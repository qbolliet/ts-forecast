# Tutoriel : imputation multi-fréquences

Ce tutoriel montre pas à pas comment donner une valeur mensuelle à une variable annuelle avec
`HighFrequencyImputer`, lire la provenance des valeurs produites, et utiliser l'imputeur dans un
pipeline de prédiction.

Voir le [concept](../concepts/mixed_frequency_imputation.md) pour une présentation plus détaillée de la classe, de son fonctionnement et de ses paramètres.

## 1. Données synthétiques

On considère un jeu de données composé de deux variables mensuelles denses (`m1`, `m2`) et d'une variable annuelle `a1`, observée seulement au
31 décembre.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tsforecast import HighFrequencyImputer

dates = pd.date_range("2019-01-31", periods=60, freq="ME")
rng = np.random.default_rng(0)

data = pd.DataFrame(
    {
        "m1": rng.normal(size=60).cumsum() + 100,
        "m2": rng.normal(size=60).cumsum() + 40,
        "a1": np.nan,
    },
    index=dates,
)
for year_end in ["2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"]:
    data.loc[year_end, "a1"] = data.loc[:year_end, "m1"].tail(12).sum() + rng.normal(0, 5)
```

## 2. Imputation directe

Cas par défaut : `a1` est imputée directement à la fréquence mensuelle, et son modèle n'apprend que
sur les ancres observées.

```python
imputer = HighFrequencyImputer(
    target_frequency="M",
    estimator=LinearRegression(),
    covariate_strategy="interpolate",       # axe 1 (défaut)
    impute_intermediate_frequencies=False,  # axe 2 (défaut)
)
result = imputer.fit_transform(data)

result.xs("M", level="frequency")["a1"].head()
```

La sortie porte un niveau `frequency` sur l'index (`(frequency, date)` ici) ;
`keep_lower_frequencies=False` le retire.

## 3. Lire la provenance

```python
imputer.imputation_provenance_.xs("M", level="frequency")["a1"].value_counts().head(1)
# model_on_true    60

imputer.provenance_statistics_["overall"]["model_on_true_pct"]
# 33.3 : les 60 cellules de a1 sur les 180 du niveau mensuel
```

Le niveau mensuel de `a1` est **entièrement produit** : ses 60 cellules, ancres de décembre
comprises, portent `MODEL_ON_TRUE`, car le modèle n'a vu que des valeurs observées (`m1` et `m2`
sont mensuelles). La matrice est catégorielle : `value_counts` liste les dix types de provenance, y compris ceux qui ne sont pas produits ici.

`provenance_statistics_` est le résumé chiffré (comptes et pourcentages par `ProvenanceType`), au
global et par colonne. Comme `imputation_provenance_`, il décrit la **dernière passe** : chaque
`transform` le réécrit, sur le niveau cible de sa sortie.

## 4. Contrainte d'agrégation

Par défaut, `aggregation_constraint="sum"` : les 12 valeurs mensuelles d'une année sont recalées
pour sommer exactement à l'ancre annuelle.

```python
monthly = result.xs("M", level="frequency")["a1"]
yearly = monthly.groupby(monthly.index.year).sum()

np.allclose(yearly.to_numpy(), data["a1"].dropna().to_numpy())
# True
```

`aggregation_constraint=None` laisse les prédictions libres.

Le recalage `"sum"` est un prorata : chaque année est multipliée par son propre rapport, si bien que
la série **saute** en janvier quand deux années voisines reçoivent des rapports différents, surtout
après une interpolation, dont le profil est lisse. `aggregation_constraint="denton"` respecte les
mêmes totaux en ajoutant la correction la plus lisse possible, calculée sur toute la série :

```python
def january_moves(constraint):
    imputer = HighFrequencyImputer(target_frequency="M", estimator=None,
                                   aggregation_constraint=constraint,
                                   keep_lower_frequencies=False)
    a1 = imputer.fit_transform(data)["a1"]
    moves = a1.diff().abs()
    return moves[moves.index.month == 1].mean(), moves[moves.index.month != 1].mean()

january_moves("sum")      # (2.13, 0.24) : le pas de janvier vaut 9 pas ordinaires
january_moves("denton")   # (0.51, 0.35) : la série tourne sans sauter
```

Denton est additif : il ne divise par rien, et convient aussi aux variables qui changent de signe.

## 5. Covariables imputées par modèle

Quand **plusieurs** variables de basse fréquence doivent se servir mutuellement de covariables, on
passe `covariate_strategy="model"` et on choisit l'ordre d'imputation. On ajoute une seconde variable
annuelle `a2` :

```python
data["a2"] = np.nan
for year_end in ["2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"]:
    data.loc[year_end, "a2"] = data.loc[:year_end, "m2"].tail(12).sum() + rng.normal(0, 5)

imputer = HighFrequencyImputer(
    target_frequency="M",
    estimator=LinearRegression(),
    covariate_strategy="model",
    fit_predict_order="cv",     # ordre par score de validation croisée
    cv=3,
    min_cv_train_size=3,
)
imputer.fit(data)

imputer.imputation_order_          # {'M': ['a1', 'a2']}  (meilleur score d'abord)
imputer.imputation_cv_scores_      # {'M': {'a1': <score>, 'a2': <score>}}
```

`a2` reçoit `a1` déjà imputée comme covariable. `imputation_cv_scores_` expose les scores qui ont
décidé de l'ordre, utiles pour le [tracking](../guides/mlflow_tracking.md).

L'ordre par défaut, `fit_predict_order="frequency"`, est décrit dans le
[concept](../concepts/mixed_frequency_imputation.md#axe-1-covariate_strategy-comment-mettre-a-disposition-du-modele-dimputation-une-covariable-peu-observee).

## 6. Panel

Sur un panel indexé `(entité, date)`, aucun paramètre supplémentaire n'est nécessaire : les
fréquences sont détectées **par (entité, colonne)**, et la sortie insère le niveau `frequency` juste
avant la date. Sur un frame plat, `time_col` et `panel_cols` désignent les colonnes de date et
d'entité.

```python
panel = pd.concat(
    {"FR": data[["m1", "m2", "a1"]], "DE": 1.1 * data[["m1", "m2", "a1"]]},
    names=["country", "date"],
)
imputer = HighFrequencyImputer(target_frequency="M", estimator=LinearRegression())
imputer.fit_transform(panel).index.names
# ['country', 'frequency', 'date']
```

Le modèle de `a1` s'entraîne sur les ancres des deux pays réunies. `target_frequency` accepte aussi
un dictionnaire `{entité: fréquence}`.

## 7. Dans un pipeline : une cible annuelle

Pour prédire `a1` à partir de `m1` et `m2`, l'imputeur multiplie les lignes d'entraînement (une par
mois au lieu d'une par an). Le modèle apprendrait alors une valeur mensuelle, alors que la cible de
test reste un total annuel : [`RollingAggregator`](../api/xy/RollingAggregator.md) ramène chaque
ligne à l'échelle de la cible, en **année glissante**.

```python
from tsforecast.xy import DropNaNTarget, RollingAggregator, XYPipeline

X, y = data[["m1", "m2"]], data["a1"]

# Passe d'entraînement : X et y au mois, puis en années glissantes
imputer = HighFrequencyImputer(target_frequency="M", estimator=LinearRegression(),
                               keep_lower_frequencies=False)
X_monthly, y_monthly = imputer.fit_transform(X, y)
roller = RollingAggregator(window="Y", agg="mean")
X_rolling, y_rolling = roller.fit_transform(X_monthly, y_monthly)

roller.alignments_
# {(): 'past'} : grille de fin de mois (ME), fenêtre [t-11, t]
np.isclose(y_rolling["2020-12-31"], y["2020-12-31"])
# True : l'année glissante de décembre redonne le total observé
```

Dans un `XYPipeline`, `DropNaNTarget` retire les lignes sans cible au `fit` et au `score`. Un scorer
textuel de sklearn compare les prédictions au `y` de test brut, NaN compris, et lèverait une erreur : le scorer
ci-dessous restreint la comparaison aux lignes observées.

```python
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate
from tsforecast import TSOutOfSampleSplit


def mae_on_observed(pipeline, X, y):
    observed = y.notna().to_numpy()
    return -mean_absolute_error(y[observed], pipeline.predict(X)[observed])


pipeline = XYPipeline([
    ("imputer", HighFrequencyImputer(target_frequency="M", estimator=LinearRegression(),
                                     keep_lower_frequencies=False)),
    ("roll", RollingAggregator(window="Y", agg="mean")),
    ("drop", DropNaNTarget()),     # lignes sans cible, au fit et au score
    ("fill", SimpleImputer()),     # premiers mois d'un pli de test : fenêtre incomplète
    ("model", LinearRegression()),
])
scores = cross_validate(pipeline, X, y, cv=TSOutOfSampleSplit(n_splits=2, test_size=12),
                        scoring=mae_on_observed)
-scores["test_score"]
# array([ 3.3, 17.5]) ; sans l'étape "roll" : array([1073. , 1165.6])
```

Sans `RollingAggregator`, le modèle compare une valeur mensuelle à un total annuel : l'erreur est
de l'ordre du total lui-même. Chaque pli de test couvre une année civile, si bien que décembre, la
seule ligne évaluée, dispose d'une fenêtre complète.

!!! warning
    `covariate_strategy="interpolate"` lit l'ancre future : la sortie reconstruit l'histoire, elle
    ne simule pas un passage en temps réel. Les délais de publication s'appliquent avant l'imputeur
    (voir le [concept](../concepts/mixed_frequency_imputation.md#dans-un-pipeline-entraine-avec-y)).

## 8. Récapitulatif de la classe

### 8.1 Arguments

Les paramètres acceptant un dictionnaire (`estimator`, `interpolation_method`,
`interpolation_anchor`, `scale_features`, `aggregation_constraint`) sont indexés par **nom de
colonne**, avec une clé `'__default__'` facultative pour les colonnes non nommées.

<!-- doc-args: tsforecast.imputation.HighFrequencyImputer -->

**Essentiels**

| Argument | Valeurs | Défaut | Description |
|----------|---------|--------|-------------|
| `target_frequency` | `str`, ou `dict` `{entité: fréquence}` sur un panel | — | Fréquence de la grille de sortie (`'M'`, `'Q'`, `'W'`, ...). Un dictionnaire nomme **toutes** les entités. |
| `estimator` | estimateur sklearn, `dict` `{colonne: estimateur}`, `None` | `None` | Modèle d'imputation de chaque variable. `None` remplace tous les modèles par une interpolation (un avertissement au `fit`). |
| `additive_transformer` | transformateur exposant `fit_transform` et `inverse_transform`, `None` | `None` | Rend les données additives avant toute imputation (log, différenciation) ; inversé en sortie. |

**Axe 1 — les colonnes du modèle**

| Argument | Valeurs | Défaut | Description |
|----------|---------|--------|-------------|
| `covariate_strategy` | `'tolerate_nan'`, `'interpolate'`, `'model'` | `'interpolate'` | Forme sous laquelle le modèle reçoit une covariable moins fréquente que la grille : telle qu'observée (l'estimateur doit tolérer les NaN), interpolée, ou imputée par son propre modèle, en cascade. `'interpolate'` est le moins coûteux en calcul, `'model'` donne une erreur d'imputation plus faible. Une covariable seulement interpolée à la prédiction et observée à la fréquence des ancres de la variable est écartée (`ImputationStep.dropped_covariates`). |
| `covariate_fallback` | `'interpolate'`, `'tolerate_nan'` | `'interpolate'` | Forme d'une covariable pas encore imputée à l'étape : interpolée depuis ses ancres d'origine, ou ses ancres brutes. Sans effet hors de `'model'`. |
| `covariate_eligibility` | `'any_entity'`, `'all_entities'` | `'any_entity'` | Panel : une colonne est covariable dès qu'une entité l'observe, ou seulement si toutes l'observent (choix sûr pour un estimateur qui ne tolère pas les NaN). |
| `interpolation_method` | méthode de `pandas.Series.interpolate`, ou `dict` | `'linear'` | Forme de toute interpolation : covariables, replis, `estimator=None`. |
| `interpolation_anchor` | `float` de `[0, 1]`, `None`, ou `dict` | `None` | Position d'une valeur dans sa période (`0` : début, `1` : fin) ; `None` garde l'ancrage détecté. |

**Axe 2 — les lignes de la cible**

| Argument | Valeurs | Défaut | Description |
|----------|---------|--------|-------------|
| `impute_intermediate_frequencies` | `bool` | `False` | `True` passe par les fréquences intermédiaires (annuel → trimestriel → mensuel) et entraîne le modèle final de chaque variable aussi sur ses imputations antérieures. `False` est le moins coûteux en calcul, `True` donne une erreur d'imputation plus faible. |
| `impute_unobserved_entities` | `bool` | `False` | Panel : impute, à partir des autres entités, une entité qui n'observe jamais la colonne (`MODEL_UNANCHORED`, sans recalage). Gouverne aussi les entités nouvelles au `transform`. |

**Rôle de `y`**

| Argument | Valeurs | Défaut | Description |
|----------|---------|--------|-------------|
| `target_as_covariate` | `bool` | `False` | `y`, fourni au `fit`, peut servir de covariable aux colonnes de `X` ; la passe de prédiction, qui ne voit jamais `y`, lève alors si un modèle en a besoin. Sans effet sans `y`. |

**Ordre d'imputation** (sans effet hors de `covariate_strategy='model'`)

| Argument | Valeurs | Défaut | Description |
|----------|---------|--------|-------------|
| `fit_predict_order` | `'frequency'`, `'cv'` | `'frequency'` | Ordre des variables d'une étape : fréquence source la plus élevée d'abord (variables sans ancre en dernier), ou meilleur score de validation croisée d'abord. |
| `cv` | `None`, entier ≥ 2, splitter, itérable de splits | `None` | Validation croisée de l'ordre `'cv'` ; `None` et un entier donnent un `KFold` mélangé. |
| `cv_scoring` | chaîne ou callable accepté par `check_scoring` | `'neg_mean_absolute_percentage_error'` | Score de l'ordre `'cv'`, plus grand = meilleur. |
| `min_cv_train_size` | entier ≥ 1 | `10` | Nombre minimal d'observations pour scorer une variable ; en dessous, elle rejoint le groupe ordonné par fréquence. |

**Fenêtres**

| Argument | Valeurs | Défaut | Description |
|----------|---------|--------|-------------|
| `imputation_scope` | `'strict'`, `'extended_backward'`, `'extended_forward'`, `'extended_both'` | `'strict'` | Fenêtre de prédiction : la période où toutes les variables sont disponibles, étendue vers le passé, l'avenir ou les deux. |
| `coverage_threshold` | `float` de `[0, 1]` | `0.5` | Couverture minimale d'une période pour entrer dans une extension. |
| `training_scope` | `None`, ou une valeur d'`imputation_scope` | `None` | Fenêtre d'entraînement ; `None` suit `imputation_scope`. Ajoute des lignes, jamais des colonnes. |
| `training_coverage_threshold` | `None`, `float` de `[0, 1]` | `None` | Seuil des extensions de la fenêtre d'entraînement ; `None` suit `coverage_threshold`. Sans effet sans `training_scope`. |

**Échelle et contraintes**

| Argument | Valeurs | Défaut | Description |
|----------|---------|--------|-------------|
| `scale_features` | `False`, `'constant'`, `'calendar'`, ou `dict` (par colonne, puis par entité) | `'constant'` | Ramène cible et covariables à l'échelle d'un pas de grille : un total trimestriel divisé par trois (`'constant'`) ou au prorata de la durée réelle de chaque mois (`'calendar'`). |
| `aggregation_constraint` | `'sum'`, `'denton'`, `None`, ou `dict` | `'sum'` | Recalage des sous-périodes sur le total observé : au prorata de chaque période, par l'étalonnage lissé de Denton, ou aucun. |

**Sortie et divers**

| Argument | Valeurs | Défaut | Description |
|----------|---------|--------|-------------|
| `keep_lower_frequencies` | `bool` | `True` | Empile les niveaux de la progression dans un niveau d'index `frequency`. Paramètre d'affichage, sans effet sur les valeurs. |
| `on_frequency_mismatch` | `'error'`, `'warn'` | `'error'` | Réaction à une fréquence cible plus élevée que celle que les données permettent. |
| `restore_original_values` | `bool` | `False` | `inverse_transform` remet toutes les valeurs observées en entrée, et non les seules cellules `ORIGINAL`. |
| `time_col` | `str`, `None` | `None` | Colonne des dates d'un frame plat. |
| `panel_cols` | liste de `str`, `None` | `None` | Colonnes des entités d'un frame plat. |
| `verbose` | `bool` | `False` | Messages de progression préfixés `[HighFrequencyImputer]`. |
<!-- /doc-args -->

Référence complète : [API `HighFrequencyImputer`](../api/imputation/HighFrequencyImputer.md).

### 8.2 Méthodes

| Méthode | Description |
|---------|-------------|
| `fit(X, y=None)` | Détecte les fréquences, construit et fige le plan d'imputation (`imputation_plan_`). |
| `fit_transform(X, y=None)` | Passe d'entraînement : rend la sortie sur la grille cible, `(X, y)` si `y` est fourni. |
| `transform(X, y=None)` | Rejoue le plan figé sans rien réajuster. Avec `X` seul : passe de prédiction, sur les lignes d'entrée, `y` rendu inchangé. |
| `inverse_transform(X)` | Défait la dernière passe : index de l'entrée, cellules `ORIGINAL` (toutes les observations sous `restore_original_values=True`). |

### 8.3 Attributs après `fit`

| Attribut | Description |
|----------|-------------|
| `effective_target_frequency_` | Fréquence cible normalisée, scalaire ou par entité. |
| `detected_frequencies_` | Fréquence détectée par colonne, ou par couple `(entité..., colonne)` sur un panel. |
| `variable_categories_` | Variables rangées en `'target_freq'`, `'impute'` et `'aggregate'`. |
| `frequency_progression_` | Fréquences des étapes, de la plus grossière à la cible. |
| `imputation_plan_` | `ImputationPlan` : l'état ajusté complet, une étape par (fréquence, variable). |
| `imputation_models_` | Vue `{(étape, variable): estimateur}` sur le plan. |
| `imputation_order_` | Ordre des variables par étape ; vide hors de `'model'`. |
| `imputation_cv_scores_` | Scores de validation croisée qui ont décidé de l'ordre `'cv'`. |
| `unanchored_pairs_` | Couples `(entité..., colonne)` imputés sans aucune ancre. |
| `imputation_window_`, `training_window_` | Bornes `(début, fin)` des fenêtres de prédiction et d'entraînement, par entité sur un panel. |
| `imputation_window_mask_`, `training_window_mask_`, `strict_window_mask_` | Masques booléens des trois fenêtres. |
| `imputation_provenance_` | Matrice de provenance catégorielle de la **dernière passe**, de même forme que sa sortie. |
| `provenance_statistics_` | Comptes et pourcentages par `ProvenanceType`, au global et par colonne, de la dernière passe. |
| `additive_transformer_` | Clone ajusté d'`additive_transformer`, `None` sans lui. |
| `feature_columns_`, `target_column_` | Colonnes de `X` reçues, nom de `y` dans le frame de travail. |
| `entities_`, `is_panel_` | Entités du panel (`None` sur une série) et nature des données. |

## Voir aussi

- [Concept : imputation multi-fréquences](../concepts/mixed_frequency_imputation.md)
- [Concept : pipeline XY](../concepts/xy_pipeline.md)
- [API : HighFrequencyImputer](../api/imputation/HighFrequencyImputer.md),
  [RollingAggregator](../api/xy/RollingAggregator.md)
- [Guide MLflow](../guides/mlflow_tracking.md)
