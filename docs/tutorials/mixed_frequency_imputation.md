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

## Voir aussi

- [Concept : imputation multi-fréquences](../concepts/mixed_frequency_imputation.md)
- [Concept : pipeline XY](../concepts/xy_pipeline.md)
- [API : HighFrequencyImputer](../api/imputation/HighFrequencyImputer.md),
  [RollingAggregator](../api/xy/RollingAggregator.md)
- [Guide MLflow](../guides/mlflow_tracking.md)
