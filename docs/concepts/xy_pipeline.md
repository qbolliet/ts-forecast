# Pipeline XY

## Le problème

`sklearn.pipeline.Pipeline` ne fait passer que `X` d'une étape à l'autre. `y`
traverse le pipeline sans jamais être transformé. Or, en prévision, beaucoup de
transformations touchent **les deux** :

- passage en log ou différenciation de la cible comme des features ;
- application d'un délai de publication qui décale `X` et `y` ;
- imputation multi-fréquences qui produit un `(X, y)` ré-indexé.

## La solution

`XYPipeline` est un remplacement direct de `Pipeline` qui reconnaît les
**transformateurs XY** et propage `y` transformé.

Une étape est « XY » quand elle est **déclarée** comme telle : elle hérite de
[`XYTransformerMixin`](../api/xy/XYTransformerMixin.md) ou de
[`XYPanelTimeSeriesTransformer`](../api/xy/XYPanelTimeSeriesTransformer.md)
(test : [`is_xy_transformer`](../api/xy/is_xy_transformer.md)). Le pipeline lui
donne alors la cible (`fit_transform(X, y)`, `transform(X, y)`,
`inverse_transform(X, y)`) et lit sa sortie comme la paire
`(X_transformed, y_transformed)` ; une étape XY qui rend autre chose qu'une paire
quand elle reçoit `y` lève une `TypeError`.

Toute autre étape est traitée **exactement comme par `Pipeline`**, quelles que
soient sa signature et sa sortie : `transform(X)` sans la cible, et la sortie
transmise telle quelle comme `X` de l'étape suivante.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from tsforecast.xy import XYPipeline
from tsforecast.delays import PublicationDelayTransformer

pipe = XYPipeline([
    ("delays", PublicationDelayTransformer(delays=delays, strategy="shift")),
    ("scaler", StandardScaler()),      # étape sklearn standard
    ("model", Ridge()),
])
pipe.fit(X, y)
```

## Écrire un transformateur XY

Hériter de `XYTransformerMixin` (ou de `XYPanelTimeSeriesTransformer` pour
bénéficier des validations séries / panel) et implémenter `_fit` / `_transform`
(et, au besoin, `_inverse_transform`) avec la convention de retour `(X_t, y_t)`
si `y` est fourni, `X_t` sinon. Le mixin fournit `fit`, `transform`,
`fit_transform` et `inverse_transform` ; une classe qui écrit elle-même ces
méthodes publiques reste XY tant qu'elle hérite du mixin.

## Retirer les lignes sans cible

Une étape qui augmente le jeu d'entraînement (`HighFrequencyImputer`, des
retards vers plusieurs horizons...) laisse volontairement des lignes sans cible.
[`DropNaNTarget`](../api/xy/DropNaNTarget.md), placée juste avant l'estimateur,
les retire au `fit` et au `score` (quand `y` traverse le pipeline) et n'en
retire aucune au `predict`. Un scorer textuel (`scoring='r2'`...) compare les
prédictions au `y` brut, hors du pipeline : il reçoit les cibles de test NaN.
Seul le score par défaut (`scoring=None`, c'est-à-dire `pipeline.score`) les
retire.

Les méthodes de prédiction (`predict`, `predict_proba`, `predict_log_proba`,
`decision_function`, `score_samples`) ne prennent pas de `y`, comme celles de
`Pipeline` : une prédiction ne dépend jamais de la cible, et elle compte
toujours une valeur par ligne de `X`. Pour évaluer le pipeline sur une cible,
utiliser `score(X, y)`.

## Agréger en fenêtres glissantes

[`RollingAggregator`](../api/xy/RollingAggregator.md) remplace chaque colonne
par son agrégat (somme par défaut) sur une **période calendaire glissante**,
entité par entité. Sur une grille mensuelle, `window='Q'` fait de chaque mois un
**trimestre glissant**. Placée après un `HighFrequencyImputer` qui a porté une
cible trimestrielle au mois, l'étape remet le modèle à l'échelle de la cible
observée, avec trois fois plus de lignes d'entraînement qu'un jeu trimestriel.

```python
pipe = XYPipeline([
    ("imputer", HighFrequencyImputer(target_frequency="M", estimator=LinearRegression(),
                                     keep_lower_frequencies=False)),
    ("roll", RollingAggregator(window="Q", agg={"taux_chomage": "mean"})),
    ("drop", DropNaNTarget()),
    ("model", HistGradientBoostingRegressor()),
])
```

**Deux passes**, comme l'imputeur :

- `fit_transform(X, y)` (entraînement) agrège `X` **et** `y` (`target_agg`) ;
- `transform(X, y)` (prédiction, `score`) agrège `X` seul et rend `y`
  **inchangé** : la cible observée est déjà un total de période.

**Sens de la fenêtre** (`alignment='auto'`) : il suit la position des dates de
la grille. Une grille de fin de période (`ME`) reçoit une fenêtre **passée**
`[t-2, t]`, une grille de début de période (`MS`) une fenêtre **future**
`[t, t+2]`. Dans les deux cas, la ligne d'ancre d'un trimestre somme exactement
les trois mois de ce trimestre. Une grille sans position (jours, semaines)
demande `alignment='past'` ou `'future'`.

**Taille de la fenêtre** : `window` est une fréquence (`'Q'`, `'Y'`, `'3M'`...)
comptée en pas de la grille de chaque entité, lue au `fit`. La grille doit être
de même base que la fenêtre ou s'y emboîter exactement : `Y` sur `Q`, `M` ou
`SM` ; `Q` sur `M` ou `SM` ; `M` sur `SM` ; `W` sur `D`. Les dates sont lues
comme des numéros de période : une date absente de la grille laisse un trou, et
jamais la fenêtre ne glisse sur une autre période.

**Fenêtres complètes seulement** : une fenêtre à laquelle il manque une valeur
(NaN, date absente, bord des données) donne NaN. À l'entraînement,
`DropNaNTarget` retire ces lignes. À la prédiction, les premières lignes d'un
jeu de test (fenêtre passée) ou ses dernières lignes (fenêtre future) sortent
en NaN, sauf si le jeu de test commence et finit sur des bornes de période.
Il faut donc découper la validation croisée sur des trimestres entiers, ou
utiliser un estimateur qui tolère les NaN.

**Composer avec d'autres transformations** : l'étape ne fait que l'agrégation.

- Pour un log ou une évolution de `X` après agrégation, ajouter une étape
  ordinaire du pipeline (`FunctionTransformer`, enveloppé dans
  `PanelwiseTransformer` pour un calcul par entité comme `pct_change`).
- Pour un log de la cible, envelopper le modèle dans un
  `TransformedTargetRegressor` : ses prédictions sont ramenées à l'échelle de
  la cible observée.
- Ne jamais transformer la cible au seul entraînement autrement que par une
  agrégation qui reconstruit la quantité observée. Un log ou une évolution
  appliqués ainsi mettraient le modèle et `y_test` sur deux échelles
  différentes, sans aucun avertissement.

`suffix='_q'` garde les colonnes d'origine et ajoute les agrégats après elles ;
`agg` accepte un dictionnaire par colonne avec une clé `'__default__'` (somme
pour un flux, moyenne pour un taux).

## Paramètres par ligne : poids d'échantillon

Un paramètre d'ajustement qui compte une valeur par ligne de `X` (la règle de
sklearn pour les paramètres de `fit` en validation croisée), typiquement
`sample_weight`, **suit les lignes gardées** par les étapes XY : derrière
`DropNaNTarget`, l'estimateur reçoit les poids de ses lignes d'entraînement.
C'est vrai pour `model__sample_weight=w` comme pour un `sample_weight` routé par
le *metadata routing*, au `fit`, au `fit_transform`, au `fit_predict` et au
`score(X, y, sample_weight=w)`.

Les lignes gardées sont localisées sur l'**index** de `y`, sinon de `X` : il faut
donc des objets pandas à index unique. Sur des tableaux numpy, si une étape XY
change le nombre de lignes, un paramètre par ligne est refusé par une
`ValueError` explicite plutôt que transmis désaligné.

Sans *metadata routing*, les paramètres de `predict`, `predict_proba` et
`predict_log_proba` sont transmis à l'estimateur final
(`pipe.predict(X, return_std=True)`), comme dans `Pipeline`.

## Mise en cache

`memory=` (un répertoire ou un `joblib.Memory`) met en cache l'ajustement des
étapes intermédiaires, étapes XY comprises : un second `fit` sur les mêmes
données les recharge au lieu de les réajuster. Les étapes données au pipeline
sont alors clonées et restent non ajustées. La clé de cache lit le **contenu**
des données (valeurs, index, colonnes, types) : un `DataFrame` daté dont l'index
a rempli ses caches internes pandas retrouve son entrée. Les types comprennent
l'unité de stockage des dates : les mêmes instants en secondes et en
nanosecondes sont deux entrées distinctes, comme le sont les sorties des étapes,
rendues à l'unité de leur entrée.

## Structure pandas des sorties

Avec `preserve_dataframe=True` (défaut), une sortie en tableau d'une entrée
pandas redevient un `DataFrame` (et la cible une `Series`), avec les étiquettes
**de l'appel en cours** :

- lignes : l'index du `X` reçu tant que le nombre de lignes est inchangé ; une
  fois des lignes retirées, l'index de la cible rendue, si `X` et `y` ont été
  passés avec le même index (`X` et `y` restent alignés) ;
- colonnes : les noms donnés par `get_feature_names_out` des étapes (que
  `DropNaNTarget` fournit), sinon les colonnes d'entrée si la largeur est
  inchangée, sinon des positions.

Une tranche `pipe[:-1]` garde `preserve_dataframe`. À l'intérieur du pipeline,
en revanche, la sortie en tableau d'une étape reste un tableau, comme dans
`Pipeline` : l'estimateur final n'apprend pas de noms de colonnes, et
`pipe[-1].predict(pipe[:-1].transform(X))` déclenche l'avertissement sklearn
« X has feature names, but … was fitted without feature names ». Pour des
sorties pandas de bout en bout (estimateur final compris), configurer
`pipe.set_output(transform="pandas")`.

## Compatibilité

`XYPipeline` reste compatible `GridSearchCV`, `RandomizedSearchCV`,
`cross_validate`, `cross_val_predict` (avec un découpage en partition), le
*metadata routing* (sklearn ≥ 1.4), `joblib` / `pickle` et les autres
méta-estimateurs. `inverse_transform` s'applique en ordre inverse et gère `X`
comme `y`.

## Pour aller plus loin

- Tutoriel : [Pipeline XY](../tutorials/xy_pipeline.md)
- API : [XYPipeline](../api/xy/XYPipeline.md),
  [XYTransformerMixin](../api/xy/XYTransformerMixin.md),
  [XYPanelTimeSeriesTransformer](../api/xy/XYPanelTimeSeriesTransformer.md),
  [DropNaNTarget](../api/xy/DropNaNTarget.md),
  [RollingAggregator](../api/xy/RollingAggregator.md),
  [is_xy_transformer](../api/xy/is_xy_transformer.md)
- Classes de base : [Classes de base des transformateurs](base_transformers.md)
