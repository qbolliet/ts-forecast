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

Un transformateur est « XY » si son `transform` :

1. accepte `X` **et** `y` ;
2. renvoie un tuple `(X_transformed, y_transformed)` quand `y` est fourni.

Les transformateurs sklearn standards (`transform(X) -> X_t`) continuent de
fonctionner sans changement dans le même pipeline.

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

Hériter de `XYTransformerMixin` (ou de `PanelTimeSeriesTransformer`) et
implémenter `_fit` / `_transform` / `_inverse_transform` avec la convention de
retour `(X_t, y_t)` si `y` est fourni, `X_t` sinon. **`fit_transform` doit être
implémenté** pour que `y` soit correctement passé.

## Retirer les lignes sans cible

Une étape qui augmente le jeu d'entraînement (`HighFrequencyImputer`, des
retards vers plusieurs horizons...) laisse volontairement des lignes sans cible.
[`DropNaNTarget`](../api/xy/DropNaNTarget.md), placée juste avant l'estimateur,
les retire au `fit` et au `score` (quand `y` traverse le pipeline) et n'en
retire aucune au `predict`. Un scorer textuel compare les prédictions au `y`
brut, hors du pipeline : il ne peut pas recevoir une cible de test NaN.

Les méthodes de prédiction (`predict`, `predict_proba`, `predict_log_proba`,
`decision_function`, `score_samples`) ne prennent pas de `y`, comme celles de
`Pipeline` : une prédiction ne dépend jamais de la cible, et elle compte
toujours une valeur par ligne de `X`. Pour évaluer le pipeline sur une cible,
utiliser `score(X, y)`.

## Structure pandas des sorties

Avec `preserve_dataframe=True` (défaut), une sortie en tableau d'une entrée
pandas redevient un `DataFrame` (et la cible une `Series`), avec les étiquettes
**de l'appel en cours** :

- lignes : l'index du `X` reçu tant que le nombre de lignes est inchangé ; une
  fois des lignes retirées, l'index de la cible rendue, si `X` et `y` ont été
  passés avec le même index (`X` et `y` restent alignés) ;
- colonnes : les noms donnés par `get_feature_names_out` des étapes, sinon les
  colonnes d'entrée si la largeur est inchangée, sinon des positions.

## Compatibilité

`XYPipeline` reste compatible `GridSearchCV`, `cross_validate`, le *metadata
routing* (sklearn ≥ 1.4) et les autres méta-estimateurs. `inverse_transform`
s'applique en ordre inverse et gère `X` comme `y`.

## Pour aller plus loin

- Tutoriel : [Pipeline XY](../tutorials/xy_pipeline.md)
- API : [XYPipeline](../api/xy/XYPipeline.md),
  [XYTransformerMixin](../api/xy/XYTransformerMixin.md),
  [DropNaNTarget](../api/xy/DropNaNTarget.md)
- Classes de base : [Classes de base des transformateurs](base_transformers.md)
