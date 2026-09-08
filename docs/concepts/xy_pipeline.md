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

## Compatibilité

`XYPipeline` reste compatible `GridSearchCV`, `cross_validate`, le *metadata
routing* (sklearn ≥ 1.4) et les autres méta-estimateurs. `inverse_transform`
s'applique en ordre inverse et gère `X` comme `y`.

## Pour aller plus loin

- Tutoriel : [Pipeline XY](../tutorials/xy_pipeline.md)
- API : [XYPipeline](../api/xy/XYPipeline.md),
  [XYTransformerMixin](../api/xy/XYTransformerMixin.md)
- Classes de base : [Classes de base des transformateurs](base_transformers.md)
