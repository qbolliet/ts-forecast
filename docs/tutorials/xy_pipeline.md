# Tutoriel : pipeline XY

`XYPipeline` remplace `sklearn.pipeline.Pipeline` quand une étape transforme
`X` **et** `y`. Voir le [concept](../concepts/xy_pipeline.md).

## 1. Le problème en une cellule

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

dates = pd.date_range("2020-01-01", periods=48, freq="MS")
X = pd.DataFrame({"x": np.abs(np.random.randn(48)) + 1}, index=dates)
y = pd.Series(np.abs(np.random.randn(48)) + 1, index=dates, name="y")

# Un Pipeline sklearn ne transforme jamais y : log appliqué à X seulement
Pipeline([("log", FunctionTransformer(np.log1p))]).fit_transform(X, y)  # y intact
```

## 2. Un transformateur XY

```python
from sklearn.base import BaseEstimator
from tsforecast.xy import XYTransformerMixin

class LogXY(BaseEstimator, XYTransformerMixin):
    """Passe X et y en log1p, réversible."""

    def _fit(self, X, y):
        return self

    def _transform(self, X, y=None):
        X_t = np.log1p(X)
        if y is not None:
            return X_t, np.log1p(y)
        return X_t

    def _inverse_transform(self, X, y=None):
        X_o = np.expm1(X)
        if y is not None:
            return X_o, np.expm1(y)
        return X_o
```

L'héritage de `XYTransformerMixin` **déclare** l'étape XY ; le mixin fournit
`fit`, `transform`, `fit_transform` et `inverse_transform` à partir des trois
méthodes privées.

## 3. Dans un `XYPipeline`

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from tsforecast.xy import XYPipeline

pipe = XYPipeline([
    ("log", LogXY()),          # transformateur XY
    ("scaler", StandardScaler()),  # étape sklearn standard, inchangée
    ("model", Ridge()),
])

pipe.fit(X, y)
pipe.predict(X.iloc[-6:])
```

`XYPipeline` traite `LogXY` comme XY parce qu'il hérite de `XYTransformerMixin`
(`is_xy_transformer(LogXY())` vaut `True`) : il lui passe `y` et lit sa sortie
comme la paire `(X, y)`. `StandardScaler`, étape sklearn ordinaire, est traité
exactement comme par `Pipeline`.

## 4. Transformation seule et inversion

```python
X_t, y_t = XYPipeline([("log", LogXY()), ("scaler", StandardScaler())]).fit_transform(X, y)
```

`inverse_transform` s'applique en ordre inverse et gère les deux côtés.

## 5. Compatibilité méta-estimateurs

```python
from sklearn.model_selection import GridSearchCV
from tsforecast.crossvals import TSOutOfSampleSplit

grid = GridSearchCV(
    pipe,
    param_grid={"model__alpha": [0.1, 1.0, 10.0]},
    cv=TSOutOfSampleSplit(n_splits=4, test_size=6, gap=1),
)
grid.fit(X, y)
```

## 6. Poids d'échantillon derrière un retrait de lignes

```python
from tsforecast.xy import DropNaNTarget

y_gaps = y.copy()
y_gaps.iloc[[3, 10]] = np.nan
weights = np.linspace(0.5, 1.5, len(X))

pipe = XYPipeline([("drop", DropNaNTarget()), ("model", Ridge())])
pipe.fit(X, y_gaps, model__sample_weight=weights)   # 46 poids pour 46 lignes gardées
```

Les poids suivent les lignes gardées, localisées sur l'index de `y` (ou de `X`).

## 7. Trimestres glissants : `y` agrégé à l'entraînement seulement

```python
from tsforecast.xy import RollingAggregator

months = pd.date_range("2024-01-31", periods=6, freq="ME")   # fin de mois : fenêtre passée
X_m = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=months)
y_m = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], index=months)

step = RollingAggregator(window="Q")
X_t, y_t = step.fit_transform(X_m, y_m)   # passe d'entraînement
y_t.tolist()                              # [nan, nan, 60.0, 90.0, 120.0, 150.0]

_, y_p = step.transform(X_m, y_m)         # passe de prédiction : y inchangé
y_p.tolist()                              # [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
```

Sur une grille de début de mois (`MS`), la fenêtre est future : la ligne du
1er janvier somme janvier, février et mars.

## Voir aussi

- [Concept : pipeline XY](../concepts/xy_pipeline.md)
- [API : XYPipeline](../api/xy/XYPipeline.md),
  [XYTransformerMixin](../api/xy/XYTransformerMixin.md),
  [RollingAggregator](../api/xy/RollingAggregator.md)
