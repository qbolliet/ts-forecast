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
from sklearn.base import BaseEstimator, TransformerMixin

class LogXY(BaseEstimator, TransformerMixin):
    """Passe X et y en log1p, réversible."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_t = np.log1p(X)
        if y is not None:
            return X_t, np.log1p(y)
        return X_t

    def fit_transform(self, X, y=None):   # obligatoire pour propager y
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X, y=None):
        X_o = np.expm1(X)
        if y is not None:
            return X_o, np.expm1(y)
        return X_o
```

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

`XYPipeline` détecte automatiquement `LogXY` comme XY (il accepte `y` et renvoie
un tuple), et passe `StandardScaler` en mode standard.

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

## Voir aussi

- [Concept : pipeline XY](../concepts/xy_pipeline.md)
- [API : XYPipeline](../api/xy/XYPipeline.md),
  [XYTransformerMixin](../api/xy/XYTransformerMixin.md)
