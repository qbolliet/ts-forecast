# Tutoriel : transformations par entité

`PanelwiseTransformer` applique un transformateur **indépendamment à chaque
entité** d'un panel. Voir le [concept](../concepts/panelwise_transforms.md).

## 1. Un panel synthétique

Panel à `MultiIndex (country, date)`. Avec un `MultiIndex`, on laisse
`time_col=None, panel_cols=None` : la structure est détectée automatiquement.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
dates = pd.date_range("2020-01-01", periods=36, freq="MS")
idx = pd.MultiIndex.from_arrays(
    [np.repeat(["FR", "DE", "IT"], 36), list(dates) * 3],
    names=["country", "date"],
)
gdp = np.concatenate([
    rng.normal(0, 1, 36),      # FR : ~0,  écart-type 1
    rng.normal(100, 5, 36),    # DE : ~100, écart-type 5
    rng.normal(-20, 0.5, 36),  # IT : ~-20, écart-type 0.5
])
panel = pd.DataFrame({"gdp": gdp}, index=idx).sort_index()
```

## 2. Mode simple — même transformateur, un ajustement par entité

```python
from sklearn.preprocessing import StandardScaler
from tsforecast.panel import PanelwiseTransformer

pt = PanelwiseTransformer(transformer=StandardScaler(), time_col=None, panel_cols=None)
scaled = pt.fit_transform(panel)

pt.transformers_          # {('DE',): StandardScaler(), ('FR',): ..., ('IT',): ...}
pt.entities_              # 3
```

Chaque pays est centré-réduit avec sa propre moyenne et son propre écart-type.

## 3. Mode fabrique — un transformateur construit par entité

```python
def scaler_for(entity_key):
    # Pas de centrage pour l'Italie
    return StandardScaler(with_mean=entity_key != ("IT",))

pt = PanelwiseTransformer(transformer=scaler_for, time_col=None, panel_cols=None)
pt.fit_transform(panel)
```

## 4. Mode `entity_kwargs` — paramètres par entité

```python
pt = PanelwiseTransformer(
    transformer=StandardScaler(),
    time_col=None,
    panel_cols=None,
    entity_kwargs={("FR",): {"with_std": False}},
    default_entity_kwargs={"with_std": True},
)
pt.fit_transform(panel)
```

## 5. Parallélisation et gestion d'erreurs

```python
pt = PanelwiseTransformer(
    transformer=StandardScaler(),
    time_col=None,
    panel_cols=None,
    n_jobs=-1,               # joblib
    error_handling="warn",   # une entité en échec n'arrête pas le reste
)
pt.fit_transform(panel)
pt.failed_entities_          # []
```

## 6. Réversibilité

Si le transformateur de base est réversible, `PanelwiseTransformer` l'est aussi :

```python
original = pt.inverse_transform(scaled)
```

!!! note "Données en colonnes plutôt qu'en index"
    Si `country` et `date` sont des colonnes (pas un `MultiIndex`), passer
    `panel_cols=["country"]` et `time_col="date"`.

## Voir aussi

- [Concept : transformations par entité](../concepts/panelwise_transforms.md)
- [API : PanelwiseTransformer](../api/panel/PanelwiseTransformer.md)
