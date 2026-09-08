# ts-forecast

**ts-forecast** est une boîte à outils Python, compatible scikit-learn, pour le
traitement et la validation de séries temporelles et de données de panel dans un
contexte de prévision *pseudo-temps réel*.

Elle répond à quatre besoins récurrents :

| Besoin | Composant | Concept | Tutoriel |
|--------|-----------|---------|----------|
| Évaluer un modèle sans fuite temporelle | `tsforecast.crossvals` | [Cross-validation temporelle](concepts/cross_validation.md) | [Ensembles train/test rigoureux](tutorials/pseudo_real_time_forecast.md) |
| Reproduire l'information réellement disponible à une date | `tsforecast.delays` | [Délais de publication](concepts/publication_delays.md) | [Délais de publication](tutorials/publication_delays.md) |
| Combiner des variables de fréquences différentes | `tsforecast.frequency` | [Imputation multi-fréquences](concepts/mixed_frequency_imputation.md) | [Imputation multi-fréquences](tutorials/mixed_frequency_imputation.md) |
| Enchaîner des transformations qui touchent X *et* y | `tsforecast.xy` | [Pipeline XY](concepts/xy_pipeline.md) | [Pipeline XY](tutorials/xy_pipeline.md) |
| Appliquer une transformation par entité d'un panel | `tsforecast.panel` | [Transformations par entité](concepts/panelwise_transforms.md) | [Transformations par entité](tutorials/panelwise_transforms.md) |
| Manipuler fréquences, durées, positions, dates | `tsforecast.utils` | [Utilitaires temporels](concepts/temporal_utils.md) | [Utilitaires temporels](tutorials/temporal_utils.md) |

## Intégration MLflow

Les composants exposent leur état ajusté (paramètres, provenance des imputations,
scores de validation croisée, description des plis) et le paquet
`tsforecast.tracking` le résume en dictionnaires plats prêts pour
`mlflow.log_metrics`. Voir le guide
[Tracking d'expériences (MLflow)](guides/mlflow_tracking.md).

## Installation

```bash
uv pip install -e .        # depuis les sources
```

## Démarrage rapide

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate

from tsforecast import TSOutOfSampleSplit, split_summary

dates = pd.date_range("2018-01-01", periods=120, freq="MS")
X = pd.DataFrame({"x1": np.random.randn(120), "x2": np.random.randn(120)}, index=dates)
y = pd.Series(np.random.randn(120), index=dates, name="y")

cv = TSOutOfSampleSplit(n_splits=5, test_size=6, gap=1)

# Validation croisée respectant l'ordre temporel
scores = cross_validate(Ridge(), X, y, cv=cv, scoring="neg_mean_absolute_error")
print(scores["test_score"])

# Description des plis, prête pour le tracking
print(split_summary(cv, X, y))
```

## Référence API

La section [API Reference](api/crossvals/OutOfSampleSplit.md) documente chaque
objet public à partir de ses docstrings (Google style).
