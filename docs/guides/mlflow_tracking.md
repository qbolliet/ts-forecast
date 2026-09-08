# Tracking d'expériences (MLflow)

`ts-forecast` est une boîte à outils : elle n'écrit rien dans MLflow elle-même,
mais elle **rend accessible** tout ce qu'un projet a besoin de logguer —
paramètres, métriques, artefacts — et fournit dans `tsforecast.tracking` des
helpers qui produisent des dictionnaires plats `{str: float}` directement
consommables par `mlflow.log_metrics`.

Aucun composant du paquet ne dépend de MLflow.

## Vue d'ensemble

| Composant | Paramètres | Métriques | Artefacts |
|-----------|-----------|-----------|-----------|
| Splitters `crossvals` | `get_params()` | `split_summary()` | — |
| `PublicationDelayTransformer` | `get_params()` | `delay_metrics()` | délais inférés (`compare_and_detect_delays` → CSV) |
| `HighFrequencyImputer2` | `get_params()` | `imputation_metrics()` | `imputation_provenance_` (CSV), `imputation_plan_` (repr), `frequency_progression_` |
| `PanelwiseTransformer` | `get_params()` | — | `failed_entities_` |
| `XYPipeline` | `get_params()` | score du modèle final | — |

## 1. Paramètres

Tous les transformateurs et splitters sont des `BaseEstimator` scikit-learn :
`get_params(deep=True)` renvoie un dict à plat, presque prêt pour
`mlflow.log_params`. Attention aux valeurs non scalaires (un estimateur imbriqué,
un dict) — les convertir en `repr` :

```python
import mlflow

def log_params(estimator, prefix=""):
    params = {}
    for key, value in estimator.get_params(deep=True).items():
        params[f"{prefix}{key}"] = value if isinstance(value, (str, int, float, bool, type(None))) else repr(value)
    mlflow.log_params(params)
```

## 2. Métriques — `tsforecast.tracking`

### `split_summary(splitter, X, y=None, groups=None)`

Décrit les plis qu'un splitter produit : `n_splits`, `train_size.{min,max,mean}`,
`test_size.{min,max,mean}`, `gap`.

```python
from tsforecast.crossvals import TSOutOfSampleSplit
from tsforecast import split_summary

cv = TSOutOfSampleSplit(n_splits=5, test_size=6, gap=1)
mlflow.log_metrics(split_summary(cv, X, y))
```

### `delay_metrics(delay_transformer)`

Résumé des délais appliqués par un `PublicationDelayTransformer` ajusté :
`n_shift_columns`, `n_mask_columns`, `n_delayed_columns`,
`shift_periods.{min,max,mean,n}`, `mask_obs.{min,max,mean,n}`,
`delay.{min,max,mean,n}`.

```python
from tsforecast import delay_metrics

pdt.fit(X)
mlflow.log_metrics(delay_metrics(pdt))
```

### `imputation_metrics(imputer, *, per_column=False)`

Résumé d'un `HighFrequencyImputer2` (ou `HighFrequencyImputer`) ajusté :

- `provenance.<type>` et `provenance.<type>_pct` pour chaque `ProvenanceType`
  (ex. `provenance.original_pct`, `provenance.model_on_true_pct`,
  `provenance.interpolated_pct`) ;
- `n_stages`, `n_unanchored_pairs` ;
- `cv_score.{min,max,mean,n}` — les scores de validation croisée qui ont décidé
  l'ordre d'imputation (présents seulement sous `covariate_strategy='model'` +
  `fit_predict_order='cv'`) ;
- avec `per_column=True`, les mêmes blocs de provenance préfixés par colonne.

```python
from tsforecast import imputation_metrics

imputer.fit(data)
mlflow.log_metrics(imputation_metrics(imputer))
mlflow.log_metrics(imputation_metrics(imputer, per_column=True))
```

Les valeurs non finies (`inf` d'une CV entièrement en échec, `nan`) sont retirées
du dictionnaire — la plupart des trackers les rejettent ; le compteur `*.n`
associé reste visible.

## 3. Artefacts

```python
# Matrice de provenance complète
imputer.imputation_provenance_.to_csv("provenance.csv")
mlflow.log_artifact("provenance.csv")

# Plan d'imputation (état ajusté lisible)
with open("imputation_plan.txt", "w") as f:
    f.write(repr(imputer.imputation_plan_))
mlflow.log_artifact("imputation_plan.txt")

# Progression des fréquences
mlflow.log_dict(imputer.frequency_progression_, "frequency_progression.json")

# Délais inférés en amont
from tsforecast.delays import compare_and_detect_delays
delays = compare_and_detect_delays(new_data, existing_data, download_date="2024-05-15")
delays.to_csv("inferred_delays.csv")
mlflow.log_artifact("inferred_delays.csv")
```

## 4. Évaluer un modèle en validation croisée

Il n'existe pas de `crossval_score` dédié dans le paquet : les splitters suivent
le contrat scikit-learn, `cross_validate` fait le travail.

```python
from sklearn.model_selection import cross_validate
from tsforecast.crossvals import TSOutOfSampleSplit
from tsforecast import split_summary

cv = TSOutOfSampleSplit(n_splits=5, test_size=6, gap=1)

cv_res = cross_validate(
    estimator, X, y, cv=cv,
    scoring=["neg_root_mean_squared_error", "neg_mean_absolute_percentage_error"],
)

with mlflow.start_run():
    mlflow.log_metrics(split_summary(cv, X, y))
    for key, scores in cv_res.items():
        if key.startswith("test_"):
            mlflow.log_metric(f"{key}.mean", float(scores.mean()))
            mlflow.log_metric(f"{key}.std", float(scores.std()))
```

## 5. Exemple complet

```python
import mlflow
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from tsforecast import HighFrequencyImputer2, TSOutOfSampleSplit
from tsforecast import imputation_metrics, split_summary

with mlflow.start_run():
    # --- imputation multi-fréquences ---
    imputer = HighFrequencyImputer2(target_frequency="M", estimator=Ridge())
    imputed = imputer.fit_transform(raw_data)

    mlflow.log_params({f"imputer__{k}": repr(v) for k, v in imputer.get_params().items()})
    mlflow.log_metrics(imputation_metrics(imputer))
    imputer.imputation_provenance_.to_csv("provenance.csv")
    mlflow.log_artifact("provenance.csv")

    # --- modèle + validation croisée temporelle ---
    X_ts = imputed.xs("M", level="frequency")
    X, y = X_ts.drop(columns="target"), X_ts["target"]
    cv = TSOutOfSampleSplit(n_splits=5, test_size=6, gap=1)

    mlflow.log_metrics(split_summary(cv, X, y))
    res = cross_validate(Ridge(), X, y, cv=cv, scoring="neg_root_mean_squared_error")
    mlflow.log_metric("cv_rmse.mean", float(-res["test_score"].mean()))
```

## Voir aussi

- [API : tracking.metrics](../api/tracking/metrics.md)
- [Concept : cross-validation temporelle](../concepts/cross_validation.md)
- [Concept : imputation multi-fréquences](../concepts/mixed_frequency_imputation.md)
