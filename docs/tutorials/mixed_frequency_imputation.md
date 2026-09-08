# Tutoriel : imputation multi-fréquences

Ce tutoriel montre pas à pas comment donner une valeur mensuelle à une variable
annuelle avec `HighFrequencyImputer`, puis comment lire la provenance des
valeurs produites.

Voir le [concept](../concepts/mixed_frequency_imputation.md) pour le modèle
mental (les deux axes, la provenance, le plan).

## 1. Données synthétiques

Deux variables mensuelles denses (`m1`, `m2`) et une variable annuelle `a1`
observée seulement au 31 décembre.

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
# a1 = total annuel, aligné (à peu près) sur 12 * moyenne de m1
for year_end in ["2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"]:
    window = data.loc[:year_end, "m1"].tail(12)
    data.loc[year_end, "a1"] = window.sum() + rng.normal(0, 5)
```

## 2. Imputation directe (axe 2 = `False`)

Cas par défaut : `a1` est imputée directement à la fréquence mensuelle, le modèle
n'apprend que sur les ancres observées.

```python
imputer = HighFrequencyImputer(
    target_frequency="M",
    estimator=LinearRegression(),
    covariate_strategy="interpolate",   # axe 1 (défaut)
    impute_intermediate_frequencies=False,  # axe 2 (défaut)
)
result = imputer.fit_transform(data)

result.xs("M", level="frequency")["a1"].head()
```

La sortie porte un niveau `frequency` sur l'index (`(frequency, date)` ici).

## 3. Lire la provenance

```python
imputer.imputation_provenance_.xs("M", level="frequency")["a1"].value_counts()
# ProvenanceType.MODEL_ON_TRUE   60   (tout le niveau mensuel produit)

imputer.provenance_statistics_["overall"]["model_on_true_pct"]
```

Le niveau mensuel de `a1` est **entièrement produit** (désagrégation modélisée
des ancres) : ses cellules portent toutes `MODEL_ON_TRUE`. Le niveau annuel
d'origine, lui, reste `ORIGINAL`.

`provenance_statistics_` est le résumé chiffré (comptes + pourcentages par
`ProvenanceType`), au global et par colonne.

## 4. Contrainte d'agrégation

Par défaut `aggregation_constraint="sum"` : les 12 valeurs mensuelles imputées
pour une année sont recalées pour sommer exactement à l'ancre annuelle.

```python
yearly_check = (
    result.xs("M", level="frequency")["a1"]
    .groupby(lambda d: d.year)
    .sum()
)
# égal (à l'arrondi près) aux ancres a1 observées
```

Passer `aggregation_constraint=None` laisse les prédictions libres.

## 5. Covariables imputées par modèle (axe 1 = `'model'`)

Quand **plusieurs** variables de basse fréquence doivent se servir mutuellement
de features, on passe `covariate_strategy="model"` et on choisit l'ordre
d'imputation. Ajoutons une seconde variable annuelle `a2` :

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

imputer.imputation_order_          # {'M': ['a1', 'a2']}  (meilleur score CV d'abord)
imputer.imputation_cv_scores_      # {'M': {'a1': <score>, 'a2': <score>}}
```

`imputation_cv_scores_` expose les scores qui ont décidé de l'ordre — utile pour
le [tracking](../guides/mlflow_tracking.md).

## 6. Panel

Sur un panel `MultiIndex (entité, date)`, `target_frequency` peut être un dict
`{entité: fréquence}`, et les fréquences sont détectées **par (entité, colonne)**.
La sortie insère le niveau `frequency` juste avant la date :
`(entité..., 'frequency', 'date')`.

```python
imputer = HighFrequencyImputer(
    target_frequency="M",
    estimator=LinearRegression(),
    panel_cols=["country"],
)
```

!!! warning
    `covariate_strategy="interpolate"` lit l'ancre future : la sortie reconstruit
    l'histoire, elle ne simule pas un passage en temps réel.

## Voir aussi

- [Concept : imputation multi-fréquences](../concepts/mixed_frequency_imputation.md)
- [API : HighFrequencyImputer](../api/frequency/HighFrequencyImputer.md)
- [Guide MLflow](../guides/mlflow_tracking.md)
