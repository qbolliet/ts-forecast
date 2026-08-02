# Tutoriel : Traitement des Fréquences Mixtes pour la Prévision

## Introduction

Dans de nombreuses applications de prévision, par exemple en économie ou en finance, les données proviennent de sources à **fréquences différentes** : certaines variables sont observées mensuellement (ventes, indices boursiers, indicateurs de production), d'autres trimestriellement (PIB, rapports financiers) ou annuellement (bilans comptables). Le **MixedFrequencyTransformer** de ce package permet de combiner ces données hétérogènes pour améliorer les prévisions en exploitant toute l'information disponible.

## 1. Concepts fondamentaux

### 1.1 Problématique des fréquences mixtes

Considérons un exemple typique :
- **Variable cible** : Croissance du PIB (trimestrielle)
- **Prédicteurs potentiels** : 
  - Indices de production industrielle (mensuelle)
  - Confiance des ménages (mensuelle)
  - Taux d'intérêt (quotidienne → agrégée mensuellement)

**Problème** : Comment utiliser l'information mensuelle pour prédire une variable trimestrielle ?

### 1.2 Principe de la solution

Le MixedFrequencyTransformer suit une approche en plusieurs étapes :

1. **Transformation additive** : Rendre les données additives
2. **Agrégation** : Agréger les variables haute fréquence à la fréquence cible et interpoler les variables à plus basse fréquence si spécifié par l'utilisateur
3. **Apprentissage** : Apprendre la relation entre les variables à haute fréquence agrégées et à basse fréquence interpolées (si spécifié) et les variables à la fréquence cible.
4. **Imputation** : Imputer les valeurs manquantes aux fréquences plus élevées que la fréquence cible
5. **Gestion des délais** : Traiter les délais de publication pour imputer des données en fin de période.
6. **Gestion des co-variables aux fréquences plus faibles** : Les co-variables à fréquence plus faibles que la fréquence cible sont imputées successivement en répétant les étapes 2 à 4 pour chaque fréquence de la plus faible à la plus élevée (si l'interpolation de ces variables n'est pas spécifiée par l'utilisateur)
7. **Transformation inverse** : Revenir à la forme originale des données si nécessaire

Le jeu de données résultat est un panel auquel la fréquence de référence de l'observation est une entité ajoutée à la série temporelle ou un niveau ajoutée aux entités d'un panel existant.

![Processus de traitement des fréquences mixtes](../assets/mixed_frequency_process.png)

## 2. Étape 1 : Agrégation et apprentissage

### 2.1 Agrégation à la fréquence cible

Pour apprendre la relation entre variables, on agrège d'abord tout à la fréquence de la variable que l'on cherche à prévoir (trimestrielle dans notre exemple) :

```python
import pandas as pd
import numpy as np
from tsforecast.transformers import MixedFrequencyTransformer
from sklearn.ensemble import RandomForestRegressor

# Données exemple
dates_monthly = pd.date_range('2020-01-01', periods=36, freq='MS')
dates_quarterly = pd.date_range('2020-01-01', periods=12, freq='QS')

# Variables mensuelles (haute fréquence)
data = pd.DataFrame({
    'industrial_production': np.random.randn(36),
    'consumer_confidence': np.random.randn(36),
}, index=dates_monthly)

# Variable trimestrielle (basse fréquence)
# Seulement disponible aux dates trimestrielles
gdp_data = pd.Series(np.random.randn(12), index=dates_quarterly, name='gdp_growth')

# Alignement sur la fréquence trimestrielle
data['gdp_growth'] = gdp_data
data = data.reindex(dates_monthly)  # Remplit avec NaN pour les mois non-trimestriels

print("Données brutes:")
print(data.head(12))
```

![Agrégation des fréquences](../assets/frequency_aggregation.png)

### 2.2 Configuration du transformer

```python
# Configuration de base
transformer = MixedFrequencyTransformer(
    target_frequency='quarterly',          # Fréquence cible
    estimator=RandomForestRegressor(100),  # Modèle pour imputation
    time_col='date',
    handle_nan=False,                      # Le modèle ne gère pas les NaN
    n_jobs=-1                              # Parallélisation
)

# Ajustement sur les données d'entraînement
transformer.fit(data)

print("\nFréquences détectées:")
for col, freq in transformer.frequency_map_.items():
    print(f"  {col}: {freq}")

print("\nTypes de variables détectées:")
for col, var_type in transformer.variable_types_.items():
    print(f"  {col}: {var_type}")
```

### 2.3 Transformation additive (optionnel)

Pour certaines variables (ex: PIB, ventes), une transformation logarithmique rend les données additives, ce qui facilite l'agrégation et l'imputation :

```python
from sklearn.preprocessing import FunctionTransformer

# Transformer logarithmique
log_transformer = FunctionTransformer(
    func=np.log1p,      # log(1 + x) pour gérer les valeurs nulles
    inverse_func=np.expm1  # exp(x) - 1
)

# Configuration avec transformation
transformer = MixedFrequencyTransformer(
    target_frequency='quarterly',
    estimator=RandomForestRegressor(100),
    transformer=log_transformer,  # Transformation avant apprentissage
    time_col='date'
)
```

## 3. Étape 2 : Imputation des fréquences élevées

### 3.1 Processus d'imputation standard

Une fois le modèle entraîné, on peut imputer les valeurs manquantes :

```python
# Transformation : agrégation + imputation
data_transformed = transformer.transform(data)

print("\nDonnées après transformation:")
print(data_transformed[['gdp_growth']].head(12))

# Vérification : plus de NaN dans gdp_growth
print(f"\nValeurs manquantes avant: {data['gdp_growth'].isna().sum()}")
print(f"Valeurs manquantes après: {data_transformed['gdp_growth'].isna().sum()}")
```

![Processus d'imputation](../assets/imputation_process.png)

### 3.2 Imputation avec fit_transform

Pour les données d'entraînement, `fit_transform` utilise une stratégie optimisée :

```python
# Sur données d'entraînement : stratégie enrichie
data_train_transformed = transformer.fit_transform(data)

# Le transformer:
# 1. Annule temporairement les délais de publication
# 2. Impute avec toutes les données disponibles
# 3. Réapplique les délais d'origine
```

## 4. Gestion des délais de publication

### 4.1 Problème des délais en fin de période

En pratique, les données à basse fréquence ont des délais de publication. Par exemple :
- Le PIB du Q1 2024 n'est publié qu'en mai 2024
- En avril 2024, on n'a que le PIB du Q4 2023

**Question** : Comment imputer le PIB pour février et mars 2024 si on n'a pas encore la valeur trimestrielle ?

### 4.2 Stratégie 1 : Sans shift (par défaut)

On impute avec le dernier modèle entraîné, en acceptant un délai de publication dans les imputations :

```python
# Configuration sans shift
transformer = MixedFrequencyTransformer(
    target_frequency='quarterly',
    estimator=RandomForestRegressor(100),
    time_col='date'
)

# Les imputations en fin de période auront la même qualité
# que celles apprises, mais avec un délai de publication
```

![Stratégie sans shift](../assets/no_shift_strategy.png)

### 4.3 Stratégie 2 : Shift complet

On shifte toutes les séries pour éliminer les délais de publication visibles :

```python
from tsforecast.transformers import ReleaseDelayTransformer

# Ajout du transformer de délais
delay_transformer = ReleaseDelayTransformer(
    prediction_date='today',
    time_col='date'
)

# Pipeline complet
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('delays', delay_transformer),
    ('mixed_freq', transformer)
])

# Les séries sont shiftées en amont
pipeline.fit_transform(data)
```

![Stratégie avec shift complet](../assets/full_shift_strategy.png)

**Avantage** : Pas de délai visible en fin de période  
**Inconvénient** : Les prédictions utilisent des données décalées

### 4.4 Stratégie 3 : Imputation progressive (multi-phases)

Pour obtenir la meilleure qualité d'imputation à chaque horizon, on peut entraîner plusieurs modèles avec différents shifts :

```python
# Configuration pour imputation progressive
transformer = MixedFrequencyTransformer(
    target_frequency='quarterly',
    estimator=RandomForestRegressor(100),
    time_col='date',
    progressive_imputation=True,  # Active l'imputation en phases
    n_phases=3  # Nombre de phases d'imputation
)

# Le transformer va :
# 1. Imputer d'abord avec shift maximal (meilleure qualité)
# 2. Réentraîner avec shift réduit
# 3. Imputer les observations les plus récentes
# 4. Répéter jusqu'à n_phases
```

![Imputation progressive](../assets/progressive_imputation.png)

**Avantage** : Meilleure qualité d'imputation adaptée à chaque horizon  
**Inconvénient** : Possible incohérence entre les observations (plus de bruit en fin de période)

## 5. Évaluation de la performance d'imputation

### 5.1 Validation croisée intégrée

Le transformer peut évaluer automatiquement la qualité de ses imputations :

```python
# Configuration avec validation
transformer = MixedFrequencyTransformer(
    target_frequency='quarterly',
    estimator=RandomForestRegressor(100),
    metric='mse',  # Métrique pour variables continues
    validation_size=0.2,  # 20% pour validation
    random_state=42
)

# Ajustement avec validation
transformer.fit(data)

# Accès aux performances
print("\nPerformance d'imputation (RMSE):")
for col, metrics in transformer.imputation_performance_.items():
    if metrics:
        print(f"  {col}: {np.sqrt(metrics['mse']):.4f}")
```

### 5.2 K-fold validation par variable

Pour une évaluation plus robuste :

```python
from sklearn.model_selection import cross_val_score

# Évaluation manuelle avec k-fold
def evaluate_imputation(transformer, data, variable, k=5):
    """Évalue la qualité d'imputation avec k-fold CV."""
    # Extraction des données à la fréquence agrégée
    freq = transformer.frequency_map_[variable]
    data_agg = transformer._aggregate_to_frequency(data, freq)
    
    # Préparation X, y
    feature_cols = [c for c in data_agg.columns 
                   if c != variable and c != transformer.time_col]
    X = data_agg[feature_cols].fillna(method='ffill')
    y = data_agg[variable].dropna()
    
    # Alignement
    X = X.loc[y.index]
    
    # K-fold CV
    model = transformer.imputation_models_.get(variable)
    if model and hasattr(model, 'predict'):
        scores = cross_val_score(
            model, X, y, 
            cv=k, 
            scoring='neg_mean_squared_error'
        )
        return np.sqrt(-scores.mean()), np.sqrt(scores.std())
    return None, None

# Évaluation
rmse_mean, rmse_std = evaluate_imputation(transformer, data, 'gdp_growth')
print(f"RMSE (5-fold CV): {rmse_mean:.4f} ± {rmse_std:.4f}")
```

### 5.3 Évaluation par shift (imputation progressive)

Pour l'imputation progressive, on peut évaluer chaque phase :

```python
# Évaluation de chaque phase d'imputation
if hasattr(transformer, 'phase_performances_'):
    print("\nPerformance par phase d'imputation:")
    for phase, performances in transformer.phase_performances_.items():
        print(f"\n  Phase {phase}:")
        for col, metrics in performances.items():
            rmse = np.sqrt(metrics.get('mse', 0))
            print(f"    {col}: RMSE = {rmse:.4f}")
```

## 6. Exemple complet : Prévision du PIB

### 6.1 Préparation des données

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tsforecast.transformers import MixedFrequencyTransformer

# Génération de données réalistes
np.random.seed(42)

# Dates
dates_monthly = pd.date_range('2015-01-01', '2024-12-01', freq='MS')
dates_quarterly = pd.date_range('2015-01-01', '2024-12-01', freq='QS')

# Variables mensuelles (haute fréquence)
n_months = len(dates_monthly)
monthly_data = pd.DataFrame({
    # Indices économiques mensuels
    'industrial_production': 100 + np.cumsum(np.random.randn(n_months) * 0.5),
    'consumer_confidence': 50 + 10 * np.sin(np.arange(n_months) * 2*np.pi/12) + np.random.randn(n_months) * 2,
    'unemployment_rate': 5 + np.random.randn(n_months) * 0.3,
    'retail_sales': 1000 + 50 * np.arange(n_months) + np.random.randn(n_months) * 20,
    
    # Variables de marché (agrégées du quotidien au mensuel)
    'stock_index': 3000 + 500 * np.cumsum(np.random.randn(n_months) * 0.05),
    'exchange_rate': 1.1 + np.cumsum(np.random.randn(n_months) * 0.01),
}, index=dates_monthly)

# Variable trimestrielle (basse fréquence) - PIB
n_quarters = len(dates_quarterly)
# Corrélation réaliste avec les variables mensuelles
quarterly_data = pd.DataFrame({
    'gdp_growth': 2.0 + 0.5 * np.sin(np.arange(n_quarters) * 2*np.pi/4) + np.random.randn(n_quarters) * 0.3,
    'gdp_level': 10000 + 100 * np.arange(n_quarters) + np.random.randn(n_quarters) * 50,
}, index=dates_quarterly)

# Fusion des données
data = monthly_data.copy()
data = data.join(quarterly_data, how='left')

print("Structure des données:")
print(f"  Observations totales: {len(data)}")
print(f"  Variables mensuelles: {len(monthly_data.columns)}")
print(f"  Variables trimestrielles: {len(quarterly_data.columns)}")
print(f"  Valeurs manquantes PIB: {data['gdp_growth'].isna().sum()}")
```

### 6.2 Configuration et entraînement

```python
# Séparation train/test
train_end = '2023-12-31'
data_train = data.loc[:train_end]
data_test = data.loc[train_end:]

# Configuration du transformer
transformer = MixedFrequencyTransformer(
    target_frequency='monthly',  # On veut des prévisions mensuelles
    estimator=GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ),
    metric='mse',
    validation_size=0.2,
    time_col='date',
    n_jobs=-1,
    random_state=42
)

# Entraînement
print("\nEntraînement du transformer...")
transformer.fit(data_train)

print("\nFréquences détectées:")
for col, freq in transformer.frequency_map_.items():
    print(f"  {col}: {freq}")

print("\nModèles d'imputation entraînés:")
for col in transformer.imputation_models_.keys():
    print(f"  {col}")
```

### 6.3 Transformation et évaluation

```python
# Transformation des données
data_train_transformed = transformer.fit_transform(data_train)
data_test_transformed = transformer.transform(data_test)

# Comparaison avant/après
print("\n=== Données d'entraînement ===")
print(f"PIB - Valeurs manquantes avant: {data_train['gdp_growth'].isna().sum()}")
print(f"PIB - Valeurs manquantes après: {data_train_transformed['gdp_growth'].isna().sum()}")

print("\n=== Données de test ===")
print(f"PIB - Valeurs manquantes avant: {data_test['gdp_growth'].isna().sum()}")
print(f"PIB - Valeurs manquantes après: {data_test_transformed['gdp_growth'].isna().sum()}")

# Visualisation des imputations
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Données d'entraînement
axes[0].plot(data_train.index, data_train['gdp_growth'], 
             'o', label='Observations réelles', markersize=8)
axes[0].plot(data_train_transformed.index, data_train_transformed['gdp_growth'], 
             '-', alpha=0.6, label='Avec imputation')
axes[0].set_title('Données d\'entraînement - PIB mensuel imputé')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Données de test
axes[1].plot(data_test.index, data_test['gdp_growth'], 
             'o', label='Observations réelles', markersize=8)
axes[1].plot(data_test_transformed.index, data_test_transformed['gdp_growth'], 
             '-', alpha=0.6, label='Avec imputation')
axes[1].set_title('Données de test - PIB mensuel imputé')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 6.4 Évaluation de la qualité d'imputation

```python
# Métriques sur les observations trimestrielles
quarterly_mask = data_train['gdp_growth'].notna()
y_true = data_train.loc[quarterly_mask, 'gdp_growth']
y_pred = data_train_transformed.loc[quarterly_mask, 'gdp_growth']

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("\n=== Performance d'imputation (validation sur données réelles) ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
print(f"R²: {r2_score(y_true, y_pred):.4f}")

# Distribution des erreurs
errors = y_true - y_pred
print(f"\nDistribution des erreurs:")
print(f"  Moyenne: {errors.mean():.4f}")
print(f"  Écart-type: {errors.std():.4f}")
print(f"  Min: {errors.min():.4f}")
print(f"  Max: {errors.max():.4f}")
```

## 7. Cas d'usage avancés

### 7.1 Données panel avec fréquences mixtes

```python
# Données panel : plusieurs pays/entreprises
entities = ['USA', 'EUR', 'JPN', 'CHN']
dates = pd.date_range('2015-01-01', '2024-12-01', freq='MS')

# Index multi-niveaux
index = pd.MultiIndex.from_product(
    [entities, dates],
    names=['entity', 'date']
)

# Données panel
panel_data = pd.DataFrame({
    'industrial_production': np.random.randn(len(index)),
    'consumer_confidence': np.random.randn(len(index)),
    'gdp_growth': np.nan,  # Trimestriel
}, index=index)

# Remplissage des valeurs trimestrielles
for entity in entities:
    quarterly_dates = pd.date_range('2015-01-01', '2024-12-01', freq='QS')
    for date in quarterly_dates:
        panel_data.loc[(entity, date), 'gdp_growth'] = np.random.randn()

# Configuration pour panel
transformer = MixedFrequencyTransformer(
    target_frequency='monthly',
    estimator=RandomForestRegressor(100),
    time_col='date',
    panel_cols=['entity'],  # Colonnes définissant les entités
    n_jobs=-1
)

# Transformation
panel_transformed = transformer.fit_transform(panel_data)
```

### 7.2 Transformation inverse

```python
# Utilisation de transformations additives
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(
    func=np.log1p,
    inverse_func=np.expm1
)

transformer = MixedFrequencyTransformer(
    target_frequency='monthly',
    estimator=RandomForestRegressor(100),
    transformer=log_transformer,
    time_col='date'
)

# Transformation (en log-space)
data_log = transformer.fit_transform(data_train)

# Application de la transformation inverse pour revenir à l'échelle originale
data_original_scale = data_log.copy()
for col in transformer.variable_types_.keys():
    if col in data_original_scale.columns:
        data_original_scale[col] = np.expm1(data_original_scale[col])

print("\nDonnées retransformées à l'échelle originale")
```

### 7.3 Optimisation des hyperparamètres

L'état entraîné du transformer est **immuable** : `imputation_plan_` (la liste ordonnée des
`ImputationStep`) est la seule source de vérité du fit, et `imputation_models_`,
`model_fitting_order_`, `stage_groups_` et `frequency_progression_` n'en sont que des vues
**dérivées, en lecture seule**. On ne remplace donc pas les modèles après coup : on passe les
estimateurs optimisés au paramètre `estimator`, sous forme d'un dict `variable -> estimateur`,
puis on ré-entraîne.

```python
from sklearn.model_selection import GridSearchCV

# Définition de la grille de recherche
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}

# Variables réellement imputées par un modèle, lues sur le plan d'imputation
variables = {
    step.var_name
    for step in transformer.imputation_plan_
    if not step.is_fallback
}

# Recherche d'hyperparamètres pour chaque variable
best_estimators = {}

for variable in variables:
    print(f"\nOptimisation pour {variable}...")

    # Préparation des données à la fréquence propre de la variable
    freq = transformer.detected_frequencies_[variable]
    data_agg = transformer._freq_aligner.aggregate_to_target(
        data_train, [c for c in data_train.columns if c != variable], freq,
        transformer.is_panel_
    )

    feature_cols = [c for c in data_agg.columns if c != variable]
    X = data_agg[feature_cols].ffill()
    y = data_agg[variable].dropna()
    X = X.loc[y.index]

    # Grid search
    base_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(X, y)
    best_estimators[variable] = grid_search.best_estimator_

    print(f"  Meilleurs paramètres: {grid_search.best_params_}")
    print(f"  RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

# Ré-entraînement avec les meilleurs modèles : le dict est passé au
# constructeur, jamais affecté à "imputation_models_" (vue en lecture seule)
tuned = HighFrequencyImputer(
    target_frequency='M',
    estimator=best_estimators,
    time_col='date',
)
tuned.fit(data_train)
```

## 8. Bonnes pratiques et recommandations

### 8.1 Choix de la stratégie d'imputation

| Situation | Stratégie recommandée |
|-----------|----------------------|
| **Prévision de court terme** | Imputation progressive (multi-phases) |
| **Analyse historique** | Sans shift (plus cohérent) |
| **Backtesting rigoureux** | Avec shift complet |
| **Performance maximale** | Imputation progressive + optimisation |

### 8.2 Sélection des variables et agrégation

```python
# Définir des fonctions d'agrégation personnalisées
custom_agg = {
    'industrial_production': 'mean',  # Moyenne pour les indices
    'stock_index': 'last',  # Dernière valeur du trimestre
    'unemployment_rate': 'mean',  # Moyenne du trimestre
    'retail_sales': 'sum'  # Somme sur le trimestre
}

# Possibilité d'override dans le transformer
# Note: Cette fonctionnalité doit être ajoutée au code
```

### 8.3 Gestion des valeurs aberrantes

```python
# Prétraitement avant transformation
def remove_outliers(data, columns, n_std=3):
    """Supprime les valeurs aberrantes."""
    data_clean = data.copy()
    for col in columns:
        mean = data[col].mean()
        std = data[col].std()
        mask = np.abs(data[col] - mean) < n_std * std
        data_clean.loc[~mask, col] = np.nan
    return data_clean

# Application
data_clean = remove_outliers(data_train, ['industrial_production', 'stock_index'])
transformer.fit(data_clean)
```

### 8.4 Validation temporelle stricte

```python
# Utilisation avec validation croisée temporelle
from tsforecast.crossvals import TSOutOfSampleSplit

splitter = TSOutOfSampleSplit(
    n_splits=5,
    test_size=12,  # 1 an de test
    gap=3  # 1 trimestre de gap
)

# Évaluation avec cross-validation
scores = []
for train_idx, test_idx in splitter.split(data_train):
    # Split temporel
    cv_train = data_train.iloc[train_idx]
    cv_test = data_train.iloc[test_idx]
    
    # Transformation
    cv_transformer = MixedFrequencyTransformer(
        target_frequency='monthly',
        estimator=RandomForestRegressor(100),
        time_col='date'
    )
    
    cv_train_transformed = cv_transformer.fit_transform(cv_train)
    cv_test_transformed = cv_transformer.transform(cv_test)
    
    # Évaluation (sur points trimestriels uniquement)
    quarterly_mask = cv_test['gdp_growth'].notna()
    if quarterly_mask.any():
        y_true = cv_test.loc[quarterly_mask, 'gdp_growth']
        y_pred = cv_test_transformed.loc[quarterly_mask, 'gdp_growth']
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        scores.append(rmse)

print(f"\nRMSE moyen (CV temporelle): {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

## 9. Limitations et précautions

### 9.1 Incohérences potentielles

⚠️ **Attention** : Avec l'imputation progressive, les observations peuvent être incohérentes entre elles :

```python
# Exemple d'incohérence potentielle
# Modèle 1 (shift=0) : prédit PIB janvier = 2.1%
# Modèle 2 (shift=1) : prédit PIB février = 1.8%
# Modèle 3 (shift=2) : prédit PIB mars = 2.5%

# Mais le PIB trimestriel observé est 2.0%
# Il y a incohérence car (2.1 + 1.8 + 2.5) / 3 ≠ 2.0
```

**Solution** : Contraindre les imputations mensuelles à respecter la valeur trimestrielle (distribution proportionnelle).

### 9.2 Overfitting sur petits échantillons

Pour les variables trimestrielles, on n'a qu'environ 40 observations sur 10 ans :

```python
# Recommandations
# - Utiliser des modèles simples (Ridge, Lasso) pour petits échantillons
# - Privilégier la régularisation
# - Valider rigoureusement avec k-fold CV

from sklearn.linear_model import Ridge

transformer = MixedFrequencyTransformer(
    target_frequency='monthly',
    estimator=Ridge(alpha=1.0),  # Régularisation
    validation_size=0.25,  # Plus de validation
    random_state=42
)
```

### 9.3 Extrapolation hors distribution

Les imputations peuvent être peu fiables si les variables haute fréquence sortent de leur distribution historique :

```python
# Détection d'outliers dans les prédictions
def detect_prediction_outliers(data, data_transformed, threshold=3):
    """Détecte les imputations suspectes."""
    suspicious = {}
    
    for col in data.columns:
        if col in data_transformed.columns:
            # Observations réelles
            real_values = data[col].dropna()
            mean, std = real_values.mean(), real_values.std()
            
            # Imputations
            imputed_mask = data[col].isna() & data_transformed[col].notna()
            imputed_values = data_transformed.loc[imputed_mask, col]
            
            # Valeurs suspectes
            outliers = imputed_values[np.abs(imputed_values - mean) > threshold * std]
            
            if len(outliers) > 0:
                suspicious[col] = {
                    'n_outliers': len(outliers),
                    'dates': outliers.index.tolist()
                }
    
    return suspicious

# Application
suspicious_imputations = detect_prediction_outliers(
    data_test, 
    data_test_transformed, 
    threshold=3
)

if suspicious_imputations:
    print("\n⚠️ Imputations suspectes détectées:")
    for col, info in suspicious_imputations.items():
        print(f"  {col}: {info['n_outliers']} valeurs aberrantes")
```

## 10. Conclusion

Le **MixedFrequencyTransformer** offre une solution modulaire pour exploiter des données à fréquences mixtes dans des modèles de prévision. Les points clés à retenir :

✅ **Avantages** :
- Exploite toute l'information disponible
- Augmente la granularité des prévisions
- Gère automatiquement les délais de publication
- Évalue la qualité des imputations

⚠️ **Précautions** :
- Valider rigoureusement les imputations
- Choisir la stratégie adaptée au cas d'usage
- Surveiller les incohérences en imputation progressive
- Régulariser pour éviter l'overfitting

📚 **Ressources** :
- Documentation API : `tsforecast.transformers.MixedFrequencyTransformer`