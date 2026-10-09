# Tutoriel : Traitement des délais de publication en prévision

## Table des matières

- [Introduction](#introduction)
- [1. Comprendre les délais de publication](#1-comprendre-les-délais-de-publication)
  - [1.1 Qu'est-ce qu'un délai de publication ?](#11-quest-ce-quun-délai-de-publication-)
  - [1.2 Impact sur la prévision](#12-impact-sur-la-prévision)
- [2. Deux approches pour gérer les délais](#2-deux-approches-pour-gérer-les-délais)
  - [2.1 Vue d'ensemble](#21-vue-densemble)
  - [2.2 Approche 1 : Conservation de l'acquis (strategy `mask`)](#22-approche-1--conservation-de-lacquis-strategy-mask)
  - [2.3 Approche 2 : Décalage des séries (strategy `shift`)](#23-approche-2--décalage-des-séries-strategy-shift)
- [3. Comparaison visuelle des deux approches](#3-comparaison-visuelle-des-deux-approches)
  - [3.1 Données brutes avant transformation](#31-données-brutes-avant-transformation)
  - [3.2 Après application de la stratégie `mask`](#32-après-application-de-la-stratégie-mask)
  - [3.3 Après application de la stratégie `shift`](#33-après-application-de-la-stratégie-shift)
- [4. Détails de l'implémentation](#4-détails-de-limplémentation)
  - [4.1 Détection des délais : `compare_and_detect_delays`](#41-détection-des-délais--compare_and_detect_delays)
  - [4.2 Calcul des délais applicables : `calculate_applicable_delay`](#42-calcul-des-délais-applicables--calculate_applicable_delay)
  - [4.3 Application des délais : `PublicationDelayTransformer`](#43-application-des-délais--publicationdelaytransformer)
- [5. Utilisation pratique dans un pipeline de prédiction](#5-utilisation-pratique-dans-un-pipeline-de-prédiction)
  - [5.1 Pipeline complet](#51-pipeline-complet)
  - [5.2 Validation croisée réaliste](#52-validation-croisée-réaliste)
- [6. Cas d'usage avancés](#6-cas-dusage-avancés)
  - [6.1 Délais variables par entité (données de panel)](#61-délais-variables-par-entité-données-de-panel)
- [7. Erreurs courantes à éviter](#7-erreurs-courantes-à-éviter)
- [8. Conclusion](#8-conclusion)

## Introduction

Les **délais de publication** (ou *publication delays*) des variables utilisées pour construire un modèle de prévision sur séries temporelles constituent un écueil dont il faut tenir compte pour simuler justement la performance prédictive de modèles en production. En pratique, les données économiques, financières ou opérationnelles ne sont en effet pas disponibles instantanément : elles sont publiées avec un retard qui peut aller de quelques jours à plusieurs mois. Ignorer ces délais lors de l'entraînement et de l'évaluation des modèles conduit à une surestimation systématique des performances.

Ce tutoriel présente deux approches complémentaires pour gérer ces délais : la **conservation de l'acquis** et le **décalage des séries**.

## 1. Comprendre les délais de publication

### 1.1 Qu'est-ce qu'un délai de publication ?

Le délai de publication est le temps qui s'écoule entre :
- La fin de la période de référence d'une observation
- Le moment où cette observation devient disponible

**Exemples** :
- Le PIB du T1 2024 (janvier-mars) est publié fin avril → délai de ~30 jours
- Les ventes d'un magasin du 15 janvier sont consolidées le 18 janvier → délai de 3 jours
- Le taux de chômage de septembre est publié début octobre → délai de ~7 jours

### 1.2 Impact sur la prévision

Au moment de faire une prévision à la date $t$, nous disposons uniquement des observations publiées avant $t$. Pour une série avec un délai de publication $d$ :

- **Dernière observation disponible** : $y_{t-d}$
- **Observations non disponibles** : $y_{t-d+1}, ..., y_{t-1}, y_t$

![Impact du délai de publication](../assets/release_delay_impact.png)

**Conséquence** : Si nous entraînons un modèle sur des données sans tenir compte des délais, nous créons un **data leakage temporel** en utilisant des informations qui ne seraient pas disponibles en production.

## 2. Deux approches pour gérer les délais

### 2.1 Vue d'ensemble

Il existe deux stratégies principales pour gérer les délais de publication, chacune avec ses avantages :

| Approche | Principe | Avantages | Cas d'usage |
|----------|----------|-----------|-------------|
| **Conservation de l'acquis** | Masquer les observations non disponibles | Préserve l'alignement temporel | Absence d'effets d'entraînement, seules les informations de la période en cours sont pertinentes |
| **Décalage des séries** | Shifter les séries selon leur délai | Utilise toute l'information disponible | Effets d'entraînement, réalisation d'une prévision en début de période |

### 2.2 Approche 1 : Conservation de l'acquis (strategy `mask`)

**Principe** : On conserve l'alignement temporel original mais on remplace par `NaN` les observations qui ne seraient pas encore disponibles au moment de la prévision.

![Conservation de l'acquis](../assets/mask_mode_approach.png)

**Algorithme** :
```
Pour chaque série x_i avec délai d_i :
    Pour chaque date t dans les données :
        Si t > date_prédiction - d_i :
            Masquer x_i(t)  # Remplacer par NaN
```

**Exemple pratique** :
```python
import pandas as pd
import numpy as np
from tsforecast.delays import PublicationDelayTransformer

# Données avec deux indicateurs
dates = pd.date_range('2024-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'date': dates,
    'GDP': np.random.randn(100),      # Délai: 30 jours
    'inflation': np.random.randn(100)  # Délai: 7 jours
})

# Configuration des délais
delays = {'GDP': 30, 'inflation': 7}

# Application du masquage
transformer = PublicationDelayTransformer(
    delays=delays,
    strategy='mask',
    prediction_date='2024-03-15',  # Date de référence
)

# Transformation
data_masked = transformer.fit_transform(data)

# Résultat :
# - GDP : masqué après 2024-02-14 (30 jours avant 2024-03-15)
# - inflation : masqué après 2024-03-08 (7 jours avant 2024-03-15)
```

**Avantages** :
- ✅ Alignement temporel préservé : `X_t` correspond toujours à la date `t`
- ✅ Simulation réaliste en production
- ✅ Facile à inverser

**Inconvénients** :
- ❌ Perte d'information (observations masquées)
- ❌ Nécessite des modèles robustes aux valeurs manquantes

### 2.3 Approche 2 : Décalage des séries (strategy `shift`)

**Principe** : On décale chaque série vers le futur selon son délai de publication, de sorte que la valeur disponible à la date `t` soit alignée avec cette date.

![Décalage des séries](../assets/shift_mode_approach.png)

**Algorithme** :
```
Pour chaque série x_i avec délai d_i :
    Shifter la série de d_i périodes vers le futur
    # x_i(t+d_i) ← x_i(t)
```

**Exemple pratique** :
```python
# Même configuration que précédemment
transformer = PublicationDelayTransformer(
    delays=delays,
    strategy='shift',  # Mode décalage
    prediction_date='2024-03-15'
)

# Transformation
data_shifted = transformer.fit_transform(data)

# Résultat :
# - GDP de date t apparaît maintenant à t+30 jours
# - inflation de date t apparaît à t+7 jours
# → Les valeurs à la ligne du 15 mars sont celles publiées ce jour-là
```

**Avantages** :
- ✅ Aucune perte d'information
- ✅ Utilise la dernière observation disponible
- ✅ Pas de valeurs manquantes générées
- ✅ Permet de tenir compte des effets d'entraînement

**Inconvénients** :
- ❌ Perd l'alignement temporel naturel
- ❌ Interprétation moins intuitive

## 3. Comparaison visuelle des deux approches

### 3.1 Données brutes avant transformation

Supposons deux séries avec des délais différents :
- Série A (bleu foncé) : délai de 5 jours
- Série B (bleu clair) : délai de 15 jours

### 3.2 Après application de la stratégie `mask`

Les observations trop récentes sont masquées (remplacées par NaN). L'alignement temporel est préservé mais certaines cellules deviennent vides.

### 3.3 Après application de la stratégie `shift`

Les séries sont décalées vers le futur. Aucune observation n'est perdue, mais les valeurs ne correspondent plus à leur date de référence originale.

![Comparaison des approches](../assets/approach_comparison.png)

## 4. Détails de l'implémentation

Cette section décrit en profondeur les trois composants clés du module de gestion des délais de publication : la fonction de détection `compare_and_detect_delays`, la fonction de calcul `calculate_applicable_delay`, et le transformeur `PublicationDelayTransformer`.

### 4.1 Détection des délais : `compare_and_detect_delays`

La fonction `compare_and_detect_delays` identifie les nouvelles observations et calcule leurs délais de publication en comparant deux jeux de données ou en analysant les valeurs manquantes d'un seul jeu de données par rapport à une date.

#### 4.1.1 Signature et arguments

```python
from tsforecast.delays import compare_and_detect_delays

df_delays = compare_and_detect_delays(
    new_data: pd.DataFrame,
    existing_data: Optional[pd.DataFrame] = None,
    download_date: Union[str, datetime] = None,
    detection_mode: str = 'new_only',
    reference_point: str = 'start',
    delay_unit: Literal['us', 's', 'D', 'microsecond', 'second', 'day'] = 'day',
    time_col: Optional[str] = None,
    panel_cols: Optional[List[str]] = None
)
```

| Argument | Type | Description |
|----------|------|-------------|
| `new_data` | `pd.DataFrame` | Nouveau jeu de données à analyser. L'index doit contenir les dates (ou un MultiIndex pour les données panel). |
| `existing_data` | `pd.DataFrame` ou `None` | Jeu de données existant pour comparaison. Si `None`, identifie l'observation non nulle la plus récente par variable. |
| `download_date` | `str` ou `datetime` | Date de téléchargement des données (`'today'` accepté). Si `None`, utilise `datetime.now()`. Une date sans fuseau horaire est lue en UTC face à une date avec fuseau (et réciproquement) ; deux dates avec fuseaux sont comparées comme des instants. |
| `detection_mode` | `'new_only'` ou `'all_changes'` | Mode de détection : `'new_only'` détecte uniquement les transitions NaN→valeur, `'all_changes'` détecte aussi les révisions. |
| `reference_point` | `'start'` ou `'end'` | Point de référence pour le calcul du délai : début ou fin de la période. |
| `delay_unit` | `str` | Unité du délai retourné : `'day'`/`'D'`, `'second'`/`'s'`, ou `'microsecond'`/`'us'`. |
| `time_col` | `str` ou `None` | Nom de la colonne temporelle (si non présente dans l'index). |
| `panel_cols` | `List[str]` ou `None` | Liste des colonnes identifiant les entités panel (si non présentes dans l'index). |

#### 4.1.2 Valeur retournée

La fonction retourne un `DataFrame` indexé par les niveaux d'entité des données (aucun pour une série) suivis d'un niveau `column` portant le nom de la variable ; une variable peut figurer sur plusieurs lignes. Les colonnes, dans l'ordre :

| Colonne | Description |
|---------|-------------|
| `observation_date` | Date de l'observation détectée |
| `has_changes` | Toujours `True` (marque les observations détectées) |
| `download_date` | Date de téléchargement des données à partir de laquelle est calculé le délai |
| `frequency` | Fréquence détectée du couple (entité, variable), en toutes lettres (`'daily'`, `'weekly'`, `'monthly'`, `'quarterly'`, `'annual'`, ...) ; `None` si elle est indétectable (une seule observation) — un avertissement est alors émis et `period_start`, `period_end` et `delay` valent `NaT` / `NaN` |
| `period_start` | Date de début de la période de référence de l'observation à la fréquence détectée |
| `period_end` | Borne de fin **exclusive** de la période de référence (premier instant de la période suivante : `2023-05-01` pour avril) |
| `reference_point` | Point de référence utilisé (`'start'` ou `'end'`) |
| `delay` | Délai de publication calculé (arrondi à l'entier supérieur de l'unité) entre le point de référence de la période de l'observation et la date de téléchargement ; négatif si le téléchargement précède ce point (par exemple une prévision utilisée comme observation) |
| `unit` | Unité du délai (`'day'`, `'second'`, `'microsecond'`) |

#### 4.1.3 Cas d'usage 1 : Comparaison de deux jeux de données

Lorsque `existing_data` est fourni, la fonction compare les deux jeux de données pour identifier les changements.

![Détection avec deux jeux de données](../assets/detect_delays_two_datasets.png)

```python
import pandas as pd
from datetime import datetime
from tsforecast.delays import compare_and_detect_delays

# Données existantes (téléchargées le 15 mars 2024)
existing_data = pd.DataFrame({
    'GDP': [1.2, 1.5, np.nan, np.nan]
}, index=pd.to_datetime(['2023-04-01', '2023-07-01', '2023-10-01', '2024-01-01']))
existing_data.index.freq = 'QS'

# Nouvelles données (téléchargées le 20 avril 2024)
new_data = pd.DataFrame({
    'GDP': [1.2, 1.6, 1.8, 2.1]  # Q3 révisé, Q4 et Q1 nouveaux
}, index=pd.to_datetime(['2023-04-01', '2023-07-01', '2023-10-01', '2024-01-01']))
new_data.index.freq = 'QS'

# Mode 'new_only' : détecte uniquement NaN → valeur
df_new_only = compare_and_detect_delays(
    new_data=new_data,
    existing_data=existing_data,
    download_date='2024-04-20',
    detection_mode='new_only',
    reference_point='end'
)
# Résultat : Q4 2023 et Q1 2024 détectés

# Mode 'all_changes' : détecte aussi les révisions
df_all_changes = compare_and_detect_delays(
    new_data=new_data,
    existing_data=existing_data,
    download_date='2024-04-20',
    detection_mode='all_changes',
    reference_point='end'
)
# Résultat : Q3 2023 (révision), Q4 2023 et Q1 2024 détectés
```

#### 4.1.4 Cas d'usage 2 : Analyse d'un jeu de données unique

Lorsque `existing_data=None`, la fonction identifie l'observation non-nulle la plus récente pour chaque variable (et chaque entité en données panel).

![Détection avec un seul jeu de données](../assets/detect_delays_single_dataset.png)

```python
# Identification des dernières observations disponibles
df_delays = compare_and_detect_delays(
    new_data=new_data,
    existing_data=None,  # Pas de données existantes
    download_date='2024-04-20',
    reference_point='end'
)
# Résultat : dernière observation non-NaN par colonne
```

Ce mode est utile pour :
- L'initialisation d'un système de suivi des délais
- L'estimation des délais à partir d'un snapshot
- La calibration sans historique de versions

#### 4.1.5 Impact du paramètre `reference_point`

Le choix du point de référence affecte significativement la valeur du délai calculé.

![Impact du point de référence](../assets/reference_point_impact.png)

| `reference_point` | Formule |
|-------------------|---------|
| `'start'` | `delay = download_date - period_start` |
| `'end'` | `delay = download_date - period_end` |

```python
# Exemple : Q1 2024 (1er jan - 31 mars), téléchargé le 15 mai 2024

# Avec reference_point='start'
# delay = 15 mai - 1er jan = 135 jours

# Avec reference_point='end'
# delay = 15 mai - 31 mars = 45 jours
```

---

### 4.2 Calcul des délais applicables : `calculate_applicable_delay`

La fonction `calculate_applicable_delay` convertit les délais détectés vers une fréquence et un point de référence cibles, puis les agrège pour obtenir un délai applicable par indicateur.

#### 4.2.1 Signature et arguments

```python
from tsforecast.delays import calculate_applicable_delay

df_applicable = calculate_applicable_delay(
    publication_delays: pd.DataFrame,
    reference_point: Literal['start', 'end'],
    frequency: Union[str, Dict[Union[str, tuple], str]],
    unit: Optional[Literal['us', 's', 'D', 'microsecond', 'second', 'day']] = None,
    indicators: Optional[List[str]] = None,
    aggregate_by_panel: bool = False,
    aggregation_method: Union[str, callable] = 'median'
)
```

| Argument | Type | Description |
|----------|------|-------------|
| `publication_delays` | `pd.DataFrame` | DataFrame retourné par `compare_and_detect_delays()`. |
| `reference_point` | `'start'` ou `'end'` | Point de référence cible pour le délai converti (la fin est la borne exclusive). |
| `frequency` | `str` ou `Dict` | Fréquence cible (`'M'`, `'Q'`, `'Y'`, `'monthly'`, etc.) ou dictionnaire : clés = indicateurs (`{'GDP': 'M'}`) et/ou couples complets de l'index `(entité, ..., indicateur)` (`{('FR', 'GDP'): 'M'}`), la clé du couple l'emportant sur celle de l'indicateur. Chaque ligne doit être couverte. |
| `unit` | `str` ou `None` | Unité cible (toute durée : `'D'`, `'h'`, `'W'`, `'s'`, ...), arrondie au supérieur. Si `None`, utilise l'unité des données d'entrée (les lignes agrégées ensemble doivent alors en partager une). |
| `indicators` | `List[str]` ou `None` | Liste des indicateurs à traiter. Si `None`, traite tous les indicateurs. |
| `aggregate_by_panel` | `bool` | Si `True`, calcule des délais séparés par entité panel. Sinon, agrège sur toutes les entités. |
| `aggregation_method` | `str` ou `callable` | Méthode d'agrégation : `'mean'`, `'median'`, `'max'`, `'min'`, ou fonction personnalisée. Un nom inconnu lève une `ValueError`. |

#### 4.2.2 Valeur de retour

Le DataFrame retourné est indexé par indicateur (ou par entité et indicateur si `aggregate_by_panel=True`) :

| Colonne | Description |
|---------|-------------|
| `delay` | Délai calculé après conversion et agrégation |
| `unit` | Unité du délai (son code, `'D'`, si `unit` est fourni) |
| `frequency` | Fréquence cible utilisée, telle que fournie |
| `reference_point` | Point de référence cible |
| `n_observations` | Nombre d'observations (de délai connu) utilisées dans l'agrégation |
| `aggregation_method` | Méthode d'agrégation utilisée |

Une ligne dont le délai est `NaN` ou la fréquence `None` (couple à fréquence indétectable renvoyé par `compare_and_detect_delays`) est **conservée avec un délai `NaN`** : elle n'est pas comptée dans `n_observations`, et un groupe composé uniquement de telles lignes a un délai `NaN` et `n_observations=0`.

#### 4.2.3 Conversion de fréquence

Lors de la conversion vers une fréquence différente, la fonction identifie la sous-période pertinente en utilisant la date d'observation (`observation_date`).

![Conversion de fréquence](../assets/frequency_conversion_delay.png)

**Cas de conversion vers une fréquence plus élevée** (ex: trimestriel → mensuel) :

La sous-période sélectionnée est celle qui contient `observation_date`. Par exemple, pour des données Q1 2024 avec `observation_date` au 15 mars :
- La période cible est **mars** (et non janvier ou février)
- Le délai est recalculé par rapport aux bornes de mars

```python
# Données trimestrielles avec observation_date = 15 mars
# Conversion Q → M

df_applicable = calculate_applicable_delay(
    publication_delays=df_delays,
    reference_point='end',
    frequency='M',  # Mensuel
    aggregation_method='median'
)
# La sous-période mars est sélectionnée car elle contient observation_date
```

**Une fréquence cible par couple (entité, indicateur)** :

```python
df_applicable = calculate_applicable_delay(
    publication_delays=df_delays,
    reference_point='start',
    frequency={('DE', 'GDP'): 'quarterly', 'GDP': 'monthly', 'CPI': 'monthly'},
    aggregate_by_panel=True,
)
# Allemagne : PIB trimestriel ; PIB des autres entités et CPI : mensuel
```

Sans `aggregate_by_panel`, les entités d'un même indicateur doivent partager la même fréquence cible (leurs délais sont agrégés ensemble) ; sinon une `ValueError` demande d'agréger par entité.

**Cas de conversion vers une fréquence plus basse** (ex: trimestriel → annuel) :

La période englobante à la fréquence cible est utilisée.

#### 4.2.4 Conversion du point de référence

La conversion entre points de référence suit un processus en trois étapes.

![Conversion du point de référence](../assets/reference_point_conversion.png)

1. **Reconstruction de la date de téléchargement `download_date`** : À partir du délai original et du point de référence original
2. **Détermination de la période cible** : Identification des bornes à la fréquence cible
3. **Calcul du délai converti** : Par rapport au nouveau point de référence

```python
# Conversion reference_point='end' (45 jours) → reference_point='start'

# Étape 1: download_date = period_end + 45 jours = 31 mars + 45 = 15 mai
# Étape 2: period_start (Q1) = 1er janvier
# Étape 3: converted_delay = 15 mai - 1er jan = 135 jours
```

#### 4.2.5 Exemple complet

```python
from tsforecast.delays import compare_and_detect_delays, calculate_applicable_delay

# 1. Détection des délais
df_delays = compare_and_detect_delays(
    new_data=economic_data,
    download_date='2024-04-20',
    reference_point='end'
)

# 2. Calcul des délais applicables (conversion vers mensuel, ref='start')
df_applicable = calculate_applicable_delay(
    publication_delays=df_delays,
    reference_point='start',
    frequency='M',
    unit='day',
    aggregation_method='median'
)

print(df_applicable)
#              delay unit frequency reference_point  n_observations aggregation_method
# column
# GDP           75.0    D         M           start               4             median
# inflation     45.0    D         M           start               4             median
```

---

### 4.3 Application des délais : `PublicationDelayTransformer`

Le `PublicationDelayTransformer` est un transformeur compatible avec l'API scikit-learn qui applique les délais de publication aux données selon la stratégie (`'shift'` ou `'mask'`).

#### 4.3.1 Signature et arguments

```python
from tsforecast.delays import PublicationDelayTransformer

transformer = PublicationDelayTransformer(
    delays: Union[Dict[str, float], pd.DataFrame],
    prediction_date: Union[str, datetime] = 'today',
    strategy: Union[Literal['shift', 'mask'], Dict[str, Literal['shift', 'mask']]] = 'shift',
    target_frequency: Optional[Union[str, Dict[str, str]]] = None,
    delay_unit: Optional[Union[str, Dict[str, str]]] = None,
    reference_point: Optional[Union[Literal['start', 'end'], Dict[str, Literal['start', 'end']]]] = None,
    handle_missing_delays: Literal['ignore', 'warn', 'error'] = 'warn',
    default_values: Optional[Dict[str, Union[int, float, str]]] = None
)
```

| Argument | Type | Description |
|----------|------|-------------|
| `delays` | `Dict` ou `pd.DataFrame` | Délais par variable. Si DataFrame : une ligne par variable, colonne `'delay'`, variable dans une colonne `'column'` ou dans l'index (niveau `'column'`, à défaut le dernier) — la sortie de `calculate_applicable_delay` convient telle quelle ; colonnes optionnelles `'unit'`, `'reference_point'`, `'frequency'`. Un délai `NaN` est inconnu. Une variable listée plusieurs fois (délais par entité) est refusée : voir §6.1 (fabrique). |
| `prediction_date` | `str` ou `datetime` | Date de prédiction de référence. Accepte `'today'` pour la date courante. |
| `strategy` | `str` ou `Dict` | Stratégie d'application : `'shift'` (décalage) ou `'mask'` (masquage). Peut être spécifié par variable ; une variable retardée absente du dictionnaire est laissée telle quelle (avertissement). |
| `target_frequency` | `str`, `Dict` ou `None` | Fréquence cible pour la stratégie `'mask'`. Ignoré pour `'shift'`. |
| `delay_unit` | `str`, `Dict` ou `None` | Unité des délais, globale ou par variable. Si `None`, inférée depuis le DataFrame, puis `default_values`. Une colonne retardée sans unité lève une `ValueError`. |
| `reference_point` | `str`, `Dict` ou `None` | Point de référence des délais (`'start'` / `'end'`), global ou par variable. Si `None`, inféré depuis le DataFrame, puis `default_values`. |
| `handle_missing_delays` | `str` | Variables de `X` sans délai connu (laissées telles quelles) : `'warn'` (défaut) les signale, `'error'` lève une `ValueError`, `'ignore'` se tait. |
| `default_values` | `Dict` ou `None` | Valeurs par défaut des colonnes de `X` qui en manquent : clés `'delay'`, `'unit'`, `'reference_point'` (et `'target_frequency'` pour `'mask'`). Avec lui, toutes les colonnes de `X` reçoivent un délai. Ignoré avec une stratégie par variable. |

#### 4.3.2 Attributs après `fit`

| Attribut | Description |
|----------|-------------|
| `prediction_date_` | Date de prédiction résolue (objet `datetime`) |
| `detected_frequencies_` | Dictionnaire des fréquences détectées par colonne |
| `shift_params` | Paramètres de décalage par colonne : `{'n_periods': int, 'frequency': str}` |
| `mask_params` | Paramètres de masquage par colonne : `{'n_obs': int, 'mask_frequency': str, 'how': str}` |
| `fit_report_` | `DelayFitReport` : paramétrage résolu de chaque colonne et son origine |
| `auxiliary_transformers_` | Transformeurs auxiliaires du dernier `transform`, utilisés par `inverse_transform` (qui exige donc un `transform` préalable) |

**Index, aller-retour et panel.** Le décalage déplace les dates sans perdre de valeur : l'index de sortie est
l'union des dates décalées des colonnes (il s'allonge, et les dates qu'aucune colonne n'occupe plus disparaissent),
trié par date (pour un panel : par entité, dans l'ordre d'entrée, puis par date).
`inverse_transform` supprime les lignes ajoutées par `transform`, si bien que `inverse_transform(transform(X))`
restitue `X` à l'identique. Un panel peut être passé directement : les **mêmes délais** s'appliquent à toutes les
entités (un avertissement le rappelle) ; pour des délais propres à chaque entité, voir la fabrique (§6.1).

**Pipeline avec une cible.** Seul `X` est transformé : dans une `XYPipeline` suivie d'un estimateur, le décalage
change les lignes de `X` sans toucher à `y`, et une étape ultérieure doit réaligner `y` sur le nouvel index. La
stratégie `'mask'` (sans bascule vers le décalage) conserve l'index et s'enchaîne directement.

#### 4.3.3 Calcul du nombre de périodes à décaler (stratégie `'shift'`)

Le cœur de la stratégie `'shift'` est le calcul du nombre de périodes à décaler pour chaque série. Il se fait
**sur le calendrier** : l'observation de la période commençant en *s* est publiée en *s + délai* (point de
référence `'start'`) ou à la fin (exclusive) de sa période + délai (`'end'`). `n_periods` vaut moins le nombre de
périodes entre la période de `prediction_date` et la **dernière période publiée à `prediction_date`** (une
période publiée le jour même compte comme publiée). La durée réelle des mois (28 à 31 jours) et les années
bissextiles sont prises en compte ; le délai, lui, est une durée (un délai en mois compte 30 jours par mois,
comme partout dans le paquet).

**Exemple détaillé :**

```python
# Configuration
prediction_date = '2024-02-15'
delay = 45  # jours
reference_point = 'end'

# Série mensuelle :
# - janvier 2024 finit le 1er février, + 45 jours = 17 mars    -> non publié au 15 février
# - décembre 2023 finit le 1er janvier, + 45 jours = 15 février -> publié le jour même
# Dernière période publiée : décembre 2023, deux mois avant février -> n_periods = -2
```

**Impact de la fréquence :**

| Fréquence | Dernière période publiée au 15 février 2024 | `n_periods` |
|-----------|---------------------------------------------|-------------|
| Mensuel (M) | Décembre 2023 (fin 1er janv. + 45 j = 15 févr.) | -2 |
| Trimestriel (Q) | T4 2023 (fin 1er janv. + 45 j = 15 févr.) | -1 |
| Annuel (Y) | 2023 (fin 1er janv. + 45 j = 15 févr.) | -1 |

Une formule approchée (`-ceil((délai - écoulé) / durée)` avec un mois de 30 jours) se trompait d'une période
près des frontières calendaires, dans les deux sens : une valeur publiée le jour de la prédiction était repoussée
après elle, ou une valeur publiée le lendemain devenait visible (ANO-DELAYS-042, corrigée).

#### 4.3.4 Calcul du nombre d'observations à masquer (stratégie `'mask'`)

La stratégie `'mask'` compte de la même façon, sur le calendrier, les **périodes de l'index** (pas celles de la
série) non encore publiées à `prediction_date` : c'est le nombre de dernières observations masquées dans chaque
période cible. Avec `'end'`, le délai part de la fin de la période de la série contenant l'observation.

**Vérification de faisabilité (`can_mask`) :** le masquage doit laisser au moins une observation dans chaque
période cible. Le nombre d'observations à masquer est comparé au plus petit nombre de périodes de l'index dans
une période cible couverte par les données, compté sur le calendrier (`FrequencyConverter.count_subperiods_per_period` :
28 jours en février 2023, 3 mois par trimestre). Si `can_mask = False`, la colonne est automatiquement basculée
vers la stratégie `'shift'` avec un avertissement : elle reçoit alors le décalage calculé comme au §4.3.3.

**Exemples :**

```python
# Index mensuel, trimestre cible, 45 jours depuis la fin, prédiction au 15 février 2024
# - février 2024 et janvier 2024 (publié le 17 mars) ne sont pas publiés, décembre 2023 l'est
# -> 2 mois masqués par trimestre ; 2 < 3 mois par trimestre : masquage possible

# Index mensuel, trimestre cible, 80 jours depuis le début, prédiction au 15 décembre 2023
# - décembre, novembre et octobre (publié le 20 décembre) ne sont pas publiés
# -> 3 mois à masquer = un trimestre entier : bascule vers 'shift' (n_periods = -3)
```

#### 4.3.5 Exemple d'utilisation complète

```python
import pandas as pd
from datetime import datetime
from tsforecast.delays import (
    compare_and_detect_delays,
    calculate_applicable_delay,
    PublicationDelayTransformer
)

# 1. Création des données
dates = pd.date_range('2024-01-01', periods=100, freq='D')
data = pd.DataFrame({
    'GDP': np.random.randn(100),
    'inflation': np.random.randn(100),
    'retail_sales': np.random.randn(100)
}, index=dates)

# 2. Définition des délais (méthode directe)
delays = {'GDP': 45, 'inflation': 30, 'retail_sales': 20}

# 3a. Stratégie 'shift' : décalage des séries
transformer_shift = PublicationDelayTransformer(
    delays=delays,
    strategy='shift',
    prediction_date='2024-03-15',
    delay_unit='D',
    reference_point='end'
)

data_shifted = transformer_shift.fit_transform(data)

# Vérification des paramètres calculés
print(transformer_shift.shift_params)
# {'GDP': {'n_periods': -2, 'frequency': 'D'},
#  'inflation': {'n_periods': -1, 'frequency': 'D'},
#  'retail_sales': {'n_periods': -1, 'frequency': 'D'}}

# 3b. Stratégie 'mask' : masquage des observations récentes
transformer_mask = PublicationDelayTransformer(
    delays=delays,
    strategy='mask',
    prediction_date='2024-03-15',
    delay_unit='D',
    reference_point='end',
    target_frequency='M'
)

data_masked = transformer_mask.fit_transform(data)

# Vérification des paramètres calculés
print(transformer_mask.mask_params)
# {'GDP': {'n_obs': 45, 'mask_frequency': 'M', 'how': 'last'},
#  'inflation': {'n_obs': 30, 'mask_frequency': 'M', 'how': 'last'},
#  'retail_sales': {'n_obs': 20, 'mask_frequency': 'M', 'how': 'last'}}

# 4. Transformation inverse
data_original = transformer_shift.inverse_transform(data_shifted)
```

#### 4.3.6 Utilisation avec le DataFrame de délais

Le transformeur peut également recevoir directement le DataFrame retourné par `calculate_applicable_delay` (variable dans l'index `column`, sans `reset_index`), ce qui permet l'inférence automatique des paramètres :

```python
# Pipeline complet avec inférence des paramètres
df_delays = compare_and_detect_delays(
    new_data=data,
    download_date='2024-03-20',
    reference_point='end'
)

df_applicable = calculate_applicable_delay(
    publication_delays=df_delays,
    reference_point='end',
    frequency='M',
    unit='D',
    aggregation_method='median'
)

# Le transformer infère delay_unit, reference_point et target_frequency
transformer = PublicationDelayTransformer(
    delays=df_applicable,  # DataFrame avec métadonnées
    strategy='mask',
    prediction_date='2024-03-15'
    # delay_unit, reference_point, target_frequency inférés automatiquement
)

data_transformed = transformer.fit_transform(data)
```

#### 4.3.7 Stratégies mixtes par variable

Il est possible de spécifier une stratégie différente pour chaque variable :

```python
transformer = PublicationDelayTransformer(
    delays=delays,
    strategy={
        'GDP': 'mask',          # Masquage pour le PIB
        'inflation': 'shift',   # Décalage pour l'inflation
        'retail_sales': 'mask'  # Masquage pour les ventes
    },
    prediction_date='2024-03-15',
    delay_unit='D',
    reference_point='end',
    target_frequency='M'
)
```

## 5. Utilisation pratique dans un pipeline de prédiction

### 5.1 Pipeline complet

```python
# Importation des modules
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from tsforecast.delays import PublicationDelayTransformer

# Définition des délais (en jours)
delays = {
    'GDP': 45,
    'CPI': 30,
    'unemployment': 15,
    'retail_sales': 20
}

# Construction du pipeline
pipeline = Pipeline([
    ('delays', PublicationDelayTransformer(
        delays=delays,
        strategy='mask',
        prediction_date='today'
    )),
    ('model', RandomForestRegressor(n_estimators=100))
])

# Entraînement
pipeline.fit(X_train, y_train)

# Prédiction (les délais sont automatiquement appliqués)
y_pred = pipeline.predict(X_test)
```

### 5.2 Validation croisée réaliste

Pour une évaluation réaliste, il est **crucial** d'appliquer les délais de publication à la fois sur les données d'entraînement et de test, car en production, le modèle sera entraîné avec les mêmes contraintes de disponibilité des données.

#### 5.2.1 Approche recommandée : Application avant le split

La meilleure pratique consiste à appliquer le `PublicationDelayTransformer` sur l'ensemble complet des données pour chaque fold, en utilisant la date de début du test comme `prediction_date`. Cela simule exactement ce qui serait disponible au moment de la prédiction.

```python
# Importation des modules
from tsforecast.crossvals import TSOutOfSampleSplit
from tsforecast.delays import PublicationDelayTransformer
from sklearn.ensemble import RandomForestRegressor

# Configuration de la validation croisée
splitter = TSOutOfSampleSplit(
    n_splits=5,
    test_size=30,
    gap=5  # Horizon de prévision
)

# Évaluation avec délais appliqués sur train ET test
results = []
for train_idx, test_idx in splitter.split(X):
    # Identification de la date de prédiction (première date du test)
    prediction_date = X.index[test_idx[0]]

    # Application des délais sur TOUTES les données avec cette date de référence
    transformer = PublicationDelayTransformer(
        delays=delays,
        strategy='mask',
        prediction_date=prediction_date,
        delay_unit='D',
        reference_point='end'
    )

    # Transformation de l'ensemble complet
    X_delayed = transformer.fit_transform(X)

    # Séparation train/test APRÈS application des délais
    X_train = X_delayed.iloc[train_idx]
    X_test = X_delayed.iloc[test_idx]
    y_train_split = y.iloc[train_idx]
    y_test_split = y.iloc[test_idx]

    # Entraînement et évaluation
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train_split)
    y_pred = model.predict(X_test)

    results.append(evaluate_predictions(y_test_split, y_pred))

# Agrégation des résultats
mean_score = np.mean(results)
std_score = np.std(results)
print(f"Score moyen : {mean_score:.4f} ± {std_score:.4f}")
```

**Pourquoi cette approche ?**
- ✅ **Réalisme** : Simule exactement les données disponibles au moment de la prédiction
- ✅ **Cohérence** : Train et test reflètent les mêmes contraintes de publication
- ✅ **Prévention du data leakage** : Aucune observation future n'est utilisée

#### 5.2.2 Approche avec Pipeline sklearn (production)

Pour la mise en production, on peut utiliser un Pipeline sklearn standard :

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from tsforecast.delays import PublicationDelayTransformer

# Définition du pipeline
pipeline = Pipeline([
    ('delays', PublicationDelayTransformer(
        delays=delays,
        strategy='mask',
        prediction_date='today',  # Date dynamique en production
        delay_unit='D',
        reference_point='end'
    )),
    ('model', RandomForestRegressor(n_estimators=100))
])

# Entraînement sur toutes les données historiques
pipeline.fit(X_train, y_train)

# Prédiction (les délais sont automatiquement appliqués)
y_pred = pipeline.predict(X_test)
```

**Note importante** : Pour la validation croisée avec un Pipeline, il faudrait créer un transformer personnalisé qui ajuste dynamiquement `prediction_date` selon le fold. L'approche manuelle (5.2.1) est donc recommandée pour l'évaluation.

## 6. Cas d'usage avancés

### 6.1 Délais variables par entité (données de panel)

Pour les données de panel, chaque entité peut avoir des délais différents. Passé directement à un `PublicationDelayTransformer`, un panel reçoit les mêmes délais pour toutes les entités (avec un avertissement), et un tableau de délais par entité est refusé. On utilise alors `PanelwiseTransformer` avec les fonctions helper pour appliquer les délais adéquats à chaque entité grâce à un `PublicationDelayTransformer` différent (une entité absente du tableau lève une `KeyError` avec la fabrique, ou garde le transformeur de base avec `entity_kwargs`) :

**Méthode 1 : Factory pattern avec `create_delay_transformer_factory`**

```python
from tsforecast.delays import (
    PublicationDelayTransformer,
    compare_and_detect_delays,
    calculate_applicable_delay,
    create_delay_transformer_factory
)
from tsforecast.panel import PanelwiseTransformer

# 1. Détection des délais par entité
df_delays = compare_and_detect_delays(
    new_data=panel_data,
    download_date='2024-03-15',
    panel_cols=['country']
)

# 2. Calcul des délais applicables par entité
df_applicable = calculate_applicable_delay(
    publication_delays=df_delays,
    reference_point='end',
    frequency='M',
    unit='D',
    aggregate_by_panel=True  # Agrège par entité
)

# 3. Création de la factory
factory = create_delay_transformer_factory(
    df_delays=df_applicable,
    strategy='mask',
    prediction_date='2024-03-15'
)

# 4. Application avec PanelwiseTransformer
panel_transformer = PanelwiseTransformer(
    transformer=factory,
    panel_cols=['country']
)

data_transformed = panel_transformer.fit_transform(panel_data)
```

**Méthode 2 : entity_kwargs avec `prepare_entity_kwargs_from_delays`**

```python
from tsforecast.delays import prepare_entity_kwargs_from_delays

# Préparation des kwargs par entité
entity_kwargs = prepare_entity_kwargs_from_delays(
    df_delays=df_applicable,
    strategy='shift'
)

# Application avec un transformer de base
panel_transformer = PanelwiseTransformer(
    transformer=PublicationDelayTransformer(
        delays={},  # Sera remplacé par entity_kwargs
        prediction_date='2024-03-15'
    ),
    entity_kwargs=entity_kwargs,
    panel_cols=['country']
)

data_transformed = panel_transformer.fit_transform(panel_data)
```


## 7. Erreurs courantes à éviter

### ❌ Erreur 1 : Ignorer les délais lors de l'évaluation

```python
# INCORRECT : Évalue sans tenir compte des délais
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # Surestimation !
```

✅ **Correction** :
```python
# CORRECT : Applique les délais avant l'évaluation
transformer = PublicationDelayTransformer(delays=delays, strategy='mask')
X_test_real = transformer.fit_transform(X_test)
score = model.score(X_test_real, y_test)
```

### ❌ Erreur 2 : Confusion entre délai et horizon

```python
# INCORRECT : Confond délai de publication et horizon de prévision
gap = horizon  # ⚠️ Incomplet !
```

✅ **Correction** :
```python
# CORRECT : Prend en compte les deux
gap = horizon + max_publication_delay
```

## 8. Conclusion

La gestion rigoureuse des délais de publication est essentielle pour :

1. **Obtenir des évaluations réalistes** : Éviter la surestimation des performances
2. **Déployer des modèles fonctionnels** : Simuler les vraies conditions de production

**Ressources complémentaires** :
- Documentation de `tsforecast.delays`
- Tutoriel sur la validation croisée temporelle