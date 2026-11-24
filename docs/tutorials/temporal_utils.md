# Tutoriel : utilitaires temporels

Ce tutoriel applique les fonctions de `tsforecast.utils` à un même jeu de données à fréquences
mixtes. Il montre comment lire et comparer des fréquences, détecter celle de chaque variable, délimiter des périodes,
changer la position des dates puis leur fréquence, convertir des durées et valider la structure d'un
panel. Le [concept](../concepts/temporal_utils.md) énonce les règles que ces fonctions appliquent.

## Table des matières

- [1. Données](#1-donnees)
- [2. Lire, normaliser et comparer des fréquences](#2-lire-normaliser-et-comparer-des-frequences)
- [3. Détecter une fréquence : `detect_frequency`](#3-detecter-une-frequence-detect_frequency)
- [4. Délimiter une période : `get_period_boundaries`](#4-delimiter-une-periode-get_period_boundaries)
- [5. Changer la position des dates : `convert_position`](#5-changer-la-position-des-dates-convert_position)
- [6. Changer de fréquence : `FrequencyConverter`](#6-changer-de-frequence-frequencyconverter)
- [7. Convertir une durée : `convert_duration`](#7-convertir-une-duree-convert_duration)
- [8. Valider un jeu de données : `validate_temporal_data`](#8-valider-un-jeu-de-donnees-validate_temporal_data)

## 1. Données

Un indice de prix mensuel (`cpi`), un PIB trimestriel (`gdp`) et une population observée une seule
fois (`population`) partagent un index mensuel, daté du premier jour de chaque mois.

```python
import numpy as np
import pandas as pd

dates = pd.date_range("2023-01-01", "2024-12-01", freq="MS", name="date")
data = pd.DataFrame(
    {"cpi": np.linspace(100.0, 104.6, len(dates)).round(1), "gdp": np.nan, "population": np.nan},
    index=dates,
)
data.loc[dates[::3], "gdp"] = [500.0, 503.0, 507.0, 510.0, 514.0, 516.0, 521.0, 525.0]
data.loc["2023-01-01", "population"] = 68.0
```

## 2. Lire, normaliser et comparer des fréquences

Une fréquence s'écrit `[N]BASE[S|E][-ANCRE]`, pour multiplicateur, base, position et ancre.
`parse_frequency` la décompose, `build_frequency_string` l'assemble, et `normalize_frequency` la
valide puis la rend au niveau de détail de `return_format` (`'base'`, `'with_position'`, `'full'`,
`'components'`). Les noms littéraux (`'monthly'`, `'quarterly'`, `'annual'`...) sont acceptés.

```python
from tsforecast.utils.frequency import (
    canonicalize_frequency, is_higher_frequency, normalize_frequency, to_pandas_freq,
)
from tsforecast.utils.parse import build_frequency_string, parse_frequency

parse_frequency("2QS-FEB")
# ParsedFrequency(freq='Q', position='S', suffix='FEB', multiplier=2)
build_frequency_string("Q", "S", "FEB", 2)                       # '2QS-FEB'
normalize_frequency("2QS-FEB")                                   # 'Q'
normalize_frequency("2QS-FEB", return_format="with_position")    # 'QS'
normalize_frequency("quarterly", return_format="full")           # 'Q'
```

Une base nue (`'M'`, `'Q'`, `'Y'`, `'SM'`) est un code du paquet, que pandas 3 refuse comme alias.
`to_pandas_freq` donne l'alias à transmettre à pandas ; `canonicalize_frequency` ramène les ancres
équivalentes à une seule écriture, celle que rendent les fonctions de détection.

```python
to_pandas_freq("Q")                                              # 'QE'
pd.date_range("2024-01-01", periods=2, freq=to_pandas_freq("Q"))
# DatetimeIndex(['2024-03-31', '2024-06-30'], dtype='datetime64[us]', freq='QE-DEC')
canonicalize_frequency("QS-OCT")                                 # 'QS-JAN'
```

`is_higher_frequency(a, b)` teste si `a` est plus fine que `b`, multiplicateur compris :

```python
is_higher_frequency("D", "M"), is_higher_frequency("MS", "2MS")  # (True, True)
is_higher_frequency("3MS", "QS"), is_higher_frequency("QS", "3MS")  # (False, False)
```

## 3. Détecter une fréquence : `detect_frequency`

### 3.1 Par variable

`detect_frequency` lit la fréquence de chaque colonne sur ses **valeurs observées** : `gdp`, portée
par un index mensuel, est trimestrielle. Une colonne de moins de deux observations reçoit `None`.

```python
from tsforecast.utils.frequency import detect_frequency, detect_index_frequency

detect_index_frequency(data.index)                               # 'M'
detect_frequency(data, return_format="full")
# {'cpi': 'MS', 'gdp': 'QS-JAN', 'population': None}
detect_frequency(data, check_consistency=True, consistency_mode="highest")   # 'M'
```

Sur un panel, la détection se fait par couple (entité, colonne) ; les clés sont des tuples
`(entité..., colonne)` :

```python
panel = pd.concat(
    {"FR": data[["cpi"]], "DE": data[["gdp"]].rename(columns={"gdp": "cpi"}).dropna()},
    names=["country"],
)
detect_frequency(panel)
# {('DE', 'cpi'): 'Q', ('FR', 'cpi'): 'M'}
```

### 3.2 Index lacunaire

Quand pandas ne reconnaît aucune fréquence, la détection retient l'espacement modal entre dates
consécutives : une grille mensuelle lacunaire reste `'MS'`. `is_regular` teste la régularité et
`regularize` complète la grille par des lignes `NaN`.

```python
from tsforecast.utils.frequency import is_regular, regularize

gappy = data["cpi"].drop(pd.to_datetime(["2023-05-01", "2023-06-01"]))
detect_index_frequency(gappy.index, return_format="full")        # 'MS'
is_regular(gappy)                                                # False
regularize(gappy)["2023-04":"2023-07"]
# date
# 2023-04-01    100.6
# 2023-05-01      NaN
# 2023-06-01      NaN
# 2023-07-01    101.2
# Freq: MS, Name: cpi, dtype: float64
```

### 3.3 Arguments

<!-- doc-args: tsforecast.utils.frequency.detect_frequency -->
| Argument | Type | Description |
|----------|------|-------------|
| `data` | `pd.Series` ou `pd.DataFrame` | Série ou jeu de données indexé par des dates, ou panel (`MultiIndex`, dates au dernier niveau). |
| `time_col` | `str` ou `None` | Colonne des dates d'un `DataFrame` plat. |
| `panel_cols` | `List[str]` ou `None` | Colonnes des entités d'un `DataFrame` plat. |
| `return_format` | `'base'`, `'with_position'`, `'full'`, `'components'` | Forme des fréquences rendues : `'M'`, `'MS'`, `'2MS'` ou `ParsedFrequency` (défaut `'base'`). |
| `check_consistency` | `bool` | Si `True`, rend une seule fréquence au lieu d'un dictionnaire (défaut `False`). |
| `consistency_mode` | `'modal'` ou `'highest'` | Fréquence retenue : la plus fréquente, ou la plus fine (défaut `'modal'`). |
| `strict` | `bool` | En mode `'modal'`, `None` dès que deux fréquences diffèrent (défaut `True`). Les colonnes indétectables sont ignorées. |
<!-- /doc-args -->

Sans `check_consistency`, le résultat est un dictionnaire `{colonne: fréquence}`, ou
`{(entité..., colonne): fréquence}` sur un panel ; pour une série, une fréquence. Référence
complète : [API `frequency`](../api/utils/frequency.md).

## 4. Délimiter une période : `get_period_boundaries`

`get_period_boundaries(date, frequency)` renvoie le début et la fin de la période qui contient
`date`. La fin est exclue : c'est le premier instant de la période suivante.

```python
from tsforecast.utils.time import get_labelled_period, get_period_boundaries, get_period_end

date = pd.Timestamp("2024-05-15")
get_period_boundaries(date, "Q")
# (Timestamp('2024-04-01 00:00:00'), Timestamp('2024-07-01 00:00:00'))
get_period_boundaries(date, "QS-FEB")       # trimestres février-avril, mai-juillet...
# (Timestamp('2024-05-01 00:00:00'), Timestamp('2024-08-01 00:00:00'))
get_period_boundaries(date, "W-WED")        # semaines du jeudi au mercredi
# (Timestamp('2024-05-09 00:00:00'), Timestamp('2024-05-16 00:00:00'))
get_period_end(date, "M")                   # Timestamp('2024-06-01 00:00:00')
```

Une fréquence multipliée découpe le temps en blocs de *n* périodes. `get_period_boundaries` les
aligne sur janvier 1970, ou sur `origin` ; `get_labelled_period` lit la date comme l'étiquette d'un
bloc d'une grille de données (`pd.date_range(..., freq="2MS")`) :

```python
label = pd.Timestamp("2024-04-01")
get_period_boundaries(label, "2MS")         # blocs janvier-février, mars-avril...
# (Timestamp('2024-03-01 00:00:00'), Timestamp('2024-05-01 00:00:00'))
get_labelled_period(label, "2MS")           # bloc qui commence à l'étiquette
# (Timestamp('2024-04-01 00:00:00'), Timestamp('2024-06-01 00:00:00'))
```

<!-- doc-args: tsforecast.utils.time.get_period_boundaries -->
| Argument | Type | Description |
|----------|------|-------------|
| `date` | `datetime`, `Timestamp`, `numpy.datetime64` ou `Period` | Date de référence ; un `Period` est représenté par son premier instant. Le fuseau horaire est conservé. |
| `frequency` | `str` | Fréquence des périodes : code pandas (ancre et multiplicateur compris) ou nom littéral. |
| `origin` | `Timestamp`, `datetime` ou `None` | Origine des blocs d'une fréquence multipliée (défaut : janvier 1970). |
<!-- /doc-args -->

`get_period_start` et `get_period_end` prennent les mêmes arguments ; `get_labelled_period` prend
`date` et `frequency`. Les bornes sont dans l'unité de `date`. Référence complète :
[API `time`](../api/utils/time.md).

## 5. Changer la position des dates : `convert_position`

`convert_position(value, from_position, to_position, freq=None)` déplace les dates d'un index, d'une
série ou d'un `DataFrame` à l'autre bord de leur période ; `convert_offset` convertit une chaîne de
fréquence. Les fréquences sans position (`D`, `W`, `h`...) sont rendues inchangées.

```python
from tsforecast.utils.position import convert_offset, convert_position

convert_position(data["cpi"].iloc[:3], "start", "end")
# date
# 2023-01-31    100.0
# 2023-02-28    100.2
# 2023-03-31    100.4
# Name: cpi, dtype: float64
convert_offset("QS-FEB", "end"), convert_offset("2MS", "end"), convert_offset("D", "end")
# ('QE-JAN', '2ME', 'D')
```

`freq`, détectée si elle est absente, délimite les périodes : une fréquence plus grossière que les
données, qui fusionnerait des dates, lève une `ValueError`.

## 6. Changer de fréquence : `FrequencyConverter`

### 6.1 Agréger et interpoler

`FrequencyConverter().convert_frequency` agrège vers une fréquence moins fine et interpole vers une
fréquence plus fine. Le `method` par défaut, `'mean'`, est une agrégation : une interpolation doit
préciser la méthode associée pour imputer une valeur aux périodes manquantes.

```python
from tsforecast.utils.frequency import FrequencyConverter

converter = FrequencyConverter()
converter.convert_frequency(data["cpi"], "Q", method="mean").head(3)
# date
# 2023-01-01    100.2
# 2023-04-01    100.8
# 2023-07-01    101.4
# Freq: QS-JAN, Name: cpi, dtype: float64
converter.convert_frequency(data["gdp"].dropna(), "M", method="linear").tail(4)
# date
# 2024-09-01    523.666667
# 2024-10-01    525.000000
# 2024-11-01    525.000000
# 2024-12-01    525.000000
# Freq: MS, Name: gdp, dtype: float64
```

Les étiquettes suivent la position de la source (`'MS'` : début de trimestre), sauf position
explicite dans la cible (`'QE'`) ou dans `target_position`. L'interpolation prolonge la dernière
observation (octobre) sur toute sa période (novembre, décembre).

### 6.2 Périodes incomplètes

Une période incomplète est agrégée sur ses seules observations ; `full_periods_only=True` la met à
`NaN` :

```python
sales = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.date_range("2024-01-01", periods=5, freq="MS"))
converter.convert_frequency(sales, "Q", method="sum").tolist()                          # [6.0, 9.0]
converter.convert_frequency(sales, "Q", method="sum", full_periods_only=True).tolist()  # [6.0, nan]
```

### 6.3 Plusieurs colonnes

Chaque colonne part de sa propre fréquence. Un dictionnaire `{colonne: fréquence}` ne convertit que
les colonnes qu'il nomme ; l'index de sortie est la grille cible, et `gdp`, déjà trimestrielle, garde
ses dates. `FrequencyAligner` convertit les mêmes colonnes en conservant l'index d'entrée :

```python
from tsforecast.utils.frequency import FrequencyAligner

converter.convert_frequency(data[["cpi", "gdp"]], {"cpi": "Q"}, method="mean").head(2)
#               cpi    gdp
# date
# 2023-01-01  100.2  500.0
# 2023-04-01  100.8  503.0
FrequencyAligner().convert_to_target(data[["cpi", "gdp"]], ["cpi"], "Q", agg_method="mean").head(4)
#               cpi    gdp
# date
# 2023-01-01  100.2  500.0
# 2023-02-01    NaN    NaN
# 2023-03-01    NaN    NaN
# 2023-04-01  100.8  503.0
```

### 6.4 Arguments

<!-- doc-args: tsforecast.utils.frequency.FrequencyConverter.convert_frequency -->
| Argument | Type | Description |
|----------|------|-------------|
| `data` | `pd.Series` ou `pd.DataFrame` | Série, jeu de données ou panel (`MultiIndex`, dates au dernier niveau), convertis entité par entité. |
| `target_freq` | `str` ou `dict` | Fréquence cible commune, ou `{colonne: fréquence}` ; sur un panel, aussi `{(entité...,): fréquence}` et `{(entité..., colonne): fréquence}`. |
| `method` | `str` | Méthode d'agrégation (`'mean'`, défaut, `'sum'`, `'first'`, `'last'`, `'min'`, `'max'`, `'median'`, `'std'`, `'count'`, `'all'`, `'any'`) ou d'interpolation (`'linear'`, `'time'`, `'index'`, `'values'`, `'nearest'`, `'zero'`, `'slinear'`, `'quadratic'`, `'cubic'`), selon la direction. |
| `alignment_method` | `'ffill'`, `'bfill'`, `'nearest'`, `'none'` | Comblement des `NaN` d'une colonne sur la réunion de grilles différentes (défaut `'ffill'`). |
| `time_col` | `str` ou `None` | Colonne des dates d'un `DataFrame` plat, passée en index. |
| `panel_cols` | `List[str]` ou `None` | Colonnes des entités d'un `DataFrame` plat, passées en index. |
| `target_position` | `'S'`, `'E'`, `'start'`, `'end'` ou `None` | Position des étiquettes cibles. Si `None` : celle de `target_freq`, sinon celle de la source, sinon la fin. |
| `full_periods_only` | `bool` | Agrégation : `NaN` pour une période incomplète (défaut `False`). |
| `limit` | `int`, `'default'` ou `None` | Interpolation : nombre maximal de valeurs consécutives comblées ; `'default'` (défaut) : nombre de sous-périodes d'une période ; `None` : sans borne. |
| `limit_direction` | `'forward'`, `'backward'`, `'both'` ou `None` | Interpolation : sens du comblement ; `None` (défaut) : `'forward'` pour une position de début, `'backward'` pour une position de fin. |
| `limit_area` | `'inside'`, `'outside'` ou `None` | Interpolation : comblement limité à l'intérieur ou à l'extérieur des observations (défaut `None`). |
<!-- /doc-args -->

La sortie a les dates de la grille cible, dans l'unité des dates d'entrée. Référence complète :
[API `frequency`](../api/utils/frequency.md), [API `aligner`](../api/utils/aligner.md).

## 7. Convertir une durée : `convert_duration`

`convert_duration(value, from_duration, to_duration, rounding=None)` convertit une quantité de
temps. Les unités s'écrivent en code ou en nom au singulier (`'day'`, `'hour'`, `'business_day'`) ; un
multiplicateur met l'unité à l'échelle ; `rounding='floor'` ou `'ceil'` rend un entier.

```python
from tsforecast.utils.duration import convert_duration

convert_duration(45, "day", "second")                  # 3888000.0
convert_duration(1, "2D", "h")                         # 48.0
convert_duration(37, "h", "D", rounding="ceil")        # 2
convert_duration(1, "Q", "D"), convert_duration(1, "Q", "h")   # (91.0, 2184.0)
convert_duration(1, "Y", "W"), convert_duration(1, "Y", "D")   # (52.0, 365.0)
```

Les unités calendaires sont conventionnelles : un trimestre dure 91 jours (13 semaines) de
24 heures. Entre une unité calendaire et une subdivision usuelle, la conversion compte les
sous-périodes : une année fait 52 semaines, bien qu'elle dure 365 jours (voir le
[concept](../concepts/temporal_utils.md#durees)).

## 8. Valider un jeu de données : `validate_temporal_data`

### 8.1 Valider puis restaurer

Un panel plat, non trié, aux dates en chaînes, devient un panel indexé par `(country, date)` et
trié. Avec `return_metadata=True`, `restore_original_structure` remet ensuite les colonnes d'index à
leur place ; l'ordre trié des lignes est conservé.

```python
from tsforecast.utils.validation import (
    restore_original_structure, validate_entities_grouped, validate_sorted_within_groups,
    validate_temporal_data,
)

raw = pd.DataFrame({
    "country": ["FR", "FR", "DE", "FR", "DE", "DE"],
    "date": ["2024-03-01", "2024-01-01", "2024-01-01", "2024-02-01", "2024-03-01", "2024-02-01"],
    "gdp": [3.0, 1.0, 10.0, 2.0, 30.0, 20.0],
})
validate_entities_grouped(raw, panel_cols=["country"])           # False
validated, metadata = validate_temporal_data(
    raw, time_col="date", panel_cols=["country"], return_metadata=True
)
validated.head(3)
#                      gdp
# country date
# DE      2024-01-01  10.0
#         2024-02-01  20.0
#         2024-03-01  30.0
validate_sorted_within_groups(validated)                         # True
restore_original_structure(validated, metadata).head(3)
#   country       date   gdp
# 0      DE 2024-01-01  10.0
# 1      DE 2024-02-01  20.0
# 2      DE 2024-03-01  30.0
```

### 8.2 Arguments

<!-- doc-args: tsforecast.utils.validation.validate_temporal_data -->
| Argument | Type | Description |
|----------|------|-------------|
| `data` | `pd.Series` ou `pd.DataFrame` | Série ou jeu de données indexé par des dates, panel indexé (`MultiIndex`, dates au dernier niveau), ou `DataFrame` plat. |
| `time_col` | `str` ou `None` | Colonne des dates d'un `DataFrame` plat, passée en index. Si `None`, l'index porte les dates. |
| `panel_cols` | `List[str]` ou `None` | Colonnes des entités d'un `DataFrame` plat ; exige `time_col`. |
| `strict` | `bool` | Si `True` (défaut), une étiquette qui n'est pas une date ou une date répétée lève une `ValueError` ; si `False`, un doublon garde sa première occurrence et un index non convertible est rendu tel quel, avec un avertissement. |
| `sort_data` | `bool` | Trie les lignes par index (défaut `True`). |
| `return_metadata` | `bool` | Si `True`, renvoie `(données, métadonnées)` pour `restore_original_structure` (défaut `False`). |
<!-- /doc-args -->

Les étiquettes temporelles acceptées sont des dates, des chaînes que `pd.to_datetime` lit et des
`Period` ; des nombres (années entières, `RangeIndex`) sont refusés. Référence complète :
[API `validation`](../api/utils/validation.md).

## Voir aussi

- [Concept : utilitaires temporels](../concepts/temporal_utils.md)
- Notebook : `notebooks/0 - Utils.ipynb`
- API : [frequency](../api/utils/frequency.md), [aligner](../api/utils/aligner.md),
  [regularizer](../api/utils/regularizer.md), [duration](../api/utils/duration.md),
  [position](../api/utils/position.md), [time](../api/utils/time.md),
  [validation](../api/utils/validation.md), [parse](../api/utils/parse.md)
