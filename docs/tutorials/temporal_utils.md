# Tutoriel : utilitaires temporels

Tour d'horizon pratique de `tsforecast.utils`. Voir le
[concept](../concepts/temporal_utils.md) pour la vue d'ensemble.

## Fréquences

### Normaliser

```python
from tsforecast.utils.frequency import normalize_frequency, to_code, is_higher_frequency

normalize_frequency("monthly")        # -> 'M'
to_code("quarterly")                  # -> 'Q'
is_higher_frequency("D", "M")         # -> True  (le jour est plus fin que le mois)
```

### Détecter

```python
import numpy as np, pandas as pd
from tsforecast.utils.frequency import detect_index_frequency, detect_frequency

idx = pd.date_range("2020-01-31", periods=24, freq="ME")
detect_index_frequency(idx)           # -> 'M'

df = pd.DataFrame({"q": [1, np.nan, np.nan, 2, np.nan, np.nan]}, index=pd.date_range("2020-01-31", periods=6, freq="ME"))
detect_frequency(df)                  # détection modale par variable -> {'q': 'Q'}
```

### Convertir

```python
from tsforecast.utils.frequency import FrequencyConverter

conv = FrequencyConverter()
# agrégation mensuelle -> trimestrielle (somme), interpolation trimestrielle -> mensuelle
```

## Durées

```python
from tsforecast.utils.duration import normalize_duration, convert_duration

normalize_duration("day")             # -> 'D'
convert_duration(45, "D", "s")        # 45 jours en secondes -> 3888000.0
```

## Position dans la période

```python
from tsforecast.utils.position import normalize_position, flip_position

normalize_position("start")           # -> 'S'
flip_position("S")                    # -> 'E'
```

## Dates et bornes de périodes

```python
from tsforecast.utils.time import resolve_date, get_period_boundaries

resolve_date("2023-03")               # -> Timestamp('2023-03-01')
get_period_boundaries(pd.Timestamp("2023-02-15"), "Q")   # (2023-01-01, 2023-04-01)
```

## Validation de structure

```python
from tsforecast.utils.validation import validate_temporal_data

series = pd.Series([1, 2, 3], index=pd.date_range("2023-01-01", periods=3))
validated = validate_temporal_data(series)   # objet validé, index temporel propre
```

## Parsing de chaînes de fréquence

```python
from tsforecast.utils.parse import parse_frequency, build_frequency_string

parse_frequency("QE-DEC")             # -> ('Q', 'E', 'DEC')
build_frequency_string("Q", "E", "DEC")   # -> 'QE-DEC'
```

## Voir aussi

- [Concept : utilitaires temporels](../concepts/temporal_utils.md)
- API : [frequency](../api/utils/frequency.md), [duration](../api/utils/duration.md),
  [position](../api/utils/position.md), [time](../api/utils/time.md),
  [validation](../api/utils/validation.md), [parse](../api/utils/parse.md)
