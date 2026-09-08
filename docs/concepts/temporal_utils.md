# Utilitaires temporels

`tsforecast.utils` regroupe les briques transverses de manipulation du temps
utilisées par tout le package. Chaque sous-paquet suit le même patron : un
**Normalizer** (représentation canonique), un **Converter** (passage d'une unité
à une autre) et des **fonctions utilitaires** de convenance.

## `utils/frequency/` — fréquences

Normalise les représentations de fréquence (codes pandas `MS`/`QE`, `DateOffset`,
noms conviviaux) et convertit entre elles.

- `FrequencyNormalizer`, `normalize_frequency()`, `to_pandas_freq()`,
  `to_dateoffset()`, `to_code()`, `to_literal()`
- `FrequencyConverter`, `convert_frequency()` — agrégation vers le bas,
  interpolation vers le haut, avec choix de la méthode et de l'ancrage
- `is_higher_frequency()`, `get_frequency_order()`, `validate_frequency()`
- **Détection** : `FrequencyDetector`, `detect_frequency()`,
  `detect_index_frequency()`, `detect_dataset_frequency()` — inférence modale, à
  partir des valeurs observées d'une variable (pas seulement de l'index)

## `utils/duration/` — durées

Normalise et convertit des durées (`'day'` ↔ `'D'`, jours ↔ secondes…), avec
arrondi optionnel.

- `DurationNormalizer`, `DurationConverter`
- `normalize_duration()`, `convert_duration()`,
  `get_duration_conversion_factor()`, `get_duration_order()`

## `utils/position/` — position dans la période

Une observation datée du 1ᵉʳ janvier peut désigner le **début** ou la **fin** de
sa période. Ce sous-paquet normalise cette position et la convertit.

- `PeriodPositionNormalizer`, `PeriodPositionConverter`
- `normalize_position()`, `flip_position()`, `convert_position()`,
  `convert_offset()`

## `utils/time/` — dates et bornes de périodes

- `resolve_date()` — conversion souple chaîne / datetime → date
- `timeseries_to_string()` / `string_to_timeseries()` — index datetime ↔ chaîne
- `get_period_start()` / `get_period_end()` / `get_period_boundaries()`

## `utils/validation/` — validation de structure

- `validate_temporal_data()` — validation complète série ou panel (index
  datetime, unicité, tri)
- `restore_original_structure()` — restauration de la structure d'entrée en
  sortie de transformateur
- `validate_entities_grouped()`, `validate_sorted_within_groups()`

## `utils/parse/` — parsing des chaînes de fréquence

- `parse_frequency()` — `'QE-DEC'` → `('Q', 'E', 'DEC')`
- `build_frequency_string()` — l'opération inverse

Ce sont les deux seules primitives de manipulation textuelle des fréquences ;
tout le reste passe par le Normalizer.

## `utils/abc/` — classes abstraites

`TemporalNormalizer` et `TemporalConverter` : le contrat commun à tous les
Normalizer / Converter ci-dessus.

## Pour aller plus loin

- Tutoriel : [Utilitaires temporels](../tutorials/temporal_utils.md)
- API : [frequency](../api/utils/frequency.md), [duration](../api/utils/duration.md),
  [position](../api/utils/position.md), [time](../api/utils/time.md),
  [validation](../api/utils/validation.md), [parse](../api/utils/parse.md)
