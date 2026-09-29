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

Conventions de conversion :

- seules les fréquences ayant une variante début / fin en pandas portent une
  position : mensuelle (`MS` / `ME`), trimestrielle (`QS` / `QE`), annuelle
  (`YS` / `YE`) et semi-mensuelle (`SMS` / `SME`), avec multiplicateur et
  ancre éventuels (`2MS`, `QS-FEB`, `YE-JUN`). Les autres (`D`, `B`, `W-SUN`,
  `h`...) sont laissées inchangées, par `convert_offset()` comme par
  `convert_position()` ;
- une date convertie tombe sur la grille native pandas de l'offset cible
  (`MS` converti en fin donne exactement `pd.date_range(freq='ME')`, à minuit),
  dans la période de sa date source ; l'ancre décrit les mêmes périodes
  (`QS-FEB` ↔ `QE-JAN`) ;
- une conversion ne fusionne jamais deux dates : une `freq` plus grossière que
  les données lève une `ValueError` (agréger avec `groupby` / `resample`).

## `utils/time/` — dates et bornes de périodes

- `resolve_date()` — conversion souple chaîne / datetime → date
- `timeseries_to_string()` / `string_to_timeseries()` — index datetime ↔ chaîne
- `get_period_start()` / `get_period_end()` / `get_period_boundaries()` — bornes
  `[début, fin)` de la période contenant une date, en `pd.Timestamp` :
  - ancres respectées : `W-WED` (semaine jeudi → mercredi), `QE-JAN` / `QS-FEB`
    (trimestres févr.-avr., mai-juil., …), `YS-JUL` (exercice juillet-juin) ;
  - multiplicateurs (`2MS`, `3D`, `2h`) : périodes de *n* unités alignées sur
    l'époque Unix, ou sur le paramètre optionnel `origin` ;
  - fuseau horaire conservé (jours de 23 h / 25 h aux changements d'heure) ;
    entrées `datetime`, `Timestamp`, `datetime64` ou `Period` (représentée par
    son premier instant).

## `utils/validation/` — validation de structure

- `validate_temporal_data()` — validation complète série ou panel (index
  datetime, unicité, tri)
- `restore_original_structure()` — restauration de la structure d'entrée en
  sortie de transformateur
- `validate_entities_grouped()`, `validate_sorted_within_groups()`

## `utils/parse/` — parsing des chaînes de fréquence

- `parse_frequency()` — `'QE-DEC'` → `ParsedFrequency('Q', 'E', 'DEC', 1)` ;
  le multiplicateur en tête fait partie de la grammaire (`'2MS'` →
  `ParsedFrequency('M', 'S', None, 2)`)
- `build_frequency_string()` — l'opération inverse (paramètre `multiplier`)

Ce sont les deux seules primitives de manipulation textuelle des fréquences ;
tout le reste passe par le Normalizer.

### Multiplicateurs (`'2MS'`, `'3QS-FEB'`)

Les Normalizer de fréquence et de durée acceptent un multiplicateur en tête :

- `normalize()` renvoie le code de base (`'2MS'` → `'M'`), comme pour la
  position et l'ancre ; `normalize_with_multiplier()` renvoie `('M', 2)`.
- `normalize_frequency()` : `'base'` (`'M'`) et `'with_position'` (`'MS'`)
  ignorent le multiplicateur ; `'full'` renvoie la chaîne d'origine ;
  `'components'` renvoie un `ParsedFrequency` dont le 4ᵉ champ est le
  multiplicateur. Les fonctions de détection (`detect_*_frequency`) suivent.
- `is_higher_frequency()` / `is_longer_duration()` tiennent compte du
  multiplicateur : `'MS'` est plus fréquent que `'2MS'`, et `'2MS'` que `'QS'`.
- `DurationConverter` met à l'échelle l'unité préfixée : `'2D'` vaut 48 h.
- `FrequencyConverter` décompose `target_freq` via `ParsedFrequency` et
  convertit vers / depuis une fréquence multipliée. Les opérations qui
  comptent des périodes de base (`full_periods_only`, `method='all'`,
  `anchor_fraction` avec une source multipliée) lèvent `NotImplementedError`.
- Les transformateurs de `tsforecast.delays` rejettent un index multiplié.

Pour les fréquences inférées par pandas, `canonicalize_frequency()` (dans
`utils/frequency/`) ramène les écritures équivalentes à un représentant
unique : `'QS-OCT'` → `'QS-JAN'`, `'QE-NOV'` → `'QE-FEB'`.

## `utils/abc/` — classes abstraites

`TemporalNormalizer` et `TemporalConverter` : le contrat commun à tous les
Normalizer / Converter ci-dessus.

## Pour aller plus loin

- Tutoriel : [Utilitaires temporels](../tutorials/temporal_utils.md)
- API : [frequency](../api/utils/frequency.md), [duration](../api/utils/duration.md),
  [position](../api/utils/position.md), [time](../api/utils/time.md),
  [validation](../api/utils/validation.md), [parse](../api/utils/parse.md)
