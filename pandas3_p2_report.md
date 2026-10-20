# Rapport P2 — Unité des dates en sortie : politique « unité de l'entrée », socle `utils`

pandas 3.0.6, scikit-learn 1.9.1 — 2026-10-20. Anomalie traitée : ANO-UTILS-072 (corrigée).

## 1. Décision appliquée

Option **(2) — unité de l'entrée** (arbitrage de l'auteur). Règle, inscrite dans `CLAUDE.md` (section
« pandas ≥ 3.0 ») et dans ANO-UTILS-072 :

1. **Principe** — toute date rendue par un composant public (index, niveau temporel d'un panel, colonne
   `datetime64`, `Timestamp`) est dans l'unité de l'entrée de référence. Instants, valeurs et calculs internes
   (nanosecondes entières d'`anchor_fraction`, bornes de période) ne changent pas : seule l'unité rendue change.
2. **Entrée de référence** — l'axe temporel de la donnée (index, dernier niveau d'un panel, colonne `time_col`) ;
   pour une date scalaire, l'unité de `pd.Timestamp(date)` (`us` pour un `datetime`, un `Period`, une chaîne ;
   `np.datetime64[D]` → `s`) ; deux jeux comparés : `new_data` ; `X` et `y` : `X`. Un paramètre date scalaire
   (`download_date`, `prediction_date`, `origin`, `test_indices`) ne fixe jamais l'unité.
3. **Date non représentable** — jamais d'arrondi : la plus grossière des unités plus fines qui la représente
   exactement (un index entier ; chaque borne d'une période séparément).
4. **Hors plage** — `OutOfBoundsDatetime` (seul cas : entrée en `ns` et date produite hors 1677-09-21 –
   2262-04-11).

## 2. Socle

- **Utilitaire unique** : `tsforecast/utils/time/resolution.py::as_exact_unit(dates, unit)`
  (`pd.Timestamp` / `pd.DatetimeIndex`), réexporté par `tsforecast.utils.time`. Constante transverse
  `DATETIME_UNITS = ('s', 'ms', 'us', 'ns')` dans `utils/_constants.py`. Module séparé, importé **dans la méthode**
  par les deux convertisseurs : `utils/time/utils.py` importe `utils.frequency`, dont l'`__init__` importe
  `converter.py` (un import en tête créerait un cycle).
- **Chemins modifiés** (sortie seule) :

| Fichier | Point d'application | Couvre |
|---|---|---|
| `utils/frequency/converter.py` | fin de `interpolate_to_higher_frequency` | `convert_frequency` (série, DataFrame, panel), `convert()`, `FrequencyAligner` |
| `utils/position/converter.py` | `_convert_datetime_index`, après les retours anticipés | `PeriodPositionConverter.convert`, `convert_position` |
| `utils/time/utils.py` | `get_period_start` / `get_period_end` / `get_period_boundaries` (helper `_at_date_unit`, chaque borne) | `delays` (bornes de période), `imputation_window` |

  `aggregate_to_lower_frequency` et les autres chemins de `convert_frequency` rendaient déjà l'unité de l'entrée
  (règle documentée et testée, aucun appel). Docstrings `Returns:` / `Raises:` complétées ; aucun doctest touché.

## 3. Unités rendues avant / après

Entrée en `s` / `ms` / `us` / `ns` ; « entrée » = `s ms us ns`.

| Composant | Chemin | Avant | Après |
|---|---|---|---|
| `aggregate_to_lower_frequency` | tous | entrée | entrée |
| `interpolate_to_higher_frequency` | grille étendue, `anchor_fraction`, cibles `W` / `SMS` / `h`, DataFrame | `us us us us` | entrée |
| idem | grille infra-seconde (`'100ms'`) | `us us us us` | `ms ms us ns` |
| idem | repli `asfreq`, jamais observée | entrée | entrée |
| `convert_frequency` | série : montée | `us us us us` | entrée |
| idem | DataFrame / panel : montée | `us us us ns` | entrée |
| idem | descentes, identité, colonnes conservées, `time_col` | entrée | entrée |
| `convert_position` | M / Q / Y, ancres, fuseau, panel | `us us us ns` | entrée |
| idem | `SMS`, fréquence sans position | entrée | entrée |
| `get_period_*` | calendaire | `us us us us` | entrée |
| idem | infra-journalier | `ns ns ns ns` | entrée |
| idem | borne plus fine que la date (`'ms'`, `'ns'`) | `ns` | début : entrée ; fin : la plus fine nécessaire |
| idem | `datetime`, `Period` | `us` / `ns` | `us` |
| **Aval (sans code modifié)** `FrequencyAligner` (index étendu) | | `us us us ns` | entrée |
| `HighFrequencyImputer` (sortie, provenance, rejeu, inversion, `fit_transform(X, y)`) | TS, PANEL-F | `us` (provenance d'un panel : entrée) | entrée |
| `ImputationWindowCalculator` (masques bruts) | | `us us us us` | entrée |
| `compare_and_detect_delays` : `period_start`, `period_end` | | `us us us us` | entrée |
| idem : `observation_date`, `download_date`, résultat vide | | entrée / `us` / `ns` | inchangés (P3) |

## 4. Garde-fous

- **Sorties de référence** (script jetable du scratchpad, copie de HEAD contre arbre modifié) : 1 228 cas —
  socle et aval (aligneur, imputeur TS / PANEL-F × `interpolate` / `model`, fenêtres, délais,
  `calculate_applicable_delay`, transformateurs de délais, `PanelwiseTransformer`) sur l'identité et les
  perturbations de `tests/support/perturbations.py`, × 4 unités ; 1 043 sorties comparées. **0 écart** d'instant
  (entiers de ns) ou de valeur (bits float64) ; **0 écart à la règle** (unité attendue recalculée à partir des
  instants) ; une seule différence d'exception, voulue (règle 4) : `get_period_end(Timestamp('2262-04-01',
  unit='ns'), 'M')` lève `OutOfBoundsDatetime` (rendait un `Timestamp[us]`).
- **Suite complète** (`-W error::FutureWarning -W error::DeprecationWarning`) : 13 969 réussis, 0 échec
  (13 802 avant).
- **Couverture** (lignes et branches) : aucun module en baisse ; total 99,368 % → 99,370 % ;
  `time/resolution.py` 100 %, `frequency/converter.py` 100 %, `position/converter.py` 98,18 → 98,20 %,
  `time/utils.py` 97,51 → 97,56 %.
- **Doctests** : même ensemble de 23 échecs (N7 / N8) ; 310 réussis (308 + `as_exact_unit`, `_at_date_unit`).
- **Conventions** (`check_conventions.py --rule all`) : 23 violations bloquantes (doctests) avant comme après,
  aucune nouvelle (décalages de lignes seulement).
- Imports à froid de `tsforecast`, `utils.frequency`, `utils.position`, `utils.time` et de leurs modules : sans cycle.

## 5. Tests

- Nouveau `tests/unit/utils/time/test_resolution.py` (53 tests).
- `tests/unit/utils/frequency/converter/test_datetime_resolution.py` : table `PATHS` (16 chemins, un par ligne,
  `datetime_units == {unit}`), méthodes d'agrégation et d'interpolation, grille infra-seconde, aller-retour
  QS → MS → QS (index d'entrée, unité comprise).
- `tests/unit/utils/time/test_utils.py::TestDatetimeResolution` : bornes à l'unité de la date (12 fréquences ×
  2 fuseaux), table des autres chemins (borne plus fine, `datetime`, `Period`, `np.datetime64`, `origin`), refus hors
  plage.
- `tests/unit/utils/position/test_converter.py::TestDatetimeResolution` : 8 chemins, aller-retour S → E → S.
- Ricochets ajustés : `tests/unit/delays/data_manager/test_datetime_resolution.py::test_single_download`
  (observation et bornes à l'unité de l'entrée, `download_date` épinglé à `us`) ;
  `tests/unit/frequency/high_frequency_imputer/test_datetime_resolution.py` (sortie, provenance, rejeu,
  inversion à l'unité de l'entrée). Aucune valeur d'or modifiée.

## 6. Points pour P3

- `compare_and_detect_delays` :
  - `download_date` → unité de `new_data` (`as_exact_unit(pd.Timestamp(download_date), unité)`, plus fine si la date
    l'exige : `'… 13:45:10.123456'` sur des données en `s` → `us`) ;
  - `observation_date` suit l'unité d'`existing_data` quand il est fourni (`new_data` en `s`, `existing_data` en
    `ns` → `ns`) → unité de `new_data` ;
  - résultat vide `datetime64[ns]` → unité de `new_data` (dtype des chaînes : P4) ; chemin `time_col`.
- `HighFrequencyImputer` : ses sorties publiques suivent déjà l'entrée ; reste à auditer l'état ajusté (plan,
  grilles, instantanés d'`inverse_transform`) de `fit` à `transform`, à tester `fit_transform(X, y)` et sortie =
  provenance, et `XYPipeline(memory=…)` (deux unités d'entrée donnent désormais deux sorties de dtypes différents :
  empreintes différentes, à documenter), puis clore ANO-FREQ-086.
- Épingler si souhaité l'unité de `ImputationWindowCalculator` (masques bruts) et de l'index étendu de
  `FrequencyAligner`, qui suivent l'entrée depuis P2.
- ANO-DELAYS-045 et ANO-FREQ-086 portent une ligne « Effet du socle » décrivant l'état après P2 (statuts : ouverts).
