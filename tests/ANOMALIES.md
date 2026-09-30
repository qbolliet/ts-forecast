# Registre des anomalies

Ce registre applique le protocole de `tests_and_refactoring_prompts.md` §4.4 et §4.5.

## §4.4 — Tri des tests existants en échec

Pour chaque test migré en échec (et chaque test existant douteux, même vert), la cause est
établie **avant** de toucher au test, puis classée :

| Catégorie | Critère | Action |
|---|---|---|
| **(a) Test obsolète** | changement **délibéré** de l'API : prouvé par l'historique git (`git log -S<symbole>`), la spec ou la mémoire de l'auteur | réécrire le test sur l'API actuelle, en conservant l'**intention** du test ; supprimer s'il n'a plus d'objet |
| **(b) Code faux** | le test exprime ce que la classe **doit** faire (spec, finalité, cohérence avec les autres composants) et le code s'en écarte | garder le test (le rendre correct si besoin), `xfail(strict=True)`, entrée ci-dessous |
| **(c) Test faux** | le test encode lui-même une attente erronée (valeur d'or fausse, hypothèse contredite par la spec) | corriger le test, justifier dans le rapport |
| **(d) Indécidable** | aucune source ne tranche | épingler le comportement actuel, entrée `à arbitrer` ci-dessous |

**Règle absolue** : ne jamais « faire passer » un test en l'alignant sur un comportement qui
semble contraire à la finalité de la classe. Un test vert qui fige un bogue est pire qu'un test
absent. En cas de doute entre (a) et (b), l'historique git tranche ; à défaut, (d).

## §4.5 — Protocole anomalie

1. Test du comportement **attendu**, marqué
   `@pytest.mark.xfail(strict=True, reason="ANO-<MOD>-NNN: <résumé>")`, avec `<MOD>` ∈
   {`UTILS`, `DELAYS`, `FREQ`} ;
2. entrée ci-dessous, dans la section du module :

   ```markdown
   ### ANO-FREQ-NNN — <titre court>
   - **Type** : [CODE] comportement | [DOC] docstring ≠ code | [SPEC] code ≠ spec
   - **Composant** : `tsforecast/<paquet>/<module>.py::<Symbole>`
   - **Sévérité** : bloquante | majeure | mineure | cosmétique | à arbitrer
   - **Observé** : <comportement, message d'erreur exact>
   - **Attendu** : <comportement et justification : spec, finalité, cohérence>
   - **Reproduction** : <5-10 lignes minimales>
   - **Test** : `tests/unit/<…>.py::<Classe>::<test>`
   - **Statut** : ouvert
   ```

3. une anomalie **[DOC]** n'a pas de `xfail` (le test suit le code, qui est juste) ;
4. numérotation continue par préfixe (lire le dernier numéro dans ce fichier).

---

## UTILS

### ANO-UTILS-001 — `build_frequency_string` perd l'ancre d'un jour de semaine
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/parse/utils.py::build_frequency_string`
- **Sévérité** : mineure
- **Observé** : `build_frequency_string(*parse_frequency('W-MON'))` renvoyait `'W'`
  (l'ancre `MON` perdue). Le code ignorait délibérément le suffixe quand
  `position is None` (« le suffixe n'a de sens qu'accolé à une position »),
  hypothèse vraie pour les trimestres (`'QE-DEC'`) mais fausse pour les
  ancres hebdomadaires, qui n'ont pas de position S/E et portent pourtant un
  suffixe significatif (le jour de la semaine). Le même chemin de code est
  emprunté par `PublicationDelayTransformer` / `ShiftTransformer` /
  `MaskTransformer` (`tsforecast/delays/transformers.py`, via
  `self.index_position_` / `self.index_suffix_`) : un index à fréquence
  hebdomadaire ancrée y perdait silencieusement son ancre.
- **Attendu** : le round trip `parse_frequency` → `build_frequency_string`
  doit être l'identité pour toute chaîne de fréquence supportée, y compris
  les ancres hebdomadaires (`'W-MON'` → `'W-MON'`), comme c'est déjà le cas
  pour les ancres trimestrielles.
- **Reproduction** :
  ```python
  from tsforecast.utils.parse.utils import parse_frequency, build_frequency_string
  freq, position, suffix = parse_frequency('W-MON')  # ('W', None, 'MON')
  build_frequency_string(freq, position, suffix)  # était 'W' au lieu de 'W-MON'
  ```
- **Correctif** : `build_frequency_string` accole désormais le suffixe
  indépendamment de la présence d'une position (seule la position reste
  conditionnée à `_POSITION_AWARE_FREQUENCIES`).
- **Test** : `tests/unit/utils/parse/test_utils.py::TestParseBuildRoundTrip::test_roundtrip_is_identity[W-MON]`
- **Statut** : corrigée

### ANO-UTILS-002 — Docstring de `DurationNormalizer` : exemple appelant une méthode inexistante
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/utils/duration/normalizer.py::DurationNormalizer`
- **Sévérité** : cosmétique
- **Observé** : le docstring de classe (`Examples:`) appelle
  `normalizer.normalize_duration('monthly')` ; `DurationNormalizer` n'expose
  pas de méthode `normalize_duration` (ce nom est celui de la fonction de
  commodité au niveau module, `tsforecast/utils/duration/utils.py::normalize_duration`,
  qui appelle en interne `DurationNormalizer.normalize`). `uv run pytest
  --doctest-modules tsforecast/utils/duration` lève
  `AttributeError: 'DurationNormalizer' object has no attribute 'normalize_duration'`.
- **Attendu** : l'exemple devrait appeler `normalizer.normalize('monthly')`
  (ou `normalize_duration('monthly')` après import de la fonction module).
- **Reproduction** :
  ```python
  from tsforecast.utils.duration.normalizer import DurationNormalizer
  DurationNormalizer().normalize_duration('monthly')  # AttributeError
  ```
- **Test** : découvert hors campagne de tests dédiés, via
  `uv run pytest --doctest-modules tsforecast/utils/duration` (non ajouté à
  `tests/`, aucun test de doctest n'existe encore pour ce module).
- **Correctif** : l'exemple appelle désormais `normalizer.normalize('month')`
  (littéral valide de `UserDurationType`, `'monthly'` n'en fait pas partie).
  `uv run pytest --doctest-modules tsforecast/utils/duration` passe.
- **Statut** : corrigée

### ANO-UTILS-003 — Docstring de `get_duration_order` : type de retour annoncé (`float`) ≠ type observé (`int`)
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/utils/duration/utils.py::get_duration_order`
- **Sévérité** : cosmétique
- **Observé** : le docstring annonce `Returns: Duration order as float` et
  l'exemple `>>> get_duration_order('day') / 7.0`, mais `_duration_order`
  (`DurationNormalizer.__init__`) stocke des `int` pour la plupart des codes
  (seuls `'B'` et `'SM'` sont des `float`, `7.5`/`8.5`, pour s'insérer entre
  deux durées standards) : `get_duration_order('day')` renvoie l'`int` `7`,
  pas le `float` `7.0`. `uv run pytest --doctest-modules
  tsforecast/utils/duration` échoue sur cet exemple.
- **Attendu** : soit le docstring documente le type réel (`int` ou `float`
  selon le code), soit le code caste systématiquement en `float`.
- **Reproduction** :
  ```python
  from tsforecast.utils.duration.utils import get_duration_order
  type(get_duration_order('day'))  # <class 'int'>, pas <class 'float'>
  ```
- **Test** : couvert côté test par
  `tests/unit/utils/duration/test_utils.py::TestGetDurationOrder::test_golden_orders`
  (assertions d'égalité numérique, insensibles au type exact).
- **Correctif** : docstring corrigé pour documenter le type réel (`int` ou
  `float` selon le code) et l'exemple mis à jour (`get_duration_order('day')
  == 7`, pas `7.0`). Code inchangé (pas de cast systématique en `float`).
- **Statut** : corrigée

### ANO-UTILS-004 — `DurationConverter.convert(rounding=...)` ignorait silencieusement une valeur de `rounding` inconnue
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/duration/converter.py::DurationConverter._round_result`
- **Sévérité** : mineure
- **Observé** : une valeur de `rounding` autre que `'floor'`, `'ceil'` ou
  `None` (ex. `'round'`) ne levait aucune erreur ; `_round_result` retournait
  silencieusement la valeur non arrondie, comme si `rounding=None` avait été
  passé. Point de vigilance déjà relevé dans
  `notebooks/utils/duration_converter.ipynb` §3.4.
- **Attendu** : une valeur de `rounding` non reconnue doit être rejetée
  explicitement (échec rapide plutôt que silencieux).
- **Reproduction** :
  ```python
  from tsforecast.utils.duration.converter import DurationConverter
  DurationConverter().convert(37, 'h', 'D', rounding='round')  # ValueError désormais
  ```
- **Correctif** : `_round_result` lève `ValueError("Unsupported rounding
  mode: ...")` pour toute valeur autre que `'floor'`/`'ceil'`.
- **Test** : `tests/unit/utils/duration/test_converter.py::TestConvert::test_unrecognized_rounding_value_raises`
- **Statut** : corrigée

### ANO-UTILS-005 — `DurationConverter.get_conversion_factor` : branche défensive inatteignable
- **Type** : [CODE] comportement (nettoyage, pas un bogue)
- **Composant** : `tsforecast/utils/duration/converter.py::DurationConverter.get_conversion_factor`
- **Sévérité** : cosmétique
- **Observé** : après recherche dans la table calendaire (`_CALENDAR_SUBPERIODS`),
  le code vérifiait `if from_code not in _CONVERSION_FACTORS_TO_SECONDS`
  (et l'équivalent pour `to_code`) avant le calcul via les secondes. Ces
  branches étaient du code mort : `from_code`/`to_code` proviennent de
  `normalize_duration`, qui ne renvoie que des codes déjà présents dans
  `_CONVERSION_FACTORS_TO_SECONDS` (mêmes clés que `_code_to_literal`) — donc
  inatteignables avec les mappings actuels, décelé lors de la campagne de tests
  U2 (couverture bloquée à 92 % sur ces deux lignes).
- **Attendu** : suppression du code mort, sans changement de comportement.
- **Correctif** : les deux `if` et leurs `raise ValueError` retirés ; le
  commentaire précédant le calcul explique désormais la garantie qui les
  rendait inutiles.
- **Test** : couverture 100 % de `converter.py` après suppression
  (`tests/unit/utils/duration/test_converter.py`).
- **Statut** : corrigée

### ANO-UTILS-006 — Conversion début → fin : dates à 23:59:59.999999999, hors grille `ME` / `QE` / `YE`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/position/converter.py::PeriodPositionConverter._convert_datetime_index`
- **Sévérité** : majeure
- **Observé** : la conversion `'S'` → `'E'` passe par
  `index.to_period(base).to_timestamp(how='end')`, qui renvoie le **dernier
  instant** de la période : `2024-02-01` (`MS`) devient
  `2024-02-29 23:59:59.999999999`, pas `2024-02-29` (minuit, convention de
  `pd.date_range(freq='ME')` et de tous les jeux `ME` du paquet, dont
  `PANEL-X`). Conséquences : un index `MS` converti en fin ne s'aligne pas
  (jointure, `reindex`) sur un index `ME` natif du même calendrier ; l'aller-
  retour fin → début → fin d'un index `ME` n'est pas l'identité ; un index
  journalier ou horaire, sans notion de position (`convert_offset('D', 'E')`
  renvoie `'D'`), est tout de même décalé (`2024-01-01` →
  `2024-01-01 23:59:59.999999999`). L'observation reste dans sa période (même
  jour calendaire) : pas de sortie de période, d'où « majeure » et non
  « bloquante ». Relevé comme « point de vigilance » dans
  `notebooks/utils/period_position_converter.ipynb` §5.1, non corrigé depuis.
- **Attendu** : la date de fin est le **jour** de fin de période à minuit
  (grille pandas `ME` / `QE` / `YE`), et la conversion est l'identité pour les
  fréquences sans position (`D`, `h`), comme `convert_offset`.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.position import convert_position
  idx = pd.date_range('2024-01-01', periods=3, freq='MS')
  convert_position(idx, 'S', 'E')[1]   # Timestamp('2024-02-29 23:59:59.999999999')
  pd.date_range('2024-01-31', periods=3, freq='ME')[1]   # Timestamp('2024-02-29 00:00:00')
  ```
- **Correctif** : `_convert_datetime_index` ne passe plus par
  `to_period(...).to_timestamp(how='end')` mais par l'arithmétique des offsets
  pandas au jour calendaire (fin = début de période + n périodes − 1 jour, à
  minuit) : un index `MS` converti est exactement `pd.date_range(freq='ME')`.
  Décision de l'auteur (2026-09-28) : fréquences sans variante début / fin en
  pandas (`D`, `B`, `W`, `h`, …) laissées **inchangées**, comme par
  `convert_offset`. Effets liés : heure du jour abandonnée (bornes à minuit),
  fuseau horaire conservé (calcul en heure locale naïve, correct au changement
  d'heure ; il était auparavant perdu avec un avertissement pandas).
- **Test** : `tests/unit/utils/position/test_converter.py::TestConvertDatetimeIndex::test_start_to_end_matches_pandas_end_grid`,
  `::test_end_start_end_roundtrip_is_identity`, `::test_frequency_without_position_is_identity`,
  `::test_time_zone_is_kept_across_dst`
- **Statut** : corrigée

### ANO-UTILS-007 — Conversion d'index : l'ancre de la fréquence est ignorée (`QS-FEB`, `QE-NOV`, `YS-JUL`, `W-WED`)
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/position/converter.py::PeriodPositionConverter._convert_datetime_index`
- **Sévérité** : majeure
- **Observé** : la fréquence est réduite à sa base
  (`normalize_frequency(freq, return_format='base')` : `'QS-FEB'` → `'Q'`),
  puis `to_period('Q')` utilise l'ancre par défaut de pandas (`Q-DEC`,
  trimestres civils). Un index `QS-FEB` (trimestres févr.-avr., mai-juil., …)
  converti en fin donne `2024-03-31`, `2024-06-30`, … au lieu de
  `2024-04-30`, `2024-07-31`, … ; `QE-NOV` → début donne le 1er janvier au lieu
  du 1er décembre ; `YS-JUL` (exercice juillet-juin) → fin donne le 31/12 au
  lieu du 30/06 ; `W-WED` → début donne le lundi au lieu du jeudi. Chaque date
  reste dans sa période d'origine, mais pas à sa borne, et l'index produit
  suit un autre découpage (trimestres civils) : l'aller-retour début → fin →
  début sur `QS-FEB` renvoie `2024-02-01` → `2024-03-31` → `2024-01-01`, date
  **hors** du trimestre d'origine (févr.-avr.). D'où « majeure ».
- **Attendu** : la conversion se fait dans les périodes de la fréquence
  ancrée (`to_period('Q-JAN')` pour `QS-FEB`, etc.) ; aller-retour identité.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.position import convert_position
  idx = pd.date_range('2024-02-01', periods=3, freq='QS-FEB')
  end = convert_position(idx, 'S', 'E')        # 2024-03-31, 2024-06-30, 2024-09-30 (au lieu de 04-30, 07-31, 10-31)
  convert_position(end, 'E', 'S')[0]           # Timestamp('2024-01-01') : hors du trimestre févr.-avr.
  ```
- **Correctif** : `PeriodPositionConverter._resolve_period_offsets` décompose
  la fréquence (multiplicateur, base, position, ancre) et construit le couple
  d'offsets début / fin décrivant les mêmes périodes (`QS-FEB` / `QE-JAN`) ;
  la conversion se fait dans ces périodes. Le cas `W-WED` n'a plus d'objet :
  les semaines n'ont pas de position (voir ANO-UTILS-006 et 009), l'index est
  laissé inchangé.
- **Test** : `tests/unit/utils/position/test_converter.py::TestConvertDatetimeIndex::test_anchored_frequency_uses_its_own_periods`,
  `::test_start_end_start_roundtrip_is_identity`, `::test_converted_index_follows_convert_offset`
- **Statut** : corrigée

### ANO-UTILS-008 — `convert_offset` perd le suffixe d'ancrage et redéfinit les périodes
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/position/converter.py::PeriodPositionConverter.convert_offset`
- **Sévérité** : mineure
- **Observé** : `freq, _, _ = parse_frequency(base_offset)` jette le suffixe,
  puis `build_frequency_string(freq, to_pos)` le recompose sans lui.
  `convert_offset('QE-NOV', 'start')` renvoie `'QS'` (= `QS-JAN`, trimestres
  civils) au lieu de `'QS-DEC'` (mêmes trimestres déc.-févr., …) ;
  `convert_offset('YS-JUL', 'end')` renvoie `'YE'` au lieu de `'YE-JUN'` ;
  même sans changement de position, `convert_offset('QS-FEB', 'start')`
  renvoie `'QS'`. Sans conséquence pour les ancres par défaut (`QE-DEC` →
  `'QS'` est équivalent à `QS-JAN`).
- **Attendu** : l'offset renvoyé décrit les **mêmes périodes** que l'offset
  source (`QS-<m>` ↔ `QE-<m-1>`, `YS-<m>` ↔ `YE-<m-1>`), multiplicateur
  conservé.
- **Reproduction** :
  ```python
  from tsforecast.utils.position import convert_offset
  convert_offset('QE-NOV', 'start')   # 'QS' au lieu de 'QS-DEC'
  convert_offset('QS-FEB', 'start')   # 'QS' au lieu de 'QS-FEB'
  ```
- **Correctif** : `convert_offset` conserve le suffixe et, pour une ancre
  mensuelle (trimestriel, annuel), le décale d'un mois selon le sens de la
  conversion (`_shift_anchor` : `QS-<m>` ↔ `QE-<m−1>`) ; une ancre sans
  position (`'Q-DEC'`) est lue comme mois de fin, conformément à pandas ; un
  mois inconnu est rejeté.
- **Test** : `tests/unit/utils/position/test_converter.py::TestConvertOffset::test_anchor_is_shifted_to_describe_the_same_periods`
- **Statut** : corrigée

### ANO-UTILS-009 — `convert_offset` renvoie des alias inconnus de pandas pour `W` et `B`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/position/converter.py::PeriodPositionConverter.convert_offset`
  (cause : `_POSITION_AWARE_FREQUENCIES` de `tsforecast/utils/parse/utils.py`,
  qui contient `'W'` et `'B'`)
- **Sévérité** : mineure
- **Observé** : `build_frequency_string` traite `'W'` et `'B'` comme
  « position-aware » et leur accole `S` / `E` : `convert_offset('W', 'end')`
  renvoie `'WE'`, `convert_offset('W-MON', 'start')` renvoie `'WS'` (ancre
  perdue en plus), `convert_offset('B', 'end')` renvoie `'BE'`. Aucun de ces
  alias n'existe en pandas : `to_offset('WE')` lève
  `ValueError: Invalid frequency: WE`. Le docstring annonce pourtant
  « Converted pandas DateOffset string ». Le notebook
  `period_position_normalizer.ipynb` montre que `WS` / `WE` / `BS` / `BE`
  étaient des codes internes de l'ancien `_legacy_offset_mapping`, supprimé
  lors de la consolidation `parse_frequency`.
- **Attendu** : pour une fréquence sans variante S/E en pandas (`W`,
  `W-<jour>`, `B`), `convert_offset` renvoie un offset pandas valide (au
  minimum l'offset d'origine, comme pour `D` / `h`).
- **Reproduction** :
  ```python
  from pandas.tseries.frequencies import to_offset
  from tsforecast.utils.position import convert_offset
  convert_offset('W', 'end')            # 'WE'
  to_offset(convert_offset('W', 'end'))  # ValueError: Invalid frequency: WE
  ```
- **Correctif** : à la source, `_POSITION_AWARE_FREQUENCIES`
  (`tsforecast/utils/parse/utils.py`) vaut désormais `('M', 'Q', 'Y', 'SM')` :
  `W` et `B` n'y figurent plus (aucune variante S/E en pandas), `SM` y entre
  (`SMS` / `SME`). `build_frequency_string` ignore donc la position pour `W`
  et `B` : `convert_offset('W-MON', 'end')` renvoie `'W-MON'` (décision de
  l'auteur : offset hebdomadaire inchangé). Le test U1
  `test_builds_expected_string[business-daily-end]`, qui épinglait `'BE'`, a
  été réécrit (même intention, comportement corrigé). Effet de bord favorable :
  `FrequencyConverter` construisait lui aussi `'WS'` / `'WE'` pour une cible
  hebdomadaire.
- **Test** : `tests/unit/utils/position/test_converter.py::TestConvertOffset::test_frequency_without_position_is_unchanged`,
  `::test_result_is_a_valid_pandas_offset`,
  `tests/unit/utils/parse/test_utils.py::TestBuiltStringIsValidPandas`
- **Statut** : corrigée

### ANO-UTILS-010 — `freq` explicite plus grossière que les données : observations déplacées hors de leur période, index dupliqué
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/position/converter.py::PeriodPositionConverter.convert`
  (`_convert_time_series`, `_convert_panel`)
- **Sévérité** : majeure
- **Observé** : une `freq` fournie est appliquée telle quelle, sans contrôle de
  cohérence avec les dates. Sur une série mensuelle, `freq='Q'` envoie
  janvier, février et mars 2024 au `2024-03-31` : les observations de janvier
  et de février quittent leur mois (leur période d'origine) et l'index
  devient dupliqué, sans erreur ni avertissement. Sur un panel, la `freq`
  s'applique à **toutes** les entités : une entité mensuelle dans un panel
  trimestriel subit le même sort. Relevé comme « point de vigilance majeur »
  dans `period_position_converter.ipynb` §5.4, non corrigé depuis.
- **Attendu** : une conversion ne fusionne jamais deux dates distinctes d'une
  même entité ; si la `freq` fournie est plus grossière que l'espacement des
  dates (deux dates dans la même période), lever une `ValueError` explicite.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.position import convert_position
  s = pd.Series([1., 2., 3.], index=pd.date_range('2024-01-01', periods=3, freq='MS'))
  convert_position(s, 'S', 'E', freq='Q').index.normalize()
  # DatetimeIndex(['2024-03-31', '2024-03-31', '2024-03-31'])
  ```
- **Correctif** : `_convert_datetime_index` vérifie que le nombre de dates
  distinctes est conservé (par entité pour un panel) et lève sinon
  `ValueError: Frequency '<freq>' is coarser than the data: …` en orientant
  vers `groupby` / `resample`. Décision de l'auteur (2026-09-28) : erreur
  plutôt qu'avertissement. Limite connue : une `freq` multipliée sur des
  données plus fines (`'2MS'` sur du mensuel) ne fusionne aucune date (blocs
  chevauchants) et n'est pas détectable sans connaître la vraie fréquence.
- **Test** : `tests/unit/utils/position/test_converter.py::TestConvertTimeSeries::test_coarser_explicit_frequency_is_rejected`,
  `::TestConvertDatetimeIndex::test_coarser_frequency_is_rejected`,
  `::TestConvertPanel::test_explicit_frequency_incompatible_with_an_entity_is_rejected`
- **Statut** : corrigée

### ANO-UTILS-011 — Panel : le repli `detect_dataset_frequency` renvoie un `dict`, message d'erreur dédié inatteignable
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/position/converter.py::PeriodPositionConverter._convert_panel`
- **Sévérité** : mineure
- **Observé** : quand `detect_index_frequency` renvoie `None` pour une entité
  (espacement irrégulier sans fréquence reconnue, ex. 45 puis 50 jours), le
  repli appelle `detect_dataset_frequency(df=group_data,
  consistency_mode='highest', strict=False)` **sans** `check_consistency=True` :
  la fonction renvoie alors la carte `{(entité, colonne): fréquence}` et non
  une fréquence. Ce `dict` (non vide, donc jamais `None`) est passé à
  `normalize_frequency`, qui lève `ValueError: Frequency must be a string, got
  <class 'dict'>` : le message dédié « Cannot infer frequency for entity … »
  n'est jamais atteint, et le repli ne peut jamais réussir.
- **Attendu** : soit le repli renvoie une fréquence unique
  (`check_consistency=True`), soit l'erreur dédiée nommant l'entité est levée.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.position import convert_position
  dates = list(pd.date_range('2024-01-01', periods=3, freq='MS')) + \
      list(pd.to_datetime(['2023-01-01', '2023-02-15', '2023-04-06']))
  idx = pd.MultiIndex.from_arrays([['A'] * 3 + ['B'] * 3, dates])
  convert_position(pd.DataFrame({'v': range(6)}, index=idx), 'S', 'E')
  # ValueError: Frequency must be a string, got <class 'dict'>
  ```
- **Correctif** : le repli appelle `detect_dataset_frequency(...,
  return_format='full', check_consistency=True, consistency_mode='highest')`
  sur le groupe réduit à son index temporel (le `MultiIndex` à niveaux sans
  nom faisait échouer la détection de structure panel) et obtient une
  fréquence unique ; une `ValueError` de la détection sur l'index (entité à
  une seule date) mène aussi au repli, puis au message dédié nommant l'entité.
- **Test** : `tests/unit/utils/position/test_converter.py::TestConvertPanel::test_undetectable_entity_frequency_raises_dedicated_error`,
  `::test_column_frequency_fallback`, `::test_single_observation_entity_without_frequency_raises`,
  `::TestDeferredFrequencyImports::test_panel_falls_back_on_dataset_detection`
- **Statut** : corrigée

### ANO-UTILS-012 — `DataFrame` converti : dtypes des colonnes perdus
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/position/converter.py::PeriodPositionConverter._convert_time_series`
  et `::_convert_panel`
- **Sévérité** : mineure
- **Observé** : le résultat est reconstruit par
  `pd.DataFrame(data.values, index=new_index, columns=data.columns)`.
  `.values` produit un tableau NumPy **commun** à toutes les colonnes : un
  `DataFrame` `int64` / `float64` / `bool` / `object` ressort entièrement en
  `object`, un `DataFrame` `int64` / `float64` entièrement en `float64`. Même
  reconstruction pour chaque segment d'un panel. Les attributs (`attrs`) sont
  aussi perdus. Changer la position ne devrait toucher que l'index.
- **Attendu** : valeurs et dtypes inchangés, seul l'index est remplacé (ex.
  `data.set_axis(new_index)` ou copie + affectation de l'index).
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.position import convert_position
  df = pd.DataFrame({'a': [1, 2], 'b': [0.5, 1.5], 'c': ['u', 'v']},
                    index=pd.date_range('2024-01-01', periods=2, freq='MS'))
  convert_position(df, 'S', 'E', freq='M').dtypes   # a, b, c : object
  ```
- **Correctif** : copie de l'objet puis remplacement de son index
  (`result = data.copy(); result.index = …`), pour les séries comme pour les
  panels : dtypes, `attrs` et nom des colonnes conservés.
- **Test** : `tests/unit/utils/position/test_converter.py::TestConvertTimeSeries::test_dataframe_keeps_mixed_dtypes`,
  `::TestConvertPanel::test_dataframe_panel_keeps_mixed_dtypes`
- **Statut** : corrigée

### ANO-UTILS-013 — Panel vide : `ValueError('No objects to concatenate')`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/position/converter.py::PeriodPositionConverter._convert_panel`
- **Sévérité** : mineure
- **Observé** : un panel sans ligne (colonnes et noms d'index conservés), même
  avec `freq` fournie, lève l'erreur interne de pandas
  `ValueError: No objects to concatenate` : aucun groupe, donc
  `pd.concat([])`. Un `DatetimeIndex` ou une série simple vides sont, eux,
  convertis sans erreur.
- **Attendu** : un panel vide ressort vide, avec la même structure (cas limite
  « jeu vide » de `CLAUDE.md`).
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.position import convert_position
  idx = pd.MultiIndex.from_arrays([['A'], pd.to_datetime(['2024-01-01'])], names=['entity', 'date'])
  convert_position(pd.Series([1.0], index=idx).iloc[0:0], 'S', 'E', freq='M')
  # ValueError: No objects to concatenate
  ```
- **Correctif** : corrigée par effet du correctif d'ANO-UTILS-012 (non demandé
  explicitement) : `_convert_panel` ne concatène plus de segments mais
  replace les dates converties à la position de leurs lignes ; sans groupe,
  le niveau temporel vide est conservé.
- **Test** : `tests/unit/utils/position/test_converter.py::TestConvertPanel::test_empty_panel_with_explicit_frequency`
- **Statut** : corrigée

### ANO-UTILS-014 — Index court à doublons détecté `'ns'` : conversion silencieusement sans effet
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/detector.py::FrequencyDetector._extend_infer_freq`
  (symptôme dans `PeriodPositionConverter.convert` sans `freq`)
- **Sévérité** : mineure
- **Observé** : sur `['2024-01-01', '2024-01-01', '2024-02-01', '2024-03-01']`,
  `pd.infer_freq` renvoie `None` (doublon) ; le repli calcule les écarts triés
  0, 31 et 29 jours, tous modaux ; `mode()[0]` retient le plus petit, **0**,
  classé comme infra-journalier → `'ns'`. La conversion début → fin à la
  nanoseconde laisse alors les dates au 1er du mois, sans erreur ni
  avertissement. Dès que les doublons sont minoritaires (plus d'observations),
  l'écart modal redevient mensuel et la conversion est correcte.
- **Attendu** : un écart nul (doublon) n'est jamais candidat à la fréquence ;
  la détection se fait sur les dates uniques (ici mensuelle, `'MS'`).
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.frequency import detect_index_frequency
  from tsforecast.utils.position import convert_position
  idx = pd.to_datetime(['2024-01-01', '2024-01-01', '2024-02-01', '2024-03-01'])
  detect_index_frequency(idx, return_format='full')   # 'ns'
  convert_position(idx, 'S', 'E')                    # dates inchangées
  ```
- **Correctif** : `FrequencyDetector.detect_time_series_frequency` dédoublonne
  l'index temporel avant `pd.infer_freq` et le calcul de l'écart modal.
- **Test** : `tests/unit/utils/position/test_converter.py::TestConvertDatetimeIndex::test_short_index_with_duplicates_is_converted`,
  `tests/unit/utils/frequency/test_detector.py::TestFrequencyDetectorDuplicatedDates`
- **Statut** : corrigée

### ANO-UTILS-015 — Index à fréquence multipliée (`'2MS'`) rejeté par la conversion de position
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/parse/utils.py::parse_frequency`,
  `tsforecast/utils/frequency/utils.py::normalize_frequency`
  (format `'full'`), `tsforecast/utils/position/converter.py::PeriodPositionConverter._convert_datetime_index`
- **Sévérité** : mineure
- **Observé** : `convert_offset('2MS', 'E')` renvoyait `'2ME'`, mais un index
  bimestriel était rejeté : `detect_index_frequency` échouait
  (`normalize_frequency('2MS', 'full')` → `Unsupported frequency: 2MS`, le
  multiplicateur n'étant pas géré par `parse_frequency`), et une `freq='2MS'`
  explicite échouait de même.
- **Attendu** : cohérence avec `convert_offset` : un index `2MS` se convertit
  par blocs de deux mois (`2024-01-01` → `2024-02-29`).
- **Correctif** : `parse_frequency` gère le multiplicateur (résultat
  `ParsedFrequency(freq, position, suffix, multiplier)`) et
  `build_frequency_string` l'accepte (`multiplier=`), ce qui supprime les trois
  contournements (`_split_multiplier`, `_strip_multiplier`, regex de
  `normalize_frequency`). `normalize_frequency(..., 'full')` valide la base et
  renvoie la chaîne complète ; la conversion d'index convertit des blocs de `n`
  périodes. Extension (2026-09-29) : `FrequencyNormalizer.normalize` et
  `DurationNormalizer.normalize` acceptent le multiplicateur (code de base
  renvoyé, `normalize_with_multiplier` le conserve) ; `normalize_frequency(...,
  'components')` renvoie un `ParsedFrequency` à 4 champs (les formats `'base'` et
  `'with_position'` sont inchangés) ; `is_higher_frequency`,
  `is_longer_duration` et `DurationConverter` en tiennent compte ;
  `FrequencyConverter` décompose `target_freq` via `ParsedFrequency` (les
  opérations à décompte de périodes de base rejettent le multiplicateur par
  `NotImplementedError`) ; `tsforecast.delays` rejette un index multiplié.
  `convert_offset('1MS', 'E')` renvoie `'ME'` (un multiplicateur 1 explicite
  est omis).
- **Test** : `tests/unit/utils/position/test_converter.py::TestConvertDatetimeIndex::test_multiplied_frequency`,
  `tests/unit/utils/frequency/test_utils.py::TestNormalizeFrequencyFullMultiplier`
- **Statut** : corrigée

### ANO-UTILS-016 — Semi-mensuel : `convert_offset('SMS', …)` renvoyait `'SM'`, index `SMS` en échec
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/parse/utils.py::build_frequency_string`,
  `tsforecast/utils/position/converter.py::PeriodPositionConverter`,
  `tsforecast/utils/frequency/detector.py::FrequencyDetector._detect_semi_monthly_frequency`
- **Sévérité** : mineure
- **Observé** : `SM` n'était pas « position-aware » : `convert_offset('SMS',
  'E')` renvoyait `'SM'` (position perdue) ; la conversion d'un index `SMS`
  levait l'erreur brute de pandas `SME-15 is not supported as period
  frequency` ; le détecteur ne reconnaissait pas la grille `SME` (15 et fin de
  mois) et renvoyait `'SM'` sans position pour `SMS`.
- **Attendu** : les grilles natives pandas `SMS` (1 et 15) et `SME` (15 et fin
  de mois) sont des positions début / fin l'une de l'autre.
- **Correctif** : `SM` ajouté à `_POSITION_AWARE_FREQUENCIES` ; conversion
  d'index par rang de la demi-période dans le mois (1er ↔ 15, 15 ↔ fin de
  mois) ; variantes multipliées ou à jour non standard (`2SMS`, `SMS-10`)
  rejetées par un message dédié ; détecteur renvoyant `'SMS'` / `'SME'` sur
  ces grilles exactes (motif approché : `'SM'`, inchangé).
- **Test** : `tests/unit/utils/position/test_converter.py::TestConvertDatetimeIndex::test_semi_monthly_pairs_native_grids`,
  `::test_unsupported_semi_monthly_variant_raises`,
  `tests/unit/utils/frequency/test_detector.py::TestFrequencyDetectorSemiMonthly`
- **Statut** : corrigée

### ANO-UTILS-017 — Détection trimestrielle : ancre non canonique (`'QS-OCT'` pour un index `QS`)
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/utils.py::detect_index_frequency`,
  `tsforecast/utils/frequency/detector.py::FrequencyDetector.detect_time_series_frequency`
- **Sévérité** : cosmétique
- **Observé** : `pd.infer_freq` renvoie une ancre quelconque de la classe
  d'équivalence (`QS-JAN` = `QS-APR` = `QS-JUL` = `QS-OCT`) : un index `QS`
  débutant en janvier est détecté `'QS-OCT'`, un index `QS-FEB` `'QS-NOV'`.
  Les dates générées sont identiques, mais la chaîne est trompeuse.
- **Attendu** : une ancre canonique, stable et lisible.
- **Correctif** : `canonicalize_frequency` (utils, réexportée par
  `utils.frequency`) : début → `JAN` /
  `FEB` / `MAR`, fin (ou ancre sans position) → mois précédent (`DEC` /
  `JAN` / `FEB`), de sorte que les défauts pandas (`QS-JAN` / `QE-DEC`) sont
  préservés et qu'un début et sa fin canoniques décrivent les mêmes
  trimestres (`QS-FEB` / `QE-JAN`). Appliqué aux deux chemins de détection.
  La fonction est générale (point d'extension pour d'autres classes
  d'écritures équivalentes) et s'appuie sur `parse_frequency` /
  `build_frequency_string` : multiplicateur conservé.
- **Test** : `tests/unit/utils/frequency/test_utils.py::TestDetectIndexFrequencyAnchors`,
  `::TestCanonicalizeFrequency`
- **Statut** : corrigée

### ANO-UTILS-018 — `get_period_end(…, 'ns')` renvoie la date d'entrée : période vide
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/time/utils.py::get_period_end`
- **Sévérité** : mineure
- **Observé** : `date + timedelta(microseconds=0.001)` est arrondi à 0 par `timedelta` :
  `get_period_end(d, 'ns') == d` et `get_period_start(d, 'ns') == d`. La période
  `[début, fin)` est vide, `début <= date < fin` est faux.
- **Attendu** : `fin > date`. `datetime` ne sait pas représenter 1 ns : soit renvoyer
  un `Timestamp` (`pd.Timestamp(date) + pd.Timedelta(1, 'ns')`), soit rejeter `'ns'`
  explicitement plutôt que renvoyer une borne fausse.
- **Reproduction** :
  ```python
  from datetime import datetime
  from tsforecast.utils.time.utils import get_period_boundaries
  d = datetime(2023, 6, 15, 1, 2, 3, 456789)
  get_period_boundaries(d, 'ns')   # (d, d)
  ```
- **Correctif** : la période d'un instant `'ns'` dure une nanoseconde ; les fonctions retournent des `pd.Timestamp` (sous-classe de `datetime`), ce qui représente la borne.
- **Test** : `tests/unit/utils/time/test_utils.py::TestNanosecond`
- **Statut** : corrigée

### ANO-UTILS-019 — `get_period_end(…, 'ms')` n'est pas tronqué à la milliseconde
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/time/utils.py::get_period_end`
- **Sévérité** : majeure
- **Observé** : la branche `'ms'` fait `date + timedelta(milliseconds=1)` sans tronquer,
  alors que `s`, `min` et `h` tronquent avant d'ajouter. Pour
  `2023-06-15 14:35:47.123456` : début `.123000` (correct), fin `.124456` au lieu de
  `.124000`. La période `[.123000 ; .124456)` dure 1,456 ms, et la période suivante ne
  commence pas à `fin` (`get_period_start(fin, 'ms') = .124000`).
- **Attendu** : `fin = début + 1 ms`, comme pour les autres unités ; périodes contiguës.
- **Reproduction** :
  ```python
  from datetime import datetime
  from tsforecast.utils.time.utils import get_period_end
  get_period_end(datetime(2023, 6, 15, 14, 35, 47, 123456), 'ms')   # …47.124456
  ```
- **Correctif** : toutes les fréquences infra-journalières se ramènent à `début = date − (temps écoulé depuis l'origine mod n·unité)` et `fin = début + n·unité` ; plus de branche spécifique à `ms`.
- **Test** : `tests/unit/utils/time/test_utils.py::TestMillisecond`, `::TestPeriodProperties` (`ms` et `ns` inclus)
- **Statut** : corrigée

### ANO-UTILS-020 — `get_period_*` : entrée `pandas.Period` acceptée pour certaines fréquences seulement
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/time/utils.py::get_period_start`, `get_period_end`
- **Sévérité** : mineure (arbitrée)
- **Observé** : la signature n'annonce que `Timestamp` / `datetime`, et seul `Timestamp` est
  converti. Un `Period` passe par duck-typing : `D`, `B`, `SM`, `M`, `Q`, `Y` fonctionnent
  (`.year` / `.month` / `.day`), mais `W` (`TypeError: 'int' object is not callable`,
  `Period.weekday` est une propriété), `h`, `min`, `s`, `ms` (`AttributeError: 'Period' object has no
  attribute 'replace'`) et `us` (`IncompatibleFrequency`) échouent avec des erreurs opaques.
  `'ns'` renvoie le `Period` tel quel.
- **Attendu** : à trancher : soit `Period` est un type d'entrée (conversion par
  `.to_timestamp()` en tête de fonction), soit il est refusé avec un message explicite.
  Le comportement actuel est épinglé en attendant.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.time.utils import get_period_start
  p = pd.Period('2023-06-15', freq='D')
  get_period_start(p, 'M')   # datetime(2023, 6, 1)
  get_period_start(p, 'W')   # TypeError
  ```
- **Correctif** : arbitrage : `Period` est un type d'entrée, représenté par son premier instant (`to_timestamp(how='start')`), pour toutes les fréquences ; `datetime64` accepté ; `NaT` lève `ValueError`, les autres types `TypeError` avec message clair.
- **Test** : `tests/unit/utils/time/test_utils.py::TestPeriodInput`
- **Statut** : corrigée

### ANO-UTILS-021 — `get_period_*` : ancres et multiplicateurs de fréquence ignorés
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/time/utils.py::get_period_start`, `get_period_end`
- **Sévérité** : mineure (arbitrée)
- **Observé** : `normalize_frequency` réduit la fréquence à sa base : `'Q-JAN'`, `'QS-FEB'`
  donnent des trimestres civils, `'YS-JUL'` des années civiles, `'2MS'` / `'3D'` /
  `'2h'` une période d'**une** unité. Pour `W-*` le code documente ce choix
  (« toujours lundi ») ; pour `Q` / `Y` et les multiplicateurs, rien n'est documenté.
  Cohérence : ANO-UTILS-007 (corrigée) traitait comme un défaut majeur l'ancre ignorée
  par `PeriodPositionConverter`.
- **Attendu** : à trancher. Les bornes d'une période `Q-JAN` (mai-juil.) diffèrent de
  celles de `Q-DEC` ; soit les ancres sont gérées, soit un rejet explicite ou la
  docstring précise la limite. Comportement actuel épinglé.
- **Reproduction** :
  ```python
  from datetime import datetime
  from tsforecast.utils.time.utils import get_period_start
  get_period_start(datetime(2023, 6, 15), 'Q-JAN')   # 2023-04-01 ; pandas : 2023-05-01
  get_period_start(datetime(2023, 6, 15), '2MS')     # 2023-06-01 (période de 1 mois)
  ```
- **Correctif** : arbitrage : ancres et multiplicateurs sont pris en compte (`W-X` finit le jour X, `Q-X` / `QE-X` finissent en X, `QS-X` commencent en X, idem `Y`) ; `nX` donne des périodes de n unités alignées sur l'époque Unix (et l'ancre) ou sur le paramètre optionnel `origin`. Ancre invalide ou sur une base sans ancre : `ValueError`.
- **Test** : `tests/unit/utils/time/test_utils.py::TestPeriodAnchorsAndMultipliers`, `::TestPeriodOrigin`, `::TestInvalidFrequency`
- **Statut** : corrigée

### ANO-UTILS-022 — `resolve_date('')` renvoie `NaT` au lieu de lever `ValueError`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/time/utils.py::resolve_date`
- **Sévérité** : mineure
- **Observé** : `pd.to_datetime('')` renvoie `NaT`, retourné tel quel (un `NaT` est aussi
  un `datetime` : `resolve_date(pd.NaT)` le renvoie inchangé). Une date vide se
  propage ensuite silencieusement.
- **Attendu** : la docstring promet un `ValueError` pour toute valeur non résoluble ;
  une chaîne vide n'est pas une date.
- **Reproduction** :
  ```python
  from tsforecast.utils.time.utils import resolve_date
  resolve_date('')   # NaT
  ```
- **Correctif** : `resolve_date` lève `ValueError` quand le résultat est `NaT` (chaîne vide, `NaT` en entrée).
- **Test** : `tests/unit/utils/time/test_utils.py::TestResolveDate::test_unresolvable_value_raises_instead_of_returning_nat`
- **Statut** : corrigée

### ANO-UTILS-023 — `get_period_start` / `get_period_end` perdent le fuseau horaire
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/time/utils.py::get_period_start`, `get_period_end`
- **Sévérité** : majeure
- **Observé** : `Timestamp.to_pydatetime()` conserve le fuseau, mais les branches qui
  reconstruisent la date avec `datetime(...)` le perdent (début : toutes les fréquences
  sauf `'ns'` ; fin : `D`, `B`, `W`, `SM`, `M`, `Q`, `Y`). Les branches `s`, `min`, `h`,
  `us`, `ms` de `get_period_end` le conservent : pour `'h'`,
  `get_period_boundaries` renvoie un **début naïf** et une **fin aware**, et
  `début <= date` lève `TypeError: can't compare offset-naive and offset-aware datetimes`.
- **Attendu** : bornes du même fuseau que la date d'entrée (`tzinfo=date.tzinfo`).
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.time.utils import get_period_boundaries
  d = pd.Timestamp('2023-06-15 14:35', tz='Europe/Paris')
  get_period_boundaries(d, 'h')   # (naïf 14:00, aware 15:00)
  get_period_boundaries(d, 'M')   # (naïf, naïf)
  ```
- **Correctif** : calcul sur `pd.Timestamp` : fuseau conservé ; fréquences calendaires sur l'horloge murale (jour de 23 h / 25 h), infra-journalières sur l'horloge murale locale avec conservation du décalage UTC (heure ambiguë, fuseaux à +5:30).
- **Test** : `tests/unit/utils/time/test_utils.py::TestTimezone`
- **Statut** : corrigée

### ANO-UTILS-024 — `restore_original_structure` réaffecte l'index d'origine par position à des lignes triées
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/validation/utils.py::restore_original_structure`
  (cause : `validate_temporal_data(sort_data=True)` réordonne les lignes, alors que les
  métadonnées ne gardent que `original_index` ; `was_sorted` est enregistré mais jamais lu)
- **Sévérité** : majeure
- **Observé** : dès que l'entrée n'est pas triée, `data_work.index = metadata['original_index']`
  colle l'index d'origine (dans l'ordre d'entrée) sur des lignes déjà triées. **Chemin index** :
  la série `[30, 10, 20]` sur `[2023-03-01, 2023-01-01, 2023-02-01]` revient avec
  `2023-03-01 → 10`, `2023-01-01 → 20`, `2023-02-01 → 30` : chaque valeur est rattachée à
  la **mauvaise date**, sans erreur ni avertissement. **Chemin colonnes** : les lignes
  reviennent triées, avec les étiquettes `0..n-1` de l'entrée (l'ordre d'origine n'est pas
  restitué). Même effet sur un panel dont les blocs d'entités ne sont pas dans l'ordre
  lexicographique (`reverse_entities`). Appelé par
  `PanelTimeSeriesTransformer._restore_structure_if_converted`
  (`tsforecast/base/transformers.py`, `convert_cols_to_index=True`). Le test existant `test_restoration_with_unsorted_data`
  comparait les dates triées des deux côtés : il ne pouvait pas le voir.
- **Attendu** : `restore_original_structure(validate_temporal_data(x, return_metadata=True))`
  restitue `x` à l'identique (valeurs, index et ordre des lignes), triée ou non à l'entrée ;
  à défaut d'ordre restituable, ne jamais rattacher une valeur à une autre étiquette.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.validation import validate_temporal_data, restore_original_structure
  s = pd.Series([30, 10, 20], name='v',
                index=pd.to_datetime(['2023-03-01', '2023-01-01', '2023-02-01']))
  v, meta = validate_temporal_data(s, return_metadata=True)
  restore_original_structure(v, meta)   # 2023-03-01 -> 10 (au lieu de 30)
  ```
- **Correctif** : décision de l'auteur (2026-09-29) : **renoncer** à restituer l'ordre
  d'origine (aucun cas d'usage à des lignes non triées). `validate_temporal_data` note dans
  les métadonnées `rows_reordered` (le tri a réellement déplacé des lignes) ;
  `restore_original_structure` n'affecte plus `original_index` quand `rows_reordered` est
  vrai (ni quand le nombre de lignes a changé) : les lignes restent triées, avec l'index
  qu'elles ont (chemin colonnes : étiquettes `0..n-1` neuves). Chaque valeur garde sa date.
- **Test** : `tests/unit/utils/validation/test_utils.py::TestRestoreOriginalStructure::test_index_path_restoration_after_sort_keeps_each_value_on_its_date`,
  `::test_column_path_restoration_after_sort_gives_the_sorted_rows`,
  `::TestReturnedMetadata::test_rows_reordered_flag`,
  `::TestRoundTripOnPerturbedDatasets::test_round_trip_of_unsorted_data_gives_the_sorted_frame`
  (`shuffle_rows` et `reverse_entities`, chemins index et colonnes) ; l'identité sans tri
  reste testée par `::test_round_trip_of_unsorted_data_without_sort_is_the_identity`
- **Statut** : corrigée

### ANO-UTILS-025 — `restore_original_structure` ne restitue pas la position des colonnes d'origine
- **Type** : [CODE] comportement (le docstring promet « column positions »)
- **Composant** : `tsforecast/utils/validation/utils.py::restore_original_structure`
- **Sévérité** : mineure
- **Observé** : `reset_index()` place les colonnes d'index (panel puis temps) **en tête** ;
  `metadata['original_columns']` n'est jamais lu. Une entrée `['date', 'v', 'entity']`
  revient en `['entity', 'date', 'v']`. Sans effet quand les colonnes d'index étaient déjà
  en tête, dans cet ordre (cas des jeux `reset_index()` de la campagne).
- **Attendu** : colonnes restituées dans l'ordre d'origine (les colonnes ajoutées après la
  validation restant en fin), comme l'annonce le docstring.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.validation import validate_temporal_data, restore_original_structure
  df = pd.DataFrame({'date': pd.date_range('2023-01-01', periods=3).tolist() * 2,
                     'v': range(6), 'entity': ['A'] * 3 + ['B'] * 3})
  v, meta = validate_temporal_data(df, time_col='date', panel_cols=['entity'], return_metadata=True)
  list(restore_original_structure(v, meta).columns)   # ['entity', 'date', 'v']
  ```
- **Correctif** : après `reset_index()`, les colonnes sont réordonnées selon
  `metadata['original_columns']` ; les colonnes ajoutées depuis restent en fin.
- **Test** : `tests/unit/utils/validation/test_utils.py::TestRestoreOriginalStructure::test_columns_are_restored_at_their_original_positions`,
  `::test_columns_added_after_validation_stay_last`, `::test_missing_column_record_leaves_the_reset_order`
- **Statut** : corrigée

### ANO-UTILS-026 — `restore_original_structure` sur une `Series` (chemin colonnes) renvoie la colonne temporelle
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/validation/utils.py::restore_original_structure`
- **Sévérité** : mineure (le type de `X` passé par `_restore_structure_if_converted` n'a pas été vérifié)
- **Observé** : la `Series` est convertie en frame, l'index (temps) est remis en colonne
  par `reset_index()`, puis `data_work.iloc[:, 0]` renvoie la **première colonne, c'est-à-dire
  les dates**, à la place des valeurs. Le test existant `test_restoration_with_series` ne
  vérifiait que le type et l'absence de NaN : il passait à vide.
- **Attendu** : les valeurs de la `Series` ne sont jamais remplacées par les dates. La forme
  exacte (valeurs sur le `RangeIndex` d'origine, ou frame avec la colonne temporelle) est à
  décider ; le test n'exige que les valeurs.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.validation import validate_temporal_data, restore_original_structure
  df = pd.DataFrame({'date': pd.date_range('2023-01-01', periods=3), 'v': [10, 20, 30]})
  v, meta = validate_temporal_data(df, time_col='date', return_metadata=True)
  restore_original_structure(v['v'], meta)   # les dates, pas [10, 20, 30]
  ```
- **Correctif** : une `Series` n'est plus convertie en frame : elle garde ses valeurs et
  récupère son index d'origine (sous les mêmes conditions qu'en ANO-024) ; les dates, portées
  par l'index validé seul, sont abandonnées quand l'index est remplacé. Après un tri, la
  `Series` garde son index temporel.
- **Test** : `tests/unit/utils/validation/test_utils.py::TestRestoreOriginalStructure::test_a_series_of_the_column_path_gets_its_original_index_back`,
  `::test_a_reordered_series_keeps_its_time_index`
- **Statut** : corrigée

### ANO-UTILS-027 — Une `Series` non nommée ressort nommée `0`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/validation/utils.py::validate_temporal_data` et `::restore_original_structure`
- **Sévérité** : mineure
- **Observé** : `Series.to_frame()` sur une série sans nom crée la colonne `0` ;
  `data.iloc[:, 0]` renvoie ensuite une `Series` **nommée `0`** au lieu de `None`. Vrai
  aussi pour `restore_original_structure`. Le nom d'une série nommée est, lui, conservé.
- **Attendu** : le nom de la `Series` en entrée (y compris `None`) est celui de la sortie.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.validation import validate_temporal_data
  s = pd.Series([1, 2, 3], index=pd.date_range('2023-01-01', periods=3))
  validate_temporal_data(s).name   # 0
  ```
- **Correctif** : `validate_temporal_data` réaffecte `data.name` à la `Series` renvoyée ;
  `restore_original_structure` ne convertit plus la `Series` en frame.
- **Test** : `tests/unit/utils/validation/test_utils.py::TestValidateTemporalDataInputContract::test_unnamed_series_stays_unnamed`,
  `::TestRestoreOriginalStructure::test_unnamed_series_round_trips_unnamed`
- **Statut** : corrigée

### ANO-UTILS-028 — Index entier (`RangeIndex`, années) converti en nanosecondes depuis 1970
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/validation/utils.py::validate_temporal_data`
  (`_check_datetime_convertible`, `_validate_index_based`)
- **Sévérité** : à arbitrer
- **Observé** : « convertible en date » signifie « `pd.to_datetime` ne lève pas ». Un
  `RangeIndex` (`0, 1, …`) ou des années entières (`2020, 2021`) passent **en mode strict**
  et deviennent `1970-01-01 00:00:00.000002020`, `…2021` : des dates sans rapport avec les
  données, sans erreur ni avertissement. Même chose pour le dernier niveau d'un `MultiIndex`.
  La même conversion nue est faite par `TimeSeriesTransformerMixin._validate_time_index`
  (`tsforecast/base/transformers.py`). La documentation (`docs/concepts/temporal_utils.md`)
  ne dit rien du cas.
- **Attendu** : à trancher. Refuser un index entier en mode strict (il ne représente pas un
  instant), ou documenter l'acceptation. Comportement actuel épinglé.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.validation import validate_temporal_data
  validate_temporal_data(pd.DataFrame({'v': [1, 2]}, index=[2020, 2021])).index[0]
  # Timestamp('1970-01-01 00:00:00.000002020')
  ```
- **Correctif** : décision de l'auteur (2026-09-29) : **rejeter** un index entier. Le
  helper `_convert_to_datetime` (qui remplace `_check_datetime_convertible`) refuse tout
  index / colonne à dtype numérique (entiers, flottants), sauf s'il est vide. Effet : erreur
  en strict, avertissement puis données inchangées sinon (index simple et dernier niveau de
  `MultiIndex`) ; erreur dans les deux modes pour `time_col`. Aligné sur les mêmes règles :
  `TimeSeriesTransformerMixin._validate_time_index` (`tsforecast/base/transformers.py`), qui
  réutilise `_convert_to_datetime` (auparavant `pd.to_datetime` nu). Effet de bord : le test
  `delays` `TestShiftTransformerEdgeCases::test_non_datetime_index` (« rejet des index
  non-datetime ») passe, retiré de `tests/legacy_failures.txt`.
- **Test** : `tests/unit/base/test_transformers.py::TestValidateTimeIndex` (numérique rejeté, `Period`, `MultiIndex`, `time_col`),
  `tests/unit/utils/validation/test_utils.py::TestIndexBasedValidation::test_numeric_index_is_rejected_when_strict`,
  `::test_default_range_index_is_rejected_when_strict`, `::test_numeric_index_warns_and_returns_data_when_not_strict`,
  `::TestMultiIndexValidation::test_integer_last_level_is_rejected_when_strict`,
  `::TestColumnBasedValidation::test_numeric_time_col_is_rejected`
- **Statut** : corrigée

### ANO-UTILS-029 — `PeriodIndex` refusé par la validation temporelle
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/validation/utils.py::validate_temporal_data`,
  `::validate_sorted_within_groups` ; même cause dans
  `tsforecast/base/transformers.py::TimeSeriesTransformerMixin._validate_time_index`
- **Sévérité** : à arbitrer
- **Observé** : `pd.to_datetime` rejette un `PeriodIndex` (`TypeError: Passing PeriodDtype
  data is invalid`). `validate_temporal_data` lève donc `ValueError: Index cannot be
  converted to datetime` en mode strict (index simple, dernier niveau de `MultiIndex` ou
  colonne `time_col` : `Column 'date' cannot be converted to datetime`), et en mode non
  strict avertit puis renvoie l'index **inchangé** (toujours `PeriodIndex`).
  `validate_sorted_within_groups` lève `Time series data must have DatetimeIndex`.
  `_validate_time_index` lève « Cannot determine time index » : c'est le message qui fait
  échouer des tests `delays` (voir le rapport U5, à signaler pour D3-D5).
- **Attendu** : à trancher. `CLAUDE.md` cite `Period` parmi les types de dates à tester « là
  où l'API les accepte » ; soit conversion par `to_timestamp()`, soit refus documenté.
  Comportement actuel épinglé.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.validation import validate_temporal_data
  s = pd.Series([1, 2, 3], index=pd.period_range('2023-01', periods=3, freq='M'))
  validate_temporal_data(s)   # ValueError: Index cannot be converted to datetime
  ```
- **Correctif** : décision de l'auteur (2026-09-29) : **accepter** `PeriodIndex`. Il est
  converti à son premier instant (`to_timestamp()`, convention d'ANO-UTILS-020) pour un index
  simple, le dernier niveau d'un `MultiIndex` et une colonne `time_col` ;
  `restore_original_structure` rend le `PeriodIndex` d'origine sur le chemin index.
  `validate_sorted_within_groups` accepte un `PeriodIndex` (série simple ou dernier niveau).
  `_validate_time_index` (`base`) est aligné : il accepte un `PeriodIndex` et lit le dernier
  niveau d'un `MultiIndex` (il échouait sur les deux avec « Cannot determine time index »),
  et renvoie un `DatetimeIndex` pour une `time_col` (c'était une `Series`). Effet de bord :
  sept tests d'intégration `delays` (`TestShiftTransformerWithPanelwise`,
  `TestMaskTransformerWithPanelwise`, `TestPerformance::test_large_panel_performance`), qui
  échouaient sur ce message avec un panel `MultiIndex`, passent ; retirés de
  `tests/legacy_failures.txt` (groupe D5).
- **Test** : `tests/unit/utils/validation/test_utils.py::TestIndexBasedValidation::test_period_index_is_converted_to_first_instants`,
  `::test_unsorted_period_index_is_sorted`, `::TestMultiIndexValidation::test_period_last_level_is_converted_to_first_instants`,
  `::TestColumnBasedValidation::test_period_time_col_is_converted_to_first_instants`,
  `::TestRestoreOriginalStructure::test_period_index_is_restored_after_conversion`,
  `::TestRoundTripOnPerturbedDatasets::test_period_index_dataset_round_trips`, `::test_period_column_dataset_is_validated`,
  `::test_period_index_panel_round_trips`, `::TestValidateSortedWithinGroups::test_plain_series_with_period_index`,
  `::test_panel_with_period_dates`
- **Statut** : corrigée

### ANO-UTILS-030 — Dernier niveau de `MultiIndex` converti sur les niveaux triés, pas sur les valeurs
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/validation/utils.py::validate_temporal_data`
  (`_validate_index_based`, branche `MultiIndex`)
- **Sévérité** : mineure
- **Observé** : le contrôle `_check_datetime_convertible` porte sur les **valeurs** du
  dernier niveau, la conversion sur ses **niveaux** (`index.levels[-1]`, uniques et triés
  lexicographiquement). Pour des chaînes non ISO, pandas déduit le format de la première
  valeur rencontrée : `['15/01/2023', '02/02/2023']` sur un index simple donne
  `2023-01-15`, `2023-02-02` (jour d'abord, déduit de `15/01`), mais sur un `MultiIndex` les
  niveaux triés commencent par `02/02/2023` (lu mois d'abord), d'où
  `ValueError: Failed to convert MultiIndex last level: time data "15/01/2023" doesn't
  match format "%m/%d/%Y"` en strict, et avertissement + niveau non converti sinon. Le
  résultat dépend de l'ordre lexicographique des chaînes, et le message d'erreur est
  contradictoire avec le contrôle qui vient de passer.
- **Attendu** : mêmes dates que l'index simple pour les mêmes valeurs (conversion des
  valeurs du niveau, pas de ses niveaux triés).
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.validation import validate_temporal_data
  idx = pd.MultiIndex.from_arrays([['A', 'A'], ['15/01/2023', '02/02/2023']])
  validate_temporal_data(pd.DataFrame({'v': [1, 2]}, index=idx))
  # ValueError: Failed to convert MultiIndex last level: ...
  ```
- **Correctif** : la conversion porte sur les **valeurs** du dernier niveau ; l'index est
  reconstruit par `MultiIndex.from_arrays`. Le `try / except` de conversion, devenu
  inatteignable, est supprimé (voir ANO-UTILS-032).
- **Test** : `tests/unit/utils/validation/test_utils.py::TestMultiIndexValidation::test_day_first_string_dates_convert_like_a_flat_index`
- **Statut** : corrigée

### ANO-UTILS-031 — Docstring de `validate_temporal_data` : `strict=False` ne « corrige » pas une colonne temporelle illisible
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/utils/validation/utils.py::validate_temporal_data`
- **Sévérité** : cosmétique
- **Observé** : le docstring décrit `strict` : « If True, raises errors on validation
  failures; if False, attempts corrections ». Sur le chemin index, `strict=False` avertit et
  renvoie les données. Sur le chemin colonnes, une `time_col` non convertible lève
  `ValueError: Column 'date' cannot be converted to datetime` **dans les deux modes**
  (aucun repli possible : pas de colonne temporelle, pas d'index). Le mode non strict ne
  couvre en fait que les doublons et la conversion d'index.
- **Attendu** : le docstring précise que `strict=False` ne concerne pas la conversion de
  `time_col` (le code est juste).
- **Correctif** : docstring de `validate_temporal_data` (`strict`, types de dates acceptés)
  précisé ; code inchangé.
- **Test** : `tests/unit/utils/validation/test_utils.py::TestColumnBasedValidation::test_non_convertible_time_col_raises_in_both_modes`
  (suit le code, pas de `xfail`)
- **Statut** : corrigée

### ANO-UTILS-032 — `validate_temporal_data` : branches inatteignables (code mort)
- **Type** : [CODE] comportement (nettoyage, pas un bogue)
- **Composant** : `tsforecast/utils/validation/utils.py::_validate_index_based`,
  `::_validate_column_based`
- **Sévérité** : cosmétique
- **Observé** : (1) dans `_validate_index_based`, le `try / except` autour de
  `pd.to_datetime(data.index)` (index simple) ne peut jamais lever : le même appel vient de
  réussir dans `_check_datetime_convertible` ; (2) dans `_validate_column_based`, même
  situation pour `pd.to_datetime(data[time_col])`, et les branches `if time_col:` fausses
  (lignes `498->508`, `511->515`) sont inatteignables : `panel_cols` sans `time_col` est
  rejeté dès le début de `validate_temporal_data`, et sans `time_col` ni `panel_cols` c'est
  le chemin index qui est pris. Décelé par la couverture : ces lignes sont les seules non couvertes
  de `utils.py` (96 %, lignes et branches).
- **Attendu** : suppression du code mort, sans changement de comportement.
- **Correctif** : les trois `try / except` de conversion et les branches `if time_col:`
  sont supprimés ; `_validate_column_based` exige `time_col` (`str`). `utils.py` : 100 %
  de couverture (lignes et branches).
- **Test** : couverture 100 % de `utils.py` par les tests de `tests/unit/utils/validation/test_utils.py`
- **Statut** : corrigée

### ANO-UTILS-033 — Avertissement « Index replaced » émis à chaque appel du chemin colonnes
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/validation/utils.py::validate_temporal_data`
- **Sévérité** : cosmétique
- **Observé** : le chemin `time_col` émettait `UserWarning: Index replaced with [...]. Use
  return_metadata=True and restore_original_structure() to revert.` même quand
  `return_metadata=True` était déjà passé.
- **Attendu** : pas d'avertissement quand l'appelant a demandé les métadonnées de restauration.
- **Correctif** : l'avertissement n'est émis que sans `return_metadata`.
- **Test** : `tests/unit/utils/validation/test_utils.py::TestColumnBasedValidation::test_replacement_of_the_index_is_announced_without_metadata`,
  `::test_replacement_of_the_index_is_silent_with_metadata`
- **Statut** : corrigée

### ANO-UTILS-034 — `validate_sorted_within_groups` : `except Exception` masque une erreur de structure en « non trié »
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/validation/utils.py::validate_sorted_within_groups`
  (et `validate_entities_grouped` pour les colonnes ambiguës)
- **Sévérité** : mineure
- **Observé** : une `time_col` (ou un `panel_col`) dont le nom désigne deux colonnes fait de
  `data[time_col]` un frame ; le contrôle vectorisé échouait, et le `except Exception` du repli
  répondait `False` (« non trié »), sans signaler le mauvais usage.
- **Attendu** : erreur explicite pour un nom de colonne ambigu ; pas de repli silencieux.
- **Correctif** : contrôle explicite `_check_columns_unique` (`ValueError: Column 'date' is not
  unique in data`) dans les deux fonctions ; le `try / except` est supprimé.
  Des dates incomparables (chaîne et entier) donnent toujours `False`
  (`is_monotonic_increasing`), sans exception.
- **Test** : `tests/unit/utils/validation/test_utils.py::TestValidateSortedWithinGroups::test_ambiguous_time_col_is_rejected`,
  `::TestValidateEntitiesGrouped::test_ambiguous_panel_col_is_rejected`
- **Statut** : corrigée

### ANO-UTILS-035 — Docstring de `get_frequency_order` : `7.0` et « renvoie 0 si inconnue » ≠ comportement observé
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/utils/frequency/utils.py::get_frequency_order`
- **Sévérité** : cosmétique
- **Observé** : le docstring annonce `Returns: Frequency order as float` et « Returns 0 if
  frequency is not found in the order mapping », avec les exemples `get_frequency_order('daily')
  -> 7.0` et `'monthly' -> 9.0`. Le code renvoie l'`int` `7` (seuls `'B'` et `'SM'` valent `7.5` /
  `8.5`), donc `--doctest-modules` échoue sur ces exemples. Une fréquence inconnue ne renvoie pas
  `0` : `normalize_frequency` lève `ValueError('Unsupported frequency: ...')` avant, et le défaut
  de `_normalizer._frequency_order.get(base_freq, 0)` est inatteignable (tout code renvoyé par
  `normalize` est une clé de la table d'ordre : même situation qu'ANO-UTILS-005).
- **Attendu** : docstring documentant le type réel (`int` ou `float`) et la levée de `ValueError` ;
  le défaut `0` peut être retiré (code mort). Même famille qu'ANO-UTILS-003 (`get_duration_order`).
- **Reproduction** :
  ```python
  from tsforecast.utils.frequency.utils import get_frequency_order
  get_frequency_order('daily')   # 7 (int), pas 7.0
  get_frequency_order('xyz')     # ValueError, pas 0
  ```
- **Test** : `tests/unit/utils/frequency/test_utils.py::TestGetFrequencyOrder::test_documented_values`,
  `::test_order_is_a_number`, `::test_unsupported_frequency_raises` (le test suit le code).
- **Correctif** : docstring de `get_frequency_order` corrigé : type réel (`int`, ou `float` pour `'B'` et `'SM'`), `Raises: ValueError`, multiplicateur ignoré ; le défaut `.get(base_freq, 0)` inatteignable est remplacé par un accès direct `[base_freq]`. `--doctest-modules` passe.
- **Statut** : corrigée

### ANO-UTILS-036 — `convert_frequency` (fonction) levait toujours `TypeError`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/utils.py::convert_frequency`
- **Sévérité** : majeure
- **Observé** : la fonction appelle `FrequencyConverter().convert(value, to_unit, **kwargs)`, alors
  que `FrequencyConverter.convert(value, from_unit, to_unit, **kwargs)` attend trois arguments
  (`from_unit` y est ignoré, la fréquence source étant détectée). Tout appel échoue :
  `TypeError: FrequencyConverter.convert() missing 1 required positional argument: 'to_unit'`.
  Le docstring est lui aussi faux : type de retour annoncé `float` (le résultat est une `Series` /
  un `DataFrame`) et exemple `convert_frequency(series, 'daily', 'monthly', method='mean')` à trois
  arguments positionnels sur une signature `(value, to_unit, **kwargs)`
  (`TypeError: convert_frequency() takes 2 positional arguments but 3 were given`). La fonction est
  exportée par `tsforecast.utils.frequency` et citée dans `docs/concepts/temporal_utils.md` ; aucun
  appelant interne (les notebooks et la spec parlent de la *méthode*
  `FrequencyConverter.convert_frequency`, qui fonctionne).
- **Attendu** : `convert_frequency(series, 'MS', method='mean')` renvoie la série convertie, comme
  `FrequencyConverter().convert_frequency(series, 'MS', method='mean')`. Valeur d'or : valeurs
  journalières 1..59 (janvier-février 2023) -> `[16.0, 45.5]` (moyenne de 1..31, puis de 32..59).
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.frequency.utils import convert_frequency
  s = pd.Series(range(1, 60), index=pd.date_range('2023-01-01', periods=59, freq='D'), dtype=float)
  convert_frequency(s, 'MS', method='mean')  # TypeError
  ```
- **Test** : test supprimé avec la fonction (`TestConvertFrequency`) ; la méthode `FrequencyConverter.convert_frequency` reste testée par `tests/unit/utils/frequency/converter/`.
- **Correctif** : `convert_frequency` **supprimée** (aucun appelant dans `tsforecast/`) de `utils.py` et de `tsforecast.utils.frequency.__all__` ; `docs/concepts/temporal_utils.md` renvoie à `FrequencyConverter.convert()` / `convert_frequency()` (la méthode, inchangée).
- **Statut** : corrigée

### ANO-UTILS-037 — Entrée non `str` : `TypeError` (ou `unhashable`) au lieu de `ValueError`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/normalizer.py::FrequencyNormalizer.to_pandas_freq` /
  `to_dateoffset` ; `tsforecast/utils/frequency/utils.py::normalize_frequency` (formats autres que
  `'base'`)
- **Sévérité** : mineure
- **Observé** : `normalize`, `to_literal`, `to_code`, `get_frequency_order` lèvent
  `ValueError("Frequency must be a string, got ...")` et `validate` renvoie `False` pour un entier,
  un flottant, une liste, un dictionnaire ou des `bytes`. `to_pandas_freq` / `to_dateoffset` et
  `normalize_frequency(..., return_format='with_position' | 'full' | 'components')` lèvent
  `TypeError: expected string or bytes-like object, got 'int'` (`re.match` dans `parse_frequency`),
  `TypeError: unhashable type: 'list'` (`x in dict`) ou `TypeError: cannot use a string pattern on
  a bytes-like object`. `None` donne un `ValueError` dont le message parle de détection d'index
  (« Could not detect index frequency »). Le contrat de `TemporalNormalizer.normalize` promet une
  `ValueError`, et `normalize_with_multiplier` intercepte déjà `(ValueError, TypeError)` :
  l'auteur attendait donc ce `TypeError` sans le traiter ailleurs.
- **Attendu** : une `ValueError` uniforme, quel que soit le point d'entrée (cohérence avec les
  autres méthodes du même normaliseur ; `notebooks/utils/frequency_normalizer.ipynb` §2.5, à propos de
  `normalize` : « lève toujours un `ValueError` (jamais un `TypeError`) »).
- **Reproduction** :
  ```python
  from tsforecast.utils.frequency.utils import normalize_frequency, to_pandas_freq
  normalize_frequency(123, return_format='full')  # TypeError (ValueError en 'base')
  to_pandas_freq(123)                              # TypeError
  ```
- **Test** : `tests/unit/utils/frequency/test_normalizer.py::TestRebuiltPandasStrings::test_non_string_input_raises_value_error`, `tests/unit/utils/frequency/test_utils.py::TestNormalizeFrequencyReturnFormats::test_non_string_raises_value_error_in_every_format`
- **Correctif** : `to_pandas_freq` vérifie le type avant tout accès à un dictionnaire ; `normalize_frequency` valide la chaîne complète par `FrequencyNormalizer.normalize` avant de la décomposer (formats `with_position`, `full`, `components`). `ValueError('Frequency must be a string, got ...')` partout, `None` compris. Les `xfail(strict)` sont retirés.
- **Statut** : corrigée

### ANO-UTILS-038 — `is_higher_frequency('2D', '2B')` : ni l'un ni l'autre n'est plus fin
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/normalizer.py::FrequencyNormalizer.is_higher_frequency`
- **Sévérité** : mineure (arbitrée par l’auteur)
- **Observé** : sans multiplicateur, `'D'` est plus fin que `'B'` (ordre 7 contre 7.5). Dès qu'un
  multiplicateur > 1 est présent, la comparaison porte sur les durées nominales, identiques pour
  `D` et `B` (`_CONVERSION_FACTORS_TO_SECONDS`) : `is_higher_frequency('2D', '2B')` et
  `('2B', '2D')` valent tous deux `False`. L'incomparabilité n'est plus transitive :
  `'D'` et `'24h'` sont incomparables, `'24h'` et `'B'` aussi, mais `'D'` est plus fin que `'B'`.
  Irréflexivité, antisymétrie et transitivité de « plus fin que » restent vérifiées (propriétés
  testées sur 75 fréquences multipliées).
- **Attendu** : à arbitrer. Un jour ouvré saute les week-ends, donc `kB` est plus grossier que
  `kD` (cohérence avec le cas sans multiplicateur), ce qui suppose de départager les durées égales
  par l'ordre des codes ; ou bien assumer l'égalité nominale, sans départage pour `k = 1` non plus.
- **Reproduction** :
  ```python
  from tsforecast.utils.frequency.utils import is_higher_frequency
  is_higher_frequency('D', 'B')                                   # True
  is_higher_frequency('2D', '2B'), is_higher_frequency('2B', '2D')  # (False, False)
  ```
- **Test** : `tests/unit/utils/frequency/test_utils.py::TestIsHigherFrequencyDayVersusBusinessDay`
- **Correctif** : `is_higher_frequency` compare les durées nominales avec un jour ouvré valant 7/5 de jour calendaire (5 observations par semaine) : `'2D'` est plus fin que `'2B'`, comme `'D'` que `'B'`. Le tableau de conversion partagé (`_CONVERSION_FACTORS_TO_SECONDS`, `'B'` = `'D'`) est inchangé ; l'écart est local à la comparaison (`_NOMINAL_SECONDS`). Deux codes de même durée (`'24h'`, `'D'`) restent incomparables, et l'incomparabilité est désormais transitive ; `'5B'` égale `'W'`.
- **Statut** : corrigée

### ANO-UTILS-039 — `to_pandas_freq` / `to_dateoffset` : codes sans position = alias dépréciés par pandas
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/normalizer.py::FrequencyNormalizer.to_pandas_freq`
  (et `to_dateoffset`)
- **Sévérité** : mineure (arbitrée par l’auteur)
- **Observé** : `to_pandas_freq('monthly')` renvoie `'M'` (valeur donnée par le docstring),
  `'quarterly'` -> `'Q'`, `'annual'` -> `'Y'`, `'semi_monthly'` -> `'SM'`. Sous pandas 2.3
  (`pandas>=2.3.1,<3` dans `pyproject.toml`), `to_dateoffset` de ces quatre codes émet
  `FutureWarning: 'M' is deprecated and will be removed in a future version, please use 'ME'
  instead` (idem `'Q'` -> `'QE'`, `'Y'` -> `'YE'`, `'SM'` -> `'SME'`) ; pandas 3 les supprimera. Les
  variantes positionnées (`MS`, `ME`, `QS`, ...) n'émettent rien.
- **Attendu** : à arbitrer. Soit conserver le code nu comme code *interne* (le docstring de
  `to_pandas_freq` promet pourtant « pandas frequency code »), soit renvoyer l'alias positionné par
  défaut (`'ME'`, `'QE'`, `'YE'`, `'SME'`) avant la montée vers pandas 3.
- **Reproduction** :
  ```python
  from tsforecast.utils.frequency.utils import to_pandas_freq, to_dateoffset
  to_pandas_freq('monthly')   # 'M'
  to_dateoffset('monthly')    # FutureWarning: 'M' is deprecated ... please use 'ME' instead
  ```
- **Test** : `tests/unit/utils/frequency/test_normalizer.py::TestBareCodesGiveTheEndVariant`, `tests/unit/utils/frequency/test_utils.py::TestToPandasFreqEveryCodeAndPosition`
- **Correctif** : `to_pandas_freq` (et donc `to_dateoffset`) renvoie la variante fin par défaut des codes sans position : `'M'` → `'ME'`, `'Q'` → `'QE'`, `'Y'` → `'YE'`, `'SM'` → `'SME'` (multiplicateur et ancre conservés : `'2M'` → `'2ME'`, `'Q-DEC'` → `'QE-DEC'`). Mêmes dates que les alias nus de pandas 2, aucun avertissement, valides en pandas 3. Le code de base (`'M'`) reste renvoyé par `normalize` / `to_code` / `normalize_frequency`. Seul appelant du package : `delays/calculator.py`, qui veut un alias pour `pd.date_range` (comportement inchangé).
- **Statut** : corrigée

### ANO-UTILS-040 — L'ancre d'une chaîne de fréquence n'est jamais validée
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/normalizer.py::FrequencyNormalizer.normalize` /
  `validate` / `to_pandas_freq`
- **Sévérité** : mineure (arbitrée par l’auteur)
- **Observé** : seule la base est contrôlée. `validate('QS-XYZ')`, `validate('MS-JAN')`,
  `validate('D-MON')`, `validate('YE-FOO')`, `validate('W-XYZ')` valent `True` ;
  `to_pandas_freq` renvoie la chaîne telle quelle ; l'erreur ne vient que de pandas quand l'offset
  est construit (`ValueError: Invalid frequency: QS-XYZ, failed to parse with error message ...`).
  `normalize` documente pourtant seulement une *extraction* de la base ; et `validate` promet de
  dire si la fréquence est « supportée ».
- **Attendu** : à arbitrer : valider l'ancre (mois pour `Q` / `Y`, jour de semaine pour `W`, aucune
  pour `D`, `M`, ...), ou documenter que `validate` ne vérifie que la base.
- **Reproduction** :
  ```python
  from tsforecast.utils.frequency.utils import validate_frequency, to_dateoffset
  validate_frequency('QS-XYZ')   # True
  to_dateoffset('QS-XYZ')        # ValueError (pandas)
  ```
- **Test** : `tests/unit/utils/frequency/test_normalizer.py::TestAnchorValidation` (dont un test de propriété confrontant 375 chaînes à `pandas.to_offset`)
- **Correctif** : `FrequencyNormalizer._validate_anchor`, appelée par `normalize` (donc par `validate`, `to_pandas_freq`, `to_dateoffset` et `normalize_frequency`) : mois pour `Q` / `Y`, jour de semaine pour `W`, jour du mois derrière une position pour `SM` (2 à 27 pour `SMS`, 1 à 27 pour `SME`), aucune ancre ailleurs ; majuscules uniquement. Message : `Unsupported frequency: QS-XYZ. Invalid anchor 'XYZ' for base frequency 'Q': expected a month among JAN, ...`. Nouvelle constante `WEEKDAY_ABBREVIATIONS` (`utils/parse/utils.py`).
- **Statut** : corrigée

### ANO-UTILS-041 — `build_frequency_string` : un code sans position sortait comme alias pandas déprécié
- **Type** : [CODE] comportement (suite d'ANO-UTILS-039)
- **Composant** : `tsforecast/utils/parse/utils.py::build_frequency_string`
- **Sévérité** : mineure (arbitrée par l’auteur : migration vers pandas 3)
- **Observé** : `build_frequency_string('M')` renvoie `'M'`, alias que pandas 2.2 déprécie et que
  pandas 3 supprime. Quatre appelants transmettaient ce résultat à pandas
  (`FrequencyConverter._with_position` -> `resample` ; `MaskTransformer` / `ShiftTransformer`
  ×3 -> `pd.date_range`, la position étant `None` pour un index journalier) : `FutureWarning`
  aujourd'hui, `ValueError` avec pandas 3. Trois autres appelants veulent au contraire le **code
  de base** (`FrequencyConverter._duration_of` et le calcul de ratio d'`_extend_index_for_upsampling`
  passent la chaîne à `DurationConverter`, `canonicalize_frequency` préserve l'orthographe
  `'Q-DEC'`) : un défaut global les aurait cassés.
- **Attendu** : distinguer les deux intentions par l'appelant, sans changer le comportement par défaut.
- **Correctif** : nouveau paramètre `default_position` (`'S'`, `'E'` ou `None`) : appliqué aux
  fréquences `M`, `Q`, `Y`, `SM` sans position explicite (`'M'` -> `'ME'`), sans effet sur les autres
  ni sur une position explicite ; `None` (défaut) laisse le code de base nu. Passé à `'E'` par les
  quatre appelants « alias pandas » et par `FrequencyNormalizer.to_pandas_freq` (qui reprend ainsi
  ANO-UTILS-039 sur le même mécanisme). Inchangés : les appelants « durée » et ceux dont la
  position est explicite ou déjà `'E'` par défaut (`imputation_window`).
  Non traité : le repli de `target_offset_for_index` (position source indétectable) renvoie la
  fréquence cible fournie, donc un `'M'` nu si l'appelant en donne un (choix documenté).
- **Test** : `tests/unit/utils/parse/test_utils.py::TestDefaultPosition`,
  `tests/unit/utils/frequency/converter/test_positions.py::TestWithPositionIsAPandasAlias`,
  `tests/unit/delays/test_transformers.py::TestMaskTransformerPandasAliases`
- **Statut** : corrigée

## DELAYS

### ANO-DELAYS-001 — Docstrings de `calculator.py` périmées
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/delays/calculator.py` (`calculate_applicable_delay`,
  `_convert_to_target_frequency_and_reference`, `_calculate_converted_delay`, `_convert_delay_unit`,
  `_aggregate_delays`, `_validate_columns`)
- **Sévérité** : cosmétique
- **Observé** : `unit` documenté comme limité à `us` / `s` / `D` alors que toute durée de
  `convert_duration` est acceptée (`'hour'`, `'W'`, ...) ; `frequency` documenté comme chaîne seule
  alors qu'un dictionnaire `{indicateur: fréquence}` est accepté ; index attendu (dernier niveau =
  indicateur) et unités d'entrée valides non décrits ; `Raises` sans `TypeError` ; colonne `unit`
  de sortie contenant le **code** (`'D'`) et non l'étiquette d'entrée (`'day'`) quand `unit` est
  fourni ; exemple de `_calculate_converted_delay` avec `'unit': 'days'` (rejeté : `Unsupported
  duration: days`), `observation_date` absent de la liste des colonnes de `row`, fin de période
  annoncée « Mar 31 » alors qu'elle est **exclusive** (`2024-04-01`), exemples non exécutables
  (`delays_df` indéfini, sorties en commentaires). Message d'erreur `TypeError` : « git a » pour « got a ».
- **Correctif** : docstrings réécrits d'après le comportement observé ; exemples exécutables sur un
  jeu de données défini, sorties calculées à la main (75 j = 15 mai - 1er mars ; médiane de 75 et 85 = 80 ;
  1920 h) ; coquille corrigée. `pytest --doctest-modules tsforecast/delays/calculator.py` passe.
- **Test** : doctests du module (le test suit le code).
- **Statut** : corrigée

### ANO-DELAYS-002 — Unités d'entrée mixtes agrégées sans conversion quand `unit=None`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/calculator.py::calculate_applicable_delay` / `_aggregate_delays`
- **Sévérité** : majeure
- **Observé** : `converted_delay` reste dans l'unité de chaque ligne ; sans `unit`, `_aggregate_delays`
  agrège ces valeurs telles quelles et étiquette le résultat avec l'unité du premier élément
  (`'unit': 'first'`). Deux lignes du même indicateur à 75 `day` et 6 480 000 `second` donnent
  un délai médian de `3240037.5` étiqueté `day`.
- **Attendu** : convertir vers une unité commune (celle de la première ligne, ou la plus fine) avant
  d'agréger, ou lever une `ValueError` si les unités diffèrent. Le docstring demande aujourd'hui que
  les lignes d'un indicateur partagent une unité, sans que le code le vérifie.
- **Reproduction** :
  ```python
  # deux lignes (FR, DE) de l'indicateur GDP : delay=[45, 3888000], unit=['day', 'second']
  calculate_applicable_delay(delays, 'start', 'M')  # delay=3240037.5, unit='day'
  ```
- **Test** : à écrire au prompt D2 (`tests/unit/delays/test_calculator.py::TestEdgeCases::test_mixed_units`
  existe, marqué hérité).
- **Statut** : ouvert

### ANO-DELAYS-003 — `frequency` en dictionnaire incomplet : message d'erreur trompeur
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/calculator.py::_convert_to_target_frequency_and_reference`
- **Sévérité** : mineure
- **Observé** : un indicateur absent du dictionnaire reçoit `NaN` comme fréquence cible (`.map`), puis
  `normalize_frequency` lève `ValueError: Frequency must be a string, got <class 'float'>`, sans
  nommer l'indicateur.
- **Attendu** : une erreur nommant les indicateurs sans fréquence cible.
- **Reproduction** :
  ```python
  calculate_applicable_delay(delays, 'start', {'GDP': 'M'})  # avec un indicateur CPI dans les données
  ```
- **Test** : à écrire au prompt D2.
- **Statut** : ouvert


## FREQ

_Aucune entrée._
