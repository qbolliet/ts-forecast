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
  `tests/unit/utils/frequency/detector/test_time_series.py::TestFrequencyDetectorDuplicatedDates`
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
  `tests/unit/utils/frequency/detector/test_time_series.py::TestFrequencyDetectorSemiMonthly`
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
- **Test** : `tests/unit/base/transformers/test_time_index.py::TestValidateTimeIndex` (numérique rejeté, `Period`, `MultiIndex`, `time_col`),
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

### ANO-UTILS-042 — Série temporelle : une colonne indétectable fait échouer toute la détection, ou disparaît
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/detector.py::FrequencyDetector._detect_time_series_frequencies`
  (via `detect_dataset_frequency` / `detect_frequency` sur un `DataFrame` non panel)
- **Sévérité** : mineure
- **Observé** : deux comportements, tous deux différents du panel. (1) Une colonne ayant moins de
  `min_observations` valeurs (colonne entièrement NaN, observée une fois) lève
  `ValueError: Series has only 1 non-null observations, minimum required is 2` pour **tout** le
  jeu : `_detect_time_series_frequencies` appelle `detect_frequency` (qui lève) et non
  `_detect_column_frequency` (qui renvoie `None`), comme le fait le chemin panel. (2) Une colonne
  dont l'espacement n'est pas reconnu (`None`) est **omise** de la carte (`elif freq_result:`).
  Un jeu de données vide lève la même `ValueError`.
- **Attendu** : la clé présente, associée à `None`, comme pour un couple (entité, colonne) depuis
  le commit `906da2e` (« rather than being silently dropped »). C'est aussi la convention de
  `HighFrequencyImputer._detect_frequencies_robustly` / `_undetected_frequencies_`, qui contourne
  aujourd'hui l'exception par une détection colonne par colonne.
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.utils.frequency import detect_dataset_frequency
  df = pd.DataFrame({'dense': np.arange(5.0), 'sparse': [np.nan] * 4 + [1.0]},
                    index=pd.date_range('2024-01-01', periods=5, freq='D'))
  detect_dataset_frequency(df)   # ValueError: Series has only 1 non-null observations ...
  # attendu : {'dense': 'D', 'sparse': None}
  ```
- **Test** : `tests/unit/utils/frequency/detector/test_panel.py::TestDetectDatasetFrequencyTimeSeries::test_column_with_too_few_observations_is_none`,
  `::test_column_with_unrecognized_spacing_is_none`, `::test_empty_frame_maps_every_column_to_none`
- **Correctif** : décision de l'auteur (2026-09-30) : `None` est la convention. `_detect_time_series_frequencies`
  passe par `_detect_column_frequency` et garde toutes les colonnes ; `_detect_column_frequency` ne rattrape plus
  d'exception : il compte les valeurs non nulles et renvoie `None` sous `min_observations`, toute autre erreur
  (index non temporel, `return_format` inconnu) remonte. Effets de bord : `ImputationWindowCalculator` traite
  désormais une colonne vide de série temporelle comme en panel (structurellement absente, exclue du
  dénominateur, avertissement) — le test `tests/unit/frequency/test_imputation_window.py::TestStructurallyAbsentColumns::test_notion_is_panel_only_on_a_time_series`,
  qui épinglait l'ancienne exception, est réécrit en `::test_empty_column_of_a_time_series_is_absent` (catégorie
  (a)) ; `delays/data_manager.py::compare_and_detect_delays` ne passe plus `None` à `to_literal`.
- **Statut** : corrigée

### ANO-UTILS-043 — Repli heuristique : ancre non par défaut perdue, offset hors de la grille des données
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/detector.py::FrequencyDetector._detect_day_frequency`
  (et `_with_detected_position` / `_detect_period_position`)
- **Sévérité** : mineure
- **Observé** : quand `pd.infer_freq` échoue (deux dates, ou grille à trous), le repli ne renvoie
  que des codes sans ancre. Semaine au lundi → `'W'` (lu `W-SUN` par pandas) ; trimestres
  février-mai-août-novembre → `'Q'` (trous) ou `'QS'` (deux dates, l'index portant sa `freq`) ;
  exercice juillet-juin → `'Y'` / `'YS'`. En format `'full'`, l'offset renvoyé ne contient
  **aucune** des dates observées (`is_on_offset` faux) ; `'components'` perd l'ancre (et la
  position, `is_quarter_start` étant calendaire sur un index sans `freq`). Les grilles à ancre
  par défaut (`W-SUN`, `MS`/`ME`, `QS-JAN`/`QE-DEC`, `YS-JAN`/`YE-DEC`) ne sont pas touchées.
- **Attendu** : l'ancre lue sur les dates (jour de semaine pour `W`, mois pour `Q` / `Y`), avec
  forme canonique (`canonicalize_frequency`), comme sur le chemin `pd.infer_freq` : sur une grille
  régulière, `QS-FEB`, `W-MON`, `YS-JUL` sont bien détectés.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.frequency import detect_index_frequency
  idx = pd.date_range('2024-01-01', periods=12, freq='W-MON').delete([3, 7, 8])
  detect_index_frequency(idx, return_format='full')   # 'W' (dimanches), attendu 'W-MON'
  detect_index_frequency(pd.date_range('2020-02-01', periods=2, freq='QS-FEB'), return_format='full')
  # 'QS' (janvier, avril, ...), attendu 'QS-FEB'
  ```
- **Test** : `tests/unit/utils/frequency/detector/test_time_series.py::TestFallbackKeepsNonDefaultAnchors`
- **Correctif** : le repli lit les grilles calendaires (dates toutes en début de mois, toutes en fin de mois,
  ou toutes le même jour du mois) en **mois** et ancre trimestres et années sur le mois des dates, sous forme
  canonique (`_detect_calendar_frequency`) ; les grilles hebdomadaires sont ancrées sur leur jour de semaine
  commun (`_detect_weekly_frequency`, `'W-SUN'` explicite comme sur le chemin `pd.infer_freq`). Remplace
  `_with_detected_position` / `_detect_period_position` (calendaires, sans ancre). Voir aussi ANO-UTILS-051.
- **Statut** : corrigée

### ANO-UTILS-044 — Détection : index entier lu comme des nanosecondes depuis 1970
- **Type** : [CODE] comportement (même cause qu'ANO-UTILS-028)
- **Composant** : `tsforecast/utils/frequency/detector.py::FrequencyDetector.detect_time_series_frequency`,
  `tsforecast/utils/frequency/utils.py::detect_index_frequency`
- **Sévérité** : mineure
- **Observé** : une série indexée par des années (`2020, 2021, …`) ou un `RangeIndex` est convertie
  par `pd.to_datetime` nu et détectée `'ns'`, sans erreur ; `detect_index_frequency` sur le même
  index lève `AttributeError: 'Index' object has no attribute 'inferred_freq'`.
- **Attendu** : `ValueError`, selon la décision de l'auteur pour ANO-UTILS-028 (2026-09-29 :
  **rejeter** un index numérique), déjà appliquée par `validate_temporal_data` et
  `TimeSeriesTransformerMixin._validate_time_index` (helper `_convert_to_datetime`).
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.frequency import detect_frequency, detect_index_frequency
  detect_frequency(pd.Series([1.0, 2.0, 3.0]))             # 'ns'
  detect_index_frequency(pd.Index([2020, 2021, 2022]))     # AttributeError
  ```
- **Test** : `tests/unit/utils/frequency/detector/test_time_series.py::TestDetectTimeSeriesFrequencyIndexTypes::test_integer_index_is_rejected`,
  `tests/unit/utils/frequency/detector/test_index_and_offset.py::TestDetectIndexFrequency::test_integer_index_raises_value_error`,
  `tests/unit/utils/frequency/detector/test_panel.py::TestDetectorDetectFrequencyPanelSeries::test_non_date_level_raises`
- **Correctif** : conversion de l'index par `tsforecast/utils/validation/utils.py::_convert_to_datetime` (le
  helper d'ANO-UTILS-028 / 029) dans `detect_time_series_frequency` et, pour un index non `DatetimeIndex`,
  dans `detect_index_frequency` : `ValueError: ... cannot be converted to datetime (numeric labels are not
  dates)`. Sur un panel, un niveau de dates entier lève aussi (il n'est plus avalé en `None`, cf. ANO-UTILS-047).
- **Statut** : corrigée

### ANO-UTILS-045 — Détection : `PeriodIndex` refusé
- **Type** : [CODE] comportement (même cause qu'ANO-UTILS-029)
- **Composant** : `tsforecast/utils/frequency/detector.py::FrequencyDetector.detect_time_series_frequency`,
  `tsforecast/utils/frequency/utils.py::detect_index_frequency`
- **Sévérité** : mineure
- **Observé** : `pd.to_datetime` refuse un `PeriodIndex` : `ValueError: Series index cannot be
  converted to datetime` pour une série, `None` pour un couple de panel (erreur avalée par
  `_detect_column_frequency`), `AttributeError: 'PeriodIndex' object has no attribute
  'inferred_freq'` pour `detect_index_frequency`.
- **Attendu** : décision de l'auteur pour ANO-UTILS-029 (2026-09-29) : **accepter** les périodes,
  converties à leur premier instant (`to_timestamp()`) ; un `PeriodIndex` mensuel est mensuel.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.frequency import detect_frequency, detect_index_frequency
  detect_frequency(pd.Series([1.0, 2.0, 3.0], index=pd.period_range('2024-01', periods=3, freq='M')))
  # ValueError: Series index cannot be converted to datetime
  detect_index_frequency(pd.period_range('2024Q1', periods=4, freq='Q'))   # AttributeError
  ```
- **Test** : `tests/unit/utils/frequency/detector/test_time_series.py::TestDetectTimeSeriesFrequencyIndexTypes::test_period_index_is_accepted`,
  `tests/unit/utils/frequency/detector/test_panel.py::TestDetectDatasetFrequencyPanel::test_period_dates`,
  `tests/unit/utils/frequency/detector/test_index_and_offset.py::TestDetectIndexFrequency::test_period_index`,
  `::test_period_date_level`
- **Correctif** : même helper `_convert_to_datetime` qu'ANO-UTILS-044 : périodes converties à leur premier
  instant ; un `PeriodIndex` mensuel est détecté `'M'` / `'MS'`, trimestriel `'QS-JAN'`.
- **Statut** : corrigée

### ANO-UTILS-046 — Panel à niveaux d'index sans nom : `TypeError`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/detector.py::FrequencyDetector._detect_panel_frequencies`
  (structure fournie par `tsforecast/panel/utils.py::detect_panel_structure`)
- **Sévérité** : mineure
- **Observé** : sur un `DataFrame` dont le `MultiIndex` (entité, date) n'a pas de noms,
  `detect_panel_structure` renvoie `panel_cols=[None]`, puis `df.groupby(level=None)` lève
  `TypeError: You have to supply one of 'by' and 'level'`. Une `Series` au même index est, elle,
  détectée (groupement par position). Le correctif d'ANO-UTILS-011 contournait déjà ce cas
  côté `PeriodPositionConverter`.
- **Attendu** : niveaux d'entité désignés par leur position quand ils n'ont pas de nom ; même
  carte qu'avec des niveaux nommés (cas « noms d'index non standards » de `CLAUDE.md`).
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.frequency import detect_dataset_frequency
  dates = pd.date_range('2023-01-01', periods=3, freq='D').tolist()
  idx = pd.MultiIndex.from_arrays([['A'] * 3 + ['B'] * 3, dates * 2])
  detect_dataset_frequency(pd.DataFrame({'v': range(6)}, index=idx))   # TypeError
  ```
- **Test** : `tests/unit/utils/frequency/detector/test_panel.py::TestDetectDatasetFrequencyPanel::test_unnamed_levels`
- **Correctif** : `_detect_panel_frequencies` groupe par **position** de niveau : position du nom pour un
  niveau nommé, rang pour un niveau sans nom (les niveaux auto-détectés sont les premiers de l'index).
  `detect_panel_structure` (`panel/utils.py`) est inchangée.
- **Statut** : corrigée

### ANO-UTILS-047 — Panel : `return_format` invalide avalé, chaque couple associé à `None`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/detector.py::FrequencyDetector._detect_column_frequency`
  (appelée par `detect_frequency` sur `MultiIndex` et `_detect_panel_frequencies`)
- **Sévérité** : mineure
- **Observé** : `_detect_column_frequency` intercepte **toute** `ValueError` pour traduire « trop
  peu d'observations » en `None` ; l'erreur `Invalid return_format` est interceptée de la même
  façon. Sur un panel, `return_format='bogus'` renvoie donc `{(entité, colonne): None, …}` sans
  erreur, alors qu'une série simple ou un `DataFrame` non panel lèvent `ValueError`.
- **Attendu** : `ValueError: Invalid return_format` quel que soit le type de données (erreur de
  l'appelant, pas propriété des données) ; seul le manque d'observations mène à `None`.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.frequency import detect_dataset_frequency
  idx = pd.MultiIndex.from_product([['A'], pd.date_range('2024-01-01', periods=3)], names=['e', 'd'])
  detect_dataset_frequency(pd.DataFrame({'v': range(3)}, index=idx), return_format='bogus')
  # {('A', 'v'): None}
  ```
- **Test** : `tests/unit/utils/frequency/detector/test_panel.py::TestDetectorDetectFrequencyPanelSeries::test_invalid_return_format_raises`,
  `::TestDetectDatasetFrequencyPanel::test_invalid_return_format_raises`
- **Correctif** : `return_format` vérifié à l'entrée de `detect_time_series_frequency`, `detect_frequency` et
  `detect_dataset_frequency` (`_check_return_format`), y compris pour un panel sans ligne ; plus aucune
  exception avalée par `_detect_column_frequency` (cf. ANO-UTILS-042).
- **Statut** : corrigée

### ANO-UTILS-048 — `target_offset_for_index` perd le multiplicateur et l'ancre de la cible
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/utils.py::target_offset_for_index`
- **Sévérité** : mineure
- **Observé** : la cible est réduite à son code de base (`normalize_frequency(…, 'base')`) avant
  d'y greffer la position de la source. `'2Q'` (semestre) sur un index `ME` devient `'QE'`
  (trimestre) ; `'QE-NOV'` sur un index `MS` devient `'QS'` (`QS-JAN` : trimestres janv.-mars au
  lieu de déc.-févr.) ; `'YE-JUN'` devient `'YS'` (année civile au lieu de l'exercice juillet-juin).
  Les périodes d'agrégation changent silencieusement. Les appelants actuels
  (`FrequencyAligner`, `ImputationWindowCalculator`, `CovariateMaterializer`) passent des codes à
  ancre par défaut sans multiplicateur : sans effet observé aujourd'hui.
- **Attendu** : seule la position change ; multiplicateur et périodes conservés (`'2QE'`,
  `'QS-DEC'` ou équivalent canonique `'QS-MAR'`, `'YS-JUL'`). Même esprit qu'ANO-UTILS-015.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.frequency import target_offset_for_index
  target_offset_for_index(pd.date_range('2024-01-31', periods=6, freq='ME'), '2Q')       # 'QE'
  target_offset_for_index(pd.date_range('2024-01-01', periods=6, freq='MS'), 'QE-NOV')   # 'QS'
  ```
- **Test** : `tests/unit/utils/frequency/detector/test_index_and_offset.py::TestTargetOffsetForIndex::test_target_multiplier_is_kept`,
  `::test_target_periods_are_kept`, `::test_reanchored_target_describes_the_same_periods`
- **Correctif** : la cible est décomposée (`'components'`) ; multiplicateur conservé ; une ancre mensuelle de
  trimestre ou d'année est déplacée sur l'autre bord des mêmes périodes (fin en novembre ↔ début en décembre,
  ancre sans position lue comme une fin, comme pandas), puis mise sous forme canonique ; l'ancre par défaut de
  pandas reste implicite (`'QS'`, pas `'QS-JAN'`, sorties inchangées pour les appelants actuels). Résultats :
  `'2Q'` → `'2QE'`, `'QE-NOV'` → `'QS-MAR'` (≡ `'QS-DEC'`), `'YE-JUN'` → `'YS-JUL'`.
- **Statut** : corrigée

### ANO-UTILS-049 — Contrôles de cohérence : les couples indétectables (`None`) comptent comme une fréquence
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/detector.py::FrequencyDetector.validate_frequency_consistency`
  (via `detect_frequency` / `detect_dataset_frequency` avec `check_consistency=True`)
- **Sévérité** : à arbitrer
- **Observé** : depuis le commit `906da2e`, les cartes de panel contiennent des `None` ; les
  contrôles de cohérence n'ont pas été adaptés. En mode strict, une seule entité indétectable
  rend le panel « incohérent » (`None`) alors que toutes les autres ont la même fréquence ; en
  mode modal non strict, `None` est compté comme une fréquence et peut être **renvoyé** s'il est
  majoritaire. Le mode `'highest'` (`_get_highest_frequency`) ignore déjà les `None`.
- **Attendu** : décision de l'auteur (2026-09-30) : ignorer les `None` (inconnu ≠ différent) dans les
  deux modes, comme `'highest'`.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.utils.frequency import detect_frequency
  idx = pd.MultiIndex.from_tuples(
      [('A', pd.Timestamp('2024-01-01')), ('B', pd.Timestamp('2024-01-01'))]
      + [('C', d) for d in pd.date_range('2024-01-01', periods=3)])
  detect_frequency(pd.Series(range(5), index=idx), check_consistency=True, strict=False)   # None
  ```
- **Test** : `tests/unit/utils/frequency/detector/test_consistency.py::TestConsistencyWithUndetectablePairs`
- **Correctif** : `validate_frequency_consistency` écarte les `None` avant tout calcul ; une carte sans aucune
  fréquence détectée donne `(False, None)` ; un vrai désaccord reste incohérent en mode strict.
- **Statut** : corrigée

### ANO-UTILS-050 — Docstrings de la détection ≠ comportement
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/utils/frequency/detector.py::FrequencyDetector`,
  `::FrequencyDetector.detect_dataset_frequency`, `tsforecast/utils/frequency/utils.py::detect_index_frequency`,
  `::detect_dataset_frequency`
- **Sévérité** : cosmétique
- **Observé** :
  - exemple de la classe `FrequencyDetector` : `detect_frequency(series)` annoncé `'monthly'`,
    renvoie `'M'` (seul doctest en échec des deux modules ; l'exemple utilise en outre l'alias
    déprécié `freq='M'`) ;
  - `detect_index_frequency` : « With 'components' format, returns a tuple of (base, position,
    suffix) » — c'est un `ParsedFrequency` à **quatre** champs (multiplicateur) ; « Raises
    ValueError … irregular spacing » — un espacement non reconnu renvoie `None` sans erreur ;
  - `detect_dataset_frequency` (méthode et fonction) : « (panel_id, column) tuples » — les clés
    sont **aplaties** `(entité…, colonne)` depuis `67529a5` (la docstring privée
    `_detect_panel_frequencies` le dit correctement).
- **Attendu** : docstrings alignées sur le code (le code est juste).
- **Test** : `tests/unit/utils/frequency/detector/test_index_and_offset.py::TestDetectIndexFrequency::test_irregular_index_is_none`,
  `::test_irregular_index_gives_the_dominant_grid`,
  `tests/unit/utils/frequency/detector/test_panel.py::TestDetectDatasetFrequencyPanel::test_three_level_index_keys_are_flat`
- **Correctif** : docstrings réalignées (exemple de classe `'M'` / `'ME'`, `ParsedFrequency` à quatre champs,
  `None` sur espacement sans écart dominant, clés aplaties, `None` des couples indétectables). Ajout, à la
  demande de l'auteur, de la documentation de l'**écart modal** : sur un index irrégulier, la fréquence
  renvoyée est celle de la grille dominante (utile pour imputer sur cette grille), pas un test de
  régularité (`is_regular`) — `detect_time_series_frequency` et `detect_index_frequency`.
- **Statut** : corrigée

### ANO-UTILS-051 — Repli heuristique sans multiplicateur
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/detector.py::FrequencyDetector._extend_infer_freq` et ses auxiliaires
- **Sévérité** : mineure
- **Observé** : le repli (grille à trous, deux dates) ne produisait que des codes simples, reconnus par
  plages d'écart modal (1, 7, 13-16, 28-31, 89-92, 365-366 jours ; 5 % autour de 1 h / 1 min / 1 s ; ordre de
  grandeur sous la seconde). Une grille bimestrielle, bihebdomadaire ou semestrielle à trous était
  indétectable (`None`), alors que `pd.infer_freq` renvoie `'2MS'`, `'2W-WED'`, `'2QS-OCT'` sur la même grille
  sans trou ; un écart de 10 ms était lu `'ms'`.
- **Attendu** : demande de l'auteur (2026-09-30) : le repli produit aussi des multiplicateurs, avec les
  mêmes chaînes que le chemin `pd.infer_freq` (forme canonique). Non retenu, à la demande de l'auteur :
  deux jours ouvrés consécutifs restent `'D'` (aucun week-end observé).
- **Correctif** : mois comptés sur les grilles calendaires (`'2MS'`, semestres `'2QS-JAN'`, `'2YS-JAN'`),
  semaines ancrées (`'2W-WED'`), sinon multiple de la plus grande unité qui divise l'écart (`'3D'`, `'90min'`,
  `'36h'`, `'10ms'`). Garde-fou `_is_dominant` : un multiplicateur exige un écart modal **observé au moins
  deux fois** et **strictement majoritaire** — deux dates seules ne donnent jamais `'45D'` (sinon toute paire de
  dates aurait une fréquence ; c'est ce qu'a révélé `tests/unit/utils/position/test_converter.py::TestConvertPanel::test_column_frequency_fallback`),
  et des écarts de 45 puis 50 jours restent `None`. Un multiple de 7 jours n'est hebdomadaire que si les dates
  partagent un jour de semaine (91 jours entre jours variables : `'Q'`, pas `'13W'`). Tolérance d'une heure
  sur les jours entiers (changement d'heure d'un index localisé). Les alias `pd.infer_freq` non pris en charge
  (`'BME'`) passent toujours par le repli.
- **Test** : `tests/unit/utils/frequency/detector/test_time_series.py::TestFallbackMultipliedFrequencies`,
  `::TestFallbackWithoutCalendarPattern`, `::TestDetectTimeSeriesFrequencyTwoObservations`,
  `::TestDetectTimeSeriesFrequencyWithGaps::test_timezone_aware_gapped_grid`
- **Statut** : corrigée

### ANO-UTILS-052 — Sur-échantillonnage à rapport non entier (M → W) : grille source conservée, valeurs écrasées
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter._extend_index_for_upsampling`
  (via `interpolate_to_higher_frequency` et `convert_frequency`)
- **Sévérité** : majeure
- **Observé** : `get_duration_conversion_factor('M', 'W')` = 30/7 n'est pas entier : l'extension renvoie
  l'index **d'origine** (mensuel). Le ré-ancrage sur `'W'` déplace chaque fin de mois au dimanche qui
  clôt sa semaine (04/02, 03/03, 31/03) ; seul le 31/03 coïncide avec l'index d'origine → `[NaN, NaN, 3]`,
  puis comblement arrière → `[3, 3, 3]` sur un index `ME`. Deux observations sur trois perdues, en
  silence, et aucune date hebdomadaire. Q → W (rapport 13) et Y → W (52) fonctionnent.
- **Attendu** : docstring de `convert_frequency` : « The output index carries the target frequency » —
  une grille `W-SUN` couvrant les périodes source, les observations interpolées dessus.
- **Reproduction** :
  ```python
  monthly = pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2024-01-31', periods=3, freq='ME'))
  FrequencyConverter().convert_frequency(monthly, 'W', method='linear')
  # 2024-01-31 3.0 / 2024-02-29 3.0 / 2024-03-31 3.0 (Freq: ME)
  ```
- **Test** : `tests/unit/utils/frequency/converter/test_interpolation.py::TestUpsamplingTargetGrid::test_monthly_to_weekly_gives_a_weekly_grid`
- **Correctif** : `_extend_index_for_upsampling` n'exige plus un rapport de durées entier : dès que la cible est plus fine (rapport ≥ 1), la grille cible est construite par `date_range` sur les bornes des périodes source. M → W : dimanches du 7 janvier au 31 mars 2024, chaque fin de mois ré-ancrée sur le dimanche qui clôt sa semaine.
- **Statut** : corrigée

### ANO-UTILS-053 — Sur-échantillonnage semi-mensuel : cible `SMS` tout-NaN, source `SM` jamais densifiée
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter._reanchor_index_to_target`,
  `::_extend_index_for_upsampling`
- **Sévérité** : majeure
- **Observé** : `pd.Period` n'a pas de fréquence `'SM'`. (1) Cible `'SMS'` : `_reanchor_index_to_target`
  retombe sur l'index source (fins de mois), qui n'intersecte pas la grille des 1er et 15 → sortie de
  6 dates **entièrement NaN**, même défaut que la régression `QS → ME` corrigée auparavant
  (`test_positions.py::TestCrossPositionUpsampling`). (2) Source `'SME'` vers `'D'` :
  `_extend_index_for_upsampling` échoue sur `pd.Period(..., freq='SM')` et renvoie l'index d'origine → la
  sortie reste semi-mensuelle, aucune date journalière. `count_subperiods_per_period` a un repli
  constant pour `'SM'` ; le chemin d'interpolation n'en a pas.
- **Attendu** : les fréquences semi-mensuelles sont supportées par le paquet (`'semi_monthly'`,
  `_with_position('SM') == 'SME'`) : grille cible produite, observations conservées.
- **Reproduction** :
  ```python
  monthly = pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2024-01-31', periods=3, freq='ME'))
  FrequencyConverter().interpolate_to_higher_frequency(monthly, 'SMS')      # 6 dates, toutes NaN
  semi = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range('2024-01-15', periods=4, freq='SME'))
  FrequencyConverter().interpolate_to_higher_frequency(semi, 'D')           # index SME inchangé
  ```
- **Test** : `tests/unit/utils/frequency/converter/test_interpolation.py::TestUpsamplingTargetGrid::test_monthly_to_semi_monthly_keeps_the_observations`,
  `::test_semi_monthly_to_daily_gives_a_daily_grid`
- **Correctif** : périodes semi-mensuelles bornées par les offsets pandas `SMS` / `SME` (`_offset_block_bounds`) ; ré-ancrage sur une cible semi-mensuelle par `rollback` (`SMS`) ou `rollforward` (`SME`). `anchor_fraction` garde son repli documenté (comportement `None`) pour une source semi-mensuelle.
- **Statut** : corrigée

### ANO-UTILS-054 — Variable portée par une grille de lignes plus fine que la cible : `cannot reindex on an axis with duplicate labels`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter.interpolate_to_higher_frequency`
  (`_reanchor_index_to_target`), `::convert_frequency` (chemin DataFrame / panel)
- **Sévérité** : majeure
- **Observé** : une variable annuelle portée par une grille mensuelle (NaN hors janvier), sur-échantillonnée
  vers `'QS'` : `_reanchor_index_to_target` ré-ancre **toutes** les lignes, NaN compris, sur le trimestre
  qui les contient → trois lignes par trimestre → `ValueError: cannot reindex on an axis with duplicate
  labels`. `convert_frequency` passe toujours la grille complète (`data[columns]`) : c'est le cas de tout
  DataFrame à fréquences mixtes dont une colonne monte vers une fréquence intermédiaire —
  `irregular_index_timeseries[['balance_commerciale_annuelle']]` → `'QS'`, dépenses annuelles de la
  France (`heterogeneous_coverage_panel`) → `'QS'`. L'extension part aussi de la première et de la dernière
  **ligne** du cadre, pas des observations de la colonne. `FrequencyAligner._interpolate_series` évite le
  défaut en ne passant que les valeurs observées (`_observed_series`) : `HighFrequencyImputer` n'est pas
  exposé, l'API publique du convertisseur l'est. La docstring de `source_freq` annonce pourtant le cas
  (« a quarterly variable carried on a monthly index »).
- **Attendu** : interpoler les seules observations de la variable (comme `FrequencyAligner`) : 100, 110,
  120, 130, 140 puis 140 jusqu'à la fin de 2021 dans la reproduction. **À arbitrer ensuite** : même
  corrigé, un DataFrame dont certaines colonnes descendent (mensuel) et d'autres montent (annuel) vers la
  même cible ne peut pas être converti en un appel, `method` servant aux deux sens (toute valeur est
  invalide pour l'un des deux).
- **Reproduction** :
  ```python
  grid = pd.date_range('2020-01-01', '2021-12-01', freq='MS')
  annual = pd.Series(np.nan, index=grid)
  annual.loc[['2020-01-01', '2021-01-01']] = [100.0, 140.0]
  FrequencyConverter().interpolate_to_higher_frequency(annual, 'QS', source_freq='YS')
  # ValueError: cannot reindex on an axis with duplicate labels
  ```
- **Test** : `tests/unit/utils/frequency/converter/test_interpolation.py::TestUpsamplingTargetGrid::test_variable_on_a_finer_row_grid`,
  `tests/unit/utils/frequency/converter/test_realistic_datasets.py::TestIrregularIndexTimeseries::test_annual_column_to_quarters`,
  `::TestHeterogeneousCoveragePanel::test_annual_spending_to_quarters`
- **Correctif** : `interpolate_to_higher_frequency` trie les lignes et ne garde que les lignes observées (`dropna(how='all')`) avant détection, extension et ré-ancrage : l'extension part de la première et de la dernière observation. Deux observations dans une même période cible lèvent une `ValueError` explicite (« Several observations fall in the same … period »). **Arbitrage de l'auteur (2026-09-30)** : le cas des colonnes qui montent et descendent vers une même cible relève de `FrequencyAligner`, pas du convertisseur (une seule `method` par appel).
- **Statut** : corrigée

### ANO-UTILS-055 — Colonnes non converties : toute la grille des lignes survit, la sortie n'est plus à la fréquence cible
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter._apply_grouped_conversions`
  (et `::_build_frequency_map`)
- **Sévérité** : majeure
- **Observé** : une colonne non convertie — fréquence déjà égale à la cible, ou indétectable (jamais
  observée) — est réattachée avec `data.index` **entier**, lignes de bourrage NaN comprises. L'union des
  index garde alors la grille source et `alignment_method='ffill'` y propage les agrégats des colonnes
  converties (une somme trimestrielle répétée sur trois mois). Cas observés : cible `str` avec une
  colonne tout-NaN ou une colonne trimestrielle portée par la grille mensuelle ; cible `dict` dont une
  clé nomme une colonne jamais observée ; `heterogeneous_coverage_panel[['inflation_ipc',
  'climat_affaires']]` → `'QS'` : l'Italie (climat jamais observé) garde ses 79 lignes mensuelles ;
  `irregular_index_timeseries[['production_industrielle', 'pib_trimestriel']]` → `'QS'` : dates
  mensuelles en sortie ; `[['depenses_publiques_pib']]` → `'YS'` : France et Italie rendues telles
  quelles (grille mensuelle).
- **Attendu** : « The output index carries the target frequency » (docstring) : pour une cible `str`,
  index = grille cible, colonne déjà à la cible → ses observations sur cette grille, colonne jamais
  observée → NaN ; pour un `dict`, seules les colonnes **absentes** du dictionnaire gardent leurs dates
  (docstring) — une clé présente mais indétectable n'en fait pas partie.
- **Reproduction** :
  ```python
  df = pd.DataFrame({'ventes': np.arange(1.0, 7.0), 'prix': np.nan},
                    index=pd.date_range('2024-01-31', periods=6, freq='ME'))
  FrequencyConverter().convert_frequency(df, 'QE', method='sum')
  # 6 lignes mensuelles : ventes NaN, NaN, 6, 6, 6, 15 ; prix NaN
  ```
- **Test** : `tests/unit/utils/frequency/converter/test_conversion.py::TestDataFrameOutputContract::test_all_nan_column_does_not_keep_the_source_grid`,
  `::test_dict_key_on_all_nan_column_does_not_keep_the_source_grid`,
  `::test_column_already_at_target_does_not_keep_the_row_grid`,
  `tests/unit/utils/frequency/converter/test_realistic_datasets.py::TestIrregularIndexTimeseries::test_quarterly_output_has_quarter_starts_only`,
  `::TestHeterogeneousCoveragePanel::test_never_observed_column_does_not_keep_the_monthly_grid`
- **Correctif** : **arbitrage de l'auteur (2026-09-30)** : les colonnes absentes du dictionnaire gardent leurs valeurs aux seules dates observées. `_apply_grouped_conversions` construit l'index de sortie comme l'union des index cibles des colonnes converties et des dates observées des colonnes conservées (absentes du dict, déjà à la cible, jamais observées) ; sans colonne à convertir, seules les lignes où au moins une colonne est observée sont rendues (un cadre jamais observé rend zéro ligne, comme une Series : ANO-UTILS-070). Une entité de panel qu'aucune clé ne cible reste rendue telle quelle.
- **Statut** : corrigée

### ANO-UTILS-056 — DataFrame et panel : une cible sans position ignore la position de la source
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter._build_frequency_map`
- **Sévérité** : majeure
- **Observé** : le chemin Series résout `target_position or target.position or source.position`
  (source `MS` + cible `'Q'` → `QS`) ; `_build_frequency_map` s'arrête à `target_position or
  target.position` et retombe sur `'E'` (→ `QE`). Même appel, étiquettes différentes selon que la
  donnée est une Series ou un DataFrame ; les panels DataFrame (convertis entité par entité en
  DataFrame) héritent du défaut, les panels Series non. Risque : jointure silencieusement décalée avec
  des données en début de période. (Le notebook `frequency_converter.ipynb` §6.3 signalait l'écart pour
  les Series, corrigé depuis pour elles seules.)
- **Attendu** : docstring de `target_position` : « If None, preserves source position when
  identifiable, otherwise uses default 'E' », pour tous les types d'entrée, colonne par colonne.
- **Reproduction** :
  ```python
  c = FrequencyConverter()
  s = pd.Series(np.arange(1.0, 7.0), index=pd.date_range('2024-01-01', periods=6, freq='MS'))
  c.convert_frequency(s, 'Q', method='sum').index                # 2024-01-01, 2024-04-01
  c.convert_frequency(s.to_frame('a'), 'Q', method='sum').index  # 2024-03-31, 2024-06-30
  ```
- **Test** : `tests/unit/utils/frequency/converter/test_conversion.py::TestDataFrameOutputContract::test_dataframe_keeps_the_source_position`,
  `tests/unit/utils/frequency/converter/test_panel.py::TestPanelStringTarget::test_source_position_is_kept`
- **Correctif** : `_build_frequency_map` résout la position cible colonne par colonne : explicite, puis celle de la cible, puis celle de la source, puis `'E'` — la règle du chemin Series.
- **Statut** : corrigée

### ANO-UTILS-057 — `interpolate_to_higher_frequency` sur un index décroissant : extension de la première période perdue
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter.interpolate_to_higher_frequency`
- **Sévérité** : mineure
- **Observé** : la méthode publique ne trie pas son entrée (contrairement à `convert_frequency`, qui passe
  par `validate_temporal_data(sort_data=True)`). Sur un index décroissant, pandas infère `'-1QE-DEC'`, que
  le normaliseur refuse ; la `ValueError` est interceptée, la fréquence source devient `None` et le repli
  `asfreq` ne couvre que mars → décembre : 10 mois au lieu de 12, janvier et février absents.
  `aggregate_to_lower_frequency` n'est pas touchée (`resample` trie).
- **Attendu** : même sortie que sur l'entrée triée (cas limite « données non triées » de `CLAUDE.md`).
- **Reproduction** :
  ```python
  q = pd.Series([10.0, 20.0, 30.0, 40.0], index=pd.date_range('2021-03-31', periods=4, freq='QE'))
  FrequencyConverter().interpolate_to_higher_frequency(q.iloc[::-1], 'ME')  # 10 mois, dès le 31 mars
  ```
- **Test** : `tests/unit/utils/frequency/converter/test_interpolation.py::TestUpsamplingTargetGrid::test_unsorted_input_gives_the_sorted_result`
- **Correctif** : lignes triées en tête d'`interpolate_to_higher_frequency` et d'`aggregate_to_lower_frequency` (`_sorted`) ; la détection trie désormais elle-même un index décroissant (ANO-UTILS-069).
- **Statut** : corrigée

### ANO-UTILS-058 — Le sur-échantillonnage perd le nom de l'index
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter.interpolate_to_higher_frequency`
  (index étendu construit par `pd.date_range`, sans nom)
- **Sévérité** : cosmétique
- **Observé** : un index nommé `'date'` ressort sans nom (`None`) d'une interpolation, directe ou via
  `convert_frequency` ; l'agrégation le conserve, et le chemin panel restaure les noms.
- **Attendu** : nom de l'index conservé, comme par l'agrégation.
- **Reproduction** :
  ```python
  q = pd.Series([1.0, 2.0], index=pd.date_range('2021-03-31', periods=2, freq='QE', name='date'))
  FrequencyConverter().interpolate_to_higher_frequency(q, 'ME').index.name  # None
  ```
- **Test** : `tests/unit/utils/frequency/converter/test_interpolation.py::TestUpsamplingTargetGrid::test_index_name_is_kept`,
  `tests/unit/utils/frequency/converter/test_conversion.py::TestDataFrameOutputContract::test_upsampling_keeps_the_index_name`
- **Correctif** : nom de l'index source réaffecté à la sortie de l'interpolation et de l'assemblage des colonnes.
- **Statut** : corrigée

### ANO-UTILS-059 — `full_periods_only` ignore en silence une `source_freq` invalide
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter.aggregate_to_lower_frequency`
- **Sévérité** : mineure
- **Observé** : le décompte attendu est calculé dans un `try / except (ValueError, KeyError)` ; une
  `source_freq='foo'` fournie par l'appelant lève `ValueError` dans `normalize_frequency`, interceptée :
  le garde-fou saute et les périodes incomplètes sont agrégées. Pour la même entrée, `method='all'` lève
  `ValueError: Unsupported frequency: foo`.
- **Attendu** : `ValueError`, comme `method='all'` : un paramètre explicite invalide n'est pas une
  fréquence « indétectable ». Le repli silencieux reste légitime quand `source_freq` est `None` et la
  détection impossible (comportement épinglé par `TestCoverageGuardsWithoutSourceFrequency`).
- **Reproduction** :
  ```python
  c = FrequencyConverter()
  m = pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2024-01-31', periods=3, freq='ME'))
  c.aggregate_to_lower_frequency(m, 'QE', 'sum', full_periods_only=True, source_freq='foo')  # [6.0]
  c.aggregate_to_lower_frequency(m > 0, 'QE', 'all', source_freq='foo')                      # ValueError
  ```
- **Test** : `tests/unit/utils/frequency/converter/test_aggregation.py::TestCoverageGuardsWithoutSourceFrequency::test_invalid_source_frequency_raises_for_full_periods_only`
- **Correctif** : une `source_freq` fournie est validée (`normalize_frequency`) avant l'agrégation ; seul l'échec de la **détection** (fréquence ni fournie ni détectable) fait sauter les gardes, dans `_expected_subperiod_counts`, pour `full_periods_only` comme pour `'all'`.
- **Statut** : corrigée

### ANO-UTILS-060 — Panel Series + dictionnaire d'entités incomplet : `ValueError` au lieu de laisser l'entité inchangée
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter._resolve_panel_target`
  (`tsforecast/panel/utils.py::get_entity_target_frequency`)
- **Sévérité** : mineure
- **Observé** : `convert_frequency(panel['x'], {('A',): 'QE'})` lève `ValueError: No target frequency found
  for entity ('B',)` ; le même dictionnaire sur le panel DataFrame laisse l'entité `B` inchangée
  (`_convert_panel_frequency` : « entité sans aucune colonne ciblée : conservation telle quelle »).
- **Attendu** : cohérence Series / DataFrame : une entité absente du dictionnaire est rendue inchangée.
- **Reproduction** :
  ```python
  idx = pd.MultiIndex.from_product([['A', 'B'], pd.date_range('2024-01-31', periods=6, freq='ME')])
  x = pd.Series(np.arange(12.0), index=idx)
  FrequencyConverter().convert_frequency(x, {('A',): 'QE'}, method='sum')  # ValueError
  ```
- **Test** : `tests/unit/utils/frequency/converter/test_panel.py::TestPanelSeries::test_untargeted_entity_is_unchanged`
- **Correctif** : `_resolve_panel_target` résout une Series de panel comme une colonne unique nommée d'après la Series (`resolve_entity_column_frequencies`) : entité non ciblée → rendue inchangée ; une clé de colonne égale au nom de la Series cible toutes les entités, comme pour un DataFrame.
- **Statut** : corrigée

### ANO-UTILS-061 — Méthode d'interpolation inconnue : message « Unsupported aggregation method »
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter.interpolate_to_higher_frequency`
- **Sévérité** : cosmétique
- **Observé** : `ValueError: Unsupported aggregation method: mean, should be in {...}` pour une
  **interpolation** — message trompeur, d'autant que `method='mean'` est la valeur par défaut de
  `convert_frequency` : tout sur-échantillonnage sans `method` explicite le produit.
- **Attendu** : « Unsupported interpolation method: mean, should be in {...} » (et, idéalement, rappeler
  que `method` doit être une méthode d'interpolation pour un sur-échantillonnage).
- **Reproduction** :
  ```python
  q = pd.Series([1.0, 2.0], index=pd.date_range('2021-03-31', periods=2, freq='QE'))
  FrequencyConverter().convert_frequency(q, 'ME')  # ValueError: Unsupported aggregation method: mean, …
  ```
- **Test** : `tests/unit/utils/frequency/converter/test_interpolation.py::TestInterpolationMethods::test_unsupported_method_message_names_interpolation`
- **Correctif** : « Unsupported interpolation method: … » ; la méthode est validée avant tout calcul.
- **Statut** : corrigée

### ANO-UTILS-062 — Docstrings de `FrequencyConverter` ≠ comportement
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter.aggregate_to_lower_frequency`,
  `::interpolate_to_higher_frequency`, `::convert_frequency`, `::_extend_index_for_upsampling`
- **Sévérité** : cosmétique
- **Observé** : `pytest --doctest-modules tsforecast/utils/frequency/converter.py` : 3 échecs sur 16.
  - exemples d'`aggregate_to_lower_frequency` (`'monthly'`) et d'`interpolate_to_higher_frequency`
    (`'daily'`) : libellés refusés (`Invalid target frequency 'monthly'`) — ces deux méthodes n'acceptent
    que des offsets pandas depuis `013d929` (« On ne normalise plus la fréquence pour préserver la
    position »), changement délibéré ; l'exemple d'interpolation utilise en outre `freq='M'` (déprécié) ;
  - exemple de `_extend_index_for_upsampling` : appel d'une fonction libre inexistante ;
  - `aggregate_to_lower_frequency` : `target_freq` documenté sans préciser « offset pandas » ; rien ne dit
    que les gardes de couverture (`full_periods_only`, `method='all'`) sont **sautées** quand la fréquence
    source n'est ni fournie ni détectable (index irrégulier) — seule la docstring privée
    `_require_full_subperiod_coverage` le dit ;
  - `convert_frequency` : `Raises` sans `NotImplementedError` (fréquence multipliée avec
    `full_periods_only` / `'all'`), ni la `ValueError` d'un sur-échantillonnage laissé à `method='mean'`
    (valeur par défaut, voir ANO-UTILS-061).
- **Attendu** : docstrings alignées sur le code (le code est juste sur ces points).
- **Test** : le comportement est épinglé par `tests/unit/utils/frequency/converter/test_aggregation.py::TestAggregationEdgeCases::test_user_label_is_not_an_offset`,
  `::TestCoverageGuardsWithoutSourceFrequency`, `tests/unit/utils/frequency/converter/test_conversion.py::TestConversionDirection::test_default_method_only_fits_downsampling`
- **Correctif** : docstrings réalignées : exemples en offsets pandas (`'ME'`, `'D'`), `Raises` complets (données vides, `NotImplementedError`, `limit`), gardes sautées sans fréquence source, `method` d'agrégation par défaut, nouveau contrat d'index de `convert_frequency`, source multipliée refusée par `anchor_fraction` (la docstring la disait « traitée comme sa base »). `pytest --doctest-modules tsforecast/utils/frequency/converter.py` : 20 passés.
- **Statut** : corrigée

### ANO-UTILS-063 — `converter.py` : branches inatteignables (code mort)
- **Type** : [CODE] comportement (nettoyage, pas un bogue)
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter`
- **Sévérité** : cosmétique
- **Observé** : seules lignes non couvertes après U8 (couverture 95 %, lignes et branches), toutes
  inatteignables par l'API publique :
  - `convert_frequency` : `target_freq` dict sur une Series simple (rejeté par la validation ; une Series
    de panel reçoit une chaîne par entité) ; « Data must be a pandas Series or DataFrame » (rejeté par
    `validate_temporal_data`) ;
  - `_validate_conversion_params` : les deux `Invalid position` — `parse_frequency` ne rend que `S`, `E`
    ou `None` (`([SE])?`), toutes valides ;
  - `_resolve_limit_direction` : `return None` et le test `if position is not None` (position toujours
    résolue en `S` / `E`) ; donc `resolved_limit_direction is None` jamais vrai ;
  - `_resolve_interpolation_limit` et `_extend_index_for_upsampling` : `except (ValueError, KeyError)`
    autour du facteur de durée — les fréquences sont déjà normalisées hors du `try` et
    `get_duration_conversion_factor` est total sur les codes normalisés (vérifié sur 13 × 13 codes,
    multiplicateurs compris) ;
  - `_apply_grouped_conversions` : branche `converted` Series (`data[columns]` est toujours un DataFrame) ;
  - `_align_mixed_frequency_columns` : `converted_columns` vide (l'appelant retourne avant), valeurs
    DataFrame (toujours des Series), branche `MultiIndex` (identique à l'autre ; les panels sont
    découpés par entité en amont), colonne de base déjà convertie ;
  - `_upsample` / `_downsample` : branches panel `groupby(...).apply` — jamais atteintes (panels découpés
    en amont) et fausses si elles l'étaient : `droplevel` + `group_keys=False` perdent le niveau entité
    (dates dupliquées sans entité).
- **Attendu** : suppression du code mort, sans changement de comportement.
- **Test** : couverture de `converter.py` par `tests/unit/utils/frequency/converter/` (lignes restantes :
  257, 315, 986→988, 1202-1203, 1243, 1282, 1325, 1622, 1679, 1687, 1696, 1705→1704, 1713, 1760-1761,
  1799-1800, 1869-1872 au commit `f61ed61`).
- **Correctif** : code mort supprimé : branches listées, `_upsample` / `_downsample` (appels directs aux méthodes publiques), `_align_mixed_frequency_columns` (remplacée par `_fill_union_gaps`), `try / except` larges de `_reanchor_index_to_target`, `_shift_index_to_anchor_fraction` et `_extend_index_for_upsampling` (base semi-mensuelle traitée explicitement), import `validate_position`. `converter.py` : 100 % de couverture, lignes et branches.
- **Statut** : corrigée

### ANO-UTILS-064 — Fréquence *Tick* multipliée sans position (`'2D'`) lue en fin de bloc
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter._extend_index_for_upsampling`
- **Sévérité** : mineure
- **Observé** : depuis `469d37f`, un horodatage multiplié couvre un bloc de n périodes « à partir de lui
  en position début, jusqu'à lui en position fin » ; sans position, la convention du paquet `'E'`
  s'applique. Une source `'2D'` (01/01 … 19/01/2024) sur-échantillonnée en `'D'` donne une grille du
  **31/12/2023** (avant la première observation, comblé vers l'arrière) au 19/01, et ampute le second jour
  du dernier bloc. Origine : échec hérité `test_positions.py::TestExtendIndexForUpsampling::test_unsupported_frequency_pair_returns_original`
  (en XPASS depuis `469d37f`), réécrit en U8.
- **Attendu** : **arbitrage de l'auteur (2026-09-30)** : une fréquence *Tick* multipliée (`D`, `h`,
  `min`, `s`, …) marque le **début** de son bloc, comme les étiquettes de `pandas.resample` ; grille
  du 01/01 au 20/01/2024. `W`, `SM` et M / Q / Y sans position restent en fin de période.
- **Reproduction** :
  ```python
  s = pd.Series(np.arange(10.0), index=pd.date_range('2024-01-01', periods=10, freq='2D'))
  FrequencyConverter().convert_frequency(s, 'D', method='linear').index[[0, -1]]
  # DatetimeIndex(['2023-12-31', '2024-01-19'])
  ```
- **Test** : `tests/unit/utils/frequency/converter/test_interpolation.py::TestMultipliedPositionlessSource::test_daily_grid_covers_the_blocks`
- **Correctif** : `_extend_index_for_upsampling` lit une source sans position à base jour ou infra-journalière (`D`, `B`, `h`, `min`, `s`, `ms`, `us`, `ns`) en début de bloc : `'2D'` → `'D'` du 01/01 au 20/01/2024. `B` suit les jours (étiquette à gauche de `resample('2B')`).
- **Statut** : corrigée

### ANO-UTILS-065 — `method='sum'` : une période sans aucune observation vaut 0.0
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter.aggregate_to_lower_frequency`
- **Sévérité** : mineure
- **Observé** : `resample(...).sum()` (pandas, `min_count=0`) rend `0.0` pour une période sans
  observation, indiscernable d'un vrai zéro ; `mean`, `median`, `min`, `max`, `first`, `last`, `std`
  rendent NaN. Sur `irregular_index_timeseries` → `'YS'`, `production_industrielle` vaut 0.0 de 2015 à
  2018 (série démarrant en 2019). `full_periods_only=True` masque ces périodes.
- **Attendu** : **arbitrage de l'auteur (2026-09-30)** : NaN, comme `mean` (« pas d'observation → pas de
  valeur ») ; `count` reste 0.
- **Reproduction** :
  ```python
  m = pd.Series([1.0, 2.0, 3.0, np.nan, np.nan, np.nan],
                index=pd.date_range('2024-01-31', periods=6, freq='ME'))
  FrequencyConverter().aggregate_to_lower_frequency(m, 'QE', method='sum')  # [6.0, 0.0]
  ```
- **Test** : `tests/unit/utils/frequency/converter/test_aggregation.py::TestPeriodsWithoutObservation::test_sum_gives_nan`,
  `tests/unit/utils/frequency/converter/test_realistic_datasets.py::TestIrregularIndexTimeseries::test_years_without_observation_do_not_sum_to_zero`
- **Correctif** : `resample(...).sum(min_count=1)`.
- **Statut** : corrigée

### ANO-UTILS-066 — `limit` : entier numpy, flottant ou chaîne quelconque ignorés en silence
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter._resolve_interpolation_limit`
- **Sévérité** : mineure
- **Observé** : seuls `None`, un `int` Python et `'default'` sont reconnus ; toute autre valeur —
  `np.int64(1)` (`isinstance(np.int64(1), int)` est faux), `1.0`, `'foo'` — est résolue en `None`, soit
  **aucune limite**, sans erreur. pandas refuserait ces valeurs (`limit` doit être un entier).
- **Attendu** : entiers numpy acceptés comme des `int` ; autre valeur → `ValueError`.
- **Reproduction** :
  ```python
  c = FrequencyConverter()
  q = pd.Series([10.0, 20.0], index=pd.date_range('2021-03-31', periods=2, freq='QE'))
  c.interpolate_to_higher_frequency(q, 'ME', limit=np.int64(1))  # janvier comblé : aucune limite
  c.interpolate_to_higher_frequency(q, 'ME', limit='foo')        # idem, sans erreur
  ```
- **Test** : `tests/unit/utils/frequency/converter/test_interpolation.py::TestInterpolationLimit::test_numpy_integer_limit_is_honoured`,
  `::test_invalid_limit_raises`
- **Correctif** : entiers numpy acceptés (`numbers.Integral`, booléens exclus) ; toute autre valeur que `None`, `'default'` ou un entier → `ValueError: Invalid limit …`.
- **Statut** : corrigée

### ANO-UTILS-067 — Sources différentes vers une même cible : l'index de sortie est celui de la première colonne
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter._apply_grouped_conversions`
  (« chemin simple »)
- **Sévérité** : mineure
- **Observé** : quand toutes les colonnes vont vers la même cible sans colonne préservée, le résultat est
  construit sur l'index de la **première** colonne convertie ; les dates propres aux autres groupes
  (sources différentes → extensions différentes) sont perdues. Sur une grille mensuelle de janvier 2020
  à juin 2021, `q` (trimestrielle, extension jusqu'à juin 2021) et `y` (annuelle, jusqu'à décembre 2021)
  vers `'MS'` : 18 lignes pour `[['q', 'y']]`, 24 pour `[['y', 'q']]`. Le chemin « fréquences mixtes »
  prend, lui, l'union des index.
- **Attendu** : résultat indépendant de l'ordre des colonnes (permuter les colonnes permute la sortie,
  rien d'autre) — l'union des index cibles, comme le chemin mixte.
- **Reproduction** :
  ```python
  c = FrequencyConverter()
  grid = pd.date_range('2020-01-01', '2021-06-01', freq='MS')
  df = pd.DataFrame({'q': np.nan, 'y': np.nan}, index=grid)
  df.loc[grid.month.isin([1, 4, 7, 10]), 'q'] = [1.0, 2, 3, 4, 5, 6]
  df.loc[['2020-01-01', '2021-01-01'], 'y'] = [100.0, 112.0]
  len(c.convert_frequency(df[['q', 'y']], 'MS', method='linear'))  # 18
  len(c.convert_frequency(df[['y', 'q']], 'MS', method='linear'))  # 24
  ```
- **Test** : `tests/unit/utils/frequency/converter/test_conversion.py::TestDataFrameOutputContract::test_result_does_not_depend_on_column_order`
- **Correctif** : index de sortie = union des index cibles, sans comblement tant qu'une seule fréquence cible est en jeu et qu'aucune colonne conservée n'est observée (`_apply_grouped_conversions`).
- **Statut** : corrigée

### ANO-UTILS-068 — Sur-échantillonnage vers une cible infra-journalière : `cannot reindex on an axis with duplicate labels`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter._reanchor_index_to_target`
- **Sévérité** : majeure
- **Observé** : le ré-ancrage appliquait `to_timestamp(how='end').normalize()` : toutes les heures d'une
  journée revenaient à minuit. Tout sur-échantillonnage vers une cible infra-journalière (6 h → h, h → min)
  levait `ValueError: cannot reindex on an axis with duplicate labels`. Décelé en corrigeant ANO-UTILS-064.
- **Attendu** : grille infra-journalière, heure de chaque observation conservée.
- **Reproduction** :
  ```python
  s = pd.Series([0.0, 6.0, 12.0, 18.0], index=pd.date_range('2024-01-01', periods=4, freq='6h'))
  FrequencyConverter().interpolate_to_higher_frequency(s, 'h')  # ValueError (duplicate labels)
  ```
- **Correctif** : les cibles sans position à base jour ou infra-journalière sont ré-ancrées au début de
  leur période, sans remise à minuit (`_BLOCK_START_BASES`).
- **Test** : `tests/unit/utils/frequency/converter/test_interpolation.py::TestUpsamplingTargetGrid::test_sub_daily_target_keeps_the_time_of_day`,
  `::test_six_hours_block_starts_at_its_stamp`
- **Statut** : corrigée

### ANO-UTILS-069 — Détection sur un index décroissant : erreur cryptique ou tri silencieux selon le chemin
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/utils.py::detect_index_frequency`
- **Sévérité** : mineure
- **Observé** : sur un index régulier décroissant, `detect_index_frequency` passait `'-1QE-DEC'` (inférence
  pandas) au normaliseur : `ValueError: Unsupported frequency: -1QE-DEC`. Avec des trous, le repli
  heuristique triait l'index en silence, comme `FrequencyDetector.detect_time_series_frequency` (donc
  `detect_frequency`, `detect_dataset_frequency`). Cause de ANO-UTILS-057 côté convertisseur.
- **Attendu** : **arbitrage de l'auteur (2026-09-30)** : un index décroissant est trié comme un index
  désordonné (même règle que la validation des séries temporelles) — un premier correctif levant une
  erreur sur l'index décroissant a été écarté, l'asymétrie avec l'index désordonné n'étant pas logique.
- **Correctif** : `detect_index_frequency` trie un index non croissant avant l'inférence pandas ;
  `detect_time_series_frequency` triait déjà.
- **Test** : `tests/unit/utils/frequency/detector/test_index_and_offset.py::TestDetectIndexFrequency::test_decreasing_index_is_sorted`,
  `tests/unit/utils/frequency/detector/test_time_series.py::TestDetectTimeSeriesFrequencyUnsortedData::test_reversed_series`,
  `tests/unit/utils/frequency/detector/test_panel.py::TestDetectDatasetFrequencyTimeSeries::test_time_column[reversed]`
- **Statut** : corrigée

### ANO-UTILS-070 — Données vides ou jamais observées : comportements différents entre Series et DataFrame
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/utils/frequency/converter.py::FrequencyConverter.convert_frequency`,
  `::aggregate_to_lower_frequency`, `::interpolate_to_higher_frequency`
- **Sévérité** : mineure
- **Observé** : un panel vide était rendu inchangé par `convert_frequency`, une Series vide refusée par
  la détection (« Series has only 0 non-null observations ») ; `aggregate_to_lower_frequency` rendait une
  sortie vide. Sans aucune valeur observée, une Series levait une erreur et un DataFrame rendait zéro
  ligne ; avec des observations mais sans fréquence détectable, une Series levait une erreur et un
  DataFrame était rendu sans conversion. Relevé comme comportement surprenant au rapport U8.
- **Attendu** : **arbitrage de l'auteur (2026-09-30)** : un panel ou une série vide renvoient une erreur ;
  même comportement pour les Series et les DataFrame, le choix entre erreur et objet vide étant laissé au
  correctif. Règle retenue, identique pour Series, DataFrame et panels :
  1. aucune ligne → `ValueError` ;
  2. des lignes mais aucune valeur observée → résultat vide (aucune date à conserver) ; dans un panel,
     l'entité jamais observée disparaît — une erreur rendrait inconvertible tout panel à colonne
     structurellement absente pour une entité (`climat_affaires` / Italie) ;
  3. des observations sans fréquence détectable (Series, ou toutes les colonnes d'un DataFrame) →
     `ValueError`.
- **Correctif** : `_reject_empty` en tête des trois méthodes (après validation pour `convert_frequency`) :
  `ValueError: Cannot <convert|aggregate|interpolate> empty data: no row to …`. Series sans observation :
  `convert_frequency` et `interpolate_to_higher_frequency` rendent un objet vide du même type ;
  `_build_frequency_map` lève « Cannot detect current frequency of any column of the data » quand des
  valeurs sont observées sans qu'aucune colonne ait de fréquence détectable ;
  `_convert_panel_frequency` écarte les entités sans observation. `aggregate_to_lower_frequency` garde,
  sur des lignes jamais observées, ses périodes à `NaN` (une période par période couverte par les lignes).
- **Test** : `tests/unit/utils/frequency/converter/test_conversion.py::TestSeriesEdgeCases::test_empty_series_raises`,
  `::TestDataFrameConversion::test_empty_dataframe_raises`,
  `::TestDataFrameOutputContract::test_never_observed_frame_gives_no_row`,
  `::test_never_observed_series_gives_no_row`, `::test_frame_without_detectable_column_raises`,
  `tests/unit/utils/frequency/converter/test_panel.py::TestPanelStringTarget::test_empty_panel_raises`,
  `::test_never_observed_entity_is_left_out`, `::test_never_observed_panel_gives_no_row`,
  `tests/unit/utils/frequency/converter/test_aggregation.py::TestAggregationEdgeCases::test_empty_series_raises`,
  `tests/unit/utils/frequency/converter/test_interpolation.py::TestEmptyOrUnobservedInput`
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
- **Test** : `tests/unit/delays/test_calculator.py::TestTargetUnit::test_mixed_input_units_are_refused_without_target_unit`,
  `::test_mixed_input_units_are_accepted_when_aggregated_per_couple`, `::test_day_name_and_code_are_the_same_unit`,
  `::test_mixed_input_units_are_converted_with_a_target_unit`.
- **Correctif** : décision de l'auteur (2026-10-05) : `ValueError` quand les lignes agrégées ensemble (par indicateur, ou par couple avec `aggregate_by_panel`) n'ont pas la même unité, `'day'` et `'D'` étant une même unité ; le message nomme les groupes et demande une `unit` commune.
- **Statut** : corrigée

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
- **Test** : `tests/unit/delays/test_calculator.py::TestTargetFrequencyDictionary::test_uncovered_indicator_is_named`,
  `::test_uncovered_couple_is_named`.
- **Correctif** : le dictionnaire est résolu ligne par ligne (`_resolve_target_frequencies`) ; les clés non couvertes sont nommées dans la `ValueError` (`'frequency' does not give a target frequency for: ['CPI']`). Le dictionnaire accepte désormais aussi des clés `(entité, ..., indicateur)` (décision de l'auteur, 2026-10-05 : une fréquence cible par couple), voir `TestTargetFrequencyDictionary`.
- **Statut** : corrigée


### ANO-DELAYS-004 — Délai en microsecondes majoré de 1 µs au-delà d'environ 800 jours
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/data_manager.py::_calculate_publication_delays` (via `compare_and_detect_delays(delay_unit='us')`)
- **Sévérité** : mineure
- **Observé** : le délai en microsecondes est `np.ceil(timedelta.total_seconds() * 1_000_000)`. Le produit flottant de
  `total_seconds()` (≈ 1,5e8 s, soit 9 chiffres entiers + 6 décimales : à la limite de la précision d'un `float64`) par
  1e6 tombe parfois juste au-dessus de l'entier exact, et `ceil` ajoute alors 1 µs. Sur 300 dates de téléchargement
  tirées au hasard entre 60 et 2 000 jours après la période, 32 sont majorées de 1 µs. Aucune erreur observée sous
  ≈ 800 jours (l'erreur d'arrondi y reste inférieure à 1e-2 µs) ; en jours et en secondes la formule est exacte.
- **Attendu** : l'entier exact de microsecondes écoulées, `(téléchargement - début) // 1 µs` : le `ceil` n'a d'objet que
  pour les fractions de l'unité (docstring : « ceil-rounded »), pas pour une erreur de représentation.
- **Reproduction** :
  ```python
  from datetime import datetime, timedelta
  import pandas as pd
  from tsforecast.delays.data_manager import compare_and_detect_delays
  data = pd.DataFrame({'PIB': [1.0, 2.0, 3.0, 4.0]}, index=pd.date_range('2023-01-01', periods=4, freq='MS'))
  dl = datetime(2027, 11, 26, 17, 18, 56, 578903)
  compare_and_detect_delays(data, None, dl, delay_unit='us')['delay'].iloc[0]   # 146942336578904.0
  (dl - datetime(2023, 4, 1)) // timedelta(microseconds=1)                      # 146942336578903
  ```
- **Test** : `tests/unit/delays/test_data_manager.py::TestReferencePointAndUnit::test_microsecond_delay_is_exact`
- **Correctif** : le délai est calculé en entiers de nanosecondes (`as_unit('ns')`, division entière par excès
  `-(-n // u)`), avec la durée de chaque unité en nanosecondes (`get_duration_nanoseconds`, alimentée par `utils/_constants.py` ;
  anciennement la table locale `_DELAY_UNITS`) : plus aucun produit flottant, arrondi
  au supérieur uniquement pour les fractions de l'unité.
- **Statut** : corrigée

### ANO-DELAYS-005 — Jeux vides ou sans observation : `TypeError` / `IndexError` / `ValueError` cryptiques
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/data_manager.py::compare_and_detect_delays`
  (`_identify_new_observations` sans `existing_data`, puis `_calculate_publication_delays`)
- **Sévérité** : mineure
- **Observé** : trois familles de jeux dégénérés, tous rejetés par une exception interne sans rapport avec le problème,
  alors que le chemin « comparaison » rend un résultat vide pour les mêmes données :
  1. série **sans aucune valeur observée**, `existing_data=None` : `TypeError: Addition/subtraction of integers and
     integer-arrays with Timestamp is no longer supported` (la liste de résultats vide donne un index d'objets, la
     soustraction `download_date - période` échoue). Panel dans le même cas : `ValueError: Length of new names must be
     1, got 2`. Avec `existing_data` fourni, les mêmes données donnent un `DataFrame` vide (colonnes de sortie
     présentes) ;
  2. jeu **sans aucune ligne** (série ou panel) : mêmes exceptions (`TypeError`, `ValueError: Length of new names…`) ;
  3. jeu **sans aucune colonne** (dates seules), avec ou sans `existing_data` : `IndexError: index 0 is out of bounds
     for axis 0 with size 0` (`freq_df.index[0]` sur une carte de fréquences vide).
- **Attendu** : règle d'**ANO-UTILS-070** (arbitrage de l'auteur du 2026-09-30 pour la conversion de fréquence, reprise
  ici par analogie, **à confirmer**) : (1) aucune ligne → `ValueError` explicite (le message contient « empty ») ; (2) des
  lignes mais aucune valeur observée, ou aucune colonne → résultat vide, avec les colonnes de sortie et les niveaux
  d'index habituels, comme le fait déjà le chemin avec `existing_data`.
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.data_manager import compare_and_detect_delays
  idx = pd.date_range('2023-01-01', periods=5, freq='MS')
  compare_and_detect_delays(pd.DataFrame({'PIB': [np.nan] * 5}, index=idx), None, '2023-06-15')   # TypeError
  compare_and_detect_delays(pd.DataFrame(index=idx), None, '2023-06-15')                          # IndexError
  compare_and_detect_delays(pd.DataFrame({'PIB': []}, index=pd.DatetimeIndex([])), None, '2023-06-15')  # TypeError
  ```
- **Test** : `tests/unit/delays/test_data_manager.py::TestEmptyAndDegenerateInputs::test_all_null_data_on_first_download_gives_an_empty_frame`,
  `::test_all_null_panel_on_first_download_gives_an_empty_frame`, `::test_dataset_without_rows_raises_a_clear_error`,
  `::test_panel_without_rows_raises_a_clear_error`, `::test_frame_without_columns_gives_an_empty_frame`
- **Correctif** : décision de l'auteur (2026-10-04) : règle d'ANO-UTILS-070. `compare_and_detect_delays` lève
  `ValueError: Cannot detect publication delays on empty data: new_data has no row` pour un `new_data` sans ligne
  (avec ou sans `existing_data` ; un `existing_data` vide signifie « rien de connu » : tout est nouveau) ; des lignes sans
  valeur observée, ou aucune colonne, donnent un résultat vide. Tous les chemins vides (rien de changé, rien
  d'observé, aucune colonne) passent par `_empty_publication_delays` : mêmes colonnes, mêmes types, mêmes niveaux
  d'index (entités, puis `column`).
- **Statut** : corrigée

### ANO-DELAYS-006 — Fréquence indétectable : message sans le nom de la colonne ; une entité fait échouer tout le panel
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/data_manager.py::_calculate_publication_delays`
- **Sévérité** : à arbitrer
- **Observé** : depuis le commit `f61ed61` (ANO-UTILS-042 : un couple indétectable est associé à `None`, non plus une
  exception), une observation dont la fréquence est indétectable (colonne observée une seule fois, ou entité à une seule
  observation dans un panel) atteint `get_period_boundaries(frequency=None)` et lève
  `ValueError: Frequency must be a string, got <class 'NoneType'>` : le message ne nomme ni la colonne ni l'entité. Dans un
  panel, un seul couple de ce type fait échouer le calcul des délais de **tous** les autres. L'ancien message du
  détecteur (« Series has only 1 non-null observations, minimum required is 2 », attendu par le test historique
  `test_single_observation`) a disparu avec ce commit : changement délibéré, le test historique est de catégorie (a).
- **Attendu** : (1) une `ValueError` nommant les couples sans fréquence détectable — attente testée ; (2) à arbitrer :
  lever pour tout le panel (comportement actuel, épinglé), ou écarter les couples indétectables avec un avertissement,
  ou les rendre avec `frequency=None` et des bornes / un délai `NaN`.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.delays.data_manager import compare_and_detect_delays
  data = pd.DataFrame({'PIB': [100.0]}, index=pd.date_range('2023-01-01', periods=1, freq='MS'))
  compare_and_detect_delays(data, None, '2023-02-15')   # ValueError: Frequency must be a string, got <class 'NoneType'>
  ```
- **Test** : `tests/unit/delays/test_data_manager.py::TestFrequencyAndPeriodBoundaries::test_undetectable_frequency_gives_a_row_without_period_or_delay`,
  `::test_undetectable_frequency_warning_names_the_column`, `::test_one_undetectable_entity_does_not_spoil_the_panel`,
  `::test_undetectable_frequency_in_comparison`
- **Correctif** : décision de l'auteur (2026-10-04) : le couple indétectable est **renvoyé** avec `frequency=None`,
  `period_start` / `period_end` à `NaT` et `delay` à `NaN` (l'unité reste renseignée), et un `UserWarning` nomme les
  couples concernés (`The frequency could not be detected for [('B', 'PIB')]: ...`). Les autres couples sont calculés.
  Conséquence en aval : `calculate_applicable_delay` rejette une ligne sans fréquence (voir ANO-DELAYS-011).
- **Statut** : corrigée

### ANO-DELAYS-007 — Une colonne de données nommée `has_changes` fait échouer la comparaison
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/data_manager.py::_identify_new_observations` (chemin `existing_data` fourni)
- **Sévérité** : mineure
- **Observé** : `pd.melt(changes_mask, var_name="column", value_name="has_changes", ...)` lève
  `ValueError: value_name (has_changes) cannot match an element in the DataFrame columns.` quand une colonne de données
  s'appelle `has_changes`. Les autres noms de colonnes de sortie (`column`, `delay`, `frequency`, `observation_date`,
  `unit`, `download_date`, `period_start`, `reference_point`) et `index` passent, et le chemin `existing_data=None`
  accepte aussi `has_changes` : l'échec est propre au chemin de comparaison.
- **Attendu** : toute colonne de données se compare comme les autres, quel que soit son nom (CLAUDE.md : noms de colonnes
  non standards).
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.data_manager import compare_and_detect_delays
  idx = pd.date_range('2023-01-01', periods=4, freq='MS')
  new = pd.DataFrame({'has_changes': [1.0, 2.0, 3.0, 4.0]}, index=idx)
  old = new.copy(); old.iloc[3] = np.nan
  compare_and_detect_delays(new, old, '2023-06-15')   # ValueError: value_name (has_changes) cannot match ...
  ```
- **Test** : `tests/unit/delays/test_data_manager.py::TestInputLayout::test_column_named_has_changes_in_comparison`
- **Correctif** : `pd.melt` est remplacé par une construction par colonne (`_flag_observations`) : aucun nom de
  colonne de données ne peut plus entrer en collision avec `has_changes`.
- **Statut** : corrigée

### ANO-DELAYS-008 — `FutureWarning` pandas à chaque appel sur un panel à deux niveaux (sans `existing_data`)
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/data_manager.py::_identify_new_observations`
- **Sévérité** : cosmétique
- **Observé** : `new_data.groupby(level=[0])` (liste d'un seul niveau, `panel_levels = list(range(nlevels - 1))`) émet
  `FutureWarning: Creating a Groupby object with a length-1 list-like level parameter will yield indexes as tuples in a
  future version` une fois **par colonne** (21 occurrences sur la suite de ce fichier de tests). Sans conséquence
  aujourd'hui (la clé du groupe n'est pas utilisée), mais le groupement est en outre refait pour chaque colonne.
- **Attendu** : aucun avertissement ; un seul `groupby` hors de la boucle sur les colonnes.
- **Reproduction** :
  ```python
  import warnings, pandas as pd
  from tsforecast.delays.data_manager import compare_and_detect_delays
  idx = pd.MultiIndex.from_product([['A', 'B'], pd.date_range('2023-01-01', periods=4, freq='MS')], names=['country', 'date'])
  panel = pd.DataFrame({'PIB': range(8), 'CPI': range(8)}, index=idx)
  warnings.simplefilter('error', FutureWarning)
  compare_and_detect_delays(panel, None, '2023-06-15')   # FutureWarning
  ```
- **Test** : `tests/unit/delays/test_data_manager.py::TestWarnings::test_panel_call_is_silent`
- **Correctif** : le premier téléchargement est réécrit sans boucle de groupes : `dropna()` puis
  `groupby(level=...).tail(1)` par colonne, avec un niveau scalaire quand il n'y en a qu'un ; l'index d'origine
  (noms de niveaux compris) est conservé sans reconstruction par tuples.
- **Statut** : corrigée

### ANO-DELAYS-009 — Docstring de `compare_and_detect_delays` et documentation : sortie, noms de colonnes, valeurs de `frequency`
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/delays/data_manager.py::compare_and_detect_delays` ; `docs/tutorials/publication_delays.md`
  §4.1.2 ; `docs/concepts/publication_delays.md` §« Inférer les délais »
- **Sévérité** : cosmétique
- **Observé** : (1) la colonne `has_changes` (toujours `True`) figure dans la sortie mais pas dans la docstring ; (2) la
  docstring et le tutoriel présentent `column` comme une colonne : c'est le dernier niveau de l'**index** (niveaux
  d'entité éventuels, puis `column`), `observation_date` étant, elle, une vraie colonne ; le tutoriel annonce un
  `DataFrame` « indexé par les dates » ; (3) le tutoriel nomme encore la colonne `release_delay` (renommée `delay` par
  `4b3d3bc`) ; (4) `frequency` vaut `'monthly'` / `'quarterly'` / `'annual'` / `'daily'` / `'weekly'` / `'hourly'` (littéraux
  de `to_literal`), non `'M'` / `'Q'` / `'A'` comme le dit le tutoriel ; (5) `period_end` est la borne **exclusive**
  (1er jour de la période suivante : `2023-05-01` pour avril) ; (6) `docs/concepts/publication_delays.md` évoque
  « une seule [extraction] avec une colonne de date de téléchargement » : aucune telle colonne n'est lue, le mode à
  un seul jeu est `existing_data=None` (dernière observation par variable) ; (7) `Raises` omet le `TypeError`
  d'un `download_date` aware avec des données naïves (et l'inverse), et le `ValueError` de `resolve_date` pour un type
  inattendu (`datetime.date`) ; (8) `download_date` est annoncé `Union[str, datetime]` mais sa valeur par défaut est `None`.
- **Correctif** : docstring et pages à réécrire d'après le comportement observé (le test suit le code).
- **Test** : `tests/unit/delays/test_data_manager.py::TestOutputContract` (colonnes, ordre, index),
  `::TestFrequencyAndPeriodBoundaries` (littéraux de fréquence, bornes), `::TestDownloadDate`
- **Correctif** : docstrings de `compare_and_detect_delays` et `_calculate_publication_delays` réécrites d'après le
  comportement (colonnes et ordre, index, littéraux de fréquence, borne de fin exclusive, délai des révisions, fuseaux,
  `Raises` / `Warns`, exemple exécutable) ; `docs/tutorials/publication_delays.md` §4.1.1-4.1.2 et
  `docs/concepts/publication_delays.md` corrigés.
- **Statut** : corrigée

### ANO-DELAYS-010 — `_calculate_publication_delays` : branches inatteignables (code mort)
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/data_manager.py::_calculate_publication_delays` (branches `elif frequency_map_raw is not None`
  / `else` de la conversion en littéral ; `if freq_df.index.nlevels == 1 and isinstance(freq_df.index[0], tuple)` ;
  `else` de la construction de `freq_df` à partir d'une fréquence unique)
- **Sévérité** : cosmétique
- **Observé** : `detect_frequency(new_data)` reçoit toujours un `DataFrame` (une `Series` lève `AttributeError: 'Series'
  object has no attribute 'columns'` dès `_identify_new_observations`) et renvoie alors toujours un `dict`. Les
  traitements d'une fréquence unique (`str`) ou `None` ne sont donc jamais exécutés ; les clés de panel sont
  aplaties, jamais des tuples sur un index à un niveau (`pd.Series(dict)` construit lui-même le `MultiIndex`). Ces
  lignes (308-311, 319, 322) restent non couvertes (couverture du module : 93 %).
- **Attendu** : supprimer ces branches, ou accepter réellement une `Series` (le type annoncé est `pd.DataFrame`) ;
  à défaut, un `TypeError` explicite plutôt que l'`AttributeError`.
- **Test** : `tests/unit/delays/test_data_manager.py::TestEmptyAndDegenerateInputs::test_series_instead_of_frame_raises` ; couverture du module : 100 %.
- **Correctif** : branches mortes supprimées (`detect_frequency` d'un `DataFrame` est toujours un `dict`) ; un
  `new_data` ou `existing_data` qui n'est pas un `DataFrame` lève `TypeError: new_data must be a pandas DataFrame, got Series`.
- **Statut** : corrigée

### ANO-DELAYS-011 — `calculate_applicable_delay` rejette les lignes sans fréquence renvoyées par `compare_and_detect_delays`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/calculator.py::calculate_applicable_delay`
- **Sévérité** : mineure
- **Observé** : depuis ANO-DELAYS-006, un couple dont la fréquence est indétectable est renvoyé avec `frequency=None` et
  un délai `NaN`. Passé à `calculate_applicable_delay`, il lève `ValueError: Frequency must be a string, got <class
  'NoneType'>` (sans nommer le couple) : la chaîne `compare_and_detect_delays` → `calculate_applicable_delay` échoue
  dès qu'un couple est indétectable.
- **Attendu** : à traiter au prompt D2 : écarter ces lignes (délai inconnu) avec un avertissement nommant les couples,
  ou les conserver avec un délai `NaN` ; dans les deux cas sans exception cryptique.
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays import compare_and_detect_delays, calculate_applicable_delay
  idx = pd.MultiIndex.from_product([['A', 'B'], pd.date_range('2023-01-01', periods=4, freq='MS')], names=['country', 'date'])
  panel = pd.DataFrame({'PIB': [1., 2, 3, 4, 1, np.nan, np.nan, np.nan]}, index=idx)
  calculate_applicable_delay(compare_and_detect_delays(panel, None, '2023-06-15'), 'start', 'M')   # ValueError
  ```
  Le même échec survient pour une ligne dont le délai est `NaN` alors que sa fréquence est valide
  (`ValueError: cannot convert float NaN to integer`, levée par `pd.Timedelta(seconds=nan)` dans
  `_calculate_converted_delay`) : le chemin « délai inconnu » n'est géré nulle part.
- **Test** : `tests/unit/delays/test_calculator.py::TestContractWithDataManager::test_undetectable_frequency_does_not_break_the_chain`,
  `::test_undetectable_couple_is_kept_with_a_nan_delay`, `::TestDelayMagnitudes::test_unknown_delay_row_is_kept_with_a_nan_delay`,
  `::test_unknown_delay_row_is_ignored_by_the_aggregation`, `::test_group_of_unknown_delays_has_a_nan_delay`,
  `::test_unknown_delay_survives_the_unit_conversion_and_the_reference_change`.
- **Correctif** : décision de l'auteur (2026-10-05) : la ligne est **conservée avec un délai `NaN`** (délai `NaN` ou fréquence source `None`). Elle n'est pas comptée dans `n_observations` ; un groupe sans délai connu a un délai `NaN` (même pour `sum`) et `n_observations=0`. Aucun avertissement n'est ajouté (celui de `compare_and_detect_delays` suffit).
- **Statut** : corrigée

### ANO-DELAYS-012 — Délai en microsecondes décalé de 1 µs par le passage par des secondes flottantes
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/calculator.py::_calculate_converted_delay` (via `calculate_applicable_delay` sur des
  lignes d'unité `'microsecond'`, sortie de `compare_and_detect_delays(delay_unit='us')`)
- **Sévérité** : mineure
- **Observé** : le délai d'entrée est converti en secondes **flottantes** (`convert_duration(..., 'us' -> 's', rounding=None)`),
  transformé en `pd.Timedelta(seconds=...)` (tronqué à la nanoseconde), puis le nouveau délai est reconverti
  par `total_seconds()` (flottant) et `convert_duration(..., 's' -> 'us', rounding='ceil')`. Sur 400 délais tirés au hasard
  sous 100 jours, **108 sortent diminués de 1 µs** (et 1 augmenté), y compris quand le point de référence ne change pas
  (`'end'` -> `'end'`, même fréquence : le délai devrait être restitué à l'identique) ; sur 800 à 2 000 jours, plus d'un
  cas sur deux. Aucune erreur en jours ni en secondes entières (200 tirages chacun). Même famille que ANO-DELAYS-004
  (`data_manager`), dans l'autre sens : la troncature à la nanoseconde de `pd.Timedelta(seconds=8183426.019069999)`
  rend `8183426019069` µs au lieu de `8183426019070`.
- **Attendu** : un calcul en entiers (nanosecondes, comme le correctif d'ANO-DELAYS-004) : le délai converti est exact, et
  restitué à l'identique quand ni la fréquence ni le point de référence ne changent.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.delays.calculator import calculate_applicable_delay
  ts = pd.Timestamp
  delays = pd.DataFrame({
      'observation_date': [ts('2023-12-15')], 'download_date': [ts('2024-05-15')], 'frequency': ['monthly'],
      'period_start': [ts('2023-12-01')], 'period_end': [ts('2024-01-01')], 'reference_point': ['end'],
      'delay': [8_183_426_019_070], 'unit': ['microsecond']}, index=pd.Index(['PIB'], name='indicator'))
  calculate_applicable_delay(delays, 'end', 'M')['delay'].iloc[0]   # 8183426019069.0
  ```
- **Test** : `tests/unit/delays/test_calculator.py::TestTargetUnit::test_microsecond_delay_is_exact` (trois cas)
- **Correctif** : délais reconstruits et reconvertis en nanosecondes **entières** (`_nanoseconds_per_unit`, `_to_nanoseconds` via `Fraction`, division entière par excès), y compris pour le changement d'unité (`_convert_delay_unit`) ; plus aucune seconde flottante. Vérifié sur 1 200 délais tirés au hasard (0 écart).
- **Statut** : corrigée

### ANO-DELAYS-013 — Jeu sans aucune ligne : `KeyError` cryptique
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/calculator.py::calculate_applicable_delay` (`_aggregate_delays`)
- **Sévérité** : mineure
- **Observé** : un `DataFrame` sans ligne mais avec les colonnes requises passe `_validate_columns` puis échoue avec
  `KeyError: "Column(s) ['converted_delay'] do not exist"` (`delays.apply(..., axis=1)` d'un frame vide ne crée aucune
  colonne). Même famille qu'ANO-DELAYS-005 côté `data_manager`.
- **Attendu** : soit un `DataFrame` vide avec les six colonnes de sortie, soit une `ValueError` explicite (« no row ») ; l'attente du
  test admet les deux.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.delays.calculator import calculate_applicable_delay
  cols = ['observation_date', 'download_date', 'frequency', 'period_start', 'period_end', 'reference_point', 'delay', 'unit']
  calculate_applicable_delay(pd.DataFrame(columns=cols, index=pd.Index([], name='indicator')), 'end', 'M')   # KeyError
  ```
- **Test** : `tests/unit/delays/test_calculator.py::TestDelayMagnitudes::test_empty_frame_raises_a_clear_error`
- **Correctif** : `ValueError: Cannot calculate applicable delays on empty data: publication_delays has no row`, comme `compare_and_detect_delays` (ANO-DELAYS-005).
- **Statut** : corrigée

### ANO-DELAYS-014 — Index sans nom : le niveau de l'indicateur est repéré par son nom
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/calculator.py::calculate_applicable_delay` (`indicator_level_name = delays.index.names[-1]`)
- **Sévérité** : mineure
- **Observé** : le dernier niveau de l'index est identifié par son **nom** puis passé à `groupby(level=[nom])`. Un index à un niveau
  sans nom (`pd.Index([...], name=None)`) lève `TypeError: '>' not supported between instances of 'NoneType' and 'int'` ; un
  `MultiIndex` dont tous les niveaux sont sans nom lève `ValueError: The name None occurs multiple times, use a level number`.
  Un `MultiIndex` dont seul le dernier niveau est sans nom fonctionne (le nom `None` est unique). La sortie de
  `compare_and_detect_delays` nomme toujours son dernier niveau `'column'`, mais la docstring décrit l'index par sa position
  (« the last level of its index is the indicator »).
- **Attendu** : le niveau est repéré par sa position (`-1`), les noms étant libres ; les noms existants sont conservés dans
  la sortie.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.delays.calculator import calculate_applicable_delay
  ts = pd.Timestamp
  delays = pd.DataFrame({
      'observation_date': [ts('2023-12-15')], 'download_date': [ts('2024-01-15')], 'frequency': ['monthly'],
      'period_start': [ts('2023-12-01')], 'period_end': [ts('2024-01-01')], 'reference_point': ['end'],
      'delay': [14], 'unit': ['day']}, index=['PIB'])                  # index sans nom
  calculate_applicable_delay(delays, 'end', 'M')                       # TypeError
  ```
- **Test** : `tests/unit/delays/test_calculator.py::TestIndicatorsAndPanel::test_unnamed_index_levels` (deux cas),
  `::test_unnamed_index_levels_stay_unnamed_in_the_panel_result`, `::test_duplicated_level_names_are_read_by_position`
- **Correctif** : les niveaux de l'index sont repérés par leur **position** (`nlevels - 1` pour l'indicateur, `range(nlevels)` pour le panel) ; les noms sont libres, absents ou dupliqués, et conservés dans la sortie.
- **Statut** : corrigée

### ANO-DELAYS-015 — `aggregation_method` inconnu : `AttributeError` de pandas
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/calculator.py::_aggregate_delays`
- **Sévérité** : cosmétique
- **Observé** : un nom de méthode inconnu (`aggregation_method='nope'`) lève `AttributeError: 'SeriesGroupBy' object has no
  attribute 'nope'` ; un objet non appelable (`3`) lève `TypeError: 'int' object is not callable`. Le `Raises` de la docstring
  ne mentionne ni l'un ni l'autre.
- **Attendu** : à arbitrer : laisser l'erreur de pandas (comportement actuel, **épinglé** par le test), ou lever une `ValueError` qui
  nomme l'argument et les méthodes supportées.
- **Reproduction** :
  ```python
  calculate_applicable_delay(delays, 'end', 'M', aggregation_method='nope')   # AttributeError
  ```
- **Test** : `tests/unit/delays/test_calculator.py::TestAggregation::test_unknown_method_name_raises_a_value_error_naming_the_argument`,
  `::test_non_callable_method_raises_a_type_error_naming_the_argument`, `::test_method_is_validated_before_the_computation`,
  `::test_callable_without_a_name_is_reported_by_its_type`
- **Correctif** : décision de l'auteur (2026-10-05) : `aggregation_method` est validé avant tout calcul (`_validate_aggregation_method`, essai sur un groupe minimal) ; un nom inconnu lève `ValueError: Unsupported aggregation_method 'nope': ...`. Un objet ni chaîne ni appelable lève `TypeError: 'aggregation_method' should be a string or a callable, got a int` (choix du TypeError : même convention que `frequency`). Un appelable sans `__name__` (`functools.partial`) est nommé par son type.
- **Statut** : corrigée

### ANO-DELAYS-016 — `MaskTransformer` : la période incomplète d'un bord n'est jamais générée
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::MaskTransformer` (`_generate_periods`)
- **Sévérité** : majeure
- **Observé** : les périodes sont générées par `pd.date_range(start=min, end=max, freq=<fréquence de masque ancrée>)`,
  qui ne renvoie que les ancres comprises dans `[min, max]`. Avec un ancrage **fin** (index journalier ou `ME` →
  `'ME'`, `'QE'`), la période qui contient la dernière date est omise dès que les données s'arrêtent avant sa fin ;
  avec un ancrage **début** (index `MS` → `'QS'`), c'est la période qui contient la première date qui est omise dès
  que les données commencent après son début. Rien n'y est masqué, même des positions présentes dans les données :
  série journalière du 1er janvier au 10 mars, `n_obs=4, how='first'` → 1-4 mars non masqués ; index `MS` de
  février à décembre, `mask_frequency='Q', how='last'` → mars non masqué ; 5 jours de janvier, `how='first'` → rien
  masqué. Le code d'extension des bords (`_extend_period_if_incomplete`, commit `76ed126`) n'est donc atteint que
  pour une moitié des cas. Le test vert `TestMaskTransformerPandasAliases[monthly]` (série de 120 jours finissant le
  29 avril) épinglait cette omission (« avril, incomplet, n'est pas masqué ») alors que le 29 avril est l'avant-dernier
  jour d'avril.
- **Attendu** : toute période qui contient au moins une observation est traitée, quel que soit l'ancrage ; les
  positions sont ensuite choisies selon la sémantique calendaire de l'extension des bords (1-4 mars masqués, mars
  masqué, 29 avril masqué). Le résultat ne doit pas dépendre de la position (début / fin) de l'index.
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import MaskTransformer
  s = pd.Series(np.arange(70.), index=pd.date_range('2024-01-01', '2024-03-10', freq='D'))
  MaskTransformer(n_obs=4, mask_frequency='M', how='first').fit_transform(s)['2024-03'].isna().sum()   # 0, attendu 4
  m = pd.Series(np.arange(11.), index=pd.date_range('2024-02-01', periods=11, freq='MS'))
  MaskTransformer(n_obs=1, mask_frequency='Q').fit_transform(m).loc['2024-03-01']                      # 1.0, attendu NaN
  ```
- **Test** : `tests/unit/delays/transformers/test_mask_transformer.py::TestBoundaries::test_series_end_with_how_first`,
  `::test_first_incomplete_quarter_on_month_start_index`, `::test_single_incomplete_period_with_how_first`,
  `::test_position_less_frequency_raises_no_pandas_deprecation[monthly]`
- **Correctif** : décision de l'auteur (2026-10-06) : les périodes sont parcourues à partir de chaque observation restante (`get_period_boundaries(date, mask_frequency)`), sans `pd.date_range` ancré : toute période contenant au moins une observation est traitée, quel que soit l'ancrage de l'index (`MaskTransformer._rows_to_mask`).
- **Statut** : corrigée

### ANO-DELAYS-017 — `MaskTransformer` : positions calendaires aux bords, positions relatives dans une période lacunaire
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::MaskTransformer` (`_mask_n_obs_per_period`, `_extend_period_if_incomplete`)
- **Sévérité** : majeure (arbitrée par l'auteur le 2026-10-06)
- **Observé** : une période coupée par le **début ou la fin des données** est complétée par des dates artificielles
  avant le choix des positions (commit `76ed126`) : les positions masquées sont calendaires et une position absente
  ne masque rien (données commençant le 10 janvier, `how='first'` → aucun jour de janvier masqué). Une période
  lacunaire **à l'intérieur** des données n'est pas complétée : les positions y sont comptées sur les dates
  disponibles (31 janvier absent, `how='last'` → le 30 janvier est masqué). Sur `irregular_index_timeseries`, les
  observations annuelles isolées du 1er janvier 2015, 2016 et 2017 (seules dans leur trimestre) sont ainsi masquées
  comme « dernier mois du trimestre », alors qu'avec la règle des bords elles ne le seraient pas ; idem pour les
  trimestres partiels d'`heterogeneous_coverage_panel` (Allemagne : janvier 2016, 2017, 2018).
- **Attendu** : une seule règle. La règle calendaire (celle des bords) est cohérente avec la finalité dans
  `PublicationDelayTransformer` (reproduire dans chaque période l'information manquante à la même position que la date
  de prédiction) et avec le commit `76ed126` ; elle a été retenue pour trier les tests hérités
  `test_series_start_with_how_first` et `test_series_end_with_how_last` (catégorie (c)), qui attendaient des positions
  relatives aux données. À confirmer par l'auteur ; le comportement intérieur actuel est **épinglé**.
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import MaskTransformer
  idx = pd.date_range('2024-01-01', '2024-02-29', freq='D').drop(pd.Timestamp('2024-01-31'))
  s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
  MaskTransformer(n_obs=1, mask_frequency='M').fit_transform(s).loc['2024-01-30']   # NaN (position relative)
  ```
- **Test** : `tests/unit/delays/transformers/test_mask_transformer.py::TestBoundaries::test_gap_inside_a_period_masks_the_last_available_date`,
  `tests/integration/delays/test_shift_mask_realistic.py::TestIrregularIndexTimeseries::test_lone_annual_dates_are_masked`
- **Correctif** : décision de l'auteur (2026-10-06) : règle calendaire partout. Dans chaque période, les positions sont celles de la grille régulière de la fréquence d'index détectée au `fit` (`pd.date_range(début, fin, freq=index_offset_, inclusive='left')`) ; une observation prend la position du dernier point de grille qui la précède ou l'égale. Une position absente (bord, lacune, index irrégulier) ne masque rien ; aucune observation n'est masquée parce qu'une voisine manque. `_extend_period_if_incomplete` et `_generate_periods` sont supprimées.
- **Statut** : corrigée

### ANO-DELAYS-018 — `MaskTransformer` non ajusté reconnu comme ajusté par sklearn
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::MaskTransformer.__init__`
- **Sévérité** : mineure
- **Observé** : `__init__` initialise `self.masked_data_ = None`. Un attribut à suffixe `_` posé dans `__init__` viole
  la convention sklearn : `check_is_fitted(MaskTransformer(1, 'Q'))` ne lève pas `NotFittedError` (et
  `check_estimator` signale un attribut posé hors paramètres). `ShiftTransformer` respecte la convention.
- **Attendu** : aucun attribut ajusté dans `__init__` ; `inverse_transform` teste `getattr(self, 'masked_data_', None)`
  (ou `check_is_fitted`) pour conserver son message « Must call transform before inverse_transform ».
- **Reproduction** :
  ```python
  from sklearn.utils.validation import check_is_fitted
  from tsforecast.delays.transformers import MaskTransformer
  check_is_fitted(MaskTransformer(n_obs=1, mask_frequency='Q'))   # ne lève rien
  ```
- **Test** : `tests/unit/delays/transformers/test_mask_transformer.py::TestSklearnProtocol::test_unfitted_masker_is_reported_unfitted`
- **Correctif** : `__init__` ne pose plus que les paramètres ; le stock (`masked_values_`, `original_dtypes_`) est créé par `fit`. `transform` et `inverse_transform` appellent `check_is_fitted` (`NotFittedError` avant `fit`).
- **Statut** : corrigée

### ANO-DELAYS-019 — `MaskTransformer` : `how` et `n_obs` non validés
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::MaskTransformer`
- **Sévérité** : mineure
- **Observé** : `how='middle'` ne masque rien, sans erreur (aucune des deux branches `'first'` / `'last'` n'est prise) ;
  `n_obs=-1` masque **toutes les positions sauf la première** de chaque période (`index[-(-1):]` = `index[1:]`) :
  58 jours masqués sur 60. Un `n_obs` négatif est atteignable depuis `PublicationDelayTransformer`
  (`_compute_mask_periods` renvoie `ceil((délai - écoulé) / période)`, négatif quand la donnée est déjà publiée, et
  `can_mask` est alors vrai) — à confirmer au prompt D4.
- **Attendu** : `ValueError` explicite pour un `how` hors `{'first', 'last'}` et pour un `n_obs` négatif (ou, pour ce
  dernier, traitement comme 0 — à arbitrer ; le test exige l'erreur).
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import MaskTransformer
  s = pd.Series(np.arange(60.), index=pd.date_range('2024-01-01', periods=60, freq='D'))
  MaskTransformer(2, 'M', how='middle').fit_transform(s).isna().sum()   # 0
  MaskTransformer(-1, 'M').fit_transform(s).isna().sum()                # 58
  ```
- **Test** : `tests/unit/delays/transformers/test_mask_transformer.py::TestParameterValidation::test_unknown_how_raises`,
  `::test_negative_n_obs_raises`
- **Correctif** : décision de l'auteur (2026-10-06) : `fit` valide `n_obs` (`TypeError` si non entier, booléens compris ; `ValueError: 'n_obs' must be a non-negative integer, got -1`) et `how` (`ValueError: how must be 'first' or 'last'`). `PublicationDelayTransformer` ramène à 0 un nombre d'observations négatif (donnée déjà publiée) et ne construit aucun `MaskTransformer` pour `n_obs == 0` (`_active_params`).
- **Statut** : corrigée

### ANO-DELAYS-020 — `MaskTransformer.inverse_transform` réinjecte des lignes absentes de son entrée
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::MaskTransformer.inverse_transform`
- **Sévérité** : mineure
- **Observé** : l'inversion concatène l'entrée et **toutes** les lignes masquées lors du dernier `transform`. Inverser
  une partie des données transformées (premier semestre de douze mois) renvoie aussi les lignes masquées du second
  semestre : 6 lignes en entrée, 8 en sortie. Même effet si l'entrée vient d'un autre jeu que le dernier `transform`.
- **Attendu** : la sortie a l'index de l'entrée (contrat sklearn de `inverse_transform`) ; seules les lignes masquées
  présentes dans l'entrée sont restituées.
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import MaskTransformer
  s = pd.Series(np.arange(12.), index=pd.date_range('2024-01-01', periods=12, freq='MS'))
  m = MaskTransformer(n_obs=1, mask_frequency='Q')
  len(m.inverse_transform(m.fit_transform(s).iloc[:6]))   # 8
  ```
- **Test** : `tests/unit/delays/transformers/test_mask_transformer.py::TestRoundTrip::test_inverse_transform_keeps_the_index_of_its_input`
- **Correctif** : décision de l'auteur (2026-10-06) : le stock cumule les cellules masquées par tous les `transform` depuis le `fit` (la dernière valeur l'emporte pour une même date) ; `inverse_transform` ne restitue que les cellules dont la date et la colonne sont dans son entrée, et renvoie l'index de son entrée. Les colonnes entières retrouvent leur type quand aucune valeur manquante ne subsiste et que la conversion est sans perte.
- **Statut** : corrigée

### ANO-DELAYS-021 — `ShiftTransformer` décale par position sur un index lacunaire ou irrégulier
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::ShiftTransformer` (`_shift_by_periods`)
- **Sévérité** : majeure
- **Observé** : le nouvel index est construit en prolongeant l'index d'entrée de `k` périodes régulières puis en
  gardant `len(data)` dates, et les valeurs sont **réaffectées par position**. Tant que l'index est régulier, cela
  revient à décaler de `k` périodes ; dès qu'il a des lacunes (détectées comme régulières par la détection de
  fréquence), chaque valeur glisse vers la date **voisine dans l'index**, pas de `k` périodes : index journalier sans
  le 3 janvier, `n_periods=1` → la valeur du 4 janvier arrive le 2. Sur `irregular_index_timeseries` (dates annuelles
  2015-2017 puis grille mensuelle), `n_periods=-1` envoie la valeur annuelle 2017 au 1er janvier 2018 (douze mois
  plus tard). L'aller-retour n'est pas non plus garanti : une lacune retirée par le décalage est reconstruite sur la
  grille régulière à l'inversion (`n_periods=-2` puis inverse : les dates 2015-01 et 2016-01 reviennent en 2016-11 et
  2016-12, données réalistes série et panel).
- **Attendu** : chaque date recule de `k` périodes exactement (`index - k * offset`), quelle que soit la régularité de
  l'index — ou, à défaut, une `ValueError` sur un index non régulier ; jamais un déplacement silencieux de durée variable.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.delays.transformers import ShiftTransformer
  idx = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-04', '2024-01-05'])
  ShiftTransformer(n_periods=1, frequency='D').fit_transform(pd.Series([1., 2., 3., 4.], index=idx))
  # 2023-12-31 1.0 / 2024-01-01 2.0 / 2024-01-02 3.0 / 2024-01-04 4.0  (attendu : 3.0 au 2024-01-03)
  ```
- **Test** : `tests/unit/delays/transformers/test_shift_transformer.py::TestIndexRobustness::test_index_with_gaps_shifts_by_calendar_periods`,
  `::test_index_with_gaps_round_trip[leading-gap-k=-2]`,
  `tests/integration/delays/test_shift_mask_realistic.py::TestIrregularIndexTimeseries::test_shift_moves_every_date_by_one_month`,
  `::test_shift_round_trip[k=-2]`, `::TestHeterogeneousCoveragePanel::test_shift_round_trip_per_entity[k=-2]`
- **Correctif** : décision de l'auteur (2026-10-06) : décalage calendaire `index - k * index_offset_` (arithmétique d'offset pandas, date par date), exactement réversible ; une date hors de la grille d'un offset calendaire (non `Tick`) lève `ValueError: ... date(s) are not on its grid`. L'extension / troncature positionnelle (`_extend_index_start`, `_extend_index_end`) est supprimée.
- **Statut** : corrigée

### ANO-DELAYS-022 — `ShiftTransformer` perd le nom de l'index
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::ShiftTransformer` (`_extend_index_start`, `_extend_index_end`)
- **Sévérité** : mineure
- **Observé** : l'index étendu est la concaténation d'un `pd.date_range` sans nom et de l'index d'origine ; le nom
  (`'periode'`, `'date'`…) disparaît après `transform` comme après `inverse_transform`. `MaskTransformer` le conserve.
- **Attendu** : le nom de l'index d'entrée est conservé.
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import ShiftTransformer
  s = pd.Series(np.arange(6.), index=pd.date_range('2024-01-01', periods=6, freq='MS', name='periode'))
  ShiftTransformer(n_periods=2, frequency='M').fit_transform(s).index.name   # None
  ```
- **Test** : `tests/unit/delays/transformers/test_shift_transformer.py::TestIndexRobustness::test_index_name_is_preserved` (deux cas)
- **Correctif** : conséquence du décalage par arithmétique d'offset, qui conserve le nom de l'index.
- **Statut** : corrigée

### ANO-DELAYS-023 — `ShiftTransformer` / `MaskTransformer` : un panel passé directement lève une `AttributeError`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::_detect_index_components` (via `ShiftTransformer.fit`, `MaskTransformer.fit`)
- **Sévérité** : mineure
- **Observé** : sur un `MultiIndex`, `detect_index_frequency` renvoie un dictionnaire par entité ; `fit` échoue avec
  `AttributeError: 'dict' object has no attribute 'multiplier'`. Le chemin panel prévu est `PanelwiseTransformer`
  (ce que fait `PublicationDelayTransformer`), mais rien ne l'indique.
- **Attendu** : `ValueError` explicite (« panel data: wrap the transformer in a PanelwiseTransformer ») ou traitement par
  entité ; le test exige une `ValueError`.
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import ShiftTransformer
  idx = pd.MultiIndex.from_product([['FR', 'DE'], pd.date_range('2024-01-01', periods=6, freq='MS')])
  ShiftTransformer(1, 'M').fit(pd.DataFrame({'a': np.arange(12.)}, index=idx))   # AttributeError
  ```
- **Test** : `tests/unit/delays/transformers/test_shift_transformer.py::TestIndexRobustness::test_multiindex_input_raises_a_clear_error`,
  `tests/unit/delays/transformers/test_mask_transformer.py::TestIndexRobustness::test_multiindex_input_raises_a_clear_error`
- **Correctif** : `_validate_time_series_input` lève `ValueError: ... got a MultiIndex (panel data): wrap it in a PanelwiseTransformer to apply it entity by entity.` dans `fit`, `transform` et `inverse_transform` des deux classes.
- **Statut** : corrigée

### ANO-DELAYS-024 — `ShiftTransformer` sur un index en jours ouvrés : une semaine vaut 7 jours ouvrés
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::ShiftTransformer._convert_shift_periods_to_index_periods`
- **Sévérité** : mineure
- **Observé** : la conversion passe par `DurationConverter.get_conversion_factor`, dont la table partagée assimile
  `'B'` à `'D'` (voir ANO-UTILS-038, correctif limité à la comparaison) : `'W' → 'B'` vaut 7, `'M' → 'B'` 30. Sur un
  index `B`, `ShiftTransformer(1, 'W')` recule de 7 jours ouvrés (≈ 1,4 semaine), `ShiftTransformer(1, 'M')` de 30
  jours ouvrés (≈ 6 semaines).
- **Attendu** : une semaine = 5 jours ouvrés (et un mois ≈ 21-22 jours ouvrés, ou un décalage calendaire ramené sur la
  grille ouvrée).
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import ShiftTransformer
  b = pd.Series(np.arange(10.), index=pd.date_range('2024-01-01', periods=10, freq='B'))
  ShiftTransformer(1, 'W').fit_transform(b).index[0]   # 2023-12-21, attendu 2023-12-25
  ```
- **Test** : `tests/unit/delays/transformers/test_shift_transformer.py::TestShiftFrequency::test_week_on_business_day_index_is_five_business_days`
- **Correctif** : sur un index `B`, le facteur est calculé en jours calendaires puis ramené à 5 jours ouvrés par semaine (`ShiftTransformer._to_index_periods`) : une semaine = 5, un mois = round(30 × 5 / 7) = 21 jours ouvrés. La table partagée de `DurationConverter` est inchangée.
- **Statut** : corrigée

### ANO-DELAYS-025 — `ShiftTransformer` ignore le multiplicateur de `frequency`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::ShiftTransformer._convert_shift_periods_to_index_periods`
- **Sévérité** : mineure
- **Observé** : `normalize_frequency('2M')` rend `'M'`, égal à la fréquence de l'index : aucune conversion, et
  `ShiftTransformer(1, '2M')` décale d'**un** mois sur un index `MS`. `MaskTransformer(…, mask_frequency='2Q')`
  honore au contraire le multiplicateur (semestres), et un index multiplié est rejeté (`_detect_index_components`).
- **Attendu** : un décalage de deux mois par période, ou une `ValueError` comme pour un index multiplié ; jamais un
  multiplicateur ignoré en silence. Le test admet les deux.
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import ShiftTransformer
  m = pd.Series(np.arange(6.), index=pd.date_range('2024-01-01', periods=6, freq='MS'))
  ShiftTransformer(1, '2M').fit_transform(m).index[0]   # 2023-12-01, attendu 2023-11-01
  ```
- **Test** : `tests/unit/delays/transformers/test_shift_transformer.py::TestShiftFrequency::test_multiplied_shift_frequency_is_honoured_or_rejected`
- **Correctif** : le facteur vient de `DurationConverter.get_conversion_factor(frequency, base de l'index)`, qui honore le multiplicateur (`'2M'` → 2 mois par période) ; plus de raccourci par égalité des fréquences normalisées.
- **Statut** : corrigée

### ANO-DELAYS-026 — Docstrings de `ShiftTransformer` et `MaskTransformer` ≠ code
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/delays/transformers.py::ShiftTransformer`, `::MaskTransformer`
- **Sévérité** : mineure
- **Observé** :
  - `ShiftTransformer` : paramètre `frequency_check` (« ignore » / « warn » / « raise ») et attribut `is_series_`
    documentés mais inexistants (l'exemple `ShiftTransformer(..., frequency_check='warn')` lève `TypeError`) ;
    l'exemple « Shift forward by 2 monthly periods » avec `n_periods=2` recule les valeurs de deux mois (un
    `n_periods` positif avance l'index vers le passé) ; l'exemple « Perfect inverse » inverse avec le second
    transformateur la sortie du premier (`original.equals(series)` est faux) ; `freq='M'` est un alias déprécié ;
    `fit` annonce une erreur `frequency_check='raise'` ; l'exemple de `_convert_shift_periods_to_index_periods`
    appelle `self` hors de la classe.
  - `MaskTransformer` : paramètre `prediction_date` et attributs `prediction_date`, `original_data_` documentés mais
    inexistants (l'exemple lève `TypeError`) ; `how` absent de la section *Parameters* ; « Mask N most recent
    observations » alors que `how='first'` masque les plus anciennes ; `transform` annoté `-> pd.Series` (renvoie
    aussi un `DataFrame`) ; ni la sémantique calendaire des bords (ANO-DELAYS-017) ni le sens du masquage d'une
    cellule à l'inversion (valeur d'origine restituée même sous une prédiction) ne sont décrits.
- **Attendu** : docstrings alignées sur le code ; sens du signe de `n_periods` explicite (positif = valeurs avancées
  vers des dates antérieures ; `PublicationDelayTransformer` passe un `n_periods` négatif).
- **Reproduction** : `ShiftTransformer(n_periods=1, frequency='M', frequency_check='warn')` → `TypeError` ;
  `MaskTransformer(n_obs=2, mask_frequency='M', prediction_date=datetime(2024, 3, 31))` → `TypeError`.
- **Test** : sans `xfail` ([DOC]) ; le comportement réel est fixé par `TestShiftGoldValues` et `TestMaskGoldValues`.
- **Correctif** : docstrings des deux classes réécrites (paramètres et attributs réels, sens du signe de `n_periods`, sémantique calendaire des positions, cumul et restitution des cellules masquées, exemples exécutés par doctest).
- **Statut** : corrigée

### ANO-DELAYS-027 — `MaskTransformer` : index hebdomadaire masqué par mois ou trimestre → `Invalid frequency: ME-SUN`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::MaskTransformer._generate_periods`
- **Sévérité** : majeure
- **Observé** : la fréquence des périodes de masque est construite avec la position **et l'ancre de l'index**
  (`build_frequency_string(mask_frequency, self.index_position_, self.index_suffix_)`). Pour un index `W-SUN`
  (ancre `SUN` conservée depuis le correctif d'ANO-UTILS-001), `mask_frequency='M'` donne `'ME-SUN'` et `transform`
  lève `ValueError: Invalid frequency: ME-SUN, failed to parse with error message: ValueError('Bad freq suffix SUN')`.
  Tout index hebdomadaire est donc impossible à masquer par mois, trimestre ou année. Pour un index `QS-FEB`, le
  même mécanisme produirait des années commençant en février (`'YS-FEB'`), ce qui peut être voulu ou non.
- **Attendu** : l'ancre d'un jour de semaine n'est pas reportée sur une fréquence mensuelle ou plus basse ; un index
  hebdomadaire se masque comme un index journalier (dernier dimanche de chaque mois pour `how='last'`).
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import MaskTransformer
  w = pd.Series(np.arange(13.), index=pd.date_range('2024-01-07', periods=13, freq='W-SUN'))
  MaskTransformer(n_obs=1, mask_frequency='M').fit_transform(w)   # ValueError: Invalid frequency: ME-SUN
  ```
- **Test** : `tests/unit/delays/transformers/test_mask_transformer.py::TestMaskGoldValues::test_weekly_index_masked_by_month` (deux cas)
- **Correctif** : les périodes de masque sont calculées par `get_period_boundaries` sur `mask_frequency` seule : l'ancre de l'index n'est plus reportée sur la fréquence de masque. Une ancre explicite (`'YE-NOV'`) est honorée.
- **Statut** : corrigée

### ANO-DELAYS-028 — `PublicationDelayTransformer` : `reference_point` en dictionnaire rejeté par `__init__`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer.__init__` (et, par ricochet,
  `create_delay_transformer_factory`)
- **Sévérité** : majeure
- **Observé** : `reference_point not in ['start', 'end']` est vrai pour tout dictionnaire :
  `ValueError: reference_point must be 'start' or 'end', got '{'GDP': 'end', 'CPI': 'start'}'`. La signature
  (`Dict[str, Literal['start', 'end']]`), la docstring et `_build_parameter_dict` acceptent pourtant un dictionnaire, et
  `_build_entity_params` en produit un dès que le point de référence varie d'une variable à l'autre d'une entité : la
  fabrique échoue alors à chaque appel. (`calculate_applicable_delay` renvoie un point de référence unique : le cas ne
  survient que pour un tableau assemblé à la main.)
- **Attendu** : chaque valeur du dictionnaire validée, le point de référence appliqué variable par variable
  (45 j depuis la fin → -3 ; depuis le début → -2 au 15/12/2023).
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import PublicationDelayTransformer
  PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D', reference_point={'GDP': 'end'})  # ValueError
  ```
- **Test** : `tests/unit/delays/transformers/test_publication_delay_transformer.py::TestReferencePoint::test_params_with_dict_reference_point`,
  `tests/unit/delays/transformers/test_factories.py::TestCreateDelayTransformerFactory::test_reference_point_varying_by_variable`
- **Correctif** : `__init__` valide un `reference_point` en dictionnaire valeur par valeur (`ValueError ... for variable 'CPI' got 'middle'`) ; `fit` résout le point de référence variable par variable (`_resolve_parameter`).
- **Statut** : corrigée

### ANO-DELAYS-029 — `PublicationDelayTransformer` : `strategy` en dictionnaire → `too many values to unpack`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer.fit` (énumération des colonnes par
  stratégie), `::_build_parameter_dict`
- **Sévérité** : majeure
- **Observé** : `[k for k, v in self.strategy if v == 'shift']` itère sur les **clés** du dictionnaire (il manque
  `.items()`) : `ValueError: too many values to unpack (expected 2)` pour tout nom de variable de plus de deux caractères.
  La stratégie par variable, acceptée et validée par `__init__`, et transmise telle quelle par la fabrique
  (`_resolve_strategy` renvoie le dictionnaire si ses clés sont des chaînes), est donc inutilisable. Accessoirement,
  `_build_parameter_dict` impute les valeurs par défaut aux colonnes **hors** du dictionnaire de stratégies (jamais
  retardées), alors que `__init__` annonce `'default_values' is ignored` dans ce cas.
- **Attendu** : colonnes `'shift'` décalées, colonnes `'mask'` masquées ; aucune imputation par défaut (comme annoncé).
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import PublicationDelayTransformer
  X = pd.DataFrame({'GDP': np.arange(12.), 'CPI': np.arange(12.)}, index=pd.date_range('2023-01-01', periods=12, freq='MS'))
  PublicationDelayTransformer(delays={'GDP': 45.0, 'CPI': 20.0}, strategy={'GDP': 'shift', 'CPI': 'mask'},
                              delay_unit='D', reference_point='start', target_frequency={'CPI': 'Q'},
                              prediction_date='2023-12-15').fit(X)   # ValueError: too many values to unpack
  ```
- **Test** : `test_publication_delay_transformer.py::TestStrategy` (`test_strategy_per_variable*`, `test_delayed_column_without_strategy_is_left_unchanged`, `test_masked_variable_needs_a_target_frequency`),
  `test_factories.py::TestPanelwiseIntegration::test_per_variable_strategy`
- **Correctif** : itération sur `self.strategy.items()` (`_split_columns_by_strategy`) ; une variable retardée absente du dictionnaire est laissée telle quelle avec un avertissement ; `default_values` n'est plus imputé avec une stratégie par variable, comme annoncé (une variable masquée sans fréquence cible lève alors la `ValueError` d'ANO-DELAYS-033).
- **Statut** : corrigée

### ANO-DELAYS-030 — `PublicationDelayTransformer` : un panel passé directement → `KeyError` dans `fit`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer.fit`
- **Sévérité** : majeure
- **Observé** : sur un `MultiIndex` (entité, date), `detect_frequency` renvoie des clés `(entité, colonne)` ; `fit` lit
  `self.detected_frequencies_[col]` → `KeyError: 'CPI'`. La docstring promet « Automatic panel wrapping with
  PanelwiseTransformer » et `transform` contient la branche panel (`is_panel`), inatteignable. Pas une régression d'un
  renommage : le contrat de `detect_dataset_frequency` renvoyait déjà des tuples `(panel_id, column)` en mars 2026
  (`3f2fcc7`), et `fit` lit `detected_frequencies_[col]` depuis `e0a664b`. Cause probable des `KeyError` dans `fit` du
  groupe D5 (`tests/integration/delays/test_integration.py`).
- **Attendu** : un jeu de délais par colonne appliqué à chaque entité (fréquence détectée par entité), ou une
  `ValueError` orientant vers `create_delay_transformer_factory` + `PanelwiseTransformer`.
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import PublicationDelayTransformer
  X = pd.DataFrame({'GDP': np.arange(12.)}, index=pd.date_range('2023-01-01', periods=12, freq='MS'))
  panel = pd.concat({'FR': X, 'DE': X}, names=['country', 'date'])
  PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D', reference_point='start').fit(panel)  # KeyError
  ```
- **Test** : `test_publication_delay_transformer.py::TestInputTypes::test_panel_gets_the_shift_of_each_column`
- **Correctif** : arbitrage de l'auteur (2026-10-08) : panel accepté, avec l'avertissement « Panel data: the same publication delays are applied to every entity. Use create_delay_transformer_factory (or prepare_entity_kwargs_from_delays) with a PanelwiseTransformer to apply per-entity delays. » La fréquence d'une colonne est celle que partagent ses entités (`_detect_column_frequencies`, une entité où la colonne est vide est ignorée) ; des fréquences différentes pour une colonne retardée, ou des fréquences d'index différentes pour le masque, lèvent une `ValueError` qui renvoie vers la fabrique. `transform` / `inverse_transform` passent par `PanelwiseTransformer` (branche panel désormais atteinte).
- **Statut** : corrigée

### ANO-DELAYS-031 — Repli masque → décalage : `n_periods` positif, les valeurs apparaissent avant leur publication
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer.fit` (branche `not result['can_mask']`)
- **Sévérité** : majeure (fuite d'information : anticipation silencieuse)
- **Observé** : quand le masque couvrirait toute la période cible, la colonne passe au décalage avec
  `n_periods = result['n_periods']`, c'est-à-dire le nombre **positif** d'observations d'index à masquer, alors que la
  convention de `ShiftTransformer` (et de `_compute_shift_periods`) est négative pour retarder. Exemple : 400 jours,
  cible trimestrielle, série mensuelle, 15/12/2023 → `n_periods = +13` : la valeur de janvier 2023 est datée de
  décembre 2021. Le nombre est en plus exprimé en périodes d'index, pas de la fréquence de la colonne.
- **Attendu** : le décalage de la stratégie 'shift' (`_compute_shift_periods`) : -13 ici, -3 pour 75 jours.
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import PublicationDelayTransformer
  X = pd.DataFrame({'GDP': np.arange(12.)}, index=pd.date_range('2023-01-01', periods=12, freq='MS'))
  t = PublicationDelayTransformer(delays={'GDP': 75.0}, strategy='mask', delay_unit='D', reference_point='start',
                                  target_frequency='Q', prediction_date='2023-12-15').fit(X)
  t.shift_params   # {'GDP': {'n_periods': 3, ...}}, attendu -3
  ```
- **Test** : `test_publication_delay_transformer.py::TestMaskParams::test_fallback_shift_moves_values_to_later_dates`,
  `test_publication_delay_transformer_transform.py::TestRoundTrip::test_fallback_never_shows_a_value_before_its_date`
- **Correctif** : le repli calcule le décalage de la stratégie 'shift' (`_compute_shift_periods` : périodes de la colonne, signe négatif).
- **Statut** : corrigée

### ANO-DELAYS-032 — `default_values` : `delay` et `unit` jamais appliqués
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer.fit`, `::_build_parameter_dict`
- **Sévérité** : majeure
- **Observé** : `__init__` exige les clés `delay`, `unit`, `reference_point` (et `target_frequency` pour 'mask'), mais :
  (1) `fit` cherche l'unité sous la clé `'delay_unit'` (`default_key='delay_unit'`), absente : l'unité par défaut n'est
  jamais imputée (avertissement « Could not impute a default 'delay_unit' » puis `KeyError`) ; (2) `default_values['delay']`
  n'est jamais lu : avec `default_values`, toutes les colonnes de `X` sont retenues mais `delays_dict[col]` lève
  `KeyError` pour une colonne sans délai. Seuls `reference_point` et `target_frequency` par défaut fonctionnent.
- **Attendu** : une colonne sans délai reçoit le délai et l'unité par défaut (20 j depuis le début → -1).
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import PublicationDelayTransformer
  X = pd.DataFrame({'GDP': np.arange(12.), 'CPI': np.arange(12.)}, index=pd.date_range('2023-01-01', periods=12, freq='MS'))
  PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D', reference_point='start', prediction_date='2023-12-15',
                              default_values={'delay': 20.0, 'unit': 'D', 'reference_point': 'start'}).fit(X)  # KeyError: 'CPI'
  ```
- **Test** : `test_publication_delay_transformer.py::TestDefaultValues::test_default_delay_for_a_column_without_delay`,
  `::test_default_unit`
- **Correctif** : `_build_delays_dict` donne `default_values['delay']` aux colonnes de `X` sans délai connu (avertissement « Imputed default delay value ... ») ; l'unité par défaut est lue sous la clé `'unit'`. Le rapport porte l'origine du délai (`ColumnDelayRecord.delay_source`, `'explicit'` / `'default'`) et `defaults_imputed` liste `(colonne, 'delay')`.
- **Statut** : corrigée

### ANO-DELAYS-033 — Unité ou point de référence non résolus : avertissement puis `KeyError` brut
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer.fit` (`_compute_shift_periods`,
  `_compute_mask_periods`)
- **Sévérité** : mineure
- **Observé** : un dictionnaire de délais sans `delay_unit` (ou `reference_point`) produit un avertissement « Could not
  impute… » puis `KeyError: 'GDP'` dans `_compute_shift_periods`. C'était la cause des 4 échecs hérités
  `TestPublicationDelayTransformerFitTransform::*` (tests corrigés, catégorie (c) : rien ne permet d'inférer l'unité
  d'un dictionnaire ; aucun renommage en cause, la signature est inchangée depuis `d16b0b2`).
- **Attendu** : `ValueError` nommant le paramètre et les colonnes concernées.
- **Reproduction** : `PublicationDelayTransformer(delays={'GDP': 45.0}, reference_point='start').fit(X)` → `KeyError: 'GDP'`.
- **Test** : `test_publication_delay_transformer.py::TestMissingDelays::test_unresolved_unit_raises_a_clear_error`
- **Correctif** : `_check_resolved` lève `ValueError: No 'delay_unit' for the delayed columns ['GDP']: give it to the constructor, in the delays table, or in 'default_values'` (idem pour `reference_point` et, en masque, `target_frequency`). Seules les colonnes retardées doivent avoir ces paramètres.
- **Statut** : corrigée

### ANO-DELAYS-034 — `handle_missing_delays` validé mais sans effet
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer`
- **Sévérité** : mineure
- **Observé** : le paramètre est validé et stocké, puis jamais lu : `'error'` ne lève rien pour une colonne de `X` sans
  délai, `'warn'` n'avertit pas, `'ignore'` est le comportement de fait (la colonne est laissée intacte et listée dans
  `fit_report_.columns_unaffected`).
- **Attendu** : `'error'` → `ValueError` nommant les colonnes sans délai ; `'warn'` → avertissement ; `'ignore'` → rien.
- **Reproduction** : `PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D', reference_point='start',
  handle_missing_delays='error').fit(X.assign(Z=1.0))` → aucun effet.
- **Test** : `test_publication_delay_transformer.py::TestMissingDelays::test_handle_missing_delays_error`
- **Correctif** : `_handle_missing_delays` : `'warn'` (défaut) → « No publication delay for the columns [...]: they are left unchanged », `'error'` → `ValueError` avec le même message, `'ignore'` → silence.
- **Statut** : corrigée

### ANO-DELAYS-035 — Délai `NaN` → `ValueError: cannot convert float NaN to integer`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer._compute_shift_periods`
- **Sévérité** : mineure
- **Observé** : `calculate_applicable_delay` renvoie un délai `NaN` pour un couple sans délai connu (`n_observations = 0`,
  cf. ANO-DELAYS-011) ; transmis à la PDT (directement ou par la fabrique), il fait échouer `math.ceil`.
- **Attendu** : un délai inconnu traité comme un délai manquant (colonne laissée intacte, signalée selon
  `handle_missing_delays`).
- **Reproduction** : `PublicationDelayTransformer(delays={'GDP': float('nan')}, delay_unit='D', reference_point='start').fit(X)`.
- **Test** : `test_publication_delay_transformer.py::TestMissingDelays::test_unknown_delay_is_treated_as_missing`
- **Correctif** : un délai `NaN` (ou `None`) est un délai inconnu (`_is_missing`) : colonne traitée comme sans délai (`handle_missing_delays`), ou complétée par `default_values['delay']`.
- **Statut** : corrigée

### ANO-DELAYS-036 — `PublicationDelayTransformer` refuse une `Series`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer.fit`
- **Sévérité** : mineure
- **Observé** : `fit` / `transform` sont annotés `Union[pd.Series, pd.DataFrame]` mais lisent `X.columns` :
  `AttributeError: 'Series' object has no attribute 'columns'`.
- **Attendu** : une `Series` nommée traitée comme un DataFrame d'une colonne (et restituée en `Series`), ou annotation
  restreinte au DataFrame avec une erreur explicite.
- **Reproduction** : `PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D', reference_point='start').fit(X['GDP'])`.
- **Test** : `test_publication_delay_transformer.py::TestInputTypes::test_series_is_accepted`
- **Correctif** : `_as_frame` / `_restore_series` : une `Series` (nommée ou non) est traitée comme un DataFrame d'une colonne et restituée en `Series` par `transform` / `inverse_transform` ; tout autre type lève `TypeError`.
- **Statut** : corrigée

### ANO-DELAYS-037 — `inverse_transform` sans `transform` préalable → `AttributeError`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer.inverse_transform`
- **Sévérité** : mineure
- **Observé** : les transformateurs auxiliaires sont créés (et ajustés) par `transform`, pas par `fit` :
  `inverse_transform` juste après `fit` lève `AttributeError: ... no attribute 'auxiliary_transformers_'` alors que
  `check_is_fitted` passe. L'inversion dépend aussi des données du **dernier** `transform` (index détecté à ce moment,
  cellules masquées mémorisées par `MaskTransformer`).
- **Attendu** : `NotFittedError` explicite (« call transform first ») — arbitrage de l'auteur.
- **Reproduction** : `t = PublicationDelayTransformer(...).fit(X); t.inverse_transform(X)` → `AttributeError`.
- **Test** : `test_publication_delay_transformer.py::TestSklearnProtocol::test_inverse_transform_before_transform`
- **Correctif** : arbitrage de l'auteur (2026-10-08) : l'inversion ne sert qu'après un `transform`. `inverse_transform` sans `transform` depuis le dernier `fit` lève `NotFittedError: inverse_transform reverses the last transform: call transform before inverse_transform` ; `fit` oublie les auxiliaires d'un `transform` précédent.
- **Statut** : corrigée

### ANO-DELAYS-038 — Sortie de `calculate_applicable_delay` non acceptée telle quelle par `PublicationDelayTransformer`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer._infer_parameters_from_delays`
- **Sévérité** : mineure
- **Observé** : la PDT attend un tableau **à plat** avec une colonne `column` ; `calculate_applicable_delay` (sans
  `aggregate_by_panel`) porte la variable dans l'**index** nommé `column` → `KeyError: "['column'] not in index"`.
  Il faut `delays=applicable.reset_index()` (contrat vérifié par
  `TestDelaysFormat::test_output_of_calculate_applicable_delay`). À l'inverse, les fabriques lisent la variable dans
  l'index : deux conventions pour le même tableau.
- **Attendu** : à trancher — accepter la variable dans l'index (niveau `column` ou dernier niveau, comme les fabriques),
  ou documenter `reset_index()`.
- **Reproduction** : `PublicationDelayTransformer(delays=calculate_applicable_delay(detected, 'start', 'M')).fit(X)`.
- **Test** : `test_publication_delay_transformer.py::TestDelaysFormat::test_raw_output_of_calculate_applicable_delay`
- **Correctif** : arbitrage de l'auteur (2026-10-08) : la variable est lue dans la colonne `column`, à défaut dans le niveau d'index `'column'`, à défaut dans le dernier niveau (convention des fabriques) — la sortie de `calculate_applicable_delay` est acceptée telle quelle. Une variable listée plusieurs fois (délais par entité) lève une `ValueError` qui renvoie vers la fabrique ; un tableau sans colonne `delay` aussi.
- **Statut** : corrigée

### ANO-DELAYS-039 — Avertissements parasites (`fit` et fabrique)
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer._build_parameter_dict`, `::__init__`
  (via `create_delay_transformer_factory`)
- **Sévérité** : cosmétique
- **Observé** : (1) sans `default_values`, chaque `fit` émet « Could not impute a default 'delay_unit' for columns set()… »
  (et idem pour `reference_point`) même quand tout est résolu (ensemble vide) ; (2) la fabrique passe toujours
  `target_frequency` (colonne `frequency` du tableau), d'où « 'target_frequency' is ignored when a shifting strategy is
  applied » pour chaque entité en stratégie 'shift'. Le bruit masque les avertissements utiles.
- **Attendu** : avertissement seulement pour un ensemble non vide ; pas d'avertissement quand la fabrique transmet
  une fréquence cible qu'elle n'a pas choisie.
- **Reproduction** : `PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D', reference_point='start').fit(X)`
  → 2 avertissements.
- **Test** : `test_publication_delay_transformer.py::TestWarnings::test_fully_resolved_fit_emits_no_warning`,
  `test_factories.py::TestCreateDelayTransformerFactory::test_shift_factory_emits_no_warning`
- **Correctif** : (1) les avertissements « Could not impute ... » ont disparu : un paramètre manquant d'une colonne retardée lève l'erreur d'ANO-DELAYS-033, une colonne sans délai n'a besoin d'aucun paramètre ; (2) la fabrique et `prepare_entity_kwargs_from_delays` ne passent `target_frequency` qu'aux entités dont la stratégie n'est pas 'shift' (`_entity_kwargs`).
- **Statut** : corrigée

### ANO-DELAYS-040 — Colonne retardée entièrement `NaN` : `ValueError` cryptique au lieu d'un avertissement
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer.fit`
- **Sévérité** : mineure
- **Observé** : une colonne retardée sans fréquence détectable (entièrement `NaN`, ou jeu d'une seule observation / vide)
  fait échouer `get_period_start` : `ValueError: Frequency must be a string, got <class 'NoneType'>`, pour **toutes**
  les colonnes. La docstring de la classe annonce « Warning generation for all-NaN columns ». (Situation réaliste par
  entité : `climat_affaires` est absente pour l'Italie dans le panel du notebook 3.)
- **Attendu** : avertissement nommant la colonne, colonne laissée telle quelle, autres colonnes traitées ; pour un jeu
  sans fréquence détectable, `ValueError` explicite.
- **Reproduction** : `PublicationDelayTransformer(delays={'GDP': 45.0, 'CPI': 20.0}, delay_unit='D',
  reference_point='start').fit(X.assign(CPI=np.nan))`.
- **Test** : `test_publication_delay_transformer_transform.py::TestRobustness::test_all_nan_delayed_column_is_left_as_is`
- **Correctif** : une colonne retardée sans fréquence détectable est signalée (« Could not detect the frequency of the delayed columns [...] ... left unchanged ») et laissée telle quelle, les autres sont traitées ; si aucune colonne n'a de fréquence détectable alors que des colonnes sont retardées, `ValueError: Could not detect the frequency of any column of X ...`.
- **Statut** : corrigée

### ANO-DELAYS-041 — Docstrings de `PublicationDelayTransformer` et des fabriques ≠ code
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer`, `::create_delay_transformer_factory`,
  `::prepare_entity_kwargs_from_delays`, `::_build_entity_params`, `::_extract_param_by_variable`, `::_resolve_strategy`
- **Sévérité** : mineure
- **Observé** :
  - `PublicationDelayTransformer` : paramètre `default_delay` documenté (le vrai est `default_values`, clés non
    décrites) ; attribut `column_transformers_` inexistant (les attributs réels sont `shift_params`, `mask_params`,
    `auxiliary_transformers_`, posé par `transform`) ; `handle_missing_delays: 'ignore', 'ignore', or 'error'` ; format
    attendu du DataFrame `delays` (`column`, `delay`, `unit`, `reference_point`, `frequency`) non décrit ; signe et
    unité de `n_periods`, croissance de l'index par le décalage et aller-retour (dates ajoutées vides) non décrits ;
    `fit` / `transform` annoncent `Series` (ANO-DELAYS-036) et l'enveloppe panel (ANO-DELAYS-030).
  - Fabriques : arguments `panel_level` / `variable_level` documentés mais inexistants (niveaux repérés par position :
    tous sauf le dernier = entité) ; `_build_entity_params` / `_extract_param_by_variable` documentent des arguments
    inexistants (`panel_level_name`, `col`, `variable_level_name`) ; `_resolve_strategy` annonce un `str` mais renvoie
    le dictionnaire par variable tel quel.
- **Attendu** : docstrings alignées sur le code.
- **Test** : sans `xfail` ([DOC]) ; comportement réel fixé par `test_factories.py` et
  `test_publication_delay_transformer*.py`.
- **Correctif** : docstrings de `PublicationDelayTransformer` (format de `delays`, calcul de `n_periods`, croissance de l'index et aller-retour, attributs réels, `Raises` / `Warns`, exemple exécuté), des fabriques et de leurs auxiliaires réécrites ; exemples exécutés par doctest (`test_factories.py::TestDocstringExamples`).
- **Statut** : corrigée

### ANO-DELAYS-042 — Périodes de 30 / 91 / 365 jours : décalage ou masque faux d'une période près d'une frontière calendaire
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer._compute_shift_periods`,
  `::_compute_mask_periods`
- **Sévérité** : majeure
- **Observé** : `n_periods = -ceil((délai - écoulé) / durée)`, la durée d'une période venant de `convert_duration`
  (un mois = 30 jours, un trimestre = 91, une année = 365 ; convention documentée par la docstring de la classe).
  Quand les mois qui séparent la période retardée de la date de prédiction ne font pas 30 jours chacun, le décalage
  (et le nombre d'observations masquées) est faux d'une période, **dans les deux sens** :
  - **fuite** : février + mars 2023 = 59 jours < 60 ; délai de 69 jours depuis le début, prédiction au 10 avril 2023 →
    `n_periods = -ceil(55 / 30) = -2` : la valeur de février, publiée le 11 avril, est visible au 10 avril ;
  - **perte** : juillet + août = 62 jours > 60 ; une valeur publiée **le jour même** de la prédiction est repoussée
    après elle. C'est le cas du flux de base : deux téléchargements (le second le 15 sept. 2024) → délai détecté =
    date de téléchargement - début de période (76 jours pour juillet) → prédiction à la date du second
    téléchargement → `-ceil((76 - 14) / 30) = -3` au lieu de -2 : la valeur de juillet, publiée le 15 septembre,
    n'est pas visible au 15 septembre (panel du notebook 3, inflation et chômage de la France et de l'Italie).
  Même convention dans `_compute_mask_periods` (45 jours depuis le début, prédiction au 15 août : 2 mois masqués au
  lieu d'un, juillet étant publié le 15 août).
- **Attendu** : le nombre de périodes du calendrier : plus petit `k ≥ 0` tel que `début(période de la prédiction) -
  k périodes (+ une période si le point de référence est la fin) + délai ≤ date de prédiction` (arithmétique
  `pd.DateOffset`, comme le décalage calendaire d'ANO-DELAYS-021). Finalité de la classe : une valeur est visible à la
  date de prédiction **si et seulement si** elle est publiée à cette date ; le sens « fuite » introduit un biais
  d'anticipation (look-ahead) dans un backtest. La convention à 30 jours est documentée, d'où la sévérité « majeure »
  plutôt que « bloquante » : à arbitrer entre calendrier exact et convention assumée (avec sa limite documentée).
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import PublicationDelayTransformer
  X = pd.DataFrame({'a': np.arange(6.)}, index=pd.date_range('2023-01-01', periods=6, freq='MS'))
  t = PublicationDelayTransformer(delays={'a': 69.0}, prediction_date='2023-04-10', delay_unit='D', reference_point='start')
  t.fit_transform(X).loc['2023-04-01', 'a']   # 1.0 (février, publié le 11 avril) ; attendu 0.0 (janvier)
  ```
- **Test** : `tests/unit/delays/transformers/test_publication_delay_transformer.py::TestCalendarBoundaries`
  (`test_value_published_after_the_prediction_date_is_hidden`, `test_value_published_on_the_prediction_date_is_visible`),
  `::test_business_days_skip_the_weekend`, `::test_negative_delay_on_business_days`,
  `::test_end_of_a_quarter_counts_from_the_next_quarter`,
  `::TestMaskParams::test_whole_february_cannot_be_masked`, `::test_period_index_is_masked_like_its_month_starts`,
  `::test_semi_monthly_target`,
  `tests/integration/delays/test_workflow.py::TestTwoDownloadsOnHeterogeneousPanel::test_number_of_periods_follows_the_calendar`,
  `::test_published_value_is_visible_at_the_prediction_date`
- **Correctif** : décision de l'auteur (2026-10-09) : décompte calendaire. `_count_unpublished_periods` cherche le plus
  petit `k` (négatif pour un délai négatif) tel que la période commençant `k` périodes avant celle de la date de
  prédiction soit publiée à cette date, en parcourant les périodes avec `get_period_start` / `get_period_end`
  (`tsforecast/utils/time`, bornes calendaires exactes déjà utilisées par le paquet ; jours ouvrés : `BDay`, un
  week-end appartenant au vendredi). Le délai reste une durée (`_delay_to_timedelta`, longueurs de
  `get_duration_nanoseconds` : un délai en mois compte 30 jours, comme dans `calculate_applicable_delay`).
  `_compute_shift_periods` et `_compute_mask_periods` l'utilisent (périodes de la colonne pour le décalage, de l'index
  pour le masque). La faisabilité du masque (`can_mask`) compare désormais le nombre d'observations à masquer au
  plus petit nombre de périodes de l'index dans une période cible des données, compté par
  `FrequencyConverter.count_subperiods_per_period` (février : 28 jours au lieu de 30). Valeurs d'or des tests D4
  tombant sur une frontière (75 jours au 15 décembre, la publication d'octobre tombant le jour même) corrigées : -2
  au lieu de -3 ; exemples de la docstring de la fabrique et du tutoriel (§4.3.3-4.3.4) corrigés.
- **Statut** : corrigée

### ANO-DELAYS-043 — Sortie panel de `PublicationDelayTransformer` non triée par date au sein des entités
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer._assemble` (via `transform` sur un panel)
- **Sévérité** : mineure
- **Observé** : sur un panel `MultiIndex` (entité, date), `_assemble` joint les colonnes décalées (index décalé) et les
  colonnes non décalées par `pd.concat(axis=1, join='outer')` ; l'union de deux `MultiIndex` n'est pas triée : les
  dates propres à l'un des index sont ajoutées **en fin de frame**. Sur `heterogeneous_coverage_panel`, les lignes de la
  France commencent par 2015-04-01, 2016-04-01, ... et se terminent par 2015-01-01 ... 2018-03-01 ; sur un panel de
  deux entités `FR` / `DE`, les lignes `(FR, 2024-01-01)` et `(DE, 2024-01-01)` arrivent après toutes les autres (les
  entités ne sont même plus contiguës). Sur une série simple, l'union de `DatetimeIndex` est triée, la sortie aussi ;
  `PanelwiseTransformer` + `ShiftTransformer` seuls rendent aussi une sortie triée par entité.
  `inverse_transform` restitue bien l'entrée (aller-retour exact).
- **Attendu** : lignes groupées par entité (ordre d'entrée), dates croissantes dans chaque entité — comme pour une
  série simple et comme `PanelwiseTransformer` ; un panel désordonné en sortie casse les composants en aval qui
  supposent des entités groupées et triées (`validate_entities_grouped`, `validate_sorted_within_groups`, splitters).
- **Reproduction** :
  ```python
  import numpy as np, pandas as pd
  from tsforecast.delays.transformers import PublicationDelayTransformer
  X = pd.DataFrame({'a': np.arange(6.), 'b': np.arange(6.)}, index=pd.date_range('2024-01-01', periods=6, freq='MS'))
  panel = pd.concat({'FR': X, 'DE': X}, names=['c', 'date'])
  out = PublicationDelayTransformer(delays={'a': 40.0}, prediction_date='2024-06-15', delay_unit='D',
                                    reference_point='start').fit_transform(panel)
  out.index[-2:].tolist()   # [('FR', 2024-01-01), ('DE', 2024-01-01)]
  ```
- **Test** : `tests/integration/delays/test_workflow.py::TestTwoDownloadsOnHeterogeneousPanel::test_output_is_ordered_by_entity_then_date`,
  `tests/unit/delays/transformers/test_publication_delay_transformer_transform.py::TestPanelOutputOrder`
- **Correctif** : `_assemble` trie la sortie d'un panel par entité (ordre d'apparition dans `X`), puis par date ; la
  sortie d'une série était déjà triée. L'aller-retour reste exact.
- **Statut** : corrigée

### ANO-DELAYS-044 — `PublicationDelayTransformer` dans un `XYPipeline` : `X` décalé n'est plus aligné sur `y`
- **Type** : [CODE] comportement (arbitré : composition de pipeline, hors classe)
- **Composant** : `tsforecast/delays/transformers.py::PublicationDelayTransformer.transform` (stratégie `'shift'`, et
  repli masque → décalage) dans `tsforecast/xy/pipeline.py::XYPipeline`
- **Sévérité** : majeure
- **Observé** : le décalage change les lignes de `X` (union des dates décalées, arbitrage du 2026-10-08) ;
  `PublicationDelayTransformer` n'est pas un transformateur XY, `y` traverse le pipeline inchangé. L'estimateur final
  reçoit donc `X` et `y` de longueurs différentes : `SpyEstimator` (sans contrôle) s'ajuste, un vrai estimateur
  lèverait ; au score, `cross_validate` lève `ValueError: Found input variables with inconsistent numbers of samples:
  [6, 8]` (pli de test de 6 mois de `irregular_index_timeseries`, 8 lignes après un décalage d'un et deux mois de
  l'inflation et du chômage ; `[6, 7]` dans la reproduction ci-dessous, inflation seule). Même échec avec
  `strategy='mask'` dès qu'une colonne bascule en décalage (« Could not mask the column 'pib_trimestriel' ... Moved
  it to the shifted columns »). Avec un masque seul (index inchangé), la chaîne fonctionne.
- **Attendu** : à trancher — utilisable avec `cross_validate` d'une manière documentée : soit `transform` restreint sa
  sortie à l'index de son entrée (les dates ajoutées n'ont pas de cible), soit un mode XY qui réaligne `y` sur les
  dates de `X`, soit une `ValueError` explicite / une mention dans la docstring orientant vers la stratégie `'mask'`.
  La finalité (préparer les variables d'une prévision évaluée en validation croisée temporelle) suppose au moins l'un
  des trois.
- **Reproduction** :
  ```python
  from sklearn.model_selection import cross_validate
  from tests.support.datasets import build_mixed_frequency_timeseries
  from tests.support.estimators import SpyEstimator
  from tsforecast.crossvals import TSOutOfSampleSplit
  from tsforecast.delays.transformers import PublicationDelayTransformer
  from tsforecast.xy import XYPipeline
  data = build_mixed_frequency_timeseries(annual_start_date='2015-01-01')
  X, y = data.drop(columns='production_industrielle'), data['production_industrielle']
  pipe = XYPipeline([('delays', PublicationDelayTransformer(delays={'inflation_ipc': 20.0}, prediction_date='2024-08-15',
                                                            delay_unit='D', reference_point='start')),
                     ('model', SpyEstimator())])
  cross_validate(pipe, X, y, cv=TSOutOfSampleSplit(n_splits=3, test_size=6), error_score='raise')  # ValueError
  ```
- **Test** : `tests/integration/delays/test_cross_validation.py::TestShiftedPipelineCrossValidation::test_scoring_without_a_realignment_step_fails`
- **Correctif** : arbitrage de l'auteur (2026-10-09) : **pas un défaut de la classe**. `transform` /
  `inverse_transform` respectent leur contrat (`X` en entrée, `X` en sortie, index agrandi par le décalage) et la
  sortie n'est pas restreinte à l'index d'entrée. Le désalignement naît de la composition dans un pipeline avec une
  cible : il sera traité par une étape ultérieure (transformateur XY réalignant `y` sur `X`), hors de
  `PublicationDelayTransformer`. Docstring de la classe et tutoriel (« Pipeline avec une cible ») le mentionnent ; le
  test épingle l'échec du score sans étape de réalignement.
- **Statut** : corrigée (arbitrée : comportement conservé, réalignement hors classe)

### Arbitrages de l'auteur (2026-10-04) sur `compare_and_detect_delays`
- **Fuseaux horaires** : une date sans fuseau est lue en UTC face à une date avec fuseau ; deux dates avec fuseaux sont
  comparées comme des instants ; la colonne `download_date` garde la date telle que fournie. (Auparavant : `TypeError`
  pandas sur un `download_date` aware avec des données naïves.) Tests : `TestDownloadDate::test_timezone_aware_*`,
  `::test_naive_download_date_with_timezone_aware_data`, `::test_aware_download_date_in_another_zone_than_the_data`.
- **Délai négatif** : légitime et conservé (une prévision utilisée comme observation d'une autre prévision anticipe sa
  valeur). Test : `TestReferencePointAndUnit::test_download_before_the_period_start_gives_a_negative_delay`.
- **Valeur retirée** (valeur dans `existing_data`, `NaN` dans `new_data`) : non signalée par aucun mode, faute de
  valeur à laquelle rattacher un délai. Test : `TestDetectionModes::test_withdrawn_value_is_not_reported`.
- **Révisions** (`all_changes`) : le point de référence est le même qu'en `new_only` (début ou fin de la période) ; le délai
  entre première publication et révision s'obtient par différence entre deux résultats. Test :
  `TestDetectionModes::test_all_changes_detects_revisions`.


### Arbitrages de l'auteur (2026-10-08) sur `PublicationDelayTransformer`
- **Index après décalage** : le décalage agrandit l'index (union des dates décalées des colonnes) et fait disparaître les
  dates qu'aucune colonne n'occupe plus — conservé. **Aller-retour** : `inverse_transform` supprime les lignes ajoutées
  par `transform` (dates hors de son entrée, entièrement vides une fois inversées), de sorte que
  `inverse_transform(transform(X))` restitue `X` exactement ; une ligne vide de l'entrée, ou une ligne non vide hors de
  l'entrée, est conservée. Tests : `test_publication_delay_transformer_transform.py::TestRoundTrip`.
- **Tableau de délais avec la variable dans l'index** : accepté (ANO-DELAYS-038).
- **Panel passé directement** : accepté avec un avertissement renvoyant vers la fabrique (ANO-DELAYS-030).
- **`inverse_transform` sans `transform`** : refusé, `NotFittedError` (ANO-DELAYS-037).

## FREQ

### ANO-FREQ-001 — `mark_*` d'un `Timestamp` absent de la matrice ajoute silencieusement une ligne
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/provenance.py::ImputationProvenanceTracker.mark_imputed`
  (et `mark_aggregated`, `mark_disaggregated`, `mark_interpolated`, `mark_model_imputed`)
- **Sévérité** : mineure
- **Observé** : `mark_imputed('a', pd.Timestamp('2030-01-01'), ...)` sur une date hors matrice agrandit la
  matrice d'une ligne (`.loc[label, col] = ...` en mode « setting with enlargement »), les autres colonnes de
  la ligne valant NaN. Un `DatetimeIndex` (même partiellement) hors matrice lève au contraire
  `KeyError: "None of [DatetimeIndex([...])] are in the [index]"`.
- **Attendu** : `KeyError` dans les deux cas. Le contrat de `extend_index` est que les lignes de la grille
  densifiée sont ajoutées par lui seul, avant tout `mark_*` ; l'agrandissement implicite contourne la règle
  « seuls les `extend_index` ajoutent des lignes » et ne trie pas la matrice.
- **Reproduction** :
  ```python
  tracker = ImputationProvenanceTracker().initialize(
      pd.DataFrame({'a': [1.0]}, index=pd.date_range('2023-01-01', periods=1)))
  tracker.mark_interpolated('a', pd.Timestamp('2030-01-01'))
  tracker.provenance_matrix_.shape   # (2, 1) au lieu d'un KeyError
  ```
- **Test** : `tests/unit/frequency/test_provenance.py::TestMarks::test_scalar_timestamp_outside_the_matrix_raises_key_error`
- **Statut** : corrigé (2026-10-09)
- **Correction** : `_check_labels_exist` (appelée par `mark_imputed` et `clear_provenance`) fait lever `KeyError` pour une étiquette scalaire ou une clé de `MultiIndex` absente ; plus de test `xfail` (`TestMarks`, `TestClearProvenance`).

### ANO-FREQ-002 — `merge` : aucune erreur quand les étiquettes ne se recoupent pas
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/provenance.py::ImputationProvenanceTracker.merge`
- **Sévérité** : à arbitrer
- **Observé** : seule la *forme* des deux matrices est contrôlée ; l'appariement se fait par étiquettes
  (`self.provenance_matrix_[mask] = other.provenance_matrix_[mask]`). Deux trackers de même forme mais de
  colonnes (ou de dates) disjointes fusionnent sans erreur ni effet.
- **Attendu** : indécidable — la docstring annonce une `ValueError` « si les matrices ont des formes
  incompatibles », ce qui peut se lire comme « étiquettes incompatibles ». `merge` n'est appelé nulle part
  dans `tsforecast/` : aucune source ne tranche. Comportement actuel épinglé.
- **Reproduction** : `a.merge(b)` avec `a` de colonnes `['a', 'b']` et `b` de colonnes `['x', 'y']`, mêmes dates.
- **Test** : `tests/unit/frequency/test_provenance.py::TestMerge::test_same_shape_but_disjoint_columns_is_a_silent_no_op`
  et `::test_same_shape_but_disjoint_index_is_a_silent_no_op`
- **Statut** : corrigé (2026-10-09) — arbitrage de l'auteur
- **Correction** : `merge(other, overwrite=True)` s'appuie sur `DataFrame.update` (le paramètre `how` disparaît : `'update'` → `overwrite=True`, `'preserve'` → `overwrite=False`) ; il lève `ValueError('Incompatible labels')` si l'index (ordre compris) ou l'ensemble des colonnes diffèrent, et n'ajoute jamais de cellule. Tests : `TestMerge`.

### ANO-FREQ-003 — `resolve_model_provenance` n'invalide pas ses arguments
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/provenance.py::resolve_model_provenance`
- **Sévérité** : à arbitrer
- **Observé** : `resolve_model_provenance('zz', 'zz')` renvoie `MODEL_ON_TRUE` ; une souillure inconnue (ou
  une faute de frappe) est traitée comme « indemne », alors que `origin_to_taint` et `max_origin` lèvent
  `KeyError` sur une origine inconnue.
- **Attendu** : indécidable — la spec (§6.3) donne l'implémentation « telle quelle », sans validation, ce
  qui fige ce comportement ; la cohérence avec les deux autres primitives plaiderait pour une erreur.
  Comportement actuel épinglé.
- **Reproduction** : `resolve_model_provenance('interpolate', 'none')` (faute de frappe pour `'interpolated'`)
  renvoie `MODEL_ON_TRUE`.
- **Test** : `tests/unit/frequency/test_provenance.py::TestResolveModelProvenance::test_unknown_taint_is_silently_treated_as_clean`
- **Statut** : corrigé (2026-10-09) — arbitrage de l'auteur
- **Correction** : `ValueError` nommant l'argument fautif (plutôt que `None`, qui serait écrit tel quel comme « non renseigné » dans la matrice par `mark_model_imputed`). Test : `TestResolveModelProvenance::test_unknown_taint_raises`.

### ANO-FREQ-004 — Représentation d'une cellule « non renseignée » : `NaN` ou `None`
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/frequency/provenance.py::ImputationProvenanceTracker.initialize`
- **Sévérité** : cosmétique
- **Observé** : la docstring d'`initialize` (texte et exemple `[<ProvenanceType.ORIGINAL...>, None, ...]`) et
  celle de `get_provenance` parlent de `None` pour une cellule non renseignée ; la matrice construite par
  `initialize` porte en réalité `NaN` (colonne `object` créée vide), alors que `extend_index` et
  `clear_provenance` écrivent bien `None`. Les lectures (`isna`, `isin`, `to_string_matrix`,
  `compute_statistics`) traitent les deux de la même façon : aucun effet fonctionnel.
- **Attendu** : le test suit le code (anomalie [DOC], pas de `xfail`) : les tests de lecture d'une cellule
  issue d'`initialize` utilisent `pd.isna`.
- **Reproduction** : `ImputationProvenanceTracker().initialize(df).get_provenance('a', ts_nan) is None` → `False`.
- **Test** : `tests/unit/frequency/test_provenance.py::TestGetProvenance::test_unfilled_cell_is_none`
- **Statut** : corrigé (2026-10-09)
- **Correction** : docstrings d'`initialize`, `extend_index` et `clear_provenance` alignées sur le code (cellule non renseignée = nulle, à tester avec `pd.isna`) ; le code n'a pas changé.

### ANO-FREQ-005 — `regularize` (mode global) réécrit une entité sur la fréquence d'une autre
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/regularizer.py::IndexRegularizer.regularize` (`per_entity=False`)
- **Sévérité** : à arbitrer
- **Observé** : sur un panel dont l'entité A est mensuelle et l'entité B trimestrielle (chacune avec un trou),
  `regularize(panel)` réindexe B sur la grille **mensuelle** : 5 trimestres deviennent 13 mois (`MS`), les
  9 mois ajoutés étant NaN. Aucune observation n'est perdue (4 valeurs avant, 4 après), mais la fréquence de B
  est modifiée. `per_entity=True` conserve `QS-OCT` pour B.
- **Attendu** : indécidable — la docstring de `per_entity` annonce « a single global frequency is used for
  every entity », ce qui décrit le comportement observé ; mais la finalité du composant (combler des trous) et
  `is_regular(per_entity=True)` (A et B « régulières ») plaident pour ne pas changer la fréquence d'une entité
  régulière. Comportement actuel épinglé.
- **Reproduction** : `regularize(pd.concat({'A': mensuelle_a_trou, 'B': trimestrielle_a_trou}))`.
- **Test** : `tests/unit/frequency/test_regularizer.py::TestRegularizeMixedPositionsAndFrequencies::test_global_mode_rewrites_the_quarterly_entity_on_the_monthly_grid`
- **Statut** : clos (2026-10-09) — arbitrage de l'auteur : le mode global est un alignement sur une fréquence unique, le comportement est voulu.
- **Correction** : docstrings de `regularize` précisées ; le test n'est plus une anomalie épinglée.

### ANO-FREQ-006 — `regularize` ne restitue pas `panel_cols` en colonnes
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/regularizer.py::IndexRegularizer.regularize`
- **Sévérité** : cosmétique
- **Observé** : avec `time_col='date', panel_cols=['country']`, le résultat a `date` en colonne mais `country`
  reste dans l'index (`reset_index(level=-1)` seulement) : la structure d'entrée (colonnes) n'est pas
  restituée, contrairement à ce que laisse attendre « restored as a column afterwards ».
- **Attendu** : indécidable — la docstring ne parle que de `time_col` ; l'entrée en colonnes et la sortie
  en index sont asymétriques. Comportement actuel épinglé.
- **Reproduction** : `regularize(df, time_col='date', panel_cols=['country']).index.names` → `['country']`.
- **Test** : `tests/unit/frequency/test_regularizer.py::TestRegularizePanel::test_panel_cols_and_time_col_are_restored_as_columns`
- **Statut** : corrigé (2026-10-09)
- **Correction** : `regularize` restitue `panel_cols` (et `time_col`) en colonnes ; `panel_cols` sans `time_col` conserve le `DatetimeIndex` comme niveau temporel. Tests : `TestRegularizePanel::test_panel_cols_and_time_col_are_restored_as_columns`, `::test_panel_cols_alone_keep_the_datetime_index`.

### ANO-FREQ-007 — `is_regular` : un index non trié mais complet est « irrégulier »
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/regularizer.py::IndexRegularizer.is_regular`
- **Sévérité** : cosmétique
- **Observé** : `is_regular(serie_melangee_sans_trou)` renvoie `False` (`pd.infer_freq` ne trie pas), alors que
  `regularize` trie d'abord et restitue une série sans trou. Même constat par entité pour un panel mélangé.
- **Attendu** : indécidable — la docstring définit la régularité par « no gaps » ; l'absence de tri ne
  devrait pas compter comme un trou, mais le contrat « `pd.infer_freq` renvoie non-None » le fige.
  Comportement actuel épinglé.
- **Reproduction** : `is_regular(pd.Series(range(6), index=dates_mensuelles).iloc[[3, 0, 5, 1, 4, 2]])`.
- **Test** : `tests/unit/frequency/test_regularizer.py::TestIsRegularTimeSeries::test_unsorted_but_complete_index_is_regular`
- **Statut** : corrigé (2026-10-09)
- **Correction** : l'index est trié avant `pd.infer_freq`. Tests : `TestIsRegularTimeSeries::test_unsorted_but_complete_index_is_regular`, `TestIsRegularPanel::test_unsorted_panel_is_regular_per_entity`.

### ANO-FREQ-008 — `regularize` détruit les observations hors grille
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/regularizer.py::IndexRegularizer.regularize` (`_regularize_ts`)
- **Sévérité** : majeure
- **Observé** : la fréquence est lue sur l'espacement modal, puis les données sont `reindex`ées sur
  `date_range(min, max, freq)`. Toute observation dont la date n'est pas sur cette grille est supprimée
  sans erreur ni avertissement. Cas minimal : 6 observations mensuelles dont une datée le 15 avril →
  la grille détectée est `ME` (fin de mois) et **les 6 observations disparaissent** (`count()` : 6 → 0).
  En mode global sur un panel, une entité sans fréquence détectable (3 dates à espacements 2 et 48 jours)
  est réindexée sur la grille mensuelle d'une autre entité et perd 1 observation sur 3 (la date non alignée)
  tout en gagnant une ligne NaN.
- **Attendu** : aucune observation non NaN ne disparaît silencieusement : soit l'entité / la série est rendue
  telle quelle (comme lorsque la fréquence est indétectable), soit une erreur est levée.
- **Reproduction** :
  ```python
  d = pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-15', '2020-05-01', '2020-06-01'])
  s = pd.Series(range(6), index=d, dtype=float)
  regularize(s).count()   # 0 au lieu de 6
  ```
- **Test** : `tests/unit/frequency/test_regularizer.py::TestOffGridObservations` et `::TestUndetectableFrequency`
- **Statut** : corrigé (2026-10-09) — arbitrage de l'auteur
- **Correction** : `regularize` lève `ValueError` quand aucune fréquence constante n'est détectable (série ou entité, y compris en mode global) ou quand aucune observation ne tombe sur la grille ; si seules certaines observations sont hors grille, elles sont écartées avec un `UserWarning` qui les liste. Les séries / entités à moins de deux dates sont rendues telles quelles. Tests : `TestOffGridObservations`, `TestUndetectableFrequency`.
- **Note** : le détecteur (`_detect_calendar_frequency`) perdait la position (`'M'` sans `S` / `E`) dès qu'une seule date sortait du motif ; au moins 75 % des dates en début (ou fin) de mois suffit désormais à la conserver (`'MS'`, `'QS-JAN'`, `'YS-JAN'` avec ancre prise sur les dates alignées). Reste sans position le cas où aucune position ne domine : `regularize` lève alors une `ValueError`. Tests : `tests/unit/utils/frequency/detector/test_time_series.py::TestFallbackWithOutlierDates`, `TestOffGridObservations::test_mid_month_outlier_keeps_the_month_start_grid`.

### ANO-FREQ-009 — `TargetFrequencyValidator.validate` : comportements permissifs
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/target_frequency_validator.py::TargetFrequencyValidator.validate`
- **Sévérité** : cosmétique
- **Observé** : (1) toute valeur de `on_frequency_mismatch` autre que `'error'` était traitée comme `'warn'` ; (2) une entité sans aucune fréquence détectée était écartée du dict renvoyé, avec un simple avertissement.
- **Attendu** : (1) `ValueError` pour une valeur non supportée ; (2) `ValueError` nommant l'entité.
- **Test** : `tests/unit/frequency/test_target_frequency_validator.py::TestOnFrequencyMismatchValues`, `::TestPanelEntityWithoutDetectedFrequency`
- **Statut** : corrigé (2026-10-09) — arbitrage de l'auteur
- **Correction** : validation du paramètre en tête de `validate` ; l'`except ValueError` qui convertissait l'erreur en avertissement est supprimé.

### ANO-FREQ-010 — Agrégation : les dates dupliquées sont sommées deux fois
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/frequency_aligner.py::FrequencyAligner.convert_to_target` (voie d'agrégation, `_aggregate_series`)
- **Sévérité** : mineure
- **Observé** : une série mensuelle dont janvier est dupliqué, agrégée en `'QS'`, donne T1 = 1+2+3+1 = **7** sans erreur ni
  avertissement (`full_periods_only` est satisfait : 4 valeurs ≥ 3 attendues), et la ligne dupliquée reçoit aussi la valeur
  agrégée. `_aggregate_series` appelle directement `FrequencyConverter.aggregate_to_lower_frequency`, qui ne contrôle pas les
  doublons.
- **Attendu** : `ValueError`, comme sur la voie d'interpolation du même `convert_to_target` (« Several observations fall in the
  same 'MS' period ») et comme `FrequencyConverter.convert_frequency` (test
  `tests/unit/utils/frequency/converter/test_conversion.py::TestSeriesEdgeCases::test_duplicated_dates_raise`) : un total gonflé en silence est
  pire qu'une erreur.
- **Reproduction** :
  ```python
  df = pd.DataFrame({'x': np.arange(1., 7.)}, index=pd.date_range('2023-01-01', periods=6, freq='MS'))
  df = pd.concat([df, df.iloc[:1]])
  FrequencyAligner().convert_to_target(df, ['x'], 'QS')['x'].iloc[0]   # 7.0
  ```
- **Test** : `tests/unit/frequency/test_frequency_aligner.py::TestAggregationContract::test_duplicated_dates_are_rejected`
- **Statut** : corrigé (2026-10-10)
- **Correction** : `_iter_variable_series` (voie commune à l'agrégation et à l'interpolation) lève `ValueError('Duplicate dates[ for entity …]: …')` dès qu'une entité portant une colonne à convertir a des dates dupliquées ; les entités sans clé ne sont pas contrôlées. Tests : `TestAggregationContract::test_duplicated_dates_are_rejected`, `::test_duplicated_dates_of_a_panel_entity_name_the_entity`, `::test_duplicates_of_an_entity_without_key_are_tolerated`, `TestConvertToTargetRouting::test_duplicated_dates_raise_on_interpolation`.

### ANO-FREQ-011 — Panel : une clé « nom de colonne » simple est ignorée en silence
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/frequency_aligner.py::FrequencyAligner.convert_to_target`
- **Sévérité** : à arbitrer
- **Observé** : sur un panel, `convert_to_target(panel, ['x'], 'QS')` renvoie le panel inchangé, sans erreur ni avertissement.
  `split_variable_key('x')` donne l'entité `()`, qu'aucun bloc d'entité du panel ne porte ; la clé est en outre orientée vers
  l'agrégation faute de fréquence trouvée (`freq_map` est indexé par tuples sur un panel).
- **Attendu** : à arbitrer entre (1) convertir la colonne pour **toutes** les entités et (2) lever une `ValueError` demandant des
  clés `(entité, colonne)`. La docstring annonce « column names or (entity..., variable) tuples » sans réserver les noms de
  colonnes aux séries temporelles ; le no-op silencieux n'est conforme à aucune des deux lectures. Le test `xfail` accepte
  l'une ou l'autre résolution.
- **Reproduction** : `convert_to_target(panel_mensuel_A_B, ['x'], 'QS').equals(panel_mensuel_A_B)` → `True`.
- **Test** : `tests/unit/frequency/test_frequency_aligner.py::TestConvertToTargetRouting::test_plain_column_name_on_a_panel_converts_every_entity`
- **Statut** : corrigé (2026-10-10) — arbitrage de l'auteur : conversion de toutes les entités, comme les autres fonctions du package
- **Correction** : `_expand_keys` remplace un nom de colonne seul par une clé `(entité..., colonne)` par entité du panel (dédoublonnée avec les clés tuples), avant l'orientation agrégation / interpolation : chaque entité garde sa propre fréquence source. Tests : `TestConvertToTargetRouting::test_plain_column_name_on_a_panel_converts_every_entity`, `::test_plain_and_tuple_keys_are_equivalent`, `::test_several_plain_column_names_on_a_panel`, `::test_plain_and_tuple_keys_for_the_same_column_are_merged`.

### ANO-FREQ-012 — `_interpolate_to_target` plante sur une colonne sans observation (garde morte)
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/frequency_aligner.py::FrequencyAligner._interpolate_series` / `_interpolate_to_target`
- **Sévérité** : cosmétique (inatteignable par l'API publique)
- **Observé** : `_interpolate_series` renvoie `None` pour une série sans observation (annotation `-> pd.Series` pourtant) ;
  `_interpolate_to_target` range ce `None` dans `interpolated`, puis `build_densified_index` lève
  `AttributeError: 'NoneType' object has no attribute 'index'`. Avec une seule observation, `_interpolate_series` lève
  `ValueError: Series has only 1 non-null observations` (appel non protégé à `detect_frequency`), alors que `_observed_series`
  prévoit justement ce repli. `convert_to_target` oriente toujours ces colonnes vers l'agrégation : la garde n'est atteinte que par
  appel direct de la méthode privée.
- **Attendu** : colonne laissée telle quelle, comme `_aggregate_to_target` le fait pour une colonne vide (`continue` sur `None`) —
  ou suppression de la garde morte. Par ailleurs, `_observed_series` est sans effet observable : le convertisseur restreint
  lui-même aux lignes observées (`dropna(how='all')`) et reçoit la fréquence source explicitement.
- **Reproduction** :
  ```python
  df = pd.DataFrame({'x': [np.nan] * 3}, index=pd.date_range('2023-01-01', periods=3, freq='QS'))
  FrequencyAligner()._interpolate_to_target(df, ['x'], 'MS')   # AttributeError
  ```
- **Test** : `tests/unit/frequency/test_frequency_aligner.py::TestPrivateGuards::test_interpolating_a_column_without_frequency_leaves_it_unchanged`
- **Statut** : corrigé (2026-10-10)
- **Correction** : `_interpolate_series` renvoie `None` sous deux observations (annotation `Optional[pd.Series]`) et `_interpolate_to_target` laisse alors la colonne telle quelle ; `_observed_series`, sans effet observable, est supprimée (le convertisseur restreint lui-même aux lignes observées).

### ANO-FREQ-013 — Documentation : `FrequencyAligner` présenté comme brique du `HighFrequencyImputer`
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/frequency/frequency_aligner.py` (docstring de module et de classe),
  `tsforecast/utils/frequency/converter.py` (docstring de module)
- **Sévérité** : cosmétique
- **Observé** : la docstring du module annonce « FrequencyAligner builds homogeneous-frequency datasets for the
  HighFrequencyImputer », celle de `converter.py` renvoie vers lui « as required by the HighFrequencyImputer » ; or
  `HighFrequencyImputer` ne l'importe plus depuis `2e6e595` (suppression de HFI v1). Les exemples des docstrings (classe,
  `_interpolate_to_target`) et la description du module documentent des méthodes **privées** (`_aggregate_to_target`,
  `_interpolate_to_target`) plutôt que `convert_to_target`.
- **Attendu** : outil autonome d'alignement, exemples sur `convert_to_target` / `build_densified_index`. Pas de `xfail`
  (le code est juste).
- **Test** : `tests/unit/frequency/test_frequency_aligner.py` (docstring de module)
- **Statut** : corrigé (2026-10-10)
- **Correction** : docstrings du module, de la classe et de `convert_to_target` réécrites (outil autonome, exemples sur `convert_to_target` / `build_densified_index`), docstring de module de `converter.py`, ligne du tableau de `docs/concepts/mixed_frequency_imputation.md` et `CLAUDE.md`. Restent périmés (non modifiés) : `notebooks/utils/frequency_aligner.ipynb` (appels privés avec l'argument `is_panel` supprimé en `6db1ca7`), textes de `notebooks/0 - QB - Utils.ipynb` et `notebooks/explore_frequency_converter.ipynb`.

### Arbitrages de l'auteur (2026-10-10) sur `FrequencyAligner`
- **Position de la cible** : une position explicite (`'QE'`, `'MS'`) est honorée ; une cible sans position (`'Q'`, `'M'`) prend
  celle de l'index source (`_resolve_target_offset`). Quand la position explicite diffère de celle de la source, les dates cibles
  absentes sont **ajoutées** à l'index (agrégation comprise) et la colonne convertie ne vit que sur la grille cible (NaN à ses
  anciennes dates) ; les autres colonnes sont intactes. L'agrégation ne garantit plus « même index » qu'à position identique ;
  les labels de périodes incomplètes (NaN) n'étendent jamais l'index. Tests : `TestPositionCoherence`.
- **Méthode d'agrégation** : paramètre `agg_method` de `convert_to_target` (défaut `'sum'`), toutes les valeurs de
  `AggregationMethod` ; `'all'` / `'any'` écrivent des booléens dans une colonne `object`. Tests : `TestAggregationMethods`.
- **Bords de l'interpolation** : dépendent de `interp_method` — `'linear'`, `'time'`, `'index'`, `'values'` maintiennent la valeur
  de bord (dans la limite), les méthodes scipy laissent NaN. Documenté, pas de changement. Tests : `TestInterpolationMethodEdges`.
- **Axe temporel non daté** : `TypeError` explicite (piste `to_timestamp` pour un `PeriodIndex`). Tests :
  `TestConvertToTargetRouting::test_non_datetime_time_series_index_raises` et suivants.

### ANO-FREQ-014 — Grille ancrée en fin de période quand aucune colonne n'est à la fréquence de l'index
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/imputation_window.py::ImputationWindowCalculator._build_index_freq_grid`
- **Sévérité** : majeure
- **Observé** : la position (début / fin) de la grille de couverture est lue sur la première colonne dont la fréquence
  détectée égale celle de l'index. Sans telle colonne — index mensuel `MS` ne portant que des séries trimestrielles et
  annuelles, cas d'usage direct d'une imputation vers une grille plus fine —, la position retombe sur `'E'` : la grille
  est `ME` alors que les données sont `MS`. Les bornes et le masque brut sont en fin de mois, et
  `get_imputation_window_mask(data)` renvoie **False sur toutes les lignes** (0 / 24) alors que la fenêtre stricte couvre
  tout le jeu. Même défaut pour `get_mask_at_frequency`, ancré sur l'index du masque.
- **Attendu** : la position de la grille suit celle de l'index des données (ou, à défaut de colonne haute fréquence,
  la position des colonnes observées) ; le masque aligné vaut True sur les 24 mois. Avec un index `ME`, le défaut
  ne se voit pas (repli `'E'` coïncident).
- **Reproduction** :
  ```python
  dates = pd.date_range('2020-01-01', periods=24, freq='MS')
  df = pd.DataFrame({'q': np.nan, 'y': np.nan}, index=dates)
  df.loc[dates.month.isin([1, 4, 7, 10]), 'q'] = 1.0
  df.loc[dates.month == 1, 'y'] = 2.0
  calc = ImputationWindowCalculator().fit(df)
  calc.imputation_strict_window_start_          # Timestamp('2020-01-31')
  calc.get_imputation_window_mask(df).sum()     # 0
  ```
- **Test** : `tests/unit/frequency/test_imputation_window.py::TestGridAnchoring::test_grid_follows_the_index_when_every_column_is_coarser`
- **Statut** : corrigé (2026-10-11)
- **Correction** : nouvelle méthode `_detect_grid_position` : la position est lue sur les dates observées des colonnes à la
  fréquence de la grille, puis sur l'index de l'entité, puis sur les dates observées des autres colonnes de fréquence
  détectée ; `'E'` ne reste qu'un dernier recours. Tests : `TestGridAnchoring::test_grid_follows_the_index_when_every_column_is_coarser`
  (`MS` et `ME`), `::test_panel_entity_grid_follows_its_own_index`.

### ANO-FREQ-015 — `fit` plante sur une colonne observée une seule fois
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/imputation_window.py::ImputationWindowCalculator._build_coverage_matrix`
- **Sévérité** : majeure
- **Observé** : depuis ANO-UTILS-042, une colonne (ou un couple (entité, colonne)) observée une seule fois a une fréquence
  détectée `None`. `_build_index_freq_grid` se replie alors sur la fréquence de l'index (`col_freqs.get(col) or
  index_freq`), mais `_build_coverage_matrix` lit `col_freqs[col]` sans repli et passe `None` à `get_period_start` :
  `ValueError: Frequency must be a string, got <class 'NoneType'>`. Sur un panel, une seule entité dans ce cas fait
  échouer le `fit` de tout le panel ; `HighFrequencyImputer._fit_imputation_window` rattrape la `ValueError` et
  poursuit **sans fenêtre**, silencieusement.
- **Attendu** : pas d'exception ; traitement cohérent entre la grille et la matrice — soit le même repli sur la
  fréquence de l'index, soit (plus cohérent avec `_undetected_frequencies_` de l'imputeur et D39) une exclusion du
  dénominateur, signalée. Le test `xfail` n'exige que l'absence d'exception.
- **Reproduction** :
  ```python
  dates = pd.date_range('2020-01-01', periods=12, freq='MS')
  df = pd.DataFrame({'a': np.arange(12.), 'b': np.arange(12.),
                     'c': [np.nan] * 5 + [1.0] + [np.nan] * 6}, index=dates)
  ImputationWindowCalculator().fit(df)   # ValueError: Frequency must be a string, got <class 'NoneType'>
  ```
- **Test** : `tests/unit/frequency/test_imputation_window.py::TestUndetectedFrequencyColumns`
- **Statut** : corrigé (2026-10-11) — arbitrage de l'auteur : exclusion (décision D40, spec §14.8)
- **Correction** : `_compute_window` recense les colonnes observées de fréquence `None` ; `_build_index_freq_grid` et
  `_build_coverage_matrix` les ignorent (plus de repli sur la fréquence de l'index), le dénominateur les écarte. Nouvel
  attribut ajusté `undetected_frequency_columns_` (même structure que `structurally_absent_columns_`), `UserWarning` agrégé
  propre, couverture rendue `(None, None)`. La branche « toutes colonnes absentes » de la couverture, devenue morte, est
  supprimée. Tests : `TestUndetectedFrequencyColumns` (8 tests : veto, rapport, avertissement unique, grille non étirée,
  espacement irrégulier, panel par entité, `get_columns_with_coverage`, série sans aucune fréquence).

### ANO-FREQ-016 — `imputation_scope='unrestricted'` : prévu par la spec, refusé par le code
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/imputation_window.py::ImputationScope`, `ImputationWindowCalculator.__init__`
- **Sévérité** : à arbitrer
- **Observé** : `ImputationScope` n'a que quatre valeurs ; `ImputationWindowCalculator(imputation_scope='unrestricted')`
  lève `ValueError`. `'unrestricted'` n'existe que dans `TrainingScope` (commit `44f5266`, commentaire explicite :
  « les quatre valeurs d'ImputationScope, plus 'unrestricted' »).
- **Attendu** : la spec §7 écrit `ImputationScope = Literal[..., 'extended_both', 'unrestricted']` et
  `TrainingScope = ImputationScope`. Une fenêtre de **prédiction** sans restriction (imputer des dates sans aucune
  donnée) a peu de sens : la spec est probablement à corriger plutôt que le code. Comportement actuel épinglé,
  sans `xfail`.
- **Reproduction** : `ImputationWindowCalculator(imputation_scope='unrestricted')` → `ValueError`.
- **Test** : `tests/unit/frequency/test_imputation_window.py::TestConstructorValidation::test_imputation_scope_rejects_unrestricted`
- **Statut** : corrigé (2026-10-11) — arbitrage de l'auteur : `'unrestricted'` n'a d'intérêt que pour `TrainingScope`
- **Correction** : la spec est corrigée (§7 : `ImputationScope` à quatre valeurs, `TrainingScope` explicite, paragraphe de
  motif) ; le code, déjà conforme, est inchangé. Le test épingle désormais le contrat.

### ANO-FREQ-017 — Entité sans aucune observation : `structurally_absent_columns_` vide
- **Type** : [DOC] docstring ≠ code (ou [CODE], selon l'arbitrage)
- **Composant** : `tsforecast/frequency/imputation_window.py::ImputationWindowCalculator._fit_panel` /
  `_compute_window`
- **Sévérité** : à arbitrer
- **Observé** : une entité d'index régulier dont toutes les colonnes sont NaN a une grille vide ; `_compute_window`
  renvoie `absent_columns=()` avant de calculer la matrice. L'entité figure dans `entities_without_window_`, mais
  `structurally_absent_columns_[entité] == ()`, sans avertissement. Même `()` pour une entité dont la fréquence
  d'index est indéterminée (là, c'est défendable : rien n'a été examiné).
- **Attendu** : la docstring réserve le tuple vide à « an entity without any such column » ; or toutes les colonnes de
  l'entité sont sans observation. Soit lister toutes ses colonnes (cohérent avec la définition), soit préciser dans la
  docstring que l'attribut ne couvre que les entités dotées d'une grille. Comportement actuel épinglé.
- **Reproduction** :
  ```python
  dates = pd.date_range('2020-01-01', periods=12, freq='MS')
  idx = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
  values = list(range(12)) + [np.nan] * 12
  calc = ImputationWindowCalculator().fit(pd.DataFrame({'a': values, 'b': values}, index=idx, dtype=float))
  calc.structurally_absent_columns_   # {('A',): (), ('B',): ()}
  ```
- **Test** : `tests/unit/frequency/test_imputation_window.py::TestEntitiesWithoutWindow::test_empty_entity_reports_no_excluded_column`
- **Statut** : corrigé (2026-10-11) — arbitrage de l'auteur : docstring précisée, code inchangé
- **Correction** : la docstring de `structurally_absent_columns_` (et de `undetected_frequency_columns_`) énonce la règle :
  l'attribut n'a de sens que pour une entité dotée d'une grille ; une entité de `entities_without_window_` porte toujours un
  tuple vide. Elle cesse aussi d'affirmer qu'une série temporelle porte toujours `()` (faux depuis ANO-UTILS-042). Règle
  reprise dans D39 (spec §14.8).

### ANO-FREQ-018 — `get_mask_at_frequency` : entités sans fenêtre traitées de deux façons
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/imputation_window.py::ImputationWindowCalculator.get_mask_at_frequency`
- **Sévérité** : mineure
- **Observé** : une entité dont la fréquence d'index est indéterminée est **absente** du résultat ; une entité d'index
  régulier sans aucune observation (grille vide) y figure **à False**. Or l'imputeur (`_window_mask_at`) et
  `TrainingSetBuilder` lisent une entité absente comme **non restreinte** et une entité à False comme entièrement hors
  fenêtre : deux entités sans fenêtre recevaient deux traitements opposés.
- **Attendu** : une seule règle pour `entities_without_window_`. La docstring de la méthode (« Entities without a valid
  fitted mask are simply absent ») et celle de l'imputeur (« An entity the calculator omits — one without a valid fitted
  mask — is left unrestricted ») désignent l'absence.
- **Reproduction** : panel `A` complet, `B` entièrement NaN sur un index mensuel régulier ;
  `calc.get_mask_at_frequency('QS').loc['B']` → 4 × `False`.
- **Test** : `tests/unit/frequency/test_imputation_window.py::TestEntitiesWithoutWindow::test_entity_without_window_is_absent_from_mask_at_frequency`
- **Statut** : corrigé (2026-10-11)
- **Correction** : la boucle par entité saute toute entité de `entities_without_window_` ; un `frequency` en dict n'a plus à
  nommer ces entités (`::test_dict_target_need_not_name_an_entity_without_window`). Les masques ajustés gardent leurs lignes
  à False (§7.2). Suites `frequency` et `tracking` inchangées (1 108 tests).

### ANO-FREQ-019 — `get_imputation_window_mask(data)` : `FutureWarning` de pandas
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/imputation_window.py::ImputationWindowCalculator.get_imputation_window_mask`
- **Sévérité** : cosmétique
- **Observé** : `source.reindex(data.index).fillna(False).astype(bool)` passe par un tableau `object` dès qu'une ligne de
  `data` manque à la grille : `FutureWarning: Downcasting object dtype arrays on .fillna ... is deprecated`.
- **Attendu** : aucun avertissement de dépréciation ; résultat booléen identique.
- **Reproduction** : `calc.get_imputation_window_mask(pd.DataFrame(index=index_avec_une_date_hors_grille))`.
- **Test** : `tests/unit/frequency/test_imputation_window.py::TestGetImputationWindowMask::test_alignment_emits_no_pandas_deprecation`
- **Statut** : corrigé (2026-10-11)
- **Correction** : `source.reindex(data.index, fill_value=False).astype(bool)`.

### ANO-FREQ-020 — Entité sans fenêtre écartée en silence
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/imputation_window.py::ImputationWindowCalculator._fit_panel`
- **Sévérité** : mineure
- **Observé** : une entité sans fenêtre (fréquence d'index indéterminée, ou aucune observation) voit toutes ses lignes à
  False dans les trois masques sans aucun avertissement — alors que l'exclusion d'une simple colonne (D39) en émet un.
- **Attendu** : un avertissement agrégé unique nommant ces entités (point de vigilance 5 du §17 de la spec : tout
  avertissement est agrégé et unique).
- **Reproduction** : panel `A` complet, `B` entièrement NaN ; `fit` ne produit aucun avertissement.
- **Test** : `tests/unit/frequency/test_imputation_window.py::TestEntitiesWithoutWindow::test_entities_without_window_are_named_by_one_warning`
- **Statut** : corrigé (2026-10-11)
- **Correction** : `_fit_panel` émet un `UserWarning` unique listant `entities_without_window_` ; un panel où toutes les
  entités ont une fenêtre reste silencieux (`::test_no_warning_when_every_entity_has_a_window`).

### ANO-FREQ-021 — Agrégat prédit quasi nul : le recalage explose et la somme n'égale plus le total
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/aggregation_constraint.py::AggregationConstraint._rescale_one` (via `rescale` / `transform`)
- **Sévérité** : mineure
- **Observé** : la garde « agrégat prédit nul » (§11.1) teste l'égalité **exacte** `aggregate == 0`. Des prédictions
  qui s'annulent mathématiquement mais pas en flottant passent la garde : `[0.1, 0.2, -0.3]` (agrégat `5.55e-17`) face à
  un total trimestriel de 12 donnent un ratio de `2.2e17`, des sous-périodes de l'ordre de `6e16`, et une somme recalée de
  **16** au lieu de 12 — l'invariant I4 (additivité, `1e-9`) est rompu sans avertissement. Mesures complémentaires
  (12 mois de signes alternés, total 78) : agrégat `1e-6` → écart `6e-8` ; `1e-9` → écart `6e-5`. Sur 2 000 tirages
  aléatoires sans compensation, l'écart maximal reste `2.3e-10`.
- **Attendu** : soit la période reste brute avec l'avertissement « aggregate to zero » (agrégat **numériquement** nul,
  p. ex. `|Σ| ≤ k·ε·Σ|x|`), soit sa somme est exacte à `1e-9`. Le seuil exact est à arbitrer par l'auteur ; une colonne
  oscillant autour de zéro (une balance commerciale mensuelle) rend le cas plausible.
- **Reproduction** :
  ```python
  grid = pd.date_range('2021-01-31', periods=3, freq='ME')
  values = pd.Series([0.1, 0.2, -0.3], index=grid)
  observations = pd.Series([np.nan, np.nan, 12.0], index=grid)
  rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Q', grid_freq='M')
  rescaled.sum()   # 16.0, mask tout à True
  ```
- **Test** : `tests/unit/frequency/test_aggregation_constraint.py::TestRescaleGuards::test_numerically_zero_aggregate_is_zero`
- **Statut** : corrigé (2026-10-11) — arbitrage de l'auteur
- **Correction** : `_rescale_one` traite comme nul tout agrégat dont la valeur absolue est inférieure à la borne d'erreur
  d'arrondi de la sommation, `n · ε · Σ|x|` (`np.finfo(float).eps`) : la période reste brute, avec l'avertissement agrégé
  « aggregate to zero ». La borne est relative : douze prédictions de `1e-20` restent recalées. Tests :
  `TestRescaleGuards::test_numerically_zero_aggregate_is_zero`, `::test_tiny_same_sign_predictions_are_still_rescaled`.
  Limite résiduelle (précision double, non corrigeable par un seuil) : un agrégat petit mais non nul (`1e-6` face à un
  total de 78) donne des sous-périodes de l'ordre de `1e8`, dont la somme ne peut être exacte qu'à `~n · ε · Σ|x_recalé|`
  près (`6e-8` mesuré). À amender au §11.1 de la spec (prompt R9) : « agrégat prédit **numériquement** nul ».

### ANO-FREQ-022 — Dates d'observation dupliquées : le total visé est compté deux fois
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/aggregation_constraint.py::AggregationConstraint._observed_totals` (via `rescale` / `transform`)
- **Sévérité** : mineure
- **Observé** : une ancre dupliquée (deux lignes `2021-12-31` à 120) est sommée : la cible de 2021 devient **240** et
  l'année est recalée dessus, sans erreur ni avertissement (la garde de troncature est satisfaite : 13 lignes ≥ 12
  attendues). `anchor_cells_mask` lève au contraire `ValueError: cannot reindex on an axis with duplicate labels`.
- **Attendu** : `ValueError` nommant les dates dupliquées, comme `FrequencyAligner` après ANO-FREQ-010 et
  `FrequencyConverter.convert_frequency` : un total gonflé en silence est pire qu'une erreur. `HighFrequencyImputer`
  n'est pas exposé (validation stricte en amont : `validate_temporal_data` lève sur un index dupliqué) ; seul l'usage
  autonome du composant l'est.
- **Reproduction** :
  ```python
  grid = pd.date_range('2021-01-31', periods=12, freq='ME')
  dup = grid.append(grid[-1:])
  values = pd.Series(np.arange(1.0, 14.0), index=dup)
  observations = pd.Series(np.nan, index=dup); observations.iloc[-2:] = 120.0
  AggregationConstraint('sum').rescale(values, observations, 'Y', grid_freq='M')[0].sum()   # 240.0
  ```
- **Test** : `tests/unit/frequency/test_aggregation_constraint.py::TestRescaleGuards::test_duplicated_observation_dates_are_rejected`
- **Statut** : corrigé (2026-10-11)
- **Correction** : `_check_unique_dates` lève `ValueError('Duplicate dates in observations|values: [...]')` dans
  `_rescale_one` (ancres non nulles des observations, et grille des valeurs : une sous-période dupliquée compterait deux
  fois dans l'agrégat) et dans `anchor_cells_mask` (même message que le recalage, au lieu de l'erreur de `reindex`).
  Tests : `TestRescaleGuards::test_duplicated_observation_dates_are_rejected`, `::test_duplicated_grid_dates_are_rejected`,
  `::test_duplicated_panel_pairs_are_rejected`, `TestAnchorCellsMask::test_duplicated_observation_dates_raise`.

### ANO-FREQ-023 — `set_params` contourne la validation d'`aggregation_constraint` (D20)
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/aggregation_constraint.py::AggregationConstraint.fit`
- **Sévérité** : mineure
- **Observé** : la validation n'a lieu qu'à `__init__`. `set_params(aggregation_constraint='mean')` (chemin de
  `GridSearchCV`, des pipelines) l'esquive ; `fit` ne revalide pas, et `transform` traite `'mean'` comme une contrainte
  active, c'est-à-dire comme `'sum'` (`resolve_constraint(...) is not None`). Un agrégateur retiré de l'API par D20 est
  donc accepté en silence et produit un résultat d'une autre nature que celle demandée.
  `HighFrequencyImputer` n'est pas exposé : il reconstruit le composant à chaque `fit`, ce qui revalide.
- **Attendu** : `fit` revalide le réglage (`validate_aggregation_constraint`) et lève le même `ValueError` qu'`__init__`
  (B3 : la normalisation a lieu au `fit`).
- **Reproduction** :
  ```python
  grid = pd.date_range('2021-01-31', periods=12, freq='ME')
  X = pd.DataFrame({'a': np.arange(1.0, 13.0)}, index=grid)
  obs = pd.DataFrame({'a': [np.nan] * 11 + [78.0]}, index=grid)
  c = AggregationConstraint('sum', 'Y', obs).set_params(aggregation_constraint='mean')
  c.fit(X).transform(X)['a'].sum()   # 78.0, aucune erreur
  ```
- **Test** : `tests/unit/frequency/test_aggregation_constraint.py::TestValidationThroughSetParams`
- **Statut** : corrigé (2026-10-11) — arbitrage de l'auteur (composants utilisables seuls, `set_params` compris)
- **Correction** : `fit`, `transform` et `rescale` (utilisable sans `fit`) revalident `aggregation_constraint` par
  `validate_aggregation_constraint` ; la validation d'`__init__` est conservée (§13.1). Tests : `TestValidationThroughSetParams`.

### ANO-FREQ-024 — Garde de troncature désactivée en silence
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/aggregation_constraint.py::AggregationConstraint._expected_subperiods`
- **Sévérité** : à arbitrer
- **Observé** : la garde de troncature calendaire (§11.1) se désactive sans signal dans deux cas : (1) un `grid_freq`
  illisible (`'zz'`) — le `ValueError` de `normalize_frequency` est absorbé par le `except (ValueError, TypeError)`
  prévu pour les couples de fréquences sur lesquels pandas ne sait pas itérer ; (2) un `grid_freq` en dict qui ne nomme
  pas une entité (y compris une clé `('FR',)` face à un index à trois niveaux, dont la clé d'entité est
  `('Zone euro', 'FR')`). Dans les deux cas une période tronquée reçoit tout son total — la sur-attribution que la garde
  devait empêcher.
- **Attendu** : indécidable — la docstring annonce bien `None` « when it cannot be computed », mais la spec ne prévoit
  la désactivation que pour un appelant qui ne passe **pas** `grid_freq`. Pistes : `ValueError` sur une fréquence
  illisible, et sur une entité absente du dict. Comportement actuel épinglé.
- **Reproduction** :
  ```python
  grid = pd.date_range('2021-10-31', periods=3, freq='ME')
  values = pd.Series(1.0, index=grid)
  observations = pd.Series([np.nan, np.nan, 120.0], index=grid)
  AggregationConstraint('sum').rescale(values, observations, 'Y', grid_freq='zz')[1].all()   # True
  ```
- **Test** : `tests/unit/frequency/test_aggregation_constraint.py::TestGridFrequencyValidation`
- **Statut** : corrigé (2026-10-11) — arbitrage de l'auteur
- **Correction** : `_resolve_grid_freq`, appelée une fois par appel en tête de `_rescale_one`, normalise `grid_freq` et lève
  `ValueError` sur une fréquence illisible (« grid_freq must be a readable frequency ») et sur un dictionnaire qui ne couvre
  pas toutes les entités des valeurs (« grid_freq does not cover entities [...] » ; une série temporelle est l'entité `()`).
  Les clés du dictionnaire sont normalisées (`'FR'` ≡ `('FR',)`) ; des clés d'entités absentes sont tolérées.
  `_expected_subperiods` ne rattrape plus aucune exception. `None` désactive toujours la garde. Appelants internes non
  concernés : `HighFrequencyImputer` et `CovariateMaterializer` passent une fréquence scalaire d'étape. Tests :
  `TestGridFrequencyValidation`.

### ANO-FREQ-025 — Une ligne datée `NaT` forme son propre bloc de recalage
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/aggregation_constraint.py::_period_membership`
- **Sévérité** : cosmétique
- **Observé** : `to_period` rend `NaT` pour une date absente, et `_is_usable_key(NaT)` vaut `True` : toutes les lignes `NaT`
  forment une « période » commune, et une observation datée `NaT` y devient un total imposé. La garde `_is_usable_key` était
  du code mort (aucune clé `None` / NaN n'était jamais produite).
- **Attendu** : une ligne `NaT` n'appartient à aucune période, ni recalée ni source de total.
- **Reproduction** : grille de 12 mois + une ligne `NaT` portant 7 et une observation à 5 → la ligne `NaT` était recalée à 5.
- **Test** : `tests/unit/frequency/test_aggregation_constraint.py::TestRescaleGuards::test_nat_rows_never_form_a_period`
- **Statut** : corrigé (2026-10-11)
- **Correction** : `_period_membership` rend `None` pour une date `NaT` (série et panel) ; `_is_usable_key` écarte ces lignes
  dans `_observed_totals` et `_rescale_one`.

### ANO-FREQ-026 — `transform` accepte d'autres colonnes que celles du `fit`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/aggregation_constraint.py::AggregationConstraint.transform`
- **Sévérité** : mineure
- **Observé** : aucune comparaison avec `feature_names_in_` ; une colonne inconnue au `fit` est recalée sans que ses clés de
  dictionnaire ni ses métadonnées aient été contrôlées, une colonne manquante passe inaperçue.
- **Attendu** : convention sklearn, `ValueError` nommant les colonnes manquantes et inattendues.
- **Reproduction** : `AggregationConstraint('sum', 'Y', obs).fit(X).transform(X.assign(q1=1.0))` ne levait rien.
- **Test** : `tests/unit/frequency/test_aggregation_constraint.py::TestSklearnProtocol::test_transform_rejects_other_columns_than_fit`
- **Statut** : corrigé (2026-10-11)
- **Correction** : `ValueError('X has different columns than during fit: missing [...], unexpected [...]')` ; l'ordre des
  colonnes reste libre (`::test_transform_accepts_reordered_columns`).

### ANO-FREQ-027 — `fit` ne purge pas `rescaled_mask_`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/aggregation_constraint.py::AggregationConstraint.fit`
- **Sévérité** : cosmétique
- **Observé** : après `fit_transform(X)` puis `fit(X_autre)`, `rescaled_mask_` porte encore le masque du transform
  précédent, sur une autre grille.
- **Attendu** : principe B19 (§12.5) — l'état du `transform` est purgé en tête de `fit`.
- **Reproduction** : `c.fit_transform(X); c.fit(X2); c.rescaled_mask_.index` → index de `X`.
- **Test** : `tests/unit/frequency/test_aggregation_constraint.py::TestSklearnProtocol::test_refit_purges_the_previous_rescaled_mask`
- **Statut** : corrigé (2026-10-11)
- **Correction** : `fit` supprime `rescaled_mask_` s'il existe.

### ANO-FREQ-028 — `'calendar'` : diviseur `1.0` au lieu d'un diviseur fractionnaire quand l'étape est plus grossière
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/stage_scaler.py::StageScaler._pair_divisor` (donc `target_divisor`,
  `fit_scale_factor`, `feature_divisors`)
- **Sévérité** : majeure
- **Observé** : sous `'constant'`, un couple « étape plus grossière que la variable » donne bien `1/3` (M -> Q),
  `1/4` (Q -> Y), `1/12` (M -> Y). Sous `'calendar'`, les mêmes couples rendent `1.0` sur chaque ligne :
  `count_subperiods_per_period(dates, low_freq='M', high_freq='Q')` compte les trimestres qui *recouvrent* un mois
  (toujours un), pas la fraction de trimestre qu'il représente.
- **Attendu** : §5.8 R5 — diviseur « **> 1** pour un bloc plus grossier que l'étape et **< 1** pour un bloc plus
  fin » ; §9.2 — la modalité s'applique à **tous** les diviseurs. Sur un jeu mutualisé en `'calendar'` (bloc `IT`
  mensuel à l'étape `Q`), les valeurs mensuelles entrent sans être ramenées à l'échelle trimestrielle : 10 au lieu
  de 30, le mélange d'échelles que la mutualisation devait supprimer (B29). Convention à arbitrer : `1/3` uniforme
  ou part de jours (`31/91`) ; dans les deux cas les sous-périodes d'une période d'étape somment à 1 — c'est la
  propriété testée.
- **Reproduction** :
  ```python
  idx = pd.date_range('2024-01-31', periods=3, freq='ME')
  StageScaler(scale_features='calendar').target_divisor('v', 'M', 'Q', index=idx).tolist()
  # [1.0, 1.0, 1.0] ; 'constant' donne 1/3
  ```
- **Test** : `tests/unit/frequency/test_stage_scaler.py::TestCalendarGoldenValues::test_fractional_divisor_under_calendar`,
  `::test_fractional_feature_divisor_under_calendar`
- **Statut** : corrigé (2026-10-11) — arbitrage de l'auteur : part de jours en `'calendar'`, fraction uniforme en `'constant'`
- **Correction** : `_pair_divisor` détecte une période portée plus fine que l'étape (`is_higher_frequency(f_to, f_from)`) et, sous `'calendar'`, calcule la part de temps réelle de la période d'étape (`_calendar_share` : durée de la période fine / durée de la période d'étape, en nanosecondes ; repli sur le facteur constant pour une fréquence sans `Period` pandas, ex. `SM`). `'constant'` est inchangé (1/3, 1/4, 1/12). Tests : `tests/unit/frequency/test_stage_scaler.py::TestCalendarGoldenValues::test_calendar_share_of_days_gold`, `::test_calendar_share_on_a_mutualized_block`, `::test_fractional_divisor_under_calendar`.

### ANO-FREQ-029 — `feature_divisors` : `grid_freq` par entité rend une `Series` quand les blocs coïncident
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/stage_scaler.py::StageScaler.feature_divisors`
- **Sévérité** : mineure
- **Observé** : avec `grid_freq` en liaison par entité, le retour est une `DataFrame` par ligne si les diviseurs
  diffèrent d'une entité à l'autre, mais une `Series` par colonne s'ils coïncident (blocs identiques, liaison à une
  seule entrée, colonnes toutes en `'raw_anchors'`…). La forme dépend des **valeurs** calculées ;
  `HighFrequencyImputer._feature_divisors_per_row` doit d'ailleurs traiter les deux formes.
- **Attendu** : §9.2 — `feature_divisors` « reçoit `grid_freq` sous forme de liaison par entité … et retourne
  **alors** une `DataFrame` de diviseurs par ligne » ; règle de forme de la campagne : la forme dépend de la
  configuration, jamais des valeurs.
- **Reproduction** :
  ```python
  idx = pd.MultiIndex.from_product([['DE', 'FR'], pd.date_range('2021-01-31', periods=3, freq='ME')])
  type(StageScaler().feature_divisors(columns=['m1'], column_frequencies={'m1': 'M'}, ways={},
       grid_freq={('DE',): 'Y', ('FR',): 'Y'}, stage_freq='M', index=idx))   # Series
  ```
- **Test** : `tests/unit/frequency/test_stage_scaler.py::TestShapeRule::test_per_entity_grid_freq_always_gives_a_dataframe`
- **Statut** : corrigé (2026-10-11)
- **Correction** : `feature_divisors` rend toujours une `DataFrame` par ligne pour un `grid_freq` par entité. Plus généralement, la forme ne dépend plus de la coïncidence des diviseurs : par ligne sous `'calendar'`, pour un `grid_freq` par entité, ou quand les fréquences effectivement lues (étape, colonnes en `'raw_anchors'`) se résolvent différemment d'une entité à l'autre ; compacte sinon. Tests : `tests/unit/frequency/test_stage_scaler.py::TestShapeRule`.

### ANO-FREQ-030 — Liaison d'étape par entité homogène : forme compacte, contrairement à `source_freq`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/stage_scaler.py::StageScaler.feature_divisors`,
  `::StageScaler._stage_divisor` (`target_divisor`, `fit_scale_factor`)
- **Sévérité** : à arbitrer
- **Observé** : `pred_freq` / `stage_freq` = `{('DE',): 'M', ('FR',): 'M'}` rend un scalaire (`target_divisor`) et
  une `Series` par colonne (`feature_divisors`) ; la même liaison avec des fréquences divergentes rend une `Series`
  / `DataFrame` par ligne. À l'inverse, `source_freq` par entité rend **toujours** une `Series`, même homogène.
  Comportement présent depuis `097012f` (test `test_homogeneous_panel_keeps_the_compact_form`).
- **Attendu** : à arbitrer. La règle de forme du prompt F6 (« la forme dépend de la configuration, jamais des
  valeurs ») plaide pour une forme fixée par la **forme** de la liaison (mapping -> par ligne), comme pour
  `source_freq` ; la forme compacte actuelle évite en revanche des `Series` inutiles sur les panels homogènes.
- **Reproduction** :
  ```python
  idx = pd.MultiIndex.from_product([['DE', 'FR'], pd.date_range('2021-01-31', periods=3, freq='ME')])
  StageScaler().target_divisor('a1', 'Y', {('DE',): 'M', ('FR',): 'M'}, index=idx)        # 12.0
  StageScaler().target_divisor('a1', {('DE',): 'Y', ('FR',): 'Y'}, 'M', index=idx)        # Series
  ```
- **Test** : `tests/unit/frequency/test_stage_scaler.py::TestShapeRule::test_homogeneous_stage_binding_keeps_the_compact_form`
  (épingle le comportement actuel)
- **Statut** : arbitré (2026-10-11) — forme compacte conservée
- **Correction** : règle retenue : une liaison d'étape (`stage_freq` / `pred_freq`) dont les entités s'accordent sur une fréquence (aux alias de position près, `'M'` = `'ME'`) reste compacte ; les liaisons `grid_freq` / `source_freq` décrivent un jeu mutualisé et sont toujours par ligne. Motif : `HighFrequencyImputer` passe une liaison par entité à toute étape de panel ; la forme par ligne figerait chaque `fit_scale_factor` en `Series` sur la grille du fit (report d'échelle exposé au désalignement NaN déjà signalé sous `'calendar'`, plan illisible : `'per-row'` au lieu de 12.0) sans bénéfice. Tests : `tests/unit/frequency/test_stage_scaler.py::TestShapeRule::test_homogeneous_stage_binding_keeps_the_compact_form`, `::test_disagreeing_stage_binding_is_per_row_even_with_equal_divisors`.

### ANO-FREQ-031 — Entité de la grille absente de la liaison d'étape : diviseur `NaN` en silence
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/stage_scaler.py::StageScaler.feature_divisors`,
  `::StageScaler._stage_divisor`, `::StageScaler._spread_one`
- **Sévérité** : mineure
- **Observé** : quand seule la liaison d'étape (`stage_freq` / `pred_freq`) est par entité, les entités servies
  sont celles **nommées par la liaison** (`_binding_entities`), pas celles de la grille ; une entité de la grille
  non couverte garde le `NaN` d'initialisation de `_spread_one`. Diviser par `NaN` efface ensuite toutes ses
  valeurs, sans erreur. Les chemins `grid_freq` / `source_freq` par entité, eux, lèvent (`_grid_entities` +
  `_block_freq`).
- **Attendu** : `ValueError` nommant l'entité, comme pour `grid_freq` / `source_freq` — « an entity of the grid
  left uncovered must be reported, never silently skipped » (docstring de `_grid_entities`), « the scale of a
  block is never guessed » (`_block_freq`). `HighFrequencyImputer` passe a priori des liaisons complètes.
- **Reproduction** :
  ```python
  idx = pd.MultiIndex.from_product([['DE', 'FR', 'IT'], pd.date_range('2021-01-31', periods=3, freq='ME')])
  StageScaler().target_divisor('a1', 'Y', {('FR',): 'M', ('DE',): 'Q'}, index=idx).loc['IT'].tolist()
  # [nan, nan, nan]
  ```
- **Test** : `tests/unit/frequency/test_stage_scaler.py::TestPerEntityFrequencies::test_grid_entity_uncovered_by_the_stage_binding_raises`
- **Statut** : corrigé (2026-10-11)
- **Correction** : la fréquence d'étape d'une entité est résolue par `_stage_freq_for` : entrée propre, liaison à une seule entrée, puis **liaison unanime** (toutes les entrées s'accordent : c'est la fréquence de l'étape, elle vaut pour tout le panel) ; seule une liaison **discordante** qui omet une entité de la grille lève un `ValueError` nommant l'entité. Les entités servies sont celles de la grille dès qu'une liaison par entité est lue (`feature_divisors`, `_stage_divisor`, chemin `produced_freq` de `target_divisor`). Une colonne d'ancres sans fréquence pour une entité retombe sur le diviseur par défaut, jamais sur NaN. Tests : `tests/unit/frequency/test_stage_scaler.py::TestPerEntityFrequencies`.
- **Effet sur `HighFrequencyImputer`** : le cas se produisait réellement. Avec `target_frequency={FR: 'M', DE: 'Q', IT: 'M'}` et `impute_intermediate_frequencies=True` sur PANEL-F, l'étape `M` lie `{FR: M, IT: M}` alors que le jeu mutualisé contient le bloc trimestriel de DE (R2) : ses lignes recevaient un diviseur NaN puis étaient retirées par `usable = y_train.notna()` (côté features, le diviseur par défaut). `n_training_rows` de l'étape `M` / `v` passe de 48 à 60 : les 12 ancres de DE entrent désormais dans l'entraînement, ramenées à l'échelle mensuelle.

### ANO-FREQ-032 — `set_params` contourne la validation de `scale_features`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/stage_scaler.py::StageScaler.fit` (et `target_divisor`,
  `fit_scale_factor`, `feature_divisors`, utilisables sans `fit`)
- **Sévérité** : mineure
- **Observé** : la validation n'a lieu qu'à `__init__`. `set_params(scale_features='exact')` est accepté ;
  `resolve_mode` rend `'exact'`, que `_pair_divisor` traite comme `'constant'` (`mode != 'calendar'`).
- **Attendu** : même arbitrage qu'ANO-FREQ-023 (`AggregationConstraint`, composants utilisables seuls,
  `set_params` compris) : `fit` et les méthodes de diviseur revalident et lèvent le `ValueError` d'`__init__`.
- **Reproduction** :
  ```python
  StageScaler().set_params(scale_features='exact').target_divisor('a1', 'Y', 'M')   # 12.0, aucune erreur
  ```
- **Test** : `tests/unit/frequency/test_stage_scaler.py::TestValidationThroughSetParams::test_inadmissible_setting_raises_after_set_params`
- **Statut** : corrigé (2026-10-11)
- **Correction** : `resolve_mode`, point de passage de `fit` et des trois méthodes de diviseur, revalide `scale_features` ; la validation d'`__init__` est conservée. Tests : `tests/unit/frequency/test_stage_scaler.py::TestValidationThroughSetParams`.

### ANO-FREQ-033 — `transform` accepte d'autres colonnes que celles du `fit`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/stage_scaler.py::StageScaler.transform` (et `inverse_transform`)
- **Sévérité** : mineure
- **Observé** : aucune comparaison avec `feature_names_in_`. Avec des diviseurs compacts, pandas aligne la `Series`
  de diviseurs sur les colonnes : une colonne inconnue au `fit` sort en `NaN`, une colonne du `fit` absente de `X`
  est **ajoutée**, entièrement `NaN`.
- **Attendu** : convention sklearn, cohérente avec ANO-FREQ-026 (corrigée) : `ValueError` nommant les colonnes
  manquantes et inattendues ; ordre des colonnes libre.
- **Reproduction** :
  ```python
  X = pd.DataFrame({'m1': [1.0], 'a2': [12.0]}, index=pd.to_datetime(['2021-12-31']))
  s = StageScaler(source_freq='Y', pred_freq='M', column_frequencies={'m1': 'M', 'a2': 'Y'}).fit(X)
  s.transform(X.rename(columns={'a2': 'q1'})).columns.tolist()   # ['a2', 'm1', 'q1'], a2 et q1 NaN
  ```
- **Test** : `tests/unit/frequency/test_stage_scaler.py::TestTransformSymmetry::test_transform_rejects_other_columns_than_fit`
- **Statut** : corrigé (2026-10-11)
- **Correction** : `transform` / `inverse_transform` d'une `DataFrame` lèvent `ValueError('X has different columns than during fit: missing [...], unexpected [...]')` ; l'ordre des colonnes reste libre. Tests : `tests/unit/frequency/test_stage_scaler.py::TestTransformSymmetry::test_transform_rejects_other_columns_than_fit`, `::test_transform_accepts_reordered_columns`.

### ANO-FREQ-034 — `0`, `0.0`, `np.False_` acceptés par la validation puis traités comme `'constant'`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/stage_scaler.py::StageScaler._validate_scale_features` /
  `::StageScaler.resolve_mode`
- **Sévérité** : mineure
- **Observé** : la validation compare par égalité (`value in (False, 'constant', 'calendar')`, or `0 == False`),
  la résolution par identité (`mode is False`). `StageScaler(scale_features=np.False_)` est donc accepté et divise
  par 12 comme `'constant'` — l'inverse de ce que l'utilisateur a demandé. Même chose pour `0` et `0.0`, et pour les
  valeurs de la forme dict. (`True` et `1` sont, eux, bien rejetés.)
- **Attendu** : validation et résolution cohérentes : rejet de tout ce qui n'est pas `False` (le `Literal[False]`
  du §9.1), ou bien traitement comme `False`.
- **Reproduction** :
  ```python
  StageScaler(scale_features=np.False_).target_divisor('a1', 'Y', 'M')   # 12.0, attendu 1.0 ou ValueError
  ```
- **Test** : `tests/unit/frequency/test_stage_scaler.py::TestValidation::test_values_equal_to_false_are_rejected_or_act_as_false`
- **Statut** : corrigé (2026-10-11)
- **Correction** : `_is_admissible_setting` compare `False` par identité : `0`, `0.0`, `np.False_` sont rejetés, en forme scalaire comme en forme dict (validation partagée avec `HighFrequencyImputer.__init__`). Tests : `tests/unit/frequency/test_stage_scaler.py::TestValidation::test_values_merely_equal_to_false_are_rejected`.

### ANO-FREQ-035 — `_feature_divisor` : ancienne règle B25 restée en code mort
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/stage_scaler.py::StageScaler._feature_divisor`
- **Sévérité** : cosmétique
- **Observé** : la méthode (normalisation vers `f_var`, règle B25) n'a plus aucun appelant depuis `35c36e1`, où
  `feature_divisors` est passé à la règle de la période couverte. Elle reste définie (lignes non couvertes par
  construction).
- **Attendu** : §9.2 — « il n'existe **qu'une** méthode de diviseur de covariable » ; suppression du code mort, sans
  changement de comportement.
- **Reproduction** : `git grep -n "_feature_divisor(" -- tsforecast` ne trouve que la définition.
- **Test** : `tests/unit/frequency/test_stage_scaler.py::TestSingleCovariateDivisorMethod::test_no_residual_covariate_divisor_method`
  (`internal`)
- **Statut** : corrigé (2026-10-11)
- **Correction** : `StageScaler._feature_divisor` supprimée ; `feature_divisors` est la seule méthode de diviseur de covariable.

### ANO-FREQ-036 — `transform` / `inverse_transform` perdent le nom de `y` sous un diviseur par ligne
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/stage_scaler.py::StageScaler.apply` / `::StageScaler.invert`
- **Sévérité** : mineure
- **Observé** : `values / divisor` entre deux `Series` de noms différents rend une `Series` sans nom ; les diviseurs
  par ligne (`'calendar'`, `produced_freq`, `source_freq` par entité) ne sont pas nommés. `transform(y).name` vaut
  donc `'v'` sous un diviseur scalaire et `None` sous un diviseur par ligne : la sortie dépend de la forme du
  diviseur.
- **Attendu** : le nom de `y` est conservé — c'est lui qui désigne la colonne imputée (§9.2, D15) ; un `fit` sur la
  cible transformée retomberait sinon sur le réglage global.
- **Reproduction** :
  ```python
  idx = pd.to_datetime(['2024-01-31', '2024-02-29'])
  y = pd.Series([31.0, 29.0], index=idx, name='m1')
  s = StageScaler(scale_features='calendar', source_freq='M', pred_freq='D').fit(y.to_frame(), y)
  s.transform(y).name   # None
  ```
- **Test** : `tests/unit/frequency/test_stage_scaler.py::TestTransformSymmetry::test_name_of_y_is_kept_under_per_row_divisors`
- **Statut** : corrigé (2026-10-11)
- **Correction** : `apply` / `invert` redonnent au résultat le nom de la `Series` d'entrée (`_keep_name`). Tests : `tests/unit/frequency/test_stage_scaler.py::TestTransformSymmetry::test_name_of_y_is_kept_under_per_row_divisors`.

### ANO-FREQ-037 — Liaison d'étape unanime : une entité de la grille non nommée reste sans fréquence
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/covariate_materializer.py::CovariateMaterializer._stage_frequency` (via `classify`,
  `decide_ways`, `materialize`, `stage_frame`) ; même lecture des clés dans `::_column_frequency`
- **Sévérité** : majeure
- **Observé** : `stage_freq={('FR',): 'Q', ('DE',): 'Q'}` sur une grille FR / DE / IT : IT n'a pas d'entrée propre, la
  liaison a deux entrées, `_stage_frequency` rend `None`, et `classify` rend `'raw_anchors'` pour IT. Ce verdict (rang 2
  de `_WAY_RANK`) l'emporte sur l'agrégation de FR et DE (rang 1) : `ways == {'m1': 'raw_anchors'}`, et `m1` vaut
  **102** (valeur ponctuelle de mars) au T1 2021 pour les trois entités, au lieu de **303** (D22). Par ailleurs, des clés
  d'entité scalaires (`{'FR': 'Q'}`, `{'v': {'FR': 'Y'}}`) ne sont jamais reconnues : le commentaire « clé brute puis clé
  normalisée » normalise l'entité cherchée — déjà un tuple, issu de `iter_entity_blocks` —, jamais les clés ; toutes les
  entités tombent en `None`, sans erreur.
- **Attendu** : la résolution de `StageScaler._stage_freq_for` (ANO-FREQ-031, arbitrée le 2026-10-11) : entrée propre,
  liaison à une entrée, puis liaison **unanime** valant pour tout le panel ; une liaison **discordante** qui omet une
  entité de la grille lève une `ValueError` nommant l'entité. Les deux composants lisent la même liaison d'étape de
  l'imputeur et ne doivent pas en tirer deux fréquences différentes. Clés scalaires : à arbitrer entre normaliser
  (précédent ANO-FREQ-011, « comme les autres fonctions du package ») et lever.
- **Exposition** : l'étape 5a de `HighFrequencyImputer` appelle `stage_frame(grid_index=stage_mask.index,
  stage_freq=stage_freq)` ; la grille couvre toutes les entités (D38), la liaison ne nomme que celles dont le groupe
  traverse l'étape (D31) — cas réel relevé par ANO-FREQ-031 (`target_frequency={FR: 'M', DE: 'Q', IT: 'M'}`, étape `M`
  liée à `{FR, IT}`). La frame d'étape étant la base de la sortie d'étape (D37), l'effet est à mesurer en F11 / F12.
- **Reproduction** :
  ```python
  panel = build_panel_reference()[['m1']]
  grid = pd.MultiIndex.from_product([['FR', 'DE', 'IT'], pd.date_range('2021-03-31', periods=12, freq='QE')])
  features, ways, _ = CovariateMaterializer().materialize(
      columns=['m1'], grid_index=grid, stage_freq={('FR',): 'Q', ('DE',): 'Q'},
      detected_frequencies={'m1': 'M'}, source_data=panel)
  ways, features['m1'].iloc[0]   # ({'m1': 'raw_anchors'}, 102.0)
  ```
- **Tests** : `tests/unit/frequency/covariate_materializer/test_classification.py::TestStageBindingResolution`
  (liaison unanime ; liaison discordante, sous `'model'` ; clés scalaires normalisées), `tests/unit/frequency/covariate_materializer/test_stage_frame.py::TestStageFramePanel::test_unanimous_stage_binding_covers_an_unbound_grid_entity`
- **Statut** : corrigé (2026-10-11) — arbitrage de l'auteur : clés scalaires normalisées
- **Correction** : `_stage_frequency` résout comme `StageScaler._stage_freq_for` : entrée propre, puis liaison **unanime** — aux alias de position près, une seule entrée comprise — valant pour tout le panel, sinon `None`. `_check_stage_coverage` lève une `ValueError` nommant les entités de la grille qu'une liaison discordante ne nomme pas, dans `decide_ways` et `_produce_column` (donc `materialize`, `stage_frame`, `interpolate_column`), avant toute lecture des registres. `_normalized_keys` normalise les clés des liaisons par entité (`detected_frequencies`, `stage_freq`) : `'FR'` vaut `('FR',)`, comme l'argument `entity` des méthodes publiques. Corollaire : sous `record=True`, plus aucune cellule n'est enregistrée avec une fréquence de production `None` (entité non liée). Même normalisation dans `StageScaler._freq_for` (arbitrage de l'auteur, 2026-10-12), par la fonction partagée `tsforecast.panel.utils.normalize_entity_mapping`, que le matérialiseur utilise aussi. Tests : `tests/unit/frequency/covariate_materializer/test_classification.py::TestStageBindingResolution`, `tests/unit/frequency/covariate_materializer/test_stage_frame.py::TestStageFramePanel::test_unanimous_stage_binding_covers_an_unbound_grid_entity`, `tests/unit/frequency/test_stage_scaler.py::TestPerEntityFrequencies::test_plain_entity_keys_behave_like_tuple_keys`, `tests/unit/panel/test_utils.py::TestNormalizeEntityMapping`.

### ANO-FREQ-038 — `decide_ways` : les égalités de rang sont départagées par l'ordre des entités
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/covariate_materializer.py::CovariateMaterializer.decide_ways` (réduction
  `max(candidates, key=_WAY_RANK)`), `::_applicable_way`
- **Sévérité** : majeure
- **Observé** : `_WAY_RANK` donne le même rang à `'interpolate'` et `'raw_anchors'` (2), et à `'stage_model'` et
  `'carried_model'` (3) ; `max` garde le premier verdict rencontré, dans l'ordre des entités de la grille.
  (a) `v` annuelle pour FR, trimestrielle pour DE, jamais observée par IT (`{'v': {('FR',): 'Y', ('DE',): 'Q'}}`), étape
  M, `'interpolate'` : grille FR, DE, IT → `'interpolate'` ; grille IT, FR, DE → `'raw_anchors'`, et FR / DE passent de
  36 cellules interpolées à leurs 3 et 12 ancres.
  (b) Sous `'model'`, FR porte des cellules d'étape M (2022-2023) et des cellules Q (2021), DE des cellules Q seules. DE
  en tête → voie `'carried_model'`, que `_applicable_way` applique à FR (le report y est testé avant le rang 2 dès que
  la voie de colonne n'est pas `'stage_model'`) : FR reporte ses cellules de 2021, ignore celles de son étape, et
  2022-2023 restent NaN. FR en tête → FR lit ses cellules d'étape.
- **Attendu** : une voie indépendante de l'ordre des entités. §4.4 : la précédence « s'arrête au premier cas
  applicable » — une entité qui porte des cellules à `f` est servie au rang 2, quelle que soit la voie de colonne ;
  §4.3 : sous `'interpolate'`, les seuls NaN résiduels sont ceux d'une entité sans la colonne — le verdict d'une entité
  sans conversion définissable (fréquence inconnue) ne doit pas dicter la voie des autres. Piste : écarter ces verdicts
  de la réduction (`_applicable_way` sert déjà leurs lignes) et départager les rangs égaux par une règle fixe.
- **Reproduction** :
  ```python
  panel = build_panel_reference()[['v']]
  dates = pd.date_range('2021-01-31', periods=36, freq='ME')
  freqs = {'v': {('FR',): 'Y', ('DE',): 'Q'}}
  for order in (['FR', 'DE', 'IT'], ['IT', 'FR', 'DE']):
      grid = pd.MultiIndex.from_product([order, dates])
      print(CovariateMaterializer().decide_ways(columns=['v'], grid_index=grid, stage_freq='M',
                                                detected_frequencies=freqs))
  # {'v': 'interpolate'} puis {'v': 'raw_anchors'}
  ```
- **Tests** : `tests/unit/frequency/covariate_materializer/test_classification.py::TestReductionOrder::test_entity_without_frequency_never_dictates_the_column_way`,
  `::TestReductionOrder::test_tied_model_verdicts_never_skip_rank_two`
- **Statut** : corrigé (2026-10-11)
- **Correction** : `decide_ways` écarte de la réduction les verdicts des entités sans conversion définissable (fréquence de colonne ou d'étape inconnue : leurs lignes restent servies par `_applicable_way`) ; une colonne qu'aucune entité ne convertit vaut `'raw_anchors'`. Les rangs égaux sont départagés par un ordre fixe, `_WAY_TIE_BREAK` (`'raw_anchors'` après `'interpolate'`, `'carried_model'` après `'stage_model'`) ; `_WAY_RANK`, lu par `TrainingSetBuilder`, est inchangé. `_applicable_way` résout une voie de modèle par la précédence propre de l'entité — rang 2 si son miroir porte des cellules à `f`, sinon rang 3, sinon repli —, quelle que soit la voie de modèle retenue pour la colonne. Tests : `tests/unit/frequency/covariate_materializer/test_classification.py::TestReductionOrder`, `tests/unit/frequency/covariate_materializer/test_precedence.py::TestModelWaysPerEntity`.

### ANO-FREQ-039 — Dates dupliquées dans `source_data` : l'agrégation exacte compte deux fois
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/covariate_materializer.py::CovariateMaterializer._produce_block` (voie
  `'aggregate'`, via `materialize` / `stage_frame`)
- **Sévérité** : mineure
- **Observé** : `TS` dont le 2023-12-31 est dupliqué, `m1` à l'étape Q : le T4 2023 vaut **537** (133 + 134 + 135 + 135)
  au lieu de **402**, sans erreur ni avertissement. Sur le même jeu, les voies `'identity'` / `'raw_anchors'` lèvent
  l'erreur brute de pandas (`cannot reindex on an axis with duplicate labels`) et `'interpolate'` celle du convertisseur
  (`Several observations fall in the same 'ME' period`).
- **Attendu** : `ValueError` nommant les dates dupliquées, uniforme sur les quatre voies, comme `FrequencyAligner`
  (ANO-FREQ-010) et `AggregationConstraint` (ANO-FREQ-022). `HighFrequencyImputer` n'est pas exposé
  (`validate_temporal_data` lève en amont sur un index dupliqué) : seul l'usage autonome du composant l'est.
- **Reproduction** :
  ```python
  ts = build_panel_reference().loc['FR', ['m1']]
  dup = pd.concat([ts, ts.iloc[-1:]])
  grid = pd.date_range('2021-03-31', periods=12, freq='QE')
  features, _, _ = CovariateMaterializer().materialize(
      columns=['m1'], grid_index=grid, stage_freq='Q', detected_frequencies={'m1': 'M'}, source_data=dup)
  features['m1'].iloc[-1]   # 537.0
  ```
- **Test** : `tests/unit/frequency/covariate_materializer/test_robustness.py::TestDuplicates::test_duplicated_source_dates_are_rejected_by_every_way`
- **Statut** : corrigé (2026-10-11)
- **Correction** : `_check_source_data` lève `ValueError('Duplicate dates in source_data: [...]')` — couples `(entité, date)` pour un panel — dans `materialize`, `stage_frame` et `interpolate_column`, avant toute production : les quatre voies échouent de la même façon. Tests : `tests/unit/frequency/covariate_materializer/test_robustness.py::TestDuplicates::test_duplicated_source_dates_are_rejected_by_every_way`, `::TestDuplicates::test_duplicated_panel_pairs_name_the_entity`, `tests/unit/frequency/covariate_materializer/test_stores.py::TestInterpolateColumn::test_duplicated_source_dates_raise`.

### ANO-FREQ-040 — Colonne absente de `source_data` : colonne NaN rendue en silence
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/covariate_materializer.py::CovariateMaterializer.materialize`, `::stage_frame`
  (garde `column not in source_block.columns` de `_produce_column`)
- **Sévérité** : mineure
- **Observé** : `materialize(columns=['m1', 'typo'], ...)` rend une colonne `typo` entièrement NaN, de voie
  `'raw_anchors'` et d'origine `'observed'`, sans erreur ; de même `stage_frame(columns=[...])`.
- **Attendu** : `KeyError` nommant la colonne, comme `interpolate_column` et `entities_without_column` du même
  composant : une coquille deviendrait sinon une feature vide transmise à l'estimateur. `eligible_columns`, qui est une
  sélection, l'écarte légitimement. `HighFrequencyImputer` ne passe que des colonnes de `X_work` : non exposé.
- **Reproduction** :
  ```python
  ts = build_panel_reference().loc['FR', ['m1']]
  features, ways, origins = CovariateMaterializer().materialize(
      columns=['m1', 'typo'], grid_index=ts.index, stage_freq='M',
      detected_frequencies={'m1': 'M'}, source_data=ts)
  ways['typo'], int(features['typo'].notna().sum())   # ('raw_anchors', 0)
  ```
- **Test** : `tests/unit/frequency/covariate_materializer/test_materialize.py::TestColumnAbsentFromSource::test_column_absent_from_the_source_raises`
  (`materialize`, `stage_frame`)
- **Statut** : corrigé (2026-10-11)
- **Correction** : `_check_source_data` lève `KeyError('Columns missing from source_data: [...]')` dans `materialize` et `stage_frame`. Test : `tests/unit/frequency/covariate_materializer/test_materialize.py::TestColumnAbsentFromSource::test_column_absent_from_the_source_raises`.

### ANO-FREQ-041 — Docstrings de `CovariateMaterializer` en écart avec le code et la spec
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/frequency/covariate_materializer.py::CovariateMaterializer` (docstring de classe),
  `::materialize`, `::stage_frame`
- **Sévérité** : cosmétique
- **Observé** :
  1. `materialize` : « The caller chooses the way once, on the training grid, then imposes it on the prediction grid »
     — l'inverse de D21 / §4.6 (« la grille de prédiction décide, la grille d'entraînement subit »), qu'applique
     l'imputeur (`decide_ways` sur `pred_grid`, puis rejeu). Même docstring : « the second one is what makes the
     uniqueness rule of enforceable » (phrase tronquée).
  2. Docstring de classe : « ranks 2 and 3 of the precedence must discriminate on `origin_store == 'model'` ». Le code,
     la docstring de `classify` (« the trigger is the mere presence of cells produced at f, whatever their origin ») et
     §4.4 (un rang 2 servi par repli porte l'origine `'interpolated'`) déclenchent les rangs 2 et 3 sur la seule
     présence de cellules produites.
  3. `stage_frame` : « an interpolation covers no date a raw anchor would leave empty » énonce l'inclusion inverse —
     l'entité servie par repli reçoit 36 cellules interpolées contre 3 ancres. Le propos voulu est vraisemblablement
     « an interpolation leaves empty no date a raw anchor would cover ».
- **Attendu** : docstrings alignées sur le code et sur D21 / §4.4.
- **Reproduction** : lecture des docstrings ; les tests ci-dessous suivent le code.
- **Tests** : `tests/unit/frequency/covariate_materializer/test_precedence.py::TestPrecedenceRanks::test_rank_two_trigger_ignores_the_origin`,
  `tests/unit/frequency/covariate_materializer/test_stage_frame.py::TestStageFrameMirror::test_entity_with_an_empty_mirror_is_capped_to_its_raw_anchors`
- **Statut** : corrigé (2026-10-11)
- **Correction** : docstrings réécrites : `materialize` (voie choisie sur la grille de PRÉDICTION puis imposée à l'entraînement, D21 ; phrase complétée), docstring de classe (rangs 2 et 3 déclenchés par la seule présence de cellules produites, origine rapportée telle quelle ; registres limités à ce qui est produit, D24), `stage_frame` (plafonnement par entité : l'exception documentée disparaît avec ANO-FREQ-042). Tests : `tests/unit/frequency/covariate_materializer/test_precedence.py::TestPrecedenceRanks::test_rank_two_trigger_ignores_the_origin`, `tests/unit/frequency/covariate_materializer/test_stage_frame.py::TestStageFrameMirror::test_entity_with_an_empty_mirror_is_capped_to_its_raw_anchors`.

### ANO-FREQ-042 — Voie `'raw_anchors'` : une entité plus fine que la grille reçoit des valeurs ponctuelles
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/covariate_materializer.py::CovariateMaterializer._applicable_way` (sortie
  anticipée `if way == 'raw_anchors': return way`), via `materialize` et `stage_frame`
- **Sévérité** : majeure
- **Observé** : `_applicable_way` rend `'raw_anchors'` avant la normalisation de rang 1 par entité ; une entité qui
  observe la colonne au moins aussi finement que la grille reçoit la valeur ponctuelle de la date de grille au lieu de
  l'identité ou de l'agrégation exacte. PANEL-F, `v` à l'étape Q : sous `'tolerate_nan'` (et `'model'` + repli
  `'tolerate_nan'`), IT (mensuelle) vaut **10 / 11 / 12,5** — le mois de fin de trimestre — au lieu des trimestres
  **30 / 33 / 37,5** (§2.5) ; sous `'interpolate'`, la même étape rend bien 30 / 33 / 37,5. `stage_frame`, qui plafonne
  toute voie `'interpolate'` à `'raw_anchors'`, le fait **sous toutes les stratégies**, contrairement à sa docstring
  (« the EXACT aggregations of the finer columns »). Même mécanisme quand une entité sans fréquence impose
  `'raw_anchors'` à la colonne (ANO-FREQ-037, -038) : colonne mensuelle pour DE, trimestrielle pour FR, absente pour
  IT, étape Q → DE vaut 102 au lieu de 303.
- **Attendu** : §4.1 classe « entité par entité » : une covariable de fréquence supérieure ou égale à la grille est
  « toujours disponible, par agrégation exacte ou par identité », `covariate_strategy` ne gouvernant que les colonnes
  strictement plus basses ; §4.6 / D22 : la voie de colonne est appliquée par entité, ramenée à ce que ses fréquences
  autorisent. `'raw_anchors'` doit passer par la même normalisation (identité si `f_c == f`, agrégation si `f_c` est
  plus fine).
- **Échelle** : `StageScaler.feature_divisors` déduit la période couverte de la **voie de colonne** — `'raw_anchors'`
  couvre une période de la colonne (§4.2, §9.2) — et divise donc la valeur de mars d'IT par 1/3 : après mise à
  l'échelle, `X` est au bon niveau (10 × 3 = 30) mais ne porte qu'un mois sur trois. La correction doit être
  coordonnée avec ce diviseur, sans quoi une somme exacte de 30 serait divisée par 1/3. La frame d'étape, elle, n'est
  pas mise à l'échelle : elle porte bien 10 là où le niveau trimestriel attend 30, et elle est la base de la sortie
  d'étape (D37) — effet à mesurer en F11 / F12 (`keep_lower_frequencies=True`).
- **Reproduction** :
  ```python
  panel_f = build_panel_reference()[['m1', 'q1', 'v']]
  grid = pd.MultiIndex.from_product([['FR', 'DE', 'IT'], pd.date_range('2021-03-31', periods=12, freq='QE')])
  freqs = {'v': {('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'}}
  frame = CovariateMaterializer().stage_frame(grid_index=grid, stage_freq='Q', detected_frequencies=freqs,
                                              source_data=panel_f, columns=['v'])
  frame.loc['IT', 'v'].iloc[:4].tolist()   # [10.0, 10.0, 10.0, 10.0] au lieu de [30.0] * 4
  ```
- **Tests** : `tests/unit/frequency/covariate_materializer/test_materialize.py::TestRawAnchorsPerEntity` (trois tests),
  `tests/unit/frequency/covariate_materializer/test_stage_frame.py::TestStageFramePanel::test_finer_entity_is_aggregated_exactly`
- **Statut** : corrigé (2026-10-11) — arbitrage de l'auteur : correction conjointe du matérialiseur et de `StageScaler`, plafonnement de `stage_frame` par entité
- **Correction** : `_applicable_way` ne court-circuite plus `'raw_anchors'` : identité si `f_c == f`, agrégation exacte si `f_c` est plus fine, ancres brutes si elle est plus grossière. `stage_frame` plafonne entité par entité, APRÈS la précédence (`_produce_column(cap_interpolation=True)`) : seule une voie d'entité `'interpolate'` est ramenée aux ancres — y compris celle d'une entité dégradée au repli sous `'model'`, qui reste à ses ancres sous les deux replis ; les rangs 1 à 3 sont conservés. `StageScaler.feature_divisors` : une cellule `'raw_anchors'` ne couvre la période de la colonne que si celle-ci est strictement plus grossière que la grille (`_coarser_than`), une période de grille sinon ; sous `fit` (grille = étape), seules les colonnes plus fines que l'étape changent de diviseur. Deux tests de `tests/unit/frequency/test_stage_scaler.py` passaient par une ancre plus fine que la grille pour obtenir une cellule mensuelle à l'étape Q (`TestConstantGoldenValues::test_feature_divisor_gold[M-cell@Q]`, `TestCalendarGoldenValues::test_fractional_feature_divisor_under_calendar`) : ils l'obtiennent désormais d'une grille de bloc mensuelle (voie `'identity'`), mêmes valeurs d'or. Tests : `tests/unit/frequency/covariate_materializer/test_materialize.py::TestRawAnchorsPerEntity`, `tests/unit/frequency/covariate_materializer/test_stage_frame.py::TestStageFramePanel::test_finer_entity_is_aggregated_exactly`, `::TestStageFrameMirror::test_carried_mirror_is_not_capped`, `tests/unit/frequency/test_stage_scaler.py::TestCarriedPeriodRule` (`test_raw_anchors_of_a_column_at_least_as_fine_as_the_grid_span_the_grid`, `test_raw_anchors_per_entity_follow_each_entity_column`).

### ANO-FREQ-043 — Grille de panel vide : `ValueError: max() iterable argument is empty`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/covariate_materializer.py::CovariateMaterializer.decide_ways` (via
  `materialize`, `stage_frame`)
- **Sévérité** : mineure
- **Observé** : une grille `MultiIndex` vide n'a aucune entité ; la réduction `max(candidates, ...)` reçoit une liste
  vide et lève `ValueError: max() iterable argument is empty`, sans rapport avec la cause. Une grille de série vide,
  elle, garde son entité dégénérée `()` : voies décidées, frame `(0, n)` rendue.
- **Attendu** : le comportement d'une série : une voie par colonne, et une frame vide aux colonnes demandées. Exposition
  de `HighFrequencyImputer` non établie.
- **Reproduction** :
  ```python
  empty = pd.MultiIndex.from_tuples([], names=['country', 'date'])
  CovariateMaterializer().decide_ways(columns=['m1'], grid_index=empty, stage_freq='M',
                                      detected_frequencies={'m1': 'M'})
  # ValueError: max() iterable argument is empty
  ```
- **Test** : `tests/unit/frequency/covariate_materializer/test_robustness.py::TestEmptyAndSingle::test_empty_panel_grid`
  (`decide_ways`, `materialize`, `stage_frame`)
- **Statut** : corrigé (2026-10-11) — arbitrage de l'auteur : une voie par colonne, comme pour une série
- **Correction** : `decide_ways` décide, pour une grille sans entité, pour l'entité dégénérée `()` — exactement le cas d'une grille de série vide ; `materialize` et `stage_frame` rendent une frame `(0, n)`. Test : `tests/unit/frequency/covariate_materializer/test_robustness.py::TestEmptyAndSingle::test_empty_panel_grid`.

### ANO-FREQ-044 — `interpolation_method` en dict sans `'__default__'` : repli `'time'` au lieu de `'linear'`
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/covariate_materializer.py::DEFAULT_INTERPOLATION_METHOD`,
  `::CovariateMaterializer.resolve_method`
- **Sévérité** : mineure
- **Observé** : `DEFAULT_INTERPOLATION_METHOD = 'time'` (depuis `bad4412`, création du composant) sert de défaut du
  composant et de repli d'une forme dict sans `'__default__'`. `HighFrequencyImputer` a pour défaut `'linear'` et
  transmet son dict tel quel : `HighFrequencyImputer(interpolation_method={'a1': 'cubic'})` interpole donc toutes les
  autres colonnes en `'time'`.
- **Attendu** : §10.1 : `interpolation_method = 'linear'` par défaut, « forme dict par feature avec `'__default__'`,
  mêmes règles de validation qu'au §9.1 » — et §9.1 : « features non couvertes et pas de `'__default__'` → défaut ».
  Une colonne non couverte reçoit `'linear'`. (`'time'` et `'linear'` diffèrent dès que les pas sont inégaux : mois
  de 28 à 31 jours.)
- **Reproduction** :
  ```python
  CovariateMaterializer(interpolation_method={'a1': 'cubic'}).resolve_method('q1')   # 'time'
  ```
- **Test** : `tests/unit/frequency/covariate_materializer/test_configuration.py::TestPerFeatureResolution::test_uncovered_column_falls_back_to_the_parameter_default`
- **Statut** : corrigé (2026-10-11)
- **Correction** : `DEFAULT_INTERPOLATION_METHOD = 'linear'` : défaut du composant et repli d'une forme dict sans `'__default__'`. Effet hors imputeur : un `CovariateMaterializer()` autonome interpole désormais en `'linear'` (positionnel sans ancrage, `'time'` sous `anchor_fraction`, comme le prévoit le convertisseur). Test : `tests/unit/frequency/covariate_materializer/test_configuration.py::TestPerFeatureResolution::test_uncovered_column_falls_back_to_the_parameter_default`.

### ANO-FREQ-045 — Cellule coïncidente écartée au profit d'une ancre que la fenêtre exclut
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/training_set_builder.py::TrainingSetBuilder.build` (`_candidate_rows`)
- **Sévérité** : mineure
- **Observé** : sous `keep_coincident_cells=False` (traduction de `aggregation_constraint='sum'`), la cellule produite
  qui partage sa date avec une observation était écartée **avant** la fenêtre `'training'` et le filtre d'origine. Or
  `get_mask_at_frequency` ne garde une période basse fréquence que si **toutes** ses sous-périodes sont dans la fenêtre :
  une fenêtre qui commence en cours d'année exclut l'ancre annuelle mais garde le T4. Les deux lignes de la période
  étaient alors perdues. Atteignable par `HighFrequencyImputer` : sur `TS` dont `m1` est vide au premier semestre 2021,
  sous `impute_intermediate_frequencies=True` et `'sum'`, `a1` s'entraîne à l'étape `M` sur 9 lignes, le T4 2021 absent.
- **Attendu** : §5.5 / §5.9 — « l'observation gagne » parce que, recalées sur elle, les sous-périodes en font une
  combinaison linéaire exacte. Une imputation n'est redondante qu'avec une observation **retenue** : si la fenêtre exclut
  l'ancre, le T4 n'est redondant avec rien et doit rester (10 lignes).
- **Reproduction** :
  ```python
  data = reference_timeseries.copy(); data.index.freq = None
  data.loc[:'2021-06-30', 'm1'] = np.nan
  HighFrequencyImputer(estimator=LinearRegression(), target_frequency='M',
                       impute_intermediate_frequencies=True, aggregation_constraint='sum').fit(data)
  # jeu de 'a1' à l'étape M : 9 lignes, aucune au 2021-12-31
  ```
- **Tests** : `tests/unit/frequency/training_set_builder/test_coincident_cells.py::TestCoincidentCells::test_produced_cell_kept_when_its_anchor_is_outside_the_window`,
  `::test_imputer_keeps_the_quarter_of_an_excluded_year`
- **Statut** : corrigé (2026-10-12) — accord de l'auteur sur le principe (le constat initial « sans effet pour
  l'imputeur » était faux)
- **Correction** : `_candidate_rows` garde toutes les cellules produites ; le départage est fait par
  `_drop_coincident_cells` **après** la fenêtre et le filtre d'origine, contre les seules observations retenues.
  Docstring de `build` (`keep_coincident_cells`) mise à jour. La spec (§5.5, §5.9) reste à amender au prompt R9.

### ANO-FREQ-046 — Entité à fréquence indétectable : elle emprunte la fréquence des autres quand elles s'accordent
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._detected_frequencies_by_column`
  (repli sur la forme scalaire), lu par `TrainingSetBuilder.blocks_of`
- **Sévérité** : mineure
- **Observé** : une entité qui observe la variable une seule fois a une fréquence indétectable
  (`_undetected_frequencies_`). Si les entités détectées sont en désaccord, la liaison reste par entité et le bloc de
  l'entité est écarté (journalisé). Si elles s'accordent, la colonne est repliée sur la forme scalaire, et l'entité en
  hérite. Sur `PANEL-F` avec `v` annuelle pour FR et DE et une seule valeur mensuelle pour IT, le bloc IT vaut `'Y'` :
  sa valeur 10.0 serait prise pour un total annuel et divisée par 12. Le sort de l'entité dépend donc des AUTRES
  entités. (La fenêtre `'training'` par défaut écarte ici la ligne, mais pas par principe.)
- **Attendu** : §5.8 R2 — `f_block(e) = f_var(e, v)`, la fréquence **propre** de l'entité. Sans fréquence propre, pas
  de diviseur : le bloc est écarté, quelles que soient les autres entités (cohérent avec D40 et la classification,
  qui laisse ces couples de côté).
- **Reproduction** :
  ```python
  data = mixed_freq_panel_multifrequency.copy()
  data.loc['IT', 'v'] = np.nan; data.loc[('IT', pd.Timestamp('2021-01-31')), 'v'] = 10.0
  data.loc['DE', 'v'] = np.nan   # puis DE annuelle : 120 / 132 / 150 aux fins d'année
  # HighFrequencyImputer(...).fit(data) : blocs de 'v' = {DE: Y, FR: Y, IT: Y}
  ```
- **Test** : `tests/unit/frequency/training_set_builder/test_blocks_and_rows.py::TestScopeOfContributors::test_imputer_never_lends_a_frequency_to_an_undetectable_entity`
- **Statut** : corrigé (2026-10-12) — arbitrage de l'auteur : écarter toujours
- **Correction** : `HighFrequencyImputer._training_frequencies` remet au constructeur la colonne cible sous forme par
  entité, chaque couple indétectable explicitement à `None` ; `blocks_of` écarte alors le bloc (journalisé). Les
  entrées des covariables ne changent pas. Docstring de `blocks_of` : forme `None` explicite documentée.

### ANO-FREQ-047 — Niveau de fréquence repéré par son nom
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._has_frequency_level`,
  `::_select_inverse_frequency_level`, `::_drop_frequency_level` ; `tsforecast/frequency/training_set_builder.py::TrainingSetBuilder.build`
- **Sévérité** : mineure
- **Observé** : `inverse_transform` reconnaissait une sortie multi-fréquences au nom `'frequency'` d'un de ses niveaux
  et l'adressait par ce nom (`xs(label, level='frequency')`). Une sortie dont l'utilisateur a renommé les niveaux
  n'était pas reconnue (le niveau de fréquence restait dans le résultat) ; un panel dont un niveau d'entité s'appelle
  déjà `'frequency'` produisait un index empilé à deux niveaux du même nom, ambigu à l'adressage par nom — de même pour
  la grille estampillée du constructeur.
- **Attendu** : §5.9 — la forme de la grille est « portée par le jeu lui-même, jamais reniflée des noms de niveaux » ;
  même principe pour la sortie empilée (§12.4), qui partage la même clé.
- **Reproduction** :
  ```python
  imputer = HighFrequencyImputer(estimator=LinearRegression(), target_frequency='M')
  out = imputer.fit_transform(reference_timeseries)
  imputer.inverse_transform(out.rename_axis(['pas', 'jour']))   # niveau de fréquence non retiré
  ```
- **Tests** : `tests/unit/frequency/high_frequency_imputer/test_transform.py::TestReservedFrequencyLevel`,
  `tests/unit/frequency/training_set_builder/test_robustness.py::TestIndexRobustness::test_reserved_level_name_is_rejected`
- **Statut** : corrigé (2026-10-12) — arbitrage de l'auteur : l'option la plus robuste, appliquée partout
- **Correction** : nom unique `FREQUENCY_LEVEL_NAME` dans `tsforecast/utils/_constants.py` (valeur `'frequency'`
  conservée : nom public de la sortie, utilisé par la documentation et le notebook 2) ; nom **réservé** — `fit`,
  `transform` et `TrainingSetBuilder.build` rejettent une entrée dont un niveau d'index (ou `panel_cols`, `time_col`)
  le porte, quelle que soit la configuration ; `inverse_transform` reconnaît une sortie empilée à sa **profondeur**
  (notée par `transform` au moment de l'empilement) et lit le niveau **par position** (l'avant-dernier).

### ANO-FREQ-048 — Estimateur dont tous les plis échouent : `ValueError` au lieu du repli `-inf`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/variable_orderer.py::VariableOrderer._order_by_cv`
- **Sévérité** : majeure
- **Observé** : `cross_val_score(..., error_score=np.nan)` lève `ValueError: All the N fits failed` dès que
  **tous** les plis échouent (sklearn ne renvoie des NaN que si au moins un pli réussit). La branche
  `np.all(np.isnan(scores))` (sentinelle `-inf` + journal) est donc inatteignable avec un vrai estimateur ;
  l'exception sort de `order()` et, sous `fit_predict_order='cv'`, interrompt `HighFrequencyImputer.fit`.
- **Attendu** : §8.2 — « sentinelle `-np.inf` pour une variable dont **tous** les plis échouent » et journal
  de ces variables ; la docstring de `order` (`log`) le promet aussi.
- **Reproduction** :
  ```python
  orderer = VariableOrderer(fit_predict_order='cv', cv=2, cv_scoring='neg_mean_absolute_error',
                            min_cv_train_size=2).fit()
  orderer.order(specs, X=frame, estimator=FailingEstimator(), log=print)   # ValueError
  ```
- **Test** : `tests/unit/frequency/test_variable_orderer.py::TestFailingFolds::test_failing_estimator_scores_minus_inf_and_logs`
- **Statut** : corrigé (2026-10-12)
- **Correction** : `_order_by_cv` rattrape la `ValueError` de `cross_val_score` dont le message contient
  `fits failed` et la traite comme des scores NaN (repli `-inf` + journal) ; toute autre `ValueError` est relancée
  (`test_unrelated_value_error_is_propagated`).

### ANO-FREQ-049 — Un imputeur ajusté ne se sérialise pas (`pickle`, `copy.deepcopy`)
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/imputation_plan.py::ImputationStep` (champs `materialization`,
  `training_blocks`), donc `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer` ajusté
- **Sévérité** : majeure
- **Observé** : `pickle.dumps(imputeur_ajusté)` et `copy.deepcopy(imputeur_ajusté)` lèvent
  `TypeError: cannot pickle 'mappingproxy' object`. `ImputationStep.__post_init__` fige `materialization` et
  `training_blocks` en `types.MappingProxyType`, que `pickle` ne sait pas sérialiser ; les attributs
  `imputation_plan_` et `_training_set_builder` échouent tous deux. Un imputeur NON ajusté se sérialise.
- **Attendu** : un estimateur sklearn ajusté se persiste (`pickle`, `joblib.dump`) et se copie, et le
  rejeu après aller-retour est identique (§12.5, conformité sklearn ; `check_estimators_pickle`). Le gel
  peut être conservé avec une forme sérialisable (p. ex. `__reduce__` / `__getstate__` de l'étape, ou un
  mapping immuable picklable).
- **Reproduction** :
  ```python
  import pickle
  imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression()).fit(reference_timeseries)
  pickle.dumps(imputer)            # TypeError: cannot pickle 'mappingproxy' object
  copy.deepcopy(imputer)           # même erreur
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_sklearn.py::TestPersistence::test_pickle_round_trip_replays_identically`,
  `::test_deepcopy_of_a_fitted_imputer_replays_identically`
- **Statut** : corrigé (2026-10-12)
- **Correction** : `ImputationStep.__getstate__` rend `materialization` et `training_blocks` en `dict`, `__setstate__` les fige de nouveau en `MappingProxyType` : le gel est conservé, l'étape — donc le plan et l'imputeur ajusté — passe `pickle` et `copy.deepcopy`, et le rejeu après aller-retour est identique. Tests : `test_sklearn.py::TestPersistence` (plus de `xfail`), `tests/unit/frequency/test_imputation_plan.py::TestImputationStepSerialization`.

### ANO-FREQ-050 — `transform` marque `ORIGINAL` les cellules NaN jamais imputées
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._transform`
  (matrice `imputation_provenance_` du rejeu)
- **Sévérité** : mineure
- **Observé** : après `fit`, une cellule NaN qu'aucune étape n'impute est **non marquée** (`isna()`) ; après
  `transform` des mêmes données, la même cellule porte `ORIGINAL`. Sur `TS` dont `q1`, `a1`, `a2` s'arrêtent
  en 2022, les 36 cellules de 2023 diffèrent entre la matrice du fit et celle du transform. Même constat sur
  `PANEL-F` sans `v` pour `IT` (36 cellules d'`IT`).
- **Attendu** : §5.10 (« précision de vocabulaire, mesurée ») et §6.4 : le tracker ne pose `ORIGINAL` que
  sur les cellules **non nulles en entrée** ; une cellule NaN jamais imputée est non marquée, au fit comme au
  transform. I1 exige en outre l'égalité des provenances entre l'ajustement et son rejeu. Effet :
  `provenance_statistics_` compte ces cellules comme originales après un transform.
- **Reproduction** :
  ```python
  data = reference_timeseries.copy(); data.loc['2023', ['q1', 'a1', 'a2']] = np.nan
  imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                 keep_lower_frequencies=False).fit(data)
  imputer.imputation_provenance_.loc['2023', 'a1'].isna().all()   # True
  imputer.transform(data)
  imputer.imputation_provenance_.loc['2023', 'a1'].unique()        # ['original']
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_transform.py::TestTransformState::test_never_imputed_cells_stay_unmarked_after_transform`
- **Statut** : corrigé (2026-10-12)
- **Correction** : `_replay_plan` ne pose `ORIGINAL` que sur les cellules de la grille absentes du tracker **et** porteuses d'une valeur (agrégations exactes) ; une cellule restée NaN est non marquée, au rejeu comme au fit. Le message de l'avertissement agrégé et les docstrings disent désormais « NaN and unmarked » au lieu de « NaN and ORIGINAL ». Tests : `test_transform.py::TestTransformState::test_never_imputed_cells_stay_unmarked_after_transform` (plus de `xfail`), `::test_fit_and_replay_provenances_agree_on_a_panel` (I1 sur `PANEL-F` sans `v` d'`IT`).

### ANO-FREQ-051 — `inverse_transform(restore_original_values=True)` juste après un `fit` : `AttributeError`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._restore_original_values`
- **Sévérité** : mineure
- **Observé** : `AttributeError: 'HighFrequencyImputer' object has no attribute '_original_X_'`.
  `_restore_original_values` lit `self._original_X_` sans garde ; seul `transform` écrit cet instantané, et
  `fit` le purge (B19). `_drop_frequency_level`, lui, le lit par `getattr(..., None)`.
- **Attendu** : l'inversion juste après un `fit` est déclarée licite (le `fit` écrit sa propre matrice,
  `test_inverse_right_after_a_fit_reads_the_fit_provenance`) ; sous `restore_original_values=True` elle doit
  restaurer les valeurs observées — celles des données passées, ou celles du fit — plutôt que lever.
- **Reproduction** :
  ```python
  imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                 restore_original_values=True).fit(reference_timeseries)
  imputer.inverse_transform(reference_timeseries)   # AttributeError
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_inverse_transform.py::TestInverseTransform::test_restoring_inverse_right_after_a_fit`
- **Statut** : corrigé (2026-10-12)
- **Correction** : le `fit` écrit en phase 6 l'instantané de sa propre entrée (`_original_X_`, `_original_y_`), comme il écrit sa propre matrice de provenance : B19 purge l'état du transform en tête de `fit`, puis le `fit` décrit la dernière passe. Les deux tests B19 deviennent publics (`test_fit_phases.py::TestFitPhases::test_fit_purges_stale_transform_state`, `test_transform.py::TestTransformState::test_transform_state_purged_at_fit` : l'inversion exacte restitue les valeurs du dernier passage) et perdent `@pytest.mark.internal`.

### ANO-FREQ-052 — `set_params` contourne la validation de cinq paramètres de l'imputeur
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._fit` (phase 0)
- **Sévérité** : mineure
- **Observé** : la validation de l'imputeur n'a lieu qu'à `__init__`. Après `set_params` (chemin de
  `GridSearchCV`), la plupart des valeurs refusées le restent au `fit`, parce que les composants reconstruits
  au `fit` revalident les leurs. Ne sont **pas** revalidés :
  - les quatre booléens (`keep_lower_frequencies`, `restore_original_values`, `verbose`,
    `impute_unobserved_entities`) : `'yes'` est accepté en silence et traité comme vrai ;
  - `impute_intermediate_frequencies='yes'` : `KeyError: 'yes'` (lecture d'`ELIGIBLE_ORIGINS`) au lieu du
    `ValueError` d'`__init__` ;
  - `estimator` / `additive_transformer` sans méthode requise : `TypeError` de `sklearn.clone`
    (« Cannot clone object… »), sans le message d'`__init__`.
- **Attendu** : même arbitrage que pour ANO-FREQ-023 / 032 (composants utilisables seuls, `set_params`
  compris) et B3 (« toute normalisation a lieu au `fit` ») : le `fit` revalide et lève l'erreur d'`__init__`.
- **Reproduction** :
  ```python
  HighFrequencyImputer(target_frequency='M').set_params(verbose='yes').fit(reference_timeseries)   # aucune erreur
  HighFrequencyImputer(target_frequency='M').set_params(
      impute_intermediate_frequencies='yes').fit(reference_timeseries)                              # KeyError
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_sklearn.py::TestValidationThroughSetParams::test_fit_raises_the_message_of_init`
- **Statut** : corrigé (2026-10-12)
- **Correction** : la validation d'`__init__` est regroupée dans `_validate_params()`, appelée par `__init__` (après le stockage tel quel) et rejouée en tête de la phase 0 du `fit` : toute valeur refusée à `__init__` l'est au `fit` avec la même exception et le même message. Test : `test_sklearn.py::TestValidationThroughSetParams::test_fit_raises_the_message_of_init` (24 paramètres).

### ANO-FREQ-053 — Clés inconnues des dictionnaires par feature ignorées au `fit`
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._fit` (transmission de
  `scale_features`, `interpolation_method`, `interpolation_anchor`, `aggregation_constraint` aux composants)
- **Sévérité** : majeure
- **Observé** : `scale_features={'zz': 'calendar'}`, `interpolation_method={'zz': 'linear'}`,
  `interpolation_anchor={'zz': 0.5}`, `aggregation_constraint={'zz': None}` ajustent sans erreur ni
  avertissement sur un jeu sans colonne `zz` ; une clé `(entité, colonne)` (`{('FR', 'v'): ...}`) est acceptée
  à `__init__` et ignorée au `fit`. Les composants disposent pourtant d'une vérification des clés (« names
  unknown columns ») que l'imputeur n'atteint pas. Une faute de frappe dans un nom de colonne applique donc le
  défaut en silence.
- **Attendu** : §9.1 (« clés inconnues au `fit` → `ValueError` listant les colonnes fautives »), §10.1
  (« mêmes règles de validation qu'au §9.1 »), §13.1 (« clés vérifiées au `fit` » pour `interpolation_method`
  et `aggregation_constraint`), D10 (clés = noms de colonne, jamais des couples). **À arbitrer** : `estimator`
  en dict, pour lequel la spec ne fixe que la clé `'__default__'` ; comportement actuel épinglé (clé inconnue
  ignorée).
- **Reproduction** :
  ```python
  HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                       scale_features={'zz': 'calendar'}).fit(reference_timeseries)   # aucune erreur
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_parameters.py::TestPerFeatureDictUnknownColumn`
- **Statut** : arbitré et corrigé (2026-10-12)
- **Correction** : **arbitrage de l'auteur** — une clé inconnue est **ignorée avec un avertissement**, et non refusée : un dict par feature doit pouvoir servir à des jeux dont les colonnes diffèrent (sélection de variables en amont, recherche sur des sous-ensembles), et l'avertissement suffit à révéler une faute de frappe. `_warn_unknown_feature_keys` émet en phase 0 du `fit` un **unique** `UserWarning` listant, paramètre par paramètre, les clés absentes des colonnes de travail (cible comprise) ; `estimator` suit la même règle. Une clé `(entité, colonne)` est traitée de même, le message rappelant que les clés sont des noms de colonne (D10). La colonne concernée retombe sur `'__default__'` ou sur le défaut. Les composants utilisés seuls gardent leur `validate_columns` strict. Spec alignée (§9.1, §10.1, §13.1). Tests : `test_parameters.py::TestPerFeatureDictUnknownColumn` (plus de `xfail`).

### ANO-FREQ-054 — Avertissement sur `min_cv_train_size` hors `covariate_strategy='model'`
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._fit` (appel de
  `VariableOrderer.fit` sous `fit_predict_order='cv'`, quelle que soit la stratégie)
- **Sévérité** : mineure
- **Observé** : sous `covariate_strategy='interpolate'`, `fit_predict_order='cv', cv=3, min_cv_train_size=2`
  émet « min_cv_train_size (2) < the effective number of CV splits (3): cross-validation… », alors que l'ordre
  et ses paramètres sont inertes hors `'model'` (aucune validation croisée n'est lancée, sortie identique).
- **Attendu** : D9 et §5.6 : une combinaison inerte est documentée, jamais signalée. La résolution du
  splitter (`cv_`, §13.2) peut rester ; l'avertissement n'a d'objet que sous `'model'`.
- **Reproduction** :
  ```python
  HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(), fit_predict_order='cv',
                       cv=3, min_cv_train_size=2).fit(reference_timeseries)   # UserWarning
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_parameters.py::TestInertCombinations::test_inert_combination_is_silent[cv_parameters]`
- **Statut** : corrigé (2026-10-12)
- **Correction** : hors `covariate_strategy='model'`, l'imputeur résout toujours `cv_` mais étouffe l'avertissement croisé de `VariableOrderer.fit` ; il reste émis sous `'model'`. Tests : `test_parameters.py::TestInertCombinations::test_inert_combination_is_silent[cv_parameters]` (plus de `xfail`), `::test_cv_warning_is_kept_under_model`.

### ANO-FREQ-055 — Avertissement `estimator=None` : `__init__` (spec) ou `fit` (code)
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._validate_estimator`
- **Sévérité** : à arbitrer
- **Observé** : `__init__` n'émet rien sous `estimator=None` ; l'avertissement unique est émis au `fit`. Le
  code le justifie (« pour ne pas se répéter à chaque clone ») et la docstring le dit (« a single warning at
  `fit` »).
- **Attendu** : le tableau §13.1 (« Validations à `__init__` ») range « avertissement **unique** si `None` »
  parmi les contrôles d'`__init__`. L'emplacement du code paraît le meilleur (B3 : `__init__` sans effet ;
  `clone` reconstruit l'objet) : la spec serait à aligner au prompt R9.
- **Reproduction** : `HighFrequencyImputer(target_frequency='M')` → aucun avertissement ; `.fit(X)` → un.
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_parameters.py::TestInitDoesNotTransform::test_init_emits_no_warning`
  (comportement actuel épinglé), `test_fit_phases.py::TestFitPhases::test_estimator_none_warns_once`
- **Statut** : arbitré (2026-10-12) — spec alignée sur le code
- **Correction** : **arbitrage de l'auteur** — l'avertissement reste au `fit`, émis en phase 0 avant tout entraînement : à `__init__` il se répéterait à chaque `clone` (une fois par pli et par candidat d'une recherche d'hyperparamètres) et manquerait `set_params(estimator=None)`, le chemin même de `GridSearchCV`. Un avertissement n'interrompt de toute façon pas l'entraînement. Le tableau §13.1 de la spec est corrigé ; aucun changement de code.

### ANO-FREQ-056 — Docstring de `HighFrequencyImputer` incomplète face au §13
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer` (docstring de classe)
- **Sévérité** : cosmétique
- **Observé** : les entrées `scale_features`, `interpolation_method` et `interpolation_anchor` ne mentionnent
  ni la clé `'__default__'` (admise par le code, testée) ni le repli d'une colonne non couverte ;
  l'attribut ajusté `additive_transformer_` (clone ajusté du paramètre, `None` sinon) n'est pas documenté.
- **Attendu** : §9.1 / §10.1 / §10.2 (forme dict avec `'__default__'`, défaut pour une colonne non
  couverte) ; section `Attributes:` complète.
- **Reproduction** : lecture de `HighFrequencyImputer.__doc__`.
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_parameters.py::TestAdmittedValues` (formes
  `'__default__'` admises), `::TestAdmittedValuesTakeEffect::test_additive_transformer_is_cloned_not_fitted`
- **Statut** : corrigé (2026-10-12)
- **Correction** : docstring de classe complétée — clé `'__default__'` et repli d'une colonne non couverte pour les cinq dicts par feature, règle des clés inconnues (ANO-FREQ-053), attribut `additive_transformer_`.

### ANO-FREQ-057 — Scalaires numpy refusés par la validation de l'imputeur
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._validate_params`,
  `tsforecast/frequency/variable_orderer.py::VariableOrderer._validate_cv`
- **Sévérité** : mineure
- **Observé** : un `np.bool_` passé à l'un des quatre booléens était refusé avec le message trompeur
  « must be a bool, got bool » ; `np.int64` était refusé pour `min_cv_train_size` (`TypeError`) et pour `cv`
  (ni entier, ni splitter) ; `np.float32` pour les seuils de couverture. Une grille construite par `np.arange`
  ou `np.linspace` échouait donc dans `GridSearchCV`.
- **Attendu** : la convention de sklearn (`Interval(Integral)`, `"boolean"`) admet les scalaires numpy ; les
  booléens restent exclus des entiers, et `impute_intermediate_frequencies` reste comparé par identité (I13).
- **Reproduction** :
  ```python
  HighFrequencyImputer(target_frequency='M', verbose=np.False_)          # TypeError: got bool
  HighFrequencyImputer(target_frequency='M', min_cv_train_size=np.int64(3))  # TypeError
  ```
- **Test** : `tests/unit/frequency/test_variable_orderer.py::TestCVResolution::test_numpy_int_cv_is_an_int`,
  `tests/unit/frequency/high_frequency_imputer/test_parameters.py::TestRejectedValues::test_numpy_booleans_take_effect`,
  `::test_numpy_integers_from_a_grid_are_admitted`, `::test_numpy_scalars_outside_the_domain_are_refused`,
  `::TestAdmittedValues` (formes numpy)
- **Statut** : corrigé (2026-10-12)
- **Correction** : `_is_integer` (`numbers.Integral`, booléens Python et numpy exclus) pour `min_cv_train_size`
  et `cv` ; `numbers.Real` pour les seuils ; `(bool, np.bool_)` pour les quatre booléens ; `VariableOrderer`
  admet un `cv` entier numpy.

### ANO-FREQ-058 — `n_features_in_` / `feature_names_in_` absents des transformateurs du paquet
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/base/transformers.py::PanelTimeSeriesTransformer.fit`
- **Sévérité** : mineure
- **Observé** : la classe de base n'écrit que `n_features_` / `feature_names_` ; `Pipeline.n_features_in_`
  d'un pipeline commençant par `HighFrequencyImputer` lève `AttributeError`.
- **Attendu** : convention sklearn — `n_features_in_` (entier) et `feature_names_in_` (tableau `object`) écrits
  par `fit`.
- **Reproduction** :
  ```python
  Pipeline([('impute', HighFrequencyImputer(target_frequency='M'))]).fit(X).n_features_in_   # AttributeError
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_sklearn.py::TestFeaturesSeenAtFit`
- **Statut** : corrigé (2026-10-12)
- **Correction** : `PanelTimeSeriesTransformer.fit` écrit `n_features_in_` et `feature_names_in_` (colonnes
  après passage des colonnes de temps et de panel dans l'index) ; les anciens `n_features_` / `feature_names_`
  sont **supprimés**, sans alias (arbitrage de l'auteur, 2026-10-12).

### Arbitrages de l'auteur (2026-10-12) sur `HighFrequencyImputer`
- **Clés inconnues des dicts par feature** (ANO-FREQ-053) : ignorées avec un `UserWarning` unique au `fit`,
  `estimator` compris ; jamais d'erreur.
- **Avertissement `estimator=None`** (ANO-FREQ-055) : au `fit`, en phase 0, avant tout entraînement.
- **Inversion juste après un `fit`** (ANO-FREQ-051) : licite, `restore_original_values=True` compris ; le `fit`
  est une passe au même titre qu'un `transform`.

### ANO-FREQ-059 — Avertissements de recalage émis une fois par (étape, colonne) et par covariable interpolée
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._fit` (phase 5) et
  `::_transform` (phase 5'), via `tsforecast/frequency/aggregation_constraint.py::AggregationConstraint.rescale`
  (appelée par `_execute_step` et par `CovariateMaterializer._interpolate_series`)
- **Sévérité** : mineure
- **Observé** : chaque appel à `rescale` émet ses propres avertissements « aggregate to zero » / « flipped the
  predicted profile ». Sur `TS` dont `a1` et `a2` sont négatives (-120 / -132 / -150, -60 / -66 / -72), un
  estimateur constant à +1 et `impute_intermediate_frequencies=True` : **quatre** messages « flipped » au `fit`
  (`a1` et `a2`, étapes Q et M, « for 3 period(s), in column(s) ['a1'] » ...), quatre de plus au `transform`. Une
  covariable `a2` d'ancres +100 / -10 / +100, interpolée pour q1 et pour a1 sur les grilles d'entraînement et de
  prédiction : **six** messages identiques pour la seule année 2022.
- **Attendu** : §12.5 — « Avertissements uniques (… périodes à signe inversé …), jamais un par variable × étape :
  accumulation puis message agrégé en fin de phase ». Un message par famille pour toute la phase, nommant les
  colonnes et le nombre total de périodes.
- **Reproduction** :
  ```python
  data = reference_timeseries.copy(); data[['a1', 'a2']] = -data[['a1', 'a2']]
  HighFrequencyImputer(target_frequency='M', estimator=ConstantEstimator(constant=1.0),
                       impute_intermediate_frequencies=True).fit(data)   # 4 avertissements « flipped »
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_fit_phases.py::TestAggregatedWarnings`
  (`test_flipped_profiles_of_two_columns_over_two_stages_warn_once`, `test_the_replay_warns_once_too`,
  `test_a_covariate_interpolated_for_several_variables_warns_once`, `test_zero_aggregates_of_two_columns_warn_once`),
  `tests/unit/frequency/test_aggregation_constraint.py::TestDeferredWarnings`
- **Statut** : corrigé (2026-10-13)
- **Correction** : `AggregationConstraint.deferred_warnings()` (gestionnaire de contexte, réentrant) : dans le bloc,
  `rescale` et `transform` recensent les périodes dégénérées au lieu d'avertir, dédoublonnées par colonne ; à la
  sortie, `_emit_warnings` émet un message par cas pour tout le bloc (rien en cas d'exception, aucun état résiduel).
  `HighFrequencyImputer` ouvre ce bloc autour de la boucle des étapes (phase 5) et du rejeu (phase 5') : les
  recalages des étapes et ceux des covariables interpolées par le matérialiseur, qui partage la même instance,
  donnent un message par famille et par phase. **Décompte** : une période (colonne, période observée) rencontrée à
  plusieurs étapes ou par plusieurs appels compte **une** fois — 6 périodes dans le scénario ci-dessus (2 colonnes
  × 3 années), et non 12 (arbitrage de l'auteur, 2026-10-13 : le décompte porte sur les périodes observées, non sur
  les recalages). Utilisé seul, `rescale` garde son comportement (un message par appel).

### ANO-FREQ-060 — Colonne à fréquence indétectable retenue comme covariable : les autres variables retombent en repli
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._select_feature_columns`
  (via `_prepare_variable`)
- **Sévérité** : majeure
- **Observé** : une colonne observée une seule fois (cible `y` à une observation, ou colonne de X) a une fréquence
  indétectable et rejoint les couples non détectés ; elle n'est pas imputée, mais reste **candidate covariable** :
  non vide sur les deux fenêtres (sa seule valeur y figure), elle est retenue. Ni agrégeable ni interpolable faute de
  fréquence, elle n'apporte que des NaN à `X_train`, et un estimateur qui ne les tolère pas échoue : sur `TS` avec
  `y = a1` réduite à 150 au 2023-12-31, q1 et a2 tombent en repli (« estimator fit failed (ValueError: Input X
  contains NaN…), interpolation fallback »). Une colonne entièrement NaN, elle, est bien écartée.
- **Attendu** : §12.5 — « Fréquence indétectable, jamais fatale : … un couple inutilisable ne fait pas échouer le
  `fit` des autres » ; §4.5 (éligibilité). Un couple sans fréquence détectée n'est pas une covariable éligible : q1
  et a2 s'ajustent comme sans a1.
- **Reproduction** :
  ```python
  y = reference_timeseries['a1'].where(reference_timeseries.index == '2023-12-31')
  imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())
  imputer.fit(reference_timeseries[['m1', 'q1', 'a2']], y)
  [step.is_fallback for step in imputer.imputation_plan_]   # [True, True]
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_robustness.py::TestInputEdgeCases::test_undetectable_column_is_not_used_as_a_covariate`
- **Statut** : corrigé (2026-10-13)
- **Correction** : `_prepare_variable` ne retient comme candidates que les colonnes dotées d'une fréquence détectée pour
  au moins une entité (clés de `_detected_frequencies_by_column`) ; une colonne indétectable — ou dont tous les
  couples sont hors de leur grille calendaire (ANO-FREQ-065) — n'est jamais covariable. Sur un panel, une entité
  indétectable d'une colonne détectée ailleurs garde le traitement arbitré d'ANO-FREQ-046 (côté covariables). Test :
  plus de `xfail`.

### ANO-FREQ-061 — Entité réduite à une ligne : la fenêtre de tout le panel est perdue
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/imputation_window.py::ImputationWindowCalculator.fit` (détection de la
  fréquence d'index sur tout le `MultiIndex`), via `HighFrequencyImputer._fit_imputation_window`
- **Sévérité** : mineure
- **Observé** : une entité qui ne compte qu'une ligne fait échouer `detect_index_frequency` sur l'index entier ; le
  calculateur lève, et l'imputeur avertit « Could not calculate imputation window: Series has only 1 non-null
  observations, minimum required is 2. Using all available data. » : les trois masques valent True partout, pour
  **toutes** les entités. Sur `PANEL-F` dont m1 de DE s'arrête fin 2022, les douze mois 2023 de DE entrent dans la
  fenêtre d'imputation. `imputation_window_` devient le tuple global au lieu du dict par entité.
- **Attendu** : la règle d'ANO-FREQ-020 — une entité sans fenêtre déterminable rejoint `entities_without_window_`
  (avertissement unique), les autres gardent leurs masques. Cohérence avec l'entité entièrement NaN, déjà traitée
  ainsi.
- **Reproduction** :
  ```python
  base = mixed_freq_panel_multifrequency.copy()   # m1 de DE à NaN à partir de 2023
  lu = pd.DataFrame({'m1': [100.0], 'q1': [np.nan], 'v': [np.nan]},
                    index=pd.MultiIndex.from_tuples([('LU', pd.Timestamp('2021-01-31'))], names=base.index.names))
  HighFrequencyImputer(target_frequency='M', estimator=LinearRegression()).fit(pd.concat([base, lu]))
  # UserWarning: Could not calculate imputation window: Series has only 1 non-null observations ...
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_robustness.py::TestInputEdgeCases::test_entity_reduced_to_one_row_leaves_the_other_windows_intact`,
  `tests/unit/frequency/test_imputation_window.py::TestEntitiesWithoutWindow::test_entity_reduced_to_one_row_has_no_window_and_costs_the_others_nothing`
- **Statut** : corrigé (2026-10-13)
- **Correction** : sur un panel, `ImputationWindowCalculator.fit` détecte la fréquence d'index entité par entité
  (`_detect_entity_index_frequencies`) ; une entité indéterminable vaut `None` et rejoint `entities_without_window_`,
  avec l'avertissement unique d'ANO-FREQ-020. Les autres entités gardent leurs fenêtres.

### ANO-FREQ-062 — Entité entièrement NaN : avertissement « entities not in data » erroné
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._fit` (phase 0 : expansion
  de `target_frequency` sur `entities_`, puis `TargetFrequencyValidator.validate` contre les seules fréquences
  détectées)
- **Sévérité** : cosmétique
- **Observé** : sur `PANEL-F` dont IT est entièrement NaN, avec `target_frequency='M'` (une chaîne) : « target_frequency
  dict contains entries for entities not in data: {('IT',)}. These will be ignored. » L'utilisateur n'a passé aucun
  dict, et IT est dans les données. L'avertissement de fenêtre la nomme déjà correctement (« 1 entity(ies) without a
  determinable window (… no observation at all) »). Résultat inchangé (IT vide et non marquée, FR et DE identiques à un
  ajustement sans IT).
- **Attendu** : aucun message affirmant l'absence d'une entité présente ; au plus un message nommant l'entité comme
  jamais observée (avertissements uniques et exacts, §12.5).
- **Reproduction** :
  ```python
  data = mixed_freq_panel_multifrequency.copy(); data.loc[('IT',), :] = np.nan
  HighFrequencyImputer(target_frequency='M', estimator=LinearRegression()).fit(data)
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_robustness.py::TestInputEdgeCases::test_all_nan_entity_is_not_said_absent_from_the_data`
- **Statut** : corrigé (2026-10-13)
- **Correction** : en phase 0, les entités présentes mais sans aucune fréquence détectée (entièrement NaN, ou réduites
  à une ligne) sont retirées de la cible effective avant la validation ; seul l'avertissement de la fenêtre les nomme.
  Une entrée de dict nommant une entité réellement absente des données reste signalée par le validateur.

### ANO-FREQ-063 — Échec de prédiction d'une étape sans ancre annoncé comme repli d'interpolation
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._predict_step` (message
  accumulé quel que soit `step.unanchored`)
- **Sévérité** : cosmétique
- **Observé** : sous `impute_unobserved_entities=True`, quand la prédiction d'une étape sans ancre lève (covariable
  NaN, estimateur intolérant), l'avertissement agrégé des étapes dégradées annonce « 'v' at stage M: prediction failed
  (…), interpolation fallback », alors qu'aucun repli n'a lieu : les cellules restent NaN et non marquées, et le
  second avertissement (« unobserved (entity, column) pair(s) could not be imputed … ») nomme le couple. Deux messages
  pour un fait, dont un inexact.
- **Attendu** : §5.10, conséquence 5 — « Le repli d'une entité sans ancre ne peut pas être l'interpolation : ses
  cellules restent NaN et non marquées, et un avertissement agrégé nomme les couples concernés. »
- **Reproduction** : `HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
  impute_unobserved_entities=True).fit(panel_f_with_a_mute_entity)` (fixture du paquet).
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_mutualization.py::TestUnobservedEntities::test_unanchored_prediction_failure_announces_no_fallback`
- **Statut** : corrigé (2026-10-13)
- **Correction** : `_predict_step` n'accumule plus de message « interpolation fallback » pour une étape sans ancre ;
  seul l'avertissement des couples non imputés la nomme, reformulé : « no prediction on the target grid (no usable
  covariate, or the prediction failed), and no interpolation fallback exists without an anchor ».

### ANO-FREQ-064 — Prédictions entièrement NaN : rien n'est écrit, ni repli, ni avertissement
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._execute_step`
- **Sévérité** : à arbitrer
- **Observé** : un estimateur dont `predict` rend des NaN (sans lever) produit une étape figée comme un succès
  (`is_fallback=False`, `n_written=0`) ; aucune cellule n'est écrite et aucun avertissement n'est émis. Une prédiction
  partiellement NaN laisse de même des trous silencieux. Un échec de prédiction par exception, lui, dégrade l'étape
  en repli d'interpolation et l'annonce.
- **Attendu** : à arbitrer — (i) comportement actuel, NaN jamais écrit (cohérent avec la règle « une ligne que les
  covariables ne nourrissent pas rend un NaN, qui n'est jamais écrit », §5.10) ; (ii) repli d'interpolation et
  avertissement, comme pour une exception (§12.3 : « le plan dise toujours ce qui a réellement été fait ») ; (iii)
  au moins un avertissement agrégé nommant les étapes restées vides.
- **Reproduction** : `HighFrequencyImputer(target_frequency='M', estimator=ConstantEstimator(constant=np.nan)).fit(reference_timeseries)`
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_fit_phases.py::TestPhaseFiveEdgeCases::test_all_nan_predictions_fall_back_and_warn`,
  `::test_partly_nan_predictions_are_left_unwritten_and_named`
- **Statut** : corrigé (2026-10-13) — option (ii) retenue, recommandée à l'auteur et à confirmer
- **Correction** : des prédictions entièrement NaN sont traitées comme un échec de prédiction — repli d'interpolation,
  étape figée `is_fallback=True`, message « the model predicted only NaN, interpolation fallback » dans
  l'avertissement agrégé des étapes dégradées. Des NaN épars restent des cellules non écrites, nommées par le même
  avertissement (« N of K prediction(s) are NaN and stay unimputed »). Une étape sans ancre n'a pas de repli : ses
  couples sont nommés par leur propre avertissement.

### ANO-FREQ-065 — Ancres basse fréquence hors du calendrier civil : aucune ligne d'entraînement
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._training_mask_at` /
  `TrainingSetBuilder` (masque 'training' lu par date exacte), face à
  `tsforecast/frequency/aggregation_constraint.py` (rattachement par période)
- **Sévérité** : à arbitrer
- **Observé** : une colonne annuelle dont les ancres ne tombent pas aux bornes civiles — exercice clos au 30 juin sur
  un index `ME`, ou ancre datée au 1er décembre sur un index `MS` (ce que produit `to_period_start` appliqué tel
  quel à `PANEL-X`) — est détectée `'Y'`. Le masque 'training' annuel ne porte que des dates civiles (31 décembre ou
  1er janvier) : les ancres en sortent, l'étape retombe sur le repli sans ligne d'entraînement (« empty training set
  (0 row(s), …) »), et le repli n'écrit qu'une partie de la grille (30 mois sur 36 pour l'exercice décalé ; 75 cellules
  sur 108 pour `to_period_start(PANEL-X)`, où **toutes** les étapes tombent en repli). `AggregationConstraint`, elle,
  rattache ces mêmes ancres à leur année civile (`Period(date, 'Y')`) : les deux composants ne lisent pas l'ancre de
  la même façon.
- **Attendu** : à arbitrer — (i) refuser ou signaler explicitement des ancres hors de la grille calendaire de leur
  fréquence ; (ii) lire les ancres par période (comme le recalage) dans le masque d'entraînement ; (iii) supporter
  les exercices décalés (fréquence ancrée `YE-JUN`). La spec ne traite que des périodes civiles.
- **Reproduction** :
  ```python
  data = reference_timeseries[['m1']].copy()
  data['a1'] = pd.Series([120.0, 132.0, 150.0],
                         index=pd.to_datetime(['2021-06-30', '2022-06-30', '2023-06-30'])).reindex(data.index)
  HighFrequencyImputer(target_frequency='M', estimator=LinearRegression()).fit(data).imputation_plan_
  # a1 : is_fallback=True, n_training_rows=0
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_robustness.py::TestPeriodPositions::test_off_calendar_anchors_are_left_out_with_one_warning`,
  `::test_off_calendar_quarter_anchors_of_one_entity_only`
- **Statut** : corrigé (2026-10-13) — option (i) retenue, recommandée à l'auteur et à confirmer
- **Correction** : `_misaligned_anchor_pairs` repère, en phase 0, les couples mensuels, trimestriels ou annuels dont
  une observation n'est pas sur la borne calendaire de sa période (fin sous un index de fin de période, début sous un
  index de début ; contrôle omis pour une entité sans position d'index, journalière ou ouvrée). Ces couples sont
  écartés de `detected_frequencies_` — ni imputés, ni covariables (ANO-FREQ-060), ni traités comme sans ancre — et un
  `UserWarning` unique les nomme en invitant à les redater. Leurs observations ressortent telles quelles. Le support
  des exercices décalés (fréquences ancrées `YE-JUN`) reste hors périmètre : il supposerait que la détection, les
  masques de fenêtre, les agrégations et le recalage portent tous le mois d'ancrage.

### ANO-FREQ-066 — Sélection des covariables lue à la fréquence du bloc, et non par couche (D30)
- **Type** : [SPEC] code ≠ spec
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._prepare_variable`
  (vue `train_view` construite à `_block_binding(blocks, …)`, lue par `_select_feature_columns`)
- **Sévérité** : mineure
- **Observé** : sous `impute_intermediate_frequencies=True`, la vue d'entraînement qui décide de la sélection des
  covariables est matérialisée à la fréquence du **bloc**, sur toutes les dates de la grille, y compris celles des
  couches plus fines. Une covariable vide à la fréquence du bloc mais pleine sur une couche plus fine est écartée.
  Cas limite mesuré : `TS[['m1', 'q1', 'a1']]` dont m1 s'arrête en novembre 2021 ; à l'étape M, les trois seules lignes
  de a1 sont les trimestres T1-T3 2021 écrits par le repli de l'étape Q (couche Q, couche Y vide) ; m1 y vaut la somme
  de chaque trimestre, mais lue à la fréquence Y sur ces dates elle est vide : m1 est écartée, et a1 retombe en repli
  avec trois lignes exploitables. Plus largement : une covariable mensuelle dont les années sont incomplètes mais les
  trimestres complets n'est jamais retenue pour une variable annuelle, même quand ses lignes trimestrielles
  pourraient la porter.
- **Attendu** : §5.4bis / D30 — « Un bloc cesse d'être homogène dès qu'une étape antérieure y a écrit des cellules
  plus fines que sa fréquence propre. Tout ce qui se lit "à la fréquence du bloc" se lit alors à la fréquence de chaque
  LIGNE, une couche à la fois. » La sélection (§7.2 : non-vacuité sur les deux fenêtres) doit lire chaque couche à sa
  fréquence ; I18 (b) : `X_train` plein sur les lignes trimestrielles.
- **Reproduction** :
  ```python
  data = reference_timeseries[['m1', 'q1', 'a1']].copy(); data.loc['2021-12-31':, 'm1'] = np.nan
  imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                 impute_intermediate_frequencies=True).fit(data)
  # étape (M, a1) : n_training_rows=3, feature_cols=(), is_fallback=True
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_fit_phases.py::TestWindowWithoutTraining::test_finer_rows_alone_keep_their_covariates`
- **Statut** : corrigé (2026-10-13)
- **Correction** : la vue d'entraînement de la sélection est construite par `_training_view`, couche par couche (une
  trame d'étape par couche, à la liaison de la couche, sur sa sous-grille), comme le masque 'training', la
  matérialisation et les diviseurs. Une seule couche — axe 2 éteint — redonne la vue d'origine.

### ANO-FREQ-067 — `to_diagnostic_frame` : `training_blocks` sérialisé dans l'ordre d'entrée des entités
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/imputation_plan.py::ImputationPlan.to_diagnostic_frame` (colonne
  `training_blocks`), alimentée par l'ordre des blocs de `TrainingSetBuilder.build`
- **Sévérité** : cosmétique
- **Observé** : sur `reverse_entities(PANEL-X)`, valeurs et provenances sont identiques à celles de l'original, mais
  la colonne `training_blocks` du diagnostic vaut `'IT=Q, FR=Q, DE=Q'` au lieu de `'DE=Q, FR=Q, IT=Q'`. La colonne
  `entities`, elle, est triée. Le diagnostic — destiné au suivi d'expériences — dépend donc de l'ordre des lignes
  d'entrée.
- **Attendu** : un diagnostic indépendant de l'ordre des entités en entrée, comme le plan (`_imputable_groups` : « l'ordre
  du plan ne doit dépendre ni de l'ordre des colonnes ni de celui des lignes en entrée ») — sérialisation triée par clé
  d'entité.
- **Reproduction** :
  ```python
  imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                 covariate_eligibility='all_entities').fit(reverse_entities(panel_reference_full))
  imputer.imputation_plan_.to_diagnostic_frame()['training_blocks'][0]   # 'IT=Q, FR=Q, DE=Q'
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_robustness.py::TestPerturbationInvariance::test_diagnostic_frame_is_unchanged_up_to_renaming[reverse_entities]`,
  `tests/unit/frequency/test_imputation_plan.py::TestDiagnosticFrame::test_training_blocks_rendering_ignores_the_entity_order`
- **Statut** : corrigé (2026-10-13)
- **Correction** : `to_diagnostic_frame` trie `training_blocks` par clé d'entité. Le test d'or
  `TestDiagnosticFrame::test_golden_row_of_a_panel_step`, qui figeait l'ordre d'insertion (`'FR=M, DE=Q'`), attend
  désormais `'DE=Q, FR=M'` (changement délibéré).

### ANO-FREQ-068 — « Panel structure issues: Missing panel columns » sous `panel_cols` converties en index
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/base/transformers.py::PanelTimeSeriesTransformer._check_panel_consistency`
- **Sévérité** : cosmétique
- **Observé** : sous `time_col` / `panel_cols` et `convert_cols_to_index=True` (le cas de `HighFrequencyImputer`), la
  validation fait passer les colonnes de panel dans l'index, puis le contrôle de cohérence les cherche parmi les
  colonnes et avertit « Panel structure issues: Missing panel columns: {'country'} ».
- **Attendu** : aucun avertissement — l'identifiant est présent, comme niveau d'index.
- **Test** : `tests/unit/base/transformers/test_structure.py::TestPanelIdentifierConsistency::test_panel_cols_moved_into_the_index_are_not_missing`
- **Statut** : corrigé (2026-10-13)
- **Correction** : l'identifiant se lit parmi les colonnes ou les niveaux d'index ; ses valeurs manquantes aussi.

### ANO-FREQ-069 — Message de tri conseillant `auto_sort=True` à un transformateur qui ne l'expose pas
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/base/transformers.py::PanelTimeSeriesTransformer` (validations MultiIndex et par colonnes)
- **Sévérité** : cosmétique
- **Observé** : un panel non trié passé à `HighFrequencyImputer` lève « Panel entities are not contiguous in MultiIndex.
  Consider sorting by entity levels or enabling auto_sort=True » ; l'imputeur fige `auto_sort=False` (§12.5) et ne
  l'expose pas.
- **Attendu** : un conseil réalisable.
- **Test** : `tests/unit/base/transformers/test_validation.py::TestSortingAdvice`,
  `tests/unit/frequency/high_frequency_imputer/test_robustness.py::TestInputEdgeCases::test_unsorted_panel_is_refused`
- **Statut** : corrigé (2026-10-13)
- **Correction** : `_sorting_advice` ne propose `auto_sort=True` que si le paramètre figure dans `get_params()` ;
  sinon « Sort the data by … before calling <classe>, which never sorts implicitly ».

### ANO-FREQ-070 — Frame à une seule colonne : « Could not calculate imputation window … min_columns=2 »
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._make_window_calculator`
- **Sévérité** : cosmétique
- **Observé** : l'imputeur imposait `min_columns=2` au calculateur (depuis l'implémentation initiale, sans
  justification) ; un frame d'une seule colonne émettait « Could not calculate imputation window: Data has 1 columns,
  but min_columns=2. Using all available data. » au `fit` et au `transform`.
- **Attendu** : une colonne suffit à définir une fenêtre (sa propre couverture) : aucun avertissement.
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_robustness.py::TestInputEdgeCases::test_no_low_frequency_column_gives_an_empty_plan`
- **Statut** : corrigé (2026-10-13)
- **Correction** : le calculateur garde son défaut (`min_columns=1`).

### ANO-FREQ-071 — Couvertures disjointes : trois avertissements pour une cause, dont un inexact
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/imputation_window.py::ImputationWindowCalculator._compute_window` /
  `_fit_panel`, `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._fit` (phase 1) et `_fit_variable`
- **Sévérité** : mineure
- **Observé** : sans période couverte par toutes les colonnes, le calculateur avertissait « There is no period in the
  DataFrame … » (une fois **par entité** sur un panel, sans la nommer), l'imputeur ajoutait « The training window is
  empty: … all imputations will fall back to interpolation » — inexact : la fenêtre d'imputation est vide aussi et
  rien n'est écrit — puis le message des étapes dégradées annonçait un repli pour chaque variable.
- **Attendu** : un message unique et exact (§12.5).
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_fit_phases.py::TestWindowWithoutTraining::test_disjoint_coverages_warn_once_and_impute_nothing`,
  `tests/unit/frequency/test_imputation_window.py::TestNoStrictWindow::test_time_series_message_says_nothing_is_imputed`,
  `::test_entities_without_strict_window_are_named_by_one_warning`
- **Statut** : corrigé (2026-10-13)
- **Correction** : `_compute_window` signale l'absence de fenêtre stricte (`no_strict_window`) sans avertir ;
  `_fit_ts` et `_fit_panel` émettent un message unique, qui nomme les entités d'un panel et dit qu'aucune valeur n'est
  imputable. L'imputeur ne double plus ce message : une fenêtre d'entraînement vide implique une fenêtre d'imputation
  vide (les deux contiennent la fenêtre stricte, aucune extension ne naît d'une fenêtre stricte vide), et un repli
  n'est plus annoncé quand la grille de prédiction est vide et qu'aucun groupe sans ancre ne prédit.

### ANO-FREQ-072 — Panel sans fenêtre stricte pour aucune entité : imputé sur toutes ses lignes
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._fit` (phase 1, repli sur des
  masques tout-vrais), face à `tsforecast/frequency/imputation_window.py::ImputationWindowCalculator._fit_panel`
- **Sévérité** : mineure
- **Observé** : sur un panel dont aucune entité n'a de période couverte par toutes ses colonnes, le calculateur lève
  « No imputation window found for any entity in the panel » (comportement voulu du composant, testé) ; l'imputeur
  attrape l'erreur, avertit « Could not calculate imputation window: … Using all available data » et impute **toutes**
  les lignes. Les mêmes données en série temporelle n'imputent **rien** (fenêtres vides, un message). Mesuré sur
  `PANEL-X[['m1', 'q1', 'a1']]`, m1 limitée à 2021 et q1 à 2023 : 108 cellules écrites pour q1 et pour a1 sur le panel,
  0 sur `TS`.
- **Attendu** : le comportement de la série temporelle, cohérent avec §7 / B23 — une entité sans fenêtre stricte n'a
  rien à imputer, que les autres entités en aient une ou non. Le repli « Using all available data » est à réserver
  aux échecs qui ne disent rien des fenêtres (il n'est plus atteint pour un frame d'une colonne, ANO-FREQ-070).
- **Reproduction** :
  ```python
  data = panel_reference_full[['m1', 'q1', 'a1']].copy(); dates = data.index.get_level_values(-1)
  data.loc[dates >= '2022-01-01', 'm1'] = np.nan; data.loc[dates <= '2022-12-31', 'q1'] = np.nan
  imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression()).fit(data)
  [step.n_written for step in imputer.imputation_plan_]   # [108, 108] ; sur data.loc['FR'] : [0, 0]
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_fit_phases.py::TestWindowWithoutTraining::test_panel_without_any_strict_window_imputes_nothing`,
  `tests/unit/frequency/test_imputation_window.py::TestFitInputValidation::test_accepts_panel_where_no_entity_has_a_window`,
  `::TestNoStrictWindow::test_panel_without_window_behaves_like_a_time_series`,
  `tests/unit/frequency/high_frequency_imputer/test_robustness.py::TestInputEdgeCases::test_all_nan_entity_is_named_once_at_transform`
- **Statut** : corrigé (2026-10-13) — arbitrage de l'auteur : aligner le panel sur la série temporelle
- **Correction** : `ImputationWindowCalculator._fit_panel` ne lève plus quand aucune entité n'a de fenêtre stricte :
  comme une série (B23), les trois masques existent — tout faux, tout vrai à l'entraînement sous
  `training_scope='unrestricted'` — et l'avertissement unique d'ANO-FREQ-071 nomme les entités ; le message « No entity
  has a strict imputation window … » disparaît, redondant. Les deux tests du calculateur qui figeaient l'erreur
  (`test_rejects_panel_where_no_entity_has_a_window`, `test_panel_without_window_raises_by_default`) sont réécrits sur le
  nouveau contrat (changement délibéré). Au `transform`, `_warn_rows_outside_window` ne recompte plus les lignes d'une
  entité — ou d'une série — sans aucune ligne dans la fenêtre : le calculateur l'a déjà nommée (même règle pour une
  entité entièrement NaN). Le repli « Using all available data » du `fit` (masques tout-vrais sur échec du
  calculateur), devenu inatteignable, est supprimé à la demande de l'auteur : un échec du calcul de fenêtre au `fit`
  lève désormais. Son pendant au `transform` reste — une seule ligne à transformer ne porte aucune fenêtre.


### ANO-FREQ-073 — Entité née au `transform` : ses colonnes observées sont servies sans ancre, ses observations écrasées
- **Type** : [SPEC] code ≠ spec (D34 appliqué à la lettre, contraire à I4 et I7)
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._new_entity_steps`
  (et `_execute_step` pour les étapes `unanchored=True`)
- **Sévérité** : majeure
- **Observé** : sous `impute_unobserved_entities=True`, une entité absente du `fit` reçoit, pour **chaque** colonne
  imputée de l'étape, une étape dérivée `unanchored=True` (`source_frequency=None`, facteurs à 1.0), qu'elle observe
  la colonne ou non dans les données du `transform`. Sur `PANEL-F` augmenté d'`ES`, copie de `DE` (donc `v` et `q1`
  observées chaque trimestre) : les 36 cellules mensuelles de `v` d'`ES` sont des prédictions libres marquées
  `MODEL_UNANCHORED`, jamais recalées — à la date d'observation `2021-03-31`, 9.66 au lieu d'une désagrégation de 28.0,
  et la somme du trimestre vaut 28.74 au lieu de 28.0 (12 trimestres sur 12 faux). Quand la prédiction d'une telle
  étape échoue (`q1` d'`ES`, dont la covariable `v` est vide quand `ES` copie `IT`), l'avertissement agrégé nomme le
  couple comme **non observé** — « 1 unobserved (entity, column) pair(s) could not be imputed … [('ES', 'q1')] » —
  alors qu'`ES` observe `q1` douze fois ; la même entité présente au `fit` aurait vu `q1` imputée normalement.
- **Attendu** : I7 — le `transform` ne détruit jamais une observation d'entrée — et I4 — chaque période complète
  observée garde son total. D34 motive l'étape sans ancre par « une entité sans historique est, par construction, une
  entité sans ancre », mais énonce aussi que sa fréquence « ne peut venir que des données du transform » : la
  prémisse ne tient que pour les couples que le `transform` n'observe pas. Deux corrections possibles, à arbitrer par
  l'auteur : (a) dériver, pour un couple observé, une étape **ancrée** (`source_frequency` = fréquence détectée au
  `transform`, recalage, provenance `MODEL_*`) ; (b) laisser intact un couple observé et ne servir sans ancre que les
  couples jamais observés. Le test ne tranche pas : il exige que chaque trimestre observé somme à son observation,
  vrai sous (a) comme sous (b).
- **Reproduction** :
  ```python
  added = panel_f.xs('DE', level=0, drop_level=False).copy()
  added.index = pd.MultiIndex.from_arrays([['ES'] * 36, added.index.get_level_values(-1)],
                                          names=panel_f.index.names)
  extended = pd.concat([panel_f, added]).sort_index()
  imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                 impute_unobserved_entities=True, keep_lower_frequencies=False).fit(panel_f)
  out = imputer.transform(extended).loc['ES', 'v']
  out.groupby(out.index.to_period('Q')).sum().head(2)   # 28.74, 29.61 ; observé : 28.0, 30.0
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_transform.py::TestNewEntityAtTransform::test_observations_of_a_new_entity_are_never_overwritten`,
  `::test_new_entity_copy_is_imputed_like_its_model`, `::test_column_empty_at_fit_and_observed_at_transform_is_anchored`,
  `::test_pairs_unknown_to_the_fit_stay_as_given_by_default`
- **Statut** : corrigé (2026-10-14) — arbitrage de l'auteur : option (a), D34 inchangée sous `False`
- **Correction** : D42 (§14.10, §5.10 « Au `transform` »). `_new_entity_steps` devient `_replay_steps` : sous
  `impute_unobserved_entities=True`, un couple que le fit n'a jamais observé — colonne d'une entité nouvelle, **ou**
  colonne qu'une entité connue laissait vide au fit (même défaut : son étape sans ancre du plan écrasait ses
  observations) — est servi comme s'il avait été présent au fit. Observé par le `transform` (fréquence détectée sur
  ses données, `_check_transform_frequencies`), il rejoint une étape **ancrée** dérivée de l'étape de sa colonne
  (même modèle `is`, mêmes `feature_cols` et voies, `source_frequency` détectée, facteur de groupe, recalage,
  `MODEL_*`), à toute étape où sa fréquence est plus grossière que celle de l'étape ; jamais observé, il garde
  l'étape sans ancre de D34. Une entité connue quitte l'étape sans ancre du plan pour l'étape ancrée. Les étapes
  dérivées suivent celles de leur colonne (ordre des colonnes du plan conservé). Un couple observé hors des bornes
  calendaires de sa fréquence est écarté comme au fit (`_misaligned_anchor_pairs`, note agrégée). Sous `False`,
  rien ne change : l'entité nouvelle reste telle quelle. Mesuré : `ES` copie de `DE` reçoit exactement les valeurs
  et provenances de `DE` ; l'avertissement « unobserved pair » ne nomme plus un couple observé.

### ANO-FREQ-074 — Colonne sans aucune observation dans les données du `transform` : prédite sans ancre, marquée `MODEL_ON_*`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._transform` (rejeu des
  fréquences du `fit` pour un couple que les données du `transform` n'observent pas), en interaction avec la fenêtre
  recalculée (`ImputationWindowCalculator`, D39)
- **Sévérité** : majeure
- **Observé** : `a1` non publiée pour 2023. `transform(TS)` entier : les lignes 2023 sortent de la fenêtre recalculée,
  `a1` y reste NaN et non marquée. `transform(TS.loc['2023'])` — les **mêmes** lignes, sans l'historique : `a1` n'a
  plus aucune observation, elle quitte le dénominateur de la couverture (D39, avertissement « Columns without any
  observation … »), la fenêtre couvre toute l'année et le modèle figé prédit les douze mois (11.90 … 12.99) sans
  ancre ni recalage, marqués `MODEL_ON_INTERPOLATED`. Même comportement sur un panel pour une entité connue du `fit`
  dont la colonne est vide au `transform` (`PANEL-X[['m1', 'q1', 'a1', 'a2']]`, `a1` de `FR` effacée : 36 cellules
  écrites), sous `impute_unobserved_entities=False`.
- **Attendu** : §5.10 / D33 — un couple sans aucune observation n'a rien à désagréger : sous
  `impute_unobserved_entities=False` (défaut) ses cellules restent NaN et non marquées ; sous `True` elles portent
  `MODEL_UNANCHORED`, la provenance qui annonce l'absence d'ancre, de recalage et de diviseur. Une famille `MODEL_ON_*`
  sur ces cellules annonce une désagrégation qui n'a pas eu lieu. Le résultat d'une ligne ne doit pas dépendre de la
  présence de l'historique dans l'appel. Au `fit`, un tel couple est déjà écarté (fréquence indétectable) ; au
  `transform`, la fréquence rejouée depuis le `fit` le rend imputable.
- **Reproduction** :
  ```python
  imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                 keep_lower_frequencies=False).fit(ts)
  data = ts.copy(); data.loc['2023', 'a1'] = np.nan
  imputer.transform(data).loc['2023', 'a1'].notna().sum()        # 0
  imputer.transform(data.loc['2023'])['a1'].notna().sum()        # 12
  imputer.imputation_provenance_['a1'].unique()                  # [MODEL_ON_INTERPOLATED]
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_transform.py::TestColumnWithoutAnchorAtTransform::test_unpublished_year_alone_is_predicted_by_its_replayed_step`,
  `::test_extended_window_predicts_it_with_its_history`, `::test_unpublished_year_stays_empty_with_its_history`
- **Statut** : clos (2026-10-14) — arbitrage de l'auteur : le comportement est voulu
- **Correction** : aucun changement de code. Le couple est **observé au fit** : §12.1 rejoue sa classification, et
  son étape reste ancrée. Si le fit a entraîné un modèle pour la colonne, il l'impute au `transform` même quand
  l'appel ne la publie pas. Les périodes sans observation sont prédites sans recalage et marquées `MODEL_*`, comme
  dans la fenêtre de prédiction étendue du §7.3 (pseudo temps réel). Sous `extended_forward`, la même année 2023 est
  imputée avec l'historique (11.84…) comme sans lui (11.90…). L'attente « reste NaN sauf sous
  `impute_unobserved_entities=True` » appliquait §5.10, qui ne concerne que les couples jamais observés au fit. La
  différence entre les deux appels tient à la fenêtre recalculée, qui coupe aussi `q1` sous `strict` : D39 élargit
  la fenêtre quand l'appel n'observe pas la colonne, et l'avertit. Le test `xfail` est réécrit sur ce contrat ; la
  spec le dit au §5.10.

### ANO-FREQ-075 — Lignes d'entrée ajoutées à l'étape finale (D38) : sortie non triée, entités non contiguës
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._replay_plan`
  (`grid = grid.append(absent)` à l'étape finale)
- **Sévérité** : majeure
- **Observé** : la grille de l'étape finale est complétée des lignes d'entrée qu'elle ne couvre pas (D38) par un
  `append` **en fin d'index**. Dès qu'une entité a une fréquence cible plus grossière que l'index — `PANEL-F` sous
  `target_frequency={'FR': 'M', 'DE': 'Q', 'IT': 'M'}`, avec ou sans axe 2, empilé ou non — ses 24 mois hors fin de
  trimestre sont rejetés après `IT` : la sortie n'est pas triée (`is_monotonic_increasing` faux), les blocs d'entité
  ne sont plus contigus, `inverse_transform` rend les lignes de X dans un autre ordre, et cette inversion, repassée à
  un `HighFrequencyImputer`, lève « Panel entities are not contiguous in MultiIndex ». Même cas sur une série
  temporelle mensuelle sous `target_frequency='Q'` (fins de trimestre d'abord, autres mois ensuite).
- **Attendu** : la sortie garde l'ordre de l'entrée — §12.5 refuse un index non trié plutôt que de le trier en
  silence ; un composant ne peut pas produire ce qu'il refuse. I8 : l'inversion restitue l'index de X, ordre compris.
  Compléter la grille puis la trier (ou réindexer sur l'union ordonnée) suffit.
- **Reproduction** :
  ```python
  imputer = HighFrequencyImputer(target_frequency={'FR': 'M', 'DE': 'Q', 'IT': 'M'},
                                 estimator=LinearRegression(), restore_original_values=True)
  out = imputer.fit_transform(panel_f)
  out.index.is_monotonic_increasing                             # False
  imputer.inverse_transform(out).index.equals(panel_f.index)    # False (même ensemble, autre ordre)
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_transform.py::TestKeepLowerFrequencies::test_output_index_stays_sorted`,
  `tests/unit/frequency/high_frequency_imputer/test_inverse_transform.py::TestRoundTripRestoresObservations::test_inverse_restores_the_row_order`
- **Statut** : corrigé (2026-10-14) — arbitrage de l'auteur : la complétion est retirée
- **Correction** : D43 (§14.10, D38 amendée). La complétion de la grille finale par `grid.append(absent)` est
  supprimée : sur huit jeux mesurés, les lignes ajoutées ne portaient jamais de valeur. Elles n'apparaissent qu'avec
  une cible plus grossière que l'index, où leur valeur est agrégée sur la ligne de sa période. Chaque grille d'étape
  passe par `_in_input_order` : entités dans leur ordre d'apparition en entrée, dates triées. Cela couvre aussi
  l'entité omise par le calculateur, que `_stage_mask` rattachait en fin de masque. L'inversion se réindexe sur
  l'entrée de la dernière passe (ANO-FREQ-076) : elle rend les lignes de `X` dans leur ordre, y compris les mois de
  `DE` absents de la sortie.

### ANO-FREQ-076 — Index irrégulier : `inverse_transform` rend les lignes que la grille cible a ajoutées
- **Type** : [SPEC] code ≠ spec (I8 : « restitue l'index »)
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._inverse_transform`
- **Sévérité** : mineure
- **Observé** : sur un index irrégulier (ancres annuelles antérieures à la grille mensuelle), la grille cible densifie
  la période (comportement voulu, `TestIrregularIndexExtendedWindow`) ; l'inversion la garde telle quelle.
  `irregular_index_timeseries` : 82 lignes en entrée, 115 rendues (33 mois de 2015-2017 ajoutés, entièrement NaN après
  inversion, sous `restore_original_values=True` comme `False`). `heterogeneous_coverage_panel` : 225 → 318 lignes.
  Aucune ligne d'entrée ne manque et aucune valeur n'est fausse ; seul l'index n'est pas celui de X. Le test existant
  `TestInverseTransform::test_inverse_transform_preserves_multilevel_entity_names` tolère déjà, par un `isin`, la même
  ligne supplémentaire sur `panel_two_level_dataset` (« ne relève pas de l'inversion »).
- **Attendu** : I8 — `inverse_transform(transform(X))` restitue l'index de X. L'inversion défait la dernière passe,
  dont elle garde l'instantané d'entrée (`_original_X_`) : se réindexer sur son index rend X sans perte, les lignes
  ajoutées ne portant plus rien après la remise à NaN. À arbitrer si l'auteur préfère garder la grille densifiée
  (l'invariant serait alors à reformuler en « contient l'index de X »).
- **Reproduction** :
  ```python
  imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                 restore_original_values=True)
  restored = imputer.inverse_transform(imputer.fit_transform(irregular_ts))
  len(irregular_ts), len(restored)                                   # 82, 115
  restored.loc[restored.index.difference(irregular_ts.index)].isna().all().all()   # True
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_inverse_transform.py::TestRoundTripRestoresObservations::test_inverse_drops_the_rows_the_grid_added`,
  `::TestInverseTransform::test_inverse_transform_preserves_multilevel_entity_names` (resserré : égalité d'index),
  `tests/unit/frequency/high_frequency_imputer/test_transform.py::TestTargetCoarserThanTheIndex::test_inverse_gives_back_the_monthly_index`
- **Statut** : corrigé (2026-10-14) — arbitrage de l'auteur : réindexer sur l'index d'entrée
- **Correction** : D43 (§12.4, §14.10). `_inverse_transform` réindexe, après la remise à NaN, sur l'index de
  l'instantané d'entrée de la dernière passe (`_original_X_`), dans son ordre. Les lignes que la grille a ajoutées
  disparaissent, et celles qu'elle ne porte pas (cible plus grossière que l'index) reviennent, vides ou restaurées
  sous `restore_original_values=True`. I8 tient tel qu'écrit, ordre compris.

### ANO-FREQ-077 — `provenance_statistics_` décrit le `fit`, ses docstrings « la dernière passe »
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer` (section `Attributes`),
  `tsforecast/tracking/metrics.py::imputation_metrics`
- **Sévérité** : mineure
- **Observé** : `provenance_statistics_` n'est écrit qu'au `fit` (phase 6) ; un `transform` le laisse intact, alors
  qu'il réécrit `imputation_provenance_`. La docstring de la classe l'annonce « computed on the last pass » ; celle
  d'`imputation_metrics` lit « `provenance_statistics_` when present (the last ``transform``), otherwise
  `provenance_statistics_fit_` (the ``fit``) » — conception de la v1, supprimée au commit `2e6e595` (L13-0) :
  `provenance_statistics_fit_` n'existe plus et la branche de repli d'`imputation_metrics` est morte.
- **Attendu** : le code est cohérent avec §12.1 (le `transform` ne mute pas l'état ajusté) ; les docstrings doivent
  dire que les statistiques décrivent le `fit`, et `imputation_metrics` ne plus mentionner l'attribut disparu. Si
  l'auteur veut au contraire des statistiques du dernier `transform`, c'est un changement de code (et une exception de
  plus à la liste des attributs réécrits par passe).
- **Reproduction** :
  ```python
  imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression()).fit(ts)
  stats = imputer.provenance_statistics_
  imputer.transform(ts.loc['2021'])
  imputer.provenance_statistics_ is stats, hasattr(imputer, 'provenance_statistics_fit_')   # True, False
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_transform.py::TestTransformState::test_provenance_statistics_describe_the_last_pass`,
  `tests/unit/tracking/test_metrics.py::test_imputation_metrics_describe_the_last_pass`,
  `tests/unit/frequency/test_provenance.py::TestComputeStatistics::test_module_function_matches_the_tracker`
- **Statut** : corrigé (2026-10-14) — arbitrage de l'auteur : changement de code, les statistiques suivent la
  dernière passe
- **Correction** : D45 (§13.2, §14.10). Le calcul de `ImputationProvenanceTracker.compute_statistics` est extrait
  dans la fonction de module `provenance_statistics(matrix)`, exportée par `tsforecast.frequency`. `_transform`
  réécrit `provenance_statistics_` sur le niveau cible de sa sortie, quel que soit `keep_lower_frequencies`, qui
  n'est qu'un affichage. `fit` le purge en phase 0 avec l'état de transform. Les docstrings de la classe disent
  « dernière passe » à bon droit. `tracking.imputation_metrics` ne lit plus que `provenance_statistics_` : le repli
  mort sur `provenance_statistics_fit_` est supprimé, et la docstring invite à loguer juste après la passe décrite.
  `PER_PASS_ATTRIBUTES` des tests compte désormais les deux attributs.

### ANO-FREQ-078 — Un `transform` qui lève a déjà écrasé l'instantané lu par l'inversion
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._transform` (phase 0' :
  `_original_X_` / `_original_y_` écrits avant `_check_transform_frequencies`)
- **Sévérité** : majeure
- **Observé** : `_transform` copie son entrée dans `_original_X_` **avant** le contrôle des colonnes (D11). Un appel
  refusé (`ValueError` « transform is missing 1 column(s) seen at fit time ») laisse donc l'instantané de l'entrée
  refusée face à la matrice de provenance de la passe précédente. `inverse_transform(sortie_précédente)` sous
  `restore_original_values=True` restaure alors les valeurs de l'appel refusé : ancres de `a1` rendues 240 / 264 / 300
  au lieu de 120 / 132 / 150, et `q1` (absente de l'entrée refusée) plus restaurée du tout (0 cellule sur 12).
- **Attendu** : un appel qui lève n'est pas une passe : il laisse l'imputeur tel qu'il l'a trouvé. Les contrôles
  (colonnes, nom de niveau réservé, alignement de `y`) doivent précéder l'instantané.
- **Reproduction** :
  ```python
  imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                 restore_original_values=True)
  out = imputer.fit_transform(ts)
  try: imputer.transform(ts.drop(columns=['q1']).assign(a1=ts['a1'] * 2))
  except ValueError: pass
  imputer.inverse_transform(out)['a1'].dropna().tolist()   # [240.0, 264.0, 300.0]
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_transform.py::TestTransformFrequencyControl::test_failed_transform_leaves_the_last_pass_intact`
- **Statut** : corrigé (2026-10-14)
- **Correction** : D45. `_transform` copie son entrée dans des variables locales et ne publie l'état de la passe
  (`_original_X_`, `_original_y_`, `_stacked_index_nlevels`, `imputation_provenance_`, `provenance_statistics_`)
  qu'en phase 6', d'un seul tenant, quand plus rien ne peut lever. Le test vérifie aussi que la matrice et les
  statistiques de la passe précédente sont intactes après l'appel refusé.

### ANO-FREQ-079 — Lignes complétant l'étape finale (D38) : vides en sortie, marquées `ORIGINAL`
- **Type** : [CODE] comportement (et question de spec à arbitrer sur le contenu de ces lignes)
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._replay_plan`
  (instantané de provenance réindexé sur la grille complétée)
- **Sévérité** : mineure
- **Observé** : sous `target_frequency={'FR': 'M', 'DE': 'Q', 'IT': 'M'}` sur `PANEL-F`, les 24 mois de `DE` hors fin
  de trimestre sont ajoutés à la frame finale (D38), mais la frame d'étape, construite à la liaison `DE: 'Q'`, n'y
  porte rien : ces lignes sont **vides** en sortie (`m1` compris, pourtant observée chaque mois). La matrice de
  provenance, elle, reprend l'instantané du traceur initialisé sur l'entrée, et marque `ORIGINAL` les 24 cellules de
  `m1` — 24 cellules `ORIGINAL` vides, empilé ou non. La règle que le rejeu énonce lui-même (« une cellule restée NaN
  n'est jamais marquée, au rejeu comme au fit », cf. ANO-FREQ-050) n'y tient pas. Sans `restore_original_values`,
  l'inversion rend ces 24 cellules vides ; avec, elle les restaure depuis l'instantané.
- **Attendu** : la matrice décrit la sortie : aucune cellule `ORIGINAL` vide. Question ouverte pour l'auteur sur le
  **contenu** des lignes complétées : D38 les ajoute au nom d'I7 (« ne détruit jamais une observation d'entrée »), ce
  qui plaide pour y porter les valeurs d'entrée ; mais `m1` y côtoierait alors, à la même fréquence de sortie, les
  sommes trimestrielles des fins de trimestre (100, 101, 303…), mélange d'échelles que D37 interdit ailleurs. Le test
  n'exige que la cohérence provenance ↔ valeurs, vraie dans les deux options (valeurs portées et `ORIGINAL`, ou lignes
  vides et non marquées).
- **Reproduction** :
  ```python
  imputer = HighFrequencyImputer(target_frequency={'FR': 'M', 'DE': 'Q', 'IT': 'M'},
                                 estimator=LinearRegression(), keep_lower_frequencies=False)
  out = imputer.fit_transform(panel_f)
  out.loc[('DE', '2021-01-31')].tolist()                           # [nan, nan, nan]
  imputer.imputation_provenance_.loc[('DE', '2021-01-31'), 'm1']   # ProvenanceType.ORIGINAL
  ```
- **Test** : `tests/unit/frequency/high_frequency_imputer/test_transform.py::TestKeepLowerFrequencies::test_no_empty_cell_of_the_output_is_marked`
  (renommé et élargi : aucune cellule vide marquée, quelle que soit la marque)
- **Statut** : corrigé (2026-10-14) — arbitrage de l'auteur : pas de mélange d'échelles ; les lignes ne sont plus
  ajoutées
- **Correction** : D43 retire les lignes complétées (cf. ANO-FREQ-075). D44 fait décrire à la matrice les cellules
  de la sortie : `_stage_provenance` ne marque jamais une cellule vide, et marque `AGGREGATED` l'agrégation exacte
  d'une colonne plus fine que la ligne. Celle-ci portait auparavant l'`ORIGINAL` de la cellule d'entrée de même
  date : une somme trimestrielle de `m1` (303) que l'inversion sans restauration rendait pour l'observation
  mensuelle (102). Le §6.4 prescrivait déjà `AGGREGATED`.

### ANO-FREQ-080 — Imputeur ajusté avec `y` : `transform(X)` sans `y` lève, `XYPipeline.predict` et les scorers textuels échouent
- **Type** : [SPEC] code ≠ spec (I9 : conformité `Pipeline`)
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._check_transform_frequencies`
  (colonne de `y` fusionnée dans le plan au `fit` par `_fit`, exigée au `transform`)
- **Sévérité** : majeure
- **Observé** : au `fit(X, y)`, `y` est fusionnée dans la frame de travail sous son nom et entre dans le plan figé. Un
  `transform(X)` sans `y` lève alors `ValueError: transform is missing 1 column(s) seen at fit time:
  ['taux_chomage']. Every column of the fit must be present, even entirely empty: it is part of the frozen plan.`
  Dans un `XYPipeline([PublicationDelayTransformer, HighFrequencyImputer, estimateur])`, `predict(X)` passe `y=None` à
  chaque étape : un pipeline ajusté ne sait donc pas prédire. Tout scorer textuel (`scoring='neg_mean_absolute_error'`)
  appelle `predict(X)` et échoue de même ; seul `scoring=None` passe, parce que `XYPipeline.score(X, y)` transporte `y`.
- **Attendu** : I9 promet la compatibilité `Pipeline` / `GridSearchCV` ; un pipeline sklearn ajusté prédit à partir de
  `X` seul. Au `transform` sans `y`, la colonne de `y` devrait être traitée comme absente sans erreur (imputation des
  seules colonnes de `X`). La correction croise ANO-FREQ-081 : si `y` reste une covariable des colonnes de `X`, son
  absence au `predict` change les features (rupture de l'invariant I11 entre `fit` et `transform`).
- **Reproduction** :
  ```python
  data = build_mixed_frequency_timeseries(annual_start_date='2015-01-01')
  data = data[data['taux_chomage'].notna()]
  X, y = data.drop(columns='taux_chomage'), data['taux_chomage']
  pipeline = XYPipeline([('imputer', HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                                          keep_lower_frequencies=False)),
                         ('model', SpyEstimator())]).fit(X.iloc[:-6], y.iloc[:-6])
  pipeline.predict(X.iloc[-6:])   # ValueError: transform is missing 1 column(s) seen at fit time: ['taux_chomage']
  ```
- **Test** : `tests/integration/frequency/test_pipelines.py::TestPredictWithoutTarget` (le `xfail` est retiré) ;
  `tests/unit/frequency/high_frequency_imputer/test_xy_passes.py::TestPredictionPass`
- **Statut** : corrigé (2026-10-15) — arbitrage de l'auteur : l'imputeur augmente un jeu d'entraînement (`X` et
  `y`) et n'a pas vocation à transformer des données de test ; conception arrêtée avec ANO-FREQ-081 et 082
- **Correction** : D47 (spec §14, I22). Ajusté avec `y`, `transform(X, y=None)` est la **passe de prédiction** : la
  colonne cible est absente du jeu de travail (`_check_transform_frequencies(optional_columns=...)`), ses étapes ne
  sont pas rejouées (`_replay_plan(skip_columns=...)`), les colonnes de `X` sont imputées par les modèles figés et
  rendues sur les lignes d'entrée, sortie plate ; `y`, s'il est passé, revient inchangé. `fit_transform(X, y)`,
  surchargé, reste la **passe d'entraînement** (rejeu complet, `y` compris). `XYPipeline.predict(X)` et les scorers
  textuels fonctionnent. Hors du §4.3 de la campagne : modification de `tsforecast/` demandée explicitement par
  l'auteur.
### ANO-FREQ-081 — `y` covariable de l'imputation de `X` : la cible du pli de test alimente les features imputées
- **Type** : [SPEC] à arbitrer (D15 appliquée hors de son cas d'usage)
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._fit` / `_transform`
  (fusion de `y` dans la frame de travail, puis sélection des covariables)
- **Sévérité** : à arbitrer
- **Observé** : D15 fait de `y` « une colonne comme une autre ». Au `fit(X, y)`, `y` devient donc une covariable des
  modèles qui imputent les colonnes basse fréquence de `X` : sur la série réaliste, `taux_chomage` figure dans les
  `feature_cols` des deux étapes (`pib_trimestriel`, `balance_commerciale_annuelle`). Dans un `XYPipeline` évalué par
  `cross_validate(scoring=None)`, le pli de test est transformé avec `y_test` : les features imputées du pli de test
  dépendent de sa cible. Multiplier `y_test` par 10 change le `X_test` imputé.
- **Attendu** : à arbitrer. Pour le cas d'usage de D15 (désagréger une cible basse fréquence `y` à l'aide de `X`), `y`
  est la colonne imputée et son rôle de covariable des autres colonnes est accessoire. Dans un pipeline de prévision,
  la cible ne devrait pas entrer dans les features : sinon le score d'un pli mesure un modèle qui a vu sa cible, et la
  prédiction sans `y` est impossible (ANO-FREQ-080). Pistes : exclure `y` des `feature_cols` des autres colonnes ;
  ou un paramètre qui choisit le rôle de `y`.
- **Reproduction** :
  ```python
  imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                 keep_lower_frequencies=False).fit(X.iloc[:-6], y.iloc[:-6])
  [step.feature_cols for step in imputer.imputation_plan_]      # 'taux_chomage' dans chaque étape
  a, _ = imputer.transform(X.iloc[-6:], y.iloc[-6:])
  b, _ = imputer.transform(X.iloc[-6:], 10 * y.iloc[-6:])
  a.equals(b)                                                    # False
  ```
- **Test** : `tests/integration/frequency/test_pipelines.py::TestTargetRole` ;
  `tests/unit/frequency/high_frequency_imputer/test_xy_passes.py::TestTargetAsCovariate`,
  `::TestPredictionPass::test_imputed_x_ignores_the_target`
- **Statut** : corrigé (2026-10-15) — arbitrage de l'auteur, après une simulation : avec `y` covariable, l'erreur
  d'imputation de `x` baisse (0,20 contre 0,77) mais le R² d'entraînement dépasse celui de l'oracle (0,99 contre 0,97),
  le coefficient aval se biaise sous bruit fort (2,76 au lieu de 2) et l'erreur de test ne s'améliore pas
  (2,18 contre 2,00)
- **Correction** : D47. Nouveau paramètre `target_as_covariate: bool = False` : `y` est imputé à partir de `X`
  mais exclu des covariables candidates des autres colonnes (`_fit_variable`). Sous `True`, l'ancien comportement
  revient et la passe de prédiction lève (`_check_prediction_pass_replayable`) dès qu'un modèle d'une colonne de `X`
  lit `y`, en nommant les colonnes concernées. La passe de prédiction ne lit jamais `y` : le `X` imputé d'un pli de
  test ne dépend plus de sa cible.
### ANO-FREQ-082 — Pli d'entraînement ou de test qui coupe une période : la grille de l'imputeur la complète
- **Type** : [CODE] à arbitrer (interaction imputeur × validation croisée)
- **Composant** : `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._replay_plan`
  (sortie = index du masque d'imputation, grille complète par période, D38 / D43)
- **Sévérité** : à arbitrer
- **Observé** : la sortie de l'imputeur court jusqu'à la fin de la dernière période observée. Dans un `XYPipeline`
  évalué par `cross_validate` + `TSOutOfSampleSplit(n_splits=3, test_size=6)` sur la série réaliste :
  (1) le pli d'entraînement qui s'arrête en 2023-06 porte l'ancre annuelle 2023-01-01, et l'estimateur final reçoit
  72 lignes au lieu de 66, de 2023-07 à 2023-12, soit des dates du pli de test, avec une cible NaN et des valeurs
  tirées du seul pli d'entraînement. Aucune information du pli de test ne fuit : perturber le pli de test ne change
  rien à l'entraînement. Mais un estimateur qui refuse les NaN de `y` échoue au `fit`.
  (2) Avec `test_size=7`, un pli de test coupe un trimestre : l'imputeur ajoute les mois manquants, leur cible est
  NaN, et `score` lève `ValueError: Input contains NaN` (`r2_score`).
- **Attendu** : à arbitrer. La complétion de période est voulue en usage autonome (D38 : grille par période).
  En mode XY, quand l'imputeur reçoit `y`, deux options : borner la sortie à l'index de `y` (les lignes ajoutées
  n'ont aucune cible), ou documenter que l'estimateur final et le scorer doivent tolérer ces lignes.
- **Reproduction** :
  ```python
  data = build_mixed_frequency_timeseries(annual_start_date='2015-01-01')
  data = data[data['taux_chomage'].notna()]
  X, y = data.drop(columns='taux_chomage'), data['taux_chomage']
  pipeline = XYPipeline([('imputer', HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                                          keep_lower_frequencies=False)),
                         ('model', SpyEstimator())])
  cross_validate(pipeline, X, y, cv=TSOutOfSampleSplit(n_splits=3, test_size=7))   # ValueError: Input contains NaN
  ```
- **Test** : `tests/integration/frequency/test_pipelines.py::TestNoLeakage::test_test_fold_cutting_a_period_is_scored`,
  `::TestFullChainWithDropNaNTarget` ; `tests/unit/xy/test_transformers.py`
- **Statut** : corrigé (2026-10-15) — arbitrage de l'auteur : ne retirer aucune ligne dans l'imputeur (un décalage
  aval vers plusieurs horizons lit encore les lignes sans cible)
- **Correction** : D48. (2) La passe de prédiction reste sur les lignes d'entrée : un pli de test qui coupe une
  période ne gagne plus de ligne, `score` aboutit. (1) La passe d'entraînement garde ses lignes de complétion, qui
  ne portent que de l'information du pli d'entraînement (vérifié par perturbation du pli de test) ; leur retrait
  revient à la nouvelle étape `tsforecast.xy.DropNaNTarget`, placée juste avant l'estimateur, qui retire les lignes
  sans cible au `fit` et au `score` et aucune au `predict`. Limite sklearn documentée : un scorer textuel compare
  les prédictions au `y` de test brut, hors du pipeline ; une cible de test NaN ne peut y être scorée.
### ANO-FREQ-083 — Sous `covariate_strategy='model'`, les ancres hors fenêtre des couples imputés sont effacées
- **Type** : [SPEC] code ≠ spec (I7 : « ne détruit jamais une observation d'entrée »)
- **Composant** : `tsforecast/frequency/covariate_materializer.py::CovariateMaterializer.stage_frame` (voie miroir,
  rangs 2-3, propre à `'model'`), lue par `HighFrequencyImputer._replay_plan`
- **Sévérité** : majeure
- **Observé** : sur `irregular_index_timeseries` (fenêtre d'imputation 2019-01 → 2024-06, faute de
  `production_industrielle` avant 2019), les observations de `pib_trimestriel` (2018-01, 04, 07, 10, 2024-01, 04) et
  de `balance_commerciale_annuelle` (2015-2018) situées hors de la fenêtre sortent **NaN et sans marque**
  sous `'model'`, quelle que soit la modalité de `impute_intermediate_frequencies`. Sous `'interpolate'` et
  `'tolerate_nan'`, les mêmes cellules gardent leur valeur et la marque `ORIGINAL`. Même chose sur
  `heterogeneous_coverage_panel`. Seule l'inversion avec `restore_original_values=True` les rend. Cause : sous
  `'model'`, la frame d'étape sert la colonne imputée depuis le miroir des cellules produites (rang 2-3). Ce miroir
  est vide hors fenêtre et ne retombe pas sur les ancres brutes, alors que la voie `'interpolate'` plafonnée les garde.
- **Attendu** : I7, et cohérence entre stratégies : la stratégie de covariables ne change rien hors de la fenêtre
  d'imputation (§7.2 : les lignes hors fenêtre gardent leurs valeurs d'entrée). Les ancres hors fenêtre restent
  intactes et `ORIGINAL`, comme sous `'interpolate'`.
- **Reproduction** :
  ```python
  data = build_mixed_frequency_timeseries(annual_start_date='2015-01-01')
  for strategy in ('interpolate', 'model'):
      imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                     covariate_strategy=strategy, keep_lower_frequencies=False)
      print(strategy, imputer.fit_transform(data).loc['2018-01-01', 'pib_trimestriel'])
  # interpolate 2500.0 ; model nan   (observé : 2500.0)
  ```
- **Test** : `tests/integration/frequency/test_hfi_on_realistic_dataset.py::TestResidualNaN::test_input_rows_outside_the_window_are_left_as_is`
  (les 18 cellules, `xfail` retiré) ; `tests/unit/frequency/covariate_materializer/test_stage_frame.py::TestStageFrameMirror::test_coarser_production_never_lands_raw_on_a_finer_grid`
  (assertion resserrée sur son intention : aucune production trimestrielle sur la grille mensuelle, l'observation
  annuelle 2021 restant à sa date)
- **Statut** : corrigé (2026-10-15) — demande de l'auteur : traiter l'anomalie de la manière la plus logique
- **Correction** : `CovariateMaterializer._produce_column`, dans la seule vue d'étape (`cap_interpolation=True`,
  chemin de `stage_frame`) : pour une entité servie par le miroir (rangs 2-3), le miroir **recouvre** les ancres
  brutes au lieu de les remplacer ; là où le modèle n'a rien produit, hors fenêtre notamment, l'observation reste.
  C'est la règle que le plafonnement appliquait déjà aux autres voies, et la docstring de `stage_frame` (« les
  valeurs d'entrée, … et le miroir »). `materialize` (`X_train` / `X_pred`) n'est pas touché : modèles, voies et
  I11 inchangés. Les ancres hors fenêtre sortent avec leur valeur et la marque `ORIGINAL`, sous les trois
  stratégies.
### ANO-FREQ-084 — Notebook 5 : la cellule 33 lit `_check_transform_frequencies` comme un dict
- **Type** : [DOC] notebook ≠ code
- **Composant** : `notebooks/5 - QB - HighFrequencyImputer pas a pas.ipynb` (cellule 33), face à
  `tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer._check_transform_frequencies`
- **Sévérité** : mineure
- **Observé** : la cellule appelle la méthode **privée** `REF_T._check_transform_frequencies(PANEL_X.copy())` et
  passe le résultat à `dict(...)`. Depuis `c9ecf79` (D42, correction d'ANO-FREQ-073), la méthode renvoie un tuple
  `(frequencies, misaligned)` : `ValueError: dictionary update sequence element #0 has length 0; 2 is required`.
  C'est la seule cellule en erreur sur les 44 cellules de code.
- **Attendu** : le notebook suit le code (changement délibéré, prouvé par l'historique). Correctif d'une ligne :
  `new_pairs, _misaligned = REF_T._check_transform_frequencies(PANEL_X.copy())`. Mieux : passer par l'API publique,
  puisqu'un notebook utilisateur ne devrait pas dépendre d'une méthode privée.
- **Reproduction** : exécuter le notebook (`jupyter nbconvert --execute`, ou le test ci-dessous).
- **Test** : `tests/integration/frequency/test_notebooks.py::TestStepByStepNotebook::test_notebook_runs_without_error`
- **Statut** : corrigé (2026-10-15) — arbitrage de l'auteur : corriger le notebook
- **Correction** : cellule 33 alignée sur la signature actuelle :
  `new_pairs, misaligned = REF_T._check_transform_frequencies(PANEL_X.copy())`, commentaire mis à jour (D42). Le
  notebook audite volontairement les méthodes privées : il reste exposé à leurs changements de signature, que
  le test d'exécution (`slow`) détecte. Le `xfail` est retiré ; le notebook s'exécute sans erreur.

## BASE

### ANO-BASE-001 — `InverseTransformer` inutilisable après `clone`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/base/transformers.py::InverseTransformer`
- **Sévérité** : majeure
- **Observé** : `fit` ne fait rien et `clone` remplace le transformateur enveloppé par une copie non ajustée :
  `clone(InverseTransformer(ajusté)).fit(X).transform(X)` lève l'erreur du transformateur enveloppé non ajusté
  (`NotFittedError` avec `CenteredScaler`, `AttributeError` pour une sous-classe sans `check_is_fitted`).
  `cross_validate`, `GridSearchCV` et `Pipeline(memory=…)` clonent : l'étape y est inutilisable.
- **Attendu** : un clone ajusté comme toute étape sklearn transforme comme l'original. Pistes : `__sklearn_clone__`
  conservant le transformateur ajusté (estimateur « figé »), ou `fit` qui ajuste un clone du transformateur
  enveloppé (sur des données de l'espace d'origine, à documenter).
- **Reproduction** :
  ```python
  from sklearn.base import clone
  from sklearn.preprocessing import StandardScaler
  from tsforecast.base.transformers import InverseTransformer
  X = [[1.0], [3.0], [5.0]]
  inv = InverseTransformer(StandardScaler().fit(X))
  clone(inv).fit(X).transform(X)   # NotFittedError
  ```
- **Correctif** : décision de l'auteur (2026-10-16) : option la plus standard, celle de
  `sklearn.frozen.FrozenEstimator` (étape pré-ajustée). `__sklearn_clone__` renvoie l'enveloppe elle-même,
  `__sklearn_is_fitted__` suit l'état du transformateur enveloppé, `transform` / `inverse_transform` appellent
  `check_is_fitted`. Un `set_params` sur un clone modifie donc l'original (documenté).
- **Test** : `tests/unit/base/transformers/test_reversible.py::TestInverseTransformer::test_clone_keeps_a_usable_transformer`,
  `::test_clone_is_the_wrapper_itself`, `::test_fitted_state_follows_the_wrapped_transformer`,
  `::test_cross_validate_with_a_pipeline`
- **Statut** : corrigée (2026-10-16)

### ANO-BASE-002 — L'entrée de l'appelant est modifiée (index converti en place)
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/base/transformers.py::PanelTimeSeriesTransformer._validate_index_time_series`
- **Sévérité** : majeure
- **Observé** : `data.index = pd.to_datetime(data.index)` s'applique à l'objet passé à `fit` / `transform` : un index
  de chaînes de dates ressort `datetime64` chez l'appelant ; un `RangeIndex` ressort en dates de 1970 (voir
  ANO-BASE-004). Même effet sur une `Series`. Le dernier niveau d'un `MultiIndex` et les colonnes ne sont pas touchés
  (`validate_temporal_data` travaille sur une copie).
- **Attendu** : aucune méthode ne modifie l'entrée (copie avant conversion, ou conversion confiée à
  `validate_temporal_data`).
- **Reproduction** :
  ```python
  import pandas as pd
  X = pd.DataFrame({'v': [1.0, 2.0]}, index=['2023-01-31', '2023-02-28'])
  MyTransformer().fit(X)      # toute sous-classe concrète de PanelTimeSeriesTransformer
  X.index.dtype               # datetime64[ns]
  ```
- **Correctif** : `_validate_index_time_series` ne convertit plus l'index : la conversion est faite par
  `validate_temporal_data`, sur une copie ; la convertibilité est vérifiée en amont par `_detect_data_structure`.
- **Test** : `tests/unit/base/transformers/test_sklearn_contract.py::TestNonMutation::test_converted_index_is_left_intact`
  (chaînes de dates en `DataFrame` et en `Series`, `PeriodIndex`), `::test_fit_transform_inverse_leave_the_input_intact`
- **Statut** : corrigée (2026-10-16)

### ANO-BASE-003 — `transform` réécrit l'attribut ajusté `conversion_metadata_`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/base/transformers.py::PanelTimeSeriesTransformer._validate_column_based`,
  `::ReversibleTransformerMixin._restore_structure_if_converted`
- **Sévérité** : mineure
- **Observé** : sous `convert_cols_to_index=True`, chaque `transform` remplace `conversion_metadata_` par les
  métadonnées de ses propres données. `inverse_transform` d'une sortie antérieure restaure alors l'index des
  dernières données transformées : `out = t.transform(A)`, `t.transform(B)`, `t.inverse_transform(out)` porte
  l'index de `B`. La convention sklearn veut que `transform` ne modifie pas l'état ajusté.
- **Attendu** : `inverse_transform(transform(X))` restitue `X` quels que soient les appels intermédiaires.
- **Reproduction** :
  ```python
  first = panel_en_colonnes                    # index 0..7
  second = first.set_axis(range(10, 18))
  t = MyReversible(time_col='date', panel_cols=['country'], convert_cols_to_index=True)
  out = t.fit(first).transform(first); t.transform(second)
  t.inverse_transform(out).index               # 10..17
  ```
- **Correctif** : décision de l'auteur (2026-10-16). `conversion_metadata_` est figé au `fit` et ne garde que la
  structure (colonnes déplacées, positions, noms : `original_index` à `None`) ; `transform` ne le modifie plus, et un
  nouveau `fit` sans conversion le retire. Les colonnes rendues reçoivent un `RangeIndex` neuf (même règle qu'après
  un tri, ANO-UTILS-024). Les métadonnées de l'appel en cours (index d'origine compris) sont gardées dans l'attribut
  privé `_current_conversion_metadata`, que `HighFrequencyImputer._align_target_index` lit pour réaligner `y` sur `X`.
- **Test** : `tests/unit/base/transformers/test_reversible.py::TestRestoreStructure::test_inverse_does_not_depend_on_later_transforms`,
  `::test_transform_leaves_the_fitted_structure_untouched`, `::test_restored_columns_get_a_fresh_range_index`,
  `::test_refit_without_conversion_drops_the_structure`
- **Statut** : corrigée (2026-10-16)

### ANO-BASE-004 — Index numérique accepté par `fit` en mode strict, lu en nanosecondes depuis 1970
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/base/transformers.py::PanelTimeSeriesTransformer._detect_data_structure` et
  `::_validate_index_time_series`
- **Sévérité** : majeure
- **Rattachement** : ANO-UTILS-028 (décision de l'auteur du 2026-09-29 : rejeter un index numérique).
  `validate_temporal_data` et `_validate_time_index` appliquent la décision (`_convert_to_datetime`), mais la
  détection de structure et la validation d'une série en index appellent `pd.to_datetime` nu.
- **Observé** : `RangeIndex`, années entières (`2020, 2021`) ou flottants passent `fit` en mode strict, sans erreur
  ni avertissement ; `_fit` reçoit `1970-01-01 00:00:00.000002020`, … (et l'entrée de l'appelant est modifiée,
  ANO-BASE-002). Le dernier niveau numérique d'un `MultiIndex` est, lui, bien rejeté.
- **Attendu** : même règle que `validate_temporal_data` : rejet en mode strict.
- **Reproduction** :
  ```python
  import pandas as pd
  t = MyTransformer().fit(pd.DataFrame({'v': [1.0, 2.0]}, index=[2020, 2021]))   # aucune erreur
  ```
- **Correctif** : `_detect_data_structure` applique `_convert_to_datetime` (règles de la validation) à un index
  simple ; un index numérique y est refusé dans les deux modes, comme un index de chaînes qui ne sont pas des dates :
  « When time_col and panel_cols are None, the index must hold dates (datetimes, date strings or periods) for a
  time series, or be a MultiIndex for panel data. »
- **Test** : `tests/unit/base/transformers/test_time_index.py::TestRejectedTimeLabels::test_numeric_index_is_rejected`
  (`range`, `integer-years`, `floats` × strict / permissif)
- **Statut** : corrigée (2026-10-16)

### ANO-BASE-005 — `PeriodIndex` refusé par `fit` ; périodes d'un panel jamais restituées
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/base/transformers.py::PanelTimeSeriesTransformer._detect_data_structure`,
  `::_validate_index_time_series`, `::ReversibleTransformerMixin._restore_structure_if_converted`
- **Sévérité** : mineure
- **Rattachement** : ANO-UTILS-029 (décision de l'auteur du 2026-09-29 : accepter `PeriodIndex`, lu au premier
  instant, et le rendre par `restore_original_structure`). Ici ce n'est pas `validate_temporal_data` qui refuse,
  mais la détection de structure de `base`, qui ne l'atteint pas.
- **Observé** : un `PeriodIndex` simple lève `ValueError: When time_col and panel_cols are None, index must be
  DatetimeIndex (for time series) or MultiIndex (for panel data)` (`pd.to_datetime` nu refuse les périodes). Un
  dernier niveau `Period` de `MultiIndex` est accepté et converti en `Timestamp`, mais ni `transform` ni
  `inverse_transform` ne rendent les périodes : l'aller-retour change le type de l'index.
- **Attendu** : `PeriodIndex` accepté comme par `validate_temporal_data` ; aller-retour à l'identité.
- **Reproduction** :
  ```python
  import pandas as pd
  X = pd.DataFrame({'v': [1.0, 2.0]}, index=pd.period_range('2023-01', periods=2, freq='M'))
  MyTransformer().fit(X)   # ValueError: When time_col and panel_cols are None, ...
  ```
- **Test** : `tests/unit/base/transformers/test_time_index.py::TestTimeIndexThroughFit::test_period_index_is_read_at_first_instants`,
  `tests/unit/base/transformers/test_reversible.py::TestRoundTripOnPerturbations::test_series[to_period_index]`,
  `::test_panel[to_period_index]`, `::TestRestoreStructure::test_periods_left_unconverted_are_kept`
- **Correctif** : un `PeriodIndex` simple est accepté (même règle que l'index numérique, ANO-BASE-004) ; `fit`
  note la fréquence des périodes de l'index (ou du dernier niveau) et `_restore_structure_if_converted` rend les
  périodes (`_restore_periods`). La sortie de `transform` porte le premier instant de chaque période, comme la donnée
  validée transmise à `_transform`.
- **Statut** : corrigée (2026-10-16)

### ANO-BASE-006 — `fit` sur une `pandas.Series` : `IndexError`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/base/transformers.py::PanelTimeSeriesTransformer.fit`
- **Sévérité** : mineure
- **Observé** : `_validate_input` route explicitement les `Series` (commit `f22bbb0` « (add) Series validation »),
  puis `fit` lève `IndexError: tuple index out of range` sur `self.n_features_in_ = X.shape[1]`, avec ou sans
  validation.
- **Attendu** : `Series` traitée comme une seule variable (`n_features_in_ == 1`), ou refus explicite et documenté.
- **Reproduction** :
  ```python
  import pandas as pd
  s = pd.Series([1.0, 2.0], index=pd.date_range('2023-01-31', periods=2, freq='ME'), name='v')
  MyTransformer().fit(s)   # IndexError: tuple index out of range
  ```
- **Correctif** : décision de l'auteur (2026-10-16) : convention sklearn, `X` est à deux dimensions. `fit` et
  `transform` (base et `XYPanelTimeSeriesTransformer`) refusent tout ce qui n'est pas un `DataFrame`, validation
  activée ou non : « X must be a pandas DataFrame, got Series; pass series.to_frame() for a single series » (même
  libellé que le refus de `HighFrequencyImputer`). Le routage des `Series` de `_validate_input` est supprimé.
- **Test** : `tests/unit/base/transformers/test_sklearn_contract.py::TestInputKinds::test_series_is_refused_at_fit`,
  `::test_series_is_refused_at_transform`, `::test_single_column_frame_of_a_series`,
  `tests/unit/base/transformers/test_structure.py::TestInvalidStructures::test_non_pandas_input`
- **Statut** : corrigée (2026-10-16)

### ANO-BASE-007 — `inverse_transform` abstrait de `ReversibleTransformerMixin` non imposé
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/base/transformers.py::ReversibleTransformerMixin`
- **Sévérité** : mineure
- **Observé** : le mixin n'a pas `ABCMeta` pour métaclasse ; `ABCMeta` de `PanelTimeSeriesTransformer` ne lit que
  les `__abstractmethods__` des bases, que le mixin n'a pas. Une sous-classe sans `inverse_transform`
  s'instancie, et `inverse_transform` renvoie `None` (corps `pass` du mixin), sans erreur.
- **Attendu** : `TypeError` à l'instanciation, comme pour `_fit` / `_transform` (le mixin dérive de `ABC`).
- **Reproduction** :
  ```python
  class NoInverse(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
      def _fit(self, X, y=None): pass
      def _transform(self, X): return X
  NoInverse().inverse_transform(X)   # None
  ```
- **Correctif** : `ReversibleTransformerMixin` dérive de `ABC`.
- **Test** : `tests/unit/base/transformers/test_reversible.py::TestReversibleContract::test_missing_inverse_transform_is_refused`
- **Statut** : corrigée (2026-10-16)

### ANO-BASE-008 — `InverseTransformer` non exporté par `tsforecast.base`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/base/__init__.py`
- **Sévérité** : mineure
- **Observé** : classe publique (sans préfixe `_`, même règle que pour `panel/utils.py`) absente de
  `tsforecast.base.__all__` et des imports du paquet.
- **Attendu** : exportée comme les trois autres classes du module.
- **Reproduction** : `'InverseTransformer' in tsforecast.base.__all__   # False`
- **Correctif** : ajouté aux imports et à `__all__` de `tsforecast/base/__init__.py`.
- **Test** : `tests/unit/base/transformers/test_reversible.py::TestInverseTransformer::test_exported_by_the_package`
- **Statut** : corrigée (2026-10-16)

### ANO-BASE-009 — `transform` ne contrôle pas les colonnes vues au `fit`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/base/transformers.py::PanelTimeSeriesTransformer.transform`
- **Sévérité** : à arbitrer
- **Observé** : `n_features_in_` et `feature_names_in_` sont enregistrés au `fit` mais jamais comparés à l'entrée de
  `transform` : une colonne en plus ou en moins est transmise telle quelle à `_transform` (l'erreur éventuelle
  vient de la sous-classe, ou n'existe pas).
- **Attendu** : à trancher. sklearn contrôle ces attributs dans `transform` (`validate_data(reset=False)`) ; les
  sous-classes du paquet (imputeur, transformateur par entité) acceptent peut-être des colonnes différentes au
  `transform`. Comportement actuel épinglé.
- **Correctif** : décision de l'auteur (2026-10-16) : contrôle au `transform`, comme sklearn
  (`_check_feature_names`, appelé par `PanelTimeSeriesTransformer.transform` et
  `XYPanelTimeSeriesTransformer.transform`, après validation) : mêmes noms, même ordre ; le message nomme les
  colonnes nouvelles, absentes, ou signale un simple réordonnancement. `HighFrequencyImputer` surcharge
  `_check_feature_names` (sans effet) : sa spec fixe son propre contrat (colonne manquante refusée en la nommant,
  colonne supplémentaire ignorée silencieusement).
- **Test** : `tests/unit/base/transformers/test_sklearn_contract.py::TestFeatureNamesAtTransform` (6 tests) ;
  contrat de l'imputeur : `tests/unit/frequency/high_frequency_imputer/test_transform.py::TestTransformFrequencyControl`
- **Statut** : corrigée (2026-10-16)

### ANO-BASE-010 — `convert_cols_to_index=False` fait quand même passer les colonnes dans l'index
- **Type** : [DOC] docstring ≠ code / à arbitrer
- **Composant** : `tsforecast/base/transformers.py::PanelTimeSeriesTransformer` (docstrings de classe et
  d'`__init__`), `::_validate_column_based`
- **Sévérité** : à arbitrer
- **Observé** : la docstring décrit `convert_cols_to_index` : « If True, converts time_col and panel_cols to index
  and stores metadata for restoration ». Or la validation par colonnes appelle toujours `validate_temporal_data`,
  qui indexe par `panel_cols + [time_col]` : avec `False`, la sortie est indexée de même, sans métadonnées, avec
  l'avertissement « Index replaced with [...] » à chaque `fit` et `transform`, et `inverse_transform` ne peut pas
  rendre les colonnes. Le paramètre ne décide en fait que de la conservation des métadonnées.
- **Attendu** : à trancher : documenter ce comportement, ou ne pas déplacer les colonnes quand le paramètre vaut `False`.
- **Test** : `tests/unit/base/transformers/test_structure.py::TestConvertColsToIndex::test_without_conversion_the_columns_still_move_into_the_index`
  (suit le code)
- **Décision de l'auteur (2026-10-16)** : avec `convert_cols_to_index=False`, `time_col` et `panel_cols` restent des
  colonnes (ce n'est donc plus le cas par défaut). Correction **reportée à B5**, avec ANO-PANEL-001 : aujourd'hui
  `PanelwiseTransformer` ne traite un panel en colonnes que parce que la validation le convertit en `MultiIndex` ;
  corrigée seule, cette anomalie le ferait passer par son chemin colonnes, cassé (trois tests de
  `tests/unit/panel/test_transformers.py` : `test_fit_transform_with_validation`, `test_multiindex_after_validation`,
  `test_scaling_with_multiindex`). Effets attendus : la vérification du tri d'une série en colonnes porte sur la
  colonne `time_col` ; `n_features_in_` compte les colonnes de temps et d'entité ; l'avertissement « Index replaced »
  disparaît ; le test ci-dessus devient un test du comportement voulu.
- **Statut** : ouvert (reporté à B5)

### ANO-BASE-011 — Identifiant d'entité manquant : simple avertissement, même en mode strict
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/base/transformers.py::PanelTimeSeriesTransformer.fit` (`_check_panel_consistency`)
- **Sévérité** : à arbitrer
- **Observé** : une valeur manquante dans un niveau d'entité ou une colonne `panel_cols` produit
  « Panel structure issues: Missing values in panel identifier… » en avertissement, que `strict_validation` vaille
  `True` ou `False`. Le test d'origine passait `strict_validation=False` avec le commentaire « Use warnings instead of
  errors », ce qui suppose une erreur en mode strict.
- **Attendu** : à trancher : erreur en mode strict (cohérent avec les autres contrôles), ou avertissement documenté.
  Comportement actuel épinglé.
- **Correctif** : décision de l'auteur (2026-10-16) : « Panel structure issues: … » lève une `ValueError` quand
  `validate_input` et `strict_validation` valent `True`, avertit sinon (validation désactivée : simple signalement).
- **Test** : `tests/unit/base/transformers/test_structure.py::TestPanelIdentifierConsistency::test_missing_entity_level_value_raises_in_strict_mode`,
  `::test_missing_entity_level_value_warns_without_validation`, `::test_missing_entity_level_value_warns_when_not_strict`
- **Statut** : corrigée (2026-10-16)

### ANO-BASE-012 — `time_col` absente : message « Cannot determine time index » qui ne la nomme pas
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/base/transformers.py::PanelTimeSeriesTransformer._validate_column_based`
  (`_validate_time_index`)
- **Sévérité** : cosmétique
- **Observé** : sur un panel en colonnes dont `time_col` n'existe pas, `fit` lève « Time validation failed: Cannot
  determine time index. Please provide a datetime index or specify time_col parameter. », alors que `time_col` est
  précisé. Si l'index porte des dates, le message devient « Cannot specify panel_cols without time_col » (strict) ou,
  en mode permissif, un avertissement suivi de « For panel data, both panel_cols and time_col must be specified » :
  aucun ne nomme la colonne manquante. C'est l'une des sources du message « Cannot determine time index » vu en
  partie D. Sans `panel_cols`, une `time_col` absente laisse l'index porter les dates (comportement épinglé).
- **Attendu** : « Time column 'period' not found in data » (message de `validate_temporal_data`).
- **Reproduction** : `MyTransformer(time_col='period', panel_cols=['country']).fit(panel_en_colonnes)`
- **Correctif** : sur un panel en colonnes, `_validate_column_based` lève « Time column 'period' not found in
  data » dans les deux modes, que l'index porte des dates ou non. Sans `panel_cols`, le repli sur l'index est
  conservé.
- **Test** : `tests/unit/base/transformers/test_time_index.py::TestRejectedTimeLabels::test_absent_time_col_of_a_panel_is_named`
  (index par défaut ou daté × strict / permissif)
- **Statut** : corrigée (2026-10-16)

### ANO-BASE-013 — Docstrings de `base/transformers.py` : défaut, exemples, noms d'arguments
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/base/transformers.py`
- **Sévérité** : cosmétique
- **Observé** : la docstring de classe de `PanelTimeSeriesTransformer` annonce `time_col` « Default is 'date' »
  (`None` dans `__init__`) ; les exemples de `fit`, `transform`, `_sorting_advice`, `_store_transformation_info` et
  `_restore_structure_if_converted` utilisent `MyTransformer`, `ConcreteTransformer` et `MyReversibleTransformer`,
  jamais définis, et instancient `PanelTimeSeriesTransformer`, abstraite (les doctests échouent ; recoupe N7) ;
  `_validate_input` et `_validate_multiindex_panel` documentent un argument `X` qui s'appelle `data` ;
  l'exemple de `fit` annonce `n_features_in_ == 3` pour un frame `date, value, feature` validé sur `time_col='date'`
  (`2` observé, la date passe dans l'index).
- **Attendu** : docstrings alignées sur le code ; exemples exécutables.
- **Correctif** : docstrings réécrites (défaut `None`, attributs ajustés, `Series`) ; chaque exemple définit sa
  sous-classe et passe en doctest (`pytest --doctest-modules tsforecast/base --noconftest` : 26 passed, aucun
  `+SKIP`).
- **Test** : doctests du module
- **Statut** : corrigée (2026-10-16)

### ANO-BASE-014 — `docs/concepts/base_transformers.md` décrit une hiérarchie et des méthodes inexistantes
- **Type** : [DOC] documentation ≠ code
- **Composant** : `docs/concepts/base_transformers.md`
- **Sévérité** : mineure
- **Observé** : (1) l'arbre « Rapport avec les autres modules » fait hériter `XYTransformerMixin` et
  `PublicationDelayTransformer` de `PanelTimeSeriesTransformer` : le premier dérive de `TransformerMixin`, le second
  de `BaseEstimator, TransformerMixin` ; seuls `PanelwiseTransformer` et `XYPanelTimeSeriesTransformer` en
  héritent. (2) `ReversibleTransformerMixin` « impose … son articulation avec `_inverse_transform` » : cette méthode
  n'existe pas. (3) `PanelTimeSeriesTransformer` « prend en charge … la restauration de la structure d'origine en
  sortie » : `transform` ne restaure rien, seule l'aide `_restore_structure_if_converted` du mixin le fait, à
  l'appel de la sous-classe. (4) `InverseTransformer` et `convert_cols_to_index` ne sont pas mentionnés.
- **Attendu** : page alignée sur le code.
- **Correctif** : page réécrite (quatre briques, hiérarchie réelle, contrôles du `transform`, `InverseTransformer`).
- **Test** : aucun
- **Statut** : corrigée (2026-10-16)

### ANO-BASE-015 — Branches inatteignables (code mort)
- **Type** : [CODE] comportement (nettoyage, pas un bogue)
- **Composant** : `tsforecast/base/transformers.py::PanelTimeSeriesTransformer`
- **Sévérité** : cosmétique
- **Observé** : décelé par la couverture : `raise ValueError("Unexpected data structure configuration")` (les
  quatre combinaisons `time_col` / `panel_cols` sont traitées avant) ; `raise ValueError("Invalid data structure
  configuration")` de `_validate_input` (un panel en index a toujours le temps en index) ; les branches d'avertissement
  du mode permissif après `validate_temporal_data` dans `_validate_multiindex_panel`, `_validate_index_time_series`
  et `_validate_column_based` (en mode permissif la fonction avertit sans lever ; pour les colonnes, les causes
  restantes sont refusées en amont) ; le test `isinstance(X_validated.index, pd.DatetimeIndex)` d'une série en
  colonnes (toujours vrai) ; les contrôles `MultiIndex` de `_validate_multiindex_panel` et la conversion de
  `_validate_index_time_series`, déjà garantis par `_detect_data_structure`.
- **Correctif** : branches supprimées, sans changement de comportement. Les entrées `Series` passent désormais par
  `_detect_data_structure`, comme les `DataFrame`. `base/transformers.py` : 100 % (lignes et branches).
- **Test** : couverture des tests de `tests/unit/base/transformers/`
- **Statut** : corrigée (2026-10-16)

## CV

### ANO-CV-001 — Données non triées : plis renvoyés avec la permutation inverse (fuite)
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::OutOfSampleSplit._split`, `::InSampleSplit._split`
  (et `::_verify_and_sort_data`, dont la docstring décrit l'inverse de ce qu'elle renvoie)
- **Sévérité** : bloquante
- **Observé** : `_verify_and_sort_data` renvoie `sort_indices` tel que `sort_indices[k]` est la position d'origine de
  la k-ième ligne triée (sa docstring dit le contraire : « mapping original positions to sorted positions »). `_split`
  inverse cette permutation (`np.argsort(sort_mapping)`) avant de l'appliquer : les plis désignent d'autres lignes
  que prévu. Sur les dates `[3, 1, 5, 2, 4]` (`n_splits=2, test_size=1`), le premier pli teste la date 1 et apprend
  sur les dates 3, 4 et 5. Une permutation qui est sa propre inverse (ordre renversé) donne par hasard le bon
  résultat. Seul l'avertissement « not sorted by date. Sorting automatically. » était testé. Touche toutes les
  variantes (série, panel avec `groups`, `TS*`, `Panel*`), en hors échantillon comme en échantillon.
- **Attendu** : les plis d'une série désordonnée désignent les mêmes dates que ceux de la série triée (remap par
  `sort_mapping[indices]`) ; aucune fuite.
- **Reproduction** :
  ```python
  import pandas as pd
  from tsforecast.crossvals import OutOfSampleSplit
  d = pd.date_range('2024-01-31', periods=5, freq='ME')
  X = pd.Series(range(5), index=d[[2, 0, 4, 1, 3]])
  train, test = next(OutOfSampleSplit(n_splits=2, test_size=1).split(X))
  X.index[train].max() > X.index[test].min()   # True : apprentissage postérieur au test
  ```
- **Correctif** : `_split` ramène les plis par `sort_mapping[indices]` ; la docstring de `_verify_and_sort_data` décrit `sort_indices[k]` comme la position d'origine de la k-ième ligne triée. Le tri d'un panel est un tri stable sur tous les niveaux.
- **Test** : `tests/unit/crossvals/base/test_index_robustness.py::TestUnsortedData::test_folds_select_the_same_dates_as_on_sorted_data`, `::test_no_leakage`, `::test_no_leakage_on_an_interleaved_panel`, `::test_sort_is_announced`, `::test_reversed_order`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-002 — `get_n_splits` renvoie `n_splits`, pas le nombre de plis produits
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::OutOfSampleSplit`, `::InSampleSplit` (`get_n_splits` hérité de
  `sklearn.model_selection._split._BaseKFold`)
- **Sévérité** : majeure
- **Observé** : `get_n_splits` renvoie toujours `n_splits`. Or avec `test_indices`, le nombre de plis est
  `len(test_indices)` ; un découpage en échantillon produit toujours un seul pli. `GridSearchCV` refuse alors le
  splitter : « cv.split and cv.get_n_splits returned inconsistent results. Expected 5 splits, got 2 » (ou « got 1 »
  pour `TSInSampleSplit`). Pour une configuration que `split` refuse (trop de plis, jeu vide, `test_size` trop
  grand), `get_n_splits` annonce quand même `n_splits` plis.
- **Attendu** : `get_n_splits(X, y, groups) == len(list(split(X, y, groups)))` ; quand `split` refuse les données,
  `get_n_splits` les refuse aussi (ce second point est à confirmer : les splitters sklearn ne regardent pas les
  données dans `get_n_splits`).
- **Reproduction** :
  ```python
  from sklearn.linear_model import Ridge
  from sklearn.model_selection import GridSearchCV
  cv = TSOutOfSampleSplit(test_indices=['2024-06-30', '2024-09-30'], test_size=3)
  GridSearchCV(Ridge(), {'alpha': [0.1, 1.0]}, cv=cv).fit(X, y)   # ValueError: ... Expected 5 splits, got 2
  ```
- **Correctif** : décision de l'auteur (2026-10-16) : convention sklearn. `get_n_splits` se calcule sur les paramètres seuls, sans lire les données : nombre de `test_indices` s'il y en a, `n_splits` sinon ; `InSampleSplit.get_n_splits` renvoie 1. Pour toute donnée acceptée par `split`, le compte est exact ; pour une donnée refusée, `get_n_splits` ne lève pas (comme `TimeSeriesSplit`). `PanelOutOfSampleSplitPerEntity` / `PanelInSampleSplitPerEntity` héritent de ce compte, qui ne multiplie pas par le nombre d'entités : reste à traiter en B3.
- **Test** : `tests/unit/crossvals/base/test_properties.py::TestNSplitsConsistency` (`test_other_configurations`, `test_panel_configurations`, `test_count_does_not_read_the_data`), `tests/unit/crossvals/test_time_series.py::TestSklearnModelSelection::test_grid_search_with_other_folds`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-003 — Plis à apprentissage vide produits sans erreur
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::OutOfSampleSplit._get_timeseries_train_indices`,
  `::_get_group_train_indices`
- **Sévérité** : à arbitrer
- **Observé** : quand `gap` dépasse l'historique disponible avant un pli (ou quand la première date est un indice de
  test explicite), le pli est produit avec un ensemble d'apprentissage vide, sans erreur ni avertissement
  (`n=12, n_splits=3, gap=7` : apprentissages `[]`, `[]`, `[0, 1]`). Un estimateur ne peut rien en faire. Le contrôle
  « Too many splits » ne tient pas compte de `gap` (commentaire du code : « On peut ajouter -self.gap si on veut
  s'assurer des données d'entraînement »).
- **Attendu** : à trancher : refuser la configuration (`ValueError` qui compte `gap`), retirer les plis sans
  apprentissage (et les décompter dans `get_n_splits`, ANO-CV-002), ou garder et documenter.
- **Reproduction** : `list(OutOfSampleSplit(n_splits=3, gap=7).split(np.zeros(12)))[0][0]   # array([])`
- **Correctif** : décision de l'auteur (2026-10-16) : configuration refusée. Un pli sans donnée d'apprentissage lève « No training data before the test set starting at row 3 (gap=7); reduce gap or the number of splits » (série) ou « No training data for the group 'A' before its test set (gap=5); … » (panel), à la production du pli.
- **Test** : `tests/unit/crossvals/base/test_out_of_sample_split.py::TestDefaultFolds::test_gap_longer_than_the_history_is_refused`, `::TestExplicitTestIndices::test_first_position_is_refused`, `::test_gap_reaching_the_first_date_is_refused`, `::TestPanelGroups::test_gap_longer_than_an_entity_history_is_refused`, `tests/unit/crossvals/test_time_series.py::TestTSOutOfSampleSplit::test_gap_leaving_no_training_data`, `::TestSklearnModelSelection::test_fold_without_training_data_is_reported`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-004 — Entité lue sur le seul premier niveau, date sur le seul deuxième
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py` (`_resolve_test_positions`, `_check_groups`, `_sorted_time_points`,
  `_rows_at`), `tsforecast/crossvals/panel.py::_extract_groups_from_panel`
- **Sévérité** : majeure
- **Observé** : la convention du paquet (`panel/utils.py::is_panel_data`, `get_entity_levels`,
  `utils/validation::validate_entities_grouped`) définit l'entité par tous les niveaux du `MultiIndex` sauf le
  dernier, et la date par le dernier. Les splitters lisaient l'entité au niveau 0 et la date au niveau 1 : sur un
  panel (pays, secteur, date), les secteurs d'un même pays étaient fusionnés en une entité et les dates cherchées
  dans le niveau des secteurs. `_extract_groups_from_panel` acceptait en outre un index simple (chaque ligne une
  entité).
- **Attendu** : entité = niveaux d'entité (valeur seule, ou tuple quand il y en a plusieurs), date = dernier
  niveau ; un index simple n'est pas un panel.
- **Correctif** : décision de l'auteur (2026-10-16). Détection du panel par `is_panel_data` ; entité de chaque ligne
  par `_entity_labels` (`index.droplevel(-1)`), dates par `_time_level` (dernier niveau) ; `groups` confronté à ces
  entités ; tables d'entités par `pd.factorize` (entités tuples) ; un tuple de test s'écrit `(pays, secteur, date)`
  ou `(('pays', 'secteur'), date)`, avec une valeur par niveau d'entité (sinon `ValueError`).
  `_extract_groups_from_panel` refuse un index simple (« Panel data requires MultiIndex (entity, time) ») et renvoie
  les entités ; les deux classes `PerEntity` itèrent sur la même table.
- **Test** : `tests/unit/crossvals/base/test_index_robustness.py::TestEntityOnSeveralLevels` (9 tests),
  `tests/unit/crossvals/panel/test_index_robustness.py::TestSeveralEntityLevels` (dont les 2 tests
  `test_groups_are_deduced_from_every_entity_level`, `test_per_entity_folds`),
  `tests/unit/crossvals/panel/test_extract_groups.py::TestRefusals::test_flat_index_is_refused` (réécrit : il épinglait
  l'acceptation d'un index simple), `::TestEntityOfEachRow::test_several_entity_levels_give_tuples`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-005 — Docstrings de `crossvals/panel.py` : dictionnaire annoncé, structure d'index et exemples inexacts
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/crossvals/panel.py::PanelOutOfSampleSplit`, `::PanelInSampleSplit`,
  `::PanelOutOfSampleSplitPerEntity`, `::PanelInSampleSplitPerEntity` ; `docs/concepts/cross_validation.md`
- **Sévérité** : mineure
- **Observé** :
  1. `PanelOutOfSampleSplitPerEntity` : « returning a dictionary mapping each entity to its (train_indices,
     test_indices) » ; `PanelInSampleSplitPerEntity` : « Returns a dictionary mapping each entity ». Les deux
     produisent des couples `(train, test)`, un par entité testée et par pli global (la docstring de `split` le dit,
     pas celle de la classe).
  2. Les deux classes `PerEntity` annoncent « Handling entities with different data availability » : faux, une entité
     qui ne porte pas une date des plis globaux fait échouer le découpage (ANO-CV-006).
  3. `PanelOutOfSampleSplit` / `PanelInSampleSplit` : « First level: entity identifiers / Second level: time
     periods ». L'entité est formée de tous les niveaux sauf le dernier, qui porte la date (index à 3 niveaux
     acceptés, entité tuple).
  4. Ce que fait une variante `PerEntity` n'est dit nulle part : elle découpe les plis de la variante globale entité
     par entité (mêmes dates de test pour toutes les entités, ordre « pli puis entité », entités dans l'ordre de leur
     première ligne) ; `get_n_splits` ne compte pas ces plis (ANO-CV-023). La page de concept dit « un découpage propre
     à chaque entité », ce qui suggère un découpage sur les dates de chaque entité.
  5. Exemples : `np.random.randn` sans graine, aucune sortie affichée (les plis ne sont jamais montrés) ; les formes de
     `test_indices` (positions, tuples qui limitent le pli à une entité) et le mode « plusieurs dates » d'`InSample`
     (`test_size` non utilisé) ne sont pas décrits ; `split` ne documente que l'erreur « MultiIndex » parmi celles de
     la base. Le module n'a pas de docstring d'en-tête.
- **Attendu** : docstrings alignées sur le code, exemples exécutables qui affichent des plis (recoupe N7), page de
  concept qui décrit précisément les deux variantes.
- **Correctif** : `crossvals/panel.py` reçoit une docstring d'en-tête ; docstrings des quatre classes réécrites (entité
  = tous les niveaux sauf le dernier, dates communes et entité en apprentissage seul pour les variantes globales,
  découpage sur les dates de chaque entité pour `PerEntity`, ordre des plis, `get_n_splits`), exemples exécutables qui
  affichent les plis ; docstrings de `OutOfSampleSplit` / `InSampleSplit` complétées ; page
  `docs/concepts/cross_validation.md` alignée (comportement après ANO-CV-006 et ANO-CV-023).
- **Test** : doctests de `tsforecast/crossvals` (`pytest --doctest-modules tsforecast/crossvals`),
  `tests/unit/crossvals/panel/test_contract.py::TestGoldenFolds`,
  `::test_per_entity_order_follows_the_first_appearance_of_entities`,
  `tests/unit/crossvals/panel/test_index_robustness.py::TestSeveralEntityLevels`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-006 — Variantes `PerEntity` : une entité sans une date des plis globaux fait échouer le découpage
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/panel.py::PanelOutOfSampleSplitPerEntity.split`,
  `::PanelInSampleSplitPerEntity.split` (qui délèguent à `tsforecast/crossvals/base.py::OutOfSampleSplit._get_group_train_indices`,
  `::InSampleSplit._get_group_train_indices`)
- **Sévérité** : majeure
- **Observé** : une variante `PerEntity` découpe les plis de sa variante globale ; les dates de test sont donc celles
  du panel entier (dernières dates de l'union, ou dates demandées), communes à toutes les entités. Une entité qui ne
  porte pas l'une d'elles est signalée (« Cannot find test period '2024-11-30 00:00:00' for entity 'B' ») puis le
  découpage lève « Cannot find test indices for the group : B ». Tout panel dont les entités ne finissent pas à la même
  date est donc refusé, y compris le panel réaliste (`heterogeneous_coverage_panel` : Allemagne finit en avril 2024),
  alors que les docstrings annoncent « Handling entities with different data availability » et la page de concept
  « un découpage propre à chaque entité ». Les variantes globales ont le même comportement, cohérent avec leur contrat
  (une même fenêtre de test pour toutes les entités) : il est épinglé, mais le message final ne nomme pas la date.
- **Attendu** (à arbitrer, proposition encodée par les tests `xfail`) : une variante `PerEntity` découpe chaque entité
  sur ses propres dates (plis par défaut : les dernières dates de l'entité) ; une date demandée qu'une entité ne porte
  pas l'exclut de ce pli, sans erreur. Variante possible : garder les dates communes et seulement exclure les entités
  qui ne les portent pas (les deux options coïncident sur un panel équilibré et sur le chemin des dates).
- **Reproduction** :
  ```python
  from tests.support.datasets import build_staggered_panel
  from tsforecast.crossvals import PanelOutOfSampleSplitPerEntity
  X = build_staggered_panel()   # A janvier-décembre, B mars-octobre, C juillet-décembre
  list(PanelOutOfSampleSplitPerEntity(n_splits=2).split(X))
  # UserWarning: Cannot find test period '2024-11-30 00:00:00' for entity 'B'
  # ValueError: Cannot find test indices for the group : B
  ```
- **Correctif** : décisions de l'auteur (2026-10-16).
  1. `PerEntity` découpe chaque entité sur ses propres dates : plis par défaut sur ses dernières dates ; une entité trop
     courte pour les plis par défaut est signalée (« Entity 'C' is too short for the default folds and gets none: … »)
     et ne reçoit pas de pli ; une date demandée que l'entité ne porte pas, ou qui ne lui laisse aucune date
     d'apprentissage, ne lui ouvre pas de pli (avertissement). En échantillon, une entité qui ne porte qu'une des dates
     demandées est testée sur cette seule ligne. Positions et couples (entité, date) gardent les plis globaux, déjà
     limités à une entité. Plis rangés découpage par découpage, entités dans l'ordre de leur première ligne.
  2. Les variantes globales gardent des dates de test communes ; une entité qui ne porte pas les dates de test d'un pli
     n'y est pas testée (avertissement « …; the entity is not tested in this fold ») et apprend sur ses dates
     antérieures à la première date de test, moins `gap`, dans la fenêtre (jusqu'à la dernière date de test en
     échantillon). Aucune date d'apprentissage d'aucune entité n'atteint la période de test. L'erreur « Cannot find test
     indices for the group » disparaît ; une entité testée sans date d'apprentissage reste refusée (ANO-CV-003).
- **Test** : `tests/unit/crossvals/panel/test_contract.py::TestGlobalFoldsWithDifferentCoverage` (7 tests),
  `::TestPerEntityFoldsWithDifferentCoverage` (10 tests),
  `tests/unit/crossvals/panel/test_panel_split.py::TestRealisticPanel::test_default_folds_when_an_entity_ends_earlier`,
  `::TestDegenerateEntities::test_single_observation_entity_in_the_default_folds_is_refused`,
  `tests/unit/crossvals/base/test_out_of_sample_split.py::TestPanelGroups::test_default_date_missing_for_one_entity`,
  `::test_date_missing_for_one_entity`, `tests/unit/crossvals/base/test_in_sample_split.py::TestPanelGroups::test_entity_without_the_test_date`
  (2 cas)
- **Statut** : corrigée (2026-10-16)

### ANO-CV-007 — `test_size` ignoré quand `InSampleSplit` reçoit plusieurs `test_indices`
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/crossvals/base.py::InSampleSplit` (docstrings de classe et d'`__init__`),
  `tsforecast/crossvals/time_series.py::TSInSampleSplit`
- **Sévérité** : mineure
- **Observé** : avec plusieurs indices de test, le pli unique teste une ligne par indice et `test_size` n'est pas lu.
  Changement délibéré (commit `1b27e04` « (fix) insample iter_test_indices behavior with multiple test indices »),
  décrit seulement dans la docstring de la méthode privée `_iter_test_indices`. La docstring publique dit « test_size:
  Size of each test set. If None, uses 1 for specific test_indices ».
- **Attendu** : docstrings publiques qui décrivent les deux modes (un début + `test_size`, ou plusieurs points).
- **Correctif** : docstrings de `InSampleSplit` (classe, `n_splits`) et de `TSInSampleSplit` réécrites : un début et `test_size`, ou plusieurs lignes testées une à une (`test_size` non utilisé).
- **Test** : `tests/unit/crossvals/base/test_in_sample_split.py::TestExplicitTestIndices::test_several_dates_form_one_test_set`, doctests du module
- **Statut** : corrigée (2026-10-16)

### ANO-CV-008 — Un apprentissage vide est un tableau `float64`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::OutOfSampleSplit._get_timeseries_train_indices`
  (`np.array([])`), `::_get_group_train_indices` (`np.array(train_indices)` sur une liste vide)
- **Sévérité** : mineure
- **Observé** : le tableau d'apprentissage vide a le type `float64` : `np.arange(12)[train]` lève « arrays used as
  indices must be of integer (or boolean) type », et `cross_validate` échoue avec un message sans rapport : « No
  valid specification of the columns. Only a scalar, list or slice of all integers or all strings, or boolean mask
  is allowed ».
- **Attendu** : positions entières dans tous les cas (`np.array([], dtype=int)`) ; l'estimateur signale alors
  lui-même l'apprentissage vide (« Found array with 0 sample(s) »).
- **Reproduction** : `next(OutOfSampleSplit(n_splits=3, gap=7).split(np.zeros(12)))[0].dtype   # float64`
- **Correctif** : toutes les positions sont des tableaux d'entiers (`dtype=int`) ; un apprentissage vide n'est de toute façon plus produit (ANO-CV-003).
- **Test** : `tests/unit/crossvals/base/test_properties.py::TestNoLeakage::test_folds_are_valid_positions`, `::TestInSampleContract::test_contract_holds`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-009 — `InSampleSplit` sans ligne de test : « max() iterable argument is empty »
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::InSampleSplit._iter_default_timeseries_test_indices`,
  `::_get_timeseries_train_indices`
- **Sévérité** : mineure
- **Observé** : quand la période de test calculée est vide (jeu vide ; moins de `n_splits + 1` lignes sans
  `test_size`, une observation comprise ; `test_size=0`), le pli de test vide atteint `max(test_indices)` et lève
  « max() iterable argument is empty ». `OutOfSampleSplit` refuse les mêmes données par un message explicite
  (« Cannot have number of folds=… greater than the number of samples=… »).
- **Attendu** : `ValueError` qui nomme la cause (`test_size` nul, nombre de lignes insuffisant).
- **Reproduction** : `list(InSampleSplit().split(pd.Series([1.0, 2.0, 3.0])))   # 3 // 6 = 0 ligne de test`
- **Correctif** : « Cannot split an empty dataset: number of samples=0 » pour un jeu vide ; « Empty test period: test_size=0 for number of samples=3 with n_splits=5; set test_size or reduce n_splits » quand la taille calculée est nulle ; `test_size=0` est refusé par la validation des paramètres (ANO-CV-010).
- **Test** : `tests/unit/crossvals/base/test_in_sample_split.py::TestDegenerateData::test_empty_test_period_is_refused_with_a_clear_message` (5 cas), `tests/unit/crossvals/test_time_series.py::TestTSInSampleSplit::test_empty_series`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-010 — `gap`, `test_size` et `max_train_size` jamais validés ; un `gap` négatif fait fuiter
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::OutOfSampleSplit`, `::InSampleSplit`
- **Sévérité** : majeure
- **Observé** : seul `n_splits` est contrôlé (par `_BaseKFold`). `gap=-1` place la première date de test de chaque
  pli dans l'apprentissage, sans erreur. `test_size=0` lève « range() arg 3 must not be zero », `test_size=-1`
  « min() iterable argument is empty ». `max_train_size=0` vaut « pas de limite » en hors échantillon (test de
  vérité) mais « apprentissage = test » en échantillon (`is not None`) ; `max_train_size=-1` donne des
  apprentissages vides.
- **Attendu** : `ValueError` nommant le paramètre pour `gap < 0`, `test_size < 1`, `max_train_size < 1` ; même
  interprétation de `max_train_size` dans les deux familles.
- **Reproduction** : `next(OutOfSampleSplit(n_splits=3, gap=-1).split(np.zeros(12)))   # ([0..3], [3, 4, 5])`
- **Correctif** : `split` valide les paramètres à l'appel : « gap must be a non-negative integer, got -1 », « test_size must be a positive integer or None, got 0 », « max_train_size must be a positive integer or None, got 0 » (booléens et flottants refusés, entiers numpy acceptés). `max_train_size` se lit `is not None` dans les deux familles.
- **Test** : `tests/unit/crossvals/base/test_out_of_sample_split.py::TestInvalidParameters` (`test_is_refused_with_its_name`, 8 cas ; `test_numpy_integers_are_accepted`), `tests/unit/crossvals/base/test_in_sample_split.py::TestInvalidParameters::test_is_refused_with_its_name` (4 cas)
- **Statut** : corrigée (2026-10-16)

### ANO-CV-011 — `test_indices` : formes de dates courantes refusées par une `TypeError` brute
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::_resolve_test_positions`
- **Sévérité** : mineure
- **Observé** : seules une liste ou un tableau dont le **premier** élément est un entier, une chaîne, un `Timestamp`,
  un `datetime` ou un tuple sont résolus. Un `DatetimeIndex` (`X.index[[3, 7]]`), des `numpy.datetime64`, des
  `datetime.date`, des `pandas.Period` (sur un `PeriodIndex`) ou une chaîne seule passent tels quels à `np.array` et
  échouent sur la comparaison `>= n_samples` (`TypeError` ou `UFuncTypeError`, et non la `ValueError` documentée).
  Des flottants (`[3.0]`) sont pris pour des positions et produisent des tableaux d'indices `float64` ; un booléen
  est pris pour la position 0 ou 1.
- **Attendu** : toute représentation d'une date résolue comme une chaîne ; refus explicite (`ValueError`) de ce qui
  n'est ni une position entière ni une date.
- **Reproduction** : `list(OutOfSampleSplit(test_indices=X.index[[3, 7]]).split(X))   # UFuncTypeError`
- **Correctif** : `_parse_test_indices` classe **chaque** élément : entier (Python ou numpy) = position, tuple = couple (entité, date), chaîne / `date` / `datetime` / `Timestamp` / `numpy.datetime64` / `Period` = date ; conteneurs acceptés : liste, tuple, tableau, `Index` (dont `DatetimeIndex`), `Series` ; une valeur seule est un indice. Refus explicites : booléen (décision de l'auteur, « got the boolean True »), flottant, `None`, mélange de natures. `_normalize_label` convertit une date au type de l'index (fuseau horaire, période) ; une date partielle qui désigne plusieurs lignes est refusée.
- **Test** : `tests/unit/crossvals/base/test_out_of_sample_split.py::TestExplicitTestIndices` (`test_other_date_forms_are_resolved` 5 cas, `test_single_date_is_one_test_start`, `test_dates_on_a_time_zone_aware_index`, `test_other_types_are_refused`, `test_booleans_are_refused`, `test_booleans_are_refused_by_get_n_splits`, `test_positions_and_dates_cannot_be_mixed`, `test_partial_date_matching_several_rows_is_refused`), `tests/unit/crossvals/base/test_index_robustness.py::TestIndexKinds::test_test_index_on_a_period_index`, `::test_partial_date_matching_one_row`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-012 — Dates répétées : une date à la fois en apprentissage et en test
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::OutOfSampleSplit` (découpage positionnel),
  `::_resolve_test_positions`
- **Sévérité** : à arbitrer
- **Observé** : un index trié qui répète une date n'est ni refusé ni signalé ; les plis sont coupés par position,
  si bien qu'une date répétée de part et d'autre de la frontière est apprise et testée dans le même pli. Une date
  répétée donnée en `test_indices` se résout en `slice` (`get_loc`) et lève « '>=' not supported between instances
  of 'slice' and 'int' ».
- **Attendu** : à trancher : refuser les dates répétées d'une série (`ValueError`), ou couper les plis sur les dates
  (toutes les lignes d'une date du même côté) ; dans les deux cas, une date répétée en `test_indices` désigne toutes
  ses lignes ou est refusée explicitement.
- **Reproduction** :
  ```python
  d = pd.to_datetime(['2024-01-31', '2024-02-29', '2024-02-29', '2024-03-31', '2024-04-30', '2024-05-31'])
  next(OutOfSampleSplit(n_splits=2, test_size=2).split(pd.Series(range(6), index=d)))   # ([0, 1], [2, 3])
  ```
- **Correctif** : décision de l'auteur (2026-10-16) : refus. `_verify_and_sort_data` lève « X has duplicated dates: [Timestamp('2024-02-29 00:00:00')]; each date must appear once » (« (entity, date) pairs … once per entity » pour un panel), avant tout tri et avant la résolution des `test_indices`.
- **Test** : `tests/unit/crossvals/base/test_index_robustness.py::TestDuplicatedDates::test_is_refused_naming_the_date` (4 cas), `::test_unsorted_duplicates_are_refused_before_sorting`, `tests/unit/crossvals/base/test_out_of_sample_split.py::TestPanelGroups::test_duplicated_entity_date_pairs_are_refused`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-013 — En échantillon, plusieurs points de test et `max_train_size` : des points de test hors apprentissage
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::InSampleSplit._get_timeseries_train_indices`,
  `::_get_group_train_indices`
- **Sévérité** : mineure
- **Observé** : la fenêtre vaut `max(max_train_size, len(test))` lignes avant la dernière date de test : elle suppose
  un test contigu. Avec des points espacés (`test_indices=[1, 7, 10], max_train_size=4`), l'apprentissage est
  `[7, 8, 9, 10]` et ne contient plus le point 1, contrairement au contrat « training includes test period ».
- **Attendu** : chaque point de test reste dans l'apprentissage (fenêtre d'au moins l'étendue du test), ou refus
  explicite de la combinaison.
- **Reproduction** : `list(InSampleSplit(test_indices=[1, 7, 10], max_train_size=4).split(np.zeros(12)))`
- **Correctif** : la fenêtre compte `max(max_train_size, étendue du test)` lignes, l'étendue allant de la première à la dernière ligne de test (de l'entité pour un panel).
- **Test** : `tests/unit/crossvals/base/test_in_sample_split.py::TestExplicitTestIndices::test_several_points_with_max_train_size_stay_in_training`, `::test_several_points_with_a_wider_window`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-014 — `X` sous forme de liste : `AttributeError`
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::OutOfSampleSplit.split`, `::InSampleSplit.split`,
  `::_verify_and_sort_data`
- **Sévérité** : mineure
- **Observé** : la détection d'un objet pandas repose sur `hasattr(X, 'index')`, vrai pour une liste (méthode
  `list.index`). `split(list)` lève « 'builtin_function_or_method' object has no attribute 'is_monotonic_increasing' »,
  et `cross_validate` sur des listes (qu'il transmet telles quelles) lève « … has no attribute 'equals' ». Les
  splitters sklearn acceptent tout tableau.
- **Attendu** : une liste est découpée comme le tableau équivalent (test `isinstance(…, (pd.Series, pd.DataFrame))`).
- **Reproduction** : `list(OutOfSampleSplit(n_splits=3).split(list(range(12))))`
- **Correctif** : la détection d'un objet pandas teste `isinstance(X, (pd.Series, pd.DataFrame))` (aide `_is_pandas`), pour `X` comme pour `y`.
- **Test** : `tests/unit/crossvals/base/test_index_robustness.py::TestContainers::test_list_gives_the_folds_of_the_array` (2 cas), `tests/unit/crossvals/test_time_series.py::TestSklearnModelSelection::test_cross_validate_on_lists`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-015 — Panel : `test_indices` en positions ou en tuples (entité, date) inutilisables
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::OutOfSampleSplit._iter_group_test_indices`,
  `::_get_group_train_indices`, `::InSampleSplit._iter_group_test_indices`
- **Sévérité** : majeure
- **Observé** : sur un panel (avec `groups`), chaque position ou tuple ouvre son propre pli, limité à son entité ; le
  calcul de l'apprentissage lève alors « Cannot find test indices for the group : B » pour les autres entités. Les
  tuples, annoncés par les docstrings (« Tuples (entity, date) for specific entity-date combinations »), ne
  fonctionnent donc jamais en hors échantillon dès qu'il y a deux entités ; en échantillon, une position seule échoue
  de même (plusieurs positions ou tuples fonctionnent : un seul pli). Même cause que l'erreur des panels
  déséquilibrés (ANO-CV-006, B3).
- **Attendu** : des plis sans fuite qui testent les lignes demandées ; une entité sans ligne de test dans un pli n'y
  est pas évaluée (ou y apprend sur son historique), sans erreur.
- **Reproduction** :
  ```python
  cv = OutOfSampleSplit(test_indices=[('A', d[3]), ('B', d[3])], test_size=2)
  list(cv.split(panel, groups=panel.index.get_level_values(0)))   # ValueError: ... group : B
  ```
- **Correctif** : une position ou un couple (entité, date) ouvre un pli qui teste **et entraîne** son entité seule ; les autres entités sont absentes du pli (`every_group=False` dans le calcul de l'apprentissage). Un test par position est coupé à la fin de son entité (en échantillon aussi). Le chemin des dates garde l'erreur « Cannot find test indices for the group » (ANO-CV-006, B3).
- **Test** : `tests/unit/crossvals/base/test_out_of_sample_split.py::TestPanelGroups::test_entity_specific_test_indices` (3 cas), `::test_entity_specific_fold_with_gap_and_window`, `tests/unit/crossvals/base/test_in_sample_split.py::TestPanelGroups::test_single_entity_specific_index` (2 cas), `::test_tuples_of_one_entity`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-016 — `groups` jamais confronté à `X` : aucun pli, ou erreur sans rapport
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::OutOfSampleSplit`, `::InSampleSplit` (chemins panel)
- **Sévérité** : mineure
- **Observé** : les entités sont lues dans `groups`, mais les positions de test par `X.index.get_loc((groupe, date))`.
  Des `groups` qui ne reprennent pas le premier niveau de l'index (autre casse, autre codage) donnent **zéro pli**,
  sans erreur, avec un avertissement « Cannot find test period … » par entité et par date ; `get_n_splits` en
  annonce `n_splits`. Des `groups` plus courts que `X` ne sont pas refusés : les lignes omises n'appartiennent à
  aucun groupe et l'erreur porte sur un autre groupe.
- **Attendu** : `ValueError` si `groups` n'a pas une étiquette par ligne ou ne correspond pas au premier niveau de
  l'index (ou bien `groups` lu comme seule source des entités, à documenter).
- **Reproduction** : `list(OutOfSampleSplit(n_splits=2).split(panel, groups=np.array(['a'] * 6 + ['b'] * 6)))   # []`
- **Correctif** : `_check_groups` refuse des `groups` de longueur différente (« groups must hold one label per row: got 6 labels for 12 rows ») ou différents du premier niveau (« groups must match the first index level (entity) of X, row by row »). Les entités sont ensuite lues dans `groups` seul ; la table `_precompute_multiindex_mappings` est supprimée.
- **Test** : `tests/unit/crossvals/base/test_out_of_sample_split.py::TestPanelGroups::test_groups_inconsistent_with_the_index_are_refused` (2 cas), `::test_groups_of_another_length_are_refused`, `tests/unit/crossvals/base/test_in_sample_split.py::TestPanelGroups::test_groups_inconsistent_with_the_index_are_refused` (3 cas)
- **Statut** : corrigée (2026-10-16)

### ANO-CV-017 — Un panel passé à un splitter de série est découpé en silence
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/time_series.py::TSOutOfSampleSplit`, `::TSInSampleSplit`
- **Sévérité** : à arbitrer
- **Observé** : un `MultiIndex` (entité, date) passé sans `groups` est découpé par position comme une seule série :
  les plis de test ne contiennent que la dernière entité, et l'apprentissage contient les autres entités aux mêmes
  dates (ou plus tard) que le test. Aucun avertissement, alors que `groups` déclenche, lui, « The groups parameter is
  ignored ».
- **Attendu** : à trancher : avertissement ou erreur orientant vers `Panel*Split`, ou comportement documenté.
- **Correctif** : décision de l'auteur (2026-10-16) : cas non toléré. `TSOutOfSampleSplit.split` / `TSInSampleSplit.split` lèvent « TSOutOfSampleSplit splits a single time series, but X has a MultiIndex (panel data); use PanelOutOfSampleSplit instead » (`Series` comme `DataFrame`), avant l'avertissement sur `groups`.
- **Test** : `tests/unit/crossvals/test_time_series.py::TestPanelGivenToATimeSeriesSplitter::test_multiindex_is_refused` (4 cas), `::test_refusal_precedes_the_groups_warning`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-018 — Étiquette d'entité manquante : erreurs sans rapport avec la cause
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py` (chemins panel ; `_precompute_multiindex_mappings` s'appuie sur
  `groupby(level=0)`, qui écarte les clés manquantes)
- **Sévérité** : mineure
- **Observé** : une entité `NaN` dans le premier niveau atteint les branches de repli des tables d'entités (seul cas
  où elles servent), puis échoue : « Cannot find test indices for the group : nan » (`groups == nan` est toujours
  faux) ou, en échantillon sans `test_indices`, « max() iterable argument is empty ».
- **Attendu** : refus explicite d'une étiquette d'entité manquante (même règle que la base des transformateurs,
  ANO-BASE-011).
- **Reproduction** : `panel` à entités `[1.0, nan]` ; `list(OutOfSampleSplit(n_splits=2).split(panel, groups=panel.index.get_level_values(0)))`
- **Correctif** : `_check_groups` lève « Entity labels must not be missing: the entity level or groups holds a missing value ».
- **Test** : `tests/unit/crossvals/base/test_index_robustness.py::TestMissingEntityLabel::test_is_refused_with_a_clear_message` (4 cas)
- **Statut** : corrigée (2026-10-16)

### ANO-CV-019 — Panel en échantillon par défaut : `test_size` réduit à une ligne par entité
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::InSampleSplit._iter_default_group_test_indices`
- **Sévérité** : mineure
- **Observé** : la période de test commence à la **dernière** date du panel ; `test_size` ne peut donc que la couper
  à la fin de l'entité : une seule ligne par entité, quel que soit `test_size`. Sur une série, la même classe teste
  bien les `test_size` dernières dates. La docstring annonce « with size controlled by test_size parameter ».
- **Attendu** : les `test_size` dernières dates de chaque entité.
- **Reproduction** : `list(InSampleSplit(test_size=2).split(panel_2x6, groups=...))   # test [5, 11]`
- **Correctif** : la période de test par défaut d'un panel en échantillon est faite des `test_size` dernières dates du panel (dates triées), testées pour chaque entité qui les porte ; une entité qui en manque une est signalée.
- **Test** : `tests/unit/crossvals/base/test_in_sample_split.py::TestPanelGroups::test_default_fold_holds_the_last_test_size_dates`, `::test_default_fold_is_the_last_date`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-020 — Docstrings et page de concept de `crossvals` : exemples et affirmations inexacts
- **Type** : [DOC] docstring ≠ code
- **Composant** : `tsforecast/crossvals/base.py`, `tsforecast/crossvals/time_series.py`,
  `docs/concepts/cross_validation.md`
- **Sévérité** : mineure
- **Observé** :
  1. `_verify_and_sort_data` : `sort_indices` décrit comme « mapping original positions to sorted positions » ; c'est
     l'inverse (origine d'ANO-CV-001).
  2. Exemples non exécutables ou trompeurs : `InSampleSplit` utilise `X` et `groups` non définis ;
     `TSOutOfSampleSplit` / `TSInSampleSplit` utilisent `pd`, `np` (non importés par le module) et `np.random.randn`
     sans graine ; `OutOfSampleSplit`, `_resolve_test_positions` et `_precompute_multiindex_mappings` affichent sans
     sortie attendue ou en commentaire ; le troisième exemple de `OutOfSampleSplit` découpe un panel sans `groups`
     (découpage positionnel, ANO-CV-017) ; `_resolve_test_positions` annonce lever des `ValueError` seulement
     (ANO-CV-011).
  3. `InSampleSplit` : « test_size … calculated size for default splits » ; sur un panel, la taille par défaut vaut 1
     (et `test_size` est réduit à une ligne, ANO-CV-019) ; `n_splits` ne fixe jamais le nombre de plis (toujours 1).
  4. `docs/concepts/cross_validation.md` : « Tous les splitters partagent l'API de `OutOfSampleSplit` » (les
     splitters en échantillon n'ont pas `gap`) ; « l'entraînement précède toujours le test » (faux en échantillon) ;
     une série est décrite par un `DatetimeIndex` alors que tout index trié convient (découpage positionnel) ;
     l'exemple `GridSearchCV(estimator, param_grid, cv=cv)` ne fonctionne qu'avec les plis par défaut hors
     échantillon (ANO-CV-002).
- **Attendu** : docstrings et page alignées sur le code, exemples exécutables (recoupe N7).
- **Correctif** : docstrings de `crossvals/base.py`, `crossvals/time_series.py` et `crossvals/__init__.py` réécrites (en-têtes de module, exemples exécutables : `pytest --doctest-modules tsforecast/crossvals` 22 passed) ; page `docs/concepts/cross_validation.md` alignée (index trié, dates répétées refusées, `gap` hors échantillon seulement, `get_n_splits`, formes de `test_indices`). Les docstrings de `crossvals/panel.py` (ANO-CV-005) restent à B3.
- **Test** : doctests du paquet `tsforecast/crossvals`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-021 — Branches inatteignables de `crossvals/base.py` (code mort)
- **Type** : [CODE] comportement (nettoyage, pas un bogue)
- **Composant** : `tsforecast/crossvals/base.py`
- **Sévérité** : cosmétique
- **Observé** : décelé par la couverture (98 %, seules ces branches manquent) : `if hasattr(X.index, 'get_loc')` de
  `OutOfSampleSplit._iter_group_test_indices` et `if hasattr(X, 'index') and hasattr(X.index, 'get_level_values')`
  de `_iter_default_test_indices` / `InSampleSplit._iter_default_group_test_indices` (toujours vrais : avec `groups`,
  `_verify_and_sort_data` exige un `MultiIndex`) ; le dernier repli `group_positions = []` des trois méthodes de
  test par groupe (`level_0_values` existe dès qu'il y a un `MultiIndex`) ; dans
  `InSampleSplit._iter_timeseries_test_indices` et `_iter_group_test_indices`, le filtre `if test_idx < n_samples`
  et le test `if test_indices:` (les positions hors bornes sont refusées en amont, la liste n'est jamais vide). Les
  autres replis (`elif level_0_values is not None`) ne servent qu'aux entités `NaN` (ANO-CV-018).
- **Attendu** : branches supprimées, sans changement de comportement.
- **Correctif** : branches supprimées par la réécriture de `base.py` (tables MultiIndex et replis inutiles retirés) : `base.py` 100 % lignes et branches.
- **Test** : couverture des tests de `tests/unit/crossvals/base/` et `tests/unit/crossvals/test_time_series.py`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-022 — Index de libellés non trié : tri alphabétique pris pour un ordre chronologique
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::_verify_and_sort_data`
- **Sévérité** : majeure
- **Observé** : un index non croissant était trié avec `argsort`, quel que soit son type. Pour des chaînes, l'ordre
  obtenu est alphabétique : avec `timeseries_to_string(ts, format='%m/%d/%Y')` sur juillet 2023 à juin 2024, le
  premier pli testait `['10/31/2023', '10/31/2024', '11/30/2023']` et apprenait sur 2024 (fuite), avec un simple
  avertissement « not sorted by date ». Même effet pour des libellés numérotés (`'period_10'` avant `'period_2'`) et
  pour le niveau des dates d'un panel.
- **Attendu** : décision de l'auteur (2026-10-16) : refus d'un index de libellés non trié.
- **Correctif** : `_is_chronologically_sortable` : dates, périodes, durées et nombres sont triés comme avant ; un index
  de libellés non croissant lève « Time series index is not sorted and holds labels of dtype object, whose sorted
  order is not chronological; convert it to dates (pd.to_datetime) or periods, or sort the rows by date » (« Panel
  date level … » pour un panel dont les dates ne sont pas croissantes dans une entité). Un index de libellés déjà
  croissant est pris dans l'ordre des lignes ; un panel à dates libellés, croissantes dans chaque entité mais aux
  entités entrelacées, est regroupé par un tri stable sur les seuls niveaux d'entité.
- **Test** : `tests/unit/crossvals/base/test_index_robustness.py::TestUnsortedData::test_unsorted_labels_are_refused`
  (6 cas), `::test_sorted_labels_are_taken_in_their_order`, `::test_numbers_and_durations_are_sorted` (3 cas),
  `::test_panel_with_label_dates_unsorted_within_an_entity_is_refused`,
  `::test_panel_with_label_dates_interleaved_entities_are_regrouped`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-023 — `get_n_splits` des variantes `PerEntity` ignore les entités ; `GridSearchCV` les refuse
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/panel.py::PanelOutOfSampleSplitPerEntity`, `::PanelInSampleSplitPerEntity`
  (`get_n_splits` hérité de `OutOfSampleSplit` / `InSampleSplit`)
- **Sévérité** : majeure
- **Observé** : `get_n_splits` renvoie le nombre de plis **globaux** (`n_splits`, nombre de `test_indices`, ou 1 en
  échantillon), alors que `split` en produit un par entité testée et par pli global. `GridSearchCV` compare les deux
  et refuse le splitter : « cv.split and cv.get_n_splits returned inconsistent results. Expected 2 splits, got 6 »
  (« Expected 1 splits, got 3 » en échantillon). `cross_validate`, qui ne lit pas `get_n_splits`, fonctionne.
- **Attendu** : `get_n_splits(X, y, groups)` égal au nombre de plis produits quand les données sont fournies
  (`GridSearchCV` les fournit). Sans données, le nombre d'entités est inconnu : à trancher (nombre de plis globaux
  documenté, ou erreur), en accord avec la convention retenue pour ANO-CV-002.
- **Reproduction** :
  ```python
  from sklearn.linear_model import Ridge
  from sklearn.model_selection import GridSearchCV
  cv = PanelOutOfSampleSplitPerEntity(n_splits=2)
  GridSearchCV(Ridge(), {'alpha': [0.1, 1.0]}, cv=cv).fit(panel_2_entities_12_dates, y)
  # ValueError: cv.split and cv.get_n_splits returned inconsistent results. Expected 2 splits, got 4
  ```
- **Correctif** : décision de l'auteur (2026-10-16) : `PanelOutOfSampleSplitPerEntity.get_n_splits` /
  `PanelInSampleSplitPerEntity.get_n_splits` exigent `X` et comptent exactement les plis de `split` (sans répéter ses
  avertissements) ; sans `X`, « … .get_n_splits requires X: the number of folds depends on the entities and their
  dates » (même choix que `LeaveOneGroupOut`, qui exige `groups`). Les variantes globales gardent la convention de
  ANO-CV-002.
- **Test** : `tests/unit/crossvals/panel/test_contract.py::TestNumberOfSplits::test_per_entity_count_with_the_data`
  (6 cas), `::test_per_entity_count_does_not_repeat_the_warnings`, `::test_per_entity_count_without_the_data_is_refused`
  (2 cas), `tests/integration/crossvals/test_integration.py::TestPanelSplittersInGridSearch::test_per_entity_splitter` (2 cas)
- **Statut** : corrigée (2026-10-16)

### ANO-CV-024 — Entité dont la clé est un tuple, dans un seul niveau : inaccessible par un tuple de test
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::_pair_key`
- **Sévérité** : mineure
- **Observé** : sur un panel à un seul niveau d'entité dont les valeurs sont des tuples (`('FR', 'b')`), le tuple de test
  `(('FR', 'b'), date)` est lu comme une entité sur deux niveaux et refusé : « Panel index (('FR', 'b'), '2024-06-30')
  does not match the index: expected 1 entity value(s) followed by a date ». La forme niveau par niveau
  `('FR', 'b', date)` est refusée de même : l'entité ne peut être désignée d'aucune façon. Les plis par défaut, par date
  ou par position fonctionnent.
- **Attendu** : avec un seul niveau d'entité, le premier élément du tuple de test est l'entité entière, tuple compris.
- **Reproduction** :
  ```python
  level = pd.Index([('FR', 'a')] * 3 + [('FR', 'b')] * 3, tupleize_cols=False)
  X = pd.DataFrame({'v': range(6)}, index=pd.MultiIndex.from_arrays([level, list(dates[:3]) * 2]))
  list(PanelOutOfSampleSplit(test_indices=[(('FR', 'b'), dates[2])]).split(X))   # ValueError
  ```
- **Correctif** : `_pair_key` lit, quand il n'y a qu'un niveau d'entité, le premier élément d'un couple comme l'entité
  entière.
- **Test** : `tests/unit/crossvals/panel/test_index_robustness.py::TestEntityKeys::test_entity_date_tuple_with_tuple_keys`,
  doctest de `_pair_key`
- **Statut** : corrigée (2026-10-16)

### ANO-CV-025 — Panel vide en échantillon : un pli sans ligne au lieu d'une erreur
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/crossvals/base.py::InSampleSplit._iter_default_group_test_indices`
- **Sévérité** : mineure
- **Observé** : `PanelInSampleSplit().split(panel.iloc[:0])` produit un pli `([], [])` ; `PanelInSampleSplitPerEntity`
  ne produit aucun pli. Une série vide est refusée par « Cannot split an empty dataset: number of samples=0 », et
  `PanelOutOfSampleSplit` refuse un panel vide (« Too many splits=2 for number of time points=0 … »).
- **Attendu** : le même refus explicite que pour une série vide.
- **Reproduction** : `list(PanelInSampleSplit().split(panel.iloc[:0]))   # [(array([]), array([]))]`
- **Correctif** : `_iter_default_group_test_indices` refuse un panel vide (« Cannot split an empty dataset: number of
  samples=0 ») ; les variantes `PerEntity` aussi, hors et dans l'échantillon.
- **Test** : `tests/unit/crossvals/panel/test_panel_split.py::TestDegenerateEntities::test_empty_panel_is_refused_in_sample`,
  `tests/unit/crossvals/panel/test_contract.py::TestPerEntityFoldsWithDifferentCoverage::test_empty_panel_is_refused` (2 cas)
- **Statut** : corrigée (2026-10-16)

## PANEL

### ANO-PANEL-009 — Sortie en tableau : le niveau des dates remplacé par des positions
- **Type** : [CODE] comportement
- **Composant** : `tsforecast/panel/transformers.py::PanelwiseTransformer._reconstruct_entity_dataframe` (appelée par
  `_transform` et `inverse_transform`)
- **Sévérité** : majeure
- **Observé** : quand le transformateur d'une entité renvoie un `ndarray` (cas de tout transformateur sklearn sans
  `set_output(transform='pandas')`, `StandardScaler` compris), le `DataFrame` reconstruit reçoit un `RangeIndex` : le
  niveau des dates de la sortie vaut 0..n-1 pour chaque entité, et `transform(X).index` diffère de `X.index`. Une étape
  suivante qui aligne sur l'index (cible, `DropNaNTarget`) ne retrouve plus les dates. Relevé par B3 (intégration
  `crossvals`) ; numéro pris après les identifiants indicatifs 001 à 008 de `tests/PLAN_B.md`.
- **Attendu** : un tableau qui garde une ligne par ligne d'entrée reprend les dates de l'entité ; `transform` et
  `inverse_transform` restituent l'index d'entrée.
- **Reproduction** :
  ```python
  out = PanelwiseTransformer(StandardScaler()).fit(panel).transform(panel)
  out.index.get_level_values(-1)[:3]   # [0, 1, 2] au lieu des dates
  ```
- **Correctif** : décision de l'auteur (2026-10-16) : `_reconstruct_entity_dataframe` reçoit l'index des lignes de
  l'entité et l'applique à un tableau de même nombre de lignes (un tableau d'une autre longueur garde des positions).
- **Test** : `tests/unit/panel/test_transformers.py::TestDatesOfAnArrayOutput` (4 tests)
- **Statut** : corrigée (2026-10-16)

## XY
