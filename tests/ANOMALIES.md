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
  `-(-n // u)`), avec la durée de chaque unité en nanosecondes (`_DELAY_UNITS`) : plus aucun produit flottant, arrondi
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


## FREQ

_Aucune entrée._
