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

## DELAYS

_Aucune entrée._

## FREQ

_Aucune entrée._
