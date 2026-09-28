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

## DELAYS

_Aucune entrée._

## FREQ

_Aucune entrée._
