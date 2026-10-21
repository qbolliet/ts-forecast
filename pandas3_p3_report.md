# Rapport P3 — Unité des dates en sortie : application à `delays` et à `HighFrequencyImputer`

pandas 3.0.6, scikit-learn 1.9.1 — 2026-10-20. Anomalies traitées : ANO-DELAYS-045 (dates : corrigée ; chaînes :
ouverte, P4), ANO-FREQ-086 (corrigée). Anomalie consignée hors périmètre : ANO-UTILS-073 (ouverte). Rien n'est commité.

## 1. Critères

| Critère | Avant (HEAD `d367a16`) | Après | Statut |
|---|---|---|---|
| Sorties de référence (895 cas) | — | **0 écart** d'instant (entiers de ns) ou de valeur (bits float64), 0 exception ni avertissement changé, **0 écart à la règle** | ✅ |
| Suite complète, `-W error::FutureWarning -W error::DeprecationWarning` | 13 969 réussis | **14 119 réussis, 2 xfail stricts** (voulus, §5), 0 échec | ✅ |
| Couverture lignes + branches | 99,3699 % | 99,3712 % ; `high_frequency_imputer.py` 99,329 → 99,332 %, `data_manager.py` 100 → 100 %, `report.py` 96,69 → 96,69 % ; aucun module en baisse | ✅ |
| Durée (même commande, sous couverture, deux passes **simultanées**) | 1 091,6 s | 1 103,5 s, **+1,1 %** (seuls les deux fichiers de tests modifiés bougent : +4,3 s, +2,2 s) | ✅ |
| `check_conventions.py --rule all` | 23 violations bloquantes (doctests) | identique, aux numéros de ligne près | ✅ hors doctests |
| Doctests des modules touchés | 7 réussis, 1 échec (`_stage_frequency_label`, préexistant) | 8 réussis (+ `_datetime_units`), même échec | ✅ |

**Mesure de durée.** Deux passes séquentielles du même code ont varié de +19 % et +41 % sur ce poste : le
ralentissement touchait un bloc contigu de fichiers de l'imputeur, que j'avais laissés intacts (`test_invariants.py`
151 → 240 s), qui durent pourtant autant isolés, sous couverture ou non (`test_invariants.py` : 162 / 166 s avant, 161 /
159 s après, en alternance). C'est donc la machine qui ralentissait par moments, pas le code. Mesure retenue : les deux
suites lancées en même temps sur deux copies du scratchpad (copie de HEAD, copie de l'arbre modifié), donc dans les
mêmes conditions.

## 2. Règle appliquée (arbitrage P2)

Toute date rendue est à l'unité de l'entrée de référence, plus fine seulement pour une date qu'elle ne représente pas,
jamais arrondie. Entrée de référence : `new_data` pour les délais (axe temporel validé : index, dernier niveau d'un
panel, colonne `time_col`) ; le `X` de la passe pour l'imputeur, `y` compris. Le fuseau des dates d'observation et des
bornes est celui des données ; `download_date` garde le sien (arbitrage du 2026-10-04).

## 3. Changements de code

| Fichier | Changement |
|---|---|
| `delays/data_manager.py` | `download_date` résolue puis exprimée par `as_exact_unit(pd.Timestamp(date), unité de new_data)` ; `observation_date` ramenée à l'unité de `new_data` (l'alignement sur `existing_data` la portait à la plus fine des deux unités quand les dates diffèrent) ; bornes de période construites par `_as_output_dates` (unité de `new_data` ; une colonne entièrement `NaT`, sans fréquence détectée, reçoit le fuseau des données, au lieu de l'unité `s` naïve que pandas infère) ; `_empty_publication_delays` type ses dates sur l'axe temporel de `new_data` et sur la date de téléchargement rendue. Helpers `_time_axis`, `_as_output_dates`. Docstring `Returns:` complétée. |
| `delays/report.py` | Docstring : `DelayDetectionReport.download_date` est la date de la colonne (`pd.Timestamp`). |
| `frequency/high_frequency_imputer.py` | `_align_target_index` : mêmes instants, autre unité → `y` repositionné sur l'index de `X` (noms de niveaux de `y` conservés) ; helper `_datetime_units` (doctest). Docstring de classe : paragraphe « Unit of the dates ». |

`calculate_applicable_delay` ne rend aucune date : aucun changement ; ses résultats sont identiques au bit près
pour les quatre unités (vérifié en référence et par test).

## 4. Unités rendues avant / après

« entrée » = unité de `new_data` (délais) ou de `X` (imputeur) ; « + fine » = plus fine si une date l'exige.

| Composant | Sortie | Avant | Après |
|---|---|---|---|
| `compare_and_detect_delays` | `observation_date`, sans `existing_data` | entrée | entrée |
| idem | `observation_date`, avec `existing_data` | plus fine des deux jeux (dates différentes) | entrée |
| idem | `period_start` / `period_end` | entrée (P2) ; colonne tout-`NaT` : `s` naïf | entrée (+ fuseau des données) |
| idem | `download_date` (chaîne, `datetime`) | `us` | entrée, + fine (`'… 13:45:10.123456'` sur `s` → `us`) |
| idem | `download_date` (`Timestamp`) | unité du `Timestamp` | entrée, + fine (fuseau donné conservé) |
| idem | résultat vide | `datetime64[ns]` naïf (4 colonnes) | dtypes du résultat non vide (unité **et** fuseau) |
| `DelayDetectionReport.download_date` | | `datetime` (chaîne) ou `Timestamp` donné | `Timestamp` de la colonne ; `to_dict()` inchangé (`isoformat` sans unité) |
| `HighFrequencyImputer` | sortie, provenance, rejeu, inversion, état ajusté | entrée (P2) | entrée (vérifié) |
| idem | `fit_transform(X, y)`, `y` à une autre unité que `X` | plus fine des deux (X, y, provenance) ; état mêlant deux unités | unité de `X` |
| idem | passe de prédiction, `y` à une autre unité | unité de `y` | unité de `X`, valeurs inchangées |
| idem | `transform` à une autre unité que le `fit` | unité des données transformées | inchangé ; état ajusté intact |

## 5. Garde-fous

- **Sorties de référence** (scripts du scratchpad `p3_reference.py` / `p3_compare.py`, copie de HEAD contre arbre
  modifié, sérialisation neutre : instants en ns, bits float64, dtypes sans unité, unités à part) — **895 cas** :
  - 723 de délais : six jeux réalistes (panel hétérogène, série, série irrégulière, panel, panel à deux niveaux,
    PANEL-X) × 4 unités ; points de référence × unités de délai ; deux instantanés (`existing_data` en `s` et `ns`,
    deux modes) ; résultats vides (aucune observation, aucun changement) ; téléchargement en `Timestamp` `s` / `ns`,
    `datetime`, avec fuseau ; données avec fuseau ; `time_col` / `panel_cols` ; index de périodes et de chaînes ;
    fréquences indétectables ; grilles de 100 ms et de 1 s ; `to_dict()` / `to_frame()` des rapports ;
    `calculate_applicable_delay` sur chaque résultat ;
  - 172 de l'imputeur : passes TS / PANEL-F / PANEL / PANEL-X × `interpolate` / `model` / sortie plate (sortie,
    provenance, rejeu, inversion, masques et bornes de fenêtres) ; passes X / y avec `y` aux quatre unités ;
    `target_as_covariate` ; `XYPipeline` ; ajustement et transformation à deux unités ; matrice de
    `realistic_runs.py` (2 jeux × 3 stratégies × 3 modalités : valeurs **et** matrices de provenance) ; × 4 unités.
  - Résultat : 0 écart de contenu hors changements voulus (type de `report.download_date` ; fuseau ajouté aux
    colonnes de dates de 40 résultats vides) ; toutes les unités conformes à la règle ; les 238 résultats vides ont
    les dtypes de dates de leur équivalent non vide (**232 en différaient avant**). Côté imputeur, seules changent les
    unités des passes dont `y` est stocké à une autre unité que `X`.
- **Audit de l'état ajusté** (parcours récursif de tout l'état, attributs privés compris : plan, modèles, registres du
  matérialiseur, masques, bornes, instantanés) en `s` et `ns` sur TS, PANEL-F, PANEL-X et les jeux réalistes : une
  seule unité, celle du `fit`, sauf dans le cas `y` ci-dessus, désormais corrigé.

## 6. Tests

- `tests/unit/delays/data_manager/test_datetime_resolution.py` (36 → 156 tests) : unités épinglées remplacées par la
  règle (`_date_units`, unité attendue du téléchargement à la microseconde explicite par unité) ; deux instantanés
  (dates de `new_data` même face à un `existing_data` plus fin) ; `TestOutputUnit` (téléchargement en `Timestamp` /
  `datetime`, plus fin que les données, avec fuseau ; données avec fuseau ; fréquence indétectable aux dtypes d'une
  détectée ; rapport ; grille de 100 ms avec valeur d'or ; `time_col`) ; `TestEmptyResult` (vide = dtypes d'un non vide :
  4 unités × série / panel × naïf / fuseau × deux dates de téléchargement × deux causes de vide) ; `TestApplicableDelay`.
  **xfail strict** `test_empty_result_strings_have_the_dtypes_of_a_non_empty_one` (×2) : reste d'ANO-DELAYS-045 que P4
  doit faire passer (`reference_point`, `unit` et niveau `column` en `object` dans un résultat vide).
- `tests/unit/frequency/high_frequency_imputer/test_datetime_resolution.py` (20 → 52 tests) : chaque date de chaque
  sortie à l'unité de l'entrée ; provenance indexée exactement comme la sortie ; `inverse_transform(fit_transform(X))`
  égal à `X`, index et unité compris ; `TestTargetPasses` (passe d'entraînement ; `y` à une autre unité : entraînement,
  inversion, prédiction à l'unité de `X`) ; `TestFittedState` (état public à l'unité du `fit` ; `transform` à une autre
  unité : valeurs d'un ajustement à cette unité, état intact) ; `TestCachedPipeline` (`XYPipeline(memory=…)`). Passes
  calculées une fois par (jeu, stratégie, unité), partagées entre tests : les deux fichiers coûtent +6,5 s dans la
  suite (152 tests de plus).
- Aucune valeur d'or modifiée.

## 7. `XYPipeline(memory=…)`

L'empreinte de cache lit les dtypes, unité des dates comprise : les mêmes instants en `s` et en `ns` sont **deux
entrées**. Sous la règle, c'est voulu : les sorties des étapes, à l'unité de leur entrée, diffèrent aussi par leurs
dtypes, et une entrée partagée servirait l'une à l'unité de l'autre. Documenté (`docs/concepts/xy_pipeline.md`,
« Mise en cache ») et épinglé (`s`, `ns`, `s` → deux ajustements, chaque sortie à son unité, mêmes valeurs).

## 8. Documentation et registre

- `docs/tutorials/publication_delays.md` (unité des colonnes de dates, résultat vide),
  `docs/concepts/mixed_frequency_imputation.md` (unité des passes), `docs/concepts/xy_pipeline.md` (cache).
- `tests/ANOMALIES.md` : ANO-DELAYS-045 (Attendu, Correctif, Reste ouvert, Test ; statut « corrigée pour les dates ;
  partie chaînes ouverte (P4) ») ; ANO-FREQ-086 (audit, Correctif, `XYPipeline`, Test ; statut « corrigée ») ;
  **nouvelle ANO-UTILS-073** (§9).
- `CLAUDE.md` : non modifié (la règle y figure depuis P2).
- `tests_and_refactoring_prompts.md` : modifié pendant la session (compléments pandas 3 de N7 / N8) par l'auteur,
  pas par P3 ; laissé tel quel.

## 9. Points ouverts

- **ANO-UTILS-073** (à arbitrer) : `resolve_date` lit une chaîne avec `pd.to_datetime` puis la rend en `datetime` Python,
  ce qui **tronque les nanosecondes** (`UserWarning` de pandas). Sur des données en `ns`, un téléchargement
  `'… 00:00:00.000000001'` donne 777 600 000 000 µs de délai au lieu de 777 600 000 001 µs (valeur du `Timestamp`
  équivalent). Antérieur à P3, inchangé par lui. Pistes : `resolve_date` rendant un `pd.Timestamp`, ou lecture des
  chaînes par `pd.Timestamp` dans `compare_and_detect_delays`.
- **P4** : dtype des chaînes du résultat vide (xfail strict prêt à basculer), index `column` du rapport.
- **N7 / N8** : aucun doctest touché ; le seul ajouté (`_datetime_units`) n'affiche aucune unité dépendante de pandas
  au-delà de l'objet de l'exemple.
