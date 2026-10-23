# Rapport P4 — dtype des chaînes en sortie (`str`), résultat vide des délais, pyarrow en extra

pandas 3.0.6, scikit-learn 1.9.1 — 2026-10-21. Décisions de l'auteur : **(2) harmoniser en `str`**, pyarrow
**(b) extra optionnel**. Anomalies : ANO-DELAYS-045 (partie chaînes : **corrigée**, statut global « corrigée »),
**ANO-FREQ-087** (nouvelle, corrigée). Rien n'est commité.

## 1. Critères

| Critère | Avant (HEAD `1a7e0ce`) | Après | Statut |
|---|---|---|---|
| Sorties de référence (284 cas, 880 objets pandas) | — | **0 valeur changée** ; dtypes changés = exactement ceux du §3 | ✅ |
| Suite complète `-W error::FutureWarning -W error::DeprecationWarning` (stockage python) | 14 119 réussis, 2 xfail stricts (ANO-DELAYS-045, partie chaînes) | **14 167 réussis**, 0 échec, 0 xfail | ✅ |
| Suite complète, mêmes `-W`, **avec pyarrow** (venv séparé du scratchpad) | 14 échecs (diagnostic du plan, §5) | **14 167 réussis**, 0 échec (pyarrow 26.0.0, stockage `pyarrow` vérifié) | ✅ |
| Couverture lignes + branches | 99,371 % | 99,372 % ; `report.py` 96,694 → 96,748 %, `delays/transformers.py` 99,608 → 99,610 %, `high_frequency_imputer.py` 99,332 → 99,333 % ; les autres modules inchangés, aucun en baisse | ✅ |
| `check_conventions.py --rule all` | 23 violations bloquantes (doctests) | identique (seuls trois « fichier long » informatifs changent de longueur) | ✅ hors doctests |
| Doctests (`--doctest-modules tsforecast`) | 23 échecs (lot N7 / N8), 311 réussis | identique (l'exemple ajouté à `_extract_param_by_variable` passe) | ✅ |

## 2. Règle appliquée (reportée dans CLAUDE.md, section pandas ≥ 3.0)

- Chaîne **produite par le paquet** (fréquence, unité, point de référence, origine, source, libellé d'étape) :
  `dtype='str'` **explicite**, quel que soit le contenu (vide, tout manquant) ; manquant = `NaN`.
- **Libellé repris de l'entrée** (colonnes, entités) : dtype inféré des valeurs ; les niveaux d'entité gardent celui
  des données ; un résultat vide a les dtypes d'un résultat non vide.
- `object` explicite seulement pour des valeurs hétérogènes (membres de `ProvenanceType`, tuples, `Period`,
  `feature_names_in_`), avec un commentaire au site.
- Hors pandas (dict, kwargs), une chaîne manquante reste `None`. Les tableaux `object` internes restent : on
  harmonise à la frontière publique.

## 3. Tableau des 35 sites `dtype=object` (`git grep` dans `tsforecast/`, 34 + une ligne de doctest)

| # | Site | Objet | Valeurs | Avant | Après | Risque traité |
|---|---|---|---|---|---|---|
| 1 | `base/transformers.py:383` | `feature_names_in_` | libellés | ndarray object | conservé (sklearn) | — |
| 2 | `crossvals/base.py:88` | `_labels_array` (interne) | libellés, tuples | object | conservé | tuples |
| 3 | `delays/data_manager.py` (résultat vide, niveau `column` et niveaux d'entité) | index | libellés | object | dtype d'un non vide (inféré des noms ; entités : celui des données ; `str` sans variable) | ANO-DELAYS-045 |
| 4–6 | idem, `frequency`, `reference_point`, `unit` vides | colonnes | chaînes | object | **str** | idem |
| 7 | `data_manager.py` (`frequency`, + `astype(object).where(…, None)`) | colonne | littéral / indétectable | object, `None` | **str**, `NaN` | 3 tests `is None` → `pd.isna` |
| 8 | `delays/report.py` `DelayDetectionReport.to_frame` | index `column` | libellés | object | inféré (`str`) | — |
| 9 | `delays/report.py` `DelayFitReport.to_frame` | index `column` | libellés | object | inféré (`str`) | — |
| 10 | `delays/transformers.py:417` `_delays_table` (interne) | colonne | libellés quelconques | object | conservé | — |
| 11–12 | `aggregation_constraint.py:119, 130` (interne) | Series | `Period` / `None`, tuples | object | conservés | — |
| 13 | `aggregation_constraint.py:1321` | `feature_names_in_` | libellés | object | conservé (sklearn) | — |
| 14 | `covariate_materializer.py:271` (doctest de `_combine_first`, privé) | — | chaînes + flottants | object | conservé | — |
| 15–16 | `covariate_materializer.py` `_mirror_blocks` (défauts `freq`, `origin`) | Series vides | codes, origines | object | **str** (= registres) | branche de `_combine_first` |
| 17–22 | `covariate_materializer.py:1414, 1415, 1475, 1548, 1581, 1609` | intermédiaires | chaînes + `None` | object | conservés (internes) | — |
| 23–25 | `high_frequency_imputer.py:2666, 2786, 3218` | lecture `to_numpy` | codes | ndarray object | conservés | — |
| 26 | `high_frequency_imputer.py:4825` | factorize | tuples | object | conservé | — |
| 27 | `high_frequency_imputer.py:5245` | niveau `frequency` de la sortie empilée | libellés d'étape | str (inféré) | **str explicite** | — |
| 28 | `frequency/provenance.py:353` | `provenance_matrix_` / `imputation_provenance_` | membres + non renseigné | object | conservé + commentaire | `is` / `isinstance` (~30 tests) |
| 29 | `stage_scaler.py:706` `row_modes` (interne) | ndarray | modalités | object | conservé | — |
| 30 | `stage_scaler.py:1809` | `feature_names_in_` | libellés | object | conservé (sklearn) | — |
| 31–32 | `training_set_builder.py:153–154` (doctest `TrainingSet`) | — | — | object | **str** | — |
| 33 | `training_set_builder.py:510` (interne) | lecture | codes | ndarray object | conservé | — |
| 34–35 | `training_set_builder.py` `row_frequency`, `row_origin` | Series | codes, origines | str (object si vide) | **`dtype='str'`** | `hfi` : filtre `origin is not None` → `pd.notna` |

## 4. Autres sorties publiques à chaînes

| Sortie | Avant | Après |
|---|---|---|
| `compare_and_detect_delays` : `reference_point`, `unit` (non vide) | str inféré | str explicite (inchangé) |
| `DelayFitReport.to_frame()` : `strategy`, `delay_unit`, `reference_point`, `frequency`, `target_frequency`, `*_source`, `delay_source` | str, str + NaN ou **object (tout `None`)** selon le contenu | **str** explicite (`None` → `NaN` dans la trame ; `to_dict()` garde `None`) |
| `prepare_entity_kwargs_from_delays` / fabrique | `None` lu dans une colonne object ; `NaN` si la table est en `str` | **`None`** dans tous les cas (normalisation à la lecture d'une colonne `str`) |
| `calculate_applicable_delay` : `unit`, `frequency`, `reference_point`, `aggregation_method` | str inféré (dtype de l'entrée) | str explicite |
| `ImputationPlan.to_diagnostic_frame()` : `stage`, `source_frequency`, `entities`, `*_taint`, `interpolation_method`, `materialization`, `training_blocks` | str, sauf `source_frequency` tout `None` → object | str explicite |
| idem `emitted_provenance` | str contenant des membres (**chaînes simples sous pyarrow**) | **object** construit tel quel (membres sous les deux stockages) |
| `ImputationProvenanceTracker.to_string_matrix()` | str inféré | str explicite |
| `imputation_provenance_` après `transform` | 584 cellules `ORIGINAL` / `AGGREGATED` en **simple `str`** (`mask(cond, membre)`) | membres (ANO-FREQ-087) |
| `CovariateMaterializer.origin_store`, `imputed_freq_store`, `snapshot()` | str inféré ; object selon l'appelant ou si tout `None` | str explicite à l'écriture (`_write_stores`) |
| `timeseries_to_string`, `split_summary`, métriques, `to_dict()` | str / dict | inchangés |

## 5. Mise en œuvre

**Code** (`tsforecast/`) : `delays/data_manager.py` (fréquence en `str`, résultat vide typé comme un non vide),
`delays/report.py` (index inféré, champs textuels en `str`), `delays/transformers.py` (kwargs : `None`),
`delays/calculator.py` (colonnes textuelles en `str`), `frequency/imputation_plan.py` (diagnostic),
`frequency/provenance.py` (`to_string_matrix`, commentaire de la matrice), `frequency/covariate_materializer.py`
(registres), `frequency/training_set_builder.py` (lignes), `frequency/high_frequency_imputer.py` (membres au rejeu,
filtre d'origine, niveau `frequency`, enregistrement). Docstrings mises à jour (`frequency` manquante = `NaN`).

**Sous pyarrow.** Une colonne `str` stocke sous pyarrow de simples chaînes : les membres d'une énumération y perdent leur
type. À HEAD, avec pyarrow installé, 14 tests du diagnostic du plan échouaient déjà (`is ProvenanceType.X`) ; la
colonne `emitted_provenance` est désormais construite en `object` (un `astype(object)` après coup ne suffit pas : la
conversion a déjà eu lieu). Aucun autre écart sous pyarrow.

**Sorties de référence** (script jetable du scratchpad : 5 jeux réalistes × 4 unités, deux points de référence, deux
unités de délai, sans / avec instantané, deux causes de vide, fréquence indétectable, rapports `to_frame` / `to_dict` /
`summary`, délais applicables, kwargs, `PublicationDelayTransformer` shift / mask / dict, HFI TS et panel ×
1 ou 2 étapes × `keep_lower_frequencies` : sorties, provenance au fit et au transform, diagnostic, matrice texte,
registres, jeux d'entraînement) : **0 valeur changée**. Marqueurs de manquant `None` → `NaN` : 20 fréquences
indétectables, 136 `target_frequency` et 136 `target_frequency_source` de `DelayFitReport.to_frame()` (stratégie
`shift`). 584 cellules de provenance redevenues des membres.

**pyarrow** : `[project.optional-dependencies] pyarrow = ["pyarrow>=13.0.0,<27"]` (13.0.0 : minimum de pandas 3 ;
résolu : 26.0.0), `uv.lock` régénéré ; job CI `tests-pyarrow` (`uv sync --group dev --extra pyarrow`, contrôle
`dtype.storage == 'pyarrow'`, tests rapides) sur PR et push ; README (Prérequis).

## 6. Tests

- Nouveau helper `string_dtypes(obj)` (`tests/support/perturbations.py`, compagnon de `datetime_units`, doctest).
- Nouveaux fichiers : `tests/unit/delays/data_manager/test_string_dtypes.py` (séries, panels à libellés `str` /
  `object`, entiers, deux causes de vide, rapport), `tests/unit/delays/calculator/test_string_dtypes.py`,
  `tests/unit/frequency/high_frequency_imputer/test_string_dtypes.py` (niveau `frequency`, membres au fit et au
  transform, diagnostic).
- Classes ajoutées : `test_report.py::TestStringDtypes`, `test_imputation_plan.py::TestDiagnosticFrameDtypes`,
  `test_stores.py::TestStoreDtypes` (dont `_combine_first` sur deux dtypes, branche que les registres, désormais
  tous `str`, n'atteignent plus), `test_origins_and_layers.py::TestRowDtypes`,
  `test_factories.py::…::test_missing_text_reaches_the_kwargs_as_none` (tables `str` et `object`), `test_tracker_reads.py::…::test_matrix_dtypes`,
  `test_string_labels.py::TestOutputStringDtypes` (libellés spéciaux × `object` / `str`).
- Adaptés : 3 `is None` → `pd.isna` (`test_delay_measure.py`), `test_imputation_plan.py` (`source_frequency`),
  `test_datetime_resolution.py` (comparaison de tous les dtypes du vide ; `xfail` strict d'ANO-DELAYS-045 retiré :
  il passe), helpers de registres en `'str'` comme en production.
- Garde-fous vérifiés sur HEAD : les tests des membres au transform et du diagnostic échouent sur HEAD.

## 7. Effet utilisateur

- `compare_and_detect_delays(...)['frequency']` : dtype `str`, fréquence indétectable = **`NaN`** (et non plus
  `None`) : `pd.isna` fonctionne comme avant, `x is None` ne fonctionne plus.
- Un code qui teste `dtype == object` sur ces sorties doit tester `'str'` (vrai avec ou sans pyarrow).
- Résultat vide des délais : mêmes dtypes qu'un résultat non vide ; une concaténation de résultats ne mélange plus
  `object` et `str`.
- `DelayFitReport.to_frame()` : colonnes textuelles toujours `str` (`None` → `NaN`) ; `to_dict()` / JSON inchangés.
- Provenance : toujours des membres de `ProvenanceType` (fit, transform, diagnostic), avec ou sans pyarrow.
- `pip install "ts-forecast[pyarrow]"` : même sémantique, stockage des chaînes plus rapide.

## 8. Points ouverts

- La matrice de provenance mêle deux marqueurs de cellule non renseignée (`NaN` à l'initialisation, `None` après
  `extend_index` / `clear_provenance`) : préexistant, hors périmètre (les lectures passent par `pd.isna`).
- `ImputationPlan.to_diagnostic_frame()` : `scale_factor` / `fit_scale_factor` (flottant ou `'per-row'`) et
  `interpolation_anchor` restent hétérogènes (dtype selon le contenu) : ce ne sont pas des colonnes de chaînes.
- Le job CI pyarrow n'exécute que les tests rapides ; la suite complète a été vérifiée localement sous pyarrow.
