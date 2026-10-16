# Plan de la campagne B — `base`, `crossvals`, `panel`, `xy`

> Fichier TRANSITOIRE (supprimé par C2). Lu par B1 à B8. Établi par B0 le 2026-10-16.

## 1. Vagues, dépendances, propriétaires

| Vague | Prompts | Dépend de |
|---|---|---|
| 1 | B1, B2, B4, B6 (en parallèle) | B0 |
| 2 | B3 (B2 **et** B4), B5 (B4 **et** B6), B7 (B6) | vague 1 |
| 3 | B8 | B1-B7 |

B3 dépend aussi de B4, et B5 aussi de B6 (ajouts au plan), pour qu'aucun fichier de support n'ait
deux auteurs en même temps. Ces dépendances, les chemins des tests réorganisés par B0 et le
découpage en paquets sont reportés dans les prompts B de `tests_and_refactoring_prompts.md`.

**Un seul propriétaire par fichier partagé.** Les autres prompts le lisent ; ceux d'une vague
ultérieure peuvent le compléter une fois le propriétaire terminé.

| Fichier | Propriétaire | Contenu à ajouter |
|---|---|---|
| `tests/support/perturbations.py` (+ `test_perturbations.py`) | **B4** | `rename_entities(df, mapping)` (clés avec espaces, accents, entiers, tuples) ; `interleave_entities(df)` (tri date puis entité) ; `unsort_within_entities(df, seed)` (trié par entité, pas par date) ; `with_single_observation_entity(df, entity)` ; `truncate_entity(df, entity, end)` (entité absente des derniers plis). Existants : `to_three_level_index`, `drop_entity`, `shuffle_rows`, `reverse_entities`, `with_duplicated_rows`, `single_observation`, `empty_like`, `to_period_index`, `to_period_start`/`end`, `with_special_column_names`, `with_index_names`. |
| `tests/support/estimators.py` (+ nouveau `test_estimators.py`) | **B6** | Mise au propre des doublures XY déplacées par B0 (`LogTransformXY`, `ScaleXY`, `DropRowsXY` : docstrings, et `LogTransformXY` non inversible sur les négatifs) ; `MixinScaleXY` (sous-classe minimale de `XYTransformerMixin`, (X, y) → (k·X, k·y)) ; `SpyTransformer` / `SpyXYTransformer` (journal des appels : méthode, index et forme de X, `y` reçu ou `None`) ; `FailingTransformer(fail_on=…, method='fit'\|'transform')` (échec ciblé par entité, pour B5). `SpyEstimator` existe : il enregistre `fit_X_`, `fit_y_` et `predict_X_`, mais exige `y`, n'enregistre ni `transform` ni `score`, et son compteur de classe `n_fits` est partagé entre tests et invisible avec `n_jobs > 1` ; ne pas le modifier (frequency en dépend), ajouter plutôt les doublures ci-dessus. |
| `tests/support/splits.py` (nouveau, + `test_splits.py`) | **B2** | `collect_folds(splitter, X, y=None, groups=None)`, `assert_no_leakage(X, folds, gap)` (sur les dates, entité par entité), `assert_folds_cover(...)`, `assert_n_splits_consistent(splitter, X, ...)`. Y transférer `TestHelpers` de `fixtures.py` s'il fait doublon. |
| `tests/support/fixtures.py` (section crossvals) | **B2**, puis B3 | B5, B6 et B7 n'y ajoutent rien : fixtures locales dans le `conftest.py` de leur paquet. |
| `tests/support/datasets.py` (+ `test_datasets.py`) | **B3** | `build_staggered_panel()` : 3 entités aux dates de début et de fin différentes, de 6 à 12 dates, valeurs calculables à la main. |
| `tests/legacy_failures.txt` | **B5** | Groupe « hors campagne » vidé. |
| `tests/ANOMALIES.md` | une section par préfixe : BASE (B1), CV (B2 puis B3), PANEL (B4 puis B5), XY (B6 puis B7) | Sections créées vides par B0. |
| `tests/integration/pipeline/` (nouveau) | **B8** | — |

## 2. Structure cible et seuil de 1 200 lignes

Estimation : symboles publics × familles de scénarios × environ 25 lignes par test, plus l'existant.

| Tests de | Actuel | Estimé | Décision |
|---|---:|---:|---|
| `base/transformers.py` | 816 | ~1 700 | **paquet `tests/unit/base/transformers/`**, créé par B1 (`git mv` de `test_transformers.py` vers `transformers/test_validation.py`) : `conftest.py` (fixtures, `ConcreteTransformer`, `SimpleScaler`), `test_time_index.py`, `test_structure.py`, `test_validation.py`, `test_sklearn_contract.py`, `test_reversible.py` |
| `crossvals/base.py` | 791 | ~1 600 | **paquet créé par B0** : `test_out_of_sample_split.py`, `test_in_sample_split.py`, `test_helpers.py` ; B2 ajoute `test_properties.py` (fuite, couverture, `get_n_splits`) et `test_index_robustness.py` |
| `crossvals/time_series.py` | 546 | ~850 | fichier |
| `crossvals/panel.py` | 725 | ~1 500 | **paquet `tests/unit/crossvals/panel/`**, créé par B3 : `test_extract_groups.py` (internal), `test_panel_split.py` (variantes globales), `test_per_entity_split.py`, `test_index_robustness.py` |
| `panel/utils.py` | 83 | ~1 000 | fichier ; si B4 dépasse 1 200 lignes → paquet `panel/utils/` (`test_entity_keys.py`, `test_structure.py`, `test_blocks.py`, `test_frequencies.py`) |
| `panel/transformers.py` | 495 | ~1 500 | **paquet `tests/unit/panel/transformers/`**, créé par B5 : `test_fit_transform.py`, `test_modes.py` (simple, fabrique, `entity_kwargs`), `test_error_handling.py`, `test_parallel.py`, `test_sklearn.py`, `test_structure.py`. Mettre à jour `legacy_failures.txt` en même temps |
| `xy/transformers.py` | 182 | ~600 | fichier |
| `xy/pipeline.py` | 715 | ~2 000 | **paquet créé par B0** : `conftest.py`, `test_core.py` (B6), `test_sklearn_surface.py` (B7). B6 ajoute `test_y_propagation.py` et `test_equivalence.py` ; B7 ajoute `test_introspection.py`, `test_delegation.py`, `test_routing.py`, `test_memory.py`, `test_model_selection.py`, `test_serialization.py`, `test_errors.py` (et peut y répartir `test_sklearn_surface.py`) |
| intégration crossvals | 477 | ~900 | fichier (B3) |
| intégration B8 | — | ~500 | `tests/integration/pipeline/test_panel_workflow.py` |

## 3. Inventaire : symbole → tests existants → trou

Indicateur de fragilité (accès à un symbole privé de `tsforecast` par fichier) :
`base/test_transformers.py` **22**, `crossvals/base/test_helpers.py` **35**, `crossvals/test_panel.py` **8**,
tous les autres **0**. Aucun test n'est marqué `@pytest.mark.internal`, alors que le marqueur existe
dans `pyproject.toml`.

### `base/transformers.py` (69 %)
| Symbole | Tests | Trou |
|---|---|---|
| `TimeSeriesTransformerMixin._validate_time_index` | `TestValidateTimeIndex` (16, appel direct) | testé seulement via le symbole privé ; voie publique = `fit` |
| `PanelTimeSeriesTransformer.__init__` (6 paramètres) | `TestParameterDefaults` (3 valeurs par défaut) | `get_params` / `set_params` / `clone` absents |
| `.fit` | colonnes (panel, série), MultiIndex à 2 et 3 niveaux, DatetimeIndex, strict / auto_sort, colonne de panel manquante, NaN au niveau entité | valeurs jamais vérifiées ; entrée `Series`, PeriodIndex, RangeIndex, doublons, jeu vide, une observation : aucun test |
| `.transform` / `fit_transform` | seulement via `fit_transform`, chemin nominal | NotFittedError, transform sur d'autres données que le fit, colonnes différentes |
| `is_panel_` | oui | — |
| `n_features_in_`, `feature_names_in_`, `conversion_metadata_` | aucun | tout |
| `ReversibleTransformerMixin` (`inverse_transform`, `_store_transformation_info`, `_restore_structure_if_converted`, `original_*_` / `transformed_*_`) | aucun (indirect via PanelwiseTransformer) | tout |
| `InverseTransformer` (`fit`, `transform`, `inverse_transform`) | aucun | tout |

### `crossvals/base.py` (84 %), `time_series.py` (100 %), `panel.py` (92 %)
| Symbole | Tests | Trou |
|---|---|---|
| `OutOfSampleSplit.__init__` / `.split` | 21 tests : série nominale, gap, `max_train_size` (`<=` faible), `test_indices` dates / entiers, panel avec `groups`, numpy, erreurs | `get_n_splits` jamais testé ; données non triées : seul l'avertissement est vérifié (fuite, ANO-CV-001) ; indices exacts expanding / rolling ; PeriodIndex ; doublons ; NaN |
| `InSampleSplit.__init__` / `.split` | 14 tests | `get_n_splits` ; « `test_size` ignoré quand plusieurs `test_indices` » épinglé sans documentation |
| `_resolve_test_positions`, `_precompute_group_mappings`, `_precompute_multiindex_mappings`, `_verify_and_sort_data` | `test_helpers.py` (27, uniquement en privé) | à marquer `internal` ou à remplacer par la voie publique |
| `TSOutOfSampleSplit.split`, `TSInSampleSplit.split` | 37 tests | `get_n_splits` ; `cross_validate` / `GridSearchCV` avec `cv=splitter` |
| `_extract_groups_from_panel` | 6 tests (privé) ; épingle qu'un index simple est accepté | index à 3 niveaux |
| `PanelOutOfSampleSplit` / `PanelInSampleSplit` / `…PerEntity` `.split` | 32 tests nominaux + erreurs | 3 niveaux, panel entrelacé ou non trié, types de clés, entité à une observation, `get_n_splits` |

### `panel/utils.py` (45 %)
Testés : `get_entity_levels` (6), `split_variable_key` (paramétré), `normalize_entity_mapping` (2),
`is_panel_data` et `iter_entity_blocks` (indirectement). **Aucun test** : `detect_panel_structure`,
`extract_time_series_from_multiindex`, `normalize_entity_key` (en direct), `get_unique_panel_entities`,
`group_keys_by_entity_and_variable`, `extract_column_names`, `get_entity_mask`,
`get_entity_target_frequency`, `resolve_entity_column_frequencies`, `build_panel_index`.
**Décision de l'auteur (2026-10-16)** : toute fonction de `panel/utils.py` sans préfixe `_` est
publique. Les quatre fonctions absentes de `tsforecast.panel.__all__` (`detect_panel_structure`,
`extract_time_series_from_multiindex`, `iter_entity_blocks`, `build_panel_index`) sont donc
publiques et se testent comme les autres (import depuis `tsforecast.panel.utils`) ; leur absence de
l'export est une anomalie [CODE] (ANO-PANEL-007).

### `panel/transformers.py` (58 %) — `PanelwiseTransformer`
| Symbole | Tests | Trou |
|---|---|---|
| `fit` / `transform` / `fit_transform` (chemin colonnes) | 9 tests, **tous en xfail** | chemin cassé (ANO-PANEL-001) |
| idem (chemin MultiIndex, `validate_input=True`) | 3 tests nominaux | 3 niveaux, une entité, désordre |
| `inverse_transform` | 1 (xfail) | tout |
| `get_params` / `set_params` / `clone` | 2 nominaux | GridSearchCV avec paramètre imbriqué, mode fabrique |
| `get_entity_transformer`, `n_entities_` | xfail | ANO-PANEL-003 |
| `transformers_`, `transform_columns_`, `entities_`, `failed_entities_`, `is_factory_mode_` | partiel | `failed_entities_`, `is_factory_mode_` |
| modes fabrique, `entity_kwargs` / `default_entity_kwargs` | aucun | tout |
| `n_jobs > 1`, `error_handling='ignore'`, `__repr__`, pickle | aucun | tout |

### `xy/transformers.py` (90 %)
`XYTransformerMixin` (fit / transform / inverse_transform / fit_transform, plus les `_…` abstraits)
n'est testé qu'à travers `DropNaNTarget`. `XYPanelTimeSeriesTransformer` : aucun test direct
(seulement via les tests de HighFrequencyImputer) ; absent de `tsforecast.xy` et de CLAUDE.md.
`DropNaNTarget` : 17 tests de bonne qualité (valeurs d'or) ; manquent : panel MultiIndex, NaN de `X`
conservés, `y` ndarray 2-D, alignement des index dans le pipeline.

### `xy/pipeline.py` (70 %) — `XYPipeline`
Testés (forme seulement) : `fit`, `predict`, `score`, `transform`, `fit_transform`, `inverse_transform`
(avec y), `predict_proba`, `decision_function`, `classes_`, NotFittedError (5 méthodes), verbose, memory,
GridSearchCV, `cross_val_score`, `preserve_dataframe`, passthrough / None, slicing, get/set_params,
`transform_input`. **Aucun test** : `predict_log_proba`, `fit_predict`, `score_samples`,
`inverse_transform` sans y, metadata routing, `sample_weight`, `n_features_in_`,
`feature_names_in_`, `get_feature_names_out`, sérialisation, `n_jobs`, alignement des index
(ANO-XY-001 et 002), équivalence avec `sklearn.pipeline.Pipeline`.

## 4. Pré-tri des tests existants (verts douteux)

**base** : 5 tests sans assertion (ils vérifient seulement que rien n'est levé) :
`test_strict_validation_sorted_time_series_passes`, `test_auto_sort_fixes_unsorted_data`,
`test_sorted_panel_data_passes`, `test_auto_sort_with_panel_data`, `test_validation_can_be_disabled`.
Assertions faibles (`isinstance` / `len`, aucune valeur) : `test_column_based_*_backward_compatibility`,
`test_multiindex_with_unnamed_levels`. Tests qui vérifient l'**entrée** au lieu de la sortie :
`test_auto_sort_preserves_sorted_data`, `test_auto_sort_only_sorts_not_modifies_values` (comparaison
d'ensembles), `test_validation_preserves_data_when_valid`. `pytest.warns` sans `match` :
`test_permissive_validation_unsorted_warns`. Fixtures **non déterministes** (`np.random.randn` sans
graine, `sample(frac=1)` sans `random_state`). Doublon : `test_multiindex_panel_fit_metadata` ≈
`test_multiindex_panel_basic`. Bloc `if __name__ == "__main__"` au milieu du fichier. Commentaires
historiques (« used to fail ») : règle 2.

**crossvals** : assertions permissives : `test_gap_larger_than_data` (`len == 0 or < 3`),
`test_timeseries_with_max_train_size` (`<= 20`), `test_max_train_size_smaller_than_test` (`>=`),
`test_per_entity_single_entity` (`>= 1`). `test_panel_split_specific_test_dates` est
quasi tautologique (`if x in test_idx: assert x in entity_test`). Les gardes
`if len(entity_train) > 0 and len(entity_test) > 0` peuvent rendre des boucles vides.
`test_unsorted_data_warning` ne vérifie que l'avertissement et masque ANO-CV-001. Noms trompeurs :
`test_mixed_index_types_error` et `test_no_multiindex_error` (vérifient un succès),
`test_missing_entity_in_test_period` (la date manque pour toutes les entités).
`pytest.raises(ValueError)` sans `match` : `test_empty_dataset` / `test_empty_series`. Intégration :
seuils de temps (`< 1.0 s`, instables), alias pandas `'H'`, `'M'`, `'Y'` (FutureWarning, B2),
`pytest.raises` sans `match`. Tests `init_*` dupliqués dans 6 classes : à paramétrer.

**panel** : `test_unknown_entity_warning` est **vert à tort** : avec `'warn'`, le fit échoue pour
chaque entité (ANO-PANEL-001), `transformers_` reste vide, et le transform avertit « not seen » pour
toutes les entités. `test_gridsearchcv_compatible` n'utilise pas GridSearchCV.
`test_fit_transform_with_validation` (MultiIndex + `len`) et `test_scaling_with_multiindex`
(moyenne seulement) sont faibles.

**xy** : assertions de forme seulement (`test_predict`, `test_transform`, `test_multiple_xy_transformers`,
qui ne regarde pas `y`). `test_score` : `isinstance float`. `test_no_cloning_when_caching_disabled`
est vide de sens : il vérifie `mean_ is not None` au lieu de `named_steps['scaler'] is original_scaler`.
`test_preserve_series_with_xy_transformer` passe des ndarray et ne vérifie qu'un tuple, contrairement
à son titre. `test_memory_caching_*` ne vérifie pas la mise en cache. `test_gridsearch_*` :
`best_params_ is not None`. `xy/test_transformers.py` est de bonne qualité (plusieurs assertions
reliées par `and` : à scinder au besoin).

### Les 14 xfail hérités (`tests/unit/panel/test_transformers.py`, tri définitif en B5)
Cause commune observée avec `--runxfail` : sur un panel en **colonnes** (`validate_input=False`, index
simple), `X.loc[group_idx, cols].xs(entity_key)` lève `KeyError(('DE',))`. Ce `.xs` a été introduit
partout par le commit `33cec68` « (fix) panelwise transformer ». `_fit` garde pourtant une branche
colonnes (`exclude_cols`) et la docstring annonce « panel_cols as columns or MultiIndex » : le mode
colonnes est voulu, ce n'est donc pas un test obsolète (a).

| Test | Symptôme | Hypothèse |
|---|---|---|
| `TestPanelwiseTransformerBasic::test_fit_transform_no_validation` | KeyError ('DE',) | (b) ANO-PANEL-001 |
| `…::test_per_entity_scaling` | idem | (b) 001 |
| `…::test_fit_and_transform_separately` | idem | (b) 001 |
| `…::test_inverse_transform` | idem (même `.xs` dans `inverse_transform`) | (b) 001 |
| `TestSklearnCompatibility::test_in_pipeline` | idem | (b) 001 |
| `…::test_nested_pipeline` | idem | (b) 001 |
| `…::test_gridsearchcv_compatible` | idem | (b) 001 + (c) le test n'utilise pas GridSearchCV : à réécrire |
| `TestMultiLevelPanel::test_multi_level_fit_transform` | KeyError ('DE','finance') | (b) 001 |
| `…::test_get_entity_transformer` | idem ; masque ensuite `AttributeError: _normalize_entity_key` | (b) 001 + (b) ANO-PANEL-003 |
| `TestErrorHandling::test_missing_panel_cols_error` | `TypeError: 'NoneType' object is not iterable` (`set(None)`) au lieu d'une ValueError explicite | (b) ANO-PANEL-005 (mineure) |
| `…::test_error_handling_raise` | KeyError au fit ; une fois corrigé, le transform lèverait `KeyError` et non la `ValueError` attendue | (b) 001 + (d) type d'exception, ANO-PANEL-006 |
| `TestStructurePreservation::test_index_preservation` | KeyError | (b) 001 |
| `…::test_column_order_preservation` | idem | (b) 001 |
| `…::test_row_order_preservation` | idem | (b) 001 |

## 5. Anomalies pressenties (ids indicatifs ; numérotation définitive par le prompt propriétaire)

Celles marquées ✔ ont été reproduites par B0.

**BASE (B1)**
- ANO-BASE-001 [CODE] majeure ✔ : `InverseTransformer` est inutilisable après `clone` : `fit` ne fait
  rien et le transformateur enveloppé, cloné, n'est pas ajusté → NotFittedError dans
  Pipeline / cross_validate.
- ANO-BASE-002 [CODE] majeure ✔ : `_validate_index_time_series` fait `data.index = pd.to_datetime(...)`
  sur l'objet de l'appelant : l'entrée est **mutée** (index chaîne → datetime64 après `fit`).
- ANO-BASE-003 [CODE] mineure ✔ : avec `convert_cols_to_index=True`, `transform` réécrit l'attribut
  ajusté `conversion_metadata_` ; `inverse_transform` dépend donc du dernier transform.
- ANO-BASE-004 à arbitrer : `is_panel_` vaut True pour tout MultiIndex d'au moins 2 niveaux et pour
  toute configuration `time_col` + `panel_cols`, même avec une seule entité ; après
  `convert_cols_to_index`, `_panel_in_index` reste False.
- ANO-BASE-005 [DOC] : la docstring de classe annonce `time_col` « Default is 'date' » alors que
  `__init__` vaut None ; les exemples appellent `MyTransformer`, `ConcreteTransformer` et
  `MyReversibleTransformer`, qui n'existent pas (recoupe N7).
- ANO-BASE-006 à vérifier : `transform` ne contrôle jamais `n_features_in_` / `feature_names_in_`.
- À l'origine de « Cannot determine time index » (vu en partie D) : labels numériques refusés par
  `_validate_time_index`. Le cas MultiIndex / PeriodIndex est déjà couvert par `TestValidateTimeIndex` ;
  B1 rattache le reste à l'anomalie U5 correspondante s'il y en a une.

**CV (B2, B3)**
- ANO-CV-001 [CODE] **bloquante** ✔ : sur des données **non triées**, `_split` renvoie les indices avec
  la permutation **inverse** (`np.argsort(sort_mapping)` au lieu de `sort_mapping`). Exemple : dates
  [3, 1, 5, 2, 4] → pli « test 01-01, apprentissage 01-03, 01-04, 01-05 » : **fuite de données**.
  Même code dans `OutOfSampleSplit._split` et `InSampleSplit._split` : toutes les variantes sont touchées.
- ANO-CV-002 [CODE] majeure ✔ : `get_n_splits` (hérité de `_BaseKFold`) renvoie `n_splits` et non le
  nombre réel de plis : `test_indices` (1 date → 5 annoncés, 1 produit), InSample (5 annoncés, 1
  produit), PerEntity (plis × entités).
- ANO-CV-003 à arbitrer : quand `gap` ≥ les données disponibles, des plis à apprentissage **vide** sont
  produits sans erreur (épinglé par `test_gap_larger_than_data` et `test_gap_leaves_no_training_data`).
- ANO-CV-004 à vérifier (B3) : `_extract_groups_from_panel` ne lit que le niveau 0 ; avec 3 niveaux,
  les entités qui partagent le premier niveau sont fusionnées.
- ANO-CV-005 [DOC] : les docstrings de `PanelOutOfSampleSplitPerEntity` et
  `PanelInSampleSplitPerEntity` annoncent « returns a dictionary mapping each entity », alors que les
  splitters produisent des tuples `(train, test)` par entité.
- ANO-CV-006 à arbitrer : sur un panel déséquilibré, une entité sans date de test reçoit un
  avertissement puis lève « Cannot find test indices for the group » (épinglé), alors que PerEntity
  annonce « Handling entities with different data availability ».
- ANO-CV-007 [DOC] / à arbitrer : `InSampleSplit` ignore `test_size` quand plusieurs `test_indices`
  sont donnés.

**PANEL (B4, B5)**
- ANO-PANEL-001 [CODE] **bloquante** ✔ : panel en colonnes cassé (`.xs(entity_key)` sur un index
  simple), origine `33cec68` ; touche fit, transform et inverse_transform.
- ANO-PANEL-002 [CODE] mineure ✔ : le message « Error fitting entity ('DE',): ('DE',). Skipping. »
  répète la clé (`str(KeyError)`) au lieu du type et de la cause.
- ANO-PANEL-003 [CODE] majeure ✔ : `get_entity_transformer` appelle `self._normalize_entity_key`, qui
  n'existe pas → AttributeError systématique.
- ANO-PANEL-004 à vérifier : `_fit_parallel` (`n_jobs != 1`) ne gère aucune erreur : `error_handling`
  est ignoré et `failed_entities_` reste vide (`n_jobs=1` et `n_jobs=2` donnent bien le même résultat
  sur un panel MultiIndex sain ✔).
- ANO-PANEL-005 [CODE] mineure : `panel_cols=None` sur un index simple → TypeError interne au lieu
  d'une ValueError explicite.
- ANO-PANEL-006 à arbitrer : entité inconnue avec `error_handling='raise'` → `KeyError` (le test attend
  `ValueError`).
- ANO-PANEL-007 à vérifier (B4) : `extract_time_series_from_multiindex` perd le nom de la Series et
  produit des dates en double sur plusieurs entités ; `normalize_entity_key(['FR'])` donne une clé
  non hachable. [CODE] mineure, à consigner : 4 fonctions publiques (sans préfixe `_`, décision de
  l'auteur) ne sont pas exportées par `tsforecast.panel` (test : chacune figure dans
  `tsforecast.panel.__all__`, xfail strict).
- ANO-PANEL-008 [DOC] : les exemples de la classe utilisent `df` et `PublicationDelayTransformer` sans
  les définir ; `__repr__` surchargé contourne le repr de sklearn.

**XY (B6, B7)**
- ANO-XY-001 [CODE] majeure ✔ : `transform` / `predict` réappliquent l'**index et les colonnes du fit**
  (métadonnées enregistrées seulement dans `_fit`) : `transform(X_test)` d'index 100..105 sort avec
  l'index 0..5 ; si les longueurs diffèrent, c'est un RangeIndex.
- ANO-XY-002 [CODE] majeure ✔ : une étape qui renvoie un ndarray (StandardScaler) suivie de
  `DropNaNTarget` donne un X à index positionnel [0..4] et un y d'index [0, 2, 3, 4, 5] : **X et y
  désalignés**.
- ANO-XY-003 à arbitrer : une étape est reconnue XY par sa signature (`y` positionnel), et tout
  résultat qui est un tuple de longueur 2 est lu comme (X, y). Or `TransformerMixin.fit_transform`
  accepte toujours `y` : la reconnaissance ne repose que sur la forme du résultat.
- ANO-XY-004 [DOC] : l'exemple de la docstring de classe utilise `X`, `y`, `np` et `BaseEstimator`
  sans les définir ; `XYPanelTimeSeriesTransformer` n'est pas exporté.

## 6. Scénarios attendus par prompt

Pour chaque composant, les cas limites de CLAUDE.md s'appliquent : non trié, doublons, entité
manquante, fréquence irrégulière, NaN (y compris colonne ou entité entièrement NaN), jeu vide, une
observation. Robustesse d'index : DatetimeIndex, RangeIndex, PeriodIndex, MultiIndex à 2 et 3 niveaux,
noms d'index non standards, positions de début et de fin de période, noms de colonnes spéciaux. Au
moins un test sur `irregular_index_timeseries` / `heterogeneous_coverage_panel`. Messages d'erreur
vérifiés par `match=`. Non-mutation de l'entrée (copie profonde avant, comparaison après).

### B1 — `base/transformers.py` → paquet `tests/unit/base/transformers/`
1. Tri §4.4 des 53 tests (section 4 ci-dessus) ; fixtures rendues déterministes.
2. Index temporel par la voie publique `fit` : index seul, `time_col`, MultiIndex 2 et 3 niveaux (via
   `to_three_level_index`), noms non standards, index non temporel (labels numériques → message),
   doublons, non trié, PeriodIndex, chaînes de dates ; garder `TestValidateTimeIndex` en `internal`.
3. Structure : détection série / panel, colonnes / index, `convert_cols_to_index` (aller-retour avec
   `restore_original_structure`), NaN dans l'identifiant, MultiIndex à 1 niveau.
4. Contrat : `_fit` / `_transform` non implémentés → TypeError à l'instanciation ; NotFittedError
   avant `fit` ; `is_panel_`, `n_features_in_`, `feature_names_in_` (valeurs d'or) ; get_params /
   set_params / clone (paramètres stockés tels quels) ; entrée `Series`.
5. `ReversibleTransformerMixin` + `InverseTransformer` : transform → inverse = identité sur chaque
   perturbation ; inverse avant fit ; échange transform / inverse ; clone (ANO-BASE-001).
6. Non-mutation (ANO-BASE-002), état modifié au transform (ANO-BASE-003).

### B2 — `crossvals/base.py`, `time_series.py` ; propriétaire de `support/splits.py`
1. Valeurs d'or écrites à la main sur 12 à 24 dates : expanding, rolling (`max_train_size`), `gap`,
   `test_size`, `n_splits`, `test_indices` (dates, positions, Timestamp, chaînes).
2. Propriétés sur tous les plis : aucune fuite (dates), `gap` exact, tests disjoints et successifs,
   `get_n_splits == len(list(split))` pour toutes les configurations (ANO-CV-002), générateur
   reproductible, `y=None` / `groups=None`, X en Series / DataFrame / ndarray.
3. Données non triées : indices renvoyés comparés aux **dates** (ANO-CV-001 : xfail strict) ; doublons
   de dates ; une observation ; jeu vide ; `gap` ou `test_size` > données (ANO-CV-003) ; paramètres
   invalides ; DatetimeIndex / RangeIndex / PeriodIndex ; NaN dans X.
4. sklearn : `cross_validate`, `cross_val_score`, `GridSearchCV` (`cv=splitter`, modèle trivial).
5. Helpers privés : marquer `internal` ce qui n'a pas de voie publique, retirer le reste au profit de
   la voie publique. Alias `'H'`, `'M'`, `'Y'` → `'h'`, `'ME'`, `'YE'` dans les tests d'intégration.
   Justifier chaque ligne de `base.py` restée non couverte.

### B3 — `crossvals/panel.py` + intégration → paquet `tests/unit/crossvals/panel/`
1. Contrat global contre PerEntity, sur `build_staggered_panel` (valeurs d'or) puis
   `heterogeneous_coverage_panel` (propriétés).
2. Entité absente d'un pli (`truncate_entity`), entité à une observation, entité plus courte que
   `gap` + `test_size`, panel déséquilibré (ANO-CV-006), MultiIndex à 3 niveaux (ANO-CV-004), niveaux
   nommés autrement, panel entrelacé (`interleave_entities`) ou trié par entité mais pas par date,
   `groups` explicite contre déduit, clés tuples / entiers / avec espaces (`rename_entities`).
3. Aucune fuite par entité ; `get_n_splits` cohérent (PerEntity × entités) ; docstrings (ANO-CV-005).
4. Intégration : splitters de panel + `cross_validate` + XYPipeline + PanelwiseTransformer (chemin
   MultiIndex, ANO-PANEL-001 en tête) + estimateur trivial ; `GridSearchCV` ;
   `irregular_index_timeseries` + `TSOutOfSampleSplit`. Seuils de temps → marqueur `slow` ou
   suppression justifiée.

### B4 — `panel/utils.py` ; propriétaire de `support/perturbations.py`
Une classe par fonction (15), valeurs d'or écrites à la main : index simple, MultiIndex 2 et 3
niveaux, niveaux nommés ou non, clé scalaire ≡ tuple dans les deux sens, clés avec espaces / accents /
types mélangés, colonnes MultiIndex, jeu vide, une entité, une colonne. Mappings : par entité, par
(entité, colonne), notations mélangées, clés en conflit après normalisation (à épingler). Aller-retour
`iter_entity_blocks` → `build_panel_index` (ordre et index), panel désordonné, entité absente.
Fréquences : globale, par entité, par colonne, par couple, ordre de précédence. Écarts docstring ↔
code → [DOC] ; ANO-PANEL-007.

### B5 — `panel/transformers.py` → paquet `tests/unit/panel/transformers/`
0. Ajout de B1 (décision de l'auteur, 2026-10-16) : corriger **ANO-BASE-010** (`convert_cols_to_index=False`
   garde `time_col` / `panel_cols` en colonnes, `tsforecast/base/transformers.py::_validate_column_based`) **en même
   temps qu'ANO-PANEL-001** : sans cela, un panel en colonnes validé passe par le chemin colonnes cassé de
   `PanelwiseTransformer`. Voir l'entrée ANO-BASE-010 de `ANOMALIES.md` pour les effets attendus. Depuis B1, la base
   refuse aussi toute entrée qui n'est pas un `DataFrame` et contrôle les colonnes au `transform` (mêmes noms, même
   ordre).
1. Tri des 14 xfail (tableau de la section 4) ; `legacy_failures.txt`, groupe « hors campagne »,
   vidé ; `test_unknown_entity_warning` retrié (vert à tort).
2. Chemins colonnes **et** MultiIndex (2 et 3 niveaux), une entité, entité inconnue au transform
   (`raise` / `warn` / `ignore`), modes simple / fabrique / `entity_kwargs` (+ défaut), `n_jobs` 1
   contre 2 (identique, ordre conservé ; ANO-PANEL-004), entités désordonnées, `FailingTransformer`
   sur une entité, `transformers_` / `failed_entities_` / `get_entity_transformer`
   (ANO-PANEL-003), get/set_params, clone, GridSearchCV avec paramètre imbriqué, aller-retour pickle
   et joblib, transformateur qui change le nombre de colonnes (PCA → `get_feature_names_out` / noms
   `feature_i`).
3. Non-mutation, index / colonnes / dtypes conservés ; `heterogeneous_coverage_panel` : transformer
   FR+DE == concaténation de FR seul et de DE seul.

### B6 — `xy/transformers.py` + cœur de `XYPipeline` ; propriétaire de `support/estimators.py`
Matrice (une dimension par classe, `parametrize` avec ids) : composition (tout XY, XY + sklearn,
passthrough / None, avec ou sans estimateur final, XY en dernier, une seule étape, pipeline vide) ×
`y` (Series, DataFrame multi-cibles, ndarray, None, index différent de X, lignes retirées, y
entièrement NaN) × méthode (fit, fit_transform, transform, predict, fit_predict, score). Les (X, y)
reçus par chaque étape sont vérifiés par `SpyTransformer`. **Alignement par l'index** après retrait
de lignes (ANO-XY-002) et index de sortie de transform / predict (ANO-XY-001). DropNaNTarget :
identité, tout NaN, NaN partiels, NaN de X conservés, panel MultiIndex, y DataFrame. XYTransformerMixin :
sous-classe minimale, méthode non implémentée, get_params / clone. `XYPanelTimeSeriesTransformer`.
Équivalence avec `sklearn.pipeline.Pipeline` sans étape XY (paramétré). Couverture :
`xy/transformers.py` ≥ 95 %, `pipeline.py` ≥ 80 %.

### B7 — surface sklearn de `XYPipeline` (fichiers listés en section 2)
Introspection, méthodes déléguées et `available_if`, routing (fixture qui restaure `set_config`),
`memory=` (le second fit ne recalcule pas les étapes), GridSearchCV / RandomizedSearchCV /
cross_validate / cross_val_predict avec `TSOutOfSampleSplit` et une étape XY qui retire des lignes,
`n_jobs=2`, joblib.dump / load, erreurs (nom dupliqué, étape sans fit / transform). Comparer avec
`Pipeline` ; ANO-XY-003. Couverture de `pipeline.py` : 95 % visés, 90 % minimum.

### B8 — intégration `tests/integration/pipeline/` et clôture
Flux complet sur `heterogeneous_coverage_panel` et `irregular_index_timeseries` : XYPipeline
[PanelwiseTransformer(PublicationDelayTransformer) → DropNaNTarget → estimateur] + `cross_validate`
+ splitter de panel ; scores finis ; aucune fuite entre plis ni entre entités ; non-mutation ;
reproductibilité ; `n_jobs` 1 contre 2. Clôture selon le prompt : tableau de couverture,
correspondance xfail ↔ ANOMALIES.md (script), aucun fichier > 1 200 lignes, aucun test > 2 s sans
`slow`, mise à jour du §2.2.
