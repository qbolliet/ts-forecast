# Audit des dtypes `object`

Audit des tables (sorties publiques **et** tables internes) qui utilisaient encore le dtype
`object` après le passage aux dtypes `str` et catégoriels, pistes de remplacement, modifications
implémentées et cas laissés en l'état.

> Date : 2026-10-23 · branche `qb-mixed-frequencies` · pandas 3.0.6, scikit-learn 1.9

---

## 1. Méthode

Un `grep` de `dtype=object` ne trouve que les cas **explicites**. La plupart des tableaux `object`
naissent implicitement : `apply(axis=1)` (chaque ligne devient une Series `object`), listes de
tuples, colonnes mêlant `None` et nombres, itération ou `unique()` d'un `MultiIndex` (matérialisation
de tuples), `groupby` sur des clés tuples. Trois mesures complémentaires :

1. **Instrumentation de pandas pendant toute la suite de tests** (plugin pytest, ~14 000 tests).
   Interception de `BlockManager.__init__`, `SingleBlockManager.__init__`, `Index._simple_new` et de
   la matérialisation `MultiIndex._values` ; chaque bloc / Index / tableau de tuples de dtype
   `object` est attribué à la ligne la plus interne de `tsforecast/` de la pile d'appel, avec le
   nombre de créations, les types Python rencontrés et la taille maximale.
2. **Scan des sorties publiques** : sorties de `fit_transform` / `transform`, attributs ajustés,
   rapports (`to_frame`, `to_diagnostic_frame`) de `HighFrequencyImputer`,
   `PublicationDelayTransformer`, `compare_and_detect_delays`, parcourus récursivement.
3. **Profilage** (`cProfile`) des chemins coûteux, pour ne retenir que les remplacements qui
   changent réellement quelque chose.

Les créations sont classées en **bruit interne de pandas** (Series `df.dtypes`, Index de noms de
niveaux, Index `[None]` d'un `concat` de Series sans nom, Index vides d'échafaudage) et en
**créations réelles**.

## 2. Résultats de l'audit (état initial)

318 830 créations `object` réelles sur la suite de tests (hors 113 168 de bruit), dont 279 669
Series, 26 207 Index, 11 528 matérialisations de tuples de `MultiIndex`.

| Catégorie | Sites principaux | Volume | Coût mesuré |
|---|---|---|---|
| **`apply(axis=1)` ligne à ligne** | `delays/calculator.py` : `_convert_to_target_frequency_and_reference`, `_convert_delay_unit` | ≈ 22 500 Series (une par ligne, valeurs Timestamp/bool/float/str mêlées) | **91 %** du temps de `calculate_applicable_delay` (7,8 s pour 7 200 lignes) |
| **Clés composites en tuples** | `aggregation_constraint.py` (`_period_membership` : Series de `Period` / `(entité, Period)`), `high_frequency_imputer.py` (`_aggregated_cells`, `_row_layers`, `_stage_frequency_of`), `stage_scaler.py` (`_entities_of`), `training_set_builder.py` (`split_training_index`), `utils/validation/utils.py`, `panel/utils.py` (`get_unique_panel_entities`, `iter_entity_blocks`), `crossvals` | ≈ 60 000 créations | `_rescale_one` : 22 % d'un `fit_transform` HFI ; `iter_entity_blocks` : 7,9 s sur 500 entités |
| **Vocabulaire fermé en `object`** | `covariate_materializer.py` : origines de cellules (`observed` / `interpolated` / `model`) et fréquences de production construites en `object` puis converties en `str` | ≈ 225 000 Series | ≈ 4 % du `fit_transform` HFI |
| **Sorties publiques `object`** | `DelayFitReport.to_frame()` (`n_obs` / `n_periods` tout `None`), `ImputationPlan.to_diagnostic_frame()` (`interpolation_anchor` ; toutes les colonnes numériques d'un plan vide), `FrequencyAligner` (booléens mêlés de NaN) | — | cohérence des dtypes |
| **Petits cas internes** | `training_set_builder.py` (`reindex(...).eq(True)`, `to_numpy(dtype=object)`, échafaudage vide), `high_frequency_imputer.py` (libellés d'étape), `delays/transformers.py` (libellés forcés en `object`), `provenance.py` (colonnes vides) | quelques milliers | négligeable |
| **Légitimes** | `feature_names_in_`, groupes de `crossvals`, kwargs de `delays/transformers.py` | — | — |

## 3. Pistes et modifications implémentées

### 3.1 `calculate_applicable_delay` : calcul par combinaison distincte

- **Constat.** Un `apply(axis=1)` transforme chaque ligne en Series `object`, puis reconstruit le
  DataFrame. Une vectorisation pure en entiers nanosecondes aurait changé le comportement : le calcul
  scalaire garde l'unité des dates (une date en `us` au-delà de 2262 reste valide), et les délais
  flottants sont convertis exactement via `Fraction`.
- **Implémentation.** La conversion (`_calculate_converted_delay`, désormais à arguments scalaires)
  est appelée **une fois par combinaison distincte** des colonnes qu'elle lit (date d'observation,
  bornes, point de référence, délai, unité, fréquences source et cible), puis diffusée sur les lignes.
  Les entités et indicateurs d'une même date partagent leur combinaison : 7 200 lignes → 12
  conversions dans le banc d'essai. Même principe pour `_convert_delay_unit` (une conversion par
  couple (délai, unité)), pour la normalisation des fréquences (`_map_distinct`) et pour
  `_resolve_target_frequencies` (une résolution par clé d'index distincte).
- **Non-régression.** Comparaison **bit à bit** avant/après sur 2 040 appels (fréquences M/Q/W/D/h,
  unités jour/us/heure, délais flottants et NaN, fréquences inconnues, dates en `s`/`ns`, panels,
  agrégations), erreurs comprises (type et message) : **0 écart**. L'ordre des erreurs est conservé
  (combinaisons visitées dans l'ordre de première apparition).
- **Gain.** 7,82 s → 0,024 s (**×330**) ; avec conversion d'unité 10,4 s → 0,025 s (**×420**).

### 3.2 Contrainte d'agrégation : blocs codés en entiers

- **Constat.** `_period_membership` construisait une Series `object` de `Period` (série) ou de
  tuples `(entité, Period)` (panel), parcourue ensuite en Python avec des opérations pandas par
  période (`iloc`, `notna`, deux `__setitem__`, `period_range`).
- **Implémentation.** `_block_key_components` renvoie les niveaux d'entité et un `PeriodIndex` ; les
  lignes prédites et les ancres sont codées ensemble par `factorize_row_keys` (codes entiers
  (entité × ordinal de période)), regroupées par un tri stable (`_positions_by_code`), et le recalage
  travaille sur des tranches numpy avec **une seule écriture finale**. Les clés `Period` /
  `(entité, Period)` ne sont construites que pour les périodes signalées dans les avertissements,
  dont le texte est inchangé. Le décompte calendaire des sous-périodes est mis en cache par
  (fréquence, période).
- **Non-régression.** Mêmes `np.sum` sur les mêmes valeurs dans le même ordre : résultats
  identiques au bit près. 16 configurations HFI (4 jeux × 4 réglages) et 6 cas dégénérés (zéro,
  inversion, troncature) comparés, avertissements compris : **0 écart**.
- **Gain.** `rescale` sur 48 000 lignes : 2,82 s → 0,128 s (**×22**).

### 3.3 Codes entiers d'entités et de clés composites (nouveaux utilitaires)

- `panel/utils.py` : **`factorize_row_keys(*keys)`** (codes de clés composites dans l'ordre de
  première apparition, sans tuple par ligne, valeurs manquantes traitées comme des valeurs) et
  **`get_entity_codes(index)`** (codes lus sur les codes des niveaux d'entité du `MultiIndex`),
  exportés par `tsforecast.panel`.
- Utilisés par :
  - `get_unique_panel_entities` et `iter_entity_blocks` (masque d'entité = comparaison de codes, au
    lieu d'une relecture des niveaux par entité) ;
  - `StageScaler` (`_entities_of` rend `(codes, entités)` ; modes, ventilation des diviseurs et
    fréquences d'étape résolus **une fois par entité** puis diffusés) ;
  - `HighFrequencyImputer` (`_row_layers` sur codes (entité × fréquence), `_stage_frequency_of`
    rend une Series `str` normalisée une fois par entité, `_aggregated_cells` sans Series de tuples) ;
  - `split_training_index`, qui rend désormais `(real_index, entity_codes, entities)` ;
  - `validate_entities_grouped` (nombre de séquences = nombre d'entités) et
    `validate_sorted_within_groups` (tri stable par code, sans index de tuples ni `np.asarray` des
    dates) ;
  - `crossvals` : `_precompute_group_mappings` en un seul tri stable (au lieu d'une comparaison
    par groupe), `_check_groups` par codes de première apparition, `_split_global_folds` par codes.
- **Gains.** `iter_entity_blocks` (500 entités, 120 000 lignes) : 7,87 s → 0,15 s (**×52**) ;
  `validate_entities_grouped` ×19 ; `validate_sorted_within_groups` ×9.

### 3.4 Origines de cellules : catégoriel ordonné

- **Implémentation.** `provenance.py` définit **`CELL_ORIGIN_DTYPE`**, catégoriel **ordonné**
  `observed < interpolated < model` (ordre croissant de souillure), et **`max_cell_origin`**,
  version vectorisée de `max_origin` sur une Series (maximum des codes). Les deux sont exportés par
  `tsforecast.frequency`.
- Le matérialiseur produit les origines sous forme de **codes entiers** (`Categorical.from_codes`)
  et les fréquences de production sous forme de codes convertis en `str`, sans tableau `object`
  intermédiaire. `origin_store`, le miroir et `TrainingSet.row_origin` sont en `CELL_ORIGIN_DTYPE` ;
  les lignes candidates du jeu d'entraînement aussi, si bien que la concaténation lignes observées +
  miroir ne repasse jamais par `object`. Les catégories fixes gardent le dtype à travers les
  `combine_first` successifs.
- **Non-régression.** Sorties HFI comparées : **0 écart**.

### 3.5 Sorties publiques typées

| Sortie | Avant | Après |
|---|---|---|
| `DelayFitReport.to_frame()` | `n_obs` / `n_periods` en `object` quand tout `None` ; `delay` en `int64` ou `object` selon le contenu ; tout `object` si vide | `delay` `float64`, `n_periods` / `n_obs` **`Int64`** (entier nullable, manquant pour l'autre stratégie), `moved_from_mask` `bool`, quel que soit le contenu |
| `ImputationPlan.to_diagnostic_frame()` | `interpolation_anchor` en `object` (`None`) ; plan vide : toutes les colonnes numériques et booléennes en `object` | `interpolation_anchor` `float64` (NaN), décomptes `int64`, drapeaux `bool`, facteurs `float64`, plan vide compris |
| `FrequencyAligner.convert_to_target(..., agg_method='all'/'any')` | colonne `object` (True / False / NaN) | **`boolean`** nullable (True / False / `<NA>`) |
| `ImputationProvenanceTracker.provenance_matrix_` sans colonne suivie | colonnes `object` vides | colonnes au dtype des libellés d'entrée (`str`) |
| `CovariateMaterializer.origin_store`, `TrainingSet.row_origin` | `str` | `CELL_ORIGIN_DTYPE` |

### 3.6 Petits cas internes

- `training_set_builder.py` : `reindex(rows.index, fill_value=False)` (plus de Series bool + NaN en
  `object`) ; `rows['freq'].dropna().unique()` au lieu de `to_numpy(dtype=object)` ; métadonnées
  `row_frequency` / `row_origin` reprises par leurs tableaux pandas (`.array`) ; jeu de lignes vide
  typé (`_empty_rows`).
- `high_frequency_imputer.py` : niveau de fréquence de la sortie empilée construit par
  `pd.array(..., dtype='str').take(...)` au lieu de tableaux `object` répétés.
- `covariate_materializer.py` : découpage d'une grille par entité sur une Series typée
  (`_grid_scaffold`) au lieu d'un DataFrame vide aux colonnes `object`.
- `delays/transformers.py` : la colonne `column` de la table de délais garde le dtype inféré des
  libellés (`variables.array`) au lieu d'être forcée en `object`.
- `provenance.py` : colonnes suivies conservées en `Index` au dtype de l'entrée.

## 4. Bilan chiffré

### Audit avant / après (créations `object` réelles sur la suite de tests)

| | Avant | Après | Variation |
|---|---:|---:|---:|
| Total réel | 318 830 | 27 225 | **−91 %** |
| Series `object` | 279 669 | 10 886 | −96 % |
| Matérialisations de tuples de `MultiIndex` | 11 528 | 2 796 | −76 % |
| Bruit interne pandas | 113 168 | 73 274 | −35 % |

| Fichier | Avant | Après |
|---|---:|---:|
| `frequency/covariate_materializer.py` | 225 230 | 2 900 |
| `frequency/aggregation_constraint.py` | 26 791 | 31 |
| `delays/calculator.py` | 22 507 | 2 344 |
| `frequency/stage_scaler.py` | 18 575 | 5 727 |
| `panel/utils.py` | 4 923 | 2 831 |
| `frequency/training_set_builder.py` | 4 552 | 0 |
| `frequency/high_frequency_imputer.py` | 2 520 | 1 404 |
| `utils/validation/utils.py` | 1 241 | 167 |
| `frequency/imputation_plan.py` | 1 234 | 669 |

### Temps d'exécution (meilleur de plusieurs passes, même machine, `HEAD` vs arbre modifié)

| Banc d'essai | Avant | Après | Facteur |
|---|---:|---:|---:|
| `calculate_applicable_delay` (7 200 lignes) | 7,820 s | 0,024 s | ×330 |
| `calculate_applicable_delay`, `unit='hour'` | 10,399 s | 0,025 s | ×420 |
| `AggregationConstraint.rescale` (48 000 lignes) | 2,822 s | 0,128 s | ×22 |
| `HighFrequencyImputer.fit_transform` (panel de 20 entités) | 6,679 s | 4,555 s | ×1,5 |
| `validate_entities_grouped` (120 000 lignes, 3 niveaux) | 0,121 s | 0,006 s | ×19 |
| `validate_sorted_within_groups` (idem) | 0,164 s | 0,018 s | ×9 |
| `iter_entity_blocks` (500 entités) | 7,867 s | 0,152 s | ×52 |

Le reste du temps de `fit_transform` HFI est dominé par l'interpolation et l'agrégation
(`_interpolate_block`, `aggregate_to_lower_frequency`), sans lien avec les dtypes.

## 5. Cas laissés inchangés (justification)

| Site | Raison |
|---|---|
| `feature_names_in_` (`base/transformers.py`, `aggregation_constraint.py`, `stage_scaler.py`) | Convention sklearn : tableau `object` de noms de colonnes, attendu par `check_feature_names` et les méta-estimateurs. |
| Groupes de `crossvals` (`_labels_array`, `_entity_labels`, `_extract_groups_from_panel`) | Contrat sklearn de `groups` : tableau 1D d'une étiquette par ligne, donc des tuples pour une entité à plusieurs niveaux. Les traitements **internes** (regroupements, contrôle de cohérence, plis par entité) passent, eux, par des codes entiers (§ 3.3). |
| `delays/transformers.py` : `_extract_param_by_variable` | Conversion **voulue** vers `None` : les valeurs partent en kwargs de transformateurs, hors pandas (règle « hors pandas, une chaîne manquante est `None` »). Tables de quelques lignes. |
| `ImputationPlan.to_diagnostic_frame()` : `scale_factor` / `fit_scale_factor` avec un facteur par ligne | Valeurs **hétérogènes** (flottant et libellé `'per-row'`) : `object` légitime selon les conventions. La colonne est `float64` dès qu'aucune étape n'a de facteur par ligne. |
| `ImputationPlan.to_diagnostic_frame()` : `variable` d'un plan vide | Libellé repris de l'entrée : sans étape, aucun libellé dont inférer le dtype. |
| `FrequencyAligner` : colonne mêlant booléens convertis et valeurs d'origine (entité non convertie) | Valeurs hétérogènes : `object` légitime. La colonne passe en `boolean` dès qu'elle est entièrement booléenne. |
| Bruit interne de pandas (Series `df.dtypes`, Index de noms de niveaux, Index `[None]` d'un `concat`, clés tuples d'un `groupby` sur plusieurs colonnes, `value_counts` d'un niveau) | Objets créés par pandas lui-même, de taille bornée par le nombre de colonnes, de niveaux ou de groupes distincts, jamais par le nombre de lignes. |
| `utils/frequency/detector.py` (`_detect_panel_frequencies`), `imputation_window.py` (`_detect_entity_index_frequencies`) | `groupby(level=[...])` dont les clés de groupe sont des tuples : une clé par **entité**, pas par ligne (≈ 400 créations sur toute la suite). Candidat éventuel à `get_entity_codes` hors du périmètre de cet audit. |
| `delays/data_manager.py` (`_build_detection_report`) | `value_counts` et `dict.fromkeys` sur les clés **détectées** distinctes, pour un rapport de quelques lignes. |
| Entrées utilisateur en `object` | Une colonne fournie en `object` par l'utilisateur reste `object` dans les copies internes : l'harmonisation se fait aux sorties publiques. |

## 6. Changements visibles

- **Dtypes de sortie** : voir le tableau du § 3.5 (`Int64`, `float64`, `bool`, `boolean`,
  `CELL_ORIGIN_DTYPE`). Avec un catégoriel, `value_counts()` liste aussi les catégories absentes
  (décompte 0) ; une cellule se relit en simple chaîne et se compare par `==`.
- **Nouveaux symboles publics** : `tsforecast.panel.factorize_row_keys`,
  `tsforecast.panel.get_entity_codes`, `tsforecast.frequency.CELL_ORIGIN_DTYPE`,
  `tsforecast.frequency.max_cell_origin`.
- **Signature** : `training_set_builder.split_training_index` rend
  `(real_index, entity_codes, entities)` au lieu de `(real_index, liste d'entités par ligne)`.
- **Documentation** : `docs/concepts/mixed_frequency_imputation.md` (origines catégorielles
  ordonnées) ; `CLAUDE.md` (liste des modules, et conventions : origines, clés composites en codes
  entiers, dtypes explicites des colonnes numériques publiques).

## 7. Vérifications

- Suite complète : **14 064 tests passés** (dont 4 nouveaux fichiers ou classes de tests :
  `tests/unit/panel/utils/test_entity_codes.py`,
  `tests/unit/delays/calculator/test_distinct_combinations.py`, `TestRescaleBlocks`,
  `TestCellOriginDtype` / `TestMaxCellOrigin`, plus les tests de dtypes des rapports).
- Tests mis à jour pour les nouveaux dtypes : stockage des origines, `row_origin`, `boolean` de
  `FrequencyAligner`, signature de `split_training_index`, nombre de fonctions publiques de
  `panel.utils`, `value_counts` sur origines catégorielles.
- Comparaisons bit à bit avant/après : `calculate_applicable_delay` (2 040 appels) et
  `HighFrequencyImputer` / `AggregationConstraint` (38 sorties, avertissements compris) : 0 écart
  hors changements de dtype voulus.
- Doctests : mêmes 10 échecs qu'avant ce chantier (préexistants, sans rapport).
  `scripts/check_conventions.py --rule all` : aucune nouvelle violation.
- Stockage pyarrow des chaînes : 14 064 tests passés également (§ 8).

## 8. Stockage pyarrow

Suite complète exécutée dans un environnement avec l'extra `pyarrow` (pyarrow 26.0.0, chaînes
`str` stockées en Arrow) : **14 064 tests passés**. Les origines et la provenance étant
catégorielles (codes entiers), elles se comportent à l'identique sous les deux stockages ; les
chaînes produites passent par des tableaux pandas (`.array`, `pd.array(..., dtype='str')`), jamais
par un tableau `object` intermédiaire.

Mêmes bancs d'essai sous pyarrow :

| Banc d'essai | Avant | Après | Facteur |
|---|---:|---:|---:|
| `calculate_applicable_delay` (7 200 lignes) | 12,467 s | 0,028 s | ×440 |
| `calculate_applicable_delay`, `unit='hour'` | 14,221 s | 0,030 s | ×470 |
| `AggregationConstraint.rescale` (48 000 lignes) | 2,643 s | 0,139 s | ×19 |
| `HighFrequencyImputer.fit_transform` (panel de 20 entités) | 6,080 s | 5,019 s | ×1,2 |
| `validate_entities_grouped` (120 000 lignes, 3 niveaux) | 0,271 s | 0,007 s | ×38 |
| `validate_sorted_within_groups` (idem) | 0,295 s | 0,018 s | ×16 |
| `iter_entity_blocks` (500 entités) | 2,095 s | 0,150 s | ×14 |

Sous pyarrow, l'ancien `apply(axis=1)` était encore plus lent (conversion Arrow → objets Python à
chaque ligne) : c'est là que le gain est le plus marqué.
