# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ts-forecast** est un package Python pour le traitement et la validation de séries temporelles et de données de panel. Il fournit des outils compatibles avec sklearn pour :
- La validation croisée temporelle (time-aware cross-validation)
- La gestion des délais de publication (publication delays)
- L'imputation de données multi-fréquences avec suivi de provenance
- Le traitement robuste de données temporelles
- L'exposition de métriques prêtes pour le tracking d'expériences (MLflow)

## Structure du Projet

### Arborescence des modules

```
tsforecast/
├── base/          # Classes de base des transformateurs temporels
├── crossvals/     # Validation croisée respectant l'ordre temporel
├── delays/        # Délais de publication (inférence, application, inversion)
├── frequency/     # Détection de fréquence + imputation multi-fréquences
├── panel/         # Transformations par entité + utilitaires panel
├── xy/            # Pipeline et mixin pour transformateurs (X, y)
├── tracking/      # Constructeurs de métriques pour trackers (MLflow, ...)
└── utils/         # Utilitaires temporels transversaux (sous-paquets)
```

### `tsforecast/base/`
Classes abstraites et mixins pour construire des transformateurs sklearn
temporels (`transformers.py`) :
- `TimeSeriesTransformerMixin` : gestion de l'index temporel
- `PanelTimeSeriesTransformer` : classe de base séries / panel
- `ReversibleTransformerMixin` : transformations réversibles

### `tsforecast/crossvals/`
- `base.py` : `OutOfSampleSplit`, `InSampleSplit`
- `time_series.py` : `TSOutOfSampleSplit`, `TSInSampleSplit`
- `panel.py` : `PanelOutOfSampleSplit`, `PanelInSampleSplit`,
  `PanelOutOfSampleSplitPerEntity`, `PanelInSampleSplitPerEntity`

Tous les splitters suivent le contrat sklearn (`split()`, `get_n_splits()`) et
sont donc utilisables directement avec `sklearn.model_selection.cross_validate`.

### `tsforecast/delays/`
- `data_manager.py` : `compare_and_detect_delays()` — inférence de délais depuis
  deux jeux de données ; `return_report=True` renvoie aussi un `DelayDetectionReport`
- `report.py` : `DelayDetectionReport`, `DelayFitReport` (attribut ajusté `fit_report_`
  de `PublicationDelayTransformer`), `ColumnDelayRecord` — rapports immuables
  (`summary()`, `to_dict()` JSON, `to_frame()`) ; le logger `tsforecast.delays.*`
  n'émet que `summary()` (INFO), aucun handler n'est configuré par le paquet
- `calculator.py` : `calculate_applicable_delay()` — délai applicable à une
  observation
- `transformers.py` : `PublicationDelayTransformer` (orchestrateur), plus les
  briques `ShiftTransformer`, `MaskTransformer`, et les fabriques
  `create_delay_transformer_factory()`, `prepare_entity_kwargs_from_delays()`

### `tsforecast/frequency/`
**Implémentation courante** : `HighFrequencyImputer` — impute les colonnes de
basse fréquence sur une grille plus fine, en cascade, avec suivi de provenance.
Ajusté avec `y`, il augmente un jeu d'entraînement : `fit_transform(X, y)` est la
passe d'entraînement (`X` et `y` sur la grille), `transform` la passe de prédiction
(`X` seul, sur ses lignes d'entrée, `y` rendu inchangé) ; `y` n'est pas covariable
de `X` sauf `target_as_covariate=True` (décisions D47 / D48 de la spec).

Composants auxiliaires de `HighFrequencyImputer` :
- `imputation_plan.py` : `ImputationPlan`, `ImputationStep` — état ajusté complet
- `provenance.py` : `ImputationProvenanceTracker`, `ProvenanceType` (10 niveaux),
  primitives de souillure (`Taint`, `resolve_model_provenance`, ...)
- `covariate_materializer.py` : `CovariateMaterializer` — production de
  `X_train` / `X_pred` (invariant central : jamais de dégradation fit → predict)
- `aggregation_constraint.py` : `AggregationConstraint` — recale les sous-périodes
  sur le total observé
- `stage_scaler.py` : `StageScaler` — mise à l'échelle de fréquence d'une étape
- `variable_orderer.py` : `VariableOrderer` — ordre des variables (`'frequency'`
  ou `'cv'`), expose `scores_` / `fallback_keys_`
- `training_set_builder.py` : jeu d'entraînement mutualisé inter-entités
- `imputation_window.py` : `ImputationWindowCalculator`, `ImputationScope`
- `frequency_aligner.py` : `FrequencyAligner` — outil autonome, **non utilisé** par
  `HighFrequencyImputer` : agrégation / interpolation de variables choisies
- `target_frequency_validator.py` : `TargetFrequencyValidator`
- `regularizer.py` : `IndexRegularizer`, `is_regular()`, `regularize()`

Détection de fréquence : `FrequencyDetector` et `detect_frequency()` /
`detect_dataset_frequency()` / `detect_index_frequency()` sont réexportés depuis
`tsforecast/utils/frequency/`.

### `tsforecast/panel/`
- `transformers.py` : `PanelwiseTransformer` — applique un transformateur
  indépendamment à chaque entité d'un panel (parallélisable via joblib)
- `utils.py` : `is_panel_data()`, `get_entity_levels()`, `get_unique_panel_entities()`,
  `normalize_entity_key()`, `get_entity_mask()`, `resolve_entity_column_frequencies()`, ...

### `tsforecast/xy/`
- `transformers.py` : `XYTransformerMixin` — transformateurs qui transforment
  simultanément `X` et `y` ; `XYPanelTimeSeriesTransformer` — classe de base
  validée (séries / panel) des transformateurs XY, dont hérite `HighFrequencyImputer` ;
  `DropNaNTarget` — retire les lignes sans cible juste avant l'estimateur (au `fit`
  et au `score`, jamais au `predict`) ; `is_xy_transformer()` — une étape est XY si et
  seulement si elle hérite de l'une de ces deux classes de base (jamais par sa signature)
- `pipeline.py` : `XYPipeline` — remplacement direct de `sklearn.pipeline.Pipeline`
  gérant le passage de `y` à travers les étapes XY (les autres étapes : comportement de
  `Pipeline`) ; les méthodes de prédiction (`predict`, `predict_proba`, ...) refusent `y` ;
  les paramètres par ligne (`sample_weight`) suivent les lignes gardées (lues sur l'index) ;
  `memory=` met en cache les étapes ; compatible `GridSearchCV`, `cross_validate`,
  metadata routing

### `tsforecast/tracking/`
- `metrics.py` : `imputation_metrics()`, `delay_metrics()`, `detection_metrics()`,
  `split_summary()` —
  transforment l'état ajusté d'un composant en `dict[str, float]` plat,
  directement loggable via `mlflow.log_metrics`. Aucune dépendance à MLflow.

### `tsforecast/utils/`
Utilitaires transversaux, organisés en sous-paquets (chacun expose une classe
`*Normalizer`, une classe `*Converter` et des fonctions utilitaires) :
- `utils/_constants.py` : constantes transverses (mois / jours, classification des
  bases de fréquence, tables de durées) — source unique, ne dépend que de la stdlib ;
  les constantes propres à un seul module restent locales à celui-ci
- `utils/frequency/` : `FrequencyNormalizer`, `FrequencyConverter`,
  `FrequencyDetector` (`detector.py`), `normalize_frequency()`, `to_pandas_freq()`,
  `is_higher_frequency()`, `get_frequency_order()`, ...
- `utils/duration/` : `DurationNormalizer`, `DurationConverter`,
  `normalize_duration()`, `convert_duration()`, `get_duration_nanoseconds()`
  (durée entière exacte d'une unité, source unique du calcul de délais), ...
- `utils/position/` : `PeriodPositionNormalizer`, `PeriodPositionConverter`,
  `normalize_position()`, `flip_position()`, ...
- `utils/time/` : `resolve_date()`, `timeseries_to_string()` /
  `string_to_timeseries()`, `get_period_start()` / `get_period_end()`,
  `get_period_boundaries()`, `as_exact_unit()` (`resolution.py` : unité de stockage des
  dates sans arrondi, source unique de la règle d'unité de sortie)
- `utils/validation/` : `validate_temporal_data()`, `restore_original_structure()`,
  `validate_entities_grouped()`, `validate_sorted_within_groups()`
- `utils/parse/` : `parse_frequency()`, `build_frequency_string()`
- `utils/abc/` : `TemporalNormalizer`, `TemporalConverter` (classes abstraites)

## Documentation

La documentation détaillée vit dans `docs/` (site mkdocs, `mkdocs.yml`). Pour
chaque composant principal : une page *concept*, un *tutoriel*, une page *API*
générée par mkdocstrings.
- Point d'entrée : `docs/index.md`
- Concepts : `docs/concepts/`
- Tutoriels : `docs/tutorials/`
- Intégration MLflow / tracking : `docs/guides/mlflow_tracking.md`
- Référence API : `docs/api/`

Build local : `uv run mkdocs serve` (ou `uv run mkdocs build --strict`).

## Objectifs Fonctionnels

### ✅ Implémenté
- Validation croisée temporelle (in-sample et out-of-sample) pour séries et panels
- Détection et inférence de délais de publication ; application / inversion via
  transformateurs sklearn
- Détection automatique de fréquence
- Imputation multi-fréquences en cascade avec suivi de provenance
  (`HighFrequencyImputer`)
- Normalisation et conversion de fréquences, durées, positions
- Validation robuste de données temporelles et panel ; manipulation de dates et
  périodes
- Métriques de tracking plates pour MLflow (`tsforecast.tracking`)

### 🚧 À implémenter
- Agrégation multi-fréquences dédiée (traitée dans un package séparé)
- Support de régresseurs / classifieurs CUDA
- Calcul d'intervalles de confiance par bootstrapping ; conformal prediction
- Test de Diebold & Mariano (+ correction de Harvey)
- Méthodes d'interprétabilité
- Helper `crossval_score` dédié (aujourd'hui : passer par
  `sklearn.model_selection.cross_validate`)

## Conventions de Code

### Style et Typage
- **Type hints** : Utiliser systématiquement les annotations de type (`typing`, `pandas`, `numpy`)
- **Compatibilité sklearn** : Les transformateurs et splitters doivent suivre la convention sklearn (méthodes `fit()`, `transform()`, `split()`, etc.)

### pandas ≥ 3.0
Le paquet requiert `pandas>=3.0,<4` et `scikit-learn>=1.9,<2` (pas de support de pandas 2).
- **Copy-on-Write** : aucun chaînage d'affectation (`df['a'][m] = v`, `df.loc[m]['a'] = v`) —
  écrire `df.loc[m, 'a'] = v` ; une fonction ne modifie jamais un objet reçu (copie, puis
  résultat renvoyé) ; les tableaux de `.to_numpy()` / `.values` / `np.asarray(df)` sont en
  **lecture seule** : `np.array(...)` ou `.to_numpy(copy=True)` avant toute écriture en place.
- **Résolution des dates explicite** : pandas infère l'unité (`s`, `ms`, `us`, `ns`), `us` par
  défaut pour des chaînes ; toute conversion en entiers passe par `as_unit('ns')` avant `.asi8` /
  `astype('int64')` (ou par `.value` d'un `Timestamp` / `Timedelta`, toujours en ns) ; la durée
  d'une période est `(periods + 1).start_time - periods.start_time`, jamais
  `end_time - start_time + 1 ns` (`end_time` n'est pas à la nanoseconde).
- **Unité des dates en sortie = unité de l'entrée** : toute date rendue par un composant public (index, niveau
  temporel, colonne `datetime64`, `Timestamp`) est dans l'unité de l'entrée de référence, via
  `tsforecast.utils.time.as_exact_unit(dates, unit)` appliqué à la sortie (les calculs internes gardent leur
  unité exacte ; instants et valeurs ne changent jamais). Entrée de référence : l'axe temporel de la donnée
  (index, dernier niveau d'un panel, colonne `time_col`) ; pour une date scalaire, l'unité de `pd.Timestamp(date)`
  (`us` pour un `datetime`, un `Period` ou une chaîne) ; deux jeux comparés : le jeu principal (`new_data`) ;
  `X` et `y` : celle de `X`. Un paramètre date scalaire (`download_date`, `prediction_date`, `origin`,
  `test_indices`) ne fixe jamais l'unité. Date non représentable dans l'unité cible : jamais d'arrondi, la plus
  grossière des unités plus fines qui la représente (une grille à `100ms` de dates en `s` passe en `ms`) ; hors de
  la plage de l'unité (`ns` : 1677-09-21 – 2262-04-11) : `OutOfBoundsDatetime`. Les tests le vérifient par
  `with_datetime_resolution` × `datetime_units` (`assert datetime_units(result) == {…: unit}`).
- **Dtype chaîne** : libellés et colonnes de chaînes sont en dtype `str` (NaN comme manquant) ;
  ne pas tester `dtype == object` ; `dtype=object` explicite seulement pour des valeurs
  hétérogènes (énumérations, tuples, `None`).
- **Alias de fréquence** : alias pandas actuels seulement (`ME`, `QE`, `YE`, `SME`, `h`, `min`,
  `s`...) ; les codes de base du paquet (`'M'`, `'Q'`, `'Y'`, `'SM'`) ne vont jamais à pandas sans
  `default_position` (`build_frequency_string`, `to_pandas_freq`).
- **`concat`** : écarter les entrées vides avant de concaténer (elles comptent dans les dtypes du
  résultat) ; préciser `sort=` dès que l'axe non concaténé peut être un `DatetimeIndex`.
- **`groupby`** : un niveau unique se passe en scalaire (`level=0`), une liste de longueur 1 donne
  des clés en tuples ; `DataFrame.stack()` conserve les cellules vides (`.dropna()` explicite).

### Commentaires et Documentation

#### Commentaires internes (FRANÇAIS)
- Écrire en **français** pour faciliter la compréhension technique
- Privilégier les formulations nominales :
  - ✅ `# Vérification des arguments`
  - ❌ `# Vérifier les arguments`
- Commenter les implémentations abstraites ou techniques
- Expliquer le "pourquoi" plutôt que le "quoi" quand le code est clair — un "pourquoi" qui
  décrit une **contrainte technique ou sémantique permanente** du code actuel (voir
  « Règles de rédaction du code » ci-dessous), jamais une justification historique

#### Règles de rédaction du code (commentaires, docstrings, en-têtes)

Le package n'a pas encore d'utilisateurs : le code doit être lisible par quelqu'un qui n'a
**aucune connaissance de l'historique de l'implémentation ni des documents de conception**.

**1. Autoportance.** Aucun commentaire ni aucune docstring (code **et** tests) ne renvoie à un
document ou à un numéro extérieur au fichier : numéro de paragraphe (`§3.2`, « section 4 »),
décision (`D47`), spécification (« voir la spec »), invariant (`I11`), anomalie (`ANO-FREQ-073`,
`[CODE]`), prompt, ticket. On **énonce la règle** au lieu de la citer.
  - ❌ `# Invariant I11 : jamais de dégradation fit → predict`
  - ✅ `# La prédiction ne dispose jamais de moins d'information que l'entraînement`
  - Seule exception : le `reason=` d'un `pytest.mark.xfail` peut commencer par un identifiant
    `ANO-…` (lien machine vers `tests/ANOMALIES.md`), à condition d'être suivi d'une phrase
    complète qui se comprend seule. `tests/ANOMALIES.md` et les documents de conception
    archivés sont, eux, le lieu de ces références.

**2. Pas d'historique ni de justification de décision.** Décrire ce que le code **est**, pas ce
qu'il était ni pourquoi il a changé : proscrire « auparavant », « désormais », « corrigé »,
« remplace l'ancien… », « contrairement à la version précédente », « pour corriger le bug ».
L'historique relève de `git log`, des messages de commit et de `tests/ANOMALIES.md`. Une
contrainte technique durable (« les mois de 28 jours imposent un calcul en entiers exacts »)
reste bienvenue, sans son récit.

**3. En-tête de fichier.** Chaque fichier `.py` commence par une docstring de module
(`""" … """`), **en anglais**, première instruction du fichier, de 1 à 4 phrases : ce que le
fichier contient (rôle, principaux symboles publics ; pour un fichier de tests : symboles testés
et familles de scénarios). Elle est elle-même autoportante (règles 1 et 2). Les `__init__.py`
décrivent le paquet et ce qu'il réexporte. Les fichiers texte de configuration (`.yml`, `.yaml`,
`.toml`) commencent par un commentaire `#` équivalent en anglais. Exemptés : JSON, verrous
(`uv.lock`), fichiers générés ; un notebook commence par une cellule markdown de présentation.

**4. Fichiers de tests courts.** Un fichier de tests ne dépasse pas environ **1 200 lignes** ; au-delà
il est remplacé par un **paquet** de même nom (sans préfixe `test_`) contenant plusieurs fichiers
thématiques (fixtures partagées dans le `conftest.py` du paquet). Chaque fichier créé reçoit son
en-tête (règle 3).

**5. Doctests exécutables.** Les exemples `Examples:` des docstrings sont des doctests qui
**passent** : imports explicites, données minuscules et déterministes, sortie stable
(`pytest --doctest-modules tsforecast`). Pas de `+SKIP`.

**Vérification mécanique** : `python scripts/check_conventions.py --rule all` (règles `refs`,
`header`, `length`, `doctest` bloquantes ; `history` à relire).

#### Docstrings (ENGLISH)
- Convention **Google Style** obligatoire
- Inclure systématiquement :
  - Description concise de la fonction/classe
  - `Args:` avec types et descriptions
  - `Returns:` avec type et description
  - `Raises:` pour les exceptions possibles
  - `Examples:` avec cas d'usage concrets, sous forme de doctest qui passe (règle 5 ci-dessus)

**Exemple de docstring :**
```python
def validate_temporal_data(
    data: Union[pd.Series, pd.DataFrame],
    time_col: Optional[str] = None,
    strict: bool = True
) -> pd.DataFrame:
    """Validate and prepare time series or panel data.

    Args:
        data: Time series or panel data to validate
        time_col: Name of the time column (if None, uses index)
        strict: If True, raises errors; if False, attempts corrections

    Returns:
        Validated DataFrame with proper temporal index

    Raises:
        ValueError: If validation fails and strict=True

    Examples:
        >>> dates = pd.date_range('2023-01-01', periods=5)
        >>> series = pd.Series([1, 2, 3, 4, 5], index=dates)
        >>> validated = validate_temporal_data(series)
    """
```

### Constantes
- **Réutiliser avant de redéfinir** : avant d'écrire une constante (ou un littéral
  « magique » : `7`, `5`, `12`, tables de durées, abréviations de mois / jours...),
  chercher dans `tsforecast/utils/_constants.py` (`Grep` sur le nom ou la valeur)
  et l'importer si elle existe — ne jamais la dupliquer localement.
- **Constante transverse manquante** : si elle est susceptible de servir à plusieurs
  modules (vocabulaire calendaire, classification de fréquences, durées), l'ajouter
  dans `utils/_constants.py` puis l'importer, plutôt que de la définir dans le module
  appelant.
- **Constante propre à un module** : seule une constante vraiment spécifique à un
  seul module reste locale à celui-ci (préfixe `_`).
- Remplacer aussi les littéraux équivalents à une constante existante (ex. `/ 7`
  → `/ DAYS_PER_WEEK`) lorsqu'on touche à la ligne.

### Notebooks
- Illustrer les **cas d'usage variés** de chaque classe/fonction
- Montrer l'impact des **différents paramètres** sur le comportement
- Inclure des exemples avec données réelles (séries temporelles, panels)

## Stratégie de Tests

### Framework et Organisation
- **Framework** : `pytest` pour tous les tests
- **Structure** :
  ```
  tests/
  ├── conftest.py        # pytest_plugins, marquage auto unit/integration, xfail transitoire
  ├── legacy_failures.txt  # TRANSITOIRE : échecs hérités en cours de tri
  ├── ANOMALIES.md       # registre des anomalies [CODE]/[DOC]/[SPEC]
  ├── run_tests.py        # --mode all | unit | integration | fast | coverage, --path
  ├── support/            # jeux de données, fixtures et doublures partagés
  │   ├── datasets.py      # constructeurs purs (notebook 2, PANEL-X, ...)
  │   ├── fixtures.py      # fixtures pytest, enregistrées via pytest_plugins
  │   ├── perturbations.py # désordre, noms spéciaux, positions, résolution des dates, ...
  │   └── estimators.py    # SpyEstimator, FailingEstimator, ...
  ├── unit/               # miroir EXACT de tsforecast/
  │   ├── base/  crossvals/  panel/  tracking/  xy/
  │   ├── delays/           # test_calculator.py, test_data_manager.py, ...
  │   ├── frequency/        # un test_<module>.py par module de tsforecast/frequency/
  │   └── utils/
  │       ├── abc/  duration/  parse/  position/  time/  validation/
  │       └── frequency/     # test_normalizer.py, test_utils.py, detector/ et converter/ (paquets)
  └── integration/
      ├── crossvals/  delays/  frequency/  pipeline/
      └── input_contract/  # entrées jamais modifiées, entrées en lecture seule, libellés str
  ```
- **Règle de miroir** : `tsforecast/<paquet>/<module>.py` ↔ `tests/unit/<paquet>/test_<module>.py`.
  Un module source trop gros pour un seul fichier de test devient un **paquet** de même nom
  sans préfixe `test_` (ex. `tests/unit/utils/frequency/converter/`) dès que le fichier de
  tests dépasse environ **1 200 lignes** (règle 4 de « Règles de rédaction du code »).
- **Frontière unit / integration** : *unit* = un composant de `tsforecast/` sur des données
  petites, à valeurs d'or calculables ; *integration* = plusieurs composants publics
  enchaînés, ou un composant sur des jeux réalistes.

### Priorités de Test

#### 1. Cas limites (Edge Cases)
Focus principal sur les **cas limites** car le package manipule des index pandas :
- Données non triées (temporal order)
- Index dupliqués
- Entités manquantes dans les panels
- Fréquences irrégulières
- Valeurs manquantes (NaN)
- Datasets vides ou avec une seule observation

#### 2. Robustesse aux index
Vérifier la robustesse à :
- **Dés-ordonnancement** : Données intentionnellement mal triées
- **Index mixtes** : Combinaisons de DatetimeIndex, RangeIndex, MultiIndex
- **Noms de colonnes** : Colonnes avec caractères spéciaux, espaces
- **Types de dates** : datetime64, Period, Timestamp
- **Résolution des dates** : mêmes instants en `s`, `ms`, `us` et `ns`
  (`with_datetime_resolution`, classes `TestDatetimeResolution`) — le résultat ne doit pas en dépendre

#### 3. Validation temporelle
Tester spécifiquement :
- Détection correcte de fréquences (régulières et irrégulières)
- Gestion des gaps temporels
- Comportement avec différentes fréquences (horaire, journalier, mensuel, etc.)
- Transitions de fréquence (mixed-frequency)

### Exécution des Tests

**Commande de base :**
```bash
uv run tests/run_tests.py --mode all
```

**Options disponibles :**
- `--mode coverage` : suite complète avec couverture et contrôle du seuil de 90 % par module
- `--mode fast` : tous les tests sauf `slow`
- `--path <cible>` : fichier, dossier ou node id passé directement à pytest
- `--mode all` : Tous les tests
- `--mode unit` : Tests unitaires uniquement
- `--mode integration` : Tests d'intégration
- `--mode conventions` : lance `scripts/check_conventions.py --rule all` (échoue si une règle bloquante est violée)
- `--mode doctest` : `pytest --doctest-modules tsforecast`

**Vérification des conventions de rédaction :**
```bash
python scripts/check_conventions.py --rule all --format summary
python scripts/check_conventions.py --rule refs tsforecast --format detail
python scripts/check_conventions.py --verify-code-unchanged HEAD  # seuls les docstrings diffèrent
```

**Tests ciblés :**
```bash
# Tester un module spécifique
uv run pytest tests/unit/crossvals/

# Tester une classe spécifique
uv run pytest tests/unit/crossvals/test_time_series.py::TestTSOutOfSampleSplit

# Tester avec verbosité
uv run pytest -v tests/

# Exclure les tests lents
uv run pytest -m "not slow" tests/
```

### Couverture de Code
- **Outil** : `pytest-cov` (lignes **et** branches) mesure, localement et en CI ; Codecov publie
  (lecture de `coverage.xml`). Configuration : `[tool.coverage.*]` dans `pyproject.toml`,
  `codecov.yml` (statut `project` informatif, `tests/` et `notebooks/` ignorés).
- **Seuil** : **≥ 90 %** par module de `tsforecast/{utils,delays,frequency,base,crossvals,panel,xy}`.
  `uv run python tests/run_tests.py --mode coverage` lit `coverage.json` et échoue en listant
  les modules fautifs.
- **Mesure locale** : `uv run pytest --cov=tsforecast --cov-branch tests/` (rapports `term-missing`,
  `htmlcov/`, `coverage.xml`, `coverage.json` via `--mode coverage`).
- **CI** : `.github/workflows/tests.yml` — pull request vers `main` : suite `-m "not slow"` sans
  couverture ; push sur `main` : même suite avec couverture, envoi à Codecov (secret
  `CODECOV_TOKEN`, `if: always()`), puis tests `slow` dans un job séparé. Aucun déclenchement
  sur les pushs des autres branches.
- **Badges** (`README.md`) : statut du workflow, couverture globale, et un badge par paquet via les
  *components* Codecov (`badge.svg?component=<paquet>`), un par paquet listé ci-dessus.
