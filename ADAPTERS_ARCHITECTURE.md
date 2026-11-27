# Architecture des adapters `tsforecast.adapters`

> **Document de travail de la branche `qb-adapters`.** Il fixe le contrat commun et les décisions
> de conception que doivent respecter toutes les sessions de code listées dans
> `ADAPTERS_PROMPTS.md`. Il n'est pas destiné à `main` : on le supprime (ou on le transforme en page
> de documentation mkdocs) avant la fusion.
>
> **Règle d'autoportance** (CLAUDE.md de `main`, « Règles de rédaction du code ») : aucun
> commentaire, docstring ou test ne cite ce document, ses numéros de section ou ses identifiants
> de décision (`AD-xx`). On énonce la règle dans le code au lieu de la citer.
>
> **Mise à jour** : chaque session qui tranche une question ouverte (§ 9) ou s'écarte d'une règle
> l'inscrit dans le journal des décisions (§ 10) avant de rendre la main.

---

## 1. Objectif et principe central

Envelopper des modèles de paquets tiers (sktime, darts, hierarchicalforecast, modèles de
fondation pour séries temporelles publiés sur Hugging Face ; AutoGluon dès qu'il supportera
pandas 3, § 5.6) dans des estimateurs **compatibles sklearn**, et fournir une agrégation d'experts
en ligne **native** (BOA, MLpol, § 5.4) qui combine leurs prévisions ; le tout utilisable tel quel
avec :

- `sklearn.model_selection.cross_val_score`, `cross_validate`, `cross_val_predict`, `GridSearchCV` ;
- les splitters de `tsforecast.crossvals` (`TSOutOfSampleSplit`, `PanelOutOfSampleSplit`, ...) ;
- `tsforecast.xy.XYPipeline` et les étapes XY (`DropNaNTarget`, `HighFrequencyImputer`, ...) ;
- `sklearn.pipeline.Pipeline`.

**Principe central — l'horizon appartient à `tsforecast`, jamais à l'adapter.**

La convention est celle du tutoriel `docs/tutorials/pseudo_real_time_forecast.md` de `main` :

- une ligne est indexée par son **origine** `t` (date à laquelle la prévision est faite) ;
- `y` à la ligne `t` vaut `y(t + h)` (cible décalée de `-h`, **entité par entité** sur un panel) ;
- `X` à la ligne `t` ne contient que ce qui est connu à l'origine `t` ;
- le splitter laisse `gap = h + d - 1` lignes entre entraînement et test.

Un adapter **apprend `y` à partir de `X` ligne à ligne** et ne décale jamais rien. Un modèle tiers
qui raisonne « en séries temporelles » (sktime, darts, modèles de fondation) voit la série des
lignes indexées par origine : quand il doit prévoir une ligne de test, la distance entre la dernière
ligne d'entraînement de l'entité et la ligne de test (en pas de la grille) est **déduite des index**,
jamais d'un paramètre `fh` / `horizon` fourni par l'utilisateur. Comme le splitter garantit
l'absence de fuite, l'adapter n'a pas à la contrôler — mais il ne doit jamais lire une valeur de `y`
postérieure à la dernière ligne d'entraînement.

---

## 2. État des lieux (au 2026-11-25)

### 2.1 Branche

- `qb-adapters` diverge de `main` depuis le 2025-08-07 : **465 commits** de retard, 36 commits
  propres. Les changements propres à la branche sont presque tous additifs :
  `tsforecast/adapters/*`, `notebooks/4 - QB - Adapters.ipynb`, `Debug - HierarchicalForescast.ipynb`
  (racine), `pyproject.toml`, `uv.lock`, et `tsforecast/crossvals/base.py` (ajout de
  `get_n_splits`, **déjà présent sur `main`** : prendre la version de `main`).
- `main` impose Python ≥ 3.13, `pandas>=3.0,<4`, `scikit-learn>=1.9,<2`, `numpy>=2.3`. La `.venv`
  locale est déjà sur pandas 3.0.6 / scikit-learn 1.9.1. Depuis la session 1, les extras
  `adapters-all` et le groupe `notebooks` y sont installés (versions au § 2.3).
- `main` a renuméroté ses notebooks (`0 - Utils`, `1 - Sklearn extensions`, `2 - Delays`,
  `3 - Crossvals`, `4 - Mixed frequencies`). Le notebook des adapters garde son nom
  `4 - QB - Adapters.ipynb` tant que l'utilisateur n'a pas décidé d'un renommage (§ 9).

### 2.2 Défauts déjà repérés (à confirmer et compléter par les sessions d'audit)

**Transverses**
- `adapters/__init__.py` : `__getattr__` attrape `ImportError` et masque les vraies erreurs
  d'import du module de l'adapter ; extra `adapters-all` mal formé (espace dans la liste) ;
  opera référencé sur la branche `@develop` (non figé) ; `sktime` ajouté au groupe `dev`.
- Ordre d'héritage `BaseEstimator, RegressorMixin` : sklearn ≥ 1.7 attend les mixins **avant**
  `BaseEstimator` (`class X(RegressorMixin, BaseEstimator)`) pour les tags.
- Les adapters modifient l'objet tiers reçu au constructeur (`self.model = ...` dans `fit`) :
  incompatible avec `clone`, `get_params` et la réutilisation d'une instance.
- Docstrings `Examples:` non exécutables (doctests de `main` obligatoires et passants).

**`SktimeAdapter`**
- `fh=0` relatif au *cutoff* : la prévision porte sur la date de fin d'entraînement, pas sur les
  dates des lignes de test (faux dès que `gap > 0` ou `test_size > 1`).
- Forecaster mutée au `fit` (pas de clone) ; sortie sktime renvoyée telle quelle (non réalignée sur
  les lignes de `X`) ; `get_params` / `set_params` réécrits inutilement (le `BaseEstimator` gère le
  préfixe `forecaster__`) ; `y` panel converti en DataFrame mais pas `X`.

**`DartsAdapter`**
- `self.model` réinstancié dans `fit` (perte des paramètres utilisateur visibles par
  `get_params`, avertissement à chaque `fit` de chaque pli).
- Panel traité par `unstack` en **une seule série multivariée** : chaque composante (entité) reçoit
  les covariables de **toutes** les entités (fuite inter-entités, `n_features × n_entities`
  colonnes) au lieu d'une liste de `TimeSeries` passée à un modèle global ; `static_covariates`
  construites mais inutilisées.
- Trou entre entraînement et test rempli par `0` et prévision récursive sur `n` pas ;
  `n_features_in_` faux quand `X is None`.
- Confirmés et complétés en session 4 (scénarios exécutés) : `LinearRegressionModel()` sans lags
  refusé par darts (exemples faux) ; `is_regressor` faux ; panel déséquilibré → `ValueError`
  (NaN), entités entières → prévisions toutes NaN sans erreur ; trous d'entraînement refusés par
  darts ; `predict` sur des lignes d'entraînement → `IndexError` ; sortie `Series` ; modèles locaux
  (`NaiveDrift`, `ExponentialSmoothing`) → `TypeError` ; `set_params(**get_params())` change
  l'empreinte (`_random_instance` tiré au hasard). La prédiction d'une série ne dépendait pas du gap.

**`HierarchicalForecastAdapter`** — inachevé, ne s'importe pas : erreurs de syntaxe
(`"id_col" = _ID_COL` dans un dict), noms indéfinis (`hierarchy`, `temporal_hierarchy`,
`df_hierarchy`, `Set`), méthodes appelées comme des fonctions (`_extract_entity_date_names`,
`_build_df_hierarchy`, `_build_unique_id_column`), docstring décrivant un paramètre `spec`
inexistant, `S_df_` / `tags_` réécrits dans `predict`, sortie au format long de hierarchicalforecast
non réalignée sur `X`. **Réécrit en session 6** (sous-paquet, mode cross-section, AD-14).

**`OperaAdapter`**
- `mixture_` et `is_fitted_` posés dans `__init__` (violation sklearn : `clone`,
  `check_is_fitted`) ; noms d'algorithmes faux dans la docstring (opera ne connaît que `BOA`,
  `MLpol`, `MLprod`, `FTRL`) ; `Mixture` exige des experts en `pandas.DataFrame` (un `ndarray` lève
  `TypeError`) ; aucun support panel. **Abandonné le 2026-11-26 (AD-18)** : remplacé par une
  agrégation native, sans dépendance à opera (§ 5.4). **Supprimé en session 10** (module, export,
  extra `adapters-opera`, entrées du `conftest.py` racine et de `test_init.py`).

**Notebook `4 - QB - Adapters.ipynb`**
- Section 2 : `X = X.shift(-horizon)` sur **tout le panel** (décalage inter-entités) et dans le
  mauvais sens (covariables futures) ; `gap = horizon + delays` au lieu de `h + d - 1`.
- Import de l'API privée `sklearn.utils.metaestimators._safe_split` ; `xgboost` et `tslearn`
  utilisés mais non déclarés ; cellule 53 syntaxiquement fausse ; section hierarchicalforecast
  appelant une API (`spec=`) qui n'existe pas.

### 2.3 Écosystème tiers (vérifié le 2026-11-25)

| Paquet | Distribution | Remarques |
|---|---|---|
| hierarchicalforecast | PyPI | `aggregate(df, spec)` (spec : liste de listes de colonnes), `aggregate_temporal(df, spec)` (spec : `{nom: nb de pas}`), `get_cross_temporal_tags`, `HierarchicalReconciliation(reconcilers).reconcile(Y_hat_df, S_df, tags, Y_df=None, temporal=False, ...)`. Colonnes de sortie `<modèle>/<Réconciliateur>[_method-<m>]`. La réconciliation temporelle ne supporte pas les méthodes *insample* (`mint_shrink`, ...). Le tutoriel cross-temporel fait deux appels séparés (cross-section, puis temporel sur toutes les séries cross-section) et n'évalue qu'avec des *tags* cross-temporels. **Vérifié en session 6 (1.5.3)** : `reconcile` mute les réconciliateurs (`fitted`, `P`, `W`, `y_hat`, `sampler`) et trie `level` en place ; `unique_id` et colonnes de `S_df` doivent être des chaînes ; sortie triée par ordre de `S` puis `ds` ; un modèle dont le nom contient `-lo`, `-hi` ou `-median` est écarté en silence ; `Y_df` à trous accepté (`is_balanced=False`, NaN gérés par `mint_shrink`), `Y_hat_df` rectangulaire exigé ; intervalles `normality` lus sur des colonnes `<modèle>-lo-<niveau>` / `-hi-` de `Y_hat_df`, `bootstrap` par blocs d'autant de dates insample consécutives que de dates à réconcilier ; `TopDown` / `MiddleOut` faux si les feuilles ne sont pas toutes au niveau le plus fin des *tags* ; **aucune réconciliation cross-temporelle native** (seulement `temporal=True`). |
| opera (`Dralliag/opera-python`) — **abandonné (AD-18)**, référence des tests de l'agrégation native seulement | **pas sur PyPI** (`pip install git+...`) ; version `0.0.1` | `Mixture(y, experts: DataFrame, awake=None, model="BOA"|"MLpol"|"MLprod"|"FTRL", coefficients="uniform"|ndarray, loss_type="mse"|"mae"|"mape"|"msle"|"mspe"|callable, loss_gradient=True, parameters=None)`. Le constructeur consomme les données (appelle `update`). `update(new_experts, new_y)` séquentiel ligne à ligne ; `predict(new_experts)` sans mise à jour (poids courants `w`). Dépendances non figées, **scipy importé mais non déclaré**. Compatibilité pandas 3 inconnue. |
| chronos-forecasting — **vérifié le 2026-11-27 (2.3.2, 2026-09-08)** | PyPI | `torch>=2.2,<3`, `transformers>=4.41,<6`, `accelerate>=1.1,<2`, `einops`, `pandas>=2,<4`, `numpy<3`, Python ≥ 3.10 ; Apache-2.0. `BaseChronosPipeline.from_pretrained(id, device_map=, dtype=, revision=)` choisit la classe selon `chronos_pipeline_class` de la configuration. **Chronos-2** (`Chronos2Pipeline`) : `predict_quantiles(inputs, prediction_length, quantile_levels, batch_size=256, context_length=, cross_learning=False, limit_prediction_length=False)` ; `inputs` = liste de dicts `{target: (T,) ou (V, T), past_covariates: {nom: (T,)}, future_covariates: {nom: (H,)}}` de longueurs libres (remplissage à gauche par NaN), clés futures ⊂ passées, longueur future exactement `H`, NaN = inconnu ; sortie : liste de `(V, H, Q)` ; NaN acceptés ; contexte tronqué à `model_context_length` (8 192) ; `H` > 1 024 déroulé avec avertissement ; niveaux appris `pipeline.quantiles` (21 pour `amazon/chronos-2`, 13 de 0,01 à 0,99 pour `autogluon/chronos-2-small`), interpolés entre eux, **bornés hors plage** (journal seulement) ; `batch_size` compte les lignes cible + covariables ; `cross_learning=True` partage l'information entre les séries du lot. **Bolt** (`ChronosBoltPipeline`) : liste de tenseurs 1-D, sans covariable, contexte 2 048, 64 pas par passe (déroulé au-delà), 9 niveaux. **T5** (`ChronosPipeline`) : trajectoires tirées au hasard. Inférence de Chronos-2 et Bolt sans aléa. Modèles : `amazon/chronos-2` (120 M), `autogluon/chronos-2-small` (28 M), `amazon/chronos-bolt-{tiny,mini,small,base}` (9 M à 205 M), `amazon/chronos-t5-*`. |
| timestral (Mistral AI) — **vérifié le 2026-11-27 (1.0.0)** | **pas sur PyPI** (`pip install git+https://github.com/mistralai/timestral.git`) ; dépendances `torch>=2.4`, `numpy`, `safetensors`, `huggingface_hub` | `TimestralForecaster.from_pretrained("mistralai/Timestral-{5M,28M,123M,209M}", device="cpu", max_context=None, per_shot_horizon_patches=32, revision=...)` ; `forecast(context (T,)/(B,T)/(B,V,T), prediction_length, observation_mask=, past_covariates=(B,C,T), known_covariates=(B,C,T+H), categorical_*=, symmetric_averaging=True, clip_nonnegative=True)` → `Forecast.quantiles (B,V,H,Q)`, `.median`, `.quantile(q)` (niveau natif exact seulement) ; 21 niveaux fixes (0,01 ; 0,05 ; 0,1 … 0,95 ; 0,99) ; NaN = manquant ; **un seul tableau par appel** (même `T` pour toutes les séries) ; contexte ≤ 32 768 (les plus récents gardés) ; blocs de 1 024 pas déroulés au-delà. Code Apache-2.0, poids « Modified MIT » avec clause d'usage. |
| t0 (The Forecasting Company) — **vérifié le 2026-11-27 (0.6.1, 2026-11-09)** | PyPI `tfc-t0[torch]>=0.6.1` ; Python ≥ 3.11, `torch>=2.4`, `huggingface-hub>=1` | `T0Forecaster.from_pretrained("theforecastingcompany/t0-gamma")` ; `predict(context (B,T)/(B,V,T), horizon, quantile_levels, mask=, group_ids=, future_covariates=(B,F,T+H))` → `.quantiles (B,H,Q)`, `.median` ; tout niveau de ]0, 1[ (interpolation entre les 21 niveaux appris, queues exponentielles) ; covariables passées = variables supplémentaires du contexte ; `group_ids` incompatible avec `future_covariates` ; `batch_series([...])` → contexte aligné à droite, masque `PAD`, `group_ids` ; NaN = manquant ; 1 024 pas par passe, déroulé au-delà ; Apache-2.0 ; ~256 M paramètres. |
| autogluon (`autogluon.timeseries`, `autogluon.tabular`) — **vérifié le 2026-11-26** | PyPI, 1.6.3 (2026-09-18) | `autogluon.common` déclare `pandas>=2.0,<2.4`, `numpy<2.6`, `scikit-learn<1.10`, Python `<3.14` ; même plafond pandas sur `master` ; support de pandas 3 suivi par le ticket #5730 (jalon « 2.0 Release », ouvert). **Non installable avec `pandas>=3`** : adapter reporté (§ 5.6). `TimeSeriesPredictor(prediction_length, target, known_covariates_names, eval_metric, quantile_levels, path, ...)` fixe l'horizon **au constructeur** ; `fit(TimeSeriesDataFrame, presets, time_limit, ...)` ; `predict(data, known_covariates=...)` ; données longues (`item_id`, `timestamp`) ; modèles écrits sur disque (`path`). |
| mapie — **vérifié le 2026-11-26** | PyPI, 1.6.0 (2026-10-19) | `numpy>=1.24.1`, `scikit-learn>=1.4`, `scipy>=1.10`, pandas optionnel. `TimeSeriesRegressor` : `method` ∈ {`"enbpi"`, `"aci"`} seulement (γ d'ACI fixé à la main) ; **pas d'AgACI** (vérifié dans le code installé). Garde les DataFrames (`_safe_indexing`) et convertit les prédictions en `ndarray`. AgACI (Zaffran et al., 2022) agrège les bornes de plusieurs ACI de γ différents par agrégation d'experts en ligne (BOA, perte pinball) : réalisable plus tard sur l'agrégation native (Q12). |

**Versions retenues en session 1** (Python 3.13, pandas 3.0.6, scikit-learn 1.9.1, numpy 2.3.2 ;
résolution `uv lock` sans changement des versions de `main`) :

| Paquet | Version | Compatibilité |
|---|---|---|
| sktime | 1.3.0 | pandas 3 depuis 1.2.0, pandas 3.1 depuis 1.3.0 ; plafonne `pandas<3.2`, `numpy<2.6`, `scikit-learn<1.10`, `joblib<1.7` |
| darts | 0.48.1 | pandas 3 supporté ; **bug connu avec pandas 3.1** (séries journalières → `TypeError`), correctif non publié ; xgboost / statsforecast hors du cœur de darts |
| hierarchicalforecast | 1.5.3 | roues cp313 ; pas de dépendance à statsforecast |
| utilsforecast | 0.2.17 | support de pandas 3 à partir de cette version (plancher déclaré dans l'extra) |
| opera (**abandonné, AD-18 ; retiré des dépendances en session 10**) | commit `9f82033` (`develop`, 2024-11-06) | pur Python, aucune écriture en place sur les entrées ; `scipy` importé non déclaré (déclaré dans l'extra) ; dépôt sans fichier LICENSE (setup.py : MIT) ; plus maintenu depuis 2024 |
| xgboost | 3.4.1 | groupe `notebooks` |
| tslearn | 0.10.0 | groupe `notebooks` |
| statsforecast 2.1.1, mlforecast 1.1.0 | — | exigent `pandas<3` : non installables, non déclarés (aucun usage) |

Il n'existe **aucune API commune** à ces modèles de fondation : chaque famille a son propre paquet,
son propre format d'entrée (DataFrame long, `ndarray`, tenseurs) et sa propre gestion des
covariables. `transformers.AutoModel` n'en couvre aucune de façon uniforme.

---

## 3. Contrat commun de tous les adapters

### 3.1 Classe

- `class XxxAdapter(RegressorMixin, BaseEstimator)` (mixins avant `BaseEstimator`).
- `__init__` **ne fait que stocker** ses arguments, sous le même nom, sans validation ni attribut
  dérivé (convention sklearn, requise par `clone` et `GridSearchCV`).
- L'objet tiers passé au constructeur (modèle, forecaster, réconciliateurs, experts) n'est
  **jamais modifié**. `fit` en fait une copie ajustée dans un attribut à suffixe `_`
  (`forecaster_`, `model_`, `hrec_`, `mixture_`, `experts_`) : `sklearn.base.clone` quand l'objet
  implémente `get_params`, sinon reconstruction depuis ses paramètres ou `copy.deepcopy`.
- `get_params` / `set_params` : ceux de `BaseEstimator` dès que l'objet tiers expose
  `get_params` / `set_params` (sktime, estimateurs sklearn, réconciliateurs à vérifier). Une
  surcharge n'existe que si l'objet tiers ne les expose pas (darts), et elle doit respecter
  `clone(est).get_params() == est.get_params()` (au sens de `clone`) et le préfixe `<param>__`.
- Les composites de plusieurs estimateurs (experts de l'agrégation en ligne) exposent leurs
  paramètres imbriqués sous la forme `experts__<nom>__<param>` ; `_BaseComposition` les nommerait
  `<nom>__<param>` (comme `StackingRegressor`) : `OnlineAggregator` surcharge donc `get_params` /
  `set_params` (AD-22).
- Attributs ajustés : `n_features_in_`, `feature_names_in_` (quand `X` a des colonnes), plus les
  attributs propres à l'adapter. `check_is_fitted` fonctionne sans attribut posé dans `__init__`.
- `__sklearn_tags__` surchargé si nécessaire (par exemple `requires_y`, `allow_nan`,
  `input_tags.two_d_array`) ; pas de `_more_tags` (retiré de sklearn 1.9).

### 3.2 Données en entrée

- `X` : `pandas.DataFrame` ; `y` : `pandas.Series` (ou `DataFrame` pour une cible multiple).
- Index : `DatetimeIndex` (série temporelle) ou `MultiIndex` dont **le dernier niveau est
  temporel** et les précédents identifient l'entité (panel) — même convention que le reste de
  `tsforecast`. Les splitters de `tsforecast.crossvals` et `sklearn._safe_indexing` conservent ces
  index : l'adapter les lit, il ne les reconstruit pas.
- La validation et l'analyse d'index **réutilisent** `tsforecast` (`tsforecast.panel.utils` :
  `is_panel_data`, `get_entity_levels`, `get_entity_codes`, `check_entity_labels` ;
  `tsforecast.utils.frequency` : détection de fréquence ; `tsforecast.utils.validation`). Aucune
  réimplémentation locale.
- Les conventions pandas 3 de `main` s'appliquent (Copy-on-Write, unités de dates, dtype `str`,
  alias de fréquence actuels, clés composites par codes entiers, jamais de `apply(axis=1)`).
- Un adapter accepte des lignes dans n'importe quel ordre et des panels déséquilibrés. Une
  grille temporelle irrégulière n'est acceptée que si le modèle tiers ne l'exige pas ; sinon
  `ValueError` explicite.

### 3.3 Sorties

- `predict(X)` renvoie un `numpy.ndarray` de forme `(n_samples,)` (ou `(n_samples, n_targets)`
  pour une cible multiple), **dans l'ordre des lignes de `X`**, comme un régresseur sklearn. C'est
  ce qui rend `cross_val_predict`, les scorers et `XYPipeline` corrects sans hypothèse sur l'index.
- Une méthode supplémentaire peut renvoyer une sortie riche (`predict_quantiles`, `reconcile`,
  `predict_frame`), toujours réalignée sur l'index de `X`.
- Une ligne de `X` qu'on ne sait pas prédire (entité inconnue d'un modèle local, période
  d'agrégation incomplète) lève une `ValueError` explicite par défaut ; aucun remplissage
  silencieux (ni `0`, ni `NaN` muet).
- `score` : celui de `RegressorMixin` (R²) suffit dès que `predict` respecte ce contrat.

### 3.4 Dépendances optionnelles

- Aucun module d'adapter n'importe de paquet tiers **au niveau module** : import différé dans les
  méthodes via un utilitaire commun qui lève une `ImportError` indiquant l'extra à installer.
- `tsforecast/adapters/__init__.py` importe donc directement ses classes (plus de `__getattr__`
  qui masque les erreurs) ; `import tsforecast.adapters` fonctionne sans aucune dépendance
  optionnelle.

### 3.5 Conventions de code de `main`

Toutes les règles du `CLAUDE.md` de `main` s'appliquent : commentaires internes en français
(formulations nominales), docstrings Google en anglais, en-tête de module en anglais, doctests
exécutables, autoportance, pas d'historique, constantes réutilisées depuis
`tsforecast/utils/_constants.py`, fichiers de tests ≤ ~1 200 lignes, vérification
`python scripts/check_conventions.py --rule all`.

Doctests d'un adapter : données minuscules et déterministes ; ils ne sont collectés que si le
paquet tiers est installé (voir § 7).

---

## 4. Socle commun : `tsforecast/adapters/_base.py`

Module privé partagé par tous les adapters (fixé en session 2, AD-08). Il réutilise
`tsforecast.panel.utils` (`check_entity_labels`, `get_entity_codes`),
`tsforecast.utils.frequency` (`detect_index_frequency`, `to_pandas_freq`) et
`sklearn.utils.validation.validate_data` :

- `import_optional(module, extra)` : import différé ; seule l'absence du module demandé (ou d'un
  paquet parent) devient une `ImportError` citant `pip install 'ts-forecast[<extra>]'`, toute autre
  erreur (dépendance transitive, paquet cassé) remonte telle quelle.
- `TemporalLayout` (dataclass figée) : `entity_names`, `time_name`, `entities` (clés tuple, ordre
  de première apparition ; une série = l'entité `()`), `codes` (code entier par ligne, lecture
  seule), `dates` (unité de l'entrée), `freq` (alias pandas complet ou `None`) ; propriétés
  `is_panel`, `n_entities`, `unit`. `from_index(index, *, freq=None, detect_freq=True)` refuse les
  dates non `datetime64` (dont `Period` : `to_timestamp()`), `NaT`, étiquettes d'entité manquantes,
  couples (entité, date) dupliqués et entités sur des grilles différentes ; `last_dates()` ;
  `match_entities(other)` (code de l'entité dans `self`, `-1` si inconnue). Au `predict`, passer la
  fréquence du `fit` (`freq=...`) : un test à une ligne par entité reste analysable.
- `steps_between(start, end, freq)` : pas de grille `end - start` élément par élément (grille
  `date_range` ancrée sur la première date + `get_indexer` ; multiplicateurs, ancrages, jours
  ouvrés ; date hors grille → `ValueError`).
- `restore_row_order(values, source_index, target_index)` : réalignement par `get_indexer`
  (dates comparées comme instants), ligne manquante ou source dupliquée → `ValueError`.
- `check_adapter_input(estimator, X, y=None, *, reset, freq=None, detect_freq=True)` : `X`
  `DataFrame`, `validate_data(..., skip_check_array=True)` (pose ou contrôle `n_features_in_` /
  `feature_names_in_`), `y` aligné sur `X.index` (un tableau y est replacé), renvoie
  `(X, y, layout)`.
- Découpage par entité : `tsforecast.panel.utils.iter_entity_blocks` (pas de doublon dans `_base`).

Le socle est une boîte à outils, pas une classe mère obligatoire. Une classe mère commune
(`_BaseAdapter`) n'est introduite que si au moins trois adapters partagent la même logique
`fit` / `predict`.

---

## 5. Spécifications par adapter

### 5.1 `SktimeAdapter`

- Paramètres : `forecaster`. Pas de `fh` utilisateur : l'horizon absolu est construit au `predict`
  à partir des dates des lignes de `X` (`ForecastingHorizon(dates, is_relative=False)`).
- Panel : format `pd-multiindex` / hiérarchique natif de sktime (vectorisation par entité gérée par
  sktime) ; `X` et `y` convertis de la même façon.
- Usage recommandé pour le paradigme « `y` depuis `X` » : `YfromX(regressor)` et les forecasters
  qui exploitent `X` ; un forecaster purement autorégressif (`NaiveForecaster`) reste accepté et
  prévoit à travers le `gap` à partir de l'historique d'entraînement.
- Contrôler que sktime ne réclame pas `X` sur les dates du `gap` (lignes absentes de `X_test`) ; si
  c'est le cas, le documenter et lever une erreur claire plutôt que d'inventer des valeurs.
- **Réalisé en session 3 (AD-11)** : tags sktime lus sur le forecaster —
  `capability:exogenous = False` → `X` jamais transmis ; `capability:non_contiguous_X = False`
  (`SARIMAX`) → `ValueError` si `X` n'a pas toutes les dates entre la fin d'entraînement et la date
  prévue ; `requires-fh-in-fit` → `ValueError` au `fit` ; trous dans l'entraînement : `y` réindexé
  avec `NaN` pour un forecaster sans `X`, lignes telles quelles sinon, `ValueError` si
  `capability:missing_values = False`. Prévision : un appel sktime quand `X` est ignoré ou couvre
  toutes les entités ajustées à toutes les dates de test ; sinon un appel par entité via la table
  publique `forecasters_` des forecasters vectorisés ; sinon (forecaster à support panel natif) `ValueError`.
  Sur un panel, sktime ajuste un forecaster local par entité (`YfromX` = une régression par entité).

### 5.2 `DartsAdapter`

**Réalisé en session 4 (AD-12)** : aucun paramètre forcé, aucun avertissement ; le mode est déduit
de la configuration du modèle (`mode_`) ; pas de covariable statique.

- Paramètre : `model` (modèle darts non ajusté, jamais modifié) ; `fit` en construit des copies
  par `untrained_model()`.
- `'row'` : modèle global dont `extreme_lags == (None, 0, None, None, 0, 0, 0)` (pas de lag de
  cible ni de covariable passée, `lags_future_covariates=[0]`, `output_chunk_length=1`,
  `output_chunk_shift=0`). Ajusté sur une `TimeSeries` par **suite contiguë** de lignes de chaque
  entité (`X` en covariables futures ; trous → segments, neutre). Au `predict`, chaque suite
  contiguë de lignes de `X` est prévue depuis une série factice d'un pas, remplie de `NaN` et
  finissant juste avant elle (darts impose une série pour situer la prévision, mais n'en lit pas les
  valeurs sans lag de cible) ; un appel darts par longueur de suite. Prédiction indépendante du gap
  (prouvée par la batterie et par un test ligne seule = lot), lignes in-sample et entités inconnues
  prédites. Encodeurs `position` refusés (`ValueError`).
- `'global'` : autre modèle global (lags de cible, ensembles ; torch par ce chemin générique, non
  testé). Une `TimeSeries` par entité ; `X` en covariables passées / futures selon les lags du
  modèle. `'local'` : `LocalForecastingModel`, une copie par entité (`entity_models_`), `X` en
  covariables futures si le modèle les accepte. Les deux modes extrapolent : ligne `k` pas après la
  dernière date d'entraînement de l'entité = `k`-ième pas prévu ; trous d'entraînement, entité
  inconnue, ligne non postérieure à l'entraînement, covariables absentes sur le gap → `ValueError`
  explicite (aucune valeur inventée).
- Ordre canonique des séries (étiquettes d'entité puis dates) : un modèle sensible à l'ordre des
  échantillons (forêt aléatoire) ne dépend pas de l'ordre des lignes.
- Classifieurs darts et objets non darts → `TypeError`.
- `get_params` / `set_params` personnalisés : `model__<clé de model_params>` et
  `model__<clé>__<param>` d'un estimateur sklearn imbriqué ; `set_params` reconstruit le modèle
  seulement si une valeur change (aller-retour stable).

### 5.3 `HierarchicalForecastAdapter`

**Réalisé en session 6 (AD-13, AD-14)** pour la cross-section, **en session 7 (AD-16, AD-17)** pour
le temporel et le cross-temporel.

**Rôle** : réconciliateur. `X` contient les **prévisions de base** (une colonne par modèle de base)
et, éventuellement, des colonnes ignorées ; `y` les réalisations. `fit` construit la structure (`S`,
*tags*) de chaque copie de la hiérarchie et mémorise les lignes d'entraînement ; `predict` réconcilie
date par date les prévisions de base des lignes de `X` (prédiction indépendante du gap).

**Code** : sous-paquet `tsforecast/adapters/hierarchicalforecast/` — `adapter.py` (classe),
`_structure.py` (`parse_hierarchy`, `HierarchySpec`, `Structure`, `tree_structure` ; numpy et pandas
seuls), `_temporal.py` (`parse_temporal_hierarchy`, `TemporalSpec`, `temporal_structure`,
`cross_structure`, `TemporalCalendar`, `read_temporal_index`, `detect_fine_frequency` ; numpy et
pandas seuls), `splits.py` (`period_splits`), `reshape.py` (`stack_nodes`, `unstack_nodes`),
`__init__.py` (réexports, repris par `tsforecast.adapters`).

**API**

```python
HierarchicalForecastAdapter(hierarchy=None, reconciler=None, *, node_level=None,
                            aggregate_label="Total", model_combination=None,
                            forecast_columns=None, temporal_hierarchy=None,
                            temporal_level=None, temporal_origin=None)
fit(X, y) ; predict(X) -> ndarray (n,) ; score (R² de RegressorMixin)
reconcile(X, reconcilers=None, *, level=None, intervals_method="normality", num_samples=-1, seed=0)
diagnose(X, reconcilers=None, *, atol=1e-6) ; summing_matrix(copy=None)
stack_nodes(data, node_level="node") ; unstack_nodes(data, node_level="node")
period_splits(X, temporal_hierarchy, *, n_splits=2, test_periods=1, gap=0,
              temporal_level=None, temporal_origin=None) -> [(train, test), ...]
```

Attributs ajustés : `reconciler_` (copie ; `BottomUp()` si `reconciler=None`), `hierarchy_levels_`
(noms des niveaux, du plus agrégé au plus fin : clés d'un `model_combination` dict, valeurs possibles
de `middle_level`), `copies_` (`((),)` sans copie), `forecast_columns_`, `n_features_in_`,
`feature_names_in_`, `temporal_freq_` (grille fine d'une hiérarchie temporelle, `None` sinon) ;
privés : `_design` (mode, hiérarchies, calendrier, structure temporelle, noms des niveaux et des
copies), `_copies` (structure réconciliée, structure cross-section, positions des couples
cross-temporels, lignes d'entraînement au format hierarchicalforecast, nœuds sans ligne,
équilibre), `_shared` (structures d'une copie inconnue : arbre et/ou temporel), `_combination`,
`_intervals`.

**Format** : long côté sklearn. Une ligne par (copie, nœud, date) ; la date est le dernier niveau
d'index ; `y` est une `Series` 1-D sur **tous** les nœuds (feuilles et agrégats) ; lignes dans
n'importe quel ordre ; aucune grille régulière exigée (`check_adapter_input(..., detect_freq=False)`).

**Deux formes de `hierarchy`**

- (a) liste de listes de niveaux d'index, comme la `spec` de hierarchicalforecast
  (`[['country'], ['country', 'region']]`), triée par longueur ; l'entrée la plus longue (niveau le
  plus fin) contient tous les niveaux ; structures groupées acceptées. Une ligne d'agrégat porte
  `aggregate_label` (défaut `"Total"`, Q6) sur les niveaux qu'elle agrège ; son niveau est l'entrée
  formée de ses niveaux différents de l'étiquette (aucune entrée → `ValueError`). Nœuds **appris au
  fit**, par copie : feuilles observées, agrégats impliqués par projection sur chaque entrée (une
  ligne d'agrégat sans feuille → `ValueError`). Noms de niveaux : `"country"`, `"country/region"`.
- (b) dict `{parent: [enfants]}` sur les valeurs du niveau `node_level` (nom ou position ; défaut :
  le niveau juste avant la date) : arbre ou forêt, un parent par enfant, sans cycle, enfants uniques
  et non vides. Niveaux par profondeur : `"depth_0"`, `"depth_1"`... Feuilles éventuellement à des
  profondeurs différentes (`a = b + c`, `c = d + e`). Un nœud de `X` absent de l'arbre → `ValueError`.
- Données larges (une colonne par nœud, `value_a..value_e`) : `stack_nodes` les met au format long
  (niveau de nœuds inséré avant la date ; colonnes `(variable, nœud)` → une colonne par variable),
  `unstack_nodes` revient au format large.

**Séries temporelles et panels** : les niveaux non date hors hiérarchie sont des **niveaux de copie** ;
chaque copie est une hiérarchie indépendante, réconciliée par un appel séparé avec ses propres
statistiques insample (panels déséquilibrés naturels). Série temporelle hiérarchique = une copie `()`
(index `(node, date)` ou `(country, region, date)`) ; « les entités forment la hiérarchie » = forme
(a) sans niveau de copie ; copies indépendantes = niveau(x) d'entité en tête. Copie inconnue au
`predict` : réconciliée si sa structure est connue sans données (forme (b)) et qu'aucune donnée
insample n'est requise ; sinon `ValueError`. Nœud inconnu d'une copie connue → `ValueError`.

**Structure** (`Structure`) : `S` dense `float64`, agrégats (plus agrégés d'abord) puis feuilles
(bloc identité) ; `unique_id` chaînes `"0".."n-1"` ; ordre déterministe (étiquettes triées quand
elles sont comparables), indépendant de l'ordre des lignes ; mapping ligne → nœud calculé une fois
par combinaison distincte (`TemporalLayout.entities`) puis diffusé par `layout.codes`.
`summing_matrix(copy)` l'expose, étiquetée par les nœuds.

**Colonnes et multi-modèle**

- `forecast_columns=None` : toutes les colonnes sauf les colonnes d'intervalle `<col>-lo-<niveau>` /
  `<col>-hi-<niveau>` d'une autre colonne ; une liste : ces colonnes seules, les autres ignorées.
- **Exogènes** : aucun rôle dans la réconciliation (l'agrégation `exog_vars` de hierarchicalforecast
  sert à construire les jeux des modèles de base, en amont) ; pas de paramètre `exog_vars` (Q11).
- `model_combination` : `None` (exactement une colonne), `"mean"` (moyenne ligne à ligne, comme le
  tutoriel *multi-model*), dict `{niveau: colonne}` couvrant tous les `hierarchy_levels_` (un modèle
  par niveau ; colonnes non choisies ignorées). La combinaison s'appelle `"combined"` dans
  `reconcile()` / `diagnose()` (colonne de base de ce nom refusée) ; elle est aussi calculée sur
  `X_train` pour l'insample.
- Noms internes `model_<i>` / `model_combined` vers hierarchicalforecast, renommés en sortie.

**Réconciliateurs insample** : `fit` mémorise par copie `(unique_id, ds, y, modèles)` des lignes à
cible finie (ordre canonique) ; ces lignes ne sont passées à `reconcile` que si `reconciler.insample`
ou pour des intervalles `bootstrap` / `permbu` / `conformal` ; tout nœud de la copie doit alors avoir
une ligne (`ValueError` au `fit`). `X_train` doit contenir des prévisions **hors échantillon**
produites pli par pli avec un splitter de `tsforecast` (`cross_validate(..., return_estimator=True,
return_indices=True)` puis `predict` de chaque estimateur sur ses lignes de test) :
`cross_val_predict` refuse les splitters temporels (ils ne partitionnent pas les lignes).

**Sorties**

- `predict` : moyenne réconciliée par `reconciler` (Q5 ; `GridSearchCV` :
  `param_grid={'reconciler': [BottomUp(), MinTrace(method='ols')]}`) de la combinaison, ou de la
  seule colonne.
- `reconcile(X, reconcilers=None, ...)` : `DataFrame` indexé comme `X` (même ordre, dates dans
  l'unité d'entrée), colonnes `<modèle>/<réconciliateur>` (nommage hierarchicalforecast) pour chaque
  colonne de base et `combined`, × chaque réconciliateur ; avec `level` : `-lo-<niveau>` /
  `-hi-<niveau>`, et `-sample-<i>` si `num_samples > 0`. `normality` exige les colonnes d'intervalle
  du modèle (choisies niveau par niveau pour un dict ; refus pour `"mean"`) ; `bootstrap` exige plus
  de dates d'entraînement que de dates à réconcilier (`ValueError` explicite).
- `diagnose(X, reconcilers=None, atol=1e-6)` : diagnostics de hierarchicalforecast (`level`,
  `metric`, une colonne par `<modèle>/<réconciliateur>`), niveaux de copie en tête sur un panel.
- `level`, `intervals_method`, `num_samples`, `seed` et les diagnostics sont des arguments de ces
  méthodes, pas des paramètres du constructeur ; `is_balanced` est calculé en interne.

**Invariants** : `reconcile` de hierarchicalforecast est appelé par copie, sur un objet
`HierarchicalReconciliation` local et des **copies profondes** des réconciliateurs (et une copie de
`level`) : ni le réconciliateur du constructeur, ni `reconciler_`, ni aucun attribut ajusté ne sont
modifiés par `predict`, `reconcile` ou `diagnose`. Sortie réalignée par clés `(unique_id, ds)`.

**Erreurs explicites** : `fit` — `hierarchy` absente ou mal formée, niveau inconnu, `node_level`
avec une liste, `aggregate_label` manquant, `y` non 1-D, colonnes de base invalides ou non
numériques, `model_combination` invalide, `reconciler` non `HReconciler` (`TypeError`), réconciliateur
strict (`TopDown`, `MiddleOut`) sur structure groupée ou feuilles hors du niveau le plus fin,
`middle_level` inconnu, nœud sans ligne pour un réconciliateur insample. `predict` — nœud ou copie
inconnus, date sans tous les nœuds de sa copie, prévision de base non finie, colonnes différentes du
fit. Modes temporels, en plus : `temporal_hierarchy` mal formée, `temporal_level` /
`temporal_origin` sans hiérarchie temporelle ou invalides, `node_level` sans `hierarchy`, étiquette
temporelle inconnue, agrégat mal daté, grille fine indétectable ou ne pavant pas les périodes,
réconciliateur ou intervalles insample (fit, `reconcile`, `diagnose`) ; au predict, période
incomplète.

**Modes temporel et cross-temporel (session 7, AD-16, AD-17)**

- **Modes** (déduits au fit) : cross-section (`temporal_hierarchy=None`), **temporel**
  (`hierarchy=None`, `temporal_hierarchy={...}` : tout niveau non temporel et non date est un niveau
  de copie, une série = une copie), **cross-temporel** (les deux).
- **Index** : `(copie..., nœud..., niveau temporel, date)`. `temporal_level` (nom ou position ;
  défaut : niveau juste avant la date) porte le nom du niveau temporel de chaque ligne (clés de
  `temporal_hierarchy`) ; `hierarchy` / `node_level` sont lus sur les autres niveaux (`node_level`
  par défaut : niveau précédant le niveau temporel). Étiquette inconnue → `ValueError`.
- **`temporal_hierarchy`** (format hierarchicalforecast) : `{nom: nombre de pas fins}`, entiers
  positifs distincts, un niveau à `1` (grille fine), chaque valeur divise la plus grande `m`
  (niveaux non emboîtés entre eux admis : 12/4/3/1 → structure groupée). Niveaux triés du plus
  agrégé au plus fin ; `hierarchy_levels_` = noms temporels (temporel) ou `'<cs>//<te>'`
  (cross-temporel, convention `get_cross_temporal_tags`) : clés de `model_combination`.
- **Grille fine et calendrier** : fréquence détectée au fit sur les seules lignes du niveau fin
  (`temporal_freq_`, réutilisée au predict ; indétectable → `ValueError`). Période du niveau le plus
  agrégé = fréquence fine × `m` (`'12MS'`, `'4QE-DEC'`, `'24h'`) alignée par
  `get_period_boundaries` sur l'époque Unix (années civiles pour `MS`/`ME`×12 et `QS`/`QE`×4, jours
  pour `h`×24) ou sur `temporal_origin` (exercices fiscaux, semaines du lundi, périodes d'origines).
  Une période doit contenir exactement `m` dates de la grille fine (`B`×5, `2MS` déphasé →
  `ValueError`). Nœud `j` d'un niveau de `n` pas = pas `[j·n, (j+1)·n)` de la période, nommé
  `'<niveau>-<j+1>'`.
- **Datation** (Q7, Q10) : grille étiquetée par le début (`S`, et `D`, `h`, `B`, `min`) → premier
  pas fin du nœud ; par la fin (`E`, et `W-<jour>`) → dernier pas fin. Agrégat daté autrement →
  `ValueError` (au fit comme au predict). Contrairement à `aggregate_temporal` (périodes comptées
  depuis la première date de chaque série du jeu passé, agrégat daté par sa dernière date), les
  périodes sont calendaires et communes à toutes les séries : pas d'`aggregation_type`.
- **Moteur** : colonne de réconciliation = (copie, début de période du niveau le plus agrégé) au lieu
  de la date ; `S_te` d'**une** période, fixée au fit. Pour tout réconciliateur non insample
  (`BottomUp`, `MinTrace` `ols` / `wls_struct`, `TopDown` `forecast_proportions`), `P` ne dépend que
  de `S` et la matrice d'un horizon de `p` périodes est `I_p ⊗ S_te` : réconcilier période par
  période est identique, aucun `S` propre à l'horizon, rien recalculé ni modifié au predict. Appel à
  `reconcile` en mode non temporel (`unique_id` = nœud, `ds` = début de période), comme la
  cross-section.
- **Périodes complètes** : au predict, chaque période de chaque copie contient tous les nœuds
  (temporels, ou cs × temporels) → sinon `ValueError` (« incomplete period »). Au fit, aucune
  exigence (aucune donnée insample lue ; `y` validé mais non lu).
- **Insample refusés** : réconciliateur `insample` (`mint_shrink`, `wls_var`, `ERM`...) au fit et
  dans `reconcile()` / `diagnose()` ; intervalles `bootstrap` / `permbu` / `conformal` → `ValueError`.
  `normality` fonctionne (colonnes `-lo-` / `-hi-` de `X`).
- **Splitters** : `PanelOutOfSampleSplit` compte `test_size` lignes **par entité** depuis une date ;
  ici entité = (nœud, niveau temporel) : un niveau annuel a une ligne par an et un agrégat daté en
  fin de période n'a pas la date de début → plis incomplets, refusés au predict. `period_splits`
  découpe `n_splits` blocs adjacents de `test_periods` périodes complètes finissant à la dernière
  période de `X` (même calendrier que l'adapter) ; entraînement = lignes dont le nœud se termine au
  moins `gap + 1` pas fins avant le test (un agrégat qui chevauche le gap n'entre pas). Liste
  `[(train, test), ...]` pour `cv=` de `cross_val_score` / `GridSearchCV`.
- **Lignes indexées par origine** : l'agrégation porte sur les dates de l'index, donc sur les
  origines ; la ligne « année » somme les cibles des origines de l'année. Pour des années **cibles**
  civiles avec un horizon de `h` pas : `temporal_origin` = début d'année − `h` pas (prouvé par un
  test : mêmes prévisions qu'avec l'index par date cible), ou indexation par date cible en amont.
- **Cross-temporel** : pas de méthode native en 1.5.3 (`get_cross_temporal_tags` ne fait que croiser
  les noms ; le tutoriel fait une passe temporelle **sur les prévisions de base**, cohérente dans la
  seule dimension temporelle). Une passe unique sur `S_ct = S_cs ⊗ S_te` par copie (lignes
  réordonnées : agrégats groupés par niveau croisé, bloc identité `(feuille cs, pas fin)` en bas) :
  `ỹ = S_ct P ŷ` vérifie toutes les contraintes des deux dimensions. Pour `BottomUp`, `ols`,
  `wls_struct`, `P_ct = P_cs ⊗ P_te` : deux passes enchaînées (dans un ordre ou l'autre) donnent le
  même résultat (test numérique pour `ols`) ; pour `MinTrace(nonnegative=True)` (programme
  quadratique) elles ne commutent pas et seule la passe unique est cohérente. `TopDown` /
  `MiddleOut` refusés (un nœud `(feuille, trimestre)` a deux parents) ; accepté en temporel seul sur
  des niveaux en chaîne (12/3/1). Copie inconnue : acceptée pour un arbre ou en temporel seul,
  refusée pour une hiérarchie de niveaux d'index.
- `summing_matrix(copy)` : `S_te` (index des nœuds temporels, nommé comme le niveau temporel) ou
  `S_ct` (`MultiIndex` nœud cs + nœud temporel), d'une période.

### 5.4 Agrégation d'experts en ligne native (BOA, MLpol)

**Décision du 2026-11-26 (AD-18)** : plus d'`OperaAdapter` ni de dépendance à opera (non publié sur
PyPI, inactif depuis 2024). Les algorithmes sont implémentés dans `tsforecast`. AgACI de MAPIE n'en
tient pas lieu : il agrège des bornes d'intervalles d'ACI, pas des prévisions de modèles hétérogènes,
et MAPIE 1.6.0 ne l'implémente pas (§ 2.3). **Implémenté en session 10 (AD-22).**

**Emplacement** : ce n'est pas un adapter (aucun paquet tiers) : sous-paquet `tsforecast/ensemble/`,
qui réexporte `OnlineAggregator` et `Loss`.
- `_algorithms.py` : noyau pur, numpy seul. `Loss` / `make_loss` (`"mse"`, `"mae"`, `"pinball"` de
  niveau `quantile`, ou `Loss(value, gradient)`) ; `excess_losses(weights, experts, y, loss,
  gradient)` : excès de perte de chaque expert sur le mélange, linéarisé
  (`loss'(p, y)·(x_k − p)`) ou non (`loss(x_k, y) − loss(p, y)`), **moyenné sur les lignes du tour** ;
  états figés `BOAState`, `MLpolState` ; registre `ALGORITHMS = {nom: (init, weights, update)}` de
  fonctions pures `(état, excès) -> état` ; `run(algorithme, état, experts, y, tours, perte, gradient)`.
  Le même noyau servira à AgACI (perte pinball) sans changement.
- `online.py` : `OnlineAggregator(RegressorMixin, MetaEstimatorMixin, BaseEstimator)`.

**Algorithmes** (paramètre `aggregation`), fidèles aux articles (choix de l'utilisateur, AD-22) :
- `"BOA"` — procédure entièrement adaptative de Wintenberger (2017) : `π_k ∝ η_k·exp(−η_k·L_k)·π_0k`
  (avant le premier tour : `π_0`) ; à chaque tour, étendue `E = 2^(k+1)` avec `2^k ≥ max|ℓ|`,
  `k ≥ −20` ; `L += ½(ℓ(1 + η_{t−1}ℓ) + E·1{η_{t−1}ℓ > ½})` ; `η = min(1/E, sqrt(log(1/π_0)/Σℓ²))`.
- `"MLpol"` — Gaillard, Stoltz et van Erven (2014) : `p_k ∝ η_k·(R_k)₊`, regrets `r = −ℓ`, taux à
  étendue inconnue d'opera R `1/η += r² + B_t² − B_{t−1}²` parti de `1/η = 0` (invariant d'échelle) ;
  tant qu'aucun regret n'est positif, poids `init_weights` (uniformes par défaut).
- **Écarts d'opera-python (commit `9f82033`), non reproduits** : BOA avec `E = 2^k`, cumul régularisé
  avec `η_t` au lieu de `η_{t−1}`, terme correctif `+E·1{ηr > ½}` sur le regret (signe et sens
  inversés), a priori ignoré (`log(1/K)`), compteurs démarrés à `2^-20` ; MLpol avec `1/η` démarré à
  `2^20` (dépend de l'échelle de `y`). Écart mesuré sur le jeu de référence (8 tours, 3 experts) :
  jusqu'à 0,20 sur un poids et 0,13 sur une prévision pour BOA.

**Modes** :
1. **méta-estimateur** : `experts=[("ridge", Ridge()), ...]`, paramètres imbriqués
   `experts__<nom>__<param>` (et `experts__<nom>` pour remplacer un expert). `fit` :
   1. prévisions **hors échantillon** des experts (un clone par couple expert × pli, `joblib`
      `n_jobs`). `cv` entier (défaut 5) : les `cv` derniers blocs de `n_dates // (cv + 1)` dates
      consécutives, communs aux entités, chacun prévu par des experts ajustés sur toutes les dates
      antérieures (fenêtre croissante, sans gap) — un splitter de panel de `tsforecast` compte
      `test_size` en lignes par entité, ses plis se chevauchent dès qu'il y a des trous. Ou un
      splitter fourni (lignes présentées triées entité puis date ; plis chevauchants → `ValueError`) ;
      les lignes de la première fenêtre ne sont pas agrégées ;
   2. agrégation séquentielle dans l'ordre des dates ;
   3. réajustement des experts sur tout l'entraînement (`experts_`).
2. **`experts="passthrough"`** : les colonnes de `X` sont des prévisions d'experts ; toutes les lignes
   sont agrégées.

- **Tours par date** : en `pooling="global"`, les lignes d'une même date forment un tour (mêmes poids,
  excès de perte moyen) : traiter A puis B ferait lire `y_A(t+h)` pour pondérer B à la même origine ;
  le résultat ne dépend pas de l'ordre des entités. `"per_entity"` : une agrégation par entité, plus
  la globale, toujours tenue, dont les poids prévoient une entité inconnue. Sur une série, les deux
  sont identiques (pas d'`entity_weights_`).
- `predict` : poids finaux, jamais mis à jour ; `ndarray` dans l'ordre des lignes de `X`.
- `partial_fit(X, y)` : `fit` si non ajusté ; sinon lignes toutes postérieures à `last_date_` (règle
  globale, y compris en `per_entity` : l'agrégation globale reste chronologique), experts non
  réajustés, entité nouvelle partie de l'état initial ; `fit(A)` puis `partial_fit(B)` en passthrough
  = `fit(A ∪ B)` (testé) ; algorithme changé depuis `fit` → `ValueError`.
- Attributs : `weights_` (`Series` globale), `entity_weights_` (`DataFrame` entité × expert, per_entity
  sur un panel), `weights_history_` (poids de chaque ligne agrégée avant sa cible, ordre
  chronologique), `oos_predictions_` (méta), `experts_`, `expert_names_`, `prior_`, `state_`,
  `entity_states_`, `entity_names_`, `last_date_` (unité de l'entrée).
- **Coût** : `n_experts × (n_plis + 1)` ajustements par `fit` en mode méta, × plis externes en
  validation croisée, × candidats en `GridSearchCV` (3 experts, `cv=5`, 5 plis, 10 candidats : 900).
  Moyens : moins de plis internes, `max_train_size` d'un splitter fourni, `n_jobs`, et pour régler
  l'agrégation seule, passthrough sur `oos_predictions_` (aucun réajustement ; même résultat, testé).
- **Fuite** : les poids finaux ne lisent que des cibles d'entraînement ; avec `h > 1`, l'historique
  interne des poids est un chemin en ligne idéalisé (documenté).
- **Vérification** : valeurs de référence calculées une fois avec opera-python dans un environnement
  jetable du scratchpad, recopiées en littéraux dans `tests/unit/ensemble/test__algorithms.py` (BOA
  et MLpol × `mse` / `mae` × avec / sans gradient, 8 tours, 3 experts) : MLpol par opera tel quel,
  l'état initial d'opera (`1/η = 2^20`) injecté dans le noyau ; BOA par une copie d'opera corrigée des
  trois lignes ci-dessus, `η_0 = Σℓ²_0 = 2^-20` injectés ; écart ≤ 1e-14 en relatif. Propriétés :
  tours calculés à la main, simplexe, invariance d'échelle, pinball(0,5) ≡ mae, convergence vers
  l'expert dominant, regret moyen décroissant sur une séquence alternée. Batterie de contrat (deux
  cas, `native_reference=None` : les littéraux en tiennent lieu).

### 5.5 Modèles de fondation (Hugging Face) : `FoundationModelAdapter`

**Réalisé en session 12 (AD-23)** : adapter, protocole, registre, backends `chronos` (Chronos-2 et
Chronos-Bolt) et `callable`. Timestral et t0 viendront en session 13, sans changement du
protocole.

Il n'existe pas d'API commune (§ 2.3). D'un côté, un **adapter** unique porte toute la logique
sklearn et `tsforecast` : index, panels, calcul des pas, absence de fuite, quantiles et lots. De
l'autre, des **backends** minces traduisent une requête en appel au paquet tiers ; le churn de
l'écosystème leur est confiné.

```
tsforecast/adapters/foundation/
├── __init__.py          # réexports (repris par tsforecast.adapters pour l'adapter et le registre)
├── adapter.py           # FoundationModelAdapter
├── protocol.py          # BackendCapabilities, ForecastRequest, ForecastResult, ForecastBackend, same_level
├── registry.py          # register_backend, available_backends, resolve_backend, load_backend, clear_model_cache
├── _series.py           # grilles par entité, fenêtres futures, rang des entités, niveaux de quantiles (numpy, pandas)
└── backends/
    ├── callable.py      # CallableBackend : tout modèle branché par une fonction
    ├── chronos.py       # ChronosBackend, load_chronos (Chronos-2, Bolt ; T5 refusé)
    ├── timestral.py     # session 13
    └── t0.py            # session 13
```

**Protocole** (`protocol.py`)
- `BackendCapabilities` (figée) :
  - `past_covariates`, `known_covariates` ;
  - `missing_values` : NaN acceptés dans le contexte et les covariables ;
  - `multivariate` : sinon, une série par cible ;
  - `max_context` : tronqué par l'adapter ;
  - `max_prediction_length` : au-delà, `ValueError` ;
  - `quantile_levels` : niveaux appris, ou `None` pour tout niveau de ]0, 1[ ;
  - `interpolates_quantiles` : si `False`, l'adapter demande les niveaux appris qui encadrent la
    demande et interpole linéairement.
- `ForecastRequest` (figée) : une requête = une longueur de prévision commune.
  - `context` : un tableau `(n_cibles, T_i)` par série, longueurs libres, plus ancien d'abord,
    NaN = manquant ; la dernière colonne est la dernière date d'entraînement ;
  - `prediction_length`, `quantile_levels` (croissants) ;
  - `past_covariates` : un tableau `(C_p, T_i)` par série ;
  - `known_covariates` : un tableau `(C_k, T_i + H)` par série, NaN = inconnu ;
  - noms des covariables, `freq`, `options` (les `predict_kwargs`, figés) ;
  - les tableaux sont des **vues en lecture seule** : un backend ne peut pas altérer les contextes
    mémorisés.
- `ForecastResult(quantiles)` : forme `(n_séries, n_cibles, H, n_niveaux)`, contrôlée par l'adapter
  (forme et valeurs finies, sinon `ValueError` qui nomme le backend).
- `ForecastBackend` (`Protocol`, `runtime_checkable`) : un attribut `capabilities` et une méthode
  `predict(request)`, sans état entre deux appels.
- Couverture des familles :
  - Chronos-2 : dicts ; le passé des covariables connues va dans `past_covariates`, leurs pas
    prévus dans `future_covariates` ;
  - Bolt : tenseurs 1-D ;
  - Timestral : un tableau `(B, V, T)` par appel, donc le backend remplit ou groupe par longueur ;
    `known_covariates (B, C, T+H)` telles quelles ;
  - t0 : `batch_series`, covariables connues telles quelles, covariables passées en variables du
    contexte.

**Registre et cache** (`registry.py`)
- `register_backend(name, load, *, prefixes, package, extra, replace=False)`. `load` a la forme
  `(model_id, *, device, **backend_kwargs)`.
- `chronos` est enregistré à l'import du module, sans importer chronos : préfixes
  `amazon/chronos`, `autogluon/chronos`, paquet `chronos`, extra `adapters-chronos`.
- Résolution :
  - `backend` est un nom enregistré (cas d'un chemin local) ;
  - sinon, le préfixe de `model_id` le plus long gagne, sans tenir compte de la casse ;
  - inconnu : `ValueError` qui liste les préfixes.
- `backend` peut aussi être un **objet** backend : il est utilisé tel quel, sans cache, et
  `model_id` doit valoir `None`.
- Cache du processus :
  - clé `(nom, model_id, device, backend_kwargs figés)`, verrou de chargement ;
  - valeur non hachable dans `backend_kwargs` : `ValueError` ;
  - `clear_model_cache()` libère les modèles ;
  - le paquet est contrôlé par `import_optional` **avant** le cache, donc un paquet manquant est
    signalé même si le modèle est déjà chargé.
- Interaction avec sklearn :
  - le modèle chargé n'est ni un paramètre ni un attribut ajusté. Le `fit` mémorise les arguments
    de chargement, et le `predict` relit le cache, ou recharge dans un processus neuf ;
  - `clone`, `pickle` et `return_estimator=True` ne copient donc jamais les poids (testé) ;
  - un `set_params` après le `fit` ne change pas le modèle utilisé par `predict` ;
  - avec `n_jobs=1`, un chargement sert à tous les plis et candidats de même clé (testé à travers
    `cross_val_score` et `GridSearchCV`) ;
  - avec `n_jobs>1`, une copie par processus, et les threads de torch se concurrencent :
    déconseillé (docstring).

**API**

```python
FoundationModelAdapter(model_id=None, *, backend=None, context_length=None, quantile=0.5,
                       covariates="auto", device="cpu", batch_size=None,
                       backend_kwargs=None, predict_kwargs=None)
fit(X, y) ; predict(X) -> ndarray ; predict_quantiles(X, quantile_levels=(0.1, 0.5, 0.9)) -> DataFrame
```

Attributs ajustés : `backend_name_`, `capabilities_`, `covariates_` (`'known'`, `'past'` ou
`None`), `context_length_`, `context_` (un tableau `(n_cibles, T_e)` en lecture seule par entité),
`context_covariates_`, `layout_`, `target_names_`, `n_features_in_`, `feature_names_in_`.

**Sémantique**
- `fit` : aucun entraînement.
  - Le modèle est chargé par le cache et ses capacités sont lues.
  - Grille de chaque entité, de sa première à sa dernière date d'entraînement ; trous et cibles NaN
    en NaN ; seuls les `min(context_length, max_context)` derniers pas sont gardés.
  - `ValueError` dans ces cas : grille irrégulière, entité sans valeur observée dans son contexte,
    NaN pour un modèle qui les refuse, valeurs infinies, covariables non numériques, `quantile` hors
    de la plage apprise.
  - `covariates='auto'` donne `'known'` si le modèle l'accepte, sinon `'past'`, sinon `None`, sans
    avertissement. Une valeur explicite non prise en charge lève une `ValueError`.
- `predict` : **un appel par pas**.
  - Une ligne à `k` pas (`steps_between` depuis la dernière date d'entraînement de son entité)
    reçoit le dernier pas d'une prévision de longueur `k`.
  - En mode `'known'`, cette prévision reçoit les covariables du contexte, puis les lignes de X de
    l'entité aux pas 1..k. Le gap est NaN, jamais une valeur inventée. Si le modèle refuse les NaN :
    `ValueError`.
  - Une ligne lit donc X de son origine et des lignes antérieures de son entité, **jamais d'une
    origine postérieure**.
  - Sans covariables connues, la prévision d'une ligne ne dépend pas des autres lignes de X (testé
    sur Chronos-2 : ligne seule = lot à 1e-5 près).
  - Mise en lots : groupes par `k`, entités dans l'ordre de leurs étiquettes (indépendant de l'ordre
    des lignes), lots d'au plus `batch_size` séries ; nombre d'appels `Σ_k ⌈n_k / batch_size⌉`.
  - `ValueError` dans ces cas : entité inconnue, ligne non postérieure à l'entraînement, `k >
    max_prediction_length`.
  - Plusieurs cibles : jointes si `multivariate`, sinon une série par (ligne, cible).
- Quantiles :
  - `predict` renvoie le niveau `quantile` (médiane par défaut ; aucune famille n'a de vraie
    moyenne) ;
  - `predict_quantiles(X, quantile_levels)` renvoie un `DataFrame` indexé comme X, une colonne
    `float64` par niveau (nom `quantile`), ou des colonnes `(target, quantile)` pour plusieurs
    cibles. Les niveaux sont un argument de méthode, pas du constructeur (principe d'AD-13) ;
  - niveaux uniques dans ]0, 1[ et dans la plage apprise, sinon `ValueError` (au lieu du bornage
    silencieux de Chronos) ;
  - croisements non corrigés.
- `GridSearchCV` ne règle que l'inférence : `model_id`, `context_length`, `quantile`, `covariates`,
  `device`, `backend__<param>` d'un objet backend qui expose `get_params` (`CallableBackend` :
  `backend__function`, capacités).

**Mesures sur `autogluon/chronos-2-small`, CPU, session 12**
- Premier chargement : 54 s, téléchargement compris. `predict` de 9 lignes (3 entités, pas 3 à 5,
  covariables connues) : 0,11 s.
- **Un appel unique lit le futur** : X modifié au pas 5 d'une prévision de 5 pas change le pas 3 de
  0,14. D'où l'appel par pas.
- **La longueur de prévision change les prévisions** au-delà d'un patch de sortie (16 pas) : le
  pas 3 vaut −0,502 avec `H = 3` et −0,543 avec `H = 20`.
- En mode `'known'`, prédire une ligne seule ou avec ses lignes antérieures diffère, jusqu'à 3,0 sur
  le jeu de la batterie (`y = 2·x1 − x2`, apprentissage en contexte) : c'est l'information
  disponible à l'origine de la ligne.
- Règle 11 : quantiles égaux au pipeline appelé directement, à `rtol = 1e-5` (Chronos-2 avec et
  sans covariables, Bolt).

**Limites et coût** (ce qu'il vaut mieux ne pas promettre)
- Le `fit` n'apprend rien. La validation croisée mesure un modèle figé ; régler `context_length` sur
  peu de plis sur-apprend les plis.
- Le modèle extrapole la série des lignes indexées par origine sans en voir les dates : le calendrier
  n'entre que par X. La stratégie directe (un estimateur par `h`) refait l'extrapolation de la même
  série décalée.
- En pseudo-temps réel, le contexte s'arrête `gap` pas avant l'origine ; X est l'information
  récente. Mais les lignes du gap, absentes de X, sont déclarées inconnues alors que leurs
  covariables seraient connues en temps réel.
- Pas de prédiction in-sample, ni d'entité sans historique. Les grilles irrégulières sont refusées.
- Coût d'une validation croisée : `n_plis × n_candidats × Σ_k ⌈n_k/batch⌉` passes avant, `fit`
  quasi gratuit, un chargement par processus et par clé. Un test long multiplie les appels (365
  lignes journalières → 365 appels).
- Pas d'égalité bit à bit entre périphériques, versions ou révisions de poids (`revision=` dans
  `backend_kwargs` les fige). Pas d'intervalles calibrés, pas de gain garanti par les covariables.
- Non couverts : GPU (non testé), *fine-tuning* (hors périmètre), Chronos-T5 (refusé, inférence par
  tirages), covariables catégorielles (X numérique seulement).
- Licences : Chronos Apache-2.0 (docstrings) ; Timestral (poids « Modified MIT » avec clause
  d'usage) et t0 (Apache-2.0) en session 13.

**Tests** :
- support :
  - `tests/support/forecast_backends.py` : `DriftBackend`, factice et déterministe ; ses capacités
    sont des paramètres, et il somme **toutes** les covariables connues reçues après le contexte,
    comme un modèle qui lirait le futur. Aussi `RecordingBackend` et `counting_loader` ;
  - cas `foundation_drift_case` et `foundation_chronos_case` (`adapter_cases.py`) ;
  - références `drift_reference` (calculée à la main) et `chronos_reference` (un `predict_quantiles`
    par ligne), dans `native_references.py` ;
- `tests/unit/adapters/foundation/` : `test_adapter.py` (batterie complète et scénarios à valeurs
  exactes), `test_registry.py`, `test_protocol.py`, `test__series.py`, `backends/test_callable.py`,
  `backends/test_chronos.py` ;
- `test_chronos.py` contient la correspondance hors ligne (paquets `chronos` et `torch` simulés) et
  le réel, marqué `slow` et `network` : batterie, règle 11 sur tous les niveaux, Bolt-tiny, X
  postérieur jamais lu, ligne seule = lot ;
- `tests/integration/adapters/test_foundation.py` : batterie de workflow, factice et Chronos réel ;
- couverture des modules `foundation/` : 100 %.

### 5.6 `AutoGluonAdapter` (reporté)

**Décision du 2026-11-26 (AD-19)** : adapter souhaité, mais **reporté** tant qu'AutoGluon plafonne
`pandas<2.4` (1.6.3 et `master`, § 2.3) : il ne peut pas cohabiter avec `pandas>=3`. Déclencheur :
publication d'une version d'AutoGluon acceptant pandas 3 (jalon « 2.0 Release », ticket #5730).
Les sessions A1–A2 de `ADAPTERS_PROMPTS.md` ne commencent qu'alors ; elles revérifient d'abord
l'API, qui peut avoir changé.

Points de conception connus, à trancher en plan mode à ce moment (Q13) :
- **Horizon fixé au constructeur** : `TimeSeriesPredictor(prediction_length=...)` entraîne ses
  modèles pour un nombre de pas donné, alors qu'ici le nombre de pas est déduit des index au
  `predict`. Pistes : `prediction_length` déduit au `fit` d'un paramètre de capacité
  `max_steps` (une ligne plus lointaine → `ValueError` explicite, jamais de décalage), ou un mode
  `'row'` sur `TabularPredictor` (régression ligne à ligne de `y` sur `X`, comme le mode `'row'` de
  `DartsAdapter`), qui ne pose pas la question.
- **Covariables** : `known_covariates_names` (connues sur l'horizon) contre covariables passées ;
  les lignes du `gap`, absentes de `X`, ne peuvent pas servir de covariables connues.
- **Coût et état disque** : AutoGluon entraîne de nombreux modèles (`presets`, `time_limit`) et les
  écrit sous `path` ; `clone` et `GridSearchCV` (un répertoire par pli, nettoyage), `n_jobs`,
  déterminisme (graines), durée d'une validation croisée.
- Extra `adapters-autogluon` (`autogluon.timeseries` et / ou `autogluon.tabular`), job CI séparé
  vu le poids des dépendances (torch).

---

## 6. Dépendances et packaging

- Extras (`[project.optional-dependencies]`) :
  - un par paquet : `adapters-sktime`, `adapters-darts`, `adapters-hierarchical`,
    `adapters-chronos` (session 12 : `chronos-forecasting>=2.3.2,<3`, qui tire torch, transformers
    et accelerate), puis `adapters-timestral` et `adapters-t0` (session 13) ;
  - deux agrégats (AD-23) : `adapters-light` regroupe sktime, darts et hierarchical, sans modèle de
    fondation ni torch ; `adapters-all` = `adapters-light` + `adapters-chronos`, auquel s'ajouteront
    timestral et t0 ;
  - `adapters-autogluon` plus tard (§ 5.6) ;
  - bornes `>=x,<y` comme le reste du `pyproject.toml` ;
  - dépendances Git figées sur un SHA. `allow-direct-references = true` a été retiré en session 10
    avec la seule dépendance Git et sera à remettre pour timestral. **Point à relire** : PyPI refuse
    un paquet dont une dépendance est une référence directe, ce qui bloque une publication de
    ts-forecast tant que timestral n'est pas sur PyPI ;
  - l'extra `adapters-opera` est supprimé (session 10) : l'agrégation native (AD-18, AD-22) n'a
    aucune dépendance hors du cœur ; `uv.lock` régénéré sans opera.
- `uv.lock` régénéré en session 12 :
  - ajouts : chronos-forecasting 2.3.2, torch 2.15.0, transformers 5.24.1, accelerate 1.16.0,
    huggingface-hub 2.5.0, et les bibliothèques CUDA de Linux (marqueurs de plateforme) ;
  - mises à jour imposées : click 8.2.1 → 8.5.1, idna 3.10 → 3.20, nvidia-nccl-cu13 aligné sur
    torch ;
  - aucune version de `main` changée pour pandas, numpy ou scikit-learn.
- Groupe `notebooks` (hors `default-groups`) : `xgboost`, `tslearn`, `statsforecast` /
  `mlforecast` si le notebook en a besoin, plus les extras d'adapters. `dev` et `docs` restent ceux
  de `main` (ne pas retirer `docs` de `default-groups`).
- CI : job `tests-adapters` séparé, `uv sync --group dev --extra adapters-light` depuis la
  session 12 : pas de torch ni de bibliothèques CUDA. Tests rapides uniquement : les tests de
  l'adapter de fondation passent par le backend factice et des paquets `chronos` / `torch` simulés,
  et les tests `network` ne tournent pas en CI.
- Installation locale : suivre les notes de la mémoire du projet (pas de `pip` dans `.venv`,
  attribut ReadOnly des dossiers sous OneDrive, `uv sync --no-install-project --inexact`).

---

## 7. Tests

- Arborescence : `tests/unit/adapters/<adapter>/` (paquet si > ~1 200 lignes) et
  `tests/integration/adapters/`. En-têtes de module en anglais, `tests/support` réutilisé
  (`datasets.py`, `splits.py`, `perturbations.py`).
- Chaque module de test commence par `pytest.importorskip("<paquet>")`. La collecte des doctests
  d'un adapter est ignorée quand son paquet manque : `conftest.py` **racine** (seul chargé par
  `pytest --doctest-modules tsforecast`), table module d'adapter → paquet, `find_spec` ; il porte
  aussi les exclusions transitoires (AD-10).
- **Batterie de contrat commune** : `tests/support/adapter_contract.py` (AD-09). `AdapterCase`
  décrit un adapter (`make_estimator`, `wrapped_param`, `param_grid`, `package` / `extra`,
  `supports_panel`, `local_model`, `row_regressor`, `invalid_params`, `make_data`,
  `frame_method`, tolérances) ; une fonction `check_*` par règle ; deux mixins :
  `AdapterContractTests` (règles 1–3, 7–10, à brancher sous `tests/unit/adapters/`) et
  `AdapterWorkflowContractTests` (règles 4–6, sous `tests/integration/adapters/`). Branchement :
  `pytest.importorskip("sktime")` puis
  `class TestSktimeContract(AdapterContractTests): case = AdapterCase(...)`. Jeux :
  `build_adapter_series` / `build_adapter_panel` de `tests/support/datasets.py`. La batterie est
  validée par `tests/support/test_adapter_contract.py` (adapters de référence `RowRegressorAdapter`
  et `LastValueAdapter` de `tests/support/estimators.py`, mutants détectés). Règles :
  1. `clone(est)` puis `fit` ne modifie pas l'objet tiers d'origine ; `get_params` stable ;
  2. sortie `ndarray` de longueur `len(X)`, dans l'ordre des lignes de `X` (test avec lignes
     mélangées) ;
  3. série temporelle **et** panel (équilibré et déséquilibré) ;
  4. `cross_val_score` avec `TSOutOfSampleSplit` et `PanelOutOfSampleSplit` (`gap > 0`) ;
  5. `GridSearchCV` sur au moins un paramètre imbriqué ;
  6. dans `XYPipeline` (avec `DropNaNTarget`) et dans `sklearn.pipeline.Pipeline` ;
  7. **absence de fuite** : modifier `y` après la dernière ligne d'entraînement ne change pas la
     prédiction ;
  8. prédiction indépendante du `gap` quand le modèle est un pur régresseur ligne à ligne ;
  9. erreurs explicites (entité inconnue d'un modèle local, paquet manquant, paramètre invalide) ;
  10. conventions de dates de `main` (`with_datetime_resolution` × `datetime_units`) quand
      l'adapter renvoie des dates ;
  11. **équivalence au paquet natif** (AD-20) : sur des données que le paquet tiers sait traiter
      seul (série régulière, panel équilibré, sans trou), le même modèle ajusté directement par
      l'API du paquet (horizon relatif `fh = gap + 1, ...` pour sktime, `predict(n)` pour darts,
      `aggregate` + `HierarchicalReconciliation.reconcile` pour hierarchicalforecast, pipeline du
      paquet pour un modèle de fondation) donne les mêmes prévisions que l'adapter, à
      `rtol` / `atol` près (modèles déterministes, graines fixées). Champ
      `AdapterCase.native_reference` (fonction `(X_train, y_train, X_test, fh) -> ndarray` qui appelle
      le paquet sans passer par `tsforecast`, sortie dans l'ordre des lignes de `X_test` ; `fh` =
      horizon relatif `(gap + 1, ..., gap + n_test)` des dates de test, fourni par le découpage de la
      batterie pour que la référence ne le déduise pas des dates comme l'adapter, AD-21) et champ
      `AdapterCase.native_gaps` (défaut `(0, 2)` ; `(0,)` pour un modèle qui lit `X` sur toutes les
      dates jusqu'à la prévision) ; règle sautée quand `native_reference` vaut `None`. Les situations
      propres à l'adapter (panel déséquilibré, trous, mode `'row'` de darts à travers un gap, lignes
      in-sample) n'ont pas d'équivalent natif : elles restent couvertes par des valeurs exactes
      calculées à la main.

  Session 9 (règle 11) : `check_native_equivalence(case, kind, gap)` (série et panel équilibré,
  gaps de `native_gaps`, 3 dates de test, tolérances du cas `rtol=1e-7` / `atol=1e-9`), test
  `test_native_equivalence` d'`AdapterContractTests` ; références dans
  `tests/support/native_references.py` (aucun import de `tsforecast`). Validée sur
  `RowRegressorAdapter` (régression sklearn) et `LastValueAdapter` (dernière valeur + drift × `fh`),
  mutant `ShiftedHorizonAdapter` (un pas de trop) détecté. Cas couverts :
  - batterie — sktime `NaiveForecaster()`, `YfromX(LinearRegression())` ; darts `'row'`
    `LinearRegressionModel` et `RandomForestModel` (régresseur sklearn mutualisé, lignes triées entité
    puis date, ordre des échantillons de darts), `NaiveSeasonal(K=1)` (local), `LinearRegressionModel(lags=3)`
    (global) ; hierarchicalforecast `BottomUp`, `ols` (moyenne), `mint_shrink` (un modèle par niveau,
    insample) sur l'arbre, `ols` sur les niveaux pays / régions — un `reconcile` par copie ;
  - `TestNativeEquivalence` de `test_sktime.py` — drift, `ExponentialSmoothing(trend='add')`,
    `ThetaForecaster(deseasonalize=False)`, `SARIMAX(order=(1, 0, 0))` avec `X` (gap 0) ; de
    `test_darts.py` — `ExponentialSmoothing()`, `LinearRegressionModel(lags=3, lags_past_covariates=2,
    lags_future_covariates=[0])` (gap 0) (`sktime_native_cases` / `darts_native_cases` de
    `adapter_cases.py`) ;
  - `TestNativeEquivalence` de `hierarchicalforecast/test_temporal.py` — temporel `BottomUp` / `ols`
    (`aggregate_temporal` + `reconcile(temporal=True)`), cross-temporel `BottomUp` / `ols` (arbre et
    niveaux, deux passes natives), série et panel, deux années de test ; `mint_shrink` refusé des deux
    côtés.

  Écarts d'API rencontrés (aucun n'empêche la comparaison) : `aggregate` ne construit pas un arbre à
  feuilles de profondeurs différentes sans dupliquer la feuille (`S_df` de l'arbre écrit à la main) ;
  `aggregate_temporal` date un agrégat par son dernier pas fin et compte les périodes depuis la
  première date de chaque série (comparaison par `(unique_id, temporal_id)`, fenêtre ouverte sur un
  début d'année) ; pas de cross-temporel natif (deux passes : cross-section par niveau temporel, les
  `ds` des niveaux coïncidant, puis temporel par nœud ; égales à la passe unique pour `BottomUp` /
  `ols`) ; `ThetaForecaster()` désaisonnalise en multiplicatif, refusé par statsmodels sur une cible
  non positive ; un modèle lisant `X` exige nativement `X` sur le gap. Aucun écart de prévision
  trouvé (≤ 6e-16 en relatif).

  Session 4 : `AdapterCase.handles_training_holes` (un adapter qui refuse les trous
  d'entraînement doit lever `ValueError` dans le contrôle `XYPipeline`) ; le contrôle
  `GridSearchCV` cherche sur les dates d'entraînement et prédit les dates de test postérieures.
  Session 6 : `AdapterCase.make_splitter` (défaut `default_splitter` : `TSOutOfSampleSplit` pour une
  série) ; les cas hiérarchiques prennent `PanelOutOfSampleSplit` pour tous les jeux (une série
  hiérarchique a un niveau de nœuds). Jeux `build_hierarchy_nodes` / `build_hierarchy_levels`
  (`tests/support/datasets.py`, `y` cohérent, `m1` / `m2` bruités) ; cas
  `hierarchical_{bottomup,ols,mint_shrink,levels}_case` (`tests/support/adapter_cases.py`).
  Session 7 : jeux `build_temporal_hierarchy` / `build_cross_temporal_hierarchy` (années,
  trimestres, mois ; `y` cohérent dans les deux dimensions) ; cas
  `hierarchical_{temporal,cross_temporal}_case`, soumis aux seules règles 1, 9 (paramètres
  invalides, paquet manquant) : les autres découpent des fenêtres de test sur des dates arbitraires,
  qu'un adapter temporel refuse (périodes incomplètes) ; elles sont couvertes par
  `tests/unit/adapters/hierarchicalforecast/test_temporal.py` et
  `tests/integration/adapters/test_hierarchicalforecast_temporal.py` (`period_splits`).
  Session 12 :
  - cas `foundation_drift_case` (backend factice `DriftBackend`, hors ligne) : batterie complète,
    règle 11 contre une référence calculée à la main ;
  - cas `foundation_chronos_case` (`autogluon/chronos-2-small`, marqué `slow` et `network`) :
    batterie complète, règle 11 contre `Chronos2Pipeline.predict_quantiles` appelé pas par pas,
    `rtol=1e-5` ;
  - le paquet manquant est contrôlé même modèle en cache.
- Commande locale : `.venv/Scripts/python.exe -m pytest tests/unit/adapters -q -p no:cacheprovider`.
  Tests réels des modèles de fondation : `-m "slow or network"`. Le premier lancement télécharge
  `autogluon/chronos-2-small` (environ 110 Mo) et `amazon/chronos-bolt-tiny` (environ 36 Mo) dans
  le cache Hugging Face.

---

## 8. Notebook `4 - QB - Adapters.ipynb`

- Cellule markdown de présentation en tête ; imports publics uniquement (pas de `_safe_split`).
- Section 2 (cadre de prévision) alignée sur `pseudo_real_time_forecast.md` de `main` :
  construction supervisée **par entité**, `gap = h + d - 1`, splitters de `tsforecast`.
- Un même gabarit par adapter :
  1. principe et mapping vers l'API du paquet tiers ;
  2. série temporelle ;
  3. panel ;
  4. dans une `Pipeline` / `XYPipeline` ;
  5. `cross_val_score` ;
  6. `GridSearchCV` ;
  7. limites et pièges.
- Section hierarchicalforecast : cross-section (série temporelle et panel), temporel (série et
  panel), multi-modèle, cross-temporel, `cross_val_score`, `GridSearchCV` ; contrôles numériques de
  cohérence affichés.
- Données synthétiques déterministes (graine fixée), exécution complète sur CPU en quelques
  minutes ; le notebook est exécuté de bout en bout avant de rendre la main (nbclient, cf. mémoire
  du projet) et sauvegardé avec ses sorties.

---

## 9. Questions ouvertes (à trancher en plan mode, avec l'utilisateur)

| Id | Question | Proposition par défaut |
|---|---|---|
| Q1 | Rattrapage de `main` : `merge` ou `rebase` ? | **Tranchée (AD-04)** : `git merge --no-commit --no-ff main` |
| Q2 | Renommer le notebook selon la numérotation de `main` (`5 - Adapters.ipynb`) ? | Garder le nom actuel jusqu'à la fin, renommer en dernière session |
| Q3 | Paquet tiers incompatible avec pandas 3 / Python 3.13 | **Tranchée (AD-05, AD-06)** : tous compatibles ; aucun plafond ajouté au cœur ; bug pandas 3.1 de darts documenté |
| Q4 | opera incompatible ou abandonné | **Tranchée (AD-07, puis AD-18)** : SHA figé en session 1 ; abandonné le 2026-11-26 au profit d'une agrégation native BOA / MLpol (§ 5.4), opera ne servant plus qu'à calculer une fois des valeurs de référence des tests ; fait en session 10 (AD-22) |
| Q5 | `HierarchicalForecastAdapter` : `reconciler` (singulier) seul, ou aussi `reconcilers` (liste) ? | **Tranchée (AD-13)** : `reconciler` seul ; `reconcile(X, reconcilers=None)` pour comparer |
| Q6 | Étiquette des lignes d'agrégat (`""`, `"Total"`, `"__all__"`) | **Tranchée (AD-13)** : `"Total"` (paramètre `aggregate_label`) |
| Q7 | Ancrage temporel des agrégats (début de période / période cible) | **Tranchée (AD-13)** : selon la position de la grille fine (`S` → début, `E` → fin) ; ancrage sur les origines documenté |
| Q8 | Backends de fondation à livrer en premier | **Tranchée (AD-23)** : Chronos (Chronos-2 et Chronos-Bolt ; T5 refusé) + `callable` en session 12, Timestral et t0 en session 13 |
| Q9 | Sort des fichiers `ADAPTERS_*.md` avant fusion dans `main` | Les supprimer ; le contenu utile passe dans `docs/concepts/adapters.md` |
| Q10 | Datation des agrégats temporels sur une grille sans position (`D`, `h`, `B`, `W-SUN`) | **Tranchée (AD-16)** : premier pas fin (`D`, `h`, `B`, `min`), sauf `W-<jour>` (semaine étiquetée par son dernier jour, convention de `get_labelled_period`) → dernier pas fin ; pas de paramètre |
| Q11 | Utilitaire public d'agrégation des feuilles vers tous les nœuds (`y` sommé, exogènes agrégés comme `aggregate(exog_vars=...)`), pour construire les jeux des modèles de base de chaque niveau | Hors périmètre de la session 6 ; à décider avant la session 8 (notebook) |
| Q12 | Compatibilité MAPIE (prédictions conformes) et AgACI | Hors chantier actuel. Vérifier en session dédiée : `SplitConformalRegressor` / `TimeSeriesRegressor` sur chaque adapter (MAPIE garde les DataFrames, donc les index) ; EnbPI tire des blocs avec remise, donc probablement des couples (entité, date) dupliqués que `TemporalLayout` refuse ; AgACI à construire sur l'ACI de MAPIE et le noyau d'agrégation native (perte pinball) |
| Q13 | `AutoGluonAdapter` : horizon fixé au constructeur d'AutoGluon (`prediction_length`) contre nombre de pas déduit des index ; mode `TimeSeriesPredictor` ou `TabularPredictor` (`'row'`) | À trancher en session A1, quand AutoGluon acceptera pandas 3 (§ 5.6) |

---

## 10. Journal des décisions

| Id | Date | Session | Décision | Raison |
|---|---|---|---|---|
| AD-01 | 2026-11-25 | préparation | Les adapters ne décalent jamais rien ; l'horizon est déduit des index | Principe central (§ 1) |
| AD-02 | 2026-11-25 | préparation | `predict` renvoie un `ndarray` dans l'ordre des lignes de `X` | Compatibilité scorers / `cross_val_predict` (§ 3.3) |
| AD-03 | 2026-11-25 | préparation | Modèles de fondation : un adapter + des backends enregistrés | Pas d'API commune dans l'écosystème (§ 5.5) |
| AD-04 | 2026-11-25 | 1 | Rattrapage par `git merge --no-commit --no-ff main` ; `crossvals/base.py` et `uv.lock` pris de `main` (lock régénéré) | Pas de réécriture des commits de la branche ; `get_n_splits` déjà sur `main` |
| AD-05 | 2026-11-25 | 1 | Extras `adapters-sktime` (sktime ≥ 1.3.0), `adapters-darts` (darts ≥ 0.48.1), `adapters-hierarchical` (hierarchicalforecast ≥ 1.5.3, utilsforecast ≥ 0.2.17), `adapters-opera`, `adapters-all` ; groupe `notebooks` (`adapters-all`, xgboost, tslearn) hors `default-groups` ; groupe `adapters` de la branche supprimé | Versions minimales supportant pandas 3 (3.1 pour sktime) ; statsforecast / mlforecast exigent `pandas<3` et ne servent pas |
| AD-06 | 2026-11-25 | 1 | Pas de plafond `pandas<3.1` dans `adapters-darts` malgré le bug pandas 3.1 de darts 0.48.1 ; `uv.lock` reste sur pandas 3.0.6 | Ne pas contraindre le cœur ; à revoir en session 4 (relever le plancher de darts dès le correctif publié) |
| AD-07 | 2026-11-25 | 1 | opera figé sur `9f8203363ba75206e9beb02002e3420037456200`, `scipy` déclaré dans l'extra | Dépôt non publié sur PyPI et inactif depuis 2024 ; BOA, MLpol, MLprod et FTRL passent sous pandas 3 ; réimplémentation laissée à la session 9 |
| AD-08 | 2026-11-25 | 2 | API de `_base` du § 4 : `check_adapter_input` (via `validate_data`) remplace `set_fitted_feature_info` ; `iter_entity_blocks` de `tsforecast.panel.utils` remplace `split_by_entity` / `iter_entities` ; dates `Period` refusées (`TypeError`, conversion par `to_timestamp()`) ; `steps_between` par grille `date_range` + `get_indexer` | Réutiliser l'existant ; une grille gère multiplicateurs et ancrages, que `xy/rolling._period_ordinals` (privé) ne gère pas |
| AD-09 | 2026-11-25 | 2 | Batterie de contrat en fonctions `check_*` (style `check_estimator`) + mixins unit / workflow, dans `tests/support/adapter_contract.py`, auto-validée sur deux adapters de référence et des mutants ; `test_init.py` (sous-processus, ~2 s, marqué `slow`) lancé explicitement par le job CI `tests-adapters` | Partagée par les tests unit et integration de chaque adapter ; une batterie qui ne détecte rien ne vaut rien |
| AD-10 | 2026-11-25 | 2 | `conftest.py` racine : doctests d'un adapter ignorés si son paquet manque ; exclusions **transitoires** des doctests de `hierarchicalforecast.py` (ne se parse pas, session 6) et de `sktime.py` (exemple faux, session 3) ; `hierarchicalforecast.py` dans `EXCLUDED_PREFIXES` de `scripts/check_conventions.py` et non exporté par `tsforecast.adapters` (session 6 : retirer ces trois exclusions) | `check_conventions --rule all` vert sans toucher aux modules réécrits plus tard |
| AD-11 | 2026-11-25 | 3 | `SktimeAdapter` : paramètre `fh` supprimé (horizon absolu tiré des dates de `X`) ; ajustement par la vectorisation de sktime (`pd-multiindex` / hiérarchique) ; prévision hybride (un appel sktime si `X` est ignoré ou complet sur toutes les entités et dates, sinon un appel par entité par `forecasters_`, sinon `ValueError` pour un forecaster à panel natif) ; capacités lues sur les tags sktime (`capability:exogenous`, `capability:non_contiguous_X`, `capability:missing_values`, `requires-fh-in-fit`) ; exclusion transitoire des doctests de `sktime.py` (AD-10) levée ; contrôle d'aller-retour `set_params(**get_params())` de la batterie comparant les valeurs des paramètres (objets imbriqués réduits à leur classe) | sktime n'a qu'un horizon commun à toutes les entités : un forecaster qui lit `X` échoue sur un test incomplet (panel déséquilibré) ; le `set_params` de sktime remplit un cache privé qui change l'empreinte `joblib` sans changer les paramètres |
| AD-12 | 2026-11-25 | 4 | `DartsAdapter` : plus de paramètres forcés (`lags=None`, `lags_future_covariates=[0]`...) ; mode déduit de `extreme_lags` : `'row'` (régression ligne à ligne, série factice `NaN` d'un pas par suite contiguë de lignes, segments contigus à l'entraînement), `'global'` et `'local'` (extrapolation de l'historique, `ValueError` si `X` manque sur le gap) ; pas d'option de covariables statiques ; batterie : `handles_training_holes`, `GridSearchCV` ajusté sur l'entraînement | Forcer la régression réduit darts à une régression sklearn sur `X` ; l'extrapolation ouvre les modèles locaux et autorégressifs de darts ; darts n'accepte que des covariables statiques numériques (l'utilisateur ajoute ses indicatrices dans `X`) ; un modèle qui extrapole ne prédit pas les lignes d'entraînement |
| AD-13 | 2026-11-25 | 6 | Q5 : `reconciler` singulier (`None` → `BottomUp()`) pour `predict` et `GridSearchCV`, `reconcile(X, reconcilers=None)` pour comparer ; Q6 : `aggregate_label="Total"` ; Q7 : datation des agrégats temporels selon la position de la grille fine (`S` → première date de la période, `E` → dernière ; grille sans position : Q10) ; `level`, `intervals_method`, `num_samples`, `seed` arguments de `reconcile()`, diagnostics par `diagnose(X, reconcilers, atol)`, `is_balanced` interne | Le constructeur ne porte que ce qui change `predict` ; un seul réconciliateur par estimateur garde `GridSearchCV` naturel ; une grille `QE` étiquette ses périodes par leur fin, une grille `QS` par leur début |
| AD-14 | 2026-11-25 | 6 | `HierarchicalForecastAdapter` en sous-paquet (`adapter.py`, `_structure.py`, `reshape.py`) ; format long ; `hierarchy` en liste de niveaux d'index (nœuds appris au fit, par copie) ou dict `{parent: [enfants]}` sur `node_level` (niveaux `depth_<d>`) ; niveaux non date restants = copies indépendantes réconciliées séparément (copie inconnue acceptée pour un arbre sans besoin insample) ; `S` construit par l'adapter (identifiants chaînes, feuilles en bas) ; `reconcile` public de hierarchicalforecast appelé par copie sur des copies profondes des réconciliateurs et de `level` ; `forecast_columns` (pas d'`exog_vars`) ; `model_combination` `None` / `"mean"` / `{niveau: colonne}`, nommée `combined` ; insample mémorisé au fit, prévisions de base hors échantillon produites pli par pli (`cross_validate(return_estimator=True, return_indices=True)` : `cross_val_predict` refuse les splitters temporels) ; `TopDown` / `MiddleOut` refusés sur structure groupée ou feuilles hors du niveau le plus fin ; `stack_nodes` / `unstack_nodes` exportés par `tsforecast.adapters` | `reconcile` mute réconciliateurs et `level` ; chaque date se réconcilie seule (`P` ne dépend que de `S` et de l'insample) ; des copies indépendantes ne partagent pas leurs covariances d'erreurs ; les réconciliateurs stricts de hierarchicalforecast lisent le niveau le plus fin des *tags* comme le bas ; les données larges de l'auteur (`value_a..value_e`) passent par `stack_nodes` |
| AD-15 | 2026-11-25 | 6 | Exclusions d'AD-10 levées (`conftest.py` racine : `tsforecast/adapters/hierarchicalforecast/adapter.py` collecté si hierarchicalforecast est installé ; `EXCLUDED_PREFIXES = ()` ; adapter exporté) ; champ `AdapterCase.make_splitter` ajouté à la batterie de contrat | Le module se parse et ses doctests passent ; une série hiérarchique a toujours un `MultiIndex`, que `TSOutOfSampleSplit` refuse |
| AD-16 | 2026-11-26 | 7 | Modes temporel (`hierarchy=None`) et cross-temporel de `HierarchicalForecastAdapter` : niveau temporel dans l'index (`temporal_level`, défaut juste avant la date) ; `temporal_hierarchy` en pas fins, chaque valeur divisant la plus grande ; grille fine détectée sur les lignes du niveau fin (`temporal_freq_`) ; périodes calendaires = fréquence fine × `m` alignée sur l'époque Unix ou sur `temporal_origin` (nouveau paramètre) ; Q10 : agrégat daté par son premier pas fin, son dernier sur une grille étiquetée par la fin (`E`, `W-<jour>`) ; `S_te` d'une période fixée au fit, colonnes = (copie, période) ; périodes complètes exigées au predict seulement ; réconciliateurs et intervalles insample refusés (`ValueError`) ; fonction publique `period_splits` (plis de périodes complètes, gap en pas fins, agrégats chevauchant le gap exclus) ; pas d'`aggregation_type` ; cas de contrat temporels limités aux règles indépendantes de la fenêtre de test | Un réconciliateur non insample a un `P` bloc-diagonal par période : aucun `S` propre à l'horizon ; l'ancrage calendaire rend les périodes indépendantes de la fenêtre d'entraînement (fenêtres glissantes) et communes aux séries d'un panel déséquilibré ; `temporal_origin` couvre exercices fiscaux, semaines et lignes indexées par origine ; les splitters de `tsforecast` comptent des lignes par entité, incompatibles avec des niveaux temporels de grilles différentes |
| AD-17 | 2026-11-26 | 7 | Cross-temporel en une passe sur `S_ct = S_cs ⊗ S_te` par copie (niveaux `'<cs>//<te>'`, bloc identité replacé en bas) ; `TopDown` / `MiddleOut` refusés (structure non stricte) | Aucune méthode native en 1.5.3 ; cohérence dans les deux dimensions par construction ; égale aux deux passes enchaînées pour `BottomUp`, `ols`, `wls_struct` (`P_ct = P_cs ⊗ P_te`, testé), seule cohérente pour `nonnegative=True` ; la composition du tutoriel (passe temporelle sur les prévisions de base) n'est pas cohérente en cross-section |
| AD-18 | 2026-11-26 | planification | `OperaAdapter` et l'extra `adapters-opera` abandonnés ; agrégation d'experts en ligne native (BOA, MLpol) dans un sous-paquet hors `adapters` (proposition `tsforecast/ensemble/`, à confirmer), noyau de mises à jour pur à perte interchangeable ; valeurs de référence calculées une fois avec opera hors du dépôt | opera non publié sur PyPI et inactif depuis 2024 ; AgACI (non implémenté dans MAPIE 1.6.0) agrège des bornes d'ACI, pas des prévisions de modèles : il ne remplace pas l'agrégation de prévisions, mais pourra s'appuyer sur le même noyau |
| AD-19 | 2026-11-26 | planification | `AutoGluonAdapter` ajouté au périmètre mais reporté (sessions A1–A2 hors séquence) jusqu'à une version d'AutoGluon acceptant pandas 3 | AutoGluon 1.6.3 et `master` déclarent `pandas>=2.0,<2.4` : non installable avec `pandas>=3` |
| AD-20 | 2026-11-26 | planification | Règle 11 de la batterie de contrat (équivalence au paquet natif, champ `AdapterCase.native_reference`) ; session dédiée insérée juste après la session 8, avant l'agrégation native et les modèles de fondation | Seule preuve que l'adapter ne change pas le modèle (horizon, covariables, ordre) ; l'infrastructure existe ainsi avant que les adapters suivants ne s'y branchent, et un défaut trouvé sur sktime, darts ou hierarchicalforecast est corrigé avant les notebooks et la documentation suivants |
| AD-21 | 2026-11-26 | 9 | Règle 11 : `native_reference(X_train, y_train, X_test, fh)` (horizon relatif fourni par la batterie, écart à la signature `(X_train, y_train, X_test)` prévue) et `native_gaps` ; série et panel équilibré, gaps 0 et 2, 3 dates de test, tolérances inchangées des cas ; modèles hors batterie dans des classes `TestNativeEquivalence` ; références hiérarchiques par copie, `S_df` de l'arbre écrit à la main, cross-temporel natif = deux passes enchaînées | Une référence qui déduirait l'horizon des dates refait le calcul de l'adapter au lieu de le contrôler ; SARIMAX et les modèles darts à covariables ne passent pas les règles à gap (X absent du gap), d'où une comparaison seule ; `aggregate` ne sait pas construire l'arbre des jeux ; hierarchicalforecast n'a pas de réconciliation cross-temporelle ; aucun écart trouvé, aucun adapter corrigé |
| AD-22 | 2026-11-27 | 10 | Sous-paquet `tsforecast/ensemble/` (`_algorithms.py` noyau numpy pur, `online.py`), `OnlineAggregator` et `Loss` exportés ; algorithmes **fidèles aux articles** (BOA : figure de la procédure adaptative de Wintenberger, a priori `π_0`, `c = 20` ; MLpol : taux d'opera R parti de `1/η = 0`) et non à opera-python ; tours par date en `global` (excès de perte moyen) ; `per_entity` tient aussi l'agrégation globale (entité inconnue) ; `weights_` (`Series` globale) et `entity_weights_` (`DataFrame`) séparés ; `init_weights` = a priori de BOA, poids de repli de MLpol ; `cv` entier = blocs de dates communs ; paramètres imbriqués `experts__<nom>__<param>` par `get_params` / `set_params` surchargés (`MetaEstimatorMixin` au lieu de `_BaseComposition`) ; `partial_fit` exige des lignes postérieures à toutes les lignes agrégées ; littéraux de référence d'opera (MLpol tel quel, BOA corrigé) avec injection de l'état initial d'opera | opera-python s'écarte de la figure de BOA sur trois lignes et rend MLpol dépendant de l'échelle (`1/η_0 = 2^20`) ; traiter séparément les entités d'une même date lirait des cibles inconnues en temps réel ; un attribut dont le type dépend d'un paramètre gêne l'utilisateur ; `_BaseComposition` nomme `<nom>__<param>`, le prompt exige `experts__ridge__alpha` ; un splitter de panel compte `test_size` en lignes par entité, ses plis se chevauchent sur des données trouées (`DropNaNTarget`) ; une règle globale de postériorité garde l'agrégation globale chronologique |
| AD-23 | 2026-11-27 | 12 | `FoundationModelAdapter` en sous-paquet `tsforecast/adapters/foundation/` ; protocole `BackendCapabilities` / `ForecastRequest` (contextes `(n_cibles, T_i)` de longueurs libres, covariables passées `(C, T_i)` et connues `(C, T_i + H)`, vues en lecture seule) / `ForecastResult` `(n, V, H, Q)` / `ForecastBackend` structurel ; registre à préfixes (le plus long gagne, sans casse), objet backend accepté tel quel ; cache du processus par (backend, `model_id`, `device`, `backend_kwargs`), modèle jamais paramètre ni attribut, paquet contrôlé avant le cache ; **un appel par pas** (ligne à `k` pas = dernier pas d'une prévision de longueur `k`, covariables connues des pas 1..k seulement) au lieu d'un appel de `max(pas)` par entité ; `covariates='auto'` (connues, sinon passées, sinon aucune) **sans avertissement**, valeur explicite non prise en charge → `ValueError`, au lieu d'une dégradation avertie ; `quantile_levels` argument de `predict_quantiles` et non du constructeur ; niveau hors de la plage apprise → `ValueError` ; interpolation linéaire par l'adapter pour un modèle qui n'interpole pas ; entités d'une requête dans l'ordre de leurs étiquettes ; backends `chronos` (Chronos-2, Bolt ; **T5 refusé**) et `callable` (`CallableBackend`, capacités en paramètres à plat) ; Q8 tranchée ; extras `adapters-chronos`, `adapters-light` (sans torch), `adapters-all` = léger + chronos ; job CI `tests-adapters` sur `adapters-light` | Mesuré sur `chronos-2-small` : un appel unique fait lire à un pas les covariables des pas suivants (X du pas 5 → +0,14 au pas 3), donc une ligne lirait une origine postérieure ; au-delà d'un patch de sortie, la longueur de prévision change les premiers pas (−0,502 contre −0,543). Un avertissement serait répété à chaque pli, et une demande explicite ne doit jamais être dégradée en silence. Le constructeur ne porte que ce qui change `predict` (principe d'AD-13). Chronos borne en silence les niveaux hors plage, et seul Timestral n'interpole pas. Ordonner les entités par étiquette rend les lots indépendants de l'ordre des lignes. T5 tire des trajectoires au hasard, donc ses prévisions ne sont pas reproductibles. torch de PyPI entraîne les bibliothèques CUDA sous Linux : l'utilisateur des adapters classiques et la CI n'en ont pas besoin |
|  |  |  |  |  |
