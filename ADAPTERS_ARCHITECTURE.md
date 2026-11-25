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

Envelopper des modèles de paquets tiers (sktime, darts, hierarchicalforecast, opera, modèles de
fondation pour séries temporelles publiés sur Hugging Face) dans des estimateurs **compatibles
sklearn**, utilisables tels quels avec :

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
non réalignée sur `X`.

**`OperaAdapter`**
- `mixture_` et `is_fitted_` posés dans `__init__` (violation sklearn : `clone`,
  `check_is_fitted`) ; noms d'algorithmes faux dans la docstring (opera ne connaît que `BOA`,
  `MLpol`, `MLprod`, `FTRL`) ; `Mixture` exige des experts en `pandas.DataFrame` (un `ndarray` lève
  `TypeError`) ; aucun support panel.

**Notebook `4 - QB - Adapters.ipynb`**
- Section 2 : `X = X.shift(-horizon)` sur **tout le panel** (décalage inter-entités) et dans le
  mauvais sens (covariables futures) ; `gap = horizon + delays` au lieu de `h + d - 1`.
- Import de l'API privée `sklearn.utils.metaestimators._safe_split` ; `xgboost` et `tslearn`
  utilisés mais non déclarés ; cellule 53 syntaxiquement fausse ; section hierarchicalforecast
  appelant une API (`spec=`) qui n'existe pas.

### 2.3 Écosystème tiers (vérifié le 2026-11-25)

| Paquet | Distribution | Remarques |
|---|---|---|
| hierarchicalforecast | PyPI | `aggregate(df, spec)` (spec : liste de listes de colonnes), `aggregate_temporal(df, spec)` (spec : `{nom: nb de pas}`), `get_cross_temporal_tags`, `HierarchicalReconciliation(reconcilers).reconcile(Y_hat_df, S_df, tags, Y_df=None, temporal=False, ...)`. Colonnes de sortie `<modèle>/<Réconciliateur>[_method-<m>]`. La réconciliation temporelle ne supporte pas les méthodes *insample* (`mint_shrink`, ...). Le tutoriel cross-temporel fait deux appels séparés (cross-section, puis temporel sur toutes les séries cross-section) et n'évalue qu'avec des *tags* cross-temporels. |
| opera (`Dralliag/opera-python`) | **pas sur PyPI** (`pip install git+...`) ; version `0.0.1` | `Mixture(y, experts: DataFrame, awake=None, model="BOA"|"MLpol"|"MLprod"|"FTRL", coefficients="uniform"|ndarray, loss_type="mse"|"mae"|"mape"|"msle"|"mspe"|callable, loss_gradient=True, parameters=None)`. Le constructeur consomme les données (appelle `update`). `update(new_experts, new_y)` séquentiel ligne à ligne ; `predict(new_experts)` sans mise à jour (poids courants `w`). Dépendances non figées, **scipy importé mais non déclaré**. Compatibilité pandas 3 inconnue. |
| chronos-forecasting | PyPI | `Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=...)` ; `predict_df(context_df, future_df=..., prediction_length, quantile_levels, id_column, timestamp_column, target)` ; covariables passées et futures en colonnes. Modèles : `amazon/chronos-2`, `autogluon/chronos-2-small`, `amazon/chronos-bolt-{tiny,mini,small,base}`, `amazon/chronos-t5-*`. `pandas>=2,<4`. |
| timestral (Mistral AI) | **pas sur PyPI** (`pip install git+https://github.com/mistralai/timestral.git`) | `TimestralForecaster.from_pretrained("mistralai/Timestral-{5M,28M,123M,209M}", device=...)` ; `model.forecast(context, prediction_length)` → `.median`, `.quantile(q)` ; covariables passées et futures annoncées ; contexte ≤ 32 768 pas ; sortie ≤ 1 024 pas par passe. Licence « Modified MIT » avec clause d'usage. Pas d'API batch documentée sur la carte. |
| t0 (The Forecasting Company) | PyPI `tfc-t0[torch]>=0.6.1` | `T0Forecaster.from_pretrained("theforecastingcompany/t0-gamma")` ; `predict(context, horizon, quantile_levels, mask=, group_ids=, future_covariates=)` sur tenseurs `(B, T)` / `(B, V, T)` ; `batch_series([...])` ; Apache-2.0 ; ~256 M paramètres. |

**Versions retenues en session 1** (Python 3.13, pandas 3.0.6, scikit-learn 1.9.1, numpy 2.3.2 ;
résolution `uv lock` sans changement des versions de `main`) :

| Paquet | Version | Compatibilité |
|---|---|---|
| sktime | 1.3.0 | pandas 3 depuis 1.2.0, pandas 3.1 depuis 1.3.0 ; plafonne `pandas<3.2`, `numpy<2.6`, `scikit-learn<1.10`, `joblib<1.7` |
| darts | 0.48.1 | pandas 3 supporté ; **bug connu avec pandas 3.1** (séries journalières → `TypeError`), correctif non publié ; xgboost / statsforecast hors du cœur de darts |
| hierarchicalforecast | 1.5.3 | roues cp313 ; pas de dépendance à statsforecast |
| utilsforecast | 0.2.17 | support de pandas 3 à partir de cette version (plancher déclaré dans l'extra) |
| opera | commit `9f82033` (`develop`, 2024-11-06) | pur Python, aucune écriture en place sur les entrées ; `scipy` importé non déclaré (déclaré dans l'extra) ; dépôt sans fichier LICENSE (setup.py : MIT) ; plus maintenu depuis 2024 |
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
- Les composites de plusieurs estimateurs (experts opera) héritent de
  `sklearn.utils.metaestimators._BaseComposition` comme `StackingRegressor`
  (paramètres imbriqués `experts__<nom>__<param>`).
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

**Rôle** : réconciliateur. `X` contient les **prévisions de base** (une colonne par modèle de base)
et, optionnellement, des covariables exogènes déclarées ; `y` les réalisations. `fit` construit la
structure (`S`, *tags*) et mémorise ce dont les réconciliateurs *insample* ont besoin ; `predict`
réconcilie les prévisions de base des lignes de `X`.

**Format** : long côté sklearn. Chaque nœud de la hiérarchie est une ligne ; le nœud est porté par
des niveaux d'index, la date reste le dernier niveau. `y` est donc une `Series` 1-D sur tous les
nœuds (feuilles et agrégats) et `predict` renvoie une valeur par ligne de `X` : `cross_val_score`,
`GridSearchCV` et les splitters panel (qui regroupent par date) fonctionnent sans adaptation.

**Trois modes**, déduits des paramètres renseignés :

| Mode | Paramètres | Exemple |
|---|---|---|
| cross-section | `hierarchy` | entités `(Country, State, Region)` ou nœuds `value_a = value_b + value_c` |
| temporel | `temporal_hierarchy={"year": 12, "quarter": 3, "month": 1}` | agrégats de fréquences inférieures |
| cross-temporel | les deux | tutoriel *Australian Domestic Tourism — cross-temporal* |

- `hierarchy` accepte deux formes : (a) une `spec` à la hierarchicalforecast (liste de listes de
  niveaux d'index, du plus agrégé au plus fin), les lignes d'agrégat portant une étiquette réservée
  (paramètre `aggregate_label`, par défaut `"Total"` ou chaîne vide : à trancher) dans les niveaux
  absents ; (b) un dict `{parent: [enfants]}` sur les valeurs d'un niveau d'index nommé
  (`node_level`). Un utilitaire public convertit un DataFrame « large » (une colonne par nœud,
  comme `value_a..value_e` du notebook) vers ce format long et inversement.
- Séries temporelles : la hiérarchie porte sur les nœuds seuls. Panel : soit les entités sont des
  **copies indépendantes** de la même hiérarchie (l'entité préfixe `unique_id`), soit elles **sont**
  la hiérarchie (forme (a)).
- Temporel : un niveau d'index (`temporal_level`, nom paramétrable) porte la clé de
  `temporal_hierarchy` ; une ligne d'agrégat est datée au **début** de sa période (à confirmer). Le
  `predict` refuse une période d'agrégation incomplète dans `X` (erreur explicite) : en validation
  croisée, le test doit couvrir des périodes complètes du niveau le plus agrégé
  (`test_size` multiple de son nombre de pas). Les méthodes *insample* sont refusées en temporel.
  Attention : avec des lignes indexées par origine, une agrégation calendaire porte sur les
  origines, pas sur les périodes cibles ; le documenter (ou proposer un ancrage sur la période
  cible).
- Cross-temporel : composition de la réconciliation temporelle et cross-section (ordre et nombre
  de passes à arrêter en session, en vérifiant d'abord si la version installée de
  hierarchicalforecast propose une méthode cross-temporelle native) ; un test vérifie la cohérence
  numérique dans les **deux** dimensions.
- Multi-modèle : plusieurs colonnes de prévisions de base. Paramètre `model_combination` :
  `None` (une seule colonne), `"mean"` (moyenne des colonnes, comme le tutoriel *multi-model*),
  ou un dict niveau → colonne (un modèle par niveau). Les colonnes non choisies sont ignorées.
- Sortie : `predict` renvoie la réconciliation d'**un** réconciliateur (paramètre `reconciler`,
  singulier, pour que `GridSearchCV` compare les réconciliateurs par
  `param_grid={"reconciler": [BottomUp(), MinTrace(method="ols"), ...]}`) ; `reconcile(X)` renvoie
  toutes les colonnes `<modèle>/<réconciliateur>` pour l'inspection. La compatibilité avec un
  paramètre `reconcilers` (liste) est une question ouverte (§ 9).
- `fit` ne dépend que de la structure et des données d'entraînement ; `predict` ne réécrit aucun
  attribut ajusté (les `S` / *tags* propres à l'horizon de test sont des variables locales).
- Prévisions de base *insample* (pour `mint_shrink`, `wls_var`, ERM) : `X_train` doit contenir des
  prévisions **hors échantillon** produites en amont (par exemple `cross_val_predict` avec un
  splitter de `tsforecast`) ; le notebook le montre.

### 5.4 `OperaAdapter`

**Rôle** : agrégation en ligne d'experts. Deux modes :

1. **méta-estimateur** (par défaut) : `experts=[("ridge", Ridge()), ("rf", RandomForestRegressor())]`
   (`_BaseComposition`). `fit` :
   1. produit des prévisions **hors échantillon** des experts sur l'entraînement, avec un splitter
      temporel interne (paramètre `cv`, splitter de `tsforecast` ; jamais de prévisions *in-sample*,
      qui biaiseraient les poids vers l'expert qui sur-apprend) ;
   2. fait tourner la `Mixture` séquentiellement, lignes triées par date, sur ces prévisions ;
   3. réajuste chaque expert sur tout l'entraînement (`experts_`).
   `predict` combine les prévisions des experts réajustés avec les **poids finaux** (`y` est inconnu
   au `predict` : aucune mise à jour).
2. **`experts="passthrough"`** : les colonnes de `X` sont déjà des prévisions d'experts (produites
   par d'autres adapters ou par `cross_val_predict`).

- Paramètres opera : `aggregation` (`"BOA"`, `"MLpol"`, `"MLprod"`, `"FTRL"`), `loss_type`,
  `loss_gradient`, `coefficients`, `parameters` ; validés au `fit`.
- Panel : `pooling="global"` (une seule mixture, lignes ordonnées par date puis entité) ou
  `"per_entity"` (une mixture par entité ; entité inconnue au `predict` → poids globaux si
  disponibles, sinon erreur explicite).
- Attributs : `weights_` (poids finaux, par entité en mode `per_entity`), `weights_history_`.
- `partial_fit(X, y)` : poursuit la séquence (nouvelles lignes postérieures aux précédentes).
- opera n'est pas sur PyPI : dépendance Git **figée sur un SHA** ; `scipy` déclaré explicitement.
  Si opera est incompatible avec pandas 3 / Python 3.13, voir § 9.

### 5.5 Modèles de fondation (Hugging Face) : `FoundationModelAdapter`

**Faisabilité et intérêt.** Un adapter générique est faisable et souhaitable **à condition de ne
pas viser une implémentation unique** : il n'existe pas d'API commune (§ 2.3). L'architecture
retenue sépare :

- un **adapter** unique, `FoundationModelAdapter`, qui porte toute la logique sklearn et
  `tsforecast` (index, panel, calcul des pas, alignement, quantiles, lots) ;
- des **backends** minces (~50–150 lignes), un par famille, qui ne font que convertir des tableaux
  en appel au paquet tiers. Le churn de l'écosystème est confiné aux backends.

```
tsforecast/adapters/foundation/
├── __init__.py          # FoundationModelAdapter, register_backend, ForecastBackend
├── adapter.py           # FoundationModelAdapter (sklearn, tsforecast)
├── protocol.py          # ForecastBackend (Protocol), ForecastRequest, ForecastResult
├── registry.py          # registre nom -> backend, résolution par préfixe d'identifiant HF
└── backends/
    ├── chronos.py       # chronos-forecasting (Chronos-2, Bolt, T5)
    ├── timestral.py     # mistralai/Timestral-*
    ├── t0.py            # theforecastingcompany/t0-*
    └── callable.py      # backend construit depuis une fonction utilisateur
```

- Protocole de backend (indicatif) : `load(model_id, device, **kwargs)` puis
  `predict(request) -> ForecastResult`, où `ForecastRequest` contient, par série, le contexte de
  cible (`ndarray`), les covariables passées et futures alignées (optionnelles), le nombre de pas
  et les niveaux de quantiles ; `ForecastResult` renvoie un tableau `(n_series, horizon, n_quantiles)`.
  Chaque backend déclare ses **capacités** (covariables passées / futures, multivarié, contexte et
  horizon maximaux, quantiles natifs) ; l'adapter dégrade proprement (avertissement unique) quand
  une capacité manque.
- Sémantique dans le paradigme `tsforecast` : un modèle de fondation **n'apprend pas** `y` depuis
  `X` ; il extrapole l'historique de `y`. Donc :
  - `fit(X, y)` mémorise, par entité, les `context_length` dernières lignes de `y` (et de `X` comme
    covariables passées si le backend les accepte) ; aucun entraînement (le *fine-tuning* est hors
    périmètre) ;
  - `predict(X)` calcule pour chaque entité le nombre de pas entre sa dernière ligne d'entraînement
    et chacune de ses lignes de test (`steps_between`), prévoit `max(pas)` pas en un appel par lot,
    puis extrait les pas demandés ; `X` de test sert de covariables futures si le backend les
    accepte (les lignes du `gap`, absentes, sont déclarées manquantes au backend, jamais inventées) ;
  - `GridSearchCV` ne règle que des paramètres d'inférence (`model_id`, `context_length`,
    `quantile`, `use_covariates`, ...).
- Paramètres indicatifs : `model_id`, `backend=None` (déduit du préfixe de `model_id`),
  `context_length`, `quantile=0.5` (valeur renvoyée par `predict`), `quantile_levels` (pour
  `predict_quantiles`), `use_covariates`, `device="cpu"`, `batch_size`, `backend_kwargs`.
- Chargement des poids : paresseux, mis en cache **au niveau du processus** (clé : backend,
  `model_id`, `device`) pour ne pas recharger le modèle à chaque pli ni à chaque `clone` ; le
  modèle chargé n'est jamais un paramètre (sinon `clone` le copierait).
- `n_jobs` de sklearn : déconseillé avec ces modèles (copie des poids par processus) ; le documenter.
- Licences : Timestral (« Modified MIT » avec clause d'usage), t0 (Apache-2.0), Chronos
  (Apache-2.0) : à rappeler dans la docstring et le notebook.
- Tests : backends factices sans réseau pour la logique de l'adapter ; tests réels marqués
  (`slow`, et un marqueur `network` à créer) avec les plus petits modèles (`chronos-bolt-tiny` ou
  `chronos-2-small`, `Timestral-5M`), sautés si le paquet ou le réseau manquent.

---

## 6. Dépendances et packaging

- Extras (`[project.optional-dependencies]`) : `adapters-sktime`, `adapters-darts`,
  `adapters-hierarchical`, `adapters-opera`, `adapters-chronos`, `adapters-timestral`,
  `adapters-t0`, et `adapters-all` (liste sans espace). Bornes `>=x,<y` comme le reste du
  `pyproject.toml` ; dépendances Git figées sur un SHA (`allow-direct-references = true`).
- Groupe `notebooks` (hors `default-groups`) : `xgboost`, `tslearn`, `statsforecast` /
  `mlforecast` si le notebook en a besoin, plus les extras d'adapters. `dev` et `docs` restent ceux
  de `main` (ne pas retirer `docs` de `default-groups`).
- CI : job `tests-adapters` séparé (`uv sync --group dev --extra adapters-...` sans les modèles de
  fondation lourds, ou avec CPU seulement), tests rapides uniquement ; les tests `network` ne
  tournent pas en CI.
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
      l'adapter renvoie des dates.

  Session 4 : `AdapterCase.handles_training_holes` (un adapter qui refuse les trous
  d'entraînement doit lever `ValueError` dans le contrôle `XYPipeline`) ; le contrôle
  `GridSearchCV` cherche sur les dates d'entraînement et prédit les dates de test postérieures.
- Commande locale : `.venv/Scripts/python.exe -m pytest tests/unit/adapters -q -p no:cacheprovider`.

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
| Q4 | opera incompatible ou abandonné | **Tranchée pour la session 1 (AD-07)** : SHA figé, fonctionne sous pandas 3 ; réimplémentation interne éventuelle à décider en session 9 |
| Q5 | `HierarchicalForecastAdapter` : `reconciler` (singulier) seul, ou aussi `reconcilers` (liste) ? | `reconciler` pour `predict` / `GridSearchCV`, `reconcile()` pour comparer plusieurs réconciliateurs |
| Q6 | Étiquette des lignes d'agrégat (`""`, `"Total"`, `"__all__"`) | `"Total"` (paramètre `aggregate_label`) |
| Q7 | Ancrage temporel des agrégats (début de période / période cible) | Début de période, ancrage sur les origines documenté |
| Q8 | Backends de fondation à livrer en premier | Chronos-2 + `callable`, puis Timestral et t0 |
| Q9 | Sort des fichiers `ADAPTERS_*.md` avant fusion dans `main` | Les supprimer ; le contenu utile passe dans `docs/concepts/adapters.md` |

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
|  |  |  |  |  |
