# Imputation multi-fréquences

## Le problème

Un jeu de données de prévision mélange souvent des fréquences : indicateurs mensuels,
trimestriels, annuels. Pour entraîner un modèle à la fréquence la plus fine, il faut donner
une valeur mensuelle aux variables observées une fois par trimestre ou par an, sans perdre ce qui a
été observé.

`HighFrequencyImputer` porte ces variables sur une **grille cible** plus fine
(`target_frequency`). Chaque total observé est réparti entre ses sous-périodes par un modèle, puis la
répartition peut être recalée pour que les sous-périodes redonnent exactement ce total. Chaque cellule
produite garde la trace de sa **provenance**.

![Principe de la désagrégation](../assets/hfi_principle.png)

## Le principe

**Additivité.** La valeur d'une période est la somme de ses sous-périodes : c'est l'hypothèse qui
justifie le recalage. Une variable non additive (niveau, taux) passe par un `additive_transformer`
(log, différenciation…) qui expose `fit_transform` et `inverse_transform`.

**Classement des variables.** Au `fit`, la fréquence de chaque colonne est détectée, par couple
(entité, colonne) sur un panel, et chaque variable est rangée dans `variable_categories_` :

| Catégorie | Variable | Traitement |
|---|---|---|
| `target_freq` | déjà à la fréquence cible | conservée |
| `impute` | plus grossière que la cible | imputée par modèle |
| `aggregate` | plus fine que la cible | agrégée exactement sur la grille |

**Un modèle par variable et par étape.** Le modèle d'une variable s'entraîne sur les dates où elle est observée (**lignes
d'ancre**). Ses covariables sont les autres colonnes du jeu de données, disponible à la fréquence de la variable que l'on cherche à prédire (cf axe 1 ci-dessous). Cible et covariables sont ramenées à l'échelle d'une période de la grille (`scale_features`) : sur une grille mensuelle, un total trimestriel est divisé par trois. Le modèle
prédit ensuite chaque sous-période, et `aggregation_constraint` recale le résultat à la bonne échelle. `estimator`
accepte un estimateur unique ou un dictionnaire par colonne ; `None` remplace tous les modèles par
une interpolation.

**Un plan figé.** `fit` enregistre les étapes de l'imputation dans un `ImputationPlan` (`imputation_plan_`), source
de vérité de l'état ajusté ; `transform` le **rejoue** sans rien réajuster. `imputation_models_`,
`frequency_progression_` et `imputation_order_` en sont des vues.

## Les deux axes du paramétrage

Le comportement de l'imputer se paramètre principalement selon **deux axes** indépendants : l'axe 1 détermine les **colonnes** données à
l'estimateur pour imputer la cible, l'axe 2 les **lignes** de sa cible.

### Axe 1 — `covariate_strategy` : comment mettre à disposition du modèle d'imputation une covariable peu observée ?

![Axe 1 : covariate_strategy](../assets/hfi_covariate_strategy.png)

| Valeur | Ce que reçoit le modèle | Prérequis |
|--------|-------------------------|-----------|
| `'interpolate'` (défaut) | la covariable est interpolée entre ses ancres | — |
| `'tolerate_nan'` | la covariable telle qu'observée, trous compris | **l'estimateur doit tolérer les NaN**, sinon l'étape part en repli par interpolation |
| `'model'` | la covariable imputée par son propre modèle, en cascade sur les variables dans l'ordre `fit_predict_order` | — |

Sous `'model'`, une covariable pas encore imputée à l'étape courante passe par `covariate_fallback`
(`'interpolate'` par défaut, ou `'tolerate_nan'`). `fit_predict_order='cv'` ordonne les variables par
score de validation croisée (`cv`, `cv_scoring`, `min_cv_train_size`) ; ces paramètres sont sans
effet hors de `'model'`. `interpolation_method` et `interpolation_anchor` règlent la forme de toute
interpolation (covariables, replis, `estimator=None`).

!!! warning "Regard vers le futur"
    Une interpolation entre deux ancres lit l'ancre **future**. L'imputer est destiné à reconstruire l'histoirique du jeu de données d'entraînement et non à réaliser une simulation en pseudo temps réel.

### Axe 2 — `impute_intermediate_frequencies` : passer par des fréquences intermédiaires ?

![Axe 2 : impute_intermediate_frequencies](../assets/hfi_intermediate_frequencies.png)

| Valeur | Étapes (annuel → mensuel) | `y_train` |
|--------|---------------------------|-----------|
| `False` (défaut) | directe : `['M']` | ancres observées seulement |
| `'covariates_only'` | en cascade : `['Q', 'M']` | ancres observées seulement |
| `True` | en cascade : `['Q', 'M']` | ancres **et** imputations antérieures de la variable |

Trois subtilités :

- `'covariates_only'` ne change les valeurs que sous `covariate_strategy='model'`, et seulement
  lorsqu'une covariable plus grossière est imputée **après** la variable qui l'utilise : sinon, sa
  valeur déjà imputée à l'étape courante prime sur celle de l'étape intermédiaire.
- Avec un estimateur linéaire non pénalisé, `True` redonne les valeurs de `False` : les imputations
  intermédiaires se trouvent sur l'hyperplan ajusté. L'écart n'apparaît qu'avec un estimateur
  pénalisé ou non linéaire.
- Sous `True`, `aggregation_constraint` agit aussi sur `y_train` : sous `'sum'`, une imputation
  antérieure tombant sur une date d'ancre en est retirée (elle serait une combinaison exacte des
  autres lignes) ; sous `None`, elle est conservée.

## Provenance

Chaque cellule du résultat porte une étiquette `ProvenanceType`. L'imputeur en émet neuf :

| Famille | Niveaux | Sens |
|---|---|---|
| Non modèle | `ORIGINAL`, `AGGREGATED`, `INTERPOLATED` | valeur d'entrée, agrégation additive exacte, interpolation |
| Modèle, par pire ingrédient vu | `MODEL_ON_TRUE` → `MODEL_ON_INTERPOLATED` → `MODEL_ON_IMPUTED` → `MODEL_ON_IMPUTED_TARGET` → `MODEL_ON_IMPUTED_BOTH` | de la plus sûre à la plus dégradée |
| Sans ancre | `MODEL_UNANCHORED` | l'entité n'observe jamais la colonne (`impute_unobserved_entities=True`) |

La dixième valeur de l'énumération, `DISAGGREGATED`, n'est jamais émise : le recalage conserve la
provenance de la valeur recalée.

La matrice de la **dernière passe** (`fit`, puis chaque `transform`) est exposée dans
`imputation_provenance_`, son résumé est disponible dans `provenance_statistics_`. Elle décrit les cellules de la sortie : une colonne plus fine que la grille y porte
l'agrégation exacte de la période (`AGGREGATED`), une cellule vide n'est jamais marquée (`NaN`). La
matrice est **catégorielle** (`PROVENANCE_DTYPE`) : une cellule se lit comme une simple chaîne
(`'original'`, `'model_on_true'`...) et se compare directement à un membre
(`provenance == ProvenanceType.ORIGINAL`).

La provenance ne voit que ce que l'imputeur produit : des NaN comblés à l'intérieur de l'estimateur
(un `SimpleImputer` dans un `Pipeline`) n'y apparaissent pas.

## Contrainte d'agrégation

Laissées libres, les sous-périodes prédites n'ont aucune raison de sommer à l'observation qu'elles
décrivent. Sous `aggregation_constraint='sum'` (défaut), les sous-périodes d'une période sont
multipliées par le rapport total observé / somme prédite : une prédiction libre devient une vraie
**désagrégation** de l'observation. `None` laisse les prédictions telles quelles ; un dictionnaire
permet de paramétrer ce comportement par colonne.

Le recalage ne change jamais la provenance d'une cellule. Une période que les bornes des données ou
de la fenêtre tronquent n'est pas recalée : un total annuel ne s'impose pas à dix mois sur douze. Un
avertissement unique par passe le signale, comme pour les périodes dont la somme prédite est nulle, qui
laissées inchangées.

## Fenêtres

Trois fenêtres coexistent, chacune exposée comme masque booléen et comme bornes :

| Fenêtre | Attributs | Réglage |
|---|---|---|
| stricte : toutes les variables disponibles | `strict_window_mask_` | — (diagnostic seulement) |
| de prédiction : les lignes écrites | `imputation_window_mask_`, `imputation_window_` | `imputation_scope`, `coverage_threshold` |
| d'entraînement : les lignes vues par les modèles | `training_window_mask_`, `training_window_` | `training_scope`, `training_coverage_threshold` |

`imputation_scope` étend la fenêtre stricte vers le passé, l'avenir ou les deux
(`'extended_backward'`, `'extended_forward'`, `'extended_both'`) ; une période n'y entre que si sa
couverture atteint `coverage_threshold`. `training_scope` suit `imputation_scope` par défaut
(`None`). Ce paramètre **ajoute des lignes, jamais des colonnes** : le choix des covariables reste fixé par
leur disponibilité au moment de la prédiction. Borner l'entraînement (`training_scope='strict'`)
tout en étendant la prédiction évite d'apprendre sur des périodes pauvres en covariables.

## Panels

Sur un panel, toutes les opérations sont effectuées par couple (entité, colonne) : une même colonne peut être annuelle
pour une entité et trimestrielle pour une autre, et `target_frequency` accepte un dictionnaire par
entité.

- **Jeu d'entraînement mutualisé.** Le modèle d'une variable s'entraîne sur toutes les entités qui
  l'observent, chacune ramenée à l'échelle de l'étape. Cela suppose des niveaux comparables entre
  entités : `scale_features` ne corrige que l'échelle de fréquence.
- **Provenance contagieuse.** Une entité qui apporte des cellules interpolées ou imputées dégrade la
  provenance de toutes les cellules que l'étape produit, y compris celles concernant d'autres entités car le modèle est entraîné sur des données approximatives.
- **Colonne absente d'une entité.** `covariate_eligibility='any_entity'` (défaut) garde comme
  covariable une colonne observée par au moins une entité ; `'all_entities'`
  écarte une colonne qui n'est pas observée par toutes les entités. Il s'agit du choix le plus sûr pour un estimateur qui ne tolère pas les NaN.
- **Entité sans observation d'une colonne.** Sous `impute_unobserved_entities=True`, l'entité reçoit
  une imputation complète apprise sur les autres entités, marquée `MODEL_UNANCHORED`, ni recalée ni divisée
  (`unanchored_pairs_`). Au `transform`, le paramètre gouverne aussi les couples que le `fit` n'a
  jamais vus (entité nouvelle, colonne laissée vide) : sous `False` ils restent tels quels ; sous
  `True`, chacun est imputé par le modèle figé de sa colonne, ancré sur ses propres observations
  quand le `transform` en a.

## Sortie

Le résultat empile les niveaux de fréquence de la progression sur le côté « entité » de l'index :
`(frequency, date)` pour une série, `(entité..., 'frequency', 'date')` pour un panel.
`keep_lower_frequencies=False` ne garde que la fréquence cible, sans niveau `frequency` dans l'index : il s'agit d'un paramètre
d'affichage, sans effet sur les valeurs. Le nom `'frequency'` est **réservé** : `fit` et `transform`
refusent une entrée dont un niveau d'index ou une colonne le porte déjà.

La sortie suit l'ordre de l'entrée et porte les lignes de la grille cible : une grille densifiée
(index irrégulier) en ajoute ; une cible plus grossière que l'index en retire, la valeur d'une ligne retirée étant agrégée sur la ligne de sa période. `inverse_transform` défait la dernière passe et 
rend l'index de son entrée : sans `restore_original_values`, seules les cellules `ORIGINAL` y
reviennent ; avec, toutes les valeurs observées.

Les dates de toute passe sont dans l'unité des dates du `X` de cette passe (`s`, `ms`, `us` ou
`ns`), y compris pour `y` ; les valeurs ne changent jamais.

## Dans un pipeline entraîné avec `y`

Entraîné avec `y`, `HighFrequencyImputer` **augmente un jeu d'entraînement** (`X` et `y`) ; il n'a
pas vocation à transformer des données de test. Un `XYPipeline` appelle `fit_transform` sur ses
étapes pendant son `fit`, et `transform` pendant `predict` / `score` : les deux passes diffèrent.

| Appel | Passe | Ce qui sort |
|---|---|---|
| `fit_transform(X, y)` | entraînement | `X` **et** `y` sur la grille cible, lignes ajoutées comprises |
| `transform(X)` / `transform(X, y)` | prédiction | les colonnes de `X` seules, imputées par les modèles figés, **sur les lignes d'entrée** ; `y` rendu **inchangé** |

Par défaut `y` est imputé comme une colonne, mais n'est **pas une covariable** des autres colonnes
(`target_as_covariate=False`) : dans un pipeline la passe de prédiction n'a en effet pas de `y` disponible pour alimenter un tel
modèle. `target_as_covariate=True` rétablit ce rôle ; la passe de prédiction lève alors une erreur si un
modèle de `X` lit `y`. Ajusté **sans** `y`, rien de cela ne s'applique : `transform` rejoue le plan
sur n'importe quelles données, et `fit_transform(X)` égale `fit(X).transform(X)`.

### Cible observée moins souvent que la grille

Quand `y` est trimestriel et la grille mensuelle, le modèle impute une **valeur mensuelle** qui n'est pas à la même échelle que les valeurs dans `y`(qui valent le **total trimestriel** sur la ligne d'ancre, NaN ailleurs). [`RollingAggregator`](../api/xy/RollingAggregator.md) (`window='Q'`)
résout l'écart : placé après l'imputeur, il fait de chaque mois un **trimestre glissant**, et, sur
une ligne d'ancre, la somme glissante redonne exactement le total observé. Il suit la même
convention que l'imputeur : `y` n'est agrégé qu'à la passe d'entraînement. Le sens de la fenêtre
suit la position de la grille (`alignment='auto'`) : passée `[t-2, t]` sur une grille de fin de
période (`ME`), future `[t, t+2]` sur une grille de début (`MS`).

![Les deux passes dans un XYPipeline](../assets/hfi_pipeline_passes.png)

[`DropNaNTarget`](../api/xy/DropNaNTarget.md), juste avant l'estimateur, retire les lignes sans
cible au `fit` et au `score`. Un scorer textuel (`scoring='neg_mean_absolute_error'`) comparerait les
prédictions au `y` de test brut, NaN compris, et lèverait une erreur. Utiliser le score par défaut du pipeline ou
un scorer restreint aux lignes observées permet de résoudre ce problème (voir le [tutoriel](../tutorials/mixed_frequency_imputation.md#7-dans-un-pipeline-une-cible-annuelle)).

Les délais de publication se comptent à la fréquence propre de chaque colonne : ils s'appliquent
**avant** l'imputeur, qui ne répartit alors que des totaux déjà publiés. `PublicationDelayTransformer`
n'est pas une étape XY et, sous `'shift'`, ajoute des dates à `X` : il s'applique donc à `X` avant
la validation croisée, ramené aux lignes de `y`. Le décalage ne dépend pas de la cible et
n'introduit pas de fuite. `prediction_date` le fixe pour toute la série : c'est la date à laquelle
les prévisions évaluées sont censées être produites ; laissée à `'today'`, elle rend le résultat
dépendant du jour d'exécution (voir [Pipeline XY](xy_pipeline.md#la-solution)).

```python
X_delayed = (
    PublicationDelayTransformer(delays=delays, strategy="shift", prediction_date="2024-01-15")
    .fit_transform(X)
    .reindex(X.index)
)

pipeline = XYPipeline([
    ("imputer", HighFrequencyImputer(target_frequency="M", estimator=LinearRegression(),
                                     keep_lower_frequencies=False)),
    ("roll", RollingAggregator(window="Q", agg="mean")),
    ("drop", DropNaNTarget()),
    ("model", HistGradientBoostingRegressor()),
])
```

## Architecture du code

`HighFrequencyImputer` est un orchestrateur : la classe valide les paramètres, publie les attributs
ajustés et enchaîne les phases ; chaque phase vit dans un module de
`tsforecast/imputation/_engine/`. Le `fit` et le `transform` partagent la **même préparation** et la
**même exécution d'étape**.

![fit construit le plan, transform le rejoue](../assets/hfi_fit_transform.png)

Le contexte d'exécution (`_RunContext`) n'est jamais stocké sur l'imputeur : chaque passe reçoit son
matérialiseur, ses registres, son calculateur de fenêtre, son traceur de provenance et ses
avertissements en attente. Un `transform` ne lit ni n'altère aucun état d'exécution du `fit`.

| Module (`_engine/`) | Rôle |
|---------------------|------|
| `params` | validateurs purs des paramètres du constructeur |
| `frequency_layout` | `FrequencyLayout` figé et `plan_frequencies` : fréquence de chaque couple (entité, colonne), catégories de variables, progression des étapes |
| `preparation` | `prepare_work_frame` : phases communes au `fit` et au `transform` |
| `run_context` | `FittedComponents` (configurés au `fit`, relus par toute passe) et `_RunContext` (état d'une passe) |
| `grids` | grilles de prédiction et masques de fenêtre d'une étape |
| `row_scaling` | `RowScaler` : diviseurs par ligne des jeux d'entraînement mutualisés |
| `stage_runner` | `StageRunner` : ajustement d'une étape (`fit_stage`) et exécution d'une étape du plan (`execute_step`) |
| `replay` | rejeu du plan, contrôles du `transform`, entités nées au `transform` |
| `output` | assemblage de la sortie et gestes de l'inversion |
| `warnings` | `WarningAccumulator` : un avertissement par famille et par passe |

Les composants réutilisables vivent dans `_components/` :

| Composant | Rôle |
|-----------|------|
| `CovariateMaterializer` | produit `X_train` / `X_pred` ; la prédiction ne dispose jamais de moins d'information que l'entraînement |
| `StageScaler` | met les deux côtés du modèle à la même échelle de fréquence |
| `AggregationConstraint` | recale les sous-périodes sur le total observé |
| `VariableOrderer` | ordonne les variables (`'frequency'` ou `'cv'`), expose `scores_` |
| `TrainingSetBuilder` | jeu d'entraînement mutualisé entre entités d'un panel |
| `ImputationWindowCalculator` | fenêtres de prédiction et d'entraînement |
| `TargetFrequencyValidator` | cohérence fréquence cible ↔ fréquences détectées |

`_engine/` et `_components/` sont des détails d'implémentation : l'API publique est celle de la
classe et des types réexportés par `tsforecast.imputation`. `FrequencyAligner` et `IndexRegularizer`
(`tsforecast.utils.frequency`) sont des outils autonomes que l'imputeur n'utilise pas.

## Pour aller plus loin

- Tutoriel : [Imputation multi-fréquences](../tutorials/mixed_frequency_imputation.md)
- Notebook : `notebooks/2 - QB - Mixed frequencies.ipynb`, qui illustre chaque paramètre
- API : [HighFrequencyImputer](../api/imputation/HighFrequencyImputer.md),
  [provenance](../api/imputation/provenance.md),
  [aggregation_constraint](../api/imputation/aggregation_constraint.md),
  [stage_runner](../api/imputation/engine/stage_runner.md) (internals),
  [RollingAggregator](../api/xy/RollingAggregator.md)
- Métriques de tracking : [`imputation_metrics`](../guides/mlflow_tracking.md)
