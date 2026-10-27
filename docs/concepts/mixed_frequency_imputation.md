# Imputation multi-fréquences

## Le problème

Un jeu de données de prévision mélange souvent des fréquences : ventes
mensuelles, PIB trimestriel, bilans annuels. Pour entraîner un modèle à la
fréquence la plus fine, il faut donner une valeur mensuelle aux variables qui ne
sont observées qu'une fois par trimestre ou par an — sans inventer d'information
que l'on ne pourrait pas justifier.

`HighFrequencyImputer` impute les colonnes de basse fréquence sur une **grille cible plus
fine**, en **cascade** (année → trimestre → mois si besoin), et **trace l'origine
de chaque cellule produite**.

L'hypothèse structurante est l'**additivité** : la valeur d'une période est la
somme de ses sous-périodes. Si les données ne sont pas additives (niveaux, taux),
on passe un `additive_transformer` (log, différenciation…) qui expose
`fit_transform` et `inverse_transform`.

## Les deux axes du paramétrage

Le comportement se règle sur **deux axes orthogonaux** :

### Axe 1 — `covariate_strategy` : comment une covariable trop peu observée est-elle donnée au modèle ?

| Valeur | Effet | Prérequis |
|--------|-------|-----------|
| `'interpolate'` (défaut) | la covariable est interpolée linéairement entre ses ancres | — |
| `'tolerate_nan'` | la covariable est donnée telle quelle, trous compris | **l'estimateur doit tolérer les NaN** (sinon repli interpolation) |
| `'model'` | la covariable est elle-même imputée par modèle, en cascade sur les variables selon `fit_predict_order` | — |

!!! warning "Regard vers le futur"
    `'interpolate'` lit l'ancre **future** pour interpoler. La sortie
    reconstruit l'histoire, elle ne simule pas un passage en temps réel : ne pas
    l'utiliser telle quelle pour un backtest pseudo-temps réel.

### Axe 2 — `impute_intermediate_frequencies` : la variable imputée passe-t-elle par des fréquences intermédiaires ?

| Valeur | Étapes intermédiaires | `y_train` |
|--------|----------------------|-----------|
| `False` (défaut) | non, saut direct vers la cible | ancres observées uniquement |
| `'covariates_only'` | oui | ancres observées uniquement (le bénéfice de la cascade va aux covariables) |
| `True` | oui | s'entraîne aussi sur ses propres imputations antérieures |

Les deux axes se composent sans se connaître : l'axe 1 gouverne les **colonnes**
données à l'estimateur, l'axe 2 les **lignes** de sa cible.

## Provenance

Chaque cellule du résultat porte une étiquette `ProvenanceType` :

- **non-modèle** : `ORIGINAL`, `AGGREGATED` (agrégation additive exacte),
  `INTERPOLATED` ;
- **modèle**, par degré de « souillure » croissant : `MODEL_ON_TRUE`,
  `MODEL_ON_INTERPOLATED`, `MODEL_ON_IMPUTED`, `MODEL_ON_IMPUTED_TARGET`,
  `MODEL_ON_IMPUTED_BOTH` ;
- `MODEL_UNANCHORED` : prédiction sans ancre (entités jamais observées, sous
  `impute_unobserved_entities=True`).

La provenance est **contagieuse** : sur un panel, une entité qui contribue des
cellules interpolées ou modélisées dégrade la provenance de toutes les cellules
que l'étape produit. La matrice complète de la **dernière passe** (`fit`, puis
chaque `transform`) est exposée dans `imputation_provenance_`, et son résumé
chiffré dans `provenance_statistics_`, compté sur le niveau cible. La matrice
d'un `transform` décrit les cellules de **sa sortie** : une colonne plus fine
que la ligne y porte l'agrégation exacte de la période (`AGGREGATED`), et une
cellule vide n'est jamais marquée (`NaN`).

La matrice est **catégorielle** (`PROVENANCE_DTYPE`) : ses catégories sont les dix
valeurs de `ProvenanceType`, une cellule se relit en simple chaîne (`'original'`,
`'model_on_true'`...) et se compare directement à un membre
(`provenance == ProvenanceType.ORIGINAL`). Chaque cellule tient sur un code
entier : comptages et masques restent rapides et peu gourmands en mémoire sur de
gros panels, et toute valeur hors vocabulaire est refusée à l'écriture.

L'**origine** d'une cellule produite (`'observed'`, `'interpolated'`, `'model'`),
lue par les étapes suivantes pour calculer leur souillure, suit le même principe
avec un catégoriel **ordonné** (`CELL_ORIGIN_DTYPE`) : ses catégories vont de la
moins à la plus souillée, si bien que l'origine la plus souillée d'une série est
son maximum (`max_cell_origin`). C'est le dtype des registres d'origines du
matérialiseur (`origin_store`) et de `row_origin` dans un jeu d'entraînement.

## Le plan, source de vérité unique

`fit` construit un `ImputationPlan` (`imputation_plan_`) : la liste ordonnée des
étapes, chacune décrivant une (fréquence, variable), sa voie de matérialisation,
son échelle et son estimateur. `transform` **rejoue** ce plan à l'identique.
Tous les autres attributs (`imputation_models_`, `frequency_progression_`,
`imputation_order_`) en sont des vues.

## Contrainte d'agrégation

Laissées libres, les sous-périodes prédites n'ont aucune raison de sommer à
l'observation qu'elles décrivent. `aggregation_constraint='sum'` (défaut) recale
les sous-périodes d'une période pour que leur agrégat égale le total observé :
une prédiction libre devient une vraie **désagrégation** de l'observation. Le
recalage ne change jamais la provenance d'une cellule. Une période que les
bornes des données (ou de la fenêtre) tronquent n'est pas recalée : un total
annuel ne s'impose pas à dix mois sur douze. Son total observé n'est alors plus
déductible de la sortie, ce qu'un avertissement unique par passe signale.

## Composants auxiliaires

| Composant | Rôle |
|-----------|------|
| `CovariateMaterializer` | produit `X_train` / `X_pred` — porte l'invariant : jamais de dégradation fit → predict |
| `StageScaler` | met les deux côtés du modèle à la même échelle de fréquence |
| `AggregationConstraint` | recale les sous-périodes sur le total observé |
| `VariableOrderer` | ordonne les variables (`'frequency'` ou `'cv'`), expose `scores_` |
| `TrainingSetBuilder` | jeu d'entraînement mutualisé entre entités d'un panel |
| `ImputationWindowCalculator` | fenêtres de prédiction / d'entraînement |
| `FrequencyAligner` | outil autonome (non utilisé par l'imputeur) : agrégation / interpolation de variables choisies entre fréquences |
| `TargetFrequencyValidator` | cohérence fréquence cible ↔ fréquences détectées |
| `IndexRegularizer` | comble les trous d'un index irrégulier |

## Sortie

Le résultat empile les niveaux de fréquence sur le côté « entité » de l'index :
`(frequency, date)` pour une série, `(entité..., 'frequency', 'date')` pour un
panel. `keep_lower_frequencies=False` ne conserve que le niveau cible (paramètre
purement d'affichage, sans effet sur les valeurs).

Le nom `'frequency'` est **réservé** : `fit` et `transform` refusent une entrée
dont un niveau d'index (ou une colonne de `panel_cols`, ou `time_col`) le porte
déjà, pour que l'index empilé n'ait jamais deux niveaux du même nom.
`inverse_transform` reconnaît une sortie empilée à sa profondeur et lit le niveau
de fréquence par position : renommer les niveaux de la sortie ne l'empêche pas
de l'inverser.

La sortie suit l'ordre de l'entrée. Elle porte les lignes de la grille cible :
une grille densifiée (index irrégulier) en ajoute, une cible plus grossière que
l'index en retire — la valeur d'une ligne retirée est agrégée sur la ligne de
sa période. `inverse_transform` défait la dernière passe et rend **l'index de
son entrée**, dans son ordre : sans `restore_original_values`, seules les
cellules `ORIGINAL` y reviennent ; avec, toutes les valeurs observées.

Les dates de toute passe (sortie, matrice de provenance, inversion, `y`) sont
stockées dans l'unité des dates du `X` de cette passe (`s`, `ms`, `us` ou `ns`) :
un `y` stocké dans une autre unité revient dans celle de `X`, ses valeurs
inchangées. L'état ajusté garde l'unité de l'entrée du `fit` ; un `transform` sur
des données dans une autre unité rend la leur, avec les mêmes valeurs.

## Dans un pipeline entraîné avec `y`

Entraîné avec `y`, `HighFrequencyImputer` **augmente un jeu d'entraînement** (`X`
et `y`) ; il n'a pas vocation à transformer des données de test. Un
`XYPipeline` appelle `fit_transform` sur ses étapes pendant son `fit`, et
`transform` pendant `predict` / `score` : les deux passes diffèrent.

| Appel | Passe | Ce qui sort |
|---|---|---|
| `fit_transform(X, y)` | entraînement | `X` **et** `y` sur la grille cible, lignes ajoutées comprises ; aucune ligne retirée |
| `transform(X)` / `transform(X, y)` | prédiction | les colonnes de `X` seules, imputées par les modèles figés, **sur les lignes d'entrée** (sortie plate) ; `y` rendu **inchangé** |

`y` est imputé comme une colonne, mais n'est **pas une covariable** des autres
colonnes par défaut (`target_as_covariate=False`) : une feature imputée à partir
de la cible en porte le bruit, gonfle les métriques sur l'entraînement et biaise
le modèle aval, et la passe de prédiction n'a pas de `y` pour alimenter un tel
modèle. `target_as_covariate=True` rétablit ce rôle ; la passe de prédiction
lève alors si un modèle de `X` lit `y`.

La passe d'entraînement garde les lignes sans cible : une étape aval peut encore
décaler `X` vers plusieurs horizons. Les retirer revient à
[`DropNaNTarget`](../api/xy/DropNaNTarget.md), placée **juste avant
l'estimateur** :

```python
pipeline = XYPipeline([
    ('delays', PublicationDelayTransformer(...)),
    ('imputer', HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                     keep_lower_frequencies=False)),
    ('drop', DropNaNTarget()),
    ('model', HistGradientBoostingRegressor()),
])
```

Ajusté **sans** `y`, rien de cela ne s'applique : `transform` rejoue le plan sur
n'importe quelles données, et `fit_transform(X)` égale `fit(X).transform(X)`.

### Limite : évaluer une cible à plus basse fréquence que la grille

Quand `y` à une fréquence plus basse que `target_frequency` (un PIB trimestriel
imputé au mois pour multiplier les lignes d'entraînement), le modèle apprend à
prédire une **valeur mensuelle**. À l'évaluation, `y_test` n'est pas imputé : il
porte le **total trimestriel** sur la ligne d'ancre, et NaN ailleurs. Après
`DropNaNTarget`, le score compare donc une prédiction mensuelle à un total
trimestriel : il s'exécute, mais **son échelle est fausse**.

Montage suggéré : agréger, **en aval** de l'imputeur, les données à la
fréquence de la cible, par exemple en **trimestres glissants** (sommes sur trois
mois). Chaque ligne mensuelle porte alors une quantité trimestrielle : la passe
d'entraînement garde trois fois plus de lignes qu'un jeu trimestriel, et, sur
les lignes d'ancre du test, la prédiction d'un trimestre glissant se compare
directement au total observé. Deux points de vigilance :

- **côté `y`**, l'agrégation n'a lieu qu'à la passe d'entraînement (le `y`
  imputé au mois devient des trimestres glissants) ; à la passe de prédiction,
  `y_test` reste tel qu'observé. Un transformateur de fenêtres glissantes qui ne
  traite que `X` ne suffit donc pas : il faut une étape XY qui suive la même
  convention que l'imputeur (`fit_transform` agrège `X` et `y`, `transform`
  agrège `X` seul) ;
- **l'alignement de la fenêtre** suit la position des ancres : une fenêtre
  **passée** (`[t-2, t]`) pour des ancres de fin de période (`ME`, `QE`), une
  fenêtre **future** (`[t, t+2]`) pour des ancres de début de période (`MS`,
  `QS`), sans quoi la somme ne couvre pas le trimestre observé.

## Entités et couples inconnus du `fit`

Au `transform`, `impute_unobserved_entities` gouverne aussi les couples
(entité, colonne) que le `fit` n'a jamais observés : les colonnes d'une entité
absente du `fit`, ou une colonne qu'une entité connue laissait vide. Sous
`False` (défaut) ils restent tels quels. Sous `True`, chacun est imputé comme
s'il avait été présent au `fit`, par le modèle figé de sa colonne : **ancré**
sur ses propres observations quand le `transform` l'observe, sans ancre
(`MODEL_UNANCHORED`) sinon.

## Architecture du code : un plan, un exécuteur

`HighFrequencyImputer` est un orchestrateur : la classe valide les paramètres, publie les attributs
ajustés et enchaîne les phases ; chaque phase vit dans un module de `tsforecast/imputation/_engine/`.
Le `fit` et le `transform` partagent la **même préparation** et la **même exécution d'étape** : le
`fit` construit chaque étape puis l'exécute, le `transform` rejoue le plan figé par cette même
méthode (`StageRunner.execute_step`).

```text
fit(X, y)                                         transform(X)
─────────                                         ────────────
validate_params, purge de la passe précédente
        │                                                 │
        ▼                                                 ▼
prepare_work_frame(fitting=True)  ◄── une seule ──►  prepare_work_frame(fitting=False)
  alignement de y, frame de travail,                   contrôle des fréquences contre le fit,
  plan des fréquences (FrequencyLayout),               fenêtres recalculées sur les données,
  fenêtres, transformateur additif,                    transformateur additif ajusté appliqué,
  traceur de provenance                                traceur de provenance
        │                                                 │
        ▼                                                 ▼
_RunContext.for_fit(FittedComponents, …)          _RunContext.for_transform(FittedComponents, …)
        │                                                 │
        ▼                                                 ▼
StageRunner.fit_stage(…)  par étape               replay_plan(runner, imputation_plan_, …)
  prepare_variable → ordre → ajustement             pour chaque étape du plan figé
  → build_step → execute_step ───────┐         ┌──► execute_step (prédire avec le modèle figé)
        │                            └─ même ──┘          │
        ▼                               méthode           ▼
ImputationPlan figé (imputation_plan_)            assemble_output(…) : niveau cible, empilement
                                                  multi-fréquences, provenance, statistiques
```

Extraits de la classe (`high_frequency_imputer.py`), réduits aux appels :

```python
# fit : préparation, contexte d'exécution neuf, étapes l'une après l'autre
prepared = prepare_work_frame(self, X, y, fitting=True)
self._publish_fit_preparation(X, prepared)
run = _RunContext.for_fit(self._fitted_components_, window_calculator=..., ...)
runner = self._stage_runner(run)
for stage_freq in self.frequency_progression_:
    stage = runner.fit_stage(prepared.X_work, stage_freq)
    ...  # les étapes de `stage.steps` rejoignent `imputation_plan_`

# transform : même préparation, contexte construit depuis l'état ajusté seul
prepared = prepare_work_frame(self, X, y, fitting=False, prediction=prediction)
run = _RunContext.for_transform(self._fitted_components_, ...)
frames, provenances = replay_plan(runner, self.imputation_plan_, prepared.X_work, ...)
output = assemble_output(frames, provenances, prediction=prediction, ...)
```

Le contexte d'exécution (`_RunContext`) n'est jamais stocké sur l'imputeur : chaque passe reçoit
son matérialiseur, ses registres, son calculateur de fenêtre, son traceur de provenance et ses
avertissements en attente. Un `transform` ne lit donc ni n'altère aucun état d'exécution du `fit`.

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

Les composants réutilisables (`_components/`) sont décrits plus haut ; `_engine/` regroupe les
étapes propres à l'algorithme de `HighFrequencyImputer`. Les deux sont des détails
d'implémentation : l'API publique est celle de la classe et des types réexportés par
`tsforecast.imputation`.

## Pour aller plus loin

- Tutoriel : [Imputation multi-fréquences](../tutorials/mixed_frequency_imputation.md)
- API : [HighFrequencyImputer](../api/imputation/HighFrequencyImputer.md),
  [provenance](../api/imputation/provenance.md),
  [aggregation_constraint](../api/imputation/aggregation_constraint.md),
  [stage_runner](../api/imputation/engine/stage_runner.md) (internals)
- Métriques de tracking : [`imputation_metrics`](../guides/mlflow_tracking.md)
