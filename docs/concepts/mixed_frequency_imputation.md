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
que l'étape produit. La matrice complète est exposée après `fit` dans
`imputation_provenance_`, et son résumé chiffré dans `provenance_statistics_`.

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
recalage ne change jamais la provenance d'une cellule.

## Composants auxiliaires

| Composant | Rôle |
|-----------|------|
| `CovariateMaterializer` | produit `X_train` / `X_pred` — porte l'invariant : jamais de dégradation fit → predict |
| `StageScaler` | met les deux côtés du modèle à la même échelle de fréquence |
| `AggregationConstraint` | recale les sous-périodes sur le total observé |
| `VariableOrderer` | ordonne les variables (`'frequency'` ou `'cv'`), expose `scores_` |
| `TrainingSetBuilder` | jeu d'entraînement mutualisé entre entités d'un panel |
| `ImputationWindowCalculator` | fenêtres de prédiction / d'entraînement |
| `FrequencyAligner` | agrégation / interpolation entre fréquences |
| `TargetFrequencyValidator` | cohérence fréquence cible ↔ fréquences détectées |
| `IndexRegularizer` | comble les trous d'un index irrégulier |

## Sortie

Le résultat empile les niveaux de fréquence sur le côté « entité » de l'index :
`(frequency, date)` pour une série, `(entité..., 'frequency', 'date')` pour un
panel. `keep_lower_frequencies=False` ne conserve que le niveau cible (paramètre
purement d'affichage, sans effet sur les valeurs).

## Pour aller plus loin

- Tutoriel : [Imputation multi-fréquences](../tutorials/mixed_frequency_imputation.md)
- API : [HighFrequencyImputer](../api/frequency/HighFrequencyImputer.md),
  [provenance](../api/frequency/provenance.md),
  [aggregation_constraint](../api/frequency/aggregation_constraint.md)
- Métriques de tracking : [`imputation_metrics`](../guides/mlflow_tracking.md)
