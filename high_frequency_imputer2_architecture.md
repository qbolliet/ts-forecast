# `HighFrequencyImputer2` — spécification d'architecture

> **Version 2 du document — 2026-09-01** (branche `qb-mixed-frequencies`).
> Rédigé initialement le 2026-08-31 à partir de la spécification orale du 2026-08-31, de la
> relecture de `_fit` dans `tsforecast/frequency/high_frequency_imputer.py`, et des deux
> documents de la campagne d'annotations : `high_frequency_imputer_annotations_architecture.md`
> (noté ci-après **[ARCH]**) et `high_frequency_imputer_annotations_prompts.md` (prompts 1 à 12
> et 22 exécutés). **Révisé le 2026-09-01** pour intégrer les arbitrages de l'auteur : toutes les
> décisions ouvertes du §11 de la version 1 sont désormais tranchées et consignées au §14.
>
> **Objet** : spécifier `HighFrequencyImputer2` (`tsforecast/frequency/high_frequency_imputer2.py`),
> réécriture from scratch de `HighFrequencyImputer` destinée à la remplacer dans le package.
>
> **Statut** : spécification **arrêtée**. Ce document est la référence unique de l'implémentation
> et de sa documentation. Il ne reste aucune décision de conception ouverte ; le §17 en dérive
> les lots d'implémentation, base de la future liste de prompts.
>
> **Convention de lecture** : `hfi:` = `tsforecast/frequency/high_frequency_imputer.py`
> (implémentation actuelle), `hfi2:` = la classe spécifiée ici,
> `iwc:` = `tsforecast/frequency/imputation_window.py`. Localiser par nom de symbole, jamais par
> numéro de ligne. Les codes `Bxx` renvoient aux défauts de [ARCH], les codes `Cxx` à ses
> constats.

---

## Table des matières

0. [Résumé et décisions structurantes](#0--résumé-et-décisions-structurantes)
1. [Contexte : pourquoi repartir d'un fichier vierge](#1--contexte--pourquoi-repartir-dun-fichier-vierge)
2. [Vocabulaire normatif et jeux de référence](#2--vocabulaire-normatif-et-jeux-de-référence)
3. [L'invariant central](#3--linvariant-central)
4. [Axe 1 — matérialisation des covariables](#4--axe-1--matérialisation-des-covariables)
5. [Axe 2 — imputation des fréquences intermédiaires](#5--axe-2--imputation-des-fréquences-intermédiaires)
6. [Provenance](#6--provenance)
7. [Fenêtres d'entraînement et d'imputation](#7--fenêtres-dentraînement-et-dimputation)
8. [Ordre d'imputation](#8--ordre-dimputation)
9. [Mise à l'échelle des données](#9--mise-à-léchelle-des-données)
10. [Interpolation et position d'ancrage](#10--interpolation-et-position-dancrage)
11. [Contrainte d'agrégation](#11--contrainte-dagrégation)
12. [Architecture logicielle](#12--architecture-logicielle)
13. [API résultante, paramètre par paramètre](#13--api-résultante-paramètre-par-paramètre)
14. [Journal des décisions arrêtées](#14--journal-des-décisions-arrêtées)
15. [Prérequis et travaux annexes](#15--prérequis-et-travaux-annexes)
16. [Stratégie de tests et invariants](#16--stratégie-de-tests-et-invariants)
17. [Lots d'implémentation](#17--lots-dimplémentation)

---

## 0 — Résumé et décisions structurantes

`HighFrequencyImputer2` remplace la sémantique d'entraînement de `HighFrequencyImputer` — un
empilement de gardes (`cascade_refitting`, `covariate_eligibility`, `train_on_partial_coverage`,
règle de matérialisation §3.17 de [ARCH]) — par **deux axes orthogonaux et explicites** :

| Axe | Paramètre | Question à laquelle il répond |
|---|---|---|
| 1 | `covariate_strategy: Literal['tolerate_nan', 'interpolate', 'model']` | comment une **covariable** de fréquence plus basse que la grille courante est-elle mise à disposition du modèle ? |
| 2 | `impute_intermediate_frequencies: Literal[False, 'covariates_only', True]` | la variable imputée traverse-t-elle des **fréquences intermédiaires**, et son modèle final s'entraîne-t-il sur ses propres imputations ? |

Le défaut structurel mesuré (B28 : 0 % de NaN au `fit` contre jusqu'à 67 % au `predict`) devient
**inexprimable par construction** : la préparation des features passe par un composant unique,
appliqué à l'identique aux deux grilles.

**Les huit décisions structurantes arrêtées** (détail et justification au §14) :

1. **Provenance en échelle de souillure** : `MODEL_ON_TRUE` → `MODEL_ON_INTERPOLATED` →
   `MODEL_ON_IMPUTED`, plus deux libellés distinguant la souillure venue de la **cible** :
   `MODEL_ON_IMPUTED_TARGET` et `MODEL_ON_IMPUTED_BOTH`. `MODEL_ON_MIXED` est **supprimé** de
   l'énumération (§6, D6).
2. **`impute_intermediate_frequencies` à trois modalités** `False` / `'covariates_only'` /
   `True`, défaut `False` (§5, D3).
3. **`aggregation_constraint: Literal['sum', None] = 'sum'`** remplace `enforce_period_totals`,
   avec une extension réservée (dict par colonne, contraintes `'mean'`/`'last'`) (§11, D8).
4. **Les ancres sont toujours désagrégées** : une variable imputée à l'étape `f` est prédite sur
   **toute** sa période, ancres comprises, pour qu'une colonne ne mélange jamais total de période
   et valeurs de sous-période. **Non paramétrable** — `disaggregate_anchors` n'existe pas (§11.2,
   D7).
5. **`cv` sklearn polymorphe** remplace `cv_n_splits`, qui est supprimé (§8.3, D4).
6. **Le mode « un seul fit réutilisé »** (`cascade_refitting=False`) est **abandonné** (D2).
7. **Paramètres inertes : silence + docstring**, jamais d'avertissement (D9).
8. **Les dictionnaires par feature sont indexés par nom de colonne**, jamais par
   `(entité, colonne)` (D10). **`transform` face à des fréquences divergentes** : avertissement
   et poursuite avec les fréquences du fit, erreur si une colonne du fit manque (D11).

---

## 1 — Contexte : pourquoi repartir d'un fichier vierge

### 1.1 — Le défaut structurel mesuré

Série temporelle `m1` mensuelle, `q1` trimestrielle, `a1`/`a2` annuelles, cible `M`,
`cascade_refitting=True` (le défaut de la classe), estimateur espion enregistrant le taux de NaN
reçu :

| Étape | Variable | `feature_cols` | NaN au `fit` | NaN au `predict` |
|-------|----------|----------------|--------------|------------------|
| `Q` | `a1` | `m1, q1, a2` | 0 % | **33 %** |
| `Q` | `a2` | `m1, q1, a1` | 0 % | 0 % |
| `M` | `a1` | `m1, q1, a2` | 0 % | **67 %** |
| `M` | `a2` | `m1, q1, a1` | 0 % | **33 %** |

Trois causes, toutes structurelles ([ARCH] §3.17, défaut B28) :

- une variable imputée à une étape **antérieure** porte les valeurs de **cette** étape : `a1`
  imputée en trimestriel reste NaN deux mois sur trois à l'étape mensuelle ;
- l'ordre intra-étape range les variables de **même** fréquence arbitrairement : la première ne
  voit aucune de ses jumelles, la dernière les voit toutes — le résultat dépend de l'ordre des
  colonnes en entrée ;
- un repli par interpolation n'alimente ni `imputed_store` ni le miroir des covariables : la
  variable en repli n'est jamais matérialisée, où qu'elle soit dans l'ordre.

Sous un estimateur ne tolérant pas les NaN, tout le groupe bascule en repli interpolation ; sous
un estimateur tolérant, il prédit à partir de features majoritairement absentes, silencieusement.

`hfi2` répond aux trois causes respectivement par : la **règle de report d'étape** (§4.6, un
imputé d'étape antérieure est reporté sur la grille courante, jamais laissé troué), le
**tie-break alphabétique** (§8.1), et la règle **« le repli matérialise »** (§4.4).

### 1.2 — Le constat de complexité

`hfi` fait ~4 500 lignes et plus de 60 méthodes ; la sémantique d'entraînement est éclatée entre
`cascade_refitting`, `covariate_eligibility`, `train_on_partial_coverage` (supprimé par [ARCH]
§3.7), la règle de matérialisation (§3.17), et trois blocs de repli qui ont déjà divergé une fois
(B27). Les correctifs des prompts 5 à 12 ont fiabilisé les fondations (symétrie fit/transform,
échelles, conformité clone, fenêtres) mais la **logique de disponibilité des covariables** reste
un empilement de gardes. `HighFrequencyImputer2` repart de cette logique-là, en conservant ce qui
est sain (phases 0 à 4 de `_fit`, le plan d'étapes rejouable, la provenance, les contraintes
additives).

### 1.3 — Ce qui est explicitement conservé de `HighFrequencyImputer`

- **Arguments conservés à l'identique** (définition, type, rôle) : `target_frequency`,
  `estimator`, `additive_transformer`, `on_frequency_mismatch`, `min_cv_train_size`,
  `cv_scoring`, `restore_original_values`, `keep_lower_frequencies`, `covariate_eligibility`,
  `imputation_scope`, `coverage_threshold`, plus les paramètres de la classe de base `time_col`,
  `panel_cols`, `verbose`.
- **Validations d'arguments** : l'intégralité du bloc de validation de `hfi:__init__` est
  reprise (format de `target_frequency`, contrat de `estimator` — y compris la forme dict avec
  clé `'__default__'` —, `additive_transformer` avec `fit_transform`/`inverse_transform`,
  bornes numériques, `Literal`s, validation groupée des booléens, avertissement unique
  `estimator=None`). La règle B3 reste impérative : **valider sans transformer**, stocker les
  paramètres tels que reçus, normaliser au `fit` dans des attributs suffixés `_`.
- **Phases 0 à 4 de `_fit`** : setup (purge de l'état, colonnes, détection panel, alignement et
  nommage de `y`, détection de fréquences, normalisation/validation de la fréquence cible,
  classification des variables), calcul de la fenêtre d'imputation, transformateur additif,
  liste des fréquences de prédiction, initialisation de la provenance — dans cet ordre, avec le
  tracker initialisé **après** le transformateur additif dans les deux chemins (B8).
- **`transform` rejoue les mêmes transformations et imputations que `fit`** (même plan
  d'étapes), et **`inverse_transform`** restitue les données originales comme aujourd'hui
  (niveau de fréquence source, inversion du transformateur additif, masque `ORIGINAL`,
  `restore_original_values`).
- **Appui sur les classes existantes de `tsforecast/frequency`** : `FrequencyAligner`,
  `FrequencyConverter`, `ImputationWindowCalculator`, `TargetFrequencyValidator`,
  `ImputationProvenanceTracker`, `ImputationStep` (adapté), la logique de
  `hfi:_write_interpolation_fallback` (portée dans le nouveau composant d'interpolation), la
  logique de `hfi:_rescale_to_period_totals` (portée dans le composant de contrainte).

### 1.4 — Ce qui disparaît, ce qui est renommé

| Paramètre `hfi` | Sort dans `hfi2` | Référence |
|---|---|---|
| `cascade_refitting` | **supprimé**, remplacé par le couple (`covariate_strategy`, `impute_intermediate_frequencies`) ; le mode « un seul fit réutilisé » est abandonné | §5, D2 |
| `train_on_partial_coverage` | **supprimé** (déjà condamné par [ARCH] §3.7) ; remplacé par `training_scope` | §7 |
| `train_on_partial_fit_order` | **renommé** `fit_predict_order`, mêmes modalités `'frequency'`/`'cv'`, champ d'application restreint | §8.1 |
| `scale_features: bool` | **élargi** en `Union[Literal[False], 'constant', 'calendar', Dict[...]]` | §9.1 |
| `enforce_period_totals: bool` | **remplacé** par `aggregation_constraint: Literal['sum', None]` | §11.1, D8 |
| `cv_n_splits: int` | **supprimé**, absorbé par `cv` (contrat sklearn) | §8.3, D4 |
| `covariate_eligibility` | **conservé**, sémantique recadrée sur le seul cas « feature absente pour la totalité d'une entité » | §4.5 |
| `keep_lower_frequencies` | **conservé** tel quel, paramètre d'affichage pur, correctif B4 inclus | §12.4 |
| — | **nouveaux** : `covariate_strategy`, `covariate_fallback`, `impute_intermediate_frequencies`, `interpolation_method`, `interpolation_anchor`, `cv`, `training_scope`, `training_coverage_threshold` | §13 |

Deux constantes de l'énumération de provenance disparaissent ou changent :
`ProvenanceType.MODEL_ON_MIXED` est **supprimée** (§6.7), et le champ
`ImputationStep.trained_on_imputed: bool` est remplacé par le couple
`(covariate_taint, target_taint)` (§6.2).

---

## 2 — Vocabulaire normatif et jeux de référence

Tout le document utilise ce vocabulaire ; l'implémentation doit reprendre ces mots dans les noms
de variables et les docstrings.

### 2.1 — Vocabulaire

| Terme | Définition normative |
|---|---|
| **fréquence cible** `f_target` | la fréquence à laquelle toutes les colonnes imputables doivent être disponibles en sortie (`target_frequency`, éventuellement par entité). |
| **fréquence détectée** `f_var` | la fréquence propre d'une colonne, inférée en phase 0 par `FrequencyDetector` sur les dates où elle est observée. |
| **étape** (*stage*) | un couple (fréquence de prédiction `f_stage`, jeu de variables à imputer à cette fréquence). Les étapes s'enchaînent de la plus basse à la plus haute fréquence. |
| **grille** | l'index des lignes d'un jeu de données à une fréquence donnée. Grille **d'entraînement** = lignes retenues pour le `fit` d'un modèle ; grille **de prédiction** = lignes pour lesquelles il produit des valeurs. |
| **ancre** (*anchor*) | une ligne où une colonne porte une valeur **observée** à sa propre fréquence. `a1` annuelle observée au 2021-12-31 a une ancre à cette date. |
| **variable imputable** | colonne dont `f_var` est strictement plus basse que `f_target` : elle doit être imputée. |
| **covariable** | toute colonne utilisée comme feature d'un modèle. Une variable imputable est covariable des autres. |
| **matérialisation** | l'opération qui rend une covariable disponible sur une grille où elle n'est pas observée : agrégation, identité, interpolation, ou imputation par modèle. |
| **voie de matérialisation** | le chemin retenu pour un triplet (étape, variable imputée, covariable). Choisi **une fois** et appliqué **aux deux grilles** (§4.6). |
| **origine d'une cellule** (`CellOrigin`) | `'observed'` \| `'interpolated'` \| `'model'` : le degré épistémique d'une cellule, tenu par un registre interne (§6.2). Distinct de la provenance publique, qui dit *où* la valeur se situe autant que *d'où* elle vient. |
| **souillure** (*taint*) | le maximum des origines des cellules **effectivement vues** par un modèle, sur l'axe covariables et sur l'axe cible (§6.2). |
| **progression de fréquences** | la liste ordonnée des `f_stage`, construite au §5.2. |

### 2.2 — Jeu de référence `TS` (série temporelle)

Utilisé dans tous les exemples chiffrés du document. Index mensuel fin de mois,
2021-01-31 → 2023-12-31 (36 lignes), `target_frequency='M'` :

| Colonne | `f_var` | Observations | Valeurs utilisées dans les exemples |
|---|---|---|---|
| `m1` | `M` | 36 lignes (toutes) | quelconques, jamais NaN |
| `q1` | `Q` | 12 ancres (fins de trimestre) | quelconques |
| `a1` | `Y` | 3 ancres (2021-12-31, 2022-12-31, 2023-12-31) | **120, 132, 150** |
| `a2` | `Y` | 3 ancres, mêmes dates | **60, 66, 72** |

Toutes les colonnes sont additives (une valeur annuelle est la **somme** de ses sous-périodes) ;
c'est le contrat par défaut, `additive_transformer` étant chargé de rendre additives les colonnes
qui ne le sont pas.

Fréquences détectées : `{m1: M, q1: Q, a1: Y, a2: Y}`. Variables imputables : `q1`, `a1`, `a2`.

### 2.3 — Jeu de référence `PANEL`

Trois entités `FR`, `DE`, `IT`, mêmes colonnes que `TS`, plus une colonne `climat_affaires`
mensuelle **entièrement absente pour `IT`** (la colonne existe dans le frame, l'entité ne
l'observe jamais). Ce jeu est produit par le notebook 3 (§15.1) ; il est le support des tests de
`covariate_eligibility` (§4.5) et de la mesure **par entité** de l'invariant central.

### 2.4 — Notation des exemples

Les tableaux d'exemple donnent, pour une étape et une variable : `X_train` (features vues au
`fit`), `y_train` (cible du `fit`), `X_pred` (features vues au `predict`) et la provenance
écrite. Les valeurs numériques sont exactes et vérifiables : les tests du §16 les reprennent
telles quelles comme cas d'or.

---

## 3 — L'invariant central

Toute la conception découle d'une règle unique, qui est la leçon de B28 :

> **Un modèle ne voit jamais, à la prédiction, un motif de disponibilité de features plus
> dégradé qu'à l'entraînement, ni des features de nature différente.** Pour chaque étape du plan
> et chaque colonne, (a) le taux de NaN de `X_pred` est ≤ celui de `X_train`, et (b) la voie de
> matérialisation employée est la même des deux côtés. L'exception unique, mesurée **par
> entité**, est celle des entités structurellement dépourvues de la colonne (§4.5).

Cet invariant est **testable mécaniquement** (estimateur espion, tests I2 et I11 du §16) et
chaque stratégie de covariables doit le garantir **par construction**, pas par filtrage
a posteriori :

- `'tolerate_nan'` : les covariables ne sont jamais matérialisées au-delà de leurs observations ;
  le motif de NaN est celui des données, identique au fit et au predict par définition — mais il
  doit être **construit par la même routine** des deux côtés ;
- `'interpolate'` : toutes les covariables sont matérialisées partout (sauf entité intégralement
  vide) ; taux de NaN ≈ 0 des deux côtés ;
- `'model'` : une covariable non encore matérialisée au moment où une variable est imputée est
  traitée par l'approche secondaire (`covariate_fallback`), **au fit comme au predict** — jamais
  « pleine au fit, vide au predict ».

**Corollaire structurel** : la préparation des features (`X_train` **et** `X_pred`) passe par
**une seule méthode partagée** (`CovariateMaterializer.materialize`, §12.2), appelée par le fit
et par le transform. Les défauts B7/B27 (« deux blocs censés être identiques qui divergent »)
deviennent impossibles : il n'existe qu'un bloc.

---

## 4 — Axe 1 — matérialisation des covariables

### 4.1 — Définition et périmètre

```python
covariate_strategy: Literal['tolerate_nan', 'interpolate', 'model'] = 'interpolate'
covariate_fallback: Literal['interpolate', 'tolerate_nan'] = 'interpolate'
```

`covariate_strategy` décide comment une covariable **non observée à la fréquence de la grille
courante** est mise à disposition du modèle. Il ne concerne **que** les covariables de fréquence
strictement plus basse que la grille : une covariable de fréquence supérieure ou égale est
toujours disponible, par agrégation exacte (`m1` sommée au trimestre) ou par identité (`q1` sur
une grille `Q`).

Classification d'une covariable `c` face à une grille `f` :

| Cas | Traitement | Origine des cellules |
|---|---|---|
| `f_c == f` | identité | `'observed'` |
| `f_c` plus **fine** que `f` | agrégation exacte selon la contrainte d'agrégation (somme par défaut), via `FrequencyConverter.aggregate_to_lower_frequency(..., full_periods_only=True)` | `'observed'` |
| `f_c` plus **basse** que `f` | **`covariate_strategy`** | §4.2 à §4.4 |

Une période incomplète en agrégation (`full_periods_only=True`) produit NaN : c'est une source
légitime de NaN, identique au fit et au predict, et le correctif B26 (§15, P1) est le prérequis
qui empêche cette règle de jeter silencieusement les mois de 28 jours.

### 4.2 — `'tolerate_nan'`

Aucune matérialisation. Une covariable plus basse fréquence que la grille porte ses valeurs aux
seules **dates-ancres** (à l'échelle de sa propre fréquence) et NaN partout ailleurs.

- **Contrat d'estimateur** : l'estimateur doit tolérer les NaN
  (`HistGradientBoostingRegressor`, `LGBMRegressor`, ou un `Pipeline` incluant un
  `SimpleImputer`). La docstring de `estimator` l'énonce déjà comme recommandation ; c'est ici un
  **prérequis dur** de la modalité, à répéter dans la docstring de `covariate_strategy`.
- **Ordre indifférent** : aucune covariable ne dépend d'une imputation antérieure ;
  `fit_predict_order` est ignoré (§8.1).
- **Diviseur d'échelle** : règle B25 conservée — `1.0` pour une colonne jamais ré-agrégée (ses
  ancres gardent l'échelle de `f_c`), `get_conversion_factor(f_stage, f_c)` sinon, avec
  `f_stage = pred_freq` si `f_c` est plus fine que l'étape et `f_c` sinon (§9.2).

Sur le jeu `TS`, étape `M`, covariable `a1` : `a1` vaut 120 au 2021-12-31, NaN sur les 11 autres
mois de 2021 — **au fit comme au predict**, taux de NaN 11/12 des deux côtés.

### 4.3 — `'interpolate'` (défaut)

Toute covariable plus basse fréquence que la grille est **interpolée** sur la grille à partir de
ses valeurs observées (les ancres servant de points de référence), puis **recalée pour préserver
les totaux de période** quand `aggregation_constraint` est active (§11). C'est le pendant exact,
sur la covariable, de ce que la classe produit sur une variable imputée, et c'est la logique déjà
unifiée dans `hfi:_write_interpolation_fallback` (B27, prompt 22).

- Méthode d'interpolation : `interpolation_method` (§10.1), globale ou par feature (dict).
- Position d'ancrage : `interpolation_anchor` (§10.2).
- Supprime **tous** les NaN, sauf (a) les features absentes pour la **totalité** d'une entité
  (§4.5) et (b) les bords de série au-delà de ce que `limit_direction` autorise.
- **Ordre indifférent** : l'interpolation est déterministe à partir des seules observations,
  sans dépendance aux imputations précédentes. `fit_predict_order` est ignoré (§8.1).
- **Origine des cellules produites** : `'interpolated'` ; provenance publique `INTERPOLATED`
  (ou `DISAGGREGATED` pour les cellules recalées d'une période complète, §11.2). Un modèle qui en
  consomme émet `MODEL_ON_INTERPOLATED` (§6.3).
- Cette même méthode sert de **repli** partout où une imputation par modèle échoue, dans toutes
  les stratégies (l'actuel `INTERPOLATE_FALLBACK`).

**Avertissement à documenter (regard vers l'aval)** : l'interpolation linéaire entre deux ancres
utilise l'ancre **future** — dans un usage pseudo-temps réel, c'est une information de futur. Ce
n'est pas un défaut pour de l'imputation d'historique, mais la docstring de `covariate_strategy`
et celle de `interpolation_method` doivent le dire, et `imputation_scope='extended_forward'`
reste le mécanisme dédié aux fins de série.

### 4.4 — `'model'` et la précédence de matérialisation

Les covariables manquantes sont imputées par le même mécanisme fit/predict que les variables
cibles, **dans l'ordre défini par `fit_predict_order`** (§8). C'est le seul mode où l'ordre a un
impact réel.

Quand `hfi2` doit matérialiser la covariable `c` sur la grille de l'étape `f`, il applique la
**précédence** suivante, dans cet ordre, et s'arrête au premier cas applicable :

| Rang | Condition | Voie retenue | Origine |
|---|---|---|---|
| 1 | `f_c >= f` | identité ou agrégation exacte (§4.1) | `'observed'` |
| 2 | `c` a déjà été imputée **à l'étape courante** `f` (elle précède la variable en cours dans l'ordre) | ses valeurs imputées à `f`, lues dans le miroir | `'model'`, ou `'interpolated'` si son étape a été produite par repli |
| 3 | `c` a été imputée à une étape **antérieure** `f'` (`f'` plus basse que `f`) — cas possible seulement sous `impute_intermediate_frequencies != False` | ses valeurs imputées à `f'`, **reportées** sur la grille `f` par la voie d'interpolation de `c` (méthode, ancrage, recalage aux totaux de `f'`) | `'model'` (l'interpolation d'une valeur de modèle reste de modèle) ; `'interpolated'` si l'imputation à `f'` était elle-même un repli |
| 4 | aucun des cas précédents | **`covariate_fallback`** : `'interpolate'` → interpolation des seules observations de `c` (§4.3) ; `'tolerate_nan'` → ancres + NaN (§4.2) | `'interpolated'` ou `'observed'` |

Le rang 3 est la réponse à la **première cause de B28** : une covariable imputée en trimestriel
n'est plus laissée NaN deux mois sur trois à l'étape mensuelle, elle est reportée. C'est aussi ce
qui donne un effet observable à `impute_intermediate_frequencies='covariates_only'` (§5.6).

Le rang 2 conserve le miroir des imputations (`imputed_store` de `hfi`) avec les correctifs B5
(`predictions.combine_first(existing)`) et **« le repli matérialise »** (troisième cause de B28) ;
il est doublé du registre de fréquences (`imputed_freq_store`, nécessaire aux diviseurs par
ligne, §9.3) et du registre d'origines (`origin_store`, §6.2).

**Règle de matérialisation** ([ARCH] §3.17, reformulée) : une covariable est « matérialisée » sur
la grille de `f` si et seulement si elle relève des rangs 1, 2 ou 3 ci-dessus. Les covariables de
rang 4 sont servies par l'approche secondaire, **au fit comme au predict**.

### 4.5 — `covariate_eligibility` : feature absente pour la totalité d'une entité

Aucune stratégie ne peut fabriquer des valeurs pour une entité où la feature n'a **aucune**
observation (cas de `climat_affaires` pour `IT`, §2.3). Le paramètre de `hfi` est conservé, sa
sémantique recadrée sur ce seul cas :

```python
covariate_eligibility: Literal['any_entity', 'all_entities'] = 'any_entity'
```

- `'any_entity'` (défaut) : la colonne est retenue dès qu'au moins une entité l'observe ; les
  lignes des entités vides restent NaN et relèvent du contrat NaN de l'estimateur — c'est
  l'**unique** source de NaN résiduels sous `covariate_strategy='interpolate'` ;
- `'all_entities'` : la colonne est écartée de `feature_cols` si une entité ne l'observe pas —
  pour les estimateurs qui ne tolèrent pas les NaN.

L'exception à l'invariant du §3 est exactement celle-là : les lignes d'une entité
structurellement vide sont NaN au predict comme au fit. **L'invariant se mesure par entité.**

### 4.6 — Règle d'unicité de la voie de matérialisation

> **Pour chaque triplet (étape `f`, variable imputée `v`, covariable `c`), la voie de
> matérialisation est choisie une seule fois, enregistrée dans l'étape du plan, et appliquée
> à `X_train` comme à `X_pred`, au `fit` comme au `transform`.**

Conséquence non intuitive mais impérative : sous `'model'`, si `c` est servie par le
`covariate_fallback` au predict (rang 4), la version vue au **fit** doit être préparée par la
**même** voie — interpolée sur la grille d'entraînement si `covariate_fallback='interpolate'`,
laissée à ses ancres si `'tolerate_nan'` — **même lorsque ses ancres suffiraient**. Sinon le
modèle apprend sur la covariable exacte et prédit sur la covariable interpolée. C'est la
généralisation de l'invariant du §3, du motif de NaN à la **nature** des valeurs.

Le champ correspondant de l'étape est
`materialization: Mapping[str, Literal['identity', 'aggregate', 'stage_model', 'carried_model', 'interpolate', 'raw_anchors']]`,
une entrée par colonne de `feature_cols`. Il est **rejoué tel quel au transform** ; c'est lui
qui rend le test I11 (§16) possible.

### 4.7 — L'exemple de référence sous chaque stratégie

Jeu `TS`, imputation de `a1` (annuelle, valeurs 120 / 132 / 150) à l'étape `Q`,
`feature_cols = [m1, q1, a2]`, `impute_intermediate_frequencies=True` (pour que l'étape `Q`
existe) :

| Stratégie | `X_train` (grille Y, 3 lignes) | `X_pred` (grille Q, 12 lignes) | NaN fit / predict de `a2` | Provenance de `a1` |
|---|---|---|---|---|
| `'tolerate_nan'` | `m1` sommée à Y, `q1` sommée à Y, `a2` à ses ancres Y (complète) | `m1`, `q1` sommées/identité à Q ; `a2` NaN sauf aux 3 fins d'année | 0 % / **75 %** → **interdit** : l'invariant impose de dégrader aussi `X_train`… voir note | `MODEL_ON_TRUE` |
| `'interpolate'` | idem, `a2` complète | `a2` interpolée sur la grille Q, recalée aux totaux annuels (60 → 15/15/15/15 si linéaire plate) | 0 % / 0 % | `MODEL_ON_INTERPOLATED` |
| `'model'`, ordre `a2` avant `a1` | idem | `a2` = son imputation Q (rang 2) | 0 % / 0 % | `MODEL_ON_IMPUTED` |
| `'model'`, ordre `a1` avant `a2` | `a2` préparée par `covariate_fallback` (interpolée sur la grille Y, donc identique à ses ancres) | `a2` interpolée sur Q (rang 4) | 0 % / 0 % | `MODEL_ON_INTERPOLATED` |

**Note sur la première ligne** : sous `'tolerate_nan'`, `a2` est à ses ancres des deux côtés — le
taux de NaN diffère uniquement parce que les deux grilles n'ont pas le même pas. L'invariant du
§3 se mesure donc **par ligne d'ancre commune**, pas par taux brut : la formulation testable est
« pour chaque couple (étape, colonne), l'ensemble des dates où la colonne est renseignée dans
`X_pred` contient l'image, sur la grille de prédiction, de l'ensemble des dates où elle est
renseignée dans `X_train` ». Sous `'tolerate_nan'`, cette inclusion est vraie (les 3 ancres
annuelles sont renseignées dans `X_pred`), et le test I2 doit être écrit ainsi, pas comme une
comparaison naïve de pourcentages. Le taux brut reste le bon critère sous `'interpolate'` et
`'model'`, où les deux grilles sont pleines.

La dernière ligne est le cas qui produisait 33–67 % de NaN silencieux dans `hfi` : il est ici
défini, symétrique et tracé.

---

## 5 — Axe 2 — imputation des fréquences intermédiaires

### 5.1 — Les trois modalités

```python
impute_intermediate_frequencies: Literal[False, 'covariates_only', True] = False
```

| Valeur | Progression de fréquences | `y_train` d'une variable `v` à l'étape `f` |
|---|---|---|
| `False` (défaut) | **une seule étape**, à la fréquence cible | ses ancres uniquement (origine `'observed'`) |
| `'covariates_only'` | progression **complète** (Y → Q → M sur `TS`) | ses ancres + ses cellules d'origine `'interpolated'` — **jamais** ses propres imputations de modèle |
| `True` | progression **complète** | ses ancres + ses cellules `'interpolated'` + **ses propres imputations des étapes antérieures** (origine `'model'`) |

Autrement dit : `'covariates_only'` et `True` produisent **le même plan d'étapes** et diffèrent
uniquement par le **filtre d'origine de `y_train`** ; `False` et `'covariates_only'` appliquent
**le même filtre** et diffèrent uniquement par le plan.

- `True` est ce qui « augmente le nombre d'observations sur lesquelles est entraîné le modèle
  servant à l'imputation finale » : sur `TS`, `a1` passe de 3 lignes d'entraînement (3 ancres
  annuelles) à 12 lignes (ses 12 imputations trimestrielles). En contrepartie `y_train` porte des
  valeurs bruitées par le modèle de l'étape antérieure, ce que la provenance signale
  (`MODEL_ON_IMPUTED_TARGET`, §6).
- `'covariates_only'` conserve le bénéfice de la cascade **du côté des covariables** — une
  covariable imputée à l'étape `Q` est reportée sur la grille `M` (rang 3 du §4.4), information
  strictement plus riche que l'interpolation de ses seules ancres annuelles — **sans** admettre
  la moindre valeur de modèle dans la cible.
- `False` est le comportement le plus simple et le plus propre : un modèle par variable, entraîné
  sur ses seules observations, prédisant directement à la fréquence cible.

Ce paramètre reprend et fusionne deux mécanismes de `hfi`/[ARCH] : la cascade de fréquences
(PHASE 5) et `train_on_own_imputations` ([ARCH] §3.10, jamais implémenté — prompt 15 non exécuté).

### 5.2 — Construction de la progression de fréquences

Algorithme, identique au fit et au transform (rejoué depuis le plan au transform) :

1. Soit `F` l'ensemble des fréquences détectées des colonnes **imputables** du périmètre, plus la
   fréquence cible `f_target`.
2. Si `impute_intermediate_frequencies is False` : `progression = [f_target]`.
3. Sinon : `progression = sorted(F \ {la plus basse}, de la plus basse à la plus haute)`, en ne
   retenant que les fréquences **strictement plus hautes** que la plus basse fréquence source et
   **inférieures ou égales** à `f_target`, et en garantissant que `f_target` en est le dernier
   élément.
4. À chaque étape `f`, les **variables imputables à `f`** sont les colonnes dont `f_var` est
   strictement plus basse que `f` et qui ne sont pas encore imputées **à `f`**.

Sur `TS` : `F = {Q, Y, M}` ; sous `False` → `['M']` ; sous `'covariates_only'`/`True` →
`['Q', 'M']` (la fréquence `Y`, la plus basse, n'est pas une étape : rien n'est à imputer à `Y`).
Étape `Q` : variables `{a1, a2}`. Étape `M` : variables `{q1, a1, a2}`.

Sur un panel, la progression est calculée **par groupe d'entités partageant la même fréquence
cible** ; `target_frequency` en dict autorise des cibles différentes par entité (validation B16 :
dict incomplet → `ValueError` nommant les entités manquantes).

### 5.3 — Composition de `y_train` : le filtre d'origine

`y_train` d'une variable `v` à l'étape `f` est composé des cellules de la **colonne `v` dans le
frame d'étape**, restreintes par la fenêtre `'training'` (§7), filtrées par leur origine :

```python
ELIGIBLE_ORIGINS = {
    False:             {'observed'},
    'covariates_only': {'observed', 'interpolated'},
    True:              {'observed', 'interpolated', 'model'},
}
```

Trois points d'implémentation impératifs :

1. **Le filtre porte sur `origin_store`, pas sur la matrice de provenance.** `DISAGGREGATED` est
   ambigu par construction (§6.5) : il marque aussi bien une cellule issue d'une interpolation
   recalée qu'une prédiction de modèle recalée. Utiliser la provenance publique comme filtre
   ferait de `'covariates_only'` un synonyme de `True` — c'est le piège principal de ce
   paragraphe.
2. **Chaque ligne de `y_train` porte la fréquence à laquelle elle a été produite**, lue dans
   `imputed_freq_store` ; le diviseur d'échelle est **par ligne** (§5.4).
3. Sous `False`, `y_train` d'une variable annuelle sur `TS` contient exactement 3 lignes : la
   garde `min_cv_train_size` et les gardes de taille de l'estimateur doivent être documentées
   comme le prix de la modalité, et le repli interpolation reste le filet.

### 5.4 — Le piège d'échelle des lignes imputées

> Les lignes de `y_train` issues d'une imputation antérieure sont à l'échelle de **leur** étape
> (des trimestres pour une variable annuelle imputée en `Q`) ; leur diviseur de mise à l'échelle
> est propre à la ligne — `get_conversion_factor(pred_freq, f_ligne)` — et non le scalaire de
> l'étape.

Le registre `imputed_freq_store` (fréquence de production de chaque imputation) et la forme
`scale_factor: Union[float, pd.Series]` de [ARCH] §3.10 sont repris tels quels, y compris le
correctif du court-circuit B12 (ne pas court-circuiter la mise à l'échelle quand le facteur est
une `Series` dont toutes les valeurs valent 1.0 par hasard).

Exemple chiffré sur `TS`, `impute_intermediate_frequencies=True`, `scale_features='constant'`,
étape `M`, variable `a1` :

| Ligne de `y_train` | Origine | Fréquence de production | Valeur brute | Diviseur | Valeur mise à l'échelle |
|---|---|---|---|---|---|
| 2021-12-31 (ancre) | `'observed'` | `Y` | 120 | `factor(M, Y) = 12` | 10.0 |
| 2021-03-31 (imputée à `Q`) | `'model'` | `Q` | 28 | `factor(M, Q) = 3` | 9.33 |
| 2021-06-30 (imputée à `Q`) | `'model'` | `Q` | 30 | `factor(M, Q) = 3` | 10.0 |

Sans le diviseur par ligne, la valeur annuelle 120 et la valeur trimestrielle 28 seraient mêlées
telles quelles dans la même cible : le modèle apprendrait un mélange de deux échelles.

### 5.5 — Exemple complet du plan sur `TS`

`covariate_strategy='model'`, `fit_predict_order='frequency'`, `impute_intermediate_frequencies`
variable :

**Sous `False`** — 1 étape, 3 modèles :

| # | Étape | Variable | `y_train` | `feature_cols` | Voies de matérialisation |
|---|---|---|---|---|---|
| 1 | `M` | `a1` | 3 ancres Y | `m1, q1, a2` | `m1`: identity ; `q1`, `a2`: fallback (interpolate) |
| 2 | `M` | `a2` | 3 ancres Y | `m1, q1, a1` | `m1`: identity ; `q1`: fallback ; `a1`: stage_model |
| 3 | `M` | `q1` | 12 ancres Q | `m1, a1, a2` | `m1`: identity ; `a1`, `a2`: stage_model |

(Ordre `'frequency'` : la fréquence la plus basse d'abord — `a1`, `a2` (annuelles) avant `q1`
(trimestrielle) — puis tie-break alphabétique entre `a1` et `a2`. Le §8.4 déroule le même
exemple sous `fit_predict_order='cv'`.)

**Sous `'covariates_only'` ou `True`** — 2 étapes, 5 modèles :

| # | Étape | Variable | `y_train` sous `'covariates_only'` | `y_train` sous `True` |
|---|---|---|---|---|
| 1 | `Q` | `a1` | 3 ancres Y | 3 ancres Y |
| 2 | `Q` | `a2` | 3 ancres Y | 3 ancres Y |
| 3 | `M` | `q1` | 12 ancres Q | 12 ancres Q |
| 4 | `M` | `a1` | 3 ancres Y | 3 ancres Y **+ 12 imputations Q** |
| 5 | `M` | `a2` | 3 ancres Y | 3 ancres Y **+ 12 imputations Q** |

Aux étapes 4 et 5, `a1` et `a2` se voient mutuellement comme covariables **reportées de l'étape
`Q`** (rang 3 du §4.4) sous les deux modalités — c'est l'apport propre de `'covariates_only'`.

### 5.6 — Combinaisons inertes, à documenter sans avertir (D9)

- `'covariates_only'` **sans** `covariate_strategy='model'` : les covariables sont matérialisées
  par interpolation de leurs observations (rangs 1 et 4 seulement), le rang 3 n'est jamais
  atteint — les étapes intermédiaires ne changent **aucune** valeur finale. Elles restent
  visibles dans la sortie multi-fréquences si `keep_lower_frequencies=True`, et coûtent du temps
  de calcul. À dire explicitement dans la docstring du paramètre.
- `True` a un effet sous **toutes** les stratégies (il modifie `y_train`).
- `covariate_fallback` est inerte hors `covariate_strategy='model'`.
- `fit_predict_order` est inerte hors `covariate_strategy='model'` (§8.1).
- `interpolation_method`, `interpolation_anchor` sont inertes sous `'tolerate_nan'` **et**
  `covariate_fallback='tolerate_nan'`, sauf s'ils servent au repli d'échec (ils servent toujours
  à cela : ils ne sont donc jamais totalement inertes).

Aucun de ces cas n'émet d'avertissement (D9) : la docstring les énonce, un `UserWarning` par
combinaison rendrait la classe pénible en exploration d'hyperparamètres.

### 5.7 — Ce que le nouvel espace de paramètres n'exprime plus

**Le mode « un seul fit, réutilisé aux étapes suivantes avec le facteur d'échelle de l'étape »**
(`cascade_refitting=False` de `hfi`) est **abandonné** (D2). C'était une économie de calcul au
prix d'une asymétrie fit/predict jamais maîtrisée (B7). Si le besoin réapparaît, il se
réintroduit comme **optimisation interne** — mémoïsation d'un modèle dont le jeu d'entraînement
et les voies de matérialisation n'ont pas changé entre deux étapes — jamais comme sémantique
publique.

En revanche, la seconde capacité perdue de la version 1 du document — « étapes intermédiaires
pour enrichir les covariables, mais cible entraînée sur les seules valeurs fiables » — est
**restaurée** : c'est exactement `impute_intermediate_frequencies='covariates_only'`.

---

## 6 — Provenance

### 6.1 — L'énumération

```python
class ProvenanceType(str, Enum):
    """Enumeration of value provenance types."""
    # --- Cellules non produites par un modèle ---
    ORIGINAL      = 'original'        # présente dans le jeu d'entrée
    AGGREGATED    = 'aggregated'      # agrégation exacte de vraies valeurs plus fines
    DISAGGREGATED = 'disaggregated'   # sous-période d'un total observé, réparti sur sa période
    INTERPOLATED  = 'interpolated'    # NOUVEAU : produite par interpolation d'observations

    # --- Cellules produites par un modèle, par degré de souillure ---
    MODEL_ON_TRUE           = 'model_on_true'
    MODEL_ON_INTERPOLATED   = 'model_on_interpolated'    # NOUVEAU
    MODEL_ON_IMPUTED        = 'model_on_imputed'         # NOUVEAU
    MODEL_ON_IMPUTED_TARGET = 'model_on_imputed_target'  # NOUVEAU
    MODEL_ON_IMPUTED_BOTH   = 'model_on_imputed_both'    # NOUVEAU

    # MODEL_ON_MIXED : SUPPRIMÉ (voir §6.7)
```

Les cinq libellés `MODEL_*` répondent à une question unique et lisible : **quel est le plus
mauvais ingrédient qu'a vu le modèle, et de quel côté ?**

- `MODEL_ON_TRUE` — le modèle n'a vu que des vraies valeurs, y compris **agrégées** (une
  agrégation additive exacte d'observations n'est pas une approximation).
- `MODEL_ON_INTERPOLATED` — au moins une valeur **interpolée** parmi les covariables **ou** dans
  `y_train`, et aucune valeur de modèle.
- `MODEL_ON_IMPUTED` — au moins une **covariable imputée par modèle** (en plus, éventuellement,
  de vraies valeurs, d'agrégées et d'interpolées) ; `y_train` reste indemne.
- `MODEL_ON_IMPUTED_TARGET` — `y_train` contient au moins une valeur imputée par modèle
  (`impute_intermediate_frequencies=True`), les covariables non.
- `MODEL_ON_IMPUTED_BOTH` — les deux.

### 6.2 — Origine des cellules et souillures

Deux notions internes, non exposées dans l'API publique mais **portées par le plan** :

```python
CellOrigin = Literal['observed', 'interpolated', 'model']   # ordre croissant de souillure
Taint      = Literal['none', 'interpolated', 'imputed']
```

- **`origin_store`** : registre `{colonne: pd.Series[CellOrigin]}` tenu par le
  `CovariateMaterializer`, aligné sur l'index du frame d'étape. Une cellule d'entrée non NaN vaut
  `'observed'` ; une agrégation exacte vaut `'observed'` ; une interpolation ou un repli
  d'interpolation valent `'interpolated'` ; une prédiction de modèle vaut `'model'`, y compris
  après recalage aux totaux (§11) et y compris après report d'étape (§4.4, rang 3).
- **`covariate_taint`** d'une étape = `max` des origines des cellules **effectivement lues** dans
  `X_train ∪ X_pred`, restreint aux **`feature_cols` effectives** du modèle (leçon C17 de [ARCH]
  §3.8 : jamais sur l'état global du store) — avec la correspondance
  `'observed'→'none'`, `'interpolated'→'interpolated'`, `'model'→'imputed'`.
- **`target_taint`** d'une étape = `max` des origines des lignes retenues dans `y_train`, même
  correspondance.

Le champ `ImputationStep.trained_on_imputed: bool` est **remplacé** par :

```python
covariate_taint: Taint
target_taint: Taint
```

### 6.3 — Table de correspondance (3 × 3 → 5)

| `covariate_taint` ↓ / `target_taint` → | `'none'` | `'interpolated'` | `'imputed'` |
|---|---|---|---|
| **`'none'`** | `MODEL_ON_TRUE` | `MODEL_ON_INTERPOLATED` | `MODEL_ON_IMPUTED_TARGET` |
| **`'interpolated'`** | `MODEL_ON_INTERPOLATED` | `MODEL_ON_INTERPOLATED` | `MODEL_ON_IMPUTED_TARGET` |
| **`'imputed'`** | `MODEL_ON_IMPUTED` | `MODEL_ON_IMPUTED` | `MODEL_ON_IMPUTED_BOTH` |

Fonction de référence, à implémenter telle quelle dans `provenance.py` :

```python
def resolve_model_provenance(covariate_taint: Taint, target_taint: Taint) -> ProvenanceType:
    """Map the two training taints of a step onto its emitted MODEL_* provenance."""
    if target_taint == 'imputed':
        return (ProvenanceType.MODEL_ON_IMPUTED_BOTH if covariate_taint == 'imputed'
                else ProvenanceType.MODEL_ON_IMPUTED_TARGET)
    if covariate_taint == 'imputed':
        return ProvenanceType.MODEL_ON_IMPUTED
    if 'interpolated' in (covariate_taint, target_taint):
        return ProvenanceType.MODEL_ON_INTERPOLATED
    return ProvenanceType.MODEL_ON_TRUE
```

**Propriétés à tester** (I6, §16) :

- la provenance est une **propriété de l'étape**, propagée identiquement à toutes les cellules
  que le modèle de cette étape produit ;
- `MODEL_ON_IMPUTED` et `MODEL_ON_IMPUTED_BOTH` ne sont émis que sous
  `covariate_strategy='model'` ;
- `MODEL_ON_IMPUTED_TARGET` et `MODEL_ON_IMPUTED_BOTH` ne sont émis que sous
  `impute_intermediate_frequencies=True` ;
- sous `covariate_strategy='interpolate'`, dès qu'une covariable de fréquence plus basse que la
  grille entre dans `feature_cols`, la provenance est `MODEL_ON_INTERPOLATED` (et non
  `MODEL_ON_TRUE`) : c'est le point de rupture avec la version 1 du document.

### 6.4 — Provenance des cellules non produites par un modèle

| Situation | Provenance | Origine (`origin_store`) |
|---|---|---|
| valeur présente dans le jeu d'entrée | `ORIGINAL` | `'observed'` |
| agrégation exacte d'une colonne plus fine sur une période complète | `AGGREGATED` | `'observed'` |
| cellule d'une période **recalée** pour sommer au total observé (§11) | `DISAGGREGATED` | inchangée : `'model'` si la prédiction venait d'un modèle, `'interpolated'` si elle venait d'une interpolation |
| **date-ancre** ré-exprimée à la fréquence d'étape, que le recalage ait eu lieu ou non | `DISAGGREGATED` | idem |
| cellule produite par interpolation (stratégie `'interpolate'`, `covariate_fallback`, ou **repli d'échec** d'un modèle) | `INTERPOLATED` | `'interpolated'` |

Deux règles héritées et conservées :

- **B2** : le marquage de provenance des ancres est **indépendant de la réussite du recalage**.
- **D6, tranché** : les cellules du **repli** (variable dont le modèle a échoué) portent
  `INTERPOLATED`, et non un `MODEL_*`. C'est plus exact et cela rend le repli visible dans les
  statistiques de provenance. Une étape en repli est marquée `is_fallback=True` dans le plan.

`DISAGGREGATED` est donc **ambigu quant au degré de confiance** — c'est assumé, il décrit une
position (« sous-période d'un total observé ») autant qu'une origine. Le filtre de `y_train`
(§5.3) et le calcul des souillures (§6.2) lisent `origin_store`, **jamais** la matrice de
provenance. La docstring de `DISAGGREGATED` doit le dire.

### 6.5 — Exemple cellule par cellule

Jeu `TS`, `covariate_strategy='interpolate'`, `impute_intermediate_frequencies=False`,
`aggregation_constraint='sum'`, imputation de `a1` (2021 = 120) à l'étape `M`.
Le modèle de `a1` a `feature_cols = [m1, q1, a2]` ; `q1` et `a2` sont plus basses que `M`, donc
interpolées → `covariate_taint = 'interpolated'` ; `y_train` = 3 ancres → `target_taint = 'none'`
→ provenance émise **`MODEL_ON_INTERPOLATED`**.

| Date | Prédiction brute | Après recalage (somme 2021 = 120) | Provenance | Origine |
|---|---|---|---|---|
| 2021-01-31 | 9.0 | 9.6 | `DISAGGREGATED` | `'model'` |
| 2021-02-28 | 9.5 | 10.13 | `DISAGGREGATED` | `'model'` |
| … | … | … | `DISAGGREGATED` | `'model'` |
| 2021-12-31 (**ancre**) | 10.5 | 11.2 | `DISAGGREGATED` | `'model'` |
| **somme 2021** | 112.5 | **120.0** | — | — |

Sous `aggregation_constraint=None`, les mêmes cellules portent **`MODEL_ON_INTERPOLATED`** et
gardent leur valeur brute, **sauf** la ligne d'ancre 2021-12-31 qui porte `DISAGGREGATED` (elle
est une ancre ré-exprimée à la fréquence d'étape, cf. B2) tout en valant 10.5 : la somme 2021 ne
fait alors plus 120, et la valeur observée n'est récupérable que par `inverse_transform` ou par
le masque `ORIGINAL` du niveau source (§11.2).

### 6.6 — `mark_model_imputed` : nouvelle signature

```python
def mark_model_imputed(
    self,
    column: str,
    index: Union[pd.Timestamp, pd.DatetimeIndex, slice],
    covariate_taint: Taint = 'none',
    target_taint: Taint = 'none',
) -> None:
    """Mark specific values as imputed by a model, at the given taint levels."""
```

Le paramètre `trained_on_imputed: bool` disparaît. Les appels de `hfi` sont mis à jour
mécaniquement : `trained_on_imputed=True` devient `covariate_taint='imputed'` — la sémantique
d'origine de `hfi` (`trained_on_imputed = train_on_partial_coverage and bool(imputed_store)`)
porte bien sur les **covariables**.

### 6.7 — Suppression de `MODEL_ON_MIXED` : impact sur `HighFrequencyImputer`

`MODEL_ON_MIXED` est supprimé de l'énumération (décision explicite de l'auteur, acceptant la
rupture). L'énumération étant partagée, `hfi` doit être mis à jour dans le **même lot** (L2,
§17), de façon purement mécanique :

| Site | Changement |
|---|---|
| `provenance.py:ProvenanceType` | suppression de `MODEL_ON_MIXED`, ajout des quatre nouvelles constantes, mise à jour de la docstring de classe |
| `provenance.py:mark_model_imputed` | nouvelle signature (§6.6) |
| `provenance.py` docstring de module et exemples | `MODEL_ON_MIXED` → `MODEL_ON_IMPUTED` |
| `hfi:_mark_predictions_provenance`, `hfi:_apply_step_predictions` et leurs appelants (les sites passant `trained_on_imputed`) | passer `covariate_taint='imputed' if step.trained_on_imputed else 'none'` ; `ImputationStep.trained_on_imputed` **reste** dans `hfi` tant que `hfi` existe |
| `tests/frequency/test_high_frequency_imputer.py` | les quatre occurrences de `MODEL_ON_MIXED` deviennent `MODEL_ON_IMPUTED` ; le test `test_transform_fallback_marks_trained_on_imputed_false` est renommé et vérifie qu'aucune cellule de repli ne porte un `MODEL_*` (elles portent `INTERPOLATED` dans `hfi2`, elles gardent le comportement `hfi` dans `hfi`) |
| notebooks 4 et suivants | remplacement du libellé dans les affichages de provenance |

Aucun alias de compatibilité n'est conservé : un `MODEL_ON_MIXED` résiduel doit produire un
`AttributeError` franc, pas un comportement silencieux.

---

## 7 — Fenêtres d'entraînement et d'imputation

Reprise intégrale de [ARCH] §3.3 et §3.7 (prévus pour `imputation_scope`/`coverage_threshold`,
prompts 10 et 13 **non exécutés** — c'est `hfi2` qui les réalise) :

```python
imputation_scope: ImputationScope = 'strict'          # fenêtre de PRÉDICTION
coverage_threshold: float = 0.5                       # seuil de ses extensions
training_scope: Optional[TrainingScope] = None        # None -> suit imputation_scope
training_coverage_threshold: Optional[float] = None   # None -> suit coverage_threshold

ImputationScope = Literal['strict', 'extended_backward', 'extended_forward',
                          'extended_both', 'unrestricted']
TrainingScope   = ImputationScope   # mêmes modalités
```

### 7.1 — Les trois masques

`ImputationWindowCalculator` expose **trois masques** via
`get_imputation_window_mask(data, kind: Literal['strict', 'imputation', 'training'])` :

| `kind` | Définition | Appelants |
|---|---|---|
| `'strict'` | lignes où **toutes** les variables du périmètre ont au moins une observation dans leur période | ordonnancement CV (§8.3), diagnostics |
| `'imputation'` | `'strict'` étendu selon `imputation_scope` / `coverage_threshold` | grille de **prédiction** |
| `'training'` | `'strict'` étendu selon `training_scope` / `training_coverage_threshold` (qui retombent sur les précédents quand ils valent `None`) | grille d'**entraînement** |

Chaque appelant **nomme explicitement** son masque : aucun appel sans `kind`. Les correctifs B23
(la branche « aucune fenêtre stricte » ignorait les extensions) et B24 (docstring) sont inclus.

### 7.2 — Règles d'usage

- **Élargir `training_scope` ajoute des lignes, jamais des colonnes.** La sélection des
  `feature_cols` reste gouvernée par la disponibilité à la prédiction (§4), indépendamment de la
  fenêtre d'entraînement. Les deux ajustements de [ARCH] §3.6 suivent : une colonne n'est gardée
  que si elle est non-vide sur **les deux** fenêtres ; les lignes d'entraînement sans aucune
  covariable observée sont écartées.
- **Type de retour unifié** (idée du prompt 20, retenue et généralisée) : pour un panel, les
  masques de fenêtre sont des `pd.Series` booléennes à MultiIndex `(entity..., date)` — plus
  jamais des `Dict[entity, Series]`. Cela vaut pour les attributs ajustés
  (`imputation_window_mask_`, `training_window_mask_`), pour `get_mask_at_frequency`, et pour
  toute structure interne par entité de `hfi2` où c'est praticable. La conversion de fréquence
  des masques se délègue à `FrequencyConverter.convert_frequency` (vérifier au passage sa gestion
  des fréquences cibles par entité ; sinon garder la boucle interne mais **unifier le type de
  retour**).
- **`transform` hors fenêtre de fit (B1)** : la fenêtre est une contrainte de **disponibilité des
  données**, pas un paramètre appris — au `transform`, elle est **recalculée sur les données
  transformées** avec les hyperparamètres du fit (option A de [ARCH] §3.14), avec deux
  garde-fous : ne jamais vider une colonne sans la réécrire, et **avertir une seule fois** quand
  des lignes du périmètre sont hors fenêtre (message agrégé nommant le nombre de lignes et les
  entités concernées). `fit_transform(X) ≡ fit(X).transform(X)` reste un invariant strict (le
  recalcul sur `X` redonne la fenêtre du fit).

### 7.3 — Exemple

Jeu `TS` tronqué : `a1` observée jusqu'en 2023 mais `q1` seulement jusqu'au 2023-06-30 (retard de
publication). Fenêtre `'strict'` = 2021-01 → 2023-06. Sous `imputation_scope='extended_forward'`
et `coverage_threshold=0.5`, la fenêtre de prédiction s'étend jusqu'au 2023-12 tant qu'au moins
50 % des variables du périmètre sont observées ; sous `training_scope=None`, la fenêtre
d'entraînement suit la même extension. Poser `training_scope='strict'` entraîne alors sur
2021-01 → 2023-06 et prédit jusqu'au 2023-12 — la configuration recommandée pour un usage
pseudo-temps réel.

---

## 8 — Ordre d'imputation

### 8.1 — `fit_predict_order`, champ d'application restreint

```python
fit_predict_order: Literal['frequency', 'cv'] = 'frequency'
```

Mêmes modalités que `train_on_partial_fit_order` de `hfi`, logiques reprises de
`hfi:_determine_imputation_order` (`'frequency'` : fréquence la plus basse d'abord, puis nombre
d'entités décroissant) et `hfi:_determine_variable_order_cv` (`'cv'` : variables les mieux
prédites d'abord, réécrite autour de `cross_val_score` — [ARCH] §3.4, corrections B9/B10
incluses).

- **Champ d'application** : l'ordre n'est calculé et appliqué que sous
  `covariate_strategy='model'` — le seul mode où il influe sur le résultat. Sous
  `'tolerate_nan'` et `'interpolate'`, aucune logique de tri n'est exécutée ; l'ordre de
  traitement est celui des colonnes d'entrée, **sans effet sur les valeurs produites**
  (propriété garantie par les tests I3 et I10, §16). Passer `fit_predict_order='cv'` avec une
  autre stratégie n'est **pas** une erreur : le paramètre est ignoré, la docstring le dit, aucun
  avertissement n'est émis (D9).
- **Justification de l'indifférence** : `'tolerate_nan'` ne matérialise rien ; `'interpolate'`
  matérialise depuis les seules observations. Dans les deux cas, le jeu d'entraînement et le jeu
  de prédiction d'une variable ne dépendent d'aucune imputation antérieure d'une autre variable.
  La seule dépendance à l'ordre qui subsiste est l'axe 2 (les imputations intermédiaires d'une
  variable alimentent **sa propre** étape suivante), qui suit l'ordre des **fréquences**, pas
  l'ordre des variables.
- **Déterminisme intra-étape** : sous `'model'`, les ex æquo du tri (`'frequency'` : même
  fréquence et même nombre d'entités ; `'cv'` : scores égaux, `NaN` compris) sont départagés par
  **ordre alphabétique du nom de variable** — jamais par l'ordre des colonnes d'entrée. C'est la
  réponse à la deuxième cause de B28 : l'asymétrie intra-étape demeure (elle est intrinsèque au
  mode) mais elle devient déterministe, documentée et indépendante de la présentation des
  données.

### 8.2 — Correctifs CV repris de [ARCH] §3.4

Restriction aux lignes exploitables **avant** scoring ; `check_scoring` pour résoudre
`cv_scoring` ; `cross_val_score(..., error_score=np.nan)` ; sentinelles `-np.inf` pour les
variables non scorables ; **tri décroissant partout** (convention *greater is better* de
sklearn) ; journal (`self._log`) des variables dont tous les plis ont échoué ; masque `'strict'`
pour construire le jeu de scoring ; note de docstring sur le traitement des zéros par le MAPE de
sklearn (division par zéro → score dégradé, pas d'erreur).

### 8.3 — Le paramètre `cv` (convention sklearn)

```python
cv: Union[int, BaseCrossValidator, Iterable, None] = None
# None -> KFold(n_splits=5, shuffle=True, random_state=42)
# int  -> KFold(n_splits=cv, shuffle=True, random_state=42)
# splitter ou itérable de splits -> utilisé tel quel
cv_scoring: Union[str, Callable] = 'neg_mean_absolute_percentage_error'
min_cv_train_size: int = 10
```

`cv_n_splits` est **supprimé** (D4). Justification :

- c'est exactement le contrat de `GridSearchCV`/`cross_val_score` : tout utilisateur sklearn sait
  passer `cv=5` ou `cv=TimeSeriesSplit(n_splits=3)` ; la validation est fournie par
  `sklearn.model_selection.check_cv` ;
- un splitter utilisateur ouvre les cas que des scalaires n'exprimeront jamais (CV temporelle,
  splitter *group-aware* sur panel — les classes de `tsforecast/crossvals` sont directement
  utilisables ici) sans multiplier les paramètres ;
- `cv_scoring` reste séparé (comme `scoring` chez sklearn) ; `min_cv_train_size` reste : c'est un
  seuil d'**éligibilité au scoring**, pas un réglage du splitter.

**Avertissement croisé** conservé, reformulé : après `check_cv`, si
`min_cv_train_size < n_splits effectifs`, un `UserWarning` unique est émis au `fit` (message
nommant les deux valeurs). Le défaut `shuffle=True, random_state=42` est repris du code actuel
avec son commentaire justificatif ; un utilisateur qui n'en veut pas passe son propre splitter.

**Validation à `__init__`** (B3, sans transformation) : `cv` doit être `None`, un `int ≥ 2`, un
objet exposant `split` et `get_n_splits`, ou un itérable. `check_cv` n'est appelé **qu'au `fit`**,
et son résultat stocké dans `cv_` — le paramètre reste inchangé pour `get_params`/`clone`.

### 8.4 — Exemple d'ordre sur `TS`

`covariate_strategy='model'`, `impute_intermediate_frequencies=False`, une seule étape `M` :

| `fit_predict_order` | Ordre calculé | Raison |
|---|---|---|
| `'frequency'` | `a1`, `a2`, `q1` | `Y` avant `Q` ; `a1` avant `a2` par tie-break alphabétique |
| `'cv'` (scores `q1`=−0.08, `a2`=−0.15, `a1`=−0.15) | `q1`, `a1`, `a2` | tri décroissant des scores ; `a1` avant `a2` par tie-break alphabétique |

Sous `'frequency'`, `a1` est imputée sans covariable de modèle (`q1` et `a2` viennent après →
rang 4, fallback) ; `a2` voit `a1` (rang 2) ; `q1` voit `a1` et `a2` (rang 2). Les provenances
résultantes sont donc `MODEL_ON_INTERPOLATED` pour `a1` (fallback interpolé) et
`MODEL_ON_IMPUTED` pour `a2` et `q1`.

---

## 9 — Mise à l'échelle des données

### 9.1 — `scale_features` à trois modalités, éventuellement par feature

```python
ScaleMode = Literal['constant', 'calendar']
scale_features: Union[Literal[False], ScaleMode,
                      Dict[str, Union[Literal[False], ScaleMode]]] = 'constant'
```

| Modalité | Diviseur | Cas d'usage |
|---|---|---|
| `False` | aucun sur les features ; **`y` reste toujours mis à l'échelle** (comportement du `False` actuel) | features déjà comparables |
| `'constant'` (défaut) | diviseur **constant** par couple de fréquences, `FrequencyConverter.get_conversion_factor` (M→Y = 12, D→M = 30.0…) | variables **corrigées des variations saisonnières** : le facteur moyen lisse ce que la CVS a déjà lissé |
| `'calendar'` | diviseur par **décompte calendaire réel** de la période, `FrequencyConverter.count_subperiods_per_period` (février → 28 ou 29, T1 → 90 ou 91) | variables **brutes** (non CVS), où le nombre de jours porte du signal |

`'calendar'` produit une `pd.Series` de diviseurs par ligne ; la plomberie
`scale_factor: Union[float, pd.Series]` du §5.4 sert les deux besoins — **un seul chemin de
code**, avec le correctif du court-circuit B12.

**Forme dict** : `{feature: modalité}` avec clé `'__default__'` optionnelle (même convention que
`estimator`). Clés inconnues au `fit` → `ValueError` listant les colonnes fautives. Features non
couvertes et pas de `'__default__'` → défaut `'constant'`. Les clés sont des **noms de colonne**,
jamais des couples `(entité, colonne)` (D10).

### 9.2 — Ce que la modalité gouverne exactement

La modalité s'applique à **tous les diviseurs** relatifs à la feature :

1. le diviseur des covariables à l'entraînement et à la prédiction
   (`_covariate_scaling_divisors`, **règle B25** : `1.0` pour une colonne jamais ré-agrégée,
   `get_conversion_factor(f_stage, f_var)` sinon, avec `f_stage = pred_freq` si `f_var` est plus
   fine que l'étape et `f_var` sinon) ;
2. le diviseur de `y` (scalaire d'étape, ou `pd.Series` par ligne dès que `y_train` mêle
   plusieurs fréquences de production, §5.4) ;
3. le report d'échelle des prédictions (`fit_scale_factor` : le facteur **cuit dans le modèle**,
   qui ne bouge plus une fois l'étape ajustée).

La modalité de `y` est celle de la **colonne imputée**. Un mélange légitime « feature en
`'calendar'`, `y` en `'constant'` » doit être couvert par un test dédié (I5).

### 9.3 — Prérequis et audit

- **B26 (prompt 21, non exécuté)** : le contrôle `full_periods_only` de
  `utils/frequency/converter.py:_require_full_subperiod_coverage` utilise le décompte
  **constant** et jette février — toute covariable journalière disparaît silencieusement du jeu
  d'entraînement. `hfi2` s'appuyant sur les mêmes agrégations, **ce correctif est un prérequis
  dur** (§15, P1 ; lot L1 du §17).
- L'audit demandé — « moyenne des features sur train et sur test, comparées » — devient
  l'invariant de test I5 (§16) et une section du notebook pas à pas (§15.2).

---

## 10 — Interpolation et position d'ancrage

### 10.1 — Méthode d'interpolation, globale ou par feature

```python
interpolation_method: Union[str, Dict[str, str]] = 'linear'
```

Valeurs admises : celles de `FrequencyConverter.interpolate_to_higher_frequency` (`'linear'`,
`'time'`, `'nearest'`, `'zero'`, `'slinear'`, `'quadratic'`, `'cubic'`, …). Forme dict par feature
avec `'__default__'`, mêmes règles de validation qu'au §9.1.

Ce paramètre sert **trois** usages avec la même valeur (cohérence voulue) :

1. la stratégie `covariate_strategy='interpolate'` (§4.3) ;
2. le `covariate_fallback='interpolate'` (§4.4, rang 4) ;
3. le **repli d'échec** d'une imputation par modèle (l'actuel `INTERPOLATE_FALLBACK`) ;

et, sous `impute_intermediate_frequencies != False`, le **report d'étape** du rang 3 (§4.4).

Le `limit_direction` reste résolu par `FrequencyConverter._resolve_limit_direction` avec ses
défauts actuels (dépendant de la position de l'index dans la période).

### 10.2 — Position d'ancrage de la valeur dans sa période

```python
interpolation_anchor: Union[None, float, Dict[str, Optional[float]]] = None
```

- `None` (défaut) : comportement actuel — la valeur s'applique à la date à laquelle elle est
  référencée (début ou fin de période, selon la position de l'index).
- `float ∈ [0, 1]` : la valeur observée d'une période est considérée **atteinte à la fraction
  correspondante de la période qu'elle couvre** — `0.0` = début, `0.5` = milieu, `1.0` = fin.
  L'interpolation entre ancres se fait entre ces points décalés, puis le résultat est réindexé
  sur la grille cible.
- Forme dict par feature avec `'__default__'`.

**Implémentation** — nouvelle option d'`interpolate_to_higher_frequency` :

```python
def interpolate_to_higher_frequency(self, data, target_freq, method='linear',
                                    limit_direction=None,
                                    anchor_fraction: Optional[float] = None): ...
```

1. décaler l'index ré-ancré de `_reanchor_index_to_target` vers
   `début_période + anchor_fraction × durée_période`, **avant** interpolation ;
2. interpoler sur l'**union** (index décalé ∪ grille cible) ;
3. ne retenir que la grille cible.

Les timestamps décalés ne tombent en général **pas** sur la grille cible : l'union est
indispensable, et les valeurs d'ancre elles-mêmes ne survivent pas telles quelles dans la sortie
— c'est le but recherché (à `0.5`, la valeur de mars ne prétend plus valoir au 31 mars).

**Exemple** — `a1` annuelle (120 en 2021, 132 en 2022), interpolation vers `Q`,
`method='linear'` :

| `anchor_fraction` | Position des ancres | Valeurs Q 2022 avant recalage | Après recalage à 132 |
|---|---|---|---|
| `None` | 2021-12-31, 2022-12-31 | interpolation entre 120 et 132 aux 4 fins de trimestre : 123, 126, 129, 132 | ×132/510 → 31.8, 32.6, 33.4, 34.2 |
| `0.5` | 2021-07-02, 2022-07-02 | ancres au milieu d'année ; T1 2022 est extrapolé/interpolé entre les deux points décalés | somme des 4 trimestres ramenée à 132 |

**Interactions à spécifier dans la docstring** :

- avec le recalage aux totaux (§11) : l'ancrage fractionnaire change la **forme** de
  l'interpolation, le recalage ré-impose ensuite le **total** — les deux se composent sans
  conflit, dans cet ordre ;
- aux bords de série, au-delà de la dernière ancre décalée, `limit_direction` décide ;
- ce paramètre s'applique à l'**interpolation seulement** (stratégie, fallback, repli, report
  d'étape) — jamais à l'agrégation, ni à la désagrégation par totaux, qui restent ancrées sur les
  périodes exactes.

---

## 11 — Contrainte d'agrégation

### 11.1 — `aggregation_constraint`

```python
aggregation_constraint: Literal['sum', None] = 'sum'
```

Remplace `enforce_period_totals: bool` (D8). `'sum'` ≡ `enforce_period_totals=True`, `None` ≡
`False`. Le type littéral est retenu **maintenant** parce qu'il ouvre, sans rupture d'API :

- d'autres contraintes (`'mean'` pour des taux, `'last'` pour des stocks) — actuellement hors
  contrat, `additive_transformer` étant chargé de rendre les colonnes additives ;
- une **forme dictionnaire** `Dict[str, Optional[str]]` par colonne, avec `'__default__'`, sur le
  modèle de `estimator` et `scale_features`.

**Validation à `__init__` (état actuel)** : seules les valeurs `'sum'` et `None` sont acceptées.
Toute autre valeur — y compris un dict — lève un `ValueError` dont le message énonce que les
formes `'mean'`, `'last'` et dictionnaire sont **réservées pour une extension ultérieure**. La
docstring décrit l'extension prévue.

Sémantique de `'sum'`, reprise de `hfi:_rescale_to_period_totals` : les sous-périodes prédites
d'une période sont multipliées par `total observé / total prédit`, de sorte que la colonne porte
une véritable **désagrégation** de l'observation plutôt qu'une prédiction libre. Gardes
conservées :

| Cas | Comportement |
|---|---|
| période **partiellement** prédite (au moins une sous-période NaN) | non recalée, prédictions brutes conservées |
| période sans aucune observation (fin de série retardée) | non recalée |
| total prédit nul, total observé non nul | non recalée (ratio indéfini) |
| total prédit de **signe opposé** au total observé | **recalée** — la contrainte prime — mais toutes les sous-périodes changent de signe : un `UserWarning` agrégé est émis |

Le masque des cellules effectivement recalées pilote le marquage `DISAGGREGATED` ; les cellules
laissées de côté gardent leur provenance `MODEL_*` ou `INTERPOLATED` (§6.4).

### 11.2 — Désagrégation des ancres : comportement **non paramétrable**

> **Une variable imputée à l'étape `f` est prédite sur la totalité de chaque période couverte,
> ancres comprises.** La ligne qui portait le total de la période reçoit, comme les autres, une
> valeur de sous-période.

`disaggregate_anchors` ([ARCH] §3.9, prompt 14 non exécuté) **n'existe pas** dans `hfi2` (D7).
Motif : une colonne ne doit jamais mélanger, à une fréquence d'imputation donnée, un total de
période et des valeurs de sous-période — cette hétérogénéité rend la colonne inexploitable comme
covariable, fausse toute agrégation en aval et rend l'échelle d'une ligne dépendante de sa
position dans la période.

Conséquences à documenter explicitement :

- sous `aggregation_constraint='sum'`, **aucune information n'est perdue** : la somme des
  sous-périodes reconstitue exactement le total observé, et la ligne d'ancre porte
  `DISAGGREGATED` ;
- sous `aggregation_constraint=None`, le total observé **est écrasé** par une prédiction libre.
  Il reste récupérable de deux manières, toutes deux à mentionner dans la docstring : par
  `inverse_transform`, et par le masque `ORIGINAL` du niveau de fréquence source dans la sortie
  multi-fréquences (`keep_lower_frequencies=True`) ;
- le marquage `DISAGGREGATED` de la ligne d'ancre est **indépendant de la réussite du recalage**
  (B2) : il dit « cette cellule occupe la place d'une observation réelle », pas « cette cellule
  respecte l'identité additive ».

**Exemple** (jeu `TS`, `a1` = 120 en 2021, étape `M`) : les 12 mois de 2021 reçoivent une valeur ;
le 2021-12-31 vaut 11.2 (et non 120) ; sous `'sum'` la somme des 12 mois vaut exactement 120.

---

## 12 — Architecture logicielle

### 12.1 — Principe : un plan, un exécuteur

La leçon des défauts B7/B27/B8 : chaque fois que `fit` et `transform` portent deux copies d'une
logique, elles divergent. `hfi2` sépare **construction du plan** et **exécution du plan** :

```
fit       = phases 0-4  ->  PlanBuilder      (construit ET exécute chaque étape au fil de l'eau,
                                              car l'étape k dépend des imputations de l'étape k-1)
transform = phases 0'-4' ->  PlanExecutor    (rejoue le plan figé)
```

- Le **plan** (`List[ImputationStep]`) est l'état ajusté complet.
- **Une seule implémentation de l'exécution d'étape** (`_execute_step`), paramétrée par
  « fit : ajuster puis prédire » vs « transform : prédire avec le modèle figé ». Les écritures
  (vidage/réécriture, recalage, provenance, stores) sont **communes par construction**.

Ce qui est **état du fit** (rejoué tel quel) vs **recalculé au transform** :

| Rejoué depuis le fit | Recalculé sur les données du transform |
|---|---|
| classification des variables, fréquences détectées, progression de fréquences, ordre des variables | fenêtres d'imputation / d'entraînement (§7, B1) |
| modèles ajustés, `feature_cols`, **voie de matérialisation par covariable** (§4.6), facteurs et modalités d'échelle | frames d'étape, valeurs interpolées, prédictions, provenance du transform |
| méthodes et ancrages d'interpolation par feature, souillures de l'étape (`covariate_taint`, `target_taint`) | masques de prédiction, recalage aux totaux, `origin_store` du transform |

**Comportement en cas de divergence des fréquences détectées au `transform` (D11)** : la
détection est refaite sur les données du transform et **comparée** à celle du fit. Divergence sur
une colonne présente → `UserWarning` unique (message listant les colonnes et les deux fréquences)
puis **poursuite avec les fréquences du fit**. Colonne du fit **absente** des données du
transform → `ValueError` nommant les colonnes manquantes. Colonnes supplémentaires au transform →
ignorées silencieusement (elles ne sont dans aucun plan).

### 12.2 — Découpage en composants

Fichiers nouveaux dans `tsforecast/frequency/` — une responsabilité par module, la classe
principale restant un orchestrateur mince :

| Composant | Fichier | Responsabilité |
|---|---|---|
| `HighFrequencyImputer2` | `high_frequency_imputer2.py` | API sklearn, validations `__init__`, normalisation au `fit`, orchestration `fit`/`transform`/`inverse_transform` |
| `CovariateMaterializer` | `covariate_materializer.py` | matérialisation des covariables sur une grille selon `covariate_strategy` / `covariate_fallback` / `interpolation_*` ; **unique** producteur de `X_train` et `X_pred` ; tient `imputed_store`, `imputed_freq_store` et `origin_store` ; applique la précédence du §4.4 |
| `StageScaler` | `stage_scaler.py` | diviseurs `'constant'`/`'calendar'`, scalaires et par ligne ; application et inversion de l'échelle ; report d'échelle des prédictions |
| `VariableOrderer` | `variable_orderer.py` | ordres `'frequency'` et `'cv'` (avec `cv`, `cv_scoring`, `min_cv_train_size`), tie-break alphabétique |
| `AggregationConstraint` | `aggregation_constraint.py` | recalage aux totaux de période, gardes du §11.1, masque des cellules recalées |
| `ImputationStep` (v2) + plan | `imputation_plan2.py` | étape immuable : + `covariate_taint`, + `target_taint`, + `materialization`, + `is_fallback` ; − `trained_on_imputed`, − `feature_means` |
| réutilisés | `imputation_window.py` (3 masques, MultiIndex), `frequency_aligner.py`, `provenance.py` (enum étendue), `target_frequency_validator.py`, `regularizer.py`, `utils/frequency/converter.py` (+ `anchor_fraction`, correctif B26) | rôle inchangé |

`hfi` et `hfi2` **coexistent** pendant la transition (exports distincts dans
`tsforecast/frequency/__init__.py`) ; le remplacement effectif (dépréciation puis suppression de
`HighFrequencyImputer`) est un chantier ultérieur, hors périmètre de ce document. Seule exception
à la non-régression de `hfi` : la suppression de `MODEL_ON_MIXED` (§6.7).

### 12.3 — Squelette de `fit`

```
PHASE 0  setup : purge de l'état de transform (B19), colonnes, détection panel,
         alignement et nommage de y (B14 : égalité des index, pas seulement des longueurs),
         détection des fréquences, normalisation/validation de target_frequency (B16),
         classification des variables
PHASE 1  calcul des fenêtres (trois masques, §7)
PHASE 2  ajustement du transformateur additif
PHASE 3  construction de la progression de fréquences (§5.2)
PHASE 4  initialisation du tracker de provenance — APRÈS le transformateur additif (B8)
PHASE 5  pour chaque étape de fréquence f de la progression :
  5a. frame d'étape : données d'origine + agrégations exactes à f + miroir des imputations
      (CovariateMaterializer, unique aussi pour le transform)
  5b. variables imputables à f ; ordre (VariableOrderer, SEULEMENT si strategy='model')
  5c. pour chaque variable v :
      - grille d'entraînement (ancres de v [+ cellules éligibles selon ELIGIBLE_ORIGINS, §5.3]),
        masque 'training' ; grille de prédiction, masque 'imputation'
      - sélection des feature_cols (non-vides sur LES DEUX fenêtres, covariate_eligibility)
      - matérialisation des covariables sur LES DEUX grilles par la MÊME voie (§4.6),
        enregistrement de materialization[col]
      - calcul des souillures covariate_taint / target_taint (§6.2)
      - mise à l'échelle (StageScaler : scalaire ou Series par ligne)
      - ajustement de l'estimateur ; en cas d'échec -> repli interpolation (méthode de v),
        étape marquée is_fallback=True, cellules marquées INTERPOLATED
      - prédiction sur TOUTE la période, ancres comprises (§11.2)
      - recalage aux totaux (AggregationConstraint) -> masque des cellules recalées
      - écriture des valeurs, marquage de provenance (§6.3 pour les cellules non recalées,
        DISAGGREGATED pour les recalées et pour les ancres)
      - mise à jour de imputed_store / imputed_freq_store / origin_store
        (y compris en repli : "le repli matérialise")
      - gel de l'ImputationStep dans le plan
PHASE 6  finalisation : plan figé, attributs de sortie, sortie multi-fréquences si demandé
```

### 12.4 — `transform`, `inverse_transform`, `keep_lower_frequencies`

- **`transform`** : phases 0'-4' data-dépendantes (alignement/nommage de `y` par la **même**
  fonction qu'au fit — B14 ; transformateur additif appliqué avec l'objet **ajusté** ; tracker
  initialisé après lui — B8 ; fenêtres **recalculées** — B1 ; contrôle des fréquences — D11),
  puis `PlanExecutor` qui rejoue les étapes dans l'ordre du plan.
- **`inverse_transform`** : reprise du chemin actuel (sélection du niveau de fréquence source,
  inversion du transformateur additif, restitution par masque `ORIGINAL`,
  `restore_original_values`), avec les invariants B4 (panels à n > 2 niveaux d'entité, noms
  d'index préservés) et B19.
- **`keep_lower_frequencies`** : conservé (nom compris), documenté comme **paramètre d'affichage
  pur** — il gouverne l'empilage multi-fréquences de la sortie, jamais la logique. Sous
  `impute_intermediate_frequencies=False`, il n'y a pas de niveau intermédiaire à empiler : la
  sortie ne contient que le niveau source et le niveau cible. À documenter tel quel.

### 12.5 — Conformité sklearn et contrat d'entrée (repris de [ARCH] §3.16, non négociables)

- **B3** : `__init__` stocke les paramètres **tels que reçus** ; `clone` et `get_params` exacts ;
  toute normalisation a lieu au `fit`, dans des attributs suffixés `_`.
- **B14** : nommage unique de `y`, vérification d'**égalité des index** (pas seulement des
  longueurs).
- **B15** : panel déclaré par `panel_cols` sur frame plat, pleinement fonctionnel.
- **B16** : dict `target_frequency` incomplet → `ValueError` nommant les entités manquantes.
- **B19** : purge de l'état de `transform` en tête de `fit`.
- **B20** : `NotFittedError` propre avant `fit`, via `check_is_fitted` avec une **liste explicite**
  d'attributs.
- **Avertissements uniques** (estimateur absent, lignes hors fenêtre, fréquences divergentes,
  périodes à signe inversé), jamais un par variable × étape : accumulation puis message agrégé en
  fin de phase.

---

## 13 — API résultante, paramètre par paramètre

```python
class HighFrequencyImputer2(XYPanelTimeSeriesTransformer):
    def __init__(
        self,
        target_frequency: Union[str, Dict[Union[str, tuple], str]],
        estimator: Optional[Union[BaseEstimator, Dict[str, BaseEstimator]]] = None,
        additive_transformer: Optional[TransformerMixin] = None,

        # --- Axe 1 : matérialisation des covariables (§4) ---
        covariate_strategy: Literal['tolerate_nan', 'interpolate', 'model'] = 'interpolate',
        covariate_fallback: Literal['interpolate', 'tolerate_nan'] = 'interpolate',
        covariate_eligibility: Literal['any_entity', 'all_entities'] = 'any_entity',
        interpolation_method: Union[str, Dict[str, str]] = 'linear',
        interpolation_anchor: Union[None, float, Dict[str, Optional[float]]] = None,

        # --- Axe 2 : fréquences intermédiaires (§5) ---
        impute_intermediate_frequencies: Literal[False, 'covariates_only', True] = False,

        # --- Ordre d'imputation (§8) ---
        fit_predict_order: Literal['frequency', 'cv'] = 'frequency',
        cv: Union[int, BaseCrossValidator, Iterable, None] = None,
        cv_scoring: Union[str, Callable] = 'neg_mean_absolute_percentage_error',
        min_cv_train_size: int = 10,

        # --- Fenêtres (§7) ---
        imputation_scope: ImputationScope = 'strict',
        coverage_threshold: float = 0.5,
        training_scope: Optional[TrainingScope] = None,
        training_coverage_threshold: Optional[float] = None,

        # --- Échelle et contraintes (§9, §11) ---
        scale_features: Union[Literal[False], ScaleMode,
                              Dict[str, Union[Literal[False], ScaleMode]]] = 'constant',
        aggregation_constraint: Literal['sum', None] = 'sum',

        # --- Sortie et divers ---
        keep_lower_frequencies: bool = True,
        on_frequency_mismatch: Literal['error', 'warn'] = 'error',
        restore_original_values: bool = False,
        time_col: Optional[str] = None,
        panel_cols: Optional[List[str]] = None,
        verbose: bool = False,
    ): ...
```

### 13.1 — Validations à `__init__` (sans transformation, B3)

| Paramètre | Contrôle |
|---|---|
| `target_frequency` | str non vide normalisable, ou dict non vide `{entité: str}` (contenu vérifié au `fit` contre les entités réelles) |
| `estimator` | `None`, objet exposant `fit`/`predict`, ou dict `{colonne: estimateur}` (clé `'__default__'` admise) ; avertissement **unique** si `None` |
| `additive_transformer` | `None` ou objet exposant `fit_transform` **et** `inverse_transform` |
| `covariate_strategy`, `covariate_fallback`, `covariate_eligibility`, `fit_predict_order`, `on_frequency_mismatch`, `imputation_scope`, `training_scope` | appartenance au `Literal` (message listant les valeurs admises) |
| `impute_intermediate_frequencies` | `is False`, `== 'covariates_only'`, ou `is True` — **jamais** un test de vérité booléenne : `'covariates_only'` est *truthy*, un `if self.impute_intermediate_frequencies:` serait un bug silencieux |
| `interpolation_method` | str, ou dict `{str: str}` (clés vérifiées au `fit`) |
| `interpolation_anchor` | `None`, float dans `[0, 1]`, ou dict de ces valeurs |
| `cv` | `None`, `int ≥ 2`, objet à `split`/`get_n_splits`, ou itérable (`check_cv` au `fit` seulement) |
| `cv_scoring` | str ou callable |
| `min_cv_train_size` | `int ≥ 1` |
| `coverage_threshold`, `training_coverage_threshold` | float dans `[0, 1]` (ou `None` pour le second) |
| `scale_features` | `False`, `'constant'`, `'calendar'`, ou dict de ces valeurs |
| `aggregation_constraint` | `'sum'` ou `None` exactement ; message d'erreur mentionnant l'extension réservée (§11.1) |
| booléens (`keep_lower_frequencies`, `restore_original_values`, `verbose`) | validation groupée, comme dans `hfi` |

Combinaisons **inertes** documentées mais **non signalées** (D9) : cf. §5.6.
`training_coverage_threshold` sans `training_scope` est inerte, à documenter.

### 13.2 — Attributs ajustés

| Attribut | Contenu |
|---|---|
| `effective_target_frequency_` | fréquence cible normalisée (scalaire ou dict par entité) |
| `detected_frequencies_` | `{colonne: fréquence}` détectée au fit |
| `variable_categories_` | classification des colonnes, format [ARCH] §3.2 |
| `frequency_progression_` | liste des `f_stage` (§5.2) |
| `imputation_order_` | ordre des variables par étape (vide hors `covariate_strategy='model'`) |
| `imputation_plan_` | `List[ImputationStep]` — l'état ajusté complet |
| `imputation_models_` | vue `{(étape, variable): estimateur}` sur le plan |
| `imputation_window_mask_`, `training_window_mask_`, `strict_window_mask_` | `pd.Series` booléennes (MultiIndex sur panel, §7.2) |
| `imputation_window_`, `training_window_` | bornes `(début, fin)` lisibles |
| `imputation_provenance_` | matrice de provenance après `fit` puis après `transform` |
| `feature_columns_`, `target_column_`, `entities_`, `is_panel_` | contrat d'entrée |
| `cv_` | splitter résolu par `check_cv` (présent seulement sous `fit_predict_order='cv'`) |

---

## 14 — Journal des décisions arrêtées

Toutes les questions ouvertes de la version 1 du document sont tranchées. Les codes `Axx`/`Dxx`
sont ceux de la version 1, conservés pour la traçabilité.

### 14.1 — Ambiguïtés de la spécification orale

| Code | Question | Arbitrage |
|---|---|---|
| A1 | phrase tronquée du point 1 | **sans objet** — les arguments non cités suivent le §1.4 |
| A2 | phrase tronquée du point 4 | **sans objet** |
| A3 | « on pourra également [spécifier] cela sous la forme d'un dictionnaire » | **confirmé** : forme dict `{feature: modalité}` pour `scale_features`, `interpolation_method`, `interpolation_anchor` (§9.1, §10) |
| A4 | défauts non spécifiés | **confirmés** : `covariate_strategy='interpolate'`, `covariate_fallback='interpolate'`, `interpolation_method='linear'`, `interpolation_anchor=None`, `scale_features='constant'` |

### 14.2 — Décisions de conception

| Code | Décision | Conséquence dans ce document |
|---|---|---|
| **D1** | reprise du bloc de validation groupée des booléens et des littéraux de `hfi` | §13.1 |
| **D2** | **abandon** du mode « un seul fit réutilisé » (`cascade_refitting=False`) ; la capacité « étapes intermédiaires sans bruit dans `y` » est en revanche **restaurée** sous la forme `impute_intermediate_frequencies='covariates_only'` | §5.7 |
| **D3** | `impute_intermediate_frequencies` par défaut **`False`** | §5.1 |
| **D4** | paramètre **`cv`** sklearn polymorphe ; **suppression** de `cv_n_splits` | §8.3 |
| **D5** | noms validés : `covariate_strategy`, `covariate_fallback`, `impute_intermediate_frequencies`, `fit_predict_order`, `interpolation_anchor` — **sauf** `enforce_aggregation_constraints`, remplacé par **`aggregation_constraint`** | §11.1, §13 |
| **D6** | provenance en **échelle de souillure** : vraies + agrégées → `MODEL_ON_TRUE` ; + interpolées → `MODEL_ON_INTERPOLATED` ; + au moins une covariable imputée → `MODEL_ON_IMPUTED` ; souillure venue de `y_train` distinguée par `MODEL_ON_IMPUTED_TARGET` / `MODEL_ON_IMPUTED_BOTH` ; **`MODEL_ON_MIXED` supprimé**, y compris pour `hfi`. Les cellules de repli portent `INTERPOLATED` | §6 |
| **D7** | **pas** de `disaggregate_anchors` : une variable est imputée sur toute sa période, ancres comprises, comportement **non paramétrable** | §11.2 |
| **D8** | `aggregation_constraint: Literal['sum', None] = 'sum'`, extensible (`'mean'`, `'last'`, forme dict) — extension **réservée**, refusée à la validation aujourd'hui | §11.1 |
| **D9** | paramètres inertes : **silence + docstring**, jamais d'avertissement | §5.6, §8.1 |
| **D10** | dictionnaires par feature indexés par **nom de colonne**, jamais par `(entité, colonne)` | §9.1, §10 |
| **D11** | `transform` face à des fréquences divergentes : **avertissement + poursuite** avec les fréquences du fit ; **`ValueError`** si une colonne du fit manque | §12.1 |

### 14.3 — Décision dérivée, introduite par cette révision

| Code | Décision | Motif |
|---|---|---|
| **D12** | introduction de `CellOrigin` / `origin_store` : le filtre de `y_train` et le calcul des souillures lisent l'**origine** des cellules, jamais la matrice de provenance | `DISAGGREGATED` est ambigu (interpolation recalée vs prédiction recalée) ; sans ce registre, `'covariates_only'` serait un synonyme silencieux de `True` (§5.3) |
| **D13** | précédence de matérialisation à quatre rangs, dont le **report d'étape** (rang 3) | c'est ce qui donne un effet observable à `'covariates_only'` et ce qui corrige la première cause de B28 (§4.4) |
| **D14** | l'invariant NaN se formule en **inclusion d'ensembles de dates renseignées**, pas en comparaison de taux bruts | sous `'tolerate_nan'`, les deux grilles n'ont pas le même pas (§4.7) |

---

## 15 — Prérequis et travaux annexes

| Code | Travail | Statut |
|---|---|---|
| **P1** | **B26 / prompt 21** : le contrôle `full_periods_only` de `utils/frequency/converter.py` utilise un décompte **constant** et jette février ; toute covariable journalière disparaît silencieusement du jeu d'entraînement. `hfi2` s'appuyant sur les mêmes agrégations, c'est un **prérequis dur**, indépendant de `hfi2`, à exécuter **en premier** | à faire (L1) |
| **P2** | extension `FrequencyConverter.interpolate_to_higher_frequency(anchor_fraction=...)` (§10.2) : capacité utilitaire testable isolément, prérequis de la stratégie `'interpolate'` complète | à faire (L1) |
| **P3** | extension `ImputationWindowCalculator` : trois masques + `training_scope` + retour MultiIndex (§7) — prompts 10/13/20 de [ARCH] jamais exécutés, réalisés ici sur `iwc`, **partagé avec `hfi` sans changement de comportement par défaut** | à faire (L3) |
| **P4** | extension `ProvenanceType` + `mark_model_imputed` (§6) — **rupture assumée** pour `hfi` (suppression de `MODEL_ON_MIXED`) | à faire (L2) |
| **P5** | jeu `PANEL` avec feature manquante par entité (§15.1) — **à faire en premier**, les jeux d'exemple servant ensuite à tous les tests et notebooks | à faire (L0) |

### 15.1 — Notebook 3 : feature absente pour certaines entités

`notebooks/3 - QB - Panel a frequences mixtes heterogene.ipynb`, fonction `create_panel_dataset` :
ajouter une variable (`climat_affaires`, indicateur mensuel d'enquête) présente pour la France et
l'Allemagne, **absente pour l'Italie** (colonne entièrement NaN pour cette entité — la colonne
existe dans le frame, c'est l'entité qui ne l'observe jamais). Compléter la section 2.3 de
vérification. Le jeu étant réutilisé en aval (notebook 4, futur notebook 5, fixtures de tests qui
s'en inspirent), vérifier et ré-exécuter les consommateurs. C'est le cas d'usage de
`covariate_eligibility` (§4.5) et la limite documentée de la stratégie `'interpolate'`.

### 15.2 — Notebook 5 : pas à pas de `HighFrequencyImputer2`

Sur le modèle de `notebooks/4 - QB - HighFrequencyImputer pas a pas.ipynb`, un notebook
`5 - QB - HighFrequencyImputer2 pas a pas.ipynb` qui, sur **les deux jeux** du notebook 3 :

- détaille variables et données à chaque phase de `fit` et de `transform` ;
- affiche, **pour chaque étape du plan** : `X_train`, `y_train` et `X_pred` exacts, la **voie de
  matérialisation** retenue par covariable, les trois masques de fenêtre appliqués, les deux
  souillures de l'étape, et la provenance après écriture ;
- **audite le scaling** : moyenne de chaque feature sur le train et sur le test, côte à côte —
  l'écart relatif doit être compatible avec la modalité choisie (`'constant'` vs `'calendar'`),
  un déséquilibre train/test étant le symptôme immédiat d'un diviseur faux ;
- montre les **six combinaisons** des deux axes (3 stratégies × 3 modalités, réduites aux six
  qui diffèrent effectivement, §5.6) avec la matrice de provenance résultante et les cinq
  familles `MODEL_*` ;
- conserve les contrôles croisés du notebook 4 : pas-à-pas vs `fit_transform`, et
  `fit_transform(X)` vs `fit(X).transform(X)`.

**Règle permanente** reprise de [ARCH] §6 : tout prompt d'implémentation qui modifie l'exécution
d'étape met à jour la cellule pas-à-pas correspondante **dans le même lot**.

### 15.3 — Coexistence et migration

`hfi2` est développé à côté de `hfi`, sans toucher au comportement par défaut des composants
partagés (`iwc`, converter, provenance — extensions rétro-compatibles, à l'exception de
`MODEL_ON_MIXED`). La dépréciation de `HighFrequencyImputer` n'intervient qu'après validation
croisée des deux classes sur les notebooks — hors périmètre du présent chantier.

---

## 16 — Stratégie de tests et invariants

Suite dédiée `tests/frequency/test_high_frequency_imputer2*.py`, fixtures partagées avec
l'existant (`tests/frequency/conftest.py`), critère « aucun échec nouveau » vs
`BASELINE_FAILURES.txt` pour les composants partagés. Chaque invariant est testé sur la **série
temporelle** ET sur le **panel** (y compris l'entité sans feature, §15.1).

| Code | Invariant | Détail |
|---|---|---|
| **I1** | symétrie | `fit_transform(X)` ≡ `fit(X).transform(X)` : égalité stricte des valeurs, des provenances et des attributs de sortie |
| **I2** | invariant NaN (§3) | par étape et par colonne, l'ensemble des dates renseignées dans `X_pred` contient l'image de celui de `X_train` (formulation D14) ; mesuré par estimateur espion, sous les trois stratégies. C'est le test qui échoue sur `hfi` aujourd'hui |
| **I3** | invariance à l'ordre des colonnes | permuter les colonnes d'entrée ne change ni les valeurs ni les provenances, sous les trois stratégies (sous `'model'`, grâce au tie-break alphabétique §8.1) |
| **I4** | additivité | sous `aggregation_constraint='sum'`, chaque colonne imputée somme au total observé de chaque période **complète** — imputations, interpolations et covariables portées comprises ; et la ligne d'ancre ne porte plus le total (§11.2) |
| **I5** | échelle train/test | pour chaque étape, moyennes des features de `X_train` et de `X_pred` comparables (tolérance dépendant de la modalité) ; inclut le cas mixte « feature `'calendar'`, `y` `'constant'` » |
| **I6** | provenance | les cinq familles `MODEL_*` sont émises exactement dans les cas du §6.3 et aucun autre ; `MODEL_ON_IMPUTED*` seulement sous `'model'` ; `*_TARGET`/`*_BOTH` seulement sous `impute_intermediate_frequencies=True` ; les cellules de repli portent `INTERPOLATED` ; `MODEL_ON_MIXED` n'existe plus |
| **I7** | `transform` hors fenêtre de fit | impute au lieu de vider, ne détruit jamais une observation d'entrée, avertit **une fois** sur les lignes inimputables |
| **I8** | aller-retour | `inverse_transform(transform(X))` restitue l'index, les noms (panels multi-niveaux compris) et, sous `restore_original_values=True`, les valeurs d'origine |
| **I9** | conformité sklearn | `clone`, `get_params`/`set_params`, `Pipeline`, `GridSearchCV` sur panel avec `target_frequency` dict ; `NotFittedError` avant `fit` |
| **I10** | indifférence de l'ordre hors `'model'` | sous `'tolerate_nan'` et `'interpolate'`, forcer deux ordres de traitement différents produit des sorties identiques |
| **I11** | unicité de la voie de matérialisation | pour chaque (étape, variable, covariable), `materialization` est identique au fit et au transform, et la nature des valeurs produites l'est aussi (§4.6) |
| **I12** | `'covariates_only'` ≠ `True` | sur un jeu où la cascade change quelque chose, `'covariates_only'` produit des `y_train` sans aucune ligne d'origine `'model'`, et des valeurs finales **différentes** de `True` ; sous `covariate_strategy='interpolate'`, `'covariates_only'` produit les mêmes valeurs finales que `False` (§5.6) |
| **I13** | `impute_intermediate_frequencies` n'est jamais testé comme booléen | test statique/grep : aucun `if self.impute_intermediate_frequencies:` dans le code (`'covariates_only'` est *truthy*) |

Cas limites à couvrir explicitement, en plus : index non trié, index dupliqué, entité à une seule
observation, variable annuelle à 2 ancres seulement (`y_train` de taille 2 sous
`impute_intermediate_frequencies=False`), période incomplète en début et en fin de série,
fréquence irrégulière détectée, colonne entièrement NaN, `estimator=None`.

---

## 17 — Lots d'implémentation

Les lots ci-dessous sont l'ossature de la future liste de prompts : chacun est indépendamment
testable, et l'ordre des dépendances est strict. Un lot ne se clôt qu'avec ses tests **et** la
mise à jour du notebook concerné quand il touche l'exécution d'étape (§15.2).

| Lot | Contenu | Dépend de | Livrables de test |
|---|---|---|---|
| **L0** | jeu `PANEL` du notebook 3 : feature `climat_affaires` absente pour `IT` ; ré-exécution des consommateurs | — | section 2.3 du notebook 3, fixtures `conftest.py` |
| **L1** | correctifs et extensions de `utils/frequency/converter.py` : B26 (décompte calendaire de `full_periods_only`) et `anchor_fraction` (§10.2) | — | `tests/utils/frequency/test_converter.py` : février, T1/T2, ancrage 0.0/0.5/1.0, union d'index |
| **L2** | `provenance.py` : nouvelle énumération (§6.1), `resolve_model_provenance`, `mark_model_imputed` (§6.6), mise à jour mécanique de `hfi` et de ses tests (§6.7) | — | `tests/frequency/test_provenance.py` ; non-régression `hfi` |
| **L3** | `imputation_window.py` : trois masques `kind=`, `training_scope`/`training_coverage_threshold`, retour `pd.Series` MultiIndex, B23/B24 | — | `tests/frequency/test_imputation_window.py` ; non-régression `hfi` |
| **L4** | `imputation_plan2.py` : `ImputationStep` v2 (champs du §12.2), plan immuable, sérialisation de diagnostic | L2 | tests unitaires du plan |
| **L5** | `stage_scaler.py` : diviseurs `'constant'`/`'calendar'`, scalaires et `Series`, B25, B12, forme dict par feature | L1 | tests unitaires isolés + I5 |
| **L6** | `covariate_materializer.py` : les trois stratégies, `covariate_fallback`, précédence à quatre rangs (§4.4), `imputed_store`/`imputed_freq_store`/`origin_store`, `covariate_eligibility` | L1, L2, L3 | tests unitaires isolés + I2, I11 |
| **L7** | `aggregation_constraint.py` : recalage aux totaux, gardes (§11.1), masque des cellules recalées, désagrégation systématique des ancres (§11.2) | L2 | tests unitaires + I4 |
| **L8** | `variable_orderer.py` : ordres `'frequency'` et `'cv'`, `check_cv`, tie-break alphabétique, correctifs §8.2 | — | tests unitaires + I3 |
| **L9** | `high_frequency_imputer2.py` — `__init__`, validations (§13.1), phases 0 à 4, attributs ajustés | L4–L8 | I9, tests de validation d'arguments |
| **L10** | `high_frequency_imputer2.py` — PHASE 5 : exécution d'étape unique, axe 1 complet, provenance, stores | L9 | I2, I3, I6, I10, I11 |
| **L11** | axe 2 : progression de fréquences, `ELIGIBLE_ORIGINS`, échelle par ligne, report d'étape | L10 | I12, I13, exemples chiffrés du §5.5 |
| **L12** | `transform`, `inverse_transform`, `keep_lower_frequencies`, contrôle des fréquences (D11), avertissements uniques | L11 | I1, I7, I8 |
| **L13** | notebook 5 pas à pas (§15.2) et documentation (`mkdocs`, docstrings de référence) | L12 | exécution complète du notebook |

**Points de vigilance à rappeler dans chaque prompt d'implémentation** :

1. commentaires internes en **français**, formulations nominales ; docstrings en **anglais**,
   Google Style, avec `Args`/`Returns`/`Raises`/`Examples` ;
2. `impute_intermediate_frequencies` ne se teste **jamais** par vérité booléenne ;
3. `X_train` et `X_pred` ne sont produits **que** par `CovariateMaterializer.materialize` ;
4. le filtre de `y_train` et les souillures lisent `origin_store`, **jamais** la provenance ;
5. tout avertissement est **agrégé et unique** ;
6. `__init__` valide sans transformer (B3).
