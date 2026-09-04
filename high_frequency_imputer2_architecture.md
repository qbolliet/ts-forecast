# `HighFrequencyImputer2` — spécification d'architecture

> **Version 2 du document — 2026-09-01** (branche `qb-mixed-frequencies`).
> Rédigé initialement le 2026-08-31 à partir de la spécification orale du 2026-08-31, de la
> relecture de `_fit` dans `tsforecast/frequency/high_frequency_imputer.py`, et des deux
> documents de la campagne d'annotations : `high_frequency_imputer_annotations_architecture.md`
> (noté ci-après **[ARCH]**) et `high_frequency_imputer_annotations_prompts.md` (prompts 1 à 12
> et 22 exécutés). **Révisé le 2026-09-01** pour intégrer les arbitrages de l'auteur : toutes les
> décisions ouvertes du §11 de la version 1 sont désormais tranchées et consignées au §14.
>
> **Révision du 2026-09-04** : intégration de la **mutualisation inter-entités du jeu
> d'entraînement** (§5.8, décisions D17 à D19, défaut mesuré B29). Une même colonne peut porter
> une fréquence différente selon l'entité d'un panel ; les vraies valeurs des entités qui
> l'observent plus finement doivent entraîner le modèle. Sections amendées : §0, §1.5 (nouvelle),
> §2.1, §2.5 (nouvelle), §5.3, §5.4, §5.8 (nouvelle), §7.2, §9.2, §12.2, §12.3, §13.2, §14.3,
> §16, §17.
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

**Les neuf décisions structurantes arrêtées** (détail et justification au §14) :

1. **Provenance en échelle de souillure** : `MODEL_ON_TRUE` → `MODEL_ON_INTERPOLATED` →
   `MODEL_ON_IMPUTED`, plus deux libellés distinguant la souillure venue de la **cible** :
   `MODEL_ON_IMPUTED_TARGET` et `MODEL_ON_IMPUTED_BOTH`. `MODEL_ON_MIXED` est **supprimé** de
   l'énumération (§6, D6).
2. **`impute_intermediate_frequencies` à trois modalités** `False` / `'covariates_only'` /
   `True`, défaut `False` (§5, D3).
3. **`aggregation_constraint: Union[Literal['sum', 'mean', 'last', None], Dict[str, ...]] = 'sum'`**
   remplace `enforce_period_totals` : quatre contraintes scalaires et une forme dictionnaire par
   colonne, avec clé `'__default__'` (§11, D8).
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
9. **Le jeu d'entraînement est mutualisé entre entités** : une colonne peut porter une fréquence
   différente selon l'entité, et toutes les entités qui l'observent entraînent le même modèle,
   chacune à sa propre fréquence, ramenée à l'échelle de l'étape par un diviseur **par ligne**.
   **Non paramétrable** (§5.8, D17 à D19).

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
| `enforce_period_totals: bool` | **remplacé** par `aggregation_constraint`, élargi en `Union[Literal['sum', 'mean', 'last', None], Dict[str, ...]]` | §11.1, D8 |
| `cv_n_splits: int` | **supprimé**, absorbé par `cv` (contrat sklearn) | §8.3, D4 |
| `covariate_eligibility` | **conservé**, sémantique recadrée sur le seul cas « feature absente pour la totalité d'une entité » | §4.5 |
| `keep_lower_frequencies` | **conservé** tel quel, paramètre d'affichage pur, correctif B4 inclus | §12.4 |
| — | **nouveaux** : `covariate_strategy`, `covariate_fallback`, `impute_intermediate_frequencies`, `interpolation_method`, `interpolation_anchor`, `cv`, `training_scope`, `training_coverage_threshold` | §13 |

Deux constantes de l'énumération de provenance disparaissent ou changent :
`ProvenanceType.MODEL_ON_MIXED` est **supprimée** (§6.7), et le champ
`ImputationStep.trained_on_imputed: bool` est remplacé par le couple
`(covariate_taint, target_taint)` (§6.2).

---

### 1.5 — Le second défaut structurel mesuré (B29) : la mutualisation implicite et fausse

Le défaut B28 du §1.1 porte sur les **covariables**. Un second défaut, de même nature mais sur
l'axe **cible**, apparaît dès qu'un panel porte une colonne à des fréquences différentes selon
l'entité — cas courant en pratique : un agrégat publié annuellement dans un pays, trimestriellement
dans un autre, mensuellement dans un troisième.

Mesure sur un panel `FR`/`DE`/`IT`, colonne `v` annuelle pour `FR` (≈ 120/an), trimestrielle pour
`DE` (≈ 30/trimestre), mensuelle pour `IT` (≈ 10/mois), covariable `m1` mensuelle partout,
`target_frequency='M'`, estimateur espion enregistrant `y_train` :

| Étape | Groupe `(variable, f_var)` | Lignes de `y_train` | Cible mise à l'échelle |
|---|---|---|---|
| `Q` | `(v, Y)` → `{FR}` | 9 : **3 FR, 3 DE, 3 IT** | FR 29.9 · DE 7.4 · IT 2.4 |
| `M` | `(v, Y)` → `{FR}` | 9 : 3 FR, 3 DE, 3 IT | FR 9.96 · DE 2.48 · IT 0.78 |
| `M` | `(v, Q)` → `{DE}` | 27 : 3 FR, 12 DE, 12 IT | FR 39.8 · DE 10.2 · IT 3.2 |

Trois constats, tous structurels dans `hfi` :

- **le mélange a lieu** : `hfi:_prepare_training_data` lit `y_source = X_original[var_name]`,
  c'est-à-dire la colonne **entière, toutes entités confondues**, et ne la filtre que par
  `notna()` et par la fenêtre ; aucune restriction aux entités du groupe n'existe avant le `fit` ;
- **l'échelle est fausse** : le diviseur est le **scalaire de l'étape**, calculé sur la clé
  représentative du groupe (`Y` ici). Les vraies valeurs trimestrielles de `DE`, déjà à l'échelle
  de l'étape `Q`, sont divisées par 4 ; les mensuelles d'`IT`, qu'il faudrait agréger, le sont
  aussi. Le modèle apprend une cible qui mêle trois régimes d'échelle contre les mêmes features ;
- **le volume n'y est pas** : seules 3 lignes par entité étrangère survivent, les covariables
  étant agrégées à `f_var` du groupe (grille annuelle) et les lignes sans covariable observée
  étant écartées. Le gisement réel à l'étape `Q` — 12 vraies valeurs trimestrielles de `DE` plus
  12 agrégats trimestriels exacts d'`IT`, contre 3 ancres `FR` — n'est pas exploité.

Autrement dit, `hfi` a le **coût** de la mutualisation (biais d'échelle silencieux) sans son
**bénéfice** (volume). `hfi2` répond par le §5.8 : la mutualisation devient explicite, chaque
entité contribue **à sa propre fréquence**, et le diviseur devient une propriété de la **ligne**.

---

## 2 — Vocabulaire normatif et jeux de référence

Tout le document utilise ce vocabulaire ; l'implémentation doit reprendre ces mots dans les noms
de variables et les docstrings.

### 2.1 — Vocabulaire

| Terme | Définition normative |
|---|---|
| **fréquence cible** `f_target` | la fréquence à laquelle toutes les colonnes imputables doivent être disponibles en sortie (`target_frequency`, éventuellement par entité). |
| **fréquence détectée** `f_var(e, c)` | la fréquence propre d'une colonne **pour une entité**, inférée en phase 0 par `FrequencyDetector` sur les dates où cette entité l'observe. Une série temporelle est le cas dégénéré d'entité `()` et l'on écrit `f_var(c)`. **Un panel peut porter la même colonne à des fréquences différentes selon l'entité** (§2.5) : `detected_frequencies_` est indexé par `(entité, colonne)` dès que les entités divergent, et tout raisonnement sur `f_var` se fait **par entité**. |
| **bloc d'entraînement** | la contribution d'une entité au jeu d'entraînement d'une variable à une étape : ses lignes, prises à sa **fréquence de bloc**, avec ses covariables matérialisées sur cette grille (§5.8). |
| **fréquence de bloc** `f_block(e)` | la plus **basse** des deux fréquences `f_var(e, v)` et `f_stage` : on n'entraîne jamais sur une grille plus fine que la grille de prédiction (D18). |
| **mutualisation** | l'assemblage des blocs de **toutes** les entités observant `v` en un jeu d'entraînement unique, chaque ligne ramenée à l'échelle de l'étape par le diviseur de son bloc (§5.8, D17). |
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

### 2.5 — Jeu de référence `PANEL-F` (fréquences hétérogènes par entité)

Support de la mutualisation (§5.8) et de ses valeurs d'or. Trois entités `FR`, `DE`, `IT`, index
mensuel fin de mois 2021-01-31 → 2023-12-31 (36 dates par entité, 108 lignes),
`target_frequency='M'`, toutes les colonnes additives :

| Colonne | `FR` | `DE` | `IT` |
|---|---|---|---|
| `m1` | mensuelle, dense (`100 + rang`) | idem | idem |
| `q1` | trimestrielle, 12 ancres (`10 × k`) | idem | idem |
| `v` | **annuelle**, 3 ancres | **trimestrielle**, 12 ancres | **mensuelle**, 36 valeurs |

Valeurs d'or de `v`, choisies pour que les trois entités portent le **même total annuel** et que
tous les agrégats tombent juste :

| Année | `FR` (ancre de fin d'année) | `DE` (4 ancres de fin de trimestre) | `IT` (valeur mensuelle constante) |
|---|---|---|---|
| 2021 | **120** | 28, 30, 31, 31 (somme 120) | **10.0** (somme 120, trimestres 30) |
| 2022 | **132** | 31, 33, 34, 34 (somme 132) | **11.0** (somme 132, trimestres 33) |
| 2023 | **150** | 36, 37, 38, 39 (somme 150) | **12.5** (somme 150, trimestres 37.5) |

Fréquences détectées : `{(FR, v): Y, (DE, v): Q, (IT, v): M, (·, q1): Q, (·, m1): M}`.
`v` est **imputable** pour `FR` (annuelle → mensuelle) et pour `DE` (trimestrielle → mensuelle),
et **ne l'est pas** pour `IT`, qui l'observe déjà à la fréquence cible.

Ce jeu est le support des invariants I14, I15 et I16 (§16) ; ses valeurs sont **gelées** au même
titre que celles du jeu `TS`.

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

Classification d'une covariable `c` face à une grille `f`, **entité par entité** (`f_c` désigne
`f_var(e, c)`, qui peut différer d'une entité à l'autre, §2.1) — c'est ce que fait déjà
`CovariateMaterializer._applicable_way`, qui dégrade la voie retenue pour la colonne à ce que
l'état de chaque entité permet :

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
- **Origine des cellules produites** : `'interpolated'` ; provenance publique `INTERPOLATED`,
  **y compris après recalage aux totaux** — la contrainte d'agrégation ne change aucune
  provenance (§11.2). Un modèle qui en consomme émet `MODEL_ON_INTERPOLATED` (§6.3).
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

Les trois modalités décrivent le contenu de `y_train` **pour une entité donnée** ; sur un panel,
ce contenu est ensuite **mutualisé entre entités** selon le §5.8, les deux mécanismes étant
orthogonaux (§5.3, point 4).

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

1. Soit `F` l'ensemble des fréquences détectées des couples **(entité, colonne) imputables** du
   périmètre, plus la fréquence cible `f_target`. Sur un panel où les entités divergent pour une
   même colonne, `F` contient donc **toutes** leurs fréquences : sur `PANEL-F` (§2.5),
   `F = {Y, Q, M}` alors qu'une lecture par colonne n'aurait vu qu'une fréquence pour `v`.
2. Si `impute_intermediate_frequencies is False` : `progression = [f_target]`.
3. Sinon : `progression = sorted(F \ {la plus basse}, de la plus basse à la plus haute)`, en ne
   retenant que les fréquences **strictement plus hautes** que la plus basse fréquence source et
   **inférieures ou égales** à `f_target`, et en garantissant que `f_target` en est le dernier
   élément.
4. À chaque étape `f`, les **variables imputables à `f`** sont les couples `(entité, colonne)`
   dont `f_var(e, c)` est strictement plus basse que `f` et qui ne sont pas encore imputés
   **à `f`**. Les couples ainsi retenus sont regroupés par `(colonne, f_var)` : un groupe par
   fréquence source, chacun donnant une étape de plan, **toutes partageant le modèle ajusté sur
   le jeu mutualisé** (§5.8 R6). Sur `PANEL-F` à l'étape `M` : deux groupes, `(v, Y) → {FR}` et
   `(v, Q) → {DE}`, `IT` n'étant imputable à aucune étape.

Sur `TS` : `F = {Q, Y, M}` ; sous `False` → `['M']` ; sous `'covariates_only'`/`True` →
`['Q', 'M']` (la fréquence `Y`, la plus basse, n'est pas une étape : rien n'est à imputer à `Y`).
Étape `Q` : variables `{a1, a2}`. Étape `M` : variables `{q1, a1, a2}`.

Sur un panel, la progression est calculée **par groupe d'entités partageant la même fréquence
cible** ; `target_frequency` en dict autorise des cibles différentes par entité (validation B16 :
dict incomplet → `ValueError` nommant les entités manquantes).

### 5.3 — Composition de `y_train` : le filtre d'origine

`y_train` d'une variable `v` à l'étape `f` est composé des cellules de la **colonne `v`**,
**bloc d'entité par bloc d'entité** (§5.8 : chaque entité contribue à sa propre fréquence
`f_block(e)`), restreintes par la fenêtre `'training'` lue **à la fréquence du bloc** (§7),
puis filtrées par leur origine :

```python
ELIGIBLE_ORIGINS = {
    False:             {'observed'},
    'covariates_only': {'observed', 'interpolated'},
    True:              {'observed', 'interpolated', 'model'},
}
```

Trois points d'implémentation impératifs :

1. **Le filtre porte sur `origin_store`, pas sur la matrice de provenance.** L'origine est une
   propriété de la **cellule**, la provenance une propriété de l'**étape** qui l'a produite
   (§6.3), propagée telle quelle à toutes ses cellules : les deux registres n'ont ni la même
   granularité ni la même durée de vie, et le store est disponible pendant l'étape, avant même
   que la provenance ne soit écrite. Lire la provenance ici, c'est risquer de faire de
   `'covariates_only'` un synonyme silencieux de `True` — c'est le piège principal de ce
   paragraphe.
2. **Chaque ligne de `y_train` porte la fréquence à laquelle elle a été produite**, lue dans
   `imputed_freq_store` ; le diviseur d'échelle est **par ligne** (§5.4).
3. Sous `False`, `y_train` d'une variable annuelle sur `TS` contient exactement 3 lignes : la
   garde `min_cv_train_size` et les gardes de taille de l'estimateur doivent être documentées
   comme le prix de la modalité, et le repli interpolation reste le filet. **Sur un panel, la
   mutualisation du §5.8 élargit ce compte** — 51 lignes au lieu de 3 sur le jeu `PANEL-F` — sans
   changer le filtre d'origine, qui s'applique **à l'identique dans chaque bloc**.
4. **Le filtre d'origine et la mutualisation sont orthogonaux** : `ELIGIBLE_ORIGINS` décide
   *quelles cellules* d'une entité sont éligibles, le §5.8 décide *quelles entités* contribuent
   et *à quelle fréquence*. Les deux se composent sans se connaître ; c'est pourquoi
   `TrainingSetBuilder` (§12.2) reçoit `ELIGIBLE_ORIGINS` en paramètre plutôt que de lire
   `impute_intermediate_frequencies`.

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

**Généralisation à l'axe des entités (§5.8)** : la fréquence d'une ligne de `y_train` n'est pas
une propriété de la **colonne** mais du couple (**ligne**, **entité**). Elle a deux sources, et
une seule règle les unifie :

| Type de ligne | Fréquence de la ligne `f_row` | Lue dans |
|---|---|---|
| cellule **observée** ou **agrégée exactement** de l'entité `e` | `f_block(e)` = la plus basse de `f_var(e, v)` et `f` | `detected_frequencies_` |
| cellule d'origine `'interpolated'` ou `'model'` produite à une étape antérieure | la fréquence de production de la cellule | `imputed_freq_store` |

Dans les deux cas le diviseur vaut `get_conversion_factor(f, f_row)` et le vecteur de diviseurs
est passé tel quel à `StageScaler.target_divisor(produced_freq=…)`. Le §5.4 (axe des étapes) et
le §5.8 (axe des entités) partagent donc **le même mécanisme** et le même chemin de code : il n'y
a pas deux plomberies d'échelle à écrire.

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

### 5.8 — Mutualisation inter-entités du jeu d'entraînement

> **Le jeu d'entraînement d'une variable `v` à une étape `f` est mutualisé entre toutes les
> entités qui observent `v`, chacune contribuant à la fréquence à laquelle elle l'observe,
> ramenée à l'échelle de l'étape par un diviseur propre à son bloc.**

C'est la réponse au défaut B29 (§1.5). Sur un panel, une colonne annuelle pour une entité,
trimestrielle pour une deuxième et mensuelle pour une troisième porte, chez les deux dernières,
des **vraies valeurs à la fréquence même où la première doit être imputée** : les ignorer, c'est
entraîner sur trois ancres ce qui pouvait l'être sur cinquante et une. **Non paramétrable**
(D17) : le mode « chaque entité son modèle » n'existe pas dans `hfi2`, pas plus que dans `hfi`,
le modèle d'un panel y étant global par construction.

#### Les six règles

**R1 — Périmètre des contributeurs.** Toute entité du panel portant **au moins une observation**
de `v` contribue, **indépendamment** : de sa fréquence pour `v`, du fait que `v` y soit imputable
ou non, et de sa fréquence cible propre (`target_frequency` en dict). Une entité qui n'observe
jamais `v` ne contribue rien — symétrique exact du §4.5 côté covariables.

**R2 — Fréquence de bloc.** `f_block(e)` = la plus **basse** des deux fréquences `f_var(e, v)` et
`f`. Motif (D18) : on n'entraîne **jamais** sur une grille plus fine que la grille de prédiction,
sinon le modèle apprendrait une relation à une échelle et un régime d'agrégation des covariables
que la prédiction ne rencontrera jamais.

**R3 — Lignes d'un bloc**, selon la position de `f_var(e, v)` face à `f` :

| Cas | Lignes retenues | `f_block(e)` | Diviseur de la ligne |
|---|---|---|---|
| `f_var(e, v)` plus **basse** que `f` (entité imputable) | les **ancres** de `v` pour `e`, à sa propre fréquence — comportement d'origine, inchangé | `f_var(e, v)` | `get_conversion_factor(f, f_var(e, v))` |
| `f_var(e, v)` **égale** à `f` | les cellules **observées** de `v` pour `e` sur la grille de l'étape | `f` | `1.0` |
| `f_var(e, v)` plus **fine** que `f` | l'**agrégat exact** de `v` sur chaque période **complète** de `f` (`full_periods_only=True`), par la contrainte résolue pour la colonne (`'sum'` par défaut, `'mean'`, `'last'` — §11.1) ; une période incomplète ne produit **aucune** ligne | `f` | `1.0` |

Dans les trois cas, les lignes sont ensuite filtrées par le masque `'training'` lu **à la
fréquence du bloc** (`get_mask_at_frequency({e: f_block(e)}, kind='training')`) et par
`ELIGIBLE_ORIGINS` (§5.3).

**R4 — Covariables du bloc.** `CovariateMaterializer.materialize` est appelé **une seule fois**
sur la grille d'entraînement mutualisée — l'union des grilles de bloc — avec
`stage_freq = {e: f_block(e)}` et `detected_frequencies` sous leur forme par entité. Le composant
lit déjà une fréquence d'étape par entité et dégrade la voie retenue entité par entité
(`_applicable_way`) : **aucune seconde implémentation, aucun contournement de la règle du §4.6**.
La voie est décidée sur cette grille, enregistrée dans l'étape, et rejouée telle quelle sur la
grille de prédiction.

**R5 — Échelle.** Diviseur **par ligne** pour la cible (§5.4, table de généralisation) et
diviseurs **par bloc** pour les features : `StageScaler.feature_divisors` reçoit `source_freq`
sous forme de liaison par entité `{e: f_block(e)}`, exactement comme il reçoit déjà `pred_freq`.
La règle B25 est appliquée **à l'intérieur de chaque bloc**, sans changement.

**R6 — Un seul ajustement par (étape, variable).** Le jeu ainsi construit ne dépend **pas** du
groupe de fréquence source : à une étape donnée, tous les groupes d'une même variable reçoivent
**le même** `X_train`, le même `y_train` et les mêmes voies de matérialisation. L'estimateur est
donc ajusté **une fois par (étape, variable)**, et le modèle ajusté est **partagé** par les
étapes du plan qui ne diffèrent que par leur `source_frequency` et leurs entités. C'est
exactement la mémoïsation autorisée par le §5.7 — « un modèle dont le jeu d'entraînement et les
voies de matérialisation n'ont pas changé » — et **non** un retour de `cascade_refitting=False`
(D2) : un modèle n'est jamais réutilisé d'une **étape** à l'autre.

#### Ce que la mutualisation ne change pas

- **La prédiction** reste par groupe : seules les entités où `v` est imputable à `f` reçoivent
  des valeurs. Sur `PANEL-F` à l'étape `M`, `IT` alimente l'entraînement mais **aucune** de ses
  cellules n'est réécrite ; elles restent `ORIGINAL`.
- **Le recalage** reste par groupe et par entité : les prédictions de `FR` somment à ses totaux
  **annuels**, celles de `DE` à ses totaux **trimestriels** (§11.1). C'est la raison pour
  laquelle les étapes restent distinctes alors même que le modèle est partagé.
- **La provenance** reste une propriété de l'étape (§6.3). Les souillures se calculent sur le jeu
  mutualisé : une entité qui contribue des cellules observées ne dégrade rien ; une entité qui
  contribue des cellules `'interpolated'` ou `'model'` (axe 2) dégrade `target_taint`, donc la
  provenance de **toutes** les cellules produites par l'étape, y compris pour les autres entités.
  À énoncer dans la docstring de la classe.
- **L'invariant central** (§3) et sa mesure **par entité** sont inchangés : les blocs ajoutent des
  **lignes**, jamais des colonnes, et la voie de matérialisation reste unique par (étape,
  variable, covariable).
- **Une série temporelle** est le cas dégénéré à une entité : le jeu mutualisé y est identique au
  jeu d'origine, et tous les exemples chiffrés des §4.7, §5.4 et §5.5 restent vrais au chiffre
  près (invariant I16).

#### Exemple chiffré sur `PANEL-F` (§2.5)

Variable `v`, `covariate_strategy='interpolate'`, `scale_features='constant'`.

**Étape `M`** (la seule étape sous `impute_intermediate_frequencies=False`) — 51 lignes
d'entraînement, un seul ajustement, deux étapes de plan :

| Bloc | `f_block` | Lignes | Valeurs brutes | Diviseur | Cible mise à l'échelle |
|---|---|---|---|---|---|
| `FR` | `Y` | 3 | 120 · 132 · 150 | 12 | **10.0 · 11.0 · 12.5** |
| `DE` | `Q` | 12 | 28 · 30 · 31 · 31 · 31 · 33 · 34 · 34 · 36 · 37 · 38 · 39 | 3 | 9.333 · 10.0 · 10.333 · 10.333 · 10.333 · 11.0 · 11.333 · 11.333 · 12.0 · 12.333 · 12.667 · 13.0 |
| `IT` | `M` | 36 | 10.0 (×12) · 11.0 (×12) · 12.5 (×12) | 1 | **10.0 · 11.0 · 12.5** |

Les trois blocs se superposent à la même échelle mensuelle — c'est le contrôle de bon sens du
jeu : `FR` annuel, `DE` trimestriel et `IT` mensuel décrivent la même trajectoire, et le modèle
apprend sur 51 lignes cohérentes au lieu de 3.

Deux étapes de plan en découlent, **partageant le modèle ajusté** (R6) :

| Étape de plan | `source_frequency` | `entities` | Recalage |
|---|---|---|---|
| `M` / `v` / `Y` | `Y` | `(FR,)` | totaux **annuels** de `FR` |
| `M` / `v` / `Q` | `Q` | `(DE,)` | totaux **trimestriels** de `DE` |

**Étape `Q`** (existe sous `impute_intermediate_frequencies='covariates_only'` ou `True`) —
27 lignes, une seule étape de plan (`(FR,)`, `source_frequency='Y'`), `DE` et `IT` n'y étant pas
imputables :

| Bloc | `f_block` | Lignes | Diviseur | Cible mise à l'échelle |
|---|---|---|---|---|
| `FR` | `Y` | 3 | 4 | **30.0 · 33.0 · 37.5** |
| `DE` | `Q` | 12 | 1 | 28 · 30 · 31 · 31 · 31 · 33 · 34 · 34 · 36 · 37 · 38 · 39 |
| `IT` | `Q` | 12 (agrégats de 3 mois) | 1 | **30.0 (×3) · 33.0 (×3) · 37.5 (×4)** |

À comparer aux **9 lignes de trois échelles différentes** que `hfi` produit sur le même jeu
(§1.5).

#### Le biais assumé (D17)

Mutualiser suppose les **niveaux comparables** entre entités : un pays dix fois plus grand tire
la cible, et `scale_features` ne corrige que l'échelle de **fréquence**, jamais celle d'entité.
L'arbitrage est tranché — **le gain de volume l'emporte sur le risque de biais** — pour trois
raisons : le modèle d'un panel est déjà global dans `hfi` (les entités de même fréquence y sont
déjà mutualisées, sans que personne n'ait jugé utile de le paramétrer) ; le comportement actuel
mutualise **déjà**, mais faux (§1.5) ; et un utilisateur qui veut un modèle par entité l'obtient
sans paramètre, en ajustant un imputeur par entité. À documenter dans la docstring de la classe,
avec cette échappatoire nommée explicitement.

---

## 6 — Provenance

### 6.1 — L'énumération

```python
class ProvenanceType(str, Enum):
    """Enumeration of value provenance types."""
    # --- Cellules non produites par un modèle ---
    ORIGINAL      = 'original'        # présente dans le jeu d'entrée
    AGGREGATED    = 'aggregated'      # agrégation exacte de vraies valeurs plus fines
    DISAGGREGATED = 'disaggregated'   # HÉRITAGE de `hfi`, JAMAIS émis par `hfi2` (§6.4)
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
| cellule d'une période **recalée** pour sommer au total observé (§11) | **inchangée** : celle que la valeur portait avant recalage (`MODEL_*` ou `INTERPOLATED`) | inchangée : `'model'` si la prédiction venait d'un modèle, `'interpolated'` si elle venait d'une interpolation |
| **date-ancre** ré-exprimée à la fréquence d'étape, que le recalage ait eu lieu ou non | celle de la valeur qui y est **écrite** (`MODEL_*` ou `INTERPOLATED`) — jamais `ORIGINAL`, la cellule ne portant plus l'observation | idem |
| cellule produite par interpolation (stratégie `'interpolate'`, `covariate_fallback`, ou **repli d'échec** d'un modèle) | `INTERPOLATED` | `'interpolated'` |

Deux règles :

- **invariance de provenance (D16)** : la provenance d'une cellule est **indépendante du
  recalage** — de son application comme de sa réussite. La ligne d'ancre n'y échappe pas : elle
  porte la provenance de la valeur qui l'occupe désormais, sous `'sum'` comme sous `None`.
- **D6, tranché** : les cellules du **repli** (variable dont le modèle a échoué) portent
  `INTERPOLATED`, et non un `MODEL_*`. C'est plus exact et cela rend le repli visible dans les
  statistiques de provenance. Une étape en repli est marquée `is_fallback=True` dans le plan.

**`DISAGGREGATED` n'est donc jamais émis par `hfi2` (D16).** Recaler une cellule, c'est déplacer
une valeur, pas la produire : la contrainte d'agrégation laisse la provenance intacte, exactement
comme la mise à l'échelle du `StageScaler`. Le libellé était de surcroît **ambigu** — il
remplaçait la marque que la cellule portait, si bien qu'il ne disait ni si l'identité additive
était respectée, ni si la valeur venait d'un modèle ou d'une interpolation. Il **reste dans
l'énumération partagée** tant que `hfi` existe, avec une docstring disant les deux choses : son
ambiguïté, et le fait que `hfi2` ne l'émet jamais. `AggregationConstraint` ne marque aucune
provenance : ses deux masques — cellules recalées, lignes d'ancre — sont des masques de
**diagnostic**. Le filtre de `y_train` (§5.3) et le calcul des souillures (§6.2) lisent
`origin_store`, **jamais** la matrice de provenance.

### 6.5 — Exemple cellule par cellule

Jeu `TS`, `covariate_strategy='interpolate'`, `impute_intermediate_frequencies=False`,
`aggregation_constraint='sum'`, imputation de `a1` (2021 = 120) à l'étape `M`.
Le modèle de `a1` a `feature_cols = [m1, q1, a2]` ; `q1` et `a2` sont plus basses que `M`, donc
interpolées → `covariate_taint = 'interpolated'` ; `y_train` = 3 ancres → `target_taint = 'none'`
→ provenance émise **`MODEL_ON_INTERPOLATED`**.

| Date | Prédiction brute | Après recalage (somme 2021 = 120) | Provenance | Origine |
|---|---|---|---|---|
| 2021-01-31 | 9.0 | 9.6 | `MODEL_ON_INTERPOLATED` | `'model'` |
| 2021-02-28 | 9.5 | 10.13 | `MODEL_ON_INTERPOLATED` | `'model'` |
| … | … | … | `MODEL_ON_INTERPOLATED` | `'model'` |
| 2021-12-31 (**ancre**) | 10.5 | 11.2 | `MODEL_ON_INTERPOLATED` | `'model'` |
| **somme 2021** | 112.5 | **120.0** | — | — |

La colonne « Provenance » est **la même sous `aggregation_constraint=None`** — c'est tout le sens
de l'invariance : seules les valeurs changent, les cellules gardant alors leur prédiction brute,
ligne d'ancre comprise (2021-12-31 vaut 10.5, et non 120). La somme 2021 ne fait alors plus 120,
et la valeur observée n'est récupérable que par `inverse_transform` ou par le masque `ORIGINAL`
du niveau source (§11.2).

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
- **Masque lu à la fréquence du bloc** : le jeu d'entraînement mutualisé (§5.8) réunit des blocs
  de fréquences différentes ; chaque bloc lit le masque `'training'` **à sa propre fréquence**,
  via `get_mask_at_frequency({entité: f_block(entité)}, kind='training')`, qui accepte déjà une
  fréquence par entité. Jamais un masque de stage appliqué à une grille de bloc.
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
| `False` | **aucun diviseur**, ni sur les features couvertes, ni sur `y` lorsque c'est la modalité résolue pour la colonne imputée (D15, rupture avec le `False` de `hfi`) | variables déjà comparables : indices, taux, ratios |
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
   fine que l'étape et `f_var` sinon). Sur un jeu d'entraînement **mutualisé** (§5.8), la règle
   s'applique **bloc par bloc** : `feature_divisors` reçoit `source_freq` sous forme de liaison
   par entité `{e: f_block(e)}` — la même forme que `pred_freq` accepte déjà — et retourne alors
   nécessairement une `DataFrame` de diviseurs par ligne ;
2. le diviseur de `y` (scalaire d'étape, ou `pd.Series` par ligne dès que `y_train` mêle
   plusieurs fréquences de production — de production **d'étape** (§5.4) comme de **bloc**
   (§5.8)) ;
3. le report d'échelle des prédictions (`fit_scale_factor` : le facteur **cuit dans le modèle**,
   qui ne bouge plus une fois l'étape ajustée).

La modalité de `y` est celle de la **colonne imputée**, lue **exactement comme celle de
n'importe quelle autre colonne** : `False` sur cette colonne laisse la cible à son échelle
d'origine, le modèle prédit alors dans l'unité de la variable et le facteur cuit vaut `1.0` — le
report d'échelle des prédictions devient un `no-op`. La cible n'est plus un cas particulier (D15).
Un mélange légitime « feature en `'calendar'`, `y` en `'constant'` », ou « features divisées, `y`
laissée telle quelle » via la forme dict, doit être couvert par un test dédié (I5).

La colonne imputée n'a plus de paramètre dédié : le composant étant un transformer sklearn
ajusté par `fit(X, y)`, le **nom de `y`** la désigne, et une cible anonyme retombe sur le réglage
global de `scale_features`. Le paramètre `target_column` est **supprimé** — il aurait dupliqué une
information que `y` porte déjà (D15).

L'arithmétique, elle, continue de distinguer `y` des features : une covariable portée à sa propre
fréquence n'est jamais ré-agrégée et divise par `1.0` (règle B25), tandis que la cible, **produite
sur la grille d'étape**, y porte `pred_freq` et divise par le décompte de sous-périodes
`pred_freq` d'une période de `f_var`. Les deux règles ne coïncident que si l'on déclare `pred_freq`
comme fréquence de la colonne — et leurs replis diffèrent : fréquence incomparable → diviseur par
défaut pour une covariable, **erreur** pour la cible. Les méthodes `feature_divisors` et
`target_divisor` restent donc distinctes.

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
ConstraintKind = Literal['sum', 'mean', 'last']
aggregation_constraint: Union[Optional[ConstraintKind],
                              Dict[str, Optional[ConstraintKind]]] = 'sum'
```

Remplace `enforce_period_totals: bool` (D8). `'sum'` ≡ `enforce_period_totals=True`, `None` ≡
`False`. Le paramètre est d'emblée extensible, sans rupture d'API :

- trois contraintes scalaires — `'sum'` pour une variable additive, `'mean'` pour un taux,
  `'last'` pour un stock — plus `None`, qui n'impose rien. `additive_transformer` reste le moyen
  de rendre additive une colonne qui ne l'est pas ; `'mean'` et `'last'` servent les colonnes
  qu'aucune transformation ne rend additives ;
- une **forme dictionnaire** `Dict[str, Optional[ConstraintKind]]` par colonne, avec
  `'__default__'`, sur le modèle de `estimator` et `scale_features`.

**Validation à `__init__`** : seules ces quatre valeurs, et les dictionnaires qui les associent à
un nom de colonne, sont acceptés. Toute autre valeur lève un `ValueError` énonçant les formes
admises. La validation et la résolution sont portées par les fonctions de module
`validate_aggregation_constraint` et `resolve_aggregation_constraint`
(`aggregation_constraint.py`), **partagées** avec `CovariateMaterializer` : les deux composants
portent le paramètre, une seule implémentation les empêche de diverger.

Sémantique, reprise de `hfi:_rescale_to_period_totals` : les sous-périodes prédites d'une période
sont multipliées par `total observé / agrégat prédit`, de sorte que la colonne porte une véritable
**désagrégation** de l'observation plutôt qu'une prédiction libre. L'agrégat est celui que nomme
la contrainte — somme, moyenne, ou dernière sous-période. **Les trois contraintes passent par ce
ratio unique**, si bien que les gardes ci-dessous et le masque des cellules recalées leur sont
rigoureusement identiques. Gardes conservées :

| Cas | Comportement |
|---|---|
| période **partiellement** prédite (au moins une sous-période NaN) | non recalée, prédictions brutes conservées |
| période sans aucune observation (fin de série retardée) | non recalée |
| agrégat prédit nul, total observé non nul | non recalée (ratio indéfini) |
| agrégat prédit de **signe opposé** au total observé | **recalée** — la contrainte prime — mais toutes les sous-périodes changent de signe : un `UserWarning` agrégé est émis |

Le masque des cellules effectivement recalées est un masque de **diagnostic** : recalées ou non,
toutes les cellules gardent la provenance qu'elles portaient avant le recalage — `MODEL_*` ou
`INTERPOLATED` (§6.4, invariance de provenance D16).

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
  sous-périodes reconstitue exactement le total observé ;
- sous `aggregation_constraint=None`, le total observé **est écrasé** par une prédiction libre.
  Il reste récupérable de deux manières, toutes deux à mentionner dans la docstring : par
  `inverse_transform`, et par le masque `ORIGINAL` du niveau de fréquence source dans la sortie
  multi-fréquences (`keep_lower_frequencies=True`) ;
- la ligne d'ancre porte la provenance de la valeur qui l'occupe désormais — le `MODEL_*` de
  l'étape, ou `INTERPOLATED` —, jamais `ORIGINAL` et jamais une marque propre : le recalage ne
  change aucune provenance (D16). `AggregationConstraint.anchor_cells_mask` **localise** les
  totaux observés écrasés — ce dont `inverse_transform` et les diagnostics ont besoin —, il ne
  les qualifie pas, et ne dépend ni de `aggregation_constraint` ni de la réussite du recalage.

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
| modèles ajustés (**un par (étape, variable)**, partagé par les groupes de fréquence source, §5.8 R6), `feature_cols`, **voie de matérialisation par covariable** (§4.6), composition du jeu d'entraînement mutualisé (`training_blocks`), facteurs et modalités d'échelle | frames d'étape, valeurs interpolées, prédictions, provenance du transform |
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
| `StageScaler` | `stage_scaler.py` | diviseurs `'constant'`/`'calendar'`, scalaires et par ligne ; application et inversion de l'échelle ; report d'échelle des prédictions ; `source_freq` en **liaison par entité** pour les jeux mutualisés (§5.8 R5) |
| `VariableOrderer` | `variable_orderer.py` | ordres `'frequency'` et `'cv'` (avec `cv`, `cv_scoring`, `min_cv_train_size`), tie-break alphabétique |
| `TrainingSetBuilder` | `training_set_builder.py` | jeu d'entraînement **mutualisé** d'une variable à une étape (§5.8) : blocs par entité, fréquence de bloc, lignes éligibles (`ELIGIBLE_ORIGINS`), appel **unique** à `CovariateMaterializer.materialize`, fréquence de production par ligne, diviseurs par bloc |
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
  5c. pour chaque variable v (COLONNE, une seule fois par étape) :
      - JEU D'ENTRAÎNEMENT MUTUALISÉ (TrainingSetBuilder, §5.8) : un bloc par entité observant v,
        fréquence de bloc f_block(e) = la plus basse de f_var(e, v) et f (R2), lignes du bloc
        selon R3 (ancres / observations / agrégats exacts de périodes complètes), masque
        'training' lu A LA FREQUENCE DU BLOC, filtre ELIGIBLE_ORIGINS (§5.3) ; grille
        d'entraînement = union des grilles de bloc ; fréquence de production PAR LIGNE
      - grille de prédiction : masque 'imputation'
      - sélection des feature_cols (non-vides sur LES DEUX fenêtres, covariate_eligibility)
      - matérialisation des covariables sur LES DEUX grilles par la MÊME voie (§4.6), l'appel
        d'entraînement portant stage_freq = {e: f_block(e)} ; enregistrement de
        materialization[col]
      - calcul des souillures covariate_taint / target_taint (§6.2), sur le jeu MUTUALISÉ
      - mise à l'échelle (StageScaler : diviseur cible PAR LIGNE, diviseurs de features PAR BLOC)
      - ajustement de l'estimateur — UN SEUL pour (étape, variable), quel que soit le nombre de
        groupes de fréquence source (§5.8 R6) ; en cas d'échec -> repli interpolation (méthode de
        v), étapes marquées is_fallback=True, cellules marquées INTERPOLATED
      - pour chaque GROUPE DE FRÉQUENCE SOURCE (entités partageant f_var(e, v), et pour
        lesquelles v est imputable à f), en PARTAGEANT le modèle ajusté ci-dessus :
          . prédiction sur TOUTE la période, ancres comprises (§11.2)
          . recalage aux totaux DE CE GROUPE (AggregationConstraint) -> masque des cellules
            recalées
          . écriture des valeurs, marquage de provenance (§6.3, IDENTIQUE pour les cellules
            recalées et non recalées, lignes d'ancre comprises : le recalage ne change aucune
            provenance)
          . mise à jour de imputed_store / imputed_freq_store / origin_store
            (y compris en repli : "le repli matérialise")
          . gel de l'ImputationStep dans le plan (mêmes model, feature_cols, materialization et
            souillures ; source_frequency, entities et recalage propres au groupe)
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
        aggregation_constraint: Union[Optional[ConstraintKind],
                                      Dict[str, Optional[ConstraintKind]]] = 'sum',

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
| `aggregation_constraint` | `'sum'`, `'mean'`, `'last'`, `None`, ou dict de ces valeurs (clé `'__default__'` admise, clés vérifiées au `fit`) ; message listant les formes admises (§11.1) |
| booléens (`keep_lower_frequencies`, `restore_original_values`, `verbose`) | validation groupée, comme dans `hfi` |

Combinaisons **inertes** documentées mais **non signalées** (D9) : cf. §5.6.
`training_coverage_threshold` sans `training_scope` est inerte, à documenter.

### 13.2 — Attributs ajustés

| Attribut | Contenu |
|---|---|
| `effective_target_frequency_` | fréquence cible normalisée (scalaire ou dict par entité) |
| `detected_frequencies_` | fréquence détectée au fit : `{colonne: fréquence}` sur une série temporelle, `{(entité, colonne): fréquence}` sur un panel — **les entités peuvent diverger pour une même colonne** (§2.1, §2.5) |
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
| **D8** | `aggregation_constraint` remplace `enforce_period_totals` : `'sum'` (défaut), `'mean'`, `'last'`, `None`, ou dict par colonne avec `'__default__'`. Les quatre formes sont **implémentées**, la validation les accepte toutes | §11.1 |
| **D9** | paramètres inertes : **silence + docstring**, jamais d'avertissement | §5.6, §8.1 |
| **D10** | dictionnaires par feature indexés par **nom de colonne**, jamais par `(entité, colonne)` | §9.1, §10 |
| **D11** | `transform` face à des fréquences divergentes : **avertissement + poursuite** avec les fréquences du fit ; **`ValueError`** si une colonne du fit manque | §12.1 |

### 14.3 — Décision dérivée, introduite par cette révision

| Code | Décision | Motif |
|---|---|---|
| **D12** | introduction de `CellOrigin` / `origin_store` : le filtre de `y_train` et le calcul des souillures lisent l'**origine** des cellules, jamais la matrice de provenance | l'origine est une propriété de la **cellule**, la provenance une propriété de l'**étape** ; sans ce registre, `'covariates_only'` serait un synonyme silencieux de `True` (§5.3) |
| **D13** | précédence de matérialisation à quatre rangs, dont le **report d'étape** (rang 3) | c'est ce qui donne un effet observable à `'covariates_only'` et ce qui corrige la première cause de B28 (§4.4) |
| **D14** | l'invariant NaN se formule en **inclusion d'ensembles de dates renseignées**, pas en comparaison de taux bruts | sous `'tolerate_nan'`, les deux grilles n'ont pas le même pas (§4.7) |
| **D15** | `scale_features=False` **dispense aussi `y`** : la cible est une colonne comme une autre, et sa modalité est celle résolue pour la colonne imputée, `False` compris | rupture assumée avec le `False` de `hfi` : il existe de bonnes raisons de ne pas mettre une variable à l'échelle (indice, taux), et l'utilisateur est mieux placé que le composant pour le décider. Corollaire : `target_column` est **supprimé**, le nom de `y` désignant la colonne imputée au `fit` (§9.2) |
| **D16** | la contrainte d'agrégation **ne change aucune provenance** : cellules recalées et lignes d'ancre gardent le `MODEL_*` ou l'`INTERPOLATED` de la valeur écrite ; `DISAGGREGATED` n'est **jamais émis** par `hfi2` et ne survit dans l'énumération partagée que pour `hfi` | recaler, c'est déplacer une valeur, pas la produire — au même titre que la mise à l'échelle du `StageScaler`. `DISAGGREGATED` **remplaçait** la marque de la cellule et effaçait l'information utile (modèle ? interpolation ? souillure ?) au profit d'une information de position que le masque des cellules recalées et `anchor_cells_mask` portent déjà, comme diagnostics (§6.4, §11.2) |
| **D17** | **mutualisation inter-entités du jeu d'entraînement**, **non paramétrable** : toute entité observant la colonne entraîne le modèle, à sa propre fréquence, ramenée à l'échelle de l'étape par un diviseur de bloc (§5.8) | le comportement actuel mutualise **déjà**, mais avec un diviseur faux et une fraction des lignes (B29, §1.5) ; le modèle d'un panel est global par construction, dans `hfi` comme dans `hfi2` ; l'auteur tranche l'arbitrage volume / biais de niveau **en faveur du volume**, l'échappatoire (un imputeur par entité) restant disponible sans paramètre |
| **D18** | `f_block(e)` = la plus **basse** de `f_var(e, v)` et `f_stage` : on n'entraîne jamais sur une grille plus fine que la grille de prédiction ; une entité plus fine contribue par **agrégation exacte** sur les périodes **complètes** | à une grille plus fine, les covariables porteraient un régime d'agrégation que la prédiction ne rencontre jamais, et la cible une échelle fractionnaire ; l'agrégation exacte d'une colonne additive n'est pas une approximation (elle vaut `'observed'`, §6.2) |
| **D19** | **un seul ajustement par (étape, variable)** : le modèle est partagé par les étapes du plan qui ne diffèrent que par `source_frequency` et `entities` | le jeu mutualisé ne dépend pas du groupe (§5.8 R6) : deux ajustements y seraient redondants et, sous un estimateur stochastique, divergents. C'est la mémoïsation explicitement autorisée par le §5.7, et non un retour de D2 : un modèle n'est jamais réutilisé d'une **étape** à l'autre |

---

## 15 — Prérequis et travaux annexes

| Code | Travail | Statut |
|---|---|---|
| **P1** | **B26 / prompt 21** : le contrôle `full_periods_only` de `utils/frequency/converter.py` utilise un décompte **constant** et jette février ; toute covariable journalière disparaît silencieusement du jeu d'entraînement. `hfi2` s'appuyant sur les mêmes agrégations, c'est un **prérequis dur**, indépendant de `hfi2`, à exécuter **en premier** | à faire (L1) |
| **P2** | extension `FrequencyConverter.interpolate_to_higher_frequency(anchor_fraction=...)` (§10.2) : capacité utilitaire testable isolément, prérequis de la stratégie `'interpolate'` complète | à faire (L1) |
| **P3** | extension `ImputationWindowCalculator` : trois masques + `training_scope` + retour MultiIndex (§7) — prompts 10/13/20 de [ARCH] jamais exécutés, réalisés ici sur `iwc`, **partagé avec `hfi` sans changement de comportement par défaut** | à faire (L3) |
| **P4** | extension `ProvenanceType` + `mark_model_imputed` (§6) — **rupture assumée** pour `hfi` (suppression de `MODEL_ON_MIXED`) | à faire (L2) |
| **P5** | jeu `PANEL` avec feature manquante par entité (§15.1) — **à faire en premier**, les jeux d'exemple servant ensuite à tous les tests et notebooks | à faire (L0) |
| **P6** | jeu `PANEL-F` à fréquences hétérogènes par entité (§2.5) : fixture, valeurs d'or gelées et tests de référence — prérequis des invariants I14 à I16 | à faire (L8b) |

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
  `fit_transform(X)` vs `fit(X).transform(X)` ;
- **déroule le jeu `PANEL-F` (§2.5)** : pour chaque étape, la composition du jeu d'entraînement
  **mutualisé** bloc par bloc (entité, `f_block`, nombre de lignes, diviseur, cible avant et
  après mise à l'échelle), la superposition des trois blocs à la même échelle, le fait qu'`IT`
  entraîne sans jamais être réécrite, et la comparaison avec le comptage de `hfi` (§1.5).

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
| **I5** | échelle train/test | pour chaque étape, moyennes des features de `X_train` et de `X_pred` comparables (tolérance dépendant de la modalité) ; inclut le cas mixte « feature `'calendar'`, `y` `'constant'` » et le cas « features divisées, `y` en `False` » (D15) |
| **I6** | provenance | les cinq familles `MODEL_*` sont émises exactement dans les cas du §6.3 et aucun autre ; `MODEL_ON_IMPUTED*` seulement sous `'model'` ; `*_TARGET`/`*_BOTH` seulement sous `impute_intermediate_frequencies=True` ; les cellules de repli portent `INTERPOLATED` ; `MODEL_ON_MIXED` n'existe plus |
| **I7** | `transform` hors fenêtre de fit | impute au lieu de vider, ne détruit jamais une observation d'entrée, avertit **une fois** sur les lignes inimputables |
| **I8** | aller-retour | `inverse_transform(transform(X))` restitue l'index, les noms (panels multi-niveaux compris) et, sous `restore_original_values=True`, les valeurs d'origine |
| **I9** | conformité sklearn | `clone`, `get_params`/`set_params`, `Pipeline`, `GridSearchCV` sur panel avec `target_frequency` dict ; `NotFittedError` avant `fit` |
| **I10** | indifférence de l'ordre hors `'model'` | sous `'tolerate_nan'` et `'interpolate'`, forcer deux ordres de traitement différents produit des sorties identiques |
| **I11** | unicité de la voie de matérialisation | pour chaque (étape, variable, covariable), `materialization` est identique au fit et au transform, et la nature des valeurs produites l'est aussi (§4.6) |
| **I12** | `'covariates_only'` ≠ `True` | sur un jeu où la cascade change quelque chose, `'covariates_only'` produit des `y_train` sans aucune ligne d'origine `'model'`, et des valeurs finales **différentes** de `True` ; sous `covariate_strategy='interpolate'`, `'covariates_only'` produit les mêmes valeurs finales que `False` (§5.6) |
| **I13** | `impute_intermediate_frequencies` n'est jamais testé comme booléen | test statique/grep : aucun `if self.impute_intermediate_frequencies:` dans le code (`'covariates_only'` est *truthy*) |
| **I14** | mutualisation (§5.8) | sur `PANEL-F`, `y_train` de `v` contient **exactement** les lignes annoncées au §5.8 — 51 à l'étape `M` (3 `FR` + 12 `DE` + 36 `IT`), 27 à l'étape `Q` — et chaque ligne est à l'**échelle de l'étape** : valeurs d'or `10.0 / 11.0 / 12.5` (blocs `FR` et `IT` à `M`), `30.0 / 33.0 / 37.5` (bloc `FR` à `Q`), agrégats `IT` `30 / 33 / 37.5` à `Q`. Aucune ligne ne mêle deux échelles ; une période incomplète ne produit aucune ligne |
| **I15** | indépendance au groupe de fréquence source | à une étape donnée, les groupes `(v, Y)` et `(v, Q)` de `PANEL-F` reçoivent le **même** `X_train`, le **même** `y_train` et les **mêmes** voies de matérialisation, et **partagent le même objet modèle** (`is`) ; leurs recalages restent distincts (totaux annuels de `FR`, trimestriels de `DE`) et `IT` n'est **jamais** réécrite |
| **I16** | non-régression de la série temporelle | sur `TS` (entité unique), le jeu mutualisé est **identique** au jeu d'origine : tous les exemples chiffrés des §4.7, §5.4 et §5.5 restent vrais au chiffre près |

Cas limites à couvrir explicitement, en plus : index non trié, index dupliqué, entité à une seule
observation, variable annuelle à 2 ancres seulement (`y_train` de taille 2 sous
`impute_intermediate_frequencies=False`), période incomplète en début et en fin de série,
fréquence irrégulière détectée, colonne entièrement NaN, `estimator=None`.

Cas limites propres à la mutualisation (§5.8) : entité observant la colonne **plus finement** que
la fréquence cible (contributrice, jamais imputée) ; entité dont la première ou la dernière
période est **incomplète** (aucune ligne produite pour cette période, les autres restant) ;
entité n'observant **jamais** la colonne (aucune contribution, aucune erreur) ; entité dont la
**fréquence cible** diffère de celle du groupe imputé (elle contribue quand même, R1) ; panel où
**toutes** les entités observent la colonne à la fréquence cible (aucune variable imputable, plan
vide) ; panel à **une seule** entité (le jeu mutualisé se réduit au bloc unique).

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
| **L8b** | jeu de référence `PANEL-F` (§2.5) : colonne à fréquence hétérogène par entité, fixtures et valeurs d'or | — | `tests/frequency/test_reference_datasets.py` |
| **L8c** | `stage_scaler.py` : `source_freq` en liaison par entité, diviseurs par bloc (§5.8 R5, §9.2) | L5 | tests unitaires isolés |
| **L8d** | `training_set_builder.py` : jeu d'entraînement mutualisé (§5.8 R1 à R5), `ImputationStep.training_blocks` | L4, L5, L6, L8b, L8c | tests unitaires + I14, I16 |
| **L9** | `high_frequency_imputer2.py` — `__init__`, validations (§13.1), phases 0 à 4, attributs ajustés, fréquences détectées **par (entité, colonne)** | L4–L8d | I9, tests de validation d'arguments |
| **L10** | `high_frequency_imputer2.py` — PHASE 5 : exécution d'étape unique, axe 1 complet, provenance, stores, **un ajustement par (étape, variable)** partagé par les groupes de fréquence source (§5.8 R6) | L9 | I2, I3, I6, I10, I11, I14, I15 |
| **L11** | axe 2 : progression de fréquences, `ELIGIBLE_ORIGINS`, échelle par ligne, report d'étape ; composition avec la mutualisation (fréquence de ligne : bloc **ou** store) | L10 | I12, I13, exemples chiffrés du §5.5 et du §5.8 |
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
