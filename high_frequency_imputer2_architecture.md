# `HighFrequencyImputer2` — spécification d'architecture

> Document rédigé le 2026-08-31 (branche `qb-mixed-frequencies`), à partir de la spécification
> orale du 2026-08-31, de la relecture de `_fit` dans
> `tsforecast/frequency/high_frequency_imputer.py`, et des deux documents de la campagne
> d'annotations : `high_frequency_imputer_annotations_architecture.md` (noté ci-après **[ARCH]**)
> et `high_frequency_imputer_annotations_prompts.md` (prompts 1 à 12 et 22 exécutés).
>
> **Objet** : spécifier `HighFrequencyImputer2` (`tsforecast/frequency/high_frequency_imputer2.py`),
> réécriture from scratch de `HighFrequencyImputer` destinée à la remplacer dans le package.
> Le document sert de base à un second prompt (revue Opus) puis à une liste de prompts
> d'implémentation ; le §11 recense les décisions encore ouvertes à trancher dans ce second
> prompt.
>
> Convention : `hfi:` = `tsforecast/frequency/high_frequency_imputer.py` (implémentation
> actuelle), `iwc:` = `tsforecast/frequency/imputation_window.py`. Localiser par nom de
> symbole, jamais par numéro de ligne.

---

## Table des matières

0. [Résumé et verdict sur la spécification](#0--résumé-et-verdict-sur-la-spécification)
1. [Contexte : pourquoi repartir d'un fichier vierge](#1--contexte--pourquoi-repartir-dun-fichier-vierge)
2. [L'invariant central](#2--linvariant-central)
3. [Les deux axes de la logique d'entraînement](#3--les-deux-axes-de-la-logique-dentraînement)
4. [Fenêtres d'entraînement et d'imputation](#4--fenêtres-dentraînement-et-dimputation)
5. [Ordre d'imputation](#5--ordre-dimputation)
6. [Mise à l'échelle des données](#6--mise-à-léchelle-des-données)
7. [Interpolation et position d'ancrage](#7--interpolation-et-position-dancrage)
8. [Contraintes d'agrégation](#8--contraintes-dagrégation)
9. [Architecture logicielle](#9--architecture-logicielle)
10. [API résultante](#10--api-résultante)
11. [Problèmes, manques et décisions à trancher](#11--problèmes-manques-et-décisions-à-trancher)
12. [Prérequis et travaux annexes](#12--prérequis-et-travaux-annexes)
13. [Stratégie de tests et invariants](#13--stratégie-de-tests-et-invariants)

---

## 0 — Résumé et verdict sur la spécification

**La spécification est cohérente et elle est meilleure que l'existant** : les deux axes
(matérialisation des covariables × imputation des fréquences intermédiaires) recouvrent, en les
rendant orthogonaux et explicites, ce que `cascade_refitting`, `train_on_partial_coverage`,
`covariate_eligibility` et la règle de matérialisation du §3.17 de [ARCH] tentaient d'exprimer
par accumulation de correctifs. Le défaut structurel mesuré (B28 : 0 % de NaN au `fit` contre
jusqu'à 67 % au `predict`) devient **inexprimable par construction** dans les trois stratégies
de covariables, au lieu d'être colmaté par un filtre d'éligibilité.

Trois familles de points restent à trancher avant l'implémentation, toutes recensées au §11 :

1. **Trois phrases de la spécification sont tronquées** (fin du point 1 « En revanche, je
   souhaite . », fin du second paramètre du point 4 « Dans ce cas », point 6 « On pourra
   également cela ») — les deux dernières se résolvent par le contexte, la première non.
2. **Deux capacités de l'implémentation actuelle disparaissent** du nouvel espace de
   paramètres sans avoir été explicitement abandonnées : le mode « un seul fit réutilisé aux
   étapes suivantes » (`cascade_refitting=False`) et la combinaison « étapes intermédiaires
   pour enrichir les covariables mais cible entraînée sur les seules vraies valeurs »
   (le comportement par défaut de `HighFrequencyImputer` aujourd'hui).
3. **Une dizaine de correctifs et garde-fous de [ARCH] restent pertinents** et ne figurent pas
   dans la spécification orale : conformité sklearn (B3, B14–B16, B19, B20), `transform` hors
   fenêtre de fit (B1), panels à n > 2 niveaux d'entité (B4), décompte calendaire de
   `full_periods_only` (B26, **toujours ouvert**, prompt 21 non exécuté), `covariate_eligibility`
   pour les features absentes d'une entité entière, avertissements uniques, etc. Le §9.5 et le
   §11 les reprennent un à un.

**Réponse à la question de conception CV** (KFold paramétrée vs crossval utilisateur) : suivre la
convention sklearn — un paramètre unique `cv` acceptant un entier ou un splitter complet, avec un
défaut raisonnable, plutôt que de multiplier les paramètres scalaires. Détail au §5.3.

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
  `cv_scoring`, `cv_n_splits` (voir toutefois §5.3), `restore_original_values`,
  `enforce_period_totals` (renommage proposé au §8.1), plus les paramètres de la classe de base
  `time_col`, `panel_cols`, `verbose`.
- **Validations d'arguments** : l'intégralité du bloc de validation de `hfi:__init__` est
  reprise (format de `target_frequency`, contrat de `estimator` — y compris la forme dict avec
  clé `'__default__'` —, `additive_transformer` avec `fit_transform`/`inverse_transform`,
  bornes numériques, `Literal`s, validation groupée des booléens, avertissement
  `min_cv_train_size < cv_n_splits`, avertissement unique `estimator=None`). La règle B3 reste
  impérative : **valider sans transformer**, stocker les paramètres tels que reçus, normaliser
  au `fit` dans des attributs `*_`.
- **Phases 0 à 4 de `_fit`** : setup (purge de l'état, colonnes, détection panel, alignement et
  nommage de `y`, détection de fréquences, normalisation/validation de la fréquence cible,
  classification des variables), calcul de la fenêtre d'imputation, transformateur additif,
  liste des fréquences de prédiction, initialisation de la provenance — dans cet ordre, avec le
  tracker initialisé **après** le transformateur additif dans les deux chemins (B8).
- **`transform` rejoue les mêmes transformations et imputations que `fit`** (même plan
  d'étapes), et **`inverse_transform`** restitue les données originales comme aujourd'hui
  (niveau de fréquence source, inversion du transformateur additif, masque `ORIGINAL`,
  `restore_original_values`).
- **`keep_lower_frequencies`** : conservé tel quel, paramètre d'affichage pur (empilage
  multi-fréquences de la sortie), sans influence sur la logique — y compris le correctif B4
  (panels à n > 2 niveaux d'entité, noms d'index préservés).
- **Appui sur les classes existantes de `tsforecast/frequency`** : `FrequencyAligner`,
  `FrequencyConverter`, `ImputationWindowCalculator`, `TargetFrequencyValidator`,
  `ImputationProvenanceTracker`, `ImputationStep` (adapté), `_write_interpolation_fallback`
  (sa logique, portée dans le nouveau composant d'interpolation).

### 1.4 — Ce qui disparaît

| Paramètre `hfi` | Sort dans `hfi2` |
|---|---|
| `cascade_refitting` | remplacé par le couple (`covariate_strategy`, `impute_intermediate_frequencies`) — voir §3.5 pour ce qui est perdu au passage |
| `covariate_eligibility` | **conservé** (voir §3.1.4) — la spécification orale ne le mentionne pas mais le cas « feature absente pour la totalité d'une entité » l'exige |
| `train_on_partial_coverage` | déjà condamné par [ARCH] §3.7 ; remplacé par `training_scope` |
| `train_on_partial_fit_order` | renommé `fit_predict_order`, mêmes modalités `'frequency'`/`'cv'`, champ d'application restreint (§5.1) |
| `scale_features: bool` | remplacé par la forme à trois modalités + dict (§6) |

---

## 2 — L'invariant central

Toute la conception de `HighFrequencyImputer2` découle d'une règle unique, qui est la leçon de
B28 :

> **Un modèle ne voit jamais, à la prédiction, un motif de disponibilité de features plus
> dégradé qu'à l'entraînement.** Pour chaque étape du plan, le taux de NaN par colonne de
> `X_pred` est inférieur ou égal au taux de NaN de la même colonne dans `X_train`, aux lignes
> d'entités structurellement absentes près (§3.1.4).

Cet invariant est **testable mécaniquement** (estimateur espion, test
`test_nan_rate_at_predict_never_exceeds_fit`) et chaque stratégie de covariables (§3.1) doit le
garantir **par construction**, pas par filtrage a posteriori :

- `'tolerate_nan'` : les covariables ne sont jamais matérialisées au-delà de leurs observations ;
  le motif de NaN est celui des données, identique au fit et au predict par définition — mais il
  doit être **construit par la même routine** des deux côtés ;
- `'interpolate'` : toutes les covariables sont matérialisées partout (sauf entité intégralement
  vide) ; taux de NaN ≈ 0 des deux côtés ;
- `'model'` : une covariable non encore matérialisée au moment où une variable est imputée est
  traitée par l'approche secondaire (`'interpolate'` ou `'tolerate_nan'`), **au fit comme au
  predict** — jamais « pleine au fit, vide au predict ».

Corollaire structurel : la préparation des features (`X_train` et `X_pred`) passe par **une seule
méthode partagée** (le `CovariateMaterializer`, §9.2), appelée par le fit et par le transform.
Les défauts B7/B27 (« deux blocs censés être identiques qui divergent ») deviennent impossibles :
il n'existe qu'un bloc.

---

## 3 — Les deux axes de la logique d'entraînement

### 3.1 — Axe 1 : la stratégie de matérialisation des covariables

```python
covariate_strategy: Literal['tolerate_nan', 'interpolate', 'model'] = 'interpolate'
```

Décide comment une covariable **non observée à la fréquence de la grille courante** (grille
d'entraînement ou de prédiction) est mise à disposition du modèle. Une covariable de fréquence
**supérieure ou égale** à la grille est toujours disponible (agrégation exacte ou identité) :
la stratégie ne concerne que les covariables de fréquence **strictement inférieure**.

#### 3.1.1 — `'tolerate_nan'`

Aucune matérialisation. Une covariable plus basse fréquence que la grille porte ses valeurs aux
seules dates-ancres (à l'échelle de sa propre fréquence) et NaN partout ailleurs. **Contrat** :
l'estimateur doit tolérer les NaN (`HistGradientBoostingRegressor`, `LGBMRegressor`, ou un
`Pipeline` avec `SimpleImputer`) — la docstring de `estimator` l'énonce déjà, elle devient ici
un prérequis dur de cette modalité. L'ordre d'imputation est **indifférent** (aucune covariable
ne dépend d'une imputation antérieure) : `fit_predict_order` est ignoré (§5.1).

Diviseur d'échelle : règle B25 conservée — `1.0` pour une colonne jamais ré-agrégée (ses ancres
gardent l'échelle `f_cov`), `get_conversion_factor(f_stage, f_var)` sinon, avec
`f_stage = pred_freq` si `f_cov` est plus fine que l'étape et `f_cov` sinon.

#### 3.1.2 — `'interpolate'` (défaut proposé)

Toute covariable plus basse fréquence que la grille est **interpolée** sur la grille à partir de
ses valeurs observées (les dates-ancres servant de points de référence), puis **recalée pour
préserver les totaux de période** quand la contrainte d'agrégation est active (§8) — c'est le
pendant exact, sur la covariable, de ce que la classe produit sur la variable imputée, et c'est
la logique déjà unifiée dans `hfi:_write_interpolation_fallback` (B27, prompt 22).

- Méthode d'interpolation : `interpolation_method` (§7.1), globale ou par feature (dict).
- Position d'ancrage de la valeur dans sa période : `interpolation_anchor` (§7.2).
- Supprime **tous** les NaN, sauf les features absentes pour la **totalité** d'une entité
  (§3.1.4) et les bords de série hors `limit_direction`.
- L'ordre d'imputation est **indifférent** : l'interpolation est déterministe à partir des
  seules observations, aucune dépendance aux imputations précédentes. `fit_predict_order`
  est ignoré (§5.1).
- Provenance des cellules de covariables interpolées : `INTERPOLATED` (§3.3), qui appartient aux
  provenances « fiables » — un modèle entraîné sur vraies valeurs + valeurs interpolées reste
  `MODEL_ON_TRUE`.
- Cette même méthode d'interpolation (globale ou par feature) sert de **repli** partout où une
  imputation par modèle échoue, dans toutes les stratégies (le repli actuel
  `INTERPOLATE_FALLBACK`).

**Avertissement à documenter (regard vers l'aval)** : l'interpolation linéaire entre deux ancres
utilise l'ancre **future** — dans un usage pseudo-temps réel, c'est une information de futur. Ce
n'est pas un défaut pour de l'imputation d'historique, mais la docstring doit le dire, et
`imputation_scope='extended_forward'` reste le mécanisme dédié aux fins de série.

#### 3.1.3 — `'model'`

Les covariables manquantes sont imputées par le même mécanisme fit/predict que les variables
cibles, **dans l'ordre défini par `fit_predict_order`** (§5). C'est le seul mode où l'ordre a un
impact réel. Règle de matérialisation ([ARCH] §3.17, reprise telle quelle) :

> une covariable est utilisable pour imputer `v` à l'étape `f` si et seulement si elle est
> matérialisée sur la grille de `f` au moment où `v` est imputée : fréquence détectée ≥ `f`,
> **ou** imputée avec succès plus tôt dans cette même étape (un repli par interpolation
> **matérialise aussi**, contrairement à `hfi` — troisième cause de B28).

Pour les covariables **pas encore** matérialisées à ce moment (celles qui viennent après dans
l'ordre, ou de même fréquence que `v`), une **approche secondaire** s'applique :

```python
covariate_fallback: Literal['interpolate', 'tolerate_nan'] = 'interpolate'
```

- `'interpolate'` : elles sont portées sur la grille par l'interpolation du §3.1.2 — le modèle
  de `v` voit alors une version interpolée de ces covariables, remplacée aux étapes suivantes
  par leur version imputée au fil de la cascade ; **le motif de disponibilité reste identique au
  fit et au predict** parce que la même règle s'applique des deux côtés ;
- `'tolerate_nan'` : elles restent NaN hors ancres — même contrat d'estimateur qu'en §3.1.1.

Le miroir des imputations (`imputed_store` de `hfi`) est conservé avec les correctifs B5
(`predictions.combine_first(existing)`) et « le repli matérialise » ; il est doublé du registre
de fréquences (`imputed_freq_store`) nécessaire aux diviseurs par ligne (§6.3).

#### 3.1.4 — Feature absente pour la totalité d'une entité — `covariate_eligibility` conservé

Aucune stratégie ne peut fabriquer des valeurs pour une entité où la feature n'a **aucune**
observation (le cas que la nouvelle feature du notebook 3 introduira, §12.1). Le paramètre
`covariate_eligibility: Literal['any_entity', 'all_entities'] = 'any_entity'` de `hfi` est donc
conservé avec sa sémantique actuelle, mais **recadré sur ce seul cas** :

- `'any_entity'` (défaut) : la colonne est retenue dès qu'au moins une entité l'observe ; les
  lignes des entités vides restent NaN et relèvent du contrat NaN de l'estimateur — c'est
  l'**unique** source de NaN résiduels sous `covariate_strategy='interpolate'` ;
- `'all_entities'` : la colonne est écartée de `feature_cols` si une entité ne l'observe pas —
  pour les estimateurs qui ne tolèrent pas les NaN.

L'exception à l'invariant du §2 est précisément celle-là : les lignes d'une entité
structurellement vide sont NaN au predict comme au fit (la variable de cette entité n'a pas ces
lignes dans son entraînement). L'invariant se mesure **par entité**.

#### 3.1.5 — L'exemple de référence sous chaque stratégie

Jeu du §1.1 (`m1` M, `q1` Q, `a1`/`a2` Y, cible M), imputation de `a1` à l'étape `Q` :

| Stratégie | `feature_cols` de `a1`@Q | X_train (grille Y, ancres de `a1`) | X_pred (grille Q) |
|---|---|---|---|
| `'tolerate_nan'` | `m1, q1, a2` | `m1`, `q1` agrégées à Y ; `a2` à ses ancres Y (complète si mêmes ancres) | `m1`, `q1` agrégées/identité à Q ; `a2` NaN sauf ancres → **NaN visibles au fit ET au predict** |
| `'interpolate'` | `m1, q1, a2` | idem, `a2` complète | `a2` interpolée sur la grille Q, recalée aux totaux annuels → **0 % NaN partout** |
| `'model'` + ordre `a2` avant `a1` | `m1, q1, a2` | idem | `a2` = son imputation Q (modèle) → 0 % NaN ; provenance de `a1` : `MODEL_ON_IMPUTED_COVARIATES` |
| `'model'` + ordre `a1` avant `a2` | `m1, q1, a2` | `a2` interpolée (fallback) dans X_train ? Non — complète à Y par ses ancres | `a2` portée par `covariate_fallback` → interpolée (ou NaN si `'tolerate_nan'`) — **jamais** « pleine au fit, vide au predict » |

La quatrième ligne est le cas qui produisait 33–67 % de NaN silencieux dans `hfi` : ici il est
défini, symétrique et tracé.

**Point de rigueur (X_train)** : sous `'model'`, si une covariable est servie par le fallback au
predict, la version vue au **fit** doit être préparée par la même voie (interpolée à la grille
d'entraînement si `'interpolate'`, brute si `'tolerate_nan'`) même lorsque ses ancres suffiraient
— sinon le modèle apprend sur la covariable exacte et prédit sur la covariable interpolée. La
règle : **pour chaque (étape, variable, covariable), une seule voie de matérialisation, choisie
une fois et appliquée aux deux grilles.** C'est la généralisation de l'invariant du §2 du motif
de NaN à la **nature** des valeurs.

### 3.2 — Axe 2 : l'imputation des fréquences intermédiaires

```python
impute_intermediate_frequencies: bool = ...   # défaut à trancher, §11-D3
```

- `False` : chaque variable imputable est imputée **directement à la fréquence cible** (une
  seule étape par variable et par groupe d'entités). Le modèle de la variable est entraîné sur
  ses seules observations vraies (ancres basse fréquence), éventuellement complétées de valeurs
  interpolées si la fenêtre l'exige — jamais d'imputations de modèle dans `y_train`.
- `True` : la progression de fréquences de `hfi` est conservée (Y → Q → M) ; une variable
  annuelle est d'abord imputée en trimestriel, puis, à l'étape mensuelle, son modèle est
  entraîné sur **ses ancres annuelles + ses imputations trimestrielles** — c'est ce qui
  « augmente le nombre d'observations sur lesquelles est entraîné le modèle servant à
  l'imputation finale ». En contrepartie, `y_train` porte des valeurs bruitées par le modèle de
  l'étape antérieure.

Ce paramètre reprend et fusionne deux mécanismes de `hfi`/[ARCH] : la cascade de fréquences
(PHASE 5) et `train_on_own_imputations` (§3.10 de [ARCH], jamais implémenté — prompt 15 non
exécuté). Il en hérite le **piège d'échelle**, qui est un prérequis dur d'implémentation :

> les lignes de `y_train` issues d'une imputation antérieure sont à l'échelle de **leur** étape
> (des trimestres pour une variable annuelle imputée en Q) ; leur diviseur de mise à l'échelle
> est propre à la ligne — `get_conversion_factor(pred_freq, f_ligne)` — et non le scalaire de
> l'étape. Le registre `imputed_freq_store` (fréquence de production de chaque imputation) et la
> forme `scale_factor: Union[float, pd.Series]` de [ARCH] §3.10 sont repris tels quels, y
> compris le correctif du court-circuit B12.

**Filtre de provenance de `y_train`** : la constante `TRUSTED_PROVENANCE`
(`ORIGINAL`, `AGGREGATED`, `DISAGGREGATED`, + `INTERPOLATED`, §3.3) reste le filtre sous
`impute_intermediate_frequencies=False` ; sous `True`, les cellules `MODEL_*` de la variable
elle-même y sont admises en plus (et **marquent** le modèle, §3.3).

### 3.3 — La matrice 2×2 dans la provenance

Les quatre combinaisons doivent être **possibles et distinctes** dans la provenance. Extension de
`ProvenanceType` (ajouts rétro-compatibles, l'enum est partagée avec `hfi` pendant la
transition) :

```python
class ProvenanceType(str, Enum):
    ORIGINAL      = 'original'
    AGGREGATED    = 'aggregated'
    DISAGGREGATED = 'disaggregated'
    INTERPOLATED  = 'interpolated'          # NOUVEAU : cellule produite par interpolation
                                            # (repli, ou covariable portée par 'interpolate')
    MODEL_ON_TRUE               = 'model_on_true'
    MODEL_ON_IMPUTED_COVARIATES = 'model_on_imputed_covariates'   # NOUVEAU
    MODEL_ON_IMPUTED_TARGET     = 'model_on_imputed_target'       # NOUVEAU
    MODEL_ON_IMPUTED_BOTH       = 'model_on_imputed_both'         # NOUVEAU
    MODEL_ON_MIXED = 'model_on_mixed'       # conservé pour hfi, non émis par hfi2
```

Règles d'attribution, par **étape** (le libellé est une propriété du modèle de l'étape,
propagée à toutes les cellules qu'il produit — remplaçant le booléen `trained_on_imputed`
d'`ImputationStep` par un champ `training_taint: Literal['none', 'covariates', 'target',
'both']`) :

| Covariables du modèle | `y_train` du modèle | Provenance émise |
|---|---|---|
| vraies, agrégées ou **interpolées** | vraies (ancres) ou **interpolées** | `MODEL_ON_TRUE` |
| ≥ 1 covariable **imputée par modèle** parmi les `feature_cols` effectives | vraies/interpolées | `MODEL_ON_IMPUTED_COVARIATES` |
| vraies/interpolées | ≥ 1 ligne issue d'une imputation par modèle | `MODEL_ON_IMPUTED_TARGET` |
| les deux | les deux | `MODEL_ON_IMPUTED_BOTH` |

Points fixés par la spécification :

- **l'interpolation compte comme « vrai »** : une désagrégation/interpolation déterministe
  d'observations réelles ne « bruite » pas le modèle au sens de cette matrice ([ARCH] §3.17,
  interaction provenance). Cela vaut pour les covariables **et** pour `y` (une ligne de
  `y_train` interpolée ne fait pas basculer vers `MODEL_ON_IMPUTED_TARGET`) — cohérence à
  confirmer en revue (§11-D6) ;
- le prédicat « covariables imputées » se calcule sur les **`feature_cols` effectives** du
  modèle, pas sur l'état global du store (leçon C17 de [ARCH] §3.8) ;
- correspondance avec les paramètres : `covariate_strategy='model'` est la **seule** source de
  `..._COVARIATES` ; `impute_intermediate_frequencies=True` est la seule source de
  `..._TARGET` ; les quatre combinaisons de paramètres produisent donc bien quatre familles de
  provenance distinctes et identifiables.

### 3.4 — `keep_lower_frequencies`

Conservé (nom compris, faute de mieux), documenté comme **paramètre d'affichage pur** : il
gouverne l'empilage multi-fréquences de la sortie (avec le correctif multi-niveaux B4), jamais la
logique. Sous `impute_intermediate_frequencies=False`, il n'y a pas de niveau intermédiaire à
empiler : la sortie multi-fréquences ne contient que le niveau source et le niveau cible —
comportement à documenter.

### 3.5 — Ce que le nouvel espace de paramètres n'exprime plus

À acter explicitement (décisions §11-D2) :

1. **Le mode « un seul fit, réutilisé avec le facteur d'échelle de l'étape »**
   (`cascade_refitting=False` de `hfi`). Recommandation : **abandon**. C'était une économie de
   calcul au prix d'une asymétrie fit/predict jamais vraiment maîtrisée (B7) ; si le besoin
   réapparaît, il se réintroduit comme optimisation interne (mémoïsation d'un modèle dont le jeu
   d'entraînement n'a pas changé entre deux étapes), pas comme sémantique publique.
2. **« Étapes intermédiaires sans bruit dans `y` »** — le comportement par défaut actuel de
   `hfi` (cascade active, cible entraînée sur les seules ancres). Sous la nouvelle
   spécification, activer les fréquences intermédiaires implique d'entraîner sur leurs
   imputations. Si cette combinaison doit rester exprimable, `impute_intermediate_frequencies`
   devient un `Literal[False, 'covariates_only', True]` — au prix d'une troisième colonne dans
   la matrice de provenance. Recommandation : commencer par le booléen strict (la matrice 2×2
   est l'axe structurant de la spécification), noter l'extension comme réserve.

---

## 4 — Fenêtres d'entraînement et d'imputation

Reprise intégrale de [ARCH] §3.3 + §3.7 (prévu pour `imputation_scope`/`coverage_threshold`,
prompts 10 et 13 **non exécutés** — c'est `hfi2` qui les réalise) :

```python
imputation_scope: ImputationScope = 'strict'          # fenêtre de PRÉDICTION
coverage_threshold: float = 0.5                       # seuil de ses extensions
training_scope: Optional[TrainingScope] = None        # None -> suit imputation_scope
training_coverage_threshold: Optional[float] = None   # None -> suit coverage_threshold
# TrainingScope = Literal['strict', 'extended_backward', 'extended_forward',
#                         'extended_both', 'unrestricted']
```

- `ImputationWindowCalculator` expose **trois masques** (`strict`, `imputation`, `training`) via
  `get_imputation_window_mask(data, kind=...)`, chaque appelant nommant explicitement son
  masque : ordonnancement CV → `'strict'`, entraînement → `'training'`, prédiction →
  `'imputation'`. Les correctifs B23 (branche « aucune fenêtre stricte » qui ignore les
  extensions) et B24 (docstring) sont inclus.
- **Élargir `training_scope` ajoute des lignes, jamais des colonnes** : la sélection des
  `feature_cols` reste gouvernée par la disponibilité à la prédiction (§3.1), indépendamment de
  la fenêtre d'entraînement. Les deux ajustements de [ARCH] §3.6 suivent : une colonne n'est
  gardée que si elle est non-vide sur **les deux** fenêtres ; les lignes d'entraînement sans
  aucune covariable observée sont écartées.
- **Idée du prompt 20, retenue et généralisée** : pour un panel, les masques de fenêtre sont des
  `pd.Series` booléennes à MultiIndex `(entity..., date)` — plus jamais des
  `Dict[entity, Series]`. Cela vaut pour les attributs (`imputation_window_mask_`, etc.), pour
  `get_mask_at_frequency`, et pour toute structure interne par entité de `hfi2` où c'est
  praticable : la conversion de fréquence des masques se délègue alors à
  `FrequencyConverter.convert_frequency` (vérifier au passage sa gestion des fréquences cibles
  par entité ; sinon garder la boucle interne mais unifier le **type de retour**).
- **`transform` hors fenêtre de fit (B1)** : la fenêtre est une contrainte de **disponibilité
  des données**, pas un paramètre appris — au `transform`, elle est **recalculée sur les données
  transformées** avec les hyperparamètres du fit (option A de [ARCH] §3.14), avec les deux
  garde-fous : ne jamais vider une colonne sans la réécrire, avertir quand des lignes du
  périmètre sont hors fenêtre. `fit_transform(X)` ≡ `fit(X).transform(X)` reste un invariant
  strict (le recalcul sur `X` redonne la fenêtre du fit).

---

## 5 — Ordre d'imputation

### 5.1 — `fit_predict_order`, champ d'application restreint

```python
fit_predict_order: Literal['frequency', 'cv'] = 'frequency'
```

Mêmes modalités que `train_on_partial_fit_order` de `hfi`, logiques reprises de
`_determine_imputation_order` (`'frequency'` : fréquence la plus basse d'abord, puis nombre
d'entités) et `_determine_variable_order_cv` (`'cv'` : variables les mieux prédites d'abord,
réécrite autour de `cross_val_score` — [ARCH] §3.4, corrections B9/B10 incluses).

**Champ d'application** : l'ordre n'est calculé et appliqué que sous
`covariate_strategy='model'` — le seul mode où il influe sur le résultat. Sous `'tolerate_nan'`
et `'interpolate'`, aucune logique de tri n'est exécutée (l'ordre de traitement est alors celui
des colonnes d'entrée, sans effet sur les valeurs produites — propriété garantie par le test
d'invariance §13-I3). Passer `fit_predict_order='cv'` avec une autre stratégie n'est pas une
erreur : le paramètre est ignoré, la docstring le dit (pas d'avertissement — trancher, §11-D9).

**Vérification de l'intuition « l'ordre est indifférent »** : exacte sous les deux stratégies —
`'tolerate_nan'` ne matérialise rien, `'interpolate'` matérialise depuis les seules observations ;
dans les deux cas le jeu d'entraînement et le jeu de prédiction d'une variable ne dépendent
d'aucune imputation antérieure d'une autre variable. La seule dépendance à l'ordre restante est
l'axe 2 (les imputations intermédiaires d'une variable alimentent **sa propre** étape suivante),
qui suit l'ordre des fréquences, pas l'ordre des variables.

**Déterminisme intra-étape** : sous `'model'`, les ex æquo du tri (`'frequency'` : même
fréquence, même nombre d'entités ; `'cv'` : scores égaux) sont départagés par **ordre
alphabétique du nom de variable** — jamais par l'ordre des colonnes d'entrée. C'est la réponse à
la deuxième cause de B28 : l'asymétrie intra-étape demeure (elle est intrinsèque au mode) mais
elle devient déterministe, documentée et indépendante de la présentation des données.

### 5.2 — Reprise des correctifs CV de [ARCH] §3.4

Restriction aux lignes exploitables avant scoring, `check_scoring`, `cross_val_score` avec
`error_score=np.nan`, sentinelles `-np.inf`, tri décroissant partout (*greater is better*),
journal des variables dont tous les plis ont échoué, masque `'strict'`, note sur le traitement
des zéros par le MAPE sklearn.

### 5.3 — La question de conception : paramètres CV scalaires ou crossval utilisateur ?

**Recommandation : la convention sklearn — un paramètre `cv` unique, polymorphe.**

```python
cv: Union[int, BaseCrossValidator, Iterable, None] = None
# None -> KFold(n_splits=5, shuffle=True, random_state=42)
# int  -> KFold(n_splits=cv, shuffle=True, random_state=42)
# splitter/itérable de splits -> utilisé tel quel
cv_scoring: Union[str, Callable] = 'neg_mean_absolute_percentage_error'
min_cv_train_size: int = 10
```

Arguments :

- c'est exactement le contrat de `GridSearchCV`/`cross_val_score` : tout utilisateur sklearn
  sait passer `cv=5` ou `cv=TimeSeriesSplit(...)` ; la validation est fournie par
  `sklearn.model_selection.check_cv` ;
- un splitter utilisateur ouvre les cas que des scalaires n'exprimeront jamais (CV temporelle,
  group-aware sur panel) sans un paramètre par réglage ;
- `cv_n_splits` devient redondant (absorbé par `cv=int`) — **c'est le point de friction avec la
  consigne « conserver `cv_n_splits` à l'identique »** du point 1 de la spécification :
  recommandation de le **supprimer** au profit de `cv`, décision §11-D4 ;
- `cv_scoring` reste un paramètre séparé (comme `scoring` chez sklearn) ; `min_cv_train_size`
  reste : c'est un seuil d'éligibilité au scoring, pas un réglage du splitter. L'avertissement
  croisé devient `min_cv_train_size < n_splits effectifs` (lus via `check_cv`).

Le défaut `shuffle=True, random_state=42` est conservé du code actuel (commentaire justificatif
repris) ; un utilisateur qui n'en veut pas passe son splitter.

---

## 6 — Mise à l'échelle des données

### 6.1 — `scale_features` à trois modalités, éventuellement par feature

```python
ScaleMode = Literal['constant', 'calendar']
scale_features: Union[Literal[False], ScaleMode, Dict[str, Union[Literal[False], ScaleMode]]] = 'constant'
```

- `False` : aucune mise à l'échelle des features (`y` reste toujours mis à l'échelle — même
  comportement que le `False` actuel).
- `'constant'` : le `True` actuel — diviseur constant par couple de fréquences, via
  `FrequencyConverter.get_conversion_factor` (M→Y = 12, D→M = 30.0…). Adapté aux variables
  **corrigées des variations saisonnières** : le facteur moyen lisse ce que la CVS a déjà lissé.
- `'calendar'` : diviseur par **décompte calendaire réel** de la période, via
  `FrequencyConverter.count_subperiods_per_period` (février → 28/29, T1 → 90/91). Adapté aux
  variables **brutes** (non CVS), où le nombre de jours ouvrés/calendaires porte du signal. Le
  diviseur devient une `pd.Series` par ligne — la plomberie `scale_factor: Union[float,
  pd.Series]` du §3.2 sert les deux besoins (synergie à exploiter : un seul chemin de code).
- **Forme dict** : `{feature: modalité}` avec clé `'__default__'` optionnelle (même convention
  que `estimator`), pour appliquer une méthode par feature — typique d'un jeu mêlant séries CVS
  et brutes. Clés inconnues → `ValueError` au `fit` (liste des colonnes fautives) ; features non
  couvertes sans `'__default__'` → le défaut `'constant'`.

### 6.2 — Ce que la modalité gouverne exactement

La modalité s'applique à **tous les diviseurs** relatifs à la feature : diviseur d'entraînement
(`_covariate_scaling_divisors`, règle B25), diviseur de `y` (scalaire d'étape ou par ligne,
§3.2), et report d'échelle des prédictions (`fit_scale_factor`). Un mélange « feature en
`'calendar'`, y en `'constant'` » est légitime et doit être couvert par un test.

### 6.3 — Prérequis

- **B26 (prompt 21, non exécuté)** : le contrôle `full_periods_only` de la somme dans
  `utils/frequency/converter.py` utilise le décompte **constant** et jette février — toute
  covariable journalière disparaît silencieusement du jeu d'entraînement. `hfi2` s'appuyant sur
  les mêmes agrégations, **ce correctif est un prérequis dur** du chantier (à faire avant ou en
  tout début).
- La vérification d'audit demandée (« moyenne des features sur train et sur test comparées »)
  devient un invariant de test (§13-I5) et une section du notebook pas à pas (§12.2).

---

## 7 — Interpolation et position d'ancrage

### 7.1 — Méthode d'interpolation, globale ou par feature

```python
interpolation_method: Union[str, Dict[str, str]] = 'linear'
```

Valeurs : celles de `FrequencyConverter.interpolate_to_higher_frequency` (`'linear'`, `'time'`,
`'nearest'`, `'zero'`, `'slinear'`, `'quadratic'`, `'cubic'`, …). Forme dict par feature avec
`'__default__'`, mêmes règles de validation qu'au §6.1. Ce paramètre sert **trois** usages avec
la même valeur (cohérence voulue par la spécification) : la stratégie `'interpolate'` (§3.1.2),
le `covariate_fallback='interpolate'` (§3.1.3), et le **repli d'échec** d'une imputation par
modèle (l'actuel `INTERPOLATE_FALLBACK`).

### 7.2 — Position d'ancrage de la valeur dans sa période

Extension de `FrequencyConverter` (et exposée par `FrequencyAligner`), utilisée par `hfi2` :

```python
interpolation_anchor: Union[None, float, Dict[str, Union[None, float]]] = None
```

- `None` (défaut) : comportement actuel — la valeur s'applique à la date à laquelle elle est
  référencée (début ou fin de période selon la position de l'index).
- `float ∈ [0, 1]` : la valeur observée d'une période est considérée comme atteinte à la
  **fraction** correspondante de la période qu'elle couvre — `0.5` = milieu de période, `0.0` =
  début, `1.0` = fin. L'interpolation entre ancres se fait alors entre ces points décalés, puis
  le résultat est réindexé sur la grille cible.
- Forme dict par feature avec `'__default__'`.

**Point d'implémentation** (dans `FrequencyConverter`, nouvelle option d'
`interpolate_to_higher_frequency`, p. ex. `anchor_fraction: Optional[float] = None`) : décaler
l'index ré-ancré de `_reanchor_index_to_target` vers `début_période + fraction × durée_période`
**avant** interpolation, interpoler sur l'union (index décalé ∪ grille cible), puis ne retenir
que la grille cible. Les timestamps décalés ne tombent en général **pas** sur la grille cible —
l'union est indispensable, et les valeurs d'ancre elles-mêmes ne survivent pas telles quelles
dans la sortie (c'est le but : à `0.5`, la valeur de mars ne prétend plus valoir au 31 mars).

**Interactions à spécifier dans la docstring** :

- avec le recalage aux totaux de période (§8) : l'ancrage fractionnaire change la **forme** de
  l'interpolation, le recalage ré-impose ensuite le total — les deux se composent sans conflit ;
- aux bords de série : au-delà de la dernière ancre décalée, `limit_direction` décide (défauts
  actuels de `_resolve_limit_direction` conservés) ;
- ce paramètre s'applique à l'interpolation **seulement** (stratégie, fallback, repli) — jamais
  à l'agrégation ni à la désagrégation par totaux, qui restent ancrées sur les périodes exactes.

---

## 8 — Contraintes d'agrégation

### 8.1 — Renommage d'`enforce_period_totals`

Rôle inchangé (recaler les sous-périodes prédites pour qu'elles somment au total observé de la
période). Proposition de nom : **`enforce_aggregation_constraints`** (alternatives :
`enforce_period_sums`, `aggregation_constraint: Literal['sum', None]` — cette dernière ouvrant la
porte à d'autres contraintes (moyenne pour les taux) mais au prix d'un type plus lourd ;
décision §11-D8). Les acquis de `hfi` sont repris :

- marquage de provenance des ancres indépendant de la réussite du recalage (B2) ;
- chemin d'interpolation unique appliquant recalage puis `dropna` dans cet ordre (B27) ;
- garde « période partiellement prédite » de `_rescale_to_period_totals`.

### 8.2 — Sort de `disaggregate_anchors` ([ARCH] §3.9, prompt 14 non exécuté)

La spécification orale ne le mentionne pas. Par défaut, `hfi2` reprend le comportement actuel de
`hfi` : une variable imputée est prédite sur **toute** sa période, ancres comprises, pour que sa
colonne ne mélange jamais total de période et valeurs de sous-période. Décision à prendre
(§11-D7) : porter ou non le paramètre `disaggregate_anchors=False` (préserver les ancres,
colonne à échelles mélangées) dans `hfi2`.

---

## 9 — Architecture logicielle

### 9.1 — Principe : un plan, un exécuteur

La leçon des défauts B7/B27/B8 : chaque fois que `fit` et `transform` portent deux copies d'une
logique, elles divergent. `hfi2` sépare donc **construction du plan** et **exécution du plan** :

```
fit       = phases 0-4  →  PlanBuilder (construit ET exécute chaque étape au fil de l'eau,
                            car l'étape k dépend des imputations de l'étape k-1)
transform = phases 0'-4' (recalculs data-dépendants)  →  PlanExecutor (rejoue le plan figé)
```

- Le **plan** (`List[ImputationStep]`, adapté) est l'état ajusté complet, comme aujourd'hui.
- **Une seule implémentation de l'exécution d'étape** (`_execute_step`), paramétrée par
  « fit : ajuster puis prédire » vs « transform : prédire avec le modèle figé ». Les écritures
  (vidage/réécriture, recalage, provenance, store) sont **communes par construction**.
- Ce qui est **état du fit** (rejoué tel quel) vs **recalculé au transform** :

| Rejoué depuis le fit | Recalculé sur les données du transform |
|---|---|
| classification des variables, fréquences détectées (validées contre les données du transform), progression de fréquences, ordre | fenêtres d'imputation/entraînement (§4, B1) |
| modèles, `feature_cols`, voie de matérialisation par covariable (§3.1.5), facteurs/modalités d'échelle | frames d'étape, valeurs interpolées, prédictions, provenance du transform |
| méthodes/ancrage d'interpolation par feature | masques de prédiction, recalage aux totaux |

### 9.2 — Découpage en composants

Fichiers nouveaux dans `tsforecast/frequency/` (une responsabilité par module, la classe
principale restant un orchestrateur mince) :

| Composant | Fichier | Responsabilité |
|---|---|---|
| `HighFrequencyImputer2` | `high_frequency_imputer2.py` | API sklearn, validations, orchestration fit/transform/inverse_transform |
| `CovariateMaterializer` | `covariate_materializer.py` | matérialisation des covariables sur une grille selon `covariate_strategy`/`covariate_fallback`/`interpolation_*` ; **unique** producteur de `X_train`/`X_pred` ; tient le miroir des imputations et le registre de fréquences |
| `StageScaler` | `stage_scaler.py` | diviseurs (`'constant'`/`'calendar'`, scalaires et par ligne), application/inversion de l'échelle, report d'échelle des prédictions |
| `VariableOrderer` | `variable_orderer.py` | ordres `'frequency'` et `'cv'` (avec `cv`, `cv_scoring`, `min_cv_train_size`), tie-break alphabétique |
| `ImputationStep` (v2) + plan | `imputation_plan.py` (étendu) ou `imputation_plan2.py` | étape immuable : + `training_taint`, + voie de matérialisation par covariable, − `trained_on_imputed`, − `feature_means` |
| réutilisés | `imputation_window.py` (3 masques, MultiIndex), `frequency_aligner.py`, `provenance.py` (enum étendue), `target_frequency_validator.py`, `regularizer.py`, `utils/frequency/converter.py` (+ `anchor_fraction`, correctif B26) | inchangés dans leur rôle |

`hfi` et `hfi2` coexistent pendant la transition (exports distincts dans
`tsforecast/frequency/__init__.py`) ; le remplacement effectif (dépréciation puis suppression de
`HighFrequencyImputer`) est un chantier ultérieur, hors périmètre de ce document.

### 9.3 — Squelette de `fit`

Phases 0 à 4 reprises de `hfi:_fit` (§1.3), puis :

```
PHASE 5 — pour chaque étape de fréquence f de la progression
          (progression = [cible] si impute_intermediate_frequencies=False) :
  5a. frame d'étape : données d'origine + agrégations à f + miroir des imputations
      (via CovariateMaterializer, unique aussi pour le transform)
  5b. variables imputables à f ; ordre (VariableOrderer, seulement si strategy='model')
  5c. pour chaque variable v :
      - grille d'entraînement (ancres de f_var [+ imputations intermédiaires de v]),
        fenêtre 'training' ; grille de prédiction, fenêtre 'imputation'
      - matérialisation des covariables sur LES DEUX grilles par la MÊME voie (§3.1.5)
      - mise à l'échelle (StageScaler)
      - ajustement ; en cas d'échec → repli interpolation (méthode de v), étape marquée fallback
      - prédiction, recalage aux totaux, écriture, provenance (§3.3), stores
      - matérialisation de v (y compris en repli) pour la suite de l'étape si strategy='model'
PHASE 6 — finalisation (plan figé, attributs de sortie, sortie multi-fréquences si demandé)
```

### 9.4 — `transform` et `inverse_transform`

- `transform` : mêmes phases 0'-4' data-dépendantes (alignement/nommage de `y` par la **même**
  fonction qu'au fit — B14 ; transformateur additif appliqué avec l'objet ajusté ; tracker
  initialisé après lui — B8 ; fenêtres recalculées — B1), puis `PlanExecutor`.
- `inverse_transform` : reprise du chemin actuel (sélection du niveau de fréquence source,
  inversion du transformateur additif, restitution par masque `ORIGINAL`,
  `restore_original_values`), avec les invariants B4 (multi-niveaux) et B19 (purge d'état au
  `fit`).

### 9.5 — Conformité sklearn et contrat d'entrée (repris de [ARCH] §3.16, non négociables)

- B3 : `__init__` stocke tel que reçu, `clone`/`get_params` exacts ; normalisations au `fit` ;
- B14 : nommage unique de `y`, vérification d'**égalité des index** (pas seulement des
  longueurs) ;
- B15 : panel déclaré par `panel_cols` sur frame plat fonctionnel ;
- B16 : dict `target_frequency` incomplet → `ValueError` nommant les entités manquantes ;
- B19 : purge de l'état de `transform` en tête de `fit` ;
- B20 : `NotFittedError` propre avant `fit` (liste d'attributs explicite à `check_is_fitted`) ;
- avertissements **uniques** (estimateur absent, lignes hors fenêtre), jamais un par
  variable×étape.

---

## 10 — API résultante

```python
class HighFrequencyImputer2(XYPanelTimeSeriesTransformer):
    def __init__(
        self,
        target_frequency: Union[str, Dict[Union[str, tuple], str]],
        estimator: Optional[Union[BaseEstimator, Dict[str, BaseEstimator]]] = None,
        additive_transformer: Optional[TransformerMixin] = None,

        # --- Axe 1 : matérialisation des covariables (§3.1) ---
        covariate_strategy: Literal['tolerate_nan', 'interpolate', 'model'] = 'interpolate',
        covariate_fallback: Literal['interpolate', 'tolerate_nan'] = 'interpolate',
        covariate_eligibility: Literal['any_entity', 'all_entities'] = 'any_entity',
        interpolation_method: Union[str, Dict[str, str]] = 'linear',
        interpolation_anchor: Union[None, float, Dict[str, Optional[float]]] = None,

        # --- Axe 2 : fréquences intermédiaires (§3.2) ---
        impute_intermediate_frequencies: bool = ...,          # défaut : décision D3

        # --- Ordre (§5) ---
        fit_predict_order: Literal['frequency', 'cv'] = 'frequency',
        cv: Union[int, BaseCrossValidator, Iterable, None] = None,   # décision D4
        cv_scoring: Union[str, Callable] = 'neg_mean_absolute_percentage_error',
        min_cv_train_size: int = 10,

        # --- Fenêtres (§4) ---
        imputation_scope: ImputationScope = 'strict',
        coverage_threshold: float = 0.5,
        training_scope: Optional[TrainingScope] = None,
        training_coverage_threshold: Optional[float] = None,

        # --- Échelle et contraintes (§6, §8) ---
        scale_features: Union[Literal[False], ScaleMode,
                              Dict[str, Union[Literal[False], ScaleMode]]] = 'constant',
        enforce_aggregation_constraints: bool = True,         # ex-enforce_period_totals

        # --- Sortie et divers ---
        keep_lower_frequencies: bool = True,
        on_frequency_mismatch: Literal['error', 'warn'] = 'error',
        restore_original_values: bool = False,
        time_col: Optional[str] = None,
        panel_cols: Optional[List[str]] = None,
        verbose: bool = False,
    ): ...
```

Validations à `__init__` (sans transformation, B3) : celles de `hfi` reprises + `Literal`s des
nouveaux paramètres + `interpolation_anchor ∈ [0, 1]` + cohérence des dicts par feature (types
des valeurs ; les **clés** ne sont vérifiables qu'au `fit`, contre les colonnes réelles) +
avertissements croisés (`covariate_fallback` ignoré hors `'model'` ? — décision D9 ;
`training_coverage_threshold` sans `training_scope` inerte, à documenter).

Attributs ajustés principaux : `effective_target_frequency_`, `detected_frequencies_`,
`variable_categories_` (format [ARCH] §3.2), `imputation_plan_`, `imputation_models_` (vue),
`imputation_window_` / `training_window_`, masques MultiIndex (§4), `imputation_provenance_`
(après `transform`), `feature_columns_`, `target_column_`, `entities_`, `is_panel_`.

---

## 11 — Problèmes, manques et décisions à trancher

À incorporer dans le second prompt (revue Opus). **A-x** = ambiguïté de la spécification orale,
**D-x** = décision de conception ouverte, **P-x** = prérequis/défaut externe.

### Ambiguïtés de la spécification orale

- **A1 — phrase tronquée, point 1** : « En revanche, je souhaite . » — la fin manque. Contexte
  probable : un renommage ou une suppression parmi les arguments non listés. **À compléter par
  l'auteur** ; en attendant, ce document suppose que les arguments non cités de `hfi`
  (`cascade_refitting`, `covariate_eligibility`, `train_on_partial_coverage`,
  `train_on_partial_fit_order`, `scale_features`, `keep_lower_frequencies`,
  `coverage_threshold`, `imputation_scope`) suivent le sort décrit aux §1.4 et §3–6.
- **A2 — phrase tronquée, point 4, second paramètre** : « Dans ce cas » — interprétée ici comme
  « dans ce cas, les imputations intermédiaires de la variable entrent dans son jeu
  d'entraînement aux étapes suivantes » (§3.2), cohérente avec la phrase suivante sur la
  provenance. À confirmer.
- **A3 — point 6** : « On pourra également cela sous la forme d'un dictionnaire » — lu comme
  « spécifier cela », forme dict feature → modalité (§6.1). À confirmer.
- **A4 — défauts non spécifiés** : la spécification ne fixe le défaut d'aucun des deux axes ni
  des nouveaux paramètres. Propositions : `covariate_strategy='interpolate'` (sûr,
  déterministe, indépendant de l'ordre), `covariate_fallback='interpolate'`,
  `interpolation_method='linear'`, `interpolation_anchor=None`, `scale_features='constant'`
  (continuité avec le `True` actuel). Reste D3 pour l'axe 2.

### Décisions de conception

- **D1 — validation des booléens et nouveaux littéraux** : reprise du bloc groupé de `hfi` —
  acté, mention pour mémoire.
- **D2 — capacités perdues** (§3.5) : abandonner le mode « un seul fit réutilisé »
  (recommandé : oui) ; rendre exprimable « étapes intermédiaires sans bruit dans `y` »
  (recommandé : non dans un premier temps, extension `Literal` notée en réserve).
- **D3 — défaut d'`impute_intermediate_frequencies`** : `True` prolonge l'esprit du défaut
  actuel (cascade) mais produit du `MODEL_ON_IMPUTED_TARGET` par défaut ; `False` donne le
  comportement le plus propre (MODEL_ON_TRUE) et le moins surprenant pour un nouvel utilisateur.
  Recommandation : **`False`**.
- **D4 — `cv` vs `cv_n_splits`** (§5.3) : la recommandation (paramètre `cv` sklearn, suppression
  de `cv_n_splits`) contredit la consigne « conserver `cv_n_splits` » — à arbitrer. Position de
  repli : garder `cv_n_splits` ET accepter `cv` (avec `cv` prioritaire), au prix d'une
  redondance.
- **D5 — noms des paramètres** : `covariate_strategy` / `covariate_fallback` /
  `impute_intermediate_frequencies` / `fit_predict_order` / `interpolation_anchor` /
  `enforce_aggregation_constraints` sont des propositions — valider ou renommer maintenant, plus
  jamais ensuite.
- **D6 — `INTERPOLATED` et `y_train`** (§3.3) : une ligne de `y_train` interpolée compte-t-elle
  comme « vraie » (recommandé, symétrie avec les covariables) ? Et faut-il que les cellules du
  **repli** d'imputation (variable en échec de modèle) portent `INTERPOLATED` plutôt que le
  marquage actuel — recommandé : oui, c'est plus exact et cela rend le repli visible dans les
  statistiques de provenance.
- **D7 — `disaggregate_anchors`** (§8.2) : porter ou non ce paramètre ([ARCH] §3.9) dans `hfi2`.
  Recommandation : non au premier jet (comportement actuel conservé), à garder dans le backlog.
- **D8 — nom et extension de la contrainte d'agrégation** (§8.1) : `enforce_aggregation_constraints`
  booléen, ou `aggregation_constraint: Literal['sum', None]` extensible (moyenne pour les taux,
  dernière valeur pour les stocks) ? Recommandation : booléen maintenant, littéral si le besoin
  « moyenne » se matérialise — il est réel pour des variables non additives type taux de
  chômage, aujourd'hui hors contrat (le transformateur additif est censé les rendre additives).
- **D9 — paramètres inertes silencieux ou bavards** : `fit_predict_order` hors `'model'`,
  `covariate_fallback` hors `'model'`, `interpolation_anchor` sous `'tolerate_nan'` pur… —
  ignorer en silence (documenté) ou `UserWarning` ? Recommandation : silence + docstring, un
  avertissement par combinaison rendrait la classe pénible en exploration d'hyperparamètres.
- **D10 — granularité des dicts par feature sur panel** : les dicts (`interpolation_method`,
  `interpolation_anchor`, `scale_features`, `estimator`) sont indexés par **nom de colonne**,
  jamais par `(entité, colonne)`. À confirmer — la forme par entité multiplierait la surface de
  validation pour un besoin non exprimé.
- **D11 — comportement de `transform` face à des fréquences détectées différentes du fit** : le
  plan est rejoué avec les fréquences du fit ; si la détection sur les données du transform
  diverge (colonne devenue plus fine, etc.), erreur ou avertissement + poursuite ?
  Recommandation : avertissement + poursuite avec les fréquences du fit, erreur si une colonne
  du fit manque.

### Prérequis et défauts externes

- **P1 — B26 / prompt 21** (§6.3) : correctif du décompte calendaire de `full_periods_only`
  dans `utils/frequency/converter.py` — **prérequis dur**, indépendant de `hfi2`, à exécuter en
  premier.
- **P2 — extension `FrequencyConverter.anchor_fraction`** (§7.2) : nouvelle capacité utilitaire,
  testable isolément, prérequis de la stratégie `'interpolate'` complète.
- **P3 — extension `ImputationWindowCalculator`** (trois masques + `training_scope` +
  MultiIndex, §4) : prompts 10/13/20 de [ARCH] jamais exécutés — leur contenu est repris par le
  chantier `hfi2` et réalisé sur `iwc` (partagé avec `hfi` sans changement de comportement par
  défaut).
- **P4 — extension `ProvenanceType`** (§3.3) : ajouts d'`INTERPOLATED` et des trois
  `MODEL_ON_IMPUTED_*`, rétro-compatibles avec `hfi`.
- **P5 — feature panel manquante par entité** (§12.1) : à faire **en premier** (demande
  explicite), car les jeux d'exemple servent ensuite à tous les tests et notebooks.

---

## 12 — Prérequis et travaux annexes

### 12.1 — Notebook 3 : ajouter une feature absente pour certaines entités

`notebooks/3 - QB - Panel a frequences mixtes heterogene.ipynb`, `create_panel_dataset` :
ajouter une variable (proposition : `climat_affaires`, indicateur mensuel d'enquête) présente
pour la France et l'Allemagne, **absente pour l'Italie** (colonne entièrement NaN pour cette
entité — la colonne existe dans le frame, c'est l'entité qui ne l'observe jamais). Compléter la
section 2.3 de vérification. Le jeu étant réutilisé en aval (notebook 4, futur notebook 5,
fixtures de tests qui s'en inspirent), vérifier et ré-exécuter les consommateurs. C'est le cas
d'usage de `covariate_eligibility` (§3.1.4) et la limite documentée de la stratégie
`'interpolate'`.

### 12.2 — Notebook 5 : pas à pas de `HighFrequencyImputer2`

Sur le modèle de `notebooks/4 - QB - HighFrequencyImputer pas a pas.ipynb`, un notebook
`5 - QB - HighFrequencyImputer2 pas a pas.ipynb` qui, sur **les deux jeux** du notebook 3
(série temporelle et panel) :

- détaille variables et données à chaque phase de `fit` et de `transform` ;
- affiche, **pour chaque étape du plan** : les jeux `X_train`/`y_train` et `X_pred` exacts, la
  voie de matérialisation retenue par covariable, les fenêtres appliquées (les trois masques),
  et la provenance après écriture ;
- **audite le scaling** : moyenne de chaque feature sur le train et sur le test, côte à côte —
  l'écart relatif doit être compatible avec la modalité choisie (`'constant'` vs `'calendar'`),
  et un déséquilibre train/test est le symptôme immédiat d'un diviseur faux (c'est le contrôle
  demandé explicitement) ;
- montre les **quatre combinaisons** des deux axes sur le même jeu, avec la matrice de
  provenance résultante (les quatre familles `MODEL_*` distinctes) ;
- conserve le contrôle croisé du notebook 4 (pas-à-pas vs `fit_transform`, et
  `fit_transform(X)` vs `fit(X).transform(X)`).

Règle permanente reprise de [ARCH] §6 : tout prompt d'implémentation qui modifie l'exécution
d'étape met à jour la cellule pas-à-pas correspondante **dans le même lot**.

### 12.3 — Coexistence et migration

`hfi2` est développé à côté de `hfi`, sans toucher au comportement par défaut des composants
partagés (`iwc`, converter, provenance — extensions rétro-compatibles uniquement). La
dépréciation de `HighFrequencyImputer` n'intervient qu'après la validation croisée des deux
classes sur les notebooks (hors périmètre du présent chantier).

---

## 13 — Stratégie de tests et invariants

Suite dédiée `tests/frequency/test_high_frequency_imputer2*.py`, fixtures partagées avec
l'existant (`tests/frequency/conftest.py`), critère « aucun échec nouveau » vs
`BASELINE_FAILURES.txt` pour les composants partagés. Les invariants ci-dessous sont l'ossature —
chacun testé sur la série temporelle ET le panel (y compris l'entité sans feature, §12.1) :

- **I1 — symétrie** : `fit_transform(X)` ≡ `fit(X).transform(X)` (égalité stricte des valeurs,
  des provenances et des attributs de sortie).
- **I2 — invariant NaN (§2)** : par étape, taux de NaN par colonne de `X_pred` ≤ celui de
  `X_train`, mesuré par estimateur espion, sous les trois stratégies — le test qui échoue sur
  `hfi` aujourd'hui.
- **I3 — invariance à l'ordre des colonnes** : permuter les colonnes d'entrée ne change ni les
  valeurs imputées ni les provenances, sous les trois stratégies (sous `'model'`, grâce au
  tie-break alphabétique §5.1).
- **I4 — additivité** : sous contrainte d'agrégation active, chaque colonne imputée somme au
  total observé de chaque période complète (imputations, interpolations et covariables portées
  comprises).
- **I5 — échelle train/test** : pour chaque étape, moyenne des features de `X_train` et de
  `X_pred` comparables (tolérance dépendant de la modalité) — le pendant test du contrôle
  notebook §12.2 et de B25.
- **I6 — provenance 2×2** : les quatre combinaisons de paramètres produisent leurs quatre
  familles `MODEL_*` respectives et aucune autre ; interpolé compte comme vrai (`MODEL_ON_TRUE`).
- **I7 — transform hors fenêtre de fit** : impute au lieu de vider, ne détruit jamais une
  observation d'entrée, avertit sur les lignes inimputables.
- **I8 — aller-retour** : `inverse_transform(transform(X))` restitue l'index, les noms (panels
  multi-niveaux compris) et, sous `restore_original_values=True`, les valeurs d'origine.
- **I9 — conformité sklearn** : `clone`, `get_params`/`set_params`, `Pipeline`, `GridSearchCV`
  sur panel avec `target_frequency` dict ; `NotFittedError` avant `fit`.
- **I10 — indifférence de l'ordre d'imputation hors `'model'`** : sous `'tolerate_nan'` et
  `'interpolate'`, forcer deux ordres de traitement différents produit des sorties identiques.
