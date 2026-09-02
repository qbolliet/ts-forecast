# Prompts d'implémentation — `HighFrequencyImputer2`

> Document généré le **2026-09-01** à partir de `high_frequency_imputer2_architecture.md`
> (noté ci-après **[SPEC]**), dont le **§17** fournit le découpage en lots `L0`–`L13`.
> Chaque prompt est **autonome** et destiné à être collé tel quel dans une session Claude Code
> **indépendante**. Les prompts sont **ordonnés** ; les dépendances dures sont indiquées en tête.
> Après chaque prompt, l'auteur relit le code ajouté ou modifié avant de lancer le suivant.
>
> **Convention de lecture** : `hfi:` = `tsforecast/frequency/high_frequency_imputer.py`,
> `hfi2:` = `tsforecast/frequency/high_frequency_imputer2.py` (à créer),
> `iwc:` = `tsforecast/frequency/imputation_window.py`,
> `conv:` = `tsforecast/utils/frequency/converter.py`.
> Les codes `Bxx`/`Cxx` renvoient à `high_frequency_imputer_annotations_architecture.md`
> (noté **[ARCH]**), les codes `Dxx` au §14 de [SPEC].

---

## État vérifié du dépôt (2026-09-01, branche `qb-mixed-frequencies`)

Constats établis par lecture du dépôt **avant** rédaction de ces prompts. Ils conditionnent le
périmètre réel de plusieurs lots — ne pas les redécouvrir en session.

- `tsforecast/frequency/` contient : `frequency_aligner.py`, `high_frequency_imputer.py`,
  `imputation_plan.py`, `imputation_window.py`, `provenance.py`, `regularizer.py`,
  `target_frequency_validator.py`. **Aucun** des six modules nouveaux du §12.2 de [SPEC]
  n'existe encore.
- ⚠️ **Le lot `L3` est déjà réalisé à ~90 %** : `iwc` expose **déjà** `training_scope`,
  `training_coverage_threshold`, `_effective_training_scope`,
  `_effective_training_coverage_threshold`, `_build_scope_mask`, `_select_mask`, les trois
  attributs `imputation_strict_window_mask_` / `imputation_window_mask_` /
  `training_window_mask_`, et l'argument `kind=` sur `get_imputation_window_mask` **et**
  `get_mask_at_frequency` (prompt 10 de la campagne précédente, exécuté ; B23/B24 inclus).
  **Ne reste que** l'unification du type de retour en `pd.Series` à MultiIndex (§7.2 de [SPEC],
  ancien prompt 20 jamais exécuté). Le **prompt 5** ci-dessous est réduit à cela.
- `provenance.py` : `ProvenanceType` porte encore `MODEL_ON_MIXED`, et **ni** `INTERPOLATED`
  **ni** les quatre nouvelles constantes `MODEL_*`. `mark_model_imputed` a toujours la signature
  `(column, index, trained_on_imputed: bool)`. `resolve_model_provenance` n'existe pas.
- `conv:_require_full_subperiod_coverage` : le chemin **somme** utilise toujours le décompte
  **constant** (défaut B26, non corrigé) ; le chemin **booléen** du même fichier utilise déjà
  `count_subperiods_per_period`. `interpolate_to_higher_frequency` n'a **pas** d'argument
  `anchor_fraction`.
- `imputation_plan.py` : `ImputationStep` est un dataclass portant `trained_on_imputed: bool`,
  `scale_factor: float`, `fit_scale_factor: float`, et une **propriété** `is_fallback` dérivée de
  `model is INTERPOLATE_FALLBACK`. Il n'y a pas de champ `feature_means`.
- Il n'existe **pas** de `tests/frequency/test_provenance.py`. Les tests de `conv` vivent dans
  `tests/frequency/test_converter_*.py` (pas dans `tests/utils/frequency/`, qui ne contient que
  `test_detector.py`, `test_normalizer.py`, `test_parser.py`) : **la convention du dépôt prime
  sur les chemins indicatifs du §17 de [SPEC]**.
- Fixtures partagées : `tests/frequency/conftest.py` expose `mixed_freq_timeseries`,
  `mixed_freq_panel`, `panel_two_level_dataset`, construites par `_build_timeseries`,
  `_build_panel`, `_build_panel_two_level`.
- Classe de base de `hfi` : `XYPanelTimeSeriesTransformer` (`tsforecast/xy/transformers.py`).

### ⚠️ Filet de tests : la suite n'est PAS verte au départ

Le critère d'acceptation n'est **jamais** « suite verte » mais **« aucun échec NOUVEAU par
rapport à la référence »**, vérifié mécaniquement :

```bash
uv run tests/frequency/check_regressions.py     # compare à tests/frequency/BASELINE_FAILURES.txt
```

Deux lots font **volontairement** bouger des valeurs numériques (prompt 2 / B26) ou **cassent
volontairement** une API partagée (prompt 4 / suppression de `MODEL_ON_MIXED`) : ils régénèrent
`BASELINE_FAILURES.txt` **en fin de lot**, après avoir listé explicitement les tests modifiés et
la raison de chaque modification.

---

## Récapitulatif

| # | Lot | Objet | Réf. [SPEC] | Modèle | Plan mode | Dépendances |
|---|-----|-------|-------------|--------|-----------|-------------|
| 1 | **L0** | Jeu `PANEL` hétérogène : `climat_affaires` absente pour `IT` | §2.3, §15.1 | **Sonnet** | Non | — |
| 2 | **L1a** | `conv` : décompte calendaire de `full_periods_only` (B26) | §9.3, P1 | **Opus** | **Oui** | — |
| 3 | **L1b** | `conv` : `interpolate_to_higher_frequency(anchor_fraction=…)` | §10.2, P2 | **Opus** | **Oui** | 2 |
| 4 | **L2** | `provenance.py` : énumération de souillure, `resolve_model_provenance`, `mark_model_imputed` | §6, §6.7, P4 | **Sonnet** | Non | — |
| 5 | **L3** | `iwc` : type de retour `pd.Series` MultiIndex unifié | §7.2, P3 | **Sonnet** | Non | — |
| 6 | **L4** | `imputation_plan2.py` : `ImputationStep` v2 + plan immuable | §12.2, §4.6, §6.2 | **Sonnet** | Non | 4 |
| 7 | **L5** | `stage_scaler.py` : diviseurs `'constant'`/`'calendar'`, scalaires et `Series` | §9, §5.4 | **Opus** | **Oui** | 2, 3 |
| 8 | **L6a** | `covariate_materializer.py` : socle, stores, `'tolerate_nan'`, `'interpolate'` | §4.1–4.3, §4.5, §6.2 | **Opus** | **Oui** | 3, 4, 5 |
| 9 | **L6b** | `covariate_materializer.py` : stratégie `'model'`, précédence à 4 rangs | §4.4, §4.6 | **Opus** | **Oui** | 8 |
| 10 | **L7** | `aggregation_constraint.py` : recalage aux totaux, gardes, ancres | §11 | **Opus** | **Oui** | 4 |
| 11 | **L8** | `variable_orderer.py` : ordres `'frequency'` et `'cv'`, `cv` sklearn | §8 | **Sonnet** | Non | 5 |
| 12 | **L9** | `hfi2` : `__init__`, validations, phases 0 à 4, attributs ajustés | §12.3, §13 | **Opus** | **Oui** | 6–11 |
| 13 | **L10** | `hfi2` : PHASE 5, exécution d'étape unique, provenance, stores | §12.3, §6.3 | **Opus** | **Oui** | 12 |
| 14 | **L11** | Axe 2 : progression, `ELIGIBLE_ORIGINS`, échelle par ligne, report d'étape | §5 | **Opus** | **Oui** | 13 |
| 15 | **L12** | `transform`, `inverse_transform`, `keep_lower_frequencies`, D11 | §12.1, §12.4 | **Opus** | **Oui** | 14 |
| 16 | **L13a** | Notebook 5 « pas à pas » | §15.2 | **Sonnet** | Non | 1, 15 |
| 17 | **L13b** | Documentation : docstrings de référence, `mkdocs`, `__init__.py` | §15.3 | **Sonnet** | Non | 16 |

**Dépendances dures** : 2 → 3 ; 4 → 6 ; 3 → 7 ; 3, 4, 5 → 8 ; 8 → 9 ; 4 → 10 ; 5 → 11 ;
6, 7, 9, 10, 11 → 12 ; 12 → 13 → 14 → 15 → 16 → 17.
**Parallélisables entre eux** : {1, 2, 4, 5} ; {7, 10, 11} une fois 3 et 4 faits ; 1 avec tout.

### Critère de choix du modèle

**Opus** pour tout lot qui raisonne simultanément sur les **échelles de fréquence**, la
**symétrie fit/transform** et la **provenance** : une erreur y produit des résultats faux sans
crash (lots 2, 3, 7, 8, 9, 10, 12, 13, 14, 15).
**Sonnet** pour les lots à surface fermée et à critère binaire : créations de dataclass,
renommages mécaniques, tri de variables, notebooks, docstrings (lots 1, 4, 5, 6, 11, 16, 17).

### Critère de choix du plan mode

**Oui** dès qu'un lot crée un composant dont l'API sera consommée par plusieurs lots ultérieurs,
ou qu'il modifie un contrat partagé entre `fit` et `transform` : le plan valide la répartition
des responsabilités **avant** écriture, et c'est le seul moment où une signature se corrige à
coût nul.
**Non** pour les lots mécaniques, dont le résultat attendu est entièrement décrit par le prompt.

### Convention applicable à TOUS les prompts

Ces six points de vigilance sont ceux du §17 de [SPEC]. Ils sont rappelés en fin de chaque
prompt pour que chaque session soit autonome.

1. Commentaires internes **en français**, formulations **nominales**
   (`# Vérification des arguments`, pas `# Vérifier les arguments`) ; docstrings **en anglais**,
   **Google Style**, avec `Args:` / `Returns:` / `Raises:` / `Examples:` ; type hints
   systématiques (`CLAUDE.md`).
2. `impute_intermediate_frequencies` ne se teste **jamais** par vérité booléenne :
   `'covariates_only'` est *truthy*. Toujours `is False`, `== 'covariates_only'`, `is True`.
3. `X_train` et `X_pred` ne sont produits **que** par `CovariateMaterializer.materialize`.
4. Le filtre de `y_train` et le calcul des souillures lisent `origin_store`, **jamais** la
   matrice de provenance.
5. Tout avertissement est **agrégé et unique** : accumulation puis message en fin de phase,
   jamais un par variable × étape.
6. `__init__` **valide sans transformer** (B3) : paramètres stockés tels que reçus, normalisation
   au `fit` dans des attributs suffixés `_`.

Localiser le code **par nom de symbole**, jamais par numéro de ligne. Ne jamais masquer un échec
de test : le rapporter tel quel.

---

## Prompt 1 — Jeu `PANEL` hétérogène : `climat_affaires` absente pour `IT` (L0)

**Modèle : Sonnet · Plan mode : Non · Dépendances : aucune · Parallélisable avec tout**

````text
Contexte : je prépare l'implémentation de HighFrequencyImputer2, spécifiée dans
high_frequency_imputer2_architecture.md (référence unique, notée [SPEC]). Tous les tests et
notebooks à venir s'appuient sur un jeu de panel comportant une feature structurellement absente
pour une entité — c'est le cas d'usage du paramètre `covariate_eligibility` ([SPEC] §4.5) et la
limite documentée de la stratégie 'interpolate' ([SPEC] §3, exception à l'invariant central).
Ce lot ne touche AUCUN fichier de tsforecast/ : il ne produit que des données d'exemple.
Référence : [SPEC] §2.3 (jeu de référence PANEL) et §15.1.

PARTIE A — notebooks/3 - QB - Panel a frequences mixtes heterogene.ipynb

Dans la fonction `create_panel_dataset` :

1) Ajouter une colonne `climat_affaires`, indicateur d'enquête de conjoncture **mensuel**,
   présent pour les entités FR et DE, et **entièrement NaN pour IT**. La colonne EXISTE dans le
   frame pour les trois entités : c'est l'entité italienne qui ne l'observe jamais, jamais la
   colonne qui est absente du schéma. Ne pas la faire disparaître par un dropna en aval.
   Valeurs FR/DE : niveau autour de 100 avec un bruit reproductible (réutiliser le mécanisme de
   graine déjà en place dans la fonction — ne pas introduire un second générateur aléatoire).

2) Compléter la section 2.3 de vérification du notebook : afficher, par entité, le nombre
   d'observations et le taux de NaN de CHAQUE colonne, de sorte que le zéro observation d'IT sur
   `climat_affaires` soit visible immédiatement. Ajouter une cellule markdown d'une phrase
   expliquant pourquoi ce jeu existe (support de `covariate_eligibility`, mesure PAR ENTITÉ de
   l'invariant NaN).

3) Ré-exécuter le notebook en entier et vérifier qu'aucune cellule aval ne casse. Le jeu est
   consommé par notebooks/4 - QB - HighFrequencyImputer pas a pas.ipynb : le vérifier également
   et corriger ce qui casse, SANS changer l'intention des cellules existantes.

PARTIE B — tests/frequency/conftest.py

4) Ajouter une fixture `mixed_freq_panel_heterogeneous` construite par une fonction
   `_build_panel_heterogeneous(seed: int = 42)`, sur le modèle exact de `_build_panel` /
   `mixed_freq_panel` déjà présents (mêmes conventions d'index, de nommage de colonnes et de
   graine). Le jeu doit être le jeu PANEL de [SPEC] §2.3 :
   - trois entités 'FR', 'DE', 'IT' ;
   - index mensuel fin de mois, 2021-01-31 -> 2023-12-31 (36 lignes par entité) ;
   - `m1` mensuelle complète, `q1` trimestrielle (ancres fins de trimestre), `a1` et `a2`
     annuelles (ancres 2021-12-31, 2022-12-31, 2023-12-31) ;
   - `climat_affaires` mensuelle pour FR et DE, entièrement NaN pour IT.
   Toutes les colonnes sont additives (une valeur annuelle est la SOMME de ses sous-périodes).

5) Ajouter également une fixture `reference_timeseries` correspondant EXACTEMENT au jeu TS de
   [SPEC] §2.2, avec les valeurs d'or du document : `a1` = 120 / 132 / 150 et `a2` = 60 / 66 / 72
   aux trois ancres annuelles. `m1` et `q1` peuvent être quelconques mais doivent être
   déterministes et JAMAIS NaN pour `m1`. Ces valeurs sont reprises telles quelles comme cas d'or
   par les tests des lots suivants : elles ne doivent plus bouger après ce lot.

6) Deux tests de fixture, dans un nouveau fichier tests/frequency/test_reference_datasets.py :
   - `test_heterogeneous_panel_it_has_no_climat_affaires` : la colonne existe, IT a 0 observation,
     FR et DE en ont 36 chacune ;
   - `test_reference_timeseries_matches_spec_anchors` : les six valeurs d'or de `a1` et `a2` sont
     exactement celles du §2.2, aux trois dates d'ancre.

Enfin : `uv run tests/frequency/check_regressions.py` et rapporter. Aucun nouvel échec attendu.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 2 — `conv` : décompte calendaire de `full_periods_only` (L1a, B26)

**Modèle : Opus · Plan mode : OUI · Dépendances : aucune**

> Plan mode justifié : le correctif change le résultat de **toute** agrégation par somme du
> package (`FrequencyAligner`, `HighFrequencyImputer`, conversions directes). L'ampleur de la
> dérive numérique doit être mesurée **avant** écriture, pas découverte dans la suite de tests.

````text
Contexte : je prépare l'implémentation de HighFrequencyImputer2, spécifiée dans
high_frequency_imputer2_architecture.md ([SPEC]). Son §9.3 et son §15 (prérequis P1) désignent ce
correctif comme un PRÉREQUIS DUR : hfi2 s'appuie sur les mêmes agrégations, et le défaut efface
silencieusement toute covariable journalière du jeu d'entraînement.
Référence complémentaire : high_frequency_imputer_annotations_architecture.md §1bis, entrée B26.

LE DÉFAUT — tsforecast/utils/frequency/converter.py, méthode `_require_full_subperiod_coverage`,
chemin SOMME. Le nombre de sous-périodes attendues est calculé avec un facteur CONSTANT :

    expected_count = int(round(get_duration_conversion_factor(target_base, source_base)))

`get_duration_conversion_factor('M', 'D')` vaut 30.0. Une agrégation D->M exige donc 30 jours par
mois : FÉVRIER (28 jours) est écarté toutes les années. Le chemin BOOLÉEN de la MÊME méthode a
déjà été migré et utilise `self.count_subperiods_per_period(result.index, target_base,
source_base)` — c'est une incohérence interne au fichier, pas un choix.

LA CONSÉQUENCE, qui est le vrai sujet : sur une grille journalière, la perte de février casse la
régularité de la série intermédiaire ; l'agrégation suivante M->Y n'y trouve plus que 11 mois sur
12 et jette l'ANNÉE ENTIÈRE. Toute covariable journalière disparaît alors du jeu d'entraînement
et l'étape bascule en repli par interpolation, avec pour seul indice « 0 usable covariate(s) ».

CIBLE

1) Aligner le chemin somme sur le chemin booléen : `count_subperiods_per_period`, qui rend un
   décompte PAR PÉRIODE (février 28 ou 29, janvier 31, trimestres 90/91/92, années bissextiles
   366). Le repli documenté sur le facteur constant reste en place quand les bases de fréquence
   ne s'expriment pas en `pd.Period` — la méthode le gère déjà elle-même, ne pas le dupliquer.

2) `expected_count` devient une `pd.Series` alignée sur l'index du résultat. La comparaison
   `valid_counts >= expected_counts` se fait élément par élément, et colonne par colonne pour un
   DataFrame (`.ge(..., axis=0)`), exactement comme au chemin booléen. Vérifier qu'aucun
   `int(round(...))` résiduel ne subsiste sur ce chemin.

3) Mettre à jour la docstring de la méthode : elle doit dire que le décompte est calendaire et
   par période, et nommer le cas de repli.

MESURER AVANT D'ÉCRIRE (à présenter dans le plan) : combien de tests de tests/frequency/ changent
de valeur, et de combien. Le correctif fait RÉAPPARAÎTRE des périodes aujourd'hui absentes, donc
des observations d'entraînement supplémentaires : les prédictions bougent partout où une grille
journalière ou hebdomadaire est en jeu. C'est un changement de comportement ATTENDU ET VOULU.

TESTS — dans tests/frequency/test_converter_subperiods.py (fichier existant, ne pas créer
tests/utils/frequency/test_converter.py : la convention du dépôt place les tests de converter
dans tests/frequency/) :
   - `test_february_survives_daily_to_monthly_sum` : sur 2015-2023, les 12 mois de chaque année
     sont agrégés, pas 11 ; les 9 févriers ne sont plus écartés ;
   - `test_leap_year_february_counted` : 2016 et 2020, 29 jours attendus ;
   - `test_quarter_lengths_respected` : T1 = 90 ou 91, T2 = 91, T3 = T4 = 92 selon l'année ;
   - `test_partial_period_still_rejected` : un mois amputé de 3 jours reste bien écarté — le
     correctif ne relâche pas le contrôle, il le rend exact ;
   - `test_daily_covariate_reaches_training_set` : test d'intégration, covariable journalière,
     variable annuelle, étape mensuelle, via HighFrequencyImputer — X_train porte des valeurs et
     l'étape ne bascule pas en repli.

À VÉRIFIER AU PASSAGE, sans forcément le corriger dans ce lot : `_aggregate_series` dans
tsforecast/frequency/frequency_aligner.py commente « la fréquence source doit être celle de la
variable, pas celle de l'index » et s'appuie sur `_observed_series`. Au SECOND passage
d'agrégation, l'entrée est déjà réindexée sur la grille d'origine et la fréquence source est
redétectée comme 'D' au lieu de 'M'. Dire si le correctif B26 rend le point sans objet ou s'il
faut un lot dédié — ne pas l'élargir ici.

ATTENTION : les tests de non-régression comparant des valeurs numériques doivent être MIS À JOUR
en connaissance de cause, PAS relâchés en tolérance. Me lister explicitement ceux que vous
modifiez et pourquoi. Puis régénérer tests/frequency/BASELINE_FAILURES.txt et rapporter le
nouveau compte avec `uv run tests/frequency/check_regressions.py`.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 3 — `conv` : `interpolate_to_higher_frequency(anchor_fraction=…)` (L1b, P2)

**Modèle : Opus · Plan mode : OUI · Dépendances : prompt 2 (même fichier)**

> Plan mode justifié : l'option modifie la sémantique d'index d'une méthode utilisée partout dans
> `hfi` ; la mécanique « décaler, interpoler sur l'union, réindexer » doit être validée avant
> écriture, en particulier son interaction avec `_reanchor_index_to_target` et
> `_resolve_limit_direction`.

````text
Contexte : je prépare l'implémentation de HighFrequencyImputer2 ([SPEC] =
high_frequency_imputer2_architecture.md). Son §10.2 spécifie un paramètre `interpolation_anchor`
qui repose sur une capacité nouvelle du convertisseur, désignée comme prérequis P2 au §15.
Ce lot livre UNIQUEMENT la capacité utilitaire, testable isolément. Il ne touche pas à hfi.
Référence : [SPEC] §10.2, intégralement — le lire avant d'écrire.

CIBLE — tsforecast/utils/frequency/converter.py, méthode `interpolate_to_higher_frequency`

1) Nouvel argument optionnel, ajouté EN FIN de signature pour ne casser aucun appelant positionnel :

       def interpolate_to_higher_frequency(self, data, target_freq, method='linear',
                                           limit=None, limit_direction=None, limit_area=None,
                                           source_freq=None,
                                           anchor_fraction: Optional[float] = None): ...

   `None` (défaut) = comportement ACTUEL, strictement inchangé : la valeur s'applique à la date à
   laquelle elle est référencée (début ou fin de période, selon la position de l'index). Le
   chemin `None` ne doit exécuter aucun code nouveau — vérifier par lecture, pas par test seul.

2) Sémantique de `anchor_fraction: float ∈ [0, 1]` : la valeur observée d'une période est
   considérée ATTEINTE à la fraction correspondante de la période qu'elle couvre. 0.0 = début de
   période, 0.5 = milieu, 1.0 = fin. Algorithme, dans cet ordre exact ([SPEC] §10.2) :
     a. décaler l'index ré-ancré produit par `_reanchor_index_to_target` vers
        `début_période + anchor_fraction × durée_période` ;
     b. interpoler sur l'UNION (index décalé ∪ grille cible) ;
     c. ne retenir que la grille cible.
   Les timestamps décalés ne tombent en général PAS sur la grille cible : l'union est
   indispensable, et les valeurs d'ancre elles-mêmes ne survivent pas telles quelles dans la
   sortie — c'est le but recherché (à 0.5, la valeur de mars ne prétend plus valoir au 31 mars).
   Ne pas « rattraper » ce point par une réinjection des valeurs d'ancre : ce serait le bug.

3) `durée_période` se calcule sur la période SOURCE réelle (calendaire), pas sur une durée
   moyenne : réutiliser les primitives existantes du fichier (`pd.Period`,
   `count_subperiods_per_period`, `_reanchor_index_to_target`) plutôt que d'introduire un
   décompte maison. Si la base de fréquence source ne s'exprime pas en `pd.Period`, retomber
   proprement sur le comportement `anchor_fraction=None` et le documenter ou si cela te semble préférable utiliser le facteur de conversion non calendaire.

4) `limit_direction` reste résolu par `_resolve_limit_direction` avec ses défauts actuels. Aux
   bords de série, au-delà de la dernière ancre décalée, c'est lui qui décide — le documenter
   dans la docstring, `Args:` de `anchor_fraction`.

5) Validation : `anchor_fraction` doit être `None` ou un float dans [0, 1] ; hors bornes ->
   `ValueError` nommant la valeur reçue et l'intervalle admis. Pas de coercition silencieuse.

6) Docstring `Examples:` : reprendre l'exemple chiffré du §10.2 de [SPEC] — `a1` annuelle
   (120 en 2021, 132 en 2022) interpolée vers 'Q', method='linear' :
     - `anchor_fraction=None` : ancres au 2021-12-31 et 2022-12-31, valeurs Q 2022 avant recalage
       123, 126, 129, 132 ;
     - `anchor_fraction=0.5` : ancres au 2021-07-02 et 2022-07-02, T1 2022 interpolé/extrapolé
       entre les deux points décalés.
   Préciser aussi que ce paramètre s'applique à l'INTERPOLATION seulement — jamais à
   l'agrégation, ni à la désagrégation par totaux, qui restent ancrées sur les périodes exactes.

TESTS — nouveau fichier tests/frequency/test_converter_anchor_fraction.py :
   - `test_anchor_fraction_none_is_unchanged` : égalité STRICTE avec la sortie actuelle sur trois
     conversions (Y->Q, Y->M, Q->M) — c'est le test de non-régression du lot ;
   - `test_anchor_fraction_zero_shifts_to_period_start` et
     `test_anchor_fraction_one_shifts_to_period_end` : positions d'ancre attendues, vérifiées sur
     l'index intermédiaire ;
   - `test_anchor_fraction_half_uses_mid_period` : sur l'exemple Y->Q du §10.2, les ancres tombent
     au 2021-07-02 / 2022-07-02 et les quatre valeurs trimestrielles diffèrent de celles du cas
     None ;
   - `test_anchor_fraction_union_index_is_used` : la sortie est indexée EXACTEMENT sur la grille
     cible (aucun timestamp décalé résiduel) ;
   - `test_anchor_fraction_out_of_range_raises` : -0.1 et 1.5 lèvent ValueError ;
   - `test_anchor_fraction_edges_follow_limit_direction` : au-delà de la dernière ancre décalée,
     le comportement suit `limit_direction`.

Puis `uv run tests/frequency/check_regressions.py` et rapporter. Aucun nouvel échec attendu : ce
lot est purement additif.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 4 — `provenance.py` : échelle de souillure et `resolve_model_provenance` (L2, P4)

**Modèle : Sonnet · Plan mode : Non · Dépendances : aucune**

> Plan mode inutile : le résultat attendu est entièrement décrit ci-dessous (énumération exacte,
> fonction de correspondance donnée telle quelle, liste exhaustive des sites d'appel à mettre à
> jour). La seule difficulté est la rigueur du renommage mécanique.

````text
Contexte : je prépare l'implémentation de HighFrequencyImputer2 ([SPEC] =
high_frequency_imputer2_architecture.md). Son §6 remplace la provenance binaire actuelle par une
ÉCHELLE DE SOUILLURE à cinq libellés MODEL_*, et supprime MODEL_ON_MIXED. L'énumération étant
PARTAGÉE avec HighFrequencyImputer, ce lot met aussi à jour hfi de façon purement mécanique.
C'est une RUPTURE ASSUMÉE, décidée explicitement par l'auteur (décision D6, [SPEC] §6.7).
Référence : [SPEC] §6 en entier (§6.1 à §6.7).

PARTIE A — tsforecast/frequency/provenance.py

1) `ProvenanceType` devient EXACTEMENT (valeurs de chaîne comprises) :

       ORIGINAL      = 'original'
       AGGREGATED    = 'aggregated'
       DISAGGREGATED = 'disaggregated'
       INTERPOLATED  = 'interpolated'                    # NOUVEAU
       MODEL_ON_TRUE           = 'model_on_true'
       MODEL_ON_INTERPOLATED   = 'model_on_interpolated'   # NOUVEAU
       MODEL_ON_IMPUTED        = 'model_on_imputed'        # NOUVEAU
       MODEL_ON_IMPUTED_TARGET = 'model_on_imputed_target' # NOUVEAU
       MODEL_ON_IMPUTED_BOTH   = 'model_on_imputed_both'   # NOUVEAU

   MODEL_ON_MIXED est SUPPRIMÉ. AUCUN alias de compatibilité : un MODEL_ON_MIXED résiduel doit
   produire un AttributeError franc, jamais un comportement silencieux.
   Mettre à jour la docstring de classe avec les cinq définitions du §6.1 de [SPEC] :
     - MODEL_ON_TRUE : le modèle n'a vu que des vraies valeurs, AGRÉGÉES COMPRISES (une
       agrégation additive exacte d'observations n'est pas une approximation) ;
     - MODEL_ON_INTERPOLATED : au moins une valeur interpolée parmi les covariables OU dans
       y_train, et aucune valeur de modèle ;
     - MODEL_ON_IMPUTED : au moins une covariable imputée par modèle ; y_train indemne ;
     - MODEL_ON_IMPUTED_TARGET : y_train contient au moins une valeur imputée par modèle, les
       covariables non ;
     - MODEL_ON_IMPUTED_BOTH : les deux.
   La docstring de DISAGGREGATED doit dire explicitement qu'elle est AMBIGUË quant au degré de
   confiance : elle décrit une POSITION (« sous-période d'un total observé ») autant qu'une
   origine, et ne doit JAMAIS servir de filtre pour composer y_train ni pour calculer une
   souillure ([SPEC] §6.4).

2) Deux alias de typage exportés par le module :

       CellOrigin = Literal['observed', 'interpolated', 'model']   # ordre croissant de souillure
       Taint      = Literal['none', 'interpolated', 'imputed']

3) Fonction de module `resolve_model_provenance`, à implémenter TELLE QUELLE ([SPEC] §6.3) :

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

   Ajouter une fonction utilitaire `origin_to_taint(origin: CellOrigin) -> Taint` réalisant la
   correspondance 'observed'->'none', 'interpolated'->'interpolated', 'model'->'imputed', et une
   fonction `max_origin(origins: Iterable[CellOrigin]) -> CellOrigin` retournant le maximum sur
   l'ordre croissant de souillure (utilitaire dont les lots suivants ont besoin ; sur un
   itérable vide, retourner 'observed').

4) `ImputationProvenanceTracker.mark_model_imputed` : nouvelle signature ([SPEC] §6.6)

       def mark_model_imputed(self, column, index, covariate_taint: Taint = 'none',
                              target_taint: Taint = 'none') -> None:
           """Mark specific values as imputed by a model, at the given taint levels."""

   Le paramètre `trained_on_imputed: bool` DISPARAÎT. Le corps délègue à
   `resolve_model_provenance` puis à `mark_imputed`.

5) Ajouter une méthode `mark_interpolated(column, index)` écrivant ProvenanceType.INTERPOLATED,
   symétrique des `mark_aggregated` / `mark_disaggregated` existantes.

6) Balayer le module : docstring de module, `Examples:` de `mark_imputed`, `get_mask`,
   `compute_statistics`, `to_string_matrix`. Toute occurrence de MODEL_ON_MIXED devient
   MODEL_ON_IMPUTED. Vérifier que `compute_statistics` et `to_string_matrix` traitent les cinq
   nouvelles constantes sans liste codée en dur qui en oublierait — si une liste explicite
   existe, la remplacer par une itération sur `ProvenanceType`.

PARTIE B — mise à jour mécanique de HighFrequencyImputer ([SPEC] §6.7)

7) `ImputationStep.trained_on_imputed` RESTE dans tsforecast/frequency/imputation_plan.py tant
   que hfi existe : ne pas y toucher. Seuls les APPELS changent. Sur chaque site passant
   `trained_on_imputed=` à `mark_model_imputed` (notamment `_mark_predictions_provenance`,
   `_apply_step_predictions` et leurs appelants), écrire :

       covariate_taint='imputed' if step.trained_on_imputed else 'none'

   Justification à mettre en commentaire français : la sémantique d'origine de hfi
   (`trained_on_imputed = train_on_partial_coverage and bool(imputed_store)`) porte bien sur les
   COVARIABLES, pas sur la cible.

8) Rechercher MODEL_ON_MIXED dans TOUT le dépôt (tsforecast/, tests/, notebooks/, docs/) et
   traiter chaque occurrence :
   - tests/frequency/test_high_frequency_imputer.py : les occurrences deviennent
     MODEL_ON_IMPUTED ; le test `test_transform_fallback_marks_trained_on_imputed_false` est
     RENOMMÉ et vérifie le comportement actuel de hfi (hfi2 fera porter INTERPOLATED aux cellules
     de repli, hfi garde son comportement) ;
   - notebooks 4 et suivants : remplacer le libellé dans les affichages de provenance ;
   - docs/ et docstrings : idem.
   Me lister les fichiers touchés et le nombre d'occurrences par fichier.

PARTIE C — tests, nouveau fichier tests/frequency/test_provenance.py

9) La suite couvre :
   - `test_resolve_model_provenance_table` : les NEUF cases du tableau 3x3 du §6.3 de [SPEC],
     écrites une par une, pas générées par une boucle sur la fonction testée ;
   - `test_model_on_mixed_is_gone` : `getattr(ProvenanceType, 'MODEL_ON_MIXED', None) is None`
     et `'model_on_mixed'` n'est la valeur d'aucun membre ;
   - `test_mark_model_imputed_defaults_to_model_on_true` ;
   - `test_mark_model_imputed_taints` : les cinq libellés sont atteignables via la nouvelle
     signature ;
   - `test_mark_interpolated_writes_interpolated` ;
   - `test_origin_to_taint_and_max_origin` : correspondance et ordre de souillure ;
   - `test_statistics_cover_all_provenance_types` : aucune constante oubliée.

Puis `uv run tests/frequency/check_regressions.py`. Ce lot casse volontairement des tests de hfi :
les corriger comme indiqué au point 8, PUIS régénérer tests/frequency/BASELINE_FAILURES.txt et me
donner le compte final ainsi que la liste des tests modifiés avec la raison de chacun.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 5 — `iwc` : type de retour `pd.Series` MultiIndex unifié (L3, P3)

**Modèle : Sonnet · Plan mode : Non · Dépendances : aucune**

> Plan mode inutile : le lot est réduit à une unification de type de retour, à surface fermée.
> ⚠️ **Le reste du lot L3 est déjà fait** — voir « État vérifié du dépôt » : les trois masques,
> `training_scope`, `training_coverage_threshold`, `kind=` et les correctifs B23/B24 existent.

````text
Contexte : je prépare l'implémentation de HighFrequencyImputer2 ([SPEC] =
high_frequency_imputer2_architecture.md). Son §7.2 impose un TYPE DE RETOUR UNIFIÉ pour les
masques de fenêtre sur panel : des `pd.Series` booléennes à MultiIndex (entity..., date), plus
jamais des `Dict[entity, Series]`. C'est le prérequis P3 de [SPEC] §15.

ÉTAT VÉRIFIÉ, à ne pas refaire : tsforecast/frequency/imputation_window.py expose DÉJÀ les trois
masques (`imputation_strict_window_mask_`, `imputation_window_mask_`, `training_window_mask_`),
les paramètres `training_scope` / `training_coverage_threshold`, les helpers
`_effective_training_scope`, `_effective_training_coverage_threshold`, `_build_scope_mask`,
`_select_mask`, l'argument `kind=` sur `get_imputation_window_mask` ET `get_mask_at_frequency`,
et les correctifs B23/B24. NE PAS y toucher.

CE QUI RESTE À FAIRE

1) Les trois attributs de masque, ainsi que `coverage_by_date_`, stockent aujourd'hui, sur panel,
   un `Dict[tuple, Optional[pd.Series]]`. Les faire porter une `pd.Series` booléenne unique à
   MultiIndex (niveaux d'entité..., puis date), dans l'ordre des niveaux du frame d'entrée.
   - Une entité dont la fenêtre est indéterminée (branche « fréquence d'index non identifiée » de
     `_fit_panel`) contribue ses lignes avec la valeur False, PAS une absence de lignes : la
     Series doit couvrir l'intégralité de l'index du frame ajusté. C'est ce qui permet aux
     appelants d'aligner sans reconstruire.
   - Conserver l'information « fenêtre indéterminée » dans un attribut dédié —
     `entities_without_window_: Tuple[tuple, ...]` — plutôt que par un None noyé dans un dict.
   - Les attributs de BORNES (`imputation_window_start_`/`_end_`,
     `imputation_strict_window_start_`/`_end_`) restent des dicts par entité : ce sont des
     scalaires, pas des masques. Ne pas les convertir.
   - `column_coverage_` reste inchangé.

2) `get_imputation_window_mask(data=None, kind=...)` : la logique d'alignement (réindexation TS,
   reconstruction MultiIndex panel) se SIMPLIFIE puisque la source est déjà une Series à
   MultiIndex — factoriser, ne pas dupliquer par `kind`. Le contrat public est inchangé du point
   de vue des appelants qui passaient déjà `data` (ils recevaient déjà une Series alignée) ;
   c'est l'appel SANS `data` sur panel qui change de type. Le signaler dans la docstring et dans
   les `Returns:`.

3) `get_mask_at_frequency(frequency, kind=...)` et `_convert_mask_to_frequency` : même
   unification. [SPEC] §7.2 demande de vérifier au passage si
   `FrequencyConverter.convert_frequency` sait traiter des fréquences cibles PAR ENTITÉ ; si oui,
   déléguer ; sinon CONSERVER la boucle interne par entité mais unifier le TYPE DE RETOUR.
   Me dire laquelle des deux voies a été retenue et pourquoi.

4) Recenser et mettre à jour tous les consommateurs internes au dépôt : `hfi` (sites d'appel de
   `get_imputation_window_mask` / `get_mask_at_frequency`, y compris `_prepare_training_data`,
   `_prediction_masks`, `_determine_variable_order_cv`), tests, notebooks. Le comportement
   fonctionnel de hfi doit être STRICTEMENT inchangé : ce lot est un changement de
   représentation, pas de sémantique. Toute différence de valeur produite est une régression.

5) Mettre à jour la docstring de classe, les descriptions d'attributs et les `Returns:` des
   méthodes touchées.

TESTS — dans tests/frequency/test_imputation_window.py (fichier existant) :
   - `test_panel_masks_are_multiindex_series` : les trois masques sont des Series booléennes à
     MultiIndex couvrant tout l'index du frame ajusté, pour les trois valeurs de `kind` ;
   - `test_entity_without_window_is_all_false` : une entité sans fenêtre déterminable a toutes
     ses lignes à False, et figure dans `entities_without_window_` ;
   - `test_timeseries_masks_unchanged` : sur série temporelle, aucun changement de type ni de
     valeur ;
   - `test_get_mask_at_frequency_returns_multiindex_series` ;
   - `test_window_bounds_still_dict_per_entity` : les bornes n'ont PAS été converties.

Puis `uv run tests/frequency/check_regressions.py` et rapporter. Aucun nouvel échec attendu :
si un test de hfi change de résultat, c'est un bug de ce lot, pas un test à relâcher.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 6 — `imputation_plan2.py` : `ImputationStep` v2 et plan immuable (L4)

**Modèle : Sonnet · Plan mode : Non · Dépendances : prompt 4**

> Plan mode inutile : la liste des champs est donnée exhaustivement ; le lot est une création de
> dataclass sans logique métier.

````text
Contexte : je prépare l'implémentation de HighFrequencyImputer2 ([SPEC] =
high_frequency_imputer2_architecture.md). L'architecture repose sur la séparation « un plan, un
exécuteur » ([SPEC] §12.1) : le PLAN est l'état ajusté complet, rejoué tel quel au transform.
Ce lot crée la structure de plan de hfi2, à côté de celle de hfi, qui reste INTACTE.
Référence : [SPEC] §12.1, §12.2, §4.6, §6.2, §13.2.

PRÉREQUIS déjà en place : tsforecast/frequency/provenance.py expose `CellOrigin`, `Taint`,
`resolve_model_provenance`, `origin_to_taint`, `max_origin` (lot précédent).

CIBLE — nouveau fichier tsforecast/frequency/imputation_plan2.py

1) `MaterializationWay = Literal['identity', 'aggregate', 'stage_model', 'carried_model',
   'interpolate', 'raw_anchors']` — les six voies du §4.6 de [SPEC], dans cet ordre, avec un
   commentaire français rattachant chacune à son rang de précédence du §4.4 :
     - 'identity'      : rang 1, f_c == f
     - 'aggregate'     : rang 1, f_c plus fine que f, agrégation exacte
     - 'stage_model'   : rang 2, imputée à l'étape courante
     - 'carried_model' : rang 3, imputée à une étape antérieure puis REPORTÉE
     - 'interpolate'   : rang 4, covariate_fallback='interpolate'
     - 'raw_anchors'   : rang 4, covariate_fallback='tolerate_nan', ou stratégie 'tolerate_nan'

2) Dataclass `ImputationStep` (v2), IMMUABLE (`frozen=True`), portant :

   Repris de la v1 (mêmes noms, même rôle) :
       pred_freq_label, pred_freq, var_key, var_name, model, feature_cols (Tuple[str, ...]),
       scale_factor, fit_scale_factor, source_frequency, entities

   Modifiés :
       scale_factor:     Union[float, pd.Series]     # §5.4 : diviseur PAR LIGNE possible
       fit_scale_factor: Union[float, pd.Series]

   Ajoutés :
       covariate_taint: Taint                        # §6.2
       target_taint:    Taint                        # §6.2
       materialization: Mapping[str, MaterializationWay]   # §4.6, une entrée par feature_col
       is_fallback:     bool                         # §6.4 : CHAMP, plus une propriété dérivée
       interpolation_method: str                     # méthode retenue pour CETTE variable
       interpolation_anchor: Optional[float]         # ancrage retenu pour CETTE variable

   Supprimés : `trained_on_imputed` (remplacé par le couple de souillures), et `feature_means`
   s'il existe quelque part dans la v1 (vérifier ; il n'est pas dans le dataclass actuel).

   Points d'implémentation :
   - `is_fallback` devient un CHAMP parce qu'une étape peut être en repli pour d'autres raisons
     que `model is INTERPOLATE_FALLBACK` ; conserver néanmoins la cohérence : si
     `model is INTERPOLATE_FALLBACK` alors `is_fallback` est True (à vérifier dans
     `__post_init__`, avec un ValueError explicite si l'invariant est rompu).
   - `materialization` doit être immuable et couvrir EXACTEMENT `feature_cols` : le vérifier en
     `__post_init__` (clés manquantes ou en trop -> ValueError nommant les colonnes fautives).
     Stocker via `types.MappingProxyType` sur un dict interne, ou un `Tuple[Tuple[str, str], ...]`
     exposé par une propriété — retenir la forme qui reste hashable et lisible en debug.
   - `scale_factor`/`fit_scale_factor` pouvant être des `pd.Series`, le dataclass ne peut pas
     utiliser l'égalité générée par défaut : poser `eq=False` et écrire un `__eq__` explicite
     comparant les Series par `.equals()`, ou documenter clairement pourquoi l'égalité n'est pas
     définie. Ne pas laisser un `==` qui lève sur une Series.
   - Conserver la propriété `stage_key -> Tuple[FrequencyLabel, GroupKey]` de la v1.

3) `ImputationPlan` : conteneur immuable enveloppant `Tuple[ImputationStep, ...]`, avec :
       - `steps` en lecture seule, itération et indexation ;
       - `by_stage() -> Dict[FrequencyLabel, Tuple[ImputationStep, ...]]`, préservant l'ordre ;
       - `models() -> Dict[Tuple[FrequencyLabel, str], Any]` — la vue `imputation_models_` de
         [SPEC] §13.2 ;
       - `to_diagnostic_frame() -> pd.DataFrame` : UNE LIGNE PAR ÉTAPE, colonnes
         `stage`, `variable`, `n_features`, `covariate_taint`, `target_taint`,
         `emitted_provenance` (via `resolve_model_provenance`), `is_fallback`,
         `interpolation_method`, `interpolation_anchor`, `materialization` (rendu lisible, une
         chaîne `col=way` séparée par des virgules). C'est la sérialisation de diagnostic
         demandée par [SPEC] §17 (L4) et le support de la cellule pas-à-pas du notebook 5.

4) Fonction `append_step(plan, step) -> ImputationPlan` renvoyant un NOUVEAU plan : le plan ne se
   mute jamais en place. La construction incrémentale de la PHASE 5 passera par elle.

5) NE PAS toucher à tsforecast/frequency/imputation_plan.py : hfi et hfi2 coexistent ([SPEC]
   §12.2 et §15.3). Ne pas non plus factoriser une base commune entre les deux ImputationStep :
   la v1 est destinée à disparaître, une hiérarchie partagée créerait un couplage à défaire.

6) Exporter `ImputationStep` (v2), `ImputationPlan`, `MaterializationWay` depuis
   tsforecast/frequency/__init__.py SOUS DES NOMS DISTINCTS de ceux de la v1 — proposer
   `ImputationStep2` / `ImputationPlan2` à l'export et me dire si un autre nommage vous paraît
   meilleur, sans changer le nom de la classe DANS le module (elle s'y appelle `ImputationStep`).

TESTS — nouveau fichier tests/frequency/test_imputation_plan2.py :
   - `test_step_is_frozen` : toute affectation lève ;
   - `test_materialization_must_cover_feature_cols` : clé manquante et clé en trop lèvent
     ValueError avec le nom de la colonne fautive ;
   - `test_fallback_invariant` : `model is INTERPOLATE_FALLBACK` avec `is_fallback=False` lève ;
   - `test_scale_factor_accepts_series` : construction et égalité avec une `pd.Series` ;
   - `test_plan_is_immutable_and_append_returns_new` ;
   - `test_by_stage_preserves_order` ;
   - `test_diagnostic_frame_columns_and_emitted_provenance` : sur trois étapes de souillures
     différentes, la colonne `emitted_provenance` vaut ce que donne `resolve_model_provenance` ;
   - `test_v1_step_untouched` : l'import de `tsforecast.frequency.imputation_plan.ImputationStep`
     fonctionne toujours et porte encore `trained_on_imputed`.

Puis `uv run tests/frequency/check_regressions.py` et rapporter. Lot purement additif : aucun
nouvel échec attendu.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 7 — `stage_scaler.py` : diviseurs `'constant'` / `'calendar'` (L5)

**Modèle : Opus · Plan mode : OUI · Dépendances : prompts 2 et 3**

> Plan mode justifié : la mise à l'échelle est le premier endroit où une erreur produit des
> résultats faux sans crash. Trois diviseurs distincts (covariables, `y`, report des prédictions),
> deux modalités, une forme scalaire et une forme `Series` par ligne : la répartition et les
> signatures doivent être arrêtées avant écriture.

````text
Contexte : je prépare l'implémentation de HighFrequencyImputer2 ([SPEC] =
high_frequency_imputer2_architecture.md). Ce lot extrait la mise à l'échelle dans un composant
autonome et l'étend à deux modalités, éventuellement par feature.
Référence : [SPEC] §9 en entier, §5.4 (piège d'échelle des lignes imputées), §12.2 (tableau des
composants). Lire aussi, dans hfi, `_covariate_scaling_divisors` et les sites consommant
`scale_factor` / `fit_scale_factor` : la logique EXISTE, ce lot la déplace, la généralise et la
corrige — il ne la réinvente pas.

CIBLE — nouveau fichier tsforecast/frequency/stage_scaler.py, classe `StageScaler`

1) Modalités ([SPEC] §9.1) :

       ScaleMode = Literal['constant', 'calendar']
       scale_features: Union[Literal[False], ScaleMode,
                             Dict[str, Union[Literal[False], ScaleMode]]] = 'constant'

   - `False` : AUCUN diviseur, ni sur les features couvertes, ni sur `y` quand c'est la modalité
     résolue pour la colonne imputée (décision D15 : rupture assumée avec le `False` de hfi — la
     cible est une colonne comme une autre, un indice ou un taux n'a pas à être divisé) ;
   - `'constant'` (défaut) : diviseur constant par couple de fréquences, via
     `FrequencyConverter.get_conversion_factor` (M->Y = 12, D->M = 30.0…). Cas d'usage : variables
     corrigées des variations saisonnières ;
   - `'calendar'` : diviseur par décompte calendaire réel, via
     `FrequencyConverter.count_subperiods_per_period` (février 28 ou 29, T1 90 ou 91). Produit une
     `pd.Series` de diviseurs PAR LIGNE. Cas d'usage : variables brutes non CVS.

   Forme dict `{feature: modalité}` avec clé `'__default__'` optionnelle, même convention que
   `estimator` dans hfi. Clés inconnues au fit -> `ValueError` LISTANT les colonnes fautives.
   Features non couvertes et pas de `'__default__'` -> défaut `'constant'`. Les clés sont des
   NOMS DE COLONNE, jamais des couples (entité, colonne) — décision D10 de [SPEC].

2) Les TROIS diviseurs gouvernés par la modalité ([SPEC] §9.2), à exposer par trois méthodes
   distinctes et nommées :

   a. `feature_divisors(...)` — diviseur des covariables à l'entraînement ET à la prédiction.
      RÈGLE B25, à reprendre telle quelle : `1.0` pour une colonne JAMAIS ré-agrégée (ses ancres
      gardent l'échelle de f_c), `get_conversion_factor(f_stage, f_var)` sinon, avec
      `f_stage = pred_freq` si f_var est plus FINE que l'étape, et `f_var` sinon.
   b. `target_divisor(...)` — diviseur de `y`. Scalaire d'étape, OU `pd.Series` par ligne dès que
      `y_train` mêle plusieurs fréquences de production ([SPEC] §5.4). La modalité appliquée est
      celle de la COLONNE IMPUTÉE, résolue comme celle de n'importe quelle colonne : `False` y
      rend `1.0`, fréquences de production comprises. La colonne imputée est désignée par le NOM
      DE `y` reçu au `fit` — PAS de paramètre `target_column`, il dupliquerait ce que `y` porte ;
      une cible anonyme retombe sur le réglage global. La méthode reste DISTINCTE de
      `feature_divisors` : la règle B25 rend `1.0` pour une colonne à sa propre fréquence, alors
      que la cible, produite sur la grille d'étape, y porte `pred_freq` (§9.2).
   c. `fit_scale_factor(...)` — le facteur CUIT DANS LE MODÈLE, qui ne bouge plus une fois l'étape
      ajustée, et qui sert au report d'échelle des prédictions.

   Chaque méthode retourne `Union[float, pd.Series]`. Une seule routine interne partagée applique
   et inverse l'échelle : `apply(values, divisor)` et `invert(values, divisor)`.

3) CORRECTIF B12, impératif : ne PAS court-circuiter la mise à l'échelle quand le facteur est une
   `pd.Series` dont toutes les valeurs valent 1.0 par hasard. Le court-circuit n'est licite que
   sur un scalaire strictement égal à 1.0. Écrire un helper unique `_is_identity(divisor)` et
   l'utiliser partout — c'est le seul endroit où la question se pose.

4) Échelle PAR LIGNE ([SPEC] §5.4) : `target_divisor` accepte une `pd.Series[str]` donnant, pour
   chaque ligne de `y_train`, la FRÉQUENCE À LAQUELLE ELLE A ÉTÉ PRODUITE (lue plus tard dans
   `imputed_freq_store`). Le diviseur de la ligne est `get_conversion_factor(pred_freq, f_ligne)`
   sous `'constant'`, son équivalent calendaire sous `'calendar'`. Reproduire l'exemple chiffré du
   §5.4 dans la docstring `Examples:` :
       ligne 2021-12-31, origine 'observed', produite en Y, valeur 120, diviseur factor(M, Y)=12
           -> 10.0
       ligne 2021-03-31, origine 'model', produite en Q, valeur 28, diviseur factor(M, Q)=3
           -> 9.33
       ligne 2021-06-30, origine 'model', produite en Q, valeur 30, diviseur factor(M, Q)=3
           -> 10.0

5) Validation : `scale_features` est validé à l'INIT du StageScaler sans transformation (les
   clés dict ne sont vérifiées contre les colonnes réelles qu'à l'appel du fit). Un
   `resolve_mode(column) -> Union[Literal[False], ScaleMode]` public rend la modalité effective
   d'une colonne et sert aux diagnostics du notebook.

6) Le composant est SANS ÉTAT persistant autre que sa configuration : pas de cache d'index, pas
   de mémorisation entre étapes. Les facteurs cuits sont portés par l'ImputationStep, pas par le
   scaler ([SPEC] §12.2). Une même instance doit pouvoir servir toutes les étapes.

TESTS — nouveau fichier tests/frequency/test_stage_scaler.py :
   - `test_constant_divisors_match_conversion_factor` : M->Y = 12, Q->M = 3, D->M = 30.0 ;
   - `test_calendar_divisors_are_per_row` : février 28 / 29, T1 90 ou 91, longueur et index de la
     Series ;
   - `test_b25_never_reaggregated_column_divides_by_one` ;
   - `test_b25_finer_column_uses_pred_freq` et `test_b25_lower_column_uses_own_freq` ;
   - `test_b12_series_of_ones_is_not_short_circuited` : une Series de 1.0 passe bien par le
     chemin de mise à l'échelle (vérifié par un espion ou par le type de sortie), un scalaire 1.0
     court-circuite ;
   - `test_per_row_target_divisor` : reproduit EXACTEMENT le tableau du §5.4 (10.0 / 9.33 / 10.0,
     à 1e-2 près pour la seconde) ;
   - `test_dict_form_and_default_key` : `{'m1': 'calendar', '__default__': 'constant'}` ;
   - `test_unknown_dict_key_raises_listing_columns` ;
   - `test_scale_features_false_spares_y_too` : `False` ne divise ni les features ni la cible
     (D15), y compris avec des fréquences de production mêlées ; et la forme dict permet de
     dispenser la seule cible en divisant les features ;
   - `test_fit_reads_the_imputed_column_from_the_name_of_y` : le nom de `y` désigne la colonne
     imputée ; une cible anonyme retombe sur le réglage global ;
   - `test_apply_invert_roundtrip` : `invert(apply(v, d), d) == v` sur scalaire et Series.

Puis `uv run tests/frequency/check_regressions.py` et rapporter. Lot purement additif : aucun
nouvel échec attendu. NE PAS modifier hfi dans ce lot : `_covariate_scaling_divisors` y reste en
place tant que hfi existe.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 8 — `covariate_materializer.py` : socle, stores, `'tolerate_nan'`, `'interpolate'` (L6a)

**Modèle : Opus · Plan mode : OUI · Dépendances : prompts 3, 4, 5**

> Plan mode justifié : c'est le composant central de l'architecture — l'**unique** producteur de
> `X_train` et `X_pred`, dont dépend l'invariant du §3. Sa signature conditionne les lots 9, 12,
> 13, 14 et 15. À arrêter avant écriture.

````text
Contexte : je prépare l'implémentation de HighFrequencyImputer2 ([SPEC] =
high_frequency_imputer2_architecture.md). Ce lot crée le composant qui porte l'INVARIANT CENTRAL
de l'architecture ([SPEC] §3) : « un modèle ne voit jamais, à la prédiction, un motif de
disponibilité de features plus dégradé qu'à l'entraînement, ni des features de nature différente ».
Le corollaire structurel est qu'il n'existe qu'UNE SEULE méthode produisant X_train et X_pred.
Référence : [SPEC] §3, §4.1, §4.2, §4.3, §4.5, §6.2, §12.2. Lire ces sections avant d'écrire.
Ce lot NE TRAITE PAS la stratégie 'model' (rangs 2 et 3 de la précédence) : c'est le lot suivant.

PRÉREQUIS déjà en place : `conv.interpolate_to_higher_frequency(..., anchor_fraction=...)` ;
`provenance.py` avec `CellOrigin`, `Taint`, `origin_to_taint`, `max_origin`,
`ProvenanceType.INTERPOLATED` ; `iwc` rendant des masques `pd.Series` à MultiIndex.

CIBLE — nouveau fichier tsforecast/frequency/covariate_materializer.py

1) `CovariateMaterializer`, configuré par : `covariate_strategy`, `covariate_fallback`,
   `covariate_eligibility`, `interpolation_method`, `interpolation_anchor`,
   `aggregation_constraint`, plus les objets partagés (`FrequencyConverter`, éventuellement le
   `StageScaler`). `interpolation_method` et `interpolation_anchor` acceptent la forme scalaire ET
   la forme dict par feature avec `'__default__'` (mêmes règles de validation que `scale_features`,
   clés = noms de colonne, décision D10). Exposer `resolve_method(column)` et
   `resolve_anchor(column)`.

2) LES TROIS STORES, tenus par ce composant et par lui seul ([SPEC] §4.4, §6.2) :
       imputed_store:      Dict[str, pd.Series]        # miroir des valeurs imputées
       imputed_freq_store: Dict[str, pd.Series]        # fréquence de PRODUCTION de chaque cellule
       origin_store:       Dict[str, pd.Series]        # CellOrigin par cellule
   Règles d'écriture :
   - correctif B5 : toute écriture se fait par `predictions.combine_first(existing)`, jamais par
     une affectation qui écraserait des valeurs antérieures ;
   - règle « LE REPLI MATÉRIALISE » (troisième cause de B28) : une valeur produite par repli
     alimente les trois stores exactement comme une prédiction de modèle. C'est un point de
     conception, pas un détail : ne pas reproduire le comportement de hfi, où
     `_write_interpolation_fallback` n'alimente ni `imputed_store` ni le miroir.
   - `origin_store` : une cellule d'entrée non NaN vaut 'observed' ; une agrégation exacte vaut
     'observed' ; une interpolation ou un repli d'interpolation valent 'interpolated' ; une
     prédiction de modèle vaut 'model', Y COMPRIS après recalage aux totaux et Y COMPRIS après
     report d'étape.
   Fournir `reset()` (purge des trois stores) et `snapshot()` (copie profonde, pour le pas-à-pas).

3) CLASSIFICATION d'une covariable `c` face à une grille de fréquence `f` ([SPEC] §4.1) :
       f_c == f            -> 'identity',  origine 'observed'
       f_c plus FINE que f -> 'aggregate', origine 'observed', via
           FrequencyConverter.aggregate_to_lower_frequency(..., full_periods_only=True)
       f_c plus BASSE      -> gouverné par covariate_strategy
   Une période incomplète en agrégation produit NaN : c'est une source LÉGITIME de NaN, identique
   au fit et au predict. Ne pas la masquer.

4) MÉTHODE UNIQUE ET PUBLIQUE, seule productrice de features :

       def materialize(self, *, columns, grid_index, stage_freq, detected_frequencies,
                       source_data, materialization=None) -> Tuple[pd.DataFrame, Dict[str, MaterializationWay], Dict[str, CellOrigin]]

   - `materialization=None` : le composant CHOISIT la voie de chaque colonne et la retourne ;
   - `materialization` fourni : le composant REJOUE les voies imposées, sans rien re-décider.
     C'est ce second mode qui garantit la règle d'unicité du §4.6 et rend le test I11 possible :
     l'appelant choisit la voie UNE FOIS, sur la grille d'entraînement, puis l'impose à la grille
     de prédiction, au fit COMME au transform.
   - Le troisième élément du retour est l'origine AGRÉGÉE par colonne (max des origines des
     cellules effectivement produites), qui servira au calcul de `covariate_taint`.
   Aucune autre méthode publique ne doit pouvoir produire un DataFrame de features.

5) STRATÉGIE 'tolerate_nan' ([SPEC] §4.2) : aucune matérialisation. Une covariable plus basse
   fréquence que la grille porte ses valeurs aux SEULES dates-ancres, à l'échelle de sa propre
   fréquence, et NaN partout ailleurs — au fit COMME au predict. Voie 'raw_anchors'.
   La docstring doit énoncer le PRÉREQUIS DUR de la modalité : l'estimateur doit tolérer les NaN
   (HistGradientBoostingRegressor, LGBMRegressor, ou un Pipeline incluant un SimpleImputer).

6) STRATÉGIE 'interpolate' ([SPEC] §4.3), le défaut : toute covariable plus basse fréquence que la
   grille est interpolée sur la grille à partir de ses valeurs observées (les ancres servant de
   points de référence), avec `resolve_method(c)` et `resolve_anchor(c)`, PUIS recalée pour
   préserver les totaux de période quand `aggregation_constraint` est active. Voie 'interpolate',
   origine 'interpolated'. Le recalage lui-même sera fourni par le composant
   AggregationConstraint (lot 10) : dans CE lot, prévoir le point d'injection (paramètre
   `aggregation_constraint_applier` optionnel, ou appel différé) et le laisser inerte si l'objet
   n'est pas fourni. Me dire quelle option a été retenue.
   Cette même méthode d'interpolation sert de REPLI partout où une imputation par modèle échoue,
   dans TOUTES les stratégies : l'exposer publiquement (`interpolate_column`) pour que le lot 13
   l'appelle plutôt que d'en écrire une seconde copie (défaut B27 : deux blocs censés être
   identiques qui divergent).
   Documenter l'avertissement « regard vers l'aval » du §4.3 : l'interpolation linéaire entre deux
   ancres utilise l'ancre FUTURE ; ce n'est pas un défaut pour de l'imputation d'historique, mais
   la docstring doit le dire, et `imputation_scope='extended_forward'` reste le mécanisme dédié
   aux fins de série.

7) `covariate_eligibility` ([SPEC] §4.5), sémantique RECADRÉE sur le seul cas « feature sans
   AUCUNE observation pour la totalité d'une entité » :
       'any_entity' (défaut) : la colonne est retenue dès qu'au moins une entité l'observe ; les
           lignes des entités vides restent NaN — c'est l'UNIQUE source de NaN résiduels sous
           covariate_strategy='interpolate' ;
       'all_entities' : la colonne est écartée de feature_cols si une entité ne l'observe pas.
   Exposer `eligible_columns(columns, data) -> Tuple[str, ...]` et
   `entities_without_column(column, data)`. L'exception à l'invariant du §3 est exactement
   celle-là : L'INVARIANT SE MESURE PAR ENTITÉ.

8) Le composant ne connaît NI le plan, NI l'estimateur, NI les fenêtres : il reçoit un index de
   grille et des données, il rend des features. Les masques de fenêtre sont appliqués par
   l'appelant, en amont, sur `grid_index`.

TESTS — nouveau fichier tests/frequency/test_covariate_materializer.py, sur le jeu TS de [SPEC]
§2.2 et le panel hétérogène (fixtures `reference_timeseries` et
`mixed_freq_panel_heterogeneous` de tests/frequency/conftest.py) :
   - `test_identity_and_aggregate_ranks` : f_c == f et f_c plus fine, origines 'observed' ;
   - `test_incomplete_period_aggregation_is_nan` ;
   - `test_tolerate_nan_keeps_anchors_only` : sur TS, étape M, covariable a1 -> a1 vaut 120 au
     2021-12-31 et NaN sur les 11 autres mois de 2021, AU FIT COMME AU PREDICT (taux 11/12 des
     deux côtés) ;
   - `test_interpolate_removes_all_nan_except_empty_entity` ;
   - `test_interpolate_uses_per_feature_method_and_anchor` : forme dict + '__default__' ;
   - `test_origin_store_values` : les quatre règles du point 2 ;
   - `test_replayed_materialization_is_identical` : `materialize` appelé une fois en mode choix
     puis une fois en mode rejeu produit EXACTEMENT les mêmes valeurs et les mêmes voies — c'est
     le socle du test I11 ;
   - `test_fallback_writes_to_all_three_stores` : la règle « le repli matérialise » ;
   - `test_combine_first_never_overwrites` (B5) ;
   - `test_covariate_eligibility_any_vs_all` sur le panel hétérogène : sous 'any_entity',
     climat_affaires est retenue et IT reste NaN ; sous 'all_entities', elle est écartée ;
   - `test_invariant_nan_by_entity` : formulation D14 de [SPEC] §4.7 — pour chaque (grille,
     colonne), l'ensemble des dates renseignées dans X_pred CONTIENT l'image, sur la grille de
     prédiction, de l'ensemble des dates renseignées dans X_train. Écrire le test AINSI, PAS
     comme une comparaison naïve de pourcentages : sous 'tolerate_nan', les deux grilles n'ont
     pas le même pas et le taux brut diffère légitimement.

Puis `uv run tests/frequency/check_regressions.py` et rapporter. Lot purement additif.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 9 — `covariate_materializer.py` : stratégie `'model'` et précédence à quatre rangs (L6b)

**Modèle : Opus · Plan mode : OUI · Dépendances : prompt 8**

> Plan mode justifié : la précédence à quatre rangs est la réponse aux trois causes de B28 ; le
> rang 3 (report d'étape) est une décision dérivée nouvelle (D13). Le raisonnement sur les
> échelles et les origines à travers le report doit être validé avant écriture.

````text
Contexte : suite immédiate du lot précédent sur tsforecast/frequency/covariate_materializer.py.
Ce lot ajoute la stratégie covariate_strategy='model' et la PRÉCÉDENCE DE MATÉRIALISATION à
quatre rangs, qui est le cœur de la correction du défaut B28 mesuré (0 % de NaN au fit contre
jusqu'à 67 % au predict).
Référence : [SPEC] = high_frequency_imputer2_architecture.md, §4.4 et §4.6, plus §1.1 pour le
défaut corrigé et §14.3 (décision D13). Lire ces sections avant d'écrire.

LA PRÉCÉDENCE ([SPEC] §4.4) — appliquée dans cet ordre, ARRÊT au premier cas applicable, pour
matérialiser la covariable `c` sur la grille de l'étape `f` :

   Rang 1 — f_c >= f : identité ou agrégation exacte. Voie 'identity' / 'aggregate'.
            Origine 'observed'. (Déjà implémenté au lot précédent.)

   Rang 2 — `c` a DÉJÀ été imputée À L'ÉTAPE COURANTE `f` (elle précède la variable en cours dans
            l'ordre d'imputation) : ses valeurs imputées à `f`, lues dans le miroir.
            Voie 'stage_model'. Origine 'model', ou 'interpolated' si son étape a été produite
            par repli — lire `origin_store`, ne pas déduire de la présence d'un modèle.

   Rang 3 — `c` a été imputée à une étape ANTÉRIEURE `f'` (f' plus basse que f) : ses valeurs
            imputées à `f'`, REPORTÉES sur la grille `f` par la voie d'interpolation DE `c`
            (méthode `resolve_method(c)`, ancrage `resolve_anchor(c)`, puis recalage aux totaux
            de `f'`). Voie 'carried_model'. Origine 'model' — l'interpolation d'une valeur de
            modèle RESTE de modèle ; 'interpolated' si l'imputation à `f'` était elle-même un
            repli. Cas atteignable uniquement sous impute_intermediate_frequencies != False.

   Rang 4 — aucun des cas précédents : covariate_fallback.
            'interpolate' -> interpolation des SEULES observations de `c` (voie 'interpolate') ;
            'tolerate_nan' -> ancres + NaN (voie 'raw_anchors').

CE QUE CE LOT DOIT LIVRER

1) L'implémentation des rangs 2 et 3 dans `materialize`, en lisant `imputed_store`,
   `imputed_freq_store` et `origin_store`. Le rang 3 doit :
   - lire la fréquence de production dans `imputed_freq_store` pour savoir depuis QUELLE grille
     reporter (ne pas la redétecter) ;
   - recaler aux totaux DE L'ÉTAPE D'ORIGINE f', pas à ceux de f ;
   - propager l'origine sans la dégrader ni l'améliorer.
   Le rang 3 est la réponse à la PREMIÈRE cause de B28 : une covariable imputée en trimestriel
   n'est plus laissée NaN deux mois sur trois à l'étape mensuelle, elle est REPORTÉE.

2) La RÈGLE D'UNICITÉ DE LA VOIE ([SPEC] §4.6), conséquence non intuitive mais impérative :
   sous 'model', si `c` est servie par le covariate_fallback au predict (rang 4), la version vue
   au FIT doit être préparée par la MÊME voie — interpolée sur la grille d'entraînement si
   covariate_fallback='interpolate', laissée à ses ancres si 'tolerate_nan' — MÊME LORSQUE SES
   ANCRES SUFFIRAIENT. Sinon le modèle apprend sur la covariable exacte et prédit sur la
   covariable interpolée. C'est la généralisation de l'invariant du §3, du motif de NaN à la
   NATURE des valeurs.
   Concrètement : la voie est décidée une fois, sur la grille d'ENTRAÎNEMENT, puis imposée en
   mode rejeu à la grille de prédiction. Le mécanisme `materialization=` du lot précédent est
   déjà là ; ce lot s'assure qu'il est bien le SEUL chemin, y compris sous 'model'.

3) Une méthode `decide_ways(...) -> Dict[str, MaterializationWay]` isolant la décision de
   précédence, testable seule, sans production de valeurs. `materialize` l'appelle quand
   `materialization is None`.

4) Vérifier explicitement, et documenter en commentaire français, que sous 'tolerate_nan' et
   'interpolate' les rangs 2 et 3 ne sont JAMAIS atteints : ces deux stratégies ne consultent
   pas les stores pour matérialiser. C'est ce qui rend l'ordre d'imputation indifférent hors
   'model' ([SPEC] §8.1) et ce que testera l'invariant I10.

TESTS — à ajouter dans tests/frequency/test_covariate_materializer.py :
   - `test_rank_two_reads_stage_mirror` : `c` imputée à l'étape courante est lue dans le miroir,
     voie 'stage_model' ;
   - `test_rank_two_origin_follows_store` : si l'étape de `c` était un repli, l'origine lue est
     'interpolated', pas 'model' ;
   - `test_rank_three_carries_from_previous_stage` : stores amorcés à la main avec une imputation
     trimestrielle de `a1`, étape M -> voie 'carried_model', aucune cellule NaN, origine 'model' ;
   - `test_rank_three_rescales_to_origin_stage_totals` : le recalage porte sur les totaux de f',
     pas de f ;
   - `test_rank_four_fallback_interpolate` et `test_rank_four_fallback_tolerate_nan` ;
   - `test_precedence_stops_at_first_applicable_rank` : une covariable présente à la fois dans le
     miroir d'étape et dans celui d'une étape antérieure prend le rang 2 ;
   - `test_way_unicity_fit_and_pred` : sur le jeu TS, `covariate_strategy='model'`, ordre
     `a1` avant `a2` — `a2` est préparée par covariate_fallback DES DEUX CÔTÉS, y compris au fit
     où ses ancres suffiraient. Vérifier la valeur, pas seulement la voie ;
   - `test_ranks_two_and_three_unreachable_outside_model` : sous 'tolerate_nan' et 'interpolate',
     amorcer les stores puis vérifier que les voies retournées ne contiennent ni 'stage_model' ni
     'carried_model' ;
   - `test_reference_example_of_spec_4_7` : reproduire les quatre lignes du tableau de [SPEC] §4.7
     (imputation de `a1` à l'étape Q, feature_cols = [m1, q1, a2]) — voies retenues et taux de NaN
     de `a2` au fit et au predict, pour les quatre configurations du tableau.

Puis `uv run tests/frequency/check_regressions.py` et rapporter.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 10 — `aggregation_constraint.py` : recalage aux totaux de période (L7)

**Modèle : Opus · Plan mode : OUI · Dépendances : prompt 4**

> Plan mode justifié : quatre gardes, un masque de cellules recalées qui pilote le marquage de
> provenance, et une règle B2 contre-intuitive (le marquage est indépendant de la réussite du
> recalage). La répartition entre « ce que le composant décide » et « ce que l'appelant marque »
> doit être arrêtée avant écriture.

````text
Contexte : je prépare l'implémentation de HighFrequencyImputer2 ([SPEC] =
high_frequency_imputer2_architecture.md). Ce lot extrait la contrainte d'agrégation dans un
composant autonome et remplace le booléen `enforce_period_totals` par un paramètre extensible.
Référence : [SPEC] §11 en entier (§11.1 et §11.2), §6.4 (provenance des cellules non produites
par un modèle), décisions D7 et D8 du §14.2. Lire aussi `hfi:_rescale_to_period_totals` : la
logique existe, ce lot la déplace et l'encadre — il ne la réinvente pas.

CIBLE — nouveau fichier tsforecast/frequency/aggregation_constraint.py

1) Paramètre public :

       aggregation_constraint: Literal['sum', None] = 'sum'

   'sum' équivaut à `enforce_period_totals=True`, `None` à `False`. VALIDATION À L'ÉTAT ACTUEL :
   seules les valeurs 'sum' 'mean', 'last' et None sont acceptées. TOUTE autre valeur — SAUF UN DICT associant à un nom de colonne l'une de ces valeurs — lève
   un `ValueError` dont le message énonce que seules formes 'sum' 'mean', 'last' et None sont acceptées. La docstring décrit l'extension (contraintes
   par colonne avec clé '__default__', sur le modèle de `estimator` et `scale_features`).
   Le type littéral est retenu MAINTENANT pour ouvrir l'extension sans rupture d'API.

2) Sémantique de 'sum', reprise de `hfi:_rescale_to_period_totals` : les sous-périodes prédites
   d'une période sont multipliées par `total observé / total prédit`, de sorte que la colonne
   porte une véritable DÉSAGRÉGATION de l'observation plutôt qu'une prédiction libre.
   LES QUATRE GARDES, à conserver telles quelles ([SPEC] §11.1) :
       période PARTIELLEMENT prédite (au moins une sous-période NaN) -> non recalée, prédictions
           brutes conservées ;
       période sans aucune observation (fin de série retardée) -> non recalée ;
       total prédit NUL et total observé non nul -> non recalée (ratio indéfini) ;
       total prédit de SIGNE OPPOSÉ au total observé -> RECALÉE, la contrainte prime, mais toutes
           les sous-périodes changent de signe : un UserWarning AGRÉGÉ est émis (un seul message
           pour toute l'opération, nommant les colonnes et le nombre de périodes concernées —
           jamais un avertissement par période).

3) Retour de la méthode principale : le DataFrame recalé ET le MASQUE BOOLÉEN des cellules
   EFFECTIVEMENT RECALÉES. C'est ce masque qui pilote le marquage DISAGGREGATED chez l'appelant ;
   les cellules laissées de côté gardent leur provenance MODEL_* ou INTERPOLATED ([SPEC] §6.4).
   Le composant NE MARQUE PAS lui-même la provenance : il rend le masque, l'appelant marque.

4) DÉSAGRÉGATION DES ANCRES, comportement NON PARAMÉTRABLE ([SPEC] §11.2, décision D7) :
   une variable imputée à l'étape `f` est prédite sur la TOTALITÉ de chaque période couverte,
   ANCRES COMPRISES. La ligne qui portait le total de la période reçoit, comme les autres, une
   valeur de sous-période. `disaggregate_anchors` N'EXISTE PAS et ne doit pas être introduit.
   Le composant expose `anchor_cells_mask(...)` identifiant les lignes d'ancre ré-exprimées à la
   fréquence d'étape, car elles portent DISAGGREGATED elles aussi.
   RÈGLE B2, contre-intuitive et impérative : le marquage DISAGGREGATED de la ligne d'ancre est
   INDÉPENDANT DE LA RÉUSSITE DU RECALAGE. Il dit « cette cellule occupe la place d'une
   observation réelle », pas « cette cellule respecte l'identité additive ». Donc
   `anchor_cells_mask` ne dépend PAS de `aggregation_constraint` : sous `None` aussi, la ligne
   d'ancre est marquée DISAGGREGATED tout en portant une valeur libre.

5) Docstring de classe : documenter les deux conséquences du §11.2 —
   - sous 'sum', AUCUNE information n'est perdue : la somme des sous-périodes reconstitue
     exactement le total observé ;
   - sous None, le total observé EST ÉCRASÉ par une prédiction libre ; il reste récupérable de
     deux manières, toutes deux à mentionner : par `inverse_transform`, et par le masque ORIGINAL
     du niveau de fréquence source dans la sortie multi-fréquences
     (`keep_lower_frequencies=True`).
   Motif de la non-paramétrabilité, à écrire : une colonne ne doit jamais mélanger, à une
   fréquence d'imputation donnée, un total de période et des valeurs de sous-période — cette
   hétérogénéité rend la colonne inexploitable comme covariable, fausse toute agrégation en aval
   et rend l'échelle d'une ligne dépendante de sa position dans la période.

6) Le composant est utilisable sur série temporelle ET sur panel (masques `pd.Series` à
   MultiIndex, conformément au §7.2), et sur plusieurs colonnes à la fois.

Si cela est possible, j'aimerais implémenter cette classe sous la forme d'un Transformer sklearn.

TESTS — nouveau fichier tests/frequency/test_aggregation_constraint.py, sur le jeu TS de [SPEC]
§2.2 (fixture `reference_timeseries`) :
   - `test_reference_example_of_spec_6_5` : `a1` = 120 en 2021, étape M, prédictions brutes
     sommant à 112.5 -> après recalage, la somme 2021 vaut EXACTEMENT 120.0, et la ligne d'ancre
     2021-12-31 vaut 11.2 (et non 120) ;
   - `test_partial_period_not_rescaled` ;
   - `test_period_without_observation_not_rescaled` ;
   - `test_zero_predicted_total_not_rescaled` ;
   - `test_opposite_sign_is_rescaled_and_warns_once` : le recalage a lieu, un SEUL UserWarning est
     émis pour N périodes concernées ;
   - `test_rescaled_mask_matches_modified_cells` : le masque retourné coïncide exactement avec
     les cellules dont la valeur a changé ;
   - `test_anchor_mask_independent_of_constraint` (B2) : identique sous 'sum' et sous None ;
   - `test_constraint_none_leaves_raw_values` ;
   - `test_dict_form_raises_with_reserved_extension_message` : le message cite 'mean', 'last' et
     la forme dictionnaire ;
   - version panel d'au moins `test_reference_example_of_spec_6_5` et
     `test_anchor_mask_independent_of_constraint`.

Puis `uv run tests/frequency/check_regressions.py` et rapporter. NE PAS modifier hfi :
`_rescale_to_period_totals` y reste en place tant que hfi existe.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 11 — `variable_orderer.py` : ordres `'frequency'` et `'cv'` (L8)

**Modèle : Sonnet · Plan mode : Non · Dépendances : prompt 5**

> Plan mode inutile : les deux logiques de tri existent déjà dans `hfi` et sont reprises ; les
> correctifs à appliquer sont énumérés un par un ci-dessous.

````text
Contexte : je prépare l'implémentation de HighFrequencyImputer2 ([SPEC] =
high_frequency_imputer2_architecture.md). Ce lot extrait l'ordonnancement des variables dans un
composant autonome, remplace `cv_n_splits` par le paramètre `cv` polymorphe de sklearn, et rend
l'ordre DÉTERMINISTE par un tie-break alphabétique.
Référence : [SPEC] §8 en entier (§8.1 à §8.4), décision D4 du §14.2. Reprendre les logiques de
`hfi:_determine_imputation_order` et `hfi:_determine_variable_order_cv`.

CIBLE — nouveau fichier tsforecast/frequency/variable_orderer.py, classe `VariableOrderer`

1) `fit_predict_order: Literal['frequency', 'cv'] = 'frequency'` — mêmes modalités que
   `train_on_partial_fit_order` de hfi, dont c'est le renommage :
   - 'frequency' : fréquence la plus BASSE d'abord, puis nombre d'entités DÉCROISSANT ;
   - 'cv' : variables les MIEUX prédites d'abord, autour de `cross_val_score`.

2) CHAMP D'APPLICATION RESTREINT ([SPEC] §8.1), à documenter et à faire respecter par l'appelant :
   l'ordre n'est calculé et appliqué que sous `covariate_strategy='model'`. Sous 'tolerate_nan' et
   'interpolate', AUCUNE logique de tri n'est exécutée ; l'ordre de traitement est celui des
   colonnes d'entrée, SANS effet sur les valeurs produites. Passer `fit_predict_order='cv'` avec
   une autre stratégie n'est PAS une erreur : le paramètre est ignoré, la docstring le dit, aucun
   avertissement n'est émis (décision D9 : paramètres inertes = silence + docstring).
   Écrire la justification en commentaire français : 'tolerate_nan' ne matérialise rien ;
   'interpolate' matérialise depuis les seules observations ; dans les deux cas le jeu
   d'entraînement et le jeu de prédiction d'une variable ne dépendent d'aucune imputation
   antérieure d'une autre variable.

3) DÉTERMINISME INTRA-ÉTAPE, impératif ([SPEC] §8.1) : les ex æquo du tri sont départagés par
   ORDRE ALPHABÉTIQUE DU NOM DE VARIABLE — jamais par l'ordre des colonnes d'entrée. Cela vaut
   pour 'frequency' (même fréquence et même nombre d'entités) ET pour 'cv' (scores égaux, NaN
   compris). C'est la réponse à la deuxième cause de B28 : l'asymétrie intra-étape demeure, mais
   elle devient déterministe, documentée et indépendante de la présentation des données.

4) Paramètre `cv`, contrat sklearn ([SPEC] §8.3, décision D4) :

       cv: Union[int, BaseCrossValidator, Iterable, None] = None
       # None -> KFold(n_splits=5, shuffle=True, random_state=42)
       # int  -> KFold(n_splits=cv, shuffle=True, random_state=42)
       # splitter ou itérable de splits -> utilisé tel quel
       cv_scoring: Union[str, Callable] = 'neg_mean_absolute_percentage_error'
       min_cv_train_size: int = 10

   `cv_n_splits` est SUPPRIMÉ, absorbé par `cv`. Le défaut `shuffle=True, random_state=42` est
   repris du code actuel AVEC son commentaire justificatif ; un utilisateur qui n'en veut pas
   passe son propre splitter. `check_cv` n'est appelé QU'AU FIT, et son résultat stocké dans
   `cv_` — le paramètre reste inchangé pour `get_params`/`clone` (règle B3).
   AVERTISSEMENT CROISÉ conservé, reformulé : après `check_cv`, si
   `min_cv_train_size < n_splits effectifs`, un UserWarning UNIQUE est émis au fit, nommant les
   deux valeurs.

5) CORRECTIFS CV à reprendre ([SPEC] §8.2, correctifs B9/B10 de [ARCH] §3.4) :
   - restriction aux lignes exploitables AVANT scoring ;
   - `check_scoring` pour résoudre `cv_scoring` ;
   - `cross_val_score(..., error_score=np.nan)` ;
   - sentinelles `-np.inf` pour les variables non scorables ;
   - TRI DÉCROISSANT PARTOUT (convention *greater is better* de sklearn) ;
   - journal (`self._log`, ou un callback de journalisation injecté) des variables dont TOUS les
     plis ont échoué ;
   - masque `kind='strict'` pour construire le jeu de scoring — justification en commentaire :
     l'ordonnancement doit comparer les variables sur des données de qualité homogène, sinon le
     score d'une variable dépend de l'étendue de son extension ;
   - note de docstring sur le traitement des zéros par le MAPE de sklearn (division par zéro ->
     score dégradé, pas d'erreur).

6) Validation à l'init, SANS transformation : `cv` doit être None, un `int >= 2`, un objet
   exposant `split` ET `get_n_splits`, ou un itérable ; `cv_scoring` str ou callable ;
   `min_cv_train_size` `int >= 1` ; `fit_predict_order` dans le Literal (message listant les
   valeurs admises).

TESTS — nouveau fichier tests/frequency/test_variable_orderer.py, sur la fixture
`reference_timeseries` :
   - `test_frequency_order_on_reference_ts` : reproduit [SPEC] §8.4 — l'ordre est `a1, a2, q1`
     (Y avant Q ; `a1` avant `a2` par tie-break alphabétique) ;
   - `test_cv_order_on_reference_ts` : scores `q1`=-0.08, `a2`=-0.15, `a1`=-0.15 injectés par un
     estimateur factice -> ordre `q1, a1, a2` (tri décroissant, tie-break alphabétique) ;
   - `test_alphabetical_tiebreak_ignores_column_order` : permuter les colonnes d'entrée ne change
     pas l'ordre, sous les deux modalités ;
   - `test_cv_accepts_int_splitter_and_iterable` ;
   - `test_cv_none_defaults_to_kfold_5_shuffled` ;
   - `test_cv_not_resolved_at_init` : `check_cv` n'est pas appelé avant le fit ; le paramètre
     reste identique dans `get_params` ;
   - `test_min_cv_train_size_warning_emitted_once` ;
   - `test_unscorable_variables_sorted_last` : sentinelles `-np.inf` ;
   - `test_all_folds_failed_is_logged` ;
   - `test_cv_n_splits_is_gone` : le composant n'accepte aucun paramètre nommé `cv_n_splits`.

Puis `uv run tests/frequency/check_regressions.py` et rapporter. NE PAS modifier hfi.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 12 — `hfi2` : `__init__`, validations, phases 0 à 4 (L9)

**Modèle : Opus · Plan mode : OUI · Dépendances : prompts 6, 7, 9, 10, 11**

> Plan mode justifié : c'est la création de la classe principale et de son contrat sklearn ;
> 24 paramètres, 14 attributs ajustés, cinq phases et six composants à câbler. Le plan valide
> l'assemblage avant écriture.

````text
Contexte : je prépare l'implémentation de HighFrequencyImputer2 ([SPEC] =
high_frequency_imputer2_architecture.md). Tous les composants du §12.2 existent désormais :
imputation_plan2.py, stage_scaler.py, covariate_materializer.py, aggregation_constraint.py,
variable_orderer.py, plus provenance.py et imputation_window.py étendus. Ce lot crée la classe
ORCHESTRATRICE et s'arrête AVANT la PHASE 5 (exécution des étapes), qui est le lot suivant.
Référence : [SPEC] §12.3 (squelette de fit, phases 0 à 4), §12.5 (conformité sklearn), §13 en
entier (API paramètre par paramètre, validations, attributs ajustés).

CIBLE — nouveau fichier tsforecast/frequency/high_frequency_imputer2.py

1) `class HighFrequencyImputer2(XYPanelTimeSeriesTransformer)` — même classe de base que hfi
   (tsforecast/xy/transformers.py). Signature d'`__init__` EXACTEMENT celle du §13 de [SPEC],
   dans le même ordre et avec les mêmes défauts. La recopier depuis le document, ne pas la
   reconstituer de mémoire. Rappel des paramètres qui N'EXISTENT PLUS et ne doivent pas
   réapparaître : `cascade_refitting`, `train_on_partial_coverage`, `train_on_partial_fit_order`,
   `enforce_period_totals`, `cv_n_splits`, `disaggregate_anchors`.

2) RÈGLE B3, non négociable : `__init__` stocke les paramètres TELS QUE REÇUS et VALIDE SANS
   TRANSFORMER. `clone` et `get_params`/`set_params` doivent être exacts. Toute normalisation a
   lieu au fit, dans des attributs suffixés `_`.
   Reprendre L'INTÉGRALITÉ du bloc de validation de `hfi:__init__` ([SPEC] §1.3) : format de
   `target_frequency`, contrat de `estimator` (y compris la forme dict avec clé '__default__'),
   `additive_transformer` exposant `fit_transform` ET `inverse_transform`, bornes numériques,
   Literals, validation GROUPÉE des booléens, avertissement UNIQUE si `estimator=None`.
   Y ajouter les contrôles du tableau du §13.1 de [SPEC], que je recopie ici pour éviter toute
   dérive :
     - covariate_strategy, covariate_fallback, covariate_eligibility, fit_predict_order,
       on_frequency_mismatch, imputation_scope, training_scope : appartenance au Literal, message
       listant les valeurs admises ;
     - impute_intermediate_frequencies : `is False`, `== 'covariates_only'`, ou `is True` —
       JAMAIS un test de vérité booléenne. 'covariates_only' est *truthy* : un
       `if self.impute_intermediate_frequencies:` serait un bug silencieux ;
     - interpolation_method : str, ou dict {str: str} (clés vérifiées au fit) ;
     - interpolation_anchor : None, float dans [0, 1], ou dict de ces valeurs ;
     - cv : None, int >= 2, objet à split/get_n_splits, ou itérable (check_cv AU FIT SEULEMENT) ;
     - cv_scoring : str ou callable ; min_cv_train_size : int >= 1 ;
     - coverage_threshold, training_coverage_threshold : float dans [0, 1] (None admis pour le
       second) ;
     - scale_features : False, 'constant', 'calendar', ou dict de ces valeurs ;
     - aggregation_constraint : 'sum' ou None EXACTEMENT, message d'erreur mentionnant
       l'extension réservée ;
     - booléens (keep_lower_frequencies, restore_original_values, verbose) : validation groupée.

3) DOCSTRING DE CLASSE — c'est un livrable de ce lot, pas un accessoire. Elle doit énoncer :
   - les DEUX AXES ORTHOGONAUX et la question à laquelle chacun répond ([SPEC] §0) ;
   - pour `covariate_strategy='tolerate_nan'`, le PRÉREQUIS DUR « l'estimateur doit tolérer les
     NaN » ;
   - pour `covariate_strategy='interpolate'` et `interpolation_method`, l'avertissement « regard
     vers l'aval » : l'interpolation linéaire entre deux ancres utilise l'ancre FUTURE ;
   - les COMBINAISONS INERTES du §5.6, documentées SANS avertissement (décision D9) :
     'covariates_only' sans covariate_strategy='model' ne change aucune valeur finale ;
     covariate_fallback, fit_predict_order inertes hors 'model' ;
     training_coverage_threshold sans training_scope ;
   - `keep_lower_frequencies` comme PARAMÈTRE D'AFFICHAGE PUR, et le fait que sous
     impute_intermediate_frequencies=False il n'y a pas de niveau intermédiaire à empiler ;
   - sous impute_intermediate_frequencies=False, `y_train` d'une variable annuelle sur trois
     ancres contient trois lignes : `min_cv_train_size` et les gardes de taille de l'estimateur
     sont le prix de la modalité, le repli interpolation restant le filet ([SPEC] §5.3, point 3).

4) PHASES 0 à 4 de `fit` ([SPEC] §12.3), dans cet ordre exact :
     PHASE 0 setup : purge de l'état de transform (B19), colonnes, détection panel, alignement et
             nommage de `y` (B14 : vérification d'ÉGALITÉ DES INDEX, pas seulement des
             longueurs), détection des fréquences, normalisation/validation de target_frequency
             (B16 : dict incomplet -> ValueError NOMMANT les entités manquantes), classification
             des variables ;
     PHASE 1 calcul des fenêtres — LES TROIS MASQUES ([SPEC] §7), chaque appelant NOMMANT
             explicitement son `kind` : jamais d'appel sans `kind` ;
     PHASE 2 ajustement du transformateur additif ;
     PHASE 3 construction de la progression de fréquences ([SPEC] §5.2) — dans ce lot, se
             limiter au cas `impute_intermediate_frequencies is False`, qui donne
             `progression = [f_target]` ; le cas complet est le lot 14, prévoir le point
             d'extension et le signaler par un TODO nommant le lot ;
     PHASE 4 initialisation du tracker de provenance — APRÈS le transformateur additif (B8), dans
             les DEUX chemins.
   Reprendre les phases 0 à 4 de `hfi:_fit`, qui sont saines ([SPEC] §1.2 et §1.3) : les adapter,
   pas les réécrire de zéro.

5) RÈGLE D'USAGE DES FENÊTRES ([SPEC] §7.2), à câbler dès ce lot : élargir `training_scope`
   AJOUTE DES LIGNES, JAMAIS DES COLONNES. La sélection des feature_cols reste gouvernée par la
   disponibilité à la PRÉDICTION, indépendamment de la fenêtre d'entraînement. Les deux
   ajustements de [ARCH] §3.6 suivent : une colonne n'est gardée que si elle est non-vide sur LES
   DEUX fenêtres ; les lignes d'entraînement sans aucune covariable observée sont écartées.

6) ATTRIBUTS AJUSTÉS — les quatorze du §13.2 de [SPEC], déclarés et typés, renseignés par ce lot
   quand la phase correspondante existe, initialisés sinon :
   effective_target_frequency_, detected_frequencies_, variable_categories_,
   frequency_progression_, imputation_order_, imputation_plan_, imputation_models_,
   imputation_window_mask_, training_window_mask_, strict_window_mask_, imputation_window_,
   training_window_, imputation_provenance_, feature_columns_, target_column_, entities_,
   is_panel_, cv_ (ce dernier présent SEULEMENT sous fit_predict_order='cv').
   `imputation_order_` est VIDE hors covariate_strategy='model'.

7) CONFORMITÉ SKLEARN ([SPEC] §12.5) : B20 — `NotFittedError` propre avant fit, via
   `check_is_fitted` avec une LISTE EXPLICITE d'attributs ; B15 — panel déclaré par `panel_cols`
   sur frame plat, pleinement fonctionnel ; avertissements UNIQUES et agrégés.

8) `transform` / `inverse_transform` : dans ce lot, lever `NotImplementedError` avec un message
   nommant le lot qui les livre. NE PAS écrire une version provisoire : le risque est qu'elle
   survive et diverge du fit (défauts B7/B27, motif de toute l'architecture).

9) Exporter `HighFrequencyImputer2` depuis tsforecast/frequency/__init__.py, À CÔTÉ de
   `HighFrequencyImputer` : les deux classes COEXISTENT pendant la transition ([SPEC] §12.2,
   §15.3). Ne rien déprécier dans ce lot.

TESTS — nouveau fichier tests/frequency/test_high_frequency_imputer2.py :
   - un test de validation PAR LIGNE du tableau du §13.1 : valeur invalide -> ValueError dont le
     message liste les valeurs admises ;
   - `test_impute_intermediate_frequencies_covariates_only_is_not_true` : le paramètre est accepté
     et n'est pas confondu avec True ;
   - `test_removed_parameters_are_rejected` : `cascade_refitting`, `train_on_partial_coverage`,
     `train_on_partial_fit_order`, `enforce_period_totals`, `cv_n_splits`, `disaggregate_anchors`
     lèvent TypeError ;
   - `test_get_params_returns_untouched_values` et `test_clone_roundtrip` sur un jeu de
     paramètres exotiques (target_frequency dict, scale_features dict, cv = splitter) ;
   - `test_not_fitted_error_before_fit` ;
   - `test_phases_zero_to_four_populate_attributes` sur `reference_timeseries` ET
     `mixed_freq_panel_heterogeneous` ;
   - `test_target_frequency_dict_incomplete_raises_naming_entities` (B16) ;
   - `test_y_index_equality_checked` (B14) : longueurs égales mais index différents -> erreur ;
   - `test_three_window_masks_are_set` ;
   - `test_transform_not_implemented_yet` ;
   - `test_no_boolean_test_on_impute_intermediate_frequencies` : test STATIQUE (grep sur le
     source du module) vérifiant qu'aucun `if self.impute_intermediate_frequencies:` ni
     `if not self.impute_intermediate_frequencies:` n'apparaît — c'est l'invariant I13 de
     [SPEC] §16.

Puis `uv run tests/frequency/check_regressions.py` et rapporter.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 13 — `hfi2` : PHASE 5, exécution d'étape unique (L10)

**Modèle : Opus · Plan mode : OUI · Dépendances : prompt 12**

> Plan mode justifié : c'est le cœur de la classe. Une seule implémentation d'exécution d'étape
> doit servir le fit et le transform ; l'ordre des opérations internes (matérialisation,
> souillures, échelle, ajustement, prédiction, recalage, écriture, marquage, stores) est
> l'architecture elle-même et doit être arrêté avant écriture.

````text
Contexte : suite du lot précédent sur tsforecast/frequency/high_frequency_imputer2.py, dont les
phases 0 à 4 sont en place. Ce lot écrit la PHASE 5, c'est-à-dire l'exécution des étapes du plan.
Il se limite au régime `impute_intermediate_frequencies is False` (progression à une seule étape,
la fréquence cible) : l'axe 2 est le lot suivant.
Référence : [SPEC] = high_frequency_imputer2_architecture.md, §12.1 (un plan, un exécuteur),
§12.3 (bloc PHASE 5, à suivre pas à pas), §6.2 et §6.3 (souillures et provenance), §4.6 (unicité
de la voie), §11 (recalage). Lire ces sections avant d'écrire.

PRINCIPE STRUCTURANT ([SPEC] §12.1), à respecter dès l'écriture et non à rattraper plus tard :
il n'existe QU'UNE SEULE implémentation de l'exécution d'étape, `_execute_step`, paramétrée par
« fit : ajuster puis prédire » vs « transform : prédire avec le modèle figé ». Les écritures
(vidage/réécriture, recalage, provenance, stores) sont COMMUNES PAR CONSTRUCTION. C'est la leçon
des défauts B7/B27/B8 : chaque fois que fit et transform portent deux copies d'une logique, elles
divergent. Le lot 15 branchera le transform sur CETTE méthode ; ne pas en préparer une seconde.

PHASE 5 — pour chaque étape de fréquence `f` de la progression, dans cet ordre exact :

  5a. FRAME D'ÉTAPE : données d'origine + agrégations exactes à `f` + miroir des imputations,
      produit par le CovariateMaterializer — le MÊME objet servira au transform.

  5b. Variables imputables à `f` ; ordre calculé par le VariableOrderer SEULEMENT si
      `covariate_strategy == 'model'`. Sinon, ordre des colonnes d'entrée, sans effet sur les
      valeurs (invariant I10).

  5c. pour chaque variable `v` :
      - GRILLE D'ENTRAÎNEMENT : ancres de `v`, masque `kind='training'`.
        GRILLE DE PRÉDICTION : masque `kind='imputation'`. Nommer explicitement le `kind` aux
        deux appels.
      - SÉLECTION DES feature_cols : non-vides sur LES DEUX fenêtres, filtrées par
        `covariate_eligibility`.
      - MATÉRIALISATION des covariables sur LES DEUX GRILLES PAR LA MÊME VOIE ([SPEC] §4.6) :
        un appel à `materialize` sur la grille d'entraînement (mode choix) qui retourne les voies,
        puis un appel sur la grille de prédiction en MODE REJEU avec ces voies. Enregistrer
        `materialization[col]` dans l'étape. C'est le seul chemin autorisé pour produire X_train
        et X_pred.
      - SOUILLURES ([SPEC] §6.2) :
        `covariate_taint` = max des origines des cellules EFFECTIVEMENT LUES dans
        `X_train ∪ X_pred`, restreint aux feature_cols EFFECTIVES du modèle — leçon C17 de [ARCH]
        §3.8 : JAMAIS sur l'état global du store ;
        `target_taint` = max des origines des lignes retenues dans `y_train`.
        Les deux lisent `origin_store`, JAMAIS la matrice de provenance.
      - MISE À L'ÉCHELLE via le StageScaler (scalaire ou Series par ligne).
      - AJUSTEMENT de l'estimateur. En cas d'ÉCHEC -> repli interpolation par la méthode de `v`
        (`CovariateMaterializer.interpolate_column`, pas une seconde implémentation), étape
        marquée `is_fallback=True`, cellules marquées `INTERPOLATED` (décision D6 : les cellules
        de repli ne portent PAS un MODEL_*, c'est plus exact et cela rend le repli visible dans
        les statistiques de provenance).
      - PRÉDICTION SUR TOUTE LA PÉRIODE, ANCRES COMPRISES ([SPEC] §11.2, non paramétrable).
      - RECALAGE AUX TOTAUX via AggregationConstraint -> masque des cellules recalées.
      - ÉCRITURE des valeurs et MARQUAGE DE PROVENANCE :
        cellules recalées ET lignes d'ancre -> `DISAGGREGATED` (marquage des ancres INDÉPENDANT
        de la réussite du recalage, règle B2) ;
        cellules non recalées produites par le modèle -> `resolve_model_provenance(covariate_taint,
        target_taint)` ; cellules de repli -> `INTERPOLATED`.
        La provenance est une PROPRIÉTÉ DE L'ÉTAPE, propagée identiquement à toutes les cellules
        que le modèle de cette étape produit.
      - MISE À JOUR de imputed_store / imputed_freq_store / origin_store, Y COMPRIS EN REPLI
        (« le repli matérialise »).
      - GEL de l'ImputationStep (v2) dans le plan, via `append_step`.

  PHASE 6 finalisation : plan figé, attributs de sortie renseignés, sortie multi-fréquences si
  demandée.

POINTS DE VIGILANCE PROPRES À CE LOT
   - `y_train` sous ce régime = les seules ancres de `v` (origine 'observed'), donc
     `target_taint == 'none'` toujours. Ne pas coder cette valeur en dur : la CALCULER par le
     filtre d'origine, pour que le lot 14 n'ait qu'à élargir `ELIGIBLE_ORIGINS`.
   - Tous les avertissements sont ACCUMULÉS puis émis en UN SEUL message en fin de phase.
   - `imputation_order_` reste vide hors 'model'.

TESTS — à ajouter dans tests/frequency/test_high_frequency_imputer2.py, sur `reference_timeseries`
et `mixed_freq_panel_heterogeneous`, avec un ESTIMATEUR ESPION enregistrant X_train, y_train,
X_pred et le taux de NaN de chaque appel :
   - **I2** `test_nan_invariant_by_stage_and_column` : formulation D14 — pour chaque (étape,
     colonne), l'ensemble des dates renseignées dans X_pred CONTIENT l'image, sur la grille de
     prédiction, de celui de X_train ; testé sous les TROIS stratégies, sur TS et sur panel, PAR
     ENTITÉ. C'est le test qui échoue sur hfi aujourd'hui ;
   - **I3** `test_column_order_invariance` : permuter les colonnes d'entrée ne change ni les
     valeurs ni les provenances, sous les trois stratégies ;
   - **I10** `test_processing_order_indifferent_outside_model` : sous 'tolerate_nan' et
     'interpolate', forcer deux ordres de traitement différents produit des sorties IDENTIQUES ;
   - **I11** `test_materialization_way_recorded_per_step` : chaque étape porte une entrée
     `materialization` par feature_col, et la voie est celle attendue par la précédence ;
   - **I6** `test_provenance_families` : les cinq familles MODEL_* sont émises exactement dans les
     cas du §6.3 et aucun autre ; MODEL_ON_IMPUTED seulement sous 'model' ; sous 'interpolate',
     dès qu'une covariable de fréquence plus basse que la grille entre dans feature_cols, la
     provenance est MODEL_ON_INTERPOLATED et NON MODEL_ON_TRUE (c'est le point de rupture avec la
     version 1 du document) ; les cellules de repli portent INTERPOLATED ;
   - **I4** `test_period_totals_additivity` : sous aggregation_constraint='sum', chaque colonne
     imputée somme au total observé de chaque période COMPLÈTE, et la ligne d'ancre ne porte plus
     le total ;
   - **I5** `test_train_test_feature_means_comparable` : pour chaque étape, moyennes des features
     de X_train et de X_pred comparables, tolérance dépendant de la modalité ; inclut le cas mixte
     « feature 'calendar', y 'constant' » ;
   - `test_reference_plan_of_spec_5_5_under_false` : reproduit le tableau « Sous False » du §5.5 —
     une étape M, trois modèles, dans l'ordre `a1`, `a2`, `q1`, avec les voies de matérialisation
     annoncées (`m1` identity ; `q1`, `a2` fallback pour `a1` ; `a1` stage_model pour `a2`…) ;
   - `test_reference_provenance_of_spec_6_5` : `a1` imputée à l'étape M sous
     covariate_strategy='interpolate' -> provenance émise MODEL_ON_INTERPOLATED, ancre
     2021-12-31 marquée DISAGGREGATED, somme 2021 = 120.0 ;
   - `test_estimator_failure_falls_back_and_marks_interpolated` : is_fallback=True, cellules
     INTERPOLATED, stores alimentés ;
   - cas limites : index non trié, index dupliqué, entité à une seule observation, variable
     annuelle à 2 ancres seulement, période incomplète en début et en fin de série, colonne
     entièrement NaN, `estimator=None`.

Puis `uv run tests/frequency/check_regressions.py` et rapporter.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 14 — Axe 2 : progression, `ELIGIBLE_ORIGINS`, échelle par ligne (L11)

**Modèle : Opus · Plan mode : OUI · Dépendances : prompt 13**

> Plan mode justifié : le piège principal du lot (`'covariates_only'` devenant un synonyme
> silencieux de `True` si le filtre lit la provenance au lieu de `origin_store`) se prévient à la
> conception, pas au débogage.

````text
Contexte : suite du lot précédent sur tsforecast/frequency/high_frequency_imputer2.py, dont la
PHASE 5 fonctionne pour `impute_intermediate_frequencies is False`. Ce lot ouvre le SECOND AXE :
la traversée des fréquences intermédiaires.
Référence : [SPEC] = high_frequency_imputer2_architecture.md, §5 en entier (§5.1 à §5.7), §4.4
rang 3 (report d'étape), §14.3 décisions D12 et D13. Lire ces sections avant d'écrire.

LE PARAMÈTRE ([SPEC] §5.1)

    impute_intermediate_frequencies: Literal[False, 'covariates_only', True] = False

    False             -> une seule étape, à la fréquence cible ;
                         y_train = ancres uniquement (origine 'observed')
    'covariates_only' -> progression COMPLÈTE ;
                         y_train = ancres + cellules d'origine 'interpolated',
                         JAMAIS ses propres imputations de modèle
    True              -> progression COMPLÈTE ;
                         y_train = ancres + 'interpolated' + SES PROPRES IMPUTATIONS des étapes
                         antérieures (origine 'model')

'covariates_only' et True produisent LE MÊME PLAN D'ÉTAPES et diffèrent uniquement par le FILTRE
D'ORIGINE de y_train ; False et 'covariates_only' appliquent LE MÊME FILTRE et diffèrent
uniquement par le PLAN. Structurer le code sur cette factorisation exacte.

CE QUE CE LOT DOIT LIVRER

1) PROGRESSION DE FRÉQUENCES ([SPEC] §5.2), algorithme identique au fit et au transform :
     a. `F` = fréquences détectées des colonnes IMPUTABLES du périmètre, plus `f_target` ;
     b. si `impute_intermediate_frequencies is False` : `progression = [f_target]` ;
     c. sinon : `sorted(F \ {la plus basse})` de la plus basse à la plus haute, en ne retenant que
        les fréquences strictement PLUS HAUTES que la plus basse fréquence source et INFÉRIEURES
        OU ÉGALES à f_target, en garantissant que f_target en est le DERNIER élément ;
     d. à chaque étape `f`, les variables imputables à `f` sont les colonnes dont `f_var` est
        strictement plus basse que `f` et qui ne sont pas encore imputées À `f`.
   Vérification attendue sur le jeu TS : `F = {Q, Y, M}` ; sous False -> `['M']` ; sous
   'covariates_only'/True -> `['Q', 'M']` (la fréquence Y, la plus basse, n'est PAS une étape :
   rien n'est à imputer à Y). Étape Q : variables {a1, a2} ; étape M : {q1, a1, a2}.
   Sur un panel, la progression est calculée PAR GROUPE D'ENTITÉS PARTAGEANT LA MÊME FRÉQUENCE
   CIBLE (`target_frequency` en dict autorise des cibles différentes par entité).

2) FILTRE D'ORIGINE DE y_train ([SPEC] §5.3) :

       ELIGIBLE_ORIGINS = {
           False:             {'observed'},
           'covariates_only': {'observed', 'interpolated'},
           True:              {'observed', 'interpolated', 'model'},
       }

   TROIS POINTS IMPÉRATIFS :
   a. LE FILTRE PORTE SUR `origin_store`, PAS SUR LA MATRICE DE PROVENANCE. `DISAGGREGATED` est
      ambigu par construction : il marque aussi bien une cellule issue d'une interpolation recalée
      qu'une prédiction de modèle recalée. Utiliser la provenance publique comme filtre ferait de
      'covariates_only' un SYNONYME de True — c'est le piège principal de ce lot (décision D12).
   b. Chaque ligne de y_train porte LA FRÉQUENCE À LAQUELLE ELLE A ÉTÉ PRODUITE, lue dans
      `imputed_freq_store` ; le diviseur d'échelle est PAR LIGNE (point 3).
   c. y_train est composé des cellules de la colonne `v` dans le FRAME D'ÉTAPE, restreintes par
      la fenêtre `kind='training'`, puis filtrées par leur origine.

3) ÉCHELLE PAR LIGNE ([SPEC] §5.4) : les lignes de y_train issues d'une imputation antérieure sont
   à l'échelle de LEUR étape (des trimestres pour une variable annuelle imputée en Q) ; leur
   diviseur est `get_conversion_factor(pred_freq, f_ligne)`, PAS le scalaire de l'étape. Passer la
   `pd.Series[str]` de fréquences de production au `StageScaler.target_divisor` — le composant sait
   déjà le faire (lot 7). Correctif B12 inclus : ne pas court-circuiter quand la Series vaut 1.0
   partout par hasard.
   Sans le diviseur par ligne, la valeur annuelle 120 et la valeur trimestrielle 28 seraient mêlées
   telles quelles dans la même cible : le modèle apprendrait un mélange de deux échelles.

4) REPORT D'ÉTAPE, rang 3 de la précédence : le CovariateMaterializer l'implémente déjà (lot 9).
   Ce lot le rend ATTEIGNABLE en produisant des imputations à des étapes antérieures, et vérifie
   que les stores sont alimentés de manière à ce que le rang 3 se déclenche. C'est l'apport propre
   de 'covariates_only' : aux étapes M, `a1` et `a2` se voient mutuellement comme covariables
   REPORTÉES de l'étape Q, information strictement plus riche que l'interpolation de leurs seules
   ancres annuelles.

5) DOCUMENTER dans la docstring du paramètre, sans émettre d'avertissement (décision D9) :
   - 'covariates_only' SANS covariate_strategy='model' est INERTE quant aux valeurs finales — les
     rangs 2 et 3 ne sont jamais atteints, les étapes intermédiaires ne changent AUCUNE valeur ;
     elles restent visibles dans la sortie multi-fréquences si keep_lower_frequencies=True et
     coûtent du temps de calcul ;
   - True a un effet sous TOUTES les stratégies (il modifie y_train) ;
   - le mode « un seul fit réutilisé » (`cascade_refitting=False` de hfi) est ABANDONNÉ
     (décision D2) : s'il revient un jour, ce sera comme optimisation interne — mémoïsation d'un
     modèle dont le jeu d'entraînement et les voies de matérialisation n'ont pas changé entre deux
     étapes — jamais comme sémantique publique.

TESTS — à ajouter dans tests/frequency/test_high_frequency_imputer2.py :
   - `test_frequency_progression_on_reference_ts` : les trois cas du point 1, valeurs exactes ;
   - `test_progression_per_target_frequency_group_on_panel` ;
   - `test_stage_plan_of_spec_5_5` : reproduit le tableau « Sous 'covariates_only' ou True » du
     §5.5 — 2 étapes, 5 modèles, dans l'ordre (Q, a1), (Q, a2), (M, q1), (M, a1), (M, a2), avec
     y_train de 3 ancres partout sous 'covariates_only', et 3 ancres + 12 imputations Q pour
     `a1` et `a2` à l'étape M sous True ;
   - **I12** `test_covariates_only_differs_from_true` : sur un jeu où la cascade change quelque
     chose, 'covariates_only' produit des y_train SANS AUCUNE ligne d'origine 'model', et des
     valeurs finales DIFFÉRENTES de True ; et sous covariate_strategy='interpolate',
     'covariates_only' produit les MÊMES valeurs finales que False ;
   - `test_y_train_filter_reads_origin_store_not_provenance` : test ciblé sur le piège D12 —
     amorcer une cellule DISAGGREGATED d'origine 'interpolated' et une d'origine 'model', vérifier
     que sous 'covariates_only' seule la première entre dans y_train ;
   - `test_per_row_scale_factor_on_mixed_frequency_y_train` : reproduit le tableau chiffré du
     §5.4 (120/Y -> 10.0 ; 28/Q -> 9.33 ; 30/Q -> 10.0) ;
   - `test_carried_model_rank_reached_under_covariates_only` : au moins une étape porte une voie
     'carried_model' ;
   - **I6 complément** `test_target_taint_families` : MODEL_ON_IMPUTED_TARGET et
     MODEL_ON_IMPUTED_BOTH ne sont émis que sous impute_intermediate_frequencies=True ;
   - **I13** `test_no_boolean_test_on_parameter` : le test statique du lot précédent doit toujours
     passer après ce lot — c'est ici qu'il est le plus facile à casser.

Puis `uv run tests/frequency/check_regressions.py` et rapporter.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 15 — `transform`, `inverse_transform`, `keep_lower_frequencies` (L12)

**Modèle : Opus · Plan mode : OUI · Dépendances : prompt 14**

> Plan mode justifié : la frontière « rejoué depuis le fit » / « recalculé sur les données du
> transform » est le contrat central de l'architecture ; le §12.1 en donne le tableau, et toute
> erreur de répartition reproduit exactement les défauts B1/B7 que la réécriture existe pour
> supprimer.

````text
Contexte : suite du lot précédent sur tsforecast/frequency/high_frequency_imputer2.py, dont le fit
est complet sur les deux axes. Ce lot livre `transform`, `inverse_transform` et la sortie
multi-fréquences, et remplace les NotImplementedError posés au lot 12.
Référence : [SPEC] = high_frequency_imputer2_architecture.md, §12.1 (tableau « rejoué » vs
« recalculé », comportement D11), §12.4, §7.2 (règle B1), §12.5. Lire ces sections avant d'écrire.

PRINCIPE — `transform` NE REDÉCIDE RIEN. Il exécute les phases 0'-4' data-dépendantes puis rejoue
le plan figé étape par étape, EN APPELANT LA MÊME `_execute_step` que le fit, en mode « prédire
avec le modèle figé ». Ne pas écrire une seconde boucle d'exécution : c'est précisément ce qui a
fait diverger hfi (B7, B27).

TABLEAU DE RÉPARTITION ([SPEC] §12.1), à respecter à la lettre :

  REJOUÉ depuis le fit : classification des variables, fréquences détectées, progression de
    fréquences, ordre des variables, modèles ajustés, feature_cols, VOIE DE MATÉRIALISATION PAR
    COVARIABLE, facteurs et modalités d'échelle, méthodes et ancrages d'interpolation par feature,
    souillures de l'étape (covariate_taint, target_taint).

  RECALCULÉ sur les données du transform : fenêtres d'imputation et d'entraînement, frames
    d'étape, valeurs interpolées, prédictions, provenance du transform, masques de prédiction,
    recalage aux totaux, origin_store du transform.

CE QUE CE LOT DOIT LIVRER

1) PHASES 0'-4' du transform : alignement et nommage de `y` par LA MÊME FONCTION qu'au fit (B14) ;
   transformateur additif appliqué avec l'objet AJUSTÉ ; tracker de provenance initialisé APRÈS
   lui (B8) ; fenêtres RECALCULÉES (B1) ; contrôle des fréquences (point 3).

2) FENÊTRES AU TRANSFORM ([SPEC] §7.2, B1) : la fenêtre est une contrainte de DISPONIBILITÉ DES
   DONNÉES, pas un paramètre appris. Elle est RECALCULÉE sur les données transformées avec les
   HYPERPARAMÈTRES DU FIT, avec deux garde-fous :
     - ne JAMAIS vider une colonne sans la réécrire ;
     - AVERTIR UNE SEULE FOIS quand des lignes du périmètre sont hors fenêtre, par un message
       agrégé nommant le NOMBRE DE LIGNES et les ENTITÉS concernées.
   `fit_transform(X) ≡ fit(X).transform(X)` reste un invariant STRICT : le recalcul sur X redonne
   la fenêtre du fit.

3) CONTRÔLE DES FRÉQUENCES AU TRANSFORM (décision D11, [SPEC] §12.1) : la détection est REFAITE
   sur les données du transform et COMPARÉE à celle du fit.
     - divergence sur une colonne présente -> UserWarning UNIQUE, message listant les colonnes et
       LES DEUX fréquences, puis POURSUITE AVEC LES FRÉQUENCES DU FIT ;
     - colonne du fit ABSENTE des données du transform -> ValueError nommant les colonnes
       manquantes ;
     - colonnes supplémentaires au transform -> ignorées SILENCIEUSEMENT (elles ne sont dans aucun
       plan).

4) `inverse_transform` ([SPEC] §12.4) : reprise du chemin actuel de hfi — sélection du niveau de
   fréquence source, inversion du transformateur additif, restitution par masque ORIGINAL,
   `restore_original_values` — avec les invariants B4 (panels à n > 2 niveaux d'entité, NOMS
   D'INDEX PRÉSERVÉS) et B19.

5) `keep_lower_frequencies` : conservé, nom compris, documenté comme PARAMÈTRE D'AFFICHAGE PUR —
   il gouverne l'empilage multi-fréquences de la sortie, JAMAIS la logique. Sous
   `impute_intermediate_frequencies is False`, il n'y a pas de niveau intermédiaire à empiler : la
   sortie ne contient que le niveau source et le niveau cible. À documenter tel quel.
   Correctif B4 inclus (tous les niveaux d'entité préservés dans `_build_multifreq_output`).

6) AVERTISSEMENTS UNIQUES ([SPEC] §12.5) : estimateur absent, lignes hors fenêtre, fréquences
   divergentes, périodes à signe inversé — jamais un par variable × étape : accumulation puis
   message agrégé en fin de phase.

7) `imputation_provenance_` porte la matrice de provenance après fit PUIS après transform
   ([SPEC] §13.2) : documenter explicitement que l'attribut est écrasé par chaque transform, et
   purger l'état de transform en tête de fit (B19).

TESTS — à ajouter dans tests/frequency/test_high_frequency_imputer2.py :
   - **I1** `test_fit_transform_equals_fit_then_transform` : égalité STRICTE des valeurs, des
     provenances ET des attributs de sortie, sous les six combinaisons significatives des deux
     axes, sur TS et sur panel ;
   - **I7** `test_transform_outside_fit_window` : impute au lieu de vider, ne détruit JAMAIS une
     observation d'entrée, avertit UNE FOIS sur les lignes inimputables ;
   - **I8** `test_inverse_transform_roundtrip` : restitue l'index, les noms (panels multi-niveaux
     compris) et, sous `restore_original_values=True`, les valeurs d'origine ;
   - **I11** `test_materialization_identical_fit_and_transform` : pour chaque (étape, variable,
     covariable), `materialization` est identique au fit et au transform, et LA NATURE DES VALEURS
     PRODUITES l'est aussi ;
   - `test_transform_diverging_frequency_warns_once_and_uses_fit_frequencies` (D11) ;
   - `test_transform_missing_column_raises_naming_columns` (D11) ;
   - `test_transform_extra_column_ignored_silently` (D11) ;
   - `test_keep_lower_frequencies_is_display_only` : les valeurs du niveau cible sont identiques
     avec True et avec False ; sous impute_intermediate_frequencies=False, aucun niveau
     intermédiaire n'apparaît ;
   - **I9** `test_sklearn_conformance` : `clone`, `get_params`/`set_params`, `Pipeline`,
     `GridSearchCV` sur panel avec `target_frequency` dict ; `NotFittedError` avant fit ;
   - `test_transform_state_purged_at_fit` (B19).

Puis `uv run tests/frequency/check_regressions.py` et rapporter. À ce stade, les treize invariants
I1 à I13 de [SPEC] §16 doivent être couverts : me donner le tableau de correspondance
invariant -> test(s), et signaler explicitement tout invariant non couvert plutôt que de le
déclarer acquis.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Prompt 16 — Notebook 5 « pas à pas » (L13a)

**Modèle : Sonnet · Plan mode : Non · Dépendances : prompts 1 et 15**

> Plan mode inutile : le contenu attendu est énuméré section par section ; le notebook 4 fournit
> le modèle de structure.

````text
Contexte : HighFrequencyImputer2 est implémenté et testé (tsforecast/frequency/
high_frequency_imputer2.py et ses cinq composants). Il manque le notebook d'audit pas-à-pas, qui
est le livrable de vérification humaine de la classe.
Référence : [SPEC] = high_frequency_imputer2_architecture.md §15.2, et le notebook existant
notebooks/4 - QB - HighFrequencyImputer pas a pas.ipynb, dont il faut reprendre la STRUCTURE et
les contrôles croisés.

CIBLE — nouveau notebook `notebooks/5 - QB - HighFrequencyImputer2 pas a pas.ipynb`, exécuté sur
LES DEUX JEUX du notebook 3 (série temporelle et panel hétérogène avec `climat_affaires` absente
pour IT).

Sections attendues :

1) Détail des variables et des données à CHAQUE PHASE de `fit` puis de `transform` (phases 0 à 6,
   puis 0' à 4' et rejeu du plan). Une cellule markdown par phase, nommant la phase comme dans
   [SPEC] §12.3.

2) POUR CHAQUE ÉTAPE DU PLAN, afficher :
   - `X_train`, `y_train` et `X_pred` EXACTS ;
   - la VOIE DE MATÉRIALISATION retenue par covariable (`step.materialization`) ;
   - les trois masques de fenêtre appliqués ('strict', 'imputation', 'training') ;
   - les DEUX SOUILLURES de l'étape (`covariate_taint`, `target_taint`) et la provenance émise
     par `resolve_model_provenance` ;
   - la provenance APRÈS écriture.
   S'appuyer sur `ImputationPlan.to_diagnostic_frame()` pour la vue d'ensemble, et sur des
   affichages détaillés pour l'étape sélectionnée.

3) AUDIT DU SCALING : moyenne de chaque feature sur le train et sur le test, CÔTE À CÔTE. L'écart
   relatif doit être compatible avec la modalité choisie ('constant' vs 'calendar') ; un
   déséquilibre train/test est le symptôme IMMÉDIAT d'un diviseur faux. Inclure le cas mixte
   « feature 'calendar', y 'constant' ».

4) LES SIX COMBINAISONS des deux axes (3 stratégies × 3 modalités, réduites aux six qui diffèrent
   effectivement — voir [SPEC] §5.6 pour les combinaisons inertes), avec la matrice de provenance
   résultante et la répartition des cinq familles MODEL_*.

5) CONTRÔLES CROISÉS repris du notebook 4 :
   - pas-à-pas vs `fit_transform` : égalité stricte ;
   - `fit_transform(X)` vs `fit(X).transform(X)` : égalité stricte.

6) Une section finale sur le PANEL HÉTÉROGÈNE : mesure de l'invariant NaN PAR ENTITÉ, et
   illustration des deux valeurs de `covariate_eligibility` sur `climat_affaires` / IT.

Contraintes :
   - le notebook doit s'exécuter INTÉGRALEMENT, sans cellule en erreur — le rapporter
     explicitement, en indiquant le temps d'exécution total ;
   - illustrer les CAS D'USAGE VARIÉS et l'IMPACT DES DIFFÉRENTS PARAMÈTRES (règle CLAUDE.md
     sur les notebooks) ;
   - ne pas dupliquer de logique métier dans le notebook : tout affichage doit lire les attributs
     ajustés et le plan, jamais recalculer une imputation à la main.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style ; cellules markdown en français.
````

---

## Prompt 17 — Documentation de référence (L13b)

**Modèle : Sonnet · Plan mode : Non · Dépendances : prompt 16**

> Plan mode inutile : lot de rédaction, à surface fermée.

````text
Contexte : HighFrequencyImputer2 est implémenté, testé et audité par le notebook 5. Ce dernier lot
aligne la documentation du package.
Référence : [SPEC] = high_frequency_imputer2_architecture.md, qui est LA RÉFÉRENCE UNIQUE de
l'implémentation et de sa documentation — la documentation ne doit rien affirmer qui n'y figure.

1) DOCSTRING DE CLASSE de `HighFrequencyImputer2` : la relire intégralement contre [SPEC] §13 et
   la compléter. Elle doit contenir, en anglais Google Style :
   - le tableau des DEUX AXES du §0 et la question à laquelle chacun répond ;
   - `Args:` couvrant les 24 paramètres, chacun avec sa sémantique, son défaut, et le renvoi à la
     section de référence ;
   - `Attributes:` couvrant les attributs ajustés du §13.2 ;
   - `Examples:` : au moins trois usages — série temporelle sous les défauts, panel avec
     `target_frequency` en dict, et une configuration pseudo-temps réel
     (`imputation_scope='extended_forward'`, `training_scope='strict'`, cf. §7.3) ;
   - `Raises:` pour les erreurs de validation.
   Y faire figurer explicitement les mises en garde déjà décidées : prérequis NaN de
   'tolerate_nan' ; regard vers l'aval de l'interpolation ; combinaisons inertes (§5.6) sans
   avertissement ; `keep_lower_frequencies` paramètre d'affichage pur ; écrasement du total
   observé sous `aggregation_constraint=None` et les deux moyens de le récupérer (§11.2).

2) Docstrings des cinq composants (`CovariateMaterializer`, `StageScaler`, `VariableOrderer`,
   `AggregationConstraint`, `ImputationStep` v2 / `ImputationPlan`) : vérifier que chacune énonce
   sa RESPONSABILITÉ UNIQUE telle que définie au §12.2, et renvoie à la section de [SPEC] qui la
   spécifie. Corriger toute docstring qui décrit un comportement que le code n'a pas.

3) `tsforecast/frequency/__init__.py` : exports complets et cohérents des symboles nouveaux
   (`HighFrequencyImputer2`, les cinq composants, `ImputationStep2`/`ImputationPlan2`,
   `MaterializationWay`, `CellOrigin`, `Taint`, `resolve_model_provenance`), `__all__` à jour, et
   docstring de module mentionnant la COEXISTENCE de `HighFrequencyImputer` et
   `HighFrequencyImputer2` pendant la transition ([SPEC] §15.3). Ne rien déprécier : la
   dépréciation de hfi est un chantier ultérieur, hors périmètre.

4) `mkdocs` : ajouter les pages de référence des nouveaux modules dans docs/ et les entrées
   correspondantes dans mkdocs.yml, sur le modèle des pages existantes. Vérifier que la
   construction passe.

5) `CLAUDE.md`, section « Structure du Projet » puis « Objectifs Fonctionnels » : ajouter
   `high_frequency_imputer2.py` et les cinq composants sous `tsforecast/frequency/`, et déplacer
   « Gestion complète de données multi-fréquences (imputation, agrégation) » de « À implémenter »
   vers « Implémenté » en précisant que `HighFrequencyImputer` reste en place pendant la
   transition. Ne pas toucher au reste du fichier.

6) `README.md` : une entrée courte, cohérente avec le reste du fichier. Ne pas le réécrire.

Enfin : `uv run tests/frequency/check_regressions.py` une dernière fois, et me donner le bilan
final — compte d'échecs vs référence, tableau invariant I1-I13 -> test(s), et liste des points de
[SPEC] que l'implémentation ne couvre PAS, s'il en reste. Ne rien déclarer acquis sans l'avoir
vérifié.

Rappels de convention (CLAUDE.md) : commentaires internes en français à formulation nominale ;
docstrings en anglais Google Style avec Args/Returns/Raises/Examples ; type hints systématiques ;
localiser le code par nom de symbole, jamais par numéro de ligne.
````

---

## Résumé opérationnel

### Après chaque prompt

1. Relire le diff (`git diff`), en particulier les points de vigilance du lot.
2. `uv run tests/frequency/check_regressions.py` — aucun nouvel échec, sauf lots 2 et 4 qui
   régénèrent explicitement la référence.
3. Commit atomique, message préfixé par le lot (`feat(L6a): …`, `fix(L1a): …`).
4. Lancer le prompt suivant dans une session **neuve**.

### Ce qui ne doit jamais être perdu de vue d'un lot à l'autre

| Règle | Lots concernés |
|---|---|
| Une seule méthode produit `X_train` et `X_pred` | 8, 9, 13, 15 |
| La voie de matérialisation est choisie une fois et rejouée | 8, 9, 13, 15 |
| Le filtre de `y_train` et les souillures lisent `origin_store` | 4, 8, 13, 14 |
| `impute_intermediate_frequencies` n'est jamais testé comme booléen | 12, 13, 14 |
| Une seule implémentation d'exécution d'étape | 13, 15 |
| Le repli matérialise (stores alimentés) | 8, 13 |
| Avertissements agrégés et uniques | 11, 12, 13, 15 |
| `__init__` valide sans transformer | 7, 10, 11, 12 |
| `hfi` et `hfi2` coexistent, sauf `MODEL_ON_MIXED` | 4, 12, 17 |
