# Prompts — Tests (`utils`, `delays`, `frequency`) puis refactoring de `HighFrequencyImputer`

> Document généré le **2026-09-27**, révisé le même jour (branche `qb-mixed-frequencies`,
> commit `6be4102`). Chaque prompt est **autonome** et destiné à être collé tel quel dans une
> session Claude Code **indépendante**. Les prompts sont **ordonnés** ; les dépendances dures
> sont indiquées en tête de chacun, avec le **modèle**, le **plan mode** et le **niveau
> d'effort** (`/effort low | medium | high | xhigh | max`) à régler **avant** de coller le
> prompt. Après chaque prompt, relire le diff et le rapport d'anomalies avant de lancer le
> suivant.
>
> **Deux campagnes, dans cet ordre.**
> 1. **Tests** (parties A, U, D, F, C) : caractériser `tsforecast/utils`, `tsforecast/delays`
>    et `tsforecast/frequency` **sans modifier le code du package**. Les tests existants en
>    échec sont **revus et corrigés**, jamais ignorés.
> 2. **Refactoring** (partie R) : renommer `tsforecast/frequency`, rendre visible la hiérarchie
>    du module et découper `HighFrequencyImputer`, **à signature et comportement strictement
>    inchangés**, sous la protection des tests de la campagne 1 et d'un jeu de sorties de
>    référence figées (*golden master*).

---

## Sommaire

1. [Pourquoi les tests avant le refactoring](#1-pourquoi-les-tests-avant-le-refactoring)
2. [État vérifié du dépôt](#2-état-vérifié-du-dépôt-2026-09-27)
3. [Architecture cible de `tests/`](#3-architecture-cible-de-tests)
4. [Conventions communes à tous les prompts](#4-conventions-communes-à-tous-les-prompts)
5. [Récapitulatif, modèles, plan mode, effort](#5-récapitulatif)
6. Prompts — [A. Socle](#partie-a--socle) · [U. utils](#partie-u--tsforecastutils) ·
   [D. delays](#partie-d--tsforecastdelays) · [F. frequency](#partie-f--tsforecastfrequency) ·
   [C. Clôture](#partie-c--clôture-de-la-campagne-de-tests)
7. [Refactoring : proposition](#7-refactoring--proposition)
8. Prompts — [R. Refactoring](#partie-r--refactoring)

---

## 1. Pourquoi les tests avant le refactoring

Oui, les tests d'abord — à trois conditions, qui structurent toute la série :

1. **Tester des contrats, pas l'implémentation.** Un filet de sécurité de refactoring doit
   survivre au refactoring. Les tests ciblent l'API publique (`__init__`, `fit`, `transform`,
   méthodes et fonctions publiques, attributs ajustés suffixés `_` documentés) et les
   invariants fonctionnels. Un test qui appelle un symbole privé (`_xxx`) n'est admis que s'il
   n'existe aucun chemin public raisonnable ; il porte alors le marqueur `internal`, pour que
   le refactoring sache qu'il peut le réécrire ou le déplacer sans que ce soit une régression.
   Les tests actuels de `tests/frequency` font **~170 appels à des symboles privés** (69 dans
   `test_high_frequency_imputer.py`).
2. **Distinguer « comportement actuel » et « comportement attendu ».** Quand un comportement
   semble être une erreur, le test décrit le comportement **attendu**, marqué
   `xfail(strict=True)` et référencé dans `tests/ANOMALIES.md` (protocole §4.5). La suite reste
   verte, l'anomalie est documentée ; pendant le refactoring, un `xfail` qui se met à passer
   signale un **changement de comportement** (interdit) ; après le refactoring, sa correction
   fera passer le test en `XPASS` → échec strict → suppression du marqueur : la correction est
   prouvée.
3. **Une base verte, obtenue en corrigeant les tests, pas en les masquant.** Les 89 échecs et
   4 erreurs de collecte actuels (§2) sont triés un par un dans les prompts de leur module. Le
   mécanisme transitoire du prompt A0 (`tests/legacy_failures.txt`) ne sert qu'à garder la suite
   exploitable entre deux prompts : chaque prompt **retire** ses entrées de la liste, et le
   prompt C2 vérifie qu'elle est vide avant de la supprimer.

---

## 2. État vérifié du dépôt (2026-09-27)

Constats établis par exécution avant rédaction — ne pas les redécouvrir en session.

### 2.1 Suite complète

`uv run pytest tests/ --continue-on-collection-errors --cov=tsforecast --cov-branch` :
**89 failed, 1032 passed, 4 erreurs de collecte, 8 min 49 s.**

**Erreurs de collecte** (le fichier ne s'importe plus) :

| Fichier | Cause | Piste |
|---|---|---|
| `tests/delays/test_data_manager.py` | `_calculate_release_delays` introuvable | renommée `_calculate_publication_delays` (`tsforecast/delays/data_manager.py`) |
| `tests/panel/test_panelwise_transformer.py` | `from panelwise_transformer import …` | chemin d'import obsolète → `tsforecast.panel.transformers` |
| `tests/utils/frequency/test_parser.py` | `detect_and_parse_frequency` n'est plus exporté par `tsforecast.frequency` | primitives actuelles : `tsforecast.utils.parse.parse_frequency` / `build_frequency_string` |
| `tests/utils/time/test_utils.py` | collision avec le module standard `time` (`tests/utils/` sans `__init__.py`) | réglé par la nouvelle arborescence (A0) |

**Échecs d'exécution**, par fichier et cause dominante :

| Fichier | Échecs | Cause dominante observée | Nature probable |
|---|---:|---|---|
| `tests/delays/test_delay_calculator.py` | 40 | colonne `release_delay` attendue, le code exige `delay` | tests obsolètes : renommages délibérés (commits `4b3d3bc` *release_delay → delay*, `f302664` *output_unit → target_unit*, `ae347b2` *arguments + colonnes renvoyées*) |
| `tests/delays/test_integration.py` | 14 | mêmes renommages ; `KeyError` dans `PublicationDelayTransformer.fit` ; `ValueError` de `base/transformers.py` (« Cannot determine time index ») | à trier |
| `tests/delays/test_transformers.py` | 10 | `KeyError` dans `PublicationDelayTransformer.fit`, bornes de `MaskTransformer`, index non datetime | à trier |
| `tests/utils/frequency/test_normalizer.py` | 19 | alias pandas `'A'` / `'AS'` / `'AE'` rejetés ; `normalize_frequency(return_format=…)` | **abandon délibéré** de l'alias `'A'` (consolidation `parse_frequency`, acceptée par l'auteur) → tests obsolètes ; `return_format` à trier |
| `tests/utils/frequency/test_detector.py` | 5 | détection sur `Series` à `MultiIndex` 2 et 3 niveaux | à trier |
| `tests/frequency/test_converter_positions.py` | 1 | `FrequencyConverter._extend_index_for_upsampling` sur couple de fréquences non supporté | à trier |

### 2.2 Couverture actuelle (lignes, branches incluses dans le %)

**`tsforecast/frequency`** (≈ 88 % au total, 468 tests) :

| Module | Stmts | Miss | Couv. | Tests |
|---|---:|---:|---:|---:|
| `aggregation_constraint.py` | 195 | 11 | 92 % | 29 |
| `covariate_materializer.py` | 370 | 22 | 92 % | 29 |
| `frequency_aligner.py` | 111 | 10 | 90 % | 27 |
| `high_frequency_imputer.py` | 1023 | 95 | **88 %** | 163 |
| `imputation_plan.py` | 122 | 4 | 95 % | 14 |
| `imputation_window.py` | 374 | 28 | 90 % | 58 |
| `provenance.py` | 172 | 50 | **64 %** | 13 |
| `regularizer.py` | 132 | 112 | **11 %** | **0** |
| `stage_scaler.py` | 243 | 18 | 91 % | 47 |
| `target_frequency_validator.py` | 90 | 7 | 93 % | 23 |
| `training_set_builder.py` | 168 | 3 | 97 % | 27 |
| `variable_orderer.py` | 122 | 8 | 90 % | 18 |

**`tsforecast/utils`** :

| Module | Stmts | Miss | Couv. | Tests existants |
|---|---:|---:|---:|---|
| `abc/converter.py` · `abc/normalizer.py` | 11 · 19 | 2 · 7 | 82 % · 63 % | aucun dédié |
| `duration/converter.py` · `normalizer.py` · `utils.py` | 37 · 47 · 23 | 4 · 26 · 4 | 85 % · **45 %** · 83 % | aucun dédié |
| `frequency/converter.py` | 434 | 106 | **72 %** | 6 fichiers `tests/frequency/test_converter_*.py` (82 tests) |
| `frequency/detector.py` | 183 | 43 | **72 %** | `test_detector.py` (53) |
| `frequency/normalizer.py` · `utils.py` | 62 · 136 | 3 · 23 | 95 % · 84 % | `test_normalizer.py` (46) |
| `parse/utils.py` | 24 | 2 | 88 % | `test_parser.py` (29, ne s'importe pas) |
| `position/converter.py` · `normalizer.py` · `utils.py` | 99 · 31 · 22 | 82 · 11 · 9 | **12 %** · 62 % · 59 % | aucun dédié |
| `time/utils.py` | 100 | 35 | 62 % | `test_utils.py` (12, ne se collecte pas) |
| `validation/utils.py` | 186 | 77 | **56 %** | `test_validation.py` (8) |

**`tsforecast/delays`** :

| Module | Stmts | Miss | Couv. | Tests existants |
|---|---:|---:|---:|---|
| `calculator.py` | 95 | 54 | **38 %** | `test_delay_calculator.py` (47, dont 40 en échec) |
| `data_manager.py` | 101 | 30 | 66 % | `test_data_manager.py` (48, ne s'importe pas) |
| `transformers.py` (1 935 lignes) | 432 | 140 | **64 %** | `test_transformers.py` (77) + `test_integration.py` (19) |

**Hors campagne** (signalé, non traité) : `panel/transformers.py` 37 %, `base/transformers.py`
69 %, `xy/pipeline.py` 70 %, `crossvals/base.py` 84 %.

### 2.3 Sources de vérité

Les **docstrings n'ont pas toutes été relues** par l'auteur et peuvent diverger de
l'implémentation. Hiérarchie à appliquer :

| Module | Source prioritaire | Puis | En dernier |
|---|---|---|---|
| `frequency` | `high_frequency_imputer2_architecture.md` (spec, §0-§17, décisions D1-D39, invariants I1-I21) | le code | docstrings, `docs/` |
| `utils` | le code **et** l'historique git des changements délibérés (`git log -p -- <fichier>`) | `docs/concepts/temporal_utils.md`, notebooks `notebooks/utils/*.ipynb` (possiblement périmés : l'auteur y a relevé des problèmes, corrigés depuis dans le code) | docstrings |
| `delays` | le code et l'historique git (renommages `4b3d3bc`, `f302664`, `ae347b2`) | `docs/concepts/publication_delays.md`, notebooks `1 - QB - Delays`, `11 - QB - Delays`, `Test Delays transformers` | docstrings |

Un écart docstring ↔ code est une **anomalie de documentation** (`[DOC]`, §4.5), jamais une
raison d'aligner un test sur la docstring contre le code, ni sur le code contre le bon sens.

### 2.4 Jeux de données

- `tests/support/datasets.py` porte `build_mixed_frequency_timeseries` / `build_mixed_frequency_panel` (répliques du
  **notebook 2**, généralisées — voir ci-dessous), `build_panel_two_level` et
  `build_panel_reference` (jeu `PANEL-X` : colonnes `m1, q1, a1, a2, climat_affaires, v`,
  entités `FR/DE/IT`, valeurs d'or du §2 de la spec) et ses projections
  (`reference_timeseries`, `mixed_freq_panel_heterogeneous`, `mixed_freq_panel_multifrequency`).
- Le jeu du **notebook 3** (`notebooks/3 - QB - Panel a frequences mixtes heterogene.ipynb`,
  `create_timeseries_dataset` / `create_panel_dataset`) n'a **pas** de constructeur dédié : il
  fait doublon avec `build_mixed_frequency_timeseries` / `build_mixed_frequency_panel` sur deux de ses trois
  caractéristiques (`climat_affaires` absente pour une entité et fréquence hétérogène par
  entité pour une même colonne existent déjà dans `PANEL-X`, via `v`). Seule sa troisième
  caractéristique — couverture temporelle propre à chaque entité et index **réellement**
  irrégulier (dates annuelles antérieures à la grille mensuelle, pas seulement des NaN dans une
  grille régulière) — est un apport réel, absent des jeux existants avant ce prompt. Plutôt que
  d'ajouter des constructeurs `build_timeseries_nb3` / `build_panel_nb3` quasi dupliqués,
  `build_mixed_frequency_timeseries` et `build_mixed_frequency_panel` ont été **étendus** avec les paramètres qui
  manquaient (`annual_start_date`, et pour le panel : couverture par entité, fréquence de
  publication par entité, covariable structurellement absente) ; les valeurs par défaut de ces
  deux fonctions restent inchangées (jeu régulier historique), et le dictionnaire
  `HETEROGENEOUS_PANEL_COUNTRIES` reproduit fidèlement `create_panel_dataset` en passant
  `countries=HETEROGENEOUS_PANEL_COUNTRIES` à `build_mixed_frequency_panel`. Les fixtures
  `irregular_index_timeseries` / `heterogeneous_coverage_panel` (§4.2.4 et tous les prompts qui
  les utilisent) sont construites par ces fonctions généralisées, pas par des constructeurs
  séparés. Ces quatre noms (les deux constructeurs et les deux fixtures) ont été renommés après
  A1 pour ne plus mentionner le notebook d'origine (`build_timeseries_nb2` →
  `build_mixed_frequency_timeseries`, `build_panel_nb2` → `build_mixed_frequency_panel`,
  `nb3_timeseries` → `irregular_index_timeseries`, `nb3_panel` → `heterogeneous_coverage_panel`) ;
  ce document reflète déjà les noms actuels partout sauf dans le texte des prompts A0/A1
  eux-mêmes (§6), laissé inchangé car déjà exécuté.
- ⚠️ Le notebook 3 amorce ses graines par `np.random.seed(seed + hash(country) % 1000)` :
  `hash` d'une `str` est salé par processus, **son panel change à chaque exécution**. Comme
  `build_mixed_frequency_panel`, `HETEROGENEOUS_PANEL_COUNTRIES` passe par `zlib.crc32` : seule la
  **structure** du notebook est reproduite, pas ses valeurs exactes.
- Références aux anciens chemins de tests : `notebooks/5 - QB - HighFrequencyImputer pas a
  pas.ipynb` (importe `_build_panel_reference` par `importlib` depuis
  `tests/frequency/conftest.py`), `notebooks/utils/{frequency_aligner,frequency_converter,
  imputation_window}.ipynb`. Les `.md` historiques de la racine (`high_frequency_imputer*_*.md`)
  sont des archives : **ne pas les modifier** (sauf la spec, explicitement, au prompt R9).

### 2.5 Divers

- Dossier `tests/utils/validation.py/` (un **dossier** nommé `.py`).
- `tests/run_tests.py` ne lance que `tests/crossvals/`, alors que `CLAUDE.md` et `README.md`
  annoncent `--mode all|unit|integration`.
- Aucune configuration pytest / coverage dans `pyproject.toml` ; marqueurs `slow` /
  `integration` déclarés dans `tests/conftest.py::pytest_configure`. Pas de CI (`.github/` absent).
- `tests/frequency/BASELINE_FAILURES.txt` + `check_regressions.py` : mécanisme hérité de la
  campagne HFI2, **remplacé** (A0).
- `tests/frequency/` seul dure **5 min 13 s** ; les tests les plus lents (5-7 s) sont dans
  `test_high_frequency_imputer.py`.

---

## 3. Architecture cible de `tests/`

```
tests/
├── conftest.py              # racine : pytest_plugins, marquage auto unit/integration,
│                            #          xfail TRANSITOIRE des échecs hérités
├── legacy_failures.txt      # TRANSITOIRE : vidé prompt après prompt, supprimé en C2
├── ANOMALIES.md             # registre des anomalies (protocole §4.5)
├── run_tests.py             # --mode all | unit | integration | fast | coverage
├── support/                 # paquet partagé : `from tests.support import ...`
│   ├── datasets.py          # constructeurs PURS : PANEL-X, notebook 2 (généralisé notebook 3)
│   ├── fixtures.py          # fixtures pytest, enregistrées par `pytest_plugins`
│   ├── perturbations.py     # désordre, noms spéciaux, positions S/E, index 3 niveaux…
│   ├── estimators.py        # SpyEstimator, FailingEstimator, ConstantEstimator
│   ├── golden/              # sorties de référence figées (partie R)
│   └── test_datasets.py     # tests des jeux eux-mêmes
├── unit/                    # miroir EXACT de tsforecast/
│   ├── base/  crossvals/  panel/  tracking/  xy/
│   ├── delays/              # test_calculator.py, test_data_manager.py, transformers/…
│   ├── frequency/           # un test_<module>.py par module ; high_frequency_imputer/ (paquet)
│   └── utils/
│       ├── abc/  duration/  parse/  position/  time/  validation/
│       └── frequency/       # test_detector.py, test_normalizer.py, test_utils.py, converter/ (paquet)
└── integration/
    ├── crossvals/  delays/  frequency/
```

**Règle de miroir.** `tsforecast/<paquet>/<module>.py` ↔ `tests/unit/<paquet>/test_<module>.py`.
Un module source trop gros pour un seul fichier de test (plus de ~1 200 lignes de tests) devient
un **paquet** de même nom sans préfixe `test_` : `tests/unit/<paquet>/<module>/test_<thème>.py`
(cas de `high_frequency_imputer`, `utils/frequency/converter`, `delays/transformers`).

**Frontière unit / integration.** *Unit* : un composant de `tsforecast/` (et ses
collaborateurs réels quand les remplacer par des doublures n'apporte rien), sur des données
**petites, construites à la main, à valeurs d'or calculables**. *Integration* : plusieurs
composants publics enchaînés, ou un composant sur les jeux **réalistes** du notebook 3.

---

## 4. Conventions communes à TOUS les prompts

Chaque prompt demande de relire cette section : elle fait partie du prompt.

### 4.1 Rédaction

1. Commentaires internes **en français**, formulations **nominales** et détaillées, expliquant
   la **logique** du test : ce qui est construit, pourquoi cette donnée, d'où vient la valeur
   d'or (`# Valeur d'or : 120 / 12 = 10 par mois, recalage exact sur le total annuel`).
   ✅ `# Construction d'un panel désordonné` — ❌ `# Construire un panel désordonné`.
2. Docstrings **en anglais**, **Google Style** :
   - module de test : résumé + paragraphe listant le périmètre couvert (symboles testés) ;
   - classe de test : le comportement ou contrat regroupé ;
   - fonction de test : une ligne de résumé (« Rescaled sub-periods sum to the observed annual
     total. »), puis au besoin un paragraphe sur le scénario ;
   - fonctions de `tests/support/` : `Args:` / `Returns:` / `Raises:` / `Examples:` (doctest
     quand c'est possible), type hints systématiques.
3. Organisation : classes `Test<Contrat>` ; `pytest.mark.parametrize` avec `ids=` lisibles
   plutôt que des boucles ; une assertion **principale** par test.

### 4.2 Données

1. Jeux de données via `tests/support/` (fixtures ou constructeurs) ; un petit frame local à
   valeurs d'or reste bienvenu **dans** le test quand il en est le sujet.
2. Déterminisme : `np.random.default_rng(seed)` ou `np.random.seed` explicite, jamais `hash()`.
3. Cas limites de `CLAUDE.md` pour chaque composant : données non triées, index dupliqués,
   entités manquantes, fréquences irrégulières, NaN (colonne / entité entièrement NaN), jeu
   vide, une seule observation ; **robustesse d'index** : `DatetimeIndex` seul vs `MultiIndex`
   2 et 3 niveaux, noms d'index non standards, positions début (`MS`, `QS`, `YS`) **et** fin
   (`ME`, `QE`, `YE`), noms de colonnes avec espaces / accents / caractères spéciaux, types
   `datetime64`, `Period`, `Timestamp` là où l'API les accepte. Utiliser
   `tests/support/perturbations.py`.
4. Chaque composant qui accepte des jeux de données est testé au moins une fois sur les jeux
   **réalistes** du notebook 3 (`irregular_index_timeseries`, `heterogeneous_coverage_panel`) — à défaut de valeur d'or, sur
   des **propriétés** (pas de NaN introduit hors fenêtre, totaux conservés, index restitué…).
5. Spécifique `frequency` : une fréquence détectée est une propriété du couple
   **(entité, colonne)** ; `impute_intermediate_frequencies` ne se teste jamais par vérité
   booléenne (`'covariates_only'` est *truthy*) : `is False`, `== 'covariates_only'`, `is True`.

### 4.3 Portée (campagne de tests)

1. **Ne jamais modifier `tsforecast/`** pendant les parties A, U, D, F, C. Si un test ne peut
   pas être écrit sans modifier le code, s'arrêter et le signaler.
2. API publique d'abord. Test d'un symbole privé : seulement sans chemin public raisonnable, et
   marqué `@pytest.mark.internal`.
3. Localiser le code **par nom de symbole**, jamais par numéro de ligne.
4. Ne pas committer : l'auteur relit et committe lui-même.

### 4.4 Esprit critique : tri des tests existants en échec

Pour **chaque** test migré en échec (et chaque test existant douteux, même vert), établir la
cause **avant** de toucher au test, et la classer :

| Catégorie | Critère | Action |
|---|---|---|
| **(a) Test obsolète** | changement **délibéré** de l'API : prouvé par l'historique git (`git log -S<symbole>`, message de commit explicite), la spec ou la mémoire de l'auteur | réécrire le test sur l'API actuelle, en conservant l'**intention** du test ; supprimer s'il n'a plus d'objet |
| **(b) Code faux** | le test exprime ce que la classe **doit** faire (spec, finalité, cohérence avec les autres composants) et le code s'en écarte | garder le test (le rendre correct si besoin), `xfail(strict=True)`, entrée dans `ANOMALIES.md` |
| **(c) Test faux** | le test encode lui-même une attente erronée (valeur d'or fausse, hypothèse contredite par la spec) | corriger le test, justifier dans le rapport |
| **(d) Indécidable** | aucune source ne tranche | épingler le comportement actuel, entrée `à arbitrer` dans `ANOMALIES.md` |

**Règle absolue** : ne jamais « faire passer » un test en l'alignant sur un comportement qui
semble contraire à la finalité de la classe. Un test vert qui fige un bogue est pire qu'un test
absent. En cas de doute entre (a) et (b), l'historique git tranche ; à défaut, (d).

Même discipline pour les **nouveaux** tests : écrire l'attente **avant** d'exécuter le code
(valeur d'or dérivée de la spec ou du calcul à la main), puis confronter. Ne jamais recopier la
sortie du code comme valeur d'or sans l'avoir vérifiée indépendamment.

### 4.5 Protocole anomalie

1. Test du comportement **attendu**, marqué
   `@pytest.mark.xfail(strict=True, reason="ANO-<MOD>-NNN: <résumé>")`, avec `<MOD>` ∈
   {`UTILS`, `DELAYS`, `FREQ`} ;
2. entrée dans `tests/ANOMALIES.md`, section du module :

   ```markdown
   ### ANO-FREQ-NNN — <titre court>
   - **Type** : [CODE] comportement | [DOC] docstring ≠ code | [SPEC] code ≠ spec
   - **Composant** : `tsforecast/<paquet>/<module>.py::<Symbole>`
   - **Sévérité** : bloquante | majeure | mineure | cosmétique | à arbitrer
   - **Observé** : <comportement, message d'erreur exact>
   - **Attendu** : <comportement et justification : spec, finalité, cohérence>
   - **Reproduction** : <5-10 lignes minimales>
   - **Test** : `tests/unit/<…>.py::<Classe>::<test>`
   - **Statut** : ouvert
   ```

3. une anomalie **[DOC]** n'a pas de `xfail` (le test suit le code, qui est juste) ;
4. numérotation continue par préfixe (lire le dernier numéro dans `tests/ANOMALIES.md`).

### 4.6 Critères d'acceptation (tous les prompts de tests)

- `uv run pytest <fichiers du prompt> -q` : **0 failed, 0 error** ; un `XPASS(strict)` est un
  échec.
- Couverture du module ciblé, lignes **et** branches :
  `uv run pytest <tests> --cov=<module pointé> --cov-branch --cov-report=term-missing`
  — objectif **≥ 95 %**, minimum **90 %** ; chaque ligne encore non couverte est **justifiée**
  (code mort ? branche défensive inatteignable ? → candidat anomalie).
- Les entrées du module dans `tests/legacy_failures.txt` sont **retirées** ; le module ne
  figure plus dans `collect_ignore`.
- `uv run pytest tests/ -q -m "not slow"` : aucun échec nouveau.
- Aucun test ne dépasse **2 s** sans le marqueur `slow`.

### 4.7 Rapport final (fin de CHAQUE prompt)

Rapport en français :
1. fichiers créés / déplacés / supprimés, nombre de tests avant → après ;
2. couverture avant → après, lignes non couvertes justifiées ;
3. **tri des tests en échec** : tableau test → catégorie (a/b/c/d) → preuve (commit, § de spec)
   → action ;
4. tests supprimés et pourquoi ;
5. **récapitulatif des anomalies** (ids, type, sévérité, une phrase chacune) et tout
   comportement surprenant non consigné ;
6. questions ouvertes pour l'auteur.

---

## 5. Récapitulatif

| # | Objet | Modèle | Plan | Effort | Dépend de |
|---|---|---|---|---|---|
| **A0** | Arborescence, migration de **tous** les tests, config pytest/coverage, liste transitoire | Sonnet | **Oui** | medium | — |
| **A1** | `tests/support` : fixtures notebook 3, perturbations, doublures | Sonnet | Non | medium | A0 |
| **U1** | `utils/parse` + `utils/abc` (réparation de `test_parser.py`) | Sonnet | Non | medium | A0 |
| **U2** | `utils/duration` | Sonnet | Non | medium | A0 |
| **U3** | `utils/position` (12 %) | Opus | Non | high | A0 |
| **U4** | `utils/time` | Sonnet | Non | medium | A0 |
| **U5** | `utils/validation` | Sonnet | Non | high | A1 |
| **U6** | `utils/frequency` : `normalizer` + `utils` (19 échecs `'A'`) | Sonnet | Non | high | U1 |
| **U7** | `utils/frequency/detector.py` (échecs `MultiIndex`) | Opus | Non | high | A1, U6 |
| **U8** | `utils/frequency/converter.py` (1 845 lignes, 72 %) | Opus | **Oui** | xhigh | U3, U6 |
| **D1** | `delays/data_manager.py` | Sonnet | Non | high | A1, U4 |
| **D2** | `delays/calculator.py` (40 échecs) | Sonnet | Non | high | D1 |
| **D3** | `delays/transformers.py` : `ShiftTransformer`, `MaskTransformer` | Opus | Non | high | A1 |
| **D4** | `delays/transformers.py` : `PublicationDelayTransformer`, fabriques | Opus | **Oui** | high | D2, D3 |
| **D5** | Intégration `delays` | Opus | Non | high | D4 |
| **F1** | `provenance.py` + `imputation_plan.py` | Sonnet | Non | medium | A1 |
| **F2** | `regularizer.py` (0 test) + `target_frequency_validator.py` | Sonnet | Non | medium | A1 |
| **F3** | `frequency_aligner.py` | Opus | Non | high | A1 |
| **F4** | `imputation_window.py` | Opus | Non | high | A1 |
| **F5** | `aggregation_constraint.py` | Opus | Non | high | A1 |
| **F6** | `stage_scaler.py` | Opus | Non | high | A1 |
| **F7** | `covariate_materializer.py` | Opus | **Oui** | xhigh | F1, F5 |
| **F8** | `training_set_builder.py` | Opus | Non | high | F4, F6, F7 |
| **F9** | `variable_orderer.py` | Sonnet | Non | medium | A1 |
| **F10** | HFI 1/3 : découpage du fichier, paramètres, conformité sklearn | Opus | **Oui** | high | F1–F9 |
| **F11** | HFI 2/3 : `fit`, cas limites, robustesse d'index | Opus | Non | xhigh | F10 |
| **F12** | HFI 3/3 : `transform`, `inverse_transform`, sortie multi-fréquences | Opus | Non | xhigh | F11 |
| **F13** | Intégration `frequency` : notebook 3, `XYPipeline`, crossvals, délais, tracking | Opus | **Oui** | high | F12, D5 |
| **C1** | Couverture : CI GitHub Actions, Codecov, badges `README.md`, `CLAUDE.md` | Sonnet | Non | medium | toutes U, D, F |
| **C2** | Bilan : liste transitoire vide, `ANOMALIES.md` consolidé | Opus | Non | high | C1 |
| **R0** | *Golden master* : sorties de référence figées de `HighFrequencyImputer` | Opus | Non | high | C2 |
| **R1** | Renommage `frequency` → `imputation`, hiérarchie, relocalisations, tests miroir | Sonnet | **Oui** | high | R0 |
| **R2** | Extraction : validation des paramètres | Sonnet | Non | medium | R1 |
| **R3** | Extraction : état d'exécution (`_RunContext`) et avertissements agrégés | Opus | **Oui** | xhigh | R2 |
| **R4** | Extraction : planification des fréquences (`FrequencyLayout`) | Opus | Non | high | R3 |
| **R5** | Mutualisation `fit` / `transform` : préparation commune (phases 0-4) | Opus | **Oui** | xhigh | R4 |
| **R6** | Extraction : mise à l'échelle par ligne (`RowScaler`) | Opus | Non | high | R5 |
| **R7** | Extraction : exécution d'étape (`StageRunner`) | Opus | **Oui** | xhigh | R6 |
| **R8** | Extraction : rejeu du plan et assemblage de la sortie | Opus | Non | xhigh | R7 |
| **R9** | Finalisation : tests `internal`, docs mkdocs, spec, `CLAUDE.md` | Sonnet | Non | medium | R8 |

**Parallélisables** : après A0, {U1, U2, U3, U4} ; après A1, {U5, D3, F1, F2, F3, F4, F5, F6,
F9} ; seul `tests/ANOMALIES.md` est partagé (numéros à relire au moment de l'écriture).
L'ordre recommandé reste **U → D → F** : `frequency` consomme `utils/frequency` (le
convertisseur surtout), et une anomalie du convertisseur trouvée en U8 évite de la
re-diagnostiquer trois fois en F. La partie R est **strictement séquentielle**.

### Critère de choix du modèle

**Opus** dès que la valeur d'or d'un test ou la sûreté d'une extraction exige de raisonner sur
des **échelles de fréquence**, des **positions de période**, des **fenêtres**, des **délais
relatifs à une date de référence** ou la **symétrie fit / transform** : un test faux y passe au
vert sans rien signaler, et c'est là que se trouvent les anomalies. **Sonnet** pour les lots à
surface fermée et critère binaire : migration mécanique, renommages délibérés à répercuter,
énumérations, configuration CI, extractions sans état.

### Critère de choix du plan mode

**Oui** quand le prompt prend une décision de **structure** consommée par les suivants
(arborescence en A0 et R1, découpage d'un gros fichier de tests en U8 / F10, matrice de
scénarios en D4 / F7 / F13, frontières des objets extraits en R3 / R5 / R7). **Non** quand le
prompt décrit entièrement le livrable.

### Critère de choix de l'effort

- **medium** : lot mécanique ou module court dont le contrat se lit d'un coup.
- **high** : module de taille moyenne avec tri d'échecs, ou valeurs d'or calendaires.
- **xhigh** : plus de 1 500 lignes de code sous test, ou extraction touchant l'état partagé entre
  `fit` et `transform` — un oubli y produit une divergence silencieuse.
- **max** : non utilisé ; le réserver à la reprise d'un prompt `xhigh` qui a échoué.

---

## Partie A — Socle

### Prompt A0 — Nouvelle arborescence, migration de tous les tests, configuration

**Modèle : Sonnet · Plan mode : Oui · Effort : medium · Dépendances : —**

```text
Contexte : dépôt ts-forecast (package Python tsforecast), branche qb-mixed-frequencies. Lis
d'abord CLAUDE.md, puis les sections 2, 3 et 4 de tests_and_refactoring_prompts.md (racine) : état
vérifié de la suite, arborescence cible, conventions. Ce prompt est une MIGRATION : aucun test
nouveau, aucun changement dans tsforecast/.

Objectif : réorganiser tests/ en tests/unit (miroir exact de tsforecast/), tests/integration et
tests/support, configurer pytest et coverage, et obtenir une suite exploitable (0 failed, 0
error) qui collecte au moins les mêmes tests qu'avant.

Étapes :

1. Relevé de référence : `uv run pytest tests/ --collect-only -q --continue-on-collection-errors`
   → nombre de tests collectés et liste des node ids, sauvegardés dans le scratchpad.

2. Déplacements avec `git mv` (historique conservé) selon le §3 :
   - tests/<module>/test_*.py → tests/unit/<module>/ pour base, crossvals, delays, panel,
     tracking, xy, utils/frequency, utils/time ;
   - tests/utils/validation.py/ → tests/unit/utils/validation/ (dossier nommé à tort .py) ;
   - tests/crossvals/test_integration.py → tests/integration/crossvals/ ;
     tests/delays/test_integration.py → tests/integration/delays/ ;
   - tests/frequency/test_converter_*.py (6 fichiers, ils testent FrequencyConverter de
     tsforecast/utils/frequency/converter.py) → tests/unit/utils/frequency/converter/ (paquet,
     règle de miroir du §3), renommés test_<thème>.py (test_positions.py, test_subperiods.py…) ;
   - tests/frequency/test_*.py restants → tests/unit/frequency/ (même nom) ;
   - tests/delays/test_delay_calculator.py → tests/unit/delays/test_calculator.py (miroir de
     calculator.py) ;
   - tests/frequency/test_reference_datasets.py → tests/support/test_datasets.py.
   Chaque dossier de test reçoit un __init__.py (règle la collision tests/utils/time ↔ module
   standard time).

3. tests/support/ :
   - datasets.py : y déplacer SANS changer la logique les constructeurs de
     tests/frequency/conftest.py, rendus publics (build_timeseries_nb2, build_panel_nb2,
     build_panel_two_level, build_panel_reference), docstrings Google complètes ;
   - fixtures.py : toutes les fixtures de tests/frequency/conftest.py et de tests/conftest.py,
     MÊMES noms ; enregistrement par `pytest_plugins = ["tests.support.fixtures"]` dans
     tests/conftest.py ;
   - estimators.py : _SpyEstimator et _FailingEstimator de test_high_frequency_imputer.py,
     renommés SpyEstimator / FailingEstimator (alias locaux tolérés jusqu'au prompt F10) ;
   - supprimer tests/frequency/conftest.py une fois vide, puis le dossier tests/frequency.

4. pyproject.toml :
   - [tool.pytest.ini_options] : testpaths = ["tests"], pythonpath = ["."],
     addopts = "--strict-markers", markers unit, integration, slow, internal (une ligne de
     description chacun) ; retirer tests/conftest.py::pytest_configure ;
   - [tool.coverage.run] source = ["tsforecast"], branch = true ;
     [tool.coverage.report] show_missing = true, exclude_also = ["if TYPE_CHECKING:",
     "raise NotImplementedError"] ;
   - hook pytest_collection_modifyitems dans tests/conftest.py : marqueur unit ou integration
     ajouté selon le chemin (tests/support compte comme unit).

5. Échecs hérités — mécanisme TRANSITOIRE, pas un masquage (§1, point 3) :
   - imports cassés : réparer ceux dont la cause est un simple chemin d'import (panel :
     `from panelwise_transformer import …` → tsforecast.panel.transformers) ; pour
     test_data_manager.py (_calculate_release_delays → _calculate_publication_delays) et
     test_parser.py (detect_and_parse_frequency supprimé), NE PAS réécrire : les lister dans
     `collect_ignore` de tests/conftest.py avec un commentaire « à traiter au prompt D1 / U1 » ;
   - échecs d'exécution : node ids (NOUVEAUX chemins) dans tests/legacy_failures.txt, un par
     ligne, groupés par fichier, chaque groupe précédé d'un commentaire « # prompt <id> »
     (D2, D3, D4, D5, U6, U7, U8) ; le hook applique
     `pytest.mark.xfail(strict=False, reason="legacy: à trier au prompt <id>")` ; les
     échecs de panel / base / xy éventuellement révélés par la réparation d'import vont sous
     « # hors campagne » ;
   - en-tête de tests/legacy_failures.txt : il est transitoire, chaque prompt retire ses
     entrées, C2 exige qu'il ne reste que « hors campagne » ;
   - créer tests/ANOMALIES.md : en-tête reprenant les §4.4 et §4.5, trois sections UTILS,
     DELAYS, FREQ vides ;
   - supprimer tests/frequency/BASELINE_FAILURES.txt et check_regressions.py.

6. tests/run_tests.py réécrit pour tout le package : modes all | unit | integration | fast
   (-m "not slow") | coverage (--cov=tsforecast --cov-branch, rapports term-missing, html, xml)
   et --path pour une cible libre ; docstrings Google, commentaires français.

7. Références aux anciens chemins : notebooks/5 - QB - HighFrequencyImputer pas a pas.ipynb
   (import importlib → `from tests.support.datasets import build_panel_reference`),
   notebooks/utils/{frequency_aligner,frequency_converter,imputation_window}.ipynb, README.md
   (section Test), CLAUDE.md (Stratégie de Tests : arborescence, commandes). Ne PAS toucher aux
   high_frequency_imputer*_*.md.

8. Marquage slow : `uv run pytest tests/ --durations=0 -q`, marquer slow tout test > 2 s.

Critères d'acceptation :
- `uv run pytest tests/ -q` : 0 failed, 0 error ; tous les xfailed viennent de
  legacy_failures.txt ;
- tests collectés = relevé de l'étape 1, moins les deux fichiers de collect_ignore, plus ceux
  de test_panelwise_transformer.py et tests/unit/utils/time désormais collectés ; lister toute
  autre différence ;
- `uv run python tests/run_tests.py --mode coverage` produit coverage.xml ;
- le notebook 5 s'exécute sans erreur (nbconvert --execute) ;
- `git status` : aucun fichier modifié sous tsforecast/.

Rapport final : §4.7, plus le tableau ancien chemin → nouveau chemin et la durée de
`-m "not slow"`.
```

### Prompt A1 — `tests/support` : fixtures notebook 3, perturbations, doublures

**Modèle : Sonnet · Plan mode : Non · Effort : medium · Dépendances : A0**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md puis les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md. A0 a créé tests/support/ (datasets.py, fixtures.py,
estimators.py, test_datasets.py). Aucun changement dans tsforecast/.

tests/support/datasets.py a déjà été étendu (avant ce prompt, hors campagne) pour couvrir les
jeux REALISTES du notebook `notebooks/3 - QB - Panel a frequences mixtes heterogene.ipynb` :
pas de nouveaux constructeurs `build_timeseries_nb3` / `build_panel_nb3` (doublons évités, §2.4),
mais `build_timeseries_nb2` (paramètre `annual_start_date`) et `build_panel_nb2` (paramètre
`countries`, dictionnaire `HETEROGENEOUS_PANEL_COUNTRIES` fourni par le module) généralisés
pour produire ce jeu à l'identique quand on le leur demande. Les valeurs par défaut de ces deux
fonctions sont inchangées (vérifié : sorties bit-identiques à avant l'extension). Objectif de
ce prompt : les fixtures, leurs tests, et une boîte à outils de perturbations d'index.

1. tests/support/fixtures.py :
   - `nb3_timeseries` : `build_timeseries_nb2(annual_start_date='2015-01-01')`.
   - `nb3_panel` : `build_panel_nb2(countries=HETEROGENEOUS_PANEL_COUNTRIES)`.
   - Fixture de session privée + fixture de fonction publique renvoyant `.copy()` (contre les
     mutations croisées) pour ces deux jeux ; même traitement pour les jeux existants coûteux
     (`mixed_freq_timeseries`, `mixed_freq_panel`, `panel_two_level_dataset`,
     `panel_reference_full` et ses projections).

2. tests/support/test_datasets.py, classe TestNotebook3Datasets : chacune des caractéristiques
   annoncées par le notebook (cellules 0, 6, 8, 10, 11), exercées via les fixtures
   `nb3_timeseries` / `nb3_panel` (pas d'appel direct à un constructeur `nb3` : il n'y en a pas) :
   - couverture propre à chaque entité (FR 2018-01→2024-07, DE 2018-07→2024-04,
     IT 2019-01→2024-07 pour la grille mensuelle) ;
   - index irrégulier (dates annuelles antérieures à la grille mensuelle, pour
     `nb3_timeseries` comme pour chaque entité de `nb3_panel`) : `is_regular` de
     tsforecast.frequency doit le dire irrégulier ;
   - depenses_publiques_pib : annuelle FR/IT, trimestrielle DE, dernière valeur retirée ;
   - climat_affaires : observée FR/DE, zéro observation IT, colonne présente pour IT ;
   - délais : dernière ligne NaN pour inflation_ipc et taux_chomage ;
   - reproductibilité : deux appels égaux ; 3-4 valeurs d'or relevées une fois et écrites en dur.
   - Vérifier aussi (régression) que `build_timeseries_nb2()` et `build_panel_nb2()` sans
     argument restent des jeux réguliers (`is_regular` vrai), sans `depenses_publiques_pib` ni
     `climat_affaires` : la généralisation ne doit pas avoir contaminé le jeu par défaut.

3. tests/support/perturbations.py — fonctions pures, documentées, testées (TestPerturbations) :
   shuffle_rows(df, seed) ; reverse_entities(df) ; with_special_column_names(df) →
   (df, mapping) (espaces, accents, '/', '%', '(') ; with_index_names(df, names) ;
   to_three_level_index(df) (niveau région ajouté) ; to_period_start(df) / to_period_end(df)
   (bascule début ↔ fin de période en conservant la fréquence propre de chaque couple
   (entité, colonne) — documenter la règle) ; to_period_index(df) (PeriodIndex, là où c'est
   pertinent) ; drop_entity(df, entity) ; with_duplicated_rows(df, n) ;
   single_observation(df) ; empty_like(df). Chaque fonction : docstring Google + Examples,
   commentaire français sur le cas limite fabriqué (renvoi à CLAUDE.md « Priorités de test »).

4. tests/support/estimators.py : SpyEstimator (enregistre X / y de chaque fit et predict),
   FailingEstimator (lève à la demande), ConstantEstimator (prédit une constante, valeurs d'or
   triviales). Docstrings complètes.

Critères : §4 ; `uv run pytest tests/support -q` et
`uv run pytest --doctest-modules tests/support -q` verts ; suite complète verte.
Rapport §4.7. Signaler toute caractéristique annoncée par le notebook que son code ne produit
PAS réellement (anomalie du jeu de référence).
```

---

## Partie U — `tsforecast/utils`

> Contexte commun aux prompts U : l'auteur a vérifié ces utilitaires avec les notebooks
> `notebooks/utils/*.ipynb`, **qui peuvent être périmés** (des problèmes qu'ils ont soulevés ont
> été corrigés depuis dans le code). Une divergence notebook ↔ code se tranche par l'historique
> git, pas par le notebook. Consolidation passée à connaître : `parse_frequency` /
> `build_frequency_string` (`tsforecast/utils/parse`) sont les **seules** primitives de chaîne
> de fréquence ; `decompose_offset`, `combine_frequency_position`,
> `extract_position_from_offset` n'existent plus, et **l'abandon des alias pandas `'A'` / `'AS'`
> / `'AE'` a été accepté par l'auteur**. `parse_frequency` / `build_frequency_string` ne gèrent
> pas de multiplicateur en tête (`'2MS'`) ; `build_frequency_string(position=…)` n'accepte que
> `'S'` / `'E'` / `None`.

### Prompt U1 — `utils/parse` et `utils/abc`

**Modèle : Sonnet · Plan mode : Non · Effort : medium · Dépendances : A0**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts U » de la partie U.
Aucun changement dans tsforecast/.

Cibles :
- tsforecast/utils/parse/utils.py (parse_frequency, build_frequency_string ; 88 %) ;
- tsforecast/utils/abc/{converter,normalizer}.py (TemporalConverter, TemporalNormalizer ;
  82 % / 63 %, aucun test dédié).

1. tests/unit/utils/frequency/test_parser.py (29 tests) est dans collect_ignore : il importe
   detect_and_parse_frequency depuis tsforecast.frequency, supprimé lors de la consolidation.
   Retrouver par `git log -S detect_and_parse_frequency` ce qu'il faisait, trier chaque test
   (§4.4), puis porter les tests encore pertinents vers tests/unit/utils/parse/test_utils.py
   (miroir de parse/utils.py) sur parse_frequency / build_frequency_string ; supprimer
   test_parser.py et son entrée de collect_ignore.
2. Compléter : aller-retour parse → build pour toutes les fréquences supportées (D, W, M, Q, Y,
   sous-journalières h, min, s, ms, us, ns ; positions S / E / absente ; ancres type 'Q-DEC',
   'W-MON'), rejet explicite de 'A' / 'AS' / 'AE' (comportement délibéré : test normal, pas
   xfail), multiplicateur '2MS' (comportement actuel à épingler, documenté comme non géré),
   position littérale 'start' refusée par build_frequency_string, casse.
3. tests/unit/utils/abc/test_converter.py et test_normalizer.py : contrat des classes
   abstraites via une sous-classe concrète minimale définie dans le test (méthodes abstraites
   non implémentées → TypeError ; méthodes concrètes héritées).

Critères §4.6 ; rapport §4.7.
```

### Prompt U2 — `utils/duration`

**Modèle : Sonnet · Plan mode : Non · Effort : medium · Dépendances : A0**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts U ». Aucun changement
dans tsforecast/. Notebooks de référence (possiblement périmés) :
notebooks/utils/duration_converter.ipynb, duration_normalizer.ipynb.

Cibles : tsforecast/utils/duration/{converter,normalizer,utils}.py (85 % / 45 % / 83 %),
DurationNormalizer, DurationConverter, normalize_duration, convert_duration et les autres
fonctions exportées par tsforecast/utils/duration/__init__.py. Aucun test dédié aujourd'hui
(une classe TestDurationConverterIntegration existe dans
tests/unit/utils/frequency/converter/test_validation.py : la laisser où elle est si elle teste
FrequencyConverter, la déplacer sinon).

Créer tests/unit/utils/duration/test_{converter,normalizer,utils}.py :
- normalisation : toutes les unités et alias acceptés (us, s, D, microsecond, second, day…,
  pluriels, casse), types en entrée (str, pd.Timedelta, datetime.timedelta, int + unité),
  rejets (message exact) ;
- conversion : table de conversions d'or par parametrize, aller-retour, valeurs négatives et
  nulles, conversions « calendaires » (mois, années) si l'API les expose — quelle convention
  (30 jours ? calendrier réel ?) : si elle n'est documentée nulle part → §4.4 (d) ;
- équivalence fonctions ↔ méthodes de classe.

Critères §4.6 ; rapport §4.7.
```

### Prompt U3 — `utils/position`

**Modèle : Opus · Plan mode : Non · Effort : high · Dépendances : A0**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts U ». Aucun changement
dans tsforecast/. Notebooks (possiblement périmés) : notebooks/utils/
period_position_converter.ipynb, period_position_normalizer.ipynb.

Cibles : tsforecast/utils/position/{converter,normalizer,utils}.py — 12 % / 62 % / 59 %, aucun
test dédié. PeriodPositionNormalizer, PeriodPositionConverter (dont convert_offset, qui gère
lui-même le multiplicateur), normalize_position, flip_position et le reste de __init__.py.

Créer tests/unit/utils/position/test_{converter,normalizer,utils}.py :
- normalisation : 'S' / 'E' / 'start' / 'end' / casse / None, rejets ;
- flip_position : involution ;
- convert_offset : chaque fréquence (M, Q, Y, W avec ancres, D) × position source × position
  cible, multiplicateurs ('2MS' → '2ME'), fréquences sans notion de position (D, h) ;
- conversion d'index / de jeux de données (MS ↔ ME, QS ↔ QE, YS ↔ YE) : valeurs d'or de dates
  calculées à la main (fin de février bissextile 2024, trimestres), séries et panels
  (heterogeneous_coverage_panel : positions début), index irrégulier (irregular_index_timeseries), index désordonné,
  MultiIndex 3 niveaux ; les imports différés de tsforecast.utils.frequency dans converter.py
  doivent être exercés.
Une conversion qui déplace une observation hors de sa période d'origine est une anomalie
[CODE] majeure.

Critères §4.6 ; rapport §4.7.
```

### Prompt U4 — `utils/time`

**Modèle : Sonnet · Plan mode : Non · Effort : medium · Dépendances : A0**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts U ». Aucun changement
dans tsforecast/.

Cible : tsforecast/utils/time/utils.py (62 %) — resolve_date, timeseries_to_string,
string_to_timeseries, get_period_start, get_period_end, get_period_boundaries. Test existant :
tests/unit/utils/time/test_utils.py (12 tests, jamais collectés avant A0 à cause de la
collision avec le module standard time : les exécuter et les trier §4.4 en premier).

Compléter :
- resolve_date : formats explicites / implicites, datetime / Timestamp / str, format invalide ;
- aller-retour timeseries_to_string ↔ string_to_timeseries (NaT, fuseau horaire si accepté) ;
- get_period_start / get_period_end / get_period_boundaries : table d'or par parametrize pour
  D, W (ancres), M, Q, Y, h, en positions début et fin, dates déjà en bord de période,
  29 février, fréquence invalide ; cohérence start ≤ date ≤ end pour toute date (propriété sur
  un échantillon de dates) ; entrées de type Period.

Critères §4.6 ; rapport §4.7.
```

### Prompt U5 — `utils/validation`

**Modèle : Sonnet · Plan mode : Non · Effort : high · Dépendances : A1**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts U ». Aucun changement
dans tsforecast/. CLAUDE.md classe utils/validation parmi les modules critiques (objectif de
couverture > 90 %).

Cible : tsforecast/utils/validation/utils.py (56 %, 8 tests dans
tests/unit/utils/validation/test_validation.py → renommer test_utils.py, miroir de utils.py).
Symboles : validate_temporal_data, restore_original_structure, validate_entities_grouped,
validate_sorted_within_groups.

Compléter, en exerçant les deux chemins (index / colonne time_col) et le paramètre strict :
- validate_temporal_data : Series, DataFrame, DatetimeIndex, MultiIndex 2 et 3 niveaux,
  time_col, panel_cols, index non datetime mais convertible, non convertible, doublons,
  non trié, vide, une ligne ; métadonnées renvoyées (_build_metadata via l'API) ;
- aller-retour validate_temporal_data → restore_original_structure == identité, pour chaque
  perturbation de tests/support/perturbations.py ;
- validate_entities_grouped / validate_sorted_within_groups : entités entrelacées, triées par
  entité mais pas par date, strict True / False (erreur vs correction), heterogeneous_coverage_panel mélangé.
Attention : le message « Cannot determine time index » de base/transformers.py qui fait échouer
des tests delays vient peut-être d'ici — si validate_temporal_data refuse un cas raisonnable,
c'est une anomalie à consigner (et à signaler pour D3-D5).

Critères §4.6 (minimum 90 %) ; rapport §4.7.
```

### Prompt U6 — `utils/frequency` : `normalizer` et `utils`

**Modèle : Sonnet · Plan mode : Non · Effort : high · Dépendances : U1**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts U » (abandon délibéré des
alias 'A' / 'AS' / 'AE'). Aucun changement dans tsforecast/. Notebook (possiblement périmé) :
notebooks/utils/frequency_normalizer.ipynb.

Cibles : tsforecast/utils/frequency/normalizer.py (95 %) et utils.py (84 %) :
FrequencyNormalizer, normalize_frequency (dont return_format), to_pandas_freq,
is_higher_frequency, get_frequency_order et le reste de utils.py (hors détection, testée en U7).

1. Tri des 19 échecs de tests/unit/utils/frequency/test_normalizer.py (liste dans
   tests/legacy_failures.txt, groupe « # prompt U6 ») :
   - TestDeprecatedAliasNormalization (8 tests) et test_converter_handle_a_alias : catégorie (a)
     — vérifier par l'historique git (consolidation parse_frequency) puis réécrire en tests de
     REJET explicite de 'A' (message d'erreur utile ?) ;
   - TestNormalizeFrequencyReturnFormats, TestFrequencyNormalizerComplexStrings,
     TestFrequencyNormalizerIntegration : établir la cause (git log -p -- normalizer.py utils.py)
     avant de choisir (a), (b), (c) ou (d).
   Retirer le groupe U6 de legacy_failures.txt.
2. Créer tests/unit/utils/frequency/test_utils.py (miroir de utils.py) et compléter :
   is_higher_frequency / get_frequency_order sur toutes les paires (table d'ordre total,
   antisymétrie, transitivité en propriété), to_pandas_freq pour chaque fréquence et position,
   return_format dans toutes ses modalités, fréquences inconnues.

Critères §4.6 ; rapport §4.7.
```

### Prompt U7 — `utils/frequency/detector.py`

**Modèle : Opus · Plan mode : Non · Effort : high · Dépendances : A1, U6**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts U ». Aucun changement
dans tsforecast/. Notebook (possiblement périmé) : notebooks/utils/frequency_detector.ipynb.
Commits utiles : f988798 (return_format du détecteur), 7488735 (détecteur déplacé dans utils).

Cible : tsforecast/utils/frequency/detector.py (72 %) — FrequencyDetector, detect_frequency,
detect_dataset_frequency, detect_index_frequency, target_offset_for_index. Tests :
tests/unit/utils/frequency/test_detector.py (53 tests, 5 échecs : Series à MultiIndex 2 et
3 niveaux, non triée, observations insuffisantes, detect_frequency sans cohérence).

1. Tri des 5 échecs (§4.4). La détection par (entité, colonne) sur panel est un contrat
   CENTRAL du package (HighFrequencyImputer en dépend) : une Series à MultiIndex qui n'est plus
   détectée est a priori (b) sauf preuve d'un changement délibéré.
2. Compléter : chaque fréquence (D, W, M, Q, Y, positions S / E), séries à trous, à une seule
   observation, à deux observations, index irrégulier (irregular_index_timeseries : fréquence de chaque
   colonne, et de l'index global), panel à fréquence hétérogène par entité (heterogeneous_coverage_panel,
   depenses_publiques_pib : Y pour FR / IT, Q pour DE), colonne jamais observée pour une
   entité (climat_affaires / IT), données désordonnées, return_format ; target_offset_for_index.

Critères §4.6 ; rapport §4.7.
```

### Prompt U8 — `utils/frequency/converter.py`

**Modèle : Opus · Plan mode : Oui · Effort : xhigh · Dépendances : U3, U6**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts U ». Aucun changement
dans tsforecast/. Notebook (possiblement périmé) : notebooks/utils/frequency_converter.ipynb.
Référence fonctionnelle complémentaire : high_frequency_imputer2_architecture.md §9.3
(décompte calendaire de full_periods_only, défaut B26 corrigé) et §10.2 (anchor_fraction).

Cible : tsforecast/utils/frequency/converter.py — FrequencyConverter, 1 845 lignes, 72 %
(106 lignes et 58 branches manquantes). Tests : paquet tests/unit/utils/frequency/converter/
(6 fichiers migrés en A0, 82 tests), dont 1 échec :
TestExtendIndexForUpsampling::test_unsupported_frequency_pair_returns_original.

En plan mode :
1. inventaire des méthodes publiques de FrequencyConverter et de leurs paramètres ;
2. matrice méthodes × scénarios (sens M→Q→Y et inverse, positions, booléens all / any, sum /
   mean / last, full_periods_only, anchor_fraction, panel, index irrégulier, NaN, données
   désordonnées) marquée couvert / à ajouter / sans objet ;
3. répartition des nouveaux tests dans les fichiers du paquet (créer au besoin
   test_aggregation.py, test_interpolation.py, test_panel.py). Faire valider.

Puis : trier l'échec existant (§4.4) ; écrire les tests manquants avec valeurs d'or à la main ;
propriété d'additivité (agrégation par somme d'une désagrégation = identité) ; heterogeneous_coverage_panel et
irregular_index_timeseries au moins une fois par méthode publique.

Critères §4.6 (objectif 90 % minimum sur ce fichier) ; rapport §4.7.
```

---

## Partie D — `tsforecast/delays`

> Contexte commun aux prompts D : pas de spécification dédiée ; sources par ordre de priorité
> §2.3. Renommages **délibérés** récents à répercuter dans les tests (catégorie (a)) :
> `release_delay` → `delay` (commit `4b3d3bc`), `output_unit` → `target_unit` (`f302664`),
> arguments et colonnes renvoyées (`ae347b2`) ; `_calculate_release_delays` →
> `_calculate_publication_delays`. Vérifier chacun par `git show <commit> --stat` et `git log
> -p`. Page de concept : `docs/concepts/publication_delays.md`. `CLAUDE.md` classe `delays`
> parmi les modules critiques (couverture > 90 %).

### Prompt D1 — `delays/data_manager.py`

**Modèle : Sonnet · Plan mode : Non · Effort : high · Dépendances : A1, U4**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts D ». Aucun changement
dans tsforecast/.

Cible : tsforecast/delays/data_manager.py (66 %) — compare_and_detect_delays (API publique :
new_data, existing_data, download_date, detection_mode, reference_point, delay_unit, time_col,
panel_cols). Test : tests/unit/delays/test_data_manager.py (48 tests), dans collect_ignore
(import de _calculate_release_delays, renommée _calculate_publication_delays).

1. Réparer l'import, retirer le fichier de collect_ignore, exécuter, trier chaque échec (§4.4).
   Les tests qui visent des fonctions privées : les réécrire via compare_and_detect_delays
   quand c'est possible, sinon `internal`.
2. Compléter : chaque detection_mode, reference_point 'start' / 'end', chaque delay_unit,
   premier téléchargement (existing_data=None), révisions de valeurs déjà publiées (sont-elles
   des « nouvelles observations » ?), panel (panel_cols) et série, time_col vs index,
   download_date str / datetime, données désordonnées, colonnes au nom spécial ; scénario
   réaliste : heterogeneous_coverage_panel « téléchargé » à deux dates (le second avec les valeurs retirées par
   les délais simulés du notebook) → délais détectés par (entité, colonne) cohérents avec
   ceux simulés.

Critères §4.6 ; rapport §4.7.
```

### Prompt D2 — `delays/calculator.py`

**Modèle : Sonnet · Plan mode : Non · Effort : high · Dépendances : D1**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts D ». Aucun changement
dans tsforecast/.

Cible : tsforecast/delays/calculator.py (38 %) — calculate_applicable_delay (conversion vers la
fréquence et le point de référence cibles, conversion d'unité, agrégation par indicateur /
panel). Test : tests/unit/delays/test_calculator.py (47 tests, 40 en échec, groupe
« # prompt D2 » de tests/legacy_failures.txt ; fixture monthly_publication_delays qui produit
encore une colonne release_delay).

1. Tri §4.4 : la majorité relève a priori de (a) (renommages délibérés) — le PROUVER commit par
   commit, puis mettre à jour fixtures et appels ; les tests qui échouent pour une autre raison
   sont triés individuellement. Idéalement, la fixture est construite via
   compare_and_detect_delays (sortie réelle de D1) plutôt qu'à la main, pour que le contrat
   entre les deux fonctions soit testé.
2. Compléter : conversions de fréquence (Q → M, M → Q, Y → M…) × point de référence (start /
   end), valeurs d'or à la main ; chaque méthode d'agrégation (mean, max, callable) ;
   target_unit ; target_frequency en dict par entité ; délais négatifs / nuls / très grands ;
   colonnes manquantes (message) ; ordre et noms des colonnes renvoyées (contrat de sortie).
Retirer le groupe D2 de legacy_failures.txt.

Critères §4.6 (minimum 90 %) ; rapport §4.7.
```

### Prompt D3 — `delays/transformers.py` : `ShiftTransformer` et `MaskTransformer`

**Modèle : Opus · Plan mode : Non · Effort : high · Dépendances : A1**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts D ». Aucun changement
dans tsforecast/.

Cible : ShiftTransformer et MaskTransformer de tsforecast/delays/transformers.py (fichier de
1 935 lignes, 64 %). Test : tests/unit/delays/test_transformers.py (77 tests) → à scinder
selon la règle de miroir du §3 en paquet tests/unit/delays/transformers/ :
test_shift_transformer.py, test_mask_transformer.py (ce prompt),
test_publication_delay_transformer.py et test_factories.py (prompt D4, y déplacer tels quels
les tests concernés).

1. Tri des échecs du groupe « # prompt D3 » (test_non_datetime_index,
   TestMaskTransformerBoundaries × 3, test_mask_dataframe_inverse_transform) §4.4.
2. Compléter pour chacun : transform / inverse_transform (aller-retour == identité sur les
   cellules non masquées ; que devient une cellule masquée à l'inversion ?), décalages positifs,
   nuls, négatifs, supérieurs à la longueur de la série ; paramètres par variable et par
   entité ; how='first' / 'last' ; bords de série ; extension d'index (_extend_index_start via
   l'API) ; panel désordonné ; index non datetime (erreur attendue ? message ?) ; protocole
   sklearn (clone, get_params, NotFitted).
   Propriété clé : un décalage de k périodes puis son inverse ne perd aucune observation dans
   la fenêtre commune.

Critères §4.6 (couverture mesurée sur les deux classes : --cov=tsforecast.delays.transformers,
lignes des deux classes) ; rapport §4.7.
```

### Prompt D4 — `PublicationDelayTransformer` et fabriques

**Modèle : Opus · Plan mode : Oui · Effort : high · Dépendances : D2, D3**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts D ». Aucun changement
dans tsforecast/.

Cible : PublicationDelayTransformer (orchestrateur), create_delay_transformer_factory,
prepare_entity_kwargs_from_delays et leurs auxiliaires (_build_entity_params,
_extract_param_by_variable, _resolve_strategy) dans tsforecast/delays/transformers.py. Tests :
tests/unit/delays/transformers/test_publication_delay_transformer.py et test_factories.py
(créés en D3), échecs du groupe « # prompt D4 » (KeyError dans fit ;
test_params_with_dict_reference_point).

En plan mode : cartographier le flux (tableau de délais → paramètres par entité et variable →
ShiftTransformer / MaskTransformer → PanelwiseTransformer ?) et proposer la matrice de
scénarios (stratégies, reference_point scalaire / dict, panel / série, délais manquants pour
une variable, variable sans délai, entité absente du tableau). Faire valider.

Puis : tri des échecs (§4.4 ; le KeyError est-il un renommage de colonne non répercuté dans le
code — ce serait (b), un bogue introduit par le renommage — ou dans le test ?) ; tests de la
matrice ; aller-retour transform / inverse_transform ; sur heterogeneous_coverage_panel avec des délais tirés de
compare_and_detect_delays (D1) → après transform, la dernière valeur disponible de chaque
(entité, colonne) est cohérente avec son délai.

Critères §4.6 ; objectif : transformers.py entier ≥ 90 % après D3 + D4. Rapport §4.7.
```

### Prompt D5 — Intégration `delays`

**Modèle : Opus · Plan mode : Non · Effort : high · Dépendances : D4**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts D ». Aucun changement
dans tsforecast/.

Cible : tests/integration/delays/test_integration.py (19 tests, 14 en échec, groupe
« # prompt D5 » : ShiftTransformer / MaskTransformer avec PanelwiseTransformer, workflows de
détection, combinaison shift puis mask, performance). Certains échouent sur « Cannot determine
time index » levé par tsforecast/base/transformers.py : établir si c'est l'intégration avec
PanelwiseTransformer / validate_temporal_data qui est cassée (b) ou le montage du test (c) —
voir aussi ce qu'a conclu U5.

1. Tri §4.4 de chaque échec, retrait du groupe D5 de legacy_failures.txt.
2. Scinder par thème si utile (test_panelwise.py, test_workflow.py) ; marquer slow le test de
   performance.
3. Ajouter le workflow de bout en bout sur heterogeneous_coverage_panel : deux « téléchargements » →
   compare_and_detect_delays → calculate_applicable_delay → PublicationDelayTransformer →
   vérification par (entité, colonne) ; puis PublicationDelayTransformer dans une XYPipeline
   évaluée par cross_validate avec TSOutOfSampleSplit sur irregular_index_timeseries (pas de fuite :
   aucune donnée du pli de test visible au fit).

Critères §4.6 (couverture globale de tsforecast/delays ≥ 90 % en fin de partie D) ;
rapport §4.7.
```

---

## Partie F — `tsforecast/frequency`

> Contexte commun aux prompts F : la **spécification** `high_frequency_imputer2_architecture.md`
> fait autorité sur les docstrings (§2.3). Un écart code ↔ spec est une anomalie `[SPEC]` ; un
> écart docstring ↔ code, `[DOC]`.

### Prompt F1 — `provenance.py` et `imputation_plan.py`

**Modèle : Sonnet · Plan mode : Non · Effort : medium · Dépendances : A1**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts F ». Référence :
high_frequency_imputer2_architecture.md §6 (provenance, échelle de souillure) et §12.2
(ImputationStep / ImputationPlan). Aucun changement dans tsforecast/.

Cibles : tests/unit/frequency/test_provenance.py (64 %, 13 tests) et
tests/unit/frequency/test_imputation_plan.py (95 %, 14 tests).

provenance.py :
- ProvenanceType : les 10 membres, hiérarchie éventuelle, sérialisation str (`str, Enum`) ;
- resolve_model_provenance, origin_to_taint, max_origin : table de vérité COMPLÈTE par
  parametrize, itérable vide pour max_origin ;
- ImputationProvenanceTracker : initialize (série, panel, colonnes absentes, index non trié,
  dupliqué), extend_index, chaque mark_* (index partiel, hors matrice, colonne inconnue,
  écrasement d'une provenance existante — quelle priorité ?), clear_provenance, get_provenance,
  get_mask, compute_statistics (tous les types présents, pourcentages cohérents, jeu vide),
  get_provenance_matrix, to_string_matrix, merge (chevauchements, colonnes disjointes,
  conflits) ;
- heterogeneous_coverage_panel : marquage de la seule entité IT, aucune cellule FR / DE touchée.

imputation_plan.py : ImputationStep (frozen, __eq__ Series-safe sur scale_factor, stage_key,
emitted_provenance pour chaque combinaison de souillures et unanchored), ImputationPlan
(immuabilité, by_stage, models, to_diagnostic_frame : colonnes exactes du §12.2 et de
l'enrichissement 2026-09-10, une ligne par étape, plan vide), append_step (ne mute pas le plan
d'origine), to_entity_tuple, INTERPOLATE_FALLBACK, MaterializationWay.

Critères §4.6 ; rapport §4.7.
```

### Prompt F2 — `regularizer.py` et `target_frequency_validator.py`

**Modèle : Sonnet · Plan mode : Non · Effort : medium · Dépendances : A1**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts F ». Aucun changement
dans tsforecast/.

Partie A — tsforecast/frequency/regularizer.py : 11 %, AUCUN test, non utilisé par
HighFrequencyImputer mais exporté (IndexRegularizer, is_regular, regularize). Lire le module et
notebooks/utils/regularizer.ipynb + notebooks/test_regularizer.ipynb (possiblement périmés),
puis créer tests/unit/frequency/test_regularizer.py :
- is_regular : série régulière / à trou / à doublon ; panel régulier, panel dont UNE entité est
  irrégulière ; D, M (début et fin), Q, Y ; irregular_index_timeseries et heterogeneous_coverage_panel (irréguliers) ;
- regularize : réindexation sur la grille régulière, valeurs préservées aux dates d'origine,
  NaN aux dates ajoutées, idempotence, non trié, panel à couvertures hétérogènes (chaque entité
  garde-t-elle SES bornes ?), positions incohérentes entre entités (quelle erreur ?) ;
- équivalence fonctions ↔ méthodes d'IndexRegularizer ; messages d'erreur.
Une observation détruite ou une fréquence d'entité modifiée par regularize → anomalie.

Partie B — tests/unit/frequency/test_target_frequency_validator.py (93 %, 23 tests) :
branches manquantes (term-missing d'abord), heterogeneous_coverage_panel (y à couverture hétérogène), cible
absente pour une entité, target_frequency en dict avec entité inconnue / manquante, fréquence
cible plus basse que celle de y.

Critères §4.6 ; rapport §4.7.
```

### Prompt F3 — `frequency_aligner.py`

**Modèle : Opus · Plan mode : Non · Effort : high · Dépendances : A1**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts F ». Aucun changement
dans tsforecast/.

Cible : tests/unit/frequency/test_frequency_aligner.py (27 tests, 90 %, 24 appels privés).
FrequencyAligner n'est pas utilisé par HighFrequencyImputer (outil autonome d'alignement,
utilisé par le notebook 3).

1. Audit des appels privés (_aggregate_series, _interpolate_series, _aggregate_to_target,
   _interpolate_to_target, _observed_series…) : réécrire via convert_to_target /
   build_densified_index, sinon `internal`. Résultat dans le rapport.
2. Compléter, valeurs d'or à la main : agrégation M→Q, M→Y, Q→Y (méthodes exposées), périodes
   incomplètes en bord, NaN au milieu d'une période ; interpolation Y→M, Q→M, positions début ET
   fin des deux côtés (le commit 967e2ad « position coherence with source index » est récent :
   zone à risque) ; panel à fréquence par (entité, colonne) — heterogeneous_coverage_panel, depenses_publiques_pib ;
   shuffle_rows → résultat identique ; noms de colonnes spéciaux ; index 3 niveaux ;
   build_densified_index : bornes exactes par position.

Critères §4.6 ; rapport §4.7.
```

### Prompt F4 — `imputation_window.py`

**Modèle : Opus · Plan mode : Non · Effort : high · Dépendances : A1**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts F ». Référence :
high_frequency_imputer2_architecture.md §7 (trois masques, type de retour §7.2) et D39
(colonnes structurellement absentes exclues du dénominateur de couverture, attribut
structurally_absent_columns_, UserWarning). Aucun changement dans tsforecast/.

Cible : tests/unit/frequency/test_imputation_window.py (58 tests, 90 %).
ImputationWindowCalculator, ImputationScope.

1. Couverture term-missing, traiter chaque bloc non couvert.
2. Un test par contrat : inclusion strict ⊆ imputation, relation avec training selon
   training_scope ; type de retour panel = pd.Series unique à MultiIndex (entity…, date), jamais
   un dict ; bornes = dict par entité ; entities_without_window_ ; coverage_threshold 0 et 1 ;
   extensions avant / arrière ; D39 sur heterogeneous_coverage_panel (climat_affaires / IT hors dénominateur,
   avertissement UNIQUE, structurally_absent_columns_ exact) ; get_mask_at_frequency pour chaque
   kind, M→Q, M→Y, positions début / fin, entité sans fenêtre, clés scalaires vs tuples ;
   get_columns_with_coverage ; NotFitted sur chaque méthode publique.
3. heterogeneous_coverage_panel : fenêtres par entité cohérentes avec les couvertures propres (DE finit en
   2024-04, IT commence en 2019-01), bornes calculées à la main ; irregular_index_timeseries : les dates
   annuelles de 2015 ne doivent pas étendre la fenêtre stricte avant 2018 (sinon : anomalie ?).
4. Robustesse : shuffle_rows, reverse_entities, to_three_level_index → mêmes masques.

Critères §4.6 ; rapport §4.7.
```

### Prompt F5 — `aggregation_constraint.py`

**Modèle : Opus · Plan mode : Non · Effort : high · Dépendances : A1**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts F ». Référence :
high_frequency_imputer2_architecture.md §11 et D20 ('sum' / None seulement, forme dict par
colonne ; 'mean' / 'last' retirés). Aucun changement dans tsforecast/.

Cible : tests/unit/frequency/test_aggregation_constraint.py (29 tests, 92 %).
AggregationConstraint (fit, transform, rescale, anchor_cells_mask, validate_columns,
resolve_constraint), validate_aggregation_constraint, resolve_aggregation_constraint,
validate_constraint_columns, ConstraintKind, DEFAULT_CONSTRAINT_KEY.

1. Validation : table exhaustive (valeurs admises, 'mean' / 'last' rejetés avec message, dict
   avec DEFAULT_CONSTRAINT_KEY, colonne inconnue, types invalides).
2. rescale — invariant « somme des sous-périodes recalées = total observé » par PROPRIÉTÉ sur
   Y→M, Y→Q, Q→M ; positions début / fin ; période incomplète en bord ; sous-période NaN ; total
   nul ; total NÉGATIF (balance commerciale d'`irregular_index_timeseries` !) ; toutes sous-périodes nulles (division
   par zéro ?) ; panel à fréquence source différente par entité.
3. anchor_cells_mask : exactement les cellules des ancres.
4. sklearn : clone, get_params / set_params, fit_transform == fit().transform(), NotFitted.
5. Avertissements agrégés et uniques.
Une somme recalée qui s'écarte du total de plus de 1e-9 est une anomalie, jamais une tolérance
à élargir.

Critères §4.6 ; rapport §4.7.
```

### Prompt F6 — `stage_scaler.py`

**Modèle : Opus · Plan mode : Non · Effort : high · Dépendances : A1**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts F ». Référence :
high_frequency_imputer2_architecture.md §9 ('constant' / 'calendar'), §5.4 (échelle par ligne,
B12), §5.8 R5 (source_freq par entité, diviseur fractionnaire 1/3). Règle de forme : la forme
du retour dépend de la CONFIGURATION, jamais des valeurs. Aucun changement dans tsforecast/.

Cible : tests/unit/frequency/test_stage_scaler.py (47 tests, 91 %, 20 appels privés).

1. Audit des appels privés (_pair_divisor, _feature_divisor, _stage_divisor, _spread…) →
   feature_divisors / target_divisor / fit_scale_factor / transform, sinon `internal`.
2. Valeurs d'or par parametrize pour (source, prédiction) ∈ {Y, Q, M}² en 'constant' (12, 4, 3,
   1, 1/3, 1/12…) et 'calendar' (jours réels : février 2024, trimestres de 90 / 91 / 92 jours),
   positions début et fin.
3. Règle de forme : Series vs DataFrame selon la configuration, y compris diviseurs tous à 1.0.
4. Symétrie inverse_transform ∘ transform == identité (1e-12) sur features et cible ; transform
   distingue features / cible par le type.
5. source_freq par entité : PANEL-F et heterogeneous_coverage_panel (depenses_publiques_pib Y pour FR / IT, Q pour
   DE).
6. sklearn : clone, NotFitted, get_params ; méthodes de diviseur utilisables sans fit.

Critères §4.6 ; rapport §4.7.
```

### Prompt F7 — `covariate_materializer.py`

**Modèle : Opus · Plan mode : Oui · Effort : xhigh · Dépendances : F1, F5**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts F ». Référence :
high_frequency_imputer2_architecture.md §4 (stratégies 'model' / 'tolerate_nan' /
'interpolate', §4.5 covariate_eligibility, §4.6 précédence à 4 rangs, MaterializationWay),
§5.4bis (couches), D37. Invariant central : jamais de dégradation fit → predict. Aucun
changement dans tsforecast/.

Cible : tests/unit/frequency/test_covariate_materializer.py (29 tests, 92 %). ≈1 900 lignes.

En plan mode, proposer la matrice : lignes = méthodes publiques (classify, decide_ways,
materialize, stage_frame, record_production, interpolate_column, eligible_columns,
entities_without_column, snapshot, reset, validate_columns, resolve_method / resolve_anchor /
resolve_aggregation_constraint) ; colonnes = scénarios (série / panel / panel à fréquence
hétérogène / entité sans la colonne / colonne tout-NaN / index désordonné / positions début-fin) ;
case = couvert / à ajouter / sans objet. Faire valider.

Puis : un test par rang de précédence, avec deux rangs candidats dont seul le plus prioritaire
s'applique ; unicité de la voie par (colonne, entité, étape) ; invariant fit → predict sur
heterogeneous_coverage_panel par entité ; covariate_eligibility (climat_affaires / IT) ; snapshot / reset isolent
deux usages successifs ; AggregationConstraintApplier (Protocol) : une doublure minimale suffit.

Critères §4.6 ; rapport §4.7.
```

### Prompt F8 — `training_set_builder.py`

**Modèle : Opus · Plan mode : Non · Effort : high · Dépendances : F4, F6, F7**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts F ». Référence :
high_frequency_imputer2_architecture.md §5.8 (R1-R6, D17-D19 : f_block(e) = fréquence propre
de l'entité, cible jamais agrégée, blocs indépendants de l'étape), §5.4bis (couches), §5.9
(cellules coïncidentes sous aggregation_constraint=None, niveau 'frequency' estampillé seulement
si une coïncidence survit au filtre d'origine). Aucun changement dans tsforecast/.

Cible : tests/unit/frequency/test_training_set_builder.py (27 tests, 97 %) — l'enjeu est la
robustesse et les propriétés.

- split_training_index : aller-retour avec / sans niveau de fréquence, noms de niveaux non
  standards (jamais de reniflage), 3 niveaux d'entité ;
- TrainingSet : champs, has_frequency_level ;
- build : lignes d'or sur PANEL-F (51 à toutes les étapes), 12 vs 15 lignes a1 / a2 à l'étape M
  sous 'sum' vs None ; entité plus fine que l'étape jamais sur la grille de prédiction ; filtre
  d'origine ; fenêtre 'training' lue par couche ;
- heterogeneous_coverage_panel, depenses_publiques_pib : blocs {FR: Y, DE: Q, IT: Y}, diviseurs cohérents avec
  StageScaler ;
- shuffle_rows / reverse_entities → mêmes lignes (au tri près).

Critères §4.6 ; rapport §4.7.
```

### Prompt F9 — `variable_orderer.py`

**Modèle : Sonnet · Plan mode : Non · Effort : medium · Dépendances : A1**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts F ». Référence :
high_frequency_imputer2_architecture.md §8 ('frequency' / 'cv', cv sklearn polymorphe ; §8.5 :
training_sets prime sur X / scoring_mask ; variable sans covariable → repli 'frequency').
Aucun changement dans tsforecast/.

Cible : tests/unit/frequency/test_variable_orderer.py (18 tests, 90 %). VariableOrderer,
VariableSpec.

- 'frequency' : tri, départage alphabétique, fréquences par entité divergentes ;
- 'cv' : cv entier, KFold, TimeSeriesSplit, TSOutOfSampleSplit, cv invalide (messages) ;
  scores_ et fallback_keys_ exacts ;
- replis : variable sans covariable, jeu vide, estimateur en échec (FailingEstimator) — logs /
  warnings agrégés ;
- déterminisme ; ordre des colonnes d'entrée permuté → même sortie (hors départage documenté) ;
- sklearn : clone, get_params.

Critères §4.6 ; rapport §4.7.
```

### Prompt F10 — `HighFrequencyImputer` (1/3) : découpage, paramètres, conformité sklearn

**Modèle : Opus · Plan mode : Oui · Effort : high · Dépendances : F1 à F9**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md, l'encadré « Contexte commun aux prompts F », puis
high_frequency_imputer2_architecture.md (§12.3 et §13 : paramètres et attributs ; §14 : D1-D39 ;
§16 : invariants I1-I21). Aucun changement dans tsforecast/.

Situation : tests/unit/frequency/test_high_frequency_imputer.py ≈3 250 lignes, 163 tests,
38 classes, 69 appels privés ; high_frequency_imputer.py à 88 % (95 lignes, 58 branches
manquantes).

Objectif : restructurer sans perdre un test, puis compléter paramètres et conformité sklearn
(fit / transform : F11, F12). ATTENTION : ce paquet de tests sera le filet du refactoring
(partie R) — chaque test doit passer par l'API publique autant que possible.

1. Plan mode : découpage en paquet tests/unit/frequency/high_frequency_imputer/
   (test_parameters.py, test_fit_phases.py, test_frequency_progression.py,
   test_mutualization.py, test_materialization.py, test_provenance.py, test_invariants.py,
   test_transform.py, test_inverse_transform.py, test_sklearn.py, conftest.py) avec la
   correspondance classe existante → fichier ; pour chacun des 69 appels privés : « réécrire via
   API publique » ou « garder + internal ». Faire valider.
2. Découpage (git mv du fichier d'origine vers celui qui reçoit la plus grosse part). Même liste
   de noms de tests, même résultat.
3. Traçabilité I1-I21 → test(s) : vérifier le tableau du §16 contre les tests réels, le placer
   dans la docstring de test_invariants.py ; invariant sans test → le signaler.
4. test_parameters.py : chaque paramètre de __init__ — valeurs admises, rejetées (message),
   inertes (silence documenté), __init__ qui valide SANS transformer (B3 : get_params renvoie
   exactement ce qui a été passé), dict par feature avec colonne inconnue. Écart docstring de
   paramètre ↔ spec §13 → [DOC] / [SPEC].
5. test_sklearn.py : clone, get_params / set_params, sklearn.utils.estimator_checks applicables
   (check_get_params_invariance, check_set_params, check_no_attributes_set_in_init,
   check_dont_overwrite_parameters ; documenter les inapplicables et pourquoi),
   NotFittedError, pickle aller-retour d'un imputeur ajusté (transform identique).

Critères §4.6 (couverture sur le paquet entier) ; rapport §4.7 avec la table ancien → nouveau.
```

### Prompt F11 — `HighFrequencyImputer` (2/3) : `fit`, cas limites, robustesse d'index

**Modèle : Opus · Plan mode : Non · Effort : xhigh · Dépendances : F10**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md, l'encadré « Contexte commun aux prompts F », puis
high_frequency_imputer2_architecture.md §5, §6 et §12.3. Paquet
tests/unit/frequency/high_frequency_imputer/ créé en F10. Aucun changement dans tsforecast/.

1. Couverture : `uv run pytest tests/unit/frequency/high_frequency_imputer
   --cov=tsforecast.frequency.high_frequency_imputer --cov-branch --cov-report=term-missing`.
   Pour chaque bloc non couvert du fit (phases 0 à 6), écrire le scénario PUBLIC qui l'atteint ;
   s'il n'en existe pas, le signaler (code mort ?).
2. Cas limites, un test chacun : X non trié ; index dupliqué ; entité de X absente de y et
   inversement ; colonne tout-NaN ; entité tout-NaN ; une seule observation de la cible ; y
   entièrement NaN ; X sans colonne de basse fréquence (plan vide ?) ; une seule entité en
   MultiIndex.
3. Invariance aux perturbations (shuffle_rows, reverse_entities, with_special_column_names,
   with_index_names, to_three_level_index) : fit_transform(perturbé) == perturbation(
   fit_transform(original)) au réordonnancement près, imputation_plan_.to_diagnostic_frame()
   identique au renommage près. Sur PANEL-X ; slow si nécessaire.
4. Positions : PANEL-X passé en to_period_start → mêmes imputations au décalage de date près.
   Constat connu non corrigé (campagne HFI2) : sur un index MS, la grille cible porte une ligne de
   plus par entité — l'épingler ici, anomalie si confirmé.
5. fit sur irregular_index_timeseries et heterogeneous_coverage_panel pour chaque covariate_strategy : pas d'exception, aucune
   cellule observée marquée imputée, avertissements agrégés (un par famille au plus).

Critères §4.6 ; rapport §4.7.
```

### Prompt F12 — `HighFrequencyImputer` (3/3) : `transform`, `inverse_transform`, sortie

**Modèle : Opus · Plan mode : Non · Effort : xhigh · Dépendances : F11**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md, l'encadré « Contexte commun aux prompts F », puis
high_frequency_imputer2_architecture.md §12.1, §12.4, D11 et D34-D38 (§14.7). Aucun
changement dans tsforecast/.

1. Couverture des chemins transform / inverse_transform / sortie multi-fréquences / rejeu /
   recouvrement restant non couverts après F11.
2. Contrats :
   - fit_transform(X) == fit(X).transform(X) pour chaque (covariate_strategy ×
     impute_intermediate_frequencies) sur PANEL-X ;
   - transform ne ré-entraîne jamais (TrainingSetBuilder patché en AssertionError) et ne mute pas
     l'état ajusté (deepcopy des attributs publics _ avant / après) ;
   - transform sur des dates postérieures au fit : nouvelles périodes imputées, valeurs du fit
     inchangées ;
   - fréquences divergentes au transform : warning + fréquences du fit ; colonne du fit
     manquante : ValueError ;
   - entité nouvelle (D34), impute_unobserved_entities True / False ;
   - inverse_transform(transform(X)) restitue les valeurs OBSERVÉES de X et son index
     (niveaux multiples, noms non standards) ;
   - keep_lower_frequencies et sortie multi-fréquences selon D35.
3. heterogeneous_coverage_panel : fit jusqu'à 2023-12, transform sur l'ensemble ; DE (fin 2024-04) n'est pas
   prolongée au-delà de sa couverture sans provenance adéquate.

Critères §4.6 ; objectif final high_frequency_imputer.py ≥ 95 %. Rapport §4.7.
```

### Prompt F13 — Intégration `frequency`

**Modèle : Opus · Plan mode : Oui · Effort : high · Dépendances : F12, D5**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 2, 3 et 4 de
tests_and_refactoring_prompts.md (le §3 définit la frontière unit / integration) et l'encadré
« Contexte commun aux prompts F ». Aucun changement dans tsforecast/.

Objectif : tests/integration/frequency/, comportement de bout en bout sur les jeux du notebook 3
et insertion de HighFrequencyImputer dans l'écosystème du package. Plan mode : proposer la
matrice de scénarios et la faire valider. Pistes :
1. test_hfi_on_realistic_dataset.py : HFI sur irregular_index_timeseries et heterogeneous_coverage_panel, chaque covariate_strategy ×
   impute_intermediate_frequencies — aucune valeur observée modifiée ; totaux de période
   conservés sous 'sum' ; chaque cellule non NaN de la sortie a une provenance ; NaN restants
   seulement là où la spec les annonce (par entité : climat_affaires / IT) ; entités à
   couverture courte non prolongées.
2. test_pipelines.py : XYPipeline([PublicationDelayTransformer, HighFrequencyImputer,
   estimateur]) ; cross_validate avec TSOutOfSampleSplit (série) et PanelOutOfSampleSplit
   (panel) ; GridSearchCV sur un paramètre de l'imputeur ; absence de fuite (SpyEstimator : aucune
   donnée postérieure à la fin du pli d'entraînement).
3. test_tracking.py : tsforecast.tracking.imputation_metrics — dict plat de float, clés stables,
   cohérent avec imputation_plan_.to_diagnostic_frame().
4. test_notebooks.py (slow) : exécution du notebook 5 via nbclient, sans erreur.
Module < 60 s avec -m "not slow".

Critères §4.6 (rapporter l'apport à la couverture globale) ; rapport §4.7.
```

---

## Partie C — Clôture de la campagne de tests

### Prompt C1 — Couverture, CI, badges `README.md`

**Modèle : Sonnet · Plan mode : Non · Effort : medium · Dépendances : toutes les parties U, D, F**

```text
Contexte : dépôt ts-forecast, remote https://github.com/qbolliet/ts-forecast. Lis CLAUDE.md
puis les sections 2 à 4 de tests_and_refactoring_prompts.md. pytest-cov est dans le groupe dev ;
[tool.coverage.*] configuré en A0. Pas de .github/.

Choix retenu : pytest-cov mesure (localement et en CI), Codecov publie. Un badge statique généré
localement serait périmé au premier commit sans régénération ; Codecov lit le coverage.xml de
pytest-cov et sert un badge à jour à chaque push.

1. .github/workflows/tests.yml : push et pull_request ; ubuntu-latest ; astral-sh/setup-uv ;
   Python 3.13 ; `uv sync --group dev` ; `uv run pytest tests/ -m "not slow" --cov=tsforecast
   --cov-branch --cov-report=xml` ; tests slow dans un job séparé (ou sur main seulement) ;
   codecov/codecov-action@v5, `token: ${{ secrets.CODECOV_TOKEN }}`, `if: always()`.
2. codecov.yml : statuts project / patch informatifs au départ ; flags ou composants par paquet
   (utils, delays, frequency) — vérifier dans la documentation Codecov lequel des deux sert un
   badge par sous-ensemble ; ignore tests/ et notebooks/.
3. README.md : badges en tête (sous le titre) — statut du workflow, couverture globale
   (https://codecov.io/gh/qbolliet/ts-forecast/branch/main/graph/badge.svg), et un badge par
   paquet si disponible. Réécrire la section « Test » (commandes run_tests.py, arborescence
   unit / integration / support, lien vers tests/ANOMALIES.md).
4. run_tests.py --mode coverage : échec si un module de tsforecast/{utils,delays,frequency} est
   sous 90 % (lecture de coverage.json / coverage.xml), liste des modules fautifs.
5. CLAUDE.md : section « Couverture de Code » (outil, seuils, badges, CI) et « Stratégie de
   Tests » à jour.
6. NE PAS pousser. Terminer par les actions manuelles : activer le dépôt sur codecov.io, créer le
   secret CODECOV_TOKEN, pousser, vérifier le badge (servi pour la branche de l'URL, main).

Vérifications : `uv run python tests/run_tests.py --mode coverage` vert ; YAML valide.
Rapport §4.7 avec la couverture finale par module.
```

### Prompt C2 — Bilan de la campagne de tests

**Modèle : Opus · Plan mode : Non · Effort : high · Dépendances : C1**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, la section 4 de tests_and_refactoring_prompts.md,
tests/ANOMALIES.md et tests/legacy_failures.txt. Aucun changement dans tsforecast/.

1. Vérifications : suite complète verte ; tests/legacy_failures.txt ne contient plus que
   « hors campagne » (sinon : lister les restes et s'arrêter) — s'il est vide, le supprimer avec
   le hook correspondant de tests/conftest.py ; collect_ignore vide ; chaque xfail « ANO-… » a
   son entrée dans ANOMALIES.md et inversement (script de contrôle dans le scratchpad) ;
   couverture ≥ 90 % par module de utils, delays, frequency.
2. ANOMALIES.md consolidé : dédoublonner, regrouper par cause racine, classer par sévérité ;
   pour chaque anomalie de frequency, préciser si le refactoring (partie R, qui ne change AUCUN
   comportement) la laisse intacte — ce doit être le cas — et quel correctif ultérieur la
   traitera.
3. Inventaire des tests internal (pytest -m internal --collect-only -q), par module : ceux que
   la partie R rendra obsolètes ou déplacera.
4. En tête de ANOMALIES.md, synthèse « Préalables au refactoring » (≤ 1 page).

Rapport §4.7 : tableau couverture par module, nombre de tests unit / integration / slow /
internal / xfail, les cinq anomalies les plus graves.
```

---

## 7. Refactoring : proposition

### 7.1 Nom du module

`tsforecast/frequency` décrit le **matériau** (des fréquences), pas la **fonction** (imputer
des variables de basse fréquence sur une grille plus fine), et entre en collision avec
`tsforecast/utils/frequency`. Candidats :

| Nom | Pour | Contre |
|---|---|---|
| **`imputation`** ✅ | dit ce que fait le module ; `from tsforecast.imputation import HighFrequencyImputer` se lit naturellement ; place pour d'autres imputeurs ; aucune collision | générique — mais le nom de la classe porte déjà la précision « haute fréquence » |
| `mixed_frequency` | décrit le problème traité | rouvre l'ambiguïté avec `utils/frequency` ; laisse croire que l'agrégation (hors package, cf. `CLAUDE.md`) y vit |
| `disaggregation` | terme technique exact (désagrégation temporelle) | décrit mal la cascade modèle + provenance ; moins parlant pour l'utilisateur |
| `nowcasting` | usage métier final | trop large (délais, crossvals relèvent aussi du nowcasting) |

**Décision de l'auteur (2026-09-27) : `tsforecast.imputation`, sans module de transition.**
`tsforecast.frequency` disparaît purement et simplement (package en 0.1.0, un seul
utilisateur) ; notebooks et docs sont mis à jour dans le même lot (R1).

### 7.2 Hiérarchie visible

Oui, un sous-paquet — et même deux, avec la convention de scikit-learn (préfixe `_` = détail
d'implémentation, API publique réexportée par `__init__.py`) :

```
tsforecast/imputation/
├── __init__.py                  # HighFrequencyImputer + types lus dans les attributs ajustés :
│                                #   ImputationPlan, ImputationStep, ProvenanceType,
│                                #   MaterializationWay, CellOrigin, Taint
├── high_frequency_imputer.py    # LA classe : docstring, __init__, fit/transform orchestrés
├── _components/                 # composants autonomes, testés isolément (existants)
│   ├── provenance.py  imputation_plan.py  imputation_window.py
│   ├── covariate_materializer.py  aggregation_constraint.py  stage_scaler.py
│   ├── variable_orderer.py  training_set_builder.py  target_frequency_validator.py
└── _engine/                     # morceaux EXTRAITS de la classe (prompts R2-R8)
    ├── params.py  run_context.py  warnings.py  frequency_layout.py
    ├── preparation.py  row_scaling.py  stage_runner.py  replay.py  output.py
```

- `_components/` : ce qui existe déjà et a une API propre ; `_engine/` : ce qui sort de la
  classe. La distinction dit au lecteur « ceci est une brique réutilisable » vs « ceci est une
  étape de l'algorithme de `HighFrequencyImputer` ».
- **`FrequencyAligner` et `IndexRegularizer`** ne sont pas utilisés par `HighFrequencyImputer` :
  ce sont des utilitaires de manipulation de fréquence de jeux de données, exactement la vocation
  de `tsforecast/utils/frequency`. Proposition : `tsforecast/utils/frequency/aligner.py` et
  `regularizer.py`, réexportés par `tsforecast.utils.frequency`.
- `FrequencyDetector` & co. ne sont plus réexportés par le module d'imputation (ils le sont déjà
  par `tsforecast.utils.frequency`).
- Les tests suivent (règle de miroir §3) : `tests/unit/imputation/high_frequency_imputer/`,
  `tests/unit/imputation/_components/test_*.py`, `tests/unit/imputation/_engine/test_*.py`,
  `tests/unit/utils/frequency/test_aligner.py`, `test_regularizer.py`,
  `tests/integration/imputation/`.

### 7.3 Découpage de `HighFrequencyImputer` (4 623 lignes, ~90 méthodes)

**Diagnostic.** La classe cumule six responsabilités : validation des paramètres (~400 lignes),
planification des fréquences (détection par couple, classification, groupes imputables, couples
sans ancre, progression : ~500 lignes), préparation des données (`_fit` : 379 lignes,
`_transform` : 161, dont les phases 0-4 et 0'-4' font la même chose avec des variantes),
mise à l'échelle par ligne (~300), exécution d'étape (`_prepare_variable` 179, `_fit_variable`
141, `_execute_step` 140, `_order_columns` 130…) et rejeu / sortie (~700). L'état d'exécution
(matérialiseur, traceur, calculateur de fenêtre, accumulateurs d'avertissements) vit dans des
dizaines d'attributs `self._…`, ce qui oblige `_replay_state` à les **échanger** puis restaurer
pour réutiliser `_execute_step` au `transform`.

**Proposition** — la classe devient un orchestrateur de ~800-1 000 lignes (docstring comprise) :

| Extrait | Contenu | Forme |
|---|---|---|
| `_engine/params.py` | tous les `_validate_*` | fonctions pures `validate_*(value) -> None` appelées par `__init__` / `fit` |
| `_engine/warnings.py` | accumulation + émission agrégée (règle « un message par famille ») | classe `WarningAccumulator` |
| `_engine/run_context.py` | matérialiseur, traceur, calculateur de fenêtre, masques, scaler, accumulateurs | dataclass `_RunContext` **construite** par `fit` et **reconstruite** par `transform` — supprime l'échange d'état de `_replay_state` |
| `_engine/frequency_layout.py` | détection robuste, fréquences par colonne, classification, groupes imputables, couples sans ancre, progression, libellés d'étape | `FrequencyLayout` (dataclass figée) + fonction `plan_frequencies(...)` |
| `_engine/preparation.py` | phases communes 0-4 : alignement de y, frame de travail, fenêtres, transformateur additif, provenance | `prepare_work_frame(imputer, X, y, *, fitting: bool) -> PreparedData` — **une** implémentation pour `fit` et `transform` |
| `_engine/row_scaling.py` | `_feature_divisors_per_row`, `_target_divisors_per_row`, `_row_layers`, `_row_entities`, `_block_binding` | `RowScaler` enveloppant `StageScaler` |
| `_engine/stage_runner.py` | `_execute_stage`, `_order_columns`, `_prepare_variable`, `_fit_variable`, `_fit_estimator`, `_build_step`, `_execute_step`, `_predict_step`, `_group_scale_factor` | `StageRunner(context, layout, params)` : `fit_stage(...)`, `replay_step(...)` |
| `_engine/replay.py` | `_replay_plan`, entités nouvelles, étapes dérivées, contrôle des fréquences au transform, avertissement hors fenêtre | fonctions sur `StageRunner` + `_RunContext` |
| `_engine/output.py` | recouvrement, liaison de sortie, sortie multi-fréquences, sélection / suppression du niveau de fréquence, restitution des valeurs d'origine | fonctions pures sur frames |

Ce qui ne change **pas** : la signature de `__init__` (mêmes arguments, mêmes défauts, même
ordre), `get_params()`, les attributs ajustés publics (suffixe `_`) et leur contenu, les
messages et catégories d'avertissements et d'erreurs, les sorties de `fit` / `transform` /
`inverse_transform` **au bit près**, la classe mère `XYPanelTimeSeriesTransformer`. Ce qui
**peut** changer : les attributs privés `self._…` (un imputeur picklé avant le refactoring ne se
dépicklera pas forcément après — à accepter explicitement) et les tests marqués `internal`.

### 7.4 Garde-fous du refactoring

1. **Golden master (R0)** : sorties figées de `HighFrequencyImputer` avant toute modification,
   comparées au bit près après chaque prompt R.
2. **Signature figée** : `inspect.signature(HighFrequencyImputer.__init__)`, liste triée des
   attributs publics ajustés, `get_params()` sur défauts — snapshots testés.
3. **Tests de la campagne 1** : tous verts ; un `xfail(strict=True)` qui passe = comportement
   changé = arrêt.
4. **Un extrait par prompt**, chacun testé unitairement dans `tests/unit/imputation/_engine/`.
5. Les anomalies consignées **ne sont pas corrigées** pendant la partie R : leur correction est
   une campagne ultérieure, justement rendue plus sûre par le découpage.

---

## Partie R — Refactoring

> Contexte commun aux prompts R : lire le §7 de ce document. **Aucun changement de
> comportement** : golden master (`tests/support/golden/`) au bit près, tous les tests verts,
> `xfail` stricts toujours en échec. Modifications autorisées de `tsforecast/` : uniquement
> celles que le prompt décrit. Commentaires français nominaux, docstrings Google en anglais,
> type hints (`CLAUDE.md`). Chaque objet extrait reçoit ses tests unitaires dans
> `tests/unit/imputation/_engine/test_<module>.py`. Les tests `internal` de
> `HighFrequencyImputer` qui visaient une méthode déplacée sont **déplacés** vers le test de
> l'extrait, pas supprimés. Rapport de fin de prompt : §4.7 plus le nombre de lignes de
> `high_frequency_imputer.py` avant → après.

### Prompt R0 — Golden master de `HighFrequencyImputer`

**Modèle : Opus · Plan mode : Non · Effort : high · Dépendances : C2**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 4 et 7 de tests_and_refactoring_prompts.md
et l'encadré « Contexte commun aux prompts R ». Aucun changement dans tsforecast/ : ce prompt
FIGE le comportement actuel avant le refactoring.

1. tests/support/golden/ : script de génération generate.py (exécutable à la main, jamais par
   pytest) qui, pour une matrice de configurations — covariate_strategy × 
   impute_intermediate_frequencies × aggregation_constraint ('sum', None) × fit_predict_order
   ('frequency', 'cv') × impute_unobserved_entities, élaguée aux combinaisons distinctes
   (voir §15.2 de la spec : classes d'équivalence) — et pour les jeux PANEL-X, ses projections,
   irregular_index_timeseries, heterogeneous_coverage_panel, enregistre : sortie de fit_transform, de transform sur un X
   postérieur, de inverse_transform, la matrice de provenance, to_diagnostic_frame() du plan,
   les attributs publics ajustés (sérialisables), et la liste (catégorie, message) des
   avertissements. Format : pickle pandas (pd.to_pickle) + un manifest JSON (configuration,
   jeu, versions pandas / numpy / sklearn). Estimateur déterministe (LinearRegression, cv à
   graine fixe).
2. tests/unit/imputation/... n'existe pas encore : placer le test de comparaison dans
   tests/integration/frequency/test_golden_master.py (déplacé en R1), marqué slow si > 60 s au
   total, comparant au bit près (pd.testing.assert_frame_equal(check_exact=True) ; égalité
   stricte des avertissements).
3. Snapshots de signature : test_signature.py dans le paquet de tests HFI — signature de
   __init__, get_params() par défaut, liste triée des attributs publics ajustés après fit.
4. Vérifier le déterminisme : générer deux fois, comparer ; exécuter le test dans un second
   processus.

Critères : suite complète verte ; taille de tests/support/golden raisonnable (< 20 Mo, sinon
élaguer la matrice et le justifier). Rapport §4.7 : matrice retenue et durée du test.
```

### Prompt R1 — Renommage `frequency` → `imputation`, hiérarchie, relocalisations

**Modèle : Sonnet · Plan mode : Oui · Effort : high · Dépendances : R0**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 3, 4 et 7 de
tests_and_refactoring_prompts.md et l'encadré « Contexte commun aux prompts R ». Ce prompt DÉPLACE du
code sans en changer une ligne de logique.

En plan mode, présenter : l'arborescence cible (§7.2), la table ancien chemin → nouveau chemin
(code ET tests), la liste des fichiers qui importent tsforecast.frequency (tsforecast/tracking/
metrics.py, docstring de tsforecast/utils/frequency/converter.py, notebooks, docs/api/frequency/*,
docs/index.md, mkdocs.yml, CLAUDE.md, README.md). Faire valider. Décision déjà prise par
l'auteur, à ne pas rouvrir : nom `imputation`, AUCUN module de transition — tsforecast/frequency
n'existe plus après ce prompt.

Puis :
1. git mv tsforecast/frequency → tsforecast/imputation ; composants → _components/ ;
   frequency_aligner.py → tsforecast/utils/frequency/aligner.py ; regularizer.py →
   tsforecast/utils/frequency/regularizer.py ; créer _engine/__init__.py vide.
2. Imports relatifs corrigés ; tsforecast/imputation/__init__.py réexporte HighFrequencyImputer
   et les types de résultat (§7.2) — __all__ explicite ; tsforecast/utils/frequency/__init__.py
   réexporte FrequencyAligner, IndexRegularizer, is_regular, regularize.
3. Tests : tests/unit/frequency → tests/unit/imputation (miroir : _components/,
   high_frequency_imputer/), test_frequency_aligner.py → tests/unit/utils/frequency/
   test_aligner.py, test_regularizer.py → tests/unit/utils/frequency/, tests/integration/
   frequency → tests/integration/imputation ; imports des tests mis à jour.
4. Docs : docs/api/frequency → docs/api/imputation (+ utils), mkdocs.yml, pages concept et
   tutoriel, CLAUDE.md (arborescence, sections), README.md, notebooks (tous, y compris
   notebooks/utils/*.ipynb qui importent ces modules) ; spec : ajouter en tête de
   high_frequency_imputer2_architecture.md une note datée « chemins renommés » avec la table,
   sans réécrire le corps.
5. `grep -rn "tsforecast.frequency\|tsforecast/frequency"` ne renvoie plus que la note de la
   spec et les archives .md ; `import tsforecast.frequency` lève ModuleNotFoundError.

Critères : suite complète verte, golden master identique, `uv run mkdocs build --strict` sans
erreur, notebook 5 exécuté sans erreur. Rapport §4.7 avec la table des déplacements.
```

### Prompt R2 — Extraction : validation des paramètres

**Modèle : Sonnet · Plan mode : Non · Effort : medium · Dépendances : R1**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, la section 7 de tests_and_refactoring_prompts.md et
l'encadré « Contexte commun aux prompts R ».

Cible : tsforecast/imputation/high_frequency_imputer.py — méthodes _validate_literal,
_validate_intermediate_frequencies, _validate_unit_interval, _validate_cv,
_validate_cv_scoring, _validate_target_frequency_format, _validate_estimator,
_validate_additive_transformer (et autres _validate_* éventuelles).

1. Les déplacer dans tsforecast/imputation/_engine/params.py sous forme de fonctions
   (paramètres explicites, aucun accès à self) ; la classe les appelle aux mêmes endroits, dans
   le même ordre (l'ordre des validations détermine QUELLE erreur est levée en premier : le
   conserver). Messages d'erreur à l'identique.
2. Repérer les validations dupliquées avec les composants (_components/*._validate_*,
   CovariateMaterializer._validate_literal…) : les signaler dans le rapport, ne PAS les
   fusionner (hors périmètre, risque de changer un message).
3. tests/unit/imputation/_engine/test_params.py : chaque fonction, valeurs admises / rejetées ;
   les tests internal de HFI qui les visaient y sont déplacés.

Critères : suite verte, golden master identique, test_signature inchangé. Rapport §4.7.
```

### Prompt R3 — Extraction : état d'exécution et avertissements agrégés

**Modèle : Opus · Plan mode : Oui · Effort : xhigh · Dépendances : R2**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, la section 7 de tests_and_refactoring_prompts.md,
l'encadré « Contexte commun aux prompts R », et high_frequency_imputer2_architecture.md
§12.1 (symétrie fit / transform) et la décision sur _replay_state (lot L12, D34-D38).

Cible : l'état d'exécution de HighFrequencyImputer — tous les attributs self._… créés dans
_fit / _transform qui ne sont PAS des attributs ajustés publics (matérialiseur, traceur de
provenance, calculateur de fenêtre, masques internes, scaler, TrainingSetBuilder, accumulateurs
d'avertissements, couples non ancrés en cours, etc.) — et le gestionnaire _replay_state.

En plan mode :
1. inventaire exhaustif de ces attributs : où ils sont écrits, où ils sont lus, lesquels
   survivent au fit (nécessaires au transform) et lesquels sont transitoires ;
2. proposition de _RunContext (dataclass) et de WarningAccumulator (familles d'avertissements
   actuelles, message agrégé identique) ;
3. stratégie pour supprimer l'échange d'état de _replay_state : transform construit un
   _RunContext neuf à partir de l'état ajusté au lieu d'échanger puis restaurer ; démontrer que
   les lectures de l'état du fit au transform sont toutes couvertes. Faire valider.

Puis implémenter dans _engine/run_context.py et _engine/warnings.py ; les méthodes de la classe
reçoivent ou lisent le contexte ; aucun attribut ajusté PUBLIC ne change. Tests unitaires des
deux extraits. Vérifier explicitement : fit_transform(X) == fit(X).transform(X), deux
transform successifs identiques, transform ne mute pas l'état ajusté (tests F12).

Critères : suite verte, golden master identique au bit près (y compris l'ordre et le texte des
avertissements). Rapport §4.7 avec la table attribut → destination.
```

### Prompt R4 — Extraction : planification des fréquences

**Modèle : Opus · Plan mode : Non · Effort : high · Dépendances : R3**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, la section 7 de tests_and_refactoring_prompts.md,
l'encadré « Contexte commun aux prompts R » et high_frequency_imputer2_architecture.md §5.1-§5.2
(progression, fusion des progressions par groupe de cible, D31), §5.10 (entités non observées).

Cible : _detect_frequencies_robustly, _detected_frequencies_by_column,
_column_frequencies_by_entity, _classify_variables_at_frequency, _imputable_groups,
_entity_target_frequency, _unanchored_pairs_at, _group_frequency_progression,
_build_frequency_progression, _stage_frequency_label, _stage_frequency_of,
_check_target_frequency_covers_entities.

1. _engine/frequency_layout.py : dataclass figée FrequencyLayout (fréquences détectées par
   couple, couples non détectés, fréquence cible effective, catégories, progression) + fonctions
   qui la calculent à partir de X_work, de la fréquence cible et des paramètres utiles —
   jamais de l'imputeur entier.
2. La classe calcule le layout une fois en phase 0-3 et publie les MÊMES attributs ajustés
   (detected_frequencies_, variable_categories_, frequency_progression_, …) qu'avant.
3. tests/unit/imputation/_engine/test_frequency_layout.py : PANEL-X, PANEL-F, heterogeneous_coverage_panel
   (depenses_publiques_pib Y / Q / Y, climat_affaires / IT non détectée), progression fusionnée
   sur cibles hétérogènes ; tests internal déplacés.

Critères : suite verte, golden master identique. Rapport §4.7.
```

### Prompt R5 — Mutualisation de la préparation `fit` / `transform`

**Modèle : Opus · Plan mode : Oui · Effort : xhigh · Dépendances : R4**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, la section 7 de tests_and_refactoring_prompts.md,
l'encadré « Contexte commun aux prompts R » et high_frequency_imputer2_architecture.md §12.1.

Cible : phases 0 à 4 de _fit et phases 0' à 4' de _transform (alignement de y, nom de cible,
frame de travail, index d'entrée, fenêtres et masques, transformateur additif, initialisation
de la provenance), qui font la même chose avec des variantes.

En plan mode : tableau phase par phase fit vs transform — étapes identiques, étapes qui
diffèrent (et pourquoi : ajustement vs réutilisation de l'objet ajusté, instantané d'entrée pour
l'inversion, contrôle des fréquences D11, recalcul des fenêtres AVANT le transformateur additif
pour garantir fit_transform == fit().transform…). Proposer prepare_work_frame(..., fitting: bool)
→ PreparedData (dataclass) et montrer que chaque différence est paramétrée, pas perdue. Faire
valider.

Puis implémenter dans _engine/preparation.py ; _fit et _transform deviennent une séquence
d'appels lisible (une ligne de commentaire nominal par phase). Tests unitaires de
prepare_work_frame en mode fit et transform.

Critères : suite verte, golden master identique ; _fit + _transform < 250 lignes à eux deux.
Rapport §4.7.
```

### Prompt R6 — Extraction : mise à l'échelle par ligne

**Modèle : Opus · Plan mode : Non · Effort : high · Dépendances : R5**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, la section 7 de tests_and_refactoring_prompts.md,
l'encadré « Contexte commun aux prompts R » et high_frequency_imputer2_architecture.md §5.4,
§5.4bis (couches), §5.9 (index estampillé : StageScaler jamais exposé à l'index estampillé,
réécriture POSITIONNELLE).

Cible : _feature_divisors_per_row, _target_divisors_per_row, _row_layers, _row_entities,
_block_binding, _group_scale_factor (si elle en relève).

1. _engine/row_scaling.py : RowScaler, construit avec le StageScaler et le layout, exposant
   feature_divisors(rows, …) et target_divisors(rows, …). Conserver la réécriture positionnelle
   et l'appel couche par couche ; « une seule couche = appel unique » reste vrai (test existant).
2. tests/unit/imputation/_engine/test_row_scaling.py : une couche, plusieurs couches, index
   estampillé d'un niveau 'frequency' (dates dupliquées), PANEL-F (diviseurs Q 4 / 1 / ⅓,
   M 12 / 3 / 1) ; tests internal déplacés.

Critères : suite verte, golden master identique. Rapport §4.7.
```

### Prompt R7 — Extraction : exécution d'étape

**Modèle : Opus · Plan mode : Oui · Effort : xhigh · Dépendances : R6**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, la section 7 de tests_and_refactoring_prompts.md,
l'encadré « Contexte commun aux prompts R » et high_frequency_imputer2_architecture.md §5-§6,
§8.5 (_prepare_variable partagé entre ordonnancement et ajustement, pas de cache 5b → 5c),
D19 (un ajustement par (étape, variable) partagé entre groupes de fréquence source).

Cible : _execute_stage, _order_columns, _prepare_variable (+ _VariableFit), _fit_variable,
_fit_estimator, _estimator_for, _select_feature_columns, _drop_empty_training_rows, _build_step,
_execute_step, _predict_step, _stage_mask, _prediction_grid, _unrestricted_grid,
_training_mask_at, _restrict_to_entities, _step_pairs.

En plan mode : graphe d'appels actuel, dépendances de chaque méthode (contexte, layout, row
scaler, paramètres de l'imputeur), et découpage proposé : StageRunner (fit d'une étape, rejeu
d'une étape) + fonctions pures pour les grilles et masques. Montrer que _execute_step reste
UNIQUE (fit et rejeu passent par le même code, contrat du lot L12). Faire valider.

Puis implémenter dans _engine/stage_runner.py ; la phase 5 de fit devient une boucle de
quelques lignes. Tests unitaires du StageRunner sur PANEL-X avec SpyEstimator (X_train, y_train,
X_pred vus par l'estimateur identiques à avant : comparer aux enregistrements du golden master
si besoin) ; tests internal déplacés.

Critères : suite verte, golden master identique au bit près. Rapport §4.7.
```

### Prompt R8 — Extraction : rejeu du plan et assemblage de la sortie

**Modèle : Opus · Plan mode : Non · Effort : xhigh · Dépendances : R7**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, la section 7 de tests_and_refactoring_prompts.md,
l'encadré « Contexte commun aux prompts R » et high_frequency_imputer2_architecture.md §12.4,
D11, D34-D38.

Cibles :
- rejeu : _replay_plan, _new_entities, _extend_stage_binding, _new_entity_steps,
  _check_transform_frequencies, _warn_rows_outside_window → _engine/replay.py ;
- sortie : _output_binding, _overlay_imputations, _build_multifreq_output, _has_frequency_level,
  _select_inverse_frequency_level, _drop_frequency_level, _restore_original_values →
  _engine/output.py (fonctions pures sur frames autant que possible).

Puis :
1. _transform et _inverse_transform deviennent des séquences d'appels lisibles.
2. tests/unit/imputation/_engine/test_replay.py et test_output.py ; tests internal déplacés.
3. Bilan de la classe : lister les méthodes restantes de HighFrequencyImputer ; objectif
   ≤ 1 000 lignes docstring comprise, aucune méthode > 80 lignes ; au-delà, proposer (sans
   l'exécuter) un découpage complémentaire.

Critères : suite verte, golden master identique. Rapport §4.7 avec la table méthode → module
pour l'ensemble de la partie R.
```

### Prompt R9 — Finalisation : tests, documentation, spec

**Modèle : Sonnet · Plan mode : Non · Effort : medium · Dépendances : R8**

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md, les sections 3 et 7 de tests_and_refactoring_prompts.md
et l'encadré « Contexte commun aux prompts R ». Plus aucune modification de logique.

1. Tests : vérifier la règle de miroir (script dans le scratchpad : chaque module de
   tsforecast/ a son test_<module>.py ou son paquet, et inversement) ; relancer l'inventaire
   des tests internal : ceux qui visent encore des méthodes privées de HighFrequencyImputer
   pointent-ils toujours vers quelque chose d'existant ? ; le golden master et test_signature
   restent en place (utiles aux correctifs d'anomalies à venir).
2. Docs : pages API mkdocs de tsforecast.imputation (la classe en premier, les composants en
   section « Internals ») et de _engine ; page concept mixed_frequency_imputation.md : schéma de
   l'architecture fit / transform avec les extraits ; `uv run mkdocs build --strict`.
3. Spec high_frequency_imputer2_architecture.md : nouvelle section datée « Architecture du code
   après refactoring » (§12.2 bis) — table méthode d'origine → module, sans réécrire l'historique.
4. CLAUDE.md : arborescence tsforecast/imputation, rôle de _components/ et _engine/,
   arborescence de tests.
5. tests/ANOMALIES.md : confirmer que chaque anomalie de l'ex-frequency est toujours ouverte,
   mettre à jour ses chemins.

Critères : suite verte ; mkdocs strict ; notebook 5 exécuté sans erreur. Rapport §4.7.
```
