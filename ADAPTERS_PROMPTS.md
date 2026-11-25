# Prompts des sessions — branche `qb-adapters`

Ce fichier liste, dans l'ordre, les sessions Claude Code qui mènent à bien le chantier des
adapters. Chaque session s'appuie sur `ADAPTERS_ARCHITECTURE.md` (contrat commun, spécifications,
questions ouvertes, journal des décisions).

**Mode d'emploi**

1. Avant la première session, committer ces deux fichiers sur `qb-adapters` pour que les diffs des
   sessions suivantes ne contiennent que le travail de la session
   (`docs: add adapters architecture and session prompts`).
2. Pour chaque session : ouvrir une **nouvelle** session Claude Code, régler le modèle (`/model`) et
   l'effort, activer le plan mode si indiqué (Shift+Tab), coller le bloc « Prompt ».
3. En fin de session : relire le diff, ajuster le message de commit proposé, committer soi-même.
   Une session ne commence que sur un arbre de travail propre.

**Choix des modèles.** Opus pour les sessions de conception, d'audit et d'implémentation non
triviale (rattrapage de `main`, contrat commun, hierarchicalforecast, opera, modèles de fondation) ;
Sonnet pour les sessions qui appliquent un contrat déjà fixé (notebook, backends supplémentaires,
documentation). Haiku n'est recommandé pour aucune session : toutes demandent du jugement sur des
API tierces mouvantes ou sur des questions de fuite temporelle.

**Plan mode.** Activé quand la session doit d'abord produire un diagnostic ou un design que vous
devez valider (audit, conception, résolution de conflits) ; désactivé quand le contrat est fixé.

| # | Session | Modèle | Effort | Plan mode |
|---|---|---|---|---|
| 1 | Rattrapage de `main` et dépendances | Opus | high | Oui |
| 2 | Socle commun, packaging, infrastructure de tests | Opus | high | Oui |
| 3 | Audit et correction de `SktimeAdapter` | Opus | high | Oui |
| 4 | Audit et correction de `DartsAdapter` | Opus | high | Oui |
| 5 | Notebook : cadre de prévision et sections sklearn → darts | Sonnet | medium | Non |
| 6 | `HierarchicalForecastAdapter` : conception et cross-section | Opus | xhigh | Oui |
| 7 | `HierarchicalForecastAdapter` : temporel et cross-temporel | Opus | high | Oui |
| 8 | Notebook : section hierarchicalforecast | Sonnet | high | Non |
| 9 | `OperaAdapter` | Opus | high | Oui |
| 10 | Notebook : section opera | Sonnet | medium | Non |
| 11 | Modèles de fondation : adapter, protocole, backends Chronos et `callable` | Opus | xhigh | Oui |
| 12 | Modèles de fondation : backends Timestral et t0 | Sonnet | high | Non |
| 13 | Notebook : section modèles de fondation | Sonnet | medium | Non |
| 14 | Documentation : page concept et tutoriel des adapters | Sonnet | high | Oui |
| 15 | Documentation : pages API, navigation, CLAUDE.md, README | Sonnet | medium | Non |
| 16 | Revue finale de la branche | Opus | high | Non |

Les sessions 3–4, 6–7, 9 et 11 sont indépendantes une fois la session 2 faite ; les sessions de
notebook suivent celle de leur adapter. L'ordre ci-dessus minimise les conflits.

---

## 0. Règles communes (rappelées par chaque prompt)

- Lire `CLAUDE.md` puis `ADAPTERS_ARCHITECTURE.md` avant toute action.
- **Ne jamais committer** : ni `git commit`, ni `git merge --continue`, ni `git rebase`, ni
  `git push`, ni `git stash`. `git add` seulement si une opération l'exige (résolution de conflits).
- Aucun commentaire, docstring ou test ne cite `ADAPTERS_ARCHITECTURE.md`, `ADAPTERS_PROMPTS.md`,
  leurs sections ou leurs identifiants (règle d'autoportance de `CLAUDE.md`).
- Une décision qui tranche une question ouverte ou s'écarte du document d'architecture s'inscrit
  dans son § 10 (journal des décisions), et le § 9 est mis à jour.
- Tests avec `.venv/Scripts/python.exe -m pytest ... -q -p no:cacheprovider` ; conventions avec
  `.venv/Scripts/python.exe scripts/check_conventions.py --rule all` ; notebooks exécutés en mémoire
  avec `nbformat` + `nbclient` et `PYTHONPATH` à la racine du repo (cf. mémoire du projet).
- Rester dans le périmètre de la session ; signaler (sans corriger) ce qui en sort.
- Fin de session : lancer les tests pertinents, afficher `git status` et `git diff --stat`, résumer
  ce qui a été fait, ce qui reste et les points à relire, puis proposer un message de commit au
  format conventionnel (`type(scope): résumé`, corps en puces).

---

## Session 1 — Rattrapage de `main` et dépendances

**Modèle** : Opus · **Effort** : high · **Plan mode** : oui

**Prompt**

```text
Lis CLAUDE.md, ADAPTERS_ARCHITECTURE.md (§ 2, § 6, § 9) et les règles communes du § 0 de
ADAPTERS_PROMPTS.md, que tu appliques strictement (en particulier : tu ne commites jamais).

Objectif : mettre la branche qb-adapters à jour avec main (465 commits de retard) et remettre ses
dépendances en phase, sans perdre le travail de la branche.

1. Plan (plan mode) : inventorie les fichiers modifiés des deux côtés depuis la base commune
   (git merge-base HEAD main), prévois les conflits (pyproject.toml, uv.lock,
   tsforecast/crossvals/base.py, notebooks renommés sur main, artefacts suivis comme
   coverage.xml / htmlcov / site) et propose une résolution pour chacun. Vérifie, pour sktime,
   darts, hierarchicalforecast (et ses dépendances statsforecast / utilsforecast si besoin),
   opera (Git, figé sur un SHA ; scipy à déclarer), xgboost et tslearn, la dernière version
   compatible Python 3.13 + pandas 3 + scikit-learn 1.9 + numpy 2.3 (métadonnées PyPI, notes de
   version). Présente-moi les incompatibilités et la question Q3 / Q4 du § 9 avant d'agir.
2. Après validation : git merge --no-commit --no-ff main, résolution des conflits (prendre main
   pour crossvals/base.py : get_n_splits y existe déjà ; garder default-groups de main ; ne pas
   réintroduire sktime dans le groupe dev).
3. pyproject.toml : extras d'adapters conformes au § 6 (adapters-sktime, adapters-darts,
   adapters-hierarchical, adapters-opera, adapters-all sans espace ; pas encore ceux des modèles de
   fondation), groupe notebooks hors default-groups, en-têtes et commentaires du fichier au style
   de main. Régénère uv.lock (uv lock).
4. Installe dans .venv les extras d'adapters et le groupe notebooks (procédure de la mémoire du
   projet : pas de pip, attribut ReadOnly, uv sync --no-install-project --inexact).
5. Vérifie : suite rapide de main inchangée (pytest -m "not slow"), import de chaque paquet
   tiers, import de chaque module d'adapter. Le module hierarchicalforecast.py ne s'importe pas
   (erreurs de syntaxe connues) : ne le corrige pas, note-le. Corrige uniquement les erreurs
   d'import ou d'API introduites par les montées de version dans sktime.py, darts.py et opera.py
   (corrections minimales, l'audit complet vient en sessions 3, 4 et 9).
6. Liste dans ton résumé : versions retenues, conflits résolus et comment, ce qui casse encore.

Ne termine pas le merge : je relis et je committe moi-même.
```

**Critères de fin** : merge en cours, sans conflit restant ; `uv.lock` régénéré ; suite rapide
de `main` verte ; versions des paquets tiers documentées dans le résumé.

**Message de commit suggéré**

```text
Merge branch 'main' into qb-adapters

- Bring the adapters branch up to date with main (pandas 3, scikit-learn 1.9, Python 3.13)
- Resolve conflicts in pyproject.toml, uv.lock and crossvals/base.py (main's get_n_splits kept)
- Declare adapter extras (sktime, darts, hierarchicalforecast, opera pinned to a commit) and a
  notebooks dependency group
- Minimal API fixes in the sktime, darts and opera adapters after the dependency upgrades
```

---

## Session 2 — Socle commun, packaging, infrastructure de tests

**Modèle** : Opus · **Effort** : high · **Plan mode** : oui

**Prompt**

```text
Lis CLAUDE.md, ADAPTERS_ARCHITECTURE.md (§ 3, § 4, § 6, § 7) et les règles communes du § 0 de
ADAPTERS_PROMPTS.md (tu ne commites jamais).

Objectif : poser le socle partagé par tous les adapters, sans encore réécrire les adapters.

1. Plan : propose l'API exacte de tsforecast/adapters/_base.py (§ 4) en réutilisant au maximum
   tsforecast.panel.utils, tsforecast.utils.frequency, tsforecast.utils.validation et
   tsforecast.utils.time (cherche d'abord ce qui existe, n'en réimplémente rien). Propose aussi la
   batterie de tests de contrat commune (§ 7) sous forme d'un module de tests paramétrable par
   adapter (fabrique d'estimateur, jeux série et panel), et la stratégie de collecte des doctests
   quand un paquet optionnel manque. Attends ma validation.
2. Implémente _base.py (docstrings Google anglaises avec doctests passants, commentaires
   français) et ses tests unitaires.
3. Réécris tsforecast/adapters/__init__.py : imports directs (plus de __getattr__ qui masque les
   erreurs), chaque module d'adapter n'important ses paquets tiers que dans ses méthodes via
   import_optional. Adapte sktime.py, darts.py et opera.py sur ce seul point. hierarchicalforecast.py
   ne s'importe pas encore : neutralise son import dans __init__ proprement et signale-le (il sera
   réécrit en session 6).
4. Infrastructure de tests : tests/unit/adapters/ (en-têtes, conftest), marqueur network déclaré
   dans pyproject.toml, collecte conditionnelle des doctests des adapters, job CI tests-adapters
   dans .github/workflows/tests.yml (tests rapides, sans modèles de fondation).
5. Vérifie : suite rapide complète, doctests (pytest --doctest-modules tsforecast),
   scripts/check_conventions.py --rule all, et que import tsforecast.adapters fonctionne dans un
   environnement sans aucun paquet optionnel (simule-le si besoin).
```

**Critères de fin** : `_base.py` testé ; batterie de contrat prête à être branchée sur chaque
adapter ; `import tsforecast.adapters` sans dépendance optionnelle ; CI adaptée.

**Message de commit suggéré**

```text
feat(adapters): add shared adapter toolkit and contract test suite

- tsforecast/adapters/_base.py: optional imports, temporal layout analysis, step counting
  between dates, row-order restoration, fitted feature info
- Direct imports in tsforecast.adapters; third-party packages imported lazily in methods
- Shared contract tests (clone, row order, series and panel, cross_val_score, GridSearchCV,
  pipelines, leakage), conditional doctest collection, network marker, adapters CI job
```

---

## Session 3 — Audit et correction de `SktimeAdapter`

**Modèle** : Opus · **Effort** : high · **Plan mode** : oui

**Prompt**

```text
Lis CLAUDE.md, ADAPTERS_ARCHITECTURE.md (§ 1, § 2.2, § 3, § 5.1, § 7) et les règles communes du
§ 0 de ADAPTERS_PROMPTS.md (tu ne commites jamais).

Objectif : vérifier qu'il n'y a ni erreur ni inefficience dans tsforecast/adapters/sktime.py, puis
le rendre conforme au contrat commun.

1. Plan = rapport d'audit. Pour chaque défaut (ceux du § 2.2 et ceux que tu trouves) : symptôme,
   scénario minimal qui le démontre (exécute-le), gravité, correction proposée. Points à
   instruire en priorité : sémantique de fh (relatif au cutoff vs dates des lignes de test, gap > 0,
   test_size > 1), panel (mtypes pd-multiindex / hiérarchiques, X et y convertis de façon
   cohérente), forecasters qui exigent X sur les dates du gap, mutation du forecaster, sortie non
   réalignée, coût des conversions. Attends ma validation.
2. Corrige selon le § 5.1 : clone du forecaster dans forecaster_, horizon absolu construit depuis
   les dates de X, get_params / set_params de BaseEstimator, sortie ndarray dans l'ordre de X,
   erreurs explicites. Supprime le paramètre fh s'il n'a plus de sens (le dire dans le résumé).
3. Branche la batterie de contrat de la session 2 sur SktimeAdapter (NaiveForecaster et
   YfromX(LinearRegression()) au moins), plus des tests ciblés de chaque défaut corrigé.
4. Vérifie doctests et check_conventions. Ne touche pas au notebook (session 5) mais liste les
   cellules qu'il faudra adapter.
```

**Message de commit suggéré**

```text
fix(adapters): make SktimeAdapter predict the rows of X without mutating the forecaster

- Absolute forecasting horizon built from the dates of X (gap and multi-row tests supported)
- Panel data passed in sktime's multi-index format for both X and y
- Forecaster cloned at fit, BaseEstimator parameter handling, ndarray output in row order
- Contract and regression tests
```

---

## Session 4 — Audit et correction de `DartsAdapter`

**Modèle** : Opus · **Effort** : high · **Plan mode** : oui

**Prompt**

```text
Lis CLAUDE.md, ADAPTERS_ARCHITECTURE.md (§ 1, § 2.2, § 3, § 5.2, § 7) et les règles communes du
§ 0 de ADAPTERS_PROMPTS.md (tu ne commites jamais).

Objectif : vérifier qu'il n'y a ni erreur ni inefficience dans tsforecast/adapters/darts.py, puis
le rendre conforme au contrat commun.

1. Plan = rapport d'audit, défaut par défaut, avec un scénario exécuté qui le démontre. Points à
   instruire : réinstanciation de self.model dans fit (clone, get_params, GridSearchCV,
   avertissement à chaque pli) ; panel par unstack en une série multivariée (covariables de toutes
   les entités vues par chaque entité, coût en colonnes) contre liste de TimeSeries passée à un
   modèle global ; static_covariates ; remplissage du gap par 0 et prévision récursive ; calcul de
   n ; grilles irrégulières ; modèles darts non « régression » (torch, statistiques locaux) : que
   doit faire l'adapter ? Montre si la prédiction dépend du gap. Attends ma validation.
2. Corrige selon le § 5.2 : model_ construit au fit, self.model intact, panel en liste de séries,
   prédiction indépendante du gap quand le modèle est un pur régresseur (prouvée par un test),
   sortie ndarray dans l'ordre de X, set_params conforme à clone.
3. Batterie de contrat (LinearRegressionModel et un modèle arborescent darts disponible sans GPU),
   plus tests ciblés. Doctests et check_conventions. Liste les cellules du notebook à adapter.
```

**Message de commit suggéré**

```text
fix(adapters): fit darts models per entity series without mutating the wrapped model

- Panel data passed as one TimeSeries per entity to a global model (no cross-entity covariates)
- Predictions independent of the train/test gap for pure regression models
- Fitted copy in model_, clone-compatible parameters, ndarray output in row order
- Contract and regression tests
```

---

## Session 5 — Notebook : cadre de prévision et sections sklearn → darts

**Modèle** : Sonnet · **Effort** : medium · **Plan mode** : non

**Prompt**

```text
Lis CLAUDE.md, ADAPTERS_ARCHITECTURE.md (§ 1, § 2.2 « Notebook », § 8) et les règles communes du
§ 0 de ADAPTERS_PROMPTS.md (tu ne commites jamais).

Objectif : remettre d'aplomb notebooks/4 - QB - Adapters.ipynb jusqu'à la section darts incluse.

1. Section 2 : aligne la construction du jeu supervisé sur docs/tutorials/pseudo_real_time_forecast.md
   (lignes indexées par origine, cible décalée de -h entité par entité, covariables connues à
   l'origine, gap = h + d - 1, splitters de tsforecast). Supprime les imports privés sklearn.
2. Sections 3.1 à 3.5 (sklearn, xgboost, tslearn, sktime, darts) : suis le gabarit du § 8
   (série temporelle, panel, Pipeline / XYPipeline, cross_val_score, GridSearchCV, limites) avec
   l'API actuelle des adapters (lis leur code et leurs tests, n'invente pas de paramètre).
3. Ne touche pas aux sections hierarchicalforecast et opera (elles seront réécrites) : si elles
   empêchent l'exécution, isole-les derrière une cellule markdown « section en cours de réécriture »
   et commente leur code, sans le supprimer.
4. Exécute le notebook de bout en bout (nbclient) et sauvegarde-le avec ses sorties ; aucune
   cellule en erreur, temps total raisonnable sur CPU.
```

**Message de commit suggéré**

```text
docs(notebooks): align the adapters notebook with the origin-indexed forecasting setup

- Supervised dataset built per entity, gap = h + d - 1, tsforecast splitters
- sklearn, xgboost, tslearn, sktime and darts sections rewritten on the current adapters API
  (time series, panel, pipelines, cross_val_score, GridSearchCV)
```

---

## Session 6 — `HierarchicalForecastAdapter` : conception et cross-section

**Modèle** : Opus · **Effort** : xhigh · **Plan mode** : oui

**Prompt**

```text
Lis CLAUDE.md, ADAPTERS_ARCHITECTURE.md (§ 1, § 2.2, § 2.3, § 3, § 5.3, § 7, § 9 : Q5, Q6, Q7) et
les règles communes du § 0 de ADAPTERS_PROMPTS.md (tu ne commites jamais). Lis aussi l'état actuel
de tsforecast/adapters/hierarchicalforecast.py, la section 3.6 de
notebooks/4 - QB - Adapters.ipynb et Debug - HierarchicalForescast.ipynb (racine) pour récupérer les
intentions de l'auteur, ainsi que les deux tutoriels :
https://nixtlaverse.nixtla.io/hierarchicalforecast/examples/australiandomestictourismcrosstemporal.html
https://nixtlaverse.nixtla.io/hierarchicalforecast/examples/australiandomestictourism-multimodel.html
et le code source de hierarchicalforecast installé (aggregate, aggregate_temporal,
HierarchicalReconciliation.reconcile, get_cross_temporal_tags, éventuelles méthodes
cross-temporelles natives).

Objectif : finir l'adapter. Cette session couvre la conception d'ensemble (les trois modes) et
l'implémentation du mode cross-section ; le temporel et le cross-temporel viennent en session 7.

1. Plan : design détaillé et définitif du § 5.3 — format des données (y long sur tous les nœuds,
   niveaux d'index portant les nœuds, étiquette des agrégats), deux formes de hierarchy (spec de
   niveaux d'index ; dict parent -> enfants sur un niveau de nœuds), utilitaire de conversion
   large <-> long, séries temporelles vs panel (entités copies indépendantes vs entités = niveaux
   de la hiérarchie), multi-modèle (model_combination), réconciliateurs insample (X_train =
   prévisions hors échantillon), sortie (reconciler singulier pour predict et GridSearchCV,
   reconcile() pour tout comparer), exogènes, paramètres transmis à reconcile (level,
   intervals_method, num_samples, seed, is_balanced, diagnostics). Montre le design des modes
   temporel et cross-temporel pour vérifier qu'il s'emboîte, sans les implémenter. Tranche Q5 à
   Q7 avec moi.
2. Réécris hierarchicalforecast.py (fichier unique ou petit sous-paquet si plus clair) : mode
   cross-section complet, conforme au § 3 ; le mode temporel lève NotImplementedError pour
   l'instant. Réintègre l'adapter dans tsforecast/adapters/__init__.py.
3. Tests : batterie de contrat + cohérence numérique (agrégats = somme des feuilles après
   réconciliation, à la tolérance près) pour : série temporelle avec hiérarchie de nœuds
   (a = b + c, c = d + e), panel avec copies indépendantes, panel dont les entités forment la
   hiérarchie, deux modèles de base (mean et un modèle par niveau), BottomUp / MinTrace(ols) /
   MinTrace(mint_shrink) ; cross_val_score avec PanelOutOfSampleSplit ; GridSearchCV sur
   reconciler.
4. Mets à jour le § 5.3 et le journal des décisions du document d'architecture avec le design
   définitif. Ne touche pas au notebook.
```

**Message de commit suggéré**

```text
feat(adapters): implement cross-sectional reconciliation in HierarchicalForecastAdapter

- Long node layout on the index (series and panel), hierarchy given as a level spec or a
  parent -> children mapping, wide <-> long conversion helper
- Multi-model base forecasts (mean or one model per level), insample reconcilers fed with
  out-of-sample base forecasts
- predict returns one reconciler's values in row order (GridSearchCV over reconciler);
  reconcile() returns every model/reconciler column
- Contract, coherence, cross_val_score and GridSearchCV tests
```

---

## Session 7 — `HierarchicalForecastAdapter` : temporel et cross-temporel

**Modèle** : Opus · **Effort** : high · **Plan mode** : oui

**Prompt**

```text
Lis CLAUDE.md, ADAPTERS_ARCHITECTURE.md (§ 5.3 tel que mis à jour en session 6, § 7, § 10) et les
règles communes du § 0 de ADAPTERS_PROMPTS.md (tu ne commites jamais). Relis le tutoriel
https://nixtlaverse.nixtla.io/hierarchicalforecast/examples/australiandomestictourismcrosstemporal.html
et le code installé de aggregate_temporal, get_cross_temporal_tags et reconcile(temporal=True).

Objectif : implémenter les modes temporel et cross-temporel du HierarchicalForecastAdapter,
compatibles cross_val_score et GridSearchCV.

1. Plan : représentation des niveaux temporels dans l'index, datation des agrégats, contrôle des
   périodes complètes dans X au predict (et conséquence pour les splitters : test couvrant des
   périodes complètes du niveau le plus agrégé), S et tags propres à l'horizon de test calculés
   sans modifier les attributs ajustés, refus des méthodes insample en temporel, interaction avec
   des lignes indexées par origine (ancrage calendaire). Cross-temporel : méthode native si la
   version installée en a une, sinon composition explicite (ordre des passes justifié) ; montre
   que la sortie est cohérente dans les deux dimensions. Attends ma validation.
2. Implémente, pour séries temporelles et panels.
3. Tests : cohérence temporelle (somme des mois = trimestre = année), cohérence cross-temporelle
   dans les deux dimensions, panel déséquilibré, période incomplète refusée, cross_val_score avec
   un splitter dont le test couvre des années complètes, GridSearchCV (reconciler et, si pertinent,
   aggregation_type).
4. Mets à jour § 5.3 et le journal des décisions. Ne touche pas au notebook.
```

**Message de commit suggéré**

```text
feat(adapters): add temporal and cross-temporal reconciliation to HierarchicalForecastAdapter

- Temporal levels carried by the index, aggregates dated at period start, incomplete periods
  rejected at predict, horizon-specific summing matrix kept local to predict
- Cross-temporal reconciliation coherent across both the hierarchy and the temporal levels
- Series and panel tests, cross_val_score with full-period test windows, GridSearchCV
```

---

## Session 8 — Notebook : section hierarchicalforecast

**Modèle** : Sonnet · **Effort** : high · **Plan mode** : non

**Prompt**

```text
Lis CLAUDE.md, ADAPTERS_ARCHITECTURE.md (§ 5.3, § 8) et les règles communes du § 0 de
ADAPTERS_PROMPTS.md (tu ne commites jamais). Lis le code et les tests de
tsforecast/adapters/hierarchicalforecast.py : n'utilise que son API réelle.

Objectif : réécrire la section 3.6 de notebooks/4 - QB - Adapters.ipynb.

Contenu attendu (gabarit du § 8, données synthétiques de la section 1.2 étendues si besoin) :
1. production des prévisions de base hors échantillon avec cross_val_predict et un splitter de
   tsforecast (jamais de prévisions in-sample pour les réconciliateurs insample) ;
2. cross-section sur une série temporelle (a = b + c, c = d + e) puis sur un panel (copies
   indépendantes, puis entités formant la hiérarchie) ;
3. temporel (mois -> trimestre -> année) sur une série puis sur un panel ;
4. multi-modèle, sur le modèle du tutoriel australiandomestictourism-multimodel (deux ou trois
   modèles de base, moyenne et un modèle par niveau) ;
5. cross-temporel, sur le modèle du tutoriel australiandomestictourismcrosstemporal ;
6. cross_val_score et GridSearchCV (comparaison des réconciliateurs) ;
7. pour chaque cas, contrôle numérique de cohérence avant / après réconciliation et erreur
   comparée des prévisions de base et réconciliées ; limites et pièges.

Supprime Debug - HierarchicalForescast.ipynb à la racine une fois que rien d'utile n'y reste
(dis-moi ce que tu en as repris). Exécute tout le notebook (nbclient) et sauvegarde-le avec ses
sorties.
```

**Message de commit suggéré**

```text
docs(notebooks): illustrate hierarchical reconciliation with HierarchicalForecastAdapter

- Cross-sectional (series and panel), temporal (series and panel), multi-model and
  cross-temporal examples with coherence checks
- cross_val_score and GridSearchCV over reconcilers
- Remove the root-level hierarchicalforecast debug notebook
```

---

## Session 9 — `OperaAdapter`

**Modèle** : Opus · **Effort** : high · **Plan mode** : oui

**Prompt**

```text
Lis CLAUDE.md, ADAPTERS_ARCHITECTURE.md (§ 2.2, § 2.3, § 3, § 5.4, § 7, § 9 : Q4) et les règles
communes du § 0 de ADAPTERS_PROMPTS.md (tu ne commites jamais). Lis le code source d'opera installé
(opera/mixture.py) : constructeur, update, predict, attributs w / weights, contrôle des colonnes.

Objectif : reconstruire OperaAdapter, pour séries temporelles et panels, compatible
cross_val_score et GridSearchCV.

1. Plan : audit rapide de l'adapter actuel (scénarios exécutés), puis design selon le § 5.4 :
   mode méta-estimateur (experts en _BaseComposition, prévisions hors échantillon des experts par
   splitter interne, Mixture séquentielle triée par date, réajustement des experts) et mode
   passthrough ; pooling global / per_entity et entité inconnue ; poids finaux au predict ;
   partial_fit ; validation des paramètres opera ; coût (nombre d'ajustements d'experts) et
   moyens de le borner. Vérifie la compatibilité d'opera avec pandas 3 et tranche Q4 avec moi.
2. Implémente, docstrings Google avec doctests (mode passthrough, données minuscules).
3. Tests : batterie de contrat (deux modes), poids qui convergent vers le meilleur expert sur un
   jeu synthétique où un expert domine, aucune prévision in-sample utilisée pour les poids,
   per_entity vs global, GridSearchCV sur aggregation et sur un paramètre imbriqué d'expert
   (experts__ridge__alpha), cross_val_score panel.
4. Mets à jour § 5.4 et le journal des décisions. Ne touche pas au notebook.
```

**Message de commit suggéré**

```text
feat(adapters): rebuild OperaAdapter as an online expert-aggregation meta-estimator

- Experts fitted out of sample with a temporal splitter, mixture run in date order,
  final weights used at predict; passthrough mode for precomputed expert forecasts
- Global or per-entity pooling for panel data, partial_fit continuation
- Nested expert parameters for GridSearchCV; contract and behaviour tests
```

---

## Session 10 — Notebook : section opera

**Modèle** : Sonnet · **Effort** : medium · **Plan mode** : non

**Prompt**

```text
Lis CLAUDE.md, ADAPTERS_ARCHITECTURE.md (§ 5.4, § 8) et les règles communes du § 0 de
ADAPTERS_PROMPTS.md (tu ne commites jamais). Lis le code et les tests de tsforecast/adapters/opera.py.

Objectif : réécrire la section 3.7 de notebooks/4 - QB - Adapters.ipynb selon le gabarit du § 8 :
série temporelle et panel (pooling global et per_entity), mode méta-estimateur avec des experts
hétérogènes (dont, si utile, un adapter d'une section précédente) et mode passthrough, évolution des
poids (graphique), comparaison aux experts seuls et à leur moyenne, cross_val_score, GridSearchCV
(algorithme d'agrégation, perte, paramètre d'expert imbriqué), partial_fit, limites. Exécute le
notebook (nbclient) et sauvegarde-le avec ses sorties.
```

**Message de commit suggéré**

```text
docs(notebooks): illustrate online expert aggregation with OperaAdapter

- Series and panel examples (global and per-entity pooling), meta-estimator and passthrough modes
- Weight trajectories, comparison with single experts, cross_val_score and GridSearchCV
```

---

## Session 11 — Modèles de fondation : adapter, protocole, backends Chronos et `callable`

**Modèle** : Opus · **Effort** : xhigh · **Plan mode** : oui

**Prompt**

```text
Lis CLAUDE.md, ADAPTERS_ARCHITECTURE.md (§ 1, § 2.3, § 3, § 4, § 5.5, § 6, § 7, § 9 : Q8) et les
règles communes du § 0 de ADAPTERS_PROMPTS.md (tu ne commites jamais). Consulte la documentation à
jour de chronos-forecasting (README GitHub amazon-science/chronos-forecasting, code de
Chronos2Pipeline) ainsi que les cartes Hugging Face mistralai/Timestral-209M et
theforecastingcompany/t0-gamma, pour que le protocole couvre aussi ces deux familles.

Objectif : FoundationModelAdapter générique à backends, compatible séries temporelles, panels,
cross_val_score et GridSearchCV.

1. Plan : protocole ForecastBackend / ForecastRequest / ForecastResult et déclaration de
   capacités, vérifié contre les trois API (Chronos-2, Timestral, t0) ; registre et résolution du
   backend depuis model_id ; sémantique fit / predict du § 5.5 (contexte mémorisé par entité,
   nombre de pas déduit des index, extraction des pas demandés, lignes du gap déclarées
   manquantes, covariables passées / futures) ; cache de modèles au niveau du processus et
   interaction avec clone, GridSearchCV et n_jobs ; quantiles (predict_quantiles) ; mise en lots ;
   périphérique ; extras adapters-chronos (et la place des futurs adapters-timestral,
   adapters-t0). Donne-moi aussi ton avis argumenté : limites du paradigme pour ces modèles, coût
   d'une validation croisée, ce qu'il vaut mieux ne pas promettre. Attends ma validation.
2. Implémente tsforecast/adapters/foundation/ (adapter, protocole, registre, backends chronos et
   callable — le second permet de brancher n'importe quel modèle par une fonction).
3. Tests : logique de l'adapter avec un backend factice sans réseau (batterie de contrat
   complète, alignement des pas avec gap > 0, panel déséquilibré, absence de fuite, covariables,
   quantiles, cache) ; test réel Chronos avec le plus petit modèle, marqué slow et network.
4. Mets à jour § 5.5, § 6 et le journal des décisions. Ne touche pas au notebook.
```

**Message de commit suggéré**

```text
feat(adapters): add FoundationModelAdapter with pluggable forecasting backends

- Backend protocol with declared capabilities, registry resolved from the model id
- Context stored per entity at fit; steps to each test row derived from the index at predict;
  past and future covariates when the backend supports them; quantile forecasts
- Process-level model cache compatible with clone and GridSearchCV
- Chronos and callable backends; offline contract tests with a fake backend, slow network test
```

---

## Session 12 — Modèles de fondation : backends Timestral et t0

**Modèle** : Sonnet · **Effort** : high · **Plan mode** : non

**Prompt**

```text
Lis CLAUDE.md, ADAPTERS_ARCHITECTURE.md (§ 2.3, § 5.5, § 6, § 10) et les règles communes du § 0 de
ADAPTERS_PROMPTS.md (tu ne commites jamais). Lis tsforecast/adapters/foundation/ (protocole,
registre, backend chronos, tests) : tu ajoutes des backends sans modifier le protocole. Si le
protocole ne suffit pas, arrête-toi et explique-moi pourquoi avant de le changer.

Objectif : backends timestral (mistralai/Timestral-*, paquet Git github.com/mistralai/timestral) et
t0 (theforecastingcompany/t0-*, paquet PyPI tfc-t0[torch]).

1. Lis les cartes Hugging Face et le code des deux paquets : format du contexte, covariables
   passées / futures, quantiles natifs, horizon maximal par passe, inférence par lots (batch_series
   et mask / group_ids pour t0), périphérique.
2. Ajoute les extras adapters-timestral (Git figé sur un SHA) et adapters-t0, régénère uv.lock,
   installe-les dans .venv (procédure de la mémoire du projet).
3. Implémente les deux backends, déclare leurs capacités, enregistre leurs préfixes d'identifiants.
4. Tests : tests hors ligne du mapping (paquet simulé) et tests réels avec Timestral-5M et t0-gamma
   sur CPU, marqués slow et network. Rappelle les licences dans les docstrings.
5. Mets à jour § 2.3 / § 5.5 et le journal des décisions.
```

**Message de commit suggéré**

```text
feat(adapters): add Timestral and t0 backends to FoundationModelAdapter

- Context, covariates and quantile mapping for mistralai/Timestral-* and
  theforecastingcompany/t0-* models, batched inference
- adapters-timestral and adapters-t0 extras; offline mapping tests and slow network tests
```

---

## Session 13 — Notebook : section modèles de fondation

**Modèle** : Sonnet · **Effort** : medium · **Plan mode** : non

**Prompt**

```text
Lis CLAUDE.md, ADAPTERS_ARCHITECTURE.md (§ 5.5, § 8) et les règles communes du § 0 de
ADAPTERS_PROMPTS.md (tu ne commites jamais). Lis le code et les tests de tsforecast/adapters/foundation/.

Objectif : ajouter une section 3.8 « Modèles de fondation (Hugging Face) » à
notebooks/4 - QB - Adapters.ipynb selon le gabarit du § 8 : principe (zéro-shot, ce que fit et
predict font dans le cadre tsforecast, ce que GridSearchCV peut régler), série temporelle et panel
avec Chronos, Timestral et t0 (plus petits modèles, CPU), covariables avec / sans,
predict_quantiles et intervalles tracés, backend callable (exemple minimal), cross_val_score,
GridSearchCV sur model_id / context_length, comparaison aux adapters précédents, licences et
limites (coût, n_jobs, téléchargements). Exécute le notebook (nbclient) et sauvegarde-le avec ses
sorties ; signale la durée d'exécution de la section.
```

**Message de commit suggéré**

```text
docs(notebooks): illustrate zero-shot foundation models with FoundationModelAdapter

- Chronos, Timestral and t0 on series and panel data, with and without covariates
- Quantile forecasts, callable backend, cross_val_score and GridSearchCV over inference settings
```

---

## Session 14 — Documentation : page concept et tutoriel des adapters

**Modèle** : Sonnet · **Effort** : high · **Plan mode** : oui

**Prompt**

```text
Lis CLAUDE.md (sections Documentation et Règles de rédaction du code), ADAPTERS_ARCHITECTURE.md en
entier et les règles communes du § 0 de ADAPTERS_PROMPTS.md (tu ne commites jamais). Prends pour
modèles de structure et de ton docs/concepts/sklearn_extensions.md et
docs/tutorials/sklearn_extensions.md, ainsi que docs/tutorials/pseudo_real_time_forecast.md pour la
construction du jeu supervisé (origine, horizon, délai, gap = h + d - 1). Lis le code, les docstrings
et les tests de tsforecast/adapters/ : la documentation décrit l'API réelle, jamais celle du
document d'architecture si les deux divergent (signale alors l'écart).

Objectif : créer docs/concepts/adapters.md et docs/tutorials/adapters.md, en français.

1. Plan : propose le plan détaillé des deux pages (titres, contenu de chaque section, exemples
   prévus, schémas éventuels) et attends ma validation. Points attendus :
   - concept : le problème (modèles tiers qui gèrent eux-mêmes l'horizon, formats propriétaires,
     incompatibilité avec cross_val_score / GridSearchCV / XYPipeline), le principe « l'horizon
     appartient à tsforecast, l'adapter apprend y depuis X ligne à ligne sans décalage », le
     contrat commun (paramètres, objet tiers jamais modifié, sortie dans l'ordre des lignes de X,
     séries temporelles et panels, erreurs explicites, dépendances optionnelles), puis une section
     par adapter (sktime, darts, hierarchicalforecast avec ses trois modes et le multi-modèle,
     opera, modèles de fondation et leurs backends) : ce que fit et predict font, ce que
     GridSearchCV peut régler, limites ; un tableau de choix « quel adapter pour quel besoin » ;
     « Pour aller plus loin » (tutoriel, pages API, notebook) ;
   - tutoriel : table des matières, données (série temporelle et panel construits entité par
     entité), puis une section par adapter avec un exemple minimal série, un exemple panel,
     cross_val_score avec un splitter de tsforecast, GridSearchCV, et le tableau d'arguments ;
     installation de chaque extra en tête de section ;
   - schéma draw.io du principe (lignes indexées par origine, gap, ce que voit chaque adapter) si
     tu le juges utile : sources dans docs/assets/drawio, PNG dans docs/assets/images (cf. mémoire
     du projet).
2. Rédige les pages en respectant les conventions de main : tableaux d'arguments sans bloc de
   signature, encadrés par <!-- doc-args: tsforecast.adapters.<Classe> --> et <!-- /doc-args -->,
   noms exactement égaux aux paramètres du constructeur ; exemples de code exécutables dont les
   sorties en commentaire sont celles de l'exécution (exécute chaque bloc et recopie la sortie,
   ne l'invente jamais) ; données minuscules et déterministes ; modèles de fondation sur CPU avec
   le plus petit modèle, en précisant que l'exemple télécharge des poids.
3. Ajoute les deux pages à la navigation de mkdocs.yml (Concepts et Tutoriels) et une ligne au
   tableau de docs/index.md.
4. Vérifie : mkdocs build --strict, scripts/check_conventions.py --rule all (doc-args en
   particulier), et réexécute tous les blocs de code du tutoriel pour confirmer les sorties.
```

**Critères de fin** : deux pages relues par rapport au code ; tous les exemples exécutés et leurs
sorties à jour ; `mkdocs build --strict` et `check_conventions` verts.

**Message de commit suggéré**

```text
docs(adapters): add concept and tutorial pages for the third-party adapters

- Concept page: the forecasting horizon belongs to tsforecast, shared adapter contract,
  per-adapter behaviour and limits, choice table
- Tutorial: series and panel examples, cross_val_score and GridSearchCV for each adapter,
  argument tables checked against the constructors
- Navigation and home page entries
```

---

## Session 15 — Documentation : pages API, navigation, CLAUDE.md, README

**Modèle** : Sonnet · **Effort** : medium · **Plan mode** : non

**Prompt**

```text
Lis CLAUDE.md (section Documentation), ADAPTERS_ARCHITECTURE.md et les règles communes du § 0 de
ADAPTERS_PROMPTS.md (tu ne commites jamais).

Objectif : compléter la documentation de référence des adapters au standard de main.

1. Pages API docs/api/adapters/*.md (mkdocstrings), une par classe publique et une pour le
   protocole et le registre des backends de modèles de fondation, sur le modèle des pages API
   existantes ; section « adapters » de la navigation de mkdocs.yml ; liens depuis
   docs/concepts/adapters.md et docs/tutorials/adapters.md.
2. CLAUDE.md : sous-paquet tsforecast/adapters dans l'arborescence et sa description (au même
   niveau de détail que les autres sous-paquets) ; mise à jour des sections « Implémenté » et
   « À implémenter ».
3. README.md : extras d'installation des adapters.
4. Renommage du notebook selon Q2 du § 9 de ADAPTERS_ARCHITECTURE.md : pose-moi la question, et si
   je confirme, renomme-le et mets à jour toutes ses références.
5. Vérifie : mkdocs build --strict, scripts/check_conventions.py --rule all, doctests.
```

**Message de commit suggéré**

```text
docs(adapters): add API reference pages and document the adapters subpackage

- mkdocstrings pages for the adapters, the foundation-model backend protocol and registry
- CLAUDE.md and README updated with the adapters subpackage and installation extras
```

---

## Session 16 — Revue finale de la branche

**Modèle** : Opus · **Effort** : high · **Plan mode** : non

**Prompt**

```text
Lis CLAUDE.md, ADAPTERS_ARCHITECTURE.md et les règles communes du § 0 de ADAPTERS_PROMPTS.md (tu
ne commites jamais).

Objectif : revue de qb-adapters par rapport à main avant la pull request.

1. Lance /code-review high sur la branche (diff contre main) et lis le rapport.
2. Vérifie toi-même le contrat commun sur chaque adapter (§ 3) et la batterie de tests (§ 7) :
   ce qui n'est pas couvert, ce qui est couvert mais faux.
3. Lance la suite complète (slow inclus, en arrière-plan : elle dépasse 10 min), les tests
   network si le réseau est disponible, les doctests, check_conventions, mkdocs build --strict, et
   l'exécution complète du notebook.
4. Corrige les défauts confirmés de gravité importante ; liste les autres.
5. Propose ce qu'il faut faire des fichiers ADAPTERS_*.md (Q9) et un texte de description de PR.
```

**Message de commit suggéré**

```text
fix(adapters): address final review findings

- <un point par correction>
```
