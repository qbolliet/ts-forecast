# Rapport P1 — Passage de ts-forecast à pandas ≥ 3.0 (et scikit-learn ≥ 1.9)

Date : 2026-10-19 · Branche : `qb-mixed-frequencies` · Rien n'est commité : l'auteur relit et committe.

**En une phrase** : le paquet tourne sous pandas 3.0.6 et scikit-learn 1.9.1, sans aucun écart de
valeur ni changement d'exception par rapport à pandas 2.3.1 (185 sorties de référence), suite
complète verte (13 802 tests) avec les avertissements de dépréciation de pandas traités comme des
erreurs ; trois anomalies « à arbitrer » sur l'**unité des dates en sortie** restent ouvertes
(prompts P2 et P3, §11).

---

## Sommaire

1. [Critères d'acceptation](#1-critères-dacceptation)
2. [Pages de documentation lues](#2-pages-de-documentation-lues)
3. [Bornes de dépendances](#3-bornes-de-dépendances)
4. [Inventaire chiffré et traitement, par famille](#4-inventaire-chiffré-et-traitement-par-famille)
5. [Changements de code (`tsforecast/`)](#5-changements-de-code-tsforecast)
6. [Tests créés et modifiés](#6-tests-créés-et-modifiés)
7. [Différences de sortie pandas 2.3 → 3, justifiées](#7-différences-de-sortie-pandas-23--3-justifiées)
8. [Anomalies](#8-anomalies)
9. [Doctests, notebooks, documentation, conventions](#9-doctests-notebooks-documentation-conventions)
10. [Environnement et incidents](#10-environnement-et-incidents)
11. [Décisions à arbitrer — vue d'ensemble](#11-décisions-à-arbitrer--vue-densemble)
12. [Prompts de résolution et d'arbitrage](#12-prompts-de-résolution-et-darbitrage)

---

## 1. Critères d'acceptation

| Critère | Base (pandas 2.3.1, sklearn 1.7.1) | État final (pandas 3.0.6, sklearn 1.9.1) | Statut |
|---|---|---|---|
| Suite complète (slow compris) | 13 192 passed, 0 xfail, 0 skip | **13 802 passed** (+618 nouveaux), 0 failed, 0 xpass ; aucun test existant n'a changé d'issue | ✅ |
| Avertissements pandas | 1 `FutureWarning` (`groupby`), alias dépréciés dans un test | 0 — passe complète avec `-W error::FutureWarning -W error::DeprecationWarning` (couvre `Pandas4Warning`) | ✅ |
| Sorties de référence (185 cas) | — | **0 écart de valeur, 0 changement d'exception** ; dtypes, unités et 1 ulp justifiés (§7) | ✅ |
| Couverture lignes + branches | 99,357 % | 99,36 % ; **aucun module en recul** (`delays/transformers.py` 99,599 → 99,608 %) | ✅ |
| Durée (même commande, sous couverture) | 789,96 s | 823,83 s, **+4,3 %** avec 618 tests de plus (seuil : +15 %) | ✅ |
| `scripts/check_conventions.py --rule all` | 18 violations bloquantes (toutes `doctest`) | 23, toutes `doctest` (5 nouvelles : représentation, §9) ; `refs` / `header` / `length` inchangés | ✅ hors doctests |
| `mkdocs build --strict` | échec, 189 avertissements griffe | mêmes 189 avertissements (ensemble identique) | ⚠️ préexistant (P5) |

Les 8 identifiants de tests « disparus » sont des tests d'alias renommés ou re-paramétrés
(`test_bare_code_is_deprecated_by_pandas` → `test_bare_code_is_rejected_by_pandas`, dates d'or
passées en paramètre), comptés dans les 618 nouveaux.

**Méthode** : migration d'abord à l'intérieur de pandas 2.3.1 avec `PANDAS_COPY_ON_WRITE=1` et
`PANDAS_FUTURE_INFER_STRING=1` (variables lues à l'import, donc propagées aux workers joblib et aux
noyaux Jupyter) et `-W error`, puis montée de version en deux marches pour attribuer chaque écart :
4a — scikit-learn 1.9.1 seul sous pandas 2.3 (aucun changement) ; 4b — pandas 3.0.6.
Sorties de référence : script rejouable sous les deux versions (HFI sur la matrice réaliste
dataset × stratégie × modalité et sur PANEL-X, délais, splitters, convertisseur / détecteur /
validation sur toutes les perturbations, `PanelwiseTransformer`, `XYPipeline`), sérialisation
neutre indépendante de pandas, déterminisme vérifié (deux passes identiques).

## 2. Pages de documentation lues

- pandas — *What's new in 3.0.0 (January 21, 2026)* : changements incompatibles, retraits de
  dépréciations, corrections ;
- pandas — *Migration guide for the new string data type (pandas 3.0)* ;
- pandas — *Copy-on-Write (CoW)* ;
- pandas — *What's new in 2.3.0* (option `future.infer_string`, repli sans pyarrow) ;
- pandas — PDEP-14 *Dedicated string data type for pandas 3.0* ;
- scikit-learn — changelogs 1.8 et 1.9 (1.9.0 : `check_array(dtype="numeric")` rejette désormais
  les colonnes `StringDtype`, qu'il acceptait en silence sous pandas 3 ; dépendance narwhals).

Faits pandas 3 vérifiés par sonde dans l'environnement (pandas 3.0.6) : alias retirés → `ValueError`
(`M Q Y SM A H T S L U N`) ; unité inférée `us` pour chaînes et `datetime` ; `date_range` garde
l'unité d'un `Timestamp` d'entrée ; `Period.end_time` en `us` ; `DatetimeIndex.equals` compare
désormais les instants entre unités (faux sous 2.3) ; `Day` n'est plus un `Tick` ;
`concat(sort=None)` sur `DatetimeIndex` trie mais émet `Pandas4Warning` ; `stack()` garde les NaN ;
`get_period_alias` n'accepte plus qu'une chaîne.

## 3. Bornes de dépendances

| Paquet | Avant | Après | Remarque |
|---|---|---|---|
| pandas | `>=2.3.1,<3` | **`>=3.0,<4`** | installé 3.0.6 |
| scikit-learn | `>=1.7.0,<2` | **`>=1.9,<2`** | validé par l'auteur ; installé 1.9.1 |
| numpy | `>=2.3.1,<3` | inchangé | 2.3.2 (Python < 3.14) ; le verrou bifurque vers 2.5.4 pour Python ≥ 3.14 |
| joblib, groupes `dev` / `docs` | — | inchangés | mkdocs, mkdocstrings, nbclient, ipykernel, pytest : indépendants de pandas |
| narwhals | — | ajouté (transitif) | dépendance de scikit-learn 1.9 |
| pytz | transitif | retiré du verrou | plus requis par pandas 3 (fuseaux `zoneinfo`) |
| pyarrow | absent | non ajouté | dtype `str` en stockage Python, même sémantique (décision : P4) |

`uv.lock` régénéré par `uv lock` (rien à refaire). scipy 1.16.1, matplotlib 3.10.5 et seaborn 0.13.2
fonctionnent sous pandas 3 (notebooks).

## 4. Inventaire chiffré et traitement, par famille

Comptes `occurrences / fichiers` (git grep ; notebooks : cellules de code seulement) ; « réel » =
après relecture des correspondances.

| Fam. | Motif | `tsforecast/` | `tests/` | notebooks | Traitement |
|---|---|---|---|---|---|
| a — CoW | chaînage `x[..][..] = v` | 0 réel (1 faux positif : dict de dicts) | 0 | 2 | aucun code à changer ; garde : `tests/integration/input_contract/` |
| a | `inplace=True` | 2 / 1 (objet local, `delays/data_manager.py`) | 0 | 0 | sans effet de bord ; inchangé |
| a | `.values` / `.to_numpy()` / `np.asarray` | 6 / 48 / 40 | 8 / 192 / 92 | 50 | 0 écriture dans un tableau extrait (écritures dans des `np.full` seulement) ; garde : entrées en lecture seule |
| b — `str` | `dtype == object`, `dtype.kind`, `astype(str)` | 0 / 3 / 4 | 2 / 9 / 15 | 0 | 7 tests rendus indépendants du nom de dtype |
| b | `dtype=object` explicite | 34 / 11 | 15 / 10 | 0 | conservés (valeurs hétérogènes) ; harmonisation = P4 |
| c — résolution | `.asi8`, `.value`, `end_time` | 2, 3, 7 | — | — | **2 défauts corrigés** (`end_time`, §5) ; `.value` toujours en ns : sûr |
| c | littéraux `datetime64[ns]`, `Timedelta(…, unit='ns')` | 11 / 3, 9 / 5 | 2 / 2 | — | 1 test réécrit ; doctests : N7 / N8 |
| c | constructeurs à unité inférée (`date_range`, `to_datetime`, `Timestamp`) | 129 / 35 / 60 | 606 / 68 / 196 | — | tests `TestDatetimeResolution` (11 fichiers) |
| d — alias | alias retirés passés à pandas | 0 hors doctests (modernisation existante) ; 1 doctest | ~0 en littéral ; 4 tests à oracle « pandas 2 » | 19 `freq=` | oracles remplacés par des valeurs d'or figées depuis pandas 2 ; 2 cellules corrigées |
| e — API | `concat` | 20 / 8, dont 6 `axis=1` | 67 / 42 | 38 | trames vides écartées ; union explicite (§5) ; 2 cellules `sort=True` |
| e | `groupby(level=[…])` itéré | 1 réel (`delays/transformers.py`) | — | — | niveau scalaire |
| e | `stack()` (NaN conservés) | 0 | 3 | 3 | 2 tests `stack().dropna()` |
| e | `fillna(method=)`, `applymap`, `Index.format`, `groupby(axis=)`… | 0 | 0 | 1 `fillna(method=)` | cellule : `.ffill()` |
| f — autres | `Day` non `Tick`, fuseaux `zoneinfo` | 1, 2 | 1 | — | aucun effet observé ; tests de garde verts |

## 5. Changements de code (`tsforecast/`)

+112 / −31 lignes, cinq fichiers. Chaque correctif donne, **sous pandas 2.3, des sorties de
référence identiques au bit près** (vérifié avant la montée de version).

| Fichier | Changement | Pourquoi |
|---|---|---|
| `delays/data_manager.py` | `concat` des seules trames de détection non vides | pandas 3 compte les entrées vides dans le dtype du résultat (`FutureWarning` en 2.3) |
| `delays/transformers.py` (`_build_entity_params`) | `groupby(level=0)` scalaire quand il n'y a qu'un niveau d'entité | une liste de longueur 1 donne des clés en tuples (`FutureWarning` en 2.3) |
| `delays/transformers.py` (`PublicationDelayTransformer._assemble`) | index de dates : union par `Index.union`, chaque groupe réindexé dessus ; autres index : `concat(sort=False)` explicite ; aucune jointure avec un jeu sans colonne | `Pandas4Warning` sur le tri par défaut ; un `sort=True` naïf perdait `freq='MS'` en sortie (constaté et corrigé, test ajouté) |
| `frequency/covariate_materializer.py` | `_combine_first` : la production remplace un registre de dtype différent qu'elle recouvre entièrement, sans concaténer de partie vide | dtype du registre (`FutureWarning` en 2.3) ; équivalence exacte avec `combine_first` vérifiée sur 3 000 tirages |
| `frequency/stage_scaler.py` | `_period_span_nanoseconds` : durée = début de la période suivante − début | ANO-FREQ-085 : `Period.end_time` en `us` sous pandas 3 → 999 ns perdues par période |
| `utils/frequency/converter.py` (`_shift_index_to_anchor_fraction`) | bornes de période en nanosecondes tirées des débuts de période | ANO-UTILS-071 : même cause, ancres décalées |

Aucune branche `if pandas_version`, aucun `# type: ignore`, aucun `warnings.filterwarnings`.

## 6. Tests créés et modifiés

**Nouveaux (618 tests)** :

| Fichier / classe | Tests | Objet |
|---|---:|---|
| `tests/support/perturbations.py` | — | `with_datetime_resolution(obj, unit)`, `datetime_units(obj)`, `DATETIME_UNITS` |
| `tests/support/test_perturbations.py` (`TestWithDatetimeResolution`, `TestDatetimeUnits`) | 19 | les deux utilitaires |
| `tests/unit/**/test_datetime_resolution.py` (validation, détecteur, convertisseur, délais ×3, HFI, crossvals ×2) | 380 | résultat indépendant de la résolution (`s`/`ms`/`us`/`ns`), unité de sortie épinglée |
| `TestDatetimeResolution` dans `utils/time/test_utils.py`, `utils/duration/test_utils.py`, `crossvals/test_time_series.py` | 141 | idem |
| `tests/integration/input_contract/test_non_mutation.py` | 25 | rien de ce que reçoit le paquet n'est modifié (valeurs, dtypes, index, `freq`) |
| `tests/integration/input_contract/test_readonly_inputs.py` | 25 | entrées sur tampons NumPy `writeable=False` : mêmes sorties |
| `tests/integration/input_contract/test_string_labels.py` | 16 | libellés `object` et `str` (caractères spéciaux) : aller-retour identique |
| exactitude `end_time`, `freq` de `PublicationDelayTransformer`, alias renommés | 12 | garde des correctifs du §5 |

Les tests de contrat détectent bien une faute injectée (tri en place, `index.freq` modifié, écriture
dans `to_numpy()`), et les tests d'exactitude échouent avec l'ancien calcul `end_time`.

**Modifiés** (uniquement pour retirer une dépendance à un détail de pandas, sans assouplir
d'assertion) : `crossvals/base/test_index_robustness.py` (dtype du message lu sur l'entrée),
`panel/transformers/test_structure.py` (dtypes attendus lus sur l'entrée),
`utils/parse/test_utils.py` et `utils/frequency/test_normalizer.py` (oracles « alias nu de pandas 2 »
→ valeurs d'or ; alias passés à pandas avec `default_position='E'`),
`frequency_aligner/test_convert_to_target.py` et `covariate_materializer/test_robustness.py`
(`stack().dropna()`), `high_frequency_imputer/test_transform_window.py` (`check_freq=False` sur une
sélection de `MultiIndex`), `converter/test_anchor_fraction.py` (ancres internes épinglées en `ns`),
`test_publication_delay_transformer_transform.py` (valeur d'or construite avec `sort=True`),
`tests/support/perturbations.py` (`to_period_index` : alias en chaîne pour `get_period_alias`).

## 7. Différences de sortie pandas 2.3 → 3, justifiées

Comparaison des 185 cas de référence (base pandas 2.3.1 / sklearn 1.7.1 contre état final) :

| Nature | Nombre | Justification | Effet utilisateur |
|---|---:|---|---|
| Valeurs | **0** | — | — |
| Exceptions | **0** | — | — |
| Flottants à ~1 ulp | 3 sorties (interpolations `cubic` / `time` du convertisseur), écart relatif ≤ 5,5e-16 | l'index en `us` change l'échelle des abscisses passées au calcul flottant | aucun (sous la tolérance 1e-12) |
| dtype `object` → `str` | 3 540 colonnes / niveaux | dtype chaîne par défaut de pandas 3 | code utilisateur testant `dtype == object` |
| dtype `datetime64[ns]` → `[us]` | 801 | les jeux d'entrée, construits depuis des chaînes, sont eux-mêmes en `us` ; plus des choix propres à chaque chemin (§8) | `assert_index_equal` / comparaisons de dtypes ; l'alignement par date fonctionne (instants) |
| Avertissements | 2 | `FutureWarning` du `groupby` supprimé ; message de sklearn ne différant que par le chemin du venv | — |

## 8. Anomalies

| Id | Type | Sévérité | Statut | Résumé |
|---|---|---|---|---|
| ANO-UTILS-071 | [CODE] | mineure | corrigée | `anchor_fraction` : longueur de période lue sur `end_time`, fausse de 999 ns sous pandas 3 |
| ANO-FREQ-085 | [CODE] | mineure | corrigée | part calendaire (`scale_features='calendar'`) : même cause, écart ≈ 8e-14 |
| ANO-UTILS-072 | [CODE] | à arbitrer | **ouverte** | unité des dates rendues par `FrequencyConverter`, `convert_position`, `get_period_*` : propre à chaque chemin |
| ANO-DELAYS-045 | [CODE] | à arbitrer | **ouverte** | `compare_and_detect_delays` : dates à trois résolutions ; résultat vide typé autrement (dates `ns`, chaînes `object`) |
| ANO-FREQ-086 | [CODE] | à arbitrer | **ouverte** | HFI : sortie toujours en `us`, provenance d'un panel à la résolution de l'entrée |

Ce sont les seules anomalies ouvertes du registre (les autres « à arbitrer » sont arbitrées ou closes).
Détail, reproductions et tests : `tests/ANOMALIES.md`.

## 9. Doctests, notebooks, documentation, conventions

- **Doctests** (`pytest --doctest-modules tsforecast -q`) : base 18 failed / 311 passed / 22 skipped →
  final **23 failed / 308 passed / 22 skipped** (2 doctests ajoutés, qui passent). Les 5 nouveaux
  échecs ne portent que sur la représentation (`dtype('O')` → `StringDtype`, `[ns]` → `[us]`) :
  `base/transformers.py::PanelTimeSeriesTransformer._validate_index_time_series`,
  `::_validate_multiindex_panel`, `panel/utils.py::extract_time_series_from_multiindex`,
  `utils/frequency/converter.py::FrequencyConverter._extend_index_for_upsampling`,
  `::_reanchor_index_to_target`. Non corrigés ici (N7 / N8, amendement en §12).
- **Notebooks** : 3 notebooks cassés par pandas 3, corrigés sur le **source** seulement (5 lignes ;
  sorties stockées laissées en l'état) : `0 - QB - Utils.ipynb` (`'H'` → `'h'`, `'M'` → `'ME'`),
  `1 - QB - Delays.ipynb` (`fillna(method='ffill')` → `.ffill()`), `test_regularizer.ipynb`
  (`concat(..., sort=True)`). Erreurs de cellules ensuite identiques à la base.
  `tests/integration/frequency/test_notebooks.py` (notebook 5) passe. **14 notebooks étaient déjà en
  erreur sous pandas 2.3** (dérive d'API, sans lien avec pandas) : hors périmètre (P6).
- **mkdocs** : `--strict` échouait déjà (189 avertissements griffe : annotations manquantes surtout
  dans `xy/` et `crossvals/`, indentation de docstrings, 3 renvois croisés) ; ensemble identique (P5).
- **Conventions** : aucune nouvelle violation bloquante hors doctests ; les signalements `history`
  restants (non bloquants) sont antérieurs.
- **README.md** : section « Prérequis » (pandas ≥ 3.0, scikit-learn ≥ 1.9). **CLAUDE.md** : section
  « pandas ≥ 3.0 » (CoW, unité explicite, dtype `str`, alias, `concat`, `groupby` / `stack`),
  arborescence des tests (`perturbations.py`, `integration/input_contract/`), robustesse à la
  résolution — **attention : `CLAUDE.md` est ignoré par git (`.gitignore:187`)**, ces ajouts
  n'apparaîtront pas dans un commit.

## 10. Environnement et incidents

- `pip` absent de `.venv` ; `uv run` / `uv sync` complet échouent (dist-info du projet sous OneDrive).
- La première installation a échoué à mi-chemin : OneDrive pose l'attribut **lecture seule** sur des
  dossiers, et Windows refuse de supprimer un dossier en lecture seule → pandas 2.3.1 à moitié
  désinstallé. Réparation : attribut retiré sur les seuls dossiers pandas / sklearn,
  `uv sync --no-install-project --inexact`, suppression d'un `pandas-2.3.1.dist-info` vide (qui
  faisait renvoyer `None` à `importlib.metadata.version('pandas')`). Vérifié : aucun fichier résiduel,
  métadonnées correctes. Procédure notée dans la mémoire du projet.
- Plusieurs passes de fond interrompues (mémoire, puis mise en veille) : relancées une par une.
- Outils de travail (sorties de référence, comparateurs) dans le scratchpad de la session — à
  recréer si besoin (P2 / P3 le précisent).

## 11. Décisions à arbitrer — vue d'ensemble

| # | Décision | Enjeu principal | Prompt |
|---|---|---|---|
| D1 | Politique d'**unité des dates en sortie** | cohérence des sorties, aller-retours exacts, golden master R0 | P2 (+ P3) |
| D2 | **dtype des chaînes** en sortie (`str` / `object`), résultat vide des délais, pyarrow | sémantique `None` / `NaN`, égalités de dtypes, performance | P4 |
| D3 | Doctests touchés par pandas 3 | à figer **après** D1 / D2 | amendement N7 / N8 |
| D4 | `mkdocs build --strict` (préexistant) | documentation API, CI | P5 |
| D5 | 14 notebooks en dérive d'API (préexistant) | exemples utilisateur exécutables | P6 |
| D6 | `CLAUDE.md` ignoré par git | les conventions pandas 3 ne sont pas versionnées | aucune (retirer `CLAUDE.md` de `.gitignore` ou non) |

**Ordre recommandé** : P2 → P3 → P4 → N7 (amendé) → N8 (amendé) → R0. P5 et P6 sont indépendants.
D1 et D2 doivent précéder N7 / N8 (les doctests affichent unités et dtypes) et R0 (le golden master
figerait sinon des unités et des dtypes appelés à changer).

---

## 12. Prompts de résolution et d'arbitrage

Choix du modèle, de l'effort et du plan mode selon les critères de `tests_and_refactoring_prompts.md`
(§5) : **Opus** dès qu'un raisonnement sur dates, périodes, fenêtres ou la symétrie
`fit` / `transform` conditionne la justesse ; **plan mode** quand le prompt prend une décision de
structure consommée par les suivants ; **xhigh** au-delà de 1 500 lignes sous test ou quand l'état
partagé entre `fit` et `transform` est touché.

| # | Objet | Modèle | Plan | Effort | Dépend de |
|---|---|---|---|---|---|
| **P2** | Politique de résolution des dates en sortie : arbitrage et socle `utils` (ANO-UTILS-072) | Opus | **Oui** | xhigh | P1 |
| **P3** | Application de la politique à `delays` et `HighFrequencyImputer` (ANO-DELAYS-045, ANO-FREQ-086) | Opus | Non | xhigh | P2 |
| **P4** | dtype des chaînes en sortie, résultat vide des délais, pyarrow (arbitrage) | Opus | **Oui** | high | P3 |
| N7 / N8 | Doctests (amendement : 5 doctests pandas 3) | Sonnet | Non | medium / high | P4 |
| **P5** | `mkdocs build --strict` vert (189 avertissements préexistants) | Sonnet | Non | medium | — |
| **P6** | 14 notebooks en dérive d'API (optionnel) | Sonnet | **Oui** | medium | — |

### Prompt P2 — Politique de résolution des dates en sortie : arbitrage et socle `utils`

**Modèle : Opus · Plan mode : Oui · Effort : xhigh · Dépendances : P1**

**Enjeux.** pandas 3 infère l'unité des dates (`us` pour des chaînes) et chaque chemin du paquet
rend aujourd'hui sa propre unité, sans écart de valeur : agrégation d'une série et descente d'un
panel suivent l'entrée ; montée d'un panel et `convert_position` rendent au moins `us` ;
`interpolate_to_higher_frequency` toujours `us` ; `get_period_*` `us` (calendaire) ou `ns`
(infra-journalier) ; validation, délais (`Shift` / `Mask` / `PublicationDelayTransformer`) et
`PanelwiseTransformer` suivent l'entrée ; HFI toujours `us`. Conséquences : un aller-retour
(`transform` → `inverse_transform`, validation → restitution) peut ne pas rendre le dtype d'entrée ;
`assert_frame_equal`, les comparaisons de dtypes et les empreintes de cache (`XYPipeline(memory=…)`,
qui hachent `repr(dtype)`) distinguent des sorties égales ; deux sorties d'unités différentes
s'alignent par instant mais leur concaténation monte à l'unité la plus fine. Toute décision prise
après R0 (golden master) et N7 / N8 (doctests) obligerait à les régénérer.

**Options.** (1) Statu quo documenté — aucun code, incohérence assumée. (2) **Unité de l'entrée**
— chaque sortie publique rend ses dates dans l'unité de l'entrée, ou plus fine si une date produite
n'y est pas représentable (grille infra-seconde) ; cohérent avec les composants qui le font déjà,
aller-retours exacts ; coût : un utilitaire appliqué à la sortie de 5 à 8 chemins. (3) **Unité fixe
documentée** — `ns` (historique, plage 1677–2262) ou `us` (défaut de pandas 3, plage étendue) ;
simple à énoncer, mais change l'unité de l'entrée de l'utilisateur.
**Recommandation : (2).**

```text
Contexte : dépôt ts-forecast (pandas 3.0.6, scikit-learn 1.9.1). Lis CLAUDE.md (section « pandas ≥
3.0 » et règles de rédaction), pandas3_migration_report.md (§7, §8, §11) et, dans tests/ANOMALIES.md,
ANO-UTILS-072, ANO-DELAYS-045, ANO-FREQ-086. Commandes : `.venv/Scripts/python.exe -m pytest …`
(uv run échoue sur cette machine).

DÉCISION DE L'AUTEUR (à compléter avant lancement ; si elle est vide, la demander en plan mode avec
AskUserQuestion, options (1) statu quo, (2) unité de l'entrée, (3) unité fixe ns / us, en rappelant
les enjeux) : (2).

Objet : arrêter la politique de résolution des dates en sortie et l'appliquer au socle `utils`.
1. Plan (plan mode) : inventaire exhaustif, par composant public, de l'unité des dates rendues
   (index, niveaux, colonnes) selon l'unité d'entrée `s`/`ms`/`us`/`ns`, mesuré avec
   `tests/support/perturbations.py::with_datetime_resolution` et `datetime_units` (tableau
   composant × chemin × unité rendue). Pour la politique retenue, liste des chemins à modifier et
   règle exacte : unité de quelle entrée quand il y en a plusieurs (deux jeux à comparer, date de
   téléchargement scalaire, `test_indices`) ; cas d'une date produite non représentable dans
   l'unité cible (refus, unité plus fine, arrondi : jamais d'arrondi silencieux).
2. Socle : un utilitaire unique dans `tsforecast/utils/time/` (nom et signature au plan), avec
   docstring Google et doctest, testé dans `tests/unit/utils/time/` ; appliqué aux sorties de
   `FrequencyConverter` (`convert_frequency`, `aggregate_to_lower_frequency`,
   `interpolate_to_higher_frequency`), `convert_position` et `get_period_start` / `get_period_end` /
   `get_period_boundaries`. Les calculs internes restent exacts (nanosecondes entières où il le
   faut, cf. ANO-UTILS-071) : seule l'unité rendue change.
3. Garde-fous : aucune valeur ni instant ne change. Avant toute modification, sorties de référence
   de ces composants sur toutes les perturbations (script jetable dans le scratchpad, sérialisation
   neutre : instants en entiers de ns, valeurs float64, dtypes en texte) ; après : instants et
   valeurs identiques au bit près, seules les unités diffèrent, conformément à la politique.
4. Tests : remplacer les unités épinglées des classes `TestDatetimeResolution` de
   `tests/unit/utils/frequency/converter/test_datetime_resolution.py` et
   `tests/unit/utils/time/test_utils.py` par l'assertion de la politique (une ligne explicite par
   chemin) ; ajouter un test d'aller-retour d'unité là où un composant a un inverse.
5. ANO-UTILS-072 : statut → corrigée (ou « arbitrée : statu quo »), ligne Correctif. CLAUDE.md :
   la règle d'unité de sortie dans la section pandas ≥ 3.0. Ne pas toucher aux doctests (N7 / N8).
Critères : suite complète verte (`-W error::FutureWarning -W error::DeprecationWarning`), aucun
écart de valeur, couverture non dégradée, `scripts/check_conventions.py --rule all` sans nouvelle
violation hors doctests. Rapport en français : tableau avant / après des unités, chemins modifiés,
décision appliquée, points à reporter dans P3 (délais, HFI).
```

### Prompt P3 — Application de la politique à `delays` et `HighFrequencyImputer`

**Modèle : Opus · Plan mode : Non · Effort : xhigh · Dépendances : P2**

**Enjeux.** Les deux composants les plus exposés : `compare_and_detect_delays` rend
`observation_date` à la résolution de l'entrée mais `period_start` / `period_end` /
`download_date` en `us`, et son **résultat vide** a des dates en `ns` — une concaténation de
résultats (calendrier de publications) mélange les unités. `HighFrequencyImputer` construit sa
grille (sortie, rejeu en `us`) mais rend la provenance d'un panel à la résolution de l'entrée : sortie
et provenance d'un même ajustement n'ont pas la même unité. Risque spécifique : l'imputeur garde un
état entre `fit` et `transform` (plan, grilles, instantané pour `inverse_transform`) — changer
l'unité d'un seul côté crée une divergence silencieuse (`Index.equals` compare les instants sous
pandas 3, mais `get_indexer`, les dtypes d'index et les empreintes de cache peuvent diverger).

```text
Contexte : dépôt ts-forecast (pandas 3.0.6). Lis CLAUDE.md (section pandas ≥ 3.0, désormais complétée
de la règle d'unité de sortie par P2), pandas3_migration_report.md, le rapport de P2 et, dans
tests/ANOMALIES.md, ANO-DELAYS-045 et ANO-FREQ-086. Commandes : `.venv/Scripts/python.exe -m pytest …`.

Objet : appliquer la politique d'unité de sortie arrêtée en P2 (utilitaire de
tsforecast/utils/time/) à :
- tsforecast/delays/data_manager.py::compare_and_detect_delays — les quatre colonnes de dates
  (`observation_date`, `period_start`, `period_end`, `download_date`) et le résultat vide
  (`_empty_publication_delays`, aujourd'hui typé `datetime64[ns]`), qui doit avoir exactement les
  dtypes d'un résultat non vide pour la même entrée ; vérifier calculate_applicable_delay et
  DelayDetectionReport / to_frame ;
- tsforecast/frequency/high_frequency_imputer.py::HighFrequencyImputer — sortie empilée, rejeu
  (`transform`), `imputation_provenance_`, `inverse_transform` (doit rendre l'index d'entrée, unité
  comprise), passe X / y (`fit_transform(X, y)`), état ajusté (plan, grilles, instantanés) :
  une seule unité cohérente de `fit` à `transform`.
Méthode : (1) sorties de référence avant modification (script jetable dans le scratchpad :
compare_and_detect_delays sur les jeux réalistes et toutes les unités d'entrée, résultat vide ;
HFI sur la matrice de tests/integration/frequency/realistic_runs.py et PANEL-X, valeurs ET matrices
de provenance) ; (2) modification ; (3) comparaison : instants et valeurs identiques au bit près,
seules les unités changent comme le prévoit la politique. Un écart de valeur est une anomalie :
s'arrêter, consigner, ne pas ajuster de valeur d'or.
Tests : remplacer les unités épinglées de
tests/unit/delays/data_manager/test_datetime_resolution.py et
tests/unit/frequency/high_frequency_imputer/test_datetime_resolution.py par l'assertion de la
politique ; ajouter : résultat vide de compare_and_detect_delays aux mêmes dtypes qu'un non vide ;
sortie et provenance de l'imputeur à la même unité ; aller-retour `inverse_transform` à l'unité
d'entrée ; XYPipeline(memory=…) : même empreinte pour deux unités d'entrée si la politique les
rend identiques en sortie (sinon, documenter). ANO-DELAYS-045 (partie dates) et ANO-FREQ-086 :
statut → corrigée, ligne Correctif ; la partie « chaînes `object` du résultat vide » d'ANO-DELAYS-045
reste à P4 (le noter dans l'entrée).
Critères : suite complète verte avec `-W error::FutureWarning -W error::DeprecationWarning`,
couverture non dégradée (high_frequency_imputer.py, data_manager.py), durée de la suite ≤ +5 %,
`check_conventions.py --rule all` sans nouvelle violation hors doctests. Rapport en français.
```

### Prompt P4 — dtype des chaînes en sortie (`str` / `object`), résultat vide des délais, pyarrow

**Modèle : Opus · Plan mode : Oui · Effort : high · Dépendances : P3**

**Enjeux.** pandas 3 infère le dtype `str` (manquant = `NaN`) pour les chaînes ; 34 `dtype=object`
explicites (11 fichiers) gardent des sorties en `object`. Résultat : des tables mixtes —
`compare_and_detect_delays` rend `reference_point` / `unit` en `str` mais `frequency` en `object`, et
son résultat vide tout en `object` ; l'index `column` de `delays/report.py` est `object`.
Certains `object` sont **nécessaires** : la matrice de provenance contient des `ProvenanceType`
(énumération `str`) et `None` — en dtype `str`, les membres deviennent de simples chaînes
(`isinstance(val, ProvenanceType)` échoue, `provenance.py`) et `None` devient `NaN` (tout test
`is None` change de sens en silence) ; idem pour les origines de cellule du matérialiseur, les clés
d'agrégation (tuples) et `feature_names_in_` (convention sklearn : tableau `object`). pyarrow : le
dtype `str` utilise aujourd'hui le stockage Python (plus lent) ; un utilisateur qui a pyarrow obtient
le stockage pyarrow (même sémantique, chaînes à substituts Unicode refusées).

**Options.** (1) Statu quo documenté. (2) **Harmoniser en `str`** toute colonne ou tout index de
chaînes pures rendu à l'utilisateur (dont `frequency` des délais, l'index `column` du rapport, le
résultat vide), garder `object` là où les valeurs sont hétérogènes (énumérations, tuples, `None`
porteur de sens) avec un commentaire qui le justifie. (3) Tout en `object` explicite (à
contre-courant de pandas 3). pyarrow : (a) non requis (statu quo), (b) extra optionnel
(`ts-forecast[pyarrow]`) avec la suite exécutée sous les deux stockages en CI, (c) requis.
**Recommandation : (2) + (b).**

```text
Contexte : dépôt ts-forecast (pandas 3.0.6). Lis CLAUDE.md (section pandas ≥ 3.0),
pandas3_migration_report.md (§4 famille b, §7, §11) et ANO-DELAYS-045 dans tests/ANOMALIES.md.

DÉCISIONS DE L'AUTEUR (à compléter ; sinon les demander en plan mode avec AskUserQuestion, en
rappelant les enjeux) : dtype des chaînes <(1) statu quo / (2) harmoniser en str / (3) tout object> : choix retenu : (2) : harmoniser en str ;
pyarrow <(a) non requis / (b) extra optionnel / (c) requis> : choix retenu : (b) extra optionnel.

1. Plan (plan mode) : tableau des 34 sites `dtype=object` (git grep "dtype\s*=\s*(object|'object')"
   dans tsforecast/) et de toute sortie publique contenant des chaînes : composant, objet rendu
   (colonne, index, niveau, attribut ajusté), valeurs possibles (chaînes pures / énumérations /
   tuples / None significatif), dtype actuel, dtype cible selon la décision, risque (`is None`,
   `isinstance`, comparaisons d'égalité, `.str`, sérialisation `to_dict()` / JSON des rapports).
   Chercher en particulier les tests `is None` / `isinstance(..., ProvenanceType)` /
   `isinstance(..., CellOrigin)` sur des valeurs lues dans ces colonnes.
2. Mise en œuvre de la décision, sans changer une valeur : sorties de référence avant / après
   (script jetable : délais réalistes, rapports to_dict(), HFI provenance, matérialiseur) — égalité
   des valeurs (NaN et None comptés comme « manquant » et rapportés à part), seuls les dtypes
   changent conformément au plan. Résultat vide de compare_and_detect_delays : mêmes dtypes qu'un
   résultat non vide (ANO-DELAYS-045, partie chaînes → statut corrigée).
3. pyarrow selon la décision : extra dans pyproject.toml (`uv lock`), job CI supplémentaire
   (.github/workflows/tests.yml) installant pyarrow, et mention au README ; vérifier la suite
   complète avec pyarrow installé dans un venv séparé du scratchpad (`uv venv` + `uv pip install`),
   sans toucher à .venv.
4. Tests : un test par sortie publique harmonisée (dtype attendu, explicite) ; libellés à caractères
   spéciaux via tests/integration/input_contract/test_string_labels.py (étendre si besoin).
5. CLAUDE.md (section pandas ≥ 3.0) : règle « dtype des chaînes en sortie ».
Critères : suite complète verte avec `-W error::FutureWarning -W error::DeprecationWarning`, aucune
valeur changée, couverture non dégradée, `check_conventions.py --rule all` sans nouvelle violation
hors doctests. Rapport en français : tableau des sites, décisions appliquées, effet utilisateur.
```

### Amendement de N7 / N8 — doctests touchés par pandas 3

**N7 : Sonnet · Plan mode : Non · Effort : medium · N8 : Sonnet · Plan mode : Non · Effort : high ·
Dépendances : P2, P3, P4 (les doctests affichent unités et dtypes)**

**Enjeux.** 23 doctests en échec : 18 préexistants dans `frequency` (N8) et 5 que pandas 3 a cassés
par la seule représentation. Les corriger avant P2-P4 obligerait à les reprendre ; les écrire de façon
indépendante de la représentation (`.tolist()`, `str(...)` de valeurs, pas de `repr` d'index avec
`dtype=`) les rend robustes à une future évolution de pandas.

Texte à ajouter à la fin des prompts N7 et N8 :

```text
Complément pandas 3 : sous pandas 3.0.6, 5 doctests échouent par représentation seulement —
base/transformers.py::PanelTimeSeriesTransformer._validate_index_time_series et
::_validate_multiindex_panel (dtype('O') → StringDtype, [ns] → [us]),
panel/utils.py::extract_time_series_from_multiindex,
utils/frequency/converter.py::FrequencyConverter._extend_index_for_upsampling et
::_reanchor_index_to_target (DatetimeIndex(..., dtype='datetime64[ns]')). Les réécrire pour qu'ils
n'affichent ni dtype ni unité (`.strftime('%Y-%m-%d').tolist()`, `str(x.dtype)` seulement si l'unité
est fixée par la politique de P2, `.tolist()`), en conservant ce qu'ils démontrent. Même règle pour
tout exemple corrigé : aucune sortie dépendant de la résolution des dates ou du dtype chaîne, sauf
quand c'est l'objet de l'exemple.
```

### Prompt P5 — `mkdocs build --strict` vert

**Modèle : Sonnet · Plan mode : Non · Effort : medium · Dépendances : aucune (après N7 / N8 si les
docstrings y changent)**

**Enjeux.** Le build strict échoue depuis avant la migration : 189 avertissements griffe — surtout
« No type or annotation for parameter / returned value » (`xy/pipeline.py` 43, `crossvals/panel.py`
26, `crossvals/base.py` 26, `xy/transformers.py` 25, `crossvals/time_series.py` 17, puis
`base/transformers.py`, `utils/frequency/utils.py`, `delays/transformers.py`…), « Confusing
indentation for continuation line » (`utils/frequency/detector.py`, `frequency/covariate_materializer.py`)
et 3 renvois croisés `mkdocs_autorefs` (listes `[0.0, 0.6931, 1.3863]` des docstrings de `xy` lues
comme des liens). Tant qu'il échoue, la CI ne peut pas garder la documentation. Options : (1)
**annoter les signatures publiques et corriger l'indentation des docstrings** (documentation API
complète, recommandé) ; (2) régler les options de mkdocstrings / griffe pour ne plus signaler les
annotations manquantes (à vérifier dans leur documentation) ; (3) renoncer à `--strict`.

```text
Contexte : dépôt ts-forecast. Lis CLAUDE.md (docstrings Google, règles de rédaction, type hints
systématiques) et pandas3_migration_report.md §9. Commande :
`.venv/Scripts/python.exe -m mkdocs build --strict -d <dossier du scratchpad>` (ne pas écrire site/).
DÉCISION DE L'AUTEUR : <(1) annoter / (2) configurer griffe / (3) sans --strict> (défaut : (1)).
Objet : 0 avertissement en mode strict. Avec (1) : ajouter les annotations de type manquantes aux
paramètres et retours des API publiques signalées (xy, crossvals d'abord), corriger l'indentation
des lignes de continuation des docstrings signalées, échapper les listes lues comme des renvois
croisés (bloc de code ou backquotes). Ne change aucun comportement : annotations et docstrings
seulement ; vérifier `git diff` (pas d'autre modification), suite `-m "not slow"` inchangée,
doctests des fichiers touchés inchangés. Ajouter le build strict à la CI
(.github/workflows/tests.yml, job séparé, groupe `docs`). Rapport : avertissements par catégorie
avant / après, fichiers touchés.
```

### Prompt P6 — Notebooks en dérive d'API (optionnel)

**Modèle : Sonnet · Plan mode : Oui · Effort : medium · Dépendances : P2-P4 recommandés (les sorties
affichées en dépendent)**

**Enjeux.** 14 notebooks échouaient déjà sous pandas 2.3 (imports disparus, renommages délibérés,
méthodes privées modifiées) ; seul le notebook 5 est gardé par un test. Ce sont des exemples
utilisateur : périmés, ils induisent en erreur. Décision par notebook : réparer, archiver
(`notebooks/archive/`, hors exécution) ou supprimer.

| Notebook | Cellules en erreur | Première erreur |
|---|---:|---|
| `0 - QB - Utils` | 4 | `ImportError: convert_frequency` (n'est plus exporté) |
| `1 - QB - Delays` | 12 | `NameError: ReleaseDataManager` |
| `11 - QB - Delays` | 27 | `KeyError: 'release_delay'` (renommé `delay`) |
| `Test Delays transformers` | 16 | `ModuleNotFoundError: tsforecast.delays.auxiliary_transformers` |
| `utils/duration_converter` | 1 | arrondi `'round'` non supporté |
| `utils/frequency_aligner` | 14 | `_aggregate_to_target(is_panel=…)` (privé, signature changée) |
| `utils/frequency_converter` | 3 | assertion du notebook sur `target_position=None` |
| `utils/frequency_detector` | 2 | assertion du notebook ; `time_col` inexistant (cellule de démonstration d'erreur) |
| `utils/frequency_normalizer` | 3 | assertion `('SM', 'SM', 'SME')` |
| `utils/period_position_converter` | 3 | fréquence `'foo'` (démonstration d'erreur) ; `'Q'` plus grossière que les données |
| `utils/period_position_normalizer` | 18 | attribut privé `_position_aware_frequencies` disparu |
| `utils/provenance` | 3 | `mark_model_imputed(trained_on_imputed=…)` |
| `utils/regularizer` | 2 | série sans fréquence constante ; `IndexRegularizer(min_observations=…)` |
| `utils/target_frequency_validator` | 1 | aucune fréquence détectée pour une entité |

```text
Contexte : dépôt ts-forecast (pandas 3). Lis CLAUDE.md (section Notebooks : cas d'usage variés,
impact des paramètres, données réelles), pandas3_migration_report.md §9 et §12 (tableau P6).
Commande d'exécution sans réécriture : nbclient en mémoire, répertoire de travail = racine
(comme tests/integration/frequency/test_notebooks.py), MPLBACKEND=Agg.
1. Plan (plan mode) : pour chacun des 14 notebooks, cellules en erreur, cause (API publique
   renommée, API privée, démonstration d'erreur volontaire non protégée, assertion périmée), et
   proposition : réparer (sur l'API publique uniquement), archiver ou supprimer ; à valider par
   l'auteur (AskUserQuestion) avant exécution.
2. Réparation : sources des cellules seulement, API publique actuelle ; une cellule qui montre
   volontairement une erreur l'encadre (try / except avec affichage du message) ; puis exécution
   complète et sauvegarde des sorties (le notebook est rejoué de bout en bout).
3. Tests : ajouter les notebooks réparés au test paramétré d'exécution (marqueur slow), sur le
   modèle de tests/integration/frequency/test_notebooks.py.
Critères : notebooks retenus exécutés sans erreur ; suite inchangée ; rapport : décision par
notebook, cellules modifiées.
```

### Hors prompt — `CLAUDE.md` ignoré par git (D6)

`CLAUDE.md` figure à la ligne 187 de `.gitignore` : les conventions ajoutées par P1 (et celles de P2
/ P4) ne sont pas versionnées. Décision : retirer la ligne de `.gitignore` et committer `CLAUDE.md`
avec la migration, ou garder le fichier local (les conventions vivent alors seulement sur ce poste).
