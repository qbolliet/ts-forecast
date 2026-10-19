# ts-forecast

[![tests](https://github.com/qbolliet/ts-forecast/actions/workflows/tests.yml/badge.svg)](https://github.com/qbolliet/ts-forecast/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/qbolliet/ts-forecast/branch/main/graph/badge.svg)](https://codecov.io/gh/qbolliet/ts-forecast)
[![utils](https://codecov.io/gh/qbolliet/ts-forecast/branch/main/graph/badge.svg?component=utils&label=utils)](https://codecov.io/gh/qbolliet/ts-forecast/components?components%5B0%5D=utils)
[![delays](https://codecov.io/gh/qbolliet/ts-forecast/branch/main/graph/badge.svg?component=delays&label=delays)](https://codecov.io/gh/qbolliet/ts-forecast/components?components%5B0%5D=delays)
[![frequency](https://codecov.io/gh/qbolliet/ts-forecast/branch/main/graph/badge.svg?component=frequency&label=frequency)](https://codecov.io/gh/qbolliet/ts-forecast/components?components%5B0%5D=frequency)
[![base](https://codecov.io/gh/qbolliet/ts-forecast/branch/main/graph/badge.svg?component=base&label=base)](https://codecov.io/gh/qbolliet/ts-forecast/components?components%5B0%5D=base)
[![crossvals](https://codecov.io/gh/qbolliet/ts-forecast/branch/main/graph/badge.svg?component=crossvals&label=crossvals)](https://codecov.io/gh/qbolliet/ts-forecast/components?components%5B0%5D=crossvals)
[![panel](https://codecov.io/gh/qbolliet/ts-forecast/branch/main/graph/badge.svg?component=panel&label=panel)](https://codecov.io/gh/qbolliet/ts-forecast/components?components%5B0%5D=panel)
[![xy](https://codecov.io/gh/qbolliet/ts-forecast/branch/main/graph/badge.svg?component=xy&label=xy)](https://codecov.io/gh/qbolliet/ts-forecast/components?components%5B0%5D=xy)

Tools for pseudo-real time forecasting on time series and panel data

# Objectifs

Ce package poursuit les objectifs suivants :
- Fournir des crossvals in ou out of sample pour des time series et des données de panel
- Appliquer et inverser des délais de publication sur un jeu de données à partir d'un dictionnaire (optionnel) et d'une date / d'un numéro (par exemple de jours) et d'une fréquence (peut être inférer du y). On fera l'hypothèse que la date en indice renvoie toujours à la première date de la période
- Traiter des jeux de données de fréquences différentes en imputant les valeurs des variables de basses fréquences à partir des données à haute fréquence ou en agrégeant à une fréquence plus faible
- Utiliser des régresseurs et des classifieurs sklearn ou cuda
- Calculer des intervalles de confiance par bootstrap de manière efficace
- Permettre l'utilisation de méthodes d'interprétabilité
- L'agrégation sera traitée dans un package différent

## Test

Les tests utilisent `pytest` ; la couverture (lignes et branches) est mesurée par `pytest-cov`
et publiée par Codecov (badges ci-dessus : global, puis un badge par paquet). Seuil visé :
90 % par module.

### Commandes

```bash
python tests/run_tests.py --mode all          # tous les tests
python tests/run_tests.py --mode unit         # tests unitaires (tests/unit/ et tests/support/)
python tests/run_tests.py --mode integration  # tests d'intégration (tests/integration/)
python tests/run_tests.py --mode fast         # tout sauf les tests marqués slow
python tests/run_tests.py --mode coverage     # suite complète + rapports + contrôle du seuil de 90 %
python tests/run_tests.py --path tests/unit/crossvals/base/   # cible libre (fichier, dossier, node id)
```

Le mode `coverage` produit `coverage.xml`, `coverage.json` et `htmlcov/`, puis échoue en listant
les modules de `tsforecast/{utils,delays,frequency,base,crossvals,panel,xy}` sous 90 %.

### Arborescence

```
tests/
├── unit/         # miroir exact de tsforecast/ : un composant, petites données à valeurs d'or
├── integration/  # plusieurs composants publics enchaînés, ou jeux de données réalistes
├── support/      # jeux de données, fixtures, perturbations et doublures partagés
├── ANOMALIES.md  # registre des anomalies détectées par la campagne de tests
└── run_tests.py  # lanceur
```

Les anomalies de comportement du code constatées par les tests sont consignées dans
[`tests/ANOMALIES.md`](tests/ANOMALIES.md).

### Intégration continue

`.github/workflows/tests.yml` vérifie que les tests (hors `slow`) passent sur chaque pull
request vers `main`. Sur chaque push sur `main`, il mesure la couverture, l'envoie à Codecov
(secret `CODECOV_TOKEN`) et lance les tests `slow` dans un job séparé. La configuration Codecov
est dans `codecov.yml`.
