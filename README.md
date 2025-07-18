# ts-forecast
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
