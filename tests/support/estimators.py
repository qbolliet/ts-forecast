"""Estimateurs sklearn factices, partagés par les tests de ``tsforecast``.

Extraits de ``tests/unit/frequency/test_high_frequency_imputer.py``
(anciennement ``_SpyEstimator`` / ``_FailingEstimator``, privés à ce module),
renommés sans préfixe pour un usage transversal.
"""
# Modules de base
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class SpyEstimator(BaseEstimator, RegressorMixin):
    """Estimateur espion, tolérant les NaN et prédisant une constante.

    Il retient ``fit_X_``, ``fit_y_`` et la liste ``predict_X_`` des trames de
    prédiction, ce qui rend mesurable l'invariant central (jamais de
    dégradation fit → predict) et l'échelle des données transmises. Le
    compteur de classe ``n_fits`` mesure la règle « un seul ajustement par
    (étape, variable) ».
    """

    n_fits = 0

    def __init__(self, constant: float = 1.0):
        self.constant = constant

    def fit(self, X, y):
        """Retient le jeu d'entraînement et la moyenne de la cible."""
        type(self)._record_fit()
        self.fit_X_ = X.copy()
        self.fit_y_ = y.copy()
        self.predict_X_ = []
        values = np.asarray(y, dtype=float)
        finite = values[~np.isnan(values)]
        self.mean_ = float(finite.mean()) if finite.size else 0.0
        return self

    def predict(self, X):
        """Retient la trame de prédiction et rend la moyenne apprise."""
        if not hasattr(self, 'predict_X_'):
            self.predict_X_ = []
        self.predict_X_.append(X.copy())
        return np.full(len(X), self.mean_)

    @classmethod
    def _record_fit(cls):
        """Incrémente le compteur d'ajustements de la classe."""
        SpyEstimator.n_fits += 1


class FailingEstimator(BaseEstimator, RegressorMixin):
    """Estimateur dont l'ajustement échoue toujours."""

    def fit(self, X, y):
        """Lève systématiquement, pour éprouver le repli d'interpolation."""
        raise RuntimeError('deliberate fit failure')

    def predict(self, X):
        """Jamais atteint : l'ajustement a déjà échoué."""
        raise RuntimeError('deliberate predict failure')
