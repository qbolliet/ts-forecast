"""``RollingAggregator`` after ``HighFrequencyImputer`` inside an ``XYPipeline``.

Chains ``tsforecast.imputation.HighFrequencyImputer`` (quarterly target
imputed onto a monthly grid), ``tsforecast.xy.RollingAggregator``
(``window='Q'``), ``tsforecast.xy.DropNaNTarget`` and a NaN-tolerant
estimator, on a month-start series (notebook 2 dataset, quarterly GDP dated
at quarter start) and on a month-end panel (``PANEL-X`` reference dataset,
quarterly ``q1`` dated at quarter end).

Families of scenarios: at the training pass, the rolling quarter of the
imputed target equals the observed quarterly total on every anchor row, with
a future window on the month-start grid and a past window on the month-end
grid; at the prediction pass, the test target comes back as observed and the
predictions on the anchors are on the quarterly scale; ``cross_validate``
with ``TSOutOfSampleSplit`` scores every fold.
"""
# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

# Jeux de données partagés
from tests.support.datasets import build_mixed_frequency_timeseries, build_panel_reference

# Composants enchaînés
from tsforecast.crossvals import TSOutOfSampleSplit
from tsforecast.imputation import HighFrequencyImputer
from tsforecast.xy import DropNaNTarget, RollingAggregator, XYPipeline


# =============================================================================
# Données et pipelines
# =============================================================================
@pytest.fixture(scope='module')
def month_start_series():
    """``(X, y)``: monthly covariates and quarterly GDP dated at quarter start (``MS`` grid)."""
    data = build_mixed_frequency_timeseries()
    X = data[['production_industrielle', 'inflation_ipc', 'taux_chomage']]
    return X, data['pib_trimestriel']


@pytest.fixture(scope='module')
def month_end_panel():
    """``(X, y)``: monthly ``m1`` and quarterly ``q1`` dated at quarter end (``ME`` grid), three entities."""
    data = build_panel_reference()
    return data[['m1']], data['q1']


def _imputer() -> HighFrequencyImputer:
    """Imputer bringing the quarterly target onto the monthly grid."""
    return HighFrequencyImputer(
        target_frequency='M', estimator=LinearRegression(), keep_lower_frequencies=False
    )


def _training_pass(X: pd.DataFrame, y: pd.Series):
    """Training pass of the imputer followed by the rolling aggregation."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return XYPipeline([('imputer', _imputer()), ('roll', RollingAggregator(window='Q'))]).fit_transform(X, y)


def _forecast_pipeline() -> XYPipeline:
    """Complete chain ending with a NaN-tolerant estimator."""
    return XYPipeline([
        ('imputer', _imputer()),
        ('roll', RollingAggregator(window='Q')),
        ('drop', DropNaNTarget()),
        ('model', HistGradientBoostingRegressor(max_iter=50, random_state=0)),
    ])


# =============================================================================
# Passe d'entraînement
# =============================================================================
class TestTrainingPass:
    """The rolling quarter of the imputed target rebuilds the observed totals."""

    def test_month_start_anchors_get_their_quarter(self, month_start_series):
        X, y = month_start_series
        _, y_t = _training_pass(X, y)
        observed = y.dropna()
        rebuilt = y_t.reindex(observed.index)
        compared = rebuilt.notna()
        assert compared.sum() >= len(observed) // 2
        np.testing.assert_allclose(rebuilt[compared], observed[compared], rtol=1e-9)

    def test_month_end_anchors_get_their_quarter(self, month_end_panel):
        X, y = month_end_panel
        _, y_t = _training_pass(X, y)
        observed = y.dropna()
        rebuilt = y_t.reindex(observed.index)
        compared = rebuilt.notna()
        assert compared.sum() >= len(observed) // 2
        np.testing.assert_allclose(rebuilt[compared], observed[compared], rtol=1e-9)

    def test_every_month_carries_a_quarterly_quantity(self, month_end_panel):
        X, y = month_end_panel
        _, y_t = _training_pass(X, y)
        # Trois fois plus de lignes d'entraînement que d'ancres trimestrielles, à un bord près
        assert y_t.notna().sum() > 2 * y.notna().sum()


# =============================================================================
# Passe de prédiction
# =============================================================================
class TestPredictionPass:
    """Observed test target, predictions on the quarterly scale."""

    @pytest.fixture(scope='class')
    def fitted(self, month_start_series):
        X, y = month_start_series
        train, test = X.index < '2023-01-01', X.index >= '2023-01-01'
        pipe = _forecast_pipeline()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipe.fit(X[train], y[train])
        return pipe, X[test], y[test]

    def test_test_target_comes_back_as_observed(self, fitted):
        pipe, X_test, y_test = fitted
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, y_t = pipe[:-1].transform(X_test, y_test)
        pd.testing.assert_series_equal(y_t, y_test.dropna(), check_freq=False)

    def test_predictions_on_anchors_are_quarterly(self, fitted):
        pipe, X_test, y_test = fitted
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            predictions = pd.Series(pipe.predict(X_test), index=X_test.index)
        observed = y_test.dropna()
        ratio = predictions.reindex(observed.index).median() / observed.median()
        assert 0.8 < ratio < 1.2

    def test_score_runs_on_the_observed_target(self, fitted):
        pipe, X_test, y_test = fitted
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            score = pipe.score(X_test, y_test)
        assert np.isfinite(score)


# =============================================================================
# Validation croisée
# =============================================================================
class TestCrossValidation:
    """Every fold of a quarter-aligned out-of-sample split is scored."""

    def test_cross_validate_scores_every_fold(self, month_start_series):
        # Données arrêtées à la fin d'un trimestre : plis de 12 mois bornés par des trimestres
        X, y = (data.loc[:'2024-06-01'] for data in month_start_series)
        n_rows = len(X)
        test_starts = [n_rows - 36, n_rows - 24, n_rows - 12]
        assert all(X.index[start].month in (1, 4, 7, 10) for start in test_starts)
        splitter = TSOutOfSampleSplit(n_splits=3, test_size=12)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = cross_validate(_forecast_pipeline(), X, y, cv=splitter, error_score='raise')
        assert len(result['test_score']) == 3
        assert np.isfinite(result['test_score']).all()
