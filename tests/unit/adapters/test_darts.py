"""Unit tests of ``tsforecast.adapters.darts.DartsAdapter``.

Runs the unit rules of the adapter contract on row-wise models
(``LinearRegressionModel``, ``RandomForestModel``) and extrapolating models
(``NaiveSeasonal``, lagged ``LinearRegressionModel``), then targeted scenarios:
wrapped model never modified, mode deduced from the model configuration and
refused objects, row-wise predictions equal to a pooled scikit-learn regression
(panels without cross-entity covariates, integer and two-level entities, training
holes, in-sample rows, unknown entities, gap and batch independence, several
targets), extrapolation through the gap (local and global models, covariates
missing on the gap, rows before the end of training), and the darts parameters
exposed to sklearn.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('darts')

from darts.models import (  # noqa: E402
    ARIMA,
    ExponentialSmoothing,
    LinearRegressionModel,
    NaiveDrift,
    NaiveSeasonal,
    RandomForestModel,
    SKLearnClassifierModel,
    SKLearnModel,
)
from sklearn.base import clone, is_regressor  # noqa: E402
from sklearn.exceptions import NotFittedError  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402

from tests.support.adapter_cases import (  # noqa: E402
    darts_forest_case,
    darts_lagged_case,
    darts_linear_case,
    darts_seasonal_case,
)
from tests.support.adapter_contract import AdapterContractTests, split_by_date  # noqa: E402
from tests.support.datasets import build_adapter_panel  # noqa: E402
from tsforecast.adapters import DartsAdapter  # noqa: E402


class TestDartsLinearContract(AdapterContractTests):
    """Unit contract of a row-wise ``LinearRegressionModel``."""

    case = darts_linear_case()


class TestDartsForestContract(AdapterContractTests):
    """Unit contract of a row-wise ``RandomForestModel``."""

    case = darts_forest_case()


class TestDartsSeasonalContract(AdapterContractTests):
    """Unit contract of the local ``NaiveSeasonal`` model."""

    case = darts_seasonal_case()


class TestDartsLaggedContract(AdapterContractTests):
    """Unit contract of a global ``LinearRegressionModel`` on target lags."""

    case = darts_lagged_case()


# ---------------------------------------------------------------------------
# Données et outils
# ---------------------------------------------------------------------------


def row_adapter(**params) -> DartsAdapter:
    """``DartsAdapter`` around a row-wise ``LinearRegressionModel``."""
    return DartsAdapter(LinearRegressionModel(lags_future_covariates=[0], **params))


def sklearn_predictions(X_train: pd.DataFrame, y_train, X_test: pd.DataFrame) -> np.ndarray:
    """Predictions of a scikit-learn linear regression pooled over all the training rows."""
    return LinearRegression().fit(X_train.to_numpy(), np.asarray(y_train)).predict(X_test.to_numpy())


def linear_series(n_dates: int = 12, slope: float = 2.0, intercept: float = 1.0) -> tuple:
    """``(X, y)`` on month starts with ``y = intercept + slope * step``: drift and AR(1) are exact."""
    dates = pd.date_range('2021-01-01', periods=n_dates, freq='MS', name='date')
    X = pd.DataFrame({'x1': np.arange(n_dates, dtype=float), 'x2': np.cos(np.arange(n_dates))}, index=dates)
    return X, pd.Series(intercept + slope * np.arange(n_dates), index=dates, name='target')


def linear_panel(slopes=(('A', 1.0, 0.0), ('B', -3.0, 10.0))) -> tuple:
    """Panel of exact linear trends, one ``(entity, slope, intercept)`` per entity."""
    frames, targets = [], []
    for entity, slope, intercept in slopes:
        X, y = linear_series(slope=slope, intercept=intercept)
        index = pd.MultiIndex.from_arrays([np.repeat(entity, len(X)), X.index], names=['entity', 'date'])
        frames.append(X.set_axis(index))
        targets.append(y.set_axis(index))
    return pd.concat(frames), pd.concat(targets)


def relabel_entities(X: pd.DataFrame, y: pd.Series, mapping: dict) -> tuple:
    """Return ``(X, y)`` whose entity labels are replaced through ``mapping``."""
    X = X.rename(index=mapping, level=0)
    return X, y.set_axis(X.index)


def entity_mask(X: pd.DataFrame, entity) -> np.ndarray:
    """Boolean mask of the rows of an entity."""
    return np.asarray(X.index.get_level_values(0) == entity)


# ---------------------------------------------------------------------------
# Modèle enveloppé, attributs et étiquettes sklearn
# ---------------------------------------------------------------------------


class TestWrappedModel:
    """The darts model of the constructor is never modified nor replaced."""

    def test_model_kept_after_fit(self, adapter_series):
        X, y = adapter_series
        model = LinearRegressionModel(lags=3)
        adapter = DartsAdapter(model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            adapter.fit(X, y)
        assert adapter.model is model
        assert adapter.get_params()['model__lags'] == 3
        assert adapter.model_ is not model
        assert not any('DartsAdapter' in str(warning.message) for warning in caught)

    def test_nested_sklearn_estimator_never_fitted(self, adapter_series):
        X, y = adapter_series
        inner = Ridge()
        DartsAdapter(SKLearnModel(lags_future_covariates=[0], model=inner)).fit(X, y)
        assert not hasattr(inner, 'coef_')

    def test_is_regressor(self):
        assert is_regressor(row_adapter())

    def test_predict_before_fit(self, adapter_series):
        X, _ = adapter_series
        with pytest.raises(NotFittedError):
            row_adapter().predict(X)

    def test_refit_with_another_mode_drops_previous_state(self, adapter_series):
        X, y = adapter_series
        adapter = DartsAdapter(LinearRegressionModel(lags=3)).fit(X, y)
        assert hasattr(adapter, 'series_')
        adapter.set_params(model=LinearRegressionModel(lags_future_covariates=[0])).fit(X, y)
        assert adapter.mode_ == 'row' and not hasattr(adapter, 'series_')


# ---------------------------------------------------------------------------
# Mode de prévision
# ---------------------------------------------------------------------------


class TestMode:
    """The mode is read from the configuration of the darts model."""

    @pytest.mark.parametrize(
        ('make_model', 'mode'),
        [
            (lambda: LinearRegressionModel(lags_future_covariates=[0]), 'row'),
            (lambda: RandomForestModel(lags_future_covariates=[0], n_estimators=2), 'row'),
            (lambda: LinearRegressionModel(lags=3), 'global'),
            (lambda: LinearRegressionModel(lags_future_covariates=[-1, 0]), 'global'),
            (lambda: LinearRegressionModel(lags_future_covariates=[0], output_chunk_length=2), 'global'),
            (lambda: LinearRegressionModel(lags_future_covariates=[0], output_chunk_shift=1), 'global'),
            (lambda: NaiveDrift(), 'local'),
            (lambda: ExponentialSmoothing(seasonal=None), 'local'),
        ],
    )
    def test_mode(self, make_model, mode):
        X, y = linear_series(n_dates=24)
        assert DartsAdapter(make_model()).fit(X, y).mode_ == mode

    @pytest.mark.parametrize(
        'model',
        [LinearRegression(), SKLearnClassifierModel(lags_future_covariates=[0])],
        ids=['sklearn-estimator', 'darts-classifier'],
    )
    def test_refused_objects(self, model, adapter_series):
        X, y = adapter_series
        with pytest.raises(TypeError):
            DartsAdapter(model).fit(X, y)

    def test_position_encoders_refused_row_wise(self, adapter_series):
        X, y = adapter_series
        adapter = row_adapter(add_encoders={'position': {'future': ['relative']}})
        with pytest.raises(ValueError, match='Position encoders'):
            adapter.fit(X, y)

    def test_irregular_grid(self, adapter_series):
        X, y = adapter_series
        rows = [0, 1, 3, 4, 9, 10, 20]
        irregular = X.iloc[rows].set_axis(X.index[rows] + pd.to_timedelta([0, 0, 3, 0, 0, 5, 0], unit='D'))
        with pytest.raises(ValueError, match='regular grid'):
            row_adapter().fit(irregular, y.iloc[rows].set_axis(irregular.index))


# ---------------------------------------------------------------------------
# Mode ligne à ligne : régression mutualisée sur X
# ---------------------------------------------------------------------------


class TestRowWise:
    """A row-wise model is a scikit-learn regression pooled over the rows of all entities."""

    def test_series_matches_sklearn(self, adapter_series):
        X_train, y_train, X_test, _ = split_by_date(*adapter_series)
        predictions = row_adapter().fit(X_train, y_train).predict(X_test)
        np.testing.assert_allclose(predictions, sklearn_predictions(X_train, y_train, X_test))

    @pytest.mark.parametrize('balanced', [True, False])
    def test_panel_matches_pooled_sklearn(self, balanced):
        X_train, y_train, X_test, _ = split_by_date(*build_adapter_panel(balanced=balanced))
        predictions = row_adapter().fit(X_train, y_train).predict(X_test)
        np.testing.assert_allclose(predictions, sklearn_predictions(X_train, y_train, X_test))

    def test_no_cross_entity_covariates(self, adapter_panel):
        X_train, y_train, X_test, _ = split_by_date(*adapter_panel)
        adapter = row_adapter().fit(X_train, y_train)
        reference = adapter.predict(X_test)
        changed = X_test.copy()
        changed.loc[entity_mask(changed, 'B'), :] += 100.0
        is_a = entity_mask(X_test, 'A')
        np.testing.assert_allclose(adapter.predict(changed)[is_a], reference[is_a])

    def test_integer_entities(self, adapter_panel):
        X, y = relabel_entities(*adapter_panel, {'A': 1, 'B': 2, 'C': 3})
        X_train, y_train, X_test, _ = split_by_date(X, y)
        predictions = row_adapter().fit(X_train, y_train).predict(X_test)
        np.testing.assert_allclose(predictions, sklearn_predictions(X_train, y_train, X_test))

    def test_two_entity_levels(self, adapter_panel):
        X, y = adapter_panel
        regions = np.where(X.index.get_level_values('entity') == 'A', 'north', 'south')
        index = pd.MultiIndex.from_arrays(
            [regions, X.index.get_level_values('entity'), X.index.get_level_values('date')],
            names=['region', 'entity', 'date'],
        )
        X_train, y_train, X_test, _ = split_by_date(X.set_axis(index), y.set_axis(index))
        predictions = row_adapter().fit(X_train, y_train).predict(X_test)
        np.testing.assert_allclose(predictions, sklearn_predictions(X_train, y_train, X_test))

    def test_training_holes_use_the_kept_rows(self, adapter_panel):
        X_train, y_train, X_test, _ = split_by_date(*adapter_panel)
        # Trous dont une ligne isolée entre deux trous (suite d'une seule ligne)
        keep = np.ones(len(X_train), dtype=bool)
        keep[[3, 5, 6, 7, 12, 40]] = False
        predictions = row_adapter().fit(X_train[keep], y_train[keep]).predict(X_test)
        np.testing.assert_allclose(predictions, sklearn_predictions(X_train[keep], y_train[keep], X_test))

    def test_in_sample_rows(self, adapter_panel):
        X_train, y_train, _, _ = split_by_date(*adapter_panel)
        adapter = row_adapter().fit(X_train, y_train)
        np.testing.assert_allclose(adapter.predict(X_train), sklearn_predictions(X_train, y_train, X_train))
        assert adapter.score(X_train, y_train) > 0.9

    def test_rows_predicted_alone_or_in_batch(self, adapter_panel):
        X_train, y_train, X_test, _ = split_by_date(*adapter_panel, n_test=6, gap=2)
        adapter = row_adapter().fit(X_train, y_train)
        # Lignes de test non contiguës : une date sur deux
        sparse = X_test.iloc[::2]
        batch = adapter.predict(sparse)
        alone = np.array([adapter.predict(sparse.iloc[[row]])[0] for row in range(len(sparse))])
        np.testing.assert_allclose(batch, alone)
        np.testing.assert_allclose(batch, sklearn_predictions(X_train, y_train, sparse))

    def test_unknown_entity_predicted(self, adapter_panel):
        X_train, y_train, X_test, _ = split_by_date(*adapter_panel)
        known = ~entity_mask(X_train, 'C')
        predictions = row_adapter().fit(X_train[known], y_train[known]).predict(X_test)
        np.testing.assert_allclose(predictions, sklearn_predictions(X_train[known], y_train[known], X_test))

    def test_several_targets(self, adapter_panel):
        X, y = adapter_panel
        targets = pd.DataFrame({'a': y, 'b': -2.0 * y + X['x2']})
        X_train, y_train, X_test, _ = split_by_date(X, targets)
        predictions = row_adapter().fit(X_train, y_train).predict(X_test)
        assert predictions.shape == (len(X_test), 2)
        np.testing.assert_allclose(predictions, sklearn_predictions(X_train, y_train, X_test))

    def test_calendar_encoders(self, adapter_series):
        X_train, y_train, X_test, _ = split_by_date(*adapter_series, n_test=6)
        adapter = row_adapter(add_encoders={'datetime_attribute': {'future': ['month']}}).fit(X_train, y_train)
        batch = adapter.predict(X_test)
        alone = np.array([adapter.predict(X_test.iloc[[row]])[0] for row in range(len(X_test))])
        np.testing.assert_allclose(batch, alone)

    def test_empty_X(self, adapter_series):
        X, y = adapter_series
        assert row_adapter().fit(X, y).predict(X.iloc[:0]).shape == (0,)

    def test_columns_seen_at_fit(self, adapter_series):
        X, y = adapter_series
        with pytest.raises(ValueError):
            row_adapter().fit(X, y).predict(X.rename(columns={'x1': 'other'}))

    def test_date_off_the_grid(self, adapter_series):
        X, y = adapter_series
        adapter = row_adapter().fit(X, y)
        with pytest.raises(ValueError, match='grid'):
            adapter.predict(X.iloc[[3]].set_axis(X.index[[3]] + pd.Timedelta(days=4)))


# ---------------------------------------------------------------------------
# Extrapolation de l'historique de la cible
# ---------------------------------------------------------------------------


class TestExtrapolation:
    """Extrapolating models forecast through the gap from the last training date of each entity."""

    def test_local_drift_through_the_gap(self):
        X, y = linear_series()
        adapter = DartsAdapter(NaiveDrift()).fit(X.iloc[:6], y.iloc[:6])
        np.testing.assert_allclose(adapter.predict(X.iloc[[11, 8]]), y.iloc[[11, 8]].to_numpy())

    def test_local_drift_per_entity(self):
        X, y = linear_panel()
        train = np.asarray(X.index.get_level_values('date') < '2021-07-01')
        test = np.asarray(X.index.get_level_values('date') >= '2021-10-01')
        adapter = DartsAdapter(NaiveDrift()).fit(X[train], y[train])
        assert len(adapter.entity_models_) == 2
        np.testing.assert_allclose(adapter.predict(X[test].iloc[::-1]), y[test].iloc[::-1].to_numpy())

    def test_global_lags_through_the_gap(self):
        # Même pente, ordonnées différentes : y(t) = y(t-1) + 2 exact pour toutes les entités
        X, y = linear_panel(slopes=(('A', 2.0, 0.0), ('B', 2.0, 30.0)))
        train = np.asarray(X.index.get_level_values('date') < '2021-08-01')
        test = np.asarray(X.index.get_level_values('date') >= '2021-10-01')
        adapter = DartsAdapter(LinearRegressionModel(lags=1)).fit(X[train], y[train])
        assert adapter.mode_ == 'global'
        np.testing.assert_allclose(adapter.predict(X[test]), y[test].to_numpy(), atol=1e-8)

    def test_global_several_targets(self):
        X, y = linear_series()
        targets = pd.DataFrame({'a': y, 'b': 3.0 * y})
        adapter = DartsAdapter(LinearRegressionModel(lags=1)).fit(X.iloc[:8], targets.iloc[:8])
        np.testing.assert_allclose(adapter.predict(X.iloc[[10, 11]]), targets.iloc[[10, 11]].to_numpy(), atol=1e-8)

    @pytest.mark.parametrize(
        'covariate_lags',
        [{'lags_past_covariates': [-1]}, {'lags_future_covariates': [0]}],
        ids=['past', 'future'],
    )
    def test_covariates_without_gap(self, covariate_lags):
        X, y = linear_series(n_dates=40)
        adapter = DartsAdapter(LinearRegressionModel(lags=1, **covariate_lags)).fit(X.iloc[:30], y.iloc[:30])
        predictions = adapter.predict(X.iloc[30:34])
        np.testing.assert_allclose(predictions, y.iloc[30:34].to_numpy(), atol=1e-6)

    @pytest.mark.parametrize(
        'make_model',
        [
            lambda: LinearRegressionModel(lags=1, lags_past_covariates=[-1]),
            lambda: LinearRegressionModel(lags=1, lags_future_covariates=[0]),
            lambda: ARIMA(p=1, d=0, q=0),
        ],
        ids=['global-past-covariates', 'global-future-covariates', 'local-future-covariates'],
    )
    def test_covariates_missing_on_the_gap(self, make_model):
        X, y = linear_series(n_dates=40)
        adapter = DartsAdapter(make_model()).fit(X.iloc[:30], y.iloc[:30])
        with pytest.raises(ValueError, match=r"no row on '2023-07-01"):
            adapter.predict(X.iloc[[32, 33]])

    def test_local_covariates_without_gap(self):
        X, y = linear_series(n_dates=40)
        adapter = DartsAdapter(ARIMA(p=1, d=0, q=0)).fit(X.iloc[:30], y.iloc[:30])
        assert np.isfinite(adapter.predict(X.iloc[30:33])).all()

    @pytest.mark.parametrize('make_model', [NaiveDrift, lambda: LinearRegressionModel(lags=2)])
    def test_rows_before_the_end_of_training(self, make_model):
        X, y = linear_series()
        adapter = DartsAdapter(make_model()).fit(X.iloc[:8], y.iloc[:8])
        with pytest.raises(ValueError, match='at or before the last training date'):
            adapter.predict(X.iloc[[7, 9]])

    @pytest.mark.parametrize('make_model', [NaiveDrift, lambda: LinearRegressionModel(lags=2)])
    def test_training_holes_refused(self, make_model):
        X, y = linear_panel()
        keep = np.ones(len(X), dtype=bool)
        keep[[14, 15]] = False
        with pytest.raises(ValueError, match=r"entities \('B',\).*contiguous"):
            DartsAdapter(make_model()).fit(X[keep], y[keep])

    def test_unknown_entity(self):
        X, y = linear_panel()
        is_a = entity_mask(X, 'A')
        adapter = DartsAdapter(NaiveDrift()).fit(X[is_a].iloc[:6], y[is_a].iloc[:6])
        with pytest.raises(ValueError, match="Entities unknown at fit: \\('B',\\)"):
            adapter.predict(X[~is_a].iloc[[8]])

    def test_univariate_local_model_refuses_several_targets(self):
        X, y = linear_series(n_dates=24)
        targets = pd.DataFrame({'a': y, 'b': 2.0 * y})
        with pytest.raises(ValueError, match='several targets'):
            DartsAdapter(ExponentialSmoothing(seasonal=None)).fit(X, targets)


# ---------------------------------------------------------------------------
# Paramètres exposés à sklearn
# ---------------------------------------------------------------------------


class TestParams:
    """darts parameters through ``get_params`` / ``set_params``, compatible with ``clone``."""

    def test_deep_params(self):
        adapter = DartsAdapter(SKLearnModel(lags_future_covariates=[0], model=Ridge(alpha=3.0)))
        params = adapter.get_params(deep=True)
        assert params['model__lags_future_covariates'] == [0]
        assert params['model__model__alpha'] == 3.0
        assert set(adapter.get_params(deep=False)) == {'model'}

    def test_set_nested_estimator_param(self):
        inner = Ridge(alpha=1.0)
        model = SKLearnModel(lags_future_covariates=[0], model=inner)
        adapter = DartsAdapter(model).set_params(model__model__alpha=2.0)
        assert adapter.model is not model and adapter.get_params()['model__model__alpha'] == 2.0
        assert inner.alpha == 1.0 and model.model_params['model'].alpha == 1.0

    def test_set_unchanged_values_keeps_the_model(self):
        model = SKLearnModel(lags_future_covariates=[0], model=Ridge())
        adapter = DartsAdapter(model)
        adapter.set_params(**adapter.get_params(deep=True))
        assert adapter.model is model

    def test_set_constructor_keyword(self):
        adapter = row_adapter().set_params(model__fit_intercept=False)
        assert adapter.get_params()['model__fit_intercept'] is False
        assert adapter.set_params(model__lags=2).get_params()['model__lags'] == 2

    @pytest.mark.parametrize(
        ('model', 'params'),
        [
            (RandomForestModel(lags_future_covariates=[0]), {'other': 1}),
            (RandomForestModel(lags_future_covariates=[0]), {'model__model__alpha': 1.0}),
            (NaiveSeasonal(K=1), {'model__unknown': 1}),
        ],
        ids=['adapter-key', 'nested-key-without-estimator', 'model-key'],
    )
    def test_invalid_keys(self, model, params):
        with pytest.raises(ValueError, match='Invalid parameter'):
            DartsAdapter(model).set_params(**params)

    def test_clone_keeps_the_values(self):
        adapter = DartsAdapter(RandomForestModel(lags_future_covariates=[0], n_estimators=7, random_state=3))
        cloned = clone(adapter)
        assert cloned.model is not adapter.model
        assert cloned.get_params()['model__n_estimators'] == 7
        assert cloned.get_params()['model__random_state'] == 3

    def test_clone_predicts_like_the_original(self, adapter_series):
        X_train, y_train, X_test, _ = split_by_date(*adapter_series)
        adapter = DartsAdapter(RandomForestModel(lags_future_covariates=[0], n_estimators=5, random_state=0))
        reference = adapter.fit(X_train, y_train).predict(X_test)
        np.testing.assert_allclose(clone(adapter).fit(X_train, y_train).predict(X_test), reference)
