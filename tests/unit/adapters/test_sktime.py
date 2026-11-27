"""Unit tests of ``tsforecast.adapters.sktime.SktimeAdapter``.

Runs the unit rules of the adapter contract on ``NaiveForecaster`` and
``YfromX(LinearRegression())``, compares drift, exponential smoothing, Theta and
``SARIMAX`` with sktime used natively, then targeted scenarios: absolute horizon read
from the dates of ``X`` (gap, several test dates, panels), wrapped forecaster
never fitted, sklearn parameters and tags, output shape and row order, unknown
entities, incomplete panel test rows (vectorized and native panel forecasters),
forecasters needing ``X`` on the gap or the horizon at fit, training rows with
holes, panel entities with too few training dates for sktime, irregular grids,
hierarchical indexes and several targets.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sktime')

from sklearn.base import is_regressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sktime.forecasting.base import BaseForecaster  # noqa: E402
from sktime.forecasting.compose import DirectReductionForecaster, YfromX  # noqa: E402
from sktime.forecasting.naive import NaiveForecaster  # noqa: E402

from tests.support.adapter_cases import (  # noqa: E402
    sktime_naive_case,
    sktime_native_cases,
    sktime_yfromx_case,
)
from tests.support.adapter_contract import (  # noqa: E402
    NATIVE_GAPS,
    AdapterContractTests,
    check_native_equivalence,
    split_by_date,
)
from tests.support.datasets import build_adapter_panel, build_adapter_series  # noqa: E402
from tests.support.perturbations import shuffle_rows  # noqa: E402
from tsforecast.adapters import SktimeAdapter  # noqa: E402

# Mtypes sktime des séries et panels acceptés par le forecaster global de test
_PANEL_MTYPES = ['pd.DataFrame', 'pd-multiindex', 'pd_multiindex_hier']


class PooledMeanForecaster(BaseForecaster):
    """Global test forecaster with native panel support: pooled target mean plus ``scale`` times the first feature."""

    _tags = {
        'y_inner_mtype': _PANEL_MTYPES,
        'X_inner_mtype': _PANEL_MTYPES,
        'capability:exogenous': True,
        'capability:missing_values': True,
        'capability:non_contiguous_X': True,
        'requires-fh-in-fit': False,
    }

    def __init__(self, scale: float = 0.0):
        self.scale = scale
        super().__init__()

    def _fit(self, y, X, fh):
        self.mean_ = float(np.nanmean(y.to_numpy()))
        self.columns_ = y.columns
        return self

    def _predict(self, fh, X):
        values = self.mean_ + self.scale * X.iloc[:, 0].to_numpy()
        return pd.DataFrame(values, index=X.index, columns=self.columns_)


class ShortHistoryForecaster(BaseForecaster):
    """Local test forecaster repeating the last target, failing on a history of fewer than three dates."""

    _tags = {
        'y_inner_mtype': 'pd.DataFrame',
        'X_inner_mtype': 'pd.DataFrame',
        'capability:exogenous': True,
        'capability:non_contiguous_X': True,
        'requires-fh-in-fit': False,
    }

    def __init__(self, fail_at: str = 'predict'):
        self.fail_at = fail_at
        super().__init__()

    def _fit(self, y, X, fh):
        self.n_dates_ = len(y)
        self.last_ = y.iloc[-1:]
        if self.n_dates_ < 3 and self.fail_at == 'fit':
            raise NotImplementedError('history too short')
        return self

    def _predict(self, fh, X):
        if self.n_dates_ < 3:
            raise TypeError('history too short')
        index = fh.to_absolute_index(self.cutoff)
        return pd.DataFrame(
            np.repeat(self.last_.to_numpy(), len(index), axis=0), index=index, columns=self.last_.columns
        )


def short_entity_panel(histories: dict, n_test: int = 2) -> tuple:
    """``(X_train, y_train, X_test)``: entity ``k`` keeps the last ``histories[k]`` of twelve training months.

    ``x1`` is the month position and ``y = 1 + x1`` for every entity.
    """
    dates = pd.date_range('2021-01-01', periods=12 + n_test, freq='MS')
    train = pd.MultiIndex.from_tuples(
        [(entity, date) for entity, n_dates in histories.items() for date in dates[12 - n_dates:12]],
        names=['entity', 'date'],
    )
    test = pd.MultiIndex.from_product([list(histories), dates[12:]], names=['entity', 'date'])
    X_train, X_test = (
        pd.DataFrame({'x1': dates.get_indexer(index.get_level_values('date')).astype(float)}, index=index)
        for index in (train, test)
    )
    return X_train, (1.0 + X_train['x1']).rename('target'), X_test


def linear_series(n_dates: int = 12, slope: float = 2.0) -> tuple:
    """``(X, y)`` on month starts with ``y = 1 + slope * step``: a drift forecast is exact."""
    dates = pd.date_range('2021-01-01', periods=n_dates, freq='MS', name='date')
    X = pd.DataFrame({'x1': np.arange(n_dates, dtype=float), 'x2': np.ones(n_dates)}, index=dates)
    return X, pd.Series(1.0 + slope * np.arange(n_dates), index=dates, name='target')


def linear_panel() -> tuple:
    """Unbalanced panel, one exact linear trend per entity (``A``: slope 1, ``B``: slope −3)."""
    frames, targets = [], []
    for entity, slope, start in (('A', 1.0, 0), ('B', -3.0, 4)):
        X, y = linear_series(n_dates=12 - start, slope=slope)
        index = pd.MultiIndex.from_arrays(
            [np.repeat(entity, len(X)), X.index + pd.DateOffset(months=start)], names=['entity', 'date']
        )
        frames.append(X.set_axis(index))
        targets.append(y.set_axis(index))
    return pd.concat(frames), pd.concat(targets)


def entity_mask(X: pd.DataFrame, entity: str) -> np.ndarray:
    """Boolean mask of the rows of an entity."""
    return np.asarray(X.index.get_level_values(0) == entity)


def per_entity_regression(X_train, y_train, X_test) -> np.ndarray:
    """Reference of a local ``YfromX``: one linear regression per entity, rows of ``X_test``."""
    result = np.empty(len(X_test))
    for entity in X_test.index.get_level_values(0).unique():
        train, test = entity_mask(X_train, entity), entity_mask(X_test, entity)
        result[test] = LinearRegression().fit(X_train[train], y_train[train]).predict(X_test[test])
    return result


# =============================================================================
# Batterie de contrat
# =============================================================================


class TestSktimeNaiveContract(AdapterContractTests):
    """Contract of ``SktimeAdapter(NaiveForecaster())``."""

    case = sktime_naive_case()


class TestSktimeYfromXContract(AdapterContractTests):
    """Contract of ``SktimeAdapter(YfromX(LinearRegression()))``."""

    case = sktime_yfromx_case()


class TestNativeEquivalence:
    """Drift, exponential smoothing, Theta and ``SARIMAX`` predict like sktime used natively."""

    @pytest.mark.parametrize('gap', NATIVE_GAPS)
    @pytest.mark.parametrize('kind', ['series', 'panel'])
    @pytest.mark.parametrize('name', ['drift', 'exponential_smoothing', 'theta', 'sarimax'])
    def test_same_forecasts_as_sktime(self, name, kind, gap):
        """Relative horizon ``fh = gap + 1, ...`` of a sktime fit on the training rows."""
        pytest.importorskip('statsmodels')
        case = sktime_native_cases()[name]
        if gap not in case.native_gaps:
            pytest.skip(f'{name} needs X on the gap')
        check_native_equivalence(case, kind, gap)


# =============================================================================
# Horizon absolu lu sur les dates de X
# =============================================================================


class TestAbsoluteHorizon:
    """Each row is forecast at its own date, counted in grid steps from the end of training."""

    def test_last_value_on_every_test_row(self):
        """Gap of two dates, three test dates: three forecasts equal to the last target."""
        X, y = build_adapter_series()
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=3, gap=2)
        predictions = SktimeAdapter(NaiveForecaster()).fit(X_train, y_train).predict(X_test)
        np.testing.assert_allclose(predictions, np.repeat(y_train.iloc[-1], 3))

    def test_drift_counts_grid_steps_through_the_gap(self):
        """A drift forecast of an exact trend recovers the target 3, 4 and 5 steps ahead."""
        X, y = linear_series()
        X_train, y_train, X_test, y_test = split_by_date(X, y, n_test=3, gap=2)
        model = SktimeAdapter(NaiveForecaster(strategy='drift')).fit(X_train, y_train)
        np.testing.assert_allclose(model.predict(X_test), y_test.to_numpy())
        np.testing.assert_allclose(model.predict(X_test.iloc[::-1]), y_test.to_numpy()[::-1])

    def test_drift_per_entity_of_an_unbalanced_panel(self):
        """Each entity is extrapolated from its own last training date."""
        X, y = linear_panel()
        X_train, y_train, X_test, y_test = split_by_date(X, y, n_test=2, gap=1)
        model = SktimeAdapter(NaiveForecaster(strategy='drift')).fit(X_train, y_train)
        np.testing.assert_allclose(model.predict(X_test), y_test.to_numpy())

    def test_yfromx_predicts_the_rows_of_x(self):
        """``YfromX`` predicts each test row from its features, not the end of training."""
        X, y = build_adapter_series()
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=3, gap=2)
        predictions = SktimeAdapter(YfromX(LinearRegression())).fit(X_train, y_train).predict(X_test)
        expected = LinearRegression().fit(X_train, y_train).predict(X_test)
        np.testing.assert_allclose(predictions, expected)

    def test_predict_does_not_depend_on_previous_predictions(self):
        """A predict on other dates leaves the next predictions unchanged."""
        X, y = build_adapter_series()
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=3, gap=0)
        model = SktimeAdapter(NaiveForecaster(strategy='drift')).fit(X_train, y_train)
        reference = model.predict(X_test)
        model.predict(X_test.iloc[-1:])
        np.testing.assert_allclose(model.predict(X_test), reference)

    def test_no_fh_parameter(self):
        """The horizon is never a parameter."""
        assert sorted(SktimeAdapter(NaiveForecaster()).get_params(deep=False)) == ['forecaster']


# =============================================================================
# Forecaster enveloppé et API sklearn
# =============================================================================


class TestWrappedForecaster:
    """The wrapped forecaster is never fitted; sklearn handles the parameters."""

    def test_forecaster_shared_by_two_adapters(self):
        """Fitting a second adapter on the same forecaster leaves the first one unchanged."""
        X, y = build_adapter_series()
        X_train, y_train, X_test, _ = split_by_date(X, y)
        forecaster = NaiveForecaster()
        first = SktimeAdapter(forecaster).fit(X_train, y_train)
        reference = first.predict(X_test)
        SktimeAdapter(forecaster).fit(X_train.iloc[:10], y_train.iloc[:10])
        np.testing.assert_allclose(first.predict(X_test), reference)
        assert not forecaster.is_fitted
        assert first.forecaster_ is not forecaster

    def test_nested_parameters(self):
        """Nested parameters go through ``forecaster__``; unknown ones are refused."""
        adapter = SktimeAdapter(NaiveForecaster())
        adapter.set_params(forecaster__strategy='mean')
        assert adapter.forecaster.strategy == 'mean'
        assert adapter.get_params()['forecaster__strategy'] == 'mean'
        with pytest.raises(ValueError, match='Invalid parameter'):
            adapter.set_params(fh=1)

    def test_is_a_regressor(self):
        """The regressor tags are those of ``RegressorMixin``."""
        assert is_regressor(SktimeAdapter(NaiveForecaster()))

    def test_not_a_forecaster(self):
        """A scikit-learn estimator is not a sktime forecaster."""
        X, y = build_adapter_series()
        with pytest.raises(TypeError, match='sktime forecaster'):
            SktimeAdapter(LinearRegression()).fit(X, y)


# =============================================================================
# Sortie
# =============================================================================


class TestOutput:
    """Array in the row order of ``X``, one column per target."""

    def test_series_target_gives_one_dimension(self):
        """A ``Series`` target gives an array of shape ``(n_samples,)``."""
        X, y = build_adapter_panel()
        X_train, y_train, X_test, _ = split_by_date(X, y)
        predictions = SktimeAdapter(NaiveForecaster()).fit(X_train, y_train.rename(None)).predict(X_test)
        assert isinstance(predictions, np.ndarray) and predictions.shape == (len(X_test),)

    @pytest.mark.parametrize('forecaster', [NaiveForecaster(), YfromX(LinearRegression())], ids=['naive', 'yfromx'])
    def test_several_targets_keep_their_column_order(self, forecaster):
        """A ``DataFrame`` target gives one column per target, in its column order."""
        X, y = build_adapter_panel(balanced=False)
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=2)
        targets = pd.DataFrame({'b': -y_train, 'a': y_train})
        model = SktimeAdapter(forecaster).fit(X_train, targets)
        single = SktimeAdapter(forecaster).fit(X_train, y_train).predict(X_test)
        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test), 2)
        np.testing.assert_allclose(predictions, np.column_stack([-single, single]))

    def test_empty_x(self):
        """No row to predict gives an empty array."""
        X, y = build_adapter_series()
        model = SktimeAdapter(NaiveForecaster()).fit(X, y)
        assert model.predict(X.iloc[:0]).shape == (0,)

    def test_shuffled_training_and_test_rows(self):
        """Rows in any order give the same model and predictions following ``X``."""
        X, y = build_adapter_panel(balanced=False)
        X_train, y_train, X_test, _ = split_by_date(X, y)
        reference = SktimeAdapter(YfromX(LinearRegression())).fit(X_train, y_train).predict(X_test)
        order = shuffle_rows(X_train, seed=3).index
        model = SktimeAdapter(YfromX(LinearRegression())).fit(X_train.loc[order], y_train.loc[order])
        shuffled = shuffle_rows(X_test, seed=4)
        np.testing.assert_allclose(model.predict(shuffled), reference[X_test.index.get_indexer(shuffled.index)])


# =============================================================================
# Panels
# =============================================================================


class TestPanel:
    """Unknown entities, incomplete test rows, hierarchical indexes."""

    def test_unknown_entity(self):
        """An entity absent from the training rows has no fitted model."""
        X, y = build_adapter_panel()
        X_train, y_train, X_test, _ = split_by_date(X, y)
        known = ~entity_mask(X_train, 'C')
        model = SktimeAdapter(NaiveForecaster()).fit(X_train[known], y_train[known])
        with pytest.raises(ValueError, match=r"Entities unknown at fit: \('C',\)"):
            model.predict(X_test)

    def test_local_yfromx_is_one_regression_per_entity(self):
        """sktime fits ``YfromX`` once per entity of a panel."""
        X, y = build_adapter_panel(balanced=False)
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=2)
        predictions = SktimeAdapter(YfromX(LinearRegression())).fit(X_train, y_train).predict(X_test)
        np.testing.assert_allclose(predictions, per_entity_regression(X_train, y_train, X_test))

    @pytest.mark.parametrize('missing', ['date', 'entity'])
    def test_incomplete_test_rows(self, missing):
        """A test entity without one of the dates, or an entity without test rows, is still predicted."""
        X, y = build_adapter_panel()
        X_train, y_train, X_test, _ = split_by_date(X, y)
        dates = X_test.index.get_level_values(-1)
        drop = entity_mask(X_test, 'B') & np.asarray(dates == dates.max() if missing == 'date' else True)
        model = SktimeAdapter(YfromX(LinearRegression())).fit(X_train, y_train)
        np.testing.assert_allclose(model.predict(X_test[~drop]), model.predict(X_test)[~drop])

    def test_native_panel_forecaster_needs_every_row(self):
        """A global forecaster forecasts all entities on common dates: missing rows of X are refused."""
        X, y = build_adapter_panel(balanced=False)
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=2)
        model = SktimeAdapter(PooledMeanForecaster(scale=1.0)).fit(X_train, y_train)
        np.testing.assert_allclose(model.predict(X_test), y_train.mean() + X_test['x1'].to_numpy())
        with pytest.raises(ValueError, match=r'X lacks 1 \(entity, date\) row'):
            model.predict(X_test.iloc[1:])

    def test_hierarchical_index(self):
        """Two entity levels: one local model per leaf, incomplete test rows accepted."""
        X, y = build_adapter_panel(balanced=False)
        groups = np.where(X.index.get_level_values(0) == 'A', 'g1', 'g2')
        index = pd.MultiIndex.from_arrays(
            [groups, X.index.get_level_values(0), X.index.get_level_values(1)], names=['group', 'entity', 'date']
        )
        X_train, y_train, X_test, _ = split_by_date(X.set_axis(index), y.set_axis(index), gap=2)
        model = SktimeAdapter(YfromX(LinearRegression())).fit(X_train, y_train)
        flat = [X_train.droplevel(0), y_train.droplevel(0), X_test.droplevel(0)]
        np.testing.assert_allclose(model.predict(X_test), per_entity_regression(*flat))
        kept = ~np.asarray(X_test.index.get_level_values('entity') == 'C')
        np.testing.assert_allclose(model.predict(X_test[kept]), model.predict(X_test)[kept])


# =============================================================================
# Contraintes des forecasters et des grilles
# =============================================================================


class TestForecasterConstraints:
    """Explicit errors instead of invented values or opaque sktime errors."""

    def test_x_needed_on_the_gap(self):
        """``SARIMAX`` needs ``X`` on every date up to the forecast: a gap absent from ``X`` is refused."""
        pytest.importorskip('statsmodels')
        from sktime.forecasting.sarimax import SARIMAX

        X, y = build_adapter_series()
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=3, gap=0)
        assert SktimeAdapter(SARIMAX()).fit(X_train, y_train).predict(X_test).shape == (3,)
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=3, gap=2)
        model = SktimeAdapter(SARIMAX()).fit(X_train, y_train)
        with pytest.raises(ValueError, match="needs X on every date.*'2022-08-01.*'2022-09-01"):
            model.predict(X_test)

    def test_x_needed_on_the_gap_names_the_entity(self):
        """On a panel, the error names the entity whose rows of ``X`` leave a gap."""
        pytest.importorskip('statsmodels')
        from sktime.forecasting.sarimax import SARIMAX

        X, y = build_adapter_panel()
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=3, gap=0)
        dates = X_test.index.get_level_values(-1)
        hole = entity_mask(X_test, 'B') & np.asarray(dates == dates.min())
        model = SktimeAdapter(SARIMAX()).fit(X_train, y_train)
        with pytest.raises(ValueError, match=r"for entity \('B',\)"):
            model.predict(X_test[~hole])

    def test_horizon_needed_at_fit(self):
        """A direct reduction needs the horizon at fit, unknown before predict."""
        X, y = build_adapter_series()
        with pytest.raises(ValueError, match='horizon at fit'):
            SktimeAdapter(DirectReductionForecaster(LinearRegression())).fit(X, y)

    def test_irregular_grid(self):
        """Dates without a common grid cannot be turned into forecast steps."""
        dates = pd.DatetimeIndex(['2024-01-01', '2024-01-15', '2024-02-03', '2024-03-01', '2024-03-20'], name='date')
        X = pd.DataFrame({'x': np.arange(5.0)}, index=dates)
        with pytest.raises(ValueError, match='regular grid'):
            SktimeAdapter(NaiveForecaster()).fit(X, pd.Series(np.arange(5.0), index=dates))


class TestTrainingHoles:
    """Training rows missing grid dates, as left by ``DropNaNTarget``."""

    @staticmethod
    def holed_split(X, y, gap: int = 2):
        """Training rows without one row in five, and the test rows."""
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=gap)
        keep = np.arange(len(X_train)) % 5 != 1
        return X_train[keep], y_train[keep], X_test

    @pytest.mark.parametrize('strategy', ['last', 'mean'])
    def test_target_only_forecaster_sees_missing_values(self, strategy):
        """A forecaster ignoring ``X`` sees the holes as missing target values on the grid."""
        X, y = build_adapter_series()
        X_train, y_train, X_test = self.holed_split(X, y)
        X_train, y_train = X_train.iloc[:-1], y_train.iloc[:-1]
        predictions = SktimeAdapter(NaiveForecaster(strategy=strategy)).fit(X_train, y_train).predict(X_test)
        expected = y_train.iloc[-1] if strategy == 'last' else y_train.mean()
        np.testing.assert_allclose(predictions, np.repeat(expected, len(X_test)))

    @pytest.mark.parametrize('shuffled', [False, True], ids=['sorted', 'shuffled'])
    def test_target_only_forecaster_on_a_panel(self, shuffled):
        """Each entity of a panel keeps its own last observed target, whatever the row order."""
        X, y = build_adapter_panel(balanced=False)
        X_train, y_train, X_test = self.holed_split(X, y)
        last = y_train.groupby(level='entity').last()
        if shuffled:
            # Entité C en tête : ordre d'apparition des entités différent de leur ordre trié
            order = shuffle_rows(X_train, seed=5).index
            order = order[np.argsort(order.get_level_values(0) != 'C', kind='stable')]
            X_train, y_train = X_train.loc[order], y_train.loc[order]
        predictions = SktimeAdapter(NaiveForecaster()).fit(X_train, y_train).predict(X_test)
        np.testing.assert_allclose(predictions, last.loc[X_test.index.get_level_values(0)].to_numpy())

    def test_row_wise_forecaster_uses_the_rows(self):
        """``YfromX`` fits on the rows given, holes included."""
        X, y = build_adapter_panel(balanced=False)
        X_train, y_train, X_test = self.holed_split(X, y)
        predictions = SktimeAdapter(YfromX(LinearRegression())).fit(X_train, y_train).predict(X_test)
        np.testing.assert_allclose(predictions, per_entity_regression(X_train, y_train, X_test))

    def test_forecaster_without_missing_values_refuses_holes(self):
        """``SARIMAX`` handles no missing value: holed training rows are refused."""
        pytest.importorskip('statsmodels')
        from sktime.forecasting.sarimax import SARIMAX

        X, y = build_adapter_panel()
        X_train, y_train, _ = self.holed_split(X, y)
        with pytest.raises(ValueError, match="entities .*do not cover every date of the 'MS' grid"):
            SktimeAdapter(SARIMAX()).fit(X_train, y_train)


# =============================================================================
# Entités trop courtes pour la lecture de leur fréquence
# =============================================================================


SHORT_MESSAGE = r"failed on a panel where the entities \('A',\) have fewer than 3 training dates"


class TestShortEntities:
    """Panel entities with fewer than three training dates, the number sktime needs to read a frequency."""

    @pytest.mark.parametrize('n_dates', [1, 2])
    def test_naive_forecaster_failure_is_explicit(self, n_dates):
        """The sktime error is raised as a ``ValueError`` naming the entity, and kept as its cause."""
        X_train, y_train, X_test = short_entity_panel({'A': n_dates, 'B': 12})
        with pytest.raises(ValueError, match=SHORT_MESSAGE) as error:
            SktimeAdapter(NaiveForecaster()).fit(X_train, y_train).predict(X_test)
        assert 'infer frequency' in str(error.value.__cause__)

    @pytest.mark.parametrize('fail_at', ['fit', 'predict'])
    def test_failure_at_fit_or_predict(self, fail_at):
        """Whatever the call that fails, the entity is named."""
        X_train, y_train, X_test = short_entity_panel({'A': 2, 'B': 12})
        with pytest.raises(ValueError, match=SHORT_MESSAGE) as error:
            SktimeAdapter(ShortHistoryForecaster(fail_at)).fit(X_train, y_train).predict(X_test)
        assert 'history too short' in str(error.value.__cause__)

    def test_failure_on_incomplete_test_rows(self):
        """Entity by entity, only the failure of a short entity is rewritten."""
        X_train, y_train, X_test = short_entity_panel({'A': 2, 'B': 12})
        model = SktimeAdapter(ShortHistoryForecaster()).fit(X_train, y_train)
        with pytest.raises(ValueError, match=SHORT_MESSAGE):
            model.predict(X_test.iloc[1:])
        # Lignes de la seule entité longue : prévision de sa dernière cible
        np.testing.assert_allclose(model.predict(X_test.loc[['B']].iloc[1:]), [12.0])

    def test_forecaster_not_needing_the_frequency(self):
        """``YfromX`` predicts an entity with a single training date."""
        X_train, y_train, X_test = short_entity_panel({'A': 1, 'B': 12})
        predictions = SktimeAdapter(YfromX(LinearRegression())).fit(X_train, y_train).predict(X_test)
        # Une observation pour A : régression constante ; droite exacte pour B
        np.testing.assert_allclose(predictions, [12.0, 12.0, 13.0, 14.0])

    def test_short_entity_that_sktime_handles(self):
        """No row is refused when sktime reads the frequency from the other entities."""
        X_train, y_train, X_test = short_entity_panel({'A': 12, 'B': 2})
        predictions = SktimeAdapter(NaiveForecaster()).fit(X_train, y_train).predict(X_test)
        np.testing.assert_allclose(predictions, np.repeat(12.0, 4))

    def test_holes_declared_missing_count_as_dates(self):
        """Two rows spanning more grid dates, passed as missing values, do not make an entity short."""
        X_train, y_train, X_test = short_entity_panel({'A': 12, 'B': 12})
        kept = ~(entity_mask(X_train, 'A') & ~np.isin(X_train['x1'], [6.0, 11.0]))
        with pytest.raises(ValueError, match='unknown strategy|strategy') as error:
            SktimeAdapter(NaiveForecaster(strategy='median')).fit(X_train[kept], y_train[kept])
        assert 'training dates' not in str(error.value)

    def test_other_errors_are_not_rewritten(self):
        """Without a short entity, the error of sktime propagates unchanged."""
        X_train, y_train, _ = short_entity_panel({'A': 12, 'B': 12})
        with pytest.raises(ValueError) as error:
            SktimeAdapter(NaiveForecaster(strategy='median')).fit(X_train, y_train)
        assert 'training dates' not in str(error.value) and error.value.__cause__ is None

