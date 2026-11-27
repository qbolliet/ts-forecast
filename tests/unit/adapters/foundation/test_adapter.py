"""Unit tests of ``tsforecast.adapters.foundation.adapter.FoundationModelAdapter``.

Runs the unit rules of the adapter contract on the fake ``DriftBackend``, then
targeted scenarios with values computed by hand: number of steps read from the
index through a gap, unbalanced panels, one forecast per number of steps, known
covariates (placement, unknown gap steps, earlier rows read, later rows never
read), past covariates, automatic and explicit covariate modes, training holes,
context length, quantile levels (order, range, interpolation by the adapter),
several targets, batches, canonical entity order, read-only requests, parameter
and backend errors.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from tests.support.adapter_cases import foundation_drift_case
from tests.support.adapter_contract import AdapterContractTests
from tests.support.forecast_backends import DriftBackend, RecordingBackend
from tsforecast.adapters import CallableBackend, FoundationModelAdapter
from tsforecast.adapters.foundation import ForecastResult

# Calendrier mensuel des jeux construits à la main
DATES = pd.date_range('2024-01-01', periods=12, freq='MS', name='date')


class TestDriftContract(AdapterContractTests):
    """Unit contract of the adapter around the fake backend with known covariates."""

    case = foundation_drift_case()


def make_series(n_train=6, test_offsets=(), x=None):
    """``(X_train, y_train, X_test)``: ``y = 1..n_train``, ``x`` column, test rows ``offset`` steps after training."""
    x = np.arange(12, dtype=float) if x is None else np.asarray(x, dtype=float)
    X = pd.DataFrame({'x': x}, index=DATES)
    y = pd.Series(np.arange(1, 13, dtype=float), index=DATES, name='target')
    test = [n_train - 1 + offset for offset in test_offsets]
    return X.iloc[:n_train], y.iloc[:n_train], X.iloc[test]


def make_panel(starts, ends, test_dates=()):
    """``(X_train, y_train, X_test)`` of entities starting and ending at the given month positions.

    ``y`` is ``100·(entity rank) + month position``, ``x`` the month position.
    """
    rows = [(entity, DATES[position]) for entity in starts for position in range(starts[entity], ends[entity] + 1)]
    index = pd.MultiIndex.from_tuples(rows, names=['entity', 'date'])
    ranks = {entity: rank for rank, entity in enumerate(starts)}
    positions = np.array([DATES.get_loc(date) for _, date in rows], dtype=float)
    X_train = pd.DataFrame({'x': positions}, index=index)
    y_train = pd.Series([100 * ranks[entity] + position for (entity, _), position in zip(rows, positions)],
                        index=index, name='target')
    test_index = pd.MultiIndex.from_tuples(list(test_dates), names=['entity', 'date'])
    X_test = pd.DataFrame({'x': [float(DATES.get_loc(date)) for _, date in test_dates]}, index=test_index)
    return X_train, y_train, X_test


class TestStepsFromIndex:
    """The number of steps of each row is read from the index, through any gap."""

    def test_series_through_a_gap(self):
        X_train, y_train, X_test = make_series(test_offsets=(3, 5))
        adapter = FoundationModelAdapter(backend=DriftBackend(drift=1.0), covariates=None).fit(X_train, y_train)
        # Dernière cible 6, dérive 1 par pas : 6 + 3 et 6 + 5
        np.testing.assert_allclose(adapter.predict(X_test), [9.0, 11.0])
        np.testing.assert_allclose(adapter.predict(X_test.iloc[::-1]), [11.0, 9.0])

    def test_unbalanced_panel_with_different_ends(self):
        X_train, y_train, X_test = make_panel(
            {'A': 0, 'B': 2}, {'A': 5, 'B': 3}, [('A', DATES[7]), ('B', DATES[7])]
        )
        backend = RecordingBackend(DriftBackend(drift=1.0))
        adapter = FoundationModelAdapter(backend=backend, covariates=None).fit(X_train, y_train)
        # A finit en juin (5), B en avril (103) : 2 et 4 pas jusqu'en août
        np.testing.assert_allclose(adapter.predict(X_test), [5.0 + 2, 103.0 + 4])
        assert [request.prediction_length for request in backend.requests] == [2, 4]
        assert [request.context[0].shape[1] for request in backend.requests] == [6, 2]

    def test_one_forecast_per_number_of_steps(self):
        X_train, y_train, X_test = make_series(test_offsets=(1, 2, 4))
        backend = RecordingBackend(DriftBackend())
        FoundationModelAdapter(backend=backend).fit(X_train, y_train).predict(X_test)
        assert [request.prediction_length for request in backend.requests] == [1, 2, 4]
        assert all(request.n_series == 1 for request in backend.requests)

    def test_rows_of_two_entity_levels(self):
        index = pd.MultiIndex.from_product([['FR', 'DE'], ['a'], DATES[:4]], names=['country', 'shop', 'date'])
        X = pd.DataFrame({'x': np.zeros(8)}, index=index)
        y = pd.Series(np.r_[np.arange(4.0), 10 + np.arange(4.0)], index=index)
        adapter = FoundationModelAdapter(backend=DriftBackend(drift=1.0), covariates=None).fit(X.iloc[[0, 1, 2, 4, 5, 6]], y.iloc[[0, 1, 2, 4, 5, 6]])
        np.testing.assert_allclose(adapter.predict(X.iloc[[3, 7]]), [3.0, 13.0])


class TestKnownCovariates:
    """Known covariates: training values, then the rows of X at most k steps after training."""

    def test_request_layout_and_values(self):
        X_train, y_train, X_test = make_series(n_train=4, test_offsets=(2, 3), x=np.arange(12) * 10.0)
        backend = RecordingBackend(DriftBackend(drift=0.0, coef=1.0))
        adapter = FoundationModelAdapter(backend=backend).fit(X_train, y_train)
        assert adapter.covariates_ == 'known'
        predictions = adapter.predict(X_test)
        first, second = backend.requests
        assert first.known_covariate_names == ('x',) and first.past_covariates is None
        # Passé (0, 10, 20, 30), pas 1 du gap inconnu, puis les lignes de test (50, 60)
        np.testing.assert_array_equal(first.known_covariates[0], [[0.0, 10.0, 20.0, 30.0, np.nan, 50.0]])
        np.testing.assert_array_equal(second.known_covariates[0], [[0.0, 10.0, 20.0, 30.0, np.nan, 50.0, 60.0]])
        # Dernière cible 4, plus la somme des covariables reçues après le contexte
        np.testing.assert_allclose(predictions, [4.0 + 50.0, 4.0 + 50.0 + 60.0])

    def test_later_rows_never_read(self):
        X_train, y_train, X_test = make_series(test_offsets=(1, 2, 3))
        adapter = FoundationModelAdapter(backend=DriftBackend(coef=1.0)).fit(X_train, y_train)
        reference = adapter.predict(X_test)
        altered = X_test.copy()
        altered.iloc[2, 0] += 1000.0
        predictions = adapter.predict(altered)
        np.testing.assert_array_equal(predictions[:2], reference[:2])
        assert predictions[2] == reference[2] + 1000.0

    def test_earlier_rows_are_read(self):
        X_train, y_train, X_test = make_series(n_train=4, test_offsets=(1, 2), x=np.arange(12) * 10.0)
        adapter = FoundationModelAdapter(backend=DriftBackend(drift=0.0, coef=1.0)).fit(X_train, y_train)
        # Ligne à 2 pas : covariables des pas 1 (40, ligne antérieure présente) et 2 (50)
        np.testing.assert_allclose(adapter.predict(X_test), [4.0 + 40.0, 4.0 + 40.0 + 50.0])
        # Seule : pas 1 inconnu
        np.testing.assert_allclose(adapter.predict(X_test.iloc[[1]]), [4.0 + 50.0])

    def test_missing_value_refused_by_a_model_without_missing_values(self):
        X_train, y_train, X_test = make_series(test_offsets=(2,))
        adapter = FoundationModelAdapter(backend=DriftBackend(missing_values=False)).fit(X_train, y_train)
        with pytest.raises(ValueError, match='no value is invented'):
            adapter.predict(X_test)
        # Sans gap : aucune valeur manquante
        _, _, X_next = make_series(test_offsets=(1,))
        assert np.isfinite(adapter.predict(X_next)).all()

    def test_missing_values_of_X_are_unknown_steps(self):
        X_train, y_train, X_test = make_series(n_train=4, test_offsets=(1,), x=np.arange(12) * 10.0)
        backend = RecordingBackend(DriftBackend(drift=0.0, coef=1.0))
        adapter = FoundationModelAdapter(backend=backend).fit(X_train, y_train)
        np.testing.assert_allclose(adapter.predict(X_test.assign(x=np.nan)), [4.0])
        assert np.isnan(backend.requests[0].known_covariates[0][0, -1])


class TestCovariateModes:
    """Automatic and explicit use of the columns of X."""

    @pytest.mark.parametrize(
        'capabilities, expected',
        [
            ({}, 'known'),
            ({'known_covariates': False}, 'past'),
            ({'known_covariates': False, 'past_covariates': False}, None),
        ],
    )
    def test_auto(self, capabilities, expected):
        X_train, y_train, _ = make_series()
        adapter = FoundationModelAdapter(backend=DriftBackend(**capabilities)).fit(X_train, y_train)
        assert adapter.covariates_ == expected
        assert (adapter.context_covariates_ is None) == (expected is None)

    @pytest.mark.parametrize('mode, capability', [('known', 'known_covariates'), ('past', 'past_covariates')])
    def test_explicit_mode_unsupported(self, mode, capability):
        X_train, y_train, _ = make_series()
        adapter = FoundationModelAdapter(backend=DriftBackend(**{capability: False}), covariates=mode)
        with pytest.raises(ValueError, match=f'does not accept {mode} covariates'):
            adapter.fit(X_train, y_train)

    def test_past_covariates_ignore_the_rows_to_predict(self):
        X_train, y_train, X_test = make_series(test_offsets=(2,), x=np.arange(12) * 10.0)
        backend = RecordingBackend(DriftBackend(drift=0.0, past_coef=1.0))
        adapter = FoundationModelAdapter(backend=backend, covariates='past').fit(X_train, y_train)
        # Dernière cible 6, plus la covariable de la dernière date d'entraînement (50)
        np.testing.assert_allclose(adapter.predict(X_test), [56.0])
        np.testing.assert_allclose(adapter.predict(X_test + 1000.0), [56.0])
        request = backend.requests[0]
        assert request.known_covariates is None and request.past_covariate_names == ('x',)

    def test_no_covariates(self):
        X_train, y_train, X_test = make_series(test_offsets=(1,))
        backend = RecordingBackend(DriftBackend(coef=1.0))
        FoundationModelAdapter(backend=backend, covariates=None).fit(X_train, y_train).predict(X_test)
        assert backend.requests[0].known_covariates is None and backend.requests[0].past_covariates is None

    def test_no_column(self):
        X_train, y_train, X_test = make_series(test_offsets=(1,))
        adapter = FoundationModelAdapter(backend=DriftBackend()).fit(X_train[[]], y_train)
        assert adapter.covariates_ is None and adapter.n_features_in_ == 0
        np.testing.assert_allclose(adapter.predict(X_test[[]]), [6.5])

    def test_non_numeric_covariates(self):
        X_train, y_train, _ = make_series()
        X_text = X_train.assign(x=list('abcdef'))
        with pytest.raises(ValueError, match='numeric'):
            FoundationModelAdapter(backend=DriftBackend()).fit(X_text, y_train)
        FoundationModelAdapter(backend=DriftBackend(), covariates=None).fit(X_text, y_train)


class TestContext:
    """Contexts on the grid of each entity: holes, length, observed values."""

    def test_holes_are_missing_values(self):
        X_train, y_train, X_test = make_series(test_offsets=(1,))
        kept = [0, 1, 3, 4, 5]
        adapter = FoundationModelAdapter(backend=DriftBackend(), covariates=None).fit(X_train.iloc[kept], y_train.iloc[kept])
        np.testing.assert_array_equal(adapter.context_[0], [[1.0, 2.0, np.nan, 4.0, 5.0, 6.0]])
        with pytest.raises(ValueError, match='does not accept missing values'):
            FoundationModelAdapter(backend=DriftBackend(missing_values=False)).fit(X_train.iloc[kept], y_train.iloc[kept])
        # Cible complète, covariable manquante
        with pytest.raises(ValueError, match='covariates have missing values'):
            FoundationModelAdapter(backend=DriftBackend(missing_values=False)).fit(X_train.assign(x=np.nan), y_train)
        FoundationModelAdapter(backend=DriftBackend(missing_values=False), covariates=None).fit(X_train.assign(x=np.nan), y_train)

    def test_missing_target_values(self):
        X_train, y_train, X_test = make_series(test_offsets=(1,))
        y_holes = y_train.mask(y_train.index == DATES[5])
        adapter = FoundationModelAdapter(backend=DriftBackend(drift=1.0), covariates=None).fit(X_train, y_holes)
        # Dernière valeur observée 5, toujours un pas après la dernière date d'entraînement
        np.testing.assert_allclose(adapter.predict(X_test), [6.0])

    def test_context_length_and_model_limit(self):
        X_train, y_train, _ = make_series()
        adapter = FoundationModelAdapter(backend=DriftBackend(), context_length=3).fit(X_train, y_train)
        assert adapter.context_length_ == 3
        np.testing.assert_array_equal(adapter.context_[0], [[4.0, 5.0, 6.0]])
        assert adapter.context_covariates_[0].shape == (1, 3)
        limited = FoundationModelAdapter(backend=DriftBackend(max_context=2), context_length=3).fit(X_train, y_train)
        assert limited.context_length_ == 2 and limited.context_[0].shape == (1, 2)
        assert FoundationModelAdapter(backend=DriftBackend()).fit(X_train, y_train).context_length_ is None

    def test_entity_without_observed_target(self):
        X_train, y_train, _ = make_panel({'A': 0, 'B': 0}, {'A': 3, 'B': 3})
        y_empty = y_train.mask(y_train.index.get_level_values('entity') == 'B')
        with pytest.raises(ValueError, match="entities \\('B',\\)"):
            FoundationModelAdapter(backend=DriftBackend()).fit(X_train, y_empty)
        # Valeur observée hors du contexte gardé
        y_old = y_train.mask((y_train.index.get_level_values('entity') == 'B') & (y_train.index.get_level_values('date') > DATES[0]))
        with pytest.raises(ValueError, match='no observed value'):
            FoundationModelAdapter(backend=DriftBackend(), context_length=2).fit(X_train, y_old)

    def test_irregular_grid_and_infinite_values(self):
        X_train, y_train, _ = make_series()
        irregular = [0, 1, 4]
        X_irregular = X_train.iloc[irregular].set_axis(pd.DatetimeIndex(['2024-01-01 00:00', '2024-01-01 07:13', '2024-01-03 01:00'], name='date'))
        with pytest.raises(ValueError, match='regular grid'):
            FoundationModelAdapter(backend=DriftBackend()).fit(X_irregular, y_train.iloc[irregular].set_axis(X_irregular.index))
        with pytest.raises(ValueError, match='infinite'):
            FoundationModelAdapter(backend=DriftBackend()).fit(X_train, y_train.replace(6.0, np.inf))
        with pytest.raises(ValueError, match='infinite'):
            FoundationModelAdapter(backend=DriftBackend()).fit(X_train.replace(5.0, np.inf), y_train)


class TestPredictionErrors:
    """Rows that cannot be forecast raise explicit errors."""

    def test_rows_not_after_training(self):
        X_train, y_train, _ = make_series()
        adapter = FoundationModelAdapter(backend=DriftBackend()).fit(X_train, y_train)
        with pytest.raises(ValueError, match='at or before the last training date'):
            adapter.predict(X_train.iloc[-2:])

    def test_unknown_entity_and_off_grid_date(self):
        X_train, y_train, _ = make_panel({'A': 0}, {'A': 3})
        adapter = FoundationModelAdapter(backend=DriftBackend()).fit(X_train, y_train)
        unknown = pd.DataFrame({'x': [0.0]}, index=pd.MultiIndex.from_tuples([('Z', DATES[5])], names=['entity', 'date']))
        with pytest.raises(ValueError, match='unknown at fit'):
            adapter.predict(unknown)
        off_grid = pd.DataFrame({'x': [0.0]}, index=pd.MultiIndex.from_tuples([('A', pd.Timestamp('2024-06-15'))], names=['entity', 'date']))
        with pytest.raises(ValueError, match='not on the'):
            adapter.predict(off_grid)

    def test_longest_forecast_of_the_model(self):
        X_train, y_train, X_test = make_series(test_offsets=(4,))
        adapter = FoundationModelAdapter(backend=DriftBackend(max_prediction_length=3)).fit(X_train, y_train)
        with pytest.raises(ValueError, match='at most 3 steps'):
            adapter.predict(X_test)

    def test_empty_X_and_not_fitted(self):
        X_train, y_train, _ = make_series()
        adapter = FoundationModelAdapter(backend=DriftBackend())
        with pytest.raises(NotFittedError):
            adapter.predict(X_train)
        adapter.fit(X_train, y_train)
        assert adapter.predict(X_train.iloc[:0]).shape == (0,)
        assert adapter.predict_quantiles(X_train.iloc[:0]).shape == (0, 3)

    def test_other_columns(self):
        X_train, y_train, X_test = make_series(test_offsets=(1,))
        adapter = FoundationModelAdapter(backend=DriftBackend()).fit(X_train, y_train)
        with pytest.raises(ValueError):
            adapter.predict(X_test.rename(columns={'x': 'z'}))


class TestQuantiles:
    """Quantile levels: order, range, interpolation between native levels."""

    def test_frame_and_level_order(self):
        X_train, y_train, X_test = make_series(test_offsets=(1, 2))
        adapter = FoundationModelAdapter(backend=DriftBackend(drift=1.0, spread=10.0), covariates=None).fit(X_train, y_train)
        frame = adapter.predict_quantiles(X_test, quantile_levels=[0.9, 0.1, 0.5])
        assert frame.index.equals(X_test.index)
        assert frame.columns.tolist() == [0.9, 0.1, 0.5] and frame.columns.name == 'quantile'
        assert (frame.dtypes == 'float64').all()
        np.testing.assert_allclose(frame.to_numpy(), [[11.0, 3.0, 7.0], [12.0, 4.0, 8.0]])
        np.testing.assert_allclose(adapter.predict(X_test), frame[0.5].to_numpy())

    def test_quantile_parameter(self):
        X_train, y_train, X_test = make_series(test_offsets=(1,))
        adapter = FoundationModelAdapter(backend=DriftBackend(drift=1.0, spread=10.0), quantile=0.8, covariates=None)
        np.testing.assert_allclose(adapter.fit(X_train, y_train).predict(X_test), [10.0])

    @pytest.mark.parametrize('levels', [[], [0.0], [1.0], [0.5, 0.5], ['a']])
    def test_invalid_levels(self, levels):
        X_train, y_train, X_test = make_series(test_offsets=(1,))
        adapter = FoundationModelAdapter(backend=DriftBackend()).fit(X_train, y_train)
        with pytest.raises(ValueError):
            adapter.predict_quantiles(X_test, quantile_levels=levels)

    def test_range_of_native_levels(self):
        X_train, y_train, X_test = make_series(test_offsets=(1,))
        backend = DriftBackend(quantile_levels=(0.1, 0.5, 0.9))
        adapter = FoundationModelAdapter(backend=backend).fit(X_train, y_train)
        with pytest.raises(ValueError, match='outside the range'):
            adapter.predict_quantiles(X_test, quantile_levels=[0.05])
        with pytest.raises(ValueError, match='outside the range'):
            FoundationModelAdapter(backend=backend, quantile=0.95).fit(X_train, y_train)

    def test_interpolation_by_the_adapter(self):
        X_train, y_train, X_test = make_series(test_offsets=(1,))
        backend = RecordingBackend(DriftBackend(drift=1.0, spread=10.0, quantile_levels=(0.1, 0.5, 0.9),
                                                interpolates_quantiles=False))
        adapter = FoundationModelAdapter(backend=backend, quantile=0.3, covariates=None).fit(X_train, y_train)
        # Modèle linéaire en niveau : l'interpolation linéaire est exacte
        np.testing.assert_allclose(adapter.predict(X_test), [7.0 - 2.0])
        assert backend.requests[-1].quantile_levels == (0.1, 0.5)
        frame = adapter.predict_quantiles(X_test, quantile_levels=[0.9, 0.3])
        np.testing.assert_allclose(frame.to_numpy(), [[11.0, 5.0]])
        assert backend.requests[-1].quantile_levels == (0.1, 0.5, 0.9)


class TestSeveralTargets:
    """Several targets: forecast jointly, or one series per target."""

    @pytest.mark.parametrize('multivariate, n_series', [(True, 1), (False, 2)])
    def test_targets(self, multivariate, n_series):
        X_train, y_train, X_test = make_series(test_offsets=(1, 2))
        Y_train = pd.DataFrame({'a': y_train, 'b': 10 * y_train})
        backend = RecordingBackend(DriftBackend(drift=1.0, multivariate=multivariate))
        adapter = FoundationModelAdapter(backend=backend, covariates=None).fit(X_train, Y_train)
        np.testing.assert_allclose(adapter.predict(X_test), [[7.0, 61.0], [8.0, 62.0]])
        assert [request.n_series for request in backend.requests] == [n_series, n_series]
        frame = adapter.predict_quantiles(X_test, quantile_levels=[0.5])
        assert frame.columns.tolist() == [('a', 0.5), ('b', 0.5)]
        assert frame.columns.names == ['target', 'quantile']
        assert adapter.target_names_ == ['a', 'b']


class TestBatches:
    """Batches of series: same forecasts, canonical entity order, read-only requests."""

    def test_batch_size(self):
        X_train, y_train, X_test = make_panel(
            {'A': 0, 'B': 0, 'C': 0}, {'A': 3, 'B': 3, 'C': 3}, [(entity, DATES[5]) for entity in 'ABC']
        )
        reference = FoundationModelAdapter(backend=DriftBackend(coef=1.0)).fit(X_train, y_train).predict(X_test)
        backend = RecordingBackend(DriftBackend(coef=1.0))
        predictions = FoundationModelAdapter(backend=backend, batch_size=2).fit(X_train, y_train).predict(X_test)
        np.testing.assert_allclose(predictions, reference)
        assert [request.n_series for request in backend.requests] == [2, 1]

    def test_canonical_entity_order(self):
        X_train, y_train, X_test = make_panel(
            {'B': 0, 'A': 0}, {'B': 3, 'A': 3}, [('B', DATES[4]), ('A', DATES[4])]
        )
        backend = RecordingBackend(DriftBackend())
        FoundationModelAdapter(backend=backend).fit(X_train, y_train).predict(X_test)
        # Entités dans l'ordre de leurs étiquettes, quel que soit l'ordre des lignes
        assert [request.context[0][0, -1] for request in backend.requests] == [103.0]
        assert backend.requests[0].context[1][0, -1] == 3.0

    def test_requests_are_read_only(self):
        X_train, y_train, X_test = make_series(test_offsets=(1,))

        def overwrite(request):
            request.context[0][0, -1] = 0.0

        adapter = FoundationModelAdapter(backend=CallableBackend(overwrite)).fit(X_train, y_train)
        with pytest.raises(ValueError, match='read-only'):
            adapter.predict(X_test)
        assert adapter.context_[0][0, -1] == 6.0


class _Backend:
    """Backend object returning the output of a given function of the request."""

    def __init__(self, output, capabilities=None):
        self.output = output
        self.capabilities = DriftBackend().capabilities if capabilities is None else capabilities

    def predict(self, request):
        return self.output(request)


class TestBackendErrors:
    """Invalid parameters, backends and outputs."""

    @pytest.mark.parametrize(
        'params',
        [
            {'quantile': True},
            {'quantile': 'median'},
            {'context_length': 2.5},
            {'batch_size': True},
            {'device': 0},
            {'backend_kwargs': ['revision']},
            {'predict_kwargs': 1},
            {'model_id': 3},
        ],
    )
    def test_invalid_parameters(self, params):
        X_train, y_train, _ = make_series()
        with pytest.raises(ValueError):
            FoundationModelAdapter(backend=DriftBackend()).set_params(**params).fit(X_train, y_train)

    def test_model_and_backend(self):
        X_train, y_train, _ = make_series()
        with pytest.raises(ValueError, match='Give a model_id'):
            FoundationModelAdapter().fit(X_train, y_train)
        with pytest.raises(ValueError, match='model_id must be None'):
            FoundationModelAdapter('amazon/chronos-2', backend=DriftBackend()).fit(X_train, y_train)
        with pytest.raises(TypeError, match='backend must be'):
            FoundationModelAdapter(backend=3).fit(X_train, y_train)
        with pytest.raises(ValueError, match='No registered backend'):
            FoundationModelAdapter('unknown-org/model').fit(X_train, y_train)
        with pytest.raises(TypeError, match='BackendCapabilities'):
            FoundationModelAdapter(backend=_Backend(None, capabilities='all')).fit(X_train, y_train)

    @pytest.mark.parametrize(
        'output, error, match',
        [
            (lambda request: np.zeros((1, 1, 1, 1)), TypeError, 'ForecastResult'),
            (lambda request: ForecastResult(np.zeros((2, 1, 1, 1))), ValueError, 'expected'),
            (lambda request: ForecastResult(np.full((1, 1, 1, 1), np.nan)), ValueError, 'non-finite'),
        ],
    )
    def test_invalid_outputs(self, output, error, match):
        X_train, y_train, X_test = make_series(test_offsets=(1,))
        adapter = FoundationModelAdapter(backend=_Backend(output)).fit(X_train, y_train)
        with pytest.raises(error, match=match):
            adapter.predict(X_test)

    def test_tags(self):
        assert FoundationModelAdapter().__sklearn_tags__().input_tags.allow_nan
