"""Unit tests of ``tsforecast.adapters.foundation.backends.timestral``: ``TimestralBackend`` and ``load_timestral``.

Offline, with a simulated ``timestral`` package: loading arguments, capabilities
read on the forecaster, one ``forecast`` call per request (series padded on the
left, known covariates over context and horizon with ``NaN`` on the gap, past
covariates, several targets), levels of the model and their interpolation by the
adapter, missing package. With ``mistralai/Timestral-5M`` on CPU (marked ``slow``
and ``network``, skipped without timestral or network): unit rules of the adapter
contract, quantiles equal to ``TimestralForecaster.forecast`` called directly on
the same context (series, balanced and unbalanced panels, known, past and no
covariates, several targets, options, several decoding blocks), later covariates
never read, row alone equal to the batch.
"""
import sys
import types
from functools import partial

import numpy as np
import pandas as pd
import pytest

from tests.support.adapter_cases import TIMESTRAL_5M, foundation_timestral_case
from tests.support.adapter_contract import AdapterContractTests, split_by_date
from tests.support.datasets import build_adapter_panel, build_adapter_series
from tests.support.native_references import foundation_quantiles, timestral_forecaster, timestral_row_quantiles
from tsforecast.adapters import FoundationModelAdapter
from tsforecast.adapters.foundation import ForecastRequest
from tsforecast.adapters.foundation.backends.timestral import TimestralBackend, load_timestral

# Niveaux du modèle simulé
FAKE_LEVELS = (0.1, 0.5, 0.9)
# Identifiant servi par le paquet simulé
FAKE_ID = 'mistralai/Timestral-test'


# ---------------------------------------------------------------------------
# Paquet simulé
# ---------------------------------------------------------------------------


class FakeForecaster:
    """Simulated ``TimestralForecaster``: last target value plus the level, calls recorded."""

    quantile_levels = FAKE_LEVELS

    def __init__(self, covariates=True, known=True, max_context=100):
        self.supports_covariates = covariates
        self.supports_known_covariates = known
        self.max_context = max_context
        self.calls = []

    def forecast(self, context, prediction_length, past_covariates=None, known_covariates=None, **kwargs):
        self.calls.append({'context': context, 'prediction_length': prediction_length,
                           'past': past_covariates, 'known': known_covariates, 'kwargs': kwargs})
        shape = context.shape[:2] + (prediction_length, len(self.quantile_levels))
        quantiles = context[:, :, -1, None, None] + np.array(self.quantile_levels)
        return types.SimpleNamespace(quantiles=np.broadcast_to(quantiles, shape))


@pytest.fixture
def fake_timestral(monkeypatch, isolated_registry):
    """Simulated ``timestral`` package; returns the list of loaded forecasters.

    A model id containing ``plain`` has no covariates, ``past-only`` no known covariates.
    """
    loaded = []

    def from_pretrained(model_id, device, **kwargs):
        forecaster = FakeForecaster(
            covariates='plain' not in model_id,
            known='past-only' not in model_id,
            max_context=kwargs.get('max_context', 100),
        )
        forecaster.load_args = (model_id, device, kwargs)
        loaded.append(forecaster)
        return forecaster

    timestral = types.ModuleType('timestral')
    timestral.TimestralForecaster = types.SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, 'timestral', timestral)
    return loaded


# ---------------------------------------------------------------------------
# Correspondance hors ligne
# ---------------------------------------------------------------------------


class TestMapping:
    """Requests translated into the arrays of ``TimestralForecaster.forecast``."""

    def test_loading_arguments_and_capabilities(self, fake_timestral):
        X, y = build_adapter_series(n_dates=8)
        adapter = FoundationModelAdapter(
            FAKE_ID, device='cuda', backend_kwargs={'revision': 'abc', 'max_context': 5}
        ).fit(X, y)
        assert fake_timestral[0].load_args == (FAKE_ID, 'cuda', {'revision': 'abc', 'max_context': 5})
        assert adapter.backend_name_ == 'timestral'
        capabilities = adapter.capabilities_
        assert (capabilities.past_covariates, capabilities.known_covariates, capabilities.missing_values,
                capabilities.multivariate) == (True, True, True, True)
        assert capabilities.max_context == 5 and capabilities.max_prediction_length is None
        assert capabilities.quantile_levels == FAKE_LEVELS and not capabilities.interpolates_quantiles
        assert adapter.covariates_ == 'known' and adapter.context_[0].shape == (1, 5)

    @pytest.mark.parametrize(
        'model_id, mode', [('mistralai/Timestral-plain', None), ('mistralai/Timestral-past-only', 'past')]
    )
    def test_covariates_of_the_checkpoint(self, fake_timestral, model_id, mode):
        X, y = build_adapter_series(n_dates=8)
        assert FoundationModelAdapter(model_id).fit(X, y).covariates_ == mode

    def test_known_covariates_arrays_per_step(self, fake_timestral):
        X, y = build_adapter_series(n_dates=12)
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=2, gap=2)
        adapter = FoundationModelAdapter(FAKE_ID, predict_kwargs={'symmetric_averaging': False})
        predictions = adapter.fit(X_train, y_train).predict(X_test)
        calls = fake_timestral[0].calls
        # Un appel par nombre de pas : 3 puis 4
        assert [call['prediction_length'] for call in calls] == [3, 4]
        assert all(call['kwargs'] == {'symmetric_averaging': False} for call in calls)
        length = len(X_train)
        for call, k in zip(calls, (3, 4)):
            np.testing.assert_array_equal(call['context'], y_train.to_numpy()[None, None])
            assert call['past'] is None and call['known'].shape == (1, 2, length + k)
            np.testing.assert_array_equal(call['known'][0, :, :length], X_train.to_numpy().T)
            # Pas du gap inconnus, lignes de test aux pas 3 et 4, rien au-delà de k
            assert np.isnan(call['known'][0, :, length:length + 2]).all()
            np.testing.assert_array_equal(call['known'][0, :, length + 2:], X_test.to_numpy()[: k - 2].T)
        np.testing.assert_allclose(predictions, y_train.iloc[-1] + 0.5)

    def test_series_of_unequal_lengths_padded_in_one_call(self, fake_timestral):
        X, y = build_adapter_panel(balanced=False, n_dates=12)
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=1)
        predictions = FoundationModelAdapter(FAKE_ID).fit(X_train, y_train).predict(X_test)
        (call,) = fake_timestral[0].calls
        # Entités A, B, C de 11, 5 et 8 dates, alignées sur la dernière
        assert call['context'].shape == (3, 1, 11) and call['known'].shape == (3, 2, 12)
        assert np.isnan(call['context']).sum(axis=(1, 2)).tolist() == [0, 6, 3]
        for position, entity in enumerate(['A', 'B', 'C']):
            target = y_train.xs(entity).to_numpy()
            start = 11 - len(target)
            np.testing.assert_array_equal(call['context'][position, 0, start:], target)
            assert np.isnan(call['known'][position, :, :start]).all()
            np.testing.assert_array_equal(call['known'][position, :, start:11], X_train.xs(entity).to_numpy().T)
            np.testing.assert_array_equal(call['known'][position, :, 11], X_test.xs(entity).to_numpy()[0])
        np.testing.assert_allclose(predictions, y_train.groupby(level='entity').last().to_numpy() + 0.5)

    def test_past_covariates_and_no_covariates(self, fake_timestral):
        X, y = build_adapter_series(n_dates=8)
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=1)
        FoundationModelAdapter(FAKE_ID, covariates='past').fit(X_train, y_train).predict(X_test)
        FoundationModelAdapter(FAKE_ID, covariates=None).fit(X_train, y_train).predict(X_test)
        past, plain = fake_timestral[0].calls
        np.testing.assert_array_equal(past['past'], X_train.to_numpy().T[None])
        assert past['known'] is None
        assert plain['past'] is None and plain['known'] is None

    def test_several_targets_jointly(self, fake_timestral):
        X, y = build_adapter_series(n_dates=8)
        Y = pd.DataFrame({'a': y, 'b': -y})
        X_train, Y_train, X_test, _ = split_by_date(X, Y, n_test=1)
        predictions = FoundationModelAdapter(FAKE_ID).fit(X_train, Y_train).predict(X_test)
        (call,) = fake_timestral[0].calls
        np.testing.assert_array_equal(call['context'], Y_train.to_numpy().T[None])
        np.testing.assert_allclose(predictions, Y_train.to_numpy()[-1:] + 0.5)

    def test_levels_between_those_of_the_model_interpolated(self, fake_timestral):
        X, y = build_adapter_series(n_dates=8)
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=1)
        adapter = FoundationModelAdapter(FAKE_ID).fit(X_train, y_train)
        frame = adapter.predict_quantiles(X_test, quantile_levels=[0.9, 0.3])
        np.testing.assert_allclose(frame.to_numpy(), y_train.iloc[-1] + np.array([[0.9, 0.3]]))

    def test_backend_predict_on_a_request(self, fake_timestral):
        backend = load_timestral(FAKE_ID, device='cpu')
        request = ForecastRequest(
            context=(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0, 5.0]])),
            prediction_length=2,
            quantile_levels=(0.1, 0.9),
            past_covariates=(np.array([[7.0, 8.0]]), np.array([[7.0, 8.0, 9.0]])),
            past_covariate_names=('p',),
            known_covariates=(np.array([[0.0, 1.0, 2.0, 3.0]]), np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])),
            known_covariate_names=('c',),
        )
        result = backend.predict(request)
        assert result.quantiles.shape == (2, 1, 2, 2)
        np.testing.assert_allclose(result.quantiles[:, 0, 0, :], [[2.1, 2.9], [5.1, 5.9]])
        (call,) = fake_timestral[0].calls
        np.testing.assert_array_equal(call['context'], [[[np.nan, 1.0, 2.0]], [[3.0, 4.0, 5.0]]])
        np.testing.assert_array_equal(call['past'], [[[np.nan, 7.0, 8.0]], [[7.0, 8.0, 9.0]]])
        np.testing.assert_array_equal(
            call['known'], [[[np.nan, 0.0, 1.0, 2.0, 3.0]], [[0.0, 1.0, 2.0, 3.0, 4.0]]]
        )

    def test_level_foreign_to_the_model_refused(self):
        request = ForecastRequest(context=(np.array([[1.0]]),), prediction_length=1, quantile_levels=(0.25,))
        with pytest.raises(ValueError, match='own quantile levels'):
            TimestralBackend(FakeForecaster()).predict(request)


class TestLoading:
    """Missing package."""

    def test_missing_package_names_the_extra(self, block_package, isolated_registry):
        block_package('timestral')
        X, y = build_adapter_series(n_dates=6)
        with pytest.raises(ImportError, match='adapters-timestral'):
            FoundationModelAdapter(TIMESTRAL_5M).fit(X, y)
        with pytest.raises(ImportError, match='adapters-timestral'):
            load_timestral(TIMESTRAL_5M, device='cpu')


# ---------------------------------------------------------------------------
# Modèle réel
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.usefixtures('timestral_5m')
class TestTimestralContract(AdapterContractTests):
    """Unit contract of the adapter around Timestral-5M, with known covariates."""

    case = foundation_timestral_case()


# Niveaux comparés, tous appris par le modèle
REAL_LEVELS = (0.01, 0.1, 0.5, 0.9, 0.99)
# Tolérances des comparaisons en float32, lots de séries remplies compris
RTOL, ATOL = 1e-5, 1e-5


def _native_quantiles(X_train, y_train, X_test, gap, covariates, levels=REAL_LEVELS):
    """Quantiles of every test row from the forecaster called directly, one series per call."""
    n_dates = X_test.index.get_level_values(-1).nunique()
    rows = partial(timestral_row_quantiles, timestral_forecaster(TIMESTRAL_5M))
    fh = tuple(range(gap + 1, gap + n_dates + 1))
    return foundation_quantiles(rows, X_train, y_train, X_test, fh, levels, covariates)


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.usefixtures('timestral_5m')
class TestNativeEquivalence:
    """The adapter gives the quantiles of ``TimestralForecaster`` called directly on the same context."""

    @pytest.mark.parametrize('kind', ['series', 'unbalanced_panel'])
    @pytest.mark.parametrize('gap', [0, 2])
    @pytest.mark.parametrize('covariates', ['known', 'past', None])
    def test_quantiles(self, kind, gap, covariates):
        X, y = build_adapter_series() if kind == 'series' else build_adapter_panel(balanced=False)
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=gap)
        adapter = FoundationModelAdapter(TIMESTRAL_5M, covariates=covariates).fit(X_train, y_train)
        frame = adapter.predict_quantiles(X_test, quantile_levels=REAL_LEVELS)
        expected = _native_quantiles(X_train, y_train, X_test, gap, covariates)
        np.testing.assert_allclose(frame.to_numpy(), expected, rtol=RTOL, atol=ATOL)

    def test_one_series_per_call(self):
        X, y = build_adapter_panel(balanced=False)
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=1)
        adapter = FoundationModelAdapter(TIMESTRAL_5M, batch_size=1).fit(X_train, y_train)
        frame = adapter.predict_quantiles(X_test, quantile_levels=REAL_LEVELS)
        expected = _native_quantiles(X_train, y_train, X_test, 1, 'known')
        np.testing.assert_allclose(frame.to_numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_level_interpolated_between_those_of_the_model(self):
        X, y = build_adapter_series()
        X_train, y_train, X_test, _ = split_by_date(X, y)
        adapter = FoundationModelAdapter(TIMESTRAL_5M).fit(X_train, y_train)
        frame = adapter.predict_quantiles(X_test, quantile_levels=[0.125])
        native = _native_quantiles(X_train, y_train, X_test, 0, 'known', levels=(0.1, 0.15))
        np.testing.assert_allclose(frame.to_numpy()[:, 0], native.mean(axis=1), rtol=RTOL, atol=ATOL)

    def test_several_targets_forecast_jointly(self):
        X, y = build_adapter_series()
        Y = pd.DataFrame({'a': y, 'b': y ** 2})
        X_train, Y_train, X_test, _ = split_by_date(X, Y, n_test=1)
        adapter = FoundationModelAdapter(TIMESTRAL_5M, covariates=None).fit(X_train, Y_train)
        forecast = timestral_forecaster(TIMESTRAL_5M).forecast(np.array(Y_train.to_numpy().T[None]), 1)
        np.testing.assert_allclose(adapter.predict(X_test), forecast.median[:, :, 0].numpy(), rtol=RTOL, atol=ATOL)

    def test_options_of_the_package(self):
        X, y = build_adapter_series()
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=1)
        options = {'symmetric_averaging': False, 'clip_nonnegative': False}
        adapter = FoundationModelAdapter(TIMESTRAL_5M, covariates=None, predict_kwargs=options)
        predictions = adapter.fit(X_train, y_train).predict(X_test)
        forecaster = timestral_forecaster(TIMESTRAL_5M)
        context = np.array(y_train.to_numpy()[None, None])
        expected = forecaster.forecast(context, 1, **options).median[0, 0].numpy()
        np.testing.assert_allclose(predictions, expected, rtol=RTOL, atol=ATOL)
        assert not np.allclose(predictions, forecaster.forecast(context, 1).median[0, 0].numpy(), rtol=1e-4)

    def test_several_decoding_blocks(self, timestral_5m):
        # 1 100 pas : deux blocs, le second lisant les médianes du premier
        rng = np.random.default_rng(0)
        context = tuple(rng.standard_normal((1, length)).cumsum(axis=1) for length in (64, 20))
        request = ForecastRequest(context=context, prediction_length=1100, quantile_levels=(0.1, 0.5))
        quantiles = timestral_5m.predict(request).quantiles
        forecaster = timestral_forecaster(TIMESTRAL_5M)
        # Chaque série comparée à sa prévision seule
        for position, series in enumerate(context):
            alone = forecaster.forecast(np.array(series[None]), 1100)
            expected = np.stack([alone.quantile(level)[0, 0].numpy() for level in (0.1, 0.5)], axis=-1)
            np.testing.assert_allclose(quantiles[position, 0], expected, rtol=1e-4, atol=1e-4)


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.usefixtures('timestral_5m')
class TestRealForecastProperties:
    """Properties of the forecasts of the real model through the adapter."""

    def test_later_covariates_never_read(self):
        X, y = build_adapter_panel(balanced=False)
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=2)
        adapter = FoundationModelAdapter(TIMESTRAL_5M).fit(X_train, y_train)
        reference = adapter.predict(X_test)
        # Covariables de la dernière date de test altérées : lignes antérieures inchangées
        last = np.asarray(X_test.index.get_level_values(-1) == X_test.index.get_level_values(-1).max())
        altered = X_test.mask(np.broadcast_to(last[:, None], X_test.shape), X_test + 10.0)
        predictions = adapter.predict(altered)
        np.testing.assert_array_equal(predictions[~last], reference[~last])
        assert not np.allclose(predictions[last], reference[last])

    def test_row_alone_equals_batch_without_known_covariates(self):
        X, y = build_adapter_panel(balanced=False)
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=2)
        adapter = FoundationModelAdapter(TIMESTRAL_5M, covariates=None).fit(X_train, y_train)
        batch = adapter.predict(X_test)
        alone = np.array([adapter.predict(X_test.iloc[[row]])[0] for row in range(len(X_test))])
        np.testing.assert_allclose(alone, batch, rtol=RTOL, atol=ATOL)
