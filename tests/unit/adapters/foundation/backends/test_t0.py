"""Unit tests of ``tsforecast.adapters.foundation.backends.t0``: ``T0Backend`` and ``load_t0``.

Offline, with simulated ``t0`` and ``torch`` packages: loading arguments and
evaluation mode, capabilities, one ``TimeSeries`` per series (targets, past
covariates as further variates whose forecasts are dropped, known covariates over
context and horizon with ``NaN`` on the gap), one call per batch, one call per
context length beyond one forward pass, any quantile level, missing package. With
``theforecastingcompany/t0-gamma`` on CPU (marked ``slow`` and ``network``,
skipped without tfc-t0 or network): unit rules of the adapter contract, quantiles
equal to ``T0Forecaster.predict`` called directly on the same context (series,
balanced and unbalanced panels, known, past and no covariates, several targets,
levels beyond the trained ones, options, beyond one forward pass), later
covariates never read, row alone equal to the batch.
"""
import sys
import types
from functools import partial

import numpy as np
import pandas as pd
import pytest

from tests.support.adapter_cases import T0_GAMMA, foundation_t0_case
from tests.support.adapter_contract import AdapterContractTests, split_by_date
from tests.support.datasets import build_adapter_panel, build_adapter_series
from tests.support.native_references import foundation_quantiles, t0_forecaster, t0_row_quantiles
from tsforecast.adapters import FoundationModelAdapter
from tsforecast.adapters.foundation import ForecastRequest
from tsforecast.adapters.foundation.backends.t0 import T0Backend, load_t0

# Identifiant servi par le paquet simulé
FAKE_ID = 'theforecastingcompany/t0-test'


# ---------------------------------------------------------------------------
# Paquets simulés
# ---------------------------------------------------------------------------


class FakeTimeSeries:
    """Simulated ``t0.TimeSeries``: keeps the arrays it is built from; a batch is their list."""

    def __init__(self, targets, future_covariates):
        self.targets = targets
        self.future_covariates = future_covariates

    @classmethod
    def from_array(cls, targets, future_covariates=None):
        return cls(targets, future_covariates)

    @staticmethod
    def batch(series):
        return list(series)


class FakeForecaster:
    """Simulated ``T0Forecaster``: last value of each variate plus the level, calls recorded."""

    def __init__(self, max_horizon=1024):
        self.max_horizon = max_horizon
        self.device = None
        self.training = True
        self.calls = []

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self

    def predict(self, model_input, horizon, quantile_levels, *, context_length, **kwargs):
        self.calls.append({'series': model_input, 'horizon': horizon, 'levels': quantile_levels,
                           'context_length': context_length, 'kwargs': kwargs})
        # Une ligne de sortie par variable, série après série
        last = np.concatenate([item.targets[0][:, -1] for item in model_input])
        shape = (len(last), horizon, len(quantile_levels))
        quantiles = last[:, None, None] + np.array(quantile_levels)
        return types.SimpleNamespace(quantiles=np.broadcast_to(quantiles, shape))


@pytest.fixture
def fake_t0(monkeypatch, isolated_registry):
    """Simulated ``t0`` and ``torch`` packages; returns the list of loaded forecasters."""
    loaded = []

    def from_pretrained(model_id, **kwargs):
        forecaster = FakeForecaster()
        forecaster.load_args = (model_id, kwargs)
        loaded.append(forecaster)
        return forecaster

    t0 = types.ModuleType('t0')
    t0.T0Forecaster = types.SimpleNamespace(from_pretrained=from_pretrained)
    t0.TimeSeries = FakeTimeSeries
    torch = types.ModuleType('torch')
    torch.float32 = 'float32'
    torch.as_tensor = lambda values, dtype=None: np.asarray(values)
    monkeypatch.setitem(sys.modules, 't0', t0)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    return loaded


# ---------------------------------------------------------------------------
# Correspondance hors ligne
# ---------------------------------------------------------------------------


class TestMapping:
    """Requests translated into the ``TimeSeries`` batches of ``T0Forecaster.predict``."""

    def test_loading_arguments_and_capabilities(self, fake_t0):
        X, y = build_adapter_series(n_dates=8)
        adapter = FoundationModelAdapter(FAKE_ID, device='cuda', backend_kwargs={'revision': 'abc'}).fit(X, y)
        forecaster = fake_t0[0]
        assert forecaster.load_args == (FAKE_ID, {'revision': 'abc'})
        assert forecaster.device == 'cuda' and not forecaster.training
        assert adapter.backend_name_ == 't0'
        capabilities = adapter.capabilities_
        assert (capabilities.past_covariates, capabilities.known_covariates, capabilities.missing_values,
                capabilities.multivariate) == (True, True, True, True)
        assert capabilities.max_context is None and capabilities.max_prediction_length is None
        assert capabilities.quantile_levels is None
        assert adapter.covariates_ == 'known'

    def test_known_covariates_per_step(self, fake_t0):
        X, y = build_adapter_series(n_dates=12)
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=2, gap=2)
        adapter = FoundationModelAdapter(FAKE_ID, predict_kwargs={'snap_integers': True})
        predictions = adapter.fit(X_train, y_train).predict(X_test)
        calls = fake_t0[0].calls
        # Un appel par nombre de pas : 3 puis 4
        assert [call['horizon'] for call in calls] == [3, 4]
        assert all(call['kwargs'] == {'snap_integers': True} and call['levels'] == [0.5] for call in calls)
        length = len(X_train)
        for call, k in zip(calls, (3, 4)):
            assert call['context_length'] == length
            (item,) = call['series']
            np.testing.assert_allclose(item.targets, y_train.to_numpy()[None, None])
            assert item.future_covariates.shape == (1, 2, length + k)
            np.testing.assert_allclose(item.future_covariates[0, :, :length], X_train.to_numpy().T)
            # Pas du gap inconnus, lignes de test aux pas 3 et 4, rien au-delà de k
            assert np.isnan(item.future_covariates[0, :, length:length + 2]).all()
            np.testing.assert_allclose(item.future_covariates[0, :, length + 2:], X_test.to_numpy()[: k - 2].T)
        np.testing.assert_allclose(predictions, y_train.iloc[-1] + 0.5)

    def test_past_covariates_as_variates_of_the_context(self, fake_t0):
        X, y = build_adapter_series(n_dates=8)
        Y = pd.DataFrame({'a': y, 'b': -y})
        X_train, Y_train, X_test, _ = split_by_date(X, Y, n_test=1)
        adapter = FoundationModelAdapter(FAKE_ID, covariates='past').fit(X_train, Y_train)
        predictions = adapter.predict(X_test)
        (item,) = fake_t0[0].calls[0]['series']
        assert item.future_covariates is None
        np.testing.assert_allclose(item.targets, np.concatenate([Y_train.to_numpy().T, X_train.to_numpy().T])[None])
        # Prévisions des covariables passées écartées
        np.testing.assert_allclose(predictions, Y_train.to_numpy()[-1:] + 0.5)

    def test_panel_in_one_batch_without_covariates(self, fake_t0):
        X, y = build_adapter_panel(balanced=False, n_dates=12)
        Y = pd.DataFrame({'a': y, 'b': 2 * y})
        X_train, Y_train, X_test, _ = split_by_date(X, Y, n_test=1)
        predictions = FoundationModelAdapter(FAKE_ID, covariates=None).fit(X_train, Y_train).predict(X_test)
        (call,) = fake_t0[0].calls
        # Entités A, B, C de 11, 5 et 8 dates, chacune à sa longueur
        assert [item.targets.shape for item in call['series']] == [(1, 2, 11), (1, 2, 5), (1, 2, 8)]
        assert all(item.future_covariates is None for item in call['series'])
        assert call['context_length'] == 11
        np.testing.assert_allclose(predictions, Y_train.groupby(level='entity').last().to_numpy() + 0.5)

    def test_one_call_per_length_beyond_one_forward_pass(self, fake_t0):
        forecaster = FakeForecaster(max_horizon=4)
        request = ForecastRequest(
            context=(np.array([[1.0, 2.0, 3.0]]), np.arange(5.0)[None], np.array([[7.0, 8.0, 9.0]])),
            prediction_length=5,
            quantile_levels=(0.25, 0.5),
        )
        result = T0Backend(forecaster).predict(request)
        assert [(len(call['series']), call['context_length']) for call in forecaster.calls] == [(2, 3), (1, 5)]
        # Résultats rangés dans l'ordre de la requête
        np.testing.assert_allclose(result.quantiles[:, 0, 4], [[3.25, 3.5], [4.25, 4.5], [9.25, 9.5]])

    def test_any_level_requested_as_such(self, fake_t0):
        X, y = build_adapter_series(n_dates=8)
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=1)
        adapter = FoundationModelAdapter(FAKE_ID).fit(X_train, y_train)
        frame = adapter.predict_quantiles(X_test, quantile_levels=[0.999, 0.001, 0.37])
        assert fake_t0[0].calls[0]['levels'] == [0.001, 0.37, 0.999]
        np.testing.assert_allclose(frame.to_numpy(), y_train.iloc[-1] + np.array([[0.999, 0.001, 0.37]]))


class TestLoading:
    """Missing package."""

    @pytest.mark.parametrize('package', ['t0', 'torch'])
    def test_missing_package_when_predicting(self, block_package, package, monkeypatch):
        # Paquet complémentaire simulé : seul le paquet bloqué manque
        for name in ('t0', 'torch'):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        block_package(package)
        request = ForecastRequest(context=(np.array([[1.0]]),), prediction_length=1, quantile_levels=(0.5,))
        with pytest.raises(ImportError, match='adapters-t0'):
            T0Backend(FakeForecaster()).predict(request)

    def test_missing_package_names_the_extra(self, block_package, isolated_registry):
        block_package('t0')
        X, y = build_adapter_series(n_dates=6)
        with pytest.raises(ImportError, match='adapters-t0'):
            FoundationModelAdapter(T0_GAMMA).fit(X, y)
        with pytest.raises(ImportError, match='adapters-t0'):
            load_t0(T0_GAMMA, device='cpu')


# ---------------------------------------------------------------------------
# Modèle réel
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.usefixtures('t0_gamma')
class TestT0Contract(AdapterContractTests):
    """Unit contract of the adapter around t0-gamma, with known covariates."""

    case = foundation_t0_case()


# Niveaux comparés : appris, interpolé et au-delà des niveaux appris
REAL_LEVELS = (0.001, 0.1, 0.5, 0.505, 0.9, 0.999)
# Tolérances des comparaisons en float32, lots de séries de longueurs inégales compris
RTOL, ATOL = 1e-4, 1e-4


def _native_quantiles(X_train, y_train, X_test, gap, covariates):
    """Quantiles of every test row from the forecaster called directly, one series per call."""
    n_dates = X_test.index.get_level_values(-1).nunique()
    rows = partial(t0_row_quantiles, t0_forecaster(T0_GAMMA))
    fh = tuple(range(gap + 1, gap + n_dates + 1))
    return foundation_quantiles(rows, X_train, y_train, X_test, fh, REAL_LEVELS, covariates)


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.usefixtures('t0_gamma')
class TestNativeEquivalence:
    """The adapter gives the quantiles of ``T0Forecaster`` called directly on the same context."""

    @pytest.mark.parametrize('kind', ['series', 'unbalanced_panel'])
    @pytest.mark.parametrize('gap', [0, 2])
    @pytest.mark.parametrize('covariates', ['known', 'past', None])
    def test_quantiles(self, kind, gap, covariates):
        X, y = build_adapter_series() if kind == 'series' else build_adapter_panel(balanced=False)
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=gap)
        adapter = FoundationModelAdapter(T0_GAMMA, covariates=covariates).fit(X_train, y_train)
        frame = adapter.predict_quantiles(X_test, quantile_levels=REAL_LEVELS)
        expected = _native_quantiles(X_train, y_train, X_test, gap, covariates)
        np.testing.assert_allclose(frame.to_numpy(), expected, rtol=RTOL, atol=ATOL)

    def test_one_series_per_call(self):
        X, y = build_adapter_panel(balanced=False)
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=1)
        adapter = FoundationModelAdapter(T0_GAMMA, batch_size=1).fit(X_train, y_train)
        frame = adapter.predict_quantiles(X_test, quantile_levels=REAL_LEVELS)
        expected = _native_quantiles(X_train, y_train, X_test, 1, 'known')
        np.testing.assert_allclose(frame.to_numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_several_targets_forecast_jointly(self):
        X, y = build_adapter_series()
        Y = pd.DataFrame({'a': y, 'b': y ** 2})
        X_train, Y_train, X_test, _ = split_by_date(X, Y, n_test=1)
        adapter = FoundationModelAdapter(T0_GAMMA, covariates=None).fit(X_train, Y_train)
        context = np.array(Y_train.to_numpy().T[None], dtype=np.float32)
        expected = t0_forecaster(T0_GAMMA).predict(context, 1, [0.5]).quantiles[:, :, 0, 0].numpy()
        np.testing.assert_allclose(adapter.predict(X_test), expected, rtol=RTOL, atol=ATOL)

    def test_options_of_the_package(self):
        X, y = build_adapter_series()
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=1)
        options = {'symmetric_averaging': True}
        adapter = FoundationModelAdapter(T0_GAMMA, covariates=None, predict_kwargs=options)
        predictions = adapter.fit(X_train, y_train).predict(X_test)
        forecaster = t0_forecaster(T0_GAMMA)
        context = np.array(y_train.to_numpy()[None], dtype=np.float32)
        expected = forecaster.predict(context, 1, [0.5], **options).quantiles[0, :, 0].numpy()
        np.testing.assert_allclose(predictions, expected, rtol=RTOL, atol=ATOL)
        default = forecaster.predict(context, 1, [0.5]).quantiles[0, :, 0].numpy()
        assert not np.allclose(predictions, default, rtol=RTOL, atol=ATOL)

    def test_beyond_one_forward_pass(self, t0_gamma):
        # 1 030 pas : prolongement autorégressif, dépendant de la largeur du lot
        rng = np.random.default_rng(0)
        context = tuple(rng.standard_normal((1, length)).cumsum(axis=1) for length in (64, 20))
        levels = (0.1, 0.5, 0.9)
        request = ForecastRequest(context=context, prediction_length=1030, quantile_levels=levels)
        quantiles = t0_gamma.predict(request).quantiles
        forecaster = t0_forecaster(T0_GAMMA)
        # Chaque série comparée à sa prévision seule
        for position, series in enumerate(context):
            alone = forecaster.predict(np.array(series, dtype=np.float32), 1030, list(levels)).quantiles[0].numpy()
            np.testing.assert_allclose(quantiles[position, 0], alone, rtol=1e-3, atol=1e-3)


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.usefixtures('t0_gamma')
class TestRealForecastProperties:
    """Properties of the forecasts of the real model through the adapter."""

    def test_later_covariates_never_read(self):
        X, y = build_adapter_panel(balanced=False)
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=2)
        adapter = FoundationModelAdapter(T0_GAMMA).fit(X_train, y_train)
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
        adapter = FoundationModelAdapter(T0_GAMMA, covariates=None).fit(X_train, y_train)
        batch = adapter.predict(X_test)
        alone = np.array([adapter.predict(X_test.iloc[[row]])[0] for row in range(len(X_test))])
        np.testing.assert_allclose(alone, batch, rtol=RTOL, atol=ATOL)
