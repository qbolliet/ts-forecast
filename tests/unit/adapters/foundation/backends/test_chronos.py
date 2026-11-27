"""Unit tests of ``tsforecast.adapters.foundation.backends.chronos``: ``ChronosBackend`` and ``load_chronos``.

Offline, with simulated ``chronos`` and ``torch`` packages: dictionaries built for
Chronos-2 (target, past and future covariates, ``NaN`` on the gap, one call per
number of steps), tensors for Chronos-Bolt, capabilities read on the pipeline,
refused Chronos-T5 and unknown pipelines, missing package. With the real models
(marked ``slow`` and ``network``, skipped without chronos-forecasting or network):
unit rules of the adapter contract on ``autogluon/chronos-2-small``, quantiles
equal to ``predict_quantiles`` of the pipeline called directly (Chronos-2 with and
without covariates, Chronos-Bolt), later covariates never read, row alone equal
to the batch.
"""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from tests.support.adapter_cases import CHRONOS_2_SMALL, CHRONOS_BOLT_TINY, foundation_chronos_case
from tests.support.adapter_contract import AdapterContractTests, split_by_date
from tests.support.datasets import build_adapter_panel, build_adapter_series
from tests.support.native_references import chronos_pipeline, chronos_row_quantiles
from tsforecast.adapters import FoundationModelAdapter
from tsforecast.adapters.foundation import ForecastRequest
from tsforecast.adapters.foundation.backends.chronos import ChronosBackend, load_chronos

# Niveaux des pipelines simulés
FAKE_LEVELS = [0.1, 0.5, 0.9]


# ---------------------------------------------------------------------------
# Paquets simulés
# ---------------------------------------------------------------------------


class FakeChronos2Pipeline:
    """Simulated ``Chronos2Pipeline``: forecasts the last target value, records its calls."""

    quantiles = FAKE_LEVELS
    model_context_length = 100

    def __init__(self):
        self.calls = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
        self.calls.append({'inputs': inputs, 'prediction_length': prediction_length,
                           'quantile_levels': quantile_levels, 'kwargs': kwargs})
        outputs = []
        for item in inputs:
            target = np.atleast_2d(item['target'])
            outputs.append(np.broadcast_to(target[:, -1:, None], (target.shape[0], prediction_length, len(quantile_levels))) + np.array(quantile_levels))
        return outputs, None


class FakeBoltPipeline:
    """Simulated ``ChronosBoltPipeline``: same forecast on a list of 1-D tensors."""

    quantiles = FAKE_LEVELS
    model_context_length = 50

    def __init__(self):
        self.calls = []

    def predict_quantiles(self, inputs, prediction_length, quantile_levels, **kwargs):
        self.calls.append({'inputs': inputs, 'prediction_length': prediction_length, 'kwargs': kwargs})
        last = np.array([np.asarray(series)[-1] for series in inputs])
        return np.broadcast_to(last[:, None, None], (len(inputs), prediction_length, len(quantile_levels))) + np.array(quantile_levels), None


class FakeOtherPipeline:
    """Simulated pipeline of another family."""

    quantiles = FAKE_LEVELS
    model_context_length = 10


@pytest.fixture
def fake_chronos(monkeypatch, isolated_registry):
    """Simulated ``chronos`` and ``torch`` packages; returns the list of loaded pipelines."""
    loaded = []

    def from_pretrained(model_id, device_map, **kwargs):
        pipeline = FakeBoltPipeline() if 'bolt' in model_id else (
            FakeOtherPipeline() if 'other' in model_id else FakeChronos2Pipeline())
        pipeline.load_args = (model_id, device_map, kwargs)
        loaded.append(pipeline)
        return pipeline

    chronos = types.ModuleType('chronos')
    chronos.BaseChronosPipeline = types.SimpleNamespace(from_pretrained=from_pretrained)
    chronos.Chronos2Pipeline = FakeChronos2Pipeline
    chronos.ChronosBoltPipeline = FakeBoltPipeline
    torch = types.ModuleType('torch')
    torch.float32 = 'float32'
    torch.as_tensor = lambda values, dtype=None: np.asarray(values)
    monkeypatch.setitem(sys.modules, 'chronos', chronos)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    return loaded


# ---------------------------------------------------------------------------
# Correspondance hors ligne
# ---------------------------------------------------------------------------


class TestChronos2Mapping:
    """Requests translated into the dictionaries of ``Chronos2Pipeline.predict_quantiles``."""

    def test_loading_arguments_and_capabilities(self, fake_chronos):
        X, y = build_adapter_series(n_dates=8)
        adapter = FoundationModelAdapter(
            'amazon/chronos-2-test', device='cuda', backend_kwargs={'revision': 'abc'}
        ).fit(X, y)
        assert fake_chronos[0].load_args == ('amazon/chronos-2-test', 'cuda', {'revision': 'abc'})
        assert adapter.backend_name_ == 'chronos'
        capabilities = adapter.capabilities_
        assert (capabilities.past_covariates, capabilities.known_covariates, capabilities.missing_values,
                capabilities.multivariate) == (True, True, True, True)
        assert capabilities.max_context == 100 and capabilities.quantile_levels == (0.1, 0.5, 0.9)
        assert adapter.covariates_ == 'known'

    def test_known_covariates_dictionaries_per_step(self, fake_chronos):
        X, y = build_adapter_series(n_dates=12)
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=2, gap=2)
        adapter = FoundationModelAdapter('amazon/chronos-2-test', predict_kwargs={'cross_learning': False})
        predictions = adapter.fit(X_train, y_train).predict(X_test)
        calls = fake_chronos[0].calls
        # Un appel par nombre de pas : 3 puis 4
        assert [call['prediction_length'] for call in calls] == [3, 4]
        assert all(call['kwargs'] == {'cross_learning': False} for call in calls)
        assert all(call['quantile_levels'] == [0.5] for call in calls)
        for call, k in zip(calls, (3, 4)):
            (item,) = call['inputs']
            np.testing.assert_array_equal(item['target'], y_train.to_numpy())
            assert set(item['past_covariates']) == set(item['future_covariates']) == {'x1', 'x2'}
            np.testing.assert_array_equal(item['past_covariates']['x1'], X_train['x1'].to_numpy())
            future = item['future_covariates']['x1']
            # Pas du gap inconnus, lignes de test aux pas 3 et 4, rien au-delà de k
            assert len(future) == k
            assert np.isnan(future[:2]).all()
            np.testing.assert_array_equal(future[2:], X_test['x1'].to_numpy()[: k - 2])
        np.testing.assert_allclose(predictions, y_train.iloc[-1] + 0.5)

    def test_past_covariates_and_no_covariates(self, fake_chronos):
        X, y = build_adapter_series(n_dates=8)
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=1)
        FoundationModelAdapter('amazon/chronos-2-test', covariates='past').fit(X_train, y_train).predict(X_test)
        FoundationModelAdapter('amazon/chronos-2-test', covariates=None).fit(X_train, y_train).predict(X_test)
        (past_item,), (plain_item,) = (call['inputs'] for call in fake_chronos[0].calls)
        assert set(past_item['past_covariates']) == {'x1', 'x2'} and 'future_covariates' not in past_item
        assert set(plain_item) == {'target'}

    def test_several_targets_as_a_two_dimensional_target(self, fake_chronos):
        X, y = build_adapter_series(n_dates=8)
        Y = pd.DataFrame({'a': y, 'b': -y})
        X_train, Y_train, X_test, _ = split_by_date(X, Y, n_test=1)
        predictions = FoundationModelAdapter('amazon/chronos-2-test', covariates=None).fit(X_train, Y_train).predict(X_test)
        (item,) = fake_chronos[0].calls[0]['inputs']
        assert item['target'].shape == (2, len(X_train))
        np.testing.assert_allclose(predictions, [[Y_train['a'].iloc[-1] + 0.5, Y_train['b'].iloc[-1] + 0.5]])

    def test_backend_predict_on_a_request(self, fake_chronos):
        backend = load_chronos('amazon/chronos-2-test', device='cpu')
        request = ForecastRequest(
            context=(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0, 5.0]])),
            prediction_length=2,
            quantile_levels=(0.1, 0.9),
            known_covariates=(np.array([[0.0, 1.0, 2.0, 3.0]]), np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])),
            known_covariate_names=('c',),
        )
        result = backend.predict(request)
        assert result.quantiles.shape == (2, 1, 2, 2)
        np.testing.assert_allclose(result.quantiles[:, 0, 0, :], [[2.1, 2.9], [5.1, 5.9]])
        first, second = fake_chronos[0].calls[0]['inputs']
        np.testing.assert_array_equal(first['past_covariates']['c'], [0.0, 1.0])
        np.testing.assert_array_equal(second['future_covariates']['c'], [3.0, 4.0])


class TestBoltMapping:
    """Chronos-Bolt: one 1-D tensor per series, one target per series, no covariates."""

    def test_capabilities_and_calls(self, fake_chronos):
        X, y = build_adapter_panel(n_dates=10)
        Y = pd.DataFrame({'a': y, 'b': 2 * y})
        X_train, Y_train, X_test, _ = split_by_date(X, Y, n_test=1, gap=1)
        adapter = FoundationModelAdapter('amazon/chronos-bolt-test').fit(X_train, Y_train)
        assert adapter.covariates_ is None
        assert not adapter.capabilities_.multivariate and adapter.capabilities_.max_context == 50
        predictions = adapter.predict(X_test)
        (call,) = fake_chronos[0].calls
        # Une série par (entité, cible), dans l'ordre des étiquettes d'entité
        assert call['prediction_length'] == 2 and len(call['inputs']) == 6
        assert all(np.asarray(series).ndim == 1 for series in call['inputs'])
        last = Y_train.groupby(level='entity').last().to_numpy()
        np.testing.assert_allclose(predictions, last + 0.5)


class TestLoading:
    """Refused models and missing package."""

    def test_sampling_models_refused_before_any_import(self, block_package):
        block_package('chronos')
        with pytest.raises(ValueError, match='Chronos-T5'):
            load_chronos('amazon/chronos-t5-tiny', device='cpu')

    def test_other_pipelines_refused(self, fake_chronos):
        with pytest.raises(ValueError, match='not supported'):
            load_chronos('amazon/chronos-other', device='cpu')

    def test_missing_package_names_the_extra(self, block_package, isolated_registry):
        block_package('chronos')
        X, y = build_adapter_series(n_dates=6)
        with pytest.raises(ImportError, match='adapters-chronos'):
            FoundationModelAdapter('amazon/chronos-2').fit(X, y)

    def test_tensor_outputs_converted(self):
        torch = pytest.importorskip('torch')
        pipeline = FakeChronos2Pipeline()
        backend = ChronosBackend(pipeline, 'chronos-2')
        pipeline.predict_quantiles = lambda inputs, **kwargs: ([torch.ones(1, 2, 1)], None)
        request = ForecastRequest(context=(np.array([[1.0]]),), prediction_length=2, quantile_levels=(0.5,))
        assert backend.predict(request).quantiles.dtype == np.float64


# ---------------------------------------------------------------------------
# Modèles réels
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def bolt_tiny():
    """``amazon/chronos-bolt-tiny`` loaded by chronos-forecasting alone, or the test skipped."""
    pytest.importorskip('chronos')
    try:
        return chronos_pipeline(CHRONOS_BOLT_TINY)
    except OSError as error:
        pytest.skip(f'{CHRONOS_BOLT_TINY} cannot be downloaded: {error}')


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.usefixtures('chronos_small')
class TestChronosContract(AdapterContractTests):
    """Unit contract of the adapter around Chronos-2 small, with known covariates."""

    case = foundation_chronos_case()


# Niveaux comparés : appris par chronos-2-small, sauf 0.25 (interpolé par Chronos)
REAL_LEVELS = (0.01, 0.1, 0.25, 0.5, 0.9, 0.99)


def _native_quantiles(model_id, X_train, y_train, X_test, gap, covariates):
    """Quantiles of every test row from the pipeline called directly, one ``k``-step forecast per row."""
    pipeline = chronos_pipeline(model_id)
    calendar = pd.DatetimeIndex(X_test.index.get_level_values(-1)).unique().sort_values()
    expected = np.empty((len(X_test), len(REAL_LEVELS)))
    entities = [None] if not isinstance(X_test.index, pd.MultiIndex) else X_test.index.get_level_values(0).unique()
    for entity in entities:
        train_mask = np.ones(len(X_train), bool) if entity is None else (X_train.index.get_level_values(0) == entity)
        test_rows = np.flatnonzero(np.ones(len(X_test), bool) if entity is None else (X_test.index.get_level_values(0) == entity))
        steps = gap + 1 + calendar.get_indexer(X_test.index.get_level_values(-1)[test_rows])
        X_context = X_train.to_numpy()[train_mask] if covariates else None
        X_future = dict(zip(steps.tolist(), X_test.to_numpy()[test_rows])) if covariates else None
        for row, k in zip(test_rows, steps.tolist()):
            expected[row] = chronos_row_quantiles(
                pipeline, y_train.to_numpy()[train_mask], X_context, X_future, k, REAL_LEVELS, list(X_train.columns)
            )
    return expected


@pytest.mark.slow
@pytest.mark.network
class TestNativeEquivalence:
    """``predict_quantiles`` equals the quantiles of the pipeline called directly on the same context."""

    @pytest.mark.parametrize('kind', ['series', 'panel'])
    @pytest.mark.parametrize('gap', [0, 2])
    @pytest.mark.parametrize('covariates', ['auto', None])
    def test_chronos_2_quantiles(self, chronos_small, kind, gap, covariates):
        X, y = build_adapter_series() if kind == 'series' else build_adapter_panel()
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=gap)
        adapter = FoundationModelAdapter(CHRONOS_2_SMALL, covariates=covariates).fit(X_train, y_train)
        frame = adapter.predict_quantiles(X_test, quantile_levels=REAL_LEVELS)
        expected = _native_quantiles(CHRONOS_2_SMALL, X_train, y_train, X_test, gap, covariates == 'auto')
        np.testing.assert_allclose(frame.to_numpy(), expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize('gap', [0, 2])
    def test_bolt_quantiles(self, bolt_tiny, isolated_registry, gap):
        X, y = build_adapter_panel()
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=gap)
        adapter = FoundationModelAdapter(CHRONOS_BOLT_TINY).fit(X_train, y_train)
        assert adapter.covariates_ is None
        frame = adapter.predict_quantiles(X_test, quantile_levels=(0.1, 0.5, 0.9))
        pipeline = chronos_pipeline(CHRONOS_BOLT_TINY)
        expected = np.empty((len(X_test), 3))
        calendar = pd.DatetimeIndex(X_test.index.get_level_values(-1)).unique().sort_values()
        for row, (entity, date) in enumerate(X_test.index):
            context = y_train.xs(entity).to_numpy()
            k = gap + 1 + calendar.get_loc(date)
            expected[row] = chronos_row_quantiles(pipeline, context, None, None, k, (0.1, 0.5, 0.9))
        np.testing.assert_allclose(frame.to_numpy(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.slow
@pytest.mark.network
class TestRealForecastProperties:
    """Properties of the forecasts of the real model through the adapter."""

    def test_later_covariates_never_read(self, chronos_small):
        X, y = build_adapter_panel(balanced=False)
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=2)
        adapter = FoundationModelAdapter(CHRONOS_2_SMALL).fit(X_train, y_train)
        reference = adapter.predict(X_test)
        # Covariables de la dernière date de test altérées : lignes antérieures inchangées
        last = np.asarray(X_test.index.get_level_values(-1) == X_test.index.get_level_values(-1).max())
        altered = X_test.mask(np.broadcast_to(last[:, None], X_test.shape), X_test + 10.0)
        predictions = adapter.predict(altered)
        np.testing.assert_array_equal(predictions[~last], reference[~last])
        assert not np.allclose(predictions[last], reference[last])

    def test_row_alone_equals_batch_without_known_covariates(self, chronos_small):
        X, y = build_adapter_panel(balanced=False)
        X_train, y_train, X_test, _ = split_by_date(X, y, gap=2)
        adapter = FoundationModelAdapter(CHRONOS_2_SMALL, covariates=None).fit(X_train, y_train)
        batch = adapter.predict(X_test)
        alone = np.array([adapter.predict(X_test.iloc[[row]])[0] for row in range(len(X_test))])
        np.testing.assert_allclose(alone, batch, rtol=1e-5, atol=1e-6)
