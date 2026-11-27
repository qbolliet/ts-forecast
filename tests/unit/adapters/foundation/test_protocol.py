"""Unit tests of ``tsforecast.adapters.foundation.protocol``.

``BackendCapabilities`` (defaults, lengths, native levels), ``ForecastRequest``
(read-only views leaving the inputs writable, frozen options, shape checks of the
contexts, levels and covariates), ``ForecastResult`` (four dimensions),
``ForecastBackend`` (structural check) and ``same_level``.
"""
import numpy as np
import pytest

from tsforecast.adapters.foundation.protocol import (
    BackendCapabilities,
    ForecastBackend,
    ForecastRequest,
    ForecastResult,
    same_level,
)


class TestBackendCapabilities:
    """Declared capabilities."""

    def test_defaults_and_levels(self):
        capabilities = BackendCapabilities(quantile_levels=[0.1, 0.9])
        assert not (capabilities.past_covariates or capabilities.known_covariates or capabilities.missing_values)
        assert capabilities.interpolates_quantiles and capabilities.quantile_levels == (0.1, 0.9)

    @pytest.mark.parametrize(
        'params',
        [{'max_context': 0}, {'max_context': True}, {'max_prediction_length': 2.5},
         {'quantile_levels': ()}, {'quantile_levels': (0.5, 0.1)}, {'quantile_levels': (0.0, 0.5)}],
    )
    def test_invalid(self, params):
        with pytest.raises(ValueError):
            BackendCapabilities(**params)


def _request(**params):
    """Request of two series, two known covariate rows, forecast over two steps."""
    defaults = dict(
        context=(np.array([[1.0, 2.0, 3.0]]), np.array([[4.0, 5.0]])),
        prediction_length=2,
        quantile_levels=(0.1, 0.5),
    )
    return ForecastRequest(**{**defaults, **params})


class TestForecastRequest:
    """Requests of a batch of series."""

    def test_read_only_views(self):
        context = np.array([[1.0, 2.0, 3.0]])
        request = _request(context=(context, np.array([[4.0, 5.0]])), options={'a': 1})
        assert not request.context[0].flags.writeable and context.flags.writeable
        assert np.shares_memory(request.context[0], context)
        assert request.n_series == 2 and request.n_targets == 1
        with pytest.raises(TypeError):
            request.options['a'] = 2

    def test_covariates(self):
        request = _request(
            past_covariates=(np.zeros((1, 3)), np.zeros((1, 2))),
            past_covariate_names=('p',),
            known_covariates=(np.zeros((2, 5)), np.zeros((2, 4))),
            known_covariate_names=('k1', 'k2'),
        )
        assert request.known_covariates[1].shape == (2, 4)
        assert not request.known_covariates[0].flags.writeable

    @pytest.mark.parametrize(
        'params',
        [
            {'context': ()},
            {'context': (np.array([1.0, 2.0]),)},
            {'context': (np.zeros((1, 2)), np.zeros((2, 2)))},
            {'context': (np.zeros((1, 0)),)},
            {'prediction_length': 0},
            {'prediction_length': None},
            {'quantile_levels': (0.5, 0.1)},
            {'past_covariate_names': ('p',)},
            {'known_covariates': (np.zeros((1, 5)),), 'known_covariate_names': ('k',)},
            {'known_covariates': (np.zeros((1, 3)), np.zeros((1, 4))), 'known_covariate_names': ('k',)},
            {'past_covariates': (np.zeros((2, 3)), np.zeros((2, 2))), 'past_covariate_names': ('p',)},
        ],
    )
    def test_invalid(self, params):
        with pytest.raises(ValueError):
            _request(**params)


class TestForecastResultAndBackend:
    """Results, structural type of a backend, equality of levels."""

    def test_result(self):
        result = ForecastResult([[[[1, 2]]]])
        assert result.quantiles.dtype == np.float64 and not result.quantiles.flags.writeable
        with pytest.raises(ValueError, match='shape'):
            ForecastResult(np.zeros((2, 3)))

    def test_backend_protocol(self):
        class WithoutPredict:
            capabilities = BackendCapabilities()

        assert not isinstance(WithoutPredict(), ForecastBackend)

    def test_same_level(self):
        assert same_level(0.7, 0.1 * 7) and not same_level(0.7, 0.7001)
