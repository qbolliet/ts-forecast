"""Unit tests of ``tsforecast.adapters.foundation.registry``.

Registration (names, prefixes, replacement, invalid entries), resolution of the
backend of a model id (longest prefix, case, explicit name, unknown ids), the
process-level model cache (one load per backend, model id, device and options,
shared by clones, cross-validation folds and grid-search candidates; release;
unhashable options; invalid loaders), the package check made before the cache,
and fitted adapters that are pickled without the model and use the model of
their fit after ``set_params``.
"""
import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score

from tests.support.adapter_contract import default_splitter, split_by_date
from tests.support.datasets import build_adapter_series
from tests.support.forecast_backends import DriftBackend, counting_loader
from tsforecast.adapters import FoundationModelAdapter


@pytest.fixture
def fake(isolated_registry):
    """Backend ``fake`` registered for the prefix ``fake-org/``; returns the list of its loads."""
    calls = []
    isolated_registry.register_backend('fake', counting_loader(calls), prefixes=('Fake-Org/',))
    return calls


class TestRegistration:
    """Registered backends and their prefixes."""

    def test_builtin_chronos(self):
        from tsforecast.adapters.foundation import available_backends

        assert available_backends()['chronos'] == ('amazon/chronos', 'autogluon/chronos')

    def test_register_and_replace(self, isolated_registry, fake):
        assert isolated_registry.available_backends()['fake'] == ('fake-org/',)
        with pytest.raises(ValueError, match='already registered'):
            isolated_registry.register_backend('fake', counting_loader([]))
        isolated_registry.register_backend('fake', counting_loader([]), prefixes=('other/',), replace=True)
        assert isolated_registry.available_backends()['fake'] == ('other/',)

    @pytest.mark.parametrize(
        'name, load, prefixes, error',
        [('', counting_loader([]), (), ValueError), ('x', 'load', (), TypeError), ('x', counting_loader([]), ('',), ValueError)],
    )
    def test_invalid_registrations(self, isolated_registry, name, load, prefixes, error):
        with pytest.raises(error):
            isolated_registry.register_backend(name, load, prefixes=prefixes)


class TestResolution:
    """Backend of a model id."""

    def test_longest_prefix_and_case(self, isolated_registry):
        isolated_registry.register_backend('short', counting_loader([]), prefixes=('org/',))
        isolated_registry.register_backend('long', counting_loader([]), prefixes=('org/model-',))
        assert isolated_registry.resolve_backend('ORG/Model-1').name == 'long'
        assert isolated_registry.resolve_backend('org/other').name == 'short'

    def test_explicit_name_and_unknown(self, isolated_registry, fake):
        assert isolated_registry.resolve_backend('./local/path', 'fake').name == 'fake'
        with pytest.raises(ValueError, match='Unknown backend'):
            isolated_registry.resolve_backend('fake-org/m', 'missing')
        with pytest.raises(ValueError, match='No registered backend'):
            isolated_registry.resolve_backend('nobody/model')


class TestModelCache:
    """One load per key in the process."""

    def test_keys(self, isolated_registry, fake):
        spec = isolated_registry.resolve_backend('fake-org/a')
        first = isolated_registry.load_backend(spec, 'fake-org/a', device='cpu')
        assert isolated_registry.load_backend(spec, 'fake-org/a', device='cpu') is first
        isolated_registry.load_backend(spec, 'fake-org/a', device='cuda')
        isolated_registry.load_backend(spec, 'fake-org/a', device='cpu', backend_kwargs={'revision': 'r1'})
        isolated_registry.load_backend(spec, 'fake-org/b', device='cpu')
        assert fake == [('fake-org/a', 'cpu', {}), ('fake-org/a', 'cuda', {}),
                        ('fake-org/a', 'cpu', {'revision': 'r1'}), ('fake-org/b', 'cpu', {})]
        assert isolated_registry.clear_model_cache() == 4
        isolated_registry.load_backend(spec, 'fake-org/a', device='cpu')
        assert len(fake) == 5

    def test_unhashable_options(self, isolated_registry, fake):
        spec = isolated_registry.resolve_backend('fake-org/a')
        with pytest.raises(ValueError, match='hashable'):
            isolated_registry.load_backend(spec, 'fake-org/a', device='cpu', backend_kwargs={'levels': [0.5]})

    def test_invalid_loader(self, isolated_registry):
        isolated_registry.register_backend('broken', lambda model_id, *, device: object(), prefixes=('broken/',))
        with pytest.raises(TypeError, match='capabilities'):
            FoundationModelAdapter('broken/model').fit(*build_adapter_series(n_dates=6))

    def test_package_checked_before_the_cache(self, isolated_registry, block_package):
        calls = []
        isolated_registry.register_backend(
            'needs', counting_loader(calls), prefixes=('needs/',), package='colorsys', extra='adapters-example'
        )
        X, y = build_adapter_series(n_dates=6)
        FoundationModelAdapter('needs/model').fit(X, y)
        block_package('colorsys')
        with pytest.raises(ImportError, match='adapters-example'):
            FoundationModelAdapter('needs/model').fit(X, y)
        assert len(calls) == 1

    def test_shared_by_clones_folds_and_candidates(self, fake):
        X, y = build_adapter_series()
        adapter = FoundationModelAdapter('fake-org/model')
        cross_val_score(clone(adapter), X, y, cv=default_splitter('series'), error_score='raise')
        GridSearchCV(adapter, {'context_length': [3, None], 'quantile': [0.4, 0.6]},
                     cv=default_splitter('series'), error_score='raise').fit(X, y)
        assert len(fake) == 1
        # Périphérique réglé par la recherche : un chargement par périphérique
        GridSearchCV(adapter, {'device': ['cpu', 'meta']}, cv=default_splitter('series'), error_score='raise').fit(X, y)
        assert [device for _, device, _ in fake] == ['cpu', 'meta']


class TestFittedAdapter:
    """The fitted adapter refers to its model through the cache."""

    def test_pickled_without_the_model(self, isolated_registry, fake):
        X, y = build_adapter_series()
        X_train, y_train, X_test, _ = split_by_date(X, y)
        adapter = FoundationModelAdapter('fake-org/model').fit(X_train, y_train)
        reference = adapter.predict(X_test)
        assert adapter._backend_object is None
        assert b'DriftBackend' not in pickle.dumps(adapter)
        # Processus sans le modèle : chargé de nouveau au predict
        isolated_registry.clear_model_cache()
        restored = pickle.loads(pickle.dumps(adapter))
        np.testing.assert_array_equal(restored.predict(X_test), reference)
        assert len(fake) == 2

    def test_model_of_the_fit_after_set_params(self, fake):
        X, y = build_adapter_series()
        X_train, y_train, X_test, _ = split_by_date(X, y)
        adapter = FoundationModelAdapter('fake-org/a').fit(X_train, y_train)
        adapter.set_params(model_id='fake-org/b').predict(X_test)
        assert [model_id for model_id, _, _ in fake] == ['fake-org/a']

    def test_backend_object_never_copied_into_the_cache(self, isolated_registry):
        backend = DriftBackend()
        X, y = build_adapter_series(n_dates=6)
        adapter = FoundationModelAdapter(backend=backend).fit(X, y)
        assert adapter._backend_object is backend and adapter.backend_name_ == 'DriftBackend'
        assert isolated_registry._MODEL_CACHE == {}
