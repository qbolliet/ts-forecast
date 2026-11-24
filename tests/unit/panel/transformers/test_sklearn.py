"""Tests of the sklearn integration of ``PanelwiseTransformer``.

Covers ``get_params`` / ``set_params`` (own and nested ``transformer__`` parameters,
factory mode), ``clone`` (unfitted and fitted, factory kept), use inside a
``Pipeline`` and with a ``Pipeline`` as entity transformer, ``GridSearchCV`` over a
nested parameter, ``pickle`` and ``joblib`` round trips of a fitted transformer,
``NotFittedError`` before ``fit``, the fitted attributes (``transformers_``,
``get_entity_transformer``, ``n_entities_``, ``entities_``) and ``__repr__``.
"""
import io
import pickle

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tsforecast.panel.transformers import PanelwiseTransformer

from .conftest import centring, month_ends


def _factory(entity_key: tuple) -> StandardScaler:
    """Module-level factory, picklable: one centring scaler per entity."""
    return StandardScaler(with_std=False)


# =============================================================================
# Paramètres
# =============================================================================
class TestParams:
    """``get_params`` / ``set_params`` expose own and nested parameters."""

    def test_shallow_params_are_the_constructor_arguments(self):
        """``get_params(deep=False)`` lists exactly the constructor arguments, as stored."""
        transformer = PanelwiseTransformer(centring(), panel_cols=['country'], n_jobs=2, error_handling='warn')

        params = transformer.get_params(deep=False)

        assert sorted(params) == sorted([
            'transformer', 'time_col', 'panel_cols', 'entity_kwargs', 'default_entity_kwargs', 'validate_input',
            'strict_validation', 'auto_sort', 'convert_cols_to_index', 'n_jobs', 'error_handling',
        ])
        assert (params['panel_cols'], params['n_jobs'], params['error_handling']) == (['country'], 2, 'warn')

    def test_deep_params_include_the_nested_transformer(self):
        """``get_params(deep=True)`` adds the ``transformer__`` parameters."""
        params = PanelwiseTransformer(StandardScaler(with_mean=False)).get_params(deep=True)
        assert params['transformer__with_mean'] is False

    def test_set_nested_param(self):
        """``set_params(transformer__…)`` reaches the base transformer."""
        transformer = PanelwiseTransformer(StandardScaler()).set_params(transformer__with_mean=False)
        assert transformer.transformer.with_mean is False

    def test_set_own_param(self):
        """``set_params`` sets own parameters and returns the transformer."""
        transformer = PanelwiseTransformer(centring())
        assert transformer.set_params(n_jobs=3) is transformer and transformer.n_jobs == 3

    def test_transformer_without_get_params_has_no_nested_params(self):
        """A transformer that does not expose ``get_params`` adds no nested parameter."""

        class Bare:
            """Object with ``fit`` and ``transform`` only."""

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X

        params = PanelwiseTransformer(Bare()).get_params(deep=True)
        assert not any(key.startswith('transformer__') for key in params)

    def test_factory_has_no_nested_params(self):
        """A factory has no ``transformer__`` parameters to expose."""
        params = PanelwiseTransformer(_factory).get_params(deep=True)
        assert not any(key.startswith('transformer__') for key in params)

    def test_nested_param_on_a_factory_is_ignored_with_a_warning(self):
        """Nested parameters cannot be set on a factory: a warning says so."""
        with pytest.warns(UserWarning, match='Cannot set nested parameters on a callable factory transformer'):
            PanelwiseTransformer(_factory).set_params(transformer__with_mean=False)


# =============================================================================
# Clonage, pipelines, recherche d'hyperparamètres
# =============================================================================
class TestClone:
    """``clone`` builds an unfitted copy with equal parameters."""

    def test_clone_copies_the_parameters(self):
        """The clone has equal parameters and its own base transformer."""
        transformer = PanelwiseTransformer(StandardScaler(with_mean=False), panel_cols=['country'], n_jobs=2)

        cloned = clone(transformer)

        assert cloned.transformer is not transformer.transformer
        assert cloned.get_params(deep=True).keys() == transformer.get_params(deep=True).keys()
        assert (cloned.panel_cols, cloned.n_jobs, cloned.transformer.with_mean) == (['country'], 2, False)

    def test_clone_of_a_fitted_transformer_is_unfitted(self, index_panel):
        """Cloning drops the fitted state."""
        cloned = clone(PanelwiseTransformer(centring(), time_col=None).fit(index_panel))
        assert not hasattr(cloned, 'transformers_')

    def test_clone_keeps_the_factory(self):
        """A factory is a function: the clone refers to the same one."""
        assert clone(PanelwiseTransformer(_factory)).transformer is _factory


class TestPipelines:
    """``PanelwiseTransformer`` works inside a ``Pipeline`` and wraps one."""

    def test_inside_a_pipeline(self, column_panel, centred_column_panel):
        """As a pipeline step, it gives the same output as alone."""
        pipe = Pipeline([('per_entity', PanelwiseTransformer(centring(), panel_cols=['country']))])
        pd.testing.assert_frame_equal(pipe.fit_transform(column_panel), centred_column_panel)

    def test_pipeline_as_entity_transformer(self, index_panel):
        """A ``Pipeline`` is fitted per entity: each entity spans [0, 1] after ``MinMaxScaler``."""
        inner = Pipeline([('scale', StandardScaler()), ('minmax', MinMaxScaler())])

        output = PanelwiseTransformer(inner, time_col=None).fit_transform(index_panel)

        # Valeur d'or : a de FR (1, 2, 6) et de DE (10, 20, 60) -> (0, 0.2, 1)
        np.testing.assert_allclose(output['a'].to_numpy(), [0.0, 0.2, 1.0] * 2)

    def test_gridsearchcv_over_a_nested_parameter(self, index_panel):
        """``GridSearchCV`` searches ``transformer__with_mean`` through the pipeline.

        The target is ``a`` centred per entity: only ``with_mean=True`` lets a linear
        model without intercept fit it exactly, so the best value is known in advance.
        """
        y = pd.Series([-2.0, -1.0, 3.0, -20.0, -10.0, 30.0], index=index_panel.index)
        pipe = Pipeline([
            ('per_entity', PanelwiseTransformer(StandardScaler(with_std=False), time_col=None)),
            ('model', LinearRegression(fit_intercept=False)),
        ])
        # Pli unique : apprentissage et test sur toutes les lignes (seul le choix du paramètre compte)
        rows = np.arange(len(index_panel))
        search = GridSearchCV(pipe, {'per_entity__transformer__with_mean': [False, True]}, cv=[(rows, rows)])

        search.fit(index_panel, y)

        assert search.best_params_ == {'per_entity__transformer__with_mean': True}


# =============================================================================
# Sérialisation
# =============================================================================
class TestSerialization:
    """A fitted transformer survives ``pickle`` and ``joblib`` round trips."""

    @pytest.mark.parametrize('transformer', [centring(), _factory], ids=['simple', 'factory'])
    def test_pickle_round_trip(self, index_panel, centred_index_panel, transformer):
        """The unpickled transformer gives the same output."""
        fitted = PanelwiseTransformer(transformer, time_col=None).fit(index_panel)

        restored = pickle.loads(pickle.dumps(fitted))

        pd.testing.assert_frame_equal(restored.transform(index_panel), centred_index_panel)

    def test_joblib_round_trip(self, column_panel, centred_column_panel):
        """``joblib.dump`` / ``joblib.load`` keep the fitted state of a column panel transformer."""
        fitted = PanelwiseTransformer(centring(), panel_cols=['country']).fit(column_panel)
        buffer = io.BytesIO()

        joblib.dump(fitted, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)

        pd.testing.assert_frame_equal(restored.transform(column_panel), centred_column_panel)


# =============================================================================
# Attributs ajustés
# =============================================================================
class TestFittedAttributes:
    """Fitted attributes describe the entities and their transformers."""

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_not_fitted_error(self, index_panel, method):
        """Before ``fit``, ``transform`` and ``inverse_transform`` raise ``NotFittedError``."""
        with pytest.raises(NotFittedError):
            getattr(PanelwiseTransformer(centring(), time_col=None), method)(index_panel)

    @pytest.mark.parametrize('name', ['n_entities_', 'get_entity_transformer'])
    def test_entity_accessors_need_a_fit(self, name):
        """``n_entities_`` and ``get_entity_transformer`` raise ``NotFittedError`` before ``fit``."""
        transformer = PanelwiseTransformer(centring())
        with pytest.raises(NotFittedError):
            attribute = getattr(transformer, name)
            attribute('FR')

    def test_counts_of_entities(self, index_panel):
        """``entities_`` counts the entities of the data, ``n_entities_`` the fitted ones."""
        transformer = PanelwiseTransformer(centring(), time_col=None).fit(index_panel)
        assert (transformer.entities_, transformer.n_entities_) == (2, 2)

    def test_transformers_follow_the_order_of_the_data(self):
        """The fitted transformers are listed like the entities appear, not sorted."""
        index = pd.MultiIndex.from_tuples(
            [(region, country, date) for region, country in [('EU', 'B'), ('EU', 'A')] for date in month_ends()],
            names=['region', 'country', 'date'],
        )
        panel = pd.DataFrame({'v': [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]}, index=index)

        transformer = PanelwiseTransformer(centring(), time_col=None).fit(panel)

        assert list(transformer.transformers_) == [('EU', 'B'), ('EU', 'A')]

    @pytest.mark.parametrize('key', ['FR', ('FR',)], ids=['scalar', 'tuple'])
    def test_entity_transformer_is_found_with_either_notation(self, index_panel, key):
        """``get_entity_transformer`` normalizes the key like the rest of the package."""
        transformer = PanelwiseTransformer(centring(), time_col=None).fit(index_panel)
        assert transformer.get_entity_transformer(key) is transformer.transformers_[('FR',)]

    def test_entity_transformer_of_a_two_column_entity(self):
        """An entity of two panel columns is fetched by its pair of labels, fitted."""
        panel = pd.DataFrame({
            'country': ['FR', 'FR', 'DE', 'DE'], 'sector': ['tech'] * 4,
            'date': list(month_ends(2)) * 2, 'v': [1.0, 3.0, 10.0, 30.0],
        })
        transformer = PanelwiseTransformer(centring(), panel_cols=['country', 'sector']).fit(panel)

        # Valeur d'or : moyenne de (DE, tech) = 20
        assert transformer.get_entity_transformer(('DE', 'tech')).mean_.tolist() == [20.0]

    def test_unknown_entity_lists_the_available_ones(self, index_panel):
        """An unknown key raises a ``KeyError`` listing the fitted entities."""
        transformer = PanelwiseTransformer(centring(), time_col=None).fit(index_panel)
        with pytest.raises(KeyError, match=r"Entity \('IT',\) not found. Available entities: \[\('FR',\), \('DE',\)\]"):
            transformer.get_entity_transformer('IT')


class TestRepr:
    """``__repr__`` is the sklearn one: every parameter that differs from its default."""

    def test_repr_shows_the_changed_parameters(self):
        """``n_jobs``, ``error_handling`` and ``panel_cols`` are shown when set."""
        text = repr(PanelwiseTransformer(StandardScaler(), panel_cols=['country'], n_jobs=2, error_handling='warn'))
        assert ' '.join(text.split()) == (
            "PanelwiseTransformer(error_handling='warn', n_jobs=2, panel_cols=['country'], "
            "transformer=StandardScaler())"
        )

    def test_repr_shows_a_none_time_col_as_none(self):
        """``time_col=None`` reads ``None``, not the string ``'None'``."""
        assert repr(PanelwiseTransformer(StandardScaler(), time_col=None)) == (
            'PanelwiseTransformer(time_col=None, transformer=StandardScaler())'
        )

    def test_repr_of_a_factory_names_the_function(self):
        """A factory is shown by its name."""
        assert '<function _factory at' in repr(PanelwiseTransformer(_factory))
