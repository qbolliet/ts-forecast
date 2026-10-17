"""Tests of the three parameterization modes of ``PanelwiseTransformer``.

Covers the simple mode (one clone of ``transformer`` per entity, the parameter left
unfitted), the factory mode (a callable called with each entity key; invalid
returns; ``entity_kwargs`` ignored with a warning; ``is_factory_mode_``) and the
``entity_kwargs`` mode (per-entity parameters set on the clone, the fallback
``default_entity_kwargs``, key notations, invalid parameter names).
"""
import pytest
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from tsforecast.panel.transformers import PanelwiseTransformer

from .conftest import centring


def _fit(index_panel, **kwargs) -> PanelwiseTransformer:
    """Fit a ``PanelwiseTransformer`` on the index panel, the time labels read on the index."""
    return PanelwiseTransformer(time_col=None, **kwargs).fit(index_panel)


def _mean_scaler_factory(entity_key: tuple) -> StandardScaler:
    """Centre ``FR`` on its means, leave the other entities uncentred."""
    return StandardScaler(with_mean=entity_key == ('FR',), with_std=False)


# =============================================================================
# Mode simple
# =============================================================================
class TestSimpleMode:
    """One clone of the transformer is fitted per entity; the parameter itself is never fitted."""

    def test_each_entity_has_its_own_clone(self, index_panel):
        """The entity transformers are distinct objects, and none is the parameter."""
        base = centring()
        transformer = _fit(index_panel, transformer=base)

        fitted = list(transformer.transformers_.values())
        assert fitted[0] is not fitted[1] and base not in fitted

    def test_parameter_is_left_unfitted(self, index_panel):
        """The ``transformer`` parameter keeps no fitted state (sklearn convention)."""
        base = centring()
        _fit(index_panel, transformer=base)
        assert not hasattr(base, 'mean_')

    def test_clones_learn_their_entity_means(self, index_panel):
        """Each clone learns the means of its own entity."""
        transformer = _fit(index_panel, transformer=centring())
        # Valeur d'or : moyennes (a, b) de FR = (3, 4), de DE = (30, 3)
        assert transformer.get_entity_transformer('FR').mean_.tolist() == [3.0, 4.0]
        assert transformer.get_entity_transformer('DE').mean_.tolist() == [30.0, 3.0]

    def test_simple_mode_is_not_a_factory(self, index_panel):
        """An estimator instance is not read as a factory."""
        assert _fit(index_panel, transformer=centring()).is_factory_mode_ is False


# =============================================================================
# Mode fabrique
# =============================================================================
class TestFactoryMode:
    """A callable receives each entity key and returns that entity's transformer."""

    def test_factory_is_called_with_each_entity_key(self, index_panel):
        """The factory receives the normalized keys, in order of appearance."""
        seen = []

        def factory(entity_key):
            seen.append(entity_key)
            return centring()

        _fit(index_panel, transformer=factory)
        assert seen == [('FR',), ('DE',)]

    def test_factory_builds_entity_specific_transformers(self, index_panel):
        """``FR`` is centred, ``DE`` is left as is, as the factory decides."""
        output = PanelwiseTransformer(_mean_scaler_factory, time_col=None).fit_transform(index_panel)
        # Valeur d'or : a de FR - 3 ; a de DE inchangé
        assert output['a'].tolist() == [-2.0, -1.0, 3.0, 10.0, 20.0, 60.0]

    def test_factory_mode_is_detected(self, index_panel):
        """A plain callable is read as a factory."""
        assert _fit(index_panel, transformer=_mean_scaler_factory).is_factory_mode_ is True

    def test_entity_kwargs_are_ignored_with_a_warning(self, index_panel):
        """With a factory, ``entity_kwargs`` cannot apply: a warning says so."""
        with pytest.warns(UserWarning, match='entity_kwargs is ignored when transformer is a callable factory'):
            transformer = _fit(
                index_panel, transformer=_mean_scaler_factory, entity_kwargs={('FR',): {'with_mean': False}}
            )
        assert transformer.get_entity_transformer('FR').with_mean is True

    def test_factory_returning_none_is_refused(self, index_panel):
        """A factory must return a transformer for every entity."""
        with pytest.raises(ValueError, match=r"Transformer factory returned None for entity \('FR',\)"):
            _fit(index_panel, transformer=lambda entity_key: None)

    def test_factory_returning_a_non_transformer_is_refused(self, index_panel):
        """An object without ``fit`` and ``transform`` is not a transformer."""
        with pytest.raises(ValueError, match='Transformer factory returned invalid object'):
            _fit(index_panel, transformer=lambda entity_key: BaseEstimator())

    def test_class_instead_of_an_instance_is_refused(self, index_panel):
        """A class is neither an estimator instance nor a factory: ``clone`` refuses it."""
        with pytest.raises(TypeError, match='provide an instance of scikit-learn estimator instead of a class'):
            _fit(index_panel, transformer=StandardScaler)


# =============================================================================
# Mode entity_kwargs
# =============================================================================
class TestEntityKwargsMode:
    """``entity_kwargs`` sets per-entity parameters on the clone before fitting."""

    def test_entity_parameters_are_applied(self, index_panel):
        """``DE`` is left uncentred, ``FR`` keeps the base parameters."""
        output = PanelwiseTransformer(
            centring(), time_col=None, entity_kwargs={('DE',): {'with_mean': False}}
        ).fit_transform(index_panel)

        # Valeur d'or : a de FR - 3 ; a de DE inchangé
        assert output['a'].tolist() == [-2.0, -1.0, 3.0, 10.0, 20.0, 60.0]

    def test_parameters_are_set_on_the_clone_only(self, index_panel):
        """The base transformer keeps its own parameters."""
        base = centring()
        _fit(index_panel, transformer=base, entity_kwargs={('DE',): {'with_mean': False}})
        assert base.with_mean is True

    def test_default_kwargs_apply_to_unlisted_entities(self, index_panel):
        """Entities missing from ``entity_kwargs`` receive ``default_entity_kwargs``."""
        transformer = _fit(
            index_panel, transformer=centring(),
            entity_kwargs={('FR',): {'with_mean': True}}, default_entity_kwargs={'with_mean': False},
        )
        assert [t.with_mean for t in transformer.transformers_.values()] == [True, False]

    def test_default_kwargs_alone_are_not_used(self, index_panel):
        """Without ``entity_kwargs``, ``default_entity_kwargs`` is not applied (documented)."""
        transformer = _fit(index_panel, transformer=centring(), default_entity_kwargs={'with_mean': False})
        assert [t.with_mean for t in transformer.transformers_.values()] == [True, True]

    def test_unlisted_entity_without_default_keeps_the_base_parameters(self, index_panel):
        """An entity absent from ``entity_kwargs`` and without default is a plain clone."""
        transformer = _fit(index_panel, transformer=centring(), entity_kwargs={('FR',): {'with_mean': False}})
        assert transformer.get_entity_transformer('DE').with_mean is True

    def test_scalar_key_designates_the_one_level_entity(self, index_panel):
        """``'FR'`` designates the one-level entity ``('FR',)``, as everywhere in the package."""
        transformer = _fit(index_panel, transformer=centring(), entity_kwargs={'FR': {'with_mean': False}})
        assert transformer.get_entity_transformer('FR').with_mean is False

    def test_conflicting_notations_are_refused(self, index_panel):
        """``'FR'`` and ``('FR',)`` with different parameters are ambiguous."""
        entity_kwargs = {'FR': {'with_mean': False}, ('FR',): {'with_mean': True}}
        with pytest.raises(ValueError, match=r"Conflicting values for entity \('FR',\)"):
            _fit(index_panel, transformer=centring(), entity_kwargs=entity_kwargs)

    def test_entity_kwargs_parameter_is_left_as_given(self, index_panel):
        """The normalization of the keys does not rewrite the ``entity_kwargs`` parameter."""
        entity_kwargs = {'FR': {'with_mean': False}}
        transformer = _fit(index_panel, transformer=centring(), entity_kwargs=entity_kwargs)
        assert transformer.entity_kwargs == {'FR': {'with_mean': False}}

    def test_invalid_parameter_name_is_refused(self, index_panel):
        """An unknown parameter is refused, the entity and the parameter named in the message."""
        with pytest.raises(
            ValueError, match=r"Invalid parameters in entity_kwargs for entity \('FR',\): Invalid parameter 'nope'"
        ):
            _fit(index_panel, transformer=centring(), entity_kwargs={('FR',): {'nope': 1}})
