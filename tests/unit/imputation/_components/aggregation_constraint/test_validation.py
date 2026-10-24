"""Validation of the ``aggregation_constraint`` setting.

Scenarios: public vocabulary, admissible and rejected forms, resolution per column, dict
keys checked against the columns at fit, validation preserved through ``set_params`` and
the ``grid_freq`` check.
"""

# Modules de base
import typing

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objets testés
from tsforecast.imputation._components.aggregation_constraint import (
    AggregationConstraint,
    ConstraintKind,
    DEFAULT_CONSTRAINT_KEY,
    resolve_aggregation_constraint,
    validate_aggregation_constraint,
    validate_constraint_columns,
)
from tsforecast.imputation._components.covariate_materializer import CovariateMaterializer

# Perturbations partagées
from tests.support.perturbations import to_three_level_index

# Outils de test partagés du paquet
from tests.unit.imputation._components.aggregation_constraint.helpers import (
    _RAW_2021,
    _annual_observations,
    _monthly_grid,
    _raw_predictions,
    _to_panel,
)


# =============================================================================
# Vocabulaire public
# =============================================================================
class TestPublicVocabulary:
    """Public constants of the module."""

    def test_constraint_kind_admits_only_sum(self):
        """ConstraintKind is Literal['sum']: 'mean' and 'last' are gone."""
        assert typing.get_args(ConstraintKind) == ('sum',)

    def test_default_constraint_key(self):
        """The fallback key of the dict form is '__default__'."""
        assert DEFAULT_CONSTRAINT_KEY == '__default__'

    def test_default_setting_is_sum(self):
        """The constructor defaults to 'sum'."""
        assert AggregationConstraint().aggregation_constraint == 'sum'


# =============================================================================
# Validation du paramètre public
# =============================================================================
# Réglages admissibles : deux scalaires, et les dicts qui les associent à une
# colonne, clé de repli comprise
_ACCEPTED_SETTINGS = {
    'sum': 'sum',
    'none': None,
    'dict-sum': {'a1': 'sum'},
    'dict-none': {'a1': None},
    'dict-default-only': {DEFAULT_CONSTRAINT_KEY: None},
    'dict-mixed-default': {'a1': None, 'a2': 'sum', DEFAULT_CONSTRAINT_KEY: 'sum'},
    'dict-non-str-key': {1: 'sum'},
    'numpy-str': np.str_('sum'),
}

# Réglages refusés à la forme scalaire : agrégateurs retirés, casse et
# espaces, types non textuels — True / 0 ne doivent pas passer pour 'sum' / None
_REJECTED_SCALARS = {
    'mean': 'mean',
    'last': 'last',
    'upper-case': 'SUM',
    'capitalized': 'Sum',
    'padded': ' sum',
    'empty-string': '',
    'other-aggregator': 'median',
    'true': True,
    'false': False,
    'zero': 0,
    'one': 1,
    'float': 1.5,
    'nan': float('nan'),
    'list': ['sum'],
    'tuple': ('sum',),
    'set': {'sum'},
}

# Réglages refusés à la forme dictionnaire : une seule valeur fautive suffit
_REJECTED_DICTS = {
    'dict-mean': {'a1': 'mean'},
    'dict-last': {'a1': 'last'},
    'dict-default-mean': {DEFAULT_CONSTRAINT_KEY: 'mean'},
    'dict-true': {'a1': True},
    'dict-list': {'a1': ['sum']},
    'dict-one-bad-among-good': {'a1': 'sum', 'a2': None, 'a3': 'last'},
}


class TestValidateAggregationConstraint:
    """Admissible forms of ``aggregation_constraint``."""

    @pytest.mark.parametrize(
        'setting', list(_ACCEPTED_SETTINGS.values()), ids=list(_ACCEPTED_SETTINGS)
    )
    def test_admissible_settings_pass(self, setting):
        """'sum', None, and dicts of these values pass silently."""
        assert validate_aggregation_constraint(setting) is None

    @pytest.mark.parametrize(
        'setting', list(_ACCEPTED_SETTINGS.values()), ids=list(_ACCEPTED_SETTINGS)
    )
    def test_constructor_stores_admissible_settings_untouched(self, setting):
        """__init__ stores the setting as received (B3)."""
        assert AggregationConstraint(setting).aggregation_constraint is setting

    @pytest.mark.parametrize(
        'setting', list(_REJECTED_SCALARS.values()), ids=list(_REJECTED_SCALARS)
    )
    def test_rejected_scalar_raises_value_error(self, setting):
        """Any other scalar raises a ValueError."""
        with pytest.raises(ValueError, match='aggregation_constraint must be one of'):
            validate_aggregation_constraint(setting)

    @pytest.mark.parametrize(
        'setting', list(_REJECTED_DICTS.values()), ids=list(_REJECTED_DICTS)
    )
    def test_rejected_dict_raises_value_error(self, setting):
        """A dict holding one inadmissible value raises a ValueError."""
        with pytest.raises(ValueError, match='aggregation_constraint values must be one of'):
            validate_aggregation_constraint(setting)

    @pytest.mark.parametrize('setting', ['mean', 'last'], ids=['mean', 'last'])
    def test_withdrawn_aggregator_message_lists_admissible_forms(self, setting):
        """The message of a withdrawn aggregator lists the forms and names additive_transformer.

        The message lists the admissible forms ('sum', None, the dict
        form with its '__default__' key) and names ``additive_transformer`` as
        the answer to a non-additive column, and repeats the value.
        """
        with pytest.raises(ValueError) as error:
            AggregationConstraint(setting)
        message = str(error.value)

        # Formes admises, échappatoire et valeur reçue, toutes nommées
        expected_fragments = [
            "('sum', None)", 'dict', repr(DEFAULT_CONSTRAINT_KEY),
            'additive_transformer', f'got {setting!r}',
        ]
        assert [fragment for fragment in expected_fragments if fragment not in message] == []

    def test_dict_message_names_only_the_offending_entries(self):
        """The dict message quotes the invalid entries, not the valid ones."""
        with pytest.raises(ValueError) as error:
            validate_aggregation_constraint({'a1': 'sum', 'a2': 'mean', 'a3': 'last'})
        message = str(error.value)

        # Valeur d'or : seules a2 et a3 sont fautives
        assert "got {'a2': 'mean', 'a3': 'last'}" in message
        assert 'additive_transformer' in message

    def test_empty_dict_raises(self):
        """An empty dict designates no column and raises."""
        with pytest.raises(ValueError, match='cannot be empty'):
            AggregationConstraint({})

    @pytest.mark.parametrize(
        'setting',
        ['mean', 'last', True, {'a1': 'mean'}, {}],
        ids=['mean', 'last', 'true', 'dict-mean', 'empty-dict'],
    )
    def test_covariate_materializer_rejects_the_same_settings(self, setting):
        """CovariateMaterializer shares the validation: same error, same message.

        Both components carry the parameter, a single implementation
        keeps them from drifting.
        """
        with pytest.raises(ValueError) as reference:
            validate_aggregation_constraint(setting)
        with pytest.raises(ValueError) as materializer:
            CovariateMaterializer(aggregation_constraint=setting)

        assert str(materializer.value) == str(reference.value)


class TestResolveAggregationConstraint:
    """Constraint effectively applied to one column."""

    # Table de résolution : (réglage, colonne, réglage attendu)
    _CASES = {
        'scalar-sum': ('sum', 'a1', 'sum'),
        'scalar-none': (None, 'a1', None),
        'scalar-global': ('sum', None, 'sum'),
        'own-entry-none': ({'a1': None, DEFAULT_CONSTRAINT_KEY: 'sum'}, 'a1', None),
        'own-entry-sum': ({'a1': 'sum', DEFAULT_CONSTRAINT_KEY: None}, 'a1', 'sum'),
        'fallback-on-default-key': ({'a1': 'sum', DEFAULT_CONSTRAINT_KEY: None}, 'q1', None),
        'fallback-on-document-default': ({'a1': None}, 'q1', 'sum'),
        'global-reads-default-key': ({'a1': 'sum', DEFAULT_CONSTRAINT_KEY: None}, None, None),
        'global-without-default-key': ({'a1': None}, None, 'sum'),
    }

    @pytest.mark.parametrize(
        'setting, column, expected', list(_CASES.values()), ids=list(_CASES)
    )
    def test_module_function(self, setting, column, expected):
        """Own entry, else '__default__', else 'sum'."""
        assert resolve_aggregation_constraint(setting, column) == expected

    @pytest.mark.parametrize(
        'setting, column, expected', list(_CASES.values()), ids=list(_CASES)
    )
    def test_method_matches_module_function(self, setting, column, expected):
        """resolve_constraint gives the same answer as the module function."""
        assert AggregationConstraint(setting).resolve_constraint(column) == expected


class TestValidateConstraintColumns:
    """Dict keys are checked against the real columns at fit."""

    @pytest.mark.parametrize('setting', ['sum', None], ids=['sum', 'none'])
    def test_scalar_form_has_no_key_to_check(self, setting):
        """A scalar setting never names a column, even against no column."""
        assert validate_constraint_columns(setting, []) is None

    def test_known_keys_and_default_key_pass(self):
        """The '__default__' key is not a column name and is never reported."""
        setting = {'a1': 'sum', DEFAULT_CONSTRAINT_KEY: None}
        assert validate_constraint_columns(setting, pd.Index(['a1', 'q1'])) is None

    def test_unknown_keys_are_listed_sorted(self):
        """Every unknown key is named, in sorted order."""
        setting = {'zz': 'sum', 'a1': None, 'yy': None, DEFAULT_CONSTRAINT_KEY: 'sum'}
        with pytest.raises(ValueError, match=r"unknown columns : \['yy', 'zz'\]"):
            validate_constraint_columns(setting, iter(['a1', 'q1']))

    def test_method_delegates_to_module_function(self):
        """validate_columns raises like the module function."""
        with pytest.raises(ValueError, match=r"unknown columns : \['zz'\]"):
            AggregationConstraint({'zz': 'sum'}).validate_columns(['a1'])

    def test_fit_checks_the_keys(self):
        """fit validates the dict keys against the columns of X."""
        grid = _monthly_grid()
        X = pd.DataFrame({'a1': _RAW_2021}, index=grid)
        constraint = AggregationConstraint(
            {'zz': 'sum'}, period_frequencies='Y',
            observations=pd.DataFrame({'a1': _annual_observations().to_numpy()}, index=grid),
        )
        with pytest.raises(ValueError, match='unknown columns'):
            constraint.fit(X)


class TestValidationThroughSetParams:
    """Settings injected by set_params are validated again.

    'mean' is not accepted by the API. ``set_params`` (GridSearchCV,
    pipelines) skips ``__init__``: every public entry point validates again,
    so that 'mean' is never silently treated as an active constraint.
    """

    @staticmethod
    def _stage():
        """Return a one-column stage frame and its annual observation."""
        grid = _monthly_grid()
        X = pd.DataFrame({'a1': _RAW_2021}, index=grid)
        observations = pd.DataFrame({'a1': _annual_observations().to_numpy()}, index=grid)
        return X, observations

    def test_rejected_at_fit(self):
        """fit raises the ValueError of __init__."""
        X, observations = self._stage()
        constraint = AggregationConstraint('sum', 'Y', observations).set_params(
            aggregation_constraint='mean'
        )

        with pytest.raises(ValueError, match='additive_transformer'):
            constraint.fit(X)

    def test_rejected_at_transform_after_fit(self):
        """A setting changed after fit is rejected at transform."""
        X, observations = self._stage()
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)
        constraint.set_params(aggregation_constraint={'a1': 'last'})

        with pytest.raises(ValueError, match='additive_transformer'):
            constraint.transform(X)

    def test_rejected_at_rescale(self):
        """rescale, usable without fit, validates too."""
        constraint = AggregationConstraint('sum').set_params(aggregation_constraint='mean')

        with pytest.raises(ValueError, match='additive_transformer'):
            constraint.rescale(_raw_predictions(), _annual_observations(), 'Y')

    def test_admissible_setting_through_set_params_is_honoured(self):
        """A valid setting injected by set_params is applied."""
        X, observations = self._stage()
        constraint = AggregationConstraint('sum', 'Y', observations).set_params(
            aggregation_constraint=None
        )

        pd.testing.assert_frame_equal(constraint.fit_transform(X), X)


class TestGridFrequencyValidation:
    """grid_freq is validated once per call, never silently ignored.

    The truncation guard only exists through ``grid_freq``; an unreadable
    frequency or a forgotten entity would switch it off and hand a
    truncated period its whole total.
    """

    def test_unreadable_grid_frequency_raises(self):
        """An unparsable grid_freq raises, naming the parameter and the value."""
        grid = pd.date_range('2021-10-31', periods=3, freq='ME')
        values = pd.Series(1.0, index=grid)
        observations = pd.Series([np.nan, np.nan, 120.0], index=grid)

        with pytest.raises(ValueError, match="grid_freq must be a readable frequency, got 'zz'"):
            AggregationConstraint('sum').rescale(values, observations, 'Y', grid_freq='zz')

    def test_unreadable_frequency_inside_the_mapping_raises(self):
        """The values of the per-entity mapping are checked too."""
        values = _to_panel(_raw_predictions())

        with pytest.raises(ValueError, match='readable frequency, got 3'):
            AggregationConstraint('sum').rescale(
                values, _to_panel(_annual_observations()), 'Y',
                grid_freq={('FR',): 'M', ('DE',): 3},
            )

    def test_entity_missing_from_the_mapping_raises(self):
        """An entity the mapping does not name raises, listing the missing entities."""
        quarters = pd.date_range('2021-03-31', periods=2, freq='QE')
        values = pd.Series(30.0, index=pd.MultiIndex.from_product(
            [['DE'], quarters], names=['country', 'date']))
        observations = pd.Series([np.nan, 120.0], index=values.index)

        with pytest.raises(ValueError, match=r"does not cover entities \[\('DE',\)\]"):
            AggregationConstraint('sum').rescale(
                values, observations, 'Y', grid_freq={('FR',): 'M'}
            )

    def test_two_level_keys_on_a_three_level_index_raise(self):
        """('FR',) does not name the entity ('Zone euro', 'FR') of a three-level index."""
        values = to_three_level_index(_to_panel(_raw_predictions()).to_frame('a1'))['a1']
        observations = to_three_level_index(
            _to_panel(_annual_observations()).to_frame('a1')
        )['a1']

        with pytest.raises(ValueError, match='does not cover entities'):
            AggregationConstraint('sum').rescale(
                values, observations, 'Y', grid_freq={('FR',): 'M', ('DE',): 'M'}
            )

    def test_scalar_entity_keys_are_normalized(self):
        """'FR' and ('FR',) designate the same entity."""
        values = _to_panel(_raw_predictions())

        _rescaled, mask = AggregationConstraint('sum').rescale(
            values, _to_panel(_annual_observations()), 'Y',
            grid_freq={'FR': 'MS', 'DE': 'ME'},
        )

        assert mask.all()

    def test_mapping_may_name_extra_entities(self):
        """Keys of entities absent from the values are harmless."""
        values = _to_panel(_raw_predictions(), entities=('FR',))

        _rescaled, mask = AggregationConstraint('sum').rescale(
            values, _to_panel(_annual_observations(), entities=('FR',)), 'Y',
            grid_freq={'FR': 'M', 'IT': 'M'},
        )

        assert mask.all()

    def test_mapping_on_a_time_series_needs_the_empty_key(self):
        """A time series is the single entity (): a mapping must name it."""
        values = _raw_predictions()

        _rescaled, mask = AggregationConstraint('sum').rescale(
            values, _annual_observations(), 'Y', grid_freq={(): 'M'}
        )
        with pytest.raises(ValueError, match=r'does not cover entities \[\(\)\]'):
            AggregationConstraint('sum').rescale(
                values, _annual_observations(), 'Y', grid_freq={'FR': 'M'}
            )

        assert mask.all()
