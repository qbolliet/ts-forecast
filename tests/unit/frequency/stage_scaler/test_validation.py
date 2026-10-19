"""Tests of ``StageScaler`` argument validation.

Scenarios: ``scale_features`` keys and mode resolution, validation preserved through
``set_params``, fallbacks for incomparable frequencies and the disabled scaling.
"""

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objets testés
from tsforecast.frequency.stage_scaler import (
    DEFAULT_SCALE_KEY,
    DEFAULT_SCALE_MODE,
    StageScaler,
)

# Perturbations partagées
from tests.support.perturbations import with_special_column_names

# Outils de test partagés du paquet
from tests.unit.frequency.stage_scaler.helpers import _make_scaler, _monthly_grid


# =============================================================================
# Validation de la configuration
# =============================================================================
class TestValidation:
    """``scale_features`` validation, dict keys and mode resolution."""

    @pytest.mark.parametrize(
        'setting',
        [False, 'constant', 'calendar', {'m1': 'calendar'},
         {'m1': False, DEFAULT_SCALE_KEY: 'calendar'}],
        ids=['false', 'constant', 'calendar', 'dict', 'dict-default'],
    )
    def test_admissible_settings_are_stored_untouched(self, setting):
        """Every admissible setting is stored as received, same object."""
        scaler = StageScaler(scale_features=setting)
        assert scaler.scale_features is setting

    @pytest.mark.parametrize(
        'setting, message',
        [
            ('exact', r"scale_features must be one of"),
            (True, r"scale_features must be one of"),
            (1, r"scale_features must be one of"),
            (None, r"scale_features must be one of"),
            ({'m1': 'exact'}, r"scale_features values must be one of"),
            ({}, r"empty"),
        ],
        ids=['unknown-literal', 'true', 'one', 'none', 'dict-bad-value', 'empty-dict'],
    )
    def test_inadmissible_settings_raise(self, setting, message):
        """Anything else than False / 'constant' / 'calendar' (or their dict) raises."""
        with pytest.raises(ValueError, match=message):
            StageScaler(scale_features=setting)

    @pytest.mark.parametrize('form', ['scalar', 'dict'])
    @pytest.mark.parametrize('setting', [0, 0.0, np.False_], ids=['int-0', 'float-0', 'numpy-false'])
    def test_values_merely_equal_to_false_are_rejected(self, setting, form):
        """0, 0.0 and np.False_ are rejected: only False itself divides nothing.

        The resolution reads ``mode is False``; a value merely equal to False
        would otherwise pass the validation and then act as 'constant'.
        """
        value = setting if form == 'scalar' else {'m1': setting}
        with pytest.raises(ValueError, match=r"scale_features"):
            StageScaler(scale_features=value)

    def test_validate_columns_lists_every_unknown_key(self):
        """Keys absent from the real columns are listed, sorted; known ones are not."""
        scaler = _make_scaler(scale_features={'zz': 'calendar', 'yy': False, 'm1': 'constant'})
        with pytest.raises(ValueError, match=r"unknown columns : \['yy', 'zz'\]"):
            scaler.validate_columns(['m1', 'q1'])

    def test_default_key_is_never_reported_as_unknown(self):
        """'__default__' is not a column name and raises nothing."""
        _make_scaler(scale_features={DEFAULT_SCALE_KEY: 'calendar'}).validate_columns(['m1'])

    @pytest.mark.parametrize('setting', [False, 'constant', 'calendar'], ids=str)
    def test_validate_columns_ignores_the_scalar_form(self, setting):
        """The scalar form names no column: nothing to check, whatever the columns."""
        StageScaler(scale_features=setting).validate_columns([])

    def test_dict_keys_with_special_characters(self):
        """Column names with spaces, accents and symbols are plain dict keys."""
        frame = pd.DataFrame({'m1': [1.0], 'q1': [2.0]}, index=_monthly_grid(periods=1))
        renamed, mapping = with_special_column_names(frame)
        scaler = StageScaler(scale_features={mapping['m1']: 'calendar', mapping['q1']: False})

        # Validation et résolution sur les noms spéciaux, sans normalisation
        scaler.validate_columns(renamed.columns)
        assert [scaler.resolve_mode(mapping[c]) for c in ('m1', 'q1')] == ['calendar', False]

    @pytest.mark.parametrize(
        'setting, column, expected',
        [
            ('calendar', 'm1', 'calendar'),
            ('calendar', None, 'calendar'),
            (False, 'm1', False),
            ({'m1': 'calendar', DEFAULT_SCALE_KEY: False}, 'm1', 'calendar'),
            ({'m1': 'calendar', DEFAULT_SCALE_KEY: False}, 'q1', False),
            ({'m1': 'calendar', DEFAULT_SCALE_KEY: False}, None, False),
            ({'m1': 'calendar'}, 'q1', DEFAULT_SCALE_MODE),
            ({'m1': 'calendar'}, None, DEFAULT_SCALE_MODE),
        ],
        ids=['scalar', 'scalar-global', 'false', 'dict-own-key', 'dict-default',
             'dict-global-default', 'dict-no-default', 'dict-global-no-default'],
    )
    def test_resolve_mode(self, setting, column, expected):
        """Own entry, then '__default__', then 'constant' — the scalar form for all."""
        assert StageScaler(scale_features=setting).resolve_mode(column) == expected

    def test_default_scale_mode_is_constant(self):
        """The documented default of the module and of __init__ is 'constant'."""
        assert DEFAULT_SCALE_MODE == 'constant'
        assert StageScaler().scale_features == 'constant'


class TestValidationThroughSetParams:
    """``set_params`` does not bypass the ``__init__`` validation."""

    @pytest.mark.parametrize(
        'entry_point', ['fit', 'target_divisor', 'fit_scale_factor', 'feature_divisors', 'resolve_mode']
    )
    def test_inadmissible_setting_raises_after_set_params(self, entry_point):
        """fit and the divisor methods usable without fit revalidate scale_features.

        As in ``AggregationConstraint``, the
        components are usable alone, ``set_params`` (``GridSearchCV``,
        pipelines) included.
        """
        scaler = _make_scaler().set_params(scale_features='exact')
        X = pd.DataFrame({'m1': [1.0]}, index=_monthly_grid(periods=1))
        calls = {
            'fit': lambda: scaler.fit(X),
            'target_divisor': lambda: scaler.target_divisor('a1', source_freq='Y', pred_freq='M'),
            'fit_scale_factor': lambda: scaler.fit_scale_factor('a1', source_freq='Y', pred_freq='M'),
            'feature_divisors': lambda: scaler.feature_divisors(
                columns=['m1'], column_frequencies={'m1': 'M'}, ways={}, grid_freq='Y', stage_freq='M'
            ),
            'resolve_mode': lambda: scaler.resolve_mode('m1'),
        }
        with pytest.raises(ValueError, match=r"scale_features"):
            calls[entry_point]()

    def test_admissible_setting_through_set_params(self):
        """A legitimate set_params keeps working and is read by the next call."""
        scaler = _make_scaler().set_params(scale_features=False)
        assert scaler.target_divisor('a1', source_freq='Y', pred_freq='M') == 1.0


class TestFallbacks:
    """Incomparable frequencies: default divisor for a covariate, error for the target."""

    @pytest.mark.parametrize(
        'column_frequencies',
        [{}, {'x': 'zz'}],
        ids=['undetected', 'unknown-literal'],
    )
    def test_covariate_falls_back_on_default_divisor(self, column_frequencies):
        """An anchor column without a comparable frequency takes default_divisor."""
        divisors = StageScaler(default_divisor=12.0).feature_divisors(
            columns=['x'], column_frequencies=column_frequencies, ways={'x': 'raw_anchors'},
            grid_freq='M', stage_freq='M',
        )
        assert divisors['x'] == 12.0

    def test_default_argument_overrides_default_divisor(self):
        """The ``default`` argument of the call takes precedence over the constructor's."""
        divisors = StageScaler(default_divisor=12.0).feature_divisors(
            columns=['x'], column_frequencies={}, ways={'x': 'raw_anchors'},
            grid_freq='M', stage_freq='M', default=4.0,
        )
        assert divisors['x'] == 4.0

    @pytest.mark.parametrize('source_freq', ['zz', None], ids=['unknown-literal', 'none'])
    def test_target_with_incomparable_frequency_raises(self, source_freq):
        """The target's scale silently governs the model: never a fallback, always an error."""
        with pytest.raises(ValueError):
            StageScaler().target_divisor('v', source_freq=source_freq, pred_freq='M')


class TestScaleFeaturesFalse:
    """``False`` divides nothing: neither the features, nor the target."""

    def test_false_spares_features_target_and_baked_factor(self):
        """The target is a column like any other: False leaves it at its own scale."""
        scaler = _make_scaler(scale_features=False)
        divisors = scaler.feature_divisors(
            columns=['m1', 'q1', 'd1'],
            column_frequencies={'m1': 'M', 'q1': 'Q', 'd1': 'D'},
            ways={'m1': 'aggregate', 'q1': 'aggregate', 'd1': 'aggregate'},
            grid_freq='Y',
            stage_freq='M',
        )
        assert (
            divisors.tolist(),
            scaler.target_divisor('a1', source_freq='Y', pred_freq='M'),
            scaler.fit_scale_factor('a1', source_freq='Y', pred_freq='M'),
        ) == ([1.0, 1.0, 1.0], 1.0, 1.0)

    def test_false_on_y_ignores_produced_frequencies(self):
        """Even with mixed production frequencies, False divides no row."""
        index = pd.to_datetime(['2021-12-31', '2021-03-31'])
        divisor = _make_scaler(scale_features=False).target_divisor(
            'a1', source_freq='Y', pred_freq='M',
            produced_freq=pd.Series(['Y', 'Q'], index=index),
        )
        assert divisor == 1.0

    def test_false_on_y_only_through_the_dict_form(self):
        """A dict may spare the target alone, the features staying divided."""
        scaler = _make_scaler(scale_features={'a1': False, DEFAULT_SCALE_KEY: 'constant'})
        divisors = scaler.feature_divisors(
            columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
            grid_freq='Y', stage_freq='M',
        )
        assert (divisors['m1'], scaler.target_divisor('a1', 'Y', 'M')) == (12.0, 1.0)

    def test_calendar_feature_with_constant_target(self):
        """'calendar' on a feature, 'constant' on y — each divisor follows its own mode."""
        scaler = _make_scaler(scale_features={'d1': 'calendar', DEFAULT_SCALE_KEY: 'constant'})
        index = pd.date_range('2024-01-31', periods=2, freq='ME')
        divisors = scaler.feature_divisors(
            columns=['d1'], column_frequencies={'d1': 'D'}, ways={'d1': 'aggregate'},
            grid_freq='M', stage_freq='D', index=index,
        )
        target = scaler.target_divisor('a1', source_freq='M', pred_freq='D', index=index)
        assert (divisors['d1'].tolist(), target) == ([31.0, 29.0], 30.0)

    def test_false_in_dict_form_only_spares_that_feature(self):
        """False in a dict only neutralizes the targeted column."""
        divisors = _make_scaler(
            scale_features={'m1': False, DEFAULT_SCALE_KEY: 'constant'}
        ).feature_divisors(
            columns=['m1', 'q1'],
            column_frequencies={'m1': 'M', 'q1': 'Q'},
            ways={'m1': 'aggregate', 'q1': 'aggregate'},
            grid_freq='Y',
            stage_freq='M',
        )
        assert divisors.to_dict() == {'m1': 1.0, 'q1': 12.0}
