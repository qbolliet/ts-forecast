"""Input guards of ``ImputationWindowCalculator``.

Scenarios: constructor validation, ``fit`` input checks, accessors before ``fit`` and
observed columns without a detectable frequency.
"""

# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest

# Objet testé
from tsforecast.imputation._components.imputation_window import ImputationScope
from tsforecast.imputation._components.imputation_window import ImputationWindowCalculator

# Outils de test partagés du paquet
from tests.unit.imputation._components.imputation_window.helpers import (
    _fit_quietly,
    _frame,
    _warning_messages,
)


# =============================================================================
# Construction et validation des entrées
# =============================================================================
class TestConstructorValidation:
    """Parameter validation at ``__init__`` and storage of the parameters as received."""

    @pytest.mark.parametrize(
        'kwargs, match',
        [
            pytest.param({'coverage_threshold': 1.5}, 'coverage_threshold', id='threshold_above_1'),
            pytest.param({'coverage_threshold': -0.1}, 'coverage_threshold', id='threshold_below_0'),
            pytest.param({'imputation_scope': 'bogus'}, 'imputation_scope', id='unknown_imputation_scope'),
            pytest.param({'training_scope': 'bogus'}, 'training_scope', id='unknown_training_scope'),
            pytest.param({'training_coverage_threshold': 1.5}, 'training_coverage_threshold',
                         id='training_threshold_above_1'),
            pytest.param({'min_columns': 0}, 'min_columns', id='min_columns_zero'),
        ],
    )
    def test_invalid_parameter_raises(self, kwargs, match):
        """Each out-of-domain parameter raises a ValueError naming it."""
        with pytest.raises(ValueError, match=match):
            ImputationWindowCalculator(**kwargs)

    @pytest.mark.parametrize('threshold', [0.0, 1.0], ids=['zero', 'one'])
    def test_threshold_bounds_are_accepted(self, threshold):
        """0 and 1 are inside the closed domain [0, 1] of both thresholds."""
        calc = ImputationWindowCalculator(
            coverage_threshold=threshold, training_coverage_threshold=threshold
        )

        assert (calc.coverage_threshold, calc.training_coverage_threshold) == (threshold, threshold)

    def test_imputation_scope_rejects_unrestricted(self):
        """'unrestricted' is a training scope only: refused as a prediction scope.

        Imputing dates without any data has no meaning; 'unrestricted'
        is restricted to ``TrainingScope``.
        """
        with pytest.raises(ValueError, match='imputation_scope'):
            ImputationWindowCalculator(imputation_scope='unrestricted')

    def test_imputation_scope_vocabulary(self):
        """``ImputationScope`` lists exactly the four accepted prediction scopes."""
        from typing import get_args

        assert get_args(ImputationScope) == (
            'strict', 'extended_backward', 'extended_forward', 'extended_both'
        )

    def test_none_defaults_are_stored_unresolved(self):
        """The None training defaults are kept as received (sklearn get_params convention)."""
        calc = ImputationWindowCalculator(coverage_threshold=0.3, imputation_scope='extended_both')

        assert (calc.training_scope, calc.training_coverage_threshold) == (None, None)


class TestFitInputValidation:
    """Input guards of ``fit``."""

    def test_rejects_non_dataframe(self):
        """A list is not a DataFrame."""
        with pytest.raises(ValueError, match='DataFrame'):
            ImputationWindowCalculator().fit([1, 2, 3])

    def test_rejects_empty_dataframe(self):
        """An empty frame is refused explicitly."""
        with pytest.raises(ValueError, match='empty'):
            ImputationWindowCalculator().fit(pd.DataFrame(index=pd.DatetimeIndex([])))

    def test_rejects_non_temporal_index(self):
        """A RangeIndex carries no time axis."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        with pytest.raises(ValueError, match='DatetimeIndex or MultiIndex'):
            ImputationWindowCalculator().fit(df)

    def test_rejects_too_few_columns(self):
        """One column is below ``min_columns=2``."""
        df = _frame({'a': list(range(12))})

        with pytest.raises(ValueError, match='min_columns'):
            ImputationWindowCalculator(min_columns=2).fit(df)

    def test_rejects_time_series_without_any_observation(self):
        """A time series whose columns are all empty has no detectable frequency."""
        # Construction d'une série entièrement NaN : aucune fréquence de colonne
        df = _frame({'a': [np.nan] * 12, 'b': [np.nan] * 12})

        with pytest.raises(ValueError, match='Cannot detect any column frequency'):
            ImputationWindowCalculator().fit(df)

    def test_accepts_panel_where_no_entity_has_a_window(self):
        """Disjoint observation ranges in every entity: no strict window anywhere, no error.

        Both columns exist (none is excluded from the denominator), but their
        joint coverage never reaches 1.0. Like a time series, the panel
        gets all-False masks and one warning naming the entities; it does not
        raise, so ``HighFrequencyImputer`` never falls back to all-True masks.
        """
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        index = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
        first_half = [1.0] * 6 + [np.nan] * 6
        second_half = [np.nan] * 6 + [1.0] * 6
        df = pd.DataFrame({'a': first_half * 2, 'b': second_half * 2}, index=index)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            calc = ImputationWindowCalculator().fit(df)

        named = [m for m in _warning_messages(records) if 'no period' in m]
        assert len(named) == 1 and "[('A',), ('B',)]" in named[0]
        assert not calc.imputation_window_mask_.any()

    def test_fit_returns_self(self):
        """``fit`` returns the calculator itself, for chaining."""
        calc = ImputationWindowCalculator()

        assert calc.fit(_frame({'a': list(range(12)), 'b': list(range(12))})) is calc


class TestNotFitted:
    """Every public accessor requires ``fit`` first."""

    @pytest.mark.parametrize(
        'call',
        [
            pytest.param(lambda c: c.get_imputation_window_mask(), id='get_imputation_window_mask'),
            pytest.param(lambda c: c.get_mask_at_frequency('QS'), id='get_mask_at_frequency'),
            pytest.param(
                lambda c: c.get_columns_with_coverage(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-01')),
                id='get_columns_with_coverage',
            ),
        ],
    )
    def test_public_method_before_fit_raises(self, call):
        """The method raises a ValueError mentioning the missing fit."""
        with pytest.raises(ValueError, match='not fitted'):
            call(ImputationWindowCalculator())


class TestUndetectedFrequencyColumns:
    """An observed column without detectable frequency leaves the grid and the denominator.

    Detection maps such a column to ``None``: observed once,
    or on unrecognized spacing. Its period being unknown, it can neither
    veto nor stretch the window.
    """

    @staticmethod
    def _with_single_observation() -> pd.DataFrame:
        """Two dense monthly columns over 2020 and ``c`` observed once (June)."""
        return _frame({'a': list(range(12)), 'b': list(range(12)),
                       'c': [np.nan] * 5 + [1.0] + [np.nan] * 6})

    def test_single_observation_column_does_not_veto_the_window(self):
        """The strict window is the whole year of the two dense columns."""
        df = self._with_single_observation()
        calc = _fit_quietly(ImputationWindowCalculator(), df)

        assert (calc.imputation_strict_window_start_, calc.imputation_strict_window_end_) == (
            df.index[0], df.index[-1]
        )

    def test_excluded_column_is_reported(self):
        """Listed in ``undetected_frequency_columns_``, not as structurally absent; coverage (None, None)."""
        calc = _fit_quietly(ImputationWindowCalculator(), self._with_single_observation())

        assert (calc.undetected_frequency_columns_, calc.structurally_absent_columns_,
                calc.column_coverage_['c']) == (('c',), (), (None, None))

    def test_single_warning_names_the_column(self):
        """One warning for the exclusion, naming the column and the attribute."""
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            ImputationWindowCalculator().fit(self._with_single_observation())

        named = [m for m in _warning_messages(records) if 'cannot be detected' in m]
        assert (len(named), "['c']" in named[0], 'undetected_frequency_columns_' in named[0]) == (1, True, True)

    def test_stray_observation_does_not_stretch_the_grid(self):
        """A lone value after the dense columns leaves the grid on their 12 months."""
        # Construction : 'a', 'b' sur 12 mois puis NaN ; 'c' observée une fois au 18e mois
        nan6 = [np.nan] * 6
        df = _frame({'a': list(range(12)) + nan6, 'b': list(range(12)) + nan6,
                     'c': [np.nan] * 17 + [1.0]})
        calc = _fit_quietly(ImputationWindowCalculator(), df)

        assert (len(calc.get_imputation_window_mask()), calc.get_imputation_window_mask().index[-1]) == (
            12, df.index[11]
        )

    def test_irregularly_spaced_column_is_excluded(self):
        """Observations on days 0, 45 and 95 of a daily frame: no frequency, excluded."""
        irregular = [np.nan] * 100
        for day in (0, 45, 95):
            irregular[day] = 1.0
        df = _frame({'a': list(range(100)), 'b': list(range(100)), 'c': irregular}, freq='D')
        calc = _fit_quietly(ImputationWindowCalculator(), df)

        assert (calc.undetected_frequency_columns_, int(calc.imputation_strict_window_mask_.sum())) == (('c',), 100)

    def test_panel_exclusion_is_per_entity(self):
        """Only the entity observing ``c`` once loses it; both keep a full window."""
        dense = self._with_single_observation().assign(c=np.arange(12, dtype=float))
        panel = pd.concat({'A': dense, 'B': self._with_single_observation()}, names=['entity', 'date'])
        calc = _fit_quietly(ImputationWindowCalculator(), panel)

        counts = calc.get_imputation_window_mask().groupby(level=0).sum().to_dict()
        assert (calc.undetected_frequency_columns_, counts) == ({('A',): (), ('B',): ('c',)}, {'A': 12, 'B': 12})

    def test_excluded_column_is_never_covered(self):
        """``get_columns_with_coverage`` ignores the excluded column, even over its observation."""
        calc = _fit_quietly(ImputationWindowCalculator(), self._with_single_observation())

        assert calc.get_columns_with_coverage(pd.Timestamp('2020-06-01'), pd.Timestamp('2020-06-01')) == ['a', 'b']

    def test_only_undetected_columns_on_a_time_series_raises(self):
        """Every column observed once: no frequency at all, explicit error."""
        df = _frame({'a': [1.0] + [np.nan] * 11, 'b': [np.nan] * 11 + [1.0]})

        with pytest.raises(ValueError, match='Cannot detect any column frequency'):
            ImputationWindowCalculator().fit(df)
