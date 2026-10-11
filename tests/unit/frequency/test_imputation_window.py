"""Tests for tsforecast.frequency.imputation_window.

``ImputationWindowCalculator`` builds, per entity, a boolean coverage matrix
on the highest detected frequency (a quarterly observation covers its three
months), derives the coverage ratio of each grid date and exposes three
window masks (spec ``high_frequency_imputer2_architecture.md`` §7.1):

- ``'strict'``: coverage == 1.0, no extension (diagnostics only);
- ``'imputation'``: the strict window extended per ``imputation_scope`` /
  ``coverage_threshold`` (prediction grid);
- ``'training'``: the strict window extended per ``training_scope`` /
  ``training_coverage_threshold``, both falling back to the prediction
  parameters when ``None``; ``'unrestricted'`` lifts the restriction.

Contracts pinned here: strict ⊆ imputation, training relation per
``training_scope``; contiguous extensions stopping at the first date under
the threshold; ``coverage_threshold`` 0 and 1; unified return type on a panel
(§7.2: one ``pd.Series`` on a MultiIndex ``(entity..., date)``, never a dict,
while window bounds stay dicts keyed by entity tuples);
``entities_without_window_``; decision D39 (commit ``a3f5783``: columns with
no observation are excluded from the coverage denominator, reported by
``structurally_absent_columns_`` and a single ``UserWarning``); D40 (observed
columns of undetectable frequency excluded from the grid and the denominator,
``undetected_frequency_columns_``, a ``UserWarning`` of their own); grid
anchoring following the data when no column sits at the grid frequency;
``get_mask_at_frequency`` for every ``kind``, M→Q, M→Y, start / end
positions, per-entity targets, scalar vs tuple keys, entities without window
left out; ``get_columns_with_coverage``; the not-fitted guard of every public
method.

Covered symbols: ``ImputationWindowCalculator`` (``fit``,
``get_imputation_window_mask``, ``get_mask_at_frequency``,
``get_columns_with_coverage`` and every fitted attribute) and the
``ImputationScope`` vocabulary. Realistic datasets: ``heterogeneous_coverage_panel``
(per-entity coverage, ``climat_affaires`` never observed for Italie) and
``irregular_index_timeseries`` (annual anchors before the monthly grid).
Robustness: ``shuffle_rows``, ``reverse_entities``, ``to_three_level_index``.
"""
# Modules de base
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

# Objet testé
from tsforecast.frequency import ImputationScope
from tsforecast.frequency.imputation_window import ImputationWindowCalculator
from tsforecast.panel.utils import get_unique_panel_entities

# Perturbations partagées
from tests.support.perturbations import (
    reverse_entities,
    shuffle_rows,
    to_three_level_index,
)

# Les trois masques exposés par le calculateur (spec §7.1)
_KINDS = ('strict', 'imputation', 'training')

# Attributs de masque correspondant à chaque kind
_MASK_ATTRIBUTES = (
    'imputation_strict_window_mask_',
    'imputation_window_mask_',
    'training_window_mask_',
)

# Attributs de bornes : des dicts indexés par tuple d'entité sur un panel
_BOUND_ATTRIBUTES = (
    'imputation_window_start_',
    'imputation_window_end_',
    'imputation_strict_window_start_',
    'imputation_strict_window_end_',
    'training_window_start_',
    'training_window_end_',
)


# =============================================================================
# Constructeurs de données et utilitaires
# =============================================================================
def _frame(columns: Dict[str, List[float]], start: str = '2020-01-01', freq: str = 'MS') -> pd.DataFrame:
    """Build a float frame on a regular grid from ``{column: values}``."""
    n_rows = len(next(iter(columns.values())))
    index = pd.date_range(start, periods=n_rows, freq=freq, name='date')
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in columns.items()}, index=index)


def _make_panel(entities: List[tuple], n_periods: int = 36) -> pd.DataFrame:
    """Build a fully observed two-column monthly panel indexed by the given entity tuples."""
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='MS')
    n_levels = len(entities[0])
    # Construction du MultiIndex (niveaux d'entité + date)
    tuples = [(*entity, date) for entity in entities for date in dates]
    names = [f'level_{i}' for i in range(n_levels)] + ['date']
    index = pd.MultiIndex.from_tuples(tuples, names=names)
    n_rows = len(index)
    return pd.DataFrame({
        'a': np.arange(n_rows, dtype=float),
        'b': np.arange(n_rows, dtype=float) * 2,
    }, index=index)


def _make_df_with_shoulders():
    """Build a monthly frame with a strict window flanked by two half-coverage shoulders.

    Returns:
        Tuple ``(dates, df)``: both columns observed on ``dates[10:15]``
        (strict window), ``a`` alone on ``dates[8:10]`` and ``dates[15:17]``
        (coverage 0.5, kept by a 0.5 threshold, dropped by 0.9).
    """
    dates = pd.date_range('2020-01-01', periods=20, freq='MS')
    df = pd.DataFrame({'b': np.nan, 'a': np.nan}, index=dates, dtype=float)
    # Fenêtre stricte : couverture totale sur [10, 15)
    df.loc[dates[10:15], ['a', 'b']] = 1.0
    # Épaules à couverture == 0.5, de part et d'autre de la fenêtre stricte
    df.loc[dates[8:10], 'a'] = 1.0
    df.loc[dates[15:17], 'a'] = 1.0
    return dates, df


def _make_df_with_gap():
    """Build a monthly frame whose backward extension must stop at a coverage gap.

    Returns:
        Tuple ``(dates, df)``: strict window ``dates[10:15]``; coverage 0.5
        on ``dates[8:10]`` and ``dates[0:4]``, coverage 0 on ``dates[4:8]``.
    """
    dates = pd.date_range('2020-01-01', periods=20, freq='MS')
    df = pd.DataFrame({'b': np.nan, 'a': np.nan}, index=dates, dtype=float)
    df.loc[dates[10:15], ['a', 'b']] = 1.0
    df.loc[dates[0:4], 'a'] = 1.0
    df.loc[dates[8:10], 'a'] = 1.0
    return dates, df


def _make_df_without_strict_window():
    """Build a monthly frame whose two columns never overlap (coverage never 1.0).

    Returns:
        Tuple ``(dates, df)``: ``b`` on ``dates[0:8]``, ``a`` on ``dates[12:20]``.
    """
    dates = pd.date_range('2020-01-01', periods=20, freq='MS')
    df = pd.DataFrame({'b': np.nan, 'a': np.nan}, index=dates, dtype=float)
    df.loc[dates[0:8], 'b'] = 1.0
    df.loc[dates[12:20], 'a'] = 1.0
    return dates, df


def _make_panel_with_none_entity() -> pd.DataFrame:
    """Build a panel whose entity ``B`` has an index frequency that cannot be detected.

    Entity ``A`` is a regular monthly series (24 months); entity ``B`` carries
    four irregularly spaced dates, so ``detect_index_frequency`` yields None
    for it.
    """
    dates_a = pd.date_range('2020-01-01', periods=24, freq='MS')
    dates_b = pd.DatetimeIndex(['2020-01-01', '2020-01-03', '2020-06-17', '2023-02-28'])
    tuples = [('A', date) for date in dates_a] + [('B', date) for date in dates_b]
    index = pd.MultiIndex.from_tuples(tuples, names=['entity', 'date'])
    n_rows = len(index)
    return pd.DataFrame({
        'b': np.arange(n_rows, dtype=float),
        'a': np.arange(n_rows, dtype=float) * 2,
    }, index=index)


def _make_panel_with_empty_entity() -> pd.DataFrame:
    """Build a regular monthly panel whose entity ``B`` observes nothing at all.

    ``B``'s index frequency is detected (regular monthly rows) but its
    coverage grid is empty: no observation, hence no period to expand.
    """
    dates = pd.date_range('2020-01-01', periods=12, freq='MS')
    index = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
    values = list(range(12)) + [np.nan] * 12
    return pd.DataFrame({'a': values, 'b': values}, index=index, dtype=float)


def _fit_quietly(calc: ImputationWindowCalculator, data: pd.DataFrame) -> ImputationWindowCalculator:
    """Fit ``calc`` on ``data``, silencing the UserWarnings that are not the test subject."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return calc.fit(data)


def _entities_of(series: pd.Series) -> set:
    """Return the set of entity key tuples present in a MultiIndex mask Series."""
    entity_index = series.index.droplevel(-1)
    return {key if isinstance(key, tuple) else (key,) for key in entity_index}


def _months(bounds: Dict[tuple, Optional[pd.Timestamp]]) -> Dict[str, Optional[str]]:
    """Render per-entity window bounds as ``{last entity level: 'YYYY-MM'}``."""
    return {
        key[-1]: (value.strftime('%Y-%m') if value is not None else None)
        for key, value in bounds.items()
    }


def _warning_messages(records) -> List[str]:
    """Extract the UserWarning messages of a ``warnings.catch_warnings(record=True)`` list."""
    return [str(r.message) for r in records if issubclass(r.category, UserWarning)]


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

        Imputing dates without any data has no meaning; spec §7 restricts
        'unrestricted' to ``TrainingScope`` (ANO-FREQ-016, spec corrected).
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

    def test_rejects_panel_where_no_entity_has_a_window(self):
        """Disjoint observation ranges in every entity: no strict window anywhere.

        Both columns exist (none is excluded from the denominator), but their
        joint coverage never reaches 1.0.
        """
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        index = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
        first_half = [1.0] * 6 + [np.nan] * 6
        second_half = [np.nan] * 6 + [1.0] * 6
        df = pd.DataFrame({'a': first_half * 2, 'b': second_half * 2}, index=index)

        with pytest.raises(ValueError, match='No imputation window'):
            _fit_quietly(ImputationWindowCalculator(), df)

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


# =============================================================================
# Fenêtre stricte
# =============================================================================
class TestStrictWindow:
    """Strict window = dates where every (present) column covers the grid date."""

    def test_single_column(self):
        """With one column, coverage reduces to the presence of that column."""
        df = _frame({'a': [np.nan, np.nan] + list(range(10))})
        calc = ImputationWindowCalculator(min_columns=1).fit(df)

        # Valeur d'or : premières observations aux deux premiers mois manquants
        assert (calc.imputation_strict_window_start_, calc.imputation_strict_window_end_) == (
            df.index[2], df.index[-1]
        )

    @pytest.mark.parametrize(
        'freq, quarter_dates, expected',
        [
            pytest.param('MS', ['2020-01-01', '2020-04-01', '2020-07-01'],
                         ('2020-01-01', '2020-09-01'), id='start_position'),
            pytest.param('ME', ['2020-03-31', '2020-06-30', '2020-09-30'],
                         ('2020-01-31', '2020-09-30'), id='end_position'),
        ],
    )
    def test_quarterly_observation_covers_its_months(self, freq, quarter_dates, expected):
        """A quarterly observation covers the three months of its quarter.

        ``a`` is monthly over 2020, ``q`` quarterly for Q1 to Q3: the strict
        window ends with the last month of Q3, whatever the anchoring.
        """
        df = _frame({'a': list(range(12)), 'q': [np.nan] * 12}, freq=freq)
        df.loc[pd.DatetimeIndex(quarter_dates), 'q'] = 1.0
        calc = ImputationWindowCalculator().fit(df)

        # Valeur d'or : T1-T3 couvrent janvier à septembre
        assert (calc.imputation_strict_window_start_, calc.imputation_strict_window_end_) == (
            pd.Timestamp(expected[0]), pd.Timestamp(expected[1])
        )

    def test_non_contiguous_strict_window_warns_and_reports_outer_bounds(self):
        """A hole in the strict window warns and the bounds stay [first, last]."""
        # Construction : 'a' manquante sur trois mois au milieu de 20 mois couverts
        dates = pd.date_range('2020-01-01', periods=20, freq='MS')
        df = pd.DataFrame({'b': 1.0, 'a': 1.0}, index=dates)
        df.loc[dates[5:8], 'a'] = np.nan
        calc = ImputationWindowCalculator()

        with pytest.warns(UserWarning, match='not contiguous'):
            calc.fit(df)

        assert (calc.imputation_strict_window_start_, calc.imputation_strict_window_end_,
                int(calc.imputation_strict_window_mask_.sum())) == (dates[0], dates[-1], 17)

    @pytest.mark.parametrize(
        'kwargs, expected',
        [
            pytest.param({}, {'Imputation', 'Training'}, id='both_windows_single_date'),
            pytest.param({'training_scope': 'unrestricted'}, {'Imputation'}, id='imputation_only'),
            pytest.param({'imputation_scope': 'extended_both', 'training_scope': 'strict'},
                         {'Training'}, id='training_only'),
        ],
    )
    def test_single_observation_window_warns_per_mask(self, kwargs, expected):
        """Each mask warns on its own when it holds a single date.

        ``b`` covers January to June, ``a`` June to December: June alone is
        strict, every other month sits at coverage 0.5.
        """
        df = _frame({'b': [1.0] * 6 + [np.nan] * 6, 'a': [np.nan] * 5 + [1.0] * 7})

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            ImputationWindowCalculator(**kwargs).fit(df)

        single = {
            message.split()[0] for message in _warning_messages(records)
            if 'contains only one observation' in message
        }
        assert single == expected

    @pytest.mark.parametrize(
        'freq, last_date',
        [('MS', '2023-12-01'), ('ME', '2023-12-31')],
        ids=['start_position', 'end_position'],
    )
    def test_grid_stops_at_the_last_observed_period(self, freq, last_date):
        """The grid upper bound is exclusive: no extra period after the last observation.

        ``pd.date_range`` treats ``end`` as inclusive; on a start-anchored grid
        the exclusive bound is itself a grid date and would add one period.
        """
        start = '2019-01-01' if freq == 'MS' else '2019-01-31'
        data = _frame({'a': list(range(60)), 'b': list(range(60))}, start=start, freq=freq)

        mask = _fit_quietly(ImputationWindowCalculator(), data).get_imputation_window_mask()

        assert (len(mask), mask.index[-1]) == (len(data), pd.Timestamp(last_date))

    def test_grid_extends_past_the_last_row_to_the_end_of_its_period(self):
        """A quarterly observation on the last row stretches the grid to the end of its quarter.

        The raw mask may therefore be longer than the data (docstring of the
        class); the aligned mask has exactly the data index.
        """
        # Construction : 7 mois, 'q' trimestrielle en janvier, avril et juillet
        df = _frame({'a': list(range(7)), 'q': [1.0, np.nan, np.nan, 1.0, np.nan, np.nan, 1.0]})
        calc = ImputationWindowCalculator().fit(df)

        raw = calc.get_imputation_window_mask()
        aligned = calc.get_imputation_window_mask(df)

        # Valeur d'or : grille janvier-septembre (T3 couvre jusqu'à septembre),
        # fenêtre stricte janvier-juillet ('a' s'arrête en juillet)
        assert (len(raw), raw.index[-1], int(raw.sum()), bool(aligned.all()), len(aligned)) == (
            9, pd.Timestamp('2020-09-01'), 7, True, 7
        )


# =============================================================================
# Extensions de la fenêtre
# =============================================================================
class TestScopeExtensions:
    """Backward / forward extensions of the strict window, governed by the threshold."""

    @pytest.mark.parametrize(
        'scope, expected',
        [
            pytest.param('strict', (10, 14), id='strict'),
            pytest.param('extended_backward', (8, 14), id='extended_backward'),
            pytest.param('extended_forward', (10, 16), id='extended_forward'),
            pytest.param('extended_both', (8, 16), id='extended_both'),
        ],
    )
    def test_window_bounds_follow_imputation_scope(self, scope, expected):
        """Each scope activates the half-coverage shoulders on its own side(s)."""
        dates, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(coverage_threshold=0.5, imputation_scope=scope, min_columns=2)
        calc.fit(df)

        # Valeur d'or : épaules [8, 10) et [15, 17) à couverture 0.5 >= seuil
        assert (calc.imputation_window_start_, calc.imputation_window_end_) == (
            dates[expected[0]], dates[expected[1]]
        )

    @pytest.mark.parametrize('scope', ['extended_backward', 'extended_forward', 'extended_both'])
    def test_strict_bounds_do_not_move_with_scope(self, scope):
        """The strict bounds stay on the strict window whatever the scope."""
        dates, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(coverage_threshold=0.5, imputation_scope=scope, min_columns=2)
        calc.fit(df)

        assert (calc.imputation_strict_window_start_, calc.imputation_strict_window_end_) == (
            dates[10], dates[14]
        )

    def test_extension_stops_at_first_gap(self):
        """Walking backward, the extension stops at the first date under the threshold.

        ``dates[0:4]`` have coverage 0.5 too, but lie beyond a zero-coverage gap.
        """
        dates, df = _make_df_with_gap()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='extended_backward', min_columns=2
        ).fit(df)

        mask = calc.imputation_window_mask_
        assert (bool(mask.loc[dates[0:4]].any()), bool(mask.loc[dates[8:10]].all())) == (False, True)

    def test_threshold_zero_extends_to_the_grid_edge(self):
        """``coverage_threshold=0``: no date is below the threshold, even at coverage 0.

        The backward extension crosses the gap and reaches the first grid date.
        """
        dates, df = _make_df_with_gap()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.0, imputation_scope='extended_backward', min_columns=2
        ).fit(df)

        # Valeur d'or : grille [0, 15), entièrement active
        assert (calc.imputation_window_start_, int(calc.imputation_window_mask_.sum())) == (dates[0], 15)

    @pytest.mark.parametrize('scope', ['extended_backward', 'extended_forward', 'extended_both'])
    def test_threshold_one_never_extends(self, scope):
        """``coverage_threshold=1``: only strict dates qualify, the scope is inert."""
        _, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(coverage_threshold=1.0, imputation_scope=scope, min_columns=2)
        calc.fit(df)

        pd.testing.assert_series_equal(calc.imputation_window_mask_, calc.imputation_strict_window_mask_)

    def test_threshold_above_shoulder_coverage_blocks_extension(self):
        """A 0.9 threshold rejects the 0.5 shoulders."""
        dates, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.9, imputation_scope='extended_both', min_columns=2
        ).fit(df)

        assert (calc.imputation_window_start_, calc.imputation_window_end_) == (dates[10], dates[14])

    def test_window_already_spanning_the_grid_is_unchanged(self):
        """Full coverage: nothing to extend on either side."""
        df = _frame({'a': list(range(12)), 'b': list(range(12))})
        calc = ImputationWindowCalculator(imputation_scope='extended_both').fit(df)

        pd.testing.assert_series_equal(calc.imputation_window_mask_, calc.imputation_strict_window_mask_)


# =============================================================================
# Les trois masques
# =============================================================================
class TestThreeMasks:
    """Relations between the strict, imputation and training masks (spec §7.1)."""

    @pytest.mark.parametrize('scope', ['strict', 'extended_backward', 'extended_forward', 'extended_both'])
    def test_strict_is_included_in_imputation(self, scope):
        """No strict date is missing from the imputation mask."""
        _, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(imputation_scope=scope, min_columns=2).fit(df)

        assert not (calc.imputation_strict_window_mask_ & ~calc.imputation_window_mask_).any()

    def test_extension_adds_dates_to_the_strict_mask(self):
        """Under 'extended_both', the inclusion is strict: the shoulders are added."""
        dates, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(imputation_scope='extended_both', min_columns=2).fit(df)

        added = calc.imputation_window_mask_ & ~calc.imputation_strict_window_mask_
        assert list(added.index[added]) == list(dates[8:10]) + list(dates[15:17])

    def test_strict_mask_does_not_depend_on_scope(self):
        """The extensions derive new masks and never mutate the strict one."""
        _, df = _make_df_with_shoulders()
        narrow = ImputationWindowCalculator(imputation_scope='strict', min_columns=2).fit(df)
        wide = ImputationWindowCalculator(
            imputation_scope='extended_both', training_scope='unrestricted', min_columns=2
        ).fit(df)

        pd.testing.assert_series_equal(wide.imputation_strict_window_mask_, narrow.imputation_strict_window_mask_)

    @pytest.mark.parametrize(
        'training_scope, expected_active',
        [
            pytest.param(None, list(range(8, 17)), id='none_follows_imputation'),
            pytest.param('strict', list(range(10, 15)), id='strict'),
            pytest.param('extended_backward', list(range(8, 15)), id='extended_backward'),
            pytest.param('extended_forward', list(range(10, 17)), id='extended_forward'),
            pytest.param('unrestricted', list(range(8, 17)), id='unrestricted_whole_grid'),
        ],
    )
    def test_training_mask_follows_training_scope(self, training_scope, expected_active):
        """The training window is set by ``training_scope``, the prediction one by ``imputation_scope``.

        ``imputation_scope='extended_both'`` throughout; the grid spans
        ``dates[8:17]`` (first to last observed period).
        """
        dates, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(
            imputation_scope='extended_both', training_scope=training_scope, min_columns=2
        ).fit(df)

        mask = calc.training_window_mask_
        assert list(mask.index[mask]) == [dates[i] for i in expected_active]

    def test_training_scope_leaves_the_prediction_window_untouched(self):
        """Widening ``training_scope`` changes the training mask only."""
        _, df = _make_df_with_shoulders()
        narrow = ImputationWindowCalculator(training_scope='strict', min_columns=2).fit(df)
        wide = ImputationWindowCalculator(training_scope='extended_both', min_columns=2).fit(df)

        pd.testing.assert_series_equal(wide.imputation_window_mask_, narrow.imputation_window_mask_)

    def test_training_bounds_follow_the_training_mask(self):
        """``training_window_start_`` / ``_end_`` match the active range of the training mask."""
        dates, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(training_scope='extended_forward', min_columns=2).fit(df)

        assert (calc.training_window_start_, calc.training_window_end_) == (dates[10], dates[16])

    def test_training_coverage_threshold_is_independent(self):
        """Two training thresholds give two training masks, one prediction mask."""
        dates, df = _make_df_with_shoulders()
        kwargs = dict(imputation_scope='extended_both', training_scope='extended_both', min_columns=2)
        permissive = ImputationWindowCalculator(training_coverage_threshold=0.5, **kwargs).fit(df)
        demanding = ImputationWindowCalculator(training_coverage_threshold=0.9, **kwargs).fit(df)

        # Le seuil 0.9 écarte les épaules à 0.5, la fenêtre de prédiction reste au seuil 0.5
        assert (int(permissive.training_window_mask_.sum()), int(demanding.training_window_mask_.sum()),
                permissive.imputation_window_mask_.equals(demanding.imputation_window_mask_)) == (9, 5, True)

    def test_training_threshold_none_follows_coverage_threshold(self):
        """``training_coverage_threshold=None`` falls back on ``coverage_threshold``."""
        _, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.9, imputation_scope='strict', training_scope='extended_both', min_columns=2
        ).fit(df)

        pd.testing.assert_series_equal(calc.training_window_mask_, calc.imputation_strict_window_mask_)


class TestNoStrictWindow:
    """B23: without any strict window, the three masks still exist (all False, or all True)."""

    def test_three_masks_exist_and_are_all_false(self):
        """The extensions cannot activate anything from an empty strict mask."""
        _, df = _make_df_without_strict_window()
        calc = ImputationWindowCalculator(imputation_scope='extended_both', min_columns=2)

        with pytest.warns(UserWarning, match='no period'):
            calc.fit(df)

        assert all(
            isinstance(getattr(calc, attr), pd.Series) and not getattr(calc, attr).any()
            for attr in _MASK_ATTRIBUTES
        )

    def test_bounds_are_none(self):
        """Every window bound stays undetermined."""
        _, df = _make_df_without_strict_window()
        calc = _fit_quietly(ImputationWindowCalculator(imputation_scope='extended_both', min_columns=2), df)

        assert [getattr(calc, attr) for attr in _BOUND_ATTRIBUTES] == [None] * len(_BOUND_ATTRIBUTES)

    def test_unrestricted_training_scope_is_honoured(self):
        """Only 'unrestricted' yields a usable training window in that case."""
        _, df = _make_df_without_strict_window()
        calc = ImputationWindowCalculator(
            imputation_scope='extended_both', training_scope='unrestricted', min_columns=2
        )

        with pytest.warns(UserWarning, match='unrestricted'):
            calc.fit(df)

        assert (bool(calc.training_window_mask_.all()), bool(calc.imputation_window_mask_.any())) == (True, False)

    def test_panel_without_window_raises_by_default(self):
        """On a panel, no strict window for any entity is a hard error by default."""
        _, df = _make_df_without_strict_window()
        panel = pd.concat({'A': df, 'B': df}, names=['entity', 'date'])

        with pytest.raises(ValueError, match='No imputation window'):
            _fit_quietly(ImputationWindowCalculator(min_columns=2), panel)

    def test_panel_without_window_is_trainable_under_unrestricted(self):
        """Under 'unrestricted', the panel stays trainable and warns that nothing is imputable."""
        _, df = _make_df_without_strict_window()
        panel = pd.concat({'A': df, 'B': df}, names=['entity', 'date'])
        calc = ImputationWindowCalculator(training_scope='unrestricted', min_columns=2)

        with pytest.warns(UserWarning, match='no value can be imputed'):
            calc.fit(panel)

        assert (bool(calc.training_window_mask_.loc['A'].all()),
                bool(calc.imputation_window_mask_.loc['A'].any())) == (True, False)


# =============================================================================
# Types de retour sur un panel (spec §7.2)
# =============================================================================
class TestPanelReturnTypes:
    """Masks are one MultiIndex Series, bounds are dicts keyed by entity tuples."""

    @pytest.mark.parametrize('kind', _KINDS)
    def test_masks_are_a_single_multiindex_series(self, kind):
        """Each mask is a boolean Series on ``(entity..., date)``, never a dict."""
        df = _make_panel([('A', 'x'), ('B', 'y')])
        calc = ImputationWindowCalculator(imputation_scope='extended_both', min_columns=2).fit(df)

        mask = calc.get_imputation_window_mask(kind=kind)

        assert (type(mask), type(mask.index), mask.dtype, list(mask.index.names),
                _entities_of(mask), set(df.index) <= set(mask.index)) == (
            pd.Series, pd.MultiIndex, bool, list(df.index.names), {('A', 'x'), ('B', 'y')}, True
        )

    @pytest.mark.parametrize('kind', _KINDS)
    def test_raw_mask_is_the_fitted_attribute(self, kind):
        """Without ``data``, the method returns the fitted attribute designated by ``kind``."""
        df = _make_panel([('A',), ('B',)])
        calc = ImputationWindowCalculator().fit(df)
        attribute = dict(zip(_KINDS, _MASK_ATTRIBUTES))[kind]

        assert calc.get_imputation_window_mask(kind=kind) is getattr(calc, attribute)

    def test_coverage_by_date_is_a_multiindex_series(self):
        """``coverage_by_date_`` follows the same unified type."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))

        assert (isinstance(calc.coverage_by_date_.index, pd.MultiIndex),
                _entities_of(calc.coverage_by_date_)) == (True, {('A',), ('B',)})

    @pytest.mark.parametrize(
        'entities',
        [
            pytest.param([('A',), ('B',)], id='one_entity_level'),
            pytest.param([('A', 'x'), ('B', 'y')], id='two_entity_levels'),
        ],
    )
    @pytest.mark.parametrize(
        'attribute',
        list(_BOUND_ATTRIBUTES) + ['column_coverage_', 'index_freq_', 'structurally_absent_columns_',
                                   'undetected_frequency_columns_'],
    )
    def test_dict_attributes_are_keyed_by_entity_tuples(self, entities, attribute):
        """Every per-entity dict uses the keys of ``get_unique_panel_entities``.

        Tuples even with a single entity level: ``('A',)``, never ``'A'``
        (§3.4 of the review), so a KeyError genuinely signals a bug.
        """
        df = _make_panel(entities)
        calc = ImputationWindowCalculator().fit(df)

        value = getattr(calc, attribute)
        assert (isinstance(value, dict), set(value)) == (True, set(get_unique_panel_entities(df)))

    def test_bounds_are_timestamps_per_entity(self):
        """Bounds stay scalar per entity: one Timestamp each."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))

        assert all(
            isinstance(v, pd.Timestamp)
            for attr in _BOUND_ATTRIBUTES for v in getattr(calc, attr).values()
        )

    def test_time_series_masks_are_on_a_datetime_index(self):
        """On a time series: Series on a DatetimeIndex, no entity without window."""
        calc = ImputationWindowCalculator(imputation_scope='extended_both').fit(
            _frame({'a': list(range(24)), 'b': list(range(24))})
        )

        assert (all(isinstance(getattr(calc, a).index, pd.DatetimeIndex) for a in _MASK_ATTRIBUTES),
                isinstance(calc.coverage_by_date_.index, pd.DatetimeIndex),
                calc.entities_without_window_) == (True, True, ())


# =============================================================================
# Entités sans fenêtre
# =============================================================================
class TestEntitiesWithoutWindow:
    """Entities whose window cannot be determined contribute False rows and are listed."""

    def test_undetected_index_frequency_gives_false_rows(self):
        """An entity with an irregular index contributes its frame rows, all False."""
        df = _make_panel_with_none_entity()
        calc = _fit_quietly(ImputationWindowCalculator(min_columns=2), df)

        b_rows = df.loc['B'].index
        assert calc.entities_without_window_ == (('B',),)
        for attr in _MASK_ATTRIBUTES:
            b_slice = getattr(calc, attr).xs('B', level=0)
            pd.testing.assert_index_equal(b_slice.index, b_rows)
            assert not b_slice.any(), attr
            assert getattr(calc, attr).xs('A', level=0).any(), attr

    def test_undetected_index_frequency_keeps_every_entity_key(self):
        """Bounds are None and ``column_coverage_`` still has an entry for the entity."""
        calc = _fit_quietly(ImputationWindowCalculator(min_columns=2), _make_panel_with_none_entity())

        assert ([getattr(calc, attr)[('B',)] for attr in _BOUND_ATTRIBUTES], calc.column_coverage_[('B',)]) == (
            [None] * len(_BOUND_ATTRIBUTES), None
        )

    def test_empty_coverage_grid_gives_false_rows(self):
        """An entity observing nothing (regular index) has no grid: False rows, listed."""
        df = _make_panel_with_empty_entity()
        calc = _fit_quietly(ImputationWindowCalculator(), df)

        counts = calc.imputation_window_mask_.groupby(level=0).agg(['sum', 'size'])
        assert (calc.entities_without_window_, counts.loc['B'].tolist(), counts.loc['A'].tolist(),
                calc.imputation_window_start_[('B',)], calc.column_coverage_[('B',)]) == (
            (('B',),), [0, 12], [12, 12], None, None
        )

    def test_empty_entity_reports_no_excluded_column(self):
        """An entity without window carries ``()`` in both exclusion attributes.

        The attributes are only meaningful for an entity with a coverage grid:
        ``B``'s columns were never weighed against one another (documented
        rule, ANO-FREQ-017).
        """
        calc = _fit_quietly(ImputationWindowCalculator(), _make_panel_with_empty_entity())

        assert (calc.structurally_absent_columns_, calc.undetected_frequency_columns_) == (
            {('A',): (), ('B',): ()}, {('A',): (), ('B',): ()}
        )

    @pytest.mark.parametrize(
        'builder',
        [
            pytest.param(_make_panel_with_none_entity, id='undetected_index_frequency'),
            pytest.param(_make_panel_with_empty_entity, id='no_observation'),
        ],
    )
    def test_entities_without_window_are_named_by_one_warning(self, builder):
        """Losing a whole entity is never silent: one warning names every such entity."""
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            ImputationWindowCalculator(min_columns=2).fit(builder())

        named = [m for m in _warning_messages(records) if 'without a determinable window' in m]
        assert (len(named), "[('B',)]" in named[0]) == (1, True)

    def test_no_warning_when_every_entity_has_a_window(self):
        """A panel where every entity has a window fits silently."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))

        assert calc.entities_without_window_ == ()

    def test_unrestricted_does_not_revive_an_entity_without_window(self):
        """'unrestricted' is all True for the fitted entity, all False for the other."""
        df = _make_panel_with_none_entity()
        calc = _fit_quietly(ImputationWindowCalculator(training_scope='unrestricted', min_columns=2), df)

        aligned = calc.get_imputation_window_mask(df, kind='training')
        assert (bool(aligned.loc['A'].all()), bool(aligned.loc['B'].any())) == (True, False)

    def test_training_scope_none_follows_imputation_scope_on_a_panel(self):
        """Default training window = prediction window, entity without window included."""
        calc = _fit_quietly(ImputationWindowCalculator(imputation_scope='extended_both', min_columns=2), _make_panel_with_none_entity())

        pd.testing.assert_series_equal(calc.training_window_mask_, calc.imputation_window_mask_)

    @pytest.mark.parametrize(
        'builder',
        [
            pytest.param(_make_panel_with_none_entity, id='undetected_index_frequency'),
            pytest.param(_make_panel_with_empty_entity, id='no_observation'),
        ],
    )
    def test_entity_without_window_is_absent_from_mask_at_frequency(self, builder):
        """``get_mask_at_frequency`` leaves out every entity without window, whatever the cause.

        Absent means "no window", unlike ``False`` ("outside the window"):
        the imputer and ``TrainingSetBuilder`` read an absent entity as
        unrestricted. An entity with a known frequency but no observation
        used to be converted to all-False rows instead.
        """
        calc = _fit_quietly(ImputationWindowCalculator(min_columns=2), builder())

        assert _entities_of(calc.get_mask_at_frequency('QS')) == {('A',)}

    def test_dict_target_need_not_name_an_entity_without_window(self):
        """A per-entity target listing only the entities with a window is enough."""
        calc = _fit_quietly(ImputationWindowCalculator(), _make_panel_with_empty_entity())

        assert calc.get_mask_at_frequency({'A': 'QS'}).tolist() == [True] * 4

    def test_columns_with_coverage_is_empty_for_an_entity_without_window(self):
        """No coverage dict for the entity: no covered column."""
        calc = _fit_quietly(ImputationWindowCalculator(min_columns=2), _make_panel_with_none_entity())

        assert calc.get_columns_with_coverage(
            pd.Timestamp('2020-01-01'), pd.Timestamp('2023-12-31'), entity=('B',)
        ) == []


# =============================================================================
# D39 : colonnes structurellement absentes
# =============================================================================
class TestStructurallyAbsentColumns:
    """D39: a column never observed (per entity) is left out of the coverage denominator.

    It says nothing about the temporal extent of its entity: kept in the
    denominator, it would cap coverage at (n-1)/n, hence an empty strict
    window and an entity where nothing can be imputed.
    """

    @staticmethod
    def _panel_with_absent_column() -> pd.DataFrame:
        """Panel where ``b`` is never observed for entity ``B`` only."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        index = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
        column_b = np.concatenate([np.arange(12, dtype=float), [np.nan] * 12])
        return pd.DataFrame({'a': np.arange(24, dtype=float), 'b': column_b}, index=index)

    def test_absent_column_does_not_veto_the_entity_window(self):
        """The entity missing a column keeps a full window."""
        calc = ImputationWindowCalculator()
        with pytest.warns(UserWarning, match='coverage denominator'):
            calc.fit(self._panel_with_absent_column())

        counts = calc.get_imputation_window_mask().groupby(level=0).sum().to_dict()
        assert (counts, calc.entities_without_window_) == ({'A': 12, 'B': 12}, ())

    def test_absent_columns_are_reported_per_entity(self):
        """The exclusion is named per entity; the reported coverage stays (None, None)."""
        calc = _fit_quietly(ImputationWindowCalculator(), self._panel_with_absent_column())

        assert (calc.structurally_absent_columns_, calc.column_coverage_[('B',)]['b']) == (
            {('A',): (), ('B',): ('b',)}, (None, None)
        )

    def test_empty_column_of_a_time_series_is_absent(self):
        """On a time series, an empty column is structurally absent too (ANO-UTILS-042).

        It is excluded from the denominator without shrinking the strict
        window of the two other columns.
        """
        df = _frame({'a': list(range(12)), 'b': list(range(12)), 'c': [np.nan] * 12})
        calc = ImputationWindowCalculator()
        with pytest.warns(UserWarning, match='coverage denominator'):
            calc.fit(df)

        assert (calc.structurally_absent_columns_,
                calc.imputation_strict_window_start_,
                calc.imputation_strict_window_end_) == (('c',), df.index[0], df.index[-1])

    def test_no_absent_column_is_silent(self):
        """Without an absent column, no warning and an empty attribute."""
        df = _frame({'a': list(range(12)), 'b': list(range(12))})
        calc = ImputationWindowCalculator()
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            calc.fit(df)

        assert calc.structurally_absent_columns_ == ()

    def test_refit_resets_panel_only_attributes(self):
        """Re-fitting a panel-fitted calculator on a time series resets the per-entity attributes."""
        calc = _fit_quietly(ImputationWindowCalculator(min_columns=2), _make_panel_with_none_entity())
        calc.fit(_frame({'a': list(range(12)), 'b': list(range(12))}))

        assert (calc.structurally_absent_columns_, calc.undetected_frequency_columns_,
                calc.entities_without_window_) == ((), (), ())


class TestStructurallyAbsentOnHeterogeneousPanel:
    """D39 on the realistic panel: ``climat_affaires`` is never observed for Italie."""

    def test_absent_columns_are_exact(self, heterogeneous_coverage_panel):
        """Only (Italie, climat_affaires) is structurally absent."""
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        assert calc.structurally_absent_columns_ == {
            ('Allemagne',): (), ('France',): (), ('Italie',): ('climat_affaires',),
        }

    def test_single_warning_names_the_entity_and_column(self, heterogeneous_coverage_panel):
        """Exactly one denominator warning per fit, naming Italie and its column."""
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            ImputationWindowCalculator().fit(heterogeneous_coverage_panel)

        denominator = [m for m in _warning_messages(records) if 'coverage denominator' in m]
        assert (len(denominator), "('Italie',): ['climat_affaires']" in denominator[0]) == (1, True)

    def test_italy_keeps_a_strict_window(self, heterogeneous_coverage_panel):
        """Without D39, Italie's coverage would never exceed 6/7: its strict window exists."""
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        # Valeur d'or : production industrielle dès 2019-06 ; dépenses et balance jusqu'en 2023-12
        assert (calc.imputation_strict_window_start_[('Italie',)],
                calc.imputation_strict_window_end_[('Italie',)]) == (
            pd.Timestamp('2019-06-01'), pd.Timestamp('2023-12-01')
        )

    @pytest.mark.parametrize(
        'entity, date, expected',
        [
            # Italie : 6 colonnes au dénominateur ; en 2019-01, seule la production manque
            pytest.param('Italie', '2019-01-01', 5 / 6, id='italie_denominator_6'),
            # France : 7 colonnes ; en 2018-01, seule la production manque
            pytest.param('France', '2018-01-01', 6 / 7, id='france_denominator_7'),
        ],
    )
    def test_coverage_ratio_uses_present_columns_only(self, heterogeneous_coverage_panel, entity, date, expected):
        """The coverage ratio divides by the entity's present columns."""
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        assert calc.coverage_by_date_.loc[(entity, pd.Timestamp(date))] == pytest.approx(expected)

    def test_reported_coverage_of_the_absent_column_is_none(self, heterogeneous_coverage_panel):
        """The exclusion bears on the denominator, not on ``column_coverage_``."""
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        assert calc.column_coverage_[('Italie',)]['climat_affaires'] == (None, None)


# =============================================================================
# Jeux réalistes
# =============================================================================
class TestHeterogeneousCoveragePanel:
    """Per-entity windows consistent with each entity's own coverage.

    Golden bounds derived by hand from ``HETEROGENEOUS_PANEL_COUNTRIES``
    (monthly grids FR 2018-01→2024-07, DE 2018-07→2024-04, IT 2019-01→2024-07;
    production from FR 2018-06, DE 2019-01, IT 2019-06; last monthly value of
    inflation / unemployment, last quarter of GDP, last year of the annual
    series withheld; annual history from 2015 (FR) / 2016 (DE, IT)).
    """

    @pytest.mark.parametrize(
        'scope, threshold, expected_start, expected_end',
        [
            pytest.param('strict', 0.5,
                         {'Allemagne': '2019-01', 'France': '2018-06', 'Italie': '2019-06'},
                         {'Allemagne': '2023-12', 'France': '2023-12', 'Italie': '2023-12'},
                         id='strict'),
            # 6/7 (FR, DE) et 5/6 (IT) avant la production ; balance seule (1/7, 1/6) plus tôt
            pytest.param('extended_backward', 0.5,
                         {'Allemagne': '2018-07', 'France': '2018-01', 'Italie': '2019-01'},
                         {'Allemagne': '2023-12', 'France': '2023-12', 'Italie': '2023-12'},
                         id='extended_backward'),
            # 2024 : sans les séries annuelles, 5/7 (FR), 6/7 (DE), 4/6 (IT) ; dernier mois 2/7, 2/7, 1/6
            pytest.param('extended_forward', 0.5,
                         {'Allemagne': '2019-01', 'France': '2018-06', 'Italie': '2019-06'},
                         {'Allemagne': '2024-03', 'France': '2024-06', 'Italie': '2024-06'},
                         id='extended_forward'),
            # Seuil 0 : jusqu'au bout de la grille propre à chaque entité (DE finit en 2024-04)
            pytest.param('extended_forward', 0.0,
                         {'Allemagne': '2019-01', 'France': '2018-06', 'Italie': '2019-06'},
                         {'Allemagne': '2024-04', 'France': '2024-07', 'Italie': '2024-07'},
                         id='extended_forward_threshold_0'),
            # Seuil 0 : jusqu'à la première ancre annuelle de chaque entité
            pytest.param('extended_backward', 0.0,
                         {'Allemagne': '2016-01', 'France': '2015-01', 'Italie': '2016-01'},
                         {'Allemagne': '2023-12', 'France': '2023-12', 'Italie': '2023-12'},
                         id='extended_backward_threshold_0'),
        ],
    )
    def test_window_bounds_per_entity(self, heterogeneous_coverage_panel, scope, threshold,
                                      expected_start, expected_end):
        """Each entity's window follows its own coverage."""
        calc = _fit_quietly(
            ImputationWindowCalculator(imputation_scope=scope, coverage_threshold=threshold),
            heterogeneous_coverage_panel,
        )

        assert (_months(calc.imputation_window_start_), _months(calc.imputation_window_end_)) == (
            expected_start, expected_end
        )

    def test_annual_anchor_rows_stay_outside_the_window(self, heterogeneous_coverage_panel):
        """Rows before an entity's monthly grid (annual anchors only) are never in the window."""
        calc = _fit_quietly(
            ImputationWindowCalculator(imputation_scope='extended_both'), heterogeneous_coverage_panel
        )
        aligned = calc.get_imputation_window_mask(heterogeneous_coverage_panel)

        # Lignes antérieures au début de la grille mensuelle de chaque entité
        monthly_start = {'France': '2018-01-01', 'Allemagne': '2018-07-01', 'Italie': '2019-01-01'}
        dates = aligned.index.get_level_values('date')
        countries = aligned.index.get_level_values('country')
        before = np.array([d < pd.Timestamp(monthly_start[c]) for c, d in zip(countries, dates)])
        assert (int(before.sum()), bool(aligned[before].any())) == (9, False)

    def test_entities_all_have_a_window(self, heterogeneous_coverage_panel):
        """Every entity of the realistic panel has a window (index frequency detected)."""
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        assert (calc.entities_without_window_, set(calc.index_freq_.values())) == ((), {'M'})


class TestIrregularIndexTimeseries:
    """Annual anchors from 2015 precede a monthly grid starting in 2018."""

    def test_annual_anchors_do_not_extend_the_strict_window(self, irregular_index_timeseries):
        """The strict window starts with industrial production (2019-01), not in 2015."""
        calc = ImputationWindowCalculator().fit(irregular_index_timeseries)

        # Valeur d'or : production dès 2019-01 ; balance annuelle (dernière année retirée) jusqu'en 2023
        assert (calc.imputation_strict_window_start_, calc.imputation_strict_window_end_) == (
            pd.Timestamp('2019-01-01'), pd.Timestamp('2023-12-01')
        )

    def test_backward_extension_stops_at_the_monthly_grid(self, irregular_index_timeseries):
        """2018: 4 columns of 5 (0.8); before 2018: the annual balance alone (0.2 < 0.5)."""
        calc = ImputationWindowCalculator(imputation_scope='extended_backward').fit(irregular_index_timeseries)

        assert calc.imputation_window_start_ == pd.Timestamp('2018-01-01')

    def test_grid_spans_the_annual_anchors_but_they_stay_outside(self, irregular_index_timeseries):
        """The grid starts in 2015-01; the 2015-2017 anchor rows are False once aligned."""
        calc = ImputationWindowCalculator(imputation_scope='extended_both').fit(irregular_index_timeseries)
        aligned = calc.get_imputation_window_mask(irregular_index_timeseries)

        anchors = aligned.loc[:'2017-12-31']
        assert (calc.imputation_window_mask_.index[0], len(anchors), bool(anchors.any())) == (
            pd.Timestamp('2015-01-01'), 3, False
        )

    def test_threshold_zero_reaches_the_first_anchor(self, irregular_index_timeseries):
        """``coverage_threshold=0`` activates every covered grid date, back to 2015-01."""
        calc = ImputationWindowCalculator(
            imputation_scope='extended_backward', coverage_threshold=0.0
        ).fit(irregular_index_timeseries)

        assert calc.imputation_window_start_ == pd.Timestamp('2015-01-01')


# =============================================================================
# get_imputation_window_mask
# =============================================================================
class TestGetImputationWindowMask:
    """Raw mask without ``data``, mask reindexed on ``data.index`` otherwise."""

    def test_aligned_mask_has_exactly_the_data_index(self):
        """Rows outside the fitted grid are False; strict dates are True."""
        df = _frame({'a': list(range(24)), 'b': list(range(24))})
        calc = ImputationWindowCalculator().fit(df)
        data_to_align = pd.DataFrame(index=df.index.append(pd.DatetimeIndex(['2022-06-01'])))

        aligned = calc.get_imputation_window_mask(data_to_align)

        assert (aligned.index.equals(data_to_align.index), aligned.dtype,
                bool(aligned.loc[pd.Timestamp('2022-06-01')]), bool(aligned.loc[df.index].all())) == (
            True, bool, False, True
        )

    def test_alignment_emits_no_pandas_deprecation(self):
        """Rows missing from the grid are filled without the deprecated object downcast."""
        df = _frame({'a': list(range(12)), 'b': list(range(12))})
        calc = ImputationWindowCalculator().fit(df)
        target = pd.DataFrame(index=df.index.append(pd.DatetimeIndex(['2023-01-01'])))

        with warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            aligned = calc.get_imputation_window_mask(target)

        assert (aligned.dtype, int(aligned.sum())) == (bool, 12)

    def test_raw_mask_without_data(self):
        """Without ``data``, the fitted imputation mask is returned."""
        calc = ImputationWindowCalculator().fit(_frame({'a': list(range(12)), 'b': list(range(12))}))

        assert calc.get_imputation_window_mask() is calc.imputation_window_mask_

    @pytest.mark.parametrize('kind', _KINDS)
    def test_aligned_mask_per_kind(self, kind):
        """``kind`` selects the mask that is aligned."""
        _, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(
            imputation_scope='extended_backward', training_scope='unrestricted', min_columns=2
        ).fit(df)
        attribute = dict(zip(_KINDS, _MASK_ATTRIBUTES))[kind]

        expected = getattr(calc, attribute).reindex(df.index, fill_value=False)
        pd.testing.assert_series_equal(calc.get_imputation_window_mask(df, kind=kind), expected)

    def test_invalid_kind_raises(self):
        """A wrong ``kind`` fails instead of falling back on the imputation window."""
        calc = ImputationWindowCalculator().fit(_frame({'a': list(range(12)), 'b': list(range(12))}))

        with pytest.raises(ValueError, match='kind'):
            calc.get_imputation_window_mask(kind='bogus')

    def test_aligns_panel_rows_directly(self):
        """On a panel, ``(entity, date)`` rows are aligned, including unknown ones (False)."""
        df = _make_panel([('A',), ('B',)], n_periods=12)
        calc = ImputationWindowCalculator().fit(df)
        extra = pd.MultiIndex.from_tuples([('C', pd.Timestamp('2020-01-01'))], names=df.index.names)
        target = pd.DataFrame(index=df.index.append(extra))

        aligned = calc.get_imputation_window_mask(target)

        assert (int(aligned.sum()), bool(aligned.loc[('C', pd.Timestamp('2020-01-01'))])) == (24, False)


# =============================================================================
# get_mask_at_frequency
# =============================================================================
class TestGetMaskAtFrequency:
    """A lower-frequency period is True iff all its sub-periods are in the window."""

    def test_month_to_quarter(self):
        """Strict window February-December: Q1 False, Q2-Q4 True."""
        df = _frame({'a': list(range(12)), 'b': [np.nan] + list(range(11))})
        calc = ImputationWindowCalculator().fit(df)

        quarterly = calc.get_mask_at_frequency('QS')

        assert (list(quarterly.index), quarterly.tolist()) == (
            list(pd.date_range('2020-01-01', periods=4, freq='QS')), [False, True, True, True]
        )

    def test_month_to_year_full_years(self):
        """Twelve months covered: the year is True."""
        calc = ImputationWindowCalculator().fit(_frame({'a': list(range(24)), 'b': list(range(24))}))

        assert calc.get_mask_at_frequency('YS').tolist() == [True, True]

    def test_month_to_year_partial_year(self):
        """Eleven months out of twelve: the year is False."""
        # Construction : un seul mois de 2021 hors fenêtre stricte ('b' manquante en juin)
        b = [float(i) for i in range(24)]
        b[17] = np.nan
        df = _frame({'a': list(range(24)), 'b': b})
        calc = _fit_quietly(ImputationWindowCalculator(), df)

        assert calc.get_mask_at_frequency('YS').tolist() == [True, False]

    def test_start_anchored_grid_keeps_start_anchoring(self):
        """'YE' asked on an MS grid gives a YS index, reindexable on start-anchored data."""
        calc = ImputationWindowCalculator().fit(_frame({'a': list(range(24)), 'b': list(range(24))}))

        yearly = calc.get_mask_at_frequency('YE')

        assert list(yearly.index) == [pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')]

    def test_end_anchored_grid_keeps_end_anchoring(self):
        """'QS' asked on an ME grid gives a quarter-end index."""
        df = _frame({'a': list(range(12)), 'b': list(range(12))}, start='2020-01-31', freq='ME')
        calc = ImputationWindowCalculator().fit(df)

        quarterly = calc.get_mask_at_frequency('QS')

        assert (list(quarterly.index), quarterly.tolist()) == (
            list(pd.date_range('2020-03-31', periods=4, freq='QE')), [True] * 4
        )

    def test_same_frequency_is_the_identity(self):
        """Asking for the grid's own frequency returns the mask unchanged."""
        calc = ImputationWindowCalculator().fit(_frame({'a': list(range(12)), 'b': list(range(12))}))

        pd.testing.assert_series_equal(calc.get_mask_at_frequency('MS'), calc.imputation_window_mask_)

    def test_higher_target_frequency_raises(self):
        """A daily target from a monthly mask is not an aggregation."""
        calc = ImputationWindowCalculator().fit(_frame({'a': list(range(12)), 'b': list(range(12))}))

        with pytest.raises(ValueError, match='should be higher'):
            calc.get_mask_at_frequency('D')

    def test_kind_is_propagated(self):
        """Strict, imputation and training masks give their own quarterly masks."""
        _, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(
            imputation_scope='extended_both', training_scope='strict', min_columns=2
        ).fit(df)

        strict = calc.get_mask_at_frequency('QS', kind='strict')
        imputation = calc.get_mask_at_frequency('QS', kind='imputation')
        training = calc.get_mask_at_frequency('QS', kind='training')

        # Valeur d'or : grille sept. 2020 - mai 2021 ; stricte nov. 2020 - mars 2021 (T1 2021 seul complet),
        # étendue sept. 2020 - mai 2021 (T4 2020 et T1 2021 complets)
        assert (int(strict.sum()), int(imputation.sum()), training.equals(strict)) == (1, 2, True)

    def test_kind_defaults_to_imputation(self):
        """The default ``kind`` preserves the historical behaviour."""
        _, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(imputation_scope='extended_both', min_columns=2).fit(df)

        pd.testing.assert_series_equal(
            calc.get_mask_at_frequency('QS'), calc.get_mask_at_frequency('QS', kind='imputation')
        )

    def test_invalid_kind_raises(self):
        """A wrong ``kind`` is rejected here too."""
        calc = ImputationWindowCalculator().fit(_frame({'a': list(range(12)), 'b': list(range(12))}))

        with pytest.raises(ValueError, match='kind'):
            calc.get_mask_at_frequency('QS', kind='bogus')

    def test_panel_result_is_a_multiindex_series(self):
        """On a panel: one Series on ``(entity..., date)``, start-anchored like the grid."""
        df = _make_panel([('A', 'x'), ('B', 'y')])
        calc = ImputationWindowCalculator().fit(df)

        yearly = calc.get_mask_at_frequency('YS')

        assert (isinstance(yearly.index, pd.MultiIndex), yearly.dtype, _entities_of(yearly),
                list(yearly.index.names), yearly.tolist()) == (
            True, bool, {('A', 'x'), ('B', 'y')}, list(df.index.names), [True] * 6
        )

    def test_scalar_and_tuple_keys_are_equivalent(self):
        """Scalar entity keys are normalized to tuples at the public boundary."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))

        pd.testing.assert_series_equal(
            calc.get_mask_at_frequency({'A': 'QS', 'B': 'YS'}),
            calc.get_mask_at_frequency({('A',): 'QS', ('B',): 'YS'}),
        )

    def test_per_entity_targets(self):
        """A dict target converts each entity to its own frequency."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))

        result = calc.get_mask_at_frequency({('A',): 'QS', ('B',): 'YS'})

        # Valeur d'or : 36 mois = 12 trimestres pour A, 3 années pour B
        assert result.groupby(level=0).size().to_dict() == {'A': 12, 'B': 3}

    def test_per_entity_targets_on_the_heterogeneous_panel(self, heterogeneous_coverage_panel):
        """Strict window at a frequency per entity, golden counts by hand.

        FR strict 2018-06→2023-12: complete quarters 2018-Q3→2023-Q4 (22) on a
        grid 2015-01→2024-07 (39 quarters); DE 2019-01→2023-12: 5 years on 9;
        IT 2019-06→2023-12: 2019-Q3→2023-Q4 (18) on 35 quarters.
        """
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        result = calc.get_mask_at_frequency(
            {'France': 'QS', 'Allemagne': 'YS', 'Italie': 'QS'}, kind='strict'
        )

        assert result.groupby(level=0).agg(['sum', 'size']).to_dict('index') == {
            'Allemagne': {'sum': 5, 'size': 9},
            'France': {'sum': 22, 'size': 39},
            'Italie': {'sum': 18, 'size': 35},
        }

    def test_year_end_on_the_heterogeneous_panel(self, heterogeneous_coverage_panel):
        """Italie, strict window at 'YE': 2019 (from June) and 2024 incomplete, 2020-2023 complete."""
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        italy = calc.get_mask_at_frequency('YE', kind='strict').xs('Italie').loc['2019':]

        assert italy.tolist() == [False, True, True, True, True, False]


# =============================================================================
# get_columns_with_coverage
# =============================================================================
class TestGetColumnsWithCoverage:
    """Columns whose coverage range overlaps the query range."""

    @staticmethod
    def _monthly_and_quarterly() -> pd.DataFrame:
        """``a`` monthly over 2020, ``q`` quarterly for Q1-Q3 (covers January to September)."""
        q = [1.0, np.nan, np.nan, 1.0, np.nan, np.nan, 1.0] + [np.nan] * 5
        return _frame({'a': list(range(12)), 'q': q})

    @pytest.mark.parametrize(
        'start, end, expected',
        [
            pytest.param('2020-01-01', '2020-03-01', ['a', 'q'], id='both_covered'),
            pytest.param('2020-10-01', '2020-12-01', ['a'], id='after_quarterly_coverage'),
            pytest.param('2020-09-01', '2020-10-01', ['a', 'q'], id='overlap_on_one_month'),
            pytest.param('2021-02-01', '2021-03-01', [], id='outside_everything'),
        ],
    )
    def test_time_series(self, start, end, expected):
        """Overlap test ``col_start <= end and col_end >= start`` on a time series."""
        calc = ImputationWindowCalculator().fit(self._monthly_and_quarterly())

        assert calc.get_columns_with_coverage(pd.Timestamp(start), pd.Timestamp(end)) == expected

    def test_panel_with_entity_returns_a_list(self):
        """With an entity: the list of that entity's covered columns."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))

        assert calc.get_columns_with_coverage(
            pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01'), entity=('A',)
        ) == ['a', 'b']

    def test_panel_scalar_entity_is_normalized(self):
        """A scalar entity is accepted and normalized to its tuple."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))
        start, end = pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01')

        assert calc.get_columns_with_coverage(start, end, entity='A') == calc.get_columns_with_coverage(
            start, end, entity=('A',)
        )

    def test_panel_without_entity_returns_a_dict(self):
        """Without an entity: a dict over every entity tuple."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A', 'x'), ('B', 'y')]))

        assert calc.get_columns_with_coverage(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01')) == {
            ('A', 'x'): ['a', 'b'], ('B', 'y'): ['a', 'b'],
        }

    def test_unknown_entity_raises(self):
        """An unknown key is a KeyError, not a silent empty list."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))

        with pytest.raises(KeyError):
            calc.get_columns_with_coverage(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01'), entity=('ZZ',))

    def test_unobserved_column_is_never_covered(self, heterogeneous_coverage_panel):
        """Italie, early 2024: annual series ended in 2023, climat_affaires never observed."""
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        assert calc.get_columns_with_coverage(
            pd.Timestamp('2024-01-01'), pd.Timestamp('2024-03-01'), entity='Italie'
        ) == ['production_industrielle', 'inflation_ipc', 'taux_chomage', 'pib_trimestriel']


# =============================================================================
# Ancrage de la grille et fréquences non détectées
# =============================================================================
class TestGridAnchoring:
    """The coverage grid is anchored like the data, so masks align on it."""

    @pytest.mark.parametrize(
        'start, freq',
        [
            pytest.param('2020-01-01', 'MS', id='month_start'),
            pytest.param('2020-01-31', 'ME', id='month_end'),
            pytest.param('2020-01-01', 'QS', id='quarter_start'),
            pytest.param('2020-03-31', 'QE', id='quarter_end'),
            pytest.param('2020-01-01', 'YS', id='year_start'),
            pytest.param('2020-12-31', 'YE', id='year_end'),
            pytest.param('2020-01-01', 'D', id='daily'),
            pytest.param('2020-01-01', 'B', id='business_daily'),
            pytest.param('2020-01-05', 'W-SUN', id='weekly'),
            pytest.param('2020-01-01', 'h', id='hourly'),
        ],
    )
    def test_grid_matches_a_fully_observed_index(self, start, freq):
        """Fully observed data: the raw mask index is the data index."""
        df = _frame({'a': list(range(30)), 'b': list(range(30))}, start=start, freq=freq)
        calc = ImputationWindowCalculator().fit(df)

        assert list(calc.get_imputation_window_mask().index) == list(df.index)

    @staticmethod
    def _coarse_columns_only(start: str, freq: str) -> pd.DataFrame:
        """Monthly index carrying only a quarterly and an annual column (2020-2021).

        Observations sit on the first (start anchoring) or last (end
        anchoring) month of each quarter / year, so both columns cover all
        24 months.
        """
        dates = pd.date_range(start, periods=24, freq=freq)
        df = pd.DataFrame({'q': np.nan, 'y': np.nan}, index=dates)
        quarter_months = [1, 4, 7, 10] if freq == 'MS' else [3, 6, 9, 12]
        year_month = 1 if freq == 'MS' else 12
        df.loc[dates.month.isin(quarter_months), 'q'] = 1.0
        df.loc[dates.month == year_month, 'y'] = 2.0
        return df

    @pytest.mark.parametrize(
        'start, freq',
        [pytest.param('2020-01-01', 'MS', id='month_start'), pytest.param('2020-01-31', 'ME', id='month_end')],
    )
    def test_grid_follows_the_index_when_every_column_is_coarser(self, start, freq):
        """No column at the grid frequency: the grid is anchored like the index (ANO-FREQ-014).

        Both columns cover every month of 2020-2021, so the strict window is
        the whole frame and every aligned row is True. The grid used to fall
        back on 'E': with a month-start index, every aligned row was False.
        """
        df = self._coarse_columns_only(start, freq)
        calc = ImputationWindowCalculator().fit(df)

        assert (list(calc.get_imputation_window_mask().index), bool(calc.get_imputation_window_mask(df).all())) == (
            list(df.index), True
        )

    def test_panel_entity_grid_follows_its_own_index(self):
        """Per entity: a month-start entity with coarse columns only is anchored at month start."""
        coarse = self._coarse_columns_only('2020-01-01', 'MS')
        dense = _frame({'q': list(range(24)), 'y': list(range(24))})
        panel = pd.concat({'A': dense, 'B': coarse}, names=['entity', 'date'])
        calc = ImputationWindowCalculator().fit(panel)

        assert bool(calc.get_imputation_window_mask(panel).all())


class TestUndetectedFrequencyColumns:
    """D40: an observed column without detectable frequency leaves the grid and the denominator.

    Detection maps such a column to ``None`` (ANO-UTILS-042): observed once,
    or on unrecognized spacing. Its period being unknown, it can neither
    veto nor stretch the window (ANO-FREQ-015: ``fit`` used to crash).
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


# =============================================================================
# Robustesse d'index
# =============================================================================
class TestIndexRobustness:
    """Row order, entity order and an extra entity level leave the masks unchanged."""

    _KWARGS = dict(imputation_scope='extended_both', training_scope='unrestricted')

    @pytest.mark.parametrize(
        'perturbation',
        [pytest.param(shuffle_rows, id='shuffle_rows'), pytest.param(reverse_entities, id='reverse_entities')],
    )
    @pytest.mark.parametrize('kind', _KINDS)
    def test_panel_masks_are_unchanged(self, heterogeneous_coverage_panel, perturbation, kind):
        """Same masks (up to the order of the entity blocks) on the perturbed panel."""
        reference = _fit_quietly(ImputationWindowCalculator(**self._KWARGS), heterogeneous_coverage_panel)
        perturbed = _fit_quietly(
            ImputationWindowCalculator(**self._KWARGS), perturbation(heterogeneous_coverage_panel)
        )

        pd.testing.assert_series_equal(
            perturbed.get_imputation_window_mask(kind=kind).sort_index(),
            reference.get_imputation_window_mask(kind=kind).sort_index(),
        )

    @pytest.mark.parametrize(
        'perturbation',
        [pytest.param(shuffle_rows, id='shuffle_rows'), pytest.param(reverse_entities, id='reverse_entities')],
    )
    def test_panel_bounds_are_unchanged(self, heterogeneous_coverage_panel, perturbation):
        """Same per-entity bounds and absent columns on the perturbed panel."""
        reference = _fit_quietly(ImputationWindowCalculator(**self._KWARGS), heterogeneous_coverage_panel)
        perturbed = _fit_quietly(
            ImputationWindowCalculator(**self._KWARGS), perturbation(heterogeneous_coverage_panel)
        )

        assert (perturbed.imputation_window_start_, perturbed.imputation_window_end_,
                perturbed.structurally_absent_columns_) == (
            reference.imputation_window_start_, reference.imputation_window_end_,
            reference.structurally_absent_columns_,
        )

    @pytest.mark.parametrize('kind', _KINDS)
    def test_three_level_index_gives_the_same_masks(self, heterogeneous_coverage_panel, kind):
        """An outer ``region`` level is carried by the masks, values unchanged."""
        reference = _fit_quietly(ImputationWindowCalculator(**self._KWARGS), heterogeneous_coverage_panel)
        three_levels = _fit_quietly(
            ImputationWindowCalculator(**self._KWARGS), to_three_level_index(heterogeneous_coverage_panel)
        )

        mask = three_levels.get_imputation_window_mask(kind=kind)
        assert list(mask.index.names) == ['region', 'country', 'date']
        pd.testing.assert_series_equal(
            mask.droplevel(0).sort_index(), reference.get_imputation_window_mask(kind=kind).sort_index()
        )

    def test_three_level_index_keys_and_mask_at_frequency(self, heterogeneous_coverage_panel):
        """Bounds keyed by (region, country); per-entity targets accept those tuples."""
        calc = _fit_quietly(ImputationWindowCalculator(), to_three_level_index(heterogeneous_coverage_panel))
        targets = {('Zone euro', 'France'): 'QS', ('Zone euro', 'Allemagne'): 'YS',
                   ('Zone euro', 'Italie'): 'QS'}

        sums = calc.get_mask_at_frequency(targets, kind='strict').groupby(level=[0, 1]).sum()

        assert (set(calc.imputation_strict_window_start_), sums.to_dict()) == (
            set(targets),
            {('Zone euro', 'Allemagne'): 5, ('Zone euro', 'France'): 22, ('Zone euro', 'Italie'): 18},
        )

    @pytest.mark.parametrize('kind', _KINDS)
    def test_shuffled_time_series_gives_the_same_masks(self, irregular_index_timeseries, kind):
        """On a time series, the grid is rebuilt sorted whatever the row order."""
        reference = ImputationWindowCalculator(**self._KWARGS).fit(irregular_index_timeseries)
        shuffled = ImputationWindowCalculator(**self._KWARGS).fit(shuffle_rows(irregular_index_timeseries))

        pd.testing.assert_series_equal(
            shuffled.get_imputation_window_mask(kind=kind), reference.get_imputation_window_mask(kind=kind)
        )

    def test_aligned_mask_follows_shuffled_rows(self, heterogeneous_coverage_panel):
        """Aligning on shuffled rows gives the same row-wise values."""
        calc = _fit_quietly(ImputationWindowCalculator(**self._KWARGS), heterogeneous_coverage_panel)

        shuffled = calc.get_imputation_window_mask(shuffle_rows(heterogeneous_coverage_panel))

        pd.testing.assert_series_equal(
            shuffled.sort_index(), calc.get_imputation_window_mask(heterogeneous_coverage_panel).sort_index()
        )
