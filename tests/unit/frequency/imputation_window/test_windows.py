"""Strict, imputation and training windows.

Scenarios: strict window definition, backward and forward extensions, relations between
the three masks, and the case without any strict window.
"""

# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest

# Objet testé
from tsforecast.frequency.imputation_window import ImputationWindowCalculator

# Outils de test partagés du paquet
from tests.unit.frequency.imputation_window.helpers import (
    _BOUND_ATTRIBUTES,
    _MASK_ATTRIBUTES,
    _fit_quietly,
    _frame,
    _make_df_with_shoulders,
    _warning_messages,
)


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
    """Relations between the strict, imputation and training masks."""

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

    def test_panel_without_window_behaves_like_a_time_series(self):
        """On a panel, no strict window for any entity gives all-False masks, as on a time series.

        This case does not raise "No imputation window found for any entity in the
        panel": the imputer does not deduce all-True masks from it, which would
        impute the whole panel.
        """
        _, df = _make_df_without_strict_window()
        panel = pd.concat({'A': df, 'B': df}, names=['entity', 'date'])
        calc = _fit_quietly(ImputationWindowCalculator(imputation_scope='extended_both', min_columns=2), panel)

        assert all(
            isinstance(getattr(calc, attr), pd.Series) and not getattr(calc, attr).any()
            for attr in _MASK_ATTRIBUTES
        )

    def test_panel_without_window_is_trainable_under_unrestricted(self):
        """Under 'unrestricted', the panel stays trainable and warns once that nothing is imputable."""
        _, df = _make_df_without_strict_window()
        panel = pd.concat({'A': df, 'B': df}, names=['entity', 'date'])
        calc = ImputationWindowCalculator(training_scope='unrestricted', min_columns=2)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            calc.fit(panel)

        named = [m for m in _warning_messages(records) if 'no value can be imputed' in m]
        assert len(named) == 1
        assert (bool(calc.training_window_mask_.loc['A'].all()),
                bool(calc.imputation_window_mask_.loc['A'].any())) == (True, False)

    def test_time_series_message_says_nothing_is_imputed(self):
        """One message, saying the imputation window is empty and what still trains."""
        _, df = _make_df_without_strict_window()

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            ImputationWindowCalculator(min_columns=2).fit(df)

        named = [m for m in _warning_messages(records) if 'no period' in m]
        assert named == [
            "There is no period in the DataFrame where all the variables are available: "
            "the imputation window is empty, so no value can be imputed, and only "
            "training_scope='unrestricted' allows training in this case."
        ]

    def test_entities_without_strict_window_are_named_by_one_warning(self):
        """Two entities without strict window next to one with: ONE message naming both.

        The message is emitted once, names the entities and does not talk about
        the whole "DataFrame".
        """
        _, without = _make_df_without_strict_window()
        with_window = without.fillna(1.0)
        panel = pd.concat({'A': with_window, 'B': without, 'C': without}, names=['entity', 'date'])

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            calc = ImputationWindowCalculator(min_columns=2).fit(panel)

        named = [m for m in _warning_messages(records) if 'no period' in m]
        assert len(named) == 1
        assert "for 2 entity(ies) [('B',), ('C',)]" in named[0]
        assert bool(calc.imputation_window_mask_.loc['A'].all())
