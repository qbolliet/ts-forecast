"""Grid anchoring and index robustness of ``ImputationWindowCalculator``.

Scenarios: masks aligned on the data grid, irregular annual anchors, row and entity
order, extra entity level.
"""

import numpy as np
import pandas as pd
import pytest

# Objet testé
from tsforecast.frequency.imputation_window import ImputationWindowCalculator

# Perturbations partagées
from tests.support.perturbations import (
    reverse_entities,
    shuffle_rows,
    to_three_level_index,
)

# Outils de test partagés du paquet
from tests.unit.frequency.imputation_window.helpers import _KINDS, _fit_quietly, _frame


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
        """No column at the grid frequency: the grid is anchored like the index.

        Both columns cover every month of 2020-2021, so the strict window is
        the whole frame and every aligned row is True.
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
