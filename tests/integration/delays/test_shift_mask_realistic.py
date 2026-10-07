"""Realistic scenarios for ``ShiftTransformer`` and ``MaskTransformer``.

Runs both transformers on the realistic datasets of notebook 3:
``irregular_index_timeseries`` (annual dates 2015-2017 before a monthly grid
starting in 2018, columns of several frequencies stored on that grid) and, through
``PanelwiseTransformer``, ``heterogeneous_coverage_panel`` (per-entity coverage:
Allemagne ends in April 2024, France and Italie in July 2024, with per-entity
gaps). No gold value per cell: the tests check properties derived from the
calendar (round trip, masked positions, last masked date per entity, dates moved
by exactly ``k`` periods).

The genuinely irregular index exposed two anomalies, now fixed (``tests/ANOMALIES.md``):
``ANO-DELAYS-021`` (shift by position on an index with gaps) and ``ANO-DELAYS-017``
(positions counted on the available dates inside the data, so that a lone January
observation was masked as the last month of its quarter).
"""
# Modules de base
import numpy as np
import pandas as pd
import pytest

# Classes à tester et collaborateur réel pour les panels
from tsforecast.delays.transformers import MaskTransformer, ShiftTransformer
from tsforecast.panel import PanelwiseTransformer

TS = pd.Timestamp

# Date de début de la grille mensuelle régulière de irregular_index_timeseries
MONTHLY_GRID_START = TS('2018-01-01')


def _panelwise(transformer) -> PanelwiseTransformer:
    """Wrap a transformer for a panel whose entity and time are index levels."""
    return PanelwiseTransformer(transformer=transformer, time_col=None, panel_cols=None)


# =============================================================================
# Série à index réellement irrégulier
# =============================================================================
class TestIrregularIndexTimeseries:
    """``irregular_index_timeseries``: annual dates followed by a monthly grid."""

    @pytest.mark.parametrize('k', [pytest.param(-2, id='k=-2'), pytest.param(3, id='k=3')])
    def test_shift_round_trip(self, irregular_index_timeseries, k):
        """Shift then inverse restores the dataset exactly, annual dates included."""
        shifter = ShiftTransformer(n_periods=k, frequency='M')
        recovered = shifter.inverse_transform(shifter.fit_transform(irregular_index_timeseries))
        pd.testing.assert_frame_equal(recovered, irregular_index_timeseries, check_freq=False, check_names=False)

    def test_shift_keeps_every_observation(self, irregular_index_timeseries):
        """No observation is lost nor created: non-null counts per column are unchanged."""
        shifted = ShiftTransformer(n_periods=-2, frequency='M').fit_transform(irregular_index_timeseries)
        pd.testing.assert_series_equal(shifted.notna().sum(), irregular_index_timeseries.notna().sum())

    def test_shift_moves_every_date_by_one_month(self, irregular_index_timeseries):
        """Every date, annual ones included, moves exactly one month later for ``n_periods=-1``."""
        shifted = ShiftTransformer(n_periods=-1, frequency='M').fit_transform(irregular_index_timeseries)
        # Valeur d'or : arithmétique d'offsets pandas (2017-01-01 -> 2017-02-01, pas 2018-01-01)
        expected = irregular_index_timeseries.index + pd.offsets.MonthBegin(1)
        pd.testing.assert_index_equal(shifted.index, expected, check_names=False)

    def test_mask_round_trip(self, irregular_index_timeseries):
        """Mask then inverse restores the dataset exactly."""
        masker = MaskTransformer(n_obs=1, mask_frequency='Q', how='last')
        recovered = masker.inverse_transform(masker.fit_transform(irregular_index_timeseries))
        pd.testing.assert_frame_equal(recovered, irregular_index_timeseries, check_freq=False)

    def test_mask_leaves_unmasked_cells_unchanged(self, irregular_index_timeseries):
        """Cells not masked keep their value (NaN included)."""
        masked = MaskTransformer(n_obs=1, mask_frequency='Q', how='last').fit_transform(irregular_index_timeseries)
        kept = ~masked.isna().all(axis=1)
        pd.testing.assert_frame_equal(masked[kept], irregular_index_timeseries[kept], check_freq=False)

    def test_mask_on_the_monthly_grid_hits_quarter_ends_only(self, irregular_index_timeseries):
        """On the regular part of the index, only the last month of each quarter is masked."""
        masked = MaskTransformer(n_obs=1, mask_frequency='Q', how='last').fit_transform(irregular_index_timeseries)
        rows = masked.index[masked.isna().all(axis=1) & irregular_index_timeseries.notna().any(axis=1)]
        assert set(rows[rows >= MONTHLY_GRID_START].month) == {3, 6, 9, 12}

    def test_lone_annual_dates_are_not_masked(self, irregular_index_timeseries):
        """Each January of 2015-2017, alone in its quarter, is not on the last position: kept (ANO-DELAYS-017)."""
        masked = MaskTransformer(n_obs=1, mask_frequency='Q', how='last').fit_transform(irregular_index_timeseries)
        rows = masked.index[masked.isna().all(axis=1) & irregular_index_timeseries.notna().any(axis=1)]
        assert rows[rows < MONTHLY_GRID_START].empty


# =============================================================================
# Panel à couverture hétérogène (via PanelwiseTransformer)
# =============================================================================
class TestHeterogeneousCoveragePanel:
    """``heterogeneous_coverage_panel``: per-entity coverage and gaps."""

    @pytest.mark.parametrize('k', [pytest.param(-2, id='k=-2'), pytest.param(3, id='k=3')])
    def test_shift_round_trip_per_entity(self, heterogeneous_coverage_panel, k):
        """Shift then inverse restores every entity, in the input order."""
        transformer = _panelwise(ShiftTransformer(n_periods=k, frequency='M'))
        recovered = transformer.inverse_transform(transformer.fit_transform(heterogeneous_coverage_panel))
        pd.testing.assert_frame_equal(recovered, heterogeneous_coverage_panel)

    def test_shift_keeps_the_coverage_of_each_entity(self, heterogeneous_coverage_panel):
        """Each entity keeps its number of rows and of observations per column."""
        shifted = _panelwise(ShiftTransformer(n_periods=-2, frequency='M')).fit_transform(heterogeneous_coverage_panel)
        counts = shifted.notna().groupby(level=0).sum()
        pd.testing.assert_frame_equal(counts, heterogeneous_coverage_panel.notna().groupby(level=0).sum())

    def test_mask_round_trip_per_entity(self, heterogeneous_coverage_panel):
        """Mask then inverse restores every entity, in the input order."""
        transformer = _panelwise(MaskTransformer(n_obs=1, mask_frequency='Q', how='last'))
        recovered = transformer.inverse_transform(transformer.fit_transform(heterogeneous_coverage_panel))
        pd.testing.assert_frame_equal(recovered, heterogeneous_coverage_panel)

    def test_last_masked_date_follows_the_coverage_of_each_entity(self, heterogeneous_coverage_panel):
        """The last masked month is the last complete quarter end of each entity.

        Allemagne ends in April 2024 and France / Italie in July 2024: in both cases
        the last position of the current quarter (June, September) is absent, so the
        last masked month is the previous quarter end.
        """
        masked = _panelwise(MaskTransformer(n_obs=1, mask_frequency='Q', how='last')).fit_transform(
            heterogeneous_coverage_panel)
        new_nan = masked.isna() & heterogeneous_coverage_panel.reindex(masked.index).notna()
        rows = masked.index[new_nan.any(axis=1)]
        last = {entity: rows[rows.get_level_values(0) == entity].get_level_values(-1).max()
                for entity in ('Allemagne', 'France', 'Italie')}
        # Valeur d'or : fin du trimestre complet précédent la fin de couverture
        assert last == {'Allemagne': TS('2024-03-01'), 'France': TS('2024-06-01'), 'Italie': TS('2024-06-01')}
