"""Anchored and multiplied frequencies in ``PublicationDelayTransformer``.

Scenarios: weeks anchored on another day than Sunday, mask over fiscal quarters,
multiplied column frequencies aligned on the data grid (single series and panel),
delays expressed at a multiplied frequency, and the count of index positions in
anchored target periods.
"""

# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest

# Classe et fonction auxiliaire à tester
from tsforecast.delays.transformers import PublicationDelayTransformer, _min_subperiods

TS = pd.Timestamp


def _fit(X: pd.DataFrame, **kwargs) -> PublicationDelayTransformer:
    """Fit a transformer on ``X``, silencing the panel and missing-delay warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return PublicationDelayTransformer(**kwargs).fit(X)


def _bimonthly_on_a_monthly_grid(first_month: str = '2023-02-01', periods: int = 12) -> pd.DataFrame:
    """Build a monthly-indexed frame whose ``x`` column is observed every other month from ``first_month``."""
    index = pd.date_range('2023-01-01', periods=periods, freq='MS')
    values = pd.Series(np.nan, index=index)
    observed = pd.date_range(first_month, index[-1], freq='2MS')
    values[observed] = np.arange(len(observed), dtype=float)
    return values.to_frame('x')


# =============================================================================
# Ancrages des fréquences
# =============================================================================
class TestAnchoredFrequencies:
    """The anchor detected on the dates sets the periods counted by the shift and the mask."""

    def test_week_ending_on_wednesday_counts_its_own_weeks(self):
        """Weeks from Thursday to Wednesday, published 3 days after their end, seen on Saturday 2023-11-25.

        Valeurs d'or : la semaine du 16 au 22 novembre finit le jeudi 23, publiée le dimanche 26,
        pas encore visible ; celle du 9 au 15 est publiée le 19 : dernière publiée, deux semaines
        avant celle de la date de prédiction (23 au 29 novembre).
        """
        index = pd.date_range('2023-10-04', '2023-11-29', freq='W-WED')
        X = pd.DataFrame({'x': np.arange(len(index), dtype=float)}, index=index)

        transformer = _fit(X, delays={'x': 3}, delay_unit='day', reference_point='end',
                           prediction_date='2023-11-25')

        assert transformer.detected_frequencies_ == {'x': 'W-WED'}
        assert transformer.shift_params == {'x': {'n_periods': -2, 'frequency': 'W-WED'}}
        # La valeur de la semaine finissant le 15 novembre est placée dans celle de la prédiction
        assert transformer.transform(X).loc[TS('2023-11-29'), 'x'] == X.loc[TS('2023-11-15'), 'x']

    def test_mask_over_fiscal_quarters(self):
        """Quarters from September to November, monthly index, prediction on 2023-11-20, 10 days after month end.

        Valeurs d'or : novembre (dernier mois de son trimestre) n'est publié que le 11 décembre,
        octobre l'est le 11 novembre : un mois masqué à la fin de chaque trimestre fiscal
        (février, mai, août, novembre).
        """
        index = pd.date_range('2023-01-01', periods=12, freq='MS')
        X = pd.DataFrame({'x': np.arange(12, dtype=float)}, index=index)

        transformer = _fit(X, delays={'x': 10}, delay_unit='day', reference_point='end', strategy='mask',
                           target_frequency='QE-NOV', prediction_date='2023-11-20')

        assert transformer.mask_params == {
            'x': {'n_obs': 1, 'mask_frequency': 'QE-NOV', 'how': 'last', 'last_position': None}}
        masked = transformer.transform(X)
        assert masked.index[masked['x'].isna()].strftime('%m').tolist() == ['02', '05', '08', '11']


# =============================================================================
# Fréquences multipliées
# =============================================================================
class TestMultipliedColumnFrequency:
    """A multiplied column frequency is counted on the grid of the data, not on the epoch."""

    def test_bimonthly_column_counts_its_own_periods(self):
        """Periods February-March, April-May, ..., published 25 days after their end, seen on 2023-12-20.

        Valeurs d'or : octobre-novembre finit le 1er décembre, publié le 26 décembre, pas encore
        visible ; août-septembre l'est le 26 octobre : deux périodes de deux mois avant celle de la
        date de prédiction (décembre-janvier), soit quatre mois d'index. Une grille alignée sur
        l'epoch (novembre-décembre, septembre-octobre) n'en compterait qu'une.
        """
        X = _bimonthly_on_a_monthly_grid()

        transformer = _fit(X, delays={'x': 25}, delay_unit='day', reference_point='end',
                           prediction_date='2023-12-20')

        assert transformer.detected_frequencies_ == {'x': '2MS'}
        assert transformer.grid_origins_ == {'x': TS('2023-02-01')}
        assert transformer.shift_params == {'x': {'n_periods': -2, 'frequency': '2MS'}}
        # La valeur d'août est placée quatre mois plus tard, en décembre
        assert transformer.transform(X).loc[TS('2023-12-01'), 'x'] == X.loc[TS('2023-08-01'), 'x']

    def test_period_index_gives_the_same_grid(self):
        """Monthly periods instead of month starts: same detected frequency, origin and shift."""
        X = _bimonthly_on_a_monthly_grid()
        periods = X.copy()
        periods.index = periods.index.to_period('M')

        transformer = _fit(periods, delays={'x': 25}, delay_unit='day', reference_point='end',
                           prediction_date='2023-12-20')

        assert transformer.grid_origins_ == {'x': TS('2023-02-01')}
        assert transformer.shift_params == {'x': {'n_periods': -2, 'frequency': '2MS'}}

    def test_single_period_columns_have_no_grid_origin(self):
        """Only a multiplied frequency needs the origin of its grid."""
        index = pd.date_range('2023-01-01', periods=12, freq='MS')
        X = pd.DataFrame({'x': np.arange(12, dtype=float)}, index=index)

        transformer = _fit(X, delays={'x': 10}, delay_unit='day', reference_point='end', prediction_date='2023-12-20')

        assert transformer.grid_origins_ == {'x': None}

    def test_panel_entities_on_the_same_grid_share_its_origin(self):
        """Two entities observed in February, April, ...: one grid, starting in February."""
        panel = pd.concat({'A': _bimonthly_on_a_monthly_grid(), 'B': _bimonthly_on_a_monthly_grid('2023-04-01')},
                          names=['entity', 'date'])

        transformer = _fit(panel, delays={'x': 25}, delay_unit='day', reference_point='end',
                           prediction_date='2023-12-20')

        assert transformer.grid_origins_ == {'x': TS('2023-02-01')}

    def test_panel_entities_out_of_phase_are_rejected(self):
        """Entities observed in February, April... and in January, March... do not share their periods."""
        panel = pd.concat({'A': _bimonthly_on_a_monthly_grid(), 'B': _bimonthly_on_a_monthly_grid('2023-01-01')},
                          names=['entity', 'date'])

        with pytest.raises(ValueError, match="grids out of phase"):
            _fit(panel, delays={'x': 25}, delay_unit='day', reference_point='end', prediction_date='2023-12-20')

    def test_delay_at_a_multiplied_frequency_other_than_the_column_is_rejected(self):
        """A delay cannot be read at '2MS' for a monthly column: a multiplied period exists only on the data grid."""
        index = pd.date_range('2023-01-01', periods=12, freq='MS')
        X = pd.DataFrame({'x': np.arange(12, dtype=float)}, index=index)

        with pytest.raises(ValueError, match="multiplied frequency other than the frequency"):
            _fit(X, delays={'x': 10}, delay_unit='day', reference_point='start', delay_frequency='2MS',
                 prediction_date='2023-12-20')

    def test_delay_at_the_multiplied_frequency_of_the_column_is_accepted(self):
        """A delay read at the frequency of a '2MS' column counts from the start of each period of the grid."""
        X = _bimonthly_on_a_monthly_grid()

        # Valeurs d'or : délai de 80 j depuis le début, soit 19 j après la fin d'août-septembre (1er octobre,
        # 61 j) : même décompte que 19 j depuis la fin
        from_start = _fit(X, delays={'x': 80}, delay_unit='day', reference_point='start', delay_frequency='2MS',
                          prediction_date='2023-12-20')
        from_end = _fit(X, delays={'x': 19}, delay_unit='day', reference_point='end', prediction_date='2023-12-20')

        assert from_start.shift_params == from_end.shift_params


# =============================================================================
# Décompte des positions de l'index
# =============================================================================
class TestMinSubperiods:
    """Smallest number of index positions in the target periods covered by the dates."""

    @pytest.mark.parametrize(
        ('dates', 'target', 'offset', 'expected'),
        [
            # Trimestres de décembre à février : 90 jours pendant l'hiver 2022-2023
            (pd.date_range('2023-01-01', '2023-06-30', freq='D'), 'QE-NOV', pd.offsets.Day(), 90.0),
            # Trimestres calendaires couverts de janvier à juin 2023 : 90 jours au premier
            (pd.date_range('2023-01-01', '2023-06-30', freq='D'), 'Q', pd.offsets.Day(), 90.0),
            # Mercredis de chaque mois de mars à mai 2023 : 5, 4 et 5
            (pd.date_range('2023-03-01', '2023-05-31', freq='W-WED'), 'M', pd.offsets.Week(weekday=2), 4.0),
            # Mois d'un trimestre fiscal, grille de fin de mois
            (pd.date_range('2023-01-31', periods=6, freq='ME'), 'QS-MAR', pd.offsets.MonthEnd(), 3.0),
        ],
        ids=['fiscal-quarter-days', 'calendar-quarter-days', 'wednesdays-per-month', 'months-per-fiscal-quarter'],
    )
    def test_count_on_the_index_grid(self, dates, target, offset, expected):
        """The positions are the points of the index grid in each target period, as the mask counts them."""
        assert _min_subperiods(dates, target, offset) == expected

    def test_period_without_date_is_not_counted(self):
        """A target period with no date between two covered ones does not lower the minimum."""
        dates = pd.DatetimeIndex(['2023-01-15', '2023-03-15'])

        # Valeurs d'or : janvier (31 j) et mars (31 j) couverts, février (28 j) non
        assert _min_subperiods(dates, 'M', pd.offsets.Day()) == 31.0

    def test_no_date_gives_an_infinite_count(self):
        """Without any date, every mask is possible."""
        assert _min_subperiods(pd.DatetimeIndex([]), 'M', pd.offsets.Day()) == float('inf')
