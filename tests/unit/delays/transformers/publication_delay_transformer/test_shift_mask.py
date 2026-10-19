"""Shift and mask resolution of ``PublicationDelayTransformer``.

Scenarios: gold shift values, calendar boundaries, mask parameters, strategy per
variable and masks with nothing to hide.
"""

# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest

# Classe à tester et producteurs réels de tableaux de délais
from tsforecast.delays.transformers import PublicationDelayTransformer

# Outils de test partagés du paquet
from tests.unit.delays.transformers.publication_delay_transformer.helpers import (
    PREDICTION,
    TS,
    _delays_frame,
    _fit,
    _monthly,
)


def _daily(columns=('GDP',), periods: int = 90) -> pd.DataFrame:
    """Build a daily frame from 2024-01-01 (January to March 30, 2024 for 90 days).

    Args:
        columns: Column names.
        periods: Number of days.

    Returns:
        Deterministic float frame indexed by days.
    """
    index = pd.date_range('2024-01-01', periods=periods, freq='D')
    return pd.DataFrame({col: np.arange(periods, dtype=float) for col in columns}, index=index)


# =============================================================================
# Valeurs d'or du décalage
# =============================================================================
class TestShiftGoldValues:
    """``n_periods`` is minus the number of months between December and the last month published on December 15."""

    @pytest.mark.parametrize(
        ('delay', 'reference_point', 'expected'),
        [
            # Valeur d'or : octobre est la dernière période publiée au 15 décembre (1er oct. + 45 j = 15 nov.,
            # 1er nov. + 45 j = 16 déc.) : -2
            pytest.param(45.0, 'start', -2, id='45d-start'),
            # Valeur d'or : septembre (fin le 1er oct. + 45 j = 15 nov. ; octobre : 1er nov. + 45 j = 16 déc.) : -3
            pytest.param(45.0, 'end', -3, id='45d-end'),
            # Valeur d'or : novembre (1er nov. + 20 j = 21 nov.) : -1
            pytest.param(20.0, 'start', -1, id='20d-start'),
            # Valeur d'or : octobre (fin le 1er nov. + 20 j = 21 nov. ; novembre : 1er déc. + 20 j = 21 déc.) : -2
            pytest.param(20.0, 'end', -2, id='20d-end'),
            # Valeur d'or : décembre publié le 1er décembre : 0
            pytest.param(0.0, 'start', 0, id='0d-start'),
            # Valeur d'or : décembre publié le 15 décembre, le jour même de la prédiction : 0
            pytest.param(14.0, 'start', 0, id='14d-start-published-on-the-day'),
            # Valeur d'or : décembre publié le 16, un jour trop tard : novembre, -1
            pytest.param(15.0, 'start', -1, id='15d-start-one-day-late'),
            # Valeur d'or : 1er nov. 2022 + 400 j = 5 déc. 2023 (publié) ; 1er déc. 2022 + 400 j = 4 janv. 2024 :
            # novembre 2022, treize mois avant décembre 2023 : -13
            pytest.param(400.0, 'start', -13, id='400d-start-longer-than-the-series'),
        ],
    )
    def test_monthly_series(self, delay, reference_point, expected):
        """Number of monthly periods to shift for a delay in days."""
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': delay}, delay_unit='D', reference_point=reference_point)
        assert transformer.shift_params['GDP']['n_periods'] == expected

    def test_quarterly_series(self):
        """On a quarterly series, the elapsed time counts from the quarter start.

        Gold value: the fourth quarter (October 1 + 45 days = November 15) is published on
        December 15: no shift.
        """
        quarterly = _monthly(('GDP',)).iloc[::3]
        transformer = _fit(quarterly, delays={'GDP': 45.0}, delay_unit='D', reference_point='start')
        assert transformer.shift_params['GDP'] == {'n_periods': 0, 'frequency': 'Q'}

    def test_frequency_is_the_detected_one(self):
        """The shift frequency of each column is its detected frequency."""
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 30.0}, delay_unit='D', reference_point='end')
        assert transformer.shift_params['GDP']['frequency'] == transformer.detected_frequencies_['GDP'] == 'M'

    def test_ordered_delays_give_ordered_shifts(self):
        """Increasing delays (10, 30, 60 days from the start) give shifts 0, -1, -2."""
        # Valeurs d'or : décembre publié le 11 déc. (0) ; novembre le 1er déc. (-1) ; octobre le 30 nov.,
        # novembre seulement le 31 déc. (-2)
        transformer = _fit(_monthly(('fast', 'medium', 'slow')), delays={'fast': 10.0, 'medium': 30.0, 'slow': 60.0},
                           delay_unit='D', reference_point='start')
        assert {col: p['n_periods'] for col, p in transformer.shift_params.items()} == {
            'fast': 0, 'medium': -1, 'slow': -2}


class TestCalendarBoundaries:
    """Near a calendar boundary, a value is visible at the prediction date if and only if it is published by then.

    ``n_periods`` counts calendar periods, not periods of 30 days: when
    the months between the delayed period and the prediction date did not last 30
    days each, the shift was one period off, in either direction.
    """

    def test_value_published_after_the_prediction_date_is_hidden(self):
        """February 2023 + 69 days = April 11: at April 10, the value of April must be January's, not February's.

        Gold value: 2023 is not a leap year, February + March = 59 days < 2 x 30;
        on the calendar, the last month published on April 10 is January
        (January 1 + 69 days = March 11), three months before April.
        """
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 69.0}, delay_unit='D', reference_point='start',
                           prediction_date='2023-04-10')
        shifted = transformer.transform(_monthly(('GDP',)))
        # Valeur d'or : valeur de janvier (0.0) ; février (1.0) n'est publié que le 11 avril
        assert shifted.loc[TS('2023-04-01'), 'GDP'] == 0.0

    def test_value_published_on_the_prediction_date_is_visible(self):
        """July 2023 + 76 days = September 15: at September 15, the value of September is July's.

        Gold value: July + August = 62 days > 2 x 30; on the calendar, July is
        published on the prediction date itself, two months before September.
        """
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 76.0}, delay_unit='D', reference_point='start',
                           prediction_date='2023-09-15')
        shifted = transformer.transform(_monthly(('GDP',)))
        # Valeur d'or : valeur de juillet (6.0), et non celle de juin (5.0)
        assert shifted.loc[TS('2023-09-01'), 'GDP'] == 6.0

    def test_business_days_skip_the_weekend(self):
        """Business-day series, 3 days from the start, prediction on Monday 2024-01-08: Friday's value lands on Monday.

        Gold value: Friday January 5 + 3 days = Monday January 8, published on the
        prediction date; Monday's own value is published on Thursday. One business
        day separates them (the weekend is not a period of a ``B`` index), hence -1.
        """
        X = pd.DataFrame({'GDP': np.arange(10.0)}, index=pd.bdate_range('2024-01-01', periods=10))
        transformer = _fit(X, delays={'GDP': 3.0}, delay_unit='D', reference_point='start',
                           prediction_date='2024-01-08')
        assert transformer.shift_params['GDP'] == {'n_periods': -1, 'frequency': 'B'}

    def test_negative_delay_on_business_days(self):
        """A delay of -1 day (forecast published the day before its date): Tuesday's value is visible on Monday.

        Gold value: Tuesday January 9 - 1 day = Monday January 8, published on the
        prediction date; Wednesday's value only on Tuesday: one business day ahead, +1.
        """
        X = pd.DataFrame({'GDP': np.arange(10.0)}, index=pd.bdate_range('2024-01-01', periods=10))
        transformer = _fit(X, delays={'GDP': -1.0}, delay_unit='D', reference_point='start',
                           prediction_date='2024-01-08')
        assert transformer.shift_params['GDP'] == {'n_periods': 1, 'frequency': 'B'}

    def test_end_of_a_quarter_counts_from_the_next_quarter(self):
        """Quarterly series, 45 days from the end, prediction on 2024-02-15: the fourth quarter of 2023 is published.

        Gold value: the fourth quarter ends on January 1, + 45 days = February 15
        (published on the prediction date), one quarter before the first quarter of
        2024, hence -1.
        """
        X = pd.DataFrame({'GDP': np.arange(8.0)}, index=pd.date_range('2022-01-01', periods=8, freq='QS'))
        transformer = _fit(X, delays={'GDP': 45.0}, delay_unit='D', reference_point='end',
                           prediction_date='2024-02-15')
        assert transformer.shift_params['GDP'] == {'n_periods': -1, 'frequency': 'Q'}


# =============================================================================
# Paramètres de masquage
# =============================================================================
class TestMaskParams:
    """Mask strategy: number of index observations hidden in each target period.

    ``n_obs`` is the number of index periods not yet published at the prediction
    date, counted on the calendar; masking is possible while
    ``n_obs`` is below the number of column periods in a target period, otherwise
    the column falls back to a shift.
    """

    @pytest.mark.parametrize(
        ('delay', 'reference_point', 'expected'),
        [
            # Valeur d'or : le 10 mars est publié le 15 ; 11 -> 15 mars non publiés -> 5
            pytest.param(5.0, 'start', 5, id='5d-start'),
            # Valeur d'or : depuis la fin du jour, le 9 mars est publié le 15 ; 10 -> 15 mars -> 6
            pytest.param(5.0, 'end', 6, id='5d-end'),
            # Valeur d'or : le 28 février (fin le 29) est publié le 15 mars ; 29 février -> 15 mars -> 16
            pytest.param(15.0, 'end', 16, id='15d-end'),
        ],
    )
    def test_daily_series_masked_by_month(self, delay, reference_point, expected):
        """On a daily series, one day is masked per day of delay not yet elapsed."""
        transformer = _fit(_daily(), delays={'GDP': delay}, strategy='mask', delay_unit='D',
                           reference_point=reference_point, target_frequency='M', prediction_date='2024-03-15')
        assert transformer.mask_params == {'GDP': {'n_obs': expected, 'mask_frequency': 'M', 'how': 'last'}}

    @pytest.mark.parametrize(
        'target_frequency',
        [pytest.param('Q', id='str'), pytest.param({'GDP': 'Q'}, id='dict'), pytest.param('quarterly', id='literal')],
    )
    def test_target_frequency_forms(self, target_frequency):
        """Scalar, per-variable and literal target frequencies give the normalized code.

        Gold value: monthly series, 20 days from the start: December (published December 21)
        is not published, November is -> 1 month masked per quarter.
        """
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 20.0}, strategy='mask', delay_unit='D',
                           reference_point='start', target_frequency=target_frequency)
        assert transformer.mask_params == {'GDP': {'n_obs': 1, 'mask_frequency': 'Q', 'how': 'last'}}

    def test_target_frequency_inferred_from_the_table(self):
        """The ``frequency`` column of the delays table is the target frequency."""
        transformer = _fit(_monthly(('GDP',)), delays=_delays_frame(columns=('GDP',), delays=(20.0,)), strategy='mask')
        assert transformer.mask_params['GDP']['mask_frequency'] == 'Q'

    @pytest.mark.parametrize(
        ('delay', 'masked'),
        [
            # Valeur d'or : 27 jours, le 16 février est publié le 15 mars ; 17 février -> 15 mars non publiés :
            # 12 + 15 = 27 jours < 28 jours de février 2023 -> masquage
            pytest.param(27.0, True, id='27d-masked'),
            # Valeur d'or : 28 jours, 16 février -> 15 mars non publiés : 13 + 15 = 28 jours, tout février
            # serait masqué -> repli sur le décalage (un mois de 30 jours le laissait masquer)
            pytest.param(28.0, False, id='28d-whole-february'),
        ],
    )
    def test_whole_february_cannot_be_masked(self, delay, masked):
        """Daily series masked by month: the check counts the 28 days of February 2023, not 30 days."""
        X = pd.DataFrame({'GDP': np.arange(90.0)}, index=pd.date_range('2023-01-01', periods=90, freq='D'))
        transformer = _fit(X, delays={'GDP': delay}, strategy='mask', delay_unit='D', reference_point='start',
                           target_frequency='M', prediction_date='2023-03-15')
        assert ('GDP' in transformer.mask_params, 'GDP' in transformer.shift_params) == (masked, not masked)

    def test_period_index_is_masked_like_its_month_starts(self):
        """A monthly ``PeriodIndex`` gets the mask of its month starts: 20 days, December unpublished, 1 month."""
        X = _monthly(('GDP',))
        X.index = X.index.to_period('M')
        transformer = _fit(X, delays={'GDP': 20.0}, strategy='mask', delay_unit='D', reference_point='start',
                           target_frequency='Q')
        assert transformer.mask_params == {'GDP': {'n_obs': 1, 'mask_frequency': 'Q', 'how': 'last'}}

    @pytest.mark.parametrize(
        ('delay', 'masked'),
        [
            # Valeur d'or : 3 jours non publiés au 20 février (18 -> 20) : moins qu'une quinzaine -> masquage
            pytest.param(3.0, True, id='3d-masked'),
            # Valeur d'or : 1er -> 20 février non publiés, 20 jours, plus qu'aucune quinzaine -> décalage
            pytest.param(20.0, False, id='20d-whole-semi-month'),
        ],
    )
    def test_semi_monthly_target(self, delay, masked):
        """Semi-monthly target periods (no pandas ``Period``): the check uses a semi-month of 15 days."""
        X = pd.DataFrame({'GDP': np.arange(60.0)}, index=pd.date_range('2024-01-01', periods=60, freq='D'))
        transformer = _fit(X, delays={'GDP': delay}, strategy='mask', delay_unit='D', reference_point='start',
                           target_frequency='SM', prediction_date='2024-02-20')
        assert ('GDP' in transformer.mask_params, 'GDP' in transformer.shift_params) == (masked, not masked)

    def test_last_maskable_number_of_months(self):
        """Up to two months of a quarter can be masked: 74 days, October published on December 14 -> 2."""
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 74.0}, strategy='mask', delay_unit='D',
                           reference_point='start', target_frequency='Q')
        assert transformer.mask_params['GDP']['n_obs'] == 2

    def test_fallback_to_shift_when_the_whole_period_would_be_masked(self):
        """80 days -> 3 unpublished months, a whole quarter: the column is shifted, with a warning.

        Gold value: on December 15, October (published December 20), November and
        December are unpublished; September is published on November 20.
        """
        transformer = PublicationDelayTransformer(delays={'GDP': 80.0}, strategy='mask', delay_unit='D',
                                                  reference_point='start', target_frequency='Q',
                                                  prediction_date=PREDICTION)
        with pytest.warns(UserWarning, match="Could not mask the column 'GDP'"):
            transformer.fit(_monthly(('GDP',)))
        assert (set(transformer.shift_params), transformer.mask_params) == ({'GDP'}, {})

    def test_fallback_shift_moves_values_to_later_dates(self):
        """The fallback shift is the one of the 'shift' strategy: 80 days from the start -> -3.

        The fallback stores a negative ``n_periods``: a positive value would move the
        values three months **earlier**, before their publication.
        """
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 80.0}, strategy='mask', delay_unit='D',
                           reference_point='start', target_frequency='Q')
        assert transformer.shift_params['GDP']['n_periods'] == -3


# =============================================================================
# Stratégie
# =============================================================================
class TestStrategy:
    """Choice of the strategy per variable."""

    def test_mask_strategy_masks_every_delayed_column(self):
        """With 'mask', every column with a delay is masked, none shifted."""
        transformer = _fit(_monthly(), delays={'GDP': 20.0, 'CPI': 50.0}, strategy='mask', delay_unit='D',
                           reference_point='start', target_frequency='Q')
        # Valeurs d'or : 20 j, novembre publié le 21 nov. -> 1 ; 50 j, octobre publié le 20 nov.,
        # novembre seulement le 21 déc. -> 2
        assert ({c: p['n_obs'] for c, p in transformer.mask_params.items()}, transformer.shift_params) == (
            {'GDP': 1, 'CPI': 2}, {})

    def test_strategy_per_variable(self):
        """A per-variable strategy shifts some columns and masks the others."""
        transformer = _fit(_monthly(), delays={'GDP': 45.0, 'CPI': 20.0}, strategy={'GDP': 'shift', 'CPI': 'mask'},
                           delay_unit='D', reference_point='start', target_frequency={'CPI': 'Q'})
        assert (set(transformer.shift_params), set(transformer.mask_params)) == ({'GDP'}, {'CPI'})

    def test_strategy_per_variable_gold_values(self):
        """Shift of ``GDP`` (45 days -> -2) and mask of ``CPI`` (20 days -> 1 month per quarter)."""
        transformer = _fit(_monthly(), delays={'GDP': 45.0, 'CPI': 20.0}, strategy={'GDP': 'shift', 'CPI': 'mask'},
                           delay_unit='D', reference_point='start', target_frequency={'CPI': 'Q'})
        assert (transformer.shift_params, transformer.mask_params) == (
            {'GDP': {'n_periods': -2, 'frequency': 'M'}},
            {'CPI': {'n_obs': 1, 'mask_frequency': 'Q', 'how': 'last'}})

    def test_delayed_column_without_strategy_is_left_unchanged(self):
        """A column with a delay but absent from the strategy dict is announced and left unchanged."""
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0, 'CPI': 20.0}, strategy={'GDP': 'shift'},
                                                  delay_unit='D', reference_point='start', prediction_date=PREDICTION)
        with pytest.warns(UserWarning, match=r"\['CPI'\] have a delay but no strategy"):
            transformer.fit(_monthly())
        assert (set(transformer.shift_params), transformer.fit_report_.columns_unaffected) == ({'GDP'}, ('CPI',))

    def test_strategy_per_variable_ignores_default_values(self):
        """As announced by ``__init__``, ``default_values`` is ignored with a strategy dict: nothing is imputed.

        ``_build_parameter_dict`` must not impute the default reference point to the
        columns *outside* the strategy dict (``Z`` here), which are never delayed.
        """
        defaults = {'delay': 30.0, 'unit': 'D', 'reference_point': 'end', 'target_frequency': 'Q'}
        transformer = _fit(_monthly(('GDP', 'CPI', 'Z')), delays=_delays_frame(),
                           strategy={'GDP': 'shift', 'CPI': 'shift'}, default_values=defaults)
        assert transformer.fit_report_.defaults_imputed == ()

    def test_masked_variable_needs_a_target_frequency(self):
        """A masked variable without target frequency (defaults ignored with a dict) is rejected, named."""
        defaults = {'delay': 30.0, 'unit': 'D', 'reference_point': 'start', 'target_frequency': 'Q'}
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0, 'CPI': 20.0},
                                                  strategy={'GDP': 'shift', 'CPI': 'mask'}, delay_unit='D',
                                                  reference_point='start', default_values=defaults,
                                                  prediction_date=PREDICTION)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match=r"No 'target_frequency' for the delayed columns \['CPI'\]"):
                transformer.fit(_monthly())


# =============================================================================
# Masques nuls ou négatifs (contrat de MaskTransformer : n_obs >= 0)
# =============================================================================
class TestZeroAndNegativeMasks:
    """A mask with nothing to mask leaves its column unchanged instead of building a ``MaskTransformer``.

    Gold values (monthly column, delays at monthly frequency, so the mask frequency
    equals the index frequency): ``n_obs`` is the number of months not yet published
    on 2023-12-15, the delay counted from the period start; a negative count (later
    months already published) is clamped at zero.
    """

    @staticmethod
    def _monthly() -> pd.DataFrame:
        index = pd.date_range('2023-01-01', periods=12, freq='MS')
        return pd.DataFrame({'GDP': np.arange(12.0), 'CPI': np.arange(12.0) * 2}, index=index)

    @staticmethod
    def _transformer(delay: float) -> PublicationDelayTransformer:
        delays = pd.DataFrame({'column': ['GDP'], 'delay': [delay], 'unit': ['D'],
                               'reference_point': ['start'], 'frequency': ['M']})
        return PublicationDelayTransformer(delays=delays, strategy='mask', prediction_date='2023-12-15')

    @pytest.mark.parametrize(
        'delay',
        [
            # Valeur d'or : décembre publié le 6 décembre -> 0
            pytest.param(5.0, id='zero'),
            # Valeur d'or : janvier publié dès le 22 novembre (1er janv. - 40 j) -> -1, ramené à 0
            pytest.param(-40.0, id='negative'),
        ],
    )
    def test_n_obs_is_never_negative(self, delay):
        """The number of observations to mask is clamped at zero."""
        transformer = self._transformer(delay).fit(self._monthly())
        assert transformer.mask_params['GDP']['n_obs'] == 0

    @pytest.mark.parametrize('delay', [pytest.param(5.0, id='zero'), pytest.param(-40.0, id='negative')])
    def test_zero_mask_leaves_the_data_unchanged(self, delay):
        """``transform`` returns the data as given: nothing is unpublished at the prediction date."""
        transformer = self._transformer(delay).fit(self._monthly())
        pd.testing.assert_frame_equal(transformer.transform(self._monthly()), self._monthly())

    def test_zero_mask_round_trip(self):
        """``inverse_transform`` of an unchanged frame returns it unchanged."""
        transformer = self._transformer(5.0).fit(self._monthly())
        recovered = transformer.inverse_transform(transformer.transform(self._monthly()))
        pd.testing.assert_frame_equal(recovered, self._monthly())
