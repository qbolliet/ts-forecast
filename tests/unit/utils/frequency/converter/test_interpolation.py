"""Unit tests of ``FrequencyConverter.interpolate_to_higher_frequency``.

Scope: calendar golden values (monthly to daily, leap years), the chaining
Y → Q → M, ``limit`` (integer, ``None``, ``'default'``, invalid values),
``limit_direction`` defaults (start, end and position-less targets),
``limit_area``, interior NaN, the interpolation methods (``time``, ``index``,
``nearest``, unsupported), the ``asfreq`` fallback when the source frequency
cannot be detected, empty and unobserved inputs, and the target grid of the
upsampling path, fixed after U8 (ANO-UTILS-052 to 068): non-integer frequency
ratio (M → W), semi-monthly frequencies, sub-daily frequencies, variables
carried on a finer row grid, unsorted input, index name, multiplied
position-less source frequencies ('2D').

``anchor_fraction`` has its own module (``test_anchor_fraction.py``); the
start / end re-anchoring is covered by ``test_positions.py``.
"""
import numpy as np
import pandas as pd
import pytest

from tsforecast.utils.frequency.converter import FrequencyConverter


@pytest.fixture
def converter() -> FrequencyConverter:
    """A fresh ``FrequencyConverter``."""
    return FrequencyConverter()


@pytest.fixture
def quarterly() -> pd.Series:
    """Quarter ends of 2021: 10, 20, 30, 40."""
    return pd.Series(
        [10.0, 20.0, 30.0, 40.0], index=pd.date_range('2021-03-31', periods=4, freq='QE')
    )


# Valeurs d'or de `quarterly` → 'ME' (interpolation positionnelle, pas d'un tiers
# entre deux fins de trimestre ; janvier et février comblés vers l'arrière depuis mars)
_QUARTERLY_TO_MONTHLY = [10.0, 10.0, 10.0, 40 / 3, 50 / 3, 20.0,
                         70 / 3, 80 / 3, 30.0, 100 / 3, 110 / 3, 40.0]


# =============================================================================
# Valeurs d'or calendaires et chaînage
# =============================================================================

class TestCalendarInterpolation:
    """Monthly to daily interpolation follows the actual length of each month."""

    @pytest.mark.parametrize('year', [2023, 2024], ids=['common-year', 'leap-year'])
    def test_month_ends_to_days(self, converter, year):
        """Cumulative day counts interpolate to the day of the year."""
        # Valeur de chaque fin de mois = son quantième (31, 59 ou 60, 90 ou 91) :
        # l'interpolation linéaire sur la grille journalière redonne le quantième de
        # chaque jour ; du 1er au 30 janvier, comblement arrière depuis le 31 (= 31)
        month_ends = pd.date_range(f'{year}-01-31', periods=3, freq='ME')
        series = pd.Series(month_ends.dayofyear.astype(float), index=month_ends)

        result = converter.interpolate_to_higher_frequency(series, 'D', method='linear')

        days = pd.date_range(f'{year}-01-01', f'{year}-03-31', freq='D')
        expected = pd.Series(np.maximum(days.dayofyear, 31).astype(float), index=days)
        pd.testing.assert_series_equal(result, expected, check_freq=False)


class TestInterpolationChain:
    """Y → Q then Q → M gives the same linear path as Y → M directly."""

    def test_chained_equals_direct(self, converter):
        """Golden monthly values of the chained interpolation."""
        yearly = pd.Series([120.0, 132.0], index=pd.date_range('2021-12-31', periods=2, freq='YE'))

        quarterly = converter.interpolate_to_higher_frequency(yearly, 'QE', method='linear')
        monthly = converter.interpolate_to_higher_frequency(quarterly, 'ME', method='linear')

        # Valeurs d'or : 2021 constant à 120 (comblement arrière depuis le 31/12/2021),
        # 2022 de 121 à 132 par pas de 1 — points alignés : le chaînage ne déforme rien
        assert monthly.tolist() == pytest.approx([120.0] * 12 + list(np.arange(121.0, 133.0)))


# =============================================================================
# limit, limit_direction, limit_area
# =============================================================================

class TestInterpolationLimit:
    """``limit`` bounds the number of consecutive NaN filled."""

    def test_no_limit_fills_every_gap(self, converter, quarterly):
        """``limit=None`` (method default) fills every month."""
        result = converter.interpolate_to_higher_frequency(quarterly, 'ME', method='linear', limit=None)
        assert result.tolist() == pytest.approx(_QUARTERLY_TO_MONTHLY)

    def test_integer_limit(self, converter, quarterly):
        """``limit=1`` fills one month backwards from each quarter end."""
        result = converter.interpolate_to_higher_frequency(quarterly, 'ME', method='linear', limit=1)
        # Valeurs d'or : dans chaque trou de deux mois, seul le mois voisin de la
        # fin de trimestre suivante est comblé (direction 'backward' par défaut)
        expected = [np.nan, 10.0, 10.0, np.nan, 50 / 3, 20.0,
                    np.nan, 80 / 3, 30.0, np.nan, 110 / 3, 40.0]
        assert result.tolist() == pytest.approx(expected, nan_ok=True)

    def test_default_limit_is_the_conversion_factor(self, converter, quarterly):
        """``'default'`` means 3 months per quarter: a missing quarter leaves two holes."""
        with_gap = quarterly.copy()
        with_gap.iloc[1] = np.nan

        result = converter.interpolate_to_higher_frequency(with_gap, 'ME', method='linear', limit='default')

        # Valeurs d'or : trou d'avril à août (5 NaN) ; limite 3 vers l'arrière depuis
        # septembre → juin, juillet, août comblés sur la droite 10 (mars) → 30 (sept.) ;
        # avril et mai restent NaN
        expected = [10.0, 10.0, 10.0, np.nan, np.nan, 20.0,
                    70 / 3, 80 / 3, 30.0, 100 / 3, 110 / 3, 40.0]
        assert result.tolist() == pytest.approx(expected, nan_ok=True)

    def test_default_limit_without_source_frequency_is_unbounded(self, converter):
        """No detectable source frequency: ``'default'`` falls back on no limit."""
        irregular = pd.Series(
            [1.0, 5.0], index=pd.DatetimeIndex(['2024-01-01', '2024-01-11'])
        )
        result = converter.interpolate_to_higher_frequency(irregular, 'D', method='linear', limit='default')
        # Valeur d'or : 11 jours, pas de 0.4 entre 1 et 5, aucun NaN
        assert result.tolist() == pytest.approx(list(np.linspace(1.0, 5.0, 11)))

    def test_numpy_integer_limit_is_honoured(self, converter, quarterly):
        """``np.int64(1)`` behaves like ``1``."""
        result = converter.interpolate_to_higher_frequency(
            quarterly, 'ME', method='linear', limit=np.int64(1)
        )
        # Valeur d'or : janvier non comblé, comme avec limit=1
        assert np.isnan(result.iloc[0])

    @pytest.mark.parametrize(
        'limit',
        [pytest.param('foo', id='string'), pytest.param(1.0, id='float'), pytest.param(True, id='boolean')],
    )
    def test_invalid_limit_raises(self, converter, quarterly, limit):
        """A limit that is neither an integer, ``'default'`` nor ``None`` is rejected."""
        with pytest.raises(ValueError, match='Invalid limit'):
            converter.interpolate_to_higher_frequency(quarterly, 'ME', method='linear', limit=limit)


class TestLimitDirectionDefaults:
    """Without ``limit_direction``, the target position decides the filling direction."""

    def test_end_target_fills_backwards(self, converter, quarterly):
        """'ME' target: months before the first quarter end are filled."""
        result = converter.interpolate_to_higher_frequency(quarterly, 'ME', method='linear')
        assert result.iloc[:2].tolist() == [10.0, 10.0]

    def test_start_target_fills_forwards(self, converter, quarterly):
        """'MS' target: the first two months stay missing, the rest is interpolated."""
        result = converter.interpolate_to_higher_frequency(quarterly, 'MS', method='linear')
        # Valeurs d'or : fins de trimestre ré-ancrées au 1er de leur mois (mars, juin,
        # sept., déc.) ; 'forward' ne comble pas janvier ni février
        expected = [np.nan, np.nan] + _QUARTERLY_TO_MONTHLY[2:]
        assert result.tolist() == pytest.approx(expected, nan_ok=True)

    def test_positionless_target_fills_backwards(self, converter):
        """'D' has no position: the end convention applies, trailing days stay missing."""
        quarter_starts = pd.Series(
            [10.0, 20.0], index=pd.date_range('2024-01-01', periods=2, freq='QS')
        )
        result = converter.interpolate_to_higher_frequency(quarter_starts, 'D', method='linear')

        # Valeur d'or : grille du 1er janvier au 30 juin (fin du 2e trimestre) ; au-delà
        # du 1er avril, 'backward' ne comble rien (90 jours NaN : 29 + 31 + 30)
        assert result.loc['2024-04-02':].isna().sum() == 90

    @pytest.mark.parametrize(
        'direction, target, expected_head',
        [
            pytest.param('forward', 'ME', [np.nan, np.nan, 10.0], id='forward-on-end-target'),
            pytest.param('both', 'MS', [10.0, 10.0, 10.0], id='both-on-start-target'),
        ],
    )
    def test_explicit_direction_wins(self, converter, quarterly, direction, target, expected_head):
        """An explicit direction overrides the positional default."""
        result = converter.interpolate_to_higher_frequency(
            quarterly, target, method='linear', limit_direction=direction
        )
        assert result.iloc[:3].tolist() == pytest.approx(expected_head, nan_ok=True)


class TestLimitArea:
    """``limit_area`` restricts the filling to inner or outer gaps."""

    def test_inside_leaves_the_edges_missing(self, converter, quarterly):
        """'inside' never extrapolates before the first quarter end."""
        result = converter.interpolate_to_higher_frequency(quarterly, 'ME', method='linear', limit_area='inside')
        expected = [np.nan, np.nan] + _QUARTERLY_TO_MONTHLY[2:]
        assert result.tolist() == pytest.approx(expected, nan_ok=True)

    def test_outside_fills_the_edges_only(self, converter, quarterly):
        """'outside' fills January and February, never the months between quarter ends."""
        result = converter.interpolate_to_higher_frequency(quarterly, 'ME', method='linear', limit_area='outside')
        expected = [10.0, 10.0, 10.0, np.nan, np.nan, 20.0,
                    np.nan, np.nan, 30.0, np.nan, np.nan, 40.0]
        assert result.tolist() == pytest.approx(expected, nan_ok=True)


# =============================================================================
# Méthodes d'interpolation
# =============================================================================

class TestInterpolationMethods:
    """Methods other than 'linear' use the index values, not the positions."""

    @pytest.mark.parametrize('method', ['time', 'index'])
    def test_time_weighted_methods(self, converter, quarterly, method):
        """'time' and 'index' weight by the actual number of days between month ends."""
        result = converter.interpolate_to_higher_frequency(quarterly, 'ME', method=method)
        # Valeurs d'or : trimestres de 91, 92 et 92 jours en 2021 ; ex. 30 avril à
        # 30 jours du 31 mars → 10 + 10 × 30 / 91
        expected = [10.0, 10.0, 10.0,
                    10 + 10 * 30 / 91, 10 + 10 * 61 / 91, 20.0,
                    20 + 10 * 31 / 92, 20 + 10 * 62 / 92, 30.0,
                    30 + 10 * 31 / 92, 30 + 10 * 61 / 92, 40.0]
        assert result.tolist() == pytest.approx(expected)

    def test_nearest(self, converter, quarterly):
        """'nearest' takes the closest quarter end, in days."""
        result = converter.interpolate_to_higher_frequency(quarterly, 'ME', method='nearest')
        # Valeurs d'or : 30 avril (30 j après mars, 61 j avant juin) → 10 ; 31 mai → 20 ;
        # 31 juil. (31 / 61) → 20 ; 31 août (62 / 30) → 30 ; 31 oct. → 30 ; 30 nov. → 40
        interior = result.loc[['2021-04-30', '2021-05-31', '2021-07-31',
                               '2021-08-31', '2021-10-31', '2021-11-30']]
        assert interior.tolist() == [10.0, 20.0, 20.0, 30.0, 30.0, 40.0]

    def test_unsupported_method_raises(self, converter, quarterly):
        """An aggregation method is not an interpolation method."""
        with pytest.raises(ValueError, match='mean'):
            converter.interpolate_to_higher_frequency(quarterly, 'ME', method='mean')

    def test_unsupported_method_message_names_interpolation(self, converter, quarterly):
        """The error message speaks of an interpolation method."""
        with pytest.raises(ValueError, match='interpolation method'):
            converter.interpolate_to_higher_frequency(quarterly, 'ME', method='mean')

    def test_invalid_target_frequency_raises(self, converter, quarterly):
        """An anchor unknown to pandas is reported with the target frequency."""
        with pytest.raises(ValueError, match="Invalid target frequency 'MS-FOO'"):
            converter.interpolate_to_higher_frequency(quarterly, 'MS-FOO')


# =============================================================================
# Repli sans fréquence source
# =============================================================================

class TestFallbackWithoutSourceFrequency:
    """Undetectable source frequency: ``asfreq`` between the first and last dates."""

    def test_irregular_index_is_interpolated_between_its_dates(self, converter):
        """The daily grid spans the observations only; values follow the calendar."""
        observed = pd.DatetimeIndex(['2024-01-01', '2024-01-03', '2024-01-10', '2024-02-20'])
        irregular = pd.Series([1.0, 2.0, 3.0, 4.0], index=observed)

        result = converter.interpolate_to_higher_frequency(irregular, 'D', method='linear')

        # Valeurs d'or : interpolation linéaire en jours écoulés (np.interp), grille
        # du 1er janvier au 20 février, sans extension de période
        days = pd.date_range('2024-01-01', '2024-02-20', freq='D')
        offsets = (days - days[0]).days
        expected = np.interp(offsets, (observed - days[0]).days, irregular.to_numpy())
        pd.testing.assert_series_equal(result, pd.Series(expected, index=days), check_freq=False)

    def test_single_observation_is_returned_alone(self, converter):
        """One point, no frequency: nothing to extend."""
        single = pd.Series([7.0], index=pd.DatetimeIndex(['2024-01-31']))
        result = converter.interpolate_to_higher_frequency(single, 'D')
        assert result.tolist() == [7.0]

    def test_supplied_source_frequency_extends_a_single_observation(self, converter):
        """With ``source_freq='ME'``, a single month end covers its whole month."""
        single = pd.Series([7.0], index=pd.DatetimeIndex(['2024-01-31']))
        result = converter.interpolate_to_higher_frequency(single, 'D', source_freq='ME')
        # Valeur d'or : 31 jours de janvier 2024, tous comblés vers l'arrière depuis le 31
        pd.testing.assert_series_equal(
            result, pd.Series(7.0, index=pd.date_range('2024-01-01', '2024-01-31')), check_freq=False
        )

    def test_duplicated_dates_raise(self, converter):
        """Two observations in the same target period are refused."""
        duplicated = pd.Series(
            [1.0, 2.0, 3.0], index=pd.DatetimeIndex(['2024-03-31', '2024-03-31', '2024-06-30'])
        )
        with pytest.raises(ValueError, match="Several observations fall in the same 'ME' period"):
            converter.interpolate_to_higher_frequency(duplicated, 'ME')

    def test_data_finer_than_the_target_raises(self, converter):
        """Monthly observations cannot be 'interpolated' to quarters."""
        monthly = pd.Series(
            np.arange(1.0, 7.0), index=pd.date_range('2024-01-01', periods=6, freq='MS')
        )
        with pytest.raises(ValueError, match='not at a lower frequency than the target'):
            converter.interpolate_to_higher_frequency(monthly, 'QS', source_freq='YS')


class TestEmptyOrUnobservedInput:
    """No row, or no observed value: nothing to interpolate."""

    def test_empty_series_raises(self, converter):
        """An empty Series is refused."""
        empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        with pytest.raises(ValueError, match='Cannot interpolate empty data'):
            converter.interpolate_to_higher_frequency(empty, 'D')

    @pytest.mark.parametrize('as_frame', [False, True], ids=['series', 'dataframe'])
    def test_unobserved_input_gives_an_empty_result(self, converter, quarterly, as_frame):
        """Rows without any observation give no date, for a Series as for a DataFrame."""
        unobserved = quarterly * np.nan
        data = unobserved.to_frame('q') if as_frame else unobserved
        result = converter.interpolate_to_higher_frequency(data, 'ME')
        assert result.empty and type(result) is type(data)


# =============================================================================
# Grille cible du sur-échantillonnage (ANO-UTILS-052 à 058, 068 corrigées)
# =============================================================================

class TestUpsamplingTargetGrid:
    """The output is the target grid over the whole source periods, from the observations."""

    def test_monthly_to_weekly_gives_a_weekly_grid(self, converter):
        """The output of a monthly to weekly interpolation is weekly."""
        monthly = pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2024-01-31', periods=3, freq='ME'))
        result = converter.interpolate_to_higher_frequency(monthly, 'W', method='linear')
        assert pd.infer_freq(result.index) == 'W-SUN'

    def test_monthly_to_weekly_golden_values(self, converter):
        """Each month end is re-stamped on the Sunday closing its week, then interpolated."""
        monthly = pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2024-01-31', periods=3, freq='ME'))
        result = converter.interpolate_to_higher_frequency(monthly, 'W', method='linear')
        # Valeurs d'or : dimanches du 7 janvier au 31 mars 2024 ; 31/01 → 04/02 (1),
        # 29/02 → 03/03 (2), 31/03 (3) ; 4 semaines entre deux ancres, pas de 0.25 ;
        # janvier comblé vers l'arrière (cible sans position → 'backward')
        expected = pd.Series(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
            index=pd.date_range('2024-01-07', '2024-03-31', freq='W'),
        )
        pd.testing.assert_series_equal(result, expected, check_freq=False)

    def test_monthly_to_semi_monthly(self, converter):
        """Each month end falls in the second half of its month (15th, 'SMS')."""
        monthly = pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2024-01-31', periods=3, freq='ME'))
        result = converter.interpolate_to_higher_frequency(monthly, 'SMS', method='linear')
        # Valeurs d'or : grille 1er / 15 de janvier à mars ; ancres aux 15 ; cible 'S'
        # → 'forward', le 1er janvier reste NaN
        expected = pd.Series(
            [np.nan, 1.0, 1.5, 2.0, 2.5, 3.0],
            index=pd.date_range('2024-01-01', '2024-03-15', freq='SMS'),
        )
        pd.testing.assert_series_equal(result, expected, check_freq=False)

    def test_semi_monthly_to_daily_gives_a_daily_grid(self, converter):
        """A semi-monthly series interpolated to days is daily."""
        semi_monthly = pd.Series(
            [1.0, 2.0, 3.0, 4.0], index=pd.date_range('2024-01-15', periods=4, freq='SME')
        )
        result = converter.interpolate_to_higher_frequency(semi_monthly, 'D', method='linear')
        assert pd.infer_freq(result.index) == 'D'

    def test_semi_monthly_to_daily_covers_the_half_months(self, converter):
        """'SME' stamps close their half-month: the days run from January 1st to February 29th."""
        semi_monthly = pd.Series(
            [1.0, 2.0, 3.0, 4.0], index=pd.date_range('2024-01-15', periods=4, freq='SME')
        )
        result = converter.interpolate_to_higher_frequency(semi_monthly, 'D', method='linear')
        # Valeurs d'or : 1er-15 janvier comblés vers l'arrière (1) ; 23 janvier à mi-chemin
        # des ancres du 15 (1) et du 31 (2) → 1.5
        assert result.loc[['2024-01-01', '2024-01-23', '2024-02-29']].tolist() == [1.0, 1.5, 4.0]
        assert result.index[[0, -1]].tolist() == [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-02-29')]

    def test_semi_monthly_start_to_daily(self, converter):
        """'SMS' stamps open their half-month: the days run from January 1st to February 29th."""
        semi_monthly = pd.Series(
            [1.0, 2.0, 3.0, 4.0], index=pd.date_range('2024-01-01', periods=4, freq='SMS')
        )
        result = converter.interpolate_to_higher_frequency(semi_monthly, 'D', method='linear')
        # Valeurs d'or : 8 janvier à mi-chemin des ancres du 1er (1) et du 15 (2) → 1.5 ;
        # 16-29 février, après la dernière ancre, non comblés ('backward' par défaut)
        assert result.index[[0, -1]].tolist() == [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-02-29')]
        assert result.loc['2024-01-08'] == 1.5 and result.loc['2024-02-16':].isna().all()

    @pytest.mark.parametrize(
        'source_step, target, expected_head',
        [
            pytest.param('6h', 'h', [0.0, 1.0, 2.0, 3.0], id='six-hours-to-hours'),
            pytest.param('h', 'min', [0.0, 1.0, 2.0, 3.0], id='hours-to-minutes'),
        ],
    )
    def test_sub_daily_target_keeps_the_time_of_day(self, converter, source_step, target, expected_head):
        """Sub-daily stamps are not brought back to midnight (ANO-UTILS-068)."""
        # Valeur = nombre d'unités cibles écoulées depuis minuit : l'interpolation
        # linéaire redonne ce décompte à chaque pas de la grille cible
        steps = pd.date_range('2024-01-01', periods=4, freq=source_step)
        units = pd.Timedelta(1, unit=target)
        series = pd.Series(((steps - steps[0]) / units).astype(float), index=steps)

        result = converter.interpolate_to_higher_frequency(series, target, method='linear')

        assert result.iloc[:4].tolist() == expected_head

    def test_six_hours_block_starts_at_its_stamp(self, converter):
        """A '6h' stamp opens its block: the hourly grid runs from 00:00 to 23:00."""
        steps = pd.date_range('2024-01-01', periods=4, freq='6h')
        series = pd.Series([0.0, 6.0, 12.0, 18.0], index=steps)
        result = converter.interpolate_to_higher_frequency(series, 'h', method='linear')
        # Valeurs d'or : 0 … 18 aux heures observées et entre elles ; 19 h - 23 h au-delà
        # de la dernière observation, non comblées ('backward' par défaut)
        expected = list(np.arange(19.0)) + [np.nan] * 5
        assert result.tolist() == pytest.approx(expected, nan_ok=True)

    def test_variable_on_a_finer_row_grid(self, converter):
        """An annual variable carried on a monthly grid interpolates to quarters."""
        # Variable annuelle (100 en 2020, 140 en 2021) portée par une grille mensuelle
        monthly_grid = pd.date_range('2020-01-01', '2021-12-01', freq='MS')
        annual = pd.Series(np.nan, index=monthly_grid)
        annual.loc['2020-01-01'] = 100.0
        annual.loc['2021-01-01'] = 140.0

        result = converter.interpolate_to_higher_frequency(annual, 'QS', method='linear', source_freq='YS')

        # Valeurs d'or : pas de 10 par trimestre entre les deux débuts d'année, puis
        # comblement vers l'avant (cible 'S') jusqu'à la fin de 2021
        expected = pd.Series(
            [100.0, 110.0, 120.0, 130.0, 140.0, 140.0, 140.0, 140.0],
            index=pd.date_range('2020-01-01', periods=8, freq='QS'),
        )
        pd.testing.assert_series_equal(result, expected, check_freq=False)

    def test_unsorted_input_gives_the_sorted_result(self, converter, quarterly):
        """A descending index gives the same twelve months as the sorted one."""
        result = converter.interpolate_to_higher_frequency(quarterly.iloc[::-1], 'ME', method='linear')
        assert result.tolist() == pytest.approx(_QUARTERLY_TO_MONTHLY)

    def test_index_name_is_kept(self, converter, quarterly):
        """The name of the source index survives the interpolation."""
        named = quarterly.rename_axis('date')
        result = converter.interpolate_to_higher_frequency(named, 'ME', method='linear')
        assert result.index.name == 'date'


class TestMultipliedPositionlessSource:
    """A multiplied tick frequency ('2D') stamps the start of its block, like pandas.

    Triage of the inherited failure
    ``test_positions.py::TestExtendIndexForUpsampling::test_unsupported_frequency_pair_returns_original``
    ('2D' → 'D', supported since ``469d37f``): the author ruled on 2026-09-30 that a
    position-less multiplied tick is read as the start of its block, as
    ``pandas.resample`` labels it (ANO-UTILS-064, fixed).
    """

    @pytest.fixture
    def every_other_day(self) -> pd.Series:
        """Ten values every other day from 2024-01-01 (last stamp 2024-01-19)."""
        return pd.Series(np.arange(10.0), index=pd.date_range('2024-01-01', periods=10, freq='2D'))

    def test_daily_grid_covers_the_blocks(self, converter, every_other_day):
        """The daily grid runs from the first stamp to the end of the last block."""
        result = converter.convert_frequency(every_other_day, 'D', method='linear')
        # Valeur d'or : blocs [01-01, 01-02], …, [01-19, 01-20] → 20 jours
        pd.testing.assert_index_equal(
            result.index, pd.date_range('2024-01-01', '2024-01-20', freq='D'), check_names=False
        )

    def test_observed_values_are_kept(self, converter, every_other_day):
        """Whatever the block reading, each stamp keeps its value and days in between are halfway."""
        result = converter.convert_frequency(every_other_day, 'D', method='linear')
        # Valeurs d'or : 0 au 1er janvier, 0.5 au 2, 9 au 19
        assert result.loc[['2024-01-01', '2024-01-02', '2024-01-19']].tolist() == [0.0, 0.5, 9.0]
