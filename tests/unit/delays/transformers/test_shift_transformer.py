"""Unit tests for ``ShiftTransformer`` (``tsforecast.delays.transformers``).

Covers the public API of the class: ``__init__`` (``n_periods``, ``frequency``),
the sklearn protocol (``get_params`` / ``set_params``, ``clone``,
``check_is_fitted``), ``fit``, ``transform`` and ``inverse_transform``. The
private helpers ``_convert_shift_periods_to_index_periods``,
``_shift_by_periods``, ``_extend_index_start`` and ``_extend_index_end`` are
exercised through ``transform`` / ``inverse_transform`` only.

Sign convention (code, confirmed by ``PublicationDelayTransformer`` which passes
a **negative** ``n_periods`` so that the value of ``t`` appears at ``t + delay``):
a positive ``n_periods = k`` moves every value ``k`` periods **earlier**, a
negative one later. The shift is calendar arithmetic on each date, on the grid
of the index frequency detected at ``fit``. Gold values are computed with pandas
offsets, independently of the code: ``shifted.index == original.index - k *
offset`` and the values keep their order, including on indexes with gaps.

Key property: shifting by ``k`` then inverting restores the input exactly, and
in the window common to the input and the shifted output no observation is
lost (``shifted[t] == original[t + k]``).

Per-entity parameters go through ``PanelwiseTransformer`` (the class itself
rejects a ``MultiIndex``); per-variable parameters through one instance per
group of columns, as ``PublicationDelayTransformer`` does. The realistic
scenario (notebook 3 datasets) lives in
``tests/integration/delays/test_shift_mask_realistic.py``.

Anomalies found while writing these tests are registered in
``tests/ANOMALIES.md`` (``ANO-DELAYS-021`` to ``-026``, all fixed).
"""
# Modules de base
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

# Classe à tester et collaborateur réel pour les panels
from tsforecast.delays.transformers import ShiftTransformer
from tsforecast.panel import PanelwiseTransformer

# Perturbations partagées
from tests.support.perturbations import (
    shuffle_rows,
    to_period_index,
    with_duplicated_rows,
    with_index_names,
    with_special_column_names,
)

TS = pd.Timestamp


# =============================================================================
# Constructeurs locaux de petits jeux à valeurs d'or calculables
# =============================================================================
def _series(freq: str = 'MS', periods: int = 6, start: str = '2024-01-01', name='x') -> pd.Series:
    """Build a regular float series ``0, 1, ..., periods - 1``.

    Args:
        freq: Pandas frequency of the index.
        periods: Number of observations.
        start: First date (rolled forward to the frequency anchor by pandas).
        name: Name of the series.

    Returns:
        The series, its values equal to their position.
    """
    index = pd.date_range(start, periods=periods, freq=freq)
    return pd.Series(np.arange(periods, dtype=float), index=index, name=name)


def _expected_index(original: pd.Series, k: int, freq: str) -> pd.DatetimeIndex:
    """Return the index expected after a shift of ``k`` periods, from pandas offsets only.

    Args:
        original: Series before the shift (its index lies on the ``freq`` anchors).
        k: Shift, positive = values moved earlier.
        freq: Pandas frequency of the index.

    Returns:
        ``original.index - k * offset``.
    """
    # Valeur d'or : arithmétique d'offsets pandas, indépendante du code testé
    return original.index - k * pd.tseries.frequencies.to_offset(freq)


# Fréquences d'index couvertes : positions début / fin, ancres, infra-journalier
INDEX_FREQUENCIES = [
    pytest.param('MS', 'M', id='MS'),
    pytest.param('ME', 'M', id='ME'),
    pytest.param('QS', 'Q', id='QS'),
    pytest.param('QE-DEC', 'Q', id='QE-DEC'),
    pytest.param('YS', 'Y', id='YS'),
    pytest.param('YE', 'Y', id='YE'),
    pytest.param('D', 'D', id='D'),
    pytest.param('W-WED', 'W', id='W-WED'),
    pytest.param('h', 'h', id='h'),
]

# Décalages couverts : nuls, positifs, négatifs, plus longs que la série (6 observations)
SHIFTS = [
    pytest.param(0, id='k=0'),
    pytest.param(1, id='k=1'),
    pytest.param(3, id='k=3'),
    pytest.param(-1, id='k=-1'),
    pytest.param(-4, id='k=-4'),
    pytest.param(13, id='k=13>len'),
    pytest.param(-13, id='k=-13>len'),
]


# =============================================================================
# Protocole sklearn
# =============================================================================
class TestSklearnProtocol:
    """``ShiftTransformer`` follows the sklearn estimator conventions."""

    def test_get_params_returns_the_constructor_arguments(self):
        """``get_params`` exposes exactly ``n_periods`` and ``frequency``."""
        assert ShiftTransformer(n_periods=-2, frequency='Q').get_params() == {'n_periods': -2, 'frequency': 'Q'}

    def test_set_params_changes_the_shift(self):
        """``set_params`` (used by ``PanelwiseTransformer`` per entity) changes the applied shift."""
        # Valeur d'or : n_periods passé de 1 à 2 -> première date 2023-11-01
        shifter = ShiftTransformer(n_periods=1, frequency='M').set_params(n_periods=2)
        assert shifter.fit_transform(_series()).index[0] == TS('2023-11-01')

    def test_clone_keeps_the_parameters(self):
        """``clone`` rebuilds an estimator with the same parameters."""
        assert clone(ShiftTransformer(n_periods=3, frequency='M')).get_params() == {'n_periods': 3, 'frequency': 'M'}

    def test_clone_drops_the_fitted_attributes(self):
        """A clone of a fitted shifter is unfitted."""
        fitted = ShiftTransformer(n_periods=1, frequency='M').fit(_series())
        assert not hasattr(clone(fitted), 'index_frequency_')

    def test_unfitted_shifter_is_reported_unfitted(self):
        """``check_is_fitted`` raises ``NotFittedError`` before ``fit``."""
        with pytest.raises(NotFittedError):
            check_is_fitted(ShiftTransformer(n_periods=1, frequency='M'))

    def test_fit_returns_self_and_is_fitted(self):
        """``fit`` returns the estimator itself, which ``check_is_fitted`` then accepts."""
        shifter = ShiftTransformer(n_periods=1, frequency='M')
        assert shifter.fit(_series()) is shifter
        check_is_fitted(shifter)

    @pytest.mark.parametrize(
        'freq, expected',
        [
            pytest.param('MS', ('M', 'S', None), id='MS'),
            pytest.param('ME', ('M', 'E', None), id='ME'),
            pytest.param('QE-DEC', ('Q', 'E', 'DEC'), id='QE-DEC'),
            pytest.param('D', ('D', None, None), id='D'),
        ],
    )
    def test_fit_stores_the_index_components(self, freq, expected):
        """``fit`` stores the base, position and anchor of the index frequency."""
        shifter = ShiftTransformer(n_periods=1, frequency=expected[0]).fit(_series(freq))
        assert (shifter.index_frequency_, shifter.index_position_, shifter.index_suffix_) == expected

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_unfitted_shifter_cannot_transform(self, method):
        """``transform`` and ``inverse_transform`` raise ``NotFittedError`` before ``fit``."""
        with pytest.raises(NotFittedError):
            getattr(ShiftTransformer(n_periods=1, frequency='M'), method)(_series())

    def test_transform_uses_the_grid_detected_at_fit(self):
        """A single observation, whose frequency cannot be detected, is shifted on the fitted grid."""
        shifter = ShiftTransformer(n_periods=1, frequency='Q').fit(_series('MS'))
        # Valeur d'or : un trimestre = trois périodes de la grille mensuelle ajustée
        assert shifter.transform(_series('MS', periods=1)).index.tolist() == [TS('2023-10-01')]

    def test_fit_stores_the_shift_in_index_periods(self):
        """``index_periods_`` is the shift converted into index periods."""
        # Valeur d'or : deux trimestres = six mois
        assert ShiftTransformer(n_periods=2, frequency='Q').fit(_series('MS')).index_periods_ == 6


# =============================================================================
# Valeurs d'or : sens du décalage, positions, ancres
# =============================================================================
class TestShiftGoldValues:
    """Each value moves exactly ``n_periods`` index periods, earlier for a positive shift."""

    def test_positive_shift_moves_values_earlier(self):
        """``n_periods=2`` on ``MS``: 2024-01-01 becomes 2023-11-01, values in the same order."""
        series = pd.Series([10., 20., 30., 40., 50.], index=pd.date_range('2024-01-01', periods=5, freq='MS'))
        shifted = ShiftTransformer(n_periods=2, frequency='M').fit_transform(series)
        # Valeur d'or : chaque date recule de deux mois, les valeurs ne bougent pas dans le tableau
        expected = pd.Series([10., 20., 30., 40., 50.], index=pd.date_range('2023-11-01', periods=5, freq='MS'))
        pd.testing.assert_series_equal(shifted, expected, check_freq=False)

    def test_negative_shift_moves_values_later(self):
        """``n_periods=-2`` on ``MS``: the last date 2024-05-01 becomes 2024-07-01."""
        series = pd.Series([10., 20., 30., 40., 50.], index=pd.date_range('2024-01-01', periods=5, freq='MS'))
        shifted = ShiftTransformer(n_periods=-2, frequency='M').fit_transform(series)
        expected = pd.Series([10., 20., 30., 40., 50.], index=pd.date_range('2024-03-01', periods=5, freq='MS'))
        pd.testing.assert_series_equal(shifted, expected, check_freq=False)

    @pytest.mark.parametrize('freq, shift_freq', INDEX_FREQUENCIES)
    @pytest.mark.parametrize('k', SHIFTS)
    def test_index_moves_by_k_periods(self, freq, shift_freq, k):
        """The shifted index is ``original.index - k * offset`` for every frequency and sign."""
        original = _series(freq)
        shifted = ShiftTransformer(n_periods=k, frequency=shift_freq).fit_transform(original)
        pd.testing.assert_index_equal(shifted.index, _expected_index(original, k, freq), exact=False, check_names=False)

    @pytest.mark.parametrize('freq, shift_freq', INDEX_FREQUENCIES)
    @pytest.mark.parametrize('k', SHIFTS)
    def test_values_keep_their_order(self, freq, shift_freq, k):
        """Only the dates change: the values array is the input one, in the same order."""
        original = _series(freq)
        shifted = ShiftTransformer(n_periods=k, frequency=shift_freq).fit_transform(original)
        np.testing.assert_array_equal(shifted.to_numpy(), original.to_numpy())

    @pytest.mark.parametrize(
        'freq, check',
        [
            pytest.param('MS', lambda d: d.is_month_start, id='month-start'),
            pytest.param('ME', lambda d: d.is_month_end, id='month-end'),
            pytest.param('QS', lambda d: d.is_quarter_start, id='quarter-start'),
            pytest.param('QE', lambda d: d.is_quarter_end, id='quarter-end'),
            pytest.param('YS', lambda d: d.is_year_start, id='year-start'),
            pytest.param('YE', lambda d: d.is_year_end, id='year-end'),
        ],
    )
    def test_position_of_the_index_is_preserved(self, freq, check):
        """Dates added by the extension keep the start / end position of the index."""
        shifted = ShiftTransformer(n_periods=4, frequency=freq[0]).fit_transform(_series(freq))
        assert all(check(d) for d in shifted.index)

    def test_weekly_anchor_is_preserved(self):
        """A ``W-WED`` index stays on Wednesdays after a negative shift."""
        shifted = ShiftTransformer(n_periods=-2, frequency='W').fit_transform(_series('W-WED'))
        # Valeur d'or : 2024-01-03 est un mercredi, la série couvre six mercredis, +2 semaines
        assert shifted.index[-1] == TS('2024-02-21')


class TestZeroAndLongShifts:
    """Zero shifts are copies; shifts longer than the series still lose nothing."""

    def test_zero_shift_returns_an_equal_series(self):
        """``n_periods=0`` returns the input unchanged."""
        original = _series()
        pd.testing.assert_series_equal(ShiftTransformer(n_periods=0, frequency='M').fit_transform(original), original)

    def test_zero_shift_returns_a_copy(self):
        """The output of a zero shift does not share its data with the input."""
        original = _series()
        shifted = ShiftTransformer(n_periods=0, frequency='M').fit_transform(original)
        shifted.iloc[0] = 99.0
        assert original.iloc[0] == 0.0

    def test_zero_shift_still_checks_the_granularity(self):
        """A zero shift by a frequency finer than the index is rejected like any other."""
        with pytest.raises(ValueError, match="cannot be more granular"):
            ShiftTransformer(n_periods=0, frequency='D').fit(_series())

    def test_shift_longer_than_the_series_leaves_no_common_date(self):
        """A shift of 10 periods on 6 observations moves the whole index before the input."""
        shifted = ShiftTransformer(n_periods=10, frequency='M').fit_transform(_series())
        # Valeur d'or : dernière date 2024-06-01 - 10 mois = 2023-08-01 < 2024-01-01
        assert shifted.index.intersection(_series().index).empty

    def test_shift_longer_than_the_series_keeps_every_value(self):
        """No observation is dropped by a shift longer than the series."""
        shifted = ShiftTransformer(n_periods=-10, frequency='M').fit_transform(_series())
        assert shifted.notna().sum() == 6


# =============================================================================
# Propriété clé : aller-retour et fenêtre commune
# =============================================================================
class TestRoundTripProperty:
    """Shifting by ``k`` then inverting loses no observation."""

    @pytest.mark.parametrize('freq, shift_freq', INDEX_FREQUENCIES)
    @pytest.mark.parametrize('k', SHIFTS)
    def test_inverse_transform_restores_the_input(self, freq, shift_freq, k):
        """``inverse_transform(transform(x)) == x`` (dates and values).

        ``DatetimeIndex.freq`` is a cached attribute dropped by the index
        concatenation; it is not compared.
        """
        original = _series(freq)
        shifter = ShiftTransformer(n_periods=k, frequency=shift_freq).fit(original)
        recovered = shifter.inverse_transform(shifter.transform(original))
        pd.testing.assert_series_equal(recovered, original, check_freq=False)

    @pytest.mark.parametrize('freq, shift_freq', INDEX_FREQUENCIES)
    @pytest.mark.parametrize('k', SHIFTS)
    def test_common_window_holds_the_value_k_periods_later(self, freq, shift_freq, k):
        """On the dates common to input and output, ``shifted[t] == original[t + k]``."""
        original = _series(freq, periods=12)
        shifted = ShiftTransformer(n_periods=k, frequency=shift_freq).fit_transform(original)
        common = shifted.index.intersection(original.index)
        # Valeur d'or : original.shift(-k) porte en t la valeur observée en t + k périodes
        expected = original.shift(-k).reindex(common)
        pd.testing.assert_series_equal(shifted.reindex(common), expected, check_freq=False)

    def test_inverse_depends_on_parameters_and_grid_only(self):
        """Another shifter with the same parameters, fitted on the same grid, inverts the shift."""
        shifted = ShiftTransformer(n_periods=3, frequency='M').fit_transform(_series())
        recovered = ShiftTransformer(n_periods=3, frequency='M').fit(shifted).inverse_transform(shifted)
        pd.testing.assert_series_equal(recovered, _series(), check_freq=False)

    def test_existing_nan_moves_with_its_date(self):
        """A missing value is shifted like any other: no NaN is created nor filled."""
        series = pd.Series([1., np.nan, 3., 4., 5., 6.], index=pd.date_range('2024-01-01', periods=6, freq='MS'))
        shifted = ShiftTransformer(n_periods=2, frequency='M').fit_transform(series)
        # Valeur d'or : le NaN de 2024-02-01 se retrouve en 2023-12-01
        assert shifted.isna().loc[TS('2023-12-01')] and shifted.isna().sum() == 1


# =============================================================================
# Fréquence du décalage différente de celle de l'index
# =============================================================================
class TestShiftFrequency:
    """``frequency`` sets the shift arithmetic, the detected index frequency the extension."""

    @pytest.mark.parametrize(
        'index_freq, shift_freq, k, expected_first',
        [
            # Valeur d'or : 1 trimestre = 3 mois exactement
            pytest.param('MS', 'Q', 1, '2023-10-01', id='Q-on-MS'),
            # Valeur d'or : 1 an = 4 trimestres avant 2024-03-31
            pytest.param('QE-DEC', 'Y', 1, '2023-03-31', id='Y-on-QE'),
            # Valeur d'or : 1 semaine = 7 jours
            pytest.param('D', 'W', 1, '2023-12-25', id='W-on-D'),
            # Valeur d'or : 1 heure = 60 minutes
            pytest.param('min', 'h', 1, '2023-12-31 23:00', id='h-on-min'),
            # Valeur d'or : 1 mois = 30 jours (durée nominale du DurationConverter,
            # approximation documentée de _convert_shift_periods_to_index_periods)
            pytest.param('D', 'M', 1, '2023-12-02', id='M-on-D'),
            # Valeur d'or : round(30 / 7) = 4 semaines avant le mercredi 2024-01-03
            pytest.param('W-WED', 'M', 1, '2023-12-06', id='M-on-W-WED'),
        ],
    )
    def test_coarser_shift_frequency_is_converted_to_index_periods(self, index_freq, shift_freq, k, expected_first):
        """A shift frequency coarser than the index is converted into index periods."""
        original = _series(index_freq, start='2024-01-03' if index_freq == 'W-WED' else '2024-01-01')
        shifted = ShiftTransformer(n_periods=k, frequency=shift_freq).fit_transform(original)
        assert shifted.index[0] == TS(expected_first)

    def test_finer_shift_frequency_raises(self):
        """Shifting a monthly index by days is rejected with an explicit message."""
        with pytest.raises(ValueError, match="cannot be more granular than index frequency 'M'"):
            ShiftTransformer(n_periods=5, frequency='D').fit_transform(_series())

    def test_unknown_shift_frequency_raises(self):
        """An unknown frequency code is rejected."""
        with pytest.raises(ValueError, match="Unsupported frequency: foo"):
            ShiftTransformer(n_periods=1, frequency='foo').fit_transform(_series())

    def test_multiplied_shift_frequency_is_honoured(self):
        """``frequency='2M'`` shifts by two months per period (ANO-DELAYS-025)."""
        shifted = ShiftTransformer(n_periods=1, frequency='2M').fit_transform(_series())
        # Valeur d'or : une période de deux mois avant 2024-01-01
        assert shifted.index[0] == TS('2023-11-01')

    def test_week_on_business_day_index_is_five_business_days(self):
        """One week on a ``B`` index is five business days (ANO-DELAYS-024)."""
        original = _series('B', periods=10)
        shifted = ShiftTransformer(n_periods=1, frequency='W').fit_transform(original)
        # Valeur d'or : lundi 2024-01-01 - 5 jours ouvrés = lundi 2023-12-25
        assert shifted.index[0] == TS('2023-12-25')

    def test_month_on_business_day_index(self):
        """One month on a ``B`` index is round(30 * 5 / 7) = 21 business days."""
        shifted = ShiftTransformer(n_periods=1, frequency='M').fit_transform(_series('B', periods=10))
        # Valeur d'or : arithmétique pandas de 21 jours ouvrés avant le lundi 2024-01-01
        assert shifted.index[0] == TS('2024-01-01') - 21 * pd.offsets.BDay()


# =============================================================================
# DataFrame, paramètres par variable
# =============================================================================
class TestDataFrameAndVariables:
    """All columns of a frame share the shift; per-variable shifts use one instance per group."""

    @staticmethod
    def _frame() -> pd.DataFrame:
        index = pd.date_range('2024-01-01', periods=6, freq='MS')
        return pd.DataFrame({
            'int_col': range(6),
            'float_col': np.arange(6) * 0.5,
            'str_col': [f'val_{i}' for i in range(6)],
        }, index=index)

    def test_every_column_is_shifted_by_the_same_periods(self):
        """A single shifter moves the rows: each column keeps its values in order."""
        frame = self._frame()
        shifted = ShiftTransformer(n_periods=1, frequency='M').fit_transform(frame)
        expected = frame.set_axis(pd.date_range('2023-12-01', periods=6, freq='MS'))
        pd.testing.assert_frame_equal(shifted, expected, check_freq=False)

    def test_dtypes_are_preserved(self):
        """Integer, float and string columns keep their dtype (no NaN is introduced)."""
        shifted = ShiftTransformer(n_periods=-2, frequency='M').fit_transform(self._frame())
        assert shifted.dtypes.tolist() == self._frame().dtypes.tolist()

    def test_dataframe_round_trip(self):
        """``inverse_transform`` restores a frame exactly."""
        shifter = ShiftTransformer(n_periods=7, frequency='M')
        recovered = shifter.inverse_transform(shifter.fit_transform(self._frame()))
        pd.testing.assert_frame_equal(recovered, self._frame(), check_freq=False)

    def test_special_column_names_are_preserved(self):
        """Columns with spaces, accents or symbols come out unchanged."""
        renamed, mapping = with_special_column_names(self._frame())
        shifted = ShiftTransformer(n_periods=1, frequency='M').fit_transform(renamed)
        assert shifted.columns.tolist() == list(mapping.values())

    def test_all_nan_column_is_accepted(self):
        """A column without any observation does not prevent the shift of the others."""
        frame = self._frame().assign(empty=np.nan)
        shifted = ShiftTransformer(n_periods=1, frequency='M').fit_transform(frame)
        assert shifted['empty'].isna().all() and shifted.index[0] == TS('2023-12-01')

    def test_per_variable_shifts_with_one_instance_per_group(self):
        """Two groups of columns shifted by 1 and 3 months each move by their own shift."""
        frame = self._frame()[['int_col', 'float_col']]
        # Construction : un ShiftTransformer par groupe de colonnes (comme PublicationDelayTransformer)
        first = ShiftTransformer(n_periods=1, frequency='M').fit_transform(frame[['int_col']])
        second = ShiftTransformer(n_periods=3, frequency='M').fit_transform(frame[['float_col']])
        # Valeur d'or : premières dates 2023-12-01 et 2023-10-01
        assert (first.index[0], second.index[0]) == (TS('2023-12-01'), TS('2023-10-01'))


# =============================================================================
# Robustesse d'index et cas limites
# =============================================================================
class TestIndexRobustness:
    """Unsorted, Period, string, duplicated, short or non-datetime indexes."""

    def test_unsorted_input_is_sorted_then_shifted(self):
        """A shuffled series gives the shift of the sorted series."""
        original = _series('D', periods=10)
        shifted = ShiftTransformer(n_periods=1, frequency='D').fit_transform(shuffle_rows(original, seed=3))
        expected = ShiftTransformer(n_periods=1, frequency='D').fit_transform(original)
        pd.testing.assert_series_equal(shifted, expected, check_freq=False)

    def test_period_index_is_returned_as_periods(self):
        """A ``PeriodIndex`` input gives a ``PeriodIndex`` output, shifted by one month."""
        shifted = ShiftTransformer(n_periods=1, frequency='M').fit_transform(to_period_index(_series('MS')))
        pd.testing.assert_index_equal(shifted.index, pd.period_range('2023-12', periods=6, freq='M'), check_names=False)

    def test_period_index_round_trip(self):
        """A ``PeriodIndex`` input is restored exactly."""
        original = to_period_index(_series('MS'))
        shifter = ShiftTransformer(n_periods=-2, frequency='M')
        pd.testing.assert_series_equal(shifter.inverse_transform(shifter.fit_transform(original)), original)

    def test_string_dates_index_is_converted(self):
        """An index of ISO date strings is converted to timestamps."""
        original = _series()
        original.index = original.index.strftime('%Y-%m-%d')
        shifted = ShiftTransformer(n_periods=1, frequency='M').fit_transform(original)
        assert shifted.index[0] == TS('2023-12-01')

    @pytest.mark.parametrize('k', [pytest.param(2, id='positive'), pytest.param(-2, id='negative')])
    def test_index_name_is_preserved(self, k):
        """A non-standard index name survives the shift (ANO-DELAYS-022)."""
        shifted = ShiftTransformer(n_periods=k, frequency='M').fit_transform(with_index_names(_series(), 'periode'))
        assert shifted.index.name == 'periode'

    def test_series_name_is_preserved(self):
        """The name of a series survives the round trip."""
        shifter = ShiftTransformer(n_periods=3, frequency='M')
        assert shifter.inverse_transform(shifter.fit_transform(_series(name='GDP'))).name == 'GDP'

    def test_duplicated_index_is_rejected(self):
        """Duplicated dates are rejected by the validation."""
        with pytest.raises(ValueError, match="Index contains duplicate values"):
            ShiftTransformer(n_periods=1, frequency='M').fit(with_duplicated_rows(_series()))

    @pytest.mark.parametrize('periods', [pytest.param(0, id='empty'), pytest.param(1, id='single')])
    def test_fewer_than_two_observations_are_rejected(self, periods):
        """An empty or single-observation series cannot carry a frequency."""
        with pytest.raises(ValueError, match="minimum required is 2"):
            ShiftTransformer(n_periods=1, frequency='M').fit(_series(periods=periods))

    def test_two_observations_are_enough(self):
        """Two observations are the minimum."""
        shifted = ShiftTransformer(n_periods=1, frequency='M').fit_transform(_series(periods=2))
        assert shifted.index.tolist() == [TS('2023-12-01'), TS('2024-01-01')]

    @pytest.mark.parametrize(
        'index',
        [pytest.param([0, 1, 2], id='range'), pytest.param([2020, 2021, 2022], id='years')],
    )
    def test_non_datetime_index_is_rejected(self, index):
        """An integer index is rejected with an explicit message (legacy ``test_non_datetime_index``)."""
        with pytest.raises(ValueError, match="Index cannot be converted to datetime"):
            ShiftTransformer(n_periods=1, frequency='D').fit_transform(pd.Series([1., 2., 3.], index=index))

    def test_multiplied_index_is_rejected_at_fit(self):
        """A ``2MS`` index is not treated as monthly."""
        series = pd.Series(range(6), index=pd.date_range('2024-01-01', periods=6, freq='2MS'), dtype=float)
        with pytest.raises(ValueError, match="Multiplied index frequency"):
            ShiftTransformer(n_periods=1, frequency='M').fit(series)

    def test_non_pandas_input_is_rejected_by_fit(self):
        """Lists are rejected by ``fit``."""
        with pytest.raises(ValueError, match="must be a pandas Series or DataFrame"):
            ShiftTransformer(n_periods=1, frequency='D').fit([1, 2, 3])

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_non_pandas_input_is_rejected_after_fit(self, method):
        """Lists are rejected by ``transform`` and ``inverse_transform``."""
        shifter = ShiftTransformer(n_periods=1, frequency='D').fit(_series('D'))
        with pytest.raises(ValueError, match="must be a pandas Series or DataFrame"):
            getattr(shifter, method)([1, 2, 3])

    @pytest.mark.parametrize(
        'value',
        [pytest.param(1.5, id='float'), pytest.param(True, id='bool'), pytest.param('1', id='str')],
    )
    def test_non_integer_n_periods_is_rejected(self, value):
        """``n_periods`` must be an integer."""
        with pytest.raises(TypeError, match="'n_periods' must be an integer"):
            ShiftTransformer(n_periods=value, frequency='M').fit(_series())

    def test_numpy_integer_n_periods_is_accepted(self):
        """A numpy integer is a valid number of periods."""
        assert ShiftTransformer(n_periods=np.int64(1), frequency='M').fit_transform(_series()).index[0] == TS('2023-12-01')

    def test_dates_off_the_calendar_grid_are_rejected(self):
        """On a month-start grid, a mid-month date cannot be shifted by whole months exactly."""
        shifter = ShiftTransformer(n_periods=1, frequency='M').fit(_series('MS'))
        off_grid = pd.Series([1.0], index=[TS('2024-01-15')])
        with pytest.raises(ValueError, match="not on its grid"):
            shifter.transform(off_grid)

    def test_dates_off_a_fixed_grid_are_shifted_exactly(self):
        """On a daily grid (fixed-length offset), any time of day is shifted by whole days."""
        shifter = ShiftTransformer(n_periods=1, frequency='D').fit(_series('D'))
        shifted = shifter.transform(pd.Series([1.0], index=[TS('2024-01-15 12:00')]))
        assert shifted.index.tolist() == [TS('2024-01-14 12:00')]

    def test_index_with_gaps_shifts_by_calendar_periods(self):
        """A daily index missing 2024-01-03 is still shifted by one calendar day (ANO-DELAYS-021)."""
        index = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-04', '2024-01-05'])
        series = pd.Series([1., 2., 3., 4.], index=index)
        shifted = ShiftTransformer(n_periods=1, frequency='D').fit_transform(series)
        # Valeur d'or : chaque date recule d'un jour, 2024-01-04 devient 2024-01-03
        pd.testing.assert_index_equal(shifted.index, index - pd.Timedelta(days=1), check_names=False)

    @pytest.mark.parametrize(
        'dates, k',
        [
            # Construction : lacune intérieure
            pytest.param(['2024-01-01', '2024-01-02', '2024-01-04', '2024-01-05'], 1, id='inner-gap-k=1'),
            # Construction : lacune au début (2 janvier absent), décalage négatif
            pytest.param(['2024-01-01', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06'], -2,
                         id='leading-gap-k=-2'),
        ],
    )
    def test_index_with_gaps_round_trip(self, dates, k):
        """On an index with gaps, ``inverse_transform`` restores the input."""
        series = pd.Series(np.arange(len(dates), dtype=float), index=pd.to_datetime(dates))
        shifter = ShiftTransformer(n_periods=k, frequency='D')
        pd.testing.assert_series_equal(shifter.inverse_transform(shifter.fit_transform(series)), series, check_freq=False)

    def test_multiindex_input_raises_a_clear_error(self):
        """A panel passed directly raises a ``ValueError`` pointing to ``PanelwiseTransformer`` (ANO-DELAYS-023)."""
        index = pd.MultiIndex.from_product([['FR', 'DE'], pd.date_range('2024-01-01', periods=6, freq='MS')])
        panel = pd.DataFrame({'a': np.arange(12.)}, index=index)
        with pytest.raises(ValueError, match="wrap it in a PanelwiseTransformer"):
            ShiftTransformer(n_periods=1, frequency='M').fit(panel)


# =============================================================================
# Panel : paramètres par entité, panel désordonné (via PanelwiseTransformer)
# =============================================================================
class TestPanelThroughPanelwise:
    """Per-entity shifts through ``PanelwiseTransformer``, the panel path of the class."""

    @staticmethod
    def _panel() -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [['FR', 'DE'], pd.date_range('2024-01-01', periods=6, freq='MS')], names=['country', 'date'])
        return pd.DataFrame({'a': np.arange(12.), 'b': np.arange(12.) * 10}, index=index)

    def test_per_entity_shift(self):
        """``entity_kwargs`` gives FR a shift of 2 months, DE keeps the default of 1."""
        transformer = PanelwiseTransformer(
            transformer=ShiftTransformer(n_periods=1, frequency='M'),
            entity_kwargs={('FR',): {'n_periods': 2}}, time_col=None, panel_cols=None)
        shifted = transformer.fit_transform(self._panel())
        first_dates = {entity: shifted.loc[entity].index[0] for entity in ('FR', 'DE')}
        # Valeur d'or : 2024-01-01 - 2 mois et - 1 mois
        assert first_dates == {'FR': TS('2023-11-01'), 'DE': TS('2023-12-01')}

    def test_per_entity_round_trip(self):
        """Each entity is restored by ``inverse_transform``, entities in their input order (FR, DE)."""
        transformer = PanelwiseTransformer(
            transformer=ShiftTransformer(n_periods=1, frequency='M'),
            entity_kwargs={('FR',): {'n_periods': -3}}, time_col=None, panel_cols=None)
        recovered = transformer.inverse_transform(transformer.fit_transform(self._panel()))
        pd.testing.assert_frame_equal(recovered, self._panel())

    def test_unsorted_panel_gives_the_sorted_result(self):
        """A shuffled panel (``auto_sort=True``) is shifted like the sorted one."""
        def shift(panel):
            return PanelwiseTransformer(
                transformer=ShiftTransformer(n_periods=1, frequency='M'),
                time_col=None, panel_cols=None, auto_sort=True).fit_transform(panel).sort_index()

        pd.testing.assert_frame_equal(shift(shuffle_rows(self._panel(), seed=0)), shift(self._panel()))
