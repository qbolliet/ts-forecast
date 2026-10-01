"""Unit tests of ``FrequencyConverter.convert_frequency`` and ``convert`` on simple indexes.

Scope: ``convert`` as a pure delegation to ``convert_frequency`` (``from_unit``
ignored), the choice of direction (downsampling, upsampling, identity), the
default ``limit='default'`` and default ``method='mean'``, ``target_position``
in both its code and literal forms, Series edge cases (unsorted, duplicated,
empty, single observation, ``PeriodIndex``, irregular index), column-wise
conversion of DataFrames (special column names, ``time_col``), and the output
contract of the DataFrame path, fixed after U8 (ANO-UTILS-055, 056, 058, 067):
unconverted columns keep their observed dates only, the source position is
preserved, the index name survives an upsampling, the result does not depend
on the order of the columns.

Panels are covered by ``test_panel.py``, ``target_freq`` dictionaries on
DataFrames by ``test_mixed_frequencies.py``, parameter validation by
``test_validation.py``.
"""
import numpy as np
import pandas as pd
import pytest

from tests.support.perturbations import with_special_column_names
from tsforecast.utils.frequency.converter import FrequencyConverter


@pytest.fixture
def converter() -> FrequencyConverter:
    """A fresh ``FrequencyConverter``."""
    return FrequencyConverter()


@pytest.fixture
def monthly() -> pd.Series:
    """Month ends of 2024 H1: 1, 2, …, 6."""
    return pd.Series(np.arange(1.0, 7.0), index=pd.date_range('2024-01-31', periods=6, freq='ME'))


@pytest.fixture
def quarterly() -> pd.Series:
    """Quarter ends of 2024: 100, 200, 300, 400."""
    return pd.Series(
        [100.0, 200.0, 300.0, 400.0], index=pd.date_range('2024-03-31', periods=4, freq='QE')
    )


# =============================================================================
# convert : délégation
# =============================================================================

class TestConvertDelegatesToConvertFrequency:
    """``convert(value, from_unit, to_unit)`` is ``convert_frequency(value, to_unit)``."""

    @pytest.mark.parametrize('from_unit', ['monthly', 'annual', 'nonsense'])
    def test_from_unit_is_ignored(self, converter, monthly, from_unit):
        """The source frequency is always detected, whatever ``from_unit`` says."""
        result = converter.convert(monthly, from_unit, 'QE', method='sum')
        # Valeurs d'or : 1 + 2 + 3, 4 + 5 + 6
        assert result.tolist() == [6.0, 15.0]

    def test_dataframe(self, converter, monthly):
        """DataFrames and keyword arguments go through unchanged."""
        frame = pd.DataFrame({'a': monthly, 'b': monthly * 10})
        result = converter.convert(frame, 'monthly', 'QE', method='sum')
        expected = pd.DataFrame({'a': [6.0, 15.0], 'b': [60.0, 150.0]}, index=result.index)
        pd.testing.assert_frame_equal(result, expected)


# =============================================================================
# Sens de conversion
# =============================================================================

class TestConversionDirection:
    """The direction is chosen from the source and target frequencies."""

    # Valeurs d'or de `monthly` (1..6) agrégé par trimestre
    @pytest.mark.parametrize(
        'method, expected',
        [
            pytest.param('sum', [6.0, 15.0], id='sum'),
            pytest.param('mean', [2.0, 5.0], id='mean'),
            pytest.param('last', [3.0, 6.0], id='last'),
        ],
    )
    def test_downsampling(self, converter, monthly, method, expected):
        """A coarser target aggregates."""
        result = converter.convert_frequency(monthly, 'QE', method=method)
        assert result.tolist() == expected

    def test_upsampling(self, converter, quarterly):
        """A finer target interpolates over the whole first quarter."""
        result = converter.convert_frequency(quarterly, 'ME', method='linear')
        # Valeurs d'or : janvier-février comblés vers l'arrière (100), puis pas de 100/3
        expected = [100.0, 100.0, 100.0, 400 / 3, 500 / 3, 200.0,
                    700 / 3, 800 / 3, 300.0, 1000 / 3, 1100 / 3, 400.0]
        assert result.tolist() == pytest.approx(expected)

    def test_default_limit_is_the_conversion_factor(self, converter, quarterly):
        """``convert_frequency`` defaults to ``limit='default'`` (3 months per quarter)."""
        with_gap = quarterly.copy()
        with_gap.iloc[1] = np.nan
        result = converter.convert_frequency(with_gap, 'ME', method='linear')
        # Valeurs d'or : trou avril-août ; 3 mois comblés vers l'arrière depuis
        # septembre (juin 200, juillet 700/3, août 800/3), avril et mai restent NaN
        assert result.loc['2024-04-30':'2024-08-31'].tolist() == pytest.approx(
            [np.nan, np.nan, 200.0, 700 / 3, 800 / 3], nan_ok=True
        )

    def test_default_method_only_fits_downsampling(self, converter, quarterly):
        """``method='mean'`` (default) is not an interpolation method."""
        with pytest.raises(ValueError, match='mean'):
            converter.convert_frequency(quarterly, 'ME')

    @pytest.mark.parametrize('target', ['ME', 'M', 'monthly'], ids=['ME', 'bare-M', 'label'])
    def test_same_frequency_is_returned_unchanged(self, converter, monthly, target):
        """A target equal to the source (position included) returns the data as is."""
        result = converter.convert_frequency(monthly, target)
        pd.testing.assert_series_equal(result, monthly)

    def test_series_keeps_the_source_position(self, converter):
        """A position-less target inherits the start position of a Series."""
        month_starts = pd.Series(
            np.arange(1.0, 7.0), index=pd.date_range('2024-01-01', periods=6, freq='MS')
        )
        result = converter.convert_frequency(month_starts, 'Q', method='sum')
        # Valeur d'or : trimestres étiquetés à leur premier jour, comme la source
        assert list(result.index) == [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-04-01')]

    @pytest.mark.parametrize(
        'position, first_label',
        [
            pytest.param('start', '2024-01-01', id='literal-start'),
            pytest.param('S', '2024-01-01', id='code-S'),
            pytest.param('end', '2024-03-31', id='literal-end'),
            pytest.param('E', '2024-03-31', id='code-E'),
        ],
    )
    def test_target_position_forms(self, converter, monthly, position, first_label):
        """``target_position`` accepts codes and literals and wins over the source position."""
        result = converter.convert_frequency(monthly, 'Q', method='sum', target_position=position)
        assert result.index[0] == pd.Timestamp(first_label)


# =============================================================================
# Cas limites sur Series
# =============================================================================

class TestSeriesEdgeCases:
    """Inputs the validation sorts, rejects or converts."""

    def test_unsorted_series_is_sorted(self, converter, monthly):
        """The input is sorted before conversion."""
        result = converter.convert_frequency(monthly.iloc[::-1], 'QE', method='sum')
        assert result.tolist() == [6.0, 15.0]

    def test_duplicated_dates_raise(self, converter):
        """Duplicated dates are rejected by the validation."""
        duplicated = pd.Series(
            [1.0, 2.0, 3.0], index=pd.DatetimeIndex(['2024-01-31', '2024-01-31', '2024-02-29'])
        )
        with pytest.raises(ValueError, match='duplicate'):
            converter.convert_frequency(duplicated, 'QE', method='sum')

    def test_empty_series_raises(self, converter):
        """An empty Series is refused before any detection."""
        empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        with pytest.raises(ValueError, match='Cannot convert empty data'):
            converter.convert_frequency(empty, 'QE', method='sum')

    def test_single_observation_raises(self, converter):
        """No frequency can be detected from a single observation."""
        single = pd.Series([1.0], index=pd.DatetimeIndex(['2024-01-31']))
        with pytest.raises(ValueError, match='non-null observations'):
            converter.convert_frequency(single, 'QE', method='sum')

    def test_irregular_series_raises(self, converter):
        """An irregular Series has no detectable frequency."""
        irregular = pd.Series(
            [1.0, 2.0, 3.0, 4.0],
            index=pd.DatetimeIndex(['2024-01-01', '2024-01-03', '2024-01-10', '2024-02-20']),
        )
        with pytest.raises(ValueError, match='Cannot detect current frequency'):
            converter.convert_frequency(irregular, 'MS', method='sum')

    def test_period_index_is_accepted(self, converter):
        """A monthly ``PeriodIndex`` is converted to quarter-end timestamps."""
        periods = pd.Series([1.0, 2.0, 3.0], index=pd.period_range('2024-01', periods=3, freq='M'))
        result = converter.convert_frequency(periods, 'QE', method='sum')
        pd.testing.assert_series_equal(
            result, pd.Series([6.0], index=pd.DatetimeIndex(['2024-03-31'])), check_freq=False
        )


# =============================================================================
# DataFrame colonne par colonne
# =============================================================================

class TestDataFrameConversion:
    """Every column is converted from its own detected frequency."""

    def test_string_target(self, converter, monthly):
        """Same target for every column, explicit start position."""
        frame = pd.DataFrame({'a': monthly, 'b': monthly * 10})
        result = converter.convert_frequency(frame, 'QS', method='sum')
        expected = pd.DataFrame(
            {'a': [6.0, 15.0], 'b': [60.0, 150.0]},
            index=pd.DatetimeIndex(['2024-01-01', '2024-04-01']),
        )
        pd.testing.assert_frame_equal(result, expected, check_freq=False)

    def test_explicit_target_position(self, converter, monthly):
        """``target_position`` applies to every column."""
        frame = pd.DataFrame({'a': monthly})
        result = converter.convert_frequency(frame, 'Q', method='sum', target_position='start')
        assert list(result.index) == [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-04-01')]

    def test_special_column_names_are_kept(self, converter, monthly):
        """Spaces, accents and symbols in column names survive, in order."""
        frame, mapping = with_special_column_names(pd.DataFrame({'a': monthly, 'b': monthly * 10}))
        result = converter.convert_frequency(frame, 'QE', method='sum')
        assert list(result.columns) == [mapping['a'], mapping['b']]
        assert result[mapping['b']].tolist() == [60.0, 150.0]

    @pytest.mark.filterwarnings('ignore:Index replaced')
    def test_time_column(self, converter, monthly):
        """With ``time_col``, the output is indexed by the time column."""
        frame = pd.DataFrame({'date': monthly.index, 'a': monthly.to_numpy()})
        result = converter.convert_frequency(frame, 'QE', method='sum', time_col='date')
        expected = pd.DataFrame(
            {'a': [6.0, 15.0]}, index=pd.DatetimeIndex(['2024-03-31', '2024-06-30'], name='date')
        )
        pd.testing.assert_frame_equal(result, expected, check_freq=False)

    def test_empty_dataframe_raises(self, converter, monthly):
        """A DataFrame without any row is refused."""
        empty = pd.DataFrame({'a': monthly}).iloc[0:0]
        with pytest.raises(ValueError, match='Cannot convert empty data'):
            converter.convert_frequency(empty, 'QE', method='sum')


class TestDataFrameOutputContract:
    """The output index is the target grid and the observed dates, nothing else."""

    def test_all_nan_column_does_not_keep_the_source_grid(self, converter, monthly):
        """A column that is never observed does not prevent the quarterly output."""
        frame = pd.DataFrame({'ventes': monthly, 'prix': np.nan})
        result = converter.convert_frequency(frame, 'QE', method='sum')
        # Valeur d'or : deux fins de trimestre, ventes 6 et 15, prix NaN
        pd.testing.assert_index_equal(
            result.index, pd.DatetimeIndex(['2024-03-31', '2024-06-30']), exact=False
        )

    def test_dict_key_on_all_nan_column_does_not_keep_the_source_grid(self, converter, monthly):
        """A never observed column named in the dictionary is not a column 'absent from the dict'."""
        frame = pd.DataFrame({'ventes': monthly, 'prix': np.nan})
        result = converter.convert_frequency(frame, {'ventes': 'QE', 'prix': 'QE'}, method='sum')
        # Valeur d'or : l'union des index cibles, soit les deux fins de trimestre
        pd.testing.assert_index_equal(
            result.index, pd.DatetimeIndex(['2024-03-31', '2024-06-30']), exact=False
        )

    def test_column_already_at_target_does_not_keep_the_row_grid(self, converter):
        """A quarterly column on a monthly grid appears at its quarter ends only."""
        month_ends = pd.date_range('2024-01-31', periods=12, freq='ME')
        frame = pd.DataFrame({'mensuel': np.arange(1.0, 13.0), 'trimestriel': np.nan}, index=month_ends)
        frame.loc[month_ends.month % 3 == 0, 'trimestriel'] = [10.0, 20.0, 30.0, 40.0]

        result = converter.convert_frequency(frame, 'QE', method='sum')

        # Valeurs d'or : sommes trimestrielles 6, 15, 24, 33 ; trimestriel inchangé
        expected = pd.DataFrame(
            {'mensuel': [6.0, 15.0, 24.0, 33.0], 'trimestriel': [10.0, 20.0, 30.0, 40.0]},
            index=pd.date_range('2024-03-31', periods=4, freq='QE'),
        )
        pd.testing.assert_frame_equal(result, expected, check_freq=False)

    def test_dataframe_keeps_the_source_position(self, converter):
        """A position-less target inherits the start position, as for a Series."""
        month_starts = pd.DataFrame(
            {'a': np.arange(1.0, 7.0)}, index=pd.date_range('2024-01-01', periods=6, freq='MS')
        )
        result = converter.convert_frequency(month_starts, 'Q', method='sum')
        assert list(result.index) == [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-04-01')]

    def test_result_does_not_depend_on_column_order(self, converter):
        """Permuting the columns permutes the output columns, nothing else."""
        # Grille mensuelle de janvier 2020 à juin 2021 : q trimestrielle, y annuelle ;
        # l'extension de q s'arrête fin juin 2021, celle de y fin décembre 2021
        grid = pd.date_range('2020-01-01', '2021-06-01', freq='MS')
        frame = pd.DataFrame({'q': np.nan, 'y': np.nan}, index=grid)
        frame.loc[grid.month.isin([1, 4, 7, 10]), 'q'] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        frame.loc[['2020-01-01', '2021-01-01'], 'y'] = [100.0, 112.0]

        forward = converter.convert_frequency(frame[['q', 'y']], 'MS', method='linear')
        backward = converter.convert_frequency(frame[['y', 'q']], 'MS', method='linear')

        pd.testing.assert_frame_equal(forward, backward[['q', 'y']])

    def test_upsampling_keeps_the_index_name(self, converter, quarterly):
        """The name of the source index survives an upsampling conversion."""
        result = converter.convert_frequency(quarterly.rename_axis('date'), 'ME', method='linear')
        assert result.index.name == 'date'

    def test_column_absent_from_the_dict_keeps_its_observed_dates(self, converter):
        """An untargeted quarterly column on a monthly grid keeps its quarter ends only."""
        month_ends = pd.date_range('2024-01-31', periods=12, freq='ME')
        frame = pd.DataFrame({'mensuel': np.arange(1.0, 13.0), 'trimestriel': np.nan}, index=month_ends)
        frame.loc[month_ends.month % 3 == 0, 'trimestriel'] = [10.0, 20.0, 30.0, 40.0]

        result = converter.convert_frequency(frame, {'mensuel': 'QE'}, method='sum')

        # Valeurs d'or : union des fins de trimestre (mensuel converti) et des dates
        # observées de trimestriel (les mêmes) ; aucune ligne de bourrage mensuelle
        expected = pd.DataFrame(
            {'mensuel': [6.0, 15.0, 24.0, 33.0], 'trimestriel': [10.0, 20.0, 30.0, 40.0]},
            index=pd.date_range('2024-03-31', periods=4, freq='QE'),
        )
        pd.testing.assert_frame_equal(result, expected, check_freq=False)

    def test_nothing_to_convert_drops_the_padding_rows(self, converter):
        """A quarterly column already at its target comes back at its quarter ends."""
        month_ends = pd.date_range('2024-01-31', periods=12, freq='ME')
        frame = pd.DataFrame({'trimestriel': np.nan}, index=month_ends)
        frame.loc[month_ends.month % 3 == 0, 'trimestriel'] = [10.0, 20.0, 30.0, 40.0]

        result = converter.convert_frequency(frame, 'QE', method='sum')

        expected = pd.DataFrame(
            {'trimestriel': [10.0, 20.0, 30.0, 40.0]},
            index=pd.date_range('2024-03-31', periods=4, freq='QE'),
        )
        pd.testing.assert_frame_equal(result, expected, check_freq=False)

    def test_never_observed_frame_gives_no_row(self, converter, monthly):
        """Without any observed value, no date is kept (but the columns are)."""
        frame = pd.DataFrame({'a': monthly * np.nan, 'b': monthly * np.nan})
        result = converter.convert_frequency(frame, 'QE', method='sum')
        assert result.empty and list(result.columns) == ['a', 'b']

    def test_never_observed_series_gives_no_row(self, converter, monthly):
        """Same rule for a Series: no observed value, no date (ANO-UTILS-070)."""
        result = converter.convert_frequency((monthly * np.nan).rename('a'), 'QE', method='sum')
        assert result.empty and result.name == 'a'

    def test_frame_without_detectable_column_raises(self, converter, monthly):
        """Observed values without any detectable frequency raise, as for a Series."""
        # Une seule observation par colonne : aucune fréquence lisible
        frame = pd.DataFrame({'a': monthly * np.nan, 'b': monthly * np.nan})
        frame.iloc[0, 0] = 1.0
        frame.iloc[3, 1] = 2.0
        with pytest.raises(ValueError, match='Cannot detect current frequency of any column'):
            converter.convert_frequency(frame, 'QE', method='sum')
