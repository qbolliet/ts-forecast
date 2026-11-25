"""Index-based ``validate_temporal_data``.

Scenarios: argument checks common to both paths, flat index validation and panel
``MultiIndex`` validation with two and three levels.
"""

import numpy as np
import pandas as pd
import pytest

# Jeux de données et perturbations partagés
from tests.support import perturbations as perturb

# Fonctions à tester
from tsforecast.utils.validation import validate_temporal_data

# Outils de test partagés du paquet
from tests.unit.utils.validation.utils.helpers import _months, _panel


# Le chemin « colonnes » émet systématiquement un avertissement de remplacement
# d'index (comportement testé explicitement dans TestColumnBasedValidation) ; le
# bruit est masqué ailleurs pour garder une sortie lisible. L'inférence de format
# de pandas sur des chaînes de dates non ISO émet aussi un avertissement.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Index replaced with"),
    pytest.mark.filterwarnings("ignore:Could not infer format"),
]


# =============================================================================
# validate_temporal_data — contrat d'entrée
# =============================================================================
class TestValidateTemporalDataInputContract:
    """Argument checks common to both validation paths."""

    @pytest.mark.parametrize(
        'bad_input',
        [[1, 2, 3], np.arange(3), None, {'a': 1}],
        ids=['list', 'ndarray', 'none', 'dict'],
    )
    def test_non_pandas_input_is_rejected(self, bad_input):
        """A non Series / DataFrame input raises ``ValueError``."""
        with pytest.raises(ValueError, match='pandas Series or DataFrame'):
            validate_temporal_data(bad_input)

    def test_panel_cols_without_time_col_is_rejected(self):
        """``panel_cols`` alone (no ``time_col``) is an inconsistent request."""
        df = pd.DataFrame({'country': ['A'], 'v': [1]}, index=_months(1))
        with pytest.raises(ValueError, match='Cannot specify panel_cols without time_col'):
            validate_temporal_data(df, panel_cols=['country'])

    def test_empty_panel_cols_without_time_col_takes_the_index_path(self):
        """``panel_cols=[]`` is falsy: the index-based path is used, no error."""
        s = pd.Series([1, 2], index=_months(2), name='v')
        out = validate_temporal_data(s, panel_cols=[])
        assert isinstance(out.index, pd.DatetimeIndex)

    def test_series_input_returns_a_series(self):
        """A ``Series`` in gives a ``Series`` out (same values, sorted index)."""
        s = pd.Series([1, 2, 3], index=_months(3), name='v')
        out = validate_temporal_data(s)
        assert isinstance(out, pd.Series)

    def test_dataframe_input_returns_a_dataframe(self):
        """A ``DataFrame`` in gives a ``DataFrame`` out."""
        df = pd.DataFrame({'v': [1, 2, 3]}, index=_months(3))
        assert isinstance(validate_temporal_data(df), pd.DataFrame)

    def test_named_series_keeps_its_name(self):
        """The name of a named ``Series`` survives the validation."""
        s = pd.Series([1, 2, 3], index=_months(3), name='chiffre d\'affaires')
        assert validate_temporal_data(s).name == 'chiffre d\'affaires'

    def test_unnamed_series_stays_unnamed(self):
        """An unnamed ``Series`` comes back unnamed, not named ``0``.

        The name ``0`` is the label of the temporary column created by
        ``Series.to_frame()``; the validation must not leak it.
        """
        s = pd.Series([1, 2, 3], index=_months(3))
        assert validate_temporal_data(s).name is None

    def test_series_with_time_col_is_rejected(self):
        """A ``Series`` has no column: ``time_col`` cannot be found."""
        s = pd.Series([1, 2, 3], index=_months(3), name='v')
        with pytest.raises(ValueError, match="Time column 'date' not found"):
            validate_temporal_data(s, time_col='date')

    @pytest.mark.parametrize('path', ['index', 'columns'])
    def test_input_is_never_mutated(self, path):
        """The caller's frame is left untouched, string dates included.

        The conversion of the index / of ``time_col`` happens on an internal
        copy: the caller's object keeps its ``object`` dtype and its order.
        """
        dates = ['2023-02-01', '2023-01-01']
        if path == 'index':
            df = pd.DataFrame({'v': [1, 2]}, index=dates)
            kwargs = {}
        else:
            df = pd.DataFrame({'date': dates, 'v': [1, 2]})
            kwargs = {'time_col': 'date'}
        before = df.copy()

        validate_temporal_data(df, **kwargs)

        pd.testing.assert_frame_equal(df, before)


# =============================================================================
# validate_temporal_data — chemin « index »
# =============================================================================
class TestIndexBasedValidation:
    """Index-based validation (``time_col=None``, ``panel_cols=None``) on a flat index."""

    def test_datetime_index_passes_through_unchanged(self):
        """An already valid, sorted, unique ``DatetimeIndex`` is returned as is."""
        s = pd.Series([1.0, 2.0, 3.0], index=_months(3), name='v')
        pd.testing.assert_series_equal(validate_temporal_data(s), s, check_freq=False)

    def test_no_warning_on_a_valid_datetime_index(self, recwarn):
        """The index path is silent when there is nothing to correct."""
        df = pd.DataFrame({'v': [1, 2, 3]}, index=_months(3))
        validate_temporal_data(df)
        assert len(recwarn) == 0

    def test_timezone_is_preserved(self):
        """A tz-aware ``DatetimeIndex`` keeps its time zone."""
        index = pd.date_range('2023-01-01', periods=3, tz='Europe/Paris')
        out = validate_temporal_data(pd.DataFrame({'v': range(3)}, index=index))
        assert str(out.index.tz) == 'Europe/Paris'

    @pytest.mark.parametrize('container', ['series', 'dataframe'])
    def test_string_index_is_converted_and_sorted(self, container):
        """A string index of ISO dates is converted to ``DatetimeIndex`` then sorted.

        Golden value: values ``[1, 2]`` attached to ``2023-02-01`` and
        ``2023-01-01`` come back as ``[2, 1]`` on the sorted dates.
        """
        index = ['2023-02-01', '2023-01-01']
        data = pd.Series([1, 2], index=index, name='v')
        if container == 'dataframe':
            data = data.to_frame()

        out = validate_temporal_data(data)

        assert list(out.index) == [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01')]
        assert list(np.ravel(out.to_numpy())) == [2, 1]

    def test_non_convertible_index_raises_when_strict(self):
        """A non-date index is an error in strict mode."""
        s = pd.Series([1, 2], index=['a', 'b'])
        with pytest.raises(ValueError, match='Index cannot be converted to datetime'):
            validate_temporal_data(s, strict=True)

    def test_non_convertible_index_warns_and_returns_data_when_not_strict(self):
        """Non strict mode warns and returns the data with their original index."""
        s = pd.Series([1, 2], index=['a', 'b'], name='v')
        with pytest.warns(UserWarning, match='Index conversion to datetime failed'):
            out = validate_temporal_data(s, strict=False)
        assert list(out.index) == ['a', 'b']
        assert list(out) == [1, 2]

    def test_duplicated_index_raises_when_strict(self):
        """Two rows on the same date are an error in strict mode."""
        index = pd.DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01'])
        with pytest.raises(ValueError, match='Index contains duplicate values'):
            validate_temporal_data(pd.Series([1, 2, 3], index=index), strict=True)

    def test_duplicated_index_keeps_first_occurrence_when_not_strict(self):
        """Non strict mode warns and keeps the first row of each duplicated date.

        Golden value: values ``[1, 2, 3]`` on ``[01-01, 01-01, 02-01]`` give
        ``[1, 3]`` on ``[01-01, 02-01]``.
        """
        index = pd.DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01'])
        with pytest.warns(UserWarning, match='Keeping first occurrence'):
            out = validate_temporal_data(pd.Series([1, 2, 3], index=index), strict=False)
        assert list(out) == [1, 3]
        assert list(out.index) == [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01')]

    def test_unsorted_data_is_sorted_by_default(self):
        """``sort_data=True`` orders the rows chronologically (values follow their dates)."""
        index = pd.to_datetime(['2023-03-01', '2023-01-01', '2023-02-01'])
        out = validate_temporal_data(pd.Series([30, 10, 20], index=index))
        assert list(out) == [10, 20, 30]

    def test_unsorted_data_keeps_its_order_when_sort_is_disabled(self):
        """``sort_data=False`` leaves the row order alone."""
        index = pd.to_datetime(['2023-03-01', '2023-01-01', '2023-02-01'])
        out = validate_temporal_data(pd.Series([30, 10, 20], index=index), sort_data=False)
        assert list(out) == [30, 10, 20]

    def test_empty_datetime_frame_is_accepted(self):
        """An empty frame with a ``DatetimeIndex`` passes (edge case: empty dataset)."""
        df = pd.DataFrame({'v': []}, index=pd.DatetimeIndex([]))
        assert validate_temporal_data(df).shape == (0, 1)

    def test_empty_range_index_frame_becomes_an_empty_datetime_index(self):
        """An empty frame with a ``RangeIndex`` is converted to an empty ``DatetimeIndex``."""
        out = validate_temporal_data(pd.DataFrame({'v': []}))
        assert isinstance(out.index, pd.DatetimeIndex) and len(out) == 0

    def test_single_observation_is_accepted(self):
        """A single row passes (edge case: no frequency can be inferred from one point)."""
        df = pd.DataFrame({'v': [1]}, index=_months(1))
        assert validate_temporal_data(df).shape == (1, 1)

    def test_a_single_nat_is_sorted_last(self):
        """A missing date (``NaT``) is kept and pushed to the end by the sort."""
        index = pd.DatetimeIndex(['2023-01-02', None, '2023-01-01'])
        out = validate_temporal_data(pd.DataFrame({'v': [0, 1, 2]}, index=index))
        assert list(out['v']) == [2, 0, 1]
        assert pd.isna(out.index[-1])

    @pytest.mark.parametrize(
        'labels',
        [[2020, 2021], [0, 1], [2020.0, 2021.0]],
        ids=['integer-years', 'range-like', 'floats'],
    )
    def test_numeric_index_is_rejected_when_strict(self, labels):
        """Numeric labels are not dates and raise in strict mode.

        ``pandas.to_datetime`` would read ``2020`` as ``1970-01-01 00:00:00.000002020``.
        """
        with pytest.raises(ValueError, match='Index cannot be converted to datetime'):
            validate_temporal_data(pd.DataFrame({'v': [1, 2]}, index=labels), strict=True)

    def test_default_range_index_is_rejected_when_strict(self):
        """A frame left with its default ``RangeIndex`` is not a time series."""
        with pytest.raises(ValueError, match='Index cannot be converted to datetime'):
            validate_temporal_data(pd.DataFrame({'v': [1, 2]}))

    def test_numeric_index_warns_and_returns_data_when_not_strict(self):
        """Non strict mode warns and returns the numeric index untouched."""
        df = pd.DataFrame({'v': [1, 2]}, index=[2020, 2021])
        with pytest.warns(UserWarning, match='Index conversion to datetime failed'):
            out = validate_temporal_data(df, strict=False)
        assert list(out.index) == [2020, 2021]

    def test_period_index_is_converted_to_first_instants(self):
        """A ``PeriodIndex`` is accepted and read at the start of each period.

        Golden value: monthly periods ``2023-01``, ``2023-02``, ``2023-03`` become
        ``2023-01-01``, ``2023-02-01``, ``2023-03-01``.
        """
        s = pd.Series([1, 2, 3], index=pd.period_range('2023-01', periods=3, freq='M'), name='v')
        out = validate_temporal_data(s, strict=True)
        assert isinstance(out.index, pd.DatetimeIndex)
        assert list(out.index) == list(_months(3))

    def test_unsorted_period_index_is_sorted(self):
        """Period labels take part in the sort like dates."""
        index = pd.PeriodIndex(['2023-03', '2023-01', '2023-02'], freq='M')
        out = validate_temporal_data(pd.Series([30, 10, 20], index=index))
        assert list(out) == [10, 20, 30]


# =============================================================================
# validate_temporal_data — chemin « index » sur un MultiIndex
# =============================================================================
class TestMultiIndexValidation:
    """Index-based validation on a panel ``MultiIndex`` (2 and 3 levels)."""

    def test_two_level_panel_passes_through(self):
        """A valid sorted ``(entity, date)`` panel is returned equal to its input."""
        panel = _panel()
        pd.testing.assert_frame_equal(validate_temporal_data(panel), panel)

    def test_three_level_panel_passes_through(self):
        """A valid ``(region, entity, date)`` panel is returned equal to its input."""
        panel = perturb.to_three_level_index(_panel())
        pd.testing.assert_frame_equal(validate_temporal_data(panel), panel)

    def test_series_with_multiindex_returns_a_series(self):
        """A panel ``Series`` keeps its ``MultiIndex`` and its name."""
        s = _panel()['v'].rename('valeur')
        out = validate_temporal_data(s)
        assert isinstance(out, pd.Series) and out.name == 'valeur'
        assert out.index.names == s.index.names

    def test_unsorted_three_level_panel_is_sorted_lexicographically(self):
        """Rows are ordered by region, then entity, then date."""
        panel = perturb.to_three_level_index(_panel(entities=('FR', 'DE')))
        out = validate_temporal_data(panel.iloc[::-1])
        assert out.index.is_monotonic_increasing
        assert out.index[0][1] == 'DE'

    def test_string_last_level_is_converted(self):
        """ISO strings on the last level become datetimes; level names are kept."""
        index = pd.MultiIndex.from_product(
            [['A', 'B'], ['2023-01-01', '2023-02-01']], names=['entity', 'date']
        )
        out = validate_temporal_data(pd.DataFrame({'v': range(4)}, index=index))
        assert isinstance(out.index.get_level_values(-1), pd.DatetimeIndex)
        assert list(out.index.names) == ['entity', 'date']

    def test_non_convertible_last_level_raises_when_strict(self):
        """A last level that is not a date is an error in strict mode."""
        index = pd.MultiIndex.from_product([['A', 'B'], ['x', 'y']])
        with pytest.raises(ValueError, match='Last level of MultiIndex cannot be converted'):
            validate_temporal_data(pd.DataFrame({'v': range(4)}, index=index), strict=True)

    def test_non_convertible_last_level_warns_when_not_strict(self):
        """Non strict mode warns and returns the frame with its index."""
        index = pd.MultiIndex.from_product([['A', 'B'], ['x', 'y']])
        with pytest.warns(UserWarning, match='MultiIndex last level conversion failed'):
            out = validate_temporal_data(pd.DataFrame({'v': range(4)}, index=index), strict=False)
        assert out.shape == (4, 1)

    def test_duplicated_combinations_raise_when_strict(self):
        """Two rows on the same ``(entity, date)`` are an error in strict mode."""
        dates = pd.to_datetime(['2023-01-01', '2023-01-01', '2023-02-01'])
        index = pd.MultiIndex.from_arrays([['A', 'A', 'A'], dates])
        with pytest.raises(ValueError, match='MultiIndex contains duplicate combinations'):
            validate_temporal_data(pd.DataFrame({'v': [1, 2, 3]}, index=index), strict=True)

    def test_duplicated_combinations_keep_first_when_not_strict(self):
        """Non strict mode keeps the first row of a duplicated ``(entity, date)``."""
        dates = pd.to_datetime(['2023-01-01', '2023-01-01', '2023-02-01'])
        index = pd.MultiIndex.from_arrays([['A', 'A', 'A'], dates])
        with pytest.warns(UserWarning, match='MultiIndex contains duplicates'):
            out = validate_temporal_data(pd.DataFrame({'v': [1, 2, 3]}, index=index), strict=False)
        assert list(out['v']) == [1, 3]

    def test_same_date_for_two_entities_is_not_a_duplicate(self):
        """The uniqueness applies to the full key, not to the date alone."""
        out = validate_temporal_data(_panel(), strict=True)
        assert out.index.is_unique

    @pytest.mark.filterwarnings("ignore:Parsing dates in")
    @pytest.mark.parametrize('strict', [True, False], ids=['strict', 'not-strict'])
    def test_day_first_string_dates_convert_like_a_flat_index(self, strict):
        """``dd/mm/yyyy`` strings on the last level convert as they do on a flat index.

        Golden value: ``15/01/2023`` and ``02/02/2023`` (day first, inferred
        from the first value ``15/01``) are ``2023-01-15`` and ``2023-02-02``.
        The ``MultiIndex`` path does not convert the *sorted levels* (``02/02/2023``
        first, read month first) and fail, whereas the flat index accepts the same values.
        """
        values = ['15/01/2023', '02/02/2023']
        flat = validate_temporal_data(pd.DataFrame({'v': [1, 2]}, index=values), strict=strict)
        index = pd.MultiIndex.from_arrays([['A', 'A'], values])

        out = validate_temporal_data(pd.DataFrame({'v': [1, 2]}, index=index), strict=strict)

        assert list(out.index.get_level_values(-1)) == list(flat.index)

    def test_integer_last_level_is_rejected_when_strict(self):
        """Integer labels on the last level are not dates."""
        index = pd.MultiIndex.from_arrays([['A', 'A'], [2020, 2021]])
        with pytest.raises(ValueError, match='Last level of MultiIndex cannot be converted'):
            validate_temporal_data(pd.DataFrame({'v': [1, 2]}, index=index), strict=True)

    def test_integer_last_level_warns_when_not_strict(self):
        """Non strict mode warns and returns the frame with its index."""
        index = pd.MultiIndex.from_arrays([['A', 'A'], [2020, 2021]])
        with pytest.warns(UserWarning, match='MultiIndex last level conversion failed'):
            out = validate_temporal_data(pd.DataFrame({'v': [1, 2]}, index=index), strict=False)
        assert list(out.index.get_level_values(-1)) == [2020, 2021]

    def test_period_last_level_is_converted_to_first_instants(self):
        """A ``PeriodIndex`` last level is read at the start of each period."""
        periods = pd.period_range('2023-01', periods=2, freq='M')
        index = pd.MultiIndex.from_arrays([['A', 'A'], periods], names=['entity', 'date'])
        out = validate_temporal_data(pd.DataFrame({'v': [1, 2]}, index=index), strict=True)
        assert list(out.index.get_level_values(-1)) == list(_months(2))
        assert list(out.index.names) == ['entity', 'date']

    def test_empty_panel_is_accepted(self):
        """An empty panel keeps its structure (edge case: empty dataset)."""
        out = validate_temporal_data(perturb.empty_like(_panel()))
        assert out.shape == (0, 1)
