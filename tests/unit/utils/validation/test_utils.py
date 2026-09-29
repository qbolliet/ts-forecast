"""Unit tests for ``tsforecast.utils.validation.utils``.

Covers the four public functions of the module — ``validate_temporal_data``,
``restore_original_structure``, ``validate_entities_grouped`` and
``validate_sorted_within_groups`` — through their public API only (the
private helpers ``_validate_index_based``, ``_validate_column_based``,
``_build_metadata``, ... are exercised via ``validate_temporal_data``).

Both validation paths are exercised (index-based and ``time_col`` /
``panel_cols`` column-based), with ``strict`` True and False, on plain time
series, ``MultiIndex`` panels of 2 and 3 levels, and on the realistic
datasets of ``tests/support`` perturbed by ``tests/support/perturbations.py``.

Anomalies found while writing these tests are registered in
``tests/ANOMALIES.md`` (``ANO-UTILS-024`` to ``ANO-UTILS-032``, all fixed).
"""
# Modules de base
import re

import numpy as np
import pandas as pd
import pytest

# Jeux de données et perturbations partagés
from tests.support import perturbations as perturb

# Fonctions à tester
from tsforecast.utils.validation import (
    restore_original_structure,
    validate_entities_grouped,
    validate_sorted_within_groups,
    validate_temporal_data,
)

# Le chemin « colonnes » émet systématiquement un avertissement de remplacement
# d'index (comportement testé explicitement dans TestColumnBasedValidation) ; le
# bruit est masqué ailleurs pour garder une sortie lisible. L'inférence de format
# de pandas sur des chaînes de dates non ISO émet aussi un avertissement.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Index replaced with"),
    pytest.mark.filterwarnings("ignore:Could not infer format"),
]


# =============================================================================
# Constructeurs locaux de petits jeux à valeurs d'or calculables
# =============================================================================
def _months(periods: int = 4, start: str = '2023-01-01') -> pd.DatetimeIndex:
    """Build a month-start ``DatetimeIndex`` (``MS``) of ``periods`` dates."""
    return pd.date_range(start, periods=periods, freq='MS')


def _panel(entities=('A', 'B'), periods: int = 3, names=('entity', 'date')) -> pd.DataFrame:
    """Build a small sorted two-level panel with values ``0..n-1`` in column ``v``.

    Args:
        entities: Entity labels, in the order of the blocks.
        periods: Number of monthly dates per entity.
        names: Names of the two index levels.

    Returns:
        DataFrame indexed by ``(entity, date)``, sorted by entity then date.
    """
    index = pd.MultiIndex.from_product([list(entities), _months(periods)], names=list(names))
    return pd.DataFrame({'v': range(len(index))}, index=index)


def _flat(df: pd.DataFrame) -> pd.DataFrame:
    """Move the index levels of a frame into columns (column-based input)."""
    return df.reset_index()


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
        """An unnamed ``Series`` comes back unnamed, not named ``0`` (``ANO-UTILS-027``).

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
        """Numeric labels are not dates and raise in strict mode (``ANO-UTILS-028``).

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
        """A ``PeriodIndex`` is accepted and read at the start of each period (``ANO-UTILS-029``).

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
        Regression of ``ANO-UTILS-030``: the ``MultiIndex`` path used to convert the
        *sorted levels* (``02/02/2023`` first, read month first) and failed, while
        the flat index accepted the same values.
        """
        values = ['15/01/2023', '02/02/2023']
        flat = validate_temporal_data(pd.DataFrame({'v': [1, 2]}, index=values), strict=strict)
        index = pd.MultiIndex.from_arrays([['A', 'A'], values])

        out = validate_temporal_data(pd.DataFrame({'v': [1, 2]}, index=index), strict=strict)

        assert list(out.index.get_level_values(-1)) == list(flat.index)

    def test_integer_last_level_is_rejected_when_strict(self):
        """Integer labels on the last level are not dates (``ANO-UTILS-028``)."""
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
        """A ``PeriodIndex`` last level is read at the start of each period (``ANO-UTILS-029``)."""
        periods = pd.period_range('2023-01', periods=2, freq='M')
        index = pd.MultiIndex.from_arrays([['A', 'A'], periods], names=['entity', 'date'])
        out = validate_temporal_data(pd.DataFrame({'v': [1, 2]}, index=index), strict=True)
        assert list(out.index.get_level_values(-1)) == list(_months(2))
        assert list(out.index.names) == ['entity', 'date']

    def test_empty_panel_is_accepted(self):
        """An empty panel keeps its structure (edge case: empty dataset)."""
        out = validate_temporal_data(perturb.empty_like(_panel()))
        assert out.shape == (0, 1)


# =============================================================================
# validate_temporal_data — chemin « colonnes »
# =============================================================================
class TestColumnBasedValidation:
    """Column-based validation (``time_col`` and optional ``panel_cols``)."""

    def test_time_col_becomes_a_named_datetime_index(self):
        """``time_col`` moves from the columns to a ``DatetimeIndex`` of the same name."""
        df = pd.DataFrame({'date': _months(4), 'value': [10, 20, 30, 40]})
        out = validate_temporal_data(df, time_col='date')
        assert isinstance(out.index, pd.DatetimeIndex) and out.index.name == 'date'
        assert 'date' not in out.columns
        assert list(out['value']) == [10, 20, 30, 40]

    def test_time_col_and_panel_col_build_a_two_level_index(self):
        """Panel columns come first in the index, the time column last."""
        out = validate_temporal_data(_flat(_panel()), time_col='date', panel_cols=['entity'])
        assert list(out.index.names) == ['entity', 'date']

    def test_two_panel_cols_build_a_three_level_index(self):
        """Panel columns keep the order in which they are given."""
        flat = _flat(perturb.to_three_level_index(_panel()))
        out = validate_temporal_data(flat, time_col='date', panel_cols=['region', 'entity'])
        assert list(out.index.names) == ['region', 'entity', 'date']

    def test_string_time_col_is_converted(self):
        """A string time column is converted to ``datetime64``."""
        df = pd.DataFrame({'date': ['2023-02-01', '2023-01-01'], 'v': [1, 2]})
        out = validate_temporal_data(df, time_col='date')
        assert list(out.index) == [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01')]
        assert list(out['v']) == [2, 1]

    def test_rows_are_sorted_by_panel_then_time(self):
        """Golden order: ``(A, 01-01)``, ``(A, 02-01)``, ``(B, 02-01)``."""
        df = pd.DataFrame({
            'entity': ['B', 'A', 'A'],
            'date': pd.to_datetime(['2023-02-01', '2023-02-01', '2023-01-01']),
            'v': [3, 2, 1],
        })
        out = validate_temporal_data(df, time_col='date', panel_cols=['entity'])
        assert list(out['v']) == [1, 2, 3]

    def test_row_order_is_kept_when_sort_is_disabled(self):
        """``sort_data=False`` leaves the input row order alone."""
        df = pd.DataFrame({
            'entity': ['B', 'A', 'A'],
            'date': pd.to_datetime(['2023-02-01', '2023-02-01', '2023-01-01']),
            'v': [3, 2, 1],
        })
        out = validate_temporal_data(df, time_col='date', panel_cols=['entity'], sort_data=False)
        assert list(out['v']) == [3, 2, 1]

    def test_previous_index_is_dropped(self):
        """The former index is replaced, whatever it was."""
        df = pd.DataFrame({'date': _months(2), 'v': [1, 2]}, index=['x', 'y'])
        out = validate_temporal_data(df, time_col='date')
        assert list(out.index) == list(_months(2))

    def test_replacement_of_the_index_is_announced_without_metadata(self):
        """Without ``return_metadata`` a ``UserWarning`` announces the index replacement."""
        df = pd.DataFrame({'date': _months(2), 'v': [1, 2]})
        with pytest.warns(UserWarning, match=re.escape("Index replaced with ['date']")):
            validate_temporal_data(df, time_col='date')

    def test_replacement_of_the_index_is_silent_with_metadata(self, recwarn):
        """With ``return_metadata=True`` the caller can restore the structure: no warning."""
        df = pd.DataFrame({'date': _months(2), 'v': [1, 2]})
        validate_temporal_data(df, time_col='date', return_metadata=True)
        assert len(recwarn) == 0

    def test_missing_time_col_is_rejected(self):
        """An unknown ``time_col`` raises with its name."""
        df = pd.DataFrame({'date': _months(2), 'v': [1, 2]})
        with pytest.raises(ValueError, match="Time column 'nope' not found in data"):
            validate_temporal_data(df, time_col='nope')

    def test_missing_panel_col_is_rejected(self):
        """An unknown panel column raises with its name."""
        df = pd.DataFrame({'date': _months(2), 'v': [1, 2]})
        with pytest.raises(ValueError, match='Panel columns not found in data'):
            validate_temporal_data(df, time_col='date', panel_cols=['nope'])

    @pytest.mark.parametrize('strict', [True, False], ids=['strict', 'not-strict'])
    def test_non_convertible_time_col_raises_in_both_modes(self, strict):
        """A non-date time column is an error even when ``strict=False``.

        Unlike the index path (which warns and returns the data), the column
        path has no fallback: without a usable time column no index can be
        built. The ``strict`` docstring now says so (``ANO-UTILS-031``).
        """
        df = pd.DataFrame({'date': ['a', 'b'], 'v': [1, 2]})
        with pytest.raises(ValueError, match="Column 'date' cannot be converted to datetime"):
            validate_temporal_data(df, time_col='date', strict=strict)

    def test_period_time_col_is_converted_to_first_instants(self):
        """A ``Period`` time column is read at the start of each period (``ANO-UTILS-029``)."""
        df = pd.DataFrame({'date': pd.period_range('2023-01', periods=2, freq='M'), 'v': [1, 2]})
        out = validate_temporal_data(df, time_col='date')
        assert list(out.index) == list(_months(2))

    @pytest.mark.parametrize('labels', [[2020, 2021], [2020.0, 2021.0]], ids=['integers', 'floats'])
    @pytest.mark.parametrize('strict', [True, False], ids=['strict', 'not-strict'])
    def test_numeric_time_col_is_rejected(self, labels, strict):
        """A numeric time column is not a date and raises in both modes (``ANO-UTILS-028``)."""
        df = pd.DataFrame({'date': labels, 'v': [1, 2]})
        with pytest.raises(ValueError, match="Column 'date' cannot be converted to datetime"):
            validate_temporal_data(df, time_col='date', strict=strict)

    def test_ambiguous_time_col_is_rejected(self):
        """A duplicated ``time_col`` label designates two columns: no date to read."""
        df = pd.DataFrame([[_months(1)[0], 1, 2]], columns=['date', 'v', 'date'])
        with pytest.raises(ValueError, match="Column 'date' cannot be converted to datetime"):
            validate_temporal_data(df, time_col='date')

    def test_duplicated_keys_raise_when_strict(self):
        """Two rows on the same ``(entity, date)`` are an error in strict mode."""
        flat = _flat(_panel())
        flat = pd.concat([flat, flat.iloc[:1]])
        with pytest.raises(
            ValueError,
            match=re.escape("Duplicate rows found for combination of columns: ['entity', 'date']"),
        ):
            validate_temporal_data(flat, time_col='date', panel_cols=['entity'], strict=True)

    def test_duplicated_keys_keep_first_when_not_strict(self):
        """Non strict mode warns and drops later duplicates of a key."""
        flat = _flat(_panel())
        duplicate = flat.iloc[[0]].assign(v=999)
        with pytest.warns(UserWarning, match='Keeping first occurrence'):
            out = validate_temporal_data(
                pd.concat([flat, duplicate]), time_col='date', panel_cols=['entity'], strict=False
            )
        assert 999 not in out['v'].tolist()
        assert len(out) == len(flat)

    def test_empty_frame_is_accepted(self):
        """An empty frame with an (empty) datetime column passes."""
        df = pd.DataFrame({'date': pd.to_datetime([]), 'v': []})
        assert validate_temporal_data(df, time_col='date').shape == (0, 1)

    def test_single_observation_is_accepted(self):
        """A single row passes."""
        df = pd.DataFrame({'date': _months(1), 'v': [1]})
        assert validate_temporal_data(df, time_col='date').shape == (1, 1)

    def test_special_column_names_are_kept(self):
        """Time / panel / value columns with spaces and accents work unchanged."""
        df = pd.DataFrame({
            'période (mois)': _months(2),
            'pays / zone': ['FR', 'FR'],
            'taux de chômage %': [7.1, 7.2],
        })
        out = validate_temporal_data(df, time_col='période (mois)', panel_cols=['pays / zone'])
        assert list(out.index.names) == ['pays / zone', 'période (mois)']
        assert list(out.columns) == ['taux de chômage %']


# =============================================================================
# validate_temporal_data — métadonnées renvoyées
# =============================================================================
class TestReturnedMetadata:
    """Metadata returned with ``return_metadata=True`` (built via the public API)."""

    def test_no_metadata_by_default(self):
        """Without ``return_metadata`` the function returns the data alone."""
        out = validate_temporal_data(pd.Series([1, 2], index=_months(2)))
        assert isinstance(out, pd.Series)

    def test_returns_a_pair_when_requested(self):
        """With ``return_metadata`` the result is a ``(data, dict)`` pair."""
        result = validate_temporal_data(pd.Series([1, 2], index=_months(2)), return_metadata=True)
        assert isinstance(result, tuple) and isinstance(result[1], dict)

    def test_metadata_of_a_flat_datetime_index(self):
        """Golden metadata for a named ``DatetimeIndex`` (nothing replaced)."""
        index = pd.date_range('2023-01-01', periods=3, freq='MS', name='periode')
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}, index=index)

        _, meta = validate_temporal_data(df, return_metadata=True)

        assert meta['index_type'] == 'DatetimeIndex'
        assert meta['index_name'] == 'periode'
        assert meta['original_columns'] == ['a', 'b']
        assert meta['index_was_replaced'] is False
        assert meta['had_time_col_in_columns'] is False
        assert meta['had_panel_cols_in_columns'] is False
        assert meta['time_col'] is None and meta['panel_cols'] is None
        assert meta['was_sorted'] is True
        assert meta['rows_reordered'] is False
        pd.testing.assert_index_equal(meta['original_index'], index)

    def test_metadata_of_a_multiindex(self):
        """A ``MultiIndex`` is described by ``index_names`` (not ``index_name``)."""
        _, meta = validate_temporal_data(_panel(), return_metadata=True)
        assert meta['index_type'] == 'MultiIndex'
        assert list(meta['index_names']) == ['entity', 'date']
        assert 'index_name' not in meta

    def test_metadata_of_the_column_path(self):
        """Golden metadata for a ``time_col`` + ``panel_cols`` request."""
        flat = _flat(_panel())

        _, meta = validate_temporal_data(
            flat, time_col='date', panel_cols=['entity'], return_metadata=True
        )

        assert meta['index_was_replaced'] is True
        assert meta['had_time_col_in_columns'] is True
        assert meta['had_panel_cols_in_columns'] is True
        assert meta['time_col'] == 'date' and meta['panel_cols'] == ['entity']
        assert meta['original_columns'] == ['entity', 'date', 'v']
        assert meta['index_type'] == 'RangeIndex'
        pd.testing.assert_index_equal(meta['original_index'], flat.index)

    def test_time_col_only_request_records_no_panel_columns(self):
        """A ``time_col`` alone replaces the index but records no panel columns."""
        df = pd.DataFrame({'timestamp': _months(4), 'revenue': [1000, 1100, 1050, 1200]})
        _, meta = validate_temporal_data(df, time_col='timestamp', return_metadata=True)
        assert meta['index_was_replaced'] is True
        assert meta['had_panel_cols_in_columns'] is False
        assert meta['panel_cols'] is None
        assert isinstance(meta['original_index'], pd.RangeIndex) and len(meta['original_index']) == 4

    def test_metadata_does_not_alias_the_callers_panel_cols(self):
        """The recorded ``panel_cols`` is a copy: later edits by the caller do not leak."""
        panel_cols = ['entity']
        _, meta = validate_temporal_data(
            _flat(_panel()), time_col='date', panel_cols=panel_cols, return_metadata=True
        )
        panel_cols.append('other')
        assert meta['panel_cols'] == ['entity']

    def test_sort_flag_is_recorded(self):
        """``was_sorted`` reflects the ``sort_data`` argument."""
        s = pd.Series([1, 2], index=_months(2))
        _, meta = validate_temporal_data(s, sort_data=False, return_metadata=True)
        assert meta['was_sorted'] is False

    @pytest.mark.parametrize(
        'dates, sort_data, expected',
        [
            (['2023-01-01', '2023-02-01'], True, False),
            (['2023-02-01', '2023-01-01'], True, True),
            (['2023-02-01', '2023-01-01'], False, False),
        ],
        ids=['already-sorted', 'reordered-by-the-sort', 'sort-disabled'],
    )
    def test_rows_reordered_flag(self, dates, sort_data, expected):
        """``rows_reordered`` is true only when the sort actually moved rows."""
        s = pd.Series([1, 2], index=pd.to_datetime(dates))
        _, meta = validate_temporal_data(s, sort_data=sort_data, return_metadata=True)
        assert meta['rows_reordered'] is expected

    def test_original_index_is_the_index_before_any_correction(self):
        """Metadata describes the input as received (unsorted, duplicates included)."""
        index = pd.DatetimeIndex(['2023-02-01', '2023-01-01', '2023-01-01'])
        s = pd.Series([1, 2, 3], index=index)
        with pytest.warns(UserWarning):
            _, meta = validate_temporal_data(s, strict=False, return_metadata=True)
        pd.testing.assert_index_equal(meta['original_index'], index)

    def test_original_index_survives_later_edits_of_the_output(self):
        """The recorded index is a copy, independent of the returned data."""
        df = pd.DataFrame({'v': [1, 2]}, index=_months(2))
        out, meta = validate_temporal_data(df, return_metadata=True)
        out.index = out.index + pd.Timedelta(days=1)
        assert meta['original_index'][0] == pd.Timestamp('2023-01-01')


# =============================================================================
# restore_original_structure
# =============================================================================
class TestRestoreOriginalStructure:
    """Restoration of the structure recorded by ``validate_temporal_data``."""

    def test_basic_restoration(self):
        """A validate → restore round trip on a sorted ``time_col`` frame is the identity."""
        original = pd.DataFrame({
            'timestamp': _months(4),
            'revenue': [1000, 1100, 1050, 1200],
            'users': [50, 55, 52, 60],
        })
        validated, meta = validate_temporal_data(original, time_col='timestamp', return_metadata=True)

        restored = restore_original_structure(validated, meta)

        pd.testing.assert_frame_equal(restored, original)

    def test_restoration_preserves_dates(self):
        """The time column comes back with its dates."""
        original = pd.DataFrame({'date': pd.date_range('2023-01-01', periods=5), 'value': range(5)})
        validated, meta = validate_temporal_data(original, time_col='date', return_metadata=True)

        restored = restore_original_structure(validated, meta)

        pd.testing.assert_series_equal(restored['date'], original['date'])

    def test_string_dates_come_back_as_datetimes(self):
        """The converted time column is restored as ``datetime64``, not as the input strings (pinned).

        The metadata records no dtype: only the converted values are available.
        """
        original = pd.DataFrame({'date': ['2023-01-01', '2023-02-01'], 'v': [1, 2]})
        validated, meta = validate_temporal_data(original, time_col='date', return_metadata=True)
        restored = restore_original_structure(validated, meta)
        assert list(restored['date']) == [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01')]

    def test_restoration_with_panel_data(self):
        """Panel and time columns come back as columns, with all values."""
        original = pd.DataFrame({
            'entity': ['US'] * 3 + ['FR'] * 3,
            'date': _months(3).tolist() * 2,
            'value': range(6),
        })
        validated, meta = validate_temporal_data(
            original, time_col='date', panel_cols=['entity'], return_metadata=True
        )

        restored = restore_original_structure(validated, meta)

        # Les lignes sont triées par entité (FR avant US) : comparaison indépendante de l'ordre
        key = ['entity', 'date']
        pd.testing.assert_frame_equal(
            restored.sort_values(key).reset_index(drop=True),
            original.sort_values(key).reset_index(drop=True),
        )

    def test_restoration_after_transformations(self):
        """Columns added to the validated frame are carried through the restoration."""
        original = pd.DataFrame({
            'timestamp': _months(4),
            'revenue': [1000, 1100, 1050, 1200],
        })
        validated, meta = validate_temporal_data(original, time_col='timestamp', return_metadata=True)
        validated['revenue_change'] = validated['revenue'].pct_change()

        restored = restore_original_structure(validated, meta)

        assert list(restored.columns) == ['timestamp', 'revenue', 'revenue_change']
        assert not restored['revenue'].isna().any()

    def test_index_path_restores_a_string_index(self):
        """On the index path the original (string) index is restored as such."""
        original = pd.DataFrame({'v': [1, 2]}, index=['2023-01-01', '2023-02-01'])
        validated, meta = validate_temporal_data(original, return_metadata=True)
        restored = restore_original_structure(validated, meta)
        pd.testing.assert_frame_equal(restored, original)

    def test_series_with_datetime_index_round_trips(self):
        """A named ``Series`` on the index path is restored equal to itself."""
        original = pd.Series([1.0, 2.0, 3.0], index=_months(3, ), name='v')
        validated, meta = validate_temporal_data(original, return_metadata=True)
        restored = restore_original_structure(validated, meta)
        pd.testing.assert_series_equal(restored, original, check_freq=False)

    def test_unnamed_series_round_trips_unnamed(self):
        """An unnamed ``Series`` is restored unnamed (``ANO-UTILS-027``)."""
        original = pd.Series([1, 2, 3], index=_months(3))
        validated, meta = validate_temporal_data(original, return_metadata=True)
        assert restore_original_structure(validated, meta).name is None

    def test_restoration_does_not_mutate_its_input(self):
        """The validated frame given to the function is left untouched."""
        original = pd.DataFrame({'date': _months(3), 'v': [1, 2, 3]})
        validated, meta = validate_temporal_data(original, time_col='date', return_metadata=True)
        before = validated.copy()

        restore_original_structure(validated, meta)

        pd.testing.assert_frame_equal(validated, before)

    def test_empty_metadata_leaves_the_data_unchanged(self):
        """Without recorded structure, the data are returned as a copy, unchanged."""
        df = pd.DataFrame({'v': [1, 2]}, index=_months(2))
        restored = restore_original_structure(df, {})
        pd.testing.assert_frame_equal(restored, df)
        assert restored is not df

    def test_index_is_not_restored_when_the_row_count_changed(self):
        """A length mismatch (rows dropped after validation) leaves the index as is (pinned).

        Typical case: ``strict=False`` dropped a duplicated row, so the
        recorded ``original_index`` is longer than the validated data. The
        function silently keeps the current index instead of raising.
        """
        index = pd.DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01'])
        original = pd.DataFrame({'v': [1, 2, 3]}, index=index)
        with pytest.warns(UserWarning):
            validated, meta = validate_temporal_data(original, strict=False, return_metadata=True)

        restored = restore_original_structure(validated, meta)

        pd.testing.assert_frame_equal(restored, validated)

    def test_index_path_restoration_after_sort_keeps_each_value_on_its_date(self):
        """After a sort, each value stays on its own date (index path, ``ANO-UTILS-024``).

        Golden value: ``30`` sits on ``2023-03-01`` in the input, so it still sits on
        ``2023-03-01`` after ``validate → restore``. The original row order is not
        restored (rows stay sorted): the code used to glue the unsorted original
        index onto the sorted rows, so ``2023-03-01`` ended up with the value ``10``.
        """
        index = pd.to_datetime(['2023-03-01', '2023-01-01', '2023-02-01'])
        original = pd.Series([30, 10, 20], index=index, name='v')
        validated, meta = validate_temporal_data(original, sort_data=True, return_metadata=True)

        restored = restore_original_structure(validated, meta)

        pd.testing.assert_series_equal(restored, original.sort_index(), check_freq=False)

    def test_column_path_restoration_after_sort_gives_the_sorted_rows(self):
        """After a sort, the rows come back sorted with fresh labels (column path).

        The former ``RangeIndex`` labels no longer line up with the sorted rows, so
        they are not reassigned; each ``(date, value)`` pair stays intact.
        """
        original = pd.DataFrame({
            'date': pd.to_datetime(['2023-03-01', '2023-01-01', '2023-02-01']),
            'v': [30, 10, 20],
        })
        validated, meta = validate_temporal_data(
            original, time_col='date', sort_data=True, return_metadata=True
        )

        restored = restore_original_structure(validated, meta)

        expected = original.sort_values('date').reset_index(drop=True)
        pd.testing.assert_frame_equal(restored, expected)

    def test_period_index_is_restored_after_conversion(self):
        """A ``PeriodIndex`` converted to timestamps for validation comes back as periods."""
        original = pd.Series(
            [1, 2, 3], index=pd.period_range('2023-01', periods=3, freq='M'), name='v'
        )
        validated, meta = validate_temporal_data(original, return_metadata=True)

        restored = restore_original_structure(validated, meta)

        pd.testing.assert_series_equal(restored, original)

    def test_columns_are_restored_at_their_original_positions(self):
        """Columns come back in their original order: ``['date', 'v', 'entity']`` (``ANO-UTILS-025``).

        ``reset_index()`` puts the former index columns first, in index order; the
        recorded ``original_columns`` puts them back where they were.
        """
        original = pd.DataFrame({
            'date': _months(3).tolist() * 2,
            'v': range(6),
            'entity': ['A'] * 3 + ['B'] * 3,
        })
        validated, meta = validate_temporal_data(
            original, time_col='date', panel_cols=['entity'], return_metadata=True
        )

        restored = restore_original_structure(validated, meta)

        assert list(restored.columns) == ['date', 'v', 'entity']

    def test_columns_added_after_validation_stay_last(self):
        """A column added to the validated frame is placed after the original columns."""
        original = pd.DataFrame({'date': _months(3), 'v': [1, 2, 3], 'w': [4, 5, 6]})
        validated, meta = validate_temporal_data(original, time_col='date', return_metadata=True)
        validated['extra'] = 0

        restored = restore_original_structure(validated, meta)

        assert list(restored.columns) == ['date', 'v', 'w', 'extra']

    def test_missing_column_record_leaves_the_reset_order(self):
        """Without recorded ``original_columns``, index columns simply come first."""
        validated = pd.DataFrame({'v': [1, 2]}, index=pd.Index(_months(2), name='date'))
        restored = restore_original_structure(validated, {'index_was_replaced': True})
        assert list(restored.columns) == ['date', 'v']

    def test_a_series_of_the_column_path_gets_its_original_index_back(self):
        """A value ``Series`` restored after the column path keeps its values (``ANO-UTILS-026``).

        The values come back on the original ``RangeIndex``; the dates, carried by the
        validated index only, are dropped (a ``Series`` has no column to put them in).
        """
        original = pd.DataFrame({'date': _months(3), 'v': [10, 20, 30]})
        validated, meta = validate_temporal_data(original, time_col='date', return_metadata=True)

        restored = restore_original_structure(validated['v'], meta)

        pd.testing.assert_series_equal(restored, original['v'])

    def test_a_reordered_series_keeps_its_time_index(self):
        """After a sort, a ``Series`` keeps the time index it has (no label to restore)."""
        original = pd.DataFrame({
            'date': pd.to_datetime(['2023-02-01', '2023-01-01']), 'v': [20, 10]
        })
        validated, meta = validate_temporal_data(original, time_col='date', return_metadata=True)

        restored = restore_original_structure(validated['v'], meta)

        assert list(restored) == [10, 20]
        assert isinstance(restored.index, pd.DatetimeIndex)


# =============================================================================
# Aller-retour validate_temporal_data → restore_original_structure
# =============================================================================
def _order_preserving_perturbations(kind: str) -> dict:
    """Return the perturbations that keep rows sorted, keyed by a readable id.

    Args:
        kind: ``'timeseries'`` or ``'panel'`` (some perturbations need a panel).

    Returns:
        Dictionary ``{id: function(df) -> df}``.
    """
    index_names = 'periode' if kind == 'timeseries' else ['pays', 'periode']
    perturbations = {
        'identity': lambda df: df,
        'special_column_names': lambda df: perturb.with_special_column_names(df)[0],
        'custom_index_names': lambda df: perturb.with_index_names(df, index_names),
        'period_start': perturb.to_period_start,
        'period_end': perturb.to_period_end,
        'single_observation': perturb.single_observation,
        'empty': perturb.empty_like,
    }
    if kind == 'panel':
        perturbations['three_level_index'] = perturb.to_three_level_index
        perturbations['drop_entity'] = lambda df: perturb.drop_entity(df, 'Italie')
    return perturbations


def _order_changing_perturbations(kind: str) -> dict:
    """Return the perturbations that leave rows unsorted, keyed by a readable id."""
    perturbations = {'shuffle_rows': lambda df: perturb.shuffle_rows(df, seed=0)}
    if kind == 'panel':
        perturbations['reverse_entities'] = perturb.reverse_entities
    return perturbations


def _round_trip_cases(perturbations_of) -> list:
    """Cross ``kind`` x ``path`` x perturbation into ``pytest.param`` cases."""
    cases = []
    for kind in ('timeseries', 'panel'):
        for path in ('index', 'columns'):
            for name in perturbations_of(kind):
                cases.append(pytest.param(kind, path, name, id=f'{kind}-{path}-{name}'))
    return cases


def _dataset(request: pytest.FixtureRequest, kind: str) -> pd.DataFrame:
    """Return the realistic dataset (notebook 3) matching ``kind``."""
    fixture = 'irregular_index_timeseries' if kind == 'timeseries' else 'heterogeneous_coverage_panel'
    return request.getfixturevalue(fixture)


def _validate_kwargs(df: pd.DataFrame, path: str) -> tuple:
    """Return ``(input frame, kwargs)`` to validate ``df`` on the requested path.

    Args:
        df: Frame with a datetime last index level, indexed by entity levels first.
        path: ``'index'`` (keep the index) or ``'columns'`` (index moved to columns).

    Returns:
        The frame to validate and the ``time_col`` / ``panel_cols`` keyword arguments.
    """
    if path == 'index':
        return df, {}
    names = list(df.index.names)
    kwargs = {'time_col': names[-1]}
    if len(names) > 1:
        kwargs['panel_cols'] = names[:-1]
    return df.reset_index(), kwargs


class TestRoundTripOnPerturbedDatasets:
    """``validate → restore`` on the realistic datasets, perturbed one way at a time."""

    @pytest.mark.parametrize(
        'kind, path, perturbation', _round_trip_cases(_order_preserving_perturbations)
    )
    def test_round_trip_is_the_identity(self, request, kind, path, perturbation):
        """A sorted, valid frame is restored exactly as it was."""
        df = _order_preserving_perturbations(kind)[perturbation](_dataset(request, kind))
        data, kwargs = _validate_kwargs(df, path)
        validated, meta = validate_temporal_data(data, return_metadata=True, **kwargs)

        restored = restore_original_structure(validated, meta)

        pd.testing.assert_frame_equal(restored, data, check_freq=False)

    @pytest.mark.parametrize(
        'kind, path, perturbation', _round_trip_cases(_order_changing_perturbations)
    )
    def test_round_trip_of_unsorted_data_gives_the_sorted_frame(
        self, request, kind, path, perturbation
    ):
        """An unsorted frame comes back sorted, every row intact (``ANO-UTILS-024``).

        The original row order is not restored. Index path: the sorted original.
        Column path: the rows sorted by ``(panel_cols, time_col)`` with fresh labels.
        """
        df = _order_changing_perturbations(kind)[perturbation](_dataset(request, kind))
        data, kwargs = _validate_kwargs(df, path)
        validated, meta = validate_temporal_data(data, return_metadata=True, **kwargs)

        restored = restore_original_structure(validated, meta)

        if path == 'index':
            expected = data.sort_index()
        else:
            keys = kwargs.get('panel_cols', []) + [kwargs['time_col']]
            expected = data.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(restored, expected, check_freq=False)

    @pytest.mark.parametrize(
        'kind, path, perturbation', _round_trip_cases(_order_changing_perturbations)
    )
    def test_round_trip_of_unsorted_data_without_sort_is_the_identity(
        self, request, kind, path, perturbation
    ):
        """With ``sort_data=False`` the round trip is the identity even on unsorted data."""
        df = _order_changing_perturbations(kind)[perturbation](_dataset(request, kind))
        data, kwargs = _validate_kwargs(df, path)
        validated, meta = validate_temporal_data(
            data, sort_data=False, return_metadata=True, **kwargs
        )

        restored = restore_original_structure(validated, meta)

        pd.testing.assert_frame_equal(restored, data, check_freq=False)

    @pytest.mark.parametrize('kind', ['timeseries', 'panel'])
    def test_validation_orders_unsorted_data_on_realistic_datasets(self, request, kind):
        """Whatever the shuffle, the validated index is monotonic increasing (index path)."""
        df = perturb.shuffle_rows(_dataset(request, kind), seed=1)
        assert validate_temporal_data(df).index.is_monotonic_increasing

    @pytest.mark.parametrize('kind', ['timeseries', 'panel'])
    def test_validation_only_reorders_rows(self, request, kind):
        """Sorting a shuffled dataset gives back the original dataset (index path).

        Every row follows its own index label: the validated frame equals the
        sorted original.
        """
        original = _dataset(request, kind)
        shuffled = perturb.shuffle_rows(original, seed=2)

        out = validate_temporal_data(shuffled)

        pd.testing.assert_frame_equal(out, original.sort_index(), check_freq=False)

    @pytest.mark.parametrize('strict', [True, False], ids=['strict', 'not-strict'])
    def test_duplicated_rows_on_a_realistic_panel(self, request, strict):
        """Duplicated rows: an error when strict, dropped (keep first) when not."""
        panel = _dataset(request, 'panel')
        duplicated = perturb.with_duplicated_rows(panel, n=3)

        if strict:
            with pytest.raises(ValueError, match='MultiIndex contains duplicate combinations'):
                validate_temporal_data(duplicated, strict=True)
        else:
            with pytest.warns(UserWarning, match='Keeping first occurrence'):
                out = validate_temporal_data(duplicated, strict=False)
            pd.testing.assert_frame_equal(out, panel.sort_index(), check_freq=False)

    def test_duplicated_rows_on_the_column_path(self, request):
        """Duplicated ``(entity, date)`` rows on the column path follow the same contract."""
        panel = _dataset(request, 'panel')
        flat = perturb.with_duplicated_rows(panel, n=3).reset_index()
        with pytest.raises(ValueError, match='Duplicate rows found'):
            validate_temporal_data(flat, time_col='date', panel_cols=['country'], strict=True)

    def test_period_index_dataset_round_trips(self, mixed_freq_timeseries):
        """A regular series on a ``PeriodIndex`` is validated, then restored as periods."""
        period_ts = perturb.to_period_index(mixed_freq_timeseries)
        assert isinstance(period_ts.index, pd.PeriodIndex), "la perturbation doit produire un PeriodIndex"

        validated, meta = validate_temporal_data(period_ts, return_metadata=True)
        restored = restore_original_structure(validated, meta)

        assert list(validated.index) == list(period_ts.index.to_timestamp())
        pd.testing.assert_frame_equal(restored, period_ts)

    def test_period_column_dataset_is_validated(self, mixed_freq_timeseries):
        """A ``Period`` time column is read at the start of each period (column path)."""
        period_ts = perturb.to_period_index(mixed_freq_timeseries)
        flat = period_ts.reset_index()

        validated = validate_temporal_data(flat, time_col='date')

        assert list(validated.index) == list(period_ts.index.to_timestamp())

    def test_period_index_panel_round_trips(self, mixed_freq_panel):
        """A regular panel with a ``PeriodIndex`` date level is validated, then restored."""
        period_panel = perturb.to_period_index(mixed_freq_panel)
        assert isinstance(period_panel.index.get_level_values(-1), pd.PeriodIndex)

        validated, meta = validate_temporal_data(period_panel, return_metadata=True)
        restored = restore_original_structure(validated, meta)

        dates = validated.index.get_level_values(-1)
        assert list(dates) == list(period_panel.index.get_level_values(-1).to_timestamp())
        pd.testing.assert_frame_equal(restored, period_panel)


# =============================================================================
# validate_entities_grouped
# =============================================================================
class TestValidateEntitiesGrouped:
    """Contiguity of each entity's observations."""

    def test_non_pandas_input_is_rejected(self):
        """A non Series / DataFrame input raises ``ValueError``."""
        with pytest.raises(ValueError, match='pandas Series or DataFrame'):
            validate_entities_grouped([1, 2, 3])

    @pytest.mark.parametrize('container', ['dataframe', 'series'])
    def test_contiguous_entities_are_grouped(self, container):
        """Entities in contiguous blocks give ``True``."""
        panel = _panel()
        data = panel if container == 'dataframe' else panel['v']
        assert validate_entities_grouped(data) is True

    @pytest.mark.parametrize('container', ['dataframe', 'series'])
    def test_interleaved_entities_are_not_grouped(self, container):
        """``A, B, A, B, A, B`` gives ``False``."""
        index = pd.MultiIndex.from_arrays(
            [['A', 'B'] * 3, _months(6)], names=['entity', 'date']
        )
        data = pd.DataFrame({'v': range(6)}, index=index)
        assert validate_entities_grouped(data if container == 'dataframe' else data['v']) is False

    def test_an_entity_reappearing_after_another_is_not_grouped(self):
        """``A, A, B, A`` gives ``False`` (a single reappearance is enough)."""
        index = pd.MultiIndex.from_arrays([['A', 'A', 'B', 'A'], _months(4)])
        assert validate_entities_grouped(pd.DataFrame({'v': range(4)}, index=index)) is False

    def test_grouping_ignores_the_date_order(self):
        """Entities grouped but dates unsorted inside a block: still ``True``."""
        panel = _panel()
        unsorted_within = panel.iloc[[2, 0, 1, 5, 3, 4]]
        assert validate_entities_grouped(unsorted_within) is True

    def test_entity_blocks_may_come_in_any_order(self):
        """Grouping does not require alphabetical order of the blocks."""
        assert validate_entities_grouped(perturb.reverse_entities(_panel())) is True

    @pytest.mark.parametrize('rows', [0, 1], ids=['empty', 'single-row'])
    def test_empty_and_single_row_panels_are_grouped(self, rows):
        """Edge cases: no row, or one row, is trivially grouped."""
        assert validate_entities_grouped(_panel().iloc[:rows]) is True

    def test_single_entity_is_grouped(self):
        """A panel with one entity is grouped."""
        assert validate_entities_grouped(_panel(entities=('A',))) is True

    def test_three_level_panel_is_grouped_by_the_full_entity_key(self):
        """On 3 levels the entity is the ``(region, country)`` pair, date excluded."""
        panel = perturb.to_three_level_index(_panel(entities=('FR', 'DE')))
        assert validate_entities_grouped(panel) is True

    def test_three_level_panel_with_interleaved_countries_is_not_grouped(self):
        """Same region for both countries, but the rows alternate: ``False``."""
        panel = perturb.to_three_level_index(_panel(entities=('FR', 'DE')))
        interleaved = panel.iloc[[0, 3, 1, 4, 2, 5]]
        assert validate_entities_grouped(interleaved) is False

    def test_missing_entity_labels_are_one_group(self):
        """``NaN`` entity labels are treated as one entity: contiguous is ``True``, split is ``False`` (pinned)."""
        dates = _months(4)
        grouped = pd.MultiIndex.from_arrays([['A', 'A', np.nan, np.nan], dates])
        split = pd.MultiIndex.from_arrays([[np.nan, 'A', np.nan, 'A'], dates])
        frame = pd.DataFrame({'v': range(4)})
        assert validate_entities_grouped(frame.set_axis(grouped)) is True
        assert validate_entities_grouped(frame.set_axis(split)) is False

    def test_panel_col_path(self):
        """With one panel column, contiguity is read from that column."""
        flat = _flat(_panel())
        assert validate_entities_grouped(flat, panel_cols=['entity']) is True
        interleaved = flat.iloc[[0, 3, 1, 4, 2, 5]]
        assert validate_entities_grouped(interleaved, panel_cols=['entity']) is False

    def test_several_panel_cols_are_read_as_one_composite_key(self):
        """With two panel columns the entity is the pair of values."""
        flat = _flat(perturb.to_three_level_index(_panel(entities=('FR', 'DE'))))
        assert validate_entities_grouped(flat, panel_cols=['region', 'entity']) is True
        assert validate_entities_grouped(flat.iloc[[0, 3, 1, 4, 2, 5]], panel_cols=['region', 'entity']) is False

    def test_missing_panel_col_is_rejected(self):
        """An unknown panel column raises with its name."""
        with pytest.raises(ValueError, match='Panel columns not found in data'):
            validate_entities_grouped(_flat(_panel()), panel_cols=['nope'])

    def test_ambiguous_panel_col_is_rejected(self):
        """A duplicated panel column label designates two columns: no entity to read."""
        df = pd.DataFrame([['A', 1, 'A']], columns=['entity', 'v', 'entity'])
        with pytest.raises(ValueError, match="Column 'entity' is not unique"):
            validate_entities_grouped(df, panel_cols=['entity'])

    def test_series_with_panel_cols_is_rejected(self):
        """``panel_cols`` cannot be used with a ``Series`` without ``MultiIndex``."""
        s = pd.Series([1, 2], index=_months(2))
        with pytest.raises(ValueError, match='Cannot use panel_cols with Series'):
            validate_entities_grouped(s, panel_cols=['entity'])

    def test_plain_time_series_without_panel_cols_is_rejected(self):
        """Neither ``MultiIndex`` nor ``panel_cols``: nothing to group, ``ValueError``."""
        s = pd.Series([1, 2], index=_months(2))
        with pytest.raises(ValueError, match='MultiIndex or panel_cols must be specified'):
            validate_entities_grouped(s)

    def test_panel_cols_are_ignored_for_a_multiindex(self):
        """For a ``MultiIndex`` the ``panel_cols`` argument is ignored, even if bogus."""
        assert validate_entities_grouped(_panel(), panel_cols=['does_not_exist']) is True

    def test_shuffled_realistic_panel_is_not_grouped(self, heterogeneous_coverage_panel):
        """The notebook 3 panel: grouped as built, not grouped once shuffled."""
        assert validate_entities_grouped(heterogeneous_coverage_panel) is True
        shuffled = perturb.shuffle_rows(heterogeneous_coverage_panel, seed=0)
        assert validate_entities_grouped(shuffled) is False

    def test_reversed_realistic_panel_is_still_grouped(self, heterogeneous_coverage_panel):
        """Reversing the entity blocks keeps each entity contiguous."""
        reversed_panel = perturb.reverse_entities(heterogeneous_coverage_panel)
        assert validate_entities_grouped(reversed_panel) is True


# =============================================================================
# validate_sorted_within_groups
# =============================================================================
class TestValidateSortedWithinGroups:
    """Chronological order of the dates inside each entity."""

    def test_non_pandas_input_is_rejected(self):
        """A non Series / DataFrame input raises ``ValueError``."""
        with pytest.raises(ValueError, match='pandas Series or DataFrame'):
            validate_sorted_within_groups([1, 2, 3])

    @pytest.mark.parametrize('container', ['dataframe', 'series'])
    def test_sorted_panel(self, container):
        """Dates increasing inside each entity give ``True``."""
        panel = _panel()
        assert validate_sorted_within_groups(panel if container == 'dataframe' else panel['v']) is True

    def test_unsorted_dates_inside_an_entity(self):
        """Golden case: entity ``B`` dates ``03, 01, 02`` give ``False``."""
        dates = pd.to_datetime(
            ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-03', '2023-01-01', '2023-01-02']
        )
        index = pd.MultiIndex.from_arrays([['A'] * 3 + ['B'] * 3, dates])
        assert validate_sorted_within_groups(pd.DataFrame({'v': range(6)}, index=index)) is False

    def test_interleaved_entities_with_sorted_dates_are_sorted(self):
        """Order is checked per entity: interleaving the entities does not matter."""
        panel = _panel()
        interleaved = panel.sort_index(level=[1, 0])
        assert validate_entities_grouped(interleaved) is False
        assert validate_sorted_within_groups(interleaved) is True

    def test_equal_dates_inside_an_entity_are_accepted(self):
        """Monotone means non-decreasing: a repeated date is not a disorder (pinned)."""
        dates = pd.to_datetime(['2023-01-01', '2023-01-01', '2023-02-01'])
        index = pd.MultiIndex.from_arrays([['A', 'A', 'A'], dates])
        assert validate_sorted_within_groups(pd.DataFrame({'v': range(3)}, index=index)) is True

    @pytest.mark.parametrize('rows', [0, 1], ids=['empty', 'single-row'])
    def test_empty_and_single_row_panels_are_sorted(self, rows):
        """Edge cases: no row, or one row, is trivially sorted."""
        assert validate_sorted_within_groups(_panel().iloc[:rows]) is True

    def test_three_level_panel_sorted_by_full_entity_key(self):
        """On 3 levels the groups are ``(region, country)``: dates restart per country."""
        panel = perturb.to_three_level_index(_panel(entities=('FR', 'DE')))
        # Deux pays d'une même région : les dates redémarrent au changement de pays,
        # ce qui ne doit pas être pris pour un désordre.
        assert validate_sorted_within_groups(panel) is True

    def test_three_level_panel_detects_disorder_in_one_country(self):
        """Reversing the dates of one country of a 3-level panel gives ``False``."""
        panel = perturb.to_three_level_index(_panel(entities=('FR', 'DE')))
        disordered = panel.iloc[[0, 1, 2, 5, 4, 3]]
        assert validate_sorted_within_groups(disordered) is False

    def test_plain_series_with_sorted_datetime_index(self):
        """Simple time series: ``True`` when the index is increasing."""
        assert validate_sorted_within_groups(pd.Series([1, 2, 3], index=_months(3))) is True

    def test_plain_series_with_unsorted_datetime_index(self):
        """Simple time series: ``False`` when the index is not increasing."""
        s = pd.Series([1, 2, 3], index=_months(3)[[1, 0, 2]])
        assert validate_sorted_within_groups(s) is False

    def test_plain_series_with_period_index(self):
        """A ``PeriodIndex`` is a temporal index too: sorted ``True``, unsorted ``False`` (``ANO-UTILS-029``)."""
        periods = pd.period_range('2023-01', periods=3, freq='M')
        assert validate_sorted_within_groups(pd.Series([1, 2, 3], index=periods)) is True
        assert validate_sorted_within_groups(pd.Series([1, 2, 3], index=periods[[1, 0, 2]])) is False

    def test_panel_with_period_dates(self):
        """Period dates on the last level of a ``MultiIndex`` are compared as periods."""
        periods = pd.period_range('2023-01', periods=3, freq='M')
        index = pd.MultiIndex.from_product([['A', 'B'], periods])
        panel = pd.DataFrame({'v': range(6)}, index=index)
        assert validate_sorted_within_groups(panel) is True
        assert validate_sorted_within_groups(panel.iloc[[1, 0, 2, 3, 4, 5]]) is False

    def test_plain_series_without_temporal_index_is_rejected(self):
        """Only a ``DatetimeIndex`` or ``PeriodIndex`` can be validated."""
        with pytest.raises(ValueError, match='must have DatetimeIndex or PeriodIndex'):
            validate_sorted_within_groups(pd.Series([1, 2, 3], index=pd.RangeIndex(3)))

    def test_panel_col_and_time_col_path(self):
        """Column path: sorted gives ``True``, unsorted dates give ``False``."""
        flat = _flat(_panel())
        assert validate_sorted_within_groups(flat, panel_cols=['entity'], time_col='date') is True
        disordered = flat.iloc[[1, 0, 2, 3, 4, 5]]
        assert validate_sorted_within_groups(disordered, panel_cols=['entity'], time_col='date') is False

    def test_several_panel_cols_on_the_column_path(self):
        """Two panel columns form the group key on the column path."""
        flat = _flat(perturb.to_three_level_index(_panel(entities=('FR', 'DE'))))
        assert validate_sorted_within_groups(flat, panel_cols=['region', 'entity'], time_col='date') is True
        disordered = flat.iloc[[0, 1, 2, 5, 4, 3]]
        assert validate_sorted_within_groups(
            disordered, panel_cols=['region', 'entity'], time_col='date'
        ) is False

    def test_missing_panel_col_is_rejected(self):
        """An unknown panel column raises with its name."""
        with pytest.raises(ValueError, match='Panel columns not found in data'):
            validate_sorted_within_groups(_flat(_panel()), panel_cols=['nope'], time_col='date')

    def test_missing_time_col_is_rejected(self):
        """An unknown time column raises with its name."""
        with pytest.raises(ValueError, match="Time column 'nope' not found in data"):
            validate_sorted_within_groups(_flat(_panel()), panel_cols=['entity'], time_col='nope')

    def test_series_with_panel_and_time_cols_is_rejected(self):
        """Column names make no sense on a ``Series`` without ``MultiIndex``."""
        s = pd.Series([1, 2], index=_months(2))
        with pytest.raises(ValueError, match='Cannot use panel_cols and time_col with Series'):
            validate_sorted_within_groups(s, panel_cols=['a'], time_col='b')

    @pytest.mark.parametrize(
        'kwargs',
        [{'panel_cols': ['entity']}, {'time_col': 'date'}],
        ids=['panel-cols-only', 'time-col-only'],
    )
    def test_incomplete_column_arguments_are_rejected(self, kwargs):
        """One of ``panel_cols`` / ``time_col`` without the other is an error (pinned).

        A single time series stored with a ``time_col`` column (no panel)
        cannot be checked either: the function has no single-series column path.
        """
        with pytest.raises(ValueError, match='both panel_cols and time_col must be specified'):
            validate_sorted_within_groups(_flat(_panel()), **kwargs)

    def test_incomparable_dates_are_not_sorted(self):
        """A group mixing a string and an integer date cannot be ordered: ``False`` (pinned).

        ``is_monotonic_increasing`` answers ``False`` on values that cannot be
        compared; no exception reaches the function.
        """
        df = pd.DataFrame({'entity': ['A', 'A'], 'date': ['2023-01-01', 5]})
        assert validate_sorted_within_groups(df, panel_cols=['entity'], time_col='date') is False

    def test_ambiguous_time_col_is_rejected(self):
        """A duplicated ``time_col`` label designates two columns: ``ValueError``, not ``False``.

        Edge case: column names that are not unique. The check used to fail inside a
        blanket ``except`` and answer ``False`` ("not sorted"), hiding the misuse.
        """
        dates = _months(2)
        df = pd.DataFrame([['A', dates[0], dates[0]], ['A', dates[1], dates[1]]],
                          columns=['entity', 'date', 'date'])
        with pytest.raises(ValueError, match="Column 'date' is not unique"):
            validate_sorted_within_groups(df, panel_cols=['entity'], time_col='date')

    def test_shuffled_realistic_panel_is_not_sorted(self, heterogeneous_coverage_panel):
        """The notebook 3 panel: sorted as built, not sorted once shuffled."""
        assert validate_sorted_within_groups(heterogeneous_coverage_panel) is True
        shuffled = perturb.shuffle_rows(heterogeneous_coverage_panel, seed=0)
        assert validate_sorted_within_groups(shuffled) is False

    def test_realistic_irregular_series_is_sorted(self, irregular_index_timeseries):
        """The irregular notebook 3 series has an increasing ``DatetimeIndex``."""
        assert validate_sorted_within_groups(irregular_index_timeseries) is True


# =============================================================================
# Articulation avec validate_temporal_data : strict True / False et tri
# =============================================================================
def _layouts(panel: pd.DataFrame) -> dict:
    """Build the three unordered layouts of a sorted panel, keyed by readable id.

    Args:
        panel: Sorted two-level panel, at least two entities.

    Returns:
        ``interleaved`` (dates globally ordered, entities alternating: not
        grouped, sorted per entity), ``unsorted_within_entity`` (entities
        contiguous, dates shuffled inside each block: grouped, not sorted) and
        ``shuffled`` (neither).
    """
    within = panel.groupby(level=0, group_keys=False, sort=False).apply(
        lambda block: perturb.shuffle_rows(block, seed=0)
    )
    return {
        'interleaved': panel.sort_index(level=[1, 0]),
        'unsorted_within_entity': within,
        'shuffled': perturb.shuffle_rows(panel, seed=0),
    }


class TestValidationRepairsPanelOrder:
    """``validate_temporal_data`` (sorting) makes both order predicates true.

    ``validate_entities_grouped`` and ``validate_sorted_within_groups`` take no
    ``strict`` argument: they only report. ``strict`` is a property of
    ``validate_temporal_data``, which sorts whatever ``strict`` is; the
    predicates then turn ``True``. Without sorting they stay as they were.
    """

    @pytest.mark.parametrize(
        'layout, grouped, sorted_within',
        [
            ('interleaved', False, True),
            ('unsorted_within_entity', True, False),
            ('shuffled', False, False),
        ],
    )
    def test_layouts_are_reported_as_expected(self, layout, grouped, sorted_within):
        """The three layouts give the golden ``(grouped, sorted)`` pairs."""
        panel = _layouts(_panel(entities=('A', 'B', 'C'), periods=6))[layout]
        assert (validate_entities_grouped(panel), validate_sorted_within_groups(panel)) == (
            grouped,
            sorted_within,
        )

    @pytest.mark.parametrize('layout', ['interleaved', 'unsorted_within_entity', 'shuffled'])
    @pytest.mark.parametrize('strict', [True, False], ids=['strict', 'not-strict'])
    @pytest.mark.parametrize('path', ['index', 'columns'])
    def test_validation_with_sort_repairs_the_order(self, layout, strict, path):
        """After ``validate_temporal_data(sort_data=True)`` both predicates are ``True``."""
        panel = _layouts(_panel(entities=('A', 'B', 'C'), periods=6))[layout]
        data, kwargs = _validate_kwargs(panel, path)

        out = validate_temporal_data(data, strict=strict, sort_data=True, **kwargs)

        assert validate_entities_grouped(out) is True
        assert validate_sorted_within_groups(out) is True

    @pytest.mark.parametrize('layout', ['interleaved', 'unsorted_within_entity', 'shuffled'])
    def test_validation_without_sort_leaves_the_order_as_it_was(self, layout):
        """With ``sort_data=False`` the reported order is unchanged."""
        panel = _layouts(_panel(entities=('A', 'B', 'C'), periods=6))[layout]
        before = (validate_entities_grouped(panel), validate_sorted_within_groups(panel))

        out = validate_temporal_data(panel, sort_data=False)

        assert (validate_entities_grouped(out), validate_sorted_within_groups(out)) == before

    def test_shuffled_realistic_panel_is_repaired(self, heterogeneous_coverage_panel):
        """The shuffled notebook 3 panel becomes grouped and sorted after validation."""
        shuffled = perturb.shuffle_rows(heterogeneous_coverage_panel, seed=3)
        out = validate_temporal_data(shuffled)
        assert validate_entities_grouped(out) is True
        assert validate_sorted_within_groups(out) is True

    def test_column_path_predicates_on_the_flat_realistic_panel(self, heterogeneous_coverage_panel):
        """The predicates accept the flat notebook 3 panel through ``panel_cols`` / ``time_col``."""
        flat = perturb.shuffle_rows(heterogeneous_coverage_panel, seed=4).reset_index()
        assert validate_entities_grouped(flat, panel_cols=['country']) is False
        assert validate_sorted_within_groups(flat, panel_cols=['country'], time_col='date') is False
        repaired = validate_temporal_data(flat, time_col='date', panel_cols=['country']).reset_index()
        assert validate_entities_grouped(repaired, panel_cols=['country']) is True
        assert validate_sorted_within_groups(repaired, panel_cols=['country'], time_col='date') is True
