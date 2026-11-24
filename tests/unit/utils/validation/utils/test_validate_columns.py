"""Column-based ``validate_temporal_data`` and its metadata.

Scenarios: ``time_col`` with optional ``panel_cols`` and the metadata returned by
``return_metadata=True``.
"""

# Modules de base
import re

import pandas as pd
import pytest

# Jeux de données et perturbations partagés
from tests.support import perturbations as perturb

# Fonctions à tester
from tsforecast.utils.validation import validate_temporal_data

# Outils de test partagés du paquet
from tests.unit.utils.validation.utils.helpers import _flat, _months, _panel


# Le chemin « colonnes » émet systématiquement un avertissement de remplacement
# d'index (comportement testé explicitement dans TestColumnBasedValidation) ; le
# bruit est masqué ailleurs pour garder une sortie lisible. L'inférence de format
# de pandas sur des chaînes de dates non ISO émet aussi un avertissement.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Index replaced with"),
    pytest.mark.filterwarnings("ignore:Could not infer format"),
]


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
        built.
        """
        df = pd.DataFrame({'date': ['a', 'b'], 'v': [1, 2]})
        with pytest.raises(ValueError, match="Column 'date' cannot be converted to datetime"):
            validate_temporal_data(df, time_col='date', strict=strict)

    def test_period_time_col_is_converted_to_first_instants(self):
        """A ``Period`` time column is read at the start of each period."""
        df = pd.DataFrame({'date': pd.period_range('2023-01', periods=2, freq='M'), 'v': [1, 2]})
        out = validate_temporal_data(df, time_col='date')
        assert list(out.index) == list(_months(2))

    @pytest.mark.parametrize('labels', [[2020, 2021], [2020.0, 2021.0]], ids=['integers', 'floats'])
    @pytest.mark.parametrize('strict', [True, False], ids=['strict', 'not-strict'])
    def test_numeric_time_col_is_rejected(self, labels, strict):
        """A numeric time column is not a date and raises in both modes."""
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
