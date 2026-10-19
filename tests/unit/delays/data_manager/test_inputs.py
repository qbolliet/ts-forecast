"""Inputs and reports of ``compare_and_detect_delays``.

Scenarios: index kinds and column layouts, empty and degenerate inputs, emitted warnings
and the ``DelayDetectionReport`` returned by ``return_report=True``.
"""

# Modules de base
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# Jeux de données et perturbations partagés
from tests.support import perturbations as perturb

# Fonction à tester
from tsforecast.delays.data_manager import compare_and_detect_delays

# Outils de test partagés du paquet
from tests.unit.delays.data_manager.helpers import (
    OUTPUT_COLUMNS,
    TS,
    _expected,
    _months,
    _records,
    _series,
)


# Le chemin « colonnes » (time_col / panel_cols) émet systématiquement un
# avertissement de remplacement d'index (comportement de validate_temporal_data) :
# le bruit est masqué pour garder une sortie lisible.
pytestmark = pytest.mark.filterwarnings("ignore:Index replaced with")


# =============================================================================
# Présentation des données en entrée
# =============================================================================
class TestInputLayout:
    """Index kinds, column-based layouts, disorder and unusual names."""

    def test_time_col_gives_the_same_result_as_the_index(self, monthly_series):
        """Passing the dates in a column (``time_col``) is equivalent to passing them as the index."""
        reference = compare_and_detect_delays(monthly_series, None, '2023-06-15')

        result = compare_and_detect_delays(
            monthly_series.rename_axis('date').reset_index(), None, '2023-06-15', time_col='date',
        )

        pd.testing.assert_frame_equal(result, reference)

    def test_time_col_applies_to_both_datasets(self):
        """``time_col`` is used to read ``new_data`` and ``existing_data`` alike."""
        existing = _series([1.0, 2.0, 3.0, np.nan]).rename_axis('date').reset_index()
        new_data = _series([1.0, 2.0, 3.0, 4.0]).rename_axis('date').reset_index()

        result = compare_and_detect_delays(new_data, existing, '2023-06-15', time_col='date')

        assert _records(result, 'observation_date', 'delay') == _expected(('PIB', TS('2023-04-01'), 75.0))

    def test_unknown_time_col_raises(self, monthly_series):
        """A ``time_col`` that is not a column is refused."""
        with pytest.raises(ValueError, match="Time column 'absent' not found"):
            compare_and_detect_delays(monthly_series.reset_index(), None, '2023-06-15', time_col='absent')

    def test_panel_cols_give_the_same_result_as_a_multiindex(self, small_panel):
        """Entity and date columns (``panel_cols`` + ``time_col``) are equivalent to a two-level index."""
        reference = compare_and_detect_delays(small_panel, None, '2023-06-15')

        result = compare_and_detect_delays(
            small_panel.reset_index(), None, '2023-06-15', time_col='date', panel_cols=['country'],
        )

        pd.testing.assert_frame_equal(result, reference)

    def test_panel_cols_with_existing_data(self, small_panel):
        """The column-based layout also works when comparing two panels."""
        existing = small_panel.copy()
        existing.iloc[3, 0] = np.nan  # (A, avril) retiré
        reference = compare_and_detect_delays(small_panel, existing, '2023-06-15')

        result = compare_and_detect_delays(
            small_panel.reset_index(), existing.reset_index(), '2023-06-15', time_col='date', panel_cols=['country'],
        )

        pd.testing.assert_frame_equal(result, reference)

    def test_string_dates_in_the_index_are_converted(self, monthly_series):
        """An index of ISO date strings is converted to dates."""
        reference = compare_and_detect_delays(monthly_series, None, '2023-06-15')
        data = monthly_series.copy()
        data.index = data.index.strftime('%Y-%m-%d')

        result = compare_and_detect_delays(data, None, '2023-06-15')

        pd.testing.assert_frame_equal(result, reference)

    def test_unsorted_time_series_is_sorted(self):
        """A shuffled series gives the result of the sorted one (the last date is the latest, not the last row)."""
        sorted_data = _series([1.0, 2.0, 3.0, 4.0])

        result = compare_and_detect_delays(perturb.shuffle_rows(sorted_data, seed=3), None, '2023-06-15')

        pd.testing.assert_frame_equal(result, compare_and_detect_delays(sorted_data, None, '2023-06-15'))

    def test_unsorted_panel_is_sorted(self, small_panel):
        """A shuffled panel gives the result of the sorted one."""
        result = compare_and_detect_delays(perturb.shuffle_rows(small_panel, seed=3), None, '2023-06-15')

        pd.testing.assert_frame_equal(result, compare_and_detect_delays(small_panel, None, '2023-06-15'))

    def test_unsorted_panel_with_existing_data(self, small_panel):
        """Disorder in both datasets does not change what is detected."""
        existing = small_panel.copy()
        existing.iloc[3, 0] = np.nan  # (A, avril) retiré
        reference = compare_and_detect_delays(small_panel, existing, '2023-06-15')

        result = compare_and_detect_delays(
            perturb.shuffle_rows(small_panel, seed=1), perturb.shuffle_rows(existing, seed=2), '2023-06-15',
        )

        pd.testing.assert_frame_equal(result.sort_index(), reference.sort_index())

    def test_entities_in_reverse_order(self, small_panel):
        """Entity blocks in reverse order give the same set of rows."""
        reference = compare_and_detect_delays(small_panel, None, '2023-06-15')

        result = compare_and_detect_delays(perturb.reverse_entities(small_panel), None, '2023-06-15')

        pd.testing.assert_frame_equal(result.sort_index(), reference.sort_index())

    def test_non_standard_index_names_are_preserved(self, small_panel):
        """Entity level names are not assumed: the result index reuses the input's entity name."""
        data = perturb.with_index_names(small_panel, ['pays', 'jour'])

        result = compare_and_detect_delays(data, None, '2023-06-15')

        assert list(result.index.names) == ['pays', 'column']

    def test_unnamed_time_index(self, monthly_series):
        """A time index without name works (the dates leave the index under ``observation_date``)."""
        data = perturb.with_index_names(monthly_series, None)

        result = compare_and_detect_delays(data, None, '2023-06-15')

        assert _records(result, 'observation_date', 'delay') == _expected(('PIB', TS('2023-04-01'), 75.0))

    def test_three_level_panel(self, small_panel):
        """Every entity level before the dates stays in the result index."""
        data = perturb.to_three_level_index(small_panel, {'A': 'Nord', 'B': 'Sud'})

        result = compare_and_detect_delays(data, None, '2023-06-15')

        assert _records(result, 'observation_date', 'delay') == _expected(
            ('Nord', 'A', 'PIB', TS('2023-04-01'), 75.0), ('Sud', 'B', 'PIB', TS('2023-03-01'), 106.0),
        )
        assert list(result.index.names) == ['region', 'country', 'column']

    def test_three_level_panel_with_existing_data(self, small_panel):
        """Three-level panels are also compared level by level."""
        data = perturb.to_three_level_index(small_panel, {'A': 'Nord', 'B': 'Sud'})
        existing = data.copy()
        existing.iloc[3, 0] = np.nan  # (Nord, A, avril) retiré

        result = compare_and_detect_delays(data, existing, '2023-06-15')

        assert _records(result, 'observation_date', 'delay') == _expected(
            ('Nord', 'A', 'PIB', TS('2023-04-01'), 75.0),
        )

    def test_period_index_is_read_at_the_first_instant(self, monthly_series):
        """A ``PeriodIndex`` is converted to the first instant of each period."""
        data = monthly_series.copy()
        data.index = data.index.to_period('M')

        result = compare_and_detect_delays(data, None, '2023-06-15')

        pd.testing.assert_frame_equal(result, compare_and_detect_delays(monthly_series, None, '2023-06-15'))

    def test_period_index_panel(self, small_panel):
        """A panel whose date level is a ``PeriodIndex`` behaves like its ``DatetimeIndex`` twin."""
        result = compare_and_detect_delays(perturb.to_period_index(small_panel), None, '2023-06-15')

        pd.testing.assert_frame_equal(result, compare_and_detect_delays(small_panel, None, '2023-06-15'))

    def test_special_column_names_are_kept_verbatim(self):
        """Spaces, accents, ``%`` and ``/`` in column names reach the ``column`` index untouched."""
        data, mapping = perturb.with_special_column_names(
            pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [5.0, 6.0, 7.0, 8.0]}, index=_months())
        )

        result = compare_and_detect_delays(data, None, '2023-06-15')

        assert sorted(result.index) == sorted(mapping.values())

    def test_special_column_names_with_existing_data(self):
        """The comparison path also keeps special names intact."""
        data, mapping = perturb.with_special_column_names(
            pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [5.0, 6.0, 7.0, 8.0]}, index=_months())
        )
        existing = data.copy()
        existing.iloc[3] = np.nan

        result = compare_and_detect_delays(data, existing, '2023-06-15')

        assert sorted(result.index) == sorted(mapping.values())

    @pytest.mark.parametrize(
        'name',
        ['column', 'delay', 'frequency', 'observation_date', 'unit', 'download_date', 'period_start', 'reference_point', 'index'],
    )
    @pytest.mark.parametrize('with_existing_data', [False, True], ids=['first-download', 'comparison'])
    def test_column_named_like_an_output_column(self, name, with_existing_data):
        """A data column sharing the name of an output column neither collides nor disappears."""
        data = pd.DataFrame({name: [1.0, 2.0, 3.0, 4.0], 'x': [5.0, 6.0, 7.0, 8.0]}, index=_months())
        existing = None
        if with_existing_data:
            existing = data.copy()
            existing.iloc[3] = np.nan

        result = compare_and_detect_delays(data, existing, '2023-06-15')

        assert sorted(result.index) == sorted([name, 'x'])

    def test_column_named_has_changes_in_comparison(self):
        """A data column called ``has_changes`` is compared like any other."""
        data = pd.DataFrame({'has_changes': [1.0, 2.0, 3.0, 4.0], 'x': [5.0, 6.0, 7.0, 8.0]}, index=_months())
        existing = data.copy()
        existing.iloc[3] = np.nan

        result = compare_and_detect_delays(data, existing, '2023-06-15')

        assert sorted(result.index) == ['has_changes', 'x']

    def test_column_named_has_changes_on_first_download(self):
        """On a first download the same column name is harmless."""
        data = pd.DataFrame({'has_changes': [1.0, 2.0, 3.0, 4.0]}, index=_months())

        result = compare_and_detect_delays(data, None, '2023-06-15')

        assert list(result.index) == ['has_changes']

    def test_duplicated_dates_raise(self):
        """A repeated date cannot be placed on a period: refused by the validation."""
        data = perturb.with_duplicated_rows(_series([1.0, 2.0, 3.0, 4.0]))

        with pytest.raises(ValueError, match="duplicate"):
            compare_and_detect_delays(data, None, '2023-06-15')

    def test_duplicated_panel_rows_raise(self, small_panel):
        """A repeated (entity, date) pair is refused."""
        with pytest.raises(ValueError, match="duplicate"):
            compare_and_detect_delays(perturb.with_duplicated_rows(small_panel), None, '2023-06-15')

    def test_duplicated_dates_in_existing_data_raise(self, monthly_series):
        """The validation also applies to ``existing_data``."""
        with pytest.raises(ValueError, match="duplicate"):
            compare_and_detect_delays(monthly_series, perturb.with_duplicated_rows(monthly_series), '2023-06-15')


# =============================================================================
# Jeux vides ou dégénérés
# =============================================================================
class TestEmptyAndDegenerateInputs:
    """Nothing to report, nothing observed, nothing at all."""

    def test_no_change_gives_an_empty_frame_with_the_output_schema(self, monthly_series):
        """Identical downloads give zero rows but the full set of output columns."""
        result = compare_and_detect_delays(monthly_series, monthly_series.copy(), '2023-06-15')

        assert result.empty and list(result.columns) == OUTPUT_COLUMNS

    def test_no_change_in_a_panel_keeps_the_index_levels(self, small_panel):
        """An empty panel result still has the (entity, ``column``) index levels."""
        result = compare_and_detect_delays(small_panel, small_panel.copy(), '2023-06-15')

        assert result.empty and list(result.index.names) == ['country', 'column']

    def test_all_null_data_in_comparison_gives_an_empty_frame(self):
        """With ``existing_data`` given, a dataset without any observation reports nothing."""
        data = _series([np.nan] * 4)

        result = compare_and_detect_delays(data, data.copy(), '2023-06-15')

        assert result.empty

    def test_all_null_data_on_first_download_gives_an_empty_frame(self):
        """Without ``existing_data``, a series with rows but no observation reports nothing (as the comparison path does)."""
        result = compare_and_detect_delays(_series([np.nan] * 5), None, '2023-06-15')

        assert result.empty and list(result.columns) == OUTPUT_COLUMNS

    def test_all_null_panel_on_first_download_gives_an_empty_frame(self, small_panel):
        """Same expectation for a panel without any observation."""
        result = compare_and_detect_delays(small_panel * np.nan, None, '2023-06-15')

        assert result.empty and list(result.index.names) == ['country', 'column']

    def test_dataset_without_rows_raises_a_clear_error(self, monthly_series):
        """A dataset with no row at all is refused with an explicit ``ValueError``."""
        with pytest.raises(ValueError, match="empty"):
            compare_and_detect_delays(perturb.empty_like(monthly_series), None, '2023-06-15')

    def test_panel_without_rows_raises_a_clear_error(self, small_panel):
        """A panel with no row at all is refused with an explicit ``ValueError``."""
        with pytest.raises(ValueError, match="empty"):
            compare_and_detect_delays(perturb.empty_like(small_panel), None, '2023-06-15')

    @pytest.mark.parametrize('with_existing_data', [False, True], ids=['first-download', 'comparison'])
    def test_frame_without_columns_gives_an_empty_frame(self, with_existing_data):
        """Dates but no variable: nothing can be observed, hence an empty result."""
        data = pd.DataFrame(index=_months())

        result = compare_and_detect_delays(data, data.copy() if with_existing_data else None, '2023-06-15')

        assert result.empty

    def test_empty_results_of_every_path_are_identical(self, monthly_series):
        """'Nothing changed', 'nothing observed' and 'no column' give the very same empty frame."""
        unchanged = compare_and_detect_delays(monthly_series, monthly_series.copy(), '2023-06-15')
        unobserved = compare_and_detect_delays(_series([np.nan] * 4), None, '2023-06-15')
        no_column = compare_and_detect_delays(pd.DataFrame(index=_months()), None, '2023-06-15')

        pd.testing.assert_frame_equal(unobserved, unchanged)
        pd.testing.assert_frame_equal(no_column, unchanged)

    def test_empty_new_data_with_existing_data_raises(self, monthly_series):
        """An empty new download is refused even when there is something to compare it with."""
        with pytest.raises(ValueError, match="empty"):
            compare_and_detect_delays(perturb.empty_like(monthly_series), monthly_series, '2023-06-15')

    def test_empty_existing_data_makes_every_observation_new(self, monthly_series):
        """An existing download without any row is 'nothing known yet': all observed values are new."""
        result = compare_and_detect_delays(monthly_series, perturb.empty_like(monthly_series), '2023-06-15')

        assert len(result) == 4

    @pytest.mark.parametrize('argument', ['new_data', 'existing_data'])
    def test_series_instead_of_frame_raises(self, monthly_series, argument):
        """Only DataFrames are accepted: a Series is refused with an explicit ``TypeError``."""
        arguments = {'new_data': monthly_series, 'existing_data': monthly_series}
        arguments[argument] = monthly_series['PIB']

        with pytest.raises(TypeError, match=f"{argument} must be a pandas DataFrame"):
            compare_and_detect_delays(**arguments, download_date='2023-06-15')

    def test_two_observations_are_enough(self):
        """The minimal usable series has two observations (one step gives a frequency)."""
        result = compare_and_detect_delays(_series([1.0, 2.0]), None, '2023-04-15')

        # Valeur d'or : dernière observation = février, période [1er fév., 1er mars) -> 15 avril :
        # 28 + 31 + 14 = 73 j depuis le début
        assert _records(result, 'frequency', 'delay') == _expected(('PIB', 'monthly', 73.0))

    def test_delay_from_end_is_exact_for_two_observations(self):
        """Gold value of the former 'precision' test, now exact (44 days after the end of February)."""
        result = compare_and_detect_delays(_series([100.0, 101.0]), None, datetime(2023, 4, 14), reference_point='end')

        # Valeur d'or : fin de la période de février = 1er mars (exclusif) -> 14 avril = 31 + 13 = 44 j
        assert result['delay'].iloc[0] == 44.0

    def test_old_data_gives_a_large_delay(self):
        """Thirty-three years of delay are computed in days without overflow."""
        data = pd.DataFrame({'PIB': [100.0, 101.0, 102.0]}, index=pd.date_range('1990-01-01', periods=3, freq='MS'))

        result = compare_and_detect_delays(data, None, '2023-12-15')

        # Valeur d'or : 1er mars 1990 -> 15 déc. 2023 = 33 ans (33 * 365 + 8 jours bissextiles) = 12 053 j
        # jusqu'au 1er mars 2023, puis 275 j jusqu'au 1er déc. et 14 j : 12 342 j
        assert result['delay'].iloc[0] == 12342.0


# =============================================================================
# Avertissements
# =============================================================================
class TestWarnings:
    """Warnings emitted by a call."""

    def test_time_series_call_is_silent(self, monthly_series):
        """A plain time-series call emits no warning at all."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            compare_and_detect_delays(monthly_series, None, '2023-06-15')

    def test_comparison_call_is_silent(self, monthly_series):
        """A time-series comparison emits no warning at all."""
        existing = monthly_series.copy()
        existing.iloc[3, 0] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            compare_and_detect_delays(monthly_series, existing, '2023-06-15')

    @pytest.mark.parametrize('with_existing_data', [False, True], ids=['first-download', 'comparison'])
    def test_panel_call_is_silent(self, small_panel, with_existing_data):
        """A two-level panel call emits no warning, pandas deprecation warnings included."""
        existing = None
        if with_existing_data:
            existing = small_panel.copy()
            existing.iloc[3, 0] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            compare_and_detect_delays(small_panel, existing, '2023-06-15')


# =============================================================================
# Rapport de détection (return_report=True)
# =============================================================================
class TestDetectionReport:
    """``return_report=True`` returns a ``DelayDetectionReport`` next to the frame."""

    EXISTING = pd.DataFrame({'PIB': [1.0, 2.0, 3.0, np.nan], 'OLD': [1.0] * 4}, index=_months(4))
    NEW = pd.DataFrame({'PIB': [1.0, 2.5, 3.0, 4.0], 'NEW': [1.0] * 4}, index=_months(4))

    def test_default_returns_the_frame_only(self):
        result = compare_and_detect_delays(self.NEW, self.EXISTING, download_date='2023-06-15')
        assert isinstance(result, pd.DataFrame)

    def test_frame_is_the_same_with_and_without_report(self):
        plain = compare_and_detect_delays(self.NEW, self.EXISTING, '2023-06-15', detection_mode='all_changes')
        frame, _ = compare_and_detect_delays(
            self.NEW, self.EXISTING, '2023-06-15', detection_mode='all_changes', return_report=True)
        pd.testing.assert_frame_equal(plain, frame)

    def test_return_report_is_keyword_only(self):
        with pytest.raises(TypeError):
            compare_and_detect_delays(  # type: ignore[call-overload]
                self.NEW, self.EXISTING, '2023-06-15', 'new_only', 'start', 'day', None, None, True)

    def test_new_only_counts(self):
        _, report = compare_and_detect_delays(self.NEW, self.EXISTING, '2023-06-15', return_report=True)
        assert (report.n_detected, report.n_new_values, report.n_revisions) == (1, 1, 0)
        assert report.n_vanished_values == 0
        assert report.n_detected_by_column == {'PIB': 1}
        assert report.columns_without_detection == ()

    def test_all_changes_counts_the_revision_with_gold_delays(self):
        frame, report = compare_and_detect_delays(
            self.NEW, self.EXISTING, '2023-06-15', detection_mode='all_changes', return_report=True)
        # Avril publié (75 j depuis le 1er avril) et février révisé (134 j depuis le 1er février)
        assert sorted(frame['delay']) == [75.0, 134.0]
        assert (report.n_detected, report.n_new_values, report.n_revisions) == (2, 1, 1)
        assert (report.delay_min, report.delay_max, report.delay_mean, report.delay_median) == (75.0, 134.0, 104.5, 104.5)
        assert (report.n_known, report.n_negative) == (2, 0)
        assert report.delay_unit == 'day'

    def test_compared_and_ignored_columns(self):
        _, report = compare_and_detect_delays(self.NEW, self.EXISTING, '2023-06-15', return_report=True)
        assert report.columns_compared == ('PIB',)
        assert report.columns_new_only == ('NEW',)
        assert report.columns_existing_only == ('OLD',)
        assert (report.n_rows_new, report.n_rows_existing, report.n_columns, report.n_entities) == (4, 4, 2, 0)

    def test_vanished_values_are_counted_and_not_detected(self):
        existing = _series([1.0, 2.0, 3.0, 4.0])
        new = _series([1.0, 2.0, 3.0, np.nan])
        frame, report = compare_and_detect_delays(
            new, existing, '2023-06-15', detection_mode='all_changes', return_report=True)
        assert frame.empty
        assert report.n_vanished_values == 1
        assert report.n_detected == 0
        assert report.columns_without_detection == ('PIB',)
        assert (report.n_known, report.delay_min, report.delay_median) == (0, None, None)

    def test_without_existing_data_the_latest_observation_is_a_new_value(self):
        _, report = compare_and_detect_delays(self.NEW, download_date='2023-06-15', return_report=True)
        assert report.has_existing_data is False
        assert report.n_rows_existing is None
        assert (report.n_detected, report.n_new_values, report.n_revisions) == (2, 2, 0)

    def test_panel_counts_and_frequencies_per_entity(self):
        index = pd.MultiIndex.from_product([['FR', 'DE'], _months(4)], names=['country', 'date'])
        existing = pd.DataFrame({'PIB': [1.0, 2.0, 3.0, np.nan] * 2}, index=index)
        new = pd.DataFrame({'PIB': [1.0, 2.0, 3.0, 4.0] * 2}, index=index)
        _, report = compare_and_detect_delays(new, existing, '2023-06-15', return_report=True)
        assert report.n_entities == 2
        assert report.n_detected == 2
        assert report.frequencies == {('DE', 'PIB'): 'monthly', ('FR', 'PIB'): 'monthly'}
        assert report.undetected_keys == ()

    def test_undetectable_frequency_is_reported(self):
        new = _series([1.0])
        with pytest.warns(UserWarning, match="frequency could not be detected"):
            frame, report = compare_and_detect_delays(new, download_date='2023-06-15', return_report=True)
        assert report.undetected_keys == ('PIB',)
        assert report.frequencies == {'PIB': None}
        assert (report.n_detected, report.n_known) == (1, 0)

    def test_negative_delays_are_counted(self):
        _, report = compare_and_detect_delays(self.NEW, download_date='2023-01-10', return_report=True)
        assert report.n_negative == report.n_known

    def test_report_is_logged_at_info_level(self, caplog):
        with caplog.at_level('INFO', logger='tsforecast.delays.data_manager'):
            _, report = compare_and_detect_delays(self.NEW, self.EXISTING, '2023-06-15', return_report=True)
        assert report.summary() in caplog.messages

    def test_unit_label_follows_the_unit(self):
        _, report = compare_and_detect_delays(
            self.NEW, self.EXISTING, '2023-06-15', delay_unit='us', return_report=True)
        assert report.delay_unit == 'microsecond'
