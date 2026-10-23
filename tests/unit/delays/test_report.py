"""Unit tests for ``tsforecast.delays.report``.

Covers the rendering of the two reports (``summary``, ``to_dict``, ``to_frame`` and the
``str`` dtype of the text in its frames),
their immutability and the ``delay_statistics`` helper. The reports are built
through the public API of the components (``compare_and_detect_delays`` and
``PublicationDelayTransformer.fit``); what they contain is checked in
the ``data_manager`` and ``transformers`` test packages.
"""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from tsforecast.delays import (
    ColumnDelayRecord, DelayDetectionReport, DelayFitReport,
    PublicationDelayTransformer, compare_and_detect_delays,
)
from tsforecast.delays.report import delay_statistics


@pytest.fixture
def detection_report() -> DelayDetectionReport:
    """Report of a panel detection with revisions."""
    index = pd.MultiIndex.from_product([['FR', 'DE'], pd.date_range('2023-01-01', periods=4, freq='MS')],
                                       names=['country', 'date'])
    existing = pd.DataFrame({'PIB': [1.0, 2.0, 3.0, np.nan] * 2}, index=index)
    new = pd.DataFrame({'PIB': [1.0, 2.5, 3.0, 4.0] * 2}, index=index)
    _, report = compare_and_detect_delays(new, existing, '2023-06-15', detection_mode='all_changes', return_report=True)
    return report


@pytest.fixture
def fit_report() -> DelayFitReport:
    """Fit report with one shifted column, one unaffected column and one ignored column."""
    X = pd.DataFrame({'GDP': range(6), 'Z': range(6)}, index=pd.date_range('2023-01-01', periods=6, freq='MS'))
    delays = pd.DataFrame({'column': ['GDP', 'OLD'], 'delay': [45.0, 5.0], 'unit': ['D', 'D'],
                           'reference_point': ['end', 'end'], 'frequency': ['M', 'M']})
    with pytest.warns(UserWarning):
        transformer = PublicationDelayTransformer(delays=delays, prediction_date='2023-07-15').fit(X)
    return transformer.fit_report_


class TestDelayStatistics:
    def test_gold_values(self):
        stats = delay_statistics(pd.Series([10.0, -2.0, np.nan, 30.0]))
        assert stats == {'n_known': 3, 'n_negative': 1, 'min': -2.0, 'max': 30.0,
                         'mean': 38.0 / 3, 'median': 10.0}

    @pytest.mark.parametrize('values', [[], [np.nan, np.nan]])
    def test_no_known_delay(self, values):
        stats = delay_statistics(pd.Series(values, dtype=float))
        assert (stats['n_known'], stats['n_negative']) == (0, 0)
        assert [stats[k] for k in ('min', 'max', 'mean', 'median')] == [None] * 4

    def test_non_numeric_values_are_ignored(self):
        assert delay_statistics(pd.Series([1.0, 'x', None], dtype=object))['n_known'] == 1


class TestDetectionReportRendering:
    def test_is_immutable(self, detection_report):
        with pytest.raises(dataclasses.FrozenInstanceError):
            detection_report.n_detected = 0  # type: ignore[misc]

    def test_summary_is_one_line_and_names_the_counts(self, detection_report):
        summary = detection_report.summary()
        assert '\n' not in summary
        assert '4 observations detected (2 new, 2 revised)' in summary
        assert str(detection_report) == summary

    def test_to_dict_is_json_serializable_with_string_keys(self, detection_report):
        payload = detection_report.to_dict()
        json.dumps(payload, allow_nan=False)
        assert payload['frequencies'] == {'DE/PIB': 'monthly', 'FR/PIB': 'monthly'}
        assert payload['download_date'] == '2023-06-15T00:00:00'
        assert payload['columns_compared'] == ['PIB']

    def test_to_frame_has_one_row_per_column(self, detection_report):
        frame = detection_report.to_frame()
        assert frame.index.name == 'column'
        assert frame.loc['PIB'].to_dict() == {'n_detected': 4, 'compared': True}


class TestFitReportRendering:
    def test_is_immutable(self, fit_report):
        with pytest.raises(dataclasses.FrozenInstanceError):
            fit_report.columns = ()  # type: ignore[misc]

    def test_summary_is_one_line(self, fit_report):
        summary = fit_report.summary()
        assert '\n' not in summary
        assert '1 shifted, 0 masked, 1 unaffected, 1 ignored, 0 defaults imputed, 0 mask fallbacks' in summary

    def test_to_dict_is_json_serializable(self, fit_report):
        payload = fit_report.to_dict()
        json.dumps(payload, allow_nan=False)
        assert payload['columns'][0]['column'] == 'GDP'
        assert payload['columns'][0]['delay_unit_source'] == 'inferred'

    def test_to_frame_has_one_row_per_delayed_column(self, fit_report):
        frame = fit_report.to_frame()
        assert list(frame.index) == ['GDP']
        assert frame.loc['GDP', 'n_periods'] == -3
        assert 'column' not in frame.columns

    def test_to_frame_of_a_report_without_column_is_empty(self, fit_report):
        assert isinstance(fit_report.columns[0], ColumnDelayRecord)
        assert dataclasses.replace(fit_report, columns=()).to_frame().empty


class TestStringDtypes:
    """The frames of the reports hold their text in the ``str`` dtype, whatever their content."""

    # Champs textuels d'un ColumnDelayRecord
    TEXT_FIELDS = ('strategy', 'delay_unit', 'reference_point', 'frequency', 'target_frequency',
                   'delay_unit_source', 'reference_point_source', 'target_frequency_source', 'delay_source')

    def test_detection_frame_index_is_str(self, detection_report):
        assert detection_report.to_frame().index.dtype == 'str'

    def test_fit_frame_text_columns_are_str(self, fit_report):
        frame = fit_report.to_frame()

        assert frame.index.dtype == 'str'
        assert {column: str(frame[column].dtype) for column in self.TEXT_FIELDS} == dict.fromkeys(self.TEXT_FIELDS, 'str')

    def test_fit_frame_all_none_field_is_missing_in_str(self, fit_report):
        """A shift has no target frequency: an all-None field is a ``str`` column of NaN."""
        assert fit_report.columns[0].target_frequency is None
        column = fit_report.to_frame()['target_frequency']

        assert column.dtype == 'str' and column.isna().all()
        # Le dictionnaire, hors pandas, garde None
        assert fit_report.to_dict()['columns'][0]['target_frequency'] is None

    def test_empty_fit_frame_text_columns_are_str(self, fit_report):
        frame = dataclasses.replace(fit_report, columns=()).to_frame()

        assert all(frame[column].dtype == 'str' for column in self.TEXT_FIELDS)
