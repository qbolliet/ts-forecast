"""String dtypes of the outputs of ``compare_and_detect_delays`` and of its detection report.

The text columns of the result (``frequency``, ``reference_point``, ``unit``) are
always in the ``str`` dtype, an undetectable frequency being missing (NaN); the
``column`` level holds the labels with the dtype pandas infers from them and the
entity levels keep the dtype of the data. An empty result (no observation, or no
change since a snapshot) has exactly the dtypes of a non-empty one, for string,
``object`` and integer labels. ``DelayDetectionReport.to_frame`` indexes its rows
by the labels in the same way.
"""
import numpy as np
import pandas as pd
import pytest

# Outils de test partagés
from tests.support.perturbations import string_dtypes

# Fonction à tester
from tsforecast.delays.data_manager import compare_and_detect_delays

# Outils de test partagés du paquet
from tests.unit.delays.data_manager.helpers import _panel, _series

# Colonnes textuelles du résultat
TEXT_COLUMNS = {('column', 'frequency'): 'str', ('column', 'reference_point'): 'str', ('column', 'unit'): 'str'}


def _labelled(structure: str, label_dtype) -> pd.DataFrame:
    """Build a two-variable series or panel whose labels are stored in ``label_dtype``."""
    if structure == 'series':
        data = _series([1.0, 2.0, 3.0, 4.0]).assign(CPI=[1.0, 2.0, 3.0, np.nan])
    else:
        data = _panel({
            'FR': {'PIB': [1.0, 2.0, 3.0, 4.0], 'CPI': [1.0, 2.0, 3.0, np.nan]},
            'DE': {'PIB': [1.0, 2.0, 3.0, 4.0], 'CPI': [1.0, 2.0, np.nan, np.nan]},
        })
        data.index = data.index.set_levels(data.index.levels[0].astype(label_dtype), level=0)
    data.columns = data.columns.astype(label_dtype)
    return data


def _level_dtypes(result: pd.DataFrame) -> list:
    """Return the dtype of each index level of a result."""
    return [result.index.get_level_values(level).dtype for level in range(result.index.nlevels)]


class TestNonEmptyResult:
    """The text columns of a result are in the ``str`` dtype."""

    def test_series_result(self):
        """Series: text columns and ``column`` level in ``str``."""
        result = compare_and_detect_delays(_series([1.0, 2.0, 3.0, 4.0]), None, '2023-06-15')

        assert string_dtypes(result) == {('index', 0): 'str', **TEXT_COLUMNS}

    @pytest.mark.parametrize('label_dtype', ['str', object], ids=['str_labels', 'object_labels'])
    def test_panel_result_keeps_the_entity_dtype(self, label_dtype):
        """Panel: the entity level keeps the dtype of the data; the ``column`` level is inferred (``str``)."""
        data = _labelled('panel', label_dtype)

        result = compare_and_detect_delays(data, None, '2023-06-15')

        expected_entity = 'str' if label_dtype == 'str' else 'object'
        assert string_dtypes(result) == {('index', 0): expected_entity, ('index', 1): 'str', **TEXT_COLUMNS}

    def test_undetectable_frequency_is_missing_in_the_str_dtype(self):
        """A single observation: the frequency column stays ``str``, the value is NaN (not None)."""
        with pytest.warns(UserWarning, match='could not be detected'):
            result = compare_and_detect_delays(
                _series([1.0, 2.0, 3.0, 4.0]).assign(CPI=[np.nan, 2.0, np.nan, np.nan]), None, '2023-06-15'
            )

        assert result['frequency'].dtype == 'str'
        assert result.loc['PIB', 'frequency'] == 'monthly'
        assert result.loc['CPI', 'frequency'] is not None and pd.isna(result.loc['CPI', 'frequency'])


class TestEmptyResult:
    """An empty result has the dtypes and level dtypes of a non-empty one, whatever the labels."""

    @pytest.mark.parametrize('emptiness', ['no_observation', 'no_change'])
    @pytest.mark.parametrize('label_dtype', ['str', object], ids=['str_labels', 'object_labels'])
    @pytest.mark.parametrize('structure', ['series', 'panel'])
    def test_same_dtypes_as_a_non_empty_result(self, structure, label_dtype, emptiness):
        """Columns and index levels typed alike; the text columns are ``str``."""
        data = _labelled(structure, label_dtype)
        non_empty = compare_and_detect_delays(data, None, '2023-06-15')

        if emptiness == 'no_observation':
            result = compare_and_detect_delays(data * np.nan, None, '2023-06-15')
        else:
            result = compare_and_detect_delays(data, data, '2023-06-15')

        assert result.empty
        pd.testing.assert_series_equal(result.dtypes, non_empty.dtypes)
        assert _level_dtypes(result) == _level_dtypes(non_empty)
        assert list(result.index.names) == list(non_empty.index.names)
        assert {key: kind for key, kind in string_dtypes(result).items() if key[0] == 'column'} == TEXT_COLUMNS

    def test_integer_labels(self):
        """Integer column labels: an ``int64`` ``column`` level, empty or not."""
        data = pd.DataFrame({1: [1.0, 2.0, 3.0], 2: [1.0, 2.0, 3.0]},
                            index=pd.date_range('2023-01-01', periods=3, freq='MS'))

        non_empty = compare_and_detect_delays(data, None, '2023-06-15')
        empty = compare_and_detect_delays(data * np.nan, None, '2023-06-15')

        assert non_empty.index.dtype == 'int64' and empty.index.dtype == 'int64'


class TestDetectionReport:
    """``DelayDetectionReport.to_frame`` indexes its rows by the labels, in their inferred dtype."""

    def test_report_frame_index_is_str(self):
        """String labels: a ``str`` index named ``column``."""
        _result, report = compare_and_detect_delays(
            _series([1.0, 2.0, 3.0, 4.0]), _series([1.0, 2.0, 3.0, np.nan]), '2023-06-15', return_report=True
        )

        frame = report.to_frame()

        assert frame.index.name == 'column' and frame.index.dtype == 'str'
        assert string_dtypes(frame) == {('index', 0): 'str'}
