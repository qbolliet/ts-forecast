"""Independence of ``OutOfSampleSplit`` and ``InSampleSplit`` from the resolution of the dates.

Twenty-four month ends stored in seconds, milliseconds, microseconds or
nanoseconds (``with_datetime_resolution``), with default folds, explicit test
dates given as strings or as ``Timestamp`` objects of another resolution, and
rolling windows: the folds equal those of the nanosecond index.
"""
# Importation des modules
import pandas as pd
import pytest

from tests.support.perturbations import DATETIME_UNITS, shuffle_rows, with_datetime_resolution
from tests.support.splits import as_lists, collect_folds
from tsforecast.crossvals.base import InSampleSplit, OutOfSampleSplit

# Configurations : classe et paramètres ; dates de test en chaînes et en Timestamp
# d'une autre résolution que l'index
CONFIGURATIONS = {
    'out-default': (OutOfSampleSplit, {'n_splits': 3}),
    'out-gap-rolling': (OutOfSampleSplit, {'n_splits': 3, 'test_size': 2, 'gap': 1, 'max_train_size': 6}),
    'out-date-strings': (OutOfSampleSplit, {'test_indices': ['2023-12-31', '2024-06-30'], 'test_size': 3}),
    'out-timestamps': (OutOfSampleSplit, {
        'test_indices': [pd.Timestamp('2024-03-31').as_unit('s'), pd.Timestamp('2024-09-30').as_unit('ns')],
    }),
    'in-default': (InSampleSplit, {'n_splits': 3, 'test_size': 2}),
    'in-timestamps': (InSampleSplit, {'test_indices': [pd.Timestamp('2024-05-31').as_unit('ms')],
                                      'max_train_size': 8}),
}


class TestDatetimeResolution:
    """Folds do not depend on how the dates of the index are stored."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('name', list(CONFIGURATIONS), ids=list(CONFIGURATIONS))
    def test_folds(self, cv_frame_24, name, unit):
        """Same train and test positions as on the nanosecond index."""
        splitter_class, params = CONFIGURATIONS[name]
        expected = as_lists(collect_folds(splitter_class(**params), with_datetime_resolution(cv_frame_24, 'ns')))

        folds = as_lists(collect_folds(splitter_class(**params), with_datetime_resolution(cv_frame_24, unit)))

        assert folds == expected

    @pytest.mark.filterwarnings('ignore:Time series data is not sorted')
    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_unsorted_index(self, cv_frame_24, unit):
        """An unsorted index is sorted by date the same way at every resolution."""
        shuffled = shuffle_rows(cv_frame_24, seed=5)
        expected = as_lists(collect_folds(OutOfSampleSplit(n_splits=3), with_datetime_resolution(shuffled, 'ns')))

        folds = as_lists(collect_folds(OutOfSampleSplit(n_splits=3), with_datetime_resolution(shuffled, unit)))

        assert folds == expected
