"""Independence of the panel splitters from the resolution of the dates.

The staggered panel and the realistic heterogeneous panel, dates stored in
seconds, milliseconds, microseconds or nanoseconds (``with_datetime_resolution``),
with default folds and explicit test dates (strings, ``Timestamp`` objects of
another resolution, ``(entity, date)`` tuples): ``PanelOutOfSampleSplit``,
``PanelInSampleSplit`` and their ``PerEntity`` variants give the folds of the
nanosecond panel.
"""
# Importation des modules
import warnings

import pandas as pd
import pytest

from tests.support.datasets import HETEROGENEOUS_PANEL_COUNTRIES, build_mixed_frequency_panel
from tests.support.perturbations import DATETIME_UNITS, with_datetime_resolution
from tests.support.splits import as_lists, collect_folds
from tsforecast.crossvals.panel import (
    PanelInSampleSplit,
    PanelInSampleSplitPerEntity,
    PanelOutOfSampleSplit,
    PanelOutOfSampleSplitPerEntity,
)

# Configurations : classe, paramètres
CONFIGURATIONS = {
    'out-default': (PanelOutOfSampleSplit, {'n_splits': 2, 'test_size': 1}),
    'out-timestamps': (PanelOutOfSampleSplit, {
        'test_indices': [pd.Timestamp('2024-08-31').as_unit('s'), '2024-10-31'], 'test_size': 2,
    }),
    'out-tuples': (PanelOutOfSampleSplit, {'test_indices': [('B', '2024-06-30'), ('C', '2024-10-31')], 'gap': 1}),
    'in-default': (PanelInSampleSplit, {'test_size': 2}),
    'out-per-entity': (PanelOutOfSampleSplitPerEntity, {'n_splits': 2, 'test_size': 1}),
    'in-per-entity': (PanelInSampleSplitPerEntity, {
        'test_indices': [pd.Timestamp('2024-09-30').as_unit('ms')], 'max_train_size': 4,
    }),
}


def _folds(splitter, X: pd.DataFrame) -> list:
    """Folds as lists of positions, the warnings about entities without test dates silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return as_lists(collect_folds(splitter, X))


class TestDatetimeResolution:
    """Folds do not depend on how the dates of the panel are stored."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('name', list(CONFIGURATIONS), ids=list(CONFIGURATIONS))
    def test_staggered_panel(self, staggered_panel, name, unit):
        """Same train and test positions as on the nanosecond panel."""
        splitter_class, params = CONFIGURATIONS[name]
        expected = _folds(splitter_class(**params), with_datetime_resolution(staggered_panel, 'ns'))

        folds = _folds(splitter_class(**params), with_datetime_resolution(staggered_panel, unit))

        assert folds == expected and folds

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize(
        'splitter_class', [PanelOutOfSampleSplit, PanelOutOfSampleSplitPerEntity],
        ids=['global', 'per-entity'],
    )
    def test_realistic_panel(self, splitter_class, unit):
        """Entities with their own coverage and isolated annual anchors: same folds at every resolution."""
        panel = build_mixed_frequency_panel(countries=HETEROGENEOUS_PANEL_COUNTRIES)
        expected = _folds(splitter_class(n_splits=3, test_size=3), with_datetime_resolution(panel, 'ns'))

        folds = _folds(splitter_class(n_splits=3, test_size=3), with_datetime_resolution(panel, unit))

        assert folds == expected
