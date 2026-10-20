"""Independence of ``validate_temporal_data`` from the resolution of the input dates.

The same series and panels, stored with dates in seconds, milliseconds,
microseconds or nanoseconds (``with_datetime_resolution``), on the index path
and on the column path: validated values and row order do not depend on the
resolution, and the dates come back at the input resolution.
"""

import pandas as pd
import pytest

# Jeux de données et perturbations partagés
from tests.support.datasets import build_mixed_frequency_panel, build_mixed_frequency_timeseries
from tests.support.perturbations import (
    DATETIME_UNITS,
    datetime_units,
    shuffle_rows,
    unsort_within_entities,
    with_datetime_resolution,
)

# Fonction à tester
from tsforecast.utils.validation import validate_temporal_data

# Outils de test partagés du paquet
from tests.unit.utils.validation.utils.helpers import _validate_kwargs


# Le chemin « colonnes » émet systématiquement un avertissement de remplacement d'index
pytestmark = pytest.mark.filterwarnings("ignore:Index replaced with")


# Jeux non triés : le tri par date est la partie du traitement qui lit les dates
_UNSORTED_DATASETS = {
    'series': lambda: shuffle_rows(build_mixed_frequency_timeseries(), seed=4),
    'panel': lambda: unsort_within_entities(build_mixed_frequency_panel(), seed=4),
}


class TestDatetimeResolution:
    """Validated output identical at every resolution, dates at the input resolution."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('path', ['index', 'columns'])
    @pytest.mark.parametrize('dataset', list(_UNSORTED_DATASETS), ids=list(_UNSORTED_DATASETS))
    def test_output_does_not_depend_on_resolution(self, dataset: str, path: str, unit: str) -> None:
        """Values and row order equal those of the nanosecond input; dates keep the input unit."""
        data = _UNSORTED_DATASETS[dataset]()
        # Référence : les mêmes données en nanosecondes
        frame, kwargs = _validate_kwargs(with_datetime_resolution(data, 'ns'), path)
        reference = validate_temporal_data(frame, **kwargs)

        frame, kwargs = _validate_kwargs(with_datetime_resolution(data, unit), path)
        result = validate_temporal_data(frame, **kwargs)

        pd.testing.assert_frame_equal(result, reference, check_index_type=False)
        assert set(datetime_units(result).values()) == {unit}

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_sortedness_metadata_does_not_depend_on_resolution(self, unit: str) -> None:
        """The scalar metadata (index type, names, sort and reorder flags) is the same at every resolution."""
        data = _UNSORTED_DATASETS['panel']()
        _, reference = validate_temporal_data(with_datetime_resolution(data, 'ns'), return_metadata=True)

        _, metadata = validate_temporal_data(with_datetime_resolution(data, unit), return_metadata=True)

        comparable = {key for key, value in reference.items() if not isinstance(value, (pd.Index, pd.DataFrame))}
        assert {key: metadata[key] for key in comparable} == {key: reference[key] for key in comparable}
