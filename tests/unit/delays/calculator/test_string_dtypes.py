"""String dtypes of the applicable delays returned by ``calculate_applicable_delay``.

The text columns (``unit``, ``frequency``, ``reference_point``,
``aggregation_method``) are in the ``str`` dtype for a series and for a panel,
aggregated by indicator or by entity, including when the delay table holds its
text as ``object`` and when the aggregation is a callable.
"""

import numpy as np
import pytest

# Outils de test partagés
from tests.support.perturbations import string_dtypes

# Fonction à tester
from tsforecast.delays.calculator import calculate_applicable_delay

# Colonnes textuelles du résultat
TEXT_COLUMNS = {
    ('column', 'unit'): 'str', ('column', 'frequency'): 'str',
    ('column', 'reference_point'): 'str', ('column', 'aggregation_method'): 'str',
}


class TestStringDtypes:
    """Text columns in ``str``; labels in their inferred dtype."""

    def test_series_delays(self, simple_time_series_delays):
        result = calculate_applicable_delay(simple_time_series_delays, 'end', 'quarterly')

        assert string_dtypes(result) == {('index', 0): 'str', **TEXT_COLUMNS}

    @pytest.mark.parametrize('aggregate_by_panel', [False, True])
    def test_panel_delays(self, monthly_publication_delays, aggregate_by_panel):
        result = calculate_applicable_delay(
            monthly_publication_delays, 'start', 'M', unit='hour', aggregate_by_panel=aggregate_by_panel
        )

        expected_index = {('index', level): 'str' for level in range(result.index.nlevels)}
        assert string_dtypes(result) == {**expected_index, **TEXT_COLUMNS}

    def test_object_text_and_callable_aggregation(self, simple_time_series_delays):
        """Text stored as ``object`` in the table, aggregation by a function: still ``str``."""
        delays = simple_time_series_delays.astype({'unit': object, 'reference_point': object})

        result = calculate_applicable_delay(delays, 'start', 'monthly', aggregation_method=np.max)

        assert {key: kind for key, kind in string_dtypes(result).items() if key[0] == 'column'} == TEXT_COLUMNS
        assert result['aggregation_method'].unique().tolist() == ['max']
