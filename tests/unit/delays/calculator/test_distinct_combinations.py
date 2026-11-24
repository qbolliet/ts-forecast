"""Conversion of ``calculate_applicable_delay`` computed once per distinct combination.

Scenarios: entities and indicators sharing an observation date share one conversion,
the result equals a row-by-row computation, unknown delays stay out of the computation,
the first faulty row is the one reported, and the unit conversion follows the same rule.
"""

# Modules de base
import numpy as np
import pandas as pd
import pytest

# Fonction à tester
from tsforecast.delays import calculator
from tsforecast.delays.calculator import calculate_applicable_delay

# Outils de test partagés du paquet
from tests.unit.delays.calculator.helpers import _frame, _monthly, _row


def _panel_of_shared_dates(n_entities: int, months=('2023-10', '2023-11', '2023-12')) -> pd.DataFrame:
    """Build ``n_entities`` x two indicators observed on the same ``months``, same delays."""
    rows, keys = [], []
    for entity in range(n_entities):
        for indicator in ('PIB', 'IPC'):
            for position, month in enumerate(months):
                rows.append(_monthly(month, delay=10 + position))
                keys.append((f'E{entity}', indicator))
    return _frame(rows, index=keys, names=('entity', 'indicator'))


@pytest.mark.internal
class TestOneConversionPerCombination:
    """``_calculate_converted_delay`` runs once per distinct combination of its inputs."""

    def test_shared_dates_are_converted_once(self, monkeypatch):
        """Thirty entities x two indicators x three months: three conversions only."""
        calls = []
        original = calculator._calculate_converted_delay

        def spy(*args):
            calls.append(args)
            return original(*args)

        monkeypatch.setattr(calculator, '_calculate_converted_delay', spy)
        result = calculate_applicable_delay(_panel_of_shared_dates(30), 'start', 'M', aggregation_method='max')
        assert len(calls) == 3
        assert result['n_observations'].tolist() == [90, 90]

    def test_unknown_delays_are_not_converted(self, monkeypatch):
        """A NaN delay or an undetermined frequency is never handed to the conversion."""
        calls = []
        original = calculator._calculate_converted_delay
        monkeypatch.setattr(calculator, '_calculate_converted_delay', lambda *a: calls.append(a) or original(*a))
        rows = [_monthly('2023-12', delay=10), _monthly('2023-12', delay=np.nan),
                _monthly('2023-11', delay=5, frequency=None)]
        calculate_applicable_delay(_frame(rows), 'start', 'M')
        assert len(calls) == 1


class TestSameResultAsRowByRow:
    """Spreading the distinct conversions gives the row-by-row result."""

    def test_aggregates_match_a_row_by_row_computation(self):
        """Each converted delay equals the conversion of its own row."""
        frame = _panel_of_shared_dates(4)
        expected = []
        for _, row in frame.iterrows():
            delay, _start, _end = calculator._calculate_converted_delay(
                row['observation_date'], row['period_start'], row['period_end'], row['reference_point'],
                row['delay'], row['unit'], 'M', 'M', 'start',
            )
            expected.append(delay)
        series = pd.Series(expected, index=frame.index)
        golden = series.groupby(level=-1, sort=True).median().astype(float)
        result = calculate_applicable_delay(frame, 'start', 'M')
        assert result['delay'].to_dict() == golden.to_dict()

    def test_mixed_known_and_unknown_delays(self):
        """Known delays are spread on their rows, unknown ones stay out of the count."""
        rows = [_monthly('2023-12', delay=10), _monthly('2023-12', delay=np.nan), _monthly('2023-12', delay=10)]
        result = calculate_applicable_delay(_frame(rows, index=['A', 'A', 'B']), 'start', 'M')
        assert result.loc['A', 'n_observations'] == 1
        assert result.loc['B', 'n_observations'] == 1
        assert result.loc['A', 'delay'] == result.loc['B', 'delay']

    def test_only_unknown_delays(self):
        """Without any known delay, every group has a NaN delay and no observation."""
        rows = [_monthly('2023-12', delay=np.nan), _monthly('2023-11', delay=np.nan)]
        result = calculate_applicable_delay(_frame(rows, index=['A', 'B']), 'start', 'M')
        assert result['delay'].isna().all()
        assert result['n_observations'].tolist() == [0, 0]

    def test_unit_conversion_spreads_over_shared_delays(self):
        """Converting to hours multiplies every shared delay by 24, NaN kept."""
        rows = [_monthly('2023-12', delay=10), _monthly('2023-12', delay=10), _monthly('2023-12', delay=np.nan)]
        result = calculate_applicable_delay(_frame(rows, index=['A', 'B', 'C']), 'start', 'M', unit='hour')
        days = calculate_applicable_delay(_frame(rows, index=['A', 'B', 'C']), 'start', 'M')
        assert result.loc['A', 'delay'] == days.loc['A', 'delay'] * 24
        assert result.loc['B', 'delay'] == result.loc['A', 'delay']
        assert np.isnan(result.loc['C', 'delay'])


class TestFirstFaultyRowIsReported:
    """Distinct combinations are visited in order of first appearance."""

    def test_first_unsupported_unit_is_named(self):
        """With two unsupported units, the error names the one of the first row."""
        rows = [_monthly('2023-12', delay=10),
                {**_monthly('2023-11', delay=3), 'unit': 'fortnight'},
                {**_monthly('2023-10', delay=3), 'unit': 'lustrum'}]
        with pytest.raises(ValueError, match='fortnight'):
            calculate_applicable_delay(_frame(rows, index=['A', 'B', 'C']), 'start', 'M')

    def test_uncovered_keys_are_listed_once_in_order(self):
        """Index keys without a target frequency are named once each, in order of appearance."""
        rows = [_row('2023-12-15', '2023-12-01', '2024-01-01', 5) for _ in range(4)]
        frame = _frame(rows, index=[('FR', 'X'), ('DE', 'Y'), ('FR', 'X'), ('IT', 'Y')],
                       names=('entity', 'indicator'))
        with pytest.raises(ValueError, match=r"\[\('FR', 'X'\), \('DE', 'Y'\), \('IT', 'Y'\)\]"):
            calculate_applicable_delay(frame, 'start', {('XX', 'Z'): 'M'}, aggregate_by_panel=True)
