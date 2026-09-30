"""Tests for ``detect_index_frequency`` and ``target_offset_for_index`` (``utils/frequency/utils.py``).

Covers the detection on a bare index (``DatetimeIndex``, or one frequency per entity
of an ``(entity, date)`` ``MultiIndex``) in every ``return_format``, with the
``FrequencyDetector`` fallback for gapped, unsorted or irregular indexes; and the
target offset anchored like a source index (position of the source over the
target's, fallbacks, lost multiplier and anchor ANO-UTILS-048).
"""
from __future__ import annotations

import pandas as pd
import pytest

from tsforecast.utils.frequency.utils import detect_index_frequency, target_offset_for_index


# =============================================================================
# detect_index_frequency
# =============================================================================

class TestDetectIndexFrequency:
    """``detect_index_frequency`` on a ``DatetimeIndex`` or an ``(entity, date)`` index."""

    @pytest.mark.parametrize(
        'freq, start, return_format, expected',
        [
            pytest.param('MS', '2023-01-01', 'base', 'M', id='base'),
            pytest.param('MS', '2023-01-01', 'with_position', 'MS', id='with-position'),
            pytest.param('QE-DEC', '2023-03-31', 'full', 'QE-DEC', id='full'),
            pytest.param('MS', '2023-01-01', 'components', ('M', 'S', None, 1), id='components'),
            pytest.param('QE-DEC', '2023-03-31', 'components', ('Q', 'E', 'DEC', 1),
                         id='components-quarterly'),
            pytest.param('D', '2023-01-01', 'components', ('D', None, None, 1), id='components-daily'),
        ],
    )
    def test_return_format(self, freq, start, return_format, expected):
        """Golden value of each format on a regular index."""
        dates = pd.date_range(start, periods=8, freq=freq)
        assert detect_index_frequency(dates, return_format=return_format) == expected

    def test_default_format_is_base(self):
        """Without ``return_format``, the base code is returned."""
        assert detect_index_frequency(pd.date_range('2023-01-01', periods=10, freq='MS')) == 'M'

    def test_fallback_on_gapped_index(self):
        """A gapped quarterly index, missed by ``inferred_freq``, is detected by the fallback."""
        dates = pd.date_range('2020-01-01', periods=10, freq='QS').delete([4])
        assert detect_index_frequency(dates, return_format='with_position') == 'QS'

    def test_unsorted_index(self):
        """An unsorted index is detected as the sorted one."""
        dates = pd.date_range('2023-01-31', periods=6, freq='ME')[[3, 0, 5, 1, 4, 2]]
        assert detect_index_frequency(dates, return_format='full') == 'ME'

    def test_irregular_index_is_none(self):
        """An index without a dominant spacing gives ``None``, not an error (ANO-UTILS-050)."""
        # Écarts de 45 puis 50 jours
        dates = pd.DatetimeIndex(['2023-01-01', '2023-02-15', '2023-04-06'])
        assert detect_index_frequency(dates) is None

    def test_irregular_index_gives_the_dominant_grid(self):
        """Isolated dates outside a regular grid do not hide the grid's frequency (modal spacing)."""
        # Deux ancres annuelles isolées avant 24 débuts de mois : l'index est irrégulier,
        # l'écart modal reste mensuel
        dates = pd.DatetimeIndex(['2020-01-01', '2021-01-01']).append(
            pd.date_range('2022-01-01', periods=24, freq='MS'))
        assert detect_index_frequency(dates, return_format='full') == 'MS'

    def test_string_labels_are_converted(self):
        """ISO date strings are parsed before detection."""
        labels = pd.Index(['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'])
        assert detect_index_frequency(labels, return_format='with_position') == 'QS'

    def test_single_date_raises(self):
        """A single date raises."""
        with pytest.raises(ValueError, match='only 1 non-null observations'):
            detect_index_frequency(pd.DatetimeIndex(['2024-01-01']))

    def test_invalid_return_format_raises(self):
        """An unknown ``return_format`` is rejected."""
        with pytest.raises(ValueError, match='Invalid return_format'):
            detect_index_frequency(pd.date_range('2024-01-01', periods=5), return_format='bogus')

    @pytest.mark.parametrize(
        'return_format, expected',
        [pytest.param('base', 'M', id='base'),
         pytest.param('components', ('M', 'S', None, 1), id='components')],
    )
    def test_multiindex(self, return_format, expected):
        """One frequency per entity, keyed by a one-element tuple."""
        index = pd.MultiIndex.from_product(
            [['entity_1', 'entity_2'], pd.date_range('2024-01-01', periods=5, freq='MS')],
            names=['entity', 'date'])
        detected = detect_index_frequency(index, return_format=return_format)
        assert detected == {('entity_1',): expected, ('entity_2',): expected}

    def test_multiindex_with_single_date_entity_raises(self):
        """Unlike ``detect_frequency``, one entity with a single date aborts the whole index."""
        index = pd.MultiIndex.from_arrays(
            [['A', 'A', 'A', 'B'],
             list(pd.date_range('2024-01-01', periods=3)) + [pd.Timestamp('2024-01-01')]])
        with pytest.raises(ValueError, match='only 1 non-null observations'):
            detect_index_frequency(index)

    def test_integer_index_raises_value_error(self):
        """An integer index is rejected with a ``ValueError`` (ANO-UTILS-044)."""
        with pytest.raises(ValueError, match='numeric labels are not dates'):
            detect_index_frequency(pd.Index([2020, 2021, 2022]))

    def test_period_index(self):
        """A quarterly ``PeriodIndex`` is read at its quarter starts (ANO-UTILS-045)."""
        index = pd.period_range('2024Q1', periods=4, freq='Q')
        assert detect_index_frequency(index, return_format='with_position') == 'QS'

    def test_period_date_level(self):
        """The ``Period`` date level of a ``MultiIndex`` is converted per entity."""
        index = pd.MultiIndex.from_product(
            [['A', 'B'], pd.period_range('2024-01', periods=4, freq='M')], names=['entity', 'date'])
        assert detect_index_frequency(index) == {('A',): 'M', ('B',): 'M'}


# =============================================================================
# target_offset_for_index
# =============================================================================

class TestTargetOffsetForIndex:
    """Target offset anchored like the source index (start / end position).

    ``target_offset_for_index`` factorise la logique utilisée par
    ``FrequencyAligner.aggregate_to_target`` et
    ``ImputationWindowCalculator._convert_mask_to_frequency`` : la position de
    l'index source l'emporte sur celle de la cible.
    """

    MS_INDEX = pd.date_range('2024-01-01', periods=12, freq='MS')
    ME_INDEX = pd.date_range('2024-01-31', periods=12, freq='ME')

    @pytest.mark.parametrize(
        'index, target, expected',
        [
            pytest.param(MS_INDEX, 'Q', 'QS', id='start-source-bare-target'),
            pytest.param(ME_INDEX, 'Q', 'QE', id='end-source-bare-target'),
            # Position cible déjà présente : écrasée par celle de la source
            pytest.param(MS_INDEX, 'QE', 'QS', id='start-source-overrides-end-target'),
            pytest.param(ME_INDEX, 'YS', 'YE', id='end-source-overrides-start-target'),
            pytest.param(MS_INDEX, 'QE-DEC', 'QS', id='start-source-default-anchor-target'),
            pytest.param(MS_INDEX, 'Y', 'YS', id='start-source-annual-target'),
            pytest.param(MS_INDEX, 'quarterly', 'QS', id='literal-target'),
            pytest.param(MS_INDEX, 'SM', 'SMS', id='semi-monthly-target'),
            # Cibles sans notion de position : inchangées
            pytest.param(MS_INDEX, 'W', 'W', id='weekly-target'),
            pytest.param(MS_INDEX, 'D', 'D', id='daily-target'),
            # Sources dont la position reste détectable malgré leur forme
            pytest.param(pd.date_range('2024-01-01', periods=2, freq='MS'), 'Q', 'QS', id='two-dates'),
            pytest.param(MS_INDEX.delete([4, 5]), 'Q', 'QS', id='gapped-source'),
            pytest.param(pd.date_range('2020-01-01', periods=5, freq='QS'), 'Y', 'YS', id='quarterly-source'),
        ],
    )
    def test_position_follows_the_source(self, index, target, expected):
        """Golden anchored offset for each (source, target) pair."""
        assert target_offset_for_index(index, target) == expected

    @pytest.mark.parametrize(
        'index, target',
        [
            # Un seul point : aucune fréquence détectable -> rien à ancrer
            pytest.param(pd.DatetimeIndex(['2024-01-01']), 'Q', id='single-date'),
            pytest.param(pd.DatetimeIndex([]), 'Q', id='empty'),
            # Source journalière (sans position) : cible déjà positionnée conservée
            pytest.param(pd.date_range('2024-01-01', periods=59, freq='D'), 'ME', id='daily-source'),
            pytest.param(pd.date_range('2024-01-01', periods=10, freq='W'), 'M', id='weekly-source'),
            # Dates au 15 du mois : mensuelles, mais ni début ni fin de période
            pytest.param(pd.date_range('2024-01-01', periods=5, freq='MS') + pd.Timedelta(days=14), 'Q',
                         id='mid-month-source'),
            # Cible inconnue : renvoyée telle quelle, sans erreur
            pytest.param(MS_INDEX, 'bogus', id='unknown-target'),
        ],
    )
    def test_target_unchanged_without_usable_position(self, index, target):
        """Without a detectable source position (or a usable target), the target is kept."""
        assert target_offset_for_index(index, target) == target

    @pytest.mark.parametrize(
        'index, target, expected',
        [
            pytest.param(ME_INDEX, '2Q', '2QE', id='half-year-end'),
            pytest.param(MS_INDEX, '2Q', '2QS', id='half-year-start'),
            pytest.param(MS_INDEX, '2MS', '2MS', id='two-months'),
        ],
    )
    def test_target_multiplier_is_kept(self, index, target, expected):
        """A half-year target stays a half-year once anchored (ANO-UTILS-048)."""
        assert target_offset_for_index(index, target) == expected

    @pytest.mark.parametrize(
        'index, target, expected',
        [
            # Trimestres déc.-févr., mars-mai, ... : débuts en déc. (forme canonique : mars)
            pytest.param(MS_INDEX, 'QE-NOV', 'QS-MAR', id='quarter-end-november'),
            # Ancre sans position : lue comme une fin, comme le fait pandas
            pytest.param(MS_INDEX, 'Q-NOV', 'QS-MAR', id='quarter-november-without-position'),
            # Exercice juillet-juin : début d'exercice en juillet
            pytest.param(MS_INDEX, 'YE-JUN', 'YS-JUL', id='year-end-june'),
            # Sens inverse : exercice commençant en juillet, source ancrée en fin
            pytest.param(ME_INDEX, 'YS-JUL', 'YE-JUN', id='year-start-july'),
            pytest.param(ME_INDEX, 'QS-FEB', 'QE-JAN', id='quarter-start-february'),
            # Même position que la source : ancre conservée
            pytest.param(MS_INDEX, 'QS-FEB', 'QS-FEB', id='same-position'),
            # Ancre hebdomadaire : sans notion de position, conservée
            pytest.param(MS_INDEX, 'W-MON', 'W-MON', id='weekly-anchor'),
        ],
    )
    def test_target_periods_are_kept(self, index, target, expected):
        """Re-anchoring a target keeps its periods: only the position changes (ANO-UTILS-048)."""
        assert target_offset_for_index(index, target) == expected

    @pytest.mark.parametrize('target', ['QE-NOV', 'YE-JUN', '2QE-NOV'])
    def test_reanchored_target_describes_the_same_periods(self, target):
        """Property: the start-anchored offset starts exactly the periods the target ends."""
        # Chaque début de période cible = lendemain d'une fin de période de la cible
        ends = pd.date_range('2019-01-01', periods=6, freq=target)
        starts = pd.date_range(ends[0] + pd.Timedelta(days=1), periods=6,
                               freq=target_offset_for_index(self.MS_INDEX, target))
        assert list(starts) == [end + pd.Timedelta(days=1) for end in ends]
