"""Tests of the grid and window-mask functions of ``tsforecast.imputation._engine.grids``.

Scenarios: restriction of an index to a group of entities (order, scalar keys, time series, empty
group, two entity levels); stage masks (kind passed through, entity omitted by the calculator,
unavailable mask with and without log); prediction and unrestricted grids; training mask binding
(scalar block frequency of a time series, panel binding, unavailable mask).
"""
# Manipulation de données
import numpy as np
import pandas as pd

from tsforecast.imputation._components.imputation_window import ImputationWindowCalculator
from tsforecast.imputation._engine.grids import (
    prediction_grid,
    restrict_to_entities,
    stage_mask,
    training_mask_at,
    unrestricted_grid,
)

# Clés d'entité des jeux de référence
FR, DE, IT = ('FR',), ('DE',), ('IT',)


# =============================================================================
# Doublures et constructeurs
# =============================================================================
# Calculateur de fenêtre minimal : un masque donné, des appels enregistrés
class _FixedMaskCalculator:
    """Window calculator double returning one given mask and recording its calls."""

    def __init__(self, mask: pd.Series) -> None:
        self.mask = mask
        self.calls = []

    def get_mask_at_frequency(self, frequency, kind='imputation'):
        """Record the call and return the given mask."""
        self.calls.append((frequency, kind))
        return self.mask


# Index de panel de trois entités sur trois mois
def _panel_index() -> pd.MultiIndex:
    """Build a ``(country, date)`` index: FR, DE, IT over three month ends."""
    dates = pd.date_range('2021-01-31', periods=3, freq='ME')
    return pd.MultiIndex.from_product([['FR', 'DE', 'IT'], dates], names=['country', 'date'])


# =============================================================================
# Grilles et masques
# =============================================================================
class TestRestrictToEntities:
    """Restriction of an index to the rows of a group, decided per entity."""

    def test_rows_of_the_group_keep_their_order(self):
        index = _panel_index()
        restricted = restrict_to_entities(index, [IT, FR])
        assert restricted.get_level_values('country').tolist() == ['FR'] * 3 + ['IT'] * 3
        assert restricted.equals(index[np.r_[0:3, 6:9]])

    def test_scalar_keys_designate_single_level_entities(self):
        index = _panel_index()
        assert restrict_to_entities(index, ['DE']).equals(restrict_to_entities(index, [DE]))

    def test_time_series_and_none_are_left_untouched(self):
        index = _panel_index()
        dates = index.get_level_values('date')[:3]
        assert restrict_to_entities(index, None) is index
        assert restrict_to_entities(dates, [FR]) is dates

    def test_empty_group_gives_an_empty_index(self):
        restricted = restrict_to_entities(_panel_index(), [])
        assert len(restricted) == 0
        assert restricted.names == ['country', 'date']

    def test_two_entity_levels(self):
        dates = pd.date_range('2021-01-31', periods=2, freq='ME')
        index = pd.MultiIndex.from_product(
            [['EU'], ['FR', 'DE'], dates], names=['zone', 'country', 'date']
        )
        restricted = restrict_to_entities(index, [('EU', 'DE')])
        assert restricted.get_level_values('country').tolist() == ['DE', 'DE']


class TestStageMask:
    """Window masks read at the frequency of a stage."""

    def test_entity_omitted_by_the_calculator_is_left_unrestricted(self):
        index = _panel_index()
        fr_rows = index[:3]
        calculator = _FixedMaskCalculator(pd.Series([False, True, True], index=fr_rows))

        mask = stage_mask(calculator, index, {FR: 'M'}, 'imputation', pass_name='fit')

        # Le "kind" est transmis tel quel ; FR garde son masque, DE et IT sont rattachées
        assert calculator.calls == [({FR: 'M'}, 'imputation')]
        assert mask.tolist() == [False, True, True] + [True] * 6
        assert mask.index[3:].equals(index[3:])

    def test_unavailable_mask_keeps_every_row_and_logs(self):
        index = _panel_index()
        messages = []
        mask = stage_mask(
            ImputationWindowCalculator(), index, 'M', 'training',
            pass_name='transform', log=messages.append,
        )
        assert mask.index.equals(index) and mask.all()
        assert messages == [
            '[transform] training mask unavailable (Calculator not fitted. Call fit() '
            'first.); every row kept'
        ]

    def test_unavailable_mask_without_log_is_silent(self):
        index = _panel_index()
        mask = stage_mask(ImputationWindowCalculator(), index, 'M', 'imputation', pass_name='fit')
        assert mask.index.equals(index) and mask.all()

    def test_prediction_grid_keeps_the_true_rows_of_the_group(self):
        index = _panel_index()
        mask = pd.Series([True, False, True, True, True, False, True, True, True], index=index)
        calculator = _FixedMaskCalculator(mask)
        grid = prediction_grid(calculator, index, 'M', [DE], pass_name='fit')
        assert grid.equals(index[[3, 4]])

    def test_unrestricted_grid_keeps_every_row_of_the_group(self):
        index = _panel_index()
        mask = pd.Series(False, index=index)
        grid = unrestricted_grid(_FixedMaskCalculator(mask), index, 'M', [IT], pass_name='fit')
        assert grid.equals(index[6:])
        assert len(prediction_grid(_FixedMaskCalculator(mask), index, 'M', [IT], pass_name='fit')) == 0

class TestTrainingMaskAt:
    """The training mask injected into the training-set builder."""

    def test_time_series_reads_the_scalar_block_frequency(self):
        calculator = _FixedMaskCalculator(pd.Series(dtype=bool))
        training_mask_at(calculator, {(): 'QE'}, is_panel=False)
        assert calculator.calls == [('QE', 'training')]

    def test_panel_passes_the_binding_as_given(self):
        calculator = _FixedMaskCalculator(pd.Series(dtype=bool))
        blocks = {FR: 'YE', DE: 'QE'}
        training_mask_at(calculator, blocks, is_panel=True)
        assert calculator.calls == [(blocks, 'training')]

    def test_unavailable_mask_means_no_restriction(self):
        messages = []
        result = training_mask_at(
            ImputationWindowCalculator(), {(): 'QE'}, is_panel=False, log=messages.append
        )
        assert result is None
        assert messages == [
            '[fit] training mask unavailable (Calculator not fitted. Call fit() first.); '
            'every row kept'
        ]
