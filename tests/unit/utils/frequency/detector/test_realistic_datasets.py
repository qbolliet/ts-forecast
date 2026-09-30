"""Frequency detection on the realistic datasets of notebook 3.

Covers ``detect_dataset_frequency``, ``detect_frequency`` and ``detect_index_frequency``
on ``irregular_index_timeseries`` (frequency of each column and of the genuinely
irregular global index) and ``heterogeneous_coverage_panel`` (frequency per
``(entity, column)``: ``depenses_publiques_pib`` annual for France / Italie and
quarterly for Allemagne, ``climat_affaires`` never observed for Italie), under the
perturbations of ``tests/support/perturbations.py``.
"""
from __future__ import annotations

import pytest

from tests.support.perturbations import (
    drop_entity,
    reverse_entities,
    shuffle_rows,
    to_period_end,
    to_three_level_index,
    with_duplicated_rows,
    with_index_names,
    with_special_column_names,
)
from tsforecast.utils.frequency.utils import (
    detect_dataset_frequency,
    detect_frequency,
    detect_index_frequency,
)


# =============================================================================
# Jeux réalistes du notebook 3
# =============================================================================

# Valeurs d'or de irregular_index_timeseries (build_mixed_frequency_timeseries) :
# grille MS 2018-01 à 2024-07, PIB aux mois 1/4/7/10, balance annuelle YS dès 2015
IRREGULAR_TIMESERIES_FREQUENCIES = {
    'production_industrielle': 'MS',
    'inflation_ipc': 'MS',
    'taux_chomage': 'MS',
    'pib_trimestriel': 'QS-JAN',
    'balance_commerciale_annuelle': 'YS-JAN',
}


class TestIrregularIndexTimeseries:
    """Detection on the realistic, genuinely irregular time series of notebook 3."""

    def test_frequency_of_each_column(self, irregular_index_timeseries):
        """Each column keeps its own publication frequency and start position."""
        detected = detect_dataset_frequency(irregular_index_timeseries, return_format='full')
        assert detected == IRREGULAR_TIMESERIES_FREQUENCIES

    @pytest.mark.parametrize(
        'return_format, expected',
        [pytest.param('base', 'M', id='base'),
         pytest.param('with_position', 'MS', id='with-position'),
         pytest.param('full', 'MS', id='full')],
    )
    def test_global_index_is_monthly(self, irregular_index_timeseries, return_format, expected):
        """The global index is monthly: annual anchors before the grid do not change the modal spacing."""
        # Trois ancres annuelles (2015-2017) isolées avant une grille de 79 mois : l'écart
        # modal reste mensuel, et toutes les dates sont des débuts de mois
        detected = detect_index_frequency(irregular_index_timeseries.index, return_format=return_format)
        assert detected == expected

    def test_end_anchored_variant(self, irregular_index_timeseries):
        """Once moved to period ends, every column is end anchored."""
        detected = detect_dataset_frequency(to_period_end(irregular_index_timeseries),
                                            return_format='with_position')
        expected = {column: frequency[0] + 'E'
                    for column, frequency in IRREGULAR_TIMESERIES_FREQUENCIES.items()}
        assert detected == expected

    def test_shuffled_rows(self, irregular_index_timeseries):
        """Shuffled rows give the same frequencies."""
        detected = detect_dataset_frequency(shuffle_rows(irregular_index_timeseries, seed=7),
                                            return_format='full')
        assert detected == IRREGULAR_TIMESERIES_FREQUENCIES

    def test_special_column_names(self, irregular_index_timeseries):
        """Spaces, accents and symbols in column names are kept as keys."""
        renamed, mapping = with_special_column_names(irregular_index_timeseries)
        expected = {mapping[column]: frequency
                    for column, frequency in IRREGULAR_TIMESERIES_FREQUENCIES.items()}
        assert detect_dataset_frequency(renamed, return_format='full') == expected

    def test_modal_consistency(self, irregular_index_timeseries):
        """Three monthly columns out of five: monthly is modal."""
        detected = detect_dataset_frequency(irregular_index_timeseries, check_consistency=True,
                                            strict=False)
        assert detected == 'M'


# Colonnes mensuelles de heterogeneous_coverage_panel, observées par toutes les entités
MONTHLY_PANEL_COLUMNS = ('production_industrielle', 'inflation_ipc', 'taux_chomage')


def _heterogeneous_panel_frequencies() -> dict:
    """Golden base frequencies of ``heterogeneous_coverage_panel``.

    Returns:
        Mapping ``(entity, column) -> base frequency``: monthly columns ``'M'``,
        GDP ``'Q'``, trade balance ``'Y'``, public spending ``'Y'`` for France /
        Italie and ``'Q'`` for Allemagne, business climate ``'M'`` except for
        Italie, which never observes it (``None``).
    """
    expected = {}
    for entity in ('Allemagne', 'France', 'Italie'):
        for column in MONTHLY_PANEL_COLUMNS:
            expected[(entity, column)] = 'M'
        expected[(entity, 'pib_trimestriel')] = 'Q'
        expected[(entity, 'depenses_publiques_pib')] = 'Q' if entity == 'Allemagne' else 'Y'
        expected[(entity, 'balance_commerciale_annuelle')] = 'Y'
        expected[(entity, 'climat_affaires')] = None if entity == 'Italie' else 'M'
    return expected


class TestHeterogeneousCoveragePanel:
    """Detection per (entity, column) on the realistic heterogeneous panel of notebook 3."""

    def test_frequency_of_each_pair(self, heterogeneous_coverage_panel):
        """Every (entity, column) pair has its golden frequency."""
        assert detect_dataset_frequency(heterogeneous_coverage_panel) == _heterogeneous_panel_frequencies()

    @pytest.mark.parametrize(
        'entity, expected',
        [pytest.param('France', 'YS', id='France'),
         pytest.param('Allemagne', 'QS', id='Allemagne'),
         pytest.param('Italie', 'YS', id='Italie')],
    )
    def test_public_spending_frequency_depends_on_entity(self, heterogeneous_coverage_panel, entity, expected):
        """``depenses_publiques_pib``: annual for France / Italie, quarterly for Allemagne."""
        detected = detect_dataset_frequency(heterogeneous_coverage_panel, return_format='with_position')
        assert detected[(entity, 'depenses_publiques_pib')] == expected

    def test_never_observed_column_is_none(self, heterogeneous_coverage_panel):
        """``climat_affaires`` is never observed for Italie: present, mapped to ``None``."""
        detected = detect_frequency(heterogeneous_coverage_panel['climat_affaires'])
        assert detected == {('Allemagne',): 'M', ('France',): 'M', ('Italie',): None}

    @pytest.mark.parametrize(
        'options, expected',
        [
            pytest.param({'strict': True}, None, id='strict'),
            # Valeur d'or : deux entités annuelles contre une trimestrielle
            pytest.param({'strict': False}, 'Y', id='modal'),
            pytest.param({'consistency_mode': 'highest'}, 'Q', id='highest'),
        ],
    )
    def test_public_spending_consistency(self, heterogeneous_coverage_panel, options, expected):
        """Golden reduction of the heterogeneous public spending column."""
        detected = detect_frequency(heterogeneous_coverage_panel['depenses_publiques_pib'],
                                    check_consistency=True, **options)
        assert detected == expected

    def test_index_frequency_per_entity(self, heterogeneous_coverage_panel):
        """Each entity index is monthly despite its own coverage and early annual anchors."""
        detected = detect_index_frequency(heterogeneous_coverage_panel.index, return_format='full')
        assert detected == {('Allemagne',): 'MS', ('France',): 'MS', ('Italie',): 'MS'}

    @pytest.mark.parametrize(
        'perturb',
        [
            pytest.param(lambda df: shuffle_rows(df, seed=11), id='shuffled-rows'),
            pytest.param(reverse_entities, id='reversed-entities'),
            pytest.param(lambda df: with_duplicated_rows(df, n=5), id='duplicated-rows'),
            pytest.param(lambda df: with_index_names(df, ['pays', 'période']), id='renamed-levels'),
        ],
    )
    def test_perturbation_keeps_the_frequencies(self, heterogeneous_coverage_panel, perturb):
        """Row order, duplicated rows and level names do not change any frequency."""
        assert detect_dataset_frequency(perturb(heterogeneous_coverage_panel)) == _heterogeneous_panel_frequencies()

    def test_three_level_index(self, heterogeneous_coverage_panel):
        """An outer ``region`` level is spliced in front of every key."""
        detected = detect_dataset_frequency(to_three_level_index(heterogeneous_coverage_panel))
        expected = {('Zone euro', *key): frequency
                    for key, frequency in _heterogeneous_panel_frequencies().items()}
        assert detected == expected

    def test_missing_entity(self, heterogeneous_coverage_panel):
        """Dropping an entity removes its pairs only."""
        detected = detect_dataset_frequency(drop_entity(heterogeneous_coverage_panel, 'Italie'))
        expected = {key: frequency for key, frequency in _heterogeneous_panel_frequencies().items()
                    if key[0] != 'Italie'}
        assert detected == expected

    def test_panel_columns(self, heterogeneous_coverage_panel):
        """Entity and date given as columns give the same map as the index."""
        flat = heterogeneous_coverage_panel.reset_index()
        detected = detect_dataset_frequency(flat, time_col='date', panel_cols=['country'])
        assert detected == _heterogeneous_panel_frequencies()
