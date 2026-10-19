"""Tests of ``covariate_eligibility``: a feature absent for a whole entity.

No strategy can
fabricate values for an entity holding NO observation of a feature
(``climat_affaires`` for ``IT``). ``'any_entity'`` keeps the column as
soon as one entity observes it — the empty entity's rows stay NaN, the unique
residual NaN under ``'interpolate'``; ``'all_entities'`` drops it. A column
observed by no entity at all, or absent from the frame, is never retained.
Tested on ``PANEL`` and on the realistic ``heterogeneous_coverage_panel``.

Covered symbols: ``CovariateMaterializer.eligible_columns``,
``entities_without_column``.
"""
# Calcul numérique
import numpy as np

# Cadre de test
import pytest

# Objet testé
from tsforecast.frequency.covariate_materializer import CovariateMaterializer

# Jeux et perturbations partagés
from tests.support.perturbations import (
    reverse_entities,
    shuffle_rows,
    to_three_level_index,
    with_special_column_names,
)
from tests.unit.frequency.covariate_materializer.helpers import (
    HETEROGENEOUS_FREQUENCIES,
    PANEL_FREQUENCIES,
    monthly_grid,
    panel_grid,
)


# Colonnes de heterogeneous_coverage_panel, dans l'ordre des fréquences
HETEROGENEOUS_COLUMNS = list(HETEROGENEOUS_FREQUENCIES)


class TestEligibleColumns:
    """``eligible_columns``: the columns admissible as features."""

    def test_covariate_eligibility_any_vs_all(self, mixed_freq_panel_heterogeneous):
        """``'any_entity'`` keeps climat_affaires; ``'all_entities'`` drops it."""
        data = mixed_freq_panel_heterogeneous
        columns = ['m1', 'q1', 'climat_affaires']

        any_entity = CovariateMaterializer(covariate_eligibility='any_entity')
        all_entities = CovariateMaterializer(covariate_eligibility='all_entities')

        # Recensement des entités qui n'observent jamais la colonne
        assert any_entity.entities_without_column('climat_affaires', data) == (('IT',),)
        assert any_entity.entities_without_column('m1', data) == ()

        # 'any_entity' retient la colonne, 'all_entities' l'écarte
        assert any_entity.eligible_columns(columns, data) == tuple(columns)
        assert all_entities.eligible_columns(columns, data) == ('m1', 'q1')

        # Sous 'any_entity', les lignes de l'entité vide restent NaN
        features, _ways, _origins = any_entity.materialize(
            columns=list(any_entity.eligible_columns(columns, data)),
            grid_index=panel_grid(data, monthly_grid()),
            stage_freq='M',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
        )
        assert features.loc['IT', 'climat_affaires'].isna().all()
        assert features.loc['FR', 'climat_affaires'].notna().all()

    @pytest.mark.parametrize(
        'eligibility, expected',
        [
            ('any_entity', tuple(HETEROGENEOUS_COLUMNS)),
            # climat_affaires écartée ; depenses_publiques_pib, à fréquence
            # hétérogène mais observée partout, retenue
            ('all_entities', tuple(c for c in HETEROGENEOUS_COLUMNS if c != 'climat_affaires')),
        ],
    )
    def test_realistic_panel(self, heterogeneous_coverage_panel, eligibility, expected):
        """heterogeneous_coverage_panel: only climat_affaires (Italie) is at stake."""
        materializer = CovariateMaterializer(covariate_eligibility=eligibility)
        assert materializer.eligible_columns(HETEROGENEOUS_COLUMNS, heterogeneous_coverage_panel) == expected

    @pytest.mark.parametrize('eligibility', ['any_entity', 'all_entities'])
    def test_column_observed_by_no_entity_is_never_retained(self, mixed_freq_panel_heterogeneous, eligibility):
        """An entirely empty column is dropped whatever the modality."""
        data = mixed_freq_panel_heterogeneous.copy()
        data['climat_affaires'] = np.nan
        materializer = CovariateMaterializer(covariate_eligibility=eligibility)
        assert materializer.eligible_columns(['m1', 'climat_affaires'], data) == ('m1',)

    @pytest.mark.parametrize('eligibility', ['any_entity', 'all_entities'])
    def test_absent_column_is_never_retained(self, mixed_freq_panel_heterogeneous, eligibility):
        """A column missing from the frame is skipped, never an error."""
        materializer = CovariateMaterializer(covariate_eligibility=eligibility)
        assert materializer.eligible_columns(['zz', 'm1'], mixed_freq_panel_heterogeneous) == ('m1',)

    def test_order_follows_the_given_columns(self, mixed_freq_panel_heterogeneous):
        """The retained columns keep the order of ``columns``, not that of the frame."""
        materializer = CovariateMaterializer()
        assert materializer.eligible_columns(['q1', 'a2', 'm1'], mixed_freq_panel_heterogeneous) == (
            'q1', 'a2', 'm1',
        )

    def test_time_series_drops_its_empty_column(self, reference_timeseries):
        """A time series is one entity: an empty column is observed by nobody."""
        blind = reference_timeseries.copy()
        blind['a1'] = np.nan
        assert CovariateMaterializer().eligible_columns(['m1', 'a1'], blind) == ('m1',)

    def test_shuffled_panel_gives_the_same_columns(self, heterogeneous_coverage_panel):
        """The row order plays no part in the eligibility."""
        materializer = CovariateMaterializer(covariate_eligibility='all_entities')
        reference = materializer.eligible_columns(HETEROGENEOUS_COLUMNS, heterogeneous_coverage_panel)
        shuffled = materializer.eligible_columns(
            HETEROGENEOUS_COLUMNS, shuffle_rows(heterogeneous_coverage_panel, seed=2)
        )
        assert shuffled == reference


class TestEntitiesWithoutColumn:
    """``entities_without_column``: the entities holding no observation at all."""

    def test_realistic_panel_lists_italy(self, heterogeneous_coverage_panel):
        """climat_affaires is structurally absent for Italie only."""
        assert CovariateMaterializer().entities_without_column(
            'climat_affaires', heterogeneous_coverage_panel
        ) == (('Italie',),)

    @pytest.mark.parametrize(
        'column', [c for c in HETEROGENEOUS_COLUMNS if c != 'climat_affaires'],
    )
    def test_observed_columns_list_nobody(self, heterogeneous_coverage_panel, column):
        """Every other column, sparse or of heterogeneous frequency, is observed by all."""
        assert CovariateMaterializer().entities_without_column(column, heterogeneous_coverage_panel) == ()

    @pytest.mark.parametrize(
        'empty, expected', [(False, ()), (True, ((),))], ids=['observed', 'empty'],
    )
    def test_time_series(self, reference_timeseries, empty, expected):
        """A time series lists its degenerate entity ``()`` when the column is empty."""
        data = reference_timeseries.copy()
        if empty:
            data['a1'] = np.nan
        assert CovariateMaterializer().entities_without_column('a1', data) == expected

    def test_empty_column_lists_every_entity_in_data_order(self, mixed_freq_panel_heterogeneous):
        """A column observed by nobody lists every entity, in the data's order."""
        data = mixed_freq_panel_heterogeneous.copy()
        data['climat_affaires'] = np.nan
        assert CovariateMaterializer().entities_without_column('climat_affaires', data) == (
            ('DE',), ('FR',), ('IT',),
        )

    def test_order_follows_the_data_entity_order(self, mixed_freq_panel_heterogeneous):
        """Reversed entities, reversed listing: the data order, never a sorted one."""
        data = mixed_freq_panel_heterogeneous.copy()
        data['climat_affaires'] = np.nan
        assert CovariateMaterializer().entities_without_column(
            'climat_affaires', reverse_entities(data)
        ) == (('IT',), ('FR',), ('DE',))

    def test_three_level_index_keys(self, mixed_freq_panel_heterogeneous):
        """With an outer region level, the entity key holds two elements."""
        data = to_three_level_index(mixed_freq_panel_heterogeneous)
        assert CovariateMaterializer().entities_without_column('climat_affaires', data) == (
            ('Zone euro', 'IT'),
        )

    def test_special_column_name(self, mixed_freq_panel_heterogeneous):
        """Spaces, accents and symbols in the column name are plain labels."""
        data, mapping = with_special_column_names(mixed_freq_panel_heterogeneous)
        assert CovariateMaterializer().entities_without_column(mapping['climat_affaires'], data) == (
            ('IT',),
        )

    def test_missing_column_raises_key_error(self, mixed_freq_panel_heterogeneous):
        """A column absent from the frame is an error, unlike in ``eligible_columns``."""
        with pytest.raises(KeyError, match='zz'):
            CovariateMaterializer().entities_without_column('zz', mixed_freq_panel_heterogeneous)
