"""Tests of the classification of covariates against a stage grid.

Covers ``classify`` — the classification made entity by entity, and
the four-rank precedence as one entity sees it — and ``decide_ways``
— one way per column, the most degraded of the per-entity verdicts,
decided without producing nor recording anything. Both read the per-entity
forms of ``detected_frequencies`` (a frequency belongs to the
``(entity, column)`` pair) and of ``stage_freq`` (layer and stage bindings).

Also covered: stage bindings resolved like ``StageScaler``:
a unanimous binding stands for the whole panel, a disagreeing one leaving a
grid entity unnamed is rejected; plain entity keys normalized; the
reduction never depends on the entity order: verdicts without a definable
conversion left out, equal ranks broken by a fixed order.

Covered symbols: ``CovariateMaterializer.classify``, ``decide_ways``.
"""
# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objet testé
from tsforecast.imputation._components.covariate_materializer import CovariateMaterializer

# Jeux et perturbations partagés
from tests.support.perturbations import reverse_entities
from tests.unit.imputation._components.covariate_materializer.helpers import (
    HETEROGENEOUS_FREQUENCIES,
    PANEL_F_FREQUENCIES,
    TS_FREQUENCIES,
    annual_grid,
    entity_grid,
    monthly_grid,
    prime,
    quarterly_grid,
)


# Configurations (stratégie, repli) éprouvées par les tables de classification
CONFIGS = [
    ('tolerate_nan', 'interpolate'),
    ('interpolate', 'interpolate'),
    ('model', 'interpolate'),
    ('model', 'tolerate_nan'),
]
CONFIG_IDS = ['tolerate_nan', 'interpolate', 'model-interpolate', 'model-tolerate_nan']


class TestClassifyTable:
    """Rank 1 whatever the strategy, the strategy below the grid."""

    @pytest.mark.parametrize('strategy, fallback', CONFIGS, ids=CONFIG_IDS)
    @pytest.mark.parametrize(
        'column, expected', [('q1', 'identity'), ('m1', 'aggregate')], ids=['equal', 'finer'],
    )
    def test_rank_one_ignores_the_strategy(self, strategy, fallback, column, expected):
        """A covariate at least as fine as the grid is identity or exact aggregation."""
        materializer = CovariateMaterializer(
            covariate_strategy=strategy, covariate_fallback=fallback
        )
        assert materializer.classify(column, 'Q', TS_FREQUENCIES) == expected

    @pytest.mark.parametrize(
        'strategy, fallback, expected',
        [
            ('tolerate_nan', 'interpolate', 'raw_anchors'),
            ('interpolate', 'interpolate', 'interpolate'),
            # Registres vides : la stratégie 'model' tombe au rang 4, son repli
            ('model', 'interpolate', 'interpolate'),
            ('model', 'tolerate_nan', 'raw_anchors'),
        ],
        ids=CONFIG_IDS,
    )
    def test_coarser_covariate_follows_the_strategy(self, strategy, fallback, expected):
        """A covariate coarser than the grid is served by the strategy."""
        materializer = CovariateMaterializer(
            covariate_strategy=strategy, covariate_fallback=fallback
        )
        assert materializer.classify('a1', 'Q', TS_FREQUENCIES) == expected

    @pytest.mark.parametrize('strategy, fallback', CONFIGS, ids=CONFIG_IDS)
    def test_column_without_detected_frequency_is_raw_anchors(self, strategy, fallback):
        """No conversion is definable without a frequency: the observations as they are."""
        materializer = CovariateMaterializer(
            covariate_strategy=strategy, covariate_fallback=fallback
        )
        assert materializer.classify('zz', 'Q', TS_FREQUENCIES) == 'raw_anchors'

    @pytest.mark.parametrize(
        'label, expected',
        [
            ('ME', 'aggregate'), ('MS', 'aggregate'), ('monthly', 'aggregate'),
            ('QE', 'identity'), ('QS', 'identity'), ('quarterly', 'identity'),
            ('YE', 'interpolate'), ('YS', 'interpolate'), ('annual', 'interpolate'),
        ],
        ids=lambda value: str(value),
    )
    def test_column_labels_are_compared_by_base_frequency(self, label, expected):
        """Start and end positions, and named aliases, denote the same frequency."""
        assert CovariateMaterializer().classify('x', 'Q', {'x': label}) == expected

    @pytest.mark.parametrize('stage', ['Q', 'QE', 'QS', 'quarterly'])
    def test_stage_labels_are_compared_by_base_frequency(self, stage):
        """The stage label is normalized like the column label."""
        assert CovariateMaterializer().classify('q1', stage, TS_FREQUENCIES) == 'identity'

    def test_unknown_frequency_label_raises(self):
        """An unrecognized label is an error, never a silent verdict."""
        with pytest.raises(ValueError, match='Unsupported frequency: zz'):
            CovariateMaterializer().classify('x', 'Q', {'x': 'zz'})


class TestClassifyPerEntity:
    """The classification is made for one entity at a time."""

    @pytest.mark.parametrize(
        'entity, stage, expected',
        [
            (('FR',), 'M', 'interpolate'),
            (('DE',), 'M', 'interpolate'),
            (('IT',), 'M', 'identity'),
            (('FR',), 'Q', 'interpolate'),
            (('DE',), 'Q', 'identity'),
            (('IT',), 'Q', 'aggregate'),
        ],
        ids=['FR-M', 'DE-M', 'IT-M', 'FR-Q', 'DE-Q', 'IT-Q'],
    )
    def test_per_entity_detected_frequency(self, entity, stage, expected):
        """PANEL-F: v annual for FR, quarterly for DE, monthly for IT."""
        materializer = CovariateMaterializer()
        assert materializer.classify('v', stage, PANEL_F_FREQUENCIES, entity) == expected

    def test_scalar_entity_argument_is_normalized(self):
        """``'IT'`` and ``('IT',)`` designate the same entity."""
        materializer = CovariateMaterializer()
        assert materializer.classify('v', 'Q', PANEL_F_FREQUENCIES, 'IT') == 'aggregate'

    def test_scalar_entity_argument_reads_a_tuple_keyed_stage_binding(self):
        """The entity argument is normalized for the stage binding too."""
        binding = {('FR',): 'Q', ('DE',): 'M'}
        assert CovariateMaterializer().classify('m1', binding, TS_FREQUENCIES, 'FR') == 'aggregate'

    def test_single_entry_frequency_mapping_stands_for_every_entity(self):
        """A per-entity mapping with one entry is the frequency of the whole panel."""
        frequencies = {'v': {('FR',): 'Y'}}
        assert CovariateMaterializer().classify('v', 'M', frequencies, ('DE',)) == 'interpolate'

    def test_entity_absent_from_a_disagreeing_mapping_has_no_frequency(self):
        """An entity that never observes the column has nothing to convert.

        Its own verdict is the raw anchors — which, for an entity without any
        observation, means NaN rows. Whether that verdict may weigh on the
        column's way is decided by the reduction of the verdicts.
        """
        frequencies = {'v': {('FR',): 'Y', ('DE',): 'Q'}}
        assert CovariateMaterializer().classify('v', 'M', frequencies, ('IT',)) == 'raw_anchors'

    @pytest.mark.parametrize(
        'entity, expected', [(('FR',), 'aggregate'), (('DE',), 'identity')], ids=['FR', 'DE'],
    )
    def test_per_entity_stage_binding(self, entity, expected):
        """The stage frequency is read for the entity classified (layer binding)."""
        binding = {('FR',): 'Q', ('DE',): 'M'}
        assert CovariateMaterializer().classify('m1', binding, TS_FREQUENCIES, entity) == expected

    def test_single_entry_stage_binding_stands_for_every_entity(self):
        """A stage binding with one entry is the stage frequency of the whole panel."""
        binding = {('FR',): 'Q'}
        assert CovariateMaterializer().classify('m1', binding, TS_FREQUENCIES, ('DE',)) == 'aggregate'

    @pytest.mark.parametrize(
        'entity, expected', [(('FR',), 'stage_model'), (('DE',), 'interpolate')], ids=['FR', 'DE'],
    )
    def test_model_reads_the_entity_own_mirror(self, entity, expected):
        """Under ``'model'``, only the entity's own mirror can grant rank 2."""
        materializer = CovariateMaterializer(covariate_strategy='model')
        # Miroir de a1 au pas M pour la seule entité FR
        french = entity_grid(['FR'], monthly_grid())
        prime(materializer, 'a1', pd.Series(7.0, index=french), 'M', 'model')
        assert materializer.classify('a1', 'M', TS_FREQUENCIES, entity) == expected


class TestDecideWays:
    """One way per column, decided once, without producing anything."""

    def test_one_way_per_column_in_column_order(self):
        """One entry per column, in the order of ``columns``, never one per entity."""
        ways = CovariateMaterializer().decide_ways(
            columns=['v', 'q1', 'm1'],
            grid_index=entity_grid(['FR', 'DE', 'IT'], monthly_grid()),
            stage_freq='M',
            detected_frequencies=PANEL_F_FREQUENCIES,
        )
        assert list(ways.items()) == [('v', 'interpolate'), ('q1', 'interpolate'), ('m1', 'identity')]

    @pytest.mark.parametrize(
        'strategy, stage, dates, expected',
        [
            # FR interpolate, DE identity, IT aggregate : l'interpolation domine
            ('interpolate', 'Q', quarterly_grid(), 'interpolate'),
            # FR raw_anchors, DE identity, IT aggregate : les ancres dominent
            ('tolerate_nan', 'Q', quarterly_grid(), 'raw_anchors'),
            # FR identity, DE aggregate, IT aggregate : l'agrégation domine
            ('interpolate', 'Y', annual_grid(), 'aggregate'),
        ],
        ids=['Q-interpolate', 'Q-tolerate_nan', 'Y-interpolate'],
    )
    def test_reduction_keeps_the_most_degraded_verdict(self, strategy, stage, dates, expected):
        """PANEL-F ``v``: the per-entity verdicts reduce to the most degraded one."""
        ways = CovariateMaterializer(covariate_strategy=strategy).decide_ways(
            columns=['v'],
            grid_index=entity_grid(['FR', 'DE', 'IT'], dates),
            stage_freq=stage,
            detected_frequencies=PANEL_F_FREQUENCIES,
        )
        assert ways == {'v': expected}

    def test_decision_produces_and_records_nothing(self):
        """``decide_ways`` writes none of the three stores."""
        materializer = CovariateMaterializer(covariate_strategy='model')
        materializer.decide_ways(
            columns=['m1', 'q1', 'a1'],
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
        )
        assert (
            materializer.imputed_store, materializer.imputed_freq_store, materializer.origin_store,
        ) == ({}, {}, {})

    def test_reversed_entities_keep_the_ways(self, heterogeneous_coverage_panel):
        """Realistic panel: the entity order of the grid changes no way."""
        data = heterogeneous_coverage_panel
        materializer = CovariateMaterializer()
        kwargs = dict(
            columns=list(HETEROGENEOUS_FREQUENCIES),
            stage_freq='M',
            detected_frequencies=HETEROGENEOUS_FREQUENCIES,
        )
        ways = materializer.decide_ways(grid_index=data.index, **kwargs)
        reversed_ways = materializer.decide_ways(grid_index=reverse_entities(data).index, **kwargs)
        assert reversed_ways == ways

    def test_start_and_end_anchored_grids_decide_alike(self):
        """The position of the grid dates plays no part in the decision."""
        materializer = CovariateMaterializer()
        kwargs = dict(columns=['m1', 'q1', 'a1'], stage_freq='Q', detected_frequencies=TS_FREQUENCIES)
        start = materializer.decide_ways(
            grid_index=pd.date_range('2021-01-01', periods=12, freq='QS'), **kwargs
        )
        end = materializer.decide_ways(grid_index=quarterly_grid(), **kwargs)
        assert start == end == {'m1': 'aggregate', 'q1': 'identity', 'a1': 'interpolate'}

    def test_empty_time_series_grid_still_decides(self):
        """An empty series grid keeps its degenerate entity: the ways are decided."""
        ways = CovariateMaterializer().decide_ways(
            columns=['m1', 'a1'],
            grid_index=monthly_grid()[:0],
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
        )
        assert ways == {'m1': 'identity', 'a1': 'interpolate'}


class TestStageBindingResolution:
    """A stage binding is resolved like ``StageScaler`` resolves it.

    ``StageScaler._stage_freq_for``: an
    own entry, a single-entry binding, then a UNANIMOUS binding stands for the
    whole panel; only a disagreeing binding leaving a grid entity uncovered
    raises, naming the entity. ``HighFrequencyImputer`` hands such partial
    bindings over on grids covering every entity.
    """

    def test_unanimous_stage_binding_covers_an_unbound_grid_entity(
        self, mixed_freq_panel_heterogeneous
    ):
        """IT, absent from ``{FR: Q, DE: Q}``, is served at the stage frequency Q.

        Valeur d'or : m1 vaut 100 + rang pour toutes les entités ; sa somme
        exacte au T1 2021 vaut 100 + 101 + 102 = 303, et la voie de la colonne
        reste l'agrégation de rang 1.
        """
        features, ways, _origins = CovariateMaterializer().materialize(
            columns=['m1'],
            grid_index=entity_grid(['FR', 'DE', 'IT'], quarterly_grid()),
            stage_freq={('FR',): 'Q', ('DE',): 'Q'},
            detected_frequencies=TS_FREQUENCIES,
            source_data=mixed_freq_panel_heterogeneous,
        )
        assert (ways, features.loc['IT', 'm1'].iloc[0]) == ({'m1': 'aggregate'}, 303.0)

    def test_disagreeing_stage_binding_omitting_a_grid_entity_raises(
        self, mixed_freq_panel_heterogeneous
    ):
        """``{FR: M, DE: Q}`` cannot tell IT's stage: an error names IT.

        Run under ``'model'`` with a mirror for IT: the binding is rejected
        before any read of the stores.
        """
        materializer = CovariateMaterializer(covariate_strategy='model')
        # Miroir de a1 au pas M pour FR et IT
        for entity in ('FR', 'IT'):
            prime(materializer, 'a1', pd.Series(7.0, index=entity_grid([entity], monthly_grid())), 'M', 'model')
        with pytest.raises(ValueError, match='IT'):
            materializer.materialize(
                columns=['a1'],
                grid_index=entity_grid(['FR', 'DE', 'IT'], monthly_grid()),
                stage_freq={('FR',): 'M', ('DE',): 'Q'},
                detected_frequencies=TS_FREQUENCIES,
                source_data=mixed_freq_panel_heterogeneous,
            )

    def test_disagreeing_stage_binding_is_rejected_by_decide_ways(self):
        """The decision walks the grid too: an unnamed entity is rejected there."""
        with pytest.raises(ValueError, match=r"\('IT',\)"):
            CovariateMaterializer().decide_ways(
                columns=['m1'],
                grid_index=entity_grid(['FR', 'DE', 'IT'], monthly_grid()),
                stage_freq={('FR',): 'M', ('DE',): 'Q'},
                detected_frequencies=TS_FREQUENCIES,
            )

    def test_unreadable_stage_label_raises(self):
        """An unanimous binding on an unrecognized label is an error, never a silent rank."""
        binding = {('FR',): 'zz', ('DE',): 'zz'}
        with pytest.raises(ValueError, match='Unsupported frequency: zz'):
            CovariateMaterializer().classify('m1', binding, TS_FREQUENCIES, ('IT',))

    def test_unanimous_binding_up_to_position_aliases(self):
        """``{FR: 'M', IT: 'ME'}`` agrees on the month: DE is at stage M too."""
        binding = {('FR',): 'M', ('IT',): 'ME'}
        assert CovariateMaterializer().classify('m1', binding, TS_FREQUENCIES, ('DE',)) == 'identity'

    @pytest.mark.parametrize(
        'mapping', ['detected_frequencies', 'stage_freq'], ids=['detected', 'stage'],
    )
    def test_plain_entity_keys_are_normalized(self, mapping):
        """``{'FR': ...}`` behaves like ``{('FR',): ...}``, as the entity argument does.

        Scalar keys are normalized, like in the other functions of the package.
        """
        materializer = CovariateMaterializer()
        grid = entity_grid(['FR', 'DE'], quarterly_grid())
        # Deux formes du même réglage : clés tuples et clés scalaires
        if mapping == 'detected_frequencies':
            column = 'v'
            forms = [{'v': {('FR',): 'Y', ('DE',): 'Q'}}, {'v': {'FR': 'Y', 'DE': 'Q'}}]
            calls = [dict(stage_freq='Q', detected_frequencies=form) for form in forms]
        else:
            column = 'm1'
            forms = [{('FR',): 'Q', ('DE',): 'M'}, {'FR': 'Q', 'DE': 'M'}]
            calls = [dict(stage_freq=form, detected_frequencies=TS_FREQUENCIES) for form in forms]
        decided = [
            materializer.decide_ways(columns=[column], grid_index=grid, **call) for call in calls
        ]
        assert decided[1] == decided[0]


class TestReductionOrder:
    """The reduction of the verdicts never depends on the entity order."""

    def test_entity_without_frequency_never_dictates_the_column_way(self):
        """IT never observes ``v``: listed first, it must not strip FR / DE of interpolation.

        IT, without a frequency, has no definable conversion: its verdict
        ('raw_anchors', of the same rank as 'interpolate') does not enter the
        reduction. Under 'interpolate', the only residual NaN are those of an
        entity without the column.
        """
        frequencies = {'v': {('FR',): 'Y', ('DE',): 'Q'}}
        materializer = CovariateMaterializer(covariate_strategy='interpolate')
        decided = [
            materializer.decide_ways(
                columns=['v'],
                grid_index=entity_grid(order, monthly_grid()),
                stage_freq='M',
                detected_frequencies=frequencies,
            )
            for order in (['FR', 'DE', 'IT'], ['IT', 'FR', 'DE'])
        ]
        assert decided == [{'v': 'interpolate'}, {'v': 'interpolate'}]

    def test_tied_model_verdicts_never_skip_rank_two(self, mixed_freq_panel_heterogeneous):
        """FR holds cells at the stage: it reads them, whichever entity comes first.

        FR : cellules d'étape M en 2022-2023 (1.0) et cellules Q en 2021 (99.0)
        — rang 2 applicable ; DE : cellules Q seules (5.0) — rang 3. Les deux
        verdicts partagent le rang 3 de ``_WAY_RANK`` : quelle que soit la voie
        de modèle retenue pour la colonne, FR suit sa propre précédence et lit
        les cellules de son étape.
        """
        materializer = CovariateMaterializer(covariate_strategy='model')
        quarters, months = quarterly_grid(), monthly_grid()
        prime(materializer, 'a1', pd.Series(99.0, index=entity_grid(['FR'], quarters[quarters.year == 2021])), 'Q', 'model')
        prime(materializer, 'a1', pd.Series(1.0, index=entity_grid(['FR'], months[months.year >= 2022])), 'M', 'model')
        prime(materializer, 'a1', pd.Series(5.0, index=entity_grid(['DE'], quarters)), 'Q', 'model')

        features, _ways, _origins = materializer.materialize(
            columns=['a1'],
            grid_index=entity_grid(['DE', 'FR'], months),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=mixed_freq_panel_heterogeneous,
            record=False,
        )
        np.testing.assert_array_equal(features.loc['FR', 'a1'].loc['2022'].to_numpy(), np.ones(12))

    def test_column_no_entity_can_convert_is_raw_anchors(self):
        """Without any detected frequency, the column is read as it is."""
        ways = CovariateMaterializer().decide_ways(
            columns=['zz'],
            grid_index=entity_grid(['FR', 'DE'], monthly_grid()),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
        )
        assert ways == {'zz': 'raw_anchors'}

    def test_tied_model_verdicts_reduce_to_one_way_in_any_order(self):
        """FR at rank 2, DE at rank 3: the column way is the same in both orders.

        Rangs égaux départagés par un ordre fixe : le report d'une étape
        antérieure, moins direct, l'emporte.
        """
        materializer = CovariateMaterializer(covariate_strategy='model')
        prime(materializer, 'a1', pd.Series(1.0, index=entity_grid(['FR'], monthly_grid())), 'M', 'model')
        prime(materializer, 'a1', pd.Series(5.0, index=entity_grid(['DE'], quarterly_grid())), 'Q', 'model')
        decided = [
            materializer.decide_ways(
                columns=['a1'],
                grid_index=entity_grid(order, monthly_grid()),
                stage_freq='M',
                detected_frequencies=TS_FREQUENCIES,
            )
            for order in (['FR', 'DE'], ['DE', 'FR'])
        ]
        assert decided == [{'a1': 'carried_model'}, {'a1': 'carried_model'}]
