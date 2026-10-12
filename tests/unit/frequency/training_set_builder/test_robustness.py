"""Robustness of ``TrainingSetBuilder`` on realistic data and perturbed indexes.

Realistic datasets of notebook 3: ``heterogeneous_coverage_panel``
(``depenses_publiques_pib`` blocks ``{France: Y, Allemagne: Q, Italie: Y}`` on
per-entity coverages, per-row target divisors equal to the block divisors of
``StageScaler``, feature divisors equal to the target divisor row by row,
``climat_affaires`` structurally absent for Italie left NaN) and
``irregular_index_timeseries`` (annual anchors before the monthly grid kept as
rows, their covariates NaN).

Index robustness: ``shuffle_rows`` / ``reverse_entities`` give the same rows
up to the order (stamped set included), three-level panel (plain and
stamped), non-standard level names, the reserved level name ``'frequency'``
rejected whatever the configuration, start positions, special column names, duplicated dates rejected. Wiring:
log callback and representation.

Covered symbols: ``TrainingSetBuilder`` (``build``, ``__repr__``),
``split_training_index``.
"""
# Modules de base
from typing import List

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objets testés
from tsforecast.frequency.training_set_builder import (
    FREQUENCY_LEVEL_NAME,
    TrainingSet,
    split_training_index,
)
from tsforecast.frequency.stage_scaler import StageScaler

# Perturbations partagées
from tests.support.perturbations import (
    reverse_entities,
    shuffle_rows,
    to_three_level_index,
    with_duplicated_rows,
    with_index_names,
    with_special_column_names,
)

# Constantes et fabriques partagées du paquet
from tests.unit.frequency.training_set_builder.helpers import (
    HETEROGENEOUS_FEATURES,
    HETEROGENEOUS_FREQUENCIES,
    PANEL_F_BLOCKS,
    PANEL_F_FREQUENCIES,
    assert_same_rows,
    build_panel_f,
    build_ts,
    make_builder,
    panel_f_at_period_start,
    seed_quarterly_mirror,
)


# =============================================================================
# Jeux réalistes
# =============================================================================
class TestRealisticDatasets:
    """Notebook-3 datasets: per-entity frequencies, coverages and absences."""

    @staticmethod
    def _build_depenses(builder, data, stage='M') -> TrainingSet:
        """Compose the training set of ``depenses_publiques_pib``."""
        return builder.build(
            column='depenses_publiques_pib',
            feature_cols=HETEROGENEOUS_FEATURES,
            stage_freq=stage,
            detected_frequencies=HETEROGENEOUS_FREQUENCIES,
            source_data=data,
            eligible_origins={'observed'},
        )

    @pytest.mark.parametrize('stage', ['Q', 'M'])
    def test_depenses_blocks_and_rows(self, heterogeneous_coverage_panel, stage):
        """Blocks ``{France: Y, Allemagne: Q, Italie: Y}`` and 6 + 23 + 5 rows."""
        data = heterogeneous_coverage_panel
        training = self._build_depenses(make_builder(), data, stage)

        assert dict(training.blocks) == {
            ('Allemagne',): 'Q', ('France',): 'Y', ('Italie',): 'Y'
        }
        # Valeur d'or : le décompte des publications de chaque pays, sur SA
        # couverture (dernière publication manquante)
        assert training.y.groupby(level=0).size().to_dict() == {
            'Allemagne': 23, 'France': 6, 'Italie': 5
        }
        # Propriété : les lignes sont exactement les publications observées
        observed = data['depenses_publiques_pib'].dropna()
        assert training.y.sort_index().equals(observed.sort_index())

    @pytest.mark.parametrize(
        'stage, expected',
        [('M', {'Allemagne': 3.0, 'France': 12.0, 'Italie': 12.0}),
         ('Q', {'Allemagne': 1.0, 'France': 4.0, 'Italie': 4.0})],
        ids=['M', 'Q'],
    )
    def test_depenses_target_divisors(self, heterogeneous_coverage_panel, stage, expected):
        """Per-row divisors equal the block divisors of ``StageScaler`` (R5)."""
        training = self._build_depenses(make_builder(), heterogeneous_coverage_panel, stage)
        scaler = StageScaler()

        per_row = scaler.target_divisor(
            'depenses_publiques_pib', source_freq=dict(training.blocks),
            pred_freq=stage, index=training.y.index,
            produced_freq=training.row_frequency,
        )
        per_block = scaler.target_divisor(
            'depenses_publiques_pib', source_freq=dict(training.blocks),
            pred_freq=stage, index=training.y.index,
        )

        # Deux chemins, un seul diviseur : la fréquence de ligne d'une cellule
        # observée EST la fréquence de son bloc
        pd.testing.assert_series_equal(per_row, per_block, check_names=False)
        assert {
            entity: set(values) for entity, values in per_row.groupby(level=0)
        } == {entity: {value} for entity, value in expected.items()}

    @pytest.mark.parametrize('stage', ['Q', 'M'])
    def test_features_and_target_share_the_stage_scale(
        self, heterogeneous_coverage_panel, stage
    ):
        """Feature divisors equal the target divisor on every row (§9.2)."""
        training = self._build_depenses(make_builder(), heterogeneous_coverage_panel, stage)
        scaler = StageScaler()

        target = scaler.target_divisor(
            'depenses_publiques_pib', source_freq=dict(training.blocks),
            pred_freq=stage, index=training.y.index,
            produced_freq=training.row_frequency,
        )
        features = scaler.feature_divisors(
            columns=HETEROGENEOUS_FEATURES,
            column_frequencies=HETEROGENEOUS_FREQUENCIES,
            ways=dict(training.ways),
            grid_freq=dict(training.blocks),
            stage_freq=stage,
            index=training.X.index,
        )

        # Toute cellule de covariable couvre une période de la grille du bloc,
        # comme la cible : un seul diviseur par ligne
        for column in HETEROGENEOUS_FEATURES:
            np.testing.assert_allclose(
                features[column].to_numpy(), target.to_numpy(), rtol=1e-12
            )

    def test_structurally_absent_covariate_stays_nan_for_its_entity(
        self, heterogeneous_coverage_panel
    ):
        """``climat_affaires`` is NaN on Italie's rows only, never invented."""
        training = self._build_depenses(make_builder(), heterogeneous_coverage_panel)
        missing = training.X['climat_affaires'].isna().groupby(level=0).sum()

        assert missing.to_dict() == {'Allemagne': 0, 'France': 0, 'Italie': 5}

    def test_irregular_index_timeseries_keeps_every_anchor(
        self, irregular_index_timeseries
    ):
        """Annual anchors before the monthly grid are rows, their covariates NaN."""
        data = irregular_index_timeseries
        training = make_builder().build(
            column='balance_commerciale_annuelle',
            feature_cols=['inflation_ipc', 'pib_trimestriel'],
            stage_freq='M',
            detected_frequencies={
                'balance_commerciale_annuelle': 'Y',
                'inflation_ipc': 'M',
                'pib_trimestriel': 'Q',
            },
            source_data=data,
            eligible_origins={'observed'},
        )

        # Sans fenêtre, toutes les ancres sont des lignes, 2015-2017 comprises
        observed = data['balance_commerciale_annuelle'].dropna()
        assert training.y.equals(observed)
        # Avant 2018, aucune covariable n'existe : NaN, jamais extrapolée
        before_grid = training.X.index < pd.Timestamp('2018-01-01')
        assert before_grid.sum() == 3
        assert training.X[before_grid].isna().all().all()
        assert training.X[~before_grid].notna().all().all()


# =============================================================================
# Robustesse des index
# =============================================================================
class TestIndexRobustness:
    """Unsorted data, entity order, index shapes, positions and names."""

    @pytest.mark.parametrize(
        'perturbation',
        [lambda df: shuffle_rows(df, seed=3), reverse_entities],
        ids=['shuffle_rows', 'reverse_entities'],
    )
    def test_panel_f_perturbation_yields_the_same_rows(
        self, mixed_freq_panel_multifrequency, perturbation
    ):
        """A row or entity permutation changes nothing but the order."""
        reference = build_panel_f(make_builder(), mixed_freq_panel_multifrequency)

        perturbed = build_panel_f(make_builder(), perturbation(mixed_freq_panel_multifrequency))

        assert_same_rows(perturbed, reference)

    @pytest.mark.parametrize(
        'perturbation',
        [lambda df: shuffle_rows(df, seed=7), reverse_entities],
        ids=['shuffle_rows', 'reverse_entities'],
    )
    def test_realistic_perturbation_yields_the_same_rows(
        self, heterogeneous_coverage_panel, perturbation
    ):
        """Same property on the realistic panel, structural absence included."""
        build = TestRealisticDatasets._build_depenses
        reference = build(make_builder(), heterogeneous_coverage_panel)

        perturbed = build(make_builder(), perturbation(heterogeneous_coverage_panel))

        assert_same_rows(perturbed, reference)

    def test_shuffled_time_series_with_coincidences(self, reference_timeseries):
        """A shuffled ``TS`` gives the same stamped set, up to the order."""
        reference = build_ts(
            seed_quarterly_mirror(make_builder(), 'a1'), reference_timeseries,
            keep_coincident_cells=True,
        )

        perturbed = build_ts(
            seed_quarterly_mirror(make_builder(), 'a1'), shuffle_rows(reference_timeseries, seed=1),
            keep_coincident_cells=True,
        )

        assert perturbed.has_frequency_level
        assert_same_rows(perturbed, reference)

    def test_three_level_panel(self, mixed_freq_panel_multifrequency):
        """Entity keys are the two outer levels, values unchanged."""
        data = to_three_level_index(mixed_freq_panel_multifrequency)
        frequencies = {
            **PANEL_F_FREQUENCIES,
            'v': {('Zone euro', entity[0]): freq for entity, freq in PANEL_F_BLOCKS.items()},
        }
        reference = build_panel_f(make_builder(), mixed_freq_panel_multifrequency)

        training = build_panel_f(make_builder(), data, frequencies=frequencies)

        assert dict(training.blocks) == {
            ('Zone euro', 'FR'): 'Y', ('Zone euro', 'DE'): 'Q', ('Zone euro', 'IT'): 'M'
        }
        assert list(training.X.index.names) == ['region', 'country', 'date']
        np.testing.assert_array_equal(training.y.to_numpy(), reference.y.to_numpy())
        np.testing.assert_allclose(training.X.to_numpy(), reference.X.to_numpy())

    def test_three_level_panel_stamped(self, mixed_freq_panel_multifrequency):
        """Stamped three-level grid: ``(region, country, frequency, date)``."""
        data = to_three_level_index(mixed_freq_panel_multifrequency)
        frequencies = {
            **PANEL_F_FREQUENCIES,
            'v': {('Zone euro', entity[0]): freq for entity, freq in PANEL_F_BLOCKS.items()},
        }
        builder = make_builder()
        index = pd.MultiIndex.from_tuples(
            [('Zone euro', 'FR', pd.Timestamp('2021-12-31'))],
            names=['region', 'country', 'date'],
        )
        builder.materializer.record_production(
            'v', pd.Series([31.0], index=index),
            pd.Series(['model'], index=index), pd.Series(['Q'], index=index),
        )

        training = build_panel_f(
            builder, data, frequencies=frequencies,
            eligible_origins=('observed', 'model'), keep_coincident_cells=True,
        )

        assert list(training.X.index.names) == [
            'region', 'country', FREQUENCY_LEVEL_NAME, 'date'
        ]
        _real, entities = split_training_index(training.X.index, has_frequency_level=True)
        assert set(entities) == set(training.blocks)
        assert len(training) == 52

    def test_non_standard_index_names(self, mixed_freq_panel_multifrequency):
        """Level names are reproduced, the frequency level inserted before the date."""
        data = with_index_names(mixed_freq_panel_multifrequency, ['pays', 'periode'])
        builder = make_builder()
        index = pd.MultiIndex.from_tuples(
            [('FR', pd.Timestamp('2021-12-31'))], names=['pays', 'periode']
        )
        builder.materializer.record_production(
            'v', pd.Series([31.0], index=index),
            pd.Series(['model'], index=index), pd.Series(['Q'], index=index),
        )

        plain = build_panel_f(make_builder(), data)
        stamped = build_panel_f(
            builder, data, eligible_origins=('observed', 'model'), keep_coincident_cells=True,
        )

        assert list(plain.X.index.names) == ['pays', 'periode']
        assert list(stamped.X.index.names) == ['pays', FREQUENCY_LEVEL_NAME, 'periode']

    @pytest.mark.parametrize(
        'names', [[FREQUENCY_LEVEL_NAME, 'date'], ['country', FREQUENCY_LEVEL_NAME]],
        ids=['entity_level', 'date_level'],
    )
    @pytest.mark.parametrize('keep_coincident_cells', [False, True], ids=['sum', 'None'])
    def test_reserved_level_name_is_rejected(
        self, mixed_freq_panel_multifrequency, names, keep_coincident_cells
    ):
        """A level named like the frequency level is rejected, whatever the configuration.

        The stacked grid would otherwise hold two levels of one name; the
        rejection does not depend on ``keep_coincident_cells``, so the same
        data is never accepted by one configuration and refused by another.
        """
        data = with_index_names(mixed_freq_panel_multifrequency, names)

        with pytest.raises(ValueError, match="'frequency' is reserved"):
            build_panel_f(
                make_builder(), data, keep_coincident_cells=keep_coincident_cells
            )

    def test_start_positions(self, mixed_freq_panel_multifrequency):
        """Start-anchored ``PANEL-F``: same rows and values, block covariates equal."""
        reference = build_panel_f(make_builder(), mixed_freq_panel_multifrequency)

        training = build_panel_f(
            make_builder(), panel_f_at_period_start(mixed_freq_panel_multifrequency)
        )

        assert dict(training.blocks) == PANEL_F_BLOCKS
        np.testing.assert_array_equal(training.y.to_numpy(), reference.y.to_numpy())
        # Ancres de début de période : 1er janvier pour FR, début de trimestre
        # pour DE
        assert training.y.loc['FR'].index[0] == pd.Timestamp('2021-01-01')
        assert training.y.loc['DE'].index[0] == pd.Timestamp('2021-01-01')
        # Agrégations et identités ne dépendent pas de la position : m1 = 1266
        # sur l'année de FR, 303 sur le T1 de DE (seule l'interpolation de q1
        # chez IT dépend légitimement de la position de ses ancres)
        for entity in ('FR', 'DE'):
            np.testing.assert_allclose(
                training.X.loc[entity].to_numpy(), reference.X.loc[entity].to_numpy()
            )

    def test_special_column_names(self, mixed_freq_panel_multifrequency):
        """Spaces, accents and symbols in column names change nothing."""
        data, mapping = with_special_column_names(mixed_freq_panel_multifrequency)
        frequencies = {mapping[name]: freq for name, freq in PANEL_F_FREQUENCIES.items()}
        reference = build_panel_f(make_builder(), mixed_freq_panel_multifrequency)

        training = build_panel_f(
            make_builder(), data, frequencies=frequencies, column=mapping['v'],
            feature_cols=[mapping['m1'], mapping['q1']],
        )

        assert training.y.name == mapping['v']
        assert list(training.X.columns) == [mapping['m1'], mapping['q1']]
        assert dict(training.ways) == {mapping['m1']: 'aggregate', mapping['q1']: 'interpolate'}
        np.testing.assert_allclose(training.X.to_numpy(), reference.X.to_numpy())

    def test_duplicated_dates_raise(self, mixed_freq_panel_multifrequency):
        """A duplicated (entity, date) is rejected with an explicit message."""
        data = with_duplicated_rows(mixed_freq_panel_multifrequency)

        with pytest.raises(ValueError, match='Duplicate dates in source_data'):
            build_panel_f(make_builder(), data)


# =============================================================================
# Journalisation et représentation
# =============================================================================
class TestWiring:
    """Injected log callback and readable representation."""

    def test_log_line_names_the_stage_and_the_blocks(self, mixed_freq_panel_multifrequency):
        """One composition line per ``build``, stage and row count included."""
        logged: List[str] = []

        build_panel_f(
            make_builder(log=logged.append), mixed_freq_panel_multifrequency, stage_freq='Q'
        )

        assert len(logged) == 1
        assert "'v' at stage 'Q' : 51 rows" in logged[0]

    @pytest.mark.parametrize(
        'training_mask, expected', [(None, 'None'), (lambda _f: None, 'set')],
        ids=['no_window', 'window'],
    )
    def test_repr(self, training_mask, expected):
        """The representation names the materializer and the window wiring."""
        text = repr(make_builder(training_mask=training_mask))

        assert text.startswith('TrainingSetBuilder(materializer=CovariateMaterializer(')
        assert text.endswith(f'training_mask={expected})')
