"""Tests of the six rules of the mutualization on ``PANEL-F``.

Blocks ``{FR: Y, DE: Q, IT: M}`` and the 51 raw golden rows (3 + 12 +
36) at EVERY stage — ``Y``, ``Q``, ``M`` and a per-entity stage —, the target
handed over raw (never aggregated, even for the monthly ``IT`` block at stage
``Q``), golden scaled targets through ``StageScaler`` (12 / 3 / 1 at ``M``,
4 / 1 / 1/3 at ``Q``, 1 / 1/4 / 1/12 at ``Y``), one single call to the
materializer writing no store, one covariate materialized three ways (one per
block), the perimeter rule (truncated entity, entity never observing the
column, own target frequency, entity without detected frequency — absent
from a disagreeing mapping or mapped to None explicitly), the degenerate
cases of a single entity and the way contract.

End to end through ``HighFrequencyImputer``: the entity finer than the stage
(``IT`` at stage ``Q``) trains the model of ``v`` but no plan step of ``v``
ever predicts it; an entity observing ``v`` once (undetectable
frequency) never borrows the frequency of the others, unanimous or not.

Covered symbols: ``TrainingSetBuilder.build``, ``TrainingSetBuilder.blocks_of``.
"""
# Modules de base
from typing import List, Tuple

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Estimateur des vérifications de bout en bout
from sklearn.linear_model import LinearRegression

# Objets testés
from tsforecast.frequency.training_set_builder import (
    TrainingSet,
    TrainingSetBuilder,
)
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer
from tsforecast.frequency.stage_scaler import StageScaler
from tsforecast.utils.frequency.converter import FrequencyConverter

# Perturbations partagées
from tests.support.perturbations import (
    empty_like,
)

# Constantes et fabriques partagées du paquet
from tests.unit.frequency.training_set_builder.helpers import (
    FEATURES,
    PANEL_F_BLOCKS,
    PANEL_F_FREQUENCIES,
    V_DE,
    V_FR,
    V_IT,
    build_panel_f,
    build_ts,
    make_builder,
    scaled_target,
    snapshots_equal,
)


# =============================================================================
# R1, R2, R3 : blocs et lignes sur PANEL-F
# =============================================================================
class TestBlocksAndRows:
    """Scope, block frequency and rows."""

    def test_blocks_on_panel_f_at_monthly_stage(self, mixed_freq_panel_multifrequency):
        """Each entity contributes at ITS frequency for the column (R2)."""
        training = build_panel_f(make_builder(), mixed_freq_panel_multifrequency)

        assert dict(training.blocks) == PANEL_F_BLOCKS

    def test_row_counts_on_panel_f_at_monthly_stage(self, mixed_freq_panel_multifrequency):
        """51 rows — 3 FR, 12 DE, 36 IT — instead of 3 without mutualization."""
        training = build_panel_f(make_builder(), mixed_freq_panel_multifrequency)

        assert len(training) == 51
        assert training.y.groupby(level=0).size().to_dict() == {
            'DE': 12, 'FR': 3, 'IT': 36
        }
        # Les features couvrent exactement la grille mutualisée
        assert training.X.index.equals(training.y.index)
        assert list(training.X.columns) == FEATURES

    @pytest.mark.parametrize(
        'stage_freq',
        ['Y', 'Q', 'M', {('FR',): 'M', ('DE',): 'M', ('IT',): 'Q'}],
        ids=['Y', 'Q', 'M', 'per_entity'],
    )
    def test_golden_rows_at_every_stage(self, mixed_freq_panel_multifrequency, stage_freq):
        """Same blocks and same 51 raw golden rows at every stage."""
        training = build_panel_f(
            make_builder(), mixed_freq_panel_multifrequency, stage_freq=stage_freq
        )

        # Valeur d'or : les ancres brutes, entité par entité, jamais
        # agrégées ni mises à l'échelle
        assert dict(training.blocks) == PANEL_F_BLOCKS
        assert len(training) == 51
        assert training.y.loc['FR'].tolist() == V_FR
        assert training.y.loc['DE'].tolist() == V_DE
        assert training.y.loc['IT'].tolist() == V_IT

    @pytest.mark.parametrize(
        'stage, expected_fr, expected_de, expected_it',
        [
            # Étape Y : FR à l'échelle (1), DE × 4, IT × 12 — les trois blocs
            # retrouvent le même total annuel
            ('Y', V_FR, [value * 4 for value in V_DE], [value * 12 for value in V_IT]),
            # Étape Q : FR / 4, DE à l'échelle, IT × 3 (diviseur 1/3)
            ('Q', [30.0, 33.0, 37.5], V_DE, [value * 3 for value in V_IT]),
            # Étape M : FR / 12, DE / 3, IT à l'échelle
            ('M', [10.0, 11.0, 12.5], [value / 3 for value in V_DE], V_IT),
        ],
        ids=['Y', 'Q', 'M'],
    )
    def test_scaled_target_golden_values(
        self, mixed_freq_panel_multifrequency, stage, expected_fr, expected_de, expected_it
    ):
        """Through ``StageScaler``, the three blocks land on the stage scale (R5)."""
        training = build_panel_f(
            make_builder(), mixed_freq_panel_multifrequency, stage_freq=stage
        )
        scaled = scaled_target(training, stage)

        np.testing.assert_allclose(scaled.loc['FR'].to_numpy(), expected_fr, rtol=1e-12)
        np.testing.assert_allclose(scaled.loc['DE'].to_numpy(), expected_de, rtol=1e-12)
        np.testing.assert_allclose(scaled.loc['IT'].to_numpy(), expected_it, rtol=1e-12)

    def test_scaled_target_on_panel_f_at_monthly_stage(self, mixed_freq_panel_multifrequency):
        """9.333 for the first German quarter at stage ``M``."""
        training = build_panel_f(make_builder(), mixed_freq_panel_multifrequency)
        scaled = scaled_target(training, 'M')

        # Valeur d'or : 28 / 3 = 9.333 (tableau de l'étape M)
        assert round(float(scaled.loc[('DE', '2021-03-31')]), 3) == 9.333

    def test_row_frequency_and_origin_of_observed_rows(self, mixed_freq_panel_multifrequency):
        """An observed row carries the frequency of its block (R5)."""
        training = build_panel_f(make_builder(), mixed_freq_panel_multifrequency)

        assert set(training.row_origin) == {'observed'}
        assert set(training.row_frequency.loc['FR']) == {'Y'}
        assert set(training.row_frequency.loc['DE']) == {'Q'}
        assert set(training.row_frequency.loc['IT']) == {'M'}

    def test_rows_are_the_observed_cells(self, mixed_freq_panel_multifrequency):
        """Each row is an observed (entity, date) cell, at its source value."""
        data = mixed_freq_panel_multifrequency
        training = build_panel_f(make_builder(), data)

        # Propriété : les lignes sont exactement les cellules non-NaN de `v`
        observed = data['v'].dropna()
        assert training.y.sort_index().equals(observed.sort_index().rename('v'))
        # Et chaque bloc est trié par date
        for entity in ('FR', 'DE', 'IT'):
            assert training.y.loc[entity].index.is_monotonic_increasing

    def test_source_data_is_never_modified(self, mixed_freq_panel_multifrequency):
        """``build`` leaves ``source_data`` untouched."""
        data = mixed_freq_panel_multifrequency
        before = data.copy()

        build_panel_f(make_builder(), data)

        pd.testing.assert_frame_equal(data, before)


# =============================================================================
# Étape trimestrielle, bloc plus fin que l'étape
# =============================================================================
class TestQuarterlyStage:
    """The blocks do not depend on the stage, the divisors do."""

    def test_italian_rows_stay_monthly_at_quarterly_stage(
        self, mixed_freq_panel_multifrequency, monkeypatch
    ):
        """The IT block holds 36 MONTH-end rows, never 12 aggregates (R3)."""
        # Compteur d'agrégations : la cible ne doit JAMAIS y passer
        aggregated_columns = []
        original = FrequencyConverter.aggregate_to_lower_frequency

        def _spy(self, series, target_freq, *args, **kwargs):
            aggregated_columns.append(getattr(series, 'name', None))
            return original(self, series, target_freq, *args, **kwargs)

        monkeypatch.setattr(
            FrequencyConverter, 'aggregate_to_lower_frequency', _spy
        )

        training = build_panel_f(
            make_builder(), mixed_freq_panel_multifrequency, stage_freq='Q'
        )

        # Lignes italiennes : 36 fins de mois, dont des mois non trimestriels
        italian_dates = training.y.loc['IT'].index
        assert len(italian_dates) == 36
        assert not set(italian_dates.month).issubset({3, 6, 9, 12})
        # Cible BRUTE : c'est le diviseur qui porte l'échelle, pas l'agrégation
        assert sorted(set(training.y.loc['IT'].round(6))) == [10.0, 11.0, 12.5]
        assert 'v' not in aggregated_columns

    def test_blocks_are_stage_independent(self, mixed_freq_panel_multifrequency):
        """Same blocks and same grid at stages ``Q`` and ``M``."""
        builder = make_builder()
        monthly = build_panel_f(
            builder, mixed_freq_panel_multifrequency, stage_freq='M'
        )
        quarterly = build_panel_f(
            builder, mixed_freq_panel_multifrequency, stage_freq='Q'
        )

        assert dict(monthly.blocks) == dict(quarterly.blocks)
        assert monthly.y.index.equals(quarterly.y.index)
        pd.testing.assert_frame_equal(monthly.X, quarterly.X)

    # Avertissements légitimes de l'imputeur (étapes dégradées) : hors sujet ici
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_finer_entity_never_reaches_a_prediction_grid(
        self, mixed_freq_panel_multifrequency, monkeypatch
    ):
        """IT trains the stage-``Q`` model of ``v`` but no step of ``v`` predicts IT.

        End to end through ``HighFrequencyImputer`` under
        ``impute_intermediate_frequencies='covariates_only'`` (stage ``Q``
        exists): the training sets of ``v`` are captured at their source.
        """
        # Espion sur la composition : jeux de `v` complets (hors sonde)
        captured: List[Tuple[object, TrainingSet]] = []
        original = TrainingSetBuilder.build

        def _spy(self, **kwargs):
            training = original(self, **kwargs)
            if kwargs['column'] == 'v' and len(kwargs['feature_cols']) > 0:
                captured.append((kwargs['stage_freq'], training))
            return training

        monkeypatch.setattr(TrainingSetBuilder, 'build', _spy)
        imputer = HighFrequencyImputer(
            estimator=LinearRegression(),
            target_frequency='M',
            impute_intermediate_frequencies='covariates_only',
        )
        imputer.fit(mixed_freq_panel_multifrequency)

        # Côté entraînement : à l'étape Q, IT contribue ses 36 lignes mensuelles
        quarterly = [
            training for stage, training in captured
            if set(dict(stage).values() if isinstance(stage, dict) else [stage]) == {'Q'}
        ]
        assert quarterly
        assert all(len(training.y.loc['IT']) == 36 for training in quarterly)
        # Côté prédiction : aucune étape du plan pour `v` ne vise IT
        v_steps = [step for step in imputer.imputation_plan_.steps if step.var_name == 'v']
        assert v_steps
        assert all(('IT',) not in step.entities for step in v_steps)


# =============================================================================
# R4 : un seul appel au matérialiseur, aucun registre écrit
# =============================================================================
class TestSingleMaterialization:
    """R4: one single call to the materializer, and no store written."""

    def test_single_call_to_materialize(self, mixed_freq_panel_multifrequency):
        """``materialize`` is called once, on the blocks, without recording."""
        builder = make_builder()
        calls = []
        original = builder.materializer.materialize

        def _spy(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        builder.materializer.materialize = _spy
        training = build_panel_f(builder, mixed_freq_panel_multifrequency)

        assert len(calls) == 1
        assert calls[0]['stage_freq'] == dict(training.blocks)
        assert calls[0]['record'] is False
        assert calls[0]['grid_index'].equals(training.y.index)

    def test_stores_untouched_by_training_materialization(
        self, mixed_freq_panel_multifrequency
    ):
        """The three stores are identical before and after ``build``."""
        builder = make_builder(covariate_strategy='model')
        before = builder.materializer.snapshot()

        build_panel_f(builder, mixed_freq_panel_multifrequency)

        assert snapshots_equal(before, builder.materializer.snapshot())

    def test_no_carry_phantom_after_build(self, mixed_freq_panel_multifrequency):
        """No rank 3 becomes eligible merely because of ``build`` (R4)."""
        builder = make_builder(covariate_strategy='model')

        build_panel_f(builder, mixed_freq_panel_multifrequency)

        # Le miroir reste vide : rien n'a jamais été imputé
        assert builder.materializer.imputed_store == {}
        # Et la classification ne peut donc pas conclure au report d'étape
        for entity in (('FR',), ('DE',), ('IT',)):
            way = builder.materializer.classify(
                'q1', 'M', PANEL_F_FREQUENCIES, entity
            )
            assert way != 'carried_model'

    def test_covariates_of_each_block_at_its_own_frequency(
        self, mixed_freq_panel_multifrequency
    ):
        """One covariate, three treatments: aggregated, read as is, interpolated (R4)."""
        training = build_panel_f(make_builder(), mixed_freq_panel_multifrequency)

        # Valeurs d'or de m1 (100 + rang) : somme de l'année 2021 sur le bloc
        # annuel de FR = 100 + ... + 111 = 1266 ; somme du T1 sur le bloc
        # trimestriel de DE = 100 + 101 + 102 = 303 ; valeur du mois chez IT
        assert training.X.loc[('FR', '2021-12-31'), 'm1'] == 1266.0
        assert training.X.loc[('DE', '2021-03-31'), 'm1'] == 303.0
        assert training.X.loc[('IT', '2021-01-31'), 'm1'] == 100.0
        # q1 (10 × k) : identité chez DE, interpolée sur la grille mensuelle
        # d'IT — rang inférieur, l'entraînement est plus dégradé que la
        # prédiction, ce que la règle de non-dégradation autorise
        assert training.X.loc[('DE', '2021-03-31'), 'q1'] == 10.0
        assert training.ways['q1'] == 'interpolate'
        assert training.column_origins['q1'] == 'interpolated'


# =============================================================================
# R1 : périmètre des contributeurs
# =============================================================================
class TestScopeOfContributors:
    """R1: who contributes, and who contributes nothing without raising."""

    def test_truncated_entity_keeps_all_its_observations(
        self, mixed_freq_panel_multifrequency
    ):
        """An incomplete quarter costs NO row."""
        data = mixed_freq_panel_multifrequency.copy()
        # Troncature des deux derniers mois d'IT : le dernier trimestre est
        # incomplet, ce que `full_periods_only` sanctionnerait sur une agrégation
        last_two = data.loc['IT'].index[-2:]
        data.loc[[('IT', date) for date in last_two], 'v'] = np.nan

        training = build_panel_f(make_builder(), data, stage_freq='Q')

        assert len(training.y.loc['IT']) == 34
        assert len(training) == 3 + 12 + 34

    def test_entity_without_the_column_contributes_nothing(
        self, mixed_freq_panel_multifrequency
    ):
        """An entity never observing the column raises nothing."""
        data = mixed_freq_panel_multifrequency.copy()
        data.loc['DE', 'v'] = np.nan

        training = build_panel_f(make_builder(), data)

        assert dict(training.blocks) == {('FR',): 'Y', ('IT',): 'M'}
        assert 'DE' not in training.y.index.get_level_values(0)
        assert len(training) == 39

    def test_entity_with_other_target_frequency_still_contributes(
        self, mixed_freq_panel_multifrequency
    ):
        """An entity's own target frequency changes nothing (R1)."""
        # IT porte une fréquence cible distincte des deux autres entités
        stage_freq = {('FR',): 'M', ('DE',): 'M', ('IT',): 'Q'}

        training = build_panel_f(
            make_builder(), mixed_freq_panel_multifrequency, stage_freq=stage_freq
        )

        assert dict(training.blocks) == PANEL_F_BLOCKS
        assert len(training.y.loc['IT']) == 36
        assert len(training) == 51

    def test_column_absent_from_the_data_raises(self, mixed_freq_panel_multifrequency):
        """A column absent from the data is a wiring error."""
        with pytest.raises(KeyError, match='missing from source_data'):
            build_panel_f(
                make_builder(), mixed_freq_panel_multifrequency, column='absente'
            )

    def test_no_entity_observes_the_column(self, mixed_freq_panel_multifrequency):
        """No contributor: empty set, without raising."""
        data = mixed_freq_panel_multifrequency.copy()
        data['v'] = np.nan

        training = build_panel_f(make_builder(), data)

        assert dict(training.blocks) == {}
        assert dict(training.ways) == {}
        assert len(training) == 0
        assert list(training.X.columns) == FEATURES
        # Grille vide, de la forme de la source
        assert list(training.X.index.names) == ['country', 'date']

    def test_entity_missing_from_a_disagreeing_mapping_is_dropped_and_logged(
        self, mixed_freq_panel_multifrequency
    ):
        """No own frequency in a per-entity mapping: no divisor, block dropped."""
        logged: List[str] = []
        frequencies = {**PANEL_F_FREQUENCIES, 'v': {('FR',): 'Y', ('DE',): 'Q'}}

        training = build_panel_f(
            make_builder(log=logged.append), mixed_freq_panel_multifrequency,
            frequencies=frequencies,
        )

        assert dict(training.blocks) == {('FR',): 'Y', ('DE',): 'Q'}
        assert len(training) == 15
        assert any("('IT',) has no detected frequency" in line for line in logged)

    def test_single_entry_mapping_stands_for_the_whole_panel(
        self, mixed_freq_panel_multifrequency
    ):
        """A single-entry mapping reads like the scalar form."""
        frequencies = {**PANEL_F_FREQUENCIES, 'v': {('FR',): 'Y'}}

        training = build_panel_f(
            make_builder(), mixed_freq_panel_multifrequency, frequencies=frequencies
        )

        # Liaison à une entrée : elle vaut pour tout le panel, comme la forme
        # scalaire. L'imputeur ne la produit jamais pour la colonne cible : il
        # explicite chaque couple indétectable par None (test suivant)
        assert dict(training.blocks) == {('FR',): 'Y', ('DE',): 'Y', ('IT',): 'Y'}

    def test_explicit_none_drops_the_block_and_logs(self, mixed_freq_panel_multifrequency):
        """An entity mapped to None explicitly has no block, even in a 2-entry mapping."""
        logged: List[str] = []
        # IT observe `v` mais sa fréquence est déclarée indétectable ; FR seule
        # porte une fréquence : sans le None explicite, la liaison à une
        # entrée la prêterait à IT
        frequencies = {**PANEL_F_FREQUENCIES, 'v': {('FR',): 'Y', ('IT',): None}}

        training = build_panel_f(
            make_builder(log=logged.append), mixed_freq_panel_multifrequency,
            frequencies=frequencies,
        )

        # DE, absente d'une liaison à deux entrées, n'a pas de fréquence non plus
        assert dict(training.blocks) == {('FR',): 'Y'}
        assert any("('IT',) has no detected frequency" in line for line in logged)

    # Avertissements légitimes de l'imputeur (étapes dégradées) : hors sujet ici
    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize(
        'de_annual', [True, False], ids=['unanimous_others', 'disagreeing_others'],
    )
    def test_imputer_never_lends_a_frequency_to_an_undetectable_entity(
        self, mixed_freq_panel_multifrequency, monkeypatch, de_annual
    ):
        """End to end: IT observing ``v`` once never gets a block, whatever FR / DE do.

        FR and DE both annual must not fold ``v`` into the scalar ``'Y'``
        lent to IT: its single MONTHLY value 10.0 would enter the training set
        as a yearly total divided by 12.
        """
        data = mixed_freq_panel_multifrequency.copy()
        # IT : une seule observation de `v`, fréquence indétectable
        data.loc['IT', 'v'] = np.nan
        data.loc[('IT', pd.Timestamp('2021-01-31')), 'v'] = 10.0
        if de_annual:
            # DE annuelle, de même total que FR : les fréquences détectées
            # s'accordent, la colonne se replie sur la forme scalaire
            data.loc['DE', 'v'] = np.nan
            for date, total in zip(['2021-12-31', '2022-12-31', '2023-12-31'], V_FR):
                data.loc[('DE', pd.Timestamp(date)), 'v'] = total
        captured: List[TrainingSet] = []
        original = TrainingSetBuilder.build

        def _spy(self, **kwargs):
            training = original(self, **kwargs)
            if kwargs['column'] == 'v':
                captured.append(training)
            return training

        monkeypatch.setattr(TrainingSetBuilder, 'build', _spy)
        imputer = HighFrequencyImputer(estimator=LinearRegression(), target_frequency='M')
        imputer.fit(data)

        expected = {('DE',): 'Y' if de_annual else 'Q', ('FR',): 'Y'}
        assert captured
        assert all(dict(training.blocks) == expected for training in captured)
        assert all('IT' not in training.y.index.get_level_values(0) for training in captured)

    def test_blocks_of_is_the_r1_perimeter(self, mixed_freq_panel_multifrequency):
        """``blocks_of`` reads the per-entity form, without any stage."""
        builder = make_builder()

        blocks = builder.blocks_of(
            'v', PANEL_F_FREQUENCIES, mixed_freq_panel_multifrequency
        )

        # Ordre des entités : celui des données (DE, FR, IT dans PANEL-X),
        # jamais celui de la liaison des fréquences détectées
        assert list(blocks.items()) == [(('DE',), 'Q'), (('FR',), 'Y'), (('IT',), 'M')]


# =============================================================================
# Cas dégénérés
# =============================================================================
class TestDegenerateCases:
    """One-entity panel: the mutualization is neutral in the singular."""

    def test_timeseries_is_the_degenerate_case(self, reference_timeseries):
        """On ``TS``, the mutualized set is the original one."""
        training = build_ts(make_builder(), reference_timeseries, eligible_origins=('observed',))

        assert dict(training.blocks) == {(): 'Y'}
        assert len(training) == 3
        assert training.y.tolist() == [120.0, 132.0, 150.0]
        assert isinstance(training.y.index, pd.DatetimeIndex)

        # Diviseur 12 à l'étape M : tous les chiffres de l'exemple de référence
        # restent vrais
        scaler = StageScaler()
        divisors = scaler.target_divisor(
            'a1', source_freq='Y', pred_freq='M',
            produced_freq=training.row_frequency,
        )
        assert divisors.tolist() == [12.0, 12.0, 12.0]
        assert scaler.apply(training.y, divisors).tolist() == [10.0, 11.0, 12.5]

    def test_single_entity_panel(self, mixed_freq_panel_multifrequency, reference_timeseries):
        """A one-entity panel yields the same set as a time series."""
        data = mixed_freq_panel_multifrequency.loc[['FR']]

        training = build_panel_f(make_builder(), data)

        assert dict(training.blocks) == {('FR',): 'Y'}
        assert len(training) == 3
        assert training.y.tolist() == reference_timeseries['a1'].dropna().tolist()
        assert scaled_target(training, 'M').tolist() == [10.0, 11.0, 12.5]

    def test_single_observation(self, mixed_freq_panel_multifrequency):
        """One observed cell: one row, its covariates materialized on the block."""
        data = mixed_freq_panel_multifrequency.loc[[('FR', pd.Timestamp('2021-12-31'))]]

        training = build_panel_f(make_builder(), data)

        assert dict(training.blocks) == {('FR',): 'Y'}
        assert training.y.tolist() == [120.0]

    def test_empty_frame(self, mixed_freq_panel_multifrequency):
        """An empty frame yields an empty set of the source shape."""
        training = build_panel_f(make_builder(), empty_like(mixed_freq_panel_multifrequency))

        assert len(training) == 0
        assert dict(training.blocks) == {}
        assert list(training.X.columns) == FEATURES
        assert list(training.X.index.names) == ['country', 'date']


# =============================================================================
# Unicité de la voie
# =============================================================================
class TestWayContract:
    """The ways handed back are those imposed on the prediction grid."""

    def test_way_uniqueness_contract(self, mixed_freq_panel_multifrequency):
        """The ways cover exactly ``feature_cols`` and can be replayed."""
        builder = make_builder()
        training = build_panel_f(builder, mixed_freq_panel_multifrequency)

        assert set(training.ways) == set(FEATURES)
        assert set(training.column_origins) == set(FEATURES)

        # Rejeu tel quel sur une autre grille : celle de prédiction de FR
        prediction_grid = mixed_freq_panel_multifrequency.loc[['FR']].index
        _features, replayed, _origins = builder.materializer.materialize(
            columns=tuple(FEATURES),
            grid_index=prediction_grid,
            stage_freq='M',
            detected_frequencies=PANEL_F_FREQUENCIES,
            source_data=mixed_freq_panel_multifrequency,
            materialization=dict(training.ways),
            record=False,
        )
        assert replayed == dict(training.ways)

    def test_imposed_ways_are_handed_back_verbatim(self, mixed_freq_panel_multifrequency):
        """Imposed ways come back as imposed, not as their block image (R4)."""
        imposed = {'m1': 'interpolate', 'q1': 'interpolate'}

        training = build_panel_f(
            make_builder(), mixed_freq_panel_multifrequency, materialization=imposed
        )

        assert dict(training.ways) == imposed
        # Sur le bloc annuel de FR, l'interpolation imposée de m1 (mensuelle)
        # est ramenée à l'agrégation exacte de rang 1 : 1266 (somme 2021)
        assert training.X.loc[('FR', '2021-12-31'), 'm1'] == 1266.0
