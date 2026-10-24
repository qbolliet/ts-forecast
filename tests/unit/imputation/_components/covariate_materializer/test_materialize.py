"""Tests of ``CovariateMaterializer.materialize``, the single producer of features.

Covers rank 1 (identity, exact aggregation always by SUM), ``'tolerate_nan'``,
``'interpolate'``, the uniqueness rule (the way decided once on the
prediction grid, replayed as-is and brought back to each entity's
frequencies), the ``record`` flag, the reference example, the empty columns,
and the ``AggregationConstraintApplier`` injection point through a minimal
double and once through the real ``AggregationConstraint``.

A column absent from the source is a ``KeyError``; the ``'raw_anchors'`` way
goes through the per-entity rank-1 normalization.

Covered symbols: ``CovariateMaterializer.materialize``,
``AggregationConstraintApplier``.
"""
# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objets testés
from tsforecast.imputation._components.aggregation_constraint import AggregationConstraint
from tsforecast.imputation._components.covariate_materializer import (
    DEFAULT_MATERIALIZATION_KEY,
    CovariateMaterializer,
)

# Jeux partagés
from tests.unit.imputation._components.covariate_materializer.helpers import (
    PANEL_FREQUENCIES,
    PANEL_F_FREQUENCIES,
    SpyApplier,
    TS_FREQUENCIES,
    annual_anchor_grid,
    annual_grid,
    block_grid,
    entity_grid,
    monthly_grid,
    origins_per_entity,
    panel_grid,
    prime,
    quarterly_grid,
)


class TestRankOneClassification:
    """Rank 1: identity and exact aggregation, ``'observed'`` origins."""

    def test_identity_and_aggregate_ranks(self, reference_timeseries):
        """``f_c == f`` gives ``'identity'``, a finer ``f_c`` gives ``'aggregate'``."""
        materializer = CovariateMaterializer()

        features, ways, origins = materializer.materialize(
            columns=['q1', 'm1'],
            grid_index=quarterly_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            record=True,
        )

        # Voies retenues
        assert ways == {'q1': 'identity', 'm1': 'aggregate'}
        # Origines agrégées : une agrégation exacte n'est pas une approximation
        assert origins == {'q1': 'observed', 'm1': 'observed'}

        # Identité : les valeurs trimestrielles portées telles quelles
        np.testing.assert_allclose(features['q1'].to_numpy(), 10.0 * np.arange(1, 13))

        # Agrégation : m1 vaut 100 + i, sommée par trimestre
        assert features['m1'].iloc[0] == pytest.approx(100 + 101 + 102)
        assert features['m1'].iloc[1] == pytest.approx(103 + 104 + 105)
        # Aucune période incomplète sur cette grille
        assert features['m1'].notna().all()

        # Registre d'origines : 'observed' partout où une valeur est produite
        for column in ('q1', 'm1'):
            assert set(materializer.origin_store[column].unique()) == {'observed'}

    def test_rank1_aggregation_is_always_a_sum(self, reference_timeseries):
        """A finer covariate is aggregated by SUM even under None.

        ``aggregation_constraint`` only governs the rescaling of INTERPOLATED
        covariates: the exact aggregation of rank 1 never obeys it.
        """
        materializer = CovariateMaterializer(aggregation_constraint=None)

        features, ways, origins = materializer.materialize(
            columns=['m1'],
            grid_index=quarterly_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        assert ways == {'m1': 'aggregate'}
        assert origins == {'m1': 'observed'}
        # m1 vaut 100 + i, sommée par trimestre, comme sous le défaut 'sum'
        assert features['m1'].iloc[0] == pytest.approx(100 + 101 + 102)
        assert features['m1'].iloc[1] == pytest.approx(103 + 104 + 105)

    def test_incomplete_period_aggregation_is_nan(self, reference_timeseries):
        """An incomplete period yields NaN, a legitimate source left unmasked."""
        materializer = CovariateMaterializer()
        # Retrait de décembre 2023 : l'année 2023 devient incomplète
        truncated = reference_timeseries.iloc[:-1]

        features, ways, _origins = materializer.materialize(
            columns=['m1'],
            grid_index=annual_grid(),
            stage_freq='Y',
            detected_frequencies=TS_FREQUENCIES,
            source_data=truncated,
            record=True,
        )

        assert ways['m1'] == 'aggregate'
        # Les deux années complètes sont sommées, la troisième est NaN
        assert features['m1'].loc['2021-12-31'] == pytest.approx(1266.0)
        assert features['m1'].loc['2022-12-31'] == pytest.approx(1410.0)
        assert np.isnan(features['m1'].loc['2023-12-31'])
        # La cellule non produite n'entre pas dans le registre
        assert pd.Timestamp('2023-12-31') not in materializer.origin_store['m1'].index


class TestTolerateNan:
    """``'tolerate_nan'`` strategy: no materialization at all."""

    def test_tolerate_nan_keeps_anchors_only(self, reference_timeseries):
        """a1 is 120 on 2021-12-31 and NaN on the 11 other months, on both sides."""
        materializer = CovariateMaterializer(covariate_strategy='tolerate_nan')

        x_train, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=monthly_grid('2021-01-31', 12),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        # La voie est imposée à la seconde grille
        x_pred, ways_pred, _ = materializer.materialize(
            columns=['a1'],
            grid_index=monthly_grid('2022-01-31', 12),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            materialization=ways,
        )

        assert ways == {'a1': 'raw_anchors'}
        assert ways_pred == ways
        assert origins == {'a1': 'observed'}

        # Valeur d'or du document à l'ancre, NaN sur les onze autres mois
        assert x_train['a1'].loc['2021-12-31'] == pytest.approx(120.0)
        assert x_train['a1'].isna().sum() == 11
        # Taux de NaN identique au predict : 11/12 des deux côtés
        assert x_pred['a1'].loc['2022-12-31'] == pytest.approx(132.0)
        assert x_pred['a1'].isna().sum() == 11


class TestInterpolate:
    """``'interpolate'`` strategy, the default."""

    def test_interpolate_removes_all_nan_except_empty_entity(
        self, mixed_freq_panel_heterogeneous
    ):
        """Only the entity without any observation keeps NaN."""
        data = mixed_freq_panel_heterogeneous
        materializer = CovariateMaterializer(covariate_strategy='interpolate')

        features, ways, origins = materializer.materialize(
            columns=['m1', 'q1', 'a1', 'a2', 'climat_affaires'],
            grid_index=panel_grid(data, monthly_grid()),
            stage_freq='M',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
        )

        # Colonnes plus basses que la grille : interpolées ; les autres, rang 1
        assert ways == {
            'm1': 'identity',
            'q1': 'interpolate',
            'a1': 'interpolate',
            'a2': 'interpolate',
            'climat_affaires': 'identity',
        }
        assert origins['q1'] == 'interpolated'
        assert origins['m1'] == 'observed'

        # Plus aucun NaN, sauf climat_affaires pour l'Italie
        assert features[['m1', 'q1', 'a1', 'a2']].notna().all().all()
        assert features.loc['IT', 'climat_affaires'].isna().all()
        for entity in ('FR', 'DE'):
            assert features.loc[entity, 'climat_affaires'].notna().all()

    def test_interpolate_uses_per_feature_method_and_anchor(self, reference_timeseries):
        """Dict form with ``'__default__'``: method and anchor resolved per feature."""
        materializer = CovariateMaterializer(
            interpolation_method={'a1': 'nearest', DEFAULT_MATERIALIZATION_KEY: 'linear'},
            interpolation_anchor={'a1': None, DEFAULT_MATERIALIZATION_KEY: 0.5},
        )

        # Résolution des réglages
        assert materializer.resolve_method('a1') == 'nearest'
        assert materializer.resolve_method('a2') == 'linear'
        assert materializer.resolve_anchor('a1') is None
        assert materializer.resolve_anchor('a2') == 0.5

        grid = monthly_grid()
        features, _ways, _origins = materializer.materialize(
            columns=['a1', 'a2'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        # 'nearest' sans ancrage recopie l'ancre sur toute la période
        assert set(np.round(features['a1'].dropna().unique(), 6)) <= {120.0, 132.0, 150.0}

        # Référence linéaire sans ancrage, pour montrer que 0.5 déplace la forme
        reference = CovariateMaterializer()
        flat, _, _ = reference.materialize(
            columns=['a2'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        assert not np.allclose(
            features['a2'].to_numpy(), flat['a2'].to_numpy(), equal_nan=True
        )
        # L'ancrage au milieu de période retire la valeur d'ancre de la sortie
        assert features['a2'].loc['2022-12-31'] != pytest.approx(66.0)


class TestEmptyColumns:
    """Nothing can be fabricated where nothing is observed."""

    @pytest.mark.parametrize(
        'strategy, fallback',
        [('tolerate_nan', 'interpolate'), ('interpolate', 'interpolate'),
         ('model', 'interpolate'), ('model', 'tolerate_nan')],
        ids=['tolerate_nan', 'interpolate', 'model-interpolate', 'model-tolerate_nan'],
    )
    def test_all_nan_column_produces_and_records_nothing(
        self, reference_timeseries, strategy, fallback
    ):
        """An empty column stays NaN, reports ``'observed'`` and enters no store."""
        blind = reference_timeseries.copy()
        blind['a1'] = np.nan
        materializer = CovariateMaterializer(covariate_strategy=strategy, covariate_fallback=fallback)

        features, _ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=blind,
        )
        assert (
            int(features['a1'].notna().sum()), origins['a1'], 'a1' in materializer.imputed_store,
        ) == (0, 'observed', False)

    def test_entity_without_observation_of_a_coarser_column(self, mixed_freq_panel_heterogeneous):
        """An entity that never observes a1 stays NaN; the others are interpolated."""
        data = mixed_freq_panel_heterogeneous.copy()
        data.loc['IT', 'a1'] = np.nan

        features, _ways, _origins = CovariateMaterializer().materialize(
            columns=['a1'],
            grid_index=panel_grid(data, monthly_grid()),
            stage_freq='M',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
        )
        filled = features['a1'].notna().groupby(level=0).sum().to_dict()
        assert filled == {'DE': 36, 'FR': 36, 'IT': 0}

    def test_grid_entity_absent_from_the_source_stays_nan(self, mixed_freq_panel_heterogeneous):
        """An entity of the grid unknown to the source gets NaN rows, the others are unchanged."""
        data = mixed_freq_panel_heterogeneous
        kwargs = dict(
            columns=['m1', 'a1'],
            stage_freq='M',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
        )
        reference, _, _ = CovariateMaterializer().materialize(
            grid_index=entity_grid(['FR', 'DE'], monthly_grid()), **kwargs
        )
        features, _, _ = CovariateMaterializer().materialize(
            grid_index=entity_grid(['FR', 'DE', 'ES'], monthly_grid()), **kwargs
        )
        # Lignes de l'entité inconnue entièrement NaN, et rien d'autre ne bouge
        assert features.loc['ES'].isna().all().all()
        pd.testing.assert_frame_equal(features.loc[['FR', 'DE']], reference)


class TestColumnAbsentFromSource:
    """An unknown column is an error, as in ``interpolate_column``."""

    @pytest.mark.parametrize('method', ['materialize', 'stage_frame'])
    def test_column_absent_from_the_source_raises(self, reference_timeseries, method):
        """A typo in ``columns`` raises ``KeyError`` instead of a silent NaN feature.

        ``interpolate_column`` and ``entities_without_column`` already raise
        ``KeyError`` on an absent column; ``eligible_columns``, a selection,
        legitimately skips it.
        """
        materializer = CovariateMaterializer()
        kwargs = dict(
            columns=['m1', 'typo'],
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        with pytest.raises(KeyError, match='typo'):
            getattr(materializer, method)(**kwargs)


class TestUniquenessOfTheWay:
    """The way is chosen once and replayed as-is."""

    def test_replayed_materialization_is_identical(self, mixed_freq_panel_heterogeneous):
        """Choosing then replaying gives exactly the same values and ways."""
        data = mixed_freq_panel_heterogeneous
        columns = ['m1', 'q1', 'a1', 'climat_affaires']
        grid = panel_grid(data, monthly_grid())

        chooser = CovariateMaterializer()
        chosen, ways, origins = chooser.materialize(
            columns=columns,
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
        )

        replayer = CovariateMaterializer()
        replayed, ways_replayed, origins_replayed = replayer.materialize(
            columns=columns,
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
            materialization=ways,
        )

        assert ways_replayed == ways
        assert origins_replayed == origins
        pd.testing.assert_frame_equal(replayed, chosen)

    def test_replay_must_cover_exactly_the_columns(self, reference_timeseries):
        """An incomplete replay is rejected."""
        materializer = CovariateMaterializer()
        with pytest.raises(ValueError, match=r"doit couvrir exactement columns"):
            materializer.materialize(
                columns=['m1', 'a1'],
                grid_index=monthly_grid(),
                stage_freq='M',
                detected_frequencies=TS_FREQUENCIES,
                source_data=reference_timeseries,
                materialization={'m1': 'identity'},
            )

    def test_orphan_way_is_rejected(self, reference_timeseries):
        """A replayed way for a column not materialized is rejected and named."""
        with pytest.raises(ValueError, match=r"Orphaned materialization paths : \['zz'\]"):
            CovariateMaterializer().materialize(
                columns=['m1'],
                grid_index=monthly_grid(),
                stage_freq='M',
                detected_frequencies=TS_FREQUENCIES,
                source_data=reference_timeseries,
                materialization={'m1': 'identity', 'zz': 'identity'},
            )

    def test_missing_and_orphan_ways_are_both_listed(self, reference_timeseries):
        """Both defects of one replay are reported together."""
        with pytest.raises(ValueError) as error:
            CovariateMaterializer().materialize(
                columns=['m1', 'a1'],
                grid_index=monthly_grid(),
                stage_freq='M',
                detected_frequencies=TS_FREQUENCIES,
                source_data=reference_timeseries,
                materialization={'m1': 'identity', 'zz': 'identity'},
            )
        message = str(error.value)
        assert (
            "Columns without a materialization path : ['a1']" in message
            and "Orphaned materialization paths : ['zz']" in message
        )

    def test_way_unicity_fit_and_pred(self, reference_timeseries):
        """Under ``'model'``, the fallback applies at fit TOO.

        ``TS`` dataset, stage ``Q``, order ``a1`` before ``a2``: when ``a1``
        is imputed, ``a2`` is materialized by no higher rank, so it falls
        under ``covariate_fallback`` — on both sides, the training grid
        included, where its anchors alone would suffice (the 3 annual anchors
        are filled). Otherwise the model would learn on the exact covariate
        and predict on the interpolated one.
        """
        materializer = CovariateMaterializer(covariate_strategy='model')
        # L'étape est Q des deux côtés : la grille d'entraînement est la grille
        # Q masquée aux 3 ancres annuelles de a1, la grille de prédiction est
        # la grille Q complète
        grid_train = annual_anchor_grid()
        grid_pred = quarterly_grid()

        x_train, ways, origins = materializer.materialize(
            columns=['m1', 'q1', 'a2'],
            grid_index=grid_train,
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        x_pred, ways_pred, _origins_pred = materializer.materialize(
            columns=['m1', 'q1', 'a2'],
            grid_index=grid_pred,
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            materialization=ways,
        )

        # Rang 4 des deux côtés, jamais le rang 1 que les ancres autoriseraient
        assert ways['a2'] == 'interpolate'
        assert ways_pred == ways
        # La NATURE des valeurs, pas seulement le motif de NaN
        assert origins['a2'] == 'interpolated'
        assert x_train['a2'].notna().all()
        assert x_pred['a2'].notna().all()

        # Valeur par valeur : celle du repli, produite par la même routine
        reference = CovariateMaterializer(covariate_strategy='model')
        expected = reference.interpolate_column(
            'a2', grid_train, 'Q', TS_FREQUENCIES, reference_timeseries
        )
        np.testing.assert_allclose(x_train['a2'].to_numpy(), expected.to_numpy())

        # Cas numériquement visible : avec un ancrage à mi-période, le repli
        # du fit ne redonne PAS les ancres 60 / 66 de a2. Valeur d'or : 60 est
        # atteint le 2021-07-02 (milieu de 2021), 66 le 2022-07-02, 365 jours
        # plus loin ; le 2021-12-31 en est à 182 jours : 60 + 6 · 182 / 365
        anchored = CovariateMaterializer(
            covariate_strategy='model',
            interpolation_anchor={'a2': 0.5, DEFAULT_MATERIALIZATION_KEY: None},
        )
        x_anchored, ways_anchored, _ = anchored.materialize(
            columns=['a2'],
            grid_index=grid_train[:2],
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        assert ways_anchored == {'a2': 'interpolate'}
        np.testing.assert_allclose(
            x_anchored['a2'].to_numpy(), [60 + 6 * 182 / 365, 66 + 6 * 182 / 365]
        )

    @pytest.mark.parametrize(
        'column, entity, expected',
        [
            # FR, bloc annuel : q1 trimestrielle sommée sur l'année
            ('q1', 'FR', [100.0, 260.0, 420.0]),
            # DE, bloc trimestriel : q1 lue telle quelle
            ('q1', 'DE', list(10.0 * np.arange(1, 13))),
            # FR, bloc annuel : m1 mensuelle sommée sur l'année
            ('m1', 'FR', [1266.0, 1410.0, 1554.0]),
            # DE, bloc trimestriel : m1 sommée par trimestre
            ('m1', 'DE', list(303.0 + 9.0 * np.arange(12))),
            # IT, bloc mensuel : m1 lue telle quelle
            ('m1', 'IT', list(100.0 + np.arange(36))),
        ],
        ids=['q1-FR-aggregate', 'q1-DE-identity', 'm1-FR-aggregate', 'm1-DE-aggregate', 'm1-IT-identity'],
    )
    def test_replayed_way_is_applied_per_entity_on_mutualized_blocks(
        self, mixed_freq_panel_multifrequency, column, entity, expected
    ):
        """On PANEL-F, one way per column, each block at its own frequency.

        The way decided on the monthly prediction grid — q1 'interpolate', m1
        'identity' — is replayed on the pooled grid of the blocks of v (FR annual,
        DE quarterly, IT monthly): the same way becomes aggregation, identity or
        interpolation depending on the block.
        """
        features, _ways, _origins = CovariateMaterializer().materialize(
            columns=['q1', 'm1'],
            grid_index=block_grid({'FR': annual_grid(), 'DE': quarterly_grid(), 'IT': monthly_grid()}),
            stage_freq={('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'},
            detected_frequencies=PANEL_F_FREQUENCIES,
            source_data=mixed_freq_panel_multifrequency,
            materialization={'q1': 'interpolate', 'm1': 'identity'},
        )
        np.testing.assert_allclose(features.loc[entity, column].to_numpy(), expected)

    def test_interpolated_block_passes_through_its_anchors(self, mixed_freq_panel_multifrequency):
        """IT block: q1 interpolated monthly keeps its quarter-end anchors."""
        features, _ways, _origins = CovariateMaterializer().materialize(
            columns=['q1'],
            grid_index=block_grid({'FR': annual_grid(), 'DE': quarterly_grid(), 'IT': monthly_grid()}),
            stage_freq={('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'},
            detected_frequencies=PANEL_F_FREQUENCIES,
            source_data=mixed_freq_panel_multifrequency,
            materialization={'q1': 'interpolate'},
        )
        italian = features.loc['IT', 'q1']
        np.testing.assert_allclose(italian.loc[quarterly_grid()].to_numpy(), 10.0 * np.arange(1, 13))

    def test_each_entity_records_the_origin_of_its_own_way(self, mixed_freq_panel_multifrequency):
        """One effective way per (column, entity, stage): one origin per entity block."""
        materializer = CovariateMaterializer()
        materializer.materialize(
            columns=['q1'],
            grid_index=block_grid({'FR': annual_grid(), 'DE': quarterly_grid(), 'IT': monthly_grid()}),
            stage_freq={('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'},
            detected_frequencies=PANEL_F_FREQUENCIES,
            source_data=mixed_freq_panel_multifrequency,
            materialization={'q1': 'interpolate'},
            record=True,
        )
        assert origins_per_entity(materializer.origin_store['q1']) == {
            'DE': {'observed'}, 'FR': {'observed'}, 'IT': {'interpolated'},
        }

    def test_rank_one_way_imposed_on_a_coarser_grid_becomes_an_aggregation(
        self, reference_timeseries
    ):
        """``'identity'`` replayed on a grid coarser than the column sums it.

        Without this normalization, the cell would carry the point value of March
        (102), a monthly quantity on a quarterly row.
        """
        features, _ways, _origins = CovariateMaterializer().materialize(
            columns=['m1'],
            grid_index=quarterly_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            materialization={'m1': 'identity'},
        )
        np.testing.assert_allclose(features['m1'].to_numpy(), 303.0 + 9.0 * np.arange(12))

    @pytest.mark.parametrize(
        'strategy, fallback, filled, origin',
        [
            ('tolerate_nan', 'interpolate', 3, 'observed'),
            ('interpolate', 'interpolate', 36, 'interpolated'),
            ('model', 'interpolate', 36, 'interpolated'),
            ('model', 'tolerate_nan', 3, 'observed'),
        ],
        ids=['tolerate_nan', 'interpolate', 'model-interpolate', 'model-tolerate_nan'],
    )
    def test_rank_one_way_imposed_on_a_finer_grid_falls_back_to_rank_four(
        self, reference_timeseries, strategy, fallback, filled, origin
    ):
        """``'identity'`` replayed on a grid finer than the column is the rank 4."""
        materializer = CovariateMaterializer(covariate_strategy=strategy, covariate_fallback=fallback)
        features, _ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            materialization={'a1': 'identity'},
        )
        assert (int(features['a1'].notna().sum()), origins['a1']) == (filled, origin)


class TestRecordFlag:
    """The stores hold what was IMPUTED; replay is what keeps the way fixed."""

    @staticmethod
    def _materialize(materializer, data, **kwargs):
        """Materialize a1 on the monthly grid of ``TS``."""
        return materializer.materialize(
            columns=['a1'],
            grid_index=monthly_grid(),
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=data,
            **kwargs,
        )

    def test_default_writes_nothing_and_keeps_the_way(self, reference_timeseries):
        """By default, ``materialize`` produces the features and leaves the three stores empty.

        Default ``record=False``: materializing a covariate is not an
        imputation, and two successive calls decide the same way.
        """
        materializer = CovariateMaterializer(covariate_strategy='model')
        _features, first_ways, _origins = self._materialize(materializer, reference_timeseries)
        _features, second_ways, _origins = self._materialize(materializer, reference_timeseries)
        assert (
            materializer.imputed_store, materializer.imputed_freq_store,
            materializer.origin_store, second_ways,
        ) == ({}, {}, {}, first_ways)

    def test_unreplayed_second_call_reads_the_first_recorded_production(self, reference_timeseries):
        """Under ``record=True``, a second decision reads what the first one wrote.

        La première matérialisation (rang 4, interpolation) s'enregistre au
        pas M ; la seconde, sans rejeu, y lit un rang 2. C'est pourquoi la
        voie d'une étape n'est jamais redécidée, et pourquoi l'enregistrement
        n'est pas le défaut.
        """
        materializer = CovariateMaterializer(covariate_strategy='model')
        _features, first_ways, _origins = self._materialize(materializer, reference_timeseries, record=True)
        _features, second_ways, _origins = self._materialize(materializer, reference_timeseries, record=True)
        assert (first_ways, second_ways) == ({'a1': 'interpolate'}, {'a1': 'stage_model'})

    def test_replay_keeps_the_way_and_the_values(self, reference_timeseries):
        """Replaying the first way gives the first features, whatever the stores hold since."""
        materializer = CovariateMaterializer(covariate_strategy='model')
        first, ways, _origins = self._materialize(materializer, reference_timeseries, record=True)
        second, replayed, _origins = self._materialize(
            materializer, reference_timeseries, materialization=ways, record=True
        )
        assert replayed == ways
        pd.testing.assert_frame_equal(second, first)


class TestRawAnchorsPerEntity:
    """``'raw_anchors'`` is brought back to each entity's rank 1.

    The classification is made entity by entity: a covariate at least as fine as the
    grid is ALWAYS available, by identity or exact aggregation.
    The column's way is applied per entity, ``_applicable_way`` bringing it
    back to what the entity's frequencies allow — ``'raw_anchors'`` included,
    so that a finer entity never receives the point value of the grid date.
    """

    def test_finer_entity_is_aggregated_under_tolerate_nan(self, mixed_freq_panel_multifrequency):
        """PANEL-F at Q under ``'tolerate_nan'``: IT's monthly v is summed per quarter.

        Golden values: quarters of 30, 33 and 37.5, not the point value of
        the quarter-end month (10, 11, 12.5).
        """
        features, _ways, _origins = CovariateMaterializer(covariate_strategy='tolerate_nan').materialize(
            columns=['v'],
            grid_index=entity_grid(['FR', 'DE', 'IT'], quarterly_grid()),
            stage_freq='Q',
            detected_frequencies=PANEL_F_FREQUENCIES,
            source_data=mixed_freq_panel_multifrequency,
        )
        np.testing.assert_allclose(
            features.loc['IT', 'v'].to_numpy(), [30.0] * 4 + [33.0] * 4 + [37.5] * 4
        )

    def test_rank_one_entities_survive_an_entity_without_frequency(self, mixed_freq_panel_multifrequency):
        """c is monthly for DE, quarterly for FR, never observed by IT: DE is summed.

        IT, without a frequency, does not vote; and even if the column way were
        'raw_anchors', DE would bring it back to the exact aggregation: the
        sum of the quarter (303), never the m1 of March (102).
        """
        data = mixed_freq_panel_multifrequency[['m1', 'q1']].copy()
        data['c'] = np.nan
        data.loc['DE', 'c'] = data.loc['DE', 'm1'].to_numpy()
        data.loc['FR', 'c'] = data.loc['FR', 'q1'].to_numpy()

        features, _ways, _origins = CovariateMaterializer().materialize(
            columns=['c'],
            grid_index=entity_grid(['FR', 'DE', 'IT'], quarterly_grid()),
            stage_freq='Q',
            detected_frequencies={'c': {('FR',): 'Q', ('DE',): 'M'}},
            source_data=data,
        )
        np.testing.assert_allclose(features.loc['DE', 'c'].to_numpy(), 303.0 + 9.0 * np.arange(12))

    def test_interpolate_way_already_aggregates_the_finer_entity(self, mixed_freq_panel_multifrequency):
        """Same PANEL-F stage under ``'interpolate'``: IT is summed (30 / 33 / 37.5)."""
        features, _ways, _origins = CovariateMaterializer(covariate_strategy='interpolate').materialize(
            columns=['v'],
            grid_index=entity_grid(['FR', 'DE', 'IT'], quarterly_grid()),
            stage_freq='Q',
            detected_frequencies=PANEL_F_FREQUENCIES,
            source_data=mixed_freq_panel_multifrequency,
        )
        np.testing.assert_allclose(
            features.loc['IT', 'v'].to_numpy(), [30.0] * 4 + [33.0] * 4 + [37.5] * 4
        )


class TestReferenceExampleTable:
    """The four rows of the reference table, golden values included."""

    # Colonnes et grilles de l'exemple : imputation de a1 à l'étape Q
    COLUMNS = ['m1', 'q1', 'a2']

    def _fit_and_predict(self, materializer, data):
        """Materialize on the training grid, then replay on the prediction grid."""
        # L'étape est Q des deux côtés ; X_train est la grille Q masquée aux
        # 3 ancres annuelles de a1, X_pred la grille Q complète (12 lignes)
        x_train, ways, origins = materializer.materialize(
            columns=self.COLUMNS,
            grid_index=annual_anchor_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=data,
        )
        x_pred, ways_pred, _ = materializer.materialize(
            columns=self.COLUMNS,
            grid_index=quarterly_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=data,
            materialization=ways,
        )
        # La voie est la même des deux côtés, par construction
        assert ways_pred == ways
        return x_train, x_pred, ways, origins

    def test_reference_example_four_rows(self, reference_timeseries):
        """The four configurations of the table: way of a2 and NaN rates."""
        data = reference_timeseries

        # Ligne 1 — 'tolerate_nan' : a2 à ses ancres des deux côtés. Le taux
        # brut diffère uniquement parce que les grilles n'ont pas le même pas
        tolerate = CovariateMaterializer(covariate_strategy='tolerate_nan')
        x_train, x_pred, ways, _origins = self._fit_and_predict(tolerate, data)
        assert ways == {'m1': 'aggregate', 'q1': 'identity', 'a2': 'raw_anchors'}
        assert x_train['a2'].isna().mean() == pytest.approx(0.0)
        assert x_pred['a2'].isna().mean() == pytest.approx(0.75)

        # Ligne 2 — 'interpolate' : a2 interpolée sur la grille Q
        interpolate = CovariateMaterializer(covariate_strategy='interpolate')
        x_train, x_pred, ways, origins = self._fit_and_predict(interpolate, data)
        assert ways['a2'] == 'interpolate'
        assert origins['a2'] == 'interpolated'
        assert x_train['a2'].isna().mean() == pytest.approx(0.0)
        assert x_pred['a2'].isna().mean() == pytest.approx(0.0)

        # Ligne 3 — 'model', ordre a2 avant a1 : a2 est son imputation Q (rang 2)
        ordered = CovariateMaterializer(covariate_strategy='model')
        prime(ordered, 'a2', pd.Series(5.0 + np.arange(12), index=quarterly_grid()), 'Q', 'model')
        x_train, x_pred, ways, origins = self._fit_and_predict(ordered, data)
        assert ways['a2'] == 'stage_model'
        assert origins['a2'] == 'model'
        assert x_train['a2'].isna().mean() == pytest.approx(0.0)
        assert x_pred['a2'].isna().mean() == pytest.approx(0.0)

        # Ligne 4 — 'model', ordre a1 avant a2 : a2 servie par covariate_fallback
        # (rang 4), des deux côtés. C'est le cas qui produisait 33-67 % de NaN
        # silencieux dans hfi
        unordered = CovariateMaterializer(covariate_strategy='model')
        x_train, x_pred, ways, origins = self._fit_and_predict(unordered, data)
        assert ways['a2'] == 'interpolate'
        assert origins['a2'] == 'interpolated'
        assert x_train['a2'].isna().mean() == pytest.approx(0.0)
        assert x_pred['a2'].isna().mean() == pytest.approx(0.0)
        # Valeurs d'or de a2 aux trois ancres annuelles
        np.testing.assert_allclose(x_train['a2'].to_numpy(), [60.0, 66.0, 72.0])


class TestAggregationConstraintApplier:
    """The injection point of the rescaling: a minimal double suffices."""

    @staticmethod
    def _materialize(materializer, data, columns=('m1', 'q1', 'a1'), grid=None, stage='Q'):
        """Materialize ``columns`` of ``TS`` on one grid."""
        return materializer.materialize(
            columns=list(columns),
            grid_index=quarterly_grid() if grid is None else grid,
            stage_freq=stage,
            detected_frequencies=TS_FREQUENCIES,
            source_data=data,
        )

    @pytest.mark.parametrize(
        'strategy, prime_mirror, expected',
        [
            # m1 agrégée, q1 identité, a1 interpolée : seule a1 est recalée
            ('interpolate', False, ['a1']),
            # a1 à ses ancres : aucune interpolation, aucun recalage
            ('tolerate_nan', False, []),
            # a1 lue dans le miroir de l'étape (rang 2) : aucun recalage
            ('model', True, []),
        ],
        ids=['interpolate', 'raw_anchors', 'stage_model'],
    )
    def test_applier_is_called_for_interpolated_columns_only(
        self, reference_timeseries, strategy, prime_mirror, expected
    ):
        """Identity, exact aggregation, raw anchors and mirror reads are never rescaled."""
        applier = SpyApplier()
        materializer = CovariateMaterializer(
            covariate_strategy=strategy, aggregation_constraint_applier=applier
        )
        if prime_mirror:
            prime(materializer, 'a1', pd.Series(7.0, index=quarterly_grid()), 'Q', 'model')
        self._materialize(materializer, reference_timeseries)
        assert [call['column'] for call in applier.calls] == expected

    def test_applier_receives_the_column_observations_and_frequencies(self, reference_timeseries):
        """Observations at their own anchors, period of the column, grid of the stage."""
        applier = SpyApplier()
        materializer = CovariateMaterializer(aggregation_constraint_applier=applier)
        self._materialize(materializer, reference_timeseries, columns=['a1'], grid=monthly_grid(), stage='M')

        call = applier.calls[0]
        assert (call['period_freq'], call['column'], call['grid_freq']) == ('Y', 'a1', 'M')
        pd.testing.assert_series_equal(
            call['observations'], reference_timeseries['a1'].dropna(), check_names=False, check_freq=False
        )

    def test_applier_is_called_per_entity_with_its_own_period(self, mixed_freq_panel_multifrequency):
        """PANEL-F at M: FR rescaled to its annual totals, DE to its quarterly ones, IT never."""
        applier = SpyApplier()
        CovariateMaterializer(aggregation_constraint_applier=applier).materialize(
            columns=['v'],
            grid_index=entity_grid(['FR', 'DE', 'IT'], monthly_grid()),
            stage_freq='M',
            detected_frequencies=PANEL_F_FREQUENCIES,
            source_data=mixed_freq_panel_multifrequency,
        )
        assert {
            (call['period_freq'], len(call['observations'])) for call in applier.calls
        } == {('Y', 3), ('Q', 12)}

    def test_applier_output_replaces_the_interpolation(self, reference_timeseries):
        """The rescaled values are the features; the returned mask is ignored."""
        plain, _, _ = self._materialize(CovariateMaterializer(), reference_timeseries, columns=['a1'])
        shifted, _, _ = self._materialize(
            CovariateMaterializer(aggregation_constraint_applier=SpyApplier(offset=1000.0)),
            reference_timeseries,
            columns=['a1'],
        )
        np.testing.assert_allclose(shifted['a1'].to_numpy(), plain['a1'].to_numpy() + 1000.0)

    @pytest.mark.parametrize(
        'setting, expected',
        [
            (None, []),
            ({'a1': None, DEFAULT_MATERIALIZATION_KEY: 'sum'}, ['a2']),
            ({'a2': None}, ['a1']),
        ],
        ids=['scalar-none', 'dict-a1-none', 'dict-a2-none-default-sum'],
    )
    def test_applier_is_skipped_when_the_constraint_resolves_to_none(
        self, reference_timeseries, setting, expected
    ):
        """A column whose constraint resolves to None is interpolated un-rescaled."""
        applier = SpyApplier()
        materializer = CovariateMaterializer(
            aggregation_constraint=setting, aggregation_constraint_applier=applier
        )
        self._materialize(materializer, reference_timeseries, columns=['a1', 'a2'])
        assert [call['column'] for call in applier.calls] == expected

    def test_real_constraint_preserves_the_annual_totals(self, reference_timeseries):
        """``AggregationConstraint('sum')``: the months of a1 sum to 120 / 132 / 150."""
        features, _, _ = self._materialize(
            CovariateMaterializer(aggregation_constraint_applier=AggregationConstraint('sum')),
            reference_timeseries,
            columns=['a1'],
            grid=monthly_grid(),
            stage='M',
        )
        yearly = features['a1'].groupby(features.index.year).sum()
        np.testing.assert_allclose(yearly.to_numpy(), [120.0, 132.0, 150.0], rtol=0, atol=1e-9)

    def test_training_value_is_the_exact_aggregate_of_the_prediction_values(
        self, reference_timeseries
    ):
        """``a2`` seen at fit is the sum of ``a2`` seen at predict.

        ``TS``, target Q: the way of a2, decided on the quarterly prediction grid,
        is 'interpolate'; replayed on the annual block of a1, it gives back the
        observations 60 / 66 / 72. Under 'sum', the Q grid is rescaled to those
        same totals (2021 sum = 60).
        """
        materializer = CovariateMaterializer(aggregation_constraint_applier=AggregationConstraint('sum'))
        x_pred, ways, _ = self._materialize(materializer, reference_timeseries, columns=['a2'])
        x_train, _, _ = materializer.materialize(
            columns=['a2'],
            grid_index=annual_grid(),
            stage_freq='Y',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            materialization=ways,
        )
        yearly_pred = x_pred['a2'].groupby(x_pred.index.year).sum().to_numpy()
        np.testing.assert_allclose(x_train['a2'].to_numpy(), yearly_pred, rtol=0, atol=1e-9)
