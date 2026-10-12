"""Tests of the origin filter (§5.3) and of the layers (§5.4bis, D30).

The origin filter decides WHICH cells of an entity are eligible, the
mutualization WHICH entities contribute: the two compose without knowing each
other. ``ELIGIBLE_ORIGINS`` of every modality of
``impute_intermediate_frequencies`` (``False`` and ``'covariates_only'``
share literally one filter), the per-row production frequency of an imputed
row, NaN mirror cells ignored.

Layers: the ``'training'`` window is read once per layer —
``[{FR: Y, DE: Q, IT: M}, {FR: Q}]`` on ``PANEL-F`` once ``FR`` was imputed at
stage ``Q`` (§5.4bis table) —, each layer reading its own window; the
materializer is called once per layer on its sub-grid; a layer emptied by the
window costs no call; the reduction keeps the most degraded way and origin;
an entity absent from the mask, a date absent from it and a provider
answering None follow §7.2; ``blocks`` stays the R1 perimeter when the window
drops every row.

Covered symbol: ``TrainingSetBuilder.build`` (``eligible_origins``,
``training_mask``).
"""
# Modules de base
from typing import Dict, List

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objets testés
from tsforecast.frequency.high_frequency_imputer import ELIGIBLE_ORIGINS

# Constantes et fabriques partagées du paquet
from tests.unit.frequency.training_set_builder.helpers import (
    PANEL_F_BLOCKS,
    build_panel_f,
    build_ts,
    layer_mask_provider,
    make_builder,
    scaled_target,
    seed_panel_mirror,
    seed_quarterly_mirror,
)


# =============================================================================
# §5.3 : filtre d'origine
# =============================================================================
class TestOriginFilter:
    """§5.3: the origin filter, orthogonal to the mutualization."""

    def test_eligible_origins_filter(self, mixed_freq_panel_multifrequency):
        """``{'observed'}`` excludes the imputed cells, a widened set admits them."""
        builder = seed_panel_mirror(make_builder(), [
            ('2021-03-31', 28.0, 'model'),
            ('2021-06-30', 30.0, 'model'),
            ('2021-09-30', 31.0, 'model'),
        ])

        strict = build_panel_f(
            builder, mixed_freq_panel_multifrequency, eligible_origins={'observed'}
        )
        assert len(strict) == 51
        assert set(strict.row_origin) == {'observed'}

        widened = build_panel_f(
            builder,
            mixed_freq_panel_multifrequency,
            eligible_origins={'observed', 'interpolated', 'model'},
        )
        assert len(widened) == 54
        # R5 : ces lignes entrent avec la fréquence lue dans imputed_freq_store
        model_rows = widened.row_origin == 'model'
        assert int(model_rows.sum()) == 3
        assert set(widened.row_frequency[model_rows.to_numpy()]) == {'Q'}
        assert widened.y[model_rows.to_numpy()].tolist() == [28.0, 30.0, 31.0]

    @pytest.mark.parametrize(
        'modality, expected_rows, expected_origins',
        [
            (False, 51, {'observed'}),
            ('covariates_only', 51, {'observed'}),
            (True, 56, {'observed', 'interpolated', 'model'}),
        ],
        ids=['False', 'covariates_only', 'True'],
    )
    def test_eligible_origins_of_each_modality(
        self, mixed_freq_panel_multifrequency, modality, expected_rows, expected_origins
    ):
        """``False`` and ``'covariates_only'`` share LITERALLY the same filter (§5.3)."""
        # Miroir de FR : trois cellules 'model' (étape Q) et deux cellules
        # 'interpolated' (repli d'un modèle en échec), hors fins d'année
        builder = seed_panel_mirror(make_builder(), [
            ('2021-03-31', 28.0, 'model'),
            ('2021-06-30', 30.0, 'model'),
            ('2021-09-30', 31.0, 'model'),
            ('2022-03-31', 31.0, 'interpolated'),
            ('2022-06-30', 33.0, 'interpolated'),
        ])

        training = build_panel_f(
            builder, mixed_freq_panel_multifrequency,
            eligible_origins=ELIGIBLE_ORIGINS[modality],
        )

        assert len(training) == expected_rows
        assert set(training.row_origin) == expected_origins

    def test_model_rows_excluded_when_origin_not_eligible(
        self, mixed_freq_panel_multifrequency
    ):
        """Under ``{'observed', 'interpolated'}``, a ``'model'`` cell stays out."""
        builder = seed_panel_mirror(make_builder(), [('2021-03-31', 28.0, 'model')])

        training = build_panel_f(
            builder,
            mixed_freq_panel_multifrequency,
            eligible_origins={'observed', 'interpolated'},
        )

        assert len(training) == 51
        assert set(training.row_origin) == {'observed'}

    def test_imputed_rows_scale_by_their_own_production_frequency(
        self, mixed_freq_panel_multifrequency
    ):
        """The divisor of an imputed row is that of ITS frequency (§5.4)."""
        builder = seed_panel_mirror(make_builder(), [('2021-03-31', 28.0, 'model')])

        training = build_panel_f(
            builder,
            mixed_freq_panel_multifrequency,
            eligible_origins={'observed', 'interpolated', 'model'},
        )
        scaled = scaled_target(training, 'M')

        # 28 produit au pas trimestriel : diviseur 3, et non le scalaire annuel
        assert round(float(scaled.loc[('FR', '2021-03-31')]), 3) == 9.333
        assert float(scaled.loc[('FR', '2021-12-31')]) == 10.0

    def test_origin_filter_applies_identically_in_each_block(
        self, mixed_freq_panel_multifrequency
    ):
        """The filter never touches the observed rows of the other blocks."""
        builder = seed_panel_mirror(make_builder(), [('2021-03-31', 28.0, 'model')])

        training = build_panel_f(
            builder, mixed_freq_panel_multifrequency,
            eligible_origins={'observed', 'model'},
        )

        # Seul FR gagne une ligne : DE et IT gardent leurs 12 et 36 ancres
        assert training.y.groupby(level=0).size().to_dict() == {
            'DE': 12, 'FR': 4, 'IT': 36
        }

    @pytest.mark.parametrize(
        'values, expected_rows',
        [([28.0, np.nan], 4), ([np.nan, np.nan], 3)],
        ids=['one_nan', 'all_nan'],
    )
    def test_mirror_cell_without_value_is_ignored(
        self, reference_timeseries, values, expected_rows
    ):
        """A NaN produced cell never becomes a training row."""
        builder = make_builder()
        dates = pd.to_datetime(['2021-03-31', '2021-06-30'])
        builder.materializer.imputed_store['a1'] = pd.Series(values, index=dates)
        builder.materializer.imputed_freq_store['a1'] = pd.Series(['Q', 'Q'], index=dates)
        builder.materializer.origin_store['a1'] = pd.Series(['model', 'model'], index=dates)

        training = build_ts(builder, reference_timeseries)

        # 3 ancres + les seules cellules produites renseignées
        assert len(training) == expected_rows
        assert pd.Timestamp('2021-06-30') not in training.y.index


# =============================================================================
# §5.4bis : couches et fenêtre 'training'
# =============================================================================
class TestTrainingWindow:
    """§5.3 point 5 and §5.4bis: the ``'training'`` window, read per layer."""

    def test_window_restricts_each_block_at_its_own_frequency(
        self, mixed_freq_panel_multifrequency
    ):
        """One single layer: the mask is asked once, at the block frequencies."""
        asked: List[Dict] = []
        provider = layer_mask_provider(lambda _entity, _freq, date: date.year < 2023, asked)

        training = build_panel_f(
            make_builder(training_mask=provider), mixed_freq_panel_multifrequency
        )

        assert asked == [PANEL_F_BLOCKS]
        # Fenêtre 2021-2022 : 2 ancres annuelles, 8 trimestrielles, 24 mensuelles
        assert training.y.groupby(level=0).size().to_dict() == {
            'DE': 8, 'FR': 2, 'IT': 24
        }

    def test_window_is_read_once_per_layer(self, mixed_freq_panel_multifrequency):
        """Axis 2 on: layers ``[{FR: Y, DE: Q, IT: M}, {FR: Q}]`` (§5.4bis table)."""
        # FR a été imputée à l'étape Q : quatre trimestres hors fins d'année
        builder = seed_panel_mirror(make_builder(), [
            ('2021-03-31', 28.0, 'model'),
            ('2021-06-30', 30.0, 'model'),
            ('2021-09-30', 31.0, 'model'),
            ('2022-03-31', 31.0, 'model'),
        ])
        asked: List[Dict] = []
        # Fenêtre : 2021-2022 pour les fréquences de bloc, 2021 seule pour la
        # couche trimestrielle de FR — chaque couche lit SA fenêtre
        def keep(entity, frequency, date):
            if entity == ('FR',) and frequency == 'Q':
                return date.year == 2021
            return date.year < 2023
        builder.training_mask = layer_mask_provider(keep, asked)

        training = build_panel_f(
            builder, mixed_freq_panel_multifrequency,
            eligible_origins=('observed', 'model'),
        )

        assert asked == [PANEL_F_BLOCKS, {('FR',): 'Q'}]
        # FR : 2 ancres annuelles + 3 trimestres 2021 (celui de 2022 est hors
        # de la fenêtre de SA couche)
        fr_frequencies = training.row_frequency.loc['FR']
        assert (fr_frequencies == 'Y').sum() == 2
        assert (fr_frequencies == 'Q').sum() == 3
        assert training.y.groupby(level=0).size().to_dict() == {
            'DE': 8, 'FR': 5, 'IT': 24
        }

    def test_materializer_called_once_per_layer(self, mixed_freq_panel_multifrequency):
        """Each layer is materialized on its own sub-grid, at its own frequency."""
        builder = seed_panel_mirror(make_builder(), [('2021-03-31', 28.0, 'model')])
        calls = []
        original = builder.materializer.materialize

        def _spy(**kwargs):
            calls.append((kwargs['stage_freq'], len(kwargs['grid_index'])))
            return original(**kwargs)

        builder.materializer.materialize = _spy
        training = build_panel_f(
            builder, mixed_freq_panel_multifrequency,
            eligible_origins=('observed', 'model'),
        )

        assert calls == [(PANEL_F_BLOCKS, 51), ({('FR',): 'Q'}, 1)]
        # La ligne trimestrielle reçoit ses covariables à l'échelle du
        # trimestre : m1 = 100 + 101 + 102 = 303, jamais la somme de l'année
        assert training.X.loc[('FR', '2021-03-31'), 'm1'] == 303.0

    def test_layer_emptied_by_the_window_is_skipped(self, reference_timeseries):
        """A block layer left without rows by the window costs no materialization."""
        builder = seed_quarterly_mirror(make_builder(), 'a1')
        calls = []
        original = builder.materializer.materialize

        def _spy(**kwargs):
            calls.append(kwargs['stage_freq'])
            return original(**kwargs)

        builder.materializer.materialize = _spy
        # Fenêtre : aucune date annuelle, toutes les dates trimestrielles
        builder.training_mask = layer_mask_provider(
            lambda _entity, frequency, _date: frequency == 'Q'
        )

        training = build_ts(builder, reference_timeseries)

        # Valeur d'or : les 3 ancres sont hors fenêtre, donc les 12
        # imputations trimestrielles restent, les 3 coïncidentes comprises —
        # elles ne sont redondantes avec aucune observation retenue
        assert len(training) == 12
        assert set(training.row_frequency) == {'Q'}
        assert calls == [{(): 'Q'}]
        assert training.X.notna().all().all()

    def test_entity_absent_from_the_mask_is_unrestricted(
        self, mixed_freq_panel_multifrequency
    ):
        """An entity without a fitted mask is not restricted (§7.2)."""
        def _mask(_frequencies):
            # Masque ne couvrant que FR, comme le calculateur le rend lorsque
            # les autres entités n'ont pas de masque valide
            index = mixed_freq_panel_multifrequency.loc[['FR']].index
            return pd.Series(
                index.get_level_values(-1).year < 2023, index=index
            )

        training = build_panel_f(
            make_builder(training_mask=_mask), mixed_freq_panel_multifrequency
        )

        assert training.y.groupby(level=0).size().to_dict() == {
            'DE': 12, 'FR': 2, 'IT': 36
        }

    def test_date_absent_from_the_mask_is_outside_the_window(
        self, mixed_freq_panel_multifrequency
    ):
        """For an entity present in the mask, a missing date drops its row."""
        def _mask(frequencies):
            # Grille annuelle de FR amputée de 2023 : date absente, pas False
            index = pd.MultiIndex.from_tuples(
                [('FR', pd.Timestamp('2021-12-31')), ('FR', pd.Timestamp('2022-12-31'))],
                names=['country', 'date'],
            )
            return pd.Series(True, index=index)

        training = build_panel_f(
            make_builder(training_mask=_mask), mixed_freq_panel_multifrequency
        )

        assert training.y.loc['FR'].tolist() == [120.0, 132.0]

    def test_mask_returning_none_restricts_nothing(self, mixed_freq_panel_multifrequency):
        """A provider answering None (no computable window) keeps every row."""
        training = build_panel_f(
            make_builder(training_mask=lambda _frequencies: None),
            mixed_freq_panel_multifrequency,
        )

        assert len(training) == 51

    def test_blocks_survive_a_window_dropping_every_row(
        self, mixed_freq_panel_multifrequency
    ):
        """``blocks`` is the R1 perimeter, computed before the window."""
        provider = layer_mask_provider(lambda _entity, _freq, _date: False)

        training = build_panel_f(
            make_builder(training_mask=provider), mixed_freq_panel_multifrequency
        )

        # Aucune ligne, mais les trois entités observent `v` : leurs blocs
        # restent la composition du jeu, aucune voie n'est décidée
        assert len(training) == 0
        assert dict(training.blocks) == PANEL_F_BLOCKS
        assert dict(training.ways) == {}

    def test_layer_reduction_keeps_the_most_degraded_way_and_origin(
        self, reference_timeseries
    ):
        """Across layers, a covariate keeps its most degraded way and origin."""
        builder = seed_quarterly_mirror(make_builder(), 'a1')

        # a2 (annuelle) : identité sur la couche annuelle, interpolée sur la
        # couche trimestrielle — la réduction retient l'interpolation
        training = build_ts(builder, reference_timeseries, feature_cols=('m1', 'a2'))

        assert training.ways == {'m1': 'aggregate', 'a2': 'interpolate'}
        assert training.column_origins == {'m1': 'observed', 'a2': 'interpolated'}
