"""Tests of the coincident cells and of the frequency level.

The 31st of December is both a year end and a quarter end. The builder never
reads ``aggregation_constraint``: the imputer translates ``'sum'`` into
``keep_coincident_cells=False`` (the observation wins) and ``None`` into
``True`` (both cells kept, distinguished by a ``'frequency'`` level on the
entity side). Pinned here: 12 vs 15 rows of ``a1`` / ``a2``
at stage ``M`` on ``TS`` —, the level stamped only when a coincidence
SURVIVES the origin filter and the window, never for two entities sharing a
date, stamped after the materialization (the materializer only sees real
grids), covariates materialized at the frequency of each layer. Under
``'sum'`` the observation wins only where it is itself retained: a produced
cell whose anchor the window excludes stays.
End to end through ``HighFrequencyImputer`` under
``impute_intermediate_frequencies=True``.

Covered symbol: ``TrainingSetBuilder.build`` (``keep_coincident_cells``).
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
from tsforecast.imputation._components.training_set_builder import (
    FREQUENCY_LEVEL_NAME,
    TrainingSet,
    TrainingSetBuilder,
)
from tsforecast.imputation.high_frequency_imputer import HighFrequencyImputer

# Constantes et fabriques partagées du paquet
from tests.unit.imputation._components.training_set_builder.helpers import (
    build_panel_f,
    build_ts,
    layer_mask_provider,
    make_builder,
    seed_panel_mirror,
    seed_quarterly_mirror,
)


# =============================================================================
# Cellules coïncidentes
# =============================================================================
class TestCoincidentCells:
    """The component, unaware of ``aggregation_constraint``."""

    @pytest.mark.parametrize('column', ['a1', 'a2'])
    @pytest.mark.parametrize(
        'keep_coincident_cells, expected_rows, stamped',
        [(False, 12, False), (True, 15, True)],
        ids=['sum', 'None'],
    )
    def test_twelve_vs_fifteen_rows_at_monthly_stage(
        self, reference_timeseries, column, keep_coincident_cells, expected_rows, stamped
    ):
        """3 anchors + 9 imputations under 'sum', + 12 under None."""
        builder = seed_quarterly_mirror(make_builder(), column)

        training = build_ts(
            builder, reference_timeseries, column=column,
            keep_coincident_cells=keep_coincident_cells,
        )

        assert len(training) == expected_rows
        assert training.has_frequency_level is stamped
        # Les ancres sont toujours conservées : l'observation gagne
        assert (training.row_origin == 'observed').sum() == 3

    def test_coincident_cell_is_dropped_by_default(self, reference_timeseries):
        """Default: the observation wins, the index stays the historical one."""
        builder = seed_quarterly_mirror(make_builder(), 'a1')
        training = build_ts(builder, reference_timeseries, feature_cols=('m1',))

        assert not training.has_frequency_level
        assert isinstance(training.X.index, pd.DatetimeIndex)
        assert not training.X.index.duplicated().any()
        anchor = pd.Timestamp('2021-12-31')
        assert training.row_frequency[anchor] == 'Y'
        assert training.y[anchor] == 120.0

    def test_coincident_cell_is_kept_and_stamped(self, reference_timeseries):
        """Under conservation, both cells coexist on a stamped index."""
        builder = seed_quarterly_mirror(make_builder(), 'a1')
        training = build_ts(
            builder, reference_timeseries, feature_cols=('m1',), keep_coincident_cells=True
        )

        assert list(training.X.index.names) == [FREQUENCY_LEVEL_NAME, 'date']

        # Les deux cellules du 2021-12-31 décrivent des périodes différentes
        anchor = pd.Timestamp('2021-12-31')
        assert training.y[('Y', anchor)] == 120.0
        assert training.y[('Q', anchor)] == 31.0
        assert training.row_origin[('Y', anchor)] == 'observed'
        assert training.row_origin[('Q', anchor)] == 'model'

        # Covariables matérialisées à la fréquence de chaque COUCHE : somme de
        # l'année 2021 (1266) sur la ligne annuelle, du T4 (109 + 110 + 111 =
        # 330) sur la ligne trimestrielle, jamais un NaN
        assert training.X['m1'].notna().all()
        assert training.X.loc[('Y', anchor), 'm1'] == 1266.0
        assert training.X.loc[('Q', anchor), 'm1'] == 330.0

    def test_stamping_needs_an_actual_coincidence(self, reference_timeseries):
        """Without coincidence the parameter is inert: same set, same index."""
        kept = build_ts(make_builder(), reference_timeseries, keep_coincident_cells=True)
        dropped = build_ts(make_builder(), reference_timeseries, keep_coincident_cells=False)

        # Miroir vide : y_train ne tient que les 3 ancres, rien à distinguer
        assert len(kept) == len(dropped) == 3
        assert not kept.has_frequency_level
        assert kept.X.index.equals(dropped.X.index)
        pd.testing.assert_series_equal(kept.y, dropped.y)

    @pytest.mark.parametrize('column', ['a1', 'a2'])
    def test_coincidence_must_survive_the_origin_filter(self, reference_timeseries, column):
        """Mirror cells filtered out: no coincidence left, no level."""
        # 'covariates_only' : le miroir porte bien la coïncidence, mais le
        # filtre d'origine l'écarte AVANT le constat
        builder = seed_quarterly_mirror(make_builder(), column)

        training = build_ts(
            builder, reference_timeseries, column=column,
            eligible_origins=('observed',), keep_coincident_cells=True,
        )

        assert len(training) == 3
        assert training.has_frequency_level is False
        assert isinstance(training.X.index, pd.DatetimeIndex)

    def test_coincidence_must_survive_the_window(self, reference_timeseries):
        """A coincident cell outside its layer window stamps nothing."""
        builder = seed_quarterly_mirror(make_builder(), 'a1')
        # Fenêtre trimestrielle sans aucune fin d'année : les trois cellules
        # coïncidentes tombent, les 9 autres trimestres restent
        builder.training_mask = layer_mask_provider(
            lambda _entity, frequency, date: frequency == 'Y' or date.month != 12
        )

        training = build_ts(builder, reference_timeseries, keep_coincident_cells=True)

        assert len(training) == 3 + 9
        assert training.has_frequency_level is False

    def test_produced_cell_kept_when_its_anchor_is_outside_the_window(
        self, reference_timeseries
    ):
        """Under 'sum', a produced cell is dropped only for a RETAINED observation.

        The window excludes the 2021 anchor (its year is not fully covered)
        but keeps the fourth quarter of 2021: the quarterly imputation of
        2021-12-31 is then redundant with nothing, and stays.
        """
        builder = seed_quarterly_mirror(make_builder(), 'a1')
        builder.training_mask = layer_mask_provider(
            lambda _entity, frequency, date: frequency == 'Q' or date.year > 2021
        )

        training = build_ts(builder, reference_timeseries, keep_coincident_cells=False)

        # Valeur d'or : 2 ancres (2022, 2023) + 12 trimestres moins les 2
        # coïncidant avec une ancre retenue = 12 lignes, index de dates seul
        assert len(training) == 2 + 10
        assert training.has_frequency_level is False
        anchor_2021 = pd.Timestamp('2021-12-31')
        assert training.row_frequency[anchor_2021] == 'Q'
        assert training.y[anchor_2021] == 31.0
        # L'observation gagne là où elle est retenue
        assert training.row_frequency[pd.Timestamp('2022-12-31')] == 'Y'

    # Avertissements légitimes de l'imputeur (étapes dégradées) : hors sujet ici
    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize(
        'aggregation_constraint, expected_rows',
        [('sum', 10), (None, 12)],
        ids=['sum', 'None'],
    )
    def test_imputer_keeps_the_quarter_of_an_excluded_year(
        self, reference_timeseries, monkeypatch, aggregation_constraint, expected_rows
    ):
        """End to end: a training window starting mid-2021 keeps Q4 2021 of ``a1``.

        ``m1`` is missing over the first half of 2021, so the strict window
        starts in July: the 2021 anchor is out at ``Y`` (a year is in the
        window only if all its months are), Q3 and Q4 2021 are in at ``Q``.
        """
        data = reference_timeseries.copy()
        data.index.freq = None
        data.loc[:'2021-06-30', 'm1'] = np.nan
        captured: List[TrainingSet] = []
        original = TrainingSetBuilder.build

        def _spy(self, **kwargs):
            training = original(self, **kwargs)
            if kwargs['column'] == 'a1' and len(kwargs['feature_cols']) > 0 \
                    and kwargs['stage_freq'] == 'M':
                captured.append(training)
            return training

        monkeypatch.setattr(TrainingSetBuilder, 'build', _spy)
        imputer = HighFrequencyImputer(
            estimator=LinearRegression(),
            target_frequency='M',
            impute_intermediate_frequencies=True,
            aggregation_constraint=aggregation_constraint,
        )
        imputer.fit(data)

        # Valeur d'or : 10 trimestres dans la fenêtre (T3 2021 → T4 2023) et
        # 2 ancres (2022, 2023) ; sous 'sum', les 2 trimestres coïncidant avec
        # une ancre RETENUE tombent, celui de 2021 reste
        training = captured[-1]
        assert len(training) == expected_rows
        quarter_2021 = (
            ('Q', pd.Timestamp('2021-12-31')) if training.has_frequency_level
            else pd.Timestamp('2021-12-31')
        )
        assert training.row_frequency[quarter_2021] == 'Q'

    def test_two_entities_sharing_a_date_is_no_coincidence(
        self, mixed_freq_panel_multifrequency
    ):
        """FR, DE and IT all observe 2021-12-31: no level, the entity tells them apart."""
        training = build_panel_f(
            make_builder(), mixed_freq_panel_multifrequency, keep_coincident_cells=True
        )

        assert len(training) == 51
        assert training.has_frequency_level is False
        assert list(training.X.index.names) == ['country', 'date']

    def test_stamped_index_on_a_panel_keeps_the_entity_levels(
        self, mixed_freq_panel_multifrequency
    ):
        """On a panel, the level goes BEFORE the date, after the entities."""
        builder = seed_panel_mirror(make_builder(), [('2021-12-31', 31.0, 'model')])

        training = build_panel_f(
            builder, mixed_freq_panel_multifrequency,
            eligible_origins=('observed', 'model'), keep_coincident_cells=True,
        )

        assert training.has_frequency_level
        assert list(training.X.index.names) == ['country', FREQUENCY_LEVEL_NAME, 'date']
        assert ('FR', 'Y', pd.Timestamp('2021-12-31')) in training.X.index
        assert ('FR', 'Q', pd.Timestamp('2021-12-31')) in training.X.index
        # Les entités sans coïncidence portent leur fréquence de bloc
        assert ('DE', 'Q', pd.Timestamp('2021-12-31')) in training.X.index
        assert ('IT', 'M', pd.Timestamp('2021-12-31')) in training.X.index
        # Les 51 lignes du jeu mutualisé, plus la cellule coïncidente conservée
        assert len(training) == 52

    def test_materializer_never_sees_the_frequency_level(self, reference_timeseries):
        """The level is stamped AFTER materialization, on the real grid."""
        builder = seed_quarterly_mirror(make_builder(), 'a1')
        grids = []
        original = builder.materializer.materialize

        def _spy(**kwargs):
            grids.append(kwargs['grid_index'])
            return original(**kwargs)

        builder.materializer.materialize = _spy
        build_ts(builder, reference_timeseries, keep_coincident_cells=True)

        # Deux couches, deux grilles réelles de dates : 3 ancres, 12 trimestres
        assert [len(grid) for grid in grids] == [3, 12]
        assert all(isinstance(grid, pd.DatetimeIndex) for grid in grids)

    # Avertissements légitimes de l'imputeur (étapes dégradées) : hors sujet ici
    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize(
        'aggregation_constraint, expected_rows',
        [('sum', 12), (None, 15)],
        ids=['sum', 'None'],
    )
    def test_imputer_translates_the_aggregation_constraint(
        self, reference_timeseries, monkeypatch, aggregation_constraint, expected_rows
    ):
        """End to end on ``TS`` under ``True``: ``a1`` trains on 12 or 15 rows at ``M``."""
        captured: List[Tuple[object, TrainingSet]] = []
        original = TrainingSetBuilder.build

        def _spy(self, **kwargs):
            training = original(self, **kwargs)
            if kwargs['column'] == 'a1' and len(kwargs['feature_cols']) > 0:
                captured.append((kwargs['stage_freq'], training))
            return training

        monkeypatch.setattr(TrainingSetBuilder, 'build', _spy)
        imputer = HighFrequencyImputer(
            estimator=LinearRegression(),
            target_frequency='M',
            impute_intermediate_frequencies=True,
            aggregation_constraint=aggregation_constraint,
        )
        imputer.fit(reference_timeseries)

        # Jeu complet de a1 à l'étape M (tableau de l'étape M)
        monthly = [training for stage, training in captured if stage == 'M']
        assert monthly
        assert {len(training) for training in monthly} == {expected_rows}
