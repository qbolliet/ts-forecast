"""Tests of the central invariant: never a degradation from fit to predict.

A model never sees, at
prediction time, a feature-availability pattern more degraded than at
training time, nor features of a different nature; the single exception,
measured PER ENTITY, is the entity structurally deprived of the column.
The testable formulation is: for each (stage, column), the dates
filled in ``X_pred`` contain the image of the dates filled in ``X_train`` —
not a naive comparison of NaN rates, the two grids having different steps.

Measured on the reference datasets (``TS``, ``PANEL``) and, entity by entity,
on the realistic ``heterogeneous_coverage_panel`` (per-entity coverage,
irregular index, ``depenses_publiques_pib`` at a per-entity frequency,
``climat_affaires`` absent for Italie), under every strategy:

- same stage on both sides: ``X_train`` is ``X_pred`` restricted to the
  training rows — "the same routine on both sides";
- training grid at the target's own frequency, prediction grid at the stage,
  ways decided on the prediction grid and replayed: inclusion of the
  filled dates;
- under ``'interpolate'``: no NaN left inside the observed span of any
  (entity, column).

Covered symbol: ``CovariateMaterializer.materialize``.
"""
# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objet testé
from tsforecast.frequency.covariate_materializer import CovariateMaterializer

# Jeux partagés
from tests.unit.frequency.covariate_materializer.helpers import (
    HETEROGENEOUS_ENTITIES,
    HETEROGENEOUS_FREQUENCIES,
    PANEL_FREQUENCIES,
    TS_FREQUENCIES,
    annual_grid,
    filled_dates,
    panel_grid,
    quarterly_grid,
)


# Configurations (stratégie, repli) sous lesquelles l'invariant est mesuré
CONFIGS = [
    ('tolerate_nan', 'interpolate'),
    ('interpolate', 'interpolate'),
    ('model', 'interpolate'),
    ('model', 'tolerate_nan'),
]
CONFIG_IDS = ['tolerate_nan', 'interpolate', 'model-interpolate', 'model-tolerate_nan']

# Variable imputée du jeu réaliste : le PIB trimestriel, ancré en début de
# trimestre ; ses covariables sont toutes les autres colonnes
TARGET = 'pib_trimestriel'
COVARIATES = [column for column in HETEROGENEOUS_FREQUENCIES if column != TARGET]


class TestCentralInvariant:
    """The NaN invariant, in its testable formulation."""

    @pytest.mark.parametrize('strategy', ['tolerate_nan', 'interpolate'])
    def test_filled_training_dates_are_included_in_the_prediction_ones(
        self, reference_timeseries, strategy
    ):
        """``TS``: the dates filled in X_pred CONTAIN those of X_train.

        Formulation par inclusion des dates, et non une comparaison de pourcentages : sous
        'tolerate_nan' la grille d'entraînement (annuelle) et la grille de
        prédiction (trimestrielle) n'ont pas le même pas, et le taux brut
        diffère légitimement.
        """
        columns = ['m1', 'q1', 'a2']
        materializer = CovariateMaterializer(covariate_strategy=strategy)
        x_train, ways, _ = materializer.materialize(
            columns=columns,
            grid_index=annual_grid(),
            stage_freq='Y',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        x_pred, ways_pred, _ = materializer.materialize(
            columns=columns,
            grid_index=quarterly_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            materialization=ways,
        )
        # La voie est la même des deux côtés, et l'inclusion vaut colonne par colonne
        violations = [
            column for column in columns
            if not filled_dates(x_train, column) <= filled_dates(x_pred, column)
        ]
        assert (ways_pred, violations) == (ways, [])

    def test_invariant_nan_by_entity(self, mixed_freq_panel_heterogeneous):
        """``PANEL``: the inclusion holds PER ENTITY; IT's empty column is no exception to it.

        L'entité sans la colonne est l'unique exception à l'invariant :
        elle est vide des deux côtés, jamais dégradée d'un seul.
        """
        data = mixed_freq_panel_heterogeneous
        panel_columns = ['m1', 'q1', 'a2', 'climat_affaires']
        materializer = CovariateMaterializer(covariate_strategy='tolerate_nan')
        x_train, ways, _ = materializer.materialize(
            columns=panel_columns,
            grid_index=panel_grid(data, annual_grid()),
            stage_freq='Y',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
        )
        x_pred, ways_pred, _ = materializer.materialize(
            columns=panel_columns,
            grid_index=panel_grid(data, quarterly_grid()),
            stage_freq='Q',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
            materialization=ways,
        )
        assert ways_pred == ways

        for entity in ('FR', 'DE', 'IT'):
            for column in panel_columns:
                trained = filled_dates(x_train, column, entity)
                predicted = filled_dates(x_pred, column, entity)
                assert trained <= predicted, (entity, column)
        # L'entité italienne est vide des deux côtés, pas dégradée d'un côté
        assert x_train.loc['IT', 'climat_affaires'].isna().all()
        assert x_pred.loc['IT', 'climat_affaires'].isna().all()


class TestRealisticPanelInvariant:
    """heterogeneous_coverage_panel, entity by entity, under every strategy."""

    @staticmethod
    def _predict(materializer, data):
        """Materialize the covariates on the whole index, at stage M.

        Returns:
            Tuple ``(x_pred, ways)``.
        """
        x_pred, ways, _origins = materializer.materialize(
            columns=COVARIATES,
            grid_index=data.index,
            stage_freq='M',
            detected_frequencies=HETEROGENEOUS_FREQUENCIES,
            source_data=data,
            record=False,
        )
        return x_pred, ways

    @pytest.mark.parametrize('strategy, fallback', CONFIGS, ids=CONFIG_IDS)
    def test_training_rows_equal_the_prediction_rows_at_one_stage(
        self, heterogeneous_coverage_panel, strategy, fallback
    ):
        """Same stage on both sides: X_train is X_pred restricted to the training rows.

        Grille d'entraînement : les débuts de trimestre de chaque entité, un
        sous-ensemble de la grille de prédiction ; la voie décidée sur la
        prédiction y est rejouée. Une seule routine produit les deux côtés.
        """
        data = heterogeneous_coverage_panel
        materializer = CovariateMaterializer(covariate_strategy=strategy, covariate_fallback=fallback)
        x_pred, ways = self._predict(materializer, data)

        dates = data.index.get_level_values('date')
        train_grid = data.index[dates.month.isin([1, 4, 7, 10])]
        x_train, _ways, _origins = materializer.materialize(
            columns=COVARIATES,
            grid_index=train_grid,
            stage_freq='M',
            detected_frequencies=HETEROGENEOUS_FREQUENCIES,
            source_data=data,
            materialization=ways,
            record=False,
        )
        pd.testing.assert_frame_equal(x_train, x_pred.loc[train_grid])

    @pytest.mark.parametrize('entity', HETEROGENEOUS_ENTITIES)
    @pytest.mark.parametrize('strategy, fallback', CONFIGS, ids=CONFIG_IDS)
    def test_filled_training_dates_are_included_in_the_prediction_ones(
        self, heterogeneous_coverage_panel, strategy, fallback, entity
    ):
        """Across frequencies: training at the target's anchors (Q), prediction at M.

        La grille d'entraînement est celle des ancres trimestrielles de la
        cible, à l'étape Q ; la grille de prédiction, tout l'index à l'étape
        M. Les voies, décidées sur la prédiction, sont rejouées à
        l'entraînement et ramenées aux fréquences de chaque entité.
        """
        data = heterogeneous_coverage_panel
        materializer = CovariateMaterializer(covariate_strategy=strategy, covariate_fallback=fallback)
        x_pred, ways = self._predict(materializer, data)
        x_train, _ways, _origins = materializer.materialize(
            columns=COVARIATES,
            grid_index=data[TARGET].dropna().index,
            stage_freq='Q',
            detected_frequencies=HETEROGENEOUS_FREQUENCIES,
            source_data=data,
            materialization=ways,
            record=False,
        )
        violations = [
            column for column in COVARIATES
            if not filled_dates(x_train, column, entity) <= filled_dates(x_pred, column, entity)
        ]
        assert violations == []

    @pytest.mark.parametrize('strategy, fallback', CONFIGS, ids=CONFIG_IDS)
    def test_structurally_absent_column_is_nan_on_both_sides(
        self, heterogeneous_coverage_panel, strategy, fallback
    ):
        """climat_affaires / Italie: empty at fit AND at predict, the only exception."""
        data = heterogeneous_coverage_panel
        materializer = CovariateMaterializer(covariate_strategy=strategy, covariate_fallback=fallback)
        x_pred, ways = self._predict(materializer, data)
        x_train, _ways, _origins = materializer.materialize(
            columns=COVARIATES,
            grid_index=data[TARGET].dropna().index,
            stage_freq='Q',
            detected_frequencies=HETEROGENEOUS_FREQUENCIES,
            source_data=data,
            materialization=ways,
            record=False,
        )
        assert (
            x_train.loc['Italie', 'climat_affaires'].isna().all(),
            x_pred.loc['Italie', 'climat_affaires'].isna().all(),
        ) == (True, True)

    @pytest.mark.parametrize('entity', HETEROGENEOUS_ENTITIES)
    def test_interpolate_leaves_no_nan_inside_an_observed_span(self, heterogeneous_coverage_panel, entity):
        """Within [first, last observation] of each (entity, column), X_pred has no NaN.

        Seuls restent permis les bords (avant la première, après la dernière
        observation) et l'entité dépourvue de la colonne.
        """
        data = heterogeneous_coverage_panel
        x_pred, _ways = self._predict(CovariateMaterializer(covariate_strategy='interpolate'), data)

        violations = []
        for column in COVARIATES:
            observed = data.loc[entity, column].dropna()
            if observed.empty:
                continue
            inside = x_pred.loc[entity, column].loc[observed.index.min():observed.index.max()]
            if inside.isna().any():
                violations.append(column)
        assert violations == []
