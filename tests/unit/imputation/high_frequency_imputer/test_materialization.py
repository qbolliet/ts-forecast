"""Covariate materialization ways of ``HighFrequencyImputer``.

Covers the way recorded per plan step, its per-entity degradation on
the pooled grid, the covariates left out because the prediction grid only
receives them interpolated while the variable's anchors observe them, the
covariates already imputed at the stage read imputed on both grids, and the
replay of the ways at ``transform``.
"""

# Modules de base
import pandas as pd
import pytest

# Objets testés
from tsforecast.imputation.high_frequency_imputer import HighFrequencyImputer
from tsforecast.imputation._components.covariate_materializer import UNINFORMATIVE_COVARIATE

# Outils de test partagés
from tests.unit.imputation.high_frequency_imputer.helpers import (
    FR,
    DE,
    IT,
    entity_block,
    fit_with_spy,
    transform_quietly,
)


# Étape de plan imputant une colonne donnée
def step_of(imputer: HighFrequencyImputer, column: str):
    """Return the first plan step imputing ``column``."""
    return next(step for step in imputer.imputation_plan_ if step.var_name == column)


class TestMaterializationWays:
    """The materialization way is recorded and follows the precedence."""

    def test_materialization_way_recorded_per_step(self, reference_timeseries):
        """Une entrée par feature_col, et la voie attendue par la précédence.

        Les deux annuelles ne se lisent pas l'une l'autre : chacune est
        observée à la fréquence des ancres de l'autre et seulement interpolée
        au mois.
        """
        imputer = fit_with_spy(reference_timeseries)
        expected = {
            'q1': {'m1': 'identity', 'a1': 'interpolate', 'a2': 'interpolate'},
            'a1': {'m1': 'identity', 'q1': 'interpolate'},
            'a2': {'m1': 'identity', 'q1': 'interpolate'},
        }
        for step in imputer.imputation_plan_:
            assert set(step.materialization) == set(step.feature_cols)
            assert dict(step.materialization) == expected[step.var_name]

    def test_ways_are_raw_anchors_under_tolerate_nan(self, reference_timeseries):
        """Sous 'tolerate_nan', aucune covariable n'est matérialisée au-delà de ses ancres."""
        imputer = fit_with_spy(reference_timeseries, covariate_strategy='tolerate_nan')
        for step in imputer.imputation_plan_:
            for column, way in step.materialization.items():
                assert way in ('identity', 'aggregate', 'raw_anchors')


class TestPerEntityMaterialization:
    """La voie unique de l'étape se dégrade entité par entité sur la grille des blocs."""

    def test_one_covariate_takes_three_ways_on_the_pooled_grid(
        self, mixed_freq_panel_multifrequency
    ):
        """Une covariable imputée à l'étape est agrégée, lue, puis lue imputée selon le bloc.

        C'est le cas que la mutualisation crée : la grille d'entraînement de
        ``v`` réunit un bloc annuel (FR), un bloc trimestriel (DE) et un bloc
        mensuel (IT). ``q1``, trimestrielle et déjà imputée au mois sous
        ``'model'``, y couvre les trois positions possibles face à la grille —
        plus fine, égale, plus basse — et reçoit la transformation propre à
        chacune, sans que la voie de l'étape, décidée une seule fois sur la
        grille de prédiction, soit remise en cause.
        """
        imputer = fit_with_spy(mixed_freq_panel_multifrequency, covariate_strategy='model')
        step = step_of(imputer, 'v')

        # Voie de l'étape : décidée sur la grille de prédiction, mensuelle
        assert step.materialization['q1'] == 'stage_model'
        assert step.materialization['m1'] == 'identity'
        # Un bloc par entité, chacun à la fréquence propre de v
        assert dict(step.training_blocks) == {FR: 'Y', DE: 'Q', IT: 'M'}

        # Dégradation, bloc par bloc, de cette voie unique, lue sur les
        # valeurs de q1 que le modèle a réellement reçues. q1 vaut 10, 20, ...,
        # 120 aux fins de trimestre, pour les trois entités ; la mise à
        # l'échelle ramène toute cellule au niveau mensuel de l'étape
        X_train = step.model.fit_X_

        # Bloc annuel de FR : q1 y est plus fine, donc AGRÉGÉE.
        # Valeur d'or : somme des quatre trimestres / 12 mois
        france = entity_block(X_train['q1'], 'FR')
        assert france.tolist() == pytest.approx([100 / 12, 260 / 12, 420 / 12])

        # Bloc trimestriel de DE : q1 y est à son pas, donc LUE telle quelle.
        # Valeur d'or : observation / 3 mois
        germany = entity_block(X_train['q1'], 'DE')
        assert germany.tolist() == pytest.approx(
            [value / 3 for value in range(10, 130, 10)]
        )

        # Bloc mensuel d'IT : q1 y est plus basse, donc lue IMPUTÉE, recalée
        # sous 'sum' : les trois mois de chaque trimestre somment à son
        # observation
        italy = entity_block(X_train['q1'], 'IT')
        by_quarter = italy.groupby(italy.index.to_period('Q')).sum()
        assert by_quarter.tolist() == pytest.approx(list(range(10, 130, 10)))

    def test_the_three_ways_land_on_the_same_scale(
        self, mixed_freq_panel_multifrequency
    ):
        """Les trois voies produisent des valeurs comparables une fois mises à l'échelle.

        C'est la contrepartie de la dégradation : trois transformations
        différentes, un seul niveau. Sans le diviseur fondé sur la période que
        la cellule couvre, l'agrégat annuel de ``q1`` chez FR entrerait dans le
        modèle douze fois plus grand que sa valeur mensuelle chez IT.
        """
        imputer = fit_with_spy(mixed_freq_panel_multifrequency, covariate_strategy='model')
        X_train = step_of(imputer, 'v').model.fit_X_

        means = {
            entity: float(entity_block(X_train['q1'], entity).mean())
            for entity in ('FR', 'DE', 'IT')
        }
        # Les trois blocs portent le même niveau mensuel, à la dispersion des
        # valeurs près : aucun rapport de 3 ni de 12 entre eux
        reference = means['IT']
        for entity, mean in means.items():
            assert mean == pytest.approx(reference, rel=0.35), entity

        # Et la prédiction lit la même échelle que l'entraînement
        for prediction_frame in step_of(imputer, 'v').model.predict_X_:
            assert float(prediction_frame['q1'].mean()) == pytest.approx(
                reference, rel=0.35
            )


class TestUninformativeCovariates:
    """A covariate the prediction grid only receives interpolated is left out if the anchors observe it.

    Observée à la fréquence des ancres de la variable, elle est exacte sur
    chaque ligne d'entraînement ; interpolée sur la grille de prédiction, elle
    n'y apporte que le niveau de la période, que le recalage impose déjà.
    """

    @pytest.mark.parametrize('strategy', ['interpolate', 'model'])
    def test_annual_covariate_of_an_annual_variable_is_left_out(self, reference_timeseries, strategy):
        """a2, annual, is not a covariate of a1, annual, and the plan says why."""
        imputer = fit_with_spy(reference_timeseries, covariate_strategy=strategy)
        step = step_of(imputer, 'a1')

        assert 'a2' not in step.feature_cols
        assert dict(step.dropped_covariates) == {'a2': UNINFORMATIVE_COVARIATE}
        frame = imputer.imputation_plan_.to_diagnostic_frame()
        assert frame.set_index('variable').loc['a1', 'dropped_covariates'] == 'a2'

    def test_finer_covariate_keeps_its_place(self, reference_timeseries):
        """q1, quarterly, is finer than the annual anchors of a1: still a covariate, interpolated."""
        step = step_of(fit_with_spy(reference_timeseries), 'a1')
        assert step.materialization['q1'] == 'interpolate'

    def test_coarser_covariates_keep_their_place(self, reference_timeseries):
        """a1 and a2, annual, are interpolated on both grids of q1: still covariates."""
        step = step_of(fit_with_spy(reference_timeseries), 'q1')
        assert dict(step.dropped_covariates) == {}
        assert (step.materialization['a1'], step.materialization['a2']) == ('interpolate', 'interpolate')


class TestImputedCovariateOnBothGrids:
    """A covariate already imputed at the stage is read imputed, never interpolated.

    Le cas : la grille de prédiction est trimestrielle, la grille
    d'entraînement de la variable annuelle ``a2`` l'est aussi, et ``a1`` a déjà
    été imputée à l'étape. La prédiction lit l'imputation trimestrielle de
    ``a1`` ; l'entraînement lit ses ancres annuelles, qui sont sous ``'sum'``
    l'agrégat exact de cette imputation.
    """

    @staticmethod
    def _second_step(reference_timeseries):
        """Fit under 'model' at a quarterly target and return the step of a2."""
        imputer = fit_with_spy(
            reference_timeseries, target_frequency='Q', covariate_strategy='model'
        )
        return step_of(imputer, 'a2')

    def test_the_imputed_covariate_is_never_left_out(self, reference_timeseries):
        """a1, observed at the annual frequency of a2's anchors, stays: it is imputed at the stage."""
        step = self._second_step(reference_timeseries)
        assert step.materialization['a1'] == 'stage_model'
        assert 'a1' not in step.dropped_covariates

    def test_the_mirror_never_overrides_an_observation(self, reference_timeseries):
        """Sous 'model', le rang 1 passe AVANT la lecture du miroir.

        Une entité qui observe 'a1' à la fréquence de sa grille
        d'entraînement y lit son observation. Le miroir ne remplace jamais une
        vraie valeur.
        """
        step = self._second_step(reference_timeseries)
        # Grille d'entraînement annuelle : les observations de 'a1',
        # 120 / 132 / 150 divisées par 4
        assert step.model.fit_X_['a1'].tolist() == [30.0, 33.0, 37.5]
        observed = reference_timeseries['a1'].dropna()
        assert observed.tolist() == [120.0, 132.0, 150.0]

    def test_the_training_value_is_the_aggregate_of_the_prediction_values(self, reference_timeseries):
        """Under ``'sum'``, the annual training value of a1 is the sum of its quarterly imputation."""
        step = self._second_step(reference_timeseries)
        predicted = step.model.predict_X_[0]['a1']

        yearly = predicted.groupby(predicted.index.year).sum()
        assert yearly.tolist() == pytest.approx([120.0, 132.0, 150.0])
        # Même niveau des deux côtés, une fois l'échelle appliquée
        assert step.model.fit_X_['a1'].mean() == pytest.approx(predicted.mean())


class TestMaterializationReplay:
    """The materialization way and its values are those of the fit."""

    def test_materialization_identical_fit_and_transform_on_panel(
        self, mixed_freq_panel_multifrequency
    ):
        """On PANEL-F, same ways and same prediction frames at replay."""
        imputer = fit_with_spy(mixed_freq_panel_multifrequency, covariate_strategy='model')
        ways_before = {
            (step.pred_freq_label, step.var_key): dict(step.materialization)
            for step in imputer.imputation_plan_
        }
        fit_frames = {
            key: [frame.copy() for frame in model.predict_X_]
            for key, model in imputer.imputation_models_.items()
        }

        transform_quietly(imputer, mixed_freq_panel_multifrequency)

        assert {
            (step.pred_freq_label, step.var_key): dict(step.materialization)
            for step in imputer.imputation_plan_
        } == ways_before
        for key, model in imputer.imputation_models_.items():
            fitted = fit_frames[key]
            assert fitted
            assert len(model.predict_X_) == 2 * len(fitted)
            for before, after in zip(fitted, model.predict_X_[len(fitted):]):
                pd.testing.assert_frame_equal(before, after)

    def test_materialization_identical_fit_and_transform(self, reference_timeseries):
        """Same ways and same values produced at fit and at transform."""
        imputer = fit_with_spy(reference_timeseries, covariate_strategy='model')

        # Voies figées, relevées avant le rejeu
        plan_before = imputer.imputation_plan_.to_diagnostic_frame()
        ways_before = {
            (step.pred_freq_label, step.var_key): dict(step.materialization)
            for step in imputer.imputation_plan_
        }
        # Trames de prédiction du fit, par modèle
        models = imputer.imputation_models_
        fit_frames = {
            key: [frame.copy() for frame in getattr(model, 'predict_X_', [])]
            for key, model in models.items()
        }

        transform_quietly(imputer, reference_timeseries)

        # Le plan n'a pas bougé : les voies sont rejouées, jamais redécidées
        pd.testing.assert_frame_equal(
            plan_before, imputer.imputation_plan_.to_diagnostic_frame()
        )
        ways_after = {
            (step.pred_freq_label, step.var_key): dict(step.materialization)
            for step in imputer.imputation_plan_
        }
        assert ways_after == ways_before

        # La NATURE des valeurs produites est la même : les trames de
        # prédiction du rejeu sont, une à une, celles du fit
        for key, model in models.items():
            produced = getattr(model, 'predict_X_', [])
            fitted = fit_frames[key]
            assert len(produced) == 2 * len(fitted)
            for before, after in zip(fitted, produced[len(fitted):]):
                pd.testing.assert_frame_equal(before, after)
