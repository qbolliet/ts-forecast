"""Covariate materialization ways of ``HighFrequencyImputer`` (§4).

Covers the way recorded per plan step (§4.6), its per-entity degradation on
the pooled grid (D22), the true values kept on the training grid, and the
replay of the ways at ``transform`` (I11).
"""

# Modules de base
import pandas as pd
import pytest

# Objets testés
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer

# Outils de test partagés
from tests.unit.frequency.high_frequency_imputer.helpers import (
    FR,
    DE,
    IT,
    entity_block,
    fit_with_spy,
    transform_quietly,
)


class TestMaterializationWays:
    """I11 — la voie de matérialisation est enregistrée et conforme à la précédence."""

    def test_materialization_way_recorded_per_step(self, reference_timeseries):
        """Une entrée par feature_col, et la voie attendue par la précédence."""
        imputer = fit_with_spy(reference_timeseries)
        expected = {
            'q1': {'m1': 'identity', 'a1': 'interpolate', 'a2': 'interpolate'},
            'a1': {'m1': 'identity', 'q1': 'interpolate', 'a2': 'interpolate'},
            'a2': {'m1': 'identity', 'q1': 'interpolate', 'a1': 'interpolate'},
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
        """Une covariable trimestrielle est agrégée, lue, puis interpolée selon le bloc.

        C'est le cas que la mutualisation crée : la grille d'entraînement de
        ``v`` réunit un bloc annuel (FR), un bloc trimestriel (DE) et un bloc
        mensuel (IT). ``q1``, trimestrielle, y couvre les trois positions
        possibles face à la grille — plus fine, égale, plus basse — et doit
        recevoir la transformation propre à chacune, sans que la voie de
        l'étape, décidée une seule fois sur la grille de prédiction, soit
        remise en cause.
        """
        imputer = fit_with_spy(mixed_freq_panel_multifrequency)
        step = next(
            step for step in imputer.imputation_plan_ if step.var_name == 'v'
        )

        # Voie de l'étape : décidée sur la grille de prédiction, mensuelle
        assert step.materialization['q1'] == 'interpolate'
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

        # Bloc mensuel d'IT : q1 y est plus basse, donc INTERPOLÉE, puis recalée
        # sous 'sum' : les trois mois de chaque trimestre somment à son
        # observation (I4, covariable matérialisée comprise)
        italy = entity_block(X_train['q1'], 'IT')
        by_quarter = italy.groupby(italy.index.to_period('Q')).sum()
        assert by_quarter.tolist() == pytest.approx(list(range(10, 130, 10)))
        # La forme intra-trimestrielle est celle de l'interpolation linéaire.
        # Valeur d'or du T2 2021 : entre les ancres 10 (mars) et 20 (juin), la
        # droite passe par 40/3, 50/3, 60/3, soit les proportions 4:5:6,
        # recalées sur le total 20
        assert italy.loc['2021-04':'2021-06'].tolist() == pytest.approx(
            [20 * 4 / 15, 20 * 5 / 15, 20 * 6 / 15]
        )

    def test_the_three_ways_land_on_the_same_scale(
        self, mixed_freq_panel_multifrequency
    ):
        """Les trois voies produisent des valeurs comparables une fois mises à l'échelle.

        C'est la contrepartie de la dégradation : trois transformations
        différentes, un seul niveau. Sans le diviseur fondé sur la période que
        la cellule couvre, l'agrégat annuel de ``q1`` chez FR entrerait dans le
        modèle douze fois plus grand que sa valeur mensuelle chez IT.
        """
        imputer = fit_with_spy(mixed_freq_panel_multifrequency)
        step = next(
            step for step in imputer.imputation_plan_ if step.var_name == 'v'
        )
        X_train = step.model.fit_X_

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
        for prediction_frame in step.model.predict_X_:
            assert float(prediction_frame['q1'].mean()) == pytest.approx(
                reference, rel=0.35
            )


class TestTrueValuesOnTheTrainingGrid:
    """Une covariable à sa propre fréquence garde ses VRAIES valeurs au fit.

    Le cas : la grille de prédiction est trimestrielle et la covariable
    annuelle — elle doit donc y être matérialisée —, tandis que la grille
    d'entraînement, elle, est annuelle, parce que la variable imputée l'est
    aussi. La covariable y est disponible à sa propre maille : le rang 1 de la
    précédence l'emporte, et ce sont ses observations qui entrent dans
    ``X_train``, jamais une version reconstruite.
    """

    @staticmethod
    def _annual_step(imputer: HighFrequencyImputer):
        """Return the plan step imputing 'a1', whose training block is annual."""
        return next(
            step for step in imputer.imputation_plan_ if step.var_name == 'a1'
        )

    def test_annual_covariate_enters_the_fit_with_its_observations(
        self, reference_timeseries
    ):
        """Sur la grille annuelle, 'a2' vaut ses ancres, pas une interpolation."""
        imputer = fit_with_spy(reference_timeseries, target_frequency='Q')
        step = self._annual_step(imputer)

        # La voie de l'étape est bien celle de la grille de PRÉDICTION
        assert step.materialization['a2'] == 'interpolate'
        # Mais la grille d'entraînement est annuelle, comme 'a2'
        assert dict(step.training_blocks) == {(): 'Y'}

        # Les valeurs vues au fit sont les VRAIES, à l'échelle de l'étape :
        # 60 / 66 / 72 divisées par les 4 trimestres d'une année
        observed = reference_timeseries['a2'].dropna()
        assert observed.tolist() == [60.0, 66.0, 72.0]
        assert step.model.fit_X_['a2'].tolist() == [15.0, 16.5, 18.0]

    def test_the_two_grids_stay_consistent_under_the_sum_constraint(
        self, reference_timeseries
    ):
        """La valeur d'entraînement est l'agrégat exact des valeurs de prédiction.

        C'est ce qui rend l'asymétrie inoffensive : le modèle apprend sur
        l'observation annuelle et prédit sur son interpolation trimestrielle,
        mais sous ``aggregation_constraint='sum'`` la seconde somme exactement
        à la première. Les deux côtés portent le même niveau ; seule la forme
        intra-annuelle diffère, et la grille annuelle ne pouvait de toute
        façon pas l'exprimer.
        """
        imputer = fit_with_spy(reference_timeseries, target_frequency='Q')
        step = self._annual_step(imputer)
        predicted = step.model.predict_X_[0]['a2']

        yearly = predicted.groupby(predicted.index.year).sum()
        assert yearly.tolist() == pytest.approx([60.0, 66.0, 72.0])
        # Même niveau des deux côtés, une fois l'échelle appliquée (I5)
        assert step.model.fit_X_['a2'].mean() == pytest.approx(predicted.mean())

    def test_the_step_is_nonetheless_marked_as_having_read_an_interpolation(
        self, reference_timeseries
    ):
        """La souillure est le MAX sur les deux grilles : la prédiction commande.

        Garder les vraies valeurs au fit ne blanchit pas l'étape : la souillure
        se calcule sur ``X_train ∪ X_pred`` (§6.2), et la grille de prédiction
        y apporte de l'interpolé. La provenance émise reste
        ``MODEL_ON_INTERPOLATED``.
        """
        imputer = fit_with_spy(reference_timeseries, target_frequency='Q')
        step = self._annual_step(imputer)

        assert step.covariate_taint == 'interpolated'
        assert step.target_taint == 'none'
        assert str(step.emitted_provenance) == 'model_on_interpolated'

    def test_the_mirror_never_overrides_an_observation(self, reference_timeseries):
        """Sous 'model', le rang 1 passe AVANT la lecture du miroir.

        Même quand 'a2' a déjà été imputée à l'étape courante — donc lisible au
        rang 2 —, une entité qui l'observe à la fréquence de sa grille
        d'entraînement y lit son observation. Le miroir ne remplace jamais une
        vraie valeur.
        """
        imputer = fit_with_spy(
            reference_timeseries, target_frequency='Q', covariate_strategy='model'
        )
        # 'a1' est imputée en premier ; à l'étape de 'a2', 'a1' est au miroir
        second = next(
            step for step in imputer.imputation_plan_ if step.var_name == 'a2'
        )
        assert second.materialization['a1'] == 'stage_model'

        # Et pourtant la grille d'entraînement de 'a2', annuelle, lit les
        # observations de 'a1' : 120 / 132 / 150 divisées par 4
        assert second.model.fit_X_['a1'].tolist() == [30.0, 33.0, 37.5]
        observed = reference_timeseries['a1'].dropna()
        assert observed.tolist() == [120.0, 132.0, 150.0]


class TestMaterializationReplay:
    """I11 — la voie de matérialisation et ses valeurs sont celles du fit."""

    def test_materialization_identical_fit_and_transform_on_panel(
        self, mixed_freq_panel_multifrequency
    ):
        """I11 sur PANEL-F — mêmes voies et mêmes trames de prédiction au rejeu."""
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
        """I11 — mêmes voies et mêmes valeurs produites au fit et au transform."""
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
