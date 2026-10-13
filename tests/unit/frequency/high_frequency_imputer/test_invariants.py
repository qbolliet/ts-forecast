"""Cross-cutting invariants of ``HighFrequencyImputer`` (§16).

Covers I1, I2, I3, I4, I5, I10, I13, I16, I17 and the worked examples of
§5.5 and §6.5. The module also holds the traceability of the twenty-one
invariants of §16 to the tests of the whole package, checked against the tests
as they are (prompt F10, 2026-10-12; gaps filled the same day). Paths are
relative to ``tests/unit/frequency/high_frequency_imputer/``; §16 asks each
invariant on the time series AND on the panel, and every one is now measured
on both where §16 names a dataset.

======  =====================================================================
Code    Tests (``module::Class::test``) — coverage notes
======  =====================================================================
I1      ``test_invariants::TestTransformSymmetry`` (3 tests, TS and PANEL-F,
        six axis combinations), ``test_transform::TestSharedModelReplay``,
        ``test_transform::TestTransformState::
        test_never_imputed_cells_stay_unmarked_after_transform`` (TS) and
        ``::test_fit_and_replay_provenances_agree_on_a_panel`` (fit matrix ==
        replay matrix, never-imputed cells unmarked, ANO-FREQ-050).
I2      ``test_invariants::TestNaNInvariant`` — TS and PANEL (entity without
        feature included), three strategies.
I3      ``test_invariants::TestOrderInvariance::test_column_order_invariance``
        (TS) and ``::test_column_order_invariance_on_panel`` (PANEL), three
        strategies each.
I4      ``test_invariants::TestAggregationAdditivity`` (TS, PANEL-F),
        ``TestReferenceExamples::test_reference_provenance_of_spec_6_5``,
        ``test_fit_phases::TestPhaseFiveEdgeCases::
        test_incomplete_leading_and_trailing_periods``,
        ``test_transform::TestIrregularIndexExtendedWindow::
        test_additivity_holds_on_the_isolated_years``, and the interpolated
        covariate of ``test_materialization::TestPerEntityMaterialization::
        test_one_covariate_takes_three_ways_on_the_pooled_grid``, and the
        covariate carried from an earlier stage (rank 3):
        ``TestAggregationAdditivity::
        test_carried_covariate_sums_to_the_period_total``.
I5      ``test_invariants::TestTrainPredictScale`` (TS, both mixed cases of
        §16), ``test_materialization::TestPerEntityMaterialization::
        test_the_three_ways_land_on_the_same_scale`` (PANEL-F).
I6      ``test_provenance::TestProvenanceFamilies``,
        ``test_frequency_progression::TestOriginFilter::
        test_target_taint_families``, ``test_fit_phases::TestFallbackPath::
        test_estimator_failure_falls_back_and_marks_interpolated``,
        ``test_mutualization::TestUnobservedEntities::
        test_model_unanchored_is_never_emitted_by_default`` (panel),
        ``test_frequency_progression::TestPerRowScale::
        test_target_taint_of_a_contributing_entity_propagates`` (panel).
I7      ``test_transform::TestTransformOutsideWindow`` (TS, and
        ``test_transform_outside_fit_window_on_panel``).
I8      ``test_inverse_transform::TestInverseTransform`` (TS and two-level
        panel), ``test_transform::TestReservedFrequencyLevel::
        test_inverse_reads_the_level_by_position``. Right after a fit under
        ``restore_original_values=True``: ``TestInverseTransform::
        test_restoring_inverse_right_after_a_fit`` (ANO-FREQ-051).
I9      ``test_sklearn`` — ``TestParameterContract`` (clone, get_params,
        set_params, NotFittedError), ``TestSklearnConformance`` (Pipeline,
        GridSearchCV on a panel with a dict target), ``TestEstimatorChecks``,
        ``TestCloneAndNestedParams``, ``TestNotFitted``, ``TestPersistence``,
        ``TestFeaturesSeenAtFit``. §16 credits
        ``TestParameterValidation`` with clone / get_params / NotFittedError:
        they live in ``TestParameterContract``.
I10     ``test_invariants::TestOrderInvariance::
        test_processing_order_indifferent_outside_model`` (TS) and
        ``::test_processing_order_indifferent_outside_model_on_panel``.
I11     ``test_materialization::TestMaterializationReplay`` (fit vs replay
        frames one by one, TS and PANEL-F), ``TestMaterializationWays``.
I12     ``test_frequency_progression::TestOriginFilter::
        test_covariates_only_differs_from_true`` (TS) and
        ``::test_covariates_only_differs_from_true_on_panel`` (PANEL; final
        values equal to False under ``'interpolate'``).
I13     ``test_invariants::TestStaticInvariants::
        test_no_boolean_test_on_impute_intermediate_frequencies``,
        ``test_sklearn::TestParameterContract::
        test_impute_intermediate_frequencies_covariates_only_is_not_true``.
I14     ``test_mutualization::TestMutualizedTrainingSet::
        test_pooled_training_set_on_panel_f`` (stage M: 51 rows, gold values),
        ``test_invariants::TestOrderingSeesTheFittedSets::
        test_panel_scored_target_is_brought_back_to_the_stage_scale``, and
        stage Q: ``test_mutualization::TestMutualizedTrainingSet::
        test_pooled_training_set_at_stage_q`` (FR divisor 4, IT divisor 1/3 on
        its 36 monthly rows, DE raw).
I15     ``test_mutualization::TestMutualizedTrainingSet::
        test_single_fit_shared_between_source_frequency_groups``,
        ``test_contributing_entity_is_never_rewritten``,
        ``TestUnobservedEntities::
        test_adding_the_unanchored_group_leaves_the_others_intact``,
        ``test_transform::TestSharedModelReplay``.
I16     ``test_invariants::TestReferenceExamples::
        test_timeseries_results_unchanged_by_pooling`` (and
        ``training_set_builder/test_blocks_and_rows.py``, component level).
I17     ``test_invariants::TestOrderingSeesTheFittedSets`` (5 tests, TS and
        PANEL-F, ``training_scope='unrestricted'``, ``'frequency'`` order).
I18     (a) and (c): component level only,
        ``training_set_builder/test_origins_and_layers.py::TestTrainingWindow``
        (mask read once per layer, materializer called once per layer, layers
        ``[{FR: Y, DE: Q, IT: M}, {FR: Q}]``). (b) partial:
        ``test_frequency_progression::TestStagePlanAxis2::
        test_stage_plan_of_spec_5_5`` (Y and Q rows, origins) and
        ``TestOriginFilter::test_target_taint_families`` (``'imputed'``),
        ``TestStagePlanAxis2::test_x_train_is_full_on_the_quarterly_rows``
        (Y and Q rows, ``'imputed'``, ``m1`` = quarter sum at stage scale).
I19     ``test_frequency_progression::TestFrequencyProgression::
        test_progression_per_target_frequency_group_on_panel``,
        ``::test_entity_stopping_early_still_feeds_the_finer_stage`` (``DE``
        feeds the ``y_train`` of stage M, no ``DE`` cell imputed at M).
I20     ``test_frequency_progression::TestCoincidentCells`` (6 tests).
I21     ``test_mutualization::TestUnobservedEntities`` (11 tests).
======  =====================================================================
"""

# Modules de base
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Objets testés
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer
from tsforecast.frequency.training_set_builder import TrainingSetBuilder

# Outils de test partagés
from tests.unit.frequency.high_frequency_imputer.helpers import (
    AXIS_COMBINATIONS,
    by_entity,
    entity_block,
    filled_dates,
    fit_quietly,
    fit_transform_quietly,
    fit_with_spy,
    imputed_cells,
    make_imputer,
    replay,
    transform_quietly,
)


# =============================================================================
# Invariants statiques
# =============================================================================
class TestStaticInvariants:
    """Invariants vérifiables sur le source du module (§16)."""

    def test_no_boolean_test_on_impute_intermediate_frequencies(self):
        """Aucun test de vérité booléenne sur l'axe 2 : invariant I13."""
        module_path = (
            Path(__file__).resolve().parents[4]
            / 'tsforecast' / 'frequency' / 'high_frequency_imputer.py'
        )
        source = module_path.read_text(encoding='utf-8')

        # 'covariates_only' est truthy : un test de vérité serait un bug silencieux
        forbidden = [
            r'if\s+self\.impute_intermediate_frequencies\s*:',
            r'if\s+not\s+self\.impute_intermediate_frequencies\s*:',
        ]
        for pattern in forbidden:
            assert re.search(pattern, source) is None, pattern

        # La comparaison retenue est bien une comparaison d'identité/égalité
        assert 'self.impute_intermediate_frequencies is False' in source

    def test_removed_parameters_absent_from_source(self):
        """Les paramètres supprimés ne réapparaissent nulle part dans le module."""
        module_path = (
            Path(__file__).resolve().parents[4]
            / 'tsforecast' / 'frequency' / 'high_frequency_imputer.py'
        )
        source = module_path.read_text(encoding='utf-8')
        for removed in ('cascade_refitting', 'train_on_partial_coverage',
                        'train_on_partial_fit_order', 'enforce_period_totals',
                        'cv_n_splits', 'disaggregate_anchors'):
            assert removed not in source, removed

    def test_class_is_exported(self):
        """La classe est exportée sous son nom canonique, sans suffixe."""
        import tsforecast.frequency as frequency_module

        assert 'HighFrequencyImputer' in frequency_module.__all__
        assert 'HighFrequencyImputer2' not in frequency_module.__all__
        assert not hasattr(frequency_module, 'HighFrequencyImputer2')
        assert frequency_module.HighFrequencyImputer is HighFrequencyImputer


class TestNaNInvariant:
    """I2 — le motif de disponibilité de X_pred contient celui de X_train."""

    @pytest.mark.parametrize('strategy', ['tolerate_nan', 'interpolate', 'model'])
    @pytest.mark.parametrize(
        'fixture_name', ['reference_timeseries', 'mixed_freq_panel_heterogeneous']
    )
    @pytest.mark.slow
    def test_nan_invariant_by_stage_and_column(self, strategy, fixture_name, request):
        """Formulation D14 : inclusion des dates, par étape, colonne ET entité."""
        data = request.getfixturevalue(fixture_name)
        imputer = fit_with_spy(data, covariate_strategy=strategy)

        checked = 0
        for step in imputer.imputation_plan_:
            if step.is_fallback:
                continue
            model = step.model
            for column in step.feature_cols:
                trained = filled_dates(model.fit_X_, column)
                for prediction_frame in model.predict_X_:
                    grid = by_entity(prediction_frame.index)
                    predicted = filled_dates(prediction_frame, column)
                    for entity, dates in trained.items():
                        # Image des dates d'entraînement SUR la grille de
                        # prédiction : le taux brut de NaN ne dit rien quand
                        # les deux grilles n'ont pas le même pas (§4.7)
                        image = dates & grid.get(entity, set())
                        assert image <= predicted.get(entity, set())
                        checked += 1
        assert checked > 0


class TestOrderInvariance:
    """I3 et I10 — les valeurs ne dépendent ni de l'ordre des colonnes ni du traitement."""

    @staticmethod
    def _outputs(imputer: HighFrequencyImputer, data: pd.DataFrame, columns) -> dict:
        """Valeurs imputées et provenances, colonne par colonne, lues sur la sortie."""
        values, provenance = replay(imputer, data)
        return {
            column: (
                imputed_cells(values, provenance, column).sort_index().round(9),
                provenance[column].astype(str).sort_index(),
            )
            for column in columns
        }

    @pytest.mark.parametrize('strategy', ['tolerate_nan', 'interpolate', 'model'])
    def test_column_order_invariance(self, reference_timeseries, strategy):
        """I3 — permuter les colonnes d'entrée ne change ni valeurs ni provenances."""
        columns = ['q1', 'a1', 'a2']
        permuted_data = reference_timeseries[['a2', 'a1', 'q1', 'm1']]
        straight = fit_with_spy(reference_timeseries, covariate_strategy=strategy)
        permuted = fit_with_spy(permuted_data, covariate_strategy=strategy)
        straight_outputs = self._outputs(straight, reference_timeseries, columns)
        permuted_outputs = self._outputs(permuted, permuted_data, columns)

        for column in columns:
            values, provenance = straight_outputs[column]
            other_values, other_provenance = permuted_outputs[column]
            pd.testing.assert_series_equal(values, other_values)
            pd.testing.assert_series_equal(provenance, other_provenance)

    @pytest.mark.parametrize('strategy', ['tolerate_nan', 'interpolate'])
    @pytest.mark.slow
    def test_processing_order_indifferent_outside_model(
        self, reference_timeseries, strategy
    ):
        """I10 — hors 'model', deux ordres de traitement donnent la MÊME sortie."""
        columns = ['q1', 'a1', 'a2']
        forward = fit_with_spy(reference_timeseries, covariate_strategy=strategy)
        # Ordre de traitement inversé : il suit l'ordre des colonnes d'entrée
        backward_data = reference_timeseries[['a2', 'a1', 'q1', 'm1']]
        backward = fit_with_spy(backward_data, covariate_strategy=strategy)
        assert [step.var_name for step in forward.imputation_plan_] != [
            step.var_name for step in backward.imputation_plan_
        ]
        forward_outputs = self._outputs(forward, reference_timeseries, columns)
        backward_outputs = self._outputs(backward, backward_data, columns)
        for column in columns:
            pd.testing.assert_series_equal(
                forward_outputs[column][0], backward_outputs[column][0]
            )

    @pytest.mark.slow
    @pytest.mark.parametrize('strategy', ['tolerate_nan', 'interpolate', 'model'])
    def test_column_order_invariance_on_panel(self, mixed_freq_panel_heterogeneous, strategy):
        """I3 sur PANEL — permuter les colonnes ne change ni valeurs ni provenances."""
        data = mixed_freq_panel_heterogeneous
        columns = ['q1', 'a1', 'a2']
        permuted_data = data[['a2', 'climat_affaires', 'a1', 'q1', 'm1']]
        straight = fit_with_spy(data, covariate_strategy=strategy)
        permuted = fit_with_spy(permuted_data, covariate_strategy=strategy)
        straight_outputs = self._outputs(straight, data, columns)
        permuted_outputs = self._outputs(permuted, permuted_data, columns)

        for column in columns:
            values, provenance = straight_outputs[column]
            other_values, other_provenance = permuted_outputs[column]
            assert not values.empty
            pd.testing.assert_series_equal(values, other_values)
            pd.testing.assert_series_equal(provenance, other_provenance)

    @pytest.mark.slow
    @pytest.mark.parametrize('strategy', ['tolerate_nan', 'interpolate'])
    def test_processing_order_indifferent_outside_model_on_panel(
        self, mixed_freq_panel_heterogeneous, strategy
    ):
        """I10 sur PANEL — hors 'model', deux ordres de traitement, une même sortie."""
        data = mixed_freq_panel_heterogeneous
        columns = ['q1', 'a1', 'a2']
        forward = fit_with_spy(data, covariate_strategy=strategy)
        backward_data = data[['a2', 'a1', 'q1', 'climat_affaires', 'm1']]
        backward = fit_with_spy(backward_data, covariate_strategy=strategy)
        assert [step.var_name for step in forward.imputation_plan_] != [
            step.var_name for step in backward.imputation_plan_
        ]
        forward_outputs = self._outputs(forward, data, columns)
        backward_outputs = self._outputs(backward, backward_data, columns)
        for column in columns:
            pd.testing.assert_series_equal(
                forward_outputs[column][0], backward_outputs[column][0]
            )

    @pytest.mark.slow
    def test_imputation_order_empty_outside_model(self, reference_timeseries):
        """imputation_order_ reste vide hors covariate_strategy='model'."""
        assert not fit_with_spy(reference_timeseries).imputation_order_
        assert fit_with_spy(
            reference_timeseries, covariate_strategy='model'
        ).imputation_order_ == {'M': ['a1', 'a2', 'q1']}


class TestOrderingSeesTheFittedSets:
    """L'ordre 'cv' score les jeux que la 5c ajuste, pas une vue brute des données."""

    @staticmethod
    def _fit_recording_cv(data, scores, **overrides):
        """Ajuste sous l'ordre 'cv' en retenant le jeu scoré de chaque variable.

        Args:
            data: Jeu de données ajusté.
            scores: Score injecté par nom de variable.
            **overrides: Paramètres supplémentaires de l'imputeur.

        Returns:
            Couple ``(imputeur, recorded)``, ``recorded`` associant à chaque
            nom de variable le couple ``(X, y)`` réellement passé à la
            validation croisée.
        """
        recorded = {}

        def _spy(estimator, X, y, cv=None, scoring=None, error_score=None):
            del estimator, cv, scoring, error_score
            recorded[y.name] = (X.copy(), y.copy())
            return np.full(2, scores[y.name])

        params = dict(
            covariate_strategy='model',
            fit_predict_order='cv',
            cv=2,
            min_cv_train_size=2,
        )
        params.update(overrides)
        with patch(
            'tsforecast.frequency.variable_orderer.cross_val_score',
            side_effect=_spy,
        ):
            imputer = fit_with_spy(data, **params)
        return imputer, recorded

    def test_scored_set_is_the_fitted_set_of_the_first_variable(
        self, reference_timeseries
    ):
        """La variable classée 1re est ajustée sur EXACTEMENT le jeu qui l'a classée.

        Seule la 1re l'est : les suivantes voient en 5c ce que les rangs
        précédents ont écrit au miroir, ce que la 5b ne peut pas connaître.
        """
        imputer, recorded = self._fit_recording_cv(
            reference_timeseries, {'a1': -0.05, 'a2': -0.15, 'q1': -0.20},
        )

        first = imputer.imputation_order_['M'][0]
        assert first == 'a1'
        step = next(
            step for step in imputer.imputation_plan_ if step.var_name == first
        )
        X_scored, y_scored = recorded[first]

        # Mêmes covariables, mêmes lignes, mêmes valeurs que l'ajustement
        assert tuple(X_scored.columns) == tuple(step.feature_cols)
        pd.testing.assert_frame_equal(X_scored, step.model.fit_X_)
        pd.testing.assert_series_equal(y_scored, step.model.fit_y_)

    def test_scored_covariates_are_materialized_not_raw(self, reference_timeseries):
        """'m1' est scorée agrégée sur la grille annuelle, non lue telle quelle.

        La vue brute de "X_work" ne porterait, aux ancres de 'a1', que la
        valeur de décembre : c'est une covariable que le modèle ne verra
        jamais.
        """
        _imputer, recorded = self._fit_recording_cv(
            reference_timeseries, {'a1': -0.05, 'a2': -0.15, 'q1': -0.20},
        )
        X_scored, _y = recorded['a1']

        # Somme annuelle de 'm1' ramenée à l'échelle mensuelle de l'étape par
        # le diviseur de "StageScaler", et non la valeur de décembre
        annual = reference_timeseries['m1'].resample('YE').sum() / 12
        december = reference_timeseries['m1'].resample('YE').last()
        scored = X_scored['m1'].to_numpy(dtype=float)
        assert scored == pytest.approx(annual.to_numpy(dtype=float))
        assert scored != pytest.approx(december.to_numpy(dtype=float))

    @pytest.mark.slow
    def test_panel_scored_target_is_brought_back_to_the_stage_scale(
        self, mixed_freq_panel_multifrequency
    ):
        """Sur PANEL-F, la cible scorée est mutualisée ET remise à l'échelle de l'étape.

        La lecture brute de la colonne 'v' empilerait les trois entités à
        leurs échelles propres (annuelle pour FR, trimestrielle pour DE,
        mensuelle pour IT) : le score mesurerait la dispersion inter-entités.
        """
        _imputer, recorded = self._fit_recording_cv(
            mixed_freq_panel_multifrequency, {'v': -0.05, 'q1': -0.20},
        )
        _X_scored, y_scored = recorded['v']

        # Les 51 lignes du jeu mutualisé (§5.8), une seule échelle
        assert len(y_scored) == 51
        counts = pd.Series(
            [key[0] for key in y_scored.index]
        ).value_counts().to_dict()
        assert counts == {'IT': 36, 'DE': 12, 'FR': 3}
        assert y_scored.min() > 9.0 and y_scored.max() < 14.0

    @pytest.mark.slow
    def test_scoring_rows_follow_the_training_window_not_the_strict_one(
        self, reference_timeseries
    ):
        """Élargir "training_scope" élargit aussi les lignes scorées.

        Les lignes viennent du jeu mutualisé, restreint par la fenêtre
        'training' : le classement voit donc le même régime de valeurs
        manquantes que l'ajustement, quel que soit le scope.
        """
        scores = {'a1': -0.05, 'a2': -0.15, 'q1': -0.20}
        strict, recorded_strict = self._fit_recording_cv(
            reference_timeseries, scores,
        )
        wide, recorded_wide = self._fit_recording_cv(
            reference_timeseries, scores, training_scope='unrestricted',
        )

        # Sous les deux scopes, le jeu scoré de la 1re variable est celui de
        # son ajustement — la fenêtre stricte ne joue plus aucun rôle
        for imputer, recorded in ((strict, recorded_strict), (wide, recorded_wide)):
            first = imputer.imputation_order_['M'][0]
            step = next(
                step for step in imputer.imputation_plan_ if step.var_name == first
            )
            pd.testing.assert_series_equal(recorded[first][1], step.model.fit_y_)

        # Le scope élargi ne retire jamais de ligne au classement
        assert len(recorded_wide['q1'][1]) >= len(recorded_strict['q1'][1])

    def test_frequency_order_prepares_nothing(self, reference_timeseries):
        """Sous l'ordre 'frequency', aucune validation croisée n'est déclenchée."""
        calls = []
        with patch(
            'tsforecast.frequency.variable_orderer.cross_val_score',
            side_effect=lambda *a, **k: calls.append(1) or np.zeros(2),
        ):
            imputer = fit_with_spy(
                reference_timeseries, covariate_strategy='model',
            )
        assert not calls
        assert imputer.imputation_order_ == {'M': ['a1', 'a2', 'q1']}


class TestAggregationAdditivity:
    """I4 — chaque période complète somme au total observé."""

    def test_period_totals_additivity(self, reference_timeseries):
        """Sous 'sum', les 12 mois d'une année somment à son total annuel."""
        imputer = fit_with_spy(reference_timeseries)
        values, _provenance = replay(imputer, reference_timeseries)
        for year, total in [('2021', 120.0), ('2022', 132.0), ('2023', 150.0)]:
            assert values['a1'].loc[year].sum() == pytest.approx(total)

    def test_anchor_row_no_longer_carries_the_total(self, reference_timeseries):
        """La ligne d'ancre porte une valeur de sous-période, jamais le total (§11.2)."""
        imputer = fit_with_spy(reference_timeseries)
        # La sortie pose l'imputation par-dessus l'ancre brute (D37)
        values, _provenance = replay(imputer, reference_timeseries)
        anchor = float(values['a1']['2021-12-31'])
        assert anchor != pytest.approx(120.0)
        assert 0.0 < anchor < 120.0

    def test_group_totals_are_those_of_the_source_frequency(
        self, mixed_freq_panel_multifrequency
    ):
        """Les totaux recalés sont annuels pour FR et trimestriels pour DE (§5.8)."""
        imputer = fit_with_spy(mixed_freq_panel_multifrequency)
        values, _provenance = replay(imputer, mixed_freq_panel_multifrequency)
        store = values['v']

        france = entity_block(store, 'FR')
        assert france.loc['2021'].sum() == pytest.approx(120.0)
        # Le trimestre de FR est libre : seul son total ANNUEL est contraint
        assert france.loc['2021-01':'2021-03'].sum() != pytest.approx(28.0)

        germany = entity_block(store, 'DE')
        assert germany.loc['2021-01':'2021-03'].sum() == pytest.approx(28.0)
        assert germany.loc['2021-04':'2021-06'].sum() == pytest.approx(30.0)

    def test_carried_covariate_sums_to_the_period_total(self, reference_timeseries):
        """I4 — une covariable PORTÉE d'une étape antérieure somme au total observé.

        Sous 'model' et 'covariates_only', ``a2`` imputée à l'étape Q est
        reportée sur la grille mensuelle de ``a1`` (voie ``'carried_model'``,
        rang 3) : ses douze mois somment au total annuel observé.
        """
        imputer = fit_with_spy(
            reference_timeseries,
            covariate_strategy='model',
            impute_intermediate_frequencies='covariates_only',
            scale_features=False,
        )
        step = next(
            step for step in imputer.imputation_plan_
            if (step.pred_freq_label, step.var_name) == ('M', 'a1')
        )
        assert step.materialization['a2'] == 'carried_model'

        carried = step.model.predict_X_[0]['a2']
        assert carried.notna().all()
        totals = carried.groupby(carried.index.year).sum()
        assert totals.to_dict() == pytest.approx({2021: 60.0, 2022: 66.0, 2023: 72.0})


class TestTrainPredictScale:
    """I5 — les features de X_train et de X_pred sont à la même échelle."""

    @staticmethod
    def _means(step) -> list:
        """Moyennes appariées (entraînement, prédiction) de chaque feature."""
        model = step.model
        pairs = []
        for column in step.feature_cols:
            trained = float(model.fit_X_[column].mean())
            predicted = float(
                np.mean([frame[column].mean() for frame in model.predict_X_])
            )
            pairs.append((column, trained, predicted))
        return pairs

    @pytest.mark.parametrize('strategy', ['interpolate', 'model'])
    def test_train_test_feature_means_comparable(self, reference_timeseries, strategy):
        """Toute covariable matérialisée porte la MÊME échelle des deux côtés."""
        imputer = fit_with_spy(reference_timeseries, covariate_strategy=strategy)
        for step in imputer.imputation_plan_:
            for column, trained, predicted in self._means(step):
                assert trained == pytest.approx(predicted, rel=1e-6), (
                    f'{step.var_name}/{column}'
                )

    def test_mixed_calendar_feature_and_constant_target(self, reference_timeseries):
        """Cas mixte : feature en 'calendar', cible en 'constant' (D15)."""
        imputer = fit_with_spy(
            reference_timeseries,
            scale_features={'m1': 'calendar', '__default__': 'constant'},
        )
        for step in imputer.imputation_plan_:
            for column, trained, predicted in self._means(step):
                assert trained == pytest.approx(predicted, rel=0.05), (
                    f'{step.var_name}/{column}'
                )

    def test_features_divided_target_untouched(self, reference_timeseries):
        """Cas « features divisées, y en False » : la cible garde son échelle brute."""
        imputer = fit_with_spy(
            reference_timeseries,
            scale_features={'a1': False, '__default__': 'constant'},
        )
        step = next(
            step for step in imputer.imputation_plan_ if step.var_name == 'a1'
        )
        # Cible non divisée : les trois ancres annuelles telles quelles
        assert sorted(step.model.fit_y_.tolist()) == [120.0, 132.0, 150.0]


class TestReferenceExamples:
    """Les exemples chiffrés du document, repris comme cas d'or."""

    def test_reference_plan_of_spec_5_5_under_false(self, reference_timeseries):
        """§5.5 « Sous False » : une étape M, trois modèles, dans l'ordre a1, a2, q1."""
        imputer = fit_with_spy(reference_timeseries, covariate_strategy='model')
        plan = list(imputer.imputation_plan_)

        assert [step.var_name for step in plan] == ['a1', 'a2', 'q1']
        assert {step.pred_freq_label for step in plan} == {'M'}
        assert len(imputer.imputation_models_) == 3

        # Nombre de lignes de y_train : 3 ancres annuelles, 12 trimestrielles
        rows = {step.var_name: len(step.model.fit_y_) for step in plan}
        assert rows == {'a1': 3, 'a2': 3, 'q1': 12}

        ways = {step.var_name: dict(step.materialization) for step in plan}
        # 'a1' d'abord : rien n'est encore imputé, ses deux covariables basses
        # passent par le repli
        assert ways['a1'] == {
            'm1': 'identity', 'q1': 'interpolate', 'a2': 'interpolate'
        }
        # 'a2' ensuite : seule 'a1' est imputée, donc seule 'a1' est lue dans
        # le miroir — les registres ne portent que ce qui a été IMPUTÉ (D24),
        # jamais la matérialisation d'une covariable
        assert ways['a2'] == {
            'm1': 'identity', 'q1': 'interpolate', 'a1': 'stage_model'
        }
        # 'q1' enfin : les deux annuelles sont imputées
        assert ways['q1'] == {
            'm1': 'identity', 'a1': 'stage_model', 'a2': 'stage_model'
        }

    def test_reference_provenance_of_spec_6_5(self, reference_timeseries):
        """§6.5 : MODEL_ON_INTERPOLATED sur les douze mois, ancre comprise, somme 120."""
        imputer = fit_with_spy(reference_timeseries)
        provenance = imputer.imputation_provenance_['a1'].loc['2021']
        assert len(provenance) == 12
        assert {str(value) for value in provenance} == {'model_on_interpolated'}
        # L'ancre ne reste NI ORIGINAL NI DISAGGREGATED
        assert str(imputer.imputation_provenance_['a1']['2021-12-31']) == (
            'model_on_interpolated'
        )
        values, _provenance = replay(imputer, reference_timeseries)
        assert values['a1'].loc['2021'].sum() == pytest.approx(120.0)

    def test_provenance_unchanged_without_aggregation_constraint(
        self, reference_timeseries
    ):
        """Sous aggregation_constraint=None, seules les VALEURS changent (D16)."""
        constrained = fit_with_spy(reference_timeseries)
        free = fit_with_spy(reference_timeseries, aggregation_constraint=None)

        pd.testing.assert_series_equal(
            constrained.imputation_provenance_['a1'].astype(str),
            free.imputation_provenance_['a1'].astype(str),
        )
        free_values, _provenance = replay(free, reference_timeseries)
        free_sum = free_values['a1'].loc['2021'].sum()
        assert free_sum != pytest.approx(120.0)

    def test_timeseries_results_unchanged_by_pooling(self, reference_timeseries):
        """I16 — à une entité, le jeu mutualisé est le jeu d'origine."""
        imputer = fit_with_spy(reference_timeseries)
        for step in imputer.imputation_plan_:
            assert dict(step.training_blocks) == {
                (): 'Q' if step.var_name == 'q1' else 'Y'
            }
            assert step.entities is None
        rows = {step.var_name: len(step.model.fit_y_) for step in imputer.imputation_plan_}
        assert rows == {'a1': 3, 'a2': 3, 'q1': 12}
        values, _provenance = replay(imputer, reference_timeseries)
        for year, total in [('2021', 120.0), ('2022', 132.0), ('2023', 150.0)]:
            assert values['a1'].loc[year].sum() == pytest.approx(total)


class TestTransformSymmetry:
    """I1 — `fit_transform(X)` est strictement `fit(X).transform(X)`."""

    @pytest.mark.parametrize('strategy, intermediate', AXIS_COMBINATIONS)
    @pytest.mark.parametrize(
        'fixture_name', ['reference_timeseries', 'mixed_freq_panel_multifrequency']
    )
    @pytest.mark.slow
    def test_fit_transform_equals_fit_then_transform(
        self, strategy, intermediate, fixture_name, request
    ):
        """I1 — valeurs, provenances et attributs de sortie strictement égaux."""
        data = request.getfixturevalue(fixture_name)
        params = dict(
            covariate_strategy=strategy,
            impute_intermediate_frequencies=intermediate,
        )

        # Les deux chemins, sur les mêmes données
        combined = fit_transform_quietly(make_imputer(**params), data)
        separate = make_imputer(**params)
        fit_quietly(separate, data)
        replayed = transform_quietly(separate, data)

        # Égalité STRICTE des valeurs
        pd.testing.assert_frame_equal(combined, replayed)

    @pytest.mark.parametrize('strategy, intermediate', AXIS_COMBINATIONS)
    def test_output_attributes_are_identical_too(
        self, strategy, intermediate, reference_timeseries
    ):
        """I1 — la provenance et les attributs ajustés coïncident aussi."""
        params = dict(
            covariate_strategy=strategy,
            impute_intermediate_frequencies=intermediate,
        )
        first = make_imputer(**params)
        fit_transform_quietly(first, reference_timeseries)

        second = make_imputer(**params)
        fit_quietly(second, reference_timeseries)
        transform_quietly(second, reference_timeseries)

        # Provenance du dernier transform
        pd.testing.assert_frame_equal(
            first.imputation_provenance_, second.imputation_provenance_
        )
        # Attributs de sortie du fit
        assert first.frequency_progression_ == second.frequency_progression_
        assert first.detected_frequencies_ == second.detected_frequencies_
        assert first.unanchored_pairs_ == second.unanchored_pairs_
        pd.testing.assert_frame_equal(
            first.imputation_plan_.to_diagnostic_frame(),
            second.imputation_plan_.to_diagnostic_frame(),
        )

    def test_transform_never_rebuilds_the_training_set(self, reference_timeseries):
        """Le TrainingSetBuilder n'est pas sur le chemin du transform (§12.1)."""
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        # Patch de la classe publique : toute instance, quel que soit son
        # chemin d'accès dans l'imputeur, lève si elle est appelée
        with patch.object(
            TrainingSetBuilder, 'build',
            side_effect=AssertionError('the training set must never be rebuilt'),
        ):
            transform_quietly(imputer, reference_timeseries)
