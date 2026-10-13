"""Phases 0 to 5 of ``HighFrequencyImputer.fit``.

Covers the fitted attributes of §13.2, the input contract (B14, B15, B16,
B19), the per-(entity, column) frequencies and classification (§2.5), the
wiring of the §12.2 components, the interpolation fallback (D6) and the
edge cases of step execution.
"""

# Modules de base
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Objets testés
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer
from tsforecast.frequency.imputation_plan import INTERPOLATE_FALLBACK
from tsforecast.frequency.training_set_builder import TrainingSetBuilder

# Outils de test partagés
from tests.support.estimators import (
    FailingEstimator,
    SpyEstimator,
)
from tests.unit.frequency.high_frequency_imputer.helpers import (
    FR,
    DE,
    IT,
    fit_quietly,
    fit_with_spy,
    imputed_cells,
    make_imputer,
    plan_groups,
    replay,
    stage_label,
    transform_quietly,
)


# =============================================================================
# Phases 0 à 4 du fit (§12.3)
# =============================================================================
class TestFitPhases:
    """Les phases 0 à 4 renseignent les attributs ajustés du §13.2."""

    @pytest.mark.parametrize('fixture_name', ['reference_timeseries', 'mixed_freq_panel_heterogeneous'])
    @pytest.mark.slow
    def test_phases_zero_to_four_populate_attributes(self, fixture_name, request):
        """Les attributs des phases 0 à 4 sont renseignés, sur TS comme sur panel."""
        data = request.getfixturevalue(fixture_name)
        imputer = fit_quietly(make_imputer(), data)
        is_panel = fixture_name != 'reference_timeseries'

        # PHASE 0 : contrat d'entrée, fréquences, classification
        assert imputer.feature_columns_ == list(data.columns)
        assert imputer.is_panel_ is is_panel
        assert imputer.target_column_ is None
        assert bool(imputer.detected_frequencies_)
        assert set(imputer.variable_categories_) == {'aggregate', 'impute', 'target_freq'}
        # 'a1' est annuelle et la cible mensuelle : elle est imputable
        imputable_columns = {
            key[-1] if isinstance(key, tuple) else key
            for key in imputer.variable_categories_['impute']
        }
        assert 'a1' in imputable_columns
        # Ordre d'imputation VIDE hors covariate_strategy='model' (§13.2)
        assert not imputer.imputation_order_

        # PHASE 1 : les trois masques et les deux bornes
        for mask_name in ('strict_window_mask_', 'imputation_window_mask_',
                          'training_window_mask_'):
            assert isinstance(getattr(imputer, mask_name), pd.Series)
        assert imputer.imputation_window_ is not None
        assert imputer.training_window_ is not None

        # PHASE 2 : aucun transformateur additif fourni
        assert imputer.additive_transformer_ is None

        # PHASE 3 : progression réduite à la cible sous False
        assert len(imputer.frequency_progression_) == 1

        # PHASE 4 : matrice de provenance initialisée
        assert isinstance(imputer.imputation_provenance_, pd.DataFrame)

        # PHASE 5 : une etape de plan par groupe imputable, et un modele par
        # (etape, variable) dans le registre
        assert len(imputer.imputation_plan_) >= 1
        assert set(imputer.imputation_models_)
        assert all(step.pred_freq_label == 'M' for step in imputer.imputation_plan_)

    @pytest.mark.slow
    def test_entities_are_set_on_panel_only(self, reference_timeseries,
                                            mixed_freq_panel_multifrequency):
        """entities_ vaut None sur une série temporelle et liste les entités sur un panel."""
        assert fit_quietly(make_imputer(), reference_timeseries).entities_ is None
        panel_imputer = fit_quietly(make_imputer(), mixed_freq_panel_multifrequency)
        assert set(panel_imputer.entities_) == {FR, DE, IT}

    def test_target_is_named_and_merged(self, reference_timeseries):
        """y est fusionné dans le frame de travail sous un nom unique (B14)."""
        features = reference_timeseries.drop(columns=['a1'])
        target = reference_timeseries['a1']
        imputer = fit_quietly(make_imputer(), features)
        assert imputer.target_column_ is None

        imputer = fit_quietly(make_imputer(), features)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(features, target)
        assert imputer.target_column_ == 'a1'
        assert 'a1' in imputer.detected_frequencies_

    def test_target_frequency_dict_incomplete_raises_naming_entities(
        self, mixed_freq_panel_multifrequency
    ):
        """Un dict de fréquences cibles incomplet nomme les entités manquantes (B16)."""
        imputer = make_imputer(target_frequency={'FR': 'M', 'DE': 'M'})
        with pytest.raises(ValueError) as excinfo:
            fit_quietly(imputer, mixed_freq_panel_multifrequency)
        message = str(excinfo.value)
        assert 'IT' in message
        assert 'incomplete' in message

    def test_target_frequency_dict_complete_is_accepted(self, mixed_freq_panel_multifrequency):
        """Un dict nommant toutes les entités passe et est normalisé en clés tuples."""
        imputer = fit_quietly(
            make_imputer(target_frequency={'FR': 'M', 'DE': 'M', 'IT': 'M'}),
            mixed_freq_panel_multifrequency,
        )
        assert set(imputer.effective_target_frequency_) == {FR, DE, IT}

    def test_y_index_equality_checked(self, reference_timeseries):
        """Des index de même longueur mais de libellés différents lèvent (B14)."""
        features = reference_timeseries.drop(columns=['a1'])
        target = reference_timeseries['a1'].copy()
        # Décalage d'un mois : même longueur, index différent
        target.index = target.index + pd.DateOffset(months=1)
        assert len(features) == len(target)

        with pytest.raises(ValueError) as excinfo:
            fit_quietly(make_imputer(), features).fit(features, target)
        message = str(excinfo.value)
        assert 'different indices' in message
        assert 'same length' in message

    @pytest.mark.slow
    def test_three_window_masks_are_set(self, mixed_freq_panel_heterogeneous):
        """Les trois masques sont des Series booléennes à MultiIndex sur panel (§7.2)."""
        imputer = fit_quietly(make_imputer(), mixed_freq_panel_heterogeneous)
        for mask_name in ('strict_window_mask_', 'imputation_window_mask_',
                          'training_window_mask_'):
            mask = getattr(imputer, mask_name)
            assert isinstance(mask, pd.Series), mask_name
            assert mask.dtype == bool, mask_name
            assert isinstance(mask.index, pd.MultiIndex), mask_name

    def test_widening_training_scope_adds_rows_not_columns(self, reference_timeseries):
        """Élargir training_scope change le masque d'entraînement, jamais les colonnes (§7.2)."""
        strict = fit_quietly(
            make_imputer(training_scope='strict'), reference_timeseries
        )
        widened = fit_quietly(
            make_imputer(training_scope='unrestricted'), reference_timeseries
        )
        # Des lignes en plus à l'entraînement
        assert widened.training_window_mask_.sum() >= strict.training_window_mask_.sum()
        # Aucune colonne gagnée ni perdue
        assert widened.feature_columns_ == strict.feature_columns_
        # Le masque de PRÉDICTION reste inchangé : le scope d'entraînement ne
        # gouverne pas la fenêtre de prédiction
        pd.testing.assert_series_equal(
            widened.imputation_window_mask_, strict.imputation_window_mask_
        )

    def test_cv_attribute_only_under_cv_order(self, reference_timeseries):
        """cv_ n'existe que sous fit_predict_order='cv' (§13.2)."""
        by_frequency = fit_quietly(make_imputer(), reference_timeseries)
        assert not hasattr(by_frequency, 'cv_')

        by_cv = fit_quietly(
            make_imputer(fit_predict_order='cv', cv=3), reference_timeseries
        )
        assert hasattr(by_cv, 'cv_')
        assert by_cv.cv_.get_n_splits() == 3

    def test_frequency_progression_is_target_only_under_false(self, reference_timeseries):
        """Sous impute_intermediate_frequencies=False, la progression est [f_target]."""
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        assert imputer.frequency_progression_ == ['M']

    def test_estimator_none_warns_once(self, reference_timeseries):
        """L'absence d'estimateur donne UN avertissement, pas un par variable."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            HighFrequencyImputer(target_frequency='M', estimator=None).fit(
                reference_timeseries
            )
        estimator_warnings = [
            w for w in caught if 'estimator=None' in str(w.message)
        ]
        assert len(estimator_warnings) == 1

    @pytest.mark.slow
    def test_intermediate_frequencies_modalities_all_fit(self, reference_timeseries):
        """Les trois modalités de l'axe 2 ajustent, aucune ne lève (§5.1)."""
        for modality in (False, 'covariates_only', True):
            imputer = make_imputer(impute_intermediate_frequencies=modality)
            fit_quietly(imputer, reference_timeseries)
            assert imputer.frequency_progression_[-1] == 'M'

    def test_panel_declared_by_panel_cols_on_flat_frame(self, mixed_freq_panel_multifrequency):
        """Un panel déclaré par panel_cols sur frame plat est pleinement fonctionnel (B15)."""
        flat = mixed_freq_panel_multifrequency.reset_index()
        imputer = fit_quietly(
            make_imputer(time_col='date', panel_cols=['country']), flat
        )
        assert imputer.is_panel_ is True
        assert set(imputer.entities_) == {FR, DE, IT}

    def test_fit_purges_stale_transform_state(self, reference_timeseries):
        """Un état laissé par un transform précédent est purgé en tête de fit (B19).

        Le transform d'un autre jeu laisse son instantané et sa matrice empilée ;
        le fit qui suit les remplace par les siens : l'inversion immédiate lit
        l'index à plat du fit et restitue ses valeurs observées.
        """
        other = reference_timeseries.iloc[:24]
        imputer = fit_quietly(
            make_imputer(restore_original_values=True), reference_timeseries
        )
        transform_quietly(imputer, other)

        fit_quietly(imputer, reference_timeseries)
        restored = imputer.inverse_transform(reference_timeseries)
        assert restored.index.equals(reference_timeseries.index)
        observed = reference_timeseries.notna()
        pd.testing.assert_frame_equal(
            restored.where(observed), reference_timeseries.where(observed),
            check_freq=False,
        )

    @pytest.mark.slow
    def test_entity_never_observing_a_column_is_left_out(
        self, mixed_freq_panel_heterogeneous
    ):
        """Un couple (entité, colonne) jamais observé sort de la classification (§4.5)."""
        imputer = fit_quietly(make_imputer(), mixed_freq_panel_heterogeneous)
        # IT n'observe jamais climat_affaires : le couple n'a pas de fréquence
        assert (IT + ('climat_affaires',)) not in imputer.detected_frequencies_
        # ... alors que l'entité et la colonne existent bien dans les données :
        # c'est un couple non détecté, et non un couple absent
        assert IT in imputer.entities_
        assert 'climat_affaires' in imputer.feature_columns_
        # Les deux autres entités l'observent bien
        assert imputer.detected_frequencies_[('FR', 'climat_affaires')] == 'M'
        # Le couple n'apparaît dans aucune catégorie
        for category in imputer.variable_categories_.values():
            assert ('IT', 'climat_affaires') not in category


# =============================================================================
# Fréquences et classification PAR (entité, colonne) — jeu PANEL-F (§2.5)
# =============================================================================
class TestPerEntityFrequencies:
    """Sur un panel, une même colonne peut porter une fréquence par entité."""

    def test_detected_frequencies_are_per_entity(self, mixed_freq_panel_multifrequency):
        """detected_frequencies_ rend Y pour FR, Q pour DE et M pour IT sur la colonne v."""
        imputer = fit_quietly(make_imputer(), mixed_freq_panel_multifrequency)
        assert imputer.detected_frequencies_[('FR', 'v')] == 'Y'
        assert imputer.detected_frequencies_[('DE', 'v')] == 'Q'
        assert imputer.detected_frequencies_[('IT', 'v')] == 'M'
        # Les colonnes homogènes gardent la même fréquence partout
        for entity in ('FR', 'DE', 'IT'):
            assert imputer.detected_frequencies_[(entity, 'q1')] == 'Q'
            assert imputer.detected_frequencies_[(entity, 'm1')] == 'M'

    def test_variable_classification_is_per_entity_pair(self, mixed_freq_panel_multifrequency):
        """v est imputable pour FR et DE, et ne l'est pas pour IT, déjà à la cible."""
        imputer = fit_quietly(make_imputer(), mixed_freq_panel_multifrequency)
        imputable = set(imputer.variable_categories_['impute'])
        at_target = set(imputer.variable_categories_['target_freq'])

        assert ('FR', 'v') in imputable
        assert ('DE', 'v') in imputable
        assert ('IT', 'v') not in imputable
        assert ('IT', 'v') in at_target

    def test_imputable_pairs_group_by_source_frequency(self, mixed_freq_panel_multifrequency):
        """Les couples imputables se regroupent en (v, Y) -> {FR} et (v, Q) -> {DE}."""
        imputer = fit_quietly(make_imputer(), mixed_freq_panel_multifrequency)
        # Groupes lus sur le plan figé : une étape par groupe de fréquence source
        groups = plan_groups(imputer, 'M')

        assert groups[('v', 'Y')] == (FR,)
        assert groups[('v', 'Q')] == (DE,)
        # IT n'est imputable dans aucun groupe de v
        for group_key, entities in groups.items():
            if group_key[0] == 'v':
                assert IT not in entities

    # Interne : adaptateur privé ; son pendant public est
    # test_detected_frequencies_are_per_entity
    @pytest.mark.internal
    def test_detected_frequencies_adapter_keeps_both_shapes(
        self, mixed_freq_panel_multifrequency
    ):
        """L'adaptateur rend la forme par colonne qu'attendent les composants."""
        imputer = fit_quietly(make_imputer(), mixed_freq_panel_multifrequency)
        by_column = imputer._detected_frequencies_by_column()

        # Colonne hétérogène : forme par entité
        assert by_column['v'] == {FR: 'Y', DE: 'Q', IT: 'M'}
        # Colonnes homogènes : repli sur la forme scalaire
        assert by_column['q1'] == 'Q'
        assert by_column['m1'] == 'M'

    def test_hyperparameter_dicts_stay_keyed_by_column(self, mixed_freq_panel_multifrequency):
        """Les dicts d'hyperparamètres restent indexés par colonne, jamais par couple (D10)."""
        imputer = fit_quietly(
            make_imputer(
                scale_features={'v': 'calendar'},
                interpolation_method={'v': 'linear'},
                interpolation_anchor={'v': 1.0},
                aggregation_constraint={'v': None},
                estimator={'v': LinearRegression(), '__default__': LinearRegression()},
            ),
            mixed_freq_panel_multifrequency,
        )
        # Les clés restent des noms de colonnes nus après le fit
        for param in ('scale_features', 'interpolation_method',
                      'interpolation_anchor', 'aggregation_constraint', 'estimator'):
            assert set(getattr(imputer, param)) <= {'v', '__default__'}


# =============================================================================
# Câblage des composants
# =============================================================================
# Interne : tests de câblage par nature. Pendants publics :
# test_scoring_rows_follow_the_training_window_not_the_strict_one (I17),
# test_per_column_constraint_is_read_per_column,
# test_widening_training_scope_adds_rows_not_columns
@pytest.mark.internal
class TestWiring:
    """Les composants du §12.2 sont instanciés et câblés au fit."""

    def test_training_set_builder_is_wired(self, mixed_freq_panel_multifrequency, monkeypatch):
        """Le callable de masque du builder appelle get_mask_at_frequency(kind='training')."""
        imputer = fit_quietly(make_imputer(), mixed_freq_panel_multifrequency)

        # Le composant est instancié et porte le matérialiseur de l'instance
        assert imputer._training_set_builder is not None
        assert imputer._training_set_builder.materializer is imputer._covariate_materializer

        # Espionnage de l'appel effectué par le callable injecté
        recorded = {}
        original = imputer._imputation_window_calc.get_mask_at_frequency

        def _spy(frequency, kind='imputation'):
            recorded['frequency'] = frequency
            recorded['kind'] = kind
            return original(frequency, kind=kind)

        monkeypatch.setattr(
            imputer._imputation_window_calc, 'get_mask_at_frequency', _spy
        )

        # Blocs de la colonne v sur PANEL-F, exactement la forme que le
        # TrainingSetBuilder passe : IT y est à la fréquence de la grille, cas
        # que le calculateur traite en identité (§5.8 R2)
        blocks = {FR: 'Y', DE: 'Q', IT: 'M'}
        mask = imputer._training_set_builder.training_mask(blocks)

        # Le "kind" est nommé explicitement par le câblage, jamais laissé au défaut
        assert recorded['kind'] == 'training'
        assert recorded['frequency'] == blocks
        # Le masque revient exploitable pour les trois entités
        assert isinstance(mask, pd.Series)
        assert set(mask.index.get_level_values(0).unique()) == {'FR', 'DE', 'IT'}

    def test_covariate_materializer_carries_the_aggregation_constraint(
        self, reference_timeseries
    ):
        """La contrainte d'agrégation est portée par le matérialiseur, pas par le builder."""
        constraint = {'a1': None, '__default__': 'sum'}
        imputer = fit_quietly(
            make_imputer(aggregation_constraint=constraint), reference_timeseries
        )
        assert imputer._covariate_materializer.aggregation_constraint is constraint
        # Le builder n'en porte aucune : la cible n'est jamais agrégée (§5.8 R3)
        assert not hasattr(imputer._training_set_builder, 'aggregation_constraint')

    def test_window_calculator_receives_the_four_window_parameters(self, reference_timeseries):
        """Le calculateur de fenêtre reçoit les quatre paramètres du §7."""
        imputer = fit_quietly(
            make_imputer(
                imputation_scope='extended_forward',
                coverage_threshold=0.75,
                training_scope='strict',
                training_coverage_threshold=0.25,
            ),
            reference_timeseries,
        )
        calculator = imputer._imputation_window_calc
        assert calculator.imputation_scope == 'extended_forward'
        assert calculator.coverage_threshold == 0.75
        assert calculator.training_scope == 'strict'
        assert calculator.training_coverage_threshold == 0.25


class TestFallbackPath:
    """Le repli d'interpolation, et ce qu'il écrit."""

    def test_estimator_failure_falls_back_and_marks_interpolated(
        self, reference_timeseries
    ):
        """D6 — is_fallback=True, cellules INTERPOLATED, registres alimentés."""
        imputer = make_imputer(estimator=FailingEstimator())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            imputer.fit(reference_timeseries)
            messages = [str(item.message) for item in caught]

        assert all(step.is_fallback for step in imputer.imputation_plan_)
        assert all(step.feature_cols == () for step in imputer.imputation_plan_)
        for column in ('q1', 'a1', 'a2'):
            assert {
                str(value) for value in imputer.imputation_provenance_[column]
            } == {'interpolated'}
        # « Le repli matérialise » : chaque cellule repliée est écrite dans la
        # sortie (le registre d'origine, lui, est mesuré par
        # test_fallback_feeds_the_origin_register)
        values, provenance = replay(imputer, reference_timeseries)
        for column in ('q1', 'a1', 'a2'):
            written = imputed_cells(values, provenance, column)
            assert not written.empty
            assert written.notna().all()

        # Avertissements AGRÉGÉS : un seul message porte les trois échecs
        degraded = [message for message in messages if 'degraded during the fit' in message]
        assert len(degraded) == 1
        assert degraded[0].count('interpolation fallback') == 3

    def test_estimator_none_interpolates_everything(self, reference_timeseries):
        """estimator=None : chaque variable retombe sur l'interpolation."""
        imputer = fit_quietly(make_imputer(estimator=None), reference_timeseries)
        assert all(step.is_fallback for step in imputer.imputation_plan_)
        assert all(
            step.model == INTERPOLATE_FALLBACK for step in imputer.imputation_plan_
        )
        # Les totaux de période restent respectés : le repli est recalé
        values, _provenance = replay(imputer, reference_timeseries)
        assert values['a1'].loc['2022'].sum() == pytest.approx(132.0)

    def test_fallback_feeds_the_origin_register(self, reference_timeseries):
        """« Le repli matérialise » : une étape repliée nourrit l'étape suivante.

        Sous ``impute_intermediate_frequencies=True``, les cellules que le repli
        écrit à l'étape Q entrent dans le ``y_train`` de l'étape M avec
        l'origine ``'interpolated'`` : c'est l'effet public du registre
        d'origine alimenté par le repli (§5.3, §12.3 5c).
        """
        origins = {}
        original = TrainingSetBuilder.build

        # Espion sur la méthode publique du constructeur de jeu : le dernier
        # appel d'une (étape, colonne) est le jeu complet de la 5c
        def _spy(builder, **kwargs):
            training = original(builder, **kwargs)
            origins[(stage_label(kwargs['stage_freq']), kwargs['column'])] = set(
                training.row_origin
            )
            return training

        with patch.object(TrainingSetBuilder, 'build', _spy):
            fit_quietly(
                make_imputer(
                    estimator=FailingEstimator(), impute_intermediate_frequencies=True
                ),
                reference_timeseries,
            )

        # Étape Q : rien n'est encore imputé, seules les ancres observées
        assert origins[('Q', 'a1')] == {'observed'}
        # Étape M : les cellules du repli trimestriel rejoignent les ancres
        assert origins[('M', 'a1')] == {'observed', 'interpolated'}


class TestPhaseFiveEdgeCases:
    """Cas limites de l'exécution des étapes."""

    def test_unsorted_index_is_refused_explicitly(self, reference_timeseries):
        """Un index désordonné est refusé par le contrat d'entrée, sans imputer à faux.

        La classe fige ``auto_sort=False`` et ``strict_validation=True``
        ([SPEC] §12.5) : trier en silence changerait les fenêtres et les
        agrégats sans que l'appelant le sache.
        """
        with pytest.raises(ValueError, match='not sorted'):
            fit_with_spy(reference_timeseries.iloc[::-1])

    def test_duplicated_index_does_not_crash(self, reference_timeseries):
        """Un index dupliqué n'interrompt pas le fit."""
        duplicated = pd.concat([reference_timeseries, reference_timeseries.iloc[[0]]])
        imputer = make_imputer(estimator=SpyEstimator())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                imputer.fit(duplicated)
            except ValueError as error:
                # Un refus explicite est un comportement acceptable
                assert 'duplicat' in str(error).lower() or 'unique' in str(error).lower()
                return
        assert len(imputer.imputation_plan_) >= 1

    def test_yearly_variable_with_two_anchors_only(self, reference_timeseries):
        """Deux ancres seulement : y_train de taille 2, sans repli forcé."""
        data = reference_timeseries.copy()
        data.loc['2023-12-31', 'a1'] = np.nan
        imputer = fit_with_spy(data)
        step = next(step for step in imputer.imputation_plan_ if step.var_name == 'a1')
        assert len(step.model.fit_y_) == 2

    def test_all_nan_column_is_not_imputed(self, reference_timeseries):
        """Une colonne entièrement NaN n'est classée nulle part et n'est jamais imputée."""
        data = reference_timeseries.copy()
        data['empty'] = np.nan
        imputer = fit_with_spy(data)
        assert all(step.var_name != 'empty' for step in imputer.imputation_plan_)
        values, provenance = replay(imputer, data)
        assert values['empty'].isna().all()
        assert imputed_cells(values, provenance, 'empty').empty

    def test_incomplete_leading_and_trailing_periods(self, reference_timeseries):
        """Une période que les bornes du jeu tronquent n'est pas recalée (§11.1).

        Deux gardes distinctes se relaient : celle des sous-périodes NaN, et
        celle de la troncature CALENDAIRE, dont les sous-périodes manquantes
        sont absentes de la grille plutôt que vides. Sans la seconde, le total
        annuel de 2021 serait réparti sur les dix mois présents, soit une
        sur-attribution de 20 % à chacun.
        """
        truncated = reference_timeseries.loc['2021-03-31':'2023-08-31']
        imputer = fit_with_spy(truncated)
        values, provenance = replay(imputer, truncated)
        imputed = imputed_cells(values, provenance, 'a1')

        # L'année 2022, complète, reste exacte
        assert imputed.loc['2022'].sum() == pytest.approx(132.0)
        # 2021, amputée de deux mois, garde ses prédictions brutes
        assert len(imputed.loc['2021']) == 10
        assert imputed.loc['2021'].sum() != pytest.approx(120.0)
        # 2023 n'a plus d'ancre dans la trame : la fenêtre stricte s'arrête
        # avant, et aucune de ses lignes n'est imputée
        assert imputed.index.max() == pd.Timestamp('2022-12-31')

    def test_single_observation_entity(self, mixed_freq_panel_multifrequency):
        """Une entité réduite à une observation ne fait pas échouer l'étape."""
        data = mixed_freq_panel_multifrequency.copy()
        italy = data.index.get_level_values(0) == 'IT'
        data.loc[italy, 'v'] = np.nan
        data.iloc[np.flatnonzero(italy)[0], data.columns.get_loc('v')] = 10.0
        imputer = fit_with_spy(data)
        assert len(imputer.imputation_plan_) >= 1

    def test_panel_with_no_imputable_variable_yields_an_empty_plan(self):
        """Toutes les colonnes déjà à la fréquence cible : plan vide, aucune erreur."""
        dates = pd.date_range('2021-01-31', periods=24, freq='ME')
        index = pd.MultiIndex.from_product([['FR', 'DE'], dates], names=['country', 'date'])
        data = pd.DataFrame(
            {'m1': np.arange(48, dtype=float), 'm2': np.arange(48, dtype=float)},
            index=index,
        )
        imputer = fit_with_spy(data)
        assert len(imputer.imputation_plan_) == 0
        assert imputer.imputation_models_ == {}
