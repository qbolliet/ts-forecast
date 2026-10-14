"""Phases 0 to 6 of ``HighFrequencyImputer.fit``.

Covers the fitted attributes of §13.2, the input contract (B14, B15, B16,
B19), the per-(entity, column) frequencies and classification (§2.5), the
wiring of the §12.2 components, the interpolation fallback (D6), the edge
cases of step execution, the public scenarios reaching the branches of the
fit that the rest of the package leaves uncovered (prompt F11: window that
cannot be computed or holds no anchor, composite stage label, additive
transformer returning a pair, target on the pre-conversion index, refit
purging ``cv_``) and the aggregated warnings of phase 5 (§12.5,
ANO-FREQ-059).
"""

# Modules de base
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

# Objets testés
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer
from tsforecast.frequency.imputation_plan import INTERPOLATE_FALLBACK
from tsforecast.frequency.training_set_builder import TrainingSetBuilder

# Outils de test partagés
from tests.support.estimators import (
    ConstantEstimator,
    FailingEstimator,
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


# Doublure : transformateur additif de type XY, identité rendant (X, y)
class _PairReturningIdentity(BaseEstimator, TransformerMixin):
    """Identity additive transformer returning the ``(X, y)`` pair of an XY transformer."""

    def fit(self, X, y=None):
        """Learn nothing.

        Args:
            X: Data.
            y: Ignored.

        Returns:
            self.
        """
        return self

    def transform(self, X, y=None):
        """Return a copy of X paired with y.

        Args:
            X: Data.
            y: Optional target.

        Returns:
            Tuple ``(X copy, y)``.
        """
        return X.copy(), y

    def fit_transform(self, X, y=None, **fit_params):
        """Fit, then return the pair.

        Args:
            X: Data.
            y: Optional target.
            **fit_params: Ignored.

        Returns:
            Tuple ``(X copy, y)``.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X, y=None):
        """Return X untouched.

        Args:
            X: Data.
            y: Ignored.

        Returns:
            X.
        """
        return X


# Doublure : estimateur qui ne prédit NaN qu'une ligne sur deux
class _EveryOtherNaNEstimator(BaseEstimator):
    """Estimator predicting 1.0 on even rows and NaN on odd rows."""

    def fit(self, X, y):
        """Learn nothing.

        Args:
            X: Features.
            y: Target.

        Returns:
            self.
        """
        return self

    def predict(self, X):
        """Return 1.0 and NaN alternately.

        Args:
            X: Features.

        Returns:
            Array of ``len(X)`` values.
        """
        values = np.ones(len(X))
        values[1::2] = np.nan
        return values


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

    def test_target_on_the_pre_conversion_index_is_repositioned(
        self, mixed_freq_panel_heterogeneous
    ):
        """y indexé comme le frame plat est replacé sur l'index converti (B14, B15).

        ``panel_cols`` et ``time_col`` font passer les colonnes dans l'index
        de X avant le fit ; y garde l'index d'entiers du frame plat, qui
        correspond ligne à ligne à l'index d'origine de X : il est replacé sur
        le MultiIndex converti au lieu d'être refusé.
        """
        flat = mixed_freq_panel_heterogeneous[['m1', 'q1', 'a2']].reset_index()
        target = mixed_freq_panel_heterogeneous['a1'].reset_index(drop=True)
        imputer = make_imputer(time_col='date', panel_cols=['country'])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(flat, target)

        # La cible est nommée, détectée par entité et imputée
        assert imputer.target_column_ == 'a1'
        assert imputer.detected_frequencies_[('FR', 'a1')] == 'Y'
        assert 'a1' in {step.var_name for step in imputer.imputation_plan_}

    @pytest.mark.parametrize(
        'reduce',
        [lambda data: data.iloc[[11]], lambda data: data * np.nan],
        ids=['single_row', 'all_nan_frame'],
    )
    def test_no_detectable_frequency_raises(self, reference_timeseries, reduce):
        """Aucune fréquence détectable sur aucune colonne : ValueError explicite.

        Une seule ligne, ou un jeu entièrement NaN : chaque couple est
        indétectable (moins de deux observations), et il ne reste rien à
        classer.
        """
        with pytest.raises(ValueError, match='Could not detect frequency for any column'):
            fit_quietly(make_imputer(), reduce(reference_timeseries))

    def test_refit_under_the_frequency_order_purges_cv_(self, reference_timeseries):
        """Un second fit sous l'ordre 'frequency' efface le cv_ du premier (§13.2)."""
        imputer = fit_quietly(
            make_imputer(fit_predict_order='cv', cv=3), reference_timeseries
        )
        assert hasattr(imputer, 'cv_')

        imputer.set_params(fit_predict_order='frequency')
        fit_quietly(imputer, reference_timeseries)

        assert not hasattr(imputer, 'cv_')

    def test_additive_transformer_returning_a_pair_is_unpacked(self, reference_timeseries):
        """Un transformateur additif de type XY, qui rend (X, y), est déballé (phase 2).

        La doublure est l'identité : le plan et les valeurs imputées sont
        exactement ceux d'un ajustement sans transformateur.
        """
        imputer = fit_quietly(
            make_imputer(additive_transformer=_PairReturningIdentity()), reference_timeseries
        )
        bare = fit_quietly(make_imputer(), reference_timeseries)

        assert [(s.pred_freq_label, s.var_name) for s in imputer.imputation_plan_] == [
            (s.pred_freq_label, s.var_name) for s in bare.imputation_plan_
        ]
        values, _provenance = replay(imputer, reference_timeseries)
        bare_values, _bare_provenance = replay(bare, reference_timeseries)
        pd.testing.assert_frame_equal(values, bare_values)

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

    def test_composite_stage_label_when_entities_diverge(self, mixed_freq_panel_multifrequency):
        """Une étape liant des fréquences différentes porte un libellé composite (§5.2).

        Sous ``impute_intermediate_frequencies=False``, l'unique étape est la
        liaison ``{FR: M, DE: Q, IT: M}`` elle-même. DE a pour cible Q, où q1
        et v sont déjà : rien ne lui est imputé ; FR reçoit q1 et v, IT q1
        seulement (v y est mensuelle).
        """
        imputer = fit_quietly(
            make_imputer(target_frequency={'FR': 'M', 'DE': 'Q', 'IT': 'M'}),
            mixed_freq_panel_multifrequency,
        )
        label = "('DE',)=Q+('FR',)=M+('IT',)=M"

        assert {step.pred_freq_label for step in imputer.imputation_plan_} == {label}
        assert plan_groups(imputer, label) == {('q1', 'Q'): (FR, IT), ('v', 'Y'): (FR,)}

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

    @pytest.mark.parametrize(
        'fixture_name', ['reference_timeseries', 'mixed_freq_panel_multifrequency']
    )
    def test_duplicated_index_is_refused(self, fixture_name, request):
        """Un index dupliqué est refusé explicitement, sur série comme sur panel.

        Une date (ou un couple (entité, date)) présente deux fois compterait
        double dans les agrégations exactes et dans les totaux de période
        (ANO-FREQ-022, ANO-FREQ-039) : la validation stricte du contrat
        d'entrée (§12.5) la refuse avant tout calcul. Le doublon est trié à
        sa place, pour que le refus porte sur la duplication et non sur le
        désordre.
        """
        data = request.getfixturevalue(fixture_name)
        duplicated = pd.concat([data, data.iloc[[5]]]).sort_index()

        with pytest.raises(ValueError, match='duplicate'):
            fit_with_spy(duplicated)

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


    def test_all_nan_predictions_fall_back_and_warn(self, reference_timeseries):
        """Un modèle qui ne prédit que des NaN est traité comme un échec de prédiction.

        ANO-FREQ-064 : rien n'étant produit, l'étape est dégradée en repli
        d'interpolation — le plan dit ce qui a réellement été fait — et
        l'avertissement agrégé des étapes dégradées le nomme, comme pour une
        exception. Valeur d'or : les 36 mois de chaque colonne interpolés.
        """
        imputer = make_imputer(estimator=ConstantEstimator(constant=np.nan))

        degraded = _messages_of_family(
            lambda: imputer.fit(reference_timeseries), 'degraded during the fit'
        )

        assert [(step.var_name, step.is_fallback, step.n_written)
                for step in imputer.imputation_plan_] == [
            ('q1', True, 36), ('a1', True, 36), ('a2', True, 36),
        ]
        assert len(degraded) == 1
        assert degraded[0].count('the model predicted only NaN, interpolation fallback') == 3

    def test_partly_nan_predictions_are_left_unwritten_and_named(self, reference_timeseries):
        """Des prédictions NaN éparses restent non écrites, et l'avertissement agrégé le dit.

        L'estimateur ne prédit NaN qu'une ligne sur deux : l'étape reste un
        modèle, ses autres cellules sont écrites ; l'année dont une
        sous-période manque n'est pas recalée (§11.1). Valeur d'or pour a1 :
        18 cellules NaN sur 36, 18 écrites.
        """
        imputer = make_imputer(estimator=_EveryOtherNaNEstimator())

        degraded = _messages_of_family(
            lambda: imputer.fit(reference_timeseries), 'degraded during the fit'
        )

        step = next(s for s in imputer.imputation_plan_ if s.var_name == 'a1')
        assert (step.is_fallback, step.n_written) == (False, 18)
        assert len(degraded) == 1
        assert "'a1' at stage M: 18 of 36 prediction(s) are NaN and stay unimputed" in degraded[0]


# =============================================================================
# Fenêtres sans entraînement possible (phases 1 et 5)
# =============================================================================
class TestWindowWithoutTraining:
    """Fenêtres vides ou sans ancre : repli d'interpolation, jamais d'exception."""

    def test_disjoint_coverages_warn_once_and_impute_nothing(self, reference_timeseries):
        """Couvertures disjointes : UN avertissement, fenêtres vides, rien d'écrit.

        m1 n'est observée qu'en 2021 et q1 qu'en 2023 : aucune période ne
        couvre toutes les colonnes, la fenêtre stricte — donc celle
        d'entraînement et celle d'imputation — est vide. Chaque étape retombe
        sur le repli sans ligne d'entraînement, et la grille de prédiction
        vide n'écrit rien. Un seul message le dit : l'imputeur n'ajoute plus
        un second avertissement annonçant un repli qui n'a pas lieu.
        """
        data = reference_timeseries[['m1', 'q1', 'a1']].copy()
        data.loc['2022-01-01':, 'm1'] = np.nan
        data.loc[:'2022-12-31', 'q1'] = np.nan
        imputer = make_imputer()

        fit_messages = _messages_of_family(lambda: imputer.fit(data), '')
        transform_messages = _messages_of_family(lambda: imputer.transform(data), '')

        assert len(fit_messages) == 1
        assert 'There is no period in the DataFrame' in fit_messages[0]
        assert 'no value can be imputed' in fit_messages[0]
        # Le transform ne recompte pas les lignes d'une fenêtre qui n'existe pas
        assert transform_messages == fit_messages
        assert not imputer.training_window_mask_.any()
        assert not imputer.imputation_window_mask_.any()
        assert [(step.var_name, step.is_fallback, step.n_training_rows, step.n_written)
                for step in imputer.imputation_plan_] == [
            ('q1', True, 0, 0), ('a1', True, 0, 0),
        ]

    def test_panel_without_any_strict_window_imputes_nothing(self, panel_reference_full):
        """Un panel sans fenêtre stricte nulle part n'impute rien, comme la série temporelle.

        Mêmes couvertures disjointes que ci-dessus, pour les trois entités.
        Le calculateur levait « No imputation window found for any entity »,
        l'imputeur retombait sur des masques tout-vrais (« Using all available
        data ») et imputait les 108 cellules de q1 et de a1 (ANO-FREQ-072).
        Désormais : masques tout-faux, rien d'écrit, un message au fit qui
        nomme les trois entités, un seul au transform.
        """
        data = panel_reference_full[['m1', 'q1', 'a1']].copy()
        dates = data.index.get_level_values(-1)
        data.loc[dates >= '2022-01-01', 'm1'] = np.nan
        data.loc[dates <= '2022-12-31', 'q1'] = np.nan
        imputer = make_imputer(keep_lower_frequencies=False)

        fit_messages = _messages_of_family(lambda: imputer.fit(data), '')
        transform_messages = _messages_of_family(lambda: imputer.transform(data), '')

        assert [step.n_written for step in imputer.imputation_plan_] == [0, 0]
        assert len(fit_messages) == 1
        assert "There is no period for 3 entity(ies)" in fit_messages[0]
        assert len(transform_messages) == 1 and 'no value can be imputed' in transform_messages[0]

    def test_training_window_without_any_anchor_falls_back(self, reference_timeseries):
        """Fenêtre d'entraînement sans aucune ancre : repli, 0 ligne, 0 covariable.

        m1 s'arrête en novembre 2021 : la fenêtre va du 2021-01-31 au
        2021-11-30, avant la première ancre annuelle de a1. Valeur d'or : 11
        mois interpolés, l'année 2021 tronquée par la fenêtre n'étant pas
        recalée.
        """
        data = reference_timeseries[['m1', 'a1']].copy()
        data.loc['2021-12-31':, 'm1'] = np.nan
        imputer = make_imputer()

        degraded = _messages_of_family(lambda: imputer.fit(data), 'degraded during the fit')

        assert imputer.training_window_ == (
            pd.Timestamp('2021-01-31'), pd.Timestamp('2021-11-30')
        )
        (step,) = imputer.imputation_plan_
        assert (step.is_fallback, step.n_training_rows, step.feature_cols, step.n_written) == (
            True, 0, (), 11,
        )
        assert len(degraded) == 1
        assert "'a1' at stage M: empty training set (0 row(s), 0 covariate(s))" in degraded[0]

    def test_finer_rows_alone_keep_their_covariates(self, reference_timeseries):
        """Sous True, des lignes d'entraînement toutes plus fines que le bloc gardent m1.

        Même fenêtre que ci-dessus, avec q1 pour créer l'étape Q. À l'étape
        Q, a1 n'a aucune ancre dans la fenêtre : repli, qui écrit T1 à T3
        2021 (origine 'interpolated'). À l'étape M, ces trois trimestres sont
        les SEULES lignes de a1 — couche Q, la couche Y étant vide — et m1 y
        vaut la somme de chaque trimestre (§5.4bis, I18 b). La sélection des
        covariables lit chaque couche à sa fréquence (ANO-FREQ-066) : lue à
        la fréquence du bloc (Y) sur ces dates trimestrielles, m1 y paraissait
        vide, était écartée, et a1 retombait en repli avec trois lignes
        exploitables.
        """
        data = reference_timeseries[['m1', 'q1', 'a1']].copy()
        data.loc['2021-12-31':, 'm1'] = np.nan
        imputer = fit_quietly(make_imputer(impute_intermediate_frequencies=True), data)

        steps = {(s.pred_freq_label, s.var_name): s for s in imputer.imputation_plan_}
        # L'étape Q a bien écrit trois trimestres par repli
        assert (steps[('Q', 'a1')].is_fallback, steps[('Q', 'a1')].n_written) == (True, 3)
        # Étape M : trois lignes trimestrielles
        step = steps[('M', 'a1')]
        assert step.n_training_rows == 3
        # Attente : m1 retenue, modèle ajusté sur une cible d'origine
        # 'interpolated' (les souillures d'une étape de repli sont remises à
        # 'none' : elles ne se lisent qu'une fois le modèle ajusté)
        assert 'm1' in step.feature_cols
        assert not step.is_fallback
        assert step.target_taint == 'interpolated'

    def test_covariate_absent_from_the_training_grid_is_not_selected(
        self, reference_timeseries
    ):
        """Une covariable vide sur la grille d'entraînement n'est jamais retenue (§7.2).

        ``late`` reprend m1 à partir de février 2023 seulement. Sous
        ``training_scope='unrestricted'``, a1 s'entraîne sur ses trois ancres
        annuelles, où ``late`` — aucune année complète — ne s'agrège nulle
        part ; la grille de prédiction (février-décembre 2023) la porte
        pourtant. Une covariable non vide sur une seule des deux grilles
        entraînerait un coefficient sans support : elle est écartée.
        """
        data = reference_timeseries.copy()
        data['late'] = data['m1'].where(data.index >= '2023-02-01')

        imputer = fit_quietly(make_imputer(training_scope='unrestricted'), data)

        step = next(s for s in imputer.imputation_plan_ if s.var_name == 'a1')
        assert step.feature_cols == ('m1', 'q1', 'a2')


# =============================================================================
# Avertissements agrégés de la phase 5 (§12.3 PHASE 6, §12.5)
# =============================================================================
# Messages d'une famille d'avertissements, relevés pendant un appel
def _messages_of_family(call, family: str) -> list:
    """Run a call and return the warning messages containing ``family``.

    Args:
        call: Zero-argument callable to run.
        family: Fragment identifying the family of the message.

    Returns:
        The matching messages, in emission order.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        call()
    return [str(item.message) for item in caught if family in str(item.message)]


class TestAggregatedWarnings:
    """Un avertissement par famille et par phase, jamais un par variable × étape."""

    # Jeu TS dont les deux colonnes annuelles portent des totaux NÉGATIFS
    @pytest.fixture
    def negative_annual_totals(self, reference_timeseries) -> pd.DataFrame:
        """``TS`` whose ``a1`` and ``a2`` are negated: -120 / -132 / -150 and -60 / -66 / -72."""
        data = reference_timeseries.copy()
        data[['a1', 'a2']] = -data[['a1', 'a2']]
        return data

    def test_flipped_profiles_of_two_columns_over_two_stages_warn_once(
        self, negative_annual_totals
    ):
        """Deux colonnes à signe inversé sur deux étapes : UN avertissement « flipped ».

        L'estimateur constant prédit +1 sur chaque sous-période : l'agrégat de
        chaque année est positif face à un total négatif, aux étapes Q ET M,
        pour a1 comme pour a2 — quatre recalages dégénérés, qui donnaient
        chacun leur message (ANO-FREQ-059). Valeur d'or : 2 colonnes × 3
        années = 6 périodes observées ; une année inversée aux deux étapes
        compte une fois, le message dénombrant des périodes observées et non
        des recalages.
        """
        imputer = make_imputer(
            estimator=ConstantEstimator(constant=1.0),
            impute_intermediate_frequencies=True,
        )

        flipped = _messages_of_family(
            lambda: imputer.fit(negative_annual_totals), 'flipped the predicted profile'
        )

        # Les deux étapes ont bien été parcourues par les deux colonnes
        assert [(s.pred_freq_label, s.var_name) for s in imputer.imputation_plan_] == [
            ('Q', 'a1'), ('Q', 'a2'), ('M', 'q1'), ('M', 'a1'), ('M', 'a2'),
        ]
        # Un seul message, nommant les deux colonnes et le total des périodes
        assert len(flipped) == 1
        assert "6 period(s), in column(s) ['a1', 'a2']" in flipped[0]

    def test_the_replay_warns_once_too(self, negative_annual_totals):
        """Le rejeu du plan au transform agrège de même : un seul avertissement."""
        imputer = fit_quietly(
            make_imputer(
                estimator=ConstantEstimator(constant=1.0),
                impute_intermediate_frequencies=True,
            ),
            negative_annual_totals,
        )

        flipped = _messages_of_family(
            lambda: imputer.transform(negative_annual_totals),
            'flipped the predicted profile',
        )

        assert len(flipped) == 1
        assert "6 period(s), in column(s) ['a1', 'a2']" in flipped[0]

    def test_a_covariate_interpolated_for_several_variables_warns_once(
        self, reference_timeseries
    ):
        """Une covariable interpolée à chaque usage n'avertit qu'une fois.

        Ancres de a2 alternées (+100, -10, +100) : l'interpolation linéaire de
        2022 descend de 100 à -10, son agrégat est positif face à un total
        négatif. a2 est matérialisée pour q1 et pour a1, sur la grille
        d'entraînement comme sur celle de prédiction — six recalages de la
        même année, et autant de messages avant la correction. Valeur d'or :
        une seule période, 2022 de a2.
        """
        data = reference_timeseries.copy()
        data['a2'] = np.nan
        data.loc['2021-12-31', 'a2'] = 100.0
        data.loc['2022-12-31', 'a2'] = -10.0
        data.loc['2023-12-31', 'a2'] = 100.0

        flipped = _messages_of_family(
            lambda: make_imputer(estimator=ConstantEstimator(constant=1.0)).fit(data),
            'flipped the predicted profile',
        )

        assert len(flipped) == 1
        assert "1 period(s), in column(s) ['a2'] (e.g. 2022)" in flipped[0]

    def test_zero_aggregates_of_two_columns_warn_once(self, reference_timeseries):
        """Famille « aggregate to zero » : un message pour toutes les colonnes et les deux étapes.

        L'estimateur constant prédit 0 : chaque année de a1 et de a2 agrège
        à zéro face à un total non nul, à l'étape Q comme à l'étape M, et
        chaque trimestre de q1 à l'étape M. Valeur d'or : 3 années × 2
        colonnes annuelles + 12 trimestres de q1 = 18 périodes, prédictions
        brutes gardées.
        """
        imputer = make_imputer(
            estimator=ConstantEstimator(constant=0.0),
            impute_intermediate_frequencies=True,
        )

        zero = _messages_of_family(
            lambda: imputer.fit(reference_timeseries), 'aggregate to zero'
        )

        assert len(zero) == 1
        assert (
            "18 period(s) with a non-zero observed total, in column(s) ['a1', 'a2', 'q1']"
            in zero[0]
        )

    def test_no_degenerate_period_no_warning(self, reference_timeseries):
        """Un ajustement nominal n'émet aucun avertissement de recalage."""
        messages = _messages_of_family(
            lambda: make_imputer().fit(reference_timeseries), 'Aggregation constraint'
        )

        assert messages == []
