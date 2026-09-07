"""Tests for tsforecast.frequency.high_frequency_imputer2.

Lot L9 de [SPEC] high_frequency_imputer2_architecture.md : la classe
orchestratrice, ses validations d'`__init__` (§13.1), ses attributs ajustés
(§13.2) et les phases 0 à 4 du fit (§12.3). La PHASE 5 est le lot L10 et le
`transform` le lot L12 : les deux lèvent ici `NotImplementedError`.

Couvre en particulier la conformité sklearn (§12.5 : B3, B14, B15, B16, B20),
les fréquences détectées PAR (entité, colonne) (§2.1, §2.5, jeu `PANEL-F`), la
classification par couple, le câblage du `TrainingSetBuilder` (§5.8, §7.2) et
l'invariant statique I13 (§16).

Lot purement additif : hfi et ses tests restent intacts.
"""
# Modules de base
import re
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# Objets testés
from tsforecast.frequency.high_frequency_imputer2 import (
    ELIGIBLE_ORIGINS,
    HighFrequencyImputer2,
)
from tsforecast.frequency.imputation_plan import INTERPOLATE_FALLBACK
from tsforecast.frequency.provenance import ProvenanceType
from tsforecast.frequency.stage_scaler import StageScaler

# Clés d'entité du jeu PANEL-F, sous forme de tuples (§2.5)
FR, DE, IT = ('FR',), ('DE',), ('IT',)


# Fabrique privée d'un imputeur valide, surchargeable paramètre par paramètre
def _make_imputer(**overrides) -> HighFrequencyImputer2:
    """Build a valid imputer, overriding the given parameters."""
    params = dict(target_frequency='M', estimator=LinearRegression())
    params.update(overrides)
    return HighFrequencyImputer2(**params)


# Fabrique privée d'un fit silencieux
def _fit_quietly(imputer: HighFrequencyImputer2, data: pd.DataFrame) -> HighFrequencyImputer2:
    """Fit the imputer, swallowing the warnings this lot legitimately emits."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return imputer.fit(data)


# =============================================================================
# Validations d'__init__ (§13.1)
# =============================================================================
class TestParameterValidation:
    """Validation ligne à ligne du tableau §13.1, sans transformation (B3)."""

    # Un cas par ligne de littéral du tableau : le message énumère les valeurs
    @pytest.mark.parametrize(
        'param, value, expected_values',
        [
            ('covariate_strategy', 'nope', ('tolerate_nan', 'interpolate', 'model')),
            ('covariate_fallback', 'model', ('interpolate', 'tolerate_nan')),
            ('covariate_eligibility', 'some', ('any_entity', 'all_entities')),
            ('fit_predict_order', 'random', ('frequency', 'cv')),
            ('on_frequency_mismatch', 'ignore', ('error', 'warn')),
            ('imputation_scope', 'wide', ('strict', 'extended_backward')),
            ('training_scope', 'wide', ('strict', 'unrestricted')),
        ],
    )
    def test_literal_membership_lists_admissible_values(self, param, value, expected_values):
        """Une valeur hors du Literal lève ValueError en listant les valeurs admises."""
        with pytest.raises(ValueError) as excinfo:
            _make_imputer(**{param: value})
        message = str(excinfo.value)
        assert param in message
        for admissible in expected_values:
            assert repr(admissible) in message or admissible in message

    def test_training_scope_admits_unrestricted_but_imputation_scope_does_not(self):
        """Seul training_scope admet 'unrestricted' : les deux Literal diffèrent."""
        # Accepté côté entraînement
        _make_imputer(training_scope='unrestricted')
        # Refusé côté prédiction, le collaborateur ne le connaissant pas
        with pytest.raises(ValueError):
            _make_imputer(imputation_scope='unrestricted')

    def test_target_frequency_wrong_type_raises(self):
        """Une fréquence cible ni str ni dict lève TypeError."""
        with pytest.raises(TypeError, match='target_frequency'):
            _make_imputer(target_frequency=12)

    def test_target_frequency_empty_dict_raises(self):
        """Un dictionnaire de fréquences cibles vide est refusé."""
        with pytest.raises(ValueError, match='cannot be empty'):
            _make_imputer(target_frequency={})

    def test_estimator_without_predict_raises(self):
        """Un estimateur sans 'predict' est refusé, le message nommant la méthode."""
        class _NoPredict:
            def fit(self, X, y):
                return self

        with pytest.raises(ValueError, match="'predict' method"):
            _make_imputer(estimator=_NoPredict())

    def test_estimator_dict_admits_default_key(self):
        """La clé '__default__' est admise dans la forme dictionnaire."""
        _make_imputer(estimator={'__default__': LinearRegression()})

    def test_estimator_empty_dict_raises(self):
        """Un dictionnaire d'estimateurs vide est refusé."""
        with pytest.raises(ValueError, match='cannot be empty'):
            _make_imputer(estimator={})

    def test_additive_transformer_requires_fit_transform_and_inverse(self):
        """Le transformateur additif doit exposer fit_transform ET inverse_transform."""
        class _OnlyFitTransform:
            def fit_transform(self, X, y=None):
                return X

        with pytest.raises(ValueError) as excinfo:
            _make_imputer(additive_transformer=_OnlyFitTransform())
        assert 'inverse_transform' in str(excinfo.value)

    @pytest.mark.parametrize('value', ['yes', 0, 1, None, 'covariates'])
    def test_impute_intermediate_frequencies_rejects_other_values(self, value):
        """Toute valeur hors des trois modalités est refusée, 0 et 1 compris."""
        with pytest.raises(ValueError, match='impute_intermediate_frequencies'):
            _make_imputer(impute_intermediate_frequencies=value)

    def test_interpolation_method_dict_of_str_accepted(self):
        """La forme dictionnaire de la méthode d'interpolation est admise."""
        _make_imputer(interpolation_method={'a1': 'linear', 'q1': 'cubic'})

    def test_interpolation_method_wrong_type_raises(self):
        """Une méthode d'interpolation non textuelle est refusée."""
        with pytest.raises((ValueError, TypeError)):
            _make_imputer(interpolation_method=3)

    @pytest.mark.parametrize('value', [-0.5, 1.5, {'a1': 2.0}])
    def test_interpolation_anchor_outside_unit_interval_raises(self, value):
        """Un ancrage hors de [0, 1] est refusé, forme dictionnaire comprise."""
        with pytest.raises((ValueError, TypeError)):
            _make_imputer(interpolation_anchor=value)

    def test_interpolation_anchor_accepts_none_float_and_dict(self):
        """None, un float de [0, 1] et un dict de ces valeurs sont admis."""
        _make_imputer(interpolation_anchor=None)
        _make_imputer(interpolation_anchor=0.5)
        _make_imputer(interpolation_anchor={'a1': 0.0, 'q1': None})

    @pytest.mark.parametrize('value', [1, 'five', 3.5])
    def test_cv_invalid_forms_raise(self, value):
        """Un cv entier < 2, ou d'un type inattendu, est refusé."""
        with pytest.raises(ValueError, match='cv'):
            _make_imputer(cv=value)

    def test_cv_accepts_none_int_splitter_and_iterable(self):
        """Les quatre formes admises de cv passent la validation."""
        _make_imputer(cv=None)
        _make_imputer(cv=5)
        _make_imputer(cv=KFold(n_splits=3))
        _make_imputer(cv=[(np.array([0, 1]), np.array([2]))])

    def test_cv_is_not_resolved_at_init(self):
        """check_cv n'est pas appelé à l'__init__ : cv reste la valeur reçue (B3)."""
        imputer = _make_imputer(cv=4)
        assert imputer.cv == 4
        assert not hasattr(imputer, 'cv_')

    @pytest.mark.parametrize('value', [3, ['a']])
    def test_cv_scoring_must_be_str_or_callable(self, value):
        """Un score de validation croisée ni textuel ni appelable est refusé."""
        with pytest.raises(ValueError, match='cv_scoring'):
            _make_imputer(cv_scoring=value)

    def test_min_cv_train_size_below_one_raises(self):
        """min_cv_train_size doit valoir au moins 1 (§13.1)."""
        with pytest.raises(ValueError, match='min_cv_train_size'):
            _make_imputer(min_cv_train_size=0)

    def test_min_cv_train_size_one_is_accepted(self):
        """La borne basse du §13.1 est 1, et non 2 comme dans hfi."""
        assert _make_imputer(min_cv_train_size=1).min_cv_train_size == 1

    def test_min_cv_train_size_wrong_type_raises(self):
        """min_cv_train_size doit être un entier."""
        with pytest.raises(TypeError, match='min_cv_train_size'):
            _make_imputer(min_cv_train_size=2.5)

    @pytest.mark.parametrize('param', ['coverage_threshold', 'training_coverage_threshold'])
    @pytest.mark.parametrize('value', [-0.1, 1.1])
    def test_coverage_thresholds_outside_unit_interval_raise(self, param, value):
        """Les deux seuils de couverture doivent tomber dans [0, 1]."""
        with pytest.raises(ValueError, match=param):
            _make_imputer(**{param: value})

    def test_training_coverage_threshold_admits_none(self):
        """Seul le seuil d'entraînement admet None, où il suit celui de prédiction."""
        assert _make_imputer(training_coverage_threshold=None).training_coverage_threshold is None

    @pytest.mark.parametrize('value', ['mean', 'minmax', {'a1': 'zscore'}])
    def test_scale_features_invalid_values_raise(self, value):
        """scale_features n'admet que False, 'constant', 'calendar' ou leur dict."""
        with pytest.raises(ValueError, match='scale_features'):
            _make_imputer(scale_features=value)

    def test_scale_features_accepts_the_four_forms(self):
        """Les quatre formes admises de scale_features passent la validation."""
        _make_imputer(scale_features=False)
        _make_imputer(scale_features='constant')
        _make_imputer(scale_features='calendar')
        _make_imputer(scale_features={'a1': 'calendar', 'q1': False})

    @pytest.mark.parametrize('value', ['mean', 'last'])
    def test_aggregation_constraint_refuses_mean_and_last(self, value):
        """'mean' et 'last' sont retirés de l'API (D20), le message nommant l'échappatoire."""
        with pytest.raises(ValueError) as excinfo:
            _make_imputer(aggregation_constraint=value)
        message = str(excinfo.value)
        assert 'additive_transformer' in message
        assert 'sum' in message

    def test_aggregation_constraint_accepts_sum_none_and_dict(self):
        """'sum', None et le dict de ces deux valeurs sont admis, clé '__default__' comprise."""
        _make_imputer(aggregation_constraint='sum')
        _make_imputer(aggregation_constraint=None)
        _make_imputer(aggregation_constraint={'a1': None, '__default__': 'sum'})

    @pytest.mark.parametrize(
        'param', ['keep_lower_frequencies', 'restore_original_values', 'verbose']
    )
    @pytest.mark.parametrize('value', ['yes', 1, None])
    def test_booleans_are_validated_as_a_group(self, param, value):
        """Les trois booléens sont validés ensemble, un entier ne passant pas pour un bool."""
        with pytest.raises(TypeError, match=param):
            _make_imputer(**{param: value})


# =============================================================================
# Contrat de paramètres : B3, paramètres supprimés, NotFittedError
# =============================================================================
class TestParameterContract:
    """Conformité sklearn de l'espace de paramètres (§12.5, B3 et B20)."""

    def test_impute_intermediate_frequencies_covariates_only_is_not_true(self):
        """'covariates_only' est accepté et n'est jamais confondu avec True."""
        imputer = _make_imputer(impute_intermediate_frequencies='covariates_only')
        # La valeur est stockée telle que reçue
        assert imputer.impute_intermediate_frequencies == 'covariates_only'
        # Elle est truthy, mais n'est PAS True : c'est tout le piège du paramètre
        assert bool(imputer.impute_intermediate_frequencies) is True
        assert imputer.impute_intermediate_frequencies is not True
        assert imputer.impute_intermediate_frequencies is not False

    @pytest.mark.parametrize(
        'removed',
        [
            'cascade_refitting',
            'train_on_partial_coverage',
            'train_on_partial_fit_order',
            'enforce_period_totals',
            'cv_n_splits',
            'disaggregate_anchors',
        ],
    )
    def test_removed_parameters_are_rejected(self, removed):
        """Les six paramètres supprimés de l'API lèvent TypeError."""
        with pytest.raises(TypeError):
            _make_imputer(**{removed: True})

    def test_get_params_returns_untouched_values(self):
        """get_params rend les objets REÇUS, sans normalisation (B3)."""
        target_frequency = {'FR': 'monthly', 'DE': 'M'}
        scale_features = {'v': 'calendar', 'q1': False}
        splitter = KFold(n_splits=3)
        anchors = {'v': 0.0}
        constraint = {'v': None, '__default__': 'sum'}

        imputer = _make_imputer(
            target_frequency=target_frequency,
            scale_features=scale_features,
            cv=splitter,
            interpolation_anchor=anchors,
            aggregation_constraint=constraint,
            fit_predict_order='cv',
        )
        params = imputer.get_params()

        # Identité, et non simple égalité : "clone" repose dessus
        assert params['target_frequency'] is target_frequency
        assert params['scale_features'] is scale_features
        assert params['cv'] is splitter
        assert params['interpolation_anchor'] is anchors
        assert params['aggregation_constraint'] is constraint
        # 'monthly' n'a pas été normalisé en 'M' au passage
        assert params['target_frequency']['FR'] == 'monthly'

    def test_clone_roundtrip(self):
        """clone reproduit l'imputeur à l'identique sur un jeu exotique."""
        imputer = _make_imputer(
            target_frequency={'FR': 'M', 'DE': 'Q'},
            scale_features={'v': 'calendar'},
            cv=KFold(n_splits=3),
            fit_predict_order='cv',
            impute_intermediate_frequencies=False,
            aggregation_constraint={'v': None},
            training_scope='unrestricted',
        )
        cloned = clone(imputer)

        original_params = imputer.get_params()
        cloned_params = cloned.get_params()
        assert set(original_params) == set(cloned_params)
        # Égalité, et non identité : "clone" copie en profondeur les paramètres
        # non-estimateurs. L'identité, elle, est le contrat de "get_params" sur
        # l'instance d'origine, vérifié par test_get_params_returns_untouched_values
        for name in ('target_frequency', 'scale_features', 'aggregation_constraint',
                     'training_scope', 'impute_intermediate_frequencies'):
            assert cloned_params[name] == original_params[name]
        # Le clone n'est jamais ajusté, et reste utilisable
        assert not hasattr(cloned, 'detected_frequencies_')

    def test_set_params_roundtrip(self):
        """set_params repose la valeur exacte, sans re-normalisation."""
        imputer = _make_imputer()
        imputer.set_params(covariate_strategy='model', min_cv_train_size=1)
        assert imputer.covariate_strategy == 'model'
        assert imputer.min_cv_train_size == 1

    def test_not_fitted_error_before_fit(self):
        """transform et inverse_transform lèvent NotFittedError avant fit (B20)."""
        imputer = _make_imputer()
        frame = pd.DataFrame(
            {'m1': [1.0, 2.0]}, index=pd.date_range('2021-01-31', periods=2, freq='ME')
        )
        with pytest.raises(NotFittedError):
            imputer.transform(frame)
        with pytest.raises(NotFittedError):
            imputer.inverse_transform(frame)

    def test_imputation_models_raises_attribute_error_before_fit(self):
        """La vue des modèles lève AttributeError, jamais NotFittedError, avant fit."""
        imputer = _make_imputer()
        assert not hasattr(imputer, 'imputation_models_')
        with pytest.raises(AttributeError, match='imputation_plan_'):
            _ = imputer.imputation_models_


# =============================================================================
# Phases 0 à 4 du fit (§12.3)
# =============================================================================
class TestFitPhases:
    """Les phases 0 à 4 renseignent les attributs ajustés du §13.2."""

    @pytest.mark.parametrize('fixture_name', ['reference_timeseries', 'mixed_freq_panel_heterogeneous'])
    def test_phases_zero_to_four_populate_attributes(self, fixture_name, request):
        """Les attributs des phases 0 à 4 sont renseignés, sur TS comme sur panel."""
        data = request.getfixturevalue(fixture_name)
        imputer = _fit_quietly(_make_imputer(), data)
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

    def test_entities_are_set_on_panel_only(self, reference_timeseries,
                                            mixed_freq_panel_multifrequency):
        """entities_ vaut None sur une série temporelle et liste les entités sur un panel."""
        assert _fit_quietly(_make_imputer(), reference_timeseries).entities_ is None
        panel_imputer = _fit_quietly(_make_imputer(), mixed_freq_panel_multifrequency)
        assert set(panel_imputer.entities_) == {FR, DE, IT}

    def test_target_is_named_and_merged(self, reference_timeseries):
        """y est fusionné dans le frame de travail sous un nom unique (B14)."""
        features = reference_timeseries.drop(columns=['a1'])
        target = reference_timeseries['a1']
        imputer = _fit_quietly(_make_imputer(), features)
        assert imputer.target_column_ is None

        imputer = _fit_quietly(_make_imputer(), features)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(features, target)
        assert imputer.target_column_ == 'a1'
        assert 'a1' in imputer.detected_frequencies_

    def test_target_frequency_dict_incomplete_raises_naming_entities(
        self, mixed_freq_panel_multifrequency
    ):
        """Un dict de fréquences cibles incomplet nomme les entités manquantes (B16)."""
        imputer = _make_imputer(target_frequency={'FR': 'M', 'DE': 'M'})
        with pytest.raises(ValueError) as excinfo:
            _fit_quietly(imputer, mixed_freq_panel_multifrequency)
        message = str(excinfo.value)
        assert 'IT' in message
        assert 'incomplete' in message

    def test_target_frequency_dict_complete_is_accepted(self, mixed_freq_panel_multifrequency):
        """Un dict nommant toutes les entités passe et est normalisé en clés tuples."""
        imputer = _fit_quietly(
            _make_imputer(target_frequency={'FR': 'M', 'DE': 'M', 'IT': 'M'}),
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
            _fit_quietly(_make_imputer(), features).fit(features, target)
        message = str(excinfo.value)
        assert 'different indices' in message
        assert 'same length' in message

    def test_three_window_masks_are_set(self, mixed_freq_panel_heterogeneous):
        """Les trois masques sont des Series booléennes à MultiIndex sur panel (§7.2)."""
        imputer = _fit_quietly(_make_imputer(), mixed_freq_panel_heterogeneous)
        for mask_name in ('strict_window_mask_', 'imputation_window_mask_',
                          'training_window_mask_'):
            mask = getattr(imputer, mask_name)
            assert isinstance(mask, pd.Series), mask_name
            assert mask.dtype == bool, mask_name
            assert isinstance(mask.index, pd.MultiIndex), mask_name

    def test_widening_training_scope_adds_rows_not_columns(self, reference_timeseries):
        """Élargir training_scope change le masque d'entraînement, jamais les colonnes (§7.2)."""
        strict = _fit_quietly(
            _make_imputer(training_scope='strict'), reference_timeseries
        )
        widened = _fit_quietly(
            _make_imputer(training_scope='unrestricted'), reference_timeseries
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
        by_frequency = _fit_quietly(_make_imputer(), reference_timeseries)
        assert not hasattr(by_frequency, 'cv_')

        by_cv = _fit_quietly(
            _make_imputer(fit_predict_order='cv', cv=3), reference_timeseries
        )
        assert hasattr(by_cv, 'cv_')
        assert by_cv.cv_.get_n_splits() == 3

    def test_frequency_progression_is_target_only_under_false(self, reference_timeseries):
        """Sous impute_intermediate_frequencies=False, la progression est [f_target]."""
        imputer = _fit_quietly(_make_imputer(), reference_timeseries)
        assert imputer.frequency_progression_ == ['M']

    def test_estimator_none_warns_once(self, reference_timeseries):
        """L'absence d'estimateur donne UN avertissement, pas un par variable."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            HighFrequencyImputer2(target_frequency='M', estimator=None).fit(
                reference_timeseries
            )
        estimator_warnings = [
            w for w in caught if 'estimator=None' in str(w.message)
        ]
        assert len(estimator_warnings) == 1

    def test_transform_not_implemented_yet(self, reference_timeseries):
        """transform et inverse_transform annoncent le lot qui les livre."""
        imputer = _fit_quietly(_make_imputer(), reference_timeseries)
        for method in ('transform', 'inverse_transform'):
            with pytest.raises(NotImplementedError) as excinfo:
                getattr(imputer, method)(reference_timeseries)
            assert 'L12' in str(excinfo.value)

    def test_intermediate_frequencies_modalities_all_fit(self, reference_timeseries):
        """Les trois modalités de l'axe 2 ajustent, aucune ne lève (§5.1)."""
        for modality in (False, 'covariates_only', True):
            imputer = _make_imputer(impute_intermediate_frequencies=modality)
            _fit_quietly(imputer, reference_timeseries)
            assert imputer.frequency_progression_[-1] == 'M'

    def test_panel_declared_by_panel_cols_on_flat_frame(self, mixed_freq_panel_multifrequency):
        """Un panel déclaré par panel_cols sur frame plat est pleinement fonctionnel (B15)."""
        flat = mixed_freq_panel_multifrequency.reset_index()
        imputer = _fit_quietly(
            _make_imputer(time_col='date', panel_cols=['country']), flat
        )
        assert imputer.is_panel_ is True
        assert set(imputer.entities_) == {FR, DE, IT}

    def test_fit_purges_stale_transform_state(self, reference_timeseries):
        """Un état laissé par un transform précédent est purgé en tête de fit (B19)."""
        imputer = _fit_quietly(_make_imputer(), reference_timeseries)
        imputer._original_X_ = 'stale'
        _fit_quietly(imputer, reference_timeseries)
        assert '_original_X_' not in imputer.__dict__

    def test_entity_never_observing_a_column_is_left_out(
        self, mixed_freq_panel_heterogeneous
    ):
        """Un couple (entité, colonne) jamais observé sort de la classification (§4.5)."""
        imputer = _fit_quietly(_make_imputer(), mixed_freq_panel_heterogeneous)
        # IT n'observe jamais climat_affaires : le couple n'a pas de fréquence
        assert (IT + ('climat_affaires',)) not in imputer.detected_frequencies_
        assert ('IT', 'climat_affaires') in imputer._undetected_frequencies_
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
        imputer = _fit_quietly(_make_imputer(), mixed_freq_panel_multifrequency)
        assert imputer.detected_frequencies_[('FR', 'v')] == 'Y'
        assert imputer.detected_frequencies_[('DE', 'v')] == 'Q'
        assert imputer.detected_frequencies_[('IT', 'v')] == 'M'
        # Les colonnes homogènes gardent la même fréquence partout
        for entity in ('FR', 'DE', 'IT'):
            assert imputer.detected_frequencies_[(entity, 'q1')] == 'Q'
            assert imputer.detected_frequencies_[(entity, 'm1')] == 'M'

    def test_variable_classification_is_per_entity_pair(self, mixed_freq_panel_multifrequency):
        """v est imputable pour FR et DE, et ne l'est pas pour IT, déjà à la cible."""
        imputer = _fit_quietly(_make_imputer(), mixed_freq_panel_multifrequency)
        imputable = set(imputer.variable_categories_['impute'])
        at_target = set(imputer.variable_categories_['target_freq'])

        assert ('FR', 'v') in imputable
        assert ('DE', 'v') in imputable
        assert ('IT', 'v') not in imputable
        assert ('IT', 'v') in at_target

    def test_imputable_pairs_group_by_source_frequency(self, mixed_freq_panel_multifrequency):
        """Les couples imputables se regroupent en (v, Y) -> {FR} et (v, Q) -> {DE}."""
        imputer = _fit_quietly(_make_imputer(), mixed_freq_panel_multifrequency)
        groups = imputer._imputable_groups(imputer.effective_target_frequency_)

        assert groups[('v', 'Y')] == (FR,)
        assert groups[('v', 'Q')] == (DE,)
        # IT n'est imputable dans aucun groupe de v
        for group_key, entities in groups.items():
            if group_key[0] == 'v':
                assert IT not in entities

    def test_detected_frequencies_adapter_keeps_both_shapes(
        self, mixed_freq_panel_multifrequency
    ):
        """L'adaptateur rend la forme par colonne qu'attendent les composants."""
        imputer = _fit_quietly(_make_imputer(), mixed_freq_panel_multifrequency)
        by_column = imputer._detected_frequencies_by_column()

        # Colonne hétérogène : forme par entité
        assert by_column['v'] == {FR: 'Y', DE: 'Q', IT: 'M'}
        # Colonnes homogènes : repli sur la forme scalaire
        assert by_column['q1'] == 'Q'
        assert by_column['m1'] == 'M'

    def test_hyperparameter_dicts_stay_keyed_by_column(self, mixed_freq_panel_multifrequency):
        """Les dicts d'hyperparamètres restent indexés par colonne, jamais par couple (D10)."""
        imputer = _fit_quietly(
            _make_imputer(
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
class TestWiring:
    """Les composants du §12.2 sont instanciés et câblés au fit."""

    def test_training_set_builder_is_wired(self, mixed_freq_panel_multifrequency, monkeypatch):
        """Le callable de masque du builder appelle get_mask_at_frequency(kind='training')."""
        imputer = _fit_quietly(_make_imputer(), mixed_freq_panel_multifrequency)

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
        imputer = _fit_quietly(
            _make_imputer(aggregation_constraint=constraint), reference_timeseries
        )
        assert imputer._covariate_materializer.aggregation_constraint is constraint
        # Le builder n'en porte aucune : la cible n'est jamais agrégée (§5.8 R3)
        assert not hasattr(imputer._training_set_builder, 'aggregation_constraint')

    def test_window_calculator_receives_the_four_window_parameters(self, reference_timeseries):
        """Le calculateur de fenêtre reçoit les quatre paramètres du §7."""
        imputer = _fit_quietly(
            _make_imputer(
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


# =============================================================================
# Invariants statiques
# =============================================================================
class TestStaticInvariants:
    """Invariants vérifiables sur le source du module (§16)."""

    def test_no_boolean_test_on_impute_intermediate_frequencies(self):
        """Aucun test de vérité booléenne sur l'axe 2 : invariant I13."""
        module_path = (
            Path(__file__).resolve().parents[2]
            / 'tsforecast' / 'frequency' / 'high_frequency_imputer2.py'
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
            Path(__file__).resolve().parents[2]
            / 'tsforecast' / 'frequency' / 'high_frequency_imputer2.py'
        )
        source = module_path.read_text(encoding='utf-8')
        for removed in ('cascade_refitting', 'train_on_partial_coverage',
                        'train_on_partial_fit_order', 'enforce_period_totals',
                        'cv_n_splits', 'disaggregate_anchors'):
            assert removed not in source, removed

    def test_both_classes_are_exported(self):
        """hfi et hfi2 coexistent dans les exports du module (§15.3)."""
        import tsforecast.frequency as frequency_module

        assert 'HighFrequencyImputer' in frequency_module.__all__
        assert 'HighFrequencyImputer2' in frequency_module.__all__
        assert frequency_module.HighFrequencyImputer2 is HighFrequencyImputer2


# =============================================================================
# PHASE 5 — Exécution des étapes (lot L10)
# =============================================================================
# Estimateur espion : il retient ce que chaque appel lui a montré
class _SpyEstimator(BaseEstimator, RegressorMixin):
    """Estimateur espion, tolérant les NaN et prédisant une constante.

    Il retient ``fit_X_``, ``fit_y_`` et la liste ``predict_X_`` des trames de
    prédiction, ce qui rend l'invariant central (I2) et l'échelle (I5)
    mesurables. Le compteur de classe ``n_fits`` mesure la règle « un seul
    ajustement par (étape, variable) » (I15).
    """

    n_fits = 0

    def __init__(self, constant: float = 1.0):
        self.constant = constant

    def fit(self, X, y):
        """Retient le jeu d'entraînement et la moyenne de la cible."""
        type(self)._record_fit()
        self.fit_X_ = X.copy()
        self.fit_y_ = y.copy()
        self.predict_X_ = []
        values = np.asarray(y, dtype=float)
        finite = values[~np.isnan(values)]
        self.mean_ = float(finite.mean()) if finite.size else 0.0
        return self

    def predict(self, X):
        """Retient la trame de prédiction et rend la moyenne apprise."""
        if not hasattr(self, 'predict_X_'):
            self.predict_X_ = []
        self.predict_X_.append(X.copy())
        return np.full(len(X), self.mean_)

    @classmethod
    def _record_fit(cls):
        """Incrémente le compteur d'ajustements de la classe."""
        _SpyEstimator.n_fits += 1


# Estimateur d'échec, pour le chemin de repli
class _FailingEstimator(BaseEstimator, RegressorMixin):
    """Estimateur dont l'ajustement échoue toujours."""

    def fit(self, X, y):
        """Lève systématiquement, pour éprouver le repli d'interpolation."""
        raise RuntimeError('deliberate fit failure')

    def predict(self, X):
        """Jamais atteint : l'ajustement a déjà échoué."""
        raise RuntimeError('deliberate predict failure')


# Fonction auxiliaire d'ajustement silencieux avec l'espion
def _fit_with_spy(data: pd.DataFrame, **overrides) -> HighFrequencyImputer2:
    """Ajuste un imputeur muni de l'estimateur espion, sans avertissement."""
    _SpyEstimator.n_fits = 0
    imputer = _make_imputer(estimator=_SpyEstimator(), **overrides)
    return _fit_quietly(imputer, data)


# Fonction auxiliaire de regroupement d'un index par entité
def _by_entity(index: pd.Index) -> dict:
    """Rend l'ensemble des dates de chaque entité d'un index."""
    if not isinstance(index, pd.MultiIndex):
        return {(): set(index)}
    grouped: dict = {}
    for key in index:
        grouped.setdefault(tuple(key[:-1]), set()).add(key[-1])
    return grouped


# Fonction auxiliaire des dates renseignées d'une colonne
def _filled_dates(frame: pd.DataFrame, column: str) -> dict:
    """Rend, par entité, les dates où la colonne est renseignée."""
    return _by_entity(frame.index[frame[column].notna().to_numpy()])


# Fonction auxiliaire d'extraction du bloc d'une entité
def _entity_block(series: pd.Series, entity: str) -> pd.Series:
    """Rend la tranche d'une entité, indexée par date seule."""
    return series.xs(entity, level=0, drop_level=True)


class TestNaNInvariant:
    """I2 — le motif de disponibilité de X_pred contient celui de X_train."""

    @pytest.mark.parametrize('strategy', ['tolerate_nan', 'interpolate', 'model'])
    @pytest.mark.parametrize(
        'fixture_name', ['reference_timeseries', 'mixed_freq_panel_heterogeneous']
    )
    def test_nan_invariant_by_stage_and_column(self, strategy, fixture_name, request):
        """Formulation D14 : inclusion des dates, par étape, colonne ET entité."""
        data = request.getfixturevalue(fixture_name)
        imputer = _fit_with_spy(data, covariate_strategy=strategy)

        checked = 0
        for step in imputer.imputation_plan_:
            if step.is_fallback:
                continue
            model = step.model
            for column in step.feature_cols:
                trained = _filled_dates(model.fit_X_, column)
                for prediction_frame in model.predict_X_:
                    grid = _by_entity(prediction_frame.index)
                    predicted = _filled_dates(prediction_frame, column)
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
    def _outputs(imputer: HighFrequencyImputer2, columns) -> dict:
        """Valeurs imputées et provenances, colonne par colonne."""
        store = imputer._covariate_materializer.imputed_store
        return {
            column: (
                store[column].sort_index().round(9),
                imputer.imputation_provenance_[column].astype(str).sort_index(),
            )
            for column in columns
        }

    @pytest.mark.parametrize('strategy', ['tolerate_nan', 'interpolate', 'model'])
    def test_column_order_invariance(self, reference_timeseries, strategy):
        """I3 — permuter les colonnes d'entrée ne change ni valeurs ni provenances."""
        columns = ['q1', 'a1', 'a2']
        straight = _fit_with_spy(reference_timeseries, covariate_strategy=strategy)
        permuted = _fit_with_spy(
            reference_timeseries[['a2', 'a1', 'q1', 'm1']],
            covariate_strategy=strategy,
        )

        for column in columns:
            values, provenance = self._outputs(straight, columns)[column]
            other_values, other_provenance = self._outputs(permuted, columns)[column]
            pd.testing.assert_series_equal(values, other_values)
            pd.testing.assert_series_equal(provenance, other_provenance)

    @pytest.mark.parametrize('strategy', ['tolerate_nan', 'interpolate'])
    def test_processing_order_indifferent_outside_model(
        self, reference_timeseries, strategy
    ):
        """I10 — hors 'model', deux ordres de traitement donnent la MÊME sortie."""
        columns = ['q1', 'a1', 'a2']
        forward = _fit_with_spy(reference_timeseries, covariate_strategy=strategy)
        # Ordre de traitement inversé : il suit l'ordre des colonnes d'entrée
        backward = _fit_with_spy(
            reference_timeseries[['a2', 'a1', 'q1', 'm1']],
            covariate_strategy=strategy,
        )
        assert [step.var_name for step in forward.imputation_plan_] != [
            step.var_name for step in backward.imputation_plan_
        ]
        for column in columns:
            pd.testing.assert_series_equal(
                self._outputs(forward, columns)[column][0],
                self._outputs(backward, columns)[column][0],
            )

    def test_imputation_order_empty_outside_model(self, reference_timeseries):
        """imputation_order_ reste vide hors covariate_strategy='model'."""
        assert not _fit_with_spy(reference_timeseries).imputation_order_
        assert _fit_with_spy(
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
            imputer = _fit_with_spy(data, **params)
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
            imputer = _fit_with_spy(
                reference_timeseries, covariate_strategy='model',
            )
        assert not calls
        assert imputer.imputation_order_ == {'M': ['a1', 'a2', 'q1']}


class TestMaterializationWays:
    """I11 — la voie de matérialisation est enregistrée et conforme à la précédence."""

    def test_materialization_way_recorded_per_step(self, reference_timeseries):
        """Une entrée par feature_col, et la voie attendue par la précédence."""
        imputer = _fit_with_spy(reference_timeseries)
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
        imputer = _fit_with_spy(reference_timeseries, covariate_strategy='tolerate_nan')
        for step in imputer.imputation_plan_:
            for column, way in step.materialization.items():
                assert way in ('identity', 'aggregate', 'raw_anchors')


class TestProvenanceFamilies:
    """I6 — les cinq familles MODEL_* sont émises exactement dans les cas du §6.3."""

    @staticmethod
    def _families(imputer: HighFrequencyImputer2, column: str) -> set:
        """Provenances distinctes portées par une colonne."""
        return {str(value) for value in imputer.imputation_provenance_[column]}

    def test_interpolate_emits_model_on_interpolated(self, reference_timeseries):
        """Une covariable plus basse que la grille suffit à souiller l'étape."""
        imputer = _fit_with_spy(reference_timeseries)
        for column in ('q1', 'a1', 'a2'):
            assert self._families(imputer, column) == {'model_on_interpolated'}

    def test_tolerate_nan_emits_model_on_true(self, reference_timeseries):
        """Aucune valeur interpolée ne circule : le modèle n'a vu que du vrai."""
        imputer = _fit_with_spy(reference_timeseries, covariate_strategy='tolerate_nan')
        for column in ('q1', 'a1', 'a2'):
            assert self._families(imputer, column) == {'model_on_true'}

    def test_model_on_imputed_only_under_model_strategy(self, reference_timeseries):
        """MODEL_ON_IMPUTED n'est émis que sous covariate_strategy='model'."""
        under_model = _fit_with_spy(reference_timeseries, covariate_strategy='model')
        # 'a1' est imputée la première : ses covariables ne sont qu'interpolées
        assert self._families(under_model, 'a1') == {'model_on_interpolated'}
        # 'a2' et 'q1' lisent ensuite l'imputation de 'a1' dans le miroir
        assert self._families(under_model, 'a2') == {'model_on_imputed'}
        assert self._families(under_model, 'q1') == {'model_on_imputed'}

        for strategy in ('interpolate', 'tolerate_nan'):
            imputer = _fit_with_spy(reference_timeseries, covariate_strategy=strategy)
            emitted = set().union(
                *(self._families(imputer, column) for column in ('q1', 'a1', 'a2'))
            )
            assert 'model_on_imputed' not in emitted

    @pytest.mark.parametrize('strategy', ['tolerate_nan', 'interpolate', 'model'])
    def test_target_families_absent_under_false(self, reference_timeseries, strategy):
        """*_TARGET et *_BOTH exigent impute_intermediate_frequencies=True."""
        imputer = _fit_with_spy(reference_timeseries, covariate_strategy=strategy)
        emitted = set().union(
            *(self._families(imputer, column) for column in ('q1', 'a1', 'a2'))
        )
        assert not emitted & {'model_on_imputed_target', 'model_on_imputed_both'}

    def test_no_disaggregated_cell_is_ever_emitted(self, reference_timeseries):
        """DISAGGREGATED n'est jamais émis par hfi2, recalage compris (D16)."""
        imputer = _fit_with_spy(reference_timeseries)
        emitted = {
            str(value) for value in imputer.imputation_provenance_.to_numpy().ravel()
        }
        assert 'disaggregated' not in emitted


class TestAggregationAdditivity:
    """I4 — chaque période complète somme au total observé."""

    def test_period_totals_additivity(self, reference_timeseries):
        """Sous 'sum', les 12 mois d'une année somment à son total annuel."""
        imputer = _fit_with_spy(reference_timeseries)
        store = imputer._covariate_materializer.imputed_store
        for year, total in [('2021', 120.0), ('2022', 132.0), ('2023', 150.0)]:
            assert store['a1'].loc[year].sum() == pytest.approx(total)

    def test_anchor_row_no_longer_carries_the_total(self, reference_timeseries):
        """La ligne d'ancre porte une valeur de sous-période, jamais le total (§11.2)."""
        imputer = _fit_with_spy(reference_timeseries)
        anchor = float(imputer._covariate_materializer.imputed_store['a1']['2021-12-31'])
        assert anchor != pytest.approx(120.0)
        assert 0.0 < anchor < 120.0

    def test_group_totals_are_those_of_the_source_frequency(
        self, mixed_freq_panel_multifrequency
    ):
        """Les totaux recalés sont annuels pour FR et trimestriels pour DE (§5.8)."""
        imputer = _fit_with_spy(mixed_freq_panel_multifrequency)
        store = imputer._covariate_materializer.imputed_store['v']

        france = _entity_block(store, 'FR')
        assert france.loc['2021'].sum() == pytest.approx(120.0)
        # Le trimestre de FR est libre : seul son total ANNUEL est contraint
        assert france.loc['2021-01':'2021-03'].sum() != pytest.approx(28.0)

        germany = _entity_block(store, 'DE')
        assert germany.loc['2021-01':'2021-03'].sum() == pytest.approx(28.0)
        assert germany.loc['2021-04':'2021-06'].sum() == pytest.approx(30.0)


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
        imputer = _fit_with_spy(reference_timeseries, covariate_strategy=strategy)
        for step in imputer.imputation_plan_:
            for column, trained, predicted in self._means(step):
                assert trained == pytest.approx(predicted, rel=1e-6), (
                    f'{step.var_name}/{column}'
                )

    def test_mixed_calendar_feature_and_constant_target(self, reference_timeseries):
        """Cas mixte : feature en 'calendar', cible en 'constant' (D15)."""
        imputer = _fit_with_spy(
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
        imputer = _fit_with_spy(
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
        imputer = _fit_with_spy(reference_timeseries, covariate_strategy='model')
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
        imputer = _fit_with_spy(reference_timeseries)
        provenance = imputer.imputation_provenance_['a1'].loc['2021']
        assert len(provenance) == 12
        assert {str(value) for value in provenance} == {'model_on_interpolated'}
        # L'ancre ne reste NI ORIGINAL NI DISAGGREGATED
        assert str(imputer.imputation_provenance_['a1']['2021-12-31']) == (
            'model_on_interpolated'
        )
        store = imputer._covariate_materializer.imputed_store['a1']
        assert store.loc['2021'].sum() == pytest.approx(120.0)

    def test_provenance_unchanged_without_aggregation_constraint(
        self, reference_timeseries
    ):
        """Sous aggregation_constraint=None, seules les VALEURS changent (D16)."""
        constrained = _fit_with_spy(reference_timeseries)
        free = _fit_with_spy(reference_timeseries, aggregation_constraint=None)

        pd.testing.assert_series_equal(
            constrained.imputation_provenance_['a1'].astype(str),
            free.imputation_provenance_['a1'].astype(str),
        )
        free_sum = free._covariate_materializer.imputed_store['a1'].loc['2021'].sum()
        assert free_sum != pytest.approx(120.0)

    def test_timeseries_results_unchanged_by_pooling(self, reference_timeseries):
        """I16 — à une entité, le jeu mutualisé est le jeu d'origine."""
        imputer = _fit_with_spy(reference_timeseries)
        for step in imputer.imputation_plan_:
            assert dict(step.training_blocks) == {
                (): 'Q' if step.var_name == 'q1' else 'Y'
            }
            assert step.entities is None
        rows = {step.var_name: len(step.model.fit_y_) for step in imputer.imputation_plan_}
        assert rows == {'a1': 3, 'a2': 3, 'q1': 12}
        store = imputer._covariate_materializer.imputed_store
        for year, total in [('2021', 120.0), ('2022', 132.0), ('2023', 150.0)]:
            assert store['a1'].loc[year].sum() == pytest.approx(total)


class TestMutualizedTrainingSet:
    """I14 et I15 — mutualisation inter-entités et ajustement unique."""

    @staticmethod
    def _variable_steps(imputer: HighFrequencyImputer2) -> list:
        """Étapes de la variable 'v' du jeu PANEL-F."""
        return [step for step in imputer.imputation_plan_ if step.var_name == 'v']

    def test_pooled_training_set_on_panel_f(self, mixed_freq_panel_multifrequency):
        """I14 — 51 lignes (3 FR + 12 DE + 36 IT), toutes à l'échelle de l'étape."""
        imputer = _fit_with_spy(mixed_freq_panel_multifrequency)
        step = self._variable_steps(imputer)[0]
        y_train = step.model.fit_y_

        assert len(y_train) == 51
        counts = pd.Series(
            [key[0] for key in y_train.index]
        ).value_counts().to_dict()
        assert counts == {'IT': 36, 'DE': 12, 'FR': 3}
        assert dict(step.training_blocks) == {('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'}

        # Valeurs d'or du §5.8 : FR et IT à 10.0 / 11.0 / 12.5, DE aux tiers
        france = _entity_block(y_train, 'FR')
        assert [round(value, 6) for value in france] == [10.0, 11.0, 12.5]

        italy = _entity_block(y_train, 'IT')
        assert [round(value, 6) for value in italy.loc['2021']] == [10.0] * 12
        assert [round(value, 6) for value in italy.loc['2022']] == [11.0] * 12
        assert [round(value, 6) for value in italy.loc['2023']] == [12.5] * 12

        germany = _entity_block(y_train, 'DE')
        assert [round(value, 3) for value in germany[:4]] == [
            round(28 / 3, 3), round(30 / 3, 3), round(31 / 3, 3), round(31 / 3, 3)
        ]

        # Aucune ligne ne mêle deux échelles : toutes tiennent dans la plage
        # mensuelle du jeu
        assert y_train.min() > 9.0 and y_train.max() < 14.0

    def test_single_fit_shared_between_source_frequency_groups(
        self, mixed_freq_panel_multifrequency
    ):
        """I15 — un seul ajustement, un seul objet modèle, deux recalages distincts."""
        imputer = _fit_with_spy(mixed_freq_panel_multifrequency)
        steps = self._variable_steps(imputer)

        # Deux étapes de plan, une par groupe de fréquence source
        assert len(steps) == 2
        assert {step.source_frequency for step in steps} == {'Y', 'Q'}
        assert {step.entities for step in steps} == {(('FR',),), (('DE',),)}

        # Un seul ajustement pour la variable, quel que soit le nombre de
        # groupes : 'q1' et 'v' font deux ajustements en tout
        assert _SpyEstimator.n_fits == 2

        # MÊME objet modèle, mêmes features, mêmes blocs, mêmes souillures
        first, second = steps
        assert first.model is second.model
        assert first.feature_cols == second.feature_cols
        assert dict(first.training_blocks) == dict(second.training_blocks)
        assert dict(first.materialization) == dict(second.materialization)
        assert (first.covariate_taint, first.target_taint) == (
            second.covariate_taint, second.target_taint
        )

        # Recalages distincts : total annuel exact pour FR, trimestriel pour DE
        store = imputer._covariate_materializer.imputed_store['v']
        assert _entity_block(store, 'FR').loc['2021'].sum() == pytest.approx(120.0)
        assert _entity_block(store, 'DE').loc['2021-01':'2021-03'].sum() == (
            pytest.approx(28.0)
        )

    def test_contributing_entity_is_never_rewritten(
        self, mixed_freq_panel_multifrequency
    ):
        """IT fournit 36 des 51 lignes d'entraînement et n'est jamais réécrite."""
        imputer = _fit_with_spy(mixed_freq_panel_multifrequency)

        # Aucune étape ne nomme IT parmi ses entités
        for step in self._variable_steps(imputer):
            assert ('IT',) not in (step.entities or ())

        # Ses cellules restent ORIGINAL
        provenance = _entity_block(imputer.imputation_provenance_['v'], 'IT')
        assert {str(value) for value in provenance} == {'original'}

        # Et aucune n'entre dans le miroir : les registres ne portent que ce
        # qui a été IMPUTÉ, et IT ne l'est jamais pour 'v'
        mirrored = _by_entity(imputer._covariate_materializer.imputed_store['v'].index)
        assert set(mirrored) == {('FR',), ('DE',)}

    def test_target_taint_is_computed_by_the_origin_filter(
        self, mixed_freq_panel_multifrequency
    ):
        """Sous False, y_train n'est fait que d'observations : souillure 'none'."""
        imputer = _fit_with_spy(mixed_freq_panel_multifrequency)
        for step in imputer.imputation_plan_:
            assert step.target_taint == 'none'


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
        imputer = _fit_with_spy(mixed_freq_panel_multifrequency)
        step = next(
            step for step in imputer.imputation_plan_ if step.var_name == 'v'
        )
        materializer = imputer._covariate_materializer
        frequencies = imputer._detected_frequencies_by_column()

        # Voie de l'étape : décidée sur la grille de prédiction, mensuelle
        assert step.materialization['q1'] == 'interpolate'
        assert step.materialization['m1'] == 'identity'

        # Dégradation, bloc par bloc, de cette voie unique
        expected = {
            ('FR',): ('Y', 'aggregate'),    # bloc annuel : q1 y est plus fine
            ('DE',): ('Q', 'identity'),     # bloc trimestriel : q1 y est à son pas
            ('IT',): ('M', 'interpolate'),  # bloc mensuel : q1 y est plus basse
        }
        assert dict(step.training_blocks) == {
            entity: freq for entity, (freq, _way) in expected.items()
        }
        for entity, (f_block, way) in expected.items():
            f_col = materializer._column_frequency(frequencies, 'q1', entity)
            assert materializer._applicable_way(
                step.materialization['q1'], f_col, f_block
            ) == way

    def test_the_three_ways_land_on_the_same_scale(
        self, mixed_freq_panel_multifrequency
    ):
        """Les trois voies produisent des valeurs comparables une fois mises à l'échelle.

        C'est la contrepartie de la dégradation : trois transformations
        différentes, un seul niveau. Sans le diviseur fondé sur la période que
        la cellule couvre, l'agrégat annuel de ``q1`` chez FR entrerait dans le
        modèle douze fois plus grand que sa valeur mensuelle chez IT.
        """
        imputer = _fit_with_spy(mixed_freq_panel_multifrequency)
        step = next(
            step for step in imputer.imputation_plan_ if step.var_name == 'v'
        )
        X_train = step.model.fit_X_

        means = {
            entity: float(_entity_block(X_train['q1'], entity).mean())
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
    def _annual_step(imputer: HighFrequencyImputer2):
        """Return the plan step imputing 'a1', whose training block is annual."""
        return next(
            step for step in imputer.imputation_plan_ if step.var_name == 'a1'
        )

    def test_annual_covariate_enters_the_fit_with_its_observations(
        self, reference_timeseries
    ):
        """Sur la grille annuelle, 'a2' vaut ses ancres, pas une interpolation."""
        imputer = _fit_with_spy(reference_timeseries, target_frequency='Q')
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
        imputer = _fit_with_spy(reference_timeseries, target_frequency='Q')
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
        imputer = _fit_with_spy(reference_timeseries, target_frequency='Q')
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
        imputer = _fit_with_spy(
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


class TestFallbackPath:
    """Le repli d'interpolation, et ce qu'il écrit."""

    def test_estimator_failure_falls_back_and_marks_interpolated(
        self, reference_timeseries
    ):
        """D6 — is_fallback=True, cellules INTERPOLATED, registres alimentés."""
        imputer = _make_imputer(estimator=_FailingEstimator())
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
            # « Le repli matérialise » : les trois registres sont alimentés
            assert column in imputer._covariate_materializer.imputed_store
            assert column in imputer._covariate_materializer.origin_store

        # Avertissements AGRÉGÉS : un seul message porte les trois échecs
        degraded = [message for message in messages if 'degraded during the fit' in message]
        assert len(degraded) == 1
        assert degraded[0].count('interpolation fallback') == 3

    def test_estimator_none_interpolates_everything(self, reference_timeseries):
        """estimator=None : chaque variable retombe sur l'interpolation."""
        imputer = _fit_quietly(_make_imputer(estimator=None), reference_timeseries)
        assert all(step.is_fallback for step in imputer.imputation_plan_)
        assert all(
            step.model == INTERPOLATE_FALLBACK for step in imputer.imputation_plan_
        )
        # Les totaux de période restent respectés : le repli est recalé
        store = imputer._covariate_materializer.imputed_store
        assert store['a1'].loc['2022'].sum() == pytest.approx(132.0)


class TestPhaseFiveEdgeCases:
    """Cas limites de l'exécution des étapes."""

    def test_unsorted_index_is_refused_explicitly(self, reference_timeseries):
        """Un index désordonné est refusé par le contrat d'entrée, sans imputer à faux.

        La classe fige ``auto_sort=False`` et ``strict_validation=True``
        ([SPEC] §12.5) : trier en silence changerait les fenêtres et les
        agrégats sans que l'appelant le sache.
        """
        with pytest.raises(ValueError, match='not sorted'):
            _fit_with_spy(reference_timeseries.iloc[::-1])

    def test_duplicated_index_does_not_crash(self, reference_timeseries):
        """Un index dupliqué n'interrompt pas le fit."""
        duplicated = pd.concat([reference_timeseries, reference_timeseries.iloc[[0]]])
        imputer = _make_imputer(estimator=_SpyEstimator())
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
        imputer = _fit_with_spy(data)
        step = next(step for step in imputer.imputation_plan_ if step.var_name == 'a1')
        assert len(step.model.fit_y_) == 2

    def test_all_nan_column_is_not_imputed(self, reference_timeseries):
        """Une colonne entièrement NaN n'est classée nulle part et n'est jamais imputée."""
        data = reference_timeseries.copy()
        data['empty'] = np.nan
        imputer = _fit_with_spy(data)
        assert all(step.var_name != 'empty' for step in imputer.imputation_plan_)
        assert 'empty' not in imputer._covariate_materializer.imputed_store

    def test_incomplete_leading_and_trailing_periods(self, reference_timeseries):
        """Une période que les bornes du jeu tronquent n'est pas recalée (§11.1).

        Deux gardes distinctes se relaient : celle des sous-périodes NaN, et
        celle de la troncature CALENDAIRE, dont les sous-périodes manquantes
        sont absentes de la grille plutôt que vides. Sans la seconde, le total
        annuel de 2021 serait réparti sur les dix mois présents, soit une
        sur-attribution de 20 % à chacun.
        """
        truncated = reference_timeseries.loc['2021-03-31':'2023-08-31']
        imputer = _fit_with_spy(truncated)
        store = imputer._covariate_materializer.imputed_store

        # L'année 2022, complète, reste exacte
        assert store['a1'].loc['2022'].sum() == pytest.approx(132.0)
        # 2021, amputée de deux mois, garde ses prédictions brutes
        assert len(store['a1'].loc['2021']) == 10
        assert store['a1'].loc['2021'].sum() != pytest.approx(120.0)
        # 2023 n'a plus d'ancre dans la trame : la fenêtre stricte s'arrête
        # avant, et aucune de ses lignes n'est imputée
        assert store['a1'].index.max() == pd.Timestamp('2022-12-31')

    def test_single_observation_entity(self, mixed_freq_panel_multifrequency):
        """Une entité réduite à une observation ne fait pas échouer l'étape."""
        data = mixed_freq_panel_multifrequency.copy()
        italy = data.index.get_level_values(0) == 'IT'
        data.loc[italy, 'v'] = np.nan
        data.iloc[np.flatnonzero(italy)[0], data.columns.get_loc('v')] = 10.0
        imputer = _fit_with_spy(data)
        assert len(imputer.imputation_plan_) >= 1

    def test_panel_with_no_imputable_variable_yields_an_empty_plan(self):
        """Toutes les colonnes déjà à la fréquence cible : plan vide, aucune erreur."""
        dates = pd.date_range('2021-01-31', periods=24, freq='ME')
        index = pd.MultiIndex.from_product([['FR', 'DE'], dates], names=['country', 'date'])
        data = pd.DataFrame(
            {'m1': np.arange(48, dtype=float), 'm2': np.arange(48, dtype=float)},
            index=index,
        )
        imputer = _fit_with_spy(data)
        assert len(imputer.imputation_plan_) == 0
        assert imputer.imputation_models_ == {}


# =============================================================================
# Axe 2 — traversée des fréquences intermédiaires (lot L11, §5)
# =============================================================================
# Fonction auxiliaire de capture des contextes d'ajustement de chaque variable
def _capture_variable_fits(data: pd.DataFrame, **overrides) -> tuple:
    """Ajuste un imputeur en retenant le contexte de chaque (étape, variable).

    Le contexte est celui que la PHASE 5c ajuste réellement : la sonde
    d'ordonnancement passe par la même implémentation, seul le dernier appel
    d'une (étape, variable) est donc conservé.

    Returns:
        Couple ``(imputer, fits)``, ``fits`` étant un dict
        ``(label d'étape, colonne) -> _VariableFit``.
    """
    fits = {}
    original = HighFrequencyImputer2._prepare_variable

    def _spy(self, **kwargs):
        fit = original(self, **kwargs)
        label = self._stage_frequency_label(kwargs['stage_freq'])
        fits[(label, kwargs['column'])] = fit
        return fit

    with patch.object(HighFrequencyImputer2, '_prepare_variable', _spy):
        imputer = _fit_with_spy(data, **overrides)
    return imputer, fits


# Fonction auxiliaire de lecture des couples (étape, variable) du plan
def _plan_pairs(imputer: HighFrequencyImputer2) -> list:
    """Rend la liste ordonnée des couples (label d'étape, variable) du plan."""
    return [(step.pred_freq_label, step.var_name) for step in imputer.imputation_plan_.steps]


class TestFrequencyProgression:
    """§5.2 — construction de la progression, identique au fit et au transform."""

    def test_frequency_progression_on_reference_ts(self, reference_timeseries):
        """Les trois modalités sur le jeu TS, valeurs exactes (§5.2)."""
        # F = {Q, Y, M} : sous False, la cible seule
        imputer = _fit_with_spy(reference_timeseries, impute_intermediate_frequencies=False)
        assert imputer.frequency_progression_ == ['M']

        # Sous les deux autres modalités, LE MÊME plan : Y, la plus basse
        # fréquence, n'est pas une étape — rien n'est à y imputer
        for modality in ('covariates_only', True):
            imputer = _fit_with_spy(
                reference_timeseries, impute_intermediate_frequencies=modality
            )
            assert imputer.frequency_progression_ == ['Q', 'M']

    def test_imputable_variables_of_each_stage_on_reference_ts(self, reference_timeseries):
        """Étape Q : {a1, a2} ; étape M : {q1, a1, a2} (§5.2, point d)."""
        imputer = _fit_with_spy(
            reference_timeseries, impute_intermediate_frequencies='covariates_only'
        )
        assert {column for column, _ in imputer._imputable_groups('Q')} == {'a1', 'a2'}
        assert {column for column, _ in imputer._imputable_groups('M')} == {'q1', 'a1', 'a2'}

    def test_progression_on_panel_f_uses_per_entity_frequencies(
        self, mixed_freq_panel_multifrequency
    ):
        """Sur PANEL-F, F est lu par couple (entité, colonne) (§2.5, §5.2)."""
        imputer = _fit_with_spy(
            mixed_freq_panel_multifrequency,
            impute_intermediate_frequencies='covariates_only',
        )
        # v est annuelle pour FR, trimestrielle pour DE, mensuelle pour IT :
        # une lecture par colonne n'aurait vu qu'une fréquence et manqué Q
        assert [
            imputer._stage_frequency_label(stage)
            for stage in imputer.frequency_progression_
        ] == ['Q', 'M']

        # Étape Q : v n'est imputable que pour FR
        assert imputer._imputable_groups({FR: 'Q', DE: 'Q', IT: 'Q'})[('v', 'Y')] == (FR,)
        assert ('v', 'Q') not in imputer._imputable_groups({FR: 'Q', DE: 'Q', IT: 'Q'})
        # Étape M : v est imputable pour FR et DE, jamais pour IT
        at_month = imputer._imputable_groups({FR: 'M', DE: 'M', IT: 'M'})
        assert at_month[('v', 'Y')] == (FR,)
        assert at_month[('v', 'Q')] == (DE,)
        # IT observe déjà v mensuellement : elle n'est imputable à aucune étape
        for (column, _source), entities in at_month.items():
            if column == 'v':
                assert IT not in entities

    def test_progression_per_target_frequency_group_on_panel(
        self, mixed_freq_panel_multifrequency
    ):
        """Un dict de cibles produit une progression par groupe, fusionnée (§5.2)."""
        imputer = _fit_with_spy(
            mixed_freq_panel_multifrequency,
            target_frequency={FR: 'M', DE: 'Q', IT: 'M'},
            impute_intermediate_frequencies=True,
        )
        # DE a Q pour cible : son groupe s'arrête à l'étape Q, dont il est absent
        assert imputer.frequency_progression_ == [
            {FR: 'Q', DE: 'Q', IT: 'Q'},
            {FR: 'M', IT: 'M'},
        ]
        # Aucune étape mensuelle n'écrit pour DE
        for step in imputer.imputation_plan_.steps:
            if step.pred_freq_label == 'M':
                assert DE not in (step.entities or ())


class TestStagePlanAxis2:
    """§5.5 — le plan d'étapes complet sur le jeu TS."""

    def test_stage_plan_of_spec_5_5(self, reference_timeseries):
        """2 étapes, 5 modèles, y_train filtré par l'origine (§5.5)."""
        expected_pairs = [
            ('Q', 'a1'), ('Q', 'a2'), ('M', 'a1'), ('M', 'a2'), ('M', 'q1'),
        ]
        # Sous 'covariates_only', y_train tient les 3 ancres partout : le
        # filtre est celui de False, seul le plan diffère
        imputer, fits = _capture_variable_fits(
            reference_timeseries,
            covariate_strategy='model',
            fit_predict_order='frequency',
            impute_intermediate_frequencies='covariates_only',
        )
        assert imputer.frequency_progression_ == ['Q', 'M']
        assert _plan_pairs(imputer) == expected_pairs
        assert _SpyEstimator.n_fits == 5
        for stage in ('Q', 'M'):
            for column in ('a1', 'a2'):
                assert len(fits[(stage, column)].y_train) == 3
        assert len(fits[('M', 'q1')].y_train) == 12

        # Sous True, les étapes M de a1 et a2 gagnent leurs imputations Q.
        # Douze lignes et non quinze : la quatrième imputation trimestrielle
        # de chaque année tombe sur l'ancre annuelle, où l'observation gagne
        imputer, fits = _capture_variable_fits(
            reference_timeseries,
            covariate_strategy='model',
            fit_predict_order='frequency',
            impute_intermediate_frequencies=True,
        )
        assert _plan_pairs(imputer) == expected_pairs
        assert _SpyEstimator.n_fits == 5
        for column in ('a1', 'a2'):
            assert len(fits[('Q', column)].y_train) == 3
            training = fits[('M', column)].training
            assert len(fits[('M', column)].y_train) == 12
            assert sorted(training.row_origin.value_counts().to_dict().items()) == [
                ('model', 9), ('observed', 3)
            ]

    def test_carried_model_rank_reached_under_covariates_only(self, reference_timeseries):
        """Le rang 3 — report d'étape — devient atteignable (§4.4, §5.6)."""
        imputer = _fit_with_spy(
            reference_timeseries,
            covariate_strategy='model',
            impute_intermediate_frequencies='covariates_only',
        )
        ways = [
            way
            for step in imputer.imputation_plan_.steps
            for way in step.materialization.values()
        ]
        assert 'carried_model' in ways
        # Sous False, aucune étape antérieure : le rang 3 reste hors d'atteinte
        without = _fit_with_spy(
            reference_timeseries,
            covariate_strategy='model',
            impute_intermediate_frequencies=False,
        )
        assert 'carried_model' not in [
            way
            for step in without.imputation_plan_.steps
            for way in step.materialization.values()
        ]


class TestOriginFilter:
    """§5.3 — le filtre d'origine de y_train, et le piège D12."""

    def test_covariates_only_differs_from_true(self, reference_timeseries):
        """I12 — même plan, filtre différent, valeurs différentes (§5.1)."""
        # Sous 'model', aucune ligne d'origine 'model' n'entre dans y_train
        _, covariates_only = _capture_variable_fits(
            reference_timeseries,
            covariate_strategy='model',
            impute_intermediate_frequencies='covariates_only',
        )
        for fit in covariates_only.values():
            assert set(fit.training.row_origin.unique()) <= {'observed'}

        # ... et les valeurs finales diffèrent de celles de True
        _, cascaded = _capture_variable_fits(
            reference_timeseries,
            covariate_strategy='model',
            impute_intermediate_frequencies=True,
        )
        assert 'model' in set(cascaded[('M', 'a1')].training.row_origin.unique())
        assert not np.allclose(
            covariates_only[('M', 'a1')].y_train.to_numpy()[:3],
            cascaded[('M', 'a1')].y_train.to_numpy()[:3],
        ) or len(cascaded[('M', 'a1')].y_train) != len(
            covariates_only[('M', 'a1')].y_train
        )

        # Sous 'interpolate', les rangs 2 et 3 sont hors d'atteinte :
        # 'covariates_only' rend exactement les valeurs de False
        store_false = _fit_with_spy(
            reference_timeseries,
            covariate_strategy='interpolate',
            impute_intermediate_frequencies=False,
        )._covariate_materializer.imputed_store
        store_only = _fit_with_spy(
            reference_timeseries,
            covariate_strategy='interpolate',
            impute_intermediate_frequencies='covariates_only',
        )._covariate_materializer.imputed_store
        for column in ('a1', 'a2', 'q1'):
            # "check_freq=False" : l'attribut de fréquence de l'index diffère
            # après deux étapes, les valeurs sont ce qui est comparé
            pd.testing.assert_series_equal(
                store_false[column],
                store_only[column],
                check_names=False,
                check_freq=False,
            )

    def test_y_train_filter_reads_origin_store_not_provenance(self, reference_timeseries):
        """D12 — le filtre lit origin_store, jamais la provenance publique."""
        imputer = _fit_with_spy(reference_timeseries)
        materializer = imputer._covariate_materializer
        materializer.reset()
        builder = imputer._training_set_builder
        frequencies = imputer._detected_frequencies_by_column()

        # Deux cellules de MÊME provenance publique — toutes deux produites
        # par une même étape trimestrielle — mais d'origines différentes
        dates = pd.to_datetime(['2021-03-31', '2021-06-30', '2021-09-30'])
        materializer.record_production(
            'a1',
            pd.Series([30.0, 31.0, 32.0], index=dates),
            pd.Series(['observed', 'model', 'interpolated'], index=dates),
            pd.Series(['Q', 'Q', 'Q'], index=dates),
        )

        def _origins(modality):
            training = builder.build(
                column='a1',
                feature_cols=(),
                stage_freq='M',
                detected_frequencies=frequencies,
                source_data=reference_timeseries,
                eligible_origins=ELIGIBLE_ORIGINS[modality],
            )
            return training.row_origin.reindex(dates).dropna().to_list()

        # Sous 'covariates_only', seule la cellule observée entre ; la cellule
        # de repli 'interpolated' est exclue au même titre que celle de modèle
        assert _origins('covariates_only') == ['observed']
        # Sous True, les trois entrent
        assert sorted(_origins(True)) == ['interpolated', 'model', 'observed']

    def test_target_taint_families(self, reference_timeseries):
        """I6 — les deux familles de souillure de cible n'existent que sous True."""
        tainted = {
            ProvenanceType.MODEL_ON_IMPUTED_TARGET,
            ProvenanceType.MODEL_ON_IMPUTED_BOTH,
        }
        for modality in (False, 'covariates_only'):
            imputer = _fit_with_spy(
                reference_timeseries,
                covariate_strategy='model',
                impute_intermediate_frequencies=modality,
            )
            for step in imputer.imputation_plan_.steps:
                assert step.target_taint == 'none'
                assert step.emitted_provenance not in tainted

        imputer = _fit_with_spy(
            reference_timeseries,
            covariate_strategy='model',
            impute_intermediate_frequencies=True,
        )
        emitted = {step.emitted_provenance for step in imputer.imputation_plan_.steps}
        assert emitted & tainted



class TestCoincidentCells:
    """§5.9 et D32 — cellules coïncidentes et niveau de fréquence de l'index."""

    # Fabrique privée : les paramètres communs des mesures du §5.9
    @staticmethod
    def _cascade(data, constraint, modality=True, **overrides):
        """Ajuste sur le jeu TS sous l'axe 2, et rend les contextes captés."""
        return _capture_variable_fits(
            data,
            covariate_strategy='model',
            fit_predict_order='frequency',
            impute_intermediate_frequencies=modality,
            aggregation_constraint=constraint,
            **overrides,
        )

    # Fabrique privée : la même capture, sous un estimateur RÉEL
    @staticmethod
    def _capture_with_regression(data, **overrides):
        """Capte les contextes d'un fit mené par une régression linéaire.

        Les valeurs mesurées du §5.9 supposent un modèle réel : l'estimateur
        espion des autres tests ne prédit rien d'exploitable.
        """
        fits = {}
        original = HighFrequencyImputer2._prepare_variable

        def _spy(self, **kwargs):
            fit = original(self, **kwargs)
            label = self._stage_frequency_label(kwargs['stage_freq'])
            fits[(label, kwargs['column'])] = fit
            return fit

        with patch.object(HighFrequencyImputer2, '_prepare_variable', _spy):
            _fit_quietly(_make_imputer(**overrides), data)
        return fits

    def test_coincident_cells_kept_only_without_constraint(self, reference_timeseries):
        """I20 — 12 lignes sous 'sum', 15 sous None, à l'étape M (§5.9)."""
        _, under_sum = self._cascade(reference_timeseries, 'sum')
        _, under_none = self._cascade(reference_timeseries, None)
        for column in ('a1', 'a2'):
            # Trois ancres annuelles + neuf imputations Q : la quatrième de
            # chaque année tombe sur l'ancre, l'observation gagne
            assert len(under_sum[('M', column)].y_train) == 12
            # Aucun recalage : les quatre imputations de chaque année entrent
            assert len(under_none[('M', column)].y_train) == 15

    def test_collinearity_is_exact_under_sum(self, reference_timeseries):
        """La règle suit la structure algébrique du jeu, pas une préférence."""
        gold = {2021: 120.0, 2022: 132.0, 2023: 150.0}

        # Sortie de l'étape Q : les quatre imputations trimestrielles de a1
        def _totals(constraint):
            imputer = _fit_with_spy(
                reference_timeseries,
                target_frequency='Q',
                covariate_strategy='model',
                impute_intermediate_frequencies=True,
                aggregation_constraint=constraint,
            )
            produced = imputer._covariate_materializer.imputed_store['a1']
            years = pd.DatetimeIndex(produced.index.get_level_values(-1)).year
            return {year: produced[years == year].sum() for year in gold}

        # Sous 'sum', le recalage impose Somme(sous-périodes) = total observé :
        # la ligne annuelle EST la somme des quatre trimestrielles
        exact = _totals('sum')
        for year, anchor in gold.items():
            assert exact[year] == pytest.approx(anchor, abs=1e-9)

        # Sous None, la colinéarité est rompue (écarts -0.65 / +2.37 / -0.60)
        free = _totals(None)
        assert any(abs(free[year] - anchor) > 0.1 for year, anchor in gold.items())

    def test_training_index_gains_a_frequency_level(self, reference_timeseries):
        """Sous None, l'index porte la fréquence ; sous 'sum', il est inchangé."""
        _, under_sum = self._cascade(reference_timeseries, 'sum')
        _, under_none = self._cascade(reference_timeseries, None)
        stamped = under_none[('M', 'a1')].training
        plain = under_sum[('M', 'a1')].training

        # Sous 'sum', l'index reste STRICTEMENT celui d'aujourd'hui : les
        # douze fins de trimestre, sans niveau ajoute
        expected = pd.DatetimeIndex(
            [
                date
                for date in pd.date_range('2021-01-31', '2023-12-31', freq='ME')
                if date.month in (3, 6, 9, 12)
            ],
            name='date',
        )
        assert not plain.has_frequency_level
        assert not isinstance(plain.X.index, pd.MultiIndex)
        assert plain.X.index.equals(expected)

        # Sous None, un niveau de plus, nommé 'frequency' et placé du côté de
        # l'entité — le bloc devient le couple (entité, fréquence)
        assert stamped.has_frequency_level
        assert list(stamped.X.index.names) == ['frequency', 'date']
        assert stamped.X.index.nlevels == plain.X.index.nlevels + 1

        # Les deux cellules du 2021-12-31 coexistent
        anchor = pd.Timestamp('2021-12-31')
        assert ('Y', anchor) in stamped.X.index
        assert ('Q', anchor) in stamped.X.index

        # Leurs diviseurs valent 12 (l'année) et 3 (le trimestre)
        raw = stamped.y
        scaled = under_none[('M', 'a1')].y_train
        assert raw[('Y', anchor)] / scaled[('Y', anchor)] == pytest.approx(12.0)
        assert raw[('Q', anchor)] / scaled[('Q', anchor)] == pytest.approx(3.0)

    def test_scaled_target_stays_homogeneous_across_levels(self, reference_timeseries):
        """Les deux cellules du 2021-12-31 valent ~10.0 de part et d'autre."""
        fits = self._capture_with_regression(
            reference_timeseries,
            covariate_strategy='model',
            fit_predict_order='frequency',
            impute_intermediate_frequencies=True,
            aggregation_constraint=None,
        )
        scaled = fits[('M', 'a1')].y_train
        anchor = pd.Timestamp('2021-12-31')
        # 120 / 12 d'un côté, ~30 / 3 de l'autre : une seule échelle mensuelle
        assert scaled[('Y', anchor)] == pytest.approx(10.0)
        assert scaled[('Q', anchor)] == pytest.approx(10.0, rel=0.1)

    @pytest.mark.parametrize('modality', [False, 'covariates_only'])
    def test_constraint_is_inert_on_y_train_without_axis_2(
        self, reference_timeseries, modality
    ):
        """Hors axe 2, aucune coïncidence n'est possible : l'effet est nul."""
        _, under_sum = self._cascade(reference_timeseries, 'sum', modality=modality)
        _, under_none = self._cascade(reference_timeseries, None, modality=modality)
        assert set(under_sum) == set(under_none)
        for key, fit in under_sum.items():
            pd.testing.assert_series_equal(fit.y_train, under_none[key].y_train)

    def test_per_column_constraint_is_read_per_column(self, reference_timeseries):
        """La forme dictionnaire est lue PAR COLONNE, jamais globalement."""
        _, fits = self._cascade(
            reference_timeseries, {'a1': None, '__default__': 'sum'}
        )
        # a1 garde ses cellules coïncidentes, a2 les perd
        assert len(fits[('M', 'a1')].y_train) == 15
        assert fits[('M', 'a1')].training.has_frequency_level
        assert len(fits[('M', 'a2')].y_train) == 12
        assert not fits[('M', 'a2')].training.has_frequency_level

class TestPerRowScale:
    """§5.4 — le diviseur d'échelle est par ligne, jamais par étape."""

    def test_per_row_scale_factor_on_mixed_frequency_y_train(self):
        """Le tableau chiffré du §5.4 : 120/Y, 28/Q, 30/Q à l'étape M."""
        scaler = StageScaler(scale_features='constant')
        index = pd.to_datetime(['2021-12-31', '2021-03-31', '2021-06-30'])
        produced = pd.Series(['Y', 'Q', 'Q'], index=index)
        divisors = scaler.target_divisor(
            'a1', source_freq='Y', pred_freq='M', index=index, produced_freq=produced
        )
        # Le diviseur est PAR LIGNE : le scalaire de l'étape ne s'applique pas
        assert isinstance(divisors, pd.Series)
        assert divisors.to_list() == pytest.approx([12.0, 3.0, 3.0])
        scaled = scaler.apply(pd.Series([120.0, 28.0, 30.0], index=index), divisors)
        assert scaled.to_list() == pytest.approx([10.0, 9.3333333, 10.0])

    def test_unit_divisors_are_not_short_circuited(self):
        """B12 — une Series valant 1.0 partout reste une Series."""
        scaler = StageScaler(scale_features='constant')
        index = pd.to_datetime(['2021-01-31', '2021-02-28'])
        divisors = scaler.target_divisor(
            'a1',
            source_freq='M',
            pred_freq='M',
            index=index,
            produced_freq=pd.Series(['M', 'M'], index=index),
        )
        assert isinstance(divisors, pd.Series)
        assert divisors.to_list() == pytest.approx([1.0, 1.0])

    def test_row_frequency_mixes_block_and_store_sources(
        self, mixed_freq_panel_multifrequency
    ):
        """Les deux sources de la fréquence de ligne coexistent (§5.4, §5.8)."""
        _, fits = _capture_variable_fits(
            mixed_freq_panel_multifrequency,
            covariate_strategy='interpolate',
            impute_intermediate_frequencies=True,
        )
        training = fits[('M', 'v')].training
        # Fréquences de BLOC : Y pour FR, Q pour DE, M pour IT
        assert dict(training.blocks) == {FR: 'Y', DE: 'Q', IT: 'M'}
        by_entity = {}
        for entity, frequency, origin in zip(
            [tuple(key[:-1]) for key in training.row_frequency.index],
            training.row_frequency,
            training.row_origin,
        ):
            by_entity.setdefault(entity, set()).add((frequency, origin))
        # FR porte ses 3 ancres annuelles ET ses imputations trimestrielles
        assert ('Y', 'observed') in by_entity[FR]
        assert ('Q', 'model') in by_entity[FR]
        # DE et IT n'apportent que des observations, à leur fréquence propre
        assert by_entity[DE] == {('Q', 'observed')}
        assert by_entity[IT] == {('M', 'observed')}

        # La cible mise à l'échelle reste homogène : le diviseur fractionnaire
        # du bloc IT (1/1 à l'étape M) n'est pas planchéré, celui de FR vaut 12
        scaled = fits[('M', 'v')].y_train
        for year, level in ((2021, 10.0), (2022, 11.0), (2023, 12.5)):
            values = scaled[
                scaled.index.get_level_values(-1).year == year
            ].to_numpy()
            assert values.mean() == pytest.approx(level, rel=0.2)

    def test_target_taint_of_a_contributing_entity_propagates(
        self, mixed_freq_panel_multifrequency
    ):
        """§5.8 — une cellule 'model' d'une entité dégrade toute l'étape."""
        imputer = _fit_with_spy(
            mixed_freq_panel_multifrequency,
            covariate_strategy='interpolate',
            impute_intermediate_frequencies=True,
        )
        monthly = [
            step
            for step in imputer.imputation_plan_.steps
            if step.pred_freq_label == 'M' and step.var_name == 'v'
        ]
        assert monthly
        # Les imputations trimestrielles de FR souillent la cible de l'étape,
        # donc la provenance des cellules produites pour DE aussi
        for step in monthly:
            assert step.target_taint == 'imputed'
        assert any(DE in (step.entities or ()) for step in monthly)
