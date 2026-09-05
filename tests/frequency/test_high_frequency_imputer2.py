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

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# Objets testés
from tsforecast.frequency.high_frequency_imputer2 import HighFrequencyImputer2

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

        # PHASE 5 non livrée par ce lot : le plan est vide
        assert len(imputer.imputation_plan_) == 0
        assert imputer.imputation_models_ == {}

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

    def test_intermediate_frequencies_not_implemented_yet(self, reference_timeseries):
        """Les deux modalités de l'axe 2 non livrées annoncent le lot L11."""
        for modality in ('covariates_only', True):
            imputer = _make_imputer(impute_intermediate_frequencies=modality)
            with pytest.raises(NotImplementedError) as excinfo:
                _fit_quietly(imputer, reference_timeseries)
            assert 'L11' in str(excinfo.value)

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
