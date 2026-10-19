"""Parameter space of ``HighFrequencyImputer.__init__``.

Covers, parameter by parameter, the admissible values (stored untouched),
the rejected values and their messages, the per-feature dict forms and
their unknown keys, the inert combinations documented and
never warned about, the defaults of the signature against the documented ones, and a
minimal observable effect of the admitted values not exercised elsewhere
(``verbose``, ``additive_transformer``, ``on_frequency_mismatch``,
``covariate_eligibility``).

``sklearn.utils.estimator_checks.check_parameters_default_constructible`` is
not applicable as such: ``target_frequency`` is required by design and the
class declares no ``_required_parameters``. Its contract is checked by hand in
:class:`TestInitDoesNotTransform`.
"""

# Modules de base
import inspect
import warnings
from typing import get_args
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Objets testés
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer
from tsforecast.frequency.imputation_window import ImputationScope, TrainingScope

# Outils de test partagés
from tests.support.estimators import SpyEstimator
from tests.unit.frequency.high_frequency_imputer.helpers import (
    fit_quietly,
    fit_transform_quietly,
    fit_with_spy,
    make_imputer,
)


# =============================================================================
# Validations d'__init__
# =============================================================================
class TestParameterValidation:
    """Validation ligne à ligne des paramètres, sans transformation."""

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
            make_imputer(**{param: value})
        message = str(excinfo.value)
        assert param in message
        for admissible in expected_values:
            assert repr(admissible) in message or admissible in message

    def test_training_scope_admits_unrestricted_but_imputation_scope_does_not(self):
        """Seul training_scope admet 'unrestricted' : les deux Literal diffèrent."""
        # Accepté côté entraînement
        make_imputer(training_scope='unrestricted')
        # Refusé côté prédiction, le collaborateur ne le connaissant pas
        with pytest.raises(ValueError):
            make_imputer(imputation_scope='unrestricted')

    def test_target_frequency_wrong_type_raises(self):
        """Une fréquence cible ni str ni dict lève TypeError."""
        with pytest.raises(TypeError, match='target_frequency'):
            make_imputer(target_frequency=12)

    def test_target_frequency_empty_dict_raises(self):
        """Un dictionnaire de fréquences cibles vide est refusé."""
        with pytest.raises(ValueError, match='cannot be empty'):
            make_imputer(target_frequency={})

    def test_estimator_without_predict_raises(self):
        """Un estimateur sans 'predict' est refusé, le message nommant la méthode."""
        class _NoPredict:
            def fit(self, X, y):
                return self

        with pytest.raises(ValueError, match="'predict' method"):
            make_imputer(estimator=_NoPredict())

    def test_estimator_dict_admits_default_key(self):
        """La clé '__default__' est admise dans la forme dictionnaire."""
        make_imputer(estimator={'__default__': LinearRegression()})

    def test_estimator_empty_dict_raises(self):
        """Un dictionnaire d'estimateurs vide est refusé."""
        with pytest.raises(ValueError, match='cannot be empty'):
            make_imputer(estimator={})

    def test_additive_transformer_requires_fit_transform_and_inverse(self):
        """Le transformateur additif doit exposer fit_transform ET inverse_transform."""
        class _OnlyFitTransform:
            def fit_transform(self, X, y=None):
                return X

        with pytest.raises(ValueError) as excinfo:
            make_imputer(additive_transformer=_OnlyFitTransform())
        assert 'inverse_transform' in str(excinfo.value)

    @pytest.mark.parametrize('value', ['yes', 0, 1, None, 'covariates'])
    def test_impute_intermediate_frequencies_rejects_other_values(self, value):
        """Toute valeur hors des trois modalités est refusée, 0 et 1 compris."""
        with pytest.raises(ValueError, match='impute_intermediate_frequencies'):
            make_imputer(impute_intermediate_frequencies=value)

    def test_interpolation_method_dict_of_str_accepted(self):
        """La forme dictionnaire de la méthode d'interpolation est admise."""
        make_imputer(interpolation_method={'a1': 'linear', 'q1': 'cubic'})

    def test_interpolation_method_wrong_type_raises(self):
        """Une méthode d'interpolation non textuelle est refusée."""
        with pytest.raises((ValueError, TypeError)):
            make_imputer(interpolation_method=3)

    @pytest.mark.parametrize('value', [-0.5, 1.5, {'a1': 2.0}])
    def test_interpolation_anchor_outside_unit_interval_raises(self, value):
        """Un ancrage hors de [0, 1] est refusé, forme dictionnaire comprise."""
        with pytest.raises((ValueError, TypeError)):
            make_imputer(interpolation_anchor=value)

    def test_interpolation_anchor_accepts_none_float_and_dict(self):
        """None, un float de [0, 1] et un dict de ces valeurs sont admis."""
        make_imputer(interpolation_anchor=None)
        make_imputer(interpolation_anchor=0.5)
        make_imputer(interpolation_anchor={'a1': 0.0, 'q1': None})

    @pytest.mark.parametrize('value', [1, 'five', 3.5])
    def test_cv_invalid_forms_raise(self, value):
        """Un cv entier < 2, ou d'un type inattendu, est refusé."""
        with pytest.raises(ValueError, match='cv'):
            make_imputer(cv=value)

    def test_cv_accepts_none_int_splitter_and_iterable(self):
        """Les quatre formes admises de cv passent la validation."""
        make_imputer(cv=None)
        make_imputer(cv=5)
        make_imputer(cv=KFold(n_splits=3))
        make_imputer(cv=[(np.array([0, 1]), np.array([2]))])

    def test_cv_is_not_resolved_at_init(self):
        """check_cv n'est pas appelé à l'__init__ : cv reste la valeur reçue (B3)."""
        imputer = make_imputer(cv=4)
        assert imputer.cv == 4
        assert not hasattr(imputer, 'cv_')

    @pytest.mark.parametrize('value', [3, ['a']])
    def test_cv_scoring_must_be_str_or_callable(self, value):
        """Un score de validation croisée ni textuel ni appelable est refusé."""
        with pytest.raises(ValueError, match='cv_scoring'):
            make_imputer(cv_scoring=value)

    def test_min_cv_train_size_below_one_raises(self):
        """min_cv_train_size doit valoir au moins 1."""
        with pytest.raises(ValueError, match='min_cv_train_size'):
            make_imputer(min_cv_train_size=0)

    def test_min_cv_train_size_one_is_accepted(self):
        """La borne basse est 1, et non 2 comme dans hfi."""
        assert make_imputer(min_cv_train_size=1).min_cv_train_size == 1

    def test_min_cv_train_size_wrong_type_raises(self):
        """min_cv_train_size doit être un entier."""
        with pytest.raises(TypeError, match='min_cv_train_size'):
            make_imputer(min_cv_train_size=2.5)

    @pytest.mark.parametrize('param', ['coverage_threshold', 'training_coverage_threshold'])
    @pytest.mark.parametrize('value', [-0.1, 1.1])
    def test_coverage_thresholds_outside_unit_interval_raise(self, param, value):
        """Les deux seuils de couverture doivent tomber dans [0, 1]."""
        with pytest.raises(ValueError, match=param):
            make_imputer(**{param: value})

    def test_training_coverage_threshold_admits_none(self):
        """Seul le seuil d'entraînement admet None, où il suit celui de prédiction."""
        assert make_imputer(training_coverage_threshold=None).training_coverage_threshold is None

    @pytest.mark.parametrize('value', ['mean', 'minmax', {'a1': 'zscore'}])
    def test_scale_features_invalid_values_raise(self, value):
        """scale_features n'admet que False, 'constant', 'calendar' ou leur dict."""
        with pytest.raises(ValueError, match='scale_features'):
            make_imputer(scale_features=value)

    def test_scale_features_accepts_the_four_forms(self):
        """Les quatre formes admises de scale_features passent la validation."""
        make_imputer(scale_features=False)
        make_imputer(scale_features='constant')
        make_imputer(scale_features='calendar')
        make_imputer(scale_features={'a1': 'calendar', 'q1': False})

    @pytest.mark.parametrize('value', ['mean', 'last'])
    def test_aggregation_constraint_refuses_mean_and_last(self, value):
        """'mean' et 'last' sont retirés de l'API, le message nommant l'échappatoire."""
        with pytest.raises(ValueError) as excinfo:
            make_imputer(aggregation_constraint=value)
        message = str(excinfo.value)
        assert 'additive_transformer' in message
        assert 'sum' in message

    def test_aggregation_constraint_accepts_sum_none_and_dict(self):
        """'sum', None et le dict de ces deux valeurs sont admis, clé '__default__' comprise."""
        make_imputer(aggregation_constraint='sum')
        make_imputer(aggregation_constraint=None)
        make_imputer(aggregation_constraint={'a1': None, '__default__': 'sum'})

    @pytest.mark.parametrize(
        'param', ['keep_lower_frequencies', 'restore_original_values', 'verbose',
                  'impute_unobserved_entities', 'target_as_covariate']
    )
    @pytest.mark.parametrize('value', ['yes', 1, None])
    def test_booleans_are_validated_as_a_group(self, param, value):
        """Les cinq booléens sont validés ensemble, un entier ne passant pas pour un bool."""
        with pytest.raises(TypeError, match=param):
            make_imputer(**{param: value})


# =============================================================================
# Valeurs admises, stockées telles que reçues
# =============================================================================
# Doublures minimales : un callable de score et deux formes de "cv"
def _custom_scorer(estimator, X, y):
    """Score callable admitted by ``cv_scoring``."""
    return 0.0


_SPLITS = [(np.array([0, 1]), np.array([2]))]
_LOG_TRANSFORMER = FunctionTransformer(np.log1p, np.expm1)
_NAN_TOLERANT = Pipeline([('fill', SimpleImputer()), ('model', LinearRegression())])

# Une ligne par (paramètre, forme admise). Les littéraux de fenêtre
# sont lus sur les types publics du module de fenêtre, pour ne jamais diverger
# de ce que le collaborateur accepte
_ADMITTED = (
    [('target_frequency', value) for value in ('M', 'Q', 'monthly')]
    + [
        ('target_frequency', {'FR': 'M', 'DE': 'Q'}),
        ('target_frequency', {('FR',): 'M', ('DE',): 'M'}),
        ('estimator', None),
        ('estimator', LinearRegression()),
        ('estimator', _NAN_TOLERANT),
        ('estimator', {'a1': LinearRegression(), '__default__': LinearRegression()}),
        ('additive_transformer', None),
        ('additive_transformer', _LOG_TRANSFORMER),
    ]
    + [('covariate_strategy', value) for value in ('tolerate_nan', 'interpolate', 'model')]
    + [('covariate_fallback', value) for value in ('interpolate', 'tolerate_nan')]
    + [('covariate_eligibility', value) for value in ('any_entity', 'all_entities')]
    + [('interpolation_method', value) for value in ('linear', 'time', 'cubic')]
    + [('interpolation_method', {'a1': 'cubic', '__default__': 'linear'})]
    + [('interpolation_anchor', value) for value in (None, 0.0, 0.5, 1.0)]
    + [('interpolation_anchor', {'a1': 0.0, 'q1': None, '__default__': 1.0})]
    + [('impute_intermediate_frequencies', value) for value in (False, 'covariates_only', True)]
    + [('impute_unobserved_entities', value) for value in (False, True)]
    + [('fit_predict_order', value) for value in ('frequency', 'cv')]
    + [('cv', value) for value in (None, 2, 5, np.int64(3), KFold(n_splits=3), _SPLITS)]
    + [('cv_scoring', value) for value in ('neg_mean_absolute_error', _custom_scorer)]
    + [('min_cv_train_size', value) for value in (1, 10, np.int64(3))]
    + [('imputation_scope', value) for value in get_args(ImputationScope)]
    + [('coverage_threshold', value) for value in (0.0, 0.5, 1.0, 0, 1, np.float32(0.5))]
    + [('training_scope', value) for value in (None,) + get_args(TrainingScope)]
    + [('training_coverage_threshold', value) for value in (None, 0.0, 1.0)]
    + [('scale_features', value) for value in (False, 'constant', 'calendar')]
    + [('scale_features', {'a1': 'calendar', 'q1': False, '__default__': 'constant'})]
    + [('aggregation_constraint', value) for value in ('sum', None)]
    + [('aggregation_constraint', {'a1': None, '__default__': 'sum'})]
    + [(name, value)
       for name in ('keep_lower_frequencies', 'restore_original_values', 'verbose')
       for value in (False, True, np.True_, np.False_)]
    + [('impute_unobserved_entities', value) for value in (np.True_, np.False_)]
    + [('target_as_covariate', value) for value in (False, True, np.True_, np.False_)]
    + [('on_frequency_mismatch', value) for value in ('error', 'warn')]
    + [('time_col', value) for value in (None, 'date')]
    + [('panel_cols', value) for value in (None, ['country'])]
)


class TestAdmittedValues:
    """Every admissible form of every parameter passes and is stored untouched."""

    @pytest.mark.parametrize(
        'param, value', _ADMITTED,
        ids=[f'{name}-{index}' for index, (name, _value) in enumerate(_ADMITTED)],
    )
    def test_admitted_value_is_stored_as_received(self, param, value):
        """``get_params`` returns the very object handed to ``__init__`` (B3)."""
        imputer = make_imputer(**{param: value})
        # Identité, et non égalité : aucune normalisation, aucune copie
        assert imputer.get_params()[param] is value

    def test_every_init_parameter_is_covered(self):
        """The admitted-values table names every parameter of the signature."""
        signature = inspect.signature(HighFrequencyImputer.__init__)
        names = {name for name in signature.parameters if name != 'self'}
        assert {name for name, _value in _ADMITTED} == names


# =============================================================================
# Valeurs refusées et messages
# =============================================================================
# Doublures incomplètes : une méthode requise manque à chacune
class _PredictOnly:
    """Object exposing ``predict`` but no ``fit``."""

    def predict(self, X):
        return X


class _FitOnly:
    """Object exposing ``fit`` but no ``predict``."""

    def fit(self, X, y):
        return self


class _InverseOnly:
    """Object exposing ``inverse_transform`` but no ``fit_transform``."""

    def inverse_transform(self, X):
        return X


class _SplitOnly:
    """Object exposing ``split`` but no ``get_n_splits``."""

    def split(self, X, y=None, groups=None):
        yield np.array([0]), np.array([1])


# (paramètre, valeur, exception, fragment attendu du message)
_REJECTED = [
    ('target_frequency', '', ValueError, 'target_frequency'),
    ('target_frequency', 'nonsense', ValueError, 'target_frequency'),
    ('target_frequency', {'FR': 3}, ValueError, "'FR'"),
    ('target_frequency', {'FR': 'nonsense'}, ValueError, 'target_frequency'),
    ('estimator', _PredictOnly(), ValueError, "'fit' method"),
    ('estimator', {'a1': _PredictOnly()}, ValueError, "'a1'"),
    ('estimator', {'a1': _FitOnly()}, ValueError, "'predict' method"),
    ('additive_transformer', _InverseOnly(), ValueError, 'Missing: fit_transform'),
    ('interpolation_method', '', ValueError, 'interpolation_method'),
    ('interpolation_method', {'a1': 3}, ValueError, 'interpolation_method'),
    ('interpolation_anchor', 'x', ValueError, 'interpolation_anchor'),
    ('interpolation_anchor', True, ValueError, 'interpolation_anchor'),
    ('cv', 0, ValueError, 'cv'),
    ('cv', -1, ValueError, 'cv'),
    ('cv', True, ValueError, 'cv'),
    ('cv', _SplitOnly(), ValueError, 'get_n_splits'),
    ('cv_scoring', None, ValueError, 'cv_scoring'),
    ('min_cv_train_size', True, TypeError, 'min_cv_train_size'),
    ('coverage_threshold', None, TypeError, 'coverage_threshold'),
    ('coverage_threshold', '0.5', TypeError, 'coverage_threshold'),
    ('coverage_threshold', True, TypeError, 'coverage_threshold'),
    ('training_coverage_threshold', 'x', TypeError, 'training_coverage_threshold'),
    ('scale_features', True, ValueError, 'scale_features'),
    ('scale_features', 0, ValueError, 'scale_features'),
    ('aggregation_constraint', 'SUM', ValueError, 'additive_transformer'),
    ('aggregation_constraint', {'a1': 'mean'}, ValueError, 'aggregation_constraint'),
    ('impute_intermediate_frequencies', np.True_, ValueError, 'impute_intermediate_frequencies'),
    ('impute_intermediate_frequencies', 'COVARIATES_ONLY', ValueError,
     'impute_intermediate_frequencies'),
]


class TestRejectedValues:
    """Values outside the admissible set are refused at ``__init__``, the message naming the culprit."""

    @pytest.mark.parametrize(
        'param, value, error, fragment', _REJECTED,
        ids=[f'{name}-{index}' for index, (name, *_rest) in enumerate(_REJECTED)],
    )
    def test_rejected_value_raises_naming_the_parameter(self, param, value, error, fragment):
        """The refusal is immediate and its message names the parameter or the key."""
        with pytest.raises(error) as excinfo:
            make_imputer(**{param: value})
        assert fragment in str(excinfo.value)

    @pytest.mark.parametrize(
        'param', ['keep_lower_frequencies', 'restore_original_values', 'verbose',
                  'impute_unobserved_entities', 'target_as_covariate'],
    )
    def test_numpy_booleans_take_effect(self, reference_timeseries, param):
        """A ``numpy.bool_`` is admitted like a ``bool``, as sklearn does, and honoured."""
        imputer = fit_quietly(make_imputer(**{param: np.False_}), reference_timeseries)
        assert imputer.get_params()[param] is np.False_

    def test_numpy_integers_from_a_grid_are_admitted(self, reference_timeseries):
        """Values of an ``np.arange`` grid pass ``__init__`` and ``fit`` alike."""
        imputer = make_imputer(
            fit_predict_order='cv', cv=np.arange(2, 4)[1],
            min_cv_train_size=np.arange(1, 5)[3],
        )
        fit_quietly(imputer, reference_timeseries)
        assert imputer.cv_.get_n_splits() == 3

    @pytest.mark.parametrize('value', [np.int64(0), np.True_, np.float64(2.0)])
    def test_numpy_scalars_outside_the_domain_are_refused(self, value):
        """A numpy scalar is held to the same domain: >= 1, integer, never boolean."""
        with pytest.raises((TypeError, ValueError), match='min_cv_train_size'):
            make_imputer(min_cv_train_size=value)

    @pytest.mark.parametrize(
        'param, value',
        [
            ('interpolation_method', 'nonsense'),
            ('cv_scoring', 'not_a_scorer'),
            ('time_col', 3),
            ('panel_cols', 'country'),
        ],
        ids=['unknown_method', 'unknown_scorer', 'non_str_time_col', 'str_panel_cols'],
    )
    def test_values_checked_later_pass_init(self, param, value):
        """Values not checked at ``__init__`` pass it untouched.

        The method name is checked by the interpolation itself at ``fit``, the
        scorer by ``cross_val_score``; ``time_col`` and ``panel_cols`` are not
        checked at all.
        """
        assert make_imputer(**{param: value}).get_params()[param] is value

    def test_unknown_interpolation_method_fails_at_fit(self, reference_timeseries):
        """An unknown method name, admitted at ``__init__``, fails at ``fit``."""
        imputer = make_imputer(interpolation_method='nonsense')
        with pytest.raises(ValueError, match='nonsense'):
            fit_quietly(imputer, reference_timeseries)


# =============================================================================
# __init__ valide sans transformer (B3)
# =============================================================================
# Défauts documentés, recopiés pour comparaison avec la signature
_SPEC_DEFAULTS = {
    'estimator': None,
    'additive_transformer': None,
    'covariate_strategy': 'interpolate',
    'covariate_fallback': 'interpolate',
    'covariate_eligibility': 'any_entity',
    'interpolation_method': 'linear',
    'interpolation_anchor': None,
    'impute_intermediate_frequencies': False,
    'impute_unobserved_entities': False,
    'target_as_covariate': False,
    'fit_predict_order': 'frequency',
    'cv': None,
    'cv_scoring': 'neg_mean_absolute_percentage_error',
    'min_cv_train_size': 10,
    'imputation_scope': 'strict',
    'coverage_threshold': 0.5,
    'training_scope': None,
    'training_coverage_threshold': None,
    'scale_features': 'constant',
    'aggregation_constraint': 'sum',
    'keep_lower_frequencies': True,
    'on_frequency_mismatch': 'error',
    'restore_original_values': False,
    'time_col': None,
    'panel_cols': None,
    'verbose': False,
}


class TestInitDoesNotTransform:
    """``__init__`` validates, stores, and does nothing else."""

    def test_signature_defaults_are_those_of_the_spec(self):
        """Every default of the signature is the documented one; ``target_frequency`` has none."""
        signature = inspect.signature(HighFrequencyImputer.__init__)
        defaults = {
            name: parameter.default
            for name, parameter in signature.parameters.items()
            if name != 'self'
        }
        assert defaults.pop('target_frequency') is inspect.Parameter.empty
        assert defaults == _SPEC_DEFAULTS

    def test_minimal_imputer_reports_the_defaults(self):
        """``get_params`` of a minimal imputer is the defaults, typed as sklearn wants.

        Hand-made counterpart of ``check_parameters_default_constructible``: each
        default is a plain immutable value (str, number, bool, None), and the
        instance reports it unchanged.
        """
        params = HighFrequencyImputer(target_frequency='M').get_params()
        assert params.pop('target_frequency') == 'M'
        assert params == _SPEC_DEFAULTS
        for name, value in params.items():
            assert value is None or isinstance(value, (str, int, float, bool)), name

    def test_no_fitted_attribute_after_init(self):
        """No attribute ending with ``_`` exists before ``fit``."""
        imputer = make_imputer(estimator=None, fit_predict_order='cv', cv=3)
        fitted = [name for name in vars(imputer) if name.endswith('_')]
        assert fitted == []
        assert imputer.__sklearn_is_fitted__() is False

    @pytest.mark.parametrize(
        'estimator', [None, LinearRegression(), {'__default__': LinearRegression()}],
        ids=['none', 'estimator', 'dict'],
    )
    def test_init_emits_no_warning(self, estimator):
        """``__init__`` is silent, ``estimator=None`` included.

        The single ``estimator=None`` warning comes at the head of ``fit``,
        before any training (see ``test_estimator_none_warns_once``): ``clone``
        calls ``__init__`` and would repeat it, and ``set_params(estimator=None)``
        would escape it.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            HighFrequencyImputer(target_frequency='M', estimator=estimator)
        assert [str(item.message) for item in caught] == []

    def test_mutable_parameters_are_not_mutated_by_init(self):
        """Dict parameters come out of ``__init__`` exactly as they went in."""
        # Clés d'entité au format scalaire : la normalisation en tuples ne doit
        # avoir lieu qu'au fit, dans effective_target_frequency_
        target_frequency = {'FR': 'monthly', 'DE': 'M'}
        scale_features = {'v': 'calendar', '__default__': 'constant'}
        anchors = {'v': 0.0}
        snapshot = (dict(target_frequency), dict(scale_features), dict(anchors))

        make_imputer(
            target_frequency=target_frequency,
            scale_features=scale_features,
            interpolation_anchor=anchors,
        )
        assert (target_frequency, scale_features, anchors) == snapshot


# =============================================================================
# Combinaisons inertes : silence documenté
# =============================================================================
class TestInertCombinations:
    """Inert combinations change nothing, warn about nothing, and are documented."""

    # Une combinaison inerte par cas, sous covariate_strategy='interpolate'
    _SILENT_CASES = [
        pytest.param(dict(covariate_fallback='tolerate_nan'), id='covariate_fallback'),
        pytest.param(dict(fit_predict_order='cv'), id='fit_predict_order'),
        pytest.param(
            dict(fit_predict_order='cv', cv=3, cv_scoring='r2', min_cv_train_size=2),
            id='cv_parameters',
        ),
        pytest.param(dict(training_coverage_threshold=0.0), id='training_coverage_low'),
        pytest.param(dict(training_coverage_threshold=1.0), id='training_coverage_high'),
    ]

    @pytest.mark.parametrize('overrides', _SILENT_CASES)
    def test_inert_combination_is_silent(self, reference_timeseries, overrides):
        """No warning mentions an inert parameter."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            make_imputer(**overrides).fit(reference_timeseries)
        mentioning = [
            str(item.message) for item in caught
            if any(name in str(item.message) for name in overrides)
        ]
        assert mentioning == []

    @pytest.mark.parametrize('overrides', [
        dict(covariate_fallback='tolerate_nan'),
        dict(fit_predict_order='cv', cv=3, cv_scoring='r2', min_cv_train_size=2),
        dict(training_coverage_threshold=0.0),
    ], ids=['covariate_fallback', 'cv_parameters', 'training_coverage'])
    def test_inert_combination_changes_nothing(self, reference_timeseries, overrides):
        """The output and the training window are those of the reference setting."""
        reference_imputer = make_imputer()
        reference = fit_transform_quietly(reference_imputer, reference_timeseries)
        imputer = make_imputer(**overrides)
        output = fit_transform_quietly(imputer, reference_timeseries)

        pd.testing.assert_frame_equal(output, reference)
        pd.testing.assert_series_equal(
            imputer.training_window_mask_, reference_imputer.training_window_mask_
        )

    def test_cv_order_outside_model_triggers_no_cross_validation(self, reference_timeseries):
        """Under ``'interpolate'``, ``fit_predict_order='cv'`` scores nothing."""
        calls = []
        with patch(
            'tsforecast.frequency.variable_orderer.cross_val_score',
            side_effect=lambda *args, **kwargs: calls.append(1) or np.zeros(2),
        ):
            fit_with_spy(reference_timeseries, fit_predict_order='cv', cv=3)
        assert calls == []

    def test_cv_warning_is_kept_under_model(self, reference_timeseries):
        """Under ``'model'`` the ``min_cv_train_size`` warning has an object, and stays."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            make_imputer(
                covariate_strategy='model', fit_predict_order='cv', cv=3,
                min_cv_train_size=2,
            ).fit(reference_timeseries)
        assert any('min_cv_train_size' in str(item.message) for item in caught)

    @pytest.mark.parametrize(
        'fragment',
        ["'covariates_only'", 'covariate_fallback', 'fit_predict_order',
         'training_coverage_threshold'],
    )
    def test_inert_combinations_are_documented(self, fragment):
        """The class docstring names each inert combination."""
        docstring = HighFrequencyImputer.__doc__
        section = docstring[
            docstring.index('Inert combinations'):docstring.index('What the parameter space')
        ]
        assert fragment in section


# =============================================================================
# Dictionnaires par feature : clés de colonne seulement
# =============================================================================
class TestPerFeatureDictUnknownColumn:
    """Per-feature dicts are keyed by column; an unknown key is ignored with one warning.

    An error would break a dict
    shared by datasets whose columns differ; silence would hide a typo. The
    key is ignored — the column falls back to ``'__default__'`` or to the
    default — and a single ``UserWarning`` at ``fit`` lists every such key.
    """

    _UNKNOWN = {
        'estimator': LinearRegression(),
        'scale_features': 'calendar',
        'interpolation_method': 'linear',
        'interpolation_anchor': 0.5,
        'aggregation_constraint': None,
    }

    @staticmethod
    def _warnings_of(imputer, data):
        """Fit and return the messages of the warnings about unknown keys."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            imputer.fit(data)
        return [
            str(item.message) for item in caught
            if 'absent from the data' in str(item.message)
        ]

    @pytest.mark.parametrize('param', sorted(_UNKNOWN))
    def test_unknown_column_warns_once_at_fit(self, reference_timeseries, param):
        """One warning naming the parameter and the key, and the fit goes on."""
        imputer = make_imputer(**{param: {'zz': self._UNKNOWN[param]}})
        messages = self._warnings_of(imputer, reference_timeseries)
        assert len(messages) == 1
        assert param in messages[0] and "'zz'" in messages[0]
        assert imputer.__sklearn_is_fitted__()

    def test_unknown_keys_of_several_parameters_share_one_warning(self, reference_timeseries):
        """Two parameters with unknown keys: a single message lists both."""
        imputer = make_imputer(
            scale_features={'zz': 'calendar', 'a1': 'calendar'},
            interpolation_anchor={'yy': 0.0},
        )
        messages = self._warnings_of(imputer, reference_timeseries)
        assert len(messages) == 1
        assert "scale_features: ['zz']" in messages[0]
        assert "interpolation_anchor: ['yy']" in messages[0]

    @pytest.mark.parametrize(
        'setting',
        [{'a1': 'calendar'}, {'__default__': 'calendar'}, 'calendar'],
        ids=['known_column', 'default_key', 'scalar'],
    )
    def test_known_keys_are_silent(self, reference_timeseries, setting):
        """A known column, the ``'__default__'`` key or the scalar form warns about nothing."""
        imputer = make_imputer(scale_features=setting)
        assert self._warnings_of(imputer, reference_timeseries) == []

    def test_target_column_is_a_known_key(self, reference_timeseries):
        """The column of ``y`` is a column of the data: its key is not unknown."""
        data = reference_timeseries.drop(columns=['a1'])
        target = reference_timeseries['a1']
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            make_imputer(scale_features={'a1': 'calendar'}).fit(data, target)
        assert not [item for item in caught if 'absent from the data' in str(item.message)]

    def test_unknown_key_is_ignored(self, reference_timeseries):
        """The output is that of the dict without the unknown key."""
        reference = fit_transform_quietly(
            make_imputer(scale_features={'a1': 'calendar'}), reference_timeseries
        )
        output = fit_transform_quietly(
            make_imputer(scale_features={'a1': 'calendar', 'zz': False}),
            reference_timeseries,
        )
        pd.testing.assert_frame_equal(output, reference)

    @pytest.mark.parametrize('param', sorted(_UNKNOWN))
    def test_entity_column_key_warns_with_the_d10_hint(
        self, mixed_freq_panel_multifrequency, param
    ):
        """An ``(entity, column)`` key is no column name; the message says so."""
        imputer = make_imputer(**{param: {('FR', 'v'): self._UNKNOWN[param]}})
        messages = self._warnings_of(imputer, mixed_freq_panel_multifrequency)
        assert len(messages) == 1
        assert "('FR', 'v')" in messages[0]
        assert 'never (entity, column) pairs' in messages[0]
        # Orientation vers la forme imbriquée de scale_features
        assert '{column: {entity: mode}}' in messages[0]

    def test_estimator_dict_unknown_column_falls_back_to_the_default(
        self, reference_timeseries
    ):
        """An estimator dict naming no column fits every column on ``'__default__'``."""
        imputer = make_imputer(
            estimator={'zz': SpyEstimator(), '__default__': LinearRegression()}
        )
        assert len(self._warnings_of(imputer, reference_timeseries)) == 1
        assert all(
            isinstance(step.model, LinearRegression) for step in imputer.imputation_plan_
        )


# =============================================================================
# Modalités d'échelle par entité
# =============================================================================
# Panel journalier : 'v' mensuelle (300 en fin de mois), 'd1' journalière
def _daily_panel(entities=('DE', 'FR')) -> pd.DataFrame:
    """Build a daily panel whose monthly column is imputed at a daily stage.

    Args:
        entities: Entity names, single index level ``country``.

    Returns:
        ``MultiIndex`` (``country``, ``date``) frame over the first half of
        2024 (leap February): ``d1`` daily and dense, ``v`` equal to 300.0 on
        month ends and NaN elsewhere.
    """
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D', name='date')
    frames = []
    for entity in entities:
        frame = pd.DataFrame({'d1': np.arange(len(dates), dtype=float)}, index=dates)
        frame['v'] = np.where(dates.is_month_end, 300.0, np.nan)
        frames.append(pd.concat({entity: frame}, names=['country']))
    return pd.concat(frames)


class TestPerEntityScaleFeatures:
    """``scale_features={column: {entity: mode}}`` through the imputer.

    Golden values of ``y_train`` at the daily stage: 300 divided by the real
    days of each month of 2024 under ``'calendar'`` (31, 29, 31, 30, 31, 30),
    by 30 under ``'constant'``.
    """

    _DAYS_2024 = [31.0, 29.0, 31.0, 30.0, 31.0, 30.0]

    def _y_train(self, scale_features) -> pd.Series:
        """Fit on the daily panel and return the target the spy model saw."""
        imputer = fit_with_spy(
            _daily_panel(), target_frequency='D', scale_features=scale_features
        )
        (step,) = imputer.imputation_plan_
        return step.model.fit_y_

    def test_y_train_is_scaled_per_entity(self):
        """DE raw (real days), FR seasonally adjusted (30), in one mutualized model."""
        y_train = self._y_train({'v': {'DE': 'calendar'}})
        np.testing.assert_allclose(
            y_train.loc['DE'].to_numpy(), [300.0 / days for days in self._DAYS_2024]
        )
        np.testing.assert_allclose(y_train.loc['FR'].to_numpy(), [10.0] * 6)

    @pytest.mark.parametrize(
        'nested, scalar',
        [
            ({'v': {'DE': 'calendar', 'FR': 'calendar'}}, 'calendar'),
            ({'v': {'__default__': 'constant'}}, 'constant'),
            ({'v': {'IT': 'calendar'}}, 'constant'),
        ],
        ids=['every-entity-calendar', 'nested-default-only', 'absent-entity'],
    )
    def test_unanimous_nested_setting_equals_the_scalar_one(self, nested, scalar):
        """A nested dict resolving to one mode behaves exactly like that mode."""
        pd.testing.assert_series_equal(self._y_train(nested), self._y_train(scalar))

    def test_nested_dict_is_closed_on_itself(self):
        """FR, unnamed, takes 'constant' — never the top-level 'calendar' default."""
        y_train = self._y_train({'v': {'DE': 'calendar'}, '__default__': 'calendar'})
        np.testing.assert_allclose(y_train.loc['FR'].to_numpy(), [10.0] * 6)

    def test_per_entity_scaled_output_keeps_the_totals(self):
        """The aggregation constraint brings each entity's month back to its total."""
        output = fit_transform_quietly(
            make_imputer(target_frequency='D', scale_features={'v': {'DE': 'calendar'}}),
            _daily_panel(),
        )
        totals = output['v'].groupby(
            [output.index.get_level_values('country'),
             output.index.get_level_values('date').to_period('M')]
        ).sum()
        np.testing.assert_allclose(totals.to_numpy(), 300.0, rtol=1e-9)

    @pytest.mark.parametrize(
        'setting',
        [{'v': {'DE': False}}, {'v': {'DE': 'calendar', '__default__': False}}],
        ids=['entity', 'nested-default'],
    )
    def test_false_within_a_nested_dict_raises_at_init(self, setting):
        """False applies to a whole column, never to some of its entities."""
        with pytest.raises(ValueError, match='never to some of its entities'):
            make_imputer(scale_features=setting)

    def test_false_within_a_nested_dict_raises_at_fit_after_set_params(self):
        """set_params bypasses __init__: the check is replayed at fit."""
        imputer = make_imputer(target_frequency='D').set_params(
            scale_features={'v': {'DE': False}}
        )
        with pytest.raises(ValueError, match='never to some of its entities'):
            fit_quietly(imputer, _daily_panel())

    # --- Entités inconnues : avertissement unique -----------------------------
    @staticmethod
    def _warnings_of(imputer, data):
        """Fit and return the messages of the warnings about unknown keys."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            imputer.fit(data)
        return [
            str(item.message) for item in caught
            if 'absent from the data' in str(item.message)
        ]

    def test_unknown_entity_warns_once(self):
        """An entity key naming no entity is ignored, with one warning naming it."""
        imputer = make_imputer(
            target_frequency='D', scale_features={'v': {'IT': 'calendar', 'DE': 'calendar'}}
        )
        messages = self._warnings_of(imputer, _daily_panel())
        assert len(messages) == 1
        assert "'v': ['IT']" in messages[0]

    def test_known_entities_are_silent(self):
        """Plain and tuple keys of real entities, and the nested default, warn about nothing."""
        imputer = make_imputer(
            target_frequency='D',
            scale_features={'v': {'DE': 'calendar', ('FR',): 'constant', '__default__': 'constant'}},
        )
        assert self._warnings_of(imputer, _daily_panel()) == []

    def test_unknown_column_and_unknown_entity_share_one_warning(self):
        """Both kinds of unknown keys are listed by the same single warning."""
        imputer = make_imputer(
            target_frequency='D',
            scale_features={'zz': 'calendar', 'v': {'IT': 'calendar'}},
        )
        messages = self._warnings_of(imputer, _daily_panel())
        assert len(messages) == 1
        assert "scale_features: ['zz']" in messages[0] and "'v': ['IT']" in messages[0]

    def test_partial_key_on_a_two_level_panel_warns(self, panel_two_level_dataset):
        """'France' names no entity of a (country, sector) panel: reported, not expanded."""
        imputer = make_imputer(
            scale_features={'indicateur_trimestriel': {'France': 'calendar'}}
        )
        messages = self._warnings_of(imputer, panel_two_level_dataset)
        assert len(messages) == 1
        assert "['France']" in messages[0]
        assert 'partial key' in messages[0]

    def test_full_key_on_a_two_level_panel_is_silent(self, panel_two_level_dataset):
        """('France', 'Industrie') names a real entity of the two-level panel."""
        imputer = make_imputer(
            scale_features={'indicateur_trimestriel': {('France', 'Industrie'): 'calendar'}}
        )
        assert self._warnings_of(imputer, panel_two_level_dataset) == []

    def test_entity_keys_on_a_time_series_warn(self, reference_timeseries):
        """A time series has no entity: every nested key is ignored and reported."""
        imputer = make_imputer(scale_features={'a1': {'FR': 'calendar'}})
        messages = self._warnings_of(imputer, reference_timeseries)
        assert len(messages) == 1
        assert "'a1': ['FR']" in messages[0]


# =============================================================================
# Effet observable minimal des valeurs admises
# =============================================================================
class TestAdmittedValuesTakeEffect:
    """An admitted value is honoured, not merely accepted."""

    def test_verbose_prints_prefixed_progress(self, reference_timeseries, capsys):
        """``verbose=True`` prints ``[HighFrequencyImputer]`` lines; ``False`` prints nothing."""
        fit_quietly(make_imputer(verbose=False), reference_timeseries)
        assert capsys.readouterr().out == ''

        fit_quietly(make_imputer(verbose=True), reference_timeseries)
        lines = capsys.readouterr().out.splitlines()
        assert lines
        assert all(line.startswith('[HighFrequencyImputer]') for line in lines)

    def test_additive_transformer_is_cloned_not_fitted(self, reference_timeseries):
        """The fitted copy is ``additive_transformer_``; the parameter stays unfitted (B3)."""
        transformer = FunctionTransformer(np.log1p, np.expm1)
        imputer = fit_quietly(
            make_imputer(additive_transformer=transformer), reference_timeseries
        )
        assert imputer.additive_transformer_ is not transformer
        assert hasattr(imputer.additive_transformer_, 'n_features_in_')
        assert not hasattr(transformer, 'n_features_in_')

    def test_on_frequency_mismatch_warn_turns_the_error_into_a_warning(
        self, reference_timeseries
    ):
        """A target finer than the data raises under ``'error'`` and warns under ``'warn'``."""
        with pytest.raises(ValueError, match='higher than'):
            fit_quietly(
                make_imputer(target_frequency='D', on_frequency_mismatch='error'),
                reference_timeseries,
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            make_imputer(target_frequency='D', on_frequency_mismatch='warn').fit(
                reference_timeseries
            )
        assert any('higher than' in str(item.message) for item in caught)

    @pytest.mark.slow
    def test_all_entities_drops_a_covariate_absent_for_one_entity(
        self, mixed_freq_panel_heterogeneous
    ):
        """``'all_entities'`` drops ``climat_affaires``, never observed by IT."""
        def _features(eligibility):
            imputer = fit_with_spy(
                mixed_freq_panel_heterogeneous, covariate_eligibility=eligibility
            )
            return {
                column for step in imputer.imputation_plan_ for column in step.feature_cols
            }

        assert 'climat_affaires' in _features('any_entity')
        assert 'climat_affaires' not in _features('all_entities')
