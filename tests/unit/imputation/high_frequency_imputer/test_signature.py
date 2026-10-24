"""Snapshots of the public surface of ``HighFrequencyImputer``.

Covers the signature of ``__init__`` (names, order, kinds and defaults of the
parameters), the output of ``get_params()`` on the defaults, and the sorted
list of the public fitted attributes after ``fit``, with and without the
``'cv'`` order (the only one that adds ``cv_``) and with and without ``y``.
Any change to one of them changes the public API of the class.
"""
# Modules de base
import inspect
import warnings

import pytest
from sklearn.linear_model import LinearRegression

# Objet testé
from tsforecast.frequency import HighFrequencyImputer

# Lecture des attributs publics partagée avec le golden master
from tests.support.golden.scenarios import public_attribute_names

# Signature figée : (nom, nature, défaut) dans l'ordre de déclaration
_POSITIONAL_OR_KEYWORD = inspect.Parameter.POSITIONAL_OR_KEYWORD
_INIT_SIGNATURE = [
    ('target_frequency', _POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ('estimator', _POSITIONAL_OR_KEYWORD, None),
    ('additive_transformer', _POSITIONAL_OR_KEYWORD, None),
    ('covariate_strategy', _POSITIONAL_OR_KEYWORD, 'interpolate'),
    ('covariate_fallback', _POSITIONAL_OR_KEYWORD, 'interpolate'),
    ('covariate_eligibility', _POSITIONAL_OR_KEYWORD, 'any_entity'),
    ('interpolation_method', _POSITIONAL_OR_KEYWORD, 'linear'),
    ('interpolation_anchor', _POSITIONAL_OR_KEYWORD, None),
    ('impute_intermediate_frequencies', _POSITIONAL_OR_KEYWORD, False),
    ('impute_unobserved_entities', _POSITIONAL_OR_KEYWORD, False),
    ('target_as_covariate', _POSITIONAL_OR_KEYWORD, False),
    ('fit_predict_order', _POSITIONAL_OR_KEYWORD, 'frequency'),
    ('cv', _POSITIONAL_OR_KEYWORD, None),
    ('cv_scoring', _POSITIONAL_OR_KEYWORD, 'neg_mean_absolute_percentage_error'),
    ('min_cv_train_size', _POSITIONAL_OR_KEYWORD, 10),
    ('imputation_scope', _POSITIONAL_OR_KEYWORD, 'strict'),
    ('coverage_threshold', _POSITIONAL_OR_KEYWORD, 0.5),
    ('training_scope', _POSITIONAL_OR_KEYWORD, None),
    ('training_coverage_threshold', _POSITIONAL_OR_KEYWORD, None),
    ('scale_features', _POSITIONAL_OR_KEYWORD, 'constant'),
    ('aggregation_constraint', _POSITIONAL_OR_KEYWORD, 'sum'),
    ('keep_lower_frequencies', _POSITIONAL_OR_KEYWORD, True),
    ('on_frequency_mismatch', _POSITIONAL_OR_KEYWORD, 'error'),
    ('restore_original_values', _POSITIONAL_OR_KEYWORD, False),
    ('time_col', _POSITIONAL_OR_KEYWORD, None),
    ('panel_cols', _POSITIONAL_OR_KEYWORD, None),
    ('verbose', _POSITIONAL_OR_KEYWORD, False),
]

# get_params() d'un imputeur minimal : clés triées par sklearn, défauts exacts
_DEFAULT_PARAMS = {
    'additive_transformer': None,
    'aggregation_constraint': 'sum',
    'covariate_eligibility': 'any_entity',
    'covariate_fallback': 'interpolate',
    'covariate_strategy': 'interpolate',
    'coverage_threshold': 0.5,
    'cv': None,
    'cv_scoring': 'neg_mean_absolute_percentage_error',
    'estimator': None,
    'fit_predict_order': 'frequency',
    'imputation_scope': 'strict',
    'impute_intermediate_frequencies': False,
    'impute_unobserved_entities': False,
    'interpolation_anchor': None,
    'interpolation_method': 'linear',
    'keep_lower_frequencies': True,
    'min_cv_train_size': 10,
    'on_frequency_mismatch': 'error',
    'panel_cols': None,
    'restore_original_values': False,
    'scale_features': 'constant',
    'target_as_covariate': False,
    'target_frequency': 'M',
    'time_col': None,
    'training_coverage_threshold': None,
    'training_scope': None,
    'verbose': False,
}

# Attributs publics ajustés, ordre alphabétique, hors ordre 'cv'
_FITTED_ATTRIBUTES = [
    'additive_transformer_',
    'detected_frequencies_',
    'effective_target_frequency_',
    'entities_',
    'feature_columns_',
    'feature_names_in_',
    'frequency_progression_',
    'imputation_cv_scores_',
    'imputation_models_',
    'imputation_order_',
    'imputation_plan_',
    'imputation_provenance_',
    'imputation_window_',
    'imputation_window_mask_',
    'is_panel_',
    'n_features_in_',
    'provenance_statistics_',
    'strict_window_mask_',
    'target_column_',
    'training_window_',
    'training_window_mask_',
    'unanchored_pairs_',
    'variable_categories_',
]


def _fit_quietly(imputer: HighFrequencyImputer, *data) -> HighFrequencyImputer:
    """Fit an imputer with every warning silenced and return it."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return imputer.fit(*data)


class TestInitSignature:
    """The signature of ``__init__`` is frozen: names, order, kinds, defaults."""

    def test_parameters_in_declaration_order(self):
        """Every parameter keeps its name, its position, its kind and its default."""
        parameters = list(inspect.signature(HighFrequencyImputer.__init__).parameters.values())
        # Premier paramètre : l'instance elle-même
        assert parameters[0].name == 'self'
        snapshot = [
            (parameter.name, parameter.kind, parameter.default) for parameter in parameters[1:]
        ]
        assert snapshot == _INIT_SIGNATURE


class TestDefaultParams:
    """``get_params()`` of a minimal imputer is frozen, key order included."""

    def test_get_params_on_defaults(self):
        """``get_params()`` returns the defaults, keys sorted, values of the same type."""
        params = HighFrequencyImputer(target_frequency='M').get_params()
        assert list(params.items()) == list(_DEFAULT_PARAMS.items())
        # Types exacts : False n'est pas 0, 0.5 n'est pas un entier
        assert [type(value) for value in params.values()] == [
            type(value) for value in _DEFAULT_PARAMS.values()
        ]


class TestFittedAttributes:
    """The sorted list of the public fitted attributes is frozen."""

    def test_after_fit_on_defaults(self, reference_timeseries):
        """A fit with the defaults exposes the listed attributes, ``cv_`` excepted."""
        imputer = _fit_quietly(
            HighFrequencyImputer(target_frequency='M', estimator=LinearRegression()),
            reference_timeseries,
        )
        assert public_attribute_names(imputer) == _FITTED_ATTRIBUTES

    def test_after_fit_with_target_on_a_panel(self, mixed_freq_panel_multifrequency):
        """A fit with ``y`` on a panel exposes the very same attributes."""
        X = mixed_freq_panel_multifrequency.drop(columns='q1')
        imputer = _fit_quietly(
            HighFrequencyImputer(target_frequency='M', estimator=LinearRegression()),
            X, mixed_freq_panel_multifrequency['q1'],
        )
        assert public_attribute_names(imputer) == _FITTED_ATTRIBUTES

    def test_cv_order_adds_cv(self, reference_timeseries):
        """The ``'cv'`` order under ``'model'`` adds ``cv_`` and nothing else."""
        imputer = _fit_quietly(
            HighFrequencyImputer(
                target_frequency='M', estimator=LinearRegression(),
                covariate_strategy='model', fit_predict_order='cv', cv=3,
            ),
            reference_timeseries,
        )
        assert public_attribute_names(imputer) == sorted(_FITTED_ATTRIBUTES + ['cv_'])

    @pytest.mark.parametrize('keep_lower_frequencies', [True, False], ids=['stacked', 'flat'])
    def test_transform_adds_no_attribute(self, reference_timeseries, keep_lower_frequencies):
        """``transform`` rewrites per-pass attributes but adds none."""
        imputer = _fit_quietly(
            HighFrequencyImputer(
                target_frequency='M', estimator=LinearRegression(),
                keep_lower_frequencies=keep_lower_frequencies,
            ),
            reference_timeseries,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.transform(reference_timeseries)
        assert public_attribute_names(imputer) == _FITTED_ATTRIBUTES
