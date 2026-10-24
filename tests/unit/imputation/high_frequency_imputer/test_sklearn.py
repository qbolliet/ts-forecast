"""sklearn conformance of ``HighFrequencyImputer``.

Covers ``clone``, ``get_params`` / ``set_params`` and nested parameters,
``NotFittedError`` and ``__sklearn_is_fitted__``, the end-to-end
``Pipeline`` / ``GridSearchCV`` use, the validation of values reaching the
imputer through ``set_params``, the parameters left untouched by ``fit``, its
idempotence, and the persistence of a fitted imputer (pickle, deepcopy).

Checks of ``sklearn.utils.estimator_checks`` run as such:
``check_get_params_invariance``, ``check_set_params``,
``check_no_attributes_set_in_init``. Checks NOT applicable as such, and why:

- ``check_dont_overwrite_parameters``, ``check_fit_idempotent``,
  ``check_estimators_pickle``, ``check_fit2d_*``, ``check_transformer_general``,
  ``check_estimators_dtypes``: they fit on bare ``numpy`` arrays, while the
  imputer requires a ``DataFrame`` carrying a temporal index (and an entity
  level for a panel) — the first three are rewritten by hand below, on the
  reference datasets;
- ``check_parameters_default_constructible``: ``target_frequency`` is required
  by design and the class declares no ``_required_parameters``; checked by
  hand in ``test_parameters.py::TestInitDoesNotTransform``;
- ``check_n_features_in``: fits on a bare array too; its contract —
  ``n_features_in_`` / ``feature_names_in_`` written by ``fit`` — is checked by
  hand in :class:`TestFeaturesSeenAtFit`.
"""

# Modules de base
import copy
import pickle
import re
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.estimator_checks import (
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_set_params,
)
from sklearn.utils.validation import check_is_fitted

# Objets testés
from tsforecast.imputation.high_frequency_imputer import HighFrequencyImputer
from tsforecast.xy import DropNaNTarget, XYPipeline

# Outils de test partagés
from tests.unit.imputation.high_frequency_imputer.helpers import (
    fit_quietly,
    make_imputer,
    transform_quietly,
)


# =============================================================================
# Contrat de paramètres : B3, paramètres supprimés, NotFittedError
# =============================================================================
class TestParameterContract:
    """sklearn conformance of the parameter space."""

    def test_impute_intermediate_frequencies_covariates_only_is_not_true(self):
        """'covariates_only' est accepté et n'est jamais confondu avec True."""
        imputer = make_imputer(impute_intermediate_frequencies='covariates_only')
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
            make_imputer(**{removed: True})

    def test_get_params_returns_untouched_values(self):
        """get_params rend les objets REÇUS, sans normalisation (B3)."""
        target_frequency = {'FR': 'monthly', 'DE': 'M'}
        scale_features = {'v': 'calendar', 'q1': False}
        splitter = KFold(n_splits=3)
        anchors = {'v': 0.0}
        constraint = {'v': None, '__default__': 'sum'}

        imputer = make_imputer(
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
        imputer = make_imputer(
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
        imputer = make_imputer()
        imputer.set_params(covariate_strategy='model', min_cv_train_size=1)
        assert imputer.covariate_strategy == 'model'
        assert imputer.min_cv_train_size == 1

    def test_not_fitted_error_before_fit(self):
        """transform et inverse_transform lèvent NotFittedError avant fit (B20)."""
        imputer = make_imputer()
        frame = pd.DataFrame(
            {'m1': [1.0, 2.0]}, index=pd.date_range('2021-01-31', periods=2, freq='ME')
        )
        with pytest.raises(NotFittedError):
            imputer.transform(frame)
        with pytest.raises(NotFittedError):
            imputer.inverse_transform(frame)

    def test_imputation_models_raises_attribute_error_before_fit(self):
        """La vue des modèles lève AttributeError, jamais NotFittedError, avant fit."""
        imputer = make_imputer()
        assert not hasattr(imputer, 'imputation_models_')
        with pytest.raises(AttributeError, match='imputation_plan_'):
            _ = imputer.imputation_models_


class TestSklearnConformance:
    """End-to-end sklearn conformance."""

    # Cible de panel par entité, forme dict
    _TARGET = {('FR',): 'M', ('DE',): 'M', ('IT',): 'M'}

    def test_pipeline_fit_transform(self, mixed_freq_panel_multifrequency):
        """L'imputeur s'insère dans un `Pipeline` sklearn et le traverse."""
        pipeline = Pipeline([
            ('imputer', HighFrequencyImputer(
                target_frequency=self._TARGET,
                estimator=LinearRegression(),
                keep_lower_frequencies=False,
            )),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = pipeline.fit_transform(mixed_freq_panel_multifrequency)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(mixed_freq_panel_multifrequency.columns)
        # Le clone du pipeline reste conforme : les paramètres traversent
        assert clone(pipeline).get_params()['imputer__target_frequency'] == self._TARGET

    @pytest.mark.slow
    def test_grid_search_on_panel_with_target_frequency_dict(
        self, mixed_freq_panel_multifrequency
    ):
        """`GridSearchCV` explores the imputer on a panel with a per-entity target.

        The target `v` is low frequency for FR and DE: the training pass
        imputes it, the prediction pass returns it as is, with its NaN
        outside the anchors; `DropNaNTarget` removes them before the regressor.
        """
        panel = mixed_freq_panel_multifrequency
        X = panel[['m1', 'q1']]
        y = panel['v']

        # Découpage par DATE, pour que les deux plis restent des panels
        # complets : un pli privé d'entité ne relèverait pas de ce test
        dates = panel.index.get_level_values(-1)
        cutoff = pd.Timestamp('2022-12-31')
        split = [(
            np.flatnonzero(dates <= cutoff),
            np.flatnonzero(dates > cutoff),
        )]

        pipeline = XYPipeline([
            ('imputer', HighFrequencyImputer(
                target_frequency=self._TARGET,
                estimator=LinearRegression(),
                keep_lower_frequencies=False,
            )),
            ('fill', SimpleImputer()),
            ('drop', DropNaNTarget()),
            ('regressor', LinearRegression()),
        ])
        search = GridSearchCV(
            pipeline,
            {'imputer__covariate_strategy': ['interpolate', 'model']},
            cv=split,
            error_score='raise',
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            search.fit(X, y)

        assert search.best_params_['imputer__covariate_strategy'] in (
            'interpolate', 'model'
        )
        assert np.isfinite(search.best_score_)


# =============================================================================
# Contrôles de sklearn.utils.estimator_checks, tels quels
# =============================================================================
# Configurations exercées : défauts, formes dictionnaire, estimateur imbriqué
_CONFIGURATIONS = {
    'defaults': dict(),
    'estimator_none': dict(estimator=None),
    'dict_forms': dict(
        target_frequency={'FR': 'M', 'DE': 'Q'},
        estimator={'v': LinearRegression(), '__default__': LinearRegression()},
        scale_features={'v': 'calendar', '__default__': 'constant'},
        interpolation_anchor={'v': 0.0},
        aggregation_constraint={'v': None, '__default__': 'sum'},
    ),
    'nested_pipeline': dict(
        estimator=Pipeline([('fill', SimpleImputer()), ('model', LinearRegression())]),
        additive_transformer=FunctionTransformer(np.log1p, np.expm1),
        cv=KFold(n_splits=3),
        fit_predict_order='cv',
    ),
}


class TestEstimatorChecks:
    """The applicable checks of ``sklearn.utils.estimator_checks`` pass."""

    @pytest.mark.parametrize('config', sorted(_CONFIGURATIONS))
    @pytest.mark.parametrize(
        'check',
        [check_get_params_invariance, check_set_params, check_no_attributes_set_in_init],
        ids=lambda check: check.__name__,
    )
    def test_sklearn_check(self, check, config):
        """The check runs on the configured imputer without raising."""
        check('HighFrequencyImputer', make_imputer(**_CONFIGURATIONS[config]))


# =============================================================================
# clone, paramètres imbriqués, set_params
# =============================================================================
class TestCloneAndNestedParams:
    """``clone`` and the nested-parameter protocol of sklearn."""

    def test_clone_of_a_fitted_imputer_is_unfitted(self, reference_timeseries):
        """``clone`` keeps the parameters and drops the fitted state."""
        imputer = fit_quietly(make_imputer(covariate_strategy='model'), reference_timeseries)
        cloned = clone(imputer)

        assert cloned.__sklearn_is_fitted__() is False
        assert not [name for name in vars(cloned) if name.endswith('_')]
        assert cloned.get_params()['covariate_strategy'] == 'model'

    def test_clone_copies_an_estimator_dict_in_depth(self):
        """The estimators of a dict are copies: the clone shares no model."""
        estimators = {'v': LinearRegression(), '__default__': LinearRegression()}
        cloned = clone(make_imputer(estimator=estimators))
        copied = cloned.get_params()['estimator']

        assert set(copied) == set(estimators)
        for key, estimator in estimators.items():
            assert copied[key] is not estimator

    def test_deep_params_expose_the_nested_estimator(self):
        """``get_params(deep=True)`` exposes ``estimator__*``, ``deep=False`` does not."""
        imputer = make_imputer(estimator=LinearRegression(fit_intercept=False))
        assert imputer.get_params(deep=True)['estimator__fit_intercept'] is False
        assert 'estimator__fit_intercept' not in imputer.get_params(deep=False)

    def test_set_params_reaches_the_nested_estimator(self):
        """``set_params(estimator__...)`` updates the nested estimator and returns self."""
        imputer = make_imputer()
        assert imputer.set_params(estimator__fit_intercept=False) is imputer
        assert imputer.estimator.fit_intercept is False

    def test_set_params_rejects_an_unknown_parameter(self):
        """An unknown parameter raises ``ValueError`` (sklearn contract)."""
        with pytest.raises(ValueError, match='cascade_refitting'):
            make_imputer().set_params(cascade_refitting=False)


# =============================================================================
# Valeurs passées par set_params : revalidées au fit
# =============================================================================
# Valeurs refusées à __init__, glissées par set_params (chemin de GridSearchCV)
_THROUGH_SET_PARAMS = [
    ('covariate_strategy', 'nope'),
    ('covariate_fallback', 'model'),
    ('covariate_eligibility', 'some'),
    ('fit_predict_order', 'random'),
    ('on_frequency_mismatch', 'ignore'),
    ('imputation_scope', 'wide'),
    ('training_scope', 'wide'),
    ('min_cv_train_size', 0),
    ('coverage_threshold', 1.5),
    ('training_coverage_threshold', -0.1),
    ('cv', 1),
    ('cv_scoring', 3),
    ('scale_features', 'mean'),
    ('aggregation_constraint', 'mean'),
    ('interpolation_method', 3),
    ('interpolation_anchor', 1.5),
    ('target_frequency', 12),
]
# Paramètres que seul l'imputeur valide : aucun
# composant reconstruit au fit ne les revalide à sa place
_THROUGH_SET_PARAMS += [
    ('impute_intermediate_frequencies', 'yes'),
    ('keep_lower_frequencies', 'yes'),
    ('restore_original_values', 'yes'),
    ('verbose', 'yes'),
    ('impute_unobserved_entities', 'yes'),
    ('estimator', object()),
    ('additive_transformer', object()),
]


class TestValidationThroughSetParams:
    """A value ``__init__`` refuses is refused at ``fit`` too when set by ``set_params``.

    ``set_params`` writes attributes without calling ``__init__``: the
    validation must therefore hold again at ``fit``, where every
    normalization takes place — for the components too, ``set_params``
    included.
    """

    @pytest.mark.parametrize(
        'param, value', _THROUGH_SET_PARAMS,
        ids=[name for name, _value in _THROUGH_SET_PARAMS],
    )
    def test_fit_refuses_a_value_init_refuses(self, reference_timeseries, param, value):
        """``fit`` raises, the message naming the parameter."""
        imputer = make_imputer().set_params(**{param: value})
        with pytest.raises((ValueError, TypeError), match=param):
            fit_quietly(imputer, reference_timeseries)

    @pytest.mark.parametrize(
        'param, value', _THROUGH_SET_PARAMS,
        ids=[name for name, _value in _THROUGH_SET_PARAMS],
    )
    def test_fit_raises_the_message_of_init(self, reference_timeseries, param, value):
        """The error at ``fit`` is the very one ``__init__`` raises (same type, same text)."""
        with pytest.raises((ValueError, TypeError)) as at_init:
            make_imputer(**{param: value})
        imputer = make_imputer().set_params(**{param: value})
        with pytest.raises(at_init.type, match=re.escape(str(at_init.value))):
            fit_quietly(imputer, reference_timeseries)


# =============================================================================
# NotFittedError et prédicat d'ajustement (B20)
# =============================================================================
class TestNotFitted:
    """``__sklearn_is_fitted__`` drives ``check_is_fitted`` (B20)."""

    def test_check_is_fitted_before_and_after_fit(self, reference_timeseries):
        """Unfitted: ``NotFittedError`` naming the class; fitted: silence."""
        imputer = make_imputer()
        assert imputer.__sklearn_is_fitted__() is False
        with pytest.raises(NotFittedError, match='HighFrequencyImputer'):
            check_is_fitted(imputer)

        fit_quietly(imputer, reference_timeseries)
        assert imputer.__sklearn_is_fitted__() is True
        check_is_fitted(imputer)


# =============================================================================
# Le fit ne touche pas aux paramètres, et est idempotent
# =============================================================================
class TestFitLeavesParametersAlone:
    """Hand-made ``check_dont_overwrite_parameters`` and ``check_fit_idempotent``."""

    @pytest.mark.parametrize(
        'fixture_name, overrides',
        [
            ('reference_timeseries', dict(
                scale_features={'a1': 'calendar', '__default__': 'constant'},
                interpolation_anchor={'q1': 0.0},
            )),
            ('mixed_freq_panel_multifrequency', dict(
                target_frequency={'FR': 'M', 'DE': 'M', 'IT': 'M'},
                aggregation_constraint={'v': None, '__default__': 'sum'},
            )),
        ],
        ids=['ts', 'panel'],
    )
    def test_fit_does_not_overwrite_parameters(self, fixture_name, overrides, request):
        """Every parameter hashes the same after ``fit``; ``fit`` adds only ``*_`` / ``_*``."""
        data = request.getfixturevalue(fixture_name)
        imputer = make_imputer(**overrides)
        before = {name: joblib.hash(value) for name, value in imputer.get_params().items()}
        attributes_before = set(vars(imputer))

        fit_quietly(imputer, data)

        after = {name: joblib.hash(value) for name, value in imputer.get_params().items()}
        assert after == before
        added = set(vars(imputer)) - attributes_before
        public = [name for name in added if not name.startswith('_')]
        assert all(name.endswith('_') for name in public), public

    def test_fit_is_idempotent(self, reference_timeseries):
        """Two fits on the same data freeze the same plan and give the same output."""
        imputer = make_imputer(covariate_strategy='model')
        fit_quietly(imputer, reference_timeseries)
        first_plan = imputer.imputation_plan_.to_diagnostic_frame()
        first_output = transform_quietly(imputer, reference_timeseries)

        fit_quietly(imputer, reference_timeseries)
        pd.testing.assert_frame_equal(
            imputer.imputation_plan_.to_diagnostic_frame(), first_plan
        )
        pd.testing.assert_frame_equal(
            transform_quietly(imputer, reference_timeseries), first_output
        )


# =============================================================================
# Persistance d'un imputeur ajusté
# =============================================================================
class TestPersistence:
    """A fitted imputer survives pickle and deepcopy, and replays identically."""

    @pytest.mark.parametrize(
        'fixture_name', ['reference_timeseries', 'mixed_freq_panel_multifrequency']
    )
    def test_pickle_round_trip_replays_identically(self, fixture_name, request):
        """``pickle`` round trip: same output, same provenance, same plan."""
        data = request.getfixturevalue(fixture_name)
        imputer = fit_quietly(make_imputer(), data)
        restored = pickle.loads(pickle.dumps(imputer))

        pd.testing.assert_frame_equal(
            transform_quietly(restored, data), transform_quietly(imputer, data)
        )
        pd.testing.assert_frame_equal(
            restored.imputation_provenance_, imputer.imputation_provenance_
        )
        pd.testing.assert_frame_equal(
            restored.imputation_plan_.to_diagnostic_frame(),
            imputer.imputation_plan_.to_diagnostic_frame(),
        )

    def test_deepcopy_of_a_fitted_imputer_replays_identically(self, reference_timeseries):
        """``copy.deepcopy`` of a fitted imputer replays like the original."""
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        duplicate = copy.deepcopy(imputer)
        pd.testing.assert_frame_equal(
            transform_quietly(duplicate, reference_timeseries),
            transform_quietly(imputer, reference_timeseries),
        )

    def test_unfitted_imputer_pickles(self):
        """Before ``fit``, the imputer pickles (what joblib needs to dispatch clones)."""
        imputer = make_imputer(target_frequency={'FR': 'M'}, scale_features={'v': False})
        restored = pickle.loads(pickle.dumps(imputer))
        assert restored.get_params(deep=False).keys() == imputer.get_params(deep=False).keys()
        assert restored.get_params()['target_frequency'] == {'FR': 'M'}


# =============================================================================
# Colonnes vues au fit (convention sklearn)
# =============================================================================
class TestFeaturesSeenAtFit:
    """``fit`` writes ``n_features_in_`` and ``feature_names_in_``, as sklearn expects."""

    @pytest.mark.parametrize(
        'fixture_name', ['reference_timeseries', 'mixed_freq_panel_multifrequency']
    )
    def test_features_seen_at_fit(self, fixture_name, request):
        """Count and names of the feature columns, entity and date levels excluded."""
        data = request.getfixturevalue(fixture_name)
        imputer = fit_quietly(make_imputer(), data)

        assert imputer.n_features_in_ == data.shape[1]
        assert isinstance(imputer.feature_names_in_, np.ndarray)
        assert list(imputer.feature_names_in_) == list(data.columns)

    def test_pipeline_reads_the_count_of_its_first_step(self, reference_timeseries):
        """``Pipeline.n_features_in_`` delegates to the imputer."""
        pipeline = Pipeline([('impute', make_imputer())])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipeline.fit(reference_timeseries)
        assert pipeline.n_features_in_ == reference_timeseries.shape[1]
