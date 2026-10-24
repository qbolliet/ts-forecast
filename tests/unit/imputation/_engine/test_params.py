"""Tests of the parameter validators in ``tsforecast.imputation._engine.params``.

Scenarios: admitted and rejected values of each ``validate_*`` function, exact error
messages, numpy integers and booleans, target-frequency normalization.
"""

# Calcul numérique
import numpy as np

# Tests
import pytest

# Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from tsforecast.imputation._engine.params import (
    INTERMEDIATE_MODALITIES,
    is_integer,
    validate_additive_transformer,
    validate_cv,
    validate_cv_scoring,
    validate_estimator,
    validate_intermediate_frequencies,
    validate_literal,
    validate_min_cv_train_size,
    validate_target_frequency_format,
    validate_unit_interval,
)


class TestIsInteger:
    @pytest.mark.parametrize('value', [3, np.int64(3), np.int8(0), -2])
    def test_integers_admitted(self, value):
        assert is_integer(value)

    @pytest.mark.parametrize('value', [True, np.bool_(True), 3.0, '3', None])
    def test_non_integers_rejected(self, value):
        assert not is_integer(value)


class TestValidateLiteral:
    def test_admitted(self):
        validate_literal('order', 'cv', ('frequency', 'cv'))

    def test_rejected_message(self):
        with pytest.raises(
            ValueError, match=r"order must be one of \('frequency', 'cv'\), got 'x'"
        ):
            validate_literal('order', 'x', ('frequency', 'cv'))


class TestValidateIntermediateFrequencies:
    @pytest.mark.parametrize('value', [False, True, 'covariates_only'])
    def test_admitted(self, value):
        validate_intermediate_frequencies(value)

    @pytest.mark.parametrize('value', [0, 1, None, 'yes', 'True'])
    def test_rejected_without_truth_conflation(self, value):
        with pytest.raises(ValueError, match='impute_intermediate_frequencies must be one of'):
            validate_intermediate_frequencies(value)

    def test_message_lists_modalities(self):
        with pytest.raises(ValueError) as exc:
            validate_intermediate_frequencies(1)
        assert str(INTERMEDIATE_MODALITIES) in str(exc.value)
        assert str(exc.value).endswith('got 1')


class TestValidateUnitInterval:
    @pytest.mark.parametrize('value', [0, 1, 0.5, np.float64(0.25)])
    def test_admitted(self, value):
        validate_unit_interval('t', value)

    @pytest.mark.parametrize('value', [True, np.bool_(False), '0.5', None])
    def test_wrong_type(self, value):
        with pytest.raises(TypeError, match='t must be a float, got'):
            validate_unit_interval('t', value)

    @pytest.mark.parametrize('value', [-0.1, 1.5])
    def test_out_of_range(self, value):
        with pytest.raises(ValueError, match='t must be a float between 0 and 1'):
            validate_unit_interval('t', value)


class TestValidateCv:
    @pytest.mark.parametrize('value', [None, 2, np.int64(5), KFold(3), [([0], [1])], iter([])])
    def test_admitted(self, value):
        validate_cv(value)

    @pytest.mark.parametrize('value', [1, 0, np.int64(1)])
    def test_int_below_two(self, value):
        with pytest.raises(ValueError, match='cv must be an int >= 2'):
            validate_cv(value)

    @pytest.mark.parametrize('value', [True, False, np.bool_(True)])
    def test_bool_rejected(self, value):
        with pytest.raises(TypeError, match='cv must not be a bool'):
            validate_cv(value)

    @pytest.mark.parametrize('value', ['kfold', b'kfold', 2.5, object()])
    def test_unsupported_form(self, value):
        with pytest.raises(TypeError, match='cv must be None, an int >= 2'):
            validate_cv(value)


class TestValidateCvScoring:
    @pytest.mark.parametrize('value', ['neg_mean_squared_error', len])
    def test_admitted(self, value):
        validate_cv_scoring(value)

    @pytest.mark.parametrize('value', [None, 3])
    def test_rejected(self, value):
        with pytest.raises(TypeError, match='cv_scoring must be a str or a callable'):
            validate_cv_scoring(value)


class TestValidateTargetFrequencyFormat:
    def test_string_normalized(self):
        assert validate_target_frequency_format('M') == 'M'

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="Invalid target_frequency 'zzz'"):
            validate_target_frequency_format('zzz')

    def test_dict_keys_become_tuples(self):
        assert validate_target_frequency_format({'FR': 'M'}) == {('FR',): 'M'}

    def test_dict_tuple_key_kept(self):
        assert validate_target_frequency_format({('FR', 'a'): 'M'}) == {('FR', 'a'): 'M'}

    def test_empty_dict(self):
        with pytest.raises(ValueError, match='target_frequency dict cannot be empty'):
            validate_target_frequency_format({})

    def test_non_string_frequency(self):
        with pytest.raises(ValueError, match="Frequency for entity 'FR' must be a string"):
            validate_target_frequency_format({'FR': 3})

    def test_invalid_frequencies_aggregated(self):
        with pytest.raises(ValueError) as exc:
            validate_target_frequency_format({'FR': 'zzz', 'DE': 'qqq'})
        message = str(exc.value)
        assert message.startswith('Invalid frequencies in target_frequency dict:')
        assert "Entity 'FR'" in message and "Entity 'DE'" in message

    @pytest.mark.parametrize('value', [3, None, ['M']])
    def test_wrong_type(self, value):
        with pytest.raises(TypeError, match='target_frequency must be a string or dict'):
            validate_target_frequency_format(value)


class _NoPredict:
    def fit(self, X, y=None):
        return self


class _NoFit:
    def predict(self, X):
        return X


class TestValidateEstimator:
    def test_none_admitted(self):
        validate_estimator(None)

    def test_estimator_admitted(self):
        validate_estimator(LinearRegression())

    def test_dict_admitted(self):
        validate_estimator({'a': LinearRegression(), '__default__': LinearRegression()})

    def test_missing_fit(self):
        with pytest.raises(ValueError, match="estimator must have a 'fit' method, got _NoFit"):
            validate_estimator(_NoFit())

    def test_missing_predict(self):
        with pytest.raises(
            ValueError, match="estimator must have a 'predict' method, got _NoPredict"
        ):
            validate_estimator(_NoPredict())

    def test_empty_dict(self):
        with pytest.raises(ValueError, match='estimator dict cannot be empty'):
            validate_estimator({})

    def test_dict_missing_fit(self):
        with pytest.raises(ValueError, match="Estimator for 'a' must have a 'fit' method"):
            validate_estimator({'a': _NoFit()})

    def test_dict_missing_predict(self):
        with pytest.raises(ValueError, match="Estimator for 'a' must have a 'predict' method"):
            validate_estimator({'a': _NoPredict()})


class TestValidateAdditiveTransformer:
    def test_admitted(self):
        validate_additive_transformer(StandardScaler())

    def test_all_missing(self):
        with pytest.raises(ValueError) as exc:
            validate_additive_transformer(object())
        assert str(exc.value) == (
            'additive_transformer must have methods: fit_transform, inverse_transform. '
            'Missing: fit_transform, inverse_transform. Got object'
        )

    def test_partially_missing(self):
        class OnlyFitTransform:
            def fit_transform(self, X):
                return X

        with pytest.raises(ValueError, match='Missing: inverse_transform. Got OnlyFitTransform'):
            validate_additive_transformer(OnlyFitTransform())


class TestValidateLiteralSorted:
    def test_message_lists_sorted_values_as_list(self):
        with pytest.raises(ValueError) as exc:
            validate_literal('mode', 'x', ('b', 'a'), sort=True)
        assert str(exc.value) == "mode must be one of ['a', 'b'], got 'x'"


class TestValidateMinCvTrainSize:
    @pytest.mark.parametrize('value', [1, 10, np.int64(3)])
    def test_admitted(self, value):
        validate_min_cv_train_size(value)

    @pytest.mark.parametrize('value', [True, np.bool_(True), 2.0, np.float64(2.0), '3', None])
    def test_wrong_type(self, value):
        with pytest.raises(TypeError, match='min_cv_train_size must be an int, got'):
            validate_min_cv_train_size(value)

    @pytest.mark.parametrize('value', [0, -1, np.int64(0)])
    def test_below_one(self, value):
        with pytest.raises(ValueError, match='min_cv_train_size must be >= 1, got'):
            validate_min_cv_train_size(value)
