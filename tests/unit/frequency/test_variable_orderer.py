"""Tests for tsforecast.frequency.variable_orderer.

'frequency' and 'cv' orders, deterministic alphabetical tie-break
(independent of insertion order / input columns), polymorphic sklearn
``cv`` parameter (replacing ``cv_n_splits``), ``cv`` resolved at ``fit``
only.
"""
# Modules de base
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import FitFailedWarning, NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, TimeSeriesSplit

# Objet testé
from tsforecast.crossvals.time_series import TSOutOfSampleSplit
from tsforecast.frequency.variable_orderer import VariableOrderer, VariableSpec
from tests.support.estimators import ConstantEstimator, FailingEstimator, SpyEstimator

# Scoring à valeur d'or calculable à la main avec ConstantEstimator(0.0)
_MAE = 'neg_mean_absolute_error'


# Fabrique de spécifications de variable pour reference_timeseries : a1/a2
# annuelles, q1 trimestrielle, m1 mensuelle (jamais imputée, exclue de
# "variables" mais gardée comme covariable dans X)
def _reference_variables():
    """Build the {key: VariableSpec} mapping for a1/a2/q1 of reference_timeseries."""
    return {
        'a1': VariableSpec(name='a1', frequency='Y'),
        'a2': VariableSpec(name='a2', frequency='Y'),
        'q1': VariableSpec(name='q1', frequency='Q'),
    }


# Simulateur de "cross_val_score" : renvoie un score constant par variable
# scorée (identifiée par le nom de la cible "y.name"), sans jamais entraîner
# de modèle réel — les tests d'ordre n'ont besoin que du score, pas de sa
# provenance numérique
def _fake_cross_val_score(scores_by_name, n_folds=2):
    """Build a cross_val_score stand-in returning a fixed score per target name.

    Args:
        scores_by_name: Mapping variable name -> score to inject.
        n_folds: Number of (identical) fold scores returned, cosmetic only.

    Returns:
        Callable with the ``cross_val_score`` positional/keyword signature.
    """
    def _fake(estimator, X, y, cv=None, scoring=None, error_score=None):
        del estimator, X, cv, scoring, error_score
        return np.full(n_folds, scores_by_name[y.name])
    return _fake


# Frame à deux variables dont les valeurs d'or de CV sont calculables à la main :
# avec ConstantEstimator(0.0) et le scoring MAE, le score d'un pli est l'opposé
# de la moyenne des |y| du pli ; a = 1..12 et b = 0.1 * a
def _two_variable_frame(n_rows=12):
    """Build a frame with targets ``a`` (1..n) and ``b`` (0.1 * a)."""
    values = np.arange(1, n_rows + 1, dtype=float)
    return pd.DataFrame({'a': values, 'b': values * 0.1}, index=pd.RangeIndex(n_rows))


def _two_variable_specs(frequency='M'):
    """Build the {key: VariableSpec} mapping of ``_two_variable_frame``."""
    return {
        'a': VariableSpec(name='a', frequency=frequency),
        'b': VariableSpec(name='b', frequency=frequency),
    }


def _mae_orderer(cv=4, min_cv_train_size=4):
    """Build a fitted 'cv' orderer scoring by negative MAE."""
    return VariableOrderer(
        fit_predict_order='cv', cv=cv, cv_scoring=_MAE,
        min_cv_train_size=min_cv_train_size,
    ).fit()


# Estimateur dont l'ajustement échoue sur le seul pli dont le test contient la
# ligne 0 (l'entraînement de ce pli ne contient pas l'index 0)
class _FailsOnFirstFold(ConstantEstimator):
    """ConstantEstimator whose fit fails when row 0 is not in its training set."""

    def fit(self, X, y):
        """Fail on the fold whose test set holds row 0."""
        if 0 not in X.index:
            raise RuntimeError('deliberate failure on the first fold')
        return self


class TestFrequencyOrder:
    """'frequency' order: lowest frequency first."""

    def test_frequency_order_on_reference_ts(self, reference_timeseries):
        """The order is a1, a2, q1 (Y before Q, alphabetical tie-break)."""
        del reference_timeseries  # les fréquences suffisent, données non lues
        orderer = VariableOrderer(fit_predict_order='frequency').fit()
        assert orderer.order(_reference_variables()) == ['a1', 'a2', 'q1']

    def test_entity_count_breaks_frequency_ties_ascending(self):
        """At equal frequency, the group with the FEWEST entities comes first.

        Behaviour opposite to hfi (which sorted by decreasing number of
        entities).
        """
        orderer = VariableOrderer(fit_predict_order='frequency').fit()
        variables = {
            'many': VariableSpec(name='many', frequency='M', entities=(('FR',), ('DE',), ('IT',))),
            'few': VariableSpec(name='few', frequency='M', entities=(('FR',),)),
        }
        assert orderer.order(variables) == ['few', 'many']


class TestCVOrder:
    """Ordre 'cv' : meilleur score en premier."""

    def test_cv_order_on_reference_ts(self, reference_timeseries):
        """Scores q1=-0.08, a2=-0.15, a1=-0.15 -> ordre q1, a1, a2."""
        orderer = VariableOrderer(
            fit_predict_order='cv', cv=2, min_cv_train_size=2,
        ).fit()
        scores = {'q1': -0.08, 'a2': -0.15, 'a1': -0.15}
        with patch(
            'tsforecast.frequency.variable_orderer.cross_val_score',
            side_effect=_fake_cross_val_score(scores),
        ):
            ordered = orderer.order(
                _reference_variables(),
                X=reference_timeseries,
                estimator=LinearRegression(),
            )
        assert ordered == ['q1', 'a1', 'a2']


class TestAlphabeticalTiebreak:
    """Intra-stage determinism: never the order of the input columns / keys."""

    def test_alphabetical_tiebreak_ignores_column_order_frequency(self, reference_timeseries):
        """Permuter les clés de "variables" ne change pas l'ordre sous 'frequency'."""
        del reference_timeseries
        orderer = VariableOrderer(fit_predict_order='frequency').fit()

        forward = _reference_variables()
        reversed_dict = dict(reversed(list(_reference_variables().items())))

        assert orderer.order(forward) == orderer.order(reversed_dict) == ['a1', 'a2', 'q1']

    def test_alphabetical_tiebreak_ignores_column_order_cv(self, reference_timeseries):
        """Permuter les colonnes de X et les clés de "variables" ne change pas l'ordre sous 'cv'."""
        orderer = VariableOrderer(
            fit_predict_order='cv', cv=2, min_cv_train_size=2,
        ).fit()
        scores = {'q1': -0.08, 'a2': -0.15, 'a1': -0.15}

        forward = _reference_variables()
        reversed_dict = dict(reversed(list(_reference_variables().items())))
        permuted_X = reference_timeseries[['a2', 'q1', 'm1', 'a1']]

        with patch(
            'tsforecast.frequency.variable_orderer.cross_val_score',
            side_effect=_fake_cross_val_score(scores),
        ):
            order_forward = orderer.order(
                forward, X=reference_timeseries, estimator=LinearRegression(),
            )
            order_reversed = orderer.order(
                reversed_dict, X=permuted_X, estimator=LinearRegression(),
            )

        assert order_forward == order_reversed == ['q1', 'a1', 'a2']


class TestCVParameter:
    """Polymorphic sklearn ``cv`` parameter, replacing ``cv_n_splits``."""

    def test_cv_accepts_int_splitter_and_iterable(self):
        """Un entier, un splitter, et un itérable de splits sont tous acceptés."""
        by_int = VariableOrderer(cv=3).fit()
        assert by_int.cv_.get_n_splits() == 3

        by_splitter = VariableOrderer(cv=KFold(n_splits=4)).fit()
        assert by_splitter.cv_.get_n_splits() == 4

        splits = [
            (np.array([0, 1]), np.array([2])),
            (np.array([1, 2]), np.array([0])),
        ]
        by_iterable = VariableOrderer(cv=splits).fit()
        assert by_iterable.cv_.get_n_splits() == 2

    def test_cv_none_defaults_to_kfold_5_shuffled(self):
        """cv=None -> KFold(n_splits=5, shuffle=True, random_state=42)."""
        orderer = VariableOrderer().fit()
        assert isinstance(orderer.cv_, KFold)
        assert orderer.cv_.n_splits == 5
        assert orderer.cv_.shuffle is True
        assert orderer.cv_.random_state == 42

    def test_cv_not_resolved_at_init(self):
        """check_cv n'est pas appelé avant fit ; "cv" reste inchangé dans get_params."""
        orderer = VariableOrderer(cv=5)
        assert not hasattr(orderer, 'cv_')
        assert orderer.get_params()['cv'] == 5

        with pytest.raises(NotFittedError):
            orderer.order(_reference_variables())

        orderer.fit()
        # "cv" (paramètre reçu) reste identique après fit ; seul "cv_" apparaît
        assert orderer.get_params()['cv'] == 5
        assert orderer.cv_.get_n_splits() == 5

    def test_min_cv_train_size_warning_emitted_once(self):
        """min_cv_train_size < n_splits effectifs -> un UserWarning unique, au fit."""
        with pytest.warns(UserWarning, match=r"min_cv_train_size"):
            orderer = VariableOrderer(cv=5, min_cv_train_size=2).fit()
        assert orderer.cv_.get_n_splits() == 5

    def test_cv_n_splits_is_gone(self):
        """No parameter named "cv_n_splits" is accepted (absorbed by "cv")."""
        assert 'cv_n_splits' not in VariableOrderer().get_params()
        with pytest.raises(TypeError):
            VariableOrderer(cv_n_splits=5)


class TestPrebuiltScoringSets:
    """Jeux de scoring fournis par l'appelant : ce qui est scoré est ce qui sera ajusté."""

    @staticmethod
    def _sets():
        """Trois jeux (X, y) disjoints de ceux que "X" produirait."""
        index = pd.RangeIndex(12)
        return {
            'a1': (
                pd.DataFrame({'f1': np.arange(12.0)}, index=index),
                pd.Series(np.arange(12.0), index=index, name='a1'),
            ),
            'a2': (
                pd.DataFrame({'f2': np.arange(12.0)}, index=index),
                pd.Series(np.arange(12.0), index=index, name='a2'),
            ),
            'q1': (
                pd.DataFrame({'f3': np.arange(12.0)}, index=index),
                pd.Series(np.arange(12.0), index=index, name='q1'),
            ),
        }

    def test_training_sets_take_precedence_over_X(self, reference_timeseries):
        """Les couples fournis sont scorés tels quels ; "X" n'est jamais lu."""
        orderer = VariableOrderer(
            fit_predict_order='cv', cv=2, min_cv_train_size=2,
        ).fit()
        sets = self._sets()
        seen = {}

        def _spy(estimator, X, y, cv=None, scoring=None, error_score=None):
            del estimator, cv, scoring, error_score
            seen[y.name] = list(X.columns)
            return np.full(2, {'q1': -0.08, 'a2': -0.15, 'a1': -0.20}[y.name])

        with patch(
            'tsforecast.frequency.variable_orderer.cross_val_score',
            side_effect=_spy,
        ):
            ordered = orderer.order(
                _reference_variables(),
                X=reference_timeseries,
                estimator=LinearRegression(),
                training_sets=sets,
            )

        # Colonnes vues par la CV : celles des jeux fournis, jamais celles
        # que "X" aurait produites (m1 et les autres cibles)
        assert seen == {'a1': ['f1'], 'a2': ['f2'], 'q1': ['f3']}
        assert ordered == ['q1', 'a2', 'a1']

    def test_training_sets_alone_need_no_X(self):
        """Sans "X", des jeux fournis suffisent : aucune ValueError."""
        orderer = VariableOrderer(
            fit_predict_order='cv', cv=2, min_cv_train_size=2,
        ).fit()
        with patch(
            'tsforecast.frequency.variable_orderer.cross_val_score',
            side_effect=_fake_cross_val_score({'q1': -0.08, 'a2': -0.15, 'a1': -0.20}),
        ):
            ordered = orderer.order(
                _reference_variables(),
                estimator=LinearRegression(),
                training_sets=self._sets(),
            )
        assert ordered == ['q1', 'a2', 'a1']

    def test_missing_X_and_missing_training_sets_raise(self):
        """Ni "X" ni jeux fournis sous 'cv' : erreur explicite."""
        orderer = VariableOrderer(fit_predict_order='cv', cv=2).fit()
        with pytest.raises(ValueError, match='X is required'):
            orderer.order(_reference_variables(), estimator=LinearRegression())

    def test_variable_without_covariate_joins_the_fallback_group(self):
        """Un jeu sans aucune covariable rejoint le groupe de repli, pas la fin du groupe scoré."""
        orderer = VariableOrderer(
            fit_predict_order='cv', cv=2, min_cv_train_size=2,
        ).fit()
        sets = self._sets()
        # "q1" (trimestrielle, donc DERNIÈRE sous l'ordre 'frequency') perd
        # toutes ses covariables : elle sera imputée par interpolation quel
        # que soit son rang
        index = sets['q1'][1].index
        sets['q1'] = (pd.DataFrame(index=index), sets['q1'][1])

        with patch(
            'tsforecast.frequency.variable_orderer.cross_val_score',
            side_effect=_fake_cross_val_score({'a2': -0.15, 'a1': -0.20}),
        ):
            ordered = orderer.order(
                _reference_variables(),
                estimator=LinearRegression(),
                training_sets=sets,
            )
        # a2 et a1 scorées, q1 en repli : le groupe de repli passe après
        assert ordered == ['a2', 'a1', 'q1']

    def test_too_short_set_joins_the_fallback_group(self):
        """Un jeu plus court que min_cv_train_size bascule aussi en repli."""
        orderer = VariableOrderer(
            fit_predict_order='cv', cv=2, min_cv_train_size=10,
        ).fit()
        sets = self._sets()
        sets['a1'] = (sets['a1'][0].iloc[:3], sets['a1'][1].iloc[:3])

        with patch(
            'tsforecast.frequency.variable_orderer.cross_val_score',
            side_effect=_fake_cross_val_score({'a2': -0.15, 'q1': -0.08}),
        ):
            ordered = orderer.order(
                _reference_variables(),
                estimator=LinearRegression(),
                training_sets=sets,
            )
        assert ordered == ['q1', 'a2', 'a1']

    def test_uncovered_key_joins_the_fallback_group(self):
        """Une clé non couverte par les jeux fournis n'est pas scorée."""
        orderer = VariableOrderer(
            fit_predict_order='cv', cv=2, min_cv_train_size=2,
        ).fit()
        sets = self._sets()
        del sets['a1']

        with patch(
            'tsforecast.frequency.variable_orderer.cross_val_score',
            side_effect=_fake_cross_val_score({'a2': -0.15, 'q1': -0.08}),
        ):
            ordered = orderer.order(
                _reference_variables(),
                estimator=LinearRegression(),
                training_sets=sets,
            )
        assert ordered == ['q1', 'a2', 'a1']


class TestCVFallbacksAndLogging:
    """CV edge cases: -inf sentinels, logging of failed folds."""

    def test_unscorable_variables_sorted_last(self, reference_timeseries):
        """Une variable sans estimateur résolu (-inf) passe après les variables scorées."""
        orderer = VariableOrderer(
            fit_predict_order='cv', cv=2, min_cv_train_size=2,
        ).fit()
        scores = {'q1': -0.08, 'a2': -0.15}
        estimator = {'q1': LinearRegression(), 'a2': LinearRegression()}
        # "a1" n'a ni entrée dédiée ni '__default__' : estimateur résolu à
        # None -> score -inf, quel que soit min_cv_train_size
        with patch(
            'tsforecast.frequency.variable_orderer.cross_val_score',
            side_effect=_fake_cross_val_score(scores),
        ):
            ordered = orderer.order(
                _reference_variables(), X=reference_timeseries, estimator=estimator,
            )
        assert ordered == ['q1', 'a2', 'a1']

    def test_all_folds_failed_is_logged(self, reference_timeseries):
        """Tous les plis en échec (scores NaN) -> score -inf et message journalisé."""
        orderer = VariableOrderer(
            fit_predict_order='cv', cv=2, min_cv_train_size=2,
        ).fit()

        def _fake_cross_val_score_with_failure(estimator, X, y, cv=None, scoring=None, error_score=None):
            del estimator, X, cv, scoring, error_score
            if y.name == 'a1':
                return np.array([np.nan, np.nan])
            return np.full(2, {'q1': -0.08, 'a2': -0.15}[y.name])

        messages = []
        with patch(
            'tsforecast.frequency.variable_orderer.cross_val_score',
            side_effect=_fake_cross_val_score_with_failure,
        ):
            ordered = orderer.order(
                _reference_variables(),
                X=reference_timeseries,
                estimator=LinearRegression(),
                log=messages.append,
            )

        assert ordered == ['q1', 'a2', 'a1']
        assert len(messages) == 1
        assert "a1" in messages[0]


class TestFrequencyOrderEdgeCases:
    """Ordre 'frequency' : toute la chaîne de départage (fréquence, entités, nom)."""

    @pytest.mark.parametrize(
        'frequencies, expected',
        [
            pytest.param(['M', 'D', 'Q', 'Y'], ['v3', 'v2', 'v0', 'v1'], id='Y-Q-M-D'),
            pytest.param(['D', 'M'], ['v1', 'v0'], id='M-before-D'),
        ],
    )
    def test_lowest_frequency_comes_first(self, frequencies, expected):
        """The lowest frequency is processed first, whatever the insertion order."""
        # Une clé "v<i>" par fréquence de la liste, dans l'ordre d'insertion donné
        variables = {
            f'v{i}': VariableSpec(name=f'v{i}', frequency=freq)
            for i, freq in enumerate(frequencies)
        }
        orderer = VariableOrderer(fit_predict_order='frequency').fit()
        assert orderer.order(variables) == expected

    def test_divergent_entity_frequencies_use_group_keys(self):
        """One variable name split into several (name, frequency) groups is ordered by frequency."""
        # Cas PANEL-F : `v` vaut Y pour FR, Q pour DE, M pour IT ; la clé est
        # le couple (nom, fréquence), la spec porte le nom seul
        variables = {
            ('v', 'M'): VariableSpec(name='v', frequency='M', entities=(('IT',),)),
            ('v', 'Y'): VariableSpec(name='v', frequency='Y', entities=(('FR',),)),
            ('v', 'Q'): VariableSpec(name='v', frequency='Q', entities=(('DE',),)),
        }
        orderer = VariableOrderer(fit_predict_order='frequency').fit()
        assert orderer.order(variables) == [('v', 'Y'), ('v', 'Q'), ('v', 'M')]

    def test_entity_count_then_name_break_remaining_ties(self):
        """Same frequency: fewer entities first, then alphabetical name."""
        variables = {
            'zz': VariableSpec(name='zz', frequency='M', entities=(('FR',),)),
            'aa': VariableSpec(name='aa', frequency='M', entities=(('FR',), ('DE',))),
            'bb': VariableSpec(name='bb', frequency='M', entities=(('IT',),)),
        }
        orderer = VariableOrderer(fit_predict_order='frequency').fit()
        # bb et zz ont 1 entité (départage par nom), aa en a 2
        assert orderer.order(variables) == ['bb', 'zz', 'aa']

    def test_empty_entities_equal_none(self):
        """An empty entity tuple counts like ``None``: zero entities."""
        variables = {
            'with_none': VariableSpec(name='with_none', frequency='M'),
            'with_empty': VariableSpec(name='with_empty', frequency='M', entities=()),
            'with_one': VariableSpec(name='with_one', frequency='M', entities=(('FR',),)),
        }
        orderer = VariableOrderer(fit_predict_order='frequency').fit()
        # Tie-break nom entre les deux groupes sans entité, puis le groupe à 1 entité
        assert orderer.order(variables) == ['with_empty', 'with_none', 'with_one']

    def test_tiebreak_uses_spec_name_not_key(self):
        """The alphabetical tie-break reads ``VariableSpec.name``, never the dict key."""
        # Clés dans l'ordre inverse des noms
        variables = {
            'k1': VariableSpec(name='zebra', frequency='M'),
            'k2': VariableSpec(name='apple', frequency='M'),
        }
        orderer = VariableOrderer(fit_predict_order='frequency').fit()
        assert orderer.order(variables) == ['k2', 'k1']

    def test_unsupported_frequency_raises(self):
        """A frequency unknown to the frequency table surfaces as a ValueError."""
        variables = {
            'a': VariableSpec(name='a', frequency='M'),
            'b': VariableSpec(name='b', frequency='not-a-frequency'),
        }
        orderer = VariableOrderer(fit_predict_order='frequency').fit()
        with pytest.raises(ValueError):
            orderer.order(variables)

    @pytest.mark.parametrize(
        'variables',
        [
            pytest.param({}, id='empty'),
            pytest.param({'only': VariableSpec(name='only', frequency='M')}, id='single'),
        ],
    )
    def test_zero_or_one_variable_returns_early(self, variables):
        """Zero or one variable is returned as-is, whatever the order mode."""
        orderer = VariableOrderer(fit_predict_order='cv').fit()
        # Aucune donnée requise : la sortie anticipée précède la vérification de X
        assert orderer.order(variables) == list(variables)

    def test_frequency_order_leaves_diagnostics_empty(self):
        """Under 'frequency', scores_ and fallback_keys_ exist and are empty."""
        orderer = VariableOrderer(fit_predict_order='frequency').fit()
        orderer.order(_reference_variables())
        assert (orderer.scores_, orderer.fallback_keys_) == ({}, [])

    def test_order_requires_fit_even_for_frequency(self):
        """``order`` before ``fit`` raises NotFittedError under both modes."""
        with pytest.raises(NotFittedError):
            VariableOrderer(fit_predict_order='frequency').order(_reference_variables())


class TestInitValidation:
    """Validation de forme des paramètres à l'initialisation (B3, sans transformation)."""

    @pytest.mark.parametrize(
        'kwargs, error, message',
        [
            pytest.param({'fit_predict_order': 'random'}, ValueError,
                         "fit_predict_order must be 'frequency' or 'cv', got 'random'",
                         id='bad-order'),
            pytest.param({'cv': 1}, ValueError,
                         'cv must be >= 2 when given as an int, got 1', id='cv-one'),
            pytest.param({'cv': 0}, ValueError, 'cv must be >= 2', id='cv-zero'),
            pytest.param({'cv': -3}, ValueError, 'cv must be >= 2', id='cv-negative'),
            pytest.param({'cv': True}, TypeError, 'cv must not be a bool', id='cv-bool'),
            pytest.param({'cv': 'abc'}, TypeError, 'cv must be None, an int >= 2', id='cv-string'),
            pytest.param({'cv': 2.5}, TypeError, 'got float', id='cv-float'),
            pytest.param({'cv': object()}, TypeError, 'got object', id='cv-object'),
            pytest.param({'cv_scoring': 5}, TypeError,
                         'cv_scoring must be a str or a callable, got int', id='scoring-int'),
            pytest.param({'min_cv_train_size': 0}, ValueError,
                         'min_cv_train_size must be >= 1, got 0', id='min-size-zero'),
        ],
    )
    def test_invalid_parameter_is_rejected_at_init(self, kwargs, error, message):
        """Each malformed parameter raises at construction, with an explicit message."""
        with pytest.raises(error, match=message):
            VariableOrderer(**kwargs)

    @pytest.mark.parametrize(
        'cv',
        [
            pytest.param(None, id='none'),
            pytest.param(2, id='int'),
            pytest.param(KFold(3), id='kfold'),
            pytest.param(TimeSeriesSplit(3), id='timeseries-split'),
            pytest.param(TSOutOfSampleSplit(n_splits=3), id='ts-out-of-sample'),
            pytest.param([(np.array([0, 1]), np.array([2]))], id='list-of-splits'),
        ],
    )
    def test_valid_cv_is_accepted_at_init(self, cv):
        """None, an int >= 2, any splitter and an iterable of splits are accepted."""
        assert VariableOrderer(cv=cv).cv is cv


class TestCVResolution:
    """Résolution de ``cv`` en ``cv_`` au seul ``fit``."""

    @pytest.mark.parametrize(
        'cv',
        [
            pytest.param(TimeSeriesSplit(n_splits=3), id='timeseries-split'),
            pytest.param(TSOutOfSampleSplit(n_splits=3, test_size=3), id='ts-out-of-sample'),
        ],
    )
    def test_splitter_is_used_as_is(self, cv):
        """A user splitter is kept unchanged (same object) and never shuffled."""
        orderer = VariableOrderer(cv=cv, min_cv_train_size=3).fit()
        assert orderer.cv_ is cv

    def test_default_split_is_reproducible(self):
        """cv=None / int build a shuffled KFold whose folds repeat across fits (seed 42)."""
        frame = _two_variable_frame()
        first = [t.tolist() for _, t in VariableOrderer(cv=3).fit().cv_.split(frame)]
        second = [t.tolist() for _, t in VariableOrderer(cv=3).fit().cv_.split(frame)]
        # Les plis ne sont pas les blocs contigus d'un KFold non mélangé
        assert first == second != [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    def test_int_cv_gives_n_shuffled_splits(self):
        """An int cv gives KFold(n_splits=cv, shuffle=True, random_state=42)."""
        cv_ = VariableOrderer(cv=7, min_cv_train_size=7).fit().cv_
        assert (cv_.n_splits, cv_.shuffle, cv_.random_state) == (7, True, 42)

    def test_numpy_int_cv_is_an_int(self):
        """A numpy integer (grid built by np.arange) resolves like an int."""
        cv_ = VariableOrderer(cv=np.int64(4), min_cv_train_size=4).fit().cv_
        assert (cv_.n_splits, cv_.shuffle, cv_.random_state) == (4, True, 42)

    def test_iterable_of_splits_is_wrapped(self):
        """An iterable of splits is wrapped by check_cv and yields exactly those splits."""
        splits = [
            (np.array([0, 1, 2]), np.array([3])),
            (np.array([0, 1, 3]), np.array([2])),
        ]
        cv_ = VariableOrderer(cv=splits, min_cv_train_size=2).fit().cv_
        produced = [(tr.tolist(), te.tolist()) for tr, te in cv_.split(np.zeros(4))]
        assert produced == [([0, 1, 2], [3]), ([0, 1, 3], [2])]

    def test_no_warning_when_min_size_reaches_n_splits(self, recwarn):
        """min_cv_train_size >= n_splits emits no warning at fit."""
        VariableOrderer(cv=5, min_cv_train_size=5).fit()
        assert not [w for w in recwarn if 'min_cv_train_size' in str(w.message)]

    def test_warning_names_both_values_and_is_independent_of_order(self):
        """The cross-warning also fires under 'frequency' and names both values."""
        with pytest.warns(UserWarning, match=r"\(2\).*\(5\)"):
            VariableOrderer(fit_predict_order='frequency', cv=5, min_cv_train_size=2).fit()


class TestCVScoresAndSplitters:
    """'cv' on a real cross_val_score: exact scores_, all splitters."""

    def test_scores_are_negated_errors_best_first(self):
        """Greater is better: the smaller error (b) ranks first, whatever the alphabet."""
        orderer = _mae_orderer()
        ordered = orderer.order(
            _two_variable_specs(), X=_two_variable_frame(), estimator=ConstantEstimator(0.0),
        )
        # Valeur d'or : 4 plis de 3 lignes, donc la moyenne des plis est la moyenne
        # globale des |y| (quel que soit le mélange) : a -> 6.5, b -> 0.65
        assert ordered == ['b', 'a']
        assert orderer.scores_ == pytest.approx({'b': -0.65, 'a': -6.5})

    def test_scores_follow_ranking_order(self):
        """scores_ lists the scored keys in the returned order."""
        orderer = _mae_orderer()
        orderer.order(
            _two_variable_specs(), X=_two_variable_frame(), estimator=ConstantEstimator(0.0),
        )
        assert list(orderer.scores_) == ['b', 'a']

    @pytest.mark.parametrize(
        'cv',
        [
            pytest.param(TimeSeriesSplit(n_splits=3), id='timeseries-split'),
            pytest.param(TSOutOfSampleSplit(n_splits=3, test_size=3), id='ts-out-of-sample'),
        ],
    )
    def test_temporal_splitters_give_expected_scores(self, cv):
        """Expanding-window splitters: tests 4-6, 7-9, 10-12 -> mean(5, 8, 11) = 8 for a."""
        orderer = _mae_orderer(cv=cv, min_cv_train_size=3)
        orderer.order(
            _two_variable_specs(), X=_two_variable_frame(), estimator=ConstantEstimator(0.0),
        )
        # Valeur d'or : plis de test {3,4,5}, {6,7,8}, {9,10,11} (positions), y = pos + 1
        assert orderer.scores_ == pytest.approx({'b': -0.8, 'a': -8.0})

    def test_iterable_of_splits_gives_expected_scores(self):
        """An explicit list of splits: tests {1..4} and {5..8} -> mean(2.5, 6.5) = 4.5."""
        splits = [
            (np.arange(4, 12), np.arange(0, 4)),
            (np.concatenate([np.arange(0, 4), np.arange(8, 12)]), np.arange(4, 8)),
        ]
        orderer = _mae_orderer(cv=splits, min_cv_train_size=2)
        orderer.order(
            _two_variable_specs(), X=_two_variable_frame(), estimator=ConstantEstimator(0.0),
        )
        assert orderer.scores_ == pytest.approx({'b': -0.45, 'a': -4.5})

    def test_real_regression_ranks_predictable_variable_first(self):
        """A target exactly linear in a covariate outranks a pure-noise target."""
        # Construction : "good" = 3 * c + 1 (R² = 1) ; "bad" = bruit gaussien
        # graîné, indépendant de c
        rng = np.random.default_rng(0)
        c = np.arange(30.0)
        frame = pd.DataFrame({
            'c': c, 'good': 3 * c + 1, 'bad': rng.normal(size=30),
        })
        variables = {
            'bad': VariableSpec(name='bad', frequency='M'),
            'good': VariableSpec(name='good', frequency='M'),
        }
        orderer = VariableOrderer(
            fit_predict_order='cv', cv=3, cv_scoring='r2', min_cv_train_size=3,
        ).fit()
        ordered = orderer.order(variables, X=frame, estimator=LinearRegression())
        assert (ordered, round(orderer.scores_['good'], 6)) == (['good', 'bad'], 1.0)

    def test_callable_scorer_is_accepted(self):
        """A callable scorer ``(estimator, X, y)`` is resolved by check_scoring."""
        def scorer(estimator, X, y):
            # Score de pli : opposé de la moyenne des |y|, identique au MAE
            return -float(np.abs(np.asarray(y)).mean())

        orderer = VariableOrderer(
            fit_predict_order='cv', cv=4, cv_scoring=scorer, min_cv_train_size=4,
        ).fit()
        orderer.order(
            _two_variable_specs(), X=_two_variable_frame(), estimator=ConstantEstimator(0.0),
        )
        assert orderer.scores_ == pytest.approx({'b': -0.65, 'a': -6.5})

    def test_unknown_scoring_string_raises(self):
        """An unknown registry string surfaces sklearn's explicit ValueError."""
        orderer = VariableOrderer(
            fit_predict_order='cv', cv=4, cv_scoring='not_a_scorer', min_cv_train_size=4,
        ).fit()
        with pytest.raises(ValueError, match='not_a_scorer'):
            orderer.order(
                _two_variable_specs(), X=_two_variable_frame(),
                estimator=ConstantEstimator(0.0),
            )


class TestScoringRows:
    """Lignes scorées : cible observée, masque, seuil ``min_cv_train_size``."""

    def test_scoring_mask_restricts_rows(self):
        """Only masked rows are scored: rows 6..11 (y = 7..12) -> mean 9.5."""
        frame = _two_variable_frame()
        mask = pd.Series(np.arange(12) >= 6, index=frame.index)
        orderer = _mae_orderer(cv=2, min_cv_train_size=2)
        orderer.order(
            _two_variable_specs(), X=frame, estimator=ConstantEstimator(0.0),
            scoring_mask=mask,
        )
        # 2 plis de 3 lignes : la moyenne des plis est la moyenne des 6 lignes
        assert orderer.scores_ == pytest.approx({'b': -0.95, 'a': -9.5})

    def test_unobserved_target_rows_are_dropped(self):
        """NaN target rows are not scored: b observed on rows 6..11 only -> mean(0.7..1.2)."""
        frame = _two_variable_frame()
        frame.loc[:5, 'b'] = np.nan
        orderer = _mae_orderer(cv=2, min_cv_train_size=2)
        orderer.order(
            _two_variable_specs(), X=frame, estimator=ConstantEstimator(0.0),
        )
        assert orderer.scores_['b'] == pytest.approx(-0.95)

    def test_input_frame_is_not_modified(self):
        """``order`` leaves X and the mask untouched."""
        frame = _two_variable_frame()
        frame.loc[:2, 'b'] = np.nan
        before = frame.copy()
        _mae_orderer().order(
            _two_variable_specs(), X=frame, estimator=ConstantEstimator(0.0),
        )
        pd.testing.assert_frame_equal(frame, before)

    @pytest.mark.parametrize(
        'n_rows, scored',
        [
            pytest.param(5, True, id='exactly-min-size'),
            pytest.param(4, False, id='one-below-min-size'),
        ],
    )
    def test_min_cv_train_size_boundary(self, n_rows, scored):
        """A variable with exactly min_cv_train_size rows is scored; one fewer falls back."""
        orderer = _mae_orderer(cv=2, min_cv_train_size=5)
        orderer.order(
            _two_variable_specs(), X=_two_variable_frame(n_rows),
            estimator=ConstantEstimator(0.0),
        )
        assert bool(orderer.scores_) is scored

    def test_target_column_is_excluded_from_its_own_features(self):
        """Each variable is scored on every column but its own."""
        seen = {}

        def _spy(estimator, X, y, cv=None, scoring=None, error_score=None):
            del estimator, cv, scoring, error_score
            seen[y.name] = list(X.columns)
            return np.zeros(2)

        frame = _two_variable_frame()
        frame['m1'] = 1.0
        with patch('tsforecast.frequency.variable_orderer.cross_val_score', side_effect=_spy):
            _mae_orderer(cv=2, min_cv_train_size=2).order(
                _two_variable_specs(), X=frame, estimator=ConstantEstimator(0.0),
            )
        assert seen == {'a': ['b', 'm1'], 'b': ['a', 'm1']}


class TestEstimatorResolution:
    """Résolution et clonage de l'estimateur par variable."""

    def test_mapping_default_entry_applies_to_other_variables(self):
        """An estimator mapping uses '__default__' for variables without a dedicated entry."""
        used = {}

        def _spy(estimator, X, y, cv=None, scoring=None, error_score=None):
            del X, cv, scoring, error_score
            used[y.name] = estimator.constant
            return np.zeros(2)

        estimator = {'a': ConstantEstimator(1.0), '__default__': ConstantEstimator(2.0)}
        with patch('tsforecast.frequency.variable_orderer.cross_val_score', side_effect=_spy):
            _mae_orderer(cv=2, min_cv_train_size=2).order(
                _two_variable_specs(), X=_two_variable_frame(), estimator=estimator,
            )
        assert used == {'a': 1.0, 'b': 2.0}

    def test_estimator_is_cloned_not_fitted(self):
        """The caller's estimator is never fitted: each score uses a clone."""
        estimator = SpyEstimator()
        _mae_orderer().order(_two_variable_specs(), X=_two_variable_frame(), estimator=estimator)
        assert not hasattr(estimator, 'fit_X_')

    def test_no_estimator_scores_minus_inf_without_cv(self):
        """estimator=None: every variable scores -inf, tie-broken by name, no CV run."""
        orderer = _mae_orderer()
        with patch('tsforecast.frequency.variable_orderer.cross_val_score') as cvs:
            ordered = orderer.order(_two_variable_specs(), X=_two_variable_frame())
        assert (ordered, orderer.scores_, orderer.fallback_keys_, cvs.called) == (
            ['a', 'b'], {'a': -np.inf, 'b': -np.inf}, [], False,
        )

    def test_unresolved_mapping_entry_scores_minus_inf(self):
        """A mapping without entry nor default leaves the variable at -inf, scored last."""
        orderer = _mae_orderer()
        ordered = orderer.order(
            _two_variable_specs(), X=_two_variable_frame(),
            estimator={'a': ConstantEstimator(0.0)},
        )
        assert (ordered, orderer.scores_['b']) == (['a', 'b'], -np.inf)


class TestFallbackGroup:
    """Fallback group: never merged with the scored group, exact diagnostics."""

    @staticmethod
    def _sets(n_rows=12):
        """Three (X, y) pairs with one covariate each, y = 1..n."""
        index = pd.RangeIndex(n_rows)
        y = pd.Series(np.arange(1.0, n_rows + 1), index=index)
        return {
            key: (pd.DataFrame({'f': np.arange(float(n_rows))}, index=index), y.rename(key))
            for key in ('a1', 'm1', 'q1')
        }

    @staticmethod
    def _specs():
        """a1 yearly, q1 quarterly, m1 monthly."""
        return {
            'm1': VariableSpec(name='m1', frequency='M'),
            'q1': VariableSpec(name='q1', frequency='Q'),
            'a1': VariableSpec(name='a1', frequency='Y'),
        }

    def test_fallback_keys_follow_frequency_order(self):
        """Without covariate everywhere: all fallback, lowest frequency first, scores_ empty."""
        sets = {
            key: (pd.DataFrame(index=x.index), y) for key, (x, y) in self._sets().items()
        }
        orderer = _mae_orderer(cv=2, min_cv_train_size=2)
        ordered = orderer.order(
            self._specs(), estimator=ConstantEstimator(0.0), training_sets=sets,
        )
        assert (ordered, orderer.fallback_keys_, orderer.scores_) == (
            ['a1', 'q1', 'm1'], ['a1', 'q1', 'm1'], {},
        )

    def test_scored_group_precedes_fallback_group(self):
        """A scored low-score variable still precedes a fallback one, whatever the frequency."""
        sets = self._sets()
        # a1 (annuelle, donc première sous 'frequency') perd sa covariable : repli
        sets['a1'] = (pd.DataFrame(index=sets['a1'][1].index), sets['a1'][1])
        orderer = _mae_orderer(cv=2, min_cv_train_size=2)
        ordered = orderer.order(
            self._specs(), estimator=ConstantEstimator(0.0), training_sets=sets,
        )
        # m1 et q1 ont le même y : même score, départage alphabétique (m1 < q1)
        assert (ordered, orderer.fallback_keys_, list(orderer.scores_)) == (
            ['m1', 'q1', 'a1'], ['a1'], ['m1', 'q1'],
        )

    def test_diagnostics_are_reset_between_calls(self):
        """scores_ and fallback_keys_ describe the last call only."""
        orderer = _mae_orderer(cv=2, min_cv_train_size=2)
        sets = self._sets()
        del sets['a1']
        orderer.order(self._specs(), estimator=ConstantEstimator(0.0), training_sets=sets)
        assert orderer.fallback_keys_ == ['a1']
        # Second appel : jeux complets, plus aucun repli
        orderer.order(
            self._specs(), estimator=ConstantEstimator(0.0), training_sets=self._sets(),
        )
        assert orderer.fallback_keys_ == []

    def test_early_return_resets_diagnostics(self):
        """A one-variable call clears the diagnostics of the previous call."""
        orderer = _mae_orderer(cv=2, min_cv_train_size=2)
        orderer.order(
            self._specs(), estimator=ConstantEstimator(0.0), training_sets=self._sets(),
        )
        orderer.order({'a1': self._specs()['a1']})
        assert (orderer.scores_, orderer.fallback_keys_) == ({}, [])

    def test_empty_training_sets_mapping_sends_everything_to_fallback(self):
        """An empty (non-None) training_sets needs no X and covers no key."""
        orderer = _mae_orderer(cv=2, min_cv_train_size=2)
        ordered = orderer.order(
            self._specs(), estimator=ConstantEstimator(0.0), training_sets={},
        )
        assert (ordered, orderer.fallback_keys_) == (['a1', 'q1', 'm1'], ['a1', 'q1', 'm1'])

    def test_empty_training_frames_fall_back(self):
        """Zero-row scoring sets fall back (0 < min_cv_train_size), no CV attempted."""
        sets = {
            key: (x.iloc[:0], y.iloc[:0]) for key, (x, y) in self._sets().items()
        }
        orderer = _mae_orderer(cv=2, min_cv_train_size=2)
        with patch('tsforecast.frequency.variable_orderer.cross_val_score') as cvs:
            ordered = orderer.order(
                self._specs(), estimator=ConstantEstimator(0.0), training_sets=sets,
            )
        assert (ordered, cvs.called) == (['a1', 'q1', 'm1'], False)

    def test_empty_working_frame_falls_back(self):
        """An empty X (columns, no rows) sends every variable to the fallback group."""
        orderer = _mae_orderer(cv=2, min_cv_train_size=2)
        ordered = orderer.order(
            _two_variable_specs(), X=_two_variable_frame().iloc[:0],
            estimator=ConstantEstimator(0.0),
        )
        assert (ordered, orderer.fallback_keys_) == (['a', 'b'], ['a', 'b'])

    def test_group_keys_with_tuples_are_returned_as_is(self):
        """Tuple (name, frequency) keys flow through scoring and fallback unchanged."""
        variables = {
            ('v', 'M'): VariableSpec(name='v', frequency='M'),
            ('v', 'Y'): VariableSpec(name='v', frequency='Y'),
            ('w', 'Q'): VariableSpec(name='w', frequency='Q'),
        }
        index = pd.RangeIndex(12)
        y = pd.Series(np.arange(1.0, 13), index=index)
        sets = {
            ('v', 'M'): (pd.DataFrame({'f': y}, index=index), y),
            ('w', 'Q'): (pd.DataFrame({'f': y}, index=index), y),
        }
        orderer = _mae_orderer(cv=2, min_cv_train_size=2)
        ordered = orderer.order(variables, estimator=ConstantEstimator(0.0), training_sets=sets)
        # Scorés : (v, M) puis (w, Q) (même score, départage par nom) ; repli : (v, Y)
        assert (ordered, orderer.fallback_keys_) == (
            [('v', 'M'), ('w', 'Q'), ('v', 'Y')], [('v', 'Y')],
        )


class TestFailingFolds:
    """Failed folds: -inf sentinel, log, mean of the successful folds."""

    def test_partial_failure_averages_successful_folds(self):
        """One failed fold of four: score = mean of the three others, with a sklearn warning."""
        orderer = _mae_orderer(cv=KFold(4), min_cv_train_size=4)
        specs = {'a': _two_variable_specs()['a'], 'b': _two_variable_specs()['b']}
        with pytest.warns(FitFailedWarning, match='1 fits failed out of a total of 4'):
            orderer.order(specs, X=_two_variable_frame(), estimator=_FailsOnFirstFold(0.0))
        # Valeur d'or : plis de test {1,2,3} (échec), {4,5,6}, {7,8,9}, {10,11,12}
        # -> a : -(5 + 8 + 11) / 3 = -8
        assert orderer.scores_['a'] == pytest.approx(-8.0)

    def test_nan_folds_are_ignored_but_logged_only_if_all_fail(self):
        """Some NaN folds: nanmean score, nothing logged."""
        def _fake(estimator, X, y, cv=None, scoring=None, error_score=None):
            del estimator, X, cv, scoring, error_score
            return np.array([-0.1, np.nan, -0.3])

        messages = []
        orderer = _mae_orderer(cv=2, min_cv_train_size=2)
        with patch('tsforecast.frequency.variable_orderer.cross_val_score', side_effect=_fake):
            orderer.order(
                _two_variable_specs(), X=_two_variable_frame(),
                estimator=ConstantEstimator(0.0), log=messages.append,
            )
        assert (orderer.scores_['a'], messages) == (pytest.approx(-0.2), [])

    def test_all_nan_without_log_callback_is_silent(self):
        """All folds NaN and log=None: -inf, no exception, no callback."""
        def _fake(estimator, X, y, cv=None, scoring=None, error_score=None):
            del estimator, X, y, cv, scoring, error_score
            return np.array([np.nan, np.nan])

        orderer = _mae_orderer(cv=2, min_cv_train_size=2)
        with patch('tsforecast.frequency.variable_orderer.cross_val_score', side_effect=_fake):
            ordered = orderer.order(
                _two_variable_specs(), X=_two_variable_frame(),
                estimator=ConstantEstimator(0.0),
            )
        assert (ordered, orderer.scores_) == (['a', 'b'], {'a': -np.inf, 'b': -np.inf})

    def test_one_log_message_per_failed_variable(self):
        """Every variable whose folds all failed is logged once, with the exact message."""
        def _fake(estimator, X, y, cv=None, scoring=None, error_score=None):
            del estimator, X, cv, scoring, error_score
            return np.array([np.nan]) if y.name != 'b' else np.array([-1.0])

        messages = []
        variables = {
            'a': VariableSpec(name='a', frequency='M'),
            'b': VariableSpec(name='b', frequency='M'),
            'c': VariableSpec(name='c', frequency='M'),
        }
        frame = _two_variable_frame()
        frame['c'] = frame['a']
        orderer = _mae_orderer(cv=2, min_cv_train_size=2)
        with patch('tsforecast.frequency.variable_orderer.cross_val_score', side_effect=_fake):
            ordered = orderer.order(
                variables, X=frame, estimator=ConstantEstimator(0.0), log=messages.append,
            )
        expected = (
            "[VariableOrderer] CV scoring failed on every fold for '{}': "
            "falling back to the lowest possible score."
        )
        assert (ordered, messages) == (['b', 'a', 'c'], [expected.format('a'), expected.format('c')])

    def test_failing_estimator_scores_minus_inf_and_logs(self):
        """A fit that always fails gives -inf and one log line, never an exception."""
        messages = []
        orderer = _mae_orderer(cv=2, min_cv_train_size=2)
        ordered = orderer.order(
            _two_variable_specs(), X=_two_variable_frame(),
            estimator=FailingEstimator(), log=messages.append,
        )
        assert (ordered, orderer.scores_, len(messages)) == (
            ['a', 'b'], {'a': -np.inf, 'b': -np.inf}, 2,
        )

    def test_unrelated_value_error_is_propagated(self):
        """A ValueError not about failed fits (here, more splits than rows) is not swallowed."""
        orderer = _mae_orderer(cv=20, min_cv_train_size=1)
        with pytest.raises(ValueError, match='greater than the number of samples'):
            orderer.order(
                _two_variable_specs(), X=_two_variable_frame(),
                estimator=ConstantEstimator(0.0),
            )


class TestDeterminism:
    """Reproductibilité et indifférence à l'ordre des colonnes d'entrée."""

    @staticmethod
    def _noisy_frame():
        """Four columns with distinct, seeded noise levels around a shared trend."""
        rng = np.random.default_rng(7)
        trend = np.arange(40.0)
        return pd.DataFrame({
            'a': trend + rng.normal(scale=1.0, size=40),
            'b': trend + rng.normal(scale=5.0, size=40),
            'c': trend + rng.normal(scale=10.0, size=40),
            'd': trend + rng.normal(scale=20.0, size=40),
        })

    def _order(self, frame, variables=None):
        """Order the four columns by cross-validated R²."""
        variables = variables or {
            name: VariableSpec(name=name, frequency='M') for name in 'abcd'
        }
        orderer = VariableOrderer(
            fit_predict_order='cv', cv=4, cv_scoring='r2', min_cv_train_size=4,
        ).fit()
        return orderer.order(variables, X=frame, estimator=LinearRegression()), orderer.scores_

    def test_repeated_calls_give_identical_scores(self):
        """Two calls on the same orderer return the same order and the same scores."""
        frame = self._noisy_frame()
        orderer = VariableOrderer(
            fit_predict_order='cv', cv=4, cv_scoring='r2', min_cv_train_size=4,
        ).fit()
        variables = {name: VariableSpec(name=name, frequency='M') for name in 'abcd'}
        first = orderer.order(variables, X=frame, estimator=LinearRegression())
        scores = dict(orderer.scores_)
        second = orderer.order(variables, X=frame, estimator=LinearRegression())
        assert (first, scores) == (second, orderer.scores_)

    def test_independent_orderers_agree(self):
        """Two fresh orderers (default seeded KFold) agree on order and scores."""
        frame = self._noisy_frame()
        assert self._order(frame) == self._order(frame)

    def test_permuted_columns_give_same_order_with_real_scores(self):
        """Permuting X columns and variable keys leaves order and scores unchanged."""
        frame = self._noisy_frame()
        order, scores = self._order(frame)
        permuted_variables = {
            name: VariableSpec(name=name, frequency='M') for name in 'dbca'
        }
        order_p, scores_p = self._order(frame[['c', 'a', 'd', 'b']], permuted_variables)
        # Régression linéaire invariante à l'ordre des colonnes à l'arrondi flottant près
        assert (order_p, scores_p) == (order, pytest.approx(scores, abs=1e-9))

    def test_tied_scores_break_alphabetically_whatever_the_input_order(self):
        """Equal scores (here three identical targets) are ordered by name."""
        frame = _two_variable_frame()
        frame['c'] = frame['a']
        names = ['c', 'a', 'b']
        # a et c ont exactement les mêmes valeurs : même score MAE (6.5)
        orderer = _mae_orderer()
        ordered = orderer.order(
            {n: VariableSpec(name=n, frequency='M') for n in names},
            X=frame[names], estimator=ConstantEstimator(0.0),
        )
        assert ordered == ['b', 'a', 'c']


class TestSklearnContract:
    """Contrat sklearn : ``get_params``, ``set_params``, ``clone``."""

    def test_get_params_exposes_the_four_constructor_parameters(self):
        """get_params lists exactly the constructor parameters, as received."""
        splitter = KFold(3)
        orderer = VariableOrderer(
            fit_predict_order='cv', cv=splitter, cv_scoring='r2', min_cv_train_size=7,
        )
        assert orderer.get_params() == {
            'fit_predict_order': 'cv', 'cv': splitter, 'cv_scoring': 'r2',
            'min_cv_train_size': 7,
        }

    def test_clone_drops_fitted_state(self):
        """A clone of a fitted orderer has the same parameters but no cv_."""
        fitted = VariableOrderer(cv=3, min_cv_train_size=3).fit()
        copy = clone(fitted)
        assert (copy.get_params(), hasattr(copy, 'cv_')) == (fitted.get_params(), False)

    def test_clone_with_splitter_and_iterable(self):
        """clone works with a splitter object and a list of splits as ``cv``."""
        splits = [(np.array([0, 1]), np.array([2]))]
        assert clone(VariableOrderer(cv=TimeSeriesSplit(3))).cv.n_splits == 3
        assert len(clone(VariableOrderer(cv=splits)).cv) == 1

    def test_set_params_then_fit_resolves_new_cv(self):
        """set_params followed by fit re-resolves cv_ from the new value."""
        orderer = VariableOrderer(cv=3, min_cv_train_size=3).fit()
        orderer.set_params(cv=4, min_cv_train_size=4).fit()
        assert orderer.cv_.get_n_splits() == 4

    def test_repr_round_trips_parameters(self):
        """repr shows the non-default parameters (BaseEstimator behaviour)."""
        assert 'fit_predict_order=\'cv\'' in repr(VariableOrderer(fit_predict_order='cv'))
