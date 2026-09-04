"""Tests for tsforecast.frequency.variable_orderer.

Focus §8 (§8.1 à §8.4) de [SPEC] high_frequency_imputer2_architecture.md :
ordres 'frequency' et 'cv', tie-break alphabétique déterministe (indépendant
de l'ordre d'insertion / des colonnes d'entrée), paramètre ``cv`` polymorphe
sklearn (remplaçant ``cv_n_splits``), résolution de ``cv`` au seul ``fit``.
"""
# Modules de base
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# Objet testé
from tsforecast.frequency.variable_orderer import VariableOrderer, VariableSpec


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


class TestFrequencyOrder:
    """Ordre 'frequency' : fréquence la plus basse d'abord (§8.1, §8.4)."""

    def test_frequency_order_on_reference_ts(self, reference_timeseries):
        """[SPEC] §8.4 : l'ordre est a1, a2, q1 (Y avant Q, tie-break alphabétique)."""
        del reference_timeseries  # les fréquences suffisent, données non lues
        orderer = VariableOrderer(fit_predict_order='frequency').fit()
        assert orderer.order(_reference_variables()) == ['a1', 'a2', 'q1']

    def test_entity_count_breaks_frequency_ties_ascending(self):
        """À fréquence égale, le groupe avec le MOINS d'entités passe en premier.

        Comportement inversé par rapport à hfi (qui triait par nombre
        d'entités décroissant) : voir [SPEC] §8.1, prompt point 1.
        """
        orderer = VariableOrderer(fit_predict_order='frequency').fit()
        variables = {
            'many': VariableSpec(name='many', frequency='M', entities=(('FR',), ('DE',), ('IT',))),
            'few': VariableSpec(name='few', frequency='M', entities=(('FR',),)),
        }
        assert orderer.order(variables) == ['few', 'many']


class TestCVOrder:
    """Ordre 'cv' : meilleur score en premier (§8.1, §8.4)."""

    def test_cv_order_on_reference_ts(self, reference_timeseries):
        """[SPEC] §8.4 : scores q1=-0.08, a2=-0.15, a1=-0.15 -> ordre q1, a1, a2."""
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
    """Déterminisme intra-étape : jamais l'ordre des colonnes/clés d'entrée (§8.1)."""

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
    """Paramètre ``cv`` polymorphe sklearn, remplaçant ``cv_n_splits`` (§8.3, D4)."""

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
        """Aucun paramètre nommé "cv_n_splits" n'est accepté (absorbé par "cv", D4)."""
        assert 'cv_n_splits' not in VariableOrderer().get_params()
        with pytest.raises(TypeError):
            VariableOrderer(cv_n_splits=5)


class TestCVFallbacksAndLogging:
    """Correctifs CV §8.2 : sentinelles -inf, journalisation des plis en échec."""

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
