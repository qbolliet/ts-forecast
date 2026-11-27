"""Workflow tests of ``tsforecast.ensemble.OnlineAggregator``.

The integration rules of the adapter contract in meta-estimator and passthrough
modes (``cross_val_score`` with ``tsforecast`` splitters and a gap, ``GridSearchCV``,
``XYPipeline`` with ``DropNaNTarget``, ``sklearn.pipeline.Pipeline``); a grid search
over the algorithm and a nested expert parameter; ``cross_val_score`` on a panel
with per-entity pooling; the passthrough mode fed with out-of-sample forecasts of
other estimators.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor

from tests.support.adapter_cases import ensemble_meta_case, ensemble_passthrough_case
from tests.support.adapter_contract import AdapterWorkflowContractTests, split_by_date
from tests.support.datasets import build_adapter_panel
from tsforecast.crossvals import PanelOutOfSampleSplit
from tsforecast.ensemble import OnlineAggregator


class TestMetaWorkflow(AdapterWorkflowContractTests):
    """Workflow rules, meta-estimator over a ridge and a tree."""

    case = ensemble_meta_case()


class TestPassthroughWorkflow(AdapterWorkflowContractTests):
    """Workflow rules, expert forecasts given as columns of ``X``."""

    case = ensemble_passthrough_case()


# Fonction de construction de l'agrégateur méta-estimateur des scénarios
def _aggregator(**params) -> OnlineAggregator:
    """Meta-estimator over a ridge, a tree and the mean, three internal blocks of dates."""
    experts = [
        ('ridge', Ridge(alpha=1.0)),
        ('tree', DecisionTreeRegressor(max_depth=3, random_state=0)),
        ('mean', DummyRegressor()),
    ]
    return OnlineAggregator(experts, cv=3, **params)


class TestWorkflows:
    """Grid search and cross-validation on a panel."""

    def test_grid_search_on_the_algorithm_and_a_nested_expert_parameter(self):
        X, y = build_adapter_panel(balanced=False)
        X_train, y_train, X_test, _ = split_by_date(X, y)
        grid = {'aggregation': ['BOA', 'MLpol'], 'experts__ridge__alpha': [0.01, 100.0]}
        search = GridSearchCV(
            _aggregator(), grid, cv=PanelOutOfSampleSplit(n_splits=3, test_size=2, gap=2), error_score='raise'
        ).fit(X_train, y_train)
        # Relation linéaire : la ridge peu régularisée l'emporte
        assert search.best_params_['experts__ridge__alpha'] == 0.01
        assert search.best_estimator_.weights_['ridge'] > 0.5
        assert search.predict(X_test).shape == (len(X_test),)

    def test_cross_val_score_on_a_panel_per_entity(self):
        X, y = build_adapter_panel(balanced=False)
        splitter = PanelOutOfSampleSplit(n_splits=3, test_size=2, gap=2)
        scores = cross_val_score(_aggregator(pooling='per_entity'), X, y, cv=splitter, error_score='raise')
        mean_scores = cross_val_score(DummyRegressor(), X, y, cv=splitter)
        assert len(scores) == 3 and np.isfinite(scores).all()
        # L'agrégat dépasse nettement l'expert constant
        assert scores.mean() > mean_scores.mean() + 0.5

    def test_passthrough_on_out_of_sample_forecasts_reproduces_the_meta_estimator(self):
        # Prévisions hors échantillon calculées une fois, puis agrégées en passthrough sans réajuster d'expert
        X, y = build_adapter_panel()
        meta = _aggregator(aggregation='MLpol').fit(X, y)
        forecasts = meta.oos_predictions_
        passthrough = OnlineAggregator('passthrough', aggregation='MLpol').fit(forecasts, y.loc[forecasts.index])
        pd.testing.assert_series_equal(passthrough.weights_, meta.weights_)
