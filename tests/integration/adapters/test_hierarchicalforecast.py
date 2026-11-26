"""Workflow tests of ``HierarchicalForecastAdapter``: the integration rules of the adapter contract.

``cross_val_score`` with a panel splitter and a gap (node tree as a series,
balanced and unbalanced copies, hierarchy of index levels), ``GridSearchCV``,
``XYPipeline`` with ``DropNaNTarget`` and ``sklearn.pipeline.Pipeline`` for four
reconciliations; then ``GridSearchCV`` comparing ``BottomUp``, ``MinTrace``
``'ols'`` and ``'mint_shrink'``, and the full chain: out-of-sample base forecasts
produced fold by fold, then reconciled with in-sample errors learnt from them.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('hierarchicalforecast')

from hierarchicalforecast.methods import BottomUp, MinTrace  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate  # noqa: E402

from tests.support.adapter_cases import (  # noqa: E402
    HIERARCHY_MODEL_BY_LEVEL,
    hierarchical_bottomup_case,
    hierarchical_levels_case,
    hierarchical_mint_shrink_case,
    hierarchical_ols_case,
)
from tests.support.adapter_contract import AdapterWorkflowContractTests  # noqa: E402
from tests.support.datasets import (  # noqa: E402
    HIERARCHY_LEVELS,
    HIERARCHY_TREE,
    build_hierarchy_levels,
    build_hierarchy_nodes,
)
from tsforecast.adapters import HierarchicalForecastAdapter  # noqa: E402
from tsforecast.crossvals import PanelOutOfSampleSplit  # noqa: E402

# Validation croisée des tests d'intégration : horizon d'un mois, gap d'un mois
CV_PARAMS = {'n_splits': 4, 'test_size': 3, 'gap': 1}


class TestBottomUpWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of ``BottomUp`` on the node tree."""

    case = hierarchical_bottomup_case()


class TestOlsWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of ``MinTrace(method='ols')`` on the mean of two base models."""

    case = hierarchical_ols_case()


class TestMintShrinkWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of ``MinTrace(method='mint_shrink')`` with one base model per level."""

    case = hierarchical_mint_shrink_case()


class TestLevelsWorkflow(AdapterWorkflowContractTests):
    """Workflow contract of ``MinTrace(method='ols')`` on a hierarchy of index levels."""

    case = hierarchical_levels_case()


@pytest.mark.parametrize(
    ('hierarchy', 'builder', 'kind'),
    [
        (HIERARCHY_TREE, build_hierarchy_nodes, 'series'),
        (HIERARCHY_TREE, build_hierarchy_nodes, 'unbalanced_panel'),
        (HIERARCHY_LEVELS, build_hierarchy_levels, 'panel'),
    ],
)
def test_cross_val_score_with_in_sample_reconciler(hierarchy, builder, kind):
    X, y = builder(kind)
    adapter = HierarchicalForecastAdapter(hierarchy, MinTrace(method='mint_shrink'), model_combination='mean')
    scores = cross_val_score(
        adapter, X, y, cv=PanelOutOfSampleSplit(**CV_PARAMS), scoring='neg_mean_squared_error', error_score='raise'
    )
    assert len(scores) == CV_PARAMS['n_splits'] and np.isfinite(scores).all()


def test_grid_search_compares_the_reconcilers():
    X, y = build_hierarchy_nodes('panel')
    grid = {
        'reconciler': [BottomUp(), MinTrace(method='ols'), MinTrace(method='mint_shrink')],
        'model_combination': ['mean', HIERARCHY_MODEL_BY_LEVEL],
    }
    search = GridSearchCV(
        HierarchicalForecastAdapter(HIERARCHY_TREE),
        grid,
        cv=PanelOutOfSampleSplit(**CV_PARAMS),
        scoring='neg_mean_squared_error',
        error_score='raise',
    ).fit(X, y)
    assert len(search.cv_results_['params']) == 6
    assert np.isfinite(search.cv_results_['mean_test_score']).all()
    # Erreurs indépendantes et de même variance sur chaque nœud : la projection bat la somme des feuilles
    assert type(search.best_params_['reconciler']) is MinTrace
    wide = pd.Series(search.predict(X), index=X.index).unstack('node')
    np.testing.assert_allclose(wide['a'], wide['b'] + wide['d'] + wide['e'])


def test_out_of_sample_base_forecasts_feed_the_in_sample_reconciler():
    _, y = build_hierarchy_nodes('panel', n_dates=60)
    # Caractéristiques connues à l'origine : valeurs passées de chaque nœud de chaque copie
    groups = y.groupby(level=['entity', 'node'])
    features = pd.DataFrame({'lag_1': groups.shift(1), 'lag_12': groups.shift(12)}).dropna()
    target = y.loc[features.index]

    # Prévisions de base hors échantillon, pli par pli
    splitter = PanelOutOfSampleSplit(n_splits=8, test_size=3, gap=0)
    folds = cross_validate(
        LinearRegression(), features, target, cv=splitter, return_estimator=True, return_indices=True
    )
    base = pd.Series(np.nan, index=features.index)
    for estimator, test in zip(folds['estimator'], folds['indices']['test']):
        base.iloc[test] = estimator.predict(features.iloc[test])
    base = base.dropna().to_frame('lr')
    realised = target.loc[base.index]

    # Réconciliation : erreurs insample apprises sur les premiers plis, test sur les derniers
    dates = base.index.get_level_values('date')
    train = np.asarray(dates <= dates.unique().sort_values()[-7])
    test = np.asarray(dates >= dates.unique().sort_values()[-3])
    adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, MinTrace(method='mint_shrink'))
    adapter.fit(base[train], realised[train])
    reconciled = pd.Series(adapter.predict(base[test]), index=base[test].index).unstack('node')
    np.testing.assert_allclose(reconciled['c'], reconciled['d'] + reconciled['e'])
    np.testing.assert_allclose(reconciled['a'], reconciled['b'] + reconciled['c'])
    assert np.isfinite(adapter.score(base[test], realised[test]))
