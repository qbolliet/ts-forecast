"""Workflow tests of the temporal and cross-temporal modes of ``HierarchicalForecastAdapter``.

``cross_val_score`` with ``period_splits`` (test folds of complete calendar
years, with a gap) on series, balanced and unbalanced panels, temporal and
cross-temporal; the panel splitter of ``tsforecast``, which counts test rows per
entity, gives incomplete years that predict refuses. ``GridSearchCV`` over the
reconciler, ``XYPipeline`` with ``DropNaNTarget`` and ``sklearn.pipeline.Pipeline``.
Rows indexed by forecast origin: shifting the periods with ``temporal_origin``
reconciles calendar years of the targets.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('hierarchicalforecast')

from hierarchicalforecast.methods import BottomUp, MinTrace  # noqa: E402
from sklearn.model_selection import GridSearchCV, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from tests.support.datasets import (  # noqa: E402
    HIERARCHY_LEVELS,
    HIERARCHY_TREE,
    TEMPORAL_HIERARCHY,
    build_cross_temporal_hierarchy,
    build_temporal_hierarchy,
)
from tsforecast.adapters import HierarchicalForecastAdapter, period_splits  # noqa: E402
from tsforecast.crossvals import PanelOutOfSampleSplit  # noqa: E402
from tsforecast.xy import DropNaNTarget, XYPipeline  # noqa: E402

# Validation croisée : deux plis d'une année, deux mois écartés avant le test
N_SPLITS = 2
GAP = 2


def _temporal_adapter(reconciler=None, **params):
    """Temporal adapter on the mean of the two base models."""
    params.setdefault('model_combination', 'mean')
    return HierarchicalForecastAdapter(None, reconciler, temporal_hierarchy=TEMPORAL_HIERARCHY, **params)


def _cross_adapter(hierarchy=HIERARCHY_TREE, reconciler=None, **params):
    """Cross-temporal adapter on the mean of the two base models."""
    params.setdefault('model_combination', 'mean')
    return HierarchicalForecastAdapter(hierarchy, reconciler, temporal_hierarchy=TEMPORAL_HIERARCHY, **params)


@pytest.mark.parametrize(
    ('make_adapter', 'make_data'),
    [
        (_temporal_adapter, lambda kind: build_temporal_hierarchy(kind, n_years=5)),
        (_cross_adapter, lambda kind: build_cross_temporal_hierarchy(kind, n_years=5)),
        (
            lambda **params: _cross_adapter(HIERARCHY_LEVELS, **params),
            lambda kind: build_cross_temporal_hierarchy(kind, n_years=5, levels=True),
        ),
    ],
    ids=['temporal', 'cross_temporal_tree', 'cross_temporal_levels'],
)
@pytest.mark.parametrize('kind', ['series', 'panel', 'unbalanced_panel'])
@pytest.mark.parametrize('reconciler', [BottomUp, lambda: MinTrace(method='ols')], ids=['bottomup', 'ols'])
def test_cross_val_score_on_complete_years(make_adapter, make_data, kind, reconciler):
    # Panel déséquilibré : la copie C commence en 2022, avant le premier pli de test (2023)
    X, y = make_data(kind)
    folds = period_splits(X, TEMPORAL_HIERARCHY, n_splits=N_SPLITS, gap=GAP)
    scores = cross_val_score(
        make_adapter(reconciler=reconciler()), X, y, cv=folds, scoring='neg_mean_squared_error', error_score='raise'
    )
    assert len(scores) == N_SPLITS and np.isfinite(scores).all()


def test_panel_splitter_cuts_incomplete_years():
    X, y = build_temporal_hierarchy('panel')
    with warnings.catch_warnings():
        # Le splitter signale les entités annuelles sans la date de début du test
        warnings.simplefilter('ignore')
        with pytest.raises(ValueError, match='incomplete period'):
            cross_val_score(
                _temporal_adapter(), X, y, cv=PanelOutOfSampleSplit(n_splits=2, test_size=12), error_score='raise'
            )


def test_grid_search_over_the_reconciler():
    X, y = build_cross_temporal_hierarchy('panel', n_years=4)
    train, test = period_splits(X, TEMPORAL_HIERARCHY, n_splits=1)[0]
    X_train, y_train, X_test = X.iloc[train], y.iloc[train], X.iloc[test]
    search = GridSearchCV(
        _cross_adapter(),
        {'reconciler': [BottomUp(), MinTrace(method='ols'), MinTrace(method='wls_struct')]},
        cv=period_splits(X_train, TEMPORAL_HIERARCHY, n_splits=2, gap=GAP),
        scoring='neg_mean_squared_error',
        error_score='raise',
    ).fit(X_train, y_train)
    assert set(search.best_params_) == {'reconciler'}
    predictions = search.predict(X_test)
    assert predictions.shape == (len(X_test),) and np.isfinite(predictions).all()
    # Le réconciliateur choisi reconcilie le test de façon cohérente : année = somme des mois
    values = pd.Series(predictions, index=X_test.index)
    levels = values.index.get_level_values('level')
    by_copy = [name for name in values.index.names if name not in ('level', 'date')]
    np.testing.assert_allclose(
        values[levels == 'year'].groupby(by_copy).sum(), values[levels == 'month'].groupby(by_copy).sum()
    )


@pytest.mark.parametrize('kind', ['series', 'unbalanced_panel'])
def test_xy_pipeline_drops_rows_without_target(kind):
    X, y = build_temporal_hierarchy(kind)
    train, test = period_splits(X, TEMPORAL_HIERARCHY, n_splits=1)[0]
    X_train, y_train, X_test = X.iloc[train], y.iloc[train], X.iloc[test]
    # Cible manquante sur une ligne sur cinq de l'entraînement
    missing = np.arange(len(y_train)) % 5 == 1
    pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', _temporal_adapter(MinTrace(method='ols')))])
    predictions = pipeline.fit(X_train, y_train.mask(missing)).predict(X_test)
    expected = _temporal_adapter(MinTrace(method='ols')).fit(X_train[~missing], y_train[~missing]).predict(X_test)
    np.testing.assert_allclose(predictions, expected)


def test_sklearn_pipeline():
    X, y = build_cross_temporal_hierarchy('panel')
    train, test = period_splits(X, TEMPORAL_HIERARCHY, n_splits=1)[0]
    pipeline = Pipeline([('model', _cross_adapter(reconciler=MinTrace(method='ols')))])
    predictions = pipeline.fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test])
    expected = _cross_adapter(reconciler=MinTrace(method='ols')).fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test])
    np.testing.assert_allclose(predictions, expected)


def test_origin_indexed_rows_reconcile_target_years_with_a_shifted_origin():
    # Lignes indexées par date cible, puis par origine t = cible - h (h = 3 mois)
    X, y = build_temporal_hierarchy('panel', n_years=3)
    horizon = pd.DateOffset(months=3)
    by_origin = X.index.set_levels(X.index.levels[-1] - horizon, level='date')
    X_origin, y_origin = X.set_axis(by_origin), y.set_axis(by_origin)
    # Années cibles civiles = années d'origines d'octobre à septembre
    target = _temporal_adapter(MinTrace(method='ols'))
    origin = _temporal_adapter(MinTrace(method='ols'), temporal_origin='2019-10-01')
    train, test = period_splits(X, TEMPORAL_HIERARCHY, n_splits=1)[0]
    train_o, test_o = period_splits(X_origin, TEMPORAL_HIERARCHY, n_splits=1, temporal_origin='2019-10-01')[0]
    np.testing.assert_array_equal(train_o, train)
    np.testing.assert_array_equal(test_o, test)
    np.testing.assert_allclose(
        origin.fit(X_origin.iloc[train], y_origin.iloc[train]).predict(X_origin.iloc[test]),
        target.fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test]),
    )
    # Sans décalage de l'origine, les années d'origines ne sont pas complètes dans ces données
    with pytest.raises(ValueError, match="level 'year' is dated"):
        _temporal_adapter(MinTrace(method='ols')).fit(X_origin.iloc[train], y_origin.iloc[train])
