"""Numerical tests of ``HierarchicalForecastAdapter``: coherence and detailed outputs.

Reconciled forecasts are coherent (each aggregate equals the sum of its bottom
nodes) for a node tree as a time series, as independent copies (balanced and
unbalanced panels) and for a hierarchy of index levels, with ``BottomUp``,
``MinTrace`` ``'ols'`` and ``'mint_shrink'`` and one, the mean of two or one per
level base models. Golden properties: bottom-up sums the bottom base forecasts,
the combinations reconcile the combined column, the orthogonal projection of
``'ols'`` never increases the squared error of a date, copies are reconciled
independently. Then the outputs of ``reconcile`` (columns, row order, date
units, intervals, samples), ``diagnose`` and ``summing_matrix``.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('hierarchicalforecast')

from hierarchicalforecast.methods import BottomUp, MinTrace  # noqa: E402

from tests.support.adapter_cases import HIERARCHY_MODEL_BY_LEVEL  # noqa: E402
from tests.support.adapter_contract import split_by_date  # noqa: E402
from tests.support.datasets import (  # noqa: E402
    HIERARCHY_LEVELS,
    HIERARCHY_TREE,
    build_hierarchy_levels,
    build_hierarchy_nodes,
)
from tests.support.perturbations import (  # noqa: E402
    DATETIME_UNITS,
    datetime_units,
    shuffle_rows,
    with_datetime_resolution,
)
from tsforecast.adapters import HierarchicalForecastAdapter  # noqa: E402

# Tolérance de la cohérence des prévisions réconciliées
COHERENCE_ATOL = 1e-8
# Jeux hiérarchiques : forme de la hiérarchie et type de données
DATASETS = {
    'tree_series': (HIERARCHY_TREE, build_hierarchy_nodes, 'series'),
    'tree_panel': (HIERARCHY_TREE, build_hierarchy_nodes, 'panel'),
    'tree_unbalanced_panel': (HIERARCHY_TREE, build_hierarchy_nodes, 'unbalanced_panel'),
    'levels_series': (HIERARCHY_LEVELS, build_hierarchy_levels, 'series'),
    'levels_panel': (HIERARCHY_LEVELS, build_hierarchy_levels, 'panel'),
}
RECONCILERS = {
    'bottomup': BottomUp,
    'ols': lambda: MinTrace(method='ols'),
    'mint_shrink': lambda: MinTrace(method='mint_shrink'),
}


def _dataset(name):
    """Hierarchy and ``(X_train, y_train, X_test, y_test)`` of a dataset of :data:`DATASETS`."""
    hierarchy, builder, kind = DATASETS[name]
    return hierarchy, split_by_date(*builder(kind))


def _combinations(hierarchy):
    """The three ways of choosing base forecasts, for a hierarchy."""
    by_level = (
        HIERARCHY_MODEL_BY_LEVEL if hierarchy is HIERARCHY_TREE else {'country': 'm2', 'country/region': 'm1'}
    )
    return {
        'single': {'forecast_columns': ['m1']},
        'mean': {'model_combination': 'mean'},
        'by_level': {'model_combination': by_level},
    }


def max_incoherence(values: pd.Series) -> float:
    """Largest gap between an aggregate and the sum of its bottom nodes, on every date and copy.

    Computed from the definition of the datasets, independently of the adapter.
    """
    if 'node' in values.index.names:
        wide = values.unstack('node')
        gaps = [wide['a'] - wide['b'] - wide['c'], wide['c'] - wide['d'] - wide['e']]
        return float(max(np.abs(gap).max() for gap in gaps))
    regions = values.index.get_level_values('region')
    groups = [name for name in values.index.names if name != 'region']
    totals = values[regions == 'Total'].droplevel('region')
    sums = values[regions != 'Total'].groupby(level=groups).sum()
    return float(np.abs(totals - sums.reindex(totals.index)).max())


class TestCoherence:
    """Every reconciled forecast is coherent."""

    @pytest.mark.parametrize('combination', ['single', 'mean', 'by_level'])
    @pytest.mark.parametrize('reconciler', sorted(RECONCILERS))
    @pytest.mark.parametrize('dataset', sorted(DATASETS))
    def test_reconciled_forecasts_are_coherent(self, dataset, reconciler, combination):
        hierarchy, (X_train, y_train, X_test, _) = _dataset(dataset)
        params = _combinations(hierarchy)[combination]
        adapter = HierarchicalForecastAdapter(hierarchy, RECONCILERS[reconciler](), **params).fit(X_train, y_train)
        base = X_test['m1']
        assert max_incoherence(base) > 0.1, 'The base forecasts of the dataset are already coherent'
        reconciled = pd.Series(adapter.predict(X_test), index=X_test.index)
        assert max_incoherence(reconciled) < COHERENCE_ATOL

    def test_every_reconcile_column_is_coherent(self):
        hierarchy, (X_train, y_train, X_test, _) = _dataset('tree_panel')
        adapter = HierarchicalForecastAdapter(hierarchy, model_combination='mean').fit(X_train, y_train)
        frame = adapter.reconcile(X_test, [BottomUp(), MinTrace(method='ols'), MinTrace(method='mint_shrink')])
        assert frame.shape[1] == 9
        for column in frame.columns:
            assert max_incoherence(frame[column]) < COHERENCE_ATOL, column


class TestGoldenProperties:
    """Values known without hierarchicalforecast."""

    def test_bottom_up_sums_the_bottom_base_forecasts(self):
        hierarchy, (X_train, y_train, X_test, _) = _dataset('tree_panel')
        adapter = HierarchicalForecastAdapter(hierarchy, BottomUp(), forecast_columns=['m2']).fit(X_train, y_train)
        reconciled = pd.Series(adapter.predict(X_test), index=X_test.index).unstack('node')
        base = X_test['m2'].unstack('node')
        np.testing.assert_allclose(reconciled[['b', 'd', 'e']], base[['b', 'd', 'e']])
        np.testing.assert_allclose(reconciled['c'], base['d'] + base['e'])
        np.testing.assert_allclose(reconciled['a'], base['b'] + base['d'] + base['e'])

    @pytest.mark.parametrize('reconciler', sorted(RECONCILERS))
    def test_mean_reconciles_the_mean_column(self, reconciler):
        hierarchy, (X_train, y_train, X_test, _) = _dataset('tree_panel')
        combined = HierarchicalForecastAdapter(hierarchy, RECONCILERS[reconciler](), model_combination='mean')
        combined.fit(X_train, y_train)
        means = [data.assign(avg=data[['m1', 'm2']].mean(axis=1)) for data in (X_train, X_test)]
        reference = HierarchicalForecastAdapter(hierarchy, RECONCILERS[reconciler](), forecast_columns=['avg'])
        reference.fit(means[0], y_train)
        np.testing.assert_allclose(combined.predict(X_test), reference.predict(means[1]), rtol=1e-10)

    @pytest.mark.parametrize('reconciler', sorted(RECONCILERS))
    def test_one_model_per_level_reconciles_the_chosen_column(self, reconciler):
        hierarchy, (X_train, y_train, X_test, _) = _dataset('tree_panel')
        combined = HierarchicalForecastAdapter(
            hierarchy, RECONCILERS[reconciler](), model_combination=HIERARCHY_MODEL_BY_LEVEL
        ).fit(X_train, y_train)
        # Profondeur 1 (b, c) : m2 ; profondeurs 0 et 2 : m1
        picked = [
            data.assign(pick=data['m1'].where(~data.index.get_level_values('node').isin(['b', 'c']), data['m2']))
            for data in (X_train, X_test)
        ]
        reference = HierarchicalForecastAdapter(hierarchy, RECONCILERS[reconciler](), forecast_columns=['pick'])
        reference.fit(picked[0], y_train)
        np.testing.assert_allclose(combined.predict(X_test), reference.predict(picked[1]), rtol=1e-10)

    @pytest.mark.parametrize('dataset', ['tree_panel', 'levels_panel'])
    def test_ols_never_increases_the_squared_error_of_a_date(self, dataset):
        # Projection orthogonale sur les prévisions cohérentes, dont la réalisation fait partie
        hierarchy, (X_train, y_train, X_test, y_test) = _dataset(dataset)
        adapter = HierarchicalForecastAdapter(hierarchy, MinTrace(method='ols'), forecast_columns=['m2'])
        reconciled = adapter.fit(X_train, y_train).predict(X_test)
        groups = [name for name in X_test.index.names if name not in ('node', 'country', 'region')]
        base_error = ((X_test['m2'] - y_test) ** 2).groupby(level=groups).sum()
        reconciled_error = ((reconciled - y_test) ** 2).groupby(level=groups).sum()
        assert (reconciled_error <= base_error + 1e-9).all()
        assert (reconciled_error < base_error).mean() > 0.9

    def test_copies_are_reconciled_independently(self):
        hierarchy, (X_train, y_train, X_test, _) = _dataset('tree_unbalanced_panel')
        params = {'model_combination': HIERARCHY_MODEL_BY_LEVEL}
        panel = HierarchicalForecastAdapter(hierarchy, MinTrace(method='mint_shrink'), **params).fit(X_train, y_train)
        predictions = pd.Series(panel.predict(X_test), index=X_test.index)
        for entity in ('A', 'B', 'C'):
            rows = [data.index.get_level_values('entity') == entity for data in (X_train, X_test)]
            alone = HierarchicalForecastAdapter(hierarchy, MinTrace(method='mint_shrink'), **params)
            alone.fit(X_train[rows[0]], y_train[rows[0]])
            np.testing.assert_allclose(predictions[rows[1]].to_numpy(), alone.predict(X_test[rows[1]]), rtol=1e-10)

    def test_predict_is_the_reconcile_column_of_the_reconciler(self):
        hierarchy, (X_train, y_train, X_test, _) = _dataset('levels_panel')
        adapter = HierarchicalForecastAdapter(hierarchy, MinTrace(method='ols'), model_combination='mean')
        adapter.fit(X_train, y_train)
        frame = adapter.reconcile(X_test)
        assert frame.columns.tolist() == ['m1/MinTrace_method-ols', 'm2/MinTrace_method-ols', 'combined/MinTrace_method-ols']
        np.testing.assert_allclose(adapter.predict(X_test), frame['combined/MinTrace_method-ols'])


class TestReconcileOutput:
    """Rows, columns, dates and intervals of ``reconcile``."""

    def test_rows_follow_x(self):
        hierarchy, (X_train, y_train, X_test, _) = _dataset('tree_panel')
        adapter = HierarchicalForecastAdapter(hierarchy, forecast_columns=['m1']).fit(X_train, y_train)
        reference = adapter.reconcile(X_test)
        shuffled = shuffle_rows(X_test, seed=4)
        frame = adapter.reconcile(shuffled)
        assert frame.index.equals(shuffled.index)
        assert frame.columns.dtype == 'str'
        np.testing.assert_allclose(frame.to_numpy(), reference.loc[shuffled.index].to_numpy())

    def test_dates_keep_their_unit(self):
        hierarchy, builder, kind = DATASETS['levels_panel']
        reference = None
        for unit in DATETIME_UNITS:
            X, y = (with_datetime_resolution(data, unit) for data in builder(kind))
            X_train, y_train, X_test, _ = split_by_date(X, y)
            adapter = HierarchicalForecastAdapter(hierarchy, MinTrace(method='mint_shrink'), forecast_columns=['m1'])
            frame = adapter.fit(X_train, y_train).reconcile(X_test)
            assert set(datetime_units(frame).values()) == {unit}
            if reference is None:
                reference = frame.to_numpy()
            np.testing.assert_allclose(frame.to_numpy(), reference)

    def test_normality_intervals_read_the_interval_columns(self):
        hierarchy, (X_train, y_train, X_test, _) = _dataset('tree_series')
        with_intervals = [
            data[['m1']].assign(**{'m1-lo-80': data['m1'] - 2.0, 'm1-hi-80': data['m1'] + 2.0})
            for data in (X_train, X_test)
        ]
        adapter = HierarchicalForecastAdapter(hierarchy, MinTrace(method='ols')).fit(with_intervals[0], y_train)
        frame = adapter.reconcile(with_intervals[1], level=[80])
        name = 'm1/MinTrace_method-ols'
        assert frame.columns.tolist() == [name, f'{name}-lo-80', f'{name}-hi-80']
        assert (frame[f'{name}-lo-80'] < frame[name]).all() and (frame[name] < frame[f'{name}-hi-80']).all()
        np.testing.assert_allclose(frame[name], adapter.predict(with_intervals[1]))

    def test_normality_intervals_need_interval_columns(self, tree_series):
        X_train, y_train, X_test, _ = tree_series
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train, y_train)
        with pytest.raises(ValueError, match="'m1-lo-<level>'"):
            adapter.reconcile(X_test, level=[80])
        mean = HierarchicalForecastAdapter(HIERARCHY_TREE, model_combination='mean')
        intervals = {f'{model}-{side}-80': X_train[model] for model in ('m1', 'm2') for side in ('lo', 'hi')}
        mean.fit(X_train.assign(**intervals), y_train)
        test_intervals = {f'{model}-{side}-80': X_test[model] for model in ('m1', 'm2') for side in ('lo', 'hi')}
        with pytest.raises(ValueError, match='a mean of models has none'):
            mean.reconcile(X_test.assign(**test_intervals), level=[80])

    def test_normality_intervals_of_one_model_per_level(self, tree_series):
        X_train, y_train, X_test, _ = tree_series

        def widen(data, width):
            # Intervalles de largeur propre à chaque modèle
            return data.assign(**{
                f'{model}-{side}-90': data[model] + sign * width[model]
                for model in ('m1', 'm2') for side, sign in (('lo', -1), ('hi', 1))
            })

        width = {'m1': 1.0, 'm2': 3.0}
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, model_combination=HIERARCHY_MODEL_BY_LEVEL)
        adapter.fit(widen(X_train, width), y_train)
        frame = adapter.reconcile(widen(X_test, width), level=[90])
        bounds = [f'combined/BottomUp-{side}-90' for side in ('lo', 'hi')]
        assert set(bounds) <= set(frame.columns)
        assert (frame[bounds[0]] < frame['combined/BottomUp']).all()

    def test_bootstrap_intervals_and_samples(self, tree_panel):
        X_train, y_train, X_test, _ = tree_panel
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, MinTrace(method='mint_shrink'), forecast_columns=['m1'])
        adapter.fit(X_train, y_train)
        frame = adapter.reconcile(X_test, level=[80], intervals_method='bootstrap', num_samples=3, seed=1)
        name = 'm1/MinTrace_method-mint_shrink'
        assert frame.columns.tolist() == [name, f'{name}-lo-80', f'{name}-hi-80'] + [
            f'{name}-sample-{i}' for i in range(3)
        ]
        assert np.isfinite(frame.to_numpy()).all()
        for sample in range(3):
            assert max_incoherence(frame[f'{name}-sample-{sample}']) < 1e-6
        again = adapter.reconcile(X_test, level=[80], intervals_method='bootstrap', num_samples=3, seed=1)
        np.testing.assert_array_equal(frame.to_numpy(), again.to_numpy())


class TestDiagnoseAndSummingMatrix:
    """Coherence diagnostics and summing matrices."""

    def test_diagnose_panel(self, tree_panel):
        X_train, y_train, X_test, _ = tree_panel
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train, y_train)
        diagnostics = adapter.diagnose(X_test, [BottomUp(), MinTrace(method='ols')])
        assert diagnostics.columns.tolist() == ['entity', 'level', 'metric', 'm1/BottomUp', 'm1/MinTrace_method-ols']
        assert diagnostics['level'].dtype == 'str' and diagnostics['metric'].dtype == 'str'
        assert set(diagnostics['entity']) == {'A', 'B', 'C'}
        assert {'depth_0', 'depth_1', 'depth_2', 'Overall'} == set(diagnostics['level'])
        coherent = diagnostics[diagnostics['metric'] == 'is_coherent']
        assert (coherent[['m1/BottomUp', 'm1/MinTrace_method-ols']] == 1.0).all().all()

    def test_diagnose_series(self, levels_series):
        X_train, y_train, X_test, _ = levels_series
        adapter = HierarchicalForecastAdapter(HIERARCHY_LEVELS, model_combination='mean').fit(X_train, y_train)
        diagnostics = adapter.diagnose(X_test)
        assert diagnostics.columns.tolist() == ['level', 'metric', 'm1/BottomUp', 'm2/BottomUp', 'combined/BottomUp']
        before = diagnostics[diagnostics['metric'] == 'coherence_residual_mae_before']
        assert (before.drop(columns=['level', 'metric']).to_numpy() >= 0).all()

    def test_summing_matrix_of_a_tree(self, tree_panel):
        X_train, y_train, _, _ = tree_panel
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train, y_train)
        matrix = adapter.summing_matrix('A')
        assert matrix.index.tolist() == ['a', 'c', 'b', 'd', 'e'] and matrix.index.name == 'node'
        assert matrix.columns.tolist() == ['b', 'd', 'e']
        assert matrix.loc['a'].tolist() == [1.0, 1.0, 1.0] and matrix.loc['c'].tolist() == [0.0, 1.0, 1.0]
        # Copie inconnue d'un arbre : structure de l'arbre
        assert adapter.summing_matrix(('Z',)).equals(matrix)
        with pytest.raises(ValueError, match='Choose a copy'):
            adapter.summing_matrix()

    def test_summing_matrix_of_index_levels(self, levels_series, levels_panel):
        X_train, y_train, _, _ = levels_series
        adapter = HierarchicalForecastAdapter(HIERARCHY_LEVELS, forecast_columns=['m1']).fit(X_train, y_train)
        matrix = adapter.summing_matrix()
        assert list(matrix.index.names) == ['country', 'region']
        assert matrix.index[:2].tolist() == [('DE', 'Total'), ('FR', 'Total')]
        assert matrix.shape == (7, 5)
        assert matrix.loc[('FR', 'Total')].sum() == 2.0
        X_train, y_train, _, _ = levels_panel
        adapter.fit(X_train, y_train)
        assert adapter.summing_matrix('B').shape == (7, 5)
        with pytest.raises(ValueError, match='unknown at fit'):
            adapter.summing_matrix('Z')
