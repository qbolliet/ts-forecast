"""Unit tests of ``HierarchicalForecastAdapter``: contract, parameters and explicit errors.

Runs the unit rules of the adapter contract on four reconciliations (``BottomUp``,
``MinTrace`` ``'ols'`` with the mean of two models, ``MinTrace`` ``'mint_shrink'``
with one model per level, ``MinTrace`` ``'ols'`` on a hierarchy of index levels),
then targeted scenarios: wrapped reconciler and fitted state never modified by a
prediction, parsing of both forms of ``hierarchy`` and of the copy levels, base
forecast columns and ``model_combination``, rows that cannot be reconciled
(unknown node or copy, incomplete date, missing forecast), reconcilers that need a
strict hierarchy or in-sample data.
"""
import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('hierarchicalforecast')

from hierarchicalforecast.methods import BottomUp, MiddleOut, MinTrace, TopDown  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.exceptions import NotFittedError  # noqa: E402

from tests.support.adapter_cases import (  # noqa: E402
    hierarchical_bottomup_case,
    hierarchical_levels_case,
    hierarchical_mint_shrink_case,
    hierarchical_ols_case,
)
from tests.support.adapter_contract import AdapterContractTests  # noqa: E402
from tests.support.datasets import HIERARCHY_LEVELS, HIERARCHY_TREE  # noqa: E402
from tsforecast.adapters import HierarchicalForecastAdapter  # noqa: E402

# Arbre dont toutes les feuilles sont à la même profondeur
BALANCED_TREE = {'t': ['x', 'y'], 'x': ['x1', 'x2'], 'y': ['y1', 'y2']}


def _rename_level(data, level, mapping):
    """Return a copy of ``data`` whose labels of an index level are renamed."""
    return data.rename(index=mapping, level=level)


def _drop_rows(data, mask):
    """Return the rows of ``data`` outside ``mask``."""
    return data[~np.asarray(mask)]


class TestBottomUpContract(AdapterContractTests):
    """Unit contract of ``BottomUp`` on the node tree."""

    case = hierarchical_bottomup_case()


class TestOlsContract(AdapterContractTests):
    """Unit contract of ``MinTrace(method='ols')`` on the mean of two base models."""

    case = hierarchical_ols_case()


class TestMintShrinkContract(AdapterContractTests):
    """Unit contract of ``MinTrace(method='mint_shrink')`` with one base model per level."""

    case = hierarchical_mint_shrink_case()


class TestLevelsContract(AdapterContractTests):
    """Unit contract of ``MinTrace(method='ols')`` on a hierarchy of index levels."""

    case = hierarchical_levels_case()


class TestWrappedReconciler:
    """The reconciler and the fitted state are never modified."""

    def test_default_reconciler_is_bottom_up(self, tree_series):
        X_train, y_train, X_test, _ = tree_series
        default = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train, y_train)
        explicit = HierarchicalForecastAdapter(HIERARCHY_TREE, BottomUp(), forecast_columns=['m1']).fit(
            X_train, y_train
        )
        assert type(default.reconciler_) is BottomUp
        assert default.reconciler is None
        np.testing.assert_allclose(default.predict(X_test), explicit.predict(X_test))

    def test_reconciler_copied_at_fit(self, tree_series):
        X_train, y_train, _, _ = tree_series
        reconciler = MinTrace(method='mint_shrink')
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, reconciler, forecast_columns=['m1']).fit(
            X_train, y_train
        )
        assert adapter.reconciler is reconciler
        assert adapter.reconciler_ is not reconciler

    def test_predictions_leave_the_fitted_state_untouched(self, tree_panel):
        X_train, y_train, X_test, _ = tree_panel
        reconciler = MinTrace(method='mint_shrink')
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, reconciler, model_combination='mean').fit(
            X_train, y_train
        )
        state, wrapped = joblib.hash(adapter.__dict__), joblib.hash(reconciler)
        first = adapter.predict(X_test)
        adapter.reconcile(X_test, [BottomUp(), MinTrace(method='ols')], level=[80], intervals_method='bootstrap')
        adapter.diagnose(X_test)
        assert joblib.hash(adapter.__dict__) == state
        assert joblib.hash(reconciler) == wrapped
        np.testing.assert_array_equal(adapter.predict(X_test), first)

    def test_level_list_not_modified(self, tree_series):
        X_train, y_train, X_test, _ = tree_series
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, MinTrace(method='ols'), forecast_columns=['m1'])
        adapter.fit(X_train, y_train)
        level = [95, 80]
        adapter.reconcile(X_test, level=level, intervals_method='bootstrap')
        assert level == [95, 80]

    def test_object_that_is_no_reconciler(self, tree_series):
        X_train, y_train, X_test, _ = tree_series
        with pytest.raises(TypeError, match='hierarchicalforecast reconciler'):
            HierarchicalForecastAdapter(HIERARCHY_TREE, 'BottomUp', forecast_columns=['m1']).fit(X_train, y_train)
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train, y_train)
        with pytest.raises(TypeError, match='hierarchicalforecast reconciler'):
            adapter.reconcile(X_test, [BottomUp(), object()])
        with pytest.raises(ValueError, match='at least one'):
            adapter.reconcile(X_test, [])

    def test_clone_keeps_the_parameters(self):
        adapter = HierarchicalForecastAdapter(
            HIERARCHY_LEVELS, MinTrace(method='ols'), aggregate_label='All', forecast_columns=['m1']
        )
        cloned = clone(adapter)
        params = cloned.get_params()
        assert params['aggregate_label'] == 'All' and params['forecast_columns'] == ['m1']
        assert type(params['reconciler']) is MinTrace and params['reconciler'] is not adapter.reconciler

    def test_not_fitted(self, tree_series):
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1'])
        for method in (adapter.predict, adapter.reconcile, adapter.diagnose):
            with pytest.raises(NotFittedError):
                method(tree_series[2])
        with pytest.raises(NotFittedError):
            adapter.summing_matrix()


class TestHierarchyParameter:
    """Both forms of ``hierarchy``, the node level and the copy levels."""

    def test_tree_levels_and_copies(self, tree_panel):
        X_train, y_train, _, _ = tree_panel
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train, y_train)
        assert adapter.hierarchy_levels_ == ('depth_0', 'depth_1', 'depth_2')
        assert adapter.copies_ == (('A',), ('B',), ('C',))
        assert adapter.forecast_columns_ == ['m1']

    def test_series_is_a_single_copy(self, tree_series):
        X_train, y_train, _, _ = tree_series
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train, y_train)
        assert adapter.copies_ == ((),)

    def test_levels_names_and_copies(self, levels_panel):
        X_train, y_train, _, _ = levels_panel
        adapter = HierarchicalForecastAdapter(HIERARCHY_LEVELS, forecast_columns=['m1']).fit(X_train, y_train)
        assert adapter.hierarchy_levels_ == ('country', 'country/region')
        assert adapter.copies_ == (('A',), ('B',), ('C',))

    def test_node_level_by_name_or_position(self, tree_panel):
        X_train, y_train, X_test, _ = tree_panel
        reference = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train, y_train)
        # Niveau des nœuds placé en tête : désigné par son nom ou sa position
        swapped = [data.reorder_levels(['node', 'entity', 'date']) for data in (X_train, y_train, X_test)]
        for node_level in ('node', 0):
            adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, node_level=node_level, forecast_columns=['m1'])
            adapter.fit(swapped[0], swapped[1])
            assert adapter.copies_ == (('A',), ('B',), ('C',))
            expected = pd.Series(reference.predict(X_test), index=X_test.index)
            actual = pd.Series(adapter.predict(swapped[2]), index=swapped[2].index)
            np.testing.assert_allclose(
                actual.to_numpy(), expected.reorder_levels(['node', 'entity', 'date']).loc[actual.index].to_numpy()
            )

    def test_custom_aggregate_label(self, levels_series):
        X_train, y_train, X_test, _ = levels_series
        relabel = {'Total': 'all'}
        adapter = HierarchicalForecastAdapter(HIERARCHY_LEVELS, aggregate_label='all', forecast_columns=['m1'])
        adapter.fit(_rename_level(X_train, 'region', relabel), _rename_level(y_train, 'region', relabel))
        reference = HierarchicalForecastAdapter(HIERARCHY_LEVELS, forecast_columns=['m1']).fit(X_train, y_train)
        np.testing.assert_allclose(
            adapter.predict(_rename_level(X_test, 'region', relabel)), reference.predict(X_test)
        )

    def test_grouped_structure(self):
        # Pays, puis deux partitions croisées : régions et motifs
        dates = pd.date_range('2024-01-01', periods=4, freq='MS')
        leaves = [('FR', r, p) for r in ('north', 'south') for p in ('work', 'leisure')]
        aggregates = [('FR', 'Total', 'Total'), ('FR', 'north', 'Total'), ('FR', 'south', 'Total'),
                      ('FR', 'Total', 'work'), ('FR', 'Total', 'leisure')]
        rows = [(*key, date) for key in aggregates + leaves for date in dates]
        index = pd.MultiIndex.from_tuples(rows, names=['country', 'region', 'purpose', 'date'])
        rng = np.random.default_rng(3)
        X = pd.DataFrame({'m': rng.uniform(1, 10, len(index))}, index=index)
        y = X['m'] + 1.0
        spec = [['country'], ['country', 'region'], ['country', 'purpose'], ['country', 'region', 'purpose']]
        adapter = HierarchicalForecastAdapter(spec, MinTrace(method='ols')).fit(X, y)
        assert adapter.hierarchy_levels_ == ('country', 'country/region', 'country/purpose', 'country/region/purpose')
        reconciled = pd.Series(adapter.predict(X), index=index)
        leaves_sum = reconciled[reconciled.index.droplevel('date').isin(leaves)].groupby(level='date').sum()
        work_sum = reconciled.xs('work', level='purpose').drop('Total', level='region').groupby(level='date').sum()
        total = reconciled.xs(('FR', 'Total', 'Total'), level=['country', 'region', 'purpose'])
        work = reconciled.xs(('FR', 'Total', 'work'), level=['country', 'region', 'purpose'])
        np.testing.assert_allclose(total.to_numpy(), leaves_sum.to_numpy())
        np.testing.assert_allclose(work.to_numpy(), work_sum.to_numpy())
        with pytest.raises(ValueError, match='strictly hierarchical'):
            HierarchicalForecastAdapter(spec, TopDown(method='forecast_proportions')).fit(X, y)

    @pytest.mark.parametrize(
        ('params', 'message'),
        [
            ({'hierarchy': None}, 'hierarchy is required'),
            ({'hierarchy': 'a'}, 'non-empty list'),
            ({'hierarchy': []}, 'non-empty list'),
            ({'hierarchy': {'a': []}}, 'non-empty list'),
            ({'hierarchy': {'a': ['b', 'b']}}, 'duplicates'),
            ({'hierarchy': {'a': ['b'], 'b': ['a']}}, 'cycle'),
            ({'hierarchy': {'a': ['b', 'c'], 'c': ['b']}}, 'several parents'),
            ({'hierarchy': {'a': ['b', None]}}, 'missing'),
            ({'hierarchy': {'a': ['b', 'c', 'z']}}, 'absent from the hierarchy'),
            ({'hierarchy': HIERARCHY_TREE, 'node_level': 'region'}, 'non-date index level'),
            ({'hierarchy': HIERARCHY_TREE, 'node_level': 5}, 'position'),
        ],
    )
    def test_invalid_tree(self, tree_series, params, message):
        X_train, y_train, _, _ = tree_series
        with pytest.raises(ValueError, match=message):
            HierarchicalForecastAdapter(forecast_columns=['m1'], **params).fit(X_train, y_train)

    def test_tree_node_without_rows(self, tree_series):
        X_train, y_train, X_test, _ = tree_series
        # Nœud de l'arbre absent des données : accepté au fit, chaque date à réconcilier le réclame
        tree = {'a': ['b', 'c'], 'c': ['d', 'e', 'f']}
        adapter = HierarchicalForecastAdapter(tree, forecast_columns=['m1']).fit(X_train, y_train)
        with pytest.raises(ValueError, match="no row for the nodes 'f'"):
            adapter.predict(X_test)

    @pytest.mark.parametrize(
        ('params', 'message'),
        [
            ({'hierarchy': [['country'], ['country', 'city']]}, 'Unknown index levels'),
            ({'hierarchy': [['region'], ['country']]}, 'finest level'),
            ({'hierarchy': [['country'], ['country', 'country']]}, 'repeats'),
            ({'hierarchy': [['country', 'region'], ['region', 'country']]}, 'twice'),
            ({'hierarchy': [['country'], 'region']}, 'non-empty list'),
            ({'hierarchy': HIERARCHY_LEVELS, 'node_level': 'region'}, 'node_level only applies'),
            ({'hierarchy': HIERARCHY_LEVELS, 'aggregate_label': None}, 'aggregate_label'),
            ({'hierarchy': HIERARCHY_LEVELS, 'aggregate_label': float('nan')}, 'aggregate_label'),
        ],
    )
    def test_invalid_levels(self, levels_series, params, message):
        X_train, y_train, _, _ = levels_series
        with pytest.raises(ValueError, match=message):
            HierarchicalForecastAdapter(forecast_columns=['m1'], **params).fit(X_train, y_train)

    def test_row_matching_no_level(self, levels_series):
        X_train, y_train, _, _ = levels_series
        relabel = {'FR': 'Total'}
        with pytest.raises(ValueError, match='match no level'):
            HierarchicalForecastAdapter(HIERARCHY_LEVELS, forecast_columns=['m1']).fit(
                _rename_level(X_train, 'country', relabel), _rename_level(y_train, 'country', relabel)
            )

    def test_aggregate_without_bottom_rows(self, levels_series):
        X_train, y_train, _, _ = levels_series
        mask = (X_train.index.get_level_values('country') == 'DE') & (
            X_train.index.get_level_values('region') != 'Total'
        )
        with pytest.raises(ValueError, match='without any bottom row'):
            HierarchicalForecastAdapter(HIERARCHY_LEVELS, forecast_columns=['m1']).fit(
                _drop_rows(X_train, mask), _drop_rows(y_train, mask)
            )

    def test_index_without_node_level(self):
        dates = pd.date_range('2024-01-01', periods=3, freq='MS')
        X = pd.DataFrame({'m': [1.0, 2.0, 3.0]}, index=dates)
        with pytest.raises(ValueError, match='before the date level'):
            HierarchicalForecastAdapter({'a': ['b']}).fit(X, X['m'])

    @pytest.mark.parametrize('params', [{'temporal_level': 'node'}, {'temporal_origin': '2020-01-01'}])
    def test_temporal_parameters_need_a_temporal_hierarchy(self, tree_series, params):
        X_train, y_train, _, _ = tree_series
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1'], **params)
        with pytest.raises(ValueError, match='only apply with a temporal_hierarchy'):
            adapter.fit(X_train, y_train)

    def test_target_must_be_a_single_series(self, tree_series):
        X_train, y_train, _, _ = tree_series
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1'])
        with pytest.raises(ValueError, match='single target'):
            adapter.fit(X_train, pd.concat([y_train, y_train], axis=1))
        with pytest.raises(ValueError, match='single target'):
            adapter.fit(X_train, None)


class TestForecastColumns:
    """Base forecast columns, interval columns, exogenous columns and model combination."""

    def test_several_columns_need_a_combination(self, tree_series):
        X_train, y_train, _, _ = tree_series
        with pytest.raises(ValueError, match='model_combination'):
            HierarchicalForecastAdapter(HIERARCHY_TREE).fit(X_train, y_train)

    @pytest.mark.parametrize(
        ('forecast_columns', 'message'),
        [('m1', 'list of columns'), (['m3'], 'are not columns'), (['m1', 'm1'], 'duplicates'), ([], 'no base')],
    )
    def test_invalid_forecast_columns(self, tree_series, forecast_columns, message):
        X_train, y_train, _, _ = tree_series
        with pytest.raises(ValueError, match=message):
            HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=forecast_columns).fit(X_train, y_train)

    def test_non_numeric_forecast(self, tree_series):
        X_train, y_train, _, _ = tree_series
        X_text = X_train[['m1']].assign(m1=X_train['m1'].astype('str'))
        with pytest.raises(ValueError, match='not numeric'):
            HierarchicalForecastAdapter(HIERARCHY_TREE).fit(X_text, y_train)

    def test_exogenous_columns_are_ignored(self, tree_series):
        X_train, y_train, X_test, _ = tree_series
        reference = HierarchicalForecastAdapter(HIERARCHY_TREE).fit(X_train[['m1']], y_train)
        with_exog = [data[['m1']].assign(gdp=np.arange(len(data), dtype=float)) for data in (X_train, X_test)]
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(with_exog[0], y_train)
        np.testing.assert_allclose(adapter.predict(with_exog[1]), reference.predict(X_test[['m1']]))

    def test_interval_columns_are_no_base_forecast(self, tree_series):
        X_train, y_train, _, _ = tree_series
        X = X_train[['m1']].assign(**{'m1-lo-80': X_train['m1'] - 1, 'm1-hi-80': X_train['m1'] + 1})
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE).fit(X, y_train)
        assert adapter.forecast_columns_ == ['m1']

    def test_names_read_as_intervals_by_hierarchicalforecast(self, tree_series):
        X_train, y_train, X_test, _ = tree_series
        renamed = [data[['m1']].rename(columns={'m1': 'ets-lo-model'}) for data in (X_train, X_test)]
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE).fit(renamed[0], y_train)
        reference = HierarchicalForecastAdapter(HIERARCHY_TREE).fit(X_train[['m1']], y_train)
        assert adapter.reconcile(renamed[1]).columns.tolist() == ['ets-lo-model/BottomUp']
        np.testing.assert_allclose(adapter.predict(renamed[1]), reference.predict(X_test[['m1']]))

    @pytest.mark.parametrize(
        ('combination', 'message'),
        [
            ('median', "None, 'mean'"),
            (['m1', 'm2'], "None, 'mean'"),
            ({'depth_0': 'm1', 'depth_1': 'm2'}, 'missing levels'),
            ({'depth_0': 'm1', 'depth_1': 'm2', 'depth_2': 'm1', 'depth_3': 'm1'}, 'unknown levels'),
            ({'depth_0': 'm1', 'depth_1': 'm3', 'depth_2': 'm1'}, 'no base forecast'),
        ],
    )
    def test_invalid_combination(self, tree_series, combination, message):
        X_train, y_train, _, _ = tree_series
        with pytest.raises(ValueError, match=message):
            HierarchicalForecastAdapter(HIERARCHY_TREE, model_combination=combination).fit(X_train, y_train)

    def test_column_named_like_the_combination(self, tree_series):
        X_train, y_train, _, _ = tree_series
        X = X_train.rename(columns={'m2': 'combined'})
        with pytest.raises(ValueError, match='rename it'):
            HierarchicalForecastAdapter(HIERARCHY_TREE, model_combination='mean').fit(X, y_train)


class TestRowsToReconcile:
    """Rows that cannot be reconciled raise an explicit error."""

    def test_incomplete_date(self, tree_panel):
        X_train, y_train, X_test, _ = tree_panel
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train, y_train)
        dates = X_test.index.get_level_values('date')
        mask = (
            (X_test.index.get_level_values('entity') == 'B')
            & (X_test.index.get_level_values('node') == 'd')
            & (dates == dates.max())
        )
        with pytest.raises(ValueError, match=r"no row for the nodes of the copy \('B',\) 'd'"):
            adapter.predict(_drop_rows(X_test, mask))

    def test_missing_base_forecast(self, tree_series):
        X_train, y_train, X_test, _ = tree_series
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, model_combination='mean').fit(X_train, y_train)
        X_missing = X_test.copy()
        X_missing.iloc[0, 1] = np.nan
        with pytest.raises(ValueError, match="'combined' is missing"):
            adapter.predict(X_missing)
        with pytest.raises(ValueError, match="'m2' is missing"):
            adapter.reconcile(X_missing)

    def test_training_forecasts_may_be_missing(self, tree_series):
        X_train, y_train, X_test, _ = tree_series
        X_holes = X_train.copy()
        X_holes.iloc[::7, 0] = np.nan
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, MinTrace(method='mint_shrink'), forecast_columns=['m1'])
        assert np.isfinite(adapter.fit(X_holes, y_train).predict(X_test)).all()

    def test_unknown_node_of_a_level_hierarchy(self, levels_series):
        X_train, y_train, X_test, _ = levels_series
        adapter = HierarchicalForecastAdapter(HIERARCHY_LEVELS, forecast_columns=['m1']).fit(X_train, y_train)
        with pytest.raises(ValueError, match='unknown at fit'):
            adapter.predict(_rename_level(X_test, 'region', {'west': 'north-west'}))

    def test_unknown_copy_of_a_tree(self, tree_panel):
        X_train, y_train, X_test, _ = tree_panel
        known = X_train.index.get_level_values('entity') != 'C'
        bottom_up = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1'])
        bottom_up.fit(X_train[known], y_train[known])
        # Structure connue sans données : la copie C est réconciliée comme une copie connue
        reference = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train, y_train)
        np.testing.assert_allclose(bottom_up.predict(X_test), reference.predict(X_test))
        mint = HierarchicalForecastAdapter(HIERARCHY_TREE, MinTrace(method='mint_shrink'), forecast_columns=['m1'])
        mint.fit(X_train[known], y_train[known])
        with pytest.raises(ValueError, match=r"unknown at fit \(the reconciliation needs their training rows\)"):
            mint.predict(X_test)
        with pytest.raises(ValueError, match='unknown at fit'):
            bottom_up.reconcile(X_test, level=[80], intervals_method='bootstrap')

    def test_unknown_copy_of_a_level_hierarchy(self, levels_panel):
        X_train, y_train, X_test, _ = levels_panel
        known = X_train.index.get_level_values('market') != 'C'
        adapter = HierarchicalForecastAdapter(HIERARCHY_LEVELS, forecast_columns=['m1'])
        adapter.fit(X_train[known], y_train[known])
        with pytest.raises(ValueError, match='nodes are learnt from the training rows'):
            adapter.predict(X_test)

    def test_columns_differ_from_fit(self, tree_series):
        X_train, y_train, X_test, _ = tree_series
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train, y_train)
        with pytest.raises(ValueError):
            adapter.predict(X_test[['m1']])

    def test_empty_rows(self, tree_series):
        X_train, y_train, X_test, _ = tree_series
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train, y_train)
        empty = X_test.iloc[:0]
        assert adapter.predict(empty).shape == (0,)
        frame = adapter.reconcile(empty)
        assert frame.shape == (0, 0) and frame.index.equals(empty.index)


class TestReconcilerRequirements:
    """Strict hierarchies for top-down and middle-out, training rows for in-sample reconcilers."""

    def test_top_down_on_a_tree_with_leaves_at_several_depths(self, tree_series):
        X_train, y_train, _, _ = tree_series
        with pytest.raises(ValueError, match='finest level'):
            HierarchicalForecastAdapter(
                HIERARCHY_TREE, TopDown(method='forecast_proportions'), forecast_columns=['m1']
            ).fit(X_train, y_train)

    def test_top_down_on_a_balanced_tree(self):
        dates = pd.date_range('2024-01-01', periods=3, freq='MS')
        nodes = ['t', 'x', 'y', 'x1', 'x2', 'y1', 'y2']
        index = pd.MultiIndex.from_product([nodes, dates], names=['node', 'date'])
        rng = np.random.default_rng(1)
        X = pd.DataFrame({'m': rng.uniform(1, 10, len(index))}, index=index)
        adapter = HierarchicalForecastAdapter(BALANCED_TREE, TopDown(method='forecast_proportions')).fit(X, X['m'])
        wide = pd.Series(adapter.predict(X), index=index).unstack('node')
        np.testing.assert_allclose(wide['t'], wide[['x1', 'x2', 'y1', 'y2']].sum(axis=1))
        np.testing.assert_allclose(wide['t'], X['m'].unstack('node')['t'])

    def test_middle_level_must_be_a_level(self, levels_series):
        X_train, y_train, X_test, _ = levels_series
        middle = MiddleOut(middle_level='region', top_down_method='forecast_proportions')
        with pytest.raises(ValueError, match='middle_level'):
            HierarchicalForecastAdapter(HIERARCHY_LEVELS, middle, forecast_columns=['m1']).fit(X_train, y_train)
        middle = MiddleOut(middle_level='country', top_down_method='forecast_proportions')
        adapter = HierarchicalForecastAdapter(HIERARCHY_LEVELS, middle, forecast_columns=['m1']).fit(X_train, y_train)
        assert np.isfinite(adapter.predict(X_test)).all()

    def test_in_sample_reconciler_needs_every_node(self, tree_series):
        X_train, y_train, _, _ = tree_series
        y_missing = y_train.mask(y_train.index.get_level_values('node') == 'c')
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, MinTrace(method='mint_shrink'), forecast_columns=['m1'])
        with pytest.raises(ValueError, match="nodes 'c' have no training row"):
            adapter.fit(X_train, y_missing)
        # Sans données insample, la cible manquante d'un nœud ne gêne pas
        HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train, y_missing)

    def test_extra_in_sample_reconciler_at_reconcile(self, tree_series):
        X_train, y_train, X_test, _ = tree_series
        y_missing = y_train.mask(y_train.index.get_level_values('node') == 'c')
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train, y_missing)
        with pytest.raises(ValueError, match='training rows of every node'):
            adapter.reconcile(X_test, MinTrace(method='mint_shrink'))

    def test_bootstrap_needs_more_training_dates(self, tree_series):
        X_train, y_train, X_test, _ = tree_series
        dates = X_train.index.get_level_values('date')
        first = dates <= dates.min() + pd.DateOffset(months=1)
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, forecast_columns=['m1']).fit(X_train[first], y_train[first])
        with pytest.raises(ValueError, match='consecutive training dates'):
            adapter.reconcile(X_test, level=[80], intervals_method='bootstrap')
