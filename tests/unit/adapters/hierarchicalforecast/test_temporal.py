"""Unit tests of the temporal and cross-temporal modes of ``HierarchicalForecastAdapter``.

Temporal coherence (months sum to quarters, quarters to years) of a series, of
balanced and unbalanced panels, on start- and end-labelled grids and on grouped
temporal levels, for ``BottomUp``, ``MinTrace`` ``'ols'`` / ``'wls_struct'`` and
``TopDown``; golden values (bottom-up sums, ``'ols'`` projection of each period).
Cross-temporal coherence in both dimensions (node tree and index levels), the
single Kronecker pass equal to two chained passes for ``'ols'``, strict
reconcilers refused. Temporal and cross-temporal forecasts equal to
hierarchicalforecast used natively (``aggregate_temporal`` and
``reconcile(temporal=True)``, two chained native passes for the cross-temporal
case), in-sample reconcilers refused by both. Then complete periods required at predict, misdated
aggregates, in-sample reconcilers and intervals refused, unknown copies and
nodes, row order, date resolution, immutability, ``temporal_level``,
``reconcile`` / ``diagnose`` / ``summing_matrix`` outputs, and the contract rules
that do not cut test windows on arbitrary dates.
"""
import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('hierarchicalforecast')

from hierarchicalforecast.methods import BottomUp, MinTrace, TopDown  # noqa: E402
from sklearn.base import clone  # noqa: E402

from tests.support import native_references as native  # noqa: E402
from tests.support.adapter_cases import hierarchical_cross_temporal_case, hierarchical_temporal_case  # noqa: E402
from tests.support.adapter_contract import (  # noqa: E402
    check_invalid_params,
    check_missing_package,
    check_wrapped_object_untouched,
    split_by_date,
)
from tests.support.datasets import (  # noqa: E402
    HIERARCHY_LEVELS,
    HIERARCHY_TREE,
    TEMPORAL_HIERARCHY,
    build_cross_temporal_hierarchy,
    build_temporal_hierarchy,
)
from tests.support.perturbations import (  # noqa: E402
    DATETIME_UNITS,
    datetime_units,
    shuffle_rows,
    with_datetime_resolution,
)
from tsforecast.adapters import HierarchicalForecastAdapter, period_splits  # noqa: E402

# Tolérance de la cohérence des prévisions réconciliées
COHERENCE_ATOL = 1e-8
# Réconciliateurs sans données insample
RECONCILERS = {
    'bottomup': BottomUp,
    'ols': lambda: MinTrace(method='ols'),
    'wls_struct': lambda: MinTrace(method='wls_struct'),
}
# Hiérarchie temporelle trimestrielle
QUARTERLY = {'year': 4, 'semester': 2, 'quarter': 1}


def _last_year(X, y, hierarchy=TEMPORAL_HIERARCHY):
    """``(X_train, y_train, X_test, y_test)``: the last calendar period for test."""
    train, test = period_splits(X, hierarchy, n_splits=1)[0]
    return X.iloc[train], y.iloc[train], X.iloc[test], y.iloc[test]


def _temporal(reconciler, **params):
    """Temporal adapter on the base model ``m1``."""
    params.setdefault('forecast_columns', ['m1'])
    params.setdefault('temporal_hierarchy', TEMPORAL_HIERARCHY)
    return HierarchicalForecastAdapter(None, reconciler, **params)


def _predict(adapter, X_train, y_train, X_test):
    """Fit, then predict ``X_test`` as a ``Series`` indexed like it."""
    return pd.Series(adapter.fit(X_train, y_train).predict(X_test), index=X_test.index)


def _assert_temporal_coherence(values, steps=(12, 3, 1), names=('year', 'quarter', 'month')):
    """Each aggregate equals the sum of the fine rows of its calendar block, series by series."""
    frame = values.rename('value').reset_index()
    copies = [name for name in values.index.names if name not in ('level', 'date')]
    dates = frame['date']
    # Rang du pas fin de chaque ligne dans l'année (grilles mensuelles ou trimestrielles)
    year_steps = steps[0]
    frame['step'] = (dates.dt.month - 1) // (12 // year_steps)
    frame['year'] = dates.dt.year
    fine = frame[frame['level'] == names[-1]]
    for name, size in zip(names[:-1], steps[:-1]):
        rows = frame[frame['level'] == name]
        # Rang du bloc d'une ligne d'agrégat : par sa date de début ou de fin selon la grille
        block = lambda part: part['step'] // size  # noqa: E731
        expected = fine.assign(block=block(fine)).groupby([*copies, 'year', 'block'])['value'].sum()
        actual = rows.assign(block=block(rows)).set_index([*copies, 'year', 'block'])['value']
        np.testing.assert_allclose(actual.sort_index().to_numpy(), expected.loc[actual.sort_index().index].to_numpy(),
                                   atol=COHERENCE_ATOL)
        assert len(actual) == len(expected)


def _assert_tree_coherence(values):
    """``a = b + c`` and ``c = d + e`` on every (copy, temporal level, date)."""
    wide = values.unstack('node')
    np.testing.assert_allclose(wide['a'], wide['b'] + wide['c'], atol=COHERENCE_ATOL)
    np.testing.assert_allclose(wide['c'], wide['d'] + wide['e'], atol=COHERENCE_ATOL)


def _assert_levels_coherence(values):
    """Each country equals the sum of its regions on every (copy, temporal level, date)."""
    frame = values.rename('value').reset_index()
    totals = frame[frame['region'] == 'Total'].drop(columns='region')
    regions = frame[frame['region'] != 'Total']
    keys = [name for name in totals.columns if name != 'value']
    expected = regions.groupby(keys)['value'].sum()
    np.testing.assert_allclose(totals.set_index(keys)['value'].loc[expected.index], expected, atol=COHERENCE_ATOL)


class TestTemporalCoherence:
    """Months sum to quarters and quarters to years after reconciliation."""

    @pytest.mark.parametrize('kind', ['series', 'panel', 'unbalanced_panel'])
    @pytest.mark.parametrize('reconciler', sorted(RECONCILERS))
    def test_coherent(self, kind, reconciler):
        X_train, y_train, X_test, _ = _last_year(*build_temporal_hierarchy(kind))
        _assert_temporal_coherence(_predict(_temporal(RECONCILERS[reconciler]()), X_train, y_train, X_test))

    def test_top_down_on_nested_levels(self):
        X_train, y_train, X_test, _ = _last_year(*build_temporal_hierarchy('panel'))
        adapter = _temporal(TopDown(method='forecast_proportions'))
        values = _predict(adapter, X_train, y_train, X_test)
        _assert_temporal_coherence(values)
        # Le total annuel garde sa prévision de base
        years = X_test.index.get_level_values('level') == 'year'
        np.testing.assert_allclose(values[years], X_test.loc[years, 'm1'])

    @pytest.mark.parametrize('freq', ['ME', 'MS'])
    def test_end_and_start_labelled_grids(self, freq):
        X_train, y_train, X_test, _ = _last_year(*build_temporal_hierarchy('panel', freq=freq))
        adapter = _temporal(MinTrace(method='ols'))
        _assert_temporal_coherence(_predict(adapter, X_train, y_train, X_test))
        assert adapter.temporal_freq_ == freq

    def test_quarterly_grid(self):
        X, y = build_temporal_hierarchy('series', freq='QS', temporal_hierarchy=QUARTERLY)
        X_train, y_train, X_test, _ = _last_year(X, y, QUARTERLY)
        values = _predict(_temporal(MinTrace(method='ols'), temporal_hierarchy=QUARTERLY), X_train, y_train, X_test)
        _assert_temporal_coherence(values, steps=(4, 2, 1), names=('year', 'semester', 'quarter'))

    def test_grouped_temporal_levels(self):
        hierarchy = {'year': 12, 'four_months': 4, 'quarter': 3, 'month': 1}
        X, y = build_temporal_hierarchy('series', temporal_hierarchy=hierarchy)
        X_train, y_train, X_test, _ = _last_year(X, y, hierarchy)
        values = _predict(_temporal(MinTrace(method='ols'), temporal_hierarchy=hierarchy), X_train, y_train, X_test)
        _assert_temporal_coherence(values, steps=(12, 4, 1), names=('year', 'four_months', 'month'))
        _assert_temporal_coherence(values)

    def test_bottom_up_sums_the_monthly_base_forecasts(self):
        X_train, y_train, X_test, _ = _last_year(*build_temporal_hierarchy('series'))
        values = _predict(_temporal(BottomUp()), X_train, y_train, X_test)
        months = X_test.index.get_level_values('level') == 'month'
        np.testing.assert_allclose(values[months], X_test.loc[months, 'm1'])
        assert values.xs('year', level='level').iloc[0] == pytest.approx(X_test.loc[months, 'm1'].sum())

    def test_ols_is_the_orthogonal_projection_of_each_period(self):
        X, y = build_temporal_hierarchy('series', n_years=4)
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=24)
        adapter = _temporal(MinTrace(method='ols')).fit(X_train, y_train)
        values = pd.Series(adapter.predict(X_test), index=X_test.index)
        summing = adapter.summing_matrix()
        projection = summing.to_numpy() @ np.linalg.pinv(summing.to_numpy())
        # Chaque année se réconcilie seule : projection de ses 17 prévisions de base
        for year in (2022, 2023):
            rows = X_test.index.get_level_values('date').year == year
            base = X_test.loc[rows, 'm1']
            order = [f'{level}-{rank}' for level, rank in _node_ranks(base.index)]
            vector = pd.Series(base.to_numpy(), index=order).loc[summing.index]
            expected = pd.Series(projection @ vector.to_numpy(), index=summing.index).loc[order]
            np.testing.assert_allclose(values[rows].to_numpy(), expected.to_numpy(), atol=1e-9)


def _node_ranks(index):
    """``(level, rank in the year)`` of each row of a monthly temporal hierarchy dated by period start."""
    levels = index.get_level_values('level')
    months = index.get_level_values('date').month
    ranks = np.where(levels == 'month', months, np.where(levels == 'quarter', (months - 1) // 3 + 1, 1))
    return list(zip(levels, ranks))


class TestCrossTemporal:
    """Coherence in the hierarchy and across the temporal levels at once."""

    @pytest.mark.parametrize('kind', ['series', 'panel', 'unbalanced_panel'])
    @pytest.mark.parametrize('reconciler', sorted(RECONCILERS))
    def test_tree_coherent_in_both_dimensions(self, kind, reconciler):
        X_train, y_train, X_test, _ = _last_year(*build_cross_temporal_hierarchy(kind))
        adapter = HierarchicalForecastAdapter(
            HIERARCHY_TREE, RECONCILERS[reconciler](), model_combination='mean', temporal_hierarchy=TEMPORAL_HIERARCHY
        )
        values = _predict(adapter, X_train, y_train, X_test)
        _assert_tree_coherence(values)
        _assert_temporal_coherence(values)

    @pytest.mark.parametrize('kind', ['series', 'panel'])
    def test_levels_coherent_in_both_dimensions(self, kind):
        X_train, y_train, X_test, _ = _last_year(*build_cross_temporal_hierarchy(kind, levels=True))
        adapter = HierarchicalForecastAdapter(
            HIERARCHY_LEVELS, MinTrace(method='ols'), forecast_columns=['m2'], temporal_hierarchy=TEMPORAL_HIERARCHY
        )
        values = _predict(adapter, X_train, y_train, X_test)
        _assert_levels_coherence(values)
        _assert_temporal_coherence(values)
        assert adapter.hierarchy_levels_[0] == 'country//year'

    def test_single_pass_equals_two_chained_ols_passes(self):
        X_train, y_train, X_test, _ = _last_year(*build_cross_temporal_hierarchy('series'))
        one_pass = HierarchicalForecastAdapter(
            HIERARCHY_TREE, MinTrace(method='ols'), forecast_columns=['m1'], temporal_hierarchy=TEMPORAL_HIERARCHY
        )
        values = _predict(one_pass, X_train, y_train, X_test)
        # Passe temporelle (chaque nœud est une copie), puis passe cross-section (chaque niveau temporel est une copie)
        temporal = _temporal(MinTrace(method='ols')).fit(X_train[['m1']], y_train)
        X_mid = X_test[['m1']].assign(m1=temporal.predict(X_test[['m1']]))
        cross = HierarchicalForecastAdapter(HIERARCHY_TREE, MinTrace(method='ols'), node_level='node')
        cross.fit(X_train[['m1']], y_train)
        np.testing.assert_allclose(values, cross.predict(X_mid), atol=1e-9)

    def test_model_per_cross_level(self):
        X_train, y_train, X_test, _ = _last_year(*build_cross_temporal_hierarchy('series'))
        adapter = HierarchicalForecastAdapter(HIERARCHY_TREE, BottomUp(), temporal_hierarchy=TEMPORAL_HIERARCHY)
        levels = [f'depth_{depth}//{level}' for depth in range(3) for level in ('year', 'quarter', 'month')]
        adapter.set_params(model_combination={level: 'm1' if 'month' in level else 'm2' for level in levels})
        values = _predict(adapter, X_train, y_train, X_test)
        _assert_tree_coherence(values)

    @pytest.mark.parametrize('reconciler', [TopDown(method='forecast_proportions')])
    def test_strict_reconcilers_refused(self, reconciler):
        X_train, y_train, _, _ = _last_year(*build_cross_temporal_hierarchy('series'))
        adapter = HierarchicalForecastAdapter(
            {'t': ['x', 'y']}, reconciler, forecast_columns=['m1'], temporal_hierarchy=TEMPORAL_HIERARCHY
        )
        # Arbre t = x + y : nœuds a, b, c renommés
        kept = X_train.index.get_level_values('node').isin(['a', 'b', 'c'])
        names = {'a': 't', 'b': 'x', 'c': 'y'}
        tree_X = X_train[kept].rename(index=names, level='node')
        tree_y = y_train[kept].rename(index=names, level='node')
        with pytest.raises(ValueError, match='strictly hierarchical'):
            adapter.fit(tree_X, tree_y)

    def test_summing_matrix_is_the_kronecker_product(self):
        X_train, y_train, _, _ = _last_year(*build_cross_temporal_hierarchy('panel'))
        adapter = HierarchicalForecastAdapter(
            HIERARCHY_TREE, BottomUp(), forecast_columns=['m1'], temporal_hierarchy=TEMPORAL_HIERARCHY
        ).fit(X_train, y_train)
        matrix = adapter.summing_matrix('A')
        assert matrix.shape == (5 * 17, 3 * 12)
        assert list(matrix.index.names) == ['node', 'level']
        assert matrix.loc[('a', 'year-1')].sum() == 36
        assert matrix.loc[('c', 'quarter-2')].sum() == 6


class TestPeriods:
    """Complete periods required at predict; dating and labels checked."""

    def test_incomplete_period_refused_at_predict_only(self):
        X, y = build_temporal_hierarchy('panel')
        X_train, y_train, X_test, _ = _last_year(X, y)
        # Entraînement : une année incomplète est acceptée (aucune donnée insample)
        dropped = (X_train.index.get_level_values('level') == 'quarter') & (
            X_train.index.get_level_values('date') == pd.Timestamp('2021-04-01')
        )
        adapter = _temporal(BottomUp()).fit(X_train[~dropped], y_train[~dropped])
        # Test : un trimestre manquant rend l'année incomplète
        missing = (X_test.index.get_level_values('level') == 'quarter') & (
            X_test.index.get_level_values('entity') == 'B'
        )
        with pytest.raises(ValueError, match=r"copy \('B',\) 'quarter-1'.*period starting 2023-01-01"):
            adapter.predict(X_test[~(missing & (X_test.index.get_level_values('date').month == 1))])

    def test_test_window_cut_inside_a_year_refused(self):
        X, y = build_temporal_hierarchy('series')
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=3)
        adapter = _temporal(BottomUp()).fit(X_train, y_train)
        with pytest.raises(ValueError, match='incomplete period'):
            adapter.predict(X_test)

    def test_misdated_aggregate_refused(self):
        X, y = build_temporal_hierarchy('series', n_years=2)
        dates = X.index.get_level_values('date')
        moved = np.where(X.index.get_level_values('level') == 'year', dates + pd.DateOffset(months=1), dates)
        X_moved = X.set_axis(pd.MultiIndex.from_arrays(
            [X.index.get_level_values('level'), pd.DatetimeIndex(moved)], names=X.index.names))
        with pytest.raises(ValueError, match="level 'year' is dated"):
            _temporal(BottomUp()).fit(X_moved, y.set_axis(X_moved.index))

    def test_unknown_temporal_label(self):
        X, y = build_temporal_hierarchy('series', n_years=2)
        with pytest.raises(ValueError, match="'quarter' of the temporal index level"):
            _temporal(BottomUp(), temporal_hierarchy={'year': 12, 'month': 1}).fit(X, y)

    def test_node_level_needs_a_hierarchy(self):
        X, y = build_temporal_hierarchy('series', n_years=2)
        with pytest.raises(ValueError, match='node_level only applies with a hierarchy'):
            _temporal(BottomUp(), node_level='level').fit(X, y)

    def test_temporal_level_by_name_before_the_nodes(self):
        X, y = build_cross_temporal_hierarchy('series')
        X_train, y_train, X_test, _ = _last_year(X, y)
        reference = _predict(
            HierarchicalForecastAdapter(HIERARCHY_TREE, MinTrace(method='ols'), forecast_columns=['m1'],
                                        temporal_hierarchy=TEMPORAL_HIERARCHY),
            X_train, y_train, X_test,
        )
        swap = ['level', 'node', 'date']
        adapter = HierarchicalForecastAdapter(
            HIERARCHY_TREE, MinTrace(method='ols'), forecast_columns=['m1'],
            temporal_hierarchy=TEMPORAL_HIERARCHY, temporal_level='level',
        )
        values = adapter.fit(X_train.reorder_levels(swap), y_train.reorder_levels(swap)).predict(
            X_test.reorder_levels(swap)
        )
        np.testing.assert_allclose(values, reference.to_numpy(), atol=1e-9)


class TestInsampleRefused:
    """Temporal reconciliation reads no in-sample errors."""

    def test_in_sample_reconciler_refused_at_fit(self):
        X_train, y_train, _, _ = _last_year(*build_temporal_hierarchy('series'))
        with pytest.raises(ValueError, match='learns from in-sample errors, which temporal'):
            _temporal(MinTrace(method='mint_shrink')).fit(X_train, y_train)

    def test_in_sample_reconciler_refused_by_reconcile(self):
        X_train, y_train, X_test, _ = _last_year(*build_cross_temporal_hierarchy('series'))
        adapter = HierarchicalForecastAdapter(
            HIERARCHY_TREE, forecast_columns=['m1'], temporal_hierarchy=TEMPORAL_HIERARCHY
        ).fit(X_train, y_train)
        with pytest.raises(ValueError, match='in-sample errors'):
            adapter.reconcile(X_test, [BottomUp(), MinTrace(method='wls_var')])

    @pytest.mark.parametrize('method', ['bootstrap', 'permbu', 'conformal'])
    def test_in_sample_intervals_refused(self, method):
        X_train, y_train, X_test, _ = _last_year(*build_temporal_hierarchy('series'))
        adapter = _temporal(BottomUp()).fit(X_train, y_train)
        with pytest.raises(ValueError, match=f"intervals_method='{method}' reads in-sample errors"):
            adapter.reconcile(X_test, level=[80], intervals_method=method)

    def test_normality_intervals(self):
        X, y = build_temporal_hierarchy('series')
        X = X.assign(**{'m1-lo-80': X['m1'] - 2.0, 'm1-hi-80': X['m1'] + 2.0})
        X_train, y_train, X_test, _ = _last_year(X, y)
        adapter = _temporal(MinTrace(method='ols')).fit(X_train, y_train)
        frame = adapter.reconcile(X_test, level=[80])
        assert frame.columns.tolist() == [
            'm1/MinTrace_method-ols', 'm1/MinTrace_method-ols-lo-80', 'm1/MinTrace_method-ols-hi-80'
        ]
        assert (frame.iloc[:, 1] < frame.iloc[:, 0]).all() and (frame.iloc[:, 2] > frame.iloc[:, 0]).all()


class TestCopies:
    """Copies unknown at fit and nodes."""

    def test_unknown_series_of_a_temporal_hierarchy(self):
        X, y = build_temporal_hierarchy('panel')
        X_train, y_train, X_test, _ = _last_year(X, y)
        adapter = _temporal(MinTrace(method='ols')).fit(X_train[X_train.index.get_level_values('entity') != 'C'],
                                                        y_train[y_train.index.get_level_values('entity') != 'C'])
        _assert_temporal_coherence(pd.Series(adapter.predict(X_test), index=X_test.index))

    def test_unknown_copy_of_a_tree(self):
        X, y = build_cross_temporal_hierarchy('panel')
        X_train, y_train, X_test, _ = _last_year(X, y)
        known = X_train.index.get_level_values('entity') == 'A'
        adapter = HierarchicalForecastAdapter(
            HIERARCHY_TREE, BottomUp(), forecast_columns=['m1'], temporal_hierarchy=TEMPORAL_HIERARCHY
        ).fit(X_train[known], y_train[known])
        _assert_tree_coherence(pd.Series(adapter.predict(X_test), index=X_test.index))

    def test_unknown_copy_of_levels_refused(self):
        X, y = build_cross_temporal_hierarchy('panel', levels=True)
        X_train, y_train, X_test, _ = _last_year(X, y)
        known = X_train.index.get_level_values('entity') == 'A'
        adapter = HierarchicalForecastAdapter(
            HIERARCHY_LEVELS, BottomUp(), forecast_columns=['m1'], temporal_hierarchy=TEMPORAL_HIERARCHY
        ).fit(X_train[known], y_train[known])
        with pytest.raises(ValueError, match='Copies of the hierarchy unknown at fit'):
            adapter.predict(X_test)

    def test_unknown_node_refused(self):
        X, y = build_cross_temporal_hierarchy('series')
        X_train, y_train, X_test, _ = _last_year(X, y)
        adapter = HierarchicalForecastAdapter(
            HIERARCHY_TREE, BottomUp(), forecast_columns=['m1'], temporal_hierarchy=TEMPORAL_HIERARCHY
        ).fit(X_train, y_train)
        renamed = X_test.rename(index={'e': 'z'}, level='node')
        with pytest.raises(ValueError, match="unknown at fit: 'z'"):
            adapter.predict(renamed)


class TestOutputs:
    """Row order, dates, immutability and detailed outputs."""

    @pytest.fixture
    def fitted(self):
        X_train, y_train, X_test, y_test = _last_year(*build_cross_temporal_hierarchy('panel'))
        adapter = HierarchicalForecastAdapter(
            HIERARCHY_TREE, MinTrace(method='ols'), model_combination='mean', temporal_hierarchy=TEMPORAL_HIERARCHY
        )
        return adapter.fit(X_train, y_train), X_test

    def test_row_order(self, fitted):
        adapter, X_test = fitted
        values = pd.Series(adapter.predict(X_test), index=X_test.index)
        shuffled = shuffle_rows(X_test)
        np.testing.assert_allclose(adapter.predict(shuffled), values.loc[shuffled.index].to_numpy())

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_datetime_resolution(self, unit):
        X_train, y_train, X_test, _ = _last_year(*build_temporal_hierarchy('panel'))
        adapter = _temporal(MinTrace(method='ols'))
        reference = adapter.fit(X_train, y_train).predict(X_test)
        X_train_u, y_train_u = with_datetime_resolution(X_train, unit), with_datetime_resolution(y_train, unit)
        X_test_u = with_datetime_resolution(X_test, unit)
        converted = clone(adapter).fit(X_train_u, y_train_u)
        np.testing.assert_allclose(converted.predict(X_test_u), reference)
        assert set(datetime_units(converted.reconcile(X_test_u)).values()) == {unit}

    def test_nothing_modified(self, fitted):
        adapter, X_test = fitted
        reconciler = adapter.reconciler
        state, wrapped = joblib.hash(adapter.__dict__), joblib.hash(reconciler)
        first = adapter.predict(X_test)
        adapter.reconcile(X_test, [BottomUp(), MinTrace(method='wls_struct')])
        adapter.diagnose(X_test)
        assert joblib.hash(adapter.__dict__) == state
        assert joblib.hash(reconciler) == wrapped
        np.testing.assert_array_equal(adapter.predict(X_test), first)

    def test_reconcile_and_diagnose(self, fitted):
        adapter, X_test = fitted
        frame = adapter.reconcile(X_test, [BottomUp(), MinTrace(method='ols')])
        assert frame.index.equals(X_test.index)
        assert 'combined/MinTrace_method-ols' in frame.columns and 'm2/BottomUp' in frame.columns
        diagnostics = adapter.diagnose(X_test)
        assert diagnostics.columns[0] == 'entity'
        assert set(diagnostics['level']) >= {'depth_0//year', 'depth_2//month', 'Overall'}
        coherent = diagnostics[(diagnostics['level'] == 'Overall') & (diagnostics['metric'] == 'is_coherent')]
        assert (coherent['combined/MinTrace_method-ols'] == 1.0).all()

    def test_fitted_attributes(self, fitted):
        adapter, _ = fitted
        assert adapter.copies_ == (('A',), ('B',), ('C',))
        assert adapter.temporal_freq_ == 'MS'
        assert len(adapter.hierarchy_levels_) == 9

    def test_temporal_summing_matrix(self):
        X_train, y_train, _, _ = _last_year(*build_temporal_hierarchy('series'))
        matrix = _temporal(BottomUp()).fit(X_train, y_train).summing_matrix()
        assert matrix.shape == (17, 12)
        assert matrix.index.name == 'level'
        assert matrix.index[:2].tolist() == ['year-1', 'quarter-1']

    def test_clone_keeps_temporal_parameters(self):
        adapter = _temporal(BottomUp(), temporal_level='level', temporal_origin='2019-01-01')
        params = clone(adapter).get_params()
        assert params['temporal_level'] == 'level' and params['temporal_origin'] == '2019-01-01'


# Fonction auxiliaire de découpage : deux dernières années civiles en test
def _last_two_years(X, y):
    """``(X_train, y_train, X_test)``: the last two calendar years for test."""
    train, test = period_splits(X, TEMPORAL_HIERARCHY, n_splits=1, test_periods=2)[0]
    return X.iloc[train], y.iloc[train], X.iloc[test]


class TestNativeEquivalence:
    """Same forecasts as hierarchicalforecast used natively, on two test years."""

    @pytest.mark.parametrize('kind', ['series', 'panel'])
    @pytest.mark.parametrize('reconciler', ['bottomup', 'ols'])
    def test_temporal_equals_aggregate_temporal_and_reconcile(self, kind, reconciler):
        """``aggregate_temporal`` then ``reconcile(temporal=True)`` on every series."""
        X_train, y_train, X_test = _last_two_years(*build_temporal_hierarchy(kind, n_years=4))
        predictions = _temporal(RECONCILERS[reconciler]()).fit(X_train, y_train).predict(X_test)
        expected = native.temporal_reference(X_test, RECONCILERS[reconciler](), 'm1')
        np.testing.assert_allclose(predictions, expected, rtol=1e-7, atol=1e-9)

    @pytest.mark.parametrize('kind', ['series', 'panel'])
    @pytest.mark.parametrize('reconciler', ['bottomup', 'ols'])
    @pytest.mark.parametrize('levels', [False, True], ids=['tree', 'levels'])
    def test_cross_temporal_equals_two_native_passes(self, kind, reconciler, levels):
        """Cross-sectional ``reconcile`` of each temporal level, then temporal ``reconcile`` of each node."""
        X_train, y_train, X_test = _last_two_years(*build_cross_temporal_hierarchy(kind, n_years=4, levels=levels))
        hierarchy, structure = (HIERARCHY_LEVELS, native.LEVELS) if levels else (HIERARCHY_TREE, native.TREE)
        adapter = HierarchicalForecastAdapter(
            hierarchy, RECONCILERS[reconciler](), forecast_columns=['m1'], temporal_hierarchy=TEMPORAL_HIERARCHY
        )
        predictions = adapter.fit(X_train, y_train).predict(X_test)
        expected = native.cross_temporal_reference(X_test, RECONCILERS[reconciler](), 'm1', structure)
        np.testing.assert_allclose(predictions, expected, rtol=1e-7, atol=1e-9)

    def test_insample_reconciler_refused_by_both(self):
        """hierarchicalforecast refuses an in-sample reconciler in temporal mode, as the adapter does."""
        from hierarchicalforecast.core import HierarchicalReconciliation
        from hierarchicalforecast.utils import aggregate_temporal

        X, y = build_temporal_hierarchy(n_years=2)
        with pytest.raises(ValueError, match='in-sample|insample'):
            _temporal(MinTrace(method='mint_shrink')).fit(X, y)
        months = X.xs('month', level='level').index
        Y_df, S_df, tags = aggregate_temporal(pd.DataFrame({'unique_id': 'series', 'ds': months, 'y': 0.0}), TEMPORAL_HIERARCHY)
        # Sans données insample, seules admises en temporel : refus du réconciliateur insample
        with pytest.raises(NotImplementedError, match='not supported for `MinTrace_method-mint_shrink`'):
            HierarchicalReconciliation([MinTrace(method='mint_shrink')]).reconcile(
                Y_df.assign(model=1.0).drop(columns='y'), tags, S_df=S_df, temporal=True
            )


@pytest.mark.parametrize('make_case', [hierarchical_temporal_case, hierarchical_cross_temporal_case])
class TestTemporalContract:
    """Contract rules that do not cut test windows on arbitrary dates."""

    def test_wrapped_object_untouched(self, make_case):
        check_wrapped_object_untouched(make_case())

    def test_invalid_params(self, make_case):
        check_invalid_params(make_case())

    def test_missing_package(self, make_case, monkeypatch):
        check_missing_package(make_case(), monkeypatch)


class TestCoherenceChecks:
    """The coherence checks of this module reject the incoherent base forecasts."""

    def test_temporal_check(self):
        X, _ = build_temporal_hierarchy('series', n_years=1)
        with pytest.raises(AssertionError):
            _assert_temporal_coherence(X['m1'])

    def test_tree_check(self):
        X, _ = build_cross_temporal_hierarchy('series', n_years=1)
        with pytest.raises(AssertionError):
            _assert_tree_coherence(X['m1'])
