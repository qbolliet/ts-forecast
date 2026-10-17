"""Integration of the cross-validation splitters with sklearn model selection and the package pipeline.

Covered: panel splitters (global and per-entity, out of sample and in sample)
driving ``cross_validate`` and ``GridSearchCV`` over an ``XYPipeline`` made of a
``PanelwiseTransformer`` and a trivial estimator on the realistic
heterogeneous-coverage panel, with the transformer fitted on training rows
only and explicit ``groups`` routed to the splitter; ``TSOutOfSampleSplit`` /
``TSInSampleSplit`` on the realistic irregular monthly series with
``DropNaNTarget``; forecasting workflows whose exact linear target gives a
known score; time-series edge cases; large datasets.
"""
# Importation des modules
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler

from tests.support.estimators import RecordingEstimator
from tests.support.splits import (
    as_lists,
    assert_folds_cover,
    assert_in_sample,
    assert_no_leakage,
    collect_folds,
)
from tsforecast.crossvals import (
    InSampleSplit,
    PanelInSampleSplit,
    PanelInSampleSplitPerEntity,
    PanelOutOfSampleSplit,
    PanelOutOfSampleSplitPerEntity,
    TSInSampleSplit,
    TSOutOfSampleSplit,
)
from tsforecast.panel import PanelwiseTransformer
from tsforecast.xy import DropNaNTarget, XYPipeline

# Dates de test portées par les trois entités du panel réaliste
PANEL_DATES = ['2022-01-01', '2023-01-01']

# Splitters de panel utilisables sur le panel réaliste (toutes les entités portent ces dates)
PANEL_SPLITTERS = [
    pytest.param(PanelOutOfSampleSplit(test_indices=PANEL_DATES, test_size=3, gap=1), id='global-out'),
    pytest.param(PanelInSampleSplit(test_indices=PANEL_DATES), id='global-in'),
    pytest.param(PanelOutOfSampleSplitPerEntity(test_indices=PANEL_DATES, test_size=3, gap=1), id='per-entity-out'),
    pytest.param(PanelInSampleSplitPerEntity(test_indices=PANEL_DATES), id='per-entity-in'),
]


@pytest.fixture
def panel_xy(heterogeneous_coverage_panel):
    """Covariate ``inflation_ipc`` and target ``taux_chomage`` of the realistic panel, rows without NaN."""
    # Lignes où covariable et cible sont observées : grille mensuelle de chaque entité
    data = heterogeneous_coverage_panel[['inflation_ipc', 'taux_chomage']].dropna()
    return data[['inflation_ipc']], data['taux_chomage']


def _scaling_pipeline(model) -> XYPipeline:
    """Per-entity standardization followed by ``model``."""
    return XYPipeline([('scale', PanelwiseTransformer(StandardScaler())), ('model', model)])


class TestPanelSplittersInCrossValidate:
    """Panel splitters driving cross_validate over XYPipeline + PanelwiseTransformer."""

    @pytest.mark.parametrize('splitter', PANEL_SPLITTERS)
    def test_cross_validate_uses_the_folds_of_the_splitter(self, panel_xy, splitter):
        """cross_validate evaluates exactly the folds that split produces."""
        X, y = panel_xy
        result = cross_validate(_scaling_pipeline(DummyRegressor()), X, y, cv=splitter,
                                scoring='neg_mean_absolute_error', return_indices=True)
        produced = [(train.tolist(), test.tolist()) for train, test in zip(*result['indices'].values())]
        assert produced == as_lists(collect_folds(splitter, X))

    @pytest.mark.parametrize('splitter', PANEL_SPLITTERS)
    def test_scores_are_finite(self, panel_xy, splitter):
        """Every fold gives a finite score."""
        X, y = panel_xy
        result = cross_validate(_scaling_pipeline(DummyRegressor()), X, y, cv=splitter,
                                scoring='neg_mean_absolute_error')
        assert np.isfinite(result['test_score']).all()

    def test_transformer_learns_on_the_training_rows_only(self, panel_xy):
        """Test rows are standardized with the mean and deviation of their entity's training rows."""
        X, y = panel_xy
        splitter = PanelOutOfSampleSplit(test_indices=PANEL_DATES, test_size=3, gap=1)
        result = cross_validate(_scaling_pipeline(RecordingEstimator(DummyRegressor())), X, y, cv=splitter,
                                scoring='neg_mean_absolute_error', return_estimator=True, return_indices=True)
        for estimator, train, test in zip(result['estimator'], *result['indices'].values()):
            # Valeur attendue : (x - moyenne) / écart-type (ddof=0) de l'entité sur l'apprentissage seul
            stats = X.iloc[train].groupby(level=0)['inflation_ipc'].agg(['mean', 'std'])
            stats['std'] = X.iloc[train].groupby(level=0)['inflation_ipc'].std(ddof=0)
            X_test = X.iloc[test]
            entity = X_test.index.get_level_values(0)
            expected = (X_test['inflation_ipc'].to_numpy() - stats.loc[entity, 'mean'].to_numpy()) \
                / stats.loc[entity, 'std'].to_numpy()
            seen = estimator.named_steps['model'].predict_X_[0]
            np.testing.assert_allclose(np.asarray(seen, dtype=float).ravel(), expected)

    def test_per_entity_folds_fit_one_entity_each(self, panel_xy):
        """Each per-entity fold fits the transformer and the model on its entity alone."""
        X, y = panel_xy
        splitter = PanelOutOfSampleSplitPerEntity(test_indices=PANEL_DATES, test_size=3, gap=1)
        result = cross_validate(_scaling_pipeline(DummyRegressor()), X, y, cv=splitter,
                                scoring='neg_mean_absolute_error', return_estimator=True)
        # Entités ajustées par le transformateur de chaque pli : une seule, dans l'ordre des entités
        fitted = [list(estimator.named_steps['scale'].transformers_) for estimator in result['estimator']]
        assert fitted == [[('Allemagne',)], [('France',)], [('Italie',)]] * 2

    def test_explicit_groups_give_the_same_folds(self, panel_xy):
        """groups passed to cross_validate reach the splitter and change no fold."""
        X, y = panel_xy
        splitter = PanelOutOfSampleSplit(test_indices=PANEL_DATES, test_size=3, gap=1)
        groups = X.index.get_level_values(0).to_numpy()
        with_groups = cross_validate(DummyRegressor(), X, y, cv=splitter, groups=groups, return_indices=True)
        without = cross_validate(DummyRegressor(), X, y, cv=splitter, return_indices=True)
        assert [t.tolist() for t in with_groups['indices']['test']] == [t.tolist() for t in without['indices']['test']]

    def test_inconsistent_groups_are_refused(self, panel_xy):
        """groups passed to cross_validate are checked against the entity level."""
        X, y = panel_xy
        groups = np.roll(X.index.get_level_values(0).to_numpy(), 1)
        splitter = PanelOutOfSampleSplit(test_indices=PANEL_DATES)
        with pytest.raises(ValueError, match='groups must match the entity levels'):
            cross_validate(DummyRegressor(), X, y, cv=splitter, groups=groups)


class TestPanelSplittersInGridSearch:
    """Panel splitters as the cv of GridSearchCV over XYPipeline + PanelwiseTransformer."""

    @staticmethod
    def _search(splitter) -> GridSearchCV:
        """Grid over the constant of the model and a parameter of the per-entity transformer."""
        grid = {'model__constant': [0.0, 5.0], 'scale__transformer__with_std': [True, False]}
        return GridSearchCV(_scaling_pipeline(DummyRegressor(strategy='constant', constant=0.0)), grid, cv=splitter,
                            scoring='neg_mean_absolute_error')

    @pytest.mark.parametrize('splitter', PANEL_SPLITTERS[:2])
    def test_selects_the_constant_of_the_target(self, panel_xy, splitter):
        """With a target constant at 5, the search picks the constant 5 with a zero error."""
        X, _ = panel_xy
        y = pd.Series(5.0, index=X.index)
        search = self._search(splitter).fit(X, y)
        # Valeur d'or : erreur absolue nulle pour la constante 5, 5 pour la constante 0
        assert (search.best_params_['model__constant'], search.best_score_) == (5.0, 0.0)

    @pytest.mark.parametrize(
        ('splitter', 'n_folds'),
        [
            pytest.param(PanelOutOfSampleSplit(test_indices=PANEL_DATES, test_size=3, gap=1), 2, id='global-out'),
            pytest.param(PanelInSampleSplit(test_indices=PANEL_DATES), 1, id='global-in'),
        ],
    )
    def test_number_of_folds(self, panel_xy, splitter, n_folds):
        """The search runs one evaluation per fold of the splitter."""
        X, y = panel_xy
        assert self._search(splitter).fit(X, y).n_splits_ == n_folds

    def test_groups_passed_to_fit(self, panel_xy):
        """groups given to fit reach the splitter and leave the scores unchanged."""
        X, y = panel_xy
        splitter = PanelOutOfSampleSplit(test_indices=PANEL_DATES, test_size=3, gap=1)
        groups = X.index.get_level_values(0).to_numpy()
        with_groups = self._search(splitter).fit(X, y, groups=groups).cv_results_['mean_test_score']
        without = self._search(splitter).fit(X, y).cv_results_['mean_test_score']
        np.testing.assert_allclose(with_groups, without)

    @pytest.mark.parametrize(
        ('splitter', 'n_folds'),
        [
            pytest.param(PanelOutOfSampleSplitPerEntity(test_indices=PANEL_DATES, test_size=3, gap=1), 6, id='out'),
            pytest.param(PanelInSampleSplitPerEntity(test_indices=PANEL_DATES), 3, id='in'),
        ],
    )
    def test_per_entity_splitter(self, panel_xy, splitter, n_folds):
        """A per-entity splitter runs one evaluation per entity and per fold."""
        X, y = panel_xy
        assert self._search(splitter).fit(X, y).n_splits_ == n_folds


class TestRealisticMonthlySeries:
    """Time-series splitters on the irregular monthly series (annual anchors before the monthly grid)."""

    def test_folds_have_no_leakage_and_tile_the_end(self, irregular_index_timeseries):
        """Four adjacent test sets of six rows end at the last row, one row after their training."""
        X = irregular_index_timeseries
        folds = collect_folds(TSOutOfSampleSplit(n_splits=4, test_size=6, gap=1), X)
        assert_no_leakage(X, folds, gap=1)
        assert_folds_cover(folds, test_size=6, n_samples=len(X))

    def test_rows_are_dates_not_calendar_months(self, irregular_index_timeseries):
        """Training before January 2018 is the three annual anchors 2015-2017."""
        X = irregular_index_timeseries
        folds = collect_folds(TSOutOfSampleSplit(test_indices=['2018-01-01'], test_size=3), X)
        assert as_lists(folds) == [([0, 1, 2], [3, 4, 5])]

    def test_in_sample_contract(self, irregular_index_timeseries):
        """In sample, the test dates are training dates and training stops at the last one."""
        X = irregular_index_timeseries
        assert_in_sample(X, collect_folds(TSInSampleSplit(test_indices=['2020-01-01', '2022-01-01']), X))

    def test_cross_validate_with_drop_nan_target(self, irregular_index_timeseries):
        """The model trains on the training rows whose target is observed; every score is finite."""
        X = irregular_index_timeseries.drop(columns='taux_chomage')
        y = irregular_index_timeseries['taux_chomage']
        pipe = XYPipeline([
            ('drop', DropNaNTarget()),
            ('impute', SimpleImputer(keep_empty_features=True)),
            ('model', RecordingEstimator(LinearRegression())),
        ])
        splitter = TSOutOfSampleSplit(n_splits=4, test_size=6, gap=1)
        result = cross_validate(pipe, X, y, cv=splitter, return_estimator=True, return_indices=True)
        # Nombre de cibles observées dans chaque apprentissage : seules lignes vues par le modèle
        observed = [int(y.iloc[train].notna().sum()) for train in result['indices']['train']]
        seen = [len(estimator.named_steps['model'].fit_y_) for estimator in result['estimator']]
        assert (seen, bool(np.isfinite(result['test_score']).all())) == (observed, True)


class TestForecastingWorkflows:
    """Workflows whose target is an exact linear function of a covariate: the error is zero."""

    def test_time_series_rolling_window(self, simple_timeseries_df):
        """target = feature1 + 30 is recovered exactly on every fold."""
        X, y = simple_timeseries_df[['feature1', 'feature2']], simple_timeseries_df['target']
        splitter = TSOutOfSampleSplit(n_splits=5, test_size=3, gap=1, max_train_size=15)
        result = cross_validate(LinearRegression(), X, y, cv=splitter, scoring='neg_mean_absolute_error')
        np.testing.assert_allclose(result['test_score'], np.zeros(5), atol=1e-8)

    def test_panel_global_folds(self, simple_panel):
        """target = feature1 + 60 in every entity is recovered exactly on every global fold."""
        X, y = simple_panel[['feature1', 'feature2']], simple_panel['target']
        result = cross_validate(LinearRegression(), X, y, cv=PanelOutOfSampleSplit(n_splits=3, test_size=2, gap=1),
                                scoring='neg_mean_absolute_error')
        np.testing.assert_allclose(result['test_score'], np.zeros(3), atol=1e-8)

    def test_panel_per_entity_folds(self, simple_panel):
        """Per-entity folds: two per entity, each fitting and testing one entity, zero error."""
        X, y = simple_panel[['feature1', 'feature2']], simple_panel['target']
        splitter = PanelOutOfSampleSplitPerEntity(n_splits=2, test_size=3, gap=1)
        result = cross_validate(LinearRegression(), X, y, cv=splitter, scoring='neg_mean_absolute_error',
                                return_indices=True)
        tested = [X.index[test].get_level_values(0).unique().tolist() for test in result['indices']['test']]
        assert tested == [['A'], ['B'], ['C']] * 2
        np.testing.assert_allclose(result['test_score'], np.zeros(6), atol=1e-8)

    def test_historical_backtesting(self, simple_timeseries_df):
        """Several in-sample dates: one fold testing those rows, training up to the last one."""
        splitter = TSInSampleSplit(test_indices=['2020-01-15', '2020-01-25'], test_size=3)
        # Valeur d'or : 15 et 25 janvier = positions 14 et 24 ; test_size non utilisé (plusieurs dates)
        assert as_lists(collect_folds(splitter, simple_timeseries_df)) == [(list(range(25)), [14, 24])]


class TestTimeSeriesEdgeCases:
    """Time-series folds on several frequencies and very small series."""

    @pytest.mark.parametrize(('freq', 'periods'), [('D', 50), ('h', 50), ('ME', 24), ('YE', 10)])
    def test_frequency_does_not_change_the_positions(self, freq, periods):
        """The folds depend on the number of rows only, not on the spacing of the dates."""
        X = pd.DataFrame({'feature': range(periods)}, index=pd.date_range('2020-01-01', periods=periods, freq=freq))
        folds = collect_folds(TSOutOfSampleSplit(n_splits=2, test_size=3), X)
        n = periods
        assert as_lists(folds) == [(list(range(n - 6)), [n - 6, n - 5, n - 4]), (list(range(n - 3)), [n - 3, n - 2, n - 1])]

    def test_very_small_series(self):
        """Three rows give two folds of one test row."""
        folds = collect_folds(TSOutOfSampleSplit(n_splits=2, test_size=1), pd.Series([1, 2, 3]))
        assert as_lists(folds) == [([0], [1]), ([0, 1], [2])]

    def test_gap_edge_cases(self):
        """A gap may leave one training row, but not zero."""
        X = pd.Series(range(10))

        # Écart qui ne laisse qu'une ligne d'apprentissage au premier pli (test à partir de la ligne 6)
        splits = list(TSOutOfSampleSplit(n_splits=2, test_size=2, gap=5).split(X))
        assert [train_idx.tolist() for train_idx, _ in splits] == [[0], [0, 1, 2]]

        # Écart qui ne laisse aucune ligne d'apprentissage : configuration refusée
        with pytest.raises(ValueError, match='No training data before the test set starting at row 6'):
            list(TSOutOfSampleSplit(n_splits=2, test_size=2, gap=7).split(X))

    def test_in_sample_window_shorter_than_the_test(self):
        """A window shorter than the test is widened to the whole test period."""
        # Valeur d'or : 10 dernières lignes testées, fenêtre max(5, 10) = 10 lignes
        folds = collect_folds(InSampleSplit(test_size=10, max_train_size=5), pd.Series(range(20)))
        assert as_lists(folds) == [(list(range(10, 20)), list(range(10, 20)))]


class TestLargeData:
    """Splitters on a thousand dates and on a panel of ten entities over two hundred dates."""

    def test_large_time_series(self, large_timeseries):
        """Ten leak-free folds of fifty rows, five rows apart from their training."""
        folds = collect_folds(TSOutOfSampleSplit(n_splits=10, test_size=50, gap=5), large_timeseries)
        assert_no_leakage(large_timeseries, folds, gap=5)
        assert_folds_cover(folds, test_size=50, n_samples=len(large_timeseries))

    def test_large_panel(self, large_panel):
        """Five leak-free folds testing ten dates of each of the ten entities."""
        folds = collect_folds(PanelOutOfSampleSplit(n_splits=5, test_size=10, gap=2), large_panel)
        assert_no_leakage(large_panel, folds, gap=2)
        assert [len(test) for _, test in folds] == [100] * 5

    def test_large_panel_per_entity(self, large_panel):
        """Three folds per entity: thirty single-entity folds."""
        folds = collect_folds(PanelOutOfSampleSplitPerEntity(n_splits=3, test_size=5), large_panel)
        entities = [large_panel.index[test].get_level_values(0).unique().tolist() for _, test in folds]
        assert entities == [[f'entity_{i:02d}'] for i in range(10)] * 3
