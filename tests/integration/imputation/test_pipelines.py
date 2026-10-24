"""``HighFrequencyImputer`` inside an ``XYPipeline``, cross-validated and grid-searched.

Chains ``tsforecast.delays.transformers.PublicationDelayTransformer``
(``strategy='mask'``), ``tsforecast.frequency.HighFrequencyImputer`` and a
final estimator (the NaN-tolerant ``SpyEstimator`` of
``tests/support/estimators.py``, which records what it is fitted on) in a
``tsforecast.xy.XYPipeline``, evaluated by ``sklearn.model_selection.cross_validate``
with ``tsforecast.crossvals.TSOutOfSampleSplit`` (series) and
``tsforecast.crossvals.PanelOutOfSampleSplit`` (panel), and tuned by
``sklearn.model_selection.GridSearchCV`` on a parameter of the imputer.

Data: the realistic datasets of notebook 3, target ``taux_chomage``, restricted
to the rows where the target is observed - the monthly grid of each entity,
without the isolated annual dates before it nor the last row (publication
delay). On the complete irregular index, the training pass adds the months of
the grid (82 -> 115 rows) whose target is NaN (:class:`TestXYContract`), and
``tsforecast.xy.DropNaNTarget`` drops them right before an estimator that
refuses a NaN target (:class:`TestFullChainWithDropNaNTarget`).

Contracts checked:

- the chain runs end to end and every fold is scored, by ``score`` as by a
  string scorer: the pipeline predicts from ``X`` alone;
- fitted with ``y``, the imputer has two passes: ``fit_transform`` augments
  the training set (``X`` and ``y`` on the grid), ``transform`` is the
  prediction pass (``X`` alone, on its input rows, ``y`` unchanged);
- ``y`` is not a covariate of the imputation of ``X`` by default, so the test
  target never feeds the imputed test features;
- no information of the test fold reaches the training (perturbing the test
  fold changes nothing the estimator is fitted on); the rows dated after the
  end of a training fold are only the completion of the period of a training
  observation by the imputer grid, with a NaN target;
- ``GridSearchCV`` explores a parameter of the imputer.

Also covered: ``predict`` without ``y`` (so string scorers work), ``y`` not
being a covariate of the imputation of ``X``, and a test fold cutting a period
being scored.
"""
# Modules de base
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils.validation import check_is_fitted

# Doublure d'estimateur partagée
from tests.support.estimators import SpyEstimator

# Composants enchaînés
from tsforecast.crossvals import PanelOutOfSampleSplit, TSOutOfSampleSplit
from tsforecast.delays.transformers import PublicationDelayTransformer
from tsforecast.frequency import HighFrequencyImputer
from tsforecast.xy import DropNaNTarget, XYPipeline

# Cible commune aux deux jeux : mensuelle, observée sur toute la grille sauf la dernière ligne
TARGET = 'taux_chomage'

# Découpage des plis : 3 plis de 6 mois
N_SPLITS = 3
TEST_SIZE = 6


# =============================================================================
# Construction des données et du pipeline
# =============================================================================
def _observed_target_rows(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a realistic dataset into ``(X, y)`` on the rows where the target is observed.

    Args:
        data: Realistic dataset of notebook 3 (series or panel).

    Returns:
        ``(X, y)``: every column but :data:`TARGET`, and :data:`TARGET`, on
        the rows where :data:`TARGET` is not NaN.
    """
    # Lignes de cible observée : grille mensuelle de chaque entité, sans
    # les ancres annuelles isolées d'avant la grille ni la dernière ligne
    kept = data[data[TARGET].notna()]
    return kept.drop(columns=TARGET), kept[TARGET]


def _make_pipeline(**imputer_params) -> XYPipeline:
    """Build the ``delays -> imputer -> model`` pipeline of this module.

    Args:
        **imputer_params: Overrides of the ``HighFrequencyImputer`` parameters.

    Returns:
        Unfitted ``XYPipeline``.
    """
    # Délai de 20 jours sur l'inflation, masque trimestriel : le décalage
    # désaligne X et y
    delays = PublicationDelayTransformer(
        delays={'inflation_ipc': 20.0},
        prediction_date='2024-08-15',
        strategy='mask',
        target_frequency='Q',
        delay_unit='D',
        reference_point='start',
        handle_missing_delays='ignore',
    )
    # Sortie plate : le niveau 'frequency' n'a pas sa place devant un estimateur
    params = dict(target_frequency='M', estimator=LinearRegression(), keep_lower_frequencies=False)
    params.update(imputer_params)
    return XYPipeline([
        ('delays', delays),
        ('imputer', HighFrequencyImputer(**params)),
        ('model', SpyEstimator()),
    ])


def _quietly(function, *args, **kwargs):
    """Call ``function`` with every warning silenced and return its result."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


def _cross_validate(X, y, cv, pipeline=None, **kwargs) -> dict:
    """Run ``cross_validate`` on the module pipeline, estimators and indices returned."""
    return _quietly(
        cross_validate,
        pipeline if pipeline is not None else _make_pipeline(),
        X, y, cv=cv,
        return_estimator=True, return_indices=True, error_score='raise',
        **kwargs,
    )


def _dates(index: pd.Index) -> pd.DatetimeIndex:
    """Return the date level of a series or panel index."""
    return index.get_level_values(-1) if isinstance(index, pd.MultiIndex) else index


def _common_test_starts(X: pd.DataFrame) -> List[pd.Timestamp]:
    """Return the start dates of the panel test folds, shared by every entity.

    ``PanelOutOfSampleSplit`` needs each entity to hold each test date:
    Allemagne stops in 2024-04, three months before the others, so the folds
    are placed on the last ``N_SPLITS * TEST_SIZE`` dates common to all.

    Args:
        X: Panel features.

    Returns:
        ``N_SPLITS`` sorted start dates, ``TEST_SIZE`` months apart.
    """
    entities = X.index.get_level_values(0).unique()
    common = sorted(set.intersection(*[set(X.loc[entity].index) for entity in entities]))
    return common[-N_SPLITS * TEST_SIZE::TEST_SIZE]


# =============================================================================
# Fixtures (une validation croisée par jeu et par module)
# =============================================================================
@pytest.fixture(scope='module')
def series_xy(_irregular_index_timeseries_session) -> Tuple[pd.DataFrame, pd.Series]:
    """``(X, y)`` of the realistic series: 78 monthly rows, 2018-01 -> 2024-06."""
    return _observed_target_rows(_irregular_index_timeseries_session.copy())


@pytest.fixture(scope='module')
def panel_xy(_heterogeneous_coverage_panel_session) -> Tuple[pd.DataFrame, pd.Series]:
    """``(X, y)`` of the realistic panel: 213 rows, each entity on its own monthly grid."""
    return _observed_target_rows(_heterogeneous_coverage_panel_session.copy())


@pytest.fixture(scope='module')
def series_cv(series_xy) -> dict:
    """``cross_validate`` of the pipeline on the series, 3 folds of 6 months."""
    X, y = series_xy
    return _cross_validate(X, y, TSOutOfSampleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE))


@pytest.fixture(scope='module')
def panel_cv(panel_xy) -> dict:
    """``cross_validate`` of the pipeline on the panel, 3 folds on common dates."""
    X, y = panel_xy
    splitter = PanelOutOfSampleSplit(
        n_splits=N_SPLITS, test_size=TEST_SIZE, test_indices=_common_test_starts(X)
    )
    return _cross_validate(X, y, splitter)


# =============================================================================
# Validation croisée : la chaîne tourne de bout en bout
# =============================================================================
class TestCrossValidateTimeSeries:
    """``cross_validate`` + ``TSOutOfSampleSplit`` on the realistic irregular series."""

    def test_every_fold_is_scored(self, series_cv):
        """The three folds yield a finite score (``scoring=None``: ``y`` crosses the steps)."""
        scores = series_cv['test_score']

        assert len(scores) == N_SPLITS and np.isfinite(scores).all()

    def test_every_step_of_every_fold_is_fitted(self, series_cv):
        """Each fold's pipeline holds a fitted delay step, imputer and estimator."""
        unfitted = []
        for fold, pipeline in enumerate(series_cv['estimator']):
            for name, step in pipeline.steps:
                try:
                    check_is_fitted(step)
                except Exception:
                    unfitted.append((fold, name))

        assert unfitted == []

    def test_every_fold_imputes_the_low_frequency_columns(self, series_cv):
        """The imputer of each fold plans the quarterly and annual columns."""
        planned = [
            {step.var_name for step in pipeline.named_steps['imputer'].imputation_plan_}
            for pipeline in series_cv['estimator']
        ]

        assert all(
            {'pib_trimestriel', 'balance_commerciale_annuelle'} <= variables
            for variables in planned
        )

    def test_spy_sees_every_training_row(self, series_cv, series_xy):
        """Every row of the training fold reaches the final estimator."""
        X, _y = series_xy
        missing = [
            int(X.index[train].difference(pipeline[-1].fit_X_.index).size)
            for pipeline, train in zip(series_cv['estimator'], series_cv['indices']['train'])
        ]

        assert missing == [0] * N_SPLITS


@pytest.mark.slow
class TestCrossValidatePanel:
    """``cross_validate`` + ``PanelOutOfSampleSplit`` on the heterogeneous panel."""

    def test_every_fold_is_scored(self, panel_cv):
        """The three folds yield a finite score."""
        scores = panel_cv['test_score']

        assert len(scores) == N_SPLITS and np.isfinite(scores).all()

    def test_every_fold_imputes_the_three_entities(self, panel_cv):
        """The imputer of each fold is fitted on the three entities."""
        entities = [
            pipeline.named_steps['imputer'].entities_ for pipeline in panel_cv['estimator']
        ]

        assert entities == [[('Allemagne',), ('France',), ('Italie',)]] * N_SPLITS

    def test_spy_sees_every_training_row(self, panel_cv, panel_xy):
        """Every row of the training fold reaches the final estimator."""
        X, _y = panel_xy
        missing = [
            int(X.index[train].difference(pipeline[-1].fit_X_.index).size)
            for pipeline, train in zip(panel_cv['estimator'], panel_cv['indices']['train'])
        ]

        assert missing == [0] * N_SPLITS


# =============================================================================
# Absence de fuite : critère informationnel
# =============================================================================
def _perturbed_from(X: pd.DataFrame, y: pd.Series, start: pd.Timestamp):
    """Multiply by 10 every value of ``X`` and ``y`` dated ``start`` or later.

    Args:
        X: Features.
        y: Target.
        start: First perturbed date (start of a test fold).

    Returns:
        Perturbed copies ``(X, y)``; the earlier rows are untouched.
    """
    after = (_dates(X.index) >= start)
    X_perturbed, y_perturbed = X.copy(), y.copy()
    X_perturbed.loc[after] = X_perturbed.loc[after] * 10
    y_perturbed.loc[after] = y_perturbed.loc[after] * 10
    return X_perturbed, y_perturbed


def _training_end_completion(X_train: pd.DataFrame) -> pd.Timestamp:
    """Return the last date the imputer grid reaches on a training fold.

    The grid of the imputer runs to the end of the latest observed period:
    the training end itself, or the end of the year (quarter) of the last
    annual (quarterly) observation when that period overruns it.

    Args:
        X_train: Features of the training fold (series).

    Returns:
        Month start of the last month of the grid.
    """
    ends = [X_train.index.max()]
    for column, frequency in (('pib_trimestriel', 'Q'), ('balance_commerciale_annuelle', 'Y')):
        last = X_train[column].last_valid_index()
        if last is not None:
            ends.append(pd.Period(last, freq=frequency).end_time.to_period('M').to_timestamp())
    return max(ends)


class TestNoLeakage:
    """No information of the test fold reaches what the estimator is fitted on."""

    @pytest.mark.parametrize('fold', range(N_SPLITS), ids=lambda fold: f'fold{fold}')
    def test_training_ignores_the_test_fold(self, series_cv, series_xy, fold):
        """Perturbing every value from the test fold on leaves ``fit_X_`` / ``fit_y_`` unchanged.

        The cross-validation is replayed on data multiplied by 10 from the
        first test date of the fold: the estimator of that fold must be fitted
        on exactly the same frame, bit for bit.
        """
        X, y = series_xy
        test_start = X.index[series_cv['indices']['test'][fold]].min()
        X_perturbed, y_perturbed = _perturbed_from(X, y, test_start)

        replay = _cross_validate(
            X_perturbed, y_perturbed, TSOutOfSampleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
        )
        reference, perturbed = series_cv['estimator'][fold][-1], replay['estimator'][fold][-1]

        pd.testing.assert_frame_equal(perturbed.fit_X_, reference.fit_X_)
        pd.testing.assert_series_equal(perturbed.fit_y_, reference.fit_y_)

    @pytest.mark.parametrize('fold', range(N_SPLITS), ids=lambda fold: f'fold{fold}')
    def test_rows_after_the_training_end_only_complete_a_period(self, series_cv, series_xy, fold):
        """Rows dated after the training end are the imputer's completion of an observed period.

        Gold value derived from the input: the grid runs to the end of the
        year (quarter) of the last annual (quarterly) training observation.
        The fold ending in 2023-06 therefore reaches 2023-12 (annual value of
        2023-01-01): six months of the test period appear as training rows,
        built from training data only.
        """
        X, _y = series_xy
        train = series_cv['indices']['train'][fold]
        fit_X = series_cv['estimator'][fold][-1].fit_X_

        # Valeur d'or : mois suivant la fin du pli jusqu'à la fin de la période complétée
        train_end = X.index[train].max()
        expected = pd.date_range(
            train_end + pd.offsets.MonthBegin(1), _training_end_completion(X.iloc[train]), freq='MS'
        )
        extra = fit_X.index[fit_X.index > train_end]

        assert list(extra) == list(expected)

    def test_some_fold_is_completed_past_its_end(self, series_cv, series_xy):
        """The completion case is exercised: the fold ending mid-year gains six rows."""
        X, _y = series_xy
        extra_rows = [
            int((pipeline[-1].fit_X_.index > X.index[train].max()).sum())
            for pipeline, train in zip(series_cv['estimator'], series_cv['indices']['train'])
        ]

        # Valeur d'or : plis arrêtés en 2022-12, 2023-06, 2023-12 -> 0, 6, 0 lignes ajoutées
        assert extra_rows == [0, 6, 0]

    def test_completed_rows_carry_no_target(self, series_cv, series_xy):
        """The completion rows carry a NaN target: no test value is attached to them."""
        X, _y = series_xy
        targets = pd.concat([
            pipeline[-1].fit_y_[pipeline[-1].fit_y_.index > X.index[train].max()]
            for pipeline, train in zip(series_cv['estimator'], series_cv['indices']['train'])
        ])

        assert len(targets) > 0 and targets.isna().all()

    def test_test_fold_cutting_a_period_is_scored(self, series_xy):
        """A 7-month test fold, which cuts a quarter, is scored.

        The prediction pass imputes ``X`` on its input rows only: no
        completion row, hence no NaN target reaches the scorer.
        """
        X, y = series_xy

        results = _cross_validate(X, y, TSOutOfSampleSplit(n_splits=N_SPLITS, test_size=7))

        assert np.isfinite(results['test_score']).all()

    @pytest.mark.slow
    def test_panel_training_ignores_the_test_fold(self, panel_cv, panel_xy):
        """Panel: perturbing the last test fold leaves the last estimator unchanged."""
        X, y = panel_xy
        starts = _common_test_starts(X)
        X_perturbed, y_perturbed = _perturbed_from(X, y, starts[-1])

        splitter = PanelOutOfSampleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, test_indices=starts)
        replay = _cross_validate(X_perturbed, y_perturbed, splitter)

        pd.testing.assert_frame_equal(
            replay['estimator'][-1][-1].fit_X_, panel_cv['estimator'][-1][-1].fit_X_
        )


# =============================================================================
# Recherche sur grille d'un paramètre de l'imputeur
# =============================================================================
@pytest.fixture(scope='module')
def grid_search(series_xy) -> GridSearchCV:
    """``GridSearchCV`` on ``imputer__covariate_strategy``, fitted on the series."""
    X, y = series_xy
    search = GridSearchCV(
        _make_pipeline(),
        {'imputer__covariate_strategy': ['interpolate', 'model']},
        cv=TSOutOfSampleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE),
        error_score='raise',
    )
    return _quietly(search.fit, X, y)


@pytest.mark.slow
class TestGridSearch:
    """``GridSearchCV`` explores a parameter of the imputer through the pipeline."""

    def test_every_candidate_is_scored(self, grid_search):
        """Both candidates get a finite mean score."""
        scores = grid_search.cv_results_['mean_test_score']

        assert len(scores) == 2 and np.isfinite(scores).all()

    def test_best_parameter_is_a_candidate(self, grid_search):
        """The retained strategy is one of the grid."""
        assert grid_search.best_params_['imputer__covariate_strategy'] in {'interpolate', 'model'}

    def test_refit_pipeline_carries_the_best_parameter(self, grid_search):
        """The refitted pipeline is fitted with the retained strategy."""
        best = grid_search.best_estimator_

        check_is_fitted(best.named_steps['imputer'])
        assert (
            best.named_steps['imputer'].covariate_strategy
            == grid_search.best_params_['imputer__covariate_strategy']
        )


# =============================================================================
# Contrat XY : X et y restent alignés quand l'imputeur ajoute des lignes
# =============================================================================
class TestXYContract:
    """On the complete irregular series, the imputer grows the rows of X and y together."""

    @pytest.fixture(scope='class')
    def transformed(self, _irregular_index_timeseries_session):
        """``(X, y, Xt, yt)``: delay step + imputer on the 82 rows of the irregular series."""
        data = _irregular_index_timeseries_session.copy()
        X, y = data.drop(columns=TARGET), data[TARGET]
        steps = XYPipeline(_make_pipeline().steps[:-1])
        Xt, yt = _quietly(steps.fit_transform, X, y)
        return X, y, Xt, yt

    def test_x_and_y_share_their_index(self, transformed):
        """``Xt`` and ``yt`` come out on one and the same index."""
        _X, _y, Xt, yt = transformed

        assert Xt.index.equals(yt.index)

    def test_imputer_adds_the_months_of_the_grid(self, transformed):
        """82 rows in, 115 out: the 33 months of 2015-2017 the annual dates leave out.

        Valeur d'or : grille mensuelle 2015-01 -> 2024-07 = 115 mois.
        """
        X, _y, Xt, _yt = transformed

        assert (len(X), len(Xt)) == (82, len(pd.date_range('2015-01-01', '2024-07-01', freq='MS')))

    def test_added_rows_carry_no_target(self, transformed):
        """The target of the added months is NaN: they hold no observation."""
        X, _y, _Xt, yt = transformed

        assert yt[yt.index.difference(X.index)].isna().all()


# =============================================================================
# Prédire sans y
# =============================================================================
@pytest.fixture(scope='module')
def series_split(series_xy):
    """``(X_train, y_train, X_test, y_test)``: the series cut before its last 6 months."""
    X, y = series_xy
    return X.iloc[:-TEST_SIZE], y.iloc[:-TEST_SIZE], X.iloc[-TEST_SIZE:], y.iloc[-TEST_SIZE:]


class TestPredictWithoutTarget:
    """A fitted pipeline predicts from ``X`` alone, as any sklearn pipeline."""

    def test_fitted_pipeline_predicts_from_x_alone(self, series_split):
        """``pipeline.predict(X_test)`` returns one prediction per test row."""
        X_train, y_train, X_test, _y_test = series_split
        pipeline = _quietly(_make_pipeline().fit, X_train, y_train)

        predictions = _quietly(pipeline.predict, X_test)

        assert len(predictions) == len(X_test)

    def test_cross_validate_accepts_a_string_scorer(self, series_xy):
        """``cross_validate(..., scoring='neg_mean_absolute_error')`` scores every fold."""
        X, y = series_xy

        results = _cross_validate(
            X, y, TSOutOfSampleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE),
            scoring='neg_mean_absolute_error',
        )

        assert np.isfinite(results['test_score']).all()

    def test_estimator_predicts_on_the_input_rows(self, series_split):
        """The estimator receives the test rows as given: no row added, none lost."""
        X_train, y_train, X_test, _y_test = series_split
        pipeline = _quietly(_make_pipeline().fit, X_train, y_train)

        _quietly(pipeline.predict, X_test)

        assert pipeline[-1].predict_X_[-1].index.equals(X_test.index)


# =============================================================================
# Rôle de y : imputé, jamais covariable par défaut
# =============================================================================
class TestTargetRole:
    """``y`` is imputed at the training pass, never read by the prediction pass."""

    @pytest.fixture(scope='class')
    def imputer(self, series_split) -> HighFrequencyImputer:
        """Imputer of the pipeline, fitted alone with ``y`` on the training rows."""
        X_train, y_train, _X_test, _y_test = series_split
        return _quietly(clone(_make_pipeline().named_steps['imputer']).fit, X_train, y_train)

    def test_target_is_not_a_feature_by_default(self, imputer):
        """No imputation step lists the target among its features (``target_as_covariate=False``)."""
        features = [tuple(step.feature_cols) for step in imputer.imputation_plan_]

        assert features and not any(TARGET in columns for columns in features)

    def test_imputed_test_features_ignore_the_test_target(self, imputer, series_split):
        """Scaling ``y_test`` by 10 leaves the imputed ``X_test`` unchanged: no leak."""
        _X_train, _y_train, X_test, y_test = series_split

        X_reference, _ = _quietly(imputer.transform, X_test, y_test)
        X_scaled, _ = _quietly(imputer.transform, X_test, y_test * 10)

        pd.testing.assert_frame_equal(X_reference, X_scaled)

    def test_prediction_pass_returns_the_target_unchanged(self, imputer, series_split):
        """``transform(X, y)`` hands ``y`` back as given: a test target is never imputed."""
        _X_train, _y_train, X_test, y_test = series_split

        _X_out, y_out = _quietly(imputer.transform, X_test, y_test)

        pd.testing.assert_series_equal(y_out, y_test)

    def test_target_as_covariate_lets_the_target_in(self, series_split):
        """``target_as_covariate=True`` restores the former behaviour at fit time."""
        X_train, y_train, _X_test, _y_test = series_split
        imputer = _quietly(
            HighFrequencyImputer(
                target_frequency='M', estimator=LinearRegression(),
                keep_lower_frequencies=False, target_as_covariate=True,
            ).fit,
            X_train, y_train,
        )

        assert any(TARGET in step.feature_cols for step in imputer.imputation_plan_)

    def test_prediction_pass_refuses_a_model_that_needs_the_target(self, series_split):
        """Under ``target_as_covariate=True``, ``transform`` raises instead of feeding a NaN target."""
        X_train, y_train, X_test, _y_test = series_split
        imputer = _quietly(
            HighFrequencyImputer(
                target_frequency='M', estimator=LinearRegression(),
                keep_lower_frequencies=False, target_as_covariate=True,
            ).fit,
            X_train, y_train,
        )

        with pytest.raises(ValueError, match='target_as_covariate=False'):
            imputer.transform(X_test)


# =============================================================================
# Chaîne complète : estimateur refusant une cible NaN, DropNaNTarget
# =============================================================================
def _full_chain() -> XYPipeline:
    """``delays -> imputer -> DropNaNTarget -> HistGradientBoostingRegressor``.

    The gradient boosting tolerates NaN features (the monthly columns before
    their first observation) but refuses a NaN target: the rows the training
    pass adds without target must go, right before it.
    """
    steps = _make_pipeline().steps[:-1]
    return XYPipeline([
        *steps,
        ('drop', DropNaNTarget()),
        ('model', HistGradientBoostingRegressor(max_iter=20, min_samples_leaf=2)),
    ])


class TestFullChainWithDropNaNTarget:
    """The complete irregular series, its NaN targets included, through a real estimator."""

    @pytest.fixture(scope='class')
    def full_xy(self, _irregular_index_timeseries_session):
        """``(X, y)`` on the 82 rows of the irregular series, target NaN on 4 of them."""
        data = _irregular_index_timeseries_session.copy()
        return data.drop(columns=TARGET), data[TARGET]

    def test_without_the_drop_step_the_estimator_refuses_the_target(self, full_xy):
        """Control: the same chain without ``DropNaNTarget`` fails on the NaN target."""
        X, y = full_xy
        pipeline = XYPipeline([
            *_make_pipeline().steps[:-1],
            ('model', HistGradientBoostingRegressor(max_iter=20, min_samples_leaf=2)),
        ])

        with pytest.raises(ValueError, match='NaN'):
            _quietly(pipeline.fit, X, y)

    @pytest.mark.slow
    def test_cross_validate_scores_every_fold(self, full_xy):
        """With ``DropNaNTarget``, every fold is scored, folds cutting a period included.

        ``scoring=None``: ``XYPipeline.score`` sends ``y_test`` through the
        steps, and ``DropNaNTarget`` drops the test row whose target is NaN.
        A string scorer compares the predictions with the raw ``y_test``,
        outside the pipeline: no step can spare it a NaN test target.
        """
        X, y = full_xy

        results = _cross_validate(
            X, y, TSOutOfSampleSplit(n_splits=N_SPLITS, test_size=7), pipeline=_full_chain(),
        )

        assert np.isfinite(results['test_score']).all()

    def test_prediction_keeps_every_test_row(self, full_xy):
        """``predict`` loses no row, the one whose target is NaN included."""
        X, y = full_xy
        pipeline = _quietly(_full_chain().fit, X.iloc[:-6], y.iloc[:-6])

        predictions = _quietly(pipeline.predict, X.iloc[-6:])

        assert len(predictions) == 6
