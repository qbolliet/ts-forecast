"""Complete panel forecasting workflow: per-entity steps, target filtering, estimator, cross-validation.

Chains ``tsforecast.panel.PanelwiseTransformer`` (around
``tsforecast.delays.transformers.PublicationDelayTransformer``, 'mask'
strategy, and around ``EntityDemeaner``, a minimal subclass of
``tsforecast.base.PanelTimeSeriesTransformer``), ``tsforecast.xy.DropNaNTarget``
and a real estimator (``SimpleImputer`` + ``Ridge``, wrapped in the
``RecordingEstimator`` of ``tests/support/estimators.py``) inside a
``tsforecast.xy.XYPipeline``, evaluated by ``sklearn.model_selection.cross_validate``
with ``PanelOutOfSampleSplit`` / ``PanelOutOfSampleSplitPerEntity`` on the
realistic heterogeneous-coverage panel, and with ``TSOutOfSampleSplit`` on the
realistic irregular series.

Families of scenarios: finite scores and folds at the expected dates; no
leakage between folds (training rows only, test period without influence on
the fit); no leakage between entities (an entity's result does not depend on
the other entities); inputs left unchanged; reproducibility; ``n_jobs=1``
against ``n_jobs=2`` (``slow``); a one-entity panel equivalent to the series.
"""
# Modules de base
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

# Doublure d'estimateur et propriétés des plis partagées
from tests.support.estimators import RecordingEstimator
from tests.support.splits import assert_no_leakage

# Composants enchaînés
from tsforecast.base import PanelTimeSeriesTransformer
from tsforecast.crossvals import PanelOutOfSampleSplit, PanelOutOfSampleSplitPerEntity, TSOutOfSampleSplit
from tsforecast.delays.transformers import PublicationDelayTransformer
from tsforecast.panel import PanelwiseTransformer
from tsforecast.xy import DropNaNTarget, XYPipeline

TS = pd.Timestamp

TARGET = 'production_industrielle'

# Date de prédiction commune à tous les plis (paramètre fixe du transformateur de délais)
PREDICTION_DATE = '2024-08-15'

# Délais en jours depuis le début du mois (inflation : 20 jours, chômage : 50 jours)
DELAYS = {'inflation_ipc': 20.0, 'taux_chomage': 50.0}

# Entités du panel réaliste, dans l'ordre de l'index (tri alphabétique du constructeur)
ENTITIES = ['Allemagne', 'France', 'Italie']

# Trois plis de test de six dates : 2023-02 -> 2023-07, 2023-08 -> 2024-01, 2024-02 -> 2024-07
N_SPLITS, TEST_SIZE = 3, 6


# =============================================================================
# Étape dérivée de la classe de base des transformateurs temporels
# =============================================================================
class EntityDemeaner(PanelTimeSeriesTransformer):
    """Subtract from each column its mean over the rows seen at ``fit``.

    Inside a ``PanelwiseTransformer``, each entity is centred on its own
    training mean: the learnt state ``means_`` is the place where a leakage
    from the test period or from another entity would show.

    Attributes:
        means_: Mean of each column over the rows seen at ``fit``
            (``NaN`` ignored; ``NaN`` for a column without observation).
    """

    def _fit(self, X, y=None) -> None:
        # Moyenne par colonne des lignes d'apprentissage (données validées par la classe de base)
        self.means_ = X.mean()

    def _transform(self, X):
        # Centrage sur la moyenne apprise ; une colonne sans observation reste NaN
        return X - self.means_


# =============================================================================
# Construction des pipelines et exécution de la validation croisée
# =============================================================================
def _delays() -> PublicationDelayTransformer:
    """Mask the unpublished months of inflation and unemployment at the end of each quarter."""
    return PublicationDelayTransformer(delays=DELAYS, prediction_date=PREDICTION_DATE, strategy='mask',
                                       target_frequency='Q', delay_unit='D', reference_point='start',
                                       handle_missing_delays='ignore')


def _model() -> RecordingEstimator:
    """NaN-tolerant linear model recording the rows it is fitted on."""
    # keep_empty_features : une colonne sans observation dans un pli (climat_affaires de l'Italie) garde sa place
    return RecordingEstimator(make_pipeline(SimpleImputer(keep_empty_features=True), Ridge()))


def _panel_pipeline(n_jobs: int = 1) -> XYPipeline:
    """Per-entity delays and centring, target filtering, then the recording model."""
    return XYPipeline([
        ('delays', PanelwiseTransformer(_delays(), n_jobs=n_jobs)),
        ('center', PanelwiseTransformer(EntityDemeaner(), n_jobs=n_jobs)),
        ('dropna', DropNaNTarget()),
        ('model', _model()),
    ])


def _series_pipeline() -> XYPipeline:
    """Same steps as :func:`_panel_pipeline`, applied directly to a single series."""
    return XYPipeline([('delays', _delays()), ('center', EntityDemeaner()), ('dropna', DropNaNTarget()),
                       ('model', _model())])


def _split(data: pd.DataFrame):
    """Return ``(X, y)``: the six other columns and the target."""
    return data.drop(columns=TARGET), data[TARGET]


def _run(X, y, cv, pipeline=None, **kwargs) -> dict:
    """Run ``cross_validate`` keeping the fitted pipelines and the fold positions."""
    pipeline = _panel_pipeline() if pipeline is None else pipeline
    return cross_validate(pipeline, X, y, cv=cv, return_estimator=True, return_indices=True, error_score='raise',
                          **kwargs)


def _folds(results: dict):
    """Iterate over ``(fitted pipeline, train positions, test positions)`` of a ``cross_validate`` result."""
    return zip(results['estimator'], results['indices']['train'], results['indices']['test'])


def _global_cv() -> PanelOutOfSampleSplit:
    """Global splitter: the three last half-years, common to the entities."""
    return PanelOutOfSampleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)


def _per_entity_cv() -> PanelOutOfSampleSplitPerEntity:
    """Per-entity splitter: the three last half-years of each entity."""
    return PanelOutOfSampleSplitPerEntity(n_splits=N_SPLITS, test_size=TEST_SIZE)


def _tested_entity(X: pd.DataFrame, test) -> str:
    """Return the single entity of the test rows of a per-entity fold."""
    (entity,) = X.index[test].get_level_values(0).unique()
    return entity


def _entity_scores(X: pd.DataFrame, results: dict, entity: str) -> list:
    """Return the scores of the per-entity folds testing ``entity``, in fold order."""
    return [score for score, test in zip(results['test_score'], results['indices']['test'])
            if _tested_entity(X, test) == entity]


def _fitted_frames(results: dict) -> list:
    """Return the frame each fold's estimator was fitted on."""
    return [pipeline[-1].fit_X_ for pipeline in results['estimator']]


def _assert_same_frames(left: list, right: list) -> None:
    """Assert that two lists of frames are equal, frame by frame."""
    assert len(left) == len(right)
    for left_frame, right_frame in zip(left, right):
        pd.testing.assert_frame_equal(left_frame, right_frame)


# =============================================================================
# Jeux de données et validations croisées partagés par le module
# =============================================================================
@pytest.fixture(scope='module')
def panel_xy(_heterogeneous_coverage_panel_session):
    """``(X, y)`` of the realistic heterogeneous-coverage panel (225 rows, 3 entities)."""
    return _split(_heterogeneous_coverage_panel_session.copy())


@pytest.fixture(scope='module')
def series_xy(_irregular_index_timeseries_session):
    """``(X, y)`` of the realistic irregular series (82 rows: 3 annual dates, then 79 months)."""
    return _split(_irregular_index_timeseries_session.copy())


@pytest.fixture(scope='module')
def global_results(panel_xy):
    """``cross_validate`` of the panel pipeline with the global splitter."""
    X, y = panel_xy
    return _run(X, y, _global_cv())


@pytest.fixture(scope='module')
def per_entity_results(panel_xy):
    """``cross_validate`` of the panel pipeline with the per-entity splitter."""
    X, y = panel_xy
    return _run(X, y, _per_entity_cv())


@pytest.fixture(scope='module')
def perturbed_others_results(panel_xy):
    """Per-entity ``cross_validate`` after scaling the features of Allemagne and Italie by 10."""
    X, y = panel_xy
    X_perturbed = X.copy()
    X_perturbed.loc[['Allemagne', 'Italie']] *= 10
    return _run(X_perturbed, y, _per_entity_cv())


@pytest.fixture(scope='module')
def series_results(series_xy):
    """``cross_validate`` of the series pipeline with ``TSOutOfSampleSplit``."""
    X, y = series_xy
    return _run(X, y, TSOutOfSampleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE), pipeline=_series_pipeline())


# =============================================================================
# Flux global : plis communs aux entités
# =============================================================================
class TestGlobalPanelCrossValidation:
    """Global splitter: the complete pipeline is scored on the common last half-years."""

    def test_every_fold_has_a_finite_score(self, global_results):
        """Three folds, each with a finite score."""
        scores = global_results['test_score']
        assert len(scores) == N_SPLITS and np.isfinite(scores).all()

    def test_test_folds_are_the_last_half_years(self, panel_xy, global_results):
        """Test dates of each fold: six months, the last half-years of the panel, in order."""
        X, _ = panel_xy
        bounds = [(X.index[test].get_level_values(1).min(), X.index[test].get_level_values(1).max())
                  for test in global_results['indices']['test']]
        assert bounds == [(TS('2023-02-01'), TS('2023-07-01')), (TS('2023-08-01'), TS('2024-01-01')),
                          (TS('2024-02-01'), TS('2024-07-01'))]

    def test_entity_ending_early_is_tested_on_its_own_dates(self, panel_xy, global_results):
        """Test rows per entity: six per fold, except Allemagne (ends 2024-04) in the last fold.

        Gold values: Allemagne holds the first test date of the last fold
        (2024-02-01) and is tested on February to April only, 3 rows.
        """
        X, _ = panel_xy
        counts = [X.index[test].get_level_values(0).value_counts().to_dict()
                  for test in global_results['indices']['test']]
        assert counts == [dict.fromkeys(ENTITIES, 6)] * 2 + [{'Allemagne': 3, 'France': 6, 'Italie': 6}]

    def test_no_training_date_reaches_the_test_period(self, panel_xy, global_results):
        """Entity by entity, every training date strictly precedes every test date of the fold."""
        X, _ = panel_xy
        folds = list(zip(global_results['indices']['train'], global_results['indices']['test']))
        assert_no_leakage(X, folds)

    def test_estimator_is_fitted_on_the_training_rows_with_a_target(self, panel_xy, global_results):
        """The rows reaching the estimator are the training rows whose target is observed."""
        X, y = panel_xy
        expected = [X.index[train][y.iloc[train].notna().to_numpy()] for train in global_results['indices']['train']]
        fitted = [frame.index for frame in _fitted_frames(global_results)]
        assert [index.equals(rows) for index, rows in zip(fitted, expected)] == [True] * N_SPLITS

    def test_delays_mask_three_cells_per_complete_quarter(self, panel_xy, global_results):
        """Cells hidden by the delays step, per entity and fold, on the training rows.

        Gold values: on 2024-08-15, the last month of each quarter of inflation
        (20 days) and the last two of unemployment (50 days) are unpublished:
        3 cells per complete quarter of each entity's monthly grid. Complete
        training quarters of the first fold (training ends January 2023):
        France 2018Q1-2022Q4 (20), Allemagne 2018Q3-2022Q4 (18), Italie
        2019Q1-2022Q4 (16); each later fold adds two quarters. The annual
        dates before the grid and the incomplete last quarter hide nothing.
        """
        X, _ = panel_xy
        hidden = []
        for pipeline, train, _ in _folds(global_results):
            X_train = X.iloc[train]
            masked = pipeline.named_steps['delays'].transform(X_train).isna() & X_train.notna()
            hidden.append(masked.groupby(level=0).sum().sum(axis=1).to_dict())
        assert hidden == [{'Allemagne': 54, 'France': 60, 'Italie': 48},
                          {'Allemagne': 60, 'France': 66, 'Italie': 54},
                          {'Allemagne': 66, 'France': 72, 'Italie': 60}]

    @pytest.mark.parametrize('fold', range(N_SPLITS), ids=lambda fold: f'fold-{fold}')
    def test_test_period_has_no_influence_on_the_fit(self, panel_xy, global_results, fold):
        """Rows dated from the first test date of a fold on, scaled by 100: that fold's fit is unchanged.

        Features and target of every entity are perturbed from that date on;
        what the estimator of the fold is fitted on (after the delays,
        the per-entity centring and the target filtering) stays identical.
        """
        X, y = panel_xy
        test_start = X.index[global_results['indices']['test'][fold]].get_level_values(1).min()
        # Perturbation de la période de test du pli et de tout ce qui la suit
        later = (X.index.get_level_values(1) >= test_start)
        X_perturbed, y_perturbed = X.copy(), y.copy()
        X_perturbed.loc[later] *= 100
        y_perturbed.loc[later] *= 100
        perturbed = _run(X_perturbed, y_perturbed, _global_cv())
        pd.testing.assert_frame_equal(_fitted_frames(perturbed)[fold], _fitted_frames(global_results)[fold])


# =============================================================================
# Indépendance des entités
# =============================================================================
class TestEntityIndependence:
    """The result for one entity does not depend on the other entities."""

    def test_each_per_entity_fold_holds_a_single_entity(self, panel_xy, per_entity_results):
        """Nine folds (3 splits x 3 entities), training and test rows of one same entity each."""
        X, _ = panel_xy
        entities = [(tuple(X.index[train].get_level_values(0).unique()), _tested_entity(X, test))
                    for _, train, test in _folds(per_entity_results)]
        assert [train for train, _ in entities] == [(test,) for _, test in entities] and len(entities) == 9

    def test_every_per_entity_fold_has_a_finite_score(self, per_entity_results):
        """Nine finite scores."""
        assert np.isfinite(per_entity_results['test_score']).all()

    @pytest.mark.parametrize('entity', ENTITIES)
    def test_scores_of_an_entity_equal_those_of_the_entity_alone(self, panel_xy, per_entity_results, entity):
        """Per-entity scores on the whole panel equal the scores of the panel restricted to that entity."""
        X, y = panel_xy
        alone = _run(X.loc[[entity]], y.loc[[entity]], _per_entity_cv())
        np.testing.assert_array_equal(_entity_scores(X, per_entity_results, entity), alone['test_score'])

    def test_perturbing_the_other_entities_leaves_an_entity_unchanged(self, panel_xy, per_entity_results,
                                                                      perturbed_others_results):
        """Features of Allemagne and Italie scaled by 10: the scores of France are identical."""
        X, _ = panel_xy
        np.testing.assert_array_equal(_entity_scores(X, perturbed_others_results, 'France'),
                                      _entity_scores(X, per_entity_results, 'France'))

    def test_perturbation_of_the_other_entities_is_effective(self, panel_xy, per_entity_results,
                                                             perturbed_others_results):
        """Guard of the previous test: the perturbation does change the scores of Allemagne."""
        X, _ = panel_xy
        # Ridge n'est pas invariant d'échelle : multiplier les variables change ses prédictions
        assert (_entity_scores(X, perturbed_others_results, 'Allemagne')
                != _entity_scores(X, per_entity_results, 'Allemagne'))

    @pytest.mark.parametrize('entity', ENTITIES)
    def test_global_pipeline_transforms_each_entity_as_if_alone(self, panel_xy, global_results, entity):
        """In every global fold, an entity's rows reaching the estimator equal those of the entity alone.

        Expected rows: the per-entity steps (delays, then centring) fitted and
        applied on the training rows of that entity only; the pooled
        estimator itself learns from every entity, by design.
        """
        X, _ = panel_xy
        for pipeline, train, _ in _folds(global_results):
            X_entity = X.iloc[train].loc[[entity]]
            delayed = PanelwiseTransformer(_delays()).fit_transform(X_entity)
            expected = PanelwiseTransformer(EntityDemeaner()).fit_transform(delayed)
            fitted = pipeline[-1].fit_X_.loc[[entity]]
            pd.testing.assert_frame_equal(fitted, expected.loc[fitted.index])


# =============================================================================
# Non-mutation des entrées
# =============================================================================
class TestInputsAreNotMutated:
    """Neither ``cross_validate`` nor ``fit`` / ``predict`` / ``score`` modify ``X`` or ``y``."""

    @pytest.mark.parametrize('splitter', [_global_cv, _per_entity_cv], ids=['global', 'per-entity'])
    def test_cross_validate_leaves_the_panel_unchanged(self, _heterogeneous_coverage_panel_session, splitter):
        """Values, index and columns of ``X`` and ``y`` are identical after ``cross_validate``."""
        X, y = _split(_heterogeneous_coverage_panel_session.copy())
        X_before, y_before = X.copy(deep=True), y.copy(deep=True)
        _run(X, y, splitter())
        pd.testing.assert_frame_equal(X, X_before)
        pd.testing.assert_series_equal(y, y_before)

    def test_fit_predict_score_leave_the_panel_unchanged(self, _heterogeneous_coverage_panel_session):
        """A direct ``fit``, ``predict`` and ``score`` of the pipeline leave ``X`` and ``y`` identical."""
        X, y = _split(_heterogeneous_coverage_panel_session.copy())
        X_before, y_before = X.copy(deep=True), y.copy(deep=True)
        pipeline = _panel_pipeline().fit(X, y)
        pipeline.predict(X)
        pipeline.score(X, y)
        pd.testing.assert_frame_equal(X, X_before)
        pd.testing.assert_series_equal(y, y_before)

    def test_cross_validate_leaves_the_series_unchanged(self, _irregular_index_timeseries_session):
        """Values and irregular index of the series are identical after ``cross_validate``."""
        X, y = _split(_irregular_index_timeseries_session.copy())
        X_before, y_before = X.copy(deep=True), y.copy(deep=True)
        _run(X, y, TSOutOfSampleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE), pipeline=_series_pipeline())
        pd.testing.assert_frame_equal(X, X_before)
        pd.testing.assert_series_equal(y, y_before)


# =============================================================================
# Reproductibilité
# =============================================================================
class TestReproducibility:
    """Two runs of the same workflow give exactly the same results."""

    def test_second_run_gives_identical_scores(self, panel_xy, global_results):
        """Scores of a second ``cross_validate`` are bitwise identical."""
        X, y = panel_xy
        np.testing.assert_array_equal(_run(X, y, _global_cv())['test_score'], global_results['test_score'])

    def test_second_run_fits_on_identical_rows(self, panel_xy, global_results):
        """Frames the estimators are fitted on are identical from one run to the next."""
        X, y = panel_xy
        _assert_same_frames(_fitted_frames(_run(X, y, _global_cv())), _fitted_frames(global_results))


# =============================================================================
# Parallélisme
# =============================================================================
@pytest.mark.slow
class TestParallelism:
    """``n_jobs=2`` gives the results of ``n_jobs=1`` (worker start-up exceeds 2 s under Windows)."""

    def test_panelwise_steps_in_parallel(self, panel_xy, global_results):
        """``PanelwiseTransformer(n_jobs=2)``: same scores and same fitted rows as sequential steps."""
        X, y = panel_xy
        parallel = _run(X, y, _global_cv(), pipeline=_panel_pipeline(n_jobs=2))
        np.testing.assert_array_equal(parallel['test_score'], global_results['test_score'])
        _assert_same_frames(_fitted_frames(parallel), _fitted_frames(global_results))

    def test_cross_validate_in_parallel(self, panel_xy, global_results):
        """``cross_validate(n_jobs=2)``: same scores and same fitted rows as sequential folds."""
        X, y = panel_xy
        parallel = _run(X, y, _global_cv(), n_jobs=2)
        np.testing.assert_array_equal(parallel['test_score'], global_results['test_score'])
        _assert_same_frames(_fitted_frames(parallel), _fitted_frames(global_results))


# =============================================================================
# Série irrégulière réaliste
# =============================================================================
class TestRealisticIrregularSeries:
    """The same steps on the irregular series, and a one-entity panel equivalent to it."""

    def test_every_fold_has_a_finite_score(self, series_results):
        """Three folds, each with a finite score."""
        scores = series_results['test_score']
        assert len(scores) == N_SPLITS and np.isfinite(scores).all()

    def test_no_training_date_reaches_the_test_period(self, series_xy, series_results):
        """Every training date strictly precedes every test date of the fold."""
        X, _ = series_xy
        assert_no_leakage(X, list(zip(series_results['indices']['train'], series_results['indices']['test'])))

    def test_rows_without_target_are_dropped_before_the_estimator(self, series_results):
        """Rows reaching the estimator: 49, 55 and 61.

        Gold values: the training folds hold 64, 70 and 76 rows (3 annual
        dates, then the months up to January 2023, July 2023, January 2024);
        the target is missing on the 3 annual dates and on the 12 months of
        2018, before industrial production starts: 15 rows dropped per fold.
        """
        assert [len(frame) for frame in _fitted_frames(series_results)] == [49, 55, 61]

    def test_one_entity_panel_gives_the_results_of_the_series(self, series_xy, series_results):
        """The series as a one-entity panel, through the panel pipeline and splitter: same scores."""
        X, y = series_xy
        # Série placée sous une entité unique : MultiIndex (country, date)
        X_panel = pd.concat({'France': X}, names=['country', 'date'])
        y_panel = pd.concat({'France': y}, names=['country', 'date'])
        panel = _run(X_panel, y_panel, _global_cv())
        np.testing.assert_allclose(panel['test_score'], series_results['test_score'], rtol=1e-12)
