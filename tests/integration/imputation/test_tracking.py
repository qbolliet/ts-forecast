"""``tsforecast.tracking.imputation_metrics`` on imputers fitted on realistic data.

Reads the metrics of the ``HighFrequencyImputer`` fitted on every cell of the
scenario matrix of ``realistic_runs.py`` (realistic datasets of notebook 3,
shared session cache with ``test_hfi_on_realistic_dataset.py``), and of an
imputer inside an ``XYPipeline``. Checks the tracking contract:

- a flat ``dict`` of ``str`` -> finite ``float``, loggable as is by
  ``mlflow.log_metrics``;
- stable keys: the same set whatever the dataset, strategy or modality,
  derived from the ``ProvenanceType`` enumeration, plus ``cv_score.*`` only
  under ``fit_predict_order='cv'`` and ``<column>.provenance.*`` only under
  ``per_column=True``;
- consistency with ``imputation_plan_.to_diagnostic_frame()``: number of
  stages, unanchored pairs, imputed cells of the target level (sum of the
  ``n_written`` of the target-stage steps) and provenance families emitted;
- in a pipeline, the metrics describe the last pass of the imputer (the
  training pass after ``fit``, the scored fold after ``score``).
"""
# Modules de base
import math
import warnings
from typing import Set

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Doublure d'estimateur et matrice des scénarios partagées
from tests.integration.imputation.realistic_runs import CASES
from tests.support.estimators import SpyEstimator

# Composants testés
from tsforecast.imputation import HighFrequencyImputer, ProvenanceType
from tsforecast.tracking import imputation_metrics
from tsforecast.xy import XYPipeline

# Familles de provenance comptées comme imputées : tout sauf l'observé et l'agrégation exacte
_IMPUTED_TYPES = [
    member.value for member in ProvenanceType
    if member not in (ProvenanceType.ORIGINAL, ProvenanceType.AGGREGATED)
]

# Clés de provenance attendues : une par famille et son pourcentage, plus le non-imputé
_PROVENANCE_KEYS: Set[str] = {
    key
    for name in [member.value for member in ProvenanceType] + ['not_imputed']
    for key in (f'provenance.{name}', f'provenance.{name}_pct')
}

# Clés globales attendues : provenance, nombre d'étapes, couples sans ancre
_BASE_KEYS: Set[str] = _PROVENANCE_KEYS | {'n_stages', 'n_unanchored_pairs'}

# Clés des scores d'ordonnancement validé (fit_predict_order='cv')
_CV_KEYS: Set[str] = {'cv_score.n', 'cv_score.min', 'cv_score.max', 'cv_score.mean'}


def _target_steps(imputer: HighFrequencyImputer) -> list:
    """Return the positions of the plan steps that belong to the target stage."""
    target_label = imputer.imputation_plan_[-1].pred_freq_label
    return [
        position for position, step in enumerate(imputer.imputation_plan_)
        if step.pred_freq_label == target_label
    ]


# =============================================================================
# Contrat de forme : dict plat de float
# =============================================================================
class TestFlatFloatDict:
    """The metrics are a flat ``str -> float`` mapping, loggable by ``mlflow.log_metrics``."""

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_metrics_are_finite_floats_keyed_by_str(self, record, dataset, strategy, modality):
        """Every key is a ``str``, every value a finite ``float``."""
        metrics = imputation_metrics(record['imputer'])

        offending = {
            key: value for key, value in metrics.items()
            if not isinstance(key, str)
            or not isinstance(value, float)
            or not math.isfinite(value)
        }

        assert isinstance(metrics, dict) and offending == {}


# =============================================================================
# Stabilité des clés
# =============================================================================
@pytest.fixture(scope='module')
def cv_ordered_imputer(_irregular_index_timeseries_session) -> HighFrequencyImputer:
    """Imputer of the realistic series under ``'model'`` with a validated ordering."""
    imputer = HighFrequencyImputer(
        target_frequency='M',
        estimator=LinearRegression(),
        covariate_strategy='model',
        fit_predict_order='cv',
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return imputer.fit(_irregular_index_timeseries_session.copy())


class TestStableKeys:
    """The key set depends neither on the data nor on the configuration."""

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_keys_are_the_same_for_every_cell(self, record, dataset, strategy, modality):
        """22 provenance keys + ``n_stages`` + ``n_unanchored_pairs``, for all 18 cells.

        Gold value derived from the ``ProvenanceType`` enumeration (10
        families, each with its percentage, plus ``not_imputed``), not from a
        previous output.
        """
        metrics = imputation_metrics(record['imputer'])

        assert set(metrics) == _BASE_KEYS

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_per_column_adds_one_prefixed_block_per_column(
        self, record, dataset, strategy, modality
    ):
        """``per_column=True`` adds ``<column>.provenance.*`` for each input column, nothing else."""
        metrics = imputation_metrics(record['imputer'], per_column=True)
        expected = _BASE_KEYS | {
            f'{column}.{key}' for column in record['data'].columns for key in _PROVENANCE_KEYS
        }

        assert set(metrics) == expected

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_successive_calls_agree(self, record, dataset, strategy, modality):
        """Reading the metrics changes nothing: two calls return equal dicts."""
        imputer = record['imputer']

        assert imputation_metrics(imputer) == imputation_metrics(imputer)

    def test_validated_ordering_adds_the_cv_score_keys(self, cv_ordered_imputer):
        """``fit_predict_order='cv'`` adds exactly ``cv_score.{n,min,max,mean}``."""
        metrics = imputation_metrics(cv_ordered_imputer)

        assert set(metrics) == _BASE_KEYS | _CV_KEYS

    def test_cv_score_count_matches_the_scored_variables(self, cv_ordered_imputer):
        """``cv_score.n`` counts the (stage, variable) scores the imputer recorded."""
        metrics = imputation_metrics(cv_ordered_imputer)
        recorded = sum(len(scores) for scores in cv_ordered_imputer.imputation_cv_scores_.values())

        assert metrics['cv_score.n'] == float(recorded)


# =============================================================================
# Cohérence avec la diagnostic frame du plan
# =============================================================================
class TestConsistencyWithDiagnosticFrame:
    """The metrics agree with ``imputation_plan_.to_diagnostic_frame()``."""

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_stage_count_matches_the_plan(self, record, dataset, strategy, modality):
        """``n_stages`` equals the number of distinct stages of the plan."""
        imputer = record['imputer']
        frame = imputer.imputation_plan_.to_diagnostic_frame()

        assert imputation_metrics(imputer)['n_stages'] == float(frame['stage'].nunique())

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_unanchored_pair_count_matches_the_plan(self, record, dataset, strategy, modality):
        """``n_unanchored_pairs`` equals the entities of the unanchored steps (0 here).

        Every entity of the realistic datasets observes each imputed column at
        least once: no step is unanchored, on either side.
        """
        imputer = record['imputer']
        frame = imputer.imputation_plan_.to_diagnostic_frame()
        planned = float(frame.loc[frame['unanchored'], 'n_entities'].sum())

        assert imputation_metrics(imputer)['n_unanchored_pairs'] == planned

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_imputed_cells_match_the_cells_written_at_the_target_stage(
        self, record, dataset, strategy, modality
    ):
        """Imputed cells counted on the target level = sum of ``n_written`` of the target-stage steps.

        The metrics count the target level of the last pass; ``fit_transform``
        replays the fit on the same data, so both counts describe the same
        cells.
        """
        imputer = record['imputer']
        metrics = imputation_metrics(imputer)
        frame = imputer.imputation_plan_.to_diagnostic_frame()

        counted = sum(metrics[f'provenance.{name}'] for name in _IMPUTED_TYPES)
        written = float(frame.loc[_target_steps(imputer), 'n_written'].sum())

        assert counted == written > 0

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_counted_families_are_those_emitted_at_the_target_stage(
        self, record, dataset, strategy, modality
    ):
        """The imputed families of non-zero count are exactly the ``emitted_provenance`` of the target stage."""
        imputer = record['imputer']
        metrics = imputation_metrics(imputer)
        frame = imputer.imputation_plan_.to_diagnostic_frame()

        counted = {name for name in _IMPUTED_TYPES if metrics[f'provenance.{name}'] > 0}
        emitted = {str(value) for value in frame.loc[_target_steps(imputer), 'emitted_provenance']}

        assert counted == emitted


# =============================================================================
# Dernière passe dans un pipeline
# =============================================================================
@pytest.fixture(scope='module')
def series_target_split(_irregular_index_timeseries_session):
    """``(X_train, y_train, X_test, y_test)`` of the realistic series, target ``taux_chomage``.

    Rows where the target is observed (monthly grid), the last 6 held out.
    """
    data = _irregular_index_timeseries_session.copy()
    kept = data[data['taux_chomage'].notna()]
    X, y = kept.drop(columns='taux_chomage'), kept['taux_chomage']
    return X.iloc[:-6], y.iloc[:-6], X.iloc[-6:], y.iloc[-6:]


def _flat_imputer() -> HighFrequencyImputer:
    """Imputer with a flat output, as placed in front of an estimator."""
    return HighFrequencyImputer(
        target_frequency='M', estimator=LinearRegression(), keep_lower_frequencies=False
    )


class TestLastPassInPipeline:
    """In a pipeline, the imputer's metrics describe its last pass."""

    @pytest.fixture(scope='class')
    def passes(self, series_target_split) -> dict:
        """Metrics after ``fit`` and after ``score``, and those of a standalone imputer."""
        X_train, y_train, X_test, y_test = series_target_split
        pipeline = XYPipeline([('imputer', _flat_imputer()), ('model', SpyEstimator())])
        standalone = _flat_imputer()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipeline.fit(X_train, y_train)
            after_fit = imputation_metrics(pipeline.named_steps['imputer'])
            # Référence : même imputeur, seul, sur la même passe d'entraînement
            standalone.fit_transform(X_train, y_train)
            training_pass = imputation_metrics(standalone)
            # Passe de scoring : le pipeline transforme le pli de test
            pipeline.score(X_test, y_test)
            after_score = imputation_metrics(pipeline.named_steps['imputer'])
            standalone.transform(X_test, y_test)
            scoring_pass = imputation_metrics(standalone)
        return {
            'after_fit': after_fit,
            'training_pass': training_pass,
            'after_score': after_score,
            'scoring_pass': scoring_pass,
        }

    def test_after_fit_the_metrics_describe_the_training_pass(self, passes):
        """After ``XYPipeline.fit``, the metrics equal those of the imputer's own ``fit_transform``."""
        assert passes['after_fit'] == passes['training_pass']

    def test_after_score_the_metrics_describe_the_scored_fold(self, passes):
        """After ``score``, the metrics describe the transformed test fold, not the training."""
        assert passes['after_score'] == passes['scoring_pass']

    def test_scoring_pass_counts_fewer_cells(self, passes):
        """The scored fold (6 months) holds fewer cells than the training pass."""
        def total(metrics: dict) -> float:
            return metrics['provenance.original'] + metrics['provenance.not_imputed'] + sum(
                metrics[f'provenance.{name}'] for name in _IMPUTED_TYPES
            )

        assert total(passes['after_score']) < total(passes['after_fit'])
