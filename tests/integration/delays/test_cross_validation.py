"""``PublicationDelayTransformer`` in an ``XYPipeline`` evaluated by ``cross_validate``.

Chains ``tsforecast.delays.transformers.PublicationDelayTransformer``,
``tsforecast.xy.XYPipeline`` (with the NaN-tolerant ``SpyEstimator`` of
``tests/support/estimators.py``, which records what it is fitted on and asked
to predict), ``tsforecast.crossvals.TSOutOfSampleSplit`` and
``sklearn.model_selection.cross_validate``, on the realistic irregular series of
notebook 3 (``irregular_index_timeseries``: annual dates 2015-2017 before a
monthly grid 2018-01 -> 2024-07). Target: ``production_industrielle``;
features: the four other columns, two of them delayed.

Contract checked: no data of the test fold is visible at ``fit`` (the
estimator is fitted on the training rows only, and no value of the test fold
reaches it), and the delays are actually applied on each training fold.

With the ``'mask'`` strategy the index is unchanged and the chain runs end to
end. With the ``'shift'`` strategy, ``transform`` moves the dates of ``X`` (its
output index is the union of the shifted dates, by design) while ``y`` goes
through the pipeline unchanged: scoring needs a later XY step realigning ``y`` on
``X``, not provided by ``PublicationDelayTransformer`` (``ANO-DELAYS-044``, decided:
not a defect of the class). The absence of leakage at ``fit`` is checked fold by fold.
"""
# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_validate

# Doublure d'estimateur partagée
from tests.support.estimators import SpyEstimator

# Composants enchaînés
from tsforecast.crossvals import TSOutOfSampleSplit
from tsforecast.delays.transformers import PublicationDelayTransformer
from tsforecast.xy import XYPipeline

TS = pd.Timestamp

# Date de prédiction commune à tous les plis (paramètre fixe du transformateur)
PREDICTION_DATE = '2024-08-15'

# Délais en jours depuis le début du mois (inflation : 20 jours, chômage : 50 jours)
DELAYS = {'inflation_ipc': 20.0, 'taux_chomage': 50.0}

TARGET = 'production_industrielle'

# Trois plis de test de six mois : 2023-02 -> 2023-07, 2023-08 -> 2024-01, 2024-02 -> 2024-07
N_SPLITS, TEST_SIZE = 3, 6


def _pipeline(strategy: str) -> XYPipeline:
    """Build the delays -> spy estimator pipeline for a strategy ('mask': quarterly target frequency)."""
    extra = {'target_frequency': 'Q'} if strategy == 'mask' else {}
    delays = PublicationDelayTransformer(delays=DELAYS, prediction_date=PREDICTION_DATE, strategy=strategy,
                                         delay_unit='D', reference_point='start', handle_missing_delays='ignore',
                                         **extra)
    return XYPipeline([('delays', delays), ('model', SpyEstimator())])


def _split(data: pd.DataFrame):
    """Return ``(X, y)``: the target and the four other columns."""
    return data.drop(columns=TARGET), data[TARGET]


def _values(frame: pd.DataFrame) -> set:
    """Return the set of non-missing values of a frame."""
    values = frame.to_numpy(dtype=float).ravel()
    return set(values[~np.isnan(values)])


@pytest.fixture(scope='module')
def masked_cv(_irregular_index_timeseries_session):
    """Run ``cross_validate`` on the 'mask' pipeline; return ``(X, results)``."""
    X, y = _split(_irregular_index_timeseries_session.copy())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results = cross_validate(_pipeline('mask'), X, y, cv=TSOutOfSampleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE),
                                 scoring='neg_mean_absolute_error', return_estimator=True, return_indices=True,
                                 error_score='raise')
    return X, results


def _folds(results):
    """Iterate over ``(fitted pipeline, train positions, test positions)`` of a ``cross_validate`` result."""
    return zip(results['estimator'], results['indices']['train'], results['indices']['test'])


class TestMaskedPipelineCrossValidation:
    """'mask' strategy: the pipeline is cross-validated without leakage."""

    def test_every_fold_is_scored(self, masked_cv):
        """Three folds, each with a finite score."""
        _, results = masked_cv
        assert len(results['test_score']) == N_SPLITS and np.isfinite(results['test_score']).all()

    def test_test_folds_are_the_last_months(self, masked_cv):
        """The test folds are the three last half-years of the monthly grid, in order (chronological split)."""
        X, results = masked_cv
        bounds = [(X.index[test][0], X.index[test][-1]) for test in results['indices']['test']]
        assert bounds == [(TS('2023-02-01'), TS('2023-07-01')), (TS('2023-08-01'), TS('2024-01-01')),
                          (TS('2024-02-01'), TS('2024-07-01'))]

    def test_estimator_is_fitted_on_the_training_rows_only(self, masked_cv):
        """The rows reaching the estimator at ``fit`` are exactly those of the training fold."""
        X, results = masked_cv
        assert all(pipeline[-1].fit_X_.index.equals(X.index[train]) for pipeline, train, _ in _folds(results))

    def test_no_value_of_the_test_fold_is_visible_at_fit(self, masked_cv):
        """No non-missing value of the test fold is among the values the estimator is fitted on."""
        X, results = masked_cv
        leaked = [_values(pipeline[-1].fit_X_) & _values(X.iloc[test]) for pipeline, _, test in _folds(results)]
        assert leaked == [set()] * N_SPLITS

    def test_delays_are_applied_on_each_training_fold(self, masked_cv):
        """Cells hidden at ``fit``: 3 per complete quarter of the monthly grid, 60 / 66 / 72 per fold.

        Gold values: on 2024-08-15, August inflation (20 days) is published on
        August 21, July on July 21: 1 unpublished month; July unemployment (50
        days) is published on August 20, June on July 21: 2 unpublished months.
        The last 1 and 2 months of each quarter are hidden, i.e. 3 cells per
        quarter. The complete quarters of the training folds
        are 2018Q1 -> 2022Q4 (20), -> 2023Q2 (22), -> 2023Q4 (24); the last,
        incomplete quarter (January 2023, July 2023, January 2024 alone) has no
        month in last position, nothing is hidden there.
        """
        X, results = masked_cv
        hidden = [int(pipeline[-1].fit_X_.isna().sum().sum() - X.iloc[train].isna().sum().sum())
                  for pipeline, train, _ in _folds(results)]
        assert hidden == [60, 66, 72]

    def test_estimator_predicts_on_the_test_rows_only(self, masked_cv):
        """At ``predict``, the estimator receives exactly the rows of the test fold."""
        X, results = masked_cv
        assert all(pipeline[-1].predict_X_[-1].index.equals(X.index[test]) for pipeline, _, test in _folds(results))


class TestShiftedPipelineCrossValidation:
    """'shift' strategy: the dates of ``X`` move, those of ``y`` do not."""

    def test_scoring_without_a_realignment_step_fails(self, irregular_index_timeseries):
        """Without an XY step realigning ``y`` on the shifted ``X``, scoring meets two lengths.

        ``transform`` keeps its contract (``X`` in, ``X`` out, index grown by the
        shift) and does not touch ``y`` (ANO-DELAYS-044, decided): a 6-month test
        fold gives 8 rows once inflation and unemployment are shifted by one and
        two months.
        """
        X, y = _split(irregular_index_timeseries)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match=r'inconsistent numbers of samples: \[6, 8\]'):
                cross_validate(_pipeline('shift'), X, y, cv=TSOutOfSampleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE),
                               scoring='neg_mean_absolute_error', error_score='raise')

    def test_no_value_of_the_test_fold_is_visible_at_fit(self, irregular_index_timeseries):
        """Fold by fold (what ``cross_validate`` does before scoring), the shift brings no test value into ``fit``.

        The shift moves the last training values to dates of the test fold
        (one month later): the dates overlap, the values come from the training rows only.
        """
        X, y = _split(irregular_index_timeseries)
        leaked = []
        for train, test in TSOutOfSampleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE).split(X, y):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pipeline = clone(_pipeline('shift')).fit(X.iloc[train], y.iloc[train])
            leaked.append(_values(pipeline[-1].fit_X_) & _values(X.iloc[test]))
        assert leaked == [set()] * N_SPLITS

    def test_shift_moves_training_values_onto_dates_of_the_test_fold(self, irregular_index_timeseries):
        """The rows reaching the estimator end one month after the training fold: 2023-02-01 for the first fold.

        Gold value: on 2024-08-15, the last published months of inflation (20 days)
        and unemployment (50 days) are July and June, hence shifts of one and two
        months: the
        January 2023 unemployment value lands on 2023-03-01, the last date seen
        at ``fit`` for a training fold ending in January 2023.
        """
        X, y = _split(irregular_index_timeseries)
        train, _ = next(TSOutOfSampleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE).split(X, y))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipeline = clone(_pipeline('shift')).fit(X.iloc[train], y.iloc[train])
        assert pipeline[-1].fit_X_.index.max() == TS('2023-03-01')
