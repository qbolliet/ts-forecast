"""Tests du paquet ``tsforecast.tracking``.

Vérifie que les trois constructeurs de métriques renvoient des dictionnaires
plats de scalaires finis (contrat ``mlflow.log_metrics``), que les clés
attendues sont stables, et que les cas dégénérés (imputeur sans modèle,
splitter à un seul pli, transformateur de délais vide) ne lèvent pas.
"""
# Modules de base
import math
import warnings

# Calcul numérique et données
import numpy as np
import pandas as pd

# Cadre de test
import pytest

# Sklearn
from sklearn.linear_model import LinearRegression

# Objets testés
from tsforecast.tracking import imputation_metrics, delay_metrics, split_summary
from tsforecast.frequency import HighFrequencyImputer
from tsforecast.delays import PublicationDelayTransformer
from tsforecast.crossvals import TSOutOfSampleSplit, PanelOutOfSampleSplit


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def mixed_frequency_ts() -> pd.DataFrame:
    """Série temporelle mensuelle avec une variable annuelle à trois ancres."""
    dates = pd.date_range("2020-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "m1": rng.normal(size=48).cumsum() + 50,
            "m2": rng.normal(size=48).cumsum() + 20,
            "a1": np.nan,
        },
        index=dates,
    )
    for date in ["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"]:
        df.loc[date, "a1"] = rng.normal(120, 5)
    return df


def _is_flat_scalar_mapping(metrics) -> bool:
    """Tout est ``{str: float fini}``."""
    return isinstance(metrics, dict) and all(
        isinstance(k, str) and isinstance(v, float) and math.isfinite(v)
        for k, v in metrics.items()
    )


# --------------------------------------------------------------------------- #
# imputation_metrics
# --------------------------------------------------------------------------- #
def test_imputation_metrics_flat_and_finite(mixed_frequency_ts):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputer = HighFrequencyImputer(
            target_frequency="M", estimator=LinearRegression()
        ).fit(mixed_frequency_ts)

    metrics = imputation_metrics(imputer)

    assert _is_flat_scalar_mapping(metrics)
    # Clés structurelles stables
    assert metrics["n_stages"] == 1.0
    assert metrics["n_unanchored_pairs"] == 0.0
    # Bloc de provenance présent et cohérent (pourcentages dans [0, 100])
    pct_keys = [k for k in metrics if k.startswith("provenance.") and k.endswith("_pct")]
    assert pct_keys
    assert all(0.0 <= metrics[k] <= 100.0 for k in pct_keys)
    # Une variable annuelle imputée : de l'original ET du modèle
    assert metrics["provenance.original"] > 0
    assert metrics["provenance.model_on_true"] > 0


def test_imputation_metrics_per_column(mixed_frequency_ts):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputer = HighFrequencyImputer(
            target_frequency="M", estimator=LinearRegression()
        ).fit(mixed_frequency_ts)

    metrics = imputation_metrics(imputer, per_column=True)

    assert _is_flat_scalar_mapping(metrics)
    assert any(k.startswith("a1.provenance.") for k in metrics)


def test_imputation_metrics_cv_scores(mixed_frequency_ts):
    """Deux colonnes imputables + ordre 'cv' => ``cv_score.*`` renseigné."""
    df = mixed_frequency_ts.copy()
    df["a2"] = np.nan
    rng = np.random.default_rng(7)
    for date in ["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"]:
        df.loc[date, "a2"] = rng.normal(80, 3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imputer = HighFrequencyImputer(
            target_frequency="M",
            estimator=LinearRegression(),
            covariate_strategy="model",
            fit_predict_order="cv",
            min_cv_train_size=3,
            cv=2,
        ).fit(df)

    assert imputer.imputation_cv_scores_  # non vide
    metrics = imputation_metrics(imputer)
    assert _is_flat_scalar_mapping(metrics)
    assert metrics["cv_score.n"] == 2.0
    assert metrics["cv_score.min"] <= metrics["cv_score.mean"] <= metrics["cv_score.max"]


def test_imputation_metrics_unfitted_raises():
    with pytest.raises(AttributeError):
        imputation_metrics(HighFrequencyImputer(target_frequency="M"))


# --------------------------------------------------------------------------- #
# delay_metrics
# --------------------------------------------------------------------------- #
def test_delay_metrics_flat_and_finite():
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"gdp": rng.normal(size=24), "inflation": rng.normal(size=24)}, index=dates
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transformer = PublicationDelayTransformer(
            delays={"gdp": 45.0, "inflation": 30.0},
            strategy="shift",
            prediction_date="2022-01-15",
            delay_unit="D",
            reference_point="end",
        ).fit(X)

    metrics = delay_metrics(transformer)

    assert _is_flat_scalar_mapping(metrics)
    assert metrics["n_delayed_columns"] == 2.0
    assert metrics["n_shift_columns"] == 2.0
    assert metrics["n_mask_columns"] == 0.0
    assert metrics["delay.max"] == 45.0
    assert metrics["shift_periods.n"] == 2.0


def test_delay_metrics_unfitted_raises():
    with pytest.raises(AttributeError):
        delay_metrics(PublicationDelayTransformer(delays={"gdp": 45.0}))


# --------------------------------------------------------------------------- #
# split_summary
# --------------------------------------------------------------------------- #
def test_split_summary_time_series():
    X = np.arange(80).reshape(-1, 1)
    summary = split_summary(
        TSOutOfSampleSplit(n_splits=4, test_size=5, gap=2), X
    )

    assert _is_flat_scalar_mapping(summary)
    assert summary["n_splits"] == 4.0
    assert summary["test_size.min"] == summary["test_size.max"] == 5.0
    assert summary["gap"] == 2.0
    assert summary["train_size.min"] <= summary["train_size.max"]


def test_split_summary_minimal_folds():
    X = np.arange(30).reshape(-1, 1)
    summary = split_summary(TSOutOfSampleSplit(n_splits=2, test_size=5), X)

    assert summary["n_splits"] == 2.0
    assert _is_flat_scalar_mapping(summary)


def test_split_summary_panel_with_groups():
    entities = np.repeat(["A", "B"], 40)
    dates = np.tile(pd.date_range("2020-01-01", periods=40, freq="D"), 2)
    X = pd.DataFrame(
        {"x": np.arange(80)},
        index=pd.MultiIndex.from_arrays([entities, dates], names=["entity", "date"]),
    )
    summary = split_summary(
        PanelOutOfSampleSplit(n_splits=3, test_size=5), X, groups=entities
    )

    assert _is_flat_scalar_mapping(summary)
    assert summary["n_splits"] == 3.0


# =============================================================================
# detection_metrics et enrichissement de delay_metrics par fit_report_
# =============================================================================
def _detection_report(**kwargs):
    from tsforecast.delays import compare_and_detect_delays
    dates = pd.date_range('2023-01-01', periods=4, freq='MS')
    existing = pd.DataFrame({'PIB': [1.0, 2.0, 3.0, np.nan], 'OLD': [1.0] * 4}, index=dates)
    new = pd.DataFrame({'PIB': [1.0, 2.5, 3.0, 4.0], 'NEW': [1.0] * 4}, index=dates)
    return compare_and_detect_delays(
        new, existing, '2023-06-15', detection_mode='all_changes', return_report=True, **kwargs)[1]


def test_detection_metrics_gold_values():
    from tsforecast.tracking import detection_metrics
    metrics = detection_metrics(_detection_report())
    assert metrics['n_detected'] == 2.0
    assert metrics['n_new_values'] == 1.0
    assert metrics['n_revisions'] == 1.0
    assert metrics['n_vanished_values'] == 0.0
    assert metrics['n_columns_compared'] == 1.0
    assert metrics['n_columns_new_only'] == 1.0
    assert metrics['n_columns_existing_only'] == 1.0
    assert metrics['n_undetected_frequencies'] == 0.0
    assert metrics['n_rows_new'] == 4.0 and metrics['n_rows_existing'] == 4.0
    assert (metrics['delay.min'], metrics['delay.max'], metrics['delay.mean'], metrics['delay.median']) == (75.0, 134.0, 104.5, 104.5)
    assert (metrics['delay.n'], metrics['delay.n_negative']) == (2.0, 0.0)


def test_detection_metrics_flat_and_finite():
    from tsforecast.tracking import detection_metrics
    metrics = detection_metrics(_detection_report(), per_column=True)
    assert _is_flat_scalar_mapping(metrics)
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics['PIB.n_detected'] == 2.0


def test_detection_metrics_per_column_is_optional():
    from tsforecast.tracking import detection_metrics
    assert not any(key.endswith('.n_detected') for key in detection_metrics(_detection_report()))


def test_detection_metrics_without_known_delay_drops_the_statistics():
    from tsforecast.delays import compare_and_detect_delays
    from tsforecast.tracking import detection_metrics
    new = pd.DataFrame({'PIB': [1.0]}, index=pd.date_range('2023-01-01', periods=1, freq='MS'))
    with pytest.warns(UserWarning):
        _, report = compare_and_detect_delays(new, download_date='2023-06-15', return_report=True)
    metrics = detection_metrics(report)
    assert 'n_rows_existing' not in metrics
    assert metrics['delay.n'] == 0.0
    assert metrics['n_undetected_frequencies'] == 1.0
    assert not any(key in metrics for key in ('delay.min', 'delay.max', 'delay.mean', 'delay.median'))


def test_detection_metrics_rejects_anything_but_a_report():
    from tsforecast.tracking import detection_metrics
    with pytest.raises(AttributeError):
        detection_metrics(object())


def test_delay_metrics_reports_the_fit_events():
    index = pd.date_range('2023-01-01', periods=12, freq='MS')
    X = pd.DataFrame({'GDP': range(12), 'Z': range(12)}, index=index)
    delays = pd.DataFrame({'column': ['GDP', 'OLD'], 'delay': [400.0, 5.0], 'unit': ['D', 'D'],
                           'reference_point': ['start', 'start'], 'frequency': ['Q', 'Q']})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        transformer = PublicationDelayTransformer(
            delays=delays, strategy='mask', prediction_date='2023-12-15').fit(X)
    metrics = delay_metrics(transformer)
    assert metrics['n_mask_fallbacks'] == 1.0
    assert metrics['n_columns_ignored'] == 1.0
    assert metrics['n_columns_unaffected'] == 1.0
    assert metrics['n_defaults_imputed'] == 0.0
    assert _is_flat_scalar_mapping(metrics)


def test_delay_metrics_keeps_working_without_a_fit_report():
    class Legacy:
        shift_params = {'a': {'n_periods': 2}}
        mask_params = {}
        delays = {'a': 10.0}

    metrics = delay_metrics(Legacy())
    assert metrics['n_shift_columns'] == 1.0
    assert 'n_mask_fallbacks' not in metrics
