"""Training and prediction passes of ``HighFrequencyImputer`` once fitted with ``y``.

Covers the role of the target (``target_as_covariate``: ``y`` imputed from
``X`` but kept out of the covariates of ``X`` by default), the training pass
(``fit_transform(X, y)``: ``X`` and ``y`` on the target grid, no row dropped)
and the prediction pass (``transform(X, y=None)`` after ``fit(X, y)``: the
columns of ``X`` alone, on the input rows, flat, ``y`` handed back unchanged,
provenance and statistics describing that output, inversion, additive
transformer, refusal of a model that needs the target). Without ``y`` at
``fit`` nothing changes (see ``test_invariants.py``, ``test_transform_state.py``).

Data: the ``TS`` reference dataset, target ``y = a2`` — annual,
gold totals 60 / 66 / 72 —, features ``m1`` (monthly), ``q1`` (quarterly),
``a1`` (annual).
"""
# Modules de base
import warnings
from typing import Tuple

import pandas as pd
import pytest
from sklearn.preprocessing import FunctionTransformer

# Objets testés
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer
from tsforecast.frequency.provenance import ProvenanceType

# Outils de test partagés
from tests.unit.frequency.high_frequency_imputer.helpers import make_imputer

# Cible : colonne annuelle a2 du jeu TS
TARGET = 'a2'


def _quietly(function, *args, **kwargs):
    """Call ``function`` with every warning silenced and return its result."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


@pytest.fixture
def xy(reference_timeseries) -> Tuple[pd.DataFrame, pd.Series]:
    """``(X, y)`` of the ``TS`` dataset: ``X = m1, q1, a1``, ``y = a2`` (annual)."""
    return reference_timeseries.drop(columns=TARGET), reference_timeseries[TARGET]


def _fitted(X: pd.DataFrame, y: pd.Series, **params) -> HighFrequencyImputer:
    """Fit an imputer with ``y``, warnings silenced."""
    return _quietly(make_imputer(**params).fit, X, y)


def _features_by_variable(imputer: HighFrequencyImputer) -> dict:
    """Return ``{imputed column: feature_cols}`` of the plan."""
    return {step.var_name: tuple(step.feature_cols) for step in imputer.imputation_plan_}


# =============================================================================
# Rôle de la cible : target_as_covariate
# =============================================================================
class TestTargetAsCovariate:
    """``y`` is imputed from ``X``, never a covariate of ``X`` unless asked."""

    def test_target_is_kept_out_of_the_covariates_by_default(self, xy):
        """Valeur d'or : q1 <- (m1, a1), a1 <- (m1, q1) ; a2 n'entre dans aucune."""
        imputer = _fitted(*xy)

        features = _features_by_variable(imputer)

        assert (features['q1'], features['a1']) == (('m1', 'a1'), ('m1', 'q1'))

    def test_target_is_still_imputed_from_x(self, xy):
        """The target keeps its own step, fed by every column of ``X``."""
        imputer = _fitted(*xy)

        assert _features_by_variable(imputer)[TARGET] == ('m1', 'q1', 'a1')

    def test_true_lets_the_target_in(self, xy):
        """``target_as_covariate=True``: a2 becomes a covariate of q1 and a1."""
        imputer = _fitted(*xy, target_as_covariate=True)

        features = _features_by_variable(imputer)

        assert TARGET in features['q1'] and TARGET in features['a1']

    def test_inert_without_target(self, reference_timeseries):
        """Without ``y``, a2 is a column like any other: both values give one plan."""
        frames = [
            _quietly(make_imputer(target_as_covariate=value).fit, reference_timeseries)
            .imputation_plan_.to_diagnostic_frame()
            for value in (False, True)
        ]

        pd.testing.assert_frame_equal(frames[0], frames[1])


# =============================================================================
# Passe d'entraînement : fit_transform(X, y)
# =============================================================================
class TestTrainingPass:
    """``fit_transform(X, y)`` augments the training set: ``X`` and ``y`` on the grid."""

    def test_imputed_target_sums_to_its_annual_totals(self, xy):
        """Valeur d'or : 60 / 66 / 72, recalage exact sous 'sum'."""
        _X_out, y_out = _quietly(make_imputer(keep_lower_frequencies=False).fit_transform, *xy)

        totals = y_out.groupby(y_out.index.year).sum().round(9).tolist()

        assert totals == [60.0, 66.0, 72.0]

    def test_target_is_filled_on_every_month(self, xy):
        """The annual target comes out monthly: 36 filled cells out of 3 observations."""
        _X_out, y_out = _quietly(make_imputer(keep_lower_frequencies=False).fit_transform, *xy)

        assert (int(xy[1].notna().sum()), int(y_out.notna().sum())) == (3, 36)

    def test_x_and_y_share_their_index(self, xy):
        """The two outputs of the training pass are aligned row for row."""
        X_out, y_out = _quietly(make_imputer().fit_transform, *xy)

        assert X_out.index.equals(y_out.index)

    def test_stacked_output_is_kept(self, xy):
        """``keep_lower_frequencies=True`` still stacks the training pass."""
        X_out, _y_out = _quietly(make_imputer().fit_transform, *xy)

        assert X_out.index.names == ['frequency', 'date']


# =============================================================================
# Passe de prédiction : transform après fit(X, y)
# =============================================================================
class TestPredictionPass:
    """``transform`` after ``fit(X, y)`` imputes ``X`` alone, on its input rows."""

    def test_output_is_on_the_input_rows(self, xy):
        """Flat output on the input index, whatever ``keep_lower_frequencies``."""
        X, y = xy
        imputer = _fitted(X, y, keep_lower_frequencies=True)

        out = _quietly(imputer.transform, X)

        assert out.index.equals(X.index)

    def test_output_holds_the_columns_of_x_only(self, xy):
        """The target column never appears in the prediction output."""
        X, y = xy
        imputer = _fitted(X, y)

        out = _quietly(imputer.transform, X)

        assert list(out.columns) == list(X.columns)

    def test_x_is_imputed_as_in_the_training_pass(self, xy):
        """On the training data, the same models give the same ``X``, bit for bit."""
        X, y = xy
        imputer = make_imputer(keep_lower_frequencies=False)
        X_train, _y_train = _quietly(imputer.fit_transform, X, y)

        out = _quietly(imputer.transform, X)

        pd.testing.assert_frame_equal(out, X_train.reindex(X.index))

    def test_x_is_filled_on_every_row(self, xy):
        """q1 and a1 come out monthly on all 36 rows: the prediction pass does impute."""
        X, y = xy
        imputer = _fitted(X, y)

        out = _quietly(imputer.transform, X)

        assert int(out.isna().sum().sum()) == 0

    def test_target_is_handed_back_unchanged(self, xy):
        """``transform(X, y)`` returns ``y`` as given: NaN outside the 3 anchors kept."""
        X, y = xy
        imputer = _fitted(X, y)

        _out, y_out = _quietly(imputer.transform, X, y)

        pd.testing.assert_series_equal(y_out, y)

    @pytest.mark.parametrize('factor', [10.0, -1.0], ids=['scaled', 'negated'])
    def test_imputed_x_ignores_the_target(self, xy, factor):
        """The values of ``y`` passed to ``transform`` change nothing in ``X``."""
        X, y = xy
        imputer = _fitted(X, y)

        reference = _quietly(imputer.transform, X)
        out, _y_out = _quietly(imputer.transform, X, y * factor)

        pd.testing.assert_frame_equal(out, reference)

    def test_provenance_describes_the_output(self, xy):
        """``imputation_provenance_`` has the shape, index and columns of the output."""
        X, y = xy
        imputer = _fitted(X, y)

        out = _quietly(imputer.transform, X)
        provenance = imputer.imputation_provenance_

        assert provenance.index.equals(out.index) and list(provenance.columns) == list(out.columns)

    def test_every_filled_cell_is_marked(self, xy):
        """Each filled cell of the prediction output carries a provenance."""
        X, y = xy
        imputer = _fitted(X, y)

        out = _quietly(imputer.transform, X)

        assert not (out.notna() & imputer.imputation_provenance_.isna()).any().any()

    def test_statistics_count_the_output_cells(self, xy):
        """Valeur d'or : 36 lignes x 3 colonnes = 108 cellules comptées, la cible exclue."""
        X, y = xy
        imputer = _fitted(X, y)

        _quietly(imputer.transform, X)
        overall = imputer.provenance_statistics_['overall']
        counted = sum(value for key, value in overall.items() if not key.endswith('_pct'))

        assert counted == 108

    def test_observed_cells_stay_original(self, xy):
        """m1, observed every month, keeps its values and ``ORIGINAL`` marks."""
        X, y = xy
        imputer = _fitted(X, y)

        out = _quietly(imputer.transform, X)

        assert out['m1'].equals(X['m1']) and (
            imputer.imputation_provenance_['m1'] == ProvenanceType.ORIGINAL
        ).all()

    def test_inverse_transform_restores_the_input(self, xy):
        """After a prediction pass, index and observed values come back."""
        X, y = xy
        imputer = _fitted(X, y, restore_original_values=True)

        out = _quietly(imputer.transform, X)
        restored = _quietly(imputer.inverse_transform, out)

        pd.testing.assert_frame_equal(restored, X, check_freq=False)

    def test_prediction_pass_on_later_rows(self, xy):
        """A test fold of the last 12 months is imputed on its own rows."""
        X, y = xy
        imputer = _fitted(X.iloc[:24], y.iloc[:24])

        out = _quietly(imputer.transform, X.iloc[24:])

        assert out.index.equals(X.index[24:]) and int(out.isna().sum().sum()) == 0

    def test_additive_transformer_fitted_with_the_target(self, xy):
        """The additive transformer, fitted on a frame holding ``y``, receives it empty."""
        X, y = xy
        identity = FunctionTransformer(validate=False)
        imputer = _fitted(X, y, additive_transformer=identity)

        out = _quietly(imputer.transform, X)

        pd.testing.assert_frame_equal(out, _quietly(_fitted(X, y).transform, X))

    def test_model_needing_the_target_is_refused(self, xy):
        """Under ``target_as_covariate=True``, the prediction pass names the dependent columns."""
        X, y = xy
        imputer = _fitted(X, y, target_as_covariate=True)

        with pytest.raises(ValueError, match=r"cannot impute \['a1', 'q1'\]"):
            imputer.transform(X)

    def test_refused_pass_leaves_the_previous_one_invertible(self, xy):
        """A refused prediction pass publishes nothing: the fit snapshot stays in place."""
        X, y = xy
        imputer = _fitted(X, y, target_as_covariate=True)
        before = imputer.imputation_provenance_.copy()

        with pytest.raises(ValueError):
            imputer.transform(X)

        pd.testing.assert_frame_equal(imputer.imputation_provenance_, before)


# =============================================================================
# Sans y au fit : rien ne change
# =============================================================================
class TestWithoutTarget:
    """Fitted without ``y``, ``transform`` keeps replaying the plan on any data."""

    def test_transform_still_stacks_the_output(self, reference_timeseries):
        """No prediction pass: the stacked output of ``keep_lower_frequencies=True`` stays."""
        imputer = _quietly(make_imputer().fit, reference_timeseries)

        out = _quietly(imputer.transform, reference_timeseries)

        assert out.index.names == ['frequency', 'date']

    def test_fit_transform_equals_fit_then_transform(self, reference_timeseries):
        """Without ``y``, the two calls give the same values."""
        first = _quietly(make_imputer().fit_transform, reference_timeseries)
        imputer = _quietly(make_imputer().fit, reference_timeseries)

        pd.testing.assert_frame_equal(first, _quietly(imputer.transform, reference_timeseries))

    def test_target_column_is_none(self, reference_timeseries):
        """No ``y``, no target column: the dispatch reads ``target_column_``."""
        imputer = _quietly(make_imputer().fit, reference_timeseries)

        assert imputer.target_column_ is None
