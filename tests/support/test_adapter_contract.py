"""Tests of the adapter contract battery of ``tests/support/adapter_contract.py``.

The whole battery passes on the reference adapters ``RowRegressorAdapter``
(row-wise regressor) and ``LastValueAdapter`` (local per-entity model), and each
``check_*`` function fails on a mutant adapter that breaks its rule: modified
wrapped object, sorted output, state kept across fits, gap-dependent or
unit-dependent predictions, silent unknown entity, accepted invalid parameter,
package imported without an installation hint.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from tests.support.adapter_contract import (
    AdapterCase,
    AdapterContractTests,
    AdapterWorkflowContractTests,
    check_datetime_resolution,
    check_gap_independence,
    check_invalid_params,
    check_missing_package,
    check_no_leakage,
    check_predict_row_order,
    check_unknown_entity,
    check_wrapped_object_untouched,
    default_data,
    shift_dates,
    split_by_date,
)
from tests.support.estimators import LastValueAdapter, RowRegressorAdapter
from tsforecast.adapters._base import check_adapter_input, steps_between

ROW_REGRESSOR_CASE = AdapterCase(
    make_estimator=lambda: RowRegressorAdapter(LinearRegression()),
    wrapped_param='regressor',
    param_grid={'regressor__fit_intercept': [True, False]},
    package='sklearn.linear_model',
    extra='adapters-reference',
)

LAST_VALUE_CASE = AdapterCase(
    make_estimator=lambda: LastValueAdapter(drift=0.5),
    wrapped_param='drift',
    param_grid={'drift': [0.0, 0.5]},
    local_model=True,
    row_regressor=False,
    invalid_params=({'drift': 'up'},),
)


class TestRowRegressorContract(AdapterContractTests, AdapterWorkflowContractTests):
    """The battery passes on a row-wise regressor adapter."""

    case = ROW_REGRESSOR_CASE


class TestLastValueContract(AdapterContractTests, AdapterWorkflowContractTests):
    """The battery passes on a local per-entity adapter."""

    case = LAST_VALUE_CASE


# =============================================================================
# Mutants : chaque adapter viole une règle du contrat
# =============================================================================


class MutatingAdapter(RowRegressorAdapter):
    """Fits the wrapped regressor itself."""

    def fit(self, X, y):
        super().fit(X, y)
        self.regressor.fit(X.to_numpy(), y.to_numpy())
        return self


class SortedOutputAdapter(RowRegressorAdapter):
    """Returns the predictions in sorted index order."""

    def predict(self, X):
        return self.regressor_.predict(X.sort_index().to_numpy())


class StatefulAdapter(RowRegressorAdapter):
    """Keeps the model of its first fit, and passes it to its clones through the regressor."""

    def fit(self, X, y):
        if not hasattr(self.regressor, 'coef_'):
            self.regressor.fit(X.to_numpy(), y.to_numpy())
        super().fit(X, y)
        self.regressor_ = self.regressor
        return self


class GapDependentAdapter(RowRegressorAdapter):
    """Adds a term growing with the distance to the training data."""

    def fit(self, X, y):
        super().fit(X, y)
        self.last_date_ = X.index.get_level_values(-1).max()
        return self

    def predict(self, X):
        _, _, layout = check_adapter_input(self, X, reset=False, freq=self.freq_)
        steps = steps_between(np.repeat(self.last_date_, len(X)), layout.dates, self.freq_)
        return super().predict(X) + 0.1 * steps


class UnitDependentAdapter(RowRegressorAdapter):
    """Reads the dates as raw integers, whose scale depends on their unit."""

    def predict(self, X):
        _, _, layout = check_adapter_input(self, X, reset=False, freq=self.freq_)
        return super().predict(X) + layout.dates.asi8 * 1e-20


class SilentLastValueAdapter(LastValueAdapter):
    """Predicts zero for an entity unknown at fit."""

    def predict(self, X):
        _, _, layout = check_adapter_input(self, X, reset=False, freq=self.layout_.freq)
        if (self.layout_.match_entities(layout) < 0).any():
            return np.zeros(len(X))
        return super().predict(X)


class PermissiveLastValueAdapter(LastValueAdapter):
    """Accepts any drift."""

    def fit(self, X, y):
        drift, self.drift = self.drift, 0.0
        try:
            return super().fit(X, y)
        finally:
            self.drift = drift


class HintlessAdapter(RowRegressorAdapter):
    """Imports its package directly, without an installation hint."""

    def fit(self, X, y):
        import sklearn.linear_model  # noqa: F401  import direct, sans message d'installation
        return super().fit(X, y)


def _mutant_case(estimator_class, base: AdapterCase = ROW_REGRESSOR_CASE) -> AdapterCase:
    """Case of ``base`` whose estimator is built from a mutant class."""
    if base is ROW_REGRESSOR_CASE:
        make = lambda: estimator_class(LinearRegression())  # noqa: E731
    else:
        make = lambda: estimator_class(drift=0.5)  # noqa: E731
    return AdapterCase(**{**base.__dict__, 'make_estimator': make})


class TestMutantsAreCaught:
    """Each check fails on the adapter that breaks its rule."""

    def test_modified_wrapped_object(self):
        with pytest.raises(AssertionError, match='modified the wrapped object'):
            check_wrapped_object_untouched(_mutant_case(MutatingAdapter))

    @pytest.mark.parametrize('kind', ['series', 'panel'])
    def test_sorted_output(self, kind):
        with pytest.raises(AssertionError, match='rows of X'):
            check_predict_row_order(_mutant_case(SortedOutputAdapter), kind)

    def test_state_kept_across_fits(self):
        with pytest.raises(AssertionError, match='previous fit'):
            check_no_leakage(_mutant_case(StatefulAdapter), 'series')

    def test_gap_dependent_prediction(self):
        with pytest.raises(AssertionError, match='depends on the gap'):
            check_gap_independence(_mutant_case(GapDependentAdapter), 'panel')

    def test_unit_dependent_prediction(self):
        with pytest.raises(AssertionError, match='change with dates'):
            check_datetime_resolution(_mutant_case(UnitDependentAdapter), 'series')

    def test_silent_unknown_entity(self):
        with pytest.raises(AssertionError, match='unknown at fit'):
            check_unknown_entity(_mutant_case(SilentLastValueAdapter, LAST_VALUE_CASE))

    def test_accepted_invalid_parameter(self):
        with pytest.raises(AssertionError, match='invalid parameters'):
            check_invalid_params(_mutant_case(PermissiveLastValueAdapter, LAST_VALUE_CASE))

    def test_missing_installation_hint(self, monkeypatch):
        with pytest.raises(AssertionError, match='does not name the extra'):
            check_missing_package(_mutant_case(HintlessAdapter), monkeypatch)


class TestBatteryHelpers:
    """Dataset split and date shift of the battery."""

    def test_split_by_date_leaves_the_gap_out(self):
        X, y = default_data('panel')
        X_train, y_train, X_test, _ = split_by_date(X, y, n_test=3, gap=2)
        calendar = X.index.get_level_values(-1).unique().sort_values()
        assert X_train.index.get_level_values(-1).max() == calendar[-6]
        assert X_test.index.get_level_values(-1).min() == calendar[-3]
        assert len(X_train) == len(y_train) == 3 * 31

    def test_shift_dates_moves_every_row(self):
        X, _ = default_data('unbalanced_panel')
        shifted = shift_dates(X, 2, 'MS')
        moved = shifted.index.get_level_values(-1) - X.index.get_level_values(-1)
        assert (moved.days >= 59).all()
        assert shifted.index.get_level_values(0).equals(X.index.get_level_values(0))

    def test_default_data_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match='Unknown data kind'):
            default_data('cube')
