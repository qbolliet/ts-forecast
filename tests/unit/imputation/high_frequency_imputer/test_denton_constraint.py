"""``HighFrequencyImputer`` under ``aggregation_constraint='denton'``.

Scenarios: annual totals of the reference series under every covariate strategy, totals of the
source frequency of each entity on PANEL-F, interpolation without step at the year boundaries
(against ``'sum'``), the coincident cells of ``impute_intermediate_frequencies=True`` dropped as
under ``'sum'``, and the per-column dict form.
"""

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Outils de test partagés
from tests.unit.imputation.high_frequency_imputer.helpers import (
    entity_block,
    fit_transform_quietly,
    fit_with_spy,
    make_imputer,
    replay,
)

# Totaux annuels observés de la série de référence
_A1_TOTALS = {'2021': 120.0, '2022': 132.0, '2023': 150.0}


# Sauts aux frontières d'années et mouvements intra-annuels d'une série mensuelle
def _boundary_and_inner_moves(values: pd.Series):
    """Split the absolute month-to-month moves into year boundaries and moves within a year."""
    moves = values.diff().abs().iloc[1:]
    at_boundary = moves.index.month == 1
    return moves[at_boundary], moves[~at_boundary]


class TestTotals:
    """Every complete period meets its total, as under ``'sum'``."""

    @pytest.mark.parametrize('strategy', ['tolerate_nan', 'interpolate', 'model'])
    def test_annual_totals_hold(self, reference_timeseries, strategy):
        """The twelve months of each year of a1 sum to its observed total."""
        imputer = fit_with_spy(
            reference_timeseries, covariate_strategy=strategy, aggregation_constraint='denton'
        )
        values, _provenance = replay(imputer, reference_timeseries)
        for year, total in _A1_TOTALS.items():
            assert values['a1'].loc[year].sum() == pytest.approx(total)

    def test_group_totals_are_those_of_the_source_frequency(self, mixed_freq_panel_multifrequency):
        """Annual totals for FR, quarterly for DE: one benchmarking per entity."""
        imputer = fit_with_spy(mixed_freq_panel_multifrequency, aggregation_constraint='denton')
        values, _provenance = replay(imputer, mixed_freq_panel_multifrequency)

        france = entity_block(values['v'], 'FR')
        assert france.loc['2021'].sum() == pytest.approx(120.0)
        germany = entity_block(values['v'], 'DE')
        assert germany.loc['2021-01':'2021-03'].sum() == pytest.approx(28.0)
        assert germany.loc['2021-04':'2021-06'].sum() == pytest.approx(30.0)


class TestShape:
    """The interpolated series does not step at the boundaries of its periods."""

    def test_interpolation_steps_under_sum_not_under_denton(self, reference_timeseries):
        """Without estimator: pro rata jumps at New Year far beyond its monthly moves, Denton bends only."""
        outputs = {
            constraint: fit_transform_quietly(
                make_imputer(estimator=None, keep_lower_frequencies=False,
                             aggregation_constraint=constraint),
                reference_timeseries,
            )['a1']
            for constraint in ('sum', 'denton')
        }
        boundary_sum, inner_sum = _boundary_and_inner_moves(outputs['sum'])
        assert boundary_sum.mean() > 3 * inner_sum.mean()

        # Rupture de janvier : écart entre le pas de janvier et celui de
        # décembre, au moins quatre fois plus faible sous Denton qu'au prorata
        for january in outputs['sum'].index[outputs['sum'].index.month == 1][1:]:
            december = january - pd.offsets.MonthEnd(1)
            november = december - pd.offsets.MonthEnd(1)
            breaks = {
                constraint: abs(
                    (series[january] - series[december]) - (series[december] - series[november])
                )
                for constraint, series in outputs.items()
            }
            assert breaks['denton'] < breaks['sum'] / 4
        # Mêmes totaux sous les deux contraintes
        for year, total in _A1_TOTALS.items():
            assert outputs['denton'].loc[year].sum() == pytest.approx(total)


class TestTrainingSet:
    """The training rows are those of ``'sum'``: both constraints make the totals exact."""

    def test_coincident_cells_are_dropped_as_under_sum(self, reference_timeseries):
        """Under ``True``, ``y_train`` of every step has the same rows under 'sum' and 'denton'."""
        rows = {}
        for constraint in ('sum', 'denton'):
            imputer = fit_with_spy(
                reference_timeseries, covariate_strategy='model',
                impute_intermediate_frequencies=True, aggregation_constraint=constraint,
            )
            rows[constraint] = {
                (step.pred_freq_label, step.var_name): len(step.model.fit_y_)
                for step in imputer.imputation_plan_
            }
        assert rows['denton'] == rows['sum']


class TestDictForm:
    """The setting resolves per column."""

    def test_one_column_smooth_the_others_pro_rata(self, reference_timeseries):
        """``{'a1': 'denton', '__default__': 'sum'}``: a1 as under 'denton', a2 as under 'sum'."""
        def output(constraint):
            return fit_transform_quietly(
                make_imputer(estimator=None, keep_lower_frequencies=False,
                             aggregation_constraint=constraint),
                reference_timeseries,
            )

        mixed = output({'a1': 'denton', '__default__': 'sum'})
        pd.testing.assert_series_equal(mixed['a1'], output('denton')['a1'])
        pd.testing.assert_series_equal(mixed['a2'], output('sum')['a2'])
        assert not np.allclose(mixed['a1'].to_numpy(), output('sum')['a1'].to_numpy())
