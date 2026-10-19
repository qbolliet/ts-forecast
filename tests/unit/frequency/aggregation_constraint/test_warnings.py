"""Warnings emitted by ``AggregationConstraint``.

Scenarios: one warning per degenerate case and per operation, and the deferred warnings
of a whole chain of calls.
"""

# Modules de base
import warnings

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Sklearn
from sklearn.base import clone

# Objets testés
from tsforecast.frequency.aggregation_constraint import (
    AggregationConstraint,
    DEFAULT_CONSTRAINT_KEY,
)
from tsforecast.frequency.covariate_materializer import CovariateMaterializer

# Outils de test partagés du paquet
from tests.unit.frequency.aggregation_constraint.helpers import (
    _annual_observations,
    _monthly_grid,
    _raw_predictions,
    _record_warnings,
)


# =============================================================================
# Avertissements agrégés et uniques
# =============================================================================
class TestWarnings:
    """One warning per degenerate case and per operation."""

    def test_nominal_rescaling_is_silent(self, two_column_stage):
        """No warning when every period is rescaled normally."""
        X, observations = two_column_stage

        _result, messages = _record_warnings(
            AggregationConstraint('sum', 'Y', observations).fit_transform, X
        )

        assert messages == []

    def test_rescale_warns_once_for_several_flipped_periods(self):
        """Two flipped years: ONE warning naming the count and the column."""
        grid = pd.date_range('2021-01-31', periods=24, freq='ME')
        values = pd.Series(-1.0, index=grid, name='a1')
        observations = pd.Series(np.nan, index=grid)
        observations.loc['2021-12-31'] = 120.0
        observations.loc['2022-12-31'] = 132.0

        _result, messages = _record_warnings(
            AggregationConstraint('sum').rescale, values, observations, 'Y'
        )

        assert len(messages) == 1
        assert "2 period(s), in column(s) ['a1'] (e.g. 2021)" in messages[0]

    def test_zero_and_flipped_cases_give_one_warning_each(self):
        """2021 aggregates to zero, 2022 is flipped: exactly two warnings, one per case."""
        grid = pd.date_range('2021-01-31', periods=24, freq='ME')
        values = pd.Series([1.0, -1.0] * 6 + [-1.0] * 12, index=grid, name='a1')
        observations = pd.Series(np.nan, index=grid)
        observations.loc['2021-12-31'] = 120.0
        observations.loc['2022-12-31'] = 132.0

        _result, messages = _record_warnings(
            AggregationConstraint('sum').rescale, values, observations, 'Y'
        )

        kinds = sorted('zero' if 'aggregate to zero' in m else 'flipped' for m in messages)
        assert kinds == ['flipped', 'zero']

    def test_truncated_periods_are_named_once(self):
        """Observed totals the grid holds in part: ONE warning, their count, the column.

        Grid March 2020 - October 2022: 2020 and 2022 hold ten months each and
        stay raw; their totals are lost in the values, which the warning
        says. The whole year 2021 is rescaled and counts for nothing.
        """
        grid = pd.date_range('2020-03-31', '2022-10-31', freq='ME')
        values = pd.Series(1.0, index=grid, name='a1')
        observations = pd.Series(
            [100.0, 120.0, 140.0], index=pd.date_range('2020-12-31', periods=3, freq='YE')
        )

        _result, messages = _record_warnings(
            AggregationConstraint('sum').rescale, values, observations, 'Y', grid_freq='M'
        )

        assert len(messages) == 1
        assert messages[0].startswith('Aggregation constraint not enforced')
        assert "2 observed period(s)" in messages[0] and "['a1']" in messages[0]

    def test_truncation_is_not_reported_without_a_grid_frequency(self):
        """Without ``grid_freq`` no truncation can be seen, hence none is reported."""
        grid = pd.date_range('2021-03-31', periods=10, freq='ME')
        values = pd.Series(1.0, index=grid, name='a1')
        observations = pd.Series(120.0, index=[grid[-1]])

        _result, messages = _record_warnings(
            AggregationConstraint('sum').rescale, values, observations, 'Y'
        )

        assert messages == []

    def test_transform_warns_once_for_every_column(self):
        """Two flipped columns: one warning naming both, with the total count."""
        grid = _monthly_grid()
        X = pd.DataFrame({'a1': [-1.0] * 12, 'a2': [-1.0] * 12}, index=grid)
        observations = pd.DataFrame(
            {'a1': [np.nan] * 11 + [120.0], 'a2': [np.nan] * 11 + [60.0]}, index=grid
        )
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)

        _result, messages = _record_warnings(constraint.transform, X)

        assert len(messages) == 1
        assert "2 period(s), in column(s) ['a1', 'a2']" in messages[0]

    def test_transform_counts_zero_periods_over_every_column(self):
        """Zero aggregates in a1 (one year) and a2 (two years): one warning, 3 periods."""
        grid = pd.date_range('2021-01-31', periods=24, freq='ME')
        X = pd.DataFrame({'a1': [0.0] * 12 + [1.0] * 12, 'a2': 0.0}, index=grid)
        observations = pd.DataFrame(np.nan, index=grid, columns=['a1', 'a2'])
        observations.loc[['2021-12-31', '2022-12-31'], :] = 12.0
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)

        _result, messages = _record_warnings(constraint.transform, X)

        assert len(messages) == 1
        assert "3 period(s) with a non-zero observed total, in column(s) ['a1', 'a2']" in messages[0]

    def test_context_is_quoted_in_the_message(self):
        """The context label follows 'for' in the message."""
        values = _raw_predictions(values=[0.0] * 12)

        _result, messages = _record_warnings(
            AggregationConstraint('sum', context="'a1' at stage M").rescale,
            values, _annual_observations(), 'Y',
        )

        assert messages[0].startswith("Aggregation constraint skipped for 'a1' at stage M:")

    def test_constraint_none_never_warns(self):
        """Under None, degenerate predictions raise no warning: nothing is rescaled."""
        values = _raw_predictions(values=[-1.0] * 12)

        _result, messages = _record_warnings(
            AggregationConstraint(None).rescale, values, _annual_observations(), 'Y'
        )

        assert messages == []


# Exécution d'une suite d'appels dans un bloc différé
def _within_deferred_block(constraint: AggregationConstraint, *calls) -> None:
    """Run each call inside one ``deferred_warnings`` block of ``constraint``.

    Args:
        constraint: Constraint whose block gathers the warnings.
        *calls: Zero-argument callables, run in order.
    """
    with constraint.deferred_warnings():
        for call in calls:
            call()


class TestDeferredWarnings:
    """``deferred_warnings``: one warning per case for a whole chain of calls.

    ``HighFrequencyImputer`` rescales every (stage, column) and every
    interpolated covariate; one warning per family is wanted, never one
    per variable × stage: the warnings accumulate, then an aggregated message
    is emitted at the end of the phase.
    """

    def test_several_rescale_calls_give_one_warning(self):
        """Two columns rescaled by two calls: ONE warning naming both, 2 periods."""
        constraint = AggregationConstraint('sum')
        flipped = _raw_predictions(values=[-1.0] * 12)

        _result, messages = _record_warnings(
            _within_deferred_block,
            constraint,
            lambda: constraint.rescale(flipped, _annual_observations(), 'Y', column='a1'),
            lambda: constraint.rescale(flipped, _annual_observations(), 'Y', column='a2'),
        )

        assert len(messages) == 1
        assert "2 period(s), in column(s) ['a1', 'a2'] (e.g. 2021)" in messages[0]

    def test_a_period_met_twice_counts_once(self):
        """The same year of the same column, rescaled at two grids, counts once.

        Valeur d'or : l'année 2021 de a1, inversée sur la grille mensuelle
        puis sur la grille trimestrielle — une seule période observée.
        """
        constraint = AggregationConstraint('sum')
        monthly = _raw_predictions(values=[-1.0] * 12)
        quarters = pd.date_range('2021-03-31', periods=4, freq='QE')
        quarterly = pd.Series(-1.0, index=quarters, name='a1')
        observations = pd.Series([np.nan, np.nan, np.nan, 120.0], index=quarters)

        _result, messages = _record_warnings(
            _within_deferred_block,
            constraint,
            lambda: constraint.rescale(monthly, _annual_observations(), 'Y', column='a1'),
            lambda: constraint.rescale(quarterly, observations, 'Y', column='a1'),
        )

        assert len(messages) == 1
        assert "1 period(s), in column(s) ['a1']" in messages[0]

    def test_both_cases_give_one_warning_each_for_the_block(self):
        """A zero aggregate in one call, a flipped profile in another: two warnings."""
        constraint = AggregationConstraint('sum')

        _result, messages = _record_warnings(
            _within_deferred_block,
            constraint,
            lambda: constraint.rescale(
                _raw_predictions(values=[0.0] * 12), _annual_observations(), 'Y', column='a1'
            ),
            lambda: constraint.rescale(
                _raw_predictions(values=[-1.0] * 12), _annual_observations(), 'Y', column='a2'
            ),
        )

        kinds = sorted('zero' if 'aggregate to zero' in m else 'flipped' for m in messages)
        assert kinds == ['flipped', 'zero']

    def test_transform_joins_the_block(self):
        """``transform`` inside the block reports to it rather than warning."""
        grid = _monthly_grid()
        X = pd.DataFrame({'a1': [-1.0] * 12}, index=grid)
        observations = pd.DataFrame({'a1': [np.nan] * 11 + [120.0]}, index=grid)
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)
        flipped = _raw_predictions(values=[-1.0] * 12).rename('a2')

        _result, messages = _record_warnings(
            _within_deferred_block,
            constraint,
            lambda: constraint.transform(X),
            lambda: constraint.rescale(flipped, _annual_observations(), 'Y', column='a2'),
        )

        assert len(messages) == 1
        assert "in column(s) ['a1', 'a2']" in messages[0]

    def test_nested_blocks_warn_once_at_the_outer_exit(self):
        """A nested block joins the outer census: nothing at the inner exit."""
        constraint = AggregationConstraint('sum')
        flipped = _raw_predictions(values=[-1.0] * 12)
        inner_messages = []

        def _nested():
            # Bloc imbriqué : rien n'est émis à sa sortie
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with constraint.deferred_warnings():
                    constraint.rescale(flipped, _annual_observations(), 'Y', column='a2')
            inner_messages.extend(str(w.message) for w in caught)

        _result, messages = _record_warnings(
            _within_deferred_block,
            constraint,
            lambda: constraint.rescale(flipped, _annual_observations(), 'Y', column='a1'),
            _nested,
        )

        assert inner_messages == []
        assert len(messages) == 1
        assert "2 period(s), in column(s) ['a1', 'a2']" in messages[0]

    def test_an_exception_drops_the_census(self):
        """A block left on an exception emits nothing, and leaves no census behind."""
        constraint = AggregationConstraint('sum')
        flipped = _raw_predictions(values=[-1.0] * 12)

        def _failing_block():
            with constraint.deferred_warnings():
                constraint.rescale(flipped, _annual_observations(), 'Y', column='a1')
                raise RuntimeError('boom')

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(RuntimeError, match='boom'):
                _failing_block()
        assert caught == []

        # Hors bloc, l'appel suivant avertit immédiatement : aucun état résiduel
        _result, messages = _record_warnings(
            constraint.rescale, flipped, _annual_observations(), 'Y', column='a1'
        )
        assert len(messages) == 1

    def test_a_silent_block_stays_silent(self, two_column_stage):
        """No degenerate period in the block: no warning at its exit."""
        X, observations = two_column_stage
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)

        _result, messages = _record_warnings(
            _within_deferred_block, constraint, lambda: constraint.transform(X)
        )

        assert messages == []

    def test_the_block_leaves_the_parameters_untouched(self):
        """``get_params`` and ``clone`` are the same before, inside and after the block."""
        constraint = AggregationConstraint({'a1': None, DEFAULT_CONSTRAINT_KEY: 'sum'})
        before = constraint.get_params()

        with constraint.deferred_warnings():
            inside = clone(constraint).get_params()

        assert inside == before
        assert constraint.get_params() == before

    def test_the_materializer_covariates_join_the_block(self):
        """Covariates interpolated through the applier report to the block.

        Deux interpolations de la même colonne, ancres 100 puis -10 : 2022
        agrège positivement face à -10 à chaque appel. Une seule période.
        """
        constraint = AggregationConstraint('sum')
        materializer = CovariateMaterializer(aggregation_constraint_applier=constraint)
        grid = pd.date_range('2021-01-31', periods=24, freq='ME')
        source = pd.DataFrame({'a2': np.nan}, index=grid)
        source.loc['2021-12-31', 'a2'] = 100.0
        source.loc['2022-12-31', 'a2'] = -10.0

        def _interpolate():
            materializer.interpolate_column(
                column='a2', grid_index=grid, stage_freq='M',
                detected_frequencies={'a2': 'Y'}, source_data=source,
            )

        _result, messages = _record_warnings(
            _within_deferred_block, constraint, _interpolate, _interpolate
        )

        assert len(messages) == 1
        assert "1 period(s), in column(s) ['a2'] (e.g. 2022)" in messages[0]
