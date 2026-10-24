"""sklearn protocol of ``AggregationConstraint``.

Scenarios: ``fit`` / ``transform`` behaviour and sklearn conventions.
"""

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Sklearn
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

# Objets testés
from tsforecast.imputation._components.aggregation_constraint import (
    AggregationConstraint,
    DEFAULT_CONSTRAINT_KEY,
)

# Perturbations partagées
from tests.support.perturbations import shuffle_rows, with_special_column_names

# Outils de test partagés du paquet
from tests.unit.imputation._components.aggregation_constraint.helpers import (
    _EXACT,
    _RAW_2021,
    _annual_observations,
    _monthly_grid,
)


class TestSklearnProtocol:
    """``fit`` / ``transform`` and sklearn conventions."""

    def test_get_params_returns_the_constructor_arguments_untouched(self, two_column_stage):
        """B3: the four parameters come back as the very objects received."""
        _X, observations = two_column_stage
        setting = {'a1': 'sum', DEFAULT_CONSTRAINT_KEY: None}
        frequencies = {'a1': 'Y', 'a2': 'Y'}
        constraint = AggregationConstraint(setting, frequencies, observations, 'ctx')

        params = constraint.get_params()

        assert set(params) == {'aggregation_constraint', 'period_frequencies',
                               'observations', 'context'}
        assert (params['aggregation_constraint'] is setting
                and params['period_frequencies'] is frequencies
                and params['observations'] is observations
                and params['context'] == 'ctx')

    def test_set_params_round_trip(self):
        """set_params changes the parameter read by the resolution."""
        constraint = AggregationConstraint('sum').set_params(aggregation_constraint=None)

        assert constraint.resolve_constraint('a1') is None

    def test_clone_is_an_unfitted_equivalent(self, two_column_stage):
        """clone of a fitted constraint: same parameters, no fitted attribute."""
        X, observations = two_column_stage
        fitted = AggregationConstraint(
            {'a1': None, DEFAULT_CONSTRAINT_KEY: 'sum'}, 'Y', observations, 'ctx'
        ).fit(X)

        cloned = clone(fitted)

        assert cloned.get_params()['aggregation_constraint'] == fitted.aggregation_constraint
        pd.testing.assert_frame_equal(cloned.observations, observations)
        assert not hasattr(cloned, 'anchor_mask_')

    def test_transform_before_fit_raises_not_fitted(self, two_column_stage):
        """B20: NotFittedError before fit."""
        X, observations = two_column_stage
        constraint = AggregationConstraint('sum', 'Y', observations)

        with pytest.raises(NotFittedError):
            constraint.transform(X)

    def test_fit_returns_self_and_sets_the_fitted_attributes(self, two_column_stage):
        """fit returns self, with anchor_mask_, n_features_in_, feature_names_in_."""
        X, observations = two_column_stage
        constraint = AggregationConstraint('sum', 'Y', observations)

        assert constraint.fit(X) is constraint
        assert constraint.n_features_in_ == 2
        assert list(constraint.feature_names_in_) == ['a1', 'a2']
        assert constraint.anchor_mask_.sum().to_dict() == {'a1': 1, 'a2': 1}

    def test_fit_transform_equals_fit_then_transform(self, two_column_stage):
        """fit_transform(X) == fit(X).transform(X), masks included."""
        X, observations = two_column_stage
        chained = AggregationConstraint('sum', 'Y', observations).fit(X)
        expected = chained.transform(X)

        combined = AggregationConstraint('sum', 'Y', observations)
        result = combined.fit_transform(X)

        pd.testing.assert_frame_equal(result, expected)
        pd.testing.assert_frame_equal(combined.rescaled_mask_, chained.rescaled_mask_)

    def test_transform_rescales_every_column(self, two_column_stage):
        """Two annual columns are rescaled in one pass; X is not mutated."""
        X, observations = two_column_stage
        original = X.copy()

        rescaled = AggregationConstraint('sum', 'Y', observations).fit(X).transform(X)

        assert rescaled.sum().to_dict() == pytest.approx({'a1': 120.0, 'a2': 60.0}, abs=_EXACT)
        pd.testing.assert_frame_equal(X, original)

    def test_transform_honours_the_per_column_dict(self, two_column_stage):
        """The dict form rescales a1 and leaves a2 raw."""
        X, observations = two_column_stage
        constraint = AggregationConstraint(
            {'a1': 'sum', DEFAULT_CONSTRAINT_KEY: None}, 'Y', observations
        ).fit(X)

        rescaled = constraint.transform(X)

        pd.testing.assert_series_equal(rescaled['a2'], X['a2'])
        assert constraint.rescaled_mask_.sum().to_dict() == {'a1': 12, 'a2': 0}

    def test_transform_on_a_panel_with_per_column_frequencies(self):
        """Panel stage frame: annual a1 and quarterly q1, per-column period frequencies."""
        months = _monthly_grid()
        index = pd.MultiIndex.from_product([['FR', 'DE'], months], names=['country', 'date'])
        X = pd.DataFrame({'a1': 1.0, 'q1': 1.0}, index=index)
        observations = pd.DataFrame(np.nan, index=index, columns=['a1', 'q1'])
        observations.loc[[('FR', months[-1]), ('DE', months[-1])], 'a1'] = [120.0, 24.0]
        observations.loc[[(entity, months[i]) for entity in ('FR', 'DE') for i in (2, 5, 8, 11)],
                         'q1'] = 6.0

        rescaled = AggregationConstraint(
            'sum', {'a1': 'Y', 'q1': 'Q'}, observations
        ).fit(X).transform(X)

        # Valeurs d'or : a1 → 10 par mois (FR), 2 par mois (DE) ; q1 → 6 / 3 = 2 par mois
        assert rescaled['a1'].groupby(level='country').first().to_dict() == {'DE': 2.0, 'FR': 10.0}
        assert (rescaled['q1'] == 2.0).all()

    def test_transform_is_positional_on_a_shuffled_frame(self, two_column_stage):
        """Shuffled rows: every cell gets the same value as on the sorted frame."""
        X, observations = two_column_stage
        reference = AggregationConstraint('sum', 'Y', observations).fit(X).transform(X)
        shuffled = shuffle_rows(X, seed=2)

        result = AggregationConstraint('sum', 'Y', observations).fit_transform(shuffled)

        pd.testing.assert_frame_equal(result.loc[X.index], reference, check_freq=False)

    def test_special_column_names(self, two_column_stage):
        """Spaces, accents and symbols in column names are carried through."""
        X, observations = two_column_stage
        X_special, mapping = with_special_column_names(X)
        observations_special = observations.rename(columns=mapping)
        setting = {mapping['a1']: 'sum', DEFAULT_CONSTRAINT_KEY: None}

        rescaled = AggregationConstraint(setting, 'Y', observations_special).fit_transform(X_special)

        assert rescaled[mapping['a1']].sum() == pytest.approx(120.0, abs=_EXACT)
        pd.testing.assert_series_equal(rescaled[mapping['a2']], X_special[mapping['a2']])

    def test_constraint_none_returns_raw_values_and_an_empty_mask(self, two_column_stage):
        """Under None, transform is the identity and rescaled_mask_ is all-False."""
        X, observations = two_column_stage
        constraint = AggregationConstraint(None, 'Y', observations).fit(X)

        rescaled = constraint.transform(X)

        pd.testing.assert_frame_equal(rescaled, X)
        assert not constraint.rescaled_mask_.any().any()

    def test_series_observations_serve_a_single_column(self):
        """observations may be one Series when X holds one column."""
        grid = _monthly_grid()
        X = pd.DataFrame({'a1': _RAW_2021}, index=grid)

        rescaled = AggregationConstraint(
            'sum', 'Y', _annual_observations()
        ).fit_transform(X)

        assert rescaled['a1'].sum() == pytest.approx(120.0, abs=_EXACT)

    # Métadonnées absentes ou incomplètes : erreurs de branchement levées au fit
    _METADATA_ERRORS = {
        'no-period-frequencies': (None, 'frame', 'requires period_frequencies'),
        'period-frequencies-missing-column': ({'a1': 'Y'}, 'frame',
                                              "does not cover column 'a2'"),
        'no-observations': ('Y', None, 'requires observations'),
        'observations-missing-column': ('Y', 'a1-only', "does not hold column 'a2'"),
    }

    @pytest.mark.parametrize(
        'frequencies, observations_kind, message',
        list(_METADATA_ERRORS.values()), ids=list(_METADATA_ERRORS),
    )
    def test_missing_metadata_raises_at_fit(
        self, two_column_stage, frequencies, observations_kind, message
    ):
        """Absent or incomplete metadata is a wiring error, raised at fit."""
        X, observations = two_column_stage
        observations = {
            'frame': observations, 'a1-only': observations[['a1']], None: None,
        }[observations_kind]

        with pytest.raises(ValueError, match=message):
            AggregationConstraint('sum', frequencies, observations).fit(X)

    def test_refit_follows_the_new_grid(self, two_column_stage, reference_timeseries):
        """A second fit replaces the anchor mask with the new grid's one."""
        X, observations = two_column_stage
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)
        X_2022 = pd.DataFrame(
            {'a1': 1.0, 'a2': 1.0}, index=reference_timeseries.loc['2022'].index
        )

        constraint.set_params(observations=reference_timeseries.loc['2022', ['a1', 'a2']]).fit(X_2022)

        assert constraint.anchor_mask_.index.equals(X_2022.index)

    def test_refit_purges_the_previous_rescaled_mask(self, two_column_stage):
        """A new fit drops the mask of the previous transform (B19 principle)."""
        X, observations = two_column_stage
        constraint = AggregationConstraint('sum', 'Y', observations)
        constraint.fit_transform(X)

        constraint.fit(X)

        assert not hasattr(constraint, 'rescaled_mask_')

    @pytest.mark.parametrize(
        'columns, message',
        [(['a1'], r"missing \['a2'\], unexpected \[\]"),
         (['a1', 'a2', 'q1'], r"missing \[\], unexpected \['q1'\]")],
        ids=['missing-column', 'unexpected-column'],
    )
    def test_transform_rejects_other_columns_than_fit(
        self, two_column_stage, columns, message
    ):
        """transform refuses a frame whose columns differ from those seen at fit."""
        X, observations = two_column_stage
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)
        other = X.assign(q1=1.0)[columns]

        with pytest.raises(ValueError, match=message):
            constraint.transform(other)

    def test_transform_accepts_reordered_columns(self, two_column_stage):
        """The same columns in another order are accepted, each on its own total."""
        X, observations = two_column_stage
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)

        rescaled = constraint.transform(X[['a2', 'a1']])

        assert rescaled.sum().to_dict() == pytest.approx({'a1': 120.0, 'a2': 60.0}, abs=_EXACT)
