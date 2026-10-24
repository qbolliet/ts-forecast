"""Mask and coverage accessors of ``ImputationWindowCalculator``.

Scenarios: raw and reindexed window mask, mask at a lower frequency and columns with
coverage over a query range.
"""

# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest

# Objet testé
from tsforecast.imputation._components.imputation_window import ImputationWindowCalculator

# Outils de test partagés du paquet
from tests.unit.imputation._components.imputation_window.helpers import (
    _KINDS,
    _MASK_ATTRIBUTES,
    _entities_of,
    _fit_quietly,
    _frame,
    _make_df_with_shoulders,
    _make_panel,
)


# =============================================================================
# get_imputation_window_mask
# =============================================================================
class TestGetImputationWindowMask:
    """Raw mask without ``data``, mask reindexed on ``data.index`` otherwise."""

    def test_aligned_mask_has_exactly_the_data_index(self):
        """Rows outside the fitted grid are False; strict dates are True."""
        df = _frame({'a': list(range(24)), 'b': list(range(24))})
        calc = ImputationWindowCalculator().fit(df)
        data_to_align = pd.DataFrame(index=df.index.append(pd.DatetimeIndex(['2022-06-01'])))

        aligned = calc.get_imputation_window_mask(data_to_align)

        assert (aligned.index.equals(data_to_align.index), aligned.dtype,
                bool(aligned.loc[pd.Timestamp('2022-06-01')]), bool(aligned.loc[df.index].all())) == (
            True, bool, False, True
        )

    def test_alignment_emits_no_pandas_deprecation(self):
        """Rows missing from the grid are filled without the deprecated object downcast."""
        df = _frame({'a': list(range(12)), 'b': list(range(12))})
        calc = ImputationWindowCalculator().fit(df)
        target = pd.DataFrame(index=df.index.append(pd.DatetimeIndex(['2023-01-01'])))

        with warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            aligned = calc.get_imputation_window_mask(target)

        assert (aligned.dtype, int(aligned.sum())) == (bool, 12)

    def test_raw_mask_without_data(self):
        """Without ``data``, the fitted imputation mask is returned."""
        calc = ImputationWindowCalculator().fit(_frame({'a': list(range(12)), 'b': list(range(12))}))

        assert calc.get_imputation_window_mask() is calc.imputation_window_mask_

    @pytest.mark.parametrize('kind', _KINDS)
    def test_aligned_mask_per_kind(self, kind):
        """``kind`` selects the mask that is aligned."""
        _, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(
            imputation_scope='extended_backward', training_scope='unrestricted', min_columns=2
        ).fit(df)
        attribute = dict(zip(_KINDS, _MASK_ATTRIBUTES))[kind]

        expected = getattr(calc, attribute).reindex(df.index, fill_value=False)
        pd.testing.assert_series_equal(calc.get_imputation_window_mask(df, kind=kind), expected)

    def test_invalid_kind_raises(self):
        """A wrong ``kind`` fails instead of falling back on the imputation window."""
        calc = ImputationWindowCalculator().fit(_frame({'a': list(range(12)), 'b': list(range(12))}))

        with pytest.raises(ValueError, match='kind'):
            calc.get_imputation_window_mask(kind='bogus')

    def test_aligns_panel_rows_directly(self):
        """On a panel, ``(entity, date)`` rows are aligned, including unknown ones (False)."""
        df = _make_panel([('A',), ('B',)], n_periods=12)
        calc = ImputationWindowCalculator().fit(df)
        extra = pd.MultiIndex.from_tuples([('C', pd.Timestamp('2020-01-01'))], names=df.index.names)
        target = pd.DataFrame(index=df.index.append(extra))

        aligned = calc.get_imputation_window_mask(target)

        assert (int(aligned.sum()), bool(aligned.loc[('C', pd.Timestamp('2020-01-01'))])) == (24, False)


# =============================================================================
# get_mask_at_frequency
# =============================================================================
class TestGetMaskAtFrequency:
    """A lower-frequency period is True iff all its sub-periods are in the window."""

    def test_month_to_quarter(self):
        """Strict window February-December: Q1 False, Q2-Q4 True."""
        df = _frame({'a': list(range(12)), 'b': [np.nan] + list(range(11))})
        calc = ImputationWindowCalculator().fit(df)

        quarterly = calc.get_mask_at_frequency('QS')

        assert (list(quarterly.index), quarterly.tolist()) == (
            list(pd.date_range('2020-01-01', periods=4, freq='QS')), [False, True, True, True]
        )

    def test_month_to_year_full_years(self):
        """Twelve months covered: the year is True."""
        calc = ImputationWindowCalculator().fit(_frame({'a': list(range(24)), 'b': list(range(24))}))

        assert calc.get_mask_at_frequency('YS').tolist() == [True, True]

    def test_month_to_year_partial_year(self):
        """Eleven months out of twelve: the year is False."""
        # Construction : un seul mois de 2021 hors fenêtre stricte ('b' manquante en juin)
        b = [float(i) for i in range(24)]
        b[17] = np.nan
        df = _frame({'a': list(range(24)), 'b': b})
        calc = _fit_quietly(ImputationWindowCalculator(), df)

        assert calc.get_mask_at_frequency('YS').tolist() == [True, False]

    def test_start_anchored_grid_keeps_start_anchoring(self):
        """'YE' asked on an MS grid gives a YS index, reindexable on start-anchored data."""
        calc = ImputationWindowCalculator().fit(_frame({'a': list(range(24)), 'b': list(range(24))}))

        yearly = calc.get_mask_at_frequency('YE')

        assert list(yearly.index) == [pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')]

    def test_end_anchored_grid_keeps_end_anchoring(self):
        """'QS' asked on an ME grid gives a quarter-end index."""
        df = _frame({'a': list(range(12)), 'b': list(range(12))}, start='2020-01-31', freq='ME')
        calc = ImputationWindowCalculator().fit(df)

        quarterly = calc.get_mask_at_frequency('QS')

        assert (list(quarterly.index), quarterly.tolist()) == (
            list(pd.date_range('2020-03-31', periods=4, freq='QE')), [True] * 4
        )

    def test_same_frequency_is_the_identity(self):
        """Asking for the grid's own frequency returns the mask unchanged."""
        calc = ImputationWindowCalculator().fit(_frame({'a': list(range(12)), 'b': list(range(12))}))

        pd.testing.assert_series_equal(calc.get_mask_at_frequency('MS'), calc.imputation_window_mask_)

    def test_higher_target_frequency_raises(self):
        """A daily target from a monthly mask is not an aggregation."""
        calc = ImputationWindowCalculator().fit(_frame({'a': list(range(12)), 'b': list(range(12))}))

        with pytest.raises(ValueError, match='should be higher'):
            calc.get_mask_at_frequency('D')

    def test_kind_is_propagated(self):
        """Strict, imputation and training masks give their own quarterly masks."""
        _, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(
            imputation_scope='extended_both', training_scope='strict', min_columns=2
        ).fit(df)

        strict = calc.get_mask_at_frequency('QS', kind='strict')
        imputation = calc.get_mask_at_frequency('QS', kind='imputation')
        training = calc.get_mask_at_frequency('QS', kind='training')

        # Valeur d'or : grille sept. 2020 - mai 2021 ; stricte nov. 2020 - mars 2021 (T1 2021 seul complet),
        # étendue sept. 2020 - mai 2021 (T4 2020 et T1 2021 complets)
        assert (int(strict.sum()), int(imputation.sum()), training.equals(strict)) == (1, 2, True)

    def test_kind_defaults_to_imputation(self):
        """The default ``kind`` preserves the historical behaviour."""
        _, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(imputation_scope='extended_both', min_columns=2).fit(df)

        pd.testing.assert_series_equal(
            calc.get_mask_at_frequency('QS'), calc.get_mask_at_frequency('QS', kind='imputation')
        )

    def test_invalid_kind_raises(self):
        """A wrong ``kind`` is rejected here too."""
        calc = ImputationWindowCalculator().fit(_frame({'a': list(range(12)), 'b': list(range(12))}))

        with pytest.raises(ValueError, match='kind'):
            calc.get_mask_at_frequency('QS', kind='bogus')

    def test_panel_result_is_a_multiindex_series(self):
        """On a panel: one Series on ``(entity..., date)``, start-anchored like the grid."""
        df = _make_panel([('A', 'x'), ('B', 'y')])
        calc = ImputationWindowCalculator().fit(df)

        yearly = calc.get_mask_at_frequency('YS')

        assert (isinstance(yearly.index, pd.MultiIndex), yearly.dtype, _entities_of(yearly),
                list(yearly.index.names), yearly.tolist()) == (
            True, bool, {('A', 'x'), ('B', 'y')}, list(df.index.names), [True] * 6
        )

    def test_scalar_and_tuple_keys_are_equivalent(self):
        """Scalar entity keys are normalized to tuples at the public boundary."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))

        pd.testing.assert_series_equal(
            calc.get_mask_at_frequency({'A': 'QS', 'B': 'YS'}),
            calc.get_mask_at_frequency({('A',): 'QS', ('B',): 'YS'}),
        )

    def test_per_entity_targets(self):
        """A dict target converts each entity to its own frequency."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))

        result = calc.get_mask_at_frequency({('A',): 'QS', ('B',): 'YS'})

        # Valeur d'or : 36 mois = 12 trimestres pour A, 3 années pour B
        assert result.groupby(level=0).size().to_dict() == {'A': 12, 'B': 3}

    def test_per_entity_targets_on_the_heterogeneous_panel(self, heterogeneous_coverage_panel):
        """Strict window at a frequency per entity, golden counts by hand.

        FR strict 2018-06→2023-12: complete quarters 2018-Q3→2023-Q4 (22) on a
        grid 2015-01→2024-07 (39 quarters); DE 2019-01→2023-12: 5 years on 9;
        IT 2019-06→2023-12: 2019-Q3→2023-Q4 (18) on 35 quarters.
        """
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        result = calc.get_mask_at_frequency(
            {'France': 'QS', 'Allemagne': 'YS', 'Italie': 'QS'}, kind='strict'
        )

        assert result.groupby(level=0).agg(['sum', 'size']).to_dict('index') == {
            'Allemagne': {'sum': 5, 'size': 9},
            'France': {'sum': 22, 'size': 39},
            'Italie': {'sum': 18, 'size': 35},
        }

    def test_year_end_on_the_heterogeneous_panel(self, heterogeneous_coverage_panel):
        """Italie, strict window at 'YE': 2019 (from June) and 2024 incomplete, 2020-2023 complete."""
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        italy = calc.get_mask_at_frequency('YE', kind='strict').xs('Italie').loc['2019':]

        assert italy.tolist() == [False, True, True, True, True, False]


# =============================================================================
# get_columns_with_coverage
# =============================================================================
class TestGetColumnsWithCoverage:
    """Columns whose coverage range overlaps the query range."""

    @staticmethod
    def _monthly_and_quarterly() -> pd.DataFrame:
        """``a`` monthly over 2020, ``q`` quarterly for Q1-Q3 (covers January to September)."""
        q = [1.0, np.nan, np.nan, 1.0, np.nan, np.nan, 1.0] + [np.nan] * 5
        return _frame({'a': list(range(12)), 'q': q})

    @pytest.mark.parametrize(
        'start, end, expected',
        [
            pytest.param('2020-01-01', '2020-03-01', ['a', 'q'], id='both_covered'),
            pytest.param('2020-10-01', '2020-12-01', ['a'], id='after_quarterly_coverage'),
            pytest.param('2020-09-01', '2020-10-01', ['a', 'q'], id='overlap_on_one_month'),
            pytest.param('2021-02-01', '2021-03-01', [], id='outside_everything'),
        ],
    )
    def test_time_series(self, start, end, expected):
        """Overlap test ``col_start <= end and col_end >= start`` on a time series."""
        calc = ImputationWindowCalculator().fit(self._monthly_and_quarterly())

        assert calc.get_columns_with_coverage(pd.Timestamp(start), pd.Timestamp(end)) == expected

    def test_panel_with_entity_returns_a_list(self):
        """With an entity: the list of that entity's covered columns."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))

        assert calc.get_columns_with_coverage(
            pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01'), entity=('A',)
        ) == ['a', 'b']

    def test_panel_scalar_entity_is_normalized(self):
        """A scalar entity is accepted and normalized to its tuple."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))
        start, end = pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01')

        assert calc.get_columns_with_coverage(start, end, entity='A') == calc.get_columns_with_coverage(
            start, end, entity=('A',)
        )

    def test_panel_without_entity_returns_a_dict(self):
        """Without an entity: a dict over every entity tuple."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A', 'x'), ('B', 'y')]))

        assert calc.get_columns_with_coverage(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01')) == {
            ('A', 'x'): ['a', 'b'], ('B', 'y'): ['a', 'b'],
        }

    def test_unknown_entity_raises(self):
        """An unknown key is a KeyError, not a silent empty list."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))

        with pytest.raises(KeyError):
            calc.get_columns_with_coverage(pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01'), entity=('ZZ',))

    def test_unobserved_column_is_never_covered(self, heterogeneous_coverage_panel):
        """Italie, early 2024: annual series ended in 2023, climat_affaires never observed."""
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        assert calc.get_columns_with_coverage(
            pd.Timestamp('2024-01-01'), pd.Timestamp('2024-03-01'), entity='Italie'
        ) == ['production_industrielle', 'inflation_ipc', 'taux_chomage', 'pib_trimestriel']
