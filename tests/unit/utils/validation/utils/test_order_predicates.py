"""Panel order predicates of ``tsforecast.utils.validation``.

Scenarios: ``validate_entities_grouped``, ``validate_sorted_within_groups`` and the
repair of panel order by ``validate_temporal_data``.
"""

import numpy as np
import pandas as pd
import pytest

# Jeux de données et perturbations partagés
from tests.support import perturbations as perturb

# Fonctions à tester
from tsforecast.utils.validation import (
    validate_entities_grouped,
    validate_sorted_within_groups,
    validate_temporal_data,
)

# Outils de test partagés du paquet
from tests.unit.utils.validation.utils.helpers import (
    _flat,
    _months,
    _panel,
    _validate_kwargs,
)


# Le chemin « colonnes » émet systématiquement un avertissement de remplacement
# d'index (comportement testé explicitement dans TestColumnBasedValidation) ; le
# bruit est masqué ailleurs pour garder une sortie lisible. L'inférence de format
# de pandas sur des chaînes de dates non ISO émet aussi un avertissement.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Index replaced with"),
    pytest.mark.filterwarnings("ignore:Could not infer format"),
]


# =============================================================================
# validate_entities_grouped
# =============================================================================
class TestValidateEntitiesGrouped:
    """Contiguity of each entity's observations."""

    def test_non_pandas_input_is_rejected(self):
        """A non Series / DataFrame input raises ``ValueError``."""
        with pytest.raises(ValueError, match='pandas Series or DataFrame'):
            validate_entities_grouped([1, 2, 3])

    @pytest.mark.parametrize('container', ['dataframe', 'series'])
    def test_contiguous_entities_are_grouped(self, container):
        """Entities in contiguous blocks give ``True``."""
        panel = _panel()
        data = panel if container == 'dataframe' else panel['v']
        assert validate_entities_grouped(data) is True

    @pytest.mark.parametrize('container', ['dataframe', 'series'])
    def test_interleaved_entities_are_not_grouped(self, container):
        """``A, B, A, B, A, B`` gives ``False``."""
        index = pd.MultiIndex.from_arrays(
            [['A', 'B'] * 3, _months(6)], names=['entity', 'date']
        )
        data = pd.DataFrame({'v': range(6)}, index=index)
        assert validate_entities_grouped(data if container == 'dataframe' else data['v']) is False

    def test_an_entity_reappearing_after_another_is_not_grouped(self):
        """``A, A, B, A`` gives ``False`` (a single reappearance is enough)."""
        index = pd.MultiIndex.from_arrays([['A', 'A', 'B', 'A'], _months(4)])
        assert validate_entities_grouped(pd.DataFrame({'v': range(4)}, index=index)) is False

    def test_grouping_ignores_the_date_order(self):
        """Entities grouped but dates unsorted inside a block: still ``True``."""
        panel = _panel()
        unsorted_within = panel.iloc[[2, 0, 1, 5, 3, 4]]
        assert validate_entities_grouped(unsorted_within) is True

    def test_entity_blocks_may_come_in_any_order(self):
        """Grouping does not require alphabetical order of the blocks."""
        assert validate_entities_grouped(perturb.reverse_entities(_panel())) is True

    @pytest.mark.parametrize('rows', [0, 1], ids=['empty', 'single-row'])
    def test_empty_and_single_row_panels_are_grouped(self, rows):
        """Edge cases: no row, or one row, is trivially grouped."""
        assert validate_entities_grouped(_panel().iloc[:rows]) is True

    def test_single_entity_is_grouped(self):
        """A panel with one entity is grouped."""
        assert validate_entities_grouped(_panel(entities=('A',))) is True

    def test_three_level_panel_is_grouped_by_the_full_entity_key(self):
        """On 3 levels the entity is the ``(region, country)`` pair, date excluded."""
        panel = perturb.to_three_level_index(_panel(entities=('FR', 'DE')))
        assert validate_entities_grouped(panel) is True

    def test_three_level_panel_with_interleaved_countries_is_not_grouped(self):
        """Same region for both countries, but the rows alternate: ``False``."""
        panel = perturb.to_three_level_index(_panel(entities=('FR', 'DE')))
        interleaved = panel.iloc[[0, 3, 1, 4, 2, 5]]
        assert validate_entities_grouped(interleaved) is False

    def test_missing_entity_labels_are_one_group(self):
        """``NaN`` entity labels are treated as one entity: contiguous is ``True``, split is ``False`` (pinned)."""
        dates = _months(4)
        grouped = pd.MultiIndex.from_arrays([['A', 'A', np.nan, np.nan], dates])
        split = pd.MultiIndex.from_arrays([[np.nan, 'A', np.nan, 'A'], dates])
        frame = pd.DataFrame({'v': range(4)})
        assert validate_entities_grouped(frame.set_axis(grouped)) is True
        assert validate_entities_grouped(frame.set_axis(split)) is False

    def test_panel_col_path(self):
        """With one panel column, contiguity is read from that column."""
        flat = _flat(_panel())
        assert validate_entities_grouped(flat, panel_cols=['entity']) is True
        interleaved = flat.iloc[[0, 3, 1, 4, 2, 5]]
        assert validate_entities_grouped(interleaved, panel_cols=['entity']) is False

    def test_several_panel_cols_are_read_as_one_composite_key(self):
        """With two panel columns the entity is the pair of values."""
        flat = _flat(perturb.to_three_level_index(_panel(entities=('FR', 'DE'))))
        assert validate_entities_grouped(flat, panel_cols=['region', 'entity']) is True
        assert validate_entities_grouped(flat.iloc[[0, 3, 1, 4, 2, 5]], panel_cols=['region', 'entity']) is False

    def test_missing_panel_col_is_rejected(self):
        """An unknown panel column raises with its name."""
        with pytest.raises(ValueError, match='Panel columns not found in data'):
            validate_entities_grouped(_flat(_panel()), panel_cols=['nope'])

    def test_ambiguous_panel_col_is_rejected(self):
        """A duplicated panel column label designates two columns: no entity to read."""
        df = pd.DataFrame([['A', 1, 'A']], columns=['entity', 'v', 'entity'])
        with pytest.raises(ValueError, match="Column 'entity' is not unique"):
            validate_entities_grouped(df, panel_cols=['entity'])

    def test_series_with_panel_cols_is_rejected(self):
        """``panel_cols`` cannot be used with a ``Series`` without ``MultiIndex``."""
        s = pd.Series([1, 2], index=_months(2))
        with pytest.raises(ValueError, match='Cannot use panel_cols with Series'):
            validate_entities_grouped(s, panel_cols=['entity'])

    def test_plain_time_series_without_panel_cols_is_rejected(self):
        """Neither ``MultiIndex`` nor ``panel_cols``: nothing to group, ``ValueError``."""
        s = pd.Series([1, 2], index=_months(2))
        with pytest.raises(ValueError, match='MultiIndex or panel_cols must be specified'):
            validate_entities_grouped(s)

    def test_panel_cols_are_ignored_for_a_multiindex(self):
        """For a ``MultiIndex`` the ``panel_cols`` argument is ignored, even if bogus."""
        assert validate_entities_grouped(_panel(), panel_cols=['does_not_exist']) is True

    def test_shuffled_realistic_panel_is_not_grouped(self, heterogeneous_coverage_panel):
        """The notebook 3 panel: grouped as built, not grouped once shuffled."""
        assert validate_entities_grouped(heterogeneous_coverage_panel) is True
        shuffled = perturb.shuffle_rows(heterogeneous_coverage_panel, seed=0)
        assert validate_entities_grouped(shuffled) is False

    def test_reversed_realistic_panel_is_still_grouped(self, heterogeneous_coverage_panel):
        """Reversing the entity blocks keeps each entity contiguous."""
        reversed_panel = perturb.reverse_entities(heterogeneous_coverage_panel)
        assert validate_entities_grouped(reversed_panel) is True


# =============================================================================
# validate_sorted_within_groups
# =============================================================================
class TestValidateSortedWithinGroups:
    """Chronological order of the dates inside each entity."""

    def test_non_pandas_input_is_rejected(self):
        """A non Series / DataFrame input raises ``ValueError``."""
        with pytest.raises(ValueError, match='pandas Series or DataFrame'):
            validate_sorted_within_groups([1, 2, 3])

    @pytest.mark.parametrize('container', ['dataframe', 'series'])
    def test_sorted_panel(self, container):
        """Dates increasing inside each entity give ``True``."""
        panel = _panel()
        assert validate_sorted_within_groups(panel if container == 'dataframe' else panel['v']) is True

    def test_unsorted_dates_inside_an_entity(self):
        """Golden case: entity ``B`` dates ``03, 01, 02`` give ``False``."""
        dates = pd.to_datetime(
            ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-03', '2023-01-01', '2023-01-02']
        )
        index = pd.MultiIndex.from_arrays([['A'] * 3 + ['B'] * 3, dates])
        assert validate_sorted_within_groups(pd.DataFrame({'v': range(6)}, index=index)) is False

    def test_interleaved_entities_with_sorted_dates_are_sorted(self):
        """Order is checked per entity: interleaving the entities does not matter."""
        panel = _panel()
        interleaved = panel.sort_index(level=[1, 0])
        assert validate_entities_grouped(interleaved) is False
        assert validate_sorted_within_groups(interleaved) is True

    def test_equal_dates_inside_an_entity_are_accepted(self):
        """Monotone means non-decreasing: a repeated date is not a disorder (pinned)."""
        dates = pd.to_datetime(['2023-01-01', '2023-01-01', '2023-02-01'])
        index = pd.MultiIndex.from_arrays([['A', 'A', 'A'], dates])
        assert validate_sorted_within_groups(pd.DataFrame({'v': range(3)}, index=index)) is True

    @pytest.mark.parametrize('rows', [0, 1], ids=['empty', 'single-row'])
    def test_empty_and_single_row_panels_are_sorted(self, rows):
        """Edge cases: no row, or one row, is trivially sorted."""
        assert validate_sorted_within_groups(_panel().iloc[:rows]) is True

    def test_three_level_panel_sorted_by_full_entity_key(self):
        """On 3 levels the groups are ``(region, country)``: dates restart per country."""
        panel = perturb.to_three_level_index(_panel(entities=('FR', 'DE')))
        # Deux pays d'une même région : les dates redémarrent au changement de pays,
        # ce qui ne doit pas être pris pour un désordre.
        assert validate_sorted_within_groups(panel) is True

    def test_three_level_panel_detects_disorder_in_one_country(self):
        """Reversing the dates of one country of a 3-level panel gives ``False``."""
        panel = perturb.to_three_level_index(_panel(entities=('FR', 'DE')))
        disordered = panel.iloc[[0, 1, 2, 5, 4, 3]]
        assert validate_sorted_within_groups(disordered) is False

    def test_plain_series_with_sorted_datetime_index(self):
        """Simple time series: ``True`` when the index is increasing."""
        assert validate_sorted_within_groups(pd.Series([1, 2, 3], index=_months(3))) is True

    def test_plain_series_with_unsorted_datetime_index(self):
        """Simple time series: ``False`` when the index is not increasing."""
        s = pd.Series([1, 2, 3], index=_months(3)[[1, 0, 2]])
        assert validate_sorted_within_groups(s) is False

    def test_plain_series_with_period_index(self):
        """A ``PeriodIndex`` is a temporal index too: sorted ``True``, unsorted ``False``."""
        periods = pd.period_range('2023-01', periods=3, freq='M')
        assert validate_sorted_within_groups(pd.Series([1, 2, 3], index=periods)) is True
        assert validate_sorted_within_groups(pd.Series([1, 2, 3], index=periods[[1, 0, 2]])) is False

    def test_panel_with_period_dates(self):
        """Period dates on the last level of a ``MultiIndex`` are compared as periods."""
        periods = pd.period_range('2023-01', periods=3, freq='M')
        index = pd.MultiIndex.from_product([['A', 'B'], periods])
        panel = pd.DataFrame({'v': range(6)}, index=index)
        assert validate_sorted_within_groups(panel) is True
        assert validate_sorted_within_groups(panel.iloc[[1, 0, 2, 3, 4, 5]]) is False

    def test_plain_series_without_temporal_index_is_rejected(self):
        """Only a ``DatetimeIndex`` or ``PeriodIndex`` can be validated."""
        with pytest.raises(ValueError, match='must have DatetimeIndex or PeriodIndex'):
            validate_sorted_within_groups(pd.Series([1, 2, 3], index=pd.RangeIndex(3)))

    def test_panel_col_and_time_col_path(self):
        """Column path: sorted gives ``True``, unsorted dates give ``False``."""
        flat = _flat(_panel())
        assert validate_sorted_within_groups(flat, panel_cols=['entity'], time_col='date') is True
        disordered = flat.iloc[[1, 0, 2, 3, 4, 5]]
        assert validate_sorted_within_groups(disordered, panel_cols=['entity'], time_col='date') is False

    def test_several_panel_cols_on_the_column_path(self):
        """Two panel columns form the group key on the column path."""
        flat = _flat(perturb.to_three_level_index(_panel(entities=('FR', 'DE'))))
        assert validate_sorted_within_groups(flat, panel_cols=['region', 'entity'], time_col='date') is True
        disordered = flat.iloc[[0, 1, 2, 5, 4, 3]]
        assert validate_sorted_within_groups(
            disordered, panel_cols=['region', 'entity'], time_col='date'
        ) is False

    def test_missing_panel_col_is_rejected(self):
        """An unknown panel column raises with its name."""
        with pytest.raises(ValueError, match='Panel columns not found in data'):
            validate_sorted_within_groups(_flat(_panel()), panel_cols=['nope'], time_col='date')

    def test_missing_time_col_is_rejected(self):
        """An unknown time column raises with its name."""
        with pytest.raises(ValueError, match="Time column 'nope' not found in data"):
            validate_sorted_within_groups(_flat(_panel()), panel_cols=['entity'], time_col='nope')

    def test_series_with_panel_and_time_cols_is_rejected(self):
        """Column names make no sense on a ``Series`` without ``MultiIndex``."""
        s = pd.Series([1, 2], index=_months(2))
        with pytest.raises(ValueError, match='Cannot use panel_cols and time_col with Series'):
            validate_sorted_within_groups(s, panel_cols=['a'], time_col='b')

    @pytest.mark.parametrize(
        'kwargs',
        [{'panel_cols': ['entity']}, {'time_col': 'date'}],
        ids=['panel-cols-only', 'time-col-only'],
    )
    def test_incomplete_column_arguments_are_rejected(self, kwargs):
        """One of ``panel_cols`` / ``time_col`` without the other is an error (pinned).

        A single time series stored with a ``time_col`` column (no panel)
        cannot be checked either: the function has no single-series column path.
        """
        with pytest.raises(ValueError, match='both panel_cols and time_col must be specified'):
            validate_sorted_within_groups(_flat(_panel()), **kwargs)

    def test_incomparable_dates_are_not_sorted(self):
        """A group mixing a string and an integer date cannot be ordered: ``False`` (pinned).

        ``is_monotonic_increasing`` answers ``False`` on values that cannot be
        compared; no exception reaches the function.
        """
        df = pd.DataFrame({'entity': ['A', 'A'], 'date': ['2023-01-01', 5]})
        assert validate_sorted_within_groups(df, panel_cols=['entity'], time_col='date') is False

    def test_ambiguous_time_col_is_rejected(self):
        """A duplicated ``time_col`` label designates two columns: ``ValueError``, not ``False``.

        Edge case: column names that are not unique. The check must not fail inside a
        blanket ``except`` and answer ``False`` ("not sorted"), which would hide the misuse.
        """
        dates = _months(2)
        df = pd.DataFrame([['A', dates[0], dates[0]], ['A', dates[1], dates[1]]],
                          columns=['entity', 'date', 'date'])
        with pytest.raises(ValueError, match="Column 'date' is not unique"):
            validate_sorted_within_groups(df, panel_cols=['entity'], time_col='date')

    def test_shuffled_realistic_panel_is_not_sorted(self, heterogeneous_coverage_panel):
        """The notebook 3 panel: sorted as built, not sorted once shuffled."""
        assert validate_sorted_within_groups(heterogeneous_coverage_panel) is True
        shuffled = perturb.shuffle_rows(heterogeneous_coverage_panel, seed=0)
        assert validate_sorted_within_groups(shuffled) is False

    def test_realistic_irregular_series_is_sorted(self, irregular_index_timeseries):
        """The irregular notebook 3 series has an increasing ``DatetimeIndex``."""
        assert validate_sorted_within_groups(irregular_index_timeseries) is True


# =============================================================================
# Articulation avec validate_temporal_data : strict True / False et tri
# =============================================================================
def _layouts(panel: pd.DataFrame) -> dict:
    """Build the three unordered layouts of a sorted panel, keyed by readable id.

    Args:
        panel: Sorted two-level panel, at least two entities.

    Returns:
        ``interleaved`` (dates globally ordered, entities alternating: not
        grouped, sorted per entity), ``unsorted_within_entity`` (entities
        contiguous, dates shuffled inside each block: grouped, not sorted) and
        ``shuffled`` (neither).
    """
    within = panel.groupby(level=0, group_keys=False, sort=False).apply(
        lambda block: perturb.shuffle_rows(block, seed=0)
    )
    return {
        'interleaved': panel.sort_index(level=[1, 0]),
        'unsorted_within_entity': within,
        'shuffled': perturb.shuffle_rows(panel, seed=0),
    }


class TestValidationRepairsPanelOrder:
    """``validate_temporal_data`` (sorting) makes both order predicates true.

    ``validate_entities_grouped`` and ``validate_sorted_within_groups`` take no
    ``strict`` argument: they only report. ``strict`` is a property of
    ``validate_temporal_data``, which sorts whatever ``strict`` is; the
    predicates then turn ``True``. Without sorting they stay as they were.
    """

    @pytest.mark.parametrize(
        'layout, grouped, sorted_within',
        [
            ('interleaved', False, True),
            ('unsorted_within_entity', True, False),
            ('shuffled', False, False),
        ],
    )
    def test_layouts_are_reported_as_expected(self, layout, grouped, sorted_within):
        """The three layouts give the golden ``(grouped, sorted)`` pairs."""
        panel = _layouts(_panel(entities=('A', 'B', 'C'), periods=6))[layout]
        assert (validate_entities_grouped(panel), validate_sorted_within_groups(panel)) == (
            grouped,
            sorted_within,
        )

    @pytest.mark.parametrize('layout', ['interleaved', 'unsorted_within_entity', 'shuffled'])
    @pytest.mark.parametrize('strict', [True, False], ids=['strict', 'not-strict'])
    @pytest.mark.parametrize('path', ['index', 'columns'])
    def test_validation_with_sort_repairs_the_order(self, layout, strict, path):
        """After ``validate_temporal_data(sort_data=True)`` both predicates are ``True``."""
        panel = _layouts(_panel(entities=('A', 'B', 'C'), periods=6))[layout]
        data, kwargs = _validate_kwargs(panel, path)

        out = validate_temporal_data(data, strict=strict, sort_data=True, **kwargs)

        assert validate_entities_grouped(out) is True
        assert validate_sorted_within_groups(out) is True

    @pytest.mark.parametrize('layout', ['interleaved', 'unsorted_within_entity', 'shuffled'])
    def test_validation_without_sort_leaves_the_order_as_it_was(self, layout):
        """With ``sort_data=False`` the reported order is unchanged."""
        panel = _layouts(_panel(entities=('A', 'B', 'C'), periods=6))[layout]
        before = (validate_entities_grouped(panel), validate_sorted_within_groups(panel))

        out = validate_temporal_data(panel, sort_data=False)

        assert (validate_entities_grouped(out), validate_sorted_within_groups(out)) == before

    def test_shuffled_realistic_panel_is_repaired(self, heterogeneous_coverage_panel):
        """The shuffled notebook 3 panel becomes grouped and sorted after validation."""
        shuffled = perturb.shuffle_rows(heterogeneous_coverage_panel, seed=3)
        out = validate_temporal_data(shuffled)
        assert validate_entities_grouped(out) is True
        assert validate_sorted_within_groups(out) is True

    def test_column_path_predicates_on_the_flat_realistic_panel(self, heterogeneous_coverage_panel):
        """The predicates accept the flat notebook 3 panel through ``panel_cols`` / ``time_col``."""
        flat = perturb.shuffle_rows(heterogeneous_coverage_panel, seed=4).reset_index()
        assert validate_entities_grouped(flat, panel_cols=['country']) is False
        assert validate_sorted_within_groups(flat, panel_cols=['country'], time_col='date') is False
        repaired = validate_temporal_data(flat, time_col='date', panel_cols=['country']).reset_index()
        assert validate_entities_grouped(repaired, panel_cols=['country']) is True
        assert validate_sorted_within_groups(repaired, panel_cols=['country'], time_col='date') is True
