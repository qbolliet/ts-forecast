"""Unit tests for the panel-structure helpers of ``tsforecast.panel.utils``.

Covers ``is_panel_data``, ``get_entity_levels``, ``detect_panel_structure``,
``extract_time_series_from_multiindex`` and ``get_unique_panel_entities`` on simple,
two- and three-level indexes, named or unnamed levels, ``MultiIndex`` columns, empty
and single-entity frames, plus the public exports of the module.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

import tsforecast.panel as panel_package
import tsforecast.panel.utils as panel_utils
from tests.support.datasets import HETEROGENEOUS_PANEL_COUNTRIES
from tests.support.perturbations import (
    empty_like,
    interleave_entities,
    rename_entities,
    reverse_entities,
    to_three_level_index,
    with_duplicated_rows,
)
from tsforecast.panel.utils import (
    detect_panel_structure,
    extract_time_series_from_multiindex,
    get_entity_levels,
    get_unique_panel_entities,
    is_panel_data,
    iter_entity_blocks,
)

# Fonctions définies dans panel.utils sans préfixe « _ » : l'interface publique du module
PUBLIC_FUNCTIONS = sorted(
    name for name, obj in vars(panel_utils).items()
    if inspect.isfunction(obj) and obj.__module__ == panel_utils.__name__ and not name.startswith('_')
)


class TestIsPanelData:
    """``is_panel_data``: a ``MultiIndex`` of at least two levels is a panel."""

    @pytest.mark.parametrize(
        'index, expected',
        [
            pytest.param(pd.date_range('2023-01-01', periods=3), False, id='datetime_index'),
            pytest.param(pd.RangeIndex(3), False, id='range_index'),
            pytest.param(pd.period_range('2023-01', periods=3, freq='M'), False, id='period_index'),
            pytest.param(pd.Index(['a', 'b', 'c']), False, id='string_index'),
            pytest.param(pd.MultiIndex.from_arrays([[1, 2, 3]]), False, id='one_level_multiindex'),
            pytest.param(pd.MultiIndex.from_product([['FR'], range(3)]), True, id='two_levels'),
            pytest.param(pd.MultiIndex.from_product([['EU'], ['FR'], range(3)]), True,
                         id='three_levels'),
            pytest.param(pd.MultiIndex.from_product([['a'], ['b'], ['c'], range(3)]), True,
                         id='four_levels'),
        ],
    )
    def test_depends_on_the_index_only(self, index, expected):
        """Series and DataFrame alike, the answer depends on the number of index levels."""
        assert is_panel_data(pd.Series(range(len(index)), index=index)) is expected
        assert is_panel_data(pd.DataFrame({'x': range(len(index))}, index=index)) is expected

    def test_docstring_examples(self):
        """The two examples of the docstring."""
        idx = pd.MultiIndex.from_tuples([('FR', '2023-01'), ('FR', '2023-02')], names=['entity', 'date'])
        assert is_panel_data(pd.Series([1, 2], index=idx)) is True
        assert is_panel_data(pd.Series([1, 2], index=pd.date_range('2023', periods=2))) is False

    def test_unnamed_levels(self):
        """Level names are irrelevant."""
        index = pd.MultiIndex.from_product([['FR', 'DE'], range(2)])
        assert is_panel_data(pd.Series(range(4), index=index)) is True

    def test_empty_panel_is_still_a_panel(self, panel_2l):
        """A frame without rows keeps the structure of its index."""
        assert is_panel_data(empty_like(panel_2l)) is True

    def test_one_entity_is_still_a_panel(self, panel_2l):
        """The structure is that of the index, not the number of entities."""
        assert is_panel_data(panel_2l.loc[['FR']]) is True

    def test_multiindex_columns_do_not_make_a_panel(self, dates):
        """Only the row index counts: ``MultiIndex`` columns on a time series is no panel."""
        columns = pd.MultiIndex.from_tuples([('a', 'x'), ('a', 'y')])
        assert is_panel_data(pd.DataFrame(np.zeros((3, 2)), index=dates, columns=columns)) is False

    def test_multiindex_columns_on_a_panel(self, panel_2l):
        """A panel index with ``MultiIndex`` columns is a panel."""
        columns = pd.MultiIndex.from_tuples([('a', 'x')])
        assert is_panel_data(pd.DataFrame({columns[0]: panel_2l['x']})) is True

    def test_realistic_datasets(self, heterogeneous_coverage_panel, irregular_index_timeseries):
        """The realistic panel is a panel, the irregular time series is not."""
        assert (is_panel_data(heterogeneous_coverage_panel), is_panel_data(irregular_index_timeseries)) == (
            True, False)


class TestGetEntityLevels:
    """``get_entity_levels``: every level but the last is an entity level."""

    @pytest.mark.parametrize(
        'index, expected',
        [
            pytest.param(pd.date_range('2023-01-01', periods=3), [], id='time_series'),
            pytest.param(pd.MultiIndex.from_arrays([[1, 2, 3]]), [], id='one_level_multiindex'),
            pytest.param(pd.MultiIndex.from_product([['FR'], range(3)]), [0], id='two_levels'),
            pytest.param(pd.MultiIndex.from_product([['EU'], ['FR'], range(3)]), [0, 1],
                         id='three_levels'),
            pytest.param(pd.MultiIndex.from_product([['a'], ['b'], ['c'], range(3)]), [0, 1, 2],
                         id='four_levels'),
        ],
    )
    def test_positions_of_the_entity_levels(self, index, expected):
        """The positions are ``0 .. nlevels - 2``, in order."""
        assert get_entity_levels(pd.Series(range(len(index)), index=index)) == expected

    def test_docstring_examples(self):
        """The two examples of the docstring."""
        idx = pd.MultiIndex.from_tuples(
            [('FR', 'a', '2023-01'), ('FR', 'b', '2023-01')], names=['c', 's', 'date']
        )
        assert get_entity_levels(pd.Series([1, 2], index=idx)) == [0, 1]
        assert get_entity_levels(pd.Series([1, 2], index=pd.date_range('2023', periods=2))) == []

    def test_time_series_has_no_entity_level(self, timeseries):
        """A time series has no entity level."""
        assert get_entity_levels(timeseries) == []

    def test_one_entity_level(self, panel_2l):
        """A two-level panel has the single entity level 0."""
        assert get_entity_levels(panel_2l) == [0]

    def test_several_entity_levels(self, panel_3l):
        """A three-level panel has the entity levels 0 and 1."""
        assert get_entity_levels(panel_3l['x']) == [0, 1]

    def test_series_and_dataframe_agree(self, panel_3l):
        """A Series and a DataFrame with the same index give the same levels."""
        assert get_entity_levels(panel_3l['x']) == get_entity_levels(panel_3l)

    def test_names_do_not_matter(self, panel_3l):
        """Unnamed levels give the same positions."""
        unnamed = panel_3l.copy()
        unnamed.index = unnamed.index.set_names([None, None, None])
        assert get_entity_levels(unnamed) == [0, 1]

    def test_empty_panel(self, panel_2l):
        """An empty panel keeps its entity levels."""
        assert get_entity_levels(empty_like(panel_2l)) == [0]

    def test_is_empty_exactly_when_the_data_is_not_a_panel(self):
        """The list of levels is empty exactly when the data is not a panel."""
        for index in (pd.RangeIndex(3), pd.date_range('2023-01-01', periods=3),
                      pd.MultiIndex.from_product([['FR'], pd.date_range('2023-01-01', periods=3)])):
            data = pd.Series(range(len(index)), index=index)
            assert bool(get_entity_levels(data)) == is_panel_data(data)

    def test_levels_drop_to_a_date_only_block(self, panel_3l):
        """Dropping the entity levels of a block leaves the date level alone."""
        for _, _, block in iter_entity_blocks(panel_3l):
            assert block.index.nlevels == panel_3l.index.nlevels - len(get_entity_levels(panel_3l))


class TestDetectPanelStructure:
    """``detect_panel_structure`` reads the panel columns from the index or from the argument."""

    def test_panel_columns_are_the_index_levels_but_the_last(self, panel_2l):
        """Automatic detection on a named two-level index."""
        assert detect_panel_structure(panel_2l, None) == (['entity'], True)

    def test_three_level_index(self, panel_3l):
        """Both entity levels are reported, the date level is not."""
        assert detect_panel_structure(panel_3l, None) == (['region', 'country'], True)

    def test_unnamed_levels_are_reported_as_none(self, panel_3l):
        """Unnamed levels come back as ``None`` names, one per entity level."""
        unnamed = panel_3l.copy()
        unnamed.index = unnamed.index.set_names([None, None, None])
        assert detect_panel_structure(unnamed, None) == ([None, None], True)

    def test_series_with_a_panel_index(self, panel_2l):
        """The structure is read from the index, so a Series works like a DataFrame."""
        assert detect_panel_structure(panel_2l['x'], None) == (['entity'], True)

    @pytest.mark.parametrize(
        'index',
        [
            pytest.param(pd.date_range('2023-01-01', periods=3), id='datetime_index'),
            pytest.param(pd.RangeIndex(3), id='range_index'),
            pytest.param(pd.MultiIndex.from_arrays([[1, 2, 3]], names=['only']), id='one_level_multiindex'),
        ],
    )
    def test_no_panel_without_specified_columns(self, index):
        """A simple index and no ``panel_cols``: there is no panel."""
        assert detect_panel_structure(pd.DataFrame({'x': [1, 2, 3]}, index=index), None) == (None, False)

    def test_multiindex_columns_on_a_time_series(self, dates):
        """``MultiIndex`` columns are not panel columns."""
        columns = pd.MultiIndex.from_tuples([('a', 'x'), ('a', 'y')])
        df = pd.DataFrame(np.zeros((3, 2)), index=dates, columns=columns)
        assert detect_panel_structure(df, None) == (None, False)

    def test_specified_columns_found_in_the_index(self, panel_3l):
        """Names given by the caller and present among the index names: panel in the index."""
        assert detect_panel_structure(panel_3l, ['region', 'country']) == (['region', 'country'], True)

    def test_specified_columns_are_returned_as_given(self, panel_2l):
        """The caller's order is kept, the date level name is accepted like any other."""
        assert detect_panel_structure(panel_2l, ['date', 'entity']) == (['date', 'entity'], True)

    def test_specified_columns_missing_from_the_index(self, panel_2l):
        """Names absent from the index: the panel is assumed to be in the columns."""
        assert detect_panel_structure(panel_2l, ['country']) == (['country'], False)

    def test_specified_columns_partly_in_the_index(self, panel_2l):
        """A single missing name is enough to place the panel in the columns."""
        assert detect_panel_structure(panel_2l, ['entity', 'country']) == (['entity', 'country'], False)

    def test_specified_columns_on_a_simple_index(self, timeseries):
        """Panel in columns: the names are returned, the index is not a panel."""
        assert detect_panel_structure(timeseries, ['entity']) == (['entity'], False)

    def test_columns_are_not_checked_against_the_frame(self, timeseries):
        """The function does not look at the columns of the frame: it only reads the index."""
        assert 'nowhere' not in timeseries.columns
        assert detect_panel_structure(timeseries, ['nowhere']) == (['nowhere'], False)

    def test_empty_list_of_columns_counts_as_found_in_the_index(self, panel_2l):
        """With no name to look for, nothing is missing from the index."""
        assert detect_panel_structure(panel_2l, []) == ([], True)

    def test_input_is_not_modified(self, panel_2l):
        """Neither the frame nor the list given by the caller changes."""
        before = panel_2l.copy()
        requested = ['entity']
        detect_panel_structure(panel_2l, requested)
        pd.testing.assert_frame_equal(panel_2l, before)
        assert requested == ['entity']

    def test_empty_panel(self, panel_2l):
        """The structure survives the absence of rows."""
        assert detect_panel_structure(empty_like(panel_2l), None) == (['entity'], True)

    def test_detected_columns_are_a_plain_list(self, panel_2l):
        """The columns detected from the index are a mutable ``list``, new at each call."""
        first = detect_panel_structure(panel_2l, None)[0]
        assert type(first) is list
        first.append('other')
        assert detect_panel_structure(panel_2l, None)[0] == ['entity']


class TestExtractTimeSeriesFromMultiindex:
    """``extract_time_series_from_multiindex`` keeps the last index level only."""

    def test_single_entity(self, dates):
        """Golden value: one entity, three dates, values read in order."""
        index = pd.MultiIndex.from_product([['FR'], dates], names=['entity', 'date'])
        result = extract_time_series_from_multiindex(pd.Series([1, 2, 3], index=index))

        assert (result.tolist(), list(result.index)) == ([1, 2, 3], list(dates))

    def test_index_is_the_last_level_with_its_name(self, panel_3l):
        """On three levels the index is the date level, with its own name."""
        result = extract_time_series_from_multiindex(panel_3l['x'])
        assert result.index.name == 'date'
        assert result.index.equals(panel_3l.index.get_level_values('date'))

    def test_values_are_kept_by_position(self, panel_2l):
        """Row order and values are those of the input."""
        assert extract_time_series_from_multiindex(panel_2l['x']).tolist() == [1, 2, 3, 4, 5, 6]

    def test_name_of_the_series_is_kept(self, panel_2l):
        """The result carries the name of the input, and an anonymous input stays anonymous."""
        assert extract_time_series_from_multiindex(panel_2l['x']).name == 'x'
        assert extract_time_series_from_multiindex(panel_2l['x'].rename(None)).name is None

    def test_several_entities_give_repeated_dates(self, panel_2l):
        """Entities are not separated: each date appears once per entity and the index is not unique."""
        result = extract_time_series_from_multiindex(panel_2l['x'])
        assert (result.index.is_unique, len(result)) == (False, 6)

    def test_period_dates(self):
        """A ``PeriodIndex`` last level is kept as such."""
        index = pd.MultiIndex.from_product([['FR'], pd.period_range('2023-01', periods=2, freq='M')])
        result = extract_time_series_from_multiindex(pd.Series([1, 2], index=index))
        assert isinstance(result.index, pd.PeriodIndex)

    def test_simple_index_is_returned_with_its_values(self, dates):
        """A series without ``MultiIndex`` also goes through: its index is its last level."""
        result = extract_time_series_from_multiindex(pd.Series([1, 2, 3], index=dates))
        assert (result.tolist(), list(result.index)) == ([1, 2, 3], list(dates))

    def test_empty_series(self, panel_2l):
        """No row gives an empty series."""
        assert len(extract_time_series_from_multiindex(panel_2l['x'].iloc[0:0])) == 0

    def test_input_is_not_modified(self, panel_2l):
        """The input series keeps its index and values."""
        before = panel_2l['x'].copy()
        extract_time_series_from_multiindex(panel_2l['x'])
        pd.testing.assert_series_equal(panel_2l['x'], before)

    def test_dataframe_is_refused(self, panel_2l):
        """The function takes a Series: a one-column DataFrame is not a one-dimensional input."""
        with pytest.raises(ValueError, match='1-dimensional'):
            extract_time_series_from_multiindex(panel_2l)


class TestGetUniquePanelEntities:
    """``get_unique_panel_entities`` lists the entity keys of a panel as tuples."""

    def test_two_level_panel(self, panel_2l):
        """Golden value: the entities in the order of first appearance."""
        assert get_unique_panel_entities(panel_2l) == [('FR',), ('DE',)]

    def test_three_level_panel(self, panel_3l):
        """An entity key holds one label per entity level."""
        assert get_unique_panel_entities(panel_3l) == [('EU', 'FR'), ('EU', 'DE'), ('AS', 'JP')]

    def test_series_and_dataframe_agree(self, panel_3l):
        """A Series and a DataFrame with the same index give the same entities."""
        assert get_unique_panel_entities(panel_3l['x']) == get_unique_panel_entities(panel_3l)

    def test_order_follows_the_rows_not_the_alphabet(self, panel_2l):
        """Reversing the entity blocks reverses the list; interleaving keeps the appearance order."""
        assert get_unique_panel_entities(reverse_entities(panel_2l)) == [('DE',), ('FR',)]
        assert get_unique_panel_entities(interleave_entities(panel_2l)) == [('FR',), ('DE',)]

    def test_unsorted_dates_do_not_change_the_entities(self, panel_2l):
        """Rows shuffled inside the frame still give each entity once."""
        shuffled = panel_2l.iloc[[4, 0, 5, 1, 3, 2]]
        assert get_unique_panel_entities(shuffled) == [('DE',), ('FR',)]

    def test_repeated_rows_do_not_repeat_entities(self, panel_2l):
        """An entity with duplicated rows is listed once."""
        assert get_unique_panel_entities(with_duplicated_rows(panel_2l, n=2)) == [('FR',), ('DE',)]

    def test_one_entity(self, panel_2l):
        """A panel with one entity lists that entity alone."""
        assert get_unique_panel_entities(panel_2l.loc[['DE']]) == [('DE',)]

    def test_one_column(self, panel_2l):
        """The columns are irrelevant."""
        assert get_unique_panel_entities(panel_2l[['x']]) == [('FR',), ('DE',)]

    def test_empty_panel(self, panel_2l):
        """A panel without rows has no entity."""
        assert get_unique_panel_entities(empty_like(panel_2l)) == []

    def test_spaces_accents_and_symbols(self, panel_2l):
        """Labels are returned verbatim."""
        renamed = rename_entities(panel_2l, {'FR': "Côte d'Ivoire", 'DE': 'New York (%)'})
        assert get_unique_panel_entities(renamed) == [("Côte d'Ivoire",), ('New York (%)',)]

    def test_integer_labels_stay_integers(self, panel_2l):
        """An integer label is not turned into a string."""
        entities = get_unique_panel_entities(rename_entities(panel_2l, {'FR': 10, 'DE': 20}))
        assert entities == [(10,), (20,)] and all(isinstance(key[0], int) for key in entities)

    def test_mixed_types_across_levels(self, dates):
        """Each level keeps its own type inside the key."""
        index = pd.MultiIndex.from_tuples([('FR', 2, dates[0]), ('DE', 3, dates[0])])
        assert get_unique_panel_entities(pd.Series([1, 2], index=index)) == [('FR', 2), ('DE', 3)]

    def test_keys_are_hashable_tuples_of_the_right_length(self, panel_3l):
        """Every key is a tuple with one label per entity level, usable as a dictionary key."""
        for key in get_unique_panel_entities(panel_3l):
            assert isinstance(key, tuple) and len(key) == len(get_entity_levels(panel_3l))
            hash(key)

    def test_three_level_index_built_from_a_two_level_panel(self, panel_2l):
        """Adding an outer level turns each key into a pair."""
        panel = to_three_level_index(panel_2l, region_by_entity={'FR': 'EU', 'DE': 'EU'})
        assert get_unique_panel_entities(panel) == [('EU', 'FR'), ('EU', 'DE')]

    def test_realistic_panel(self, heterogeneous_coverage_panel):
        """The three entities of the realistic panel, once each."""
        entities = get_unique_panel_entities(heterogeneous_coverage_panel)
        assert sorted(entities) == sorted((country,) for country in HETEROGENEOUS_PANEL_COUNTRIES)

    @pytest.mark.parametrize('missing', [np.nan, None], ids=['nan', 'none'])
    def test_missing_label_is_refused(self, dates, missing):
        """A missing entity label is an error: no entity key could select its rows."""
        index = pd.MultiIndex.from_arrays([['A', 'A', missing, missing], list(dates[:2]) * 2])
        with pytest.raises(ValueError, match='must not be missing'):
            get_unique_panel_entities(pd.Series(range(4), index=index))

    def test_missing_label_error_names_the_level(self, panel_3l):
        """The error says which entity level holds the missing values."""
        index = panel_3l.index.to_frame(index=False)
        index.loc[0, 'country'] = np.nan
        broken = panel_3l.set_axis(pd.MultiIndex.from_frame(index))
        with pytest.raises(ValueError, match="level 'country'"):
            get_unique_panel_entities(broken)

    def test_missing_date_is_not_an_entity_label(self, panel_2l):
        """Only the entity levels are checked: a missing date does not refuse the panel."""
        index = panel_2l.index.to_frame(index=False)
        index.loc[0, 'date'] = pd.NaT
        data = panel_2l.set_axis(pd.MultiIndex.from_frame(index))
        assert get_unique_panel_entities(data) == [('FR',), ('DE',)]

    def test_simple_index_is_refused(self, timeseries):
        """A time series has no entity level to list."""
        with pytest.raises(ValueError, match='at least 2 levels'):
            get_unique_panel_entities(timeseries)

    def test_one_level_multiindex_is_refused(self, dates):
        """A ``MultiIndex`` with a single level has no entity level either."""
        data = pd.Series([1, 2, 3], index=pd.MultiIndex.from_arrays([dates]))
        with pytest.raises(ValueError, match='at least 2 levels'):
            get_unique_panel_entities(data)


class TestPublicExports:
    """Every function of ``panel.utils`` without a leading underscore is public API."""

    @pytest.mark.parametrize('name', PUBLIC_FUNCTIONS)
    def test_every_public_function_is_exported_unchanged(self, name):
        """The package lists and re-exports the very function defined in ``panel.utils``."""
        assert name in panel_package.__all__
        assert getattr(panel_package, name) is getattr(panel_utils, name)

    def test_the_eighteen_functions_are_found(self):
        """Guard for the introspection: the module holds the expected public functions."""
        assert len(PUBLIC_FUNCTIONS) == 18

    def test_every_exported_name_exists(self):
        """``__all__`` lists nothing that the package does not define."""
        assert all(hasattr(panel_package, name) for name in panel_package.__all__)

    def test_private_helpers_are_not_exported(self):
        """Helpers with a leading underscore stay out of the package interface."""
        assert not [name for name in panel_package.__all__ if name.startswith('_')]
