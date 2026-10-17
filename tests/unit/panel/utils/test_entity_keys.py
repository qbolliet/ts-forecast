"""Unit tests for the entity-key helpers of ``tsforecast.panel.utils``.

Covers ``normalize_entity_key``, ``normalize_entity_mapping``, ``split_variable_key``,
``group_keys_by_entity_and_variable`` and ``extract_column_names``: scalar versus tuple
keys, keys with spaces, accents and mixed types, ``MultiIndex`` column keys, conflicting
notations, and the examples of each docstring.
"""
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from tsforecast.panel.utils import (
    extract_column_names,
    get_unique_panel_entities,
    group_keys_by_entity_and_variable,
    normalize_entity_key,
    normalize_entity_mapping,
    split_variable_key,
)

ACCENTED = "Côte d'Ivoire"


class TestNormalizeEntityKey:
    """``normalize_entity_key`` turns a scalar or tuple key into a tuple."""

    @pytest.mark.parametrize(
        'key, expected',
        [
            pytest.param('FR', ('FR',), id='scalar_string'),
            pytest.param(('FR',), ('FR',), id='one_element_tuple'),
            pytest.param(('EU', 'FR'), ('EU', 'FR'), id='two_levels'),
            pytest.param(('EU', 'FR', 'industry'), ('EU', 'FR', 'industry'), id='three_levels'),
            pytest.param(3, (3,), id='scalar_integer'),
            pytest.param(ACCENTED, (ACCENTED,), id='scalar_with_space_and_accent'),
            pytest.param(('Zone euro', 'Île-de-France'), ('Zone euro', 'Île-de-France'),
                         id='tuple_with_spaces_and_accents'),
            pytest.param(('FR', 2), ('FR', 2), id='tuple_of_mixed_types'),
            pytest.param((), (), id='empty_tuple'),
            pytest.param(None, (None,), id='none_is_a_scalar'),
            pytest.param((('a', 'b'),), (('a', 'b'),), id='tuple_holding_a_tuple'),
        ],
    )
    def test_golden_values(self, key, expected):
        """A scalar becomes a one-element tuple; a tuple is returned as it is."""
        assert normalize_entity_key(key) == expected

    @pytest.mark.parametrize('key', ['FR', 3, ACCENTED, ('EU', 'FR'), (), None])
    def test_result_is_a_tuple(self, key):
        """Whatever the key, the result is a tuple."""
        assert isinstance(normalize_entity_key(key), tuple)

    @pytest.mark.parametrize('key', ['FR', 3, ACCENTED, ('EU', 'FR'), ()])
    def test_normalization_is_idempotent(self, key):
        """Normalizing a normalized key changes nothing."""
        once = normalize_entity_key(key)
        assert normalize_entity_key(once) == once

    def test_scalar_and_one_element_tuple_are_the_same_key(self):
        """``'FR'`` and ``('FR',)`` designate the same entity, in both directions."""
        assert normalize_entity_key('FR') == normalize_entity_key(('FR',))
        assert normalize_entity_key(('FR',)) == normalize_entity_key('FR')

    def test_both_notations_address_the_same_dictionary_entry(self):
        """A dictionary keyed by normalized keys is reached with either notation."""
        table = {normalize_entity_key('FR'): 'Q'}
        assert table[normalize_entity_key(('FR',))] == 'Q'

    def test_keys_of_a_panel_index_are_already_normalized(self, panel_3l):
        """The tuples read from a ``MultiIndex`` are returned unchanged."""
        for entity in get_unique_panel_entities(panel_3l):
            assert normalize_entity_key(entity) == entity

    @pytest.mark.parametrize(
        'key, expected',
        [
            pytest.param(['FR'], ('FR',), id='one_label'),
            pytest.param(['EU', 'FR'], ('EU', 'FR'), id='two_labels'),
            pytest.param([], (), id='empty'),
        ],
    )
    def test_list_is_read_as_the_tuple_of_its_labels(self, key, expected):
        """A key coming from a JSON or YAML configuration is a list of labels."""
        assert normalize_entity_key(key) == expected

    @pytest.mark.parametrize(
        'key',
        [
            pytest.param({'a': 1}, id='dict'),
            pytest.param({'a'}, id='set'),
            pytest.param(np.array([1, 2]), id='array'),
            pytest.param((['a'], 'b'), id='tuple_holding_a_list'),
        ],
    )
    def test_unhashable_key_is_refused(self, key):
        """The result is always usable as a dictionary key, or the call fails at once."""
        with pytest.raises(TypeError, match='must be hashable'):
            normalize_entity_key(key)

    @pytest.mark.parametrize('key', ['FR', ['EU', 'FR'], ('EU', 'FR'), 3])
    def test_result_is_hashable(self, key):
        """Whatever the accepted notation, the result can index a dictionary."""
        hash(normalize_entity_key(key))


class TestNormalizeEntityMapping:
    """``normalize_entity_mapping`` re-keys a per-entity binding on tuples."""

    @pytest.mark.parametrize(
        'mapping, expected',
        [
            pytest.param({'FR': 'Q'}, {('FR',): 'Q'}, id='scalar_key'),
            pytest.param({('FR',): 'Q'}, {('FR',): 'Q'}, id='tuple_key'),
            pytest.param({('Zone euro', 'IT'): 'Y'}, {('Zone euro', 'IT'): 'Y'}, id='two_entity_levels'),
            pytest.param({'FR': 'Q', ('DE',): 'M'}, {('FR',): 'Q', ('DE',): 'M'}, id='mixed_keys'),
            pytest.param({ACCENTED: 'Q', 'New York': 'M'}, {(ACCENTED,): 'Q', ('New York',): 'M'},
                         id='spaces_and_accents'),
            pytest.param({1: 'Q', 2: 'M'}, {(1,): 'Q', (2,): 'M'}, id='integer_keys'),
            pytest.param({('FR', 2): 'Y'}, {('FR', 2): 'Y'}, id='mixed_types_in_a_key'),
            pytest.param({}, {}, id='empty'),
        ],
    )
    def test_keys_become_tuples(self, mapping, expected):
        """Scalar keys become one-element tuples; tuple keys are kept."""
        assert normalize_entity_mapping(mapping) == expected

    def test_docstring_examples(self):
        """The two examples of the docstring."""
        assert normalize_entity_mapping({'FR': 'Q', ('DE',): 'M'}) == {('FR',): 'Q', ('DE',): 'M'}
        assert normalize_entity_mapping({('Zone euro', 'IT'): 'Y'}) == {('Zone euro', 'IT'): 'Y'}

    def test_order_is_kept_and_input_untouched(self):
        """A new dict, in the order of the input, which stays as it was."""
        mapping = {'IT': 'M', 'FR': 'Y'}
        normalized = normalize_entity_mapping(mapping)
        assert (list(normalized), mapping) == ([('IT',), ('FR',)], {'IT': 'M', 'FR': 'Y'})

    def test_result_is_a_new_dict(self):
        """The result is a distinct plain ``dict``, even from a tuple-keyed input."""
        mapping = {('FR',): 'Q'}
        normalized = normalize_entity_mapping(mapping)
        assert normalized is not mapping and type(normalized) is dict

    def test_values_are_kept_as_they_are(self):
        """Values are not copied nor transformed."""
        value = ['not', 'a', 'frequency']
        assert normalize_entity_mapping({'FR': value})[('FR',)] is value

    def test_accepts_any_mapping(self):
        """A read-only mapping is accepted like a dict."""
        assert normalize_entity_mapping(MappingProxyType({'FR': 'Q'})) == {('FR',): 'Q'}

    def test_entity_and_column_keys_pass_through_unchanged(self):
        """An ``(entity, column)`` key is a tuple like any other; a plain column name is wrapped.

        The function cannot tell a column from a one-level entity: it only
        normalizes the notation of the keys.
        """
        mapping = {('FR', 'gdp'): 'Q', 'cpi': 'M'}
        assert normalize_entity_mapping(mapping) == {('FR', 'gdp'): 'Q', ('cpi',): 'M'}

    @pytest.mark.parametrize(
        'mapping',
        [
            pytest.param({'FR': 'Q', ('FR',): 'M'}, id='tuple_listed_last'),
            pytest.param({('FR',): 'M', 'FR': 'Q'}, id='scalar_listed_last'),
            pytest.param({'FR': 'Q', 'DE': 'M', ('FR',): 'Y'}, id='other_entity_in_between'),
        ],
    )
    def test_conflicting_notations_are_refused(self, mapping):
        """The same entity declared twice with different values is ambiguous: no tie-break."""
        with pytest.raises(ValueError, match=r"Conflicting values for entity \('FR',\)"):
            normalize_entity_mapping(mapping)

    def test_conflict_message_names_both_values(self):
        """The error shows the two values in competition."""
        with pytest.raises(ValueError, match="'Q' and 'M'"):
            normalize_entity_mapping({'FR': 'Q', ('FR',): 'M'})

    def test_same_value_under_both_notations_is_one_binding(self):
        """A redundant declaration is harmless: it keeps the position of the first key."""
        result = normalize_entity_mapping({'FR': 'Q', 'DE': 'M', ('FR',): 'Q'})
        assert list(result.items()) == [(('FR',), 'Q'), (('DE',), 'M')]

    def test_values_without_a_boolean_equality_count_as_different(self):
        """Arrays cannot be compared to a single boolean: two notations holding them conflict."""
        with pytest.raises(ValueError, match='Conflicting values'):
            normalize_entity_mapping({'FR': np.array([1, 2]), ('FR',): np.array([1, 2])})

    def test_list_key_is_read_as_a_tuple(self):
        """A key given as a list of labels designates the same entity as the tuple."""
        mapping = {('EU', 'FR'): 'Q'}
        assert normalize_entity_key(['EU', 'FR']) in normalize_entity_mapping(mapping)


class TestSplitVariableKey:
    """``split_variable_key`` always yields an entity tuple, ``()`` outside a panel."""

    @pytest.mark.parametrize(
        'key, expected',
        [
            pytest.param(('FR', 'gdp'), (('FR',), 'gdp'), id='one_entity_level'),
            pytest.param(('FR', 'manufacturing', 'gdp'), (('FR', 'manufacturing'), 'gdp'),
                         id='two_entity_levels'),
            pytest.param(('EU', 'FR', 'industry', 'gdp'), (('EU', 'FR', 'industry'), 'gdp'),
                         id='three_entity_levels'),
            pytest.param('gdp', ((), 'gdp'), id='time_series_column'),
            pytest.param(('gdp',), ((), 'gdp'), id='one_element_tuple_is_a_time_series_column'),
            pytest.param(('FR', 'PIB brut (%)'), (('FR',), 'PIB brut (%)'), id='special_column_name'),
            pytest.param((ACCENTED, 'prix à la production'), ((ACCENTED,), 'prix à la production'),
                         id='accents_and_spaces'),
            pytest.param((1, 'gdp'), ((1,), 'gdp'), id='integer_entity'),
            pytest.param(('FR', 3), (('FR',), 3), id='integer_column'),
            pytest.param(3, ((), 3), id='integer_time_series_column'),
        ],
    )
    def test_entity_part_is_always_a_tuple(self, key, expected):
        """The entity part is a tuple at every level, the empty tuple for a plain column."""
        assert split_variable_key(key) == expected

    def test_docstring_examples(self):
        """The three examples of the docstring."""
        assert split_variable_key(('FR', 'gdp')) == (('FR',), 'gdp')
        assert split_variable_key(('FR', 'manufacturing', 'gdp')) == (('FR', 'manufacturing'), 'gdp')
        assert split_variable_key('gdp') == ((), 'gdp')

    def test_multiindex_columns_are_read_as_entity_and_column(self):
        """A column of a ``MultiIndex`` of columns is split like a panel variable key."""
        columns = pd.MultiIndex.from_tuples([('a', 'x'), ('a', 'y')])
        assert [split_variable_key(key) for key in columns] == [(('a',), 'x'), (('a',), 'y')]

    @pytest.mark.parametrize('entity', [(), ('FR',), ('EU', 'FR'), ('EU', 'FR', 'industry')])
    def test_inverse_of_building_a_key(self, entity):
        """Splitting ``(*entity, column)`` gives back the entity and the column."""
        assert split_variable_key((*entity, 'gdp')) == (entity, 'gdp')

    def test_entity_matches_the_unique_entities_of_a_panel(self, panel_2l, panel_3l):
        """The entity of a variable key equals the matching key of ``get_unique_panel_entities``."""
        for panel in (panel_2l, panel_3l):
            for entity in get_unique_panel_entities(panel):
                assert split_variable_key((*entity, 'x'))[0] == entity

    def test_empty_tuple_has_no_column(self):
        """An empty key designates nothing: it is refused."""
        with pytest.raises(IndexError):
            split_variable_key(())


class TestGroupKeysByEntityAndVariable:
    """``group_keys_by_entity_and_variable`` gathers the column names of each entity."""

    def test_docstring_examples(self):
        """The two examples of the docstring."""
        keys = [('FR', 'gdp'), ('FR', 'cpi'), ('DE', 'gdp')]
        assert group_keys_by_entity_and_variable(keys) == {('FR',): ['gdp', 'cpi'], ('DE',): ['gdp']}
        assert group_keys_by_entity_and_variable(['gdp', 'cpi']) == {(): ['gdp', 'cpi']}

    def test_entities_and_columns_follow_first_appearance(self):
        """Neither entities nor columns are sorted: they keep their order of first appearance."""
        keys = [('DE', 'b'), ('FR', 'z'), ('DE', 'a'), ('FR', 'y')]
        grouped = group_keys_by_entity_and_variable(keys)
        assert (list(grouped), grouped) == ([('DE',), ('FR',)], {('DE',): ['b', 'a'], ('FR',): ['z', 'y']})

    def test_repeated_variable_is_listed_once(self):
        """The same ``(entity, column)`` key twice gives one column."""
        grouped = group_keys_by_entity_and_variable([('FR', 'gdp'), ('FR', 'gdp')])
        assert grouped == {('FR',): ['gdp']}

    def test_same_column_in_two_entities(self):
        """A column shared by two entities appears under each."""
        grouped = group_keys_by_entity_and_variable([('FR', 'gdp'), ('DE', 'gdp')])
        assert grouped == {('FR',): ['gdp'], ('DE',): ['gdp']}

    def test_entities_of_different_depth_stay_distinct(self):
        """``('FR',)`` and ``('FR', 'industry')`` are two entities, not one."""
        keys = [('FR', 'gdp'), ('FR', 'industry', 'gdp')]
        grouped = group_keys_by_entity_and_variable(keys)
        assert grouped == {('FR',): ['gdp'], ('FR', 'industry'): ['gdp']}

    def test_plain_names_and_panel_keys_are_grouped_apart(self):
        """Plain column names go under the empty entity, panel keys under their entity."""
        grouped = group_keys_by_entity_and_variable(['cpi', ('FR', 'gdp')])
        assert grouped == {(): ['cpi'], ('FR',): ['gdp']}

    def test_plain_name_and_one_element_tuple_are_the_same_column(self):
        """``'gdp'`` and ``('gdp',)`` both designate the column of a time series."""
        assert group_keys_by_entity_and_variable(['gdp', ('gdp',)]) == {(): ['gdp']}

    def test_three_level_entities(self):
        """An entity of two levels is the tuple of both."""
        keys = [('EU', 'FR', 'gdp'), ('EU', 'DE', 'gdp'), ('EU', 'FR', 'cpi')]
        grouped = group_keys_by_entity_and_variable(keys)
        assert grouped == {('EU', 'FR'): ['gdp', 'cpi'], ('EU', 'DE'): ['gdp']}

    def test_special_names(self):
        """Spaces, accents and symbols in entities and columns are kept verbatim."""
        keys = [(ACCENTED, 'PIB brut (%)'), (ACCENTED, 'prix/mois'), (1, 'gdp')]
        grouped = group_keys_by_entity_and_variable(keys)
        assert grouped == {(ACCENTED,): ['PIB brut (%)', 'prix/mois'], (1,): ['gdp']}

    def test_multiindex_columns(self):
        """The columns of a ``MultiIndex`` of columns are grouped by their first level."""
        columns = pd.MultiIndex.from_tuples([('a', 'x'), ('a', 'y'), ('b', 'x')])
        assert group_keys_by_entity_and_variable(list(columns)) == {('a',): ['x', 'y'], ('b',): ['x']}

    def test_empty_input(self):
        """No key gives no group."""
        assert group_keys_by_entity_and_variable([]) == {}

    def test_regrouped_keys_are_the_input_keys(self):
        """Rebuilding ``(*entity, column)`` from the groups gives back the distinct input keys."""
        keys = [('FR', 'gdp'), ('DE', 'gdp'), ('FR', 'cpi'), ('EU', 'FR', 'gdp')]
        grouped = group_keys_by_entity_and_variable(keys)
        rebuilt = {(*entity, column) for entity, columns in grouped.items() for column in columns}
        assert rebuilt == set(keys)

    def test_empty_tuple_has_no_column(self):
        """An empty key designates nothing: it is refused."""
        with pytest.raises(IndexError):
            group_keys_by_entity_and_variable([()])


class TestExtractColumnNames:
    """``extract_column_names`` lists the distinct column names of a set of variable keys."""

    def test_docstring_example(self):
        """The example of the docstring."""
        assert extract_column_names([('FR', 'gdp'), ('DE', 'gdp'), 'cpi']) == ['gdp', 'cpi']

    @pytest.mark.parametrize(
        'keys, expected',
        [
            pytest.param([('FR', 'gdp'), ('DE', 'gdp')], {'gdp'}, id='same_column_in_two_entities'),
            pytest.param(['gdp', 'cpi'], {'gdp', 'cpi'}, id='time_series_names'),
            pytest.param([('FR', 'gdp'), 'cpi'], {'gdp', 'cpi'}, id='panel_keys_and_plain_names'),
            pytest.param([('EU', 'FR', 'gdp'), ('EU', 'DE', 'cpi')], {'gdp', 'cpi'}, id='two_entity_levels'),
            pytest.param([('FR', 'PIB brut (%)'), ('FR', 'prix à la production')],
                         {'PIB brut (%)', 'prix à la production'}, id='special_names'),
            pytest.param([('FR', 3), 4], {3, 4}, id='integer_names'),
            pytest.param([('gdp',)], {'gdp'}, id='one_element_tuple'),
            pytest.param([], set(), id='empty'),
        ],
    )
    def test_distinct_names(self, keys, expected):
        """The result holds each column name once, whatever the entity."""
        names = extract_column_names(keys)
        assert (len(names), set(names)) == (len(expected), expected)

    def test_result_is_a_list(self):
        """A list, as annotated."""
        assert isinstance(extract_column_names([('FR', 'gdp')]), list)

    def test_multiindex_columns(self):
        """The last level of each column of a ``MultiIndex`` of columns."""
        columns = pd.MultiIndex.from_tuples([('a', 'x'), ('a', 'y'), ('b', 'x')])
        assert sorted(extract_column_names(list(columns))) == ['x', 'y']

    def test_empty_tuple_has_no_column(self):
        """An empty key designates nothing: it is refused."""
        with pytest.raises(IndexError):
            extract_column_names([()])

    def test_names_keep_their_order_of_first_appearance(self):
        """Like the other key helpers, the order of first appearance is kept."""
        names = list('zyxwvutsrqpo')
        assert extract_column_names([('FR', name) for name in names]) == names

    def test_repeated_names_keep_the_position_of_their_first_appearance(self):
        """A name met again later does not move."""
        keys = [('FR', 'gdp'), 'cpi', ('DE', 'gdp'), ('DE', 'unemployment')]
        assert extract_column_names(keys) == ['gdp', 'cpi', 'unemployment']
