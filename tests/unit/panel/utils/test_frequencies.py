"""Unit tests for the per-entity frequency helpers of ``tsforecast.panel.utils``.

Covers ``get_entity_target_frequency`` (one frequency per entity) and
``resolve_entity_column_frequencies`` (one frequency per ``(entity, column)`` pair):
global, per-entity, per-column and per-pair declarations, their order of precedence,
entities without any declared column, single- and multi-level entities, and the
examples of each docstring.
"""
import pytest

from tsforecast.panel.utils import (
    get_entity_target_frequency,
    get_unique_panel_entities,
    normalize_entity_mapping,
    resolve_entity_column_frequencies,
)

ACCENTED = "Côte d'Ivoire"
COLUMNS = ['gdp', 'cpi']


class TestGetEntityTargetFrequency:
    """``get_entity_target_frequency`` gives the one target frequency of an entity."""

    @pytest.mark.parametrize('entity', [('FR',), ('EU', 'FR'), ()], ids=['one_level', 'two_levels', 'time_series'])
    def test_string_applies_to_every_entity(self, entity):
        """A single frequency string is the answer for any entity."""
        assert get_entity_target_frequency(entity, 'M') == 'M'

    def test_docstring_examples(self):
        """The two examples of the docstring."""
        assert get_entity_target_frequency(('FR',), 'M') == 'M'
        assert get_entity_target_frequency(('FR',), {'FR': 'M', 'DE': 'Q'}) == 'M'

    @pytest.mark.parametrize(
        'entity, mapping, expected',
        [
            pytest.param(('FR',), {('FR',): 'M', ('DE',): 'Q'}, 'M', id='tuple_key'),
            pytest.param(('DE',), {'FR': 'M', 'DE': 'Q'}, 'Q', id='scalar_key'),
            pytest.param(('FR',), {'DE': 'Q', ('FR',): 'M'}, 'M', id='mixed_notations'),
            pytest.param(('EU', 'FR'), {('EU', 'FR'): 'M', ('EU', 'DE'): 'Q'}, 'M', id='two_level_tuple_key'),
            pytest.param((ACCENTED,), {ACCENTED: 'M'}, 'M', id='accents_and_apostrophe'),
            pytest.param(('New York (%)',), {('New York (%)',): 'Q'}, 'Q', id='spaces_and_symbols'),
            pytest.param((1,), {1: 'M'}, 'M', id='integer_scalar_key'),
            pytest.param((1,), {(1,): 'Y'}, 'Y', id='integer_tuple_key'),
            pytest.param((), {(): 'M'}, 'M', id='time_series_entity'),
        ],
    )
    def test_dictionary_lookup(self, entity, mapping, expected):
        """The entity is looked up as a tuple, then as a scalar when it has one label."""
        assert get_entity_target_frequency(entity, mapping) == expected

    def test_frequency_is_returned_verbatim(self):
        """No normalization of the frequency string."""
        assert get_entity_target_frequency(('FR',), {'FR': 'ME'}) == 'ME'

    @pytest.mark.parametrize(
        'mapping',
        [{('FR',): 'Q', 'FR': 'M'}, {'FR': 'M', ('FR',): 'Q'}],
        ids=['tuple_listed_first', 'scalar_listed_first'],
    )
    def test_both_notations_with_different_values_are_refused(self, mapping):
        """The same entity declared twice with different frequencies is ambiguous, in any order."""
        with pytest.raises(ValueError, match=r"Conflicting values for entity \('FR',\)"):
            get_entity_target_frequency(('FR',), mapping)

    def test_both_notations_with_the_same_value_are_one_declaration(self):
        """A redundant declaration is harmless."""
        assert get_entity_target_frequency(('FR',), {'FR': 'Q', ('FR',): 'Q'}) == 'Q'

    def test_conflict_of_another_entity_is_not_read(self):
        """Only the notations of the entity being looked up are compared."""
        assert get_entity_target_frequency(('DE',), {'FR': 'M', ('FR',): 'Q', 'DE': 'Y'}) == 'Y'

    @pytest.mark.parametrize('entity', ['FR', ['FR']], ids=['scalar', 'list'])
    def test_entity_may_be_given_in_any_notation(self, entity):
        """The entity itself is normalized like the keys of the mapping."""
        assert get_entity_target_frequency(entity, {('FR',): 'M'}) == 'M'

    def test_same_reading_as_the_normalized_mapping(self):
        """A mapping normalized first is read like the mapping itself (or refused by both)."""
        mapping = {'FR': 'M', ('DE',): 'Q'}
        normalized = normalize_entity_mapping(mapping)
        for entity in [('FR',), ('DE',)]:
            assert (get_entity_target_frequency(entity, normalized)
                    == get_entity_target_frequency(entity, mapping))
        with pytest.raises(ValueError):
            normalize_entity_mapping({'FR': 'M', ('FR',): 'Q'})

    @pytest.mark.parametrize(
        'entity, mapping',
        [
            pytest.param(('XX',), {'FR': 'M'}, id='entity_not_declared'),
            pytest.param(('FR',), {}, id='empty_mapping'),
            pytest.param(('FR', 'a'), {'FR': 'M'}, id='scalar_key_does_not_cover_two_levels'),
            pytest.param(('FR',), {('FR', 'gdp'): 'Q'}, id='pair_key_is_not_an_entity_key'),
            pytest.param(('FR',), {'gdp': 'Q'}, id='column_key_is_not_an_entity_key'),
            pytest.param((), {'gdp': 'M'}, id='time_series_without_empty_key'),
            pytest.param(('FR',), {('FR',): None}, id='declared_as_none'),
        ],
    )
    def test_no_frequency_for_the_entity(self, entity, mapping):
        """An entity that no key designates is an error, naming the entity."""
        with pytest.raises(ValueError, match='No target frequency found for entity'):
            get_entity_target_frequency(entity, mapping)

    def test_error_message_names_the_entity(self):
        """The error message gives the entity key that was looked up."""
        with pytest.raises(ValueError, match=r"\('XX',\)"):
            get_entity_target_frequency(('XX',), {'FR': 'M'})

    def test_input_mapping_is_not_modified(self):
        """The mapping given by the caller is left as it was."""
        mapping = {'FR': 'M'}
        get_entity_target_frequency(('FR',), mapping)
        assert mapping == {'FR': 'M'}


class TestResolveEntityColumnFrequencies:
    """``resolve_entity_column_frequencies`` resolves the frequency of each ``(entity, column)`` pair."""

    def test_docstring_examples(self):
        """The three examples of the docstring."""
        assert resolve_entity_column_frequencies(('FR',), COLUMNS, 'M') == {'gdp': 'M', 'cpi': 'M'}
        assert resolve_entity_column_frequencies(
            ('FR',), COLUMNS, {('FR', 'gdp'): 'Q', ('FR',): 'M'}
        ) == {'gdp': 'Q', 'cpi': 'M'}
        assert resolve_entity_column_frequencies(('DE',), COLUMNS, {'cpi': 'ME'}) == {'cpi': 'ME'}

    # ---- Déclarations globale, par entité, par colonne, par couple --------------------------

    def test_global_string_covers_every_column(self):
        """A frequency string is given to every column."""
        assert resolve_entity_column_frequencies(('FR',), COLUMNS, 'M') == {'gdp': 'M', 'cpi': 'M'}

    def test_no_column_gives_no_entry(self):
        """Nothing to resolve, nothing returned, whatever the declaration."""
        assert resolve_entity_column_frequencies(('FR',), [], 'M') == {}
        assert resolve_entity_column_frequencies(('FR',), [], {'FR': 'M'}) == {}

    @pytest.mark.parametrize(
        'declaration',
        [
            pytest.param({('FR',): 'Q'}, id='tuple_key'),
            pytest.param({'FR': 'Q'}, id='scalar_key'),
        ],
    )
    def test_per_entity_declaration_covers_all_columns_of_that_entity(self, declaration):
        """An entity key gives its frequency to every column of the entity, in either notation."""
        assert resolve_entity_column_frequencies(('FR',), COLUMNS, declaration) == {'gdp': 'Q', 'cpi': 'Q'}

    def test_per_column_declaration_covers_that_column_in_every_entity(self):
        """A column key applies to the column wherever it is found; other columns are left out."""
        declaration = {'gdp': 'Q'}
        assert resolve_entity_column_frequencies(('FR',), COLUMNS, declaration) == {'gdp': 'Q'}
        assert resolve_entity_column_frequencies(('DE',), COLUMNS, declaration) == {'gdp': 'Q'}

    def test_per_pair_declaration_covers_that_pair_only(self):
        """An ``(entity, column)`` key does not reach the other columns nor the other entities."""
        declaration = {('FR', 'gdp'): 'Q'}
        assert resolve_entity_column_frequencies(('FR',), COLUMNS, declaration) == {'gdp': 'Q'}
        assert resolve_entity_column_frequencies(('DE',), COLUMNS, declaration) == {}

    def test_frequency_belongs_to_the_pair_not_to_the_column(self):
        """The same column has a different frequency in each entity, and each column its own in an entity."""
        declaration = {('FR', 'gdp'): 'Q', ('DE', 'gdp'): 'M', ('FR', 'cpi'): 'Y'}
        assert resolve_entity_column_frequencies(('FR',), COLUMNS, declaration) == {'gdp': 'Q', 'cpi': 'Y'}
        assert resolve_entity_column_frequencies(('DE',), COLUMNS, declaration) == {'gdp': 'M'}

    # ---- Précédence ------------------------------------------------------------------------

    @pytest.mark.parametrize(
        'declaration, expected',
        [
            pytest.param({('FR', 'gdp'): 'pair', ('FR',): 'entity', 'gdp': 'column'}, 'pair',
                         id='pair_beats_entity_and_column'),
            pytest.param({('FR',): 'entity', 'gdp': 'column'}, 'entity', id='entity_beats_column'),
            pytest.param({('FR', 'gdp'): 'pair', 'gdp': 'column'}, 'pair', id='pair_beats_column'),
            pytest.param({'FR': 'entity', 'gdp': 'column'}, 'entity', id='scalar_entity_beats_column'),
            pytest.param({'gdp': 'column'}, 'column', id='column_alone'),
        ],
    )
    def test_most_specific_key_wins(self, declaration, expected):
        """Precedence ``(entity, column)`` > ``(entity,)`` > ``column``."""
        assert resolve_entity_column_frequencies(('FR',), COLUMNS, declaration)['gdp'] == expected

    def test_precedence_is_resolved_column_by_column(self):
        """A column covered by the pair key and another covered by the entity key in the same entity."""
        declaration = {('FR', 'gdp'): 'A1', ('FR',): 'E', 'gdp': 'C', 'cpi': 'C2'}
        assert resolve_entity_column_frequencies(('FR',), COLUMNS, declaration) == {'gdp': 'A1', 'cpi': 'E'}

    def test_other_entity_only_sees_column_keys(self):
        """Pair and entity keys of FR do not leak into DE; the column key does."""
        declaration = {('FR', 'gdp'): 'Q', ('FR',): 'M', 'cpi': 'C'}
        assert resolve_entity_column_frequencies(('DE',), COLUMNS, declaration) == {'cpi': 'C'}

    @pytest.mark.parametrize(
        'declaration',
        [{'FR': 'Q', ('FR',): 'M'}, {('FR',): 'M', 'FR': 'Q'}],
        ids=['tuple_listed_last', 'scalar_listed_last'],
    )
    def test_both_notations_of_an_entity_with_different_values_are_refused(self, declaration):
        """The same entity declared twice with different frequencies is ambiguous, in any order."""
        with pytest.raises(ValueError, match=r"Conflicting values for entity \('FR',\)"):
            resolve_entity_column_frequencies(('FR',), COLUMNS, declaration)

    def test_both_notations_of_an_entity_with_the_same_value_are_one_declaration(self):
        """A redundant declaration is harmless."""
        result = resolve_entity_column_frequencies(('FR',), COLUMNS, {'FR': 'Q', ('FR',): 'Q'})
        assert result == {'gdp': 'Q', 'cpi': 'Q'}

    def test_conflict_of_another_entity_is_not_read(self):
        """Only the notations of the entity being resolved are compared."""
        declaration = {'FR': 'Q', ('FR',): 'M', ('DE',): 'Y'}
        assert resolve_entity_column_frequencies(('DE',), COLUMNS, declaration) == {'gdp': 'Y', 'cpi': 'Y'}

    def test_none_counts_as_undeclared(self):
        """A key mapped to None designates nothing, and the next most specific key applies."""
        declaration = {('FR', 'gdp'): None, ('FR',): 'M'}
        assert resolve_entity_column_frequencies(('FR',), COLUMNS, declaration) == {'gdp': 'M', 'cpi': 'M'}
        assert resolve_entity_column_frequencies(('FR',), COLUMNS, {('FR',): None}) == {}

    @pytest.mark.parametrize('entity', [('FR',), ['FR'], 'FR'], ids=['tuple', 'list', 'scalar'])
    def test_entity_may_be_given_in_any_notation(self, entity):
        """The entity itself is normalized like the keys of the declaration."""
        assert resolve_entity_column_frequencies(entity, COLUMNS, {'FR': 'Q'}) == {'gdp': 'Q', 'cpi': 'Q'}

    def test_scalar_key_is_the_entity_before_being_a_column(self):
        """A scalar key that is both the name of the entity and of a column designates the entity."""
        result = resolve_entity_column_frequencies(('FR',), ['FR', 'cpi'], {'FR': 'Q'})
        assert result == {'FR': 'Q', 'cpi': 'Q'}

    def test_scalar_key_of_another_entity_is_still_a_column_key(self):
        """For an entity that the key does not name, the same scalar key is read as a column."""
        result = resolve_entity_column_frequencies(('DE',), ['FR', 'cpi'], {'FR': 'Q'})
        assert result == {'FR': 'Q'}

    # ---- Colonnes non couvertes -------------------------------------------------------------

    def test_entity_without_any_declared_column(self):
        """No key concerns the entity or its columns: the result is empty, the caller leaves it unchanged."""
        assert resolve_entity_column_frequencies(('IT',), COLUMNS, {('FR', 'gdp'): 'Q', ('FR',): 'M'}) == {}

    def test_empty_declaration_covers_nothing(self):
        """With no key at all, no column is covered."""
        assert resolve_entity_column_frequencies(('FR',), COLUMNS, {}) == {}

    def test_uncovered_columns_are_omitted_not_set_to_none(self):
        """A column that no key designates is absent from the result, not mapped to None."""
        result = resolve_entity_column_frequencies(('FR',), COLUMNS, {('FR', 'gdp'): 'Q'})
        assert 'cpi' not in result

    # ---- Entités à plusieurs niveaux, séries temporelles ------------------------------------

    def test_two_level_entity_pair_key(self):
        """The pair key of a two-level entity has three labels."""
        result = resolve_entity_column_frequencies(('EU', 'FR'), COLUMNS, {('EU', 'FR', 'gdp'): 'Q'})
        assert result == {'gdp': 'Q'}

    def test_two_level_entity_key(self):
        """The key of a two-level entity gives its frequency to every column."""
        result = resolve_entity_column_frequencies(('EU', 'FR'), COLUMNS, {('EU', 'FR'): 'Q'})
        assert result == {'gdp': 'Q', 'cpi': 'Q'}

    def test_scalar_key_does_not_cover_a_two_level_entity(self):
        """A scalar key designates a one-label entity: it does not cover ``('EU', 'FR')``."""
        assert resolve_entity_column_frequencies(('EU', 'FR'), COLUMNS, {'EU': 'Q'}) == {}

    def test_time_series_column_key(self):
        """A time series has the empty entity; its columns are declared by name."""
        assert resolve_entity_column_frequencies((), COLUMNS, {'gdp': 'Q'}) == {'gdp': 'Q'}

    def test_time_series_one_element_pair_key(self):
        """The pair key of a time series is the one-element tuple of the column."""
        assert resolve_entity_column_frequencies((), COLUMNS, {('gdp',): 'Q'}) == {'gdp': 'Q'}

    def test_time_series_empty_entity_key(self):
        """The empty tuple designates the whole time series."""
        assert resolve_entity_column_frequencies((), COLUMNS, {(): 'Q'}) == {'gdp': 'Q', 'cpi': 'Q'}

    # ---- Noms et types ----------------------------------------------------------------------

    def test_accents_spaces_and_symbols(self):
        """Entity and column names are matched verbatim."""
        result = resolve_entity_column_frequencies(
            (ACCENTED,), ['PIB brut (%)', 'prix/mois'], {(ACCENTED, 'PIB brut (%)'): 'Q', (ACCENTED,): 'M'}
        )
        assert result == {'PIB brut (%)': 'Q', 'prix/mois': 'M'}

    def test_integer_entity_and_integer_column(self):
        """Integer labels are matched by value."""
        assert resolve_entity_column_frequencies((1,), [3], {(1, 3): 'Q'}) == {3: 'Q'}
        assert resolve_entity_column_frequencies((1,), ['gdp'], {1: 'Q'}) == {'gdp': 'Q'}

    def test_tuple_column_name(self):
        """A column of a ``MultiIndex`` of columns is a tuple, matched as one label."""
        result = resolve_entity_column_frequencies(('FR',), [('a', 'x')], {('FR', ('a', 'x')): 'Q'})
        assert result == {('a', 'x'): 'Q'}

    def test_result_follows_the_order_of_the_columns(self):
        """The result lists the columns in the order they were given."""
        result = resolve_entity_column_frequencies(('FR',), ['c', 'a', 'b'], 'M')
        assert list(result) == ['c', 'a', 'b']

    def test_repeated_column_is_resolved_once(self):
        """A column given twice appears once in the result."""
        assert resolve_entity_column_frequencies(('FR',), ['gdp', 'gdp'], {'gdp': 'Q'}) == {'gdp': 'Q'}

    def test_frequencies_are_returned_verbatim(self):
        """No normalization of the frequency strings, position suffixes included."""
        result = resolve_entity_column_frequencies(('FR',), COLUMNS, {'gdp': 'QS', 'cpi': 'ME'})
        assert result == {'gdp': 'QS', 'cpi': 'ME'}

    def test_input_declaration_is_not_modified(self):
        """The declaration given by the caller is left as it was."""
        declaration = {('FR', 'gdp'): 'Q', 'cpi': 'M'}
        resolve_entity_column_frequencies(('FR',), COLUMNS, declaration)
        assert declaration == {('FR', 'gdp'): 'Q', 'cpi': 'M'}

    # ---- Cohérence entre les deux fonctions et jeux réalistes -------------------------------

    @pytest.mark.parametrize('declaration', [{'FR': 'Q', 'DE': 'M'}, {('FR',): 'Q', ('DE',): 'M'}])
    def test_agrees_with_get_entity_target_frequency_on_entity_declarations(self, declaration):
        """A declaration made of entity keys gives each column the frequency of its entity."""
        for entity in [('FR',), ('DE',)]:
            expected = get_entity_target_frequency(entity, declaration)
            resolved = resolve_entity_column_frequencies(entity, COLUMNS, declaration)
            assert set(resolved.values()) == {expected} and set(resolved) == set(COLUMNS)

    def test_string_declaration_on_the_realistic_panel(self, heterogeneous_coverage_panel):
        """Every column of every entity of the realistic panel receives the global frequency."""
        columns = list(heterogeneous_coverage_panel.columns)
        for entity in get_unique_panel_entities(heterogeneous_coverage_panel):
            assert resolve_entity_column_frequencies(entity, columns, 'MS') == dict.fromkeys(columns, 'MS')

    def test_per_entity_declaration_on_the_realistic_panel(self, heterogeneous_coverage_panel):
        """One frequency per entity, read for each entity of the realistic panel."""
        entities = get_unique_panel_entities(heterogeneous_coverage_panel)
        declaration = {entity: freq for entity, freq in zip(entities, ['MS', 'QS', 'YS'])}
        columns = list(heterogeneous_coverage_panel.columns)
        for entity, freq in declaration.items():
            assert set(resolve_entity_column_frequencies(entity, columns, declaration).values()) == {freq}
