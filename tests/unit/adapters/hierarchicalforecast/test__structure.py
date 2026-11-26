"""Unit tests of ``tsforecast.adapters.hierarchicalforecast._structure``.

``tree_structure`` (node order, summing matrix, depth levels, forests, leaves at
several depths, malformed trees), ``Structure`` (identifiers, tags, summing frame,
node lookup), ``parse_hierarchy`` (both forms, node and copy levels, errors) and
``HierarchySpec.build_structure`` (aggregates implied by the bottom nodes, grouped
structures, unmatched rows, aggregates without bottom rows, unsortable labels).
"""
import numpy as np
import pytest

from tsforecast.adapters.hierarchicalforecast._structure import (
    HierarchySpec,
    parse_hierarchy,
    tree_structure,
)

# Arbre de référence : a = b + c, c = d + e
TREE = {'a': ['b', 'c'], 'c': ['d', 'e']}


class TestTreeStructure:
    """Structure of a ``{parent: [children]}`` hierarchy."""

    def test_nodes_summing_matrix_and_levels(self):
        structure = tree_structure(TREE)
        assert structure.node_keys == ('a', 'c', 'b', 'd', 'e')
        np.testing.assert_array_equal(
            structure.summing, [[1, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        assert structure.levels == ('depth_0', 'depth_1', 'depth_2')
        assert structure.node_levels.tolist() == [0, 1, 1, 2, 2]
        assert (structure.n_nodes, structure.n_bottom) == (5, 3)

    def test_arrays_are_read_only(self):
        structure = tree_structure(TREE)
        for array in (structure.summing, structure.node_levels):
            with pytest.raises(ValueError):
                array[0] = 0

    def test_forest(self):
        structure = tree_structure({'north': ['n1', 'n2'], 'south': ['s1']})
        assert structure.node_keys == ('north', 'south', 'n1', 'n2', 's1')
        assert structure.summing.tolist()[:2] == [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def test_leaves_at_several_depths(self):
        assert not tree_structure(TREE).bottom_is_finest_level()
        assert tree_structure({'t': ['x', 'y'], 'x': ['x1'], 'y': ['y1']}).bottom_is_finest_level()

    def test_integer_labels(self):
        structure = tree_structure({0: [1, 2], 2: [3, 4]})
        assert structure.node_keys == (0, 2, 1, 3, 4)
        assert structure.locate([3, 9]).tolist() == [3, -1]

    @pytest.mark.parametrize(
        ('tree', 'message'),
        [
            ({}, 'non-empty mapping'),
            (['a'], 'non-empty mapping'),
            ({'a': 'bc'}, 'non-empty list'),
            ({'a': []}, 'non-empty list'),
            ({'a': ['b', 'b']}, 'duplicates'),
            ({'a': [float('nan')]}, 'missing'),
            ({None: ['b']}, 'missing'),
            ({'a': ['b'], 'c': ['b']}, 'several parents'),
            ({'a': ['b'], 'b': ['a']}, 'every parent is also a child'),
            ({'r': ['x'], 'p': ['q'], 'q': ['p']}, 'cycle through'),
        ],
    )
    def test_malformed_trees(self, tree, message):
        with pytest.raises(ValueError, match=message):
            tree_structure(tree)


class TestStructureFormats:
    """Identifiers, tags and summing frame given to hierarchicalforecast."""

    def test_ids_tags_and_summing_frame(self):
        structure = tree_structure(TREE)
        assert structure.ids.tolist() == ['0', '1', '2', '3', '4']
        assert {name: ids.tolist() for name, ids in structure.tags().items()} == {
            'depth_0': ['0'], 'depth_1': ['1', '2'], 'depth_2': ['3', '4'],
        }
        assert {name: positions.tolist() for name, positions in structure.level_positions().items()} == {
            'depth_0': [0], 'depth_1': [1, 2], 'depth_2': [3, 4],
        }
        frame = structure.summing_frame()
        assert frame.columns.tolist() == ['unique_id', '2', '3', '4']
        assert frame['unique_id'].tolist() == ['0', '1', '2', '3', '4']
        np.testing.assert_array_equal(frame[['2', '3', '4']].to_numpy(), structure.summing)


class TestParseHierarchy:
    """Both forms of ``hierarchy`` against the index levels."""

    def test_levels(self):
        spec = parse_hierarchy(
            [['country', 'region'], ['country']], ['market', 'country', 'region', 'date'], aggregate_label='All'
        )
        assert spec.kind == 'levels'
        assert spec.level_names == ('country', 'country/region')
        assert spec.node_levels == (1, 2) and spec.copy_levels == (0,)
        assert spec.copy_names == ('market',) and spec.node_names == ('country', 'region')
        assert spec.entries == ((0,), (0, 1)) and spec.aggregate_label == 'All'
        assert spec.split_key(('A', 'FR', 'north')) == (('A',), ('FR', 'north'))

    def test_tree_default_and_explicit_node_level(self):
        names = ['entity', 'node', 'date']
        default = parse_hierarchy(TREE, names)
        assert default.kind == 'tree' and default.node_levels == (1,) and default.copy_names == ('entity',)
        assert default.split_key(('A', 'b')) == (('A',), 'b')
        assert parse_hierarchy(TREE, names, node_level='entity').node_levels == (0,)
        assert parse_hierarchy(TREE, names, node_level=0).node_levels == (0,)
        assert parse_hierarchy(TREE, names, node_level=-1).node_levels == (1,)
        assert parse_hierarchy(TREE, [None, 'date']).node_levels == (0,)

    def test_integer_level_names_take_precedence_over_positions(self):
        assert parse_hierarchy(TREE, [1, 0, 'date'], node_level=0).node_levels == (1,)

    @pytest.mark.parametrize(
        ('hierarchy', 'params', 'message'),
        [
            (TREE, {'node_level': 'date'}, 'non-date index level'),
            (TREE, {'node_level': 2}, 'position'),
            ([['country']], {'node_level': 'country'}, 'node_level only applies'),
            ([['country']], {'aggregate_label': None}, 'aggregate_label'),
            ([['country']], {'aggregate_label': []}, 'aggregate_label'),
            ('country', {}, 'non-empty list'),
            ([['country'], []], {}, 'non-empty list'),
            ([['date']], {}, 'Unknown index levels'),
            ([['country'], ['region']], {}, 'finest level'),
        ],
    )
    def test_invalid(self, hierarchy, params, message):
        with pytest.raises(ValueError, match=message):
            parse_hierarchy(hierarchy, ['country', 'region', 'date'], **params)

    def test_single_date_level(self):
        with pytest.raises(ValueError, match='before the date level'):
            parse_hierarchy(TREE, ['date'])


class TestLevelsStructure:
    """Structure implied by the bottom nodes of a hierarchy of index levels."""

    @staticmethod
    def _spec(hierarchy, names=('country', 'region', 'date')) -> HierarchySpec:
        return parse_hierarchy(hierarchy, list(names))

    def test_aggregates_implied_by_the_bottom_nodes(self):
        spec = self._spec([['country'], ['country', 'region']])
        keys = [('FR', 'south'), ('DE', 'west'), ('FR', 'north'), ('FR', 'Total')]
        structure = spec.build_structure(keys)
        assert structure.node_keys == (
            ('DE', 'Total'), ('FR', 'Total'), ('DE', 'west'), ('FR', 'north'), ('FR', 'south')
        )
        assert structure.summing[:2].tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]
        assert structure.node_levels.tolist() == [0, 0, 1, 1, 1]
        assert spec.entry_of(('FR', 'Total')) == 0 and spec.entry_of(('Total', 'north')) == -1

    def test_grouped_structure(self):
        spec = self._spec(
            [['country'], ['country', 'region'], ['country', 'purpose'], ['country', 'region', 'purpose']],
            ('country', 'region', 'purpose', 'date'),
        )
        keys = [('FR', region, purpose) for region in ('n', 's') for purpose in ('w', 'l')]
        structure = spec.build_structure(keys)
        assert structure.levels == ('country', 'country/region', 'country/purpose', 'country/region/purpose')
        assert structure.n_nodes == 1 + 2 + 2 + 4
        # Chaque feuille compte une fois dans chaque partition
        np.testing.assert_array_equal(structure.summing[:5].sum(axis=0), [3.0, 3.0, 3.0, 3.0])

    def test_unmatched_rows_and_orphan_aggregates(self):
        spec = self._spec([['country'], ['country', 'region']])
        with pytest.raises(ValueError, match='match no level'):
            spec.build_structure([('FR', 'north'), ('Total', 'north')])
        with pytest.raises(ValueError, match='without any bottom row'):
            spec.build_structure([('FR', 'north'), ('DE', 'Total')])
        with pytest.raises(ValueError, match='no node of the finest level'):
            spec.build_structure([('FR', 'Total')])

    def test_unsortable_labels_keep_their_order(self):
        spec = self._spec([['country'], ['country', 'region']])
        structure = spec.build_structure([(1, 'x'), ('FR', 'y')])
        assert structure.node_keys[-2:] == ((1, 'x'), ('FR', 'y'))

    def test_tree_spec_checks_the_nodes(self):
        spec = parse_hierarchy(TREE, ['node', 'date'])
        assert spec.build_structure(['a', 'e']) is spec.tree
        with pytest.raises(ValueError, match="absent from the hierarchy: 'z'"):
            spec.build_structure(['a', 'z'])
