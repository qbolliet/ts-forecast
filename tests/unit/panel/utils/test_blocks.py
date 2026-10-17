"""Unit tests for the entity-block helpers of ``tsforecast.panel.utils``.

Covers ``get_entity_mask``, ``iter_entity_blocks`` and ``build_panel_index``, then the
round trip ``iter_entity_blocks`` -> ``build_panel_index`` on sorted, unsorted,
interleaved, truncated and relabelled panels, with one, two and three entity levels.
"""
import numpy as np
import pandas as pd
import pytest

from tests.support.perturbations import (
    drop_entity,
    empty_like,
    interleave_entities,
    rename_entities,
    reverse_entities,
    to_period_index,
    to_three_level_index,
    truncate_entity,
    unsort_within_entities,
    with_duplicated_rows,
    with_single_observation_entity,
)
from tsforecast.panel.utils import (
    build_panel_index,
    get_entity_mask,
    get_unique_panel_entities,
    iter_entity_blocks,
)

ACCENTED = "Côte d'Ivoire"


def _rebuilt_index(data: pd.DataFrame) -> pd.MultiIndex:
    """Rebuild the index of ``data`` from its blocks, each row going back to its own position."""
    names = data.index.names
    keys = [None] * len(data)
    for entity, mask, block in iter_entity_blocks(data):
        rebuilt = build_panel_index(entity, block.index, names=names)
        for position, key in zip(np.flatnonzero(mask), rebuilt):
            keys[position] = key
    return pd.MultiIndex.from_tuples(keys, names=names)


class TestGetEntityMask:
    """``get_entity_mask`` selects the rows of one entity, in the order of the data."""

    def test_golden_mask_of_each_entity(self, panel_2l):
        """FR holds the first three rows, DE the last three."""
        assert get_entity_mask(panel_2l, ('FR',)).tolist() == [True, True, True, False, False, False]
        assert get_entity_mask(panel_2l, ('DE',)).tolist() == [False, False, False, True, True, True]

    def test_mask_is_a_boolean_array_of_the_length_of_the_data(self, panel_2l):
        """The mask is a boolean numpy array with one entry per row."""
        mask = get_entity_mask(panel_2l, ('FR',))
        assert (type(mask), mask.dtype, mask.shape) == (np.ndarray, np.dtype(bool), (len(panel_2l),))

    def test_three_level_entity(self, panel_3l):
        """Both entity labels must match: ``('EU', 'DE')`` is the second block, not all of ``EU``."""
        mask = get_entity_mask(panel_3l, ('EU', 'DE'))
        assert mask.tolist() == [False, False, True, True, False, False]

    def test_first_level_alone_does_not_select_a_three_level_entity(self, panel_3l):
        """``('AS', 'JP')`` selects JP only, whereas ``('EU', 'JP')`` selects nothing."""
        assert get_entity_mask(panel_3l, ('EU', 'JP')).any() == False  # noqa: E712 - numpy bool

    def test_series_input(self, panel_2l):
        """A Series with a panel index is handled like a DataFrame."""
        assert get_entity_mask(panel_2l['x'], ('DE',)).tolist() == [False] * 3 + [True] * 3

    def test_absent_entity_selects_nothing(self, panel_2l):
        """An entity that is not in the panel gives an all-False mask."""
        assert not get_entity_mask(panel_2l, ('IT',)).any()

    def test_entity_dropped_from_the_panel(self, panel_2l):
        """After removing FR, its mask is empty and DE covers every row."""
        remaining = drop_entity(panel_2l, 'FR')
        assert (get_entity_mask(remaining, ('FR',)).any(), get_entity_mask(remaining, ('DE',)).all()) == (False, True)

    def test_interleaved_rows(self, panel_2l):
        """Rows of an entity that are not consecutive are all selected."""
        mask = get_entity_mask(interleave_entities(panel_2l), ('FR',))
        assert mask.tolist() == [True, False, True, False, True, False]

    def test_unsorted_panel(self, panel_2l):
        """The mask follows the physical order of the rows."""
        shuffled = panel_2l.iloc[[4, 0, 5, 1, 3, 2]]
        assert get_entity_mask(shuffled, ('FR',)).tolist() == [False, True, False, True, False, True]

    def test_duplicated_rows_are_all_selected(self, panel_2l):
        """A row repeated at the end belongs to the same entity."""
        duplicated = with_duplicated_rows(panel_2l, n=2)
        assert get_entity_mask(duplicated, ('FR',)).sum() == 5

    def test_entity_with_one_observation(self, panel_2l):
        """An entity reduced to one row has a mask with a single True."""
        reduced = with_single_observation_entity(panel_2l, 'DE')
        assert get_entity_mask(reduced, ('DE',)).sum() == 1

    def test_one_entity_panel(self, panel_2l):
        """The only entity covers every row."""
        assert get_entity_mask(panel_2l.loc[['FR']], ('FR',)).all()

    def test_empty_panel(self, panel_2l):
        """No row gives an empty mask."""
        assert get_entity_mask(empty_like(panel_2l), ('FR',)).shape == (0,)

    def test_columns_do_not_matter(self, panel_2l):
        """One column or a ``MultiIndex`` of columns gives the same mask."""
        multi = panel_2l.copy()
        multi.columns = pd.MultiIndex.from_tuples([('a', 'x')])
        assert get_entity_mask(multi, ('DE',)).tolist() == get_entity_mask(panel_2l[['x']], ('DE',)).tolist()

    @pytest.mark.parametrize(
        'label',
        [pytest.param(ACCENTED, id='accents_and_apostrophe'),
         pytest.param('New York (%)', id='spaces_and_symbols'),
         pytest.param(10, id='integer')],
    )
    def test_labels_of_any_kind(self, panel_2l, label):
        """The entity is found whatever the type or the spelling of its label."""
        renamed = rename_entities(panel_2l, {'FR': label})
        assert get_entity_mask(renamed, (label,)).tolist() == [True] * 3 + [False] * 3

    def test_a_label_is_matched_by_value_and_type(self, panel_2l):
        """The integer ``10`` is not the string ``'10'``."""
        renamed = rename_entities(panel_2l, {'FR': 10})
        assert not get_entity_mask(renamed, ('10',)).any()

    @pytest.mark.parametrize('missing', [np.nan, None, pd.NaT], ids=['nan', 'none', 'nat'])
    def test_missing_label_in_the_key_is_refused(self, dates, missing):
        """A missing label equals nothing, itself included: it cannot designate an entity."""
        index = pd.MultiIndex.from_arrays([['A', 'A', 'B', 'B'], list(dates[:2]) * 2])
        with pytest.raises(ValueError, match='missing label'):
            get_entity_mask(pd.Series(range(4), index=index), (missing,))

    def test_missing_label_in_a_two_level_key_is_refused(self, panel_3l):
        """One missing label among several is enough to refuse the key."""
        with pytest.raises(ValueError, match='missing label'):
            get_entity_mask(panel_3l, ('EU', np.nan))

    def test_time_series_has_the_empty_entity_only(self, timeseries):
        """Without entity level the empty key selects every row; any label is refused."""
        assert get_entity_mask(timeseries, ()).all()
        with pytest.raises(ValueError, match='0 entity level'):
            get_entity_mask(timeseries, ('FR',))

    def test_masks_of_all_entities_partition_the_rows(self, heterogeneous_coverage_panel):
        """On the realistic panel each row belongs to exactly one entity."""
        masks = [get_entity_mask(heterogeneous_coverage_panel, entity)
                 for entity in get_unique_panel_entities(heterogeneous_coverage_panel)]
        assert np.sum(masks, axis=0).tolist() == [1] * len(heterogeneous_coverage_panel)

    @pytest.mark.parametrize(
        'key, n_labels',
        [
            pytest.param(('EU',), 1, id='too_short'),
            pytest.param((), 0, id='empty'),
            pytest.param(('EU', 'DE', 'something else'), 3, id='too_long'),
            pytest.param('EU', 1, id='scalar_on_two_entity_levels'),
        ],
    )
    def test_key_with_the_wrong_number_of_labels_is_refused(self, panel_3l, key, n_labels):
        """A key needs one label per entity level, no fewer and no more."""
        with pytest.raises(ValueError, match=rf'has {n_labels} label\(s\), the data has 2 entity level'):
            get_entity_mask(panel_3l, key)

    def test_empty_key_is_refused_on_a_panel(self, panel_2l):
        """An empty key cannot designate an entity of a panel."""
        with pytest.raises(ValueError, match='the data has 1 entity level'):
            get_entity_mask(panel_2l, ())

    @pytest.mark.parametrize(
        'scalar',
        [pytest.param('FR', id='string'), pytest.param(10, id='integer')],
    )
    def test_scalar_key_is_the_same_entity_as_its_tuple(self, panel_2l, scalar):
        """Like everywhere else in the module, ``'FR'`` and ``('FR',)`` designate one entity."""
        renamed = rename_entities(panel_2l, {'FR': scalar})
        assert get_entity_mask(renamed, scalar).tolist() == [True] * 3 + [False] * 3
        assert get_entity_mask(renamed, scalar).tolist() == get_entity_mask(renamed, (scalar,)).tolist()

    def test_list_key_is_read_as_a_tuple(self, panel_3l):
        """A key read from a configuration file may be a list of labels."""
        assert (get_entity_mask(panel_3l, ['EU', 'DE']).tolist()
                == get_entity_mask(panel_3l, ('EU', 'DE')).tolist())


class TestIterEntityBlocks:
    """``iter_entity_blocks`` yields ``(entity, mask, block)`` with the block indexed by date only."""

    def test_docstring_examples(self):
        """The two examples of the docstring."""
        dates = pd.date_range('2023-01-01', periods=2, freq='MS')
        df = pd.DataFrame({'x': range(4)}, index=pd.MultiIndex.from_product([['FR', 'DE'], dates]))
        assert [entity for entity, _, _ in iter_entity_blocks(df)] == [('FR',), ('DE',)]
        ts = pd.DataFrame({'x': range(2)}, index=dates)
        assert [entity for entity, _, _ in iter_entity_blocks(ts)] == [()]

    def test_golden_blocks_of_a_two_level_panel(self, panel_2l, dates):
        """FR then DE, each with its mask, its dates and its values."""
        blocks = list(iter_entity_blocks(panel_2l))

        assert [entity for entity, _, _ in blocks] == [('FR',), ('DE',)]
        assert [mask.tolist() for _, mask, _ in blocks] == [[True] * 3 + [False] * 3, [False] * 3 + [True] * 3]
        assert [block['x'].tolist() for _, _, block in blocks] == [[1, 2, 3], [4, 5, 6]]
        assert all(block.index.equals(dates.rename('date')) for _, _, block in blocks)

    def test_blocks_are_indexed_by_date_only(self, panel_3l):
        """The entity levels are dropped, on three levels as on two."""
        for _, _, block in iter_entity_blocks(panel_3l):
            assert (block.index.nlevels, block.index.name) == (1, 'date')

    def test_three_level_panel(self, panel_3l):
        """Entities of a three-level panel are pairs, in order of appearance."""
        entities = [entity for entity, _, _ in iter_entity_blocks(panel_3l)]
        assert entities == [('EU', 'FR'), ('EU', 'DE'), ('AS', 'JP')]

    def test_series_input(self, panel_2l):
        """A Series yields Series blocks."""
        blocks = [block for _, _, block in iter_entity_blocks(panel_2l['x'])]
        assert all(isinstance(block, pd.Series) for block in blocks)
        assert [block.tolist() for block in blocks] == [[1, 2, 3], [4, 5, 6]]

    def test_time_series_is_one_degenerate_block(self, timeseries):
        """The time series itself comes out under the empty entity, with a mask of all True."""
        ((entity, mask, block),) = list(iter_entity_blocks(timeseries))
        assert (entity, mask.tolist(), block is timeseries) == ((), [True] * 3, True)

    def test_time_series_with_multiindex_columns(self, dates):
        """Only the row index decides: ``MultiIndex`` columns do not make a panel."""
        columns = pd.MultiIndex.from_tuples([('a', 'x'), ('a', 'y')])
        df = pd.DataFrame(np.zeros((3, 2)), index=dates, columns=columns)
        assert [entity for entity, _, _ in iter_entity_blocks(df)] == [()]

    def test_panel_with_multiindex_columns(self, panel_2l):
        """The blocks keep the columns of the panel."""
        panel = panel_2l.copy()
        panel.columns = pd.MultiIndex.from_tuples([('a', 'x')])
        for _, _, block in iter_entity_blocks(panel):
            assert block.columns.equals(panel.columns)

    def test_unsorted_panel_keeps_the_row_order_inside_each_block(self, panel_2l):
        """Entities in order of appearance, rows as they come: the data is not sorted."""
        shuffled = panel_2l.iloc[[4, 0, 5, 1, 3, 2]]
        blocks = {entity: block for entity, _, block in iter_entity_blocks(shuffled)}

        assert list(blocks) == [('DE',), ('FR',)]
        assert blocks[('DE',)]['x'].tolist() == [5, 6, 4]
        assert blocks[('FR',)]['x'].tolist() == [1, 2, 3]

    def test_interleaved_panel(self, panel_2l):
        """Non-contiguous rows of an entity make one block, in the order of the data."""
        blocks = {entity: (mask.tolist(), block['x'].tolist())
                  for entity, mask, block in iter_entity_blocks(interleave_entities(panel_2l))}
        assert blocks == {
            ('FR',): ([True, False, True, False, True, False], [1, 2, 3]),
            ('DE',): ([False, True, False, True, False, True], [4, 5, 6]),
        }

    def test_duplicated_dates_are_kept(self, panel_2l):
        """A repeated row stays in its block."""
        blocks = {entity: len(block) for entity, _, block in iter_entity_blocks(with_duplicated_rows(panel_2l, n=1))}
        assert blocks == {('FR',): 4, ('DE',): 3}

    def test_entity_with_one_observation(self, panel_2l):
        """An entity with one observation yields a one-row block."""
        reduced = with_single_observation_entity(panel_2l, 'DE')
        assert {entity: len(block) for entity, _, block in iter_entity_blocks(reduced)} == {('FR',): 3, ('DE',): 1}

    def test_entity_that_stops_earlier(self, panel_2l):
        """Each block has the length of its own history."""
        truncated = truncate_entity(panel_2l, 'DE', '2023-02-01')
        assert {entity: len(block) for entity, _, block in iter_entity_blocks(truncated)} == {('FR',): 3, ('DE',): 2}

    def test_one_entity_one_column(self, panel_2l):
        """A single entity and a single column give one block holding that column."""
        ((entity, mask, block),) = list(iter_entity_blocks(panel_2l.loc[['DE'], ['x']]))
        assert (entity, mask.all(), block['x'].tolist()) == (('DE',), True, [4, 5, 6])

    def test_empty_panel_yields_nothing(self, panel_2l):
        """No row, no entity, no block."""
        assert list(iter_entity_blocks(empty_like(panel_2l))) == []

    def test_empty_time_series_yields_one_empty_block(self, timeseries):
        """The degenerate block of a time series exists even without rows."""
        ((entity, mask, block),) = list(iter_entity_blocks(empty_like(timeseries)))
        assert (entity, len(mask), len(block)) == ((), 0, 0)

    def test_period_dates(self, panel_2l):
        """A ``PeriodIndex`` last level gives period-indexed blocks."""
        panel = to_period_index(panel_2l)
        for _, _, block in iter_entity_blocks(panel):
            assert isinstance(block.index, pd.PeriodIndex)

    def test_forcing_time_series_on_a_panel(self, panel_2l):
        """``is_panel=False`` hands the whole frame over as a single block."""
        ((entity, mask, block),) = list(iter_entity_blocks(panel_2l, is_panel=False))
        assert (entity, mask.all(), block is panel_2l) == ((), True, True)

    def test_forcing_panel_on_a_simple_index_is_refused(self, timeseries):
        """There is no entity level to drop from a simple index."""
        with pytest.raises(ValueError, match='at least 2 levels'):
            list(iter_entity_blocks(timeseries, is_panel=True))

    def test_inferred_structure_matches_the_explicit_one(self, panel_2l):
        """``is_panel=True`` on a panel gives what the inference gives."""
        inferred = [(e, m.tolist(), b['x'].tolist()) for e, m, b in iter_entity_blocks(panel_2l)]
        explicit = [(e, m.tolist(), b['x'].tolist()) for e, m, b in iter_entity_blocks(panel_2l, is_panel=True)]
        assert inferred == explicit

    def test_blocks_of_a_panel_are_copies(self, panel_2l):
        """Changing a block leaves the panel as it was."""
        before = panel_2l.copy()
        for _, _, block in iter_entity_blocks(panel_2l):
            block['x'] = -1
        pd.testing.assert_frame_equal(panel_2l, before)

    def test_blocks_are_produced_lazily(self, panel_2l):
        """A generator: nothing is computed before the first ``next``."""
        iterator = iter_entity_blocks(panel_2l)
        assert iter(iterator) is iterator

    def test_masks_cover_every_row_once_on_the_realistic_panel(self, heterogeneous_coverage_panel):
        """Property: the masks partition the rows and each block has the size of its mask."""
        panel = heterogeneous_coverage_panel
        total = np.zeros(len(panel), dtype=int)
        for _, mask, block in iter_entity_blocks(panel):
            total += mask
            assert len(block) == mask.sum()
        assert total.tolist() == [1] * len(panel)

    def test_rows_with_a_missing_entity_label_are_refused(self, dates):
        """No row is silently left out of the blocks: a missing entity label is an error."""
        index = pd.MultiIndex.from_arrays([['A', 'A', np.nan, np.nan], list(dates[:2]) * 2])
        data = pd.Series(range(4), index=index)
        with pytest.raises(ValueError, match='must not be missing'):
            list(iter_entity_blocks(data))


class TestBuildPanelIndex:
    """``build_panel_index`` pairs an entity key with each of its dates."""

    def test_docstring_example(self):
        """The example of the docstring."""
        dates = pd.date_range('2023-01-01', periods=2, freq='MS')
        index = build_panel_index(('FR',), dates, names=['entity', 'date'])
        assert index.tolist() == [('FR', pd.Timestamp('2023-01-01')), ('FR', pd.Timestamp('2023-02-01'))]
        assert list(index.names) == ['entity', 'date']

    def test_golden_two_level_index(self, dates):
        """One entity label repeated over the three dates."""
        index = build_panel_index(('FR',), dates, names=['entity', 'date'])
        assert index.tolist() == [('FR', d) for d in dates]

    def test_three_level_index(self, dates):
        """Every label of the entity key fills its own level."""
        index = build_panel_index(('EU', 'FR'), dates[:2], names=['region', 'country', 'date'])
        assert index.tolist() == [('EU', 'FR', dates[0]), ('EU', 'FR', dates[1])]
        assert list(index.names) == ['region', 'country', 'date']

    def test_number_of_levels_is_the_entity_depth_plus_one(self, dates):
        """The index has one level per entity label, plus the date level."""
        for entity in [('a',), ('a', 'b'), ('a', 'b', 'c')]:
            assert build_panel_index(entity, dates).nlevels == len(entity) + 1

    def test_levels_keep_the_type_of_the_labels(self, dates):
        """Strings stay strings, integers stay integers, in the same key."""
        index = build_panel_index(('FR', 2), dates[:1])
        assert index.tolist() == [('FR', 2, dates[0])]
        assert index.get_level_values(1).dtype == np.dtype('int64')

    @pytest.mark.parametrize('label', [ACCENTED, 'New York (%)', 10])
    def test_labels_of_any_kind(self, dates, label):
        """Spaces, accents, symbols and integers pass through unchanged."""
        assert build_panel_index((label,), dates[:2]).get_level_values(0).tolist() == [label, label]

    def test_dates_keep_their_order_and_repetitions(self, dates):
        """Dates are paired as given: unsorted and repeated dates stay so."""
        times = pd.DatetimeIndex([dates[2], dates[0], dates[0]])
        assert build_panel_index(('FR',), times).get_level_values(-1).tolist() == list(times)

    def test_period_and_integer_times(self):
        """Any kind of time index can be paired."""
        periods = pd.period_range('2023-01', periods=2, freq='M')
        assert build_panel_index(('FR',), periods).get_level_values(-1).tolist() == list(periods)
        assert build_panel_index(('FR',), pd.RangeIndex(2)).tolist() == [('FR', 0), ('FR', 1)]

    def test_no_dates_gives_an_empty_named_index(self):
        """Without dates the index is empty but keeps its level names."""
        index = build_panel_index(('FR',), pd.DatetimeIndex([]), names=['entity', 'date'])
        assert (len(index), list(index.names)) == (0, ['entity', 'date'])

    def test_time_level_keeps_the_name_of_the_dates_by_default(self, dates):
        """Without ``names`` the entity levels are unnamed and the date level keeps its name."""
        index = build_panel_index(('EU', 'FR'), dates.rename('date'))
        assert list(index.names) == [None, None, 'date']

    def test_unnamed_dates_give_unnamed_levels(self, dates):
        """Without ``names`` and with unnamed dates, no level is named."""
        assert list(build_panel_index(('FR',), dates).names) == [None, None]

    def test_explicit_names_win_over_the_name_of_the_dates(self, dates):
        """``names`` is used as given, even when the dates carry another name."""
        index = build_panel_index(('FR',), dates.rename('date'), names=['entity', 'month'])
        assert list(index.names) == ['entity', 'month']

    def test_missing_label_in_the_key_is_refused(self, dates):
        """An entity with a missing label could not be selected afterwards."""
        with pytest.raises(ValueError, match='missing label'):
            build_panel_index((np.nan,), dates)

    def test_names_must_cover_every_level(self, dates):
        """Fewer names than levels is refused."""
        with pytest.raises(ValueError, match='Length of names must match number of levels'):
            build_panel_index(('FR',), dates, names=['entity'])

    def test_empty_entity_gives_a_one_level_index(self, dates):
        """An empty key (time series) adds no entity level: the result has the date level only."""
        index = build_panel_index((), dates[:2])
        assert (index.nlevels, index.get_level_values(0).tolist()) == (1, list(dates[:2]))

    @pytest.mark.parametrize(
        'scalar',
        [pytest.param('FR', id='string'), pytest.param(10, id='integer')],
    )
    def test_scalar_key_is_the_same_entity_as_its_tuple(self, dates, scalar):
        """Like everywhere else in the module, ``'FR'`` and ``('FR',)`` designate one entity.

        The string ``'FR'`` builds one entity level holding ``'FR'``, not one level per character.
        """
        index = build_panel_index(scalar, dates)
        assert index.nlevels == 2
        assert index.equals(build_panel_index((scalar,), dates))

    def test_list_key_is_read_as_a_tuple(self, dates):
        """A key read from a configuration file may be a list of labels."""
        assert build_panel_index(['EU', 'FR'], dates).equals(build_panel_index(('EU', 'FR'), dates))


class TestEntityBlockRoundTrip:
    """Splitting a panel into blocks and rebuilding each index gives the panel back."""

    @pytest.mark.parametrize(
        'perturb',
        [
            pytest.param(lambda df: df, id='sorted'),
            pytest.param(reverse_entities, id='entities_reversed'),
            pytest.param(lambda df: unsort_within_entities(df, seed=1), id='dates_unsorted_within_entities'),
            pytest.param(lambda df: to_three_level_index(df, region_by_entity={'FR': 'EU', 'DE': 'EU'}),
                         id='three_levels'),
            pytest.param(lambda df: rename_entities(df, {'FR': ACCENTED, 'DE': 'New York (%)'}),
                         id='accents_and_spaces'),
            pytest.param(lambda df: rename_entities(df, {'FR': 1, 'DE': 2}), id='integer_labels'),
            pytest.param(to_period_index, id='period_dates'),
            pytest.param(lambda df: truncate_entity(df, 'DE', '2023-02-01'), id='shorter_entity'),
            pytest.param(lambda df: with_single_observation_entity(df, 'DE'), id='single_observation_entity'),
        ],
    )
    def test_concatenated_blocks_give_back_the_panel(self, panel_2l, perturb):
        """Panels with contiguous entity blocks: same rows, same order, same index."""
        panel = perturb(panel_2l)
        names = panel.index.names
        rebuilt = pd.concat([
            block.set_axis(build_panel_index(entity, block.index, names=names), axis=0)
            for entity, _, block in iter_entity_blocks(panel)
        ])
        pd.testing.assert_frame_equal(rebuilt, panel)

    @pytest.mark.parametrize(
        'perturb',
        [
            pytest.param(interleave_entities, id='interleaved'),
            pytest.param(lambda df: df.iloc[[4, 0, 5, 1, 3, 2]], id='shuffled'),
            pytest.param(lambda df: with_duplicated_rows(df, n=2), id='duplicated_rows'),
        ],
    )
    def test_each_row_goes_back_to_its_own_index_entry(self, panel_2l, perturb):
        """Panels whose entity rows are not contiguous: the mask puts each key back in place."""
        panel = perturb(panel_2l)
        assert _rebuilt_index(panel).equals(panel.index)

    def test_block_index_matches_the_rows_of_its_mask(self, panel_2l):
        """For every entity, the rebuilt index equals the index of the rows of the mask."""
        panel = interleave_entities(panel_2l)
        for entity, mask, block in iter_entity_blocks(panel):
            rebuilt = build_panel_index(entity, block.index, names=panel.index.names)
            assert rebuilt.equals(panel.index[mask])

    def test_absent_entity_has_no_block_and_an_empty_mask(self, panel_2l):
        """An entity removed from the panel appears neither in the blocks nor in the masks."""
        remaining = drop_entity(panel_2l, 'FR')
        assert [entity for entity, _, _ in iter_entity_blocks(remaining)] == [('DE',)]
        assert not get_entity_mask(remaining, ('FR',)).any()

    def test_realistic_panel(self, heterogeneous_coverage_panel):
        """The three entities of the realistic panel, with their own coverage."""
        assert _rebuilt_index(heterogeneous_coverage_panel).equals(heterogeneous_coverage_panel.index)

    def test_realistic_panel_interleaved(self, heterogeneous_coverage_panel):
        """The realistic panel, interleaved by date, still gets every index entry back."""
        panel = interleave_entities(heterogeneous_coverage_panel)
        assert _rebuilt_index(panel).equals(panel.index)
