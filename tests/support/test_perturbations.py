"""Tests of ``tests/support/perturbations.py``.

One pure function per edge case of ``CLAUDE.md`` - each test checks
that the perturbation produces exactly the shape announced by its
docstring, on small hand-built datasets and on the realistic datasets
of notebook 3 (``heterogeneous_coverage_panel``) for panel-specific
perturbations.
"""
# Manipulation de données
import pandas as pd
import pytest

from tests.support.perturbations import (
    drop_entity,
    empty_like,
    interleave_entities,
    rename_entities,
    reverse_entities,
    shuffle_rows,
    single_observation,
    to_period_end,
    to_period_index,
    to_period_start,
    to_period_start_by_column,
    to_three_level_index,
    truncate_entity,
    unsort_within_entities,
    with_duplicated_rows,
    with_index_names,
    with_single_observation_entity,
    with_special_column_names,
)


@pytest.fixture
def small_timeseries() -> pd.DataFrame:
    """Small hand-built time series, month-start anchored, 4 rows."""
    dates = pd.date_range('2020-01-01', periods=4, freq='MS')
    return pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [10.0, 20.0, 30.0, 40.0]}, index=dates)


@pytest.fixture
def small_panel() -> pd.DataFrame:
    """Small hand-built panel, two entities, MultiIndex (entity, date)."""
    dates = pd.date_range('2020-01-01', periods=2, freq='MS')
    idx = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
    return pd.DataFrame({'v': [1.0, 2.0, 3.0, 4.0]}, index=idx)


class TestShuffleRows:
    """``shuffle_rows``: pure reordering, no row lost nor gained."""

    def test_same_rows_different_order(self, small_timeseries: pd.DataFrame) -> None:
        """The set of rows (index, values) is kept, not the order."""
        shuffled = shuffle_rows(small_timeseries, seed=0)

        assert sorted(shuffled.index) == sorted(small_timeseries.index)
        assert not shuffled.index.equals(small_timeseries.index)
        for date in small_timeseries.index:
            assert (shuffled.loc[date] == small_timeseries.loc[date]).all()

    def test_reproducible_with_same_seed(self, small_timeseries: pd.DataFrame) -> None:
        """The same seed gives the same permutation."""
        first = shuffle_rows(small_timeseries, seed=1)
        second = shuffle_rows(small_timeseries, seed=1)
        pd.testing.assert_frame_equal(first, second)

    def test_does_not_mutate_input(self, small_timeseries: pd.DataFrame) -> None:
        """The input stays sorted after the call (pure function)."""
        original_index = small_timeseries.index.copy()
        shuffle_rows(small_timeseries, seed=0)
        assert small_timeseries.index.equals(original_index)


class TestReverseEntities:
    """``reverse_entities``: entity block order reversed, inner order intact."""

    def test_entity_block_order_is_reversed(self, small_panel: pd.DataFrame) -> None:
        """Entities appear in the reverse order of their first appearance."""
        reversed_panel = reverse_entities(small_panel)

        seen_order = list(dict.fromkeys(reversed_panel.index.get_level_values('entity')))
        assert seen_order == ['B', 'A']

    def test_within_entity_row_order_preserved(self, small_panel: pd.DataFrame) -> None:
        """Within each entity, the order of dates is unchanged."""
        reversed_panel = reverse_entities(small_panel)

        for entity in ('A', 'B'):
            pd.testing.assert_frame_equal(
                reversed_panel.loc[[entity]], small_panel.loc[[entity]]
            )

    def test_raises_on_non_multiindex(self, small_timeseries: pd.DataFrame) -> None:
        """A plain time series (single-level index) has no entity to reverse."""
        with pytest.raises(TypeError):
            reverse_entities(small_timeseries)

    def test_heterogeneous_coverage_panel_entity_order_is_reversed(
        self, heterogeneous_coverage_panel: pd.DataFrame
    ) -> None:
        """Property (no golden value) on the realistic dataset: same entities, reversed order."""
        reversed_panel = reverse_entities(heterogeneous_coverage_panel)
        original_entities = list(
            dict.fromkeys(heterogeneous_coverage_panel.index.get_level_values('country'))
        )
        reversed_entities = list(dict.fromkeys(reversed_panel.index.get_level_values('country')))
        assert reversed_entities == list(reversed(original_entities))
        assert len(reversed_panel) == len(heterogeneous_coverage_panel)


class TestWithSpecialColumnNames:
    """``with_special_column_names``: spaces, accents, ``/ % (`` in names."""

    def test_mapping_matches_renamed_columns(self, small_timeseries: pd.DataFrame) -> None:
        """``mapping`` maps each original name to its special replacement."""
        renamed, mapping = with_special_column_names(small_timeseries)

        assert set(mapping.keys()) == set(small_timeseries.columns)
        assert list(renamed.columns) == [mapping[col] for col in small_timeseries.columns]

    def test_special_characters_present(self) -> None:
        """At least one name contains each targeted special character (5 columns: one per suffix)."""
        df = pd.DataFrame({f'col{i}': [0] for i in range(5)})
        renamed, _ = with_special_column_names(df)
        joined = ' '.join(renamed.columns)

        for char in (' ', 'é', '/', '%', '('):
            assert char in joined

    def test_values_untouched(self, small_timeseries: pd.DataFrame) -> None:
        """Only column names change, values stay identical."""
        renamed, _ = with_special_column_names(small_timeseries)
        pd.testing.assert_frame_equal(
            renamed.set_axis(small_timeseries.columns, axis=1), small_timeseries
        )


class TestWithIndexNames:
    """``with_index_names`` : renommage de l'index simple ou de chaque niveau."""

    def test_single_index_renamed(self, small_timeseries: pd.DataFrame) -> None:
        """A single index takes the new name directly."""
        renamed = with_index_names(small_timeseries, 'periode')
        assert renamed.index.name == 'periode'

    def test_multiindex_levels_renamed(self, small_panel: pd.DataFrame) -> None:
        """Each level of a ``MultiIndex`` takes the matching name."""
        renamed = with_index_names(small_panel, ['pays', 'periode'])
        assert list(renamed.index.names) == ['pays', 'periode']


class TestToThreeLevelIndex:
    """``to_three_level_index``: a region level added above the entity."""

    def test_adds_outer_level_with_default_region(self, small_panel: pd.DataFrame) -> None:
        """Without mapping, every row gets the same default region."""
        three_level = to_three_level_index(small_panel)

        assert list(three_level.index.names) == ['region', 'entity', 'date']
        assert set(three_level.index.get_level_values('region')) == {'Zone euro'}

    def test_entity_and_date_levels_unchanged(self, small_panel: pd.DataFrame) -> None:
        """Entity and date levels keep their original values."""
        three_level = to_three_level_index(small_panel)

        assert list(three_level.index.get_level_values('entity')) == list(
            small_panel.index.get_level_values('entity')
        )
        assert list(three_level.index.get_level_values('date')) == list(
            small_panel.index.get_level_values('date')
        )

    def test_custom_region_mapping(self, small_panel: pd.DataFrame) -> None:
        """An explicit mapping assigns a different region per entity."""
        three_level = to_three_level_index(small_panel, region_by_entity={'A': 'Nord'})

        regions = dict(zip(
            three_level.index.get_level_values('entity'),
            three_level.index.get_level_values('region'),
        ))
        assert regions['A'] == 'Nord'
        assert regions['B'] == 'Zone euro'  # absente du mappage : région par défaut

    def test_raises_on_two_level_requirement(self, small_timeseries: pd.DataFrame) -> None:
        """A single-level index is not a two-level panel."""
        with pytest.raises(TypeError):
            to_three_level_index(small_timeseries)


class TestPeriodPosition:
    """``to_period_start`` / ``to_period_end`` / ``to_period_index``."""

    def test_start_to_end_moves_off_month_start(self, small_timeseries: pd.DataFrame) -> None:
        """A ``MS`` index switches to period-end dates."""
        end_anchored = to_period_end(small_timeseries)
        assert not any(date.day == 1 for date in end_anchored.index)

    def test_round_trip_recovers_month_start(self, small_timeseries: pd.DataFrame) -> None:
        """A start -> end -> start round trip falls back on the original grid."""
        round_tripped = to_period_start(to_period_end(small_timeseries))
        pd.testing.assert_index_equal(
            round_tripped.index.sort_values(), small_timeseries.index.sort_values()
        )

    def test_already_start_anchored_is_a_no_op(self, small_timeseries: pd.DataFrame) -> None:
        """Requesting period start on an already ``MS`` index changes nothing."""
        result = to_period_start(small_timeseries)
        pd.testing.assert_frame_equal(result, small_timeseries)

    def test_panel_positions_flip_per_entity(self, small_panel: pd.DataFrame) -> None:
        """The conversion applies to every entity of the panel, not only the first one."""
        end_anchored = to_period_end(small_panel)
        dates = end_anchored.index.get_level_values('date')
        assert not any(date.day == 1 for date in dates)

    def test_irregular_index_still_converts_the_common_grid(
        self, irregular_index_timeseries: pd.DataFrame
    ) -> None:
        """An irregular index (annual anchors outside the grid) is converted nonetheless.

        ``convert_position`` only needs a locally detectable step, not a fully
        regular grid: unlike ``to_period_index`` (``pd.infer_freq``, strict),
        irregularity does not block the conversion here.
        """
        result = to_period_end(irregular_index_timeseries)
        assert not any(date.day == 1 and date.hour == 0 for date in result.index)

    def test_empty_index_is_a_no_op(self, small_timeseries: pd.DataFrame) -> None:
        """An empty dataset has no date to infer the position from: documented no-op."""
        empty = small_timeseries.iloc[0:0]
        result = to_period_end(empty)
        assert len(result) == 0

    def test_to_period_index_returns_period_index(self, small_timeseries: pd.DataFrame) -> None:
        """On a regular index, the result is a ``PeriodIndex``."""
        converted = to_period_index(small_timeseries)
        assert isinstance(converted.index, pd.PeriodIndex)
        assert converted.index.freqstr == 'M'

    def test_to_period_index_on_panel_converts_date_level_only(self) -> None:
        """On a panel, only the last level (date) becomes a ``PeriodIndex``.

        ``pd.infer_freq`` needs at least 3 distinct dates: a panel with 3
        periods per entity (instead of the 2 of ``small_panel``) is required
        here.
        """
        dates = pd.date_range('2020-01-01', periods=3, freq='MS')
        idx = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
        panel = pd.DataFrame({'v': range(6)}, index=idx)

        converted = to_period_index(panel)
        assert isinstance(converted.index.get_level_values('date'), pd.PeriodIndex)
        assert list(converted.index.get_level_values('entity')) == list(
            panel.index.get_level_values('entity')
        )

    def test_to_period_index_irregular_is_a_no_op(self, irregular_index_timeseries: pd.DataFrame) -> None:
        """``pd.infer_freq`` fails on an irregular index: the conversion is skipped."""
        result = to_period_index(irregular_index_timeseries)
        assert isinstance(result.index, pd.DatetimeIndex)


class TestPeriodStartByColumn:
    """``to_period_start_by_column``: each observation at the start of its own period."""

    def test_each_frequency_moves_to_its_own_period_start(
        self, reference_timeseries: pd.DataFrame
    ) -> None:
        """Y on January 1st, Q on the quarter's first day, M on the month's first day."""
        start = to_period_start_by_column(reference_timeseries)

        # Valeurs d'or : ancres annuelles 120 / 132 / 150 au 1er janvier
        assert start['a1'].dropna().to_dict() == {
            pd.Timestamp('2021-01-01'): 120.0,
            pd.Timestamp('2022-01-01'): 132.0,
            pd.Timestamp('2023-01-01'): 150.0,
        }
        # Ancres trimestrielles au premier jour du trimestre, mêmes valeurs
        quarters = start['q1'].dropna()
        assert [date.month for date in quarters.index[:4]] == [1, 4, 7, 10]
        assert quarters.tolist() == reference_timeseries['q1'].dropna().tolist()
        # Colonne mensuelle : mêmes valeurs, au 1er du mois
        assert start['m1'].tolist() == reference_timeseries['m1'].tolist()
        assert all(date.day == 1 for date in start.index)

    def test_panel_reads_the_frequency_of_each_entity(
        self, mixed_freq_panel_multifrequency: pd.DataFrame
    ) -> None:
        """``v`` annual for FR, quarterly for DE, monthly for IT: three different starts."""
        start = to_period_start_by_column(mixed_freq_panel_multifrequency)

        first_dates = {
            entity: start.loc[entity, 'v'].dropna().index[:2].strftime('%Y-%m-%d').tolist()
            for entity in ('FR', 'DE', 'IT')
        }
        assert first_dates == {
            'FR': ['2021-01-01', '2022-01-01'],
            'DE': ['2021-01-01', '2021-04-01'],
            'IT': ['2021-01-01', '2021-02-01'],
        }
        # Aucune observation perdue ni créée
        assert start.notna().sum().equals(mixed_freq_panel_multifrequency.notna().sum())

    def test_undetectable_pair_raises(self, reference_timeseries: pd.DataFrame) -> None:
        """A pair observed once has no frequency to read its period from."""
        data = reference_timeseries.copy()
        data.loc[:'2022-12-31', 'a1'] = float('nan')

        with pytest.raises(ValueError, match='Cannot detect the frequency'):
            to_period_start_by_column(data)

    def test_does_not_mutate_input(self, reference_timeseries: pd.DataFrame) -> None:
        """The input frame is left untouched."""
        before = reference_timeseries.copy()
        to_period_start_by_column(reference_timeseries)
        pd.testing.assert_frame_equal(reference_timeseries, before)


class TestDropEntity:
    """``drop_entity``: missing entity, neither observed nor present as NaN."""

    def test_entity_rows_removed(self, small_panel: pd.DataFrame) -> None:
        """The removed entity no longer appears in the index at all."""
        dropped = drop_entity(small_panel, 'A')
        assert 'A' not in dropped.index.get_level_values('entity')
        assert set(dropped.index.get_level_values('entity')) == {'B'}

    def test_other_entities_untouched(self, small_panel: pd.DataFrame) -> None:
        """The rows of the other entities stay identical."""
        dropped = drop_entity(small_panel, 'A')
        pd.testing.assert_frame_equal(dropped.loc[['B']], small_panel.loc[['B']])

    def test_raises_on_non_multiindex(self, small_timeseries: pd.DataFrame) -> None:
        """No notion of entity on a plain time series."""
        with pytest.raises(TypeError):
            drop_entity(small_timeseries, 'A')


class TestWithDuplicatedRows:
    """``with_duplicated_rows``: duplicated index, at the end of the frame."""

    def test_length_increases_by_n(self, small_timeseries: pd.DataFrame) -> None:
        """The length grows by exactly ``n``."""
        duplicated = with_duplicated_rows(small_timeseries, n=2)
        assert len(duplicated) == len(small_timeseries) + 2

    def test_duplicated_index_values_appear_twice(self, small_timeseries: pd.DataFrame) -> None:
        """Duplicated dates appear twice in the resulting index."""
        duplicated = with_duplicated_rows(small_timeseries, n=1)
        first_date = small_timeseries.index[0]
        assert (duplicated.index == first_date).sum() == 2


class TestSingleObservation:
    """``single_observation``: dataset reduced to a single row."""

    def test_returns_one_row(self, small_timeseries: pd.DataFrame) -> None:
        """A single row, the original one, columns unchanged."""
        single = single_observation(small_timeseries)
        assert len(single) == 1
        pd.testing.assert_frame_equal(single, small_timeseries.iloc[[0]])


class TestEmptyLike:
    """``empty_like``: empty dataset, shape (columns, dtypes, index names) kept."""

    def test_zero_rows_same_columns(self, small_timeseries: pd.DataFrame) -> None:
        """Zero rows, same columns and dtypes as the original."""
        empty = empty_like(small_timeseries)
        assert len(empty) == 0
        assert list(empty.columns) == list(small_timeseries.columns)
        pd.testing.assert_series_equal(empty.dtypes, small_timeseries.dtypes)

    def test_index_name_preserved(self, small_timeseries: pd.DataFrame) -> None:
        """The index name stays set despite the absence of rows."""
        empty = empty_like(small_timeseries)
        assert empty.index.name == small_timeseries.index.name


@pytest.fixture
def panel_3x3() -> pd.DataFrame:
    """Three entities (B, A, C in this order), three monthly dates, values 0..8."""
    dates = pd.date_range('2020-01-01', periods=3, freq='MS')
    idx = pd.MultiIndex.from_product([['B', 'A', 'C'], dates], names=['entity', 'date'])
    return pd.DataFrame({'v': range(9)}, index=idx)


@pytest.fixture
def panel_region() -> pd.DataFrame:
    """Three-level panel (region, country, date): (EU, FR), (EU, DE), (AS, JP), 2 dates each."""
    dates = pd.date_range('2020-01-01', periods=2, freq='MS')
    keys = [('EU', 'FR'), ('EU', 'DE'), ('AS', 'JP')]
    idx = pd.MultiIndex.from_tuples(
        [(*key, date) for key in keys for date in dates], names=['region', 'country', 'date']
    )
    return pd.DataFrame({'v': range(6)}, index=idx)


class TestRenameEntities:
    """``rename_entities``: entity labels replaced, rows and level names kept."""

    def test_scalar_labels_with_spaces_and_accents(self, panel_3x3: pd.DataFrame) -> None:
        """Golden value: B -> 'Côte d'Ivoire' and C -> 'New York', A kept."""
        renamed = rename_entities(panel_3x3, {'B': "Côte d'Ivoire", 'C': 'New York'})

        labels = list(dict.fromkeys(renamed.index.get_level_values('entity')))
        assert labels == ["Côte d'Ivoire", 'A', 'New York']

    def test_integer_labels(self, panel_3x3: pd.DataFrame) -> None:
        """Labels can become integers (the level is then integer-typed)."""
        renamed = rename_entities(panel_3x3, {'B': 1, 'A': 2, 'C': 3})

        assert list(dict.fromkeys(renamed.index.get_level_values('entity'))) == [1, 2, 3]

    def test_rows_values_and_names_unchanged(self, panel_3x3: pd.DataFrame) -> None:
        """Only the entity labels change: values, dates, order and names are kept."""
        renamed = rename_entities(panel_3x3, {'A': 'Z'})

        assert renamed['v'].tolist() == panel_3x3['v'].tolist()
        assert renamed.index.names == panel_3x3.index.names
        assert renamed.index.get_level_values('date').equals(panel_3x3.index.get_level_values('date'))

    def test_tuple_keys_on_three_levels(self, panel_region: pd.DataFrame) -> None:
        """A multi-level key is replaced as a whole by a tuple of the same arity."""
        renamed = rename_entities(panel_region, {('EU', 'FR'): ('Zone euro', 'France')})

        keys = list(dict.fromkeys(zip(
            renamed.index.get_level_values('region'), renamed.index.get_level_values('country')
        )))
        assert keys == [('Zone euro', 'France'), ('EU', 'DE'), ('AS', 'JP')]

    def test_input_is_not_mutated(self, panel_3x3: pd.DataFrame) -> None:
        """The input panel keeps its labels."""
        before = panel_3x3.copy()
        rename_entities(panel_3x3, {'A': 'Z'})
        pd.testing.assert_frame_equal(panel_3x3, before)

    def test_unknown_entity_raises(self, panel_3x3: pd.DataFrame) -> None:
        """A key matching no entity is a mistake of the test, not a silent no-op."""
        with pytest.raises(KeyError):
            rename_entities(panel_3x3, {'X': 'Y'})

    def test_wrong_arity_raises(self, panel_region: pd.DataFrame) -> None:
        """A new key must have one label per entity level."""
        with pytest.raises(ValueError, match='expected 2'):
            rename_entities(panel_region, {('EU', 'FR'): 'France'})

    def test_merging_two_entities_raises(self, panel_3x3: pd.DataFrame) -> None:
        """Two entities cannot receive the same label (the rename is a bijection)."""
        with pytest.raises(ValueError, match='same new key'):
            rename_entities(panel_3x3, {'A': 'C'})

    def test_raises_on_non_multiindex(self, small_timeseries: pd.DataFrame) -> None:
        """No notion of entity on a plain time series."""
        with pytest.raises(TypeError):
            rename_entities(small_timeseries, {'A': 'B'})


class TestInterleaveEntities:
    """``interleave_entities``: rows ordered by date, then by entity."""

    def test_golden_order(self, panel_3x3: pd.DataFrame) -> None:
        """Entities B, A, C over three dates: values 0..8 by entity become 0,3,6,1,4,7,2,5,8."""
        assert interleave_entities(panel_3x3)['v'].tolist() == [0, 3, 6, 1, 4, 7, 2, 5, 8]

    def test_entity_rows_are_no_longer_contiguous(self, panel_3x3: pd.DataFrame) -> None:
        """No two consecutive rows belong to the same entity."""
        entities = interleave_entities(panel_3x3).index.get_level_values('entity')
        assert all(entities[i] != entities[i + 1] for i in range(len(entities) - 1))

    def test_dates_are_non_decreasing(self, panel_3x3: pd.DataFrame) -> None:
        """The date level is sorted."""
        assert interleave_entities(panel_3x3).index.get_level_values('date').is_monotonic_increasing

    def test_same_rows_as_input(self, panel_3x3: pd.DataFrame) -> None:
        """No row lost or created: the rows are a permutation of the input."""
        out = interleave_entities(panel_3x3)
        assert sorted(out.index) == sorted(panel_3x3.index)
        assert out.loc[out.index[0], 'v'] == panel_3x3.loc[out.index[0], 'v']

    def test_three_level_entities(self, panel_region: pd.DataFrame) -> None:
        """A three-level panel is interleaved entity key by entity key."""
        out = interleave_entities(panel_region)
        assert out['v'].tolist() == [0, 2, 4, 1, 3, 5]

    def test_irregular_coverage_keeps_every_row(self, heterogeneous_coverage_panel: pd.DataFrame) -> None:
        """Property on the realistic panel: same number of rows, sorted dates."""
        out = interleave_entities(heterogeneous_coverage_panel)
        assert len(out) == len(heterogeneous_coverage_panel)
        assert out.index.get_level_values(-1).is_monotonic_increasing

    def test_raises_on_non_multiindex(self, small_timeseries: pd.DataFrame) -> None:
        """No notion of entity on a plain time series."""
        with pytest.raises(TypeError):
            interleave_entities(small_timeseries)


class TestUnsortWithinEntities:
    """``unsort_within_entities``: contiguous entity blocks, permuted dates."""

    def test_entity_blocks_stay_contiguous_in_first_appearance_order(self, panel_3x3: pd.DataFrame) -> None:
        """Entities B, A, C each keep their three consecutive rows."""
        out = unsort_within_entities(panel_3x3, seed=3)
        assert list(out.index.get_level_values('entity')) == list('BBBAAACCC')

    def test_every_entity_is_out_of_date_order(self, panel_3x3: pd.DataFrame) -> None:
        """Whatever the seed, no entity block is left sorted by date."""
        for seed in range(20):
            out = unsort_within_entities(panel_3x3, seed=seed)
            for entity in 'ABC':
                dates = out.loc[entity].index
                assert not dates.is_monotonic_increasing

    def test_rows_are_a_permutation_of_the_input(self, panel_3x3: pd.DataFrame) -> None:
        """Index and values keep their pairing."""
        out = unsort_within_entities(panel_3x3, seed=1)
        pd.testing.assert_frame_equal(out.sort_index(), panel_3x3.sort_index())

    def test_reproducible_with_same_seed(self, panel_3x3: pd.DataFrame) -> None:
        """The same seed gives the same permutation."""
        pd.testing.assert_frame_equal(
            unsort_within_entities(panel_3x3, seed=5), unsort_within_entities(panel_3x3, seed=5)
        )

    def test_single_row_entity_is_untouched(self, panel_3x3: pd.DataFrame) -> None:
        """An entity with one row has nothing to permute."""
        panel = with_single_observation_entity(panel_3x3, 'A')
        out = unsort_within_entities(panel, seed=0)
        assert out.loc[['A']].equals(panel.loc[['A']])

    def test_raises_on_non_multiindex(self, small_timeseries: pd.DataFrame) -> None:
        """No notion of entity on a plain time series."""
        with pytest.raises(TypeError):
            unsort_within_entities(small_timeseries)


class TestWithSingleObservationEntity:
    """``with_single_observation_entity``: one entity reduced to its first row."""

    def test_entity_keeps_only_its_first_row(self, panel_3x3: pd.DataFrame) -> None:
        """Entity A (rows 3, 4, 5) keeps value 3, at its first date."""
        out = with_single_observation_entity(panel_3x3, 'A')

        assert out.loc[['A'], 'v'].tolist() == [3]
        assert out.loc['A'].index[0] == pd.Timestamp('2020-01-01')

    def test_other_entities_and_row_order_are_unchanged(self, panel_3x3: pd.DataFrame) -> None:
        """Rows 0-2 (B) and 6-8 (C) remain, in the original order."""
        out = with_single_observation_entity(panel_3x3, 'A')
        assert out['v'].tolist() == [0, 1, 2, 3, 6, 7, 8]

    def test_tuple_key_on_three_levels(self, panel_region: pd.DataFrame) -> None:
        """A multi-level key selects one entity."""
        out = with_single_observation_entity(panel_region, ('EU', 'DE'))
        assert out['v'].tolist() == [0, 1, 2, 4, 5]

    def test_unknown_entity_raises(self, panel_3x3: pd.DataFrame) -> None:
        """An entity absent from the panel is a mistake of the test."""
        with pytest.raises(KeyError):
            with_single_observation_entity(panel_3x3, 'X')

    def test_wrong_arity_raises(self, panel_region: pd.DataFrame) -> None:
        """A scalar key does not designate an entity of a three-level panel."""
        with pytest.raises(ValueError, match='expected 2'):
            with_single_observation_entity(panel_region, 'EU')


class TestTruncateEntity:
    """``truncate_entity``: an entity whose history stops earlier than the others."""

    def test_rows_after_end_are_dropped(self, panel_3x3: pd.DataFrame) -> None:
        """Entity A keeps January and February; B and C keep their three dates."""
        out = truncate_entity(panel_3x3, 'A', '2020-02-01')

        assert out.loc['A'].index.tolist() == [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-01')]
        assert out['v'].tolist() == [0, 1, 2, 3, 4, 6, 7, 8]

    def test_end_date_is_inclusive(self, panel_3x3: pd.DataFrame) -> None:
        """A row dated exactly ``end`` is kept."""
        out = truncate_entity(panel_3x3, 'C', pd.Timestamp('2020-03-01'))
        assert len(out) == len(panel_3x3)

    def test_end_before_first_date_removes_the_entity(self, panel_3x3: pd.DataFrame) -> None:
        """The entity is then absent from the result."""
        out = truncate_entity(panel_3x3, 'A', '2019-12-31')
        assert list(dict.fromkeys(out.index.get_level_values('entity'))) == ['B', 'C']

    def test_period_index(self, panel_3x3: pd.DataFrame) -> None:
        """The bound is read in the period type of the date level."""
        panel = to_period_index(panel_3x3)
        out = truncate_entity(panel, 'A', '2020-02')
        assert out.loc['A'].index.tolist() == [pd.Period('2020-01', 'M'), pd.Period('2020-02', 'M')]

    def test_tuple_key_on_three_levels(self, panel_region: pd.DataFrame) -> None:
        """A multi-level key truncates one entity only."""
        out = truncate_entity(panel_region, ('AS', 'JP'), '2020-01-01')
        assert out['v'].tolist() == [0, 1, 2, 3, 4]

    def test_unknown_entity_raises(self, panel_3x3: pd.DataFrame) -> None:
        """An entity absent from the panel is a mistake of the test."""
        with pytest.raises(KeyError):
            truncate_entity(panel_3x3, 'X', '2020-01-01')

    def test_realistic_panel_entity_stops_early(self, heterogeneous_coverage_panel: pd.DataFrame) -> None:
        """Property on the realistic panel: the truncated entity ends at ``end``, the others do not."""
        panel = heterogeneous_coverage_panel
        entity = panel.index.get_level_values(0)[0]
        out = truncate_entity(panel, entity, '2020-12-31')

        assert out.loc[entity].index.max() <= pd.Timestamp('2020-12-31')
        assert len(out) < len(panel)
        pd.testing.assert_frame_equal(
            out.drop(index=entity, level=0), panel.drop(index=entity, level=0)
        )
