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
    reverse_entities,
    shuffle_rows,
    single_observation,
    to_period_end,
    to_period_index,
    to_period_start,
    to_three_level_index,
    with_duplicated_rows,
    with_index_names,
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
