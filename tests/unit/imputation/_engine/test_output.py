"""Tests of the output assembly and inversion helpers of ``tsforecast.imputation._engine.output``.

Scenarios: the output binding of a stage (time series, carried frequency, target frequency, stage
frequency, entity without any frequency); the input order of a grid (time series, entity blocks,
entities unknown to the input in their own blocks, empty grid, two entity levels); the provenance of a stage frame and its aggregated
cells (finer column, unmarked filled cell, empty cell, per-entity frequencies of PANEL-F, pairs
unknown to the fit); the overlay of the imputations (no production, frequency filter, priority over
the anchor, input untouched); the frame and provenance of a replayed stage on PANEL-F; the
multi-frequency stack (time series, one or two entity levels, names); the output of a pass
(prediction, stacked, flat, statistics of the target level); the merge and split of the target;
and the moves of the inversion (stacked-frame test, level selection and its fallbacks, level
removal, mask of the original cells, additive transformer, restoration of the observed values).
"""
# Modules de base
import warnings
from collections import OrderedDict

# Manipulation de données
import numpy as np
import pandas as pd

# Tests
import pytest
from sklearn.linear_model import LinearRegression

from tsforecast.imputation import HighFrequencyImputer, ProvenanceType
from tsforecast.imputation._components.covariate_materializer import CovariateMaterializer
from tsforecast.imputation._components.provenance import PROVENANCE_DTYPE, provenance_statistics
from tsforecast.imputation._engine.frequency_layout import FrequencyLayout
from tsforecast.imputation._engine.output import (
    PassOutput,
    aggregated_cells,
    assemble_output,
    build_multifreq_output,
    drop_frequency_level,
    has_frequency_level,
    in_input_order,
    invert_additive_transformer,
    keep_original_cells,
    merge_target,
    output_binding,
    overlay_imputations,
    restore_original_values,
    select_inverse_frequency_level,
    split_target,
    stage_output,
    stage_provenance,
)
from tsforecast.imputation._engine.preparation import prepare_work_frame
from tsforecast.imputation._engine.run_context import _RunContext

# Clés d'entité des jeux de référence
FR, DE, IT = ('FR',), ('DE',), ('IT',)
# Quelques fins de mois
_DATES = pd.date_range('2021-01-31', periods=3, freq='ME')


# =============================================================================
# Constructeurs
# =============================================================================
# Plan des fréquences minimal
def _layout(detected=None, target='M', *, is_panel=False) -> FrequencyLayout:
    """Build a frequency layout carrying only what the output reads.

    Args:
        detected: Detected frequencies, keyed like ``detected_frequencies_``.
        target: Effective target frequency, scalar or per entity.
        is_panel: Whether the layout describes a panel.

    Returns:
        The layout.
    """
    return FrequencyLayout(
        is_panel=is_panel, detected_frequencies=dict(detected or {}), undetected=(),
        misaligned=(), target_frequency=target, categories={}, progression=[],
    )


# Index de panel (pays, date)
def _panel_index(entities=('FR', 'DE'), dates=_DATES) -> pd.MultiIndex:
    """Build a ``(country, date)`` index over the given entities and dates."""
    return pd.MultiIndex.from_product([list(entities), dates], names=['country', 'date'])


# Provenance d'une colonne
def _marks(values, index) -> pd.DataFrame:
    """Build a one-column provenance matrix ``v`` of dtype ``PROVENANCE_DTYPE``."""
    return pd.DataFrame({'v': values}, index=index, dtype=PROVENANCE_DTYPE)


# Matérialiseur portant un miroir donné
def _materializer(values: pd.Series, frequencies: pd.Series, column='v') -> CovariateMaterializer:
    """Build a materializer whose mirror holds one column of imputations."""
    materializer = CovariateMaterializer()
    materializer.imputed_store[column] = values
    materializer.imputed_freq_store[column] = frequencies
    return materializer


# Transformateurs additifs de test
class _Doubling:
    """Transformer whose inverse halves the data."""

    def inverse_transform(self, X):
        """Halve the frame."""
        return X / 2


class _PairReturning:
    """Transformer whose inverse returns an ``(X, y)`` pair."""

    def inverse_transform(self, X):
        """Return the frame plus one, and no target."""
        return X + 1, None


class _Failing:
    """Transformer whose inverse always raises."""

    def inverse_transform(self, X):
        """Raise deliberately."""
        raise RuntimeError('deliberate inverse failure')


# =============================================================================
# Frame d'étape
# =============================================================================
class TestOutputBinding:
    """Every entity of the data is bound in the output of a stage."""

    def test_time_series_binding_is_returned_as_is(self):
        """The scalar binding of a time series is the only meaningful form."""
        assert output_binding('M', {}, _DATES, _layout()) == 'M'

    def test_left_out_entity_takes_its_last_stage(self):
        """An entity the stage leaves out keeps the frequency of its last stage."""
        layout = _layout(target={FR: 'M', DE: 'M'}, is_panel=True)
        assert output_binding({FR: 'M'}, {DE: 'Q'}, _panel_index(), layout) == {
            FR: 'M', DE: 'Q',
        }

    def test_untravelled_entity_takes_its_target(self):
        """An entity that has not travelled any stage yet takes its target frequency."""
        layout = _layout(target={FR: 'M', DE: 'ME'}, is_panel=True)
        assert output_binding({FR: 'Q'}, {}, _panel_index(), layout) == {FR: 'Q', DE: 'M'}

    def test_entity_without_target_takes_the_stage_frequency(self):
        """As a last resort, the frequency of the stage keeps the entity in the grid."""
        layout = _layout(target={FR: 'M'}, is_panel=True)
        assert output_binding({FR: 'Q'}, {}, _panel_index(), layout) == {FR: 'Q', DE: 'Q'}

    def test_entity_without_any_frequency_is_left_unbound(self):
        """An empty binding, no carried stage and no target leave nothing to bind."""
        layout = _layout(target={}, is_panel=True)
        assert output_binding({}, {}, _panel_index(), layout) == {}

    def test_binding_is_not_modified(self):
        """The completion is a new mapping: the execution binding stays as given."""
        binding = {FR: 'M'}
        output_binding(binding, {}, _panel_index(), _layout(target='M', is_panel=True))
        assert binding == {FR: 'M'}


class TestInInputOrder:
    """A stage grid comes back in the order of the input."""

    def test_time_series_dates_are_sorted(self):
        """A flat grid is sorted by date."""
        grid = pd.DatetimeIndex(_DATES[::-1])
        assert in_input_order(grid, _DATES).equals(_DATES)

    def test_entity_blocks_follow_the_input(self):
        """Entities keep their order of appearance in the input, dates sorted inside."""
        reference = _panel_index(('DE', 'FR'))
        grid = _panel_index(('FR', 'DE'))[::-1]
        ordered = in_input_order(grid, reference)
        assert ordered.equals(reference)

    def test_entity_unknown_to_the_input_comes_last(self):
        """An entity the calculator appended, absent from the input, is placed last."""
        reference = _panel_index(('DE',))
        grid = _panel_index(('IT', 'DE'))
        ordered = in_input_order(grid, reference)
        assert ordered.get_level_values(0).tolist() == ['DE'] * 3 + ['IT'] * 3

    def test_entities_unknown_to_the_input_keep_their_own_blocks(self):
        """Two entities absent from the input are never interleaved by the date sort."""
        reference = _panel_index(('DE',))
        grid = _panel_index(('IT', 'ES', 'DE'))
        ordered = in_input_order(grid, reference)
        assert ordered.get_level_values(0).tolist() == ['DE'] * 3 + ['IT'] * 3 + ['ES'] * 3
        assert ordered.get_level_values(1).equals(pd.DatetimeIndex(np.tile(_DATES, 3)))

    def test_empty_panel_grid(self):
        """An empty grid comes back empty, its levels untouched."""
        grid = _panel_index()[:0]
        ordered = in_input_order(grid, _panel_index())
        assert len(ordered) == 0 and ordered.names == ['country', 'date']

    def test_two_entity_levels(self):
        """Composite entities are ranked as normalized tuples."""
        reference = pd.MultiIndex.from_product(
            [['FR'], ['b', 'a'], _DATES[:1]], names=['country', 'sector', 'date']
        )
        grid = pd.MultiIndex.from_product([['FR'], ['a', 'b'], _DATES[:1]])
        ordered = in_input_order(grid, reference)
        assert ordered.get_level_values(1).tolist() == ['b', 'a']


class TestStageProvenance:
    """The provenance of a stage frame describes the cells the frame carries."""

    def test_finer_column_is_aggregated(self):
        """A column finer than the row carries its aggregation, marked ``AGGREGATED``."""
        layout = _layout({'v': 'M'}, target='Q')
        index = pd.to_datetime(['2021-03-31'])
        frame = pd.DataFrame({'v': [6.0]}, index=index)
        result = stage_provenance(layout, {}, _marks(['original'], index), frame, 'Q')
        assert result['v'].tolist() == [ProvenanceType.AGGREGATED]

    def test_unmarked_filled_cell_is_original(self):
        """A value no mark covers, on a column not finer than the row, is ``ORIGINAL``."""
        layout = _layout({'v': 'M'}, target='M')
        frame = pd.DataFrame({'v': [1.0]}, index=_DATES[:1])
        result = stage_provenance(layout, {}, _marks([None], _DATES[:1]), frame, 'M')
        assert result['v'].tolist() == [ProvenanceType.ORIGINAL]

    def test_empty_cell_is_never_marked(self):
        """Whatever the tracker says of the input cell, an empty cell stays unmarked."""
        layout = _layout({'v': 'Q'}, target='M')
        frame = pd.DataFrame({'v': [np.nan]}, index=_DATES[:1])
        result = stage_provenance(layout, {}, _marks(['model_on_true'], _DATES[:1]), frame, 'M')
        assert result['v'].isna().all()

    def test_imputed_marks_are_kept(self):
        """A filled cell the tracker marked keeps its mark."""
        layout = _layout({'v': 'Q'}, target='M')
        frame = pd.DataFrame({'v': [1.0]}, index=_DATES[:1])
        result = stage_provenance(layout, {}, _marks(['model_on_true'], _DATES[:1]), frame, 'M')
        assert result['v'].tolist() == [ProvenanceType.MODEL_ON_TRUE]

    def test_column_absent_from_the_tracker_is_categorical(self):
        """A column added by the reindexing, in float, is brought back to the categorical dtype."""
        layout = _layout({'v': 'M'}, target='M')
        frame = pd.DataFrame({'v': [1.0, np.nan]}, index=_DATES[:2])
        untracked = pd.DataFrame({'v': [np.nan, np.nan]}, index=_DATES[:2])
        result = stage_provenance(layout, {}, untracked, frame, 'M')
        assert result['v'].dtype == PROVENANCE_DTYPE
        assert result['v'].tolist()[0] == ProvenanceType.ORIGINAL
        assert pd.isna(result['v'].tolist()[1])


class TestAggregatedCells:
    """The cells whose column is finer than the row, per entity on a panel."""

    def test_per_entity_frequencies_of_panel_f(self):
        """At a quarterly stage, only the monthly ``v`` of IT is finer than the row."""
        layout = _layout(
            {('FR', 'v'): 'Y', ('DE', 'v'): 'Q', ('IT', 'v'): 'M'},
            target={FR: 'Q', DE: 'Q', IT: 'Q'}, is_panel=True,
        )
        index = _panel_index(('FR', 'DE', 'IT'), _DATES[2:])
        frame = pd.DataFrame({'v': [1.0, 2.0, 3.0]}, index=index)
        result = aggregated_cells(layout, {}, frame, {FR: 'Q', DE: 'Q', IT: 'Q'})
        assert result['v'].tolist() == [False, False, True]

    def test_pair_unknown_to_the_fit_reads_the_transform_frequency(self):
        """A pair the fit never observed is judged on the frequency detected at transform."""
        layout = _layout({('FR', 'v'): 'Y'}, target={FR: 'Q'}, is_panel=True)
        index = _panel_index(('FR', 'ES'), _DATES[2:])
        frame = pd.DataFrame({'v': [1.0, 2.0]}, index=index)
        result = aggregated_cells(
            layout, {('ES', 'v'): 'M'}, frame, {FR: 'Q', ('ES',): 'Q'}
        )
        assert result['v'].tolist() == [False, True]

    def test_column_without_frequency_is_never_aggregated(self):
        """A column the layout does not know is left unflagged."""
        frame = pd.DataFrame({'extra': [1.0]}, index=_DATES[:1])
        assert not aggregated_cells(_layout(), {}, frame, 'M').any().any()


class TestOverlayImputations:
    """The imputations of the mirror are laid over the stage frame."""

    def test_no_production_returns_the_frame_itself(self):
        """Without any imputation the stage frame is already the output."""
        frame = pd.DataFrame({'v': [1.0]}, index=_DATES[:1])
        assert overlay_imputations(CovariateMaterializer(), frame, 'M') is frame

    def test_only_cells_produced_at_the_row_frequency_are_laid(self):
        """A value produced at a coarser stage never lands on a finer grid."""
        materializer = _materializer(
            pd.Series([1.0, 2.0, 3.0], index=_DATES), pd.Series(['M', 'Q', 'M'], index=_DATES)
        )
        frame = pd.DataFrame({'v': [np.nan, np.nan, np.nan]}, index=_DATES)
        result = overlay_imputations(materializer, frame, 'M')
        assert result['v'].tolist()[0] == 1.0
        assert np.isnan(result['v'].tolist()[1])

    def test_imputations_win_over_the_anchor(self):
        """An anchor re-expressed at the row frequency replaces the raw total."""
        materializer = _materializer(
            pd.Series([4.0], index=_DATES[2:]), pd.Series(['M'], index=_DATES[2:])
        )
        frame = pd.DataFrame({'v': [np.nan, np.nan, 12.0], 'm1': [1.0, 2.0, 3.0]}, index=_DATES)
        result = overlay_imputations(materializer, frame, 'M')
        assert result['v'].tolist()[2] == 4.0
        # Colonne sans production : intacte
        assert result['m1'].tolist() == [1.0, 2.0, 3.0]

    def test_frame_is_not_modified(self):
        """The overlay works on a copy."""
        materializer = _materializer(
            pd.Series([4.0], index=_DATES[:1]), pd.Series(['M'], index=_DATES[:1])
        )
        frame = pd.DataFrame({'v': [np.nan]}, index=_DATES[:1])
        overlay_imputations(materializer, frame, 'M')
        assert frame['v'].isna().all()


class TestStageOutput:
    """The frame and provenance of one replayed stage, on PANEL-F."""

    @staticmethod
    def _stage(data: pd.DataFrame):
        """Replay the monthly stage of PANEL-F, then assemble its output on the full index."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer = HighFrequencyImputer(
                target_frequency='M', estimator=LinearRegression()
            ).fit(data)
            prepared = prepare_work_frame(imputer, data, fitting=False)
        run = _RunContext.for_transform(
            imputer._fitted_components_,
            window_calculator=prepared.window_calculator,
            provenance_tracker=prepared.provenance_tracker,
            transform_frequencies=prepared.new_frequencies,
            transform_misaligned=prepared.new_misaligned,
        )
        runner = imputer._stage_runner(run)
        binding = {DE: 'M', FR: 'M', IT: 'M'}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for step in imputer.imputation_plan_:
                runner.replay_step(step, X_work=prepared.X_work, stage_freq=binding)
        # Grille sans noms de niveaux, comme celle d'un masque converti
        grid = prepared.X_work.index.set_names([None, None])
        frame, provenance = stage_output(
            run, imputer._frequency_layout_, prepared.X_work, grid, binding
        )
        return frame, provenance, imputer

    def test_index_names_of_the_input_are_restored(self, mixed_freq_panel_multifrequency):
        """The grid of a converted mask carries no names; the output carries the input's."""
        frame, provenance, _ = self._stage(mixed_freq_panel_multifrequency)
        assert list(frame.index.names) == ['country', 'date']
        assert list(provenance.index.names) == ['country', 'date']

    def test_imputations_are_laid_and_described(self, mixed_freq_panel_multifrequency):
        """Every filled cell is marked, every imputed one by a model provenance."""
        frame, provenance, _ = self._stage(mixed_freq_panel_multifrequency)
        assert provenance.dtypes.eq(PROVENANCE_DTYPE).all()
        assert frame.notna().equals(provenance.notna())
        assert frame['q1'].notna().all()
        # Covariable v de FR annuelle, donc interpolée : souillure de l'étape
        assert set(provenance.loc['FR', 'q1'].unique()) == {
            ProvenanceType.MODEL_ON_INTERPOLATED.value
        }

    def test_output_matches_the_transform(self, mixed_freq_panel_multifrequency):
        """The assembled stage is the target level ``transform`` returns."""
        frame, provenance, imputer = self._stage(mixed_freq_panel_multifrequency)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            output = imputer.transform(mixed_freq_panel_multifrequency)
        pd.testing.assert_frame_equal(frame, output.xs('M', level=1), check_exact=True)
        pd.testing.assert_frame_equal(
            provenance, imputer.imputation_provenance_.xs('M', level=1), check_exact=True
        )


# =============================================================================
# Sortie d'une passe
# =============================================================================
class TestBuildMultifreqOutput:
    """The stage frames stacked on a frequency level, on the entity side."""

    def test_time_series_index(self):
        """A time series gets ``(frequency, date)``, the date level named ``'date'`` by default."""
        quarterly = pd.DataFrame({'v': [3.0]}, index=pd.DatetimeIndex(_DATES[2:]))
        monthly = pd.DataFrame({'v': [1.0, 1.0, 1.0]}, index=pd.DatetimeIndex(_DATES))
        stacked = build_multifreq_output(OrderedDict(Q=quarterly, M=monthly), is_panel=False)
        assert list(stacked.index.names) == ['frequency', 'date']
        assert stacked.index.get_level_values(0).tolist() == ['Q', 'M', 'M', 'M']
        assert stacked.index.get_level_values(0).dtype == 'str'
        assert stacked['v'].tolist() == [3.0, 1.0, 1.0, 1.0]

    def test_single_stage_level_is_a_string_level(self):
        """One stage gives one level, still in string dtype."""
        monthly = pd.DataFrame({'v': [1.0]}, index=pd.DatetimeIndex(_DATES[:1], name='when'))
        stacked = build_multifreq_output(OrderedDict(M=monthly), is_panel=False)
        assert list(stacked.index.names) == ['frequency', 'when']
        assert stacked.index.get_level_values(0).dtype == 'str'

    def test_panel_keeps_its_entity_names(self):
        """The frequency level sits between the entity and the date, names preserved."""
        frame = pd.DataFrame({'v': np.arange(6.0)}, index=_panel_index())
        stacked = build_multifreq_output(OrderedDict(M=frame), is_panel=True)
        assert list(stacked.index.names) == ['country', 'frequency', 'date']

    @pytest.mark.parametrize(
        'nlevels, expected',
        [(2, ['entity', 'frequency', 'date']), (3, ['entity_0', 'entity_1', 'frequency', 'date'])],
    )
    def test_unnamed_entity_levels_are_named(self, nlevels, expected):
        """Unnamed entity levels are named ``'entity'``, or ``'entity_<i>'`` when several."""
        levels = [['FR'], ['a']][: nlevels - 1] + [_DATES[:1]]
        frame = pd.DataFrame({'v': [1.0]}, index=pd.MultiIndex.from_product(levels))
        stacked = build_multifreq_output(OrderedDict(M=frame), is_panel=True)
        assert list(stacked.index.names) == expected


class TestAssembleOutput:
    """The output of a pass, under its three shapes."""

    @staticmethod
    def _stages():
        """Return quarterly and monthly stage frames with their provenances."""
        quarterly_index = pd.DatetimeIndex(_DATES[2:], name='date')
        monthly_index = pd.DatetimeIndex(_DATES, name='date')
        frames = OrderedDict(
            Q=pd.DataFrame({'v': [3.0]}, index=quarterly_index),
            M=pd.DataFrame({'v': [1.0, 1.0, 1.0]}, index=monthly_index),
        )
        provenances = OrderedDict(
            Q=pd.DataFrame({'v': ['original']}, index=quarterly_index, dtype=PROVENANCE_DTYPE),
            M=pd.DataFrame(
                {'v': ['model_on_true'] * 3}, index=monthly_index, dtype=PROVENANCE_DTYPE
            ),
        )
        return frames, provenances

    def test_flat_output_is_the_target_level(self):
        """Without stacking, the output is the last frame itself."""
        frames, provenances = self._stages()
        output = assemble_output(
            frames, provenances, prediction=False, keep_lower_frequencies=False,
            is_panel=False, input_index=_DATES, input_columns=pd.Index(['v']),
        )
        assert isinstance(output, PassOutput)
        assert output.data is frames['M'] and output.provenance is provenances['M']
        assert output.stacked_nlevels is None

    def test_stacked_output_records_its_depth(self):
        """Stacked, the output and its provenance share a two-level index."""
        frames, provenances = self._stages()
        output = assemble_output(
            frames, provenances, prediction=False, keep_lower_frequencies=True,
            is_panel=False, input_index=_DATES, input_columns=pd.Index(['v']),
        )
        assert output.stacked_nlevels == 2
        assert output.data.index.equals(output.provenance.index)
        assert len(output.data) == 4

    def test_statistics_describe_the_target_level(self):
        """Stacked or not, the statistics count the target level only."""
        frames, provenances = self._stages()
        output = assemble_output(
            frames, provenances, prediction=False, keep_lower_frequencies=True,
            is_panel=False, input_index=_DATES, input_columns=pd.Index(['v']),
        )
        assert output.statistics == provenance_statistics(provenances['M'])
        assert output.statistics['v']['original'] == 0

    def test_prediction_pass_keeps_the_input_rows_and_columns(self):
        """The prediction pass reindexes the target level on the input, never stacked."""
        frames, provenances = self._stages()
        input_index = pd.DatetimeIndex(_DATES[1:], name='date')
        output = assemble_output(
            frames, provenances, prediction=True, keep_lower_frequencies=True,
            is_panel=False, input_index=input_index, input_columns=pd.Index(['v', 'extra']),
        )
        assert output.stacked_nlevels is None
        assert output.data.index.equals(input_index)
        assert list(output.data.columns) == ['v', 'extra']
        # Colonne absente de l'étape : réindexée en float, ramenée au dtype catégoriel
        assert output.provenance['extra'].dtype == PROVENANCE_DTYPE
        assert output.statistics == provenance_statistics(output.provenance)


class TestMergeAndSplitTarget:
    """The target joins the working frame under its name, and leaves under its own."""

    def test_merge_without_target_is_a_copy(self):
        """Without ``y`` the features come back as a new object."""
        X = pd.DataFrame({'a': [1.0]})
        merged = merge_target(X, None, None)
        assert merged is not X and merged.equals(X)

    def test_merge_appends_the_target_column(self):
        """The target becomes the last column, under its working-frame name."""
        X = pd.DataFrame({'a': [1.0]})
        merged = merge_target(X, pd.Series([2.0], name='sales'), '__target__')
        assert list(merged.columns) == ['a', '__target__']

    def test_split_restores_the_given_name(self):
        """The target leaves the frame under the name ``y`` had."""
        data = pd.DataFrame({'a': [1.0], '__target__': [2.0]})
        X, y = split_target(data, pd.Series([0.0], name='sales'), '__target__')
        assert list(X.columns) == ['a'] and y.name == 'sales' and y.tolist() == [2.0]

    def test_split_without_target_returns_the_frame(self):
        """Without ``y``, or without its column, the frame itself comes back."""
        data = pd.DataFrame({'a': [1.0]})
        assert split_target(data, None, None) is data
        assert split_target(data, pd.Series([0.0]), 'missing') is data


# =============================================================================
# Inversion
# =============================================================================
# Frame empilée (fréquence, date)
def _stacked(labels=('Q', 'M'), names=('frequency', 'date')) -> pd.DataFrame:
    """Build a stacked frame: one row per label, all on the same date."""
    index = pd.MultiIndex.from_arrays(
        [list(labels), [_DATES[2]] * len(labels)], names=list(names)
    )
    return pd.DataFrame({'v': np.arange(float(len(labels)))}, index=index)


class TestHasFrequencyLevel:
    """A stacked frame is recognized by its depth, never by its level names."""

    def test_depth_of_the_last_stack_is_required(self):
        """Without a stacked last pass, no frame is stacked."""
        assert not has_frequency_level(_stacked(), None)

    def test_flat_frame_is_not_stacked(self):
        """A flat index is never stacked."""
        assert not has_frequency_level(pd.DataFrame({'v': [1.0]}, index=_DATES[:1]), 2)

    def test_renamed_levels_do_not_fool_the_test(self):
        """Renamed levels keep the depth; a deeper frame is not the stacked output."""
        assert has_frequency_level(_stacked(names=('x', 'y')), 2)
        assert not has_frequency_level(_stacked(), 3)


class TestSelectInverseFrequencyLevel:
    """The level of the source index first, then the target, then the last stacked."""

    def test_source_level_wins(self):
        """The level of the input index of the fit is kept when stacked."""
        assert select_inverse_frequency_level(
            _stacked(), _stacked(), stacked_nlevels=2, source_label='Q', target_frequency='M'
        ) == 'Q'

    def test_target_level_is_the_fallback(self):
        """A source level that was never a stage falls back on the target."""
        assert select_inverse_frequency_level(
            _stacked(), _stacked(), stacked_nlevels=2, source_label='W', target_frequency='M'
        ) == 'M'

    def test_last_level_with_a_warning(self):
        """Neither the source nor the target: the last stacked level, with a warning."""
        with pytest.warns(UserWarning, match="Falling back on the last stacked level 'M'"):
            label = select_inverse_frequency_level(
                _stacked(), _stacked(), stacked_nlevels=2, source_label=None,
                target_frequency='Y',
            )
        assert label == 'M'

    def test_flat_data_read_the_provenance(self):
        """Flat data to invert take the levels of a stacked provenance."""
        flat = pd.DataFrame({'v': [1.0]}, index=_DATES[2:])
        assert select_inverse_frequency_level(
            flat, _stacked(), stacked_nlevels=2, source_label='Q', target_frequency='M'
        ) == 'Q'

    def test_stacked_data_win_over_the_provenance(self):
        """Stacked data are read even when the provenance carries other levels."""
        assert select_inverse_frequency_level(
            _stacked(('M',)), _stacked(('Q', 'M')), stacked_nlevels=2, source_label='Q',
            target_frequency='M',
        ) == 'M'

    def test_nothing_stacked_selects_nothing(self):
        """Neither frame stacked: no level to select."""
        flat = pd.DataFrame({'v': [1.0]}, index=_DATES[:1])
        assert select_inverse_frequency_level(
            flat, flat, stacked_nlevels=None, source_label='M', target_frequency='M'
        ) is None


class TestDropFrequencyLevel:
    """A stacked frame reduced to one level, under the names of the input."""

    def test_no_label_or_flat_frame_is_untouched(self):
        """None, or a frame that is not stacked, comes back as is."""
        stacked = _stacked()
        assert drop_frequency_level(stacked, None, stacked_nlevels=2, index_names=['d']) is stacked
        flat = pd.DataFrame({'v': [1.0]}, index=_DATES[:1])
        assert drop_frequency_level(flat, 'M', stacked_nlevels=2, index_names=['d']) is flat

    def test_level_is_dropped_by_position_and_names_restored(self):
        """The level before the date is removed, the input names given back."""
        reduced = drop_frequency_level(
            _stacked(names=('x', 'y')), 'M', stacked_nlevels=2, index_names=['when']
        )
        assert reduced.index.name == 'when' and reduced['v'].tolist() == [1.0]

    def test_panel_keeps_its_entity_level(self):
        """On a panel the entity level stays, the frequency level goes."""
        index = pd.MultiIndex.from_arrays(
            [['FR', 'FR'], ['Q', 'M'], [_DATES[2]] * 2], names=['country', 'frequency', 'date']
        )
        stacked = pd.DataFrame({'v': [1.0, 2.0]}, index=index)
        reduced = drop_frequency_level(
            stacked, 'Q', stacked_nlevels=3, index_names=['country', 'date']
        )
        assert list(reduced.index.names) == ['country', 'date']
        assert reduced['v'].tolist() == [1.0]

    def test_level_the_last_pass_did_not_produce_raises(self):
        """A level absent from the stacked frame means data of another pass."""
        with pytest.raises(ValueError, match="The frequency level 'Y' of the data to invert"):
            drop_frequency_level(_stacked(), 'Y', stacked_nlevels=2, index_names=['date'])


class TestKeepOriginalCells:
    """Only the ``ORIGINAL`` cells survive the inversion."""

    def test_non_original_and_uncovered_cells_are_blanked(self):
        """A model mark, or a cell the provenance does not cover, is set back to NaN."""
        data = pd.DataFrame({'v': [1.0, 2.0, 3.0]}, index=_DATES)
        provenance = _marks(['original', 'model_on_true'], _DATES[:2])
        result = keep_original_cells(data, provenance)
        assert result['v'].tolist()[0] == 1.0
        assert result['v'].isna().tolist() == [False, True, True]

    def test_untracked_column_is_never_masked(self):
        """A column outside the provenance is kept whole."""
        data = pd.DataFrame({'v': [1.0], 'extra': [5.0]}, index=_DATES[:1])
        result = keep_original_cells(data, _marks(['model_on_true'], _DATES[:1]))
        assert result['extra'].tolist() == [5.0]

    def test_data_are_not_modified(self):
        """The mask builds a new frame."""
        data = pd.DataFrame({'v': [1.0]}, index=_DATES[:1])
        keep_original_cells(data, _marks(['model_on_true'], _DATES[:1]))
        assert data['v'].tolist() == [1.0]


class TestInvertAdditiveTransformer:
    """The additive transformer is undone last, never fatally."""

    def test_without_transformer_the_data_come_back(self):
        """No transformer, or one without ``inverse_transform``: nothing to undo."""
        data = pd.DataFrame({'v': [1.0]})
        assert invert_additive_transformer(None, data) is data
        assert invert_additive_transformer(object(), data) is data

    def test_inverse_is_applied(self):
        """The inverse of the fitted transformer is applied."""
        data = pd.DataFrame({'v': [4.0]})
        assert invert_additive_transformer(_Doubling(), data)['v'].tolist() == [2.0]

    def test_pair_returning_inverse_gives_its_features(self):
        """An XY transformer returns ``(X, y)``: its features are kept."""
        data = pd.DataFrame({'v': [1.0]})
        assert invert_additive_transformer(_PairReturning(), data)['v'].tolist() == [2.0]

    def test_failing_inverse_warns_and_keeps_the_data(self):
        """A failure is warned about, and the data stay as they are."""
        data = pd.DataFrame({'v': [1.0]})
        with pytest.warns(UserWarning, match='deliberate inverse failure'):
            assert invert_additive_transformer(_Failing(), data) is data


class TestRestoreOriginalValues:
    """The observed values of the last pass input win over the restored ones."""

    def test_observed_values_win(self):
        """Every cell observed in the snapshot takes its value back."""
        data = pd.DataFrame({'v': [np.nan, 2.0, 9.0]}, index=_DATES)
        snapshot = pd.DataFrame({'v': [1.0, np.nan, 3.0]}, index=_DATES)
        result = restore_original_values(data, snapshot, None, None)
        assert result['v'].tolist() == [1.0, 2.0, 3.0]
        # Jeu d'entrée intact
        assert np.isnan(data['v'].iloc[0])

    def test_target_snapshot_is_read_under_its_name(self):
        """The target of the snapshot joins it under its working-frame name."""
        data = pd.DataFrame({'v': [1.0], '__target__': [np.nan]}, index=_DATES[:1])
        snapshot = pd.DataFrame({'v': [1.0]}, index=_DATES[:1])
        result = restore_original_values(
            data, snapshot, pd.Series([7.0], index=_DATES[:1], name='sales'), '__target__'
        )
        assert result['__target__'].tolist() == [7.0]

    def test_rows_absent_from_the_snapshot_keep_their_values(self):
        """A row the snapshot does not carry is left as restored."""
        data = pd.DataFrame({'v': [5.0, 6.0]}, index=_DATES[:2])
        snapshot = pd.DataFrame({'v': [1.0]}, index=_DATES[:1])
        assert restore_original_values(data, snapshot, None, None)['v'].tolist() == [1.0, 6.0]

    def test_no_common_column_returns_the_data(self):
        """Without a shared column there is nothing to restore."""
        data = pd.DataFrame({'v': [1.0]}, index=_DATES[:1])
        snapshot = pd.DataFrame({'other': [2.0]}, index=_DATES[:1])
        assert restore_original_values(data, snapshot, None, None) is data
