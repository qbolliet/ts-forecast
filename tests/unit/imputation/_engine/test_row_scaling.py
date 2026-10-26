"""Tests of ``row_layers``, ``TrainingRows`` and ``RowScaler`` in ``tsforecast.imputation._engine.row_scaling``.

Scenarios: layer enumeration (block frequency first, rows of no layer, entities without block),
decomposition of plain and frequency-stamped grids, binding shape per data kind, a single layer
served by a single scaler call, several layers served one call each, stamped grids whose
duplicated dates are written back by position, and the PANEL-F divisors (stage Q: 4 / 1 / 1/3,
stage M: 12 / 3 / 1).
"""

# Manipulation de données
import numpy as np
import pandas as pd

# Tests
import pytest

from tsforecast.imputation._components.provenance import CELL_ORIGIN_DTYPE
from tsforecast.imputation._components.stage_scaler import StageScaler
from tsforecast.imputation._components.training_set_builder import TrainingSet
from tsforecast.imputation._engine.frequency_layout import FrequencyLayout
from tsforecast.imputation._engine.row_scaling import RowScaler, TrainingRows, row_layers

# Clés d'entité des jeux de référence
FR, DE, IT = ('FR',), ('DE',), ('IT',)
# Fin d'année partagée par une ancre annuelle et une imputation du T4
YEAR_END = pd.Timestamp('2021-12-31')


# Plan de fréquences minimal : seule la nature des données est lue
def _layout(is_panel: bool) -> FrequencyLayout:
    """Build a minimal frequency layout.

    Args:
        is_panel: Whether the data is a panel.

    Returns:
        A layout whose only meaningful field is ``is_panel``.
    """
    return FrequencyLayout(
        is_panel=is_panel,
        detected_frequencies={},
        undetected=(),
        misaligned=(),
        target_frequency={FR: 'M'} if is_panel else 'M',
        categories={},
        progression=['M'],
    )


# Lignes d'un jeu d'entraînement construit à la main
def _rows(
    index: pd.Index,
    frequencies,
    blocks,
    *,
    has_frequency_level: bool = False,
) -> TrainingRows:
    """Decompose a hand-built training set.

    Args:
        index: Training grid, stamped or not.
        frequencies: Production frequency of each row.
        blocks: Block frequency of each entity.
        has_frequency_level: Whether ``index`` carries the frequency level.

    Returns:
        The decomposed rows.
    """
    training = TrainingSet(
        X=pd.DataFrame(index=index),
        y=pd.Series(np.arange(len(index), dtype=float), index=index),
        row_frequency=pd.Series(list(frequencies), index=index, dtype='str'),
        row_origin=pd.Series('observed', index=index, dtype=CELL_ORIGIN_DTYPE),
        blocks=blocks,
        ways={},
        column_origins={},
        has_frequency_level=has_frequency_level,
    )
    return TrainingRows.of(training)


# Grille annuelle + trimestrielle d'une série, estampillée ou non
def _timeseries_axis2_rows(*, stamped: bool) -> TrainingRows:
    """Rows of ``a1`` at stage M under axis 2: three yearly anchors, two quarterly imputations.

    Args:
        stamped: Whether the grid carries the frequency level, the fourth
            quarter of 2021 then sharing its date with the 2021 anchor.

    Returns:
        The decomposed rows, in the order Y, Q, Y, Q, Y.
    """
    dates = pd.to_datetime(
        ['2021-12-31', '2021-09-30', '2022-12-31', '2022-03-31', '2023-12-31']
    )
    if stamped:
        dates = dates.where(dates != pd.Timestamp('2021-09-30'), YEAR_END)
    frequencies = ['Y', 'Q', 'Y', 'Q', 'Y']
    if stamped:
        index = pd.MultiIndex.from_arrays(
            [pd.Index(frequencies, dtype='str'), dates], names=['frequency', 'date']
        )
    else:
        index = pd.DatetimeIndex(dates, name='date')
    return _rows(index, frequencies, {(): 'Y'}, has_frequency_level=stamped)


# Metteur à l'échelle espion : enregistre chaque appel puis délègue
class _SpyStageScaler(StageScaler):
    """Stage scaler recording the keyword arguments of each divisor call."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.target_calls = []
        self.feature_calls = []

    def target_divisor(self, column, source_freq, pred_freq, index=None, produced_freq=None):
        self.target_calls.append({
            'source_freq': source_freq, 'index': index, 'produced_freq': produced_freq,
        })
        return super().target_divisor(
            column, source_freq=source_freq, pred_freq=pred_freq,
            index=index, produced_freq=produced_freq,
        )

    def feature_divisors(self, **kwargs):
        self.feature_calls.append(kwargs)
        return super().feature_divisors(**kwargs)


# Jeu PANEL-F réduit aux lignes observées de v, une couche par entité
@pytest.fixture
def panel_f_rows(mixed_freq_panel_multifrequency) -> TrainingRows:
    """Training rows of ``v`` on PANEL-F with axis 2 off: every row at its block frequency.

    Returns:
        3 yearly ``FR`` rows, 12 quarterly ``DE`` rows, 36 monthly ``IT`` rows.
    """
    index = mixed_freq_panel_multifrequency['v'].dropna().index
    blocks = {FR: 'Y', DE: 'Q', IT: 'M'}
    entity = index.get_level_values('country')
    frequencies = [blocks[(name,)] for name in entity]
    return _rows(index, frequencies, blocks)


# =============================================================================
# Énumération des couches
# =============================================================================
class TestRowLayers:
    """``row_layers``: one frequency per entity per layer, block frequency first."""

    def test_block_frequency_opens_each_entity(self):
        """FR imputed at Q under a Y block: layers [{FR: Y, DE: Q}, {FR: Q}]."""
        layers = row_layers(
            np.array([0, 0, 1]), [FR, DE],
            pd.Series(['Y', 'Q', 'Q'], dtype='str'), {FR: 'Y', DE: 'Q'},
        )
        assert [(layer, selector.tolist()) for layer, selector in layers] == [
            ({FR: 'Y', DE: 'Q'}, [True, False, True]),
            ({FR: 'Q'}, [False, True, False]),
        ]

    def test_block_first_even_when_a_finer_row_comes_first(self):
        """The order of the rows never moves the block frequency off rank 0."""
        layers = row_layers(
            np.array([0, 0]), [FR], pd.Series(['Q', 'Y'], dtype='str'), {FR: 'Y'}
        )
        assert [layer for layer, _ in layers] == [{FR: 'Y'}, {FR: 'Q'}]

    def test_rows_at_block_frequency_make_a_single_layer(self):
        """Axis 2 injecting nothing: exactly one layer, equal to the blocks."""
        layers = row_layers(
            np.array([0, 1, 2, 2]), [FR, DE, IT],
            pd.Series(['Y', 'Q', 'M', 'M'], dtype='str'), {FR: 'Y', DE: 'Q', IT: 'M'},
        )
        assert len(layers) == 1
        layer, selector = layers[0]
        assert layer == {FR: 'Y', DE: 'Q', IT: 'M'}
        assert selector.all()

    def test_missing_frequency_belongs_to_no_layer(self):
        """A row without production frequency is in no selector."""
        layers = row_layers(
            np.array([0, 0]), [FR], pd.Series(['Y', np.nan], dtype='str'), {FR: 'Y'}
        )
        assert [(layer, selector.tolist()) for layer, selector in layers] == [
            ({FR: 'Y'}, [True, False]),
        ]

    def test_empty_block_layer_is_dropped(self):
        """Finer rows alone: the block layer, without rows, is not listed."""
        layers = row_layers(
            np.array([0, 0]), [FR], pd.Series(['Q', 'Q'], dtype='str'), {FR: 'Y'}
        )
        assert [layer for layer, _ in layers] == [{FR: 'Q'}]

    def test_entity_without_block_reads_its_rows(self):
        """An entity absent from the blocks enters with its row frequencies only."""
        layers = row_layers(
            np.array([0, 1]), [FR, DE], pd.Series(['Y', 'Q'], dtype='str'), {FR: 'Y'}
        )
        assert [layer for layer, _ in layers] == [{FR: 'Y', DE: 'Q'}]

    def test_empty_grid_has_no_layer(self):
        """No row, no layer."""
        layers = row_layers(
            np.array([], dtype=np.intp), [], pd.Series([], dtype='str'), {FR: 'Y'}
        )
        assert layers == []


# =============================================================================
# Décomposition des lignes
# =============================================================================
class TestTrainingRows:
    """``TrainingRows.of``: real grid, entities and layers decomposed once."""

    def test_plain_panel_grid_is_its_own_real_grid(self, panel_f_rows):
        """Without frequency level, the real grid is the training grid itself."""
        assert panel_f_rows.real_index.equals(panel_f_rows.row_frequency.index)
        # Entités dans l'ordre de première apparition des lignes
        appearance = panel_f_rows.real_index.get_level_values('country').unique()
        assert panel_f_rows.entities == [(name,) for name in appearance]
        assert sorted(panel_f_rows.entities) == sorted([FR, DE, IT])
        assert panel_f_rows.blocks == {FR: 'Y', DE: 'Q', IT: 'M'}

    def test_stamped_grid_drops_the_frequency_level(self):
        """Stamped time series: the real grid is the date level, duplicates kept."""
        rows = _timeseries_axis2_rows(stamped=True)
        assert isinstance(rows.real_index, pd.DatetimeIndex)
        assert list(rows.real_index[:2]) == [YEAR_END, YEAR_END]
        assert rows.entities == [()]
        assert rows.entity_codes.tolist() == [0] * 5

    def test_layers_are_those_of_row_layers(self):
        """The layers held are the enumeration of the decomposed grid."""
        rows = _timeseries_axis2_rows(stamped=True)
        assert [(layer, selector.tolist()) for layer, selector in rows.layers] == [
            ({(): 'Y'}, [True, False, True, False, True]),
            ({(): 'Q'}, [False, True, False, True, False]),
        ]

    def test_blocks_are_copied(self):
        """The blocks held are a copy: the training set is never aliased."""
        blocks = {(): 'Y'}
        rows = _rows(pd.DatetimeIndex([YEAR_END]), ['Y'], blocks)
        assert rows.blocks == blocks and rows.blocks is not blocks


# =============================================================================
# Liaison d'une couche
# =============================================================================
class TestLayerBinding:
    """``RowScaler.layer_binding``: the shape :class:`StageScaler` reads."""

    def test_time_series_gets_the_scalar_frequency(self):
        """A time series has no entity level: its single block is a string."""
        scaler = RowScaler(StageScaler(), _layout(is_panel=False))
        assert scaler.layer_binding({(): 'Q'}) == 'Q'

    def test_panel_gets_a_copy_of_the_mapping(self):
        """A panel reads one frequency per entity, never the caller's object."""
        layer = {FR: 'Y', DE: 'Q'}
        binding = RowScaler(StageScaler(), _layout(is_panel=True)).layer_binding(layer)
        assert binding == layer and binding is not layer

    def test_single_entity_panel_keeps_the_mapping(self):
        """One entity on a panel is still a mapping: its grid has an entity level."""
        binding = RowScaler(StageScaler(), _layout(is_panel=True)).layer_binding({FR: 'Y'})
        assert binding == {FR: 'Y'}


# =============================================================================
# Une seule couche : un seul appel
# =============================================================================
class TestSingleLayer:
    """A grid where axis 2 injected nothing is a single layer, hence a single call."""

    def test_one_target_call_at_the_block_binding(self, panel_f_rows):
        """One ``target_divisor`` call, on the whole grid, at the blocks."""
        spy = _SpyStageScaler()
        RowScaler(spy, _layout(is_panel=True)).target_divisors(
            panel_f_rows, column='v', stage_freq='M'
        )
        assert len(spy.target_calls) == 1
        call = spy.target_calls[0]
        assert call['source_freq'] == {FR: 'Y', DE: 'Q', IT: 'M'}
        assert call['index'].equals(panel_f_rows.real_index)

    def test_one_feature_call_at_the_block_binding(self, panel_f_rows):
        """One ``feature_divisors`` call, ``grid_freq`` being the blocks."""
        spy = _SpyStageScaler()
        RowScaler(spy, _layout(is_panel=True)).feature_divisors(
            panel_f_rows,
            feature_cols=['m1'],
            freqs_by_column={'m1': 'M'},
            ways={'m1': 'aggregate'},
            stage_freq='M',
        )
        assert len(spy.feature_calls) == 1
        assert spy.feature_calls[0]['grid_freq'] == {FR: 'Y', DE: 'Q', IT: 'M'}

    def test_divisors_equal_a_direct_scaler_call(self, panel_f_rows):
        """A single layer gives exactly what the scaler gives on the whole grid."""
        scaler = StageScaler()
        row_scaler = RowScaler(scaler, _layout(is_panel=True))
        features = row_scaler.feature_divisors(
            panel_f_rows,
            feature_cols=['m1', 'q1'],
            freqs_by_column={'m1': 'M', 'q1': 'Q'},
            ways={'m1': 'aggregate', 'q1': 'interpolate'},
            stage_freq='Q',
        )
        direct = scaler.feature_divisors(
            columns=['m1', 'q1'],
            column_frequencies={'m1': 'M', 'q1': 'Q'},
            ways={'m1': 'aggregate', 'q1': 'interpolate'},
            grid_freq=dict(panel_f_rows.blocks),
            stage_freq='Q',
            index=panel_f_rows.real_index,
        )
        pd.testing.assert_frame_equal(features, direct, check_exact=True)

    def test_time_series_scalar_divisors_are_spread(self):
        """Homogeneous time series: the per-column scalar fills every row."""
        rows = _rows(pd.DatetimeIndex(pd.to_datetime(['2021-12-31', '2022-12-31'])),
                     ['Y', 'Y'], {(): 'Y'})
        divisors = RowScaler(StageScaler(), _layout(is_panel=False)).feature_divisors(
            rows,
            feature_cols=['m1', 'q1'],
            freqs_by_column={'m1': 'M', 'q1': 'Q'},
            ways={'m1': 'aggregate', 'q1': 'aggregate'},
            stage_freq='M',
        )
        assert divisors.to_dict('list') == {'m1': [12.0, 12.0], 'q1': [12.0, 12.0]}
        assert divisors.index.equals(rows.row_frequency.index)


# =============================================================================
# Plusieurs couches : un appel par couche
# =============================================================================
class TestSeveralLayers:
    """Axis 2 on: each layer is served by its own call, at its own frequency."""

    def test_target_divisor_per_layer(self):
        """``a1`` at stage M: yearly anchors divided by 12, quarterly rows by 3."""
        spy = _SpyStageScaler()
        rows = _timeseries_axis2_rows(stamped=False)
        divisors = RowScaler(spy, _layout(is_panel=False)).target_divisors(
            rows, column='a1', stage_freq='M'
        )
        assert divisors.tolist() == [12.0, 3.0, 12.0, 3.0, 12.0]
        assert [call['source_freq'] for call in spy.target_calls] == ['Y', 'Q']

    def test_feature_divisor_per_layer(self):
        """A covariate on a quarterly row spans a quarter, never the year."""
        spy = _SpyStageScaler()
        rows = _timeseries_axis2_rows(stamped=False)
        divisors = RowScaler(spy, _layout(is_panel=False)).feature_divisors(
            rows,
            feature_cols=['m1'],
            freqs_by_column={'m1': 'M'},
            ways={'m1': 'aggregate'},
            stage_freq='M',
        )
        assert divisors['m1'].tolist() == [12.0, 3.0, 12.0, 3.0, 12.0]
        assert [call['grid_freq'] for call in spy.feature_calls] == ['Y', 'Q']

    def test_panel_layer_only_holds_the_entities_reaching_its_rank(self):
        """PANEL-F, FR imputed at Q: the second call reads FR alone, on FR rows."""
        spy = _SpyStageScaler()
        index = pd.MultiIndex.from_tuples(
            [('FR', YEAR_END), ('FR', pd.Timestamp('2021-09-30')), ('DE', YEAR_END)],
            names=['country', 'date'],
        )
        rows = _rows(index, ['Y', 'Q', 'Q'], {FR: 'Y', DE: 'Q'})
        divisors = RowScaler(spy, _layout(is_panel=True)).target_divisors(
            rows, column='v', stage_freq='M'
        )
        assert divisors.tolist() == [12.0, 3.0, 3.0]
        assert [call['source_freq'] for call in spy.target_calls] == [
            {FR: 'Y', DE: 'Q'}, {FR: 'Q'},
        ]
        assert spy.target_calls[1]['index'].equals(index[[1]])

    def test_row_of_no_layer_keeps_a_missing_divisor(self):
        """A row without production frequency is never divided."""
        rows = _rows(pd.DatetimeIndex(pd.to_datetime(['2021-12-31', '2022-12-31'])),
                     ['Y', np.nan], {(): 'Y'})
        divisors = RowScaler(StageScaler(), _layout(is_panel=False)).target_divisors(
            rows, column='a1', stage_freq='M'
        )
        assert divisors.iloc[0] == 12.0 and np.isnan(divisors.iloc[1])

    def test_exempt_target_divides_by_one(self):
        """``scale_features`` False for the imputed column: every divisor is 1.0."""
        rows = _timeseries_axis2_rows(stamped=False)
        divisors = RowScaler(
            StageScaler(scale_features={'a1': False}), _layout(is_panel=False)
        ).target_divisors(rows, column='a1', stage_freq='M')
        assert divisors.tolist() == [1.0] * 5


# =============================================================================
# Grille estampillée : dates dupliquées, réécriture positionnelle
# =============================================================================
class TestStampedGrid:
    """Coincident cells share a date: the scaler sees duplicate-free sub-grids."""

    def test_target_divisors_written_back_by_position(self):
        """The 2021 anchor and the fourth quarter keep 12 and 3, though sharing a date."""
        rows = _timeseries_axis2_rows(stamped=True)
        divisors = RowScaler(StageScaler(), _layout(is_panel=False)).target_divisors(
            rows, column='a1', stage_freq='M'
        )
        assert divisors.index.equals(rows.row_frequency.index)
        assert divisors.tolist() == [12.0, 3.0, 12.0, 3.0, 12.0]

    def test_feature_divisors_written_back_by_position(self):
        """Same date, two scales on one column: the per-row frame carries both."""
        rows = _timeseries_axis2_rows(stamped=True)
        divisors = RowScaler(StageScaler(), _layout(is_panel=False)).feature_divisors(
            rows,
            feature_cols=['m1', 'q1'],
            freqs_by_column={'m1': 'M', 'q1': 'Q'},
            ways={'m1': 'aggregate', 'q1': 'aggregate'},
            stage_freq='M',
        )
        assert divisors.index.equals(rows.row_frequency.index)
        assert divisors.loc[('Y', YEAR_END)].tolist() == [12.0, 12.0]
        assert divisors.loc[('Q', YEAR_END)].tolist() == [3.0, 3.0]

    def test_scaler_never_sees_a_duplicated_date(self):
        """Each call reads the real ``(date)`` sub-grid of its layer, without duplicates."""
        spy = _SpyStageScaler()
        rows = _timeseries_axis2_rows(stamped=True)
        row_scaler = RowScaler(spy, _layout(is_panel=False))
        row_scaler.target_divisors(rows, column='a1', stage_freq='M')
        row_scaler.feature_divisors(
            rows, feature_cols=['m1'], freqs_by_column={'m1': 'M'},
            ways={'m1': 'aggregate'}, stage_freq='M',
        )
        indexes = [call['index'] for call in spy.target_calls + spy.feature_calls]
        assert len(indexes) == 4
        for index in indexes:
            assert isinstance(index, pd.DatetimeIndex)
            assert not index.has_duplicates

    def test_stamped_panel_grid(self):
        """Panel ``(country, frequency, date)``: the FR fourth quarter keeps its own divisor."""
        index = pd.MultiIndex.from_arrays(
            [
                pd.Index(['FR', 'FR', 'DE'], dtype='str'),
                pd.Index(['Y', 'Q', 'Q'], dtype='str'),
                pd.DatetimeIndex([YEAR_END] * 3),
            ],
            names=['country', 'frequency', 'date'],
        )
        rows = _rows(index, ['Y', 'Q', 'Q'], {FR: 'Y', DE: 'Q'}, has_frequency_level=True)
        divisors = RowScaler(StageScaler(), _layout(is_panel=True)).target_divisors(
            rows, column='v', stage_freq='M'
        )
        assert divisors.index.equals(index)
        assert divisors.tolist() == [12.0, 3.0, 3.0]


# =============================================================================
# PANEL-F : diviseurs de v par entité
# =============================================================================
class TestPanelF:
    """PANEL-F, ``v`` yearly for FR, quarterly for DE, monthly for IT."""

    @pytest.mark.parametrize(
        'stage, expected',
        [
            ('Q', {'FR': 4.0, 'DE': 1.0, 'IT': 1.0 / 3.0}),
            ('M', {'FR': 12.0, 'DE': 3.0, 'IT': 1.0}),
        ],
    )
    def test_target_divisors(self, panel_f_rows, stage, expected):
        """One divisor per entity, fractional for a block finer than the stage."""
        divisors = RowScaler(StageScaler(), _layout(is_panel=True)).target_divisors(
            panel_f_rows, column='v', stage_freq=stage
        )
        by_entity = divisors.groupby(level='country').unique()
        assert {name: values.tolist() for name, values in by_entity.items()} == {
            name: [value] for name, value in expected.items()
        }

    @pytest.mark.parametrize(
        'stage, expected',
        [
            ('Q', {'FR': 4.0, 'DE': 1.0, 'IT': 1.0 / 3.0}),
            ('M', {'FR': 12.0, 'DE': 3.0, 'IT': 1.0}),
        ],
    )
    def test_feature_divisors_follow_the_block_grid(self, panel_f_rows, stage, expected):
        """``m1`` aggregated on each block spans one block period."""
        divisors = RowScaler(StageScaler(), _layout(is_panel=True)).feature_divisors(
            panel_f_rows,
            feature_cols=['m1'],
            freqs_by_column={'m1': 'M'},
            ways={'m1': 'aggregate'},
            stage_freq=stage,
        )
        by_entity = divisors['m1'].groupby(level='country').unique()
        assert {name: values.tolist() for name, values in by_entity.items()} == {
            name: [value] for name, value in expected.items()
        }
