"""Output shape and round trip of ``StageScaler``.

Scenarios: the return type follows the configuration, and ``inverse_transform`` after
``transform`` is the identity on features and target.
"""

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objets testés
from tsforecast.frequency.stage_scaler import DEFAULT_SCALE_KEY, StageScaler

# Outils de test partagés du paquet
from tests.unit.frequency.stage_scaler.helpers import (
    _EXACT,
    _PANEL_F_BLOCKS,
    _make_scaler,
    _monthly_grid,
    _panel_grid,
)


# =============================================================================
# Règle de forme
# =============================================================================
class TestShapeRule:
    """The return type follows the CONFIGURATION, never the values."""

    @pytest.mark.parametrize(
        'setting, source, pred, with_index, expected_type',
        [
            ('constant', 'Y', 'M', False, float),
            ('constant', 'Y', 'M', True, float),
            ('constant', 'M', 'M', True, float),
            (False, 'Y', 'M', True, float),
            ('calendar', 'Y', 'M', True, pd.Series),
            ('calendar', 'M', 'M', True, pd.Series),
        ],
        ids=['constant', 'constant-with-index', 'constant-all-ones', 'false',
             'calendar', 'calendar-all-ones'],
    )
    def test_target_divisor_shape(self, setting, source, pred, with_index, expected_type):
        """Scalar under 'constant' / False, Series under 'calendar' — even when every divisor is 1.0."""
        index = _monthly_grid(periods=3) if with_index else None
        divisor = StageScaler(scale_features=setting).target_divisor(
            'v', source_freq=source, pred_freq=pred, index=index
        )
        assert type(divisor) is expected_type

    @pytest.mark.parametrize(
        'setting, grid_freq, expected_type',
        [
            ('constant', 'Y', pd.Series),
            ('constant', 'M', pd.Series),
            (False, 'Y', pd.Series),
            ('calendar', 'M', pd.DataFrame),
            ({'m1': 'calendar', DEFAULT_SCALE_KEY: False}, 'M', pd.DataFrame),
        ],
        ids=['constant', 'constant-all-ones', 'false', 'calendar-all-ones',
             'one-calendar-column-among-false'],
    )
    def test_feature_divisors_shape(self, setting, grid_freq, expected_type):
        """Series per column unless one column is in 'calendar', 1.0 values included."""
        index = _monthly_grid(periods=3)
        divisors = StageScaler(scale_features=setting).feature_divisors(
            columns=['m1', 'q1'],
            column_frequencies={'m1': 'M', 'q1': 'Q'},
            ways={'m1': 'identity', 'q1': 'interpolate'},
            grid_freq=grid_freq,
            stage_freq='M',
            index=index,
        )
        assert type(divisors) is expected_type

    def test_calendar_divisors_of_one_are_not_collapsed(self):
        """'calendar' M -> M: a DataFrame of 1.0 aligned on the grid, never a scalar (B12)."""
        index = _monthly_grid(periods=3)
        divisors = StageScaler(scale_features='calendar').feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': 'M'},
            ways={'m1': 'identity'},
            grid_freq='M',
            stage_freq='M',
            index=index,
        )
        pd.testing.assert_frame_equal(divisors, pd.DataFrame({'m1': 1.0}, index=index))

    @pytest.mark.parametrize('produced', [['Y', 'Y'], ['Y', 'Q']], ids=['uniform', 'mixed'])
    def test_produced_freq_always_gives_a_series(self, produced):
        """With produced_freq the target divisor is a Series, even when every row agrees."""
        index = pd.to_datetime(['2021-12-31', '2022-12-31'])
        divisor = StageScaler().target_divisor(
            'a1', source_freq='Y', pred_freq='M',
            produced_freq=pd.Series(produced, index=index),
        )
        assert isinstance(divisor, pd.Series)

    def test_per_entity_source_freq_always_gives_a_series(self):
        """A per-entity source_freq gives a Series, even when every block agrees."""
        index = _panel_grid()
        divisor = StageScaler().target_divisor(
            'v', source_freq={('DE',): 'Y', ('FR',): 'Y'}, pred_freq='M', index=index
        )
        pd.testing.assert_series_equal(divisor, pd.Series(12.0, index=index))

    def test_disagreeing_blocks_give_a_dataframe(self):
        """A per-entity grid_freq whose blocks disagree gives a per-row DataFrame."""
        index = _panel_grid()
        divisors = StageScaler().feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': 'M'},
            ways={'m1': 'aggregate'},
            grid_freq={('DE',): 'Q', ('FR',): 'Y'},
            stage_freq='M',
            index=index,
        )
        assert isinstance(divisors, pd.DataFrame)

    @pytest.mark.parametrize(
        'grid_freq, entities',
        [({('DE',): 'Y', ('FR',): 'Y'}, ('DE', 'FR')), ({('IT',): 'M'}, ('IT',))],
        ids=['agreeing-blocks', 'single-entry'],
    )
    def test_per_entity_grid_freq_always_gives_a_dataframe(self, grid_freq, entities):
        """A per-entity grid_freq returns a per-row DataFrame, whatever the divisors."""
        divisors = StageScaler().feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': 'M'},
            ways={'m1': 'aggregate'},
            grid_freq=grid_freq,
            stage_freq='Q',
            index=_panel_grid(entities),
        )
        assert isinstance(divisors, pd.DataFrame)

    def test_homogeneous_stage_binding_keeps_the_compact_form(self):
        """A stage binding whose entities agree on one frequency stays compact.

        Arbitration: a stage / prediction binding is compact as long as its
        entities agree on the frequency — the configuration, not the
        divisors, decides. ``grid_freq`` / ``source_freq`` mappings describe
        a mutualized training set and are always per row.
        """
        index = _panel_grid()
        binding = {('DE',): 'M', ('FR',): 'M'}
        scaler = StageScaler()

        features = scaler.feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': {('FR',): 'M', ('DE',): 'M'}},
            ways={'m1': 'aggregate'},
            grid_freq='Y',
            stage_freq=binding,
            index=index,
        )
        target = scaler.target_divisor('a1', source_freq='Y', pred_freq=binding, index=index)

        assert (type(features), features['m1'], type(target), target) == (pd.Series, 12.0, float, 12.0)

    def test_position_aliases_are_one_stage_frequency(self):
        """'M' for one entity and 'ME' for the other are one frequency: compact form."""
        index = _panel_grid()
        target = StageScaler().target_divisor(
            'a1', source_freq='Y', pred_freq={('DE',): 'M', ('FR',): 'ME'}, index=index
        )
        assert target == 12.0

    def test_disagreeing_stage_binding_is_per_row_even_with_equal_divisors(self):
        """Stage frequencies differing per entity give a DataFrame, even when every column is False.

        The divisors are all 1.0 and coincide, but the configuration varies
        per entity: the shape follows it, not the values.
        """
        divisors = StageScaler(scale_features=False).feature_divisors(
            columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
            grid_freq='Y', stage_freq={('DE',): 'Q', ('FR',): 'M'}, index=_panel_grid(),
        )
        pd.testing.assert_frame_equal(divisors, pd.DataFrame({'m1': 1.0}, index=_panel_grid()))

    def test_unread_column_binding_does_not_force_the_per_row_form(self):
        """A per-entity column frequency is not read by a materialized way: compact form.

        Only 'raw_anchors' reads the column's own frequency; aggregated onto
        the yearly grid, the column spans one year for every entity.
        """
        divisors = StageScaler().feature_divisors(
            columns=['ip'], column_frequencies={'ip': {('DE',): 'Q', ('FR',): 'M'}},
            ways={'ip': 'aggregate'}, grid_freq='Y', stage_freq='M', index=_panel_grid(),
        )
        assert (type(divisors), divisors['ip']) == (pd.Series, 12.0)


# =============================================================================
# Symétrie transform / inverse_transform
# =============================================================================
# Configurations de symétrie : (paramètres du scaler, fabrique de (X, y, produced))
def _symmetry_case(case: str):
    """Build one (scaler, X, y, produced_freq) configuration of the symmetry tests.

    Args:
        case: Configuration identifier.

    Returns:
        Tuple ``(scaler, X, y, produced_freq)``; values drawn from a seeded
        generator, negative values and NaN included.
    """
    rng = np.random.default_rng(42)
    produced = None
    if case == 'panel-fractional':
        # Panel PANEL-F réduit : blocs Y / Q / M, étape Q, diviseur 1/3 pour IT
        index = _panel_grid(('DE', 'FR', 'IT'), periods=6)
        scaler = StageScaler(
            source_freq=_PANEL_F_BLOCKS, pred_freq='Q', column_frequencies={'m1': 'M', 'q1': 'Q'},
            materialization={'q1': 'raw_anchors'},
        )
    elif case == 'calendar-daily':
        # Étape journalière du premier trimestre 2024 : diviseurs par ligne
        index = pd.date_range('2024-01-01', '2024-03-31', freq='D')
        scaler = StageScaler(
            scale_features='calendar', source_freq='Q', pred_freq='D',
            column_frequencies={'m1': 'M', 'q1': 'Q'}, materialization={'m1': 'raw_anchors', 'q1': 'raw_anchors'},
        )
    else:
        index = _monthly_grid(periods=12)
        settings = {
            'constant': 'constant',
            'dict-mixed': {'m1': 'calendar', 'q1': False, DEFAULT_SCALE_KEY: 'constant'},
            'produced-freq': 'constant',
        }[case]
        scaler = StageScaler(
            scale_features=settings, source_freq='Y', pred_freq='M',
            column_frequencies={'m1': 'M', 'q1': 'Q'}, materialization={'q1': 'raw_anchors'},
        )
        if case == 'produced-freq':
            produced = pd.Series(['Y', 'Q', 'M'] * 4, index=index)

    X = pd.DataFrame(
        {'m1': rng.normal(0.0, 50.0, len(index)), 'q1': rng.normal(100.0, 30.0, len(index))},
        index=index,
    )
    X.iloc[::5, 0] = np.nan
    y = pd.Series(rng.normal(120.0, 40.0, len(index)), index=index, name='v')
    y.iloc[1::4] = np.nan
    return scaler, X, y, produced


_SYMMETRY_CASES = ['constant', 'dict-mixed', 'calendar-daily', 'panel-fractional', 'produced-freq']


class TestTransformSymmetry:
    """``inverse_transform ∘ transform`` is the identity, on features and target."""

    @pytest.mark.parametrize('case', _SYMMETRY_CASES)
    def test_roundtrip_on_features(self, case):
        """inverse_transform(transform(X)) == X to 1e-12, NaN kept in place."""
        scaler, X, y, produced = _symmetry_case(case)
        scaler.fit(X, y, produced_freq=produced)
        pd.testing.assert_frame_equal(
            scaler.inverse_transform(scaler.transform(X)), X, rtol=_EXACT, atol=0
        )

    @pytest.mark.parametrize('case', _SYMMETRY_CASES)
    def test_roundtrip_on_target(self, case):
        """inverse_transform(transform(y)) == y to 1e-12, NaN kept in place."""
        scaler, X, y, produced = _symmetry_case(case)
        scaler.fit(X, y, produced_freq=produced)
        pd.testing.assert_series_equal(
            scaler.inverse_transform(scaler.transform(y)), y, rtol=_EXACT, atol=0
        )

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    @pytest.mark.parametrize('case', ['calendar-daily', 'panel-fractional', 'produced-freq'])
    def test_name_of_y_is_kept_under_per_row_divisors(self, case, method):
        """The name of y designates the imputed column: scaling never erases it.

        pandas drops the name of a Series divided by an unnamed per-row
        divisor; the scaler restores it.
        """
        scaler, X, y, produced = _symmetry_case(case)
        scaler.fit(X, y, produced_freq=produced)
        assert getattr(scaler, method)(y).name == 'v'

    def test_name_of_y_is_kept_under_a_scalar_divisor(self):
        """Contrast: under a scalar target divisor the name of y is kept."""
        scaler, X, y, produced = _symmetry_case('constant')
        scaler.fit(X, y, produced_freq=produced)
        assert (scaler.transform(y).name, scaler.inverse_transform(y).name) == ('v', 'v')

    @pytest.mark.parametrize('case', _SYMMETRY_CASES)
    def test_transform_introduces_no_nan(self, case):
        """Scaling never creates nor removes a missing value."""
        scaler, X, y, produced = _symmetry_case(case)
        scaler.fit(X, y, produced_freq=produced)
        assert scaler.transform(X).isna().equals(X.isna())

    def test_transform_tells_features_and_target_apart_by_type(self):
        """The same values divide by the covariate divisor as a DataFrame, by the target's as a Series."""
        index = _monthly_grid(periods=3)
        values = [120.0, 132.0, 150.0]
        X = pd.DataFrame({'m1': values}, index=index)
        y = pd.Series(values, index=index, name='m1')
        # 'm1' identité sur la grille d'étape : diviseur 1 ; cible Y -> M : diviseur 12
        scaler = _make_scaler(materialization={'m1': 'identity'}).fit(X, y)
        assert (scaler.transform(X)['m1'].tolist(), scaler.transform(y).tolist()) == (
            values, [10.0, 11.0, 12.5]
        )

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_other_types_are_rejected(self, method):
        """Neither a Series nor a DataFrame: TypeError naming the type."""
        X = pd.DataFrame({'m1': [1.0, 2.0]}, index=_monthly_grid(periods=2))
        scaler = _make_scaler().fit(X)
        with pytest.raises(TypeError, match=r"ndarray"):
            getattr(scaler, method)(np.arange(2.0))

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_transform_rejects_other_columns_than_fit(self, method):
        """sklearn convention: missing and unexpected columns are named."""
        index = _monthly_grid(periods=3)
        X = pd.DataFrame({'m1': [1.0, 2.0, 3.0], 'a2': [12.0, 24.0, 36.0]}, index=index)
        scaler = _make_scaler(materialization={'a2': 'raw_anchors'}).fit(X)
        with pytest.raises(ValueError, match=r"missing \['a2'\], unexpected \['q1'\]"):
            getattr(scaler, method)(X.rename(columns={'a2': 'q1'}))

    def test_transform_accepts_reordered_columns(self):
        """The column order is free: each column keeps its own divisor."""
        index = _monthly_grid(periods=3)
        X = pd.DataFrame({'m1': [1.0, 2.0, 3.0], 'a2': [12.0, 24.0, 36.0]}, index=index)
        scaler = _make_scaler(materialization={'a2': 'raw_anchors'}).fit(X)
        assert scaler.transform(X[['a2', 'm1']]).to_dict('list') == {'a2': [1.0, 2.0, 3.0], 'm1': [1.0, 2.0, 3.0]}
