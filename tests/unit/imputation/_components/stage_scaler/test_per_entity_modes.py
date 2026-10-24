"""``scale_features`` given as ``{column: {entity: mode}}``.

Scenarios: every per-entity mode combination on single- and two-level panels.
"""

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Sklearn
from sklearn.base import clone

# Objets testés
from tsforecast.frequency.stage_scaler import DEFAULT_SCALE_KEY, StageScaler

# Perturbations partagées
from tests.support.perturbations import shuffle_rows

# Outils de test partagés du paquet
from tests.unit.frequency.stage_scaler.helpers import _EXACT, _monthly_grid, _panel_grid


# =============================================================================
# Modalités par entité
# =============================================================================
# Réglage de référence : 'v' brute pour DE, CVS pour toutes les autres entités
_DE_RAW = {'v': {'DE': 'calendar', DEFAULT_SCALE_KEY: 'constant'}}


# Grille de panel à deux niveaux d'entité (zone, pays, date)
def _two_level_grid(periods: int = 2) -> pd.MultiIndex:
    """Build a month-end grid with two entity levels.

    Args:
        periods: Number of months per entity.

    Returns:
        A ``MultiIndex`` (``zone``, ``country``, ``date``): entities
        ``('Zone euro', 'DE')`` then ``('Zone euro', 'FR')``.
    """
    return pd.MultiIndex.from_product(
        [['Zone euro'], ['DE', 'FR'], _monthly_grid(periods=periods)],
        names=['zone', 'country', 'date'],
    )


class TestPerEntityModes:
    """``scale_features`` as ``{column: {entity: mode}}``.

    Golden values: a monthly variable at a daily stage divides by 30.0 under
    ``'constant'`` and by the real days of each month under ``'calendar'``
    (January / February 2021: 31 / 28); a quarterly one by 91.0 under
    ``'constant'``.
    """

    # --- Validation ----------------------------------------------------------
    @pytest.mark.parametrize(
        'setting',
        [
            _DE_RAW,
            {'v': {'DE': 'calendar'}},
            {'v': {('Zone euro', 'DE'): 'calendar'}},
            {DEFAULT_SCALE_KEY: {'IT': 'calendar', DEFAULT_SCALE_KEY: 'constant'}},
            {'v': {'DE': 'calendar'}, 'q1': False},
        ],
        ids=['nested-default', 'no-nested-default', 'two-level-key', 'top-default', 'mixed-columns'],
    )
    def test_admissible_nested_settings_are_stored_untouched(self, setting):
        """The nested form passes the validation and is stored as the same object."""
        assert StageScaler(scale_features=setting).scale_features is setting

    @pytest.mark.parametrize(
        'setting',
        [
            {'v': {'DE': False}},
            {'v': {'DE': 'calendar', DEFAULT_SCALE_KEY: False}},
            {DEFAULT_SCALE_KEY: {'DE': False}},
        ],
        ids=['entity', 'nested-default', 'top-default'],
    )
    def test_false_within_a_nested_dict_raises(self, setting):
        """False applies to a whole column, never to some of its entities."""
        with pytest.raises(ValueError, match=r"never to some of its entities"):
            StageScaler(scale_features=setting)

    @pytest.mark.parametrize(
        'setting, message',
        [
            ({'v': {}}, r"per-entity dict cannot be empty"),
            ({'v': {'DE': 'exact'}}, r"only admits \('constant', 'calendar'\)"),
            ({'v': {'DE': np.False_}}, r"only admits"),
            ({'v': {'DE': 0}}, r"only admits"),
            ({'v': {'DE': {'x': 'calendar'}}}, r"only admits"),
        ],
        ids=['empty', 'unknown-literal', 'numpy-false', 'zero', 'third-level'],
    )
    def test_inadmissible_nested_values_raise(self, setting, message):
        """A nested dict holds 'constant' or 'calendar' only, and is never empty."""
        with pytest.raises(ValueError, match=message):
            StageScaler(scale_features=setting)

    def test_false_within_a_nested_dict_raises_after_set_params(self):
        """set_params bypasses __init__: the resolution validates again."""
        scaler = StageScaler().set_params(scale_features={'v': {'DE': False}})
        with pytest.raises(ValueError, match=r"never to some of its entities"):
            scaler.resolve_mode('v', 'DE')

    # --- Résolution -----------------------------------------------------------
    @pytest.mark.parametrize(
        'setting, column, entity, expected',
        [
            (_DE_RAW, 'v', 'DE', 'calendar'),
            (_DE_RAW, 'v', ('DE',), 'calendar'),
            (_DE_RAW, 'v', 'FR', 'constant'),
            (_DE_RAW, 'v', None, 'constant'),
            (_DE_RAW, 'q1', 'DE', 'constant'),
            ({'v': {'DE': 'constant'}, DEFAULT_SCALE_KEY: 'calendar'}, 'v', 'FR', 'constant'),
            ({'v': {'DE': 'constant'}, DEFAULT_SCALE_KEY: False}, 'v', 'FR', 'constant'),
            ({'v': {'DE': 'constant'}, DEFAULT_SCALE_KEY: 'calendar'}, 'q1', 'FR', 'calendar'),
            ({DEFAULT_SCALE_KEY: {'IT': 'calendar'}}, 'q1', 'IT', 'calendar'),
            ({DEFAULT_SCALE_KEY: {'IT': 'calendar'}, 'v': False}, 'v', 'IT', False),
            ({'v': {('DE',): 'calendar'}}, 'v', 'DE', 'calendar'),
        ],
        ids=['own-entity', 'tuple-entity', 'nested-default', 'no-entity', 'other-column',
             'closed-on-calendar-default', 'closed-on-false-default', 'top-default-elsewhere',
             'nested-top-default', 'column-scalar-wins', 'tuple-key-scalar-entity'],
    )
    def test_resolve_mode(self, setting, column, entity, expected):
        """Column entry first; a nested entry is resolved within itself only."""
        assert StageScaler(scale_features=setting).resolve_mode(column, entity) == expected

    @pytest.mark.parametrize(
        'entity, expected',
        [
            (('Zone euro', 'DE'), 'calendar'),
            (('Zone euro', 'FR'), 'constant'),
            ('Zone euro', 'constant'),
            (('Zone euro',), 'constant'),
            ('DE', 'constant'),
        ],
        ids=['full-key', 'other-entity', 'partial-scalar', 'partial-tuple', 'inner-level'],
    )
    def test_two_level_keys_match_full_entities_only(self, entity, expected):
        """A partial key — outer or inner level — designates no entity."""
        scaler = StageScaler(scale_features={'v': {('Zone euro', 'DE'): 'calendar'}})
        assert scaler.resolve_mode('v', entity) == expected

    def test_partial_key_does_not_cover_its_sub_entities(self):
        """'Zone euro' as a key leaves every ('Zone euro', *) entity on the nested default."""
        scaler = StageScaler(scale_features={'v': {'Zone euro': 'calendar'}})
        modes = [scaler.resolve_mode('v', e) for e in (('Zone euro', 'DE'), ('Zone euro', 'FR'))]
        assert modes == ['constant', 'constant']

    # --- Diviseurs de la cible ------------------------------------------------
    def test_target_divisor_gold(self):
        """DE divides by its real days (31, 28), FR by the constant 30."""
        index = _panel_grid(periods=2)
        divisors = StageScaler(scale_features=_DE_RAW).target_divisor(
            'v', source_freq='M', pred_freq='D', index=index
        )
        pd.testing.assert_series_equal(
            divisors, pd.Series([31.0, 28.0, 30.0, 30.0], index=index)
        )

    def test_fit_scale_factor_gold(self):
        """The baked factor follows the same per-entity modes, frozen on the fit grid."""
        index = _panel_grid(periods=2)
        factor = StageScaler(scale_features=_DE_RAW).fit_scale_factor(
            'v', source_freq='M', pred_freq='D', index=index
        )
        pd.testing.assert_series_equal(factor, pd.Series([31.0, 28.0, 30.0, 30.0], index=index))

    def test_target_divisor_with_production_frequencies(self):
        """Per-row production frequencies combine with per-entity modes."""
        index = _panel_grid(periods=2)
        produced = pd.Series(['M', 'D', 'M', 'D'], index=index)
        divisors = StageScaler(scale_features=_DE_RAW).target_divisor(
            'v', source_freq='M', pred_freq='D', index=index, produced_freq=produced
        )
        assert divisors.tolist() == [31.0, 1.0, 30.0, 1.0]

    def test_target_divisor_with_per_entity_blocks(self):
        """Per-entity block frequencies and per-entity modes, row by row.

        DE : bloc mensuel, jours réels ; FR : bloc trimestriel, facteur
        constant D -> Q = 91.
        """
        index = _panel_grid(periods=2)
        divisors = StageScaler(scale_features=_DE_RAW).target_divisor(
            'v', source_freq={'DE': 'M', 'FR': 'Q'}, pred_freq='D', index=index
        )
        assert divisors.tolist() == [31.0, 28.0, 91.0, 91.0]

    def test_rows_of_each_entity_keep_their_mode_on_a_shuffled_grid(self):
        """The divisors land on their own rows whatever the order of the grid."""
        frame = shuffle_rows(pd.DataFrame({'v': 0.0}, index=_panel_grid(periods=2)))
        divisors = StageScaler(scale_features=_DE_RAW).target_divisor(
            'v', source_freq='M', pred_freq='D', index=frame.index
        )
        january, february = pd.Timestamp('2021-01-31'), pd.Timestamp('2021-02-28')
        assert divisors.to_dict() == {
            ('DE', january): 31.0, ('DE', february): 28.0,
            ('FR', january): 30.0, ('FR', february): 30.0,
        }

    def test_two_level_entities(self):
        """On a two-level panel, only the full key ('Zone euro', 'DE') is raw."""
        divisors = StageScaler(
            scale_features={'v': {('Zone euro', 'DE'): 'calendar'}}
        ).target_divisor('v', source_freq='M', pred_freq='D', index=_two_level_grid())
        assert divisors.tolist() == [31.0, 28.0, 30.0, 30.0]

    def test_partial_key_on_a_two_level_panel_divides_by_the_nested_default(self):
        """'Zone euro' matches no entity: every row takes the nested default."""
        divisors = StageScaler(
            scale_features={'v': {'Zone euro': 'calendar'}}
        ).target_divisor('v', source_freq='M', pred_freq='D', index=_two_level_grid())
        assert divisors == 30.0

    def test_timeseries_grid_reads_the_nested_default(self):
        """No entity level: the nested default applies to every row."""
        scaler = StageScaler(scale_features={'v': {'DE': 'constant', DEFAULT_SCALE_KEY: 'calendar'}})
        divisors = scaler.target_divisor(
            'v', source_freq='M', pred_freq='D', index=_monthly_grid(periods=2)
        )
        assert divisors.tolist() == [31.0, 28.0]

    def test_grid_without_named_entity_keeps_the_scalar_form(self):
        """Every entity of the grid on one mode: the result is that of the scalar setting."""
        divisors = StageScaler(scale_features={'v': {'IT': 'calendar'}}).target_divisor(
            'v', source_freq='M', pred_freq='D', index=_panel_grid(periods=2)
        )
        assert divisors == 30.0

    def test_unanimous_nested_dict_needs_no_index(self):
        """Modes that cannot differ need no grid, exactly like the scalar form."""
        scaler = StageScaler(scale_features={'v': {'DE': 'constant'}})
        assert scaler.target_divisor('v', source_freq='M', pred_freq='D') == 30.0

    @pytest.mark.parametrize('method', ['target_divisor', 'fit_scale_factor'])
    def test_disagreeing_modes_without_index_raise(self, method):
        """Spreading per-entity modes needs a grid: never a silent nested default."""
        with pytest.raises(ValueError, match=r"per-entity modes that differ"):
            getattr(StageScaler(scale_features=_DE_RAW), method)('v', source_freq='M', pred_freq='D')

    def test_false_column_beside_a_nested_column(self):
        """A column in False stays exempt; the nested column next to it is per entity."""
        scaler = StageScaler(scale_features={'v': {'DE': 'calendar'}, 'w': False})
        index = _panel_grid(periods=2)
        assert scaler.target_divisor('w', source_freq='M', pred_freq='D', index=index) == 1.0
        divisors = scaler.target_divisor('v', source_freq='M', pred_freq='D', index=index)
        assert divisors.tolist() == [31.0, 28.0, 30.0, 30.0]

    # --- Diviseurs des covariables --------------------------------------------
    def test_feature_divisors_gold(self):
        """A daily covariate aggregated on months: real days for DE, 30 for FR."""
        divisors = StageScaler(
            scale_features={'d1': {'DE': 'calendar'}, 'd2': 'constant'}
        ).feature_divisors(
            columns=['d1', 'd2'],
            column_frequencies={'d1': 'D', 'd2': 'D'},
            ways={'d1': 'aggregate', 'd2': 'aggregate'},
            grid_freq='M',
            stage_freq='D',
            index=_panel_grid(periods=2),
        )
        assert isinstance(divisors, pd.DataFrame)
        assert divisors['d1'].tolist() == [31.0, 28.0, 30.0, 30.0]
        assert divisors['d2'].tolist() == [30.0] * 4

    def test_feature_divisors_on_a_mutualized_grid(self):
        """Per-entity blocks and per-entity modes on the same training grid."""
        divisors = StageScaler(scale_features={'d1': {'DE': 'calendar'}}).feature_divisors(
            columns=['d1'],
            column_frequencies={'d1': 'D'},
            ways={'d1': 'aggregate'},
            grid_freq={'DE': 'M', 'FR': 'M'},
            stage_freq='D',
            index=_panel_grid(periods=2),
        )
        assert divisors['d1'].tolist() == [31.0, 28.0, 30.0, 30.0]

    def test_feature_divisors_of_unanimous_modes_keep_the_compact_form(self):
        """A nested dict resolving to 'constant' everywhere changes nothing to the shape."""
        kwargs = dict(
            columns=['d1'], column_frequencies={'d1': 'D'}, ways={'d1': 'aggregate'},
            grid_freq='M', stage_freq='D', index=_panel_grid(periods=2),
        )
        nested = StageScaler(scale_features={'d1': {'IT': 'calendar'}}).feature_divisors(**kwargs)
        scalar = StageScaler(scale_features='constant').feature_divisors(**kwargs)
        pd.testing.assert_series_equal(nested, scalar)

    def test_feature_divisors_with_disagreeing_modes_without_index_raise(self):
        """Same prerequisite on the covariate side."""
        with pytest.raises(ValueError, match=r"per-entity modes that differ"):
            StageScaler(scale_features={'d1': {'DE': 'calendar'}}).feature_divisors(
                columns=['d1'], column_frequencies={'d1': 'D'}, ways={'d1': 'aggregate'},
                grid_freq='M', stage_freq='D',
            )

    def test_feature_divisors_two_level_entities(self):
        """Full two-level key on the covariate side."""
        divisors = StageScaler(
            scale_features={'d1': {('Zone euro', 'FR'): 'calendar'}}
        ).feature_divisors(
            columns=['d1'], column_frequencies={'d1': 'D'}, ways={'d1': 'aggregate'},
            grid_freq='M', stage_freq='D', index=_two_level_grid(),
        )
        assert divisors['d1'].tolist() == [30.0, 30.0, 31.0, 28.0]

    # --- Protocole sklearn ----------------------------------------------------
    def test_fit_transform_roundtrip(self):
        """fit / transform / inverse_transform of the target under per-entity modes."""
        index = _panel_grid(periods=2)
        X = pd.DataFrame({'d1': [1.0, 2.0, 3.0, 4.0]}, index=index)
        y = pd.Series([310.0, 280.0, 300.0, 300.0], index=index, name='v')
        scaler = StageScaler(
            scale_features=_DE_RAW, source_freq='M', pred_freq='D', column_frequencies={'d1': 'D'},
        )
        scaled = scaler.fit(X, y).transform(y)
        assert scaled.tolist() == [10.0, 10.0, 10.0, 10.0]
        np.testing.assert_allclose(scaler.inverse_transform(scaled), y, rtol=0, atol=_EXACT)

    def test_clone_keeps_the_nested_setting(self):
        """clone and get_params carry the nested dict as a plain parameter."""
        twin = clone(StageScaler(scale_features=_DE_RAW))
        assert twin.get_params()['scale_features'] == _DE_RAW
        assert twin.resolve_mode('v', 'DE') == 'calendar'
