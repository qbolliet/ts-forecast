"""Tests for tsforecast.frequency.provenance.

Reference: §6 of ``high_frequency_imputer2_architecture.md`` (provenance, taint
scale) and its table of §6.3.

Covered symbols: ``ProvenanceType`` (members, ``str`` serialization),
``resolve_model_provenance`` (full 3 x 3 truth table), ``origin_to_taint``,
``max_origin`` and ``ImputationProvenanceTracker`` (``initialize``,
``extend_index``, every ``mark_*``, ``clear_provenance``, ``get_provenance``,
``get_mask``, ``compute_statistics``, ``get_provenance_matrix``,
``to_string_matrix``, ``merge``, ``__repr__``), including its behavior on a
heterogeneous-coverage panel.
"""
# Modules de base
import itertools
import json

import numpy as np
import pandas as pd
import pytest

# Objet testé
from tsforecast.frequency.provenance import (
    ImputationProvenanceTracker,
    ProvenanceType,
    max_origin,
    origin_to_taint,
    resolve_model_provenance,
)

P = ProvenanceType

# Les cinq libellés MODEL_* de l'échelle de souillure (§6.1)
_MODEL_SCALE = (
    P.MODEL_ON_TRUE,
    P.MODEL_ON_INTERPOLATED,
    P.MODEL_ON_IMPUTED,
    P.MODEL_ON_IMPUTED_TARGET,
    P.MODEL_ON_IMPUTED_BOTH,
)


# Fabrique de tracker : série mensuelle de quatre lignes, deux colonnes
def _tracker(values_a=(1.0, np.nan, 3.0, np.nan), values_b=(np.nan,) * 4):
    """Build an initialized tracker on a 4-row monthly frame.

    Args:
        values_a: Values of column ``a``.
        values_b: Values of column ``b``.

    Returns:
        Tuple ``(tracker, dates)``.
    """
    dates = pd.date_range('2023-01-01', periods=len(values_a), freq='MS')
    data = pd.DataFrame({'a': list(values_a), 'b': list(values_b)}, index=dates)
    return ImputationProvenanceTracker().initialize(data), dates


# =============================================================================
# ProvenanceType
# =============================================================================
class TestProvenanceType:
    """Members, partition and ``str`` serialization of the enumeration."""

    def test_exactly_ten_members_with_expected_values(self):
        """The enumeration holds the ten members of §6.1, values in snake_case."""
        # Valeurs d'or : copie littérale du §6.1 de la spec
        expected = {
            'ORIGINAL': 'original',
            'AGGREGATED': 'aggregated',
            'DISAGGREGATED': 'disaggregated',
            'INTERPOLATED': 'interpolated',
            'MODEL_ON_TRUE': 'model_on_true',
            'MODEL_ON_INTERPOLATED': 'model_on_interpolated',
            'MODEL_ON_IMPUTED': 'model_on_imputed',
            'MODEL_ON_IMPUTED_TARGET': 'model_on_imputed_target',
            'MODEL_ON_IMPUTED_BOTH': 'model_on_imputed_both',
            'MODEL_UNANCHORED': 'model_unanchored',
        }
        assert {m.name: m.value for m in P} == expected

    def test_members_partition_into_non_model_scale_and_unanchored(self):
        """4 non-model members + the 5-label MODEL_* scale + MODEL_UNANCHORED."""
        # Hiérarchie : aucune relation d'ordre entre membres, seulement une
        # partition 4 + 5 + 1 lue sur le préfixe du nom
        non_model = {m for m in P if not m.name.startswith('MODEL_')}
        assert non_model == {P.ORIGINAL, P.AGGREGATED, P.DISAGGREGATED, P.INTERPOLATED}
        assert set(_MODEL_SCALE) | {P.MODEL_UNANCHORED} == set(P) - non_model

    def test_values_are_unique(self):
        """No two members share a value (no silent alias)."""
        assert len({m.value for m in P}) == len(P) == 10

    @pytest.mark.parametrize('member', list(P), ids=lambda m: m.name)
    def test_member_is_a_str_equal_to_its_value(self, member):
        """``str, Enum``: a member is a ``str`` equal to its value."""
        assert isinstance(member, str)
        assert member == member.value

    @pytest.mark.parametrize('member', list(P), ids=lambda m: m.name)
    def test_str_is_the_bare_value(self, member):
        """``str(member)`` is the value, ``repr`` keeps the Enum form."""
        assert str(member) == member.value
        # repr distingue un ProvenanceType d'un str brut
        assert repr(member).startswith('<ProvenanceType.')

    @pytest.mark.parametrize('member', list(P), ids=lambda m: m.name)
    def test_lookup_by_value_round_trips(self, member):
        """``ProvenanceType(value)`` returns the very same member."""
        assert P(member.value) is member

    def test_lookup_of_unknown_value_raises(self):
        """An unknown (or removed) value raises ValueError."""
        with pytest.raises(ValueError):
            P('model_on_mixed')

    def test_json_serialization_is_the_value(self):
        """A member serializes to its bare string in JSON."""
        assert json.dumps([P.ORIGINAL, P.MODEL_ON_TRUE]) == '["original", "model_on_true"]'

    def test_pandas_display_uses_the_bare_value(self):
        """A Series of members prints ``original``, not ``ProvenanceType.ORIGINAL``."""
        rendered = repr(pd.Series([P.ORIGINAL, P.INTERPOLATED]))
        assert 'original' in rendered
        assert 'ProvenanceType' not in rendered

    def test_model_on_mixed_is_gone(self):
        """D6 break: MODEL_ON_MIXED is no longer a member, with no alias."""
        with pytest.raises(AttributeError):
            P.MODEL_ON_MIXED  # noqa: B018
        assert 'model_on_mixed' not in {m.value for m in P}


# =============================================================================
# resolve_model_provenance / origin_to_taint / max_origin
# =============================================================================
class TestResolveModelProvenance:
    """Truth table 3 x 3 -> 5 of §6.3."""

    # Valeurs d'or écrites une à une depuis la table du §6.3 (pas de boucle
    # rejouant l'implémentation)
    @pytest.mark.parametrize(
        'covariate_taint, target_taint, expected',
        [
            ('none', 'none', P.MODEL_ON_TRUE),
            ('none', 'interpolated', P.MODEL_ON_INTERPOLATED),
            ('none', 'imputed', P.MODEL_ON_IMPUTED_TARGET),
            ('interpolated', 'none', P.MODEL_ON_INTERPOLATED),
            ('interpolated', 'interpolated', P.MODEL_ON_INTERPOLATED),
            ('interpolated', 'imputed', P.MODEL_ON_IMPUTED_TARGET),
            ('imputed', 'none', P.MODEL_ON_IMPUTED),
            ('imputed', 'interpolated', P.MODEL_ON_IMPUTED),
            ('imputed', 'imputed', P.MODEL_ON_IMPUTED_BOTH),
        ],
        ids=[
            'none-none', 'none-interp', 'none-imputed',
            'interp-none', 'interp-interp', 'interp-imputed',
            'imputed-none', 'imputed-interp', 'imputed-imputed',
        ],
    )
    def test_truth_table(self, covariate_taint, target_taint, expected):
        """Each of the nine (covariate_taint, target_taint) pairs of §6.3."""
        assert resolve_model_provenance(covariate_taint, target_taint) is expected

    @pytest.mark.parametrize(
        'covariate_taint, target_taint',
        list(itertools.product(['none', 'interpolated', 'imputed'], repeat=2)),
        ids=lambda v: str(v),
    )
    def test_always_a_model_scale_label_never_unanchored(self, covariate_taint, target_taint):
        """The result is always one of the five MODEL_* scale labels."""
        # MODEL_UNANCHORED relève d'un autre arbitrage (étape, pas souillure)
        assert resolve_model_provenance(covariate_taint, target_taint) in _MODEL_SCALE

    @pytest.mark.parametrize('covariate_taint', ['none', 'interpolated', 'imputed'])
    def test_imputed_target_dominates_interpolated_and_clean_covariates(self, covariate_taint):
        """A model-imputed target never yields a label below IMPUTED_TARGET."""
        assert resolve_model_provenance(covariate_taint, 'imputed') in (
            P.MODEL_ON_IMPUTED_TARGET, P.MODEL_ON_IMPUTED_BOTH,
        )

    @pytest.mark.parametrize(
        'covariate_taint, target_taint, culprit',
        [
            ('interpolate', 'none', 'covariate_taint'),
            ('none', 'Imputed', 'target_taint'),
            ('zz', 'zz', 'covariate_taint'),
            (None, 'none', 'covariate_taint'),
        ],
        ids=['typo-covariate', 'case-target', 'both-unknown', 'none-object'],
    )
    def test_unknown_taint_raises(self, covariate_taint, target_taint, culprit):
        """ANO-FREQ-003: a mistyped taint raises instead of reading as clean."""
        with pytest.raises(ValueError, match=culprit):
            resolve_model_provenance(covariate_taint, target_taint)

    def test_mark_model_imputed_rejects_an_unknown_taint(self):
        """The validation reaches ``mark_model_imputed``, which writes nothing."""
        tracker, dates = _tracker()
        with pytest.raises(ValueError, match='covariate_taint'):
            tracker.mark_model_imputed('a', dates[1], covariate_taint='interpolate')
        assert pd.isna(tracker.get_provenance('a', dates[1]))


class TestOriginToTaint:
    """Origin -> taint correspondence of §6.2."""

    @pytest.mark.parametrize(
        'origin, expected',
        [('observed', 'none'), ('interpolated', 'interpolated'), ('model', 'imputed')],
    )
    def test_correspondence(self, origin, expected):
        """The three origins map onto the three taints."""
        assert origin_to_taint(origin) == expected

    def test_unknown_origin_raises_key_error(self):
        """An origin outside the CellOrigin literals raises KeyError."""
        with pytest.raises(KeyError):
            origin_to_taint('aggregated')


class TestMaxOrigin:
    """Maximum over the increasing taint order observed < interpolated < model."""

    _ORDER = ['observed', 'interpolated', 'model']

    # Les neuf couples ordonnés : le maximum est l'origine de rang le plus haut
    @pytest.mark.parametrize(
        'left, right', list(itertools.product(_ORDER, repeat=2)), ids=lambda v: str(v)
    )
    def test_pairwise_maximum(self, left, right):
        """The max of two origins is the one with the higher rank."""
        expected = max(left, right, key=self._ORDER.index)
        assert max_origin([left, right]) == expected

    def test_empty_iterable_is_observed(self):
        """Empty input returns the neutral, least-tainted origin."""
        assert max_origin([]) == 'observed'

    @pytest.mark.parametrize('origin', _ORDER)
    def test_single_element(self, origin):
        """A single origin is its own maximum."""
        assert max_origin([origin]) == origin

    @pytest.mark.parametrize('permutation', list(itertools.permutations(_ORDER)))
    def test_independent_of_input_order(self, permutation):
        """All six permutations of the three origins give 'model'."""
        assert max_origin(permutation) == 'model'

    def test_accepts_a_one_shot_generator(self):
        """Any iterable is accepted, a generator included."""
        assert max_origin(o for o in ['observed', 'model', 'observed']) == 'model'

    def test_unknown_origin_raises_key_error(self):
        """An unknown origin raises KeyError instead of being ignored."""
        with pytest.raises(KeyError):
            max_origin(['observed', 'bogus'])


# =============================================================================
# ImputationProvenanceTracker — initialize
# =============================================================================
class TestInitialize:
    """Initialization of the provenance matrix from the input data."""

    def test_non_null_cells_are_original_and_null_cells_none(self):
        """Golden values: ORIGINAL where the value exists, None elsewhere."""
        tracker, _ = _tracker()
        matrix = tracker.provenance_matrix_
        assert matrix['a'].tolist()[0] is P.ORIGINAL
        assert matrix['a'].tolist()[2] is P.ORIGINAL
        assert matrix['a'].isna().tolist() == [False, True, False, True]
        # Colonne entièrement NaN : aucune provenance
        assert matrix['b'].isna().all()

    def test_returns_self_and_keeps_index_and_columns(self):
        """``initialize`` is chainable and the matrix mirrors the data's labels."""
        data = pd.DataFrame({'a': [1.0, np.nan]}, index=pd.date_range('2023-01-01', periods=2, freq='MS'))
        tracker = ImputationProvenanceTracker()
        assert tracker.initialize(data) is tracker
        assert tracker.provenance_matrix_.index.equals(data.index)
        assert list(tracker.provenance_matrix_.columns) == ['a']

    @pytest.mark.parametrize('bad', [[1.0, 2.0], np.array([1.0]), None], ids=['list', 'ndarray', 'none'])
    def test_non_dataframe_raises(self, bad):
        """Anything but a DataFrame raises ValueError."""
        with pytest.raises(ValueError, match='DataFrame'):
            ImputationProvenanceTracker().initialize(bad)

    def test_series_is_rejected(self):
        """A Series is not a DataFrame: rejected like any other input."""
        series = pd.Series([1.0], index=pd.date_range('2023-01-01', periods=1))
        with pytest.raises(ValueError, match='Series'):
            ImputationProvenanceTracker().initialize(series)

    @pytest.mark.parametrize(
        'empty',
        [pd.DataFrame(), pd.DataFrame({'a': []})],
        ids=['no-column-no-row', 'column-no-row'],
    )
    def test_empty_frame_raises(self, empty):
        """An empty dataset (no row or no column) raises ValueError."""
        with pytest.raises(ValueError, match='empty'):
            ImputationProvenanceTracker().initialize(empty)

    def test_single_observation(self):
        """A one-row frame is a valid dataset."""
        tracker = ImputationProvenanceTracker().initialize(
            pd.DataFrame({'a': [5.0]}, index=pd.date_range('2023-01-31', periods=1))
        )
        assert tracker.provenance_matrix_.shape == (1, 1)
        assert tracker.provenance_matrix_.iloc[0, 0] is P.ORIGINAL

    def test_unsorted_index_keeps_input_order_and_pairing(self):
        """Disorder is preserved, each label keeps the provenance of its own value."""
        dates = pd.date_range('2023-01-01', periods=4, freq='MS')
        data = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan]}, index=dates).iloc[[2, 0, 3, 1]]
        matrix = ImputationProvenanceTracker().initialize(data).provenance_matrix_
        assert matrix.index.equals(data.index)
        # Ordre d'entrée : mars (valeur), janvier (valeur), avril (NaN), février (NaN)
        assert matrix['a'].isna().tolist() == [False, False, True, True]

    def test_duplicated_index_keeps_every_row(self):
        """Duplicated labels are kept one row each, provenance follows each value."""
        dates = pd.DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01'])
        data = pd.DataFrame({'a': [1.0, np.nan, 3.0]}, index=dates)
        matrix = ImputationProvenanceTracker().initialize(data).provenance_matrix_
        assert len(matrix) == 3
        assert matrix['a'].isna().tolist() == [False, True, False]

    def test_mark_on_duplicated_label_marks_every_duplicate(self):
        """A label-based mark reaches all rows sharing the label."""
        dates = pd.DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01'])
        data = pd.DataFrame({'a': [1.0, np.nan, 3.0]}, index=dates)
        tracker = ImputationProvenanceTracker().initialize(data)
        tracker.mark_interpolated('a', pd.Timestamp('2023-01-01'))
        assert tracker.provenance_matrix_['a'].tolist() == [
            P.INTERPOLATED, P.INTERPOLATED, P.ORIGINAL,
        ]

    @pytest.mark.parametrize(
        'index',
        [
            pd.period_range('2023-01', periods=4, freq='M'),
            pd.RangeIndex(4),
            pd.date_range('2023-01-31', periods=4, freq='ME'),
        ],
        ids=['period', 'range', 'month-end'],
    )
    def test_index_types(self, index):
        """Period, Range and month-end indexes initialize and mark by label."""
        data = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan]}, index=index)
        tracker = ImputationProvenanceTracker().initialize(data)
        tracker.mark_interpolated('a', index[1])
        column = tracker.provenance_matrix_['a']
        assert column.iloc[:3].tolist() == [P.ORIGINAL, P.INTERPOLATED, P.ORIGINAL]
        assert pd.isna(column.iloc[3])

    def test_special_column_names(self):
        """Spaces, accents and symbols in column names are carried unchanged."""
        names = ['pib trimestriel', 'taux_chômage (%)', 'a/b']
        data = pd.DataFrame(
            {name: [1.0, np.nan] for name in names},
            index=pd.date_range('2023-01-01', periods=2, freq='MS'),
        )
        tracker = ImputationProvenanceTracker().initialize(data)
        assert list(tracker.provenance_matrix_.columns) == names
        tracker.mark_aggregated('taux_chômage (%)', data.index[1])
        assert tracker.get_provenance('taux_chômage (%)', data.index[1]) is P.AGGREGATED

    def test_multiindex_panel_is_auto_detected(self):
        """A MultiIndex panel tracks every column and records the entity level."""
        index = pd.MultiIndex.from_product(
            [['FR', 'DE'], pd.date_range('2023-01-01', periods=2, freq='MS')],
            names=['country', 'date'],
        )
        data = pd.DataFrame({'a': [1.0, np.nan, np.nan, 4.0]}, index=index)
        tracker = ImputationProvenanceTracker().initialize(data)
        assert tracker._panel_cols == ['country']
        assert list(tracker.provenance_matrix_.columns) == ['a']
        assert tracker.provenance_matrix_.index.equals(index)

    def test_three_level_panel_records_both_entity_levels(self):
        """On a 3-level index all levels but the last are entity levels."""
        index = pd.MultiIndex.from_tuples(
            [('EU', 'FR', pd.Timestamp('2023-01-01')), ('EU', 'FR', pd.Timestamp('2023-02-01'))],
            names=['region', 'country', 'date'],
        )
        data = pd.DataFrame({'a': [1.0, np.nan]}, index=index)
        tracker = ImputationProvenanceTracker().initialize(data)
        assert tracker._panel_cols == ['region', 'country']
        assert list(tracker.provenance_matrix_.columns) == ['a']

    def test_flat_panel_excludes_declared_entity_columns(self):
        """``panel_cols`` naming ordinary columns removes them from the tracked ones."""
        data = pd.DataFrame(
            {'country': ['x', 'x', 'y', 'y'], 'a': [1.0, np.nan, np.nan, 4.0]},
            index=pd.date_range('2023-01-01', periods=4, freq='MS'),
        )
        tracker = ImputationProvenanceTracker().initialize(data, panel_cols=['country'])
        assert list(tracker.provenance_matrix_.columns) == ['a']

    def test_flat_frame_without_panel_cols_tracks_every_column(self):
        """Without ``panel_cols`` and without MultiIndex, nothing is excluded."""
        data = pd.DataFrame(
            {'country': ['x', 'y'], 'a': [1.0, np.nan]},
            index=pd.date_range('2023-01-01', periods=2, freq='MS'),
        )
        tracker = ImputationProvenanceTracker().initialize(data)
        assert list(tracker.provenance_matrix_.columns) == ['country', 'a']

    def test_declared_panel_cols_matching_index_levels(self):
        """``panel_cols`` naming index levels is taken as already in the index."""
        index = pd.MultiIndex.from_product(
            [['FR'], pd.date_range('2023-01-01', periods=2, freq='MS')], names=['country', 'date']
        )
        data = pd.DataFrame({'a': [1.0, np.nan]}, index=index)
        tracker = ImputationProvenanceTracker().initialize(data, panel_cols=['country'])
        assert list(tracker.provenance_matrix_.columns) == ['a']


# =============================================================================
# extend_index
# =============================================================================
class TestExtendIndex:
    """Extension of the matrix to a densified grid wider than the input."""

    @staticmethod
    def _tracker(index):
        data = pd.DataFrame({'a': [1.0] * len(index), 'b': [2.0] * len(index)}, index=index)
        return ImputationProvenanceTracker().initialize(data)

    def test_new_rows_are_unfilled_and_sorted_in(self):
        """Added dates are None and take their chronological place."""
        tracker = self._tracker(pd.to_datetime(['2015-01-01', '2015-04-01']))
        grid = pd.date_range('2015-01-01', '2015-04-01', freq='MS')
        tracker.extend_index(grid)

        matrix = tracker.provenance_matrix_
        assert list(matrix.index) == list(grid)
        assert matrix['a'].tolist() == [P.ORIGINAL, None, None, P.ORIGINAL]

    def test_new_rows_are_unfilled_in_every_column(self):
        """The added rows are None in all columns, not only one."""
        tracker = self._tracker(pd.to_datetime(['2015-01-01', '2015-03-01']))
        tracker.extend_index(pd.to_datetime(['2015-02-01']))
        assert tracker.provenance_matrix_.loc['2015-02-01'].isna().all()

    def test_extension_makes_mark_possible(self):
        """Without extension, marking an absent set of dates raises KeyError."""
        tracker = self._tracker(pd.to_datetime(['2015-01-01', '2015-04-01']))
        grid = pd.date_range('2015-01-01', '2015-04-01', freq='MS')
        with pytest.raises(KeyError):
            tracker.mark_interpolated('a', grid)

        tracker.extend_index(grid)
        tracker.mark_interpolated('a', grid)
        assert (tracker.provenance_matrix_['a'] == P.INTERPOLATED).all()

    def test_known_labels_are_a_no_op(self):
        """A grid already covered leaves the matrix strictly intact."""
        index = pd.date_range('2015-01-01', periods=3, freq='MS')
        tracker = self._tracker(index)
        before = tracker.provenance_matrix_.copy()
        tracker.extend_index(index[:2])
        pd.testing.assert_frame_equal(tracker.provenance_matrix_, before)

    def test_empty_index_is_a_no_op(self):
        """An empty index adds nothing."""
        tracker = self._tracker(pd.date_range('2015-01-01', periods=3, freq='MS'))
        before = tracker.provenance_matrix_.copy()
        tracker.extend_index(pd.DatetimeIndex([]))
        pd.testing.assert_frame_equal(tracker.provenance_matrix_, before)

    def test_existing_provenance_is_preserved(self):
        """Marks already written survive an extension."""
        tracker = self._tracker(pd.to_datetime(['2015-01-01', '2015-03-01']))
        tracker.mark_aggregated('a', pd.Timestamp('2015-03-01'))
        tracker.extend_index(pd.to_datetime(['2015-02-01']))
        assert tracker.get_provenance('a', pd.Timestamp('2015-03-01')) is P.AGGREGATED
        assert tracker.get_provenance('b', pd.Timestamp('2015-01-01')) is P.ORIGINAL

    def test_unsorted_matrix_keeps_its_order(self):
        """An unsorted matrix is not reordered: the new rows go at the end."""
        tracker = self._tracker(pd.to_datetime(['2015-03-01', '2015-01-01']))
        tracker.extend_index(pd.to_datetime(['2015-02-01']))
        assert list(tracker.provenance_matrix_.index) == list(
            pd.to_datetime(['2015-03-01', '2015-01-01', '2015-02-01'])
        )

    def test_panel_rows_land_inside_their_entity(self):
        """On a sorted MultiIndex the added row stays in its entity's block."""
        index = pd.MultiIndex.from_tuples(
            [('FR', pd.Timestamp('2015-01-01')), ('FR', pd.Timestamp('2015-03-01')),
             ('IT', pd.Timestamp('2015-01-01'))],
            names=['country', 'date'],
        )
        tracker = self._tracker(index)
        tracker.extend_index(pd.MultiIndex.from_tuples(
            [('FR', pd.Timestamp('2015-02-01'))], names=['country', 'date']
        ))
        assert tracker.provenance_matrix_.index.get_level_values('country').tolist() == [
            'FR', 'FR', 'FR', 'IT'
        ]
        assert tracker.provenance_matrix_['a'].isna().sum() == 1

    def test_uninitialized_raises(self):
        """Extending an uninitialized tracker raises ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().extend_index(pd.date_range('2015-01-01', periods=2))


# =============================================================================
# mark_* / clear_provenance / get_provenance
# =============================================================================
class TestMarks:
    """Writing provenance: ``mark_imputed`` and its convenience wrappers."""

    @pytest.mark.parametrize('provenance', list(P), ids=lambda m: m.name)
    def test_mark_imputed_writes_every_provenance_type(self, provenance):
        """``mark_imputed`` accepts all ten members."""
        tracker, dates = _tracker()
        tracker.mark_imputed('a', dates[1], provenance)
        assert tracker.get_provenance('a', dates[1]) is provenance

    @pytest.mark.parametrize(
        'method, expected',
        [
            ('mark_aggregated', P.AGGREGATED),
            ('mark_disaggregated', P.DISAGGREGATED),
            ('mark_interpolated', P.INTERPOLATED),
        ],
    )
    def test_convenience_wrappers(self, method, expected):
        """Each wrapper writes its own provenance type."""
        tracker, dates = _tracker()
        getattr(tracker, method)('a', dates[1])
        assert tracker.get_provenance('a', dates[1]) is expected

    @pytest.mark.parametrize(
        'covariate_taint, target_taint, expected',
        [
            ('none', 'none', P.MODEL_ON_TRUE),
            ('interpolated', 'none', P.MODEL_ON_INTERPOLATED),
            ('imputed', 'none', P.MODEL_ON_IMPUTED),
            ('none', 'imputed', P.MODEL_ON_IMPUTED_TARGET),
            ('imputed', 'imputed', P.MODEL_ON_IMPUTED_BOTH),
        ],
        ids=['true', 'interpolated', 'imputed', 'target', 'both'],
    )
    def test_mark_model_imputed_reaches_the_five_labels(self, covariate_taint, target_taint, expected):
        """The five MODEL_* labels are reachable through the taint signature."""
        tracker, dates = _tracker()
        tracker.mark_model_imputed(
            'a', dates[1], covariate_taint=covariate_taint, target_taint=target_taint
        )
        assert tracker.get_provenance('a', dates[1]) is expected

    def test_mark_model_imputed_defaults_to_model_on_true(self):
        """Without taint arguments the cell is MODEL_ON_TRUE."""
        tracker, dates = _tracker()
        tracker.mark_model_imputed('a', dates[1])
        assert tracker.get_provenance('a', dates[1]) is P.MODEL_ON_TRUE

    def test_mark_model_imputed_has_no_trained_on_imputed_argument(self):
        """§6.6: the ``trained_on_imputed`` boolean is gone."""
        tracker, dates = _tracker()
        with pytest.raises(TypeError):
            tracker.mark_model_imputed('a', dates[1], trained_on_imputed=True)

    def test_mark_with_datetimeindex(self):
        """A DatetimeIndex marks several cells at once, others untouched."""
        tracker, dates = _tracker()
        tracker.mark_interpolated('a', dates[[1, 3]])
        assert tracker.provenance_matrix_['a'].tolist() == [
            P.ORIGINAL, P.INTERPOLATED, P.ORIGINAL, P.INTERPOLATED,
        ]

    def test_mark_with_slice_is_label_inclusive(self):
        """A label slice includes both bounds."""
        tracker, dates = _tracker()
        tracker.mark_aggregated('a', slice(dates[1], dates[2]))
        column = tracker.provenance_matrix_['a']
        assert column.iloc[:3].tolist() == [P.ORIGINAL, P.AGGREGATED, P.AGGREGATED]
        assert pd.isna(column.iloc[3])

    def test_mark_leaves_other_columns_untouched(self):
        """Writing column ``a`` never changes column ``b``."""
        tracker, dates = _tracker(values_b=(1.0, np.nan, np.nan, np.nan))
        before = tracker.provenance_matrix_['b'].copy()
        tracker.mark_interpolated('a', dates)
        pd.testing.assert_series_equal(tracker.provenance_matrix_['b'], before)

    def test_partially_absent_index_raises_key_error(self):
        """A list-like with a label outside the matrix raises KeyError."""
        tracker, dates = _tracker()
        partial = pd.DatetimeIndex([dates[1], pd.Timestamp('2030-01-01')])
        with pytest.raises(KeyError):
            tracker.mark_interpolated('a', partial)

    def test_fully_absent_index_raises_key_error(self):
        """A list-like entirely outside the matrix raises KeyError."""
        tracker, _ = _tracker()
        with pytest.raises(KeyError):
            tracker.mark_interpolated('a', pd.DatetimeIndex(['2030-01-01', '2030-02-01']))
        assert len(tracker.provenance_matrix_) == 4

    def test_slice_outside_the_matrix_is_a_no_op(self):
        """A label slice entirely past the matrix marks nothing and adds no row."""
        tracker, _ = _tracker()
        before = tracker.provenance_matrix_.copy()
        tracker.mark_interpolated(
            'a', slice(pd.Timestamp('2030-01-01'), pd.Timestamp('2031-01-01'))
        )
        pd.testing.assert_frame_equal(tracker.provenance_matrix_, before)

    @pytest.mark.parametrize(
        'method', ['mark_aggregated', 'mark_disaggregated', 'mark_interpolated', 'mark_model_imputed'],
    )
    def test_scalar_timestamp_outside_the_matrix_raises_key_error(self, method):
        """ANO-FREQ-001: an absent scalar raises like a list-like, adding no row."""
        tracker, _ = _tracker()
        with pytest.raises(KeyError):
            getattr(tracker, method)('a', pd.Timestamp('2030-01-01'))
        assert tracker.provenance_matrix_.shape == (4, 2)

    def test_scalar_timestamp_inside_the_matrix_still_marks(self):
        """A present scalar label is written as before."""
        tracker, dates = _tracker()
        tracker.mark_interpolated('a', dates[1])
        assert tracker.get_provenance('a', dates[1]) is P.INTERPOLATED

    def test_multiindex_key_outside_the_matrix_raises_key_error(self):
        """On a panel, an absent (entity, date) key raises; a present one marks."""
        index = pd.MultiIndex.from_product(
            [['FR', 'DE'], pd.date_range('2023-01-01', periods=2, freq='MS')],
            names=['country', 'date'],
        )
        tracker = ImputationProvenanceTracker().initialize(
            pd.DataFrame({'a': [1.0, np.nan, np.nan, 4.0]}, index=index)
        )
        with pytest.raises(KeyError):
            tracker.mark_interpolated('a', ('IT', pd.Timestamp('2023-01-01')))
        assert tracker.provenance_matrix_.shape == (4, 1)
        tracker.mark_interpolated('a', ('FR', pd.Timestamp('2023-02-01')))
        assert tracker.get_provenance('a', ('FR', pd.Timestamp('2023-02-01'))) is P.INTERPOLATED

    def test_unknown_column_raises(self):
        """Marking an unknown column raises ValueError naming it."""
        tracker, dates = _tracker()
        with pytest.raises(ValueError, match="'zz'"):
            tracker.mark_imputed('zz', dates[0], P.INTERPOLATED)

    @pytest.mark.parametrize('bad', ['original', None, 3], ids=['str', 'none', 'int'])
    def test_non_provenance_type_raises(self, bad):
        """Only ProvenanceType members are accepted, not even their string value."""
        tracker, dates = _tracker()
        with pytest.raises(ValueError, match='ProvenanceType'):
            tracker.mark_imputed('a', dates[0], bad)

    def test_uninitialized_raises(self):
        """Marking before ``initialize`` raises ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().mark_imputed('a', pd.Timestamp('2023-01-01'), P.ORIGINAL)

    # Priorité d'écrasement : aucune. La dernière écriture l'emporte, y compris
    # sur ORIGINAL (§6.4 : une date-ancre ré-exprimée ne porte plus l'observation)
    # et y compris vers une provenance « plus propre » (pas de max de souillure :
    # la souillure se calcule dans le plan, jamais dans la matrice).
    @pytest.mark.parametrize(
        'first, second',
        [
            (P.ORIGINAL, P.MODEL_ON_TRUE),
            (P.MODEL_ON_IMPUTED_BOTH, P.INTERPOLATED),
            (P.INTERPOLATED, P.MODEL_ON_IMPUTED_BOTH),
            (P.MODEL_ON_TRUE, P.MODEL_UNANCHORED),
            (P.AGGREGATED, P.AGGREGATED),
        ],
        ids=['original-overwritten', 'dirty-to-clean', 'clean-to-dirty', 'to-unanchored', 'same'],
    )
    def test_last_write_wins(self, first, second):
        """Overwriting an existing provenance: the last write wins, no priority."""
        tracker, dates = _tracker()
        tracker.mark_imputed('a', dates[0], first)
        tracker.mark_imputed('a', dates[0], second)
        assert tracker.get_provenance('a', dates[0]) is second

    def test_marking_an_original_cell_overwrites_original(self):
        """§6.4: an anchor date overwritten by a prediction is no longer ORIGINAL."""
        tracker, dates = _tracker()
        assert tracker.get_provenance('a', dates[0]) is P.ORIGINAL
        tracker.mark_model_imputed('a', dates[0])
        assert tracker.get_provenance('a', dates[0]) is P.MODEL_ON_TRUE


class TestClearProvenance:
    """Reset of cells to "not filled"."""

    def test_clear_resets_to_none(self):
        """A cleared cell goes back to the NaN-cell convention (None)."""
        tracker, dates = _tracker()
        tracker.clear_provenance('a', dates[0])
        assert tracker.get_provenance('a', dates[0]) is None

    def test_clear_then_mark(self):
        """A cleared cell can be marked again."""
        tracker, dates = _tracker()
        tracker.clear_provenance('a', dates[0])
        tracker.mark_aggregated('a', dates[0])
        assert tracker.get_provenance('a', dates[0]) is P.AGGREGATED

    def test_clear_several_cells_and_slice(self):
        """DatetimeIndex and slice clear several cells, others untouched."""
        tracker, dates = _tracker(values_a=(1.0, 2.0, 3.0, 4.0))
        tracker.clear_provenance('a', dates[[0, 1]])
        tracker.clear_provenance('a', slice(dates[3], dates[3]))
        assert tracker.provenance_matrix_['a'].isna().tolist() == [True, True, False, True]

    def test_clear_already_empty_cell_is_idempotent(self):
        """Clearing a None cell leaves it None."""
        tracker, dates = _tracker()
        tracker.clear_provenance('a', dates[1])
        assert tracker.get_provenance('a', dates[1]) is None

    def test_unknown_column_raises(self):
        """An unknown column raises ValueError."""
        tracker, dates = _tracker()
        with pytest.raises(ValueError, match="'zz'"):
            tracker.clear_provenance('zz', dates[0])

    def test_scalar_timestamp_outside_the_matrix_raises_key_error(self):
        """ANO-FREQ-001: clearing an absent scalar adds no row either."""
        tracker, _ = _tracker()
        with pytest.raises(KeyError):
            tracker.clear_provenance('a', pd.Timestamp('2030-01-01'))
        assert tracker.provenance_matrix_.shape == (4, 2)

    def test_uninitialized_raises(self):
        """Clearing before ``initialize`` raises ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().clear_provenance('a', pd.Timestamp('2023-01-01'))


class TestGetProvenance:
    """Reading the provenance of cells."""

    def test_scalar_returns_the_member(self):
        """A single timestamp returns the ProvenanceType itself."""
        tracker, dates = _tracker()
        assert tracker.get_provenance('a', dates[0]) is P.ORIGINAL

    def test_several_labels_return_a_series(self):
        """A DatetimeIndex returns a Series restricted to those labels."""
        tracker, dates = _tracker()
        result = tracker.get_provenance('a', dates[:2])
        assert isinstance(result, pd.Series)
        assert result.index.equals(dates[:2])

    def test_slice_returns_a_series(self):
        """A slice returns the matching rows."""
        tracker, dates = _tracker()
        assert len(tracker.get_provenance('a', slice(dates[1], dates[3]))) == 3

    def test_unfilled_cell_is_none(self):
        """A NaN input cell reads back as a null (NaN after ``initialize``, ANO-FREQ-004)."""
        tracker, dates = _tracker()
        assert pd.isna(tracker.get_provenance('a', dates[1]))

    def test_unknown_column_raises(self):
        """An unknown column raises ValueError."""
        tracker, dates = _tracker()
        with pytest.raises(ValueError, match="'zz'"):
            tracker.get_provenance('zz', dates[0])

    def test_unknown_label_raises_key_error(self):
        """A label outside the matrix raises KeyError."""
        tracker, _ = _tracker()
        with pytest.raises(KeyError):
            tracker.get_provenance('a', pd.Timestamp('2030-01-01'))

    def test_uninitialized_raises(self):
        """Reading before ``initialize`` raises ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().get_provenance('a', pd.Timestamp('2023-01-01'))


# =============================================================================
# get_mask
# =============================================================================
class TestGetMask:
    """Boolean masks over the provenance matrix."""

    @staticmethod
    def _filled():
        tracker, dates = _tracker(values_b=(1.0, 2.0, np.nan, np.nan))
        tracker.mark_aggregated('a', dates[1])
        tracker.mark_interpolated('a', dates[3])
        tracker.mark_interpolated('b', dates[2])
        return tracker, dates

    def test_single_type_whole_frame(self):
        """A single type returns a boolean DataFrame over all columns."""
        tracker, _ = self._filled()
        mask = tracker.get_mask(P.INTERPOLATED)
        assert isinstance(mask, pd.DataFrame)
        assert mask.sum().to_dict() == {'a': 1, 'b': 1}

    def test_list_of_types_is_a_union(self):
        """A list selects cells matching ANY of the types."""
        tracker, _ = self._filled()
        mask = tracker.get_mask([P.INTERPOLATED, P.AGGREGATED], column='a')
        assert mask.tolist() == [False, True, False, True]

    def test_column_returns_a_series(self):
        """With ``column`` the mask is a Series."""
        tracker, dates = self._filled()
        mask = tracker.get_mask(P.ORIGINAL, column='a')
        assert isinstance(mask, pd.Series)
        assert mask.index.equals(dates)
        assert mask.tolist() == [True, False, True, False]

    def test_unfilled_cells_never_match(self):
        """None cells match no type."""
        tracker, _ = self._filled()
        # Colonne « a » entièrement renseignée, colonne « b » : dernière cellule vide
        assert tracker.get_mask(list(P), column='a').all()
        assert not tracker.get_mask(list(P), column='b').iloc[3]

    def test_empty_list_matches_nothing(self):
        """An empty type list gives an all-False mask."""
        tracker, _ = self._filled()
        assert not tracker.get_mask([]).to_numpy().any()

    def test_unknown_column_raises(self):
        """An unknown column raises ValueError."""
        tracker, _ = self._filled()
        with pytest.raises(ValueError, match="'zz'"):
            tracker.get_mask(P.ORIGINAL, column='zz')

    def test_uninitialized_raises(self):
        """A mask before ``initialize`` raises ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().get_mask(P.ORIGINAL)


# =============================================================================
# compute_statistics
# =============================================================================
class TestComputeStatistics:
    """Counts and percentages per provenance type, overall and per column."""

    @staticmethod
    def _all_types_tracker():
        # Colonne « a » : une cellule par type (10) + 2 cellules vides ; colonne « b » vide
        dates = pd.date_range('2023-01-01', periods=12, freq='MS')
        data = pd.DataFrame({'a': [np.nan] * 12, 'b': [np.nan] * 12}, index=dates)
        tracker = ImputationProvenanceTracker().initialize(data)
        for offset, prov_type in enumerate(P):
            tracker.mark_imputed('a', dates[offset], prov_type)
        return tracker

    def test_keys_cover_overall_columns_and_all_types(self):
        """One entry per column + 'overall', each with a count and a pct per type."""
        stats = self._all_types_tracker().compute_statistics()
        assert set(stats) == {'overall', 'a', 'b'}
        expected_keys = (
            {m.value for m in P} | {f'{m.value}_pct' for m in P} | {'not_imputed', 'not_imputed_pct'}
        )
        for scope in stats.values():
            assert set(scope) == expected_keys

    def test_golden_counts_and_percentages(self):
        """Golden values on 12 cells: 10 typed, 2 unfilled in ``a``; ``b`` empty."""
        stats = self._all_types_tracker().compute_statistics()
        # Colonne « a » : 1 cellule par type sur 12 => 100/12 % chacune
        for member in P:
            assert stats['a'][member.value] == 1
            assert stats['a'][f'{member.value}_pct'] == pytest.approx(100 / 12)
        assert stats['a']['not_imputed'] == 2
        # Colonne « b » : 12 cellules vides
        assert stats['b']['not_imputed'] == 12
        assert stats['b']['not_imputed_pct'] == pytest.approx(100.0)
        # Global : 24 cellules, 10 typées, 14 vides
        assert stats['overall']['model_on_true'] == 1
        assert stats['overall']['model_on_true_pct'] == pytest.approx(100 / 24)
        assert stats['overall']['not_imputed'] == 14

    def test_percentages_sum_to_one_hundred(self):
        """For each scope, type percentages + not_imputed_pct total 100."""
        stats = self._all_types_tracker().compute_statistics()
        for scope in stats.values():
            total = sum(v for k, v in scope.items() if k.endswith('_pct'))
            assert total == pytest.approx(100.0)

    def test_counts_sum_to_the_number_of_cells(self):
        """Counts of all types + not_imputed total the matrix size."""
        tracker = self._all_types_tracker()
        stats = tracker.compute_statistics()
        counts = sum(v for k, v in stats['overall'].items() if not k.endswith('_pct'))
        assert counts == tracker.provenance_matrix_.size

    def test_result_is_stored_in_statistics_attribute(self):
        """The returned dict is also kept in ``statistics_``."""
        tracker = self._all_types_tracker()
        assert tracker.statistics_ is None
        assert tracker.compute_statistics() is tracker.statistics_

    def test_recomputation_reflects_new_marks(self):
        """Statistics are recomputed from the current matrix at each call."""
        tracker, dates = _tracker()
        assert tracker.compute_statistics()['a']['interpolated'] == 0
        tracker.mark_interpolated('a', dates[1])
        assert tracker.compute_statistics()['a']['interpolated'] == 1

    def test_fully_unfilled_dataset(self):
        """A dataset with no value at all: everything is not_imputed, no ZeroDivision."""
        tracker, _ = _tracker(values_a=(np.nan,) * 4)
        stats = tracker.compute_statistics()
        assert stats['overall']['not_imputed_pct'] == pytest.approx(100.0)
        assert stats['overall']['original'] == 0

    def test_zero_row_matrix_yields_zero_percentages(self):
        """A matrix emptied of rows reports 0.0 percentages instead of dividing by zero."""
        tracker, _ = _tracker()
        tracker.provenance_matrix_ = tracker.provenance_matrix_.iloc[0:0]
        stats = tracker.compute_statistics()
        assert stats['overall']['original_pct'] == 0.0
        assert stats['a']['not_imputed_pct'] == 0.0

    def test_uninitialized_raises(self):
        """Statistics before ``initialize`` raise ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().compute_statistics()


# =============================================================================
# get_provenance_matrix / to_string_matrix / repr
# =============================================================================
class TestMatrixViews:
    """Copies and string rendering of the provenance matrix."""

    def test_get_provenance_matrix_is_an_independent_copy(self):
        """Mutating the returned frame does not alter the tracker."""
        tracker, dates = _tracker()
        copy = tracker.get_provenance_matrix()
        copy.loc[dates[0], 'a'] = P.MODEL_ON_TRUE
        assert tracker.get_provenance('a', dates[0]) is P.ORIGINAL

    def test_get_provenance_matrix_equals_the_attribute(self):
        """The copy has the same content as ``provenance_matrix_``."""
        tracker, _ = _tracker()
        pd.testing.assert_frame_equal(tracker.get_provenance_matrix(), tracker.provenance_matrix_)

    def test_get_provenance_matrix_uninitialized_raises(self):
        """Reading the matrix before ``initialize`` raises ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().get_provenance_matrix()

    def test_to_string_matrix_golden(self):
        """Members become their value, unfilled cells 'not_imputed'."""
        tracker, dates = _tracker()
        tracker.mark_model_imputed('a', dates[1], covariate_taint='imputed')
        result = tracker.to_string_matrix()
        assert result['a'].tolist() == ['original', 'model_on_imputed', 'original', 'not_imputed']
        assert result['b'].tolist() == ['not_imputed'] * 4

    def test_to_string_matrix_keeps_shape_and_labels(self):
        """Index and columns are those of the matrix."""
        tracker, _ = _tracker()
        result = tracker.to_string_matrix()
        assert result.index.equals(tracker.provenance_matrix_.index)
        assert list(result.columns) == ['a', 'b']

    def test_to_string_matrix_holds_plain_strings(self):
        """No ProvenanceType instance is left: every cell is a bare str value."""
        tracker, _ = _tracker()
        cells = tracker.to_string_matrix().to_numpy().ravel()
        assert all(type(cell) is str for cell in cells)

    @pytest.mark.internal
    def test_to_string_matrix_stringifies_foreign_values(self):
        """A non-ProvenanceType, non-null cell is rendered with ``str``."""
        tracker, dates = _tracker()
        tracker.provenance_matrix_.loc[dates[0], 'a'] = 'legacy_value'
        assert tracker.to_string_matrix().loc[dates[0], 'a'] == 'legacy_value'

    def test_to_string_matrix_uninitialized_raises(self):
        """Rendering before ``initialize`` raises ValueError."""
        with pytest.raises(ValueError, match='not initialized'):
            ImputationProvenanceTracker().to_string_matrix()

    def test_repr(self):
        """The repr gives the matrix shape, or 'not initialized'."""
        assert repr(ImputationProvenanceTracker()) == 'ImputationProvenanceTracker(not initialized)'
        tracker, _ = _tracker()
        assert repr(tracker) == 'ImputationProvenanceTracker(rows=4, cols=2)'


# =============================================================================
# merge
# =============================================================================
class TestMerge:
    """Combination of the matrices of two trackers of the same shape."""

    @staticmethod
    def _pair():
        # self : a = [ORIGINAL, AGGREGATED, ORIGINAL, None]
        # other: a = [INTERPOLATED, INTERPOLATED, ORIGINAL, None]
        base, dates = _tracker()
        base.mark_aggregated('a', dates[1])
        other, _ = _tracker()
        other.mark_interpolated('a', dates[[0, 1]])
        return base, other, dates

    def test_update_other_overwrites_where_it_is_set(self):
        """``overwrite=True``: filled cells of ``other`` win, overlapping or not."""
        base, other, _ = self._pair()
        base.merge(other, overwrite=True)
        column = base.provenance_matrix_['a']
        assert column.iloc[:3].tolist() == [P.INTERPOLATED, P.INTERPOLATED, P.ORIGINAL]
        assert pd.isna(column.iloc[3])

    def test_update_never_erases_with_a_none(self):
        """``overwrite=True``: a null in ``other`` leaves ``self`` untouched."""
        base, other, dates = self._pair()
        other.clear_provenance('a', dates[2])
        base.merge(other, overwrite=True)
        assert base.get_provenance('a', dates[2]) is P.ORIGINAL

    def test_overwrite_is_the_default(self):
        """No ``overwrite`` argument means ``overwrite=True``."""
        base, other, _ = self._pair()
        base.merge(other)
        assert base.provenance_matrix_['a'].iloc[0] is P.INTERPOLATED

    def test_preserve_keeps_self_and_fills_none(self):
        """``overwrite=False``: conflicts resolve to ``self``, null cells are filled from ``other``."""
        base, other, dates = self._pair()
        other.mark_aggregated('a', dates[3])
        base.merge(other, overwrite=False)
        assert base.provenance_matrix_['a'].tolist() == [
            P.ORIGINAL, P.AGGREGATED, P.ORIGINAL, P.AGGREGATED,
        ]

    def test_merge_returns_self(self):
        """``merge`` is chainable and mutates the receiver."""
        base, other, _ = self._pair()
        assert base.merge(other) is base

    def test_merge_does_not_mutate_other(self):
        """The merged-in tracker is left untouched."""
        base, other, _ = self._pair()
        before = other.provenance_matrix_.copy()
        base.merge(other)
        pd.testing.assert_frame_equal(other.provenance_matrix_, before)

    def test_merge_invalidates_statistics(self):
        """Statistics computed before a merge are reset to None."""
        base, other, _ = self._pair()
        base.compute_statistics()
        base.merge(other)
        assert base.statistics_ is None

    def test_merge_handles_each_column_independently(self):
        """Overlap is handled per column: ``b`` is merged, ``a`` is not touched."""
        base, dates = _tracker(values_b=(1.0, np.nan, np.nan, np.nan))
        other, _ = _tracker(values_b=(1.0, np.nan, np.nan, np.nan))
        other.mark_interpolated('b', dates[2])
        base.merge(other)
        column_b = base.provenance_matrix_['b']
        assert column_b.iloc[[0, 2]].tolist() == [P.ORIGINAL, P.INTERPOLATED]
        assert pd.isna(column_b.iloc[1])
        assert base.provenance_matrix_['a'].iloc[0] is P.ORIGINAL

    def test_how_argument_is_gone(self):
        """The former ``how`` strategy is replaced by ``overwrite``."""
        base, other, _ = self._pair()
        with pytest.raises(TypeError):
            base.merge(other, how='update')

    def test_merge_never_adds_cells(self):
        """Only cells ``self`` already carries are updated: shape and labels are unchanged."""
        base, other, _ = self._pair()
        before_index, before_columns = base.provenance_matrix_.index, base.provenance_matrix_.columns
        base.merge(other)
        assert base.provenance_matrix_.index.equals(before_index)
        assert base.provenance_matrix_.columns.equals(before_columns)

    def test_column_order_may_differ(self):
        """The same columns in another order are compatible (label alignment)."""
        base, dates = _tracker()
        other, _ = _tracker()
        other.mark_interpolated('a', dates[1])
        other.provenance_matrix_ = other.provenance_matrix_[['b', 'a']]
        base.merge(other)
        assert base.get_provenance('a', dates[1]) is P.INTERPOLATED

    def test_unsorted_identical_index_merges(self):
        """Two unsorted matrices sharing the very same row order merge."""
        dates = pd.date_range('2023-01-01', periods=4, freq='MS')
        data = pd.DataFrame({'a': [1.0, np.nan, 3.0, np.nan]}, index=dates).iloc[[2, 0, 3, 1]]
        base = ImputationProvenanceTracker().initialize(data)
        other = ImputationProvenanceTracker().initialize(data)
        other.mark_interpolated('a', dates[0])
        base.merge(other)
        assert base.get_provenance('a', dates[0]) is P.INTERPOLATED

    def test_incompatible_shapes_raise(self):
        """Different shapes raise ValueError naming both."""
        base, _, _ = self._pair()
        narrow = ImputationProvenanceTracker().initialize(
            pd.DataFrame({'a': [1.0] * 4}, index=pd.date_range('2023-01-01', periods=4, freq='MS'))
        )
        with pytest.raises(ValueError, match='Incompatible shapes'):
            base.merge(narrow)

    def test_uninitialized_self_raises(self):
        """An uninitialized receiver raises ValueError."""
        _, other, _ = self._pair()
        with pytest.raises(ValueError, match="This tracker"):
            ImputationProvenanceTracker().merge(other)

    def test_uninitialized_other_raises(self):
        """An uninitialized argument raises ValueError."""
        base, _, _ = self._pair()
        with pytest.raises(ValueError, match="Other tracker"):
            base.merge(ImputationProvenanceTracker())

    def test_same_shape_but_other_columns_raises(self):
        """ANO-FREQ-002: same shape is not enough, unmatched columns raise."""
        base, dates = _tracker()
        other = ImputationProvenanceTracker().initialize(
            pd.DataFrame({'x': [1.0] * 4, 'y': [2.0] * 4}, index=dates)
        )
        before = base.provenance_matrix_.copy()
        with pytest.raises(ValueError, match='Incompatible labels'):
            base.merge(other)
        pd.testing.assert_frame_equal(base.provenance_matrix_, before)

    def test_same_shape_but_other_index_raises(self):
        """ANO-FREQ-002: same shape but other row labels raises."""
        base, _ = _tracker()
        shifted = pd.date_range('2030-01-01', periods=4, freq='MS')
        other = ImputationProvenanceTracker().initialize(
            pd.DataFrame({'a': [1.0] * 4, 'b': [1.0] * 4}, index=shifted)
        )
        with pytest.raises(ValueError, match='Incompatible labels'):
            base.merge(other)

    def test_same_labels_in_another_row_order_raises(self):
        """Rows must be identical, order included, so rows are paired as expected."""
        base, _ = _tracker()
        other, _ = _tracker()
        other.provenance_matrix_ = other.provenance_matrix_.iloc[::-1]
        with pytest.raises(ValueError, match='Incompatible labels'):
            base.merge(other)


# =============================================================================
# Panel à couverture hétérogène
# =============================================================================
class TestHeterogeneousCoveragePanel:
    """Marking one entity of the realistic panel leaves the others untouched."""

    @staticmethod
    def _entity_index(panel, entity):
        return pd.MultiIndex.from_product(
            [[entity], panel.loc[entity].index], names=panel.index.names
        )

    def test_initialization_reflects_per_entity_coverage(self, heterogeneous_coverage_panel):
        """climat_affaires has no ORIGINAL cell for Italie, one per observation elsewhere."""
        panel = heterogeneous_coverage_panel
        tracker = ImputationProvenanceTracker().initialize(panel)
        by_entity = tracker.get_mask(P.ORIGINAL, column='climat_affaires').groupby(level=0).sum()
        # Valeur d'or : le nombre de cellules ORIGINAL est le nombre d'observations
        # de l'entité ; Italie n'observe jamais la colonne (couverture structurelle)
        assert by_entity['Italie'] == 0
        assert by_entity.to_dict() == {
            e: int(panel.loc[e, 'climat_affaires'].notna().sum())
            for e in ['France', 'Allemagne', 'Italie']
        }

    def test_marking_italie_leaves_france_and_allemagne_untouched(self, heterogeneous_coverage_panel):
        """Interpolating Italie's climat_affaires modifies no other cell."""
        panel = heterogeneous_coverage_panel
        tracker = ImputationProvenanceTracker().initialize(panel)
        before = tracker.provenance_matrix_.copy()

        tracker.mark_interpolated('climat_affaires', self._entity_index(panel, 'Italie'))
        after = tracker.provenance_matrix_

        # Aucune cellule France / Allemagne (toutes colonnes) n'a bougé
        others = after.index.get_level_values(0) != 'Italie'
        pd.testing.assert_frame_equal(after[others], before[others])
        # Italie : colonne ciblée entièrement INTERPOLATED, autres colonnes inchangées
        italie = after.loc['Italie']
        assert (italie['climat_affaires'] == P.INTERPOLATED).all()
        pd.testing.assert_frame_equal(
            italie.drop(columns='climat_affaires'),
            before.loc['Italie'].drop(columns='climat_affaires'),
        )

    def test_marking_with_a_multiindex_slice_stays_inside_the_entity(self, heterogeneous_coverage_panel):
        """A label slice from (Italie, first) to (Italie, last) stays in Italie."""
        panel = heterogeneous_coverage_panel
        tracker = ImputationProvenanceTracker().initialize(panel)
        dates = panel.loc['Italie'].index
        tracker.mark_aggregated('climat_affaires', slice(('Italie', dates[0]), ('Italie', dates[-1])))

        mask = tracker.get_mask(P.AGGREGATED, column='climat_affaires')
        assert mask.groupby(level=0).sum().to_dict() == {
            'Allemagne': 0, 'France': 0, 'Italie': len(dates),
        }

    def test_statistics_over_the_marked_entity(self, heterogeneous_coverage_panel):
        """Counts add up to the cells marked for Italie, and percentages total 100."""
        panel = heterogeneous_coverage_panel
        tracker = ImputationProvenanceTracker().initialize(panel)
        tracker.mark_interpolated('climat_affaires', self._entity_index(panel, 'Italie'))
        stats = tracker.compute_statistics()
        assert stats['climat_affaires']['interpolated'] == len(panel.loc['Italie'])
        assert sum(v for k, v in stats['overall'].items() if k.endswith('_pct')) == pytest.approx(100.0)
