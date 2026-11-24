"""Provenance vocabulary and taint primitives.

Scenarios: ``ProvenanceType`` members and serialization, ``resolve_model_provenance``
truth table, origin-to-taint mapping, the maximum over the taint order (iterable and
series forms) and the ordered categorical dtype of cell origins.
"""

# Modules de base
import itertools
import json

import pandas as pd
import pytest

# Objet testé
from tsforecast.imputation._components.provenance import (
    CELL_ORIGIN_DTYPE,
    max_cell_origin,
    max_origin,
    origin_to_taint,
    resolve_model_provenance,
)

# Outils de test partagés du paquet
from tests.unit.imputation._components.provenance.helpers import P, _tracker


# Les cinq libellés MODEL_* de l'échelle de souillure
_MODEL_SCALE = (
    P.MODEL_ON_TRUE,
    P.MODEL_ON_INTERPOLATED,
    P.MODEL_ON_IMPUTED,
    P.MODEL_ON_IMPUTED_TARGET,
    P.MODEL_ON_IMPUTED_BOTH,
)


# =============================================================================
# ProvenanceType
# =============================================================================
class TestProvenanceType:
    """Members, partition and ``str`` serialization of the enumeration."""

    def test_exactly_ten_members_with_expected_values(self):
        """The enumeration holds the ten members, values in snake_case."""
        # Valeurs d'or : copie littérale des valeurs documentées
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
        """MODEL_ON_MIXED is not a member, and has no alias."""
        with pytest.raises(AttributeError):
            P.MODEL_ON_MIXED  # noqa: B018
        assert 'model_on_mixed' not in {m.value for m in P}


# =============================================================================
# resolve_model_provenance / origin_to_taint / max_origin
# =============================================================================
class TestResolveModelProvenance:
    """Truth table 3 x 3 -> 5."""

    # Valeurs d'or écrites une à une depuis la table de vérité (pas de boucle
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
        """Each of the nine (covariate_taint, target_taint) pairs."""
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
        """A mistyped taint raises instead of reading as clean."""
        with pytest.raises(ValueError, match=culprit):
            resolve_model_provenance(covariate_taint, target_taint)

    def test_mark_model_imputed_rejects_an_unknown_taint(self):
        """The validation reaches ``mark_model_imputed``, which writes nothing."""
        tracker, dates = _tracker()
        with pytest.raises(ValueError, match='covariate_taint'):
            tracker.mark_model_imputed('a', dates[1], covariate_taint='interpolate')
        assert pd.isna(tracker.get_provenance('a', dates[1]))


class TestOriginToTaint:
    """Origin -> taint correspondence."""

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


class TestCellOriginDtype:
    """``CELL_ORIGIN_DTYPE``: closed vocabulary, ordered by increasing taint."""

    def test_categories_follow_the_taint_order(self):
        """The categories are the three origins, ordered."""
        assert list(CELL_ORIGIN_DTYPE.categories) == TestMaxOrigin._ORDER
        assert CELL_ORIGIN_DTYPE.ordered

    def test_value_outside_the_vocabulary_is_refused(self):
        """Writing an unknown origin into a typed series raises."""
        origins = pd.Series(['observed', 'model'], dtype=CELL_ORIGIN_DTYPE)
        with pytest.raises(TypeError):
            origins.iloc[0] = 'aggregated'

    def test_cells_read_back_as_plain_strings(self):
        """A cell compares equal to its string, a missing one is NaN."""
        origins = pd.Series(['model', None], dtype=CELL_ORIGIN_DTYPE)
        assert origins.iloc[0] == 'model' and pd.isna(origins.iloc[1])


class TestMaxCellOrigin:
    """``max_cell_origin`` is the series form of ``max_origin``."""

    @pytest.mark.parametrize(
        'left, right', list(itertools.product(TestMaxOrigin._ORDER, repeat=2)), ids=lambda v: str(v)
    )
    def test_agrees_with_max_origin(self, left, right):
        """Same maximum as ``max_origin`` on every pair, typed or plain strings."""
        expected = max_origin([left, right])
        assert max_cell_origin(pd.Series([left, right], dtype=CELL_ORIGIN_DTYPE)) == expected
        assert max_cell_origin(pd.Series([left, right], dtype='str')) == expected

    @pytest.mark.parametrize(
        'origins',
        [pd.Series([], dtype=CELL_ORIGIN_DTYPE), pd.Series([None, None], dtype=CELL_ORIGIN_DTYPE)],
        ids=['empty', 'all_missing'],
    )
    def test_no_origin_is_observed(self, origins):
        """Without any origin, the neutral, least-tainted origin."""
        assert max_cell_origin(origins) == 'observed'

    def test_missing_cells_are_ignored(self):
        """Missing cells take no part in the maximum."""
        assert max_cell_origin(pd.Series([None, 'interpolated', None])) == 'interpolated'

    def test_result_is_a_plain_string(self):
        """The result is a ``str``, usable as a dictionary key of ``origin_to_taint``."""
        result = max_cell_origin(pd.Series(['model'], dtype=CELL_ORIGIN_DTYPE))
        assert type(result) is str and origin_to_taint(result) == 'imputed'

    def test_unknown_origin_raises(self):
        """An unknown origin raises instead of being ignored."""
        with pytest.raises(ValueError, match='bogus'):
            max_cell_origin(pd.Series(['observed', 'bogus']))
