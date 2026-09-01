"""Tests for tsforecast.frequency.provenance.

Focus §6 de [SPEC] high_frequency_imputer2_architecture.md : échelle de
souillure à cinq libellés MODEL_*, suppression de MODEL_ON_MIXED, primitives
de souillure partagées (resolve_model_provenance, origin_to_taint, max_origin).
"""
# Modules de base
import numpy as np
import pandas as pd
import pytest

# Objet testé
from tsforecast.frequency.provenance import (
    ImputationProvenanceTracker,
    ProvenanceType,
    resolve_model_provenance,
    origin_to_taint,
    max_origin,
)


class TestResolveModelProvenance:
    """Table de correspondance 3x3 -> 5 du §6.3 de [SPEC]."""

    # Les neuf cases sont écrites une par une, pas générées par une boucle sur
    # la fonction testée : la boucle rejouerait l'implémentation.
    def test_resolve_model_provenance_table(self):
        """Les neuf couples (covariate_taint, target_taint) du §6.3."""
        # Ligne covariate_taint = 'none'
        assert resolve_model_provenance('none', 'none') is ProvenanceType.MODEL_ON_TRUE
        assert resolve_model_provenance('none', 'interpolated') is ProvenanceType.MODEL_ON_INTERPOLATED
        assert resolve_model_provenance('none', 'imputed') is ProvenanceType.MODEL_ON_IMPUTED_TARGET

        # Ligne covariate_taint = 'interpolated'
        assert resolve_model_provenance('interpolated', 'none') is ProvenanceType.MODEL_ON_INTERPOLATED
        assert resolve_model_provenance('interpolated', 'interpolated') is ProvenanceType.MODEL_ON_INTERPOLATED
        assert resolve_model_provenance('interpolated', 'imputed') is ProvenanceType.MODEL_ON_IMPUTED_TARGET

        # Ligne covariate_taint = 'imputed'
        assert resolve_model_provenance('imputed', 'none') is ProvenanceType.MODEL_ON_IMPUTED
        assert resolve_model_provenance('imputed', 'interpolated') is ProvenanceType.MODEL_ON_IMPUTED
        assert resolve_model_provenance('imputed', 'imputed') is ProvenanceType.MODEL_ON_IMPUTED_BOTH


class TestModelOnMixedRemoval:
    """Rupture D6 : MODEL_ON_MIXED disparaît sans alias de compatibilité."""

    def test_model_on_mixed_is_gone(self):
        """Ni membre, ni valeur de chaîne résiduelle."""
        # Aucun membre du même nom
        assert getattr(ProvenanceType, 'MODEL_ON_MIXED', None) is None
        # Un accès direct doit lever un AttributeError franc
        with pytest.raises(AttributeError):
            ProvenanceType.MODEL_ON_MIXED  # noqa: B018
        # 'model_on_mixed' n'est la valeur d'aucun membre
        assert 'model_on_mixed' not in {member.value for member in ProvenanceType}


class TestMarkModelImputed:
    """Nouvelle signature de mark_model_imputed (§6.6)."""

    @staticmethod
    def _tracker():
        dates = pd.date_range('2023-01-01', periods=5, freq='MS')
        data = pd.DataFrame({'a': [1.0, np.nan, np.nan, np.nan, np.nan]}, index=dates)
        tracker = ImputationProvenanceTracker()
        tracker.initialize(data)
        return tracker, dates

    def test_mark_model_imputed_defaults_to_model_on_true(self):
        """Sans souillure passée, la cellule est MODEL_ON_TRUE."""
        tracker, dates = self._tracker()
        tracker.mark_model_imputed('a', dates[1])
        assert tracker.get_provenance('a', dates[1]) is ProvenanceType.MODEL_ON_TRUE

    def test_mark_model_imputed_taints(self):
        """Les cinq libellés MODEL_* sont atteignables via la signature."""
        tracker, dates = self._tracker()

        tracker.mark_model_imputed('a', dates[0], covariate_taint='none', target_taint='none')
        tracker.mark_model_imputed('a', dates[1], covariate_taint='interpolated', target_taint='none')
        tracker.mark_model_imputed('a', dates[2], covariate_taint='imputed', target_taint='none')
        tracker.mark_model_imputed('a', dates[3], covariate_taint='none', target_taint='imputed')
        tracker.mark_model_imputed('a', dates[4], covariate_taint='imputed', target_taint='imputed')

        assert tracker.get_provenance('a', dates[0]) is ProvenanceType.MODEL_ON_TRUE
        assert tracker.get_provenance('a', dates[1]) is ProvenanceType.MODEL_ON_INTERPOLATED
        assert tracker.get_provenance('a', dates[2]) is ProvenanceType.MODEL_ON_IMPUTED
        assert tracker.get_provenance('a', dates[3]) is ProvenanceType.MODEL_ON_IMPUTED_TARGET
        assert tracker.get_provenance('a', dates[4]) is ProvenanceType.MODEL_ON_IMPUTED_BOTH


class TestMarkInterpolated:
    """Nouvelle méthode de commodité mark_interpolated (§6.1)."""

    def test_mark_interpolated_writes_interpolated(self):
        """La cellule marquée porte ProvenanceType.INTERPOLATED."""
        dates = pd.date_range('2023-01-01', periods=3, freq='MS')
        data = pd.DataFrame({'a': [1.0, np.nan, np.nan]}, index=dates)
        tracker = ImputationProvenanceTracker()
        tracker.initialize(data)

        tracker.mark_interpolated('a', dates[1:3])

        assert tracker.get_provenance('a', dates[1]) is ProvenanceType.INTERPOLATED
        assert tracker.get_provenance('a', dates[2]) is ProvenanceType.INTERPOLATED


class TestTaintPrimitives:
    """origin_to_taint et max_origin (§6.2)."""

    def test_origin_to_taint_and_max_origin(self):
        """Correspondance origine -> souillure et maximum sur l'ordre."""
        # Correspondance
        assert origin_to_taint('observed') == 'none'
        assert origin_to_taint('interpolated') == 'interpolated'
        assert origin_to_taint('model') == 'imputed'

        # Ordre croissant de souillure : observed < interpolated < model
        assert max_origin(['observed', 'observed']) == 'observed'
        assert max_origin(['observed', 'interpolated']) == 'interpolated'
        assert max_origin(['interpolated', 'model', 'observed']) == 'model'
        assert max_origin(['model', 'interpolated']) == 'model'

        # Itérable vide -> 'observed' (origine la moins souillée, neutre)
        assert max_origin([]) == 'observed'


class TestStatisticsCoverAllProvenanceTypes:
    """compute_statistics n'oublie aucune constante (§6, point 6)."""

    def test_statistics_cover_all_provenance_types(self):
        """Chaque membre de ProvenanceType a une entrée dans les statistiques."""
        dates = pd.date_range('2023-01-01', periods=len(ProvenanceType) + 1, freq='MS')
        data = pd.DataFrame(
            {'a': [1.0] + [np.nan] * len(ProvenanceType)}, index=dates
        )
        tracker = ImputationProvenanceTracker()
        tracker.initialize(data)

        # Écriture d'une cellule par type de provenance
        for offset, prov_type in enumerate(ProvenanceType, start=1):
            tracker.mark_imputed('a', dates[offset], prov_type)

        stats = tracker.compute_statistics()

        for prov_type in ProvenanceType:
            assert prov_type.value in stats['overall']
            assert f'{prov_type.value}_pct' in stats['overall']
            # Chaque type a été écrit au moins une fois : aucune constante
            # n'est oubliée par l'itération de compute_statistics
            assert stats['overall'][prov_type.value] >= 1
            assert prov_type.value in stats['a']

    def test_to_string_matrix_covers_all_provenance_types(self):
        """to_string_matrix rend la valeur de chaîne de chaque membre."""
        dates = pd.date_range('2023-01-01', periods=len(ProvenanceType) + 1, freq='MS')
        data = pd.DataFrame(
            {'a': [1.0] + [np.nan] * len(ProvenanceType)}, index=dates
        )
        tracker = ImputationProvenanceTracker()
        tracker.initialize(data)
        for offset, prov_type in enumerate(ProvenanceType, start=1):
            tracker.mark_imputed('a', dates[offset], prov_type)

        string_col = set(tracker.to_string_matrix()['a'])
        for prov_type in ProvenanceType:
            assert prov_type.value in string_col
