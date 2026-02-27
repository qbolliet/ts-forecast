"""Tests for TargetFrequencyValidator class.

Tests validation of target frequencies against detected data frequencies,
for both time series and panel data scenarios.
"""
import pytest
import warnings

from tsforecast.frequency.target_frequency_validator import TargetFrequencyValidator


class TestInferStructure:
    """Tests for _infer_structure: detection of panel vs time series from keys."""

    def setup_method(self):
        self.validator = TargetFrequencyValidator()

    def test_string_keys_inferred_as_timeseries(self):
        """Clés string → séries temporelles."""
        detected = {'col_a': 'M', 'col_b': 'Q'}
        is_panel, entities = self.validator._infer_structure(detected)
        assert is_panel is False
        assert entities is None

    def test_tuple_keys_inferred_as_panel(self):
        """Clés tuple → panel."""
        detected = {('FR', 'gdp'): 'M', ('DE', 'gdp'): 'Q'}
        is_panel, entities = self.validator._infer_structure(detected)
        assert is_panel is True
        assert set(entities) == {('FR',), ('DE',)}

    def test_multi_level_entity_keys(self):
        """Entités multi-niveaux (ex: pays + région)."""
        detected = {('FR', 'IDF', 'gdp'): 'M', ('DE', 'BAY', 'gdp'): 'Q'}
        is_panel, entities = self.validator._infer_structure(detected)
        assert is_panel is True
        assert set(entities) == {('FR', 'IDF'), ('DE', 'BAY')}

    def test_empty_detected_raises(self):
        """detected_frequencies vide → ValueError."""
        with pytest.raises(ValueError, match="empty"):
            self.validator._infer_structure({})

    def test_mixed_keys_raises(self):
        """Clés mixtes (str + tuple) → ValueError."""
        detected = {'col_a': 'M', ('FR', 'gdp'): 'Q'}
        with pytest.raises(ValueError, match="mixed key types"):
            self.validator._infer_structure(detected)


class TestTargetFrequencyValidatorTimeseries:
    """Tests for time series validation."""

    def setup_method(self):
        self.validator = TargetFrequencyValidator()

    def test_valid_target_lower_than_highest(self):
        """Fréquence cible inférieure ou égale à la plus haute : pas d'erreur."""
        detected = {'col_daily': 'D', 'col_monthly': 'M'}
        result = self.validator.validate(
            target_frequency='M',
            detected_frequencies=detected,
        )
        assert result == 'M'

    def test_valid_target_equals_highest(self):
        """Fréquence cible exactement égale à la plus haute."""
        detected = {'col_daily': 'D', 'col_monthly': 'M'}
        result = self.validator.validate(
            target_frequency='D',
            detected_frequencies=detected,
        )
        assert result == 'D'

    def test_mismatch_error_mode(self):
        """Fréquence cible plus haute que données → ValueError par défaut."""
        detected = {'col_monthly': 'M', 'col_quarterly': 'Q'}
        with pytest.raises(ValueError, match="higher than"):
            self.validator.validate(
                target_frequency='D',
                detected_frequencies=detected,
                on_frequency_mismatch='error',
            )

    def test_mismatch_warn_mode(self):
        """Fréquence cible plus haute → warning + ajustement en mode 'warn'."""
        detected = {'col_monthly': 'M', 'col_quarterly': 'Q'}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.validator.validate(
                target_frequency='D',
                detected_frequencies=detected,
                on_frequency_mismatch='warn',
            )
        assert result == 'M'
        assert len(w) >= 1
        assert "Adjusting" in str(w[-1].message)

    def test_dict_target_for_timeseries_raises(self):
        """Un dict pour target_frequency avec des données non-panel → ValueError."""
        detected = {'col_daily': 'D'}
        with pytest.raises(ValueError, match="cannot be a dict"):
            self.validator.validate(
                target_frequency={'entity_A': 'M'},
                detected_frequencies=detected,
            )

    def test_no_valid_frequencies_raises(self):
        """Aucune fréquence valide détectée → ValueError."""
        detected = {'col_a': None, 'col_b': None}
        with pytest.raises(ValueError, match="No valid frequencies"):
            self.validator.validate(
                target_frequency='M',
                detected_frequencies=detected,
            )

    def test_single_column(self):
        """Validation avec une seule colonne détectée."""
        detected = {'col_monthly': 'M'}
        result = self.validator.validate(
            target_frequency='Q',
            detected_frequencies=detected,
        )
        assert result == 'Q'

    def test_empty_detected_raises(self):
        """detected_frequencies vide → ValueError (structure indéterminable)."""
        with pytest.raises(ValueError, match="empty"):
            self.validator.validate(
                target_frequency='M',
                detected_frequencies={},
            )


class TestTargetFrequencyValidatorPanel:
    """Tests for panel data validation.

    Keys in detected_frequencies follow the format produced by
    detect_dataset_frequency: flat tuples (entity..., var) where entity
    components are strings (e.g. ('FR', 'gdp') for a single-level panel).
    """

    def setup_method(self):
        self.validator = TargetFrequencyValidator()

    def test_valid_panel_all_entities(self):
        """Toutes les entités ont une fréquence cible compatible."""
        detected = {
            ('FR', 'gdp'): 'M',
            ('FR', 'cpi'): 'Q',
            ('DE', 'gdp'): 'M',
            ('DE', 'cpi'): 'Q',
        }
        target = {('FR',): 'Q', ('DE',): 'Q'}
        result = self.validator.validate(
            target_frequency=target,
            detected_frequencies=detected,
        )
        assert ('FR',) in result
        assert ('DE',) in result
        assert result[('FR',)] == 'Q'

    def test_missing_entity_raises(self):
        """Entités manquantes dans target_frequency → ValueError."""
        detected = {
            ('FR', 'gdp'): 'M',
            ('DE', 'gdp'): 'M',
        }
        target = {('FR',): 'M'}  # DE manquant
        with pytest.raises(ValueError, match="missing entries"):
            self.validator.validate(
                target_frequency=target,
                detected_frequencies=detected,
            )

    def test_extra_entity_warns(self):
        """Entités supplémentaires dans target_frequency → warning."""
        detected = {
            ('FR', 'gdp'): 'M',
        }
        target = {('FR',): 'M', ('US',): 'Q'}  # US pas dans les données
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.validator.validate(
                target_frequency=target,
                detected_frequencies=detected,
            )
        assert len(w) >= 1
        assert "not in data" in str(w[-1].message)

    def test_panel_mismatch_error(self):
        """Fréquence cible plus haute par entité → ValueError en mode error."""
        detected = {
            ('FR', 'gdp'): 'Q',
        }
        target = {('FR',): 'D'}  # D > Q
        with pytest.raises(ValueError, match="higher than highest"):
            self.validator.validate(
                target_frequency=target,
                detected_frequencies=detected,
                on_frequency_mismatch='error',
            )

    def test_panel_mismatch_warn(self):
        """Fréquence cible plus haute par entité → ajustement en mode warn."""
        detected = {
            ('FR', 'gdp'): 'Q',
        }
        target = {('FR',): 'D'}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.validator.validate(
                target_frequency=target,
                detected_frequencies=detected,
                on_frequency_mismatch='warn',
            )
        assert result[('FR',)] == 'Q'

    def test_string_target_panel_applied_to_all_entities(self):
        """String target_frequency pour panel : appliqué à chaque entité."""
        detected = {
            ('FR', 'gdp'): 'M',
            ('DE', 'gdp'): 'M',
        }
        result = self.validator.validate(
            target_frequency='Q',
            detected_frequencies=detected,
        )
        assert result[('FR',)] == 'Q'
        assert result[('DE',)] == 'Q'

    def test_multi_level_entity(self):
        """Entités multi-niveaux correctement inférées."""
        detected = {
            ('FR', 'IDF', 'gdp'): 'M',
            ('DE', 'BAY', 'gdp'): 'Q',
        }
        target = {('FR', 'IDF'): 'Q', ('DE', 'BAY'): 'Y'}
        result = self.validator.validate(
            target_frequency=target,
            detected_frequencies=detected,
        )
        assert result[('FR', 'IDF')] == 'Q'
        assert result[('DE', 'BAY')] == 'Y'


class TestGetHighestFrequency:
    """Tests for internal frequency detection helpers."""

    def setup_method(self):
        self.validator = TargetFrequencyValidator()

    def test_highest_frequency_timeseries(self):
        """Identifie correctement la fréquence la plus granulaire."""
        detected = {'col_daily': 'D', 'col_monthly': 'M', 'col_quarterly': 'Q'}
        result = self.validator._get_highest_frequency_timeseries(detected)
        assert result == 'D'

    def test_highest_frequency_entity(self):
        """Identifie la fréquence la plus granulaire pour une entité."""
        detected = {
            ('FR', 'gdp'): 'M',
            ('FR', 'cpi'): 'Q',
            ('DE', 'gdp'): 'D',
        }
        result = self.validator._get_highest_frequency_entity(('FR',), detected)
        assert result == 'M'

    def test_highest_frequency_entity_not_found(self):
        """Entité absente des données détectées → ValueError."""
        detected = {
            ('FR', 'gdp'): 'M',
        }
        with pytest.raises(ValueError, match="No valid frequencies"):
            self.validator._get_highest_frequency_entity(('US',), detected)
