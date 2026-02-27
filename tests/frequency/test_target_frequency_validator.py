"""Tests for TargetFrequencyValidator class.

Tests validation of target frequencies against detected data frequencies,
for both time series and panel data scenarios.
"""
import pytest
import warnings

from tsforecast.frequency.target_frequency_validator import TargetFrequencyValidator


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
            is_panel=False,
            entities=None,
        )
        assert result == 'M'

    def test_valid_target_equals_highest(self):
        """Fréquence cible exactement égale à la plus haute."""
        detected = {'col_daily': 'D', 'col_monthly': 'M'}
        result = self.validator.validate(
            target_frequency='D',
            detected_frequencies=detected,
            is_panel=False,
            entities=None,
        )
        assert result == 'D'

    def test_mismatch_error_mode(self):
        """Fréquence cible plus haute que données → ValueError par défaut."""
        detected = {'col_monthly': 'M', 'col_quarterly': 'Q'}
        with pytest.raises(ValueError, match="higher than"):
            self.validator.validate(
                target_frequency='D',
                detected_frequencies=detected,
                is_panel=False,
                entities=None,
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
                is_panel=False,
                entities=None,
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
                is_panel=False,
                entities=None,
            )

    def test_no_valid_frequencies_raises(self):
        """Aucune fréquence valide détectée → ValueError."""
        detected = {'col_a': None, 'col_b': None}
        with pytest.raises(ValueError, match="No valid frequencies"):
            self.validator.validate(
                target_frequency='M',
                detected_frequencies=detected,
                is_panel=False,
                entities=None,
            )

    def test_single_column(self):
        """Validation avec une seule colonne détectée."""
        detected = {'col_monthly': 'M'}
        result = self.validator.validate(
            target_frequency='Q',
            detected_frequencies=detected,
            is_panel=False,
            entities=None,
        )
        assert result == 'Q'


class TestTargetFrequencyValidatorPanel:
    """Tests for panel data validation."""

    def setup_method(self):
        self.validator = TargetFrequencyValidator()

    def test_valid_panel_all_entities(self):
        """Toutes les entités ont une fréquence cible compatible."""
        detected = {
            (('FR',), 'gdp'): 'M',
            (('FR',), 'cpi'): 'Q',
            (('DE',), 'gdp'): 'M',
            (('DE',), 'cpi'): 'Q',
        }
        entities = [('FR',), ('DE',)]
        target = {('FR',): 'Q', ('DE',): 'Q'}
        result = self.validator.validate(
            target_frequency=target,
            detected_frequencies=detected,
            is_panel=True,
            entities=entities,
        )
        assert ('FR',) in result
        assert ('DE',) in result
        assert result[('FR',)] == 'Q'

    def test_missing_entity_raises(self):
        """Entités manquantes dans target_frequency → ValueError."""
        detected = {
            (('FR',), 'gdp'): 'M',
            (('DE',), 'gdp'): 'M',
        }
        entities = [('FR',), ('DE',)]
        target = {('FR',): 'M'}  # DE manquant
        with pytest.raises(ValueError, match="missing entries"):
            self.validator.validate(
                target_frequency=target,
                detected_frequencies=detected,
                is_panel=True,
                entities=entities,
            )

    def test_extra_entity_warns(self):
        """Entités supplémentaires dans target_frequency → warning."""
        detected = {
            (('FR',), 'gdp'): 'M',
        }
        entities = [('FR',)]
        target = {('FR',): 'M', ('US',): 'Q'}  # US pas dans les données
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.validator.validate(
                target_frequency=target,
                detected_frequencies=detected,
                is_panel=True,
                entities=entities,
            )
        assert len(w) >= 1
        assert "not in data" in str(w[-1].message)

    def test_panel_mismatch_error(self):
        """Fréquence cible plus haute par entité → ValueError en mode error."""
        detected = {
            (('FR',), 'gdp'): 'Q',
        }
        entities = [('FR',)]
        target = {('FR',): 'D'}  # D > Q
        with pytest.raises(ValueError, match="higher than highest"):
            self.validator.validate(
                target_frequency=target,
                detected_frequencies=detected,
                is_panel=True,
                entities=entities,
                on_frequency_mismatch='error',
            )

    def test_panel_mismatch_warn(self):
        """Fréquence cible plus haute par entité → ajustement en mode warn."""
        detected = {
            (('FR',), 'gdp'): 'Q',
        }
        entities = [('FR',)]
        target = {('FR',): 'D'}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.validator.validate(
                target_frequency=target,
                detected_frequencies=detected,
                is_panel=True,
                entities=entities,
                on_frequency_mismatch='warn',
            )
        assert result[('FR',)] == 'Q'

    def test_string_target_panel_delegates(self):
        """String target_frequency pour panel : traité comme target pour chaque entité."""
        detected = {
            (('FR',), 'gdp'): 'M',
            (('DE',), 'gdp'): 'M',
        }
        entities = [('FR',), ('DE',)]
        result = self.validator.validate(
            target_frequency='Q',
            detected_frequencies=detected,
            is_panel=True,
            entities=entities,
        )
        assert result[('FR',)] == 'Q'
        assert result[('DE',)] == 'Q'


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
            (('FR',), 'gdp'): 'M',
            (('FR',), 'cpi'): 'Q',
            (('DE',), 'gdp'): 'D',
        }
        result = self.validator._get_highest_frequency_entity(('FR',), detected)
        assert result == 'M'

    def test_highest_frequency_entity_not_found(self):
        """Entité absente des données détectées → ValueError."""
        detected = {
            (('FR',), 'gdp'): 'M',
        }
        with pytest.raises(ValueError, match="No valid frequencies"):
            self.validator._get_highest_frequency_entity(('US',), detected)
