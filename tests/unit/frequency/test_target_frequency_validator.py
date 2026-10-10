"""Tests for TargetFrequencyValidator class.

Tests validation of target frequencies against detected data frequencies,
for both time series and panel data scenarios.
"""
import pytest
import warnings

from tsforecast.frequency.target_frequency_validator import TargetFrequencyValidator
from tsforecast.utils.frequency.utils import detect_dataset_frequency


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


class TestUnusableDetectedFrequencies:
    """Detected frequencies that are missing (None) or unknown to the frequency scale."""

    def setup_method(self):
        self.validator = TargetFrequencyValidator()

    def test_timeseries_with_only_none_raises(self):
        """Time series whose columns are all undetected → ValueError."""
        with pytest.raises(ValueError, match="No valid frequencies"):
            self.validator.validate('M', {'a': None, 'b': None})

    def test_timeseries_with_unknown_frequency_label_raises(self):
        """A label with no known order cannot be ranked → ValueError."""
        with pytest.raises(ValueError, match="Could not determine frequency order"):
            self.validator.validate('M', {'a': 'zzz'})

    def test_timeseries_ignores_none_columns(self):
        """A column without detected frequency does not take part in the ranking."""
        assert self.validator.validate('M', {'a': 'D', 'b': None}) == 'M'

    def test_unknown_label_is_skipped_when_another_one_is_ranked(self):
        """An unknown label is dropped, the known one decides."""
        detected = {'a': 'zzz', 'b': 'M'}
        assert self.validator._get_highest_frequency_timeseries(detected) == 'M'

    def test_unknown_target_frequency_raises(self):
        """A target that is not a frequency → ValueError from the frequency scale."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            self.validator.validate('zzz', {'a': 'M'})


class TestPanelEntityWithoutDetectedFrequency:
    """Panel entity whose columns all lack a detected frequency (target absent for it)."""

    def setup_method(self):
        self.validator = TargetFrequencyValidator()
        # L'entité A n'observe aucune colonne : fréquences toutes None
        self.detected = {('A', 'y'): None, ('A', 'x'): None, ('B', 'y'): 'MS'}

    def test_entity_without_frequency_raises(self):
        """An entity left without any frequency is an error, not a silent omission."""
        with pytest.raises(ValueError, match="No valid frequencies detected for entity"):
            self.validator.validate('MS', self.detected)

    def test_error_is_raised_in_warn_mode_too(self):
        """``on_frequency_mismatch='warn'`` only softens frequency mismatches."""
        with pytest.raises(ValueError, match="No valid frequencies detected for entity"):
            self.validator.validate('MS', self.detected, on_frequency_mismatch='warn')


class TestOnFrequencyMismatchValues:
    """``on_frequency_mismatch`` must be one of the supported strategies."""

    @pytest.mark.parametrize('detected', [{'a': 'M'}, {('A', 'y'): 'MS'}], ids=['timeseries', 'panel'])
    @pytest.mark.parametrize('value', ['ignore', 'ERROR', '', None])
    def test_unsupported_value_raises(self, detected, value):
        """Any value other than 'error' / 'warn' → ValueError listing the admitted values."""
        with pytest.raises(ValueError, match=r"on_frequency_mismatch must be one of \['error', 'warn'\]"):
            TargetFrequencyValidator().validate('M', detected, on_frequency_mismatch=value)

    def test_invalid_value_is_rejected_even_without_mismatch(self):
        """The check does not wait for a mismatch to happen."""
        with pytest.raises(ValueError, match="on_frequency_mismatch"):
            TargetFrequencyValidator().validate('M', {'a': 'D'}, on_frequency_mismatch='ignore')


class TestPanelManyInvalidEntities:
    """Error message when more than five entities exceed their highest frequency."""

    def setup_method(self):
        self.validator = TargetFrequencyValidator()
        self.detected = {(f'E{i}', 'y'): 'MS' for i in range(7)}

    def test_message_lists_five_entities_and_counts_the_rest(self):
        """Seven invalid entities: five listed, 'and 2 more entities'."""
        with pytest.raises(ValueError, match=r"(?s)7 entities.*and 2 more entities"):
            self.validator.validate('D', self.detected)

    def test_warn_mode_adjusts_every_entity(self):
        """Warn mode returns the entity-specific highest frequency for all seven."""
        with pytest.warns(UserWarning, match="Adjusting target frequencies"):
            result = self.validator.validate('D', self.detected, on_frequency_mismatch='warn')
        assert result == {(f'E{i}',): 'MS' for i in range(7)}


class TestPanelTargetLowerThanData:
    """Target frequency lower than the data frequency is always accepted."""

    def setup_method(self):
        self.validator = TargetFrequencyValidator()

    def test_lower_target_is_kept(self):
        """Yearly target on monthly data is returned unchanged."""
        assert self.validator.validate('YS', {('A', 'y'): 'MS'}) == {('A',): 'YS'}

    def test_lower_target_in_dict_is_kept_per_entity(self):
        """Each entity keeps its own, lower, target."""
        detected = {('A', 'y'): 'MS', ('B', 'y'): 'D'}
        result = self.validator.validate({('A',): 'QS', ('B',): 'MS'}, detected)
        assert result == {('A',): 'QS', ('B',): 'MS'}


class TestHeterogeneousCoveragePanel:
    """Realistic panel: per-entity coverage, publication frequency and absent column."""

    @pytest.fixture
    def detected(self, heterogeneous_coverage_panel):
        """Detected frequencies, full format, of the heterogeneous-coverage panel."""
        return detect_dataset_frequency(heterogeneous_coverage_panel, return_format='full')

    def test_monthly_target_is_valid_for_every_entity(self, detected):
        """Every entity observes a monthly column: a monthly target is valid."""
        result = TargetFrequencyValidator().validate('MS', detected)
        assert result == {('France',): 'MS', ('Allemagne',): 'MS', ('Italie',): 'MS'}

    def test_daily_target_is_rejected_in_error_mode(self, detected):
        """Daily target exceeds the monthly grid of all three entities."""
        with pytest.raises(ValueError, match="3 entities"):
            TargetFrequencyValidator().validate('D', detected)

    def test_daily_target_is_clamped_to_monthly_in_warn_mode(self, detected):
        """Warn mode lowers every entity to its own highest frequency, 'MS'."""
        with pytest.warns(UserWarning):
            result = TargetFrequencyValidator().validate('D', detected, on_frequency_mismatch='warn')
        assert set(result.values()) == {'MS'}

    def test_dict_target_with_entity_specific_frequencies(self, detected):
        """A dict target gives each entity its own frequency."""
        target = {('France',): 'MS', ('Allemagne',): 'QS', ('Italie',): 'YS'}
        assert TargetFrequencyValidator().validate(target, detected) == target

    def test_dict_target_with_unknown_entity_warns_and_ignores_it(self, detected):
        """An entity absent from the data is ignored with a warning."""
        target = {('France',): 'MS', ('Allemagne',): 'MS', ('Italie',): 'MS', ('Espagne',): 'MS'}
        with pytest.warns(UserWarning, match="not in data"):
            result = TargetFrequencyValidator().validate(target, detected)
        assert ('Espagne',) not in result

    def test_dict_target_missing_an_entity_raises(self, detected):
        """Entities missing from the dict are named in the error."""
        with pytest.raises(ValueError, match="missing entries for entities"):
            TargetFrequencyValidator().validate({('France',): 'MS'}, detected)

    def test_dict_target_with_bare_string_keys_is_rejected(self, detected):
        """Entity keys must be tuples: plain strings do not match."""
        target = {'France': 'MS', 'Allemagne': 'MS', 'Italie': 'MS'}
        with pytest.raises(ValueError, match="missing entries for entities"):
            TargetFrequencyValidator().validate(target, detected)

    def test_target_column_absent_for_one_entity(self, detected):
        """'climat_affaires' is undetected for Italie: the entity still has other columns."""
        # Valeur d'or : la colonne cible est sans fréquence pour Italie, mais pas l'entité
        assert detected[('Italie', 'climat_affaires')] is None
        result = TargetFrequencyValidator().validate('MS', detected)
        assert ('Italie',) in result

    def test_entity_whose_columns_are_all_undetected_raises(self, detected):
        """Blanking every Italie column is an error naming the entity."""
        blanked = {k: (None if k[0] == 'Italie' else v) for k, v in detected.items()}
        with pytest.raises(ValueError, match="No valid frequencies detected for entity 'Italie'|Italie"):
            TargetFrequencyValidator().validate('MS', blanked)
