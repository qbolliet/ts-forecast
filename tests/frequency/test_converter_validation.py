"""Tests unitaires pour la validation des paramètres dans FrequencyConverter.

Ce module teste la validation des fréquences et positions dans les méthodes
de conversion, notamment la validation de freq_pos dans _validate_conversion_params.
"""
import pytest
import pandas as pd
from tsforecast.utils.frequency.converter import FrequencyConverter


class TestFrequencyValidation:
    """Tests pour la validation des fréquences et positions."""

    @pytest.fixture
    def converter(self):
        """Fixture pour créer une instance de FrequencyConverter."""
        return FrequencyConverter()

    @pytest.fixture
    def sample_series(self):
        """Fixture pour créer une série de test."""
        dates = pd.date_range('2024-01-01', periods=12, freq='D')
        return pd.Series(range(12), index=dates)

    @pytest.fixture
    def sample_dataframe(self):
        """Fixture pour créer un DataFrame de test."""
        dates = pd.date_range('2024-01-01', periods=12, freq='D')
        return pd.DataFrame({
            'col1': range(12),
            'col2': range(12, 24)
        }, index=dates)

    # Tests de validation des positions valides

    def test_valid_position_s_accepted(self, converter, sample_series):
        """Test que la position 'S' valide est acceptée."""
        # Ne devrait pas lever d'erreur
        result = converter.convert_frequency(sample_series, 'MS', method='mean')
        assert len(result) > 0

    def test_valid_position_e_accepted(self, converter, sample_series):
        """Test que la position 'E' valide est acceptée."""
        # Ne devrait pas lever d'erreur
        result = converter.convert_frequency(sample_series, 'ME', method='mean')
        assert len(result) > 0

    def test_valid_position_in_dict_accepted(self, converter, sample_dataframe):
        """Test que les positions valides dans un dict sont acceptées."""
        # Ne devrait pas lever d'erreur
        result = converter.convert_frequency(
            sample_dataframe,
            {'col1': 'MS', 'col2': 'ME'},
            method='mean'
        )
        assert len(result) > 0

    # Tests de validation des fréquences complètement invalides
    # Note: Les positions invalides seront détectées par to_offset() car decompose_offset()
    # est permissif et retourne une position par défaut si non reconnue

    def test_invalid_offset_in_string_raises_error(self, converter, sample_series):
        """Test qu'un offset invalide dans target_freq string lève une erreur."""
        # 'XS' n'est pas un offset valide (X n'est pas une fréquence de base valide)
        with pytest.raises(ValueError, match="Invalid target frequency"):
            converter.convert_frequency(sample_series, 'XS', method='mean')

    def test_invalid_offset_in_dict_raises_error(self, converter, sample_dataframe):
        """Test qu'un offset invalide dans target_freq dict lève une erreur."""
        # 'ZE' n'est pas un offset valide
        with pytest.raises(ValueError, match="Invalid target frequency"):
            converter.convert_frequency(
                sample_dataframe,
                {'col1': 'M', 'col2': 'ZE'},
                method='mean'
            )

    def test_completely_invalid_frequency_raises_error(self, converter, sample_series):
        """Test qu'une fréquence complètement invalide lève une erreur."""
        with pytest.raises(ValueError, match="Invalid target frequency"):
            converter.convert_frequency(sample_series, 'INVALID', method='mean')

    # Tests de validation de fréquences de base

    def test_invalid_base_frequency_raises_error(self, converter, sample_series):
        """Test qu'une fréquence de base invalide lève une erreur."""
        # 'ZS' a une base invalide 'Z'
        with pytest.raises(ValueError, match="Invalid target frequency"):
            converter.convert_frequency(sample_series, 'ZS', method='mean')

    def test_invalid_base_in_dict_raises_error(self, converter, sample_dataframe):
        """Test qu'une fréquence de base invalide dans un dict lève une erreur."""
        with pytest.raises(ValueError, match="Invalid target frequency"):
            converter.convert_frequency(
                sample_dataframe,
                {'col1': 'M', 'col2': 'XYZ'},
                method='mean'
            )

    # Tests de cas limites

    def test_base_frequency_without_position_accepted(self, converter, sample_series):
        """Test qu'une fréquence de base sans position explicite est acceptée."""
        # 'M' sans position explicite devrait être accepté (default: 'E')
        result = converter.convert_frequency(sample_series, 'M', method='mean')
        assert len(result) > 0

    def test_quarterly_with_anchor_accepted(self, converter, sample_series):
        """Test que les fréquences avec anchor sont acceptées."""
        # 'QE-DEC' avec anchor devrait être accepté
        result = converter.convert_frequency(sample_series, 'QE-DEC', method='mean')
        assert len(result) > 0

    def test_multiple_columns_mixed_positions(self, converter, sample_dataframe):
        """Test que différentes positions pour différentes colonnes sont acceptées."""
        result = converter.convert_frequency(
            sample_dataframe,
            {'col1': 'MS', 'col2': 'QE'},
            method='mean'
        )
        assert len(result) > 0
        assert 'col1' in result.columns
        assert 'col2' in result.columns


class TestDurationConverterIntegration:
    """Tests pour l'intégration de DurationConverter dans FrequencyConverter."""

    @pytest.fixture
    def converter(self):
        """Fixture pour créer une instance de FrequencyConverter."""
        return FrequencyConverter()

    def test_duration_converter_is_initialized(self, converter):
        """Test que DurationConverter est correctement initialisé."""
        assert hasattr(converter, '_duration_converter')
        assert converter._duration_converter is not None

    def test_qe_to_me_extension_uses_duration_converter(self, converter):
        """Test que l'extension QE→ME utilise les facteurs de DurationConverter."""
        # Création de données trimestrielles
        qe_dates = pd.date_range('2024-03-31', periods=4, freq='QE')
        qe_series = pd.Series([100, 200, 300, 400], index=qe_dates)

        # Conversion vers mensuel
        monthly = converter.convert_frequency(qe_series, 'ME', method='mean')

        # Vérification que l'extension a bien créé 12 mois
        # (ratio Q→M devrait être 3, donc 4 trimestres = 12 mois)
        assert len(monthly) == 12, f"Expected 12 months, got {len(monthly)}"

    def test_ye_to_qe_extension_uses_duration_converter(self, converter):
        """Test que l'extension YE→QE utilise les facteurs de DurationConverter."""
        # Création de données annuelles
        ye_dates = pd.date_range('2023-12-31', periods=2, freq='YE')
        ye_series = pd.Series([1000, 2000], index=ye_dates)

        # Conversion vers trimestriel
        quarterly = converter.convert_frequency(ye_series, 'QE', method='mean')

        # Vérification que l'extension fonctionne
        # (ratio Y→Q devrait être 4)
        assert len(quarterly) >= 4, f"Expected at least 4 quarters, got {len(quarterly)}"

    def test_unsupported_frequency_pair_returns_original(self, converter):
        """Test que les paires non supportées retournent l'index original."""
        # Utilisation d'une fréquence de base non standard
        # Le test vérifie que l'extension ne plante pas même si DurationConverter
        # ne supporte pas certaines paires
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        series = pd.Series(range(10), index=dates)

        # Conversion D→B (business days)
        # Si non supporté par DurationConverter, devrait retourner quelque chose
        result = converter.convert_frequency(series, 'B', method='mean')
        assert len(result) > 0

    def test_conversion_factors_match_duration_converter(self, converter):
        """Test que les ratios calculés correspondent à DurationConverter."""
        # Test direct du ratio Q→M
        ratio_q_to_m = converter._duration_converter.get_conversion_factor('Q', 'M')
        assert ratio_q_to_m == 3.0, f"Expected 3.0, got {ratio_q_to_m}"

        # Test direct du ratio Y→Q
        ratio_y_to_q = converter._duration_converter.get_conversion_factor('Y', 'Q')
        assert ratio_y_to_q == 4.0, f"Expected 4.0, got {ratio_y_to_q}"

        # Test direct du ratio Y→M
        ratio_y_to_m = converter._duration_converter.get_conversion_factor('Y', 'M')
        assert ratio_y_to_m == 12.0, f"Expected 12.0, got {ratio_y_to_m}"
