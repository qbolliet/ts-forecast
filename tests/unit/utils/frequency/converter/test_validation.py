"""Tests unitaires pour la validation des paramètres dans FrequencyConverter.

Ce module teste la validation des fréquences et positions dans les méthodes
de conversion, notamment la validation de freq_pos dans _validate_conversion_params.
"""
import pytest
import pandas as pd
from tsforecast.utils.frequency.converter import FrequencyConverter
from tsforecast.utils.duration.utils import get_duration_conversion_factor


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


class TestConvertFrequencyParameterErrors:
    """``convert_frequency`` rejects malformed targets and data before converting anything."""

    @pytest.fixture
    def converter(self):
        """A fresh ``FrequencyConverter``."""
        return FrequencyConverter()

    @pytest.fixture
    def monthly_series(self):
        """Month ends of 2024 H1."""
        return pd.Series(range(6), index=pd.date_range('2024-01-31', periods=6, freq='ME'), dtype=float)

    @pytest.fixture
    def monthly_frame(self, monthly_series):
        """Two monthly columns ``a`` and ``b``."""
        return pd.DataFrame({'a': monthly_series, 'b': monthly_series * 2})

    @pytest.mark.parametrize(
        'target, message',
        [
            pytest.param('', 'Target frequency cannot be empty', id='empty-string'),
            pytest.param({}, 'Target frequency cannot be empty', id='empty-dict'),
            pytest.param(3, 'target_freq must be a string or dictionary', id='integer'),
            pytest.param(['QE'], 'target_freq must be a string or dictionary', id='list'),
        ],
    )
    def test_malformed_target(self, converter, monthly_frame, target, message):
        """Empty targets and targets that are neither str nor dict."""
        with pytest.raises(ValueError, match=message):
            converter.convert_frequency(monthly_frame, target, method='sum')

    def test_dict_target_on_simple_series(self, converter, monthly_series):
        """A dictionary needs columns or entities to map."""
        with pytest.raises(ValueError, match='Dictionary target_freq is only valid for DataFrame or panel inputs'):
            converter.convert_frequency(monthly_series, {'a': 'QE'}, method='sum')

    def test_tuple_key_without_panel(self, converter, monthly_frame):
        """Entity keys make no sense without a panel index."""
        with pytest.raises(ValueError, match='Tuple keys in target_freq are only valid for panel inputs'):
            converter.convert_frequency(monthly_frame, {('FR', 'a'): 'QE'}, method='sum')

    def test_unknown_column(self, converter, monthly_frame):
        """Every dictionary key must be a column of the DataFrame."""
        with pytest.raises(ValueError, match="Columns in target_freq not found in data: {'zz'}"):
            converter.convert_frequency(monthly_frame, {'a': 'QE', 'zz': 'QE'}, method='sum')

    def test_non_pandas_data(self, converter):
        """Only pandas objects are converted."""
        with pytest.raises(ValueError, match='Input data must be a pandas Series or DataFrame'):
            converter.convert_frequency([1.0, 2.0, 3.0], 'QE', method='sum')


class TestDurationConverterIntegration:
    """Tests pour l'intégration de DurationConverter dans FrequencyConverter."""

    @pytest.fixture
    def converter(self):
        """Fixture pour créer une instance de FrequencyConverter."""
        return FrequencyConverter()

    def test_qe_to_me_extension_uses_duration_converter(self, converter):
        """Test que l'extension QE→ME utilise les facteurs de DurationConverter."""
        # Création de données trimestrielles
        qe_dates = pd.date_range('2024-03-31', periods=4, freq='QE')
        qe_series = pd.Series([100, 200, 300, 400], index=qe_dates)

        # Conversion vers mensuel (upsampling : method doit être une méthode
        # d'interpolation, pas 'mean')
        monthly = converter.convert_frequency(qe_series, 'ME', method='linear')

        # Vérification que l'extension a bien créé 12 mois
        # (ratio Q→M devrait être 3, donc 4 trimestres = 12 mois)
        assert len(monthly) == 12, f"Expected 12 months, got {len(monthly)}"

    def test_ye_to_qe_extension_uses_duration_converter(self, converter):
        """Test que l'extension YE→QE utilise les facteurs de DurationConverter."""
        # Création de données annuelles
        ye_dates = pd.date_range('2023-12-31', periods=2, freq='YE')
        ye_series = pd.Series([1000, 2000], index=ye_dates)

        # Conversion vers trimestriel (upsampling : method doit être une
        # méthode d'interpolation, pas 'mean')
        quarterly = converter.convert_frequency(ye_series, 'QE', method='linear')

        # Valeur d'or : rapport Y → Q = 4 par année, sur les deux années entières
        # 2023 et 2024 → 8 trimestres exactement
        assert len(quarterly) == 8

    def test_daily_to_business_daily_is_an_aggregation(self, converter):
        """Test que D → B agrège les week-ends dans le vendredi qui les précède.

        Réécriture (catégorie c) de l'ancien ``test_unsupported_frequency_pair_returns_original`` :
        D → B n'est pas un sur-échantillonnage (un jour ouvré est moins fin qu'un jour
        calendaire), l'extension d'index n'y intervient pas ; l'ancien test se bornait
        à ``len(result) > 0``.
        """
        # Du lundi 1er au mercredi 10 janvier 2024, valeurs 0 … 9
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        series = pd.Series(range(10), index=dates, dtype=float)

        result = converter.convert_frequency(series, 'B', method='mean')

        # Valeurs d'or : un compartiment par jour ouvré ; celui du vendredi 5 couvre
        # samedi 6 et dimanche 7 → moyenne de 4, 5, 6 = 5
        expected = pd.Series(
            [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 8.0, 9.0],
            index=pd.bdate_range('2024-01-01', '2024-01-10'),
        )
        pd.testing.assert_series_equal(result, expected, check_freq=False)

    def test_conversion_factors_match_duration_converter(self, converter):
        """Test que les ratios calculés correspondent à DurationConverter.

        `FrequencyConverter` ne conserve plus de `DurationConverter` en
        attribut interne : les facteurs de conversion sont obtenus via la
        fonction utilitaire `get_duration_conversion_factor`.
        """
        # Test direct du ratio Q→M
        ratio_q_to_m = get_duration_conversion_factor('Q', 'M')
        assert ratio_q_to_m == 3.0, f"Expected 3.0, got {ratio_q_to_m}"

        # Test direct du ratio Y→Q
        ratio_y_to_q = get_duration_conversion_factor('Y', 'Q')
        assert ratio_y_to_q == 4.0, f"Expected 4.0, got {ratio_y_to_q}"

        # Test direct du ratio Y→M
        ratio_y_to_m = get_duration_conversion_factor('Y', 'M')
        assert ratio_y_to_m == 12.0, f"Expected 12.0, got {ratio_y_to_m}"


class TestModernizedResampleAliases:
    """§4.6 : les alias bruts dépréciés par pandas ('Y', 'A', 'Q', 'M') sont
    modernisés ('YE', 'YE', 'QE', 'ME') avant tout resample/asfreq, sans
    changer l'ancrage ni le résultat."""

    @pytest.fixture
    def converter(self):
        return FrequencyConverter()

    @pytest.mark.parametrize("bare,modern", [('Y', 'YE'), ('A', 'YE'), ('Q', 'QE'), ('M', 'ME')])
    def test_aggregate_bare_alias_emits_no_future_warning(self, converter, bare, modern):
        """L'alias brut ne déclenche plus de FutureWarning pandas et produit
        le même résultat que son équivalent moderne."""
        import warnings

        dates = pd.date_range('2023-01-01', periods=400, freq='D')
        series = pd.Series(range(400), index=dates)

        with warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            result_bare = converter.aggregate_to_lower_frequency(series, bare, method='sum')

        result_modern = converter.aggregate_to_lower_frequency(series, modern, method='sum')
        pd.testing.assert_series_equal(result_bare, result_modern)

    def test_positioned_alias_left_untouched(self, converter):
        """Un alias déjà positionné ('QS') n'est pas altéré."""
        dates = pd.date_range('2023-01-01', periods=400, freq='D')
        series = pd.Series(range(400), index=dates)

        result = converter.aggregate_to_lower_frequency(series, 'QS', method='sum')
        assert result.index.freqstr.startswith('QS')
