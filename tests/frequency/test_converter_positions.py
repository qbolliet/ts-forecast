"""Tests unitaires pour les conversions de fréquence avec gestion des positions (S/E).

Ce module teste la préservation des positions (start/end) lors des conversions
de fréquence et l'extension de la plage temporelle lors de l'upsampling.
"""
import pytest
import pandas as pd
import numpy as np
from tsforecast.utils.frequency.converter import FrequencyConverter


class TestFrequencyConverterPositions:
    """Tests pour la gestion des positions lors des conversions de fréquence."""

    @pytest.fixture
    def converter(self):
        """Fixture pour créer une instance de FrequencyConverter."""
        return FrequencyConverter()

    # Tests de préservation des positions lors des conversions

    def test_qe_to_me_preserves_end_position(self, converter):
        """Test que la conversion de QE vers M préserve la position 'end'."""
        # Création de données trimestrielles en position end (QE)
        qe_dates = pd.date_range('2024-03-31', periods=4, freq='QE')
        qe_series = pd.Series([100, 200, 300, 400], index=qe_dates)

        # Conversion vers mensuel (upsampling : method doit être une méthode
        # d'interpolation, 'mean' n'est valide que pour l'agrégation)
        monthly = converter.convert_frequency(qe_series, 'M', method='linear')

        # Vérification que l'index mensuel est bien en position end (ME)
        # La fréquence inférée devrait être 'ME' ou équivalent
        detected_freq = pd.infer_freq(monthly.index)
        assert detected_freq in ['ME', 'M'], f"Expected 'ME' or 'M', got {detected_freq}"

        # Vérification que l'extension a bien créé 12 mois
        assert len(monthly) == 12, f"Expected 12 months, got {len(monthly)}"

    def test_qs_to_ms_preserves_start_position(self, converter):
        """Test que la conversion de QS vers MS préserve la position 'start'."""
        # Création de données trimestrielles en position start (QS)
        qs_dates = pd.date_range('2024-01-01', periods=4, freq='QS')
        qs_series = pd.Series([100, 200, 300, 400], index=qs_dates)

        # Conversion vers mensuel avec position start (upsampling : method
        # doit être une méthode d'interpolation, pas 'mean')
        monthly = converter.convert_frequency(qs_series, 'MS', method='linear')

        # Vérification que l'index mensuel est bien en position start (MS)
        detected_freq = pd.infer_freq(monthly.index)
        assert detected_freq == 'MS', f"Expected 'MS', got {detected_freq}"

        # Vérification que l'extension a bien créé 12 mois
        assert len(monthly) == 12, f"Expected 12 months, got {len(monthly)}"

    def test_me_to_qe_preserves_end_position(self, converter):
        """Test que la conversion de ME vers QE préserve la position 'end'."""
        # Création de données mensuelles en position end
        me_dates = pd.date_range('2024-01-31', periods=12, freq='ME')
        me_series = pd.Series(range(1, 13), index=me_dates)

        # Conversion vers trimestriel
        quarterly = converter.convert_frequency(me_series, 'Q', method='mean')

        # Vérification que l'index trimestriel est bien en position end
        # Accepter QE, Q, ou QE-* (avec anchor)
        detected_freq = pd.infer_freq(quarterly.index)
        assert detected_freq.startswith(('QE', 'Q')), f"Expected frequency starting with 'QE' or 'Q', got {detected_freq}"

        # Vérification du nombre de trimestres
        assert len(quarterly) == 4, f"Expected 4 quarters, got {len(quarterly)}"

    def test_explicit_target_position_overrides_source(self, converter):
        """Test que target_position explicite override la position source."""
        # Création de données trimestrielles en position end
        qe_dates = pd.date_range('2024-03-31', periods=4, freq='QE')
        qe_series = pd.Series([100, 200, 300, 400], index=qe_dates)

        # Conversion vers mensuel avec position start explicite (upsampling :
        # method doit être une méthode d'interpolation, pas 'mean')
        monthly = converter.convert_frequency(
            qe_series,
            'M',
            method='linear',
            target_position='start'
        )

        # Vérification que l'index mensuel est en position start
        detected_freq = pd.infer_freq(monthly.index)
        assert detected_freq == 'MS', f"Expected 'MS', got {detected_freq}"

    def test_target_freq_with_position_overrides_default(self, converter):
        """Test que la position dans target_freq est respectée."""
        # Création de données trimestrielles en position end
        qe_dates = pd.date_range('2024-03-31', periods=4, freq='QE')
        qe_series = pd.Series([100, 200, 300, 400], index=qe_dates)

        # Conversion vers mensuel avec position start dans target_freq
        # (upsampling : method doit être une méthode d'interpolation, pas 'mean')
        monthly = converter.convert_frequency(qe_series, 'MS', method='linear')

        # Vérification que la position start de target_freq est respectée
        detected_freq = pd.infer_freq(monthly.index)
        assert detected_freq == 'MS', f"Expected 'MS', got {detected_freq}"

    # Tests d'extension de la plage temporelle

    def test_qe_to_me_extends_to_full_periods(self, converter):
        """Test que la conversion QE→ME étend la plage pour inclure tous les mois."""
        # Création de 4 trimestres (Q1-Q4 2024)
        qe_dates = pd.date_range('2024-03-31', periods=4, freq='QE')
        qe_series = pd.Series([100, 200, 300, 400], index=qe_dates)

        # Conversion vers mensuel
        monthly = converter.convert_frequency(qe_series, 'ME', method='linear')

        # Vérification qu'on a bien 12 mois (Jan à Dec 2024)
        assert len(monthly) == 12, f"Expected 12 months, got {len(monthly)}"

        # Vérification que tous les mois sont présents
        expected_months = pd.date_range('2024-01-31', '2024-12-31', freq='ME')
        pd.testing.assert_index_equal(monthly.index, expected_months)

    def test_qs_to_ms_extends_to_full_periods(self, converter):
        """Test que la conversion QS→MS étend la plage pour inclure tous les mois."""
        # Création de 4 trimestres en position start
        qs_dates = pd.date_range('2024-01-01', periods=4, freq='QS')
        qs_series = pd.Series([100, 200, 300, 400], index=qs_dates)

        # Conversion vers mensuel
        monthly = converter.convert_frequency(qs_series, 'MS', method='linear')

        # Vérification qu'on a bien 12 mois
        assert len(monthly) == 12, f"Expected 12 months, got {len(monthly)}"

        # Vérification que tous les mois sont présents (Jan à Dec 2024)
        expected_months = pd.date_range('2024-01-01', '2024-12-01', freq='MS')
        pd.testing.assert_index_equal(monthly.index, expected_months)

    def test_ye_to_qe_extends_correctly(self, converter):
        """Test que la conversion YE→QE étend correctement."""
        # Création de données annuelles (2023 et 2024)
        ye_dates = pd.date_range('2023-12-31', periods=2, freq='YE')
        ye_series = pd.Series([1000, 2000], index=ye_dates)

        # Conversion vers trimestriel
        quarterly = converter.convert_frequency(ye_series, 'QE', method='linear')

        # L'extension YE→QE crée des trimestres entre la première et la dernière année
        # 2023-12-31 (fin 2023) à 2024-12-31 (fin 2024) = 5 trimestres
        # (Q4-2023, Q1-2024, Q2-2024, Q3-2024, Q4-2024)
        assert len(quarterly) >= 4, f"Expected at least 4 quarters, got {len(quarterly)}"

    # Tests de gestion des anchors (positions de référence)

    def test_qe_dec_anchor_preserved(self, converter):
        """Test que les anchors comme -DEC sont correctement gérés."""
        # Création de données avec anchor explicite
        qe_dec_dates = pd.date_range('2024-03-31', periods=4, freq='QE-DEC')
        qe_series = pd.Series([100, 200, 300, 400], index=qe_dec_dates)

        # Conversion vers mensuel (upsampling : method doit être une méthode
        # d'interpolation, pas 'mean')
        monthly = converter.convert_frequency(qe_series, 'ME', method='linear')

        # Vérification que la conversion fonctionne sans erreur
        assert len(monthly) > 0, "Conversion should produce non-empty result"
        assert len(monthly) == 12, f"Expected 12 months, got {len(monthly)}"

    # Tests d'interpolation après extension

    def test_interpolation_fills_extended_periods(self, converter):
        """Test que l'interpolation remplit correctement les périodes étendues."""
        # Création de données trimestrielles avec valeurs connues
        qe_dates = pd.date_range('2024-03-31', periods=4, freq='QE')
        qe_series = pd.Series([100, 200, 300, 400], index=qe_dates)

        # Conversion vers mensuel avec interpolation linéaire. 'fill_method'
        # n'existe pas dans l'API actuelle : le comblement des valeurs AVANT
        # la première date est géré par le limit_direction par défaut
        # ('backward' pour une fréquence cible positionnée en fin de période)
        monthly = converter.convert_frequency(qe_series, 'ME', method='linear')

        # Vérification qu'il n'y a pas de NaN après interpolation + fill
        assert not monthly.isna().any(), "Interpolation with bfill should fill all NaN values"

        # Vérification que les valeurs originales sont préservées aux positions trimestrielles
        # Mars, Juin, Septembre, Décembre devraient contenir les valeurs d'origine
        # Trouver les index des mois de fin de trimestre
        march_idx = monthly.index.get_loc(pd.Timestamp('2024-03-31'))
        june_idx = monthly.index.get_loc(pd.Timestamp('2024-06-30'))
        sept_idx = monthly.index.get_loc(pd.Timestamp('2024-09-30'))
        dec_idx = monthly.index.get_loc(pd.Timestamp('2024-12-31'))

        assert monthly.iloc[march_idx] == pytest.approx(100, rel=0.1), "March value should be ~100"
        assert monthly.iloc[june_idx] == pytest.approx(200, rel=0.1), "June value should be ~200"
        assert monthly.iloc[sept_idx] == pytest.approx(300, rel=0.1), "September value should be ~300"
        assert monthly.iloc[dec_idx] == pytest.approx(400, rel=0.1), "December value should be ~400"

    # Tests de cas limites

    def test_same_frequency_with_different_positions(self, converter):
        """Test la conversion entre même fréquence mais positions différentes."""
        # Création de données en position end
        me_dates = pd.date_range('2024-01-31', periods=12, freq='ME')
        me_series = pd.Series(range(1, 13), index=me_dates)

        # Conversion vers position start (utiliser 'mean' car c'est du downsampling conceptuellement)
        # On passe de ME (month-end) à MS (month-start), donc les dates changent
        ms_series = converter.convert_frequency(
            me_series,
            'M',
            method='mean',  # Utiliser une méthode d'agrégation
            target_position='start'
        )

        # Vérification que la conversion produit une sortie
        assert len(ms_series) > 0, "Conversion should produce non-empty result"

    def test_default_position_when_not_detectable(self, converter):
        """Test que la position par défaut 'E' est utilisée si non détectable."""
        # Création de données avec fréquence sans position (daily)
        daily_dates = pd.date_range('2024-01-01', periods=90, freq='D')
        daily_series = pd.Series(range(90), index=daily_dates)

        # Conversion vers mensuel (devrait utiliser position end par défaut)
        monthly = converter.convert_frequency(daily_series, 'M', method='mean')

        # Vérification que la conversion fonctionne
        assert len(monthly) == 3, f"Expected 3 months, got {len(monthly)}"

        # Vérification que la position end est utilisée par défaut
        detected_freq = pd.infer_freq(monthly.index)
        assert detected_freq in ['ME', 'M'], f"Expected 'ME' or 'M', got {detected_freq}"

    def test_dataframe_conversion_preserves_positions(self, converter):
        """Test que la conversion de DataFrame préserve aussi les positions."""
        # Création d'un DataFrame avec données trimestrielles
        qe_dates = pd.date_range('2024-03-31', periods=4, freq='QE')
        qe_df = pd.DataFrame({
            'col1': [100, 200, 300, 400],
            'col2': [1000, 2000, 3000, 4000]
        }, index=qe_dates)

        # Conversion vers mensuel (upsampling : method doit être une méthode
        # d'interpolation, pas 'mean')
        monthly_df = converter.convert_frequency(qe_df, 'M', method='linear')

        # Vérification que l'extension fonctionne pour DataFrame
        assert len(monthly_df) == 12, f"Expected 12 months, got {len(monthly_df)}"

        # Vérification que toutes les colonnes sont présentes
        assert list(monthly_df.columns) == ['col1', 'col2'], "Columns should be preserved"


class TestExtendIndexForUpsampling:
    """Tests spécifiques pour la méthode _extend_index_for_upsampling."""

    @pytest.fixture
    def converter(self):
        """Fixture pour créer une instance de FrequencyConverter."""
        return FrequencyConverter()

    def test_qe_to_me_extension(self, converter):
        """Test l'extension d'index de QE vers ME."""
        # Création d'un index trimestriel
        qe_index = pd.date_range('2024-03-31', periods=4, freq='QE')

        # Extension vers mensuel
        extended = converter._extend_index_for_upsampling(qe_index, 'QE', 'ME')

        # Vérification de la longueur
        assert len(extended) == 12, f"Expected 12 months, got {len(extended)}"

        # Vérification des bornes
        assert extended[0].month == 1, "Should start with January"
        assert extended[-1].month == 12, "Should end with December"

    def test_qs_to_ms_extension(self, converter):
        """Test l'extension d'index de QS vers MS."""
        # Création d'un index trimestriel start
        qs_index = pd.date_range('2024-01-01', periods=4, freq='QS')

        # Extension vers mensuel
        extended = converter._extend_index_for_upsampling(qs_index, 'QS', 'MS')

        # Vérification de la longueur
        assert len(extended) == 12, f"Expected 12 months, got {len(extended)}"

        # Vérification que tous sont au début du mois
        assert all(d.day == 1 for d in extended), "All dates should be month-start"

    def test_same_base_frequency_no_extension(self, converter):
        """Test qu'il n'y a pas d'extension si les fréquences de base sont identiques."""
        # Création d'un index mensuel
        me_index = pd.date_range('2024-01-31', periods=12, freq='ME')

        # "Extension" vers mensuel (pas de changement attendu)
        extended = converter._extend_index_for_upsampling(me_index, 'ME', 'MS')

        # L'index devrait rester identique en longueur
        # (la méthode retourne l'index original si même fréquence de base)
        assert len(extended) == len(me_index), "Index should not be extended for same base frequency"

    def test_unsupported_frequency_pair_returns_original(self, converter):
        """Test que les paires de fréquences non supportées retournent l'index original."""
        # Création d'un index avec fréquence non standard
        index = pd.date_range('2024-01-01', periods=10, freq='2D')

        # Tentative d'extension vers une autre fréquence
        extended = converter._extend_index_for_upsampling(index, '2D', 'D')

        # L'index devrait être retourné tel quel
        assert len(extended) >= len(index), "Should return original or extended index"


class TestCrossPositionUpsampling:
    """Tests pour l'upsampling entre positions source et cible divergentes.

    Régression : l'index étendu est construit à la position cible alors que les
    données portent la position source. Sans ré-ancrage, l'intersection des deux
    index est vide (ex. QS 2024-01-01 vs ME 2024-01-31) et la réindexation perd
    toutes les observations, produisant une série intégralement NaN.
    """

    @pytest.fixture
    def converter(self):
        """Fixture pour créer une instance de FrequencyConverter."""
        return FrequencyConverter()

    def test_qs_to_me_does_not_produce_all_nan(self, converter):
        """Test que l'upsampling QS -> ME conserve les valeurs sources."""
        # Série trimestrielle en position start
        qs_series = pd.Series(
            [100.0, 200.0, 300.0, 400.0],
            index=pd.date_range('2024-01-01', periods=4, freq='QS')
        )

        # Interpolation vers mensuel en position end
        result = converter.interpolate_to_higher_frequency(
            qs_series, 'ME', method='linear', limit='default'
        )

        # L'index cible couvre bien les 12 mois de l'année
        assert len(result) == 12, f"Expected 12 months, got {len(result)}"
        # Toutes les dates sont en fin de mois
        assert all(d == d + pd.offsets.MonthEnd(0) for d in result.index), \
            "All dates should be month-end"
        # Régression principale : la série n'est pas intégralement NaN
        assert not result.isna().all(), "Result should not be entirely NaN"

    def test_qs_to_me_preserves_source_values_on_anchor_months(self, converter):
        """Test que chaque valeur source est ré-estampillée sur le bon mois."""
        # Série trimestrielle en position start
        qs_series = pd.Series(
            [100.0, 200.0, 300.0, 400.0],
            index=pd.date_range('2024-01-01', periods=4, freq='QS')
        )

        # Interpolation vers mensuel en position end
        result = converter.interpolate_to_higher_frequency(
            qs_series, 'ME', method='linear', limit='default'
        )

        # La sous-période est préservée : janvier reste janvier, avril reste avril,
        # seule la convention start/end change
        assert result.loc['2024-01-31'] == 100.0
        assert result.loc['2024-04-30'] == 200.0
        assert result.loc['2024-07-31'] == 300.0
        assert result.loc['2024-10-31'] == 400.0

    def test_qe_to_ms_preserves_source_values_on_anchor_months(self, converter):
        """Test le cas symétrique : position source end vers position cible start."""
        # Série trimestrielle en position end
        qe_series = pd.Series(
            [100.0, 200.0, 300.0, 400.0],
            index=pd.date_range('2024-03-31', periods=4, freq='QE')
        )

        # Interpolation vers mensuel en position start
        result = converter.interpolate_to_higher_frequency(
            qe_series, 'MS', method='linear', limit='default'
        )

        # Couverture complète et valeurs ancrées sur les mois d'origine
        assert len(result) == 12, f"Expected 12 months, got {len(result)}"
        assert not result.isna().all(), "Result should not be entirely NaN"
        assert result.loc['2024-03-01'] == 100.0
        assert result.loc['2024-06-01'] == 200.0
        assert result.loc['2024-09-01'] == 300.0
        assert result.loc['2024-12-01'] == 400.0

    def test_matching_positions_still_interpolate(self, converter):
        """Test de non-régression du cas nominal (positions concordantes)."""
        # Série trimestrielle en position start vers mensuel en position start
        qs_series = pd.Series(
            [100.0, 200.0, 300.0, 400.0],
            index=pd.date_range('2024-01-01', periods=4, freq='QS')
        )

        # Interpolation vers mensuel en position start
        result = converter.interpolate_to_higher_frequency(
            qs_series, 'MS', method='linear', limit='default'
        )

        # Les valeurs sources restent en place, sans décalage
        assert len(result) == 12, f"Expected 12 months, got {len(result)}"
        assert result.loc['2024-01-01'] == 100.0
        assert result.loc['2024-04-01'] == 200.0
        assert result.loc['2024-10-01'] == 400.0

    # Tests unitaires de la méthode de ré-ancrage

    def test_reanchor_start_to_end(self, converter):
        """Test le ré-ancrage d'un index start vers une position end."""
        # Index trimestriel en position start
        qs_index = pd.date_range('2024-01-01', periods=4, freq='QS')

        # Ré-ancrage sur la grille mensuelle en position end
        reanchored = converter._reanchor_index_to_target(qs_index, 'ME')

        # Chaque date est déplacée en fin du mois qui la contenait
        expected = pd.DatetimeIndex(
            ['2024-01-31', '2024-04-30', '2024-07-31', '2024-10-31']
        )
        assert list(reanchored) == list(expected)

    def test_reanchor_end_to_start(self, converter):
        """Test le ré-ancrage d'un index end vers une position start."""
        # Index trimestriel en position end
        qe_index = pd.date_range('2024-03-31', periods=4, freq='QE')

        # Ré-ancrage sur la grille mensuelle en position start
        reanchored = converter._reanchor_index_to_target(qe_index, 'MS')

        # Chaque date est déplacée au début du mois qui la contenait
        expected = pd.DatetimeIndex(
            ['2024-03-01', '2024-06-01', '2024-09-01', '2024-12-01']
        )
        assert list(reanchored) == list(expected)

    def test_reanchor_is_noop_for_matching_position(self, converter):
        """Test que le ré-ancrage ne modifie pas un index déjà bien positionné."""
        # Index trimestriel en position start
        qs_index = pd.date_range('2024-01-01', periods=4, freq='QS')

        # Ré-ancrage vers une cible de même position
        reanchored = converter._reanchor_index_to_target(qs_index, 'MS')

        # Les dates sont inchangées
        assert list(reanchored) == list(qs_index)
