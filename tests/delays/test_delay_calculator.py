"""Tests pour le module calculator.

Ce module teste les fonctions de calcul des délais applicables:
- calculate_applicable_delay(): Fonction principale de calcul du délai applicable
- _validate_columns(): Validation des colonnes requises
- _convert_to_target_frequency_and_reference(): Conversion vers fréquence et référence cibles
- _calculate_converted_delay(): Calcul du délai converti pour une ligne
- _convert_delay_unit(): Conversion d'unité des délais
- _aggregate_delays(): Agrégation des délais
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from tsforecast.delays.calculator import (
    calculate_applicable_delay,
    _validate_columns,
    _convert_to_target_frequency_and_reference,
    _calculate_converted_delay,
    _convert_delay_unit,
    _aggregate_delays
)
from tsforecast.delays.data_manager import compare_and_detect_delays


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def monthly_publication_delays():
    """Crée des délais de publication mensuels typiques."""
    # Création de l'index (pays, indicateur)
    index_tuples = [
        ('France', 'PIB'),
        ('France', 'inflation'),
        ('Germany', 'PIB'),
        ('Germany', 'inflation'),
    ]
    index = pd.MultiIndex.from_tuples(index_tuples, names=['country', 'indicator'])

    # Données des délais
    data = {
        'observation_date': [
            datetime(2023, 12, 1), datetime(2023, 12, 1),
            datetime(2023, 12, 1), datetime(2023, 12, 1)
        ],
        'download_date': [
            datetime(2024, 1, 15), datetime(2024, 1, 10),
            datetime(2024, 1, 20), datetime(2024, 1, 12)
        ],
        'frequency': ['M', 'M', 'M', 'M'],
        'period_start': [
            datetime(2023, 12, 1), datetime(2023, 12, 1),
            datetime(2023, 12, 1), datetime(2023, 12, 1)
        ],
        'period_end': [
            datetime(2024, 1, 1), datetime(2024, 1, 1),
            datetime(2024, 1, 1), datetime(2024, 1, 1)
        ],
        'reference_point': ['end', 'end', 'end', 'end'],
        'release_delay': [14, 9, 19, 11],  # Jours après la fin de la période
        'unit': ['day', 'day', 'day', 'day']
    }

    return pd.DataFrame(data, index=index)


@pytest.fixture
def quarterly_publication_delays():
    """Crée des délais de publication trimestriels."""
    index_tuples = [
        ('France', 'PIB'),
        ('Germany', 'PIB'),
    ]
    index = pd.MultiIndex.from_tuples(index_tuples, names=['country', 'indicator'])

    data = {
        'observation_date': [datetime(2023, 10, 15), datetime(2023, 11, 20)],
        'download_date': [datetime(2024, 1, 15), datetime(2024, 1, 20)],
        'frequency': ['Q', 'Q'],
        'period_start': [datetime(2023, 10, 1), datetime(2023, 10, 1)],
        'period_end': [datetime(2024, 1, 1), datetime(2024, 1, 1)],
        'reference_point': ['end', 'end'],
        'release_delay': [14, 19],
        'unit': ['day', 'day']
    }

    return pd.DataFrame(data, index=index)


@pytest.fixture
def multi_observation_delays():
    """Crée plusieurs observations de délais pour le même indicateur."""
    index_tuples = [
        ('France', 'PIB'),
        ('France', 'PIB'),
        ('France', 'PIB'),
        ('France', 'PIB'),
        ('Germany', 'PIB'),
        ('Germany', 'PIB'),
    ]
    index = pd.MultiIndex.from_tuples(index_tuples, names=['country', 'indicator'])

    data = {
        'observation_date': [
            datetime(2023, 9, 1), datetime(2023, 10, 1),
            datetime(2023, 11, 1), datetime(2023, 12, 1),
            datetime(2023, 11, 1), datetime(2023, 12, 1)
        ],
        'download_date': [
            datetime(2023, 10, 15), datetime(2023, 11, 14),
            datetime(2023, 12, 16), datetime(2024, 1, 15),
            datetime(2023, 12, 20), datetime(2024, 1, 18)
        ],
        'frequency': ['M', 'M', 'M', 'M', 'M', 'M'],
        'period_start': [
            datetime(2023, 9, 1), datetime(2023, 10, 1),
            datetime(2023, 11, 1), datetime(2023, 12, 1),
            datetime(2023, 11, 1), datetime(2023, 12, 1)
        ],
        'period_end': [
            datetime(2023, 10, 1), datetime(2023, 11, 1),
            datetime(2023, 12, 1), datetime(2024, 1, 1),
            datetime(2023, 12, 1), datetime(2024, 1, 1)
        ],
        'reference_point': ['end', 'end', 'end', 'end', 'end', 'end'],
        'release_delay': [14, 13, 15, 14, 19, 17],
        'unit': ['day', 'day', 'day', 'day', 'day', 'day']
    }

    return pd.DataFrame(data, index=index)


@pytest.fixture
def simple_time_series_delays():
    """Crée des délais pour une série temporelle simple (sans panel)."""
    index = pd.Index(['PIB', 'inflation', 'unemployment'], name='indicator')

    data = {
        'observation_date': [
            datetime(2023, 12, 1), datetime(2023, 12, 1), datetime(2023, 12, 1)
        ],
        'download_date': [
            datetime(2024, 1, 15), datetime(2024, 1, 10), datetime(2024, 1, 5)
        ],
        'frequency': ['M', 'M', 'M'],
        'period_start': [
            datetime(2023, 12, 1), datetime(2023, 12, 1), datetime(2023, 12, 1)
        ],
        'period_end': [
            datetime(2024, 1, 1), datetime(2024, 1, 1), datetime(2024, 1, 1)
        ],
        'reference_point': ['end', 'end', 'end'],
        'release_delay': [14, 9, 4],
        'unit': ['day', 'day', 'day']
    }

    return pd.DataFrame(data, index=index)


@pytest.fixture
def delays_with_seconds_unit():
    """Crée des délais en secondes."""
    index = pd.Index(['PIB'], name='indicator')

    data = {
        'observation_date': [datetime(2023, 12, 1)],
        'download_date': [datetime(2024, 1, 15, 12, 30, 45)],
        'frequency': ['M'],
        'period_start': [datetime(2023, 12, 1)],
        'period_end': [datetime(2024, 1, 1)],
        'reference_point': ['end'],
        'release_delay': [14 * 86400 + 12 * 3600 + 30 * 60 + 45],  # 14 jours + 12h30m45s
        'unit': ['second']
    }

    return pd.DataFrame(data, index=index)


# ============================================================================
# Tests pour _validate_columns
# ============================================================================

class TestValidateColumns:
    """Tests pour la fonction _validate_columns."""

    def test_valid_columns(self, monthly_publication_delays):
        """Test avec toutes les colonnes requises présentes."""
        # Ne devrait pas lever d'exception
        _validate_columns(monthly_publication_delays)

    def test_missing_columns(self):
        """Test avec colonnes manquantes."""
        df = pd.DataFrame({
            'observation_date': [datetime(2023, 12, 1)],
            'download_date': [datetime(2024, 1, 15)]
            # Colonnes manquantes: frequency, period_start, etc.
        })

        with pytest.raises(ValueError, match="Missing required columns"):
            _validate_columns(df)

    def test_partial_missing_columns(self, monthly_publication_delays):
        """Test avec certaines colonnes manquantes."""
        df = monthly_publication_delays.drop(columns=['frequency', 'unit'])

        with pytest.raises(ValueError, match="Missing required columns"):
            _validate_columns(df)

    def test_empty_dataframe(self):
        """Test avec DataFrame vide mais avec les bonnes colonnes."""
        columns = [
            'observation_date', 'download_date', 'frequency',
            'period_start', 'period_end', 'reference_point',
            'release_delay', 'unit'
        ]
        df = pd.DataFrame(columns=columns)

        # Ne devrait pas lever d'exception
        _validate_columns(df)


# ============================================================================
# Tests pour _calculate_converted_delay
# ============================================================================

class TestCalculateConvertedDelay:
    """Tests pour la fonction _calculate_converted_delay."""

    def test_same_frequency_same_reference(self):
        """Test avec même fréquence et même point de référence."""
        row = pd.Series({
            'observation_date': pd.Timestamp('2023-12-15'),
            'period_start': pd.Timestamp('2023-12-01'),
            'period_end': pd.Timestamp('2024-01-01'),
            'reference_point': 'end',
            'release_delay': 14,
            'unit': 'day',
            'target_frequency_normalized': 'M',
            'current_frequency_normalized': 'M'
        })

        result = _calculate_converted_delay(row, 'end')

        assert 'converted_delay' in result.index
        assert 'target_period_start' in result.index
        assert 'target_period_end' in result.index
        assert result['target_reference_point'] == 'end'
        # Délai devrait rester le même
        assert result['converted_delay'] == 14

    def test_same_frequency_different_reference(self):
        """Test avec même fréquence mais point de référence différent."""
        row = pd.Series({
            'observation_date': pd.Timestamp('2023-12-15'),
            'period_start': pd.Timestamp('2023-12-01'),
            'period_end': pd.Timestamp('2024-01-01'),
            'reference_point': 'end',
            'release_delay': 14,
            'unit': 'day',
            'target_frequency_normalized': 'M',
            'current_frequency_normalized': 'M'
        })

        result = _calculate_converted_delay(row, 'start')

        # Le délai depuis le début devrait être plus grand
        # (14 jours après fin = 14 + 31 jours après début)
        assert result['converted_delay'] > 14
        assert result['target_reference_point'] == 'start'

    def test_higher_frequency_conversion(self):
        """Test de conversion vers une fréquence plus haute (Q → M)."""
        row = pd.Series({
            'observation_date': pd.Timestamp('2023-12-15'),  # En décembre
            'period_start': pd.Timestamp('2023-10-01'),  # Q4 commence en octobre
            'period_end': pd.Timestamp('2024-01-01'),
            'reference_point': 'end',
            'release_delay': 45,
            'unit': 'day',
            'target_frequency_normalized': 'M',
            'current_frequency_normalized': 'Q'
        })

        result = _calculate_converted_delay(row, 'end')

        # La sous-période cible devrait être décembre (car observation_date est en décembre)
        assert result['target_period_start'] == datetime(2023, 12, 1)
        assert result['target_period_end'] == datetime(2024, 1, 1)

    def test_lower_frequency_conversion(self):
        """Test de conversion vers une fréquence plus basse (M → Q)."""
        row = pd.Series({
            'observation_date': pd.Timestamp('2023-12-15'),
            'period_start': pd.Timestamp('2023-12-01'),
            'period_end': pd.Timestamp('2024-01-01'),
            'reference_point': 'end',
            'release_delay': 14,
            'unit': 'day',
            'target_frequency_normalized': 'Q',
            'current_frequency_normalized': 'M'
        })

        result = _calculate_converted_delay(row, 'end')

        # La période cible devrait être Q4 (oct-déc)
        assert result['target_period_start'] == datetime(2023, 10, 1)
        assert result['target_period_end'] == datetime(2024, 1, 1)

    def test_unit_conversion_in_delay(self):
        """Test que les délais en secondes sont correctement traités."""
        row = pd.Series({
            'observation_date': pd.Timestamp('2023-12-15'),
            'period_start': pd.Timestamp('2023-12-01'),
            'period_end': pd.Timestamp('2024-01-01'),
            'reference_point': 'end',
            'release_delay': 86400 * 14,  # 14 jours en secondes
            'unit': 'second',
            'target_frequency_normalized': 'M',
            'current_frequency_normalized': 'M'
        })

        result = _calculate_converted_delay(row, 'end')

        # Le résultat devrait être en secondes également
        assert result['converted_delay'] == pytest.approx(86400 * 14, rel=0.01)


# ============================================================================
# Tests pour _convert_delay_unit
# ============================================================================

class TestConvertDelayUnit:
    """Tests pour la fonction _convert_delay_unit."""

    def test_days_to_seconds(self, monthly_publication_delays):
        """Test de conversion jours → secondes."""
        delays = monthly_publication_delays.copy()
        delays['converted_delay'] = delays['release_delay']

        result = _convert_delay_unit(delays, 's')

        assert (result['unit'] == 's').all()
        # 14 jours = 14 * 86400 secondes
        assert result['converted_delay'].iloc[0] == pytest.approx(14 * 86400, rel=0.01)

    def test_days_to_microseconds(self, monthly_publication_delays):
        """Test de conversion jours → microsecondes."""
        delays = monthly_publication_delays.copy()
        delays['converted_delay'] = delays['release_delay']

        result = _convert_delay_unit(delays, 'us')

        assert (result['unit'] == 'us').all()
        # 14 jours = 14 * 86400 * 1_000_000 microsecondes
        assert result['converted_delay'].iloc[0] == pytest.approx(14 * 86400 * 1_000_000, rel=0.01)

    def test_seconds_to_days(self, delays_with_seconds_unit):
        """Test de conversion secondes → jours."""
        delays = delays_with_seconds_unit.copy()
        delays['converted_delay'] = delays['release_delay']

        result = _convert_delay_unit(delays, 'D')

        assert (result['unit'] == 'D').all()
        # Le résultat devrait être arrondi au plafond
        assert result['converted_delay'].iloc[0] >= 14

    def test_same_unit_no_conversion(self, monthly_publication_delays):
        """Test avec même unité (pas de conversion nécessaire)."""
        delays = monthly_publication_delays.copy()
        delays['converted_delay'] = delays['release_delay']
        original_values = delays['converted_delay'].copy()

        result = _convert_delay_unit(delays, 'day')

        assert (result['unit'] == 'D').all()
        # Les valeurs devraient être identiques
        pd.testing.assert_series_equal(
            result['converted_delay'],
            original_values,
            check_names=False
        )


# ============================================================================
# Tests pour _aggregate_delays
# ============================================================================

class TestAggregateDelays:
    """Tests pour la fonction _aggregate_delays."""

    def test_aggregate_by_indicator_only(self, multi_observation_delays):
        """Test d'agrégation par indicateur uniquement."""
        delays = multi_observation_delays.copy()
        delays['converted_delay'] = delays['release_delay']
        delays['target_frequency'] = 'M'

        result = _aggregate_delays(
            delays=delays,
            indicator_level_name='indicator',
            aggregate_by_panel=False,
            aggregation_method='median',
            target_reference_point='end',
            target_freq_map={'PIB': 'M'}
        )

        # Devrait avoir une seule ligne (PIB agrégé)
        assert len(result) == 1
        assert 'applicable_delay' in result.columns
        assert 'n_observations' in result.columns
        assert result.loc['PIB', 'n_observations'] == 6  # Total des observations

    def test_aggregate_by_panel(self, multi_observation_delays):
        """Test d'agrégation par entité de panel."""
        delays = multi_observation_delays.copy()
        delays['converted_delay'] = delays['release_delay']
        delays['target_frequency'] = 'M'

        result = _aggregate_delays(
            delays=delays,
            indicator_level_name='indicator',
            aggregate_by_panel=True,
            aggregation_method='median',
            target_reference_point='end',
            target_freq_map={'PIB': 'M'}
        )

        # Devrait avoir deux lignes (France/PIB et Germany/PIB)
        assert len(result) == 2
        assert ('France', 'PIB') in result.index
        assert ('Germany', 'PIB') in result.index

    def test_aggregation_method_mean(self, multi_observation_delays):
        """Test avec méthode d'agrégation 'mean'."""
        delays = multi_observation_delays.copy()
        delays['converted_delay'] = delays['release_delay']
        delays['target_frequency'] = 'M'

        result = _aggregate_delays(
            delays=delays,
            indicator_level_name='indicator',
            aggregate_by_panel=True,
            aggregation_method='mean',
            target_reference_point='end',
            target_freq_map={'PIB': 'M'}
        )

        # France: mean(14, 13, 15, 14) = 14
        assert result.loc[('France', 'PIB'), 'applicable_delay'] == pytest.approx(14, rel=0.1)

    def test_aggregation_method_max(self, multi_observation_delays):
        """Test avec méthode d'agrégation 'max'."""
        delays = multi_observation_delays.copy()
        delays['converted_delay'] = delays['release_delay']
        delays['target_frequency'] = 'M'

        result = _aggregate_delays(
            delays=delays,
            indicator_level_name='indicator',
            aggregate_by_panel=True,
            aggregation_method='max',
            target_reference_point='end',
            target_freq_map={'PIB': 'M'}
        )

        # France: max(14, 13, 15, 14) = 15
        assert result.loc[('France', 'PIB'), 'applicable_delay'] == 15

    def test_aggregation_metadata(self, multi_observation_delays):
        """Test des métadonnées d'agrégation."""
        delays = multi_observation_delays.copy()
        delays['converted_delay'] = delays['release_delay']
        delays['target_frequency'] = 'M'

        result = _aggregate_delays(
            delays=delays,
            indicator_level_name='indicator',
            aggregate_by_panel=False,
            aggregation_method='median',
            target_reference_point='end',
            target_freq_map={'PIB': 'M'}
        )

        assert result.loc['PIB', 'target_reference_point'] == 'end'
        assert result.loc['PIB', 'aggregation_method'] == 'median'
        assert result.loc['PIB', 'target_frequency'] == 'M'


# ============================================================================
# Tests pour calculate_applicable_delay
# ============================================================================

class TestCalculateApplicableDelay:
    """Tests pour la fonction principale calculate_applicable_delay."""

    def test_basic_usage(self, monthly_publication_delays):
        """Test d'utilisation basique."""
        result = calculate_applicable_delay(
            publication_delays=monthly_publication_delays,
            target_reference_point='end',
            target_frequency='M'
        )

        assert isinstance(result, pd.DataFrame)
        assert 'applicable_delay' in result.columns
        assert 'unit' in result.columns
        assert 'target_frequency' in result.columns
        assert 'n_observations' in result.columns

    def test_with_specific_indicators(self, monthly_publication_delays):
        """Test avec indicateurs spécifiques."""
        result = calculate_applicable_delay(
            publication_delays=monthly_publication_delays,
            target_reference_point='end',
            target_frequency='M',
            indicators=['PIB']
        )

        # Seul PIB devrait être dans le résultat
        indicator_level = result.index.get_level_values(-1)
        assert set(indicator_level) == {'PIB'}

    def test_with_invalid_indicators(self, monthly_publication_delays):
        """Test avec indicateurs inexistants."""
        with pytest.raises(ValueError, match="No data found for specified indicators"):
            calculate_applicable_delay(
                publication_delays=monthly_publication_delays,
                target_reference_point='end',
                target_frequency='M',
                indicators=['inexistant']
            )

    def test_aggregate_by_panel_true(self, monthly_publication_delays):
        """Test avec agrégation par panel activée."""
        result = calculate_applicable_delay(
            publication_delays=monthly_publication_delays,
            target_reference_point='end',
            target_frequency='M',
            aggregate_by_panel=True
        )

        # L'index devrait contenir (country, indicator)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.nlevels == 2

    def test_aggregate_by_panel_false(self, monthly_publication_delays):
        """Test avec agrégation par panel désactivée."""
        result = calculate_applicable_delay(
            publication_delays=monthly_publication_delays,
            target_reference_point='end',
            target_frequency='M',
            aggregate_by_panel=False
        )

        # L'index devrait contenir uniquement l'indicateur
        assert not isinstance(result.index, pd.MultiIndex)
        assert set(result.index) == {'PIB', 'inflation'}

    def test_different_aggregation_methods(self, multi_observation_delays):
        """Test avec différentes méthodes d'agrégation."""
        for method in ['mean', 'median', 'max', 'min']:
            result = calculate_applicable_delay(
                publication_delays=multi_observation_delays,
                target_reference_point='end',
                target_frequency='M',
                aggregation_method=method
            )

            assert result.loc['PIB', 'aggregation_method'] == method

    def test_output_unit_conversion(self, monthly_publication_delays):
        """Test de conversion de l'unité de sortie."""
        result_days = calculate_applicable_delay(
            publication_delays=monthly_publication_delays,
            target_reference_point='end',
            target_frequency='M',
            output_unit='D'
        )

        result_seconds = calculate_applicable_delay(
            publication_delays=monthly_publication_delays,
            target_reference_point='end',
            target_frequency='M',
            output_unit='s'
        )

        # Les délais en secondes devraient être 86400 fois plus grands
        ratio = result_seconds['applicable_delay'].iloc[0] / result_days['applicable_delay'].iloc[0]
        assert ratio == pytest.approx(86400, rel=0.01)

    def test_target_frequency_as_dict(self, monthly_publication_delays):
        """Test avec fréquence cible comme dictionnaire."""
        result = calculate_applicable_delay(
            publication_delays=monthly_publication_delays,
            target_reference_point='end',
            target_frequency={'PIB': 'M', 'inflation': 'M'}
        )

        assert len(result) > 0

    def test_reference_point_start(self, monthly_publication_delays):
        """Test avec point de référence 'start'."""
        result = calculate_applicable_delay(
            publication_delays=monthly_publication_delays,
            target_reference_point='start',
            target_frequency='M'
        )

        assert (result['target_reference_point'] == 'start').all()
        # Les délais depuis le début devraient être plus grands
        # (car ils incluent la durée de la période)

    def test_reference_point_validation(self, monthly_publication_delays):
        """Test de validation du point de référence."""
        with pytest.raises(ValueError, match="target_reference_point must be"):
            calculate_applicable_delay(
                publication_delays=monthly_publication_delays,
                target_reference_point='middle',
                target_frequency='M'
            )

    def test_frequency_conversion_quarterly_to_monthly(self, quarterly_publication_delays):
        """Test de conversion de fréquence Q → M."""
        result = calculate_applicable_delay(
            publication_delays=quarterly_publication_delays,
            target_reference_point='end',
            target_frequency='M',
            aggregate_by_panel=True
        )

        assert (result['target_frequency'] == 'M').all()

    def test_frequency_conversion_monthly_to_quarterly(self, monthly_publication_delays):
        """Test de conversion de fréquence M → Q."""
        result = calculate_applicable_delay(
            publication_delays=monthly_publication_delays,
            target_reference_point='end',
            target_frequency='Q',
            aggregate_by_panel=True
        )

        assert (result['target_frequency'] == 'Q').all()

    def test_simple_time_series_no_panel(self, simple_time_series_delays):
        """Test avec série temporelle simple (sans données de panel)."""
        result = calculate_applicable_delay(
            publication_delays=simple_time_series_delays,
            target_reference_point='end',
            target_frequency='M'
        )

        assert len(result) == 3  # PIB, inflation, unemployment
        assert set(result.index) == {'PIB', 'inflation', 'unemployment'}

    def test_custom_aggregation_function(self, multi_observation_delays):
        """Test avec fonction d'agrégation personnalisée."""
        def custom_agg(x):
            return x.quantile(0.75)

        result = calculate_applicable_delay(
            publication_delays=multi_observation_delays,
            target_reference_point='end',
            target_frequency='M',
            aggregation_method=custom_agg
        )

        assert 'applicable_delay' in result.columns

    def test_invalid_target_frequency_type(self, monthly_publication_delays):
        """Test avec type de fréquence cible invalide."""
        with pytest.raises(TypeError, match="'target_frequency' should be"):
            calculate_applicable_delay(
                publication_delays=monthly_publication_delays,
                target_reference_point='end',
                target_frequency=123  # Type invalide
            )


# ============================================================================
# Tests d'intégration avec compare_and_detect_delays
# ============================================================================

class TestIntegrationWithDataManager:
    """Tests d'intégration avec le module data_manager."""

    def test_full_pipeline_simple_time_series(self):
        """Test du pipeline complet pour une série temporelle simple."""
        # Création des données
        dates = pd.date_range('2023-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'PIB': list(range(12)),
            'inflation': [2.5 + i * 0.1 for i in range(12)]
        }, index=dates)

        # Détection des délais
        publication_delays = compare_and_detect_delays(
            new_data=data,
            existing_data=None,
            download_date='2024-01-15',
            reference_point='end'
        )

        # Calcul du délai applicable
        result = calculate_applicable_delay(
            publication_delays=publication_delays,
            target_reference_point='end',
            target_frequency='M'
        )

        assert len(result) == 2  # PIB et inflation
        assert (result['applicable_delay'] > 0).all()

    def test_full_pipeline_panel_data(self):
        """Test du pipeline complet pour des données de panel."""
        # Création des données de panel
        countries = ['France', 'Germany']
        dates = pd.date_range('2023-01-01', periods=6, freq='MS')

        index_tuples = []
        pib_values = []

        for country in countries:
            for i, date in enumerate(dates):
                index_tuples.append((country, date))
                pib_values.append(100 + i)

        index = pd.MultiIndex.from_tuples(index_tuples, names=['country', 'date'])
        data = pd.DataFrame({'PIB': pib_values}, index=index)

        # Détection des délais
        publication_delays = compare_and_detect_delays(
            new_data=data,
            existing_data=None,
            download_date='2023-07-15'
        )

        # Calcul du délai applicable par pays
        result = calculate_applicable_delay(
            publication_delays=publication_delays,
            target_reference_point='end',
            target_frequency='M',
            aggregate_by_panel=True
        )

        # Devrait avoir 2 résultats (France/PIB, Germany/PIB)
        assert len(result) == 2

    def test_full_pipeline_frequency_conversion(self):
        """Test du pipeline avec conversion de fréquence."""
        # Création de données trimestrielles
        dates = pd.date_range('2023-01-01', periods=4, freq='QS')
        data = pd.DataFrame({
            'PIB': [100, 102, 104, 106]
        }, index=dates)

        # Détection des délais
        publication_delays = compare_and_detect_delays(
            new_data=data,
            existing_data=None,
            download_date='2024-01-15',
            reference_point='end'
        )

        # Calcul du délai applicable avec conversion vers mensuel
        result = calculate_applicable_delay(
            publication_delays=publication_delays,
            target_reference_point='end',
            target_frequency='M'
        )

        assert (result['target_frequency'] == 'M').all()


# ============================================================================
# Tests pour les cas limites (edge cases)
# ============================================================================

class TestEdgeCases:
    """Tests pour les cas limites."""

    def test_single_observation(self):
        """Test avec une seule observation."""
        index = pd.Index(['PIB'], name='indicator')
        data = {
            'observation_date': [datetime(2023, 12, 1)],
            'download_date': [datetime(2024, 1, 15)],
            'frequency': ['M'],
            'period_start': [datetime(2023, 12, 1)],
            'period_end': [datetime(2024, 1, 1)],
            'reference_point': ['end'],
            'release_delay': [14],
            'unit': ['day']
        }
        delays = pd.DataFrame(data, index=index)

        result = calculate_applicable_delay(
            publication_delays=delays,
            target_reference_point='end',
            target_frequency='M'
        )

        assert len(result) == 1
        assert result.loc['PIB', 'n_observations'] == 1

    def test_negative_delay(self):
        """Test avec délai négatif (publication avant fin de période)."""
        index = pd.Index(['PIB'], name='indicator')
        data = {
            'observation_date': [datetime(2023, 12, 15)],
            'download_date': [datetime(2023, 12, 25)],  # Avant fin de période
            'frequency': ['M'],
            'period_start': [datetime(2023, 12, 1)],
            'period_end': [datetime(2024, 1, 1)],
            'reference_point': ['end'],
            'release_delay': [-7],  # 7 jours avant la fin
            'unit': ['day']
        }
        delays = pd.DataFrame(data, index=index)

        result = calculate_applicable_delay(
            publication_delays=delays,
            target_reference_point='end',
            target_frequency='M'
        )

        # Le délai négatif devrait être préservé
        assert result.loc['PIB', 'applicable_delay'] < 0

    def test_very_large_delay(self):
        """Test avec délai très grand."""
        index = pd.Index(['PIB'], name='indicator')
        data = {
            'observation_date': [datetime(2020, 1, 1)],
            'download_date': [datetime(2024, 1, 15)],  # 4 ans plus tard
            'frequency': ['M'],
            'period_start': [datetime(2020, 1, 1)],
            'period_end': [datetime(2020, 2, 1)],
            'reference_point': ['end'],
            'release_delay': [1440],  # ~4 ans en jours
            'unit': ['day']
        }
        delays = pd.DataFrame(data, index=index)

        result = calculate_applicable_delay(
            publication_delays=delays,
            target_reference_point='end',
            target_frequency='M'
        )

        assert result.loc['PIB', 'applicable_delay'] > 1000

    def test_zero_delay(self):
        """Test avec délai nul."""
        index = pd.Index(['PIB'], name='indicator')
        data = {
            'observation_date': [datetime(2023, 12, 15)],
            'download_date': [datetime(2024, 1, 1)],  # Exactement à la fin de période
            'frequency': ['M'],
            'period_start': [datetime(2023, 12, 1)],
            'period_end': [datetime(2024, 1, 1)],
            'reference_point': ['end'],
            'release_delay': [0],
            'unit': ['day']
        }
        delays = pd.DataFrame(data, index=index)

        result = calculate_applicable_delay(
            publication_delays=delays,
            target_reference_point='end',
            target_frequency='M'
        )

        assert result.loc['PIB', 'applicable_delay'] == 0

    def test_mixed_units(self):
        """Test avec unités mixtes."""
        index = pd.MultiIndex.from_tuples(
            [('France', 'PIB'), ('Germany', 'PIB')],
            names=['country', 'indicator']
        )
        data = {
            'observation_date': [datetime(2023, 12, 1), datetime(2023, 12, 1)],
            'download_date': [datetime(2024, 1, 15), datetime(2024, 1, 15)],
            'frequency': ['M', 'M'],
            'period_start': [datetime(2023, 12, 1), datetime(2023, 12, 1)],
            'period_end': [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            'reference_point': ['end', 'end'],
            'release_delay': [14, 14 * 86400],  # Jours et secondes
            'unit': ['day', 'second']
        }
        delays = pd.DataFrame(data, index=index)

        result = calculate_applicable_delay(
            publication_delays=delays,
            target_reference_point='end',
            target_frequency='M',
            output_unit='D'
        )

        # Les deux devraient donner ~14 jours après conversion
        assert all(result['applicable_delay'].between(13, 15))

    def test_empty_dataframe_after_filtering(self, monthly_publication_delays):
        """Test avec DataFrame vide après filtrage des indicateurs."""
        with pytest.raises(ValueError, match="No data found for specified indicators"):
            calculate_applicable_delay(
                publication_delays=monthly_publication_delays,
                target_reference_point='end',
                target_frequency='M',
                indicators=['inexistant']
            )

    def test_special_frequency_aliases(self, monthly_publication_delays):
        """Test avec différents alias de fréquence supportés."""
        # Note: Seuls les alias supportés par le normalisateur sont testés
        for freq_alias in ['monthly', 'M']:
            result = calculate_applicable_delay(
                publication_delays=monthly_publication_delays,
                target_reference_point='end',
                target_frequency=freq_alias
            )
            assert len(result) > 0

    def test_yearly_frequency(self):
        """Test avec fréquence annuelle."""
        index = pd.Index(['PIB'], name='indicator')
        data = {
            'observation_date': [datetime(2023, 6, 15)],
            'download_date': [datetime(2024, 3, 15)],
            'frequency': ['Y'],
            'period_start': [datetime(2023, 1, 1)],
            'period_end': [datetime(2024, 1, 1)],
            'reference_point': ['end'],
            'release_delay': [74],  # ~2.5 mois après fin d'année
            'unit': ['day']
        }
        delays = pd.DataFrame(data, index=index)

        result = calculate_applicable_delay(
            publication_delays=delays,
            target_reference_point='end',
            target_frequency='Y'
        )

        assert len(result) == 1

    def test_weekly_frequency(self):
        """Test avec fréquence hebdomadaire."""
        index = pd.Index(['sales'], name='indicator')
        data = {
            'observation_date': [datetime(2023, 12, 15)],
            'download_date': [datetime(2023, 12, 20)],
            'frequency': ['W'],
            'period_start': [datetime(2023, 12, 11)],  # Lundi
            'period_end': [datetime(2023, 12, 18)],    # Lundi suivant
            'reference_point': ['end'],
            'release_delay': [2],
            'unit': ['day']
        }
        delays = pd.DataFrame(data, index=index)

        result = calculate_applicable_delay(
            publication_delays=delays,
            target_reference_point='end',
            target_frequency='W'
        )

        assert len(result) == 1

    def test_preserve_index_names(self, monthly_publication_delays):
        """Test que les noms d'index sont préservés."""
        result = calculate_applicable_delay(
            publication_delays=monthly_publication_delays,
            target_reference_point='end',
            target_frequency='M',
            aggregate_by_panel=True
        )

        # Les noms d'index devraient être préservés
        assert result.index.names == ['country', 'indicator']

    def test_result_column_order(self, monthly_publication_delays):
        """Test de l'ordre des colonnes du résultat."""
        result = calculate_applicable_delay(
            publication_delays=monthly_publication_delays,
            target_reference_point='end',
            target_frequency='M'
        )

        expected_columns = [
            'applicable_delay',
            'unit',
            'target_frequency',
            'target_reference_point',
            'n_observations',
            'aggregation_method'
        ]

        assert list(result.columns) == expected_columns
