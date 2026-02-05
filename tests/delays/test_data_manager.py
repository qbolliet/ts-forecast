"""Tests pour le module data_manager.

Ce module teste les fonctions de détection et de calcul des délais de publication:
- compare_and_detect_delays(): Fonction principale de comparaison et détection
- _validate_input_data(): Validation des données en entrée
- _identify_new_observations(): Identification des nouvelles observations
- _calculate_release_delays(): Calcul des délais de publication
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

from tsforecast.delays.data_manager import (
    compare_and_detect_delays,
    _validate_input_data,
    _identify_new_observations,
    _calculate_release_delays
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_time_series_data():
    """Crée des données de série temporelle simple."""
    dates = pd.date_range('2023-01-01', periods=12, freq='MS')
    data = pd.DataFrame({
        'PIB': [100 + i for i in range(12)],
        'inflation': [2.5 + i * 0.1 for i in range(12)],
        'unemployment': [5.0 - i * 0.1 for i in range(12)]
    }, index=dates)
    return data


@pytest.fixture
def simple_time_series_with_nulls():
    """Crée des données de série temporelle avec des valeurs manquantes."""
    dates = pd.date_range('2023-01-01', periods=12, freq='MS')
    data = pd.DataFrame({
        'PIB': [100, 101, 102, np.nan, 104, 105, np.nan, 107, 108, 109, np.nan, 111],
        'inflation': [2.5, 2.6, 2.7, 2.8, np.nan, 3.0, 3.1, 3.2, np.nan, 3.4, 3.5, 3.6]
    }, index=dates)
    return data


@pytest.fixture
def panel_data():
    """Crée des données de panel avec MultiIndex."""
    countries = ['France', 'Germany', 'Italy']
    dates = pd.date_range('2023-01-01', periods=6, freq='MS')

    index_tuples = []
    pib_values = []
    inflation_values = []

    for country in countries:
        for i, date in enumerate(dates):
            index_tuples.append((country, date))
            pib_values.append(100 + i + (0 if country == 'France' else 10 if country == 'Germany' else 20))
            inflation_values.append(2.5 + i * 0.1)

    index = pd.MultiIndex.from_tuples(index_tuples, names=['country', 'date'])
    data = pd.DataFrame({
        'PIB': pib_values,
        'inflation': inflation_values
    }, index=index)

    return data


@pytest.fixture
def panel_data_with_nulls():
    """Crée des données de panel avec des valeurs manquantes."""
    countries = ['France', 'Germany']
    dates = pd.date_range('2023-01-01', periods=4, freq='MS')

    index_tuples = []
    pib_values = []

    for country in countries:
        for i, date in enumerate(dates):
            index_tuples.append((country, date))
            # France a une valeur manquante en mars, Germany en février
            if country == 'France' and i == 2:
                pib_values.append(np.nan)
            elif country == 'Germany' and i == 1:
                pib_values.append(np.nan)
            else:
                pib_values.append(100 + i)

    index = pd.MultiIndex.from_tuples(index_tuples, names=['country', 'date'])
    data = pd.DataFrame({'PIB': pib_values}, index=index)

    return data


@pytest.fixture
def quarterly_data():
    """Crée des données trimestrielles."""
    dates = pd.date_range('2023-01-01', periods=4, freq='QS')
    data = pd.DataFrame({
        'PIB': [100, 102, 104, 106],
        'consumption': [80, 82, 84, 86]
    }, index=dates)
    return data


# ============================================================================
# Tests pour _validate_input_data
# ============================================================================

class TestValidateInputData:
    """Tests pour la fonction _validate_input_data."""

    def test_validate_simple_time_series(self, simple_time_series_data):
        """Test de validation d'une série temporelle simple."""
        validated = _validate_input_data(simple_time_series_data)

        assert isinstance(validated, pd.DataFrame)
        assert isinstance(validated.index, pd.DatetimeIndex)
        assert len(validated) == len(simple_time_series_data)

    def test_validate_with_time_col(self):
        """Test de validation avec colonne temporelle explicite."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value': [1, 2, 3, 4, 5]
        })

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            validated = _validate_input_data(df, time_col='date')

        assert isinstance(validated.index, pd.DatetimeIndex)
        assert 'value' in validated.columns

    def test_validate_panel_data(self, panel_data):
        """Test de validation de données de panel."""
        validated = _validate_input_data(panel_data)

        assert isinstance(validated, pd.DataFrame)
        assert isinstance(validated.index, pd.MultiIndex)

    def test_validate_with_panel_cols(self):
        """Test de validation avec colonnes de panel explicites."""
        df = pd.DataFrame({
            'country': ['France', 'France', 'Germany', 'Germany'],
            'date': pd.date_range('2023-01-01', periods=2, freq='MS').tolist() * 2,
            'value': [1, 2, 3, 4]
        })

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            validated = _validate_input_data(df, time_col='date', panel_cols=['country'])

        assert isinstance(validated.index, pd.MultiIndex)
        assert validated.index.nlevels == 2

    def test_validate_unsorted_data(self):
        """Test de validation avec données non triées."""
        dates = pd.to_datetime(['2023-03-01', '2023-01-01', '2023-02-01'])
        df = pd.DataFrame({'value': [3, 1, 2]}, index=dates)

        validated = _validate_input_data(df)

        # Les données doivent être triées
        assert validated.index.is_monotonic_increasing

    def test_validate_non_datetime_index(self):
        """Test avec index non datetime qui peut être converti."""
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=['2023-01-01', '2023-01-02', '2023-01-03'])

        validated = _validate_input_data(df)

        assert isinstance(validated.index, pd.DatetimeIndex)


# ============================================================================
# Tests pour _identify_new_observations
# ============================================================================

class TestIdentifyNewObservations:
    """Tests pour la fonction _identify_new_observations."""

    def test_identify_with_no_existing_data_simple(self, simple_time_series_data):
        """Test d'identification sans données existantes (cas série temporelle simple)."""
        # Validation préalable
        validated = _validate_input_data(simple_time_series_data)

        new_obs = _identify_new_observations(validated, existing_data=None)

        # Devrait retourner la dernière observation non-nulle pour chaque colonne
        assert not new_obs.empty
        assert 'column' in new_obs.columns
        assert 'has_changes' in new_obs.columns

        # Une observation par colonne (la plus récente)
        columns_found = new_obs['column'].unique()
        assert set(columns_found) == {'PIB', 'inflation', 'unemployment'}

    def test_identify_with_no_existing_data_with_nulls(self, simple_time_series_with_nulls):
        """Test d'identification sans données existantes avec valeurs nulles."""
        validated = _validate_input_data(simple_time_series_with_nulls)

        new_obs = _identify_new_observations(validated, existing_data=None)

        # La dernière observation non-nulle devrait être retournée
        # Pour PIB: dernière non-nulle est index 11 (111)
        # Pour inflation: dernière non-nulle est index 11 (3.6)
        assert len(new_obs) == 2  # Une par colonne

    def test_identify_with_no_existing_data_panel(self, panel_data):
        """Test d'identification sans données existantes pour données de panel."""
        validated = _validate_input_data(panel_data)

        new_obs = _identify_new_observations(validated, existing_data=None)

        # Devrait retourner la dernière observation pour chaque (entité, colonne)
        assert not new_obs.empty

        # 3 pays × 2 colonnes = 6 observations
        assert len(new_obs) == 6

    def test_identify_new_only_mode(self):
        """Test du mode 'new_only' (null → non-null)."""
        dates = pd.date_range('2023-01-01', periods=3, freq='MS')

        existing = pd.DataFrame({
            'PIB': [100, np.nan, 102],
            'inflation': [2.5, 2.6, np.nan]
        }, index=dates)

        new_data = pd.DataFrame({
            'PIB': [100, 101, 102],  # 101 est nouveau (était null)
            'inflation': [2.5, 2.6, 2.7]  # 2.7 est nouveau (était null)
        }, index=dates)

        existing_validated = _validate_input_data(existing)
        new_validated = _validate_input_data(new_data)

        new_obs = _identify_new_observations(
            new_validated,
            existing_data=existing_validated,
            detection_mode='new_only'
        )

        # 2 changements: PIB en février et inflation en mars
        assert len(new_obs) == 2

    def test_identify_all_changes_mode(self):
        """Test du mode 'all_changes' (toute modification)."""
        dates = pd.date_range('2023-01-01', periods=3, freq='MS')

        existing = pd.DataFrame({
            'PIB': [100, 101, 102],
            'inflation': [2.5, 2.6, np.nan]
        }, index=dates)

        new_data = pd.DataFrame({
            'PIB': [100, 105, 102],  # 105 est une modification (était 101)
            'inflation': [2.5, 2.6, 2.7]  # 2.7 est nouveau (était null)
        }, index=dates)

        existing_validated = _validate_input_data(existing)
        new_validated = _validate_input_data(new_data)

        new_obs = _identify_new_observations(
            new_validated,
            existing_data=existing_validated,
            detection_mode='all_changes'
        )

        # 2 changements: PIB modifié en février et inflation nouvelle en mars
        assert len(new_obs) == 2

    def test_identify_no_changes(self):
        """Test quand il n'y a pas de changements."""
        dates = pd.date_range('2023-01-01', periods=3, freq='MS')

        data = pd.DataFrame({
            'PIB': [100, 101, 102],
            'inflation': [2.5, 2.6, 2.7]
        }, index=dates)

        validated = _validate_input_data(data)

        new_obs = _identify_new_observations(
            validated,
            existing_data=validated.copy(),
            detection_mode='new_only'
        )

        # Aucun changement
        assert len(new_obs) == 0

    def test_identify_panel_changes(self, panel_data_with_nulls):
        """Test d'identification de changements dans des données de panel."""
        validated = _validate_input_data(panel_data_with_nulls)

        # Créer des nouvelles données avec les valeurs remplies
        new_data = validated.copy()
        new_data['PIB'] = new_data['PIB'].fillna(999)

        new_obs = _identify_new_observations(
            new_data,
            existing_data=validated,
            detection_mode='new_only'
        )

        # 2 changements: France en mars et Germany en février
        assert len(new_obs) == 2

    def test_identify_empty_result(self):
        """Test avec données entièrement nulles."""
        dates = pd.date_range('2023-01-01', periods=3, freq='MS')

        data = pd.DataFrame({
            'PIB': [np.nan, np.nan, np.nan]
        }, index=dates)

        validated = _validate_input_data(data)

        new_obs = _identify_new_observations(validated, existing_data=None)

        # Aucune observation non-nulle
        assert len(new_obs) == 0


# ============================================================================
# Tests pour _calculate_release_delays
# ============================================================================

class TestCalculateReleaseDelays:
    """Tests pour la fonction _calculate_release_delays."""

    def test_calculate_delays_basic(self, simple_time_series_data):
        """Test de calcul basique des délais."""
        validated = _validate_input_data(simple_time_series_data)
        new_obs = _identify_new_observations(validated, existing_data=None)

        download_date = datetime(2023, 12, 15)

        delays = _calculate_release_delays(
            new_observations=new_obs,
            new_data=validated,
            download_date=download_date,
            reference_point='end',
            unit='day'
        )

        assert 'observation_date' in delays.columns
        assert 'download_date' in delays.columns
        assert 'frequency' in delays.columns
        assert 'period_start' in delays.columns
        assert 'period_end' in delays.columns
        assert 'reference_point' in delays.columns
        assert 'release_delay' in delays.columns
        assert 'unit' in delays.columns

    def test_calculate_delays_reference_point_start(self, simple_time_series_data):
        """Test avec point de référence 'start'."""
        validated = _validate_input_data(simple_time_series_data)
        new_obs = _identify_new_observations(validated, existing_data=None)

        download_date = datetime(2023, 12, 15)

        delays = _calculate_release_delays(
            new_observations=new_obs,
            new_data=validated,
            download_date=download_date,
            reference_point='start',
            unit='day'
        )

        assert (delays['reference_point'] == 'start').all()
        # Le délai depuis le début devrait être plus grand que depuis la fin
        assert (delays['release_delay'] > 0).all()

    def test_calculate_delays_reference_point_end(self, simple_time_series_data):
        """Test avec point de référence 'end'."""
        validated = _validate_input_data(simple_time_series_data)
        new_obs = _identify_new_observations(validated, existing_data=None)

        download_date = datetime(2023, 12, 15)

        delays = _calculate_release_delays(
            new_observations=new_obs,
            new_data=validated,
            download_date=download_date,
            reference_point='end',
            unit='day'
        )

        assert (delays['reference_point'] == 'end').all()

    def test_calculate_delays_unit_day(self, simple_time_series_data):
        """Test avec unité en jours."""
        validated = _validate_input_data(simple_time_series_data)
        new_obs = _identify_new_observations(validated, existing_data=None)

        download_date = datetime(2023, 12, 15)

        delays = _calculate_release_delays(
            new_observations=new_obs,
            new_data=validated,
            download_date=download_date,
            reference_point='end',
            unit='day'
        )

        assert (delays['unit'] == 'day').all()
        # Vérifier que les délais sont des entiers (ceiling)
        assert all(delays['release_delay'] == delays['release_delay'].astype(int))

    def test_calculate_delays_unit_second(self, simple_time_series_data):
        """Test avec unité en secondes."""
        validated = _validate_input_data(simple_time_series_data)
        new_obs = _identify_new_observations(validated, existing_data=None)

        # Date de téléchargement après la dernière période pour avoir des délais positifs
        download_date = datetime(2024, 1, 15, 12, 30, 45)

        delays = _calculate_release_delays(
            new_observations=new_obs,
            new_data=validated,
            download_date=download_date,
            reference_point='end',
            unit='second'
        )

        assert (delays['unit'] == 'second').all()
        # Les délais en secondes devraient être positifs (au moins quelques jours après la fin de période)
        assert (delays['release_delay'] > 0).all()

    def test_calculate_delays_unit_microsecond(self, simple_time_series_data):
        """Test avec unité en microsecondes."""
        validated = _validate_input_data(simple_time_series_data)
        new_obs = _identify_new_observations(validated, existing_data=None)

        download_date = datetime(2023, 12, 15)

        delays = _calculate_release_delays(
            new_observations=new_obs,
            new_data=validated,
            download_date=download_date,
            reference_point='end',
            unit='us'
        )

        assert (delays['unit'] == 'microsecond').all()

    def test_calculate_delays_invalid_unit(self, simple_time_series_data):
        """Test avec unité invalide."""
        validated = _validate_input_data(simple_time_series_data)
        new_obs = _identify_new_observations(validated, existing_data=None)

        download_date = datetime(2023, 12, 15)

        with pytest.raises(ValueError, match="Unit must be one of"):
            _calculate_release_delays(
                new_observations=new_obs,
                new_data=validated,
                download_date=download_date,
                reference_point='end',
                unit='invalid'
            )

    def test_calculate_delays_panel_data(self, panel_data):
        """Test de calcul des délais pour données de panel."""
        validated = _validate_input_data(panel_data)
        new_obs = _identify_new_observations(validated, existing_data=None)

        download_date = datetime(2023, 7, 15)

        delays = _calculate_release_delays(
            new_observations=new_obs,
            new_data=validated,
            download_date=download_date,
            reference_point='end',
            unit='day'
        )

        # L'index devrait être un MultiIndex (pays, indicateur)
        assert isinstance(delays.index, pd.MultiIndex)
        assert len(delays) == 6  # 3 pays × 2 colonnes

    def test_calculate_delays_quarterly_data(self, quarterly_data):
        """Test de calcul des délais pour données trimestrielles."""
        validated = _validate_input_data(quarterly_data)
        new_obs = _identify_new_observations(validated, existing_data=None)

        download_date = datetime(2023, 12, 15)

        delays = _calculate_release_delays(
            new_observations=new_obs,
            new_data=validated,
            download_date=download_date,
            reference_point='end',
            unit='day'
        )

        # Vérifier que la fréquence détectée est trimestrielle
        assert all(freq in ['Q', 'quarterly', 'QS', 'QE'] for freq in delays['frequency'])


# ============================================================================
# Tests pour compare_and_detect_delays
# ============================================================================

class TestCompareAndDetectDelays:
    """Tests pour la fonction principale compare_and_detect_delays."""

    def test_basic_usage_with_existing_data(self, simple_time_series_data):
        """Test d'utilisation basique avec données existantes."""
        # Créer des données existantes avec des valeurs manquantes
        existing = simple_time_series_data.copy()
        existing.iloc[-1, :] = np.nan  # Dernière ligne manquante

        download_date = datetime(2023, 12, 15)

        result = compare_and_detect_delays(
            new_data=simple_time_series_data,
            existing_data=existing,
            download_date=download_date,
            reference_point='end'
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'release_delay' in result.columns

    def test_basic_usage_new_data_only(self, simple_time_series_data):
        """Test d'utilisation avec nouvelles données uniquement (sans existing_data)."""
        download_date = datetime(2023, 12, 15)

        result = compare_and_detect_delays(
            new_data=simple_time_series_data,
            existing_data=None,
            download_date=download_date,
            reference_point='end'
        )

        assert isinstance(result, pd.DataFrame)
        # Devrait contenir une observation par colonne (la plus récente)
        assert len(result) == 3  # 3 colonnes: PIB, inflation, unemployment

    def test_new_data_only_with_nulls(self, simple_time_series_with_nulls):
        """Test avec nouvelles données contenant des nulls (sans existing_data)."""
        download_date = datetime(2023, 12, 15)

        result = compare_and_detect_delays(
            new_data=simple_time_series_with_nulls,
            existing_data=None,
            download_date=download_date
        )

        # Devrait retourner la dernière observation non-nulle pour chaque colonne
        assert len(result) == 2  # 2 colonnes: PIB, inflation

    def test_new_data_only_panel(self, panel_data):
        """Test avec données de panel uniquement (sans existing_data)."""
        download_date = datetime(2023, 7, 15)

        result = compare_and_detect_delays(
            new_data=panel_data,
            existing_data=None,
            download_date=download_date
        )

        # 3 pays × 2 colonnes = 6 observations
        assert len(result) == 6
        assert isinstance(result.index, pd.MultiIndex)

    def test_download_date_none_uses_now(self, simple_time_series_data):
        """Test que download_date=None utilise la date actuelle."""
        result = compare_and_detect_delays(
            new_data=simple_time_series_data,
            existing_data=None,
            download_date=None
        )

        # La date de téléchargement devrait être proche de maintenant
        download_dates = result['download_date'].unique()
        assert len(download_dates) == 1

        now = datetime.now()
        assert (now - download_dates[0]).total_seconds() < 60  # Moins d'une minute

    def test_download_date_string(self, simple_time_series_data):
        """Test avec download_date comme chaîne de caractères."""
        result = compare_and_detect_delays(
            new_data=simple_time_series_data,
            existing_data=None,
            download_date='2023-12-15'
        )

        download_dates = result['download_date'].unique()
        assert download_dates[0] == datetime(2023, 12, 15)

    def test_detection_mode_new_only(self):
        """Test du mode de détection 'new_only'."""
        dates = pd.date_range('2023-01-01', periods=3, freq='MS')

        existing = pd.DataFrame({
            'PIB': [100, np.nan, 102]
        }, index=dates)

        new_data = pd.DataFrame({
            'PIB': [100, 101, 102]
        }, index=dates)

        result = compare_and_detect_delays(
            new_data=new_data,
            existing_data=existing,
            download_date='2023-04-15',
            detection_mode='new_only'
        )

        # Seul le passage null -> non-null devrait être détecté
        assert len(result) == 1

    def test_detection_mode_all_changes(self):
        """Test du mode de détection 'all_changes'."""
        dates = pd.date_range('2023-01-01', periods=3, freq='MS')

        existing = pd.DataFrame({
            'PIB': [100, 101, 102]
        }, index=dates)

        new_data = pd.DataFrame({
            'PIB': [100, 105, 102]  # Modification en février
        }, index=dates)

        result = compare_and_detect_delays(
            new_data=new_data,
            existing_data=existing,
            download_date='2023-04-15',
            detection_mode='all_changes'
        )

        # La modification devrait être détectée
        assert len(result) == 1

    def test_reference_point_validation(self, simple_time_series_data):
        """Test de validation du point de référence."""
        with pytest.raises(ValueError, match="reference_point must be"):
            compare_and_detect_delays(
                new_data=simple_time_series_data,
                existing_data=None,
                reference_point='middle'
            )

    def test_detection_mode_validation(self, simple_time_series_data):
        """Test de validation du mode de détection."""
        with pytest.raises(ValueError, match="detection_mode must be"):
            compare_and_detect_delays(
                new_data=simple_time_series_data,
                existing_data=simple_time_series_data,
                detection_mode='invalid_mode'
            )

    def test_with_time_col_parameter(self):
        """Test avec paramètre time_col."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='MS'),
            'PIB': [100, 101, 102, 103, 104]
        })

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compare_and_detect_delays(
                new_data=df,
                existing_data=None,
                download_date='2023-06-15',
                time_col='date'
            )

        assert len(result) == 1  # Une seule colonne (PIB)

    def test_with_panel_cols_parameter(self):
        """Test avec paramètre panel_cols."""
        df = pd.DataFrame({
            'country': ['France', 'France', 'Germany', 'Germany'],
            'date': pd.date_range('2023-01-01', periods=2, freq='MS').tolist() * 2,
            'PIB': [100, 101, 110, 111]
        })

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compare_and_detect_delays(
                new_data=df,
                existing_data=None,
                download_date='2023-03-15',
                time_col='date',
                panel_cols=['country']
            )

        # 2 pays × 1 colonne = 2 observations
        assert len(result) == 2


# ============================================================================
# Tests pour les cas limites (edge cases)
# ============================================================================

class TestEdgeCases:
    """Tests pour les cas limites."""

    def test_single_observation(self):
        """Test avec une seule observation.

        Note: La détection de fréquence nécessite au moins 2 observations,
        donc ce test vérifie qu'une erreur appropriée est levée.
        """
        dates = pd.date_range('2023-01-01', periods=1, freq='MS')
        data = pd.DataFrame({'PIB': [100]}, index=dates)

        # La détection de fréquence nécessite au moins 2 observations
        with pytest.raises(ValueError, match="minimum required is 2"):
            compare_and_detect_delays(
                new_data=data,
                existing_data=None,
                download_date='2023-02-15'
            )

    def test_empty_dataframe(self):
        """Test avec DataFrame vide.

        Note: La détection de fréquence nécessite au moins 2 observations,
        donc ce test vérifie qu'une erreur appropriée est levée.
        """
        data = pd.DataFrame({'PIB': []}, index=pd.DatetimeIndex([]))

        # La détection de fréquence nécessite des observations
        with pytest.raises(ValueError, match="minimum required is 2"):
            compare_and_detect_delays(
                new_data=data,
                existing_data=None,
                download_date='2023-02-15'
            )

    def test_all_null_values(self):
        """Test avec toutes les valeurs nulles.

        Note: La détection de fréquence nécessite des observations non-nulles,
        donc ce test vérifie qu'une erreur appropriée est levée.
        """
        dates = pd.date_range('2023-01-01', periods=5, freq='MS')
        data = pd.DataFrame({'PIB': [np.nan] * 5}, index=dates)

        # La détection de fréquence nécessite des observations non-nulles
        with pytest.raises(ValueError, match="minimum required is 2"):
            compare_and_detect_delays(
                new_data=data,
                existing_data=None,
                download_date='2023-06-15'
            )

    def test_duplicate_index(self):
        """Test avec index dupliqué (devrait être géré par la validation)."""
        dates = pd.to_datetime(['2023-01-01', '2023-01-01', '2023-02-01'])
        data = pd.DataFrame({'PIB': [100, 101, 102]}, index=dates)

        # La validation devrait lever une erreur ou supprimer les doublons
        with pytest.raises(ValueError):
            compare_and_detect_delays(
                new_data=data,
                existing_data=None,
                download_date='2023-03-15'
            )

    def test_unsorted_data(self):
        """Test avec données non triées."""
        dates = pd.to_datetime(['2023-03-01', '2023-01-01', '2023-02-01'])
        data = pd.DataFrame({'PIB': [103, 101, 102]}, index=dates)

        result = compare_and_detect_delays(
            new_data=data,
            existing_data=None,
            download_date='2023-04-15'
        )

        # Les données devraient être triées automatiquement
        # La dernière observation triée devrait être celle de mars
        assert len(result) == 1

    def test_different_columns_between_new_and_existing(self):
        """Test avec colonnes différentes entre new et existing."""
        dates = pd.date_range('2023-01-01', periods=3, freq='MS')

        existing = pd.DataFrame({
            'PIB': [100, 101, np.nan],
            'inflation': [2.5, 2.6, 2.7]
        }, index=dates)

        new_data = pd.DataFrame({
            'PIB': [100, 101, 102],
            'unemployment': [5.0, 5.1, 5.2]  # Nouvelle colonne
        }, index=dates)

        result = compare_and_detect_delays(
            new_data=new_data,
            existing_data=existing,
            download_date='2023-04-15',
            detection_mode='new_only'
        )

        # Seules les colonnes communes devraient être comparées
        # PIB avait un null qui est maintenant rempli
        assert len(result) == 1

    def test_no_common_columns(self):
        """Test sans colonnes communes entre new et existing."""
        dates = pd.date_range('2023-01-01', periods=3, freq='MS')

        existing = pd.DataFrame({
            'PIB': [100, 101, 102]
        }, index=dates)

        new_data = pd.DataFrame({
            'inflation': [2.5, 2.6, 2.7]
        }, index=dates)

        result = compare_and_detect_delays(
            new_data=new_data,
            existing_data=existing,
            download_date='2023-04-15'
        )

        # Aucune colonne commune, donc aucun changement détecté
        assert len(result) == 0

    def test_mixed_frequency_detection(self):
        """Test de détection avec données de fréquences mixtes."""
        # Données mensuelles
        monthly_dates = pd.date_range('2023-01-01', periods=6, freq='MS')
        monthly_data = pd.DataFrame({
            'monthly_indicator': list(range(6))
        }, index=monthly_dates)

        result = compare_and_detect_delays(
            new_data=monthly_data,
            existing_data=None,
            download_date='2023-07-15'
        )

        # La fréquence devrait être détectée comme mensuelle
        assert len(result) == 1
        assert result['frequency'].iloc[0] in ['M', 'monthly', 'MS', 'ME']

    def test_very_old_data(self):
        """Test avec des données très anciennes."""
        dates = pd.date_range('1990-01-01', periods=3, freq='MS')
        data = pd.DataFrame({'PIB': [100, 101, 102]}, index=dates)

        result = compare_and_detect_delays(
            new_data=data,
            existing_data=None,
            download_date='2023-12-15'
        )

        # Les délais devraient être très grands (en jours)
        assert len(result) == 1
        # Plus de 30 ans en jours
        assert result['release_delay'].iloc[0] > 10000

    def test_future_download_date(self):
        """Test avec date de téléchargement dans le futur par rapport aux données."""
        dates = pd.date_range('2023-01-01', periods=3, freq='MS')
        data = pd.DataFrame({'PIB': [100, 101, 102]}, index=dates)

        # Date de téléchargement avant certaines périodes
        result = compare_and_detect_delays(
            new_data=data,
            existing_data=None,
            download_date='2023-02-15'  # Avant la fin de la dernière période
        )

        # Les délais pourraient être négatifs ou très petits
        assert len(result) == 1

    def test_multiple_panel_levels(self):
        """Test avec plusieurs niveaux de panel."""
        index_tuples = [
            ('Europe', 'France', pd.Timestamp('2023-01-01')),
            ('Europe', 'France', pd.Timestamp('2023-02-01')),
            ('Europe', 'Germany', pd.Timestamp('2023-01-01')),
            ('Europe', 'Germany', pd.Timestamp('2023-02-01')),
            ('Asia', 'Japan', pd.Timestamp('2023-01-01')),
            ('Asia', 'Japan', pd.Timestamp('2023-02-01')),
        ]

        index = pd.MultiIndex.from_tuples(index_tuples, names=['region', 'country', 'date'])
        data = pd.DataFrame({'PIB': [100, 101, 110, 111, 120, 121]}, index=index)

        result = compare_and_detect_delays(
            new_data=data,
            existing_data=None,
            download_date='2023-03-15'
        )

        # 3 entités × 1 colonne = 3 observations
        assert len(result) == 3

    def test_delay_calculation_precision(self):
        """Test de précision du calcul des délais."""
        # Utilisation de 2 observations pour permettre la détection de fréquence
        dates = pd.date_range('2023-01-01', periods=2, freq='MS')
        data = pd.DataFrame({'PIB': [100, 101]}, index=dates)

        # Download date exactement 45 jours après la fin du mois de février
        # Février finit le 28, donc 45 jours après = 14 avril
        download_date = datetime(2023, 4, 14)

        result = compare_and_detect_delays(
            new_data=data,
            existing_data=None,
            download_date=download_date,
            reference_point='end',
            delay_unit='day'
        )

        # La dernière observation est février 2023
        # La période de février se termine le 1er mars (exclusif)
        # Donc le délai devrait être de 44 jours (du 1er mars au 14 avril)
        assert result['release_delay'].iloc[0] in [43, 44, 45]

    def test_index_preservation_after_processing(self, panel_data):
        """Test que l'index est préservé correctement après traitement."""
        result = compare_and_detect_delays(
            new_data=panel_data,
            existing_data=None,
            download_date='2023-07-15'
        )

        # Vérifier que l'index contient les bonnes informations
        assert isinstance(result.index, pd.MultiIndex)
        # Le dernier niveau devrait être l'indicateur
        indicator_level = result.index.get_level_values(-1)
        assert set(indicator_level) == {'PIB', 'inflation'}
