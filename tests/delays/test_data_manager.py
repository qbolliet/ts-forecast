"""Tests pour le module data_manager."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from tsforecast.delays.database_schema import create_database_manager
from tsforecast.delays.data_manager import ReleaseDataManager
from tsforecast.frequency.detector import FrequencyDetector


@pytest.fixture
def temp_db_manager():
    """Crée un gestionnaire de base de données temporaire."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_file.close()

    connection_string = f"sqlite:///{temp_file.name}"
    db_manager = create_database_manager(connection_string, create_tables=True)

    yield db_manager

    try:
        os.unlink(temp_file.name)
    except:
        pass


@pytest.fixture
def sample_time_series_data():
    """Crée des données de série temporelle exemple."""
    dates = pd.date_range('2023-01-01', '2023-12-01', freq='MS')  # Monthly
    data = {
        'date': dates,
        'PIB': np.random.randn(len(dates)) + 100,
        'inflation': np.random.randn(len(dates)) + 2.5,
        'unemployment': np.random.randn(len(dates)) + 5.0
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_panel_data():
    """Crée des données de panel exemple."""
    countries = ['France', 'Germany', 'Italy']
    dates = pd.date_range('2023-01-01', '2023-06-01', freq='MS')  # 6 months

    data = []
    for country in countries:
        for date in dates:
            data.append({
                'country': country,
                'date': date,
                'PIB': np.random.randn() + 100,
                'inflation': np.random.randn() + 2.5
            })

    return pd.DataFrame(data)


class TestReleaseDataManager:
    """Tests pour la classe ReleaseDataManager."""

    def test_initialization(self, temp_db_manager):
        """Test de l'initialisation du gestionnaire."""
        manager = ReleaseDataManager(
            db_manager=temp_db_manager,
            time_col='date',
            panel_cols=None
        )

        assert manager.db_manager_ == temp_db_manager
        assert manager.time_col_ == 'date'
        assert manager.panel_cols_ == []
        assert manager.default_reference_point_ == 'end'
        assert isinstance(manager.frequency_detector_, FrequencyDetector)

    def test_initialization_with_panel(self, temp_db_manager):
        """Test de l'initialisation avec données panel."""
        manager = ReleaseDataManager(
            db_manager=temp_db_manager,
            time_col='date',
            panel_cols=['country', 'indicator'],
            default_reference_point='start'
        )

        assert manager.panel_cols_ == ['country', 'indicator']
        assert manager.default_reference_point_ == 'start'

    def test_invalid_reference_point(self, temp_db_manager):
        """Test avec point de référence invalide."""
        with pytest.raises(ValueError, match="default_reference_point must be"):
            ReleaseDataManager(
                db_manager=temp_db_manager,
                default_reference_point='middle'
            )

    def test_validate_input_data_valid(self, temp_db_manager, sample_time_series_data):
        """Test de validation des données valides."""
        manager = ReleaseDataManager(db_manager=temp_db_manager, time_col='date')

        validated = manager._validate_input_data(sample_time_series_data)

        assert isinstance(validated, pd.DataFrame)
        assert validated.shape == sample_time_series_data.shape
        assert pd.api.types.is_datetime64_any_dtype(validated['date'])

    def test_validate_input_data_invalid(self, temp_db_manager):
        """Test de validation avec données invalides."""
        manager = ReleaseDataManager(db_manager=temp_db_manager, time_col='date')

        # Test avec données non-DataFrame
        with pytest.raises(ValueError, match="Les données doivent être un pandas DataFrame"):
            manager._validate_input_data("not a dataframe")

        # Test avec colonne temporelle manquante
        invalid_df = pd.DataFrame({'value': [1, 2, 3]})
        with pytest.raises(ValueError, match="Colonne temporelle 'date' non trouvée"):
            manager._validate_input_data(invalid_df)

    def test_validate_input_data_missing_panel_cols(self, temp_db_manager):
        """Test avec colonnes panel manquantes."""
        manager = ReleaseDataManager(
            db_manager=temp_db_manager,
            time_col='date',
            panel_cols=['country']
        )

        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3),
            'value': [1, 2, 3]
        })

        with pytest.raises(ValueError, match="Colonnes panel manquantes"):
            manager._validate_input_data(df)

    def test_parse_download_date(self, temp_db_manager):
        """Test du parsing de la date de téléchargement."""
        manager = ReleaseDataManager(db_manager=temp_db_manager)

        # Test avec string
        date_str = '2023-12-01'
        parsed = manager._parse_download_date(date_str)
        assert isinstance(parsed, datetime)
        assert parsed.year == 2023
        assert parsed.month == 12

        # Test avec datetime
        date_dt = datetime(2023, 12, 1)
        parsed = manager._parse_download_date(date_dt)
        assert parsed == date_dt

        # Test avec format invalide
        with pytest.raises(ValueError, match="Format de date invalide"):
            manager._parse_download_date("invalid date")

        # Test avec type invalide
        with pytest.raises(ValueError, match="download_date doit être"):
            manager._parse_download_date(123)

    def test_identify_new_observations_time_series(self, temp_db_manager):
        """Test d'identification des nouvelles observations pour série temporelle."""
        manager = ReleaseDataManager(db_manager=temp_db_manager, time_col='date')

        # Données existantes
        existing = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3, freq='D'),
            'PIB': [100, 101, np.nan],  # Valeur manquante le 3ème jour
            'inflation': [2.5, 2.6, 2.7]
        })

        # Nouvelles données avec observation précédemment manquante
        new_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=4, freq='D'),
            'PIB': [100, 101, 102, 103],  # Nouvelle valeur le 3ème jour + nouveau jour
            'inflation': [2.5, 2.6, 2.7, 2.8]
        })

        new_obs = manager._identify_new_observations(new_data, existing)

        assert not new_obs.empty
        # Devrait contenir : PIB du 3ème jour (était NaN) + PIB et inflation du 4ème jour
        assert len(new_obs) == 3

        # Vérification des contenus
        pib_obs = new_obs[new_obs['indicator_name'] == 'PIB']
        assert len(pib_obs) == 2  # 3ème et 4ème jour

    def test_identify_new_observations_panel(self, temp_db_manager):
        """Test d'identification pour données panel."""
        manager = ReleaseDataManager(
            db_manager=temp_db_manager,
            time_col='date',
            panel_cols=['country']
        )

        # Données existantes
        existing = pd.DataFrame({
            'country': ['France', 'France', 'Germany', 'Germany'],
            'date': pd.date_range('2023-01-01', periods=2, freq='MS').tolist() * 2,
            'PIB': [100, 101, 105, np.nan]  # Valeur manquante pour Germany en février
        })

        # Nouvelles données
        new_data = pd.DataFrame({
            'country': ['France', 'France', 'Germany', 'Germany'],
            'date': pd.date_range('2023-01-01', periods=2, freq='MS').tolist() * 2,
            'PIB': [100, 101, 105, 107]  # PIB Germany février maintenant disponible
        })

        new_obs = manager._identify_new_observations(new_data, existing)

        assert not new_obs.empty
        assert len(new_obs) == 1  # Seule la valeur Germany février
        assert new_obs.iloc[0]['country'] == 'Germany'
        assert new_obs.iloc[0]['indicator_name'] == 'PIB'

    def test_calculate_release_delays(self, temp_db_manager):
        """Test du calcul des délais de publication."""
        manager = ReleaseDataManager(db_manager=temp_db_manager, time_col='date')

        new_observations = pd.DataFrame({
            'observation_date': [datetime(2023, 10, 1)],
            'indicator_name': ['PIB'],
            'value': [100.0]
        })

        download_date = datetime(2023, 11, 15)  # 45 jours après début octobre

        delays = manager._calculate_release_delays(
            new_observations, download_date, 'end'
        )

        assert len(delays) == 1
        delay = delays[0]
        assert delay['indicator_name'] == 'PIB'
        assert delay['download_date'] == download_date
        assert delay['is_period_start_reference'] == False
        assert delay['release_delay_days'] == 15.0  # 15 jours après fin octobre (31 oct)

    def test_determine_period_boundaries(self, temp_db_manager):
        """Test de détermination des limites de période."""
        manager = ReleaseDataManager(db_manager=temp_db_manager)

        # Test pour fréquence mensuelle
        obs_date = datetime(2023, 10, 15)  # Milieu d'octobre
        period_info = manager._determine_period_boundaries(obs_date, 'PIB')

        assert period_info['period_start'] == datetime(2023, 10, 1)
        assert period_info['period_end'].month == 10
        assert period_info['period_end'].day == 31

        # Test pour fréquence trimestrielle
        # Forcer la fréquence en modifiant temporairement la méthode
        original_method = manager._get_indicator_frequency
        manager._get_indicator_frequency = lambda x: 'quarterly'

        period_info = manager._determine_period_boundaries(obs_date, 'PIB')

        # Octobre est dans le Q4 (oct-déc)
        assert period_info['period_start'] == datetime(2023, 10, 1)
        assert period_info['period_end'].month == 12
        assert period_info['period_end'].day == 31

        # Restauration de la méthode originale
        manager._get_indicator_frequency = original_method

    def test_compare_and_store_delays_no_new_data(self, temp_db_manager, sample_time_series_data):
        """Test avec aucune nouvelle donnée."""
        manager = ReleaseDataManager(db_manager=temp_db_manager, time_col='date')

        # Mêmes données pour new et existing
        result = manager.compare_and_store_delays(
            sample_time_series_data,
            sample_time_series_data,
            download_date='2023-12-01'
        )

        assert result['new_observations_count'] == 0
        assert result['delays_calculated'] == 0
        assert result['delays_stored'] == 0

    def test_compare_and_store_delays_with_new_data(self, temp_db_manager):
        """Test avec nouvelles données."""
        manager = ReleaseDataManager(db_manager=temp_db_manager, time_col='date')

        # Données existantes
        existing = pd.DataFrame({
            'date': [datetime(2023, 10, 1)],
            'PIB': [np.nan]  # Valeur manquante
        })

        # Nouvelles données avec valeur
        new_data = pd.DataFrame({
            'date': [datetime(2023, 10, 1)],
            'PIB': [100.0]
        })

        result = manager.compare_and_store_delays(
            new_data,
            existing,
            download_date=datetime(2023, 11, 15),
            source_name='Test Source'
        )

        assert result['new_observations_count'] == 1
        assert result['delays_calculated'] == 1
        assert result['delays_stored'] == 1
        assert result['source_stored'] == True

    def test_get_stored_delays_summary_empty(self, temp_db_manager):
        """Test du résumé avec base vide."""
        manager = ReleaseDataManager(db_manager=temp_db_manager)

        summary = manager.get_stored_delays_summary()

        assert summary['record_count'] == 0
        assert 'message' in summary

    def test_get_stored_delays_summary_with_data(self, temp_db_manager):
        """Test du résumé avec données."""
        manager = ReleaseDataManager(db_manager=temp_db_manager, time_col='date')

        # Ajout de quelques enregistrements via compare_and_store_delays
        existing = pd.DataFrame({
            'date': [datetime(2023, 10, 1), datetime(2023, 11, 1)],
            'PIB': [np.nan, np.nan]
        })

        new_data = pd.DataFrame({
            'date': [datetime(2023, 10, 1), datetime(2023, 11, 1)],
            'PIB': [100.0, 101.0]
        })

        manager.compare_and_store_delays(
            new_data, existing, download_date=datetime(2023, 12, 1)
        )

        # Test du résumé
        summary = manager.get_stored_delays_summary(days_back=60)

        assert summary['record_count'] == 2
        assert 'median_delay' in summary
        assert 'mean_delay' in summary
        assert summary['unique_indicators'] == 1

    def test_error_handling_invalid_data(self, temp_db_manager):
        """Test de gestion d'erreurs avec données invalides."""
        manager = ReleaseDataManager(db_manager=temp_db_manager, time_col='date')

        # Test avec données corrompues
        invalid_new = pd.DataFrame({
            'date': ['not a date'],
            'PIB': [100]
        })

        valid_existing = pd.DataFrame({
            'date': [datetime(2023, 10, 1)],
            'PIB': [99]
        })

        with pytest.raises(Exception):  # Devrait lever une erreur de parsing de date
            manager.compare_and_store_delays(
                invalid_new, valid_existing, download_date='2023-12-01'
            )


class TestPrivateMethods:
    """Tests pour les méthodes privées spécifiques."""

    def test_store_source_info(self, temp_db_manager, sample_time_series_data):
        """Test du stockage des informations de source."""
        manager = ReleaseDataManager(db_manager=temp_db_manager)

        result = manager._store_source_info(
            'Test Source',
            sample_time_series_data,
            datetime.utcnow()
        )

        assert result == True

        # Vérification en base
        session = temp_db_manager.get_session()
        from tsforecast.delays.database_schema import DataSourceInfo

        source = session.query(DataSourceInfo).first()
        assert source is not None
        assert source.source_name == 'Test Source'
        assert source.record_count == len(sample_time_series_data)

        session.close()

    def test_get_indicator_frequency_from_db(self, temp_db_manager):
        """Test de récupération de fréquence depuis la base."""
        manager = ReleaseDataManager(db_manager=temp_db_manager)

        # Pas de données en base, devrait retourner None
        freq = manager._get_indicator_frequency('PIB')
        assert freq is None

        # Ajout d'un enregistrement avec fréquence
        session = temp_db_manager.get_session()
        from tsforecast.delays.database_schema import ReleaseDelayRecord

        record = ReleaseDelayRecord(
            indicator_name='PIB',
            observation_date=datetime.utcnow(),
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow(),
            download_date=datetime.utcnow(),
            release_delay_days=30.0,
            is_period_start_reference=False,
            data_frequency='monthly'
        )

        session.add(record)
        session.commit()
        session.close()

        # Maintenant devrait retourner 'monthly'
        freq = manager._get_indicator_frequency('PIB')
        assert freq == 'monthly'