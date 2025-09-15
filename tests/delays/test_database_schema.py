"""Tests pour le module database_schema."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os

from tsforecast.delays.database_schema import (
    DatabaseManager, ReleaseDelayRecord, ReleaseDelayStats, DataSourceInfo,
    create_database_manager, get_sample_connection_string
)


@pytest.fixture
def temp_db_manager():
    """Crée un gestionnaire de base de données temporaire pour les tests."""
    # Utilisation d'une base SQLite temporaire pour les tests
    temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_file.close()

    connection_string = f"sqlite:///{temp_file.name}"
    db_manager = create_database_manager(connection_string, create_tables=True)

    yield db_manager

    # Nettoyage
    try:
        os.unlink(temp_file.name)
    except:
        pass


@pytest.fixture
def sample_delay_record():
    """Crée un enregistrement de délai exemple pour les tests."""
    return {
        'indicator_name': 'PIB',
        'entity_id': 'France',
        'observation_date': datetime(2023, 10, 1),
        'period_start': datetime(2023, 10, 1),
        'period_end': datetime(2023, 10, 31),
        'download_date': datetime(2023, 11, 15),
        'release_delay_days': 45.0,
        'is_period_start_reference': False,
        'data_frequency': 'monthly'
    }


class TestDatabaseManager:
    """Tests pour la classe DatabaseManager."""

    def test_initialization(self, temp_db_manager):
        """Test de l'initialisation du gestionnaire."""
        assert temp_db_manager is not None
        assert temp_db_manager.engine is not None
        assert temp_db_manager.SessionLocal is not None

    def test_create_tables(self, temp_db_manager):
        """Test de la création des tables."""
        # Les tables devraient déjà être créées par le fixture
        info = temp_db_manager.get_table_info()

        assert 'release_delay_records' in info
        assert 'release_delay_stats' in info
        assert 'data_source_info' in info
        assert info['release_delay_records']['count'] == 0

    def test_session_management(self, temp_db_manager):
        """Test de la gestion des sessions."""
        session = temp_db_manager.get_session()

        assert session is not None

        # Test d'une opération simple
        count = session.query(ReleaseDelayRecord).count()
        assert count == 0

        temp_db_manager.close_session(session)

    def test_get_table_info(self, temp_db_manager):
        """Test de récupération des informations sur les tables."""
        info = temp_db_manager.get_table_info()

        assert isinstance(info, dict)
        assert len(info) == 3

        for table_info in info.values():
            assert 'count' in table_info
            assert 'table_name' in table_info


class TestReleaseDelayRecord:
    """Tests pour le modèle ReleaseDelayRecord."""

    def test_create_record(self, temp_db_manager, sample_delay_record):
        """Test de création d'un enregistrement."""
        session = temp_db_manager.get_session()

        record = ReleaseDelayRecord(**sample_delay_record)
        session.add(record)
        session.commit()

        # Vérification
        saved_record = session.query(ReleaseDelayRecord).first()
        assert saved_record is not None
        assert saved_record.indicator_name == 'PIB'
        assert saved_record.entity_id == 'France'
        assert saved_record.release_delay_days == 45.0

        session.close()

    def test_unique_constraint(self, temp_db_manager, sample_delay_record):
        """Test de la contrainte d'unicité."""
        session = temp_db_manager.get_session()

        # Premier enregistrement
        record1 = ReleaseDelayRecord(**sample_delay_record)
        session.add(record1)
        session.commit()

        # Deuxième enregistrement identique (devrait échouer)
        record2 = ReleaseDelayRecord(**sample_delay_record)
        session.add(record2)

        with pytest.raises(Exception):  # Violation de contrainte
            session.commit()

        session.rollback()
        session.close()

    def test_query_by_indicator(self, temp_db_manager, sample_delay_record):
        """Test de requête par indicateur."""
        session = temp_db_manager.get_session()

        # Ajout de plusieurs enregistrements
        record1 = ReleaseDelayRecord(**sample_delay_record)

        record2_data = sample_delay_record.copy()
        record2_data['indicator_name'] = 'inflation'
        record2_data['observation_date'] = datetime(2023, 11, 1)
        record2 = ReleaseDelayRecord(**record2_data)

        session.add_all([record1, record2])
        session.commit()

        # Requête par indicateur
        pib_records = session.query(ReleaseDelayRecord)\
                           .filter(ReleaseDelayRecord.indicator_name == 'PIB')\
                           .all()

        assert len(pib_records) == 1
        assert pib_records[0].indicator_name == 'PIB'

        session.close()


class TestReleaseDelayStats:
    """Tests pour le modèle ReleaseDelayStats."""

    def test_create_stats_record(self, temp_db_manager):
        """Test de création d'un enregistrement de statistiques."""
        session = temp_db_manager.get_session()

        stats = ReleaseDelayStats(
            indicator_name='PIB',
            entity_id='France',
            data_frequency='monthly',
            is_period_start_reference=False,
            median_delay_days=45.0,
            mean_delay_days=47.5,
            std_delay_days=12.3,
            min_delay_days=30.0,
            max_delay_days=65.0,
            count_observations=25
        )

        session.add(stats)
        session.commit()

        # Vérification
        saved_stats = session.query(ReleaseDelayStats).first()
        assert saved_stats is not None
        assert saved_stats.median_delay_days == 45.0
        assert saved_stats.count_observations == 25

        session.close()


class TestDataSourceInfo:
    """Tests pour le modèle DataSourceInfo."""

    def test_create_source_info(self, temp_db_manager):
        """Test de création d'un enregistrement de source."""
        session = temp_db_manager.get_session()

        source_info = DataSourceInfo(
            source_name='OCDE',
            source_url='https://stats.oecd.org/',
            download_date=datetime.utcnow(),
            file_hash='abc123',
            file_size_bytes=1024,
            record_count=100,
            processing_status='completed'
        )

        session.add(source_info)
        session.commit()

        # Vérification
        saved_info = session.query(DataSourceInfo).first()
        assert saved_info is not None
        assert saved_info.source_name == 'OCDE'
        assert saved_info.processing_status == 'completed'

        session.close()


class TestFactoryFunctions:
    """Tests pour les fonctions factory."""

    def test_create_database_manager(self):
        """Test de la fonction create_database_manager."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_file.close()

        try:
            connection_string = f"sqlite:///{temp_file.name}"
            db_manager = create_database_manager(connection_string)

            assert db_manager is not None
            assert isinstance(db_manager, DatabaseManager)

            # Test avec create_tables=False
            db_manager2 = create_database_manager(connection_string, create_tables=False)
            assert db_manager2 is not None

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    def test_get_sample_connection_string(self):
        """Test de la fonction get_sample_connection_string."""
        sample = get_sample_connection_string()

        assert isinstance(sample, str)
        assert 'postgresql://' in sample
        assert 'username:password' in sample


class TestIntegration:
    """Tests d'intégration pour le schéma de base de données."""

    def test_complete_workflow(self, temp_db_manager):
        """Test d'un workflow complet d'utilisation."""
        session = temp_db_manager.get_session()

        # 1. Ajout d'une source de données
        source = DataSourceInfo(
            source_name='Test Source',
            download_date=datetime.utcnow(),
            processing_status='completed',
            record_count=2
        )
        session.add(source)
        session.commit()

        # 2. Ajout d'enregistrements de délais
        records = [
            ReleaseDelayRecord(
                indicator_name='PIB',
                entity_id='France',
                observation_date=datetime(2023, 10, 1),
                period_start=datetime(2023, 10, 1),
                period_end=datetime(2023, 10, 31),
                download_date=datetime(2023, 11, 15),
                release_delay_days=45.0,
                is_period_start_reference=False,
                data_frequency='monthly'
            ),
            ReleaseDelayRecord(
                indicator_name='PIB',
                entity_id='France',
                observation_date=datetime(2023, 11, 1),
                period_start=datetime(2023, 11, 1),
                period_end=datetime(2023, 11, 30),
                download_date=datetime(2023, 12, 18),
                release_delay_days=48.0,
                is_period_start_reference=False,
                data_frequency='monthly'
            )
        ]

        session.add_all(records)
        session.commit()

        # 3. Calcul et ajout de statistiques
        delays = [r.release_delay_days for r in records]
        stats = ReleaseDelayStats(
            indicator_name='PIB',
            entity_id='France',
            data_frequency='monthly',
            is_period_start_reference=False,
            median_delay_days=np.median(delays),
            mean_delay_days=np.mean(delays),
            std_delay_days=np.std(delays),
            min_delay_days=np.min(delays),
            max_delay_days=np.max(delays),
            count_observations=len(delays)
        )

        session.add(stats)
        session.commit()

        # 4. Vérifications
        assert session.query(DataSourceInfo).count() == 1
        assert session.query(ReleaseDelayRecord).count() == 2
        assert session.query(ReleaseDelayStats).count() == 1

        # Vérification des statistiques calculées
        saved_stats = session.query(ReleaseDelayStats).first()
        assert saved_stats.median_delay_days == 46.5  # Médiane de [45, 48]
        assert saved_stats.count_observations == 2

        session.close()

    def test_error_handling(self, temp_db_manager):
        """Test de la gestion des erreurs."""
        session = temp_db_manager.get_session()

        # Test avec données invalides
        invalid_record = ReleaseDelayRecord(
            indicator_name=None,  # Champ requis
            observation_date=datetime.utcnow(),
            download_date=datetime.utcnow(),
            release_delay_days=30.0
        )

        session.add(invalid_record)

        with pytest.raises(Exception):
            session.commit()

        session.rollback()
        session.close()