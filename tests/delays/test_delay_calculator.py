"""Tests pour le module delay_calculator."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from tsforecast.delays.database_schema import (
    create_database_manager, ReleaseDelayRecord, ReleaseDelayStats
)
from tsforecast.delays.delay_calculator import ReleaseDelayCalculator


@pytest.fixture
def temp_db_manager():
    """Crée un gestionnaire de base de données temporaire avec données test."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_file.close()

    connection_string = f"sqlite:///{temp_file.name}"
    db_manager = create_database_manager(connection_string, create_tables=True)

    # Ajout de données test
    _populate_test_data(db_manager)

    yield db_manager

    try:
        os.unlink(temp_file.name)
    except:
        pass


def _populate_test_data(db_manager):
    """Peuple la base de données avec des données test."""
    session = db_manager.get_session()

    # Données test pour PIB France
    pib_france_records = []
    base_date = datetime(2023, 1, 1)

    for i in range(12):  # 12 mois de données
        obs_date = base_date + timedelta(days=i*30)
        download_date = obs_date + timedelta(days=45 + np.random.randint(-5, 6))  # ~45 jours avec variation

        record = ReleaseDelayRecord(
            indicator_name='PIB',
            entity_id='France',
            observation_date=obs_date,
            period_start=obs_date.replace(day=1),
            period_end=(obs_date.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1),
            download_date=download_date,
            release_delay_days=(download_date - (obs_date.replace(day=1) + timedelta(days=32)).replace(day=1) + timedelta(days=1)).days,
            is_period_start_reference=False,
            data_frequency='monthly'
        )
        pib_france_records.append(record)

    # Données test pour inflation France
    inflation_france_records = []
    for i in range(10):  # 10 mois de données
        obs_date = base_date + timedelta(days=i*30)
        download_date = obs_date + timedelta(days=30 + np.random.randint(-3, 4))  # ~30 jours

        record = ReleaseDelayRecord(
            indicator_name='inflation',
            entity_id='France',
            observation_date=obs_date,
            period_start=obs_date.replace(day=1),
            period_end=(obs_date.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1),
            download_date=download_date,
            release_delay_days=(download_date - (obs_date.replace(day=1) + timedelta(days=32)).replace(day=1) + timedelta(days=1)).days,
            is_period_start_reference=False,
            data_frequency='monthly'
        )
        inflation_france_records.append(record)

    # Données test pour PIB Germany
    pib_germany_records = []
    for i in range(8):  # 8 mois de données
        obs_date = base_date + timedelta(days=i*30)
        download_date = obs_date + timedelta(days=38 + np.random.randint(-3, 4))  # ~38 jours

        record = ReleaseDelayRecord(
            indicator_name='PIB',
            entity_id='Germany',
            observation_date=obs_date,
            period_start=obs_date.replace(day=1),
            period_end=(obs_date.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1),
            download_date=download_date,
            release_delay_days=(download_date - (obs_date.replace(day=1) + timedelta(days=32)).replace(day=1) + timedelta(days=1)).days,
            is_period_start_reference=False,
            data_frequency='monthly'
        )
        pib_germany_records.append(record)

    # Ajout à la session
    all_records = pib_france_records + inflation_france_records + pib_germany_records
    session.add_all(all_records)
    session.commit()
    session.close()


class TestReleaseDelayCalculator:
    """Tests pour la classe ReleaseDelayCalculator."""

    def test_initialization(self, temp_db_manager):
        """Test de l'initialisation du calculateur."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager)

        assert calculator.db_manager_ == temp_db_manager
        assert calculator.default_reference_point_ == 'end'
        assert calculator.min_observations_ == 5
        assert calculator.cache_results_ == True
        assert isinstance(calculator._cache, dict)

    def test_initialization_with_params(self, temp_db_manager):
        """Test de l'initialisation avec paramètres personnalisés."""
        calculator = ReleaseDelayCalculator(
            db_manager=temp_db_manager,
            default_reference_point='start',
            min_observations=10,
            cache_results=False
        )

        assert calculator.default_reference_point_ == 'start'
        assert calculator.min_observations_ == 10
        assert calculator.cache_results_ == False

    def test_invalid_reference_point(self, temp_db_manager):
        """Test avec point de référence invalide."""
        with pytest.raises(ValueError, match="default_reference_point must be"):
            ReleaseDelayCalculator(
                db_manager=temp_db_manager,
                default_reference_point='middle'
            )

    def test_invalid_min_observations(self, temp_db_manager):
        """Test avec nombre minimum d'observations invalide."""
        with pytest.raises(ValueError, match="min_observations must be at least 1"):
            ReleaseDelayCalculator(
                db_manager=temp_db_manager,
                min_observations=0
            )

    def test_calculate_median_delays_by_indicator(self, temp_db_manager):
        """Test du calcul des délais médians par indicateur."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager, min_observations=5)

        delays = calculator.calculate_median_delays(group_by_entity=False)

        assert isinstance(delays, dict)
        assert 'PIB' in delays
        assert 'inflation' in delays

        # Vérification que les valeurs sont raisonnables
        assert 30 < delays['PIB'] < 60  # Autour de 45 jours
        assert 20 < delays['inflation'] < 40  # Autour de 30 jours

    def test_calculate_median_delays_by_entity(self, temp_db_manager):
        """Test du calcul des délais médians par entité."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager, min_observations=5)

        delays = calculator.calculate_median_delays(group_by_entity=True)

        assert isinstance(delays, dict)
        assert ('France', 'PIB') in delays
        assert ('France', 'inflation') in delays
        assert ('Germany', 'PIB') in delays

        # Vérification des valeurs
        france_pib = delays[('France', 'PIB')]
        germany_pib = delays[('Germany', 'PIB')]

        assert 30 < france_pib < 60
        assert 30 < germany_pib < 50

    def test_calculate_median_delays_with_filters(self, temp_db_manager):
        """Test avec filtres."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager)

        # Filtre par indicateur
        delays = calculator.calculate_median_delays(
            indicators=['PIB'],
            group_by_entity=False
        )

        assert 'PIB' in delays
        assert 'inflation' not in delays

        # Filtre par entité
        delays = calculator.calculate_median_delays(
            entities=['France'],
            group_by_entity=True
        )

        france_keys = [k for k in delays.keys() if isinstance(k, tuple) and k[0] == 'France']
        assert len(france_keys) > 0

        germany_keys = [k for k in delays.keys() if isinstance(k, tuple) and k[0] == 'Germany']
        assert len(germany_keys) == 0

    def test_calculate_median_delays_with_date_range(self, temp_db_manager):
        """Test avec plage de dates."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager)

        # Restriction à une période spécifique
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 6, 1)

        delays = calculator.calculate_median_delays(
            start_date=start_date,
            end_date=end_date,
            group_by_entity=False
        )

        assert isinstance(delays, dict)
        assert len(delays) > 0

    def test_calculate_comprehensive_stats(self, temp_db_manager):
        """Test du calcul des statistiques complètes."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager, min_observations=5)

        stats = calculator.calculate_comprehensive_stats(group_by_entity=False)

        assert isinstance(stats, dict)
        assert 'PIB' in stats

        pib_stats = stats['PIB']
        required_keys = ['median', 'mean', 'std', 'min', 'max', 'count', 'percentile_25', 'percentile_75']

        for key in required_keys:
            assert key in pib_stats
            assert isinstance(pib_stats[key], (int, float))

        # Vérifications de cohérence
        assert pib_stats['min'] <= pib_stats['median'] <= pib_stats['max']
        assert pib_stats['percentile_25'] <= pib_stats['median'] <= pib_stats['percentile_75']
        assert pib_stats['count'] > 0

    def test_calculate_comprehensive_stats_with_entity(self, temp_db_manager):
        """Test des statistiques complètes par entité."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager, min_observations=5)

        stats = calculator.calculate_comprehensive_stats(group_by_entity=True)

        assert isinstance(stats, dict)
        assert ('France', 'PIB') in stats

        france_pib_stats = stats[('France', 'PIB')]
        assert 'median' in france_pib_stats
        assert 'count' in france_pib_stats

    def test_update_delay_statistics_table(self, temp_db_manager):
        """Test de mise à jour de la table des statistiques."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager, min_observations=5)

        result = calculator.update_delay_statistics_table()

        assert isinstance(result, dict)
        assert 'records_updated' in result
        assert 'records_created' in result
        assert 'total_indicators' in result
        assert 'processing_time_seconds' in result

        # Au moins quelques enregistrements devraient être créés
        assert result['records_created'] > 0
        assert result['total_indicators'] > 0

        # Vérification en base de données
        session = temp_db_manager.get_session()
        stats_count = session.query(ReleaseDelayStats).count()
        assert stats_count > 0
        session.close()

    def test_get_delays_from_stats_table(self, temp_db_manager):
        """Test de récupération depuis la table des statistiques."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager, min_observations=5)

        # Première mise à jour de la table des stats
        calculator.update_delay_statistics_table()

        # Récupération depuis la table des stats
        delays = calculator.get_delays_from_stats_table(group_by_entity=False)

        assert isinstance(delays, dict)
        assert len(delays) > 0

        # Comparaison avec calcul direct
        direct_delays = calculator.calculate_median_delays(group_by_entity=False)

        # Les résultats devraient être similaires
        for indicator in delays:
            if indicator in direct_delays:
                # Tolérance de 1 jour de différence
                assert abs(delays[indicator] - direct_delays[indicator]) <= 1.0

    def test_get_delays_from_stats_table_with_entity(self, temp_db_manager):
        """Test de récupération par entité depuis la table des stats."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager, min_observations=5)

        calculator.update_delay_statistics_table()
        delays = calculator.get_delays_from_stats_table(group_by_entity=True)

        assert isinstance(delays, dict)
        assert len(delays) > 0

        # Vérification du format des clés
        for key in delays:
            assert isinstance(key, (str, tuple))

    def test_cache_functionality(self, temp_db_manager):
        """Test de la fonctionnalité de cache."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager, cache_results=True)

        # Premier appel (devrait remplir le cache)
        delays1 = calculator.calculate_median_delays()

        # Vérification que le cache contient quelque chose
        assert len(calculator._cache) > 0

        # Deuxième appel identique (devrait utiliser le cache)
        delays2 = calculator.calculate_median_delays()

        # Les résultats devraient être identiques
        assert delays1 == delays2

        # Test du nettoyage du cache
        calculator.clear_cache()
        assert len(calculator._cache) == 0

    def test_cache_info(self, temp_db_manager):
        """Test des informations de cache."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager, cache_results=True)

        # Cache vide initialement
        info = calculator.get_cache_info()
        assert info['cache_enabled'] == True
        assert info['cached_entries'] == 0

        # Après un calcul
        calculator.calculate_median_delays()
        info = calculator.get_cache_info()
        assert info['cached_entries'] > 0
        assert len(info['cache_keys']) > 0

    def test_no_cache(self, temp_db_manager):
        """Test sans cache."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager, cache_results=False)

        delays1 = calculator.calculate_median_delays()
        delays2 = calculator.calculate_median_delays()

        # Le cache ne devrait pas être utilisé
        assert len(calculator._cache) == 0

        info = calculator.get_cache_info()
        assert info['cache_enabled'] == False

    def test_insufficient_data(self, temp_db_manager):
        """Test avec données insuffisantes."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager, min_observations=100)

        # Avec un minimum très élevé, aucun résultat ne devrait être retourné
        delays = calculator.calculate_median_delays()
        assert len(delays) == 0

    def test_empty_database(self):
        """Test avec base de données vide."""
        # Création d'une base temporaire vide
        temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_file.close()

        try:
            connection_string = f"sqlite:///{temp_file.name}"
            db_manager = create_database_manager(connection_string, create_tables=True)

            calculator = ReleaseDelayCalculator(db_manager=db_manager)

            with pytest.warns(UserWarning, match="Aucune donnée de délai trouvée"):
                delays = calculator.calculate_median_delays()

            assert len(delays) == 0

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    def test_reference_point_consistency(self, temp_db_manager):
        """Test de cohérence du point de référence."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager)

        # Test avec différents points de référence
        delays_end = calculator.calculate_median_delays(reference_point='end')
        delays_start = calculator.calculate_median_delays(reference_point='start')

        # Les délais devraient être différents selon le point de référence
        # (mais on ne peut pas garantir lesquels sont plus grands car cela dépend des données test)
        assert isinstance(delays_end, dict)
        assert isinstance(delays_start, dict)

    def test_error_handling(self, temp_db_manager):
        """Test de gestion des erreurs."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager)

        # Test avec dates invalides
        with pytest.raises(Exception):
            calculator.calculate_median_delays(
                start_date="not a date",
                end_date=datetime.utcnow()
            )


class TestPrivateMethods:
    """Tests pour les méthodes privées."""

    def test_fetch_delay_data(self, temp_db_manager):
        """Test de récupération des données."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager)

        data = calculator._fetch_delay_data('end')

        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert 'indicator_name' in data.columns
        assert 'release_delay_days' in data.columns

    def test_compute_median_delays(self, temp_db_manager):
        """Test du calcul des délais médians."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager)

        # Récupération des données brutes
        data = calculator._fetch_delay_data('end')

        # Calcul des délais par indicateur
        delays = calculator._compute_median_delays(data, group_by_entity=False)

        assert isinstance(delays, dict)
        assert len(delays) > 0

        for delay in delays.values():
            assert isinstance(delay, float)
            assert delay > 0

    def test_generate_cache_key(self, temp_db_manager):
        """Test de génération de clé de cache."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager)

        # Test avec différents arguments
        key1 = calculator._generate_cache_key('end', False, ['PIB'], None, None, None, None)
        key2 = calculator._generate_cache_key('start', False, ['PIB'], None, None, None, None)
        key3 = calculator._generate_cache_key('end', True, ['PIB'], None, None, None, None)

        # Les clés devraient être différentes
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

        # Les clés devraient être des strings
        assert isinstance(key1, str)
        assert isinstance(key2, str)
        assert isinstance(key3, str)

    def test_should_update_stats(self, temp_db_manager):
        """Test de la logique de mise à jour des statistiques."""
        calculator = ReleaseDelayCalculator(db_manager=temp_db_manager)

        # Création d'un enregistrement de stats récent
        session = temp_db_manager.get_session()

        existing_stats = ReleaseDelayStats(
            indicator_name='PIB',
            entity_id='France',
            data_frequency='monthly',
            is_period_start_reference=False,
            median_delay_days=45.0,
            mean_delay_days=47.0,
            std_delay_days=5.0,
            min_delay_days=40.0,
            max_delay_days=50.0,
            count_observations=10,
            last_updated=datetime.utcnow()
        )

        session.add(existing_stats)
        session.commit()

        # Nouvelles statistiques similaires
        new_stats = {
            'median_delay_days': 46.0,
            'mean_delay_days': 48.0,
            'std_delay_days': 5.5,
            'min_delay_days': 41.0,
            'max_delay_days': 51.0,
            'count_observations': 11
        }

        # Ne devrait pas nécessiter de mise à jour (changement < 5%)
        should_update = calculator._should_update_stats(existing_stats, new_stats)
        assert should_update == False

        # Test avec changement significatif
        new_stats['median_delay_days'] = 50.0  # Changement > 5%
        should_update = calculator._should_update_stats(existing_stats, new_stats)
        assert should_update == True

        # Test avec enregistrement ancien
        existing_stats.last_updated = datetime.utcnow() - timedelta(days=10)
        should_update = calculator._should_update_stats(existing_stats, new_stats)
        assert should_update == True

        session.close()