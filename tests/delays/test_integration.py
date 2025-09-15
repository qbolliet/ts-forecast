"""Tests d'intégration pour le module delays."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from tsforecast.delays.database_schema import create_database_manager
from tsforecast.delays.data_manager import ReleaseDataManager
from tsforecast.delays.delay_calculator import ReleaseDelayCalculator
from tsforecast.delays.transformers import ReleaseDelayTransformer


@pytest.fixture
def temp_db_manager():
    """Crée un gestionnaire de base de données temporaire pour l'intégration."""
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
def historical_data():
    """Crée des données historiques simulées."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-10-31', freq='MS')  # Monthly

    data = {
        'date': dates,
        'PIB': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),  # Trend with noise
        'inflation': 2.5 + np.random.randn(len(dates)) * 0.3,
        'unemployment': 8.0 + np.random.randn(len(dates)) * 0.5
    }

    # Simulation de quelques valeurs manquantes dans les données historiques
    df = pd.DataFrame(data)
    missing_indices = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)

    for idx in missing_indices:
        col = np.random.choice(['PIB', 'inflation', 'unemployment'])
        df.loc[idx, col] = np.nan

    return df


@pytest.fixture
def recent_data():
    """Crée des données récentes avec certaines observations qui étaient manquantes."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='MS')  # Inclut données plus récentes

    data = {
        'date': dates,
        'PIB': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'inflation': 2.5 + np.random.randn(len(dates)) * 0.3,
        'unemployment': 8.0 + np.random.randn(len(dates)) * 0.5
    }

    return pd.DataFrame(data)


@pytest.fixture
def panel_historical_data():
    """Crée des données de panel historiques."""
    np.random.seed(42)
    countries = ['France', 'Germany', 'Italy']
    dates = pd.date_range('2020-01-01', '2023-10-31', freq='MS')

    data = []
    for country in countries:
        for date in dates:
            data.append({
                'country': country,
                'date': date,
                'PIB': 100 + np.random.randn() * 2,
                'inflation': 2.5 + np.random.randn() * 0.5
            })

    df = pd.DataFrame(data)

    # Simulation de valeurs manquantes
    missing_indices = np.random.choice(len(df), size=int(len(df) * 0.03), replace=False)
    for idx in missing_indices:
        col = np.random.choice(['PIB', 'inflation'])
        df.loc[idx, col] = np.nan

    return df


@pytest.fixture
def panel_recent_data():
    """Crée des données de panel récentes."""
    np.random.seed(42)
    countries = ['France', 'Germany', 'Italy']
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='MS')

    data = []
    for country in countries:
        for date in dates:
            data.append({
                'country': country,
                'date': date,
                'PIB': 100 + np.random.randn() * 2,
                'inflation': 2.5 + np.random.randn() * 0.5
            })

    return pd.DataFrame(data)


class TestFullWorkflowTimeSeries:
    """Tests d'intégration pour le workflow complet avec séries temporelles."""

    def test_complete_workflow_time_series(self, temp_db_manager, historical_data, recent_data):
        """Test du workflow complet : stockage -> calcul -> transformation."""

        # 1. Stockage des délais de publication
        data_manager = ReleaseDataManager(
            db_manager=temp_db_manager,
            time_col='date'
        )

        # Simulation de plusieurs téléchargements avec délais
        download_dates = [
            datetime(2023, 11, 15),  # Données octobre disponibles mi-novembre
            datetime(2023, 12, 18),  # Données novembre disponibles mi-décembre
        ]

        total_delays = 0
        for download_date in download_dates:
            result = data_manager.compare_and_store_delays(
                recent_data,
                historical_data,
                download_date=download_date,
                source_name=f'Source_{download_date.strftime("%Y%m%d")}'
            )
            total_delays += result['delays_stored']

        assert total_delays > 0

        # 2. Calcul des délais médians
        calculator = ReleaseDelayCalculator(
            db_manager=temp_db_manager,
            min_observations=2  # Baissé pour les tests
        )

        median_delays = calculator.calculate_median_delays()
        assert len(median_delays) > 0
        assert all(delay > 0 for delay in median_delays.values())

        # Mise à jour de la table des statistiques
        stats_result = calculator.update_delay_statistics_table()
        assert stats_result['records_created'] > 0

        # 3. Application des transformations
        transformer = ReleaseDelayTransformer(
            delays_dict=median_delays,
            mode='mask',
            prediction_date='2023-12-01'
        )

        X_transformed = transformer.fit_transform(recent_data)

        # Vérifications
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == recent_data.shape

        # En mode mask, certaines observations récentes devraient être masquées
        original_nan_count = recent_data[list(median_delays.keys())].isna().sum().sum()
        transformed_nan_count = X_transformed[list(median_delays.keys())].isna().sum().sum()
        assert transformed_nan_count >= original_nan_count

        # 4. Test de l'inverse transform
        X_restored = transformer.inverse_transform(X_transformed)
        assert X_restored.shape == recent_data.shape

        # 5. Test avec mode shift
        transformer_shift = ReleaseDelayTransformer(
            delays_dict=median_delays,
            mode='shift',
            prediction_date='2023-12-01'
        )

        X_shifted = transformer_shift.fit_transform(recent_data)
        assert X_shifted.shape == recent_data.shape

    def test_workflow_with_calculator_transformer(self, temp_db_manager, historical_data, recent_data):
        """Test du workflow avec transformer utilisant directement le calculateur."""

        # Stockage initial des délais
        data_manager = ReleaseDataManager(
            db_manager=temp_db_manager,
            time_col='date'
        )

        data_manager.compare_and_store_delays(
            recent_data,
            historical_data,
            download_date=datetime(2023, 11, 15)
        )

        # Transformer avec calculateur intégré
        calculator = ReleaseDelayCalculator(
            db_manager=temp_db_manager,
            min_observations=1
        )

        transformer = ReleaseDelayTransformer(
            delay_calculator=calculator,
            mode='mask',
            prediction_date='2023-12-01'
        )

        # Le transformer devrait récupérer automatiquement les délais
        X_transformed = transformer.fit_transform(recent_data)

        assert isinstance(X_transformed, pd.DataFrame)
        assert hasattr(transformer, 'delays_dict_')
        assert len(transformer.delays_dict_) > 0


class TestFullWorkflowPanel:
    """Tests d'intégration pour les données de panel."""

    def test_complete_workflow_panel(self, temp_db_manager, panel_historical_data, panel_recent_data):
        """Test du workflow complet avec données de panel."""

        # 1. Stockage des délais pour données panel
        data_manager = ReleaseDataManager(
            db_manager=temp_db_manager,
            time_col='date',
            panel_cols=['country']
        )

        result = data_manager.compare_and_store_delays(
            panel_recent_data,
            panel_historical_data,
            download_date=datetime(2023, 11, 15),
            source_name='Panel_Source'
        )

        assert result['delays_stored'] > 0

        # 2. Calcul des délais par entité
        calculator = ReleaseDelayCalculator(
            db_manager=temp_db_manager,
            min_observations=1
        )

        # Délais par indicateur (groupé)
        delays_by_indicator = calculator.calculate_median_delays(group_by_entity=False)
        assert len(delays_by_indicator) > 0

        # Délais par entité et indicateur
        delays_by_entity = calculator.calculate_median_delays(group_by_entity=True)
        assert len(delays_by_entity) > 0

        # 3. Application aux données panel
        transformer = ReleaseDelayTransformer(
            delays_dict=delays_by_entity,
            mode='mask',
            prediction_date='2023-12-01',
            panel_cols=['country']
        )

        X_transformed = transformer.fit_transform(panel_recent_data)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == panel_recent_data.shape
        assert transformer.is_panel_ == True

    def test_panel_statistics_calculation(self, temp_db_manager, panel_historical_data, panel_recent_data):
        """Test du calcul de statistiques pour données panel."""

        # Stockage des délais
        data_manager = ReleaseDataManager(
            db_manager=temp_db_manager,
            time_col='date',
            panel_cols=['country']
        )

        data_manager.compare_and_store_delays(
            panel_recent_data,
            panel_historical_data,
            download_date=datetime(2023, 11, 15)
        )

        # Calcul de statistiques complètes
        calculator = ReleaseDelayCalculator(
            db_manager=temp_db_manager,
            min_observations=1
        )

        stats = calculator.calculate_comprehensive_stats(group_by_entity=True)

        assert isinstance(stats, dict)
        assert len(stats) > 0

        # Vérification du format des clés et des statistiques
        for key, stat_dict in stats.items():
            if isinstance(key, tuple):
                assert len(key) == 2  # (entity, indicator)

            required_stats = ['median', 'mean', 'std', 'min', 'max', 'count']
            for stat in required_stats:
                assert stat in stat_dict


class TestSklearnCompatibility:
    """Tests de compatibilité avec l'écosystème sklearn."""

    def test_pipeline_integration(self, temp_db_manager, historical_data, recent_data):
        """Test d'intégration dans un pipeline sklearn."""

        # Préparation des délais
        data_manager = ReleaseDataManager(
            db_manager=temp_db_manager,
            time_col='date'
        )

        data_manager.compare_and_store_delays(
            recent_data,
            historical_data,
            download_date=datetime(2023, 11, 15)
        )

        calculator = ReleaseDelayCalculator(
            db_manager=temp_db_manager,
            min_observations=1
        )

        delays = calculator.calculate_median_delays()

        # Création d'un pipeline
        pipeline = Pipeline([
            ('delay_transformer', ReleaseDelayTransformer(
                delays_dict=delays,
                mode='mask',
                prediction_date='2023-11-01'
            )),
            ('regressor', RandomForestRegressor(n_estimators=10, random_state=42))
        ])

        # Préparation des données pour l'entraînement
        X = recent_data.drop(['date'], axis=1)
        y = X['PIB'].shift(-1).dropna()  # Prédire PIB du mois suivant
        X = X.iloc[:-1]  # Aligner avec y

        # Fit et prédiction
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        assert len(predictions) == len(y)
        assert not np.isnan(predictions).all()

    def test_multiple_transformations(self, temp_db_manager, historical_data, recent_data):
        """Test de transformations multiples successives."""

        # Stockage des délais
        data_manager = ReleaseDataManager(
            db_manager=temp_db_manager,
            time_col='date'
        )

        data_manager.compare_and_store_delays(
            recent_data,
            historical_data,
            download_date=datetime(2023, 11, 15)
        )

        calculator = ReleaseDelayCalculator(
            db_manager=temp_db_manager,
            min_observations=1
        )

        delays = calculator.calculate_median_delays()

        # Première transformation : mask
        transformer1 = ReleaseDelayTransformer(
            delays_dict=delays,
            mode='mask',
            prediction_date='2023-11-01'
        )

        X1 = transformer1.fit_transform(recent_data)

        # Deuxième transformation : shift avec délais modifiés
        modified_delays = {k: v/2 for k, v in delays.items()}  # Délais réduits de moitié
        transformer2 = ReleaseDelayTransformer(
            delays_dict=modified_delays,
            mode='shift',
            prediction_date='2023-11-01'
        )

        X2 = transformer2.fit_transform(X1)

        # Vérifications
        assert X2.shape == recent_data.shape
        assert not X2.equals(recent_data)

        # Inverse transformations
        X1_restored = transformer2.inverse_transform(X2)
        X_original = transformer1.inverse_transform(X1_restored)

        assert X_original.shape == recent_data.shape


class TestErrorHandlingIntegration:
    """Tests de gestion d'erreurs en intégration."""

    def test_empty_database_workflow(self, temp_db_manager, recent_data):
        """Test avec base de données vide."""

        calculator = ReleaseDelayCalculator(
            db_manager=temp_db_manager,
            min_observations=1
        )

        # Calculateur avec base vide devrait retourner dict vide
        with pytest.warns(UserWarning):
            delays = calculator.calculate_median_delays()
        assert len(delays) == 0

        # Transformer avec dict vide devrait fonctionner mais ne rien faire
        transformer = ReleaseDelayTransformer(delays_dict=delays)
        X_transformed = transformer.fit_transform(recent_data)

        pd.testing.assert_frame_equal(X_transformed, recent_data, check_dtype=False)

    def test_inconsistent_data_workflow(self, temp_db_manager):
        """Test avec données inconsistantes."""

        # Données avec structures différentes
        data1 = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3),
            'PIB': [100, 101, 102]
        })

        data2 = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3),
            'inflation': [2.5, 2.6, 2.7]  # Indicateur différent
        })

        data_manager = ReleaseDataManager(
            db_manager=temp_db_manager,
            time_col='date'
        )

        # Devrait fonctionner mais ne trouver aucune nouvelle observation
        result = data_manager.compare_and_store_delays(
            data2, data1, download_date=datetime(2023, 2, 1)
        )

        assert result['new_observations_count'] == 1  # inflation nouvelle

    def test_database_error_handling(self):
        """Test de gestion d'erreurs de base de données."""

        # Base de données avec chemin invalide
        invalid_connection = "postgresql://invalid:invalid@nonexistent/db"

        # Devrait lever une erreur lors de la création
        with pytest.raises(Exception):
            create_database_manager(invalid_connection, create_tables=True)


class TestPerformanceIntegration:
    """Tests de performance pour les workflows d'intégration."""

    def test_large_dataset_workflow(self, temp_db_manager):
        """Test avec dataset de taille importante."""
        import time

        # Génération d'un dataset plus large
        np.random.seed(42)
        dates = pd.date_range('2000-01-01', '2023-12-31', freq='D')  # 24 ans de données quotidiennes

        large_data = pd.DataFrame({
            'date': dates,
            'indicator1': np.random.randn(len(dates)),
            'indicator2': np.random.randn(len(dates)),
            'indicator3': np.random.randn(len(dates)),
            'indicator4': np.random.randn(len(dates)),
            'indicator5': np.random.randn(len(dates))
        })

        # Simulation de données historiques avec quelques valeurs manquantes
        historical_large = large_data.copy()
        missing_mask = np.random.random(large_data.shape) < 0.01  # 1% de valeurs manquantes
        historical_large[missing_mask] = np.nan

        # Test de performance du workflow
        start_time = time.time()

        data_manager = ReleaseDataManager(
            db_manager=temp_db_manager,
            time_col='date'
        )

        result = data_manager.compare_and_store_delays(
            large_data,
            historical_large,
            download_date=datetime(2023, 12, 31)
        )

        workflow_time = time.time() - start_time

        # Vérifications
        assert result['delays_stored'] > 0
        assert workflow_time < 60  # Devrait prendre moins d'une minute

        # Test de performance du calcul de délais
        start_time = time.time()

        calculator = ReleaseDelayCalculator(
            db_manager=temp_db_manager,
            min_observations=10
        )

        delays = calculator.calculate_median_delays()
        calc_time = time.time() - start_time

        assert len(delays) > 0
        assert calc_time < 30  # Devrait prendre moins de 30 secondes

    def test_cache_performance(self, temp_db_manager, historical_data, recent_data):
        """Test de performance du cache."""

        # Stockage initial
        data_manager = ReleaseDataManager(
            db_manager=temp_db_manager,
            time_col='date'
        )

        data_manager.compare_and_store_delays(
            recent_data,
            historical_data,
            download_date=datetime(2023, 11, 15)
        )

        calculator = ReleaseDelayCalculator(
            db_manager=temp_db_manager,
            cache_results=True
        )

        # Premier calcul (remplissage du cache)
        import time
        start_time = time.time()
        delays1 = calculator.calculate_median_delays()
        first_calc_time = time.time() - start_time

        # Deuxième calcul (utilisation du cache)
        start_time = time.time()
        delays2 = calculator.calculate_median_delays()
        second_calc_time = time.time() - start_time

        # Vérifications
        assert delays1 == delays2
        assert second_calc_time < first_calc_time  # Le cache devrait être plus rapide
        assert len(calculator._cache) > 0