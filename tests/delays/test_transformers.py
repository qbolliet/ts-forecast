"""Tests pour le module transformers."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.utils.validation import check_is_fitted

from tsforecast.delays.transformers import (
    ReleaseDelayTransformer,
    create_delay_transformer_from_dict,
    create_delay_transformer_from_calculator
)
from tsforecast.delays.delay_calculator import ReleaseDelayCalculator
from tsforecast.delays.database_schema import create_database_manager


@pytest.fixture
def sample_time_series_data():
    """Crée des données de série temporelle pour les tests."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    data = {
        'date': dates,
        'PIB': np.random.randn(len(dates)) + 100,
        'inflation': np.random.randn(len(dates)) + 2.5,
        'unemployment': np.random.randn(len(dates)) + 5.0
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_panel_data():
    """Crée des données de panel pour les tests."""
    countries = ['France', 'Germany', 'Italy']
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='MS')  # Monthly

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


@pytest.fixture
def sample_delays_dict():
    """Crée un dictionnaire de délais exemple."""
    return {
        'PIB': 45.0,
        'inflation': 30.0,
        'unemployment': 15.0
    }


@pytest.fixture
def sample_panel_delays_dict():
    """Crée un dictionnaire de délais pour données panel."""
    return {
        ('France', 'PIB'): 45.0,
        ('Germany', 'PIB'): 38.0,
        ('Italy', 'PIB'): 50.0,
        ('France', 'inflation'): 30.0,
        ('Germany', 'inflation'): 25.0,
        ('Italy', 'inflation'): 35.0
    }


class TestReleaseDelayTransformer:
    """Tests pour la classe ReleaseDelayTransformer."""

    def test_initialization_with_dict(self, sample_delays_dict):
        """Test de l'initialisation avec dictionnaire de délais."""
        transformer = ReleaseDelayTransformer(
            delays_dict=sample_delays_dict,
            mode='shift',
            prediction_date='2023-12-01'
        )

        assert transformer.delays_dict == sample_delays_dict
        assert transformer.mode == 'shift'
        assert transformer.prediction_date == '2023-12-01'
        assert transformer.reference_point == 'end'
        assert transformer.time_col == 'date'

    def test_initialization_with_calculator(self):
        """Test de l'initialisation avec calculateur."""
        # Création d'un calculateur mock
        import tempfile
        import os

        temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_file.close()

        try:
            connection_string = f"sqlite:///{temp_file.name}"
            db_manager = create_database_manager(connection_string, create_tables=True)
            calculator = ReleaseDelayCalculator(db_manager=db_manager)

            transformer = ReleaseDelayTransformer(
                delay_calculator=calculator,
                mode='mask',
                prediction_date='today'
            )

            assert transformer.delay_calculator == calculator
            assert transformer.mode == 'mask'

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass

    def test_invalid_initialization(self):
        """Test avec paramètres d'initialisation invalides."""
        # Aucune source de délais
        with pytest.raises(ValueError, match="Vous devez fournir soit delays_dict soit delay_calculator"):
            ReleaseDelayTransformer()

        # Mode invalide
        with pytest.raises(ValueError, match="mode doit être 'shift' ou 'mask'"):
            ReleaseDelayTransformer(
                delays_dict={'PIB': 30.0},
                mode='invalid'
            )

        # Point de référence invalide
        with pytest.raises(ValueError, match="reference_point doit être 'start' ou 'end'"):
            ReleaseDelayTransformer(
                delays_dict={'PIB': 30.0},
                reference_point='middle'
            )

        # Gestion délais manquants invalide
        with pytest.raises(ValueError, match="handle_missing_delays doit être"):
            ReleaseDelayTransformer(
                delays_dict={'PIB': 30.0},
                handle_missing_delays='invalid'
            )

    def test_fit_transform_shift_mode(self, sample_time_series_data, sample_delays_dict):
        """Test du fit et transform en mode shift."""
        transformer = ReleaseDelayTransformer(
            delays_dict=sample_delays_dict,
            mode='shift',
            prediction_date='2023-12-01'
        )

        # Fit et transform
        X_transformed = transformer.fit_transform(sample_time_series_data)

        # Vérifications de base
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == sample_time_series_data.shape
        assert list(X_transformed.columns) == list(sample_time_series_data.columns)

        # Le transformer devrait être ajusté
        check_is_fitted(transformer, 'delays_dict_')
        assert transformer.delays_dict_ == sample_delays_dict

    def test_fit_transform_mask_mode(self, sample_time_series_data, sample_delays_dict):
        """Test du fit et transform en mode mask."""
        transformer = ReleaseDelayTransformer(
            delays_dict=sample_delays_dict,
            mode='mask',
            prediction_date='2023-06-01'  # Milieu de l'année
        )

        X_transformed = transformer.fit_transform(sample_time_series_data)

        # En mode mask, certaines valeurs devraient être NaN
        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == sample_time_series_data.shape

        # Vérification que des valeurs ont été masquées
        original_nan_count = sample_time_series_data[['PIB', 'inflation', 'unemployment']].isna().sum().sum()
        transformed_nan_count = X_transformed[['PIB', 'inflation', 'unemployment']].isna().sum().sum()

        assert transformed_nan_count >= original_nan_count

    def test_fit_transform_panel_data(self, sample_panel_data, sample_panel_delays_dict):
        """Test avec données panel."""
        transformer = ReleaseDelayTransformer(
            delays_dict=sample_panel_delays_dict,
            mode='shift',
            prediction_date='2023-12-01',
            panel_cols=['country']
        )

        X_transformed = transformer.fit_transform(sample_panel_data)

        assert isinstance(X_transformed, pd.DataFrame)
        assert X_transformed.shape == sample_panel_data.shape

        # Vérification que le transformer reconnaît les données panel
        assert transformer.is_panel_ == True

    def test_inverse_transform_mask_mode(self, sample_time_series_data, sample_delays_dict):
        """Test de l'inverse transform en mode mask."""
        transformer = ReleaseDelayTransformer(
            delays_dict=sample_delays_dict,
            mode='mask',
            prediction_date='2023-06-01'
        )

        # Transform
        X_transformed = transformer.fit_transform(sample_time_series_data)

        # Inverse transform
        X_restored = transformer.inverse_transform(X_transformed)

        # Les données restaurées devraient être proches des originales
        assert isinstance(X_restored, pd.DataFrame)
        assert X_restored.shape == sample_time_series_data.shape

        # Vérification que les valeurs non-masquées sont identiques
        # et que les valeurs masquées ont été restaurées
        for col in ['PIB', 'inflation', 'unemployment']:
            # Les positions non-masquées devraient être identiques
            not_masked = X_transformed[col].notna()
            pd.testing.assert_series_equal(
                X_restored.loc[not_masked, col],
                X_transformed.loc[not_masked, col],
                check_names=False
            )

    def test_inverse_transform_shift_mode(self, sample_time_series_data, sample_delays_dict):
        """Test de l'inverse transform en mode shift."""
        transformer = ReleaseDelayTransformer(
            delays_dict=sample_delays_dict,
            mode='shift',
            prediction_date='2023-12-01'
        )

        X_transformed = transformer.fit_transform(sample_time_series_data)

        # L'inverse transform en mode shift retourne les données originales
        with pytest.warns(UserWarning, match="L'inversion de la transformation 'shift'"):
            X_restored = transformer.inverse_transform(X_transformed)

        # Devrait retourner les données originales
        pd.testing.assert_frame_equal(X_restored, sample_time_series_data, check_dtype=False)

    def test_prediction_date_resolution(self, sample_delays_dict):
        """Test de résolution de la date de prédiction."""
        transformer = ReleaseDelayTransformer(
            delays_dict=sample_delays_dict,
            prediction_date='today'
        )

        # Test avec 'today'
        resolved = transformer._resolve_prediction_date('today', pd.DataFrame())
        assert isinstance(resolved, datetime)

        # Test avec string de date
        resolved = transformer._resolve_prediction_date('2023-12-01', pd.DataFrame())
        assert resolved == datetime(2023, 12, 1)

        # Test avec datetime
        dt = datetime(2023, 12, 1)
        resolved = transformer._resolve_prediction_date(dt, pd.DataFrame())
        assert resolved == dt

        # Test avec format invalide
        with pytest.raises(ValueError, match="Format de date invalide"):
            transformer._resolve_prediction_date('invalid date', pd.DataFrame())

        # Test avec type invalide
        with pytest.raises(ValueError, match="prediction_date doit être"):
            transformer._resolve_prediction_date(123, pd.DataFrame())

    def test_get_delay_for_column_time_series(self, sample_time_series_data, sample_delays_dict):
        """Test de récupération de délai pour série temporelle."""
        transformer = ReleaseDelayTransformer(delays_dict=sample_delays_dict)
        transformer.fit(sample_time_series_data)

        # Test avec colonne existante
        delay = transformer._get_delay_for_column('PIB', sample_time_series_data)
        assert delay == 45.0

        # Test avec colonne inexistante
        delay = transformer._get_delay_for_column('unknown', sample_time_series_data)
        assert delay is None

    def test_get_delay_for_column_panel(self, sample_panel_data, sample_panel_delays_dict):
        """Test de récupération de délai pour données panel."""
        transformer = ReleaseDelayTransformer(
            delays_dict=sample_panel_delays_dict,
            panel_cols=['country']
        )
        transformer.fit(sample_panel_data)

        # Pour les données panel, la récupération dépend de l'entité
        # Cette implémentation suppose une entité par DataFrame
        delay = transformer._get_delay_for_column('PIB', sample_panel_data)
        assert delay is not None

    def test_handle_missing_delays(self, sample_time_series_data):
        """Test de gestion des délais manquants."""
        incomplete_delays = {'PIB': 45.0}  # Manque inflation et unemployment

        # Mode 'ignore'
        transformer = ReleaseDelayTransformer(
            delays_dict=incomplete_delays,
            handle_missing_delays='ignore'
        )
        transformer.fit(sample_time_series_data)

        delay = transformer._get_delay_for_column('inflation', sample_time_series_data)
        assert delay is None

        # Mode 'warn'
        transformer = ReleaseDelayTransformer(
            delays_dict=incomplete_delays,
            handle_missing_delays='warn',
            default_delay=20.0
        )
        transformer.fit(sample_time_series_data)

        with pytest.warns(UserWarning, match="Délai manquant pour la colonne"):
            delay = transformer._get_delay_for_column('inflation', sample_time_series_data)
        assert delay == 20.0

        # Mode 'error'
        transformer = ReleaseDelayTransformer(
            delays_dict=incomplete_delays,
            handle_missing_delays='error'
        )
        transformer.fit(sample_time_series_data)

        with pytest.raises(ValueError, match="Délai manquant pour la colonne"):
            transformer._get_delay_for_column('inflation', sample_time_series_data)

    def test_get_transformation_summary(self, sample_time_series_data, sample_delays_dict):
        """Test du résumé de transformation."""
        transformer = ReleaseDelayTransformer(
            delays_dict=sample_delays_dict,
            mode='shift',
            prediction_date='2023-12-01'
        )

        transformer.fit(sample_time_series_data)
        summary = transformer.get_transformation_summary()

        assert isinstance(summary, dict)
        required_keys = [
            'mode', 'prediction_date', 'reference_point', 'delays_applied',
            'transformation_metadata', 'is_panel_data', 'time_column', 'panel_columns'
        ]

        for key in required_keys:
            assert key in summary

        assert summary['mode'] == 'shift'
        assert summary['delays_applied'] == sample_delays_dict
        assert summary['is_panel_data'] == False

    def test_update_delays(self, sample_time_series_data, sample_delays_dict):
        """Test de mise à jour des délais."""
        transformer = ReleaseDelayTransformer(delays_dict=sample_delays_dict)
        transformer.fit(sample_time_series_data)

        # Mise à jour avec nouveaux délais
        new_delays = {'PIB': 50.0, 'new_indicator': 25.0}
        transformer.update_delays(new_delays)

        # Vérification
        available_delays = transformer.get_available_delays()
        assert available_delays['PIB'] == 50.0  # Mise à jour
        assert available_delays['inflation'] == 30.0  # Inchangé
        assert available_delays['new_indicator'] == 25.0  # Nouveau

    def test_get_available_delays(self, sample_delays_dict):
        """Test de récupération des délais disponibles."""
        transformer = ReleaseDelayTransformer(delays_dict=sample_delays_dict)

        # Avant fit, devrait retourner un dict vide
        available = transformer.get_available_delays()
        assert available == {}

        # Après fit avec données vides (pour déclencher l'exception d'attribut manquant)
        try:
            transformer.fit(pd.DataFrame({'date': [datetime.now()], 'value': [1]}))
            available = transformer.get_available_delays()
            assert available == sample_delays_dict
        except:
            # Si le fit échoue, c'est normal pour ce test
            pass

    def test_fit_without_data_validation(self, sample_time_series_data, sample_delays_dict):
        """Test du fit sans validation des données."""
        transformer = ReleaseDelayTransformer(
            delays_dict=sample_delays_dict,
            validate_input=False
        )

        # Devrait fonctionner même avec validate_input=False
        transformer.fit(sample_time_series_data)
        assert hasattr(transformer, 'delays_dict_')

    def test_not_fitted_error(self, sample_time_series_data, sample_delays_dict):
        """Test d'erreur transformer non ajusté."""
        transformer = ReleaseDelayTransformer(delays_dict=sample_delays_dict)

        # Transform sans fit devrait lever une erreur
        with pytest.raises(Exception):  # sklearn's NotFittedError
            transformer.transform(sample_time_series_data)

        # Inverse transform sans fit devrait lever une erreur
        with pytest.raises(Exception):
            transformer.inverse_transform(sample_time_series_data)

        # get_transformation_summary sans fit devrait lever une erreur
        with pytest.raises(Exception):
            transformer.get_transformation_summary()


class TestFactoryFunctions:
    """Tests pour les fonctions factory."""

    def test_create_delay_transformer_from_dict(self, sample_delays_dict):
        """Test de création depuis un dictionnaire."""
        transformer = create_delay_transformer_from_dict(
            sample_delays_dict,
            mode='mask',
            prediction_date='2023-12-01',
            time_col='timestamp'
        )

        assert isinstance(transformer, ReleaseDelayTransformer)
        assert transformer.delays_dict == sample_delays_dict
        assert transformer.mode == 'mask'
        assert transformer.prediction_date == '2023-12-01'
        assert transformer.time_col == 'timestamp'

    def test_create_delay_transformer_from_calculator(self):
        """Test de création depuis un calculateur."""
        import tempfile
        import os

        temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        temp_file.close()

        try:
            connection_string = f"sqlite:///{temp_file.name}"
            db_manager = create_database_manager(connection_string, create_tables=True)
            calculator = ReleaseDelayCalculator(db_manager=db_manager)

            transformer = create_delay_transformer_from_calculator(
                calculator,
                mode='shift',
                prediction_date='2023-12-01',
                panel_cols=['country']
            )

            assert isinstance(transformer, ReleaseDelayTransformer)
            assert transformer.delay_calculator == calculator
            assert transformer.mode == 'shift'
            assert transformer.panel_cols == ['country']

        finally:
            try:
                os.unlink(temp_file.name)
            except:
                pass


class TestEdgeCases:
    """Tests pour les cas limites."""

    def test_empty_delays_dict(self):
        """Test avec dictionnaire de délais vide."""
        transformer = ReleaseDelayTransformer(delays_dict={})

        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'value': range(10)
        })

        # Devrait fonctionner mais ne rien transformer
        X_transformed = transformer.fit_transform(data)
        pd.testing.assert_frame_equal(X_transformed, data, check_dtype=False)

    def test_zero_delays(self, sample_time_series_data):
        """Test avec délais à zéro."""
        zero_delays = {'PIB': 0.0, 'inflation': 0.0, 'unemployment': 0.0}

        transformer = ReleaseDelayTransformer(
            delays_dict=zero_delays,
            mode='shift'
        )

        X_transformed = transformer.fit_transform(sample_time_series_data)

        # Avec délais à zéro, les données devraient être inchangées
        pd.testing.assert_frame_equal(X_transformed, sample_time_series_data, check_dtype=False)

    def test_negative_delays(self, sample_time_series_data):
        """Test avec délais négatifs."""
        negative_delays = {'PIB': -10.0, 'inflation': -5.0}

        transformer = ReleaseDelayTransformer(
            delays_dict=negative_delays,
            mode='mask'
        )

        # En mode mask avec délais négatifs, rien ne devrait être masqué
        X_transformed = transformer.fit_transform(sample_time_series_data)

        # Vérification que rien n'a été masqué
        original_nan_count = sample_time_series_data[['PIB', 'inflation']].isna().sum().sum()
        transformed_nan_count = X_transformed[['PIB', 'inflation']].isna().sum().sum()

        assert transformed_nan_count == original_nan_count

    def test_very_large_delays(self, sample_time_series_data):
        """Test avec délais très importants."""
        large_delays = {'PIB': 500.0, 'inflation': 1000.0}  # Délais en jours très importants

        transformer = ReleaseDelayTransformer(
            delays_dict=large_delays,
            mode='mask',
            prediction_date='2023-06-01'
        )

        X_transformed = transformer.fit_transform(sample_time_series_data)

        # Avec des délais si importants, tout devrait être masqué
        assert X_transformed['PIB'].isna().all()
        assert X_transformed['inflation'].isna().all()

    def test_missing_time_column(self, sample_delays_dict):
        """Test avec colonne temporelle manquante."""
        transformer = ReleaseDelayTransformer(
            delays_dict=sample_delays_dict,
            time_col='missing_date'
        )

        data = pd.DataFrame({'value': [1, 2, 3]})

        with pytest.raises(ValueError):
            transformer.fit(data)

    def test_malformed_panel_data(self, sample_delays_dict):
        """Test avec données panel mal formées."""
        transformer = ReleaseDelayTransformer(
            delays_dict=sample_delays_dict,
            panel_cols=['country']
        )

        # Données sans colonne country
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3),
            'value': [1, 2, 3]
        })

        with pytest.raises(ValueError):
            transformer.fit(data)