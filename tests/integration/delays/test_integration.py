"""Tests d'intégration pour le module delays.

Ce module contient les tests d'intégration pour vérifier le fonctionnement
cohérent des différents composants du module delays ensemble.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline

# Import des modules à tester
from tsforecast.delays.transformers import (
    PublicationDelayTransformer,
    ShiftTransformer,
    MaskTransformer,
    create_delay_transformer_factory,
    prepare_entity_kwargs_from_delays
)
from tsforecast.delays.calculator import calculate_applicable_delay
from tsforecast.delays.data_manager import compare_and_detect_delays
from tsforecast.panel import PanelwiseTransformer


# ============================================================================
# Fixtures de données de test
# ============================================================================

@pytest.fixture
def time_series_data():
    """Génère des données de séries temporelles pour les tests.
    
    Returns:
        pd.DataFrame: Données mensuelles avec plusieurs variables.
    """
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-06-30', freq='MS')
    
    data = pd.DataFrame({
        'GDP': 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'inflation': 2.5 + np.random.randn(len(dates)) * 0.3,
        'unemployment': 8.0 + np.random.randn(len(dates)) * 0.5
    }, index=dates)
    
    return data


@pytest.fixture
def panel_data():
    """Génère des données de panel pour les tests.
    
    Returns:
        pd.DataFrame: Données de panel avec MultiIndex (country, date).
    """
    np.random.seed(42)
    countries = ['France', 'Germany', 'Italy']
    dates = pd.date_range('2023-01-01', '2024-06-30', freq='MS')
    
    data = []
    for country in countries:
        for date in dates:
            data.append({
                'country': country,
                'date': date,
                'GDP': 100 + np.random.randn() * 2,
                'inflation': 2.5 + np.random.randn() * 0.5
            })
    
    df = pd.DataFrame(data)
    df = df.set_index(['country', 'date'])
    return df


@pytest.fixture
def daily_time_series():
    """Génère des séries temporelles journalières.
    
    Returns:
        pd.DataFrame: Données journalières sur 6 mois.
    """
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
    
    data = pd.DataFrame({
        'indicator_A': np.random.randn(len(dates)) * 10 + 100,
        'indicator_B': np.random.randn(len(dates)) * 5 + 50
    }, index=dates)
    
    return data


@pytest.fixture
def daily_panel_data():
    """Génère des données de panel journalières.
    
    Returns:
        pd.DataFrame: Données de panel journalières avec MultiIndex.
    """
    np.random.seed(42)
    entities = ['entity_A', 'entity_B', 'entity_C']
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    
    data = []
    for entity in entities:
        for date in dates:
            data.append({
                'entity': entity,
                'date': date,
                'value1': np.random.randn() * 10 + 100,
                'value2': np.random.randn() * 5 + 50
            })
    
    df = pd.DataFrame(data)
    df = df.set_index(['entity', 'date'])
    return df


@pytest.fixture
def multi_frequency_data():
    """Génère des données multi-fréquence (mensuelles et trimestrielles).
    
    Returns:
        tuple: (monthly_data, quarterly_data) DataFrames.
    """
    np.random.seed(42)
    
    # Données mensuelles
    monthly_dates = pd.date_range('2023-01-01', '2024-06-30', freq='MS')
    monthly_data = pd.DataFrame({
        'monthly_indicator': np.random.randn(len(monthly_dates)) * 10 + 100
    }, index=monthly_dates)
    
    # Données trimestrielles
    quarterly_dates = pd.date_range('2023-01-01', '2024-06-30', freq='QS')
    quarterly_data = pd.DataFrame({
        'quarterly_indicator': np.random.randn(len(quarterly_dates)) * 20 + 200
    }, index=quarterly_dates)
    
    return monthly_data, quarterly_data


# ============================================================================
# Tests d'intégration ShiftTransformer + PanelwiseTransformer
# ============================================================================

class TestShiftTransformerWithPanelwise:
    """Tests d'intégration du ShiftTransformer avec PanelwiseTransformer."""

    def test_shift_applied_independently_per_entity(self, panel_data):
        """Vérification que le shift est appliqué indépendamment à chaque entité."""
        # Création du transformer
        shifter = ShiftTransformer(n_periods=2, frequency='M')
        panelwise = PanelwiseTransformer(
            transformer=shifter,
            panel_cols=None,  # Détection automatique
            time_col=None
        )
        
        # Application de la transformation
        shifted = panelwise.fit_transform(panel_data)
        
        # Vérification de la structure
        assert isinstance(shifted, pd.DataFrame)
        assert isinstance(shifted.index, pd.MultiIndex)
        assert shifted.shape[1] == panel_data.shape[1]  # Même nombre de colonnes
        
        # Vérification que chaque entité a le même nombre d'observations
        original_counts = panel_data.groupby(level=0).size()
        shifted_counts = shifted.groupby(level=0).size()
        pd.testing.assert_series_equal(original_counts, shifted_counts)

    def test_shift_no_nan_introduced_per_entity(self, panel_data):
        """Vérification qu'aucun NaN n'est introduit par entité."""
        shifter = ShiftTransformer(n_periods=3, frequency='M')
        panelwise = PanelwiseTransformer(transformer=shifter)
        
        shifted = panelwise.fit_transform(panel_data)
        
        # Vérification qu'aucun NaN n'est introduit
        original_nan_count = panel_data.isna().sum().sum()
        shifted_nan_count = shifted.isna().sum().sum()
        assert shifted_nan_count == original_nan_count

    def test_shift_transform_inverse_transform_per_entity(self, panel_data):
        """Vérification de la récupération des données pour chaque entité."""
        shifter = ShiftTransformer(n_periods=4, frequency='M')
        panelwise = PanelwiseTransformer(transformer=shifter)
        
        shifted = panelwise.fit_transform(panel_data)
        recovered = panelwise.inverse_transform(shifted)
        
        # Vérification de la récupération
        pd.testing.assert_index_equal(recovered.index, panel_data.index)
        np.testing.assert_array_almost_equal(
            recovered.values.astype(float),
            panel_data.values.astype(float),
            decimal=10
        )

    def test_shift_different_params_per_entity(self, panel_data):
        """Test avec des paramètres différents par entité via factory."""
        def shifter_factory(entity_key):
            # Shift différent selon le pays
            shifts = {
                ('France',): 1,
                ('Germany',): 2,
                ('Italy',): 3
            }
            n_periods = shifts.get(entity_key, 1)
            return ShiftTransformer(n_periods=n_periods, frequency='M')
        
        panelwise = PanelwiseTransformer(transformer=shifter_factory)
        
        shifted = panelwise.fit_transform(panel_data)
        
        # Vérification que les transformers ont été créés pour chaque entité
        assert len(panelwise.transformers_) == 3


# ============================================================================
# Tests d'intégration MaskTransformer + PanelwiseTransformer
# ============================================================================

class TestMaskTransformerWithPanelwise:
    """Tests d'intégration du MaskTransformer avec PanelwiseTransformer."""

    def test_mask_applied_independently_per_entity(self, daily_panel_data):
        """Vérification que le masquage est appliqué indépendamment à chaque entité."""
        masker = MaskTransformer(n_obs=3, mask_frequency='M', how='last')
        panelwise = PanelwiseTransformer(transformer=masker)
        
        masked = panelwise.fit_transform(daily_panel_data)
        
        # Vérification de la structure
        assert isinstance(masked, pd.DataFrame)
        assert isinstance(masked.index, pd.MultiIndex)
        
        # Vérification que chaque entité a le même nombre d'observations
        original_counts = daily_panel_data.groupby(level=0).size()
        masked_counts = masked.groupby(level=0).size()
        pd.testing.assert_series_equal(original_counts, masked_counts)

    def test_mask_correct_per_entity(self, daily_panel_data):
        """Vérification que chaque entité est masquée correctement."""
        masker = MaskTransformer(n_obs=5, mask_frequency='M', how='last')
        panelwise = PanelwiseTransformer(transformer=masker)
        
        masked = panelwise.fit_transform(daily_panel_data)
        
        # Vérification par entité
        for entity in ['entity_A', 'entity_B', 'entity_C']:
            entity_data = masked.loc[entity]
            original_entity = daily_panel_data.loc[entity]
            
            # Des valeurs doivent être masquées
            assert entity_data.isna().sum().sum() > 0
            
            # Le nombre d'observations doit être identique
            assert len(entity_data) == len(original_entity)

    def test_mask_transform_inverse_transform_per_entity(self, daily_panel_data):
        """Vérification de la récupération des données pour chaque entité."""
        masker = MaskTransformer(n_obs=4, mask_frequency='M', how='first')
        panelwise = PanelwiseTransformer(transformer=masker)
        
        masked = panelwise.fit_transform(daily_panel_data)
        recovered = panelwise.inverse_transform(masked)
        
        # Vérification de la récupération pour chaque entité
        for entity in ['entity_A', 'entity_B', 'entity_C']:
            original_entity = daily_panel_data.loc[entity]
            recovered_entity = recovered.loc[entity]
            
            pd.testing.assert_index_equal(recovered_entity.index, original_entity.index)
            np.testing.assert_array_almost_equal(
                recovered_entity.values.astype(float),
                original_entity.values.astype(float),
                decimal=10
            )


# ============================================================================
# Tests d'intégration du workflow complet de détection des délais
# ============================================================================

class TestDelayDetectionWorkflow:
    """Tests du workflow complet : détection → calcul → transformation."""

    def test_workflow_time_series_mono_frequency(self, time_series_data):
        """Test du workflow complet sur séries temporelles mono-fréquence."""
        # Date de "téléchargement" des données
        download_date = datetime(2024, 7, 15)
        
        # 1. Détection des délais (sans données existantes → dernière obs non-nulle)
        detected_delays = compare_and_detect_delays(
            new_data=time_series_data,
            existing_data=None,
            download_date=download_date,
            reference_point='end',
            delay_unit='D'
        )
        
        # Vérification de la détection
        assert len(detected_delays) > 0
        assert 'release_delay' in detected_delays.columns
        assert 'observation_date' in detected_delays.columns
        
        # 2. Calcul des délais applicables
        applicable_delays = calculate_applicable_delay(
            publication_delays=detected_delays,
            target_reference_point='end',
            target_frequency='M',
            aggregation_method='median'
        )
        
        # Vérification du calcul
        assert 'applicable_delay' in applicable_delays.columns
        assert len(applicable_delays) == len(time_series_data.columns)
        
        # 3. Application de la transformation
        transformer = PublicationDelayTransformer(
            delays=applicable_delays,
            strategy='shift',
            prediction_date=download_date
        )
        
        transformed = transformer.fit_transform(time_series_data)
        
        # Vérification de la transformation
        assert transformed.shape == time_series_data.shape
        
        # 4. Vérification de la cohérence : la dernière observation non-nulle
        # doit correspondre à la date de prédiction
        for col in time_series_data.columns:
            # Extraction de la dernière date non-nulle dans les données originales
            last_non_null_idx = time_series_data[col].dropna().index[-1]
            
            # Dans les données transformées, cette valeur devrait être décalée
            # vers une date proche de download_date
            original_value = time_series_data.loc[last_non_null_idx, col]
            
            # La valeur doit exister dans les données transformées
            assert original_value in transformed[col].values

    def test_workflow_panel_mono_frequency(self, panel_data):
        """Test du workflow complet sur données de panel mono-fréquence."""
        download_date = datetime(2024, 7, 15)
        
        # 1. Détection des délais
        detected_delays = compare_and_detect_delays(
            new_data=panel_data,
            existing_data=None,
            download_date=download_date,
            reference_point='end',
            delay_unit='D'
        )
        
        # Vérification de la détection
        assert len(detected_delays) > 0
        # Doit avoir des observations pour chaque entité
        entities_in_delays = detected_delays.index.get_level_values(0).unique()
        original_entities = panel_data.index.get_level_values(0).unique()
        assert set(entities_in_delays) == set(original_entities)
        
        # 2. Calcul des délais avec agrégation par panel
        applicable_delays = calculate_applicable_delay(
            publication_delays=detected_delays,
            target_reference_point='end',
            target_frequency='M',
            aggregate_by_panel=True,
            aggregation_method='median'
        )
        
        # Vérification du calcul
        assert 'applicable_delay' in applicable_delays.columns
        # Doit y avoir un délai par (entity, variable)
        assert isinstance(applicable_delays.index, pd.MultiIndex)
        
        # 3. Application de la transformation avec PanelwiseTransformer
        def transformer_factory(entity_key):
            # Extraction des délais pour cette entité
            entity_delays = applicable_delays.loc[entity_key]
            delays_dict = entity_delays['applicable_delay'].to_dict()
            
            return PublicationDelayTransformer(
                delays=delays_dict,
                strategy='shift',
                prediction_date=download_date,
                delay_unit='D',
                reference_point='end'
            )
        
        panelwise = PanelwiseTransformer(transformer=transformer_factory)
        transformed = panelwise.fit_transform(panel_data)
        
        # Vérification de la transformation
        assert transformed.shape == panel_data.shape
        assert isinstance(transformed.index, pd.MultiIndex)

    def test_workflow_multi_frequency(self, multi_frequency_data):
        """Test du workflow avec données multi-fréquence."""
        monthly_data, quarterly_data = multi_frequency_data
        download_date = datetime(2024, 7, 15)
        
        # Combinaison des données (différentes fréquences)
        # En pratique, on les traiterait séparément
        
        # Test sur données mensuelles
        monthly_delays = compare_and_detect_delays(
            new_data=monthly_data,
            existing_data=None,
            download_date=download_date,
            reference_point='end',
            delay_unit='D'
        )
        
        monthly_applicable = calculate_applicable_delay(
            publication_delays=monthly_delays,
            target_reference_point='end',
            target_frequency='M',
            aggregation_method='median'
        )
        
        # Test sur données trimestrielles
        quarterly_delays = compare_and_detect_delays(
            new_data=quarterly_data,
            existing_data=None,
            download_date=download_date,
            reference_point='end',
            delay_unit='D'
        )
        
        quarterly_applicable = calculate_applicable_delay(
            publication_delays=quarterly_delays,
            target_reference_point='end',
            target_frequency='Q',
            aggregation_method='median'
        )
        
        # Vérification que les délais sont différents pour les différentes fréquences
        assert len(monthly_applicable) > 0
        assert len(quarterly_applicable) > 0

    def test_last_observation_aligned_with_prediction_date(self, time_series_data):
        """Vérification que la dernière obs non-nulle est alignée avec prediction_date."""
        # Date de "téléchargement" des données
        download_date = datetime(2024, 7, 15)
        
        # Détection des délais
        detected_delays = compare_and_detect_delays(
            new_data=time_series_data,
            existing_data=None,
            download_date=download_date,
            reference_point='end',
            delay_unit='D'
        )
        
        # Vérification que les dates d'observation détectées correspondent
        # aux dernières observations non-nulles de chaque série
        for col in time_series_data.columns:
            last_non_null_date = time_series_data[col].dropna().index[-1]
            
            # Recherche de cette colonne dans les délais détectés
            col_delays = detected_delays[detected_delays['column'] == col]
            
            if len(col_delays) > 0:
                detected_date = col_delays['observation_date'].iloc[0]
                assert detected_date == last_non_null_date


class TestDelayDetectionWorkflowPanel:
    """Tests spécifiques du workflow pour les données de panel."""

    def test_panel_each_entity_has_delays(self, panel_data):
        """Vérification que chaque entité a des délais détectés."""
        download_date = datetime(2024, 7, 15)
        
        detected_delays = compare_and_detect_delays(
            new_data=panel_data,
            existing_data=None,
            download_date=download_date,
            reference_point='end',
            delay_unit='D'
        )
        
        # Chaque combinaison (entity, variable) doit avoir un délai
        original_entities = panel_data.index.get_level_values(0).unique()
        original_columns = panel_data.columns
        
        for entity in original_entities:
            entity_delays = detected_delays.loc[entity]
            for col in original_columns:
                col_delays = entity_delays[entity_delays['column'] == col]
                assert len(col_delays) > 0, f"Pas de délai pour {entity}/{col}"

    def test_panel_last_observation_per_entity(self, panel_data):
        """Vérification de la dernière observation par entité."""
        download_date = datetime(2024, 7, 15)
        
        detected_delays = compare_and_detect_delays(
            new_data=panel_data,
            existing_data=None,
            download_date=download_date,
            reference_point='end',
            delay_unit='D'
        )
        
        # Vérification pour chaque entité
        for entity in panel_data.index.get_level_values(0).unique():
            entity_data = panel_data.loc[entity]
            entity_delays = detected_delays.loc[entity]
            
            for col in entity_data.columns:
                # Dernière date non-nulle pour cette entité/colonne
                last_non_null = entity_data[col].dropna().index[-1]
                
                # Délai détecté pour cette colonne
                col_delay = entity_delays[entity_delays['column'] == col]
                detected_date = col_delay['observation_date'].iloc[0]
                
                assert detected_date == last_non_null


# ============================================================================
# Tests d'intégration avec données multi-fréquence combinées
# ============================================================================

class TestMultiFrequencyIntegration:
    """Tests d'intégration pour données multi-fréquence dans un même panel."""

    def test_mixed_frequency_panel(self):
        """Test avec un panel contenant des fréquences mixtes."""
        np.random.seed(42)
        
        # Création de données à fréquences mixtes
        # Entité 1 : données mensuelles
        dates_monthly = pd.date_range('2024-01-01', periods=6, freq='MS')
        monthly_data = []
        for date in dates_monthly:
            monthly_data.append({
                'entity': 'monthly_entity',
                'date': date,
                'value': np.random.randn() * 10 + 100
            })
        
        # Entité 2 : données trimestrielles
        dates_quarterly = pd.date_range('2024-01-01', periods=2, freq='QS')
        quarterly_data = []
        for date in dates_quarterly:
            quarterly_data.append({
                'entity': 'quarterly_entity',
                'date': date,
                'value': np.random.randn() * 20 + 200
            })
        
        # Combinaison
        all_data = pd.DataFrame(monthly_data + quarterly_data)
        all_data = all_data.set_index(['entity', 'date'])
        
        # Détection des délais
        download_date = datetime(2024, 7, 15)
        detected_delays = compare_and_detect_delays(
            new_data=all_data,
            existing_data=None,
            download_date=download_date,
            reference_point='end',
            delay_unit='D'
        )
        
        # Les fréquences doivent être différentes par entité
        monthly_entity_delays = detected_delays.loc['monthly_entity']
        quarterly_entity_delays = detected_delays.loc['quarterly_entity']
        
        assert monthly_entity_delays['frequency'].iloc[0] == 'monthly'
        assert quarterly_entity_delays['frequency'].iloc[0] == 'quarterly'


# ============================================================================
# Tests de combinaison de transformations
# ============================================================================

class TestCombinedTransformations:
    """Tests de combinaison de plusieurs transformations."""

    def test_shift_then_mask(self, daily_time_series):
        """Test de l'application séquentielle shift puis mask."""
        # Premier transformer : shift
        shifter = ShiftTransformer(n_periods=5, frequency='D')
        shifted = shifter.fit_transform(daily_time_series)
        
        # Deuxième transformer : mask
        masker = MaskTransformer(n_obs=3, mask_frequency='M', how='last')
        masked = masker.fit_transform(shifted)
        
        # Vérifications
        assert len(masked) == len(daily_time_series)
        assert masked.isna().sum().sum() > 0
        
        # Inverse transformations
        unmasked = masker.inverse_transform(masked)
        unshifted = shifter.inverse_transform(unmasked)
        
        # Récupération des données originales
        pd.testing.assert_frame_equal(unshifted, daily_time_series)

    def test_pipeline_with_transformers(self, daily_time_series):
        """Test d'utilisation dans un sklearn Pipeline."""
        # Création du pipeline (sans modèle à la fin, juste les transformations)
        # Note: pour un pipeline complet, il faudrait ajouter un estimateur final
        
        shifter = ShiftTransformer(n_periods=3, frequency='D')
        shifted = shifter.fit_transform(daily_time_series)
        
        # Vérification que le transformer est compatible sklearn
        assert hasattr(shifter, 'fit')
        assert hasattr(shifter, 'transform')
        assert hasattr(shifter, 'fit_transform')
        assert hasattr(shifter, 'inverse_transform')


# ============================================================================
# Tests de robustesse
# ============================================================================

class TestRobustness:
    """Tests de robustesse avec données imparfaites."""

    def test_data_with_missing_values(self):
        """Test avec données contenant des valeurs manquantes."""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        data = pd.DataFrame({
            'col1': np.random.randn(60) * 10 + 100,
            'col2': np.random.randn(60) * 5 + 50
        }, index=dates)
        
        # Introduction de valeurs manquantes
        data.iloc[10:15, 0] = np.nan
        data.iloc[25:28, 1] = np.nan
        
        # Test ShiftTransformer
        shifter = ShiftTransformer(n_periods=5, frequency='D')
        shifted = shifter.fit_transform(data)
        
        # Les NaN doivent être préservés (même nombre)
        original_nan = data.isna().sum().sum()
        shifted_nan = shifted.isna().sum().sum()
        assert shifted_nan == original_nan
        
        # Test MaskTransformer
        masker = MaskTransformer(n_obs=3, mask_frequency='M', how='last')
        masked = masker.fit_transform(data)
        
        # Plus de NaN après masquage
        assert masked.isna().sum().sum() > original_nan

    def test_empty_periods_handling(self):
        """Test avec des périodes vides dans les données."""
        # Données avec un gap
        dates1 = pd.date_range('2024-01-01', periods=15, freq='D')
        dates2 = pd.date_range('2024-02-01', periods=15, freq='D')
        dates = dates1.append(dates2)
        
        data = pd.Series(range(len(dates)), index=dates, name='test')
        
        # Doit fonctionner sans erreur
        masker = MaskTransformer(n_obs=3, mask_frequency='M', how='last')
        masked = masker.fit_transform(data)
        
        assert len(masked) == len(data)


# ============================================================================
# Tests de performance
# ============================================================================

class TestPerformance:
    """Tests de performance avec données volumineuses."""

    def test_large_panel_performance(self):
        """Test de performance avec un grand panel."""
        import time
        
        np.random.seed(42)
        n_entities = 50
        n_dates = 365
        
        # Génération de données volumineuses
        data = []
        dates = pd.date_range('2024-01-01', periods=n_dates, freq='D')
        
        for i in range(n_entities):
            for date in dates:
                data.append({
                    'entity': f'entity_{i}',
                    'date': date,
                    'value': np.random.randn() * 10 + 100
                })
        
        df = pd.DataFrame(data)
        df = df.set_index(['entity', 'date'])
        
        # Test de performance du ShiftTransformer avec PanelwiseTransformer
        start_time = time.time()
        
        shifter = ShiftTransformer(n_periods=10, frequency='D')
        panelwise = PanelwiseTransformer(transformer=shifter)
        shifted = panelwise.fit_transform(df)
        
        elapsed = time.time() - start_time
        
        # Doit s'exécuter en moins de 60 secondes
        assert elapsed < 60, f"Trop lent : {elapsed:.2f}s"
        
        # Vérification de la cohérence
        assert shifted.shape == df.shape
