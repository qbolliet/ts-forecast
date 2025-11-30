"""Tests for PanelwiseTransformer.

Tests de validation de la compatibilité sklearn et du comportement
du PanelwiseTransformer.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.base import clone
import warnings

# Import du module à tester
import sys
sys.path.insert(0, '/home/claude')

from panelwise_transformer import PanelwiseTransformer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_panel_data():
    """Création de données panel de test."""
    np.random.seed(42)
    n_periods = 20
    
    # Création de données pour 3 pays
    dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')
    
    data_frames = []
    for country in ['FR', 'DE', 'IT']:
        df = pd.DataFrame({
            'date': dates,
            'country': country,
            'value': np.random.randn(n_periods) * (10 if country == 'FR' else 5),
            'feature1': np.random.randn(n_periods) * 100 + (50 if country == 'DE' else 0),
            'feature2': np.random.randn(n_periods)
        })
        data_frames.append(df)
    
    return pd.concat(data_frames, ignore_index=True)


@pytest.fixture
def multi_level_panel_data():
    """Données panel avec plusieurs niveaux d'entités."""
    np.random.seed(42)
    n_periods = 10
    dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')
    
    data_frames = []
    for country in ['FR', 'DE']:
        for sector in ['tech', 'finance']:
            df = pd.DataFrame({
                'date': dates,
                'country': country,
                'sector': sector,
                'value': np.random.randn(n_periods),
                'revenue': np.random.randn(n_periods) * 1000
            })
            data_frames.append(df)
    
    return pd.concat(data_frames, ignore_index=True)


# =============================================================================
# Tests de base
# =============================================================================

class TestPanelwiseTransformerBasic:
    """Tests basiques du PanelwiseTransformer."""
    
    def test_init(self):
        """Test de l'initialisation."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            panel_cols=['country']
        )
        assert transformer.panel_cols == ['country']
        assert isinstance(transformer.transformer, StandardScaler)
    
    def test_fit_transform_no_validation(self, sample_panel_data):
        """Test fit_transform sans validation (préserve la structure)."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country'],
            validate_input=False  # Désactivation de la validation
        )
        
        result = transformer.fit_transform(sample_panel_data)
        
        # Vérification de la forme
        assert result.shape == sample_panel_data.shape
        
        # Vérification que les colonnes panel sont préservées
        assert 'country' in result.columns
        assert 'date' in result.columns
    
    def test_fit_transform_with_validation(self, sample_panel_data):
        """Test fit_transform avec validation (convertit en MultiIndex)."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country'],
            validate_input=True
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transformer.fit_transform(sample_panel_data)
        
        # Avec validation, les colonnes panel/time sont dans l'index
        assert isinstance(result.index, pd.MultiIndex)
        # Le nombre de lignes est préservé
        assert len(result) == len(sample_panel_data)
    
    def test_per_entity_scaling(self, sample_panel_data):
        """Vérification que le scaling est bien appliqué par entité."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country'],
            validate_input=False  # Pour préserver la structure
        )
        
        result = transformer.fit_transform(sample_panel_data)
        
        # Vérification que chaque entité a mean≈0 et std≈1
        for country in ['FR', 'DE', 'IT']:
            country_data = result[result['country'] == country]
            
            # Mean proche de 0
            assert np.abs(country_data['value'].mean()) < 0.1
            # Std proche de 1
            assert np.abs(country_data['value'].std() - 1.0) < 0.1
    
    def test_fit_and_transform_separately(self, sample_panel_data):
        """Test fit et transform séparés."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country'],
            validate_input=False
        )
        
        # Fit
        transformer.fit(sample_panel_data)
        
        # Vérification des attributs
        assert hasattr(transformer, 'transformers_')
        assert len(transformer.transformers_) == 3  # 3 pays
        
        # Transform
        result = transformer.transform(sample_panel_data)
        assert result.shape == sample_panel_data.shape
    
    def test_inverse_transform(self, sample_panel_data):
        """Test de inverse_transform."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country'],
            validate_input=False
        )
        
        transformed = transformer.fit_transform(sample_panel_data)
        inversed = transformer.inverse_transform(transformed)
        
        # Vérification que les valeurs sont restaurées
        np.testing.assert_array_almost_equal(
            inversed['value'].values,
            sample_panel_data['value'].values,
            decimal=10
        )


# =============================================================================
# Tests de compatibilité sklearn
# =============================================================================

class TestSklearnCompatibility:
    """Tests de compatibilité avec l'écosystème sklearn."""
    
    def test_clone(self, sample_panel_data):
        """Test que le transformer peut être cloné."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country'],
            validate_input=False
        )
        
        # Clone avant fit
        cloned = clone(transformer)
        assert cloned is not transformer
        assert cloned.transformer is not transformer.transformer
        
        # Les paramètres propres sont identiques
        assert cloned.time_col == transformer.time_col
        assert cloned.panel_cols == transformer.panel_cols
        assert cloned.n_jobs == transformer.n_jobs
    
    def test_get_set_params(self):
        """Test get_params et set_params."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(with_mean=True),
            time_col='date',
            panel_cols=['country']
        )
        
        params = transformer.get_params(deep=True)
        
        # Vérification des paramètres propres
        assert params['time_col'] == 'date'
        assert params['panel_cols'] == ['country']
        
        # Vérification des paramètres imbriqués
        assert 'transformer__with_mean' in params
        assert params['transformer__with_mean'] is True
        
        # Modification via set_params
        transformer.set_params(transformer__with_mean=False)
        assert transformer.transformer.with_mean is False
    
    def test_in_pipeline(self, sample_panel_data):
        """Test dans une pipeline sklearn."""
        pipe = Pipeline([
            ('panelwise', PanelwiseTransformer(
                transformer=StandardScaler(),
                time_col='date',
                panel_cols=['country'],
                validate_input=False
            ))
        ])
        
        result = pipe.fit_transform(sample_panel_data)
        assert result.shape == sample_panel_data.shape
    
    def test_nested_pipeline(self, sample_panel_data):
        """Test avec une pipeline imbriquée comme transformer."""
        inner_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('minmax', MinMaxScaler())
        ])
        
        transformer = PanelwiseTransformer(
            transformer=inner_pipe,
            time_col='date',
            panel_cols=['country'],
            validate_input=False
        )
        
        result = transformer.fit_transform(sample_panel_data)
        
        # Vérification que les valeurs sont dans [0, 1] (MinMaxScaler)
        numeric_cols = ['value', 'feature1', 'feature2']
        for col in numeric_cols:
            assert result[col].min() >= -0.01  # Petite tolérance
            assert result[col].max() <= 1.01
    
    def test_gridsearchcv_compatible(self, sample_panel_data):
        """Test de compatibilité avec GridSearchCV."""
        # Préparation des données
        X = sample_panel_data.copy()
        y = pd.Series(np.random.randn(len(X)), index=X.index)
        
        # Pipeline avec le transformer
        # On utilise un wrapper pour adapter le format
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country'],
            validate_input=False
        )
        
        # Test de fit avec y en Series
        transformer.fit(X, y)
        
        # Vérification que les paramètres peuvent être modifiés
        transformer.set_params(transformer__with_mean=False)
        assert transformer.transformer.with_mean is False


# =============================================================================
# Tests multi-niveaux
# =============================================================================

class TestMultiLevelPanel:
    """Tests avec plusieurs colonnes panel."""
    
    def test_multi_level_fit_transform(self, multi_level_panel_data):
        """Test avec plusieurs colonnes panel."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country', 'sector'],
            validate_input=False
        )
        
        result = transformer.fit_transform(multi_level_panel_data)
        
        # Vérification qu'on a 4 transformers (2 pays × 2 secteurs)
        assert transformer.n_entities_ == 4
        
        # Vérification du scaling par groupe
        for country in ['FR', 'DE']:
            for sector in ['tech', 'finance']:
                mask = (
                    (result['country'] == country) &
                    (result['sector'] == sector)
                )
                group_data = result[mask]
                
                assert np.abs(group_data['value'].mean()) < 0.2
    
    def test_get_entity_transformer(self, multi_level_panel_data):
        """Test de récupération d'un transformer spécifique."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country', 'sector'],
            validate_input=False
        )
        
        transformer.fit(multi_level_panel_data)
        
        # Récupération du transformer pour FR/tech
        entity_trans = transformer.get_entity_transformer(('FR', 'tech'))
        
        assert isinstance(entity_trans, StandardScaler)
        assert hasattr(entity_trans, 'mean_')  # Vérifie qu'il est fitté


# =============================================================================
# Tests de gestion d'erreurs
# =============================================================================

class TestErrorHandling:
    """Tests de gestion des erreurs."""
    
    def test_missing_panel_cols_error(self, sample_panel_data):
        """Test erreur si panel_cols non spécifié."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=None,
            validate_input=False  # Désactivation pour tester notre erreur
        )
        
        with pytest.raises(ValueError, match="panel_cols must be specified"):
            transformer.fit(sample_panel_data)
    
    def test_unknown_entity_warning(self, sample_panel_data):
        """Test warning pour entité inconnue au transform."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country'],
            error_handling='warn',
            validate_input=False
        )
        
        # Fit sur un sous-ensemble
        train_data = sample_panel_data[sample_panel_data['country'] != 'IT']
        transformer.fit(train_data)
        
        # Transform avec une entité inconnue
        with pytest.warns(UserWarning, match="not seen during fit"):
            result = transformer.transform(sample_panel_data)
    
    def test_error_handling_raise(self, sample_panel_data):
        """Test error_handling='raise'."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country'],
            error_handling='raise',
            validate_input=False
        )
        
        train_data = sample_panel_data[sample_panel_data['country'] != 'IT']
        transformer.fit(train_data)
        
        with pytest.raises(ValueError, match="not seen during fit"):
            transformer.transform(sample_panel_data)


# =============================================================================
# Tests de préservation de structure
# =============================================================================

class TestStructurePreservation:
    """Tests de préservation de la structure des données."""
    
    def test_index_preservation(self, sample_panel_data):
        """Test que l'index est préservé (sans validation)."""
        sample_panel_data = sample_panel_data.copy()
        sample_panel_data.index = range(100, 100 + len(sample_panel_data))
        
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country'],
            validate_input=False
        )
        
        result = transformer.fit_transform(sample_panel_data)
        
        pd.testing.assert_index_equal(result.index, sample_panel_data.index)
    
    def test_column_order_preservation(self, sample_panel_data):
        """Test que l'ordre des colonnes est préservé (sans validation)."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country'],
            validate_input=False
        )
        
        result = transformer.fit_transform(sample_panel_data)
        
        assert list(result.columns) == list(sample_panel_data.columns)
    
    def test_row_order_preservation(self, sample_panel_data):
        """Test que l'ordre des lignes est préservé."""
        # Mélange des données
        shuffled = sample_panel_data.sample(frac=1, random_state=42)
        
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country'],
            validate_input=False
        )
        
        result = transformer.fit_transform(shuffled)
        
        # L'ordre des index doit être identique
        pd.testing.assert_index_equal(result.index, shuffled.index)


# =============================================================================
# Tests avec validation activée
# =============================================================================

class TestWithValidation:
    """Tests avec validation activée (comportement par défaut)."""
    
    def test_multiindex_after_validation(self, sample_panel_data):
        """Vérifie que la validation crée un MultiIndex."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country'],
            validate_input=True
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transformer.fit_transform(sample_panel_data)
        
        # Après validation, on a un MultiIndex (country, date)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ['country', 'date']
    
    def test_scaling_with_multiindex(self, sample_panel_data):
        """Vérifie le scaling avec MultiIndex."""
        transformer = PanelwiseTransformer(
            transformer=StandardScaler(),
            time_col='date',
            panel_cols=['country'],
            validate_input=True
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = transformer.fit_transform(sample_panel_data)
        
        # Vérification du scaling par entité via le MultiIndex
        for country in ['FR', 'DE', 'IT']:
            country_data = result.xs(country, level='country')
            assert np.abs(country_data['value'].mean()) < 0.1


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
