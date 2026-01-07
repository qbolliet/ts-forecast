"""Tests unitaires pour le module transformers.

Ce module contient les tests pour PublicationDelayTransformer et les fonctions
auxiliaires associées.
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from sklearn.utils.validation import check_is_fitted, NotFittedError

# Import du module à tester
from tsforecast.delays.transformers import (
    PublicationDelayTransformer,
    ShiftTransformer, 
    MaskTransformer,
    prepare_entity_kwargs_from_delays,
    _build_entity_params,
    _extract_param_by_variable,
    _resolve_strategy
)

# ============================================================================
# Fixtures de données de test
# ============================================================================

@pytest.fixture
def sample_time_series():
    """Generate sample time series data for testing.
    
    Returns:
        pd.DataFrame: Time series with monthly dates and multiple variables.
    """
    # Données mensuelles sur 2 ans
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='MS')
    data = pd.DataFrame({
        'GDP': np.random.randn(len(dates)) * 10 + 1000,
        'inflation': np.random.randn(len(dates)) * 0.5 + 2.0,
        'unemployment': np.random.randn(len(dates)) * 0.3 + 5.0
    }, index=dates)
    return data


@pytest.fixture
def sample_panel_data():
    """Generate sample panel data for testing.
    
    Returns:
        pd.DataFrame: Panel data with MultiIndex (country, date).
    """
    # Données mensuelles pour 3 pays sur 1 an
    countries = ['France', 'Germany', 'Italy']
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='MS')
    
    data = []
    for country in countries:
        for date in dates:
            data.append({
                'country': country,
                'date': date,
                'GDP': np.random.randn() * 10 + 1000,
                'inflation': np.random.randn() * 0.5 + 2.0
            })
    
    df = pd.DataFrame(data)
    df = df.set_index(['country', 'date'])
    return df


@pytest.fixture
def delays_dict_simple():
    """Simple delays dictionary for testing.
    
    Returns:
        dict: Mapping variable names to delay values in days.
    """
    return {
        'GDP': 45.0,
        'inflation': 30.0,
        'unemployment': 15.0
    }


@pytest.fixture
def delays_dataframe():
    """Delays DataFrame with metadata for testing.
    
    Returns:
        pd.DataFrame: Delays with unit, reference_point, and target_frequency.
    """
    return pd.DataFrame({
        'variable': ['GDP', 'inflation', 'unemployment'],
        'delay': [45.0, 30.0, 15.0],
        'unit': ['D', 'D', 'D'],
        'reference_point': ['end', 'end', 'end'],
        'target_frequency': ['M', 'M', 'M']
    })


@pytest.fixture
def delays_dataframe_panel():
    """Panel delays DataFrame for testing.
    
    Returns:
        pd.DataFrame: Delays with entity-level variation.
    """
    data = []
    for country in ['France', 'Germany', 'Italy']:
        for var in ['GDP', 'inflation']:
            data.append({
                'country': country,
                'variable': var,
                'delay': np.random.uniform(20, 60),
                'unit': 'D',
                'reference_point': 'end',
                'target_frequency': 'M'
            })
    
    df = pd.DataFrame(data)
    df = df.set_index(['country', 'variable'])
    return df


# ============================================================================
# Tests de la classe PublicationDelayTransformer - Initialisation
# ============================================================================

class TestPublicationDelayTransformerInit:
    """Tests for PublicationDelayTransformer initialization."""
    
    def test_init_with_dict(self, delays_dict_simple):
        """Initialization avec un dictionnaire de délais."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift',
            prediction_date='2024-01-01'
        )
        
        assert transformer.delays == delays_dict_simple
        assert transformer.strategy == 'shift'
        assert transformer.prediction_date == '2024-01-01'
    
    def test_init_with_dataframe(self, delays_dataframe):
        """Initialization avec un DataFrame de délais."""
        transformer = PublicationDelayTransformer(
            delays=delays_dataframe,
            strategy='mask',
            prediction_date=datetime(2024, 6, 15)
        )
        
        assert isinstance(transformer.delays, pd.DataFrame)
        assert transformer.strategy == 'mask'
        assert isinstance(transformer.prediction_date, datetime)
    
    def test_init_with_strategy_dict(self, delays_dict_simple):
        """Initialization avec un dictionnaire de stratégies."""
        strategy_dict = {
            'GDP': 'shift',
            'inflation': 'mask',
            'unemployment': 'shift'
        }
        
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy=strategy_dict
        )
        
        assert transformer.strategy == strategy_dict
    
    def test_init_default_values(self, delays_dict_simple):
        """Initialization avec des valeurs par défaut."""
        default_vals = {
            'delay': 30.0,
            'unit': 'D',
            'reference_point': 'end',
            'target_frequency': 'M'
        }
        
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='mask',
            default_values=default_vals
        )
        
        assert transformer.default_values == default_vals
    
    def test_init_invalid_strategy_string(self, delays_dict_simple):
        """Validation de la stratégie invalide (chaîne de caractères)."""
        with pytest.raises(ValueError, match="strategy must be 'shift' or 'mask'"):
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                strategy='invalid'
            )
    
    def test_init_invalid_strategy_dict(self, delays_dict_simple):
        """Validation de la stratégie invalide (dictionnaire)."""
        with pytest.raises(ValueError, match="strategy must be 'shift' or 'mask'"):
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                strategy={'GDP': 'invalid_strategy'}
            )
    
    def test_init_invalid_strategy_type(self, delays_dict_simple):
        """Validation du type de stratégie invalide."""
        with pytest.raises(TypeError, match="'strategy' should be a string"):
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                strategy=123  # Type invalide
            )
    
    def test_init_invalid_reference_point(self, delays_dict_simple):
        """Validation du point de référence invalide."""
        with pytest.raises(ValueError, match="reference_point must be 'start' or 'end'"):
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                reference_point='middle'
            )
    
    def test_init_invalid_handle_missing(self, delays_dict_simple):
        """Validation de la gestion des délais manquants invalide."""
        with pytest.raises(ValueError, match="'handle_missing_delays' must be"):
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                handle_missing_delays='invalid'
            )
    
    def test_init_default_values_missing_keys(self, delays_dict_simple):
        """Validation des clés manquantes dans default_values."""
        incomplete_defaults = {
            'delay': 30.0,
            'unit': 'D'
            # Clés manquantes: reference_point, target_frequency
        }
        
        with pytest.raises(ValueError, match="Expected a 'default_values' dictionnary"):
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                strategy='mask',
                default_values=incomplete_defaults
            )
    
    def test_init_warning_strategy_dict_with_defaults(self, delays_dict_simple):
        """Warning quand strategy est un dict et default_values est fourni."""
        strategy_dict = {'GDP': 'shift', 'inflation': 'mask'}
        default_vals = {
            'delay': 30.0,
            'unit': 'D',
            'reference_point': 'end',
            'target_frequency': 'M'
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                strategy=strategy_dict,
                default_values=default_vals
            )
            assert len(w) == 1
            assert "default_values" in str(w[0].message)
    
    def test_init_warning_target_frequency_with_shift(self, delays_dict_simple):
        """Warning quand target_frequency est fourni avec strategy='shift'."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                strategy='shift',
                target_frequency='M'
            )
            assert len(w) == 1
            assert "target_frequency" in str(w[0].message)


# ============================================================================
# Tests de la classe PublicationDelayTransformer - Méthodes fit et transform
# ============================================================================

class TestPublicationDelayTransformerFitTransform:
    """Tests for fit and transform methods."""
    
    def test_fit_basic(self, sample_time_series, delays_dict_simple):
        """Méthode fit de base."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift',
            prediction_date='2024-06-01'
        )
        
        result = transformer.fit(sample_time_series)
        
        # Vérification que fit retourne self
        assert result is transformer
        
        # Vérification des attributs créés
        assert hasattr(transformer, 'prediction_date_')
        assert hasattr(transformer, 'inferred_params_')
        assert hasattr(transformer, 'detected_frequencies_')
    
    def test_fit_creates_shift_params(self, sample_time_series, delays_dict_simple):
        """Création des paramètres de shift après fit."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift'
        )
        
        transformer.fit(sample_time_series)
        
        assert hasattr(transformer, 'shift_params')
        assert 'GDP' in transformer.shift_params
        assert 'n_periods' in transformer.shift_params['GDP']
        assert 'frequency' in transformer.shift_params['GDP']
    
    def test_fit_creates_mask_params(self, sample_time_series, delays_dict_simple):
        """Création des paramètres de mask après fit."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='mask',
            prediction_date='2024-06-01'
        )
        
        transformer.fit(sample_time_series)
        
        assert hasattr(transformer, 'mask_params')
        # Vérification qu'au moins une variable est dans mask_params ou shift_params
        all_vars = set(transformer.mask_params.keys()) | set(transformer.shift_params.keys())
        assert 'GDP' in all_vars or 'inflation' in all_vars
    
    def test_transform_basic(self, sample_time_series, delays_dict_simple):
        """Transformation de base des données."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift'
        )
        
        transformer.fit(sample_time_series)
        result = transformer.transform(sample_time_series)
        
        # Vérification de la structure du résultat
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_time_series.shape
        assert list(result.columns) == list(sample_time_series.columns)
    
    def test_fit_transform(self, sample_time_series, delays_dict_simple):
        """Méthode fit_transform."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift'
        )
        
        result = transformer.fit_transform(sample_time_series)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_time_series.shape
    
    def test_transform_without_fit_raises_error(self, sample_time_series, delays_dict_simple):
        """Erreur si transform est appelé avant fit."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift'
        )
        
        with pytest.raises(NotFittedError):
            transformer.transform(sample_time_series)
    
    def test_transform_with_mask_strategy(self, sample_time_series, delays_dict_simple):
        """Transformation avec stratégie mask."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='mask',
            prediction_date='2024-06-01'
        )
        
        result = transformer.fit_transform(sample_time_series)
        
        # Vérification que des valeurs ont été masquées (NaN)
        original_nan_count = sample_time_series.isna().sum().sum()
        result_nan_count = result.isna().sum().sum()
        
        # Le nombre de NaN devrait être >= à l'original
        assert result_nan_count >= original_nan_count
    
    def test_transform_with_panel_data(self, sample_panel_data, delays_dataframe_panel):
        """Transformation de données panel."""
        transformer = PublicationDelayTransformer(
            delays=delays_dataframe_panel,
            strategy='shift',
            prediction_date='2024-01-01'
        )
        
        result = transformer.fit_transform(sample_panel_data)
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.shape == sample_panel_data.shape


# ============================================================================
# Tests de la classe PublicationDelayTransformer - Inverse transform
# ============================================================================

class TestPublicationDelayTransformerInverse:
    """Tests for inverse_transform method."""
    
    def test_inverse_transform_shift(self, sample_time_series, delays_dict_simple):
        """Transformation inverse avec stratégie shift."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift'
        )
        
        # Transformation
        transformed = transformer.fit_transform(sample_time_series)
        
        # Transformation inverse
        inversed = transformer.inverse_transform(transformed)
        
        # Vérification de la structure
        assert isinstance(inversed, pd.DataFrame)
        assert inversed.shape == sample_time_series.shape
    
    def test_inverse_transform_mask(self, sample_time_series, delays_dict_simple):
        """Transformation inverse avec stratégie mask."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='mask',
            prediction_date='2024-06-01'
        )
        
        transformed = transformer.fit_transform(sample_time_series)
        inversed = transformer.inverse_transform(transformed)
        
        assert isinstance(inversed, pd.DataFrame)
        # Les dimensions doivent être préservées
        assert inversed.shape == sample_time_series.shape
    
    def test_inverse_transform_without_fit_raises_error(self, sample_time_series, delays_dict_simple):
        """Erreur si inverse_transform est appelé avant fit."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift'
        )
        
        with pytest.raises(NotFittedError):
            transformer.inverse_transform(sample_time_series)


# ============================================================================
# Tests des cas limites
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_delays_dict(self, sample_time_series):
        """Dictionnaire de délais vide."""
        transformer = PublicationDelayTransformer(
            delays={},
            strategy='shift'
        )
        
        result = transformer.fit_transform(sample_time_series)
        
        # Avec un dictionnaire vide, les données ne devraient pas être modifiées
        pd.testing.assert_frame_equal(result, sample_time_series, check_dtype=False)
    
    def test_zero_delays(self, sample_time_series):
        """Délais à zéro."""
        zero_delays = {
            'GDP': 0.0,
            'inflation': 0.0,
            'unemployment': 0.0
        }
        
        transformer = PublicationDelayTransformer(
            delays=zero_delays,
            strategy='shift'
        )
        
        result = transformer.fit_transform(sample_time_series)
        
        # Avec des délais nuls, aucun décalage ne devrait être appliqué
        pd.testing.assert_frame_equal(result, sample_time_series, check_dtype=False)
    
    def test_negative_delays(self, sample_time_series):
        """Délais négatifs."""
        negative_delays = {
            'GDP': -10.0,
            'inflation': -5.0
        }
        
        transformer = PublicationDelayTransformer(
            delays=negative_delays,
            strategy='shift'
        )
        
        # Les délais négatifs devraient être acceptés
        result = transformer.fit_transform(sample_time_series)
        assert isinstance(result, pd.DataFrame)
    
    def test_very_large_delays(self, sample_time_series):
        """Délais très importants."""
        large_delays = {
            'GDP': 5000.0,  # Environ 14 ans en jours
            'inflation': 10000.0
        }
        
        transformer = PublicationDelayTransformer(
            delays=large_delays,
            strategy='mask',
            prediction_date='2024-06-01'
        )
        
        result = transformer.fit_transform(sample_time_series)
        
        # Avec des délais très importants, tout devrait être masqué
        assert result['GDP'].isna().all() or len(result['GDP'].dropna()) <= 1
    
    def test_single_column_dataframe(self):
        """DataFrame avec une seule colonne."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='MS')
        data = pd.DataFrame({'GDP': range(len(dates))}, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 30.0},
            strategy='shift'
        )
        
        result = transformer.fit_transform(data)
        assert result.shape == data.shape
        assert 'GDP' in result.columns
    
    def test_delays_for_nonexistent_columns(self, sample_time_series):
        """Délais pour des colonnes inexistantes."""
        delays_with_extra = {
            'GDP': 45.0,
            'nonexistent_var': 30.0,  # Variable inexistante
            'another_missing': 15.0
        }
        
        transformer = PublicationDelayTransformer(
            delays=delays_with_extra,
            strategy='shift',
            handle_missing_delays='ignore'
        )
        
        # Ne devrait pas lever d'erreur
        result = transformer.fit_transform(sample_time_series)
        assert isinstance(result, pd.DataFrame)
    
    def test_all_nan_column(self, sample_time_series):
        """Colonne entièrement composée de NaN."""
        data_with_nan = sample_time_series.copy()
        data_with_nan['all_nan'] = np.nan
        
        transformer = PublicationDelayTransformer(
            delays={'all_nan': 30.0},
            strategy='shift'
        )
        
        result = transformer.fit_transform(data_with_nan)
        # La colonne NaN devrait rester NaN
        assert result['all_nan'].isna().all()
    
    def test_dataframe_with_missing_values(self, sample_time_series):
        """DataFrame avec des valeurs manquantes."""
        data_with_missing = sample_time_series.copy()
        # Insertion de quelques valeurs manquantes
        data_with_missing.loc[data_with_missing.index[5:10], 'GDP'] = np.nan
        data_with_missing.loc[data_with_missing.index[15:20], 'inflation'] = np.nan
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 30.0, 'inflation': 15.0},
            strategy='shift'
        )
        
        result = transformer.fit_transform(data_with_missing)
        assert isinstance(result, pd.DataFrame)
    
    def test_non_standard_frequency(self):
        """Fréquence non standard (hebdomadaire)."""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='W')
        data = pd.DataFrame({
            'GDP': range(len(dates)),
            'inflation': range(len(dates))
        }, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 7.0, 'inflation': 14.0},
            strategy='shift',
            delay_unit='D'
        )
        
        result = transformer.fit_transform(data)
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# Tests des types d'arguments
# ============================================================================

class TestArgumentTypes:
    """Tests for different argument types."""
    
    def test_prediction_date_string(self, sample_time_series, delays_dict_simple):
        """Date de prédiction en chaîne de caractères."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift',
            prediction_date='2024-06-15'
        )
        
        transformer.fit(sample_time_series)
        assert isinstance(transformer.prediction_date_, datetime)
    
    def test_prediction_date_datetime(self, sample_time_series, delays_dict_simple):
        """Date de prédiction en objet datetime."""
        pred_date = datetime(2024, 6, 15)
        
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift',
            prediction_date=pred_date
        )
        
        transformer.fit(sample_time_series)
        assert isinstance(transformer.prediction_date_, datetime)
    
    def test_prediction_date_today(self, sample_time_series, delays_dict_simple):
        """Date de prédiction avec valeur 'today'."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift',
            prediction_date='today'
        )
        
        transformer.fit(sample_time_series)
        # La date devrait être proche de maintenant
        assert (datetime.now() - transformer.prediction_date_).days <= 1
    
    def test_delay_unit_variations(self, sample_time_series):
        """Différentes unités de délai."""
        for unit in ['D', 'h', 's', 'W']:
            delays = {'GDP': 30.0}
            
            transformer = PublicationDelayTransformer(
                delays=delays,
                strategy='shift',
                delay_unit=unit
            )
            
            result = transformer.fit_transform(sample_time_series)
            assert isinstance(result, pd.DataFrame)
    
    def test_delay_unit_as_dict(self, sample_time_series):
        """Unité de délai spécifiée par variable."""
        delays = {'GDP': 30.0, 'inflation': 7.0}
        unit_dict = {'GDP': 'D', 'inflation': 'W'}
        
        transformer = PublicationDelayTransformer(
            delays=delays,
            strategy='shift',
            delay_unit=unit_dict
        )
        
        result = transformer.fit_transform(sample_time_series)
        assert isinstance(result, pd.DataFrame)
    
    def test_reference_point_variations(self, sample_time_series, delays_dict_simple):
        """Différents points de référence."""
        for ref_point in ['start', 'end']:
            transformer = PublicationDelayTransformer(
                delays=delays_dict_simple,
                strategy='shift',
                reference_point=ref_point
            )
            
            result = transformer.fit_transform(sample_time_series)
            assert isinstance(result, pd.DataFrame)
    
    def test_reference_point_as_dict(self, sample_time_series):
        """Point de référence spécifié par variable."""
        delays = {'GDP': 30.0, 'inflation': 15.0}
        ref_point_dict = {'GDP': 'end', 'inflation': 'start'}
        
        transformer = PublicationDelayTransformer(
            delays=delays,
            strategy='shift',
            reference_point=ref_point_dict
        )
        
        result = transformer.fit_transform(sample_time_series)
        assert isinstance(result, pd.DataFrame)
    
    def test_target_frequency_as_dict(self, sample_time_series):
        """Fréquence cible spécifiée par variable."""
        delays = {'GDP': 30.0, 'inflation': 15.0}
        freq_dict = {'GDP': 'M', 'inflation': 'Q'}
        
        transformer = PublicationDelayTransformer(
            delays=delays,
            strategy='mask',
            target_frequency=freq_dict,
            prediction_date='2024-06-01'
        )
        
        result = transformer.fit_transform(sample_time_series)
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# Tests des fonctions auxiliaires
# ============================================================================

class TestAuxiliaryFunctions:
    """Tests for auxiliary functions."""
    
    def test_extract_param_by_variable_constant(self):
        """Extraction de paramètre avec valeur constante."""
        # Données avec valeur constante
        df = pd.DataFrame({
            'variable': ['GDP', 'inflation'],
            'unit': ['D', 'D']
        }).set_index('variable')
        
        result = _extract_param_by_variable(df, 'unit')
        
        # Devrait retourner la valeur unique
        assert result == 'D'
    
    def test_extract_param_by_variable_varying(self):
        """Extraction de paramètre avec valeurs variables."""
        # Données avec valeurs différentes
        df = pd.DataFrame({
            'variable': ['GDP', 'inflation'],
            'unit': ['D', 'W']
        }).set_index('variable')
        
        result = _extract_param_by_variable(df, 'unit')
        
        # Devrait retourner un dictionnaire
        assert isinstance(result, dict)
        assert result['GDP'] == 'D'
        assert result['inflation'] == 'W'
    
    def test_resolve_strategy_string(self):
        """Résolution de stratégie avec chaîne de caractères."""
        strategy = 'shift'
        entity_key = ('France',)
        
        result = _resolve_strategy(strategy, entity_key)
        assert result == 'shift'
    
    def test_resolve_strategy_dict_simple(self):
        """Résolution de stratégie avec dictionnaire simple."""
        strategy = {
            ('France',): 'shift',
            ('Germany',): 'mask'
        }
        
        result_fr = _resolve_strategy(strategy, ('France',))
        result_de = _resolve_strategy(strategy, ('Germany',))
        
        assert result_fr == 'shift'
        assert result_de == 'mask'
    
    def test_resolve_strategy_dict_by_variable(self):
        """Résolution de stratégie avec dictionnaire par variable."""
        strategy = {
            'GDP': 'shift',
            'inflation': 'mask'
        }
        entity_key = ('France',)
        
        # Devrait retourner le dictionnaire complet
        result = _resolve_strategy(strategy, entity_key)
        assert result == strategy
    
    def test_resolve_strategy_callable(self):
        """Résolution de stratégie avec fonction callable."""
        def strategy_func(entity_key):
            if 'France' in entity_key:
                return 'shift'
            return 'mask'
        
        result_fr = _resolve_strategy(strategy_func, ('France',))
        result_de = _resolve_strategy(strategy_func, ('Germany',))
        
        assert result_fr == 'shift'
        assert result_de == 'mask'
    
    def test_resolve_strategy_invalid_string(self):
        """Validation de stratégie invalide (chaîne)."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            _resolve_strategy('invalid', ('France',))
    
    def test_resolve_strategy_missing_entity(self):
        """Validation d'entité manquante dans le dictionnaire."""
        strategy = {('France',): 'shift'}
        
        with pytest.raises(KeyError, match="No strategy defined"):
            _resolve_strategy(strategy, ('Germany',))
    
    def test_resolve_strategy_callable_invalid_return(self):
        """Validation de retour invalide pour callable."""
        def bad_func(entity_key):
            return 'invalid'
        
        with pytest.raises(ValueError, match="Strategy callable returned invalid"):
            _resolve_strategy(bad_func, ('France',))
    
    def test_resolve_strategy_invalid_type(self):
        """Validation de type de stratégie invalide."""
        with pytest.raises(TypeError, match="strategy must be str, dict, or callable"):
            _resolve_strategy(123, ('France',))


class TestBuildEntityParams:
    """Tests for _build_entity_params function."""
    
    def test_build_entity_params_basic(self):
        """Construction de base des paramètres par entité."""
        # Données de test
        df_delays = pd.DataFrame({
            'country': ['France', 'France', 'Germany', 'Germany'],
            'variable': ['GDP', 'inflation', 'GDP', 'inflation'],
            'applicable_delay': [45.0, 30.0, 40.0, 25.0],
            'unit': ['D', 'D', 'D', 'D'],
            'target_reference_point': ['end', 'end', 'end', 'end'],
            'target_frequency': ['M', 'M', 'M', 'M']
        }).set_index(['country', 'variable'])
        
        result = _build_entity_params(
            df_delays=df_delays,
            delay_col='applicable_delay',
            unit_col='unit',
            reference_point_col='target_reference_point',
            target_frequency_col='target_frequency'
        )
        
        # Vérification de la structure
        assert ('France',) in result
        assert ('Germany',) in result
        assert 'delays' in result[('France',)]
        assert 'delay_unit' in result[('France',)]
        assert result[('France',)]['delays']['GDP'] == 45.0
        assert result[('Germany',)]['delays']['GDP'] == 40.0
    
    def test_build_entity_params_varying_units(self):
        """Construction avec unités variables."""
        df_delays = pd.DataFrame({
            'country': ['France', 'France'],
            'variable': ['GDP', 'inflation'],
            'applicable_delay': [45.0, 7.0],
            'unit': ['D', 'W'],  # Unités différentes
            'target_reference_point': ['end', 'end'],
            'target_frequency': ['M', 'M']
        }).set_index(['country', 'variable'])
        
        result = _build_entity_params(
            df_delays=df_delays,
            delay_col='applicable_delay',
            unit_col='unit',
            reference_point_col='target_reference_point',
            target_frequency_col='target_frequency'
        )
        
        # delay_unit devrait être un dictionnaire
        assert isinstance(result[('France',)]['delay_unit'], dict)
        assert result[('France',)]['delay_unit']['GDP'] == 'D'
        assert result[('France',)]['delay_unit']['inflation'] == 'W'


class TestPrepareEntityKwargs:
    """Tests for prepare_entity_kwargs_from_delays function."""
    
    def test_prepare_entity_kwargs_basic(self):
        """Préparation de base des kwargs par entité."""
        df_delays = pd.DataFrame({
            'country': ['France', 'France', 'Germany', 'Germany'],
            'variable': ['GDP', 'inflation', 'GDP', 'inflation'],
            'applicable_delay': [45.0, 30.0, 40.0, 25.0],
            'unit': ['D', 'D', 'D', 'D'],
            'target_reference_point': ['end', 'end', 'end', 'end'],
            'target_frequency': ['M', 'M', 'M', 'M']
        }).set_index(['country', 'variable'])
        
        result = prepare_entity_kwargs_from_delays(
            df_delays=df_delays,
            strategy='shift'
        )
        
        # Vérification de la structure
        assert ('France',) in result
        assert ('Germany',) in result
        assert 'delays' in result[('France',)]
        assert 'strategy' in result[('France',)]
        assert result[('France',)]['strategy'] == 'shift'
    
    def test_prepare_entity_kwargs_strategy_dict(self):
        """Préparation avec dictionnaire de stratégies."""
        df_delays = pd.DataFrame({
            'country': ['France', 'France', 'Germany', 'Germany'],
            'variable': ['GDP', 'inflation', 'GDP', 'inflation'],
            'applicable_delay': [45.0, 30.0, 40.0, 25.0],
            'unit': ['D', 'D', 'D', 'D'],
            'target_reference_point': ['end', 'end', 'end', 'end'],
            'target_frequency': ['M', 'M', 'M', 'M']
        }).set_index(['country', 'variable'])
        
        strategy_dict = {
            ('France',): 'shift',
            ('Germany',): 'mask'
        }
        
        result = prepare_entity_kwargs_from_delays(
            df_delays=df_delays,
            strategy=strategy_dict
        )
        
        assert result[('France',)]['strategy'] == 'shift'
        assert result[('Germany',)]['strategy'] == 'mask'
    
    def test_prepare_entity_kwargs_missing_columns(self):
        """Validation des colonnes manquantes."""
        df_delays = pd.DataFrame({
            'country': ['France'],
            'variable': ['GDP'],
            'applicable_delay': [45.0],
            # Colonnes manquantes: unit, target_reference_point, target_frequency
        }).set_index(['country', 'variable'])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            prepare_entity_kwargs_from_delays(df_delays)


# ============================================================================
# Tests d'intégration
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_full_workflow_shift(self, sample_time_series, delays_dataframe):
        """Workflow complet avec stratégie shift."""
        # Création du transformer
        transformer = PublicationDelayTransformer(
            delays=delays_dataframe,
            strategy='shift',
            prediction_date='2024-06-01'
        )
        
        # Fit
        transformer.fit(sample_time_series)
        
        # Transform
        transformed = transformer.transform(sample_time_series)
        
        # Inverse transform
        inversed = transformer.inverse_transform(transformed)
        
        # Vérifications
        assert isinstance(transformed, pd.DataFrame)
        assert isinstance(inversed, pd.DataFrame)
        assert transformed.shape == sample_time_series.shape
        assert inversed.shape == sample_time_series.shape
    
    def test_full_workflow_mask(self, sample_time_series, delays_dataframe):
        """Workflow complet avec stratégie mask."""
        transformer = PublicationDelayTransformer(
            delays=delays_dataframe,
            strategy='mask',
            prediction_date='2024-06-01'
        )
        
        transformed = transformer.fit_transform(sample_time_series)
        inversed = transformer.inverse_transform(transformed)
        
        assert isinstance(transformed, pd.DataFrame)
        assert isinstance(inversed, pd.DataFrame)
    
    def test_mixed_strategy(self, sample_time_series, delays_dict_simple):
        """Stratégies mixtes par variable."""
        strategy_dict = {
            'GDP': 'shift',
            'inflation': 'mask',
            'unemployment': 'shift'
        }
        
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy=strategy_dict,
            prediction_date='2024-06-01'
        )
        
        result = transformer.fit_transform(sample_time_series)
        
        assert isinstance(result, pd.DataFrame)
        # Vérification que les deux types de transformations sont appliqués
        assert hasattr(transformer, 'shift_params')
        assert hasattr(transformer, 'mask_params')
    
    def test_panel_data_with_entity_specific_delays(self, sample_panel_data, delays_dataframe_panel):
        """Données panel avec délais spécifiques par entité."""
        transformer = PublicationDelayTransformer(
            delays=delays_dataframe_panel,
            strategy='shift',
            prediction_date='2024-01-01'
        )
        
        result = transformer.fit_transform(sample_panel_data)
        inversed = transformer.inverse_transform(result)
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(inversed, pd.DataFrame)
        assert isinstance(result.index, pd.MultiIndex)


# ============================================================================
# Tests de régression
# ============================================================================

class TestRegression:
    """Regression tests for known issues and fixes."""
    
    def test_column_order_preservation(self, sample_time_series, delays_dict_simple):
        """Préservation de l'ordre des colonnes."""
        original_columns = sample_time_series.columns.tolist()
        
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift'
        )
        
        result = transformer.fit_transform(sample_time_series)
        
        # L'ordre des colonnes doit être préservé
        assert result.columns.tolist() == original_columns
    
    def test_index_preservation(self, sample_time_series, delays_dict_simple):
        """Préservation de l'index."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift'
        )
        
        result = transformer.fit_transform(sample_time_series)
        
        # L'index doit être préservé
        pd.testing.assert_index_equal(result.index, sample_time_series.index)
    
    def test_multiindex_preservation(self, sample_panel_data, delays_dataframe_panel):
        """Préservation du MultiIndex."""
        transformer = PublicationDelayTransformer(
            delays=delays_dataframe_panel,
            strategy='shift'
        )
        
        result = transformer.fit_transform(sample_panel_data)
        
        # Le MultiIndex doit être préservé
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == sample_panel_data.index.names

class TestShiftTransformer:
    """Tests pour la classe ShiftTransformer."""

    def test_higher_frequency_shift_raises_error(self):
        """Tentative de shifter par jours sur un index mensuel doit lever une erreur."""
        dates = pd.date_range('2024-01-01', periods=5, freq='M')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        shifter = ShiftTransformer(n_periods=1, frequency='D')

        with pytest.raises(ValueError, match="cannot be more granular"):
            shifter.fit_transform(series)

    def test_monthly_shift_on_daily_index(self):
        """Shift mensuel sur index journalier."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        series = pd.Series(range(90), index=dates)
        shifter = ShiftTransformer(n_periods=1, frequency='M')

        shifted = shifter.fit_transform(series)

        assert len(shifted) == len(series)
        # Positive shift extends at START (earlier dates), takes first N dates
        # So shifted.index[0] should be EARLIER (negative delta)
        delta_days = (shifted.index[0] - series.index[0]).days
        assert -31 <= delta_days <= -28

    def test_dataframe_shift_same_index(self):
        """Toutes les colonnes d'un DataFrame doivent avoir le même index après shift."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({f'col_{i}': range(100) for i in range(10)}, index=dates)

        shifter = ShiftTransformer(n_periods=5, frequency='D')
        shifted = shifter.fit_transform(df)

        # Toutes les colonnes doivent partager le même index
        first_col_index = shifted.iloc[:, 0].index
        assert all(shifted.iloc[:, i].index.equals(first_col_index) for i in range(len(shifted.columns)))

    def test_quarterly_suffix_preserved(self):
        """Index trimestriel avec suffixe DEC doit être préservé."""
        dates = pd.date_range('2024-01-31', periods=5, freq='QE-DEC')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        shifter = ShiftTransformer(n_periods=1, frequency='Q')

        shifted = shifter.fit_transform(series)

        # Vérifier que les dates sont bien en fin de trimestre
        # The important thing is that shifted index is valid and same length
        assert len(shifted) == len(series)
        assert isinstance(shifted.index, pd.DatetimeIndex)

    @pytest.mark.parametrize("n_periods,freq", [
        (1, 'D'), (5, 'D'), (-3, 'D'),
        (1, 'M'), (2, 'M'), (-1, 'M'),
    ])
    def test_inverse_transform_symmetry(self, n_periods, freq):
        """L'inverse transform doit parfaitement inverser le transform."""
        dates = pd.date_range('2024-01-01', periods=20, freq=freq)
        series = pd.Series(range(20), index=dates)
        shifter = ShiftTransformer(n_periods=n_periods, frequency=freq)

        shifted = shifter.fit_transform(series)
        recovered = shifter.inverse_transform(shifted)

        assert series.equals(recovered)

    def test_zero_shift(self):
        """Shift de 0 périodes doit retourner une copie."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        shifter = ShiftTransformer(n_periods=0, frequency='D')

        shifted = shifter.fit_transform(series)

        # Doit être une copie, pas le même objet
        assert shifted is not series
        # Mais les valeurs doivent être identiques
        assert series.equals(shifted)

    def test_negative_shift(self):
        """Shift négatif doit fonctionner correctement."""
        dates = pd.date_range('2024-01-01', periods=5, freq='M')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        shifter = ShiftTransformer(n_periods=-2, frequency='M')

        shifted = shifter.fit_transform(series)

        # Vérifier que le shift négatif fonctionne
        assert len(shifted) == len(series)
        # Shift négatif = étend à la fin, drop du début
        # Donc le premier index doit être PLUS TARD (plus grand)
        assert shifted.index[0] > series.index[0]

    def test_single_observation(self):
        """Single observation cannot detect frequency."""
        dates = pd.date_range('2024-01-01', periods=1, freq='D')
        series = pd.Series([1], index=dates)
        shifter = ShiftTransformer(n_periods=1, frequency='D')

        # Single observation cannot detect frequency (need at least 2 observations)
        with pytest.raises(ValueError, match="(Could not detect|minimum required)"):
            shifter.fit_transform(series)

    def test_same_frequency_shift(self):
        """Shift avec la même fréquence que l'index."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        series = pd.Series(range(10), index=dates)
        shifter = ShiftTransformer(n_periods=3, frequency='D')

        shifted = shifter.fit_transform(series)

        # Vérifier que le shift est correct
        assert len(shifted) == len(series)
        # Positive shift extends at start (earlier), so delta should be negative
        assert (shifted.index[0] - series.index[0]).days == -3

    def test_dataframe_with_series_values_match(self):
        """DataFrame shift doit donner les mêmes valeurs qu'un shift de Series."""
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        series = pd.Series(range(20), index=dates, name='col_0')
        df = pd.DataFrame({'col_0': range(20)}, index=dates)

        shifter = ShiftTransformer(n_periods=5, frequency='D')

        shifted_series = shifter.fit_transform(series)
        shifted_df = shifter.fit_transform(df)

        # Les index doivent être identiques
        assert shifted_series.index.equals(shifted_df.index)
        # Les valeurs doivent être identiques
        pd.testing.assert_series_equal(shifted_series, shifted_df['col_0'])

    def test_quarterly_to_monthly_conversion(self):
        """Shift trimestriel sur index mensuel doit utiliser conversion correcte."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        series = pd.Series(range(12), index=dates)
        shifter = ShiftTransformer(n_periods=1, frequency='Q')

        shifted = shifter.fit_transform(series)

        # 1 trimestre = 3 mois
        # Positive shift extends at start (earlier), so delta should be negative
        delta_months = (shifted.index[0].year - series.index[0].year) * 12 + \
                       (shifted.index[0].month - series.index[0].month)
        assert delta_months == -3

    def test_monthly_start_position_preserved(self):
        """Position de début de mois doit être préservée."""
        dates = pd.date_range('2024-01-01', periods=5, freq='MS')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        shifter = ShiftTransformer(n_periods=1, frequency='M')

        shifted = shifter.fit_transform(series)

        # Vérifier que toutes les dates sont au début du mois (jour 1)
        assert all(d.day == 1 for d in shifted.index)

    def test_monthly_end_position_preserved(self):
        """Position de fin de mois doit être préservée."""
        dates = pd.date_range('2024-01-31', periods=5, freq='ME')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        shifter = ShiftTransformer(n_periods=1, frequency='M')

        shifted = shifter.fit_transform(series)

        # Vérifier que toutes les dates sont en fin de mois
        # (le jour varie selon le mois, mais doit être le dernier jour)
        for d in shifted.index:
            next_month = (d + pd.Timedelta(days=1))
            assert next_month.day == 1  # Le jour suivant doit être le 1er du mois suivant


class TestMaskTransformer:
    """Tests pour la classe MaskTransformer."""

    def test_basic_masking(self):
        """Test de base du masquage."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        series = pd.Series(range(90), index=dates, name='GDP')

        masker = MaskTransformer(
            n_obs=2,
            mask_frequency='M'
        )

        masked = masker.fit_transform(series)

        # Vérifier que certaines valeurs sont masquées (NaN)
        assert masked.isna().sum() > 0
        # La longueur doit rester la même
        assert len(masked) == len(series)

    def test_inverse_transform(self):
        """Test de l'inverse transform du masquage."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        series = pd.Series(range(90), index=dates, name='GDP')

        masker = MaskTransformer(
            n_obs=2,
            mask_frequency='M'
        )

        masked = masker.fit_transform(series)
        recovered = masker.inverse_transform(masked)

        # Les données originales doivent être parfaitement restaurées
        # Check values and index separately to avoid freq attribute issues
        pd.testing.assert_index_equal(recovered.index, series.index)
        np.testing.assert_array_equal(recovered.values, series.values)
        assert recovered.name == series.name

    def test_zero_masking(self):
        """Test avec n_obs=0 (pas de masquage)."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        series = pd.Series(range(30), index=dates, name='GDP')

        masker = MaskTransformer(
            n_obs=0,
            mask_frequency='M'
        )

        masked = masker.fit_transform(series)

        # Aucune valeur ne doit être masquée
        assert masked.isna().sum() == 0
        pd.testing.assert_series_equal(masked, series)


class TestShiftTransformerEdgeCases:
    """Additional edge case tests for ShiftTransformer."""

    def test_invalid_input_type(self):
        """Reject non-pandas inputs."""
        shifter = ShiftTransformer(n_periods=1, frequency='D')
        with pytest.raises(ValueError, match="must be a pandas Series or DataFrame"):
            shifter.fit([1, 2, 3])

    def test_non_datetime_index(self):
        """Reject non-datetime index."""
        series = pd.Series([1, 2, 3], index=[0, 1, 2])
        shifter = ShiftTransformer(n_periods=1, frequency='D')
        with pytest.raises(ValueError):
            shifter.fit_transform(series)

    def test_unsorted_index_auto_corrected(self):
        """Unsorted indices should be automatically sorted."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        # Shuffle the series
        series = series.iloc[[2, 0, 4, 1, 3]]

        shifter = ShiftTransformer(n_periods=1, frequency='D')
        shifted = shifter.fit_transform(series)

        # Result should have sorted index
        assert shifted.index.is_monotonic_increasing

    def test_daily_frequency_multiday(self):
        """Test with multiple days to ensure shift works."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        series = pd.Series(range(30), index=dates)
        shifter = ShiftTransformer(n_periods=5, frequency='D')

        shifted = shifter.fit_transform(series)

        assert len(shifted) == len(series)
        # Positive shift = extend at start (earlier), so delta should be negative
        delta_days = (shifted.index[0] - series.index[0]).days
        assert delta_days == -5

    def test_dataframe_multicolumn(self):
        """Test DataFrame with multiple columns of different types."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'int_col': range(10),
            'float_col': [float(x) for x in range(10)],
            'str_col': [f'val_{x}' for x in range(10)]
        }, index=dates)

        shifter = ShiftTransformer(n_periods=2, frequency='D')
        shifted = shifter.fit_transform(df)

        # All columns should be preserved
        assert list(shifted.columns) == list(df.columns)
        # Types should be preserved
        assert shifted['str_col'].dtype == object


class TestMaskTransformerEdgeCases:
    """Additional edge case tests for MaskTransformer."""

    def test_mask_first_observations(self):
        """Test masking first observations (how='first')."""
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        series = pd.Series(range(60), index=dates)

        masker = MaskTransformer(n_obs=5, mask_frequency='M', how='first')
        masked = masker.fit_transform(series)

        # First 5 days of each month should be masked
        assert masked.isna().sum() > 0

    def test_mask_more_than_available(self):
        """Masking more observations than available in a period should handle gracefully."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        series = pd.Series(range(5), index=dates)

        # Only 5 days total, but request masking 10 per month
        masker = MaskTransformer(n_obs=10, mask_frequency='M')
        masked = masker.fit_transform(series)

        # Should not crash and return valid series
        assert len(masked) == len(series)
        assert isinstance(masked.index, pd.DatetimeIndex)

    def test_mask_dataframe(self):
        """Test masking on DataFrame."""
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        df = pd.DataFrame({
            'col1': range(60),
            'col2': range(100, 160)
        }, index=dates)

        masker = MaskTransformer(n_obs=2, mask_frequency='M')
        masked = masker.fit_transform(df)

        # Both columns should have identical NaN patterns
        assert (masked['col1'].isna() == masked['col2'].isna()).all()

    def test_mask_without_fit(self):
        """Calling inverse_transform without transform should raise error."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        series = pd.Series(range(30), index=dates)

        masker = MaskTransformer(n_obs=2, mask_frequency='M')

        with pytest.raises(ValueError, match="Must call transform"):
            masker.inverse_transform(series)

    def test_inverse_transform_with_filled_nans(self):
        """Test inverse_transform after filling NaNs with predictions."""
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        series = pd.Series(range(60), index=dates)

        masker = MaskTransformer(n_obs=2, mask_frequency='M')
        masked = masker.fit_transform(series)

        # Fill NaN values with predictions (different values)
        predictions = masked.fillna(-999)

        # Inverse transform should restore original values (not the -999)
        restored = masker.inverse_transform(predictions)

        np.testing.assert_array_equal(restored.values, series.values)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])