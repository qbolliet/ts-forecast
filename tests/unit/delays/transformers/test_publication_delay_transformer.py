"""Unit tests for ``PublicationDelayTransformer`` (``tsforecast.delays.transformers``).

Moved as-is from ``tests/unit/delays/test_transformers.py`` by prompt D3 (split of
the module tests into one file per tested symbol, mirror rule of
``tests_and_refactoring_prompts.md`` §3). Triage of the inherited failures and
completion of the scenarios: prompt D4.
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from sklearn.utils.validation import check_is_fitted, NotFittedError

# Import du module à tester
from tsforecast.delays.transformers import PublicationDelayTransformer

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
        """Initialisation avec un dictionnaire de délais."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift',
            prediction_date='2024-01-01'
        )
        
        assert transformer.delays == delays_dict_simple
        assert transformer.strategy == 'shift'
        assert transformer.prediction_date == '2024-01-01'
    
    def test_init_with_dataframe(self, delays_dataframe):
        """Initialisation avec un DataFrame de délais."""
        transformer = PublicationDelayTransformer(
            delays=delays_dataframe,
            strategy='mask',
            prediction_date=datetime(2024, 6, 15)
        )
        
        assert isinstance(transformer.delays, pd.DataFrame)
        assert transformer.strategy == 'mask'
        assert isinstance(transformer.prediction_date, datetime)
    
    def test_init_with_strategy_dict(self, delays_dict_simple):
        """Initialisation avec un dictionnaire de stratégies."""
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
        """Initialisation avec des valeurs par défaut."""
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


# ============================================================================
# Tests de vérification des paramètres calculés (shift_params, mask_params)
# ============================================================================

class TestPublicationDelayTransformerParams:
    """Tests de vérification des paramètres shift_params et mask_params."""

    # -------------------------------------------------------------------------
    # Tests des shift_params
    # -------------------------------------------------------------------------

    def test_shift_params_structure(self):
        """Vérification de la structure des shift_params."""
        # Données mensuelles simples
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({'GDP': range(12)}, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 30.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        
        transformer.fit(data)
        
        # Vérification de la structure
        assert 'GDP' in transformer.shift_params
        assert 'n_periods' in transformer.shift_params['GDP']
        assert 'frequency' in transformer.shift_params['GDP']
        assert isinstance(transformer.shift_params['GDP']['n_periods'], int)

    def test_shift_params_n_periods_is_integer(self):
        """Vérification que n_periods est un entier."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'GDP': range(12),
            'inflation': range(12)
        }, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 45.0, 'inflation': 30.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        
        transformer.fit(data)
        
        for col, params in transformer.shift_params.items():
            assert isinstance(params['n_periods'], int), f"n_periods pour {col} n'est pas un entier"

    def test_shift_params_frequency_matches_detected(self):
        """Vérification que la fréquence dans shift_params correspond à la fréquence détectée."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({'GDP': range(12)}, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 30.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        
        transformer.fit(data)
        
        # La fréquence dans shift_params doit correspondre à la fréquence détectée
        assert transformer.shift_params['GDP']['frequency'] == transformer.detected_frequencies_['GDP']

    def test_shift_params_larger_delay_means_more_periods(self):
        """Vérification qu'un délai plus grand implique plus de périodes à shifter."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'short_delay': range(12),
            'long_delay': range(12)
        }, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'short_delay': 15.0, 'long_delay': 60.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        
        transformer.fit(data)
        
        # Plus le délai est grand, plus le n_periods (en valeur absolue) est grand
        # Note: n_periods est négatif (shift vers le passé)
        short_n = transformer.shift_params['short_delay']['n_periods']
        long_n = transformer.shift_params['long_delay']['n_periods']
        
        # En valeur absolue, long_delay doit avoir plus de périodes
        assert abs(long_n) >= abs(short_n), \
            f"Délai long ({long_n}) devrait avoir >= périodes que délai court ({short_n})"

    def test_shift_params_reference_point_end_vs_start(self):
        """Vérification de l'impact du reference_point sur n_periods."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({'GDP': range(12)}, index=dates)
        
        # Transformer avec reference_point='end'
        transformer_end = PublicationDelayTransformer(
            delays={'GDP': 45.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        transformer_end.fit(data)
        
        # Transformer avec reference_point='start'
        transformer_start = PublicationDelayTransformer(
            delays={'GDP': 45.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='start'
        )
        transformer_start.fit(data)
        
        # Les n_periods doivent être différents (environ 1 période de différence)
        n_end = transformer_end.shift_params['GDP']['n_periods']
        n_start = transformer_start.shift_params['GDP']['n_periods']
        
        # Avec reference_point='end', le délai effectif est plus court
        # donc on devrait shifter moins (ou la différence ~1 période)
        assert n_end != n_start or abs(n_end - n_start) <= 1

    def test_shift_params_zero_delay(self):
        """Vérification du comportement avec un délai de 0."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({'GDP': range(12)}, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 0.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        
        transformer.fit(data)
        
        # Avec un délai de 0, n_periods devrait être proche de 0 ou positif
        # (dépend de la position dans la période)
        n_periods = transformer.shift_params['GDP']['n_periods']
        assert isinstance(n_periods, int)

    def test_shift_params_all_columns_present(self):
        """Vérification que toutes les colonnes avec délai sont dans shift_params."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'GDP': range(12),
            'inflation': range(12),
            'unemployment': range(12)
        }, index=dates)
        
        delays = {'GDP': 30.0, 'inflation': 45.0, 'unemployment': 15.0}
        
        transformer = PublicationDelayTransformer(
            delays=delays,
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        
        transformer.fit(data)
        
        # Toutes les colonnes avec délai doivent être dans shift_params
        for col in delays.keys():
            assert col in transformer.shift_params, f"Colonne {col} absente de shift_params"

    def test_shift_params_different_delay_units(self):
        """Vérification du calcul avec différentes unités de délai."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({'GDP': range(12)}, index=dates)
        
        # Même délai exprimé en jours et en heures
        delay_days = 30.0
        delay_hours = 30.0 * 24  # 720 heures = 30 jours
        
        transformer_days = PublicationDelayTransformer(
            delays={'GDP': delay_days},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        transformer_days.fit(data)
        
        transformer_hours = PublicationDelayTransformer(
            delays={'GDP': delay_hours},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='h',
            reference_point='end'
        )
        transformer_hours.fit(data)
        
        # Les n_periods devraient être identiques (même délai effectif)
        assert transformer_days.shift_params['GDP']['n_periods'] == \
               transformer_hours.shift_params['GDP']['n_periods']

    # -------------------------------------------------------------------------
    # Tests des mask_params
    # -------------------------------------------------------------------------

    def test_mask_params_structure(self):
        """Vérification de la structure des mask_params."""
        # Données journalières
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        data = pd.DataFrame({'GDP': range(90)}, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 10.0},
            strategy='mask',
            prediction_date='2024-03-15',
            delay_unit='D',
            reference_point='end',
            target_frequency='M'
        )
        
        transformer.fit(data)
        
        # Vérification de la structure
        # Note: si can_mask est False, la colonne sera dans shift_params
        if 'GDP' in transformer.mask_params:
            assert 'n_obs' in transformer.mask_params['GDP']
            assert 'mask_frequency' in transformer.mask_params['GDP']
            assert 'how' in transformer.mask_params['GDP']

    def test_mask_params_n_obs_is_positive_integer(self):
        """Vérification que n_obs est un entier positif."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        data = pd.DataFrame({'GDP': range(90)}, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 5.0},
            strategy='mask',
            prediction_date='2024-03-15',
            delay_unit='D',
            reference_point='end',
            target_frequency='M'
        )
        
        transformer.fit(data)
        
        if 'GDP' in transformer.mask_params:
            n_obs = transformer.mask_params['GDP']['n_obs']
            assert isinstance(n_obs, int), "n_obs doit être un entier"
            assert n_obs >= 0, "n_obs doit être positif ou nul"

    def test_mask_params_how_is_last(self):
        """Vérification que 'how' est toujours 'last' (comportement par défaut)."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        data = pd.DataFrame({'GDP': range(90)}, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 5.0},
            strategy='mask',
            prediction_date='2024-03-15',
            delay_unit='D',
            reference_point='end',
            target_frequency='M'
        )
        
        transformer.fit(data)
        
        if 'GDP' in transformer.mask_params:
            assert transformer.mask_params['GDP']['how'] == 'last'

    def test_mask_params_larger_delay_means_more_obs(self):
        """Vérification qu'un délai plus grand implique plus d'observations à masquer."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        data = pd.DataFrame({
            'short_delay': range(90),
            'long_delay': range(90)
        }, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'short_delay': 5.0, 'long_delay': 15.0},
            strategy='mask',
            prediction_date='2024-03-15',
            delay_unit='D',
            reference_point='end',
            target_frequency='M'
        )
        
        transformer.fit(data)
        
        # Vérification si les colonnes sont dans mask_params
        # (sinon elles ont été déplacées vers shift_params car can_mask=False)
        if 'short_delay' in transformer.mask_params and 'long_delay' in transformer.mask_params:
            short_n = transformer.mask_params['short_delay']['n_obs']
            long_n = transformer.mask_params['long_delay']['n_obs']
            
            assert long_n >= short_n, \
                f"Délai long ({long_n} obs) devrait masquer >= que délai court ({short_n} obs)"

    def test_mask_params_target_frequency_used(self):
        """Vérification que la target_frequency est utilisée dans mask_params."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        data = pd.DataFrame({'GDP': range(90)}, index=dates)
        
        target_freq = 'M'
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 5.0},
            strategy='mask',
            prediction_date='2024-03-15',
            delay_unit='D',
            reference_point='end',
            target_frequency=target_freq
        )
        
        transformer.fit(data)
        
        if 'GDP' in transformer.mask_params:
            # La mask_frequency doit correspondre à la target_frequency normalisée
            assert 'mask_frequency' in transformer.mask_params['GDP']

    def test_mask_fallback_to_shift_when_cannot_mask(self):
        """Vérification du fallback vers shift quand le masquage n'est pas possible."""
        # Données mensuelles avec un délai très long (impossible de masquer sans tout rendre NaN)
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({'GDP': range(12)}, index=dates)
        
        # Délai de 60 jours avec fréquence mensuelle = impossible de masquer
        transformer = PublicationDelayTransformer(
            delays={'GDP': 60.0},
            strategy='mask',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end',
            target_frequency='M'
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            transformer.fit(data)
        
        # La colonne doit être dans shift_params (fallback) ou mask_params selon le calcul
        assert 'GDP' in transformer.shift_params or 'GDP' in transformer.mask_params

    # -------------------------------------------------------------------------
    # Tests de cohérence entre delays fournis et params calculés
    # -------------------------------------------------------------------------

    def test_shift_params_consistent_with_delays(self):
        """Vérification de la cohérence entre les délais fournis et les paramètres calculés."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'fast': range(12),
            'medium': range(12),
            'slow': range(12)
        }, index=dates)
        
        # Délais croissants
        delays = {'fast': 10.0, 'medium': 30.0, 'slow': 60.0}
        
        transformer = PublicationDelayTransformer(
            delays=delays,
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        
        transformer.fit(data)
        
        # Vérification de l'ordre : plus le délai est grand, plus le shift est important
        n_fast = abs(transformer.shift_params['fast']['n_periods'])
        n_medium = abs(transformer.shift_params['medium']['n_periods'])
        n_slow = abs(transformer.shift_params['slow']['n_periods'])
        
        assert n_fast <= n_medium <= n_slow, \
            f"Ordre incohérent: fast={n_fast}, medium={n_medium}, slow={n_slow}"

    def test_mask_params_consistent_with_delays(self):
        """Vérification de la cohérence entre les délais fournis et les paramètres de masquage."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        data = pd.DataFrame({
            'fast': range(90),
            'medium': range(90),
            'slow': range(90)
        }, index=dates)
        
        # Délais croissants mais suffisamment petits pour permettre le masquage
        delays = {'fast': 3.0, 'medium': 7.0, 'slow': 12.0}
        
        transformer = PublicationDelayTransformer(
            delays=delays,
            strategy='mask',
            prediction_date='2024-03-15',
            delay_unit='D',
            reference_point='end',
            target_frequency='M'
        )
        
        transformer.fit(data)
        
        # Vérification de l'ordre pour les colonnes dans mask_params
        mask_cols = [col for col in ['fast', 'medium', 'slow'] if col in transformer.mask_params]
        
        if len(mask_cols) >= 2:
            for i in range(len(mask_cols) - 1):
                col1, col2 = mask_cols[i], mask_cols[i + 1]
                n1 = transformer.mask_params[col1]['n_obs']
                n2 = transformer.mask_params[col2]['n_obs']
                # Le délai plus grand devrait avoir plus d'observations à masquer
                assert n1 <= n2, f"Ordre incohérent: {col1}={n1}, {col2}={n2}"

    def test_params_with_dict_delay_unit(self):
        """Vérification des paramètres avec delay_unit spécifié par variable."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'GDP': range(12),
            'inflation': range(12)
        }, index=dates)
        
        # Délais avec unités différentes mais équivalents
        # 30 jours pour GDP, 4 semaines (~28 jours) pour inflation
        transformer = PublicationDelayTransformer(
            delays={'GDP': 30.0, 'inflation': 4.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit={'GDP': 'D', 'inflation': 'W'},
            reference_point='end'
        )
        
        transformer.fit(data)
        
        # Les deux devraient avoir des n_periods similaires (30 jours vs 28 jours)
        n_gdp = abs(transformer.shift_params['GDP']['n_periods'])
        n_inflation = abs(transformer.shift_params['inflation']['n_periods'])
        
        # Différence d'au plus 1 période attendue
        assert abs(n_gdp - n_inflation) <= 1, \
            f"Différence trop importante: GDP={n_gdp}, inflation={n_inflation}"

    def test_params_with_dict_reference_point(self):
        """Vérification des paramètres avec reference_point spécifié par variable."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'GDP': range(12),
            'inflation': range(12)
        }, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 45.0, 'inflation': 45.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point={'GDP': 'end', 'inflation': 'start'}
        )
        
        transformer.fit(data)
        
        # Les n_periods doivent différer d'environ 1 période
        n_gdp = transformer.shift_params['GDP']['n_periods']
        n_inflation = transformer.shift_params['inflation']['n_periods']
        
        # Avec start, le délai effectif est plus long, donc plus de périodes
        assert n_gdp != n_inflation or abs(n_gdp - n_inflation) <= 1


# ============================================================================
# Rapport d'ajustement (fit_report_)
# ============================================================================

class TestFitReport:
    """``fit_report_`` gathers what ``fit`` resolved, beyond the warnings it emits."""

    @staticmethod
    def _monthly(columns=('GDP', 'CPI'), periods=12):
        index = pd.date_range('2023-01-01', periods=periods, freq='MS')
        return pd.DataFrame({col: range(periods) for col in columns}, index=index)

    @staticmethod
    def _delays(columns=('GDP', 'CPI'), delays=(45.0, 20.0), **extra):
        data = {'column': list(columns), 'delay': list(delays), 'unit': ['D'] * len(columns),
                'reference_point': ['start'] * len(columns), 'frequency': ['Q'] * len(columns)}
        data.update(extra)
        return pd.DataFrame(data)

    def test_report_exists_only_after_fit(self):
        transformer = PublicationDelayTransformer(delays=self._delays(), prediction_date='2023-12-15')
        assert not hasattr(transformer, 'fit_report_')
        transformer.fit(self._monthly())
        assert transformer.fit_report_.prediction_date == datetime(2023, 12, 15)

    def test_report_matches_the_fitted_parameters(self):
        transformer = PublicationDelayTransformer(delays=self._delays(), prediction_date='2023-12-15')
        report = transformer.fit(self._monthly()).fit_report_
        records = {record.column: record for record in report.columns}
        assert set(records) == set(transformer.shift_params)
        for col, params in transformer.shift_params.items():
            assert records[col].strategy == 'shift'
            assert records[col].n_periods == params['n_periods']
            assert records[col].frequency == params['frequency']
        assert (records['GDP'].delay, records['GDP'].delay_unit, records['GDP'].reference_point) == (45.0, 'D', 'start')

    def test_columns_unaffected_and_ignored(self):
        X = self._monthly(columns=('GDP', 'Z'))
        delays = self._delays(columns=('GDP', 'OLD'), delays=(45.0, 5.0))
        with pytest.warns(UserWarning):
            report = PublicationDelayTransformer(delays=delays, prediction_date='2023-12-15').fit(X).fit_report_
        assert report.columns_unaffected == ('Z',)
        assert report.columns_ignored == ('OLD',)

    def test_parameter_sources_inferred_and_explicit(self):
        X = self._monthly()
        inferred = PublicationDelayTransformer(
            delays=self._delays(), prediction_date='2023-12-15').fit(X).fit_report_
        assert {r.reference_point_source for r in inferred.columns} == {'inferred'}
        assert {r.delay_unit_source for r in inferred.columns} == {'inferred'}
        explicit = PublicationDelayTransformer(
            delays=self._delays(), prediction_date='2023-12-15', reference_point='end').fit(X).fit_report_
        assert {r.reference_point_source for r in explicit.columns} == {'explicit'}
        assert {r.reference_point for r in explicit.columns} == {'end'}

    def test_explicit_dict_wins_for_its_columns_only(self):
        X = self._monthly()
        report = PublicationDelayTransformer(
            delays=self._delays(), prediction_date='2023-12-15', delay_unit={'GDP': 'W'}
        ).fit(X).fit_report_
        sources = {r.column: (r.delay_unit, r.delay_unit_source) for r in report.columns}
        assert sources == {'GDP': ('W', 'explicit'), 'CPI': ('D', 'inferred')}

    def test_defaults_imputed_are_listed(self):
        X = self._monthly()
        delays = self._delays().drop(columns='reference_point')
        defaults = {'delay': 1.0, 'unit': 'D', 'reference_point': 'end'}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            report = PublicationDelayTransformer(
                delays=delays, prediction_date='2023-12-15', default_values=defaults).fit(X).fit_report_
        assert report.defaults_imputed == (('GDP', 'reference_point'), ('CPI', 'reference_point'))
        assert {r.reference_point_source for r in report.columns} == {'default'}

    def test_mask_strategy_records_the_masked_observations(self):
        X = self._monthly()
        delays = self._delays(delays=(20.0, 20.0))
        report = PublicationDelayTransformer(
            delays=delays, strategy='mask', prediction_date='2023-12-15').fit(X).fit_report_
        for record in report.columns:
            assert record.strategy == 'mask'
            assert record.n_obs == 1 and record.n_periods is None
            assert (record.target_frequency, record.target_frequency_source) == ('Q', 'inferred')
            assert record.moved_from_mask is False
        assert report.mask_fallbacks == ()

    def test_mask_fallback_is_reported(self):
        X = self._monthly()
        delays = self._delays(delays=(400.0, 20.0))
        with pytest.warns(UserWarning, match="Could not mask the column 'GDP'"):
            transformer = PublicationDelayTransformer(delays=delays, strategy='mask', prediction_date='2023-12-15')
            report = transformer.fit(X).fit_report_
        assert report.mask_fallbacks == ('GDP',)
        records = {record.column: record for record in report.columns}
        assert (records['GDP'].strategy, records['GDP'].moved_from_mask) == ('shift', True)
        assert records['GDP'].n_periods == transformer.shift_params['GDP']['n_periods']
        assert records['CPI'].strategy == 'mask'

    def test_refit_replaces_the_report(self):
        transformer = PublicationDelayTransformer(delays=self._delays(), prediction_date='2023-12-15')
        first = transformer.fit(self._monthly()).fit_report_
        second = transformer.fit(self._monthly(columns=('GDP',))).fit_report_
        assert first is not second
        assert [r.column for r in second.columns] == ['GDP']

    def test_fit_is_logged_at_info_level(self, caplog):
        transformer = PublicationDelayTransformer(delays=self._delays(), prediction_date='2023-12-15')
        with caplog.at_level('INFO', logger='tsforecast.delays.transformers'):
            report = transformer.fit(self._monthly()).fit_report_
        assert report.summary() in caplog.messages

    def test_clone_does_not_copy_the_report(self):
        from sklearn.base import clone
        transformer = PublicationDelayTransformer(delays=self._delays(), prediction_date='2023-12-15')
        transformer.fit(self._monthly())
        assert not hasattr(clone(transformer), 'fit_report_')


# ============================================================================
# Masques nuls ou négatifs (ajout D3 : contrat de MaskTransformer, n_obs >= 0)
# ============================================================================

class TestZeroAndNegativeMasks:
    """A mask with nothing to mask leaves its column unchanged instead of building a ``MaskTransformer``.

    Gold values (monthly column, delays at monthly frequency, so the mask frequency
    equals the index frequency): ``n_obs = ceil((delay - elapsed) / 30)`` with 14
    days elapsed since 1 December 2023 at the prediction date 2023-12-15 and the
    reference point at the period start.
    """

    @staticmethod
    def _monthly() -> pd.DataFrame:
        index = pd.date_range('2023-01-01', periods=12, freq='MS')
        return pd.DataFrame({'GDP': np.arange(12.0), 'CPI': np.arange(12.0) * 2}, index=index)

    @staticmethod
    def _transformer(delay: float) -> PublicationDelayTransformer:
        delays = pd.DataFrame({'column': ['GDP'], 'delay': [delay], 'unit': ['D'],
                               'reference_point': ['start'], 'frequency': ['M']})
        return PublicationDelayTransformer(delays=delays, strategy='mask', prediction_date='2023-12-15')

    @pytest.mark.parametrize(
        'delay',
        [
            # Valeur d'or : ceil((5 - 14) / 30) = 0
            pytest.param(5.0, id='zero'),
            # Valeur d'or : ceil((-40 - 14) / 30) = -1, ramené à 0
            pytest.param(-40.0, id='negative'),
        ],
    )
    def test_n_obs_is_never_negative(self, delay):
        """The number of observations to mask is clamped at zero."""
        transformer = self._transformer(delay).fit(self._monthly())
        assert transformer.mask_params['GDP']['n_obs'] == 0

    @pytest.mark.parametrize('delay', [pytest.param(5.0, id='zero'), pytest.param(-40.0, id='negative')])
    def test_zero_mask_leaves_the_data_unchanged(self, delay):
        """``transform`` returns the data as given: nothing is unpublished at the prediction date."""
        transformer = self._transformer(delay).fit(self._monthly())
        pd.testing.assert_frame_equal(transformer.transform(self._monthly()), self._monthly())

    def test_zero_mask_round_trip(self):
        """``inverse_transform`` of an unchanged frame returns it unchanged."""
        transformer = self._transformer(5.0).fit(self._monthly())
        recovered = transformer.inverse_transform(transformer.transform(self._monthly()))
        pd.testing.assert_frame_equal(recovered, self._monthly())
