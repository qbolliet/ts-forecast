"""Unit tests for the module-level helpers of ``tsforecast.delays.transformers``.

Moved as-is from ``tests/unit/delays/test_transformers.py`` by prompt D3:
``_extract_param_by_variable`` and ``_resolve_strategy`` (helpers of
``prepare_entity_kwargs_from_delays`` / ``create_delay_transformer_factory``) and
``_detect_index_components`` (index components read by ``ShiftTransformer`` and
``MaskTransformer``, whose public path is tested in ``test_shift_transformer.py``
and ``test_mask_transformer.py``). Completion: prompt D4.
"""

import pytest
import pandas as pd

# Import du module à tester
from tsforecast.delays.transformers import (
    _extract_param_by_variable,
    _resolve_strategy,
    _detect_index_components
)


class TestMultipliedIndexFrequency:
    """Les transformateurs décalent par périodes d'index entières : un index multiplié est rejeté."""

    def test_detect_index_components(self):
        """Base, position et ancre de l'index (sans multiplicateur)."""
        assert _detect_index_components(pd.date_range('2024-01-01', periods=6, freq='QS')) == ('Q', 'S', 'JAN')

    def test_multiplied_index_is_rejected(self):
        """Un index '2MS' ne doit pas être traité comme mensuel."""
        with pytest.raises(ValueError, match="Multiplied index frequency"):
            _detect_index_components(pd.date_range('2024-01-01', periods=6, freq='2MS'))


# ============================================================================
# Tests des fonctions auxiliaires
# ============================================================================

class TestAuxiliaryFunctions:
    """Tests for auxiliary functions."""
    
    def test_extract_param_by_variable_constant(self):
        """Extraction de paramètre avec valeur constante."""
        df = pd.DataFrame({
            'variable': ['GDP', 'inflation'],
            'unit': ['D', 'D']
        }).set_index('variable')
        
        result = _extract_param_by_variable(df, 'unit')
        
        # Doit retourner la valeur unique
        assert result == 'D'
    
    def test_extract_param_by_variable_varying(self):
        """Extraction de paramètre avec valeurs variables."""
        df = pd.DataFrame({
            'variable': ['GDP', 'inflation'],
            'unit': ['D', 'W']
        }).set_index('variable')
        
        result = _extract_param_by_variable(df, 'unit')
        
        # Doit retourner un dictionnaire
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
        
        # Doit retourner le dictionnaire complet
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
