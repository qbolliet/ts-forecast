"""
Publication delays module for tsforecast.

This module provides tools for managing and applying publication delays
to time series and panel data using sklearn-compatible transformers.
"""
# Importation des modules
# Importation du module d'inférence des délais de publication à partir de deux jeux de données
from .data_manager import ReleaseDataManager
# Importation du module de calcul du délai de publication à partir des dates de publication de différentes observations
from .delay_calculator import ReleaseDelayCalculator
# Importation du module d'application des délais de publication à des données
from .transformers import (
    ReleaseDelayTransformer,
    create_delay_transformer_from_calculator,
    create_delay_transformer_from_dict
)

__all__ = [
    'ReleaseDataManager',
    'ReleaseDelayCalculator',
    'ReleaseDelayTransformer',
    'create_delay_transformer_from_calculator',
    'create_delay_transformer_from_dict'
]