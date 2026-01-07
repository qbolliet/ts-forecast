"""Temporal data validation utilities.

This module provides functions for validating and preparing time series
and panel data structures, including validation of datetime indexes,
uniqueness checks, and proper index configuration.
"""

# Import des fonctions de validation temporelle
from .validation import validate_temporal_data, restore_original_structure


__all__ = [
    # Fonctions de validation
    'validate_temporal_data',
    'restore_original_structure'
]
