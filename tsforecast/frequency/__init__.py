"""Frequency detection, conversion, and imputation utilities for time series processing.

This module provides tools for detecting and converting between different time frequencies,
with support for pandas frequency codes, DateOffsets, and user-friendly frequency names.
It also provides mixed-frequency imputation capabilities with provenance tracking.
"""

# Import des classes principales
from .detector import FrequencyDetector, detect_frequency, detect_dataset_frequency, detect_index_frequency, detect_and_parse_index_frequency
from .high_frequency_imputer import HighFrequencyImputer
from .parser import parse_frequency, build_frequency_string
from .provenance import ImputationProvenanceTracker, ProvenanceType
from .imputation_window import ImputationWindowCalculator, ImputationScope

__all__ = [
    # Classes et fonctions principales
    # Détection
    'FrequencyDetector',
    'detect_frequency',
    'detect_dataset_frequency',
    'detect_index_frequency',
    'detect_and_parse_index_frequency',
    # Imputation
    'HighFrequencyImputer',
    # Provenance tracking
    'ImputationProvenanceTracker',
    'ProvenanceType',
    # P1 window calculation
    'ImputationWindowCalculator',
    'ImputationScope',
    # Parser functions
    'parse_frequency',
    'build_frequency_string',
]