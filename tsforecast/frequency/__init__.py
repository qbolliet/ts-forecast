"""Frequency detection, conversion, and imputation utilities for time series processing.

This module provides tools for detecting and converting between different time frequencies,
with support for pandas frequency codes, DateOffsets, and user-friendly frequency names.
It also provides mixed-frequency imputation capabilities with provenance tracking.
"""

# Import des classes principales
from .detector import FrequencyDetector, detect_frequency, detect_dataset_frequency, detect_index_frequency
from .high_frequency_imputer import HighFrequencyImputer
from .provenance import ImputationProvenanceTracker, ProvenanceType
from .imputation_window import ImputationWindowCalculator, ImputationScope
from .target_frequency_validator import TargetFrequencyValidator
from .frequency_aligner import FrequencyAligner
from .regularizer import IndexRegularizer, is_regular, regularize

__all__ = [
    # Classes et fonctions principales
    # Détection
    'FrequencyDetector',
    'detect_frequency',
    'detect_dataset_frequency',
    'detect_index_frequency',
    # Imputation
    'HighFrequencyImputer',
    # Provenance tracking
    'ImputationProvenanceTracker',
    'ProvenanceType',
    # Imputation window calculation
    'ImputationWindowCalculator',
    'ImputationScope',
    # Validation et alignement
    'TargetFrequencyValidator',
    'FrequencyAligner',
    # Régularisation d'index
    'IndexRegularizer',
    'is_regular',
    'regularize',
]