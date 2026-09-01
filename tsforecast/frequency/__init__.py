"""Frequency detection, conversion, and imputation utilities for time series processing.

This module provides tools for detecting and converting between different time frequencies,
with support for pandas frequency codes, DateOffsets, and user-friendly frequency names.
It also provides mixed-frequency imputation capabilities with provenance tracking.
"""

# Import des classes principales
from ..utils.frequency import FrequencyDetector, detect_frequency, detect_dataset_frequency, detect_index_frequency
from .high_frequency_imputer import HighFrequencyImputer
from .imputation_plan import ImputationStep, INTERPOLATE_FALLBACK
from .provenance import (
    ImputationProvenanceTracker,
    ProvenanceType,
    CellOrigin,
    Taint,
    resolve_model_provenance,
    origin_to_taint,
    max_origin,
)
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
    'ImputationStep',
    'INTERPOLATE_FALLBACK',
    # Provenance tracking
    'ImputationProvenanceTracker',
    'ProvenanceType',
    'CellOrigin',
    'Taint',
    'resolve_model_provenance',
    'origin_to_taint',
    'max_origin',
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