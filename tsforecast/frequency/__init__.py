"""Frequency detection, conversion, and imputation utilities for time series processing.

This module provides tools for detecting and converting between different time frequencies,
with support for pandas frequency codes, DateOffsets, and user-friendly frequency names.
It also provides mixed-frequency imputation capabilities with provenance tracking.
"""

# Import des classes principales
from .detector import FrequencyDetector
from .high_frequency_imputer import HighFrequencyImputer
from .parser import detect_and_parse_frequency, build_frequency_string
from .provenance import ImputationProvenanceTracker, ProvenanceType
from .p1_window import P1WindowCalculator, ImputationScope

__all__ = [
    # Classes principales
    'FrequencyDetector',
    'HighFrequencyImputer',
    # Provenance tracking
    'ImputationProvenanceTracker',
    'ProvenanceType',
    # P1 window calculation
    'P1WindowCalculator',
    'ImputationScope',
    # Parser functions
    'detect_and_parse_frequency',
    'build_frequency_string',
]