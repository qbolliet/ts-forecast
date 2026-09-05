"""Frequency detection, conversion, and imputation utilities for time series processing.

This module provides tools for detecting and converting between different time frequencies,
with support for pandas frequency codes, DateOffsets, and user-friendly frequency names.
It also provides mixed-frequency imputation capabilities with provenance tracking.
"""

# Import des classes principales
from ..utils.frequency import FrequencyDetector, detect_frequency, detect_dataset_frequency, detect_index_frequency
from .high_frequency_imputer import HighFrequencyImputer
from .high_frequency_imputer2 import HighFrequencyImputer2
from .imputation_plan import ImputationStep, INTERPOLATE_FALLBACK
from .imputation_plan2 import (
    ImputationStep as ImputationStep2,
    ImputationPlan as ImputationPlan2,
    MaterializationWay,
)
from .provenance import (
    ImputationProvenanceTracker,
    ProvenanceType,
    CellOrigin,
    Taint,
    resolve_model_provenance,
    origin_to_taint,
    max_origin,
)
from .covariate_materializer import (
    CovariateMaterializer,
    AggregationConstraintApplier,
    DEFAULT_MATERIALIZATION_KEY,
)
from .aggregation_constraint import (
    AggregationConstraint,
    ConstraintKind,
    ConstraintSetting,
    AggregationConstraintSetting,
    DEFAULT_CONSTRAINT_KEY,
    validate_aggregation_constraint,
    resolve_aggregation_constraint,
    validate_constraint_columns,
)
from .stage_scaler import StageScaler, ScaleMode
from .variable_orderer import VariableOrderer, VariableSpec
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
    'HighFrequencyImputer2',
    'ImputationStep',
    'INTERPOLATE_FALLBACK',
    'ImputationStep2',
    'ImputationPlan2',
    'MaterializationWay',
    # Provenance tracking
    'ImputationProvenanceTracker',
    'ProvenanceType',
    'CellOrigin',
    'Taint',
    'resolve_model_provenance',
    'origin_to_taint',
    'max_origin',
    # Matérialisation des covariables
    'CovariateMaterializer',
    'AggregationConstraintApplier',
    'DEFAULT_MATERIALIZATION_KEY',
    # Contrainte d'agrégation
    'AggregationConstraint',
    'ConstraintKind',
    'ConstraintSetting',
    'AggregationConstraintSetting',
    'DEFAULT_CONSTRAINT_KEY',
    'validate_aggregation_constraint',
    'resolve_aggregation_constraint',
    'validate_constraint_columns',
    # Mise à l'échelle d'étape
    'StageScaler',
    'ScaleMode',
    # Ordonnancement des variables
    'VariableOrderer',
    'VariableSpec',
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