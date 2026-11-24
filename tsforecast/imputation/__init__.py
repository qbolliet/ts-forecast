"""Mixed-frequency imputation with provenance tracking.

This package provides ``HighFrequencyImputer``, which imputes low-frequency columns on a
finer grid, cascading from one frequency to the next while recording the provenance of every
cell. It re-exports the types read from the fitted attributes of the imputer: the imputation
plan, the provenance vocabulary and the cell origins. Reusable building blocks live in the
private ``_components`` sub-package.
"""

from .high_frequency_imputer import HighFrequencyImputer
from ._components.imputation_plan import (
    ImputationPlan,
    ImputationStep,
    MaterializationWay,
)
from ._components.provenance import (
    CELL_ORIGIN_DTYPE,
    PROVENANCE_DTYPE,
    CellOrigin,
    ProvenanceType,
    Taint,
)

__all__ = [
    # Imputeur
    'HighFrequencyImputer',
    # Plan d'imputation (attributs ajustés)
    'ImputationPlan',
    'ImputationStep',
    'MaterializationWay',
    # Provenance
    'ProvenanceType',
    'PROVENANCE_DTYPE',
    'CellOrigin',
    'CELL_ORIGIN_DTYPE',
    'Taint',
]
