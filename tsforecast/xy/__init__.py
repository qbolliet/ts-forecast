"""Package for XY transformers, which process features and target together. Re-exports the
transformer base classes, ``XYPipeline``, ``DropNaNTarget``, ``RollingAggregator`` and
``is_xy_transformer``."""
# Importation des éléments d'intérêt du module
from .transformers import (
    DropNaNTarget,
    XYPanelTimeSeriesTransformer,
    XYTransformerMixin,
    is_xy_transformer,
)
from .pipeline import XYPipeline
from .rolling import RollingAggregator

# Ré-exports des éléments du module
__all__ = [
    "XYTransformerMixin",
    "XYPanelTimeSeriesTransformer",
    "XYPipeline",
    "DropNaNTarget",
    "RollingAggregator",
    "is_xy_transformer",
]
