"""Module XY pour transformateurs qui transforment à la fois X et y.

Ce module fournit :
- XYTransformerMixin : classe mixin pour créer des transformateurs XY
- XYPanelTimeSeriesTransformer : classe de base des transformateurs XY validés
  (séries temporelles et panels)
- XYPipeline : pipeline compatible avec les transformateurs XY
- DropNaNTarget : étape de retrait des lignes sans cible, avant l'estimateur
- is_xy_transformer : reconnaissance d'une étape XY déclarée
"""
# Importation des éléments d'intérêt du module
from .transformers import (
    DropNaNTarget,
    XYPanelTimeSeriesTransformer,
    XYTransformerMixin,
    is_xy_transformer,
)
from .pipeline import XYPipeline

# Ré-exports des éléments du module
__all__ = [
    "XYTransformerMixin",
    "XYPanelTimeSeriesTransformer",
    "XYPipeline",
    "DropNaNTarget",
    "is_xy_transformer",
]
