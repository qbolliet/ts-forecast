"""Module XY pour transformateurs qui transforment à la fois X et y.

Ce module fournit :
- XYTransformerMixin : classe mixin pour créer des transformateurs XY
- XYPanelTimeSeriesTransformer : classe de base des transformateurs XY validés
  (séries temporelles et panels)
- XYPipeline : pipeline compatible avec les transformateurs XY
- DropNaNTarget : étape de retrait des lignes sans cible, avant l'estimateur
"""
# Importation des éléments d'intérêt du module
from .transformers import DropNaNTarget, XYPanelTimeSeriesTransformer, XYTransformerMixin
from .pipeline import XYPipeline

# Ré-exports des éléments du module
__all__ = [
    "XYTransformerMixin",
    "XYPanelTimeSeriesTransformer",
    "XYPipeline",
    "DropNaNTarget",
]
