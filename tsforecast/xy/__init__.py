"""Module XY pour transformateurs qui transforment à la fois X et y.

Ce module fournit :
- XYTransformerMixin : classe mixin pour créer des transformateurs XY
- XYPipeline : pipeline compatible avec les transformateurs XY
"""
# Importation des éléments d'intérêt du module
from .transformers import XYTransformerMixin
from .pipeline import XYPipeline

# Ré-exports des éléments du module
__all__ = [
    "XYTransformerMixin",
    "XYPipeline",
]
