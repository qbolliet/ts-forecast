"""Literal type aliases for period positions: canonical codes (``PositionType``, 'S' and
'E') and user-facing names (``UserPositionType``, 'start' and 'end')."""
# Importation des modules
from typing import Literal

# Types supportés pour les positions de période
PositionType = Literal['S', 'E']
UserPositionType = Literal['start', 'end']