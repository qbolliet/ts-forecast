"""Shared builders of the ``StageScaler`` tests (not a test module).

Scaler factory, monthly and panel grids, feature-row readers and calendar helpers used
by several test files of the package.
"""

# Modules de base
from typing import Tuple

# Manipulation de données
import pandas as pd

# Objets testés
from tsforecast.frequency.stage_scaler import StageScaler


# Seuil de la symétrie transform / inverse_transform : un écart supérieur est
# une anomalie, jamais une tolérance à élargir
_EXACT = 1e-12

# Fréquences de bloc du jeu PANEL-F : v annuelle, trimestrielle, mensuelle
_PANEL_F_BLOCKS = {('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'}


# Fabrique de scaler : configuration par défaut cohérente, surchargeable
def _make_scaler(**overrides) -> StageScaler:
    """Build a StageScaler with a coherent default configuration.

    Args:
        **overrides: Constructor arguments replacing the defaults.

    Returns:
        An unfitted ``StageScaler`` (stage ``M``, variable ``Y``).
    """
    params = dict(
        scale_features='constant',
        source_freq='Y',
        pred_freq='M',
        column_frequencies={'m1': 'M', 'q1': 'Q', 'a2': 'Y'},
    )
    params.update(overrides)
    return StageScaler(**params)


# Grille mensuelle de fin de mois, calquée sur le jeu TS de référence
def _monthly_grid(start: str = '2021-01-31', periods: int = 12) -> pd.DatetimeIndex:
    """Build a month-end grid.

    Args:
        start: First month end.
        periods: Number of months.

    Returns:
        A ``DatetimeIndex`` of month ends.
    """
    return pd.date_range(start=start, periods=periods, freq='ME')


# Grille de panel (entité, date) en fin de mois
def _panel_grid(entities: Tuple[str, ...] = ('DE', 'FR'), periods: int = 3) -> pd.MultiIndex:
    """Build a month-end panel grid.

    Args:
        entities: Entity names, outer level.
        periods: Number of months per entity.

    Returns:
        A ``MultiIndex`` (``country``, ``date``).
    """
    return pd.MultiIndex.from_product(
        [list(entities), _monthly_grid(periods=periods)], names=['country', 'date']
    )
