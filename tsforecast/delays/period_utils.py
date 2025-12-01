"""Period-based delay calculation utilities for publication delay handling.

This module provides functions to convert publication delays from duration units
(days, seconds, etc.) to number of periods at a given frequency.
"""
# Importation des modules
from typing import Literal, Optional
from datetime import datetime
import math

# Import du core pandas
import pandas as pd

# Import des utilitaires du package
from tsforecast.utils.frequency import normalize_frequency
from tsforecast.utils.duration import convert_duration, normalize_duration
from tsforecast.utils.time import get_period_boundaries


def calculate_n_periods_delay(
    applicable_delay: float,
    delay_unit: str,
    prediction_date: datetime,
    reference_point: Literal['start', 'end'],
    observation_date: pd.Timestamp,
    column_frequency: str,
    target_frequency: Optional[str] = None
) -> int:
    """Calculate number of periods to shift/mask based on publication delay.

    Formula:
        n_periods = ceil((applicable_delay - elapsed_duration) / period_duration)

    All durations are expressed in delay_unit (not converted to seconds).
    Can return negative values for future predictions.

    Args:
        applicable_delay: Publication delay in delay_unit
        delay_unit: Unit of the delay ('D', 's', 'ms', 'h', 'min', etc.)
        prediction_date: Date of prediction
        reference_point: 'start' or 'end' of period (reference for delay calculation)
        observation_date: Date of the observation (for period boundaries)
        column_frequency: Detected frequency of the column
        target_frequency: Target frequency for delay calculation (defaults to column_frequency)

    Returns:
        Number of periods to shift/mask (can be negative for future predictions)

    Raises:
        ValueError: If frequencies or durations are not supported

    Examples:
        >>> from datetime import datetime
        >>> import pandas as pd
        >>> # Delay of 45 days, prediction on 2024-12-15, period end on 2024-12-01
        >>> # Monthly frequency: period is 2024-12-01 to 2025-01-01
        >>> # Reference point 'end' = 2025-01-01
        >>> # Elapsed = 2024-12-15 - 2025-01-01 = -17 days
        >>> # n_periods = ceil((45 - (-17)) / 30.44) = ceil(62 / 30.44) = ceil(2.04) = 3
        >>> calculate_n_periods_delay(
        ...     applicable_delay=45,
        ...     delay_unit='D',
        ...     prediction_date=datetime(2024, 12, 15),
        ...     reference_point='end',
        ...     observation_date=pd.Timestamp('2024-12-01'),
        ...     column_frequency='M',
        ...     target_frequency='M'
        ... )
        3

        >>> # With future prediction (negative delay)
        >>> calculate_n_periods_delay(
        ...     applicable_delay=-30,
        ...     delay_unit='D',
        ...     prediction_date=datetime(2024, 12, 15),
        ...     reference_point='end',
        ...     observation_date=pd.Timestamp('2024-12-01'),
        ...     column_frequency='M',
        ...     target_frequency='M'
        ... )
        -2
    """
    # Normalisation de l'unité de délai
    delay_unit_code = normalize_duration(delay_unit)

    # Détermination de la fréquence cible (défaut = fréquence de la colonne)
    if target_frequency is None:
        target_frequency = column_frequency

    # Normalisation de la fréquence cible
    target_freq_code = normalize_frequency(target_frequency)

    # Obtention des bornes de période pour l'observation
    period_start, period_end = get_period_boundaries(observation_date, target_freq_code)

    # Calcul de la date de référence selon le point de référence
    if reference_point == 'start':
        reference_date = period_start
    elif reference_point == 'end':
        reference_date = period_end
    else:
        raise ValueError(f"reference_point must be 'start' or 'end', got '{reference_point}'")

    # Calcul du temps écoulé (prediction_date - reference_date) en secondes
    elapsed_timedelta = prediction_date - reference_date
    elapsed_seconds = elapsed_timedelta.total_seconds()

    # Conversion du temps écoulé dans l'unité de délai
    elapsed_duration = convert_duration(
        value=elapsed_seconds,
        from_duration='s',
        to_duration=delay_unit_code,
        rounding=None
    )

    # Obtention de la durée d'une période dans l'unité de délai
    period_duration = get_period_duration(target_freq_code, delay_unit_code)

    # Calcul du nombre de périodes
    # Note : peut être négatif si la prédiction est dans le futur
    n_periods_float = (applicable_delay - elapsed_duration) / period_duration

    # Arrondi à l'entier supérieur (ceil)
    # Pour les valeurs négatives, ceil(-2.1) = -2 (vers zéro)
    n_periods = math.ceil(n_periods_float)

    return n_periods


def get_period_duration(frequency: str, unit: str) -> float:
    """Return duration of one period at given frequency, in specified unit.

    Uses average durations for irregular periods:
    - Month: 30.44 days (365.25/12)
    - Quarter: 91.31 days (365.25/4)
    - Year: 365.25 days

    Args:
        frequency: Frequency code ('D', 'M', 'Q', 'W', 'h', etc.)
        unit: Desired output unit ('D', 's', 'ms', 'h', 'min', etc.)

    Returns:
        Duration of one period in specified unit

    Raises:
        ValueError: If frequency or unit is not supported

    Examples:
        >>> get_period_duration('M', 'D')  # Monthly in days
        30.4375
        >>> get_period_duration('Q', 'D')  # Quarterly in days
        91.3125
        >>> get_period_duration('D', 'h')  # Daily in hours
        24.0
        >>> get_period_duration('h', 'min')  # Hourly in minutes
        60.0
        >>> get_period_duration('W', 'D')  # Weekly in days
        7.0
    """
    # Normalisation de la fréquence et de l'unité
    freq_code = normalize_frequency(frequency)
    unit_code = normalize_duration(unit)

    # Mapping des fréquences vers leur durée en jours
    # Utilise des moyennes pour les périodes irrégulières
    freq_to_days = {
        # Fréquences infra-journalières (en fraction de jour)
        'ns': 1 / (24 * 3600 * 1e9),        # nanoseconde
        'us': 1 / (24 * 3600 * 1e6),        # microseconde
        'ms': 1 / (24 * 3600 * 1000),       # milliseconde
        's': 1 / (24 * 3600),               # seconde
        'min': 1 / (24 * 60),               # minute
        'h': 1 / 24,                         # heure
        # Fréquences journalières et supérieures
        'D': 1.0,                            # jour
        'B': 1.0,                            # jour ouvrable (approximé à 1 jour)
        'W': 7.0,                            # semaine
        'SM': 15.22,                         # semi-mensuel (30.44/2)
        'M': 30.4375,                        # mois (365.25/12)
        'Q': 91.3125,                        # trimestre (365.25/4)
        'Y': 365.25,                         # année (avec année bissextile)
    }

    # Vérification que la fréquence est supportée
    if freq_code not in freq_to_days:
        raise ValueError(
            f"Unsupported frequency: {frequency} (normalized: {freq_code}). "
            f"Supported frequencies: {', '.join(freq_to_days.keys())}"
        )

    # Obtention de la durée en jours
    duration_in_days = freq_to_days[freq_code]

    # Conversion dans l'unité demandée
    period_duration = convert_duration(
        value=duration_in_days,
        from_duration='D',
        to_duration=unit_code,
        rounding=None
    )

    return period_duration
