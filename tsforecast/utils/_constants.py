"""Shared constants of the ``tsforecast.utils`` sub-packages.

This private module is the single source of truth for the temporal vocabulary
(month and weekday anchors), the classification of frequency bases (which ones
accept an ``S``/``E`` position, which ones mark the start of their period...)
and the duration tables. It only depends on the standard library, so that any
sub-package can import it without creating an import cycle.

Constants used by a single module (and meaningless elsewhere) stay in that module.
"""
# Importation des modules
from typing import Dict, FrozenSet, Tuple

# ---------------------------------------------------------------------------
# Vocabulaire calendaire
# ---------------------------------------------------------------------------

# Abréviations pandas des mois, dans l'ordre (ancres trimestrielles et annuelles)
MONTH_ABBREVIATIONS: Tuple[str, ...] = (
    'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',
)

# Abréviations pandas des jours de la semaine (ancres hebdomadaires, ex: 'W-MON')
WEEKDAY_ABBREVIATIONS: Tuple[str, ...] = ('MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN')

# ---------------------------------------------------------------------------
# Durées
# ---------------------------------------------------------------------------

# Durée en nanosecondes des fréquences infra-journalières (source unique : les
# autres tables de durées infra-journalières en sont dérivées)
SUBDAILY_NS: Dict[str, int] = {
    'ns': 1,
    'us': 1_000,
    'ms': 1_000_000,
    's': 1_000_000_000,
    'min': 60 * 1_000_000_000,
    'h': 3_600 * 1_000_000_000,
}

# Unités infra-journalières de la plus grande à la plus petite, avec leur durée
# en nanosecondes
INTRADAY_UNITS: Tuple[Tuple[str, int], ...] = tuple(
    sorted(SUBDAILY_NS.items(), key=lambda item: item[1], reverse=True)
)

# Facteurs de conversion vers les secondes. Les unités infra-journalières sont
# dérivées de SUBDAILY_NS ; les unités calendaires sont des approximations
CONVERSION_FACTORS_TO_SECONDS: Dict[str, float] = {
    **{unit: nanoseconds / 1e9 for unit, nanoseconds in SUBDAILY_NS.items()},
    'D': 86400,  # 24 * 3600
    'B': 86400,  # Même que 'D' pour la conversion
    'W': 604800,  # 7 * 24 * 3600
    'SM': 1296000,  # 15 * 24 * 3600 (approximation)
    'M': 2592000,  # 30 * 24 * 3600 (approximation)
    'Q': 7776000,  # 90 * 24 * 3600 (approximation)
    'Y': 31536000,  # 365 * 24 * 3600 (approximation)
}

# Table des nombres de sous-périodes calendaires (clé : fréquence basse,
# fréquence haute). Les paires emboîtées (Y/Q/M, W/D) y sont exactes ; les
# autres (jours dans un mois, semaines dans une année) portent la valeur
# conventionnelle, aucune valeur exacte constante n'existant. Dans les deux
# cas la table corrige le ratio de durées, inexact par construction
# (365/30 = 12.17 mois dans une année).
CALENDAR_SUBPERIODS: Dict[Tuple[str, str], int] = {
    ('Y', 'Q'): 4, ('Y', 'M'): 12, ('Y', 'SM'): 24, ('Y', 'W'): 52, ('Y', 'D'): 365,
    ('Q', 'M'): 3, ('Q', 'SM'): 6, ('Q', 'W'): 13, ('Q', 'D'): 91,
    ('M', 'SM'): 2, ('M', 'D'): 30,
    ('W', 'D'): 7,
}

# ---------------------------------------------------------------------------
# Classification des bases de fréquence
# ---------------------------------------------------------------------------

# Nombre de mois par unité des fréquences calendaires à base mensuelle. Ces
# fréquences sont aussi celles dont les périodes sont décrites par un couple
# d'offsets début / fin
MONTH_BASED_FREQUENCIES: Dict[str, int] = {'M': 1, 'Q': 3, 'Y': 12}

# Bases de fréquence sans équivalent pd.Period (semi-mensuelle) : leurs périodes
# sont bornées par les offsets pandas 'SMS' / 'SME'
NON_PERIOD_FREQUENCIES: FrozenSet[str] = frozenset({'SM'})

# Fréquences pandas supportant un suffixe de position S/E (ex: 'MS', 'QE', 'SMS').
# 'W' et 'B' n'en font pas partie : pandas ne connaît ni 'WS'/'WE' ni 'BS'/'BE'
# (l'ancre hebdomadaire est un jour de la semaine, pas une position)
POSITION_AWARE_FREQUENCIES: FrozenSet[str] = frozenset(MONTH_BASED_FREQUENCIES) | NON_PERIOD_FREQUENCIES

# Bases de fréquence sans position S/E, comptées en jours ou en unités
# infra-journalières : un horodatage y marque le DÉBUT de sa période (ou de son
# bloc de n périodes), comme les étiquettes de pandas.resample
BLOCK_START_FREQUENCIES: FrozenSet[str] = frozenset({'D', 'B'}) | frozenset(SUBDAILY_NS)
