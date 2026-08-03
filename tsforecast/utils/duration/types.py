# Importation des modules
from typing import Literal, Optional

# Création des types
# Types supportés pour les durées
DurationType = Literal['ns', 'us', 'ms', 's', 'min', 'h', 'D', 'B', 'W', 'SM', 'M', 'Q', 'Y']
UserDurationType = Literal[
    'nanosecond', 'microsecond', 'millisecond', 'second', 'minute', 'hour',
    'day', 'business_day', 'week', 'semi_month', 'month', 'quarter', 'year'
]

# Types pour le rounding
RoundingType = Optional[Literal['floor', 'ceil']]