# Importation des modules
from typing import Literal

# Création des types
# Types supportés pour les fréquences
FrequencyType = Literal['ns', 'us', 'ms', 's', 'min', 'h', 'D', 'B', 'W', 'SM', 'M', 'Q', 'Y']
UserFrequencyType = Literal[
    'nanosecond', 'microsecond', 'millisecond', 'second', 'minute', 'hourly', 'daily', 'weekly', 'semi_monthly', 'monthly', 'quarterly', 'annual', 'business_daily'
]
