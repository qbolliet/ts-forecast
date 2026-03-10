# Importation des modules
from typing import Literal

# Création des types
# Types supportés pour les fréquences
FrequencyType = Literal['ns', 'us', 'ms', 's', 'min', 'T', 'h', 'D', 'B', 'W', 'SM', 'M', 'Q', 'A', 'Y']
UserFrequencyType = Literal[
    'daily', 'weekly', 'monthly', 'quarterly', 'annual', 'business_daily'
]
