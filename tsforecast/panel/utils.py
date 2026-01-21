# Importation des modules
# Modules de base
import pandas as pd
from typing import List, Union

# Fonction de normalisation de l'entité du panel
def normalize_entity_key(key) -> tuple:
    """Normalize entity key to tuple format.

    Args:
        key: Entity key (scalar or tuple).

    Returns:
        Normalized entity key as a tuple.
    """
    if isinstance(key, tuple):
        return key
    return (key,)

# Fonction d'identification des unités d'un panel
def get_unique_panel_entities(panel_data: Union[pd.DataFrame, pd.Series]) -> List[tuple]:
    """Extract unique entities from panel data.

    Assumes panel_data has a MultiIndex where the first n-1 levels represent
    entities and the last (n-th) level represents time/date. This function
    extracts all unique entity combinations and normalizes them as tuples.

    Args:
        panel_data: Panel data (DataFrame or Series) with MultiIndex structure
            where entity levels come first, followed by a time/date level.

    Returns:
        List of unique normalized entity keys as tuples.

    Raises:
        IndexError: If panel_data has fewer than 2 index levels.
    """
    # Extraction des levels d'entité (tous sauf le dernier qui est la date)
    entity_index = panel_data.index.droplevel(-1)
    
    # Récupération des combinaisons uniques d'entités
    unique_entities = entity_index.unique()
    
    # Normalisation des clés d'entité
    normalized_entities = [normalize_entity_key(entity) for entity in unique_entities]
    
    return normalized_entities