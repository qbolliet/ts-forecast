
# Méthode auxiliaire de normalisation de l'entité du panel
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
