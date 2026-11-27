"""Conversions between request arrays and framework tensors, shared by the backends.

``to_tensor`` copies a (read-only) request array into a ``float32`` tensor of the
framework passed to it, ``to_numpy`` brings the output of a model back as a
``float64`` numpy array. No framework is imported here.
"""
# Importation des modules
# Modules de base
from typing import Any

# Modules de manipulation de données
import numpy as np


# Fonction de conversion d'un tableau d'une requête en tenseur
def to_tensor(torch: Any, values: np.ndarray) -> Any:
    """Copy an array into a ``float32`` tensor on the CPU.

    Args:
        torch: The imported ``torch`` module.
        values: Array, possibly read-only.

    Returns:
        A tensor owning its data (torch refuses to share a read-only array).

    Examples:
        >>> import types
        >>> import numpy as np
        >>> framework = types.SimpleNamespace(float32='float32', as_tensor=lambda values, dtype: (values, dtype))
        >>> values, dtype = to_tensor(framework, np.array([1.0, 2.0]))
        >>> values.flags.writeable, dtype
        (True, 'float32')
    """
    return torch.as_tensor(np.array(values), dtype=torch.float32)


# Fonction de conversion de la sortie d'un modèle en tableau numpy
def to_numpy(values: Any) -> np.ndarray:
    """Convert a tensor (or an array) returned by a model to a ``float64`` array.

    Args:
        values: Torch tensor on any device, or array-like.

    Returns:
        A ``float64`` numpy array.

    Examples:
        >>> to_numpy([[1, 2]]).tolist()
        [[1.0, 2.0]]
    """
    # Tenseur torch : détaché et ramené sur le processeur
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)
