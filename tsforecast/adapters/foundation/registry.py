"""Registry of the forecasting backends of ``FoundationModelAdapter`` and process-level model cache.

``register_backend`` records a backend under a name with the model-id prefixes it
serves, ``available_backends`` lists them, ``resolve_backend`` finds the backend of
a model id, ``load_backend`` loads a model once per process (key: backend, model
id, device, loading options) and ``clear_model_cache`` releases the loaded models.
The ``chronos`` (``amazon/chronos-*``, ``autogluon/chronos-*``), ``timestral``
(``mistralai/Timestral-*``) and ``t0`` (``theforecastingcompany/t0-*``) backends are
registered on import, without importing their packages.
"""
# Importation des modules
# Modules de base
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Tuple

# tsforecast
from tsforecast.adapters._base import import_optional
from tsforecast.adapters.foundation.backends.chronos import CHRONOS_PREFIXES, load_chronos
from tsforecast.adapters.foundation.backends.t0 import T0_PREFIXES, load_t0
from tsforecast.adapters.foundation.backends.timestral import TIMESTRAL_PREFIXES, load_timestral
from tsforecast.adapters.foundation.protocol import BackendCapabilities, ForecastBackend

# Signature d'une fonction de chargement : (model_id, *, device, **options) -> modèle chargé
Loader = Callable[..., ForecastBackend]


# Classe de description d'un backend enregistré
@dataclass(frozen=True)
class BackendSpec:
    """A registered backend: how to load its models and which model ids it serves.

    Attributes:
        name: Name of the backend (``'chronos'``).
        load: Function ``(model_id, *, device, **backend_kwargs)`` returning a
            loaded model (:class:`ForecastBackend`).
        prefixes: Model-id prefixes served by the backend, lower case.
        package: Module imported by the backend, checked before each load, or None.
        extra: ``ts-forecast`` extra installing ``package``, or None.
    """

    # Nom du backend
    name: str
    # Fonction de chargement d'un modèle
    load: Loader
    # Préfixes d'identifiants servis, en minuscules
    prefixes: Tuple[str, ...]
    # Module requis par le backend
    package: Optional[str]
    # Extra installant ce module
    extra: Optional[str]


# Registre nom -> backend
_REGISTRY: Dict[str, BackendSpec] = {}
# Cache du processus : (backend, model_id, device, options figées) -> modèle chargé
_MODEL_CACHE: Dict[Tuple[Hashable, ...], ForecastBackend] = {}
# Verrou du cache : un seul chargement à la fois, jamais deux copies d'un même modèle
_CACHE_LOCK = threading.RLock()


# Fonction d'enregistrement d'un backend
def register_backend(
    name: str,
    load: Loader,
    *,
    prefixes: Tuple[str, ...] = (),
    package: Optional[str] = None,
    extra: Optional[str] = None,
    replace: bool = False,
) -> None:
    """Register a backend so that ``FoundationModelAdapter`` can load its models by name or model id.

    Args:
        name: Name of the backend, passed as ``backend=name`` to the adapter.
        load: Function ``(model_id, *, device, **backend_kwargs)`` returning a
            loaded model that follows :class:`ForecastBackend`.
        prefixes: Model-id prefixes served by the backend (case-insensitive);
            the longest matching prefix wins.
        package: Module the backend imports; its absence turns a load into an
            ``ImportError`` naming ``extra``.
        extra: ``ts-forecast`` extra installing ``package``.
        replace: Whether to replace a backend registered under the same name.

    Raises:
        ValueError: If ``name`` is empty or already registered without
            ``replace``, or if a prefix is empty.
        TypeError: If ``load`` is not callable.

    Examples:
        >>> from tsforecast.adapters.foundation import CallableBackend
        >>> register_backend('example', lambda model_id, *, device: CallableBackend(None),
        ...                  prefixes=('example-org/',))
        >>> resolve_backend('Example-Org/model-1').name
        'example'
        >>> _ = _REGISTRY.pop('example')
    """
    # Vérification du nom, de la fonction et des préfixes
    if not isinstance(name, str) or not name:
        raise ValueError(f"name must be a non-empty string, got {name!r}")
    if name in _REGISTRY and not replace:
        raise ValueError(f"A backend named {name!r} is already registered; pass replace=True to replace it")
    if not callable(load):
        raise TypeError(f"load must be callable, got {type(load).__name__}")
    lowered = tuple(str(prefix).lower() for prefix in prefixes)
    if any(not prefix for prefix in lowered):
        raise ValueError("Model-id prefixes must not be empty")
    _REGISTRY[name] = BackendSpec(name=name, load=load, prefixes=lowered, package=package, extra=extra)


# Fonction de liste des backends enregistrés
def available_backends() -> Dict[str, Tuple[str, ...]]:
    """List the registered backends and the model-id prefixes they serve.

    Returns:
        Backend names mapped to their prefixes, in registration order.

    Examples:
        >>> available_backends()['chronos']
        ('amazon/chronos', 'autogluon/chronos')
    """
    return {name: spec.prefixes for name, spec in _REGISTRY.items()}


# Fonction de résolution du backend d'un identifiant de modèle
def resolve_backend(model_id: Optional[str], backend: Optional[str] = None) -> BackendSpec:
    """Find the registered backend named ``backend``, or the one serving ``model_id``.

    Args:
        model_id: Hugging Face model id or local path.
        backend: Name of a registered backend; None to deduce it from the
            longest registered prefix of ``model_id`` (case-insensitive).

    Returns:
        The registered backend.

    Raises:
        ValueError: If ``backend`` is not registered, or no registered prefix
            matches ``model_id``.

    Examples:
        >>> resolve_backend('amazon/chronos-2').name
        'chronos'
        >>> resolve_backend('./my-local-model', backend='chronos').name
        'chronos'
    """
    # Backend désigné par son nom
    if backend is not None:
        if backend not in _REGISTRY:
            raise ValueError(f"Unknown backend {backend!r}; registered backends: {sorted(_REGISTRY)}")
        return _REGISTRY[backend]
    # Backend déduit du préfixe le plus long de l'identifiant
    lowered = str(model_id).lower()
    matches = [
        (len(prefix), spec)
        for spec in _REGISTRY.values()
        for prefix in spec.prefixes
        if lowered.startswith(prefix)
    ]
    if not matches:
        raise ValueError(
            f"No registered backend serves the model id {model_id!r}; pass backend=<name> for a local path. "
            f"Registered prefixes: {available_backends()}"
        )
    return max(matches, key=lambda match: match[0])[1]


# Fonction auxiliaire de gel des options de chargement
def _freeze(options: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Turn loading options into a hashable cache key.

    Args:
        options: Keyword arguments of the loading function.

    Returns:
        The items sorted by name.

    Raises:
        ValueError: If a value is not hashable.
    """
    frozen = tuple(sorted(options.items()))
    try:
        hash(frozen)
    except TypeError:
        raise ValueError(
            f"backend_kwargs values must be hashable to identify a loaded model, got {dict(options)!r}"
        ) from None
    return frozen


# Fonction de chargement d'un modèle, une fois par processus
def load_backend(
    spec: BackendSpec,
    model_id: str,
    *,
    device: str,
    backend_kwargs: Optional[Mapping[str, Any]] = None,
) -> ForecastBackend:
    """Return the loaded model of a backend, loading it on its first use in the process.

    The package of the backend is checked at every call, even when the model is
    already loaded, so that a missing package is always reported.

    Args:
        spec: Registered backend.
        model_id: Model id or local path.
        device: Device the model is loaded on (``'cpu'``, ``'cuda'``...).
        backend_kwargs: Loading options; part of the cache key.

    Returns:
        The loaded model, shared by every caller with the same key.

    Raises:
        ImportError: If the package of the backend is not installed.
        ValueError: If an option is not hashable.
        TypeError: If the loading function does not return a backend.

    Examples:
        >>> from tsforecast.adapters.foundation import CallableBackend
        >>> spec = BackendSpec('example', lambda model_id, *, device: CallableBackend(None), (), None, None)
        >>> load_backend(spec, 'model', device='cpu') is load_backend(spec, 'model', device='cpu')
        True
        >>> clear_model_cache() >= 1
        True
    """
    # Présence du paquet du backend, contrôlée avant le cache
    if spec.package is not None:
        import_optional(spec.package, spec.extra or "adapters-all")
    options = dict(backend_kwargs or {})
    key = (spec.name, model_id, device, _freeze(options))
    # Chargement sous verrou : jamais deux chargements du même modèle
    with _CACHE_LOCK:
        # Modèle déjà chargé dans le processus
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        loaded = spec.load(model_id, device=device, **options)
        if not isinstance(loaded, ForecastBackend) or not isinstance(loaded.capabilities, BackendCapabilities):
            raise TypeError(
                f"The loader of backend {spec.name!r} must return an object with BackendCapabilities "
                f"capabilities and a predict method, got {type(loaded).__name__}"
            )
        _MODEL_CACHE[key] = loaded
        return loaded


# Fonction de libération des modèles chargés
def clear_model_cache() -> int:
    """Release every model loaded by ``FoundationModelAdapter`` in the process.

    Returns:
        The number of models released.

    Examples:
        >>> clear_model_cache() >= 0
        True
    """
    with _CACHE_LOCK:
        released = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()
    return released


# Enregistrement des backends fournis par le paquet
register_backend(
    "chronos",
    load_chronos,
    prefixes=CHRONOS_PREFIXES,
    package="chronos",
    extra="adapters-chronos",
)
register_backend(
    "timestral",
    load_timestral,
    prefixes=TIMESTRAL_PREFIXES,
    package="timestral",
    extra="adapters-timestral",
)
register_backend(
    "t0",
    load_t0,
    prefixes=T0_PREFIXES,
    package="t0",
    extra="adapters-t0",
)
