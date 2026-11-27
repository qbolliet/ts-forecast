"""Protocol between ``FoundationModelAdapter`` and the backends of forecasting foundation models.

``BackendCapabilities`` declares what a loaded model accepts (past and known
covariates, missing values, several targets, longest context and horizon, native
quantile levels), ``ForecastRequest`` carries one batch of series to forecast over
a common number of steps, ``ForecastResult`` the quantile forecasts of that batch,
and ``ForecastBackend`` is the structural type of a loaded model.
"""
# Importation des modules
# Modules de base
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, Tuple, runtime_checkable

# Modules de manipulation de données
import numpy as np


# Fonction auxiliaire de vue en lecture seule d'un tableau
def _read_only(values: Any) -> np.ndarray:
    """Return a read-only ``float64`` view (or copy) of an array, leaving the input writable.

    Args:
        values: Array-like of numbers.

    Returns:
        A ``float64`` array whose writes are refused; it shares the memory of
        ``values`` when ``values`` already is a ``float64`` array.
    """
    # Vue sur les données (copie seulement en cas de conversion de type)
    view = np.asarray(values, dtype=np.float64).view()
    view.setflags(write=False)
    return view


# Fonction auxiliaire de contrôle d'une suite de niveaux de quantiles
def _check_levels(levels: Any, name: str) -> Tuple[float, ...]:
    """Check that quantile levels are strictly increasing and inside ``]0, 1[``.

    Args:
        levels: Sequence of levels.
        name: Name of the argument, for the error message.

    Returns:
        The levels as a tuple of floats.

    Raises:
        ValueError: If the sequence is empty, not strictly increasing, or holds
            a level outside ``]0, 1[``.
    """
    # Conversion en tuple de flottants
    values = tuple(float(level) for level in levels)
    # Suite vide, niveau hors de ]0, 1[ ou suite non strictement croissante
    if not values:
        raise ValueError(f"{name} must hold at least one level")
    if not all(0.0 < level < 1.0 for level in values):
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {values}")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing, got {values}")
    return values


# Fonction auxiliaire de contrôle d'un entier positif facultatif
def _check_positive(value: Optional[int], name: str) -> Optional[int]:
    """Check that an optional count is a positive integer.

    Args:
        value: Count, or None.
        name: Name of the field, for the error message.

    Returns:
        ``value`` unchanged.

    Raises:
        ValueError: If ``value`` is not None and not an integer of at least 1.
    """
    # Booléens exclus : True n'est pas une longueur
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be None or an integer of at least 1, got {value!r}")
    return value


# Classe de déclaration des capacités d'un modèle chargé
@dataclass(frozen=True)
class BackendCapabilities:
    """What a loaded forecasting model accepts, read by the adapter before each request.

    Attributes:
        past_covariates: Whether the model reads covariates known over the
            context only.
        known_covariates: Whether the model reads covariates known over the
            context and the forecast horizon.
        missing_values: Whether missing values (``NaN``) are accepted in the
            context and in the covariates.
        multivariate: Whether several targets of a series are forecast jointly;
            otherwise the adapter sends one series per target.
        max_context: Longest context read by the model, in steps; the adapter
            keeps the most recent steps. None when unbounded.
        max_prediction_length: Longest forecast, in steps; the adapter refuses
            rows further away. None when unbounded.
        quantile_levels: Quantile levels the model was trained on, increasing;
            requested levels must lie within their range. None when any level
            of ``]0, 1[`` can be produced.
        interpolates_quantiles: Whether the model returns any level within
            the range of ``quantile_levels``; otherwise the adapter requests the
            native levels around each requested one and interpolates linearly.

    Raises:
        ValueError: If a length is not a positive integer or the levels are
            not strictly increasing inside ``]0, 1[``.

    Examples:
        >>> capabilities = BackendCapabilities(known_covariates=True, quantile_levels=(0.1, 0.5, 0.9))
        >>> capabilities.known_covariates, capabilities.max_context, capabilities.quantile_levels
        (True, None, (0.1, 0.5, 0.9))
    """

    # Covariables connues sur le contexte seul
    past_covariates: bool = False
    # Covariables connues sur le contexte et l'horizon
    known_covariates: bool = False
    # Valeurs manquantes acceptées dans le contexte et les covariables
    missing_values: bool = False
    # Plusieurs cibles d'une série prévues conjointement
    multivariate: bool = False
    # Contexte le plus long lu par le modèle, en pas
    max_context: Optional[int] = None
    # Prévision la plus longue, en pas
    max_prediction_length: Optional[int] = None
    # Niveaux de quantiles appris par le modèle
    quantile_levels: Optional[Tuple[float, ...]] = None
    # Production par le modèle de tout niveau compris entre ses niveaux appris
    interpolates_quantiles: bool = True

    # Vérification des champs
    def __post_init__(self) -> None:
        """Validate the lengths and the native levels."""
        # Longueurs positives facultatives
        _check_positive(self.max_context, "max_context")
        _check_positive(self.max_prediction_length, "max_prediction_length")
        # Niveaux natifs mis sous forme de tuple de flottants
        if self.quantile_levels is not None:
            object.__setattr__(self, "quantile_levels", _check_levels(self.quantile_levels, "quantile_levels"))


# Classe de requête de prévision d'un lot de séries
@dataclass(frozen=True)
class ForecastRequest:
    """One batch of series to forecast over a common number of steps.

    Every series is on a regular grid, oldest step first; its last column is
    the last training date of its entity, and the forecast covers the
    ``prediction_length`` following steps. Arrays are read-only views: a
    backend never modifies them.

    Attributes:
        context: One ``(n_targets, n_steps)`` array of target values per
            series, ``NaN`` for a missing value; lengths may differ between
            series.
        prediction_length: Number of steps to forecast after the context.
        quantile_levels: Levels to return, strictly increasing inside ``]0, 1[``.
        past_covariates: One ``(n_past, n_steps)`` array per series, aligned
            with its context, or None.
        known_covariates: One ``(n_known, n_steps + prediction_length)`` array
            per series: the values over the context, then over the forecast
            steps, ``NaN`` for a value that is not known; or None.
        past_covariate_names: Names of the rows of ``past_covariates``.
        known_covariate_names: Names of the rows of ``known_covariates``.
        freq: Pandas alias of the grid frequency, or None.
        options: Backend-specific options, passed unchanged.

    Raises:
        ValueError: If the shapes are inconsistent.

    Examples:
        >>> import numpy as np
        >>> request = ForecastRequest(
        ...     context=(np.array([[1.0, 2.0, 3.0]]),), prediction_length=2, quantile_levels=(0.5,))
        >>> request.n_series, request.n_targets, request.context[0].flags.writeable
        (1, 1, False)
    """

    # Contexte de cible de chaque série
    context: Tuple[np.ndarray, ...]
    # Nombre de pas à prévoir après le contexte
    prediction_length: int
    # Niveaux de quantiles à renvoyer
    quantile_levels: Tuple[float, ...]
    # Covariables connues sur le contexte seul, par série
    past_covariates: Optional[Tuple[np.ndarray, ...]] = None
    # Covariables connues sur le contexte et l'horizon, par série
    known_covariates: Optional[Tuple[np.ndarray, ...]] = None
    # Noms des covariables passées
    past_covariate_names: Tuple[str, ...] = ()
    # Noms des covariables connues
    known_covariate_names: Tuple[str, ...] = ()
    # Alias pandas de la fréquence de la grille
    freq: Optional[str] = None
    # Options propres au backend
    options: Mapping[str, Any] = field(default_factory=dict)

    # Vérification et gel des champs
    def __post_init__(self) -> None:
        """Freeze the arrays and check their shapes."""
        # Contextes : tableaux à deux dimensions en lecture seule, même nombre de cibles
        context = tuple(_read_only(values) for values in self.context)
        if not context:
            raise ValueError("A forecast request needs at least one series")
        if any(values.ndim != 2 or values.shape[1] == 0 for values in context):
            raise ValueError("Each context must be a non-empty (n_targets, n_steps) array")
        if len({values.shape[0] for values in context}) > 1:
            raise ValueError("Every series of a request must have the same number of targets")
        object.__setattr__(self, "context", context)
        # Longueur de prévision et niveaux
        _check_positive(self.prediction_length, "prediction_length")
        if self.prediction_length is None:
            raise ValueError("prediction_length must be an integer of at least 1")
        object.__setattr__(self, "quantile_levels", _check_levels(self.quantile_levels, "quantile_levels"))
        # Covariables passées puis connues : un tableau par série, longueur attendue en pas
        lengths = [values.shape[1] for values in context]
        for name, extra in (("past_covariates", 0), ("known_covariates", self.prediction_length)):
            covariates = getattr(self, name)
            names = getattr(self, f"{name[:-1]}_names")
            # Covariables absentes : aucun nom attendu
            if covariates is None:
                if names:
                    raise ValueError(f"{name[:-1]}_names is given without {name}")
                continue
            covariates = tuple(_read_only(values) for values in covariates)
            if len(covariates) != len(context):
                raise ValueError(f"{name} must hold one array per series")
            for values, length in zip(covariates, lengths):
                if values.ndim != 2 or values.shape[1] != length + extra or values.shape[0] != len(names):
                    raise ValueError(
                        f"Each array of {name} must have shape ({len(names)}, {length + extra}), "
                        f"got {values.shape}"
                    )
            object.__setattr__(self, name, covariates)
        # Options figées
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    @property
    def n_series(self) -> int:
        """Number of series of the request."""
        return len(self.context)

    @property
    def n_targets(self) -> int:
        """Number of targets of each series."""
        return self.context[0].shape[0]


# Classe de résultat d'une requête de prévision
@dataclass(frozen=True)
class ForecastResult:
    """Quantile forecasts of a :class:`ForecastRequest`.

    Attributes:
        quantiles: Array of shape ``(n_series, n_targets, prediction_length,
            n_levels)``, series in the order of the request, levels in the
            order of ``request.quantile_levels``.

    Raises:
        ValueError: If ``quantiles`` does not have four dimensions.

    Examples:
        >>> import numpy as np
        >>> ForecastResult(np.zeros((2, 1, 3, 1))).quantiles.shape
        (2, 1, 3, 1)
    """

    # Prévisions par série, cible, pas et niveau
    quantiles: np.ndarray

    # Vérification de la forme
    def __post_init__(self) -> None:
        """Convert the forecasts to a read-only ``float64`` array of four dimensions."""
        quantiles = _read_only(self.quantiles)
        if quantiles.ndim != 4:
            raise ValueError(
                "quantiles must have shape (n_series, n_targets, prediction_length, n_levels), "
                f"got {quantiles.shape}"
            )
        object.__setattr__(self, "quantiles", quantiles)


# Type structurel d'un modèle chargé
@runtime_checkable
class ForecastBackend(Protocol):
    """A loaded forecasting model, as the adapter uses it.

    Any object with a ``capabilities`` attribute and a ``predict`` method
    follows the protocol. A backend must not keep state between calls: two
    identical requests give the same forecasts.

    Attributes:
        capabilities: What the model accepts.

    Examples:
        >>> import numpy as np
        >>> class Constant:
        ...     capabilities = BackendCapabilities()
        ...     def predict(self, request):
        ...         shape = (request.n_series, request.n_targets, request.prediction_length,
        ...                  len(request.quantile_levels))
        ...         return ForecastResult(np.zeros(shape))
        >>> isinstance(Constant(), ForecastBackend)
        True
    """

    capabilities: BackendCapabilities

    # Prévision d'un lot de séries
    def predict(self, request: ForecastRequest) -> ForecastResult:
        """Forecast the series of ``request``.

        Args:
            request: Series, number of steps and levels to forecast.

        Returns:
            The quantile forecasts of every series.
        """
        ...


# Fonction de test de l'égalité approchée de deux niveaux
def same_level(left: float, right: float) -> bool:
    """Whether two quantile levels are equal up to floating-point rounding.

    Args:
        left: First level.
        right: Second level.

    Returns:
        True when the levels differ by less than ``1e-12``.

    Examples:
        >>> same_level(0.1 + 0.2, 0.3), same_level(0.1, 0.2)
        (True, False)
    """
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
