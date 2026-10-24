"""Parameter validators of ``HighFrequencyImputer``, written as pure functions.

Each ``validate_*`` function checks one constructor parameter from explicit arguments and
raises ``ValueError`` or ``TypeError`` with a message naming the parameter. Also holds the
strict integer predicate ``is_integer`` and the admissible modalities of
``impute_intermediate_frequencies``.
"""
# Importation des modules
# Modules de base
import numbers
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# Manipulation de données
import numpy as np

# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin

# Utilitaires du package
from ...panel.utils import normalize_entity_key
from ...utils.frequency.utils import normalize_frequency

# Modalités admissibles de "impute_intermediate_frequencies"
INTERMEDIATE_MODALITIES: Tuple[Any, ...] = (False, 'covariates_only', True)


# Prédicat d'entier strict, booléens exclus
def is_integer(value: Any) -> bool:
    """Tell whether a value is an integer, numpy integers included.

    Python and numpy booleans are excluded: ``True`` is an ``int`` for
    Python, never a sensible count.

    Args:
        value: Value to test.

    Returns:
        True for ``int`` and ``numpy`` integers, False otherwise.

    Examples:
        >>> is_integer(3), is_integer(np.int64(3)), is_integer(True)
        (True, True, False)
    """
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


# Validation d'un littéral
def validate_literal(
    name: str,
    value: Any,
    admissible: Sequence[Any],
    sort: bool = False,
) -> None:
    """Check that a parameter belongs to its ``Literal``.

    Args:
        name: Parameter name, quoted in the message.
        value: Value handed to ``__init__``.
        admissible: Admissible values, listed in the message.
        sort: If True, the message lists the values sorted, as a list;
            otherwise in the given order, as a tuple.

    Raises:
        ValueError: If the value is not one of the admissible ones.

    Examples:
        >>> validate_literal('fit_predict_order', 'cv', ('frequency', 'cv'))
        >>> validate_literal('fit_predict_order', 'x', ('frequency', 'cv'))
        Traceback (most recent call last):
            ...
        ValueError: fit_predict_order must be one of ('frequency', 'cv'), got 'x'
        >>> validate_literal('order', 'x', ('b', 'a'), sort=True)
        Traceback (most recent call last):
            ...
        ValueError: order must be one of ['a', 'b'], got 'x'
    """
    # Appartenance au littéral, message énumérant les valeurs admises
    if value not in admissible:
        listed = sorted(admissible) if sort else tuple(admissible)
        raise ValueError(f"{name} must be one of {listed}, got {value!r}")


# Validation de la modalité de l'axe 2
def validate_intermediate_frequencies(value: Any) -> None:
    """Check ``impute_intermediate_frequencies`` without truth testing.

    The three modalities are compared by identity for the booleans and by
    equality for the string. A membership test against a tuple would
    conflate ``0`` with ``False`` and ``1`` with ``True``; a truth test
    would silently promote ``'covariates_only'`` — which is truthy — into
    ``True``.

    Args:
        value: Value handed to ``__init__``.

    Raises:
        ValueError: If the value is not one of the three modalities.

    Examples:
        >>> validate_intermediate_frequencies('covariates_only')
        >>> validate_intermediate_frequencies(1)
        Traceback (most recent call last):
            ...
        ValueError: impute_intermediate_frequencies must be one of (False, 'covariates_only', True), got 1
    """
    # Comparaison modalité par modalité, jamais par vérité booléenne
    if value is False or value is True or value == 'covariates_only':
        return
    raise ValueError(
        f"impute_intermediate_frequencies must be one of "
        f"{INTERMEDIATE_MODALITIES}, got {value!r}"
    )


# Validation d'un réel de l'intervalle unité
def validate_unit_interval(name: str, value: Any) -> None:
    """Check that a parameter is a float within ``[0, 1]``.

    Args:
        name: Parameter name, quoted in the message.
        value: Value handed to ``__init__``.

    Raises:
        TypeError: If the value is not a real number.
        ValueError: If the value falls outside ``[0, 1]``.

    Examples:
        >>> validate_unit_interval('coverage_threshold', 0.5)
        >>> validate_unit_interval('coverage_threshold', 1.5)
        Traceback (most recent call last):
            ...
        ValueError: coverage_threshold must be a float between 0 and 1, got 1.5
    """
    # Exclusion explicite des booléens, que Python range parmi les entiers
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a float, got {type(value).__name__}"
        )
    if not 0 <= value <= 1:
        raise ValueError(
            f"{name} must be a float between 0 and 1, got {value}"
        )


# Validation de la stratégie de validation croisée
def validate_cv(cv: Any) -> None:
    """Check the format of ``cv`` without resolving it.

    ``check_cv`` is deliberately not called here: it would materialize a
    splitter object that ``get_params`` would then have to hand back
    unchanged. Resolution happens at ``fit``, into ``cv_``.

    Args:
        cv: Value handed to ``__init__``.

    Raises:
        TypeError: If ``cv`` is a bool, or is none of None, an int, a
            splitter (``split`` and ``get_n_splits``) nor an iterable of
            splits (a string is not one).
        ValueError: If ``cv`` is an int below 2.

    Examples:
        >>> validate_cv(None)
        >>> validate_cv(3)
        >>> validate_cv(1)
        Traceback (most recent call last):
            ...
        ValueError: cv must be an int >= 2, got 1
        >>> validate_cv(True)
        Traceback (most recent call last):
            ...
        TypeError: cv must not be a bool, got True
    """
    # Absence de stratégie : sklearn choisira son défaut au fit
    if cv is None:
        return
    # Un booléen est un entier pour Python, jamais un nombre de plis sensé
    if isinstance(cv, (bool, np.bool_)):
        raise TypeError(f"cv must not be a bool, got {cv!r}")
    # Forme entière : nombre de plis
    if is_integer(cv):
        if cv < 2:
            raise ValueError(f"cv must be an int >= 2, got {cv}")
        return
    # Forme objet : contrat de splitter sklearn
    if hasattr(cv, 'split') and hasattr(cv, 'get_n_splits'):
        return
    # Forme itérable : suite de couples d'index. Les chaînes sont exclues
    # explicitement — itérables au sens de Python, jamais une suite de
    # découpages
    if hasattr(cv, '__iter__') and not isinstance(cv, (str, bytes)):
        return
    raise TypeError(
        f"cv must be None, an int >= 2, a splitter exposing 'split' and "
        f"'get_n_splits', or an iterable of splits, got "
        f"{type(cv).__name__}"
    )


# Validation du score de validation croisée
def validate_cv_scoring(cv_scoring: Any) -> None:
    """Check that ``cv_scoring`` is a string or a callable.

    Args:
        cv_scoring: Value handed to ``__init__``.

    Raises:
        TypeError: If the value is neither a string nor a callable.

    Examples:
        >>> validate_cv_scoring('neg_mean_squared_error')
        >>> validate_cv_scoring(3)
        Traceback (most recent call last):
            ...
        TypeError: cv_scoring must be a str or a callable, got int
    """
    # Chaîne du registre sklearn, ou scorer appelable
    if isinstance(cv_scoring, str) or callable(cv_scoring):
        return
    raise TypeError(
        f"cv_scoring must be a str or a callable, "
        f"got {type(cv_scoring).__name__}"
    )


# Validation du nombre minimal d'observations de la validation croisée
def validate_min_cv_train_size(min_cv_train_size: Any) -> None:
    """Check that ``min_cv_train_size`` is an integer of at least 1.

    Numpy integers are admitted, booleans never.

    Args:
        min_cv_train_size: Value handed to ``__init__``.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If the value is below 1.

    Examples:
        >>> validate_min_cv_train_size(10)
        >>> validate_min_cv_train_size(0)
        Traceback (most recent call last):
            ...
        ValueError: min_cv_train_size must be >= 1, got 0
        >>> validate_min_cv_train_size(2.0)
        Traceback (most recent call last):
            ...
        TypeError: min_cv_train_size must be an int, got float
    """
    # Type d'abord : la comparaison n'a de sens que sur un entier
    if not is_integer(min_cv_train_size):
        raise TypeError(
            f"min_cv_train_size must be an int, "
            f"got {type(min_cv_train_size).__name__}"
        )
    if min_cv_train_size < 1:
        raise ValueError(
            f"min_cv_train_size must be >= 1, got {min_cv_train_size}"
        )


# Validation et normalisation du format de la fréquence cible
def validate_target_frequency_format(
    target_frequency: Union[str, Dict[Union[str, tuple], str]]
) -> Union[str, Dict[Union[str, tuple], str]]:
    """Validate the format and values of the ``target_frequency`` parameter.

    Args:
        target_frequency: Target frequency (string or dict mapping
            entities to frequencies). Dict entity keys may be given in
            scalar form for a single-level panel (``{'FR': 'M'}``).

    Returns:
        Normalized target_frequency. For a dict, BOTH sides are
        normalized: the frequency values via ``normalize_frequency`` and
        the entity keys into tuples via ``normalize_entity_key``, so that
        ``{'FR': 'M'}`` becomes ``{('FR',): 'M'}``. This is the only place
        where user-supplied entity keys enter the imputer, and every
        downstream consumer then indexes ``effective_target_frequency_``
        by tuple without any defensive fallback.

    Raises:
        ValueError: If the format is invalid or a frequency is not
            normalizable.
        TypeError: If target_frequency is neither a string nor a dict.

    Examples:
        >>> validate_target_frequency_format('M')
        'M'
        >>> validate_target_frequency_format({'FR': 'M'})
        {('FR',): 'M'}
    """
    # Distinction suivant le type de la fréquence cible
    # Cas où la fréquence cible est une chaîne de caractères
    if isinstance(target_frequency, str):
        # Normalisation
        try:
            return normalize_frequency(target_frequency, return_format="base")
        except ValueError as e:
            raise ValueError(f"Invalid target_frequency '{target_frequency}': {e}")

    # Cas où la fréquence cible est un dictionnaire
    elif isinstance(target_frequency, dict):
        # Vérification que le dictionnaire est non vide
        if not target_frequency:
            raise ValueError("target_frequency dict cannot be empty")

        # Initialisation des dictionnaires de fréquences valides et invalides
        validated_freqs: Dict[Tuple[Any, ...], str] = {}
        invalid_freqs: Dict[Any, str] = {}

        # Parcours des fréquences associées à chaque entité
        for entity, freq in target_frequency.items():
            # Vérification que la fréquence est une chaîne de caractères
            if not isinstance(freq, str):
                raise ValueError(
                    f"Frequency for entity '{entity}' must be a string, "
                    f"got {type(freq).__name__}"
                )
            # Normalisation de la clé d'entité en tuple ET de la fréquence
            try:
                validated_freqs[normalize_entity_key(entity)] = normalize_frequency(freq)
            except ValueError as e:
                invalid_freqs[entity] = str(e)

        # Construction du message d'erreur s'il existe des fréquences invalides
        if invalid_freqs:
            error_msg = "Invalid frequencies in target_frequency dict:\n"
            for entity, error in invalid_freqs.items():
                error_msg += f"  - Entity '{entity}': {error}\n"
            raise ValueError(error_msg.rstrip())
        return validated_freqs

    else:
        raise TypeError(
            f"target_frequency must be a string or dict, "
            f"got {type(target_frequency).__name__}"
        )


# Validation de l'estimateur
def validate_estimator(
    estimator: Optional[Union[BaseEstimator, Dict[str, BaseEstimator]]]
) -> None:
    """Validate that the estimator exposes ``fit`` and ``predict``.

    Args:
        estimator: Estimator or dict of estimators to validate. The dict
            form admits a ``'__default__'`` key, treated like any other.

    Raises:
        ValueError: If an estimator lacks a required method, or if the
            dict form is empty.

    Examples:
        >>> from sklearn.linear_model import LinearRegression
        >>> validate_estimator(LinearRegression())
        >>> validate_estimator({})
        Traceback (most recent call last):
            ...
        ValueError: estimator dict cannot be empty
    """
    # Cas où l'estimateur n'est pas spécifié : l'avertissement unique est
    # émis au fit, pas ici, pour ne pas se répéter à chaque clone
    if estimator is None:
        return

    # Distinction suivant le type de l'estimateur
    # Cas où il s'agit d'un dictionnaire
    if isinstance(estimator, dict):
        # Vérification que le dictionnaire est non vide
        if not estimator:
            raise ValueError("estimator dict cannot be empty")
        # Parcours des éléments du dictionnaire
        for var_name, est in estimator.items():
            # Vérification qu'il possède la méthode "fit"
            if not hasattr(est, 'fit') or not callable(getattr(est, 'fit')):
                raise ValueError(
                    f"Estimator for '{var_name}' must have a 'fit' method, "
                    f"got {type(est).__name__}"
                )
            # Vérification qu'il possède la méthode "predict"
            if not hasattr(est, 'predict') or not callable(getattr(est, 'predict')):
                raise ValueError(
                    f"Estimator for '{var_name}' must have a 'predict' method, "
                    f"got {type(est).__name__}"
                )
    # Cas où il s'agit d'un estimateur
    else:
        # Vérification qu'il possède la méthode "fit"
        if not hasattr(estimator, 'fit') or not callable(getattr(estimator, 'fit')):
            raise ValueError(
                f"estimator must have a 'fit' method, "
                f"got {type(estimator).__name__}"
            )
        # Vérification qu'il possède la méthode "predict"
        if not hasattr(estimator, 'predict') or not callable(getattr(estimator, 'predict')):
            raise ValueError(
                f"estimator must have a 'predict' method, "
                f"got {type(estimator).__name__}"
            )


# Validation du transformateur additif
def validate_additive_transformer(transformer: TransformerMixin) -> None:
    """Validate that the additive transformer exposes its two methods.

    The contract is ``fit_transform`` **and** ``inverse_transform``:
    the transformer is fitted and applied in one call at
    phase 2 of the fit, and inverted at ``inverse_transform``.

    Args:
        transformer: Transformer to validate.

    Raises:
        ValueError: If the transformer lacks a required method.

    Examples:
        >>> from sklearn.preprocessing import StandardScaler
        >>> validate_additive_transformer(StandardScaler())
        >>> validate_additive_transformer(object())
        Traceback (most recent call last):
            ...
        ValueError: additive_transformer must have methods: fit_transform, inverse_transform. Missing: fit_transform, inverse_transform. Got object
    """
    # Liste des méthodes requises
    required_methods = ['fit_transform', 'inverse_transform']
    # Initialisation de la liste des méthodes manquantes
    missing_methods = []

    # Parcours des méthodes requises
    for method_name in required_methods:
        # Vérification que le transformer possède la méthode en attribut
        if not hasattr(transformer, method_name) or not callable(
            getattr(transformer, method_name)
        ):
            missing_methods.append(method_name)

    # Construction du message d'erreur si des méthodes sont manquantes
    if missing_methods:
        raise ValueError(
            f"additive_transformer must have methods: "
            f"{', '.join(required_methods)}. "
            f"Missing: {', '.join(missing_methods)}. "
            f"Got {type(transformer).__name__}"
        )
