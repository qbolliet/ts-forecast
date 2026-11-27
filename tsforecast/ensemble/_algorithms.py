"""Pure update kernel of online expert aggregation: BOA and MLpol with pluggable losses.

``Loss`` and ``make_loss`` describe a loss and its gradient (``'mse'``, ``'mae'``,
``'pinball'``); ``excess_losses`` gives the excess loss of each expert over the
mixture on one round; ``BOAState`` / ``MLpolState`` hold the state of each
algorithm, updated by pure functions registered in ``ALGORITHMS``; ``run`` plays a
whole sequence of rounds. The module depends on numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from numbers import Real
from typing import Callable, Dict, NamedTuple, Tuple, Union

import numpy as np

# Plus petit exposant de l'étendue estimée des excès de perte de BOA : 2^-20 est jugé négligeable
_RANGE_EXPONENT_FLOOR = 20
# Noms des pertes prédéfinies
LOSS_NAMES: Tuple[str, ...] = ("mse", "mae", "pinball")


# Fonction de perte quadratique
def _squared_error(prediction: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared error ``(prediction - y)**2``, element-wise."""
    return np.square(prediction - y)


# Gradient de la perte quadratique par rapport à la prévision
def _squared_error_gradient(prediction: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient ``2 * (prediction - y)`` of the squared error, element-wise."""
    return 2.0 * (prediction - y)


# Fonction de perte absolue
def _absolute_error(prediction: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Absolute error ``|prediction - y|``, element-wise."""
    return np.abs(prediction - y)


# Sous-gradient de la perte absolue par rapport à la prévision
def _absolute_error_gradient(prediction: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sub-gradient ``sign(prediction - y)`` of the absolute error, element-wise."""
    return np.sign(prediction - y)


# Fonction de perte pinball d'un quantile
def _pinball(prediction: np.ndarray, y: np.ndarray, *, quantile: float) -> np.ndarray:
    """Pinball loss ``max(q * (y - p), (q - 1) * (y - p))`` of level ``q``, element-wise."""
    residual = y - prediction
    return np.maximum(quantile * residual, (quantile - 1.0) * residual)


# Sous-gradient de la perte pinball par rapport à la prévision
def _pinball_gradient(prediction: np.ndarray, y: np.ndarray, *, quantile: float) -> np.ndarray:
    """Sub-gradient ``1{p > y} - q`` of the pinball loss of level ``q``, element-wise."""
    return (prediction > y).astype(float) - quantile


# Classe de description d'une perte et de son gradient
@dataclass(frozen=True)
class Loss:
    """A loss of a forecast and its (sub-)gradient with respect to the forecast.

    Both functions take ``(prediction, y)`` arrays of the same shape (or
    broadcastable) and return an array of that shape. The gradient is used by
    the linearised ("gradient trick") version of the aggregation, the value by
    the version on the loss itself.

    Attributes:
        value: Loss ``value(prediction, y)``, element-wise.
        gradient: Derivative of the loss in the prediction,
            ``gradient(prediction, y)``, element-wise.
        name: Label of the loss, for display only.

    Examples:
        >>> import numpy as np
        >>> loss = Loss(value=lambda p, y: (p - y) ** 4, gradient=lambda p, y: 4 * (p - y) ** 3, name='quartic')
        >>> float(loss.value(np.array(3.0), np.array(1.0)))
        16.0
    """

    # Valeur de la perte, élément par élément
    value: Callable[[np.ndarray, np.ndarray], np.ndarray]
    # Dérivée de la perte par rapport à la prévision, élément par élément
    gradient: Callable[[np.ndarray, np.ndarray], np.ndarray]
    # Libellé de la perte
    name: str = "custom"


# Fonction de construction d'une perte depuis son nom
def make_loss(loss: Union[str, Loss], quantile: float = 0.5) -> Loss:
    """Return the :class:`Loss` named ``loss``, or ``loss`` itself when it is one.

    Args:
        loss: ``'mse'`` (squared error), ``'mae'`` (absolute error),
            ``'pinball'`` (quantile loss of level ``quantile``) or a
            :class:`Loss`.
        quantile: Level of the pinball loss, in ``(0, 1)``; ignored by the
            other losses.

    Returns:
        The loss and its gradient.

    Raises:
        ValueError: If the name is unknown, or the pinball level is not in
            ``(0, 1)``.

    Examples:
        >>> import numpy as np
        >>> pinball = make_loss('pinball', quantile=0.9)
        >>> pinball.value(np.array([1.0, 3.0]), np.array([2.0, 2.0])).round(2).tolist()
        [0.9, 0.1]
        >>> make_loss('mse').gradient(np.array([3.0]), np.array([1.0])).tolist()
        [4.0]
    """
    # Cas d'une perte déjà construite
    if isinstance(loss, Loss):
        return loss
    # Cas des pertes prédéfinies
    if loss == "mse":
        return Loss(_squared_error, _squared_error_gradient, "mse")
    if loss == "mae":
        return Loss(_absolute_error, _absolute_error_gradient, "mae")
    if loss == "pinball":
        # Niveau de quantile strictement entre 0 et 1
        if not isinstance(quantile, Real) or isinstance(quantile, bool) or not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be a number in (0, 1), got {quantile!r}")
        return Loss(
            partial(_pinball, quantile=float(quantile)),
            partial(_pinball_gradient, quantile=float(quantile)),
            "pinball",
        )
    # Cas d'un nom inconnu
    raise ValueError(f"loss must be one of {LOSS_NAMES} or a Loss instance, got {loss!r}")


# Fonction de calcul de la prévision du mélange et des excès de perte d'un tour
def excess_losses(
    weights: np.ndarray,
    experts: np.ndarray,
    y: np.ndarray,
    loss: Loss,
    gradient: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forecast of the mixture and excess loss of each expert on one round.

    A round holds one or several rows (the rows of one date of a panel) that
    share the same weights. The excess loss of expert ``k`` is the mean over
    the rows of the round of

    - ``loss'(p_i, y_i) * (x_ik - p_i)`` with ``gradient=True`` (loss
      linearised at the forecast ``p_i`` of the mixture);
    - ``loss(x_ik, y_i) - loss(p_i, y_i)`` with ``gradient=False``.

    A positive excess loss means the expert did worse than the mixture.

    Args:
        weights: Weights of the experts, shape ``(K,)``.
        experts: Forecasts of the experts, shape ``(n, K)``.
        y: Observations, shape ``(n,)``.
        loss: Loss and gradient.
        gradient: Whether to use the linearised loss.

    Returns:
        ``(prediction, excess)``: forecast of the mixture on each row, shape
        ``(n,)``, and excess loss of each expert, shape ``(K,)``.

    Examples:
        >>> import numpy as np
        >>> experts = np.array([[1.0, 3.0]])
        >>> prediction, excess = excess_losses(np.array([0.5, 0.5]), experts, np.array([1.0]), make_loss('mse'), True)
        >>> prediction.tolist(), excess.tolist()
        ([2.0], [-2.0, 2.0])
    """
    # Prévision du mélange sur chaque ligne du tour
    prediction = experts @ weights
    # Excès de perte de chaque expert sur chaque ligne, forme (n, K)
    if gradient:
        # Perte linéarisée en la prévision du mélange
        per_row = loss.gradient(prediction, y)[:, None] * (experts - prediction[:, None])
    else:
        # Perte elle-même
        per_row = loss.value(experts, y[:, None]) - loss.value(prediction, y)[:, None]
    # Moyenne sur les lignes du tour
    return prediction, per_row.mean(axis=0)


# Fonction de normalisation d'un vecteur de poids a priori
def _normalized(prior: np.ndarray) -> np.ndarray:
    """Return ``prior`` divided by its sum, as a new float array."""
    prior = np.array(prior, dtype=float)
    return prior / prior.sum()


# ---------------------------------------------------------------------------
# BOA : Bernstein Online Aggregation à taux d'apprentissage multiples
# ---------------------------------------------------------------------------


# Classe de l'état de BOA
@dataclass(frozen=True)
class BOAState:
    """State of the fully adaptive BOA procedure (one learning rate per expert).

    Attributes:
        log_prior: Logarithm of the prior weights ``pi_0``, shape ``(K,)``.
        learning_rates: Current learning rates ``eta``, shape ``(K,)``
            (zero before the first round).
        sq_sums: Sums of the squared excess losses, shape ``(K,)``.
        max_abs: Largest absolute excess loss seen, shape ``(K,)``.
        losses: Regularised cumulative excess losses ``L``, shape ``(K,)``.
    """

    # Logarithme des poids a priori
    log_prior: np.ndarray
    # Taux d'apprentissage courants (nuls avant le premier tour)
    learning_rates: np.ndarray
    # Sommes des carrés des excès de perte
    sq_sums: np.ndarray
    # Plus grand excès de perte en valeur absolue
    max_abs: np.ndarray
    # Excès de perte cumulés régularisés
    losses: np.ndarray


# Fonction d'initialisation de l'état de BOA
def boa_init(prior: np.ndarray) -> BOAState:
    """Return the initial BOA state for strictly positive prior weights.

    Args:
        prior: Prior weights, shape ``(K,)``; normalised to sum to one.

    Returns:
        The state before any round: weights equal to the prior.

    Examples:
        >>> import numpy as np
        >>> boa_weights(boa_init(np.array([1.0, 3.0]))).tolist()
        [0.25, 0.75]
    """
    log_prior = np.log(_normalized(prior))
    zeros = np.zeros_like(log_prior)
    return BOAState(log_prior, zeros, zeros, zeros, zeros)


# Fonction de calcul des poids de BOA
def boa_weights(state: BOAState) -> np.ndarray:
    """Weights ``pi_k ∝ eta_k * exp(-eta_k * L_k) * pi_0k`` of a BOA state.

    Before the first round (all learning rates zero) the weights are the
    prior. An expert whose learning rate is zero (prior weight one, i.e. a
    single expert) keeps its prior share.

    Args:
        state: BOA state.

    Returns:
        Weights on the simplex, shape ``(K,)``.

    Examples:
        >>> import numpy as np
        >>> state = boa_update(boa_init(np.ones(2)), np.array([-1.0, 1.0]))
        >>> boa_weights(state).round(4).tolist()
        [0.6225, 0.3775]
    """
    # Cas de l'état initial : poids a priori
    rates = state.learning_rates
    if not np.any(rates > 0):
        return np.exp(state.log_prior)
    # Logarithme des poids non normalisés, -inf pour un taux nul
    with np.errstate(divide="ignore"):
        log_weights = np.log(rates) + state.log_prior - rates * state.losses
    # Normalisation stable (soustraction du maximum avant l'exponentielle)
    weights = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()


# Fonction de mise à jour de l'état de BOA après un tour
def boa_update(state: BOAState, excess: np.ndarray) -> BOAState:
    """Update a BOA state with the excess losses of one round.

    With ``l`` the excess losses and ``eta`` the learning rates of the
    previous round:

    1. range ``E = 2**(k + 1)``, ``k >= -20`` the smallest integer with
       ``max |l| <= 2**k`` over the rounds seen;
    2. ``L += (l * (1 + eta * l) + E * 1{eta * l > 1/2}) / 2``;
    3. ``eta = min(1 / E, sqrt(log(1 / pi_0) / sum of l**2))``.

    Args:
        state: State before the round.
        excess: Excess loss of each expert on the round, shape ``(K,)``.

    Returns:
        The new state (the input state is not modified).

    Examples:
        >>> import numpy as np
        >>> state = boa_update(boa_init(np.ones(2)), np.array([-1.0, 1.0]))
        >>> state.learning_rates.tolist(), state.losses.tolist()
        ([0.5, 0.5], [-0.5, 0.5])
    """
    # Étendue estimée des excès de perte, puissance de 2 par expert
    max_abs = np.maximum(state.max_abs, np.abs(excess))
    with np.errstate(divide="ignore"):
        exponent = np.maximum(np.ceil(np.log2(max_abs)), -_RANGE_EXPONENT_FLOOR)
    ranges = np.power(2.0, exponent + 1.0)
    # Excès de perte cumulés régularisés, avec les taux du tour précédent
    previous = state.learning_rates
    losses = state.losses + 0.5 * (excess * (1.0 + previous * excess) + ranges * (previous * excess > 0.5))
    # Taux d'apprentissage adaptatifs (somme nulle des carrés : taux infini, borné par 1 / E)
    sq_sums = state.sq_sums + np.square(excess)
    adaptive = np.full_like(sq_sums, np.inf)
    np.divide(-state.log_prior, sq_sums, out=adaptive, where=sq_sums > 0)
    adaptive = np.sqrt(adaptive)
    learning_rates = np.minimum(1.0 / ranges, adaptive)
    return replace(state, learning_rates=learning_rates, sq_sums=sq_sums, max_abs=max_abs, losses=losses)


# ---------------------------------------------------------------------------
# MLpol : potentiel polynomial à taux d'apprentissage multiples
# ---------------------------------------------------------------------------


# Classe de l'état de MLpol
@dataclass(frozen=True)
class MLpolState:
    """State of the polynomially weighted average with multiple learning rates.

    Attributes:
        fallback_weights: Weights used while no expert has a positive
            cumulative regret, shape ``(K,)``.
        inverse_rates: Inverse learning rates ``1 / eta``, shape ``(K,)``
            (zero before the first round).
        regrets: Cumulative regrets ``R``, shape ``(K,)``.
        max_sq: Largest squared regret seen, ``B**2``.
    """

    # Poids utilisés tant qu'aucun regret cumulé n'est positif
    fallback_weights: np.ndarray
    # Inverses des taux d'apprentissage
    inverse_rates: np.ndarray
    # Regrets cumulés
    regrets: np.ndarray
    # Plus grand regret au carré
    max_sq: float


# Fonction d'initialisation de l'état de MLpol
def mlpol_init(prior: np.ndarray) -> MLpolState:
    """Return the initial MLpol state; ``prior`` gives the fallback weights.

    Args:
        prior: Strictly positive weights, shape ``(K,)``; normalised.

    Returns:
        The state before any round: weights equal to ``prior``.

    Examples:
        >>> import numpy as np
        >>> mlpol_weights(mlpol_init(np.ones(4))).tolist()
        [0.25, 0.25, 0.25, 0.25]
    """
    fallback = _normalized(prior)
    zeros = np.zeros_like(fallback)
    return MLpolState(fallback, zeros, zeros, 0.0)


# Fonction de calcul des poids de MLpol
def mlpol_weights(state: MLpolState) -> np.ndarray:
    """Weights ``p_k ∝ eta_k * max(R_k, 0)`` of an MLpol state.

    Args:
        state: MLpol state.

    Returns:
        Weights on the simplex, shape ``(K,)``; the fallback weights when no
        cumulative regret is positive.

    Examples:
        >>> import numpy as np
        >>> state = mlpol_update(mlpol_init(np.ones(2)), np.array([-1.0, 1.0]))
        >>> mlpol_weights(state).tolist()
        [1.0, 0.0]
    """
    # Parts positives des regrets cumulés, divisées par les inverses des taux
    positive = np.maximum(state.regrets, 0.0)
    # Un regret positif implique un inverse de taux positif (regret non nul observé)
    weights = np.divide(positive, state.inverse_rates, out=np.zeros_like(positive), where=positive > 0)
    total = weights.sum()
    # Cas d'aucun regret positif : poids de repli
    if total <= 0:
        return state.fallback_weights.copy()
    return weights / total


# Fonction de mise à jour de l'état de MLpol après un tour
def mlpol_update(state: MLpolState, excess: np.ndarray) -> MLpolState:
    """Update an MLpol state with the excess losses of one round.

    With the instantaneous regrets ``r = -excess`` and ``B_t**2`` the
    largest squared regret seen up to round ``t``:
    ``R += r`` and ``1 / eta += r**2 + B_t**2 - B_{t-1}**2``. Starting from
    ``1 / eta = 0`` makes the weights invariant to the scale of the losses.

    Args:
        state: State before the round.
        excess: Excess loss of each expert on the round, shape ``(K,)``.

    Returns:
        The new state (the input state is not modified).

    Examples:
        >>> import numpy as np
        >>> state = mlpol_update(mlpol_init(np.ones(2)), np.array([-1.0, 2.0]))
        >>> state.regrets.tolist(), state.inverse_rates.tolist(), state.max_sq
        ([1.0, -2.0], [5.0, 8.0], 4.0)
    """
    # Regrets instantanés et plus grand regret au carré
    regret = -excess
    squared = np.square(regret)
    max_sq = max(state.max_sq, float(squared.max()))
    # Inverses des taux : carrés des regrets plus la croissance de l'étendue
    inverse_rates = state.inverse_rates + squared + (max_sq - state.max_sq)
    return replace(state, inverse_rates=inverse_rates, regrets=state.regrets + regret, max_sq=max_sq)


# ---------------------------------------------------------------------------
# Registre et déroulement d'une séquence
# ---------------------------------------------------------------------------


# Classe regroupant les trois fonctions pures d'un algorithme
class Algorithm(NamedTuple):
    """The three pure functions of an aggregation algorithm."""

    # Construction de l'état initial depuis les poids a priori
    init: Callable[[np.ndarray], object]
    # Poids courants d'un état
    weights: Callable[[object], np.ndarray]
    # Nouvel état après les excès de perte d'un tour
    update: Callable[[object, np.ndarray], object]


# Dictionnaire nom d'algorithme -> fonctions pures
ALGORITHMS: Dict[str, Algorithm] = {
    "BOA": Algorithm(boa_init, boa_weights, boa_update),
    "MLpol": Algorithm(mlpol_init, mlpol_weights, mlpol_update),
}


# Fonction de déroulement d'une séquence de tours
def run(
    algorithm: str,
    state: object,
    experts: np.ndarray,
    y: np.ndarray,
    rounds: np.ndarray,
    loss: Loss,
    gradient: bool,
) -> Tuple[object, np.ndarray, np.ndarray]:
    """Play a sequence of rounds and return the final state.

    Rows are sorted by round; the rows of one round share the weights computed
    before the round, and the state is updated once per round with their mean
    excess loss.

    Args:
        algorithm: Key of :data:`ALGORITHMS` (``'BOA'`` or ``'MLpol'``).
        state: State before the sequence (from the ``init`` function of the
            algorithm, or the end of a previous sequence).
        experts: Forecasts of the experts, shape ``(n, K)``.
        y: Observations, shape ``(n,)``.
        rounds: Round of each row, non-decreasing, shape ``(n,)``.
        loss: Loss and gradient.
        gradient: Whether to use the linearised loss.

    Returns:
        ``(state, weights, prediction)``: final state, weights used for each
        row, shape ``(n, K)``, and forecast of the mixture on each row, shape
        ``(n,)``.

    Examples:
        >>> import numpy as np
        >>> experts = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        >>> y = np.array([1.0, 2.0, 3.0])
        >>> mlpol = ALGORITHMS['MLpol']
        >>> state, weights, prediction = run('MLpol', mlpol.init(np.ones(2)), experts, y, np.arange(3), make_loss('mse'), True)
        >>> weights.tolist(), prediction.tolist()
        ([[0.5, 0.5], [1.0, 0.0], [1.0, 0.0]], [2.5, 2.0, 3.0])
    """
    # Fonctions pures de l'algorithme
    functions = ALGORITHMS[algorithm]
    # Poids et prévision de chaque ligne, remplis tour par tour
    weights = np.empty(experts.shape, dtype=float)
    prediction = np.empty(len(y), dtype=float)
    # Cas d'une séquence vide : état inchangé
    if len(y) == 0:
        return state, weights, prediction
    # Bornes des tours : positions où le code de tour change
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(rounds)) + 1, [len(y)]))
    # Parcours des tours dans l'ordre : poids avant observation, puis mise à jour
    for start, stop in zip(bounds[:-1], bounds[1:]):
        current = functions.weights(state)
        round_prediction, excess = excess_losses(current, experts[start:stop], y[start:stop], loss, gradient)
        weights[start:stop] = current
        prediction[start:stop] = round_prediction
        state = functions.update(state, excess)
    return state, weights, prediction
