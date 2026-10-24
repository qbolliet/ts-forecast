"""Aggregated warnings of one ``HighFrequencyImputer`` pass, held back then emitted once.

``WarningAccumulator`` gathers the per-step incidents of a fit or of a replay — degraded
steps, never-observed pairs left empty, entities outside the frozen plan — and emits one
``UserWarning`` per family at the end of the pass, never one per variable and per stage.
"""
# Importation des modules
# Modules de base
import warnings
from typing import Any, Iterable, List, Literal, Tuple

# Passe dont relèvent les avertissements, nommée dans le message des étapes dégradées
PassName = Literal['fit', 'transform']


# Accumulateur des avertissements agrégés d'une passe
class WarningAccumulator:
    """Hold back the aggregated warnings of one pass, then emit one per family.

    Three families, emitted in this order when non-empty:

    1. **degraded steps** — a fallback announced at fit time, or a prediction
       that failed or produced NaN; every message is kept, repeats included;
    2. **unimputed pairs** — ``(entity..., column)`` pairs a step without
       anchor could not impute, named once each, sorted by ``repr``;
    3. **entities outside the frozen plan** — the cases a replay meets and
       the plan does not cover (entities born at transform time, pairs
       observed off the calendar bounds of their frequency).

    Attributes:
        pass_name: Pass the warnings belong to, ``'fit'`` or ``'transform'``.
        degraded_steps: Messages of the degraded steps, in order of arrival.
        unimputed_pairs: Pairs left empty, in order of arrival, repeats included.
        outside_plan_cases: Messages of the cases outside the frozen plan.

    Examples:
        >>> accumulator = WarningAccumulator('fit')
        >>> accumulator.add_degraded_step("'q1' at stage M: empty training set")
        >>> accumulator.add_unimputed_pairs([('IT', 'v'), ('IT', 'v')])
        >>> for message in accumulator.messages():
        ...     print(message.split(':')[0])
        1 imputation step(s) degraded during the fit
        1 unobserved (entity, column) pair(s) could not be imputed
    """

    # Initialisation des trois familles, vides
    def __init__(self, pass_name: PassName) -> None:
        """Start an empty accumulator for one pass.

        Args:
            pass_name: Pass the warnings belong to, ``'fit'`` or
                ``'transform'``; it is named by the message of the degraded
                steps.
        """
        self.pass_name = pass_name
        self.degraded_steps: List[str] = []
        self.unimputed_pairs: List[Tuple[Any, ...]] = []
        self.outside_plan_cases: List[str] = []

    # Enregistrement d'une étape dégradée
    def add_degraded_step(self, message: str) -> None:
        """Record one degraded step.

        Args:
            message: Description of the step and of the cause of its
                degradation, one line of the aggregated message.

        Examples:
            >>> accumulator = WarningAccumulator('transform')
            >>> accumulator.add_degraded_step('x')
            >>> accumulator.degraded_steps
            ['x']
        """
        self.degraded_steps.append(message)

    # Enregistrement de couples restés vides
    def add_unimputed_pairs(self, pairs: Iterable[Tuple[Any, ...]]) -> None:
        """Record the pairs a step without anchor could not impute.

        Args:
            pairs: ``(entity..., column)`` keys of the pairs left empty.

        Examples:
            >>> accumulator = WarningAccumulator('fit')
            >>> accumulator.add_unimputed_pairs([('IT', 'v')])
            >>> accumulator.unimputed_pairs
            [('IT', 'v')]
        """
        self.unimputed_pairs.extend(pairs)

    # Enregistrement d'un cas hors du plan figé
    def add_outside_plan_case(self, message: str) -> None:
        """Record one case the frozen plan does not cover.

        Args:
            message: Description of the case, one line of the aggregated
                message.

        Examples:
            >>> accumulator = WarningAccumulator('transform')
            >>> accumulator.add_outside_plan_case('entity ES left untouched')
            >>> accumulator.outside_plan_cases
            ['entity ES left untouched']
        """
        self.outside_plan_cases.append(message)

    # Messages agrégés, un par famille non vide
    def messages(self) -> List[str]:
        """Render the aggregated messages, one per non-empty family, in emission order.

        Returns:
            The messages :meth:`emit` would emit, in the same order; empty
            when nothing was recorded.

        Examples:
            >>> accumulator = WarningAccumulator('transform')
            >>> accumulator.messages()
            []
            >>> accumulator.add_outside_plan_case('a')
            >>> accumulator.add_outside_plan_case('b')
            >>> print(accumulator.messages()[0])
            2 case(s) of entities outside the frozen plan at transform time:
              - a
              - b
        """
        # Initialisation de la liste des messages
        messages: List[str] = []
        # Étapes dégradées : chaque message compte, répétitions comprises
        if self.degraded_steps:
            messages.append(
                f"{len(self.degraded_steps)} imputation step(s) degraded during the "
                f"{self.pass_name}:\n  - " + "\n  - ".join(self.degraded_steps)
            )
        # Couples restés vides : nommés une fois chacun, dans un ordre déterministe
        if self.unimputed_pairs:
            named = sorted(set(self.unimputed_pairs), key=repr)
            messages.append(
                f"{len(named)} unobserved (entity, column) pair(s) could not be "
                f"imputed: no prediction on the target grid (no usable covariate, "
                f"or the prediction failed), and no interpolation fallback exists "
                f"without an anchor; their cells stay NaN and unmarked: {named}"
            )
        # Cas hors du plan figé, propres au rejeu
        if self.outside_plan_cases:
            messages.append(
                f"{len(self.outside_plan_cases)} case(s) of entities outside the frozen "
                f"plan at transform time:\n  - " + "\n  - ".join(self.outside_plan_cases)
            )
        return messages

    # Émission des avertissements agrégés
    def emit(self) -> None:
        """Emit one ``UserWarning`` per non-empty family, in the order of :meth:`messages`.

        Each warning is attributed to the line that calls this method: the
        warnings of a pass keep the module of the imputer — what a
        ``module=`` filter matches — and each pass keeps its own duplicate
        registry.

        Examples:
            >>> import warnings
            >>> accumulator = WarningAccumulator('fit')
            >>> accumulator.add_degraded_step('x')
            >>> with warnings.catch_warnings(record=True) as caught:
            ...     warnings.simplefilter('always')
            ...     accumulator.emit()
            >>> [item.category.__name__ for item in caught]
            ['UserWarning']
        """
        for message in self.messages():
            # Attribution à l'appelant, jamais à ce module
            warnings.warn(message, UserWarning, stacklevel=2)
