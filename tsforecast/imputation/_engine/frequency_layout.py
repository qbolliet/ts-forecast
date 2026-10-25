"""Frequency layout of a ``HighFrequencyImputer`` fit: frequency of each (entity, column) pair,
effective target frequency, variable categories and stage progression.

``plan_frequencies`` computes the frozen ``FrequencyLayout`` from the working frame, the target
frequency and the parameters it depends on; the layout then answers the per-stage questions of
the fit and of the replay (imputable groups, pairs without anchor, frequencies by column). The
module also holds the pure helpers both passes share: calendar check of the anchors
(``misaligned_anchor_pairs``), stage labels and per-row stage frequency.
"""
# Importation des modules
# Modules de base
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# Manipulation de données
import pandas as pd

# Utilitaires du package
from ...panel.utils import (
    get_entity_codes,
    iter_entity_blocks,
    normalize_entity_key,
    split_variable_key,
)
from ...utils._constants import MONTH_BASED_FREQUENCIES
from ...utils.frequency.utils import (
    detect_frequency,
    detect_index_frequency,
    get_frequency_order,
    is_higher_frequency,
    normalize_frequency,
)
from .._components.target_frequency_validator import TargetFrequencyValidator

# Alias de types
EntityKey = Tuple[Any, ...]
# Clé d'un couple : nom de colonne (série) ou (entité..., colonne) (panel)
PairKey = Union[str, Tuple[Any, ...]]
# Fréquence d'une étape : scalaire (série) ou liaison par entité (panel)
StageFrequency = Union[str, Dict[EntityKey, str]]
# Fréquences par colonne, forme lue par les composants
ColumnFrequencies = Dict[str, Union[str, Dict[EntityKey, Optional[str]]]]
# Rappel de journalisation
LogCallback = Callable[[str], None]


# Plan des fréquences d'un fit
@dataclass(frozen=True)
class FrequencyLayout:
    """Frequencies of a fit, as detected, classified and ordered into stages.

    Computed once per fit by :func:`plan_frequencies`, then read unchanged by
    the stage execution and by every replay. The imputer publishes three of
    its fields as fitted attributes — ``detected_frequencies_``,
    ``variable_categories_``, ``frequency_progression_`` — and its target as
    ``effective_target_frequency_``, the very same objects.

    Attributes:
        is_panel: Whether the data is a panel: keys are then
            ``(entity..., column)`` tuples and stages are per-entity bindings.
        detected_frequencies: Frequency of each pair the fit observes on the
            calendar bounds of that frequency.
        undetected: Pairs observed too little (fewer than two observations,
            or none) to carry a frequency, in detection order.
        misaligned: Pairs observed off the calendar bounds of their own
            frequency, left out of the classification, sorted.
        target_frequency: Effective target frequency: a string on a time
            series, a binding entity -> frequency on a panel (entities with
            no detected pair left out).
        categories: Classification of the detected pairs at the target
            frequency, under the keys ``'aggregate'``, ``'impute'`` and
            ``'target_freq'``.
        progression: Ordered stage frequencies, the target frequency last.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2021-01-31', periods=24, freq='ME')
        >>> data = pd.DataFrame({'m1': range(24), 'q1': None}, index=dates, dtype=float)
        >>> data.loc[dates[2::3], 'q1'] = range(8)
        >>> layout = plan_frequencies(
        ...     data, 'M', is_panel=False, entities=None,
        ...     impute_intermediate_frequencies=False, on_frequency_mismatch='error',
        ... )
        >>> layout.detected_frequencies, layout.progression
        ({'m1': 'M', 'q1': 'Q'}, ['M'])
        >>> layout.categories['impute']
        ['q1']
    """

    is_panel: bool
    detected_frequencies: Dict[PairKey, str]
    undetected: Tuple[PairKey, ...]
    misaligned: Tuple[PairKey, ...]
    target_frequency: StageFrequency
    categories: Dict[str, List[PairKey]]
    progression: List[StageFrequency]

    # Fréquences d'une colonne, entité par entité
    def column_frequencies_by_entity(self, column: str) -> Dict[EntityKey, str]:
        """Return the frequency of one column for each entity observing it.

        Args:
            column: Bare column name.

        Returns:
            Mapping from entity key tuple to detected frequency. A time
            series yields the single key ``()``; a column with no detected
            frequency yields an empty dict.

        Examples:
            >>> layout = FrequencyLayout(
            ...     is_panel=True,
            ...     detected_frequencies={('FR', 'v'): 'Y', ('DE', 'v'): 'Q', ('FR', 'm1'): 'M'},
            ...     undetected=(), misaligned=(), target_frequency={('FR',): 'M'},
            ...     categories={}, progression=[],
            ... )
            >>> layout.column_frequencies_by_entity('v')
            {('FR',): 'Y', ('DE',): 'Q'}
        """
        # Itération + filtre :
        # un panel peut porter la même colonne à des fréquences différentes
        # selon l'entité
        return {
            split_variable_key(key)[0]: freq
            for key, freq in self.detected_frequencies.items()
            if split_variable_key(key)[1] == column
        }

    # Conversion des fréquences détectées à la forme des consommateurs
    def frequencies_by_column(
        self,
        transform_frequencies: Optional[Mapping[PairKey, str]] = None,
    ) -> ColumnFrequencies:
        """Return the detected frequencies keyed by column.

        Two shapes coexist on purpose. ``detected_frequencies`` is keyed by
        ``(entity..., column)`` on a panel. ``CovariateMaterializer``,
        ``StageScaler`` and ``TrainingSetBuilder`` all read the other shape,
        ``{column: frequency | {entity: frequency}}``. This adapter is the
        single crossing point between the two, so no consumer ever
        re-implements the split.

        Args:
            transform_frequencies: Frequencies a replay detected for the pairs
                the fit never observed. None or empty at fit time. They never
                override a pair of the fit.

        Returns:
            Mapping from column name to a scalar frequency (time series, or a
            panel where every entity agrees) or to a per-entity mapping.

        Examples:
            >>> layout = FrequencyLayout(
            ...     is_panel=True,
            ...     detected_frequencies={
            ...         ('FR', 'm1'): 'M', ('DE', 'm1'): 'M', ('FR', 'v'): 'Y', ('DE', 'v'): 'Q',
            ...     },
            ...     undetected=(), misaligned=(), target_frequency={('FR',): 'M', ('DE',): 'M'},
            ...     categories={}, progression=[],
            ... )
            >>> layout.frequencies_by_column()
            {'m1': 'M', 'v': {('FR',): 'Y', ('DE',): 'Q'}}
            >>> layout.frequencies_by_column({('ES', 'm1'): 'Q', ('FR', 'v'): 'M'})['m1']
            {('ES',): 'Q', ('FR',): 'M', ('DE',): 'M'}
        """
        # Couples apparus au transform : aucune fréquence de fit ne les porte,
        # la détection faite sur les données du transform est leur seule
        # source. Les couples du fit, eux, ne sont jamais recouverts
        pairs: Dict[PairKey, str] = {
            **(transform_frequencies or {}),
            **self.detected_frequencies,
        }

        # Cas des séries temporelles : les clés sont déjà des noms de colonnes
        if not self.is_panel:
            return dict(pairs)

        # Cas du panel : regroupement par colonne, puis repli sur la forme
        # scalaire quand toutes les entités s'accordent
        by_column: Dict[str, Dict[EntityKey, str]] = {}
        for key, freq in pairs.items():
            entity, column = split_variable_key(key)
            by_column.setdefault(column, {})[entity] = freq

        result: ColumnFrequencies = {}
        for column, per_entity in by_column.items():
            unique = set(per_entity.values())
            result[column] = per_entity if len(unique) > 1 else unique.pop()
        return result

    # Fréquences remises au constructeur du jeu d'entraînement
    def training_frequencies(
        self,
        column: str,
        freqs_by_column: ColumnFrequencies,
    ) -> ColumnFrequencies:
        """Return the frequencies by column with the target column made explicit.

        :meth:`frequencies_by_column` folds a column into its scalar form
        when every detected entity agrees, which lends that frequency to an
        entity observing the column at an undetectable frequency (a single
        observation). The training set must not borrow it: ``f_block(e)`` is
        the entity's OWN frequency, and a lent one would scale its rows by
        the wrong divisor. The target column is therefore handed over per
        entity, each undetectable pair — or pair observed off the calendar
        bounds of its frequency — mapped to None explicitly — the
        ``TrainingSetBuilder`` then drops that block, whatever the other
        entities do. Covariate entries are left untouched.

        Args:
            column: Column whose training set is composed.
            freqs_by_column: Output of :meth:`frequencies_by_column`.

        Returns:
            ``freqs_by_column`` itself when no pair of ``column`` is
            undetectable, otherwise a copy whose ``column`` entry is the
            explicit per-entity mapping.

        Examples:
            >>> layout = FrequencyLayout(
            ...     is_panel=True,
            ...     detected_frequencies={('FR', 'v'): 'Y', ('DE', 'v'): 'Y'},
            ...     undetected=(('IT', 'v'),), misaligned=(),
            ...     target_frequency={('FR',): 'M', ('DE',): 'M', ('IT',): 'M'},
            ...     categories={}, progression=[],
            ... )
            >>> layout.training_frequencies('v', {'v': 'Y'})
            {'v': {('FR',): 'Y', ('DE',): 'Y', ('IT',): None}}
        """
        # Couples indétectables ou observés hors de leur grille calendaire de
        # la colonne : rien à expliciter sans eux
        undetected = [
            split_variable_key(key)[0]
            for key in (*self.undetected, *self.misaligned)
            if split_variable_key(key)[1] == column
        ]
        if not self.is_panel or not undetected:
            return freqs_by_column

        # Liaison par entité : fréquences propres, None pour les indétectables
        binding: Dict[EntityKey, Optional[str]] = dict(
            self.column_frequencies_by_entity(column)
        )
        binding.update({entity: None for entity in undetected})
        return {**freqs_by_column, column: binding}

    # Classification des couples relativement à une fréquence de prédiction
    def classify(self, prediction_frequency: StageFrequency) -> Dict[str, List[PairKey]]:
        """Classify the detected pairs relative to a prediction frequency.

        See :func:`classify_variables_at_frequency`.

        Args:
            prediction_frequency: Frequency of the stage.

        Returns:
            Dict with keys ``'aggregate'``, ``'impute'`` and
            ``'target_freq'``, each holding a list of pair keys.

        Examples:
            >>> layout = FrequencyLayout(
            ...     is_panel=False, detected_frequencies={'m1': 'M', 'a1': 'Y'},
            ...     undetected=(), misaligned=(), target_frequency='Q',
            ...     categories={}, progression=[],
            ... )
            >>> layout.classify('Q')
            {'aggregate': ['m1'], 'impute': ['a1'], 'target_freq': []}
        """
        return classify_variables_at_frequency(
            self.detected_frequencies, prediction_frequency, is_panel=self.is_panel
        )

    # Regroupement des couples imputables par fréquence source
    def imputable_groups(
        self,
        prediction_frequency: StageFrequency,
        *,
        impute_unobserved_entities: bool,
    ) -> Dict[Tuple[str, Optional[str]], Tuple[EntityKey, ...]]:
        """Group the imputable ``(entity, column)`` pairs of a stage by source frequency.

        Each ``(column, f_var)`` group yields one plan step at the stage,
        all of them sharing the model fitted on the mutualized training set.

        Under ``impute_unobserved_entities=True`` a further group
        ``(column, None)`` gathers the entities that never observe the column,
        at every stage binding them. This is the **single** entry point of
        that capability: the classification is left untouched, which is
        exactly what keeps ``categories`` and ``progression`` identical with
        and without the parameter -- such a pair has no source frequency, so
        it joins no frequency set and, travelling the stages, adds none.

        Args:
            prediction_frequency: Frequency of the stage.
            impute_unobserved_entities: Whether the pairs never observed are
                imputed from the other entities.

        Returns:
            Mapping from ``(column, source frequency)`` to the tuple of
            entity keys of the group, each tuple sorted for determinism. On a
            time series the entity key is ``()``. The source frequency is
            ``None`` for an unanchored group.

        Examples:
            >>> layout = FrequencyLayout(
            ...     is_panel=True,
            ...     detected_frequencies={('FR', 'm1'): 'M', ('IT', 'm1'): 'M', ('FR', 'v'): 'Y'},
            ...     undetected=(('IT', 'v'),), misaligned=(),
            ...     target_frequency={('FR',): 'M', ('IT',): 'M'},
            ...     categories={}, progression=[],
            ... )
            >>> layout.imputable_groups({('FR',): 'M', ('IT',): 'M'}, impute_unobserved_entities=True)
            {('v', 'Y'): (('FR',),), ('v', None): (('IT',),)}
        """
        # Regroupement des couples imputables par (colonne, fréquence source)
        groups: Dict[Tuple[str, Optional[str]], List[EntityKey]] = {}
        categories = self.classify(prediction_frequency)
        for key in categories['impute']:
            entity, column = split_variable_key(key)
            source_freq = normalize_frequency(
                self.detected_frequencies[key], return_format='base'
            )
            groups.setdefault((column, source_freq), []).append(entity)

        # Groupes sans ancre, ajoutés après les groupes ancrés : l'ordre du
        # plan reste celui des fréquences sources, la nouveauté en queue
        for entity, column in self.unanchored_pairs_at(
            prediction_frequency, impute_unobserved_entities=impute_unobserved_entities
        ):
            groups.setdefault((column, None), []).append(entity)

        # Tri des entités de chaque groupe : l'ordre du plan ne doit dépendre
        # ni de l'ordre des colonnes ni de celui des lignes en entrée
        return {
            group_key: tuple(sorted(entities, key=repr))
            for group_key, entities in groups.items()
        }

    # Lecture de la fréquence cible d'une entité
    def entity_target_frequency(self, entity: Optional[EntityKey]) -> Optional[str]:
        """Read the target frequency bound to one entity.

        Args:
            entity: Entity key, None on a time series.

        Returns:
            The normalized target frequency of that entity, or None when the
            binding does not name it.

        Examples:
            >>> layout = FrequencyLayout(
            ...     is_panel=True, detected_frequencies={}, undetected=(), misaligned=(),
            ...     target_frequency={('FR',): 'ME'}, categories={}, progression=[],
            ... )
            >>> layout.entity_target_frequency(('FR',)), layout.entity_target_frequency(('IT',))
            ('M', None)
        """
        # Les deux formes de la cible, scalaire et par entité, lues ici seulement
        target = self.target_frequency
        frequency = target.get(entity) if isinstance(target, dict) else target
        if frequency is None:
            return None
        return normalize_frequency(frequency, return_format='base')

    # Sélection des couples imputables sans aucune ancre
    def unanchored_pairs_at(
        self,
        prediction_frequency: StageFrequency,
        *,
        impute_unobserved_entities: bool,
    ) -> List[Tuple[EntityKey, str]]:
        """Select the never-observed pairs imputable at one stage.

        A pair the fit could detect no frequency for joins every stage of its
        entity's progression, not only the last: the rule is the general one,
        of which the single stage of
        ``impute_intermediate_frequencies=False`` is the degenerate case.
        Reading the binding rather than the target frequency is what keeps the
        parameter independent of axis 2 -- one rule, whose consequences follow
        the progression, instead of a behavior that changes with the modality.

        Two conditions gate a pair, beyond ``impute_unobserved_entities``:

        1. the stage binds the entity, which a stage its target-frequency
           group does not travel through never does;
        2. at least one other entity observes the column, without which the
           mutualized training set would be empty and the step would have no
           model to share.

        No condition is put on the level the entity reaches: the cells
        produced at an intermediate stage are anchored on nothing, exactly
        like those of the last one -- being anchored on an earlier prediction
        of itself is not being anchored -- so every stage rescales nothing and
        emits ``MODEL_UNANCHORED``.

        Args:
            prediction_frequency: Frequency of the stage — on a panel, the
                only case where the capability plays, always a binding per
                entity.
            impute_unobserved_entities: Whether the pairs never observed are
                imputed from the other entities.

        Returns:
            List of ``(entity key, column)`` pairs, in the order of
            :attr:`undetected`. Empty under
            ``impute_unobserved_entities=False``, and empty on a time series,
            which has no other entity to learn from.

        Examples:
            >>> layout = FrequencyLayout(
            ...     is_panel=True,
            ...     detected_frequencies={('FR', 'v'): 'Y'},
            ...     undetected=(('IT', 'v'), ('IT', 'w')), misaligned=(),
            ...     target_frequency={('FR',): 'M', ('IT',): 'M'},
            ...     categories={}, progression=[],
            ... )
            >>> layout.unanchored_pairs_at({('IT',): 'M'}, impute_unobserved_entities=True)
            [(('IT',), 'v')]
        """
        # Capacité fermée par défaut, et sans objet sur une série temporelle :
        # l'imputation sans ancre est apprise sur les AUTRES entités
        if not impute_unobserved_entities or not self.is_panel:
            return []

        # Parcours des couples jamais observés
        pairs: List[Tuple[EntityKey, str]] = []
        for key in self.undetected:
            entity, column = split_variable_key(key)

            # Fréquence de prédiction liée à l'entité par l'étape : sur un
            # panel, une étape est toujours une liaison par entité
            if prediction_frequency.get(entity) is None:
                continue

            # Colonne observée par au moins une autre entité : sans elle, le
            # jeu mutualisé est vide et il n'y a aucun modèle à partager
            if not self.column_frequencies_by_entity(column):
                continue

            pairs.append((entity, column))
        return pairs


# Calcul du plan des fréquences d'un fit
def plan_frequencies(
    X_work: pd.DataFrame,
    target_frequency: Union[str, Dict[EntityKey, str]],
    *,
    is_panel: bool,
    entities: Optional[Sequence[EntityKey]],
    impute_intermediate_frequencies: Any,
    on_frequency_mismatch: str,
    log: Optional[LogCallback] = None,
) -> FrequencyLayout:
    """Detect, classify and order the frequencies of a fit.

    Steps, in this order — the order decides which error is raised first:

    1. expansion of the target frequency into a binding per entity on a
       panel, a dict target naming every entity (:func:`expand_target_frequency`);
    2. detection of the frequency of each ``(entity, column)`` pair, the pairs
       observed off the calendar bounds of their frequency being left out
       under a single warning (:func:`misaligned_anchor_pairs`), the
       undetectable ones being set aside;
    3. withdrawal from the target of the entities with no detected pair,
       then validation of the target against the detected frequencies;
    4. classification of the pairs at the target frequency, then
       progression of stage frequencies.

    Args:
        X_work: Working frame of the fit, before the additive transformer.
        target_frequency: Target frequency, already normalized by the format
            validation (a string, or a dict keyed by entity tuples).
        is_panel: Whether ``X_work`` is a panel.
        entities: Normalized entity keys of the panel, None on a time series.
        impute_intermediate_frequencies: Axis-2 modality: ``False`` yields a
            single stage, the target; ``'covariates_only'`` and ``True`` the
            complete progression.
        on_frequency_mismatch: ``'error'`` or ``'warn'``, handed to the
            target-frequency validation.
        log: Logging callback, or None.

    Returns:
        The frozen layout of the fit.

    Raises:
        ValueError: If a dict ``target_frequency`` is given on a time
            series or misses entities of the panel, if no frequency can be
            detected for any pair, or if the
            target frequency does not fit the detected frequencies under
            ``on_frequency_mismatch='error'``.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> dates = pd.date_range('2021-01-31', periods=24, freq='ME')
        >>> idx = pd.MultiIndex.from_product([['FR', 'DE'], dates], names=['entity', 'date'])
        >>> panel = pd.DataFrame({'m1': np.arange(48.0), 'v': np.nan}, index=idx)
        >>> panel.loc[('FR', dates[11::12]), 'v'] = [120.0, 132.0]
        >>> panel.loc[('DE', dates[2::3]), 'v'] = np.arange(8.0) * 5 + 30
        >>> layout = plan_frequencies(
        ...     panel, 'M', is_panel=True, entities=[('FR',), ('DE',)],
        ...     impute_intermediate_frequencies=True, on_frequency_mismatch='error',
        ... )
        >>> layout.categories['impute']
        [('DE', 'v'), ('FR', 'v')]
        >>> [sorted(stage.items()) for stage in layout.progression]
        [[(('DE',), 'Q'), (('FR',), 'Q')], [(('DE',), 'M'), (('FR',), 'M')]]
    """
    # Expansion de la fréquence cible en liaison par entité sur un panel
    effective_target = expand_target_frequency(
        target_frequency, is_panel=is_panel, entities=entities
    )

    # Détection des fréquences par (entité, colonne) : sur un panel, la
    # même colonne peut porter une fréquence différente selon l'entité
    # et les clés sont alors des tuples (entité..., colonne). Un couple
    # indétectable (moins de deux observations) vaut None, sans lever d'erreur
    raw_frequencies = detect_frequency(data=X_work)

    # Couples observés hors de la grille calendaire de leur fréquence
    # (exercice décalé, ancre annuelle au 1er décembre d'un index MS) :
    # écartés de la classification comme des covariables, et nommés par un
    # avertissement unique
    misaligned = misaligned_anchor_pairs(X_work, raw_frequencies, is_panel=is_panel)
    if misaligned:
        warnings.warn(
            f"{len(misaligned)} (entity, column) pair(s) are observed "
            f"off the calendar bounds of their own frequency and are left out "
            f"(neither imputed nor used as covariates): "
            f"{list(misaligned)}. A monthly, quarterly or yearly "
            f"value must be dated at the end of its calendar period on a "
            f"period-end index, at its start on a period-start index: re-date "
            f"a calendar value dated elsewhere (a yearly value dated 1 December "
            f"on a period-start index belongs on 1 January). A non-calendar "
            f"period, such as a fiscal year closing in June, has no calendar "
            f"reading and cannot be imputed by this class.",
            UserWarning,
        )

    # Écartement des couples sans fréquence détectable : une entité
    # n'observant jamais une colonne n'a pas de fréquence pour elle.
    # "CovariateMaterializer.eligible_columns" côté covariables et la
    # progression côté cible en tirent chacun les conséquences
    undetected = tuple(key for key, freq in raw_frequencies.items() if freq is None)
    detected = {
        key: freq for key, freq in raw_frequencies.items()
        if freq is not None and key not in misaligned
    }
    # Vérification que le dictionnaire des fréquences détectées est non vide
    if not detected:
        raise ValueError("Could not detect frequency for any column")
    # Journalisation des couples jamais observés
    if undetected and log is not None:
        log(
            f"[fit] {len(undetected)} (entity, column) "
            f"pair(s) never observed, left out of the classification: "
            f"{undetected}"
        )

    # Entités présentes mais jamais observées (entièrement NaN, ou réduites
    # à une ligne) : sans fréquence détectée, elles n'ont rien à imputer et
    # le calculateur de fenêtre les nomme déjà. Elles sont retirées de la
    # cible avant validation, qui les dirait à tort absentes des données ;
    # une entrée de dict nommant une entité réellement inconnue reste,
    # elle, signalée par le validateur
    if is_panel and isinstance(effective_target, dict):
        observed = {split_variable_key(key)[0] for key in detected}
        silent = set(entities or ()) - observed
        effective_target = {
            entity: frequency
            for entity, frequency in effective_target.items()
            if entity not in silent
        }

    # Validation de la fréquence cible contre les fréquences détectées
    effective_target = TargetFrequencyValidator().validate(
        target_frequency=effective_target,
        detected_frequencies=detected,
        on_frequency_mismatch=on_frequency_mismatch,
    )

    # Classification des variables, par couple (entité, colonne), puis
    # progression des fréquences d'étape
    categories = classify_variables_at_frequency(
        detected, effective_target, is_panel=is_panel
    )
    progression = build_frequency_progression(
        detected,
        categories['impute'],
        effective_target,
        impute_intermediate_frequencies=impute_intermediate_frequencies,
    )
    return FrequencyLayout(
        is_panel=is_panel,
        detected_frequencies=detected,
        undetected=undetected,
        misaligned=misaligned,
        target_frequency=effective_target,
        categories=categories,
        progression=progression,
    )


# Expansion de la fréquence cible par entité
def expand_target_frequency(
    target_frequency: Union[str, Dict[EntityKey, str]],
    *,
    is_panel: bool,
    entities: Optional[Sequence[EntityKey]],
) -> Union[str, Dict[EntityKey, str]]:
    """Give the target frequency the shape of the data.

    Args:
        target_frequency: Target frequency, already normalized by the format
            validation.
        is_panel: Whether the data is a panel.
        entities: Normalized entity keys of the panel, None on a time series.

    Returns:
        On a panel, a binding entity -> frequency (a string target is
        broadcast to every entity, a dict is copied); on a time series, the
        string itself.

    Raises:
        ValueError: If a dict target is given on a time series, or misses
            entities of the panel
            (:func:`check_target_frequency_covers_entities`).

    Examples:
        >>> expand_target_frequency('M', is_panel=True, entities=[('FR',), ('DE',)])
        {('FR',): 'M', ('DE',): 'M'}
        >>> expand_target_frequency('M', is_panel=False, entities=None)
        'M'
        >>> expand_target_frequency({('FR',): 'M'}, is_panel=False, entities=None)
        Traceback (most recent call last):
        ...
        ValueError: target_frequency cannot be a dict for simple time series data. Use a string frequency instead.
    """
    # Chaîne sur un panel : diffusion à toutes les entités
    if is_panel and isinstance(target_frequency, str) and entities:
        return {entity: target_frequency for entity in entities}
    # Dictionnaire sur une série temporelle : refus avant toute détection,
    # une série n'ayant pas d'entités à lier
    if isinstance(target_frequency, dict) and not is_panel:
        raise ValueError(
            "target_frequency cannot be a dict for simple time series data. "
            "Use a string frequency instead."
        )
    # Dictionnaire sur un panel : un dict incomplet est une erreur nommant
    # les entités manquantes, et non un silence dont la conséquence
    # n'apparaît qu'à la classification, sous la forme d'entités jamais
    # imputées
    if isinstance(target_frequency, dict):
        check_target_frequency_covers_entities(target_frequency, entities)
        return target_frequency.copy()
    return target_frequency


# Vérification de la couverture des entités par une cible en dictionnaire
def check_target_frequency_covers_entities(
    target_frequency: Dict[EntityKey, str],
    entities: Sequence[EntityKey],
) -> None:
    """Check that a ``target_frequency`` dict names every entity.

    Args:
        target_frequency: Target frequency dict, entity keys already
            normalized into tuples.
        entities: Entity keys of the panel.

    Raises:
        ValueError: If entities of the panel are missing from the dict.
            The message names them — a silent gap would only surface much
            later, as entities the classification never imputes.

    Examples:
        >>> check_target_frequency_covers_entities({('FR',): 'M'}, [('FR',)])
        >>> check_target_frequency_covers_entities({('FR',): 'M'}, [('FR',), ('DE',)])
        Traceback (most recent call last):
        ...
        ValueError: target_frequency dict is incomplete: no frequency given for 1 entity/entities (('DE',),). A dict target_frequency must name every entity of the panel.
    """
    # Énumération des entités absentes du dictionnaire. Seul un panel
    # atteint ce contrôle : un dict sur une série temporelle est refusé à
    # l'expansion de la cible, et un panel porte toujours ses entités
    # (panel_cols converties en index avant le fit)
    missing = [
        entity for entity in entities
        if normalize_entity_key(entity) not in target_frequency
    ]
    if missing:
        raise ValueError(
            f"target_frequency dict is incomplete: no frequency given for "
            f"{len(missing)} entity/entities "
            f"{tuple(missing)}. A dict target_frequency must name every "
            f"entity of the panel."
        )


# Repérage des couples observés hors de la grille calendaire de leur fréquence
def misaligned_anchor_pairs(
    X_work: pd.DataFrame,
    frequencies: Mapping[PairKey, Optional[str]],
    *,
    is_panel: bool,
) -> Tuple[PairKey, ...]:
    """Find the pairs observed off the calendar bounds of their own frequency.

    A monthly, quarterly or yearly value describes a calendar period, and
    the position of the index says which bound dates it: the last day on
    a period-end index (``ME``: 31 December for a year), the first day on
    a period-start index (``MS``: 1 January). A pair observed elsewhere —
    a fiscal year closing in June, a yearly value dated 1 December on a
    ``MS`` index — has no consistent reading downstream: the training
    mask, built on calendar periods, drops every one of its anchors, while
    the rescaling attaches each of them to its calendar year. Such a pair
    is left out rather than imputed on a wrong reading.

    Only the monthly, quarterly and yearly frequencies are checked, and
    only for an entity whose index carries a position: a daily or
    business-day index implies no calendar bound.

    Args:
        X_work: Working frame, time series or panel.
        frequencies: Detected frequency of each pair, keyed like
            ``detected_frequencies_``, None when undetectable.
        is_panel: Whether ``X_work`` is a panel.

    Returns:
        The keys of the misaligned pairs, sorted for determinism.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> dates = pd.date_range('2021-01-31', periods=36, freq='ME')
        >>> fiscal_years = pd.DataFrame({'m1': np.arange(36.0), 'a1': np.nan}, index=dates)
        >>> fiscal_years.loc[dates[5::12], 'a1'] = [1.0, 2.0, 3.0]
        >>> misaligned_anchor_pairs(fiscal_years, {'m1': 'M', 'a1': 'Y'}, is_panel=False)
        ('a1',)
    """
    # Initialisation de la liste des couples mal placés
    misaligned: List[PairKey] = []
    # Parcours des blocs d'entité, indexés par la seule date
    for entity, _mask, block in iter_entity_blocks(X_work, is_panel=is_panel):
        prefix = normalize_entity_key(entity) if is_panel else ()

        # Position de l'index de l'entité : sans position (index journalier,
        # ouvré, ou fréquence indéterminable), aucune borne n'est impliquée
        try:
            parsed = detect_index_frequency(block.index, return_format='components')
        except (ValueError, TypeError):
            continue
        position = getattr(parsed, 'position', None)
        if position not in ('S', 'E'):
            continue

        # Contrôle des seules fréquences calendaires de chaque couple
        for column in block.columns:
            key = (*prefix, column) if prefix else column
            frequency = frequencies.get(key)
            if frequency is None:
                continue
            base = normalize_frequency(frequency, return_format='base')
            if base not in MONTH_BASED_FREQUENCIES:
                continue
            # Borne attendue de chaque observation : début ou fin de SA période
            dates = pd.DatetimeIndex(block.index[block[column].notna().to_numpy()])
            periods = dates.to_period(base)
            bounds = (
                periods.start_time if position == 'S'
                else periods.end_time.normalize()
            )
            if not bool((dates.normalize() == bounds).all()):
                misaligned.append(key)
    return tuple(sorted(misaligned, key=repr))


# Classification des couples relativement à une fréquence de prédiction
def classify_variables_at_frequency(
    detected_frequencies: Mapping[PairKey, str],
    prediction_frequency: StageFrequency,
    *,
    is_panel: bool,
) -> Dict[str, List[PairKey]]:
    """Classify the detected pairs relative to a prediction frequency.

    On a panel the classification is done **per (entity, column) pair**,
    never per column: the same column may be annual for ``FR``, quarterly
    for ``DE`` and monthly for ``IT``, hence imputable for the first two
    and already at target for the third. An entity the binding does not
    name is not classified at that frequency.

    Args:
        detected_frequencies: Frequency of each pair.
        prediction_frequency: Frequency at which predictions will be made
            (str for a time series, dict entity -> frequency for a panel:
            a panel target is always expanded into a binding, and a dict
            target on a time series is refused by its expansion).
        is_panel: Whether the keys are ``(entity..., column)`` tuples.

    Returns:
        Dict with keys ``'aggregate'``, ``'impute'`` and
        ``'target_freq'``, each holding a list of pair keys, in the order of
        ``detected_frequencies``.

    Examples:
        >>> classify_variables_at_frequency(
        ...     {('FR', 'v'): 'Y', ('DE', 'v'): 'Q', ('IT', 'v'): 'M'},
        ...     {('FR',): 'M', ('DE',): 'Q', ('IT',): 'Q'},
        ...     is_panel=True,
        ... )
        {'aggregate': [('IT', 'v')], 'impute': [('FR', 'v')], 'target_freq': [('DE', 'v')]}
    """
    # Initialisation du dictionnaire résultat
    result: Dict[str, List[PairKey]] = {
        'aggregate': [], 'impute': [], 'target_freq': []
    }

    # Distinction suivant la structure des données
    # Données de panel
    if is_panel:
        # Parcours des fréquences détectées, couple par couple
        for key, freq in detected_frequencies.items():
            # Décomposition de la clé : entité toujours en tuple
            entity, _ = split_variable_key(key)

            # Fréquence cible liée à l'entité : une entité que la liaison
            # ne nomme pas n'est pas classée à cette fréquence
            pred_freq = prediction_frequency.get(entity)
            if pred_freq is None:
                continue

            # Normalisation des deux fréquences comparées
            freq_normalized = normalize_frequency(freq)
            pred_normalized = normalize_frequency(pred_freq)

            # Comparaison des fréquences, POUR CETTE ENTITÉ
            if is_higher_frequency(freq, pred_freq):
                # Agrégation si la fréquence source est plus fine que la cible
                result['aggregate'].append(key)
            elif freq_normalized == pred_normalized:
                # Cas d'égalité : déjà à la fréquence cible
                result['target_freq'].append(key)
            else:
                # Imputation si la fréquence source est plus basse que la cible
                result['impute'].append(key)
    # Cas des séries temporelles
    else:
        # Parcours des fréquences détectées
        for col, freq in detected_frequencies.items():
            # Normalisation des deux fréquences comparées
            freq_normalized = normalize_frequency(freq)
            pred_normalized = normalize_frequency(prediction_frequency)

            # Comparaison des fréquences
            if is_higher_frequency(freq, prediction_frequency):
                result['aggregate'].append(col)
            elif freq_normalized == pred_normalized:
                result['target_freq'].append(col)
            else:
                result['impute'].append(col)

    return result


# Construction de la progression d'un groupe d'entités
def group_frequency_progression(
    detected_frequencies: Mapping[PairKey, str],
    imputable: Sequence[PairKey],
    entities: Optional[Sequence[EntityKey]],
    target_frequency: str,
) -> List[str]:
    """Build the stage frequencies of one target-frequency group.

    Args:
        detected_frequencies: Frequency of each pair.
        imputable: Pairs classified ``'impute'`` at the target frequency.
        entities: Entity keys of the group, ``None`` for a time series —
            the degenerate one-entity case, where nothing is filtered.
        target_frequency: Target frequency shared by the group.

    Returns:
        Stage frequencies in base form, from the lowest to the highest,
        the target frequency last. The lowest frequency of ``F`` is never
        a stage: nothing is imputable at it.

    Examples:
        >>> group_frequency_progression(
        ...     {'m1': 'M', 'q1': 'Q', 'a1': 'Y'}, ['q1', 'a1'], None, 'M'
        ... )
        ['Q', 'M']
        >>> group_frequency_progression({'m1': 'M', 'a1': 'Y'}, ['a1'], None, 'M')
        ['M']
    """
    # Fréquence cible du groupe, forme canonique de comparaison
    f_target = normalize_frequency(target_frequency, return_format='base')

    # Ensemble des fréquences des couples (entité, colonne) imputables du
    # périmètre du groupe, plus la fréquence cible
    admissible = None if entities is None else set(entities)
    frequencies = {f_target}
    for key in imputable:
        entity, _ = split_variable_key(key)
        if admissible is not None and entity not in admissible:
            continue
        frequencies.add(
            normalize_frequency(detected_frequencies[key], return_format='base')
        )

    # Fréquence la plus basse de l'ensemble : jamais une étape, rien n'y étant
    # à imputer. "get_frequency_order" croît quand la fréquence baisse
    lowest = max(frequencies, key=get_frequency_order)

    # Tri de la plus basse à la plus haute, borné des deux côtés :
    # strictement au-dessus de la plus basse source, au plus à la cible
    stages = [
        frequency
        for frequency in sorted(frequencies, key=get_frequency_order, reverse=True)
        if is_higher_frequency(frequency, lowest)
        and (frequency == f_target or is_higher_frequency(f_target, frequency))
    ]

    # Garantie que la cible est le dernier élément de la progression
    if f_target in stages:
        stages.remove(f_target)
    stages.append(f_target)
    return stages


# Construction de la progression de fréquences
def build_frequency_progression(
    detected_frequencies: Mapping[PairKey, str],
    imputable: Sequence[PairKey],
    target_frequency: StageFrequency,
    *,
    impute_intermediate_frequencies: Any,
) -> List[StageFrequency]:
    """Build the ordered list of stage frequencies.

    Under ``impute_intermediate_frequencies is False`` the progression is
    the target frequency alone: the imputed variable jumps straight from
    its own frequency to the target, with no intermediate stage. Under the
    two other modalities the progression is complete and **identical**:
    ``'covariates_only'`` and ``True`` share the very same stage plan and
    differ only by the origin filter of ``y_train``.

    On a panel the progression is computed **per group of entities sharing
    the same target frequency**, then merged into global stages: an entity
    whose group does not travel through a stage is simply absent from that
    stage's binding, which leaves it unclassified — hence never imputed,
    and never written — at that stage.

    Args:
        detected_frequencies: Frequency of each pair.
        imputable: Pairs classified ``'impute'`` at the target frequency.
        target_frequency: Effective target frequency, a string on a time
            series, a binding per entity on a panel.
        impute_intermediate_frequencies: Axis-2 modality; only ``False``
            (tested by identity, ``'covariates_only'`` being truthy) yields
            a single stage.

    Returns:
        Ordered list of stage frequencies, the target frequency last. A
        time series yields a list of strings; a panel yields a list of
        ``{entity: frequency}`` bindings.

    Examples:
        >>> detected = {('FR', 'v'): 'Y', ('FR', 'q1'): 'Q', ('DE', 'v'): 'Y'}
        >>> build_frequency_progression(
        ...     detected, list(detected), {('FR',): 'M', ('DE',): 'Q'},
        ...     impute_intermediate_frequencies=True,
        ... )
        [{('FR',): 'Q', ('DE',): 'Q'}, {('FR',): 'M'}]
        >>> build_frequency_progression(
        ...     detected, list(detected), {('FR',): 'M', ('DE',): 'Q'},
        ...     impute_intermediate_frequencies=False,
        ... )
        [{('FR',): 'M', ('DE',): 'Q'}]
    """
    # Modalité sans étape intermédiaire : une seule étape, la cible.
    if impute_intermediate_frequencies is False:
        return [target_frequency]

    # Cas dégénéré de la série temporelle : un seul groupe, cible scalaire
    if not isinstance(target_frequency, dict):
        return list(group_frequency_progression(
            detected_frequencies, imputable, None, target_frequency
        ))

    # Panel : un groupe d'entités par fréquence cible partagée
    groups: Dict[str, List[EntityKey]] = {}
    for entity, frequency in target_frequency.items():
        groups.setdefault(
            normalize_frequency(frequency, return_format='base'), []
        ).append(entity)
    per_group = {
        f_target: group_frequency_progression(
            detected_frequencies, imputable, entities, f_target
        )
        for f_target, entities in groups.items()
    }

    # Fusion en étapes globales : union ordonnée des fréquences d'étape,
    # chaque étape ne liant que les entités dont le groupe la traverse
    stage_order = sorted(
        {stage for stages in per_group.values() for stage in stages},
        key=get_frequency_order,
        reverse=True,
    )
    return [
        {
            entity: stage
            for f_target, entities in groups.items()
            if stage in per_group[f_target]
            for entity in entities
        }
        for stage in stage_order
    ]


# Construction d'un label de fréquence lisible pour une étape
def stage_frequency_label(pred_freq: StageFrequency) -> str:
    """Build a human-readable frequency label for a stage.

    Labeling rule for panel stages: if every entity of the stage shares
    the same frequency, that shared frequency is the label; otherwise a
    composite label lists each entity's frequency, sorted by entity key
    for determinism.

    Args:
        pred_freq: Prediction frequency of the stage (str for a time
            series, dict entity -> frequency for a panel).

    Returns:
        Frequency label string.

    Examples:
        >>> stage_frequency_label('M')
        'M'
        >>> stage_frequency_label({('FR',): 'Q', ('DE',): 'Q'})
        'Q'
        >>> stage_frequency_label({('FR',): 'Y', ('DE',): 'Q'})
        "('DE',)=Q+('FR',)=Y"
    """
    # Cas d'une fréquence unique (séries temporelles)
    if not isinstance(pred_freq, dict):
        return str(pred_freq)

    # Cas d'un panel : fréquence partagée par toutes les entités de l'étape
    unique_freqs = {
        normalize_frequency(freq, return_format='base')
        for freq in pred_freq.values()
    }
    if len(unique_freqs) == 1:
        return unique_freqs.pop()

    # Fréquences hétérogènes : label composite, trié pour être déterministe
    parts = sorted(f"{entity}={freq}" for entity, freq in pred_freq.items())
    return '+'.join(parts)


# Fréquence d'étape ligne à ligne
def stage_frequency_of(stage_freq: StageFrequency, index: pd.Index) -> pd.Series:
    """Give the stage frequency of each row of a grid.

    Args:
        stage_freq: Frequency of the stage, scalar or per entity.
        index: Grid the frequencies are read on.

    Returns:
        Series indexed on ``index`` holding the base frequency code of
        each row (``str`` dtype).

    Examples:
        >>> import pandas as pd
        >>> idx = pd.MultiIndex.from_product(
        ...     [['FR', 'DE'], pd.date_range('2021-03-31', periods=2, freq='QE')]
        ... )
        >>> stage_frequency_of({('FR',): 'QE', ('DE',): 'monthly'}, idx).tolist()
        ['Q', 'Q', 'M', 'M']
    """
    # Fréquence unique : la même pour toutes les lignes
    if not isinstance(stage_freq, dict):
        return pd.Series(
            normalize_frequency(stage_freq, return_format='base'), index=index, dtype='str'
        )

    # Fréquence par entité : normalisée une fois par entité, puis diffusée
    # sur ses lignes par le code d'entité
    entity_codes, entities = get_entity_codes(index)
    per_entity = pd.array(
        [normalize_frequency(stage_freq[entity], return_format='base') for entity in entities],
        dtype='str',
    )
    return pd.Series(per_entity.take(entity_codes), index=index)
