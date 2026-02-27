"""Target frequency validation for mixed frequency imputation.

This module provides the TargetFrequencyValidator class to validate that
target frequencies are compatible with detected data frequencies,
for both time series and panel data.
"""
# Modules de base
import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

# Utilitaires du package
from ..utils.frequency.utils import (
    normalize_frequency,
    is_higher_frequency,
    get_frequency_order,
)
from ..panel.utils import normalize_entity_key


# Classe de validation de la fréquence cible au regard des fréquences détectées
class TargetFrequencyValidator:
    """Validate target frequencies against detected data frequencies.

    Ensures that target frequencies are not higher than the highest
    (most granular) frequency present in the data, for both time series
    and panel data. When a mismatch is found, either raises an error or
    adjusts the target frequency depending on ``on_frequency_mismatch``.

    The data structure (time series vs panel) and the list of entities are
    inferred automatically from the keys of ``detected_frequencies``:
    tuple keys indicate panel data, string keys indicate time series.

    Examples:
        >>> validator = TargetFrequencyValidator()
        >>> detected = {'col_a': 'M', 'col_b': 'Q'}
        >>> result = validator.validate(
        ...     target_frequency='M',
        ...     detected_frequencies=detected,
        ... )
        >>> result
        'M'
    """
    # Méthode de validation de la fréquence cible
    def validate(
        self,
        target_frequency: Union[str, Dict],
        detected_frequencies: Dict,
        on_frequency_mismatch: Literal['error', 'warn'] = 'error',
    ) -> Union[str, Dict]:
        """Validate target frequency against detected frequencies.

        Infers whether the data is panel (tuple keys) or time series (string
        keys) from ``detected_frequencies``, then delegates to
        ``_validate_timeseries`` or ``_validate_panel`` accordingly.

        Args:
            target_frequency: Target frequency (str for TS, dict for panel).
            detected_frequencies: Dict mapping variable keys to detected
                frequency strings. For time series, keys are column name
                strings. For panel, keys are ``(entity..., var)`` tuples as
                produced by ``detect_dataset_frequency``.
            on_frequency_mismatch: How to handle mismatches:
                - 'error': Raise ValueError (default)
                - 'warn': Warn and adjust to highest available

        Returns:
            Validated (possibly adjusted) target frequency.

        Raises:
            ValueError: If detected_frequencies is empty, if a mismatch is
                found and on_frequency_mismatch='error', or if the keys have
                an inconsistent format (mixed tuple/string keys).
        """
        # Inférence de la structure (panel ou séries temporelles) et des entités
        is_panel, entities = self._infer_structure(detected_frequencies)

        # Distinction suivant la structure
        if not is_panel:
            # Cas de séries temporelles
            return self._validate_timeseries(
                target_frequency, detected_frequencies, on_frequency_mismatch
            )
        else:
            # Cas de données de panel
            return self._validate_panel(
                target_frequency, detected_frequencies, entities, on_frequency_mismatch
            )

    # Méthode auxiliaire d'inférence de la structure (panel ou séries temporelles) et des entités
    def _infer_structure(
        self,
        detected_frequencies: Dict,
    ) -> Tuple[bool, Optional[List]]:
        """Infer data structure and entity list from detected_frequencies keys.

        Panel data produces tuple keys of the form ``(entity..., var)`` as
        returned by ``detect_dataset_frequency``. Time series data produces
        plain string keys (column names).

        Args:
            detected_frequencies: Dict mapping variable keys to frequencies.

        Returns:
            Tuple ``(is_panel, entities)`` where ``entities`` is ``None`` for
            time series data and a list of entity tuples for panel data.

        Raises:
            ValueError: If detected_frequencies is empty (structure cannot be
                determined) or if keys have an inconsistent format.
        """
        # Cas d'un dictionnaire vide : structure indéterminable
        if not detected_frequencies:
            raise ValueError(
                "detected_frequencies is empty: cannot infer data structure "
                "(time series vs panel)."
            )

        # Classification des clés
        tuple_keys = [k for k in detected_frequencies if isinstance(k, tuple)]
        string_keys = [k for k in detected_frequencies if isinstance(k, str)]

        # Vérification de la cohérence du format des clés
        if tuple_keys and string_keys:
            raise ValueError(
                "detected_frequencies contains mixed key types (both str and tuple). "
                "Expected either all string keys (time series) or all tuple keys (panel)."
            )

        # Cas des séries temporelles (clés string)
        if not tuple_keys:
            return False, None

        # Cas panel : extraction des entités uniques (tous les niveaux sauf le dernier = variable)
        entities = list({k[:-1] for k in tuple_keys})
        return True, entities

    # Méthode auxiliaire de validation de la fréquence cible dans le cas de données de séries temporelles
    def _validate_timeseries(
        self,
        target_frequency: str,
        detected_frequencies: Dict[str, str],
        on_frequency_mismatch: Literal['error', 'warn'],
    ) -> str:
        """Validate target frequency for time series data.

        Args:
            target_frequency: Target frequency string.
            detected_frequencies: Dict mapping columns to frequencies.
            on_frequency_mismatch: Mismatch handling strategy.

        Returns:
            Validated target frequency string.

        Raises:
            ValueError: If target is a dict or higher than data frequency
                and on_frequency_mismatch='error'.
        """
        # Vérification que target_frequency est un string
        if isinstance(target_frequency, dict):
            raise ValueError(
                "target_frequency cannot be a dict for simple time series data. "
                "Use a string frequency instead."
            )

        # Obtention de la fréquence la plus élevée
        highest_freq = self._get_highest_frequency_timeseries(detected_frequencies)

        # Vérification si la fréquence cible est plus haute que la plus haute fréquence
        if is_higher_frequency(target_frequency, highest_freq):
            error_msg = (
                f"Target frequency '{target_frequency}' is higher than "
                f"the highest frequency '{highest_freq}' in the data. "
                f"Target frequency must be equal to or lower than the highest frequency."
            )

            if on_frequency_mismatch == 'error':
                raise ValueError(error_msg)
            else:  # 'warn'
                warnings.warn(
                    f"{error_msg} Adjusting target_frequency to '{highest_freq}'.",
                    UserWarning
                )
                return highest_freq

        return target_frequency

    # Méthode auxiliaire de validation de la fréquence cible dans le cas de données de panel
    def _validate_panel(
        self,
        target_frequency: Union[str, Dict],
        detected_frequencies: Dict[Tuple, str],
        entities: List,
        on_frequency_mismatch: Literal['error', 'warn'],
    ) -> Dict:
        """Validate target frequency for panel data.

        Args:
            target_frequency: Target frequency dict mapping entities to
                frequencies, or a single string applied to all entities.
            detected_frequencies: Dict mapping (entity..., var) to frequency.
            entities: List of unique entity tuples inferred from
                detected_frequencies keys.
            on_frequency_mismatch: Mismatch handling strategy.

        Returns:
            Validated (possibly adjusted) target frequency dict.

        Raises:
            ValueError: If entities are missing from target_frequency dict
                or frequency mismatch with on_frequency_mismatch='error'.
        """
        # Vérification que toutes les entités ont une fréquence cible
        if isinstance(target_frequency, dict):
            missing_entities = set(entities) - set(target_frequency.keys())
            if missing_entities:
                raise ValueError(
                    f"target_frequency dict is missing entries for entities: "
                    f"{missing_entities}"
                )

            # Vérification des entités supplémentaires
            extra_entities = set(target_frequency.keys()) - set(entities)
            if extra_entities:
                warnings.warn(
                    f"target_frequency dict contains entries for entities not in data: "
                    f"{extra_entities}. These will be ignored.",
                    UserWarning
                )

        # Validation de chaque fréquence cible par entité
        # Initialisation de la liste des entités invalides
        invalid_entities = []
        # Initialisation du dictionnaire des fréquences ajustées
        adjusted_freqs = {}

        # Parcours des entités
        for entity in entities:
            # Extraction de la fréquence cible
            if isinstance(target_frequency, dict):
                target_freq = target_frequency[entity]
            else:
                target_freq = target_frequency

            # Vérification que la fréquence cible n'est pas plus élevée que la fréquence la plus élevée pour l'entité
            try:
                # Extraction de la fréquence la plus élevée de l'entité
                highest_freq = self._get_highest_frequency_entity(
                    entity, detected_frequencies
                )

                # Vérification que la fréquence cible n'est pas plus élevée que la fréquence la plus élevée pour l'entité
                if is_higher_frequency(target_freq, highest_freq):
                    # Ajout aux fréquences invalides
                    invalid_entities.append((entity, target_freq, highest_freq))
                    # Ajustement de la fréquence cible à la fréquence la plus élevée de l'entité
                    adjusted_freqs[entity] = highest_freq
                else:
                    # Ajustement à l'identique
                    adjusted_freqs[entity] = target_freq

            except ValueError as e:
                warnings.warn(f"Entity '{entity}': {e}", UserWarning)
                continue

        # Traitement des entités invalides
        if invalid_entities:
            # Construction du message d'erreur
            error_msg = (
                f"Target frequencies are higher than highest frequencies for "
                f"{len(invalid_entities)} entities:\n"
            )
            for entity, target, highest in invalid_entities[:5]:
                error_msg += (
                    f"  - Entity '{entity}': target '{target}' > highest '{highest}'\n"
                )
            if len(invalid_entities) > 5:
                error_msg += f"  ... and {len(invalid_entities) - 5} more entities\n"

            if on_frequency_mismatch == 'error':
                raise ValueError(error_msg.rstrip())
            else:  # 'warn'
                warnings.warn(
                    f"{error_msg.rstrip()}\n"
                    f"Adjusting target frequencies to entity-specific highest frequencies.",
                    UserWarning
                )
                return adjusted_freqs

        return adjusted_freqs

    # Méthode auxiliaire d'extraction de la fréquence la plus élevée d'un jeu de données de séries temporelles
    def _get_highest_frequency_timeseries(
        self,
        detected_frequencies: Dict[str, str],
    ) -> str:
        """Get the highest (most granular) frequency for time series data.

        Args:
            detected_frequencies: Dict mapping columns to frequencies.

        Returns:
            Highest frequency string.

        Raises:
            ValueError: If no valid frequencies found.
        """
        # Extraction des fréquences valides
        valid_freqs = [freq for freq in detected_frequencies.values() if freq is not None]
        if not valid_freqs:
            raise ValueError("No valid frequencies detected in the dataset")

        # Détermination de la fréquence avec l'ordre le plus bas (plus granulaire)
        freq_orders = {}
        for freq in set(valid_freqs):
            try:
                freq_orders[freq] = get_frequency_order(freq)
            except ValueError:
                continue

        # Cas au aucun ordre ne peut être extrait
        if not freq_orders:
            raise ValueError("Could not determine frequency order for detected frequencies")

        # Retour de la fréquence avec l'ordre le plus bas
        return min(freq_orders.keys(), key=lambda x: freq_orders[x])

    # Méthode auxiliaire d'extraction de la fréquence la plus élevée d'une entité
    def _get_highest_frequency_entity(
        self,
        entity: tuple,
        detected_frequencies: Dict[Tuple, str],
    ) -> str:
        """Get the highest frequency for a specific entity in panel data.

        Args:
            entity: Entity identifier tuple (e.g. ``('FR',)`` or
                ``('FR', 'EU')``), matching the ``key[:-1]`` format of
                detected_frequencies keys.
            detected_frequencies: Dict mapping (entity..., var) to frequency.

        Returns:
            Highest frequency for the entity.

        Raises:
            ValueError: If no valid frequencies found for the entity.
        """
        # Normalisation de l'entité
        normalized_entity = normalize_entity_key(entity)
        # Extraction des fréquences pour cette entité
        entity_freqs = {}
        # Parcours des fréquences détectées
        for key, freq in detected_frequencies.items():
            var = key[-1]
            ent = normalize_entity_key(key[:-1])
            if ent == normalized_entity and freq is not None:
                entity_freqs[var] = freq

        # Cas où aucune fréquence est détectée pour l'entité
        if not entity_freqs:
            raise ValueError(f"No valid frequencies detected for entity '{entity}'")

        # Extraction de la fréquence la plus élevée parmi les fréquences de l'entité
        return self._get_highest_frequency_timeseries(entity_freqs)
