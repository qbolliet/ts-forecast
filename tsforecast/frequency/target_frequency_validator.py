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


class TargetFrequencyValidator:
    """Validate target frequencies against detected data frequencies.

    Ensures that target frequencies are not higher than the highest
    (most granular) frequency present in the data, for both time series
    and panel data. When a mismatch is found, either raises an error or
    adjusts the target frequency depending on ``on_frequency_mismatch``.

    Examples:
        >>> validator = TargetFrequencyValidator()
        >>> detected = {'col_a': 'M', 'col_b': 'Q'}
        >>> result = validator.validate(
        ...     target_frequency='M',
        ...     detected_frequencies=detected,
        ...     is_panel=False,
        ...     entities=None,
        ... )
        >>> result
        'M'
    """

    def validate(
        self,
        target_frequency: Union[str, Dict],
        detected_frequencies: Dict,
        is_panel: bool,
        entities: Optional[List],
        on_frequency_mismatch: Literal['error', 'warn'] = 'error',
    ) -> Union[str, Dict]:
        """Validate target frequency against detected frequencies.

        Delegates to ``_validate_timeseries`` or ``_validate_panel``
        depending on whether the data is panel data.

        Args:
            target_frequency: Target frequency (str for TS, dict for panel).
            detected_frequencies: Dict mapping variable keys to detected
                frequency strings (column names for TS, (entity..., var)
                tuples for panel).
            is_panel: Whether the data is panel data.
            entities: List of unique entity tuples (panel only).
            on_frequency_mismatch: How to handle mismatches:
                - 'error': Raise ValueError (default)
                - 'warn': Warn and adjust to highest available

        Returns:
            Validated (possibly adjusted) target frequency.

        Raises:
            ValueError: If mismatch found and on_frequency_mismatch='error'.
        """
        if not is_panel:
            return self._validate_timeseries(
                target_frequency, detected_frequencies, on_frequency_mismatch
            )
        else:
            return self._validate_panel(
                target_frequency, detected_frequencies, entities, on_frequency_mismatch
            )

    def _validate_timeseries(
        self,
        target_frequency: Union[str, Dict],
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

    def _validate_panel(
        self,
        target_frequency: Union[str, Dict],
        detected_frequencies: Dict[Tuple, str],
        entities: Optional[List],
        on_frequency_mismatch: Literal['error', 'warn'],
    ) -> Union[str, Dict]:
        """Validate target frequency for panel data.

        Args:
            target_frequency: Target frequency dict mapping entities to
                frequencies, or a single string.
            detected_frequencies: Dict mapping (entity..., var) to frequency.
            entities: List of unique entity tuples.
            on_frequency_mismatch: Mismatch handling strategy.

        Returns:
            Validated (possibly adjusted) target frequency dict.

        Raises:
            ValueError: If entities are missing or frequency mismatch with
                on_frequency_mismatch='error'.
        """
        # Vérification que toutes les entités ont une fréquence cible
        if entities is not None and isinstance(target_frequency, dict):
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
        invalid_entities = []
        adjusted_freqs = {}

        for entity in (entities or []):
            # Extraction de la fréquence cible
            if isinstance(target_frequency, dict):
                target_freq = target_frequency[entity]
            else:
                target_freq = target_frequency

            try:
                highest_freq = self._get_highest_frequency_entity(
                    entity, detected_frequencies
                )

                if is_higher_frequency(target_freq, highest_freq):
                    invalid_entities.append((entity, target_freq, highest_freq))
                    adjusted_freqs[entity] = highest_freq
                else:
                    adjusted_freqs[entity] = target_freq

            except ValueError as e:
                warnings.warn(f"Entity '{entity}': {e}", UserWarning)
                continue

        # Traitement des entités invalides
        if invalid_entities:
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

        if not freq_orders:
            raise ValueError("Could not determine frequency order for detected frequencies")

        # Retour de la fréquence avec l'ordre le plus bas
        return min(freq_orders.keys(), key=lambda x: freq_orders[x])

    def _get_highest_frequency_entity(
        self,
        entity: Union[str, tuple],
        detected_frequencies: Dict[Tuple, str],
    ) -> str:
        """Get the highest frequency for a specific entity in panel data.

        Args:
            entity: Entity identifier.
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
        for key, freq in detected_frequencies.items():
            var = key[-1]
            ent = normalize_entity_key(key[:-1])
            if ent == normalized_entity and freq is not None:
                entity_freqs[var] = freq

        if not entity_freqs:
            raise ValueError(f"No valid frequencies detected for entity '{entity}'")

        return self._get_highest_frequency_timeseries(entity_freqs)
