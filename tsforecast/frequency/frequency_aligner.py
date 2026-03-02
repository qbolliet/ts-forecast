"""Frequency alignment utilities for mixed frequency imputation.

This module provides the FrequencyAligner class to aggregate and interpolate
data between different frequencies, handling both time series and panel data.
"""
# Modules de base
from typing import Dict, List, Tuple, Union

# Manipulation de données
import numpy as np
import pandas as pd

# Utilitaires du package
from ..utils.frequency.converter import FrequencyConverter
from ..utils.frequency.utils import normalize_frequency
from ..panel.utils import normalize_entity_key


# Classe d'alignement des fréquences de jeu de données avec des fréquences cibles
class FrequencyAligner:
    """Align data to target frequencies via aggregation or interpolation.

    Handles both time series and panel data, delegating actual frequency
    conversion to ``FrequencyConverter``. Panel data is processed per entity
    to respect entity-specific target frequencies.

    Examples:
        >>> import pandas as pd
        >>> aligner = FrequencyAligner()
        >>> dates = pd.date_range('2023-01-01', periods=90, freq='D')
        >>> df = pd.DataFrame({'daily_var': range(90)}, index=dates)
        >>> agg = aligner.aggregate_to_target(
        ...     df, ['daily_var'], 'M', is_panel=False
        ... )
    """

    # Initialisation
    def __init__(self):
        """Initialize the FrequencyAligner."""
        # Initialisation du convertisseur de fréquences
        self._freq_converter = FrequencyConverter()

    # Méthode de regroupement des variables par entité
    def group_keys_by_entity_and_variable(
        self,
        keys: List[Union[str, Tuple]],
    ) -> Dict[tuple, List[str]]:
        """Group variable keys by entity, extracting column names.

        Args:
            keys: List of (entity..., variable) tuples or plain column names.

        Returns:
            Dict mapping entity tuples to lists of column names.
            For time series (non-panel), returns {(): [col_names]}.
        """
        # Initialisation du dictionnaire résultat
        grouped: Dict[tuple, List[str]] = {}

        # Parcours des clés
        for key in keys:
            # Distinction suivant le type de la clé
            if isinstance(key, tuple):
                # Extraction de l'entité et de la colonne de la clé
                entity = normalize_entity_key(key[:-1])
                col = key[-1]
            else:
                # Cas où il n'y a pas d'entité
                entity = ()
                col = key
            # Initialisation de l'entité si elle n'existe pas déjà
            if entity not in grouped:
                grouped[entity] = []
            # Ajout de la colonne à l'entité
            if col not in grouped[entity]:
                grouped[entity].append(col)

        return grouped

    # Méthode d'extraction de la fréquence cible d'une entité
    def get_entity_target_frequency(
        self,
        entity: tuple,
        target_frequency: Union[str, Dict],
    ) -> str:
        """Get the target frequency for a specific entity.

        Args:
            entity: Entity tuple (e.g., ('FR',) or ('FR', 'GDP')).
            target_frequency: Either a single frequency string or a dict
                mapping entities to frequencies.

        Returns:
            Normalized target frequency string for the entity.

        Raises:
            ValueError: If no target frequency found for the entity.
        """
        # Cas où la fréquence cible est un dictionnaire
        if isinstance(target_frequency, dict):
            # Extraction de la fréquence cible
            freq = target_frequency.get(entity)
            # Recherche en extrayant l'entité du tuple
            if freq is None and len(entity) == 1:
                freq = target_frequency.get(entity[0])
            # Cas où la fréquence n'est pas trouvée
            if freq is None:
                raise ValueError(
                    f"No target frequency found for entity {entity}"
                )
            return freq
        # Dans le cas où la fréquence cible est une chaîne de caractère, elle s'applique à toutes les entités
        return target_frequency

    # Méthode de construction du masque des observations associées à une entité
    def get_entity_mask(
        self,
        X: pd.DataFrame,
        entity: tuple,
    ) -> np.ndarray:
        """Get a boolean mask for rows belonging to a specific entity.

        Args:
            X: DataFrame with MultiIndex (entity levels + time level).
            entity: Entity tuple to filter on.

        Returns:
            Boolean numpy array of shape (len(X),).
        """
        # Extraction de l'index
        index = X.index
        # Extraction des niveaux associés aux entités
        n_entity_levels = index.nlevels - 1

        # Cas où les entités sont sur un seul niveau
        if n_entity_levels == 1:
            return (index.get_level_values(0) == entity[0])
        else:
            # Construction d'un masque pour les entités multi-niveaux
            # Initialisation du masque
            mask = np.ones(len(X), dtype=bool)
            # Parcours des niveaux associés aux entités
            for i in range(n_entity_levels):
                # Mise à jour du masque
                mask &= (index.get_level_values(i) == entity[i])
            return mask

    # Méthode d'aggrégation des données à une fréquence cible
    def aggregate_to_target(
        self,
        df: pd.DataFrame,
        aggregate_keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        is_panel: bool,
    ) -> pd.DataFrame:
        """Aggregate high-frequency columns to target frequency.

        For panel data, aggregation is performed per entity to respect
        entity-specific target frequencies.

        Args:
            df: Input DataFrame.
            aggregate_keys: Variable keys to aggregate (column names or
                (entity..., variable) tuples).
            target_frequency: Target frequency (str or per-entity dict).
            is_panel: Whether the data is panel data.

        Returns:
            DataFrame with aggregated columns.
        """
        # Cas où les clés d'aggrégation ne sont pas spécifiées
        if not aggregate_keys:
            return df

        # Initialisation du jeu de données résultat
        result = df.copy()

        # Cas séries temporelles : agrégation globale
        if not is_panel:
            # Extraction des colonnes des tuples
            columns = self.extract_column_names(aggregate_keys)
            # Parcours des colonnes
            for col in columns:
                # Vérification que la colonne est dans le jeu de données
                if col not in df.columns:
                    continue
                # Aggrégation à la fréquence cible
                aggregated = self._freq_converter.aggregate_to_lower_frequency(
                    df[col], target_frequency, method='sum', full_periods_only=True
                )
                # Reproduction de l'index original
                result[col] = aggregated.reindex(df.index)
            # Suppression des lignes de Nan
            result.dropna(axis=0, how='all', inplace=True)
            return result

        # Cas panel : agrégation par entité
        # Extraction du dictionnaire associant une liste de variables à chaque entité
        grouped = self.group_keys_by_entity_and_variable(aggregate_keys)

        # Parcours des entités
        for entity, cols in grouped.items():
            # Extraction de la fréquence cible associée à l'enité
            entity_target = self.get_entity_target_frequency(entity, target_frequency)
            # Création du masque des observations de l'entité
            entity_mask = self.get_entity_mask(df, entity)
            
            # Parcours des colonnes
            for col in cols:
                # Cas où la colonne n'est pas présente dans le jeu de données
                if col not in df.columns:
                    continue
                
                # Extraction des observations de l'entité pour la colonne d'intérêt
                entity_series = df.loc[entity_mask, col]
                # Suppression des niveaux afférents à l'entité
                entity_series = entity_series.droplevel(
                    list(range(df.index.nlevels - 1))
                )
                # Agrégation à la fréquence souhaitée
                aggregated = self._freq_converter.aggregate_to_lower_frequency(
                    entity_series, entity_target, method='sum', full_periods_only=True
                )
                # Réindexation à l'index temporel original
                reindexed = aggregated.reindex(entity_series.index)
                # Ajout au DataFrame résultat
                result.loc[entity_mask, col] = reindexed.values

        # Suppression des lignes de Nan
        result.dropna(axis=0, how='all', inplace=True)
        return result

    # Méthode d'interpolation d'un jeu de données à une fréquence cible
    def interpolate_to_target(
        self,
        df: pd.DataFrame,
        interpolate_keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        is_panel: bool,
    ) -> pd.DataFrame:
        """Interpolate low-frequency columns to target frequency.

        For panel data, interpolation is performed per entity to respect
        entity-specific target frequencies.

        Args:
            df: Input DataFrame.
            interpolate_keys: Variable keys to interpolate (column names or
                (entity..., variable) tuples).
            target_frequency: Target frequency (str or per-entity dict).
            is_panel: Whether the data is panel data.

        Returns:
            DataFrame with interpolated columns.
        """
        # Cas où aucune clé d'interpolation n'est spécifiée, retourne le jeu de données inchangé
        if not interpolate_keys:
            return df

        # Initialisation du jeu de données résultat
        result = df.copy()

        # Cas séries temporelles : interpolation globale
        if not is_panel:
            # Extraction des colonnes
            columns = self.extract_column_names(interpolate_keys)
            # Parcours des colonnes
            for col in columns:
                # Vérification que la colonne est bien dans le jeu de données
                if col not in df.columns:
                    continue
                # Interpolation à la fréquence cible
                interpolated = self._freq_converter.interpolate_to_higher_frequency(
                    df[col], target_frequency, method='linear'
                )
                # Réindexation à l'index initial
                result[col] = interpolated.reindex(df.index)

            return result

        # Cas panel : interpolation par entité
        # Regroupement des colonnes par entité
        grouped = self.group_keys_by_entity_and_variable(interpolate_keys)
        
        # Parcours des entités
        for entity, cols in grouped.items():
            # Extraction de la fréquence cible associée à l'entité
            entity_target = self.get_entity_target_frequency(entity, target_frequency)
            # Extraction du masque des observations associées à l'entité
            entity_mask = self.get_entity_mask(df, entity)

            # Parcours des colonnes
            for col in cols:
                # Vérification que la colonne est dans le jeu de données
                if col not in df.columns:
                    continue
                
                # Extraction des observations de la colonne associées à l'entité
                entity_series = df.loc[entity_mask, col]
                # Suppression des niveaux associés à l'entité
                entity_series = entity_series.droplevel(
                    list(range(df.index.nlevels - 1))
                )
                # Interpolation à la fréquence cible
                interpolated = self._freq_converter.interpolate_to_higher_frequency(
                    entity_series, entity_target, method='linear'
                )
                # Réindexation à la série temporelle originale
                reindexed = interpolated.reindex(entity_series.index)
                # Ajout au résultat
                result.loc[entity_mask, col] = reindexed.values

        return result

    # Méthode d'extraction des noms de colonnes
    def extract_column_names(
        self,
        keys: List[Union[str, Tuple]],
    ) -> List[str]:
        """Extract unique column names from variable keys.

        Args:
            keys: List of variable identifiers (column names or
                (entity, column) tuples).

        Returns:
            List of unique column names.
        """
        # Initialisation des noms de colonnes
        column_names = set()
        # Parcours des clés
        for key in keys:
            # Cas où la clé est un tuple
            if isinstance(key, tuple):
                # Extraction du dernier élément du tuple
                column_names.add(key[-1])
            else:
                column_names.add(key)
        return list(column_names)
