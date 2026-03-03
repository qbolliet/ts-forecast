"""Frequency alignment utilities for mixed frequency imputation.

This module provides the FrequencyAligner class to aggregate and interpolate
data between different frequencies, handling both time series and panel data.
"""
# Modules de base
from typing import Dict, List, Literal, Optional, Tuple, Union

# Manipulation de données
import numpy as np
import pandas as pd

# Utilitaires du package
from ..utils.frequency.converter import FrequencyConverter
from ..utils.frequency.utils import normalize_frequency, is_higher_frequency
from ..panel.utils import normalize_entity_key

# Détection de la fréquence de l'index
from .detector import detect_index_frequency


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
        method: str = 'linear',
        limit: Optional[int] = 'default',
        limit_direction: Optional[Literal['forward', 'backward', 'both']] = None,
        limit_area: Optional[Literal['inside', 'outside']] = None,
    ) -> pd.DataFrame:
        """Interpolate low-frequency columns to target frequency.

        For panel data, interpolation is performed per entity to respect
        entity-specific target frequencies.

        If ``limit`` is not provided, it is computed automatically as the
        number of index-frequency periods within one target-frequency period
        (e.g. 3 for quarterly-to-monthly interpolation).

        Args:
            df: Input DataFrame.
            interpolate_keys: Variable keys to interpolate (column names or
                (entity..., variable) tuples).
            target_frequency: Target frequency (str or per-entity dict).
            is_panel: Whether the data is panel data.
            method: Interpolation method passed to
                :meth:`FrequencyConverter.interpolate_to_higher_frequency`
                (e.g. ``'linear'``, ``'time'``). Defaults to ``'linear'``.
            limit: Maximum number of consecutive NaN values to fill. If
                ``'default'``, computed automatically as the frequency conversion
                factor between the source (index) frequency and the target
                frequency (e.g. 3 for quarterly→monthly). Pass an explicit
                integer to override this behaviour.
            limit_direction: Direction in which to fill NaN values. If None,
                defaults to ``'forward'`` when target_position is start
                (``'S'``/``'start'``) and ``'backward'`` when target_position is
                end (``'E'``/``'end'``). See
                :meth:`pandas.DataFrame.interpolate` for details.
            limit_area: Restriction area for filling NaN values
                (``'inside'`` or ``'outside'``). Passed directly to
                :meth:`FrequencyConverter.interpolate_to_higher_frequency`.

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
                    df[col],
                    target_frequency,
                    method=method,
                    limit=limit,
                    limit_direction=limit_direction,
                    limit_area=limit_area,
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
                    entity_series,
                    entity_target,
                    method=method,
                    limit=limit,
                    limit_direction=limit_direction,
                    limit_area=limit_area,
                )
                # Réindexation à la série temporelle originale
                reindexed = interpolated.reindex(entity_series.index)
                # Ajout au résultat
                result.loc[entity_mask, col] = reindexed.values

        return result

    # Méthode générique de conversion vers la fréquence cible (agrégation ou interpolation)
    def convert_to_target(
        self,
        df: pd.DataFrame,
        keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        is_panel: bool,
        interp_method: str = 'linear',
        interp_limit: Optional[int] = None,
        interp_limit_direction: Optional[Literal['forward', 'backward', 'both']] = None,
        interp_limit_area: Optional[Literal['inside', 'outside']] = None,
    ) -> pd.DataFrame:
        """Convert columns to a target frequency, aggregating or interpolating as needed.

        For each variable key (or per entity for panel data), the source
        frequency is detected from the index. If the target frequency is
        **higher** (more granular) than the source, the column is
        interpolated via :meth:`interpolate_to_target`. Otherwise it is
        aggregated via :meth:`aggregate_to_target`.

        Args:
            df: Input DataFrame.
            keys: Variable keys to convert (column names or
                (entity..., variable) tuples).
            target_frequency: Target frequency (str or per-entity dict).
            is_panel: Whether the data is panel data.
            interp_method: Interpolation method used when upsampling.
                Defaults to ``'linear'``.
            limit: Maximum number of consecutive NaN values to fill during
                interpolation. If ``None``, computed automatically from the
                frequency conversion factor. See
                :meth:`interpolate_to_target` for details.
            limit_direction: Direction in which to fill NaN values. If None,
                defaults to ``'forward'`` when target_position is start
                (``'S'``/``'start'``) and ``'backward'`` when target_position is
                end (``'E'``/``'end'``). See
                :meth:`pandas.DataFrame.interpolate` for details.
            limit_area: Restriction area for NaN filling during
                interpolation (``'inside'`` or ``'outside'``). Forwarded to
                :meth:`interpolate_to_target`.

        Returns:
            DataFrame with converted columns.
        """
        # Cas où aucune clé n'est spécifiée
        if not keys:
            return df

        # Initialisation des listes de clés selon la direction de conversion
        aggregate_keys: List[Union[str, Tuple]] = []
        interpolate_keys: List[Union[str, Tuple]] = []

        # Cas séries temporelles : une seule fréquence source pour tout le jeu de données
        if not is_panel:
            source_freq = detect_index_frequency(df.index)
            target_freq_str = (
                target_frequency if isinstance(target_frequency, str)
                else next(iter(target_frequency.values()))
            )
            # Orientation de la conversion selon la relation entre les fréquences
            if source_freq and is_higher_frequency(
                normalize_frequency(target_freq_str),
                normalize_frequency(source_freq),
            ):
                interpolate_keys = list(keys)
            else:
                aggregate_keys = list(keys)

        # Cas panel : orientation déterminée par entité
        else:
            # Regroupement des clés par entité
            grouped = self.group_keys_by_entity_and_variable(keys)

            # Parcours des entités
            for entity, cols in grouped.items():
                # Reconstruction des clés complètes (entité + colonne)
                entity_keys: List[Union[str, Tuple]] = [
                    entity + (col,) if entity else col for col in cols
                ]

                # Extraction de la fréquence cible et du masque de l'entité
                entity_target = self.get_entity_target_frequency(entity, target_frequency)
                entity_mask = self.get_entity_mask(df, entity)

                # Détection de la fréquence source depuis l'index temporel de l'entité
                entity_time_index = pd.DatetimeIndex(
                    df.loc[entity_mask].index.get_level_values(-1)
                )
                source_freq = detect_index_frequency(entity_time_index)

                # Orientation de la conversion pour cette entité
                if source_freq and is_higher_frequency(
                    normalize_frequency(entity_target),
                    normalize_frequency(source_freq),
                ):
                    interpolate_keys.extend(entity_keys)
                else:
                    aggregate_keys.extend(entity_keys)

        # Application des conversions
        result = df
        if aggregate_keys:
            result = self.aggregate_to_target(
                result, aggregate_keys, target_frequency, is_panel
            )
        if interpolate_keys:
            result = self.interpolate_to_target(
                result, interpolate_keys, target_frequency, is_panel,
                method=interp_method,
                limit=interp_limit,
                limit_direction=interp_limit_direction,
                limit_area=limit_area,
            )
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
