"""Imputation window calculation for mixed frequency imputation.

This module provides the ImputationWindowCalculator class to compute the
temporal window where all series have real (non-NaN) values using a
multi-frequency coverage matrix approach, and to extend this window based
on coverage threshold and imputation scope parameters.
"""
# Importation des modules
# Modules de base
import warnings
from typing import Dict, List, Literal, Optional, Union, Tuple, cast

# Manipulation de données
import numpy as np
import pandas as pd

# Modules du package
from .detector import detect_dataset_frequency, detect_index_frequency
from ..panel.utils import get_unique_panel_entities, is_panel_data, normalize_entity_key
from ..utils.frequency.utils import is_higher_frequency, normalize_frequency
from ..utils.position.utils import combine_frequency_position
from ..utils.time.utils import get_period_start, get_period_end
from ..utils.frequency.converter import FrequencyConverter

# Type pour le scope d'imputation
ImputationScope = Literal['strict', 'extended_backward', 'extended_forward', 'extended_both']


# Classe de calcul de la fenêtre d'imputation à partir des couvertures multi-fréquences
class ImputationWindowCalculator:
    """Calculate the imputation window and extended training windows for imputation.

    The strict imputation window is the temporal interval where ALL series
    in the dataset have data (directly or via sub-period coverage). A
    quarterly observation, for example, is considered to cover all
    high-frequency sub-periods (e.g., months) within that quarter.

    The class builds a boolean coverage matrix at the highest detected
    frequency, then derives coverage (fraction of columns covered) at
    each date. The strict window is where coverage equals 1.0. The
    extended training window grows backward or forward as long as coverage
    meets the specified threshold.

    For panel data (MultiIndex with entity levels + time level), windows
    are computed independently per entity.

    Attributes:
        imputation_window_start_: Start of the strict imputation window
            (coverage == 1.0). Scalar for time series,
            Dict[tuple, Optional[Timestamp]] for panel.
        imputation_window_end_: End of the strict imputation window.
            Same type as imputation_window_start_.
        imputation_window_mask_: Boolean mask on the high-frequency grid
            identifying observations in the strict window the imputation_window. 
            pd.Series for time series,Dict[tuple, Optional[pd.Series]] for panel.
        coverage_by_date_: Ratio of columns covered per high-freq date.
            pd.Series for time series, Dict[tuple, Optional[pd.Series]]
            for panel.
        index_freq_: Detected highest frequency. str for time series,
            Dict[tuple, Optional[str]] for panel.
        column_coverage_: Dict mapping column names to (start, end)
            coverage timestamps. Only populated for time series data.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2023-01-31', periods=12, freq='ME')
        >>> data = pd.DataFrame({
        ...     'var1': [np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan],
        ...     'var2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        ... }, index=dates)
        >>> calc = ImputationWindowCalculator(imputation_scope='extended_both')
        >>> calc.fit(data)
        >>> print(calc.imputation_window_start_, calc.imputation_window_end_)
    """

    # Initialisation
    def __init__(
        self,
        coverage_threshold: float = 0.5,
        imputation_scope: ImputationScope = 'strict',
        min_columns: int = 2,
    ):
        """Initialize the ImputationWindowCalculator.

        Args:
            coverage_threshold: Minimum fraction of columns that must have
                coverage for a date to be included in the extended window.
                Value between 0 and 1. Default 0.5.
            imputation_scope: How to determine the imputation window:
                - 'strict': Only dates where all columns have coverage.
                - 'extended_backward': Extend before the strict window
                  where coverage >= threshold.
                - 'extended_forward': Extend after the strict window
                  where coverage >= threshold.
                - 'extended_both': Extend in both directions.
            min_columns: Minimum number of data columns required.
                Must be at least 2. Default 2.

        Raises:
            ValueError: If coverage_threshold not in [0, 1], invalid
                imputation_scope, or min_columns < 2.
        """
        # Validation des paramètres
        if not 0 <= coverage_threshold <= 1:
            raise ValueError(
                f"coverage_threshold must be between 0 and 1, got {coverage_threshold}"
            )
        if imputation_scope not in ('strict', 'extended_backward', 'extended_forward', 'extended_both'):
            raise ValueError(
                f"imputation_scope must be one of 'strict', 'extended_backward', "
                f"'extended_forward', 'extended_both', got '{imputation_scope}'"
            )
        if min_columns < 2:
            raise ValueError(f"min_columns must be at least 2, got {min_columns}")

        # Stockage des paramètres
        self.coverage_threshold = coverage_threshold
        self.imputation_scope = imputation_scope
        self.min_columns = min_columns

        # Attributs de fenêtre d'imputation — scalaires pour TS, dict par entité pour panel
        self.imputation_window_start_: Optional[Union[pd.Timestamp, Dict[tuple, Optional[pd.Timestamp]]]] = None
        self.imputation_window_end_: Optional[Union[pd.Timestamp, Dict[tuple, Optional[pd.Timestamp]]]] = None
        self.imputation_window_mask_: Optional[Union[pd.Series, Dict[tuple, Optional[pd.Series]]]] = None

        # Attributs auxiliaires
        self.coverage_by_date_: Optional[Union[pd.Series, Dict[tuple, Optional[pd.Series]]]] = None
        self.index_freq_: Optional[Union[str, Dict[tuple, Optional[str]]]] = None
        self.column_coverage_: Optional[Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]] = None

        # Attributs internes
        self._detected_frequencies: Optional[Dict] = None
        self._is_panel: bool = False
        self._is_fitted: bool = False
        self._converter = FrequencyConverter()

    # Méthode d'entraînement du calculateur
    def fit(self, data: pd.DataFrame) -> 'ImputationWindowCalculator':
        """Calculate the imputation and training windows from data.

        Detects column frequencies, builds a coverage matrix at the
        highest detected frequency, then computes the strict imputation
        window (all columns covered) and the scope-extended imputation
        window. Boolean masks for the imputation window are built and stored
        alongside the window bounds.

        Args:
            data: DataFrame with DatetimeIndex (time series) or MultiIndex
                where the last level is time and earlier levels identify
                panel entities.

        Returns:
            self: The fitted calculator.

        Raises:
            ValueError: If data is invalid, has too few columns, or no
                valid imputation window exists.

        Examples:
            >>> calc = ImputationWindowCalculator()
            >>> calc.fit(data)
            >>> print(calc.imputation_window_start_, calc.imputation_window_end_)
        """
        # Validation des données d'entrée
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pandas DataFrame, got {type(data).__name__}")
        if data.empty:
            raise ValueError("data cannot be empty")
        if not isinstance(data.index, (pd.DatetimeIndex, pd.MultiIndex)):
            raise ValueError("data must have a DatetimeIndex or MultiIndex with datetime level")

        # Détection du type de données (panel ou séries temporelles)
        self._is_panel = is_panel_data(data=data)

        # Vérification du nombre minimal de colonnes
        if len(data.columns) < self.min_columns:
            raise ValueError(
                f"Data has {len(data.columns)} columns, but min_columns={self.min_columns}"
            )

        # Détection des fréquences par colonne (et par entité pour panel)
        self._detected_frequencies = detect_dataset_frequency(data)
        # Détection de la fréquence de l'index
        self.index_freq_ = detect_index_frequency(data.index)

        # Calcul des fenêtres selon le type de données
        if self._is_panel:
            self._fit_panel(data)
        else:
            self._fit_ts(data)

        self._is_fitted = True
        return self

    # Méthode auxiliaire d'estimation de la fenêtre sur des données de séries temporelles
    def _fit_ts(self, data: pd.DataFrame) -> None:
        """Compute windows and masks for time series data (DatetimeIndex).

        Delegates to _compute_window and stores all resulting attributes:
        imputation window bounds and mask, coverage series, and column coverage mapping.

        Args:
            data: DataFrame with a simple DatetimeIndex.
        """
        # Fréquence la plus élevée parmi toutes les colonnes
        valid_freqs = [f for f in self._detected_frequencies.values() if f is not None]
        if not valid_freqs:
            raise ValueError("Cannot detect any column frequency in the data")

        # Calcul de la fenêtre pour l'unique entité TS
        result = self._compute_window(data, self._detected_frequencies, self.index_freq_)

        # Valorisation des attributs de fenêtre stricte
        self.imputation_window_start_ = result['imputation_start']
        self.imputation_window_end_ = result['imputation_end']
        self.imputation_window_mask_ = result['imputation_window_mask']

        # Valorisation des attributs auxiliaires
        self.coverage_by_date_ = result['coverage']
        self.column_coverage_ = result['column_coverage']

    # Méthode auxiliaire d'estimation de la fenêtre sur des données de panel
    def _fit_panel(self, data: pd.DataFrame) -> None:
        """Compute per-entity windows and masks for panel data (MultiIndex).

        Iterates over each panel entity, extracts its sub-DataFrame, and
        calls _compute_window. Results are aggregated into dicts keyed by
        entity tuple: strict window bounds, strict and training boolean
        masks on each entity's high-frequency grid, and coverage series.

        Args:
            data: Panel DataFrame with MultiIndex whose last level is time.
        """
        # Extraction des entités des données
        entities = get_unique_panel_entities(data)

        # Initialisation des dictionnaires de résultats
        self.imputation_window_start_ = {}
        self.imputation_window_end_ = {}
        self.imputation_window_mask_ = {}
        self.coverage_by_date_ = {}
        self.column_coverage_ = {}

        # Parcours des entités
        for entity in entities:
            # Extraction du sous-DataFrame de l'entité avec index temporel simple
            entity_row_mask = self._get_entity_row_mask(data, entity)
            entity_df = data[entity_row_mask].copy()
            entity_df.index = entity_df.index.get_level_values(-1)

            # Fréquences des colonnes pour cette entité
            col_freqs = {
                col: self._detected_frequencies.get(entity + (col,))
                for col in data.columns
            }

            # Extraction de la clé associée à l'entité
            entity_key = entity[0] if len(entity)==1 else entity

            # Cas où la fréquence de l'index n'a pas pu être identifiée
            if self.index_freq_[entity_key] is None:
                self.imputation_window_start_[entity_key] = None
                self.imputation_window_end_[entity_key] = None
                self.imputation_window_mask_[entity_key] = None
                self.coverage_by_date_[entity_key] = None
                continue

            # Calcul de la fenêtre d'imputation
            result = self._compute_window(entity_df, col_freqs, self.index_freq_[entity_key])

            # Complétion des dictionnaires de fenêtre d'imputation
            self.imputation_window_start_[entity_key] = result['imputation_start']
            self.imputation_window_end_[entity_key] = result['imputation_end']
            self.imputation_window_mask_[entity_key] = result['imputation_window_mask']

            # Complétion des attributs auxiliaires
            self.coverage_by_date_[entity_key] = result['coverage']
            self.column_coverage_ = result['column_coverage']

        # Vérification qu'au moins une entité a une fenêtre valide
        valid_starts = [v for v in self.imputation_window_start_.values() if v is not None]
        if not valid_starts:
            raise ValueError("No imputation window found for any entity in the panel")

    # Méthode auxiliaire permettant d'extraire des observations relatives à une entité du panel
    def _get_entity_row_mask(self, data: pd.DataFrame, entity: tuple) -> np.ndarray:
        """Build a boolean row mask selecting the given entity from panel data.

        Args:
            data: Panel DataFrame with MultiIndex.
            entity: Normalized entity key tuple.

        Returns:
            Boolean numpy array of length len(data).
        """
        # Initialisation du masque
        mask = np.ones(len(data), dtype=bool)
        # Parcours des éléments définissant l'entité
        for i, val in enumerate(entity):
            # Mise à jour du masque
            mask &= (data.index.get_level_values(i) == val)
        return mask

    # Méthode de calcul de la fenêtre d'imputation et d'entraînement pour une entité
    def _compute_window(
        self,
        df: pd.DataFrame,
        col_freqs: Dict[str, Optional[str]],
        index_freq: str,
    ) -> Dict:
        """Compute imputation and training windows for one entity.

        Builds the high-frequency grid and coverage matrix, derives the
        coverage series, then constructs the strict imputation window mask
        (coverage == 1.0). Window bounds are derived from this mask.
        The imputation window mask is then extended according to
        imputation_scope by passing the mask to _extend_backward and/or
        _extend_forward.

        Args:
            df: DataFrame with simple DatetimeIndex for the entity.
            col_freqs: Dict mapping column names to detected frequencies.
            index_freq: Highest (most granular) frequency for this entity,
                used as the reference grid frequency.

        Returns:
            Dict with keys:
                - 'imputation_start': Start of the strict window.
                - 'imputation_end': End of the strict window.
                - 'imputation_window_mask': Boolean pd.Series on the grid
                  identifying observations in the strict window.
                - 'coverage': coverage pd.Series on the high-freq grid.
                - 'column_coverage': Dict of per-column (start, end)
                  tuples.
        """
        # Initialisation du dictionnaire résultat par défaut
        _none_result = {
            'imputation_start': None,
            'imputation_end': None,
            'imputation_window_mask': None,
            'coverage': None,
            'column_coverage': None,
        }

        # Construction de la grille haute fréquence de référence
        grid = self._build_index_freq_grid(df, col_freqs, index_freq)

        # Cas où la grille est vide
        if grid is None or len(grid) == 0:
            return _none_result

        # Construction de la matrice de couverture booléenne
        coverage_matrix = self._build_coverage_matrix(df, col_freqs, grid, index_freq)

        # Calcul de l'coverage (proportion de colonnes couvertes par date)
        coverage = coverage_matrix.mean(axis=1)

        # Extraction de la couverture par colonne (premier/dernier True dans la grille)
        column_coverage = {}
        for col in coverage_matrix.columns:
            col_bool = np.asarray(coverage_matrix[col], dtype=bool)
            covered_grid_dates = grid[col_bool]
            if len(covered_grid_dates) > 0:
                column_coverage[col] = (covered_grid_dates[0], covered_grid_dates[-1])
            else:
                column_coverage[col] = (None, None)

        # Construction du masque de la fenêtre stricte sur la grille haute fréquence
        # Utilisation d'un seuil légèrement inférieur à 1.0 pour les erreurs d'arrondi
        imputation_window_mask = pd.Series(
            np.asarray(coverage >= (1.0 - 1e-10), dtype=bool),
            index=grid,
        )

        # Dérivation des bornes de la fenêtre stricte depuis le masque
        strict_dates = imputation_window_mask.index[imputation_window_mask]

        # Cas où pour aucune période l'ensemble des variables sont disponibles simultanément
        if len(strict_dates) == 0:
            warnings.warn(
                "There is no period in the DataFrame where all the variables are available.",
                UserWarning
            )
            return {
                **_none_result,
                'coverage': coverage,
                'column_coverage': column_coverage,
                'imputation_window_mask': imputation_window_mask,
            }

        # Extraction des bornes strictes depuis le masque
        imputation_start = strict_dates.min()
        imputation_end = strict_dates.max()

        # Vérification de la contiguïté de la fenêtre stricte
        if len(strict_dates) > 1:
            grid_positions = np.searchsorted(grid, strict_dates)
            gaps = np.diff(grid_positions)
            if np.any(gaps > 1):
                warnings.warn(
                    "The strict imputation window is not contiguous: there are gaps where "
                    "not all columns have data. Reporting [first, last] bounds.",
                    UserWarning
                )

        # Extension du masque d'entraînement selon le scope
        if self.imputation_scope in ('extended_backward', 'extended_both'):
            imputation_window_mask = self._extend_backward(coverage, imputation_window_mask)

        if self.imputation_scope in ('extended_forward', 'extended_both'):
            imputation_window_mask = self._extend_forward(coverage, imputation_window_mask)

        # Cas où la fenêtre d'entraînement ne contient qu'une seule observation
        imputation_dates = imputation_window_mask.index[imputation_window_mask]
        if len(imputation_dates) <= 1:
            warnings.warn(
                "Imputation window contains only one observation. Consider relaxing "
                "constraints or using a different imputation_scope.",
                UserWarning
            )

        return {
            'imputation_start': imputation_start,
            'imputation_end': imputation_end,
            'imputation_window_mask': imputation_window_mask,
            'coverage': coverage,
            'column_coverage': column_coverage,
        }

    # Construction de la grille à l'indice de la fréquence pour le calcul de la couverture
    def _build_index_freq_grid(
        self,
        df: pd.DataFrame,
        col_freqs: Dict[str, Optional[str]],
        index_freq: str,
    ) -> Optional[pd.DatetimeIndex]:
        """Build the reference high-frequency date grid for coverage computation.

        The grid spans from the earliest period start to the latest period
        end across all non-NaN observations. Period boundaries are computed
        using get_period_start/get_period_end, so a quarterly observation
        expands the grid to cover all three constituent months.

        Args:
            df: Entity sub-DataFrame with simple DatetimeIndex.
            col_freqs: Dict mapping column names to frequencies.
            index_freq: Highest (most granular) frequency code.

        Returns:
            pd.DatetimeIndex grid, or None if no data.
        """
        # Calcul des bornes de la grille en étendant chaque observation sur la période qu'elle couvre
        # Initialisation des listes de début et de fin de période
        expanded_starts = []
        expanded_ends = []

        # Parcours des colonnes du jeu de données
        for col in df.columns:
            # Extraction de la fréquence de la colonne (fallback sur la fréquence de l'index)
            col_freq = col_freqs.get(col) or index_freq
            # Extraction des observations valides (i.e. non nulles)
            valid = df[col].dropna()
            if len(valid) == 0:
                continue
            # Parcours des dates de l'index
            for d in valid.index:
                # Extraction des débuts et fins de période
                expanded_starts.append(pd.Timestamp(get_period_start(d, col_freq)))
                expanded_ends.append(pd.Timestamp(get_period_end(d, col_freq)))

        # Cas où la liste est vide
        if not expanded_starts:
            return None

        # Initialisation des dates de début et de fin de grille
        # Début de la grille
        grid_start = min(expanded_starts)
        # Fin de la grille — get_period_end est exclusif (borne supérieure exclue)
        grid_end_exclusive = max(expanded_ends)

        # Détection de la position (S/E) à partir des colonnes à la fréquence la plus élevée
        # Extraction des colonnes à haute fréquence
        hf_cols = [col for col in df.columns if col_freqs.get(col) == index_freq]
        # Initialisation de la position
        hf_pos = 'E'  # convention par défaut : fin de période
        if hf_cols:
            # Extraction des dates
            hf_dates = df[hf_cols[0]].dropna().index
            if len(hf_dates) >= 2:
                # Détection de la position
                try:
                    _, pos, _ = detect_index_frequency(
                        cast(pd.DatetimeIndex, hf_dates),
                        return_format='components'
                    )
                    if pos is not None:
                        hf_pos = pos
                except Exception:
                    pass

        # Construction de l'offset pandas complet (fréquence + position)
        try:
            grid_offset = combine_frequency_position(
                index_freq, hf_pos  # type: ignore[arg-type]
            )
        except Exception:
            grid_offset = index_freq

        # Génération de la grille
        try:
            grid = pd.date_range(start=grid_start, end=grid_end_exclusive, freq=grid_offset)
        except Exception:
            # Fallback sans position si l'offset est invalide
            try:
                grid = pd.date_range(start=grid_start, end=grid_end_exclusive, freq=index_freq)
            except Exception:
                return None

        return grid

    # Méthode auxiliaire de construction de la matrice de couverture
    def _build_coverage_matrix(
        self,
        df: pd.DataFrame,
        col_freqs: Dict[str, Optional[str]],
        grid: pd.DatetimeIndex,
        index_freq: str,
    ) -> pd.DataFrame:
        """Build the boolean coverage matrix at the high-frequency grid.

        Each cell (date, column) is True if the column has a non-NaN
        observation whose period contains that grid date.

        Args:
            df: Entity sub-DataFrame with simple DatetimeIndex.
            col_freqs: Dict mapping column names to frequencies.
            grid: Reference high-frequency DatetimeIndex.
            index_freq: Highest (most granular) frequency code.

        Returns:
            Boolean DataFrame with grid as index and columns as columns.
        """
        # Initialisation de la matrice de couverture
        coverage = pd.DataFrame(False, index=grid, columns=df.columns)

        # Parcours des colonnes
        for col in df.columns:
            # Extraction de la fréquence de la colonne
            col_freq = col_freqs[col]
            # Extraction des observations non nulles
            valid = df[col].dropna()
            if len(valid) == 0:
                continue
            # Initialisation du masque de la colonne
            col_mask = np.zeros(len(grid), dtype=bool)
            # Parcours des dates associées aux observations non nulles
            for d in valid.index:
                # Détermination des bornes de la période
                p_start = pd.Timestamp(get_period_start(d, col_freq))
                p_end = pd.Timestamp(get_period_end(d, col_freq))
                # Convention [p_start, p_end) : p_end est exclusif
                col_mask |= np.asarray((grid >= p_start) & (grid < p_end), dtype=bool)
            # Ajout du masque associé à la colonne
            coverage[col] = col_mask

        return coverage

    # Méthode auxiliaire d'extension du masque d'entraînement avant le début de la fenêtre stricte
    def _extend_backward(
        self,
        coverage: pd.Series,
        mask: pd.Series,
    ) -> pd.Series:
        """Extend a boolean window mask backward where coverage meets threshold.

        Searches for dates strictly before the earliest True date of the
        mask where coverage >= coverage_threshold, and sets them to True
        in the returned mask.

        Args:
            coverage: coverage series on the high-freq grid.
            mask: Boolean pd.Series on the high-freq grid representing the
                current window (strict or partially extended).

        Returns:
            New boolean pd.Series with additional True values prepended
            where coverage meets the threshold. Returns an unchanged copy
            if no extension is possible.
        """
        # Identification de la borne de début actuelle du masque
        masked_dates = mask.index[mask]
        if len(masked_dates) == 0:
            return mask.copy()
        window_start = masked_dates.min()

        # Extraction des observations de la série d'coverage antérieures à la borne actuelle
        before = coverage[coverage.index < window_start]
        # Si la borne actuelle correspond au début de la grille, le masque est inchangé
        if before.empty:
            return mask.copy()
        # Extraction des observations pour lesquelles l'coverage dépasse le seuil
        valid = before[before >= self.coverage_threshold]
        # Si aucune observation ne satisfait le seuil, le masque est inchangé
        if valid.empty:
            return mask.copy()

        # Extension du masque : activation des dates satisfaisant le seuil
        new_mask = mask.copy()
        new_mask[valid.index] = True
        return new_mask

    # Méthode auxiliaire d'extension du masque d'entraînement après la fin de la fenêtre stricte
    def _extend_forward(
        self,
        coverage: pd.Series,
        mask: pd.Series,
    ) -> pd.Series:
        """Extend a boolean window mask forward where coverage meets threshold.

        Searches for dates strictly after the latest True date of the mask
        where coverage >= coverage_threshold, and sets them to True in
        the returned mask.

        Args:
            coverage: coverage series on the high-freq grid.
            mask: Boolean pd.Series on the high-freq grid representing the
                current window (strict or partially extended).

        Returns:
            New boolean pd.Series with additional True values appended
            where coverage meets the threshold. Returns an unchanged copy
            if no extension is possible.
        """
        # Identification de la borne de fin actuelle du masque
        masked_dates = mask.index[mask]
        if len(masked_dates) == 0:
            return mask.copy()
        window_end = masked_dates.max()

        # Extraction des observations de la série d'coverage postérieures à la borne actuelle
        after = coverage[coverage.index > window_end]
        # Si la borne actuelle correspond à la fin de la grille, le masque est inchangé
        if after.empty:
            return mask.copy()
        # Extraction des observations pour lesquelles l'coverage dépasse le seuil
        valid = after[after >= self.coverage_threshold]
        # Si aucune observation ne satisfait le seuil, le masque est inchangé
        if valid.empty:
            return mask.copy()

        # Extension du masque : activation des dates satisfaisant le seuil
        new_mask = mask.copy()
        new_mask[valid.index] = True
        return new_mask

    # Méthode d'extraction du masque booléen de la fenêtre d'imputation
    def get_imputation_window_mask(
        self,
        data: Optional[Union[pd.DataFrame, pd.Series]] = None,
    ) -> Union[pd.Series, Dict[tuple, Optional[pd.Series]]]:
        """Get the boolean imputation-window mask, optionally aligned to data.

        Args:
            data: If provided, the mask is re-indexed on ``data.index``
                (dates absent from the fitted grid are False). For panel
                data, a single boolean Series aligned on the MultiIndex
                rows of ``data`` is returned instead of a dict.

        Returns:
            Boolean Series aligned on ``data.index`` if ``data`` is given.
            Otherwise the raw fitted mask (Series for time series, dict
            mapping entity tuples to Series for panel).

        Raises:
            ValueError: If calculator not fitted.
        """
        # Vérification de l'entraînement du calculateur
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        # Sans données : retour du masque brut sur la grille interne
        if data is None:
            return self.imputation_window_mask_

        # Extraction de l'index
        index = data.index

        # Cas des données de panel : reconstruction d'un masque aligné sur le MultiIndex
        if is_panel_data(data=data):
            values = np.zeros(len(index), dtype=bool)
            entity_levels = [index.get_level_values(i) for i in range(index.nlevels - 1)]
            dates = index.get_level_values(-1)
            # Parcours des masques par entité
            for entity, entity_mask in (self.imputation_window_mask_ or {}).items():
                if entity_mask is None:
                    continue
                # Normalisation de la clé d'entité en tuple
                entity_tuple = normalize_entity_key(entity)
                # Sélection des lignes de l'entité
                rows = np.ones(len(index), dtype=bool)
                for level_values, wanted in zip(entity_levels, entity_tuple):
                    rows &= (level_values == wanted)
                # Alignement du masque de l'entité sur les dates de ses lignes
                aligned = (
                    entity_mask.reindex(dates[rows])
                    .fillna(False)
                    .to_numpy(dtype=bool)
                )
                values[np.flatnonzero(rows)] = aligned
            return pd.Series(values, index=index)

        # Cas des séries temporelles : simple réindexation sur l'index des données
        return (
            self.imputation_window_mask_
            .reindex(index)
            .fillna(False)
            .astype(bool)
        )


    # Méthode de conversion du masque d'imputation vers une fréquence inférieure
    # /!\ Voir si dans le cas de données de panel, plutôt que de retourner comme masque un dictionnaire Dict[entity, Series], il n'est pas préférable de retourner une séries avec un multiIndex, ce qui permettrait d'utiliser convert_frequency ici
    def get_mask_at_frequency(
        self,
        frequency: Union[str, Dict[tuple, str]],
    ) -> Union[pd.Series, Dict[tuple, Optional[pd.Series]]]:
        """Get the strict imputation window mask resampled to a lower frequency.

        A lower-frequency period is True if and only if all its high-frequency
        sub-periods fall within the imputation window. The conversion sums the
        boolean mask (cast to int) at the target frequency, divides by the
        conversion factor between source and target frequencies, takes the
        floor, and casts to bool.

        For panel data, the conversion factor is computed independently per
        entity, as each entity may have a different source frequency.

        Args:
            frequency: Target (lower) frequency offset string (e.g., 'QE',
                'YE') or dictionnary entity -> frequency string.

        Returns:
            Boolean Series at the target frequency (time series), or dict
            mapping each entity tuple to a boolean Series at the target
            frequency (panel). Entities without a valid fitted mask map to
            None.

        Raises:
            ValueError: If calculator not fitted, or if the target frequency
                is not lower than the mask frequency.

        Examples:
            >>> monthly_calc.fit(data)
            >>> quarterly_mask = monthly_calc.get_mask_at_frequency('QE')
        """
        # Vérification de l'entraînement du calculateur
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        # Cas des séries temporelles : conversion unique
        if not self._is_panel:
            return self._convert_mask_to_frequency(
                mask=self.imputation_window_mask_,
                source_freq=self.index_freq_,
                target_freq=frequency,
            )

        # Cas des données de panel : conversion indépendante par entité, chacune
        # pouvant avoir sa propre fréquence source et sa propre fréquence cible
        return {
            entity: self._convert_mask_to_frequency(
                mask=entity_mask,
                source_freq=self.index_freq_.get(entity),
                target_freq=frequency[entity] if isinstance(frequency, dict) else frequency,
            )
            for entity, entity_mask in self.imputation_window_mask_.items()
        }

    # Méthode auxiliaire de conversion d'un masque booléen vers une fréquence inférieure
    def _convert_mask_to_frequency(
        self,
        mask: Optional[pd.Series],
        source_freq: Optional[str],
        target_freq: str,
    ) -> Optional[pd.Series]:
        """Resample a single boolean window mask to a lower frequency.

        A target period is True if and only if all its sub-periods at the
        source frequency are covered by the mask: the boolean mask is summed
        at the target frequency and compared to the number of sub-periods the
        target period contains. Partially covered target periods (at the edges
        of the grid) therefore yield False.

        Args:
            mask: Boolean Series on the source-frequency grid, or None.
            source_freq: Frequency of the mask index, or None.
            target_freq: Target (lower) frequency offset string.

        Returns:
            Boolean Series at the target frequency, or None if either the
            mask or the source frequency is missing.

        Raises:
            ValueError: If source_freq is not higher than target_freq.

        Examples:
            >>> grid = pd.date_range('2023-01-31', periods=12, freq='ME')
            >>> mask = pd.Series(True, index=grid)
            >>> calc._convert_mask_to_frequency(mask, 'M', 'YE').tolist()
            [True]
        """
        # Absence de masque ou de fréquence source exploitable
        if mask is None or source_freq is None:
            return None

        # Vérification que la fréquence source est plus élevée que la fréquence cible
        if not is_higher_frequency(source_freq, target_freq):
            raise ValueError(
                f"Index mask frequency : {source_freq} should be higher than "
                f"frequency : {target_freq}"
            )

        # Agrégation par somme des sous-périodes couvertes
        summed = self._converter.aggregate_to_lower_frequency(
            mask.astype(int), target_freq, method='sum'
        )

        # Comparaison au nombre de sous-périodes attendues : une période cible
        # n'est retenue que si elle est intégralement couverte. Le comptage est
        # délégué au convertisseur, seul dépositaire de la logique de comptage
        # des sous-périodes (cf. FrequencyConverter.count_subperiods)
        expected = self._converter.count_subperiods_per_period(
            summed.index, target_freq, source_freq
        )
        return pd.Series(np.asarray(summed) >= expected, index=summed.index)

    # Méthode d'extraction de la couverture associée à chaque série
    def get_columns_with_coverage(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> List[str]:
        """Get list of columns that have coverage in a given time range.

        Only supported for time series data. Uses the first and last dates
        where the column contributes True values to the coverage matrix.

        Args:
            start: Start of the query range.
            end: End of the query range.

        Returns:
            List of column names with coverage overlapping [start, end].

        Raises:
            ValueError: If calculator not fitted.
        """
        # Vérification que le calculateur est estimé
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        # Cas où la couverture des colonnes n'est pas disponible
        if self.column_coverage_ is None:
            return []

        # Initialisation de la liste résultat
        columns_with_coverage = []
        # Parcours des colonnes
        for col, (col_start, col_end) in self.column_coverage_.items():
            # Exclusion des colonnes sans couverture définie
            if col_start is None or col_end is None:
                continue
            # Chevauchement si col_start <= end ET col_end >= start
            if col_start <= end and col_end >= start:
                columns_with_coverage.append(col)

        return columns_with_coverage