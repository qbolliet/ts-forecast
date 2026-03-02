"""Imputation window calculation for mixed frequency imputation.

This module provides the ImputationWindowCalculator class to compute the
temporal window where all series have real (non-NaN) values using a
multi-frequency coverage matrix approach, and to extend this window based
on attrition threshold and imputation scope parameters.
"""
# Importation des modules
# Modules de base
import warnings
from typing import Dict, List, Literal, Optional, Union, Tuple, cast

# Manipulation de données
import numpy as np
import pandas as pd

# Modules du package
from .detector import detect_dataset_frequency, detect_index_frequency, detect_and_parse_index_frequency
from ..panel.utils import get_unique_panel_entities
from ..utils.frequency.utils import get_frequency_order
from ..utils.position.utils import combine_frequency_position
from ..utils.time.utils import get_period_start, get_period_end

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
    frequency, then derives attrition (fraction of columns covered) at
    each date. The strict window is where attrition equals 1.0. The
    extended window grows backward or forward as long as attrition meets
    the specified threshold.

    For panel data (MultiIndex with entity levels + time level), windows
    are computed independently per entity.

    Attributes:
        imputation_window_start_: Start of the strict window. Scalar for
            time series, Dict[tuple, Timestamp] for panel.
        imputation_window_end_: End of the strict window. Same type.
        training_start_: Start of the training window (scope-dependent).
            Same type as imputation_window_start_.
        training_end_: End of the training window. Same type.
        attrition_by_date_: Ratio of columns covered per high-freq date.
            pd.Series for time series, Dict[tuple, pd.Series] for panel.
        index_freq_: Detected highest frequency. str for time series,
            Dict[tuple, str] for panel.
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
        attrition_threshold: float = 0.5,
        imputation_scope: ImputationScope = 'strict',
        min_columns: int = 2,
    ):
        """Initialize the ImputationWindowCalculator.

        Args:
            attrition_threshold: Minimum fraction of columns that must have
                coverage for a date to be included in the extended window.
                Value between 0 and 1. Default 0.5.
            imputation_scope: How to determine the imputation window:
                - 'strict': Only dates where all columns have coverage.
                - 'extended_backward': Extend before the strict window
                  where attrition >= threshold.
                - 'extended_forward': Extend after the strict window
                  where attrition >= threshold.
                - 'extended_both': Extend in both directions.
            min_columns: Minimum number of data columns required.
                Must be at least 2. Default 2.

        Raises:
            ValueError: If attrition_threshold not in [0, 1], invalid
                imputation_scope, or min_columns < 2.
        """
        # Validation des paramètres
        if not 0 <= attrition_threshold <= 1:
            raise ValueError(
                f"attrition_threshold must be between 0 and 1, got {attrition_threshold}"
            )
        if imputation_scope not in ('strict', 'extended_backward', 'extended_forward', 'extended_both'):
            raise ValueError(
                f"imputation_scope must be one of 'strict', 'extended_backward', "
                f"'extended_forward', 'extended_both', got '{imputation_scope}'"
            )
        if min_columns < 2:
            raise ValueError(f"min_columns must be at least 2, got {min_columns}")

        # Stockage des paramètres
        self.attrition_threshold = attrition_threshold
        self.imputation_scope = imputation_scope
        self.min_columns = min_columns

        # Attributs calculés — scalaires pour TS, dict par entité pour panel
        self.imputation_window_start_: Optional[Union[pd.Timestamp, Dict[tuple, Optional[pd.Timestamp]]]] = None
        self.imputation_window_end_: Optional[Union[pd.Timestamp, Dict[tuple, Optional[pd.Timestamp]]]] = None
        self.attrition_by_date_: Optional[Union[pd.Series, Dict[tuple, Optional[pd.Series]]]] = None
        self.index_freq_: Optional[Union[str, Dict[tuple, Optional[str]]]] = None
        self.column_coverage_: Optional[Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]] = None

        # Attributs internes
        self._detected_frequencies: Optional[Dict] = None
        self._is_panel: bool = False
        self._is_fitted: bool = False

    # Méthode d'entraînement du calculateur
    def fit(self, data: pd.DataFrame) -> 'ImputationWindowCalculator':
        """Calculate the imputation and training windows from data.

        Detects column frequencies, builds a coverage matrix at the
        highest detected frequency, then computes the strict window
        (all columns covered) and the scope-extended training window.

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
        self._is_panel = isinstance(data.index, pd.MultiIndex)

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
        """Compute windows for time series data (DatetimeIndex)."""
        # Fréquence la plus élevée parmi toutes les colonnes
        valid_freqs = [f for f in self._detected_frequencies.values() if f is not None]
        if not valid_freqs:
            raise ValueError("Cannot detect any column frequency in the data")
        
        # Calcul de la fenêtre pour l'unique entité TS
        result = self._compute_window(data, self._detected_frequencies, self.index_freq_)

        # Valorisation des attributs
        self.imputation_window_start_ = result['imputation_start']
        self.imputation_window_end_ = result['imputation_end']
        self.attrition_by_date_ = result['attrition']
        self.column_coverage_ = result['column_coverage']

    # Méthode auxiliaire d'estimation de la fenêtre sur des données de panel
    def _fit_panel(self, data: pd.DataFrame) -> None:
        """Compute per-entity windows for panel data (MultiIndex)."""
        # Extraction des entités des données
        entities = get_unique_panel_entities(data)

        # Initialisation des dictionnaires de résultats
        self.imputation_window_start_ = {}
        self.imputation_window_end_ = {}
        self.attrition_by_date_ = {}
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

            # Cas où la fréquence de l'index n'a pas pu être identifié
            if self.index_freq_[entity] is None:
                self.imputation_window_start_[entity] = None
                self.imputation_window_end_[entity] = None
                self.attrition_by_date_[entity] = None
                continue
            
            # Calcul de la fenêtre d'imputation
            result = self._compute_window(entity_df, col_freqs, self.index_freq_[entity])

            # Complétion des dictionnaires
            self.imputation_window_start_[entity] = result['imputation_start']
            self.imputation_window_end_[entity] = result['imputation_end']
            self.attrition_by_date_[entity] = result['attrition']
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

    # Méthode auxiliaire de la fenêtre pour une série temporelle
    def _compute_window(
        self,
        df: pd.DataFrame,
        col_freqs: Dict[str, Optional[str]],
        index_freq: str,
    ) -> Dict:
        """Compute the imputation and training windows for one entity.

        Args:
            df: DataFrame with simple DatetimeIndex for the entity.
            col_freqs: Dict mapping column names to detected frequencies.
            index_freq: Highest (most granular) frequency for this
                entity, used as the reference grid frequency.

        Returns:
            Dict with keys: 'imputation_start', 'imputation_end',
            'attrition', 'column_coverage'.
        """
        # Initialisation du dictionnaire résultat par défaut
        _none_result = {
            'imputation_start': None, 'imputation_end': None,
            'attrition': None, 'column_coverage': None,
        }

        # Construction de la grille haute fréquence de référence
        grid = self._build_index_freq_grid(df, col_freqs, index_freq)

        # Cas où la grille est vide
        if grid is None or len(grid) == 0:
            return _none_result

        # Construction de la matrice de couverture booléenne
        coverage_matrix = self._build_coverage_matrix(df, col_freqs, grid, index_freq)

        # Calcul de l'attrition (proportion de colonnes couvertes par date)
        attrition = coverage_matrix.mean(axis=1)

        # Extraction de la couverture par colonne (premier/dernier True dans la grille)
        column_coverage = {}
        for col in coverage_matrix.columns:
            col_bool = np.asarray(coverage_matrix[col], dtype=bool)
            covered_grid_dates = grid[col_bool]
            if len(covered_grid_dates) > 0:
                column_coverage[col] = (covered_grid_dates[0], covered_grid_dates[-1])
            else:
                column_coverage[col] = (None, None)

        # Détermination de la fenêtre stricte (attrition == 1.0)
        # Utilisation d'un seuil légèrement inférieur à 1.0 pour les erreurs d'arrondis
        strict_mask = attrition >= (1.0 - 1e-10)
        strict_indices = np.where(np.asarray(strict_mask, dtype=bool))[0]

        # Cas où pour aucune période l'ensemble des variables sont disponibles simultanément
        if len(strict_indices) == 0:
            warnings.warn(
                "There is no period in the DataFrame where all the variables are available.",
                UserWarning
            )

        # Initialisation des bornes à celles de la période stricte
        window_start = grid[strict_indices[0]]
        window_end = grid[strict_indices[-1]]

        # Calcul de la fenêtre d'entraînement selon imputation_scope
        if self.imputation_scope in ('extended_backward', 'extended_both'):
            window_start = self._extend_backward(attrition, window_start)

        if self.imputation_scope in ('extended_forward', 'extended_both'):
            window_end = self._extend_forward(attrition, window_end)
        
        # Cas où la fenêtre ne contient qu'une seule observation
        if window_start == window_end:
            warnings.warn(
                "Imputation window contains only one observation. Consider relaxing "
                "constraints or using a different imputation_scope.",
                UserWarning
            )
        
        # Vérification de la contiguïté de la fenêtre stricte
        if len(strict_indices) > 1:
            gaps = np.diff(strict_indices)
            if np.any(gaps > 1):
                warnings.warn(
                    "The strict imputation window is not contiguous: there are gaps where "
                    "not all columns have data. Reporting [first, last] bounds.",
                    UserWarning
                )

        return {
            'imputation_start': window_start,
            'imputation_end': window_end,
            'attrition': attrition,
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
            # Extraction de la fréquence de 
            col_freq = col_freqs.get(col) or index_freq
            # Extraction des observations valides (i.e. non null)
            valid = df[col].dropna()
            if len(valid) == 0:
                continue
            # Parcours des dates de l'index
            for d in valid.index:
                # Extraction des débuts et fin de période
                expanded_starts.append(pd.Timestamp(get_period_start(d, col_freq)))
                expanded_ends.append(pd.Timestamp(get_period_end(d, col_freq)))

        # Cas au la liste est vide
        if not expanded_starts:
            return None

        # Initialisation de des dates de début et de fin de grille
        # Début de la grille
        grid_start = min(expanded_starts)
        # Fin de la grille
        # get_period_end est exclusif (borne supérieure exclue)
        # pd.date_range avec cette borne inclura jusqu'au dernier point avant cette date
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
                    _, pos, _ = detect_and_parse_index_frequency(
                        cast(pd.DatetimeIndex, hf_dates)
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
            # Extraction de la fréquence des colonnes
            col_freq = col_freqs.get(col) or index_freq
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
                # Mise à jour du mask
                col_mask |= np.asarray((grid >= p_start) & (grid < p_end), dtype=bool)
            # Ajout du masque associé à la colonne
            coverage[col] = col_mask

        return coverage

    # Méthode auxiliaire d'extension de la fenêtre d'imputation avant le début de la période stricte
    def _extend_backward(
        self,
        attrition: pd.Series,
        window_start: pd.Timestamp,
    ) -> pd.Timestamp:
        """Extend the window start backward where attrition meets threshold.

        The extension searches only before window_start. The upper bound
        of the backward extension region is window_start itself.

        Args:
            attrition: Attrition series on the high-freq grid.
            window_start: Current strict window start (upper limit of extension).

        Returns:
            New (earlier or equal) start timestamp.
        """
        # Extraction des observations de la série d'attrition antérieures au début de la fenêtre stricte
        before = attrition[attrition.index < window_start]
        # Si le début de la fenêtre stricte correspond au début de la série, la borne est inchangée
        if before.empty:
            return window_start
        # Extraction des observations pour lesquelles l'attrition est supérieure au seuil
        valid = before[before >= self.attrition_threshold]
        # Si aucune observation n'a une attrition supérieure au seuil, la borne stricte est conservée
        if valid.empty:
            return window_start
        # Retourne le minimum des observations valides
        return valid.index.min()

    # Méthode auxiliaire d'extension de la fenêtre d'imputation après la fin de la période stricte
    def _extend_forward(
        self,
        attrition: pd.Series,
        window_end: pd.Timestamp,
    ) -> pd.Timestamp:
        """Extend the window end forward where attrition meets threshold.

        The extension searches only after window_end. The lower bound
        of the forward extension region is window_end itself.

        Args:
            attrition: Attrition series on the high-freq grid.
            window_end: Current strict window end (lower limit of extension).

        Returns:
            New (later or equal) end timestamp.
        """
        # Extraction des observations de la série d'attrition postérieures à la fin de la fenêtre stricte
        after = attrition[attrition.index > window_end]
        # Si la fin de la fenêtre stricte correspond à la fin de la série, la borne est inchangée
        if after.empty:
            return window_end
        # Extraction des observations pour lesquelles l'attrition est supérieure au seuil
        valid = after[after >= self.attrition_threshold]
        # Si aucune observation n'a une attrition supérieure au seuil, la borne stricte est conservée
        if valid.empty:
            return window_end
        # Retourne le maximum des observations valides
        return valid.index.max()


    def get_imputation_window_mask(self, data: pd.DataFrame) -> pd.Series:
        """Get a boolean mask for observations in the strict imputation window.

        The strict window is the period where all columns have data
        (attrition == 1.0), independent of imputation_scope.

        Args:
            data: DataFrame to create mask for.

        Returns:
            Boolean Series aligned with data.index.

        Raises:
            ValueError: If calculator not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        if self._is_panel:
            return self._get_panel_temporal_mask(data, use_training=False)

        # Cas séries temporelles
        time_index = data.index
        mask = (
            (time_index >= self.imputation_window_start_)
            & (time_index <= self.imputation_window_end_)
        )
        return pd.Series(mask, index=data.index)

    def _get_panel_temporal_mask(
        self,
        data: pd.DataFrame,
        use_training: bool,
        column: Optional[str] = None,
    ) -> pd.Series:
        """Build a temporal mask for panel data using per-entity windows.

        Args:
            data: Panel DataFrame with MultiIndex.
            use_training: If True, use training window; otherwise strict window.
            column: Optional column for additional non-null filtering.

        Returns:
            Boolean Series aligned with data.index.
        """
        from ..panel.utils import get_unique_panel_entities

        combined = pd.Series(False, index=data.index)
        entities = get_unique_panel_entities(data)
        time_vals = data.index.get_level_values(-1)

        training_start_map = cast(Dict, self.training_start_)
        training_end_map = cast(Dict, self.training_end_)
        window_start_map = cast(Dict, self.imputation_window_start_)
        window_end_map = cast(Dict, self.imputation_window_end_)

        for entity in entities:
            if use_training:
                t_start = training_start_map.get(entity)
                t_end = training_end_map.get(entity)
            else:
                t_start = window_start_map.get(entity)
                t_end = window_end_map.get(entity)

            if t_start is None or t_end is None:
                continue

            entity_row_mask = self._get_entity_row_mask(data, entity)
            temporal_mask = (time_vals >= t_start) & (time_vals <= t_end)
            combined = combined | (entity_row_mask & temporal_mask)

        if column is not None:
            combined = combined & data[column].notna()

        return combined

    def get_mask_at_frequency(
        self,
        data: pd.DataFrame,
        frequency: str,
        column: Optional[str] = None,
    ) -> pd.Series:
        """Get the strict imputation window mask resampled to a lower frequency.

        A lower-frequency period is included in the mask only if ALL its
        high-frequency sub-periods fall within the imputation window
        (uses the 'all' aggregation method via FrequencyConverter).

        Only supported for time series data (DatetimeIndex). For panel
        data, raises ValueError.

        Args:
            data: DataFrame with simple DatetimeIndex.
            frequency: Target (lower) frequency offset string (e.g., 'QE', 'Y').
            column: Optional column name for additional non-null filtering.

        Returns:
            Boolean Series at the target frequency.

        Raises:
            ValueError: If calculator not fitted or data has MultiIndex.

        Examples:
            >>> monthly_calc.fit(data)
            >>> quarterly_mask = monthly_calc.get_mask_at_frequency(data, 'QE')
        """
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        if isinstance(data.index, pd.MultiIndex):
            raise ValueError(
                "get_mask_at_frequency is not supported for panel data (MultiIndex). "
                "Use get_imputation_window_mask per entity instead."
            )

        from ..utils.frequency.converter import FrequencyConverter

        # Construction du masque booléen à la fréquence native
        time_index = data.index
        mask = (
            (time_index >= self.imputation_window_start_)
            & (time_index <= self.imputation_window_end_)
        )
        mask_series = pd.Series(mask.astype(int), index=time_index)

        if column is not None and column in data.columns:
            mask_series = mask_series & data[column].notna().astype(int)

        # Agrégation vers la fréquence cible : True seulement si toutes les sous-périodes le sont
        converter = FrequencyConverter()
        resampled = converter.aggregate_to_lower_frequency(mask_series, frequency, method='all')

        return pd.Series(resampled.fillna(False), dtype=bool)

    # Méthode d'extrction de la couverture associée à chaque série
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

        # Cas où la couverture des colonnes n'es
        if self.column_coverage_ is None:
            return []

        # Initialisation de la liste résultat
        columns_with_coverage = []
        # Parcours des colonnes
        for col, (col_start, col_end) in self.column_coverage_.items():
            # Identification des colonnes avec une couverture non nulle
            if col_start is None or col_end is None:
                continue
            # Chevauchement si col_start <= end ET col_end >= start
            if col_start <= end and col_end >= start:
                columns_with_coverage.append(col)

        return columns_with_coverage
