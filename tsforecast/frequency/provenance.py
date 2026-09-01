"""Provenance tracking for imputed values in time series data.

This module provides the ImputationProvenanceTracker class to track the origin
of each value in an imputed dataset, distinguishing original values, exact
aggregations, disaggregated sub-periods, interpolated values, and the five
levels of model-imputation taint (see ProvenanceType). It also exposes the
taint primitives shared by the imputation plan: the CellOrigin / Taint aliases
and the functions resolve_model_provenance, origin_to_taint and max_origin.
"""
# Importation des modules
# Modules de base
from typing import Dict, Iterable, List, Literal, Optional, Union, Tuple
from enum import Enum
# Manipulation de données
import numpy as np
import pandas as pd

# Utilitaires de panel
from ..panel.utils import detect_panel_structure


# Enumération des types de provenance
class ProvenanceType(str, Enum):
    """Enumeration of value provenance types.

    Non-model cells:
        ORIGINAL: Value was present in the input dataset.
        AGGREGATED: Value is the exact additive aggregation of finer true
            values over a complete period. An exact additive aggregation of
            observations is not an approximation, so it never taints a model
            that only ever saw such values.
        DISAGGREGATED: Value describes a POSITION as much as an origin: it
            sits in a sub-period of an observed lower-frequency total spread
            over its whole period. Two cases carry the mark: a cell whose
            period was rescaled so its sub-periods sum back to the observed
            total (``enforce_period_totals=True``), and an ANCHOR DATE
            re-expressed at the stage frequency — the row that held the
            low-frequency observation itself — whether or not the rescaling
            ran. It is therefore AMBIGUOUS as to confidence level: only the
            first case guarantees the additive identity, the second only
            states that the cell sits where a real observation was. Because
            of that ambiguity DISAGGREGATED must NEVER be used as a filter to
            compose ``y_train`` nor to compute a taint level — those read the
            origin store, never the provenance matrix.
        INTERPOLATED: Value was produced by interpolating observations
            (strategy ``'interpolate'``, a covariate fallback, or the
            interpolation fallback of a model whose fit failed).

    Model cells, by worst ingredient the model saw and on which side:
        MODEL_ON_TRUE: The model only saw true values, AGGREGATED ones
            included (an exact additive aggregation of observations is not an
            approximation).
        MODEL_ON_INTERPOLATED: At least one interpolated value among the
            covariates OR in ``y_train``, and no model value anywhere.
        MODEL_ON_IMPUTED: At least one covariate imputed by a model (possibly
            alongside true, aggregated and interpolated values); ``y_train``
            stays clean.
        MODEL_ON_IMPUTED_TARGET: ``y_train`` holds at least one model-imputed
            value (``impute_intermediate_frequencies=True``); the covariates
            do not.
        MODEL_ON_IMPUTED_BOTH: Both sides carry a model-imputed value.
    """
    ORIGINAL = 'original'
    AGGREGATED = 'aggregated'
    DISAGGREGATED = 'disaggregated'
    INTERPOLATED = 'interpolated'

    MODEL_ON_TRUE = 'model_on_true'
    MODEL_ON_INTERPOLATED = 'model_on_interpolated'
    MODEL_ON_IMPUTED = 'model_on_imputed'
    MODEL_ON_IMPUTED_TARGET = 'model_on_imputed_target'
    MODEL_ON_IMPUTED_BOTH = 'model_on_imputed_both'

    # Représentation lisible (utilisée par l'affichage tabulaire de pandas,
    # p.ex. provenance_matrix_) : 'original' plutôt que 'ProvenanceType.ORIGINAL'.
    # __repr__ reste celui d'Enum (<ProvenanceType.ORIGINAL: 'original'>), qui
    # continue de distinguer un ProvenanceType d'un str brut dans les affichages
    # scalaires (repr de liste, retour de get_provenance, etc.).
    def __str__(self) -> str:
        return self.value


# Alias de typage : origine d'une cellule et niveau de souillure associé.
# L'ordre des littéraux de CellOrigin est l'ordre CROISSANT de souillure,
# exploité par "max_origin".
CellOrigin = Literal['observed', 'interpolated', 'model']
Taint = Literal['none', 'interpolated', 'imputed']

# Ordre de souillure des origines, pour "max_origin"
_ORIGIN_ORDER: Dict[str, int] = {'observed': 0, 'interpolated': 1, 'model': 2}

# Correspondance origine -> souillure, pour "origin_to_taint"
_ORIGIN_TO_TAINT: Dict[str, Taint] = {
    'observed': 'none',
    'interpolated': 'interpolated',
    'model': 'imputed',
}


# Résolution de la provenance émise par une étape à modèle
def resolve_model_provenance(covariate_taint: Taint, target_taint: Taint) -> ProvenanceType:
    """Map the two training taints of a step onto its emitted MODEL_* provenance.

    Args:
        covariate_taint: Worst taint among the covariates the model read
            (``'none'``, ``'interpolated'`` or ``'imputed'``).
        target_taint: Worst taint among the ``y_train`` rows the model fit on.

    Returns:
        The MODEL_* provenance every cell produced by the step's model carries.

    Examples:
        >>> resolve_model_provenance('none', 'none')
        <ProvenanceType.MODEL_ON_TRUE: 'model_on_true'>
        >>> resolve_model_provenance('imputed', 'imputed')
        <ProvenanceType.MODEL_ON_IMPUTED_BOTH: 'model_on_imputed_both'>
    """
    if target_taint == 'imputed':
        return (ProvenanceType.MODEL_ON_IMPUTED_BOTH if covariate_taint == 'imputed'
                else ProvenanceType.MODEL_ON_IMPUTED_TARGET)
    if covariate_taint == 'imputed':
        return ProvenanceType.MODEL_ON_IMPUTED
    if 'interpolated' in (covariate_taint, target_taint):
        return ProvenanceType.MODEL_ON_INTERPOLATED
    return ProvenanceType.MODEL_ON_TRUE


# Correspondance origine d'une cellule -> niveau de souillure
def origin_to_taint(origin: CellOrigin) -> Taint:
    """Map a cell origin onto its taint level.

    Args:
        origin: One of ``'observed'``, ``'interpolated'``, ``'model'``.

    Returns:
        ``'none'`` for ``'observed'``, ``'interpolated'`` for
        ``'interpolated'``, ``'imputed'`` for ``'model'``.

    Raises:
        KeyError: If ``origin`` is not a valid CellOrigin.

    Examples:
        >>> origin_to_taint('observed')
        'none'
        >>> origin_to_taint('model')
        'imputed'
    """
    return _ORIGIN_TO_TAINT[origin]


# Maximum d'un itérable d'origines sur l'ordre croissant de souillure
def max_origin(origins: Iterable[CellOrigin]) -> CellOrigin:
    """Return the most-tainted origin over the increasing taint order.

    Args:
        origins: Iterable of cell origins.

    Returns:
        The origin with the highest taint level; ``'observed'`` on an empty
        iterable (the least-tainted origin, neutral for the operation).

    Examples:
        >>> max_origin(['observed', 'model', 'interpolated'])
        'model'
        >>> max_origin([])
        'observed'
    """
    best: CellOrigin = 'observed'
    for origin in origins:
        if _ORIGIN_ORDER[origin] > _ORIGIN_ORDER[best]:
            best = origin
    return best


# Classe de suivi de la provenance des valeurs imputées
class ImputationProvenanceTracker:
    """Track the provenance (origin) of each value in an imputed dataset.

    This class maintains a matrix parallel to the data that records how each
    value was obtained: from the input data, through exact aggregation or
    disaggregation, through interpolation, or through model-based imputation
    at one of the five taint levels (see ProvenanceType).

    Attributes:
        provenance_matrix_: DataFrame with same shape as data, containing ProvenanceType values.
        statistics_: Dictionary with counts and percentages per provenance type.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> tracker = ImputationProvenanceTracker()
        >>> dates = pd.date_range('2023-01-01', periods=12, freq='M')
        >>> data = pd.DataFrame({
        ...     'var1': [1, 2, np.nan, 4, np.nan, 6, 7, 8, 9, 10, 11, 12],
        ...     'var2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        ... }, index=dates)
        >>> tracker.initialize(data)
        >>> tracker.mark_imputed('var1', dates[2], ProvenanceType.MODEL_ON_TRUE)
        >>> tracker.get_provenance('var1', dates[2])
        <ProvenanceType.MODEL_ON_TRUE: 'model_on_true'>
    """
    # Initialisation
    def __init__(self):
        """Initialize the ImputationProvenanceTracker."""
        # Initialisation de la matrice de provenance
        self.provenance_matrix_: Optional[pd.DataFrame] = None
        # Initialisation des statistiques
        self.statistics_: Optional[Dict[str, Dict[str, Union[int, float]]]] = None
        # Colonnes de panel (résolues lors de initialize())
        self._panel_cols: Optional[List[str]] = None

    # Méthode d'initialisation sur le jeu de données initial
    def initialize(
        self,
        data: pd.DataFrame,
        panel_cols: Optional[List[str]] = None
    ) -> 'ImputationProvenanceTracker':
        """Initialize the provenance matrix from input data.

        Creates a provenance matrix with the same shape as the input data,
        marking all non-null values as ORIGINAL and null values as None
        (to be filled during imputation).

        Args:
            data: Input DataFrame with potential NaN values to be imputed.
                Panel data can be passed either as a MultiIndex (entity levels
                followed by a time level) or as a flat DataFrame carrying the
                entity as ordinary column(s).
            panel_cols: Names identifying panel entities. If None, panel
                structure is auto-detected from a MultiIndex.
                If provided and matching MultiIndex level names, the panel is
                assumed to already be in the index. Otherwise, panel_cols are
                treated as ordinary DataFrame columns and excluded from
                provenance tracking.

        Returns:
            self: The initialized tracker.

        Raises:
            ValueError: If data is empty or not a DataFrame.

        Examples:
            >>> tracker = ImputationProvenanceTracker()
            >>> data = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, np.nan]})
            >>> tracker.initialize(data)
            >>> tracker.provenance_matrix_['a'].tolist()
            [<ProvenanceType.ORIGINAL: 'original'>, None, <ProvenanceType.ORIGINAL: 'original'>]
        """
        # Validation des entrées
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pandas DataFrame, got {type(data).__name__}")
        if data.empty:
            raise ValueError("data cannot be empty")

        # Détection de la structure de panel : colonnes explicites, ou
        # auto-détection depuis un MultiIndex
        panel_cols, panel_in_index = detect_panel_structure(data, panel_cols)
        self._panel_cols = panel_cols
        is_panel = panel_cols is not None and len(panel_cols) > 0

        # Détermination des colonnes à tracker : seul un panel porté par des
        # colonnes ordinaires nécessite une exclusion, un panel en index ne
        # laisse déjà que les variables dans les colonnes
        if is_panel and not panel_in_index:
            track_cols = [col for col in data.columns if col not in panel_cols]
        else:
            track_cols = list(data.columns)

        # Création de la matrice de provenance
        self.provenance_matrix_ = pd.DataFrame(
            index=data.index,
            columns=track_cols,
            dtype=object
        )

        # Initialisation: valeurs non-nulles -> ORIGINAL, valeurs nulles -> None
        for col in track_cols:
            mask_not_null = data[col].notna()
            self.provenance_matrix_.loc[mask_not_null, col] = ProvenanceType.ORIGINAL
            # Les valeurs nulles restent à None (à remplir lors de l'imputation)

        return self

    # Méthode de marquage d'observartions comme "imputées"
    def mark_imputed(
        self,
        column: str,
        index: Union[pd.Timestamp, pd.DatetimeIndex, slice],
        provenance: ProvenanceType
    ) -> None:
        """Mark specific values as imputed with given provenance type.

        Args:
            column: Name of the column containing the imputed values.
            index: Index location(s) of the imputed value(s). Can be:
                - Single timestamp
                - DatetimeIndex for multiple values
                - Slice for a range of values
            provenance: Type of provenance to assign (e.g. MODEL_ON_TRUE,
                INTERPOLATED, AGGREGATED, MODEL_ON_IMPUTED).

        Raises:
            ValueError: If provenance matrix not initialized or invalid inputs.

        Examples:
            >>> tracker.mark_imputed('var1', pd.Timestamp('2023-03-31'), ProvenanceType.MODEL_ON_TRUE)
            >>> tracker.mark_imputed('var1', data.index[2:5], ProvenanceType.MODEL_ON_IMPUTED)
        """
        # Validation de l'initialisation
        if self.provenance_matrix_ is None:
            raise ValueError("Provenance matrix not initialized. Call initialize() first.")

        # Validation de la colonne
        if column not in self.provenance_matrix_.columns:
            raise ValueError(f"Column '{column}' not found in provenance matrix")

        # Validation de la provenance
        if not isinstance(provenance, ProvenanceType):
            raise ValueError(f"provenance must be a ProvenanceType, got {type(provenance).__name__}")

        # Marquage de la provenance
        self.provenance_matrix_.loc[index, column] = provenance

    # Méthode de marquage de certaines observations comme "agrégées"
    def mark_aggregated(
        self,
        column: str,
        index: Union[pd.Timestamp, pd.DatetimeIndex, slice]
    ) -> None:
        """Mark specific values as obtained through aggregation.

        Convenience method for marking values as AGGREGATED.

        Args:
            column: Name of the column containing the aggregated values.
            index: Index location(s) of the aggregated value(s).

        Examples:
            >>> tracker.mark_aggregated('daily_var', monthly_index)
        """
        self.mark_imputed(column, index, ProvenanceType.AGGREGATED)

    # Méthpode de marquage de certaines observations comme "désagrégées"
    def mark_disaggregated(
        self,
        column: str,
        index: Union[pd.Timestamp, pd.DatetimeIndex, slice]
    ) -> None:
        """Mark specific values as obtained through disaggregation.

        Convenience method for marking values as DISAGGREGATED, i.e. the
        sub-periods of a lower-frequency observation spread over its whole
        period. The mark is AMBIGUOUS as to confidence level: it describes a
        POSITION (``sub-period of an observed total``) as much as an origin.
        Only a rescaled period guarantees the additive identity; an anchor
        date re-expressed at the stage frequency merely sits where a real
        observation was. It must therefore NEVER be used as a filter to
        compose ``y_train`` nor to compute a taint level — those read the
        origin store, never the provenance matrix.

        Args:
            column: Name of the column containing the disaggregated values.
            index: Index location(s) of the disaggregated value(s).

        Examples:
            >>> tracker.mark_disaggregated('quarterly_var', monthly_index)
        """
        self.mark_imputed(column, index, ProvenanceType.DISAGGREGATED)

    # Méthode de marquage de certaines observations comme "interpolées"
    def mark_interpolated(
        self,
        column: str,
        index: Union[pd.Timestamp, pd.DatetimeIndex, slice]
    ) -> None:
        """Mark specific values as obtained through interpolation.

        Convenience method for marking values as INTERPOLATED, i.e. produced
        by interpolating observations — strategy ``'interpolate'``, a
        covariate fallback, or the interpolation fallback of a model whose
        fit failed. Symmetric of :meth:`mark_aggregated` and
        :meth:`mark_disaggregated`.

        Args:
            column: Name of the column containing the interpolated values.
            index: Index location(s) of the interpolated value(s).

        Examples:
            >>> tracker.mark_interpolated('quarterly_var', monthly_index)
        """
        self.mark_imputed(column, index, ProvenanceType.INTERPOLATED)

    # Méthode de marquage de certaines observations comme imputées par un modèle
    def mark_model_imputed(
        self,
        column: str,
        index: Union[pd.Timestamp, pd.DatetimeIndex, slice],
        covariate_taint: Taint = 'none',
        target_taint: Taint = 'none',
    ) -> None:
        """Mark specific values as imputed by a model, at the given taint levels.

        The emitted provenance is resolved from the two training taints of
        the step by :func:`resolve_model_provenance`.

        Args:
            column: Name of the column containing the imputed values.
            index: Index location(s) of the imputed value(s).
            covariate_taint: Worst taint among the covariates the model read
                (``'none'``, ``'interpolated'`` or ``'imputed'``).
            target_taint: Worst taint among the ``y_train`` rows the model
                fit on.

        Examples:
            >>> # Model that only saw true values
            >>> tracker.mark_model_imputed('var1', dates[2])
            >>> # Model that read a model-imputed covariate
            >>> tracker.mark_model_imputed('var1', dates[5], covariate_taint='imputed')
        """
        provenance = resolve_model_provenance(covariate_taint, target_taint)
        self.mark_imputed(column, index, provenance)

    # Méthode de remise à zéro de la provenance de certaines observations
    def clear_provenance(
        self,
        column: str,
        index: Union[pd.Timestamp, pd.DatetimeIndex, slice]
    ) -> None:
        """Reset the provenance of specific cells to "not filled".

        Symmetric of :meth:`mark_imputed`: it removes a provenance instead of
        setting one. The cells go back to ``None``, the value
        :meth:`initialize` gives to a NaN cell, so that a cell emptied by the
        cascade stops being declared ORIGINAL while it no longer carries any
        value.

        Args:
            column: Name of the column holding the cleared values.
            index: Index location(s) of the cell(s) to reset.

        Raises:
            ValueError: If provenance matrix not initialized or column unknown.

        Examples:
            >>> tracker.clear_provenance('pib_trimestriel', anchor_dates)
            >>> tracker.get_provenance('pib_trimestriel', anchor_dates[0]) is None
            True
        """
        # Validation de l'initialisation
        if self.provenance_matrix_ is None:
            raise ValueError("Provenance matrix not initialized. Call initialize() first.")

        # Validation de la colonne
        if column not in self.provenance_matrix_.columns:
            raise ValueError(f"Column '{column}' not found in provenance matrix")

        # Remise à l'état "non renseigné" (convention de initialize pour un NaN)
        self.provenance_matrix_.loc[index, column] = None

    # Méthode d'extraction de la provenance
    def get_provenance(
        self,
        column: str,
        index: Union[pd.Timestamp, pd.DatetimeIndex, slice]
    ) -> Union[ProvenanceType, pd.Series]:
        """Get the provenance of specific value(s).

        Args:
            column: Name of the column.
            index: Index location(s) to query.

        Returns:
            ProvenanceType for single index, or Series of ProvenanceType for multiple indices.

        Raises:
            ValueError: If provenance matrix not initialized or invalid inputs.

        Examples:
            >>> tracker.get_provenance('var1', pd.Timestamp('2023-03-31'))
            <ProvenanceType.MODEL_ON_TRUE: 'model_on_true'>
        """
        # Validation de l'initialisation
        if self.provenance_matrix_ is None:
            raise ValueError("Provenance matrix not initialized. Call initialize() first.")

        # Validation de la colonne
        if column not in self.provenance_matrix_.columns:
            raise ValueError(f"Column '{column}' not found in provenance matrix")

        return self.provenance_matrix_.loc[index, column]

    # Méthode d'extraction du masque
    def get_mask(
        self,
        provenance_types: Union[ProvenanceType, List[ProvenanceType]],
        column: Optional[str] = None
    ) -> pd.DataFrame:
        """Get a boolean mask for values with specified provenance type(s).

        Args:
            provenance_types: Single ProvenanceType or list of types to filter.
            column: Optional column name to filter. If None, returns mask for all columns.

        Returns:
            Boolean DataFrame/Series indicating positions with matching provenance.

        Raises:
            ValueError: If provenance matrix not initialized.

        Examples:
            >>> # Get mask of all model-imputed values
            >>> mask = tracker.get_mask([ProvenanceType.MODEL_ON_TRUE, ProvenanceType.MODEL_ON_IMPUTED])
            >>> # Get mask of original values for specific column
            >>> mask = tracker.get_mask(ProvenanceType.ORIGINAL, column='var1')
        """
        # Validation de l'initialisation
        if self.provenance_matrix_ is None:
            raise ValueError("Provenance matrix not initialized. Call initialize() first.")

        # Normalisation en liste
        if not isinstance(provenance_types, list):
            provenance_types = [provenance_types]

        # Sélection du sous-ensemble
        if column is not None:
            if column not in self.provenance_matrix_.columns:
                raise ValueError(f"Column '{column}' not found in provenance matrix")
            data = self.provenance_matrix_[column]
        else:
            data = self.provenance_matrix_

        # Création du masque
        return data.isin(provenance_types)

    # Méthode de calcule de statistique sur la provenance des observations
    def compute_statistics(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Compute statistics about provenance distribution.

        Calculates counts and percentages for each provenance type,
        both overall and per column.

        Returns:
            Dictionary with statistics per column and overall.
            Structure: {
                'overall': {'original': count, 'original_pct': pct, ...},
                'column_name': {'original': count, 'original_pct': pct, ...},
                ...
            }

        Raises:
            ValueError: If provenance matrix not initialized.

        Examples:
            >>> stats = tracker.compute_statistics()
            >>> print(f"Original values: {stats['overall']['original_pct']:.1f}%")
            Original values: 75.0%
        """
        # Validation de l'initialisation
        if self.provenance_matrix_ is None:
            raise ValueError("Provenance matrix not initialized. Call initialize() first.")

        # Initialisation du dictionnaire de statistiques
        self.statistics_ = {}

        # Calcul des statistiques globales
        total_values = self.provenance_matrix_.size
        overall_stats = {}

        for prov_type in ProvenanceType:
            mask = self.provenance_matrix_ == prov_type
            count = mask.sum().sum()
            overall_stats[prov_type.value] = count
            overall_stats[f'{prov_type.value}_pct'] = (count / total_values * 100) if total_values > 0 else 0.0

        # Comptage des valeurs non-imputées (None dans la matrice)
        null_count = self.provenance_matrix_.isna().sum().sum()
        overall_stats['not_imputed'] = null_count
        overall_stats['not_imputed_pct'] = (null_count / total_values * 100) if total_values > 0 else 0.0

        self.statistics_['overall'] = overall_stats

        # Calcul des statistiques par colonne
        for col in self.provenance_matrix_.columns:
            col_stats = {}
            col_values = len(self.provenance_matrix_[col])

            for prov_type in ProvenanceType:
                mask = self.provenance_matrix_[col] == prov_type
                count = mask.sum()
                col_stats[prov_type.value] = count
                col_stats[f'{prov_type.value}_pct'] = (count / col_values * 100) if col_values > 0 else 0.0

            # Comptage des valeurs non-imputées
            null_count = self.provenance_matrix_[col].isna().sum()
            col_stats['not_imputed'] = null_count
            col_stats['not_imputed_pct'] = (null_count / col_values * 100) if col_values > 0 else 0.0

            self.statistics_[col] = col_stats

        return self.statistics_

    # Méthode d'extraction de la matrice de provenance
    def get_provenance_matrix(self) -> pd.DataFrame:
        """Get the full provenance matrix.

        Returns:
            DataFrame containing ProvenanceType values for each cell.

        Raises:
            ValueError: If provenance matrix not initialized.
        """
        if self.provenance_matrix_ is None:
            raise ValueError("Provenance matrix not initialized. Call initialize() first.")
        return self.provenance_matrix_.copy()

    # Méthode de conversion de la matrice de provenance en chaîne de caractères
    def to_string_matrix(self) -> pd.DataFrame:
        """Convert provenance matrix to string representation.

        Returns:
            DataFrame with string values ('original', 'model_on_true', etc.) instead of enums.

        Raises:
            ValueError: If provenance matrix not initialized.

        Examples:
            >>> string_matrix = tracker.to_string_matrix()
            >>> string_matrix.iloc[0, 0]
            'original'
        """
        if self.provenance_matrix_ is None:
            raise ValueError("Provenance matrix not initialized. Call initialize() first.")

        # Conversion des ProvenanceType en chaînes
        def convert_to_string(val):
            if val is None or pd.isna(val):
                return 'not_imputed'
            elif isinstance(val, ProvenanceType):
                return val.value
            else:
                return str(val)

        return self.provenance_matrix_.map(convert_to_string)

    # Méthode d'appariement de deux jeux de données
    def merge(
        self,
        other: 'ImputationProvenanceTracker',
        how: Literal['update', 'preserve'] = 'update'
    ) -> 'ImputationProvenanceTracker':
        """Merge another tracker's provenance information into this one.

        Useful for combining provenance from multiple imputation stages.

        Args:
            other: Another ImputationProvenanceTracker to merge.
            how: Merge strategy:
                - 'update': Values from other overwrite values in self
                - 'preserve': Keep values from self, only fill None values from other

        Returns:
            self: The merged tracker.

        Raises:
            ValueError: If matrices have incompatible shapes or one is not initialized.

        Examples:
            >>> tracker1.merge(tracker2, how='update')
        """
        # Validation
        if self.provenance_matrix_ is None:
            raise ValueError("This tracker's provenance matrix is not initialized")
        if other.provenance_matrix_ is None:
            raise ValueError("Other tracker's provenance matrix is not initialized")

        # Vérification des formes compatibles
        if self.provenance_matrix_.shape != other.provenance_matrix_.shape:
            raise ValueError(
                f"Incompatible shapes: {self.provenance_matrix_.shape} vs {other.provenance_matrix_.shape}"
            )

        # Application de la stratégie de fusion
        if how == 'update':
            # Les valeurs non-None de other écrasent celles de self
            mask_not_none = other.provenance_matrix_.notna()
            self.provenance_matrix_[mask_not_none] = other.provenance_matrix_[mask_not_none]
        elif how == 'preserve':
            # On ne remplit que les valeurs None de self avec celles de other
            mask_none = self.provenance_matrix_.isna()
            self.provenance_matrix_[mask_none] = other.provenance_matrix_[mask_none]
        else:
            raise ValueError(f"Invalid merge strategy: {how}. Must be 'update' or 'preserve'")

        # Invalidation des statistiques (doivent être recalculées)
        self.statistics_ = None

        return self

    # Reprénsentation de la classe sous-forme de chaîne de caractères
    def __repr__(self) -> str:
        """String representation of the tracker."""
        if self.provenance_matrix_ is None:
            return "ImputationProvenanceTracker(not initialized)"

        shape = self.provenance_matrix_.shape
        return f"ImputationProvenanceTracker(rows={shape[0]}, cols={shape[1]})"
