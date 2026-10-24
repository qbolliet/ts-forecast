"""Provenance tracking for imputed values in time series data.

This module provides the ImputationProvenanceTracker class to track the origin
of each value in an imputed dataset, distinguishing original values, exact
aggregations, disaggregated sub-periods, interpolated values, and the five
levels of model-imputation taint (see ProvenanceType), in a categorical matrix
of dtype PROVENANCE_DTYPE. It also exposes the taint primitives shared by the
imputation plan: the CellOrigin / Taint aliases and the functions
resolve_model_provenance, origin_to_taint and max_origin, plus
provenance_statistics, the per-type census of a provenance matrix.
"""
# Importation des modules
# Modules de base
from typing import Dict, Iterable, List, Literal, Optional, Union, Tuple
from enum import Enum
# Manipulation de données
import numpy as np
import pandas as pd

# Utilitaires de panel
from ...panel.utils import detect_panel_structure


# Enumération des types de provenance
class ProvenanceType(str, Enum):
    """Enumeration of value provenance types.

    Non-model cells:
        ORIGINAL: Value was present in the input dataset.
        AGGREGATED: Value is the exact additive aggregation of finer true
            values over a complete period. An exact additive aggregation of
            observations is not an approximation, so it never taints a model
            that only ever saw such values.
        DISAGGREGATED: Never written by ``HighFrequencyImputer`` (its
            aggregation constraint leaves provenance untouched), but available
            through :meth:`ImputationProvenanceTracker.mark_disaggregated`:
            the cell sits in a sub-period of an observed
            lower-frequency total spread over its whole period, either because
            its period was rescaled so its sub-periods sum back to the total
            (``enforce_period_totals=True``) or because it is an ANCHOR DATE
            re-expressed at the stage frequency. It is AMBIGUOUS as to
            confidence level — it replaces the mark the value would otherwise
            carry, so it says neither whether the additive identity holds nor
            whether the value came from a model or from an interpolation — and
            must NEVER be used as a filter to compose ``y_train`` nor to
            compute a taint level; those read the origin store, never the
            provenance matrix. ``HighFrequencyImputer`` NEVER emits it:
            rescaling to the period totals leaves provenance untouched, so a
            rescaled cell and an overwritten anchor row keep the ``MODEL_*``
            or INTERPOLATED mark of the value written there.
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

    Cell produced without any anchor of its own column, for its own entity:
        MODEL_UNANCHORED: The entity never observes the column, and received a
            complete imputation learned on the other entities of the panel
            (``impute_unobserved_entities=True``). It answers another question
            than the five MODEL_* levels above — not "what is the worst
            ingredient the model saw?" but "is this cell tied to an
            observation of its own column, for its own entity?" — which is why
            it does not fit on their scale and primes over them: whatever the
            two taints, which stay computed and frozen in the plan step for
            diagnosis.
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

    MODEL_UNANCHORED = 'model_unanchored'

    # Représentation lisible (formatage, messages) : 'original' plutôt que
    # 'ProvenanceType.ORIGINAL'. __repr__ reste celui d'Enum
    # (<ProvenanceType.ORIGINAL: 'original'>), qui distingue un membre d'une
    # simple chaîne dans les affichages scalaires (retour de get_provenance, etc.).
    def __str__(self) -> str:
        return self.value


# Dtype des matrices de provenance : catégoriel à vocabulaire fermé, une catégorie
# par valeur de ProvenanceType (ordre de déclaration, sans relation d'ordre).
# Une cellule est stockée sur un code entier : comptages et masques vectorisés
# sans dépendre du stockage des chaînes, et toute écriture hors vocabulaire lève.
# Un membre s'écrit tel quel (il est égal à sa valeur) ; une cellule se relit
# en simple chaîne, NaN pour une cellule non renseignée.
PROVENANCE_DTYPE = pd.CategoricalDtype([member.value for member in ProvenanceType])


# Alias de typage : origine d'une cellule et niveau de souillure associé.
# L'ordre des littéraux de CellOrigin est l'ordre CROISSANT de souillure,
# exploité par "max_origin".
CellOrigin = Literal['observed', 'interpolated', 'model']
Taint = Literal['none', 'interpolated', 'imputed']

# Niveaux de souillure admis, pour "resolve_model_provenance"
_TAINTS: Tuple[str, ...] = ('none', 'interpolated', 'imputed')

# Ordre de souillure des origines, pour "max_origin"
_ORIGIN_ORDER: Dict[str, int] = {'observed': 0, 'interpolated': 1, 'model': 2}

# Dtype des origines de cellules (registres du matérialiseur, lignes d'un jeu
# d'entraînement) : catégoriel ORDONNÉ dans l'ordre croissant de souillure. Les
# catégories sont fixes, si bien que deux séries d'origines se combinent sans
# perdre leur dtype, et que l'origine la plus souillée d'une série est son
# maximum. Une cellule se relit en simple chaîne, NaN quand aucune origine
CELL_ORIGIN_DTYPE = pd.CategoricalDtype(list(_ORIGIN_ORDER), ordered=True)

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

    Raises:
        ValueError: If a taint is not one of ``'none'``, ``'interpolated'``
            or ``'imputed'``. Returning a provenance for a mistyped taint
            would silently write a misleading mark in the provenance matrix.

    Examples:
        >>> resolve_model_provenance('none', 'none')
        <ProvenanceType.MODEL_ON_TRUE: 'model_on_true'>
        >>> resolve_model_provenance('imputed', 'imputed')
        <ProvenanceType.MODEL_ON_IMPUTED_BOTH: 'model_on_imputed_both'>
    """
    # Validation des souillures : une faute de frappe ne doit pas être lue comme "indemne"
    for name, taint in (('covariate_taint', covariate_taint), ('target_taint', target_taint)):
        if taint not in _TAINTS:
            raise ValueError(f"{name} must be one of {_TAINTS}, got {taint!r}")
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


# Origine la plus souillée d'une série d'origines
def max_cell_origin(origins: pd.Series) -> CellOrigin:
    """Return the most-tainted origin of a series of cell origins, vectorized.

    Series counterpart of :func:`max_origin`: the maximum of the ordered
    categorical :data:`CELL_ORIGIN_DTYPE`, missing cells ignored.

    Args:
        origins: Cell origins, in :data:`CELL_ORIGIN_DTYPE` or as strings of
            the same vocabulary; missing values are ignored.

    Returns:
        The origin with the highest taint level; ``'observed'`` when no cell
        holds an origin (the least-tainted origin, neutral for the operation).

    Raises:
        ValueError: If a value is outside the vocabulary of :data:`CellOrigin`.

    Examples:
        >>> max_cell_origin(pd.Series(['observed', None, 'interpolated']))
        'interpolated'
        >>> max_cell_origin(pd.Series([], dtype=CELL_ORIGIN_DTYPE))
        'observed'
    """
    # Contrôle du vocabulaire avant conversion, une série typée y étant conforme
    if origins.dtype != CELL_ORIGIN_DTYPE:
        unknown = origins.notna() & ~origins.isin(CELL_ORIGIN_DTYPE.categories)
        if unknown.any():
            raise ValueError(
                f"Unknown cell origin(s) {sorted(set(origins[unknown]))}: expected one of "
                f"{list(CELL_ORIGIN_DTYPE.categories)}"
            )
    # Codes de souillure : -1 pour une cellule sans origine
    codes = origins.astype(CELL_ORIGIN_DTYPE).cat.codes.to_numpy()
    if not len(codes) or codes.max() < 0:
        return 'observed'
    return CELL_ORIGIN_DTYPE.categories[codes.max()]


# Statistiques de répartition d'une matrice de provenance
def provenance_statistics(
    matrix: pd.DataFrame,
) -> Dict[str, Dict[str, Union[int, float]]]:
    """Count the cells of a provenance matrix per provenance type.

    Shared by :meth:`ImputationProvenanceTracker.compute_statistics` and by
    the matrices ``HighFrequencyImputer.transform`` assembles stage by stage,
    which no tracker holds.

    Args:
        matrix: Provenance matrix, one ``ProvenanceType`` value (or a missing
            value for a cell never marked) per cell, preferably of dtype
            ``PROVENANCE_DTYPE``; any other dtype is converted to it first, a
            value outside the vocabulary then counting as not marked.

    Returns:
        ``{'overall': {...}, column: {...}}``. Each block holds, for every
        ``ProvenanceType``, the count under its value and the percentage
        under ``<value>_pct``, plus ``not_imputed`` / ``not_imputed_pct`` for
        the unmarked cells. Percentages are 0.0 on an empty matrix.

    Examples:
        >>> matrix = pd.DataFrame(
        ...     {'a1': [ProvenanceType.ORIGINAL, None]}, dtype=PROVENANCE_DTYPE
        ... )
        >>> stats = provenance_statistics(matrix)
        >>> stats['a1']['original'], stats['overall']['not_imputed_pct']
        (1, 50.0)
    """
    # Bloc de statistiques d'un ensemble de codes de catégorie (-1 : non renseignée)
    def _block(codes: np.ndarray) -> Dict[str, Union[int, float]]:
        total = codes.size
        # Décalage de 1 : la case 0 compte les cellules non renseignées,
        # la case i + 1 la catégorie i (ordre de ProvenanceType)
        counts = np.bincount(codes.ravel() + 1, minlength=len(ProvenanceType) + 1)
        block: Dict[str, Union[int, float]] = {}
        for position, prov_type in enumerate(ProvenanceType, start=1):
            count = int(counts[position])
            block[prov_type.value] = count
            block[f'{prov_type.value}_pct'] = (count / total * 100) if total > 0 else 0.0
        null_count = int(counts[0])
        block['not_imputed'] = null_count
        block['not_imputed_pct'] = (null_count / total * 100) if total > 0 else 0.0
        return block

    # Codes de catégorie de chaque cellule, colonne par colonne (par position :
    # des noms de colonnes dupliqués restent distincts)
    matrix = matrix.astype(PROVENANCE_DTYPE)
    codes = np.empty(matrix.shape, dtype=np.int64)
    for position in range(matrix.shape[1]):
        codes[:, position] = matrix.iloc[:, position].cat.codes.to_numpy()

    # Bloc global, puis un bloc par colonne
    statistics: Dict[str, Dict[str, Union[int, float]]] = {'overall': _block(codes)}
    for position, column in enumerate(matrix.columns):
        statistics[column] = _block(codes[:, position])
    return statistics


# Classe de suivi de la provenance des valeurs imputées
class ImputationProvenanceTracker:
    """Track the provenance (origin) of each value in an imputed dataset.

    This class maintains a matrix parallel to the data that records how each
    value was obtained: from the input data, through exact aggregation or
    disaggregation, through interpolation, or through model-based imputation
    at one of the five taint levels (see ProvenanceType).

    Attributes:
        provenance_matrix_: DataFrame with the index of the data and its
            tracked columns, of dtype ``PROVENANCE_DTYPE``: one
            ``ProvenanceType`` value per cell (read back as a plain string),
            NaN for a cell not marked.
        statistics_: Dictionary with counts and percentages per provenance type.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> tracker = ImputationProvenanceTracker()
        >>> dates = pd.date_range('2023-01-01', periods=12, freq='ME')
        >>> data = pd.DataFrame({
        ...     'var1': [1, 2, np.nan, 4, np.nan, 6, 7, 8, 9, 10, 11, 12],
        ...     'var2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        ... }, index=dates)
        >>> _ = tracker.initialize(data)
        >>> tracker.mark_imputed('var1', dates[2], ProvenanceType.MODEL_ON_TRUE)
        >>> tracker.get_provenance('var1', dates[2])
        <ProvenanceType.MODEL_ON_TRUE: 'model_on_true'>
        >>> tracker.provenance_matrix_['var1'].tolist()[:3]
        ['original', 'original', 'model_on_true']
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

        Creates a provenance matrix of dtype ``PROVENANCE_DTYPE`` with the
        same shape as the input data, marking all non-null values as ORIGINAL
        and leaving null values unfilled (NaN, to be filled during imputation;
        test them with ``pd.isna``).

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
            >>> _ = tracker.initialize(data)
            >>> tracker.provenance_matrix_['a'].isna().tolist()
            [False, True, False]
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
        # laisse déjà que les variables dans les colonnes. Les libellés restent
        # un Index au dtype de l'entrée, même vide
        if is_panel and not panel_in_index:
            track_cols = data.columns[~data.columns.isin(panel_cols)]
        else:
            track_cols = data.columns

        # Matrice catégorielle : valeurs non nulles -> ORIGINAL, valeurs nulles
        # non renseignées (NaN, à remplir lors de l'imputation)
        unfilled = pd.DataFrame(index=data.index, columns=track_cols, dtype=PROVENANCE_DTYPE)
        self.provenance_matrix_ = unfilled.mask(data[track_cols].notna(), ProvenanceType.ORIGINAL)

        return self

    # Méthode d'extension de la matrice à des lignes absentes de l'entrée
    def extend_index(self, index: pd.Index) -> None:
        """Add the rows of ``index`` the provenance matrix does not carry yet.

        The matrix is initialized on the input index, but an imputation stage
        writes on a densified grid: on an irregular index (a few isolated
        annual observations before the start of the monthly grid, say), the
        grid carries dates the input never had. The added rows are "not
        filled" (NaN, like the cells :meth:`initialize` leaves unfilled),
        so that only the subsequent ``mark_*`` call decides their provenance.

        The existing rows keep their order; the matrix is re-sorted only when
        it was already sorted, so that an added date lands in its period
        instead of at the end.

        Args:
            index: Row labels about to be written. Labels already present are
                ignored.

        Raises:
            ValueError: If provenance matrix not initialized.

        Examples:
            >>> tracker = ImputationProvenanceTracker()
            >>> data = pd.DataFrame({'a': [1.0, 2.0]},
            ...                     index=pd.to_datetime(['2015-01-01', '2015-03-01']))
            >>> _ = tracker.initialize(data)
            >>> tracker.extend_index(pd.to_datetime(['2015-02-01']))
            >>> tracker.provenance_matrix_['a'].tolist()
            ['original', nan, 'original']
        """
        # Validation de l'initialisation
        if self.provenance_matrix_ is None:
            raise ValueError("Provenance matrix not initialized. Call initialize() first.")

        # Lignes réellement nouvelles
        missing = index.difference(self.provenance_matrix_.index)
        if len(missing) == 0:
            return

        # Ajout des lignes "non renseignées" (NaN de "reindex", dtype catégoriel
        # conservé), tri conservé s'il existait
        was_sorted = self.provenance_matrix_.index.is_monotonic_increasing
        extended = self.provenance_matrix_.reindex(
            self.provenance_matrix_.index.append(missing)
        )
        self.provenance_matrix_ = extended.sort_index() if was_sorted else extended

    # Garde : une écriture ne crée jamais de ligne (réservé à extend_index)
    def _check_labels_exist(self, index) -> None:
        """Raise KeyError when a scalar label is absent from the matrix.

        ``.loc`` setting enlarges the frame for an absent scalar label, while
        it raises for an absent list-like one; rows are meant to be added by
        :meth:`extend_index` only, so the scalar case is made to raise too.

        Args:
            index: Label(s) about to be written. Slices and list-likes are
                left to pandas (a tuple is a scalar key on a MultiIndex).

        Raises:
            KeyError: If ``index`` is a scalar label (or MultiIndex key)
                absent from the matrix.
        """
        # Extraction de l'index de la matrice
        matrix_index = self.provenance_matrix_.index
        is_scalar_key = not isinstance(index, slice) and (
            not pd.api.types.is_list_like(index)
            or (isinstance(index, tuple) and isinstance(matrix_index, pd.MultiIndex))
        )
        # get_loc lève KeyError pour une étiquette (ou clé partielle) absente
        if is_scalar_key:
            matrix_index.get_loc(index)

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
            KeyError: If a label of ``index`` is absent from the matrix (rows
                are added by :meth:`extend_index` only).

        Examples:
            >>> dates = pd.date_range('2023-01-31', periods=6, freq='ME')
            >>> data = pd.DataFrame({'var1': [1.0, 2.0, np.nan, np.nan, np.nan, 6.0]}, index=dates)
            >>> tracker = ImputationProvenanceTracker().initialize(data)
            >>> tracker.mark_imputed('var1', pd.Timestamp('2023-03-31'), ProvenanceType.MODEL_ON_TRUE)
            >>> tracker.mark_imputed('var1', dates[3:5], ProvenanceType.MODEL_ON_IMPUTED)
            >>> tracker.provenance_matrix_['var1'].tolist()[2:5]
            ['model_on_true', 'model_on_imputed', 'model_on_imputed']
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

        # Marquage de la provenance : le membre s'écrit comme sa catégorie
        self._check_labels_exist(index)
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
            >>> dates = pd.date_range('2023-01-31', periods=3, freq='ME')
            >>> data = pd.DataFrame({'var': [1.0, np.nan, np.nan]}, index=dates)
            >>> tracker = ImputationProvenanceTracker().initialize(data)
            >>> tracker.mark_aggregated('var', dates[1:])
            >>> tracker.provenance_matrix_['var'].tolist()
            ['original', 'aggregated', 'aggregated']
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
        origin store, never the provenance matrix. ``HighFrequencyImputer``
        never calls it, its aggregation constraint leaving provenance
        untouched.

        Args:
            column: Name of the column containing the disaggregated values.
            index: Index location(s) of the disaggregated value(s).

        Examples:
            >>> dates = pd.date_range('2023-01-31', periods=3, freq='ME')
            >>> data = pd.DataFrame({'var': [1.0, np.nan, np.nan]}, index=dates)
            >>> tracker = ImputationProvenanceTracker().initialize(data)
            >>> tracker.mark_disaggregated('var', dates[1:])
            >>> tracker.provenance_matrix_['var'].tolist()
            ['original', 'disaggregated', 'disaggregated']
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
            >>> dates = pd.date_range('2023-01-31', periods=3, freq='ME')
            >>> data = pd.DataFrame({'var': [1.0, np.nan, np.nan]}, index=dates)
            >>> tracker = ImputationProvenanceTracker().initialize(data)
            >>> tracker.mark_interpolated('var', dates[1:])
            >>> tracker.provenance_matrix_['var'].tolist()
            ['original', 'interpolated', 'interpolated']
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
            >>> dates = pd.date_range('2023-01-31', periods=3, freq='ME')
            >>> data = pd.DataFrame({'var': [1.0, np.nan, np.nan]}, index=dates)
            >>> tracker = ImputationProvenanceTracker().initialize(data)
            >>> # Model that only saw true values
            >>> tracker.mark_model_imputed('var', dates[1])
            >>> # Model that read a model-imputed covariate
            >>> tracker.mark_model_imputed('var', dates[2], covariate_taint='imputed')
            >>> tracker.provenance_matrix_['var'].tolist()
            ['original', 'model_on_true', 'model_on_imputed']
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
        setting one. The cells go back to "not filled" (NaN, like the cells
        :meth:`initialize` leaves unfilled), so that a cell emptied by the
        cascade stops being declared ORIGINAL while it no longer carries any
        value.

        Args:
            column: Name of the column holding the cleared values.
            index: Index location(s) of the cell(s) to reset.

        Raises:
            ValueError: If provenance matrix not initialized or column unknown.
            KeyError: If a label of ``index`` is absent from the matrix.

        Examples:
            >>> dates = pd.date_range('2023-01-31', periods=2, freq='ME')
            >>> data = pd.DataFrame({'var': [1.0, 2.0]}, index=dates)
            >>> tracker = ImputationProvenanceTracker().initialize(data)
            >>> tracker.clear_provenance('var', dates[:1])
            >>> tracker.get_provenance('var', dates[0]) is None
            True
        """
        # Validation de l'initialisation
        if self.provenance_matrix_ is None:
            raise ValueError("Provenance matrix not initialized. Call initialize() first.")

        # Validation de la colonne
        if column not in self.provenance_matrix_.columns:
            raise ValueError(f"Column '{column}' not found in provenance matrix")

        # Remise à l'état "non renseigné" (NaN, comme les cellules vides de initialize)
        self._check_labels_exist(index)
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
            For a single label, the ``ProvenanceType`` member of the cell, or
            ``None`` when it is not marked; otherwise the selected cells, of
            dtype ``PROVENANCE_DTYPE``.

        Raises:
            ValueError: If provenance matrix not initialized or invalid inputs.

        Examples:
            >>> dates = pd.date_range('2023-01-31', periods=3, freq='ME')
            >>> data = pd.DataFrame({'var1': [1.0, np.nan, np.nan]}, index=dates)
            >>> tracker = ImputationProvenanceTracker().initialize(data)
            >>> tracker.mark_model_imputed('var1', dates[1])
            >>> tracker.get_provenance('var1', pd.Timestamp('2023-02-28'))
            <ProvenanceType.MODEL_ON_TRUE: 'model_on_true'>
            >>> tracker.get_provenance('var1', dates[2]) is None
            True
            >>> tracker.get_provenance('var1', dates).tolist()
            ['original', 'model_on_true', nan]
        """
        # Validation de l'initialisation
        if self.provenance_matrix_ is None:
            raise ValueError("Provenance matrix not initialized. Call initialize() first.")

        # Validation de la colonne
        if column not in self.provenance_matrix_.columns:
            raise ValueError(f"Column '{column}' not found in provenance matrix")

        selected = self.provenance_matrix_.loc[index, column]
        # Cellule unique : le membre d'énumération plutôt que la chaîne stockée
        if isinstance(selected, pd.Series):
            return selected
        return None if pd.isna(selected) else ProvenanceType(selected)

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
            >>> data = pd.DataFrame({'var1': [1.0, np.nan, np.nan], 'var2': [1.0, 2.0, 3.0]})
            >>> tracker = ImputationProvenanceTracker().initialize(data)
            >>> tracker.mark_model_imputed('var1', 1)
            >>> tracker.mark_model_imputed('var1', 2, covariate_taint='imputed')
            >>> # Mask of all model-imputed values
            >>> mask = tracker.get_mask([ProvenanceType.MODEL_ON_TRUE, ProvenanceType.MODEL_ON_IMPUTED])
            >>> mask['var1'].tolist()
            [False, True, True]
            >>> # Mask of original values for a specific column
            >>> tracker.get_mask(ProvenanceType.ORIGINAL, column='var1').tolist()
            [True, False, False]
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
            >>> data = pd.DataFrame({'var1': [1.0, np.nan], 'var2': [1.0, 2.0]})
            >>> stats = ImputationProvenanceTracker().initialize(data).compute_statistics()
            >>> print(f"Original values: {stats['overall']['original_pct']:.1f}%")
            Original values: 75.0%
        """
        # Validation de l'initialisation
        if self.provenance_matrix_ is None:
            raise ValueError("Provenance matrix not initialized. Call initialize() first.")

        # Calcul des statistiques de provenance
        self.statistics_ = provenance_statistics(self.provenance_matrix_)
        return self.statistics_

    # Méthode d'extraction de la matrice de provenance
    def get_provenance_matrix(self) -> pd.DataFrame:
        """Get the full provenance matrix.

        Returns:
            Copy of the matrix, of dtype ``PROVENANCE_DTYPE``.

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
            DataFrame with string values ('original', 'model_on_true', etc.),
            ``'not_imputed'`` for a cell not marked, in the ``str`` dtype.

        Raises:
            ValueError: If provenance matrix not initialized.

        Examples:
            >>> data = pd.DataFrame({'var1': [1.0, np.nan]})
            >>> string_matrix = ImputationProvenanceTracker().initialize(data).to_string_matrix()
            >>> string_matrix['var1'].tolist()
            ['original', 'not_imputed']
        """
        if self.provenance_matrix_ is None:
            raise ValueError("Provenance matrix not initialized. Call initialize() first.")

        # Catégories rendues en chaînes, cellules non renseignées nommées
        return self.provenance_matrix_.astype('str').fillna('not_imputed')

    # Méthode d'appariement de deux jeux de données
    def merge(
        self,
        other: 'ImputationProvenanceTracker',
        overwrite: bool = True
    ) -> 'ImputationProvenanceTracker':
        """Merge another tracker's provenance information into this one.

        Useful for combining provenance from multiple imputation stages. The
        merge relies on :meth:`pandas.DataFrame.update`: it is aligned on
        labels and only updates cells ``self`` already carries, it never adds
        a row or a column, and a null cell of ``other`` never erases anything.
        Both matrices must therefore carry the same labels.

        Args:
            other: Another ImputationProvenanceTracker to merge.
            overwrite: Merge strategy:
                - True: filled values of ``other`` overwrite those of ``self``
                - False: values of ``self`` are kept, only its null cells are
                  filled from ``other``

        Returns:
            self: The merged tracker.

        Raises:
            ValueError: If one tracker is not initialized, or if the two
                matrices do not carry the same index and columns (same
                shape is not enough).

        Examples:
            >>> data = pd.DataFrame({'var': [1.0, np.nan, np.nan]})
            >>> tracker1 = ImputationProvenanceTracker().initialize(data)
            >>> tracker2 = ImputationProvenanceTracker().initialize(data)
            >>> tracker2.mark_interpolated('var', [1, 2])
            >>> tracker1.merge(tracker2, overwrite=True).provenance_matrix_['var'].tolist()
            ['original', 'interpolated', 'interpolated']
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
        # Même forme ne suffit pas : "update" apparie par étiquettes et ignorerait
        # sans bruit des étiquettes qui ne se recoupent pas
        if (not self.provenance_matrix_.index.equals(other.provenance_matrix_.index)
                or set(self.provenance_matrix_.columns) != set(other.provenance_matrix_.columns)):
            raise ValueError("Incompatible labels: both matrices must carry the same index and columns")

        # Fusion sur place : jamais d'ajout de ligne ni de colonne
        self.provenance_matrix_.update(other.provenance_matrix_, overwrite=overwrite)

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
