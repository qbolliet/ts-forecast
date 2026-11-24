"""Rolling aggregation step for ``XYPipeline``: ``RollingAggregator`` sums (or
averages...) each column over a calendar window sliding along the grid of each
entity, aggregating the target at the training pass only."""
# Importation des modules
# Modules de base
import warnings
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union, get_args
# Manipulation de données
import numpy as np
import pandas as pd
# Sklearn
from sklearn.utils.validation import check_is_fitted
# Module du package
from ..panel.utils import check_entity_labels, get_entity_codes
from ..utils._constants import CALENDAR_SUBPERIODS, EXACT_CALENDAR_SUBPERIOD_PAIRS, FREQUENCY_LEVEL_NAME
from ..utils.frequency import detect_index_frequency, normalize_frequency
from ..utils.parse import parse_frequency
from .transformers import XYPanelTimeSeriesTransformer

# Type aliases
Aggregation = Literal['sum', 'mean', 'min', 'max', 'first', 'last']
Alignment = Literal['auto', 'past', 'future']
EntityKey = Tuple[Any, ...]

# Valeurs admissibles des littéraux
_AGGREGATIONS: Tuple[str, ...] = get_args(Aggregation)
_ALIGNMENTS: Tuple[str, ...] = get_args(Alignment)

# Clé du dictionnaire par colonne couvrant les colonnes qu'il ne nomme pas
_DEFAULT_KEY = '__default__'

# Alignement déduit de la position des dates de la grille dans leur période
_ALIGNMENT_BY_POSITION: Dict[str, str] = {'E': 'past', 'S': 'future'}


# Fenêtre résolue pour une entité : grille, nombre de pas et sens de la fenêtre
class _EntityWindow(NamedTuple):
    """Window of one entity: its grid, its length in grid steps and its direction."""

    grid_base: str
    grid_position: Optional[str]
    grid_label: str
    length: int
    alignment: str


# Disposition des lignes : dates, entités et colonnes à agréger
class _RowLayout(NamedTuple):
    """Dates, entity codes and columns to aggregate of validated data."""

    dates: pd.DatetimeIndex
    codes: np.ndarray
    entities: List[EntityKey]
    value_columns: List[Any]


# Étape XY d'agrégation glissante ; sorties pandas construites par la classe,
# d'où l'absence d'enveloppe set_output
class RollingAggregator(XYPanelTimeSeriesTransformer, auto_wrap_output_keys=None):
    """Aggregate every column over a calendar window sliding along the grid.

    Each row ``t`` receives the aggregate (sum by default) of its column over
    the ``window`` period that ends at ``t`` or starts at ``t``, entity by
    entity. A monthly grid with ``window='Q'`` thus turns each month into a
    **rolling quarter**: three times as many quarterly rows as a quarterly
    dataset holds. Placed after a ``HighFrequencyImputer`` that brought a
    quarterly target onto a monthly grid, it puts the model back on the scale
    of the observed target.

    **Training and prediction passes**, as for ``HighFrequencyImputer``:

    - ``fit_transform(X, y)`` is the **training pass**: ``X`` **and** ``y``
      are aggregated (``y`` with ``target_agg``);
    - ``transform(X, y=None)`` is the **prediction pass**: ``X`` alone is
      aggregated, and ``y`` — when given, as ``score`` does — comes back
      **unchanged**: the observed target already is a period total.

    ``fit_transform(X, y)`` therefore differs from ``fit(X, y).transform(X,
    y)`` when ``y`` is given. Only an aggregation that rebuilds the observed
    quantity follows this convention: a log or a growth rate of the target
    must transform the test target too (``TransformedTargetRegressor`` around
    the model), never the training target alone.

    **Window direction**: with ``alignment='auto'``, the window follows the
    position of the grid dates in their period. A period-end grid (``ME``)
    gets a **past** window ``[t-2, t]``, a period-start grid (``MS``) a
    **future** window ``[t, t+2]``; on such a grid, the row of a quarter's
    anchor thus sums exactly the three months of that quarter. A grid without
    position (daily, weekly) requires an explicit ``'past'`` or ``'future'``.

    **Complete windows only**: a window missing a value — a NaN, a grid date
    absent from the data, the edge of the data — gives NaN. At the training
    pass, ``DropNaNTarget`` drops those rows; at the prediction pass, the
    first rows of a test set (past window) or its last ones (future window)
    come out NaN unless the test set starts and ends on period bounds.

    The window is counted in **grid steps**, read on the period of each date:
    the grid of each entity is detected at ``fit`` (two dates at least) and
    must be finer than the window with an exact, constant number of
    sub-periods (``Y`` over ``Q``, ``M`` or ``SM``; ``Q`` over ``M`` or
    ``SM``; ``M`` over ``SM``; ``W`` over ``D``), or the same base (``'3M'``
    over a monthly grid: three steps). An entity unknown at ``fit`` takes
    the window shared by every fitted entity when there is one, otherwise the
    window of its own detected grid.

    The time and entity columns (``time_col``, ``panel_cols``) come back
    unchanged; every other column must be numeric and comes back as
    ``float64``. Index, row order and date unit are those of the validated
    input. The aggregated columns replace their source, unless ``suffix`` is
    given: the sources are then kept and the aggregates appended after them.

    Args:
        window: Frequency of the window (``'Q'``, ``'QE'``, ``'Y'``,
            ``'3M'``...). Its base and multiplier count, its position does
            not: the grid gives it.
        agg: Aggregation of the columns of ``X``: one of ``'sum'``,
            ``'mean'``, ``'min'``, ``'max'``, ``'first'``, ``'last'``, or a
            dict ``{column: aggregation}`` with an optional ``'__default__'``
            key (else ``'sum'``). A key naming no column of the data is
            ignored, with a single ``UserWarning`` at ``fit``.
        target_agg: Aggregation of the target at the training pass, applied
            to every column of a multi-output target.
        alignment: ``'auto'`` (from the grid position), ``'past'`` or
            ``'future'``.
        suffix: None (default) to replace each column by its aggregate, or a
            suffix appended to the names of the aggregates, the sources being
            kept.
        time_col: Name of the time column. Default is None (dates in the
            index, last level for a ``MultiIndex``).
        panel_cols: Columns identifying panel entities. Requires
            ``time_col``.
        validate_input: Whether to validate the input data.
        strict_validation: If True, raises on validation failures; if False,
            warns.
        auto_sort: If True, sorts unsorted data by entity and date.
        convert_cols_to_index: If True, moves ``time_col`` and
            ``panel_cols`` into the index.

    Attributes:
        grid_frequencies_ (dict): Detected grid of each entity (``'ME'``,
            ``'MS'``, ``'QE'``...), keyed by entity key (``()`` for a time
            series).
        window_lengths_ (dict): Length of the window of each entity, in grid
            steps.
        alignments_ (dict): Resolved direction of the window of each entity,
            ``'past'`` or ``'future'``.
        aggregations_ (dict): Aggregation of each aggregated column.

    Raises:
        ValueError: At ``fit``, for an invalid parameter, a grid that cannot
            be detected, coarser than the window or without an exact number
            of sub-periods in it, an ``'auto'`` alignment on a grid without
            position, or a stacked frequency level in the index.
        TypeError: At ``fit``, for a non-numeric column to aggregate.

    Examples:
        >>> import pandas as pd
        >>> months = pd.date_range('2024-01-31', periods=6, freq='ME')
        >>> X = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=months)
        >>> y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], index=months)
        >>> step = RollingAggregator(window='Q')
        >>> X_t, y_t = step.fit_transform(X, y)
        >>> X_t['x'].tolist()
        [nan, nan, 6.0, 9.0, 12.0, 15.0]
        >>> y_t.tolist()
        [nan, nan, 60.0, 90.0, 120.0, 150.0]
        >>> X_p, y_p = step.transform(X, y)
        >>> y_p.tolist()
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        >>> starts = pd.date_range('2024-01-01', periods=6, freq='MS')
        >>> RollingAggregator(window='Q').fit_transform(X.set_axis(starts))['x'].tolist()
        [6.0, 9.0, 12.0, 15.0, nan, nan]
    """

    # Initialisation
    def __init__(
        self,
        window: str,
        agg: Union[Aggregation, Dict[Any, Aggregation]] = 'sum',
        target_agg: Aggregation = 'sum',
        alignment: Alignment = 'auto',
        suffix: Optional[str] = None,
        time_col: Optional[str] = None,
        panel_cols: Optional[List[str]] = None,
        validate_input: bool = True,
        strict_validation: bool = True,
        auto_sort: bool = False,
        convert_cols_to_index: bool = False,
    ):
        """Initialize the step.

        Args:
            window: Frequency of the window.
            agg: Aggregation of the columns of ``X``, or a dict per column.
            target_agg: Aggregation of the target at the training pass.
            alignment: ``'auto'``, ``'past'`` or ``'future'``.
            suffix: None to replace the columns, or the suffix of the
                appended aggregates.
            time_col: Name of the time column.
            panel_cols: Columns identifying panel entities.
            validate_input: Whether to validate the input data.
            strict_validation: If True, raises on validation failures.
            auto_sort: If True, sorts unsorted data.
            convert_cols_to_index: If True, moves the time and entity
                columns into the index.

        Examples:
            >>> RollingAggregator(window='Q', agg='mean').get_params()['agg']
            'mean'
        """
        # Initialisation du parent XYPanelTimeSeriesTransformer
        super().__init__(
            time_col=time_col,
            panel_cols=panel_cols,
            validate_input=validate_input,
            strict_validation=strict_validation,
            auto_sort=auto_sort,
            convert_cols_to_index=convert_cols_to_index,
        )
        # Instanciation des attributs
        self.window = window
        self.agg = agg
        self.target_agg = target_agg
        self.alignment = alignment
        self.suffix = suffix

    # Apprentissage : fenêtre de chaque entité et agrégation de chaque colonne
    def _fit(self, X: pd.DataFrame, y: Any = None) -> None:
        """Resolve the window of each entity and the aggregation of each column.

        Args:
            X: Validated features.
            y: Target, unused at this stage.

        Raises:
            ValueError: For an invalid parameter or grid.
            TypeError: For a non-numeric column to aggregate.
        """
        # Vérification des paramètres
        window_base, window_multiplier = self._parse_window()
        self._check_choice('target_agg', self.target_agg, _AGGREGATIONS)
        self._check_choice('alignment', self.alignment, _ALIGNMENTS)
        if self.suffix is not None and (not isinstance(self.suffix, str) or not self.suffix):
            raise ValueError(f"suffix must be None or a non-empty string, got {self.suffix!r}")

        # Disposition des lignes et colonnes à agréger
        layout = self._row_layout(X)
        self._check_numeric(X, layout.value_columns)
        self.aggregations_ = self._resolve_aggregations(layout.value_columns)
        self._check_suffix_collisions(X, layout.value_columns)

        # Fenêtre de chaque entité, lue sur sa grille détectée
        windows: Dict[EntityKey, _EntityWindow] = {}
        for code, entity in enumerate(layout.entities):
            entity_dates = layout.dates[layout.codes == code]
            windows[entity] = self._entity_window(entity, entity_dates, window_base, window_multiplier)
        self._windows = windows
        self._window_spec = (window_base, window_multiplier)

        # Attributs publics : vues des fenêtres résolues
        self.grid_frequencies_ = {entity: window.grid_label for entity, window in windows.items()}
        self.window_lengths_ = {entity: window.length for entity, window in windows.items()}
        self.alignments_ = {entity: window.alignment for entity, window in windows.items()}

    # Passe d'entraînement : X et y agrégés
    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> Any:
        """Fit, then aggregate ``X`` and, when given, ``y`` (training pass).

        Args:
            X: Features.
            y: Target (Series, DataFrame or array), optional. A pandas target
                must carry the rows of ``X`` (same labels, any order); an
                array is matched by position.

        Returns:
            ``X`` aggregated if ``y`` is None, the pair ``(X_t, y_t)`` with
            ``y`` aggregated by ``target_agg`` otherwise.

        Raises:
            ValueError: If ``y`` does not match the rows of ``X``.

        Examples:
            >>> import pandas as pd
            >>> months = pd.date_range('2024-01-31', periods=4, freq='ME')
            >>> X = pd.DataFrame({'x': [1.0, 1.0, 1.0, 1.0]}, index=months)
            >>> y = pd.Series([1.0, 2.0, 3.0, 4.0], index=months)
            >>> RollingAggregator(window='Q').fit_transform(X, y)[1].tolist()
            [nan, nan, 6.0, 9.0]
        """
        # Entrainement
        self.fit(X, y)
        # Préparation
        X_valid = self._prepare(X)
        # Transformation
        X_out = self._aggregate_features(X_valid)
        if y is None:
            return X_out
        return X_out, self._aggregate_target(X_valid, y)

    # Passe de prédiction : X seul agrégé, y rendu tel quel
    def _transform(self, X: pd.DataFrame, y: Any = None) -> Any:
        """Aggregate ``X``; give ``y`` back unchanged (prediction pass).

        Args:
            X: Validated features.
            y: Observed target, optional, returned as is.

        Returns:
            ``X`` aggregated if ``y`` is None, the pair ``(X_t, y)`` otherwise.
        """
        X_out = self._aggregate_features(X)
        return X_out if y is None else (X_out, y)

    # Noms des colonnes de sortie
    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        """Give the names of the output columns.

        Args:
            input_features: Ignored, the names are those seen at ``fit``.

        Returns:
            Object array of the output names: the input names, followed by
            the suffixed aggregates when ``suffix`` is given.

        Examples:
            >>> import pandas as pd
            >>> X = pd.DataFrame({'x': [1.0, 2.0]}, index=pd.date_range('2024-01-31', periods=2, freq='ME'))
            >>> RollingAggregator(window='Q', suffix='_q').fit(X).get_feature_names_out().tolist()
            ['x', 'x_q']
        """
        # Vérification que le transformer est entrainé
        check_is_fitted(self)
        # Extraction des noms en entrée
        names = list(self.feature_names_in_)
        # Ajout d'un suffixe si spécifié
        if self.suffix is not None:
            names += [f'{column}{self.suffix}' for column in self.aggregations_]
        return np.asarray(names, dtype=object)

    # Sorties construites par la classe : configuration de sortie sans effet
    def set_output(self, *, transform: Any = None) -> 'RollingAggregator':
        """Accept an output configuration and ignore it.

        The step always returns pandas objects; it accepts ``set_output`` so
        that ``XYPipeline.set_output`` can configure the other steps.

        Args:
            transform: Output container requested, ignored.

        Returns:
            self

        Examples:
            >>> step = RollingAggregator(window='Q')
            >>> step.set_output(transform='pandas') is step
            True
        """
        return self

    # ------------------------------------------------------------------
    # Paramètres
    # ------------------------------------------------------------------

    # Base et multiplicateur de la fenêtre
    def _parse_window(self) -> Tuple[str, int]:
        """Read the base frequency and the multiplier of ``window``.

        Returns:
            Tuple ``(base, multiplier)``, e.g. ``('Q', 1)`` or ``('M', 3)``.

        Raises:
            ValueError: If ``window`` is not a supported frequency string.
        """
        if not isinstance(self.window, str):
            raise ValueError(f"window must be a frequency string such as 'Q', got {self.window!r}")
        try:
            # Parsing et normalisation de la fréquence
            parsed = parse_frequency(self.window)
            return normalize_frequency(parsed.freq), parsed.multiplier
        except ValueError as error:
            raise ValueError(f"window must be a frequency string such as 'Q', got {self.window!r}: {error}") from error

    # Contrôle d'un paramètre à valeurs fermées
    @staticmethod
    def _check_choice(name: str, value: Any, choices: Tuple[str, ...]) -> None:
        """Raise unless ``value`` is one of ``choices``."""
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f"{name} must be one of {choices}, got {value!r}")

    # Agrégation de chaque colonne, depuis la chaîne ou le dictionnaire 'agg'
    def _resolve_aggregations(self, columns: List[Any]) -> Dict[Any, str]:
        """Resolve the aggregation of each column to aggregate.

        Args:
            columns: Columns to aggregate.

        Returns:
            Dict ``{column: aggregation}``, in column order.

        Raises:
            ValueError: If an aggregation is not supported.
        """
        # Agrégation unique
        if not isinstance(self.agg, dict):
            self._check_choice('agg', self.agg, _AGGREGATIONS)
            return {column: self.agg for column in columns}

        # Dictionnaire par colonne : valeurs admissibles, clés sans colonne signalées une fois
        for key, value in self.agg.items():
            self._check_choice(f'agg[{key!r}]', value, _AGGREGATIONS)
        unknown = [key for key in self.agg if key != _DEFAULT_KEY and key not in columns]
        if unknown:
            warnings.warn(
                f"agg names columns absent from the data, ignored: {unknown}",
                UserWarning,
                stacklevel=4,
            )
        default = self.agg.get(_DEFAULT_KEY, 'sum')
        return {column: self.agg.get(column, default) for column in columns}

    # ------------------------------------------------------------------
    # Structure des données
    # ------------------------------------------------------------------

    # Validation d'une entrée hors fit, comme au transform
    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate ``X`` as ``transform`` does, and check its columns."""
        # Vérification du type d'entrée
        self._check_input_type(X)
        # Validation
        if self.validate_input:
            X = self._validate_input(X)
        self._check_feature_names(X)
        return X

    # Dates, entités et colonnes à agréger d'une donnée validée
    def _row_layout(self, X: pd.DataFrame) -> _RowLayout:
        """Read the dates, the entity of each row and the columns to aggregate.

        Args:
            X: Validated data, dates and entities in the index or in the
                ``time_col`` / ``panel_cols`` columns.

        Returns:
            The row layout.

        Raises:
            ValueError: If the index carries a stacked frequency level, or an
                entity label is missing.
        """
        # Niveau de fréquence d'une sortie empilée : plusieurs grilles par entité
        if FREQUENCY_LEVEL_NAME in X.index.names:
            raise ValueError(
                f"The index carries a '{FREQUENCY_LEVEL_NAME}' level, as the stacked output of "
                "HighFrequencyImputer(keep_lower_frequencies=True): the step needs one grid per "
                "entity, use keep_lower_frequencies=False upstream."
            )

        # Dates et entités portées par des colonnes
        if self.time_col is not None and self.time_col in X.columns:
            key_columns = [self.time_col] + list(self.panel_cols or [])
            dates = _as_datetime_index(X[self.time_col])
            if self.panel_cols:
                check_entity_labels(X, columns=self.panel_cols)
                keys = pd.MultiIndex.from_arrays([X[column] for column in self.panel_cols] + [dates])
                codes, entities = get_entity_codes(keys)
            else:
                codes, entities = get_entity_codes(dates)
            value_columns = [column for column in X.columns if column not in key_columns]
            return _RowLayout(dates, codes, entities, value_columns)

        # Dates et entités portées par l'index
        if isinstance(X.index, pd.MultiIndex):
            check_entity_labels(X.index)
            dates = _as_datetime_index(X.index.get_level_values(-1))
        else:
            dates = _as_datetime_index(X.index)
        codes, entities = get_entity_codes(X.index)
        return _RowLayout(dates, codes, entities, list(X.columns))

    # Colonnes à agréger numériques
    @staticmethod
    def _check_numeric(X: pd.DataFrame, columns: List[Any]) -> None:
        """Raise a ``TypeError`` naming the non-numeric columns to aggregate."""
        # Extraction des colonnes non numriques
        non_numeric = [
            column for column in columns
            if not (pd.api.types.is_numeric_dtype(X[column]) or pd.api.types.is_bool_dtype(X[column]))
        ]
        if non_numeric:
            raise TypeError(f"RollingAggregator aggregates numeric columns only, got non-numeric {non_numeric}")

    # Noms suffixés distincts des colonnes existantes
    def _check_suffix_collisions(self, X: pd.DataFrame, columns: List[Any]) -> None:
        """Raise when a suffixed name already names a column of ``X``."""
        if self.suffix is None:
            return
        clashes = [f'{column}{self.suffix}' for column in columns if f'{column}{self.suffix}' in X.columns]
        if clashes:
            raise ValueError(f"suffix {self.suffix!r} gives names already in the data: {clashes}")

    # ------------------------------------------------------------------
    # Fenêtres
    # ------------------------------------------------------------------

    # Fenêtre d'une entité, lue sur sa grille détectée
    def _entity_window(
        self,
        entity: EntityKey,
        dates: pd.DatetimeIndex,
        window_base: str,
        window_multiplier: int,
    ) -> _EntityWindow:
        """Detect the grid of an entity and resolve its window.

        Args:
            entity: Entity key, for the error messages.
            dates: Dates of the entity.
            window_base: Base frequency of the window.
            window_multiplier: Multiplier of the window.

        Returns:
            The resolved window.

        Raises:
            ValueError: If the grid cannot be detected, has a multiplier, or
                does not fit in the window, or if ``'auto'`` meets a grid
                without position.
        """
        # Nom du champ
        where = f"entity {entity!r}" if entity else "the time series"

        # Détection de la grille
        try:
            grid = detect_index_frequency(dates.unique().sort_values(), return_format='components')
        except ValueError:
            grid = None
        if grid is None or grid.freq is None:
            raise ValueError(
                f"The grid of {where} cannot be detected (at least two regularly spaced dates are needed)"
            )
        if grid.multiplier != 1:
            raise ValueError(f"The grid of {where} has a multiplier ({grid.multiplier}{grid.freq}), not supported")
        grid_base = grid.freq

        # Nombre de pas de grille dans la fenêtre
        if window_base == grid_base:
            length = window_multiplier
        elif (window_base, grid_base) in EXACT_CALENDAR_SUBPERIOD_PAIRS:
            length = CALENDAR_SUBPERIODS[(window_base, grid_base)] * window_multiplier
        else:
            raise ValueError(
                f"window={self.window!r} does not hold an exact number of periods of the grid of {where} "
                f"('{grid_base}'): the grid must be the same frequency or nest exactly in the window "
                f"(pairs {sorted(EXACT_CALENDAR_SUBPERIOD_PAIRS)})"
            )

        # Sens de la fenêtre
        position = grid.position
        if self.alignment != 'auto':
            alignment = self.alignment
        elif position in _ALIGNMENT_BY_POSITION:
            alignment = _ALIGNMENT_BY_POSITION[position]
        else:
            raise ValueError(
                f"alignment='auto' needs a period-start or period-end grid, the grid of {where} "
                f"('{grid_base}') has no position: pass alignment='past' or 'future'"
            )

        label = f"{grid_base}{position or ''}"
        return _EntityWindow(grid_base, position, label, length, alignment)

    # Fenêtre d'une entité au transform : figée au fit, ou résolue pour une entité nouvelle
    def _window_for(self, entity: EntityKey, dates: pd.DatetimeIndex) -> _EntityWindow:
        """Give the window of an entity at a transform pass.

        Args:
            entity: Entity key.
            dates: Dates of the entity in the data of the pass.

        Returns:
            The window frozen at ``fit``; for an entity unknown at ``fit``,
            the window shared by every fitted entity, else the window of its
            own detected grid.
        """
        # Cas où l'entité est présente dans la fenêtre
        if entity in self._windows:
            return self._windows[entity]
        # Extraction des valeurs
        shared = set(self._windows.values())
        if len(shared) == 1:
            return next(iter(shared))
        return self._entity_window(entity, dates, *self._window_spec)

    # ------------------------------------------------------------------
    # Agrégation
    # ------------------------------------------------------------------

    # Agrégation des colonnes de X
    def _aggregate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aggregate the columns of validated features.

        Args:
            X: Validated features.

        Returns:
            Copy of ``X`` whose columns to aggregate are replaced by their
            aggregates, or followed by them under ``suffix``.
        """
        # Extraction des colonnes et des valeurs
        layout = self._row_layout(X)
        columns = list(self.aggregations_)
        values = X[columns].to_numpy(dtype=float, na_value=np.nan)
        # Aggrégation
        aggregated = self._rolling_values(values, [self.aggregations_[column] for column in columns], layout)

        # Assemblage de la sortie
        output = X.copy()
        if self.suffix is None:
            for position, column in enumerate(columns):
                output[column] = aggregated[:, position]
            return output
        appended = pd.DataFrame(
            aggregated,
            index=X.index,
            columns=pd.Index([f'{column}{self.suffix}' for column in columns]),
        )
        return pd.concat([output, appended], axis=1, sort=False)

    # Agrégation de la cible de la passe d'entraînement
    def _aggregate_target(self, X: pd.DataFrame, y: Any) -> Any:
        """Aggregate the target on the rows of validated features.

        Args:
            X: Validated features.
            y: Target (Series, DataFrame or array).

        Returns:
            The target aggregated by ``target_agg``, in its own type: a pandas
            target on the rows of ``X``, an array by position.

        Raises:
            ValueError: If ``y`` does not match the rows of ``X``.
        """
        # Extraction des éléments d'intérêt
        layout = self._row_layout(X)

        # Alignement de la cible sur les lignes de X
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = self._align_target(X, y)
            frame = y.to_frame() if isinstance(y, pd.Series) else y
            values = frame.to_numpy(dtype=float, na_value=np.nan)
        else:
            values = np.asarray(y, dtype=float)
            if len(values) != len(X):
                raise ValueError(f"X and y must have the same length, got {len(X)} and {len(values)}")
            values = values.reshape(len(values), -1)

        # Agrégation
        aggregated = self._rolling_values(values, [self.target_agg] * values.shape[1], layout)

        # Restitution dans le type de la cible
        if isinstance(y, pd.Series):
            return pd.Series(aggregated[:, 0], index=y.index, name=y.name, dtype='float64')
        if isinstance(y, pd.DataFrame):
            return pd.DataFrame(aggregated, index=y.index, columns=y.columns, dtype='float64')
        return aggregated.reshape(np.shape(y))

    # Cible pandas remise dans l'ordre des lignes de X
    @staticmethod
    def _align_target(X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Put a pandas target on the rows of ``X``.

        Raises:
            ValueError: If ``y`` does not carry exactly the row labels of ``X``.
        """
        # Alignement des index
        if y.index.equals(X.index):
            return y
        if (
            len(y) == len(X)
            and y.index.is_unique
            and X.index.is_unique
            and len(y.index.difference(X.index)) == 0
        ):
            return y.reindex(X.index)
        raise ValueError("y must carry the rows of X (same index labels, in any order)")

    # Calcul glissant de chaque colonne, entité par entité
    def _rolling_values(self, values: np.ndarray, aggregations: List[str], layout: _RowLayout) -> np.ndarray:
        """Aggregate each column of ``values`` over the window of each entity.

        The dates of an entity are read as consecutive period numbers of its
        grid: a grid date absent from the data leaves a hole, so a window
        always spans the same calendar periods.

        Args:
            values: Values, one row per row of the data, one column per
                column to aggregate.
            aggregations: Aggregation of each column.
            layout: Row layout of the data.

        Returns:
            Float array of the aggregates, shaped like ``values``.

        Raises:
            ValueError: If two rows of an entity fall in the same grid period.
        """
        # Initialisation du résultat
        result = np.full(values.shape, np.nan)
        if values.size == 0:
            return result

        # Parcours des entités
        for code, entity in enumerate(layout.entities):
            # Extraction des fenêtres
            rows = np.flatnonzero(layout.codes == code)
            if len(rows) == 0:
                continue
            window = self._window_for(entity, layout.dates[rows])
            ordinals = _period_ordinals(layout.dates[rows], window)
            if len(np.unique(ordinals)) != len(ordinals):
                where = f"entity {entity!r}" if entity else "the time series"
                raise ValueError(f"Several rows of {where} fall in the same '{window.grid_label}' grid period")

            # Axe dense des périodes de l'entité, trous compris
            dense = pd.DataFrame(values[rows], index=ordinals).reindex(
                np.arange(ordinals.min(), ordinals.max() + 1)
            )
            rolled = pd.DataFrame(index=dense.index, columns=dense.columns, dtype=float)
            for aggregation in set(aggregations):
                selected = [position for position, name in enumerate(aggregations) if name == aggregation]
                rolling = dense[selected].rolling(window.length, min_periods=window.length)
                rolled[selected] = getattr(rolling, aggregation)()

            # Fenêtre future : agrégat de [t, t + k - 1] porté par t
            if window.alignment == 'future':
                rolled = rolled.shift(-(window.length - 1))
            result[rows] = rolled.loc[ordinals].to_numpy(dtype=float)
        return result


# Conversion des étiquettes temporelles en DatetimeIndex
def _as_datetime_index(labels: Any) -> pd.DatetimeIndex:
    """Read time labels as a ``DatetimeIndex``, a period as its first instant."""
    if isinstance(labels, (pd.PeriodIndex, pd.Series)) and isinstance(getattr(labels, 'dtype', None), pd.PeriodDtype):
        return pd.PeriodIndex(labels).to_timestamp()
    return pd.DatetimeIndex(labels)


# Numéro de période de chaque date sur la grille d'une entité
def _period_ordinals(dates: pd.DatetimeIndex, window: _EntityWindow) -> np.ndarray:
    """Number each date by its period on the grid, consecutive periods differing by one.

    A semi-monthly grid has no pandas period: its two halves of a month are
    numbered from the day, the 15th belonging to the second half of a
    period-start grid (1st and 15th) and to the first half of a period-end
    grid (15th and last day).

    Args:
        dates: Dates of an entity.
        window: Window of the entity, carrying its grid.

    Returns:
        Integer array of period numbers.
    """
    if window.grid_base == 'SM':
        months = dates.to_period('M').asi8
        days = np.asarray(dates.day)
        second_half = days >= 15 if window.grid_position == 'S' else days > 15
        return months * 2 + second_half.astype(np.int64)
    return np.asarray(dates.to_period(window.grid_base).asi8, dtype=np.int64)
