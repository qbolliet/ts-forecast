"""Shared toolkit of the third-party adapters of ``tsforecast.adapters``.

Provides ``import_optional`` (deferred import of an optional package with an
installation hint), ``TemporalLayout`` (analysis of a time series or panel index:
entities, integer entity codes, dates, grid frequency), ``steps_between`` (grid
steps between two dates), ``restore_row_order`` (realignment of a third-party
output on the rows of ``X``) and ``check_adapter_input`` (common validation of
the inputs of ``fit`` and ``predict``).
"""
# Importation des modules
# Modules de base
import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Hashable, Optional, Tuple, Union

# Modules de manipulation de données
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data

# tsforecast
from tsforecast.panel.utils import check_entity_labels, get_entity_codes
from tsforecast.utils.frequency import detect_index_frequency, to_pandas_freq

# Nombre maximal d'étiquettes citées dans un message d'erreur
_MAX_LABELS_IN_MESSAGE = 5


# Fonction d'import différé d'un paquet optionnel
def import_optional(module: str, extra: str) -> ModuleType:
    """Import an optional third-party module, with an installation hint if it is missing.

    Only the absence of ``module`` itself (or of one of its parent packages)
    is turned into an installation hint; any other import error (missing
    transitive dependency, broken installation, error raised by the package)
    propagates unchanged.

    Args:
        module: Dotted name of the module to import (``'darts'``,
            ``'sktime.forecasting.base'``).
        extra: Name of the ``ts-forecast`` extra that installs it
            (``'adapters-darts'``).

    Returns:
        The imported module.

    Raises:
        ImportError: If ``module`` or one of its parent packages is not
            installed; the message names the package and the extra.

    Examples:
        >>> import_optional('json', 'adapters-example').dumps([1])
        '[1]'
        >>> try:
        ...     import_optional('tsforecast_missing_package.sub', 'adapters-example')
        ... except ImportError as error:
        ...     print(error)
        Module 'tsforecast_missing_package.sub' requires the 'tsforecast_missing_package' package. Install it with: pip install 'ts-forecast[adapters-example]'
    """
    # Tentative d'import
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        missing = error.name
        # Absence du module demandé ou d'un paquet parent : message d'installation
        if missing is not None and (module == missing or module.startswith(missing + ".")):
            package = module.split(".")[0]
            raise ImportError(
                f"Module '{module}' requires the '{package}' package. "
                f"Install it with: pip install 'ts-forecast[{extra}]'"
            ) from error
        # Toute autre absence (dépendance du paquet tiers) remonte telle quelle
        raise


# Fonction auxiliaire de description courte d'étiquettes
def _preview(labels: Any) -> str:
    """Render the first labels of a sequence for an error message.

    Args:
        labels: Labels to render, in any iterable.

    Returns:
        The ``repr`` of the first labels, comma-separated, followed by
        ``", ..."`` when some labels are left out.
    """
    labels = list(labels)
    shown = ", ".join(repr(label) for label in labels[:_MAX_LABELS_IN_MESSAGE])
    return shown + (", ..." if len(labels) > _MAX_LABELS_IN_MESSAGE else "")


# Fonction auxiliaire de détection d'une fréquence commune aux entités
def _detect_common_frequency(dates: pd.DatetimeIndex, codes: np.ndarray, n_entities: int) -> Optional[str]:
    """Detect the grid frequency shared by the entities, ignoring single-date entities.

    Args:
        dates: Date of each row.
        codes: Entity code of each row, aligned with ``dates``.
        n_entities: Number of distinct entity codes.

    Returns:
        The full pandas alias of the common frequency, or None when no entity
        has a detectable frequency.

    Raises:
        ValueError: If two entities are on different grids.
    """
    # Regroupement des lignes par entité par un tri stable des codes
    order = np.argsort(codes, kind="stable")
    bounds = np.cumsum(np.bincount(codes, minlength=n_entities))[:-1]
    frequencies = {}
    for code, rows in enumerate(np.split(order, bounds)):
        entity_dates = dates[rows].unique()
        # Entité à une seule date : aucune fréquence lisible
        if len(entity_dates) < 2:
            continue
        frequency = detect_index_frequency(entity_dates, return_format="full")
        if frequency is not None:
            frequencies.setdefault(frequency, code)
    if len(frequencies) > 1:
        raise ValueError(
            f"The entities are not on a common time grid: detected frequencies {sorted(frequencies)}"
        )
    return next(iter(frequencies), None)


# Classe de gestion de données temporelles et de panel
@dataclass(frozen=True, eq=False)
class TemporalLayout:
    """Temporal and entity structure of the index of a time series or panel.

    A time series is a single degenerate entity ``()``; a panel has a
    ``MultiIndex`` whose last level holds the dates and whose other levels
    identify the entity. Entities are identified by integer codes, never by a
    tuple per row.

    Attributes:
        entity_names: Names of the entity levels (``()`` for a time series).
        time_name: Name of the time level.
        entities: Distinct entity keys (tuples), in the order of the codes,
            i.e. of first appearance.
        codes: Entity code of each row (read-only ``np.intp`` array).
        dates: Date of each row, in the unit of the input.
        freq: Pandas alias of the grid frequency (``'MS'``, ``'QE-DEC'``,
            ``'2W-SUN'``), or None when it is not known.

    Examples:
        >>> import pandas as pd
        >>> index = pd.MultiIndex.from_product(
        ...     [['FR', 'DE'], pd.date_range('2024-01-01', periods=3, freq='MS')],
        ...     names=['country', 'date'])
        >>> layout = TemporalLayout.from_index(index)
        >>> layout.is_panel, layout.entities, layout.freq
        (True, (('FR',), ('DE',)), 'MS')
        >>> layout.codes.tolist()
        [0, 0, 0, 1, 1, 1]
        >>> TemporalLayout.from_index(pd.date_range('2024-01-01', periods=3, freq='D')).entities
        ((),)
    """

    entity_names: Tuple[Hashable, ...]
    time_name: Hashable
    entities: Tuple[tuple, ...]
    codes: np.ndarray
    dates: pd.DatetimeIndex
    freq: Optional[str]

    @property
    def is_panel(self) -> bool:
        """Whether the index has entity levels."""
        return len(self.entity_names) > 0

    @property
    def n_entities(self) -> int:
        """Number of distinct entities (1 for a time series)."""
        return len(self.entities)

    @property
    def unit(self) -> str:
        """Resolution of the dates (``'s'``, ``'ms'``, ``'us'`` or ``'ns'``)."""
        return self.dates.unit

    # Construction depuis un index
    @classmethod
    def from_index(
        cls,
        index: pd.Index,
        *,
        freq: Optional[str] = None,
        detect_freq: bool = True,
    ) -> "TemporalLayout":
        """Analyse a ``DatetimeIndex`` or an ``(entity..., date)`` ``MultiIndex``.

        Args:
            index: Index of the data. Rows may come in any order and entities
                may cover different dates.
            freq: Grid frequency to record, in any notation accepted by
                :func:`tsforecast.utils.frequency.to_pandas_freq`; when given,
                no detection takes place.
            detect_freq: When ``freq`` is None, whether to detect the
                frequency from the dates. The entities with at least two dates
                must agree; entities with a single date are ignored.

        Returns:
            The layout of the index.

        Raises:
            TypeError: If the dates are not ``datetime64`` (``Period`` labels
                included: convert them with ``to_timestamp()``), or if the
                ``MultiIndex`` has a single level.
            ValueError: If a date or an entity label is missing, if an
                ``(entity, date)`` pair is duplicated, or if the entities are
                on different grids.

        Examples:
            >>> import pandas as pd
            >>> dates = pd.DatetimeIndex(['2024-03-31', '2023-12-31', '2024-06-30'])
            >>> TemporalLayout.from_index(dates).freq
            'QE-DEC'
            >>> TemporalLayout.from_index(dates[:1], freq='Q').freq
            'QE'
        """
        # Séparation des niveaux d'entité et du niveau temporel
        if isinstance(index, pd.MultiIndex):
            if index.nlevels < 2:
                raise TypeError("A panel MultiIndex needs entity levels and a last date level")
            time_values = index.get_level_values(-1)
            entity_names = tuple(index.names[:-1])
            time_name = index.names[-1]
        else:
            time_values = index
            entity_names = ()
            time_name = index.name

        # Vérification du type des dates
        if isinstance(time_values, pd.PeriodIndex):
            raise TypeError(
                "Period labels are not supported: convert them to dates with to_timestamp()"
            )
        if not isinstance(time_values, pd.DatetimeIndex):
            raise TypeError(
                f"The time labels must be datetime64, got dtype {time_values.dtype}"
            )
        if time_values.hasnans:
            raise ValueError("The time labels contain missing dates (NaT)")

        # Étiquettes d'entité manquantes refusées, puis codes entiers d'entité
        if entity_names:
            check_entity_labels(index)
        codes, entities = get_entity_codes(index)
        codes = np.array(codes, dtype=np.intp)
        codes.setflags(write=False)

        # Couples (entité, date) uniques : condition du réalignement des sorties
        if index.has_duplicates:
            duplicated = index[index.duplicated()]
            raise ValueError(f"Duplicated (entity, date) rows: {_preview(duplicated.unique())}")

        dates = pd.DatetimeIndex(time_values, name=time_name)
        # Fréquence imposée, détectée, ou inconnue
        if freq is not None:
            frequency = to_pandas_freq(freq)
        elif detect_freq:
            frequency = _detect_common_frequency(dates, codes, len(entities))
        else:
            frequency = None
        return cls(
            entity_names=entity_names,
            time_name=time_name,
            entities=tuple(entities),
            codes=codes,
            dates=dates,
            freq=frequency,
        )

    # Dernière date de chaque entité
    def last_dates(self) -> pd.DatetimeIndex:
        """Return the last date of each entity, in the order of the codes.

        Returns:
            One date per entity, in the unit of the input.

        Examples:
            >>> import pandas as pd
            >>> index = pd.MultiIndex.from_arrays(
            ...     [['A', 'B', 'A'], pd.to_datetime(['2024-01-01', '2024-03-01', '2024-02-01'])])
            >>> TemporalLayout.from_index(index).last_dates().strftime('%Y-%m').tolist()
            ['2024-02', '2024-03']
        """
        last = pd.Series(self.dates).groupby(self.codes, sort=True).max()
        return pd.DatetimeIndex(last, name=self.time_name)

    # Correspondance des entités d'un autre layout
    def match_entities(self, other: "TemporalLayout") -> np.ndarray:
        """Locate the entity of each row of another layout among the entities of this one.

        Typical use: ``self`` describes the training data and ``other`` the
        data to predict.

        Args:
            other: Layout of the rows to locate.

        Returns:
            For each row of ``other``, the code of its entity in ``self``, or
            ``-1`` when the entity is unknown (``np.intp`` array).

        Raises:
            ValueError: If the two layouts do not have the same number of
                entity levels.

        Examples:
            >>> import pandas as pd
            >>> dates = pd.date_range('2024-01-01', periods=2, freq='MS')
            >>> train = TemporalLayout.from_index(pd.MultiIndex.from_product([['A', 'B'], dates]))
            >>> test = TemporalLayout.from_index(pd.MultiIndex.from_product([['B', 'C'], dates[-1:]]))
            >>> train.match_entities(test).tolist()
            [1, -1]
        """
        # Vérification de la correspondance des entités
        if len(other.entity_names) != len(self.entity_names):
            raise ValueError(
                f"Expected {len(self.entity_names)} entity level(s), got {len(other.entity_names)}"
            )
        # Correspondance calculée une fois par entité distincte, puis diffusée par les codes
        lookup = {entity: code for code, entity in enumerate(self.entities)}
        per_entity = np.array([lookup.get(entity, -1) for entity in other.entities], dtype=np.intp)
        return per_entity[other.codes]


# Fonction de calcul du nombre de pas de grille entre deux dates
def steps_between(
    start: Union[pd.DatetimeIndex, pd.Series, np.ndarray],
    end: Union[pd.DatetimeIndex, pd.Series, np.ndarray],
    freq: str,
) -> np.ndarray:
    """Count the grid steps from each ``start`` date to the matching ``end`` date.

    Both dates must lie on the grid of ``freq`` that contains the earliest of
    all the dates (multipliers, anchors and business days included). The count
    is ``end - start`` in steps: zero or negative when ``end`` does not come
    after ``start``.

    Args:
        start: Start dates (for example the last training date of the entity
            of each row).
        end: End dates, as many as ``start``.
        freq: Grid frequency, in any notation accepted by
            :func:`tsforecast.utils.frequency.to_pandas_freq`.

    Returns:
        ``np.int64`` array of step counts, one per pair.

    Raises:
        ValueError: If the lengths differ, if a date is missing, or if a date
            is not on the grid.

    Examples:
        >>> import pandas as pd
        >>> start = pd.to_datetime(['2024-01-01', '2024-01-01'])
        >>> end = pd.to_datetime(['2024-04-01', '2023-11-01'])
        >>> steps_between(start, end, 'MS').tolist()
        [3, -2]
        >>> steps_between(start[:1], pd.to_datetime(['2024-05-01']), '2MS').tolist()
        [2]
    """
    # Conversion en datetime
    start = pd.DatetimeIndex(start)
    end = pd.DatetimeIndex(end)
    # Vérification que les débuts et les fins sont de même longueur
    if len(start) != len(end):
        raise ValueError(f"start and end must have the same length, got {len(start)} and {len(end)}")
    # Vérification que les débuts et les fins sont non vides
    if len(start) == 0:
        return np.empty(0, dtype=np.int64)
    # Vérification que les débuts et les fins sont spécifiés
    if start.hasnans or end.hasnans:
        raise ValueError("start and end must not contain missing dates (NaT)")

    # Grille unique couvrant toutes les dates, ancrée sur la première date
    offset = to_pandas_freq(freq)
    first = min(start.min(), end.min())
    last = max(start.max(), end.max())
    grid = pd.date_range(first, last, freq=offset)

    # Positions sur la grille ; une date absente n'est pas sur la grille
    positions = []
    # Parcours des dates
    for dates in (start, end):
        # Extraction de l'index de la date
        located = grid.get_indexer(dates)
        off_grid = located < 0
        if off_grid.any():
            raise ValueError(
                f"Dates not on the '{offset}' grid starting at {first}: "
                f"{_preview(dates[off_grid].unique())}"
            )
        positions.append(located)
    return (positions[1] - positions[0]).astype(np.int64)


# Fonction de réalignement d'une sortie sur l'ordre des lignes cibles
def restore_row_order(
    values: Union[np.ndarray, pd.Series, pd.DataFrame],
    source_index: pd.Index,
    target_index: pd.Index,
) -> np.ndarray:
    """Reorder values indexed by ``source_index`` along the rows of ``target_index``.

    Dates are compared as instants, whatever their unit.

    Args:
        values: Values in the order of ``source_index`` (one row per label),
            of shape ``(n,)`` or ``(n, k)``.
        source_index: Labels of the rows of ``values``, unique.
        target_index: Labels in the wanted order (typically ``X.index``).

    Returns:
        A new array holding the row of ``values`` of each label of
        ``target_index``.

    Raises:
        ValueError: If ``values`` and ``source_index`` have different lengths,
            if ``source_index`` has duplicates, or if a label of
            ``target_index`` is missing from ``source_index``.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-01', periods=3, freq='MS')
        >>> restore_row_order([1.0, 2.0, 3.0], dates, dates[::-1]).tolist()
        [3.0, 2.0, 1.0]
    """
    # Conversion en numpy array
    values = np.asarray(values)
    # Vérification que les valeurs sont de même longueur que l'index
    if len(values) != len(source_index):
        raise ValueError(
            f"values has {len(values)} rows but source_index has {len(source_index)} labels"
        )
    # Vérification de l'unicité de l'index
    if source_index.has_duplicates:
        raise ValueError(
            f"source_index has duplicated labels: {_preview(source_index[source_index.duplicated()])}"
        )
    # Extraction des positions
    positions = source_index.get_indexer(target_index)
    missing = positions < 0
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} row(s) to restore are missing from the output: "
            f"{_preview(target_index[missing])}"
        )
    # Indexation par tableau d'entiers : copie inscriptible
    return values[positions]


# Fonction de validation commune des entrées d'un adapter
def check_adapter_input(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Any = None,
    *,
    reset: bool,
    freq: Optional[str] = None,
    detect_freq: bool = True,
) -> Tuple[pd.DataFrame, Optional[Union[pd.Series, pd.DataFrame]], TemporalLayout]:
    """Validate the inputs of an adapter's ``fit`` or ``predict`` and analyse their index.

    With ``reset=True`` (``fit``), sets ``n_features_in_`` and
    ``feature_names_in_`` on ``estimator``; with ``reset=False``
    (``predict``), checks that the columns match those seen at fit. ``X`` is
    neither copied nor modified.

    Args:
        estimator: Adapter whose fitted feature attributes are set or checked.
        X: Features, indexed by a ``DatetimeIndex`` or an ``(entity..., date)``
            ``MultiIndex``.
        y: Target: ``Series`` / ``DataFrame`` indexed like ``X``, array-like
            with one row per row of ``X`` (placed on ``X.index``), or None.
        reset: True at fit, False at predict.
        freq: Grid frequency passed to :meth:`TemporalLayout.from_index`
            (typically the frequency recorded at fit, when predicting).
        detect_freq: Whether to detect the frequency when ``freq`` is None.

    Returns:
        ``(X, y, layout)``: ``X`` unchanged, ``y`` as a pandas object indexed
        like ``X`` (or None) and the layout of ``X.index``.

    Raises:
        TypeError: If ``X`` is not a ``DataFrame``, or its dates are not
            ``datetime64``.
        ValueError: If ``y`` is not aligned with ``X``, if the columns differ
            from those seen at fit, or if the index is invalid (see
            :meth:`TemporalLayout.from_index`).

    Examples:
        >>> import pandas as pd
        >>> from sklearn.base import BaseEstimator
        >>> X = pd.DataFrame({'x': [1.0, 2.0]}, index=pd.date_range('2024-01-01', periods=2, freq='MS'))
        >>> estimator = BaseEstimator()
        >>> _, y, layout = check_adapter_input(estimator, X, [3.0, 4.0], reset=True)
        >>> estimator.n_features_in_, estimator.feature_names_in_.tolist(), layout.freq
        (1, ['x'], 'MS')
        >>> y.index.equals(X.index)
        True
    """
    # Vérification que l'entrée est un DataFrame
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}")
    # Attributs de colonnes : posés au fit, contrôlés au predict
    validate_data(estimator, X, reset=reset, skip_check_array=True)

    # Alignement de la cible sur les lignes de X
    if y is not None:
        if isinstance(y, (pd.Series, pd.DataFrame)):
            if not y.index.equals(X.index):
                raise ValueError("y must have the same index as X, in the same order")
        else:
            values = np.asarray(y)
            if values.ndim not in (1, 2) or len(values) != len(X):
                raise ValueError(
                    f"y must have one row per row of X ({len(X)}), got shape {values.shape}"
                )
            y = (
                pd.Series(values, index=X.index)
                if values.ndim == 1
                else pd.DataFrame(values, index=X.index)
            )

    layout = TemporalLayout.from_index(X.index, freq=freq, detect_freq=detect_freq)
    return X, y, layout
