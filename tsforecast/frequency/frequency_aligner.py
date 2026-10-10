"""Frequency alignment of selected variables of a time series or a panel.

This module provides :class:`FrequencyAligner`, a standalone tool that brings
selected variables of a dataset to a target frequency: variables finer than
the target are aggregated, coarser ones are interpolated. The actual
conversions are delegated to
:class:`tsforecast.utils.frequency.converter.FrequencyConverter`, but the
aligner follows its own index conventions, designed to keep the other columns
of the dataset in place:

- The output index is the input index, extended by the target dates it does
  not already hold. Aggregating onto labels already present in the index (same
  position as the source) leaves the rows untouched, number and order;
  interpolating adds the sub-periods of the target grid covering the **full
  source periods** of the first and last observations (a quarterly series in
  position start ending on 2023-07-01 densifies to monthly up to 2023-09-01).
- A converted column lives on the target grid only: it is NaN on every other
  date (and, once aggregated, NaN for incomplete periods).
- An explicit target position (``'QE'``, ``'MS'``) is honoured; a position-less
  target (``'Q'``, ``'M'``) follows the position of the source index.
- Source frequencies are detected per (entity, column) from the observed
  values (not from the index) and passed explicitly to the converter, so a
  quarterly variable carried on a monthly index is handled as quarterly.
  Detection is modal: an isolated missing observation costs only its own
  target period.
"""
# Modules de base
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union, get_args

# Manipulation de données
import numpy as np
import pandas as pd

# Utilitaires du package
from ..utils.frequency.converter import AggregationMethod, FrequencyConverter
from ..utils.frequency.utils import normalize_frequency, is_higher_frequency
from ..utils.parse import parse_frequency
from ..panel.utils import (
    build_panel_index,
    extract_column_names,
    get_entity_target_frequency,
    get_unique_panel_entities,
    group_keys_by_entity_and_variable,
    is_panel_data,
    iter_entity_blocks,
    split_variable_key,
)

# Détection de la fréquence de l'index
from ..utils.frequency.utils import (
    detect_frequency,
    detect_dataset_frequency,
    target_offset_for_index,
)

# Méthodes d'agrégation acceptées, alignées sur celles du convertisseur
_AGGREGATION_METHODS = get_args(AggregationMethod)


# Classe d'alignement des fréquences de jeu de données avec des fréquences cibles
class FrequencyAligner:
    """Align selected variables to target frequencies via aggregation or interpolation.

    Handles time series (``DatetimeIndex``) and panel data (``MultiIndex``
    whose last level is a ``DatetimeIndex``), delegating the actual frequency
    conversions to ``FrequencyConverter``. Panel data is processed per entity
    to respect entity-specific target frequencies. The public entry points are
    :meth:`convert_to_target` and :meth:`build_densified_index`.

    Examples:
        >>> import pandas as pd
        >>> aligner = FrequencyAligner()
        >>> dates = pd.date_range('2023-01-01', periods=6, freq='MS')
        >>> df = pd.DataFrame({'sales': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=dates)
        >>> aligner.convert_to_target(df, ['sales'], 'Q')['sales'].dropna().tolist()
        [6.0, 15.0]
    """

    # Initialisation
    def __init__(self):
        """Initialize the FrequencyAligner."""
        # Initialisation du convertisseur de fréquences
        self._freq_converter = FrequencyConverter()

    # -------------------------------------------------------------------------
    # Validation et normalisation des entrées
    # -------------------------------------------------------------------------
    # Méthode auxiliaire de validation de l'axe temporel
    @staticmethod
    def _check_time_index(df: pd.DataFrame, is_panel: bool) -> None:
        """Check that the time axis of ``df`` is a DatetimeIndex.

        Args:
            df: Input DataFrame.
            is_panel: Whether the data is panel data (time axis = last level).

        Raises:
            TypeError: If the time axis is not a DatetimeIndex, with a
                conversion hint for a PeriodIndex.
        """
        # Axe temporel : index d'une série, dernier niveau d'un panel
        time_index = df.index.get_level_values(-1) if is_panel else df.index
        if isinstance(time_index, pd.DatetimeIndex):
            return

        # Message explicite, avec une piste de conversion pour un PeriodIndex
        where = 'the last level of the panel MultiIndex' if is_panel else 'the index'
        hint = ''
        if isinstance(time_index, pd.PeriodIndex):
            hint = (
                " Convert the periods to timestamps first, with "
                "PeriodIndex.to_timestamp(how='start') or how='end' according to "
                "the position of the data."
            )
        raise TypeError(
            f"FrequencyAligner requires a DatetimeIndex as {where}, "
            f"got {type(time_index).__name__}.{hint}"
        )

    # Méthode auxiliaire d'extension des noms de colonnes à toutes les entités
    @staticmethod
    def _expand_keys(
        df: pd.DataFrame,
        keys: List[Union[str, Tuple]],
        is_panel: bool,
    ) -> List[Union[str, Tuple]]:
        """Expand plain column names to one (entity..., column) key per entity.

        On a panel, a plain column name designates the column for every
        entity; tuple keys are kept as they are. Time series keys are returned
        unchanged.

        Args:
            df: Input DataFrame.
            keys: Variable keys (column names or (entity..., variable) tuples).
            is_panel: Whether the data is panel data.

        Returns:
            Keys without duplicates, in their first-appearance order.
        """
        # Série temporelle : clés inchangées
        if not is_panel:
            return list(keys)

        # Entités du panel, calculées une seule fois et seulement si nécessaire
        entities: Optional[List[tuple]] = None
        expanded: List[Union[str, Tuple]] = []
        for key in keys:
            # Clé déjà rattachée à une entité
            if isinstance(key, tuple):
                expanded.append(key)
                continue
            # Nom de colonne seul : la colonne de chaque entité
            if entities is None:
                entities = get_unique_panel_entities(df)
            expanded.extend(entity + (key,) for entity in entities)

        # Dédoublonnage, ordre de première apparition conservé
        return list(dict.fromkeys(expanded))

    # Méthode auxiliaire de résolution de l'offset cible
    @staticmethod
    def _resolve_target_offset(index: pd.DatetimeIndex, target_frequency: str) -> str:
        """Resolve the pandas offset of the target frequency for a source index.

        An explicit target position (``'QE'``, ``'MS'``) is honoured as is; a
        position-less target (``'Q'``, ``'M'``) takes the position of the
        source index, so that its labels land on the source grid.

        Args:
            index: Source DatetimeIndex.
            target_frequency: Target frequency, with or without position.

        Returns:
            Pandas offset alias of the target frequency.

        Raises:
            ValueError: If ``target_frequency`` cannot be parsed.
        """
        # Position explicite de la cible : prioritaire
        if parse_frequency(frequency_str=target_frequency).position is not None:
            return target_frequency

        # Repli : position de l'index source
        return target_offset_for_index(index, target_frequency)

    # -------------------------------------------------------------------------
    # Itération unifiée séries temporelles / panel
    # -------------------------------------------------------------------------
    # Méthode auxiliaire d'itération sur les séries à convertir
    def _iter_variable_series(
        self,
        df: pd.DataFrame,
        keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        is_panel: bool,
    ) -> Iterator[Tuple[tuple, Union[np.ndarray, slice], str, pd.Series, str]]:
        """Yield each variable to convert as a date-indexed series.

        Single entry point of the time series / panel disjunction: a time
        series is iterated as a panel holding one entity ``()`` covering
        every row, so aggregation and interpolation are written once, against
        a plain date-indexed series.

        Args:
            df: Input DataFrame.
            keys: Variable keys to convert (column names or
                (entity..., variable) tuples).
            target_frequency: Target frequency (str or per-entity dict).
            is_panel: Whether the data is panel data.

        Yields:
            Tuple ``(entity, mask, col, series, entity_target)`` where
            ``mask`` selects the entity's rows in ``df``, ``series`` is the
            column restricted to the entity and indexed by date only, and
            ``entity_target`` the target frequency resolved for the entity.

        Raises:
            ValueError: If an entity holding a column to convert has
                duplicated dates.
        """
        # Regroupement des colonnes par entité : hors panel, l'entité portée par
        # une éventuelle clé-tuple est ignorée (une seule série par colonne)
        grouped = (
            group_keys_by_entity_and_variable(self._expand_keys(df, keys, is_panel))
            if is_panel else {(): extract_column_names(keys)}
        )

        # Parcours des blocs d'entités (bloc unique hors panel)
        for entity, mask, block in iter_entity_blocks(df, is_panel=is_panel):
            # Colonnes de l'entité présentes dans le jeu de données
            cols = [col for col in grouped.get(entity, []) if col in df.columns]
            if not cols:
                continue

            # Dates dupliquées : une somme ou une interpolation n'aurait pas de sens
            if block.index.has_duplicates:
                duplicated = block.index[block.index.duplicated()].unique()
                where = f" for entity {entity}" if entity else ''
                raise ValueError(
                    f"Duplicate dates{where}: "
                    f"{', '.join(str(date) for date in duplicated[:5])}. "
                    f"Aggregation and interpolation require one row per date."
                )

            # Fréquence cible de l'entité : l'entité vide d'une série temporelle
            # ne peut être servie que par une cible globale
            entity_target = get_entity_target_frequency(entity, target_frequency)

            # Parcours des colonnes de l'entité
            for col in cols:
                yield entity, mask, col, block[col], entity_target

    # Méthode auxiliaire d'affectation d'une colonne sur les lignes d'une entité
    @staticmethod
    def _assign_entity_values(
        result: pd.DataFrame,
        mask: Union[np.ndarray, slice],
        col: str,
        values: np.ndarray,
    ) -> None:
        """Assign values to the rows of one entity, in place.

        Args:
            result: Frame to update in place.
            mask: Boolean array selecting the entity's rows.
            col: Column to update.
            values: Values to write, positionally aligned on ``mask``.
        """
        # Promotion en flottant des colonnes entières ou booléennes : les valeurs
        # converties portent des NaN (hors bornes de période, périodes
        # incomplètes) qu'un dtype entier ne peut pas accueillir
        if result[col].dtype.kind in 'iub':
            result[col] = result[col].astype(float)

        # Valeurs booléennes mêlées de NaN (agrégations 'all' / 'any') : colonne
        # passée en object, seul dtype capable de porter True / False / NaN
        if np.asarray(values).dtype.kind in 'bO' and result[col].dtype.kind != 'O':
            result[col] = result[col].astype(object)

        # Affectation positionnelle sur les lignes de l'entité
        result.loc[mask, col] = values

    # Méthode auxiliaire d'écriture des séries converties
    def _assemble(
        self,
        df: pd.DataFrame,
        converted: Dict[tuple, Dict[str, pd.Series]],
        is_panel: bool,
    ) -> pd.DataFrame:
        """Write converted series on the index extended by their dates.

        Each converted column is written on the target grid only (NaN on the
        other dates of its entity). When no converted date is missing from the
        index, the original index is kept, row order included.

        Args:
            df: Original DataFrame.
            converted: Converted series, keyed by entity then column.
            is_panel: Whether the data is panel data.

        Returns:
            DataFrame with the converted columns.
        """
        # Index étendu des dates converties absentes de l'index d'origine
        target_index = self.build_densified_index(df, converted, is_panel)

        # Aucune date ajoutée (l'union contient l'index d'origine) : index
        # d'origine conservé, ordre des lignes compris
        if len(target_index) == len(df.index):
            result = df.copy()
        else:
            result = df.reindex(target_index)

        # Affectation des colonnes converties, entité par entité
        # Parcours des entités
        for entity, mask, block in iter_entity_blocks(result, is_panel=is_panel):
            # Parcours des colonnes
            for col, series in converted.get(entity, {}).items():
                # Colonne portée par la seule grille cible
                self._assign_entity_values(
                    result, mask, col, series.reindex(block.index).values
                )

        return result

    # -------------------------------------------------------------------------
    # Conversions unitaires, sur une série indexée par dates
    # -------------------------------------------------------------------------
    # Méthode auxiliaire d'agrégation d'une série indexée par dates
    def _aggregate_series(
        self,
        series: pd.Series,
        target_frequency: str,
        method: AggregationMethod = 'sum',
    ) -> Optional[pd.Series]:
        """Aggregate one date-indexed series to a lower frequency.

        Args:
            series: Column to aggregate, indexed by date only.
            target_frequency: Target frequency of the series.
            method: Aggregation method passed to the converter.

        Returns:
            Aggregated series indexed on the target period labels (NaN for
            incomplete periods), or None when the series holds no
            observation at all.
        """
        # Série sans aucune observation : rien à agréger
        if series.dropna().empty:
            return None

        # Fréquence propre de la variable, détectée sur ses valeurs observées
        # puis transmise au convertisseur : celui-ci n'a plus à la déduire de
        # la forme des données qu'on lui tend, l'index restant à la fréquence
        # de la grille (journalière pour une variable mensuelle creuse)
        try:
            source_freq = detect_frequency(series, return_format='full')
        except (ValueError, TypeError):
            # Moins de deux observations : repli sur la détection interne du
            # convertisseur, qui lit la fréquence de l'index
            source_freq = None

        # Agrégation à la fréquence cible, périodes complètes seulement
        return self._freq_converter.aggregate_to_lower_frequency(
            series,
            self._resolve_target_offset(series.index, target_frequency),
            method=method,
            full_periods_only=True,
            source_freq=source_freq,
        )

    # Méthode auxiliaire d'interpolation d'une série indexée par dates
    def _interpolate_series(
        self,
        series: pd.Series,
        target_frequency: str,
        method: str = 'linear',
        limit: Union[int, Literal['default'], None] = 'default',
        limit_direction: Optional[Literal['forward', 'backward', 'both']] = None,
        limit_area: Optional[Literal['inside', 'outside']] = None,
    ) -> Optional[pd.Series]:
        """Interpolate one date-indexed series to a higher frequency.

        The converter interpolates from the observed values only, on the
        variable's own grid: a quarterly variable carried on a monthly index
        is interpolated from its quarterly observations.

        Args:
            series: Column to interpolate, indexed by date only.
            target_frequency: Target frequency of the series.
            method: Interpolation method passed to the converter.
            limit: Maximum number of consecutive NaN values to fill.
            limit_direction: Direction in which to fill NaN values.
            limit_area: Restriction area for filling NaN values.

        Returns:
            Interpolated series, indexed on the target frequency grid covering
            the full source periods of the first and last observations, or
            None when the series holds fewer than two observations (no
            frequency of its own to interpolate from).
        """
        # Moins de deux observations : aucune fréquence propre, rien à interpoler
        if series.count() < 2:
            return None

        # Interpolation à la fréquence cible. La fréquence propre de la variable
        # est détectée sur ses valeurs observées (et non sur l'index) puis
        # transmise au convertisseur ; indétectable (None), le convertisseur
        # se replie sur sa propre détection
        return self._freq_converter.interpolate_to_higher_frequency(
            series,
            self._resolve_target_offset(series.index, target_frequency),
            method=method,
            limit=limit,
            limit_direction=limit_direction,
            limit_area=limit_area,
            source_freq=detect_frequency(series, return_format='with_position'),
        )

    # -------------------------------------------------------------------------
    # Conversions de jeux de données
    # -------------------------------------------------------------------------
    # Méthode d'aggrégation des données à une fréquence cible
    def _aggregate_to_target(
        self,
        df: pd.DataFrame,
        aggregate_keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        method: AggregationMethod = 'sum',
    ) -> pd.DataFrame:
        """Aggregate high-frequency columns to target frequency.

        For panel data, aggregation is performed per entity to respect
        entity-specific target frequencies. When the period labels already
        exist in the index (target in the position of the source, or without
        position), the output index is identical to ``df``'s, row order
        included; otherwise the missing labels are added (see
        :meth:`_assemble`).

        Args:
            df: Input DataFrame.
            aggregate_keys: Variable keys to aggregate (column names or
                (entity..., variable) tuples).
            target_frequency: Target frequency (str or per-entity dict).
            method: Aggregation method passed to the converter.

        Returns:
            DataFrame with aggregated columns.
        """
        # Cas où les clés d'aggrégation ne sont pas spécifiées
        if not aggregate_keys:
            return df

        # Détection unique de la structure du jeu de données (série ou panel)
        is_panel = is_panel_data(df)

        # Agrégation de chaque série (séries temporelles et panel confondus)
        converted: Dict[tuple, Dict[str, pd.Series]] = {}
        for entity, _, col, series, entity_target in self._iter_variable_series(
            df, aggregate_keys, target_frequency, is_panel
        ):
            aggregated = self._aggregate_series(series, entity_target, method=method)
            # Colonne sans observation pour l'entité : laissée telle quelle
            if aggregated is None:
                continue
            # Périodes incomplètes écartées : leur label, NaN, n'a pas à étendre
            # l'index (cas d'une période tronquée au-delà de la dernière date)
            converted.setdefault(entity, {})[col] = aggregated.dropna()

        return self._assemble(df, converted, is_panel)

    # Méthode auxiliaire de construction de l'index densifié
    def build_densified_index(
        self,
        df: pd.DataFrame,
        interpolated: Dict[tuple, Dict[str, pd.Series]],
        is_panel: bool,
    ) -> pd.Index:
        """Build the index densified with the dates of converted series.

        The time index of each entity is the union of its original dates and
        the dates of its converted series, so that entity-specific target
        frequencies are respected. The union is kept whole: interpolation
        covers the full source periods of the first and last observations,
        and those sub-periods legitimately belong to the data (a quarterly
        observation in position start covers the two months that follow it,
        in position end the two months that precede it).

        Args:
            df: Original DataFrame (DatetimeIndex or panel MultiIndex).
            interpolated: Converted (interpolated or aggregated) series, keyed
                by entity then column.
            is_panel: Whether the data is panel data.

        Returns:
            DatetimeIndex for a time series, MultiIndex combining each entity
            with its densified time index for a panel.

        Raises:
            TypeError: If the time axis of ``df`` is not a DatetimeIndex.

        Examples:
            >>> import pandas as pd
            >>> aligner = FrequencyAligner()
            >>> df = pd.DataFrame(
            ...     {'gdp': [1.0, 2.0]},
            ...     index=pd.date_range('2023-01-01', periods=2, freq='QS'),
            ... )
            >>> monthly = pd.Series(0.0, index=pd.date_range('2023-01-01', '2023-06-01', freq='MS'))
            >>> len(aligner.build_densified_index(df, {(): {'gdp': monthly}}, is_panel=False))
            6
        """
        # Validation de l'axe temporel
        self._check_time_index(df, is_panel)

        # Initialisation des morceaux d'index par entité
        parts: List[pd.Index] = []

        # Parcours des blocs d'entités (bloc unique hors panel)
        for entity, _, block in iter_entity_blocks(df, is_panel=is_panel):
            # Union des dates d'origine et des dates des séries converties
            densified_times = block.index
            for series in interpolated.get(entity, {}).values():
                densified_times = densified_times.union(series.index)

            # Reconstruction du MultiIndex de l'entité pour un panel
            parts.append(
                build_panel_index(entity, densified_times, names=df.index.names)
                if is_panel else densified_times
            )

        # Panel sans aucune entité : index d'origine conservé
        if not parts:
            return df.index

        # Concaténation des morceaux, entité par entité
        return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]

    # Méthode d'interpolation d'un jeu de données à une fréquence cible
    def _interpolate_to_target(
        self,
        df: pd.DataFrame,
        interpolate_keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        method: str = 'linear',
        limit: Union[int, Literal['default'], None] = 'default',
        limit_direction: Optional[Literal['forward', 'backward', 'both']] = None,
        limit_area: Optional[Literal['inside', 'outside']] = None,
    ) -> pd.DataFrame:
        """Interpolate low-frequency columns to target frequency.

        The resulting index is **densified**: it is the union of the original
        index and the dates generated by interpolation. Passing a sparse frame
        (e.g. quarterly observations only) therefore yields a frame at the
        target frequency, while passing a frame already indexed at the target
        frequency simply fills its NaN holes. Columns that are not interpolated
        keep their original values and are NaN on the newly created dates;
        interpolated columns live on the target grid only.

        Densification spans the **full source periods** of the first and last
        observations, whose sub-periods are covered by the data: a ``QS``
        series ending on 2023-07-01 densifies to ``MS`` up to 2023-09-01, a
        ``QE`` series starting on 2023-03-31 densifies to ``ME`` down to
        2023-01-31.

        For panel data, interpolation and densification are performed per
        entity to respect entity-specific target frequencies. Columns with
        fewer than two observations for an entity are left as they are.

        Args:
            df: Input DataFrame.
            interpolate_keys: Variable keys to interpolate (column names or
                (entity..., variable) tuples).
            target_frequency: Target frequency (str or per-entity dict).
            method: Interpolation method passed to
                :meth:`FrequencyConverter.interpolate_to_higher_frequency`.
            limit: Maximum number of consecutive NaN values to fill. If
                ``'default'``, the frequency conversion factor between the
                variable's own frequency and the target frequency (e.g. 3 for
                quarterly→monthly); None for no limit.
            limit_direction: Direction in which to fill NaN values. If None,
                ``'forward'`` for a target in position start and
                ``'backward'`` for a target in position end.
            limit_area: Restriction area for filling NaN values
                (``'inside'`` or ``'outside'``).

        Returns:
            DataFrame with interpolated columns, indexed on the densified index.
        """
        # Cas où aucune clé d'interpolation n'est spécifiée, retourne le jeu de données inchangé
        if not interpolate_keys:
            return df

        # Détection unique de la structure du jeu de données (série ou panel)
        is_panel = is_panel_data(df)

        # Interpolation de chaque série (séries temporelles et panel confondus)
        converted: Dict[tuple, Dict[str, pd.Series]] = {}
        for entity, _, col, series, entity_target in self._iter_variable_series(
            df, interpolate_keys, target_frequency, is_panel
        ):
            interpolated = self._interpolate_series(
                series,
                entity_target,
                method=method,
                limit=limit,
                limit_direction=limit_direction,
                limit_area=limit_area,
            )
            # Moins de deux observations pour l'entité : colonne laissée telle quelle
            if interpolated is None:
                continue
            converted.setdefault(entity, {})[col] = interpolated

        return self._assemble(df, converted, is_panel)

    # Méthode générique de conversion vers la fréquence cible (agrégation ou interpolation)
    def convert_to_target(
        self,
        df: pd.DataFrame,
        keys: List[Union[str, Tuple]],
        target_frequency: Union[str, Dict],
        agg_method: AggregationMethod = 'sum',
        interp_method: str = 'linear',
        interp_limit: Union[int, Literal['default'], None] = 'default',
        interp_limit_direction: Optional[Literal['forward', 'backward', 'both']] = None,
        interp_limit_area: Optional[Literal['inside', 'outside']] = None,
    ) -> pd.DataFrame:
        """Convert columns to a target frequency, aggregating or interpolating as needed.

        For each variable key (per entity for panel data), the source
        frequency is detected from the variable's observed values. If the
        target frequency is **higher** (more granular) than the source, the
        column is interpolated; otherwise it is aggregated with
        ``agg_method``.

        Output index: the input index, extended by the target dates it does
        not hold. Aggregation onto labels already present in the index keeps
        it unchanged, row order included; interpolation adds the sub-periods
        covering the full source periods of the first and last observations.
        Each converted column lives on the target grid only (NaN elsewhere,
        and NaN for incomplete aggregated periods); the other columns keep
        their values and are NaN on the added dates.

        Target position: an explicit position (``'QE'``, ``'MS'``) is
        honoured; a position-less target (``'Q'``, ``'M'``) follows the
        position of the source index.

        Args:
            df: Input DataFrame, indexed by a DatetimeIndex (time series) or
                by a MultiIndex whose last level is a DatetimeIndex (panel).
            keys: Variable keys to convert: column names or
                (entity..., variable) tuples. On a panel, a column name
                designates the column for every entity. Keys naming an absent
                column or entity are ignored.
            target_frequency: Target frequency (str or per-entity dict).
            agg_method: Aggregation method used when downsampling, any value
                supported by
                :meth:`FrequencyConverter.aggregate_to_lower_frequency`
                (``'sum'``, ``'mean'``, ``'first'``, ``'last'``, ``'min'``,
                ``'max'``, ``'median'``, ``'std'``, ``'count'``, ``'all'``,
                ``'any'``). Defaults to ``'sum'``.
            interp_method: Interpolation method used when upsampling.
                Defaults to ``'linear'``. ``'linear'``, ``'time'``,
                ``'index'`` and ``'values'`` hold the edge observations
                constant beyond the first / last one (within the limit);
                the scipy methods (``'nearest'``, ``'zero'``, ``'slinear'``,
                ``'quadratic'``, ``'cubic'``) leave them NaN.
            interp_limit: Maximum number of consecutive NaN values to fill
                during interpolation. If ``'default'``, the frequency
                conversion factor (e.g. 3 for quarterly→monthly); None for no
                limit.
            interp_limit_direction: Direction in which to fill NaN values. If
                None, ``'forward'`` for a target in position start and
                ``'backward'`` for a target in position end (position
                resolved as above).
            interp_limit_area: Restriction area for NaN filling during
                interpolation (``'inside'`` or ``'outside'``).

        Returns:
            DataFrame with converted columns.

        Raises:
            ValueError: If ``agg_method`` is not supported, if a target dict
                misses a keyed entity, or if an entity holding a column to
                convert has duplicated dates.
            TypeError: If the time axis is not a DatetimeIndex (e.g. a
                PeriodIndex).

        Examples:
            >>> import pandas as pd
            >>> aligner = FrequencyAligner()
            >>> dates = pd.date_range('2023-01-01', periods=3, freq='QS')
            >>> df = pd.DataFrame({'gdp': [100.0, 110.0, 120.0]}, index=dates)
            >>> out = aligner.convert_to_target(df, ['gdp'], 'M')
            >>> len(out)
            9
            >>> monthly = pd.DataFrame(
            ...     {'rate': [1.0, 2.0, 3.0]},
            ...     index=pd.date_range('2023-01-01', periods=3, freq='MS'),
            ... )
            >>> aligner.convert_to_target(monthly, ['rate'], 'Q', agg_method='mean')['rate'].iloc[0]
            np.float64(2.0)
        """
        # Validation de la méthode d'agrégation, avant tout calcul
        if agg_method not in _AGGREGATION_METHODS:
            raise ValueError(
                f"Unsupported aggregation method: {agg_method!r}, "
                f"should be in {sorted(_AGGREGATION_METHODS)}"
            )

        # Cas où aucune clé n'est spécifiée
        if not keys:
            return df

        # Détection unique de la structure du jeu de données (série ou panel)
        is_panel = is_panel_data(df)

        # Validation de l'axe temporel
        self._check_time_index(df, is_panel)

        # Noms de colonnes seuls sur un panel : une clé par entité
        keys = self._expand_keys(df, keys, is_panel)

        # Initialisation des listes de clés selon la direction de conversion
        aggregate_keys: List[Union[str, Tuple]] = []
        interpolate_keys: List[Union[str, Tuple]] = []

        # Détection des fréquences sources (col → freq pour TS, (entité, col) → freq pour panel)
        freq_map = detect_dataset_frequency(df)

        # Classification des clés selon la relation fréquence source / fréquence cible
        for key in keys:
            # Décomposition de la clé : entité toujours en tuple, () hors panel
            entity, col = split_variable_key(key)
            # Vérification que la colonne est dans le jeu de données
            if col not in df.columns:
                continue
            # Clé de lookup : tuple complet pour un panel, nom de colonne sinon
            lookup_key = key if is_panel else col
            # Extraction de la fréquence source
            source_freq = freq_map.get(lookup_key)
            # Fréquence cible : globale pour TS, par entité pour panel
            key_target = get_entity_target_frequency(
                entity if is_panel else (), target_frequency
            )
            # Orientation de la conversion
            if source_freq and is_higher_frequency(
                normalize_frequency(key_target),
                normalize_frequency(source_freq),
            ):
                interpolate_keys.append(key)
            else:
                aggregate_keys.append(key)

        # Application des conversions
        result = df
        if aggregate_keys:
            result = self._aggregate_to_target(
                result, aggregate_keys, target_frequency, method=agg_method
            )
        if interpolate_keys:
            result = self._interpolate_to_target(
                result, interpolate_keys, target_frequency,
                method=interp_method,
                limit=interp_limit,
                limit_direction=interp_limit_direction,
                limit_area=interp_limit_area,
            )
        return result
