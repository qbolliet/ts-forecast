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
from ..utils.frequency.utils import (
    detect_dataset_frequency,
    detect_index_frequency,
    target_offset_for_index,
    is_higher_frequency,
    normalize_frequency,
)
from ..panel.utils import (
    get_entity_mask,
    get_unique_panel_entities,
    is_panel_data,
    normalize_entity_key,
)
from ..utils.parse.utils import build_frequency_string
from ..utils.time.utils import get_period_start, get_period_end
from ..utils.frequency.converter import FrequencyConverter

# Type pour le scope d'imputation (fenêtre de prédiction)
ImputationScope = Literal['strict', 'extended_backward', 'extended_forward', 'extended_both']

# Type pour le scope d'entraînement : les quatre valeurs d'ImputationScope, plus
# 'unrestricted' qui supprime toute restriction de fenêtre à l'entraînement
TrainingScope = Literal['strict', 'extended_backward', 'extended_forward',
                        'extended_both', 'unrestricted']


# Classe de calcul de la fenêtre d'imputation à partir des couvertures multi-fréquences
class ImputationWindowCalculator:
    """Calculate the imputation window and extended training windows for imputation.

    The strict imputation window is the temporal interval where ALL series
    in the dataset have data (directly or via sub-period coverage). A
    quarterly observation, for example, is considered to cover all
    high-frequency sub-periods (e.g., months) within that quarter.

    The class builds a boolean coverage matrix at the highest detected
    frequency, then derives coverage (fraction of columns covered) at
    each date. The strict window is where coverage equals 1.0. An extended
    window grows backward or forward from it as long as coverage meets the
    specified threshold.

    THREE masks are derived from the same coverage computation, so that the
    range a model is fitted on can be set independently of the range it
    imputes:

    - ``imputation_strict_window_mask_`` — coverage == 1.0, no extension.
    - ``imputation_window_mask_`` — the prediction window, extended per
      ``imputation_scope`` with ``coverage_threshold``.
    - ``training_window_mask_`` — the training window, extended per
      ``training_scope`` with ``training_coverage_threshold``; the value
      'unrestricted' lifts the restriction entirely.

    Both training parameters default to None, meaning "follow the prediction
    window": the three masks then reduce to the historical single window.
    Callers pick one through the ``kind`` argument of
    :meth:`get_imputation_window_mask` and :meth:`get_mask_at_frequency`,
    whose default 'imputation' preserves the historical behaviour.

    For panel data (MultiIndex with entity levels + time level), windows
    are computed independently per entity.

    The internal high-frequency grid (used for ``imputation_window_mask_``,
    ``coverage_by_date_``, etc.) can extend past the last date actually
    present in the fitted data: it spans to the end of the last period
    covered by a low-frequency observation, which may fall after the last
    high-frequency row (e.g. a quarterly observation covers 3 months of
    grid even if only the first month exists as a row). Consumers must
    therefore never intersect this mask directly (e.g. via a bare ``&``)
    with a mask computed on their own data index; always go through
    :meth:`get_imputation_window_mask` (optionally passing ``data``), which
    reindexes/aligns the mask onto the caller's index.

    For panel data, every dict-valued attribute
    (``imputation_window_start_``, ``imputation_window_end_``,
    ``imputation_strict_window_start_``, ``imputation_strict_window_end_``,
    ``column_coverage_`` and ``index_freq_``) is keyed by the entity
    **tuple**, even when the panel has a single entity level:
    ``('France',)``, never ``'France'``. These are exactly the keys returned
    by :func:`tsforecast.panel.utils.get_unique_panel_entities`, so a consumer
    can index them directly and a ``KeyError`` genuinely signals a bug rather
    than a key-format mismatch.

    The three window masks (``imputation_strict_window_mask_``,
    ``imputation_window_mask_``, ``training_window_mask_``) and
    ``coverage_by_date_`` are NOT dicts: for panel data they are a single
    ``pd.Series`` on a MultiIndex ``(entity..., date)``, in the
    level order of the input frame, so a caller can align them without
    rebuilding a per-entity structure. An entity whose window could not be
    determined (index frequency not identified, or empty coverage grid)
    still contributes its rows, all ``False``; ``entities_without_window_``
    lists those entities.

    Methods taking an entity from the caller
    (:meth:`get_columns_with_coverage`, :meth:`get_mask_at_frequency`)
    normalize it via ``normalize_entity_key``, so scalars remain accepted at
    the public boundary only.

    Attributes:
        imputation_window_start_: Start of the imputation window, following
            ``imputation_scope`` — identical to
            ``imputation_strict_window_start_`` when ``imputation_scope='strict'``
            (the default), extended otherwise. Always matches the active
            range of ``imputation_window_mask_``. Scalar for time series,
            Dict[tuple, Optional[Timestamp]] for panel.
        imputation_window_end_: End of the imputation window, following
            ``imputation_scope``. Same type as imputation_window_start_.
        imputation_strict_window_start_: Start of the STRICT imputation
            window (coverage == 1.0), independent of ``imputation_scope``.
            Scalar for time series, Dict[tuple, Optional[Timestamp]] for panel.
        imputation_strict_window_end_: End of the strict imputation window,
            independent of ``imputation_scope``. Same type as
            imputation_strict_window_start_.
        training_window_start_: Start of the training window, following
            ``training_scope`` — identical to
            ``imputation_strict_window_start_`` when ``imputation_scope='strict'``
            (the default), extended otherwise. Always matches the active
            range of ``training_window_mask_``. Scalar for time series,
            Dict[tuple, Optional[Timestamp]] for panel.
        training_window_end_: End of the training window, following
            ``training_scope``. Same type as training_window_start_.
        imputation_strict_window_mask_: Boolean mask on the high-frequency
            grid identifying observations in the STRICT window
            (coverage == 1.0), before any extension. ``pd.Series`` on a
            DatetimeIndex for time series; ``pd.Series`` on a MultiIndex
            ``(entity..., date)`` for panel, covering every row
            of the fitted frame — entities without a window contribute
            ``False`` rows.
        imputation_window_mask_: Boolean mask on the high-frequency grid
            identifying observations in the imputation (prediction) window,
            i.e. the strict window extended per ``imputation_scope``. Same
            type as imputation_strict_window_mask_.
        training_window_mask_: Boolean mask on the high-frequency grid
            identifying observations in the TRAINING window, i.e. the strict
            window extended per ``training_scope`` — all True when that scope
            is 'unrestricted'. Same type as imputation_strict_window_mask_.
        coverage_by_date_: Ratio of columns covered per high-freq date.
            ``pd.Series`` on a DatetimeIndex for time series; ``pd.Series``
            on a MultiIndex ``(entity..., date)`` for panel, restricted to
            entities that have a coverage grid.
        entities_without_window_: Tuple of entity key tuples whose imputation
            window could not be determined (index frequency not identified,
            or empty coverage grid). Their rows appear in the three masks as
            ``False``. Empty tuple for time series data.
        index_freq_: Detected highest frequency. str for time series,
            Dict[tuple, Optional[str]] for panel.
        column_coverage_: Dict mapping column names to (start, end)
            coverage timestamps, for time series data. For panel data,
            Dict[tuple, Optional[Dict[str, Tuple]]] mapping each entity
            key to its own per-column coverage dict (one entry per
            entity, not just the last one fitted).

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2023-01-31', periods=12, freq='ME')
        >>> data = pd.DataFrame({
        ...     'var1': [np.nan, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan],
        ...     'var2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        ... }, index=dates)
        >>> calc = ImputationWindowCalculator(
        ...     imputation_scope='extended_both',
        ...     training_scope='unrestricted',
        ... )
        >>> calc.fit(data)
        >>> print(calc.imputation_window_start_, calc.imputation_window_end_)
        >>> training = calc.get_imputation_window_mask(data, kind='training')
    """

    # Initialisation
    def __init__(
        self,
        coverage_threshold: float = 0.5,
        imputation_scope: ImputationScope = 'strict',
        training_scope: Optional[TrainingScope] = None,
        training_coverage_threshold: Optional[float] = None,
        min_columns: int = 1,
    ):
        """Initialize the ImputationWindowCalculator.

        Args:
            coverage_threshold: Minimum fraction of columns that must have
                coverage for a date to be included in the extended PREDICTION
                window. Value between 0 and 1. Default 0.5.
            imputation_scope: How to determine the imputation (prediction)
                window:
                - 'strict': Only dates where all columns have coverage.
                - 'extended_backward': Extend before the strict window
                  where coverage >= threshold.
                - 'extended_forward': Extend after the strict window
                  where coverage >= threshold.
                - 'extended_both': Extend in both directions.
            training_scope: How to determine the training window. Accepts the
                four ``imputation_scope`` values plus 'unrestricted', which
                lifts the window restriction altogether (mask everywhere True)
                so a model can be fitted on a wider range than it imputes.
                None (the default) follows ``imputation_scope``, which keeps a
                single shared window.
            training_coverage_threshold: Coverage threshold of the training
                window extensions. None (the default) follows
                ``coverage_threshold``.
            min_columns: Minimum number of data columns required.
                Must be at least 1. Default 1.

        Raises:
            ValueError: If coverage_threshold or training_coverage_threshold
                is not in [0, 1], if imputation_scope or training_scope is
                invalid, or if min_columns < 1.
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
        if training_scope is not None and training_scope not in (
            'strict', 'extended_backward', 'extended_forward', 'extended_both', 'unrestricted'
        ):
            raise ValueError(
                f"training_scope must be None or one of 'strict', 'extended_backward', "
                f"'extended_forward', 'extended_both', 'unrestricted', got '{training_scope}'"
            )
        if training_coverage_threshold is not None and not 0 <= training_coverage_threshold <= 1:
            raise ValueError(
                f"training_coverage_threshold must be between 0 and 1, "
                f"got {training_coverage_threshold}"
            )
        if min_columns < 1:
            raise ValueError(f"min_columns must be at least 1, got {min_columns}")

        # Stockage des paramètres, tels que reçus : les défauts None ne sont pas
        # résolus ici, la convention sklearn imposant que get_params()/clone()
        # retrouvent la valeur fournie par l'appelant (cf. _effective_* plus bas)
        self.coverage_threshold = coverage_threshold
        self.imputation_scope = imputation_scope
        self.training_scope = training_scope
        self.training_coverage_threshold = training_coverage_threshold
        self.min_columns = min_columns

        # Attributs de fenêtre d'imputation — scalaires pour TS, dict par entité pour panel
        # Bornes suivant imputation_scope (== bornes strictes en scope 'strict')
        self.imputation_window_start_: Optional[Union[pd.Timestamp, Dict[tuple, Optional[pd.Timestamp]]]] = None
        self.imputation_window_end_: Optional[Union[pd.Timestamp, Dict[tuple, Optional[pd.Timestamp]]]] = None
        # Bornes de la fenêtre stricte, indépendantes de imputation_scope
        self.imputation_strict_window_start_: Optional[Union[pd.Timestamp, Dict[tuple, Optional[pd.Timestamp]]]] = None
        self.imputation_strict_window_end_: Optional[Union[pd.Timestamp, Dict[tuple, Optional[pd.Timestamp]]]] = None
        # Masques de fenêtre : pd.Series sur DatetimeIndex pour une série
        # temporelle ; pd.Series unique sur MultiIndex (entity..., date) pour un panel
        # Masque de la fenêtre stricte, avant toute extension
        self.imputation_strict_window_mask_: Optional[pd.Series] = None
        # Masque de la fenêtre d'imputation (prédiction), étendu selon imputation_scope
        self.imputation_window_mask_: Optional[pd.Series] = None
        # Masque de la fenêtre d'entraînement, étendu selon training_scope
        self.training_window_mask_: Optional[pd.Series] = None
        # Entités de panel sans fenêtre déterminable (fréquence d'index non
        # identifiée ou grille de couverture vide) : leurs lignes figurent à
        # False dans les trois masques, cet attribut les recense explicitement
        self.entities_without_window_: Tuple[tuple, ...] = ()

        # Attributs auxiliaires
        self.coverage_by_date_: Optional[pd.Series] = None
        self.index_freq_: Optional[Union[str, Dict[tuple, Optional[str]]]] = None
        # Dict[str, Tuple] pour TS ; Dict[tuple, Optional[Dict[str, Tuple]]] par entité pour panel
        self.column_coverage_: Optional[Union[
            Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]],
            Dict[tuple, Optional[Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]]],
        ]] = None

        # Attributs internes
        self._detected_frequencies: Optional[Dict] = None
        self._is_panel: bool = False
        self._is_fitted: bool = False
        self._converter = FrequencyConverter()
        # Noms des niveaux de l'index du frame ajusté, capturés au fit pour
        # renommer le MultiIndex des masques de panel
        self._index_names: Optional[List[str]] = None

    # Résolution du défaut "suit la fenêtre de prédiction" pour le scope d'entraînement
    @property
    def _effective_training_scope(self) -> str:
        """Training scope actually applied, falling back to imputation_scope.

        Returns:
            ``training_scope`` when set, otherwise ``imputation_scope``.
        """
        return self.training_scope if self.training_scope is not None else self.imputation_scope

    # Résolution du défaut "suit la fenêtre de prédiction" pour le seuil d'entraînement
    @property
    def _effective_training_coverage_threshold(self) -> float:
        """Training coverage threshold actually applied, falling back to coverage_threshold.

        Returns:
            ``training_coverage_threshold`` when set, otherwise
            ``coverage_threshold``.
        """
        return (
            self.training_coverage_threshold
            if self.training_coverage_threshold is not None
            else self.coverage_threshold
        )

    # Méthode d'entraînement du calculateur
    def fit(self, data: pd.DataFrame) -> 'ImputationWindowCalculator':
        """Calculate the imputation and training windows from data.

        Detects column frequencies, builds a coverage matrix at the
        highest detected frequency, then derives the strict window (all
        columns covered) and extends it in two independent directions.
        three boolean masks are stored alongside the window bounds:
        ``imputation_strict_window_mask_`` (no extension),
        ``imputation_window_mask_`` (extended per ``imputation_scope``) and
        ``training_window_mask_`` (extended per ``training_scope``).

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
        # Capture des noms de niveaux d'index pour le renommage du MultiIndex
        # des masques de panel
        self._index_names = list(data.index.names)

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
        imputation window bounds, the three window masks (strict, imputation
        and training), the coverage series and the column coverage mapping.

        Args:
            data: DataFrame with a simple DatetimeIndex.
        """
        # Fréquence la plus élevée parmi toutes les colonnes
        valid_freqs = [f for f in self._detected_frequencies.values() if f is not None]
        if not valid_freqs:
            raise ValueError("Cannot detect any column frequency in the data")

        # Calcul de la fenêtre pour l'unique entité TS
        result = self._compute_window(data, self._detected_frequencies, self.index_freq_)

        # Valorisation des bornes de fenêtre (scope-dépendantes et strictes)
        self.imputation_window_start_ = result['imputation_start']
        self.imputation_window_end_ = result['imputation_end']
        self.training_window_start_ = result['training_start']
        self.training_window_end_ = result['training_end']
        self.imputation_strict_window_start_ = result['imputation_strict_start']
        self.imputation_strict_window_end_ = result['imputation_strict_end']

        # Valorisation des trois masques
        self.imputation_strict_window_mask_ = result['imputation_strict_window_mask']
        self.imputation_window_mask_ = result['imputation_window_mask']
        self.training_window_mask_ = result['training_window_mask']

        # Valorisation des attributs auxiliaires
        self.coverage_by_date_ = result['coverage']
        self.column_coverage_ = result['column_coverage']
        # Notion sans objet hors panel (réinitialisation en cas de ré-estimation)
        self.entities_without_window_ = ()

    # Méthode auxiliaire d'estimation de la fenêtre sur des données de panel
    def _fit_panel(self, data: pd.DataFrame) -> None:
        """Compute per-entity windows and masks for panel data (MultiIndex).

        Iterates over each panel entity, extracts its sub-DataFrame, and
        calls _compute_window. Window bounds (strict and scope-dependent) and
        ``column_coverage_`` are aggregated into dicts keyed by entity tuple.
        The three boolean masks and ``coverage_by_date_`` are instead
        assembled into a single ``pd.Series`` on a MultiIndex
        ``(entity..., date)`` (spec §7.2): an entity whose window cannot be
        determined (index frequency not identified, or empty coverage grid)
        contributes its frame rows as ``False`` and is recorded in
        ``entities_without_window_``.

        Args:
            data: Panel DataFrame with MultiIndex whose last level is time.
        """
        # Extraction des entités des données
        entities = get_unique_panel_entities(data)

        # Initialisation des dictionnaires de bornes (scalaires par entité)
        self.imputation_window_start_ = {}
        self.imputation_window_end_ = {}
        self.training_window_start_ = {}
        self.training_window_end_ = {}
        self.imputation_strict_window_start_ = {}
        self.imputation_strict_window_end_ = {}
        self.column_coverage_ = {}

        # Contributions par entité, fusionnées ensuite en une unique Series à
        # MultiIndex (entity..., date)
        strict_mask_parts: List[pd.Series] = []
        imputation_mask_parts: List[pd.Series] = []
        training_mask_parts: List[pd.Series] = []
        coverage_parts: List[pd.Series] = []
        entities_without_window: List[tuple] = []

        # Parcours des entités
        for entity in entities:
            # Extraction du sous-DataFrame de l'entité avec index temporel simple
            entity_row_mask = self._get_entity_row_mask(data, entity)
            entity_df = data[entity_row_mask].copy()
            entity_df.index = entity_df.index.get_level_values(-1)
            # Dates réelles de l'entité dans le frame ajusté : servent à couvrir
            # ses lignes à False quand aucune fenêtre n'est déterminable
            entity_frame_dates = entity_df.index

            # Fréquences des colonnes pour cette entité
            col_freqs = {
                col: self._detected_frequencies.get(entity + (col,))
                for col in data.columns
            }

            # Clé d'entité TOUJOURS le tuple rendu par get_unique_panel_entities,
            # y compris à un seul niveau : ('France',) et jamais 'France'.
            entity_key = entity

            # Cas où la fréquence de l'index n'a pas pu être identifiée : aucune
            # fenêtre exploitable, mais l'entité a toutes ses lignes à False
            if self.index_freq_[entity_key] is None:
                # Ajout au registre des entités sans fenêtre
                self._store_entity_without_window(entity_key)
                entities_without_window.append(entity_key)
                # Série de booléens False
                self._append_false_mask_parts(
                    entity_key, entity_frame_dates,
                    strict_mask_parts, imputation_mask_parts, training_mask_parts,
                )
                continue

            # Calcul de la fenêtre d'imputation
            result = self._compute_window(entity_df, col_freqs, self.index_freq_[entity_key])

            # Complétion des dictionnaires de bornes (scope-dépendantes et strictes)
            self.imputation_window_start_[entity_key] = result['imputation_start']
            self.imputation_window_end_[entity_key] = result['imputation_end']
            self.training_window_start_[entity_key] = result['training_start']
            self.training_window_end_[entity_key] = result['training_end']
            self.imputation_strict_window_start_[entity_key] = result['imputation_strict_start']
            self.imputation_strict_window_end_[entity_key] = result['imputation_strict_end']
            self.column_coverage_[entity_key] = result['column_coverage']

            # Cas où la grille de couverture est vide : même traitement que la
            # fréquence non identifiée, l'entité n'a pas de masque exploitable
            if result['imputation_window_mask'] is None:
                entities_without_window.append(entity_key)
                self._append_false_mask_parts(
                    entity_key, entity_frame_dates,
                    strict_mask_parts, imputation_mask_parts, training_mask_parts,
                )
                continue

            # Contribution des trois masques et de la couverture, préfixés par
            # la clé d'entité pour former le MultiIndex final
            strict_mask_parts.append(self._entity_series_with_multiindex(
                result['imputation_strict_window_mask'], entity_key))
            imputation_mask_parts.append(self._entity_series_with_multiindex(
                result['imputation_window_mask'], entity_key))
            training_mask_parts.append(self._entity_series_with_multiindex(
                result['training_window_mask'], entity_key))
            coverage_parts.append(self._entity_series_with_multiindex(
                result['coverage'], entity_key))

        # Fusion des contributions en une Series unique à MultiIndex
        self.imputation_strict_window_mask_ = self._assemble_panel_series(strict_mask_parts)
        self.imputation_window_mask_ = self._assemble_panel_series(imputation_mask_parts)
        self.training_window_mask_ = self._assemble_panel_series(training_mask_parts)
        self.coverage_by_date_ = self._assemble_panel_series(coverage_parts)
        self.entities_without_window_ = tuple(entities_without_window)

        # Vérification qu'au moins une entité a une fenêtre valide. Une fenêtre
        # stricte absente partout reste rédhibitoire : l'extension ne peut jamais
        # créer de fenêtre là où aucune fenêtre stricte n'existe (cf.
        # _compute_window). Seul training_scope='unrestricted' y échappe, en
        # couvrant toute la grille — le panel reste alors entraînable, même si
        # aucune valeur n'y est imputable
        valid_starts = [v for v in self.imputation_strict_window_start_.values() if v is not None]
        if not valid_starts:
            if not bool(self.training_window_mask_.any()):
                raise ValueError("No imputation window found for any entity in the panel")
            # Warning
            warnings.warn(
                "No entity has a strict imputation window; training proceeds on the "
                "unrestricted training window, but no value can be imputed.",
                UserWarning
            )

    # Méthode auxiliaire d'enregistrement des bornes None d'une entité sans fenêtre
    def _store_entity_without_window(self, entity_key: tuple) -> None:
        """Record None window bounds and coverage for an entity with no window.

        ``column_coverage_`` must stay keyed by every entity, otherwise
        :meth:`get_columns_with_coverage` raises a ``KeyError`` on those.

        Args:
            entity_key: Entity key tuple whose index frequency could not be
                identified or whose coverage grid is empty.
        """
        # Bornes indéterminées (dicts par entité, inchangé)
        self.imputation_window_start_[entity_key] = None
        self.imputation_window_end_[entity_key] = None
        self.training_window_start_[entity_key] = None
        self.training_window_end_[entity_key] = None
        self.imputation_strict_window_start_[entity_key] = None
        self.imputation_strict_window_end_[entity_key] = None
        self.column_coverage_[entity_key] = None

    # Méthode auxiliaire d'ajout d'une contribution tout-False pour une entité
    def _append_false_mask_parts(
        self,
        entity_key: tuple,
        entity_frame_dates: pd.DatetimeIndex,
        *mask_parts_lists: List[pd.Series],
    ) -> None:
        """Append an all-False contribution covering the entity's frame rows.

        An entity without a determinable window must still contribute its
        rows (value ``False``, not an absence of rows) so callers can align
        the mask without rebuilding it.

        Args:
            entity_key: Entity key tuple, prepended as MultiIndex levels.
            entity_frame_dates: Dates of the entity's rows in the fitted frame.
            *mask_parts_lists: Accumulator lists to append the contribution to.
        """
        # Contribution tout-faux sur les dates réelles du frame ajusté
        false_series = pd.Series(False, index=entity_frame_dates)
        for parts in mask_parts_lists:
            parts.append(self._entity_series_with_multiindex(false_series, entity_key))

    # Méthode auxiliaire de préfixage de l'index d'une Series par la clé d'entité
    def _entity_series_with_multiindex(
        self,
        series: pd.Series,
        entity_key: tuple,
    ) -> pd.Series:
        """Prefix a per-entity Series index with the entity key levels.

        Args:
            series: Series indexed by a simple DatetimeIndex (one entity's
                high-frequency grid, or one entity's frame dates).
            entity_key: Entity key tuple prepended as leading index levels.

        Returns:
            Series carrying the same values on a MultiIndex ``(entity..., date)``.
        """
        # Construction du MultiIndex (niveaux d'entité..., date)
        tuples = [(*entity_key, date) for date in series.index]
        multi_index = pd.MultiIndex.from_tuples(tuples)
        return pd.Series(series.to_numpy(), index=multi_index)

    # Méthode auxiliaire de fusion des contributions par entité en une Series unique
    def _assemble_panel_series(self, parts: List[pd.Series]) -> pd.Series:
        """Concatenate per-entity Series into one MultiIndex Series.

        Args:
            parts: Per-entity Series already carrying a ``(entity..., date)``
                MultiIndex.

        Returns:
            Single Series spanning every contributed entity, its MultiIndex
            levels named after the fitted frame's index. An empty (dtype-bool)
            Series when no entity contributed.
        """
        # Concaténation ; repli sur une Series vide si aucune contribution
        combined = pd.concat(parts) if parts else pd.Series(dtype=bool)
        if isinstance(combined.index, pd.MultiIndex) and self._index_names is not None:
            combined.index = combined.index.set_names(self._index_names)
        return combined

    # Méthode auxiliaire permettant d'extraire des observations relatives à une entité du panel
    # Délégation à la fonction utilitaire partagée de tsforecast.panel.utils
    def _get_entity_row_mask(self, data: pd.DataFrame, entity: tuple) -> np.ndarray:
        """Build a boolean row mask selecting the given entity from panel data.

        Thin wrapper over :func:`tsforecast.panel.utils.get_entity_mask`.

        Args:
            data: Panel DataFrame with MultiIndex.
            entity: Normalized entity key tuple.

        Returns:
            Boolean numpy array of length len(data).
        """
        return get_entity_mask(data, entity)

    # Méthode de calcul de la fenêtre d'imputation et d'entraînement pour une entité
    def _compute_window(
        self,
        df: pd.DataFrame,
        col_freqs: Dict[str, Optional[str]],
        index_freq: str,
    ) -> Dict:
        """Compute imputation and training windows for one entity.

        Builds the high-frequency grid and coverage matrix, derives the
        coverage series, then constructs the strict window mask
        (coverage == 1.0). Strict window bounds are derived from this mask,
        independently of imputation_scope. TWO further masks are then derived
        from it by _build_scope_mask, which never mutates it: the imputation
        (prediction) mask, extended per imputation_scope / coverage_threshold,
        and the training mask, extended per training_scope /
        training_coverage_threshold. Scope-dependent bounds are derived from
        the imputation mask only.

        Args:
            df: DataFrame with simple DatetimeIndex for the entity.
            col_freqs: Dict mapping column names to detected frequencies.
            index_freq: Highest (most granular) frequency for this entity,
                used as the reference grid frequency.

        Returns:
            Dict with keys:
                - 'imputation_start': Start of the scope-dependent window
                  (== 'imputation_strict_start' when imputation_scope is
                  'strict').
                - 'imputation_end': End of the scope-dependent window.
                - 'training_start': Start of the scope-dependent window
                   (== 'imputation_strict_start' when training_scope is
                    'strict').
                - 'training_end': End of the scope-dependent window.
                - 'imputation_strict_start': Start of the strict window,
                  independent of imputation_scope.
                - 'imputation_strict_end': End of the strict window,
                  independent of imputation_scope.
                - 'imputation_strict_window_mask': Boolean pd.Series on the
                  grid where coverage == 1.0, BEFORE any extension.
                - 'imputation_window_mask': Same, extended per
                  imputation_scope / coverage_threshold.
                - 'training_window_mask': Same, extended per training_scope
                  / training_coverage_threshold (all True when the effective
                  training scope is 'unrestricted').
                - 'coverage': coverage pd.Series on the high-freq grid.
                - 'column_coverage': Dict of per-column (start, end)
                  tuples.
        """
        # Initialisation du dictionnaire résultat par défaut
        _none_result = {
            'imputation_start': None,
            'imputation_end': None,
            'training_start': None,
            'training_end': None,
            'imputation_strict_start': None,
            'imputation_strict_end': None,
            'imputation_strict_window_mask': None,
            'imputation_window_mask': None,
            'training_window_mask': None,
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

        # Construction du masque de la fenêtre STRICTE sur la grille haute fréquence :
        # les trois masques en dérivent, il n'est donc jamais réaffecté ni muté
        # Utilisation d'un seuil légèrement inférieur à 1.0 pour les erreurs d'arrondi
        imputation_strict_window_mask = pd.Series(
            np.asarray(coverage >= (1.0 - 1e-10), dtype=bool),
            index=grid,
        )

        # Dérivation des bornes de la fenêtre stricte depuis le masque
        strict_dates = imputation_strict_window_mask.index[imputation_strict_window_mask]

        # Cas où pour aucune période l'ensemble des variables sont disponibles simultanément
        if len(strict_dates) == 0:
            warnings.warn(
                "There is no period in the DataFrame where all the variables are available. "
                "Only training_scope='unrestricted' allows training in this case.",
                UserWarning
            )
            # Les extensions sont tout de même appliquées : partant d'un masque strict
            # vide, leur garde "len(masked_dates) == 0" les rend inopérantes et le
            # résultat reste tout-faux, sauf pour 'unrestricted' qui devient tout-vrai
            return {
                **_none_result,
                'coverage': coverage,
                'column_coverage': column_coverage,
                'imputation_strict_window_mask': imputation_strict_window_mask,
                'imputation_window_mask': self._build_scope_mask(
                    coverage,
                    imputation_strict_window_mask,
                    self.imputation_scope,
                    self.coverage_threshold,
                ),
                'training_window_mask': self._build_scope_mask(
                    coverage,
                    imputation_strict_window_mask,
                    self._effective_training_scope,
                    self._effective_training_coverage_threshold,
                ),
            }

        # Extraction des bornes strictes depuis le masque, indépendamment du scope
        imputation_strict_start = strict_dates.min()
        imputation_strict_end = strict_dates.max()

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

        # Dérivation des deux masques étendus, chacun avec son scope et son seuil
        imputation_window_mask = self._build_scope_mask(
            coverage,
            imputation_strict_window_mask,
            self.imputation_scope,
            self.coverage_threshold,
        )
        training_window_mask = self._build_scope_mask(
            coverage,
            imputation_strict_window_mask,
            self._effective_training_scope,
            self._effective_training_coverage_threshold,
        )

        # Cas où la fenêtre d'imputation ne contient qu'une seule observation.
        # Avertissement rattaché au seul masque d'imputation
        imputation_dates = imputation_window_mask.index[imputation_window_mask]
        if len(imputation_dates) <= 1:
            warnings.warn(
                "Imputation window contains only one observation. Consider relaxing "
                "constraints or using a different imputation_scope.",
                UserWarning
            )

        # Cas où la fenêtre d'entraînement ne contient qu'une seule observation.
        # Avertissement rattaché au seul masque d'entraînement
        training_dates = training_window_mask.index[training_window_mask]
        if len(training_dates) <= 1:
            warnings.warn(
                "Training window contains only one observation. Consider relaxing "
                "constraints or using a different training_scope.",
                UserWarning
            )

        # Bornes scope-dépendantes, dérivées du masque final (identiques aux bornes
        # strictes en scope 'strict', puisqu'aucune extension n'a alors eu lieu)
        # Imputation
        imputation_start = imputation_dates.min()
        imputation_end = imputation_dates.max()
        # Entraînement
        training_start = training_dates.min()
        training_end = training_dates.max()

        return {
            'imputation_start': imputation_start,
            'imputation_end': imputation_end,
            'training_start': training_start,
            'training_end': training_end,
            'imputation_strict_start': imputation_strict_start,
            'imputation_strict_end': imputation_strict_end,
            'imputation_strict_window_mask': imputation_strict_window_mask,
            'imputation_window_mask': imputation_window_mask,
            'training_window_mask': training_window_mask,
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
            grid_offset = build_frequency_string(
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

    # Méthode auxiliaire de dérivation d'un masque de scope depuis le masque strict
    def _build_scope_mask(
        self,
        coverage: pd.Series,
        strict_mask: pd.Series,
        scope: str,
        threshold: float,
    ) -> pd.Series:
        """Derive one scope mask from the strict mask, without mutating it.

        Args:
            coverage: Coverage series on the high-freq grid.
            strict_mask: Boolean pd.Series of the strict window
                (coverage == 1.0), shared by every derived mask.
            scope: One of 'strict', 'extended_backward', 'extended_forward',
                'extended_both' or 'unrestricted'.
            threshold: Coverage threshold governing the extensions of this
                mask, so that prediction and training windows may differ.

        Returns:
            New boolean pd.Series on the grid. All True for 'unrestricted';
            an unchanged copy of strict_mask for 'strict'.
        """
        # Aucune restriction : toute la grille est retenue
        if scope == 'unrestricted':
            return pd.Series(True, index=coverage.index)

        # Copie systématique : jamais l'original, les trois masques en dérivent
        mask = strict_mask.copy()
        if scope in ('extended_backward', 'extended_both'):
            mask = self._extend_backward(coverage, mask, threshold)
        if scope in ('extended_forward', 'extended_both'):
            mask = self._extend_forward(coverage, mask, threshold)
        return mask

    # Méthode auxiliaire d'extension du masque d'entraînement avant le début de la fenêtre stricte
    def _extend_backward(
        self,
        coverage: pd.Series,
        mask: pd.Series,
        threshold: float,
    ) -> pd.Series:
        """Extend a boolean window mask backward while coverage meets the threshold.

        Extension is contiguous: walking backward from the window start,
        it stops at the first date whose coverage falls below the
        threshold — a date beyond a below-threshold gap is never
        activated, even if its own coverage would otherwise qualify.

        Args:
            coverage: coverage series on the high-freq grid.
            mask: Boolean pd.Series on the high-freq grid representing the
                current window (strict or partially extended).
            threshold: Minimum coverage a date must reach to be activated.
                Passed in rather than read off the instance, so that the
                training window can use its own threshold.

        Returns:
            New boolean pd.Series with additional True values prepended
            where coverage contiguously meets the threshold. Returns an
            unchanged copy if no extension is possible.
        """
        # Identification de la borne de début actuelle du masque
        masked_dates = mask.index[mask]
        if len(masked_dates) == 0:
            return mask.copy()
        window_start = masked_dates.min()

        # Extraction des observations antérieures, de la plus récente à la plus ancienne
        before = coverage[coverage.index < window_start].sort_index(ascending=False)
        # Si la borne actuelle correspond au début de la grille, le masque est inchangé
        if before.empty:
            return mask.copy()

        # Extension contiguë : n_valid = position du premier point sous le seuil
        # (ou la longueur totale si aucun point ne le viole)
        below = (before < threshold).to_numpy()
        n_valid = int(np.argmax(below)) if below.any() else len(before)

        # Activation des dates contiguës satisfaisant le seuil (no-op si n_valid == 0)
        new_mask = mask.copy()
        new_mask[before.index[:n_valid]] = True
        return new_mask

    # Méthode auxiliaire d'extension du masque d'entraînement après la fin de la fenêtre stricte
    def _extend_forward(
        self,
        coverage: pd.Series,
        mask: pd.Series,
        threshold: float,
    ) -> pd.Series:
        """Extend a boolean window mask forward while coverage meets the threshold.

        Extension is contiguous: walking forward from the window end, it
        stops at the first date whose coverage falls below the threshold —
        a date beyond a below-threshold gap is never activated, even if
        its own coverage would otherwise qualify.

        Args:
            coverage: coverage series on the high-freq grid.
            mask: Boolean pd.Series on the high-freq grid representing the
                current window (strict or partially extended).
            threshold: Minimum coverage a date must reach to be activated.
                Passed in rather than read off the instance, so that the
                training window can use its own threshold.

        Returns:
            New boolean pd.Series with additional True values appended
            where coverage contiguously meets the threshold. Returns an
            unchanged copy if no extension is possible.
        """
        # Identification de la borne de fin actuelle du masque
        masked_dates = mask.index[mask]
        if len(masked_dates) == 0:
            return mask.copy()
        window_end = masked_dates.max()

        # Extraction des observations postérieures, de la plus ancienne à la plus récente
        after = coverage[coverage.index > window_end].sort_index(ascending=True)
        # Si la borne actuelle correspond à la fin de la grille, le masque est inchangé
        if after.empty:
            return mask.copy()

        # Extension contiguë : n_valid = position du premier point sous le seuil
        # (ou la longueur totale si aucun point ne le viole)
        below = (after < threshold).to_numpy()
        n_valid = int(np.argmax(below)) if below.any() else len(after)

        # Activation des dates contiguës satisfaisant le seuil (no-op si n_valid == 0)
        new_mask = mask.copy()
        new_mask[after.index[:n_valid]] = True
        return new_mask

    # Méthode auxiliaire de sélection du masque désigné par "kind"
    def _select_mask(
        self,
        kind: str,
    ) -> Optional[pd.Series]:
        """Return the fitted mask attribute designated by ``kind``.

        Args:
            kind: 'imputation' for the prediction window, 'strict' for the
                unextended window, 'training' for the training window.

        Returns:
            The corresponding fitted mask attribute, untouched — a
            ``pd.Series`` on a DatetimeIndex (time series) or on a MultiIndex
            ``(entity..., date)`` (panel).

        Raises:
            ValueError: If kind is not one of the three accepted values.
        """
        # Validation explicite : un kind fautif doit échouer, pas retomber
        # silencieusement sur la fenêtre d'imputation
        if kind not in ('imputation', 'strict', 'training'):
            raise ValueError(
                f"kind must be one of 'imputation', 'strict', 'training', got '{kind}'"
            )
        return {
            'imputation': self.imputation_window_mask_,
            'strict': self.imputation_strict_window_mask_,
            'training': self.training_window_mask_,
        }[kind]

    # Méthode d'extraction du masque booléen de la fenêtre d'imputation
    def get_imputation_window_mask(
        self,
        data: Optional[Union[pd.DataFrame, pd.Series]] = None,
        kind: Literal['imputation', 'strict', 'training'] = 'imputation',
    ) -> pd.Series:
        """Get one of the boolean window masks, optionally aligned to data.

        Args:
            data: If provided, the mask is re-indexed on ``data.index``
                (rows absent from the fitted grid are False). For panel data
                this aligns the MultiIndex ``(entity..., date)`` rows of
                ``data`` directly.
            kind: Which window to read:
                - 'imputation' (default): the prediction window, extended
                  per ``imputation_scope``. The default preserves the
                  historical behaviour of this method.
                - 'strict': the unextended window (coverage == 1.0).
                - 'training': the training window, extended per
                  ``training_scope``.

        Returns:
            Boolean Series aligned on ``data.index`` if ``data`` is given.
            Otherwise the raw fitted mask: a ``pd.Series`` on a DatetimeIndex
            for time series, or a ``pd.Series`` on a
            MultiIndex ``(entity..., date)`` for panel.

        Raises:
            ValueError: If calculator not fitted, or if kind is invalid.
        """
        # Vérification de l'entraînement du calculateur
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        # Sélection de la source : déjà une Series (à MultiIndex pour un panel),
        # l'alignement ci-dessous est donc le même pour les deux structures
        source = self._select_mask(kind)

        # Sans données : retour du masque brut sur la grille interne
        if data is None:
            return source

        # Avec données : réindexation directe sur l'index de l'appelant. Les
        # paires (entity..., date) absentes de la grille ajustée tombent à False
        return (
            source
            .reindex(data.index)
            .fillna(False)
            .astype(bool)
        )


    # Méthode de conversion du masque d'imputation vers une fréquence inférieure
    def get_mask_at_frequency(
        self,
        frequency: Union[str, Dict[tuple, str]],
        kind: Literal['imputation', 'strict', 'training'] = 'imputation',
    ) -> pd.Series:
        """Get the selected window mask resampled to a lower frequency.

        A lower-frequency period is True if and only if all its high-frequency
        sub-periods fall within the selected window. The conversion delegates
        to :meth:`FrequencyConverter.aggregate_to_lower_frequency` with
        ``method='all'``, anchoring the target offset on the same start/end
        position as the source mask's grid (see
        :func:`tsforecast.utils.frequency.utils.target_offset_for_index`) so the
        result stays reindexable against the original data.

        For panel data, the conversion is computed independently per entity,
        as each entity may have a different source frequency; the per-entity
        results are then assembled into a single ``pd.Series`` on a MultiIndex
        ``(entity..., date)``. The internal per-entity loop is
        kept rather than delegating to
        :meth:`FrequencyConverter.convert_frequency` (which does route an
        ``{entity: freq}`` mapping over a panel): that path aggregates per
        entity through ``convert_frequency`` on a simple index, which exposes
        neither the boolean ``method='all'`` aggregation (a target period is
        True iff every sub-period is) nor the start/end offset anchoring via
        ``target_offset_for_index`` that keeps the result reindexable — so
        delegating would change the mask semantics.

        Args:
            frequency: Target (lower) frequency offset string (e.g., 'QE',
                'YE') or dictionnary entity -> frequency string. Entity keys
                may be given in scalar form for a single-level panel: they
                are normalized to tuples before lookup.
            kind: Which window to convert — 'imputation' (default), 'strict'
                or 'training'. See :meth:`get_imputation_window_mask`.

        Returns:
            Boolean Series at the target frequency: on a DatetimeIndex for
            time series, or on a MultiIndex
            ``(entity..., date)`` for panel. Entities without a valid
            fitted mask are simply absent from the panel result.

        Raises:
            ValueError: If calculator not fitted, if kind is invalid, or if
                the target frequency is not lower than the mask frequency.

        Examples:
            >>> monthly_calc.fit(data)
            >>> quarterly_mask = monthly_calc.get_mask_at_frequency('QE')
        """
        # Vérification de l'entraînement du calculateur
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        # Sélection de la source, identique à get_imputation_window_mask
        source = self._select_mask(kind)

        # Cas des séries temporelles : conversion unique
        if not self._is_panel:
            return self._convert_mask_to_frequency(
                mask=source,
                source_freq=self.index_freq_,
                target_freq=frequency,
            )

        # Normalisation des clés du dictionnaire fourni par l'appelant : c'est la
        # seule frontière où une clé scalaire peut encore entrer
        if isinstance(frequency, dict):
            frequency = {
                normalize_entity_key(entity): freq
                for entity, freq in frequency.items()
            }

        # Cas des données de panel : conversion indépendante par entité (source
        # et cible propres à chacune), puis réassemblage en une Series unique
        converted_parts: List[pd.Series] = []
        for entity in self.index_freq_:
            entity_mask = self._entity_grid_series(source, entity)
            target = frequency[entity] if isinstance(frequency, dict) else frequency
            converted = self._convert_mask_to_frequency(
                mask=entity_mask,
                source_freq=self.index_freq_[entity],
                target_freq=target,
            )
            # Entité sans masque exploitable : simplement absente du résultat
            if converted is None:
                continue
            converted_parts.append(
                self._entity_series_with_multiindex(converted, entity)
            )
        return self._assemble_panel_series(converted_parts)

    # Méthode auxiliaire d'extraction de la tranche d'une entité depuis un masque à MultiIndex
    def _entity_grid_series(
        self,
        source: Optional[pd.Series],
        entity_key: tuple,
    ) -> Optional[pd.Series]:
        """Extract one entity's slice from a MultiIndex mask, on a simple index.

        Args:
            source: Unified MultiIndex mask Series, or None.
            entity_key: Entity key tuple to extract.

        Returns:
            Boolean Series indexed by a simple DatetimeIndex for that entity,
            or None if the entity is absent from the source.
        """
        # Absence de source exploitable
        if source is None or not isinstance(source.index, pd.MultiIndex):
            return None

        # Sélection des lignes de l'entité (clés d'entité = tuples, cf. §3.4)
        rows = np.ones(len(source), dtype=bool)
        for level, wanted in enumerate(entity_key):
            rows &= (source.index.get_level_values(level) == wanted)
        if not rows.any():
            return None

        # Retour sur un index temporel simple
        entity_series = source[rows]
        entity_series.index = entity_series.index.get_level_values(-1)
        return entity_series

    # Méthode auxiliaire de conversion d'un masque booléen vers une fréquence inférieure
    def _convert_mask_to_frequency(
        self,
        mask: Optional[pd.Series],
        source_freq: Optional[str],
        target_freq: str,
    ) -> Optional[pd.Series]:
        """Resample a single boolean window mask to a lower frequency.

        A target period is True if and only if all its sub-periods at the
        source frequency are covered by the mask. The conversion delegates to
        :meth:`FrequencyConverter.aggregate_to_lower_frequency` with
        ``method='all'``, which already forces False on periods only
        partially present in the mask's grid (e.g. a target period straddling
        the edge of the fitted grid). The target offset is anchored on the
        same start/end position as the source mask's index (via
        :func:`tsforecast.utils.frequency.utils.target_offset_for_index`), so
        the returned index stays reindexable against the original data.

        Args:
            mask: Boolean Series on the source-frequency grid, or None.
            source_freq: Frequency of the mask index, or None.
            target_freq: Target (lower) frequency offset string.

        Returns:
            Boolean Series at the target frequency, anchored like the source
            mask's index, or None if either the mask or the source frequency
            is missing.

        Raises:
            ValueError: If source_freq is not higher than target_freq.

        Examples:
            >>> grid = pd.date_range('2023-01-01', periods=12, freq='MS')
            >>> mask = pd.Series(True, index=grid)
            >>> calc._convert_mask_to_frequency(mask, 'M', 'Y').tolist()
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

        # Ancrage de l'offset cible sur la position (début/fin) de la grille
        # source, pour que l'index résultat retombe sur des dates réindexables
        # sur les données (et non sur des labels de fin de période inutilisables
        # face à une grille source ancrée en début de période)
        target_offset = target_offset_for_index(mask.index, target_freq)

        # Une période cible est dans la fenêtre ssi TOUTES ses sous-périodes le
        # sont : délégué entièrement à 'all', qui encode déjà à la fois la
        # comparaison booléenne et le garde-fou de couverture intégrale.
        # La fréquence source est transmise plutôt que redevinée : elle est
        # déjà un paramètre de cette méthode, et fait autorité sur l'index
        return self._converter.aggregate_to_lower_frequency(
            mask, target_offset, method='all', source_freq=source_freq
        ).astype(bool)

    # Méthode d'extraction de la couverture associée à chaque série
    def get_columns_with_coverage(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        entity: Optional[tuple] = None,
    ) -> Union[List[str], Dict[tuple, List[str]]]:
        """Get list of columns that have coverage in a given time range.

        For time series data, returns the list of covered columns directly.
        For panel data, ``column_coverage_`` holds one per-column coverage
        dict per entity: pass ``entity`` to query a single entity, or omit
        it to get a dict mapping every entity key to its own covered-column
        list.

        Args:
            start: Start of the query range.
            end: End of the query range.
            entity: Panel entity key, as a tuple — a scalar is accepted and
                normalized, since it is a caller-supplied value. Ignored for
                time series data. Required to get a List[str] result for
                panel data; if omitted for panel data, a dict over all
                entities is returned instead.

        Returns:
            List of column names with coverage overlapping [start, end]
            (time series, or panel with ``entity`` given). Dict mapping
            each entity key (always a tuple) to such a list (panel,
            ``entity`` omitted).

        Raises:
            ValueError: If calculator not fitted.
            KeyError: If ``entity`` is not one of the fitted panel entities.
        """
        # Vérification que le calculateur est estimé
        if not self._is_fitted:
            raise ValueError("Calculator not fitted. Call fit() first.")

        # Cas où la couverture des colonnes n'est pas disponible
        if self.column_coverage_ is None:
            return {} if (self._is_panel and entity is None) else []

        # Cas des données de panel : couverture par entité
        if self._is_panel:
            if entity is not None:
                # Normalisation de la clé fournie par l'appelant : les clés
                # internes sont des tuples, y compris à un seul niveau (§3.4)
                col_coverage = self.column_coverage_[normalize_entity_key(entity)]
                return self._columns_with_coverage(col_coverage, start, end)
            return {
                entity_key: self._columns_with_coverage(col_coverage, start, end)
                for entity_key, col_coverage in self.column_coverage_.items()
            }

        # Cas des séries temporelles
        return self._columns_with_coverage(self.column_coverage_, start, end)

    # Méthode auxiliaire d'extraction des colonnes couvertes depuis un dict de couverture
    def _columns_with_coverage(
        self,
        column_coverage: Optional[Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> List[str]:
        """Filter a single per-column coverage dict for overlap with [start, end].

        Args:
            column_coverage: Dict mapping column names to (start, end)
                coverage timestamps, or None.
            start: Start of the query range.
            end: End of the query range.

        Returns:
            List of column names with coverage overlapping [start, end].
        """
        # Cas où la couverture des colonnes n'est pas disponible
        if column_coverage is None:
            return []

        # Initialisation de la liste résultat
        columns_with_coverage = []
        # Parcours des colonnes
        for col, (col_start, col_end) in column_coverage.items():
            # Exclusion des colonnes sans couverture définie
            if col_start is None or col_end is None:
                continue
            # Chevauchement si col_start <= end ET col_end >= start
            if col_start <= end and col_end >= start:
                columns_with_coverage.append(col)

        return columns_with_coverage