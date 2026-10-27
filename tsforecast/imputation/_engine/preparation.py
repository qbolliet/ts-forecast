"""Preparation of the working data of one ``HighFrequencyImputer`` pass, shared by fit and transform.

``prepare_work_frame`` runs phases 0-4 of a pass — input contract, alignment and naming of the
target, working frame, frequency plan (fit) or frequency check (transform), windows, additive
transformer, provenance tracker — and returns them as ``PreparedData``; the fit-only part is
grouped in ``FitPreparation``. Also holds the helpers of those phases: target naming and
alignment, reserved level name check, window calculator factory and readable window bounds.
"""
# Importation des modules
# Modules de base
import warnings
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
# Manipulation de données
import numpy as np
import pandas as pd
# Sklearn
from sklearn.base import TransformerMixin, clone
# Utilitaires du package
from ...panel.utils import get_unique_panel_entities, is_panel_data
from ...utils._constants import FREQUENCY_LEVEL_NAME
from ...utils.frequency.utils import detect_index_frequency
# Composants de l'imputation
from .._components.imputation_window import ImputationWindowCalculator
from .._components.provenance import ImputationProvenanceTracker
from .frequency_layout import FrequencyLayout, plan_frequencies, stage_frequency_label
from .params import validate_target_frequency_format
from .replay import (
    check_prediction_pass_replayable,
    check_transform_frequencies,
    warn_rows_outside_window,
)
from .run_context import PairKey

if TYPE_CHECKING:
    from ..high_frequency_imputer import HighFrequencyImputer

# Clé d'entité normalisée : tuple des valeurs des niveaux d'entité
EntityKey = Tuple[Any, ...]
# Bornes (début, fin) d'une fenêtre : un couple, ou un couple par entité
WindowBounds = Union[Tuple, Dict[EntityKey, Tuple]]

# Nom de colonne de la cible anonyme une fois fusionnée au jeu de travail
_UNNAMED_TARGET = '__target__'


# Ce que seul le fit dérive de ses données
@dataclass(frozen=True)
class FitPreparation:
    """What only the fit derives from its data, published by the fit as fitted attributes.

    Attributes:
        is_panel: Whether the data are handled as a panel — declared by
            ``panel_cols`` or detected on the index.
        entities: Entity keys of the panel, None on a time series.
        source_index_frequency_label: Frequency label of the input index, which
            identifies at inversion the level to keep in a multi-frequency
            output; None for an irregular or too short index.
        layout: Frequency plan of the fit: detected frequencies, target,
            categories and progression.
        strict_window_mask: Mask of the strict window.
        imputation_window_mask: Mask of the prediction window.
        training_window_mask: Mask of the training window.
        imputation_window: Readable ``(start, end)`` bounds of the prediction
            window, or one pair per entity.
        training_window: Readable bounds of the training window.
        additive_transformer: Fitted clone of the additive transformer, None
            without one.
    """

    is_panel: bool
    entities: Optional[List[EntityKey]]
    source_index_frequency_label: Optional[str]
    layout: FrequencyLayout
    strict_window_mask: pd.Series
    imputation_window_mask: pd.Series
    training_window_mask: pd.Series
    imputation_window: WindowBounds
    training_window: WindowBounds
    additive_transformer: Optional[TransformerMixin]


# Données de travail d'une passe, préparées à l'identique par le fit et le transform
@dataclass(frozen=True)
class PreparedData:
    """Working data of one pass, prepared by the same code at fit and at transform.

    Attributes:
        X_work: Frame the pass imputes — ``X``, plus the target when it is
            merged — after the additive transformer.
        y: Target aligned onto ``X``'s index, None without one. The prediction
            pass returns it as is.
        target_column: Name of the target in ``X_work``, None when it is not
            merged (no target, or the prediction pass).
        snapshot_X: Copy of the input, read by the inversion once the caller
            publishes it at the end of the pass.
        snapshot_y: Copy of the aligned target when it is merged, else None.
        window_calculator: Window calculator fitted on the working frame
            before the additive transformer — unfitted at transform when the
            window could not be computed.
        provenance_tracker: Provenance tracker initialized on ``X_work``.
        fit: Fit-only part, None at transform.
        new_frequencies: Frequencies detected at transform for the pairs the
            fit never observed; empty at fit.
        new_misaligned: Those of these pairs observed off the calendar bounds
            of their frequency; empty at fit.
        skipped_columns: Columns of the fit the pass does not replay — the
            target in the prediction pass; empty otherwise.
    """

    X_work: pd.DataFrame
    y: Optional[pd.Series]
    target_column: Optional[str]
    snapshot_X: pd.DataFrame
    snapshot_y: Optional[pd.Series]
    window_calculator: ImputationWindowCalculator
    provenance_tracker: ImputationProvenanceTracker
    fit: Optional[FitPreparation] = None
    new_frequencies: Dict[PairKey, str] = field(default_factory=dict)
    new_misaligned: Tuple[PairKey, ...] = ()
    skipped_columns: FrozenSet[str] = frozenset()


# -----------------------------------------------------------------------------
# Préparation d'une passe
# -----------------------------------------------------------------------------
# Phases 0 à 4 d'une passe, fit ou transform
def prepare_work_frame(
    imputer: 'HighFrequencyImputer',
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    *,
    fitting: bool,
    prediction: bool = False,
) -> PreparedData:
    """Prepare the working data of one pass: phases 0-4 of the fit, 0'-4' of the transform.

    One implementation for both passes, each difference being a parameter:

    - input contract (a DataFrame, no index level named ``'frequency'``),
      alignment and naming of ``y``, input snapshot and working frame: common.
      The prediction pass (``prediction=True``) never merges ``y``, checks
      first that no model of ``X`` reads the target, and snapshots ``X``
      alone;
    - frequencies: the fit **plans** them (entities, label of the input index,
      per-feature keys naming nothing, frequency layout); the transform
      **checks** its data against that plan — a missing column of the fit
      raises, a diverging frequency warns once and the fit frequency is kept;
    - windows: the same calculator factory on the same frame, the one before
      the additive transformer, so that recomputing the window on the data of
      the fit gives the window of the fit back. The fit raises when the window
      cannot be computed and keeps its three masks; the transform warns,
      replays without restriction, and warns once about the rows left out;
    - additive transformer: cloned and fitted by the fit, the fitted object
      applied by the transform — in the prediction pass the target it was
      fitted with is handed to it empty, then dropped;
    - provenance tracker: initialized after the additive transformer, on the
      frame actually imputed.

    Nothing is written on the imputer: the caller publishes what it needs.

    Args:
        imputer: Imputer whose parameters, and at transform fitted state, are
            read.
        X: Features, already validated by the base class.
        y: Target, optional.
        fitting: True for the fit, False for a transform.
        prediction: Whether the transform is the prediction pass of an
            imputer fitted with ``y``.

    Returns:
        The prepared data of the pass.

    Raises:
        ValueError: If ``prediction`` is requested at fit, if ``X`` is not a
            DataFrame, if an index level bears the reserved name, if the
            indices of ``X`` and ``y`` disagree, if the prediction pass meets a
            model that needs the target, if the frequency plan fails (fit), if
            a column of the fit is missing (transform), or if the window
            cannot be computed at fit.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.linear_model import LinearRegression
        >>> from tsforecast.imputation import HighFrequencyImputer
        >>> dates = pd.date_range('2021-01-31', periods=24, freq='ME')
        >>> data = pd.DataFrame({'m1': np.arange(24.0), 'q1': np.nan}, index=dates)
        >>> data.loc[dates[2::3], 'q1'] = np.arange(8.0) * 3 + 10
        >>> y = pd.Series(np.arange(24.0), index=dates, name='y')
        >>> imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())
        >>> prepared = prepare_work_frame(imputer, data, y, fitting=True)
        >>> list(prepared.X_work.columns), prepared.target_column
        (['m1', 'q1', 'y'], 'y')
        >>> prepared.fit.source_index_frequency_label, prepared.fit.layout.progression
        ('M', ['M'])
        >>> replay = prepare_work_frame(imputer.fit(data), data, fitting=False)
        >>> replay.fit is None, replay.target_column, replay.new_frequencies
        (True, None, {})
    """
    # Garde : la passe de prédiction est une passe de transform
    if fitting and prediction:
        raise ValueError(
            "The prediction pass is a transform pass: it cannot be prepared "
            "with fitting=True."
        )

    # Contrat d'entrée : un DataFrame, aucun niveau au nom réservé
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"X must be a pandas DataFrame, got {type(X).__name__}")
    check_reserved_level_name(X, panel_cols=imputer.panel_cols, time_col=imputer.time_col)

    # Structure des données : établie par le fit, relue par le transform
    is_panel = (
        bool(imputer.panel_cols) or is_panel_data(data=X) if fitting else imputer.is_panel_
    )

    # Alignement de l'index de y sur celui de X, par la même fonction pour les
    # deux passes : deux règles divergentes ont déjà fait perdre la cible
    # parmi les colonnes d'étapes
    if y is not None:
        y = align_target_index(
            X, y, getattr(imputer, '_current_conversion_metadata', None)
        )
    # Passe de prédiction : aucun modèle de X ne doit lire la cible absente
    if prediction:
        check_prediction_pass_replayable(imputer.imputation_plan_, imputer.target_column_)
    # Colonne du fit absente de la passe de prédiction : la cible, jamais lue
    skipped_columns = frozenset((imputer.target_column_,)) if prediction else frozenset()
    # Cible fusionnée au jeu de travail, sauf en passe de prédiction
    merge_target = y is not None and not prediction

    # Instantané de l'entrée, publié par l'appelant en fin de passe seulement :
    # un appel qui lève n'est pas une passe, et l'inversion de la précédente
    # doit continuer de lire l'entrée de celle-ci
    snapshot_X = X.copy()
    snapshot_y = y.copy() if merge_target else None

    # Concaténation en un unique jeu de travail. L'index est déjà temporel :
    # la validation de base a converti "time_col" et "panel_cols" en index
    # et refusé tout autre index
    target_column: Optional[str] = None
    if merge_target:
        target_column = resolve_target_column_name(y)
        X_work = pd.concat([X, y.to_frame(name=target_column)], axis=1)
    else:
        X_work = X.copy()

    # Fréquences : plan établi par le fit ; au transform, contrôle contre ce
    # plan — erreur sur colonne manquante, avertissement unique sur
    # divergence, silence sur colonne en trop
    new_frequencies: Dict[PairKey, str] = {}
    new_misaligned: Tuple[PairKey, ...] = ()
    if fitting:
        entities, source_label, layout = _plan_fit_frequencies(
            imputer, X, X_work, is_panel=is_panel
        )
    else:
        new_frequencies, new_misaligned = check_transform_frequencies(
            imputer._frequency_layout_, X_work, optional_columns=skipped_columns
        )

    # Fenêtres, calculées au même stade dans les deux passes : avant le
    # transformateur additif
    window_calculator = _fit_window(imputer, X_work, fitting=fitting)
    # Masques du fit : chaque appelant nomme explicitement le sien
    if fitting:
        masks = {
            kind: window_calculator.get_imputation_window_mask(X_work, kind=kind)
            for kind in ('strict', 'imputation', 'training')
        }

    # Passage en représentation additive, unique échappatoire à l'hypothèse
    # d'additivité de la classe entière
    X_stage, additive_transformer = _apply_additive_transformer(
        imputer, X_work, fitting=fitting, prediction=prediction
    )

    # Initialisation du traceur après le transformateur additif : il scanne le
    # jeu de données réellement imputé, et non celui d'avant la
    # transformation, dont les cellules non nulles ne coïncident pas
    # nécessairement
    tracker = ImputationProvenanceTracker()
    tracker.initialize(X_stage, panel_cols=imputer.panel_cols)

    # Part propre au fit, publiée par lui comme attributs ajustés. Bornes
    # lisibles : chaque attribut porte celles de son masque
    fit_part: Optional[FitPreparation] = None
    if fitting:
        fit_part = FitPreparation(
            is_panel=is_panel,
            entities=entities,
            source_index_frequency_label=source_label,
            layout=layout,
            strict_window_mask=masks['strict'],
            imputation_window_mask=masks['imputation'],
            training_window_mask=masks['training'],
            imputation_window=zip_window_bounds(
                window_calculator.imputation_window_start_,
                window_calculator.imputation_window_end_,
            ),
            training_window=zip_window_bounds(
                window_calculator.training_window_start_,
                window_calculator.training_window_end_,
            ),
            additive_transformer=additive_transformer,
        )

    return PreparedData(
        X_work=X_stage,
        y=y,
        target_column=target_column,
        snapshot_X=snapshot_X,
        snapshot_y=snapshot_y,
        window_calculator=window_calculator,
        provenance_tracker=tracker,
        fit=fit_part,
        new_frequencies=new_frequencies,
        new_misaligned=tuple(new_misaligned),
        skipped_columns=skipped_columns,
    )


# Plan des fréquences du fit et ce qu'il demande des données
def _plan_fit_frequencies(
    imputer: 'HighFrequencyImputer',
    X: pd.DataFrame,
    X_work: pd.DataFrame,
    *,
    is_panel: bool,
) -> Tuple[Optional[List[EntityKey]], Optional[str], FrequencyLayout]:
    """Plan the frequencies of the fit, with the entities and the input index label.

    Args:
        imputer: Imputer whose parameters are read.
        X: Features of the fit.
        X_work: Working frame of the fit, before the additive transformer.
        is_panel: Whether the data are handled as a panel.

    Returns:
        Tuple ``(entities, source_index_frequency_label, layout)``.

    Raises:
        ValueError: If the frequency plan fails (see
            :func:`~tsforecast.imputation._engine.frequency_layout.plan_frequencies`).
    """
    # Identification des entités
    entities = (
        get_unique_panel_entities(X)
        if is_panel and isinstance(X.index, pd.MultiIndex)
        else None
    )

    # Label de fréquence de l'index d'entrée : il identifie, à l'inversion,
    # le niveau à conserver dans une sortie multi-fréquences
    try:
        index_freq = detect_index_frequency(X_work.index, return_format='base')
        source_label: Optional[str] = stage_frequency_label(index_freq)
    except (ValueError, TypeError):
        # Index irrégulier ou trop court : le repli sur le niveau cible suffit
        source_label = None

    # Clés des dictionnaires par feature absentes des données
    imputer._warn_unknown_feature_keys(X_work.columns, entities)

    # Normalisation de target_frequency (la fréquence de prédiction)
    normalized_target_frequency = validate_target_frequency_format(
        imputer.target_frequency
    )

    # Plan des fréquences : expansion et validation de la cible, détection
    # par couple (entité, colonne), classification et progression. Calculé
    # une fois, sur les données d'avant le transformateur additif, puis lu
    # tel quel pour l'exécution des étapes et pour chaque rejeu
    layout = plan_frequencies(
        X_work,
        normalized_target_frequency,
        is_panel=is_panel,
        entities=entities,
        impute_intermediate_frequencies=imputer.impute_intermediate_frequencies,
        on_frequency_mismatch=imputer.on_frequency_mismatch,
        log=imputer._log,
    )
    return entities, source_label, layout


# Calcul de la fenêtre d'une passe
def _fit_window(
    imputer: 'HighFrequencyImputer',
    X_work: pd.DataFrame,
    *,
    fitting: bool,
) -> ImputationWindowCalculator:
    """Fit a fresh window calculator on the data of the pass.

    The window is a constraint on data availability, not an estimated
    parameter: it is a deterministic function of the frame it is computed on.
    Both passes therefore build their calculator with the same factory and
    compute the window on their own data — the frame BEFORE the additive
    transformer, on both paths, or the two windows would not coincide —,
    which is what preserves ``fit_transform(X) == fit(X).transform(X)``.

    Args:
        imputer: Imputer whose four window parameters are read; at transform,
            also its data structure, which names the entities of the rows
            left out.
        X_work: Working frame of the pass, before the additive transformer.
        fitting: True for the fit, False for a transform.

    Returns:
        The fitted calculator — or, at transform, an unfitted one when the
        window could not be computed: its readers then restrict nothing.

    Raises:
        ValueError: At fit, if the window cannot be computed.
    """
    # Calculateur portant les hyperparamètres de l'imputeur
    calculator = _window_calculator_of(imputer)

    # Fit : un échec lève, un repli sur des masques tout-vrais imputerait tout
    # le jeu en silence ; une fenêtre vide n'est jamais un échec (masques
    # tout-faux et avertissement du calculateur). Aucun avertissement propre à
    # l'imputeur pour une fenêtre d'entraînement vide : les masques
    # d'imputation et d'entraînement contiennent tous deux la fenêtre stricte,
    # et aucune extension ne crée de fenêtre là où la stricte est vide. Une
    # fenêtre d'entraînement vide implique une fenêtre d'imputation vide, que le
    # calculateur signale déjà (« no period … where all the variables are
    # available ») — rien n'est alors imputé, ni par modèle ni par repli
    if fitting:
        return calculator.fit(X_work)

    # Transform : seul à tolérer un échec — une seule ligne à transformer ne
    # porte aucune fenêtre — et rejoue alors le plan sans restriction
    try:
        calculator.fit(X_work)
    except ValueError as error:
        warnings.warn(
            f"Could not calculate the imputation window of the data being "
            f"transformed: {error}. Using all available data.",
            UserWarning,
        )
        # Calculateur non ajusté : les gardes des consommateurs le neutralisent
        return _window_calculator_of(imputer)

    # Avertissement unique sur les lignes laissées hors de la fenêtre recalculée
    warn_rows_outside_window(calculator, X_work, is_panel=imputer.is_panel_)
    return calculator


# Application du transformateur additif d'une passe
def _apply_additive_transformer(
    imputer: 'HighFrequencyImputer',
    X_work: pd.DataFrame,
    *,
    fitting: bool,
    prediction: bool,
) -> Tuple[pd.DataFrame, Optional[TransformerMixin]]:
    """Move the working frame into the additive representation of the pass.

    Args:
        imputer: Imputer whose ``additive_transformer`` (fit) or fitted
            ``additive_transformer_`` (transform) is applied.
        X_work: Working frame, before the additive transformer.
        fitting: True to clone and fit the transformer, False to apply the
            fitted one.
        prediction: Whether the pass is the prediction pass: the target the
            transformer was fitted with is handed to it empty, then dropped.

    Returns:
        Tuple ``(frame, transformer)``: the frame the pass imputes — the
        working frame itself without a transformer — and the transformer
        applied, None without one.
    """
    # Transformateur de la passe : clone neuf au fit, objet ajusté au transform
    if fitting:
        transformer = (
            None if imputer.additive_transformer is None
            else clone(imputer.additive_transformer)
        )
    else:
        transformer = imputer.additive_transformer_
    if transformer is None:
        return X_work, None

    # Ajustement au fit ; au transform, application de l'objet ajusté sur un
    # jeu portant la cible : elle lui est remise vide en passe de prédiction
    if fitting:
        X_stage = transformer.fit_transform(X_work)
    else:
        additive_input = (
            X_work.assign(**{imputer.target_column_: np.nan}) if prediction else X_work
        )
        X_stage = transformer.transform(additive_input)

    # Déballage du couple (X, y) que renvoie un transformateur XY
    if isinstance(X_stage, tuple):
        X_stage = X_stage[0]
    # Retrait de la cible remise vide à la passe de prédiction
    if prediction and imputer.target_column_ in X_stage.columns:
        X_stage = X_stage.drop(columns=[imputer.target_column_])
    return X_stage, transformer


# -----------------------------------------------------------------------------
# Cible
# -----------------------------------------------------------------------------
# Résolution du nom de colonne de la cible
def resolve_target_column_name(y: pd.Series) -> str:
    """Resolve the column name used for ``y`` once merged into a frame.

    Single naming rule shared by the fit, the transform and the inversion.
    The fit-time name and the transform-time name must be identical:
    otherwise the target is not found among the stage columns and silently
    skips imputation.

    Args:
        y: Target series, possibly unnamed.

    Returns:
        ``y.name`` if set, else the fallback ``'__target__'``.

    Examples:
        >>> import pandas as pd
        >>> resolve_target_column_name(pd.Series([1.0], name='gdp'))
        'gdp'
        >>> resolve_target_column_name(pd.Series([1.0]))
        '__target__'
    """
    return y.name if y.name is not None else _UNNAMED_TARGET


# Alignement de l'index de la cible sur celui de X
def align_target_index(
    X: pd.DataFrame,
    y: pd.Series,
    conversion_metadata: Optional[Mapping[str, Any]] = None,
) -> pd.Series:
    """Align ``y``'s index onto ``X``'s, tolerating the col->index step.

    The check is an **index equality** check, not a length check.
    Two series of the same length carrying different labels are a caller
    bug, not a case to fix silently — ``pd.concat`` aligns on index value,
    so reindexing would grow the working frame with NaN rows instead of
    raising.

    When ``time_col``/``panel_cols`` are used, the base transformer
    converts ``X``'s columns into a Multi/DatetimeIndex before the imputer
    ever sees it, but never touches ``y``. If ``y`` still carries the
    index recorded before that conversion, it is repositioned onto
    ``X.index``.

    The same instants stored at another unit (``y`` in nanoseconds, ``X``
    in seconds) match: ``y`` then takes ``X``'s dates, so that both come
    out at the unit of ``X``.

    Args:
        X: Working features, already validated and converted.
        y: Target series as received by the pass.
        conversion_metadata: Conversion metadata of the data of the current
            call (fit or transform), as recorded by the base class
            validation; None when no column was converted.

    Returns:
        ``y`` repositioned onto ``X.index`` when that is safe to do, its
        values and index names unchanged.

    Raises:
        ValueError: If the two indices neither match nor derive from the
            same pre-conversion index.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-31', periods=2, freq='ME')
        >>> X = pd.DataFrame({'m1': [1.0, 2.0]}, index=dates.as_unit('s'))
        >>> y = pd.Series([3.0, 4.0], index=dates.as_unit('ns'))
        >>> align_target_index(X, y).index.unit
        's'
        >>> align_target_index(X, y.iloc[:1])
        Traceback (most recent call last):
            ...
        ValueError: X and y have different indices: lengths differ (2 vs 1). ...
    """
    # Cas nominal : les deux index coïncident déjà
    if X.index.equals(y.index):
        # Mêmes instants exprimés dans une autre unité : y prend les dates de
        # X, faute de quoi le jeu de travail, concaténé, passerait à la plus
        # fine des deux unités
        if datetime_units(X.index) != datetime_units(y.index):
            return y.set_axis(X.index.set_names(y.index.names))
        return y

    # Cas colonnes -> index : la conversion appliquée à X, jamais à y,
    # est réappliquée à y en réutilisant les métadonnées de conversion des
    # données de l'appel en cours (fit ou transform)
    if (
        conversion_metadata is not None
        and conversion_metadata.get('index_was_replaced')
        and conversion_metadata['original_index'].equals(y.index)
    ):
        return y.set_axis(X.index)

    # Aucun des deux cas : décompte des libellés divergents, pour que le
    # message distingue une longueur différente d'un désaccord de valeurs
    if len(X) != len(y):
        detail = f"lengths differ ({len(X)} vs {len(y)})"
    else:
        mismatches = int((~X.index.isin(y.index)).sum())
        detail = (
            f"same length ({len(X)}) but {mismatches} label(s) of X's "
            f"index are absent from y's"
        )
    raise ValueError(
        f"X and y have different indices: {detail}. y must share X's "
        f"index, or — for column-based panel/time-series data "
        f"(time_col/panel_cols) — the index X had before those columns "
        f"were converted to the index."
    )


# Lecture des unités de stockage des dates d'un index
def datetime_units(index: pd.Index) -> Tuple[Optional[str], ...]:
    """Return the storage unit of the dates of each level of an index.

    Args:
        index: Flat index or ``MultiIndex``.

    Returns:
        One entry per level: the unit (``'s'``, ``'ms'``, ``'us'`` or
        ``'ns'``) of a date level, None for any other level.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-31', periods=2, freq='ME').as_unit('s')
        >>> datetime_units(pd.MultiIndex.from_product([['FR'], dates]))
        (None, 's')
    """
    # Extraction des niveaux de l'index
    levels = index.levels if isinstance(index, pd.MultiIndex) else [index]
    # Extraction des unités associées à chaque niveau
    return tuple(
        level.unit if isinstance(level, pd.DatetimeIndex) else None
        for level in levels
    )


# -----------------------------------------------------------------------------
# Contrat d'entrée
# -----------------------------------------------------------------------------
# Contrôle du nom de niveau réservé
def check_reserved_level_name(
    X: pd.DataFrame,
    *,
    panel_cols: Union[str, Sequence[str], None],
    time_col: Optional[str],
) -> None:
    """Reject an input whose index would carry the reserved level name.

    :data:`~tsforecast.utils._constants.FREQUENCY_LEVEL_NAME` names the
    frequency level of the multi-frequency output and of the training grid.
    An input level of the same name would give the stacked index two levels
    of one name — an ambiguity for anyone addressing the level by name. The
    check does not depend on the configuration, so a ``set_params`` between
    two calls never turns an accepted input into an ambiguous output.

    Args:
        X: Input of ``fit`` or ``transform``, before any index setting.
        panel_cols: Panel columns of the imputer, set as index levels.
        time_col: Time column of the imputer, set as the date level.

    Raises:
        ValueError: If an index level, a ``panel_cols`` entry or
            ``time_col`` is named after the reserved level.

    Examples:
        >>> import pandas as pd
        >>> frame = pd.DataFrame({'frequency': ['M'], 'v': [1.0]})
        >>> check_reserved_level_name(frame, panel_cols=None, time_col=None)
        >>> check_reserved_level_name(frame, panel_cols='frequency', time_col=None)
        Traceback (most recent call last):
            ...
        ValueError: The index level name 'frequency' is reserved ...
    """
    # Noms des niveaux d'index, présents ou à venir (colonnes de panel et
    # colonne temporelle passées en index)
    names = [name for name in X.index.names if name is not None]
    panel_cols = panel_cols or []
    names += [panel_cols] if isinstance(panel_cols, str) else list(panel_cols)
    if time_col:
        names.append(time_col)
    if FREQUENCY_LEVEL_NAME in names:
        raise ValueError(
            f"The index level name {FREQUENCY_LEVEL_NAME!r} is reserved for "
            f"the frequency level of the multi-frequency output; rename that "
            f"level (or column) of X before calling fit / transform."
        )


# -----------------------------------------------------------------------------
# Fenêtres
# -----------------------------------------------------------------------------
# Fabrique du calculateur de fenêtre
def make_window_calculator(
    *,
    coverage_threshold: float,
    imputation_scope: str,
    training_scope: Optional[str],
    training_coverage_threshold: Optional[float],
) -> ImputationWindowCalculator:
    """Build a window calculator carrying the four window parameters of the imputer.

    Args:
        coverage_threshold: Coverage threshold of the prediction window.
        imputation_scope: Scope of the prediction window.
        training_scope: Scope of the training window, None to follow
            ``imputation_scope``.
        training_coverage_threshold: Coverage threshold of the training
            window, None to follow ``coverage_threshold``.

    Returns:
        Unfitted :class:`ImputationWindowCalculator`.

    Examples:
        >>> calculator = make_window_calculator(
        ...     coverage_threshold=0.75, imputation_scope='strict',
        ...     training_scope=None, training_coverage_threshold=None,
        ... )
        >>> calculator.coverage_threshold, calculator.training_scope
        (0.75, None)
    """
    # Les deux paramètres d'entraînement retombent sur ceux de prédiction
    # quand ils valent None : le calculateur porte lui-même cette règle.
    # Une seule colonne suffit à définir une fenêtre — sa propre
    # couverture : aucun minimum supérieur n'est imposé
    return ImputationWindowCalculator(
        coverage_threshold=coverage_threshold,
        imputation_scope=imputation_scope,
        training_scope=training_scope,
        training_coverage_threshold=training_coverage_threshold,
    )


# Calculateur de fenêtre configuré par les paramètres d'un imputeur
def _window_calculator_of(imputer: 'HighFrequencyImputer') -> ImputationWindowCalculator:
    """Build an unfitted calculator from the window parameters of an imputer.

    Args:
        imputer: Imputer whose four window parameters are read.

    Returns:
        Unfitted :class:`ImputationWindowCalculator`.
    """
    return make_window_calculator(
        coverage_threshold=imputer.coverage_threshold,
        imputation_scope=imputer.imputation_scope,
        training_scope=imputer.training_scope,
        training_coverage_threshold=imputer.training_coverage_threshold,
    )


# Mise en correspondance des bornes (start, end) d'une fenêtre
def zip_window_bounds(
    start: Union[pd.Timestamp, Dict[EntityKey, Optional[pd.Timestamp]], None],
    end: Union[pd.Timestamp, Dict[EntityKey, Optional[pd.Timestamp]], None],
) -> WindowBounds:
    """Pair per-entity (or scalar) start/end bounds into tuples.

    Args:
        start: Window start(s), as returned by
            ``ImputationWindowCalculator`` (scalar for a time series, dict
            keyed by entity tuple for a panel).
        end: Window end(s), same shape as ``start``.

    Returns:
        ``(start, end)`` tuple, or dict mapping entities to
        ``(start, end)``.

    Examples:
        >>> import pandas as pd
        >>> zip_window_bounds(pd.Timestamp('2021-01-31'), pd.Timestamp('2021-12-31'))
        (Timestamp('2021-01-31 00:00:00'), Timestamp('2021-12-31 00:00:00'))
        >>> zip_window_bounds({('FR',): pd.Timestamp('2021-01-31')}, {})
        {('FR',): (Timestamp('2021-01-31 00:00:00'), None)}
    """
    # Cas des données de panel : appariement par entité
    if isinstance(start, dict):
        return {entity: (start[entity], end.get(entity)) for entity in start}
    # Cas des séries temporelles
    return (start, end)
