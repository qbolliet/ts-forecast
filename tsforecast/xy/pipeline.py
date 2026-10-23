"""XY pipeline: a drop-in ``sklearn.pipeline.Pipeline`` whose XY steps transform ``X`` and ``y``.

``XYPipeline`` gives the target to the steps declared as XY transformers
(``XYTransformerMixin``, ``XYPanelTimeSeriesTransformer``) and reads their output
as the pair ``(X, y)``; every other step behaves as in ``Pipeline``. Fit
parameters aligned on the rows follow the rows an XY step keeps, ``memory``
caches the fitted steps, and the pipeline works with ``GridSearchCV``,
``cross_validate`` and metadata routing.
"""
# Importation des modules
# Modules de base
from typing import Any, Iterable, Optional
# manipulation de données
import joblib
import numpy as np
import pandas as pd
# Sklearn
from sklearn.base import clone, _fit_context
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch, _safe_indexing
from sklearn.utils.metaestimators import available_if
from sklearn.utils.metadata_routing import (
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.validation import _is_arraylike_not_scalar, _num_samples, check_memory, check_is_fitted
from sklearn.utils._user_interface import _print_elapsed_time
# Module du package
from .transformers import is_xy_transformer


# Fonction auxiliaire de vérification de l'existence d'un attribut sur l'estimateur final de la pipeline
def _final_estimator_has(attr: str):
    """Check if the final estimator has a given attribute."""
    def check(self):
        return (
            self.steps[-1][1] is not None
            and hasattr(self.steps[-1][1], attr)
        )
    return check


# Refus de la cible dans les passes de prédiction : une prédiction ne dépend
# jamais de la cible, et une étape XY qui la lirait pourrait retirer des lignes
def _refuse_target(method: str, params: dict) -> None:
    """Raise if a prediction method is given a target as keyword argument.

    Args:
        method: Name of the prediction method called.
        params: Keyword arguments it received.

    Raises:
        TypeError: If ``params`` contains ``y``.

    Examples:
        >>> _refuse_target('predict', {})
        >>> _refuse_target('predict', {'y': [1.0]})
        Traceback (most recent call last):
            ...
        TypeError: XYPipeline.predict does not take y: a prediction never depends on the target; use score(X, y) to evaluate the pipeline
    """
    if "y" in params:
        raise TypeError(
            f"XYPipeline.{method} does not take y: a prediction never depends on "
            "the target; use score(X, y) to evaluate the pipeline"
        )


# Lecture de la sortie d'une étape XY qui a reçu la cible
def _xy_output(result: Any, name: str) -> tuple:
    """Return the pair ``(X_t, y_t)`` produced by an XY step given a target.

    Args:
        result: Output of the step.
        name: Name of the step, for the error message.

    Returns:
        ``result`` itself, a pair.

    Raises:
        TypeError: If ``result`` is not a tuple of length 2.

    Examples:
        >>> _xy_output(([[1.0]], [2.0]), 'step')
        ([[1.0]], [2.0])
        >>> _xy_output([[1.0]], 'step')
        Traceback (most recent call last):
            ...
        TypeError: XY step 'step' must return the pair (X, y) when it is given a target, got list
    """
    if isinstance(result, tuple) and len(result) == 2:
        return result
    raise TypeError(
        f"XY step {name!r} must return the pair (X, y) when it is given a target, "
        f"got {type(result).__name__}"
    )


# Empreinte du contenu des données pour la clé de cache : le pickle d'un objet
# pandas inclut les caches internes de son index, remplis à la première lecture
def _fingerprint(data: Any) -> Optional[str]:
    """Hash of the content of ``data``, independent of the internal caches of pandas.

    A ``DataFrame`` or ``Series`` is hashed through its values, the values and
    names of its index levels, its labels and its dtypes; anything else through
    ``joblib.hash``.

    Args:
        data: Features or target, or None.

    Returns:
        The hash, or None when ``data`` is None.

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-31', periods=3, freq='ME')
        >>> X = pd.DataFrame({'a': [1.0, 2.0, 3.0]}, index=dates)
        >>> before = _fingerprint(X)
        >>> _ = X.index.is_normalized  # fills an internal cache of the index
        >>> _fingerprint(X) == before
        True
    """
    if data is None:
        return None
    if isinstance(data, (pd.DataFrame, pd.Series)):
        index = data.index
        levels = [index.get_level_values(level).to_numpy() for level in range(index.nlevels)]
        if isinstance(data, pd.DataFrame):
            labels, dtypes = list(data.columns), [repr(dtype) for dtype in data.dtypes]
        else:
            labels, dtypes = data.name, [repr(data.dtype)]
        return joblib.hash((
            type(data).__name__, data.to_numpy(), levels, list(index.names),
            getattr(index, "freqstr", None), labels, dtypes,
        ))
    return joblib.hash(data)


# Ajustement d'une étape et transformation de son entrée : unité mise en cache
# par memory, dont la clé lit l'empreinte des données au lieu de X et y
def _fit_transform_one(transformer, X, y, fingerprint=None, params=None, message_clsname="", message=None):
    """Fit one step on ``(X, y)`` and transform ``X`` (and ``y`` for an XY step).

    The call is the one of ``Pipeline``: ``fit_transform(X, y)``, or
    ``fit(X, y).transform(X)`` for a step without ``fit_transform``. Cached by
    ``memory`` with ``X``, ``y`` and the messages ignored: ``fingerprint``
    stands for the data in the cache key.

    Args:
        transformer: Step to fit (a clone when caching).
        X: Input features.
        y: Input target, or None.
        fingerprint: Hash of ``(X, y)`` for the cache key, None without cache.
        params: Routed parameters, a dict per method.
        message_clsname: Class name of the verbose message.
        message: Verbose message, or None.

    Returns:
        Tuple ``(result, fitted_transformer)``.
    """
    params = params or {}
    with _print_elapsed_time(message_clsname, message):
        if hasattr(transformer, "fit_transform"):
            result = transformer.fit_transform(X, y, **params.get("fit_transform", {}))
        else:
            result = transformer.fit(X, y, **params.get("fit", {})).transform(
                X, **params.get("transform", {})
            )
    return result, transformer


# Positions, dans l'entrée d'une étape, des lignes de sa sortie, lues sur l'index
def _kept_positions(before: Any, after: Any) -> Optional[np.ndarray]:
    """Positions in ``before`` of the rows of ``after``, read on their index.

    Args:
        before: Input of a step.
        after: Matching output of the step.

    Returns:
        Integer positions, or None when they cannot be read (no pandas index,
        duplicated input labels, output label absent from the input).

    Examples:
        >>> import pandas as pd
        >>> _kept_positions(pd.Series([1.0, 2.0, 3.0]), pd.Series([3.0, 1.0], index=[2, 0])).tolist()
        [2, 0]
        >>> _kept_positions([1.0, 2.0], [1.0]) is None
        True
    """
    index_before = getattr(before, "index", None)
    index_after = getattr(after, "index", None)
    if not isinstance(index_before, pd.Index) or not isinstance(index_after, pd.Index):
        return None
    if not index_before.is_unique:
        return None
    positions = index_before.get_indexer(index_after)
    return None if (positions < 0).any() else positions


# Suivi des lignes gardées par les étapes XY et alignement des paramètres par ligne
class _RowTracker:
    """Rows of the input of a call kept so far by the XY steps.

    A parameter is aligned on the rows when it is an array-like as long as
    the input of the call (the rule sklearn applies to fit parameters in
    cross-validation); it is taken at the kept positions before reaching a
    step. The positions are read on the index of ``y``, else of ``X``. When
    the row count changes and neither index locates the kept rows, such a
    parameter cannot be aligned and is refused.

    Args:
        X: Input features of the call.
        exclude: Names of the parameters never aligned (``transform_input``).

    Examples:
        >>> import pandas as pd
        >>> X = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        >>> y = pd.Series([1.0, None, 3.0])
        >>> rows = _RowTracker(X)
        >>> rows.update('drop', X, y, X.iloc[[0, 2]], y.iloc[[0, 2]])
        >>> rows.align_dict({'sample_weight': [10.0, 20.0, 30.0], 'alpha': 0.5}, 'model')
        {'sample_weight': [10.0, 30.0], 'alpha': 0.5}
    """

    def __init__(self, X: Any, exclude: Optional[Iterable[str]] = None):
        self.n_rows = _num_samples(X)
        self.exclude = set(exclude or ())
        # Positions gardées (None : lignes intactes) et étape qui les rend introuvables
        self.positions: Optional[np.ndarray] = None
        self.lost_by: Optional[str] = None
        # Vrai dès qu'une étape XY a reçu la cible
        self.target_transformed = False

    def update(self, name: str, X: Any, y: Any, X_t: Any, y_t: Any) -> None:
        """Record the rows an XY step kept, from its input and output.

        Args:
            name: Name of the step.
            X: Features given to the step.
            y: Target given to the step.
            X_t: Features returned.
            y_t: Target returned.
        """
        self.target_transformed = True
        if self.lost_by is not None:
            return
        # Lecture sur l'index de la cible, puis sur celui des features
        step_positions = _kept_positions(y, y_t)
        if step_positions is None:
            step_positions = _kept_positions(X, X_t)
        if step_positions is None:
            # Sans index lisible, un nombre de lignes inchangé vaut lignes inchangées
            if _num_samples(X_t) != _num_samples(X):
                self.lost_by = name
            return
        if self.positions is None and np.array_equal(step_positions, np.arange(self.n_rows)):
            return
        self.positions = step_positions if self.positions is None else self.positions[step_positions]

    def align_dict(self, params: dict, owner: str) -> dict:
        """Take the row-aligned parameters of ``params`` at the kept positions.

        Args:
            params: Parameters of one method of one step.
            owner: Name of the step receiving them, for the error message.

        Returns:
            ``params`` itself when no row was dropped, an aligned copy otherwise.

        Raises:
            ValueError: If a row-aligned parameter follows a step whose kept
                rows cannot be located.
        """
        if self.positions is None and self.lost_by is None:
            return params
        aligned = {}
        for key, value in params.items():
            if key in self.exclude or not self._is_row_aligned(value):
                aligned[key] = value
                continue
            if self.lost_by is not None:
                raise ValueError(
                    f"The parameter {key!r} of step {owner!r} has one value per input row, "
                    f"but the rows kept by step {self.lost_by!r} cannot be located: give X or y "
                    "a unique pandas index so that XYPipeline can align it"
                )
            aligned[key] = _safe_indexing(value, self.positions)
        return type(params)(**aligned) if isinstance(params, Bunch) else aligned

    def align(self, step_params: dict, owner: str) -> dict:
        """Align the parameters of every method of a step (``{method: params}``).

        Args:
            step_params: Routed parameters of a step, a dict per method.
            owner: Name of the step.

        Returns:
            ``step_params`` itself when no row was dropped, an aligned copy otherwise.
        """
        if self.positions is None and self.lost_by is None:
            return step_params
        return Bunch(**{method: self.align_dict(params, owner) for method, params in step_params.items()})

    def _is_row_aligned(self, value: Any) -> bool:
        """Whether ``value`` is an array-like with one value per input row."""
        if not _is_arraylike_not_scalar(value):
            return False
        try:
            return _num_samples(value) == self.n_rows
        except TypeError:
            return False


# Pipeline étendu qui supporte les transformers retournant (X, y).
# Compatible avec GridSearchCV, cross_val_score, et le metadata routing.
class XYPipeline(Pipeline):
    """Pipeline that supports transformers returning (X_transformed, y_transformed).

    This class extends sklearn's Pipeline to handle transformers that transform
    both X and y simultaneously. It maintains full compatibility with sklearn's
    API, including GridSearchCV, cross_val_score, metadata routing, and other
    meta-estimators.

    A step is an "XY transformer" when it is declared as such: it inherits
    ``XYTransformerMixin`` or ``XYPanelTimeSeriesTransformer`` (see
    ``is_xy_transformer``). The pipeline gives it the target and reads its
    output as the pair ``(X_transformed, y_transformed)``. Every other step is
    handled exactly as by ``Pipeline``, whatever its signature or output.

    Args:
        steps: List of (name, transform) tuples that are chained in sequential
            order. The last step can be any estimator.
        memory: Used to cache fitted transformers. By default, no caching.
        verbose: If True, the fitting time of each step will be printed.
        preserve_dataframe: If True, an array output of a pandas input gets the
            pandas structure back, with the labels of the call inputs (see
            ``fit_transform`` and ``transform``). Default is True.

    Attributes:
        named_steps: Access the steps by name.
        classes_: The classes labels (only if final estimator is a classifier).
        n_features_in_: Number of features seen during fit.
        feature_names_in_: Names of features seen during fit.

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.base import BaseEstimator
        >>> from sklearn.preprocessing import MinMaxScaler
        >>> from tsforecast.xy import XYPipeline, XYTransformerMixin
        >>>
        >>> class LogTransformXY(BaseEstimator, XYTransformerMixin):
        ...     def _fit(self, X, y):
        ...         return self
        ...     def _transform(self, X, y=None):
        ...         X_t = np.log1p(X)
        ...         if y is not None:
        ...             return X_t, np.log1p(y)
        ...         return X_t
        ...
        >>> X = pd.DataFrame({'x': [0.0, 1.0, 2.0]})
        >>> y = pd.Series([0.0, 1.0, 3.0], name='y')
        >>> pipe = XYPipeline([
        ...     ('scaler', MinMaxScaler()),
        ...     ('log_xy', LogTransformXY()),
        ... ])
        >>> X_t, y_t = pipe.fit_transform(X, y)
        >>> print(*[round(value, 4) for value in y_t])
        0.0 0.6931 1.3863
        >>> list(X_t.columns)
        ['x']

    Notes:
        - When calling transform(X) without y, XY transformers receive X only.
        - The prediction methods (``predict``, ``predict_proba``,
          ``predict_log_proba``, ``decision_function``, ``score_samples``) take
          no y, like those of ``Pipeline``: the steps receive X only, so a step
          dropping the rows without target never drops a prediction.
        - A fit (or ``score``) parameter with one value per input row, such as
          ``sample_weight``, follows the rows an XY step keeps, located on the
          index of ``y`` or ``X``.
        - inverse_transform works in reverse order and handles both X and y.
        - Inside the pipeline, an array output of a step stays an array: the
          final estimator learns no column names, whereas a slice
          ``pipe[:-1]`` gives its output the pandas structure back. Use
          ``set_output(transform='pandas')`` for pandas outputs end to end.
        - Metadata routing is supported (sklearn >= 1.4 required).
    """

    # Initialisation du pipeline
    def __init__(
        self,
        steps: list,
        *,
        transform_input=None,
        memory: Any = None,
        verbose: bool = False,
        preserve_dataframe: bool = True,
    ):
        """Initialize the XY pipeline.

        Args:
            steps: List of (name, transform) tuples.
            transform_input: Names of the metadata transformed by the preceding
                steps before reaching a step (requires metadata routing).
            memory: Caching directory or joblib.Memory object.
            verbose: If True, print fitting times.
            preserve_dataframe: If True, preserve pandas DataFrame structure when possible.
        """
        # Initialisation du parent
        super().__init__(steps, memory=memory, transform_input=transform_input, verbose=verbose)
        # Instanciation des attributs
        self.preserve_dataframe = preserve_dataframe

    # Accès aux étapes : une tranche garde aussi preserve_dataframe
    def __getitem__(self, ind):
        """Return a step (by position or name) or a sub-pipeline (slice).

        A slice is built as by ``Pipeline`` (same ``memory`` and ``verbose``)
        and keeps ``preserve_dataframe``.

        Args:
            ind: Position, name or slice of steps.

        Returns:
            The step, or an ``XYPipeline`` of the sliced steps.

        Raises:
            ValueError: If the slice has a step other than 1.
        """
        item = super().__getitem__(ind)
        if isinstance(ind, slice):
            item.preserve_dataframe = self.preserve_dataframe
        return item

    # Restauration de la structure pandas des sorties d'un appel
    def _restore_pandas_output(self, X, y, Xt, yt=None):
        """Give the array outputs of a call the pandas structure of its inputs.

        An array output of a ``DataFrame`` input becomes a ``DataFrame``; a 1-D
        array target of a ``Series`` input becomes a ``Series`` with its name.
        The labels come from the inputs of this very call:

        - rows: the index of the input when the row count is unchanged; once
          rows are dropped, the index of the other pandas output of the same
          length, provided the input ``X`` and ``y`` share their index; positions
          ``0..n-1`` otherwise;
        - columns: the names given by ``get_feature_names_out`` when every step
          provides it, the input columns when the width is unchanged, positions
          ``0..n-1`` otherwise.

        Args:
            X: Features given to the call.
            y: Target given to the call, or None.
            Xt: Transformed features.
            yt: Transformed target, or None to return ``Xt`` alone.

        Returns:
            ``Xt`` when ``yt`` is None, the pair ``(Xt, yt)`` otherwise.
        """
        # Vérification que la structure de DataFrame doit être préservée
        if not self.preserve_dataframe:
            return Xt if yt is None else (Xt, yt)

        # Entrées appariées par l'index : les étiquettes de y sont celles de X
        aligned = (
            isinstance(X, pd.DataFrame)
            and isinstance(y, (pd.Series, pd.DataFrame))
            and y.index.equals(X.index)
        )

        # Restauration de X comme DataFrame si l'entrée l'était
        X_out = Xt
        if isinstance(X, pd.DataFrame) and isinstance(Xt, np.ndarray) and Xt.ndim == 2:
            partner = yt if aligned and isinstance(yt, (pd.Series, pd.DataFrame)) else None
            X_out = pd.DataFrame(
                Xt,
                index=self._row_labels(X.index, len(Xt), partner),
                columns=self._output_columns(X, Xt.shape[1]),
            )

        # Restauration de y comme Series si l'entrée l'était
        y_out = yt
        if isinstance(y, pd.Series) and isinstance(yt, np.ndarray) and yt.ndim == 1:
            partner = X_out if aligned and isinstance(X_out, pd.DataFrame) else None
            y_out = pd.Series(yt, index=self._row_labels(y.index, len(yt), partner), name=y.name)

        # Retourne X et y si y est spécifié, X sinon
        if yt is None:
            return X_out
        return X_out, y_out

    # Étiquettes de lignes d'une sortie restaurée
    @staticmethod
    def _row_labels(index: pd.Index, n_rows: int, partner: Optional[Any] = None) -> Optional[pd.Index]:
        """Row labels of a restored output.

        Args:
            index: Index of the matching input.
            n_rows: Row count of the output.
            partner: Other pandas output of the call, whose index is borrowed
                when ``index`` no longer fits (rows dropped), or None.

        Returns:
            ``index`` when the row count is unchanged, the index of ``partner``
            when its length fits, None (positions) otherwise.
        """
        # Restitution d'un index sur la base de l'égalité des nombres de lignes
        if len(index) == n_rows:
            return index
        if partner is not None and len(partner) == n_rows:
            return partner.index
        return None

    # Noms de colonnes d'une sortie restaurée
    def _output_columns(self, X: pd.DataFrame, n_columns: int) -> Optional[list]:
        """Column names of a restored ``X`` output.

        Args:
            X: Features given to the call.
            n_columns: Width of the output.

        Returns:
            The names given by ``get_feature_names_out`` when every step
            provides them, the columns of ``X`` when the width is unchanged,
            None (positions) otherwise.
        """
        # Noms produits par les étapes, quand toutes savent les donner
        try:
            names = list(self.get_feature_names_out())
        except (AttributeError, NotFittedError, TypeError, ValueError):
            names = None
        if names is not None and len(names) == n_columns:
            return names
        # Repli : largeur inchangée, colonnes d'entrée conservées
        if X.shape[1] == n_columns:
            return list(X.columns)
        return None

    # Méthode d'entrainement de l'estimateur et de transformation des données
    def _fit(self, X, y=None, routed_params=None, raw_params=None):
        """Fit the intermediate steps and transform X (and y through the XY steps).

        Args:
            X: Training data.
            y: Training targets.
            routed_params: Parameters routed to steps.
            raw_params: Raw parameters passed by user (for transform_input).

        Returns:
            Tuple ``(X_transformed, y_transformed, rows)``, ``rows`` being the
            ``_RowTracker`` of the rows kept by the XY steps.
        """
        # Copie et validation des étapes
        self.steps = list(self.steps)
        self._validate_steps()

        # Instanciation du cache mémoire ; sans emplacement, ni cache ni clonage
        memory = check_memory(self.memory)
        caching = not (hasattr(memory, "location") and memory.location is None)
        fit_transform_one_cached = memory.cache(
            _fit_transform_one, ignore=["X", "y", "message_clsname", "message"]
        )

        # Initialisation des X et y transformés et du suivi des lignes
        Xt, yt = X, y
        rows = _RowTracker(X, exclude=self.transform_input)

        # Itération sur les steps intermédiaires
        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            # Ignore l'étape de transformation si elle n'est pas spécifiée
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            # Clonage seulement avec un cache : l'objet donné reste alors non ajusté
            cloned_transformer = clone(transformer) if caching else transformer

            # Paramètres de l'étape, alignés sur les lignes encore présentes
            step_params = self._get_metadata_for_step(
                step_idx=step_idx,
                step_params=routed_params[name],
                all_params=raw_params,
            )
            step_params = rows.align(step_params, name)

            # Fit et transformation, chargés du cache quand l'étape y figure
            fingerprint = (_fingerprint(Xt), _fingerprint(yt)) if caching else None
            result, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                Xt,
                yt,
                fingerprint=fingerprint,
                params=step_params,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
            )

            # Lecture de la paire (X, y) d'une étape XY qui a reçu la cible
            if is_xy_transformer(fitted_transformer) and yt is not None:
                X_new, y_new = _xy_output(result, name)
                rows.update(name, Xt, yt, X_new, y_new)
                Xt, yt = X_new, y_new
            else:
                Xt = result

            # Mise à jour du step avec le transformateur fitté
            self.steps[step_idx] = (name, fitted_transformer)

        return Xt, yt, rows

    # Transformation par les étapes ajustées, avec la cible quand elle est fournie
    def _transform_steps(self, X, y, routed_params, with_final):
        """Transform ``X`` (and ``y``, through the XY steps) by the fitted steps, in order.

        ``'passthrough'`` and None steps are skipped by ``_iter``.

        Args:
            X: Features to transform.
            y: Targets to transform, or None: the steps then receive ``X`` only.
            routed_params: Routed parameters of the method, or None (no routing).
            with_final: Whether the last step is a transformer to apply too.

        Returns:
            Tuple ``(X_t, y_t, rows)``, ``rows`` being the ``_RowTracker`` of
            the rows kept by the XY steps (``rows.target_transformed`` tells
            whether an XY step received the target).
        """
        # Initialisation des éléments transformés et du suivi des lignes
        Xt, yt = X, y
        rows = _RowTracker(X, exclude=self.transform_input)

        # Itération sur les steps ajustés
        for _, name, transformer in self._iter(with_final=with_final):
            # Paramètres de transformation, alignés sur les lignes encore présentes
            params = {} if routed_params is None else rows.align_dict(routed_params[name].transform, name)

            # Transformation de la paire par une étape XY, de X seul sinon
            if is_xy_transformer(transformer) and yt is not None:
                X_new, y_new = _xy_output(transformer.transform(Xt, yt, **params), name)
                rows.update(name, Xt, yt, X_new, y_new)
                Xt, yt = X_new, y_new
            else:
                Xt = transformer.transform(Xt, **params)

        return Xt, yt, rows

    # Apprentissage du pipeline
    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X: Any, y: Any = None, **params: Any) -> 'XYPipeline':
        """Fit the pipeline.

        Fit all the transformers one after the other and sequentially transform
        the data. Finally, fit the transformed data using the final estimator.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Training targets of shape (n_samples,) or (n_samples, n_targets).
            **params: Parameters passed to the fit method of each step.
                Parameters are routed according to each step's routing
                configuration (metadata routing). A parameter with one value
                per row of ``X`` follows the rows kept by the XY steps.

        Returns:
            self: The fitted pipeline.

        Raises:
            ValueError: If a row-aligned parameter follows an XY step whose
                kept rows cannot be located on an index.
        """
        # Validation de transform_input (comme sklearn)
        if not _routing_enabled() and self.transform_input is not None:
            raise ValueError(
                "The `transform_input` parameter can only be set if metadata "
                "routing is enabled. You can enable metadata routing using "
                "`sklearn.set_config(enable_metadata_routing=True)`."
            )

        # Routage des paramètres
        routed_params = self._check_method_params(method="fit", props=params)

        # Fit et transformation des steps intermédiaires
        Xt, yt, rows = self._fit(X, y, routed_params, raw_params=params)

        # Fit de l'estimateur final avec verbose
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                # Extraction des paramètres de la dernière étape
                last_step_params = self._get_metadata_for_step(
                    step_idx=len(self) - 1,
                    step_params=routed_params[self.steps[-1][0]],
                    all_params=params,
                )
                last_step_params = rows.align(last_step_params, self.steps[-1][0])

                # Entraînement sur les données transformées, comme Pipeline
                self._final_estimator.fit(Xt, yt, **last_step_params["fit"])

        return self

    # Apprentissage et transformation combinés
    @available_if(Pipeline._can_fit_transform)
    @_fit_context(prefer_skip_nested_validation=False)
    def fit_transform(self, X: Any, y: Any = None, **params: Any) -> Any:
        """Fit and transform the data.

        An array output of a pandas input gets the pandas structure back when
        ``preserve_dataframe`` is True: the index of the input while the row
        count is unchanged, else the index of the pandas target returned (for
        an ``X`` and a ``y`` given with the same index), so that ``X_t`` and
        ``y_t`` stay aligned once rows are dropped.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Training targets of shape (n_samples,) or (n_samples, n_targets).
            **params: Parameters passed to fit_transform of each step.

        Returns:
            X_transformed if no XY step received y.
            (X_transformed, y_transformed) if at least one XY step received y.
        """
        # Validation de transform_input (comme sklearn)
        if not _routing_enabled() and self.transform_input is not None:
            raise ValueError(
                "The `transform_input` parameter can only be set if metadata "
                "routing is enabled. You can enable metadata routing using "
                "`sklearn.set_config(enable_metadata_routing=True)`."
            )

        # Routage des paramètres
        routed_params = self._check_method_params(method="fit_transform", props=params)

        # Fit et transformation des steps intermédiaires
        Xt, yt, rows = self._fit(X, y, routed_params, raw_params=params)
        y_was_transformed = rows.target_transformed

        # Traitement du dernier step : transformateur ou 'passthrough'
        # (la méthode n'est disponible que dans ces deux cas)
        last_name, last_step = self.steps[-1][0], self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step != "passthrough":
                # Extraction des paramètres de la dernière étape
                last_step_params = self._get_metadata_for_step(
                    step_idx=len(self) - 1,
                    step_params=routed_params[last_name],
                    all_params=params,
                )
                last_step_params = rows.align(last_step_params, last_name)

                # Fit et transformation du dernier step
                if is_xy_transformer(last_step) and yt is not None:
                    result = last_step.fit_transform(Xt, yt, **last_step_params.get("fit_transform", {}))
                    Xt, yt = _xy_output(result, last_name)
                    y_was_transformed = True
                elif hasattr(last_step, "fit_transform"):
                    Xt = last_step.fit_transform(Xt, yt, **last_step_params.get("fit_transform", {}))
                else:
                    Xt = last_step.fit(Xt, yt, **last_step_params.get("fit", {})).transform(
                        Xt, **last_step_params.get("transform", {})
                    )

        # Restauration de la structure pandas si nécessaire
        if y_was_transformed and yt is not None:
            return self._restore_pandas_output(X, y, Xt, yt)
        return self._restore_pandas_output(X, y, Xt)

    # Transformation des données
    @available_if(Pipeline._can_transform)
    def transform(self, X: Any, y: Any = None, **params: Any) -> Any:
        """Transform the data.

        An array output of a pandas input gets the pandas structure back when
        ``preserve_dataframe`` is True, with the labels of this call's ``X`` and
        ``y`` (see ``fit_transform``).

        Args:
            X: Data to transform of shape (n_samples, n_features).
            y: Targets to transform, optional.
            **params: Parameters passed to transform of each step (requires
                metadata routing to be enabled).

        Returns:
            X_transformed if y is None or no XY step received it.
            (X_transformed, y_transformed) if y is provided and an XY step received it.

        Raises:
            ValueError: If ``params`` are given while metadata routing is disabled.
        """
        # Vérification que le pipeline est fitted
        check_is_fitted(self)

        # Paramètres refusés sans routage, comme Pipeline, puis routage
        _raise_for_params(params, self, "transform")
        routed_params = process_routing(self, "transform", **params)

        # Transformation par tous les steps
        Xt, yt, rows = self._transform_steps(X, y, routed_params, with_final=True)

        # Restauration de la structure pandas
        if y is not None and rows.target_transformed:
            return self._restore_pandas_output(X, y, Xt, yt)
        return self._restore_pandas_output(X, y, Xt)

    # Transformation inverse des données
    @available_if(Pipeline._can_inverse_transform)
    def inverse_transform(self, X: Any, y: Any = None, **params: Any) -> Any:
        """Inverse transform the data.

        Args:
            X: Data to inverse transform.
            y: Targets to inverse transform, optional.
            **params: Parameters passed to inverse_transform of each step
                (requires metadata routing to be enabled).

        Returns:
            X_inverse if y is None or no XY step received it.
            (X_inverse, y_inverse) if y is provided and an XY step received it.

        Raises:
            ValueError: If ``params`` are given while metadata routing is disabled.
        """
        # Vérification que le pipeline est fitted
        check_is_fitted(self)

        # Paramètres refusés sans routage, comme Pipeline, puis routage
        _raise_for_params(params, self, "inverse_transform")
        routed_params = process_routing(self, "inverse_transform", **params)

        # Initialisation des éléments transformés
        Xt = X
        yt = y
        y_was_transformed = False

        # Parcours des transformers en ordre inverse
        for _, name, transformer in reversed(list(self._iter())):
            # Extraction des paramètres de la transformation inverse
            inv_params = routed_params[name].inverse_transform
            # Transformation inverse de la paire par une étape XY, de X seul sinon
            if is_xy_transformer(transformer) and yt is not None:
                Xt, yt = _xy_output(transformer.inverse_transform(Xt, yt, **inv_params), name)
                y_was_transformed = True
            else:
                Xt = transformer.inverse_transform(Xt, **inv_params)

        if y is not None and y_was_transformed:
            return Xt, yt
        return Xt

    # Transformation et prédiction
    @available_if(_final_estimator_has("predict"))
    def predict(self, X: Any, **params: Any) -> Any:
        """Transform and predict.

        The steps receive ``X`` only: a step that drops the rows without target
        keeps every row, so there is one prediction per row of ``X``.

        Args:
            X: Data to transform and predict.
            **params: Without metadata routing, parameters passed to the
                predict of the final estimator (``return_std=True``...); with
                routing, parameters routed to the steps.

        Returns:
            Predictions from the final estimator.

        Raises:
            TypeError: If a target ``y`` is passed.
        """
        # Refus de la cible, puis vérification que le pipeline est fitted
        _refuse_target("predict", params)
        check_is_fitted(self)

        # Sans routage : paramètres donnés à l'estimateur final, comme Pipeline
        if not _routing_enabled():
            Xt, _, _ = self._transform_steps(X, None, None, with_final=False)
            return self.steps[-1][1].predict(Xt, **params)

        # Routage des paramètres via process_routing
        routed_params = process_routing(self, "predict", **params)

        # Transformation des steps intermédiaires, sans la cible
        Xt, _, _ = self._transform_steps(X, None, routed_params, with_final=False)

        # Prédiction avec l'estimateur final
        return self.steps[-1][1].predict(
            Xt, **routed_params[self.steps[-1][0]].predict
        )

    # Prédiction de probabilités
    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X: Any, **params: Any) -> Any:
        """Transform and predict_proba.

        Args:
            X: Data to transform and predict.
            **params: Without metadata routing, parameters passed to the
                predict_proba of the final estimator; with routing, parameters
                routed to the steps.

        Returns:
            Probability predictions from the final estimator.

        Raises:
            TypeError: If a target ``y`` is passed.
        """
        # Refus de la cible, puis vérification que le pipeline est fitted
        _refuse_target("predict_proba", params)
        check_is_fitted(self)

        # Sans routage : paramètres donnés à l'estimateur final, comme Pipeline
        if not _routing_enabled():
            Xt, _, _ = self._transform_steps(X, None, None, with_final=False)
            return self.steps[-1][1].predict_proba(Xt, **params)

        # Routage des paramètres via process_routing
        routed_params = process_routing(self, "predict_proba", **params)

        # Transformation des steps intermédiaires, sans la cible
        Xt, _, _ = self._transform_steps(X, None, routed_params, with_final=False)

        # Prédiction avec l'estimateur final
        return self.steps[-1][1].predict_proba(
            Xt, **routed_params[self.steps[-1][0]].predict_proba
        )

    # Fonction de décision
    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X: Any, **params: Any) -> Any:
        """Transform and decision_function.

        Args:
            X: Data to transform.
            **params: Parameters routed to the steps (requires metadata
                routing to be enabled).

        Returns:
            Decision function output from the final estimator.

        Raises:
            TypeError: If a target ``y`` is passed.
            ValueError: If ``params`` are given while metadata routing is disabled.
        """
        # Refus de la cible, puis vérification que le pipeline est fitted
        _refuse_target("decision_function", params)
        check_is_fitted(self)

        # Paramètres refusés sans routage, comme Pipeline, puis routage
        _raise_for_params(params, self, "decision_function")
        routed_params = process_routing(self, "decision_function", **params)

        # Transformation des steps intermédiaires, sans la cible
        Xt, _, _ = self._transform_steps(X, None, routed_params, with_final=False)

        # Fonction de décision avec l'estimateur final
        return self.steps[-1][1].decision_function(
            Xt, **routed_params[self.steps[-1][0]].decision_function
        )

    # Calcul du score
    @available_if(_final_estimator_has("score"))
    def score(self, X: Any, y: Any = None, sample_weight: Any = None, **params: Any) -> float:
        """Transform and score.

        ``y`` goes through the XY steps: a step dropping the rows without
        target drops them before scoring, and ``sample_weight`` follows the
        kept rows.

        Args:
            X: Data to transform and score.
            y: Targets for scoring.
            sample_weight: Sample weights for scoring, one per row of ``X``.
            **params: Parameters routed to the steps (requires metadata
                routing to be enabled).

        Returns:
            Score from the final estimator.

        Raises:
            ValueError: If ``sample_weight`` follows an XY step whose kept rows
                cannot be located on an index.
        """
        # Vérification que le pipeline est fitted
        check_is_fitted(self)
        final_name = self.steps[-1][0]

        # Sans routage : poids donnés au score de l'estimateur final, comme Pipeline
        if not _routing_enabled():
            Xt, yt, rows = self._transform_steps(X, y, None, with_final=False)
            score_params = {} if sample_weight is None else {"sample_weight": sample_weight}
            return self.steps[-1][1].score(Xt, yt, **rows.align_dict(score_params, final_name))

        # Routage des paramètres via process_routing
        routed_params = process_routing(
            self, "score", sample_weight=sample_weight, **params
        )

        # Transformation des steps intermédiaires, avec la cible
        Xt, yt, rows = self._transform_steps(X, y, routed_params, with_final=False)

        # Scoring avec l'estimateur final, paramètres alignés sur les lignes gardées
        return self.steps[-1][1].score(
            Xt, yt, **rows.align_dict(routed_params[final_name].score, final_name)
        )

    # Prédiction des log-probabilités
    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X: Any, **params: Any) -> Any:
        """Transform and predict_log_proba.

        Args:
            X: Data to transform and predict.
            **params: Without metadata routing, parameters passed to the
                predict_log_proba of the final estimator; with routing,
                parameters routed to the steps.

        Returns:
            Log probability predictions from the final estimator.

        Raises:
            TypeError: If a target ``y`` is passed.
        """
        # Refus de la cible, puis vérification que le pipeline est fitted
        _refuse_target("predict_log_proba", params)
        check_is_fitted(self)

        # Sans routage : paramètres donnés à l'estimateur final, comme Pipeline
        if not _routing_enabled():
            Xt, _, _ = self._transform_steps(X, None, None, with_final=False)
            return self.steps[-1][1].predict_log_proba(Xt, **params)

        # Routage des paramètres via process_routing
        routed_params = process_routing(self, "predict_log_proba", **params)

        # Transformation des steps intermédiaires, sans la cible
        Xt, _, _ = self._transform_steps(X, None, routed_params, with_final=False)

        # Prédiction avec l'estimateur final
        return self.steps[-1][1].predict_log_proba(
            Xt, **routed_params[self.steps[-1][0]].predict_log_proba
        )

    # Fit et prédiction combinés
    @available_if(_final_estimator_has("fit_predict"))
    @_fit_context(prefer_skip_nested_validation=False)
    def fit_predict(self, X: Any, y: Any = None, **params: Any) -> Any:
        """Fit the pipeline and predict with the final estimator.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Training targets of shape (n_samples,) or (n_samples, n_targets).
            **params: Parameters passed to fit_predict of the final estimator.

        Returns:
            Predictions from the final estimator, one per row it was fitted on.
        """
        # Validation de transform_input (comme sklearn)
        if not _routing_enabled() and self.transform_input is not None:
            raise ValueError(
                "The `transform_input` parameter can only be set if metadata "
                "routing is enabled. You can enable metadata routing using "
                "`sklearn.set_config(enable_metadata_routing=True)`."
            )

        # Routage des paramètres
        routed_params = self._check_method_params(method="fit_predict", props=params)

        # Fit et transformation des steps intermédiaires
        Xt, yt, rows = self._fit(X, y, routed_params, raw_params=params)

        # Extraction des paramètres du dernier step, alignés sur les lignes gardées
        last_step_params = rows.align(routed_params[self.steps[-1][0]], self.steps[-1][0])

        # Fit et prédiction avec l'estimateur final
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            y_pred = self._final_estimator.fit_predict(
                Xt, yt, **last_step_params.get("fit_predict", {})
            )

        return y_pred

    # Score des échantillons
    @available_if(_final_estimator_has("score_samples"))
    def score_samples(self, X: Any) -> Any:
        """Transform and score_samples with the final estimator.

        Args:
            X: Data to transform and score.

        Returns:
            Score samples from the final estimator.
        """
        # Vérification que le pipeline est fitted
        check_is_fitted(self)

        # Transformation des steps intermédiaires, sans la cible
        Xt, _, _ = self._transform_steps(X, None, None, with_final=False)

        # Score des échantillons avec l'estimateur final
        return self._final_estimator.score_samples(Xt)
