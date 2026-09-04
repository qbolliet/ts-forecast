"""Variable ordering for the ``model`` covariate strategy of ``HighFrequencyImputer2``."""
# Importation des modules
# Modules de base
import warnings
from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Literal, Mapping, Optional, Tuple, Union

# Manipulation de données
import numpy as np
import pandas as pd

# Sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator, KFold, check_cv, cross_val_score
from sklearn.utils.validation import check_is_fitted

# Utilitaires du package
from ..utils.frequency.utils import get_frequency_order

# Identifiant retourné par order() : le nom de variable pour une série
# temporelle, ou la clé de groupe (nom, fréquence) de hfi2 pour un panel dont
# les entités divergent de fréquence
VariableKey = Any


# Description d'une variable à ordonner
@dataclass(frozen=True)
class VariableSpec:
    """Ordering-relevant metadata of one variable, keyed by its ``VariableKey``.

    A ``VariableOrderer`` never re-groups entities by itself: each
    :class:`VariableSpec` already represents one imputable group as built by
    the caller (one variable name, one detected frequency, and the entities
    sharing that frequency for a panel group) — grouping heterogeneous
    entities into several same-frequency groups is the caller's
    responsibility (``hfi2``'s ``GroupKey``, see ``imputation_plan2.py``).

    Attributes:
        name: Column name of the variable in the data passed to
            :meth:`VariableOrderer.order` — the string compared for the
            alphabetical tie-break. Equal to the group's
            ``VariableKey`` for a time series.
        frequency: Detected source frequency of the variable (e.g. ``'M'``,
            ``'Q'``, ``'A'``), used by the ``'frequency'`` order and as the
            fallback score of the ``'cv'`` order.
        entities: Entity tuples backing this group, ``None`` for a time
            series. An empty tuple is treated identically to ``None``.
    """

    name: str
    frequency: str
    entities: Optional[Tuple[Any, ...]] = None


# Composant d'ordonnancement des variables
class VariableOrderer(BaseEstimator):
    """Order variables for the cascading imputation of ``HighFrequencyImputer2``.

    Two orders are available, selected by ``fit_predict_order``:

    - ``'frequency'``: lowest frequency first, then ascending entity count,
      then alphabetical variable name — deterministic tie-break.
    - ``'cv'``: variables best predicted by cross-validation first (sklearn's
      "greater is better" convention), scores tied (``NaN`` included) broken
      alphabetically. A variable with too few scorable observations
      (``min_cv_train_size``) falls back to the ``'frequency'`` group instead.

    **Restricted scope**: this ordering only matters — and is only
    meant to be invoked by the caller — under ``covariate_strategy='model'``.
    Under ``tolerate_nan``: no values are imputed, so there is nothing to
    order. Under ``interpolate``: imputation is based solely on the
    observations specific to each variable. In both cases, the
    training set and the prediction set for a given variable do not depend
    on any prior imputation of another variable: the order of
    processing—that of the input columns—therefore has no effect on the
    values produced.

    Attributes:
        cv_: Cross-validator resolved from ``cv`` at :meth:`fit`
            (``sklearn.model_selection.check_cv`` contract). ``None`` and an
            ``int`` build a ``KFold(shuffle=True, random_state=42)`` instead
            of ``check_cv``'s unshuffled default; a splitter or an iterable
            of splits is resolved through ``check_cv`` unchanged.

    Examples:
        >>> import pandas as pd
        >>> orderer = VariableOrderer(fit_predict_order='frequency').fit()
        >>> variables = {
        ...     'a1': VariableSpec(name='a1', frequency='Y'),
        ...     'a2': VariableSpec(name='a2', frequency='Y'),
        ...     'q1': VariableSpec(name='q1', frequency='Q'),
        ... }
        >>> orderer.order(variables)
        ['a1', 'a2', 'q1']
    """
    # Initialisation
    def __init__(
        self,
        fit_predict_order: Literal['frequency', 'cv'] = 'frequency',
        cv: Union[int, BaseCrossValidator, Iterable, None] = None,
        cv_scoring: Union[str, Callable] = 'neg_mean_absolute_percentage_error',
        min_cv_train_size: int = 10,
    ):
        """Initialize the VariableOrderer.

        Args:
            fit_predict_order: Order used to sort variables:
                - ``'frequency'`` (default): lowest frequency first, then
                  ascending entity count, then alphabetical name.
                - ``'cv'``: highest cross-validated score first (any value
                  accepted by ``sklearn.metrics.check_scoring`` — a registry
                  string, a ``make_scorer`` scorer, or a callable — higher is
                  better). A variable with too few scorable observations
                  falls back to the ``'frequency'`` ordering group.
                Ignored — silently, see class docstring — outside
                ``covariate_strategy='model'``.
            cv: Cross-validation splitting strategy, following sklearn's
                ``cross_val_score`` contract:
                - ``None`` (default): ``KFold(n_splits=5, shuffle=True,
                  random_state=42)``.
                - ``int``: ``KFold(n_splits=cv, shuffle=True,
                  random_state=42)``.
                - a cross-validation splitter (exposing ``split`` and
                  ``get_n_splits``), or an iterable of ``(train, test)``
                  splits: used as-is (via ``check_cv``).
                ``shuffle=True, random_state=42`` is deliberate here — the
                goal is to produce an ORDER of variables, not an honest
                time-series evaluation of a model, and the rows being scored
                are aggregated onto a shared grid rather than left as a
                temporally ordered series. A caller who does not want this
                passes its own splitter (e.g. a ``tsforecast.crossvals``
                one).
            cv_scoring: Scoring used by ``fit_predict_order='cv'``, any value
                accepted by ``sklearn.metrics.check_scoring``.
            min_cv_train_size: Minimum number of scorable observations a
                variable must have for ``fit_predict_order='cv'`` to score it
                by cross-validation; below this it falls back to the
                ``'frequency'`` ordering group. Must be >= 1.

        Raises:
            ValueError: If ``fit_predict_order`` is not ``'frequency'`` or
                ``'cv'``; if ``cv`` is an ``int`` < 2; if ``min_cv_train_size``
                < 1.
            TypeError: If ``cv`` is neither ``None``, an ``int``, an object
                exposing ``split`` and ``get_n_splits``, nor an iterable; if
                ``cv_scoring`` is neither a string nor a callable.
        """
        # Validation à l'init : aucune transformation, seulement un
        # contrôle de forme — la résolution effective de "cv" (check_cv)
        # n'a lieu qu'au fit
        if fit_predict_order not in ('frequency', 'cv'):
            raise ValueError(
                f"fit_predict_order must be 'frequency' or 'cv', "
                f"got '{fit_predict_order}'"
            )
        self._validate_cv(cv)
        if not (isinstance(cv_scoring, str) or callable(cv_scoring)):
            raise TypeError(
                f"cv_scoring must be a str or a callable, got {type(cv_scoring).__name__}"
            )
        if min_cv_train_size < 1:
            raise ValueError(
                f"min_cv_train_size must be >= 1, got {min_cv_train_size}"
            )

        # Instanciation des attributs
        self.fit_predict_order = fit_predict_order
        self.cv = cv
        self.cv_scoring = cv_scoring
        self.min_cv_train_size = min_cv_train_size

    # Méthode auxiliaire de validation de "cv"
    @staticmethod
    def _validate_cv(cv: Any) -> None:
        """Validate the shape of ``cv``, without resolving it.

        Args:
            cv: Value received for the ``cv`` parameter.

        Raises:
            ValueError: If ``cv`` is an ``int`` < 2.
            TypeError: If ``cv`` is a string, or is none of ``None``, an
                ``int``, a splitter (``split`` + ``get_n_splits``), or an
                iterable.
        """
        # Cas de la cv par défaut
        if cv is None:
            return
        # Cas d'erreur qui pourrait être confondu avec un entier
        if isinstance(cv, bool):
            # bool est un sous-type de int en Python ; jamais une valeur de
            # "cv" sensée
            raise TypeError(f"cv must not be a bool, got {cv!r}")
        # Cas d'un entier
        if isinstance(cv, int):
            if cv < 2:
                raise ValueError(f"cv must be >= 2 when given as an int, got {cv}")
            return
        # Cas d'un cv spliter respectant l'API sklearn
        if hasattr(cv, 'split') and hasattr(cv, 'get_n_splits'):
            return
        # Une chaîne est itérable caractère par caractère : jamais une
        # collection de splits valide
        if isinstance(cv, str) or not isinstance(cv, IterableABC):
            raise TypeError(
                f"cv must be None, an int >= 2, a cross-validation splitter "
                f"(exposing split and get_n_splits), or an iterable of "
                f"splits, got {type(cv).__name__}"
            )

    # Résolution de "cv" et avertissement croisé
    def fit(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> 'VariableOrderer':
        """Resolve ``cv`` into ``cv_`` and check it against ``min_cv_train_size``.

        Args:
            X: Unused; kept for sklearn signature parity. The variables and
                data actually scored under ``fit_predict_order='cv'`` are
                passed to :meth:`order`, not here — resolving ``cv`` does not
                depend on them.
            y: Unused, same reason as ``X``.

        Returns:
            self.
        """
        del X, y

        # "None" et un entier imposent shuffle=True/random_state=42 (voir
        # docstring de "cv") : check_cv seul construirait un KFold NON
        # mélangé, donc ces deux cas sont résolus avant de lui être délégués
        if self.cv is None:
            self.cv_ = KFold(n_splits=5, shuffle=True, random_state=42)
        elif isinstance(self.cv, int):
            self.cv_ = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        else:
            # Splitter déjà construit, ou itérable de splits : utilisé tel
            # quel (check_cv l'enveloppe si besoin d'un get_n_splits)
            self.cv_ = check_cv(self.cv, y=None, classifier=False)

        # Avertissement croisé, unique, indépendant de "fit_predict_order"
        n_splits = self.cv_.get_n_splits()
        if self.min_cv_train_size < n_splits:
            warnings.warn(
                f"min_cv_train_size ({self.min_cv_train_size}) < the "
                f"effective number of CV splits ({n_splits}): cross-validation "
                f"will systematically fall back to the 'frequency' ordering "
                f"for every variable.",
                UserWarning,
            )

        return self

    # Résolution de l'estimateur d'une variable
    @staticmethod
    def _resolve_estimator(
        estimator: Union[BaseEstimator, Mapping[str, BaseEstimator], None],
        var_name: str,
    ) -> Optional[BaseEstimator]:
        """Resolve the estimator to score ``var_name`` with.

        Args:
            estimator: A single estimator (applied to every variable), a
                dict mapping variable name -> estimator (with an optional
                ``'__default__'`` fallback entry), or ``None``.
            var_name: Name of the variable being scored.

        Returns:
            A clone of the resolved estimator, or ``None`` if none applies.
        """
        # Estimateur par défaut
        if estimator is None:
            return None
        # Extraction de l'estimateur associé à la colonne
        if isinstance(estimator, Mapping):
            est = estimator.get(var_name, estimator.get('__default__'))
            return clone(est) if est is not None else None
        # Estimateur unique pour l'ensemble des colonnes
        return clone(estimator)

    # Point d'entrée principal : calcul de l'ordre des variables
    def order(
        self,
        variables: Mapping[VariableKey, VariableSpec],
        X: Optional[pd.DataFrame] = None,
        estimator: Union[BaseEstimator, Mapping[str, BaseEstimator], None] = None,
        scoring_mask: Optional[pd.Series] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> List[VariableKey]:
        """Order ``variables`` according to ``fit_predict_order``.

        Args:
            variables: Mapping of ``VariableKey`` -> :class:`VariableSpec` to
                order. The keys are returned as-is, ordered; ties are broken
                by ``VariableSpec.name``.
            X: Working data to score variables on. Its columns other than a
                variable's own ``name`` are used as its features. Required
                (and used) only when ``fit_predict_order='cv'``.
            estimator: Estimator(s) used to score variables under
                ``fit_predict_order='cv'`` — a single estimator, a dict
                mapping variable name -> estimator (optional
                ``'__default__'`` entry), or ``None``. Ignored under
                ``'frequency'``.
            scoring_mask: Boolean mask (aligned on ``X``'s index) restricting
                the rows used to build the CV scoring set — typically the
                strict imputation window (``kind='strict'``): the ordering
                must compare variables on data of homogeneous quality, or a
                variable's score would depend on the extent of its own
                extension rather than on how well it is predicted. Defaults
                to keeping every row. Ignored under ``'frequency'``.
            log: Callback invoked with a message string when every CV fold
                fails for a variable, so the fallback to the worst possible
                score is never silent. No message is emitted if ``None``
                (default). Ignored under ``'frequency'``.

        Returns:
            List of the keys of ``variables``, ordered.

        Raises:
            NotFittedError: If :meth:`fit` has not been called.
            ValueError: If ``fit_predict_order='cv'`` and ``X`` is ``None``.
        """
        # Vérification que l'estimateur est entraîné
        check_is_fitted(self, attributes=['cv_'])

        # Extraction des variables à ordonner
        var_keys = list(variables)
        if len(var_keys) <= 1:
            return var_keys

        # Ordination par fréquence
        if self.fit_predict_order == 'frequency':
            return self._order_by_frequency(variables)

        # Vérification que des données sont spécifiées pour l'ordination par validation croisée
        if X is None:
            raise ValueError("X is required when fit_predict_order='cv'")
        # Ordination par validation croisée
        return self._order_by_cv(variables, X, estimator, scoring_mask, log)

    # Ordre par fréquence (croissant en nombre d'entités) puis tie-break alphabétique
    def _order_by_frequency(
        self,
        variables: Mapping[VariableKey, VariableSpec],
    ) -> List[VariableKey]:
        """Order variables by frequency, then ascending entity count, then name.

        Args:
            variables: Mapping of ``VariableKey`` -> :class:`VariableSpec`.

        Returns:
            List of keys of ``variables``, ordered.
        """
        # Fonction de tri
        def sort_key(var_key: VariableKey) -> Tuple[float, int, str]:
            # Spécification de la clé (contient son nom, sa fréquence et ses entités)
            spec = variables[var_key]
            # Calcul du nombre d'entités
            n_entities = len(spec.entities) if spec.entities else 0
            # Fréquence la plus faible d'abord : ordre décroissant de
            # "get_frequency_order" (plus grand = fréquence plus faible),
            # obtenu par une clé croissante sur son opposé
            return (-get_frequency_order(spec.frequency), n_entities, spec.name)

        return sorted(variables, key=sort_key)

    # Ordre par validation croisée
    # /!\ Vérifier que les covariables utilisées en X pour faire la CV sont bien les seules qui seront disponibles pour prédire à la fréquence d'imputation.
    def _order_by_cv(
        self,
        variables: Mapping[VariableKey, VariableSpec],
        X: pd.DataFrame,
        estimator: Union[BaseEstimator, Mapping[str, BaseEstimator], None],
        scoring_mask: Optional[pd.Series],
        log: Optional[Callable[[str], None]],
    ) -> List[VariableKey]:
        """Order variables by descending cross-validated score.

        Args:
            variables: Mapping of ``VariableKey`` -> :class:`VariableSpec`.
            X: Working data; every column but a variable's own name is used
                as its features.
            estimator: Estimator(s) resolved per variable, see :meth:`order`.
            scoring_mask: Row mask restricting the scoring set, see
                :meth:`order`.
            log: Fold-failure logging callback, see :meth:`order`.

        Returns:
            List of keys of ``variables``, CV-scored variables first
            (highest score first), then the ``min_cv_train_size`` fallback
            group (lowest frequency first) — the two groups are sorted
            separately and never merged : their scores live on unrelated
            scales (a "scoring" value versus a frequency-order integer), so
            sorting them together would send fallback variables to an
            arbitrary rank depending on that scale.
        """
        # Lorsqu'aucun masque n'est spécifié, l'ensemble des observations sont considérées
        if scoring_mask is None:
            scoring_mask = pd.Series(True, index=X.index)

        # Initialisation des deux groupes, non comparables entre eux (voir la docstring)
        cv_scored: List[Tuple[VariableKey, float]] = []
        fallback_scored: List[Tuple[VariableKey, float]] = []

        # Parcours des clés
        for var_key in variables:
            # Extraction de la spec (qui contient le nom, la fréquence et les entités)
            spec = variables[var_key]
            # Extraction du nom de la variable
            var_name = spec.name

            # Toutes les colonnes de X sauf la variable elle-même
            feature_cols = [c for c in X.columns if c != var_name]
            # Score -inf (pire score possible, convention "greater is
            # better") si la série est univariée : elle ne doit jamais
            # passer en tête
            if not feature_cols:
                cv_scored.append((var_key, -np.inf))
                continue

            # Restriction aux lignes réellement exploitables : la cible doit
            # être observée sur les lignes du masque de scoring
            scoring_rows = scoring_mask & X[var_name].notna()
            X_sub = X.loc[scoring_rows, feature_cols]
            y_sub = X.loc[scoring_rows, var_name]

            # Fallback si moins de "min_cv_train_size" observations exploitables
            if len(X_sub) < self.min_cv_train_size:
                fallback_scored.append((var_key, get_frequency_order(spec.frequency)))
                continue

            # Extraction de l'estimateur ; -inf si non spécifié
            est = self._resolve_estimator(estimator, var_name)
            if est is None:
                cv_scored.append((var_key, -np.inf))
                continue

            # Résolution du "scoring" : lève un ValueError explicite si la
            # valeur n'est pas reconnue par sklearn
            scorer = check_scoring(est, scoring=self.cv_scoring)

            # "cross_val_score" absorbe l'échec d'un pli isolé via
            # "error_score=np.nan"
            scores = cross_val_score(
                est, X_sub, y_sub,
                cv=self.cv_,
                scoring=scorer,
                error_score=np.nan,
            )
            # Score -inf si tous les plis ont échoué, journalisé explicitement :
            # sans ce journal l'échec systématique resterait silencieux
            if np.all(np.isnan(scores)):
                if log is not None:
                    log(
                        f"[VariableOrderer] CV scoring failed on every fold "
                        f"for '{var_name}': falling back to the lowest "
                        f"possible score."
                    )
                score = -np.inf
            else:
                score = float(np.nanmean(scores))
            cv_scored.append((var_key, score))

        # Tri décroissant (convention sklearn "greater is better") au sein
        # de chaque groupe, tie-break alphabétique explicite sur le nom de
        # variable
        cv_scored.sort(key=lambda item: (-item[1], variables[item[0]].name))
        fallback_scored.sort(key=lambda item: (-item[1], variables[item[0]].name))

        return [v for v, _ in cv_scored] + [v for v, _ in fallback_scored]
