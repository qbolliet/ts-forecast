"""Pipeline XY supportant les transformateurs qui retournent (X_transformed, y_transformed).

Ce module fournit XYPipeline, un remplacement direct de sklearn.pipeline.Pipeline
qui peut gérer les transformateurs avec les signatures suivantes :
- Standard : transform(X) -> X_t
- XY transformateurs : transform(X, y) -> (X_t, y_t)

Le pipeline est entièrement compatible avec GridSearchCV de sklearn
"""
# Importation des modules
# Modules de base
import inspect
from typing import Any
# manipulation de données
import numpy as np
import pandas as pd
# Sklearn
from sklearn.base import clone, _fit_context
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if
from sklearn.utils.metadata_routing import (
    process_routing,
    _routing_enabled,
)
from sklearn.utils.validation import check_memory, check_is_fitted
from sklearn.utils._user_interface import _print_elapsed_time


# Fonctions utilitaires pour la détection des transformateurs XY
# Vérification qu'une méthode accepte y comme paramètre
def _accepts_y_param(transformer: Any, method_name: str) -> bool:
    """Check if a method accepts y as a parameter.

    Args:
        transformer: A transformer object.
        method_name: Name of the method to check.

    Returns:
        True if the method accepts y as a non-keyword-only parameter.
    """
    # Vérification de l'existence de la méthode
    if not hasattr(transformer, method_name):
        return False
    
    # Extraction de la méthode
    method = getattr(transformer, method_name)

    # Inspection de la signature
    try:
        sig = inspect.signature(method)
    except (ValueError, TypeError):
        return False
    
    # Extraction des paramètres de la signature
    params = list(sig.parameters.values())

    # Recherche du paramètre y
    for param in params[1:]:  # Ignore self/X
        if param.name == "y":
            if param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            ):
                return True
    return False


# Fonction auxiliaire de vérification de l'existence d'un attribut sur l'estimateur final de la pipeline
def _final_estimator_has(attr: str):
    """Check if the final estimator has a given attribute."""
    def check(self):
        return (
            self.steps[-1][1] is not None
            and hasattr(self.steps[-1][1], attr)
        )
    return check


# Pipeline étendu qui supporte les transformers retournant (X, y).
# Compatible avec GridSearchCV, cross_val_score, et le metadata routing.
class XYPipeline(Pipeline):
    """Pipeline that supports transformers returning (X_transformed, y_transformed).

    This class extends sklearn's Pipeline to handle transformers that transform
    both X and y simultaneously. It maintains full compatibility with sklearn's
    API, including GridSearchCV, cross_val_score, metadata routing, and other
    meta-estimators.

    A transformer is considered an "XY transformer" if its transform method:
    1. Accepts both X and y as arguments
    2. Returns a tuple (X_transformed, y_transformed) when y is provided

    Standard sklearn transformers (transform(X) -> X_t) work normally.

    Args:
        steps: List of (name, transform) tuples that are chained in sequential
            order. The last step can be any estimator.
        memory: Used to cache fitted transformers. By default, no caching.
        verbose: If True, the fitting time of each step will be printed.
        preserve_dataframe: If True, preserve pandas DataFrame structure through
            transformations when possible. Default is True.

    Attributes:
        named_steps: Access the steps by name.
        classes_: The classes labels (only if final estimator is a classifier).
        n_features_in_: Number of features seen during fit.
        feature_names_in_: Names of features seen during fit.

    Example:
        >>> from sklearn.preprocessing import StandardScaler
        >>> from tsforecast.xy import XYPipeline
        >>>
        >>> class LogTransformXY(BaseEstimator, TransformerMixin):
        ...     def fit(self, X, y=None):
        ...         return self
        ...     def transform(self, X, y=None):
        ...         X_t = np.log1p(X)
        ...         if y is not None:
        ...             return X_t, np.log1p(y)
        ...         return X_t
        ...     def fit_transform(self, X, y=None):
        ...         return self.fit(X, y).transform(X, y)
        ...
        >>> pipe = XYPipeline([
        ...     ('scaler', StandardScaler()),
        ...     ('log_xy', LogTransformXY()),
        ... ])
        >>> X_t, y_t = pipe.fit_transform(X, y)

    Notes:
        - XY transformers MUST implement fit_transform to properly pass y.
        - When calling transform(X) without y, XY transformers receive y=None.
        - inverse_transform works in reverse order and handles both X and y.
        - Metadata routing is supported (sklearn >= 1.4 required).
    """

    # Initialisation du pipeline
    def __init__(
        self,
        steps: list,
        *,
        transform_input=None,
        memory=None,
        verbose: bool = False,
        preserve_dataframe: bool = True,
    ):
        """Initialize the XY pipeline.

        Args:
            steps: List of (name, transform) tuples.
            memory: Caching directory or joblib.Memory object.
            verbose: If True, print fitting times.
            preserve_dataframe: If True, preserve pandas DataFrame structure when possible.
        """
        # Initialisation du parent
        super().__init__(steps, memory=memory, transform_input=transform_input, verbose=verbose)
        # Instanciation des attributs
        self.preserve_dataframe = preserve_dataframe

    # Sauvegarde des métadonnées pandas pour restauration ultérieure
    def _save_pandas_metadata(self, X, y=None):
        """Save pandas metadata for later restoration.

        Args:
            X: Input features.
            y: Input targets.
        """
        # Détection du type de X
        self._X_is_dataframe = isinstance(X, pd.DataFrame)
        self._y_is_series = isinstance(y, pd.Series)

        # Sauvegarde des métadonnées de X
        if self._X_is_dataframe:
            self._X_columns = X.columns.tolist()
            self._X_index = X.index.copy()
        else:
            self._X_columns = None
            self._X_index = None

        # Sauvegarde des métadonnées de y
        if self._y_is_series:
            self._y_name = y.name
            self._y_index = y.index.copy()
        else:
            self._y_name = None
            self._y_index = None

    # Restauration de la structure pandas après transformation
    def _restore_pandas_output(self, X, y=None, restore_X=True, restore_y=True):
        """Restore pandas structure to outputs if applicable.

        Args:
            X: Transformed features.
            y: Transformed targets.
            restore_X: Whether to restore X structure.
            restore_y: Whether to restore y structure.

        Returns:
            Restored X, y (or just X if y is None).
        """
        # Vérification que la structure de DataFrame doit être préservée
        if not self.preserve_dataframe:
            return (X, y) if y is not None else X

        # Initialisation des output à transformer
        X_out = X
        y_out = y

        # Restauration de X comme DataFrame si l'original l'était
        if restore_X and self._X_is_dataframe and isinstance(X, np.ndarray):
            X_out = pd.DataFrame(X)
            # Restauration de l'index si la taille correspond
            if self._X_index is not None and len(X) == len(self._X_index):
                X_out.index = self._X_index
            # Restauration des colonnes si la taille correspond
            if self._X_columns is not None and X.shape[1] == len(self._X_columns):
                X_out.columns = self._X_columns

        # Restauration de y comme Series si l'original l'était
        if restore_y and y is not None and self._y_is_series:
            if isinstance(y, np.ndarray) and y.ndim == 1:
                y_out = pd.Series(y, name=self._y_name)
                if self._y_index is not None and len(y) == len(self._y_index):
                    y_out.index = self._y_index
        # Retourne X et y si y est spécifié
        if y is not None:
            return X_out, y_out
        # Retourne X sinon
        return X_out

    # Méthode d'entrainement de l'estimateur et de transformation des données
    def _fit(self, X, y=None, routed_params=None, raw_params=None):
        """Fit the pipeline and transform X (and optionally y).

        Args:
            X: Training data.
            y: Training targets.
            routed_params: Parameters routed to steps.
            raw_params: Raw parameters passed by user (for transform_input).

        Returns:
            Tuple of (X_transformed, y_transformed).
        """
        # Copie et validation des étapes
        self.steps = list(self.steps)
        self._validate_steps()

        # Instanciation du cache mémoire
        memory = check_memory(self.memory)

        # Sauvegarde des métadonnées pandas
        self._save_pandas_metadata(X, y)

        if routed_params is None:
            routed_params = {}

        # Initialisation des X et y transformés à leurs valeurs initiales
        Xt, yt = X, y

        # Itération sur les steps intermédiaires
        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            # Ignore l'étape de transformation si elle n'est pas spécifiée
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            # Clonage conditionnel
            # Ne clone pas si le caching est désactivé pour préserver la compatibilité
            if hasattr(memory, "location") and memory.location is None:
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)

            # Récupération des paramètres pour cette étape
            step_params = self._get_metadata_for_step(
                step_idx=step_idx,
                step_params=routed_params.get(name, {}),
                all_params=raw_params,
            )

            # Fit et transformation
            result, fitted_transformer = self._fit_transform_step(
                cloned_transformer,
                Xt,
                yt,
                weight=None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                params=step_params,
            )

            # Gestion du résultat (tuple ou valeur simple)
            if isinstance(result, tuple) and len(result) == 2:
                Xt, yt = result
            else:
                Xt = result

            # Mise à jour du step avec le transformateur fitté
            self.steps[step_idx] = (name, fitted_transformer)

        return Xt, yt

    # Fit et transformation d'un step individuel
    def _fit_transform_step(
        self, transformer, X, y, weight=None,
        message_clsname="", message=None, params=None
    ):
        """Fit and transform a single step, handling XY transformers.

        Args:
            transformer: The transformer to fit.
            X: Input features.
            y: Input targets.
            weight: Weight to apply to result (for FeatureUnion compatibility).
            message_clsname: Class name for timing message.
            message: Timing message.
            params: Routed parameters dict with keys 'fit', 'transform', 'fit_transform'.

        Returns:
            Tuple (result, transformer) where result is X_t or (X_t, y_t).
        """
        # Initialisation des paramètres
        if params is None:
            params = {}

        with _print_elapsed_time(message_clsname, message):
            if hasattr(transformer, "fit_transform"):
                # Utilisation de fit_transform avec y si supporté
                if _accepts_y_param(transformer, "fit_transform") and y is not None:
                    result = transformer.fit_transform(
                        X, y, **params.get("fit_transform", {})
                    )
                else:
                    result = transformer.fit_transform(
                        X, **params.get("fit_transform", {})
                    )
            else:
                # Fallback : fit puis transform
                if _accepts_y_param(transformer, "fit") and y is not None:
                    transformer.fit(X, y, **params.get("fit", {}))
                else:
                    transformer.fit(X, **params.get("fit", {}))

                result = self._transform_step(
                    transformer, X, y, params.get("transform", {})
                )

        # Gestion du poids (compatibilité FeatureUnion)
        if weight is not None:
            if isinstance(result, tuple):
                result = (result[0] * weight, result[1])
            else:
                result = result * weight

        return result, transformer

    # Transformation d'un step individuel
    def _transform_step(self, transformer, X, y, transform_params=None):
        """Transform a single step, handling XY transformers.

        Args:
            transformer: The fitted transformer.
            X: Input features.
            y: Input targets.
            transform_params: Parameters for transform.

        Returns:
            X_transformed or (X_transformed, y_transformed).
        """
        # Initialisation des paramètres
        if transform_params is None:
            transform_params = {}

        # Transformation avec y si supporté
        if _accepts_y_param(transformer, "transform") and y is not None:
            return transformer.transform(X, y, **transform_params)

        # Transformation standard (X seulement)
        return transformer.transform(X, **transform_params)

    # Transformation inverse d'un step individuel
    def _inverse_transform_step(self, transformer, X, y, params=None):
        """Inverse transform a single step, handling XY transformers.

        Args:
            transformer: The fitted transformer.
            X: Transformed features.
            y: Transformed targets.
            params: Parameters for inverse_transform.

        Returns:
            X_original or (X_original, y_original).
        """
        # Initialisation des paramètres
        if params is None:
            params = {}

        # Transformation inverse avec y si supporté
        if _accepts_y_param(transformer, "inverse_transform") and y is not None:
            return transformer.inverse_transform(X, y, **params)

        # Transformation inverse standard
        return transformer.inverse_transform(X, **params)

    # Apprentissage du pipeline
    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None, **params):
        """Fit the pipeline.

        Fit all the transformers one after the other and sequentially transform
        the data. Finally, fit the transformed data using the final estimator.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Training targets of shape (n_samples,) or (n_samples, n_targets).
            **params: Parameters passed to the fit method of each step.
                Parameters are routed according to each step's routing
                configuration (metadata routing).

        Returns:
            self: The fitted pipeline.
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
        Xt, yt = self._fit(X, y, routed_params, raw_params=params)

        # Fit de l'estimateur final avec verbose
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                # Extraction des paramètres de la dernière étape
                last_step_params = self._get_metadata_for_step(
                    step_idx=len(self) - 1,
                    step_params=routed_params[self.steps[-1][0]],
                    all_params=params,
                )

                # Entraînement avec y si supporté
                if _accepts_y_param(self._final_estimator, "fit") and yt is not None:
                    self._final_estimator.fit(Xt, yt, **last_step_params["fit"])
                else:
                    self._final_estimator.fit(Xt, **last_step_params["fit"])

        return self

    # Apprentissage et transformation combinés
    @available_if(Pipeline._can_fit_transform)
    @_fit_context(prefer_skip_nested_validation=False)
    def fit_transform(self, X, y=None, **params):
        """Fit and transform the data.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Training targets of shape (n_samples,) or (n_samples, n_targets).
            **params: Parameters passed to fit_transform of each step.

        Returns:
            X_transformed if y was not transformed by any step.
            (X_transformed, y_transformed) if at least one step transformed y.
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
        Xt, yt = self._fit(X, y, routed_params, raw_params=params)

        # Suivi de la transformation de y
        y_was_transformed = (y is not None) and (yt is not y)

        # Traitement du dernier step
        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                pass
            elif hasattr(last_step, "fit_transform") or hasattr(last_step, "transform"):
                # Extraction des paramètres de la dernière étape
                last_step_params = self._get_metadata_for_step(
                    step_idx=len(self) - 1,
                    step_params=routed_params[self.steps[-1][0]],
                    all_params=params,
                )

                # Fit et transformation du dernier step (timing déjà géré dans _fit_transform_step)
                # Mais ici on utilise le timing externe pour cohérence avec l'original
                if _accepts_y_param(last_step, "fit_transform") and yt is not None:
                    result = last_step.fit_transform(
                        Xt, yt, **last_step_params.get("fit_transform", {})
                    )
                elif hasattr(last_step, "fit_transform"):
                    result = last_step.fit_transform(
                        Xt, **last_step_params.get("fit_transform", {})
                    )
                else:
                    # fit puis transform
                    if _accepts_y_param(last_step, "fit") and yt is not None:
                        last_step.fit(Xt, yt, **last_step_params.get("fit", {}))
                    else:
                        last_step.fit(Xt, **last_step_params.get("fit", {}))
                    result = self._transform_step(
                        last_step, Xt, yt, last_step_params.get("transform", {})
                    )

                # Gestion du résultat
                if isinstance(result, tuple) and len(result) == 2:
                    Xt, yt = result
                    y_was_transformed = True
                else:
                    Xt = result
            else:
                # Estimateur final sans transform (fit seulement)
                last_step_params = self._get_metadata_for_step(
                    step_idx=len(self) - 1,
                    step_params=routed_params[self.steps[-1][0]],
                    all_params=params,
                )
                if _accepts_y_param(last_step, "fit") and yt is not None:
                    last_step.fit(Xt, yt, **last_step_params.get("fit", {}))
                else:
                    last_step.fit(Xt, **last_step_params.get("fit", {}))

        # Restauration de la structure pandas si nécessaire
        if y_was_transformed and yt is not None:
            return self._restore_pandas_output(Xt, yt)
        return self._restore_pandas_output(Xt, None)

    # Transformation des données
    @available_if(Pipeline._can_transform)
    def transform(self, X, y=None, **params):
        """Transform the data.

        Args:
            X: Data to transform of shape (n_samples, n_features).
            y: Targets to transform, optional.
            **params: Parameters passed to transform of each step (requires
                metadata routing to be enabled).

        Returns:
            X_transformed if y is None or no step transforms y.
            (X_transformed, y_transformed) if y is provided and transformed.
        """
        # Vérification que le pipeline est fitted
        check_is_fitted(self)

        # Routage des paramètres via process_routing
        routed_params = process_routing(self, "transform", **params)

        # Initialisation des éléments transformés
        Xt = X
        yt = y
        y_was_transformed = False

        # Itération sur tous les steps
        for _, name, transformer in self._iter():
            # Vérification que le transformer est spécifié
            if transformer is None or transformer == "passthrough":
                continue
            # Extraction des paramètres de transformation
            transform_params = routed_params[name].transform
            # Transformation
            result = self._transform_step(transformer, Xt, yt, transform_params)

            # Gestion du résultat
            if isinstance(result, tuple) and len(result) == 2:
                Xt, yt = result
                y_was_transformed = True
            else:
                Xt = result

        # Restauration de la structure pandas
        if y is not None and y_was_transformed:
            return self._restore_pandas_output(Xt, yt)
        return self._restore_pandas_output(Xt, None)

    # Transformation inverse des données
    @available_if(Pipeline._can_inverse_transform)
    def inverse_transform(self, X, y=None, **params):
        """Inverse transform the data.

        Args:
            X: Data to inverse transform.
            y: Targets to inverse transform, optional.
            **params: Parameters passed to inverse_transform of each step.

        Returns:
            X_inverse if y is None or no step inverse transforms y.
            (X_inverse, y_inverse) if y is provided and inverse transformed.
        """
        # Vérification que le pipeline est fitted
        check_is_fitted(self)

        # Routage des paramètres via process_routing
        routed_params = process_routing(self, "inverse_transform", **params)

        # Initialisation des éléments transformés
        Xt = X
        yt = y
        y_was_transformed = False

        # Itération sur les steps en ordre inverse
        reverse_steps = reversed(list(self._iter()))

        # Parcours des transformers
        for step_idx, name, transformer in reverse_steps:
            # Vérification que le transformer est spécifié
            if transformer is None or transformer == "passthrough":
                continue

            # Extraction des paramètres de la transformation inverse
            inv_params = routed_params[name].inverse_transform
            # Transformation inverse
            result = self._inverse_transform_step(transformer, Xt, yt, inv_params)

            # Gestion du résultat
            if isinstance(result, tuple) and len(result) == 2:
                Xt, yt = result
                y_was_transformed = True
            else:
                Xt = result

        if y is not None and y_was_transformed:
            return Xt, yt
        return Xt

    # Transformation et prédiction
    @available_if(_final_estimator_has("predict"))
    def predict(self, X, y=None, **params):
        """Transform and predict.

        Args:
            X: Data to transform and predict.
            y: Optional y for XY transformers during transform.
            **params: Parameters passed to predict of the final estimator.

        Returns:
            Predictions from the final estimator.
        """
        # Vérification que le pipeline est fitted
        check_is_fitted(self)

        # Routage des paramètres via process_routing
        routed_params = process_routing(self, "predict", **params)

        # Initialisation des éléments transformés
        Xt = X
        yt = y

        # Transformation des steps intermédiaires
        for step_idx, name, transformer in self._iter(with_final=False):
            # Vérification que le transformer est spécifié
            if transformer is None or transformer == "passthrough":
                continue

            # Extraction des paramètres de transformation
            transform_params = routed_params[name].transform
            # Transformation
            result = self._transform_step(transformer, Xt, yt, transform_params)

            # Gestion du résultat
            if isinstance(result, tuple) and len(result) == 2:
                Xt, yt = result
            else:
                Xt = result

        # Prédiction avec l'estimateur final
        return self.steps[-1][1].predict(
            Xt, **routed_params[self.steps[-1][0]].predict
        )

    # Prédiction de probabilités
    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X, y=None, **params):
        """Transform and predict_proba.

        Args:
            X: Data to transform and predict.
            y: Optional y for XY transformers during transform.
            **params: Parameters passed to predict_proba of the final estimator.

        Returns:
            Probability predictions from the final estimator.
        """
        # Vérification que le pipeline est fitted
        check_is_fitted(self)

        # Routage des paramètres via process_routing
        routed_params = process_routing(self, "predict_proba", **params)

        # Initialisation des éléments transformés
        Xt = X
        yt = y

        # Transformation des steps intermédiaires
        for step_idx, name, transformer in self._iter(with_final=False):
            # Vérification que le transformer est spécifié
            if transformer is None or transformer == "passthrough":
                continue
            # Extraction des paramètres de transformation
            transform_params = routed_params[name].transform
            # Transformation
            result = self._transform_step(transformer, Xt, yt, transform_params)

            # Gestion du résultat
            if isinstance(result, tuple) and len(result) == 2:
                Xt, yt = result
            else:
                Xt = result

        # Prédiction avec l'estimateur final
        return self.steps[-1][1].predict_proba(
            Xt, **routed_params[self.steps[-1][0]].predict_proba
        )

    # Fonction de décision
    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X, y=None, **params):
        """Transform and decision_function.

        Args:
            X: Data to transform.
            y: Optional y for XY transformers during transform.
            **params: Parameters passed to decision_function of the final estimator.

        Returns:
            Decision function output from the final estimator.
        """
        # Vérification que le pipeline est fitted
        check_is_fitted(self)

        # Routage des paramètres via process_routing
        routed_params = process_routing(self, "decision_function", **params)

        # Initialisation des éléments transformés
        Xt = X
        yt = y

        # Transformation des steps intermédiaires
        for step_idx, name, transformer in self._iter(with_final=False):
            # Vérification que le transformer est spécifié
            if transformer is None or transformer == "passthrough":
                continue
            # Extraction des paramètres de transformation
            transform_params = routed_params[name].transform
            # Transformation
            result = self._transform_step(transformer, Xt, yt, transform_params)

            # Gestion du résultat
            if isinstance(result, tuple) and len(result) == 2:
                Xt, yt = result
            else:
                Xt = result

        # Fonction de décision avec l'estimateur final
        return self.steps[-1][1].decision_function(
            Xt, **routed_params[self.steps[-1][0]].decision_function
        )

    # Calcul du score
    @available_if(_final_estimator_has("score"))
    def score(self, X, y=None, sample_weight=None, **params):
        """Transform and score.

        Args:
            X: Data to transform and score.
            y: Targets for scoring.
            sample_weight: Sample weights for scoring.
            **params: Parameters passed to score of the final estimator.

        Returns:
            Score from the final estimator.
        """
        # Vérification que le pipeline est fitted
        check_is_fitted(self)

        # Routage des paramètres via process_routing
        routed_params = process_routing(
            self, "score", sample_weight=sample_weight, **params
        )

        # Initialisation des éléments transformés
        Xt = X
        yt = y

        # Transformation des steps intermédiaires
        for step_idx, name, transformer in self._iter(with_final=False):
            # Vérification que le transformer est spécifié
            if transformer is None or transformer == "passthrough":
                continue

            # Extraction des paramètres de transformation
            transform_params = routed_params[name].transform
            # Transformation
            result = self._transform_step(transformer, Xt, yt, transform_params)

            # Gestion du résultat
            if isinstance(result, tuple) and len(result) == 2:
                Xt, yt = result
            else:
                Xt = result

        # Scoring avec l'estimateur final
        return self.steps[-1][1].score(
            Xt, yt, **routed_params[self.steps[-1][0]].score
        )

    # Prédiction des log-probabilités
    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, y=None, **params):
        """Transform and predict_log_proba.

        Args:
            X: Data to transform and predict.
            y: Optional y for XY transformers during transform.
            **params: Parameters passed to predict_log_proba of the final estimator.

        Returns:
            Log probability predictions from the final estimator.
        """
        # Vérification que le pipeline est fitted
        check_is_fitted(self)

        # Routage des paramètres via process_routing
        routed_params = process_routing(self, "predict_log_proba", **params)

        # Initialisation des éléments transformés
        Xt = X
        yt = y

        # Transformation des steps intermédiaires
        for step_idx, name, transformer in self._iter(with_final=False):
            # Vérification que le transformer est spécifié
            if transformer is None or transformer == "passthrough":
                continue
            # Extraction des paramètres de transformation
            transform_params = routed_params[name].transform
            # Transformation
            result = self._transform_step(transformer, Xt, yt, transform_params)

            # Gestion du résultat
            if isinstance(result, tuple) and len(result) == 2:
                Xt, yt = result
            else:
                Xt = result

        # Prédiction avec l'estimateur final
        return self.steps[-1][1].predict_log_proba(
            Xt, **routed_params[self.steps[-1][0]].predict_log_proba
        )

    # Fit et prédiction combinés
    @available_if(_final_estimator_has("fit_predict"))
    @_fit_context(prefer_skip_nested_validation=False)
    def fit_predict(self, X, y=None, **params):
        """Fit the pipeline and predict with the final estimator.

        Args:
            X: Training data of shape (n_samples, n_features).
            y: Training targets of shape (n_samples,) or (n_samples, n_targets).
            **params: Parameters passed to fit_predict of the final estimator.

        Returns:
            Predictions from the final estimator.
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
        Xt, yt = self._fit(X, y, routed_params, raw_params=params)

        # Extraction des paramètres du dernier step
        last_step_params = routed_params[self.steps[-1][0]]

        # Fit et prédiction avec l'estimateur final
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            y_pred = self._final_estimator.fit_predict(
                Xt, yt, **last_step_params.get("fit_predict", {})
            )

        return y_pred

    # Score des échantillons
    @available_if(_final_estimator_has("score_samples"))
    def score_samples(self, X, y=None):
        """Transform and score_samples with the final estimator.

        Args:
            X: Data to transform and score.
            y: Optional y for XY transformers during transform.

        Returns:
            Score samples from the final estimator.
        """
        # Vérification que le pipeline est fitted
        check_is_fitted(self)

        # Initialisation des éléments transformés
        Xt = X
        yt = y

        # Transformation des steps intermédiaires
        for step_idx, name, transformer in self._iter(with_final=False):
            # Vérification que le transformer est spécifié
            if transformer is None or transformer == "passthrough":
                continue
            # Transformation
            result = self._transform_step(transformer, Xt, yt, {})

            # Gestion du résultat
            if isinstance(result, tuple) and len(result) == 2:
                Xt, yt = result
            else:
                Xt = result

        # Score des échantillons avec l'estimateur final
        return self._final_estimator.score_samples(Xt)
