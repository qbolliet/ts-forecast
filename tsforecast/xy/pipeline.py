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
from typing import Any, Literal
# manipulation de données
import numpy as np
import pandas as pd
# Sklearn
from sklearn import get_config
from sklearn.base import clone, TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    process_routing,
    _routing_enabled,
)


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
        y_transformer_indices_: List of step indices that transform y.

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
        - The pipeline tracks which steps transform y via y_transformer_indices_.
        - Metadata routing is supported (sklearn >= 1.4 required).
    """

    # Initialisation du pipeline
    def __init__(
        self,
        steps: list,
        *,
        memory=None,
        verbose: bool = False,
        preserve_dataframe: bool = True,
    ):
        """Initialize the XY pipeline.

        Args:
            steps: List of (name, transform) tuples.
            memory: Caching directory or joblib.Memory object.
            verbose: If True, print fitting times.
            preserve_dataframe: If True, preserve pandas DataFrame structure.
        """
        super().__init__(steps, memory=memory, verbose=verbose)
        self.preserve_dataframe = preserve_dataframe
        self._y_transformers: list[int] = []

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
        if not self.preserve_dataframe:
            return (X, y) if y is not None else X

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

        if y is not None:
            return X_out, y_out
        return X_out

    # Apprentissage et transformation des données
    def _fit(self, X, y=None, routed_params=None):
        """Fit the pipeline and transform X (and optionally y).

        Args:
            X: Training data.
            y: Training targets.
            routed_params: Parameters routed to steps.

        Returns:
            Tuple of (X_transformed, y_transformed).
        """
        self.steps = list(self.steps)
        self._validate_steps()

        # Sauvegarde des métadonnées pandas
        self._save_pandas_metadata(X, y)

        if routed_params is None:
            routed_params = {}

        # Réinitialisation de la liste des transformateurs y
        self._y_transformers = []

        Xt = X
        yt = y

        # Itération sur tous les steps sauf le dernier
        for step_idx in range(len(self.steps) - 1):
            name, transformer = self.steps[step_idx]

            # Skip des steps passthrough
            if transformer is None or transformer == "passthrough":
                continue

            # Clonage du transformateur
            cloned_transformer = clone(transformer)

            # Récupération des paramètres routés pour ce step
            fit_params = routed_params.get(name, {}).get("fit", {})
            transform_params = routed_params.get(name, {}).get("transform", {})

            # Fit et transformation
            result = self._fit_transform_step(
                cloned_transformer, Xt, yt, fit_params, transform_params, step_idx
            )

            # Gestion du résultat (tuple ou valeur simple)
            if isinstance(result, tuple) and len(result) == 2:
                Xt, yt = result
                self._y_transformers.append(step_idx)
            else:
                Xt = result

            # Mise à jour du step avec le transformateur fitté
            self.steps[step_idx] = (name, cloned_transformer)

        return Xt, yt

    # Fit et transformation d'un step individuel
    def _fit_transform_step(
        self, transformer, X, y, fit_params, transform_params, step_idx
    ):
        """Fit and transform a single step, handling XY transformers.

        Args:
            transformer: The transformer to fit.
            X: Input features.
            y: Input targets.
            fit_params: Parameters for fit.
            transform_params: Parameters for transform.
            step_idx: Index of the current step.

        Returns:
            X_transformed or (X_transformed, y_transformed).
        """
        if hasattr(transformer, "fit_transform"):
            # Tentative de fit_transform avec y
            if _accepts_y_param(transformer, "fit_transform") and y is not None:
                try:
                    all_params = {**fit_params, **transform_params}
                    result = transformer.fit_transform(X, y, **all_params)
                    return result
                except TypeError:
                    pass

            # Fallback : fit puis transform
            try:
                transformer.fit(X, y, **fit_params)
            except TypeError:
                transformer.fit(X, **fit_params)

            return self._transform_step(transformer, X, y, transform_params)
        else:
            # Pas de fit_transform, utilisation de fit puis transform
            try:
                transformer.fit(X, y, **fit_params)
            except TypeError:
                transformer.fit(X, **fit_params)

            return self._transform_step(transformer, X, y, transform_params)

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
        if transform_params is None:
            transform_params = {}

        # Tentative de transformation avec y
        if _accepts_y_param(transformer, "transform") and y is not None:
            try:
                result = transformer.transform(X, y, **transform_params)
                return result
            except TypeError:
                pass

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
        if params is None:
            params = {}

        # Tentative de transformation inverse avec y
        if _accepts_y_param(transformer, "inverse_transform") and y is not None:
            try:
                result = transformer.inverse_transform(X, y, **params)
                return result
            except TypeError:
                pass

        # Transformation inverse standard
        return transformer.inverse_transform(X, **params)

    # Apprentissage du pipeline
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
        # Routage des paramètres
        routed_params = self._route_params("fit", params)

        # Fit et transformation des steps intermédiaires
        Xt, yt = self._fit(X, y, routed_params)

        # Fit de l'estimateur final
        last_step_idx = len(self.steps) - 1
        name, final_estimator = self.steps[last_step_idx]

        if final_estimator is not None and final_estimator != "passthrough":
            cloned_final = clone(final_estimator)
            fit_params = routed_params.get(name, {}).get("fit", {})

            try:
                cloned_final.fit(Xt, yt, **fit_params)
            except TypeError:
                cloned_final.fit(Xt, **fit_params)

            self.steps[last_step_idx] = (name, cloned_final)

        return self

    # Apprentissage et transformation combinés
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
        # Routage des paramètres
        routed_params = self._route_params("fit_transform", params)

        # Fit et transformation des steps intermédiaires
        Xt, yt = self._fit(X, y, routed_params)

        # Traitement du dernier step s'il est un transformer
        last_step_idx = len(self.steps) - 1
        name, final_estimator = self.steps[last_step_idx]

        if final_estimator is not None and final_estimator != "passthrough":
            cloned_final = clone(final_estimator)
            fit_params = routed_params.get(name, {}).get("fit", {})
            transform_params = routed_params.get(name, {}).get("transform", {})

            # Vérification si c'est un transformer
            if hasattr(cloned_final, "fit_transform") or hasattr(
                cloned_final, "transform"
            ):
                result = self._fit_transform_step(
                    cloned_final, Xt, yt, fit_params, transform_params, last_step_idx
                )

                if isinstance(result, tuple) and len(result) == 2:
                    Xt, yt = result
                    self._y_transformers.append(last_step_idx)
                else:
                    Xt = result

                self.steps[last_step_idx] = (name, cloned_final)
            else:
                # Estimateur final sans transform
                try:
                    cloned_final.fit(Xt, yt, **fit_params)
                except TypeError:
                    cloned_final.fit(Xt, **fit_params)
                self.steps[last_step_idx] = (name, cloned_final)

        # Restauration de la structure pandas si nécessaire
        if self._y_transformers and yt is not None:
            return self._restore_pandas_output(Xt, yt)
        return self._restore_pandas_output(Xt, None)

    # Transformation des données
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
        # Routage des paramètres
        routed_params = self._route_params("transform", params)

        Xt = X
        yt = y
        y_was_transformed = False

        # Itération sur tous les steps
        for step_idx, name, transformer in self._iter():
            if transformer is None or transformer == "passthrough":
                continue

            transform_params = routed_params.get(name, {}).get("transform", {})
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
        # Routage des paramètres
        routed_params = self._route_params("inverse_transform", params)

        Xt = X
        yt = y
        y_was_transformed = False

        # Itération sur les steps en ordre inverse
        reverse_steps = list(self._iter())[::-1]

        for step_idx, name, transformer in reverse_steps:
            if transformer is None or transformer == "passthrough":
                continue

            # Vérification de l'existence de inverse_transform
            if not hasattr(transformer, "inverse_transform"):
                raise TypeError(
                    f"Transformer '{name}' does not have inverse_transform method."
                )

            inv_params = routed_params.get(name, {}).get("inverse_transform", {})
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
        # Routage des paramètres
        routed_params = self._route_params("predict", params)

        Xt = X
        yt = y

        # Transformation des steps intermédiaires
        for step_idx, name, transformer in self._iter(with_final=False):
            if transformer is None or transformer == "passthrough":
                continue

            transform_params = routed_params.get(name, {}).get("transform", {})
            result = self._transform_step(transformer, Xt, yt, transform_params)

            if isinstance(result, tuple) and len(result) == 2:
                Xt, yt = result
            else:
                Xt = result

        # Prédiction avec l'estimateur final
        predict_params = routed_params.get(self.steps[-1][0], {}).get("predict", {})
        return self.steps[-1][1].predict(Xt, **predict_params)

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
        # Routage des paramètres
        routed_params = self._route_params("predict_proba", params)

        Xt = X
        yt = y

        # Transformation des steps intermédiaires
        for step_idx, name, transformer in self._iter(with_final=False):
            if transformer is None or transformer == "passthrough":
                continue

            transform_params = routed_params.get(name, {}).get("transform", {})
            result = self._transform_step(transformer, Xt, yt, transform_params)

            if isinstance(result, tuple) and len(result) == 2:
                Xt, yt = result
            else:
                Xt = result

        # Prédiction avec l'estimateur final
        predict_params = routed_params.get(self.steps[-1][0], {}).get(
            "predict_proba", {}
        )
        return self.steps[-1][1].predict_proba(Xt, **predict_params)

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
        # Routage des paramètres
        routed_params = self._route_params("decision_function", params)

        Xt = X
        yt = y

        # Transformation des steps intermédiaires
        for step_idx, name, transformer in self._iter(with_final=False):
            if transformer is None or transformer == "passthrough":
                continue

            transform_params = routed_params.get(name, {}).get("transform", {})
            result = self._transform_step(transformer, Xt, yt, transform_params)

            if isinstance(result, tuple) and len(result) == 2:
                Xt, yt = result
            else:
                Xt = result

        # Decision function avec l'estimateur final
        df_params = routed_params.get(self.steps[-1][0], {}).get(
            "decision_function", {}
        )
        return self.steps[-1][1].decision_function(Xt, **df_params)

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
        # Routage des paramètres
        routed_params = self._route_params("score", params)

        Xt = X
        yt = y

        # Transformation des steps intermédiaires
        for step_idx, name, transformer in self._iter(with_final=False):
            if transformer is None or transformer == "passthrough":
                continue

            transform_params = routed_params.get(name, {}).get("transform", {})
            result = self._transform_step(transformer, Xt, yt, transform_params)

            if isinstance(result, tuple) and len(result) == 2:
                Xt, yt = result
            else:
                Xt = result

        # Scoring
        score_params = routed_params.get(self.steps[-1][0], {}).get("score", {})
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight

        return self.steps[-1][1].score(Xt, yt, **score_params)

    # Gestion du routage des paramètres
    # Routage des paramètres vers les steps
    def _route_params(self, method: str, params: dict) -> dict:
        """Route parameters to steps based on method and routing configuration.

        Uses sklearn's metadata routing system (sklearn >= 1.4 required).

        Args:
            method: Name of the method (fit, transform, etc.).
            params: Dictionary of parameters.

        Returns:
            Dictionary mapping step names to their parameters by method.
        """
        # Vérification si le metadata routing est activé
        if _routing_enabled():
            return self._route_params_metadata_routing(method, params)
        else:
            # Fallback vers le routage par syntaxe dunder
            return self._route_params_dunder(method, params)

    # Routage via metadata routing de sklearn
    def _route_params_metadata_routing(self, method: str, params: dict) -> dict:
        """Route parameters using sklearn's metadata routing system.

        Args:
            method: Name of the method.
            params: Dictionary of parameters.

        Returns:
            Dictionary mapping step names to their parameters.
        """
        if not params:
            return {}

        try:
            # Utilisation de process_routing de sklearn
            routed = process_routing(self, method, **params)

            # Conversion au format attendu par nos méthodes
            result = {}
            for name, step_routing in routed.items():
                if name == "router":
                    continue
                result[name] = {}
                for method_name in ["fit", "transform", "predict", "score",
                                   "inverse_transform", "predict_proba",
                                   "decision_function"]:
                    if hasattr(step_routing, method_name):
                        result[name][method_name] = dict(
                            getattr(step_routing, method_name)
                        )
                    else:
                        result[name][method_name] = {}

            return result
        except Exception:
            # Fallback vers le routage dunder
            return self._route_params_dunder(method, params)

    # Routage via syntaxe dunder (step__param)
    def _route_params_dunder(self, method: str, params: dict) -> dict:
        """Route parameters using dunder (__) syntax.

        Args:
            method: Name of the method.
            params: Dictionary of parameters.

        Returns:
            Dictionary mapping step names to their parameters.
        """
        routed = {}

        for key, value in params.items():
            if "__" in key:
                # Syntaxe step__param
                step_name, param_name = key.split("__", 1)
                if step_name not in routed:
                    routed[step_name] = {
                        "fit": {},
                        "transform": {},
                        "predict": {},
                        "score": {},
                        "inverse_transform": {},
                        "predict_proba": {},
                        "decision_function": {},
                    }
                # Routage vers toutes les méthodes par défaut
                for m in routed[step_name]:
                    routed[step_name][m][param_name] = value
            else:
                # Paramètre pour le dernier step
                last_step_name = self.steps[-1][0]
                if last_step_name not in routed:
                    routed[last_step_name] = {
                        "fit": {},
                        "transform": {},
                        "predict": {},
                        "score": {},
                        "inverse_transform": {},
                        "predict_proba": {},
                        "decision_function": {},
                    }
                routed[last_step_name][method][key] = value

        return routed

    # Configuration du metadata routing pour sklearn
    def get_metadata_routing(self):
        """Get metadata routing configuration.

        Returns:
            MetadataRouter configuration for this pipeline.
        """
        router = MetadataRouter(owner=self.__class__.__name__)

        # Configuration du routage pour chaque step
        for name, step in self.steps:
            if step is None or step == "passthrough":
                continue

            method_mapping = MethodMapping()

            # Mapping des méthodes du pipeline vers les méthodes des steps
            # fit -> fit, transform
            method_mapping.add(caller="fit", callee="fit")
            method_mapping.add(caller="fit", callee="transform")

            # fit_transform -> fit, transform
            method_mapping.add(caller="fit_transform", callee="fit")
            method_mapping.add(caller="fit_transform", callee="transform")

            # transform -> transform
            method_mapping.add(caller="transform", callee="transform")

            # inverse_transform -> inverse_transform
            method_mapping.add(caller="inverse_transform", callee="inverse_transform")

            # predict, score, etc. -> transform (pour les steps intermédiaires)
            if step != self.steps[-1][1]:
                method_mapping.add(caller="predict", callee="transform")
                method_mapping.add(caller="predict_proba", callee="transform")
                method_mapping.add(caller="decision_function", callee="transform")
                method_mapping.add(caller="score", callee="transform")
            else:
                # Dernier step : mapping vers les méthodes correspondantes
                method_mapping.add(caller="predict", callee="predict")
                method_mapping.add(caller="predict_proba", callee="predict_proba")
                method_mapping.add(caller="decision_function", callee="decision_function")
                method_mapping.add(caller="score", callee="score")

            router.add(method_mapping=method_mapping, **{name: step})

        return router

    # Propriétés et accesseurs
    # Indices des steps qui transforment y
    @property
    def y_transformer_indices_(self) -> list[int]:
        """Indices of steps that transform y.

        Returns:
            List of step indices (0-based) that transformed y during fit.
        """
        if not hasattr(self, "_y_transformers"):
            raise AttributeError(
                "y_transformer_indices_ is not available before fitting."
            )
        return self._y_transformers

    # Récupération des transformateurs de y
    def get_y_transformers(self) -> list[tuple[str, Any]]:
        """Get the transformers that modify y.

        Returns:
            List of (name, transformer) tuples for steps that transform y.
        """
        if not hasattr(self, "_y_transformers"):
            raise AttributeError(
                "Pipeline must be fitted before accessing y transformers."
            )
        return [(self.steps[i][0], self.steps[i][1]) for i in self._y_transformers]