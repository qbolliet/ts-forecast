"""Panelwise transformer for applying transformations independently per entity.

This module provides a transformer that wraps any sklearn-compatible transformer
and applies it independently to each entity in panel data. Supports entity-specific
parameterization via callable factories or entity_kwargs.
"""
# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Any, Callable
import warnings
# Joblib
from joblib import Parallel, delayed
# Sklearn
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import _BaseComposition

# Importation des utilitaires de gestion de package
from ..base.transformers import (
    PanelTimeSeriesTransformer,
    ReversibleTransformerMixin
)
from .utils import normalize_entity_key

# Classe d'application d'un transformer indépendamment sur chaque entité du panel
class PanelwiseTransformer(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
    """Transformer that applies a base transformer independently to each panel entity.

    This transformer wraps any sklearn-compatible transformer (or pipeline) and
    applies it separately to each entity in panel data. Each entity gets its own
    fitted transformer instance, enabling entity-specific transformations such as
    scaling, encoding, or feature engineering.

    Supports three modes for entity-specific parameterization:
    1. Simple mode: Same transformer cloned for all entities
    2. Factory mode: Callable that creates a transformer per entity
    3. Entity kwargs mode: Dict of kwargs applied via set_params per entity

    The transformer is fully compatible with sklearn pipelines, GridSearchCV,
    cross-validation, and other sklearn utilities.

    Args:
        transformer: Base transformer, pipeline, or callable factory.
            - If BaseEstimator: Cloned for each entity (default sklearn behavior)
            - If Callable[[tuple], BaseEstimator]: Called with entity_key to create
              entity-specific transformer instances
        time_col: Name of the time column. Defaults to 'date'.
        panel_cols: Columns identifying panel entities. Required for panel data.
        entity_kwargs: Dict mapping entity keys to kwargs dicts. When provided
            (and transformer is not callable), these kwargs are passed to
            set_params() on the cloned transformer for each entity before fitting.
            Ignored if transformer is a callable factory.
        default_entity_kwargs: Default kwargs applied to entities not found in
            entity_kwargs. Only used when entity_kwargs is provided.
        validate_input: Whether to validate input data. Defaults to True.
        strict_validation: If True, raises errors on validation failure.
        auto_sort: If True, automatically sorts unsorted data. Defaults to False.
        convert_cols_to_index: If True, converts time_col and panel_cols to index.
        n_jobs: Number of parallel jobs for fitting/transforming.
            -1 means using all processors. Defaults to 1 (no parallelism).
        error_handling: How to handle errors during entity transformation.
            Options: 'raise', 'warn', 'ignore'. Defaults to 'raise'.

    Attributes:
        transformers_: Dict mapping entity keys to fitted transformers.
        transform_columns_: List of columns to transform for each entity.
        entities_: Number of unique entities found during fit.
        failed_entities_: List of entities that failed during fitting/transforming.
        is_factory_mode_: Whether transformer is a callable factory.

    Examples:
        Simple mode - same transformer for all entities:

        >>> from sklearn.preprocessing import StandardScaler
        >>> transformer = PanelwiseTransformer(
        ...     transformer=StandardScaler(),
        ...     panel_cols=['country']
        ... )
        >>> df_scaled = transformer.fit_transform(df)

        Factory mode - callable creates entity-specific transformers:

        >>> def scaler_factory(entity_key):
        ...     # Different scaling strategy per country
        ...     if entity_key == ('FR',):
        ...         return StandardScaler(with_mean=True)
        ...     return StandardScaler(with_mean=False)
        >>>
        >>> transformer = PanelwiseTransformer(
        ...     transformer=scaler_factory,
        ...     panel_cols=['country']
        ... )

        Entity kwargs mode - same base transformer with per-entity params:

        >>> transformer = PanelwiseTransformer(
        ...     transformer=PublicationDelayTransformer(
        ...         strategy='shift',
        ...         prediction_date='2024-12-15',
        ...         delays={}  # Placeholder
        ...     ),
        ...     entity_kwargs={
        ...         ('FR',): {'delays': {'GDP': 45}, 'delay_unit': 'D'},
        ...         ('DE',): {'delays': {'GDP': 60}, 'delay_unit': 'D'},
        ...     },
        ...     panel_cols=['country']
        ... )

    Notes:
        - Factory mode takes precedence over entity_kwargs mode
        - Each entity receives its own transformer instance
        - The transformer preserves the original DataFrame structure
        - Supports inverse_transform if the base transformer does
        - Memory usage scales with the number of entities
    """

    # Initialisation
    def __init__(
        self,
        transformer: Union[BaseEstimator, Callable[[tuple], BaseEstimator]],
        time_col: str = 'date',
        panel_cols: Optional[List[str]] = None,
        entity_kwargs: Optional[Dict[tuple, Dict[str, Any]]] = None,
        default_entity_kwargs: Optional[Dict[str, Any]] = None,
        validate_input: bool = True,
        strict_validation: bool = True,
        auto_sort: bool = False,
        convert_cols_to_index: bool = False,
        n_jobs: int = 1,
        error_handling: str = 'raise'
    ):
        """Initialize the PanelwiseTransformer.

        Args:
            transformer: Sklearn-compatible transformer, pipeline, or callable factory.
                If callable, must accept entity_key (tuple) and return a transformer.
            time_col: Name of the time column. Defaults to 'date'.
            panel_cols: Columns identifying panel entities.
            entity_kwargs: Dict mapping entity keys (tuples) to kwargs dicts.
                Applied via set_params() when transformer is not a callable.
            default_entity_kwargs: Default kwargs for entities not in entity_kwargs.
            validate_input: Whether to validate input data. Defaults to True.
            strict_validation: If True, raises errors; otherwise emits warnings.
            auto_sort: If True, automatically sorts the data. Defaults to False.
            convert_cols_to_index: If True, converts time_col and panel_cols to index.
            n_jobs: Number of parallel jobs (-1 = all processors). Defaults to 1.
            error_handling: Error handling strategy ('raise', 'warn', 'ignore').
        """
        # Initialisation du parent
        super().__init__(
            time_col=time_col,
            panel_cols=panel_cols,
            validate_input=validate_input,
            strict_validation=strict_validation,
            auto_sort=auto_sort,
            convert_cols_to_index=convert_cols_to_index
        )
        # Instanciation des paramètres
        self.transformer = transformer
        self.entity_kwargs = entity_kwargs
        self.default_entity_kwargs = default_entity_kwargs
        self.n_jobs = n_jobs
        self.error_handling = error_handling

    # Méthode auxiliaire de création du transformer pour une entité
    def _create_entity_transformer(self, entity_key: tuple) -> BaseEstimator:
        """Create a transformer instance for a specific entity.

        Handles three cases in order of priority:
        1. Factory mode: transformer is callable -> call it with entity_key
        2. Entity kwargs mode: clone transformer and apply entity-specific kwargs
        3. Simple mode: just clone the transformer

        Args:
            entity_key: Normalized entity identifier as tuple.

        Returns:
            Configured transformer instance for the entity.

        Raises:
            ValueError: If factory returns None or invalid transformer.
            TypeError: If entity_kwargs contains invalid parameter names.
        """
        # Cas 1 : Mode factory - le transformer est un callable
        if self.is_factory_mode_:
            transformer = self.transformer(entity_key)
            # Validation du résultat de la factory
            if transformer is None:
                raise ValueError(
                    f"Transformer factory returned None for entity {entity_key}. "
                    "Factory must return a valid transformer instance."
                )
            if not hasattr(transformer, 'fit') or not hasattr(transformer, 'transform'):
                raise ValueError(
                    f"Transformer factory returned invalid object for entity {entity_key}. "
                    f"Got {type(transformer)}, expected sklearn-compatible transformer."
                )
            return transformer

        # Cas 2 et 3 : Clone du transformer de base
        transformer = clone(self.transformer)

        # Cas 2 : Mode entity_kwargs - application des kwargs spécifiques
        if self.entity_kwargs is not None:
            # Recherche des kwargs pour cette entité
            kwargs = self.entity_kwargs.get(entity_key)

            # Fallback sur les kwargs par défaut si l'entité n'est pas trouvée
            if kwargs is None and self.default_entity_kwargs is not None:
                kwargs = self.default_entity_kwargs

            # Application des kwargs si disponibles
            if kwargs is not None:
                try:
                    transformer.set_params(**kwargs)
                except TypeError as e:
                    raise TypeError(
                        f"Invalid parameters in entity_kwargs for entity {entity_key}: {e}"
                    ) from e

        return transformer

    # Méthode auxiliaire de détection du mode factory
    def _detect_factory_mode(self) -> bool:
        """Detect if transformer is a callable factory.

        A callable is considered a factory if:
        - It's callable
        - It's not a class (type)
        - It's not an sklearn estimator instance with fit/transform methods

        Returns:
            True if transformer should be treated as a factory callable.
        """
        # Si c'est une classe (type), ce n'est pas une factory
        if isinstance(self.transformer, type):
            return False

        # Si c'est un estimator sklearn (a fit et transform), ce n'est pas une factory
        if hasattr(self.transformer, 'fit') and hasattr(self.transformer, 'transform'):
            return False

        # Sinon, c'est une factory si c'est callable
        return callable(self.transformer)

    # Méthode auxiliaire d'entraînement du transformer
    def _fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> None:
        """Fit a separate transformer for each panel entity.

        Args:
            X: Input features with panel_cols as columns or MultiIndex.
            y: Target variable aligned with X.

        Raises:
            ValueError: If panel_cols is not specified.
        """
        # Détection du mode factory (une seule fois au fit)
        self.is_factory_mode_ = self._detect_factory_mode()

        # Warning si entity_kwargs fourni mais mode factory actif
        if self.is_factory_mode_ and self.entity_kwargs is not None:
            warnings.warn(
                "entity_kwargs is ignored when transformer is a callable factory. "
                "The factory is responsible for entity-specific parameterization."
            )

        # Détection du format des données (colonnes vs MultiIndex)
        # Après validation, les colonnes peuvent avoir été converties en index
        self._has_multiindex = isinstance(X.index, pd.MultiIndex)

        # Détermination des colonnes à transformer
        if self._has_multiindex:
            # Les colonnes panel sont dans l'index, on transforme toutes les colonnes
            self.transform_columns_ = list(X.columns)
            # Récupération des entités depuis l'index
            entity_level_indices = list(range(X.index.nlevels - 1))
        else:
            # Les colonnes panel sont dans le DataFrame
            exclude_cols = set(self.panel_cols)
            # Ajout de la colonne de temps aux colonnes à exclure si elle est dans le jeu de données
            if self.time_col in X.columns:
                exclude_cols.add(self.time_col)
            # Liste des colonnes à transformer
            self.transform_columns_ = [c for c in X.columns if c not in exclude_cols]

        # Stockage des transformers par entité
        self.transformers_: Dict[tuple, BaseEstimator] = {}
        self.failed_entities_: List[tuple] = []

        # Conversion de y en Series si c'est un array pour permettre l'indexation
        if y is not None and isinstance(y, np.ndarray):
            y = pd.Series(y, index=X.index)

        # Groupement par entité
        if self._has_multiindex:
            # Groupement par les niveaux de l'index
            entity_groups = X.groupby(level=entity_level_indices)
        else:
            entity_groups = X.groupby(self.panel_cols)

        # Extraction du nombre d'entités
        self.entities_ = entity_groups.ngroups

        # Fit parallèle ou séquentiel
        if self.n_jobs == 1:
            self._fit_sequential(entity_groups, X, y)
        else:
            self._fit_parallel(entity_groups, X, y)

    # Méthode auxiliaire d'entraînement séquenciel des transformers
    def _fit_sequential(
        self,
        entity_groups,
        X: pd.DataFrame,
        y: Optional[pd.Series]
    ) -> None:
        """Fit transformers sequentially for each entity.

        Args:
            entity_groups: GroupBy object grouping entities.
            X: Complete input DataFrame.
            y: Target variable aligned with X.
        """
        # Parcours des entités
        for entity_key, group_idx in entity_groups.groups.items():
            # Normalisation de la clé en tuple
            entity_key = normalize_entity_key(entity_key)

            try:
                # Création du transformer pour cette entité
                entity_transformer = self._create_entity_transformer(entity_key)

                # Extraction des données de l'entité
                X_entity = X.loc[group_idx, self.transform_columns_].xs(entity_key)

                # Extraction de y pour cette entité si fourni
                y_entity = y.loc[group_idx].xs(entity_key) if y is not None else None

                # Fit du transformer
                entity_transformer.fit(X_entity, y_entity)

                # Ajout du transformer au dictionnaire
                self.transformers_[entity_key] = entity_transformer

            except Exception as e:
                self._handle_entity_error(entity_key, e, "fitting")

    # Méthode auxiliaire d'entraînement parallèle des transformers
    def _fit_parallel(
        self,
        entity_groups,
        X: pd.DataFrame,
        y: Optional[pd.Series]
    ) -> None:
        """Fit transformers in parallel for each entity.

        Args:
            entity_groups: GroupBy object grouping entities.
            X: Complete input DataFrame.
            y: Target variable aligned with X.
        """
        # Référence à self pour la closure
        parent = self

        def fit_single_entity(entity_key, group_idx) -> tuple:
            """Fit a transformer for a single entity.

            Args:
                entity_key: Entity identifier (scalar or tuple).
                group_idx: Index locations for the entity group.

            Returns:
                Tuple of (normalized_entity_key, fitted_transformer).
            """
            # Normalisation de la clé de l'entité
            entity_key = normalize_entity_key(entity_key)
            # Création du transformer pour cette entité
            entity_transformer = parent._create_entity_transformer(entity_key)
            # Identification des observations afférentes au groupe
            X_entity = X.loc[group_idx, parent.transform_columns_].xs(entity_key)
            y_entity = y.loc[group_idx].xs(entity_key) if y is not None else None
            # Entrainement du transformer
            entity_transformer.fit(X_entity, y_entity)
            return entity_key, entity_transformer

        # Exécution parallèle
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_single_entity)(entity_key, group_idx)
            for entity_key, group_idx in entity_groups.groups.items()
        )

        # Collecte des résultats
        for entity_key, fitted_transformer in results:
            self.transformers_[entity_key] = fitted_transformer

    # Méthode de transformation des données
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using entity-specific transformers.

        Args:
            X: Input data with same structure as during fit.

        Returns:
            Transformed data with same structure as input.
        """
        # Détection du format des données
        has_multiindex = isinstance(X.index, pd.MultiIndex)

        # Groupement par entité
        if has_multiindex:
            entity_level_indices = list(range(X.index.nlevels - 1))
            entity_groups = X.groupby(level=entity_level_indices)
        else:
            entity_groups = X.groupby(self.panel_cols)

        # Liste pour stocker les DataFrames transformés
        transformed_parts = []

        # Parcours des entités
        for entity_key, group_idx in entity_groups.groups.items():
            # Normalisation de la clé de l'entité
            entity_key = normalize_entity_key(entity_key)

            # Vérification que l'entité a été vue pendant le fit
            if entity_key not in self.transformers_:
                # Gestion des identités inconnues
                self._handle_unknown_entity(entity_key)
                # Conservation des données non transformées
                transformed_parts.append(X.loc[group_idx])
                continue

            try:
                # Récupération du transformer
                entity_transformer = self.transformers_[entity_key]

                # Extraction et transformation des données
                X_entity = X.loc[group_idx, self.transform_columns_].xs(entity_key)
                X_entity_transformed = entity_transformer.transform(X_entity)

                # Reconstruction du DataFrame avec la structure panel et l'index du transformer
                df_part = self._reconstruct_entity_dataframe(
                    X_entity_transformed=X_entity_transformed,
                    entity_key=entity_key,
                    original_index=X.index
                )

                transformed_parts.append(df_part)

            except Exception as e:
                # Gestion de l'erreur de transformation
                self._handle_entity_error(entity_key, e, "transforming")
                # Conservation des données non transformées
                transformed_parts.append(X.loc[group_idx])

        # Concaténation des résultats
        X_result = pd.concat(transformed_parts, axis=0)

        return X_result

    # Méthode de transformation inverse
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform data using entity-specific transformers.

        Args:
            X: Transformed data.

        Returns:
            Original data format.

        Raises:
            AttributeError: If base transformer doesn't support inverse_transform.
        """
        # Vérification que l'estimateur a été entraîné
        check_is_fitted(self)

        # Vérification que le transformer supporte inverse_transform
        # En mode factory, on vérifie que le premier transformer est entraîné
        if self.is_factory_mode_:
            sample_transformer = next(iter(self.transformers_.values()), None)
            if sample_transformer and not hasattr(sample_transformer, 'inverse_transform'):
                raise AttributeError(
                    f"The transformer {type(sample_transformer).__name__} "
                    "does not support inverse_transform"
                )
        elif not hasattr(self.transformer, 'inverse_transform'):
            raise AttributeError(
                f"The base transformer {type(self.transformer).__name__} "
                "does not support inverse_transform"
            )

        # Détection du format des données
        has_multiindex = isinstance(X.index, pd.MultiIndex)

        # Groupement par entité
        if has_multiindex:
            entity_level_indices = list(range(X.index.nlevels - 1))
            entity_groups = X.groupby(level=entity_level_indices)
        else:
            entity_groups = X.groupby(self.panel_cols)

        # Liste pour stocker les DataFrames inversés
        inverted_parts = []

        # Parcours des entités
        for entity_key, group_idx in entity_groups.groups.items():
            # Normalisation de l'entité
            entity_key = normalize_entity_key(entity_key)

            # Renvoie une erreur si l'entité n'a pas de transformer associé
            if entity_key not in self.transformers_:
                # Gestion des identités inconnues
                self._handle_unknown_entity(entity_key)
                # Conservation des données non transformées
                inverted_parts.append(X.loc[group_idx])
                continue

            try:
                # Extraction du transformer associé à l'entité
                entity_transformer = self.transformers_[entity_key]

                # Extraction des colonnes transformées et transformation inverse
                X_entity = X.loc[group_idx, self.transform_columns_].xs(entity_key)
                X_entity_inverted = entity_transformer.inverse_transform(X_entity)

                # Reconstruction du DataFrame avec la structure panel et l'index retourné
                df_part = self._reconstruct_entity_dataframe(
                    X_entity_transformed=X_entity_inverted,
                    entity_key=entity_key,
                    original_index=X.index
                )

                # Ajout à la liste des données transformées
                inverted_parts.append(df_part)

            except Exception as e:
                # Gestion de l'erreur de transformation
                self._handle_entity_error(entity_key, e, "inverse_transforming")
                # Conservation des données non transformées
                inverted_parts.append(X.loc[group_idx])

        # Concaténation et tri
        X_result = pd.concat(inverted_parts, axis=0)

        # Restauration de la structure originale si conversion appliquée
        X_result = self._restore_structure_if_converted(X_result)

        return X_result

    # Méthode auxiliaire de reconstitution de la structure de panel par entité
    def _reconstruct_entity_dataframe(
        self,
        X_entity_transformed: Union[np.ndarray, pd.DataFrame],
        entity_key: tuple,
        original_index: pd.Index
    ) -> pd.DataFrame:
        """Reconstruct a DataFrame with proper panel structure from transformed entity data.

        Args:
            X_entity_transformed: Transformed data (array or DataFrame) with temporal index.
            entity_key: Entity identifier as tuple.
            original_index: Original MultiIndex or Index from the input data.

        Returns:
            Reconstructed DataFrame with proper panel structure and index.
        """
        # Conversion en DataFrame si nécessaire
        if isinstance(X_entity_transformed, np.ndarray):
            # Gestion des noms de colonnes
            if X_entity_transformed.shape[1] != len(self.transform_columns_):
                # Le transformer a changé le nombre de colonnes
                cols = self._generate_column_names(
                    X_entity_transformed.shape[1],
                    self.transformers_[entity_key]
                )
            else:
                cols = self.transform_columns_

            # Création d'un DataFrame avec l'index temporel
            df_transformed = pd.DataFrame(
                X_entity_transformed,
                columns=cols
            )
        else:
            df_transformed = X_entity_transformed

        # Reconstruction du MultiIndex avec l'entité
        if isinstance(original_index, pd.MultiIndex):
            # Extraction des noms des niveaux panel
            panel_level_names = list(original_index.names[:-1])
            time_level_name = original_index.names[-1]

            # Création du MultiIndex pour cette entité
            n_rows = len(df_transformed)
            entity_arrays = [[entity_key[i]] * n_rows for i in range(len(entity_key))]
            time_array = df_transformed.index

            new_index = pd.MultiIndex.from_arrays(
                entity_arrays + [time_array],
                names=panel_level_names + [time_level_name]
            )
            df_transformed.index = new_index

        return df_transformed

    # Méthode auxiliaire de génération des noms de colonnes
    def _generate_column_names(
        self,
        n_cols: int,
        transformer: BaseEstimator
    ) -> List[str]:
        """Generate column names for transformed data.

        Args:
            n_cols: Number of columns in transformed data.
            transformer: Fitted transformer instance.

        Returns:
            List of generated column names.
        """
        # Tentative de récupération des noms via get_feature_names_out
        if hasattr(transformer, 'get_feature_names_out'):
            try:
                return list(transformer.get_feature_names_out())
            except Exception:
                pass

        # Noms génériques
        return [f"feature_{i}" for i in range(n_cols)]

    # Méthode auxiliaire de gestion des erreurs par entité
    def _handle_entity_error(self, entity_key: tuple, error: Exception, operation: str) -> None:
        """Handle errors during entity operations.

        Args:
            entity_key: Entity identifier.
            error: Exception that occurred.
            operation: Name of the operation ('fitting', 'transforming', etc.).

        Raises:
            Exception: If error_handling is 'raise'.
        """
        # Ajout à la liste des entités pour lesquelles une erreur a été constatée
        self.failed_entities_.append(entity_key)

        # Distinction suivant la méthode de gestion des erreurs
        if self.error_handling == 'raise':
            raise error
        elif self.error_handling == 'warn':
            warnings.warn(
                f"Error {operation} entity {entity_key}: {error}. Skipping."
            )
        # 'ignore' : ne rien faire

    # Méthode auxiliaire de gestion des entités inconnues
    def _handle_unknown_entity(self, entity_key: tuple) -> None:
        """Handle unknown entities encountered during transform.

        Args:
            entity_key: Unknown entity identifier.
        """
        # Initialisation du message d'erreur
        msg = f"Entity {entity_key} was not seen during fit"

        # Distinction suivant la méthode de gestion des erreurs
        if self.error_handling == 'raise':
            raise KeyError(msg)
        elif self.error_handling == 'warn':
            warnings.warn(f"{msg}. Using untransformed data.")
        # 'ignore' : ne rien faire


    # Méthode auxiliaire d'extraction des paramètres
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Args:
            deep: If True, returns parameters of nested estimators.

        Returns:
            Parameter names mapped to their values.
        """
        # Extraction des paramètres du parent
        params = super().get_params(deep=False)

        # Ajout des paramètres propres
        params['transformer'] = self.transformer
        params['entity_kwargs'] = self.entity_kwargs
        params['default_entity_kwargs'] = self.default_entity_kwargs
        params['n_jobs'] = self.n_jobs
        params['error_handling'] = self.error_handling

        # Ajout des paramètres imbriqués si deep=True et mode non-factory
        if deep and not callable(self.transformer):
            if hasattr(self.transformer, 'get_params'):
                transformer_params = self.transformer.get_params(deep=True)
                for key, value in transformer_params.items():
                    params[f'transformer__{key}'] = value

        return params

    # Méthode d'instanciation des paramètres
    def set_params(self, **params) -> 'PanelwiseTransformer':
        """Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            Self for method chaining.
        """
        # Séparation des paramètres du transformer
        transformer_params = {}
        own_params = {}

        # Parcours des paramètres
        for key, value in params.items():
            if key.startswith('transformer__'):
                # Paramètre du transformer imbriqué
                transformer_params[key[len('transformer__'):]] = value
            else:
                own_params[key] = value

        # Application des paramètres propres
        if own_params:
            super().set_params(**own_params)

        # Application des paramètres du transformer (seulement si pas une factory)
        if transformer_params:
            if callable(self.transformer) and not hasattr(self.transformer, 'set_params'):
                warnings.warn(
                    "Cannot set nested parameters on a callable factory transformer. "
                    "Ignoring transformer__ parameters."
                )
            else:
                self.transformer.set_params(**transformer_params)

        return self

    # Méthode d'extraction du transformer associé à une entité
    def get_entity_transformer(self, entity_key: Union[tuple, Any]) -> BaseEstimator:
        """Get the fitted transformer for a specific entity.

        Args:
            entity_key: Entity identifier (tuple or scalar value).

        Returns:
            Fitted transformer for the entity.

        Raises:
            KeyError: If entity not found.
            NotFittedError: If transformer not fitted.
        """
        # Vérification que les transformers sont estimés
        check_is_fitted(self, ['transformers_'])
        # Normalisation de l'entité
        entity_key = self._normalize_entity_key(entity_key)
        # Renvoie une erreur si la clé n'est pas dans le dictionnaire de transformers
        if entity_key not in self.transformers_:
            raise KeyError(
                f"Entity {entity_key} not found. "
                f"Available entities: {list(self.transformers_.keys())}"
            )
        # Retourne le transformer de l'entité
        return self.transformers_[entity_key]

    # Nombre d'entités traitées par le transformer
    @property
    def n_entities_(self) -> int:
        """Number of entities with fitted transformers.

        Returns:
            Number of unique entities found during fit.

        Raises:
            NotFittedError: If transformer has not been fitted.
        """
        check_is_fitted(self, ['transformers_'])
        return len(self.transformers_)

    # Représentation sous forme de chaîne de caractères du transformer
    def __repr__(self) -> str:
        """Return a string representation of the transformer.

        Returns:
            String representation showing key parameters.
        """
        if callable(self.transformer) and not hasattr(self.transformer, 'fit'):
            transformer_repr = f"<factory: {self.transformer.__name__ if hasattr(self.transformer, '__name__') else 'callable'}>"
        else:
            transformer_repr = repr(self.transformer)

        return (
            f"PanelwiseTransformer(\n"
            f"    transformer={transformer_repr},\n"
            f"    panel_cols={self.panel_cols},\n"
            f"    time_col='{self.time_col}',\n"
            f"    entity_kwargs={'provided' if self.entity_kwargs else None}\n"
            f")"
        )