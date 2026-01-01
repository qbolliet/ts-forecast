"""Panelwise transformer for applying transformations independently per entity.

This module provides a transformer that wraps any sklearn-compatible transformer
and applies it independently to each entity in panel data.
"""
# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List, Any
import warnings
# Joblib
from joblib import Parallel, delayed
# Sklearn
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import _BaseComposition

# Importation des utilitaires de gestion de package
from ..utils.base_transformers import (
    PanelTimeSeriesTransformer,
    ReversibleTransformerMixin
)


# Classe d'application d'un transformer indépendamment sur chaque entité du panel
class PanelwiseTransformer(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
    """Transformer that applies a base transformer independently to each panel entity.

    This transformer wraps any sklearn-compatible transformer (or pipeline) and
    applies it separately to each entity in panel data. Each entity gets its own
    fitted transformer instance, enabling entity-specific transformations such as
    scaling, encoding, or feature engineering.

    The transformer is fully compatible with sklearn pipelines, GridSearchCV,
    cross-validation, and other sklearn utilities.

    Args:
        transformer (BaseEstimator): Base transformer or pipeline to apply to each entity.
            Must be sklearn-compatible (implement fit/transform).
        time_col (str): Name of the time column. Defaults to 'date'.
        panel_cols (Optional[List[str]]): Columns identifying panel entities.
            Required for panel data operations.
        validate_input (bool): Whether to validate input data. Defaults to True.
        strict_validation (bool): If True, raises errors on validation failure.
            Defaults to True.
        auto_sort (bool): If True, automatically sorts unsorted data. Defaults to False.
        convert_cols_to_index (bool): If True, converts time_col and panel_cols
            to index. Defaults to False.
        n_jobs (int): Number of parallel jobs for fitting/transforming.
            -1 means using all processors. Defaults to 1 (no parallelism).
        error_handling (str): How to handle errors during entity transformation.
            Options: 'raise' (raises exception), 'warn' (emits warning and skips entity),
            'ignore' (silently skips entity). Defaults to 'raise'.

    Attributes:
        transformers_ (Dict[tuple, BaseEstimator]): Dictionary mapping entity keys to fitted transformers.
        transform_columns_ (List[str]): List of columns to transform for each entity.
        entities_ (int): Number of unique entities found during fit.
        failed_entities_ (List[tuple]): List of entities that failed during fitting/transforming.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from sklearn.preprocessing import StandardScaler
        >>>
        >>> # Create panel data
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2023-01-01', periods=10).tolist() * 2,
        ...     'country': ['FR'] * 10 + ['DE'] * 10,
        ...     'value': np.random.randn(20),
        ...     'feature': np.random.randn(20)
        ... })
        >>>
        >>> # Apply StandardScaler per country
        >>> transformer = PanelwiseTransformer(
        ...     transformer=StandardScaler(),
        ...     time_col='date',
        ...     panel_cols=['country']
        ... )
        >>> df_scaled = transformer.fit_transform(df)
        >>>
        >>> # Use in sklearn pipeline
        >>> from sklearn.pipeline import Pipeline
        >>> pipe = Pipeline([
        ...     ('panelwise_scale', PanelwiseTransformer(
        ...         transformer=StandardScaler(),
        ...         panel_cols=['country']
        ...     )),
        ...     ('other_step', SomeOtherTransformer())
        ... ])
        >>>
        >>> # Compatible with GridSearchCV
        >>> from sklearn.model_selection import GridSearchCV
        >>> param_grid = {
        ...     'panelwise_scale__transformer__with_mean': [True, False]
        ... }

    Notes:
        - Each entity receives its own clone of the base transformer
        - The transformer preserves the original DataFrame structure
        - Supports inverse_transform if the base transformer does
        - Memory usage scales with the number of entities
    """

    # Initialisation
    def __init__(
        self,
        transformer: BaseEstimator,
        time_col: str = 'date',
        panel_cols: Optional[List[str]] = None,
        validate_input: bool = True,
        strict_validation: bool = True,
        auto_sort: bool = False,
        convert_cols_to_index: bool = False,
        n_jobs: int = 1,
        error_handling: str = 'raise'
    ):
        """Initialize the PanelwiseTransformer.

        Args:
            transformer (BaseEstimator): Sklearn-compatible transformer to apply per entity.
            time_col (str): Name of the time column. Defaults to 'date'.
            panel_cols (Optional[List[str]]): Columns identifying panel entities.
            validate_input (bool): Whether to validate input data. Defaults to True.
            strict_validation (bool): If True, raises errors; otherwise emits warnings.
                Defaults to True.
            auto_sort (bool): If True, automatically sorts the data. Defaults to False.
            convert_cols_to_index (bool): If True, converts time_col and panel_cols to index.
                Defaults to False.
            n_jobs (int): Number of parallel jobs (-1 = all processors). Defaults to 1.
            error_handling (str): Error handling strategy. Options: 'raise', 'warn', 'ignore'.
                Defaults to 'raise'.
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
        # Initialisation 
        self.transformer = transformer
        self.n_jobs = n_jobs
        self.error_handling = error_handling

    # Méthode auxiliaire d'entraînement du transformer
    def _fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> None:
        """Fit a separate transformer for each panel entity.

        Args:
            X (pd.DataFrame): Input features with panel_cols as columns or MultiIndex.
            y (Optional[Union[pd.Series, np.ndarray]]): Target variable aligned with X.

        Raises:
            ValueError: If panel_cols is not specified.
        """
        # Détection du format des données (colonnes vs MultiIndex)
        # Après validation, les colonnes peuvent avoir été converties en index
        self._has_multiindex = isinstance(X.index, pd.MultiIndex)

        # Détermination des colonnes à transformer
        if self._has_multiindex:
            # Les colonnes panel sont dans l'index, on transforme toutes les colonnes
            self.transform_columns_ = list(X.columns)
            # Récupération des entités depuis l'index
            entity_level_indices = list(range(X.index.nlevels -1))
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

    def _fit_sequential(
        self,
        entity_groups,
        X: pd.DataFrame,
        y: Optional[pd.Series]
    ) -> None:
        """Fit transformers sequentially for each entity.

        Args:
            entity_groups (pandas.core.groupby.GroupBy): GroupBy object grouping entities.
            X (pd.DataFrame): Complete input DataFrame.
            y (Optional[pd.Series]): Target variable aligned with X.
        """
        # Parcours des entités
        for entity_key, group_idx in entity_groups.groups.items():
            # Normalisation de la clé en tuple
            entity_key = self._normalize_entity_key(entity_key)

            try:
                # Clone du transformer pour cette entité
                entity_transformer = clone(self.transformer)

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

    def _fit_parallel(
        self,
        entity_groups,
        X: pd.DataFrame,
        y: Optional[pd.Series]
    ) -> None:
        """Fit transformers in parallel for each entity.

        Args:
            entity_groups (pandas.core.groupby.GroupBy): GroupBy object grouping entities.
            X (pd.DataFrame): Complete input DataFrame.
            y (Optional[pd.Series]): Target variable aligned with X.
        """

        def fit_single_entity(entity_key, group_idx) -> tuple:
            """Fit a transformer for a single entity.

            Args:
                entity_key: Entity identifier (scalar or tuple).
                group_idx: Index locations for the entity group.

            Returns:
                Tuple of (normalized_entity_key, fitted_transformer).
            """
            # Normalisation de la clé de l'entité
            entity_key = self._normalize_entity_key(entity_key)
            # Clone du transformer
            entity_transformer = clone(self.transformer)
            # Identification des observations afférentes au groupe
            X_entity = X.loc[group_idx, self.transform_columns_].xs(entity_key)
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
        # Parcours des résultats
        for entity_key, fitted_transformer in results:
            # Ajout au dictionnaire
            self.transformers_[entity_key] = fitted_transformer

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using entity-specific transformers.

        Args:
            X (pd.DataFrame): Input data with same structure as during fit.

        Returns:
            pd.DataFrame: Transformed data with same structure as input.
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
            entity_key = self._normalize_entity_key(entity_key)

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

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from sklearn.preprocessing import StandardScaler
            >>>
            >>> # Create panel data
            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2023-01-01', periods=10).tolist() * 2,
            ...     'country': ['FR'] * 10 + ['DE'] * 10,
            ...     'value': np.random.randn(20)
            ... })
            >>>
            >>> # Fit and transform
            >>> transformer = PanelwiseTransformer(
            ...     transformer=StandardScaler(),
            ...     time_col='date',
            ...     panel_cols=['country']
            ... )
            >>> df_scaled = transformer.fit_transform(df)
            >>>
            >>> # Inverse transform to get original scale
            >>> df_original = transformer.inverse_transform(df_scaled)
        """
        # Vérification que l'estimateur a été entraîné
        check_is_fitted(self)

        # Vérification que le transformer supporte inverse_transform
        if not hasattr(self.transformer, 'inverse_transform'):
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
            entity_key = self._normalize_entity_key(entity_key)

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

    # Méthode auxiliaire de normalisation de l'entité du panel
    def _normalize_entity_key(self, key) -> tuple:
        """Normalize entity key to tuple format.

        Args:
            key: Entity key (scalar or tuple).

        Returns:
            tuple: Normalized entity key as a tuple.
        """
        if isinstance(key, tuple):
            return key
        return (key,)

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
            original_index: Original MultiIndex or Index from the input data (for structure reference).

        Returns:
            pd.DataFrame: Reconstructed DataFrame with proper panel structure and index from transformer.
        """
        # Conversion en DataFrame si nécessaire
        if isinstance(X_entity_transformed, np.ndarray):
            # Gestion des noms de colonnes
            if X_entity_transformed.shape[1] != len(self.transform_columns_):
                # Le transformer a changé le nombre de colonnes
                # Génération de nouveaux de noms de colonnes
                cols = self._generate_column_names(
                    X_entity_transformed.shape[1],
                    self.transformers_[entity_key]
                )
            else:
                cols = self.transform_columns_

            # Création d'un DataFrame temporaire pour obtenir l'index
            # L'index est celui qui correspond aux lignes de X_entity_transformed
            df_transformed = pd.DataFrame(
                X_entity_transformed,
                columns=cols
            )
        else:
            df_transformed = X_entity_transformed.copy()

        # Reconstruction de la structure panel
        if self._has_multiindex:
            # Création d'un MultiIndex combinant entity_key et l'index temporel
            # entity_key est un tuple avec les valeurs pour chaque niveau panel
            # L'index de df_transformed est le niveau temporel

            # Construction des arrays pour chaque niveau du MultiIndex
            n_rows = len(df_transformed)
            index_arrays = []

            # Ajout des niveaux pour les panel_cols
            for i, value in enumerate(entity_key):
                index_arrays.append([value] * n_rows)

            # Ajout du niveau temporel (dernier niveau)
            index_arrays.append(df_transformed.index)

            # Création du MultiIndex
            new_index = pd.MultiIndex.from_arrays(index_arrays, names=original_index.names)
            df_transformed.index = new_index

        else:
            # Les panel_cols sont des colonnes
            # Ajout des colonnes panel avec les valeurs de entity_key
            for i, col in enumerate(self.panel_cols):
                df_transformed[col] = entity_key[i]

            # Ajout de la colonne time_col si elle existe dans les colonnes originales
            if self.time_col in original_index.names or not isinstance(original_index, pd.MultiIndex):
                # L'index de df_transformed est déjà l'index temporel
                # On le nomme correctement
                df_transformed.index.name = self.time_col

        return df_transformed

    # Méthode auxiliaire de gestion des erreurs concernant une entité
    def _handle_entity_error(
        self,
        entity_key: tuple,
        error: Exception,
        operation: str
    ) -> None:
        """Handle errors during entity processing.

        Args:
            entity_key (tuple): Entity identifier.
            error (Exception): Exception that was raised.
            operation (str): Type of operation ('fitting', 'transforming', etc.).
        """
        # Initialisation du message
        msg = f"Error {operation} entity {entity_key}: {str(error)}"

        if self.error_handling == 'raise':
            raise RuntimeError(msg) from error
        elif self.error_handling == 'warn':
            warnings.warn(msg)
            self.failed_entities_.append(entity_key)
        else:  # 'ignore'
            self.failed_entities_.append(entity_key)

    # Méthode auxiliaire de gestion des entités pour lesquelles aucun transformer n'est entraîné
    def _handle_unknown_entity(
        self,
        entity_key: tuple
    ) -> None:
        """Handle entities not seen during fit.

        Args:
            entity_key (tuple): Identifier of the unknown entity.

        Raises:
            ValueError: If error_handling is 'raise'.
        """
        # Génération du message d'erreur
        msg = (
            f"Entity {entity_key} was not seen during fit. "
            "Data will be returned unchanged."
        )
        # Distinction suivant la méthode de gestion des erreurs
        if self.error_handling == 'raise':
            raise ValueError(msg)
        elif self.error_handling == 'warn':
            warnings.warn(msg)

    # Méthode auxiliaire de génération de noms de colonnes
    def _generate_column_names(
        self,
        n_cols: int,
        transformer: BaseEstimator
    ) -> List[str]:
        """Generate column names for transformed output.

        Args:
            n_cols (int): Number of columns in the output.
            transformer (BaseEstimator): Transformer that generated the columns.

        Returns:
            List[str]: List of generated column names.
        """
        # Tentative de récupération des noms via get_feature_names_out
        if hasattr(transformer, 'get_feature_names_out'):
            try:
                return list(transformer.get_feature_names_out())
            except Exception:
                pass

        # Noms génériques
        return [f"feature_{i}" for i in range(n_cols)]

    # Méthode auxiliaire d'extraction des paramètres
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Args:
            deep: If True, returns parameters of nested estimators.

        Returns:
            Parameter names mapped to their values.

        Examples:
            >>> from sklearn.preprocessing import StandardScaler
            >>>
            >>> transformer = PanelwiseTransformer(
            ...     transformer=StandardScaler(),
            ...     panel_cols=['country'],
            ...     n_jobs=2
            ... )
            >>>
            >>> # Get all parameters
            >>> params = transformer.get_params(deep=True)
            >>> print(params['n_jobs'])
            2
            >>> print(params['transformer__with_mean'])  # nested parameter
            True
        """
        # Extraction des paramètres du parent
        params = super().get_params(deep=False)
        # Ajout des paramètres de la classe
        params['transformer'] = self.transformer
        params['n_jobs'] = self.n_jobs
        params['error_handling'] = self.error_handling
        # Ajout des paramètres
        if deep and hasattr(self.transformer, 'get_params'):
            # Ajout des paramètres du transformer avec préfixe
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

        Examples:
            >>> from sklearn.preprocessing import StandardScaler
            >>>
            >>> transformer = PanelwiseTransformer(
            ...     transformer=StandardScaler(),
            ...     panel_cols=['country']
            ... )
            >>>
            >>> # Update n_jobs parameter
            >>> transformer.set_params(n_jobs=4)
            PanelwiseTransformer(...)
            >>>
            >>> # Update nested transformer parameter
            >>> transformer.set_params(transformer__with_std=False)
            PanelwiseTransformer(...)
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

        # Application des paramètres du transformer
        if transformer_params:
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

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from sklearn.preprocessing import StandardScaler
            >>>
            >>> # Create and fit transformer
            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2023-01-01', periods=10).tolist() * 2,
            ...     'country': ['FR'] * 10 + ['DE'] * 10,
            ...     'value': np.random.randn(20)
            ... })
            >>>
            >>> transformer = PanelwiseTransformer(
            ...     transformer=StandardScaler(),
            ...     time_col='date',
            ...     panel_cols=['country']
            ... )
            >>> transformer.fit(df)
            >>>
            >>> # Get transformer for France
            >>> fr_scaler = transformer.get_entity_transformer('FR')
            >>> print(fr_scaler.mean_)  # Access fitted attributes
            >>>
            >>> # For multi-column entities, use tuple
            >>> de_scaler = transformer.get_entity_transformer(('DE',))
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

    @property
    def n_entities_(self) -> int:
        """Number of entities with fitted transformers.

        Returns:
            int: Number of unique entities found during fit.

        Raises:
            NotFittedError: If transformer has not been fitted.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from sklearn.preprocessing import StandardScaler
            >>>
            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2023-01-01', periods=10).tolist() * 3,
            ...     'country': ['FR'] * 10 + ['DE'] * 10 + ['IT'] * 10,
            ...     'value': np.arange(30)
            ... })
            >>>
            >>> transformer = PanelwiseTransformer(
            ...     transformer=StandardScaler(),
            ...     time_col='date',
            ...     panel_cols=['country']
            ... )
            >>> transformer.fit(df)
            >>> print(transformer.n_entities_)
            3
        """
        check_is_fitted(self, ['transformers_'])
        return len(self.transformers_)

    def __repr__(self) -> str:
        """Return a string representation of the transformer.

        Returns:
            str: String representation showing key parameters.
        """
        transformer_repr = repr(self.transformer)
        return (
            f"PanelwiseTransformer(\n"
            f"    transformer={transformer_repr},\n"
            f"    panel_cols={self.panel_cols},\n"
            f"    time_col='{self.time_col}'\n"
            f")"
        )
