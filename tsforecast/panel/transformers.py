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

    Parameters:
        transformer: Base transformer or pipeline to apply to each entity.
            Must be sklearn-compatible (implement fit/transform).
        time_col (str): Name of the time column. Default is 'date'.
        panel_cols (Optional[List[str]]): Columns identifying panel entities.
            Required for panel data operations.
        validate_input (bool): Whether to validate input data. Default is True.
        strict_validation (bool): If True, raises errors on validation failure.
            Default is True.
        auto_sort (bool): If True, automatically sorts unsorted data. Default is False.
        convert_cols_to_index (bool): If True, converts time_col and panel_cols
            to index. Default is False.
        n_jobs (int): Number of parallel jobs for fitting/transforming.
            -1 means using all processors. Default is 1 (no parallelism).
        error_handling (str): How to handle errors during entity transformation.
            'raise': raise the exception (default)
            'warn': emit warning and skip entity
            'ignore': silently skip entity

    Attributes:
        transformers_ (Dict): Dictionary mapping entity keys to fitted transformers.
        entity_columns_ (List[str]): List of columns used for entity identification.
        entities_ (List): List of unique entity identifiers found during fit.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.pipeline import Pipeline
        >>>
        >>> # Création de données panel
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2023-01-01', periods=10).tolist() * 2,
        ...     'country': ['FR'] * 10 + ['DE'] * 10,
        ...     'value': np.random.randn(20),
        ...     'feature': np.random.randn(20)
        ... })
        >>>
        >>> # Application d'un StandardScaler par pays
        >>> transformer = PanelwiseTransformer(
        ...     transformer=StandardScaler(),
        ...     time_col='date',
        ...     panel_cols=['country']
        ... )
        >>> df_scaled = transformer.fit_transform(df)
        >>>
        >>> # Utilisation dans une pipeline sklearn
        >>> from sklearn.pipeline import Pipeline
        >>> pipe = Pipeline([
        ...     ('panelwise_scale', PanelwiseTransformer(
        ...         transformer=StandardScaler(),
        ...         panel_cols=['country']
        ...     )),
        ...     ('other_step', SomeOtherTransformer())
        ... ])
        >>>
        >>> # Compatible GridSearchCV
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
            transformer: Sklearn-compatible transformer to apply per entity.
            time_col: Name of the time column.
            panel_cols: Columns identifying panel entities.
            validate_input: Whether to validate input data.
            strict_validation: If True, raises errors; otherwise emits warnings.
            auto_sort: If True, automatically sorts the data.
            convert_cols_to_index: If True, converts time_col and panel_cols to index.
            n_jobs: Number of parallel jobs (-1 = all processors).
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
        # Initialisation 
        self.transformer = transformer
        self.n_jobs = n_jobs
        self.error_handling = error_handling

    # Méthode d'entraînement des transformers
    def _fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> None:
        """Fit a separate transformer for each panel entity.

        Args:
            X: Input features (DataFrame avec panel_cols ou MultiIndex)
            y: Target variable (optional, aligné sur X)
        """
        # Validation des panel_cols
        if not self.panel_cols:
            raise ValueError(
                "panel_cols must be specified for PanelwiseTransformer. "
                "For non-panel data, use the base transformer directly."
            )

        # Détection du format des données (colonnes vs MultiIndex)
        # Après validation, les colonnes peuvent avoir été converties en index
        self._has_multiindex = isinstance(X.index, pd.MultiIndex)

        # Détermination des colonnes à transformer
        if self._has_multiindex:
            # Les colonnes panel sont dans l'index, on transforme toutes les colonnes
            self.transform_columns_ = list(X.columns)
            # Récupération des entités depuis l'index
            entity_level_indices = list(range(len(self.panel_cols)))
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

    # Méthode d'estimation séquentielle des transformers sur chaque individu
    def _fit_sequential(
        self,
        entity_groups,
        X: pd.DataFrame,
        y: Optional[pd.Series]
    ) -> None:
        """Fit transformers sequentially for each entity.

        Args:
            entity_groups: GroupBy object des entités
            X: DataFrame complet
            y: Target variable (optional)
        """
        # Parcours des entités
        for entity_key, group_idx in entity_groups.groups.items():
            # Normalisation de la clé en tuple
            entity_key = self._normalize_entity_key(entity_key)

            try:
                # Clone du transformer pour cette entité
                entity_transformer = clone(self.transformer)

                # Extraction des données de l'entité
                X_entity = X.loc[group_idx, self.transform_columns_]

                # Extraction de y pour cette entité si fourni
                y_entity = y.loc[group_idx] if y is not None else None

                # Fit du transformer
                entity_transformer.fit(X_entity, y_entity)

                # Ajout du transformer au dictionnaire
                self.transformers_[entity_key] = entity_transformer

            except Exception as e:
                self._handle_entity_error(entity_key, e, "fitting")

    # Méthode d'estimation paramllèle des transformers sur chaque individu
    def _fit_parallel(
        self,
        entity_groups,
        X: pd.DataFrame,
        y: Optional[pd.Series]
    ) -> None:
        """Fit transformers in parallel for each entity.

        Args:
            entity_groups: GroupBy object des entités
            X: DataFrame complet
            y: Target variable (optional)
        """

        # Fonction d'estimation d'un transformer sur une entité
        def fit_single_entity(entity_key, group_idx) -> Dict[tuple, BaseEstimator]:
            # TODO : Traduire la docstring + respecter la convention de Google
            # TODO : Typer la fonction
            """Fit un transformer pour une seule entité."""
            # Normalisation de la clé de l'entité
            entity_key = self._normalize_entity_key(entity_key)
            # Clone du transformer
            entity_transformer = clone(self.transformer)
            # Identification des observations afférentes au groupe
            X_entity = X.loc[group_idx, self.transform_columns_]
            y_entity = y.loc[group_idx] if y is not None else None
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

    # Fonction de transformation
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using entity-specific transformers.

        Args:
            X: Input data

        Returns:
            Transformed DataFrame with same structure
        """
        # Détection du format des données
        has_multiindex = isinstance(X.index, pd.MultiIndex)

        # Groupement par entité
        if has_multiindex:
            entity_level_indices = list(range(len(self.panel_cols)))
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
                X_entity = X.loc[group_idx, self.transform_columns_]
                X_entity_transformed = entity_transformer.transform(X_entity)

                # Reconstruction du DataFrame avec toutes les colonnes
                df_part = X.loc[group_idx].copy()

                # Gestion du résultat de transform (array ou DataFrame)
                if isinstance(X_entity_transformed, np.ndarray):
                    # Si le transformer retourne un array avec plus/moins de colonnes
                    if X_entity_transformed.shape[1] != len(self.transform_columns_):
                        # Nouvelles colonnes générées
                        new_cols = self._generate_column_names(
                            X_entity_transformed.shape[1],
                            entity_transformer
                        )
                        # Suppression des anciennes colonnes
                        df_part = df_part.drop(columns=self.transform_columns_)
                        # Ajout des nouvelles colonnes
                        for i, col in enumerate(new_cols):
                            df_part[col] = X_entity_transformed[:, i]
                    else:
                        # Même nombre de colonnes
                        for i, col in enumerate(self.transform_columns_):
                            df_part[col] = X_entity_transformed[:, i]
                else:
                    # DataFrame retourné
                    for col in X_entity_transformed.columns:
                        if col in self.transform_columns_:
                            df_part[col] = X_entity_transformed[col].values
                        else:
                            # Nouvelle colonne
                            df_part[col] = X_entity_transformed[col].values

                transformed_parts.append(df_part)

            except Exception as e:
                # Gestion de l'erreur de transformation
                self._handle_entity_error(entity_key, e, "transforming")
                # Conservation des données non transformées
                transformed_parts.append(X.loc[group_idx])

        # Concaténation des résultats
        X_result = pd.concat(transformed_parts, axis=0)

        # Restauration de l'ordre original
        X_result = X_result.loc[X.index]

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
            entity_level_indices = list(range(len(self.panel_cols)))
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

                # Extraction des colonnes transformées
                X_entity = X.loc[group_idx, self.transform_columns_]
                X_entity_inverted = entity_transformer.inverse_transform(X_entity)

                # Reconstruction du DataFrame
                df_part = X.loc[group_idx].copy()

                # Remplacement des valeurs des colonnes par la transformation inverse
                # Le cas où le nombre de colonnes retrourné par le Transformer est différent du nombre d'origine n'est pas traité
                if isinstance(X_entity_inverted, np.ndarray):
                    if X_entity_inverted.shape[1] != len(self.transform_columns_):
                        raise NotImplementedError("Do not handle the case when the number of columns returned is different from the number of transformed columns for 'inverse_tranform'")
                    else :
                        for i, col in enumerate(self.transform_columns_):
                            df_part[col] = X_entity_inverted[:, i]
                else:
                    for col in X_entity_inverted.columns:
                        df_part[col] = X_entity_inverted[col].values

                # Ajout à la liste des données transformées
                inverted_parts.append(df_part)

            except Exception as e:
                # Gestion de l'erreur de transformation
                self._handle_entity_error(entity_key, e, "inverse_transforming")
                # Conservation des données non transformées
                inverted_parts.append(X.loc[group_idx])

        # Concaténation et restauration de l'ordre
        X_result = pd.concat(inverted_parts, axis=0)
        X_result = X_result.loc[X.index]

        # Restauration de la structure originale si conversion appliquée
        X_result = self._restore_structure_if_converted(X_result)

        return X_result

    # Méthode auxiliaire de normalisation de l'entité sous forme de tuple
    def _normalize_entity_key(self, key) -> tuple:
        """Normalize entity key to tuple format.

        Args:
            key: Entity key (peut être scalaire ou tuple)

        Returns:
            Clé normalisée en tuple
        """
        if isinstance(key, tuple):
            return key
        return (key,)

    # Méthode auxiliaire de gestion de l'erreur
    def _handle_entity_error(
        self,
        entity_key: tuple,
        error: Exception,
        operation: str
    ) -> None:
        """Handle errors during entity processing.

        Args:
            entity_key: Identifiant de l'entité
            error: Exception levée
            operation: Type d'opération ('fitting', 'transforming', etc.)
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

    # Méthode auxiliaire de gestion des entités inconnues
    def _handle_unknown_entity(
        self,
        entity_key: tuple
    ) -> None:
        """Handle entities not seen during fit.

        Args:
            entity_key: Identifiant de l'entité inconnue
            entity_data: Données de l'entité
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

    # Méthode auxilaire de générationn de noms de colonnes
    def _generate_column_names(
        self,
        n_cols: int,
        transformer: BaseEstimator
    ) -> List[str]:
        """Generate column names for transformed output.

        Args:
            n_cols: Nombre de colonnes dans la sortie
            transformer: Transformer ayant généré les colonnes

        Returns:
            Liste des noms de colonnes
        """
        # Tentative de récupération des noms via get_feature_names_out
        if hasattr(transformer, 'get_feature_names_out'):
            try:
                return list(transformer.get_feature_names_out())
            except Exception:
                pass

        # Noms génériques
        return [f"feature_{i}" for i in range(n_cols)]

    # Méthode auxiliaire des paramètres
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

    # Propriété d'extraction du nombre d'entités pour lesquelles un transformer est estimé
    @property
    def n_entities_(self) -> int:
        """Number of entities with fitted transformers.

        Examples:
            >>> import pandas as pd
            >>> from sklearn.preprocessing import StandardScaler
            >>>
            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2023-01-01', periods=10).tolist() * 3,
            ...     'country': ['FR'] * 10 + ['DE'] * 10 + ['IT'] * 10,
            ...     'value': range(30)
            ... })
            >>>
            >>> transformer = PanelwiseTransformer(
            ...     transformer=StandardScaler(),
            ...     time_col='date',
            ...     panel_cols=['country']
            ... )
            >>> transformer.fit(df)
            >>>
            >>> # Check number of entities
            >>> print(transformer.n_entities_)
            3
        """
        # Vérification que les transformers sont entraîné
        check_is_fitted(self, ['transformers_'])
        return len(self.transformers_)

    # Constructeur représentant le transformer sous forme de chaîne de caractère
    def __repr__(self) -> str:
        """String representation of the transformer."""
        transformer_repr = repr(self.transformer)
        return (
            f"PanelwiseTransformer(\n"
            f"    transformer={transformer_repr},\n"
            f"    panel_cols={self.panel_cols},\n"
            f"    time_col='{self.time_col}'\n"
            f")"
        )
