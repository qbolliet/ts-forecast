"""Base classes for time series transformers.

This module provides abstract base classes and mixins for creating
sklearn-compatible time series transformers.
"""
# Importation des modules
# Modules de base
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import warnings
# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Importation des fonctions de validation
from .validation import (
    restore_original_structure,
    validate_temporal_data,
    validate_entities_grouped,
    validate_sorted_within_groups
)


# Mixin pour les séries temporelles
class TimeSeriesTransformerMixin:
    """Mixin class for time series transformers.
    
    Provides common functionality for time series data handling.
    """
    
    # Méthode auxiliaire de validation de l'index
    def _validate_time_index(self, X: pd.DataFrame, time_col: Optional[str] = None) -> pd.DatetimeIndex:
        """Validate and extract time index from data.
        
        Args:
            X: Input data
            time_col: Name of time column
            
        Returns:
            DatetimeIndex
            
        Raises:
            ValueError: If time index cannot be determined
        """
        if time_col and time_col in X.columns:
            # Utilisation de la colonne temporelle spécifiée
            time_index = pd.to_datetime(X[time_col])
        elif isinstance(X.index, pd.DatetimeIndex):
            # L'index est déjà un DatetimeIndex
            time_index = X.index
        else:
            # Tentative de conversion de l'index
            try:
                time_index = pd.to_datetime(X.index)
            except:
                raise ValueError(
                    "Cannot determine time index. Please provide a datetime index "
                    "or specify time_col parameter."
                )
        
        return time_index
    

# Transformer pour les données de panel
class PanelTimeSeriesTransformer(BaseEstimator, TransformerMixin, TimeSeriesTransformerMixin, ABC):
    """Abstract base class for panel time series transformers.
    
    This class provides a template for creating transformers that handle
    both univariate time series and panel data.
    
    Parameters:
        time_col (str): Name of the time column
        panel_cols (Optional[List[str]]): Columns identifying panel dimensions
        validate_input (bool): Whether to validate input data
        strict_validation (bool): If True, raises errors; if False, emits warnings
        auto_sort (bool): If True, automatically sorts unsorted data
    """

    # Initialisation
    def __init__(self,
                 time_col: str = 'date',
                 panel_cols: Optional[List[str]] = None,
                 validate_input: bool = True,
                 strict_validation: bool = True,
                 auto_sort: bool = False,
                 convert_cols_to_index: bool = False):
        """Initialize the transformer.

        Args:
            time_col: Nom de la colonne temporelle
            panel_cols: Colonnes identifiant les dimensions du panel
            validate_input: Active ou désactive la validation des données d'entrée
            strict_validation: Si True, lève des erreurs; si False, émet des warnings
            auto_sort: Si True, trie automatiquement les données mal ordonnées
            convert_cols_to_index: Si True, convertit time_col et panel_cols en index
                                 et stocke les métadonnées pour restauration ultérieure
        """
        # Initialisation des paramètres en attributs
        self.time_col = time_col
        self.panel_cols = panel_cols
        self.validate_input = validate_input
        self.strict_validation = strict_validation
        self.auto_sort = auto_sort
        self.convert_cols_to_index = convert_cols_to_index

        # Attribut pour stocker les métadonnées de conversion
        self.conversion_metadata_ = None
    
    # Méthode d'entraînement
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PanelTimeSeriesTransformer':
        """Fit the transformer.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            Self for method chaining
        """
        # Validation des données si demandée
        if self.validate_input:
            X = self._validate_input(X)
        
        # Détermination du type de données
        self.is_panel_ = self.panel_cols is not None and len(self.panel_cols) > 0
        
        # Stockage des métadonnées
        self.n_features_ = X.shape[1]
        self.feature_names_ = X.columns.tolist()
        
        # Vérification de la cohérence des individus du panel
        if self.is_panel_:
            is_consistent, issues = self._check_panel_consistency(X, self.panel_cols)
            if not is_consistent:
                warnings.warn(f"Panel structure issues: {'; '.join(issues)}")
        
        # Appel de la méthode spécifique à implémenter
        self._fit(X, y)
        
        return self
    
    # Méthode auxiliaire d'entraînement
    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Fit implementation to be provided by subclasses.
        
        Args:
            X: Input features
            y: Target variable (optional)
        """
        pass
    
    # Méthode de transformation
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        # Vérification que le transformer est ajusté
        check_is_fitted(self)
        
        # Validation des données
        if self.validate_input:
            X = self._validate_input(X)
        
        # Appel de la méthode spécifique
        return self._transform(X)
    
    # Méthode auxiliaire de transformation
    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform implementation to be provided by subclasses.
        
        Args:
            X: Input data
            
        Returns:
            Transformed data
        """
        pass
    
    # Méthode auxiliaire de validation des données
    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate input data with comprehensive temporal checks.

        Validation complète incluant:
        - Type et présence des colonnes
        - Validation temporelle (index ou colonne)
        - Unicité des combinaisons (entity, date)
        - Tri temporel global et intra-groupe (panel)
        - Groupement contigü des entités (panel)

        Args:
            X: Input data

        Returns:
            Validated (et potentiellement trié si auto_sort=True)

        Raises:
            ValueError: If validation fails
        """
        # 1. Vérification type de base
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # 2. Vérification colonnes panel
        if self.panel_cols:
            missing_cols = set(self.panel_cols) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Panel columns not found: {missing_cols}")

        # 3. Vérification colonne/index temporel (méthode existante)
        try:
            self._validate_time_index(X, self.time_col)
        except ValueError as e:
            raise ValueError(f"Time validation failed: {str(e)}")

        # 4. Validation complète via validate_temporal_data
        # Si convert_cols_to_index=True, demander les métadonnées pour restauration
        try:
            if self.convert_cols_to_index:
                # Demande des métadonnées pour restauration ultérieure
                X_validated, metadata = validate_temporal_data(
                    data=X,
                    time_col=self.time_col if self.time_col in X.columns else None,
                    panel_cols=self.panel_cols,
                    strict=self.strict_validation,
                    sort_data=self.auto_sort,
                    return_metadata=True
                )
                # Stockage des métadonnées pour inverse_transform
                self.conversion_metadata_ = metadata
            else:
                # Validation sans métadonnées
                X_validated = validate_temporal_data(
                    data=X,
                    time_col=self.time_col if self.time_col in X.columns else None,
                    panel_cols=self.panel_cols,
                    strict=self.strict_validation,
                    sort_data=self.auto_sort,
                    return_metadata=False
                )
        except ValueError as e:
            if self.strict_validation:
                raise ValueError(f"Temporal data validation failed: {str(e)}")
            else:
                warnings.warn(f"Temporal data validation warning: {str(e)}")
                X_validated = X

        # 5. Validations spécifiques panel
        if self.panel_cols:
            # Vérification groupement des entités
            if not validate_entities_grouped(X_validated, panel_cols=self.panel_cols):
                msg = (
                    "Panel entities are not contiguous. "
                    "Consider sorting by panel_cols or enabling auto_sort=True"
                )
                if self.strict_validation:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)

            # Vérification tri intra-groupe
            if not validate_sorted_within_groups(
                X_validated,
                panel_cols=self.panel_cols,
                time_col=self.time_col if self.time_col in X.columns else None
            ):
                msg = (
                    "Time series not sorted within panel groups. "
                    "Consider enabling auto_sort=True"
                )
                if self.strict_validation:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)

        # 6. Validation série temporelle simple (non-panel)
        elif isinstance(X_validated.index, pd.DatetimeIndex):
            if not X_validated.index.is_monotonic_increasing:
                msg = "Time series index is not sorted"
                if self.strict_validation:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)

        return X_validated
    
    # Fonction auxiliaire de vérification de la consistence des éléments du panel
    def _check_panel_consistency(self, 
                               X: pd.DataFrame, 
                               panel_cols: List[str]) -> Tuple[bool, List[str]]:
        """Check consistency of panel structure.
        
        Args:
            X: Input data
            panel_cols: Panel identifier columns
            
        Returns:
            Tuple of (is_consistent, list_of_issues)
        """
        # Initialisation de la liste des problèmes
        issues = []
        
        # Vérification de la présence des colonnes panel
        missing_cols = set(panel_cols) - set(X.columns)
        if missing_cols:
            issues.append(f"Missing panel columns: {missing_cols}")
            return False, issues
        
        # Vérification des valeurs manquantes dans les identifiants
        for col in panel_cols:
            if X[col].isnull().any():
                issues.append(f"Missing values in panel identifier: {col}")
        
        return len(issues) == 0, issues

# Mixin d'inversion des transformations
class ReversibleTransformerMixin:
    """Mixin for transformers that support inverse transformation.
    
    Provides template for implementing reversible transformations.
    """
    
    # Méthode d'inversion de la transformation
    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the transformation.
        
        Args:
            X: Transformed data
            
        Returns:
            Original data format
        """
        pass
    
    # Méthode de stockage des méta-données (index, colonnes, dimensions) des jeux de données initial et transformé
    def _store_transformation_info(self, X: pd.DataFrame, X_transformed: pd.DataFrame) -> None:
        """Store information needed for inverse transformation.

        Args:
            X: Original data
            X_transformed: Transformed data
        """
        # Stockage des informations de forme
        self.original_shape_ = X.shape
        self.transformed_shape_ = X_transformed.shape

        # Stockage des colonnes
        self.original_columns_ = X.columns.tolist()
        self.transformed_columns_ = X_transformed.columns.tolist()

        # Stockage de l'index si différent
        if not X.index.equals(X_transformed.index):
            self.original_index_ = X.index
            self.transformed_index_ = X_transformed.index

    # FOnction de restauration de la strcuture originale des données
    def _restore_structure_if_converted(self, X: pd.DataFrame) -> pd.DataFrame:
        """Restaurer la structure originale si conversion a été appliquée.

        Restaure la structure originale des données (colonnes au lieu d'index)
        si une conversion a été effectuée via convert_cols_to_index=True.

        Args:
            X: Données transformées avec index modifié

        Returns:
            Données avec structure originale restaurée

        Examples:
            >>> # Données après transformation avec convert_cols_to_index=True
            >>> X_transformed = transformer.transform(X)  # Index = (entity, date)
            >>> # Restauration dans inverse_transform
            >>> X_restored = transformer.inverse_transform(X_transformed)
            >>> # X_restored a maintenant 'entity' et 'date' comme colonnes
        """
        # Vérification que les métadonnées existent
        if hasattr(self, 'conversion_metadata_') and self.conversion_metadata_ is not None:
            # Restauration de la structure originale
            return restore_original_structure(X, self.conversion_metadata_)
        # Pas de conversion appliquée : retour inchangé
        return X
