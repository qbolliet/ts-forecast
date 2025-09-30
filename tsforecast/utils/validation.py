"""Time series and panel data validation utilities.

This module provides sklearn-compatible transformers for validating and preparing
time series and panel data structures.
"""
# Importation des modules
# Modules de base
import pandas as pd
from typing import Optional, List, Union
import warnings
# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Modules de package
from .base_transformers import ReversibleTransformerMixin


class TimeSeriesValidator(BaseEstimator, TransformerMixin, ReversibleTransformerMixin):
    """Validate and prepare time series or panel data with reversible index transformations.

    This transformer validates temporal data structures and optionally converts
    time and panel columns into a proper index. It supports both simple time series
    and panel data, and can reverse the transformation to restore the original structure.

    Parameters:
        time_col: Name of the time column (if None, uses index)
        panel_cols: List of panel identifier columns (optional)
        strict: If True, raises errors on validation failures; if False, attempts corrections
        sort_data: If True, sorts data by index after validation

    Attributes:
        original_index_ (pd.Index): Original index before transformation
        original_structure_ (Dict): Dictionary containing original structure metadata
        had_explicit_time_col_ (bool): Whether time_col was in columns before transform
        had_explicit_panel_cols_ (bool): Whether panel_cols were in columns before transform
        index_was_replaced_ (bool): Whether the original index was replaced during transform
        time_col_ (str): Validated time column name
        panel_cols_ (List[str]): Validated panel column names

    Examples:
        >>> import pandas as pd
        >>> from tsforecast.utils.validation import TimeSeriesValidator
        >>>
        >>> # Example 1: Simple time series with index
        >>> dates = pd.date_range('2023-01-01', periods=5, freq='D')
        >>> data = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=dates)
        >>> validator = TimeSeriesValidator()
        >>> validated = validator.fit_transform(data)
        >>>
        >>> # Example 2: Panel data with time_col and panel_cols
        >>> data = pd.DataFrame({
        ...     'date': pd.date_range('2023-01-01', periods=6, freq='D'),
        ...     'country': ['US', 'US', 'US', 'FR', 'FR', 'FR'],
        ...     'value': [10, 20, 30, 40, 50, 60]
        ... })
        >>> validator = TimeSeriesValidator(time_col='date', panel_cols=['country'])
        >>> validated = validator.fit_transform(data)
        >>> # Index is now MultiIndex with (country, date)
        >>>
        >>> # Example 3: Reverse transformation
        >>> original = validator.inverse_transform(validated)
        >>> # Original structure restored
    """

    # Initialisation
    def __init__(self,
                 time_col: Optional[str] = None,
                 panel_cols: Optional[List[str]] = None,
                 strict: bool = True,
                 sort_data: bool = True):
        """Initialize the time series validator.

        Args:
            time_col: Name of the time column in data (None for index-based validation)
            panel_cols: List of panel identifier column names
            strict: Whether to raise errors (True) or attempt corrections (False)
            sort_data: Whether to sort data by index after validation
        """
        self.time_col = time_col
        self.panel_cols = panel_cols or []
        self.strict = strict
        self.sort_data = sort_data

    # Méthode fit
    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[pd.Series] = None) -> 'TimeSeriesValidator':
        """Fit the validator (stores metadata but no actual fitting needed).

        Args:
            X: Input data to validate
            y: Target variable (ignored, for sklearn compatibility)

        Returns:
            Self for method chaining
        """
        # Validation que l'entrée est pandas Series ou DataFrame
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            raise ValueError("Input must be a pandas Series or DataFrame")

        # Stockage du type d'entrée
        self.input_type_ = 'series' if isinstance(X, pd.Series) else 'dataframe'

        # Marqueur indiquant que le transformer est ajusté
        self.is_fitted_ = True

        return self

    # Méthode transform
    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Validate and transform data to proper time series format.

        Args:
            X: Input data to validate and transform

        Returns:
            Validated and transformed data with proper datetime index

        Raises:
            ValueError: If validation fails and strict=True
        """
        # Vérification que le transformer est ajusté
        check_is_fitted(self, 'is_fitted_')

        # Validation que l'entrée est du bon type
        if not isinstance(X, (pd.Series, pd.DataFrame)):
            raise ValueError("Input must be a pandas Series or DataFrame")

        # Conversion Series en DataFrame pour traitement uniforme
        X_work = X.to_frame() if isinstance(X, pd.Series) else X.copy()

        # Stockage de la structure originale
        self.original_index_ = X_work.index.copy()
        self.original_structure_ = {
            'index_name': X_work.index.name,
            'index_names': X_work.index.names if isinstance(X_work.index, pd.MultiIndex) else None,
            'columns': X_work.columns.tolist(),
            'index_type': type(X_work.index).__name__
        }

        # Détermination du mode de validation
        if self.time_col is None and not self.panel_cols:
            # Mode 1: Validation basée sur l'index
            X_validated = self._validate_index_based(X_work)
            self.index_was_replaced_ = False
            self.had_explicit_time_col_ = False
            self.had_explicit_panel_cols_ = False
        else:
            # Mode 2: Validation basée sur les colonnes
            X_validated = self._validate_column_based(X_work)
            self.index_was_replaced_ = True
            self.had_explicit_time_col_ = self.time_col is not None and self.time_col in X_work.columns
            self.had_explicit_panel_cols_ = bool(self.panel_cols) and all(col in X_work.columns for col in self.panel_cols)

        # Tri des données si demandé
        if self.sort_data:
            X_validated = X_validated.sort_index()

        # Retour au format Series si l'entrée était une Series
        if isinstance(X, pd.Series):
            return X_validated.iloc[:, 0]

        return X_validated

    # Méthode de validation basée sur l'index
    def _validate_index_based(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate data using index as time reference.

        Args:
            X: Input DataFrame

        Returns:
            Validated DataFrame

        Raises:
            ValueError: If index validation fails
        """
        # Cas 1: Index simple
        if not isinstance(X.index, pd.MultiIndex):
            # Vérification et conversion en DatetimeIndex
            if not isinstance(X.index, pd.DatetimeIndex):
                try:
                    X.index = pd.to_datetime(X.index)
                except Exception as e:
                    if self.strict:
                        raise ValueError(f"Index cannot be converted to datetime: {e}")
                    else:
                        warnings.warn(f"Index conversion to datetime failed: {e}")
                        return X

            # Vérification de l'unicité
            if X.index.duplicated().any():
                if self.strict:
                    raise ValueError("Index contains duplicate values")
                else:
                    warnings.warn("Index contains duplicates. Keeping first occurrence.")
                    X = X[~X.index.duplicated(keep='first')]

        # Cas 2: MultiIndex
        else:
            # Vérification que le dernier niveau est datetime
            last_level_idx = -1
            last_level = X.index.get_level_values(last_level_idx)

            if not isinstance(last_level, pd.DatetimeIndex):
                try:
                    # Conversion du dernier niveau en datetime
                    new_levels = list(X.index.levels)
                    new_levels[last_level_idx] = pd.to_datetime(new_levels[last_level_idx])

                    # Reconstruction du MultiIndex avec le niveau converti
                    new_codes = X.index.codes
                    X.index = pd.MultiIndex(levels=new_levels, codes=new_codes, names=X.index.names)
                except Exception as e:
                    if self.strict:
                        raise ValueError(f"Last level of MultiIndex cannot be converted to datetime: {e}")
                    else:
                        warnings.warn(f"MultiIndex last level conversion failed: {e}")
                        return X

            # Vérification de l'unicité de la combinaison des niveaux
            if X.index.duplicated().any():
                if self.strict:
                    raise ValueError("MultiIndex contains duplicate combinations")
                else:
                    warnings.warn("MultiIndex contains duplicates. Keeping first occurrence.")
                    X = X[~X.index.duplicated(keep='first')]

        return X

    # Méthode de validation basée sur les colonnes
    def _validate_column_based(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate data using time_col and panel_cols, then set as index.

        Args:
            X: Input DataFrame

        Returns:
            Validated DataFrame with new index

        Raises:
            ValueError: If column validation fails
        """
        # Vérification de la cohérence des paramètres
        if self.panel_cols and self.time_col is None:
            raise ValueError("Cannot specify panel_cols without time_col")

        # Vérification de la présence de time_col
        if self.time_col and self.time_col not in X.columns:
            raise ValueError(f"Time column '{self.time_col}' not found in data")

        # Vérification de la présence des panel_cols
        if self.panel_cols:
            missing_cols = set(self.panel_cols) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Panel columns not found: {missing_cols}")

        # Conversion de la colonne temporelle
        if self.time_col:
            try:
                X[self.time_col] = pd.to_datetime(X[self.time_col])
            except Exception as e:
                raise ValueError(f"Cannot convert time column '{self.time_col}' to datetime: {e}")

        # Création des colonnes d'index
        index_cols = []
        if self.panel_cols:
            index_cols.extend(self.panel_cols)
        if self.time_col:
            index_cols.append(self.time_col)

        # Vérification de l'unicité de la combinaison
        if X.duplicated(subset=index_cols).any():
            if self.strict:
                raise ValueError(f"Duplicate rows found for combination of columns: {index_cols}")
            else:
                warnings.warn(f"Duplicates found for {index_cols}. Keeping first occurrence.")
                X = X.drop_duplicates(subset=index_cols, keep='first')

        # Stockage des colonnes qui vont devenir index
        self.time_col_ = self.time_col
        self.panel_cols_ = self.panel_cols.copy() if self.panel_cols else []

        # Définition du nouvel index
        X = X.set_index(index_cols)

        # Message d'avertissement sur le remplacement de l'index
        warnings.warn(
            f"Index replaced with {index_cols}. Original index stored for inverse transformation.",
            UserWarning
        )

        return X

    # Méthode inverse_transform
    def inverse_transform(self, X: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Reverse transformation to restore original data structure.

        Args:
            X: Transformed data

        Returns:
            Data with original structure restored
        """
        # Vérification que le transformer est ajusté
        check_is_fitted(self, 'is_fitted_')

        # Conversion Series en DataFrame pour traitement uniforme
        X_work = X.to_frame() if isinstance(X, pd.Series) else X.copy()

        # Si l'index n'a pas été remplacé, retourner tel quel avec l'index original
        if not self.index_was_replaced_:
            X_work.index = self.original_index_
            return X_work.iloc[:, 0] if isinstance(X, pd.Series) else X_work

        # Restauration de l'index en colonnes si nécessaire
        if self.had_explicit_time_col_ or self.had_explicit_panel_cols_:
            X_work = X_work.reset_index()

        # Restauration de l'index original
        X_work.index = self.original_index_

        # Restauration du nom de l'index si applicable
        if self.original_structure_.get('index_name'):
            X_work.index.name = self.original_structure_['index_name']

        # Retour au format Series si l'entrée était une Series
        if isinstance(X, pd.Series):
            return X_work.iloc[:, 0]

        return X_work
