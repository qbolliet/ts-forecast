"""Adapter for hierarchicalforecast with sklearn-like API.

This module provides a wrapper around the hierarchicalforecast library
to enable seamless integration with sklearn pipelines and workflows.
"""
# Importation des modules
# Modules de base
from typing import Dict, List, Union
# Sklearn
from sklearn.base import BaseEstimator

# Wrapper permettant l'intégration des modèles du package "hierarchicalforecast" dans un syntaxe "sklearn-like"
class HierarchicalForecastAdapter(BaseEstimator):
    """Adapter for hierarchicalforecast reconciliation with sklearn-like API.

    This adapter wraps the hierarchicalforecast library's aggregation and
    reconciliation functionality, allowing users to work with pandas DataFrames
    that have datetime or MultiIndex indices directly.

    Args:
        reconcilers: List of instantiated reconciliation method objects from
            hierarchicalforecast.methods (e.g., BottomUp(), MinTrace()).
        spec: Hierarchy specification. Either a list of lists of column names
            for cross-sectional hierarchies (used with `aggregate`), or a dict
            mapping temporal level names to aggregation factors (used with
            `aggregate_temporal`).
        exog_vars: Dictionary specifying how to aggregate exogenous variables.
            Keys are column names, values are aggregation functions ('sum', 'mean')
            or lists of functions.
        sparse_s: If True, return the summing matrix S as a sparse matrix.
        id_col: Name of the column to use for unique series identifiers.
        time_col: Name of the column to use for timestamps.
        id_time_col: Name of the column for temporal hierarchy identifiers.
            Defaults to None for cross-sectional and 'temporal_id' for temporal.
        target_cols: Sequence of column names containing target values to aggregate.
        aggregation_type: Type of temporal aggregation ('local' or 'global').
            Only used when spec is a dict (temporal hierarchy). Ignored otherwise.
        level: Confidence levels for prediction intervals (e.g., [80, 95]).
        intervals_method: Method for computing prediction intervals.
            One of 'normality', 'bootstrap', 'permbu'.
        num_samples: Number of samples for probabilistic reconciliation.
            If positive, returns probabilistic coherent samples.

    Attributes:
        Y_df_: DataFrame containing aggregated time series after fit.
        S_df_: DataFrame containing the summing matrix after fit.
        tags_: Dictionary mapping hierarchy levels to their unique_id values.
        hrec_: HierarchicalReconciliation instance created during fit.

    Examples:
        Cross-sectional hierarchy::

            from hierarchicalforecast.methods import BottomUp, MinTrace

            spec = [
                ['Country'],
                ['Country', 'State'],
                ['Country', 'State', 'Region'],
            ]
            adapter = HierarchicalForecastAdapter(
                reconcilers=[BottomUp(), MinTrace(method='mint_shrink')],
                spec=spec,
            )
            # y has MultiIndex (Country, State, Region, date)
            adapter.fit(None, y_train)
            y_reconciled = adapter.predict(y_hat)

        Temporal hierarchy::

            spec_temporal = {
                '4-periods': 4,
                '2-periods': 2,
                '1-period': 1,
            }
            adapter = HierarchicalForecastAdapter(
                reconcilers=[MinTrace(method='ols')],
                spec=spec_temporal,
                aggregation_type='global',
            )
            adapter.fit(None, y_train)
            y_reconciled = adapter.predict(y_hat)
    """

    # Initialisation
    def __init__(
        self,
        reconcilers: List,
        spec: Union[List[List[str]], Dict[str, int]],
        exog_vars: Optional[Dict[str, Union[str, List[str]]]] = None,
        sparse_s: bool = False,
        id_col: str = "unique_id",
        time_col: str = "ds",
        id_time_col: Optional[str] = None,
        target_cols: Sequence[str] = ("y",),
        aggregation_type: str = "local",
        level: Optional[List[float]] = None,
        intervals_method: Optional[str] = None,
        num_samples: int = -1,
    ):
        # Initialisation des attributs à partir des arguments
        self.reconcilers = reconcilers
        self.spec = spec
        self.exog_vars = exog_vars
        self.sparse_s = sparse_s
        self.id_col = id_col
        self.time_col = time_col
        self.id_time_col = id_time_col
        self.target_cols = target_cols
        self.aggregation_type = aggregation_type
        self.level = level
        self.intervals_method = intervals_method
        self.num_samples = num_samples

    # Le type de hiérarchie dépend de type de l'"argument" spec qui est un dictionnaire s'il s'agit de d'une agrégation temporelle et d'une liste sinon
    @property
    def _is_temporal_hierarchy(self) -> bool:
        """Check if spec defines a temporal hierarchy."""
        return isinstance(self.spec, dict)

    # Méthode auxiliaire de détection du niveau du datetime
    def _find_datetime_level(self, index: pd.Index) -> Optional[int]:
        """Find the index level that contains datetime values.

        Args:
            index: pandas Index or MultiIndex to analyze.

        Returns:
            Index position of the datetime level, or None if not found.
            For MultiIndex, returns the level number. For simple Index, returns 0.
        """
        
        # Un MultiIndex correspond à des données de panel, on vérifie que le dernier niveau correspond bien à des dates
        if isinstance(index, pd.MultiIndex):
            # Extraction des valeurs du dernier niveau
            last_level_values = index.get_level_values(-1)
            # Vérification que le type est un index
            if isinstance(last_level_values, pd.DatetimeIndex) | pd.api.types.is_datetime64_any_dtype(last_level_values):
                return len(index.levels) - 1

        # Vérification que l'index est un datetime ou contient des datetime
        if isinstance(index, pd.DatetimeIndex) | pd.api.types.is_datetime64_any_dtype(index):
            return 0

        return None

    # Méthode de conversion de l'index en colonnes
    def _convert_index_to_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime index or MultiIndex to columns.

        Transforms a DataFrame with datetime/MultiIndex into the flat format
        expected by hierarchicalforecast's aggregate functions.

        Args:
            df: DataFrame with datetime Index or MultiIndex where the last
                level (or a detected datetime level) represents time.

        Returns:
            DataFrame with index levels converted to columns. The datetime
            level is renamed to match `time_col`.

        Raises:
            ValueError: If no datetime level can be identified in the index.
        """
        # Copie indépendante du jeu de données
        df = df.copy()

        # Extraction du niveau de l'index de dates
        datetime_level = self._find_datetime_level(df.index)

        # Renvoie un message d'erreur si l'index de date n'est pas trouvé
        if datetime_level is None:
            raise ValueError(
                "Could not identify a datetime level in the index. "
                "Ensure the DataFrame has a DatetimeIndex or a MultiIndex "
                "with a datetime level."
            )

        # Distinction suivant le type d'index
        if isinstance(df.index, pd.MultiIndex):
            # Extraction des noms des niveaux avant reset
            level_names = list(df.index.names)
            # Comptage du nombre de niveaux
            n_levels = len(level_names)

            # Réinitialisation de l'index 
            df.reset_index(inplace=True)

            # Identification du nom de la colonne 'datetime' après la réinitilisation de l'index
            datetime_col_name = level_names[datetime_level]

            # Renommage de la colonne de temps
            if datetime_col_name is None:
                datetime_col_name = self.time_col
            elif datetime_col_name != self.time_col:
                df.rename({datetime_col_name: self.time_col}, axis=1, inplace=True)
        # Index simple (DatetimeIndex ou autre avec datetime)
        else:
            # Extraction du nom de l'index
            index_name = df.index.name
            # Réinitialisation de l'index
            df.reset_index(inplace=True)

            # Renommage la colonne d'index en 'time_col'
            col_to_rename = index_name if index_name is not None else "index"
            if col_to_rename != self.time_col :
                df = df.rename(columns={col_to_rename: self.time_col})

        return df

    # Méthode d'entraînement
    def fit(self, X, y):
        """Fit the hierarchical reconciliation adapter.

        Aggregates the input time series according to the specified hierarchy
        and prepares internal structures for reconciliation.

        Args:
            X: Ignored. Present for sklearn API compatibility.
            y: Time series data as a DataFrame or Series with datetime Index
                or MultiIndex. For panel data, the MultiIndex should have
                entity levels followed by a datetime level (typically last).

        Returns:
            self: The fitted adapter instance.

        Raises:
            ValueError: If y has no identifiable datetime index level.
        """
        # Importation pour éviter les erreurs si le package n'est pas installé
        from hierarchicalforecast.core import HierarchicalReconciliation
        from hierarchicalforecast.utils import aggregate, aggregate_temporal

        # Conversion de la Series en DataFrame si nécessaire
        if isinstance(y, pd.Series):
            col_name = y.name if y.name is not None else self.target_cols[0]
            y = y.to_frame(name=col_name)

        # Conversion de l'index en colonnes
        df = self._convert_index_to_columns(y)

        # Appel de la fonction d'agrégation appropriée selon le type de spec
        if self._is_temporal_hierarchy:
            # Extraction du nom de la colonne temporelle
            effective_id_time_col = (
                self.id_time_col if self.id_time_col is not None else "temporal_id"
            )
            # Extraction des jeux de données au format adapté
            self.Y_df_, self.S_df_, self.tags_ = aggregate_temporal(
                df=df,
                spec=self.spec,
                exog_vars=self.exog_vars,
                sparse_s=self.sparse_s,
                id_col=self.id_col,
                time_col=self.time_col,
                id_time_col=effective_id_time_col,
                target_cols=self.target_cols,
                aggregation_type=self.aggregation_type,
            )
        else:
            # Extraction des jeux de données au format adapté
            self.Y_df_, self.S_df_, self.tags_ = aggregate(
                df=df,
                spec=self.spec,
                exog_vars=self.exog_vars,
                sparse_s=self.sparse_s,
                id_col=self.id_col,
                time_col=self.time_col,
                id_time_col=self.id_time_col,
                target_cols=self.target_cols,
            )

        # Création de l'objet de réconciliation
        self.hrec_ = HierarchicalReconciliation(reconcilers=self.reconcilers)

        return self

    # Méthode de prédiction
    def predict(self, X):
        """Reconcile base forecasts using the fitted hierarchy.

        Args:
            X: Base forecasts as a DataFrame with the same index structure
                as the y provided during fit. Should contain columns for each
                model's predictions to be reconciled.

        Returns:
            DataFrame containing reconciled forecasts for all hierarchy levels.
            Includes columns for each reconciliation method applied.

        Raises:
            RuntimeError: If called before fit().
            ValueError: If X has incompatible structure with fitted hierarchy.
        """
        # Vérification qu'un réconciliateur avait bien été initialisé
        if not hasattr(self, "hrec_"):
            raise RuntimeError(
                "This adapter instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'predict'."
            )

        # Convertion de l'index en colonnes
        Y_hat_df = self._convert_index_to_columns(X)

        # Construction des arguments pour reconcile à partir des attributs
        reconcile_kwargs = {
            "Y_hat_df": Y_hat_df,
            "Y_df": self.Y_df_,
            "S": self.S_df_,
            "tags": self.tags_,
        }

        # Ajout des arguments optionnels s'ils sont définis
        if self.level is not None:
            reconcile_kwargs["level"] = self.level

        if self.intervals_method is not None:
            reconcile_kwargs["intervals_method"] = self.intervals_method

        if self.num_samples != -1:
            reconcile_kwargs["num_samples"] = self.num_samples

        # Réconciliation
        Y_rec = self.hrec_.reconcile(**reconcile_kwargs)

        return Y_rec

    # Méthode combinée d'entraînement et de prédiction
    def fit_predict(self, X, y):
        """Fit the adapter and reconcile forecasts in one step.

        Convenience method that calls fit() followed by predict().

        Args:
            X: Base forecasts to reconcile (passed to predict).
            y: Historical time series data (passed to fit).

        Returns:
            DataFrame containing reconciled forecasts.
        """
        return self.fit(X=None, y=y).predict(X)

    # Méthode d'extraction d'informations sur la hiérarchie
    def get_hierarchy_info(self) -> Dict:
        """Get information about the fitted hierarchy.

        Returns:
            Dictionary containing:
                - 'n_series': Total number of series in hierarchy
                - 'n_bottom': Number of bottom-level series
                - 'levels': List of hierarchy level names
                - 'tags': The tags dictionary mapping levels to series

        Raises:
            RuntimeError: If called before fit().
        """
        # Vérification qu'un attribut "tags_" est instancié
        if not hasattr(self, "tags_"):
            raise RuntimeError(
                "This adapter instance is not fitted yet. "
                "Call 'fit' before accessing hierarchy information."
            )

        # Extraction du nombre total de séries
        n_total = len(self.S_df_) if hasattr(self, "S_df_") else 0

        # Le nombre de séries bottom correspond aux colonnes de S
        # (excluant la colonne id si présente)
        if hasattr(self, "S_df_") and self.S_df_ is not None:
            s_cols = [c for c in self.S_df_.columns if c != self.id_col]
            n_bottom = len(s_cols)
        else:
            n_bottom = 0

        return {
            "n_series": n_total,
            "n_bottom": n_bottom,
            "levels": list(self.tags_.keys()),
            "tags": self.tags_,
        }