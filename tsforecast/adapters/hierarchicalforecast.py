"""Adapter for hierarchicalforecast with sklearn-like API.

This module provides a wrapper around the hierarchicalforecast library
to enable seamless integration with sklearn pipelines and workflows.
"""
# Importation des modules
# Modules de base
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import warnings
# Sklearn
from sklearn.base import BaseEstimator, RegressorMixin


# Constantes internes correspondant aux conventions de hierarchicalforecast
_ID_COL = "unique_id"
_TIME_COL = "ds"


# Wrapper permettant l'intégration des modèles du package "hierarchicalforecast" dans une syntaxe "sklearn-like"
# /!\ La méthode "score" (nécessaire pour GridSearchCV) et l'héritage de RegressorMixin font que l'on wrappe des régresseurs et non des classifieurs par défaut
class HierarchicalForecastAdapter(BaseEstimator, RegressorMixin):
    """Adapter for hierarchicalforecast reconciliation with sklearn-like API.

    This adapter wraps the hierarchicalforecast library's aggregation and
    reconciliation functionality, allowing users to work with pandas DataFrames
    that have datetime or MultiIndex indices directly, without manually
    constructing the ``unique_id`` strings expected by hierarchicalforecast.

    The ``unique_id`` is built internally by joining the non-date index level
    values with ``"/"``. Aggregate-level rows in the prediction DataFrame
    should therefore use an empty string ``""`` for any index level that is
    absent at that aggregation depth (see ``predict`` for details).

    Note:
        ``id_col``, ``time_col``, and ``target_cols`` are not constructor
        parameters: they are handled automatically. The unique-id column is
        always ``"unique_id"``, the time column is always ``"ds"``, and the
        target column is inferred from the name of the ``y`` argument in
        ``fit``.

    Args:
        reconcilers: List of instantiated reconciliation method objects from
            ``hierarchicalforecast.methods`` (e.g., ``BottomUp()``,
            ``MinTrace()``).
        spec: Hierarchy specification. Either a list of lists of column names
            for cross-sectional hierarchies (used with ``aggregate``), or a
            dict mapping temporal level names to aggregation factors (used
            with ``aggregate_temporal``). Lists must be ordered from the most
            aggregated level to the most granular (bottom) level.
        exog_vars: Dictionary specifying how to aggregate exogenous variables.
            Keys are column names present in the ``X`` DataFrame passed to
            ``fit``; values are aggregation functions (``"sum"``, ``"mean"``)
            or lists of such functions.
        sparse_s: If ``True``, return the summing matrix ``S`` as a sparse
            matrix.
        id_time_col: Name of the column used for temporal hierarchy
            identifiers inside ``aggregate_temporal``. Only relevant when
            ``spec`` is a dict (temporal hierarchy). Defaults to
            ``"temporal_id"`` if ``None``.
        aggregation_type: Temporal aggregation strategy (``"local"`` or
            ``"global"``). Only used when ``spec`` is a dict.
        level: Confidence levels for prediction intervals (e.g.,
            ``[80, 95]``).
        intervals_method: Method for computing prediction intervals. One of
            ``"normality"``, ``"bootstrap"``, ``"permbu"``.
        num_samples: Number of samples for probabilistic reconciliation.
            If positive, returns probabilistic coherent samples.

    Attributes:
        Y_df_: DataFrame containing aggregated time series after ``fit``.
        S_df_: DataFrame containing the summing matrix after ``fit``.
        tags_: Dictionary mapping hierarchy levels to their ``unique_id``
            values.
        hrec_: ``HierarchicalReconciliation`` instance created during
            ``fit``.

    Examples:
        Cross-sectional hierarchy without exogenous variables::

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
            # y has MultiIndex (Country, State, Region, date) — bottom level only
            adapter.fit(None, y_train)
            # Y_hat has MultiIndex (Country, State, Region, date) for all levels;
            # aggregate rows use "" for absent lower levels
            y_reconciled = adapter.predict(Y_hat)

        Cross-sectional hierarchy with exogenous variables::

            adapter = HierarchicalForecastAdapter(
                reconcilers=[BottomUp()],
                spec=spec,
                exog_vars={"gdp": "sum", "population": "mean"},
            )
            # X has the same MultiIndex as y, with columns "gdp" and "population"
            adapter.fit(X_train, y_train)

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
            y_reconciled = adapter.predict(Y_hat)
    """

    # Initialisation
    def __init__(
        self,
        reconcilers: List,
        spec: Union[List[List[str]], Dict[str, int]],
        exog_vars: Optional[Dict[str, Union[str, List[str]]]] = None,
        sparse_s: bool = False,
        id_time_col: Optional[str] = None,
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
        self.id_time_col = id_time_col
        self.aggregation_type = aggregation_type
        self.level = level
        self.intervals_method = intervals_method
        self.num_samples = num_samples

    # Le type de hiérarchie dépend du type de l'argument "spec" : dictionnaire pour une agrégation temporelle, liste sinon
    @property
    def _is_temporal_hierarchy(self) -> bool:
        """Check if spec defines a temporal hierarchy."""
        return isinstance(self.spec, dict)

    # Méthode auxiliaire de détection du niveau du datetime dans l'index
    # /!\ Voir si on ne peut pas hybrider cette méthode avec les méthodes de validation dans utils/validation 
    def _find_datetime_level(self, index: pd.Index) -> Optional[int]:
        """Find the index level that contains datetime values.

        The last level of a MultiIndex is always assumed to be the date level
        when it contains datetime values. For a simple Index, level 0 is
        returned if it is datetime-typed.

        Args:
            index: pandas Index or MultiIndex to analyze.

        Returns:
            Index position of the datetime level, or ``None`` if not found.
        """
        # Vérification du dernier niveau pour un MultiIndex (données de panel)
        if isinstance(index, pd.MultiIndex):
            # Extraction des valeurs du dernier niveau de l'index (qui correspond à la date par convention)
            last_level_values = index.get_level_values(-1)
            if isinstance(last_level_values, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(last_level_values):
                return len(index.levels) - 1

        # Vérification de l'index simple (série temporelle)
        if isinstance(index, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(index):
            return 0

        return None

    # Méthode de conversion de l'index en colonnes plates
    def _index_to_flat(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str], str]:
        """Reset the index of a hierarchical DataFrame to flat columns.

        Transforms a DataFrame whose index encodes entity and time information
        into the flat column format expected by hierarchicalforecast's
        ``aggregate`` functions. The datetime level is always renamed to
        ``"ds"``; unnamed index levels are assigned placeholder names
        ``"level_i"``.

        Args:
            df: DataFrame with a DatetimeIndex (time series) or MultiIndex
                where the last level is the date.

        Returns:
            Tuple of:
                - ``df_flat``: DataFrame with the full index reset to columns
                  and the date column renamed to ``"ds"``.
                - ``non_date_cols``: Ordered list of column names that
                  originated from non-date index levels (entity and hierarchy
                  levels). Empty for a simple DatetimeIndex.
                - ``date_col_name``: Original name of the datetime level
                  before renaming to ``"ds"``.

        Raises:
            ValueError: If no datetime level can be identified in the index.
        """
        # Copie indépendante pour éviter toute modification en place du jeu de données original
        df = df.copy()

        # Extraction du niveau de l'index correspondant aux dates
        datetime_level = self._find_datetime_level(df.index)

        if datetime_level is None:
            raise ValueError(
                "Could not identify a datetime level in the index. "
                "Ensure the DataFrame has a DatetimeIndex or a MultiIndex "
                "whose last level contains datetime values."
            )

        # Traitement d'un MultiIndex (données de panel)
        if isinstance(df.index, pd.MultiIndex):
            # Attribution d'un nom aux niveaux anonymes
            level_names = [
                n if n is not None else f"level_{i}"
                for i, n in enumerate(df.index.names)
            ]
            df.index.names = level_names

            # Réinitialisation de l'index en colonnes
            df.reset_index(inplace=True)

            # Séparation du nom de la colonne date et des niveaux non-date
            date_col_name = level_names[datetime_level]
            non_date_cols = [
                n for i, n in enumerate(level_names) if i != datetime_level
            ]

        # Traitement d'un index simple (série temporelle)
        else:
            date_col_name = df.index.name if df.index.name is not None else "index"
            df.index.name = date_col_name
            df.reset_index(inplace=True)
            non_date_cols = []

        # Renommage de la colonne date vers la convention interne
        if date_col_name != _TIME_COL:
            df = df.rename(columns={date_col_name: _TIME_COL})

        return df, non_date_cols, date_col_name

    # Méthode de construction du "unique_id" à partir des colonnes non-date
    def _build_unique_id_series(
        self, df: pd.DataFrame, non_date_cols: List[str]
    ) -> pd.Series:
        """Build the ``unique_id`` column from non-date column values.

        For each row, joins the non-empty, non-null values of the specified
        columns with ``"/"``. Rows at an aggregate hierarchy level should use
        an empty string ``""`` for any column that does not apply at that
        depth; those empty parts are silently omitted from the identifier.

        Args:
            df: Flat DataFrame (index already reset) containing ``non_date_cols``.
            non_date_cols: Ordered list of column names whose values form the
                unique series identifier.

        Returns:
            Series of ``unique_id`` strings aligned with ``df``.

        Examples:
            Given columns ``["entity", "category", "component"]``::

                ("DE", "Total", "")  -> "DE/Total"
                ("DE", "Total", "B") -> "DE/Total/B"
        """
        def _row_to_id(row: pd.Series) -> str:
            # Filtrage des valeurs nulles ou vides avant la concaténation
            parts = []
            # Parcours des colonnes qui ne sont pas des dates
            for col in non_date_cols:
                # Extraction de la valeur
                val = row[col]
                # Construction de l'id unique
                is_nan = isinstance(val, float) and np.isnan(val)
                if val is not None and not is_nan and str(val) != "":
                    parts.append(str(val))
            return "/".join(parts)

        return df.apply(_row_to_id, axis=1)

    # Méthode d'entraînement
    def fit(self, X, y):
        """Fit the hierarchical reconciliation adapter.

        Aggregates ``y`` according to the specified hierarchy and stores the
        internal structures needed for reconciliation. When ``X`` is provided,
        its exogenous columns (keys of ``exog_vars``) are joined onto ``y``
        on the shared index levels and passed to the aggregation function.

        Args:
            X: DataFrame of exogenous variables whose index is a superset of
                (or equal to) ``y``'s index. Only the columns listed as keys
                in ``exog_vars`` are used; the rest are ignored. Pass ``None``
                when no exogenous variables are needed.
            y: Bottom-level time series data as a Series or single-column
                DataFrame with a DatetimeIndex (time series) or a MultiIndex
                whose last level is the date (panel data). Must contain only
                the most granular observations; upper-level aggregates are
                computed automatically by ``aggregate``.

        Returns:
            self: The fitted adapter instance.

        Raises:
            ValueError: If ``y`` has no identifiable datetime index level.
            ValueError: If ``X`` is provided but ``exog_vars`` is ``None``
                or empty.
        """
        # Importation différée pour éviter les erreurs si le package n'est pas installé
        from hierarchicalforecast.core import HierarchicalReconciliation
        from hierarchicalforecast.utils import aggregate, aggregate_temporal

        # Conversion de la Series en DataFrame avec conservation du nom de la colonne cible
        if isinstance(y, pd.Series):
            # Mémorisation du nom de la colonne cible pour une utilisation ultérieure
            self._target_col_ = y.name if y.name is not None else "y"
            # Conversion en DataFrame
            y = y.to_frame(name=self._target_col_)
        else:
            raise TypeError(f"'y' should be a pandas.Series")

        # Conversion de l'index de y en colonnes plates
        df_y, non_date_cols, date_col_name = self._index_to_flat(y)
        self._non_date_cols_ = non_date_cols

        # Jointure des variables exogènes de X si fourni.
        # Si exog_vars est None, X est ignoré silencieusement (compatibilité
        # avec les Pipelines sklearn qui transmettent toujours X à chaque étape).
        if X is not None and self.exog_vars:
            # Conversion de l'index de X en colonnes plates
            df_x, _, _ = self._index_to_flat(
                X if isinstance(X, pd.DataFrame) else X.to_frame()
            )

            # Extraction des colonnes de jointure et des colonnes exogènes disponibles
            join_keys = [_TIME_COL] + non_date_cols
            exog_cols = [c for c in self.exog_vars if c in df_x.columns]
            df_x = df_x[[c for c in join_keys + exog_cols if c in df_x.columns]]

            # Jointure à gauche sur df_y pour conserver toutes les observations d'entraînement
            df_y = df_y.merge(df_x, on=join_keys, how="left")
        elif X is not None:
            warnings.warn(
                "X was provided but 'exog_vars' is None or empty. "
                "Specify the exogenous columns and their aggregation "
                "functions via the 'exog_vars' constructor parameter."
            )

        # Sélection de la fonction d'agrégation selon le type de hiérarchie
        if self._is_temporal_hierarchy:
            # Résolution du nom du niveau temporel (valeur par défaut si non fourni)
            self._effective_id_time_col_ = (
                self.id_time_col if self.id_time_col is not None else "temporal_id"
            )
            # Pré-construction du unique_id requise par aggregate_temporal.
            # Contrairement à aggregate (cross-sectionnel) qui construit lui-même le
            # unique_id à partir du spec, aggregate_temporal s'attend à trouver la
            # colonne unique_id déjà présente dans df avant de trier les séries.
            if non_date_cols:
                # Données de panel : jointure des valeurs des niveaux d'entité
                df_y[_ID_COL] = self._build_unique_id_series(df_y, non_date_cols)
                df_y = df_y.drop(columns=non_date_cols)
            else:
                # Série temporelle simple (DatetimeIndex) : unique_id constant = nom de la cible
                df_y[_ID_COL] = target_col
            self.Y_df_, self.S_df_, self.tags_ = aggregate_temporal(
                df=df_y,
                spec=self.spec,
                exog_vars=self.exog_vars,
                sparse_s=self.sparse_s,
                id_col=_ID_COL,
                time_col=_TIME_COL,
                id_time_col=self._effective_id_time_col_,
                target_cols=(self._target_col_,),
                aggregation_type=self.aggregation_type,
            )
        else:
            self.Y_df_, self.S_df_, self.tags_ = aggregate(
                df=df_y,
                spec=self.spec,
                exog_vars=self.exog_vars,
                sparse_s=self.sparse_s,
                id_col=_ID_COL,
                time_col=_TIME_COL,
                id_time_col=self.id_time_col,
                target_cols=(self._target_col_,),
            )

        # Création de l'objet de réconciliation
        self.hrec_ = HierarchicalReconciliation(reconcilers=self.reconcilers)

        return self

    # Méthode de prédiction
    def predict(self, X):
        """Reconcile base forecasts using the fitted hierarchy.

        Accepts base forecasts either as a MultiIndex DataFrame (preferred)
        or as a flat DataFrame already containing ``"unique_id"`` and ``"ds"``
        columns (advanced / legacy use).

        **MultiIndex format (preferred)**:
            ``X`` must have the same index structure as ``y`` in ``fit``, i.e.
            a MultiIndex whose last level is the date and whose preceding
            levels encode entity and hierarchy information. Predictions for
            *all* levels of the hierarchy must be present, including
            aggregate-level rows. For a row at an intermediate aggregation
            depth, levels that do not apply at that depth should be set to an
            empty string ``""``.

            Example for a two-level cross-sectional hierarchy
            ``[["entity", "category"], ["entity", "category", "component"]]``::

                # Aggregate row (entity/category level):
                index = ("DE", "Total", "", date)  -> unique_id "DE/Total"

                # Bottom-level rows:
                index = ("DE", "Total", "B", date) -> unique_id "DE/Total/B"
                index = ("DE", "Total", "C", date) -> unique_id "DE/Total/C"

            Model prediction columns (e.g., ``"LinearRegression"``) are
            preserved as-is and passed to the reconciler.

        Args:
            X: Base forecasts as a MultiIndex DataFrame with one column per
                model, or a flat DataFrame with ``"unique_id"`` and ``"ds"``
                columns.

        Returns:
            DataFrame containing reconciled forecasts for all hierarchy
            levels, with one column per reconciliation method applied.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        # Vérification préalable de l'état d'entraînement
        if not hasattr(self, "hrec_"):
            raise RuntimeError(
                "This adapter instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'predict'."
            )

        # Détection du format d'entrée : format plat (unique_id + ds) ou MultiIndex
        if (
            isinstance(X, pd.DataFrame)
            and _ID_COL in X.columns
            and _TIME_COL in X.columns
        ):
            # Utilisation directe du format plat (usage avancé)
            Y_hat_df = X
        else:
            # Conversion du MultiIndex en colonnes plates
            df_flat, non_date_cols, _ = self._index_to_flat(X)

            # Construction du unique_id à partir des niveaux non-date
            df_flat[_ID_COL] = self._build_unique_id_series(df_flat, non_date_cols)

            # Suppression des colonnes d'index absorbées dans le unique_id
            df_flat = df_flat.drop(columns=non_date_cols)
            Y_hat_df = df_flat

        # Construction des arguments pour la réconciliation.
        # Les hiérarchies temporelles et cross-sectionnelles ont des interfaces
        # distinctes dans reconcile :
        # - Cross-sectionnel : Y_df requis, id_col='unique_id' dans S_df
        # - Temporel : Y_df=None, temporal=True, id_col='unique_id' (entité),
        #   id_time_col=effective_id_time_col_ (niveau temporel dans S_df et Y_hat)
        if self._is_temporal_hierarchy:
            #from hierarchicalforecast.utils import aggregate_temporal

            # S_df dépend de la longueur de l'horizon de prévision et ne peut
            # pas être réutilisé tel quel depuis l'entraînement. On le reconstruit
            # à partir des lignes bottom-level de Y_hat_df en utilisant une cible
            # fictive (la structure de S ne dépend que des dates et du spec).
            #id_time_col = self._effective_id_time_col_
            #bottom_level = min(self.spec, key=lambda k: self.spec[k])
            #bottom_prefix = bottom_level + "-"
            #Y_hat_bottom = Y_hat_df[
            #    Y_hat_df[id_time_col].str.startswith(bottom_prefix)
            #][[_ID_COL, _TIME_COL]].copy()
            #Y_hat_bottom[self._target_col_] = 1.0  # cible factice : seule la structure de S importe

            #_, S_df_pred, tags_pred = aggregate_temporal(
            #    df=Y_hat_bottom,
            #    spec=self.spec,
            #    id_col=_ID_COL,
            #    time_col=_TIME_COL,
            #    id_time_col=id_time_col,
            #    target_cols=(self._target_col_,),
            #    aggregation_type=self.aggregation_type,
            #)

            #reconcile_kwargs = {
            #    "Y_hat_df": Y_hat_df,
            #    "Y_df": None,
            #    "S_df": S_df_pred,
            #    "tags": tags_pred,
            #    "temporal": True,
            #    "id_col": _ID_COL,
            #    "id_time_col": id_time_col,
            #}
            reconcile_kwargs = {
                "Y_hat_df": Y_hat_df,
                "Y_df": None,
                "S_df": self.S_df_,
                "tags": self.tags_,
                "temporal": True,
            }
        else:
            reconcile_kwargs = {
                "Y_hat_df": Y_hat_df,
                "Y_df": self.Y_df_,
                "S_df": self.S_df_,
                "tags": self.tags_,
            }

        # Ajout des arguments optionnels de prédiction probabiliste
        if self.level is not None:
            reconcile_kwargs["level"] = self.level

        if self.intervals_method is not None:
            reconcile_kwargs["intervals_method"] = self.intervals_method

        if self.num_samples != -1:
            reconcile_kwargs["num_samples"] = self.num_samples

        # Réconciliation et retour des prévisions
        return self.hrec_.reconcile(**reconcile_kwargs)

    # Méthode d'extraction d'informations sur la hiérarchie construite
    def get_hierarchy_info(self) -> Dict:
        """Get information about the fitted hierarchy.

        Returns:
            Dictionary containing:
                - ``"n_series"``: Total number of series across all hierarchy
                  levels.
                - ``"n_bottom"``: Number of bottom-level (most granular)
                  series.
                - ``"levels"``: List of hierarchy level names.
                - ``"tags"``: The tags dictionary mapping level names to their
                  ``unique_id`` arrays.

        Raises:
            RuntimeError: If called before ``fit``.

        Examples:
            Inspecting the fitted hierarchy::

                adapter = HierarchicalForecastAdapter(
                    reconcilers=[BottomUp()],
                    spec=[['Country'], ['Country', 'State']]
                )
                adapter.fit(None, y_train)

                info = adapter.get_hierarchy_info()
                print(f"Total series  : {info['n_series']}")
                print(f"Bottom series : {info['n_bottom']}")
                print(f"Levels        : {info['levels']}")
        """
        # Vérification de l'état d'entraînement
        if not hasattr(self, "tags_"):
            raise RuntimeError(
                "This adapter instance is not fitted yet. "
                "Call 'fit' before accessing hierarchy information."
            )

        # Extraction du nombre total de séries (lignes de S)
        n_total = len(self.S_df_) if hasattr(self, "S_df_") else 0

        # Extraction du nombre de séries bottom-level (colonnes de S, hors colonne d'id)
        if hasattr(self, "S_df_") and self.S_df_ is not None:
            s_cols = [c for c in self.S_df_.columns if c != _ID_COL]
            n_bottom = len(s_cols)
        else:
            n_bottom = 0

        return {
            "n_series": n_total,
            "n_bottom": n_bottom,
            "levels": list(self.tags_.keys()),
            "tags": self.tags_,
        }
