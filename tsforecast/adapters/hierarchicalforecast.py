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
_TARGET_COL = "y"
_ID_COL = "unique_id"
_TIME_COL = "ds"
_ID_TIME_COL = "temporal_id"


# Wrapper permettant l'intégration des modèles du package "hierarchicalforecast" dans une syntaxe "sklearn-like"
# /!\ L'héritage de RegressorMixin fournit une méthode score() par défaut basée sur R²,
#     que l'on surcharge pour la rendre cohérente avec la sortie multi-colonnes de reconcile
class HierarchicalForecastAdapter(BaseEstimator, RegressorMixin):
    """sklearn-compatible adapter for hierarchicalforecast reconciliation.

    The ``reconcile`` method of ``hierarchicalforecast`` is conceptually a
    ``fit_predict``: it both fits the reconciler internally and produces
    reconciled forecasts in a single call. To respect the sklearn split
    between training-time state and prediction-time computation, the
    adapter splits the workflow as follows:

    - ``fit(X, y)`` calls :func:`~hierarchicalforecast.utils.aggregate` (or
      :func:`~hierarchicalforecast.utils.aggregate_temporal`) on ``y`` to
      derive the summing matrix ``S``, the level ``tags`` and the
      training-time reference frame ``Y_df``. These quantities depend only
      on the hierarchy structure and on the insample values, not on the
      out-of-sample forecasts that will be reconciled. When ``X`` is
      provided, its insample base-forecast columns are joined into ``Y_df``
      so that reconcilers requiring ``y_hat_insample``
      (``MinTrace('mint_shrink' | 'mint_cov' | 'wls_var')``, ``ERM``) can
      access them.
    - ``predict(X)`` reconciles out-of-sample base forecasts. The summing
      matrix is reused from ``fit`` for cross-sectional hierarchies; for
      temporal hierarchies, it depends on the prediction horizon and is
      rebuilt from ``X`` itself before calling ``reconcile``.

    Both ``X`` and ``y`` use pandas indices to encode entity and time
    information; no manual construction of ``id_col``, ``time_col``, 
    and ``target_cols``, ``unique_id`` / ``ds`` strings is required.

    Args:
        reconcilers: List of instantiated reconciliation method objects
            from ``hierarchicalforecast.methods`` (e.g., ``BottomUp()``,
            ``MinTrace(method='ols')``).
        hierarchy: Hierarchy specification. Dictionnary mapping each 
            aggregated level to its bottom level
        temporal_hierarchy: Dict mapping temporal level names to the
            number of bottom-level timesteps in the aggregation
            (forwarded to ``aggregate_temporal``). When specified, the
            observations are no longer reconciliated in cross-section. If
            you want to reconciale both in cross-section and across time
            step you should perform a cross-section reconcialiation first, 
            then a temporal reconcialiation as described in 
            https://nixtlaverse.nixtla.io/hierarchicalforecast/examples/australiandomestictourismcrosstemporal.html#3-temporal-reconciliation
        exog_vars: Dictionary specifying how to aggregate exogenous variables.
            Keys are column names present in the ``X`` DataFrame passed to
            ``fit``; values are aggregation functions (``"sum"``, ``"mean"``)
            or lists of such functions.
        aggregation_type: Temporal aggregation strategy, ``"local"`` or
            ``"global"``. Only relevant for temporal hierarchies.
        level: Confidence levels for probabilistic reconciliation, e.g.
            ``[80, 95]``.
        intervals_method: Sampler for prediction intervals, one of
            ``"normality"``, ``"bootstrap"``, ``"permbu"``.
        num_samples: Number of probabilistic coherent samples returned by
            ``reconcile``. Defaults to ``-1`` (no sampling).
        is_balanced: Whether the training set is balanced. Forwarded to
            ``reconcile``; setting it to ``True`` speeds reconciliation up
            when applicable.

    Attributes:
        Y_df_: Aggregated training DataFrame, optionally enriched with
            insample base forecasts joined from ``X``.
        S_df_: Summing matrix derived during ``fit``. Reused at prediction
            time for cross-sectional hierarchies; for temporal
            hierarchies, this is the *training-set* version and is
            **not** reused during ``predict``.
        tags_: Mapping from level name to the array of identifiers
            (``unique_id`` for cross-sectional, ``temporal_id`` for
            temporal) belonging to that level.
        hrec_: Underlying ``HierarchicalReconciliation`` instance.

    Note:
        The exogenous variables and the forecasts to reconcile are assumed to be in the X argument of the fit and predict methods
        
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
        hierarchy: Dict[str, List[str]],
        temporal_hierarchy: Dict[str, int],
        exog_vars: Optional[Dict[str, Union[str, List[str]]]] = None,
        aggregation_type: str = "local",
        level: Optional[List[int]] = None,
        intervals_method: str = "normality",
        num_samples: int = -1,
        seed: int = 0,
        is_balanced: bool = False,
        diagnostics: bool = False,
        diagnostics_atol: float = 1e-6,
    ):
        # Initialisation des attributs à partir des arguments
        self.reconcilers = reconcilers
        self.hierarchy = hierarchy
        self.temporal_hierarchy = temporal_hierarchy
        self.exog_vars = exog_vars
        self.aggregation_type = aggregation_type
        self.level = level
        self.intervals_method = intervals_method
        self.num_samples = num_samples
        self.seed = seed
        self.is_balanced = is_balanced
        self.diagnostics = diagnostics
        self.diagnostics_atol = diagnostics_atol

    # ------------------------------------------------------------------
    # Méthodes utilitaires
    # ------------------------------------------------------------------
    
    # Méthode auxiliaire de construction des branches parent-enfant
    def _build_parent_map(self, hierarchy: Dict[str, List[str]]) -> Dict[str, str]:
        """Build a child → parent mapping from the hierarchy dict.
     
        Args:
            hierarchy: Mapping from parent component to its direct children.
     
        Returns:
            Dict mapping each non-root node to its direct parent.
        """
        return {
            child: parent
            for parent, children in hierarchy.items()
            for child in children
        }
     
    # Méthode auxiliaire de description du chemin entre un noeud et la racine
    def _path_from_root(self, node: str, parent_map: Dict[str, str]) -> List[str]:
        """Return the path from root to node, excluding the root, top-down.
     
        Args:
            node: Target node name.
            parent_map: Dict mapping each node to its direct parent.
     
        Returns:
            Ordered list of node names from the first child of root down to
            ``node`` (inclusive).  Empty list if ``node`` is the root itself.
        """
        # Remontée vers la racine puis inversion
        path: list[str] = []
        current = node
        while current in parent_map:
            path.append(current)
            current = parent_map[current]
        # Ajout de la racine (dernier nœud sans parent)
        path.append(current)
        return list(reversed(path))
    
    # Méthode auxiliaire de construction du DataFrame implémentant la hiérarchie
    def _build_df_hierarchy(self, hierarchy: Dict[str, List[str]]) -> pd.DataFrame:
        # Identification des branches
        parent_map = self._build_parent_map(hierarchy=hierarchy)
        # Identification de l'ensemble des noeuds
        all_nodes = set(np.concatenate(list(hierarchy.values())).tolist()) | set(hierarchy.keys())
        
        # Chemins feuille → liste de noeuds depuis la racine (racine exclue)
        node_paths: dict[str, list[str]] = {
            node: self._path_from_root(node, parent_map) for node in all_nodes
        }
        # Calcul de la profondeur maximale
        max_depth = max(len(p) for p in node_paths.values())
        
        # Noms de colonnes pour chaque niveau
        level_cols = [f"_hierarchical_level{i}" for i in range(max_depth)]
        
        # Table de correspondance feuille → chemin padé (répétition du dernier élément pour avoir des éléments de profondeur homogène)
        node_to_padded: dict[str, list[str]] = {
            node: path + [""] * (max_depth - len(path))
            for node, path in node_paths.items()
        }
        
        # Construction du DataFrame associant à chaque colonne son nom dans la hiérarchie
        df_hierarchy = (
            pd.DataFrame.from_dict(node_to_padded, orient="index", columns=level_cols)
            .rename_axis("node")
            .reset_index()
        )
    
        return df_hierarchy
    
    # Méthode auxiliaire des noms de colonnes associés aux entités et aux dates
    def _extract_entity_date_names(self, index: pd.Index) -> Tuple[List[str], str]:
        # Distinction du cas d'un MultiIndex de celui d'un index simple
        if isinstance(index, pd.MultiIndex):
            # Extraction des noms
            names = index.names
            # Remplacement des valeurs manquantes par f"level_{i}"
            filled_names = [name if name is not None else f"level_{i}" for i, name in enumerate(names)]
            # Extraction des noms correspondant à la date et aux entités
            entity_cols, date_col = filled_names[:-1], filled_names[-1]
        else :
            date_col = index.name if index.name is not None else "index"
            entity_cols = []
        return entity_cols, date_col
    
    # Méthode auxiliaire de construction du jeu de données et des specs associées
    def _build_df_formatted(self, df : Union[pd.DataFrame, pd.Series], df_hierarchy : pd.DataFrame, hierarchy_cols: List[str], exog_var_names: Set[str], value_vars: List[str], value_name: str) -> Tuple[pd.DataFrame, List[List[str]]]:
        # Extraction des noms des entités et de la date
        entity_cols, date_col = _extract_entity_date_names(index=df.index)
        # Réinitialisation de l'index
        df_flat = df.reset_index()
        # Extraction des colonnes exogènes
        exog_cols = list(exog_var_names.intersection(set(df.columns))) if isinstance(df, pd.DataFrame) else []
        
        # Réorganimsation du jeu de données
        df_formatted = pd.melt(
            frame=df_flat,
            id_vars=entity_cols + [date_col] + exog_cols,
            value_vars=value_vars,
            var_name="node",
            value_name=value_name,
        )
        # Appariement avec les données de hierarchie
        if not df_hierarchy.empty :
            df_formatted = pd.merge(
                left=df_formatted,
                right=df_hierarchy,
                how="left",
                on="node",
                validate="many_to_one"
            ).drop("node", axis=1) # Suppression de la colonne d'appariement
        # Renomination de la colonne de dates
        df_formatted.rename({date_col : _TIME_COL}, axis=1, inplace=True)
    
        # Construction des specs attendues par HierarchicalForecast
        spec: list[list[str]] = [
            entity_cols + hierarchy_cols[: i + 1] for i in range(len(hierarchy_cols))
        ]
    
        return df_formatted, spec, entity_cols

    # Méthode auxiliaire d'ajout de la fonction de création de la colonne d'identifiants
    def _build_unique_id_column(self, df : pd.DataFrame, entity_cols : List[str], hierarchy_cols : List[str]) -> pd.DataFrame:
        # Création du jeu de données avec les différents niveaux (y compris intermédiaires)
        df[_ID_COL] = df[entity_cols + hierarchy_cols].astype(str).agg(list, axis=1).str.join("/").str.rstrip("/")
        return df.drop(entity_cols + hierarchy_cols, axis=1)

    # ------------------------------------------------------------------
    # API sklearn : fit / predict / score
    # ------------------------------------------------------------------
    
    # Méthode d'entraînement
    def fit(self, X, y):
        # Importation différée pour éviter les erreurs si le package n'est pas installé
        from hierarchicalforecast.core import HierarchicalReconciliation
        from hierarchicalforecast.utils import aggregate, aggregate_temporal

        # Reformattage des données d'entrée et création des specs
        # Traitement de la hiérarchie
        if hierarchy:
            # Construction du jeu de données associé à la hiérarchie
            self.df_hierarchy = _build_df_hierarchy(hierarchy=self.hierarchy)
            # Extraction de l'ensemble des noeuds de la hiérarchie qui correspondent aux variables à agréger
            self.value_vars = list(set(np.concatenate(list(self.hierarchy.values())).tolist()) | set(self.hierarchy.keys()))
            # Extraction des colonnes associées à la hiérarchie
            self.hierarchy_cols = sorted(list(set(df_hierarchy.columns.tolist()) - {'node'}))
        else:
            # Création d'un jeu de données vide
            self.df_hierarchy = pd.DataFrame()
            # L'ensemble des colonnes de y correspond aux variables d'intérêt
            self.value_vars = y.columns.tolist() if isinstance(y, pd.DataFrame) else [y.name]
            # Colonnes associées à la hiérarchie
            self.hierarchy_cols = []
            
        # Extraction des colonnes correspondant aux variables exogènes
        self.exog_var_names = set(self.exog_vars.keys()) if self.exog_vars is not None else set()
        
        # Reformattage du jeu de données
        # Reformattage des données X
        # Renommage y -> y_hat parmi les co-variables. 
        # Correspondra à y_hat_insample comme argument de la méthode "fit" des classes de réconciliation
        X_formatted, _, _ = self._build_df_formatted(df=X, df_hierarchy=self.df_hierarchy, hierarchy_cols=self.hierarchy_cols, exog_var_names=self.exog_var_names, value_vars=self.value_vars, value_name=f'{_TARGET_COL}_hat')
        # Formattage des données y
        y_formatted, spec, self.entity_cols = self._build_df_formatted(df=y, df_hierarchy=self.df_hierarchy, hierarchy_cols=self.hierarchy_cols, exog_var_names=set(), value_vars=self.value_vars, value_name=_TARGET_COL)
        
        # Appariement des deux jeux de données
        df = pd.merge(
            left=X_formatted,
            right=y_formatted,
            on=self.entity_cols + self.hierarchy_cols + [_TIME_COL],
            how='outer',
            validate='one_to_one'
        )
        
        # Distinction du cas de l'agrégation temporelle de celui de l'agrégation en cross-section
        if temporal_hierarchy :
            # Création du booléen indiquant si l'on réalise ou non une aggrégation temporelle
            self.temporal=True
        else:
            # Utilisation de la fonction "aggregate" pour créer les utilitaires nécessaires
            _, self.S_df_, self.tags_ = aggregate(
                df=df.loc[df[self.hierarchy_cols[-1]] != ""] if self.hierarchy_cols else df,
                spec=spec,
                exog_vars=self.exog_vars,
                sparse_s=True,
                id_col=_ID_COL,
                time_col=_TIME_COL,
                id_time_col=None,
                target_cols=(_TARGET_COL, f'{_TARGET_COL}_hat')
            )
            # Création du booléen indiquant si l'on réalise ou non une aggrégation temporelle
            self.temporal=False
        
        # Création du jeu de données avec les différents niveaux (y compris intermédiaires)
        self.Y_df_ = _build_unique_id_column(df=df, entity_cols=self.entity_cols, hierarchy_cols=self.hierarchy_cols)

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
                "Call 'fit' before using 'predict'."
            )

        # Reformattage des données X
        # Renommage y -> y_hat parmi les co-variables. 
        # Correspondra à y_hat_insample comme argument de la méthode "fit" des classes de réconciliation
        Y_hat_df, _, _ = self._build_df_formatted(df=X, df_hierarchy=self.df_hierarchy, hierarchy_cols=self.hierarchy_cols, exog_var_names=self.exog_var_names, value_vars=self.value_vars, value_name=f'{_TARGET_COL}_hat')
        # Création du jeu de données avec les différents niveaux (y compris intermédiaires)
        Y_hat_df = _build_unique_id_column(df=Y_hat_df, entity_cols=self.entity_cols, hierarchy_cols=self.hierarchy_cols)
        
        # Dans le cas d'une aggrégation temporelle, il faut reconstruire les matrices S et le dictionnaire de tags 
        if self.temporal:
            # Utilisation de "aggregate_temporal" pour créer les utilitaires nécessaires à partir des bottom-level
            Y_hat_df, self.S_df_, self.tags_ = aggregate_temporal(
                df=Y_hat_df,
                spec=self.temporal_hierarchy,
                exog_vars=self.exog_vars,
                sparse_s=True,
                id_col=_ID_COL,
                time_col=_TIME_COL,
                id_time_col=_ID_TIME_COL,
                target_cols=(_TARGET_COL, f'{_TARGET_COL}_hat'),
                aggregation_type=self.aggregation_type
            )
        
        # Initialisation des kwargs de réconciliation
        reconcile_kwargs = {
            "Y_hat_df": Y_hat_df,
            "tags": self.tags_,
            "S_df": self.S_df_,
            "Y_df": self.Y_df_,
            "level": self.level,
            "intervals_method": self.intervals_method,
            "num_samples" : self.num_samples,
            "seed": self.seed,
            "is_balanced": self.is_balanced,
            "id_col" = _ID_COL,
            "time_col" = _TIME_COL, 
            "target_col" = _TARGET_COL,
            "id_time_col" : _ID_TIME_COL,
            "temporal" : self.temporal,
            "diagnostics" : self.diagnostics,
            "diagnostics_atol" : self.diagnostics_atol
        }

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
