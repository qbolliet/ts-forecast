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
_DEFAULT_ID_TIME_COL = "temporal_id"


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
        spec: Hierarchy specification.

            - **Cross-sectional**: list of lists of index-level names
              ordered from the most aggregated level to the bottom level
              (forwarded to ``aggregate``).
            - **Temporal**: dict mapping temporal level names to the
              number of bottom-level timesteps in the aggregation
              (forwarded to ``aggregate_temporal``).

        exog_vars: Dictionary specifying how to aggregate exogenous variables.
            Keys are column names present in the ``X`` DataFrame passed to
            ``fit``; values are aggregation functions (``"sum"``, ``"mean"``)
            or lists of such functions.
        sparse_s: Whether to return the summing matrix as a sparse
            ``SMatrix``. Forwarded to ``aggregate`` / ``aggregate_temporal``.
        id_time_col: Name of the temporal-id column used by
            ``aggregate_temporal`` and ``reconcile``. Only relevant for
            temporal hierarchies. Defaults to ``"temporal_id"``.
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
        For the cross-sectional case, ``predict`` accepts ``X`` containing
        only bottom-level forecasts (and exogenous variables eventually): aggregate-level rows are computed
        automatically by summation along the spec. To preserve
        level-specific upstream forecasts (e.g., a dedicated model at each
        hierarchy level), pass ``X`` with rows
        already present at every level -- empty strings ``""`` in the
        index encode the levels that do not apply at the aggregate depth.

        For the temporal case, ``predict`` expects a mapping
        ``{level_name: DataFrame}`` so that each temporal level can carry
        its own forecasts. A single DataFrame
        is also accepted: it is then treated as the bottom-level forecasts
        and upper temporal levels are auto-aggregated by summation.
        
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
        is_balanced: bool = False,
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
        self.is_balanced = is_balanced

    # ------------------------------------------------------------------
    # Propriétés et méthodes utilitaires
    # ------------------------------------------------------------------
    
    # Le type de hiérarchie dépend du type de l'argument "spec" : dictionnaire pour une agrégation temporelle, liste sinon
    @property
    def _is_temporal_hierarchy(self) -> bool:
        """Check if spec defines a temporal hierarchy."""
        return isinstance(self.spec, dict)

    # Nom du niveau bottom de la hiérarchie
    @property
    def _bottom_level_name(self) -> str:
        """Return the name of the bottom (most granular) hierarchy level."""
        if self._is_temporal:
            # Le niveau bottom temporel est celui dont le facteur d'agrégation est minimal
            return min(self.spec, key=lambda k: self.spec[k])
        # Niveau bottom cross-section : dernière liste du spec
        return "/".join(self.spec[-1])
    
    # Méthode auxiliaire de détection du niveau du datetime dans l'index
    # /!\ Voir si on ne peut pas hybrider cette méthode avec les méthodes de validation dans utils/validation 
    def _find_datetime_level(self, index: pd.Index) -> Optional[int]:
        """Find the index level that contains datetime values.

        The last level of a MultiIndex is always assumed to be the date level
        when it contains datetime values. For a simple Index, level 0 is
        returned if it is datetime-typed.

        Args:
            index: pandas ``Index`` or ``MultiIndex``.

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

        Transforms a DataFrame whose index encodes entity and time
        information into the flat column format expected by
        ``hierarchicalforecast``. The datetime level is always renamed to
        ``"ds"``; unnamed index levels are assigned placeholder names
        ``"level_i"``.

        Args:
            df: DataFrame with a ``DatetimeIndex`` (time series) or a
                ``MultiIndex`` whose last level is the date.

        Returns:
            Tuple of ``(df_flat, non_date_cols)``:

            - ``df_flat``: DataFrame with the index reset to columns and
              the date column renamed to ``"ds"``.
            - ``non_date_cols``: Ordered list of non-date index level
              names.

        Raises:
            ValueError: If no datetime level can be identified.
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

        return df, non_date_cols

    # Méthode de construction du "unique_id" à partir des colonnes non-date
    def _build_unique_id_series(
        self, df: pd.DataFrame, non_date_cols: List[str]
    ) -> pd.Series:
        """Build the ``unique_id`` column from non-date column values.

        For each row, joins the non-empty values of the specified columns
        with ``"/"``. Empty strings ``""`` are silently omitted so that
        aggregate-level rows naturally produce shorter identifiers.

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
        # Récupération des colonnes en tant que tableau de chaînes de caractères
        arr = df[non_date_cols].astype(str).to_numpy()

        # Définition des marqueurs de valeur "absente" (chaîne vide ou NaN converti)
        nan_markers = {"nan", "None", "<NA>", ""}

        # Construction ligne-à-ligne avec filtrage des valeurs absentes
        def _join_row(row):
            return "/".join(v for v in row if v not in nan_markers)

        return pd.Series([_join_row(row) for row in arr], index=df.index)

    # ------------------------------------------------------------------
    # Préparation des prédictions in-sample / out-of-sample
    # ------------------------------------------------------------------

    # Conversion d'une entrée X (Series, DataFrame ou dict) en DataFrame plat
    # Retourne (df_plat, non_date_cols, model_cols, df_temporal_levels)
    def _prepare_forecasts_flat(
        self,
        X: Union[pd.Series, pd.DataFrame, Dict[str, Union[pd.Series, pd.DataFrame]]],
        non_date_cols_reference: List[str],
    ) -> Tuple[pd.DataFrame, List[str], Optional[Dict[str, pd.DataFrame]]]:
        """Convert ``X`` into the flat format expected by hierarchicalforecast.

        Handles three input shapes uniformly:

        - ``Series`` (a single model) → DataFrame with one model column.
        - ``DataFrame`` (one or several models) → kept as-is.
        - ``dict`` mapping temporal level name to ``Series``/``DataFrame``
          (temporal hierarchy only) → each entry is treated as the
          forecasts at the given temporal level.

        Args:
            X: User-facing base-forecast input.
            non_date_cols_reference: Non-date index level names observed
                during ``fit``. Used to detect index-level mismatches.

        Returns:
            Tuple ``(df_flat, model_cols, by_level)``:

            - ``df_flat``: Flat DataFrame with ``unique_id``, ``ds`` and
              one column per model. ``temporal_id`` is included only when
              ``by_level`` is non-empty.
            - ``model_cols``: List of model column names.
            - ``by_level``: For temporal inputs given as a dict, mapping
              ``level_name -> flat sub-DataFrame``. ``None`` otherwise.
        """
        # Cas spécifique du dictionnaire (hiérarchie temporelle multi-niveaux)
        if isinstance(X, dict):
            if not self._is_temporal:
                raise TypeError(
                    "Dict input is only supported for temporal hierarchies."
                )
            # Vérification de la cohérence des clés avec le spec
            unknown = set(X.keys()) - set(self.spec.keys())
            if unknown:
                raise ValueError(
                    f"Unknown temporal levels in X: {sorted(unknown)}. "
                    f"Expected levels from spec: {sorted(self.spec.keys())}."
                )
            # Conversion récursive de chaque niveau
            by_level = {}
            for level_name, df_level in X.items():
                df_flat_level, _, _ = self._prepare_forecasts_flat(
                    df_level, non_date_cols_reference
                )
                by_level[level_name] = df_flat_level
            # Extraction des colonnes de modèle communes
            model_cols = [
                c
                for c in next(iter(by_level.values())).columns
                if c not in (_ID_COL, _TIME_COL)
            ]
            # Concaténation en un seul DataFrame
            df_flat = pd.concat(by_level.values(), ignore_index=True)
            return df_flat, model_cols, by_level

        # Conversion Series → DataFrame
        if isinstance(X, pd.Series):
            name = X.name if X.name is not None else "y_hat"
            X = X.to_frame(name=name)

        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X must be a Series, DataFrame, or dict (temporal only); "
                f"got {type(X).__name__}."
            )

        # Conversion de l'index en colonnes plates
        df_flat, non_date_cols = self._index_to_flat(X)

        # Construction du unique_id à partir des colonnes non-date (hors série temporelle simple)
        if non_date_cols:
            df_flat[_ID_COL] = self._build_unique_id_series(df_flat, non_date_cols)
            df_flat = df_flat.drop(columns=non_date_cols)
        else:
            # Cas dégénéré d'une série temporelle simple
            df_flat[_ID_COL] = self._target_col_

        # Identification des colonnes de modèles (toutes sauf identifiants)
        # /!\ Voir si'il ne faut pas également exclure les variables exogènes
        model_cols = [c for c in df_flat.columns if c not in (_ID_COL, _TIME_COL)]

        return df_flat, model_cols, None

    # Calcul automatique des prédictions aux niveaux supérieurs par sommation
    def _autoaggregate_cross_sectional(
        self, df_flat: pd.DataFrame, model_cols: List[str]
    ) -> pd.DataFrame:
        """Auto-aggregate bottom-level forecasts to all hierarchy levels.

        For each upper level in the spec, the predictions are summed over
        the bottom-level rows that belong to the corresponding aggregate.
        Existing rows for an upper level (matched by ``unique_id``) are
        preserved unchanged.

        Args:
            df_flat: Flat DataFrame with ``unique_id``, ``ds`` and
                ``model_cols``.
            model_cols: Columns to aggregate.

        Returns:
            DataFrame containing rows at every level of the cross-sectional
            spec.
        """
        # Récupération des identifiants existants pour ne pas les écraser
        existing_ids = set(df_flat[_ID_COL].unique())
        all_levels_dfs = [df_flat]

        # Itération sur tous les niveaux du spec à l'exception du niveau bottom
        for level_cols in self.spec[:-1]:
            # Identifiants attendus à ce niveau, déduits de tags_
            level_name = "/".join(level_cols)
            target_ids = set(self.tags_[level_name])
            missing_ids = target_ids - existing_ids
            if not missing_ids:
                continue

            # Mapping unique_id bottom → unique_id agrégé via le préfixe du chemin
            depth = len(level_cols)
            bottom_ids = self.tags_[self._bottom_level_name]
            agg_id_for_bottom = {
                bid: "/".join(bid.split("/")[:depth]) for bid in bottom_ids
            }

            # Sommation des prédictions au niveau agrégé
            df_bottom = df_flat[df_flat[_ID_COL].isin(bottom_ids)].copy()
            df_bottom[_ID_COL] = df_bottom[_ID_COL].map(agg_id_for_bottom)
            df_agg = (
                df_bottom.groupby([_ID_COL, _TIME_COL], as_index=False)[model_cols]
                .sum()
            )
            # Filtrage des identifiants déjà présents
            df_agg = df_agg[df_agg[_ID_COL].isin(missing_ids)]
            all_levels_dfs.append(df_agg)
            existing_ids = existing_ids.union(missing_ids)

        # Concaténation des DataFrames de niveau agrégé et de niveau bottom
        return pd.concat(all_levels_dfs, ignore_index=True)

    # Construction de la matrice S et des tags pour la hiérarchie temporelle out-of-sample
    def _build_temporal_s_for_prediction(
        self,
        df_flat_bottom: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """Rebuild ``S``, ``tags`` and Y_hat at all temporal levels.

        For temporal hierarchies, the summing matrix depends on the
        prediction horizon, so it must be re-derived from the
        out-of-sample bottom-level data on each call.

        Args:
            df_flat_bottom: Flat DataFrame of bottom-level out-of-sample
                forecasts with ``unique_id``, ``ds`` and one column per
                model.

        Returns:
            Tuple ``(S_pred, tags_pred, Y_hat_all_levels)`` where
            ``Y_hat_all_levels`` contains the bottom-level forecasts
            extended with auto-aggregated rows at every upper temporal
            level.
        """
        # Importation différée pour éviter une dépendance dure
        from hierarchicalforecast.utils import aggregate_temporal

        # Détection des colonnes de modèles
        model_cols = [
            c for c in df_flat_bottom.columns if c not in (_ID_COL, _TIME_COL)
        ]

        # Agrégation temporelle des prédictions bottom-level vers tous les niveaux
        # /!\ aggregate_temporal somme les target_cols selon le spec, ce qui produit
        #     les prédictions agrégées attendues pour les niveaux supérieurs.
        Y_hat_all_levels, S_pred, tags_pred = aggregate_temporal(
            df=df_flat_bottom,
            spec=self.spec,
            sparse_s=self.sparse_s,
            id_col=_ID_COL,
            time_col=_TIME_COL,
            id_time_col=self._effective_id_time_col_,
            target_cols=tuple(model_cols),
            aggregation_type=self.aggregation_type,
        )

        return S_pred, tags_pred, Y_hat_all_levels

    # Construction du Y_hat_df temporel à partir d'un dictionnaire par niveau
    def _build_temporal_y_hat_from_dict(
        self,
        by_level: Dict[str, pd.DataFrame],
        S_pred: pd.DataFrame,
        tags_pred: Dict,
    ) -> pd.DataFrame:
        """Combine per-level forecasts into a single Y_hat_df with temporal_id.

        Each entry of ``by_level`` provides forecasts at one temporal
        aggregation. The ``temporal_id`` of each row is recovered by
        sorting the dates within each ``unique_id`` and assigning the
        position-based identifier expected by hierarchicalforecast
        (``"<level>-<k>"``).

        Args:
            by_level: Mapping ``temporal_level_name -> flat DataFrame``
                with ``unique_id``, ``ds`` and model columns.
            S_pred: Summing matrix produced by aggregate_temporal on the
                bottom-level forecasts (used to discover the temporal_id
                of each row).
            tags_pred: Tags produced by aggregate_temporal.

        Returns:
            Single DataFrame with columns ``unique_id``,
            ``temporal_id``, ``ds`` and the model columns, ready to be
            passed as ``Y_hat_df`` to ``reconcile``.
        """
        # Récupération du nom de la colonne d'identifiant temporel
        id_time = self._effective_id_time_col_

        # Reconstruction du mapping (unique_id, ds) → temporal_id par niveau
        # /!\ aggregate_temporal trie chronologiquement à l'intérieur de chaque entité
        per_level_dfs = []
        for level_name, df_lvl in by_level.items():
            ids_for_level = tags_pred.get(level_name)
            if ids_for_level is None:
                raise ValueError(
                    f"Level '{level_name}' not present in tags built from "
                    f"the bottom-level out-of-sample data."
                )

            # Tri chronologique par entité pour aligner les temporal_id
            df_sorted = df_lvl.sort_values([_ID_COL, _TIME_COL]).reset_index(drop=True)
            # Attribution du temporal_id par position dans chaque entité
            df_sorted[id_time] = df_sorted.groupby(_ID_COL).cumcount().map(
                lambda k: f"{level_name}-{k + 1}"
            )
            per_level_dfs.append(df_sorted)

        # Concaténation et réordonnancement des colonnes
        Y_hat_df = pd.concat(per_level_dfs, ignore_index=True)
        id_cols = [_ID_COL, id_time, _TIME_COL]
        model_cols = [c for c in Y_hat_df.columns if c not in id_cols]
        return Y_hat_df[id_cols + model_cols]

    # ------------------------------------------------------------------
    # API sklearn : fit / predict / score
    # ------------------------------------------------------------------
    
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
