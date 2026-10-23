"""Panelwise transformer for applying transformations independently per entity.

This module provides a transformer that wraps any sklearn-compatible transformer
and applies it independently to each entity in panel data. Supports entity-specific
parameterization via callable factories or entity_kwargs.
"""
# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
from typing import Dict, Iterator, Optional, Union, List, Any, Callable, Tuple
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
from .utils import (
    build_panel_index,
    check_entity_labels,
    is_panel_data,
    iter_entity_blocks,
    normalize_entity_key,
    normalize_entity_mapping,
)

# Stratégies admises de gestion des erreurs par entité
_ERROR_HANDLING_OPTIONS = ('raise', 'warn', 'ignore')

# Fonction auxiliaire d'itération sur les groupes d'entités dans l'ordre des données
def _groups_in_order_of_appearance(entity_groups) -> List[Tuple[Any, np.ndarray]]:
    """List the entity groups in the order of their first row in the data.

    ``GroupBy.indices`` is sorted by key: iterating over it would return the
    entities sorted, not in the order of the input. Used for entities held in
    columns, where the blocks of ``iter_entity_blocks`` do not apply. Rows are
    given by position, so that repeated row labels cannot mix two rows.

    Args:
        entity_groups: Pandas ``GroupBy`` object over the entity keys.

    Returns:
        List of ``(entity_key, row_positions)`` pairs, in order of first
        appearance, positions increasing within each entity.

    Examples:
        >>> df = pd.DataFrame({'country': ['FR', 'DE', 'FR'], 'v': [1, 2, 3]})
        >>> [(key, rows.tolist()) for key, rows in _groups_in_order_of_appearance(df.groupby('country'))]
        [('FR', [0, 2]), ('DE', [1])]
    """
    # Positions des lignes de chaque groupe
    indices = entity_groups.indices
    # Classement des groupes selon leur première ligne
    return [(key, indices[key]) for key in sorted(indices, key=lambda key: indices[key].min())]


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
        time_col: Name of the time column. Defaults to 'date'. When it is not a
            column of a ``MultiIndex`` panel, the dates are read on the last level.
        panel_cols: Columns identifying the entities of a panel held in columns.
            Not needed for a ``MultiIndex`` panel (entities in the first levels).
        entity_kwargs: Dict mapping entity keys to kwargs dicts. When provided
            (and transformer is not callable), these kwargs are passed to
            set_params() on the cloned transformer for each entity before fitting.
            A scalar key designates a one-level entity (``'FR'`` is ``('FR',)``).
            Ignored if transformer is a callable factory.
        default_entity_kwargs: Default kwargs applied to entities not found in
            entity_kwargs. Only used when entity_kwargs is provided.
        validate_input: Whether to validate input data. Defaults to True.
        strict_validation: If True, raises errors on validation failure.
        auto_sort: If True, automatically sorts unsorted data. Defaults to False.
        convert_cols_to_index: If True, converts time_col and panel_cols to index.
        n_jobs: Number of parallel jobs for fitting (transforming is sequential).
            -1 means using all processors. Defaults to 1 (no parallelism).
        error_handling: How to handle an entity unseen at fit or an entity
            transformer that fails: 'raise' (ValueError for an unseen entity, the
            original error otherwise), 'warn' or 'ignore' (rows left untransformed).
            Defaults to 'raise'.

    Attributes:
        transformers_: Dict mapping entity keys to fitted transformers.
        transform_columns_: List of columns to transform for each entity.
        entities_: Number of unique entities found during fit.
        failed_entities_: Entities whose transformer failed to fit (failures at
            transform time are reported by warnings only).
        is_factory_mode_: Whether transformer is a callable factory.

    Examples:
        Simple mode - same transformer for all entities, panel in the index:

        >>> import pandas as pd
        >>> from sklearn.preprocessing import StandardScaler
        >>> dates = pd.date_range('2024-01-31', periods=3, freq='ME', name='date')
        >>> index = pd.MultiIndex.from_product([['FR', 'DE'], dates], names=['country', 'date'])
        >>> panel = pd.DataFrame({'gdp': [1.0, 2.0, 6.0, 10.0, 20.0, 60.0]}, index=index)
        >>> centring = PanelwiseTransformer(StandardScaler(with_std=False))
        >>> centring.fit_transform(panel)['gdp'].tolist()
        [-2.0, -1.0, 3.0, -20.0, -10.0, 30.0]

        Panel held in columns: ``panel_cols`` names the entity columns, which
        are kept in the output:

        >>> columns = panel.reset_index()
        >>> out = PanelwiseTransformer(StandardScaler(with_std=False), panel_cols=['country']).fit_transform(columns)
        >>> list(out.columns), out['gdp'].tolist()
        (['country', 'date', 'gdp'], [-2.0, -1.0, 3.0, -20.0, -10.0, 30.0])

        Factory mode - callable creates entity-specific transformers:

        >>> def scaler_factory(entity_key):
        ...     # Centrage de FR seulement
        ...     return StandardScaler(with_mean=entity_key == ('FR',), with_std=False)
        >>> PanelwiseTransformer(scaler_factory).fit_transform(panel)['gdp'].tolist()
        [-2.0, -1.0, 3.0, 10.0, 20.0, 60.0]

        Entity kwargs mode - same base transformer with per-entity params:

        >>> per_entity = PanelwiseTransformer(
        ...     StandardScaler(with_std=False),
        ...     entity_kwargs={'DE': {'with_mean': False}},
        ... )
        >>> per_entity.fit_transform(panel)['gdp'].tolist()
        [-2.0, -1.0, 3.0, 10.0, 20.0, 60.0]

    Notes:
        - Factory mode takes precedence over entity_kwargs mode
        - Each entity receives its own transformer instance
        - The transformer preserves the original DataFrame structure
        - Supports inverse_transform if the base transformer does
        - Memory usage scales with the number of entities
        - Empty data gives an empty output: an empty copy of its input, since
          no entity transformer can be called without rows
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
            entity_kwargs: Dict mapping entity keys (tuples, or scalars for
                one-level entities) to kwargs dicts. Applied via set_params()
                when transformer is not a callable.
            default_entity_kwargs: Default kwargs for entities not in entity_kwargs.
            validate_input: Whether to validate input data. Defaults to True.
            strict_validation: If True, raises errors; otherwise emits warnings.
            auto_sort: If True, automatically sorts the data. Defaults to False.
            convert_cols_to_index: If True, converts time_col and panel_cols to index.
            n_jobs: Number of parallel fitting jobs (-1 = all processors). Defaults to 1.
            error_handling: Error handling strategy ('raise', 'warn', 'ignore'),
                checked at fit time.
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
            ValueError: If factory returns None or invalid transformer, or if the
                entity_kwargs of the entity hold an invalid parameter name.
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
            # Recherche des kwargs pour cette entité, sur les clés normalisées au fit
            kwargs = self._entity_kwargs.get(entity_key)

            # Fallback sur les kwargs par défaut si l'entité n'est pas trouvée
            if kwargs is None and self.default_entity_kwargs is not None:
                kwargs = self.default_entity_kwargs

            # Application des kwargs si disponibles ; sklearn refuse un nom inconnu
            if kwargs is not None:
                try:
                    transformer.set_params(**kwargs)
                except ValueError as e:
                    raise ValueError(
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

    # Méthode auxiliaire de découpage des données en blocs d'entités
    def _entity_blocks(self, X: pd.DataFrame) -> Iterator[Tuple[tuple, np.ndarray, pd.DataFrame]]:
        """Iterate over the entities of the data, in order of first appearance.

        Panel data (entities in the ``MultiIndex``) is cut by
        :func:`tsforecast.panel.utils.iter_entity_blocks`. Entities held in
        columns are grouped by ``panel_cols``; each block is indexed by the
        values of ``time_col`` (by the row labels when ``time_col`` is not a
        column), so that the entity transformer sees dated rows in both layouts.

        Args:
            X: Input data.

        Yields:
            Tuple ``(entity_key, mask, block)``: the normalized entity key, the
            boolean mask of its rows in ``X`` and its rows indexed by date only.

        Raises:
            ValueError: If an entity label is missing.
        """
        # Entités portées par l'index : blocs datés fournis par l'utilitaire de panel
        if is_panel_data(X):
            yield from iter_entity_blocks(X, is_panel=True)
            return

        # Entités portées par des colonnes : une étiquette manquante écarterait ses
        # lignes de tout groupe, comme pour les entités portées par l'index
        check_entity_labels(X, columns=self.panel_cols)
        # Groupes dans l'ordre d'apparition, blocs indexés par leurs dates
        has_time_col = self.time_col in X.columns
        for entity_key, positions in _groups_in_order_of_appearance(X.groupby(self.panel_cols, sort=False)):
            mask = np.zeros(len(X), dtype=bool)
            mask[positions] = True
            block = X.iloc[positions]
            if has_time_col:
                block = block.set_axis(pd.Index(block[self.time_col], name=self.time_col))
            yield normalize_entity_key(entity_key), mask, block

    # Méthode auxiliaire d'extraction des colonnes à transformer d'un bloc d'entité
    def _entity_data(self, block: pd.DataFrame) -> pd.DataFrame:
        """Select the columns to transform in the block of an entity.

        Args:
            block: Rows of the entity indexed by date.

        Returns:
            The block restricted to ``transform_columns_``.
        """
        return block[self.transform_columns_]

    # Méthode auxiliaire d'entraînement du transformer
    def _fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> None:
        """Fit a separate transformer for each panel entity.

        Args:
            X: Input features with panel_cols as columns or MultiIndex.
            y: Target variable aligned with X: on the same panel ``MultiIndex``,
                or row by row when the entities are columns of ``X``.

        Raises:
            ValueError: If ``error_handling`` is unknown, if ``entity_kwargs``
                declares one entity twice with different values, if the entities
                of a simple index are not named by ``panel_cols``, if an entity
                label is missing, or if ``y`` does not have the rows of a column
                panel ``X``.
        """
        # Vérification de la stratégie de gestion des erreurs
        self._check_error_handling()

        # Détection du mode factory (une seule fois au fit)
        self.is_factory_mode_ = self._detect_factory_mode()

        # Warning si entity_kwargs fourni mais mode factory actif
        if self.is_factory_mode_ and self.entity_kwargs is not None:
            warnings.warn(
                "entity_kwargs is ignored when transformer is a callable factory. "
                "The factory is responsible for entity-specific parameterization."
            )

        # Paramètres par entité sur des clés normalisées : 'FR' désigne ('FR',)
        self._entity_kwargs = normalize_entity_mapping(self.entity_kwargs or {})

        # Détection du format des données (colonnes vs MultiIndex)
        # Après validation, les colonnes peuvent avoir été converties en index
        self._has_multiindex = is_panel_data(X)

        # Détermination des colonnes à transformer
        if self._has_multiindex:
            # Les colonnes panel sont dans l'index, on transforme toutes les colonnes
            self.transform_columns_ = list(X.columns)
        else:
            # Sur un index simple, seules les colonnes panel_cols désignent les entités
            if not self.panel_cols:
                raise ValueError(
                    "panel_cols must name the entity columns when X has a simple index; "
                    "for a panel held in the index, use a MultiIndex (entity levels, then dates)"
                )
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

        # Découpage par entité (dans l'ordre d'apparition des données)
        blocks = list(self._entity_blocks(X))

        # Blocs de y par entité : lus sur l'index de panel de y, ou, quand les
        # entités sont des colonnes de X, sur les positions des lignes de X
        y_blocks = None
        if y is not None:
            if self._has_multiindex:
                y_blocks = {entity: block for entity, _, block in iter_entity_blocks(y)}
            else:
                if len(y) != len(X):
                    raise ValueError(
                        f"y has {len(y)} rows, X has {len(X)}: with the entities held in "
                        "columns, y is matched to the rows of X by position"
                    )
                y_blocks = {
                    entity: y.iloc[np.flatnonzero(mask)].set_axis(block.index)
                    for entity, mask, block in blocks
                }

        # Extraction du nombre d'entités
        self.entities_ = len(blocks)

        # Fit parallèle ou séquentiel
        if self.n_jobs == 1:
            self._fit_sequential(blocks, y_blocks)
        else:
            self._fit_parallel(blocks, y_blocks)

    # Méthode auxiliaire d'entraînement séquenciel des transformers
    def _fit_sequential(
        self,
        blocks: List[Tuple[tuple, np.ndarray, Optional[pd.DataFrame]]],
        y_blocks: Optional[Dict[tuple, Union[pd.Series, pd.DataFrame]]]
    ) -> None:
        """Fit transformers sequentially for each entity.

        Args:
            blocks: Entity blocks ``(entity_key, mask, block)`` of the input data.
            y_blocks: Target block of each entity, or None without target.
        """
        # Parcours des entités
        for entity_key, _, block in blocks:
            try:
                # Création du transformer pour cette entité
                entity_transformer = self._create_entity_transformer(entity_key)

                # Extraction des données de l'entité
                X_entity = self._entity_data(block)

                # Extraction de y pour cette entité si fourni
                y_entity = y_blocks[entity_key] if y_blocks is not None else None

                # Fit du transformer
                entity_transformer.fit(X_entity, y_entity)

                # Ajout du transformer au dictionnaire
                self.transformers_[entity_key] = entity_transformer

            except Exception as e:
                self._handle_entity_error(entity_key, e, "fitting")

    # Méthode auxiliaire d'entraînement parallèle des transformers
    def _fit_parallel(
        self,
        blocks: List[Tuple[tuple, np.ndarray, Optional[pd.DataFrame]]],
        y_blocks: Optional[Dict[tuple, Union[pd.Series, pd.DataFrame]]]
    ) -> None:
        """Fit transformers in parallel for each entity.

        The errors raised in the workers are sent back with the results and
        handled in the order of the entities, as in the sequential fit.

        Args:
            blocks: Entity blocks ``(entity_key, mask, block)`` of the input data.
            y_blocks: Target block of each entity, or None without target.
        """
        # Référence à self pour la closure
        parent = self

        def fit_single_entity(entity_key, X_entity, y_entity) -> tuple:
            """Fit a transformer for a single entity.

            Args:
                entity_key: Normalized entity key.
                X_entity: Rows of the entity, indexed by date.
                y_entity: Target of the entity, or None.

            Returns:
                Tuple of (entity_key, fitted_transformer, error): the transformer
                is None and the error is set when the fit failed.
            """
            try:
                # Création du transformer pour cette entité
                entity_transformer = parent._create_entity_transformer(entity_key)
                # Entrainement du transformer
                entity_transformer.fit(X_entity, y_entity)
            except Exception as e:
                # L'erreur est rendue au processus principal, qui applique error_handling
                return entity_key, None, e
            return entity_key, entity_transformer, None

        # Exécution parallèle
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_single_entity)(
                entity_key,
                self._entity_data(block),
                y_blocks[entity_key] if y_blocks is not None else None,
            )
            for entity_key, _, block in blocks
        )

        # Collecte des résultats, erreurs traitées comme en séquentiel
        for entity_key, fitted_transformer, error in results:
            if error is not None:
                self._handle_entity_error(entity_key, error, "fitting")
            else:
                self.transformers_[entity_key] = fitted_transformer

    # Méthode de transformation des données
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using entity-specific transformers.

        Args:
            X: Input data with same structure as during fit.

        Returns:
            Transformed data with the layout of the input (entities in the index
            or in columns), its rows in the input order when every entity keeps
            its rows. Empty data gives an empty copy of the input.

        Raises:
            ValueError: If an entity is unseen at fit under ``error_handling='raise'``.
        """
        return self._apply_per_entity(X, 'transform', 'transforming')

    # Méthode de transformation inverse
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform data using entity-specific transformers.

        Args:
            X: Transformed data.

        Returns:
            Original data format. Empty data gives an empty copy of the input.

        Raises:
            AttributeError: If base transformer doesn't support inverse_transform.
            ValueError: If an entity is unseen at fit under ``error_handling='raise'``.
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

        # Inversion entité par entité
        X_result = self._apply_per_entity(X, 'inverse_transform', 'inverse_transforming')

        # Restauration de la structure originale si conversion appliquée
        return self._restore_structure_if_converted(X_result)

    # Méthode auxiliaire d'application d'une méthode des transformers par entité
    def _apply_per_entity(self, X: pd.DataFrame, method: str, operation: str) -> pd.DataFrame:
        """Apply a method of each fitted entity transformer and reassemble the data.

        The rows of an entity unseen at ``fit``, or whose transformer fails
        without raising (``error_handling`` other than ``'raise'``), are kept
        untransformed. When every entity output is made of input rows (same
        dates, or some of them), the output follows the row order of ``X``;
        otherwise the entities are concatenated in order of first appearance
        (with fresh row labels ``0..n-1`` when the entities are columns). Data
        without rows holds no entity: its empty copy is returned, since no
        entity transformer can be called.

        Args:
            X: Data to process.
            method: Name of the entity transformer method (``'transform'`` or
                ``'inverse_transform'``).
            operation: Name of the operation in error messages.

        Returns:
            The reassembled data.

        Raises:
            ValueError: If ``error_handling`` is unknown.
        """
        # Vérification de la stratégie, modifiable par set_params après le fit
        self._check_error_handling()

        # Données vides : aucune entité, copie vide de l'entrée
        if len(X) == 0:
            return X.copy()

        # Parties de sortie et positions des lignes d'entrée qu'elles reprennent
        parts, positions = [], []

        # Parcours des entités, dans l'ordre d'apparition des données
        for entity_key, mask, block in self._entity_blocks(X):
            rows = np.flatnonzero(mask)
            # Entité absente du fit : données conservées telles quelles
            if entity_key not in self.transformers_:
                self._handle_unknown_entity(entity_key)
                parts.append(X.iloc[rows])
                positions.append(rows)
                continue

            try:
                # Application de la méthode du transformer de l'entité
                X_entity = self._entity_data(block)
                output = getattr(self.transformers_[entity_key], method)(X_entity)
                # Réattachement de l'entité à la sortie
                part, kept = self._reassemble_entity(output, entity_key, X, rows, X_entity.index)
            except Exception as e:
                # Gestion de l'erreur, puis conservation des données non transformées
                self._handle_entity_error(entity_key, e, operation)
                part, kept = X.iloc[rows], rows
            parts.append(part)
            positions.append(kept)

        # Concaténation des résultats
        X_result = pd.concat(parts, axis=0)

        # Ordre des lignes d'entrée restitué quand chaque sortie reprend des lignes d'entrée
        if all(kept is not None for kept in positions):
            return X_result.iloc[np.argsort(np.concatenate(positions), kind='stable')]
        # Sinon, entités en colonnes : étiquettes neuves, faute d'étiquettes d'entrée communes
        if not is_panel_data(X):
            return X_result.reset_index(drop=True)
        return X_result

    # Méthode auxiliaire de reconstitution de la structure de panel par entité
    def _reassemble_entity(
        self,
        output: Union[np.ndarray, pd.DataFrame],
        entity_key: tuple,
        X: pd.DataFrame,
        rows: np.ndarray,
        entity_index: pd.Index
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Give the output of an entity transformer the layout of the input data.

        Entities in the index: the entity levels are re-attached to the dates
        of the output. Entities in columns: the identifier columns (``panel_cols``,
        ``time_col``) and the row labels of the input rows are re-attached; an
        output whose dates are not all input dates of the entity (an array of
        another length is indexed by positions) gets the entity labels and its
        own index as identifier columns.

        Args:
            output: Output of the entity transformer (array or DataFrame).
            entity_key: Normalized entity key.
            X: Data passed to ``transform`` or ``inverse_transform``.
            rows: Positions of the entity rows in ``X``.
            entity_index: Dates of the entity rows passed to the transformer.

        Returns:
            Tuple ``(part, kept)``: the reassembled rows of the entity, and the
            positions in ``X`` of the input rows they correspond to (None when
            the output rows are not input rows).
        """
        # Un tableau d'une autre longueur n'est indexé que par des positions, jamais des lignes d'entrée
        positional = isinstance(output, np.ndarray) and output.shape[0] != len(entity_index)
        # Sortie tabulaire indexée par les dates de l'entité
        output = self._as_entity_frame(output, entity_key, entity_index)

        # Positions, parmi les lignes de l'entité, des lignes de la sortie
        if positional:
            local = None
        elif output.index.equals(entity_index):
            local = np.arange(len(entity_index))
        elif entity_index.is_unique:
            local = entity_index.get_indexer(output.index)
            local = local if (local >= 0).all() else None
        else:
            local = None
        kept = rows[local] if local is not None else None

        # Entités portées par l'index : réattachement des niveaux d'entité aux dates de la sortie
        if is_panel_data(X):
            index = build_panel_index(entity_key, output.index, names=X.index.names)
            return output.set_axis(index), kept

        # Entités portées par des colonnes : identifiants et étiquettes des lignes d'entrée
        id_columns = [col for col in X.columns if col not in self.transform_columns_]
        if kept is not None:
            identifiers = X.iloc[kept][id_columns]
            values = output.set_axis(identifiers.index)
        else:
            # Lignes nouvelles : étiquettes d'entité constantes, dates de la sortie
            identifiers = pd.DataFrame(
                {col: [label] * len(output) for col, label in zip(self.panel_cols, entity_key)}
            )
            if self.time_col in X.columns:
                identifiers[self.time_col] = output.index.to_numpy()
            values = output.reset_index(drop=True)
        part = pd.concat([identifiers, values], axis=1)

        # Colonnes dans l'ordre de l'entrée, colonnes créées par le transformer en fin
        ordered = [col for col in X.columns if col in part.columns]
        ordered += [col for col in part.columns if col not in ordered]
        return part[ordered], kept

    # Méthode auxiliaire de conversion de la sortie d'un transformer d'entité
    def _as_entity_frame(
        self,
        output: Union[np.ndarray, pd.DataFrame],
        entity_key: tuple,
        entity_index: pd.Index
    ) -> pd.DataFrame:
        """Turn the output of an entity transformer into a DataFrame.

        Args:
            output: Output of the transformer (array or DataFrame).
            entity_key: Entity identifier as tuple.
            entity_index: Dates of the entity rows passed to the transformer. An
                array output with one row per input row takes these dates as
                index; an array of another length keeps positions.

        Returns:
            The output as a DataFrame.
        """
        # Une sortie tabulaire garde son index et ses colonnes
        if not isinstance(output, np.ndarray):
            return output

        # Un tableau n'a pas d'index : les dates de l'entité sont reprises ligne à ligne
        # quand le transformateur conserve le nombre de lignes
        same_rows = len(entity_index) == output.shape[0]
        # Gestion des noms de colonnes
        if output.shape[1] != len(self.transform_columns_):
            # Le transformer a changé le nombre de colonnes
            cols = self._generate_column_names(output.shape[1], self.transformers_[entity_key])
        else:
            cols = self.transform_columns_

        # Création d'un DataFrame avec l'index temporel
        return pd.DataFrame(output, columns=cols, index=entity_index if same_rows else None)

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

        Only the failures of the fit are recorded in ``failed_entities_``, a
        fitted attribute that ``transform`` leaves unchanged; later failures are
        reported by the warning (``'warn'``).

        Args:
            entity_key: Entity identifier.
            error: Exception that occurred.
            operation: Name of the operation ('fitting', 'transforming', etc.).

        Raises:
            Exception: If error_handling is 'raise'.
        """
        # Entités en échec au fit seulement : l'état ajusté ne dépend pas des appels suivants
        if operation == 'fitting':
            self.failed_entities_.append(entity_key)

        # Distinction suivant la méthode de gestion des erreurs
        if self.error_handling == 'raise':
            raise error
        elif self.error_handling == 'warn':
            # Type de l'erreur explicité : str() d'une KeyError n'est que la clé
            warnings.warn(
                f"Error {operation} entity {entity_key}: {type(error).__name__}: {error}. Skipping."
            )
        # 'ignore' : ne rien faire

    # Méthode auxiliaire de gestion des entités inconnues
    def _handle_unknown_entity(self, entity_key: tuple) -> None:
        """Handle unknown entities encountered during transform.

        Args:
            entity_key: Unknown entity identifier.

        Raises:
            ValueError: If error_handling is 'raise'.
        """
        # Initialisation du message d'erreur
        msg = f"Entity {entity_key} was not seen during fit"

        # Distinction suivant la méthode de gestion des erreurs ; une valeur inconnue
        # en entrée est une ValueError, comme une catégorie inconnue pour sklearn
        if self.error_handling == 'raise':
            raise ValueError(msg)
        elif self.error_handling == 'warn':
            warnings.warn(f"{msg}. Using untransformed data.")
        # 'ignore' : ne rien faire

    # Méthode auxiliaire de vérification de la stratégie de gestion des erreurs
    def _check_error_handling(self) -> None:
        """Check that ``error_handling`` is one of the accepted strategies.

        Raises:
            ValueError: If ``error_handling`` is not 'raise', 'warn' or 'ignore'.
        """
        if self.error_handling not in _ERROR_HANDLING_OPTIONS:
            raise ValueError(
                f"error_handling must be one of {list(_ERROR_HANDLING_OPTIONS)}, "
                f"got {self.error_handling!r}"
            )


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
    def set_params(self, **params: Any) -> 'PanelwiseTransformer':
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
        entity_key = normalize_entity_key(entity_key)
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