"""Base classes for sklearn-compatible time series and panel transformers.

``TimeSeriesTransformerMixin`` extracts the time index, ``PanelTimeSeriesTransformer``
is the abstract transformer that validates series and panels before calling the
subclass, ``ReversibleTransformerMixin`` adds the ``inverse_transform`` contract and
``InverseTransformer`` swaps ``transform`` and ``inverse_transform`` of a fitted
transformer.
"""
# Importation des modules
# Modules de base
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import warnings
# Sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

# Importation des fonctions de validation
from ..utils.validation import (
    restore_original_structure,
    validate_temporal_data,
    validate_entities_grouped,
    validate_sorted_within_groups
)
# Règles de conversion en dates communes à toute la validation temporelle
from ..utils.validation.utils import _convert_to_datetime


# Mixin pour les séries temporelles
class TimeSeriesTransformerMixin:
    """Mixin class for time series transformers.

    Provides the extraction of a ``DatetimeIndex`` from a time column, a
    ``DatetimeIndex``, the last level of a panel ``MultiIndex`` or any index whose
    labels are dates.

    Examples:
        >>> import pandas as pd
        >>> class MyTransformer(TimeSeriesTransformerMixin):
        ...     def transform(self, X):
        ...         self.time_index_ = self._validate_time_index(X)
        ...         return X
        >>> dates = pd.date_range('2023-01-01', periods=5)
        >>> df = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=dates)
        >>> transformer = MyTransformer()
        >>> _ = transformer.transform(df)
        >>> isinstance(transformer.time_index_, pd.DatetimeIndex)
        True
    """

    # Méthode auxiliaire de validation de l'index
    def _validate_time_index(self, X: pd.DataFrame, time_col: Optional[str] = None) -> pd.DatetimeIndex:
        """Validate and extract time index from data.

        Extracts or converts time index from various formats:
        - From a specified time column
        - From an existing DatetimeIndex
        - From the last level of a MultiIndex (panel data)
        - By converting a non-datetime index

        The conversion follows the rules of ``validate_temporal_data``: datetimes,
        strings parsable as dates and ``Period`` values (read at the first instant of
        the period) are accepted; numeric labels (``RangeIndex``, integer years) are not
        dates and are rejected, except on an empty index.

        Args:
            X: Input data with time information
            time_col: Name of time column. If None, or if it is not a column of
                ``X``, the index is used.

        Returns:
            Validated DatetimeIndex extracted from the data

        Raises:
            ValueError: If time index cannot be determined or converted

        Examples:
            >>> import pandas as pd
            >>> dates = pd.date_range('2023-01-01', periods=3)
            >>> df = pd.DataFrame({'value': [1, 2, 3]}, index=dates)
            >>> mixin = TimeSeriesTransformerMixin()
            >>> len(mixin._validate_time_index(df))
            3
            >>> df = pd.DataFrame({
            ...     'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            ...     'value': [1, 2, 3]
            ... })
            >>> isinstance(mixin._validate_time_index(df, time_col='date'), pd.DatetimeIndex)
            True
        """
        if time_col and time_col in X.columns:
            # Utilisation de la colonne temporelle spécifiée
            converted = _convert_to_datetime(X[time_col])
            if converted is None:
                raise ValueError(
                    f"Cannot determine time index. Column '{time_col}' cannot be "
                    "converted to datetime."
                )
        else:
            # Index temporel : dernier niveau pour un MultiIndex (entités, puis dates)
            index = X.index.get_level_values(-1) if isinstance(X.index, pd.MultiIndex) else X.index
            converted = _convert_to_datetime(index)
            if converted is None:
                raise ValueError(
                    "Cannot determine time index. Please provide a datetime index "
                    "or specify time_col parameter."
                )

        # Une colonne convertie est une Series : le contrat annonce un DatetimeIndex
        return pd.DatetimeIndex(converted)


# Transformer pour les données de panel
class PanelTimeSeriesTransformer(BaseEstimator, TransformerMixin, TimeSeriesTransformerMixin, ABC):
    """Abstract base class for panel time series transformers.

    This class provides a template for creating sklearn-compatible transformers
    that handle both univariate time series and panel data (multiple entities
    observed over time). It validates the input (time labels, duplicates, sorting,
    contiguity of entities), optionally sorts it, and checks at ``transform`` that
    the columns are those seen at ``fit``. Subclasses implement ``_fit`` and
    ``_transform``.

    Accepted input: a two-dimensional ``DataFrame`` (sklearn convention) whose time
    labels are in the index (a ``MultiIndex`` with the entities first and the dates
    last for a panel) or in the ``time_col`` / ``panel_cols`` columns. A single
    series is passed as ``series.to_frame()``.

    Parameters:
        time_col (Optional[str]): Name of the time column. Default is None (time
            labels in the index, last level for a ``MultiIndex``).
        panel_cols (Optional[List[str]]): Columns identifying panel dimensions
            (e.g., ['country', 'sector']). Requires ``time_col``. Default is None
            (entities, if any, in the first levels of a ``MultiIndex``).
        validate_input (bool): Whether to validate input data structure and
            temporal ordering. Default is True.
        strict_validation (bool): If True, raises errors on validation failures;
            if False, emits warnings. Default is True.
        auto_sort (bool): If True, automatically sorts unsorted data by panel
            and time columns. Default is False.
        convert_cols_to_index (bool): If True, the validated ``time_col`` and
            ``panel_cols`` columns are moved into the index (entities first, dates
            last) and the metadata needed to give them back is stored. If False,
            they stay columns: the validation checks and converts them (dates read
            as ``datetime64``) without changing the layout. Default is False.

    Attributes:
        is_panel_ (bool): Whether the data seen at ``fit`` is a panel, read from
            the layout (``MultiIndex`` or ``panel_cols``), whatever the number of
            entities.
        n_features_in_ (int): Number of columns seen at ``fit`` (sklearn
            convention, read by ``Pipeline.n_features_in_``): the time and panel
            columns count while they are columns, not once moved into the index.
        feature_names_in_ (np.ndarray): Names of those columns, as an object
            array (sklearn convention).
        conversion_metadata_ (dict): Structure of the column data seen at ``fit``
            (moved columns, their positions and names), recorded by
            ``validate_temporal_data`` when ``convert_cols_to_index=True``. It is
            set at ``fit`` only; it keeps no row label, so the restored columns
            come back with a fresh ``RangeIndex``.

    Examples:
        >>> import pandas as pd
        >>> class SimpleScaler(PanelTimeSeriesTransformer):
        ...     def _fit(self, X, y=None):
        ...         self.mean_ = X.select_dtypes(include='number').mean()
        ...
        ...     def _transform(self, X):
        ...         numeric_cols = X.select_dtypes(include='number').columns
        ...         X_transformed = X.copy()
        ...         X_transformed[numeric_cols] = X[numeric_cols] - self.mean_
        ...         return X_transformed
        >>> dates = pd.date_range('2023-01-01', periods=5)
        >>> df = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
        >>> SimpleScaler().fit_transform(df)['value'].tolist()
        [-2.0, -1.0, 0.0, 1.0, 2.0]
        >>> idx = pd.MultiIndex.from_product(
        ...     [['A', 'B'], pd.date_range('2023-01-01', periods=3)], names=['entity', 'date']
        ... )
        >>> panel = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=idx)
        >>> scaler = SimpleScaler().fit(panel)
        >>> scaler.is_panel_, scaler.n_features_in_
        (True, 1)
    """

    # Initialisation
    def __init__(self,
                 time_col: Optional[str] = None,
                 panel_cols: Optional[List[str]] = None,
                 validate_input: bool = True,
                 strict_validation: bool = True,
                 auto_sort: bool = False,
                 convert_cols_to_index: bool = False):
        """Initialize the transformer.

        Args:
            time_col: Name of the temporal column. If None, uses the index
                (last level for MultiIndex). Default is None.
            panel_cols: Columns identifying panel dimensions (entity identifiers).
                If None with MultiIndex, auto-detects from first n-1 levels.
                Default is None.
            validate_input: Whether to activate input data validation
            strict_validation: If True, raises errors; if False, emits warnings
            auto_sort: If True, automatically sorts improperly ordered data
            convert_cols_to_index: If True, moves the time_col and panel_cols
                columns into the index and stores the metadata needed to restore
                them; if False, they stay columns

        Examples:
            >>> class Identity(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X): return X
            >>> transformer = Identity(time_col='date', panel_cols=['entity'])
            >>> transformer.get_params()['panel_cols']
            ['entity']
        """
        # Initialisation des paramètres en attributs
        self.time_col = time_col
        self.panel_cols = panel_cols
        self.validate_input = validate_input
        self.strict_validation = strict_validation
        self.auto_sort = auto_sort
        self.convert_cols_to_index = convert_cols_to_index


    # Méthode de détection de la structure des données
    def _detect_data_structure(self, X: pd.DataFrame) -> Tuple[bool, bool, bool]:
        """Detect data structure (panel vs time series, columns vs index).

        Determines whether data is panel or time series, and whether temporal
        and panel information is in columns or index levels. A ``time_col``
        without ``panel_cols`` designates a time series in columns, unless it
        is not a column of ``X`` and ``X`` has a ``MultiIndex``: the panel is
        then read from the index, as with ``time_col=None``.

        Args:
            X: Input DataFrame

        Returns:
            Tuple of (is_panel, panel_in_index, time_in_index):
                - is_panel: True if panel data (multiple entities)
                - panel_in_index: True if panel columns are in index levels
                - time_in_index: True if time column is in index

        Raises:
            ValueError: If structure is ambiguous or invalid

        Examples:
            >>> import pandas as pd
            >>> class Identity(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X): return X
            >>> df = pd.DataFrame({
            ...     'entity': ['A', 'A', 'B', 'B'],
            ...     'date': pd.date_range('2023-01-01', periods=2).tolist() * 2,
            ...     'value': [1, 2, 3, 4]
            ... })
            >>> Identity(time_col='date', panel_cols=['entity'])._detect_data_structure(df)
            (True, False, False)
            >>> panel = df.set_index(['entity', 'date'])
            >>> Identity()._detect_data_structure(panel)
            (True, True, True)
            >>> Identity(time_col='date')._detect_data_structure(panel)
            (True, True, True)
            >>> Identity()._detect_data_structure(df.iloc[:2].set_index('date'))
            (False, False, True)
        """
        # Cas 1: time_col et panel_cols spécifiés → panel colonnes
        if self.time_col is not None and self.panel_cols is not None:
            return True, False, False

        # Cas 2: Seulement time_col → série temporelle colonnes, sauf quand time_col
        # n'est pas une colonne d'un MultiIndex : le panel se lit alors dans l'index
        if self.time_col is not None:
            if (
                self.time_col not in X.columns
                and isinstance(X.index, pd.MultiIndex)
                and X.index.nlevels >= 2
            ):
                return True, True, True
            return False, False, False

        # Cas 3: panel_cols sans time_col → invalide
        if self.panel_cols is not None:
            raise ValueError(
                "Invalid configuration: panel_cols specified but time_col is None. "
                "For column-based panel data, both time_col and panel_cols must be specified. "
                "For MultiIndex panel data, both should be None (auto-detected from index)."
            )

        # Cas 4: Les deux None → détection depuis l'index
        if isinstance(X.index, pd.MultiIndex):
            if X.index.nlevels < 2:
                raise ValueError(
                    "MultiIndex must have at least 2 levels for panel data "
                    "(entity levels + time level). For simple time series, use DatetimeIndex."
                )
            return True, True, True

        # Index simple : étiquettes temporelles selon les règles de la validation
        # (dates, chaînes de dates, périodes ; pas d'étiquettes numériques)
        if _convert_to_datetime(X.index) is None:
            raise ValueError(
                "When time_col and panel_cols are None, the index must hold dates "
                "(datetimes, date strings or periods) for a time series, or be a "
                "MultiIndex for panel data."
            )
        return False, False, True

    # Méthode d'entraînement
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PanelTimeSeriesTransformer':
        """Fit the transformer.

        Validates input data, determines panel structure, and calls the
        subclass-specific _fit() method for learning transformation parameters.

        Args:
            X: Input features: DataFrame with time and optional panel information
                in the index or in columns
            y: Target variable (optional), passed to ``_fit`` as is

        Returns:
            Self for method chaining (sklearn convention)

        Raises:
            ValueError: If ``X`` is not a DataFrame, if the data fails the
                validation, or if a panel identifier is missing while
                ``validate_input`` and ``strict_validation`` are True

        Examples:
            >>> import pandas as pd
            >>> class Identity(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X): return X
            >>> df = pd.DataFrame({
            ...     'date': pd.date_range('2023-01-01', periods=10),
            ...     'value': range(10),
            ...     'feature': range(10, 20)
            ... })
            >>> transformer = Identity(time_col='date', convert_cols_to_index=True).fit(df)
            >>> transformer.is_panel_
            False
            >>> transformer.n_features_in_
            2
        """
        # Données à deux dimensions seulement (convention sklearn)
        self._check_input_type(X)

        # Périodes de l'index d'origine, rendues par l'inversion
        self._time_period_freq = self._period_frequency(X)

        # Validation des données si demandée
        if self.validate_input:
            # Validation de X
            X = self._validate_input(X)

        # Structure des colonnes passées dans l'index, figée au fit : sans
        # étiquette de ligne, propre aux seules données de cet appel
        self.__dict__.pop('conversion_metadata_', None)
        metadata = getattr(self, '_current_conversion_metadata', None) if self.validate_input else None
        if metadata is not None:
            self.conversion_metadata_ = {**metadata, 'original_index': None}

        # Détection de la structure
        is_panel, panel_in_index, time_in_index = self._detect_data_structure(X)

        # Stockage métadonnées structure
        self.is_panel_ = is_panel
        self._panel_in_index = panel_in_index
        self._time_in_index = time_in_index

        # Extraction noms niveaux panel si MultiIndex
        if is_panel and panel_in_index:
            self._panel_level_names = X.index.names[:-1]  # Tous sauf dernier (temps)
        else:
            self._panel_level_names = None

        # Stockage des métadonnées, sous les noms de la convention sklearn
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        # Vérification de la cohérence des individus du panel : erreur sous
        # validation stricte, avertissement sinon
        if self.is_panel_:
            is_consistent, issues = self._check_panel_consistency(X)
            if not is_consistent:
                msg = f"Panel structure issues: {'; '.join(issues)}"
                if self.validate_input and self.strict_validation:
                    raise ValueError(msg)
                warnings.warn(msg)

        # Appel de la méthode d'entraînement spécifique à implémenter
        self._fit(X, y)

        return self

    # Méthode auxiliaire d'entraînement
    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Fit implementation to be provided by subclasses.

        This method should implement the transformer-specific learning logic,
        such as computing statistics, learning parameters, or storing reference
        values needed for transformation.

        Args:
            X: Validated input features
            y: Target variable (optional)

        Examples:
            >>> class MeanNormalizer(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None):
            ...         self.means_ = X.select_dtypes(include='number').mean()
            ...
            ...     def _transform(self, X):
            ...         return X - self.means_
            >>> import pandas as pd
            >>> df = pd.DataFrame({'v': [1.0, 3.0]}, index=pd.date_range('2023-01-01', periods=2))
            >>> MeanNormalizer().fit(df).means_.tolist()
            [2.0]
        """

    # Méthode de transformation
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.

        Validates the transformer is fitted, validates input data if enabled,
        checks that the columns are those seen at ``fit`` (same names, same order)
        and applies the transformation defined in the subclass's _transform() method.

        Args:
            X: Input data to transform

        Returns:
            Transformed data with same structure as input

        Raises:
            NotFittedError: If transformer has not been fitted yet
            ValueError: If ``X`` is not a DataFrame, if the data fails the
                validation or if its columns differ from those seen at ``fit``

        Examples:
            >>> import pandas as pd
            >>> class Identity(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X): return X
            >>> train = pd.DataFrame({'value': [1, 2, 3]}, index=pd.date_range('2023-01-01', periods=3))
            >>> test = pd.DataFrame({'value': [4, 5]}, index=pd.date_range('2023-01-04', periods=2))
            >>> Identity().fit(train).transform(test)['value'].tolist()
            [4, 5]
        """
        # Vérification que le transformer est ajusté
        check_is_fitted(self)

        # Données à deux dimensions seulement (convention sklearn)
        self._check_input_type(X)

        # Validation des données
        if self.validate_input:
            X = self._validate_input(X)

        # Vérification des colonnes vues au fit
        self._check_feature_names(X)

        # Appel de la méthode spécifique
        return self._transform(X)

    # Méthode auxiliaire de transformation
    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform implementation to be provided by subclasses.

        This method should implement the core transformation logic using
        parameters learned during fit().

        Args:
            X: Validated input data

        Returns:
            Transformed data maintaining original structure

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> class LogTransformer(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None):
            ...         pass
            ...
            ...     def _transform(self, X):
            ...         return np.log1p(X)
            >>> df = pd.DataFrame({'v': [0.0]}, index=pd.date_range('2023-01-01', periods=1))
            >>> LogTransformer().fit_transform(df)['v'].tolist()
            [0.0]
        """

    # Méthode auxiliaire de contrôle du type des données
    @staticmethod
    def _check_input_type(X) -> None:
        """Check that the data is a two-dimensional pandas ``DataFrame``.

        Args:
            X: Data passed to ``fit`` or ``transform``

        Raises:
            ValueError: If ``X`` is not a DataFrame; a Series is pointed to ``to_frame()``

        Examples:
            >>> import pandas as pd
            >>> PanelTimeSeriesTransformer._check_input_type(pd.Series([1.0], name='v'))
            Traceback (most recent call last):
                ...
            ValueError: X must be a pandas DataFrame, got Series; pass series.to_frame() for a single series
        """
        if isinstance(X, pd.DataFrame):
            return
        hint = "; pass series.to_frame() for a single series" if isinstance(X, pd.Series) else ""
        raise ValueError(f"X must be a pandas DataFrame, got {type(X).__name__}{hint}")

    # Méthode auxiliaire de contrôle des colonnes vues au fit
    def _check_feature_names(self, X: pd.DataFrame) -> None:
        """Check that the data has the columns seen at ``fit``, in the same order.

        This is the sklearn rule. A subclass whose contract tolerates other columns
        at ``transform`` (``HighFrequencyImputer`` carries extra columns through)
        overrides this method and applies its own check.

        Args:
            X: Data passed to ``transform``, after validation

        Raises:
            ValueError: If a column is unseen at ``fit``, missing, or out of order

        Examples:
            >>> import pandas as pd
            >>> class Identity(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X): return X
            >>> df = pd.DataFrame({'a': [1], 'b': [2]}, index=pd.date_range('2023-01-01', periods=1))
            >>> transformer = Identity().fit(df)
            >>> transformer._check_feature_names(df[['b', 'a']])
            Traceback (most recent call last):
                ...
            ValueError: The feature names should match those that were passed during fit; same names in another order: ['b', 'a'] instead of ['a', 'b']
        """
        # Extraction des colonnes
        names = list(X.columns)
        fitted = list(self.feature_names_in_)
        if names == fitted:
            return

        # Détail de l'écart : colonnes nouvelles, absentes, ou seulement réordonnées
        unseen = [name for name in names if name not in fitted]
        missing = [name for name in fitted if name not in names]
        details = []
        if unseen:
            details.append(f"unseen at fit time: {unseen}")
        if missing:
            details.append(f"seen at fit time, yet now missing: {missing}")
        if not details:
            details.append(f"same names in another order: {names} instead of {fitted}")
        raise ValueError(
            "The feature names should match those that were passed during fit; "
            + '; '.join(details)
        )

    # Méthode auxiliaire de repérage des périodes de l'index
    @staticmethod
    def _period_frequency(X: pd.DataFrame) -> Optional[str]:
        """Return the frequency of a ``PeriodIndex`` (or ``Period`` last level), else None.

        Args:
            X: Input DataFrame

        Returns:
            The period frequency string (e.g. ``'M'``), or None if the time labels of
            the index are not periods.

        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'v': [1, 2]}, index=pd.period_range('2023-01', periods=2, freq='M'))
            >>> PanelTimeSeriesTransformer._period_frequency(df)
            'M'
        """
        # Extraction de l'index
        index = X.index
        # Extraction de la fréquence du nom de l'index
        labels = index.get_level_values(-1) if isinstance(index, pd.MultiIndex) else index
        return labels.freqstr if isinstance(labels, pd.PeriodIndex) else None

    # Méthode auxiliaire de validation des données
    def _validate_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate input data with comprehensive temporal checks.

        Performs comprehensive validation including:
        - Data type and column presence
        - Temporal validation (index or column)
        - Uniqueness of (entity, date) combinations
        - Global and intra-group temporal sorting (panel)
        - Contiguous grouping of entities (panel)

        The caller's data is never modified: conversions and sorting apply to copies.

        Args:
            data: Input data to validate

        Returns:
            Validated data (potentially sorted if auto_sort=True)

        Raises:
            ValueError: If validation fails and strict_validation=True

        Examples:
            >>> import pandas as pd
            >>> class Identity(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X): return X
            >>> panel = pd.DataFrame({
            ...     'value': [1, 2, 3, 4]
            ... }, index=pd.MultiIndex.from_arrays([
            ...     ['B', 'B', 'A', 'A'],
            ...     pd.to_datetime(['2023-01-01', '2023-01-02'] * 2)
            ... ]))
            >>> validated = Identity(auto_sort=True)._validate_input(panel)
            >>> validated.index.get_level_values(0).tolist()
            ['A', 'A', 'B', 'B']
        """
        # 1. Vérification du type, et métadonnées de conversion propres à cet appel
        self._check_input_type(data)
        self._current_conversion_metadata = None

        # 2. Détection structure (un panel en index a toujours le temps en index)
        _, panel_in_index, time_in_index = self._detect_data_structure(data)

        # 3. Routage selon structure
        if panel_in_index:
            return self._validate_multiindex_panel(data)
        if time_in_index:
            return self._validate_index_time_series(data)
        return self._validate_column_based(data)

    # Méthode auxiliaire du conseil de tri des messages de validation
    def _sorting_advice(self, sort_keys: str) -> str:
        """Phrase the sorting advice closing a validation message.

        ``auto_sort`` is worth suggesting only when the transformer exposes it
        as a parameter. A subclass that fixes it — ``HighFrequencyImputer``
        refuses any implicit sort, which would silently move its windows and
        aggregates — must ask the caller to sort instead.

        Args:
            sort_keys: What the data must be sorted by, e.g. ``'panel_cols'``.

        Returns:
            The advice sentence.

        Examples:
            >>> class Identity(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X): return X
            >>> Identity()._sorting_advice('panel_cols')
            'Consider sorting by panel_cols or enabling auto_sort=True'
        """
        # Paramètre exposé : le tri automatique est une solution
        if 'auto_sort' in self.get_params(deep=False):
            return f"Consider sorting by {sort_keys} or enabling auto_sort=True"
        # Paramètre figé par la sous-classe : seul un tri par l'appelant est possible
        return (
            f"Sort the data by {sort_keys} before calling "
            f"{type(self).__name__}, which never sorts implicitly"
        )

    # Méthodes auxiliaires de validation selon la structure
    # Méthode auxiliaire de validation de données de panel sous forme de multi-index
    def _validate_multiindex_panel(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate MultiIndex panel data.

        Validates panel data where entities are identified by the first n-1 levels
        of a MultiIndex (at least two levels, checked by ``_detect_data_structure``)
        and time is in the last level.

        Args:
            data: DataFrame with MultiIndex (entity levels + time level)

        Returns:
            Validated data

        Raises:
            ValueError: If validation fails and strict_validation=True

        Examples:
            >>> import pandas as pd
            >>> class Identity(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X): return X
            >>> idx = pd.MultiIndex.from_arrays([
            ...     ['A', 'A', 'B', 'B'],
            ...     ['2023-01-01', '2023-01-02'] * 2
            ... ], names=['entity', 'date'])
            >>> df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=idx)
            >>> validated = Identity()._validate_multiindex_panel(df)
            >>> validated.index.get_level_values('date').dtype
            dtype('<M8[ns]')
        """
        # Validation via validate_temporal_data : en mode permissif elle avertit
        # sans lever, seule une erreur stricte remonte ici
        try:
            data_validated = validate_temporal_data(
                data=data,
                time_col=None,  # Temps dans index (dernier niveau)
                panel_cols=None,  # Panel dans index (premiers niveaux)
                strict=self.strict_validation,
                sort_data=self.auto_sort,
                return_metadata=False
            )
        except ValueError as e:
            raise ValueError(f"MultiIndex panel validation failed: {e}") from e

        # Vérification groupement entités
        if not validate_entities_grouped(data_validated, panel_cols=None):
            msg = (
                "Panel entities are not contiguous in MultiIndex. "
                + self._sorting_advice('entity levels (e.g. X.sort_index())')
            )
            if self.strict_validation:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

        # Vérification tri intra-groupe
        if not validate_sorted_within_groups(
            data_validated,
            panel_cols=None,  # Auto-détection depuis MultiIndex
            time_col=None     # Temps dans dernier niveau
        ):
            msg = (
                "Time series not sorted within panel groups. "
                + self._sorting_advice('date within each entity (e.g. X.sort_index())')
            )
            if self.strict_validation:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

        return data_validated

    # Méthode auxiliaire de validation de données dont la structure est dans des colonnes
    def _validate_column_based(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate column-based data (panel or time series).

        Validates data where temporal and panel information is stored in columns
        rather than index levels. A ``time_col`` that is not a column leaves the
        index in charge of the time labels, unless ``panel_cols`` are given.

        The checks run on the data indexed by ``panel_cols`` then ``time_col``.
        With ``convert_cols_to_index=True`` this indexed data is returned; otherwise
        the columns are given back at their positions (time column converted to
        dates), with the original row labels unless rows were sorted or dropped.

        Args:
            X: DataFrame with time and panel information in columns

        Returns:
            Validated DataFrame, indexed by ``panel_cols`` then ``time_col`` under
            ``convert_cols_to_index=True``, in columns otherwise

        Raises:
            ValueError: If a panel or time column is missing, if the time labels are
                not dates, or if validation fails and strict_validation=True

        Examples:
            >>> import pandas as pd
            >>> class Identity(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X): return X
            >>> df = pd.DataFrame({
            ...     'entity': ['A', 'A', 'B', 'B'],
            ...     'date': pd.date_range('2023-01-01', periods=2).tolist() * 2,
            ...     'value': [1, 2, 3, 4]
            ... })
            >>> transformer = Identity(time_col='date', panel_cols=['entity'], convert_cols_to_index=True)
            >>> transformer._validate_column_based(df).index.names
            FrozenList(['entity', 'date'])
            >>> list(Identity(time_col='date', panel_cols=['entity'])._validate_column_based(df).columns)
            ['entity', 'date', 'value']
        """
        # Vérification colonnes panel, puis de la colonne temporelle sans laquelle
        # aucun index (entité, date) ne peut être construit
        if self.panel_cols:
            missing_cols = set(self.panel_cols) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Panel columns not found: {missing_cols}")
            if self.time_col not in X.columns:
                raise ValueError(f"Time column '{self.time_col}' not found in data")

        # Vérification colonne/index temporel
        try:
            self._validate_time_index(X, self.time_col)
        except ValueError as e:
            raise ValueError(f"Time validation failed: {e}") from e

        # Validation complète du séquencement temporel via validate_temporal_data :
        # colonnes absentes et dates illisibles étant refusées ci-dessus, seule
        # une erreur stricte (doublons) peut encore remonter
        time_col = self.time_col if self.time_col in X.columns else None
        try:
            X_validated, metadata = validate_temporal_data(
                data=X,
                time_col=time_col,
                panel_cols=self.panel_cols,
                strict=self.strict_validation,
                sort_data=self.auto_sort,
                return_metadata=True
            )
        except ValueError as e:
            raise ValueError(f"Temporal data validation failed: {e}") from e
        # Métadonnées des données de cet appel (index d'origine compris) ; le fit
        # en tire la structure figée ``conversion_metadata_``
        if self.convert_cols_to_index:
            self._current_conversion_metadata = metadata

        # Validations spécifiques panel
        if self.panel_cols:
            # Validation du groupement des entités
            if not validate_entities_grouped(X_validated, panel_cols=self.panel_cols):
                # Message
                msg = (
                    "Panel entities are not contiguous. "
                    + self._sorting_advice('panel_cols')
                )
                # Affichage
                if self.strict_validation:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)
            # validation du tri intra-groupe
            if not validate_sorted_within_groups(
                X_validated,
                panel_cols=self.panel_cols,
                time_col=time_col
            ):
                # Message
                msg = (
                    "Time series not sorted within panel groups. "
                    + self._sorting_advice('panel_cols then time_col')
                )
                # Affichage
                if self.strict_validation:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)

        # Validation série temporelle simple : l'index validé est un DatetimeIndex
        # (colonne temporelle, ou index de dates converti)
        elif not X_validated.index.is_monotonic_increasing:
            msg = "Time series index is not sorted"
            if self.strict_validation:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

        # Sans conversion demandée, colonnes temporelle et d'entité rendues à leur place
        if not self.convert_cols_to_index and metadata['index_was_replaced']:
            return restore_original_structure(X_validated, metadata)
        return X_validated

    # Méthode auxiliaire de validation d'une série temporelle en index
    def _validate_index_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate index-based time series (dates in a flat index, no panel).

        The index labels are dates (checked by ``_detect_data_structure``);
        ``validate_temporal_data`` converts them on a copy.

        Args:
            data: DataFrame whose index holds dates

        Returns:
            Validated data, with a DatetimeIndex

        Raises:
            ValueError: If validation fails and strict_validation=True

        Examples:
            >>> import pandas as pd
            >>> class Identity(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X): return X
            >>> df = pd.DataFrame({'value': [1, 2]}, index=['2023-01-01', '2023-01-02'])
            >>> isinstance(Identity()._validate_index_time_series(df).index, pd.DatetimeIndex)
            True
            >>> df.index.dtype
            dtype('O')
        """
        # Validation du séquencement temporel via validate_temporal_data : en mode
        # permissif elle avertit sans lever, seule une erreur stricte remonte ici
        try:
            data_validated = validate_temporal_data(
                data=data,
                time_col=None,  # Temps dans index
                panel_cols=None,  # Pas de panel
                strict=self.strict_validation,
                sort_data=self.auto_sort,
                return_metadata=False
            )
        except ValueError as e:
            raise ValueError(f"Time series validation failed: {e}") from e

        # Vérification du tri
        if not data_validated.index.is_monotonic_increasing:
            # Message
            msg = "Time series index is not sorted"
            # Affichage
            if self.strict_validation:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

        return data_validated

    # Fonction auxiliaire de vérification de la consistence des éléments du panel
    def _check_panel_consistency(self, X: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check consistency of panel structure.

        Handles both column-based and MultiIndex-based panel data. Verifies that
        panel identifier columns or index levels are present and contain no
        missing values, which would compromise panel integrity.

        Args:
            X: Input data with panel structure (columns or MultiIndex)

        Returns:
            Tuple of (is_consistent, list_of_issues) where is_consistent is
            True if no issues found, and list_of_issues contains error messages

        Examples:
            >>> import pandas as pd
            >>> class Identity(PanelTimeSeriesTransformer):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X): return X
            >>> idx = pd.MultiIndex.from_arrays([
            ...     ['A', None],
            ...     pd.date_range('2023-01-01', periods=2)
            ... ], names=['entity', 'date'])
            >>> df = pd.DataFrame({'value': [1, 2]}, index=idx)
            >>> Identity()._check_panel_consistency(df)
            (False, ['Missing values in panel identifier level: entity'])
        """
        # Initialisation de la liste des problèmes
        issues = []

        # Détection structure si pas déjà fait
        if not hasattr(self, '_panel_in_index'):
            _, panel_in_index, _ = self._detect_data_structure(X)
        else:
            panel_in_index = self._panel_in_index

        if panel_in_index:
            # Panel MultiIndex
            if not isinstance(X.index, pd.MultiIndex):
                issues.append("Expected MultiIndex for panel data")
                return False, issues
            # Vérification qu'il contient au moins deux niveaux
            if X.index.nlevels < 2:
                issues.append("MultiIndex must have at least 2 levels")
                return False, issues

            # Vérification valeurs manquantes dans niveaux entité
            for i in range(X.index.nlevels - 1):  # Tous sauf dernier (temps)
                level_values = X.index.get_level_values(i)
                if level_values.isna().any():
                    level_name = X.index.names[i] or f"level_{i}"
                    issues.append(f"Missing values in panel identifier level: {level_name}")

        else:
            # Panel déclaré par colonnes : la validation a pu les faire passer dans
            # l'index, un identifiant se lit donc parmi les colonnes OU parmi les
            # niveaux d'index
            index_levels = [name for name in X.index.names if name is not None]
            missing_cols = set(self.panel_cols) - set(X.columns) - set(index_levels)
            # Vérification des colonnes manquantes
            if missing_cols:
                issues.append(f"Missing panel columns: {missing_cols}")
                return False, issues
            # Parcours des colonnes de panel
            for col in self.panel_cols:
                values = X[col] if col in X.columns else X.index.get_level_values(col)
                # Vérification des valeurs manquantes parmi les entités
                if values.isnull().any():
                    issues.append(f"Missing values in panel identifier: {col}")

        return len(issues) == 0, issues

# Mixin d'inversion des transformations
class ReversibleTransformerMixin(ABC):
    """Mixin for transformers that support inverse transformation.

    Declares the abstract ``inverse_transform`` (a subclass that does not
    implement it cannot be instantiated) and provides helpers to record the
    structure of the transformed data and to restore the original structure:
    ``Period`` dates of the index and columns moved into the index under
    ``convert_cols_to_index=True``.

    Examples:
        >>> import pandas as pd
        >>> class ReversibleScaler(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
        ...     def _fit(self, X, y=None):
        ...         self.mean_ = X.select_dtypes(include='number').mean()
        ...
        ...     def _transform(self, X):
        ...         X_transformed = X - self.mean_
        ...         self._store_transformation_info(X, X_transformed)
        ...         return X_transformed
        ...
        ...     def inverse_transform(self, X):
        ...         return self._restore_structure_if_converted(X + self.mean_)
        >>> df = pd.DataFrame({'value': [10.0, 20.0, 30.0]}, index=pd.date_range('2023-01-01', periods=3))
        >>> scaler = ReversibleScaler().fit(df)
        >>> scaler.transform(df)['value'].tolist()
        [-10.0, 0.0, 10.0]
        >>> scaler.inverse_transform(scaler.transform(df)).equals(df)
        True
    """

    # Méthode d'inversion de la transformation
    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reverse the transformation.

        Converts transformed data back to its original form using stored
        transformation parameters. This method must be implemented by subclasses.

        Args:
            X: Transformed data to reverse

        Returns:
            Data in original format before transformation

        Examples:
            >>> class NoInverse(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X): return X
            >>> NoInverse()
            Traceback (most recent call last):
                ...
            TypeError: Can't instantiate abstract class NoInverse without an implementation for abstract method 'inverse_transform'
        """

    # Méthode de stockage des méta-données (index, colonnes, dimensions) des jeux de données initial et transformé
    def _store_transformation_info(self, X: pd.DataFrame, X_transformed: pd.DataFrame) -> None:
        """Store information needed for inverse transformation.

        Saves metadata about the original and transformed data structures,
        including shapes, column names, and index information (only when the
        transformation changed the index).

        Args:
            X: Original data before transformation
            X_transformed: Data after transformation

        Examples:
            >>> import pandas as pd
            >>> class Doubler(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X):
            ...         X_transformed = X * 2
            ...         self._store_transformation_info(X, X_transformed)
            ...         return X_transformed
            ...     def inverse_transform(self, X): return X / 2
            >>> df = pd.DataFrame({'value': [1, 2, 3]}, index=pd.date_range('2023-01-01', periods=3))
            >>> transformer = Doubler().fit(df)
            >>> _ = transformer.transform(df)
            >>> transformer.original_shape_, transformer.transformed_columns_
            ((3, 1), ['value'])
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

    # Fonction de restauration de la structure originale des données
    def _restore_structure_if_converted(self, X: pd.DataFrame) -> pd.DataFrame:
        """Restore the original structure of the data seen at ``fit``.

        Gives back the ``Period`` dates of an index that held periods at ``fit``
        (validation reads them at their first instant), and the time and panel
        columns moved into the index under ``convert_cols_to_index=True``, at the
        positions seen at ``fit`` and with a fresh ``RangeIndex``: the result does
        not depend on the data transformed in between. Data already indexed
        (``MultiIndex`` panel, ``DatetimeIndex`` series) keeps its index. Before
        ``fit``, the data is returned as is.

        Args:
            X: Transformed data with modified index

        Returns:
            Data with original structure restored

        Examples:
            >>> import pandas as pd
            >>> class Identity(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X): return X
            ...     def inverse_transform(self, X): return self._restore_structure_if_converted(X)
            >>> df = pd.DataFrame({
            ...     'entity': ['A', 'A', 'B', 'B'],
            ...     'date': pd.date_range('2023-01-01', periods=2).tolist() * 2,
            ...     'value': [1, 2, 3, 4]
            ... })
            >>> transformer = Identity(time_col='date', panel_cols=['entity'], convert_cols_to_index=True)
            >>> X_transformed = transformer.fit_transform(df)
            >>> X_transformed.index.names
            FrozenList(['entity', 'date'])
            >>> transformer.inverse_transform(X_transformed).equals(df)
            True
        """
        # Périodes de l'index d'origine
        X = self._restore_periods(X)

        # Si données déjà MultiIndex, pas de conversion à restaurer
        if getattr(self, '_panel_in_index', False):
            # Structure préservée (MultiIndex in, MultiIndex out)
            return X

        # Restauration des colonnes passées dans l'index, si les métadonnées existent
        metadata = getattr(self, 'conversion_metadata_', None)
        if metadata is not None:
            return restore_original_structure(X, metadata)

        # Pas de conversion appliquée : retour inchangé
        return X

    # Fonction de restauration des périodes de l'index
    def _restore_periods(self, X: pd.DataFrame) -> pd.DataFrame:
        """Turn the dates of the index back into periods if ``fit`` saw periods.

        Args:
            X: Data whose index (or last level) holds the first instant of each period

        Returns:
            A copy with a ``PeriodIndex`` (or ``Period`` last level) of the fitted
            frequency, or ``X`` itself when there is nothing to restore.

        Examples:
            >>> import pandas as pd
            >>> class Identity(PanelTimeSeriesTransformer, ReversibleTransformerMixin):
            ...     def _fit(self, X, y=None): pass
            ...     def _transform(self, X): return X
            ...     def inverse_transform(self, X): return self._restore_structure_if_converted(X)
            >>> df = pd.DataFrame({'v': [1, 2]}, index=pd.period_range('2023-01', periods=2, freq='M'))
            >>> transformer = Identity().fit(df)
            >>> transformer.inverse_transform(transformer.transform(df)).index.equals(df.index)
            True
        """
        freq = getattr(self, '_time_period_freq', None)
        if freq is None:
            return X

        # Dernier niveau d'un panel
        if isinstance(X.index, pd.MultiIndex):
            dates = X.index.get_level_values(-1)
            if not isinstance(dates, pd.DatetimeIndex):
                return X
            arrays = [X.index.get_level_values(i) for i in range(X.index.nlevels - 1)]
            arrays.append(dates.to_period(freq))
            return X.set_axis(pd.MultiIndex.from_arrays(arrays, names=X.index.names))

        # Index simple
        if isinstance(X.index, pd.DatetimeIndex):
            return X.set_axis(X.index.to_period(freq))
        return X


# Classe d'inversion des transformations directes et inverse d'un transformer
class InverseTransformer(BaseEstimator, TransformerMixin):
    """Wrapper that inverts the transform and inverse_transform methods of a transformer.

    This class takes a fitted transformer and swaps its transform and
    inverse_transform methods. Like ``sklearn.frozen.FrozenEstimator``, it is a
    pre-fitted step: ``fit`` does nothing, it is fitted as soon as the wrapped
    transformer is, and ``clone`` returns the wrapper itself, so the fitted
    transformer survives ``Pipeline``, ``cross_validate`` or ``GridSearchCV``.
    A ``set_params`` on a clone therefore changes the original.

    Args:
        transformer: A fitted sklearn transformer with transform and inverse_transform methods.

    Attributes:
        transformer: The wrapped transformer instance.

    Examples:
        >>> import numpy as np
        >>> from sklearn.base import clone
        >>> from sklearn.preprocessing import StandardScaler
        >>> X = np.array([[1.0], [3.0], [5.0]])
        >>> scaler = StandardScaler().fit(X)
        >>> inverse_scaler = InverseTransformer(scaler)
        >>> inverse_scaler.transform(scaler.transform(X)).ravel().round(6).tolist()
        [1.0, 3.0, 5.0]
        >>> clone(inverse_scaler) is inverse_scaler
        True
    """
    # Initialisation
    def __init__(self, transformer):
        """Initialize the InverseTransformer.

        Args:
            transformer: A fitted sklearn transformer with transform and
                inverse_transform methods.
        """
        # Instanciation des attributs
        self.transformer = transformer

    # Clonage : l'étape pré-ajustée est conservée telle quelle
    def __sklearn_clone__(self) -> 'InverseTransformer':
        """Return the wrapper itself: cloning would drop the fitted transformer.

        Returns:
            self
        """
        return self

    # Prédicat d'ajustement consulté par "check_is_fitted"
    def __sklearn_is_fitted__(self) -> bool:
        """Report whether the wrapped transformer is fitted.

        Returns:
            True if ``check_is_fitted`` accepts the wrapped transformer.
        """
        try:
            check_is_fitted(self.transformer)
        except NotFittedError:
            return False
        return True

    # Méthode d'entraînement
    def fit(self, X, y=None):
        """Fit method (no-op since transformer is already fitted).

        Args:
            X: Input data (not used).
            y: Target values (not used).

        Returns:
            self: Returns self for method chaining.
        """
        # Pas d'entraînement nécessaire car le transformer est déjà entraîné
        return self

    # Méthode de transformation
    def transform(self, X):
        """Apply the inverse_transform method of the wrapped transformer.

        Args:
            X: Input data to transform.

        Returns:
            Transformed data using the inverse_transform of the wrapped transformer.

        Raises:
            NotFittedError: If the wrapped transformer is not fitted.
        """
        # Application de la transformation inverse du transformer original
        check_is_fitted(self)
        return self.transformer.inverse_transform(X)

    # Méthode de transformation inverse
    def inverse_transform(self, X):
        """Apply the transform method of the wrapped transformer.

        Args:
            X: Input data to inverse transform.

        Returns:
            Transformed data using the transform of the wrapped transformer.

        Raises:
            NotFittedError: If the wrapped transformer is not fitted.
        """
        # Application de la transformation du transformer original
        check_is_fitted(self)
        return self.transformer.transform(X)
