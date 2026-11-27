"""Contract battery shared by every third-party adapter of ``tsforecast.adapters``.

``AdapterCase`` describes one adapter configuration; the ``check_*`` functions
assert one rule of the common contract each (wrapped object never modified,
predictions in the row order of ``X``, series and panels, no leakage, gap
independence of row-wise regressors, explicit errors, datetime resolution,
equivalence to the third-party package used natively, ``cross_val_score``,
``GridSearchCV``, pipelines) and raise ``AssertionError`` on a violation. ``AdapterContractTests`` and ``AdapterWorkflowContractTests``
turn them into tests for a subclass that sets the ``case`` attribute.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from tests.support.datasets import build_adapter_panel, build_adapter_series
from tests.support.perturbations import (
    DATETIME_UNITS,
    datetime_units,
    shuffle_rows,
    with_datetime_resolution,
)
from tsforecast.adapters._base import TemporalLayout
from tsforecast.crossvals import PanelOutOfSampleSplit, TSOutOfSampleSplit
from tsforecast.utils.frequency import to_dateoffset
from tsforecast.xy import DropNaNTarget, XYPipeline

# Jeux de données de la batterie : série temporelle, panel équilibré, panel déséquilibré
DATA_KINDS: Tuple[str, ...] = ('series', 'panel', 'unbalanced_panel')
# Découpage des contrôles hors validation croisée : dates de test et écart avant le test
N_TEST_DATES = 3
GAP_DATES = 2
# Validation croisée de la batterie
CV_PARAMS: Mapping[str, int] = {'n_splits': 3, 'test_size': 2, 'gap': 2}
# Décalage des dates de test du contrôle d'indépendance à l'écart, en pas de grille
GAP_SHIFT_STEPS = 5
# Altération de la cible future du contrôle de fuite
LEAK_OFFSET = 1000.0
# Écarts entre entraînement et test du contrôle d'équivalence au paquet natif
NATIVE_GAPS: Tuple[int, ...] = (0, GAP_DATES)

# Signature d'une référence native : (X_train, y_train, X_test, horizon relatif des dates de test)
NativeReference = Callable[[pd.DataFrame, Any, pd.DataFrame, Tuple[int, ...]], np.ndarray]


def default_data(kind: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return the ``(X, y)`` dataset of a kind of :data:`DATA_KINDS`.

    Args:
        kind: ``'series'``, ``'panel'`` or ``'unbalanced_panel'``.

    Returns:
        Origin-indexed features and target of the adapter datasets.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind == 'series':
        return build_adapter_series()
    if kind == 'panel':
        return build_adapter_panel(balanced=True)
    if kind == 'unbalanced_panel':
        return build_adapter_panel(balanced=False)
    raise ValueError(f'Unknown data kind {kind!r}; expected one of {DATA_KINDS}')


def default_splitter(kind: str) -> Any:
    """Return the out-of-sample splitter of a dataset kind, with a gap.

    Args:
        kind: ``'series'`` (``TSOutOfSampleSplit``) or a panel kind
            (``PanelOutOfSampleSplit``).

    Returns:
        A splitter with the parameters :data:`CV_PARAMS`.
    """
    if kind == 'series':
        return TSOutOfSampleSplit(**CV_PARAMS)
    return PanelOutOfSampleSplit(**CV_PARAMS)


@dataclass(frozen=True)
class AdapterCase:
    """One adapter configuration checked by the contract battery.

    Attributes:
        make_estimator: Builds a new, unfitted adapter.
        wrapped_param: Constructor parameter holding the third-party object
            (``'forecaster'``, ``'model'``...), which ``fit`` must never modify.
        param_grid: Grid of ``GridSearchCV``, with at least one nested
            parameter (``'<wrapped_param>__<param>'``).
        package: Module whose absence must turn ``fit`` into an
            ``ImportError`` naming ``extra``, or None.
        extra: Extra named by that ``ImportError``.
        supports_panel: Whether the adapter accepts panel data.
        local_model: Whether the model is fitted per entity, so that an entity
            unknown at fit raises ``ValueError`` at predict.
        row_regressor: Whether the prediction of a row only depends on the
            features of that row (it must then not depend on the gap between
            training and test).
        invalid_params: Parameter sets with which ``fit`` raises ``ValueError``.
        handles_training_holes: Whether ``fit`` accepts training rows that
            leave grid dates of an entity uncovered (rows dropped by
            ``DropNaNTarget``); otherwise it must raise ``ValueError``.
        make_data: Builds ``(X, y)`` for a kind of :data:`DATA_KINDS`.
        make_splitter: Builds the splitter of ``cross_val_score`` and
            ``GridSearchCV`` for a kind of :data:`DATA_KINDS` (a panel splitter
            for data whose time series kind still has a ``MultiIndex``).
        frame_method: Name of a method returning a pandas object with dates,
            whose units must be those of the input, or None.
        native_reference: The same model fitted and predicted through the
            third-party API alone, without ``tsforecast``:
            ``(X_train, y_train, X_test, fh) -> ndarray`` in the row order of
            ``X_test``, ``fh`` being the relative horizon of the test dates
            (``(gap + 1, ..., gap + n_test)``, common to every entity of a
            balanced panel); None when the case has no native equivalent.
        native_gaps: Gaps between training and test that the third-party API
            handles natively (``(0,)`` for a model reading ``X`` on every date
            up to the forecast).
        rtol: Relative tolerance of the comparisons of predictions.
        atol: Absolute tolerance of the comparisons of predictions.
    """

    make_estimator: Callable[[], BaseEstimator]
    wrapped_param: str
    param_grid: Mapping[str, Sequence[Any]]
    package: Optional[str] = None
    extra: Optional[str] = None
    supports_panel: bool = True
    local_model: bool = False
    row_regressor: bool = True
    invalid_params: Tuple[Mapping[str, Any], ...] = ()
    handles_training_holes: bool = True
    make_data: Callable[[str], Tuple[pd.DataFrame, Any]] = field(default=default_data)
    make_splitter: Callable[[str], Any] = field(default=default_splitter)
    frame_method: Optional[str] = None
    native_reference: Optional[NativeReference] = None
    native_gaps: Tuple[int, ...] = NATIVE_GAPS
    rtol: float = 1e-7
    atol: float = 1e-9


# ---------------------------------------------------------------------------
# Outils de la batterie
# ---------------------------------------------------------------------------


def _dates(X: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the date of each row of ``X``."""
    return pd.DatetimeIndex(X.index.get_level_values(-1))


def split_by_date(
    X: pd.DataFrame, y: Any, n_test: int = N_TEST_DATES, gap: int = 0
) -> Tuple[pd.DataFrame, Any, pd.DataFrame, Any]:
    """Split on the dates: the last ``n_test`` dates for test, ``gap`` dates left out before.

    Args:
        X: Features of a series or a panel.
        y: Target indexed like ``X``.
        n_test: Number of test dates.
        gap: Number of dates between the last training date and the first test date.

    Returns:
        ``(X_train, y_train, X_test, y_test)``.
    """
    dates = _dates(X)
    calendar = dates.unique().sort_values()
    train = np.asarray(dates <= calendar[-(n_test + gap) - 1])
    test = np.asarray(dates >= calendar[-n_test])
    return X[train], y[train], X[test], y[test]


def shift_dates(X: pd.DataFrame, steps: int, freq: str) -> pd.DataFrame:
    """Return a copy of ``X`` whose dates are moved ``steps`` grid steps later."""
    shifted = _dates(X) + steps * to_dateoffset(freq)
    out = X.copy()
    if isinstance(X.index, pd.MultiIndex):
        arrays = [X.index.get_level_values(level) for level in range(X.index.nlevels - 1)] + [shifted]
        out.index = pd.MultiIndex.from_arrays(arrays, names=X.index.names)
    else:
        out.index = pd.DatetimeIndex(shifted, name=X.index.name)
    return out


def _assert_predictions(actual: Any, expected: np.ndarray, case: AdapterCase, message: str) -> None:
    """Assert that predictions equal the expected ones within the case tolerances."""
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=case.rtol, atol=case.atol, err_msg=message)


def _param_fingerprint(estimator: BaseEstimator) -> str:
    """Hash of the deep parameters, each nested object reduced to its class."""
    params = estimator.get_params(deep=True)
    return joblib.hash({key: type(value) if hasattr(value, 'get_params') else value for key, value in params.items()})


def _assert_output_shape(predictions: Any, X: pd.DataFrame, y: Any) -> None:
    """Assert the sklearn output contract: a finite ndarray with one row per row of ``X``."""
    assert isinstance(predictions, np.ndarray), (
        f'predict must return a numpy.ndarray, got {type(predictions).__name__}'
    )
    expected = (len(X),) if np.ndim(y) == 1 else (len(X), np.shape(y)[1])
    assert predictions.shape == expected, f'predict returned shape {predictions.shape}, expected {expected}'
    assert np.isfinite(predictions).all(), 'predict returned non-finite values'


# ---------------------------------------------------------------------------
# Contrôles unitaires du contrat
# ---------------------------------------------------------------------------


def check_wrapped_object_untouched(case: AdapterCase) -> None:
    """Fit works on a copy: the wrapped object and the parameters are never modified.

    Also checks that the adapter is not fitted before ``fit``, that a clone
    fits without touching its own wrapped object, and that
    ``set_params(**get_params())`` leaves the parameters unchanged.
    """
    X, y = case.make_data('series')
    estimator = case.make_estimator()
    try:
        check_is_fitted(estimator)
    except NotFittedError:
        pass
    else:
        raise AssertionError('The adapter looks fitted before fit (fitted attribute set in __init__)')

    params = estimator.get_params(deep=False)
    wrapped = params[case.wrapped_param]
    fingerprint = joblib.hash(wrapped)
    estimator.fit(X, y)
    assert joblib.hash(wrapped) == fingerprint, f'fit modified the wrapped object {case.wrapped_param!r}'
    after = estimator.get_params(deep=False)
    assert after.keys() == params.keys() and all(after[key] is params[key] for key in params), (
        'fit replaced a constructor parameter'
    )

    # Clone d'un adapter ajusté : nouvel objet tiers, lui aussi intact après fit
    cloned = clone(estimator)
    cloned_wrapped = cloned.get_params(deep=False)[case.wrapped_param]
    cloned_fingerprint = joblib.hash(cloned_wrapped)
    cloned.fit(X, y)
    assert joblib.hash(cloned_wrapped) == cloned_fingerprint, 'fit modified the wrapped object of a clone'

    # Aller-retour des paramètres, comparés par leurs valeurs : un objet imbriqué est décrit
    # par sa classe, ses propres paramètres figurant dans get_params(deep=True) ; son état
    # privé (caches remplis par set_params, comme la configuration des objets sktime) n'est
    # pas un paramètre
    fresh = case.make_estimator()
    before = _param_fingerprint(fresh)
    fresh.set_params(**fresh.get_params(deep=True))
    assert _param_fingerprint(fresh) == before, 'set_params(**get_params()) changed the parameters'


def check_fit_predict(case: AdapterCase, kind: str) -> None:
    """Fit and predict on a dataset kind: finite ndarray with one row per row of ``X``."""
    X, y = case.make_data(kind)
    X_train, y_train, X_test, _ = split_by_date(X, y)
    estimator = case.make_estimator().fit(X_train, y_train)
    _assert_output_shape(estimator.predict(X_test), X_test, y_train)


def check_predict_row_order(case: AdapterCase, kind: str) -> None:
    """Predictions follow the rows of ``X``, whatever the order of the training and test rows."""
    X, y = case.make_data(kind)
    X_train, y_train, X_test, _ = split_by_date(X, y)
    reference = case.make_estimator().fit(X_train, y_train).predict(X_test)

    # Lignes de test mélangées : prédictions permutées à l'identique
    estimator = case.make_estimator().fit(X_train, y_train)
    shuffled = shuffle_rows(X_test, seed=1)
    assert not shuffled.index.equals(X_test.index), 'The shuffle left the rows in place'
    expected = reference[X_test.index.get_indexer(shuffled.index)]
    _assert_predictions(estimator.predict(shuffled), expected, case, 'predict does not follow the rows of X')

    # Lignes d'entraînement mélangées : même modèle
    train_order = shuffle_rows(X_train, seed=2).index
    refit = case.make_estimator().fit(X_train.loc[train_order], y_train.loc[train_order])
    _assert_predictions(refit.predict(X_test), reference, case, 'The fit depends on the order of the training rows')


def check_no_leakage(case: AdapterCase, kind: str) -> None:
    """A fit never sees the target after its training rows, even through a previous fit.

    The same instance, and a clone of it, first fitted on the full data with
    an altered future target, then fitted on the training rows, predict like a
    new adapter fitted on the training rows.
    """
    X, y = case.make_data(kind)
    X_train, y_train, X_test, _ = split_by_date(X, y, gap=GAP_DATES)
    reference = case.make_estimator().fit(X_train, y_train).predict(X_test)

    # Cible altérée après la dernière date d'entraînement
    future = np.asarray(_dates(X) > _dates(X_train).max())
    condition = future if np.ndim(y) == 1 else np.broadcast_to(future[:, None], np.shape(y))
    y_altered = y.mask(condition, y + LEAK_OFFSET)

    estimator = case.make_estimator().fit(X, y_altered)
    cloned = clone(estimator)
    estimator.fit(X_train, y_train)
    _assert_predictions(estimator.predict(X_test), reference, case, 'A previous fit leaks into the next one')
    cloned.fit(X_train, y_train)
    _assert_predictions(cloned.predict(X_test), reference, case, 'A clone carries the state of a previous fit')


def check_gap_independence(case: AdapterCase, kind: str) -> None:
    """A row-wise regressor predicts a row the same way, however far it is from the training data."""
    X, y = case.make_data(kind)
    X_train, y_train, X_test, _ = split_by_date(X, y)
    freq = TemporalLayout.from_index(X.index).freq
    estimator = case.make_estimator().fit(X_train, y_train)
    reference = estimator.predict(X_test)
    far = shift_dates(X_test, GAP_SHIFT_STEPS, freq)
    _assert_predictions(estimator.predict(far), reference, case, 'The prediction depends on the gap')


def check_unknown_entity(case: AdapterCase) -> None:
    """A local model refuses to predict an entity unknown at fit with a ``ValueError``."""
    X, y = case.make_data('panel')
    X_train, y_train, X_test, _ = split_by_date(X, y)
    known = np.asarray(X_train.index.get_level_values(0) != X_train.index.get_level_values(0)[-1])
    estimator = case.make_estimator().fit(X_train[known], y_train[known])
    try:
        estimator.predict(X_test)
    except ValueError:
        return
    raise AssertionError('predict accepted an entity unknown at fit without a ValueError')


def check_missing_package(case: AdapterCase, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without its third-party package, ``fit`` raises an ``ImportError`` naming the extra."""
    X, y = case.make_data('series')
    estimator = case.make_estimator()
    # Module et sous-modules masqués : leur import échoue
    for name in [name for name in sys.modules if name.startswith(case.package + '.')]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, case.package, None)
    try:
        estimator.fit(X, y)
    except ImportError as error:
        assert case.extra in str(error), f'The ImportError does not name the extra {case.extra!r}: {error}'
        return
    raise AssertionError(f'fit succeeded without the package {case.package!r}')


def check_invalid_params(case: AdapterCase) -> None:
    """Each invalid parameter set makes ``fit`` raise a ``ValueError``."""
    X, y = case.make_data('series')
    for params in case.invalid_params:
        estimator = case.make_estimator().set_params(**params)
        try:
            estimator.fit(X, y)
        except ValueError:
            continue
        raise AssertionError(f'fit accepted the invalid parameters {dict(params)!r}')


def check_datetime_resolution(case: AdapterCase, kind: str) -> None:
    """Predictions do not depend on the unit of the dates; returned dates keep the input unit."""
    X, y = case.make_data(kind)
    reference = None
    for unit in DATETIME_UNITS:
        X_unit = with_datetime_resolution(X, unit)
        y_unit = with_datetime_resolution(y, unit)
        X_train, y_train, X_test, _ = split_by_date(X_unit, y_unit)
        estimator = case.make_estimator().fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        if reference is None:
            reference = predictions
        else:
            _assert_predictions(predictions, reference, case, f'Predictions change with dates in {unit!r}')
        if case.frame_method is not None:
            units = datetime_units(getattr(estimator, case.frame_method)(X_test))
            assert units and set(units.values()) == {unit}, f'Dates returned in {units}, input in {unit!r}'


def check_native_equivalence(case: AdapterCase, kind: str, gap: int) -> None:
    """The adapter predicts like the same model used through its package alone.

    On a series or a balanced panel, with ``gap`` dates left out before the
    :data:`N_TEST_DATES` test dates, the predictions of the adapter equal those
    of ``case.native_reference``, which receives the relative horizon of the
    test dates from the split itself.
    """
    X, y = case.make_data(kind)
    X_train, y_train, X_test, _ = split_by_date(X, y, n_test=N_TEST_DATES, gap=gap)
    # Horizon relatif des dates de test, compté depuis la dernière date d'entraînement
    fh = tuple(range(gap + 1, gap + N_TEST_DATES + 1))
    expected = np.asarray(case.native_reference(X_train, y_train, X_test, fh), dtype=float)
    predictions = case.make_estimator().fit(X_train, y_train).predict(X_test)
    _assert_predictions(
        predictions, expected, case, f'The predictions differ from the package used natively (kind={kind!r}, gap={gap})'
    )


# ---------------------------------------------------------------------------
# Contrôles d'intégration du contrat
# ---------------------------------------------------------------------------


def check_cross_val_score(case: AdapterCase, kind: str) -> None:
    """``cross_val_score`` with a ``tsforecast`` splitter and a gap gives finite scores."""
    X, y = case.make_data(kind)
    scores = cross_val_score(case.make_estimator(), X, y, cv=case.make_splitter(kind), error_score='raise')
    assert len(scores) == CV_PARAMS['n_splits'], f'{len(scores)} scores for {CV_PARAMS["n_splits"]} folds'
    assert np.isfinite(scores).all(), f'Non-finite scores: {scores}'


def check_grid_search(case: AdapterCase) -> None:
    """``GridSearchCV`` explores the (nested) grid of the case and refits the best adapter.

    The search runs on the training dates and the refitted adapter predicts the
    later test dates, as in a forecasting workflow.
    """
    X, y = case.make_data('series')
    X_train, y_train, X_test, _ = split_by_date(X, y)
    search = GridSearchCV(
        case.make_estimator(), case.param_grid, cv=case.make_splitter('series'), error_score='raise'
    ).fit(X_train, y_train)
    assert set(search.best_params_) == set(case.param_grid), f'best_params_ = {search.best_params_}'
    _assert_output_shape(search.predict(X_test), X_test, y)


def check_xy_pipeline(case: AdapterCase, kind: str) -> None:
    """In an ``XYPipeline`` after ``DropNaNTarget``, the adapter fits on the rows with a target.

    An adapter refusing training rows with holes must raise ``ValueError`` instead.
    """
    X, y = case.make_data(kind)
    X_train, y_train, X_test, _ = split_by_date(X, y)
    # Cible manquante sur une ligne sur cinq de l'entraînement
    missing = np.arange(len(y_train)) % 5 == 1
    condition = missing if np.ndim(y_train) == 1 else np.broadcast_to(missing[:, None], np.shape(y_train))
    y_holes = y_train.mask(condition)

    pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', case.make_estimator())])
    if not case.handles_training_holes:
        try:
            pipeline.fit(X_train, y_holes)
        except ValueError:
            return
        raise AssertionError('fit accepted training rows with holes without a ValueError')
    predictions = pipeline.fit(X_train, y_holes).predict(X_test)
    expected = case.make_estimator().fit(X_train[~missing], y_train[~missing]).predict(X_test)
    _assert_predictions(predictions, expected, case, 'XYPipeline does not reproduce a fit on the kept rows')


def check_sklearn_pipeline(case: AdapterCase, kind: str) -> None:
    """In a ``sklearn.pipeline.Pipeline`` after a pandas-output step, the adapter sees the index."""
    X, y = case.make_data(kind)
    X_train, y_train, X_test, _ = split_by_date(X, y)
    pipeline = Pipeline([
        ('scale', StandardScaler().set_output(transform='pandas')),
        ('model', case.make_estimator()),
    ])
    predictions = pipeline.fit(X_train, y_train).predict(X_test)
    scaler = StandardScaler().set_output(transform='pandas').fit(X_train)
    expected = case.make_estimator().fit(scaler.transform(X_train), y_train).predict(scaler.transform(X_test))
    _assert_predictions(predictions, expected, case, 'Pipeline does not reproduce the manual steps')


# ---------------------------------------------------------------------------
# Mixins de tests
# ---------------------------------------------------------------------------


class _DataKindFixture:
    """Fixture ``data_kind`` over :data:`DATA_KINDS`, panels skipped when unsupported."""

    case: AdapterCase

    @pytest.fixture(params=DATA_KINDS)
    def data_kind(self, request: pytest.FixtureRequest) -> str:
        """Dataset kind of the test."""
        if request.param != 'series' and not self.case.supports_panel:
            pytest.skip('The adapter does not support panel data')
        return request.param


class AdapterContractTests(_DataKindFixture):
    """Unit rules of the adapter contract; subclasses set ``case``."""

    def test_wrapped_object_untouched(self) -> None:
        """Fit never modifies the wrapped object nor the parameters."""
        check_wrapped_object_untouched(self.case)

    def test_fit_predict(self, data_kind: str) -> None:
        """Series and panels give an ndarray with one row per row of X."""
        check_fit_predict(self.case, data_kind)

    def test_predict_row_order(self, data_kind: str) -> None:
        """Predictions follow the rows of X."""
        check_predict_row_order(self.case, data_kind)

    def test_no_leakage(self, data_kind: str) -> None:
        """No previous fit leaks the future target."""
        check_no_leakage(self.case, data_kind)

    def test_gap_independence(self, data_kind: str) -> None:
        """A row-wise regressor does not depend on the gap."""
        if not self.case.row_regressor:
            pytest.skip('The adapter is not a row-wise regressor')
        check_gap_independence(self.case, data_kind)

    def test_unknown_entity(self) -> None:
        """A local model refuses an entity unknown at fit."""
        if not (self.case.local_model and self.case.supports_panel):
            pytest.skip('The adapter is not a local panel model')
        check_unknown_entity(self.case)

    def test_missing_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without its package, fit names the extra to install."""
        if self.case.package is None:
            pytest.skip('The adapter has no optional package')
        check_missing_package(self.case, monkeypatch)

    def test_invalid_params(self) -> None:
        """Invalid parameters are refused at fit."""
        if not self.case.invalid_params:
            pytest.skip('The case declares no invalid parameters')
        check_invalid_params(self.case)

    def test_datetime_resolution(self, data_kind: str) -> None:
        """The unit of the dates changes nothing."""
        check_datetime_resolution(self.case, data_kind)

    @pytest.mark.parametrize('gap', NATIVE_GAPS)
    def test_native_equivalence(self, data_kind: str, gap: int) -> None:
        """The adapter predicts like its package used natively."""
        if self.case.native_reference is None:
            pytest.skip('The case has no native reference')
        if data_kind == 'unbalanced_panel':
            pytest.skip('An unbalanced panel has no common native horizon')
        if gap not in self.case.native_gaps:
            pytest.skip(f'The package cannot forecast through a gap of {gap} dates natively')
        check_native_equivalence(self.case, data_kind, gap)


class AdapterWorkflowContractTests(_DataKindFixture):
    """Integration rules of the adapter contract (sklearn workflows); subclasses set ``case``."""

    def test_cross_val_score(self, data_kind: str) -> None:
        """cross_val_score with a tsforecast splitter and a gap."""
        check_cross_val_score(self.case, data_kind)

    def test_grid_search(self) -> None:
        """GridSearchCV over a nested parameter."""
        check_grid_search(self.case)

    def test_xy_pipeline(self, data_kind: str) -> None:
        """XYPipeline with DropNaNTarget."""
        check_xy_pipeline(self.case, data_kind)

    def test_sklearn_pipeline(self, data_kind: str) -> None:
        """sklearn Pipeline with a pandas-output step."""
        check_sklearn_pipeline(self.case, data_kind)
