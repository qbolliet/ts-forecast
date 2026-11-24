"""Unit tests of ``tsforecast.xy.transformers``.

Covers ``DropNaNTarget`` — rows dropped by the training pass (``fit_transform``)
and by ``transform(X, y)`` for ``Series``, ``DataFrame`` (multi-output) and
array targets, the prediction pass without ``y`` (no row lost), edge cases (no
NaN, all NaN, empty input, NaN in ``X``, panel ``MultiIndex``, unsorted rows,
length mismatch), non-mutation of the inputs, the sklearn contract of a
step without learnt parameter (input columns recorded, output names,
``set_output``) and its place in an ``XYPipeline`` —; the contract of
``XYTransformerMixin`` seen from a minimal subclass (``fit_transform(X, y)``
gives ``(X', y')``, unimplemented methods, ``get_params`` / ``clone``, fitted
state seen by ``check_is_fitted`` and by ``XYPipeline``); and ``XYPanelTimeSeriesTransformer`` (abstract methods, validation of
``X``, fitted-state checks, ``y`` propagation); and ``is_xy_transformer``
(declaration by inheritance, not by signature).
"""
# Modules de base
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# Objets testés
from tsforecast.xy import DropNaNTarget, XYPipeline, XYTransformerMixin, is_xy_transformer
from tsforecast.xy.transformers import XYPanelTimeSeriesTransformer
from tests.support.estimators import MixinScaleXY, SpyEstimator


@pytest.fixture
def frame_xy():
    """``(X, y)``: 5 dated rows, target NaN on the 2nd and the 5th."""
    dates = pd.date_range('2024-01-01', periods=5, freq='MS', name='date')
    X = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0], 'z': [0.0, 1.0, 0.0, 1.0, 0.0]}, index=dates)
    y = pd.Series([10.0, np.nan, 30.0, 40.0, np.nan], index=dates, name='y')
    return X, y


# =============================================================================
# DropNaNTarget : lignes retirées
# =============================================================================
class TestRowsDropped:
    """The rows whose target is missing leave ``X`` and ``y`` together."""

    def test_series_target_drops_its_nan_rows(self, frame_xy):
        """Valeur d'or : lignes 1, 3, 4 conservées (cible 10, 30, 40)."""
        X_kept, y_kept = DropNaNTarget().fit_transform(*frame_xy)

        assert (X_kept['x'].tolist(), y_kept.tolist()) == ([1.0, 3.0, 4.0], [10.0, 30.0, 40.0])

    def test_kept_rows_keep_their_index(self, frame_xy):
        """The index of the kept rows is the input index minus the dropped dates."""
        X, y = frame_xy

        X_kept, _y_kept = DropNaNTarget().fit_transform(X, y)

        assert X_kept.index.equals(X.index[[0, 2, 3]])

    def test_x_and_y_share_their_index(self, frame_xy):
        """``X`` and ``y`` come out with the same index, row for row."""
        X_kept, y_kept = DropNaNTarget().fit_transform(*frame_xy)

        assert y_kept.index.equals(X_kept.index)

    def test_transform_with_target_drops_too(self, frame_xy):
        """``transform(X, y)`` — the ``score`` path — drops like ``fit_transform``."""
        X, y = frame_xy
        step = DropNaNTarget().fit(X, y)

        X_kept, _y_kept = step.transform(X, y)

        assert X_kept['x'].tolist() == [1.0, 3.0, 4.0]

    def test_multi_output_target_drops_a_row_with_any_nan(self, frame_xy):
        """A row goes as soon as one of its targets is missing."""
        X, y = frame_xy
        targets = pd.DataFrame({'a': y, 'b': [1.0, 2.0, np.nan, 4.0, 5.0]}, index=X.index)

        X_kept, _y_kept = DropNaNTarget().fit_transform(X, targets)

        # Valeur d'or : NaN de a aux lignes 2 et 5, NaN de b à la ligne 3 → lignes 1 et 4
        assert X_kept['x'].tolist() == [1.0, 4.0]

    def test_multi_output_target_stays_a_dataframe(self, frame_xy):
        """A ``DataFrame`` target comes out as a ``DataFrame`` with its columns."""
        X, y = frame_xy
        targets = pd.DataFrame({'a': y, 'b': y * 2}, index=X.index)

        _X_kept, y_kept = DropNaNTarget().fit_transform(X, targets)

        assert list(y_kept.columns) == ['a', 'b']

    def test_nan_on_a_single_target_column_drops_the_row(self, frame_xy):
        """One NaN in one of the target columns is enough, the other being complete.

        The row goes even though its other target is observed: the estimator would
        have to learn from a partial target otherwise.
        """
        X, _y = frame_xy
        targets = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0], 'b': [1.0, np.nan, 3.0, 4.0, 5.0]}, index=X.index)

        _X_kept, y_kept = DropNaNTarget().fit_transform(X, targets)

        assert y_kept['a'].tolist() == [1.0, 3.0, 4.0, 5.0]

    def test_arrays_are_supported(self, frame_xy):
        """NumPy inputs are filtered by position."""
        X, y = frame_xy

        _X_kept, y_kept = DropNaNTarget().fit_transform(X.to_numpy(), y.to_numpy())

        assert y_kept.tolist() == [10.0, 30.0, 40.0]

    def test_arrays_stay_arrays(self, frame_xy):
        """NumPy inputs come out as arrays."""
        X, y = frame_xy

        X_kept, _y_kept = DropNaNTarget().fit_transform(X.to_numpy(), y.to_numpy())

        assert isinstance(X_kept, np.ndarray)

    def test_two_dimensional_array_target(self, frame_xy):
        """A 2-D array target drops a row as soon as one of its columns is NaN."""
        X, _y = frame_xy
        targets = np.array([[1.0, 1.0], [np.nan, 1.0], [1.0, 1.0], [1.0, np.nan], [1.0, 1.0]])

        X_kept, _y_kept = DropNaNTarget().fit_transform(X.to_numpy(), targets)

        # Valeur d'or : NaN aux lignes 2 et 4 → x = 1, 3, 5
        assert X_kept[:, 0].tolist() == [1.0, 3.0, 5.0]

    def test_rows_are_matched_by_position(self, frame_xy):
        """``X`` and ``y`` with different index labels are still filtered row by row."""
        X, y = frame_xy

        _X_kept, y_kept = DropNaNTarget().fit_transform(X, y.reset_index(drop=True))

        assert y_kept.index.tolist() == [0, 2, 3]

    def test_nan_in_x_is_kept(self, frame_xy):
        """Only the target decides: a NaN feature on a row with a target stays."""
        X, y = frame_xy
        X = X.assign(x=[np.nan, 2.0, 3.0, 4.0, 5.0])

        X_kept, _y_kept = DropNaNTarget().fit_transform(X, y)

        assert X_kept['x'].isna().tolist() == [True, False, False]

    def test_panel_multiindex(self):
        """On a panel, each (entity, date) row is kept or dropped on its own target."""
        dates = pd.date_range('2024-01-31', periods=3, freq='ME')
        index = pd.MultiIndex.from_product([['FR', 'DE'], dates], names=['country', 'date'])
        X = pd.DataFrame({'a': np.arange(6.0)}, index=index)
        y = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0, 6.0], index=index)

        X_kept, _y_kept = DropNaNTarget().fit_transform(X, y)

        # Valeur d'or : FR perd février, DE perd janvier
        assert X_kept.index.tolist() == [('FR', dates[0]), ('FR', dates[2]), ('DE', dates[1]), ('DE', dates[2])]

    def test_order_of_unsorted_rows_is_kept(self, frame_xy):
        """Rows come out in their input order, without sorting."""
        X, y = frame_xy
        order = [3, 0, 4, 2, 1]

        X_kept, _y_kept = DropNaNTarget().fit_transform(X.iloc[order], y.iloc[order])

        # Valeur d'or : ordre d'entrée 4, 1, 5, 3, 2 privé des lignes 5 et 2
        assert X_kept['x'].tolist() == [4.0, 1.0, 3.0]

    def test_inputs_are_not_mutated(self, frame_xy):
        """``X`` and ``y`` are left as they were."""
        X, y = frame_xy
        X_before, y_before = X.copy(deep=True), y.copy(deep=True)

        DropNaNTarget().fit_transform(X, y)

        pd.testing.assert_frame_equal(X, X_before)
        pd.testing.assert_series_equal(y, y_before)


# =============================================================================
# DropNaNTarget : passe de prédiction et cas limites
# =============================================================================
class TestEdgeCases:
    """No ``y``, nothing to drop, everything to drop, inconsistent lengths."""

    def test_without_target_x_goes_through(self, frame_xy):
        """``transform(X)`` — the ``predict`` path — returns ``X`` untouched."""
        X, y = frame_xy
        step = DropNaNTarget().fit(X, y)

        assert step.transform(X) is X

    def test_complete_target_keeps_x(self, frame_xy):
        """A target without NaN leaves every row of ``X``."""
        X, y = frame_xy

        X_kept, _y_kept = DropNaNTarget().fit_transform(X, y.fillna(0.0))

        assert X_kept.equals(X)

    def test_complete_target_keeps_y(self, frame_xy):
        """A target without NaN comes out equal to itself."""
        X, y = frame_xy

        _X_kept, y_kept = DropNaNTarget().fit_transform(X, y.fillna(0.0))

        assert y_kept.equals(y.fillna(0.0))

    def test_all_nan_target_drops_everything(self, frame_xy):
        """A target entirely NaN leaves two empty outputs."""
        X, y = frame_xy

        X_kept, y_kept = DropNaNTarget().fit_transform(X, y * np.nan)

        assert (len(X_kept), len(y_kept)) == (0, 0)

    def test_all_nan_target_keeps_the_structure(self, frame_xy):
        """The empty ``X`` keeps its columns and dtypes, the empty ``y`` its name.

        Behaviour pinned: the step does not refuse an empty result; the estimator
        downstream is the one to raise.
        """
        X, y = frame_xy

        X_kept, y_kept = DropNaNTarget().fit_transform(X, y * np.nan)

        assert (X_kept.dtypes.equals(X.dtypes), y_kept.name) == (True, 'y')

    def test_empty_input_stays_empty(self, frame_xy):
        """Zero rows in, zero rows out."""
        X, y = frame_xy

        X_kept, _y_kept = DropNaNTarget().fit_transform(X.iloc[:0], y.iloc[:0])

        assert X_kept.empty

    def test_length_mismatch_is_refused(self, frame_xy):
        """``X`` and ``y`` of different lengths raise, naming both lengths."""
        X, y = frame_xy

        with pytest.raises(ValueError, match='same length, got 5 and 4'):
            DropNaNTarget().fit_transform(X, y.iloc[:4])


# =============================================================================
# DropNaNTarget : contrat sklearn d'une étape sans paramètre appris
# =============================================================================
class TestSklearnContract:
    """Step without learnt parameter: fitted from the start, no parameter, clonable, input columns recorded."""

    def test_reported_fitted_without_fit(self):
        """``check_is_fitted`` passes on a fresh instance: there is nothing to learn."""
        check_is_fitted(DropNaNTarget())

    def test_has_no_parameter(self):
        """``get_params`` is empty."""
        assert DropNaNTarget().get_params() == {}

    def test_clone_gives_a_new_step(self):
        """``clone`` returns another instance of the class."""
        step = DropNaNTarget()

        assert clone(step) is not step

    def test_clone_keeps_the_class(self):
        """The clone is a ``DropNaNTarget``."""
        assert type(clone(DropNaNTarget())) is DropNaNTarget

    def test_records_the_input_columns(self, frame_xy):
        """``fit`` records the width and the names of the columns of ``X``."""
        step = DropNaNTarget().fit(*frame_xy)

        assert (step.n_features_in_, list(step.feature_names_in_)) == (2, ['x', 'z'])

    def test_feature_names_out_are_the_input_names(self, frame_xy):
        """The rows are filtered, the columns kept: the output names are the input ones."""
        assert list(DropNaNTarget().fit(*frame_xy).get_feature_names_out()) == ['x', 'z']

    def test_feature_names_out_of_an_array_follow_input_features(self, frame_xy):
        """Fitted on an array, the output names are the ``input_features`` given."""
        X, y = frame_xy
        step = DropNaNTarget().fit(X.to_numpy(), y.to_numpy())

        assert list(step.get_feature_names_out(['u', 'v'])) == ['u', 'v']

    def test_series_features_record_no_width(self, frame_xy):
        """A one-dimensional ``X`` has no columns to record."""
        X, y = frame_xy

        assert not hasattr(DropNaNTarget().fit(X['x'], y), 'n_features_in_')

    def test_feature_names_out_need_a_fit(self):
        """Before ``fit``, the output names are unknown."""
        with pytest.raises(NotFittedError):
            DropNaNTarget().get_feature_names_out()

    def test_set_output_keeps_the_input_types(self, frame_xy):
        """``set_output(transform='default')`` is accepted: a ``DataFrame`` input stays a ``DataFrame``."""
        X, y = frame_xy

        X_kept, _y_kept = DropNaNTarget().set_output(transform='default').fit_transform(X, y)

        assert isinstance(X_kept, pd.DataFrame)

    def test_set_output_keeps_the_kept_labels(self, frame_xy):
        """With ``set_output(transform='pandas')``, the kept rows keep their dates."""
        X, y = frame_xy

        X_kept, _y_kept = DropNaNTarget().set_output(transform='pandas').fit_transform(X, y)

        assert X_kept.index.equals(X.index[[0, 2, 3]])


# =============================================================================
# DropNaNTarget dans un XYPipeline
# =============================================================================
class TestInXYPipeline:
    """Placed before the estimator: rows dropped at ``fit`` and ``score``, kept at ``predict``."""

    @pytest.fixture
    def pipeline(self):
        """``DropNaNTarget -> LinearRegression``, which refuses a NaN target."""
        return XYPipeline([('drop', DropNaNTarget()), ('model', LinearRegression())])

    def test_fit_succeeds_despite_nan_targets(self, pipeline, frame_xy):
        """The estimator is fitted on the 3 rows with a target."""
        pipeline.fit(*frame_xy)

        assert pipeline.named_steps['model'].n_features_in_ == 2

    def test_estimator_sees_only_the_rows_with_a_target(self, frame_xy):
        """The estimator receives the 3 dates whose target is observed."""
        X, y = frame_xy
        pipeline = XYPipeline([('drop', DropNaNTarget()), ('model', SpyEstimator())]).fit(X, y)

        assert pipeline.named_steps['model'].fit_X_.index.equals(X.index[[0, 2, 3]])

    def test_predict_keeps_every_row(self, pipeline, frame_xy):
        """``predict`` returns one value per input row, NaN targets included."""
        X, y = frame_xy

        predictions = pipeline.fit(X, y).predict(X)

        assert len(predictions) == len(X)

    def test_score_ignores_the_nan_targets(self, pipeline, frame_xy):
        """``score`` drops the rows without target before scoring: a finite R²."""
        X, y = frame_xy

        assert np.isfinite(pipeline.fit(X, y).score(X, y))


# =============================================================================
# XYTransformerMixin
# =============================================================================
class _BareXY(XYTransformerMixin):
    """Subclass of the mixin that implements none of the hooks."""


class TestXYTransformerMixin:
    """A minimal subclass writes ``_fit`` / ``_transform`` and gets the public methods."""

    def test_fit_returns_self(self, frame_xy):
        """``fit`` follows the sklearn convention."""
        step = MixinScaleXY()

        assert step.fit(*frame_xy) is step

    def test_fit_transform_returns_both(self, frame_xy):
        """Valeur d'or : k = 3 → X = 3x, y = 3y."""
        X, y = frame_xy

        X_t, y_t = MixinScaleXY(k=3.0).fit_transform(X, y.fillna(0.0))

        assert (X_t['x'].tolist(), y_t.tolist()) == ([3.0, 6.0, 9.0, 12.0, 15.0], [30.0, 0.0, 90.0, 120.0, 0.0])

    def test_fit_transform_without_target_returns_x(self, frame_xy):
        """Without ``y``, ``fit_transform`` returns ``X'`` alone."""
        X, _y = frame_xy

        assert isinstance(MixinScaleXY().fit_transform(X), pd.DataFrame)

    def test_transform_with_target_returns_a_pair(self, frame_xy):
        """``transform(X, y)`` returns the pair ``(X', y')``."""
        X, y = frame_xy
        step = MixinScaleXY(k=2.0).fit(X, y)

        _X_t, y_t = step.transform(X, y.fillna(1.0))

        assert y_t.tolist() == [20.0, 2.0, 60.0, 80.0, 2.0]

    @pytest.mark.parametrize(
        ('method', 'message'),
        [
            ('fit', 'Subclasses must implement _fit'),
            ('transform', 'Subclasses must implement _transform'),
            ('inverse_transform', '_BareXY does not implement _inverse_transform'),
        ],
        ids=['fit', 'transform', 'inverse_transform'],
    )
    def test_unimplemented_hook_raises(self, method, message, frame_xy):
        """A public method whose hook is missing raises ``NotImplementedError`` naming it."""
        with pytest.raises(NotImplementedError, match=message):
            getattr(_BareXY(), method)(*frame_xy)

    def test_inverse_transform_left_unimplemented(self, frame_xy):
        """``MixinScaleXY`` writes no ``_inverse_transform``: the class is named in the error."""
        X, y = frame_xy
        step = MixinScaleXY().fit(X, y)

        with pytest.raises(NotImplementedError, match='MixinScaleXY does not implement'):
            step.inverse_transform(X, y)

    def test_get_params_lists_the_init_arguments(self):
        """``get_params`` reads the ``__init__`` arguments of the subclass."""
        assert MixinScaleXY(k=4.0).get_params() == {'k': 4.0}

    def test_set_params_changes_the_transform(self, frame_xy):
        """``set_params(k=...)`` takes effect at the next ``transform``."""
        X, y = frame_xy
        step = MixinScaleXY(k=2.0).fit(X, y)

        step.set_params(k=10.0)

        assert step.transform(X)['x'].tolist()[0] == 10.0

    def test_clone_keeps_the_parameters(self):
        """``clone`` copies ``k`` as it was given."""
        assert clone(MixinScaleXY(k=5.0)).k == 5.0

    def test_clone_is_not_fitted(self, frame_xy):
        """The fitted state of the mixin is not carried over by ``clone``."""
        step = MixinScaleXY().fit(*frame_xy)

        with pytest.raises(NotFittedError):
            check_is_fitted(clone(step))

    def test_unfitted_subclass_fails_check_is_fitted(self):
        """Before ``fit``, a minimal subclass is reported not fitted."""
        with pytest.raises(NotFittedError):
            check_is_fitted(MixinScaleXY())

    def test_fitted_subclass_passes_check_is_fitted(self, frame_xy):
        """After ``fit``, a minimal subclass is reported fitted by sklearn."""
        step = MixinScaleXY().fit(*frame_xy)

        check_is_fitted(step)

    def test_subclass_overriding_fit_is_fitted_by_its_learnt_attributes(self, frame_xy):
        """A subclass whose own ``fit`` skips the mixin is fitted once it has a learnt attribute."""
        class _OwnFit(MixinScaleXY):
            def fit(self, X, y=None):
                self.scale_ = self.k
                return self

        check_is_fitted(_OwnFit().fit(*frame_xy))

    def test_last_step_of_a_fitted_pipeline_transforms(self, frame_xy):
        """A fitted ``XYPipeline`` ending with a minimal subclass can ``transform``."""
        X, y = frame_xy
        pipeline = XYPipeline([('scale', MixinScaleXY(k=2.0))]).fit(X, y)

        assert pipeline.transform(X)['x'].tolist()[0] == 2.0

    def test_intermediate_step_propagates_y(self, frame_xy):
        """In the middle of a pipeline, the subclass passes ``k * y`` to the estimator."""
        X, y = frame_xy
        pipeline = XYPipeline([('scale', MixinScaleXY(k=2.0)), ('model', SpyEstimator())])

        pipeline.fit(X, y.fillna(0.0))

        assert pipeline.named_steps['model'].fit_y_.tolist() == [20.0, 0.0, 60.0, 80.0, 0.0]


# =============================================================================
# XYPanelTimeSeriesTransformer
# =============================================================================
class _CentreXY(XYPanelTimeSeriesTransformer):
    """Centres ``X`` on its fitted means and doubles ``y``."""

    def _fit(self, X, y=None):
        self.mean_ = X.mean()

    def _transform(self, X, y=None):
        X_t = X - self.mean_
        if y is None:
            return X_t
        return X_t, y * 2


@pytest.fixture
def series_xy():
    """``(X, y)``: 4 month ends, ``a`` = 1, 2, 3, 4 (mean 2.5), ``y`` = 1, 2, 3, 4."""
    dates = pd.date_range('2024-01-31', periods=4, freq='ME', name='date')
    return pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0]}, index=dates), pd.Series([1.0, 2.0, 3.0, 4.0], index=dates)


class TestXYPanelTimeSeriesTransformer:
    """Validated transformer of the base hierarchy that transforms ``X`` and ``y``."""

    def test_missing_transform_hook_prevents_instantiation(self):
        """``_transform`` is abstract: a subclass without it cannot be instantiated."""
        class _NoTransform(XYPanelTimeSeriesTransformer):
            def _fit(self, X, y=None):
                pass

        with pytest.raises(TypeError, match='abstract'):
            _NoTransform()

    def test_fit_transform_returns_both(self, series_xy):
        """Valeur d'or : a - 2.5 et 2y."""
        X_t, y_t = _CentreXY().fit_transform(*series_xy)

        assert (X_t['a'].tolist(), y_t.tolist()) == ([-1.5, -0.5, 0.5, 1.5], [2.0, 4.0, 6.0, 8.0])

    def test_transform_without_target_returns_x(self, series_xy):
        """Without ``y``, ``transform`` returns ``X'`` alone."""
        X, y = series_xy

        assert isinstance(_CentreXY().fit(X, y).transform(X), pd.DataFrame)

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_unfitted_call_raises(self, method, series_xy):
        """``transform`` and ``inverse_transform`` require a fitted transformer."""
        X, _y = series_xy

        with pytest.raises(NotFittedError):
            getattr(_CentreXY(), method)(X)

    def test_other_columns_are_refused(self, series_xy):
        """The columns seen at ``fit`` are checked at ``transform``."""
        X, y = series_xy
        step = _CentreXY().fit(X, y)

        with pytest.raises(ValueError, match="unseen at fit time: \\['z'\\]"):
            step.transform(X.rename(columns={'a': 'z'}), y)

    def test_unvalidated_transform_still_checks_the_columns(self, series_xy):
        """With ``validate_input=False``, the columns seen at ``fit`` are still enforced."""
        X, y = series_xy
        step = _CentreXY(validate_input=False).fit(X, y)

        with pytest.raises(ValueError, match="seen at fit time, yet now missing: \\['a'\\]"):
            step.transform(X.rename(columns={'a': 'z'}), y)

    def test_series_input_is_refused(self, series_xy):
        """A ``Series`` ``X`` is refused, with a pointer to ``to_frame()``."""
        X, y = series_xy
        step = _CentreXY().fit(X, y)

        with pytest.raises(ValueError, match='to_frame'):
            step.transform(X['a'], y)

    def test_default_inverse_transform_raises(self, series_xy):
        """Without ``_inverse_transform``, the inverse names the class."""
        X, y = series_xy
        step = _CentreXY().fit(X, y)

        with pytest.raises(NotImplementedError, match='_CentreXY does not implement _inverse_transform'):
            step.inverse_transform(X, y)

    def test_in_pipeline_the_estimator_receives_the_transformed_target(self, series_xy):
        """In an ``XYPipeline``, the estimator is fitted on ``2y``."""
        X, y = series_xy
        pipeline = XYPipeline([('centre', _CentreXY()), ('model', SpyEstimator())]).fit(X, y)

        assert pipeline.named_steps['model'].fit_y_.tolist() == [2.0, 4.0, 6.0, 8.0]


# =============================================================================
# is_xy_transformer : déclaration par héritage
# =============================================================================
class _SignatureOnlyXY:
    """Object with XY-like methods that inherits no XY base class."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X if y is None else (X, y)


class TestIsXYTransformer:
    """A step is XY exactly when it inherits ``XYTransformerMixin`` or ``XYPanelTimeSeriesTransformer``."""

    @pytest.mark.parametrize(
        'make, expected',
        [
            (DropNaNTarget, True),
            (MixinScaleXY, True),
            (_CentreXY, True),
            (StandardScaler, False),
            (_SignatureOnlyXY, False),
        ],
        ids=['drop_nan_target', 'mixin_subclass', 'panel_xy_subclass', 'sklearn_transformer', 'signature_only'],
    )
    def test_declaration(self, make, expected):
        """The declaration, not the signature, makes a step XY."""
        assert is_xy_transformer(make()) is expected

    def test_strings_are_not_xy(self):
        """``'passthrough'`` is not an XY step."""
        assert is_xy_transformer('passthrough') is False
