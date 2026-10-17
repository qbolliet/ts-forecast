"""Tests of the sklearn contract of ``PanelTimeSeriesTransformer``.

Covers the abstract ``_fit`` / ``_transform`` (``TypeError`` at instantiation),
``fit`` / ``transform`` / ``fit_transform`` on a series and a panel stored in a
``DataFrame``, ``NotFittedError`` before ``fit``, the check at ``transform`` of the
columns seen at ``fit``, the fitted attributes ``is_panel_``,
``n_features_in_`` and ``feature_names_in_``, ``get_params`` / ``set_params`` /
``clone`` (constructor parameters stored as given), the refusal of a
``pandas.Series`` (``X`` is two-dimensional), empty and single-observation data,
and the non-mutation of the caller's input.
"""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from tsforecast.base.transformers import PanelTimeSeriesTransformer
from tests.support.perturbations import (
    empty_like,
    shuffle_rows,
    single_observation,
    to_three_level_index,
    with_special_column_names,
)

from .conftest import CenteredScaler, ConcreteTransformer, build_column_panel, build_index_panel, build_series_frame


# =============================================================================
# Méthodes abstraites
# =============================================================================
class TestAbstractMethods:
    """``_fit`` and ``_transform`` must be implemented before instantiation."""

    def test_base_class_cannot_be_instantiated(self):
        """The abstract base itself cannot be instantiated."""
        with pytest.raises(TypeError, match="abstract methods '_fit', '_transform'"):
            PanelTimeSeriesTransformer()

    def test_missing_transform_is_refused(self):
        """A subclass implementing only ``_fit`` is still abstract."""
        class OnlyFit(PanelTimeSeriesTransformer):
            def _fit(self, X, y=None):
                pass

        with pytest.raises(TypeError, match="abstract method '_transform'"):
            OnlyFit()

    def test_missing_fit_is_refused(self):
        """A subclass implementing only ``_transform`` is still abstract."""
        class OnlyTransform(PanelTimeSeriesTransformer):
            def _transform(self, X):
                return X

        with pytest.raises(TypeError, match="abstract method '_fit'"):
            OnlyTransform()


# =============================================================================
# fit / transform / fit_transform
# =============================================================================
class TestFitTransform:
    """``fit`` returns the estimator, hands ``y`` over, and ``transform`` applies ``_transform``."""

    def test_fit_returns_self(self, series_frame):
        """``fit`` returns the estimator itself, for chaining."""
        transformer = ConcreteTransformer()
        assert transformer.fit(series_frame) is transformer

    def test_target_is_handed_to_fit_as_is(self, series_frame):
        """``y`` reaches ``_fit`` unchanged (same object)."""
        y = pd.Series([0.0, 1.0, 0.0, 1.0], index=series_frame.index)
        assert ConcreteTransformer().fit(series_frame, y).fit_y_ is y

    @pytest.mark.parametrize(
        ('build', 'kwargs'),
        [(build_series_frame, {}), (build_index_panel, {}),
         (build_column_panel, {'time_col': 'date', 'panel_cols': ['country']})],
        ids=['series', 'index-panel', 'column-panel'],
    )
    def test_fit_transform_equals_fit_then_transform(self, build, kwargs):
        """``fit_transform(X)`` and ``fit(X).transform(X)`` give the same frame."""
        data = build()
        expected = CenteredScaler(**kwargs).fit(data).transform(data)
        pd.testing.assert_frame_equal(CenteredScaler(**kwargs).fit_transform(data), expected)

    def test_transform_uses_the_state_learned_at_fit(self, series_frame):
        """Data transformed after ``fit`` is centred on the training mean."""
        transformer = CenteredScaler().fit(series_frame)
        later = pd.DataFrame({'value': [5.0, 6.0]}, index=pd.date_range('2023-05-31', periods=2, freq='ME'))

        # Valeur d'or : moyenne d'entraînement 2.5, (5 - 2.5) * 2 = 5, (6 - 2.5) * 2 = 7
        assert transformer.transform(later)['value'].tolist() == [5.0, 7.0]

    def test_transform_validates_its_input(self, series_frame):
        """``transform`` applies the same validation as ``fit``."""
        transformer = ConcreteTransformer().fit(series_frame)
        with pytest.raises(ValueError, match='Time series index is not sorted'):
            transformer.transform(series_frame.iloc[::-1])

class TestFeatureNamesAtTransform:
    """``transform`` accepts only the columns seen at ``fit``, in the same order."""

    def test_unseen_column_is_refused(self, series_frame):
        """A column absent at ``fit`` is named in the error."""
        transformer = ConcreteTransformer().fit(series_frame)
        with pytest.raises(ValueError, match=r"unseen at fit time: \['extra'\]"):
            transformer.transform(series_frame.assign(extra=0.0))

    def test_missing_column_is_refused(self, index_panel):
        """A column seen at ``fit`` and now absent is named in the error."""
        transformer = ConcreteTransformer().fit(index_panel)
        with pytest.raises(ValueError, match=r"seen at fit time, yet now missing: \['b'\]"):
            transformer.transform(index_panel[['a']])

    def test_reordered_columns_are_refused(self, index_panel):
        """The same columns in another order are refused, as in sklearn."""
        transformer = ConcreteTransformer().fit(index_panel)
        with pytest.raises(ValueError, match='same names in another order'):
            transformer.transform(index_panel[['b', 'a']])

    def test_column_panel_is_checked_on_its_value_columns(self, column_panel):
        """Time and panel columns moved into the index are not part of the check."""
        transformer = ConcreteTransformer(time_col='date', panel_cols=['country']).fit(column_panel)
        with pytest.raises(ValueError, match=r"unseen at fit time: \['c'\]"):
            transformer.transform(column_panel.assign(c=0.0))

    def test_without_validation_every_column_is_checked(self, column_panel):
        """Without validation, the raw columns of ``fit`` are expected."""
        transformer = ConcreteTransformer(time_col='date', panel_cols=['country'], validate_input=False)
        transformer.fit(column_panel)
        with pytest.raises(ValueError, match=r"seen at fit time, yet now missing: \['country'\]"):
            transformer.transform(column_panel.drop(columns='country'))


class TestNotFitted:
    """Using a transformer before ``fit`` raises ``NotFittedError``."""

    @pytest.mark.parametrize('validate_input', [True, False], ids=['validated', 'not-validated'])
    def test_transform_before_fit(self, series_frame, validate_input):
        """``transform`` refuses to run before ``fit``, whatever the validation setting."""
        with pytest.raises(NotFittedError, match='is not fitted yet'):
            ConcreteTransformer(validate_input=validate_input).transform(series_frame)

    def test_check_is_fitted_passes_after_fit(self, series_frame):
        """After ``fit``, the trailing-underscore attributes make the estimator fitted."""
        transformer = ConcreteTransformer()
        with pytest.raises(NotFittedError):
            check_is_fitted(transformer)

        check_is_fitted(transformer.fit(series_frame))


# =============================================================================
# Attributs ajustés
# =============================================================================
class TestFittedAttributes:
    """``n_features_in_`` and ``feature_names_in_`` follow the sklearn convention."""

    @pytest.mark.parametrize(
        ('build', 'kwargs', 'names'),
        [(build_series_frame, {}, ['value']),
         (build_index_panel, {}, ['a', 'b']),
         (build_column_panel, {'time_col': 'date', 'panel_cols': ['country'], 'convert_cols_to_index': True},
          ['a', 'b'])],
        ids=['series', 'index-panel', 'converted-column-panel'],
    )
    def test_feature_names_exclude_time_and_panel_columns(self, build, kwargs, names):
        """Only value columns count: time and panel columns moved into the index do not."""
        transformer = ConcreteTransformer(**kwargs).fit(build())

        assert transformer.n_features_in_ == len(names)
        np.testing.assert_array_equal(transformer.feature_names_in_, np.array(names, dtype=object))

    def test_feature_names_are_an_object_array(self, index_panel):
        """``feature_names_in_`` is a numpy array of dtype ``object``."""
        assert ConcreteTransformer().fit(index_panel).feature_names_in_.dtype == object

    def test_unconverted_time_and_panel_columns_count(self, column_panel):
        """Validated without conversion, the time and panel columns stay columns, and count as features."""
        transformer = ConcreteTransformer(time_col='date', panel_cols=['country']).fit(column_panel)

        assert list(transformer.feature_names_in_) == ['country', 'date', 'a', 'b']

    def test_without_validation_every_column_counts(self, column_panel):
        """Without validation the time and panel columns stay columns, and count as features."""
        transformer = ConcreteTransformer(time_col='date', panel_cols=['country'], validate_input=False)

        transformer.fit(column_panel)

        assert list(transformer.feature_names_in_) == ['country', 'date', 'a', 'b']

    def test_special_column_names_are_kept(self, index_panel):
        """Column names with spaces, accents and symbols are recorded verbatim."""
        panel, mapping = with_special_column_names(index_panel)
        transformer = ConcreteTransformer().fit(panel)
        assert list(transformer.feature_names_in_) == [mapping['a'], mapping['b']]

    def test_attributes_are_absent_before_fit(self):
        """No fitted attribute exists on an unfitted transformer."""
        transformer = ConcreteTransformer()
        assert not any(hasattr(transformer, name) for name in ('is_panel_', 'n_features_in_', 'feature_names_in_'))


# =============================================================================
# get_params / set_params / clone
# =============================================================================
class TestParams:
    """Constructor parameters are stored as given and handled by sklearn's tools."""

    def test_parameters_are_stored_as_given(self):
        """``get_params`` returns the very objects passed to the constructor."""
        panel_cols = ['country', 'sector']
        transformer = ConcreteTransformer(time_col='date', panel_cols=panel_cols, auto_sort=True)

        params = transformer.get_params()

        assert params['panel_cols'] is panel_cols
        assert params == {
            'time_col': 'date', 'panel_cols': panel_cols, 'validate_input': True,
            'strict_validation': True, 'auto_sort': True, 'convert_cols_to_index': False,
        }

    def test_subclass_parameters_join_the_base_ones(self):
        """A subclass parameter is listed next to the six base parameters."""
        assert set(CenteredScaler(factor=3.0).get_params()) == {
            'factor', 'time_col', 'panel_cols', 'validate_input',
            'strict_validation', 'auto_sort', 'convert_cols_to_index',
        }

    def test_set_params_updates_and_returns_self(self, series_frame):
        """``set_params`` returns the estimator, and the next ``fit`` uses the new value."""
        transformer = CenteredScaler()

        assert transformer.set_params(factor=3.0) is transformer

        # Valeur d'or : (v - 2.5) * 3
        assert transformer.fit_transform(series_frame)['value'].tolist() == [-4.5, -1.5, 1.5, 4.5]

    def test_set_params_rejects_an_unknown_parameter(self):
        """An unknown parameter name raises."""
        with pytest.raises(ValueError, match='Invalid parameter'):
            ConcreteTransformer().set_params(unknown=1)

    def test_clone_is_unfitted_with_equal_parameters(self, index_panel):
        """``clone`` copies the parameters, not the fitted state."""
        transformer = CenteredScaler(factor=3.0, auto_sort=True).fit(index_panel)

        cloned = clone(transformer)

        assert cloned.get_params() == transformer.get_params()
        assert not hasattr(cloned, 'center_') and not hasattr(cloned, 'is_panel_')

    def test_clone_fits_to_the_same_output(self, index_panel):
        """A clone fitted on the same data gives the same output."""
        transformer = CenteredScaler(factor=3.0)
        expected = transformer.fit_transform(index_panel)
        pd.testing.assert_frame_equal(clone(transformer).fit_transform(index_panel), expected)


# =============================================================================
# Entrée Series, jeux vides, une observation
# =============================================================================
class TestInputKinds:
    """Series, empty and single-row inputs."""

    @pytest.mark.parametrize('validate_input', [True, False], ids=['validated', 'not-validated'])
    def test_series_is_refused_at_fit(self, series_frame, validate_input):
        """``X`` is two-dimensional, as in sklearn: a ``Series`` is pointed to ``to_frame()``."""
        with pytest.raises(ValueError, match=r'X must be a pandas DataFrame, got Series; pass series.to_frame\(\)'):
            ConcreteTransformer(validate_input=validate_input).fit(series_frame['value'])

    def test_series_is_refused_at_transform(self, series_frame):
        """A ``Series`` is refused at ``transform`` too, before any validation."""
        transformer = ConcreteTransformer().fit(series_frame)
        with pytest.raises(ValueError, match='X must be a pandas DataFrame, got Series'):
            transformer.transform(series_frame['value'])

    def test_single_column_frame_of_a_series(self, series_frame):
        """``series.to_frame()`` is fitted as a single feature named after the series."""
        transformer = ConcreteTransformer().fit(series_frame['value'].to_frame())
        np.testing.assert_array_equal(transformer.feature_names_in_, np.array(['value'], dtype=object))

    @pytest.mark.parametrize('build', [build_series_frame, build_index_panel], ids=['series', 'panel'])
    def test_empty_input(self, build):
        """An empty frame is fitted and transformed into an empty frame with the same columns."""
        data = empty_like(build())

        output = CenteredScaler().fit_transform(data)

        assert output.empty and list(output.columns) == list(data.columns)

    @pytest.mark.parametrize('build', [build_series_frame, build_index_panel], ids=['series', 'panel'])
    def test_single_observation(self, build):
        """A single row is its own mean: it is centred to zero."""
        data = single_observation(build())

        output = CenteredScaler().fit_transform(data)

        pd.testing.assert_index_equal(output.index, data.index)
        assert (output.to_numpy() == 0.0).all()


# =============================================================================
# Non-mutation de l'entrée
# =============================================================================
def _string_dated_frame() -> pd.DataFrame:
    """Monthly frame indexed by ISO date strings."""
    return pd.DataFrame({'value': [1.0, 2.0]}, index=['2023-01-31', '2023-02-28'])


class TestNonMutation:
    """No call modifies the caller's data: a deep copy taken before equals the data after."""

    @pytest.mark.parametrize(
        ('build', 'kwargs'),
        [
            (build_series_frame, {}),
            (build_index_panel, {}),
            (lambda: to_three_level_index(build_index_panel()), {}),
            (build_column_panel, {'time_col': 'date', 'panel_cols': ['country']}),
            (build_column_panel, {'time_col': 'date', 'panel_cols': ['country'], 'convert_cols_to_index': True}),
            (lambda: shuffle_rows(build_index_panel(), seed=0), {'auto_sort': True}),
            (lambda: shuffle_rows(build_column_panel(), seed=0),
             {'time_col': 'date', 'panel_cols': ['country'], 'auto_sort': True}),
            (lambda: build_index_panel().rename(index=lambda d: d.strftime('%Y-%m-%d'), level=1), {}),
        ],
        ids=['series', 'index-panel', 'three-levels', 'column-panel', 'column-panel-converted',
             'shuffled-index-panel', 'shuffled-column-panel', 'string-dates-last-level'],
    )
    def test_fit_transform_inverse_leave_the_input_intact(self, build, kwargs):
        """``fit``, ``transform`` and ``inverse_transform`` leave the input as it was."""
        data = build()
        before = copy.deepcopy(data)

        transformer = CenteredScaler(**kwargs).fit(data)
        transformer.inverse_transform(transformer.transform(data))

        pd.testing.assert_frame_equal(data, before)

    @pytest.mark.parametrize(
        'build',
        [_string_dated_frame,
         lambda: pd.DataFrame({'value': [1.0, 2.0]}, index=pd.period_range('2023-01', periods=2, freq='M'))],
        ids=['string-dates', 'period-index'],
    )
    def test_converted_index_is_left_intact(self, build):
        """An index converted to dates for validation is not converted in the caller's data."""
        data = build()
        before = copy.deepcopy(data)

        transformer = ConcreteTransformer().fit(data)
        transformer.transform(data)

        assert data.equals(before) and data.index.dtype == before.index.dtype

    def test_string_dated_index_is_converted_for_the_subclass(self):
        """The data handed to ``_fit`` carries the converted dates."""
        transformer = ConcreteTransformer().fit(_string_dated_frame())
        assert isinstance(transformer.fit_input_.index, pd.DatetimeIndex)
