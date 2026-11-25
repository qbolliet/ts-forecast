"""Introspection and parameters of ``XYPipeline``, compared with ``sklearn.pipeline.Pipeline``.

Covers the access to the steps (``steps``, ``named_steps``, ``__getitem__`` by
position and by name, ``__len__``), slicing (a slice of a fitted pipeline is a
working pipeline sharing the fitted steps; the constructor parameters a slice
keeps), ``get_params`` / ``set_params`` (nested parameters, replacing or
skipping a step, ``preserve_dataframe``, unknown parameters), ``clone``
(unfitted copies, same parameters), the text and HTML representations, and the
``verbose`` messages.
"""
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from tsforecast.xy import DropNaNTarget, XYPipeline

from .conftest import GAPPED_ROWS


def _steps():
    """Fresh steps of the reference pipeline: scale, drop the rows without target, ridge."""
    return [('scale', StandardScaler()), ('drop', DropNaNTarget()), ('model', Ridge(alpha=1.0))]


def _observed(X: pd.DataFrame, y: pd.Series):
    """Rows of ``(X, y)`` whose target is observed."""
    keep = y.notna().to_numpy()
    return X[keep], y[keep]


# ==============================================================================
# Accès aux étapes
# ==============================================================================


class TestStepAccess:
    """``steps``, ``named_steps``, ``pipe[i]``, ``pipe['name']`` and ``len`` behave as in ``Pipeline``."""

    def test_steps_keep_the_given_order(self):
        """``steps`` lists the names in the given order."""
        pipeline = XYPipeline(_steps())

        assert [name for name, _ in pipeline.steps] == ['scale', 'drop', 'model']

    def test_named_steps_by_attribute(self):
        """``named_steps.<name>`` is the given step object."""
        steps = _steps()

        assert XYPipeline(steps).named_steps.drop is steps[1][1]

    @pytest.mark.parametrize('key, position', [(0, 0), (-1, 2), ('drop', 1)], ids=['first', 'last', 'by_name'])
    def test_getitem_returns_the_step(self, key, position):
        """``pipe[0]``, ``pipe[-1]`` and ``pipe['drop']`` return the step objects themselves."""
        steps = _steps()

        assert XYPipeline(steps)[key] is steps[position][1]

    def test_unknown_name_raises_key_error(self):
        """An unknown step name raises ``KeyError``, as in ``Pipeline``."""
        with pytest.raises(KeyError, match='nope'):
            XYPipeline(_steps())['nope']

    def test_position_out_of_range_raises_index_error(self):
        """A position beyond the last step raises ``IndexError``."""
        with pytest.raises(IndexError):
            XYPipeline(_steps())[3]

    def test_len_is_that_of_pipeline(self):
        """``len`` counts every step, ``'passthrough'`` included, like ``Pipeline``."""
        steps = [('scale', StandardScaler()), ('skip', 'passthrough'), ('model', Ridge())]

        assert len(XYPipeline(steps)) == len(Pipeline(steps)) == 3


# ==============================================================================
# Découpage
# ==============================================================================


class TestSlicing:
    """``pipe[:-1]`` of a fitted pipeline is a working ``XYPipeline`` sharing the fitted steps."""

    @pytest.fixture
    def fitted(self, gapped_linear_xy):
        """Reference pipeline fitted on ``gapped_linear_xy``."""
        return XYPipeline(_steps()).fit(*gapped_linear_xy)

    def test_slice_shares_the_fitted_step_objects(self, fitted):
        """The steps of the slice are the fitted objects of the pipeline, not copies."""
        assert fitted[:-1]['scale'] is fitted['scale']

    def test_slice_transforms_with_the_fitted_parameters(self, fitted, gapped_linear_xy):
        """``pipe[:-1].transform(X)`` scales with the means and deviations learnt on all of ``X``."""
        X, _y = gapped_linear_xy

        # Référence indépendante : StandardScaler ajusté hors de tout pipeline
        np.testing.assert_allclose(fitted[:-1].transform(X).to_numpy(), StandardScaler().fit_transform(X))

    def test_slice_with_target_drops_the_rows_without_target(self, fitted, gapped_linear_xy):
        """``pipe[:-1].transform(X, y)`` returns the observed rows, as the final estimator saw them."""
        X, y = gapped_linear_xy

        X_t, _y_t = fitted[:-1].transform(X, y)

        assert X_t.index.equals(X.index.delete(list(GAPPED_ROWS)))

    def test_final_step_on_the_slice_output_predicts_like_the_pipeline(self, fitted, gapped_linear_xy):
        """``pipe[-1].predict(pipe[:-1].transform(X))`` equals ``pipe.predict(X)``."""
        X, _y = gapped_linear_xy

        # Valeurs du tableau seulement : l'estimateur final a été ajusté sur des tableaux
        np.testing.assert_allclose(fitted[-1].predict(fitted[:-1].transform(X).to_numpy()), fitted.predict(X))

    def test_slice_output_warns_the_array_fitted_final_step(self, fitted, gapped_linear_xy):
        """The slice returns a ``DataFrame``, whereas the final step was fitted on arrays: sklearn warns.

        Inside the pipeline the scaler output stays an array, as in ``Pipeline``,
        so the final step learns no feature names; the slice, a pipeline on its
        own, gives its output the pandas structure back.
        """
        X, _y = gapped_linear_xy

        with pytest.warns(UserWarning, match='X has feature names, but Ridge was fitted without feature names'):
            fitted[-1].predict(fitted[:-1].transform(X))

    def test_pandas_output_end_to_end_names_the_final_features(self, gapped_linear_xy):
        """With ``set_output(transform='pandas')``, the final step learns the column names."""
        X, y = gapped_linear_xy

        pipeline = XYPipeline(_steps()).set_output(transform='pandas').fit(X, y)

        assert list(pipeline[-1].feature_names_in_) == ['a', 'b']

    def test_pandas_output_end_to_end_does_not_warn(self, gapped_linear_xy):
        """With pandas outputs end to end, ``pipe[-1].predict(pipe[:-1].transform(X))`` is silent."""
        X, y = gapped_linear_xy
        pipeline = XYPipeline(_steps()).set_output(transform='pandas').fit(X, y)

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            prediction = pipeline[-1].predict(pipeline[:-1].transform(X))

        np.testing.assert_allclose(prediction, pipeline.predict(X))

    def test_slice_with_a_step_is_refused(self):
        """A slice with a step other than 1 is refused with the message of ``Pipeline``."""
        with pytest.raises(ValueError, match='Pipeline slicing only supports a step of 1'):
            XYPipeline(_steps())[::2]

    @pytest.mark.parametrize('param, value', [('memory', 'cache_dir'), ('verbose', True)], ids=['memory', 'verbose'])
    def test_slice_keeps_the_sklearn_parameters(self, param, value):
        """Like ``Pipeline``, a slice keeps ``memory`` and ``verbose``."""
        pipeline = XYPipeline(_steps(), **{param: value})

        assert getattr(pipeline[:-1], param) == value

    def test_slice_keeps_preserve_dataframe(self):
        """A slice of a pipeline built with ``preserve_dataframe=False`` keeps that setting."""
        pipeline = XYPipeline(_steps(), preserve_dataframe=False)

        assert pipeline[:-1].preserve_dataframe is False

    def test_slice_output_follows_preserve_dataframe(self, gapped_linear_xy):
        """The slice of a pipeline returning arrays returns arrays too."""
        X, y = gapped_linear_xy
        pipeline = XYPipeline(_steps(), preserve_dataframe=False).fit(X, y)

        assert isinstance(pipeline[:-1].transform(X), np.ndarray)


# ==============================================================================
# get_params / set_params
# ==============================================================================


class TestParams:
    """``get_params`` / ``set_params``: the parameters of ``Pipeline`` plus ``preserve_dataframe``."""

    def test_shallow_params_are_the_constructor_arguments(self):
        """``get_params(deep=False)`` lists exactly the constructor arguments."""
        params = XYPipeline(_steps()).get_params(deep=False)

        assert set(params) == {'steps', 'transform_input', 'memory', 'verbose', 'preserve_dataframe'}

    def test_deep_params_are_those_of_pipeline_plus_preserve_dataframe(self):
        """The deep parameter names are those of ``Pipeline`` on the same steps, plus ``preserve_dataframe``."""
        xy_names = set(XYPipeline(_steps()).get_params(deep=True))
        sk_names = set(Pipeline(_steps()).get_params(deep=True))

        assert xy_names == sk_names | {'preserve_dataframe'}

    def test_deep_params_hold_the_step_objects(self):
        """``get_params(deep=True)['drop']`` is the step object itself."""
        steps = _steps()

        assert XYPipeline(steps).get_params(deep=True)['drop'] is steps[1][1]

    def test_set_params_returns_the_pipeline(self):
        """``set_params`` returns the pipeline, for chaining."""
        pipeline = XYPipeline(_steps())

        assert pipeline.set_params(model__alpha=2.0) is pipeline

    def test_set_params_replaces_a_step(self):
        """``set_params(scale=MinMaxScaler())`` puts the new object in place of the step."""
        new_scaler = MinMaxScaler()

        pipeline = XYPipeline(_steps()).set_params(scale=new_scaler)

        assert pipeline['scale'] is new_scaler

    def test_replaced_step_is_used_by_fit(self, gapped_linear_xy):
        """After replacing the scaler, the predictions are those of ``Pipeline`` built with the new scaler."""
        X, y = gapped_linear_xy
        X_obs, y_obs = _observed(X, y)
        pipeline = XYPipeline(_steps()).set_params(scale=MinMaxScaler(), drop='passthrough')
        reference = Pipeline([('scale', MinMaxScaler()), ('model', Ridge(alpha=1.0))])

        # Cibles observées seulement : sans DropNaNTarget, Pipeline refuserait les NaN
        np.testing.assert_allclose(pipeline.fit(X_obs, y_obs).predict(X), reference.fit(X_obs, y_obs).predict(X))

    @pytest.mark.parametrize('skipped', ['passthrough', None], ids=['passthrough', 'none'])
    def test_skipped_scaler_fits_on_the_raw_features(self, skipped, gapped_linear_xy):
        """``set_params(scale='passthrough')`` (or None): the model is fitted on the raw observed rows."""
        X, y = gapped_linear_xy
        X_obs, y_obs = _observed(X, y)

        pipeline = XYPipeline(_steps()).set_params(scale=skipped).fit(X, y)

        # Référence indépendante : Ridge ajusté sur les lignes observées, sans mise à l'échelle
        np.testing.assert_allclose(pipeline.predict(X), Ridge(alpha=1.0).fit(X_obs, y_obs).predict(X))

    def test_set_params_steps_replaces_every_step(self):
        """``set_params(steps=...)`` replaces the whole list of steps."""
        pipeline = XYPipeline(_steps()).set_params(steps=[('model', LinearRegression())])

        assert [name for name, _ in pipeline.steps] == ['model']

    def test_set_params_preserve_dataframe(self, gapped_linear_xy):
        """``set_params(preserve_dataframe=False)``: the output of ``transform`` is an array again."""
        X, y = gapped_linear_xy
        pipeline = XYPipeline([('scale', StandardScaler())]).set_params(preserve_dataframe=False)

        assert isinstance(pipeline.fit(X, y).transform(X), np.ndarray)

    def test_unknown_parameter_is_refused(self):
        """An unknown nested parameter raises the ``ValueError`` of ``Pipeline``."""
        with pytest.raises(ValueError, match="Invalid parameter 'nope' for estimator"):
            XYPipeline(_steps()).set_params(model__nope=1.0)


# ==============================================================================
# clone
# ==============================================================================


class TestClone:
    """``clone`` gives an unfitted ``XYPipeline`` of new, unfitted steps with the same parameters."""

    @pytest.fixture
    def fitted(self, gapped_linear_xy):
        """Reference pipeline with non-default parameters, fitted on ``gapped_linear_xy``."""
        return XYPipeline(_steps(), preserve_dataframe=False).fit(*gapped_linear_xy)

    def test_clone_is_an_xy_pipeline(self, fitted):
        """The clone is an ``XYPipeline``."""
        assert type(clone(fitted)) is XYPipeline

    def test_clone_is_not_fitted(self, fitted):
        """The clone of a fitted pipeline is not fitted."""
        with pytest.raises(NotFittedError):
            check_is_fitted(clone(fitted))

    @pytest.mark.parametrize('name', ['scale', 'drop', 'model'])
    def test_clone_steps_are_new_objects(self, name, fitted):
        """Each step of the clone is a new object."""
        assert clone(fitted)[name] is not fitted[name]

    def test_clone_scaler_has_no_learnt_attribute(self, fitted):
        """The scaler of the clone has not learnt any mean."""
        assert not hasattr(clone(fitted)['scale'], 'mean_')

    def test_clone_keeps_the_parameters(self, fitted):
        """The clone has the same shallow parameters, ``preserve_dataframe=False`` included."""
        cloned = clone(fitted)

        assert (cloned.preserve_dataframe, cloned['model'].alpha) == (False, 1.0)

    def test_refitted_clone_predicts_like_the_original(self, fitted, gapped_linear_xy):
        """Fitted on the same data, the clone gives the same predictions."""
        X, y = gapped_linear_xy

        np.testing.assert_array_equal(clone(fitted).fit(X, y).predict(X), fitted.predict(X))

    def test_fitting_the_clone_leaves_the_original_unchanged(self, fitted, gapped_linear_xy):
        """Fitting the clone on other data does not touch the fitted original."""
        X, y = gapped_linear_xy
        before = fitted.predict(X)

        clone(fitted).fit(X * 10.0, y)

        np.testing.assert_array_equal(fitted.predict(X), before)


# ==============================================================================
# Représentations
# ==============================================================================


def _flat(text: str) -> str:
    """``text`` with every run of whitespace replaced by one space (alignment-independent)."""
    return ' '.join(text.split())


class TestRepr:
    """Text and HTML representations."""

    def test_repr_is_that_of_pipeline_under_its_own_name(self):
        """``repr`` is the one of ``Pipeline`` on the same steps, with the class name ``XYPipeline``."""
        steps = [('scale', StandardScaler()), ('model', Ridge(alpha=2.0))]

        assert _flat(repr(XYPipeline(steps))) == _flat('XY' + repr(Pipeline(steps)))

    def test_repr_hides_the_default_preserve_dataframe(self):
        """``preserve_dataframe`` at its default value is not shown, like any default parameter."""
        assert 'preserve_dataframe' not in repr(XYPipeline(_steps()))

    def test_repr_shows_a_changed_preserve_dataframe(self):
        """``preserve_dataframe=False`` is shown."""
        assert 'preserve_dataframe=False' in repr(XYPipeline(_steps(), preserve_dataframe=False))

    def test_html_repr_names_every_step(self):
        """The HTML representation (notebooks) names each step."""
        html = XYPipeline(_steps())._repr_html_()

        assert all(name in html for name in ('XYPipeline', 'StandardScaler', 'DropNaNTarget', 'Ridge'))


# ==============================================================================
# verbose
# ==============================================================================


def _without_times(text: str) -> str:
    """Verbose output with the elapsed times replaced by a placeholder."""
    return re.sub(r'total=\s*[\d.]+s', 'total=<t>', text)


class TestVerbose:
    """``verbose=True`` prints the messages of ``Pipeline``, step by step."""

    @pytest.mark.parametrize(
        'steps, method',
        [
            (lambda: [('scale', StandardScaler()), ('model', LinearRegression())], 'fit'),
            (lambda: [('scale', StandardScaler()), ('skip', 'passthrough'), ('model', LinearRegression())], 'fit'),
            (lambda: [('scale', StandardScaler()), ('minmax', MinMaxScaler())], 'fit_transform'),
        ],
        ids=['fit', 'fit_with_passthrough', 'fit_transform'],
    )
    def test_messages_are_those_of_pipeline(self, steps, method, linear_xy, capsys):
        """Same lines as ``Pipeline``, elapsed times aside, ``'passthrough'`` steps included."""
        X, y = linear_xy
        getattr(Pipeline(steps(), verbose=True), method)(X, y)
        expected = _without_times(capsys.readouterr().out)

        getattr(XYPipeline(steps(), verbose=True), method)(X, y)

        assert _without_times(capsys.readouterr().out) == expected

    def test_message_names_each_step(self, linear_xy, capsys):
        """One line per step, ``(step i of n) Processing <name>``."""
        XYPipeline([('scale', StandardScaler()), ('model', LinearRegression())], verbose=True).fit(*linear_xy)

        lines = capsys.readouterr().out
        assert ('(step 1 of 2) Processing scale' in lines, '(step 2 of 2) Processing model' in lines) == (True, True)

    def test_silent_by_default(self, linear_xy, capsys):
        """Nothing is printed when ``verbose`` is False."""
        XYPipeline([('scale', StandardScaler()), ('model', LinearRegression())]).fit(*linear_xy)

        assert capsys.readouterr().out == ''
