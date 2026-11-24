"""sklearn protocol and index robustness of ``StageScaler``.

Scenarios: ``fit`` / ``transform`` / ``inverse_transform`` conventions, the single
scaling routine shared by apply and invert, shuffled, duplicated, empty and single-row
grids.
"""

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Sklearn
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

# Objets testés
from tsforecast.imputation._components.stage_scaler import DEFAULT_SCALE_KEY, StageScaler

# Perturbations partagées
from tests.support.perturbations import shuffle_rows, with_duplicated_rows

# Outils de test partagés du paquet
from tests.unit.imputation._components.stage_scaler.helpers import (
    _EXACT,
    _make_scaler,
    _monthly_grid,
)


# =============================================================================
# Application, inversion, B12
# =============================================================================
class TestApplyInvert:
    """Single scaling routine, and the B12 short-circuit rule."""

    def test_scalar_one_short_circuits(self):
        """Only a scalar strictly equal to 1.0 returns the input object itself."""
        values = pd.Series([10.0, 20.0], index=_monthly_grid(periods=2))
        scaler = StageScaler()
        assert scaler.apply(values, 1.0) is values and scaler.invert(values, 1.0) is values

    @pytest.mark.parametrize('kind', ['series', 'frame'])
    def test_b12_divisors_of_ones_go_through_the_scaling_path(self, kind):
        """A Series / DataFrame of 1.0 is NOT short-circuited: a new, equal object comes back (B12)."""
        index = _monthly_grid(periods=3)
        values = pd.DataFrame({'a': [10.0, 20.0, 30.0]}, index=index)
        ones = pd.DataFrame({'a': 1.0}, index=index)
        if kind == 'series':
            values, ones = values['a'], ones['a']

        scaled = StageScaler().apply(values, ones)

        assert scaled is not values and scaled.equals(values)

    def test_b12_through_transform(self):
        """Public path: a fitted 'calendar' M -> M stage divides by a Series of 1.0, never skipped."""
        X = pd.DataFrame({'m1': [1.0, 2.0, 3.0]}, index=_monthly_grid(periods=3))
        y = pd.Series([1.0, 2.0, 3.0], index=X.index)
        scaler = StageScaler(
            scale_features='calendar', source_freq='M', pred_freq='M', column_frequencies={'m1': 'M'}
        ).fit(X, y)

        scaled = scaler.transform(y)

        assert scaled is not y and scaled.equals(y)

    def test_heterogeneous_column_divisors(self):
        """A 1.0 divisor among others leaves the other columns divided."""
        frame = pd.DataFrame({'a': [12.0, 24.0], 'b': [3.0, 6.0]}, index=_monthly_grid(periods=2))
        scaled = StageScaler().apply(frame, pd.Series({'a': 1.0, 'b': 3.0}))
        assert scaled.to_dict('list') == {'a': [12.0, 24.0], 'b': [1.0, 2.0]}

    @pytest.mark.parametrize('divisor_kind', ['scalar', 'per-row', 'per-column'])
    def test_apply_invert_roundtrip(self, divisor_kind):
        """invert(apply(v, d), d) == v for every divisor form."""
        index = _monthly_grid(periods=4)
        values = pd.DataFrame({'a': [120.0, 132.0, 150.0, 90.0], 'b': [1.0, -2.0, 3.5, np.nan]}, index=index)
        divisor = {
            'scalar': 12.0,
            'per-row': pd.DataFrame({'a': [12.0, 3.0, 4.0, 1.0], 'b': [1 / 3, 3.0, 4.0, 1.0]}, index=index),
            'per-column': pd.Series({'a': 12.0, 'b': 1 / 3}),
        }[divisor_kind]
        scaler = StageScaler()
        pd.testing.assert_frame_equal(
            scaler.invert(scaler.apply(values, divisor), divisor), values, rtol=_EXACT
        )


# =============================================================================
# Robustesse d'index
# =============================================================================
class TestIndexRobustness:
    """Shuffled, duplicated, empty and single-row grids; index names kept."""

    def test_shuffled_grid_keeps_each_divisor_on_its_row(self):
        """Per-row divisors follow their label, not their position."""
        frame = pd.DataFrame({'v': 0.0}, index=pd.date_range('2024-01-31', periods=12, freq='ME'))
        shuffled = shuffle_rows(frame, seed=3).index
        scaler = StageScaler(scale_features='calendar')
        sorted_divisors = scaler.target_divisor('v', source_freq='M', pred_freq='D', index=frame.index)
        shuffled_divisors = scaler.target_divisor('v', source_freq='M', pred_freq='D', index=shuffled)
        pd.testing.assert_series_equal(shuffled_divisors, sorted_divisors.reindex(shuffled))

    def test_duplicated_rows_are_scaled_positionally(self):
        """Duplicated dates: every row divides by its own period's count, no realignment."""
        index = pd.date_range('2024-01-31', periods=3, freq='ME')
        X = with_duplicated_rows(pd.DataFrame({'m1': [31.0, 29.0, 31.0]}, index=index), n=2)
        scaler = StageScaler(
            scale_features='calendar', source_freq='M', pred_freq='D',
            column_frequencies={'m1': 'M'}, materialization={'m1': 'raw_anchors'},
        ).fit(X, X['m1'])
        # Valeur d'or : chaque valeur vaut le nombre de jours de son mois -> 1.0 partout
        assert (scaler.transform(X)['m1'].tolist(), scaler.transform(X['m1']).tolist()) == (
            [1.0] * 5, [1.0] * 5
        )

    def test_empty_grid(self):
        """An empty grid gives empty divisors and an empty transform, without error."""
        X = pd.DataFrame({'m1': []}, index=pd.DatetimeIndex([]), dtype=float)
        scaler = StageScaler(
            scale_features='calendar', source_freq='M', pred_freq='D', column_frequencies={'m1': 'M'}
        ).fit(X)
        assert (scaler.target_divisor_.size, scaler.transform(X).shape) == (0, (0, 1))

    def test_single_observation(self):
        """A single row under 'calendar': one divisor, the days of its month."""
        index = pd.DatetimeIndex(['2024-02-10'])
        divisor = StageScaler(scale_features='calendar').target_divisor(
            'v', source_freq='M', pred_freq='D', index=index
        )
        assert divisor.tolist() == [29.0]

    def test_index_name_is_kept(self):
        """A non-standard index name is carried over to the per-row divisors."""
        index = pd.DatetimeIndex(['2024-05-15'], name='période de référence')
        divisor = StageScaler(scale_features='calendar').target_divisor(
            'v', source_freq='Q', pred_freq='D', index=index
        )
        assert (divisor.index.name, divisor.tolist()) == ('période de référence', [91.0])


# =============================================================================
# Protocole sklearn
# =============================================================================
class TestSklearnProtocol:
    """fit / transform / inverse_transform and sklearn conventions."""

    @staticmethod
    def _stage_frame() -> pd.DataFrame:
        """Build a small monthly stage frame with two covariates."""
        return pd.DataFrame(
            {'m1': [1200.0, 2400.0, 3600.0], 'a2': [120.0, 240.0, 360.0]},
            index=_monthly_grid(periods=3),
        )

    @staticmethod
    def _scaler(**overrides) -> StageScaler:
        """Scaler of the stage frame: 'm1' on the stage grid, 'a2' read at its yearly anchors."""
        params = dict(
            column_frequencies={'m1': 'M', 'a2': 'Y'},
            materialization={'m1': 'identity', 'a2': 'raw_anchors'},
        )
        params.update(overrides)
        return _make_scaler(**params)

    def test_fit_stores_the_three_divisors(self):
        """fit freezes the covariate divisors, the target divisor and the baked factor."""
        X = self._stage_frame()
        scaler = self._scaler().fit(X)
        assert (
            scaler.feature_divisors_.to_dict(), scaler.target_divisor_, scaler.fit_scale_factor_,
            scaler.n_features_in_, list(scaler.feature_names_in_),
        ) == ({'m1': 1.0, 'a2': 12.0}, 12.0, 12.0, 2, ['m1', 'a2'])

    def test_fit_returns_self(self):
        """fit returns the instance itself."""
        scaler = self._scaler()
        assert scaler.fit(self._stage_frame()) is scaler

    def test_transform_values(self):
        """transform divides 'a2' by 12 and leaves 'm1' (already at the stage scale)."""
        scaled = self._scaler().fit(self._stage_frame()).transform(self._stage_frame())
        assert scaled.to_dict('list') == {'m1': [1200.0, 2400.0, 3600.0], 'a2': [10.0, 20.0, 30.0]}

    def test_fit_transform_equals_fit_then_transform(self):
        """TransformerMixin.fit_transform gives the same frame as fit().transform()."""
        X = self._stage_frame()
        pd.testing.assert_frame_equal(
            self._scaler().fit_transform(X), self._scaler().fit(X).transform(X)
        )

    def test_fit_accepts_production_frequencies(self):
        """produced_freq feeds target_divisor_ without touching fit_scale_factor_."""
        X = self._stage_frame()
        y = pd.Series([120.0, 28.0, 30.0], index=X.index)
        scaler = self._scaler().fit(X, y, produced_freq=pd.Series(['Y', 'Q', 'Q'], index=X.index))
        assert (scaler.fit_scale_factor_, scaler.target_divisor_.tolist()) == (12.0, [12.0, 3.0, 3.0])

    def test_target_grid_defaults_to_the_grid_of_x(self):
        """Without y, the per-row target divisor runs on the grid of X."""
        X = pd.DataFrame({'m1': [1.0, 2.0]}, index=pd.to_datetime(['2024-01-31', '2024-02-29']))
        scaler = StageScaler(scale_features='calendar', source_freq='M', pred_freq='D').fit(X)
        pd.testing.assert_series_equal(scaler.target_divisor_, pd.Series([31.0, 29.0], index=X.index))

    def test_name_of_y_designates_the_imputed_column(self):
        """The name of y alone designates the imputed column whose mode applies."""
        X = self._stage_frame()
        y = pd.Series([120.0, 132.0, 150.0], index=X.index, name='a1')
        scaler = self._scaler(scale_features={'a1': False, DEFAULT_SCALE_KEY: 'constant'}).fit(X, y)
        assert (scaler.target_divisor_, scaler.fit_scale_factor_, scaler.feature_divisors_.to_dict()) == (
            1.0, 1.0, {'m1': 1.0, 'a2': 12.0}
        )

    def test_unnamed_y_falls_back_on_the_global_setting(self):
        """An anonymous target resolves no entry: the '__default__' setting applies."""
        X = self._stage_frame()
        y = pd.Series([120.0, 132.0, 150.0], index=X.index)
        scaler = self._scaler(scale_features={'a1': False, DEFAULT_SCALE_KEY: 'constant'}).fit(X, y)
        assert scaler.target_divisor_ == 12.0

    @pytest.mark.parametrize('missing', ['source_freq', 'pred_freq'])
    def test_fit_without_frequencies_raises(self, missing):
        """source_freq and pred_freq are mandatory at fit."""
        with pytest.raises(ValueError, match=missing):
            self._scaler(**{missing: None}).fit(self._stage_frame())

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_not_fitted_raises(self, method):
        """transform and inverse_transform before fit raise NotFittedError."""
        with pytest.raises(NotFittedError):
            getattr(self._scaler(), method)(self._stage_frame())

    def test_get_params_lists_every_constructor_argument(self):
        """get_params returns the seven constructor arguments, as received."""
        setting = {'m1': 'calendar'}
        params = self._scaler(scale_features=setting).get_params()
        assert sorted(params) == sorted([
            'scale_features', 'source_freq', 'pred_freq', 'column_frequencies',
            'materialization', 'default_divisor', 'converter',
        ]) and params['scale_features'] is setting

    def test_set_params_roundtrip(self):
        """set_params changes a parameter and returns the instance."""
        scaler = self._scaler()
        assert scaler.set_params(pred_freq='Q') is scaler and scaler.pred_freq == 'Q'

    def test_clone_of_a_fitted_scaler_is_unfitted_with_equal_params(self):
        """clone copies the configuration, not the fitted divisors."""
        scaler = self._scaler(scale_features={'m1': 'calendar'}).fit(self._stage_frame())
        twin = clone(scaler)
        assert twin.get_params() == scaler.get_params() and not hasattr(twin, 'feature_divisors_')

    def test_refit_replaces_every_divisor(self):
        """A second fit on another grid and stage leaves nothing of the first."""
        scaler = self._scaler()
        scaler.fit(self._stage_frame())
        scaler.set_params(pred_freq='Q', source_freq='Y', column_frequencies={'m1': 'M', 'a2': 'Y'})
        scaler.fit(self._stage_frame())
        assert (scaler.feature_divisors_.to_dict(), scaler.target_divisor_) == ({'m1': 1.0, 'a2': 4.0}, 4.0)

    def test_divisor_methods_need_no_fit(self):
        """The three divisor methods work on an unfitted, unconfigured instance."""
        scaler = StageScaler()
        features = scaler.feature_divisors(
            columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
            grid_freq='Y', stage_freq='M',
        )
        assert (
            scaler.target_divisor('a1', source_freq='Y', pred_freq='M'),
            scaler.fit_scale_factor('a1', source_freq='Y', pred_freq='M'),
            features['m1'],
        ) == (12.0, 12.0, 12.0)

    def test_one_instance_serves_every_stage(self):
        """Stateless: a shared instance gives the same divisors as fresh ones, in any order."""
        kwargs = dict(columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'}, stage_freq='M')
        shared = StageScaler()
        sequence = [shared.feature_divisors(grid_freq=grid, **kwargs)['m1'] for grid in ('Y', 'Q', 'Y')]
        fresh = [StageScaler().feature_divisors(grid_freq=grid, **kwargs)['m1'] for grid in ('Y', 'Q', 'Y')]
        assert sequence == fresh == [12.0, 3.0, 12.0]
