"""Unit tests of ``tsforecast.ensemble.online.OnlineAggregator``.

Scenarios: the adapter contract battery in meta-estimator and passthrough modes;
nested parameters; parameter and input validation; equality with the kernel run in
date order (time series, panel rounds by date, invariance to the row order);
per-entity pooling and the global weights of an unknown entity; out-of-sample
expert forecasts (equal to a manual loop over the folds, never a row seen at fit,
an overfitting expert left aside); ``partial_fit``; refitting in another mode;
resolution of the dates.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

from tests.support.adapter_cases import ensemble_meta_case, ensemble_passthrough_case, expert_forecast_data
from tests.support.adapter_contract import AdapterContractTests
from tests.support.datasets import build_adapter_panel, build_adapter_series
from tests.support.perturbations import DATETIME_UNITS, shuffle_rows, with_datetime_resolution
from tsforecast.crossvals import PanelOutOfSampleSplit
from tsforecast.ensemble import Loss, OnlineAggregator
from tsforecast.ensemble._algorithms import ALGORITHMS, make_loss, run


class TestMetaContract(AdapterContractTests):
    """Contract battery, meta-estimator over a ridge and a tree."""

    case = ensemble_meta_case()


class TestPassthroughContract(AdapterContractTests):
    """Contract battery, expert forecasts given as columns of ``X``."""

    case = ensemble_passthrough_case()


# Classe d'expert refusant de prévoir une ligne vue au fit
class _NoInSampleExpert(RegressorMixin, BaseEstimator):
    """Linear regression that raises when asked to forecast a row it was fitted on."""

    def fit(self, X, y):
        """Record the training rows and fit a linear regression."""
        self.seen_ = X.index
        self.model_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        """Forecast rows never seen at fit."""
        overlap = X.index.isin(self.seen_)
        # Le réajustement final prévoit au predict de l'agrégateur, jamais au fit
        if overlap.any() and not getattr(self, 'allow_in_sample_', False):
            raise AssertionError(f'In-sample forecast requested for {X.index[overlap][:3].tolist()}')
        return self.model_.predict(X)


# Classe d'expert renvoyant des prévisions de mauvaise forme
class _WideExpert(RegressorMixin, BaseEstimator):
    """Expert returning two columns per row."""

    def fit(self, X, y):
        """Nothing to learn."""
        return self

    def predict(self, X):
        """Two columns per row."""
        return np.zeros((len(X), 2))


# Classe de splitter ne testant aucune ligne
class _EmptySplit:
    """Splitter whose single fold tests no row."""

    def split(self, X, y=None, groups=None):
        """Yield one fold without test rows."""
        yield np.arange(len(X)), np.array([], dtype=int)


# Classe de splitter dont deux plis testent les mêmes lignes
class _OverlappingSplit:
    """Splitter whose two folds test the same last rows."""

    def split(self, X, y=None, groups=None):
        """Yield the same fold twice."""
        n = len(X)
        for _ in range(2):
            yield np.arange(n - 4), np.arange(n - 4, n)


# Fonction de construction de prévisions d'experts sur un panel, chaque expert précis sur une entité
def _entity_experts():
    """Panel ``A``, ``B`` on 36 month starts: ``p`` accurate on ``A``, ``q`` accurate on ``B``."""
    index = pd.MultiIndex.from_product(
        [['A', 'B'], pd.date_range('2021-01-01', periods=36, freq='MS')], names=['entity', 'date']
    )
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=72), index=index)
    on_a = index.get_level_values('entity') == 'A'
    X = pd.DataFrame({
        'p': y + np.where(on_a, 0.05, 2.0) * rng.normal(size=72),
        'q': y + np.where(on_a, 2.0, 0.05) * rng.normal(size=72),
    }, index=index)
    return X, y


class TestParameters:
    """Nested parameters of the experts, cloning."""

    def test_nested_expert_parameters(self):
        experts = [('ridge', Ridge(alpha=1.0)), ('mean', DummyRegressor())]
        model = OnlineAggregator(experts)
        params = model.get_params(deep=True)
        assert params['experts__ridge__alpha'] == 1.0 and params['experts__mean__strategy'] == 'mean'
        assert params['experts__ridge'] is experts[0][1]
        model.set_params(experts__ridge__alpha=5.0)
        assert model.experts[0][1].alpha == 5.0
        assert clone(model).get_params()['experts__ridge__alpha'] == 5.0

    def test_replace_an_expert_by_name(self):
        experts = [('ridge', Ridge()), ('mean', DummyRegressor())]
        model = OnlineAggregator(experts).set_params(experts__mean=DummyRegressor(strategy='median'))
        assert model.get_params()['experts__mean__strategy'] == 'median'
        # Liste reçue au constructeur intacte
        assert experts[1][1].strategy == 'mean' and model.experts is not experts

    def test_set_params_round_trip_and_unknown_expert(self):
        model = OnlineAggregator([('ridge', Ridge(alpha=3.0))])
        model.set_params(**model.get_params(deep=True))
        assert model.get_params()['experts__ridge__alpha'] == 3.0
        with pytest.raises(ValueError, match='Invalid expert name'):
            model.set_params(experts__tree__max_depth=2)

    def test_passthrough_parameters(self):
        model = OnlineAggregator('passthrough', aggregation='MLpol')
        assert model.get_params(deep=True)['experts'] == 'passthrough'
        assert clone(model).set_params(aggregation='BOA').aggregation == 'BOA'

    def test_not_fitted(self):
        X, y = expert_forecast_data('series')
        with pytest.raises(NotFittedError):
            OnlineAggregator('passthrough').predict(X)


class TestValidation:
    """Invalid parameters and inputs raise explicit errors."""

    @pytest.mark.parametrize('experts', ['average', [], [('ridge',)], [(0, Ridge())], [('a', 'b')]])
    def test_invalid_experts(self, experts):
        X, y = build_adapter_series()
        with pytest.raises(ValueError):
            OnlineAggregator(experts).fit(X, y)

    def test_duplicated_or_nested_names(self):
        X, y = build_adapter_series()
        with pytest.raises(ValueError, match='unique'):
            OnlineAggregator([('a', Ridge()), ('a', Ridge())]).fit(X, y)
        with pytest.raises(ValueError, match="'__'"):
            OnlineAggregator([('a__b', Ridge())]).fit(X, y)

    @pytest.mark.parametrize('init_weights', [[1.0, 1.0], [1.0, 0.0, 1.0], [1.0, np.nan, 1.0], 'prior'])
    def test_invalid_initial_weights(self, init_weights):
        X, y = expert_forecast_data('series')
        with pytest.raises(ValueError, match='init_weights'):
            OnlineAggregator('passthrough', init_weights=init_weights).fit(X, y)

    @pytest.mark.parametrize('cv', [0, -1, True, 'five'])
    def test_invalid_cv(self, cv):
        X, y = build_adapter_series()
        with pytest.raises(ValueError, match='cv'):
            OnlineAggregator([('ridge', Ridge())], cv=cv).fit(X, y)

    def test_panel_too_short_for_the_folds(self):
        X, y = build_adapter_panel(n_dates=4)
        with pytest.raises(ValueError, match='too few'):
            OnlineAggregator([('ridge', Ridge())], cv=5).fit(X, y)

    def test_overlapping_or_empty_internal_folds(self):
        X, y = build_adapter_series()
        with pytest.raises(ValueError, match='several folds'):
            OnlineAggregator([('ridge', Ridge())], cv=_OverlappingSplit()).fit(X, y)
        with pytest.raises(ValueError, match='tests no row'):
            OnlineAggregator([('ridge', Ridge())], cv=_EmptySplit()).fit(X, y)

    def test_expert_with_several_columns(self):
        X, y = build_adapter_series()
        with pytest.raises(ValueError, match='shape'):
            OnlineAggregator([('wide', _WideExpert())], cv=3).fit(X, y)

    def test_target_and_forecasts_must_be_finite(self):
        X, y = expert_forecast_data('series')
        # Une valeur manquante dans la cible, puis dans les prévisions d'un expert
        y_missing = y.copy()
        y_missing.iloc[3] = np.nan
        X_missing = X.copy()
        X_missing.iloc[3, 1] = np.nan
        with pytest.raises(ValueError, match='y contains'):
            OnlineAggregator('passthrough').fit(X, y_missing)
        with pytest.raises(ValueError, match='non-finite'):
            OnlineAggregator('passthrough').fit(X_missing, y)
        model = OnlineAggregator('passthrough').fit(X, y)
        with pytest.raises(ValueError, match='non-finite'):
            model.predict(X_missing)

    def test_single_target_only(self):
        X, y = expert_forecast_data('series')
        with pytest.raises(ValueError, match='single target'):
            OnlineAggregator('passthrough').fit(X, pd.DataFrame({'a': y, 'b': y}))
        # Cible en DataFrame d'une colonne acceptée
        OnlineAggregator('passthrough').fit(X, y.to_frame())

    def test_entity_levels_must_match_the_fit(self):
        X, y = expert_forecast_data('series')
        panel_X, _ = expert_forecast_data('panel')
        model = OnlineAggregator('passthrough').fit(X, y)
        with pytest.raises(ValueError, match='entity level'):
            model.predict(panel_X)


# Fonction de déroulement direct du noyau sur des prévisions d'experts triées par date
def _kernel_weights(X, y, algorithm='BOA', loss='mse', gradient=True):
    """Final weights of the kernel on ``(X, y)`` sorted by date, one round per date."""
    dates = X.index.get_level_values(-1)
    order = np.argsort(dates.to_numpy(), kind='stable')
    rounds = pd.factorize(dates[order])[0]
    functions = ALGORITHMS[algorithm]
    state, _, _ = run(
        algorithm, functions.init(np.ones(X.shape[1])), X.to_numpy()[order], y.to_numpy()[order],
        rounds, make_loss(loss), gradient,
    )
    return functions.weights(state)


class TestSequence:
    """The aggregation is the kernel run in date order."""

    @pytest.mark.parametrize('aggregation', ['BOA', 'MLpol'])
    @pytest.mark.parametrize('kind', ['series', 'panel', 'unbalanced_panel'])
    def test_equal_to_the_kernel_run_by_date(self, aggregation, kind):
        X, y = expert_forecast_data(kind)
        model = OnlineAggregator('passthrough', aggregation=aggregation, loss_gradient=False).fit(X, y)
        expected = _kernel_weights(X, y, aggregation, gradient=False)
        np.testing.assert_allclose(model.weights_.to_numpy(), expected, rtol=1e-12)
        assert model.weights_.index.tolist() == ['sharp', 'rough', 'biased']

    def test_rows_of_a_date_share_their_weights(self):
        X, y = expert_forecast_data('unbalanced_panel')
        history = OnlineAggregator('passthrough').fit(X, y).weights_history_
        dates = history.index.get_level_values('date')
        assert dates.is_monotonic_increasing
        assert (history.groupby(dates).nunique() == 1).all().all()
        assert len(history) == len(X) and set(history.index) == set(X.index)

    def test_independent_of_the_row_order(self):
        X, y = expert_forecast_data('panel')
        reference = OnlineAggregator('passthrough').fit(X, y)
        shuffled = shuffle_rows(X, seed=3)
        model = OnlineAggregator('passthrough').fit(shuffled, y.loc[shuffled.index])
        np.testing.assert_allclose(model.weights_.to_numpy(), reference.weights_.to_numpy(), rtol=1e-12)

    def test_custom_loss_and_prior(self):
        X, y = expert_forecast_data('series')
        squared = Loss(value=lambda p, t: (p - t) ** 2, gradient=lambda p, t: 2.0 * (p - t))
        custom = OnlineAggregator('passthrough', loss=squared).fit(X, y)
        builtin = OnlineAggregator('passthrough').fit(X, y)
        np.testing.assert_allclose(custom.weights_.to_numpy(), builtin.weights_.to_numpy(), rtol=1e-12)
        prior = OnlineAggregator('passthrough', init_weights=[1.0, 1.0, 8.0]).fit(X.iloc[:1], y.iloc[:1])
        np.testing.assert_allclose(prior.weights_history_.iloc[0], [0.1, 0.1, 0.8])
        np.testing.assert_allclose(prior.prior_, [0.1, 0.1, 0.8])

    def test_predict_combines_the_columns_with_the_final_weights(self):
        X, y = expert_forecast_data('series')
        model = OnlineAggregator('passthrough', aggregation='MLpol').fit(X.iloc[:30], y.iloc[:30])
        predictions = model.predict(X.iloc[30:])
        np.testing.assert_allclose(predictions, X.iloc[30:].to_numpy() @ model.weights_.to_numpy())
        assert isinstance(predictions, np.ndarray) and predictions.shape == (6,)
        assert model.weights_['sharp'] > 0.9


class TestPerEntity:
    """One aggregation per entity, the global weights for an unknown entity."""

    @pytest.mark.parametrize('aggregation', ['BOA', 'MLpol'])
    def test_each_entity_weights_its_accurate_expert(self, aggregation):
        X, y = _entity_experts()
        model = OnlineAggregator('passthrough', aggregation=aggregation, pooling='per_entity').fit(X, y)
        weights = model.entity_weights_
        assert weights.index.tolist() == ['A', 'B'] and weights.index.name == 'entity'
        assert weights.loc['A', 'p'] > 0.8 and weights.loc['B', 'q'] > 0.8
        # Agrégation globale : ni l'un ni l'autre ne domine partout
        assert model.weights_.max() < weights.loc['A', 'p']
        # Prévision : poids de l'entité de chaque ligne
        expected = (X * weights.loc[X.index.get_level_values('entity')].to_numpy()).sum(axis=1)
        np.testing.assert_allclose(model.predict(X), expected.to_numpy())

    def test_unknown_entity_takes_the_global_weights(self):
        X, y = _entity_experts()
        model = OnlineAggregator('passthrough', pooling='per_entity').fit(X, y)
        new = X.loc[['A']].rename(index={'A': 'Z'}, level='entity')
        np.testing.assert_allclose(model.predict(new), new.to_numpy() @ model.weights_.to_numpy())

    def test_entity_history_and_global_weights(self):
        X, y = _entity_experts()
        model = OnlineAggregator('passthrough', pooling='per_entity').fit(X, y)
        on_a = X.index.get_level_values('entity') == 'A'
        alone = OnlineAggregator('passthrough').fit(X[on_a].droplevel('entity'), y[on_a].droplevel('entity'))
        np.testing.assert_allclose(model.entity_weights_.loc['A'].to_numpy(), alone.weights_.to_numpy(), rtol=1e-12)
        history_a = model.weights_history_.xs('A', level='entity')
        np.testing.assert_allclose(history_a.to_numpy(), alone.weights_history_.to_numpy(), rtol=1e-12)
        global_model = OnlineAggregator('passthrough').fit(X, y)
        np.testing.assert_allclose(model.weights_.to_numpy(), global_model.weights_.to_numpy(), rtol=1e-12)

    def test_two_entity_levels(self):
        X, y = _entity_experts()
        index = pd.MultiIndex.from_arrays(
            [X.index.get_level_values(0), np.where(X.index.get_level_values(0) == 'A', 'x', 'y'),
             X.index.get_level_values(1)],
            names=['entity', 'site', 'date'],
        )
        model = OnlineAggregator('passthrough', pooling='per_entity').fit(X.set_axis(index), y.set_axis(index))
        assert model.entity_weights_.index.tolist() == [('A', 'x'), ('B', 'y')]
        assert model.entity_weights_.index.names == ['entity', 'site']

    def test_a_series_has_a_single_aggregation(self):
        X, y = expert_forecast_data('series')
        model = OnlineAggregator('passthrough', pooling='per_entity').fit(X, y)
        reference = OnlineAggregator('passthrough').fit(X, y)
        assert not hasattr(model, 'entity_weights_') and not hasattr(model, 'entity_states_')
        np.testing.assert_allclose(model.weights_.to_numpy(), reference.weights_.to_numpy())


class TestOutOfSample:
    """The meta-estimator aggregates out-of-sample expert forecasts only."""

    def test_forecasts_equal_a_manual_loop_over_the_folds(self):
        X, y = build_adapter_panel(balanced=False)
        splitter = PanelOutOfSampleSplit(n_splits=4, test_size=5, gap=1)
        experts = [('ridge', Ridge(alpha=0.5)), ('mean', DummyRegressor())]
        model = OnlineAggregator(experts, cv=splitter).fit(X, y)
        expected = pd.DataFrame(np.nan, index=X.index, columns=['ridge', 'mean'])
        for train, test in splitter.split(X, y):
            for name, expert in experts:
                fitted = clone(expert).fit(X.iloc[train], y.iloc[train])
                expected.iloc[test, expected.columns.get_loc(name)] = fitted.predict(X.iloc[test])
        expected = expected.dropna()
        pd.testing.assert_frame_equal(model.oos_predictions_, expected, check_exact=False, rtol=1e-12)
        assert set(model.weights_history_.index) == set(expected.index)

    def test_default_internal_splitters(self):
        X, y = build_adapter_series()
        series = OnlineAggregator([('ridge', Ridge())], cv=5).fit(X, y)
        # Série : 5 blocs de 36 // 6 = 6 lignes
        assert len(series.oos_predictions_) == 30
        panel_X, panel_y = build_adapter_panel()
        panel = OnlineAggregator([('ridge', Ridge())], cv=5).fit(panel_X, panel_y)
        # Panel : 5 blocs de 6 dates pour chacune des 3 entités
        assert len(panel.oos_predictions_) == 90

    def test_integer_cv_cuts_common_date_blocks_despite_holes(self):
        X, y = build_adapter_panel(balanced=False)
        holes = np.arange(len(X)) % 5 != 1
        X, y = X[holes], y[holes]
        model = OnlineAggregator([('guard', _NoInSampleExpert())], cv=3).fit(X, y)
        dates = X.index.get_level_values('date')
        calendar = dates.unique().sort_values()
        # Trois blocs de len(calendrier) // 4 dates : les dernières dates sont testées, chaque ligne une fois
        n_tested = 3 * (len(calendar) // 4)
        tested = model.oos_predictions_.index.get_level_values('date')
        assert set(tested) == set(calendar[-n_tested:])
        assert len(model.oos_predictions_) == int((dates >= calendar[-n_tested]).sum())

    def test_no_expert_forecasts_a_row_it_was_fitted_on(self):
        X, y = build_adapter_panel()
        model = OnlineAggregator([('guard', _NoInSampleExpert())], cv=4).fit(X, y)
        assert len(model.weights_history_) > 0
        # Expert réajusté sur tout l'entraînement : il prévoit les lignes suivantes
        assert model.experts_[0].seen_.equals(X.index)

    def test_an_overfitting_expert_gets_little_weight(self):
        # Mémoriseur : plus proche voisin sur un identifiant de ligne, parfait en échantillon, inutile hors échantillon
        X, y = build_adapter_series(n_dates=60)
        X = X.assign(row=np.arange(len(X), dtype=float))
        model = OnlineAggregator(
            [('ridge', Ridge(alpha=0.1)), ('memory', KNeighborsRegressor(n_neighbors=1))], cv=5
        ).fit(X, y)
        assert model.weights_['ridge'] > 0.9

    def test_experts_are_cloned_and_refitted(self):
        X, y = build_adapter_series()
        ridge = Ridge()
        model = OnlineAggregator([('ridge', ridge)], cv=3).fit(X, y)
        assert not hasattr(ridge, 'coef_') and model.experts_[0] is not ridge
        np.testing.assert_allclose(model.experts_[0].coef_, Ridge().fit(X, y).coef_)

    def test_parallel_fits_give_the_same_weights(self):
        X, y = build_adapter_series()
        experts = [('ridge', Ridge()), ('mean', DummyRegressor())]
        sequential = OnlineAggregator(experts, cv=3).fit(X, y)
        parallel = OnlineAggregator(experts, cv=3, n_jobs=2).fit(X, y)
        pd.testing.assert_series_equal(parallel.weights_, sequential.weights_)


class TestPartialFit:
    """``partial_fit`` continues the sequence."""

    @pytest.mark.parametrize('pooling', ['global', 'per_entity'])
    @pytest.mark.parametrize('aggregation', ['BOA', 'MLpol'])
    def test_fit_then_partial_fit_equals_a_single_fit(self, pooling, aggregation):
        X, y = expert_forecast_data('unbalanced_panel')
        dates = X.index.get_level_values('date')
        first = dates < pd.Timestamp('2021-06-01')
        params = {'aggregation': aggregation, 'pooling': pooling}
        online = OnlineAggregator('passthrough', **params).fit(X[first], y[first]).partial_fit(X[~first], y[~first])
        batch = OnlineAggregator('passthrough', **params).fit(X, y)
        pd.testing.assert_series_equal(online.weights_, batch.weights_)
        pd.testing.assert_frame_equal(online.weights_history_, batch.weights_history_)
        if pooling == 'per_entity':
            pd.testing.assert_frame_equal(online.entity_weights_, batch.entity_weights_)
        assert online.last_date_ == dates.max()

    def test_new_entity_starts_from_the_initial_weights(self):
        X, y = _entity_experts()
        on_a = X.index.get_level_values('entity') == 'A'
        dates = X.index.get_level_values('date')
        late = on_a & (dates > dates[5]) | ~on_a & (dates > dates[5])
        model = OnlineAggregator('passthrough', pooling='per_entity').fit(X[on_a & ~late], y[on_a & ~late])
        assert model.entity_weights_.index.tolist() == ['A']
        model.partial_fit(X[late], y[late])
        assert model.entity_weights_.index.tolist() == ['A', 'B']
        first_b = model.weights_history_.xs('B', level='entity').iloc[0]
        np.testing.assert_allclose(first_b.to_numpy(), [0.5, 0.5])

    def test_rows_must_follow_the_aggregated_rows(self):
        X, y = expert_forecast_data('series')
        model = OnlineAggregator('passthrough').fit(X.iloc[:20], y.iloc[:20])
        with pytest.raises(ValueError, match='after the last aggregated date'):
            model.partial_fit(X.iloc[19:25], y.iloc[19:25])

    def test_before_fit_partial_fit_is_fit(self):
        X, y = expert_forecast_data('series')
        model = OnlineAggregator('passthrough').partial_fit(X, y)
        pd.testing.assert_series_equal(model.weights_, OnlineAggregator('passthrough').fit(X, y).weights_)

    def test_empty_batch_changes_nothing(self):
        X, y = expert_forecast_data('series')
        model = OnlineAggregator('passthrough').fit(X, y)
        weights = model.weights_.copy()
        model.partial_fit(X.iloc[:0], y.iloc[:0])
        pd.testing.assert_series_equal(model.weights_, weights)

    def test_changed_algorithm_is_refused(self):
        X, y = expert_forecast_data('series')
        model = OnlineAggregator('passthrough').fit(X.iloc[:20], y.iloc[:20]).set_params(aggregation='MLpol')
        with pytest.raises(ValueError, match='aggregation changed'):
            model.partial_fit(X.iloc[20:], y.iloc[20:])

    def test_meta_mode_keeps_its_experts(self):
        X, y = build_adapter_series()
        model = OnlineAggregator([('ridge', Ridge()), ('mean', DummyRegressor())], cv=3).fit(X.iloc[:24], y.iloc[:24])
        experts = list(model.experts_)
        n_rows = len(model.oos_predictions_)
        model.partial_fit(X.iloc[24:], y.iloc[24:])
        assert all(after is before for after, before in zip(model.experts_, experts))
        # Prévisions des experts ajustés au fit, ajoutées aux prévisions agrégées
        assert len(model.oos_predictions_) == n_rows + 12
        np.testing.assert_allclose(model.oos_predictions_['ridge'].iloc[-12:], experts[0].predict(X.iloc[24:]))
        assert len(model.weights_history_) == n_rows + 12


class TestRefit:
    """A new fit forgets the attributes of the previous one."""

    def test_refit_in_another_mode(self):
        X, y = _entity_experts()
        model = OnlineAggregator([('ridge', Ridge())], pooling='per_entity', cv=3).fit(X, y)
        assert hasattr(model, 'experts_') and hasattr(model, 'entity_weights_')
        model.set_params(experts='passthrough', pooling='global').fit(X, y)
        for attribute in ('experts_', 'oos_predictions_', 'entity_weights_', 'entity_states_'):
            assert not hasattr(model, attribute), attribute


class TestDatetimeResolution:
    """Returned dates keep the unit of the input; weights do not depend on it."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_units(self, unit):
        X, y = expert_forecast_data('panel')
        reference = OnlineAggregator('passthrough').fit(X, y)
        X_unit, y_unit = with_datetime_resolution(X, unit), with_datetime_resolution(y, unit)
        model = OnlineAggregator('passthrough').fit(X_unit, y_unit)
        np.testing.assert_allclose(model.weights_.to_numpy(), reference.weights_.to_numpy(), rtol=1e-12)
        assert model.last_date_.unit == unit
        assert model.weights_history_.index.get_level_values('date').unit == unit
