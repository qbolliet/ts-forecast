"""Safety and coverage properties of ``OutOfSampleSplit`` and ``InSampleSplit``.

Checks, on every fold of a set of configurations over 24 month-end dates:
no leakage and an exact ``gap`` (out-of-sample), the expanding or rolling
training window, disjoint and successive test sets of the documented size,
the in-sample contract, ``get_n_splits`` against the folds actually
produced (including configurations whose ``split`` raises), ``split`` as a
reproducible generator independent of the container of ``X``, and the
absence of any change to the input.
"""
# Importation des modules
import copy
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from tests.support.splits import (
    as_lists,
    assert_folds_cover,
    assert_in_sample,
    assert_n_splits_consistent,
    assert_no_leakage,
    assert_train_window,
    assert_valid_folds,
    collect_folds,
)
from tsforecast.crossvals.base import InSampleSplit, OutOfSampleSplit

N_ROWS = 24

# Configurations hors échantillon par défaut : (paramètres, taille de test attendue)
# Valeur d'or : sans test_size, la taille vaut 24 // (n_splits + 1)
DEFAULT_OOS = {
    'expanding': ({'n_splits': 3}, 6),
    'expanding-test_size-2': ({'n_splits': 4, 'test_size': 2}, 2),
    'gap-2': ({'n_splits': 3, 'gap': 2}, 6),
    'rolling-3': ({'n_splits': 5, 'max_train_size': 3}, 4),
    'rolling-gap': ({'n_splits': 2, 'test_size': 3, 'gap': 1, 'max_train_size': 4}, 3),
    'max-splits': ({'n_splits': 23}, 1),
}

# Configurations hors échantillon à indices de test explicites
EXPLICIT_OOS = {
    'dates': {'test_indices': ['2023-06-30', '2024-01-31', '2024-06-30'], 'test_size': 2, 'gap': 1},
    'positions-rolling': {'test_indices': [5, 15], 'test_size': 4, 'max_train_size': 6},
}

ALL_OOS = {**{key: kwargs for key, (kwargs, _) in DEFAULT_OOS.items()}, **EXPLICIT_OOS}

IN_SAMPLE = {
    'default': {},
    'n_splits-3': {'n_splits': 3},
    'test_size-5': {'test_size': 5},
    'window': {'test_size': 3, 'max_train_size': 6},
    'window-shorter-than-test': {'test_size': 6, 'max_train_size': 2},
    'single-date': {'test_indices': ['2023-06-30'], 'test_size': 4},
    'single-window': {'test_indices': [10], 'test_size': 3, 'max_train_size': 5},
    'several-dates': {'test_indices': ['2023-03-31', '2023-09-30', '2024-02-29']},
}


def _params(configs: dict) -> list:
    """Turn a configuration dictionary into ``pytest.param`` items with readable ids."""
    return [pytest.param(value, id=key) for key, value in configs.items()]


class TestNoLeakage:
    """Out-of-sample training dates strictly precede test dates by exactly gap dates."""

    @pytest.mark.parametrize('kwargs', _params(ALL_OOS))
    def test_training_precedes_the_test_by_the_gap(self, cv_frame_24, kwargs):
        """Every fold respects the gap exactly, measured on dates."""
        folds = collect_folds(OutOfSampleSplit(**kwargs), cv_frame_24)
        assert_no_leakage(cv_frame_24, folds, gap=kwargs.get('gap', 0))

    @pytest.mark.parametrize('kwargs', _params(ALL_OOS))
    def test_folds_are_valid_positions(self, cv_frame_24, kwargs):
        """Positions are integer, in range and not repeated."""
        assert_valid_folds(cv_frame_24, collect_folds(OutOfSampleSplit(**kwargs), cv_frame_24))

    def test_panel_folds_per_entity(self, cv_panel_2x6):
        """On a panel, the gap and the window hold inside each entity."""
        groups = cv_panel_2x6.index.get_level_values(0).to_numpy()
        splitter = OutOfSampleSplit(n_splits=2, test_size=2, gap=1, max_train_size=2)
        folds = collect_folds(splitter, cv_panel_2x6, groups=groups)
        assert_no_leakage(cv_panel_2x6, folds, gap=1, groups=groups)
        assert_train_window(cv_panel_2x6, folds, gap=1, max_train_size=2, groups=groups)


class TestTrainingWindow:
    """Expanding window without max_train_size, rolling window with it."""

    @pytest.mark.parametrize('kwargs', _params(ALL_OOS))
    def test_window_matches_the_parameters(self, cv_frame_24, kwargs):
        """Training is the run of dates ending gap dates before the test, of the expected length."""
        folds = collect_folds(OutOfSampleSplit(**kwargs), cv_frame_24)
        assert_train_window(cv_frame_24, folds, gap=kwargs.get('gap', 0),
                            max_train_size=kwargs.get('max_train_size'))

    def test_expanding_sizes_grow_by_test_size(self, cv_frame_24):
        """Without a limit, each fold trains on test_size more dates than the previous."""
        folds = collect_folds(OutOfSampleSplit(n_splits=3), cv_frame_24)
        assert [len(train) for train, _ in folds] == [6, 12, 18]

    def test_rolling_sizes_are_constant(self, cv_frame_24):
        """With enough history, every fold trains on max_train_size dates."""
        folds = collect_folds(OutOfSampleSplit(n_splits=5, max_train_size=3), cv_frame_24)
        assert [len(train) for train, _ in folds] == [3] * 5


class TestCoverage:
    """Test sets: disjoint, successive, of the documented size and number."""

    @pytest.mark.parametrize(('kwargs', 'test_size'), [
        pytest.param(kwargs, size, id=key) for key, (kwargs, size) in DEFAULT_OOS.items()
    ])
    def test_default_folds_tile_the_end_of_the_data(self, cv_frame_24, kwargs, test_size):
        """n_splits adjacent test sets of test_size rows end at the last date."""
        folds = collect_folds(OutOfSampleSplit(**kwargs), cv_frame_24)
        assert len(folds) == kwargs['n_splits']
        assert_folds_cover(folds, test_size=test_size, n_samples=N_ROWS)

    @pytest.mark.parametrize('kwargs', _params(EXPLICIT_OOS))
    def test_explicit_folds_follow_the_test_indices(self, cv_frame_24, kwargs):
        """One fold per test index, in date order, each of test_size rows."""
        folds = collect_folds(OutOfSampleSplit(**kwargs), cv_frame_24)
        assert len(folds) == len(kwargs['test_indices'])
        assert_folds_cover(folds, test_size=kwargs['test_size'], contiguous=False)


class TestInSampleContract:
    """One fold whose training holds the test rows and ends at the last test date."""

    @pytest.mark.parametrize('kwargs', _params(IN_SAMPLE))
    def test_contract_holds(self, cv_frame_24, kwargs):
        """Test rows are trained on; training is contiguous and stops at the last test date."""
        folds = collect_folds(InSampleSplit(**kwargs), cv_frame_24)
        assert len(folds) == 1
        assert_in_sample(cv_frame_24, folds)
        assert_valid_folds(cv_frame_24, folds)

    def test_panel_contract(self, cv_panel_2x6):
        """On a panel, the contract holds inside each entity."""
        groups = cv_panel_2x6.index.get_level_values(0).to_numpy()
        folds = collect_folds(InSampleSplit(test_indices=['2024-03-31'], test_size=2, max_train_size=3),
                              cv_panel_2x6, groups=groups)
        assert_in_sample(cv_panel_2x6, folds, groups=groups)


class TestNSplitsConsistency:
    """get_n_splits announces the folds that split produces."""

    @pytest.mark.parametrize('kwargs', _params({key: kwargs for key, (kwargs, _) in DEFAULT_OOS.items()}))
    def test_default_out_of_sample_folds(self, cv_frame_24, kwargs):
        """Default out-of-sample folds: n_splits announced, n_splits produced."""
        assert_n_splits_consistent(OutOfSampleSplit(**kwargs), cv_frame_24)

    def test_without_arguments_returns_n_splits(self):
        """Called without data, get_n_splits returns n_splits."""
        assert OutOfSampleSplit(n_splits=4).get_n_splits() == 4

    @pytest.mark.parametrize('splitter', [
        *[pytest.param(OutOfSampleSplit(**kwargs), id=f'oos-{key}') for key, kwargs in EXPLICIT_OOS.items()],
        pytest.param(OutOfSampleSplit(test_indices=[5, 5, 9]), id='oos-repeated-test-index'),
        pytest.param(OutOfSampleSplit(test_indices=10), id='oos-scalar'),
        pytest.param(OutOfSampleSplit(n_splits=3, test_indices=[]), id='oos-empty-list'),
        *[pytest.param(InSampleSplit(**kwargs), id=f'is-{key}') for key, kwargs in IN_SAMPLE.items()],
    ])
    def test_other_configurations(self, cv_frame_24, splitter):
        """Explicit test indices and in-sample splits announce their real fold count."""
        assert_n_splits_consistent(splitter, cv_frame_24)

    @pytest.mark.parametrize('splitter', [
        pytest.param(OutOfSampleSplit(n_splits=2), id='oos-default'),
        pytest.param(OutOfSampleSplit(test_indices=['2024-03-31', '2024-05-31']), id='oos-dates'),
        pytest.param(OutOfSampleSplit(test_indices=[('A', '2024-04-30'), ('B', '2024-05-31')]), id='oos-tuples'),
        pytest.param(OutOfSampleSplit(test_indices=[3, 10]), id='oos-positions'),
        pytest.param(InSampleSplit(test_indices=[2]), id='is-single-position'),
    ])
    def test_panel_configurations(self, cv_panel_2x6, splitter):
        """On a panel, the count is the number of dates, tuples or positions asked for."""
        groups = cv_panel_2x6.index.get_level_values(0).to_numpy()
        assert_n_splits_consistent(splitter, cv_panel_2x6, groups=groups)

    @pytest.mark.parametrize(('splitter', 'X', 'expected'), [
        pytest.param(OutOfSampleSplit(n_splits=24), np.zeros(N_ROWS), 24, id='too-many-folds'),
        pytest.param(OutOfSampleSplit(n_splits=2, test_size=13), np.zeros(N_ROWS), 2, id='test_size-too-large'),
        pytest.param(OutOfSampleSplit(n_splits=2), np.zeros(0), 2, id='oos-empty'),
        pytest.param(OutOfSampleSplit(n_splits=3, gap=30), np.zeros(N_ROWS), 3, id='no-training-data'),
        pytest.param(InSampleSplit(), np.zeros(0), 1, id='in-sample-empty'),
    ])
    def test_count_does_not_read_the_data(self, splitter, X, expected):
        """As for sklearn splitters, data that split refuses still gets the parameter count."""
        assert splitter.get_n_splits(X) == expected
        with pytest.raises(ValueError):
            collect_folds(splitter, X)


class TestGenerator:
    """split is a reproducible generator, independent of the container of X."""

    def test_split_returns_a_generator(self, cv_frame_24):
        """split returns a generator object."""
        assert inspect.isgenerator(OutOfSampleSplit(n_splits=3).split(cv_frame_24))

    def test_generator_is_single_use(self, cv_frame_24):
        """A consumed generator yields nothing more."""
        generator = OutOfSampleSplit(n_splits=3).split(cv_frame_24)
        list(generator)
        assert list(generator) == []

    @pytest.mark.parametrize(('cls', 'kwargs'), [
        *[pytest.param(OutOfSampleSplit, kwargs, id=f'oos-{key}') for key, kwargs in ALL_OOS.items()],
        *[pytest.param(InSampleSplit, kwargs, id=f'is-{key}') for key, kwargs in IN_SAMPLE.items()],
    ])
    def test_two_calls_give_the_same_folds(self, cv_frame_24, cls, kwargs):
        """Two calls on the same data yield identical folds."""
        splitter = cls(**kwargs)
        assert as_lists(collect_folds(splitter, cv_frame_24)) == as_lists(collect_folds(splitter, cv_frame_24))

    @pytest.mark.parametrize('cls', [OutOfSampleSplit, InSampleSplit], ids=['oos', 'is'])
    def test_explicit_none_equals_omitted_arguments(self, cv_frame_24, cls):
        """y=None and groups=None are the defaults."""
        splitter = cls(n_splits=3)
        explicit = as_lists(splitter.split(cv_frame_24, y=None, groups=None))
        assert explicit == as_lists(splitter.split(cv_frame_24))

    @pytest.mark.parametrize('to_input', [
        pytest.param(lambda X: X, id='DataFrame'),
        pytest.param(lambda X: X['v'], id='Series'),
        pytest.param(lambda X: X.to_numpy(), id='ndarray-2d'),
        pytest.param(lambda X: X['v'].to_numpy(), id='ndarray-1d'),
    ])
    @pytest.mark.parametrize('cls', [OutOfSampleSplit, InSampleSplit], ids=['oos', 'is'])
    def test_container_does_not_change_the_folds(self, cv_frame_24, cls, to_input):
        """A Series, a DataFrame and arrays of the same rows give the same folds."""
        splitter = cls(n_splits=3)
        assert as_lists(collect_folds(splitter, to_input(cv_frame_24))) == as_lists(
            collect_folds(splitter, cv_frame_24)
        )


class TestNonMutation:
    """split leaves its input untouched."""

    @pytest.mark.parametrize('cls', [OutOfSampleSplit, InSampleSplit], ids=['oos', 'is'])
    @pytest.mark.parametrize('order', ['sorted', 'unsorted'])
    def test_input_is_left_intact(self, cv_frame_24, cls, order):
        """X and y are identical before and after split, sorted or not."""
        X = cv_frame_24 if order == 'sorted' else cv_frame_24.iloc[::-1]
        y = X['v'] * 2
        X_before, y_before = copy.deepcopy(X), copy.deepcopy(y)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            collect_folds(cls(n_splits=3), X, y)
        pd.testing.assert_frame_equal(X, X_before)
        pd.testing.assert_series_equal(y, y_before)
