"""Unit tests of ``tsforecast.adapters._base``, the toolkit shared by the adapters.

Covers ``import_optional`` (installed module, missing package or submodule,
missing transitive dependency propagated), ``TemporalLayout`` (series, panels
with one or two entity levels, unsorted and unbalanced rows, refused labels,
frequency detection, datetime resolution, ``last_dates``, ``match_entities``),
``steps_between`` (grids, anchors, multipliers, off-grid dates),
``restore_row_order`` and ``check_adapter_input``.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from tests.support.perturbations import DATETIME_UNITS, shuffle_rows, with_datetime_resolution
from tsforecast.adapters._base import (
    TemporalLayout,
    check_adapter_input,
    import_optional,
    restore_row_order,
    steps_between,
)


def month_starts(periods: int, start: str = '2024-01-01') -> pd.DatetimeIndex:
    """Month starts from ``start``, named ``date``."""
    return pd.date_range(start, periods=periods, freq='MS', name='date')


def panel_index(entities, dates, names=('entity', 'date')) -> pd.MultiIndex:
    """``(entity, date)`` index, entity by entity."""
    return pd.MultiIndex.from_product([entities, dates], names=list(names))


# =============================================================================
# import_optional
# =============================================================================


class TestImportOptional:
    """Deferred import with an installation hint limited to the missing package itself."""

    def test_installed_module_is_returned(self):
        assert import_optional('json', 'adapters-x').dumps({'a': 1}) == '{"a": 1}'

    def test_installed_submodule_is_returned(self):
        module = import_optional('os.path', 'adapters-x')
        assert module.join('a', 'b').replace('\\', '/') == 'a/b'

    def test_missing_package_names_package_and_extra(self):
        with pytest.raises(ImportError, match=r"'tsf_absent_pkg' package.*ts-forecast\[adapters-x\]") as raised:
            import_optional('tsf_absent_pkg', 'adapters-x')
        assert isinstance(raised.value.__cause__, ModuleNotFoundError)

    def test_missing_submodule_of_installed_package(self):
        with pytest.raises(ImportError, match=r"Module 'json\.tsf_absent'.*'json' package"):
            import_optional('json.tsf_absent', 'adapters-x')

    def test_missing_parent_of_submodule(self):
        with pytest.raises(ImportError, match=r"'tsf_absent_pkg' package"):
            import_optional('tsf_absent_pkg.sub.module', 'adapters-x')

    def test_blocked_package(self, tmp_path, monkeypatch, block_package):
        (tmp_path / 'tsf_fake_pkg').mkdir()
        (tmp_path / 'tsf_fake_pkg' / '__init__.py').write_text('VALUE = 1\n')
        monkeypatch.syspath_prepend(str(tmp_path))
        assert import_optional('tsf_fake_pkg', 'adapters-x').VALUE == 1
        block_package('tsf_fake_pkg')
        with pytest.raises(ImportError, match=r'adapters-x'):
            import_optional('tsf_fake_pkg', 'adapters-x')

    def test_missing_transitive_dependency_propagates(self, tmp_path, monkeypatch):
        (tmp_path / 'tsf_needs_dep').mkdir()
        (tmp_path / 'tsf_needs_dep' / '__init__.py').write_text('import tsf_absent_dependency\n')
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(ModuleNotFoundError) as raised:
            import_optional('tsf_needs_dep', 'adapters-x')
        assert raised.value.name == 'tsf_absent_dependency'
        assert 'ts-forecast[' not in str(raised.value)

    def test_error_raised_by_the_package_propagates(self, tmp_path, monkeypatch):
        (tmp_path / 'tsf_broken_pkg').mkdir()
        (tmp_path / 'tsf_broken_pkg' / '__init__.py').write_text("raise ImportError('broken build')\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        with pytest.raises(ImportError, match='^broken build$'):
            import_optional('tsf_broken_pkg', 'adapters-x')


# =============================================================================
# TemporalLayout
# =============================================================================


class TestTemporalLayoutStructure:
    """Entities, codes and dates read from series and panel indexes."""

    def test_series(self):
        layout = TemporalLayout.from_index(month_starts(4))
        assert not layout.is_panel
        assert layout.entity_names == ()
        assert layout.time_name == 'date'
        assert layout.entities == ((),)
        assert layout.n_entities == 1
        assert layout.codes.tolist() == [0, 0, 0, 0]
        assert layout.dates.equals(month_starts(4))
        assert layout.freq == 'MS'

    def test_panel_one_entity_level(self):
        layout = TemporalLayout.from_index(panel_index(['FR', 'DE'], month_starts(3)))
        assert layout.is_panel
        assert layout.entity_names == ('entity',)
        assert layout.entities == (('FR',), ('DE',))
        assert layout.codes.tolist() == [0, 0, 0, 1, 1, 1]
        assert layout.freq == 'MS'

    def test_panel_two_entity_levels(self):
        index = pd.MultiIndex.from_product(
            [['EU'], ['FR', 'DE'], month_starts(2)], names=['region', 'country', 'date']
        )
        layout = TemporalLayout.from_index(index)
        assert layout.entity_names == ('region', 'country')
        assert layout.entities == (('EU', 'FR'), ('EU', 'DE'))
        assert layout.codes.tolist() == [0, 0, 1, 1]

    def test_unsorted_rows(self):
        frame = pd.DataFrame({'v': range(6)}, index=panel_index(['A', 'B'], month_starts(3)))
        shuffled = shuffle_rows(frame, seed=3)
        layout = TemporalLayout.from_index(shuffled.index)
        assert layout.freq == 'MS'
        # Codes dans l'ordre de première apparition, cohérents ligne à ligne
        entities = [layout.entities[code][0] for code in layout.codes]
        assert entities == shuffled.index.get_level_values('entity').tolist()
        assert layout.dates.equals(pd.DatetimeIndex(shuffled.index.get_level_values('date')))

    def test_unbalanced_panel(self):
        index = pd.MultiIndex.from_tuples(
            [('A', d) for d in month_starts(6)] + [('B', d) for d in month_starts(2, '2024-05-01')],
            names=['entity', 'date'],
        )
        layout = TemporalLayout.from_index(index)
        assert layout.freq == 'MS'
        assert layout.last_dates().equals(pd.DatetimeIndex(['2024-06-01', '2024-06-01'], name='date'))

    def test_codes_are_read_only(self):
        layout = TemporalLayout.from_index(month_starts(3))
        with pytest.raises(ValueError):
            layout.codes[0] = 1


class TestTemporalLayoutRefusals:
    """Labels that cannot be analysed are refused explicitly."""

    def test_period_index(self):
        with pytest.raises(TypeError, match='to_timestamp'):
            TemporalLayout.from_index(pd.period_range('2024-01', periods=3, freq='M'))

    def test_period_level(self):
        index = pd.MultiIndex.from_product([['A'], pd.period_range('2024-01', periods=2, freq='M')])
        with pytest.raises(TypeError, match='to_timestamp'):
            TemporalLayout.from_index(index)

    def test_integer_index(self):
        with pytest.raises(TypeError, match='datetime64'):
            TemporalLayout.from_index(pd.RangeIndex(3))

    def test_single_level_multiindex(self):
        with pytest.raises(TypeError, match='entity levels'):
            TemporalLayout.from_index(pd.MultiIndex.from_arrays([month_starts(2)]))

    def test_missing_date(self):
        with pytest.raises(ValueError, match='NaT'):
            TemporalLayout.from_index(pd.DatetimeIndex(['2024-01-01', None]))

    def test_missing_entity_label(self):
        index = pd.MultiIndex.from_arrays([['A', np.nan], pd.to_datetime(['2024-01-01', '2024-01-01'])])
        with pytest.raises(ValueError):
            TemporalLayout.from_index(index)

    def test_duplicated_rows(self):
        index = panel_index(['A'], month_starts(2)).append(panel_index(['A'], month_starts(1)))
        with pytest.raises(ValueError, match='Duplicated'):
            TemporalLayout.from_index(index)

    def test_duplicated_dates_of_a_series(self):
        with pytest.raises(ValueError, match='Duplicated'):
            TemporalLayout.from_index(pd.DatetimeIndex(['2024-01-01', '2024-01-01']))


class TestTemporalLayoutFrequency:
    """Grid frequency: imposed, detected in common, or unknown."""

    def test_imposed_frequency_is_normalized(self):
        assert TemporalLayout.from_index(month_starts(1), freq='M').freq == 'ME'

    def test_imposed_frequency_skips_detection(self):
        # Une seule date : rien à détecter, la fréquence imposée suffit
        assert TemporalLayout.from_index(month_starts(1), freq='QE-DEC').freq == 'QE-DEC'

    def test_invalid_imposed_frequency(self):
        with pytest.raises(ValueError):
            TemporalLayout.from_index(month_starts(2), freq='not-a-frequency')

    def test_detection_disabled(self):
        assert TemporalLayout.from_index(month_starts(3), detect_freq=False).freq is None

    @pytest.mark.parametrize('freq', ['QE-DEC', '2MS', 'W-SUN', 'B', 'h'])
    def test_detected_frequencies(self, freq):
        dates = pd.date_range('2024-01-01', periods=8, freq=freq)
        assert TemporalLayout.from_index(dates).freq == freq

    def test_single_date_entities_are_ignored(self):
        index = pd.MultiIndex.from_tuples(
            [('A', d) for d in month_starts(4)] + [('B', pd.Timestamp('2024-02-01'))]
        )
        assert TemporalLayout.from_index(index).freq == 'MS'

    def test_only_single_date_entities(self):
        index = pd.MultiIndex.from_tuples([('A', pd.Timestamp('2024-01-01')), ('B', pd.Timestamp('2024-02-01'))])
        assert TemporalLayout.from_index(index).freq is None

    def test_entities_on_different_grids(self):
        index = pd.MultiIndex.from_tuples(
            [('A', d) for d in month_starts(4)]
            + [('B', d) for d in pd.date_range('2024-01-01', periods=4, freq='QS')]
        )
        with pytest.raises(ValueError, match='common time grid'):
            TemporalLayout.from_index(index)


class TestTemporalLayoutDatetimeResolution:
    """Same instants in s, ms, us and ns: same layout, dates kept in the input unit."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    def test_unit_is_kept(self, unit):
        frame = pd.DataFrame({'v': range(6)}, index=panel_index(['A', 'B'], month_starts(3)))
        converted = with_datetime_resolution(frame, unit)
        layout = TemporalLayout.from_index(converted.index)
        assert layout.unit == unit
        assert layout.freq == 'MS'
        assert layout.last_dates().unit == unit
        assert layout.codes.tolist() == [0, 0, 0, 1, 1, 1]


class TestLastDatesAndMatchEntities:
    """Last date of each entity and location of the entities of another layout."""

    def test_last_dates_of_unsorted_panel(self):
        index = pd.MultiIndex.from_arrays(
            [['B', 'A', 'B', 'A'], pd.to_datetime(['2024-03-01', '2024-02-01', '2024-01-01', '2024-01-01'])],
            names=['entity', 'date'],
        )
        last = TemporalLayout.from_index(index).last_dates()
        assert last.strftime('%Y-%m').tolist() == ['2024-03', '2024-02']
        assert last.name == 'date'

    def test_last_date_of_series(self):
        assert TemporalLayout.from_index(month_starts(3)).last_dates().tolist() == [pd.Timestamp('2024-03-01')]

    def test_match_known_and_unknown_entities(self):
        train = TemporalLayout.from_index(panel_index(['A', 'B'], month_starts(2)))
        test = TemporalLayout.from_index(
            pd.MultiIndex.from_arrays([['C', 'B', 'A', 'B'], month_starts(4)]), detect_freq=False
        )
        assert train.match_entities(test).tolist() == [-1, 1, 0, 1]

    def test_match_two_entity_levels(self):
        dates = month_starts(1)
        train = TemporalLayout.from_index(pd.MultiIndex.from_product([['EU'], ['FR', 'DE'], dates]))
        test = TemporalLayout.from_index(pd.MultiIndex.from_product([['EU', 'US'], ['DE'], dates]))
        assert train.match_entities(test).tolist() == [1, -1]

    def test_match_series(self):
        train = TemporalLayout.from_index(month_starts(3))
        test = TemporalLayout.from_index(month_starts(2, '2024-06-01'))
        assert train.match_entities(test).tolist() == [0, 0]

    def test_match_different_entity_levels(self):
        train = TemporalLayout.from_index(panel_index(['A'], month_starts(2)))
        with pytest.raises(ValueError, match='entity level'):
            train.match_entities(TemporalLayout.from_index(month_starts(2)))


# =============================================================================
# steps_between
# =============================================================================


class TestStepsBetween:
    """Grid steps between two dates, vectorized."""

    def test_monthly(self):
        start = pd.to_datetime(['2024-01-01', '2024-03-01', '2024-03-01'])
        end = pd.to_datetime(['2024-02-01', '2024-03-01', '2023-12-01'])
        steps = steps_between(start, end, 'MS')
        assert steps.tolist() == [1, 0, -3]
        assert steps.dtype == np.int64

    def test_anchored_quarter(self):
        steps = steps_between(pd.to_datetime(['2023-12-31']), pd.to_datetime(['2024-09-30']), 'QE-DEC')
        assert steps.tolist() == [3]

    def test_multiplied_frequency(self):
        start = pd.to_datetime(['2024-01-01', '2024-01-01'])
        end = pd.to_datetime(['2024-07-01', '2024-03-01'])
        assert steps_between(start, end, '2MS').tolist() == [3, 1]

    def test_date_off_a_multiplied_grid(self):
        with pytest.raises(ValueError, match='not on the'):
            steps_between(pd.to_datetime(['2024-01-01']), pd.to_datetime(['2024-02-01']), '2MS')

    def test_weekly_anchor(self):
        assert steps_between(pd.to_datetime(['2024-01-07']), pd.to_datetime(['2024-02-04']), 'W-SUN').tolist() == [4]

    def test_business_days(self):
        # Vendredi -> lundi : un jour ouvré
        assert steps_between(pd.to_datetime(['2024-01-05']), pd.to_datetime(['2024-01-08']), 'B').tolist() == [1]

    def test_base_code_frequency(self):
        assert steps_between(pd.to_datetime(['2024-01-31']), pd.to_datetime(['2024-04-30']), 'M').tolist() == [3]

    def test_off_grid_date(self):
        with pytest.raises(ValueError, match="2024-02-15"):
            steps_between(pd.to_datetime(['2024-01-01']), pd.to_datetime(['2024-02-15']), 'MS')

    def test_missing_date(self):
        with pytest.raises(ValueError, match='NaT'):
            steps_between(pd.to_datetime(['2024-01-01']), pd.DatetimeIndex([None]), 'MS')

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match='same length'):
            steps_between(month_starts(2), month_starts(1), 'MS')

    def test_empty(self):
        steps = steps_between(month_starts(0), month_starts(0), 'MS')
        assert steps.shape == (0,) and steps.dtype == np.int64

    def test_mixed_units(self):
        start = month_starts(2).as_unit('s')
        end = month_starts(2, '2024-05-01').as_unit('ns')
        assert steps_between(start, end, 'MS').tolist() == [4, 4]

    def test_series_and_arrays(self):
        start = pd.Series(month_starts(2))
        end = month_starts(2, '2024-02-01').to_numpy()
        assert steps_between(start, end, 'MS').tolist() == [1, 1]


# =============================================================================
# restore_row_order
# =============================================================================


class TestRestoreRowOrder:
    """Realignment of an output on the rows of X."""

    def test_reversed_series(self):
        dates = month_starts(3)
        assert restore_row_order(np.array([1.0, 2.0, 3.0]), dates, dates[::-1]).tolist() == [3.0, 2.0, 1.0]

    def test_shuffled_panel(self):
        frame = pd.DataFrame({'v': np.arange(6.0)}, index=panel_index(['A', 'B'], month_starts(3)))
        shuffled = shuffle_rows(frame, seed=4)
        restored = restore_row_order(frame['v'], frame.index, shuffled.index)
        assert restored.tolist() == shuffled['v'].tolist()

    def test_two_dimensional_values(self):
        values = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=month_starts(2))
        restored = restore_row_order(values, values.index, values.index[::-1])
        assert restored.tolist() == [[2, 4], [1, 3]]

    def test_subset_of_rows(self):
        dates = month_starts(4)
        assert restore_row_order(np.arange(4), dates, dates[[2]]).tolist() == [2]

    def test_different_units(self):
        frame = pd.DataFrame({'v': np.arange(4.0)}, index=panel_index(['A', 'B'], month_starts(2)))
        source = with_datetime_resolution(frame, 'ns').index
        target = with_datetime_resolution(frame, 's').index[::-1]
        assert restore_row_order(frame['v'], source, target).tolist() == [3.0, 2.0, 1.0, 0.0]

    def test_result_is_writable(self):
        dates = month_starts(2)
        values = pd.Series([1.0, 2.0], index=dates).to_numpy()
        restored = restore_row_order(values, dates, dates)
        restored[0] = 5.0
        assert values[0] == 1.0

    def test_missing_row(self):
        with pytest.raises(ValueError, match='1 row.*missing'):
            restore_row_order(np.arange(2), month_starts(2), month_starts(3))

    def test_duplicated_source(self):
        dates = pd.DatetimeIndex(['2024-01-01', '2024-01-01'])
        with pytest.raises(ValueError, match='duplicated'):
            restore_row_order(np.arange(2), dates, dates[:1])

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match='rows but source_index'):
            restore_row_order(np.arange(3), month_starts(2), month_starts(2))


# =============================================================================
# check_adapter_input
# =============================================================================


class TestCheckAdapterInput:
    """Common validation of the inputs of fit and predict."""

    def test_fit_sets_feature_attributes(self, adapter_series):
        X, y = adapter_series
        estimator = BaseEstimator()
        X_out, y_out, layout = check_adapter_input(estimator, X, y, reset=True)
        assert X_out is X and y_out is y
        assert estimator.n_features_in_ == 2
        assert estimator.feature_names_in_.tolist() == ['x1', 'x2']
        assert layout.freq == 'MS' and not layout.is_panel

    def test_predict_checks_columns(self, adapter_panel):
        X, y = adapter_panel
        estimator = BaseEstimator()
        check_adapter_input(estimator, X, y, reset=True)
        _, y_out, layout = check_adapter_input(estimator, X.iloc[:2], reset=False, freq='MS')
        assert y_out is None and layout.freq == 'MS'
        with pytest.raises(ValueError, match='feature names'):
            check_adapter_input(estimator, X[['x2', 'x1']], reset=False)

    def test_predict_without_detection(self, adapter_series):
        X, y = adapter_series
        estimator = BaseEstimator()
        check_adapter_input(estimator, X, y, reset=True)
        _, _, layout = check_adapter_input(estimator, X.iloc[:1], reset=False, detect_freq=False)
        assert layout.freq is None

    def test_array_target_is_placed_on_x(self, adapter_series):
        X, y = adapter_series
        _, y_out, _ = check_adapter_input(BaseEstimator(), X, y.to_numpy(), reset=True)
        assert isinstance(y_out, pd.Series) and y_out.index.equals(X.index)

    def test_two_dimensional_array_target(self, adapter_series):
        X, y = adapter_series
        _, y_out, _ = check_adapter_input(BaseEstimator(), X, np.column_stack([y, y]), reset=True)
        assert isinstance(y_out, pd.DataFrame) and y_out.shape == (len(X), 2)

    def test_misaligned_target(self, adapter_series):
        X, y = adapter_series
        with pytest.raises(ValueError, match='same index'):
            check_adapter_input(BaseEstimator(), X, y.iloc[::-1], reset=True)

    def test_target_of_wrong_length(self, adapter_series):
        X, y = adapter_series
        with pytest.raises(ValueError, match='one row per row'):
            check_adapter_input(BaseEstimator(), X, y.to_numpy()[:-1], reset=True)

    def test_x_must_be_a_dataframe(self, adapter_series):
        X, y = adapter_series
        with pytest.raises(TypeError, match='DataFrame'):
            check_adapter_input(BaseEstimator(), X.to_numpy(), y, reset=True)

    def test_x_is_not_modified(self, adapter_unbalanced_panel):
        X, y = adapter_unbalanced_panel
        before = X.copy()
        check_adapter_input(BaseEstimator(), X, y, reset=True)
        pd.testing.assert_frame_equal(X, before)
