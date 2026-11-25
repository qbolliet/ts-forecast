"""``restore_original_structure`` after ``validate_temporal_data``.

Scenarios: restoration of the recorded structure and the validate-then-restore round
trip on perturbed realistic datasets.
"""

import pandas as pd
import pytest

# Jeux de données et perturbations partagés
from tests.support import perturbations as perturb

# Fonctions à tester
from tsforecast.utils.validation import (
    restore_original_structure,
    validate_temporal_data,
)

# Outils de test partagés du paquet
from tests.unit.utils.validation.utils.helpers import _months, _validate_kwargs


# Le chemin « colonnes » émet systématiquement un avertissement de remplacement
# d'index (comportement testé explicitement dans TestColumnBasedValidation) ; le
# bruit est masqué ailleurs pour garder une sortie lisible. L'inférence de format
# de pandas sur des chaînes de dates non ISO émet aussi un avertissement.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Index replaced with"),
    pytest.mark.filterwarnings("ignore:Could not infer format"),
]


# =============================================================================
# restore_original_structure
# =============================================================================
class TestRestoreOriginalStructure:
    """Restoration of the structure recorded by ``validate_temporal_data``."""

    def test_basic_restoration(self):
        """A validate → restore round trip on a sorted ``time_col`` frame is the identity."""
        original = pd.DataFrame({
            'timestamp': _months(4),
            'revenue': [1000, 1100, 1050, 1200],
            'users': [50, 55, 52, 60],
        })
        validated, meta = validate_temporal_data(original, time_col='timestamp', return_metadata=True)

        restored = restore_original_structure(validated, meta)

        pd.testing.assert_frame_equal(restored, original)

    def test_restoration_preserves_dates(self):
        """The time column comes back with its dates."""
        original = pd.DataFrame({'date': pd.date_range('2023-01-01', periods=5), 'value': range(5)})
        validated, meta = validate_temporal_data(original, time_col='date', return_metadata=True)

        restored = restore_original_structure(validated, meta)

        pd.testing.assert_series_equal(restored['date'], original['date'])

    def test_string_dates_come_back_as_datetimes(self):
        """The converted time column is restored as ``datetime64``, not as the input strings (pinned).

        The metadata records no dtype: only the converted values are available.
        """
        original = pd.DataFrame({'date': ['2023-01-01', '2023-02-01'], 'v': [1, 2]})
        validated, meta = validate_temporal_data(original, time_col='date', return_metadata=True)
        restored = restore_original_structure(validated, meta)
        assert list(restored['date']) == [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01')]

    def test_restoration_with_panel_data(self):
        """Panel and time columns come back as columns, with all values."""
        original = pd.DataFrame({
            'entity': ['US'] * 3 + ['FR'] * 3,
            'date': _months(3).tolist() * 2,
            'value': range(6),
        })
        validated, meta = validate_temporal_data(
            original, time_col='date', panel_cols=['entity'], return_metadata=True
        )

        restored = restore_original_structure(validated, meta)

        # Les lignes sont triées par entité (FR avant US) : comparaison indépendante de l'ordre
        key = ['entity', 'date']
        pd.testing.assert_frame_equal(
            restored.sort_values(key).reset_index(drop=True),
            original.sort_values(key).reset_index(drop=True),
        )

    def test_restoration_after_transformations(self):
        """Columns added to the validated frame are carried through the restoration."""
        original = pd.DataFrame({
            'timestamp': _months(4),
            'revenue': [1000, 1100, 1050, 1200],
        })
        validated, meta = validate_temporal_data(original, time_col='timestamp', return_metadata=True)
        validated['revenue_change'] = validated['revenue'].pct_change()

        restored = restore_original_structure(validated, meta)

        assert list(restored.columns) == ['timestamp', 'revenue', 'revenue_change']
        assert not restored['revenue'].isna().any()

    def test_index_path_restores_a_string_index(self):
        """On the index path the original (string) index is restored as such."""
        original = pd.DataFrame({'v': [1, 2]}, index=['2023-01-01', '2023-02-01'])
        validated, meta = validate_temporal_data(original, return_metadata=True)
        restored = restore_original_structure(validated, meta)
        pd.testing.assert_frame_equal(restored, original)

    def test_series_with_datetime_index_round_trips(self):
        """A named ``Series`` on the index path is restored equal to itself."""
        original = pd.Series([1.0, 2.0, 3.0], index=_months(3, ), name='v')
        validated, meta = validate_temporal_data(original, return_metadata=True)
        restored = restore_original_structure(validated, meta)
        pd.testing.assert_series_equal(restored, original, check_freq=False)

    def test_unnamed_series_round_trips_unnamed(self):
        """An unnamed ``Series`` is restored unnamed."""
        original = pd.Series([1, 2, 3], index=_months(3))
        validated, meta = validate_temporal_data(original, return_metadata=True)
        assert restore_original_structure(validated, meta).name is None

    def test_restoration_does_not_mutate_its_input(self):
        """The validated frame given to the function is left untouched."""
        original = pd.DataFrame({'date': _months(3), 'v': [1, 2, 3]})
        validated, meta = validate_temporal_data(original, time_col='date', return_metadata=True)
        before = validated.copy()

        restore_original_structure(validated, meta)

        pd.testing.assert_frame_equal(validated, before)

    def test_empty_metadata_leaves_the_data_unchanged(self):
        """Without recorded structure, the data are returned as a copy, unchanged."""
        df = pd.DataFrame({'v': [1, 2]}, index=_months(2))
        restored = restore_original_structure(df, {})
        pd.testing.assert_frame_equal(restored, df)
        assert restored is not df

    def test_index_is_not_restored_when_the_row_count_changed(self):
        """A length mismatch (rows dropped after validation) leaves the index as is (pinned).

        Typical case: ``strict=False`` dropped a duplicated row, so the
        recorded ``original_index`` is longer than the validated data. The
        function silently keeps the current index instead of raising.
        """
        index = pd.DatetimeIndex(['2023-01-01', '2023-01-01', '2023-02-01'])
        original = pd.DataFrame({'v': [1, 2, 3]}, index=index)
        with pytest.warns(UserWarning):
            validated, meta = validate_temporal_data(original, strict=False, return_metadata=True)

        restored = restore_original_structure(validated, meta)

        pd.testing.assert_frame_equal(restored, validated)

    def test_index_path_restoration_after_sort_keeps_each_value_on_its_date(self):
        """After a sort, each value stays on its own date (index path).

        Golden value: ``30`` sits on ``2023-03-01`` in the input, so it still sits on
        ``2023-03-01`` after ``validate → restore``. The original row order is not
        restored (rows stay sorted): the unsorted original index is not glued onto the
        sorted rows, so ``2023-03-01`` keeps the value ``30``.
        """
        index = pd.to_datetime(['2023-03-01', '2023-01-01', '2023-02-01'])
        original = pd.Series([30, 10, 20], index=index, name='v')
        validated, meta = validate_temporal_data(original, sort_data=True, return_metadata=True)

        restored = restore_original_structure(validated, meta)

        pd.testing.assert_series_equal(restored, original.sort_index(), check_freq=False)

    def test_column_path_restoration_after_sort_gives_the_sorted_rows(self):
        """After a sort, the rows come back sorted with fresh labels (column path).

        The original ``RangeIndex`` labels do not line up with the sorted rows, so
        they are not reassigned; each ``(date, value)`` pair stays intact.
        """
        original = pd.DataFrame({
            'date': pd.to_datetime(['2023-03-01', '2023-01-01', '2023-02-01']),
            'v': [30, 10, 20],
        })
        validated, meta = validate_temporal_data(
            original, time_col='date', sort_data=True, return_metadata=True
        )

        restored = restore_original_structure(validated, meta)

        expected = original.sort_values('date').reset_index(drop=True)
        pd.testing.assert_frame_equal(restored, expected)

    def test_period_index_is_restored_after_conversion(self):
        """A ``PeriodIndex`` converted to timestamps for validation comes back as periods."""
        original = pd.Series(
            [1, 2, 3], index=pd.period_range('2023-01', periods=3, freq='M'), name='v'
        )
        validated, meta = validate_temporal_data(original, return_metadata=True)

        restored = restore_original_structure(validated, meta)

        pd.testing.assert_series_equal(restored, original)

    def test_columns_are_restored_at_their_original_positions(self):
        """Columns come back in their original order: ``['date', 'v', 'entity']``.

        ``reset_index()`` puts the former index columns first, in index order; the
        recorded ``original_columns`` puts them back where they were.
        """
        original = pd.DataFrame({
            'date': _months(3).tolist() * 2,
            'v': range(6),
            'entity': ['A'] * 3 + ['B'] * 3,
        })
        validated, meta = validate_temporal_data(
            original, time_col='date', panel_cols=['entity'], return_metadata=True
        )

        restored = restore_original_structure(validated, meta)

        assert list(restored.columns) == ['date', 'v', 'entity']

    def test_columns_added_after_validation_stay_last(self):
        """A column added to the validated frame is placed after the original columns."""
        original = pd.DataFrame({'date': _months(3), 'v': [1, 2, 3], 'w': [4, 5, 6]})
        validated, meta = validate_temporal_data(original, time_col='date', return_metadata=True)
        validated['extra'] = 0

        restored = restore_original_structure(validated, meta)

        assert list(restored.columns) == ['date', 'v', 'w', 'extra']

    def test_missing_column_record_leaves_the_reset_order(self):
        """Without recorded ``original_columns``, index columns simply come first."""
        validated = pd.DataFrame({'v': [1, 2]}, index=pd.Index(_months(2), name='date'))
        restored = restore_original_structure(validated, {'index_was_replaced': True})
        assert list(restored.columns) == ['date', 'v']

    def test_a_series_of_the_column_path_gets_its_original_index_back(self):
        """A value ``Series`` restored after the column path keeps its values.

        The values come back on the original ``RangeIndex``; the dates, carried by the
        validated index only, are dropped (a ``Series`` has no column to put them in).
        """
        original = pd.DataFrame({'date': _months(3), 'v': [10, 20, 30]})
        validated, meta = validate_temporal_data(original, time_col='date', return_metadata=True)

        restored = restore_original_structure(validated['v'], meta)

        pd.testing.assert_series_equal(restored, original['v'])

    def test_a_reordered_series_keeps_its_time_index(self):
        """After a sort, a ``Series`` keeps the time index it has (no label to restore)."""
        original = pd.DataFrame({
            'date': pd.to_datetime(['2023-02-01', '2023-01-01']), 'v': [20, 10]
        })
        validated, meta = validate_temporal_data(original, time_col='date', return_metadata=True)

        restored = restore_original_structure(validated['v'], meta)

        assert list(restored) == [10, 20]
        assert isinstance(restored.index, pd.DatetimeIndex)


# =============================================================================
# Aller-retour validate_temporal_data → restore_original_structure
# =============================================================================
def _order_preserving_perturbations(kind: str) -> dict:
    """Return the perturbations that keep rows sorted, keyed by a readable id.

    Args:
        kind: ``'timeseries'`` or ``'panel'`` (some perturbations need a panel).

    Returns:
        Dictionary ``{id: function(df) -> df}``.
    """
    index_names = 'periode' if kind == 'timeseries' else ['pays', 'periode']
    perturbations = {
        'identity': lambda df: df,
        'special_column_names': lambda df: perturb.with_special_column_names(df)[0],
        'custom_index_names': lambda df: perturb.with_index_names(df, index_names),
        'period_start': perturb.to_period_start,
        'period_end': perturb.to_period_end,
        'single_observation': perturb.single_observation,
        'empty': perturb.empty_like,
    }
    if kind == 'panel':
        perturbations['three_level_index'] = perturb.to_three_level_index
        perturbations['drop_entity'] = lambda df: perturb.drop_entity(df, 'Italie')
    return perturbations


def _order_changing_perturbations(kind: str) -> dict:
    """Return the perturbations that leave rows unsorted, keyed by a readable id."""
    perturbations = {'shuffle_rows': lambda df: perturb.shuffle_rows(df, seed=0)}
    if kind == 'panel':
        perturbations['reverse_entities'] = perturb.reverse_entities
    return perturbations


def _round_trip_cases(perturbations_of) -> list:
    """Cross ``kind`` x ``path`` x perturbation into ``pytest.param`` cases."""
    cases = []
    for kind in ('timeseries', 'panel'):
        for path in ('index', 'columns'):
            for name in perturbations_of(kind):
                cases.append(pytest.param(kind, path, name, id=f'{kind}-{path}-{name}'))
    return cases


def _dataset(request: pytest.FixtureRequest, kind: str) -> pd.DataFrame:
    """Return the realistic dataset (notebook 3) matching ``kind``."""
    fixture = 'irregular_index_timeseries' if kind == 'timeseries' else 'heterogeneous_coverage_panel'
    return request.getfixturevalue(fixture)


class TestRoundTripOnPerturbedDatasets:
    """``validate → restore`` on the realistic datasets, perturbed one way at a time."""

    @pytest.mark.parametrize(
        'kind, path, perturbation', _round_trip_cases(_order_preserving_perturbations)
    )
    def test_round_trip_is_the_identity(self, request, kind, path, perturbation):
        """A sorted, valid frame is restored exactly as it was."""
        df = _order_preserving_perturbations(kind)[perturbation](_dataset(request, kind))
        data, kwargs = _validate_kwargs(df, path)
        validated, meta = validate_temporal_data(data, return_metadata=True, **kwargs)

        restored = restore_original_structure(validated, meta)

        pd.testing.assert_frame_equal(restored, data, check_freq=False)

    @pytest.mark.parametrize(
        'kind, path, perturbation', _round_trip_cases(_order_changing_perturbations)
    )
    def test_round_trip_of_unsorted_data_gives_the_sorted_frame(
        self, request, kind, path, perturbation
    ):
        """An unsorted frame comes back sorted, every row intact.

        The original row order is not restored. Index path: the sorted original.
        Column path: the rows sorted by ``(panel_cols, time_col)`` with fresh labels.
        """
        df = _order_changing_perturbations(kind)[perturbation](_dataset(request, kind))
        data, kwargs = _validate_kwargs(df, path)
        validated, meta = validate_temporal_data(data, return_metadata=True, **kwargs)

        restored = restore_original_structure(validated, meta)

        if path == 'index':
            expected = data.sort_index()
        else:
            keys = kwargs.get('panel_cols', []) + [kwargs['time_col']]
            expected = data.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(restored, expected, check_freq=False)

    @pytest.mark.parametrize(
        'kind, path, perturbation', _round_trip_cases(_order_changing_perturbations)
    )
    def test_round_trip_of_unsorted_data_without_sort_is_the_identity(
        self, request, kind, path, perturbation
    ):
        """With ``sort_data=False`` the round trip is the identity even on unsorted data."""
        df = _order_changing_perturbations(kind)[perturbation](_dataset(request, kind))
        data, kwargs = _validate_kwargs(df, path)
        validated, meta = validate_temporal_data(
            data, sort_data=False, return_metadata=True, **kwargs
        )

        restored = restore_original_structure(validated, meta)

        pd.testing.assert_frame_equal(restored, data, check_freq=False)

    @pytest.mark.parametrize('kind', ['timeseries', 'panel'])
    def test_validation_orders_unsorted_data_on_realistic_datasets(self, request, kind):
        """Whatever the shuffle, the validated index is monotonic increasing (index path)."""
        df = perturb.shuffle_rows(_dataset(request, kind), seed=1)
        assert validate_temporal_data(df).index.is_monotonic_increasing

    @pytest.mark.parametrize('kind', ['timeseries', 'panel'])
    def test_validation_only_reorders_rows(self, request, kind):
        """Sorting a shuffled dataset gives back the original dataset (index path).

        Every row follows its own index label: the validated frame equals the
        sorted original.
        """
        original = _dataset(request, kind)
        shuffled = perturb.shuffle_rows(original, seed=2)

        out = validate_temporal_data(shuffled)

        pd.testing.assert_frame_equal(out, original.sort_index(), check_freq=False)

    @pytest.mark.parametrize('strict', [True, False], ids=['strict', 'not-strict'])
    def test_duplicated_rows_on_a_realistic_panel(self, request, strict):
        """Duplicated rows: an error when strict, dropped (keep first) when not."""
        panel = _dataset(request, 'panel')
        duplicated = perturb.with_duplicated_rows(panel, n=3)

        if strict:
            with pytest.raises(ValueError, match='MultiIndex contains duplicate combinations'):
                validate_temporal_data(duplicated, strict=True)
        else:
            with pytest.warns(UserWarning, match='Keeping first occurrence'):
                out = validate_temporal_data(duplicated, strict=False)
            pd.testing.assert_frame_equal(out, panel.sort_index(), check_freq=False)

    def test_duplicated_rows_on_the_column_path(self, request):
        """Duplicated ``(entity, date)`` rows on the column path follow the same contract."""
        panel = _dataset(request, 'panel')
        flat = perturb.with_duplicated_rows(panel, n=3).reset_index()
        with pytest.raises(ValueError, match='Duplicate rows found'):
            validate_temporal_data(flat, time_col='date', panel_cols=['country'], strict=True)

    def test_period_index_dataset_round_trips(self, mixed_freq_timeseries):
        """A regular series on a ``PeriodIndex`` is validated, then restored as periods."""
        period_ts = perturb.to_period_index(mixed_freq_timeseries)
        assert isinstance(period_ts.index, pd.PeriodIndex), "la perturbation doit produire un PeriodIndex"

        validated, meta = validate_temporal_data(period_ts, return_metadata=True)
        restored = restore_original_structure(validated, meta)

        assert list(validated.index) == list(period_ts.index.to_timestamp())
        pd.testing.assert_frame_equal(restored, period_ts)

    def test_period_column_dataset_is_validated(self, mixed_freq_timeseries):
        """A ``Period`` time column is read at the start of each period (column path)."""
        period_ts = perturb.to_period_index(mixed_freq_timeseries)
        flat = period_ts.reset_index()

        validated = validate_temporal_data(flat, time_col='date')

        assert list(validated.index) == list(period_ts.index.to_timestamp())

    def test_period_index_panel_round_trips(self, mixed_freq_panel):
        """A regular panel with a ``PeriodIndex`` date level is validated, then restored."""
        period_panel = perturb.to_period_index(mixed_freq_panel)
        assert isinstance(period_panel.index.get_level_values(-1), pd.PeriodIndex)

        validated, meta = validate_temporal_data(period_panel, return_metadata=True)
        restored = restore_original_structure(validated, meta)

        dates = validated.index.get_level_values(-1)
        assert list(dates) == list(period_panel.index.get_level_values(-1).to_timestamp())
        pd.testing.assert_frame_equal(restored, period_panel)
