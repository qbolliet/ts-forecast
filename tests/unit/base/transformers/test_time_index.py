"""Tests of how ``PanelTimeSeriesTransformer`` determines and checks the time index.

Covers ``TimeSeriesTransformerMixin._validate_time_index`` and the index-based
validation of ``PanelTimeSeriesTransformer.fit``: time labels read from the index,
from a ``time_col`` column, from the last level of a two- or three-level
``MultiIndex``; non-standard index names; date strings and ``Period`` values;
numeric and non-date labels; duplicated and unsorted time labels. The private
method is also tested directly (marked ``internal``): it is the only member of
the mixin.
"""
import pandas as pd
import pytest

from tsforecast.base.transformers import TimeSeriesTransformerMixin
from tests.support.perturbations import (
    shuffle_rows,
    to_period_index,
    to_three_level_index,
    with_duplicated_rows,
    with_index_names,
)

from .conftest import ConcreteTransformer, month_ends


# =============================================================================
# Index temporel déterminé par la voie publique ``fit``
# =============================================================================
class TestTimeIndexThroughFit:
    """``fit`` reads the time labels where the configuration says they are."""

    def test_datetime_index_is_used_as_is(self, series_frame):
        """A ``DatetimeIndex`` reaches ``_fit`` unchanged."""
        transformer = ConcreteTransformer().fit(series_frame)
        pd.testing.assert_frame_equal(transformer.fit_input_, series_frame)

    def test_time_col_becomes_the_index(self, series_frame):
        """A ``time_col`` column is moved into a ``DatetimeIndex`` named after it."""
        # Construction d'une série en colonnes : la date est une colonne ordinaire
        frame = series_frame.reset_index()

        transformer = ConcreteTransformer(time_col='date').fit(frame)

        pd.testing.assert_frame_equal(transformer.fit_input_, series_frame, check_freq=False)

    def test_string_time_col_is_converted(self):
        """ISO date strings in ``time_col`` become datetimes."""
        frame = pd.DataFrame({'date': ['2023-01-31', '2023-02-28'], 'value': [1.0, 2.0]})

        transformer = ConcreteTransformer(time_col='date').fit(frame)

        pd.testing.assert_index_equal(transformer.fit_input_.index, month_ends(2))

    def test_two_level_multiindex_uses_its_last_level(self, index_panel):
        """A ``(country, date)`` panel reaches ``_fit`` with the same index."""
        transformer = ConcreteTransformer().fit(index_panel)
        pd.testing.assert_index_equal(transformer.fit_input_.index, index_panel.index)

    def test_three_level_multiindex_uses_its_last_level(self, index_panel):
        """A ``(region, country, date)`` panel is accepted, its dates read on the last level."""
        # Ajout d'un niveau extérieur constant : le temps reste au dernier niveau
        panel = to_three_level_index(index_panel)

        transformer = ConcreteTransformer().fit(panel)

        pd.testing.assert_index_equal(transformer.fit_input_.index, panel.index)

    @pytest.mark.parametrize(
        'names', ['période', 'Date de publication', None], ids=['accent', 'space', 'unnamed']
    )
    def test_index_name_does_not_matter_for_a_series(self, series_frame, names):
        """The time index is found by position, whatever its name."""
        frame = with_index_names(series_frame, names)

        output = ConcreteTransformer().fit_transform(frame)

        pd.testing.assert_frame_equal(output, frame)

    @pytest.mark.parametrize(
        'names', [['pays', 'horodatage'], ['entity id', 'publication date'], [None, None]],
        ids=['french', 'spaces', 'unnamed'],
    )
    def test_level_names_do_not_matter_for_a_panel(self, index_panel, names):
        """Entity and time levels are found by position, whatever their names."""
        panel = with_index_names(index_panel, names)

        output = ConcreteTransformer().fit_transform(panel)

        pd.testing.assert_frame_equal(output, panel)

    def test_string_dates_on_the_index_are_converted(self):
        """ISO date strings on a flat index reach ``_fit`` as datetimes."""
        frame = pd.DataFrame({'value': [1.0, 2.0]}, index=['2023-01-31', '2023-02-28'])

        transformer = ConcreteTransformer().fit(frame)

        assert list(transformer.fit_input_.index) == list(month_ends(2))

    def test_string_dates_on_the_last_level_are_converted(self):
        """ISO date strings on the last level of a ``MultiIndex`` become datetimes."""
        index = pd.MultiIndex.from_product([['FR'], ['2023-01-31', '2023-02-28']])
        frame = pd.DataFrame({'value': [1.0, 2.0]}, index=index)

        transformer = ConcreteTransformer().fit(frame)

        assert list(transformer.fit_input_.index.get_level_values(-1)) == list(month_ends(2))

    def test_period_last_level_is_read_at_first_instants(self, index_panel):
        """``Period`` dates on the last level are read at the first instant of each period."""
        # Valeur d'or : la période 2023-01 commence le 2023-01-01
        panel = to_period_index(index_panel)

        transformer = ConcreteTransformer().fit(panel)

        expected = pd.date_range('2023-01-01', periods=4, freq='MS')
        assert list(transformer.fit_input_.index.get_level_values(-1)) == list(expected) * 2

    def test_period_index_is_read_at_first_instants(self, series_frame):
        """A flat ``PeriodIndex`` is accepted like the last level of a panel."""
        frame = to_period_index(series_frame)

        transformer = ConcreteTransformer().fit(frame)

        expected = pd.date_range('2023-01-01', periods=4, freq='MS')
        assert list(transformer.fit_input_.index) == list(expected)


# =============================================================================
# Étiquettes temporelles refusées
# =============================================================================
class TestRejectedTimeLabels:
    """Labels that are not dates are refused with an explicit message."""

    @pytest.mark.parametrize('strict', [True, False], ids=['strict', 'permissive'])
    @pytest.mark.parametrize(
        'labels', [range(2), [2020, 2021], [2020.0, 2021.0]], ids=['range', 'integer-years', 'floats']
    )
    def test_numeric_index_is_rejected(self, labels, strict):
        """Numeric labels are not dates: ``fit`` refuses them, as for non-date strings."""
        frame = pd.DataFrame({'value': [1.0, 2.0]}, index=pd.Index(labels))
        with pytest.raises(ValueError, match=r'the index must hold dates \(datetimes, date strings or periods\)'):
            ConcreteTransformer(strict_validation=strict).fit(frame)

    def test_numeric_last_level_is_rejected(self):
        """Numeric labels on the last level of a panel are refused in strict mode."""
        index = pd.MultiIndex.from_arrays([['FR', 'FR'], [2020, 2021]])
        frame = pd.DataFrame({'value': [1.0, 2.0]}, index=index)

        with pytest.raises(
            ValueError,
            match='MultiIndex panel validation failed: Last level of MultiIndex cannot be converted',
        ):
            ConcreteTransformer().fit(frame)

    def test_numeric_last_level_warns_when_not_strict(self):
        """In permissive mode, numeric last-level labels are left as they are, with a warning."""
        index = pd.MultiIndex.from_arrays([['FR', 'FR'], [2020, 2021]])
        frame = pd.DataFrame({'value': [1.0, 2.0]}, index=index)

        with pytest.warns(UserWarning, match='MultiIndex last level conversion failed'):
            transformer = ConcreteTransformer(strict_validation=False).fit(frame)

        assert list(transformer.fit_input_.index.get_level_values(-1)) == [2020, 2021]

    def test_non_date_strings_on_the_index_are_rejected(self):
        """Strings that are not dates leave no way to read a time index."""
        frame = pd.DataFrame({'value': [1.0, 2.0]}, index=['a', 'b'])
        with pytest.raises(ValueError, match='the index must hold dates'):
            ConcreteTransformer().fit(frame)

    @pytest.mark.parametrize('strict', [True, False], ids=['strict', 'permissive'])
    @pytest.mark.parametrize('labels', [['a', 'b'], [2020, 2021]], ids=['non-date-strings', 'integers'])
    def test_non_date_time_col_is_rejected_in_both_modes(self, labels, strict):
        """A ``time_col`` that is not made of dates raises whatever the mode, naming the column."""
        frame = pd.DataFrame({'date': labels, 'value': [1.0, 2.0]})
        with pytest.raises(
            ValueError,
            match="Time validation failed: Cannot determine time index. Column 'date' cannot be converted",
        ):
            ConcreteTransformer(time_col='date', strict_validation=strict).fit(frame)

    def test_absent_time_col_without_panel_cols_falls_back_on_the_index(self, series_frame):
        """A ``time_col`` that is not a column leaves the index in charge of the dates."""
        transformer = ConcreteTransformer(time_col='period').fit(series_frame)
        pd.testing.assert_frame_equal(transformer.fit_input_, series_frame)

    @pytest.mark.parametrize('strict', [True, False], ids=['strict', 'permissive'])
    @pytest.mark.parametrize('dated_index', [False, True], ids=['range-index', 'dated-index'])
    def test_absent_time_col_of_a_panel_is_named(self, column_panel, dated_index, strict):
        """A column panel whose ``time_col`` is missing is refused, the column being named.

        The dates may even sit in the index: a column panel needs its time column to
        build the ``(entity, date)`` index.
        """
        panel = column_panel.set_index('date') if dated_index else column_panel
        transformer = ConcreteTransformer(time_col='period', panel_cols=['country'], strict_validation=strict)

        with pytest.raises(ValueError, match="Time column 'period' not found in data"):
            transformer.fit(panel)


# =============================================================================
# Doublons et désordre de l'index temporel
# =============================================================================
class TestDuplicatedTimeLabels:
    """A time label (or an ``(entity, date)`` pair) appears at most once."""

    def test_duplicated_dates_are_rejected_in_strict_mode(self, series_frame):
        """A repeated date of a series raises."""
        frame = with_duplicated_rows(series_frame, n=1)
        with pytest.raises(ValueError, match='Time series validation failed: Index contains duplicate values'):
            ConcreteTransformer().fit(frame)

    def test_duplicated_dates_keep_the_first_occurrence_when_not_strict(self, series_frame):
        """In permissive mode the first occurrence of a repeated date is kept, with a warning."""
        # Doublon de la première ligne ajouté en queue de frame
        frame = with_duplicated_rows(series_frame, n=1)

        with pytest.warns(UserWarning, match='Index contains duplicates. Keeping first occurrence.'):
            transformer = ConcreteTransformer(strict_validation=False).fit(frame)

        pd.testing.assert_frame_equal(transformer.fit_input_, series_frame, check_freq=False)

    def test_duplicated_entity_date_pairs_are_rejected(self, index_panel):
        """A repeated ``(entity, date)`` pair of a panel raises."""
        panel = with_duplicated_rows(index_panel, n=1)
        with pytest.raises(
            ValueError, match='MultiIndex panel validation failed: MultiIndex contains duplicate combinations'
        ):
            ConcreteTransformer().fit(panel)

    def test_duplicated_entity_date_pairs_keep_the_first_when_not_strict(self, index_panel):
        """In permissive mode the panel keeps the first occurrence of a repeated pair."""
        panel = with_duplicated_rows(index_panel, n=1)

        with pytest.warns(UserWarning, match='MultiIndex contains duplicates. Keeping first occurrence.'):
            transformer = ConcreteTransformer(strict_validation=False).fit(panel)

        pd.testing.assert_frame_equal(transformer.fit_input_, index_panel)

    def test_duplicated_rows_of_a_column_panel_are_rejected(self, column_panel):
        """A repeated ``(country, date)`` row of a column panel raises."""
        panel = with_duplicated_rows(column_panel, n=1)
        with pytest.raises(ValueError, match='Temporal data validation failed: Duplicate rows found'):
            ConcreteTransformer(time_col='date', panel_cols=['country']).fit(panel)


class TestUnsortedTimeLabels:
    """Unsorted dates are refused, sorted on request, or let through with a warning."""

    @pytest.fixture
    def shuffled(self, series_frame) -> pd.DataFrame:
        """Monthly frame whose rows are shuffled (fixed seed)."""
        frame = shuffle_rows(series_frame, seed=0)
        # Précondition : la permutation tirée n'est pas l'identité
        assert not frame.index.is_monotonic_increasing
        return frame

    def test_unsorted_index_is_rejected_in_strict_mode(self, shuffled):
        """Unsorted dates of a series raise by default."""
        with pytest.raises(ValueError, match='Time series index is not sorted'):
            ConcreteTransformer().fit(shuffled)

    def test_unsorted_index_is_sorted_by_auto_sort(self, shuffled, series_frame):
        """``auto_sort=True`` sorts the dates, each value staying on its date."""
        output = ConcreteTransformer(auto_sort=True).fit_transform(shuffled)
        pd.testing.assert_frame_equal(output, series_frame, check_freq=False)

    def test_unsorted_index_is_kept_with_a_warning_when_not_strict(self, shuffled):
        """In permissive mode, the rows keep their order and a warning is emitted."""
        with pytest.warns(UserWarning, match='Time series index is not sorted'):
            transformer = ConcreteTransformer(strict_validation=False).fit(shuffled)

        pd.testing.assert_frame_equal(transformer.fit_input_, shuffled)

    def test_unsorted_dates_within_an_entity_are_rejected(self, index_panel):
        """Dates out of order inside one entity of a panel raise."""
        # Permutation des deux premières dates de FR, les blocs d'entités restent contigus
        panel = index_panel.iloc[[1, 0, 2, 3, 4, 5, 6, 7]]
        with pytest.raises(ValueError, match='Time series not sorted within panel groups'):
            ConcreteTransformer().fit(panel)

    def test_interleaved_entities_warn_when_not_strict(self, index_panel):
        """In permissive mode, entities of a ``MultiIndex`` alternating row by row only warn."""
        # Entrelacement FR, DE, FR, DE… : chaque entité reste triée par date
        panel = index_panel.iloc[[0, 4, 1, 5, 2, 6, 3, 7]]

        with pytest.warns(UserWarning, match='Panel entities are not contiguous in MultiIndex'):
            transformer = ConcreteTransformer(strict_validation=False).fit(panel)

        pd.testing.assert_frame_equal(transformer.fit_input_, panel)

    def test_unsorted_dates_within_an_entity_warn_when_not_strict(self, index_panel):
        """In permissive mode, dates out of order within an entity only warn."""
        panel = index_panel.iloc[[1, 0, 2, 3, 4, 5, 6, 7]]

        with pytest.warns(UserWarning, match='Time series not sorted within panel groups'):
            transformer = ConcreteTransformer(strict_validation=False).fit(panel)

        pd.testing.assert_frame_equal(transformer.fit_input_, panel)


# =============================================================================
# Méthode privée du mixin, testée directement
# =============================================================================
@pytest.mark.internal
class TestValidateTimeIndex:
    """``TimeSeriesTransformerMixin._validate_time_index`` follows the rules of ``validate_temporal_data``.

    The method is the only member of the mixin, so it is tested directly. It must
    accept and reject the same time labels as ``validate_temporal_data``: datetimes,
    date strings and periods are accepted; numeric labels are rejected.
    """

    @staticmethod
    def _months(periods: int = 3) -> pd.DatetimeIndex:
        """Build a month-start ``DatetimeIndex`` (``MS``)."""
        return pd.date_range('2023-01-01', periods=periods, freq='MS')

    def _validate(self, df: pd.DataFrame, time_col=None) -> pd.DatetimeIndex:
        """Call the method on a bare mixin."""
        return TimeSeriesTransformerMixin()._validate_time_index(df, time_col)

    def test_datetime_index_is_returned(self):
        """A ``DatetimeIndex`` comes back equal."""
        df = pd.DataFrame({'v': range(3)}, index=self._months())
        pd.testing.assert_index_equal(self._validate(df), self._months(), check_names=False)

    def test_string_index_is_converted(self):
        """ISO strings on the index become datetimes."""
        df = pd.DataFrame({'v': [1, 2]}, index=['2023-01-01', '2023-02-01'])
        assert list(self._validate(df)) == list(self._months(2))

    def test_multiindex_uses_its_last_level(self):
        """A panel ``MultiIndex`` gives the dates of its last level, one per row."""
        index = pd.MultiIndex.from_product([['A', 'B'], self._months(2)])
        result = self._validate(pd.DataFrame({'v': range(4)}, index=index))
        assert list(result) == list(self._months(2)) * 2

    def test_multiindex_with_string_dates(self):
        """String dates on the last level of a ``MultiIndex`` are converted."""
        index = pd.MultiIndex.from_product([['A'], ['2023-01-01', '2023-02-01']])
        result = self._validate(pd.DataFrame({'v': [1, 2]}, index=index))
        assert list(result) == list(self._months(2))

    def test_period_index_is_read_at_first_instants(self):
        """A ``PeriodIndex`` gives the first instant of each period."""
        df = pd.DataFrame({'v': range(3)}, index=pd.period_range('2023-01', periods=3, freq='M'))
        assert list(self._validate(df)) == list(self._months())

    def test_period_last_level_of_a_multiindex(self):
        """Periods on the last level of a ``MultiIndex`` are read at their first instant."""
        index = pd.MultiIndex.from_product([['A'], pd.period_range('2023-01', periods=2, freq='M')])
        result = self._validate(pd.DataFrame({'v': [1, 2]}, index=index))
        assert list(result) == list(self._months(2))

    def test_time_col_gives_a_datetime_index(self):
        """A time column is returned as a ``DatetimeIndex``, not as a ``Series``."""
        df = pd.DataFrame({'date': ['2023-01-01', '2023-02-01'], 'v': [1, 2]})
        result = self._validate(df, time_col='date')
        assert isinstance(result, pd.DatetimeIndex)
        assert list(result) == list(self._months(2))

    def test_period_time_col_is_read_at_first_instants(self):
        """A ``Period`` time column is accepted."""
        df = pd.DataFrame({'date': pd.period_range('2023-01', periods=2, freq='M'), 'v': [1, 2]})
        assert list(self._validate(df, time_col='date')) == list(self._months(2))

    def test_missing_time_col_falls_back_on_the_index(self):
        """A ``time_col`` that is not a column leaves the index in charge."""
        df = pd.DataFrame({'v': range(3)}, index=self._months())
        assert list(self._validate(df, time_col='date')) == list(self._months())

    @pytest.mark.parametrize(
        'labels', [[2020, 2021], [0, 1], [2020.0, 2021.0]], ids=['integer-years', 'range-like', 'floats']
    )
    def test_numeric_index_is_rejected(self, labels):
        """Numeric labels are not dates: they would be read as nanoseconds since 1970."""
        df = pd.DataFrame({'v': [1, 2]}, index=labels)
        with pytest.raises(ValueError, match='Cannot determine time index'):
            self._validate(df)

    def test_numeric_last_level_is_rejected(self):
        """Numeric labels on the last level of a ``MultiIndex`` are rejected too."""
        index = pd.MultiIndex.from_arrays([['A', 'A'], [2020, 2021]])
        with pytest.raises(ValueError, match='Cannot determine time index'):
            self._validate(pd.DataFrame({'v': [1, 2]}, index=index))

    def test_non_date_strings_are_rejected(self):
        """Strings that are not dates raise a ``ValueError``."""
        df = pd.DataFrame({'v': [1, 2]}, index=['a', 'b'])
        with pytest.raises(ValueError, match='Cannot determine time index'):
            self._validate(df)

    @pytest.mark.parametrize(
        'labels', [[2020, 2021], ['a', 'b']], ids=['integers', 'non-date-strings']
    )
    def test_invalid_time_col_is_rejected_with_its_name(self, labels):
        """A time column that is not made of dates raises a ``ValueError`` naming it."""
        df = pd.DataFrame({'date': labels, 'v': [1, 2]})
        with pytest.raises(ValueError, match="Column 'date' cannot be converted"):
            self._validate(df, time_col='date')

    def test_empty_frame_is_accepted(self):
        """An empty frame with the default index has no label to refuse."""
        assert len(self._validate(pd.DataFrame({'v': []}))) == 0
