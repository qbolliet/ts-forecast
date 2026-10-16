"""Tests of the input validation modes of ``PanelTimeSeriesTransformer``.

Covers ``strict_validation`` (errors or warnings), ``auto_sort`` (sorting of
unsorted series and column panels), ``validate_input=False`` (data passed through
untouched), the contiguity and intra-entity order checks of column panels, the
missing ``panel_cols`` error, and the sorting advice that suggests ``auto_sort``
only to a transformer exposing it.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from tests.support.perturbations import shuffle_rows

from .conftest import ConcreteTransformer, FixedSortTransformer


# =============================================================================
# Jeux locaux déterministes
# =============================================================================
@pytest.fixture
def daily_frame() -> pd.DataFrame:
    """Daily series of 2023 in columns (``date``, ``value``), sorted, ``value`` = day number."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    return pd.DataFrame({'date': dates, 'value': np.arange(len(dates), dtype=float)})


@pytest.fixture
def shuffled_daily_frame(daily_frame) -> pd.DataFrame:
    """The daily series with its rows shuffled (fixed seed)."""
    return shuffle_rows(daily_frame, seed=0)


@pytest.fixture
def country_panel() -> pd.DataFrame:
    """Sorted column panel of three countries over the 12 month starts of 2023.

    ``value`` = 100 * country rank + month rank, so each row is identifiable.
    """
    countries = ['France', 'Germany', 'Italy']
    dates = pd.date_range('2023-01-01', '2023-12-01', freq='MS')
    rows = [
        {'country': country, 'date': date, 'value': 100.0 * i + j}
        for i, country in enumerate(countries)
        for j, date in enumerate(dates)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def interleaved_panel() -> pd.DataFrame:
    """Column panel whose entities alternate row by row (not contiguous)."""
    return pd.DataFrame({
        'country': ['France', 'Germany', 'France', 'Germany'],
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-02-01', '2023-02-01']),
        'value': [1.0, 2.0, 3.0, 4.0],
    })


def _indexed(frame: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Expected validated frame: sorted by ``keys`` and indexed by them."""
    return frame.sort_values(keys).set_index(keys)


# =============================================================================
# Mode strict et mode permissif
# =============================================================================
class TestStrictValidationMode:
    """``strict_validation=True`` raises on unsorted dates; ``False`` warns."""

    def test_sorted_series_passes_unchanged(self, daily_frame):
        """Sorted data reaches ``_fit`` with its values on their dates, without warning."""
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message='Time series index is not sorted')
            transformer = ConcreteTransformer(time_col='date').fit(daily_frame)

        pd.testing.assert_frame_equal(transformer.fit_input_, _indexed(daily_frame, ['date']))

    def test_unsorted_series_raises(self, shuffled_daily_frame):
        """Unsorted dates raise in strict mode."""
        with pytest.raises(ValueError, match='Time series index is not sorted'):
            ConcreteTransformer(time_col='date').fit(shuffled_daily_frame)

    def test_unsorted_series_warns_when_not_strict(self, shuffled_daily_frame):
        """Unsorted dates only warn in permissive mode, the row order being kept."""
        with pytest.warns(UserWarning, match='Time series index is not sorted'):
            transformer = ConcreteTransformer(time_col='date', strict_validation=False).fit(shuffled_daily_frame)

        assert transformer.fit_input_.index.tolist() == shuffled_daily_frame['date'].tolist()


# =============================================================================
# Tri automatique
# =============================================================================
class TestAutoSortFeature:
    """``auto_sort=True`` sorts the rows, every value staying on its date."""

    def test_unsorted_series_is_sorted(self, shuffled_daily_frame, daily_frame):
        """Shuffled rows come out sorted, identical to the sorted series."""
        output = ConcreteTransformer(time_col='date', auto_sort=True).fit_transform(shuffled_daily_frame)
        pd.testing.assert_frame_equal(output, _indexed(daily_frame, ['date']))

    def test_sorted_series_is_left_in_order(self, daily_frame):
        """Already sorted data comes out in the same order."""
        output = ConcreteTransformer(time_col='date', auto_sort=True).fit_transform(daily_frame)
        assert output['value'].tolist() == daily_frame['value'].tolist()

    def test_unsorted_panel_is_sorted(self, country_panel):
        """A shuffled column panel comes out sorted by country then date."""
        shuffled = shuffle_rows(country_panel, seed=1)

        output = ConcreteTransformer(time_col='date', panel_cols=['country'], auto_sort=True).fit_transform(shuffled)

        pd.testing.assert_frame_equal(output, _indexed(country_panel, ['country', 'date']))


# =============================================================================
# Validation des panels en colonnes
# =============================================================================
class TestPanelDataValidation:
    """Column panels must be contiguous by entity and sorted by date within each entity."""

    def test_sorted_panel_passes(self, country_panel):
        """A sorted, grouped panel reaches ``_fit`` indexed by ``(country, date)``."""
        transformer = ConcreteTransformer(time_col='date', panel_cols=['country']).fit(country_panel)
        pd.testing.assert_frame_equal(transformer.fit_input_, _indexed(country_panel, ['country', 'date']))

    def test_interleaved_entities_raise(self, interleaved_panel):
        """Entities alternating row by row raise in strict mode."""
        with pytest.raises(ValueError, match='Panel entities are not contiguous'):
            ConcreteTransformer(time_col='date', panel_cols=['country']).fit(interleaved_panel)

    def test_interleaved_entities_warn_when_not_strict(self, interleaved_panel):
        """In permissive mode, interleaved entities warn and keep their order."""
        transformer = ConcreteTransformer(time_col='date', panel_cols=['country'], strict_validation=False)

        with pytest.warns(UserWarning, match='Panel entities are not contiguous'):
            transformer.fit(interleaved_panel)

        assert transformer.fit_input_['value'].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_unsorted_dates_within_an_entity_raise(self):
        """Dates out of order inside one entity raise in strict mode."""
        panel = pd.DataFrame({
            'country': ['France', 'France', 'Germany', 'Germany'],
            'date': pd.to_datetime(['2023-02-01', '2023-01-01', '2023-01-01', '2023-02-01']),
            'value': [1.0, 2.0, 3.0, 4.0],
        })
        with pytest.raises(ValueError, match='Time series not sorted within panel groups'):
            ConcreteTransformer(time_col='date', panel_cols=['country']).fit(panel)

    def test_unsorted_dates_within_an_entity_warn_when_not_strict(self):
        """In permissive mode, dates out of order within an entity only warn."""
        panel = pd.DataFrame({
            'country': ['France', 'France'],
            'date': pd.to_datetime(['2023-02-01', '2023-01-01']),
            'value': [1.0, 2.0],
        })
        with pytest.warns(UserWarning, match='Time series not sorted within panel groups'):
            ConcreteTransformer(time_col='date', panel_cols=['country'], strict_validation=False).fit(panel)

    def test_missing_panel_column_raises(self, country_panel):
        """A ``panel_cols`` column absent from the data raises, naming it."""
        with pytest.raises(ValueError, match="Panel columns not found: {'missing_column'}"):
            ConcreteTransformer(time_col='date', panel_cols=['missing_column']).fit(country_panel)

    def test_duplicated_rows_warn_when_not_strict(self, country_panel):
        """In permissive mode, a repeated ``(country, date)`` row keeps its first occurrence."""
        panel = pd.concat([country_panel, country_panel.iloc[[0]]], ignore_index=True)
        panel.loc[len(panel) - 1, 'value'] = -1.0

        with pytest.warns(UserWarning, match='Duplicates found for'):
            transformer = ConcreteTransformer(time_col='date', panel_cols=['country'], strict_validation=False)
            transformer.fit(panel)

        assert -1.0 not in transformer.fit_input_['value'].tolist()


# =============================================================================
# Conseil de tri des messages
# =============================================================================
class TestSortingAdvice:
    """The sorting advice suggests ``auto_sort`` only where the parameter exists."""

    @pytest.mark.parametrize(
        'transformer',
        [ConcreteTransformer(time_col='date', panel_cols=['country']),
         FixedSortTransformer(time_col='date', panel_cols=['country'])],
        ids=['auto_sort_exposed', 'auto_sort_fixed'],
    )
    def test_interleaved_entities(self, interleaved_panel, transformer):
        """``auto_sort`` is suggested if and only if the transformer exposes it."""
        exposed = 'auto_sort' in transformer.get_params()

        with pytest.raises(ValueError, match='not contiguous') as excinfo:
            transformer.fit(interleaved_panel)

        assert ('auto_sort=True' in str(excinfo.value)) is exposed
        assert ('never sorts implicitly' in str(excinfo.value)) is not exposed

    def test_unsorted_multiindex_under_a_fixed_sort(self):
        """A ``MultiIndex`` unsorted within an entity: the caller is told to sort."""
        index = pd.MultiIndex.from_tuples(
            [('A', pd.Timestamp('2023-02-01')), ('A', pd.Timestamp('2023-01-01'))],
            names=['entity', 'date'],
        )
        data = pd.DataFrame({'value': [1.0, 2.0]}, index=index)

        with pytest.raises(ValueError, match='not sorted within panel groups') as excinfo:
            FixedSortTransformer().fit(data)

        assert 'auto_sort' not in str(excinfo.value)
        assert 'Sort the data by date within each entity' in str(excinfo.value)

    def test_interleaved_multiindex_under_a_fixed_sort(self):
        """A ``MultiIndex`` with interleaved entities names ``FixedSortTransformer`` in the advice."""
        index = pd.MultiIndex.from_arrays(
            [['A', 'B', 'A', 'B'], list(pd.date_range('2023-01-01', periods=2).repeat(2))],
            names=['entity', 'date'],
        )
        data = pd.DataFrame({'value': [1.0, 2.0, 3.0, 4.0]}, index=index)

        with pytest.raises(ValueError, match='before calling FixedSortTransformer, which never sorts implicitly'):
            FixedSortTransformer().fit(data)


# =============================================================================
# Valeurs par défaut et validation désactivée
# =============================================================================
class TestParameterDefaults:
    """Constructor defaults: index-based, validated, strict, unsorted, unconverted."""

    @pytest.mark.parametrize(
        ('name', 'default'),
        [('time_col', None), ('panel_cols', None), ('validate_input', True),
         ('strict_validation', True), ('auto_sort', False), ('convert_cols_to_index', False)],
        ids=lambda value: str(value),
    )
    def test_default_value(self, name, default):
        """Each constructor parameter has its documented default."""
        assert ConcreteTransformer().get_params()[name] is default


class TestValidationCanBeDisabled:
    """``validate_input=False`` hands the caller's data to ``_fit`` and ``_transform`` untouched."""

    def test_unsorted_data_is_passed_through(self, shuffled_daily_frame):
        """Unsorted data in columns is neither sorted, nor indexed, nor rejected."""
        transformer = ConcreteTransformer(time_col='date', validate_input=False)

        output = transformer.fit_transform(shuffled_daily_frame)

        pd.testing.assert_frame_equal(transformer.fit_input_, shuffled_daily_frame)
        assert output is shuffled_daily_frame
