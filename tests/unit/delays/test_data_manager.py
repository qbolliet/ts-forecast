"""Unit tests for ``tsforecast.delays.data_manager``.

Covers the single public symbol of the module, ``compare_and_detect_delays``
(parameters ``new_data``, ``existing_data``, ``download_date``,
``detection_mode``, ``reference_point``, ``delay_unit``, ``time_col`` and
``panel_cols``), through its public API only: the private helpers
``_validate_input_data``, ``_identify_new_observations`` and
``_calculate_publication_delays`` are exercised through it.

Gold values are small enough to be computed by hand. The reference series is
a monthly series ``Jan -> Apr 2023`` (month-start labels) downloaded on
``2023-06-15``: its last observation (April) covers ``[2023-04-01,
2023-05-01)``, hence a delay of 75 days from the period start (30 + 31 + 14)
and 45 days from the period end (31 + 14).

The realistic scenarios (notebook 3 datasets, successive downloads) live in
``tests/integration/delays/test_detect_delays_realistic.py``.

Anomalies found while writing these tests are registered in
``tests/ANOMALIES.md`` (``ANO-DELAYS-004`` to ``ANO-DELAYS-010``, all fixed).
"""
# Modules de base
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Jeux de données et perturbations partagés
from tests.support import perturbations as perturb

# Fonction à tester
from tsforecast.delays.data_manager import compare_and_detect_delays

# Le chemin « colonnes » (time_col / panel_cols) émet systématiquement un
# avertissement de remplacement d'index (comportement de validate_temporal_data,
# ANO-UTILS-033) : le bruit est masqué pour garder une sortie lisible.
pytestmark = pytest.mark.filterwarnings("ignore:Index replaced with")

TS = pd.Timestamp

# Colonnes de sortie, dans l'ordre (`has_changes` marque les observations détectées, ANO-DELAYS-009).
OUTPUT_COLUMNS = [
    'observation_date', 'has_changes', 'download_date', 'frequency',
    'period_start', 'period_end', 'reference_point', 'delay', 'unit',
]


# =============================================================================
# Constructeurs locaux de petits jeux à valeurs d'or calculables
# =============================================================================
def _months(periods: int = 4, start: str = '2023-01-01') -> pd.DatetimeIndex:
    """Build a month-start ``DatetimeIndex`` (``MS``) of ``periods`` dates."""
    return pd.date_range(start, periods=periods, freq='MS')


def _series(values, column: str = 'PIB', start: str = '2023-01-01') -> pd.DataFrame:
    """Build a one-column monthly frame from a list of values (``NaN`` allowed)."""
    return pd.DataFrame({column: values}, index=_months(len(values), start))


def _panel(entities: dict, entity_name: str = 'country') -> pd.DataFrame:
    """Build a monthly panel from per-entity values.

    Args:
        entities: ``{entity: {column: values}}``, every entity starting on
            ``2023-01-01`` (``MS``) with as many months as values.
        entity_name: Name of the entity level.

    Returns:
        A sorted two-level ``MultiIndex`` (entity, ``date``) frame.
    """
    frames = []
    for entity, cols in entities.items():
        n_rows = len(next(iter(cols.values())))
        frame = pd.DataFrame(cols, index=_months(n_rows))
        frame.index = pd.MultiIndex.from_product([[entity], frame.index], names=[entity_name, 'date'])
        frames.append(frame)
    return pd.concat(frames).sort_index()


def _records(result: pd.DataFrame, *fields: str) -> list:
    """Flatten a result into sorted ``(*index_key, *fields)`` tuples for exact comparison."""
    records = []
    for key, row in zip(result.index, result[list(fields)].itertuples(index=False)):
        key = key if isinstance(key, tuple) else (key,)
        records.append((*key, *row))
    return sorted(records, key=repr)


def _expected(*rows: tuple) -> list:
    """Sort hand-written expected rows the same way as :func:`_records`."""
    return sorted(rows, key=repr)


@pytest.fixture
def monthly_series() -> pd.DataFrame:
    """Reference monthly series ``PIB`` = 1..4 on ``2023-01-01 .. 2023-04-01``."""
    return _series([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def small_panel() -> pd.DataFrame:
    """Two-entity monthly panel: ``A`` observed Jan-Apr, ``B`` observed Jan-Mar."""
    return _panel({'A': {'PIB': [1.0, 2.0, 3.0, 4.0]}, 'B': {'PIB': [10.0, 20.0, 30.0]}})


# =============================================================================
# Contrat de la sortie
# =============================================================================
class TestOutputContract:
    """Shape, order and types of the returned frame."""

    def test_columns_and_their_order(self, monthly_series):
        """The output exposes the documented columns plus ``has_changes``, in a fixed order."""
        result = compare_and_detect_delays(monthly_series, None, '2023-06-15')

        assert list(result.columns) == OUTPUT_COLUMNS

    def test_time_series_is_indexed_by_column_name(self, monthly_series):
        """A time series result is indexed by the variable name, the index level being called ``column``."""
        result = compare_and_detect_delays(monthly_series, None, '2023-06-15')

        assert list(result.index) == ['PIB'] and result.index.name == 'column'

    def test_panel_is_indexed_by_entity_then_column(self, small_panel):
        """A panel result keeps the entity level name of the input and appends ``column``."""
        result = compare_and_detect_delays(small_panel, None, '2023-06-15')

        assert list(result.index.names) == ['country', 'column']

    def test_gold_row_of_the_reference_series(self, monthly_series):
        """Every column of the single result row matches the hand-computed value."""
        result = compare_and_detect_delays(monthly_series, None, '2023-06-15')

        # Valeur d'or : dernière observation = avril, période [1er avril, 1er mai),
        # délai depuis le début = 75 j (30 + 31 + 14)
        expected = {
            'observation_date': TS('2023-04-01'), 'has_changes': True,
            'download_date': TS('2023-06-15'), 'frequency': 'monthly',
            'period_start': TS('2023-04-01'), 'period_end': TS('2023-05-01'),
            'reference_point': 'start', 'delay': 75.0, 'unit': 'day',
        }
        assert result.iloc[0].to_dict() == expected

    def test_delays_are_whole_numbers(self, small_panel):
        """Delays are ceil-rounded: every value is an integer (stored as float)."""
        result = compare_and_detect_delays(small_panel, None, '2023-06-15 12:00')

        assert (result['delay'] == np.ceil(result['delay'])).all()

    def test_download_date_is_constant_over_rows(self, small_panel):
        """The ``download_date`` column holds the same resolved date on every row."""
        result = compare_and_detect_delays(small_panel, None, '2023-06-15')

        assert set(result['download_date']) == {TS('2023-06-15')}

    def test_inputs_are_not_modified(self):
        """Neither dataset is changed in place, even when unsorted."""
        new_data = perturb.shuffle_rows(_series([1.0, 2.0, 3.0, 4.0]))
        existing = perturb.shuffle_rows(_series([1.0, 2.0, 3.0, np.nan]))
        new_copy, existing_copy = new_data.copy(), existing.copy()

        compare_and_detect_delays(new_data, existing, '2023-06-15')

        pd.testing.assert_frame_equal(new_data, new_copy)
        pd.testing.assert_frame_equal(existing, existing_copy)


# =============================================================================
# Premier téléchargement : existing_data=None
# =============================================================================
class TestFirstDownload:
    """``existing_data=None``: the most recent non-null observation of each variable."""

    def test_last_observation_per_column(self):
        """Each column yields its own last non-null date, whatever the other columns hold."""
        data = pd.DataFrame(
            {'PIB': [1.0, 2.0, 3.0, 4.0, 5.0, np.nan], 'inflation': [1.0, 2.0, 3.0, np.nan, np.nan, np.nan]},
            index=_months(6),
        )

        result = compare_and_detect_delays(data, None, '2023-08-15')

        # Valeurs d'or : PIB mai -> 15 août = 31 + 30 + 31 + 14 = 106 j ;
        # inflation mars -> 15 août = 31 + 30 + 31 + 30 + 31 + 14 = 167 j
        assert _records(result, 'observation_date', 'delay') == _expected(
            ('PIB', TS('2023-05-01'), 106.0), ('inflation', TS('2023-03-01'), 167.0),
        )

    def test_interior_gap_does_not_move_the_last_observation(self):
        """A missing month inside the history leaves the last observation unchanged."""
        data = _series([1.0, 2.0, np.nan, 4.0, 5.0, np.nan])

        result = compare_and_detect_delays(data, None, '2023-08-15')

        assert _records(result, 'observation_date', 'frequency') == _expected(('PIB', TS('2023-05-01'), 'monthly'))

    def test_all_null_column_is_left_out(self):
        """A column without any observation produces no row; the others are unaffected."""
        data = pd.DataFrame({'PIB': [1.0, 2.0, 3.0, 4.0], 'vide': [np.nan] * 4}, index=_months())

        result = compare_and_detect_delays(data, None, '2023-06-15')

        assert list(result.index) == ['PIB']

    @pytest.mark.parametrize('mode', ['new_only', 'all_changes'])
    def test_detection_mode_is_ignored_without_existing_data(self, monthly_series, mode):
        """Both detection modes give the same result when there is nothing to compare with."""
        reference = compare_and_detect_delays(monthly_series, None, '2023-06-15')

        result = compare_and_detect_delays(monthly_series, None, '2023-06-15', detection_mode=mode)

        pd.testing.assert_frame_equal(result, reference)

    def test_panel_last_observation_per_entity_and_column(self, small_panel):
        """Each (entity, column) couple has its own last date: the panel is unbalanced."""
        result = compare_and_detect_delays(small_panel, None, '2023-06-15')

        # Valeurs d'or : A avril -> 75 j ; B mars -> 15 juin = 31 + 30 + 31 + 14 = 106 j
        assert _records(result, 'observation_date', 'delay') == _expected(
            ('A', 'PIB', TS('2023-04-01'), 75.0), ('B', 'PIB', TS('2023-03-01'), 106.0),
        )

    def test_entity_without_observation_for_a_column_is_left_out(self):
        """An entity entirely NaN on a column yields no row for it (as ``Italie`` / ``climat_affaires``)."""
        panel = _panel({
            'A': {'PIB': [1.0, 2.0, 3.0, 4.0]},
            'B': {'PIB': [np.nan] * 4},
            'C': {'PIB': [5.0, 6.0, 7.0, 8.0]},
        })

        result = compare_and_detect_delays(panel, None, '2023-06-15')

        assert sorted(result.index.get_level_values('country')) == ['A', 'C']

    def test_single_entity_panel(self):
        """A panel with a single entity keeps a two-level result index."""
        panel = _panel({'A': {'PIB': [1.0, 2.0, 3.0, 4.0]}})

        result = compare_and_detect_delays(panel, None, '2023-06-15')

        assert _records(result, 'delay') == _expected(('A', 'PIB', 75.0))


# =============================================================================
# Modes de détection : existing_data fourni
# =============================================================================
class TestDetectionModes:
    """What ``new_only`` and ``all_changes`` consider a new observation."""

    def test_new_only_detects_null_to_value(self):
        """``new_only``: a value filling a former ``NaN`` is a new observation."""
        existing = _series([1.0, 2.0, 3.0, np.nan])

        result = compare_and_detect_delays(_series([1.0, 2.0, 3.0, 4.0]), existing, '2023-06-15')

        assert _records(result, 'observation_date', 'delay') == _expected(('PIB', TS('2023-04-01'), 75.0))

    def test_new_only_ignores_revisions(self):
        """``new_only``: a revised value (value -> other value) is not a new observation."""
        existing = _series([1.0, 2.0, 3.0, 4.0])

        result = compare_and_detect_delays(_series([1.0, 20.0, 3.0, 4.0]), existing, '2023-06-15')

        assert result.empty

    def test_new_only_reports_the_fill_but_not_the_revision(self):
        """``new_only`` on a download holding both a fill and a revision keeps only the fill."""
        existing = _series([1.0, 2.0, 3.0, np.nan])

        result = compare_and_detect_delays(_series([1.0, 20.0, 3.0, 4.0]), existing, '2023-06-15')

        assert _records(result, 'observation_date') == _expected(('PIB', TS('2023-04-01')))

    def test_all_changes_detects_revisions(self):
        """``all_changes``: a revised value is reported, with its delay counted from the period start."""
        existing = _series([1.0, 2.0, 3.0, 4.0])

        result = compare_and_detect_delays(
            _series([1.0, 20.0, 3.0, 4.0]), existing, '2023-06-15', detection_mode='all_changes',
        )

        # Valeur d'or : février -> 15 juin = 28 + 31 + 30 + 31 + 14 = 134 j.
        # Le délai d'une révision est donc compté depuis le début de la période,
        # non depuis la première publication de la valeur.
        assert _records(result, 'observation_date', 'delay') == _expected(('PIB', TS('2023-02-01'), 134.0))

    def test_all_changes_also_detects_null_to_value(self):
        """``all_changes`` is a superset of ``new_only``: fills and revisions are both reported."""
        existing = _series([1.0, 2.0, 3.0, np.nan])

        result = compare_and_detect_delays(
            _series([1.0, 20.0, 3.0, 4.0]), existing, '2023-06-15', detection_mode='all_changes',
        )

        assert _records(result, 'observation_date') == _expected(
            ('PIB', TS('2023-02-01')), ('PIB', TS('2023-04-01')),
        )

    @pytest.mark.parametrize('mode', ['new_only', 'all_changes'])
    def test_unchanged_data_gives_no_row(self, monthly_series, mode):
        """Two identical downloads detect nothing, in both modes."""
        result = compare_and_detect_delays(monthly_series, monthly_series.copy(), '2023-06-15', detection_mode=mode)

        assert result.empty

    @pytest.mark.parametrize('mode', ['new_only', 'all_changes'])
    def test_withdrawn_value_is_not_reported(self, mode):
        """A value turned back into ``NaN`` is reported by neither mode (only appearances and revisions are)."""
        existing = _series([1.0, 2.0, 3.0, 4.0])

        result = compare_and_detect_delays(
            _series([1.0, 2.0, 3.0, np.nan]), existing, '2023-06-15', detection_mode=mode,
        )

        assert result.empty

    def test_nan_to_nan_is_not_a_change_in_all_changes_mode(self):
        """``NaN != NaN`` must not make an unobserved date look changed."""
        data = _series([1.0, np.nan, 3.0, 4.0])

        result = compare_and_detect_delays(data, data.copy(), '2023-06-15', detection_mode='all_changes')

        assert result.empty

    def test_equal_values_of_different_dtypes_are_not_a_change(self):
        """An integer column and its float twin hold the same values: no change."""
        existing = _series([1, 2, 3, 4])

        result = compare_and_detect_delays(
            _series([1.0, 2.0, 3.0, 4.0]), existing, '2023-06-15', detection_mode='all_changes',
        )

        assert result.empty

    @pytest.mark.parametrize('mode', ['new_only', 'all_changes'])
    def test_dates_absent_from_existing_data_are_new(self, mode):
        """Dates the existing download did not cover are new observations (aligned as ``NaN``)."""
        existing = _series([1.0, 2.0, 3.0, 4.0])
        new_data = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        result = compare_and_detect_delays(new_data, existing, '2023-08-15', detection_mode=mode)

        # Valeurs d'or : mai -> 15 août = 31 + 30 + 31 + 14 = 106 j ; juin -> 15 août = 31 + 30 + 14 = 75 j
        assert _records(result, 'observation_date', 'delay') == _expected(
            ('PIB', TS('2023-05-01'), 106.0), ('PIB', TS('2023-06-01'), 75.0),
        )

    @pytest.mark.parametrize('mode', ['new_only', 'all_changes'])
    def test_dates_absent_from_new_data_are_ignored(self, mode):
        """Dates only the existing download covered are not new observations."""
        existing = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        result = compare_and_detect_delays(_series([1.0, 2.0, 3.0, 4.0]), existing, '2023-08-15', detection_mode=mode)

        assert result.empty

    def test_only_common_columns_are_compared(self):
        """A column present in one dataset only is neither compared nor reported."""
        existing = pd.DataFrame({'PIB': [1.0, 2.0, 3.0, np.nan], 'inflation': [1.0, 2.0, 3.0, 4.0]}, index=_months())
        new_data = pd.DataFrame({'PIB': [1.0, 2.0, 3.0, 4.0], 'chomage': [5.0, 6.0, 7.0, 8.0]}, index=_months())

        result = compare_and_detect_delays(new_data, existing, '2023-06-15')

        assert list(result.index) == ['PIB']

    def test_no_common_column_gives_no_row(self):
        """Datasets without a shared column have nothing to compare: empty result."""
        result = compare_and_detect_delays(
            _series([1.0, 2.0, 3.0, 4.0], 'a'), _series([1.0, 2.0, 3.0, 4.0], 'b'), '2023-06-15',
        )

        assert result.empty

    def test_row_and_column_order_of_existing_data_is_irrelevant(self):
        """The comparison aligns on labels: a shuffled, column-reversed copy shows no change."""
        data = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [5.0, 6.0, 7.0, 8.0]}, index=_months())
        existing = perturb.shuffle_rows(data).iloc[:, ::-1]

        result = compare_and_detect_delays(data, existing, '2023-06-15', detection_mode='all_changes')

        assert result.empty

    def test_each_changed_column_is_reported(self):
        """Changes in several columns give one row per (column, date)."""
        existing = pd.DataFrame({'a': [1.0, 2.0, 3.0, np.nan], 'b': [5.0, 6.0, np.nan, np.nan]}, index=_months())
        new_data = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [5.0, 6.0, 7.0, np.nan]}, index=_months())

        result = compare_and_detect_delays(new_data, existing, '2023-06-15')

        # Valeurs d'or : a en avril -> 75 j ; b en mars -> 15 juin = 31 + 30 + 31 + 14 = 106 j
        assert _records(result, 'observation_date', 'delay') == _expected(
            ('a', TS('2023-04-01'), 75.0), ('b', TS('2023-03-01'), 106.0),
        )

    def test_panel_changes_are_reported_per_entity(self):
        """In a panel, a fill is attributed to the entity it happened in."""
        existing = _panel({'A': {'PIB': [1.0, 2.0, 3.0, np.nan]}, 'B': {'PIB': [10.0, 20.0, np.nan, 40.0]}})
        new_data = _panel({'A': {'PIB': [1.0, 2.0, 3.0, 4.0]}, 'B': {'PIB': [10.0, 20.0, 30.0, 40.0]}})

        result = compare_and_detect_delays(new_data, existing, '2023-06-15')

        # Valeurs d'or : A en avril -> 75 j ; B en mars -> 106 j
        assert _records(result, 'observation_date', 'delay') == _expected(
            ('A', 'PIB', TS('2023-04-01'), 75.0), ('B', 'PIB', TS('2023-03-01'), 106.0),
        )

    def test_entity_absent_from_existing_data_is_entirely_new(self, small_panel):
        """An entity unknown to the existing download has all its observed values reported."""
        existing = perturb.drop_entity(small_panel, 'B')

        result = compare_and_detect_delays(small_panel, existing, '2023-06-15')

        assert _records(result, 'observation_date') == _expected(
            ('B', 'PIB', TS('2023-01-01')), ('B', 'PIB', TS('2023-02-01')), ('B', 'PIB', TS('2023-03-01')),
        )

    def test_entity_absent_from_new_data_is_ignored(self, small_panel):
        """An entity only the existing download knew gives no row."""
        result = compare_and_detect_delays(perturb.drop_entity(small_panel, 'B'), small_panel, '2023-06-15')

        assert result.empty

    def test_frequency_is_detected_on_new_data(self):
        """The frequency of each row comes from ``new_data``, not from the older download."""
        # L'ancien téléchargement n'a que deux valeurs espacées d'un trimestre (fréquence trimestrielle) ;
        # le nouveau est dense (fréquence mensuelle) : les périodes des valeurs apparues sont des mois
        existing = _series([1.0, np.nan, np.nan, 4.0, np.nan, np.nan])
        new_data = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        result = compare_and_detect_delays(new_data, existing, '2023-08-15')

        # Valeurs d'or : février, mars, mai et juin apparaissent ; chaque période est un mois (début = date)
        assert _records(result, 'frequency', 'period_start', 'period_end') == _expected(
            ('PIB', 'monthly', TS('2023-02-01'), TS('2023-03-01')),
            ('PIB', 'monthly', TS('2023-03-01'), TS('2023-04-01')),
            ('PIB', 'monthly', TS('2023-05-01'), TS('2023-06-01')),
            ('PIB', 'monthly', TS('2023-06-01'), TS('2023-07-01')),
        )

    def test_invalid_detection_mode_raises(self, monthly_series):
        """An unknown detection mode is rejected."""
        with pytest.raises(ValueError, match="detection_mode must be 'new_only' or 'all_changes'"):
            compare_and_detect_delays(monthly_series, monthly_series, '2023-06-15', detection_mode='everything')

    def test_invalid_detection_mode_raises_even_without_existing_data(self, monthly_series):
        """Mode validation does not depend on ``existing_data`` (although the mode is then unused)."""
        with pytest.raises(ValueError, match="detection_mode must be"):
            compare_and_detect_delays(monthly_series, None, '2023-06-15', detection_mode='everything')


# =============================================================================
# Exemple de la documentation (tutoriel, section 4.1.3)
# =============================================================================
class TestDocumentedExample:
    """The quarterly GDP example of ``docs/tutorials/publication_delays.md`` §4.1.3."""

    @staticmethod
    def _datasets() -> tuple:
        """Return the (existing, new) GDP frames of the tutorial."""
        index = pd.to_datetime(['2023-04-01', '2023-07-01', '2023-10-01', '2024-01-01'])
        existing = pd.DataFrame({'GDP': [1.2, 1.5, np.nan, np.nan]}, index=index)
        new_data = pd.DataFrame({'GDP': [1.2, 1.6, 1.8, 2.1]}, index=index)
        return existing, new_data

    def test_new_only_reports_the_two_new_quarters(self):
        """``new_only``: the two filled quarters, delays counted from the period end."""
        existing, new_data = self._datasets()

        result = compare_and_detect_delays(
            new_data, existing, '2024-04-20', detection_mode='new_only', reference_point='end',
        )

        # Valeurs d'or : fin du T4 2023 = 1er janv. 2024 -> 20 avril 2024 = 31 + 29 + 31 + 19 = 110 j ;
        # fin du T1 2024 = 1er avril -> 20 avril = 19 j
        assert _records(result, 'observation_date', 'frequency', 'delay') == _expected(
            ('GDP', TS('2023-10-01'), 'quarterly', 110.0), ('GDP', TS('2024-01-01'), 'quarterly', 19.0),
        )

    def test_all_changes_adds_the_revised_quarter(self):
        """``all_changes``: the revised Q3 2023 comes on top of the two new quarters."""
        existing, new_data = self._datasets()

        result = compare_and_detect_delays(
            new_data, existing, '2024-04-20', detection_mode='all_changes', reference_point='end',
        )

        # Valeur d'or pour la révision : fin du T3 2023 = 1er oct. 2023 -> 20 avril 2024 = 92 + 110 = 202 j
        assert _records(result, 'observation_date', 'delay') == _expected(
            ('GDP', TS('2023-07-01'), 202.0), ('GDP', TS('2023-10-01'), 110.0), ('GDP', TS('2024-01-01'), 19.0),
        )


# =============================================================================
# Point de référence, unité et arrondi
# =============================================================================
class TestReferencePointAndUnit:
    """Delay measured from the start or the end of the period, in days, seconds or microseconds."""

    @pytest.mark.parametrize(
        ('reference_point', 'expected_delay'),
        [('start', 75.0), ('end', 45.0)],
        ids=['start', 'end'],
    )
    def test_reference_point(self, monthly_series, reference_point, expected_delay):
        """The delay is counted from the period start or from the (exclusive) period end."""
        result = compare_and_detect_delays(monthly_series, None, '2023-06-15', reference_point=reference_point)

        # Valeurs d'or : 1er avril -> 15 juin = 75 j ; 1er mai -> 15 juin = 45 j
        assert result[['reference_point', 'delay']].iloc[0].tolist() == [reference_point, expected_delay]

    def test_reference_point_defaults_to_start(self, monthly_series):
        """Without ``reference_point`` the delay is counted from the period start."""
        result = compare_and_detect_delays(monthly_series, None, '2023-06-15')

        assert result['reference_point'].iloc[0] == 'start'

    def test_invalid_reference_point_raises(self, monthly_series):
        """Only ``'start'`` and ``'end'`` are accepted."""
        with pytest.raises(ValueError, match="reference_point must be 'start' or 'end'"):
            compare_and_detect_delays(monthly_series, None, '2023-06-15', reference_point='middle')

    @pytest.mark.parametrize(
        ('delay_unit', 'expected_unit', 'expected_delay'),
        [
            ('D', 'day', 75.0), ('day', 'day', 75.0),
            ('s', 'second', 6_480_000.0), ('second', 'second', 6_480_000.0),
            ('us', 'microsecond', 6_480_000_000_000.0), ('microsecond', 'microsecond', 6_480_000_000_000.0),
        ],
        ids=['D', 'day', 's', 'second', 'us', 'microsecond'],
    )
    def test_delay_unit(self, monthly_series, delay_unit, expected_unit, expected_delay):
        """Short and long unit names are equivalent; the ``unit`` column carries the long name."""
        result = compare_and_detect_delays(monthly_series, None, '2023-06-15', delay_unit=delay_unit)

        # Valeur d'or : 75 j = 75 * 86 400 s = 6 480 000 s = 6,48e12 µs
        assert result[['unit', 'delay']].iloc[0].tolist() == [expected_unit, expected_delay]

    def test_delay_unit_defaults_to_day(self, monthly_series):
        """Without ``delay_unit`` the delay is in days."""
        result = compare_and_detect_delays(monthly_series, None, '2023-06-15')

        assert result['unit'].iloc[0] == 'day'

    @pytest.mark.parametrize('delay_unit', ['hour', 'invalid', 'Day', 'ms'])
    def test_invalid_delay_unit_raises(self, monthly_series, delay_unit):
        """Units other than day / second / microsecond are rejected."""
        with pytest.raises(ValueError, match="Unit must be one of"):
            compare_and_detect_delays(monthly_series, None, '2023-06-15', delay_unit=delay_unit)

    @pytest.mark.parametrize(
        ('download_date', 'delay_unit', 'expected_delay'),
        [
            ('2023-06-15 12:00', 'day', 76.0),
            ('2023-06-15 00:00:00.5', 'second', 6_480_001.0),
            ('2023-06-15 00:00:00.000001', 'second', 6_480_001.0),
            ('2023-06-15 00:00:00.000001', 'day', 76.0),
        ],
        ids=['half-day', 'half-second', 'one-microsecond-in-seconds', 'one-microsecond-in-days'],
    )
    def test_delay_is_rounded_up(self, monthly_series, download_date, delay_unit, expected_delay):
        """Any fraction of the unit rounds the delay up to the next integer."""
        result = compare_and_detect_delays(monthly_series, None, download_date, delay_unit=delay_unit)

        # Valeurs d'or : 75,5 j -> 76 ; 6 480 000,5 s -> 6 480 001 ; 75 j + 1 µs -> 76 j
        assert result['delay'].iloc[0] == expected_delay

    def test_exact_whole_unit_is_not_rounded_up(self, monthly_series):
        """A delay of exactly 75 days stays 75 (no spurious ``+ 1``)."""
        result = compare_and_detect_delays(monthly_series, None, '2023-06-15 00:00:00')

        assert result['delay'].iloc[0] == 75.0

    def test_download_before_the_period_start_gives_a_negative_delay(self, monthly_series):
        """A download dated before the observed period yields a negative delay, rounded towards ``+inf``."""
        result = compare_and_detect_delays(monthly_series, None, '2023-03-20 12:00')

        # Valeur d'or : 20 mars 12:00 -> 1er avril = 11,5 j avant, soit -11,5 arrondi à -11.
        # Le signe est conservé (cohérent avec calculate_applicable_delay), sans erreur ni avertissement.
        assert result['delay'].iloc[0] == -11.0

    @pytest.mark.parametrize(
        'download_date',
        [
            datetime(2027, 11, 26, 17, 18, 56, 578903),
            datetime(2027, 3, 25, 11, 39, 36, 422959),
            datetime(2026, 12, 26, 8, 35, 13, 531343),
        ],
        ids=['2027-11', '2027-03', '2026-12'],
    )
    def test_microsecond_delay_is_exact(self, monthly_series, download_date):
        """The microsecond delay equals the exact integer count of microseconds elapsed."""
        result = compare_and_detect_delays(monthly_series, None, download_date, delay_unit='us')

        # Valeur d'or par arithmétique entière : (téléchargement - 1er avril 2023) // 1 µs
        exact = (download_date - datetime(2023, 4, 1)) // timedelta(microseconds=1)
        assert int(result['delay'].iloc[0]) == exact

    def test_microsecond_delay_is_exact_over_decades(self):
        """Thirty-three years of delay are still counted to the microsecond."""
        data = pd.DataFrame({'PIB': [1.0, 2.0, 3.0]}, index=pd.date_range('1990-01-01', periods=3, freq='MS'))
        download_date = datetime(2023, 12, 15, 12, 34, 56, 123457)

        result = compare_and_detect_delays(data, None, download_date, delay_unit='us')

        # Valeur d'or par arithmétique entière : (téléchargement - 1er mars 1990) // 1 µs
        exact = (download_date - datetime(1990, 3, 1)) // timedelta(microseconds=1)
        assert int(result['delay'].iloc[0]) == exact


# =============================================================================
# Date de téléchargement
# =============================================================================
class TestDownloadDate:
    """Accepted forms of ``download_date`` and their resolution."""

    @pytest.mark.parametrize(
        'download_date',
        ['2023-06-15', '2023-06-15 00:00:00', datetime(2023, 6, 15), TS('2023-06-15')],
        ids=['iso-date', 'iso-datetime', 'datetime', 'timestamp'],
    )
    def test_equivalent_forms_give_the_same_result(self, monthly_series, download_date):
        """A string, a ``datetime`` and a ``Timestamp`` naming the same instant are interchangeable."""
        result = compare_and_detect_delays(monthly_series, None, download_date)

        assert result[['download_date', 'delay']].iloc[0].tolist() == [TS('2023-06-15'), 75.0]

    def test_time_of_day_is_kept(self, monthly_series):
        """The time of day of a datetime is kept in the ``download_date`` column."""
        result = compare_and_detect_delays(monthly_series, None, datetime(2023, 6, 15, 12, 30, 45))

        assert result['download_date'].iloc[0] == TS('2023-06-15 12:30:45')

    def test_none_uses_the_current_time(self, monthly_series):
        """``download_date=None`` stamps the rows with the current time."""
        before = datetime.now()
        result = compare_and_detect_delays(monthly_series, None, None)
        after = datetime.now()

        assert before <= result['download_date'].iloc[0] <= after

    def test_default_is_the_current_time(self, monthly_series):
        """Omitting ``download_date`` behaves as ``None``."""
        before = datetime.now()
        result = compare_and_detect_delays(monthly_series)
        after = datetime.now()

        assert before <= result['download_date'].iloc[0] <= after

    def test_today_string_uses_the_current_time(self, monthly_series):
        """The string ``'today'`` (handled by ``resolve_date``) means the current time."""
        before = datetime.now()
        result = compare_and_detect_delays(monthly_series, None, 'today')
        after = datetime.now()

        assert before <= result['download_date'].iloc[0] <= after

    def test_timezone_aware_download_date_with_timezone_aware_data(self):
        """Aware data and an aware download date in the same zone give the same delay as naive ones."""
        data = _series([1.0, 2.0, 3.0, 4.0])
        data.index = data.index.tz_localize('Europe/Paris')

        result = compare_and_detect_delays(data, None, TS('2023-06-15', tz='Europe/Paris'))

        assert result['delay'].iloc[0] == 75.0

    @pytest.mark.parametrize(
        'download_date',
        [TS('2023-06-15', tz='UTC'), TS('2023-06-15 02:00', tz='Europe/Paris'), TS('2023-06-14 20:00', tz='America/New_York')],
        ids=['utc', 'paris', 'new-york'],
    )
    def test_timezone_aware_download_date_with_naive_data(self, monthly_series, download_date):
        """Naive data are read as UTC: an aware download date is compared as an instant."""
        result = compare_and_detect_delays(monthly_series, None, download_date)

        # Valeur d'or : 2023-06-15 00:00 UTC pour les trois dates (Paris est à UTC+2 en juin, New York à UTC-4),
        # soit 75 j après le 1er avril 00:00 lu en UTC
        assert result['delay'].iloc[0] == 75.0

    def test_naive_download_date_with_timezone_aware_data(self):
        """A naive download date is read as UTC when the data carry a time zone."""
        data = _series([1.0, 2.0, 3.0, 4.0])
        data.index = data.index.tz_localize('Europe/Paris')

        result = compare_and_detect_delays(data, None, '2023-06-15')

        # Valeur d'or : 1er avril 00:00 à Paris (UTC+2) = 31 mars 22:00 UTC ; 31 mars 22:00 -> 15 juin 00:00
        # = 75 j + 2 h, arrondi à 76 j
        assert result['delay'].iloc[0] == 76.0

    def test_aware_download_date_in_another_zone_than_the_data(self):
        """Aware data and an aware download date of another zone are compared as instants."""
        data = _series([1.0, 2.0, 3.0, 4.0])
        data.index = data.index.tz_localize('Europe/Paris')

        result = compare_and_detect_delays(data, None, TS('2023-06-14 18:00', tz='America/New_York'))

        # Valeur d'or : 18:00 à New York (UTC-4) = 22:00 UTC le 14 juin ; du 31 mars 22:00 UTC au 14 juin 22:00 UTC = 75 j
        assert result['delay'].iloc[0] == 75.0

    def test_download_date_column_keeps_the_given_time_zone(self, monthly_series):
        """The ``download_date`` column holds the date as given, time zone included."""
        given = TS('2023-06-15 02:00', tz='Europe/Paris')

        result = compare_and_detect_delays(monthly_series, None, given)

        assert result['download_date'].iloc[0] == given

    def test_timezone_aware_panel_with_timezone_aware_download_date(self, small_panel):
        """The comparison of instants also holds for panels."""
        data = small_panel.copy()
        data.index = data.index.set_levels(data.index.levels[1].tz_localize('UTC'), level='date')

        result = compare_and_detect_delays(data, None, TS('2023-06-15', tz='UTC'))

        assert _records(result, 'delay') == _expected(('A', 'PIB', 75.0), ('B', 'PIB', 106.0))

    @pytest.mark.parametrize('download_date', ['', 'not a date'], ids=['empty-string', 'garbage'])
    def test_unresolvable_string_raises(self, monthly_series, download_date):
        """A string that is not a date is refused."""
        with pytest.raises(ValueError):
            compare_and_detect_delays(monthly_series, None, download_date)

    def test_nat_raises(self, monthly_series):
        """``NaT`` is refused instead of producing ``NaN`` delays."""
        with pytest.raises(ValueError, match="NaT"):
            compare_and_detect_delays(monthly_series, None, pd.NaT)

    @pytest.mark.parametrize('download_date', [date(2023, 6, 15), 20230615, 1.5], ids=['date', 'int', 'float'])
    def test_other_types_raise(self, monthly_series, download_date):
        """Only strings and ``datetime`` objects are accepted (a plain ``date`` is not a ``datetime``)."""
        with pytest.raises(ValueError, match="'date' must be 'today', a string or datetime"):
            compare_and_detect_delays(monthly_series, None, download_date)


# =============================================================================
# Fréquence détectée par (entité, colonne) et bornes de période
# =============================================================================
class TestFrequencyAndPeriodBoundaries:
    """The frequency is detected per (entity, column) and drives the period bounds."""

    @pytest.mark.parametrize(
        ('index', 'download_date', 'delay_unit', 'frequency', 'period_start', 'period_end', 'delay_from_start', 'delay_from_end'),
        [
            (pd.date_range('2023-03-01', '2023-03-10', freq='D'), '2023-03-20', 'day',
             'daily', '2023-03-10', '2023-03-11', 10.0, 9.0),
            (pd.date_range('2023-02-26', periods=4, freq='W-SUN'), '2023-03-20', 'day',
             'weekly', '2023-03-13', '2023-03-20', 7.0, 0.0),
            (pd.date_range('2023-01-01', periods=4, freq='MS'), '2023-06-15', 'day',
             'monthly', '2023-04-01', '2023-05-01', 75.0, 45.0),
            (pd.date_range('2023-01-01', periods=4, freq='QS'), '2024-02-15', 'day',
             'quarterly', '2023-10-01', '2024-01-01', 137.0, 45.0),
            (pd.date_range('2020-01-01', periods=4, freq='YS'), '2024-02-15', 'day',
             'annual', '2023-01-01', '2024-01-01', 410.0, 45.0),
            (pd.date_range('2023-03-01 00:00', periods=5, freq='h'), '2023-03-01 06:30', 'second',
             'hourly', '2023-03-01 04:00', '2023-03-01 05:00', 9000.0, 5400.0),
        ],
        ids=['daily', 'weekly', 'monthly', 'quarterly', 'annual', 'hourly'],
    )
    @pytest.mark.parametrize('reference_point', ['start', 'end'])
    def test_frequency_and_bounds_of_the_last_period(
        self, index, download_date, delay_unit, frequency, period_start, period_end,
        delay_from_start, delay_from_end, reference_point,
    ):
        """The last observation's period is the one of the detected frequency, and the delay follows."""
        data = pd.DataFrame({'v': np.arange(len(index), dtype=float)}, index=index)

        result = compare_and_detect_delays(
            data, None, download_date, reference_point=reference_point, delay_unit=delay_unit,
        )

        expected_delay = delay_from_start if reference_point == 'start' else delay_from_end
        assert result[['frequency', 'period_start', 'period_end', 'delay']].iloc[0].tolist() == [
            frequency, TS(period_start), TS(period_end), expected_delay,
        ]

    def test_frequency_is_a_property_of_the_entity_and_column_couple(self, mixed_freq_panel_multifrequency):
        """One column observed annually, quarterly and monthly in three entities gets three frequencies."""
        result = compare_and_detect_delays(mixed_freq_panel_multifrequency, None, '2024-02-15')

        # Valeurs d'or (index de fin de mois, dernière date 2023-12-31) :
        #   mensuel   : période [1er déc. 2023, 1er janv. 2024) -> 15 fév. : 31 + 31 + 14 = 76 j
        #   trimestr. : période [1er oct. 2023, ...)            -> 15 fév. : 31 + 30 + 31 + 31 + 14 = 137 j
        #   annuel    : période [1er janv. 2023, ...)           -> 15 fév. : 365 + 31 + 14 = 410 j
        assert _records(result, 'frequency', 'period_start', 'delay') == _expected(
            ('DE', 'm1', 'monthly', TS('2023-12-01'), 76.0),
            ('FR', 'm1', 'monthly', TS('2023-12-01'), 76.0),
            ('IT', 'm1', 'monthly', TS('2023-12-01'), 76.0),
            ('DE', 'q1', 'quarterly', TS('2023-10-01'), 137.0),
            ('FR', 'q1', 'quarterly', TS('2023-10-01'), 137.0),
            ('IT', 'q1', 'quarterly', TS('2023-10-01'), 137.0),
            ('FR', 'v', 'annual', TS('2023-01-01'), 410.0),
            ('DE', 'v', 'quarterly', TS('2023-10-01'), 137.0),
            ('IT', 'v', 'monthly', TS('2023-12-01'), 76.0),
        )

    def test_one_column_per_frequency_in_a_time_series(self):
        """Columns of different frequencies in one frame each get their own period bounds."""
        data = pd.DataFrame(
            {
                'mensuelle': np.arange(12, dtype=float),
                'trimestrielle': [1.0, np.nan, np.nan, 2.0, np.nan, np.nan, 3.0, np.nan, np.nan, 4.0, np.nan, np.nan],
            },
            index=_months(12),
        )

        result = compare_and_detect_delays(data, None, '2024-02-15')

        # Valeurs d'or : mensuelle -> décembre 2023, 15 fév. : 76 j ;
        # trimestrielle -> dernier trimestre observé = octobre 2023 : 137 j
        assert _records(result, 'frequency', 'period_start', 'delay') == _expected(
            ('mensuelle', 'monthly', TS('2023-12-01'), 76.0),
            ('trimestrielle', 'quarterly', TS('2023-10-01'), 137.0),
        )

    @pytest.mark.parametrize(
        ('anchored', 'last_observation'),
        [
            (lambda df: df, '2023-04-01'),
            (perturb.to_period_end, '2023-04-30'),
        ],
        ids=['start-anchored', 'end-anchored'],
    )
    def test_period_position_of_the_index_does_not_change_the_period(self, monthly_series, anchored, last_observation):
        """Month-start and month-end labels designate the same period, hence the same delay."""
        result = compare_and_detect_delays(anchored(monthly_series), None, '2023-06-15')

        assert result[['observation_date', 'period_start', 'period_end', 'delay']].iloc[0].tolist() == [
            TS(last_observation), TS('2023-04-01'), TS('2023-05-01'), 75.0,
        ]

    @pytest.mark.parametrize('reference_point', ['start', 'end'])
    def test_undetectable_frequency_gives_a_row_without_period_or_delay(self, reference_point):
        """A single observation has no detectable frequency: the row is kept, without period nor delay."""
        with pytest.warns(UserWarning, match='PIB'):
            result = compare_and_detect_delays(_series([100.0]), None, '2023-02-15', reference_point=reference_point)

        row = result.iloc[0]
        assert list(result.index) == ['PIB'] and row['observation_date'] == TS('2023-01-01')
        assert row['frequency'] is None
        assert pd.isna(row['period_start']) and pd.isna(row['period_end']) and pd.isna(row['delay'])
        assert (row['unit'], row['reference_point'], row['has_changes']) == ('day', reference_point, True)

    def test_undetectable_frequency_warning_names_the_column(self):
        """The warning points at the variable whose frequency could not be detected."""
        with pytest.warns(UserWarning, match=r"frequency could not be detected for \['PIB'\]"):
            compare_and_detect_delays(_series([100.0]), None, '2023-02-15')

    def test_one_undetectable_entity_does_not_spoil_the_panel(self):
        """An entity with a single observation is returned with ``frequency=None``; the others are computed."""
        panel = _panel({'A': {'PIB': [1.0, 2.0, 3.0, 4.0]}, 'B': {'PIB': [1.0, np.nan, np.nan, np.nan]}})

        with pytest.warns(UserWarning, match=r"\('B', 'PIB'\)") as caught:
            result = compare_and_detect_delays(panel, None, '2023-06-15')

        rows = {key: row for key, row in result.iterrows()}
        assert rows[('A', 'PIB')][['frequency', 'delay']].tolist() == ['monthly', 75.0]
        assert rows[('B', 'PIB')]['frequency'] is None and pd.isna(rows[('B', 'PIB')]['delay'])
        # Le message ne cite que le couple indétectable
        assert "('A', 'PIB')" not in str(caught[0].message)

    def test_undetectable_frequency_in_comparison(self):
        """The same holds on the comparison path: a lone new observation is reported without delay."""
        data = _series([1.0, np.nan, np.nan, np.nan])
        existing = _series([np.nan] * 4)

        with pytest.warns(UserWarning, match='PIB'):
            result = compare_and_detect_delays(data, existing, '2023-06-15')

        assert list(result.index) == ['PIB'] and pd.isna(result['delay'].iloc[0])

    def test_single_observation_series_gives_a_row_without_delay(self):
        """A frame cut down to its first row behaves like any single observation."""
        with pytest.warns(UserWarning, match='PIB'):
            result = compare_and_detect_delays(
                perturb.single_observation(_series([1.0, 2.0, 3.0, 4.0])), None, '2023-06-15',
            )

        assert result['frequency'].tolist() == [None] and pd.isna(result['delay'].iloc[0])


# =============================================================================
# Présentation des données en entrée
# =============================================================================
class TestInputLayout:
    """Index kinds, column-based layouts, disorder and unusual names."""

    def test_time_col_gives_the_same_result_as_the_index(self, monthly_series):
        """Passing the dates in a column (``time_col``) is equivalent to passing them as the index."""
        reference = compare_and_detect_delays(monthly_series, None, '2023-06-15')

        result = compare_and_detect_delays(
            monthly_series.rename_axis('date').reset_index(), None, '2023-06-15', time_col='date',
        )

        pd.testing.assert_frame_equal(result, reference)

    def test_time_col_applies_to_both_datasets(self):
        """``time_col`` is used to read ``new_data`` and ``existing_data`` alike."""
        existing = _series([1.0, 2.0, 3.0, np.nan]).rename_axis('date').reset_index()
        new_data = _series([1.0, 2.0, 3.0, 4.0]).rename_axis('date').reset_index()

        result = compare_and_detect_delays(new_data, existing, '2023-06-15', time_col='date')

        assert _records(result, 'observation_date', 'delay') == _expected(('PIB', TS('2023-04-01'), 75.0))

    def test_unknown_time_col_raises(self, monthly_series):
        """A ``time_col`` that is not a column is refused."""
        with pytest.raises(ValueError, match="Time column 'absent' not found"):
            compare_and_detect_delays(monthly_series.reset_index(), None, '2023-06-15', time_col='absent')

    def test_panel_cols_give_the_same_result_as_a_multiindex(self, small_panel):
        """Entity and date columns (``panel_cols`` + ``time_col``) are equivalent to a two-level index."""
        reference = compare_and_detect_delays(small_panel, None, '2023-06-15')

        result = compare_and_detect_delays(
            small_panel.reset_index(), None, '2023-06-15', time_col='date', panel_cols=['country'],
        )

        pd.testing.assert_frame_equal(result, reference)

    def test_panel_cols_with_existing_data(self, small_panel):
        """The column-based layout also works when comparing two panels."""
        existing = small_panel.copy()
        existing.iloc[3, 0] = np.nan  # (A, avril) retiré
        reference = compare_and_detect_delays(small_panel, existing, '2023-06-15')

        result = compare_and_detect_delays(
            small_panel.reset_index(), existing.reset_index(), '2023-06-15', time_col='date', panel_cols=['country'],
        )

        pd.testing.assert_frame_equal(result, reference)

    def test_string_dates_in_the_index_are_converted(self, monthly_series):
        """An index of ISO date strings is converted to dates."""
        reference = compare_and_detect_delays(monthly_series, None, '2023-06-15')
        data = monthly_series.copy()
        data.index = data.index.strftime('%Y-%m-%d')

        result = compare_and_detect_delays(data, None, '2023-06-15')

        pd.testing.assert_frame_equal(result, reference)

    def test_unsorted_time_series_is_sorted(self):
        """A shuffled series gives the result of the sorted one (the last date is the latest, not the last row)."""
        sorted_data = _series([1.0, 2.0, 3.0, 4.0])

        result = compare_and_detect_delays(perturb.shuffle_rows(sorted_data, seed=3), None, '2023-06-15')

        pd.testing.assert_frame_equal(result, compare_and_detect_delays(sorted_data, None, '2023-06-15'))

    def test_unsorted_panel_is_sorted(self, small_panel):
        """A shuffled panel gives the result of the sorted one."""
        result = compare_and_detect_delays(perturb.shuffle_rows(small_panel, seed=3), None, '2023-06-15')

        pd.testing.assert_frame_equal(result, compare_and_detect_delays(small_panel, None, '2023-06-15'))

    def test_unsorted_panel_with_existing_data(self, small_panel):
        """Disorder in both datasets does not change what is detected."""
        existing = small_panel.copy()
        existing.iloc[3, 0] = np.nan  # (A, avril) retiré
        reference = compare_and_detect_delays(small_panel, existing, '2023-06-15')

        result = compare_and_detect_delays(
            perturb.shuffle_rows(small_panel, seed=1), perturb.shuffle_rows(existing, seed=2), '2023-06-15',
        )

        pd.testing.assert_frame_equal(result.sort_index(), reference.sort_index())

    def test_entities_in_reverse_order(self, small_panel):
        """Entity blocks in reverse order give the same set of rows."""
        reference = compare_and_detect_delays(small_panel, None, '2023-06-15')

        result = compare_and_detect_delays(perturb.reverse_entities(small_panel), None, '2023-06-15')

        pd.testing.assert_frame_equal(result.sort_index(), reference.sort_index())

    def test_non_standard_index_names_are_preserved(self, small_panel):
        """Entity level names are not assumed: the result index reuses the input's entity name."""
        data = perturb.with_index_names(small_panel, ['pays', 'jour'])

        result = compare_and_detect_delays(data, None, '2023-06-15')

        assert list(result.index.names) == ['pays', 'column']

    def test_unnamed_time_index(self, monthly_series):
        """A time index without name works (the dates leave the index under ``observation_date``)."""
        data = perturb.with_index_names(monthly_series, None)

        result = compare_and_detect_delays(data, None, '2023-06-15')

        assert _records(result, 'observation_date', 'delay') == _expected(('PIB', TS('2023-04-01'), 75.0))

    def test_three_level_panel(self, small_panel):
        """Every entity level before the dates stays in the result index."""
        data = perturb.to_three_level_index(small_panel, {'A': 'Nord', 'B': 'Sud'})

        result = compare_and_detect_delays(data, None, '2023-06-15')

        assert _records(result, 'observation_date', 'delay') == _expected(
            ('Nord', 'A', 'PIB', TS('2023-04-01'), 75.0), ('Sud', 'B', 'PIB', TS('2023-03-01'), 106.0),
        )
        assert list(result.index.names) == ['region', 'country', 'column']

    def test_three_level_panel_with_existing_data(self, small_panel):
        """Three-level panels are also compared level by level."""
        data = perturb.to_three_level_index(small_panel, {'A': 'Nord', 'B': 'Sud'})
        existing = data.copy()
        existing.iloc[3, 0] = np.nan  # (Nord, A, avril) retiré

        result = compare_and_detect_delays(data, existing, '2023-06-15')

        assert _records(result, 'observation_date', 'delay') == _expected(
            ('Nord', 'A', 'PIB', TS('2023-04-01'), 75.0),
        )

    def test_period_index_is_read_at_the_first_instant(self, monthly_series):
        """A ``PeriodIndex`` is converted to the first instant of each period (ANO-UTILS-029)."""
        data = monthly_series.copy()
        data.index = data.index.to_period('M')

        result = compare_and_detect_delays(data, None, '2023-06-15')

        pd.testing.assert_frame_equal(result, compare_and_detect_delays(monthly_series, None, '2023-06-15'))

    def test_period_index_panel(self, small_panel):
        """A panel whose date level is a ``PeriodIndex`` behaves like its ``DatetimeIndex`` twin."""
        result = compare_and_detect_delays(perturb.to_period_index(small_panel), None, '2023-06-15')

        pd.testing.assert_frame_equal(result, compare_and_detect_delays(small_panel, None, '2023-06-15'))

    def test_special_column_names_are_kept_verbatim(self):
        """Spaces, accents, ``%`` and ``/`` in column names reach the ``column`` index untouched."""
        data, mapping = perturb.with_special_column_names(
            pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [5.0, 6.0, 7.0, 8.0]}, index=_months())
        )

        result = compare_and_detect_delays(data, None, '2023-06-15')

        assert sorted(result.index) == sorted(mapping.values())

    def test_special_column_names_with_existing_data(self):
        """The comparison path also keeps special names intact."""
        data, mapping = perturb.with_special_column_names(
            pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [5.0, 6.0, 7.0, 8.0]}, index=_months())
        )
        existing = data.copy()
        existing.iloc[3] = np.nan

        result = compare_and_detect_delays(data, existing, '2023-06-15')

        assert sorted(result.index) == sorted(mapping.values())

    @pytest.mark.parametrize(
        'name',
        ['column', 'delay', 'frequency', 'observation_date', 'unit', 'download_date', 'period_start', 'reference_point', 'index'],
    )
    @pytest.mark.parametrize('with_existing_data', [False, True], ids=['first-download', 'comparison'])
    def test_column_named_like_an_output_column(self, name, with_existing_data):
        """A data column sharing the name of an output column neither collides nor disappears."""
        data = pd.DataFrame({name: [1.0, 2.0, 3.0, 4.0], 'x': [5.0, 6.0, 7.0, 8.0]}, index=_months())
        existing = None
        if with_existing_data:
            existing = data.copy()
            existing.iloc[3] = np.nan

        result = compare_and_detect_delays(data, existing, '2023-06-15')

        assert sorted(result.index) == sorted([name, 'x'])

    def test_column_named_has_changes_in_comparison(self):
        """A data column called ``has_changes`` is compared like any other."""
        data = pd.DataFrame({'has_changes': [1.0, 2.0, 3.0, 4.0], 'x': [5.0, 6.0, 7.0, 8.0]}, index=_months())
        existing = data.copy()
        existing.iloc[3] = np.nan

        result = compare_and_detect_delays(data, existing, '2023-06-15')

        assert sorted(result.index) == ['has_changes', 'x']

    def test_column_named_has_changes_on_first_download(self):
        """On a first download the same column name is harmless."""
        data = pd.DataFrame({'has_changes': [1.0, 2.0, 3.0, 4.0]}, index=_months())

        result = compare_and_detect_delays(data, None, '2023-06-15')

        assert list(result.index) == ['has_changes']

    def test_duplicated_dates_raise(self):
        """A repeated date cannot be placed on a period: refused by the validation."""
        data = perturb.with_duplicated_rows(_series([1.0, 2.0, 3.0, 4.0]))

        with pytest.raises(ValueError, match="duplicate"):
            compare_and_detect_delays(data, None, '2023-06-15')

    def test_duplicated_panel_rows_raise(self, small_panel):
        """A repeated (entity, date) pair is refused."""
        with pytest.raises(ValueError, match="duplicate"):
            compare_and_detect_delays(perturb.with_duplicated_rows(small_panel), None, '2023-06-15')

    def test_duplicated_dates_in_existing_data_raise(self, monthly_series):
        """The validation also applies to ``existing_data``."""
        with pytest.raises(ValueError, match="duplicate"):
            compare_and_detect_delays(monthly_series, perturb.with_duplicated_rows(monthly_series), '2023-06-15')


# =============================================================================
# Jeux vides ou dégénérés
# =============================================================================
class TestEmptyAndDegenerateInputs:
    """Nothing to report, nothing observed, nothing at all."""

    def test_no_change_gives_an_empty_frame_with_the_output_schema(self, monthly_series):
        """Identical downloads give zero rows but the full set of output columns."""
        result = compare_and_detect_delays(monthly_series, monthly_series.copy(), '2023-06-15')

        assert result.empty and list(result.columns) == OUTPUT_COLUMNS

    def test_no_change_in_a_panel_keeps_the_index_levels(self, small_panel):
        """An empty panel result still has the (entity, ``column``) index levels."""
        result = compare_and_detect_delays(small_panel, small_panel.copy(), '2023-06-15')

        assert result.empty and list(result.index.names) == ['country', 'column']

    def test_all_null_data_in_comparison_gives_an_empty_frame(self):
        """With ``existing_data`` given, a dataset without any observation reports nothing."""
        data = _series([np.nan] * 4)

        result = compare_and_detect_delays(data, data.copy(), '2023-06-15')

        assert result.empty

    def test_all_null_data_on_first_download_gives_an_empty_frame(self):
        """Without ``existing_data``, a series with rows but no observation reports nothing (as the comparison path does)."""
        result = compare_and_detect_delays(_series([np.nan] * 5), None, '2023-06-15')

        assert result.empty and list(result.columns) == OUTPUT_COLUMNS

    def test_all_null_panel_on_first_download_gives_an_empty_frame(self, small_panel):
        """Same expectation for a panel without any observation."""
        result = compare_and_detect_delays(small_panel * np.nan, None, '2023-06-15')

        assert result.empty and list(result.index.names) == ['country', 'column']

    def test_dataset_without_rows_raises_a_clear_error(self, monthly_series):
        """A dataset with no row at all is refused with an explicit ``ValueError``."""
        with pytest.raises(ValueError, match="empty"):
            compare_and_detect_delays(perturb.empty_like(monthly_series), None, '2023-06-15')

    def test_panel_without_rows_raises_a_clear_error(self, small_panel):
        """A panel with no row at all is refused with an explicit ``ValueError``."""
        with pytest.raises(ValueError, match="empty"):
            compare_and_detect_delays(perturb.empty_like(small_panel), None, '2023-06-15')

    @pytest.mark.parametrize('with_existing_data', [False, True], ids=['first-download', 'comparison'])
    def test_frame_without_columns_gives_an_empty_frame(self, with_existing_data):
        """Dates but no variable: nothing can be observed, hence an empty result."""
        data = pd.DataFrame(index=_months())

        result = compare_and_detect_delays(data, data.copy() if with_existing_data else None, '2023-06-15')

        assert result.empty

    def test_empty_results_of_every_path_are_identical(self, monthly_series):
        """'Nothing changed', 'nothing observed' and 'no column' give the very same empty frame."""
        unchanged = compare_and_detect_delays(monthly_series, monthly_series.copy(), '2023-06-15')
        unobserved = compare_and_detect_delays(_series([np.nan] * 4), None, '2023-06-15')
        no_column = compare_and_detect_delays(pd.DataFrame(index=_months()), None, '2023-06-15')

        pd.testing.assert_frame_equal(unobserved, unchanged)
        pd.testing.assert_frame_equal(no_column, unchanged)

    def test_empty_new_data_with_existing_data_raises(self, monthly_series):
        """An empty new download is refused even when there is something to compare it with."""
        with pytest.raises(ValueError, match="empty"):
            compare_and_detect_delays(perturb.empty_like(monthly_series), monthly_series, '2023-06-15')

    def test_empty_existing_data_makes_every_observation_new(self, monthly_series):
        """An existing download without any row is 'nothing known yet': all observed values are new."""
        result = compare_and_detect_delays(monthly_series, perturb.empty_like(monthly_series), '2023-06-15')

        assert len(result) == 4

    @pytest.mark.parametrize('argument', ['new_data', 'existing_data'])
    def test_series_instead_of_frame_raises(self, monthly_series, argument):
        """Only DataFrames are accepted: a Series is refused with an explicit ``TypeError``."""
        arguments = {'new_data': monthly_series, 'existing_data': monthly_series}
        arguments[argument] = monthly_series['PIB']

        with pytest.raises(TypeError, match=f"{argument} must be a pandas DataFrame"):
            compare_and_detect_delays(**arguments, download_date='2023-06-15')

    def test_two_observations_are_enough(self):
        """The minimal usable series has two observations (one step gives a frequency)."""
        result = compare_and_detect_delays(_series([1.0, 2.0]), None, '2023-04-15')

        # Valeur d'or : dernière observation = février, période [1er fév., 1er mars) -> 15 avril :
        # 28 + 31 + 14 = 73 j depuis le début
        assert _records(result, 'frequency', 'delay') == _expected(('PIB', 'monthly', 73.0))

    def test_delay_from_end_is_exact_for_two_observations(self):
        """Gold value of the former 'precision' test, now exact (44 days after the end of February)."""
        result = compare_and_detect_delays(_series([100.0, 101.0]), None, datetime(2023, 4, 14), reference_point='end')

        # Valeur d'or : fin de la période de février = 1er mars (exclusif) -> 14 avril = 31 + 13 = 44 j
        assert result['delay'].iloc[0] == 44.0

    def test_old_data_gives_a_large_delay(self):
        """Thirty-three years of delay are computed in days without overflow."""
        data = pd.DataFrame({'PIB': [100.0, 101.0, 102.0]}, index=pd.date_range('1990-01-01', periods=3, freq='MS'))

        result = compare_and_detect_delays(data, None, '2023-12-15')

        # Valeur d'or : 1er mars 1990 -> 15 déc. 2023 = 33 ans (33 * 365 + 8 jours bissextiles) = 12 053 j
        # jusqu'au 1er mars 2023, puis 275 j jusqu'au 1er déc. et 14 j : 12 342 j
        assert result['delay'].iloc[0] == 12342.0


# =============================================================================
# Avertissements
# =============================================================================
class TestWarnings:
    """Warnings emitted by a call."""

    def test_time_series_call_is_silent(self, monthly_series):
        """A plain time-series call emits no warning at all."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            compare_and_detect_delays(monthly_series, None, '2023-06-15')

    def test_comparison_call_is_silent(self, monthly_series):
        """A time-series comparison emits no warning at all."""
        existing = monthly_series.copy()
        existing.iloc[3, 0] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            compare_and_detect_delays(monthly_series, existing, '2023-06-15')

    @pytest.mark.parametrize('with_existing_data', [False, True], ids=['first-download', 'comparison'])
    def test_panel_call_is_silent(self, small_panel, with_existing_data):
        """A two-level panel call emits no warning, pandas deprecation warnings included (ANO-DELAYS-008)."""
        existing = None
        if with_existing_data:
            existing = small_panel.copy()
            existing.iloc[3, 0] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            compare_and_detect_delays(small_panel, existing, '2023-06-15')


# =============================================================================
# Rapport de détection (return_report=True)
# =============================================================================
class TestDetectionReport:
    """``return_report=True`` returns a ``DelayDetectionReport`` next to the frame."""

    EXISTING = pd.DataFrame({'PIB': [1.0, 2.0, 3.0, np.nan], 'OLD': [1.0] * 4}, index=_months(4))
    NEW = pd.DataFrame({'PIB': [1.0, 2.5, 3.0, 4.0], 'NEW': [1.0] * 4}, index=_months(4))

    def test_default_returns_the_frame_only(self):
        result = compare_and_detect_delays(self.NEW, self.EXISTING, download_date='2023-06-15')
        assert isinstance(result, pd.DataFrame)

    def test_frame_is_the_same_with_and_without_report(self):
        plain = compare_and_detect_delays(self.NEW, self.EXISTING, '2023-06-15', detection_mode='all_changes')
        frame, _ = compare_and_detect_delays(
            self.NEW, self.EXISTING, '2023-06-15', detection_mode='all_changes', return_report=True)
        pd.testing.assert_frame_equal(plain, frame)

    def test_return_report_is_keyword_only(self):
        with pytest.raises(TypeError):
            compare_and_detect_delays(  # type: ignore[call-overload]
                self.NEW, self.EXISTING, '2023-06-15', 'new_only', 'start', 'day', None, None, True)

    def test_new_only_counts(self):
        _, report = compare_and_detect_delays(self.NEW, self.EXISTING, '2023-06-15', return_report=True)
        assert (report.n_detected, report.n_new_values, report.n_revisions) == (1, 1, 0)
        assert report.n_vanished_values == 0
        assert report.n_detected_by_column == {'PIB': 1}
        assert report.columns_without_detection == ()

    def test_all_changes_counts_the_revision_with_gold_delays(self):
        frame, report = compare_and_detect_delays(
            self.NEW, self.EXISTING, '2023-06-15', detection_mode='all_changes', return_report=True)
        # Avril publié (75 j depuis le 1er avril) et février révisé (134 j depuis le 1er février)
        assert sorted(frame['delay']) == [75.0, 134.0]
        assert (report.n_detected, report.n_new_values, report.n_revisions) == (2, 1, 1)
        assert (report.delay_min, report.delay_max, report.delay_mean, report.delay_median) == (75.0, 134.0, 104.5, 104.5)
        assert (report.n_known, report.n_negative) == (2, 0)
        assert report.delay_unit == 'day'

    def test_compared_and_ignored_columns(self):
        _, report = compare_and_detect_delays(self.NEW, self.EXISTING, '2023-06-15', return_report=True)
        assert report.columns_compared == ('PIB',)
        assert report.columns_new_only == ('NEW',)
        assert report.columns_existing_only == ('OLD',)
        assert (report.n_rows_new, report.n_rows_existing, report.n_columns, report.n_entities) == (4, 4, 2, 0)

    def test_vanished_values_are_counted_and_not_detected(self):
        existing = _series([1.0, 2.0, 3.0, 4.0])
        new = _series([1.0, 2.0, 3.0, np.nan])
        frame, report = compare_and_detect_delays(
            new, existing, '2023-06-15', detection_mode='all_changes', return_report=True)
        assert frame.empty
        assert report.n_vanished_values == 1
        assert report.n_detected == 0
        assert report.columns_without_detection == ('PIB',)
        assert (report.n_known, report.delay_min, report.delay_median) == (0, None, None)

    def test_without_existing_data_the_latest_observation_is_a_new_value(self):
        _, report = compare_and_detect_delays(self.NEW, download_date='2023-06-15', return_report=True)
        assert report.has_existing_data is False
        assert report.n_rows_existing is None
        assert (report.n_detected, report.n_new_values, report.n_revisions) == (2, 2, 0)

    def test_panel_counts_and_frequencies_per_entity(self):
        index = pd.MultiIndex.from_product([['FR', 'DE'], _months(4)], names=['country', 'date'])
        existing = pd.DataFrame({'PIB': [1.0, 2.0, 3.0, np.nan] * 2}, index=index)
        new = pd.DataFrame({'PIB': [1.0, 2.0, 3.0, 4.0] * 2}, index=index)
        _, report = compare_and_detect_delays(new, existing, '2023-06-15', return_report=True)
        assert report.n_entities == 2
        assert report.n_detected == 2
        assert report.frequencies == {('DE', 'PIB'): 'monthly', ('FR', 'PIB'): 'monthly'}
        assert report.undetected_keys == ()

    def test_undetectable_frequency_is_reported(self):
        new = _series([1.0])
        with pytest.warns(UserWarning, match="frequency could not be detected"):
            frame, report = compare_and_detect_delays(new, download_date='2023-06-15', return_report=True)
        assert report.undetected_keys == ('PIB',)
        assert report.frequencies == {'PIB': None}
        assert (report.n_detected, report.n_known) == (1, 0)

    def test_negative_delays_are_counted(self):
        _, report = compare_and_detect_delays(self.NEW, download_date='2023-01-10', return_report=True)
        assert report.n_negative == report.n_known

    def test_report_is_logged_at_info_level(self, caplog):
        with caplog.at_level('INFO', logger='tsforecast.delays.data_manager'):
            _, report = compare_and_detect_delays(self.NEW, self.EXISTING, '2023-06-15', return_report=True)
        assert report.summary() in caplog.messages

    def test_unit_label_follows_the_unit(self):
        _, report = compare_and_detect_delays(
            self.NEW, self.EXISTING, '2023-06-15', delay_unit='us', return_report=True)
        assert report.delay_unit == 'microsecond'


class TestDelayUnitResolution:
    """The delay units rest on ``utils.duration``; only day, second and microsecond are accepted."""

    @pytest.mark.parametrize('unit', ['h', 'hour', 'W', 'ms', 'days', 'D ', '', None, 3])
    def test_other_units_are_rejected_with_the_same_message(self, unit):
        with pytest.raises(ValueError, match="Unit must be one of"):
            compare_and_detect_delays(_series([1.0, 2.0, 3.0]), download_date='2023-06-15', delay_unit=unit)
