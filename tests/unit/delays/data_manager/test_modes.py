"""Output contract and detection modes of ``compare_and_detect_delays``.

Scenarios: shape and types of the returned frame, first download, what counts as a new
observation under ``new_only`` and ``all_changes``, and the documented quarterly
example.
"""

import numpy as np
import pandas as pd
import pytest

# Jeux de données et perturbations partagés
from tests.support import perturbations as perturb

# Fonction à tester
from tsforecast.delays.data_manager import compare_and_detect_delays

# Outils de test partagés du paquet
from tests.unit.delays.data_manager.helpers import (
    OUTPUT_COLUMNS,
    TS,
    _expected,
    _months,
    _panel,
    _records,
    _series,
)


# Le chemin « colonnes » (time_col / panel_cols) émet systématiquement un
# avertissement de remplacement d'index (comportement de validate_temporal_data) :
# le bruit est masqué pour garder une sortie lisible.
pytestmark = pytest.mark.filterwarnings("ignore:Index replaced with")


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
# Exemple de la documentation (tutoriel sur les délais de publication)
# =============================================================================
class TestDocumentedExample:
    """The quarterly GDP example of ``docs/tutorials/publication_delays.md``."""

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
