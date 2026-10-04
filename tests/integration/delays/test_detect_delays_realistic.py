"""Realistic scenarios for ``compare_and_detect_delays``.

Runs ``tsforecast.delays.data_manager.compare_and_detect_delays`` on the
realistic datasets of notebook 3 (``heterogeneous_coverage_panel`` and
``irregular_index_timeseries``: per-entity coverage, per-entity publication
frequency, genuinely irregular indexes) and chains several downloads.

The main scenario replays a **publication calendar**. The panel fixture is a
snapshot in which the notebook simulated publication delays by blanking the
last observation of ``inflation_ipc``, ``taux_chomage``, ``pib_trimestriel``,
``balance_commerciale_annuelle`` and ``depenses_publiques_pib`` of each entity.
Each blanked value is "published" ``k`` months after the start of its period
(1 month for inflation / unemployment, 2 for GDP, 3 for the trade balance, as
the time-series cell of the notebook states; the notebook gives none for
``depenses_publiques_pib``, 3 months is chosen here). Downloading the panel
at each release date and comparing it with the previous download must detect
exactly the values released that day, with a delay equal to the simulated one.

Gold values are derived by hand from the notebook construction, never copied
from the output of the code.
"""
# Modules de base
import pandas as pd
import pytest

# Perturbations partagées
from tests.support import perturbations as perturb

# Fonction à tester
from tsforecast.delays.data_manager import compare_and_detect_delays

# Le chemin « colonnes » émet systématiquement un avertissement de remplacement
# d'index (ANO-UTILS-033) : le bruit est masqué pour garder une sortie lisible.
pytestmark = pytest.mark.filterwarnings("ignore:Index replaced with")

TS = pd.Timestamp

# Fréquence de chaque colonne du panel hétérogène (``depenses_publiques_pib`` dépend de l'entité)
COLUMN_FREQUENCY = {
    'production_industrielle': 'monthly',
    'inflation_ipc': 'monthly',
    'taux_chomage': 'monthly',
    'climat_affaires': 'monthly',
    'pib_trimestriel': 'quarterly',
    'balance_commerciale_annuelle': 'annual',
}
DEPENSES_FREQUENCY = {'France': 'annual', 'Allemagne': 'quarterly', 'Italie': 'annual'}

# Longueur de chaque période, en mois
PERIOD_MONTHS = {'monthly': 1, 'quarterly': 3, 'annual': 12}

# Dates masquées par les délais simulés du notebook : dernière valeur de la grille propre à
# chaque entité (France et Italie jusqu'en 2024-07, Allemagne jusqu'en 2024-04)
#   inflation / chômage : dernier mois de la grille ;
#   PIB                 : dernier début de trimestre de la grille (janv. / avril / juil. / oct.) ;
#   dépenses publiques  : dernière publication (annuelle en janvier pour FR / IT, trimestrielle pour DE) ;
#   balance commerciale : dernier 1er janvier avant la fin de la grille (2024-01-01 pour tous).
HIDDEN_DATES = {
    ('France', 'inflation_ipc'): '2024-07-01',
    ('France', 'taux_chomage'): '2024-07-01',
    ('France', 'pib_trimestriel'): '2024-07-01',
    ('France', 'depenses_publiques_pib'): '2024-01-01',
    ('France', 'balance_commerciale_annuelle'): '2024-01-01',
    ('Allemagne', 'inflation_ipc'): '2024-04-01',
    ('Allemagne', 'taux_chomage'): '2024-04-01',
    ('Allemagne', 'pib_trimestriel'): '2024-04-01',
    ('Allemagne', 'depenses_publiques_pib'): '2024-04-01',
    ('Allemagne', 'balance_commerciale_annuelle'): '2024-01-01',
    ('Italie', 'inflation_ipc'): '2024-07-01',
    ('Italie', 'taux_chomage'): '2024-07-01',
    ('Italie', 'pib_trimestriel'): '2024-07-01',
    ('Italie', 'depenses_publiques_pib'): '2024-01-01',
    ('Italie', 'balance_commerciale_annuelle'): '2024-01-01',
}

# Délais simulés (en mois depuis le début de la période) — voir la docstring du module
SIMULATED_LAG_MONTHS = {
    'inflation_ipc': 1,
    'taux_chomage': 1,
    'pib_trimestriel': 2,
    'balance_commerciale_annuelle': 3,
    'depenses_publiques_pib': 3,
}

# Délais attendus en jours, calculés à la main pour chaque couple (entité, colonne)
# (début de période -> début de période + délai simulé), 2024 étant bissextile :
#   1 mois : 1er avril -> 1er mai = 30 j ; 1er juillet -> 1er août = 31 j
#   2 mois : 1er avril -> 1er juin = 30 + 31 = 61 j ; 1er juillet -> 1er sept. = 31 + 31 = 62 j
#   3 mois : 1er janv. -> 1er avril = 31 + 29 + 31 = 91 j ; 1er avril -> 1er juillet = 30 + 31 + 30 = 91 j
EXPECTED_SIMULATED_DELAY_DAYS = {
    ('France', 'inflation_ipc'): 31, ('France', 'taux_chomage'): 31, ('France', 'pib_trimestriel'): 62,
    ('France', 'depenses_publiques_pib'): 91, ('France', 'balance_commerciale_annuelle'): 91,
    ('Allemagne', 'inflation_ipc'): 30, ('Allemagne', 'taux_chomage'): 30, ('Allemagne', 'pib_trimestriel'): 61,
    ('Allemagne', 'depenses_publiques_pib'): 91, ('Allemagne', 'balance_commerciale_annuelle'): 91,
    ('Italie', 'inflation_ipc'): 31, ('Italie', 'taux_chomage'): 31, ('Italie', 'pib_trimestriel'): 62,
    ('Italie', 'depenses_publiques_pib'): 91, ('Italie', 'balance_commerciale_annuelle'): 91,
}


def _frequency_of(entity: str, column: str) -> str:
    """Return the expected publication frequency of an (entity, column) couple."""
    if column == 'depenses_publiques_pib':
        return DEPENSES_FREQUENCY[entity]
    return COLUMN_FREQUENCY[column]


def _release_date(entity: str, column: str) -> pd.Timestamp:
    """Return the simulated release date: period start of the blanked value + simulated lag."""
    return TS(HIDDEN_DATES[(entity, column)]) + pd.DateOffset(months=SIMULATED_LAG_MONTHS[column])


def _as_dict(result: pd.DataFrame) -> dict:
    """Index a panel result by (entity, column) and fail on duplicates (one row per couple expected)."""
    keys = list(result.index)
    assert len(keys) == len(set(keys)), "one row per (entity, column) expected"
    return {key: row for key, row in result.iterrows()}


# =============================================================================
# Premier téléchargement sur les jeux réalistes
# =============================================================================
class TestFirstDownloadOnRealisticDatasets:
    """``existing_data=None`` on the notebook 3 panel and time series."""

    def test_panel_reports_the_last_observation_of_each_couple(self, heterogeneous_coverage_panel):
        """One row per (entity, column) holding at least one observation, at its last non-null date."""
        panel = heterogeneous_coverage_panel
        expected = {}
        for entity in panel.index.get_level_values('country').unique():
            for column in panel.columns:
                last = panel.loc[entity, column].last_valid_index()
                if last is not None:
                    expected[(entity, column)] = last

        result = compare_and_detect_delays(panel, None, '2024-10-15')

        # 3 entités x 7 colonnes - (Italie, climat_affaires) structurellement absente = 20 couples
        assert len(expected) == 20
        assert {key: row['observation_date'] for key, row in _as_dict(result).items()} == expected

    def test_panel_frequency_is_detected_per_couple(self, heterogeneous_coverage_panel):
        """The same column gets the frequency of each entity (``depenses_publiques_pib``: annual / quarterly)."""
        result = compare_and_detect_delays(heterogeneous_coverage_panel, None, '2024-10-15')

        detected = {key: row['frequency'] for key, row in _as_dict(result).items()}
        assert detected == {key: _frequency_of(*key) for key in detected}

    def test_panel_delays_and_period_bounds(self, heterogeneous_coverage_panel):
        """Every observation is a period start: bounds follow the frequency and the delay counts days from it."""
        result = compare_and_detect_delays(heterogeneous_coverage_panel, None, '2024-10-15')

        for (entity, column), row in _as_dict(result).items():
            months = PERIOD_MONTHS[_frequency_of(entity, column)]
            observation = row['observation_date']
            assert row['period_start'] == observation, (entity, column)
            assert row['period_end'] == observation + pd.DateOffset(months=months), (entity, column)
            # Valeur d'or : 15 octobre 2024 à minuit - début de période, en jours
            assert row['delay'] == (TS('2024-10-15') - observation).days, (entity, column)

    def test_panel_end_reference_subtracts_the_period_length(self, heterogeneous_coverage_panel):
        """With ``reference_point='end'`` each delay is the ``'start'`` one minus the period length."""
        start = compare_and_detect_delays(heterogeneous_coverage_panel, None, '2024-10-15', reference_point='start')
        end = compare_and_detect_delays(heterogeneous_coverage_panel, None, '2024-10-15', reference_point='end')

        gap = (start['delay'] - end['delay']).to_dict()
        # Valeur d'or : l'écart vaut la durée en jours de la période de l'observation
        expected = (end['period_end'] - end['period_start']).dt.days.astype(float).to_dict()
        assert gap == expected

    def test_irregular_time_series_frequencies_and_last_dates(self, irregular_index_timeseries):
        """Isolated annual anchors before the monthly grid do not disturb per-column detection."""
        result = compare_and_detect_delays(irregular_index_timeseries, None, '2024-10-15')

        # Valeurs d'or : notebook 3, dernière valeur de chaque colonne
        #   production : grille mensuelle jusqu'en 2024-07, non masquée ;
        #   inflation / chômage : 2024-07 masqué -> 2024-06 ;
        #   PIB : 2024-07 masqué -> 2024-04 ; balance annuelle : 2024-01 masqué -> 2023-01
        assert {key: (row['frequency'], row['observation_date']) for key, row in _as_dict_series(result).items()} == {
            'production_industrielle': ('monthly', TS('2024-07-01')),
            'inflation_ipc': ('monthly', TS('2024-06-01')),
            'taux_chomage': ('monthly', TS('2024-06-01')),
            'pib_trimestriel': ('quarterly', TS('2024-04-01')),
            'balance_commerciale_annuelle': ('annual', TS('2023-01-01')),
        }


def _as_dict_series(result: pd.DataFrame) -> dict:
    """Index a time-series result by column name."""
    return {key: row for key, row in result.iterrows()}


# =============================================================================
# Rejeu d'un calendrier de publication
# =============================================================================
class TestPublicationCalendarReplay:
    """Successive downloads of the panel, each one releasing the values due that day."""

    @staticmethod
    def _replay(panel: pd.DataFrame, **kwargs) -> list:
        """Replay the release calendar and return ``(release_date, due_couples, result)`` per step.

        Args:
            panel: Snapshot with the simulated delays applied (values blanked).
            **kwargs: Extra arguments forwarded to ``compare_and_detect_delays``.

        Returns:
            One tuple per distinct release date, in chronological order.
        """
        release = {couple: _release_date(*couple) for couple in HIDDEN_DATES}
        snapshot = panel.copy()
        steps = []
        for when in sorted(set(release.values())):
            newer = snapshot.copy()
            due = sorted(couple for couple, date in release.items() if date == when)
            for entity, column in due:
                # Valeur publiée = reconduction de la dernière valeur connue : seule
                # l'apparition (NaN -> valeur) compte pour la détection
                known = snapshot.loc[entity, column].dropna().iloc[-1]
                newer.loc[(entity, TS(HIDDEN_DATES[(entity, column)])), column] = known
            result = compare_and_detect_delays(newer, snapshot, download_date=when, reference_point='start', **kwargs)
            steps.append((when, due, result))
            snapshot = newer
        return steps

    def test_blanked_values_are_where_the_calendar_says(self, heterogeneous_coverage_panel):
        """The hand-written table matches the fixture: each date is blank and is the entity's last publication."""
        panel = heterogeneous_coverage_panel
        for (entity, column), hidden in HIDDEN_DATES.items():
            series = panel.loc[entity, column]
            step = PERIOD_MONTHS[_frequency_of(entity, column)]
            assert pd.isna(series.loc[hidden]), (entity, column)
            assert series.loc[:hidden].dropna().index[-1] == TS(hidden) - pd.DateOffset(months=step), (entity, column)
            assert series.loc[hidden:].isna().all(), (entity, column)

    def test_calendar_has_six_release_dates(self, heterogeneous_coverage_panel):
        """The 15 blanked values are released on 6 distinct dates (hand count)."""
        # Valeur d'or : 2024-04-01 (balances + dépenses FR / IT), 05-01 (DE inflation / chômage),
        # 06-01 (DE PIB), 07-01 (DE dépenses), 08-01 (FR / IT inflation / chômage), 09-01 (FR / IT PIB)
        steps = self._replay(heterogeneous_coverage_panel)

        assert [when for when, _, _ in steps] == [
            TS('2024-04-01'), TS('2024-05-01'), TS('2024-06-01'), TS('2024-07-01'), TS('2024-08-01'), TS('2024-09-01'),
        ]

    def test_each_download_detects_exactly_the_values_released_that_day(self, heterogeneous_coverage_panel):
        """The detected (entity, column) couples of a step are the ones released at that date, no more."""
        for when, due, result in self._replay(heterogeneous_coverage_panel):
            assert sorted(result.index) == due, when

    def test_detected_observation_dates_are_the_blanked_ones(self, heterogeneous_coverage_panel):
        """Each detected observation is the blanked value of its couple."""
        for when, _, result in self._replay(heterogeneous_coverage_panel):
            detected = {key: row['observation_date'] for key, row in _as_dict(result).items()}
            assert detected == {key: TS(HIDDEN_DATES[key]) for key in detected}, when

    def test_detected_delays_equal_the_simulated_ones(self, heterogeneous_coverage_panel):
        """The detected delay of each couple is the simulated lag, converted to days by the calendar."""
        for when, _, result in self._replay(heterogeneous_coverage_panel):
            detected = {key: row['delay'] for key, row in _as_dict(result).items()}
            assert detected == {key: float(EXPECTED_SIMULATED_DELAY_DAYS[key]) for key in detected}, when

    def test_detected_delays_are_consistent_with_the_simulated_lag_in_months(self, heterogeneous_coverage_panel):
        """Whatever the calendar, ``k`` months last between 28 k and 31 k days: delays and lags agree."""
        for _, _, result in self._replay(heterogeneous_coverage_panel):
            for (_, column), row in _as_dict(result).items():
                months = SIMULATED_LAG_MONTHS[column]
                assert 28 * months <= row['delay'] <= 31 * months, column

    def test_detected_frequencies_are_per_couple(self, heterogeneous_coverage_panel):
        """Frequencies come out per couple, whichever entity releases ``depenses_publiques_pib``."""
        for when, _, result in self._replay(heterogeneous_coverage_panel):
            detected = {key: row['frequency'] for key, row in _as_dict(result).items()}
            assert detected == {key: _frequency_of(*key) for key in detected}, when

    def test_same_replay_with_all_changes_gives_the_same_detection(self, heterogeneous_coverage_panel):
        """Nothing but appearances happens in the calendar: ``all_changes`` agrees with ``new_only``."""
        new_only = self._replay(heterogeneous_coverage_panel, detection_mode='new_only')
        all_changes = self._replay(heterogeneous_coverage_panel, detection_mode='all_changes')

        for (_, _, expected), (_, _, result) in zip(new_only, all_changes):
            pd.testing.assert_frame_equal(result, expected)

    def test_single_download_after_the_whole_calendar(self, heterogeneous_coverage_panel):
        """A single comparison against the initial snapshot finds the 15 releases, delays counted to that date."""
        snapshot = heterogeneous_coverage_panel
        final = snapshot.copy()
        for (entity, column), hidden in HIDDEN_DATES.items():
            final.loc[(entity, TS(hidden)), column] = snapshot.loc[entity, column].dropna().iloc[-1]

        result = compare_and_detect_delays(final, snapshot, download_date='2024-10-15')

        # Valeurs d'or : 15 oct. 2024 - 1er juil. = 106 j ; - 1er avril = 197 j ; - 1er janv. = 288 j
        expected_days = {'2024-07-01': 106, '2024-04-01': 197, '2024-01-01': 288}
        assert {key: row['delay'] for key, row in _as_dict(result).items()} == {
            key: float(expected_days[hidden]) for key, hidden in HIDDEN_DATES.items()
        }

    def test_nothing_new_after_the_last_release(self, heterogeneous_coverage_panel):
        """Re-downloading the final state detects nothing."""
        final = heterogeneous_coverage_panel.copy()
        for (entity, column), hidden in HIDDEN_DATES.items():
            final.loc[(entity, TS(hidden)), column] = 1.0

        result = compare_and_detect_delays(final, final.copy(), download_date='2024-10-15', detection_mode='all_changes')

        assert result.empty


# =============================================================================
# Robustesse du scénario aux présentations des données
# =============================================================================
class TestPerturbedRealisticPanel:
    """The realistic first download is invariant to the way the panel is presented."""

    @pytest.fixture
    def reference(self, heterogeneous_coverage_panel) -> pd.DataFrame:
        """First download of the unperturbed panel, sorted for comparison."""
        return compare_and_detect_delays(heterogeneous_coverage_panel, None, '2024-10-15').sort_index()

    def test_shuffled_rows(self, heterogeneous_coverage_panel, reference):
        """Rows in random order give the same result."""
        result = compare_and_detect_delays(perturb.shuffle_rows(heterogeneous_coverage_panel, seed=5), None, '2024-10-15')

        pd.testing.assert_frame_equal(result.sort_index(), reference)

    def test_entities_in_reverse_order(self, heterogeneous_coverage_panel, reference):
        """Entity blocks in reverse order give the same result."""
        result = compare_and_detect_delays(perturb.reverse_entities(heterogeneous_coverage_panel), None, '2024-10-15')

        pd.testing.assert_frame_equal(result.sort_index(), reference)

    def test_columns_instead_of_index(self, heterogeneous_coverage_panel, reference):
        """Entities and dates held in columns give the same result."""
        result = compare_and_detect_delays(
            heterogeneous_coverage_panel.reset_index(), None, '2024-10-15', time_col='date', panel_cols=['country'],
        )

        pd.testing.assert_frame_equal(result.sort_index(), reference)

    def test_three_level_index(self, heterogeneous_coverage_panel, reference):
        """An extra outer level changes the index of the result, not the detected values."""
        result = compare_and_detect_delays(perturb.to_three_level_index(heterogeneous_coverage_panel), None, '2024-10-15')

        flat = result.droplevel('region').sort_index()
        pd.testing.assert_frame_equal(flat, reference)

    def test_special_column_names(self, heterogeneous_coverage_panel, reference):
        """Renamed columns carry the same rows under their new names."""
        renamed, mapping = perturb.with_special_column_names(heterogeneous_coverage_panel)

        result = compare_and_detect_delays(renamed, None, '2024-10-15')

        restored = result.rename(index={new: old for old, new in mapping.items()}, level='column').sort_index()
        pd.testing.assert_frame_equal(restored, reference)

    def test_month_end_labels(self, heterogeneous_coverage_panel, reference):
        """End-of-period labels designate the same periods: same bounds, frequencies and delays."""
        end_labelled = perturb.to_period_end(heterogeneous_coverage_panel)

        result = compare_and_detect_delays(end_labelled, None, '2024-10-15').sort_index()

        # Garde-fou : la conversion a bien eu lieu (une conversion silencieusement sans effet
        # ferait réussir ce test pour la mauvaise raison) — chaque étiquette est une fin de mois ;
        # une valeur trimestrielle ou annuelle y est étiquetée à la fin de son premier mois
        assert result['observation_date'].dt.is_month_end.all()
        columns = ['frequency', 'period_start', 'period_end', 'delay']
        pd.testing.assert_frame_equal(result[columns], reference[columns])
