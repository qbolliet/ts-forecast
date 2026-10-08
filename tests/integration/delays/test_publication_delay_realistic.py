"""Realistic scenario for ``PublicationDelayTransformer`` and its per-entity factory.

Chains the delay detection and the delay application on the notebook 3 panel
(``heterogeneous_coverage_panel``: per-entity coverage, ``depenses_publiques_pib``
annual for France / Italie and quarterly for Allemagne, ``climat_affaires``
absent for Italie), as a user preparing a forecast would:

1. the publication calendar of ``test_detect_delays_realistic.py`` is replayed
   (each blanked value released ``k`` months after the start of its period) and
   ``compare_and_detect_delays`` measures the delay of every released
   (entity, column) couple;
2. ``calculate_applicable_delay(aggregate_by_panel=True)`` keeps one delay per
   couple, in days from the start of the period of its own frequency;
3. ``create_delay_transformer_factory`` + ``PanelwiseTransformer`` apply them to
   the fully released panel at the prediction date 2024-10-15.

Property checked (no cell-level gold value): for every couple, the last
available value after ``transform`` is the last value of the data, moved by the
number of periods ``k`` that separates the period of the prediction date from
the last period already published at that date. ``k`` is derived from the
calendar only: the smallest ``k`` such that the period starting ``k`` periods
before the period of the prediction date, plus its delay, is not after the
prediction date.
"""
# Modules de base
import warnings

import pandas as pd
import pytest

# Composants enchaînés
from tsforecast.delays.calculator import calculate_applicable_delay
from tsforecast.delays.data_manager import compare_and_detect_delays
from tsforecast.delays.transformers import create_delay_transformer_factory
from tsforecast.panel import PanelwiseTransformer

# Calendrier de publication simulé, établi (et vérifié contre la fixture) par les tests de détection
from tests.integration.delays.test_detect_delays_realistic import HIDDEN_DATES, SIMULATED_LAG_MONTHS

# Le chemin « colonnes » émet systématiquement un avertissement de remplacement d'index (ANO-UTILS-033)
pytestmark = pytest.mark.filterwarnings("ignore:Index replaced with")

TS = pd.Timestamp
PREDICTION = TS('2024-10-15')

# Longueur d'une période en mois, par fréquence détectée
PERIOD_MONTHS = {'monthly': 1, 'quarterly': 3, 'annual': 12}


def _release(panel: pd.DataFrame, couples) -> pd.DataFrame:
    """Return a copy of ``panel`` where the blanked value of each couple is published.

    The published value repeats the last known value: only the appearance (NaN -> value) matters.
    """
    released = panel.copy()
    for entity, column in couples:
        known = panel.loc[entity, column].dropna().iloc[-1]
        released.loc[(entity, TS(HIDDEN_DATES[(entity, column)])), column] = known
    return released


@pytest.fixture(scope='module')
def realistic_case(_heterogeneous_coverage_panel_session):
    """Replay the calendar and return ``(final_panel, applicable_delays)``.

    Returns:
        The panel with every value released, and the delays table indexed by
        (country, column): one row per released couple, delays in days from the
        start of the period of the couple's own frequency.
    """
    snapshot = _heterogeneous_coverage_panel_session.copy()
    release_dates = {couple: TS(HIDDEN_DATES[couple]) + pd.DateOffset(months=SIMULATED_LAG_MONTHS[couple[1]])
                     for couple in HIDDEN_DATES}
    detected = []
    for when in sorted(set(release_dates.values())):
        due = [couple for couple, date in release_dates.items() if date == when]
        newer = _release(snapshot, due)
        detected.append(compare_and_detect_delays(newer, snapshot, download_date=when, reference_point='start'))
        snapshot = newer
    publication_delays = pd.concat(detected)
    # Fréquence cible de chaque couple : sa propre fréquence (le délai détecté est alors inchangé)
    frequency = dict(zip(publication_delays.index, publication_delays['frequency']))
    applicable = calculate_applicable_delay(publication_delays, 'start', frequency, aggregate_by_panel=True, unit='D')
    return snapshot, applicable


def _calendar_shift(delay_days: float, months: int) -> int:
    """Return the number of periods between the prediction period and the last period published at the prediction date.

    Args:
        delay_days: Delay in days from the period start.
        months: Length of a period in months.

    Returns:
        ``k >= 0``.
    """
    # Début de la période (mois, trimestre ou année) contenant la date de prédiction
    month_index = (PREDICTION.month - 1) // months * months
    period_start = TS(year=PREDICTION.year, month=month_index + 1, day=1)
    k = 0
    while period_start - pd.DateOffset(months=k * months) + pd.Timedelta(days=delay_days) > PREDICTION:
        k += 1
    return k


@pytest.fixture(scope='module')
def transformed(realistic_case):
    """Apply the applicable delays to the fully released panel through the factory."""
    panel, applicable = realistic_case
    factory = create_delay_transformer_factory(applicable, strategy='shift', prediction_date=PREDICTION)
    transformer = PanelwiseTransformer(transformer=factory, time_col=None, panel_cols=None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return transformer.fit_transform(panel), transformer


class TestDelaysFromDetection:
    """Contract between detection, conversion and application on the realistic panel."""

    def test_one_delay_per_released_couple(self, realistic_case):
        """The 15 released couples get exactly one applicable delay each."""
        _, applicable = realistic_case
        assert sorted(applicable.index) == sorted(HIDDEN_DATES)

    def test_each_entity_transformer_gets_its_own_delays(self, transformed, realistic_case):
        """Every entity is fitted with the delays of its own rows (Allemagne: 30-day monthly delays)."""
        _, transformer = transformed
        _, applicable = realistic_case
        for entity, fitted in transformer.transformers_.items():
            expected = applicable.xs(entity[0], level='country')['delay'].to_dict()
            assert fitted.delays == expected, entity


class TestLastAvailableValue:
    """After ``transform``, the last value of each couple sits where its delay puts it."""

    @pytest.mark.parametrize('couple', sorted(HIDDEN_DATES), ids=lambda c: f'{c[0]}-{c[1]}')
    def test_last_value_moved_by_the_calendar_shift(self, couple, transformed, realistic_case):
        """Last date = last date of the data + ``k`` periods, ``k`` derived from the calendar; same value."""
        panel, applicable = realistic_case
        result, _ = transformed
        entity, column = couple
        months = _frequency_months(applicable, couple)
        k = _calendar_shift(applicable.loc[couple, 'delay'], months)
        original = panel.loc[entity, column].dropna()
        shifted = result.loc[entity, column].dropna()
        assert (shifted.index[-1], shifted.iloc[-1]) == (
            original.index[-1] + pd.DateOffset(months=k * months), original.iloc[-1])

    def test_shift_parameters_match_the_calendar(self, transformed, realistic_case):
        """The fitted ``n_periods`` of every couple is ``-k``."""
        _, applicable = realistic_case
        _, transformer = transformed
        fitted = {(entity[0], column): params['n_periods']
                  for entity, t in transformer.transformers_.items() for column, params in t.shift_params.items()}
        expected = {couple: -_calendar_shift(applicable.loc[couple, 'delay'], _frequency_months(applicable, couple))
                    for couple in applicable.index}
        assert fitted == expected

    def test_columns_without_delay_are_untouched(self, transformed, realistic_case):
        """Columns never released late (industrial production, business climate) keep their values."""
        panel, _ = realistic_case
        result, _ = transformed
        for column in ('production_industrielle', 'climat_affaires'):
            pd.testing.assert_series_equal(result[column].reindex(panel.index), panel[column])

    def test_round_trip(self, transformed, realistic_case):
        """``inverse_transform`` restores the panel exactly (dates added by the shift dropped)."""
        panel, _ = realistic_case
        result, transformer = transformed
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            recovered = transformer.inverse_transform(result)
        pd.testing.assert_frame_equal(recovered, panel)


def _frequency_months(applicable: pd.DataFrame, couple) -> int:
    """Return the period length, in months, of the target frequency of a couple."""
    return PERIOD_MONTHS[applicable.loc[couple, 'frequency']]
