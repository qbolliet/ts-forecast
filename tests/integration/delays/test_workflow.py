"""End-to-end workflows of the ``delays`` module.

Chains the public components of ``tsforecast.delays`` as a user preparing a
forecast would:

- ``compare_and_detect_delays`` (first download, ``existing_data=None``, or
  comparison of two downloads) -> ``calculate_applicable_delay`` ->
  ``PublicationDelayTransformer`` (directly, or per entity through
  ``PanelwiseTransformer``), on small time series / panels with gold values
  computed by hand from the calendar;
- the same chain on the realistic panel of notebook 3
  (``heterogeneous_coverage_panel``): two downloads, delays measured per
  (entity, column), one delay per column applied to the whole panel, and a
  check per (entity, column) of what is visible at the prediction date;
- ``ShiftTransformer`` then ``MaskTransformer``, sequentially and in an sklearn
  ``Pipeline``.

The tests migrated from ``tests/integration/delays/test_integration.py``
(prompt D5) were rewritten on the current API: ``release_delay`` -> ``delay``
(``4b3d3bc``), ``target_reference_point`` / ``target_frequency`` /
``applicable_delay`` -> ``reference_point`` / ``frequency`` / ``delay``
(``ae347b2``), and the variable read from the index level ``'column'`` of the
detected delays (an index level since ``0dd824f``, ``ANO-DELAYS-009``).

Anomalies exposed here, now fixed: ``ANO-DELAYS-042`` (periods of 30 / 91 / 365
days in the conversion of a delay into a number of periods: a value published on
the prediction date was pushed one month too far; now counted on the calendar)
and ``ANO-DELAYS-043`` (panel output of ``PublicationDelayTransformer`` not ordered
by date within each entity).
"""
# Modules de base
import itertools
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

# Composants enchaînés
from tsforecast.delays.calculator import calculate_applicable_delay
from tsforecast.delays.data_manager import compare_and_detect_delays
from tsforecast.delays.transformers import MaskTransformer, PublicationDelayTransformer, ShiftTransformer
from tsforecast.panel import PanelwiseTransformer

# Dates masquées par les délais simulés du notebook 3, établies (et vérifiées contre la fixture)
# par les tests de détection réalistes
from tests.integration.delays.test_detect_delays_realistic import HIDDEN_DATES

# Le chemin « colonnes » émet systématiquement un avertissement de remplacement d'index (ANO-UTILS-033)
pytestmark = pytest.mark.filterwarnings("ignore:Index replaced with")

TS = pd.Timestamp

# Date de téléchargement des petits jeux : 14 jours après la fin (exclusive) de juin 2024
DOWNLOAD_DATE = datetime(2024, 7, 15)
COUNTRIES = ['France', 'Germany', 'Italy']


# =============================================================================
# Jeux de données (déterministes)
# =============================================================================
@pytest.fixture
def time_series_data() -> pd.DataFrame:
    """Complete monthly (``MS``) series, January 2023 to June 2024, three variables."""
    rng = np.random.default_rng(42)
    dates = pd.date_range('2023-01-01', '2024-06-01', freq='MS')
    return pd.DataFrame({
        'GDP': 100 + np.cumsum(rng.normal(0, 0.5, len(dates))),
        'inflation': 2.5 + rng.normal(0, 0.3, len(dates)),
        'unemployment': 8.0 + rng.normal(0, 0.5, len(dates)),
    }, index=dates)


# Dernière observation de chaque couple (entité, colonne) du panel ci-dessous
PANEL_LAST_OBSERVATION = {
    ('France', 'GDP'): TS('2024-06-01'),
    ('France', 'inflation'): TS('2024-06-01'),
    ('Germany', 'GDP'): TS('2024-05-01'),
    ('Germany', 'inflation'): TS('2024-05-01'),
    ('Italy', 'GDP'): TS('2024-05-01'),
    ('Italy', 'inflation'): TS('2024-06-01'),
}


@pytest.fixture
def panel_data() -> pd.DataFrame:
    """Monthly panel (country, date), January 2023 to June 2024, with per-couple last observations.

    Germany stops in May 2024 and the Italian GDP of June 2024 is missing, so
    that the last observation is a property of the (entity, column) couple
    (``PANEL_LAST_OBSERVATION``), not of the panel.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range('2023-01-01', '2024-06-01', freq='MS', name='date')
    frames = {
        country: pd.DataFrame({'GDP': 100 + rng.normal(0, 2, len(dates)),
                               'inflation': 2.5 + rng.normal(0, 0.5, len(dates))}, index=dates)
        for country in COUNTRIES
    }
    # Couverture propre à chaque couple : Allemagne arrêtée en mai, PIB italien de juin manquant
    frames['Germany'] = frames['Germany'].iloc[:-1]
    frames['Italy'].loc[TS('2024-06-01'), 'GDP'] = np.nan
    return pd.concat(frames, names=['country', 'date'])


@pytest.fixture
def multi_frequency_data() -> pd.DataFrame:
    """Monthly indicator and quarterly indicator (``QS``) in one frame, January 2023 to June 2024."""
    rng = np.random.default_rng(42)
    monthly_dates = pd.date_range('2023-01-01', '2024-06-01', freq='MS')
    quarterly_dates = pd.date_range('2023-01-01', '2024-04-01', freq='QS')
    monthly = pd.Series(100 + rng.normal(0, 10, len(monthly_dates)), index=monthly_dates, name='monthly_indicator')
    quarterly = pd.Series(200 + rng.normal(0, 20, len(quarterly_dates)), index=quarterly_dates,
                          name='quarterly_indicator')
    # Indicateur trimestriel stocké sur la grille mensuelle (NaN hors débuts de trimestre)
    return pd.concat([monthly, quarterly], axis=1)


@pytest.fixture
def daily_time_series() -> pd.DataFrame:
    """Complete daily series, January 1 to June 30, 2024, two indicators."""
    rng = np.random.default_rng(42)
    dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
    return pd.DataFrame({'indicator_A': 100 + rng.normal(0, 10, len(dates)),
                         'indicator_B': 50 + rng.normal(0, 5, len(dates))}, index=dates)


def _silently(function, *args, **kwargs):
    """Call ``function`` with the warnings silenced (expected warnings are tested elsewhere)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


# =============================================================================
# Premier téléchargement d'une série : détection -> délai applicable -> décalage
# =============================================================================
class TestTimeSeriesWorkflow:
    """First download of a monthly series, delays counted from the period end."""

    @pytest.fixture
    def detected(self, time_series_data):
        """Delays of the last observation of each column, downloaded on 2024-07-15."""
        return compare_and_detect_delays(new_data=time_series_data, existing_data=None,
                                         download_date=DOWNLOAD_DATE, reference_point='end', delay_unit='D')

    def test_first_download_reports_the_last_observation_of_each_column(self, detected, time_series_data):
        """Without ``existing_data``, the observation of each column is its last non-null date.

        Rewritten from ``test_last_observation_aligned_with_prediction_date``: the
        column name is the index level ``'column'``, not a column of the result.
        """
        assert detected['observation_date'].to_dict() == {col: TS('2024-06-01') for col in time_series_data}

    def test_delay_counted_from_the_end_of_june(self, detected):
        """June ends on July 1 (exclusive bound): downloaded on July 15, the delay is 14 days."""
        assert detected['delay'].to_dict() == {'GDP': 14.0, 'inflation': 14.0, 'unemployment': 14.0}

    def test_applicable_delays_shift_june_into_july(self, detected, time_series_data):
        """End to end: every value moves one month later, June being published by July 15.

        ``calculate_applicable_delay`` keeps 14 days from the end of the month;
        ``PublicationDelayTransformer`` reads the variable, unit and reference point
        from its output as is. Gold value: June ends on July 1, + 14 days = July 15,
        published on the prediction date: June is the last available month, placed
        in July (one period).
        """
        applicable = calculate_applicable_delay(detected, reference_point='end', frequency='M',
                                                aggregation_method='median')
        transformed = _silently(PublicationDelayTransformer(delays=applicable, strategy='shift',
                                                            prediction_date=DOWNLOAD_DATE).fit_transform,
                                time_series_data)
        expected = time_series_data.set_axis(time_series_data.index + pd.DateOffset(months=1))
        pd.testing.assert_frame_equal(transformed, expected, check_freq=False)


# =============================================================================
# Premier téléchargement d'un panel : délais par couple (entité, colonne)
# =============================================================================
class TestPanelWorkflow:
    """First download of a panel whose last observation depends on the (entity, column) couple."""

    @pytest.fixture
    def detected(self, panel_data):
        """Delays of the last observation of each couple, downloaded on 2024-07-15, from the period end."""
        return compare_and_detect_delays(new_data=panel_data, existing_data=None,
                                         download_date=DOWNLOAD_DATE, reference_point='end', delay_unit='D')

    def test_each_entity_and_column_has_a_delay(self, detected):
        """One row per (entity, column) couple, indexed by (country, column)."""
        expected = list(itertools.product(COUNTRIES, ['GDP', 'inflation']))
        assert sorted(detected.index) == expected

    def test_last_observation_per_entity_and_column(self, detected):
        """The observation of each couple is its own last non-null date (Germany in May, Italian GDP in May)."""
        assert detected['observation_date'].to_dict() == PANEL_LAST_OBSERVATION

    def test_delay_per_entity_and_column(self, detected):
        """14 days after the end of June, 44 days after the end of May (June 1 + 44 days = July 15)."""
        expected = {couple: 14.0 if date == TS('2024-06-01') else 44.0
                    for couple, date in PANEL_LAST_OBSERVATION.items()}
        assert detected['delay'].to_dict() == expected

    def test_freshest_value_of_each_couple_lands_in_the_prediction_month(self, detected, panel_data):
        """End to end with one transformer per entity: every last value is placed in July 2024.

        ``calculate_applicable_delay(aggregate_by_panel=True)`` keeps the delay of
        each couple; a factory gives each entity its own delays. Gold values: a
        14-day delay from the end shifts one month (June -> July), a 44-day delay
        two months (May ends on June 1, + 44 days = July 15; May -> July): on the prediction
        date, the freshest published value of every couple is the one of its last
        observed month, placed in the month of the prediction date.
        """
        applicable = calculate_applicable_delay(detected, reference_point='end', frequency='M',
                                                aggregate_by_panel=True, aggregation_method='median')

        def transformer_factory(entity_key):
            # Délais de l'entité, indexés par variable
            entity_delays = applicable.xs(entity_key[0], level='country')['delay'].to_dict()
            return PublicationDelayTransformer(delays=entity_delays, strategy='shift', prediction_date=DOWNLOAD_DATE,
                                               delay_unit='D', reference_point='end')

        transformed = _silently(PanelwiseTransformer(transformer=transformer_factory, time_col=None,
                                                     panel_cols=None).fit_transform, panel_data)
        in_july = {(entity, column): transformed.loc[(entity, TS('2024-07-01')), column]
                   for entity, column in PANEL_LAST_OBSERVATION}
        last_values = {(entity, column): panel_data.loc[(entity, date), column]
                       for (entity, column), date in PANEL_LAST_OBSERVATION.items()}
        assert in_july == last_values


# =============================================================================
# Plusieurs fréquences dans un même jeu
# =============================================================================
class TestMultiFrequencyWorkflow:
    """Monthly and quarterly indicators of one frame, downloaded the same day."""

    @pytest.fixture
    def detected(self, multi_frequency_data):
        """Delays from the period start, downloaded on 2024-07-15."""
        return compare_and_detect_delays(new_data=multi_frequency_data, existing_data=None,
                                         download_date=DOWNLOAD_DATE, reference_point='start', delay_unit='D')

    def test_frequency_detected_per_column(self, detected):
        """Each column gets its own frequency, the quarterly one despite the NaN of the monthly grid."""
        assert detected['frequency'].to_dict() == {'monthly_indicator': 'monthly', 'quarterly_indicator': 'quarterly'}

    def test_quarterly_delay_from_the_period_start_is_longer(self, detected):
        """Each indicator in its own frequency: 44 days for June, 105 days for the second quarter.

        Gold values: June 1 -> July 15 = 30 + 14 = 44 days; April 1 -> July 15 =
        30 + 31 + 30 + 14 = 105 days.
        """
        applicable = calculate_applicable_delay(
            detected, reference_point='start',
            frequency={'monthly_indicator': 'M', 'quarterly_indicator': 'Q'}, aggregation_method='median')
        assert applicable['delay'].to_dict() == {'monthly_indicator': 44.0, 'quarterly_indicator': 105.0}


# =============================================================================
# Combinaison décalage puis masque
# =============================================================================
class TestCombinedTransformations:
    """``ShiftTransformer`` followed by ``MaskTransformer``."""

    def test_shift_then_mask_round_trip(self, daily_time_series):
        """Unmasking then unshifting restores the original series exactly."""
        shifter = ShiftTransformer(n_periods=5, frequency='D')
        masker = MaskTransformer(n_obs=3, mask_frequency='M', how='last')
        masked = masker.fit_transform(shifter.fit_transform(daily_time_series))
        unshifted = shifter.inverse_transform(masker.inverse_transform(masked))
        pd.testing.assert_frame_equal(unshifted, daily_time_series)

    def test_sklearn_pipeline_equals_the_sequential_application(self, daily_time_series):
        """In an sklearn ``Pipeline``, shift then mask gives the same frame as applying them in turn.

        Replaces ``test_pipeline_with_transformers``, which only checked the
        presence of the methods (covered by the sklearn protocol tests of D3).
        """
        sequential = MaskTransformer(n_obs=3, mask_frequency='M', how='last').fit_transform(
            ShiftTransformer(n_periods=5, frequency='D').fit_transform(daily_time_series))
        pipeline = Pipeline([('shift', ShiftTransformer(n_periods=5, frequency='D')),
                             ('mask', MaskTransformer(n_obs=3, mask_frequency='M', how='last'))])
        pd.testing.assert_frame_equal(pipeline.fit_transform(daily_time_series), sequential)

    def test_sklearn_pipeline_inverse_transform_restores_the_data(self, daily_time_series):
        """``Pipeline.inverse_transform`` unmasks then unshifts, restoring the input."""
        pipeline = Pipeline([('shift', ShiftTransformer(n_periods=5, frequency='D')),
                             ('mask', MaskTransformer(n_obs=3, mask_frequency='M', how='last'))])
        recovered = pipeline.inverse_transform(pipeline.fit_transform(daily_time_series))
        pd.testing.assert_frame_equal(recovered, daily_time_series)


# =============================================================================
# Scénario réaliste : deux téléchargements du panel hétérogène du notebook 3
# =============================================================================
# Second téléchargement : toutes les valeurs retenues par les délais simulés sont publiées
SECOND_DOWNLOAD = TS('2024-09-15')

# Délai détecté (jours depuis le début de la période) selon la date de la valeur publiée,
# toutes les valeurs masquées étant des débuts de période (mois, trimestre ou année) :
#   2024-07-01 -> 2024-09-15 : 31 + 31 + 14 = 76 jours ;
#   2024-04-01 -> 2024-09-15 : 30 + 31 + 30 + 31 + 31 + 14 = 167 jours ;
#   2024-01-01 -> 2024-09-15 : 244 (2024 bissextile) + 14 = 258 jours.
DETECTED_DELAY_BY_DATE = {'2024-07-01': 76.0, '2024-04-01': 167.0, '2024-01-01': 258.0}

# Colonnes retardées dont la fréquence est commune aux entités (``depenses_publiques_pib`` est
# annuelle pour la France et l'Italie, trimestrielle pour l'Allemagne), avec cette fréquence
SHARED_FREQUENCY = {
    'inflation_ipc': 'monthly',
    'taux_chomage': 'monthly',
    'pib_trimestriel': 'quarterly',
    'balance_commerciale_annuelle': 'annual',
}

# Délai appliqué à tout le panel : médiane des trois entités (France, Allemagne, Italie)
#   mensuelles et PIB : médiane(76, 167, 76) = 76 ; balance : médiane(258, 258, 258) = 258
APPLICABLE_DELAY = {'inflation_ipc': 76.0, 'taux_chomage': 76.0, 'pib_trimestriel': 76.0,
                    'balance_commerciale_annuelle': 258.0}

# Décalage calendaire attendu au 2024-09-15 : plus petit k tel que la période commençant k périodes
# avant celle de la date de prédiction, plus son délai, ne dépasse pas la date de prédiction
#   mensuelles (76 j) : 1er sept. -> k=1 : 1er août + 76 j = 16 oct. > 15 sept. ; k=2 : 1er juil. + 76 j = 15 sept.
#   PIB (76 j)        : 1er juil. + 76 j = 15 sept. -> k=0 (le troisième trimestre est publié le jour même)
#   balance (258 j)   : 1er janv. + 258 j = 15 sept. -> k=0
CALENDAR_N_PERIODS = {'inflation_ipc': -2, 'taux_chomage': -2, 'pib_trimestriel': 0,
                      'balance_commerciale_annuelle': 0}


def _released_value(couple) -> float:
    """Return the distinctive value published at the second download for a couple.

    Distinct values (9000 + rank of the couple) make each released value
    traceable after the shift, unlike a repetition of the last known value.
    """
    return 9000.0 + sorted(HIDDEN_DATES).index(couple)


def _visible_couples():
    """Return the delayed couples of a column of shared frequency, as parameters.

    France and Italie, inflation and unemployment are the couples published on the
    prediction date itself (July + 76 days = September 15), which the former 30-day
    convention pushed one month too far (ANO-DELAYS-042).
    """
    return [pytest.param(couple, id=f'{couple[0]}-{couple[1]}')
            for couple in sorted(HIDDEN_DATES) if couple[1] in SHARED_FREQUENCY]


@pytest.fixture(scope='module')
def two_downloads(_heterogeneous_coverage_panel_session):
    """Return ``(first_download, second_download)`` of the notebook 3 panel.

    The first download is the fixture itself (the last value of five columns per
    entity withheld by the simulated delays); the second one, on 2024-09-15,
    publishes every withheld value.
    """
    first = _heterogeneous_coverage_panel_session.copy()
    second = first.copy()
    for entity, column in HIDDEN_DATES:
        second.loc[(entity, TS(HIDDEN_DATES[(entity, column)])), column] = _released_value((entity, column))
    return first, second


@pytest.fixture(scope='module')
def detected_between_downloads(two_downloads):
    """Delays (days from the period start) of the values published between the two downloads."""
    first, second = two_downloads
    return compare_and_detect_delays(second, first, download_date=SECOND_DOWNLOAD, reference_point='start')


@pytest.fixture(scope='module')
def applicable_per_column(detected_between_downloads):
    """One delay per column of shared frequency, median over the entities, in days from the period start."""
    return calculate_applicable_delay(detected_between_downloads, 'start', SHARED_FREQUENCY,
                                      indicators=list(SHARED_FREQUENCY), unit='D')


@pytest.fixture(scope='module')
def fitted_on_panel(two_downloads, applicable_per_column):
    """``PublicationDelayTransformer`` fitted on the second download and its output at 2024-09-15."""
    _, second = two_downloads
    transformer = PublicationDelayTransformer(delays=applicable_per_column, prediction_date=SECOND_DOWNLOAD,
                                              handle_missing_delays='ignore')
    return transformer, _silently(transformer.fit_transform, second)


class TestTwoDownloadsOnHeterogeneousPanel:
    """Two downloads -> detection -> one delay per column -> ``PublicationDelayTransformer`` on the whole panel."""

    def test_detection_finds_every_published_value_with_its_delay(self, detected_between_downloads):
        """The 15 withheld values are detected, each with ``download - period start`` days."""
        delays = detected_between_downloads['delay'].to_dict()
        expected = {couple: DETECTED_DELAY_BY_DATE[date] for couple, date in HIDDEN_DATES.items()}
        assert delays == expected

    def test_applicable_delay_is_the_median_over_the_entities(self, applicable_per_column):
        """Monthly columns and GDP: median(76, 167, 76) = 76 days; trade balance: 258 days."""
        assert applicable_per_column['delay'].to_dict() == APPLICABLE_DELAY

    def test_column_with_per_entity_frequencies_is_rejected_on_the_panel(self, detected_between_downloads,
                                                                         two_downloads):
        """``depenses_publiques_pib`` (annual / quarterly by entity) cannot share one shift: error naming the factory.

        Gold value of its applicable delay, before the error: the German
        quarterly value of April is converted to its year (January 1 -> 258 days),
        hence median(258, 258, 258).
        """
        _, second = two_downloads
        applicable = calculate_applicable_delay(detected_between_downloads, 'start',
                                                {**SHARED_FREQUENCY, 'depenses_publiques_pib': 'annual'}, unit='D')
        transformer = PublicationDelayTransformer(delays=applicable, prediction_date=SECOND_DOWNLOAD,
                                                  handle_missing_delays='ignore')
        with pytest.raises(ValueError, match='create_delay_transformer_factory'):
            _silently(transformer.fit, second)

    @pytest.mark.parametrize('column', list(SHARED_FREQUENCY))
    def test_number_of_periods_follows_the_calendar(self, fitted_on_panel, column):
        """The shift of each column is the calendar one at 2024-09-15 (``CALENDAR_N_PERIODS``)."""
        transformer, _ = fitted_on_panel
        assert transformer.shift_params[column]['n_periods'] == CALENDAR_N_PERIODS[column]

    @pytest.mark.parametrize('couple', _visible_couples())
    def test_published_value_is_visible_at_the_prediction_date(self, fitted_on_panel, couple):
        """Per (entity, column): the value published on 2024-09-15 is the last one visible on that date.

        The prediction date is the second download: every value it publishes was
        available on that day and must therefore sit at a date not after it, and
        be the most recent such value of its couple.
        """
        _, transformed = fitted_on_panel
        entity, column = couple
        visible = transformed.loc[entity, column].dropna().sort_index().loc[:SECOND_DOWNLOAD]
        assert visible.iloc[-1] == _released_value(couple)

    def test_columns_without_delay_are_untouched(self, fitted_on_panel, two_downloads):
        """Columns without applied delay keep their values on the input rows."""
        _, second = two_downloads
        _, transformed = fitted_on_panel
        untouched = ['production_industrielle', 'depenses_publiques_pib', 'climat_affaires']
        pd.testing.assert_frame_equal(transformed[untouched].reindex(second.index), second[untouched])

    def test_output_is_ordered_by_entity_then_date(self, fitted_on_panel):
        """Like the input (and like a time series output), the panel output is sorted by (entity, date)."""
        _, transformed = fitted_on_panel
        assert transformed.index.is_monotonic_increasing

    def test_round_trip(self, fitted_on_panel, two_downloads):
        """``inverse_transform`` restores the second download exactly."""
        _, second = two_downloads
        transformer, transformed = fitted_on_panel
        pd.testing.assert_frame_equal(_silently(transformer.inverse_transform, transformed), second)
