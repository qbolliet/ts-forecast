"""Integration of ``ShiftTransformer`` / ``MaskTransformer`` with ``PanelwiseTransformer``.

Chains ``tsforecast.delays.transformers.ShiftTransformer`` and
``MaskTransformer`` with ``tsforecast.panel.PanelwiseTransformer`` on small
balanced panels (``MultiIndex`` (entity, date)): each entity is transformed
independently, with one shared transformer or one transformer per entity built
by a factory, and the round trip ``inverse_transform(transform(X))`` restores
every entity. Gold values are calendar arithmetic done by hand (dates moved by
``k`` months, last days of each month masked).

Also contains the performance check on a 50-entity daily panel.

Migrated from ``tests/integration/delays/test_integration.py`` (prompt D5); the
failures ``Cannot determine time index`` of these tests were fixed by
``ANO-UTILS-029`` (``_validate_time_index`` reading the last level of a
``MultiIndex``, prompt U5).
"""
# Modules de base
import time

import numpy as np
import pandas as pd
import pytest

# Composants enchaînés
from tsforecast.delays.transformers import MaskTransformer, ShiftTransformer
from tsforecast.panel import PanelwiseTransformer

TS = pd.Timestamp

COUNTRIES = ['France', 'Germany', 'Italy']
ENTITIES = ['entity_A', 'entity_B', 'entity_C']


# =============================================================================
# Jeux de données (déterministes)
# =============================================================================
@pytest.fixture
def panel_data() -> pd.DataFrame:
    """Balanced monthly (``MS``) panel, January 2023 to June 2024, indexed by (country, date)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range('2023-01-01', '2024-06-01', freq='MS', name='date')
    frames = {
        country: pd.DataFrame({'GDP': 100 + rng.normal(0, 2, len(dates)),
                               'inflation': 2.5 + rng.normal(0, 0.5, len(dates))}, index=dates)
        for country in COUNTRIES
    }
    return pd.concat(frames, names=['country', 'date'])


@pytest.fixture
def daily_panel_data() -> pd.DataFrame:
    """Balanced daily panel, January 1 to March 31, 2024, indexed by (entity, date)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D', name='date')
    frames = {
        entity: pd.DataFrame({'value1': 100 + rng.normal(0, 10, len(dates)),
                              'value2': 50 + rng.normal(0, 5, len(dates))}, index=dates)
        for entity in ENTITIES
    }
    return pd.concat(frames, names=['entity', 'date'])


def _panelwise(transformer) -> PanelwiseTransformer:
    """Wrap a transformer (or a factory) for a panel whose entity and time are index levels."""
    return PanelwiseTransformer(transformer=transformer, time_col=None, panel_cols=None)


def _shift_dates(panel: pd.DataFrame, months_later: int) -> pd.DataFrame:
    """Return ``panel`` with the dates of every entity moved ``months_later`` months later.

    Args:
        panel: Panel indexed by (entity, date).
        months_later: Number of months (negative: earlier dates).

    Returns:
        Same values, dates moved by calendar months.
    """
    entities = panel.index.get_level_values(0)
    dates = panel.index.get_level_values(1) + pd.DateOffset(months=months_later)
    return panel.set_axis(pd.MultiIndex.from_arrays([entities, dates], names=panel.index.names))


# =============================================================================
# ShiftTransformer appliqué entité par entité
# =============================================================================
class TestShiftTransformerWithPanelwise:
    """A shared ``ShiftTransformer`` moves the dates of every entity, independently."""

    def test_each_entity_is_shifted_by_calendar_months(self, panel_data):
        """``n_periods=2`` moves every date of every entity two months earlier, values unchanged."""
        shifted = _panelwise(ShiftTransformer(n_periods=2, frequency='M')).fit_transform(panel_data)
        # Valeur d'or : chaque entité garde ses 18 valeurs, dates reculées de deux mois
        # (janvier 2023 -> novembre 2022), entités dans l'ordre d'entrée
        pd.testing.assert_frame_equal(shifted, _shift_dates(panel_data, -2), check_freq=False)

    def test_shift_introduces_no_nan(self, panel_data):
        """A complete panel stays complete: the shift moves dates, it does not blank values."""
        shifted = _panelwise(ShiftTransformer(n_periods=3, frequency='M')).fit_transform(panel_data)
        assert shifted.isna().sum().sum() == 0

    def test_round_trip_per_entity(self, panel_data):
        """``inverse_transform(transform(X))`` restores every entity exactly."""
        panelwise = _panelwise(ShiftTransformer(n_periods=4, frequency='M'))
        recovered = panelwise.inverse_transform(panelwise.fit_transform(panel_data))
        pd.testing.assert_frame_equal(recovered, panel_data, check_freq=False)

    def test_one_shift_per_entity_through_a_factory(self, panel_data):
        """A factory gives each entity its own shift: 1, 2 and 3 months earlier.

        The factory receives the entity key as a tuple; a key mismatch would
        silently give every entity the default shift, hence the check on the
        first date of each entity rather than on the number of transformers.
        """
        shifts = {('France',): 1, ('Germany',): 2, ('Italy',): 3}

        def shifter_factory(entity_key):
            # Décalage propre à chaque pays, 0 pour une clé inattendue
            return ShiftTransformer(n_periods=shifts.get(entity_key, 0), frequency='M')

        shifted = _panelwise(shifter_factory).fit_transform(panel_data)
        first_dates = {entity: shifted.loc[entity].index.min() for entity in COUNTRIES}
        # Valeur d'or : 2023-01-01 reculé de 1, 2 et 3 mois
        assert first_dates == {'France': TS('2022-12-01'), 'Germany': TS('2022-11-01'), 'Italy': TS('2022-10-01')}


# =============================================================================
# MaskTransformer appliqué entité par entité
# =============================================================================
class TestMaskTransformerWithPanelwise:
    """A shared ``MaskTransformer`` hides the same calendar positions in every entity."""

    def test_index_is_preserved(self, daily_panel_data):
        """Masking blanks cells, it neither adds nor drops rows nor reorders entities."""
        masked = _panelwise(MaskTransformer(n_obs=3, mask_frequency='M', how='last')).fit_transform(daily_panel_data)
        pd.testing.assert_index_equal(masked.index, daily_panel_data.index)

    @pytest.mark.parametrize('entity', ENTITIES)
    def test_last_five_days_of_each_month_are_masked(self, daily_panel_data, entity):
        """``n_obs=5, how='last'``: January 27-31, February 25-29 and March 27-31 are hidden, both columns."""
        masked = _panelwise(MaskTransformer(n_obs=5, mask_frequency='M', how='last')).fit_transform(daily_panel_data)
        entity_masked = masked.loc[entity]
        # Valeur d'or : cinq derniers jours de chaque mois (2024 bissextile : février finit le 29)
        expected = (list(pd.date_range('2024-01-27', '2024-01-31')) + list(pd.date_range('2024-02-25', '2024-02-29'))
                    + list(pd.date_range('2024-03-27', '2024-03-31')))
        masked_dates = {column: list(entity_masked.index[entity_masked[column].isna()]) for column in entity_masked}
        assert masked_dates == {'value1': expected, 'value2': expected}

    def test_round_trip_per_entity(self, daily_panel_data):
        """``inverse_transform`` gives back the masked cells of every entity (``how='first'``)."""
        panelwise = _panelwise(MaskTransformer(n_obs=4, mask_frequency='M', how='first'))
        recovered = panelwise.inverse_transform(panelwise.fit_transform(daily_panel_data))
        pd.testing.assert_frame_equal(recovered, daily_panel_data, check_freq=False)


# =============================================================================
# Performance
# =============================================================================
class TestPerformance:
    """Order of magnitude of the cost of a per-entity shift on a large panel."""

    def test_large_panel_performance(self):
        """50 entities x 365 days shifted per entity well under a minute, shape preserved."""
        rng = np.random.default_rng(42)
        dates = pd.date_range('2024-01-01', periods=365, freq='D', name='date')
        frames = {f'entity_{i}': pd.DataFrame({'value': 100 + rng.normal(0, 10, len(dates))}, index=dates)
                  for i in range(50)}
        df = pd.concat(frames, names=['entity', 'date'])

        start_time = time.perf_counter()
        shifted = _panelwise(ShiftTransformer(n_periods=10, frequency='D')).fit_transform(df)
        elapsed = time.perf_counter() - start_time

        # Seuil large : détection d'une régression d'ordre de grandeur, pas d'un micro-ralentissement
        assert elapsed < 60 and shifted.shape == df.shape, f"{elapsed:.2f}s, shape {shifted.shape}"
