"""Fixtures pour les tests de HighFrequencyImputer et de ses dépendances.

Les jeux de données reproduisent par code (sans lecture du notebook) la
structure des données de `notebooks/2 - QB - Mixed frequencies.ipynb`
(`df_timeseries` et `df_panel`), utilisées comme référence empirique dans
`high_frequency_imputer_review.md`.
"""
# Modules de base
import numpy as np
import pandas as pd
import pytest


def _build_timeseries(seed: int = 42) -> pd.DataFrame:
    """Build the mixed-frequency time series dataset (see mixed_freq_timeseries)."""
    np.random.seed(seed)

    dates = pd.date_range(start='2018-01-01', end='2024-07-01', freq='MS')
    n_periods = len(dates)

    df = pd.DataFrame(index=dates)
    df.index.name = 'date'

    # ----- Variable mensuelle dense : production industrielle (~100-115) -----
    trend = np.linspace(100, 115, n_periods)
    seasonal = 3 * np.sin(2 * np.pi * np.arange(n_periods) / 12)
    noise = np.random.normal(0, 1.5, n_periods)
    df['production_industrielle'] = trend + seasonal + noise

    # ----- Variable mensuelle dense : inflation (IPC, ~0.5-5) -----
    inflation_trend = np.linspace(1.2, 2.8, n_periods)
    inflation_noise = np.random.normal(0, 0.3, n_periods)
    df['inflation_ipc'] = np.clip(inflation_trend + inflation_noise, 0.5, 5.0)

    # ----- Variable mensuelle dense : taux de chômage (~4-15) -----
    chomage_trend = np.concatenate([
        np.linspace(8.5, 7.0, n_periods // 3),
        np.linspace(7.0, 9.5, n_periods // 3),
        np.linspace(9.5, 7.5, n_periods - 2 * (n_periods // 3))
    ])
    chomage_noise = np.random.normal(0, 0.2, n_periods)
    df['taux_chomage'] = np.clip(chomage_trend + chomage_noise, 4.0, 15.0)

    # ----- Variable trimestrielle : PIB (~2500-2600), NaN hors fin de trimestre -----
    pib_base = 2500
    pib_growth_quarterly = 0.5
    df['pib_trimestriel'] = np.nan
    quarter_start_months = [1, 4, 7, 10]
    quarter_idx = 0
    for date in dates:
        if date.month in quarter_start_months:
            growth = pib_growth_quarterly + np.random.normal(0, 0.3)
            df.loc[date, 'pib_trimestriel'] = pib_base * (1 + growth / 100) ** quarter_idx
            quarter_idx += 1

    # ----- Variable annuelle : balance commerciale (~-26 à -9), NaN sauf en janvier -----
    df['balance_commerciale_annuelle'] = np.nan
    for date in dates:
        if date.month == 1:
            year_factor = date.year - 2018
            base_balance = -25 + year_factor * 3 + np.random.normal(0, 5)
            df.loc[date, 'balance_commerciale_annuelle'] = base_balance

    # ----- Simulation de délais de publication (dernières valeurs retirées) -----
    df.loc[df.index[-1], 'inflation_ipc'] = np.nan
    df.loc[df.index[-1], 'taux_chomage'] = np.nan

    pib_available = df[df['pib_trimestriel'].notna()].index
    if len(pib_available) > 0:
        df.loc[pib_available[-1], 'pib_trimestriel'] = np.nan

    bc_available = df[df['balance_commerciale_annuelle'].notna()].index
    if len(bc_available) > 0:
        df.loc[bc_available[-1], 'balance_commerciale_annuelle'] = np.nan

    # ----- Historique limité : la production industrielle démarre en 2019 -----
    mask_before_2019 = df.index < '2019-01-01'
    df.loc[mask_before_2019, 'production_industrielle'] = np.nan

    return df


def _build_panel(seed: int = 42) -> pd.DataFrame:
    """Build the mixed-frequency panel dataset (see mixed_freq_panel)."""
    countries = {
        'France': {
            'pib_base': 2800,
            'inflation_base': 1.5,
            'chomage_base': 8.0,
            'prod_ind_start': '2018-06-01',
        },
        'Allemagne': {
            'pib_base': 3500,
            'inflation_base': 1.2,
            'chomage_base': 5.5,
            'prod_ind_start': '2019-01-01',
        },
        'Italie': {
            'pib_base': 2200,
            'inflation_base': 1.8,
            'chomage_base': 10.5,
            'prod_ind_start': '2019-06-01',
        },
    }

    dates = pd.date_range(start='2018-01-01', end='2024-07-01', freq='MS')
    n_periods = len(dates)

    all_data = []
    for country, params in countries.items():
        np.random.seed(seed + hash(country) % 1000)

        df_country = pd.DataFrame(index=dates)
        df_country['country'] = country

        trend = np.linspace(100, 112 + np.random.uniform(-3, 3), n_periods)
        seasonal = 2.5 * np.sin(2 * np.pi * np.arange(n_periods) / 12)
        noise = np.random.normal(0, 1.2, n_periods)
        df_country['production_industrielle'] = trend + seasonal + noise

        prod_start = pd.Timestamp(params['prod_ind_start'])
        df_country.loc[df_country.index < prod_start, 'production_industrielle'] = np.nan

        infl_trend = np.linspace(
            params['inflation_base'],
            params['inflation_base'] + np.random.uniform(0.5, 2.0),
            n_periods
        )
        infl_noise = np.random.normal(0, 0.25, n_periods)
        df_country['inflation_ipc'] = np.clip(infl_trend + infl_noise, 0.3, 6.0)

        chomage_base = params['chomage_base']
        chomage_evolution = np.concatenate([
            np.linspace(chomage_base, chomage_base - 1, n_periods // 3),
            np.linspace(chomage_base - 1, chomage_base + 2, n_periods // 3),
            np.linspace(chomage_base + 2, chomage_base + 0.5, n_periods - 2 * (n_periods // 3))
        ])
        chomage_noise = np.random.normal(0, 0.15, n_periods)
        df_country['taux_chomage'] = np.clip(chomage_evolution + chomage_noise, 2.5, 15.0)

        df_country['pib_trimestriel'] = np.nan
        quarter_end_months = [1, 4, 7, 10]
        quarter_idx = 0
        for date in dates:
            if date.month in quarter_end_months:
                growth = 0.4 + np.random.normal(0, 0.35)
                df_country.loc[date, 'pib_trimestriel'] = (
                    params['pib_base'] * (1 + growth / 100) ** quarter_idx
                )
                quarter_idx += 1

        df_country['balance_commerciale_annuelle'] = np.nan
        for date in dates:
            if date.month == 1:
                year_factor = date.year - 2018
                base = -20 + np.random.uniform(-10, 10) + year_factor * 2
                df_country.loc[date, 'balance_commerciale_annuelle'] = base

        df_country.loc[df_country.index[-1], 'inflation_ipc'] = np.nan
        df_country.loc[df_country.index[-1], 'taux_chomage'] = np.nan

        pib_available = df_country[df_country['pib_trimestriel'].notna()].index
        if len(pib_available) > 0:
            df_country.loc[pib_available[-1], 'pib_trimestriel'] = np.nan

        bc_available = df_country[df_country['balance_commerciale_annuelle'].notna()].index
        if len(bc_available) > 0:
            df_country.loc[bc_available[-1], 'balance_commerciale_annuelle'] = np.nan

        all_data.append(df_country)

    df_panel = pd.concat(all_data, ignore_index=False)
    df_panel = df_panel.reset_index().rename(columns={'index': 'date'})
    df_panel = df_panel.set_index(['country', 'date'])
    df_panel = df_panel.sort_index()

    return df_panel


@pytest.fixture
def mixed_freq_timeseries() -> pd.DataFrame:
    """Mixed-frequency macroeconomic time series (mirrors df_timeseries).

    DatetimeIndex named ``date``, month-start anchored (``MS``), 79 rows
    from 2018-01-01 to 2024-07-01. Columns and their orders of magnitude:

    - ``production_industrielle`` (monthly, dense from 2019-01, NaN before):
      ~100-115.
    - ``inflation_ipc`` (monthly, dense): ~0.5-5.0, last observation NaN
      (simulated 1-month publication delay).
    - ``taux_chomage`` (monthly, dense): ~4.0-15.0, last observation NaN
      (simulated 1-month publication delay).
    - ``pib_trimestriel`` (quarterly: non-NaN only at quarter-start months
      1/4/7/10, NaN elsewhere): ~2500-2600, last available quarter NaN
      (simulated 2-month publication delay).
    - ``balance_commerciale_annuelle`` (annual: non-NaN only in January,
      NaN elsewhere): ~-26 to -9, last available year NaN (simulated
      3-month publication delay).
    """
    return _build_timeseries()


def _build_panel_two_level(seed: int = 7) -> pd.DataFrame:
    """Build a two-level entity panel (country x sector), 2x2 entities."""
    countries = ['France', 'Allemagne']
    sectors = ['Industrie', 'Services']
    dates = pd.date_range(start='2019-01-01', end='2022-12-01', freq='MS')
    n_periods = len(dates)

    all_data = []
    for country in countries:
        for sector in sectors:
            np.random.seed(seed + hash((country, sector)) % 1000)

            df = pd.DataFrame(index=dates)
            df['country'] = country
            df['sector'] = sector

            # ----- Variable mensuelle dense (~100-110), aucune valeur manquante -----
            trend = np.linspace(100, 110, n_periods)
            noise = np.random.normal(0, 1.0, n_periods)
            df['indicateur_mensuel'] = trend + noise

            # ----- Variable trimestrielle à imputer, NaN hors fin de trimestre -----
            df['indicateur_trimestriel'] = np.nan
            quarter_start_months = [1, 4, 7, 10]
            quarter_idx = 0
            for date in dates:
                if date.month in quarter_start_months:
                    df.loc[date, 'indicateur_trimestriel'] = (
                        500 + quarter_idx * 5 + np.random.normal(0, 3)
                    )
                    quarter_idx += 1

            all_data.append(df)

    df_panel = pd.concat(all_data, ignore_index=False)
    df_panel = df_panel.reset_index().rename(columns={'index': 'date'})
    df_panel = df_panel.set_index(['country', 'sector', 'date'])
    df_panel = df_panel.sort_index()

    return df_panel


@pytest.fixture
def panel_two_level_dataset() -> pd.DataFrame:
    """Two-level entity panel (country x sector), 2x2 = 4 entities.

    MultiIndex (``country``, ``sector``, ``date``) with 2 countries
    (``France``, ``Allemagne``) x 2 sectors (``Industrie``, ``Services``),
    each with the same 48 month-start (``MS``) dates from 2019-01-01 to
    2022-12-01 (192 rows total). Columns:

    - ``indicateur_mensuel`` (monthly, dense, no NaN): ~100-110.
    - ``indicateur_trimestriel`` (quarterly: non-NaN only at quarter-start
      months 1/4/7/10, NaN elsewhere) — the variable to impute: ~500-575.
    """
    return _build_panel_two_level()


@pytest.fixture
def mixed_freq_panel() -> pd.DataFrame:
    """Mixed-frequency macroeconomic panel (mirrors df_panel).

    MultiIndex (``country``, ``date``) with 3 entities (``France``,
    ``Allemagne``, ``Italie``), each with the same 79 month-start (``MS``)
    dates from 2018-01-01 to 2024-07-01 (237 rows total). Same columns and
    orders of magnitude as :func:`mixed_freq_timeseries`, per entity:

    - ``production_industrielle`` (monthly, dense from an entity-specific
      start date: France 2018-06, Allemagne 2019-01, Italie 2019-06):
      ~100-115.
    - ``inflation_ipc`` (monthly, dense): entity-specific base level
      (France ~1.5-3.5, Allemagne ~1.2-3.2, Italie ~1.8-3.8), last
      observation per entity NaN.
    - ``taux_chomage`` (monthly, dense): entity-specific base level
      (France ~7-10, Allemagne ~4.5-7.5, Italie ~9.5-12.5), last
      observation per entity NaN.
    - ``pib_trimestriel`` (quarterly, non-NaN only at months 1/4/7/10):
      entity-specific base (France ~2800, Allemagne ~3500, Italie
      ~2200), last available quarter per entity NaN.
    - ``balance_commerciale_annuelle`` (annual, non-NaN only in January):
      ~-30 to +5, last available year per entity NaN.
    """
    return _build_panel()
