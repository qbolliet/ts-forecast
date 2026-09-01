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


def _build_reference_timeseries(seed: int = 42) -> pd.DataFrame:
    """Build the ``TS`` reference dataset of ``high_frequency_imputer2_architecture.md`` §2.2.

    Month-end index (``ME``) from 2021-01-31 to 2023-12-31 (36 rows). Every
    column is additive (an annual value is the sum of its sub-periods). The
    annual columns carry the document's gold values, reused verbatim as gold
    cases by later implementation lots and therefore frozen.

    Args:
        seed: Unused; kept for signature parity with the other builders of
            this module (the construction is fully deterministic).

    Returns:
        Time series ``DataFrame`` with a ``DatetimeIndex`` named ``date`` and
        columns ``m1`` (monthly, never NaN), ``q1`` (quarterly, non-NaN only
        at quarter-end months), ``a1`` and ``a2`` (annual, non-NaN only at
        the three year-end anchors, values 120/132/150 and 60/66/72).

    Examples:
        >>> df = _build_reference_timeseries()
        >>> df.loc['2022-12-31', ['a1', 'a2']].tolist()
        [132.0, 66.0]
    """
    # Graine conservée pour l'homogénéité de signature : aucun tirage aléatoire.
    del seed

    dates = pd.date_range(start='2021-01-31', end='2023-12-31', freq='ME')
    df = pd.DataFrame(index=dates)
    df.index.name = 'date'

    # ----- m1 : mensuelle, dense, jamais NaN -----
    df['m1'] = 100.0 + np.arange(len(dates), dtype=float)

    # ----- q1 : trimestrielle, valeur uniquement aux fins de trimestre -----
    quarter_end_mask = dates.month.isin([3, 6, 9, 12])
    df['q1'] = np.nan
    df.loc[quarter_end_mask, 'q1'] = 10.0 * np.arange(1, quarter_end_mask.sum() + 1)

    # ----- a1 / a2 : annuelles, valeurs d'or du document (§2.2) -----
    annual_anchors = pd.to_datetime(['2021-12-31', '2022-12-31', '2023-12-31'])
    df['a1'] = np.nan
    df['a2'] = np.nan
    df.loc[annual_anchors, 'a1'] = [120.0, 132.0, 150.0]
    df.loc[annual_anchors, 'a2'] = [60.0, 66.0, 72.0]

    return df


def _build_panel_heterogeneous(seed: int = 42) -> pd.DataFrame:
    """Build the ``PANEL`` reference dataset of ``high_frequency_imputer2_architecture.md`` §2.3.

    Three entities ``FR`` / ``DE`` / ``IT`` sharing a month-end index
    (``ME``) from 2021-01-31 to 2023-12-31 (36 rows per entity, 108 total).
    Same mixed-frequency columns as :func:`_build_reference_timeseries`, plus
    ``climat_affaires``: a monthly business-survey indicator observed for
    ``FR`` and ``DE`` but structurally absent for ``IT`` (the column exists
    in the frame for every entity, the Italian entity simply never observes
    it). This is the support of the ``covariate_eligibility`` parameter
    (§4.5) and of the per-entity measurement of the central NaN invariant
    (§3). Every column is additive.

    Args:
        seed: Base random seed; each entity draws its ``climat_affaires``
            noise after ``np.random.seed(seed + hash(entity) % 1000)`` — the
            same seeding mechanism as :func:`_build_panel`.

    Returns:
        Panel ``DataFrame`` with a ``MultiIndex`` (``country``, ``date``) and
        entities ``FR`` / ``DE`` / ``IT``.

    Examples:
        >>> df = _build_panel_heterogeneous()
        >>> int(df.loc['IT', 'climat_affaires'].notna().sum())
        0
        >>> int(df.loc['FR', 'climat_affaires'].notna().sum())
        36
    """
    entities = ('FR', 'DE', 'IT')
    dates = pd.date_range(start='2021-01-31', end='2023-12-31', freq='ME')
    quarter_end_mask = dates.month.isin([3, 6, 9, 12])
    annual_anchors = pd.to_datetime(['2021-12-31', '2022-12-31', '2023-12-31'])

    all_data = []
    for entity in entities:
        # Même mécanisme de graine par entité que _build_panel.
        np.random.seed(seed + hash(entity) % 1000)

        df_entity = pd.DataFrame(index=dates)
        df_entity.index.name = 'date'
        df_entity['country'] = entity

        # ----- m1 : mensuelle, dense, jamais NaN -----
        df_entity['m1'] = 100.0 + np.arange(len(dates), dtype=float)

        # ----- q1 : trimestrielle, valeur uniquement aux fins de trimestre -----
        df_entity['q1'] = np.nan
        df_entity.loc[quarter_end_mask, 'q1'] = 10.0 * np.arange(1, quarter_end_mask.sum() + 1)

        # ----- a1 / a2 : annuelles, valeurs d'or du document (§2.2) -----
        df_entity['a1'] = np.nan
        df_entity['a2'] = np.nan
        df_entity.loc[annual_anchors, 'a1'] = [120.0, 132.0, 150.0]
        df_entity.loc[annual_anchors, 'a2'] = [60.0, 66.0, 72.0]

        # ----- climat_affaires : mensuelle pour FR et DE, jamais observée pour IT -----
        # La colonne existe pour les trois entités ; seule l'Italie ne l'observe
        # jamais (cas d'usage de covariate_eligibility, §4.5).
        df_entity['climat_affaires'] = np.nan
        if entity != 'IT':
            climat_noise = np.random.normal(0, 2.0, len(dates))
            df_entity['climat_affaires'] = 100.0 + climat_noise

        all_data.append(df_entity)

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


@pytest.fixture
def reference_timeseries() -> pd.DataFrame:
    """``TS`` reference dataset of ``high_frequency_imputer2_architecture.md`` §2.2.

    ``DatetimeIndex`` named ``date``, month-end anchored (``ME``), 36 rows
    from 2021-01-31 to 2023-12-31. Columns:

    - ``m1`` (monthly, dense, never NaN).
    - ``q1`` (quarterly: non-NaN only at quarter-end months 3/6/9/12).
    - ``a1`` (annual: non-NaN only at the three year-end anchors) — gold
      values 120 / 132 / 150.
    - ``a2`` (annual, same anchors) — gold values 60 / 66 / 72.

    The annual gold values match §2.2 verbatim and are reused as gold cases
    by later implementation lots; they must not change.
    """
    return _build_reference_timeseries()


@pytest.fixture
def mixed_freq_panel_heterogeneous() -> pd.DataFrame:
    """``PANEL`` reference dataset of ``high_frequency_imputer2_architecture.md`` §2.3.

    ``MultiIndex`` (``country``, ``date``) with 3 entities (``FR``, ``DE``,
    ``IT``), each sharing the same 36 month-end (``ME``) dates from
    2021-01-31 to 2023-12-31 (108 rows total). Columns ``m1`` / ``q1`` /
    ``a1`` / ``a2`` as in :func:`reference_timeseries`, plus:

    - ``climat_affaires`` (monthly business survey): observed for ``FR`` and
      ``DE`` (level ~100 with reproducible noise), entirely NaN for ``IT``.
      The column exists for every entity; only the Italian entity never
      observes it — the support of ``covariate_eligibility`` (§4.5) and of
      the per-entity NaN invariant (§3).
    """
    return _build_panel_heterogeneous()
