"""Pure builders of the reference datasets of the tests.

Gathers the builders historically private to
``tests/frequency/conftest.py`` (code replicas, without reading the
notebook, of notebook 2 ``notebooks/2 - QB - Mixed frequencies.ipynb``
and of the unified ``PANEL-X`` dataset of
``high_frequency_imputer2_architecture.md`` §2.6), made public to be
imported directly by tests and notebooks
(``from tests.support.datasets import build_panel_reference``).
"""
# Modules de base
import zlib
from typing import Optional

import numpy as np
import pandas as pd


def build_mixed_frequency_timeseries(
    start_date: str = '2018-01-01',
    end_date: str = '2024-07-01',
    annual_start_date: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the mixed-frequency time series dataset of notebook 2.

    Code replica (without reading the notebook) of ``df_timeseries`` in
    ``notebooks/2 - QB - Mixed frequencies.ipynb``, used as empirical
    reference in ``high_frequency_imputer_review.md``. Generalized to also
    cover the more realistic dataset of ``create_timeseries_dataset`` in
    ``notebooks/3 - QB - Panel a frequences mixtes heterogene.ipynb``: an
    ``annual_start_date`` earlier than ``start_date`` (default parameters
    of that notebook) makes the global index irregular - a few isolated
    annual anchors before the start of the monthly grid. This parameter
    defaults to ``start_date``, which preserves the historical regular
    dataset.

    Args:
        start_date: Start of the monthly grid (monthly and quarterly
            variables).
        end_date: End of the dataset (monthly grid and annual series).
        annual_start_date: Start of the annual trade balance series.
            ``None`` (default) bounds it to ``start_date``: the series stays
            within the monthly grid and the resulting index is regular. A
            date earlier than ``start_date`` adds annual anchors outside the
            grid and makes the index irregular.
        seed: Seed of the NumPy pseudo-random generator.

    Returns:
        ``DataFrame`` with a ``DatetimeIndex`` named ``date``, anchored at
        month start (``MS``) on the monthly grid, with columns:
        ``production_industrielle`` (monthly, dense from 2019-01, NaN
        before), ``inflation_ipc`` (monthly, dense, last value NaN),
        ``taux_chomage`` (monthly, dense, last value NaN),
        ``pib_trimestriel`` (quarterly, NaN outside quarter ends, last
        available quarter NaN), ``balance_commerciale_annuelle`` (annual,
        anchored at year start (``YS``) from ``annual_start_date``, last
        available year NaN).

    Examples:
        >>> df = build_mixed_frequency_timeseries()
        >>> list(df.columns)
        ['production_industrielle', 'inflation_ipc', 'taux_chomage', 'pib_trimestriel', 'balance_commerciale_annuelle']
        >>> len(df)
        79

        With a trade balance history earlier than the monthly grid
        (notebook 3 parameters), the global index becomes irregular:

        >>> from tsforecast.frequency import is_regular
        >>> df_irregular = build_mixed_frequency_timeseries(annual_start_date='2015-01-01')
        >>> is_regular(df_irregular)
        False
    """
    np.random.seed(seed)
    annual_start_date = annual_start_date or start_date

    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
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

    # ----- Variable annuelle : balance commerciale (~-26 à -9), ancrée en YS -----
    # Historique disponible dès `annual_start_date` : quand celui-ci précède
    # `start_date`, l'union avec la grille mensuelle introduit des ancres
    # annuelles isolées avant son début, et l'index global devient irrégulier.
    annual_dates = pd.date_range(start=annual_start_date, end=end_date, freq='YS')
    df = df.reindex(df.index.union(annual_dates))
    df.index.name = 'date'

    df['balance_commerciale_annuelle'] = np.nan
    for date in annual_dates:
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


# Dictionnaire par défaut de build_mixed_frequency_panel : 3 entités, grille mensuelle commune,
# sans depenses_publiques_pib ni climat_affaires (jeu régulier historique).
_DEFAULT_PANEL_COUNTRIES = {
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


def build_mixed_frequency_panel(
    seed: int = 42,
    countries: Optional[dict] = None,
) -> pd.DataFrame:
    """Build the mixed-frequency panel dataset of notebook 2.

    Code replica of ``df_panel`` in
    ``notebooks/2 - QB - Mixed frequencies.ipynb``. Generalized to also
    cover the more realistic and heterogeneous dataset of
    ``create_panel_dataset`` in ``notebooks/3 - QB - Panel a frequences
    mixtes heterogene.ipynb``: passing
    ``countries=HETEROGENEOUS_PANEL_COUNTRIES`` (defined below in this
    module) reproduces that dataset's structure - coverage specific to each
    entity, ``depenses_publiques_pib`` with a heterogeneous publication
    frequency, ``climat_affaires`` structurally absent for one entity. The
    default dictionary sets none of these keys: the historical regular
    dataset (3 entities on the same monthly grid, without
    ``depenses_publiques_pib`` nor ``climat_affaires``) is unchanged.

    Args:
        seed: Base seed of the NumPy pseudo-random generator; each entity
            draws after ``np.random.seed(seed + zlib.crc32(country) %
            1000)`` - ``crc32`` (not ``hash``) keeps the seed reproducible
            from one run to the next.
        countries: Dictionary ``{entity_name: parameters}``. ``None``
            (default) uses 3 entities (``France``, ``Allemagne``,
            ``Italie``) on a common monthly grid. Keys recognized per
            entity: ``pib_base``, ``inflation_base``, ``chomage_base``,
            ``prod_ind_start`` (always required); ``start_date`` /
            ``end_date`` (default ``'2018-01-01'`` / ``'2024-07-01'``,
            specific to each entity when given - heterogeneous coverage);
            ``annual_start_date`` (default ``start_date``; an earlier date
            makes the entity's index irregular, as in
            :func:`build_mixed_frequency_timeseries`); ``depenses_base`` /
            ``depenses_frequency`` (``'annuelle'`` or ``'trimestrielle'`` -
            absent: column ``depenses_publiques_pib`` omitted);
            ``climat_affaires_observe`` (``bool`` - absent: column
            ``climat_affaires`` omitted).

    Returns:
        Sorted panel ``DataFrame`` with a ``MultiIndex`` (``country``,
        ``date``). With the default dictionary: 3 entities (``France``,
        ``Allemagne``, ``Italie``), each on the same 79 monthly dates
        (``MS``) from 2018-01-01 to 2024-07-01 (237 rows). Same columns
        and orders of magnitude as :func:`build_mixed_frequency_timeseries`,
        with start dates and base levels specific to each entity.

    Examples:
        >>> df = build_mixed_frequency_panel()
        >>> df.index.names
        FrozenList(['country', 'date'])
        >>> sorted(df.index.get_level_values('country').unique())
        ['Allemagne', 'France', 'Italie']

        Heterogeneous coverage, per-entity publication frequency and
        irregular index (notebook 3 parameters):

        >>> df_heterogeneous = build_mixed_frequency_panel(countries=HETEROGENEOUS_PANEL_COUNTRIES)
        >>> int(df_heterogeneous.loc['Italie', 'climat_affaires'].notna().sum())
        0
    """
    if countries is None:
        countries = _DEFAULT_PANEL_COUNTRIES

    all_data = []
    for country, params in countries.items():
        # Graine déterministe par entité : ``hash`` d'une ``str`` est salé par
        # processus (non reproductible d'une exécution à l'autre), ``crc32`` non.
        np.random.seed(seed + zlib.crc32(country.encode()) % 1000)

        start_date = params.get('start_date', '2018-01-01')
        end_date = params.get('end_date', '2024-07-01')
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        n_periods = len(dates)

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

        # ----- depenses_publiques_pib : fréquence de publication propre à l'entité -----
        # Absente du dictionnaire par défaut (colonne omise) ; annuelle ou
        # trimestrielle selon l'entité dans ``HETEROGENEOUS_PANEL_COUNTRIES``.
        if 'depenses_frequency' in params:
            df_country['depenses_publiques_pib'] = np.nan
            publication_months = (
                [1] if params['depenses_frequency'] == 'annuelle' else [1, 4, 7, 10]
            )
            depenses_idx = 0
            for date in dates:
                if date.month in publication_months:
                    value = (
                        params['depenses_base']
                        + 0.1 * depenses_idx
                        + np.random.normal(0, 1.0)
                    )
                    df_country.loc[date, 'depenses_publiques_pib'] = value
                    depenses_idx += 1

        # ----- balance_commerciale_annuelle : ancrée en YS, historique propre à l'entité -----
        # Historique disponible dès `annual_start_date` (défaut : `start_date`,
        # index régulier). Une date antérieure à `start_date` introduit des
        # ancres annuelles isolées avant le début de la grille mensuelle de
        # l'entité, et rend son index irrégulier (cf. build_mixed_frequency_timeseries).
        annual_start_date = params.get('annual_start_date', start_date)
        annual_dates = pd.date_range(start=annual_start_date, end=end_date, freq='YS')
        df_country = df_country.reindex(df_country.index.union(annual_dates))
        df_country['country'] = country

        df_country['balance_commerciale_annuelle'] = np.nan
        for date in annual_dates:
            year_factor = date.year - 2018
            base = -20 + np.random.uniform(-10, 10) + year_factor * 2
            df_country.loc[date, 'balance_commerciale_annuelle'] = base

        # ----- climat_affaires : structurellement absente pour certaines entités -----
        # Absente du dictionnaire par défaut (colonne omise) ; la colonne
        # existe pour toutes les entités de ``HETEROGENEOUS_PANEL_COUNTRIES``
        # mais reste entièrement NaN pour celles dont
        # ``climat_affaires_observe`` vaut ``False`` (support de
        # ``covariate_eligibility``, §4.5 de la spec HFI2).
        if 'climat_affaires_observe' in params:
            df_country['climat_affaires'] = np.nan
            if params['climat_affaires_observe']:
                climat_noise = np.random.normal(0, 2.0, n_periods)
                df_country.loc[dates, 'climat_affaires'] = 100.0 + climat_noise

        df_country.loc[df_country.index[-1], 'inflation_ipc'] = np.nan
        df_country.loc[df_country.index[-1], 'taux_chomage'] = np.nan

        pib_available = df_country[df_country['pib_trimestriel'].notna()].index
        if len(pib_available) > 0:
            df_country.loc[pib_available[-1], 'pib_trimestriel'] = np.nan

        bc_available = df_country[df_country['balance_commerciale_annuelle'].notna()].index
        if len(bc_available) > 0:
            df_country.loc[bc_available[-1], 'balance_commerciale_annuelle'] = np.nan

        if 'depenses_publiques_pib' in df_country.columns:
            depenses_available = df_country[df_country['depenses_publiques_pib'].notna()].index
            if len(depenses_available) > 0:
                df_country.loc[depenses_available[-1], 'depenses_publiques_pib'] = np.nan

        all_data.append(df_country)

    df_panel = pd.concat(all_data, ignore_index=False)
    df_panel = df_panel.reset_index().rename(columns={'index': 'date'})
    df_panel = df_panel.set_index(['country', 'date'])
    df_panel = df_panel.sort_index()

    return df_panel


# Réplique fidèle du dictionnaire ``countries`` de ``create_panel_dataset``
# (cellule 7, ``notebooks/3 - QB - Panel a frequences mixtes heterogene.ipynb``) :
# couverture propre à chaque entité, dépenses publiques à fréquence de
# publication hétérogène (annuelle FR/IT, trimestrielle DE), climat_affaires
# jamais observée pour l'Italie, historique de balance commerciale antérieur
# au début de la grille mensuelle (index irrégulier). Le notebook amorce ses
# graines par ``seed + hash(country) % 1000`` — salé par processus, donc non
# reproductible d'une exécution à l'autre ; build_mixed_frequency_panel utilise toujours
# ``zlib.crc32``, seule la structure du notebook est reproduite ici.
HETEROGENEOUS_PANEL_COUNTRIES = {
    'France': {
        'climat_affaires_observe': True,
        'pib_base': 2800,
        'inflation_base': 1.5,
        'chomage_base': 8.0,
        'depenses_base': 55.0,
        'start_date': '2018-01-01',
        'end_date': '2024-07-01',
        'prod_ind_start': '2018-06-01',
        'depenses_frequency': 'annuelle',
        'annual_start_date': '2015-01-01',
    },
    'Allemagne': {
        'climat_affaires_observe': True,
        'pib_base': 3500,
        'inflation_base': 1.2,
        'chomage_base': 5.5,
        'depenses_base': 45.0,
        'start_date': '2018-07-01',
        'end_date': '2024-04-01',
        'prod_ind_start': '2019-01-01',
        'depenses_frequency': 'trimestrielle',
        'annual_start_date': '2016-01-01',
    },
    'Italie': {
        'climat_affaires_observe': False,
        'pib_base': 2200,
        'inflation_base': 1.8,
        'chomage_base': 10.5,
        'depenses_base': 50.0,
        'start_date': '2019-01-01',
        'end_date': '2024-07-01',
        'prod_ind_start': '2019-06-01',
        'depenses_frequency': 'annuelle',
        'annual_start_date': '2016-01-01',
    },
}


def build_panel_two_level(seed: int = 7) -> pd.DataFrame:
    """Build a two-level entity panel (country x sector), 2x2 entities.

    Args:
        seed: Base seed of the NumPy pseudo-random generator; each
            (country, sector) pair draws after
            ``np.random.seed(seed + zlib.crc32(f"{country}|{sector}") % 1000)``.

    Returns:
        Panel ``DataFrame`` with a ``MultiIndex`` (``country``, ``sector``,
        ``date``): 2 countries (``France``, ``Allemagne``) x 2 sectors
        (``Industrie``, ``Services``), each on the same 48 monthly dates
        (``MS``) from 2019-01-01 to 2022-12-01 (192 rows). Columns:
        ``indicateur_mensuel`` (monthly, dense, no NaN),
        ``indicateur_trimestriel`` (quarterly, NaN outside quarter ends,
        variable to impute).

    Examples:
        >>> df = build_panel_two_level()
        >>> df.index.names
        FrozenList(['country', 'sector', 'date'])
        >>> len(df)
        192
    """
    countries = ['France', 'Allemagne']
    sectors = ['Industrie', 'Services']
    dates = pd.date_range(start='2019-01-01', end='2022-12-01', freq='MS')
    n_periods = len(dates)

    all_data = []
    for country in countries:
        for sector in sectors:
            # Graine déterministe par entité (cf. build_mixed_frequency_panel) : ``crc32`` sur
            # la clé ``country|sector``, ``hash`` d'un tuple étant salé par processus.
            np.random.seed(
                seed + zlib.crc32(f"{country}|{sector}".encode()) % 1000
            )

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


def build_panel_reference(seed: int = 42) -> pd.DataFrame:
    """Build the unified ``PANEL-X`` reference dataset.

    Single source of truth behind the three frozen reference datasets of
    the spec (``high_frequency_imputer2_architecture.md`` §2.6): ``TS``
    (§2.2), ``PANEL`` (§2.3) and ``PANEL-F`` (§2.5) are strict projections
    of it (fixtures ``reference_timeseries``,
    ``mixed_freq_panel_heterogeneous``, ``mixed_freq_panel_multifrequency``).
    Three entities ``FR`` / ``DE`` / ``IT`` share a month-end index
    (``ME``) from 2021-01-31 to 2023-12-31 (36 dates per entity, 108
    rows). Each column is additive (an annual value is the sum of its
    sub-periods). Columns, in the union order of the three datasets:

    - ``m1`` (monthly, dense, never NaN): ``100 + rank``, identical across
      entities.
    - ``q1`` (quarterly: non-NaN only at quarter-end months 3/6/9/12):
      ``10 * k``, identical across entities.
    - ``a1`` (annual: non-NaN only at the three year-end anchors) - golden
      values §2.2 120 / 132 / 150, identical across entities.
    - ``a2`` (annual, same anchors) - golden values §2.2 60 / 66 / 72.
    - ``climat_affaires`` (monthly survey): observed for ``FR`` and ``DE``
      (level ~100 with reproducible noise), entirely NaN for ``IT`` (the
      column exists for each entity, only ``IT`` never observes it) -
      support of ``covariate_eligibility`` (§4.5).
    - ``v`` (frequency heterogeneous across entities): annual for ``FR``
      (3 anchors), quarterly for ``DE`` (12 anchors), monthly for ``IT``
      (36 values), chosen so that the three entities carry the same
      annual total (120 / 132 / 150) - support of cross-entity pooling
      (§5.8) and of invariants I14 to I16 (§16).

    Args:
        seed: Base seed for the ``climat_affaires`` noise; each entity
            draws after ``np.random.seed(seed + zlib.crc32(entity) %
            1000)``. ``crc32`` (not ``hash``) keeps the seed reproducible
            from one process to the next - no golden value depends on
            ``climat_affaires``, whose values are informative only.

    Returns:
        Sorted panel ``DataFrame`` with a ``MultiIndex`` (``country``,
        ``date``), entities ``FR`` / ``DE`` / ``IT``, columns ``m1``,
        ``q1``, ``a1``, ``a2``, ``climat_affaires``, ``v``.

    Examples:
        >>> df = build_panel_reference()
        >>> list(df.columns)
        ['m1', 'q1', 'a1', 'a2', 'climat_affaires', 'v']
        >>> df.loc['FR', ['a1', 'a2']].dropna().values.tolist()
        [[120.0, 60.0], [132.0, 66.0], [150.0, 72.0]]
        >>> df.loc['DE', 'v'].dropna().tolist()
        [28.0, 30.0, 31.0, 31.0, 31.0, 33.0, 34.0, 34.0, 36.0, 37.0, 38.0, 39.0]
        >>> int(df.loc['IT', 'climat_affaires'].notna().sum())
        0
    """
    entities = ('FR', 'DE', 'IT')
    dates = pd.date_range(start='2021-01-31', end='2023-12-31', freq='ME')
    quarter_end_mask = dates.month.isin([3, 6, 9, 12])
    quarter_end_dates = dates[quarter_end_mask]
    annual_anchors = pd.to_datetime(['2021-12-31', '2022-12-31', '2023-12-31'])

    # ----- Valeurs d'or de v, par entité (§2.5) : même total annuel pour les trois -----
    v_by_entity = {
        'FR': pd.Series([120.0, 132.0, 150.0], index=annual_anchors),
        'DE': pd.Series(
            [28.0, 30.0, 31.0, 31.0, 31.0, 33.0, 34.0, 34.0, 36.0, 37.0, 38.0, 39.0],
            index=quarter_end_dates,
        ),
        'IT': pd.Series([10.0] * 12 + [11.0] * 12 + [12.5] * 12, index=dates),
    }

    all_data = []
    for entity in entities:
        # Graine déterministe par entité (cf. build_mixed_frequency_panel) : seule climat_affaires
        # consomme le générateur, et seulement pour FR et DE.
        np.random.seed(seed + zlib.crc32(entity.encode()) % 1000)

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

        # ----- v : fréquence hétérogène par entité (annuelle / trimestrielle / mensuelle) -----
        df_entity['v'] = np.nan
        entity_v = v_by_entity[entity]
        df_entity.loc[entity_v.index, 'v'] = entity_v.to_numpy()

        all_data.append(df_entity)

    df_panel = pd.concat(all_data, ignore_index=False)
    df_panel = df_panel.reset_index().rename(columns={'index': 'date'})
    df_panel = df_panel.set_index(['country', 'date'])
    df_panel = df_panel.sort_index()

    return df_panel
