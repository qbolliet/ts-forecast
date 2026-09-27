"""Constructeurs purs des jeux de données de référence des tests.

Regroupe les constructeurs historiquement privés de
``tests/frequency/conftest.py`` (répliques par code, sans lecture de
notebook, du notebook 2 ``notebooks/2 - QB - Mixed frequencies.ipynb`` et du
jeu unifié ``PANEL-X`` de ``high_frequency_imputer2_architecture.md`` §2.6),
rendus publics pour être importés directement par les tests et les
notebooks (``from tests.support.datasets import build_panel_reference``).
"""
# Modules de base
import zlib

import numpy as np
import pandas as pd


def build_timeseries_nb2(seed: int = 42) -> pd.DataFrame:
    """Build the mixed-frequency time series dataset of notebook 2.

    Réplique par code (sans lecture du notebook) de ``df_timeseries`` dans
    ``notebooks/2 - QB - Mixed frequencies.ipynb``, utilisée comme référence
    empirique dans ``high_frequency_imputer_review.md``.

    Args:
        seed: Graine du générateur pseudo-aléatoire NumPy.

    Returns:
        ``DataFrame`` à ``DatetimeIndex`` nommé ``date``, ancré en début de
        mois (``MS``), 79 lignes de 2018-01-01 à 2024-07-01. Colonnes :
        ``production_industrielle`` (mensuelle, dense à partir de 2019-01,
        NaN avant), ``inflation_ipc`` (mensuelle, dense, dernière valeur
        NaN), ``taux_chomage`` (mensuelle, dense, dernière valeur NaN),
        ``pib_trimestriel`` (trimestrielle, NaN hors fin de trimestre,
        dernier trimestre disponible NaN), ``balance_commerciale_annuelle``
        (annuelle, NaN hors janvier, dernière année disponible NaN).

    Examples:
        >>> df = build_timeseries_nb2()
        >>> list(df.columns)
        ['production_industrielle', 'inflation_ipc', 'taux_chomage', 'pib_trimestriel', 'balance_commerciale_annuelle']
        >>> len(df)
        79
    """
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


def build_panel_nb2(seed: int = 42) -> pd.DataFrame:
    """Build the mixed-frequency panel dataset of notebook 2.

    Réplique par code de ``df_panel`` dans
    ``notebooks/2 - QB - Mixed frequencies.ipynb``.

    Args:
        seed: Graine de base du générateur pseudo-aléatoire NumPy ; chaque
            entité tire après ``np.random.seed(seed + zlib.crc32(pays) %
            1000)`` — ``crc32`` (et non ``hash``) garde la graine
            reproductible d'une exécution à l'autre.

    Returns:
        ``DataFrame`` panel à ``MultiIndex`` (``country``, ``date``) avec 3
        entités (``France``, ``Allemagne``, ``Italie``), chacune sur les 79
        mêmes dates mensuelles (``MS``) de 2018-01-01 à 2024-07-01 (237
        lignes). Mêmes colonnes et ordres de grandeur que
        :func:`build_timeseries_nb2`, avec dates de démarrage et niveaux de
        base spécifiques à chaque entité.

    Examples:
        >>> df = build_panel_nb2()
        >>> df.index.names
        FrozenList(['country', 'date'])
        >>> sorted(df.index.get_level_values('country').unique())
        ['Allemagne', 'France', 'Italie']
    """
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
        # Graine déterministe par entité : ``hash`` d'une ``str`` est salé par
        # processus (non reproductible d'une exécution à l'autre), ``crc32`` non.
        np.random.seed(seed + zlib.crc32(country.encode()) % 1000)

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


def build_panel_two_level(seed: int = 7) -> pd.DataFrame:
    """Build a two-level entity panel (country x sector), 2x2 entities.

    Args:
        seed: Graine de base du générateur pseudo-aléatoire NumPy ; chaque
            couple (pays, secteur) tire après
            ``np.random.seed(seed + zlib.crc32(f"{pays}|{secteur}") % 1000)``.

    Returns:
        ``DataFrame`` panel à ``MultiIndex`` (``country``, ``sector``,
        ``date``) avec 2 pays (``France``, ``Allemagne``) x 2 secteurs
        (``Industrie``, ``Services``), chacun sur les 48 mêmes dates
        mensuelles (``MS``) de 2019-01-01 à 2022-12-01 (192 lignes).
        Colonnes : ``indicateur_mensuel`` (mensuelle, dense, sans NaN),
        ``indicateur_trimestriel`` (trimestrielle, NaN hors fin de
        trimestre, variable à imputer).

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
            # Graine déterministe par entité (cf. build_panel_nb2) : ``crc32`` sur
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

    Source de vérité unique derrière les trois jeux de référence figés de la
    spec (``high_frequency_imputer2_architecture.md`` §2.6) : ``TS`` (§2.2),
    ``PANEL`` (§2.3) et ``PANEL-F`` (§2.5) en sont des projections strictes
    (fixtures ``reference_timeseries``, ``mixed_freq_panel_heterogeneous``,
    ``mixed_freq_panel_multifrequency``). Trois entités ``FR`` / ``DE`` /
    ``IT`` partagent un index en fin de mois (``ME``) de 2021-01-31 à
    2023-12-31 (36 dates par entité, 108 lignes). Chaque colonne est
    additive (une valeur annuelle est la somme de ses sous-périodes).
    Colonnes, dans l'ordre d'union des trois jeux :

    - ``m1`` (mensuelle, dense, jamais NaN) : ``100 + rang``, identique
      entre entités.
    - ``q1`` (trimestrielle : non-NaN uniquement aux mois de fin de
      trimestre 3/6/9/12) : ``10 * k``, identique entre entités.
    - ``a1`` (annuelle : non-NaN uniquement aux trois ancres de fin
      d'année) — valeurs d'or §2.2 120 / 132 / 150, identiques entre
      entités.
    - ``a2`` (annuelle, mêmes ancres) — valeurs d'or §2.2 60 / 66 / 72.
    - ``climat_affaires`` (enquête mensuelle) : observée pour ``FR`` et
      ``DE`` (niveau ~100 avec bruit reproductible), entièrement NaN pour
      ``IT`` (la colonne existe pour chaque entité, seule ``IT`` ne
      l'observe jamais) — support de ``covariate_eligibility`` (§4.5).
    - ``v`` (fréquence hétérogène par entité) : annuelle pour ``FR`` (3
      ancres), trimestrielle pour ``DE`` (12 ancres), mensuelle pour ``IT``
      (36 valeurs), choisie pour que les trois entités portent le même
      total annuel (120 / 132 / 150) — support de la mutualisation
      inter-entités (§5.8) et des invariants I14 à I16 (§16).

    Args:
        seed: Graine de base pour le bruit de ``climat_affaires`` ; chaque
            entité tire après ``np.random.seed(seed + zlib.crc32(entité) %
            1000)``. ``crc32`` (et non ``hash``) garde la graine
            reproductible d'un processus à l'autre — aucune valeur d'or ne
            dépend de ``climat_affaires``, dont les valeurs sont
            informatives uniquement.

    Returns:
        ``DataFrame`` panel à ``MultiIndex`` trié (``country``, ``date``),
        entités ``FR`` / ``DE`` / ``IT``, colonnes ``m1``, ``q1``, ``a1``,
        ``a2``, ``climat_affaires``, ``v``.

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
        # Graine déterministe par entité (cf. build_panel_nb2) : seule climat_affaires
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
