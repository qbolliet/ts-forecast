"""Pytest fixtures shared by the whole ``tests/`` suite.

Gathers the fixtures historically defined in ``tests/conftest.py``
(generic synthetic datasets) and ``tests/frequency/conftest.py``
(mixed-frequency datasets reproducing notebook 2 and the ``PANEL-X``
dataset of the ``HighFrequencyImputer`` spec). Registered by
``pytest_plugins = ["tests.support.fixtures"]`` in ``tests/conftest.py``.
"""
# Modules de base
import numpy as np
import pandas as pd
import pytest

from tests.support.datasets import (
    HETEROGENEOUS_PANEL_COUNTRIES,
    build_mixed_frequency_panel,
    build_mixed_frequency_timeseries,
    build_panel_reference,
    build_panel_two_level,
)


# =============================================================================
# Fixtures génériques (anciennement tests/conftest.py)
# =============================================================================


@pytest.fixture
def simple_timeseries():
    """Simple time series data for testing."""
    dates = pd.date_range('2020-01-01', periods=30, freq='D')
    return pd.Series(range(30), index=dates, name='value')


@pytest.fixture
def simple_timeseries_df():
    """Simple time series DataFrame for testing."""
    dates = pd.date_range('2020-01-01', periods=30, freq='D')
    return pd.DataFrame({
        'feature1': range(30),
        'feature2': np.sin(np.arange(30) * 0.1),
        'target': range(30, 60)
    }, index=dates)


@pytest.fixture
def simple_panel():
    """Simple panel data for testing."""
    entities = ['A', 'B', 'C']
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
    return pd.DataFrame({
        'feature1': range(60),
        'feature2': np.random.randn(60),
        'target': range(60, 120)
    }, index=idx)


@pytest.fixture
def unbalanced_panel():
    """Unbalanced panel data for testing."""
    data = []
    indices = []

    # Entity A: 10 periods
    entity_a_dates = pd.date_range('2020-01-01', periods=10, freq='D')
    for i, date in enumerate(entity_a_dates):
        data.append([i, np.random.randn(), i + 100])
        indices.append(('A', date))

    # Entity B: 15 periods
    entity_b_dates = pd.date_range('2020-01-01', periods=15, freq='D')
    for i, date in enumerate(entity_b_dates):
        data.append([i + 10, np.random.randn(), i + 200])
        indices.append(('B', date))

    # Entity C: 8 periods
    entity_c_dates = pd.date_range('2020-01-01', periods=8, freq='D')
    for i, date in enumerate(entity_c_dates):
        data.append([i + 25, np.random.randn(), i + 300])
        indices.append(('C', date))

    idx = pd.MultiIndex.from_tuples(indices, names=['entity', 'date'])
    return pd.DataFrame(data, columns=['feature1', 'feature2', 'target'], index=idx)


@pytest.fixture
def large_timeseries():
    """Large time series for performance testing."""
    dates = pd.date_range('2000-01-01', periods=1000, freq='D')
    return pd.DataFrame({
        'feature1': np.cumsum(np.random.randn(1000)),
        'feature2': np.sin(np.arange(1000) * 0.01),
        'feature3': np.random.exponential(1, 1000),
        'target': np.cumsum(np.random.randn(1000)) + np.random.randn(1000) * 0.1
    }, index=dates)


@pytest.fixture
def large_panel():
    """Large panel data for performance testing."""
    entities = [f'entity_{i:02d}' for i in range(10)]
    dates = pd.date_range('2010-01-01', periods=200, freq='D')
    idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])

    n_obs = len(entities) * len(dates)
    return pd.DataFrame({
        'feature1': np.random.randn(n_obs),
        'feature2': np.cumsum(np.random.randn(n_obs).reshape(len(entities), -1), axis=1).flatten(),
        'feature3': np.random.exponential(1, n_obs),
        'target': np.random.randn(n_obs) * 0.5 + np.arange(n_obs) * 0.01
    }, index=idx)


@pytest.fixture
def numpy_array_data():
    """Simple numpy array data for testing."""
    return np.random.randn(50, 3)


@pytest.fixture
def mismatched_xy():
    """X and y with mismatched indices for testing error handling."""
    dates_x = pd.date_range('2020-01-01', periods=20, freq='D')
    dates_y = pd.date_range('2020-01-02', periods=20, freq='D')  # Offset by 1 day

    X = pd.DataFrame({'feature': range(20)}, index=dates_x)
    y = pd.Series(range(20, 40), index=dates_y)

    return X, y


@pytest.fixture
def unsorted_timeseries():
    """Unsorted time series data for testing sorting functionality."""
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    # Shuffle the dates
    shuffled_indices = [2, 0, 7, 4, 1, 9, 3, 6, 8, 5]
    shuffled_dates = [dates[i] for i in shuffled_indices]
    values = [i * 10 for i in shuffled_indices]  # Values corresponding to original order

    return pd.Series(values, index=shuffled_dates, name='value')


@pytest.fixture
def unsorted_panel():
    """Unsorted panel data for testing sorting functionality."""
    # Create data with mixed entity and date order
    data_tuples = [
        ('B', pd.Timestamp('2020-01-02'), 10),
        ('A', pd.Timestamp('2020-01-01'), 0),
        ('B', pd.Timestamp('2020-01-01'), 5),
        ('A', pd.Timestamp('2020-01-03'), 15),
        ('A', pd.Timestamp('2020-01-02'), 12),
        ('B', pd.Timestamp('2020-01-03'), 20),
    ]

    indices = [(entity, date) for entity, date, _ in data_tuples]
    values = [value for _, _, value in data_tuples]

    idx = pd.MultiIndex.from_tuples(indices, names=['entity', 'date'])
    return pd.DataFrame({'value': values}, index=idx)


@pytest.fixture(params=[
    'simple_timeseries',
    'simple_timeseries_df',
    'numpy_array_data'
])
def various_input_types(request):
    """Parametrized fixture providing various input data types."""
    if request.param == 'simple_timeseries':
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        return pd.Series(range(20), index=dates)
    elif request.param == 'simple_timeseries_df':
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        return pd.DataFrame({'feature': range(20)}, index=dates)
    elif request.param == 'numpy_array_data':
        return np.arange(20).reshape(-1, 1)


@pytest.fixture(params=[
    ('simple_panel', None),
    ('unbalanced_panel', None),
])
def various_panel_types(request):
    """Parametrized fixture providing various panel data types."""
    panel_type, groups = request.param

    if panel_type == 'simple_panel':
        entities = ['A', 'B']
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        X = pd.DataFrame({'feature': range(20)}, index=idx)
        return X, groups
    elif panel_type == 'unbalanced_panel':
        # Create unbalanced panel
        indices = [
            ('A', pd.Timestamp('2020-01-01')),
            ('A', pd.Timestamp('2020-01-02')),
            ('A', pd.Timestamp('2020-01-03')),
            ('B', pd.Timestamp('2020-01-01')),
            ('B', pd.Timestamp('2020-01-02')),
        ]
        idx = pd.MultiIndex.from_tuples(indices, names=['entity', 'date'])
        X = pd.DataFrame({'feature': range(5)}, index=idx)
        return X, groups


class TestHelpers:
    """Helper methods for testing cross-validation functionality."""

    @staticmethod
    def assert_valid_split(train_idx, test_idx, X):
        """Assert that a train/test split is valid."""
        # Check types
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)

        # Check bounds
        n_samples = len(X)
        assert np.all(train_idx >= 0)
        assert np.all(train_idx < n_samples)
        assert np.all(test_idx >= 0)
        assert np.all(test_idx < n_samples)

        # Check no duplicates within each set
        assert len(np.unique(train_idx)) == len(train_idx)
        assert len(np.unique(test_idx)) == len(test_idx)

    @staticmethod
    def assert_temporal_order_preserved(train_idx, test_idx, gap=0):
        """Assert that temporal order is preserved (for out-of-sample)."""
        if len(train_idx) > 0 and len(test_idx) > 0:
            actual_gap = min(test_idx) - max(train_idx) - 1
            assert actual_gap >= gap, f"Gap should be at least {gap}, got {actual_gap}"

    @staticmethod
    def assert_insample_property(train_idx, test_idx):
        """Assert in-sample property (test indices included in training)."""
        assert np.all(np.isin(test_idx, train_idx)), "Test indices should be subset of training indices"


@pytest.fixture
def test_helpers():
    """Provide test helper methods."""
    return TestHelpers


# =============================================================================
# Fixtures de fréquences mixtes (anciennement tests/frequency/conftest.py)
# =============================================================================


# -----------------------------------------------------------------------------
# Jeux coûteux : construits UNE fois par session (``_..._session``), chaque
# fixture publique en renvoie une ``.copy()`` — un test qui mute son jeu (tri
# en place, renommage de colonnes...) ne contamine jamais les tests suivants,
# y compris ceux d'un autre fichier partageant la même session pytest.
# -----------------------------------------------------------------------------


@pytest.fixture(scope='session')
def _mixed_freq_timeseries_session() -> pd.DataFrame:
    """Session-scoped build of :func:`mixed_freq_timeseries` — see its docstring."""
    return build_mixed_frequency_timeseries()


@pytest.fixture
def mixed_freq_timeseries(_mixed_freq_timeseries_session: pd.DataFrame) -> pd.DataFrame:
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
    return _mixed_freq_timeseries_session.copy()


@pytest.fixture(scope='session')
def _panel_two_level_dataset_session() -> pd.DataFrame:
    """Session-scoped build of :func:`panel_two_level_dataset` — see its docstring."""
    return build_panel_two_level()


@pytest.fixture
def panel_two_level_dataset(_panel_two_level_dataset_session: pd.DataFrame) -> pd.DataFrame:
    """Two-level entity panel (country x sector), 2x2 = 4 entities.

    MultiIndex (``country``, ``sector``, ``date``) with 2 countries
    (``France``, ``Allemagne``) x 2 sectors (``Industrie``, ``Services``),
    each with the same 48 month-start (``MS``) dates from 2019-01-01 to
    2022-12-01 (192 rows total). Columns:

    - ``indicateur_mensuel`` (monthly, dense, no NaN): ~100-110.
    - ``indicateur_trimestriel`` (quarterly: non-NaN only at quarter-start
      months 1/4/7/10, NaN elsewhere) — the variable to impute: ~500-575.
    """
    return _panel_two_level_dataset_session.copy()


@pytest.fixture(scope='session')
def _mixed_freq_panel_session() -> pd.DataFrame:
    """Session-scoped build of :func:`mixed_freq_panel` — see its docstring."""
    return build_mixed_frequency_panel()


@pytest.fixture
def mixed_freq_panel(_mixed_freq_panel_session: pd.DataFrame) -> pd.DataFrame:
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
    return _mixed_freq_panel_session.copy()


@pytest.fixture(scope='session')
def _panel_reference_full_session() -> pd.DataFrame:
    """Session-scoped build of :func:`panel_reference_full` — see its docstring."""
    return build_panel_reference()


@pytest.fixture
def panel_reference_full(_panel_reference_full_session: pd.DataFrame) -> pd.DataFrame:
    """Unified ``PANEL-X`` reference dataset (``high_frequency_imputer2_architecture.md`` §2.6).

    The single frame all three frozen reference datasets project from:
    ``TS`` (:func:`reference_timeseries`, §2.2), ``PANEL``
    (:func:`mixed_freq_panel_heterogeneous`, §2.3) and ``PANEL-F``
    (:func:`mixed_freq_panel_multifrequency`, §2.5) are strict column (and,
    for ``TS``, entity) projections of this dataset — the projection status
    changes none of their values, §2.2 / §2.3 / §2.5 stay normative and
    frozen.

    ``MultiIndex`` (``country``, ``date``) with 3 entities (``FR``, ``DE``,
    ``IT``), each sharing the same 36 month-end (``ME``) dates from
    2021-01-31 to 2023-12-31 (108 rows total). Columns ``m1``, ``q1``,
    ``a1``, ``a2``, ``climat_affaires``, ``v`` — see
    :func:`~tests.support.datasets.build_panel_reference` for their
    definitions.
    """
    return _panel_reference_full_session.copy()


@pytest.fixture
def reference_timeseries(_panel_reference_full_session: pd.DataFrame) -> pd.DataFrame:
    """``TS`` reference dataset of ``high_frequency_imputer2_architecture.md`` §2.2.

    Strict projection of ``PANEL-X`` (§2.6, :func:`panel_reference_full`):
    ``panel_x.loc['FR', ['m1', 'q1', 'a1', 'a2']]``. ``DatetimeIndex`` named
    ``date``, month-end anchored (``ME``), 36 rows from 2021-01-31 to
    2023-12-31. Columns:

    - ``m1`` (monthly, dense, never NaN).
    - ``q1`` (quarterly: non-NaN only at quarter-end months 3/6/9/12).
    - ``a1`` (annual: non-NaN only at the three year-end anchors) — gold
      values 120 / 132 / 150.
    - ``a2`` (annual, same anchors) — gold values 60 / 66 / 72.

    The annual gold values match §2.2 verbatim and are reused as gold cases
    by later implementation lots; they must not change.
    """
    df = _panel_reference_full_session.loc['FR', ['m1', 'q1', 'a1', 'a2']].copy()
    # Restauration de la fréquence d'index, perdue au découpage du MultiIndex :
    # la projection reste identique bit à bit à l'ancien constructeur dédié.
    df.index.freq = df.index.inferred_freq
    return df


@pytest.fixture
def mixed_freq_panel_heterogeneous(_panel_reference_full_session: pd.DataFrame) -> pd.DataFrame:
    """``PANEL`` reference dataset of ``high_frequency_imputer2_architecture.md`` §2.3.

    Strict projection of ``PANEL-X`` (§2.6, :func:`panel_reference_full`):
    ``panel_x[['m1', 'q1', 'a1', 'a2', 'climat_affaires']]``. ``MultiIndex``
    (``country``, ``date``) with 3 entities (``FR``, ``DE``, ``IT``), each
    sharing the same 36 month-end (``ME``) dates from 2021-01-31 to
    2023-12-31 (108 rows total). Columns ``m1`` / ``q1`` / ``a1`` / ``a2``
    as in :func:`reference_timeseries`, plus:

    - ``climat_affaires`` (monthly business survey): observed for ``FR`` and
      ``DE`` (level ~100 with reproducible noise), entirely NaN for ``IT``.
      The column exists for every entity; only the Italian entity never
      observes it — the support of ``covariate_eligibility`` (§4.5) and of
      the per-entity NaN invariant (§3).
    """
    return _panel_reference_full_session[['m1', 'q1', 'a1', 'a2', 'climat_affaires']].copy()


@pytest.fixture
def mixed_freq_panel_multifrequency(_panel_reference_full_session: pd.DataFrame) -> pd.DataFrame:
    """``PANEL-F`` reference dataset of ``high_frequency_imputer2_architecture.md`` §2.5.

    Strict projection of ``PANEL-X`` (§2.6, :func:`panel_reference_full`):
    ``panel_x[['m1', 'q1', 'v']]``. ``MultiIndex`` (``country``, ``date``)
    with 3 entities (``FR``, ``DE``, ``IT``), each sharing the same 36
    month-end (``ME``) dates as :func:`mixed_freq_panel_heterogeneous` (108
    rows total). Columns ``m1`` / ``q1`` as in :func:`reference_timeseries`,
    identical across the three entities, plus:

    - ``v`` (the heterogeneous-frequency column): observed annually for
      ``FR`` (3 year-end anchors), quarterly for ``DE`` (12 quarter-end
      anchors), monthly for ``IT`` (36 observations, constant per year).
      Chosen so the three entities carry the same annual total (120.0 /
      132.0 / 150.0) — the support of inter-entity mutualisation of the
      training set (§5.8) and of the ``B29`` defect it measures (§1.5).

    No ``climat_affaires`` column: this dataset is not ``PANEL`` and does
    not replace it.
    """
    return _panel_reference_full_session[['m1', 'q1', 'v']].copy()


@pytest.fixture(scope='session')
def _irregular_index_timeseries_session() -> pd.DataFrame:
    """Session-scoped build of :func:`irregular_index_timeseries` — see its docstring."""
    return build_mixed_frequency_timeseries(annual_start_date='2015-01-01')


@pytest.fixture
def irregular_index_timeseries(_irregular_index_timeseries_session: pd.DataFrame) -> pd.DataFrame:
    """Realistic mixed-frequency time series of notebook 3.

    ``build_mixed_frequency_timeseries(annual_start_date='2015-01-01')`` — same schema as
    :func:`mixed_freq_timeseries`, with ``balance_commerciale_annuelle``'s
    history starting in 2015, three years before the monthly grid
    (2018-01-01). The union with the monthly grid introduces isolated annual
    anchors before its start, and the resulting index is genuinely
    irregular (:func:`tsforecast.frequency.is_regular` is ``False``) — not
    only a regular grid dotted with NaN, unlike every other fixture in this
    module.
    """
    return _irregular_index_timeseries_session.copy()


@pytest.fixture(scope='session')
def _heterogeneous_coverage_panel_session() -> pd.DataFrame:
    """Session-scoped build of :func:`heterogeneous_coverage_panel` — see its docstring."""
    return build_mixed_frequency_panel(countries=HETEROGENEOUS_PANEL_COUNTRIES)


@pytest.fixture
def heterogeneous_coverage_panel(_heterogeneous_coverage_panel_session: pd.DataFrame) -> pd.DataFrame:
    """Realistic heterogeneous mixed-frequency panel of notebook 3.

    ``build_mixed_frequency_panel(countries=HETEROGENEOUS_PANEL_COUNTRIES)`` — 3 entities
    (``France``, ``Allemagne``, ``Italie``), each with its own monthly-grid
    coverage (France 2018-01 to 2024-07, Allemagne 2018-07 to 2024-04, Italie
    2019-01 to 2024-07) rather than one common period truncated per entity.
    Adds two columns absent from :func:`mixed_freq_panel`:

    - ``depenses_publiques_pib``: publication frequency differs by entity
      (annual for France and Italie, quarterly for Allemagne), last
      publication per entity NaN.
    - ``climat_affaires``: observed for France and Allemagne, structurally
      absent (column present, zero observation) for Italie.

    Each entity's ``balance_commerciale_annuelle`` history also starts
    earlier than its monthly grid, so every entity's index is individually
    irregular (see :func:`irregular_index_timeseries`).
    """
    return _heterogeneous_coverage_panel_session.copy()
