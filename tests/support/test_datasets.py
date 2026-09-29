"""Tests of the reference datasets of ``high_frequency_imputer2_architecture.md``.

The ``PANEL`` (§2.3) and ``TS`` (§2.2) datasets support every test and
notebook of ``HighFrequencyImputer``. Their structure and golden values
are locked here: they must not move anymore once this batch is delivered.

Since batch L13-0b (§2.6), the three datasets are PROJECTIONS of a
single ``PANEL-X`` dataset (fixture ``panel_reference_full``). The
projection identities are tested by ``TestPanelXProjections``: they
guarantee that the factorization does not drift.
"""
# Manipulation de données
import pandas as pd
import pytest

# Détecteur de fréquence utilisé par HighFrequencyImputer
from tsforecast.utils.frequency.utils import detect_frequency
from tsforecast.frequency import is_regular
from tests.support.datasets import (
    HETEROGENEOUS_PANEL_COUNTRIES,
    build_mixed_frequency_panel,
    build_mixed_frequency_timeseries,
)


class TestHeterogeneousPanel:
    """``PANEL`` dataset: covariate structurally absent for one entity (§2.3, §4.5)."""

    def test_heterogeneous_panel_it_has_no_climat_affaires(
        self, mixed_freq_panel_heterogeneous: pd.DataFrame
    ) -> None:
        """``climat_affaires`` exists for every entity but IT never observes it."""
        df = mixed_freq_panel_heterogeneous

        # La colonne appartient au schéma, pour les trois entités.
        assert 'climat_affaires' in df.columns

        # ``count`` exclut les NaN : décompte des observations réelles par entité.
        n_obs = df.groupby(level='country')['climat_affaires'].count()

        assert n_obs['IT'] == 0
        assert n_obs['FR'] == 36
        assert n_obs['DE'] == 36


class TestReferenceTimeseries:
    """``TS`` dataset: annual golden values of the document (§2.2)."""

    def test_reference_timeseries_matches_spec_anchors(
        self, reference_timeseries: pd.DataFrame
    ) -> None:
        """The six golden values of ``a1`` and ``a2`` are those of §2.2, at the three anchors."""
        df = reference_timeseries
        anchors = pd.to_datetime(['2021-12-31', '2022-12-31', '2023-12-31'])

        assert df.loc[anchors, 'a1'].tolist() == [120.0, 132.0, 150.0]
        assert df.loc[anchors, 'a2'].tolist() == [60.0, 66.0, 72.0]


class TestMultiFrequencyPanel:
    """``PANEL-F`` dataset: one column with three detected frequencies across entities (§2.5, §5.8)."""

    def test_shape_and_index(self, mixed_freq_panel_multifrequency: pd.DataFrame) -> None:
        """108 rows, sorted ``MultiIndex`` (``country``, ``date``), 36 dates per entity."""
        df = mixed_freq_panel_multifrequency

        assert df.shape[0] == 108
        assert list(df.index.names) == ['country', 'date']
        assert df.index.is_monotonic_increasing

        n_dates = df.groupby(level='country').size()
        assert (n_dates == 36).all()

    def test_v_observation_counts_per_entity(
        self, mixed_freq_panel_multifrequency: pd.DataFrame
    ) -> None:
        """``v`` is observed 3 times for FR, 12 times for DE, 36 times for IT."""
        df = mixed_freq_panel_multifrequency

        # ``count`` exclut les NaN : décompte des observations réelles par entité.
        n_obs = df.groupby(level='country')['v'].count()

        assert n_obs['FR'] == 3
        assert n_obs['DE'] == 12
        assert n_obs['IT'] == 36

    def test_v_gold_values(self, mixed_freq_panel_multifrequency: pd.DataFrame) -> None:
        """The golden values of ``v`` of §2.5, copied as is per entity."""
        df = mixed_freq_panel_multifrequency

        assert df.loc['FR', 'v'].dropna().tolist() == [120.0, 132.0, 150.0]
        assert df.loc['DE', 'v'].dropna().tolist() == [
            28.0, 30.0, 31.0, 31.0,
            31.0, 33.0, 34.0, 34.0,
            36.0, 37.0, 38.0, 39.0,
        ]
        assert df.loc['IT', 'v'].dropna().tolist() == [10.0] * 12 + [11.0] * 12 + [12.5] * 12

    def test_annual_totals_agree_across_entities(
        self, mixed_freq_panel_multifrequency: pd.DataFrame
    ) -> None:
        """The annual sum of ``v`` is 120 / 132 / 150 for each of the three entities."""
        df = mixed_freq_panel_multifrequency

        for entity in ('FR', 'DE', 'IT'):
            v = df.loc[entity, 'v']
            annual_totals = v.groupby(v.index.year).sum()
            assert annual_totals.tolist() == [120.0, 132.0, 150.0]

    def test_italian_quarterly_aggregates(
        self, mixed_freq_panel_multifrequency: pd.DataFrame
    ) -> None:
        """The quarterly aggregation (sum) of ``v`` for IT is 30 / 33 / 37.5, x4 per year."""
        df = mixed_freq_panel_multifrequency

        v_it = df.loc['IT', 'v']
        quarterly_totals = v_it.resample('QE').sum()

        assert quarterly_totals.tolist() == [30.0] * 4 + [33.0] * 4 + [37.5] * 4

    def test_m1_and_q1_match_the_ts_reference(
        self,
        mixed_freq_panel_multifrequency: pd.DataFrame,
        reference_timeseries: pd.DataFrame,
    ) -> None:
        """``m1`` and ``q1`` of each entity are exactly those of the ``TS`` dataset."""
        df = mixed_freq_panel_multifrequency

        for entity in ('FR', 'DE', 'IT'):
            pd.testing.assert_series_equal(
                df.loc[entity, 'm1'], reference_timeseries['m1'], check_freq=False
            )
            pd.testing.assert_series_equal(
                df.loc[entity, 'q1'], reference_timeseries['q1'], check_freq=False
            )

    def test_detected_frequencies_disagree_across_entities(
        self, mixed_freq_panel_multifrequency: pd.DataFrame
    ) -> None:
        """``v`` carries three different detected frequencies depending on the entity (§2.1)."""
        df = mixed_freq_panel_multifrequency

        detected = detect_frequency(data=df)

        assert detected[('FR', 'v')] == 'Y'
        assert detected[('DE', 'v')] == 'Q'
        assert detected[('IT', 'v')] == 'M'

        for entity in ('FR', 'DE', 'IT'):
            assert detected[(entity, 'm1')] == 'M'
            assert detected[(entity, 'q1')] == 'Q'


class TestPanelXProjections:
    """§2.6 - ``TS``, ``PANEL`` and ``PANEL-F`` are projections of ``PANEL-X``."""

    def test_reference_timeseries_is_the_fr_projection(
        self,
        reference_timeseries: pd.DataFrame,
        panel_reference_full: pd.DataFrame,
    ) -> None:
        """``TS`` == ``PANEL-X.loc['FR', ['m1', 'q1', 'a1', 'a2']]``, bit for bit."""
        projection = panel_reference_full.loc['FR', ['m1', 'q1', 'a1', 'a2']]

        # ``check_freq=False`` : le découpage d'un MultiIndex perd l'attribut
        # ``freq`` de l'index ; la fixture ``reference_timeseries`` le restaure
        # pour rester un remplacement exact de l'ancien constructeur dédié.
        pd.testing.assert_frame_equal(
            reference_timeseries, projection, check_freq=False
        )

    def test_heterogeneous_panel_is_the_column_projection(
        self,
        mixed_freq_panel_heterogeneous: pd.DataFrame,
        panel_reference_full: pd.DataFrame,
    ) -> None:
        """``PANEL`` == ``PANEL-X[['m1', 'q1', 'a1', 'a2', 'climat_affaires']]``."""
        projection = panel_reference_full[
            ['m1', 'q1', 'a1', 'a2', 'climat_affaires']
        ]

        pd.testing.assert_frame_equal(mixed_freq_panel_heterogeneous, projection)

    def test_multifrequency_panel_is_the_column_projection(
        self,
        mixed_freq_panel_multifrequency: pd.DataFrame,
        panel_reference_full: pd.DataFrame,
    ) -> None:
        """``PANEL-F`` == ``PANEL-X[['m1', 'q1', 'v']]``."""
        projection = panel_reference_full[['m1', 'q1', 'v']]

        pd.testing.assert_frame_equal(mixed_freq_panel_multifrequency, projection)

    def test_projections_share_one_index(
        self,
        panel_reference_full: pd.DataFrame,
        mixed_freq_panel_heterogeneous: pd.DataFrame,
        mixed_freq_panel_multifrequency: pd.DataFrame,
    ) -> None:
        """Both panel projections carry the index of ``PANEL-X``, identically."""
        assert mixed_freq_panel_heterogeneous.index.equals(
            panel_reference_full.index
        )
        assert mixed_freq_panel_multifrequency.index.equals(
            panel_reference_full.index
        )


class TestRealisticMixedFrequencyDatasets:
    """Realistic dataset of notebook 3 (§2.4 of ``tests_and_refactoring_prompts.md``).

    ``irregular_index_timeseries`` / ``heterogeneous_coverage_panel`` are
    not dedicated builders: they are specific calls of
    :func:`build_mixed_frequency_timeseries` and
    :func:`build_mixed_frequency_panel`, generalized to reproduce
    ``create_timeseries_dataset`` / ``create_panel_dataset`` of the
    notebook ``notebooks/3 - QB - Panel a frequences mixtes heterogene.ipynb``
    (cells 5 and 7). Each test below checks a feature announced by this
    notebook (cells 0, 6, 8, 10, 11), exercised through the fixtures.
    """

    # ----- Couverture propre à chaque entité (cellule 6, notebook 3) -----

    def test_panel_entities_have_their_own_monthly_grid_coverage(
        self, heterogeneous_coverage_panel: pd.DataFrame
    ) -> None:
        """Each entity covers its own monthly grid (§2.2 of the notebook)."""
        df = heterogeneous_coverage_panel

        # Valeurs d'or : dates de couverture mensuelle du dictionnaire
        # ``countries`` de ``create_panel_dataset`` (cellule 7 du notebook),
        # recopiées telles quelles (et non relues depuis
        # ``HETEROGENEOUS_PANEL_COUNTRIES``, pour ne pas rendre le test
        # tautologique avec sa propre source).
        expected_grids = {
            'France': pd.date_range('2018-01-01', '2024-07-01', freq='MS'),
            'Allemagne': pd.date_range('2018-07-01', '2024-04-01', freq='MS'),
            'Italie': pd.date_range('2019-01-01', '2024-07-01', freq='MS'),
        }

        for entity, monthly_grid in expected_grids.items():
            entity_index = df.loc[entity].index
            # La grille mensuelle propre à l'entité est incluse dans son index
            # (qui porte en plus les ancres annuelles hors grille, testées à
            # part) : c'est elle qui porte la couverture temporelle annoncée.
            assert monthly_grid.isin(entity_index).all()
            monthly_positions = entity_index[entity_index.isin(monthly_grid)]
            assert monthly_positions.min() == monthly_grid.min()
            assert entity_index.max() == monthly_grid.max()

    def test_timeseries_and_each_entity_index_is_irregular(
        self, irregular_index_timeseries: pd.DataFrame, heterogeneous_coverage_panel: pd.DataFrame
    ) -> None:
        """The annual history earlier than the monthly grid makes the index irregular.

        CLAUDE.md edge case "irregular frequencies": unlike the other datasets
        of the suite (regular grid dotted with NaN),
        ``balance_commerciale_annuelle`` introduces annual anchors genuinely
        outside the grid - ``is_regular`` must detect it, for the series alone
        as for each entity of the panel.
        """
        assert is_regular(irregular_index_timeseries) is False

        for entity in ('France', 'Allemagne', 'Italie'):
            assert is_regular(heterogeneous_coverage_panel.loc[entity]) is False

    # ----- depenses_publiques_pib : fréquence de publication par entité (cellule 9) -----

    def test_depenses_publiques_pib_publication_frequency_per_entity(
        self, heterogeneous_coverage_panel: pd.DataFrame
    ) -> None:
        """Annual publication for France / Italie, quarterly for Allemagne, last value NaN."""
        df = heterogeneous_coverage_panel

        annual_entities = {'France': 6, 'Italie': 5}
        for entity, n_observations in annual_entities.items():
            serie = df.loc[entity, 'depenses_publiques_pib'].dropna()
            assert len(serie) == n_observations
            # Valeur d'or : écart d'environ un an entre publications.
            gaps_days = (serie.index[1:] - serie.index[:-1]).days
            assert (gaps_days >= 360).all()

        serie_de = df.loc['Allemagne', 'depenses_publiques_pib'].dropna()
        assert len(serie_de) == 23
        gaps_days_de = (serie_de.index[1:] - serie_de.index[:-1]).days
        assert (gaps_days_de < 100).all()

        # Dernière valeur retirée (délai de publication simulé) : la dernière
        # date de publication candidate (mois de janvier pour une fréquence
        # annuelle, mois 1/4/7/10 pour une fréquence trimestrielle) au sein de
        # la grille mensuelle de l'entité reste NaN, alors qu'une publication
        # aurait dû y figurer.
        expected_last_candidate = {
            'France': pd.Timestamp('2024-01-01'),
            'Allemagne': pd.Timestamp('2024-04-01'),
            'Italie': pd.Timestamp('2024-01-01'),
        }
        for entity, last_candidate in expected_last_candidate.items():
            assert pd.isna(df.loc[(entity, last_candidate), 'depenses_publiques_pib'])

    # ----- climat_affaires : structurellement absente pour l'Italie (cellules 10-11) -----

    def test_climat_affaires_structurally_absent_for_italy(
        self, heterogeneous_coverage_panel: pd.DataFrame
    ) -> None:
        """Column present for the three entities, zero observation for Italie."""
        df = heterogeneous_coverage_panel

        assert 'climat_affaires' in df.columns

        n_obs = df.groupby(level=0)['climat_affaires'].count()
        assert n_obs['Italie'] == 0
        assert n_obs['France'] == 79  # taille de la grille mensuelle de la France
        assert n_obs['Allemagne'] == 70  # taille de la grille mensuelle de l'Allemagne

    # ----- Délais : dernière valeur NaN pour inflation_ipc et taux_chomage -----

    def test_last_row_of_inflation_and_chomage_is_nan(
        self, irregular_index_timeseries: pd.DataFrame, heterogeneous_coverage_panel: pd.DataFrame
    ) -> None:
        """Simulated one-month publication delay: last observation removed."""
        assert pd.isna(irregular_index_timeseries['inflation_ipc'].iloc[-1])
        assert pd.isna(irregular_index_timeseries['taux_chomage'].iloc[-1])

        for entity in ('France', 'Allemagne', 'Italie'):
            df_entity = heterogeneous_coverage_panel.loc[entity]
            assert pd.isna(df_entity['inflation_ipc'].iloc[-1])
            assert pd.isna(df_entity['taux_chomage'].iloc[-1])

    # ----- Reproductibilité -----

    def test_irregular_index_timeseries_is_reproducible(self, irregular_index_timeseries: pd.DataFrame) -> None:
        """Two calls with the same arguments return a bit-identical dataset."""
        rebuilt = build_mixed_frequency_timeseries(annual_start_date='2015-01-01')
        pd.testing.assert_frame_equal(irregular_index_timeseries, rebuilt)

    def test_heterogeneous_coverage_panel_is_reproducible(self, heterogeneous_coverage_panel: pd.DataFrame) -> None:
        """Two calls with the same arguments return a bit-identical dataset."""
        rebuilt = build_mixed_frequency_panel(countries=HETEROGENEOUS_PANEL_COUNTRIES)
        pd.testing.assert_frame_equal(heterogeneous_coverage_panel, rebuilt)

    def test_irregular_index_timeseries_gold_values(self, irregular_index_timeseries: pd.DataFrame) -> None:
        """Four golden values read once on the built dataset, then hard-coded."""
        df = irregular_index_timeseries

        assert df.loc['2019-01-01', 'production_industrielle'] == pytest.approx(102.67063571504136)
        assert pd.isna(df.loc['2018-12-01', 'production_industrielle'])
        assert df.loc['2018-01-01', 'inflation_ipc'] == pytest.approx(0.6037293256197321)
        assert df.loc['2015-01-01', 'balance_commerciale_annuelle'] == pytest.approx(-35.2628407569658)

    def test_heterogeneous_coverage_panel_gold_values(self, heterogeneous_coverage_panel: pd.DataFrame) -> None:
        """Three golden values read once on the built dataset, then hard-coded."""
        df = heterogeneous_coverage_panel

        assert df.loc[('France', '2018-01-01'), 'inflation_ipc'] == pytest.approx(1.8353430756890152)
        assert df.loc[('Allemagne', '2018-07-01'), 'climat_affaires'] == pytest.approx(99.96786883370062)
        assert df.loc[('Italie', '2019-01-01'), 'depenses_publiques_pib'] == pytest.approx(49.53371034228826)

    # ----- Régression : les valeurs par défaut restent le jeu historique régulier -----

    def test_default_build_mixed_frequency_timeseries_stays_regular(self) -> None:
        """Without ``annual_start_date``, the index stays regular (non-contaminating generalization)."""
        df = build_mixed_frequency_timeseries()
        assert is_regular(df) is True

    def test_default_build_mixed_frequency_panel_stays_regular_and_without_heterogeneous_columns(self) -> None:
        """Without ``countries``, the panel stays regular and without the columns specific to notebook 3."""
        df = build_mixed_frequency_panel()

        assert 'depenses_publiques_pib' not in df.columns
        assert 'climat_affaires' not in df.columns

        for entity in df.index.get_level_values('country').unique():
            assert is_regular(df.loc[entity]) is True
