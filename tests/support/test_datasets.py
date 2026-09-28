"""Tests des jeux de référence de ``high_frequency_imputer2_architecture.md``.

Le jeu ``PANEL`` (§2.3) et le jeu ``TS`` (§2.2) servent de support à tous les
tests et notebooks de ``HighFrequencyImputer``. Leur structure et leurs valeurs
d'or sont verrouillées ici : elles ne doivent plus bouger une fois ce lot livré.

Depuis le lot L13-0b (§2.6), les trois jeux sont des PROJECTIONS d'un jeu unique
``PANEL-X`` (fixture ``panel_reference_full``). Les identités de projection sont
testées par ``TestPanelXProjections`` : elles garantissent que la factorisation
ne dérive pas.
"""
# Manipulation de données
import pandas as pd
import pytest

# Détecteur de fréquence utilisé par HighFrequencyImputer
from tsforecast.utils.frequency.utils import detect_frequency
from tsforecast.frequency import is_regular
from tests.support.datasets import (
    HETEROGENEOUS_PANEL_COUNTRIES,
    build_panel_nb2,
    build_timeseries_nb2,
)


class TestHeterogeneousPanel:
    """Jeu ``PANEL`` : covariable structurellement absente pour une entité (§2.3, §4.5)."""

    def test_heterogeneous_panel_it_has_no_climat_affaires(
        self, mixed_freq_panel_heterogeneous: pd.DataFrame
    ) -> None:
        """``climat_affaires`` existe pour toutes les entités mais IT ne l'observe jamais."""
        df = mixed_freq_panel_heterogeneous

        # La colonne appartient au schéma, pour les trois entités.
        assert 'climat_affaires' in df.columns

        # ``count`` exclut les NaN : décompte des observations réelles par entité.
        n_obs = df.groupby(level='country')['climat_affaires'].count()

        assert n_obs['IT'] == 0
        assert n_obs['FR'] == 36
        assert n_obs['DE'] == 36


class TestReferenceTimeseries:
    """Jeu ``TS`` : valeurs d'or annuelles du document (§2.2)."""

    def test_reference_timeseries_matches_spec_anchors(
        self, reference_timeseries: pd.DataFrame
    ) -> None:
        """Les six valeurs d'or de ``a1`` et ``a2`` sont celles du §2.2, aux trois ancres."""
        df = reference_timeseries
        anchors = pd.to_datetime(['2021-12-31', '2022-12-31', '2023-12-31'])

        assert df.loc[anchors, 'a1'].tolist() == [120.0, 132.0, 150.0]
        assert df.loc[anchors, 'a2'].tolist() == [60.0, 66.0, 72.0]


class TestMultiFrequencyPanel:
    """Jeu ``PANEL-F`` : une même colonne à trois fréquences détectées par entité (§2.5, §5.8)."""

    def test_shape_and_index(self, mixed_freq_panel_multifrequency: pd.DataFrame) -> None:
        """108 lignes, MultiIndex (``country``, ``date``) trié, 36 dates par entité."""
        df = mixed_freq_panel_multifrequency

        assert df.shape[0] == 108
        assert list(df.index.names) == ['country', 'date']
        assert df.index.is_monotonic_increasing

        n_dates = df.groupby(level='country').size()
        assert (n_dates == 36).all()

    def test_v_observation_counts_per_entity(
        self, mixed_freq_panel_multifrequency: pd.DataFrame
    ) -> None:
        """``v`` est observée 3 fois pour FR, 12 fois pour DE, 36 fois pour IT."""
        df = mixed_freq_panel_multifrequency

        # ``count`` exclut les NaN : décompte des observations réelles par entité.
        n_obs = df.groupby(level='country')['v'].count()

        assert n_obs['FR'] == 3
        assert n_obs['DE'] == 12
        assert n_obs['IT'] == 36

    def test_v_gold_values(self, mixed_freq_panel_multifrequency: pd.DataFrame) -> None:
        """Les valeurs d'or de ``v`` du §2.5, recopiées telles quelles par entité."""
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
        """La somme annuelle de ``v`` vaut 120 / 132 / 150 pour chacune des trois entités."""
        df = mixed_freq_panel_multifrequency

        for entity in ('FR', 'DE', 'IT'):
            v = df.loc[entity, 'v']
            annual_totals = v.groupby(v.index.year).sum()
            assert annual_totals.tolist() == [120.0, 132.0, 150.0]

    def test_italian_quarterly_aggregates(
        self, mixed_freq_panel_multifrequency: pd.DataFrame
    ) -> None:
        """L'agrégation trimestrielle (somme) de ``v`` pour IT vaut 30 / 33 / 37.5, ×4 par an."""
        df = mixed_freq_panel_multifrequency

        v_it = df.loc['IT', 'v']
        quarterly_totals = v_it.resample('QE').sum()

        assert quarterly_totals.tolist() == [30.0] * 4 + [33.0] * 4 + [37.5] * 4

    def test_m1_and_q1_match_the_ts_reference(
        self,
        mixed_freq_panel_multifrequency: pd.DataFrame,
        reference_timeseries: pd.DataFrame,
    ) -> None:
        """``m1`` et ``q1`` de chaque entité sont exactement celles du jeu ``TS``."""
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
        """``v`` porte trois fréquences détectées différentes selon l'entité (§2.1)."""
        df = mixed_freq_panel_multifrequency

        detected = detect_frequency(data=df)

        assert detected[('FR', 'v')] == 'Y'
        assert detected[('DE', 'v')] == 'Q'
        assert detected[('IT', 'v')] == 'M'

        for entity in ('FR', 'DE', 'IT'):
            assert detected[(entity, 'm1')] == 'M'
            assert detected[(entity, 'q1')] == 'Q'


class TestPanelXProjections:
    """§2.6 — ``TS``, ``PANEL`` et ``PANEL-F`` sont des projections de ``PANEL-X``."""

    def test_reference_timeseries_is_the_fr_projection(
        self,
        reference_timeseries: pd.DataFrame,
        panel_reference_full: pd.DataFrame,
    ) -> None:
        """``TS`` == ``PANEL-X.loc['FR', ['m1', 'q1', 'a1', 'a2']]``, au bit près."""
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
        """Les deux projections panel portent l'index de ``PANEL-X``, à l'identique."""
        assert mixed_freq_panel_heterogeneous.index.equals(
            panel_reference_full.index
        )
        assert mixed_freq_panel_multifrequency.index.equals(
            panel_reference_full.index
        )


class TestNotebook3Datasets:
    """Jeu réaliste du notebook 3 (§2.4 de ``tests_and_refactoring_prompts.md``).

    ``nb3_timeseries`` / ``nb3_panel`` ne sont pas des constructeurs dédiés :
    ce sont des appels particuliers de :func:`build_timeseries_nb2` et
    :func:`build_panel_nb2`, généralisés pour reproduire
    ``create_timeseries_dataset`` / ``create_panel_dataset`` du notebook
    ``notebooks/3 - QB - Panel a frequences mixtes heterogene.ipynb`` (cellules
    5 et 7). Chaque test ci-dessous vérifie une caractéristique annoncée par ce
    notebook (cellules 0, 6, 8, 10, 11), exercée via les fixtures.
    """

    # ----- Couverture propre à chaque entité (cellule 6, notebook 3) -----

    def test_panel_entities_have_their_own_monthly_grid_coverage(
        self, nb3_panel: pd.DataFrame
    ) -> None:
        """Chaque entité couvre sa propre grille mensuelle (§2.2 du notebook)."""
        df = nb3_panel

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
        self, nb3_timeseries: pd.DataFrame, nb3_panel: pd.DataFrame
    ) -> None:
        """L'historique annuel antérieur à la grille mensuelle rend l'index irrégulier.

        Cas limite CLAUDE.md « fréquences irrégulières » : contrairement aux
        autres jeux de la suite (grille régulière ponctuée de NaN),
        ``balance_commerciale_annuelle`` introduit ici des ancres annuelles
        réellement hors grille — ``is_regular`` doit le détecter, pour la
        série seule comme pour chaque entité du panel.
        """
        assert is_regular(nb3_timeseries) is False

        for entity in ('France', 'Allemagne', 'Italie'):
            assert is_regular(nb3_panel.loc[entity]) is False

    # ----- depenses_publiques_pib : fréquence de publication par entité (cellule 9) -----

    def test_depenses_publiques_pib_publication_frequency_per_entity(
        self, nb3_panel: pd.DataFrame
    ) -> None:
        """Publication annuelle pour France/Italie, trimestrielle pour Allemagne, dernière valeur NaN."""
        df = nb3_panel

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
        self, nb3_panel: pd.DataFrame
    ) -> None:
        """Colonne présente pour les trois entités, zéro observation pour l'Italie."""
        df = nb3_panel

        assert 'climat_affaires' in df.columns

        n_obs = df.groupby(level=0)['climat_affaires'].count()
        assert n_obs['Italie'] == 0
        assert n_obs['France'] == 79  # taille de la grille mensuelle de la France
        assert n_obs['Allemagne'] == 70  # taille de la grille mensuelle de l'Allemagne

    # ----- Délais : dernière valeur NaN pour inflation_ipc et taux_chomage -----

    def test_last_row_of_inflation_and_chomage_is_nan(
        self, nb3_timeseries: pd.DataFrame, nb3_panel: pd.DataFrame
    ) -> None:
        """Délai de publication d'un mois simulé : dernière observation retirée."""
        assert pd.isna(nb3_timeseries['inflation_ipc'].iloc[-1])
        assert pd.isna(nb3_timeseries['taux_chomage'].iloc[-1])

        for entity in ('France', 'Allemagne', 'Italie'):
            df_entity = nb3_panel.loc[entity]
            assert pd.isna(df_entity['inflation_ipc'].iloc[-1])
            assert pd.isna(df_entity['taux_chomage'].iloc[-1])

    # ----- Reproductibilité -----

    def test_nb3_timeseries_is_reproducible(self, nb3_timeseries: pd.DataFrame) -> None:
        """Deux appels avec les mêmes arguments rendent un jeu bit-identique."""
        rebuilt = build_timeseries_nb2(annual_start_date='2015-01-01')
        pd.testing.assert_frame_equal(nb3_timeseries, rebuilt)

    def test_nb3_panel_is_reproducible(self, nb3_panel: pd.DataFrame) -> None:
        """Deux appels avec les mêmes arguments rendent un jeu bit-identique."""
        rebuilt = build_panel_nb2(countries=HETEROGENEOUS_PANEL_COUNTRIES)
        pd.testing.assert_frame_equal(nb3_panel, rebuilt)

    def test_nb3_timeseries_gold_values(self, nb3_timeseries: pd.DataFrame) -> None:
        """Quatre valeurs d'or relevées une fois sur le jeu construit, puis écrites en dur."""
        df = nb3_timeseries

        assert df.loc['2019-01-01', 'production_industrielle'] == pytest.approx(102.67063571504136)
        assert pd.isna(df.loc['2018-12-01', 'production_industrielle'])
        assert df.loc['2018-01-01', 'inflation_ipc'] == pytest.approx(0.6037293256197321)
        assert df.loc['2015-01-01', 'balance_commerciale_annuelle'] == pytest.approx(-35.2628407569658)

    def test_nb3_panel_gold_values(self, nb3_panel: pd.DataFrame) -> None:
        """Trois valeurs d'or relevées une fois sur le jeu construit, puis écrites en dur."""
        df = nb3_panel

        assert df.loc[('France', '2018-01-01'), 'inflation_ipc'] == pytest.approx(1.8353430756890152)
        assert df.loc[('Allemagne', '2018-07-01'), 'climat_affaires'] == pytest.approx(99.96786883370062)
        assert df.loc[('Italie', '2019-01-01'), 'depenses_publiques_pib'] == pytest.approx(49.53371034228826)

    # ----- Régression : les valeurs par défaut restent le jeu historique régulier -----

    def test_default_build_timeseries_nb2_stays_regular(self) -> None:
        """Sans ``annual_start_date``, l'index reste régulier (généralisation non contaminante)."""
        df = build_timeseries_nb2()
        assert is_regular(df) is True

    def test_default_build_panel_nb2_stays_regular_and_without_nb3_columns(self) -> None:
        """Sans ``countries``, le panel reste régulier et sans les colonnes propres au notebook 3."""
        df = build_panel_nb2()

        assert 'depenses_publiques_pib' not in df.columns
        assert 'climat_affaires' not in df.columns

        for entity in df.index.get_level_values('country').unique():
            assert is_regular(df.loc[entity]) is True
