"""Tests des jeux de référence de ``high_frequency_imputer2_architecture.md``.

Le jeu ``PANEL`` (§2.3) et le jeu ``TS`` (§2.2) servent de support à tous les
tests et notebooks de ``HighFrequencyImputer2``. Leur structure et leurs valeurs
d'or sont verrouillées ici : elles ne doivent plus bouger une fois ce lot livré.
"""
# Manipulation de données
import pandas as pd

# Détecteur de fréquence utilisé par HighFrequencyImputer2
from tsforecast.utils.frequency.utils import detect_frequency


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
