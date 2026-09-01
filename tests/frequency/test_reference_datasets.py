"""Tests des jeux de référence de ``high_frequency_imputer2_architecture.md``.

Le jeu ``PANEL`` (§2.3) et le jeu ``TS`` (§2.2) servent de support à tous les
tests et notebooks de ``HighFrequencyImputer2``. Leur structure et leurs valeurs
d'or sont verrouillées ici : elles ne doivent plus bouger une fois ce lot livré.
"""
# Manipulation de données
import pandas as pd


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
