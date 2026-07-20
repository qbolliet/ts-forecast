"""Tests unitaires pour le FrequencyAligner.

Ce module épingle les contrats d'index du FrequencyAligner, dont dépend le
HighFrequencyImputer :
- aggregate_to_target préserve l'index dense d'origine (valeurs agrégées
  réindexées, NaN hors bornes de période, NaN pour les périodes incomplètes) ;
- interpolate_to_target densifie l'index (union avec les dates interpolées,
  restreinte à la plage d'origine) ;
- convert_to_target oriente chaque clé vers l'agrégation ou l'interpolation
  selon la fréquence source observée de la variable.
"""
import numpy as np
import pandas as pd
import pytest

from tsforecast.frequency.frequency_aligner import FrequencyAligner


@pytest.fixture
def aligner():
    """Fixture pour créer une instance de FrequencyAligner."""
    return FrequencyAligner()


class TestAggregateToTarget:
    """Tests du contrat « même index, NaN hors bornes » de aggregate_to_target."""

    @pytest.fixture
    def monthly_df(self):
        """DataFrame mensuel à deux colonnes : 2 trimestres complets + 1 partiel."""
        dates = pd.date_range('2023-01-31', periods=8, freq='ME')
        return pd.DataFrame(
            {
                'x': np.ones(len(dates)),
                'y': np.arange(len(dates), dtype=float),
            },
            index=dates,
        )

    def test_preserves_original_index(self, aligner, monthly_df):
        """L'index de sortie reste l'index dense d'origine (hors lignes tout-NaN)."""
        result = aligner.aggregate_to_target(monthly_df, ['x'], 'QE', is_panel=False)

        # La colonne 'y' n'étant pas agrégée, aucune ligne n'est tout-NaN :
        # l'index d'origine est intégralement préservé
        pd.testing.assert_index_equal(result.index, monthly_df.index)

    def test_aggregated_values_on_period_boundaries_only(self, aligner, monthly_df):
        """Les valeurs agrégées ne sont portées que par les fins de période complètes."""
        result = aligner.aggregate_to_target(monthly_df, ['x'], 'QE', is_panel=False)

        # Somme des mois sur les trimestres complets (3 mois chacun)
        assert result.loc[pd.Timestamp('2023-03-31'), 'x'] == 3.0
        assert result.loc[pd.Timestamp('2023-06-30'), 'x'] == 3.0

        # NaN partout ailleurs (mois intermédiaires et trimestre incomplet T3)
        boundaries = pd.DatetimeIndex(['2023-03-31', '2023-06-30'])
        assert result.loc[result.index.difference(boundaries), 'x'].isna().all()

        # La colonne non agrégée est inchangée
        pd.testing.assert_series_equal(result['y'], monthly_df['y'])

    def test_panel_same_index_contract(self, aligner):
        """En panel, le contrat d'index est respecté entité par entité."""
        dates = pd.date_range('2023-01-31', periods=6, freq='ME')
        index = pd.MultiIndex.from_product(
            [['A', 'B'], dates], names=['entity', 'date']
        )
        df = pd.DataFrame(
            {
                'x': np.ones(len(index)),
                'y': np.arange(len(index), dtype=float),
            },
            index=index,
        )

        result = aligner.aggregate_to_target(
            df, [('A', 'x'), ('B', 'x')], 'QE', is_panel=True
        )

        # Index préservé (aucune ligne tout-NaN grâce à 'y')
        pd.testing.assert_index_equal(result.index, df.index)

        # Valeurs agrégées aux fins de trimestre complètes, pour chaque entité
        for entity in ['A', 'B']:
            assert result.loc[(entity, pd.Timestamp('2023-03-31')), 'x'] == 3.0
            assert result.loc[(entity, pd.Timestamp('2023-06-30')), 'x'] == 3.0


class TestInterpolateToTarget:
    """Tests de la densification d'index de interpolate_to_target."""

    def test_densifies_index_to_target_frequency(self, aligner):
        """Un index trimestriel est densifié en index mensuel sur la plage d'origine."""
        dates = pd.date_range('2023-01-01', periods=3, freq='QS')
        df = pd.DataFrame({'gdp': [100.0, 110.0, 120.0]}, index=dates)

        result = aligner.interpolate_to_target(df, ['gdp'], 'MS', is_panel=False)

        # 3 trimestres densifiés en mois, restreints à la plage d'origine
        expected_index = pd.date_range('2023-01-01', '2023-07-01', freq='MS')
        pd.testing.assert_index_equal(result.index, expected_index)

        # Les observations d'origine sont conservées aux débuts de trimestre
        assert result.loc[dates, 'gdp'].tolist() == [100.0, 110.0, 120.0]

        # Les mois intermédiaires sont comblés par interpolation
        assert not result['gdp'].isna().any()


class TestConvertToTarget:
    """Tests de l'orientation agrégation/interpolation de convert_to_target."""

    def test_downsampling_routes_to_aggregation(self, aligner):
        """Une variable plus fine que la cible est agrégée (index préservé)."""
        dates = pd.date_range('2023-01-01', '2023-02-28', freq='D')
        df = pd.DataFrame(
            {
                'x': np.ones(len(dates)),
                'y': np.arange(len(dates), dtype=float),
            },
            index=dates,
        )

        result = aligner.convert_to_target(df, ['x'], 'ME', is_panel=False)

        # Contrat d'agrégation : index d'origine préservé, sommes aux fins de mois
        pd.testing.assert_index_equal(result.index, df.index)
        assert result.loc[pd.Timestamp('2023-01-31'), 'x'] == 31.0

    def test_upsampling_routes_to_interpolation(self, aligner):
        """Une variable moins fine que la cible est interpolée (index densifié)."""
        dates = pd.date_range('2023-01-01', periods=3, freq='QS')
        df = pd.DataFrame({'gdp': [100.0, 110.0, 120.0]}, index=dates)

        result = aligner.convert_to_target(df, ['gdp'], 'MS', is_panel=False)

        # Contrat d'interpolation : index densifié à la fréquence cible
        assert len(result) > len(df)
        assert pd.infer_freq(result.index) == 'MS'
