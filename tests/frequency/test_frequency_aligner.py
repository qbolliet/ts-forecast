"""Tests unitaires pour le FrequencyAligner.

Ce module épingle les contrats d'index du FrequencyAligner, dont dépend le
HighFrequencyImputer :
- aggregate_to_target préserve l'index dense d'origine (valeurs agrégées
  réindexées, NaN hors bornes de période, NaN pour les périodes incomplètes) ;
- interpolate_to_target densifie l'index (union avec les dates interpolées,
  couvrant les périodes sources complètes des observations extrêmes) ;
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
        result = aligner._aggregate_to_target(monthly_df, ['x'], 'QE')

        # La colonne 'y' n'étant pas agrégée, aucune ligne n'est tout-NaN :
        # l'index d'origine est intégralement préservé
        pd.testing.assert_index_equal(result.index, monthly_df.index)

    def test_aggregated_values_on_period_boundaries_only(self, aligner, monthly_df):
        """Les valeurs agrégées ne sont portées que par les fins de période complètes."""
        result = aligner._aggregate_to_target(monthly_df, ['x'], 'QE')

        # Somme des mois sur les trimestres complets (3 mois chacun)
        assert result.loc[pd.Timestamp('2023-03-31'), 'x'] == 3.0
        assert result.loc[pd.Timestamp('2023-06-30'), 'x'] == 3.0

        # NaN partout ailleurs (mois intermédiaires et trimestre incomplet T3)
        boundaries = pd.DatetimeIndex(['2023-03-31', '2023-06-30'])
        assert result.loc[result.index.difference(boundaries), 'x'].isna().all()

        # La colonne non agrégée est inchangée
        pd.testing.assert_series_equal(result['y'], monthly_df['y'])

    def test_aggregate_labels_land_on_source_index(self, aligner):
        """Agrégation M->Q d'un index MS : labels sur QS, reindex sans perte."""
        dates = pd.date_range('2023-01-01', periods=8, freq='MS')
        df = pd.DataFrame({'x': np.ones(len(dates))}, index=dates)

        result = aligner._aggregate_to_target(df, ['x'], 'Q')

        # Le résultat ne doit pas être vide : les labels doivent atterrir sur l'index MS
        assert len(result) > 0
        assert result['x'].notna().any()

    def test_aggregate_preserves_row_count(self, aligner):
        """aggregate_to_target ne supprime aucune ligne (contrat d'alignement)."""
        dates = pd.date_range('2023-01-01', periods=8, freq='MS')
        df = pd.DataFrame({'x': np.ones(len(dates))}, index=dates)

        result = aligner._aggregate_to_target(df, ['x'], 'QS')

        assert len(result) == len(df)

    def test_panel_aggregate_preserves_row_count(self, aligner):
        """Panel : aggregate_to_target ne supprime aucune ligne, même avec un index MS
        mal ancré vis-à-vis d'une fréquence cible sans position ('Q')."""
        dates = pd.date_range('2023-01-01', periods=8, freq='MS')
        index = pd.MultiIndex.from_product(
            [['A', 'B'], dates], names=['entity', 'date']
        )
        df = pd.DataFrame({'x': np.ones(len(index))}, index=index)

        result = aligner._aggregate_to_target(
            df, [('A', 'x'), ('B', 'x')], 'Q'
        )

        assert len(result) == len(df)
        # Les labels agrégés doivent atterrir sur l'index MS d'origine
        assert result['x'].notna().any()

    def test_aggregate_sparse_column_uses_variable_frequency(self, aligner):
        """Une colonne trimestrielle portée par un index mensuel s'agrège à l'année.

        La fréquence source doit être détectée sur les valeurs observées de la
        variable (4 trimestres par an) et non sur l'index (12 mois par an) :
        sinon `full_periods_only` masque toutes les périodes et la colonne
        agrégée devient intégralement NaN.
        """
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        quarterly = pd.Series(np.nan, index=dates)
        quarterly[quarterly.index.month.isin([1, 4, 7, 10])] = 10.0
        df = pd.DataFrame({'gdp': quarterly, 'dense': np.ones(len(dates))}, index=dates)

        result = aligner._aggregate_to_target(df, ['gdp'], 'Y')

        # Somme des 4 trimestres, portée par le 1er janvier de chaque année complète
        expected_labels = pd.DatetimeIndex(['2020-01-01', '2021-01-01'])
        assert result.loc[expected_labels, 'gdp'].tolist() == [40.0, 40.0]
        # Aucune valeur ailleurs, et l'index reste intact
        assert result['gdp'].notna().sum() == 2
        pd.testing.assert_index_equal(result.index, df.index)

    def test_aggregate_sparse_irregular_column_falls_back_to_index(self, aligner):
        """Observations irrégulières : repli sur la fréquence de l'index sans erreur."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        values = np.arange(12, dtype=float)
        values[[2, 7]] = np.nan
        df = pd.DataFrame({'x': values}, index=dates)

        result = aligner._aggregate_to_target(df, ['x'], 'Q')

        # Index préservé et sémantique full_periods_only inchangée : les
        # trimestres amputés d'un mois (T1 et T3) restent NaN
        pd.testing.assert_index_equal(result.index, df.index)
        assert np.isnan(result.loc[pd.Timestamp('2020-01-01'), 'x'])
        assert np.isnan(result.loc[pd.Timestamp('2020-07-01'), 'x'])
        assert result.loc[pd.Timestamp('2020-04-01'), 'x'] == 12.0
        assert result.loc[pd.Timestamp('2020-10-01'), 'x'] == 30.0

    def test_skips_missing_column(self, aligner, monthly_df):
        """Une clé absente du DataFrame est ignorée, pas une erreur."""
        result = aligner._aggregate_to_target(monthly_df, ['missing'], 'QE')

        pd.testing.assert_frame_equal(result, monthly_df)

    def test_panel_skips_missing_column(self, aligner):
        """Idem en panel : une colonne absente pour l'entité est ignorée."""
        dates = pd.date_range('2023-01-01', periods=4, freq='MS')
        index = pd.MultiIndex.from_product([['A'], dates], names=['entity', 'date'])
        df = pd.DataFrame({'x': np.ones(len(index))}, index=index)

        result = aligner._aggregate_to_target(df, [('A', 'missing')], 'QS')

        pd.testing.assert_frame_equal(result, df)

    def test_skips_column_with_no_observations(self, aligner):
        """Une colonne entièrement NaN est laissée telle quelle, sans erreur."""
        dates = pd.date_range('2023-01-01', periods=8, freq='MS')
        df = pd.DataFrame({'x': [np.nan] * 8}, index=dates)

        result = aligner._aggregate_to_target(df, ['x'], 'QS')

        assert result['x'].isna().all()

    def test_panel_skips_column_with_no_observations(self, aligner):
        """Idem en panel : une colonne entièrement NaN pour l'entité est ignorée."""
        dates = pd.date_range('2023-01-01', periods=8, freq='MS')
        index = pd.MultiIndex.from_product([['A'], dates], names=['entity', 'date'])
        df = pd.DataFrame({'x': [np.nan] * len(index)}, index=index)

        result = aligner._aggregate_to_target(df, [('A', 'x')], 'QS')

        assert result['x'].isna().all()

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

        result = aligner._aggregate_to_target(
            df, [('A', 'x'), ('B', 'x')], 'QE'
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
        """Un index trimestriel est densifié en index mensuel sur ses périodes sources."""
        dates = pd.date_range('2023-01-01', periods=3, freq='QS')
        df = pd.DataFrame({'gdp': [100.0, 110.0, 120.0]}, index=dates)

        result = aligner._interpolate_to_target(df, ['gdp'], 'MS')

        # 3 trimestres densifiés en 9 mois : l'observation du T3, en position
        # start, couvre aussi août et septembre
        expected_index = pd.date_range('2023-01-01', '2023-09-01', freq='MS')
        pd.testing.assert_index_equal(result.index, expected_index)

        # Les observations d'origine sont conservées aux débuts de trimestre
        assert result.loc[dates, 'gdp'].tolist() == [100.0, 110.0, 120.0]

        # Les mois intermédiaires sont comblés par interpolation
        assert not result['gdp'].isna().any()

    def test_empty_keys_returns_df_unchanged(self, aligner):
        """Aucune clé à interpoler : le DataFrame est retourné inchangé."""
        dates = pd.date_range('2023-01-01', periods=3, freq='QS')
        df = pd.DataFrame({'gdp': [100.0, 110.0, 120.0]}, index=dates)

        result = aligner._interpolate_to_target(df, [], 'MS')

        pd.testing.assert_frame_equal(result, df)

    def test_panel_densifies_index_per_entity(self, aligner):
        """Panel : chaque entité est densifiée indépendamment à sa fréquence cible."""
        dates = pd.date_range('2023-01-01', periods=3, freq='QS')
        index = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
        df = pd.DataFrame(
            {'gdp': [100.0, 110.0, 120.0, 200.0, 210.0, 220.0]}, index=index
        )

        result = aligner._interpolate_to_target(
            df, [('A', 'gdp'), ('B', 'gdp')], 'MS'
        )

        expected_dates = pd.date_range('2023-01-01', '2023-09-01', freq='MS', name='date')
        for entity in ['A', 'B']:
            entity_frame = result.xs(entity, level='entity')
            pd.testing.assert_index_equal(entity_frame.index, expected_dates)
            assert not entity_frame['gdp'].isna().any()

        # Les observations d'origine sont conservées à leurs dates
        assert result.loc[('A', pd.Timestamp('2023-01-01')), 'gdp'] == 100.0
        assert result.loc[('B', pd.Timestamp('2023-04-01')), 'gdp'] == 210.0

    def test_skips_missing_column(self, aligner):
        """Une clé absente du DataFrame est ignorée, pas une erreur."""
        dates = pd.date_range('2023-01-01', periods=3, freq='QS')
        df = pd.DataFrame({'gdp': [100.0, 110.0, 120.0]}, index=dates)

        result = aligner._interpolate_to_target(df, ['missing'], 'MS')

        pd.testing.assert_frame_equal(result, df)

    def test_panel_skips_missing_column(self, aligner):
        """Idem en panel : une colonne absente pour l'entité est ignorée."""
        dates = pd.date_range('2023-01-01', periods=3, freq='QS')
        index = pd.MultiIndex.from_product([['A'], dates], names=['entity', 'date'])
        df = pd.DataFrame({'gdp': [100.0, 110.0, 120.0]}, index=index)

        result = aligner._interpolate_to_target(df, [('A', 'missing')], 'MS')

        pd.testing.assert_frame_equal(result, df)


class TestDensifiedSpanFollowsPosition:
    """Tests de la couverture de la densification selon la position de période."""

    def test_start_position_extends_forward(self, aligner):
        """Position start : les sous-périodes suivant la dernière observation sont couvertes."""
        dates = pd.date_range('2023-01-01', periods=3, freq='QS')
        df = pd.DataFrame({'gdp': [100.0, 110.0, 120.0]}, index=dates)

        result = aligner._interpolate_to_target(df, ['gdp'], 'MS')

        # L'observation 2023-07-01 couvre juillet, août et septembre
        assert result.index[0] == pd.Timestamp('2023-01-01')
        assert result.index[-1] == pd.Timestamp('2023-09-01')

    def test_end_position_extends_backward(self, aligner):
        """Position end : les sous-périodes précédant la première observation sont couvertes."""
        dates = pd.date_range('2023-03-31', periods=3, freq='QE')
        df = pd.DataFrame({'gdp': [100.0, 110.0, 120.0]}, index=dates)

        result = aligner._interpolate_to_target(df, ['gdp'], 'ME')

        # L'observation 2023-03-31 couvre janvier, février et mars
        assert result.index[0] == pd.Timestamp('2023-01-31')
        assert result.index[-1] == pd.Timestamp('2023-09-30')

    def test_dense_index_is_not_extended_beyond_its_own_periods(self, aligner):
        """Une frame déjà à la fréquence cible n'est pas étendue : les trous sont comblés."""
        dates = pd.date_range('2023-01-01', periods=9, freq='MS')
        values = [100.0, np.nan, np.nan, 110.0, np.nan, np.nan, 120.0, np.nan, np.nan]
        df = pd.DataFrame({'gdp': values}, index=dates)

        result = aligner._interpolate_to_target(df, ['gdp'], 'MS')

        pd.testing.assert_index_equal(result.index, dates)


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

        result = aligner.convert_to_target(df, ['x'], 'ME')

        # Contrat d'agrégation : index d'origine préservé, sommes aux fins de mois
        pd.testing.assert_index_equal(result.index, df.index)
        assert result.loc[pd.Timestamp('2023-01-31'), 'x'] == 31.0

    def test_upsampling_routes_to_interpolation(self, aligner):
        """Une variable moins fine que la cible est interpolée (index densifié)."""
        dates = pd.date_range('2023-01-01', periods=3, freq='QS')
        df = pd.DataFrame({'gdp': [100.0, 110.0, 120.0]}, index=dates)

        result = aligner.convert_to_target(df, ['gdp'], 'MS')

        # Contrat d'interpolation : index densifié à la fréquence cible
        assert len(result) > len(df)
        assert pd.infer_freq(result.index) == 'MS'

    def test_empty_keys_returns_df_unchanged(self, aligner):
        """Aucune clé à convertir : le DataFrame est retourné inchangé."""
        dates = pd.date_range('2023-01-01', periods=4, freq='MS')
        df = pd.DataFrame({'x': np.ones(len(dates))}, index=dates)

        result = aligner.convert_to_target(df, [], 'QS')

        pd.testing.assert_frame_equal(result, df)

    def test_skips_missing_column(self, aligner):
        """Une clé absente du DataFrame est ignorée, pas une erreur."""
        dates = pd.date_range('2023-01-01', periods=4, freq='MS')
        df = pd.DataFrame({'x': np.ones(len(dates))}, index=dates)

        result = aligner.convert_to_target(df, ['missing'], 'QS')

        pd.testing.assert_frame_equal(result, df)
