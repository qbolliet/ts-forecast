"""Tests unitaires pour les conversions de fréquences mixtes (target_freq dict).

Ce module épingle le contrat d'index de FrequencyConverter.convert_frequency
lorsqu'un dictionnaire {colonne: fréquence} est fourni : l'index de sortie est
l'union des index cibles des colonnes converties. L'index source disparaît dès
que toutes les colonnes sont converties ; seules les colonnes absentes du
dictionnaire (ou déjà à la fréquence cible) conservent leurs dates d'origine.
"""
import numpy as np
import pandas as pd
import pytest

from tsforecast.utils.frequency.converter import FrequencyConverter


class TestMixedFrequencyIndexContract:
    """Tests du contrat d'index pour les conversions avec target_freq dict."""

    @pytest.fixture
    def converter(self):
        """Fixture pour créer une instance de FrequencyConverter."""
        return FrequencyConverter()

    @pytest.fixture
    def daily_df(self):
        """DataFrame journalier à deux colonnes sur 3 mois complets."""
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        return pd.DataFrame(
            {
                'ventes': np.arange(len(dates), dtype=float),
                'temperature': np.linspace(0.0, 20.0, len(dates)),
            },
            index=dates,
        )

    def test_mixed_dict_index_is_union_of_target_indexes(self, converter, daily_df):
        """L'index de sortie est l'union des index cibles, sans dates source."""
        # Conversion mixte : une colonne mensuelle, une colonne hebdomadaire
        result = converter.convert_frequency(
            daily_df,
            target_freq={'ventes': 'monthly', 'temperature': 'weekly'},
            method='mean',
            alignment_method='ffill',
        )

        # Conversions de référence colonne par colonne
        monthly = converter.convert_frequency(daily_df['ventes'], 'monthly', method='mean')
        weekly = converter.convert_frequency(daily_df['temperature'], 'weekly', method='mean')
        expected_index = monthly.index.union(weekly.index)

        # L'index de sortie est exactement l'union des index cibles
        pd.testing.assert_index_equal(result.index, expected_index)

        # Aucune date journalière hors cibles ne survit (l'index source disparaît)
        assert len(result) < len(daily_df)
        assert not daily_df.index.difference(expected_index).isin(result.index).any()

    def test_mixed_dict_nan_pattern_without_alignment(self, converter, daily_df):
        """Avec alignment_method='none', les NaN de l'union ne sont pas comblés."""
        result = converter.convert_frequency(
            daily_df,
            target_freq={'ventes': 'monthly', 'temperature': 'weekly'},
            method='mean',
            alignment_method='none',
        )

        # Index cibles de référence
        monthly = converter.convert_frequency(daily_df['ventes'], 'monthly', method='mean')
        weekly = converter.convert_frequency(daily_df['temperature'], 'weekly', method='mean')

        # La colonne mensuelle est NaN sur les dates purement hebdomadaires
        weekly_only_dates = weekly.index.difference(monthly.index)
        assert result.loc[weekly_only_dates, 'ventes'].isna().all()

        # La colonne mensuelle porte ses valeurs aux fins de mois
        pd.testing.assert_series_equal(
            result.loc[monthly.index, 'ventes'], monthly,
            check_names=False, check_freq=False,
        )

    def test_dict_with_single_freq_equivalent_to_scalar(self, converter, daily_df):
        """Un dict où toutes les colonnes ont la même cible ≡ cible scalaire."""
        result_dict = converter.convert_frequency(
            daily_df,
            target_freq={'ventes': 'monthly', 'temperature': 'monthly'},
            method='mean',
        )
        result_scalar = converter.convert_frequency(daily_df, 'monthly', method='mean')

        pd.testing.assert_frame_equal(result_dict, result_scalar)

    def test_partial_dict_preserves_unconverted_column_dates(self, converter, daily_df):
        """Les colonnes hors dict gardent leurs dates d'origine dans l'union."""
        result = converter.convert_frequency(
            daily_df,
            target_freq={'ventes': 'monthly'},
            method='mean',
            alignment_method='none',
        )

        # Les dates journalières de la colonne préservée restent présentes
        assert daily_df.index.isin(result.index).all()

        # La colonne préservée conserve ses valeurs d'origine sur ces dates
        pd.testing.assert_series_equal(
            result.loc[daily_df.index, 'temperature'],
            daily_df['temperature'],
            check_names=False,
        )

    def test_column_already_at_target_freq_left_unchanged(self, converter):
        """Une colonne déjà à la fréquence cible est retournée telle quelle."""
        dates = pd.date_range('2023-01-31', periods=6, freq='ME')
        df = pd.DataFrame({'x': np.arange(6, dtype=float)}, index=dates)

        result = converter.convert_frequency(df, target_freq={'x': 'monthly'}, method='mean')

        pd.testing.assert_frame_equal(result, df)
