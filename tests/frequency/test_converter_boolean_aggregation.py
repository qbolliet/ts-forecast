"""Tests unitaires des méthodes d'agrégation booléennes de FrequencyConverter.

Ce module épingle le contrat des méthodes `'all'`/`'any'` introduites par le
correctif §3.3 de `high_frequency_imputer_review.md`, qui remplacent le couple
somme/comptage utilisé jusque là par `ImputationWindowCalculator._convert_mask_to_frequency`
pour déterminer si une période basse fréquence est intégralement couverte par
une fenêtre d'imputation à plus haute fréquence :

- `'all'` : une période cible est vraie ssi TOUTES ses sous-périodes à la
  fréquence source sont présentes ET vraies. Une période vide, ou seulement
  partiellement présente (bord de grille), est toujours fausse — même si les
  valeurs présentes sont toutes vraies.
- `'any'` : une période cible est vraie ssi au moins une sous-période présente
  est vraie. Une période vide est fausse (comportement déjà natif de
  `Series.any()` sur une entrée vide).
"""
import numpy as np
import pandas as pd
import pytest

from tsforecast.utils.frequency.converter import FrequencyConverter


@pytest.fixture
def converter():
    """Fixture pour créer une instance de FrequencyConverter."""
    return FrequencyConverter()


class TestAggregateAll:
    """Tests de la méthode d'agrégation `'all'`."""

    def test_fully_covered_period_is_true(self, converter):
        """Une année dont les 12 mois sont vrais donne True."""
        index = pd.date_range('2020-01-01', periods=12, freq='MS')
        mask = pd.Series(True, index=index)

        result = converter.aggregate_to_lower_frequency(mask, 'YS', method='all')

        assert result.tolist() == [True]

    def test_one_false_subperiod_makes_the_period_false(self, converter):
        """Un seul mois faux au sein d'une année intégralement présente suffit."""
        index = pd.date_range('2020-01-01', periods=12, freq='MS')
        mask = pd.Series(True, index=index)
        mask.iloc[5] = False

        result = converter.aggregate_to_lower_frequency(mask, 'YS', method='all')

        assert result.tolist() == [False]

    def test_partially_present_period_is_false_even_if_all_present_are_true(self, converter):
        """Bord de grille : une année incomplète (7 mois sur 12) est fausse,
        même si les 7 mois présents sont tous vrais (coeur du correctif §3.3 :
        empêche une lecture vacuously true de la partie présente)."""
        # Grille démarrant en juin : seuls 7 mois de l'année 2020 existent
        index = pd.date_range('2020-06-01', periods=7, freq='MS')
        mask = pd.Series(True, index=index)

        result = converter.aggregate_to_lower_frequency(mask, 'YS', method='all')

        assert result.tolist() == [False]

    def test_two_full_years_are_independently_evaluated(self, converter):
        """Deux années pleines, l'une entièrement vraie et l'autre non."""
        index = pd.date_range('2020-01-01', periods=24, freq='MS')
        mask = pd.Series(True, index=index)
        mask.iloc[18] = False  # un mois de la deuxième année est faux

        result = converter.aggregate_to_lower_frequency(mask, 'YS', method='all')

        assert result.tolist() == [True, False]

    def test_dataframe_columns_are_evaluated_independently(self, converter):
        """Sur un DataFrame, chaque colonne est agrégée indépendamment."""
        index = pd.date_range('2020-01-01', periods=12, freq='MS')
        df = pd.DataFrame({
            'fully_true': [True] * 12,
            'one_false': [True] * 5 + [False] + [True] * 6,
        }, index=index)

        result = converter.aggregate_to_lower_frequency(df, 'YS', method='all')

        assert result['fully_true'].tolist() == [True]
        assert result['one_false'].tolist() == [False]

    def test_full_periods_only_is_ignored_for_all(self, converter):
        """full_periods_only n'affecte pas 'all' : la garantie est déjà intrinsèque."""
        index = pd.date_range('2020-06-01', periods=7, freq='MS')
        mask = pd.Series(True, index=index)

        with_flag = converter.aggregate_to_lower_frequency(
            mask, 'YS', method='all', full_periods_only=True
        )
        without_flag = converter.aggregate_to_lower_frequency(
            mask, 'YS', method='all', full_periods_only=False
        )

        # Toujours un booléen (jamais NaN), et identique dans les deux cas
        assert with_flag.tolist() == [False]
        assert with_flag.tolist() == without_flag.tolist()
        assert with_flag.dtype == bool


class TestAggregateAny:
    """Tests de la méthode d'agrégation `'any'`."""

    def test_at_least_one_true_subperiod_is_true(self, converter):
        """Une seule sous-période vraie suffit à rendre la période vraie."""
        index = pd.date_range('2020-01-01', periods=12, freq='MS')
        mask = pd.Series(False, index=index)
        mask.iloc[3] = True

        result = converter.aggregate_to_lower_frequency(mask, 'YS', method='any')

        assert result.tolist() == [True]

    def test_all_false_subperiods_is_false(self, converter):
        """Aucune sous-période vraie donne False."""
        index = pd.date_range('2020-01-01', periods=12, freq='MS')
        mask = pd.Series(False, index=index)

        result = converter.aggregate_to_lower_frequency(mask, 'YS', method='any')

        assert result.tolist() == [False]

    def test_partially_present_period_does_not_require_full_coverage(self, converter):
        """Contrairement à 'all', 'any' ne pénalise pas une période partielle :
        une seule sous-période présente et vraie suffit."""
        index = pd.date_range('2020-06-01', periods=7, freq='MS')
        mask = pd.Series(True, index=index)

        result = converter.aggregate_to_lower_frequency(mask, 'YS', method='any')

        assert result.tolist() == [True]

    def test_dataframe_columns_are_evaluated_independently(self, converter):
        """Sur un DataFrame, chaque colonne est agrégée indépendamment."""
        index = pd.date_range('2020-01-01', periods=12, freq='MS')
        df = pd.DataFrame({
            'all_false': [False] * 12,
            'one_true': [False] * 5 + [True] + [False] * 6,
        }, index=index)

        result = converter.aggregate_to_lower_frequency(df, 'YS', method='any')

        assert result['all_false'].tolist() == [False]
        assert result['one_true'].tolist() == [True]
