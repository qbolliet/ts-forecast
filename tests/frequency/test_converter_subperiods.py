"""Tests unitaires du comptage de sous-périodes de FrequencyConverter.

Ce module épingle le contrat des deux méthodes de comptage introduites par le
correctif §2.1(b) de `high_frequency_imputer_review.md` :

- `count_subperiods(low, high)` : comptage constant, indépendant de tout index,
  exact pour les paires calendaires emboîtées (12 mois dans une année, et non
  les 12.17 du ratio de durées) ;
- `count_subperiods_per_period(index, low, high)` : comptage exact période par
  période, qui respecte les irrégularités calendaires (février compte 28 jours).

Les deux sont la seule source de vérité du package en matière de comptage de
sous-périodes : `ImputationWindowCalculator._convert_mask_to_frequency` et
`HighFrequencyImputer._prepare_training_data` les consomment toutes deux.
"""
import numpy as np
import pandas as pd
import pytest

from tsforecast.utils.frequency.converter import FrequencyConverter


@pytest.fixture
def converter():
    """Fixture pour créer une instance de FrequencyConverter."""
    return FrequencyConverter()


class TestCountSubperiods:
    """Tests du comptage constant `count_subperiods`."""

    @pytest.mark.parametrize(
        'low_freq, high_freq, expected',
        [
            ('Y', 'M', 12.0),
            ('Q', 'M', 3.0),
            ('Y', 'Q', 4.0),
            ('M', 'M', 1.0),
        ],
    )
    def test_calendar_pairs_are_exact(self, converter, low_freq, high_freq, expected):
        """Les paires calendaires emboîtées donnent un compte entier exact."""
        assert converter.count_subperiods(low_freq, high_freq) == expected

    def test_differs_from_duration_ratio(self, converter):
        """Le comptage corrige le ratio de durées, inexact par construction."""
        # Le ratio de durées annonce 12.1667 mois dans une année (365/30)
        assert converter.get_conversion_factor('M', 'Y') != 12.0
        assert converter.count_subperiods('Y', 'M') == 12.0

    def test_accepts_user_and_anchored_frequencies(self, converter):
        """Les libellés utilisateur et les offsets ancrés sont normalisés."""
        assert converter.count_subperiods('annual', 'quarterly') == 4.0
        assert converter.count_subperiods('YS', 'MS') == 12.0
        assert converter.count_subperiods('YE-DEC', 'ME') == 12.0

    def test_conventional_count_for_irregular_pairs(self, converter):
        """Les paires sans compte constant portent la valeur conventionnelle."""
        # Un mois compte 28 à 31 jours : la convention retenue est 30
        assert converter.count_subperiods('M', 'D') == 30.0
        # Une année compte 365 ou 366 jours, et 52 semaines entamées ou non
        assert converter.count_subperiods('Y', 'D') == 365.0
        assert converter.count_subperiods('Y', 'W') == 52.0

    def test_inverted_pair_returns_fraction(self, converter):
        """Un appel inversé répond par la fraction de période correspondante."""
        # Cas rencontré à l'entraînement : covariable annuelle dans une période
        # trimestrielle (cf. HighFrequencyImputer._covariate_subperiod_counts)
        assert converter.count_subperiods('Q', 'Y') == 0.25
        assert converter.count_subperiods('M', 'Y') == pytest.approx(1 / 12)

    def test_unknown_pair_falls_back_on_duration_ratio(self, converter):
        """Les paires absentes de la table retombent sur le ratio de durées."""
        # Semaines dans un mois : aucun compte calendaire constant n'existe
        assert converter.count_subperiods('M', 'W') == pytest.approx(30 / 7)

    def test_unsupported_frequency_raises(self, converter):
        """Une fréquence inconnue remonte une ValueError."""
        with pytest.raises(ValueError):
            converter.count_subperiods('Y', 'not_a_frequency')


class TestCountSubperiodsPerPeriod:
    """Tests du comptage exact période par période."""

    def test_days_per_month_respect_calendar(self, converter):
        """Le comptage journalier suit la longueur réelle de chaque mois."""
        index = pd.date_range('2023-01-31', periods=4, freq='ME')
        counts = converter.count_subperiods_per_period(index, 'M', 'D')
        np.testing.assert_array_equal(counts, [31.0, 28.0, 31.0, 30.0])

    def test_leap_year_is_detected(self, converter):
        """Février 2024 compte 29 jours, là où le compte constant donne 30."""
        index = pd.DatetimeIndex(['2024-02-29'])
        counts = converter.count_subperiods_per_period(index, 'M', 'D')
        np.testing.assert_array_equal(counts, [29.0])

    def test_months_per_year_are_constant(self, converter):
        """Le comptage mensuel vaut 12 pour toute année, bissextile ou non."""
        index = pd.date_range('2023-12-31', periods=2, freq='YE')
        counts = converter.count_subperiods_per_period(index, 'Y', 'M')
        np.testing.assert_array_equal(counts, [12.0, 12.0])

    def test_start_anchored_index_is_supported(self, converter):
        """Un index ancré en début de période donne les mêmes comptes."""
        index = pd.date_range('2023-01-01', periods=3, freq='MS')
        counts = converter.count_subperiods_per_period(index, 'MS', 'D')
        np.testing.assert_array_equal(counts, [31.0, 28.0, 31.0])

    def test_empty_index_returns_empty_array(self, converter):
        """Un index vide donne un tableau vide, sans lever."""
        index = pd.DatetimeIndex([], dtype='datetime64[ns]')
        counts = converter.count_subperiods_per_period(index, 'M', 'D')
        assert len(counts) == 0

    def test_falls_back_on_constant_count(self, converter):
        """Une base non exprimable en Period retombe sur le compte constant."""
        # 'SM' (semi-mensuel) n'est pas une fréquence de Period pandas
        index = pd.date_range('2023-01-31', periods=2, freq='ME')
        counts = converter.count_subperiods_per_period(index, 'M', 'SM')
        np.testing.assert_array_equal(counts, [2.0, 2.0])
