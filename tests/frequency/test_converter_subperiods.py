"""Tests unitaires du comptage de sous-périodes de FrequencyConverter.

Ce module épingle le contrat des deux méthodes de comptage :

- `get_conversion_factor(high, low)` : comptage constant, indépendant de tout
  index, exact pour les paires calendaires emboîtées (12 mois dans une année,
  et non les 12.17 d'un ratio de durées naïf) — remplaçant de l'ancien
  `count_subperiods(low, high)`, dont il inverse l'ordre des arguments ;
- `count_subperiods_per_period(index, low, high)` : comptage exact période par
  période, qui respecte les irrégularités calendaires (février compte 28 jours).

Les deux sont la seule source de vérité du package en matière de comptage de
sous-périodes : `ImputationWindowCalculator._convert_mask_to_frequency` et
`HighFrequencyImputer._prepare_training_data` les consomment toutes deux.

`TestFullPeriodsOnlyUsesCalendarCounts` épingle le consommateur interne du
comptage exact : le garde-fou `full_periods_only` d'`aggregate_to_lower_frequency`,
aligné sur celui de `method='all'`. Avant cet alignement, un facteur constant
(`get_duration_conversion_factor('M', 'D') == 30.0`) exigeait 30 jours par mois
et écartait février toutes les années, ce qui cassait la régularité de la série
mensuelle intermédiaire et faisait disparaître toute covariable journalière du
jeu d'entraînement de `HighFrequencyImputer`.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer
from tsforecast.utils.frequency.converter import FrequencyConverter


@pytest.fixture
def converter():
    """Fixture pour créer une instance de FrequencyConverter."""
    return FrequencyConverter()


class TestCountSubperiods:
    """Tests du comptage constant `get_conversion_factor(high, low)`.

    L'appel se lit « combien de périodes hautes dans une période basse » :
    l'ordre des arguments est donc l'inverse de celui de l'ancien
    `count_subperiods(low, high)`, dont cette méthode reprend le contrat.
    """

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
        assert converter.get_conversion_factor(high_freq, low_freq) == expected

    def test_corrects_the_naive_duration_ratio(self, converter):
        """Le comptage est calendaire, et non un ratio de durées moyennes."""
        # Un ratio de durées naïf annoncerait 12.1667 mois dans une année (365/30)
        assert converter.get_conversion_factor('M', 'Y') == 12.0

    def test_accepts_user_and_anchored_frequencies(self, converter):
        """Les libellés utilisateur et les offsets ancrés sont normalisés."""
        assert converter.get_conversion_factor('quarterly', 'annual') == 4.0
        assert converter.get_conversion_factor('MS', 'YS') == 12.0
        assert converter.get_conversion_factor('ME', 'YE-DEC') == 12.0

    def test_conventional_count_for_irregular_pairs(self, converter):
        """Les paires sans compte constant portent la valeur conventionnelle."""
        # Un mois compte 28 à 31 jours : la convention retenue est 30
        assert converter.get_conversion_factor('D', 'M') == 30.0
        # Une année compte 365 ou 366 jours, et 52 semaines entamées ou non
        assert converter.get_conversion_factor('D', 'Y') == 365.0
        assert converter.get_conversion_factor('W', 'Y') == 52.0

    def test_inverted_pair_returns_fraction(self, converter):
        """Un appel inversé répond par la fraction de période correspondante."""
        # Comportement propre au convertisseur : une covariable annuelle dans
        # une période trimestrielle rend 0.25. HighFrequencyImputer ne s'y
        # expose plus — "_covariate_scaling_divisors" rend 1.0 pour une colonne
        # jamais ré-agrégée, au lieu de diviser par 0.25 donc de multiplier par 4
        assert converter.get_conversion_factor('Y', 'Q') == 0.25
        assert converter.get_conversion_factor('Y', 'M') == pytest.approx(1 / 12)

    def test_unknown_pair_falls_back_on_duration_ratio(self, converter):
        """Les paires absentes de la table retombent sur le ratio de durées."""
        # Semaines dans un mois : aucun compte calendaire constant n'existe
        assert converter.get_conversion_factor('W', 'M') == pytest.approx(30 / 7)

    def test_unsupported_frequency_raises(self, converter):
        """Une fréquence inconnue remonte une ValueError."""
        with pytest.raises(ValueError):
            converter.get_conversion_factor('not_a_frequency', 'Y')


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
# Grille journalière de référence : 9 années pleines, dont deux bissextiles
# (2016 et 2020), soit 108 mois et 36 trimestres complets
_DAILY_START = '2015-01-01'
_DAILY_END = '2023-12-31'


@pytest.fixture
def daily_ones() -> pd.Series:
    """Série journalière de 1.0 sur 2015-2023 : chaque somme compte ses jours."""
    index = pd.date_range(_DAILY_START, _DAILY_END, freq='D')
    return pd.Series(1.0, index=index)


class TestFullPeriodsOnlyUsesCalendarCounts:
    """Tests du garde-fou `full_periods_only` d'`aggregate_to_lower_frequency`.

    Le nombre de sous-périodes attendu vient de `count_subperiods_per_period`
    (décompte calendaire par période) et non d'un facteur constant : une
    période complète est retenue quelle que soit sa longueur réelle, une
    période amputée reste écartée.
    """

    def test_february_survives_daily_to_monthly_sum(self, converter, daily_ones):
        """Les 12 mois de chaque année sont agrégés, février compris."""
        result = converter.aggregate_to_lower_frequency(
            daily_ones, 'ME', 'sum', full_periods_only=True
        )

        # 12 mois x 9 années : aucun mois n'est écarté (un facteur constant de
        # 30 jours en retenait 99, les 9 févriers étant jetés)
        assert len(result) == 108
        assert result.notna().sum() == 108

        # Les 9 févriers sont présents et portent leur nombre de jours
        februaries = result[result.index.month == 2]
        assert len(februaries) == 9
        assert februaries.notna().all()

    def test_leap_year_february_counted(self, converter, daily_ones):
        """Février attend 29 jours en 2016 et 2020, 28 les autres années."""
        result = converter.aggregate_to_lower_frequency(
            daily_ones, 'ME', 'sum', full_periods_only=True
        )

        februaries = result[result.index.month == 2]
        assert februaries.tolist() == [28.0, 29.0, 28.0, 28.0, 28.0, 29.0, 28.0, 28.0, 28.0]
        assert februaries.loc['2016-02-29'] == 29.0
        assert februaries.loc['2020-02-29'] == 29.0

        # Le 29e jour est bien EXIGÉ : un février bissextile réduit à 28 jours
        # observés est écarté, là où un février commun de 28 jours passe
        truncated = daily_ones.copy()
        truncated.loc['2016-02-29'] = np.nan
        truncated_result = converter.aggregate_to_lower_frequency(
            truncated, 'ME', 'sum', full_periods_only=True
        )
        assert pd.isna(truncated_result.loc['2016-02-29'])
        assert truncated_result.loc['2015-02-28'] == 28.0

    def test_quarter_lengths_respected(self, converter, daily_ones):
        """Chaque trimestre attend sa longueur réelle : 90/91, 91, 92, 92."""
        result = converter.aggregate_to_lower_frequency(
            daily_ones, 'QE', 'sum', full_periods_only=True
        )

        # 4 trimestres x 9 années, aucun écarté
        assert len(result) == 36
        assert result.notna().sum() == 36

        # Année commune : T1 = 90 ; année bissextile : T1 = 91
        assert result[result.index.year == 2015].tolist() == [90.0, 91.0, 92.0, 92.0]
        assert result[result.index.year == 2016].tolist() == [91.0, 91.0, 92.0, 92.0]
        assert result[result.index.year == 2020].tolist() == [91.0, 91.0, 92.0, 92.0]

        # T2 = 91 et T3 = T4 = 92 quelle que soit l'année
        for quarter_month, expected in ((6, 91.0), (9, 92.0), (12, 92.0)):
            quarter_values = result[result.index.month == quarter_month]
            assert quarter_values.tolist() == [expected] * 9

    def test_partial_period_still_rejected(self, converter, daily_ones):
        """Un mois amputé de 3 jours reste écarté : le contrôle devient exact."""
        partial = daily_ones.copy()
        partial.loc['2015-03-10':'2015-03-12'] = np.nan

        result = converter.aggregate_to_lower_frequency(
            partial, 'ME', 'sum', full_periods_only=True
        )

        # Mars 2015 n'observe que 28 de ses 31 jours : écarté
        assert pd.isna(result.loc['2015-03-31'])
        # Un seul mois perdu sur les 108 : le correctif ne relâche pas le contrôle
        assert result.notna().sum() == 107

    def test_daily_covariate_reaches_training_set(self):
        """Une covariable journalière atteint le jeu d'entraînement de l'imputeur.

        Test d'intégration du bout de chaîne : covariable journalière dense,
        variable annuelle à imputer, étape mensuelle. Avec un décompte
        constant, la perte des févriers cassait la régularité de la série
        mensuelle, l'agrégation M->Y suivante jetait chaque année entière et
        l'étape basculait en repli par interpolation, avec pour seul indice
        « 0 usable covariate(s) ».
        """
        dates = pd.date_range(_DAILY_START, _DAILY_END, freq='D')
        rng = np.random.default_rng(0)

        df = pd.DataFrame(index=dates)
        df.index.name = 'date'
        # Covariable journalière dense
        df['daily_cov'] = 10.0 + rng.normal(0, 1.0, len(dates))
        # Variable annuelle : observée au seul dernier jour de chaque année
        year_ends = dates[(dates.month == 12) & (dates.day == 31)]
        df['annual_var'] = np.nan
        df.loc[year_ends, 'annual_var'] = 1000.0 + 10.0 * np.arange(len(year_ends))

        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
        )

        # Neutralisation des avertissements préexistants et étrangers à ce test
        # (provenance des cellules vides, dates sans covariable observée sur une
        # grille journalière) : seul le contenu du plan d'imputation est en jeu
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            imputer.fit(df)

        # Une seule étape, celle de la variable annuelle au stade mensuel
        assert len(imputer.imputation_plan_) == 1
        step = imputer.imputation_plan_[0]

        # L'étape porte un modèle entraîné, pas le repli par interpolation
        assert not step.is_fallback
        # La covariable journalière a bien atteint le jeu d'entraînement
        assert step.feature_cols == ('daily_cov',)
