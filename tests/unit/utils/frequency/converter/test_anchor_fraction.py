"""Tests unitaires pour `interpolate_to_higher_frequency(anchor_fraction=...)`.

Ce module couvre la position d'ancrage de la valeur dans sa période (prérequis
P2 de `HighFrequencyImputer`, §10.2 de la spécification d'architecture) : le
décalage des ancres à une fraction de leur période source, l'interpolation sur
l'union (ancres décalées ∪ grille cible), la restriction finale à la grille
cible, et la non-régression stricte du chemin `anchor_fraction=None`.
"""
import pytest
import pandas as pd
import numpy as np
from tsforecast.utils.frequency.converter import FrequencyConverter


class TestAnchorFraction:
    """Tests pour le paramètre `anchor_fraction` de l'interpolation."""

    @pytest.fixture
    def converter(self):
        """Fixture pour créer une instance de FrequencyConverter."""
        return FrequencyConverter()

    @pytest.fixture
    def yearly_series(self):
        """Série annuelle de l'exemple §10.2 : 120 en 2021, 132 en 2022."""
        return pd.Series(
            [120.0, 132.0],
            index=pd.date_range('2021-12-31', periods=2, freq='YE')
        )

    @pytest.fixture
    def quarterly_series(self):
        """Série trimestrielle de 2021, en position end."""
        return pd.Series(
            [10.0, 20.0, 30.0, 40.0],
            index=pd.date_range('2021-03-31', periods=4, freq='QE')
        )

    # Test de non-régression du lot

    def test_anchor_fraction_none_is_unchanged(self, converter, yearly_series,
                                               quarterly_series):
        """Test que anchor_fraction=None reproduit strictement l'ancien comportement."""
        # Valeurs figées, relevées sur l'implémentation antérieure au paramètre
        expected = {
            'Y->Q': (yearly_series, 'Q', [120.0, 120.0, 120.0, 120.0,
                                          123.0, 126.0, 129.0, 132.0]),
            'Y->M': (yearly_series, 'M', [120.0] * 12 + [121.0, 122.0, 123.0, 124.0,
                                                         125.0, 126.0, 127.0, 128.0,
                                                         129.0, 130.0, 131.0, 132.0]),
            'Q->M': (quarterly_series, 'M', [10.0, 10.0, 10.0,
                                             40.0 / 3, 50.0 / 3, 20.0,
                                             70.0 / 3, 80.0 / 3, 30.0,
                                             100.0 / 3, 110.0 / 3, 40.0]),
        }

        for label, (series, target_freq, expected_values) in expected.items():
            # Appel sans le nouvel argument : sortie de référence
            implicit = converter.interpolate_to_higher_frequency(
                series, target_freq, method='linear'
            )
            assert implicit.tolist() == pytest.approx(expected_values), (
                f"{label} : la sortie a changé par rapport à la référence figée"
            )

            # Passage explicite de None : sortie strictement identique
            explicit = converter.interpolate_to_higher_frequency(
                series, target_freq, method='linear', anchor_fraction=None
            )
            pd.testing.assert_series_equal(implicit, explicit)

    # Tests des positions d'ancre sur l'index intermédiaire

    @pytest.mark.internal
    def test_anchor_fraction_zero_shifts_to_period_start(self, converter,
                                                         yearly_series):
        """Test que anchor_fraction=0.0 place les ancres en début de période."""
        shifted = converter._shift_index_to_anchor_fraction(
            index=yearly_series.index,
            source_freq='YE',
            target_freq='QE',
            anchor_fraction=0.0
        )

        expected = pd.DatetimeIndex(['2021-01-01', '2022-01-01'])
        pd.testing.assert_index_equal(shifted, expected)

    @pytest.mark.internal
    def test_anchor_fraction_one_shifts_to_period_end(self, converter,
                                                      yearly_series):
        """Test que anchor_fraction=1.0 place les ancres en fin de période."""
        shifted = converter._shift_index_to_anchor_fraction(
            index=yearly_series.index,
            source_freq='YE',
            target_freq='QE',
            anchor_fraction=1.0
        )

        # Bornage à la fin de période : pas de débordement sur la période suivante
        expected = pd.DatetimeIndex(['2021-12-31', '2022-12-31'])
        pd.testing.assert_index_equal(shifted, expected)

    @pytest.mark.internal
    def test_anchor_fraction_half_uses_mid_period(self, converter, yearly_series):
        """Test que anchor_fraction=0.5 ancre au milieu de l'année (exemple §10.2)."""
        # Positions d'ancre attendues : milieu de 2021 et de 2022
        shifted = converter._shift_index_to_anchor_fraction(
            index=yearly_series.index,
            source_freq='YE',
            target_freq='QE',
            anchor_fraction=0.5
        )
        pd.testing.assert_index_equal(
            shifted, pd.DatetimeIndex(['2021-07-02', '2022-07-02'])
        )

        # Interpolation vers le trimestre avec et sans ancrage
        plain = converter.interpolate_to_higher_frequency(
            yearly_series, 'Q', method='linear'
        )
        mid = converter.interpolate_to_higher_frequency(
            yearly_series, 'Q', method='linear', anchor_fraction=0.5
        )

        # Les quatre valeurs trimestrielles de 2022 diffèrent du cas None
        assert not np.allclose(
            mid['2022'].to_numpy(), plain['2022'].to_numpy(), equal_nan=True
        )

        # Pente proportionnelle au temps entre les deux ancres décalées :
        # 2021-09-30 est à 90 jours de l'ancre 2021-07-02, sur 365 jours
        assert mid['2021-09-30'] == pytest.approx(120.0 + 12.0 * 90 / 365)
        # 2022-06-30 est à 363 jours de cette même ancre, soit 2 jours avant la suivante
        assert mid['2022-06-30'] == pytest.approx(120.0 + 12.0 * 363 / 365)

    # Test de l'index de sortie

    def test_anchor_fraction_union_index_is_used(self, converter, yearly_series):
        """Test que la sortie est indexée exactement sur la grille cible."""
        plain = converter.interpolate_to_higher_frequency(
            yearly_series, 'Q', method='linear'
        )
        mid = converter.interpolate_to_higher_frequency(
            yearly_series, 'Q', method='linear', anchor_fraction=0.5
        )

        # Aucun timestamp décalé résiduel : l'union n'existe qu'en interne
        pd.testing.assert_index_equal(mid.index, plain.index)
        assert pd.Timestamp('2021-07-02') not in mid.index
        assert len(mid) == 8

    # Test de validation

    def test_anchor_fraction_out_of_range_raises(self, converter, yearly_series):
        """Test que les valeurs hors de [0, 1] lèvent une ValueError explicite."""
        for invalid in (-0.1, 1.5):
            with pytest.raises(ValueError) as excinfo:
                converter.interpolate_to_higher_frequency(
                    yearly_series, 'Q', method='linear', anchor_fraction=invalid
                )

            # Le message nomme la valeur reçue et l'intervalle admis
            message = str(excinfo.value)
            assert str(invalid) in message
            assert '[0, 1]' in message

    # Test du comportement aux bords

    def test_anchor_fraction_edges_follow_limit_direction(self, converter,
                                                          yearly_series):
        """Test qu'au-delà de la dernière ancre décalée, limit_direction décide."""
        # Défaut pour une cible en position end : 'backward', qui ne remplit pas
        # les trimestres postérieurs à l'ancre 2022-07-02
        default = converter.interpolate_to_higher_frequency(
            yearly_series, 'Q', method='linear', anchor_fraction=0.5
        )
        assert np.isnan(default['2022-09-30'])
        assert np.isnan(default['2022-12-31'])

        # Direction explicite 'both' : extrapolation plate au-delà de l'ancre
        both = converter.interpolate_to_higher_frequency(
            yearly_series, 'Q', method='linear',
            limit_direction='both', anchor_fraction=0.5
        )
        assert both['2022-09-30'] == pytest.approx(132.0)
        assert both['2022-12-31'] == pytest.approx(132.0)


# Valeurs trimestrielles de 2021 (fins de trimestre) : T1 compte 90 jours, T2 91,
# T3 et T4 92 — les ancres décalées rendent l'interpolation pondérée par le temps
_QUARTERLY_2021 = pd.Series(
    [10.0, 20.0, 30.0, 40.0], index=pd.date_range('2021-03-31', periods=4, freq='QE')
)


class TestAnchorFractionValidation:
    """Only real numbers in ``[0, 1]`` are accepted, without coercion."""

    @pytest.fixture
    def converter(self):
        """A fresh ``FrequencyConverter``."""
        return FrequencyConverter()

    @pytest.mark.parametrize(
        'invalid',
        [
            pytest.param('0.5', id='string'),
            pytest.param(True, id='boolean'),
            pytest.param(float('nan'), id='nan'),
            pytest.param([0.5], id='list'),
        ],
    )
    def test_rejected_values(self, converter, invalid):
        """Strings, booleans, NaN and containers are rejected."""
        with pytest.raises(ValueError, match='Invalid anchor_fraction'):
            converter.interpolate_to_higher_frequency(_QUARTERLY_2021, 'ME', anchor_fraction=invalid)

    def test_integer_one_is_the_period_end(self, converter):
        """``anchor_fraction=1`` (int) anchors at the period ends, interpolated in time."""
        result = converter.interpolate_to_higher_frequency(_QUARTERLY_2021, 'ME', anchor_fraction=1)
        # Valeurs d'or : ancres aux fins de trimestre, 'linear' traité comme 'time'
        # (30 avril à 30 jours du 31 mars, sur 91) ; janvier-février comblés vers l'arrière
        expected = [10.0, 10.0, 10.0,
                    10 + 10 * 30 / 91, 10 + 10 * 61 / 91, 20.0,
                    20 + 10 * 31 / 92, 20 + 10 * 62 / 92, 30.0,
                    30 + 10 * 31 / 92, 30 + 10 * 61 / 92, 40.0]
        assert result.tolist() == pytest.approx(expected)

    def test_numpy_zero_is_the_period_start(self, converter):
        """``np.float32(0.0)`` anchors at the quarter starts."""
        result = converter.interpolate_to_higher_frequency(
            _QUARTERLY_2021, 'ME', anchor_fraction=np.float32(0.0)
        )
        # Valeurs d'or : ancres au 1er janvier, 1er avril, 1er juillet, 1er octobre ;
        # 31 janvier à 30 jours de la première sur 90 ; après le 1er octobre, rien
        # n'est comblé vers l'avant (cible 'E' → 'backward')
        expected = [10 + 10 * 30 / 90, 10 + 10 * 58 / 90, 10 + 10 * 89 / 90,
                    20 + 10 * 29 / 91, 20 + 10 * 60 / 91, 20 + 10 * 90 / 91,
                    30 + 10 * 30 / 92, 30 + 10 * 61 / 92, 30 + 10 * 91 / 92,
                    np.nan, np.nan, np.nan]
        assert result.tolist() == pytest.approx(expected, nan_ok=True)


class TestAnchorFractionPaths:
    """Method, target granularity, fallback and calendar lengths under ``anchor_fraction``."""

    @pytest.fixture
    def converter(self):
        """A fresh ``FrequencyConverter``."""
        return FrequencyConverter()

    def test_non_linear_method_is_kept(self, converter):
        """'nearest' is not turned into 'time': values stay on the observed levels."""
        result = converter.interpolate_to_higher_frequency(
            _QUARTERLY_2021, 'ME', method='nearest', anchor_fraction=0.5
        )
        # Valeurs d'or : ancres au 15 février, 16 mai, 16 août, 16 novembre ; chaque fin
        # de mois prend la valeur de l'ancre la plus proche en jours ; hors des ancres,
        # 'nearest' n'extrapole pas
        expected = [np.nan, 10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 40.0, np.nan, np.nan]
        assert result.tolist() == pytest.approx(expected, nan_ok=True)

    def test_sub_daily_target_keeps_the_time_of_day(self, converter):
        """An hourly target keeps the anchors at noon instead of midnight."""
        daily = pd.Series([1.0, 2.0], index=pd.date_range('2024-01-01', periods=2, freq='D'))
        result = converter.interpolate_to_higher_frequency(daily, 'h', anchor_fraction=0.5)
        # Valeurs d'or : ancres à midi ; 18 h est à 6 h de la première sur 24 → 1.25 ;
        # 6 h le lendemain → 1.75 ; avant midi le 1er, comblement arrière → 1.0
        hours = pd.DatetimeIndex(['2024-01-01 06:00', '2024-01-01 18:00', '2024-01-02 06:00'])
        assert result.loc[hours].tolist() == pytest.approx([1.0, 1.25, 1.75])

    def test_non_period_source_falls_back_on_none(self, converter):
        """A semi-monthly source has no ``pd.Period``: same output as ``anchor_fraction=None``."""
        semi_monthly = pd.Series(
            [1.0, 2.0, 3.0, 4.0], index=pd.date_range('2024-01-15', periods=4, freq='SME')
        )
        anchored = converter.interpolate_to_higher_frequency(semi_monthly, 'D', anchor_fraction=0.5)
        plain = converter.interpolate_to_higher_frequency(semi_monthly, 'D')
        pd.testing.assert_series_equal(anchored, plain)

    @pytest.mark.parametrize(
        'index',
        [
            pytest.param(pd.date_range('2023-12-31', periods=2, freq='YE'), id='year-end'),
            pytest.param(pd.date_range('2023-01-01', periods=2, freq='YS'), id='year-start'),
        ],
    )
    def test_source_position_does_not_matter(self, converter, index):
        """Year-start and year-end stamps anchor at the same middle of each year."""
        yearly = pd.Series([1.0, 2.0], index=index)
        result = converter.interpolate_to_higher_frequency(
            yearly, 'QE', anchor_fraction=0.5, limit_direction='both'
        )
        # Valeurs d'or : ancres au 2 juillet 2023 et au 2 juillet 2024, distantes de
        # 366 jours (29 février 2024) ; 30 sept. 2023 à 90 jours de la première
        expected = [1.0, 1.0, 1 + 90 / 366, 1 + 182 / 366, 1 + 273 / 366, 1 + 364 / 366, 2.0, 2.0]
        assert result.tolist() == pytest.approx(expected)

    def test_february_anchor_uses_its_own_length(self, converter):
        """The middle of February 2023 is the 15th (28 days), not the 16th."""
        month_ends = pd.date_range('2023-01-31', periods=3, freq='ME')
        monthly = pd.Series([31.0, 59.0, 90.0], index=month_ends)
        result = converter.interpolate_to_higher_frequency(
            monthly, 'D', anchor_fraction=0.5, limit_direction='both'
        )
        # Valeurs d'or : ancres au 16 janvier (15.5 j ramenés à minuit), 15 février
        # (14 j), 16 mars : chaque valeur observée se retrouve à son ancre
        anchors = pd.DatetimeIndex(['2023-01-16', '2023-02-15', '2023-03-16'])
        assert result.loc[anchors].tolist() == [31.0, 59.0, 90.0]
