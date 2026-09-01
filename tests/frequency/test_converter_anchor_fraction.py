"""Tests unitaires pour `interpolate_to_higher_frequency(anchor_fraction=...)`.

Ce module couvre la position d'ancrage de la valeur dans sa période (prérequis
P2 de `HighFrequencyImputer2`, §10.2 de la spécification d'architecture) : le
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
