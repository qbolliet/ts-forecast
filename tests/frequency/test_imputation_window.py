"""Tests ciblés pour ImputationWindowCalculator (cf. high_frequency_imputer_review.md §3, §7).

Chaque test épingle un comportement précis identifié dans la revue. Les tests
marqués ``xfail(strict=True)`` documentent le comportement souhaité (pas le
comportement actuel bogué) et référencent la section de la revue concernée.
"""
import numpy as np
import pandas as pd
import pytest

from tsforecast.frequency.imputation_window import ImputationWindowCalculator


class TestGetImputationWindowMaskAlignment:
    """§1.3 : get_imputation_window_mask(data) aligne le masque sur data.index."""

    def test_get_imputation_window_mask_aligned_to_data(self):
        """Le masque aligné a exactement l'index des données, False hors grille."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        df = pd.DataFrame({
            'a': np.arange(24, dtype=float),
            'b': np.arange(24, dtype=float) * 2,
        }, index=dates)
        calc = ImputationWindowCalculator(coverage_threshold=0.5, imputation_scope='strict')
        calc.fit(df)

        # Données à aligner : l'index d'origine plus une date hors de la grille ajustée
        extra_dates = dates.append(pd.DatetimeIndex(['2022-06-01']))
        data_to_align = pd.DataFrame(index=extra_dates)

        aligned = calc.get_imputation_window_mask(data_to_align)

        # Index identique à celui des données passées en argument
        pd.testing.assert_index_equal(aligned.index, data_to_align.index)
        # La date hors grille est nécessairement False
        assert aligned.loc[pd.Timestamp('2022-06-01')] == False
        # Les dates de la fenêtre stricte (couverture totale) sont True
        assert aligned.loc[dates].all()


class TestExtensionContiguity:
    """§3.2 : l'extension du masque doit s'arrêter au premier trou de couverture."""

    @pytest.mark.xfail(
        strict=True,
        reason="§3.2 high_frequency_imputer_review.md : _extend_backward active toutes "
               "les dates antérieures dont la couverture dépasse le seuil, même séparées "
               "de la fenêtre stricte par un trou sous le seuil (extension non contiguë).",
    )
    def test_extension_stops_at_first_gap(self):
        """extended_backward n'active rien au-delà d'un trou sous le seuil."""
        dates = pd.date_range('2020-01-01', periods=20, freq='MS')
        df = pd.DataFrame({'a': np.nan, 'b': np.nan}, index=dates, dtype=float)

        # Fenêtre stricte : couverture totale sur [10, 15)
        df.loc[dates[10:15], 'a'] = 1.0
        df.loc[dates[10:15], 'b'] = 1.0
        # Avant la fenêtre, au-delà d'un trou : couverture 50% (>= seuil) sur [0, 4)
        df.loc[dates[0:4], 'a'] = 1.0
        # Trou : couverture nulle (< seuil) sur [4, 8) -- doit bloquer l'extension
        # Immédiatement avant la fenêtre stricte : couverture 50% sur [8, 10)
        df.loc[dates[8:10], 'a'] = 1.0

        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='extended_backward', min_columns=2
        )
        calc.fit(df)

        extended_dates = calc.imputation_window_mask_.index[calc.imputation_window_mask_]

        # L'extension doit s'arrêter au trou : les dates [0, 4) restent hors fenêtre
        assert not calc.imputation_window_mask_.loc[dates[0:4]].any()
        # Les dates [8, 10) juste avant la fenêtre stricte sont, elles, incluses
        assert calc.imputation_window_mask_.loc[dates[8:10]].all()


class TestMaskAtFrequency:
    """§3.3 : get_mask_at_frequency ne doit pas systématiquement retourner False."""

    def test_mask_at_frequency_full_year_is_true(self):
        """12 mois couverts -> le masque annuel vaut True (bug du floor/12.1667)."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        df = pd.DataFrame({
            'a': np.arange(24, dtype=float),
            'b': np.arange(24, dtype=float) * 2,
        }, index=dates)
        calc = ImputationWindowCalculator(coverage_threshold=0.5, imputation_scope='strict')
        calc.fit(df)

        mask_year = calc.get_mask_at_frequency('YS')

        # Les deux années 2020 et 2021 sont intégralement couvertes par le masque mensuel
        assert mask_year.loc[pd.Timestamp('2020-01-01')] == True
        assert mask_year.loc[pd.Timestamp('2021-01-01')] == True


class TestColumnCoveragePanel:
    """§3.1 : column_coverage_ doit contenir une entrée par entité pour un panel."""

    @pytest.mark.xfail(
        strict=True,
        reason="§3.1 high_frequency_imputer_review.md : _fit_panel écrase "
               "self.column_coverage_ à chaque entité au lieu de l'indexer par entité "
               "(hfi/iwc:L279), seule la dernière entité survit après fit().",
    )
    def test_column_coverage_is_per_entity_for_panel(self):
        """column_coverage_ contient une entrée par entité, pas la dernière seule."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        entities = ['A', 'B']
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        df = pd.DataFrame({
            'a': np.arange(48, dtype=float),
            'b': np.arange(48, dtype=float) * 2,
        }, index=idx)

        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        calc.fit(df)

        # Un dict par entité (clés tuples), pas un unique dict de colonnes
        assert set(calc.column_coverage_.keys()) == {('A',), ('B',)}
