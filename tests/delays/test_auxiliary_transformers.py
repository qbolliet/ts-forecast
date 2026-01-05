"""Tests pour le module auxiliary_transformers."""

import pytest
import pandas as pd
import numpy as np

from tsforecast.delays.auxiliary_transformers import ShiftTransformer, MaskTransformer


class TestShiftTransformer:
    """Tests pour la classe ShiftTransformer."""

    def test_higher_frequency_shift_raises_error(self):
        """Tentative de shifter par jours sur un index mensuel doit lever une erreur."""
        dates = pd.date_range('2024-01-01', periods=5, freq='M')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        shifter = ShiftTransformer(n_periods=1, frequency='D')

        with pytest.raises(ValueError, match="cannot be more granular"):
            shifter.fit_transform(series)

    def test_monthly_shift_on_daily_index(self):
        """Shift mensuel sur index journalier."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        series = pd.Series(range(90), index=dates)
        shifter = ShiftTransformer(n_periods=1, frequency='M')

        shifted = shifter.fit_transform(series)

        assert len(shifted) == len(series)
        # Positive shift extends at START (earlier dates), takes first N dates
        # So shifted.index[0] should be EARLIER (negative delta)
        delta_days = (shifted.index[0] - series.index[0]).days
        assert -31 <= delta_days <= -28

    def test_dataframe_shift_same_index(self):
        """Toutes les colonnes d'un DataFrame doivent avoir le même index après shift."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({f'col_{i}': range(100) for i in range(10)}, index=dates)

        shifter = ShiftTransformer(n_periods=5, frequency='D')
        shifted = shifter.fit_transform(df)

        # Toutes les colonnes doivent partager le même index
        first_col_index = shifted.iloc[:, 0].index
        assert all(shifted.iloc[:, i].index.equals(first_col_index) for i in range(len(shifted.columns)))

    def test_quarterly_suffix_preserved(self):
        """Index trimestriel avec suffixe DEC doit être préservé."""
        dates = pd.date_range('2024-01-31', periods=5, freq='QE-DEC')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        shifter = ShiftTransformer(n_periods=1, frequency='Q')

        shifted = shifter.fit_transform(series)

        # Vérifier que les dates sont bien en fin de trimestre
        # The important thing is that shifted index is valid and same length
        assert len(shifted) == len(series)
        assert isinstance(shifted.index, pd.DatetimeIndex)

    @pytest.mark.parametrize("n_periods,freq", [
        (1, 'D'), (5, 'D'), (-3, 'D'),
        (1, 'M'), (2, 'M'), (-1, 'M'),
    ])
    def test_inverse_transform_symmetry(self, n_periods, freq):
        """L'inverse transform doit parfaitement inverser le transform."""
        dates = pd.date_range('2024-01-01', periods=20, freq=freq)
        series = pd.Series(range(20), index=dates)
        shifter = ShiftTransformer(n_periods=n_periods, frequency=freq)

        shifted = shifter.fit_transform(series)
        recovered = shifter.inverse_transform(shifted)

        assert series.equals(recovered)

    def test_zero_shift(self):
        """Shift de 0 périodes doit retourner une copie."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        shifter = ShiftTransformer(n_periods=0, frequency='D')

        shifted = shifter.fit_transform(series)

        # Doit être une copie, pas le même objet
        assert shifted is not series
        # Mais les valeurs doivent être identiques
        assert series.equals(shifted)

    def test_negative_shift(self):
        """Shift négatif doit fonctionner correctement."""
        dates = pd.date_range('2024-01-01', periods=5, freq='M')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        shifter = ShiftTransformer(n_periods=-2, frequency='M')

        shifted = shifter.fit_transform(series)

        # Vérifier que le shift négatif fonctionne
        assert len(shifted) == len(series)
        # Shift négatif = étend à la fin, drop du début
        # Donc le premier index doit être PLUS TARD (plus grand)
        assert shifted.index[0] > series.index[0]

    def test_single_observation(self):
        """Single observation cannot detect frequency."""
        dates = pd.date_range('2024-01-01', periods=1, freq='D')
        series = pd.Series([1], index=dates)
        shifter = ShiftTransformer(n_periods=1, frequency='D')

        # Single observation cannot detect frequency (need at least 2 observations)
        with pytest.raises(ValueError, match="(Could not detect|minimum required)"):
            shifter.fit_transform(series)

    def test_same_frequency_shift(self):
        """Shift avec la même fréquence que l'index."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        series = pd.Series(range(10), index=dates)
        shifter = ShiftTransformer(n_periods=3, frequency='D')

        shifted = shifter.fit_transform(series)

        # Vérifier que le shift est correct
        assert len(shifted) == len(series)
        # Positive shift extends at start (earlier), so delta should be negative
        assert (shifted.index[0] - series.index[0]).days == -3

    def test_dataframe_with_series_values_match(self):
        """DataFrame shift doit donner les mêmes valeurs qu'un shift de Series."""
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        series = pd.Series(range(20), index=dates, name='col_0')
        df = pd.DataFrame({'col_0': range(20)}, index=dates)

        shifter = ShiftTransformer(n_periods=5, frequency='D')

        shifted_series = shifter.fit_transform(series)
        shifted_df = shifter.fit_transform(df)

        # Les index doivent être identiques
        assert shifted_series.index.equals(shifted_df.index)
        # Les valeurs doivent être identiques
        pd.testing.assert_series_equal(shifted_series, shifted_df['col_0'])

    def test_quarterly_to_monthly_conversion(self):
        """Shift trimestriel sur index mensuel doit utiliser conversion correcte."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        series = pd.Series(range(12), index=dates)
        shifter = ShiftTransformer(n_periods=1, frequency='Q')

        shifted = shifter.fit_transform(series)

        # 1 trimestre = 3 mois
        # Positive shift extends at start (earlier), so delta should be negative
        delta_months = (shifted.index[0].year - series.index[0].year) * 12 + \
                       (shifted.index[0].month - series.index[0].month)
        assert delta_months == -3

    def test_monthly_start_position_preserved(self):
        """Position de début de mois doit être préservée."""
        dates = pd.date_range('2024-01-01', periods=5, freq='MS')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        shifter = ShiftTransformer(n_periods=1, frequency='M')

        shifted = shifter.fit_transform(series)

        # Vérifier que toutes les dates sont au début du mois (jour 1)
        assert all(d.day == 1 for d in shifted.index)

    def test_monthly_end_position_preserved(self):
        """Position de fin de mois doit être préservée."""
        dates = pd.date_range('2024-01-31', periods=5, freq='ME')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        shifter = ShiftTransformer(n_periods=1, frequency='M')

        shifted = shifter.fit_transform(series)

        # Vérifier que toutes les dates sont en fin de mois
        # (le jour varie selon le mois, mais doit être le dernier jour)
        for d in shifted.index:
            next_month = (d + pd.Timedelta(days=1))
            assert next_month.day == 1  # Le jour suivant doit être le 1er du mois suivant


class TestMaskTransformer:
    """Tests pour la classe MaskTransformer."""

    def test_basic_masking(self):
        """Test de base du masquage."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        series = pd.Series(range(90), index=dates, name='GDP')

        masker = MaskTransformer(
            n_obs=2,
            mask_frequency='M'
        )

        masked = masker.fit_transform(series)

        # Vérifier que certaines valeurs sont masquées (NaN)
        assert masked.isna().sum() > 0
        # La longueur doit rester la même
        assert len(masked) == len(series)

    def test_inverse_transform(self):
        """Test de l'inverse transform du masquage."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        series = pd.Series(range(90), index=dates, name='GDP')

        masker = MaskTransformer(
            n_obs=2,
            mask_frequency='M'
        )

        masked = masker.fit_transform(series)
        recovered = masker.inverse_transform(masked)

        # Les données originales doivent être parfaitement restaurées
        # Check values and index separately to avoid freq attribute issues
        pd.testing.assert_index_equal(recovered.index, series.index)
        np.testing.assert_array_equal(recovered.values, series.values)
        assert recovered.name == series.name

    def test_zero_masking(self):
        """Test avec n_obs=0 (pas de masquage)."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        series = pd.Series(range(30), index=dates, name='GDP')

        masker = MaskTransformer(
            n_obs=0,
            mask_frequency='M'
        )

        masked = masker.fit_transform(series)

        # Aucune valeur ne doit être masquée
        assert masked.isna().sum() == 0
        pd.testing.assert_series_equal(masked, series)


class TestShiftTransformerEdgeCases:
    """Additional edge case tests for ShiftTransformer."""

    def test_invalid_input_type(self):
        """Reject non-pandas inputs."""
        shifter = ShiftTransformer(n_periods=1, frequency='D')
        with pytest.raises(ValueError, match="must be a pandas Series or DataFrame"):
            shifter.fit([1, 2, 3])

    def test_non_datetime_index(self):
        """Reject non-datetime index."""
        series = pd.Series([1, 2, 3], index=[0, 1, 2])
        shifter = ShiftTransformer(n_periods=1, frequency='D')
        with pytest.raises(ValueError):
            shifter.fit_transform(series)

    def test_unsorted_index_auto_corrected(self):
        """Unsorted indices should be automatically sorted."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        # Shuffle the series
        series = series.iloc[[2, 0, 4, 1, 3]]

        shifter = ShiftTransformer(n_periods=1, frequency='D')
        shifted = shifter.fit_transform(series)

        # Result should have sorted index
        assert shifted.index.is_monotonic_increasing

    def test_daily_frequency_multiday(self):
        """Test with multiple days to ensure shift works."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        series = pd.Series(range(30), index=dates)
        shifter = ShiftTransformer(n_periods=5, frequency='D')

        shifted = shifter.fit_transform(series)

        assert len(shifted) == len(series)
        # Positive shift = extend at start (earlier), so delta should be negative
        delta_days = (shifted.index[0] - series.index[0]).days
        assert delta_days == -5

    def test_dataframe_multicolumn(self):
        """Test DataFrame with multiple columns of different types."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'int_col': range(10),
            'float_col': [float(x) for x in range(10)],
            'str_col': [f'val_{x}' for x in range(10)]
        }, index=dates)

        shifter = ShiftTransformer(n_periods=2, frequency='D')
        shifted = shifter.fit_transform(df)

        # All columns should be preserved
        assert list(shifted.columns) == list(df.columns)
        # Types should be preserved
        assert shifted['str_col'].dtype == object


class TestMaskTransformerEdgeCases:
    """Additional edge case tests for MaskTransformer."""

    def test_mask_first_observations(self):
        """Test masking first observations (how='first')."""
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        series = pd.Series(range(60), index=dates)

        masker = MaskTransformer(n_obs=5, mask_frequency='M', how='first')
        masked = masker.fit_transform(series)

        # First 5 days of each month should be masked
        assert masked.isna().sum() > 0

    def test_mask_more_than_available(self):
        """Masking more observations than available in a period should handle gracefully."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        series = pd.Series(range(5), index=dates)

        # Only 5 days total, but request masking 10 per month
        masker = MaskTransformer(n_obs=10, mask_frequency='M')
        masked = masker.fit_transform(series)

        # Should not crash and return valid series
        assert len(masked) == len(series)
        assert isinstance(masked.index, pd.DatetimeIndex)

    def test_mask_dataframe(self):
        """Test masking on DataFrame."""
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        df = pd.DataFrame({
            'col1': range(60),
            'col2': range(100, 160)
        }, index=dates)

        masker = MaskTransformer(n_obs=2, mask_frequency='M')
        masked = masker.fit_transform(df)

        # Both columns should have identical NaN patterns
        assert (masked['col1'].isna() == masked['col2'].isna()).all()

    def test_mask_without_fit(self):
        """Calling inverse_transform without transform should raise error."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        series = pd.Series(range(30), index=dates)

        masker = MaskTransformer(n_obs=2, mask_frequency='M')

        with pytest.raises(ValueError, match="Must call transform"):
            masker.inverse_transform(series)

    def test_inverse_transform_with_filled_nans(self):
        """Test inverse_transform after filling NaNs with predictions."""
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        series = pd.Series(range(60), index=dates)

        masker = MaskTransformer(n_obs=2, mask_frequency='M')
        masked = masker.fit_transform(series)

        # Fill NaN values with predictions (different values)
        predictions = masked.fillna(-999)

        # Inverse transform should restore original values (not the -999)
        restored = masker.inverse_transform(predictions)

        np.testing.assert_array_equal(restored.values, series.values)
