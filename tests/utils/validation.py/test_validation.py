"""Tests unitaires pour les fonctions de validation des données temporelles."""
# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
# Module de test
import pytest

# Fonctions à tester
from tsforecast.utils.validation import validate_temporal_data, restore_original_structure


class TestValidateTemporalData:
    """Suite de tests pour validate_temporal_data."""

    def test_basic_validation_with_time_col(self):
        """Test validation basique avec time_col."""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value': [10, 20, 30, 40, 50]
        })

        validated = validate_temporal_data(df, time_col='date')

        # Vérification que l'index est bien un DatetimeIndex
        assert isinstance(validated.index, pd.DatetimeIndex)
        # Vérification que les colonnes ne contiennent plus 'date'
        assert 'date' not in validated.columns
        # Vérification que les valeurs sont préservées
        assert list(validated['value']) == [10, 20, 30, 40, 50]

    def test_validation_with_metadata(self):
        """Test validation avec return_metadata=True."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=4, freq='M'),
            'revenue': [1000, 1100, 1050, 1200],
            'users': [50, 55, 52, 60]
        })

        validated, metadata = validate_temporal_data(df, time_col='timestamp', return_metadata=True)

        # Vérification de la structure des métadonnées
        assert 'original_index' in metadata
        assert 'original_columns' in metadata
        assert 'time_col' in metadata
        assert metadata['time_col'] == 'timestamp'
        assert metadata['index_was_replaced'] is True

        # Vérification que l'original_index est bien un RangeIndex
        assert isinstance(metadata['original_index'], pd.RangeIndex)
        assert len(metadata['original_index']) == 4


class TestRestoreOriginalStructure:
    """Suite de tests pour restore_original_structure."""

    def test_basic_restoration(self):
        """Test restauration basique de la structure originale."""
        # Données originales
        df_original = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=4, freq='M'),
            'revenue': [1000, 1100, 1050, 1200],
            'users': [50, 55, 52, 60]
        })

        # Validation avec métadonnées
        validated, metadata = validate_temporal_data(df_original, time_col='timestamp', return_metadata=True)

        # Restauration
        restored = restore_original_structure(validated, metadata)

        # Vérifications
        # 1. Vérification du nom de la colonne temporelle
        assert 'timestamp' in restored.columns

        # 2. Vérification que les colonnes sont les mêmes
        assert set(restored.columns) == set(df_original.columns)

        # 3. Vérification que les valeurs sont préservées
        pd.testing.assert_series_equal(
            restored['revenue'].reset_index(drop=True),
            df_original['revenue'].reset_index(drop=True)
        )
        pd.testing.assert_series_equal(
            restored['users'].reset_index(drop=True),
            df_original['users'].reset_index(drop=True)
        )

        # 4. Vérification que l'index est restauré (RangeIndex)
        assert isinstance(restored.index, pd.RangeIndex)
        assert len(restored.index) == len(df_original.index)

    def test_restoration_preserves_dates(self):
        """Test que la restauration préserve les valeurs de dates."""
        # Données originales
        df_original = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value': [10, 20, 30, 40, 50]
        })

        # Validation avec métadonnées
        validated, metadata = validate_temporal_data(df_original, time_col='date', return_metadata=True)

        # Restauration
        restored = restore_original_structure(validated, metadata)

        # Vérification que les dates sont préservées
        pd.testing.assert_series_equal(
            restored['date'].reset_index(drop=True),
            df_original['date'].reset_index(drop=True)
        )

    def test_restoration_with_unsorted_data(self):
        """Test restauration avec données non triées initialement."""
        # Données originales non triées
        df_original = pd.DataFrame({
            'date': pd.to_datetime(['2023-03-01', '2023-01-01', '2023-04-01', '2023-02-01']),
            'value': [30, 10, 40, 20]
        })

        # Validation avec tri et métadonnées
        validated, metadata = validate_temporal_data(
            df_original,
            time_col='date',
            sort_data=True,
            return_metadata=True
        )

        # Vérification que validated est trié
        assert validated.index.is_monotonic_increasing

        # Restauration
        restored = restore_original_structure(validated, metadata)

        # Vérification que la colonne 'date' existe
        assert 'date' in restored.columns

        # Vérification que toutes les valeurs sont présentes (pas de NaN)
        assert not restored['value'].isna().any()
        assert not restored['date'].isna().any()

        # Vérification que toutes les dates originales sont présentes
        original_dates_sorted = sorted(df_original['date'].tolist())
        restored_dates_sorted = sorted(restored['date'].tolist())
        assert original_dates_sorted == restored_dates_sorted

    def test_restoration_with_panel_data(self):
        """Test restauration avec données de panel."""
        # Données de panel originales
        df_original = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=6, freq='D').tolist() * 2,
            'country': ['US'] * 6 + ['FR'] * 6,
            'value': range(12)
        })

        # Validation avec métadonnées
        validated, metadata = validate_temporal_data(
            df_original,
            time_col='date',
            panel_cols=['country'],
            return_metadata=True
        )

        # Restauration
        restored = restore_original_structure(validated, metadata)

        # Vérifications
        assert 'date' in restored.columns
        assert 'country' in restored.columns
        assert 'value' in restored.columns

        # Vérification que toutes les valeurs sont présentes
        assert not restored.isna().any().any()

    def test_restoration_after_transformations(self):
        """Test restauration après transformations sur les données validées."""
        # Données originales
        df_original = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=4, freq='M'),
            'revenue': [1000, 1100, 1050, 1200],
            'users': [50, 55, 52, 60]
        })

        # Validation avec métadonnées
        validated, metadata = validate_temporal_data(
            df_original,
            time_col='timestamp',
            return_metadata=True
        )

        # Ajout de colonnes transformées (simule du feature engineering)
        validated['revenue_change'] = validated['revenue'].pct_change()
        validated['users_change'] = validated['users'].diff()

        # Restauration
        restored = restore_original_structure(validated, metadata)

        # Vérifications
        # Les colonnes originales doivent être présentes
        assert 'timestamp' in restored.columns
        assert 'revenue' in restored.columns
        assert 'users' in restored.columns

        # Les colonnes transformées doivent aussi être présentes
        assert 'revenue_change' in restored.columns
        assert 'users_change' in restored.columns

        # Les valeurs doivent être préservées (pas de NaN sauf pour les transformations)
        assert not restored['revenue'].isna().any()
        assert not restored['users'].isna().any()
        assert not restored['timestamp'].isna().any()

    def test_restoration_with_series(self):
        """Test restauration avec une Series."""
        # Création d'un DataFrame pour validation
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5, freq='D'),
            'value': [10, 20, 30, 40, 50]
        })

        # Validation
        validated, metadata = validate_temporal_data(df, time_col='date', return_metadata=True)

        # Conversion en Series
        series_validated = validated['value']

        # Restauration
        restored = restore_original_structure(series_validated, metadata)

        # Vérification
        assert isinstance(restored, pd.Series)
        assert not restored.isna().any()
