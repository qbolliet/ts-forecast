"""Tests for delays parameter formats (dict and DataFrame)."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from tsforecast.delays.transformers import (
    PublicationDelayTransformer,
    ReleaseDelayTransformer,  # Alias
    delays_dict_to_dataframe,
    delays_dataframe_to_dict,
    _validate_delays_parameter
)


@pytest.fixture
def sample_time_series_data():
    """Crée des données de série temporelle pour les tests."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    data = {
        'PIB': np.random.randn(len(dates)) + 100,
        'inflation': np.random.randn(len(dates)) + 2.5,
        'unemployment': np.random.randn(len(dates)) + 5.0
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'date'
    return df


@pytest.fixture
def sample_panel_data():
    """Crée des données de panel pour les tests."""
    countries = ['France', 'Germany', 'Italy']
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='MS')  # Monthly

    data = []
    for country in countries:
        for date in dates:
            data.append({
                'country': country,
                'date': date,
                'PIB': np.random.randn() + 100,
                'inflation': np.random.randn() + 2.5
            })

    return pd.DataFrame(data)


@pytest.fixture
def sample_delays_dict():
    """Crée un dictionnaire de délais exemple."""
    return {
        'PIB': 45.0,
        'inflation': 30.0,
        'unemployment': 15.0
    }


@pytest.fixture
def sample_panel_delays_dict():
    """Crée un dictionnaire de délais pour données panel."""
    return {
        ('France', 'PIB'): 45.0,
        ('Germany', 'PIB'): 38.0,
        ('Italy', 'PIB'): 50.0,
        ('France', 'inflation'): 30.0,
        ('Germany', 'inflation'): 25.0,
        ('Italy', 'inflation'): 35.0
    }


@pytest.fixture
def sample_delays_dataframe():
    """Crée un DataFrame de délais (format calculate_applicable_delay)."""
    data = {
        'applicable_delay': [45.0, 30.0, 15.0],
        'unit': ['D', 'D', 'D']
    }
    df = pd.DataFrame(data, index=pd.Index(['PIB', 'inflation', 'unemployment'], name='indicator'))
    return df


@pytest.fixture
def sample_panel_delays_dataframe():
    """Crée un DataFrame de délais panel (format calculate_applicable_delay)."""
    data = {
        'applicable_delay': [45.0, 38.0, 50.0, 30.0, 25.0, 35.0],
        'unit': ['D'] * 6
    }
    index = pd.MultiIndex.from_tuples([
        ('France', 'PIB'),
        ('Germany', 'PIB'),
        ('Italy', 'PIB'),
        ('France', 'inflation'),
        ('Germany', 'inflation'),
        ('Italy', 'inflation')
    ], names=['entity', 'indicator'])
    df = pd.DataFrame(data, index=index)
    return df


class TestDelaysConversionFunctions:
    """Tests pour les fonctions de conversion dict/DataFrame."""

    def test_dict_to_dataframe_simple(self, sample_delays_dict):
        """Test conversion dict -> DataFrame (série temporelle simple)."""
        df = delays_dict_to_dataframe(sample_delays_dict, unit='D')

        assert isinstance(df, pd.DataFrame)
        assert 'applicable_delay' in df.columns
        assert 'unit' in df.columns
        assert df.index.name == 'indicator'
        assert len(df) == 3
        assert df.loc['PIB', 'applicable_delay'] == 45.0
        assert (df['unit'] == 'D').all()

    def test_dict_to_dataframe_panel(self, sample_panel_delays_dict):
        """Test conversion dict -> DataFrame (données panel)."""
        df = delays_dict_to_dataframe(sample_panel_delays_dict, unit='D')

        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ['entity', 'indicator']
        assert len(df) == 6
        assert df.loc[('France', 'PIB'), 'applicable_delay'] == 45.0

    def test_dataframe_to_dict_simple(self, sample_delays_dataframe):
        """Test conversion DataFrame -> dict (série temporelle simple)."""
        delays_dict = delays_dataframe_to_dict(sample_delays_dataframe)

        assert isinstance(delays_dict, dict)
        assert len(delays_dict) == 3
        assert delays_dict['PIB'] == 45.0
        assert delays_dict['inflation'] == 30.0

    def test_dataframe_to_dict_panel(self, sample_panel_delays_dataframe):
        """Test conversion DataFrame -> dict (données panel)."""
        delays_dict = delays_dataframe_to_dict(sample_panel_delays_dataframe)

        assert isinstance(delays_dict, dict)
        assert len(delays_dict) == 6
        assert delays_dict[('France', 'PIB')] == 45.0
        assert delays_dict[('Germany', 'inflation')] == 25.0

    def test_roundtrip_conversion(self, sample_delays_dict):
        """Test dict -> DataFrame -> dict préserve les valeurs."""
        df = delays_dict_to_dataframe(sample_delays_dict)
        delays_dict_restored = delays_dataframe_to_dict(df)

        assert delays_dict_restored == sample_delays_dict

    def test_roundtrip_panel_conversion(self, sample_panel_delays_dict):
        """Test dict -> DataFrame -> dict (panel) préserve les valeurs."""
        df = delays_dict_to_dataframe(sample_panel_delays_dict)
        delays_dict_restored = delays_dataframe_to_dict(df)

        assert delays_dict_restored == sample_panel_delays_dict

    def test_dataframe_to_dict_missing_column(self):
        """Test erreur si colonne applicable_delay absente."""
        df = pd.DataFrame({'delay': [45.0, 30.0]}, index=['PIB', 'inflation'])

        with pytest.raises(ValueError, match="must contain 'applicable_delay' column"):
            delays_dataframe_to_dict(df)


class TestDelaysValidation:
    """Tests pour la validation du paramètre delays."""

    def test_validate_dict_valid(self, sample_delays_dict):
        """Test validation d'un dictionnaire valide."""
        # Ne devrait pas lever d'erreur
        _validate_delays_parameter(sample_delays_dict, panel_cols=None)

    def test_validate_dataframe_valid(self, sample_delays_dataframe):
        """Test validation d'un DataFrame valide."""
        # Ne devrait pas lever d'erreur
        _validate_delays_parameter(sample_delays_dataframe, panel_cols=None)

    def test_validate_dict_empty_warning(self):
        """Test warning sur dictionnaire vide."""
        with pytest.warns(UserWarning, match="Empty delays dictionary"):
            _validate_delays_parameter({}, panel_cols=None)

    def test_validate_dict_non_numeric_value(self):
        """Test erreur sur valeur non-numérique."""
        delays = {'PIB': 'invalid'}

        with pytest.raises(ValueError, match="must be numeric"):
            _validate_delays_parameter(delays, panel_cols=None)

    def test_validate_dataframe_missing_column(self):
        """Test erreur DataFrame sans applicable_delay."""
        df = pd.DataFrame({'delay': [45.0]}, index=['PIB'])

        with pytest.raises(ValueError, match="must contain 'applicable_delay' column"):
            _validate_delays_parameter(df, panel_cols=None)

    def test_validate_dataframe_non_numeric(self):
        """Test erreur DataFrame avec valeurs non-numériques."""
        df = pd.DataFrame(
            {'applicable_delay': ['invalid']},
            index=['PIB']
        )

        with pytest.raises(ValueError, match="must contain numeric values"):
            _validate_delays_parameter(df, panel_cols=None)

    def test_validate_invalid_type(self):
        """Test erreur sur type invalide."""
        with pytest.raises(TypeError, match="delays must be Dict or DataFrame"):
            _validate_delays_parameter("invalid", panel_cols=None)


class TestPublicationDelayTransformer:
    """Tests pour PublicationDelayTransformer avec différents formats."""

    def test_initialization_with_dict(self, sample_delays_dict):
        """Test initialisation avec dictionnaire."""
        transformer = PublicationDelayTransformer(
            delays=sample_delays_dict,
            mode='shift',
            prediction_date='2023-12-01'
        )

        assert transformer.delays == sample_delays_dict
        assert transformer.mode == 'shift'

    def test_initialization_with_dataframe(self, sample_delays_dataframe):
        """Test initialisation avec DataFrame."""
        transformer = PublicationDelayTransformer(
            delays=sample_delays_dataframe,
            mode='shift',
            prediction_date='2023-12-01'
        )

        assert isinstance(transformer.delays, pd.DataFrame)
        assert 'applicable_delay' in transformer.delays.columns

    def test_fit_with_dict(self, sample_time_series_data, sample_delays_dict):
        """Test fit avec dictionnaire de délais."""
        transformer = PublicationDelayTransformer(
            delays=sample_delays_dict,
            mode='shift',
            prediction_date='2023-12-01'
        )

        transformer.fit(sample_time_series_data)

        assert transformer.delays_dict_ is not None
        assert isinstance(transformer.delays_dict_, dict)
        assert transformer.delays_dict_['PIB'] == 45.0

    def test_fit_with_dataframe(self, sample_time_series_data, sample_delays_dataframe):
        """Test fit avec DataFrame de délais."""
        transformer = PublicationDelayTransformer(
            delays=sample_delays_dataframe,
            mode='shift',
            prediction_date='2023-12-01'
        )

        transformer.fit(sample_time_series_data)

        assert transformer.delays_dict_ is not None
        assert isinstance(transformer.delays_dict_, dict)
        assert transformer.delays_dict_['PIB'] == 45.0

    def test_get_delays_as_dataframe(self, sample_time_series_data, sample_delays_dict):
        """Test récupération des délais au format DataFrame."""
        transformer = PublicationDelayTransformer(
            delays=sample_delays_dict,
            mode='shift',
            prediction_date='2023-12-01'
        )

        transformer.fit(sample_time_series_data)
        delays_df = transformer.get_delays_as_dataframe()

        assert isinstance(delays_df, pd.DataFrame)
        assert 'applicable_delay' in delays_df.columns
        assert 'unit' in delays_df.columns
        assert len(delays_df) == 3
        assert delays_df.loc['PIB', 'applicable_delay'] == 45.0

    def test_get_delays_as_dataframe_not_fitted(self, sample_delays_dict):
        """Test erreur si get_delays_as_dataframe appelé avant fit."""
        transformer = PublicationDelayTransformer(
            delays=sample_delays_dict,
            mode='shift',
            prediction_date='2023-12-01'
        )

        with pytest.raises(RuntimeError, match="has not been fitted"):
            transformer.get_delays_as_dataframe()

    def test_alias_works(self, sample_delays_dict):
        """Test que l'alias ReleaseDelayTransformer fonctionne."""
        transformer = ReleaseDelayTransformer(
            delays=sample_delays_dict,
            mode='shift',
            prediction_date='2023-12-01'
        )

        assert isinstance(transformer, PublicationDelayTransformer)
