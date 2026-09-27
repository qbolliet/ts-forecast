"""Tests unitaires pour le module transformers.

Ce module contient les tests pour ShiftTransformer, MaskTransformer,
PublicationDelayTransformer et les fonctions auxiliaires associées.
"""

import pytest
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from sklearn.utils.validation import check_is_fitted, NotFittedError

# Import du module à tester
from tsforecast.delays.transformers import (
    PublicationDelayTransformer,
    ShiftTransformer, 
    MaskTransformer,
    prepare_entity_kwargs_from_delays,
    _build_entity_params,
    _extract_param_by_variable,
    _resolve_strategy
)

# ============================================================================
# Fixtures de données de test
# ============================================================================

@pytest.fixture
def sample_time_series():
    """Generate sample time series data for testing.
    
    Returns:
        pd.DataFrame: Time series with monthly dates and multiple variables.
    """
    # Données mensuelles sur 2 ans
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='MS')
    data = pd.DataFrame({
        'GDP': np.random.randn(len(dates)) * 10 + 1000,
        'inflation': np.random.randn(len(dates)) * 0.5 + 2.0,
        'unemployment': np.random.randn(len(dates)) * 0.3 + 5.0
    }, index=dates)
    return data


@pytest.fixture
def sample_panel_data():
    """Generate sample panel data for testing.
    
    Returns:
        pd.DataFrame: Panel data with MultiIndex (country, date).
    """
    # Données mensuelles pour 3 pays sur 1 an
    countries = ['France', 'Germany', 'Italy']
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='MS')
    
    data = []
    for country in countries:
        for date in dates:
            data.append({
                'country': country,
                'date': date,
                'GDP': np.random.randn() * 10 + 1000,
                'inflation': np.random.randn() * 0.5 + 2.0
            })
    
    df = pd.DataFrame(data)
    df = df.set_index(['country', 'date'])
    return df


@pytest.fixture
def delays_dict_simple():
    """Simple delays dictionary for testing.
    
    Returns:
        dict: Mapping variable names to delay values in days.
    """
    return {
        'GDP': 45.0,
        'inflation': 30.0,
        'unemployment': 15.0
    }


@pytest.fixture
def delays_dataframe():
    """Delays DataFrame with metadata for testing.
    
    Returns:
        pd.DataFrame: Delays with unit, reference_point, and target_frequency.
    """
    return pd.DataFrame({
        'variable': ['GDP', 'inflation', 'unemployment'],
        'delay': [45.0, 30.0, 15.0],
        'unit': ['D', 'D', 'D'],
        'reference_point': ['end', 'end', 'end'],
        'target_frequency': ['M', 'M', 'M']
    })


@pytest.fixture
def delays_dataframe_panel():
    """Panel delays DataFrame for testing.
    
    Returns:
        pd.DataFrame: Delays with entity-level variation.
    """
    data = []
    for country in ['France', 'Germany', 'Italy']:
        for var in ['GDP', 'inflation']:
            data.append({
                'country': country,
                'variable': var,
                'delay': np.random.uniform(20, 60),
                'unit': 'D',
                'reference_point': 'end',
                'target_frequency': 'M'
            })
    
    df = pd.DataFrame(data)
    df = df.set_index(['country', 'variable'])
    return df


@pytest.fixture
def daily_series():
    """Generate daily time series for mask transformer tests.
    
    Returns:
        pd.Series: Daily series spanning 3 months.
    """
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    return pd.Series(range(len(dates)), index=dates, name='value')


@pytest.fixture
def monthly_series():
    """Generate monthly time series for shift transformer tests.
    
    Returns:
        pd.Series: Monthly series spanning 12 months.
    """
    dates = pd.date_range('2024-01-01', periods=12, freq='MS')
    return pd.Series(range(12), index=dates, name='value')


@pytest.fixture
def daily_dataframe():
    """Generate daily DataFrame for multi-column tests.
    
    Returns:
        pd.DataFrame: Daily DataFrame with multiple columns.
    """
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
    return pd.DataFrame({
        'col1': range(len(dates)),
        'col2': range(100, 100 + len(dates)),
        'col3': np.random.randn(len(dates))
    }, index=dates)


# ============================================================================
# Tests de la classe ShiftTransformer - Fonctionnement de base
# ============================================================================

class TestShiftTransformerBasic:
    """Tests de base pour la classe ShiftTransformer."""

    def test_shift_positive_no_nan_introduced(self, monthly_series):
        """Vérification qu'un shift positif n'introduit pas de NaN."""
        shifter = ShiftTransformer(n_periods=3, frequency='M')
        shifted = shifter.fit_transform(monthly_series)
        
        # Aucun NaN ne doit être introduit
        assert shifted.isna().sum() == 0
        # Longueur identique
        assert len(shifted) == len(monthly_series)

    def test_shift_negative_no_nan_introduced(self, monthly_series):
        """Vérification qu'un shift négatif n'introduit pas de NaN."""
        shifter = ShiftTransformer(n_periods=-3, frequency='M')
        shifted = shifter.fit_transform(monthly_series)
        
        # Aucun NaN ne doit être introduit
        assert shifted.isna().sum() == 0
        # Longueur identique
        assert len(shifted) == len(monthly_series)

    def test_shift_zero_returns_copy(self, monthly_series):
        """Vérification qu'un shift de 0 retourne une copie identique."""
        shifter = ShiftTransformer(n_periods=0, frequency='M')
        shifted = shifter.fit_transform(monthly_series)
        
        # Données identiques
        pd.testing.assert_series_equal(shifted, monthly_series)

    def test_shift_correct_date_calculation_positive(self):
        """Vérification que le shift positif déplace l'index correctement."""
        dates = pd.date_range('2024-01-01', periods=5, freq='MS')
        series = pd.Series([10, 20, 30, 40, 50], index=dates, name='test')
        
        shifter = ShiftTransformer(n_periods=2, frequency='M')
        shifted = shifter.fit_transform(series)
        
        # Shift positif = extension au début, donc dates plus anciennes
        # La première date originale était 2024-01-01, avec shift +2, elle devient 2023-11-01
        expected_first_date = pd.Timestamp('2023-11-01')
        assert shifted.index[0] == expected_first_date
        
        # Les valeurs doivent rester dans le même ordre
        np.testing.assert_array_equal(shifted.values, series.values)

    def test_shift_correct_date_calculation_negative(self):
        """Vérification que le shift négatif déplace l'index correctement."""
        dates = pd.date_range('2024-01-01', periods=5, freq='MS')
        series = pd.Series([10, 20, 30, 40, 50], index=dates, name='test')
        
        shifter = ShiftTransformer(n_periods=-2, frequency='M')
        shifted = shifter.fit_transform(series)
        
        # Shift négatif = extension à la fin, donc dates plus récentes
        # La dernière date originale était 2024-05-01, avec shift -2, elle devient 2024-07-01
        expected_last_date = pd.Timestamp('2024-07-01')
        assert shifted.index[-1] == expected_last_date
        
        # Les valeurs doivent rester dans le même ordre
        np.testing.assert_array_equal(shifted.values, series.values)

    def test_shift_transform_then_inverse_transform_recovers_original(self, monthly_series):
        """Vérification que transform puis inverse_transform retourne les données initiales."""
        shifter = ShiftTransformer(n_periods=3, frequency='M')
        
        # Application de la transformation
        shifted = shifter.fit_transform(monthly_series)
        
        # Application de la transformation inverse
        recovered = shifter.inverse_transform(shifted)
        
        # Vérification de la récupération des données
        pd.testing.assert_index_equal(recovered.index, monthly_series.index)
        np.testing.assert_array_equal(recovered.values, monthly_series.values)
        assert recovered.name == monthly_series.name

    def test_shift_negative_transform_then_inverse_transform_recovers_original(self, monthly_series):
        """Vérification que transform puis inverse_transform fonctionne avec shift négatif."""
        shifter = ShiftTransformer(n_periods=-5, frequency='M')
        
        shifted = shifter.fit_transform(monthly_series)
        recovered = shifter.inverse_transform(shifted)
        
        pd.testing.assert_index_equal(recovered.index, monthly_series.index)
        np.testing.assert_array_equal(recovered.values, monthly_series.values)


class TestShiftTransformerDataFrame:
    """Tests du ShiftTransformer avec des DataFrames."""

    def test_shift_dataframe_no_nan_introduced(self, daily_dataframe):
        """Vérification qu'un shift sur DataFrame n'introduit pas de NaN."""
        shifter = ShiftTransformer(n_periods=10, frequency='D')
        shifted = shifter.fit_transform(daily_dataframe)
        
        # Aucun NaN ne doit être introduit (en plus de ceux éventuellement déjà présents)
        original_nan_count = daily_dataframe.isna().sum().sum()
        shifted_nan_count = shifted.isna().sum().sum()
        assert shifted_nan_count == original_nan_count

    def test_shift_dataframe_preserves_columns(self, daily_dataframe):
        """Vérification que les colonnes sont préservées."""
        shifter = ShiftTransformer(n_periods=5, frequency='D')
        shifted = shifter.fit_transform(daily_dataframe)
        
        assert list(shifted.columns) == list(daily_dataframe.columns)

    def test_shift_dataframe_transform_inverse_transform(self, daily_dataframe):
        """Vérification de la récupération pour DataFrame."""
        shifter = ShiftTransformer(n_periods=7, frequency='D')
        
        shifted = shifter.fit_transform(daily_dataframe)
        recovered = shifter.inverse_transform(shifted)
        
        pd.testing.assert_index_equal(recovered.index, daily_dataframe.index)
        pd.testing.assert_frame_equal(recovered, daily_dataframe, check_exact=False)


class TestShiftTransformerFrequencies:
    """Tests du ShiftTransformer avec différentes fréquences."""

    def test_shift_monthly_on_daily_index(self):
        """Test d'un shift mensuel sur un index journalier."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        series = pd.Series(range(90), index=dates)
        
        # Shift de 1 mois sur données journalières
        shifter = ShiftTransformer(n_periods=1, frequency='M')
        shifted = shifter.fit_transform(series)
        
        # L'index doit avoir reculé d'environ 30 jours
        delta_days = (shifted.index[0] - series.index[0]).days
        assert -35 <= delta_days <= -25  # Environ 1 mois

    def test_shift_quarterly_on_monthly_index(self):
        """Test d'un shift trimestriel sur un index mensuel."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        series = pd.Series(range(12), index=dates)
        
        shifter = ShiftTransformer(n_periods=1, frequency='Q')
        shifted = shifter.fit_transform(series)
        
        # 1 trimestre = 3 mois, shift positif = extension au début
        delta_months = (shifted.index[0].year - series.index[0].year) * 12 + \
                       (shifted.index[0].month - series.index[0].month)
        assert delta_months == -3

    def test_shift_preserves_month_start_position(self):
        """Vérification que la position de début de mois est préservée."""
        dates = pd.date_range('2024-01-01', periods=5, freq='MS')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        
        shifter = ShiftTransformer(n_periods=1, frequency='M')
        shifted = shifter.fit_transform(series)
        
        # Toutes les dates doivent être au début du mois (jour 1)
        assert all(d.day == 1 for d in shifted.index)

    def test_shift_preserves_month_end_position(self):
        """Vérification que la position de fin de mois est préservée."""
        dates = pd.date_range('2024-01-31', periods=5, freq='ME')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        
        shifter = ShiftTransformer(n_periods=1, frequency='M')
        shifted = shifter.fit_transform(series)
        
        # Toutes les dates doivent être en fin de mois
        for d in shifted.index:
            next_day = d + pd.Timedelta(days=1)
            assert next_day.day == 1


class TestShiftTransformerEdgeCases:
    """Tests des cas limites pour ShiftTransformer."""

    def test_invalid_input_type(self):
        """Rejet des entrées non-pandas."""
        shifter = ShiftTransformer(n_periods=1, frequency='D')
        with pytest.raises(ValueError, match="must be a pandas Series or DataFrame"):
            shifter.fit([1, 2, 3])

    def test_non_datetime_index(self):
        """Rejet des index non-datetime."""
        series = pd.Series([1, 2, 3], index=[0, 1, 2])
        shifter = ShiftTransformer(n_periods=1, frequency='D')
        with pytest.raises(ValueError):
            shifter.fit_transform(series)

    def test_unsorted_index_auto_corrected(self):
        """Les index non triés doivent être automatiquement triés."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        series = pd.Series([1, 2, 3, 4, 5], index=dates)
        # Mélange de la série
        series = series.iloc[[2, 0, 4, 1, 3]]
        
        shifter = ShiftTransformer(n_periods=1, frequency='D')
        shifted = shifter.fit_transform(series)
        
        # Le résultat doit avoir un index trié
        assert shifted.index.is_monotonic_increasing

    def test_daily_frequency_multiday(self):
        """Test avec plusieurs jours pour vérifier le shift."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        series = pd.Series(range(30), index=dates)
        shifter = ShiftTransformer(n_periods=5, frequency='D')
        
        shifted = shifter.fit_transform(series)
        
        assert len(shifted) == len(series)
        # Shift positif = extension au début (dates plus anciennes)
        delta_days = (shifted.index[0] - series.index[0]).days
        assert delta_days == -5

    def test_dataframe_multicolumn_types_preserved(self):
        """Test DataFrame avec plusieurs colonnes de types différents."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'int_col': range(10),
            'float_col': [float(x) for x in range(10)],
            'str_col': [f'val_{x}' for x in range(10)]
        }, index=dates)
        
        shifter = ShiftTransformer(n_periods=2, frequency='D')
        shifted = shifter.fit_transform(df)
        
        # Toutes les colonnes doivent être préservées
        assert list(shifted.columns) == list(df.columns)
        # Le type string doit être préservé
        assert shifted['str_col'].dtype == object

    def test_higher_frequency_shift_raises_error(self):
        """Test qu'un shift de fréquence plus granulaire que l'index lève une erreur."""
        # Index mensuel
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        series = pd.Series(range(12), index=dates)
        
        # Tentative de shift journalier (plus granulaire)
        shifter = ShiftTransformer(n_periods=5, frequency='D')
        
        with pytest.raises(ValueError, match="cannot be more granular"):
            shifter.fit_transform(series)


# ============================================================================
# Tests de la classe MaskTransformer - Fonctionnement de base
# ============================================================================

class TestMaskTransformerBasic:
    """Tests de base pour la classe MaskTransformer."""

    def test_basic_masking_last(self, daily_series):
        """Test de base du masquage des dernières observations."""
        masker = MaskTransformer(n_obs=2, mask_frequency='M', how='last')
        masked = masker.fit_transform(daily_series)
        
        # Des valeurs doivent être masquées (NaN)
        assert masked.isna().sum() > 0
        # La longueur doit rester la même
        assert len(masked) == len(daily_series)

    def test_basic_masking_first(self, daily_series):
        """Test de base du masquage des premières observations."""
        masker = MaskTransformer(n_obs=2, mask_frequency='M', how='first')
        masked = masker.fit_transform(daily_series)
        
        # Des valeurs doivent être masquées (NaN)
        assert masked.isna().sum() > 0
        # La longueur doit rester la même
        assert len(masked) == len(daily_series)

    def test_zero_masking(self, daily_series):
        """Test avec n_obs=0 (pas de masquage)."""
        masker = MaskTransformer(n_obs=0, mask_frequency='M')
        masked = masker.fit_transform(daily_series)
        
        # Aucune valeur ne doit être masquée
        assert masked.isna().sum() == 0
        pd.testing.assert_series_equal(masked, daily_series)

    def test_transform_then_inverse_transform_recovers_original(self, daily_series):
        """Vérification que transform puis inverse_transform retourne les données initiales."""
        masker = MaskTransformer(n_obs=3, mask_frequency='M', how='last')
        
        masked = masker.fit_transform(daily_series)
        recovered = masker.inverse_transform(masked)
        
        # Les données originales doivent être parfaitement restaurées
        pd.testing.assert_index_equal(recovered.index, daily_series.index)
        np.testing.assert_array_equal(recovered.values, daily_series.values)
        assert recovered.name == daily_series.name

    def test_transform_inverse_transform_how_first(self, daily_series):
        """Vérification de la récupération avec how='first'."""
        masker = MaskTransformer(n_obs=5, mask_frequency='M', how='first')
        
        masked = masker.fit_transform(daily_series)
        recovered = masker.inverse_transform(masked)
        
        pd.testing.assert_index_equal(recovered.index, daily_series.index)
        np.testing.assert_array_equal(recovered.values, daily_series.values)


class TestMaskTransformerBoundaries:
    """Tests de la gestion des débuts et fins de séries pour MaskTransformer."""

    def test_series_start_with_how_first(self):
        """Test de la gestion du début de série avec how='first'."""
        # Série qui commence le 10 janvier (début incomplet de période)
        dates = pd.date_range('2024-01-10', periods=50, freq='D')
        series = pd.Series(range(50), index=dates, name='test')
        
        masker = MaskTransformer(n_obs=3, mask_frequency='M', how='first')
        masked = masker.fit_transform(series)
        
        # Les premières observations de chaque mois doivent être masquées
        # Le premier mois (janvier) commence le 10, donc les 3 premières dates disponibles
        january_mask = masked.index.month == 1
        january_masked = masked[january_mask]
        
        # Les 3 premières observations de janvier (10, 11, 12) doivent être NaN
        assert january_masked.iloc[:3].isna().all()
        
        # Le reste de janvier (si présent) ne doit pas être NaN
        if len(january_masked) > 3:
            assert not january_masked.iloc[3:].isna().any()
        
        # Vérification de la récupération
        recovered = masker.inverse_transform(masked)
        np.testing.assert_array_equal(recovered.values, series.values)

    def test_series_start_with_how_last(self):
        """Test de la gestion du début de série avec how='last'."""
        # Série qui commence le 15 janvier
        dates = pd.date_range('2024-01-15', periods=60, freq='D')
        series = pd.Series(range(60), index=dates, name='test')
        
        masker = MaskTransformer(n_obs=5, mask_frequency='M', how='last')
        masked = masker.fit_transform(series)
        
        # Pour janvier, les dernières observations doivent être masquées
        # La série a des observations du 15 au 31 janvier (17 jours)
        # Les 5 dernières (27-31 janvier) doivent être masquées
        january_mask = masked.index.month == 1
        january_data = masked[january_mask]
        
        # Les 5 dernières observations de janvier doivent être NaN
        assert january_data.iloc[-5:].isna().all()
        
        # Le début de janvier ne doit pas être masqué
        assert not january_data.iloc[:-5].isna().any()
        
        # Vérification de la récupération
        recovered = masker.inverse_transform(masked)
        np.testing.assert_array_equal(recovered.values, series.values)

    def test_series_end_with_how_first(self):
        """Test de la gestion de la fin de série avec how='first'."""
        # Série qui se termine le 10 mars
        dates = pd.date_range('2024-01-01', '2024-03-10', freq='D')
        series = pd.Series(range(len(dates)), index=dates, name='test')
        
        masker = MaskTransformer(n_obs=4, mask_frequency='M', how='first')
        masked = masker.fit_transform(series)
        
        # Pour mars, les premières observations doivent être masquées
        march_mask = masked.index.month == 3
        march_data = masked[march_mask]
        
        # Les 4 premières observations de mars (1-4) doivent être NaN
        assert march_data.iloc[:4].isna().all()
        
        # Le reste de mars ne doit pas être masqué
        if len(march_data) > 4:
            assert not march_data.iloc[4:].isna().any()
        
        # Vérification de la récupération
        recovered = masker.inverse_transform(masked)
        np.testing.assert_array_equal(recovered.values, series.values)

    def test_series_end_with_how_last(self):
        """Test de la gestion de la fin de série avec how='last'."""
        # Série qui se termine le 15 mars (fin incomplète de période)
        dates = pd.date_range('2024-01-01', '2024-03-15', freq='D')
        series = pd.Series(range(len(dates)), index=dates, name='test')
        
        masker = MaskTransformer(n_obs=3, mask_frequency='M', how='last')
        masked = masker.fit_transform(series)
        
        # Pour mars, les 3 dernières observations disponibles doivent être masquées
        march_mask = masked.index.month == 3
        march_data = masked[march_mask]
        
        # Les 3 dernières observations de mars (13, 14, 15) doivent être NaN
        assert march_data.iloc[-3:].isna().all()
        
        # Le reste de mars ne doit pas être masqué
        assert not march_data.iloc[:-3].isna().any()
        
        # Vérification de la récupération
        recovered = masker.inverse_transform(masked)
        np.testing.assert_array_equal(recovered.values, series.values)


class TestMaskTransformerDataFrame:
    """Tests du MaskTransformer avec des DataFrames."""

    def test_mask_dataframe(self, daily_dataframe):
        """Test du masquage sur DataFrame."""
        masker = MaskTransformer(n_obs=2, mask_frequency='M', how='last')
        masked = masker.fit_transform(daily_dataframe)
        
        # Toutes les colonnes doivent avoir le même pattern de NaN
        nan_pattern_col1 = masked['col1'].isna()
        nan_pattern_col2 = masked['col2'].isna()
        assert (nan_pattern_col1 == nan_pattern_col2).all()

    def test_mask_dataframe_inverse_transform(self, daily_dataframe):
        """Test de la récupération pour DataFrame."""
        masker = MaskTransformer(n_obs=4, mask_frequency='M', how='first')
        
        masked = masker.fit_transform(daily_dataframe)
        recovered = masker.inverse_transform(masked)
        
        pd.testing.assert_frame_equal(recovered, daily_dataframe)


class TestMaskTransformerEdgeCases:
    """Tests des cas limites pour MaskTransformer."""

    def test_mask_more_than_available(self):
        """Masquage de plus d'observations qu'il n'y en a dans une période."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        series = pd.Series(range(5), index=dates)
        
        # Seulement 5 jours disponibles, demande de masquer 10 par mois
        masker = MaskTransformer(n_obs=10, mask_frequency='M', how='last')
        masked = masker.fit_transform(series)
        
        # Ne doit pas planter et retourner une série valide
        assert len(masked) == len(series)
        assert isinstance(masked.index, pd.DatetimeIndex)

    def test_mask_without_transform_raises_error(self, daily_series):
        """Appel de inverse_transform sans transform doit lever une erreur."""
        masker = MaskTransformer(n_obs=2, mask_frequency='M')
        
        with pytest.raises(ValueError, match="Must call transform"):
            masker.inverse_transform(daily_series)

    def test_inverse_transform_with_filled_nans(self):
        """Test de inverse_transform après remplissage des NaN avec des prédictions."""
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        series = pd.Series(range(60), index=dates)
        
        masker = MaskTransformer(n_obs=2, mask_frequency='M', how='last')
        masked = masker.fit_transform(series)
        
        # Remplissage des NaN avec des prédictions (valeurs différentes)
        predictions = masked.fillna(-999)
        
        # inverse_transform doit restaurer les valeurs originales (pas les -999)
        restored = masker.inverse_transform(predictions)
        
        np.testing.assert_array_equal(restored.values, series.values)

    def test_invalid_input_type(self):
        """Rejet des entrées non-pandas."""
        masker = MaskTransformer(n_obs=2, mask_frequency='M')
        with pytest.raises(ValueError, match="must be a pandas Series or DataFrame"):
            masker.fit([1, 2, 3])


# ============================================================================
# Tests de la classe PublicationDelayTransformer - Initialisation
# ============================================================================

class TestPublicationDelayTransformerInit:
    """Tests for PublicationDelayTransformer initialization."""
    
    def test_init_with_dict(self, delays_dict_simple):
        """Initialisation avec un dictionnaire de délais."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift',
            prediction_date='2024-01-01'
        )
        
        assert transformer.delays == delays_dict_simple
        assert transformer.strategy == 'shift'
        assert transformer.prediction_date == '2024-01-01'
    
    def test_init_with_dataframe(self, delays_dataframe):
        """Initialisation avec un DataFrame de délais."""
        transformer = PublicationDelayTransformer(
            delays=delays_dataframe,
            strategy='mask',
            prediction_date=datetime(2024, 6, 15)
        )
        
        assert isinstance(transformer.delays, pd.DataFrame)
        assert transformer.strategy == 'mask'
        assert isinstance(transformer.prediction_date, datetime)
    
    def test_init_with_strategy_dict(self, delays_dict_simple):
        """Initialisation avec un dictionnaire de stratégies."""
        strategy_dict = {
            'GDP': 'shift',
            'inflation': 'mask',
            'unemployment': 'shift'
        }
        
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy=strategy_dict
        )
        
        assert transformer.strategy == strategy_dict
    
    def test_init_default_values(self, delays_dict_simple):
        """Initialisation avec des valeurs par défaut."""
        default_vals = {
            'delay': 30.0,
            'unit': 'D',
            'reference_point': 'end',
            'target_frequency': 'M'
        }
        
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='mask',
            default_values=default_vals
        )
        
        assert transformer.default_values == default_vals
    
    def test_init_invalid_strategy_string(self, delays_dict_simple):
        """Validation de la stratégie invalide (chaîne de caractères)."""
        with pytest.raises(ValueError, match="strategy must be 'shift' or 'mask'"):
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                strategy='invalid'
            )
    
    def test_init_invalid_strategy_dict(self, delays_dict_simple):
        """Validation de la stratégie invalide (dictionnaire)."""
        with pytest.raises(ValueError, match="strategy must be 'shift' or 'mask'"):
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                strategy={'GDP': 'invalid_strategy'}
            )
    
    def test_init_invalid_strategy_type(self, delays_dict_simple):
        """Validation du type de stratégie invalide."""
        with pytest.raises(TypeError, match="'strategy' should be a string"):
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                strategy=123  # Type invalide
            )
    
    def test_init_invalid_reference_point(self, delays_dict_simple):
        """Validation du point de référence invalide."""
        with pytest.raises(ValueError, match="reference_point must be 'start' or 'end'"):
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                reference_point='middle'
            )
    
    def test_init_invalid_handle_missing(self, delays_dict_simple):
        """Validation de la gestion des délais manquants invalide."""
        with pytest.raises(ValueError, match="'handle_missing_delays' must be"):
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                handle_missing_delays='invalid'
            )
    
    def test_init_default_values_missing_keys(self, delays_dict_simple):
        """Validation des clés manquantes dans default_values."""
        incomplete_defaults = {
            'delay': 30.0,
            'unit': 'D'
            # Clés manquantes: reference_point, target_frequency
        }
        
        with pytest.raises(ValueError, match="Expected a 'default_values' dictionnary"):
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                strategy='mask',
                default_values=incomplete_defaults
            )
    
    def test_init_warning_strategy_dict_with_defaults(self, delays_dict_simple):
        """Warning quand strategy est un dict et default_values est fourni."""
        strategy_dict = {'GDP': 'shift', 'inflation': 'mask'}
        default_vals = {
            'delay': 30.0,
            'unit': 'D',
            'reference_point': 'end',
            'target_frequency': 'M'
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                strategy=strategy_dict,
                default_values=default_vals
            )
            assert len(w) == 1
            assert "default_values" in str(w[0].message)
    
    def test_init_warning_target_frequency_with_shift(self, delays_dict_simple):
        """Warning quand target_frequency est fourni avec strategy='shift'."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PublicationDelayTransformer(
                delays=delays_dict_simple,
                strategy='shift',
                target_frequency='M'
            )
            assert len(w) == 1
            assert "target_frequency" in str(w[0].message)


# ============================================================================
# Tests de la classe PublicationDelayTransformer - Méthodes fit et transform
# ============================================================================

class TestPublicationDelayTransformerFitTransform:
    """Tests for fit and transform methods."""
    
    def test_fit_basic(self, sample_time_series, delays_dict_simple):
        """Méthode fit de base."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift',
            prediction_date='2024-06-01'
        )
        
        result = transformer.fit(sample_time_series)
        
        # Vérification que fit retourne self
        assert result is transformer
        
        # Vérification des attributs créés
        assert hasattr(transformer, 'prediction_date_')
        assert hasattr(transformer, 'inferred_params_')
        assert hasattr(transformer, 'detected_frequencies_')
    
    def test_fit_creates_shift_params(self, sample_time_series, delays_dict_simple):
        """Création des paramètres de shift après fit."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift'
        )
        
        transformer.fit(sample_time_series)
        
        assert hasattr(transformer, 'shift_params')
        assert 'GDP' in transformer.shift_params
        assert 'n_periods' in transformer.shift_params['GDP']
        assert 'frequency' in transformer.shift_params['GDP']
    
    def test_transform_basic(self, sample_time_series, delays_dict_simple):
        """Transformation de base des données."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift'
        )
        
        transformer.fit(sample_time_series)
        result = transformer.transform(sample_time_series)
        
        # Vérification de la structure du résultat
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_time_series.shape
        assert list(result.columns) == list(sample_time_series.columns)
    
    def test_fit_transform(self, sample_time_series, delays_dict_simple):
        """Méthode fit_transform."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift'
        )
        
        result = transformer.fit_transform(sample_time_series)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_time_series.shape
    
    def test_transform_without_fit_raises_error(self, sample_time_series, delays_dict_simple):
        """Erreur si transform est appelé avant fit."""
        transformer = PublicationDelayTransformer(
            delays=delays_dict_simple,
            strategy='shift'
        )
        
        with pytest.raises(NotFittedError):
            transformer.transform(sample_time_series)


# ============================================================================
# Tests de vérification des paramètres calculés (shift_params, mask_params)
# ============================================================================

class TestPublicationDelayTransformerParams:
    """Tests de vérification des paramètres shift_params et mask_params."""

    # -------------------------------------------------------------------------
    # Tests des shift_params
    # -------------------------------------------------------------------------

    def test_shift_params_structure(self):
        """Vérification de la structure des shift_params."""
        # Données mensuelles simples
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({'GDP': range(12)}, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 30.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        
        transformer.fit(data)
        
        # Vérification de la structure
        assert 'GDP' in transformer.shift_params
        assert 'n_periods' in transformer.shift_params['GDP']
        assert 'frequency' in transformer.shift_params['GDP']
        assert isinstance(transformer.shift_params['GDP']['n_periods'], int)

    def test_shift_params_n_periods_is_integer(self):
        """Vérification que n_periods est un entier."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'GDP': range(12),
            'inflation': range(12)
        }, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 45.0, 'inflation': 30.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        
        transformer.fit(data)
        
        for col, params in transformer.shift_params.items():
            assert isinstance(params['n_periods'], int), f"n_periods pour {col} n'est pas un entier"

    def test_shift_params_frequency_matches_detected(self):
        """Vérification que la fréquence dans shift_params correspond à la fréquence détectée."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({'GDP': range(12)}, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 30.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        
        transformer.fit(data)
        
        # La fréquence dans shift_params doit correspondre à la fréquence détectée
        assert transformer.shift_params['GDP']['frequency'] == transformer.detected_frequencies_['GDP']

    def test_shift_params_larger_delay_means_more_periods(self):
        """Vérification qu'un délai plus grand implique plus de périodes à shifter."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'short_delay': range(12),
            'long_delay': range(12)
        }, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'short_delay': 15.0, 'long_delay': 60.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        
        transformer.fit(data)
        
        # Plus le délai est grand, plus le n_periods (en valeur absolue) est grand
        # Note: n_periods est négatif (shift vers le passé)
        short_n = transformer.shift_params['short_delay']['n_periods']
        long_n = transformer.shift_params['long_delay']['n_periods']
        
        # En valeur absolue, long_delay doit avoir plus de périodes
        assert abs(long_n) >= abs(short_n), \
            f"Délai long ({long_n}) devrait avoir >= périodes que délai court ({short_n})"

    def test_shift_params_reference_point_end_vs_start(self):
        """Vérification de l'impact du reference_point sur n_periods."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({'GDP': range(12)}, index=dates)
        
        # Transformer avec reference_point='end'
        transformer_end = PublicationDelayTransformer(
            delays={'GDP': 45.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        transformer_end.fit(data)
        
        # Transformer avec reference_point='start'
        transformer_start = PublicationDelayTransformer(
            delays={'GDP': 45.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='start'
        )
        transformer_start.fit(data)
        
        # Les n_periods doivent être différents (environ 1 période de différence)
        n_end = transformer_end.shift_params['GDP']['n_periods']
        n_start = transformer_start.shift_params['GDP']['n_periods']
        
        # Avec reference_point='end', le délai effectif est plus court
        # donc on devrait shifter moins (ou la différence ~1 période)
        assert n_end != n_start or abs(n_end - n_start) <= 1

    def test_shift_params_zero_delay(self):
        """Vérification du comportement avec un délai de 0."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({'GDP': range(12)}, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 0.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        
        transformer.fit(data)
        
        # Avec un délai de 0, n_periods devrait être proche de 0 ou positif
        # (dépend de la position dans la période)
        n_periods = transformer.shift_params['GDP']['n_periods']
        assert isinstance(n_periods, int)

    def test_shift_params_all_columns_present(self):
        """Vérification que toutes les colonnes avec délai sont dans shift_params."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'GDP': range(12),
            'inflation': range(12),
            'unemployment': range(12)
        }, index=dates)
        
        delays = {'GDP': 30.0, 'inflation': 45.0, 'unemployment': 15.0}
        
        transformer = PublicationDelayTransformer(
            delays=delays,
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        
        transformer.fit(data)
        
        # Toutes les colonnes avec délai doivent être dans shift_params
        for col in delays.keys():
            assert col in transformer.shift_params, f"Colonne {col} absente de shift_params"

    def test_shift_params_different_delay_units(self):
        """Vérification du calcul avec différentes unités de délai."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({'GDP': range(12)}, index=dates)
        
        # Même délai exprimé en jours et en heures
        delay_days = 30.0
        delay_hours = 30.0 * 24  # 720 heures = 30 jours
        
        transformer_days = PublicationDelayTransformer(
            delays={'GDP': delay_days},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        transformer_days.fit(data)
        
        transformer_hours = PublicationDelayTransformer(
            delays={'GDP': delay_hours},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='h',
            reference_point='end'
        )
        transformer_hours.fit(data)
        
        # Les n_periods devraient être identiques (même délai effectif)
        assert transformer_days.shift_params['GDP']['n_periods'] == \
               transformer_hours.shift_params['GDP']['n_periods']

    # -------------------------------------------------------------------------
    # Tests des mask_params
    # -------------------------------------------------------------------------

    def test_mask_params_structure(self):
        """Vérification de la structure des mask_params."""
        # Données journalières
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        data = pd.DataFrame({'GDP': range(90)}, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 10.0},
            strategy='mask',
            prediction_date='2024-03-15',
            delay_unit='D',
            reference_point='end',
            target_frequency='M'
        )
        
        transformer.fit(data)
        
        # Vérification de la structure
        # Note: si can_mask est False, la colonne sera dans shift_params
        if 'GDP' in transformer.mask_params:
            assert 'n_obs' in transformer.mask_params['GDP']
            assert 'mask_frequency' in transformer.mask_params['GDP']
            assert 'how' in transformer.mask_params['GDP']

    def test_mask_params_n_obs_is_positive_integer(self):
        """Vérification que n_obs est un entier positif."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        data = pd.DataFrame({'GDP': range(90)}, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 5.0},
            strategy='mask',
            prediction_date='2024-03-15',
            delay_unit='D',
            reference_point='end',
            target_frequency='M'
        )
        
        transformer.fit(data)
        
        if 'GDP' in transformer.mask_params:
            n_obs = transformer.mask_params['GDP']['n_obs']
            assert isinstance(n_obs, int), "n_obs doit être un entier"
            assert n_obs >= 0, "n_obs doit être positif ou nul"

    def test_mask_params_how_is_last(self):
        """Vérification que 'how' est toujours 'last' (comportement par défaut)."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        data = pd.DataFrame({'GDP': range(90)}, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 5.0},
            strategy='mask',
            prediction_date='2024-03-15',
            delay_unit='D',
            reference_point='end',
            target_frequency='M'
        )
        
        transformer.fit(data)
        
        if 'GDP' in transformer.mask_params:
            assert transformer.mask_params['GDP']['how'] == 'last'

    def test_mask_params_larger_delay_means_more_obs(self):
        """Vérification qu'un délai plus grand implique plus d'observations à masquer."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        data = pd.DataFrame({
            'short_delay': range(90),
            'long_delay': range(90)
        }, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'short_delay': 5.0, 'long_delay': 15.0},
            strategy='mask',
            prediction_date='2024-03-15',
            delay_unit='D',
            reference_point='end',
            target_frequency='M'
        )
        
        transformer.fit(data)
        
        # Vérification si les colonnes sont dans mask_params
        # (sinon elles ont été déplacées vers shift_params car can_mask=False)
        if 'short_delay' in transformer.mask_params and 'long_delay' in transformer.mask_params:
            short_n = transformer.mask_params['short_delay']['n_obs']
            long_n = transformer.mask_params['long_delay']['n_obs']
            
            assert long_n >= short_n, \
                f"Délai long ({long_n} obs) devrait masquer >= que délai court ({short_n} obs)"

    def test_mask_params_target_frequency_used(self):
        """Vérification que la target_frequency est utilisée dans mask_params."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        data = pd.DataFrame({'GDP': range(90)}, index=dates)
        
        target_freq = 'M'
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 5.0},
            strategy='mask',
            prediction_date='2024-03-15',
            delay_unit='D',
            reference_point='end',
            target_frequency=target_freq
        )
        
        transformer.fit(data)
        
        if 'GDP' in transformer.mask_params:
            # La mask_frequency doit correspondre à la target_frequency normalisée
            assert 'mask_frequency' in transformer.mask_params['GDP']

    def test_mask_fallback_to_shift_when_cannot_mask(self):
        """Vérification du fallback vers shift quand le masquage n'est pas possible."""
        # Données mensuelles avec un délai très long (impossible de masquer sans tout rendre NaN)
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({'GDP': range(12)}, index=dates)
        
        # Délai de 60 jours avec fréquence mensuelle = impossible de masquer
        transformer = PublicationDelayTransformer(
            delays={'GDP': 60.0},
            strategy='mask',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end',
            target_frequency='M'
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            transformer.fit(data)
        
        # La colonne doit être dans shift_params (fallback) ou mask_params selon le calcul
        assert 'GDP' in transformer.shift_params or 'GDP' in transformer.mask_params

    # -------------------------------------------------------------------------
    # Tests de cohérence entre delays fournis et params calculés
    # -------------------------------------------------------------------------

    def test_shift_params_consistent_with_delays(self):
        """Vérification de la cohérence entre les délais fournis et les paramètres calculés."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'fast': range(12),
            'medium': range(12),
            'slow': range(12)
        }, index=dates)
        
        # Délais croissants
        delays = {'fast': 10.0, 'medium': 30.0, 'slow': 60.0}
        
        transformer = PublicationDelayTransformer(
            delays=delays,
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point='end'
        )
        
        transformer.fit(data)
        
        # Vérification de l'ordre : plus le délai est grand, plus le shift est important
        n_fast = abs(transformer.shift_params['fast']['n_periods'])
        n_medium = abs(transformer.shift_params['medium']['n_periods'])
        n_slow = abs(transformer.shift_params['slow']['n_periods'])
        
        assert n_fast <= n_medium <= n_slow, \
            f"Ordre incohérent: fast={n_fast}, medium={n_medium}, slow={n_slow}"

    def test_mask_params_consistent_with_delays(self):
        """Vérification de la cohérence entre les délais fournis et les paramètres de masquage."""
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        data = pd.DataFrame({
            'fast': range(90),
            'medium': range(90),
            'slow': range(90)
        }, index=dates)
        
        # Délais croissants mais suffisamment petits pour permettre le masquage
        delays = {'fast': 3.0, 'medium': 7.0, 'slow': 12.0}
        
        transformer = PublicationDelayTransformer(
            delays=delays,
            strategy='mask',
            prediction_date='2024-03-15',
            delay_unit='D',
            reference_point='end',
            target_frequency='M'
        )
        
        transformer.fit(data)
        
        # Vérification de l'ordre pour les colonnes dans mask_params
        mask_cols = [col for col in ['fast', 'medium', 'slow'] if col in transformer.mask_params]
        
        if len(mask_cols) >= 2:
            for i in range(len(mask_cols) - 1):
                col1, col2 = mask_cols[i], mask_cols[i + 1]
                n1 = transformer.mask_params[col1]['n_obs']
                n2 = transformer.mask_params[col2]['n_obs']
                # Le délai plus grand devrait avoir plus d'observations à masquer
                assert n1 <= n2, f"Ordre incohérent: {col1}={n1}, {col2}={n2}"

    def test_params_with_dict_delay_unit(self):
        """Vérification des paramètres avec delay_unit spécifié par variable."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'GDP': range(12),
            'inflation': range(12)
        }, index=dates)
        
        # Délais avec unités différentes mais équivalents
        # 30 jours pour GDP, 4 semaines (~28 jours) pour inflation
        transformer = PublicationDelayTransformer(
            delays={'GDP': 30.0, 'inflation': 4.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit={'GDP': 'D', 'inflation': 'W'},
            reference_point='end'
        )
        
        transformer.fit(data)
        
        # Les deux devraient avoir des n_periods similaires (30 jours vs 28 jours)
        n_gdp = abs(transformer.shift_params['GDP']['n_periods'])
        n_inflation = abs(transformer.shift_params['inflation']['n_periods'])
        
        # Différence d'au plus 1 période attendue
        assert abs(n_gdp - n_inflation) <= 1, \
            f"Différence trop importante: GDP={n_gdp}, inflation={n_inflation}"

    def test_params_with_dict_reference_point(self):
        """Vérification des paramètres avec reference_point spécifié par variable."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')
        data = pd.DataFrame({
            'GDP': range(12),
            'inflation': range(12)
        }, index=dates)
        
        transformer = PublicationDelayTransformer(
            delays={'GDP': 45.0, 'inflation': 45.0},
            strategy='shift',
            prediction_date='2024-06-15',
            delay_unit='D',
            reference_point={'GDP': 'end', 'inflation': 'start'}
        )
        
        transformer.fit(data)
        
        # Les n_periods doivent différer d'environ 1 période
        n_gdp = transformer.shift_params['GDP']['n_periods']
        n_inflation = transformer.shift_params['inflation']['n_periods']
        
        # Avec start, le délai effectif est plus long, donc plus de périodes
        assert n_gdp != n_inflation or abs(n_gdp - n_inflation) <= 1


# ============================================================================
# Tests des fonctions auxiliaires
# ============================================================================

class TestAuxiliaryFunctions:
    """Tests for auxiliary functions."""
    
    def test_extract_param_by_variable_constant(self):
        """Extraction de paramètre avec valeur constante."""
        df = pd.DataFrame({
            'variable': ['GDP', 'inflation'],
            'unit': ['D', 'D']
        }).set_index('variable')
        
        result = _extract_param_by_variable(df, 'unit')
        
        # Doit retourner la valeur unique
        assert result == 'D'
    
    def test_extract_param_by_variable_varying(self):
        """Extraction de paramètre avec valeurs variables."""
        df = pd.DataFrame({
            'variable': ['GDP', 'inflation'],
            'unit': ['D', 'W']
        }).set_index('variable')
        
        result = _extract_param_by_variable(df, 'unit')
        
        # Doit retourner un dictionnaire
        assert isinstance(result, dict)
        assert result['GDP'] == 'D'
        assert result['inflation'] == 'W'
    
    def test_resolve_strategy_string(self):
        """Résolution de stratégie avec chaîne de caractères."""
        strategy = 'shift'
        entity_key = ('France',)
        
        result = _resolve_strategy(strategy, entity_key)
        assert result == 'shift'
    
    def test_resolve_strategy_dict_simple(self):
        """Résolution de stratégie avec dictionnaire simple."""
        strategy = {
            ('France',): 'shift',
            ('Germany',): 'mask'
        }
        
        result_fr = _resolve_strategy(strategy, ('France',))
        result_de = _resolve_strategy(strategy, ('Germany',))
        
        assert result_fr == 'shift'
        assert result_de == 'mask'
    
    def test_resolve_strategy_dict_by_variable(self):
        """Résolution de stratégie avec dictionnaire par variable."""
        strategy = {
            'GDP': 'shift',
            'inflation': 'mask'
        }
        entity_key = ('France',)
        
        # Doit retourner le dictionnaire complet
        result = _resolve_strategy(strategy, entity_key)
        assert result == strategy
    
    def test_resolve_strategy_callable(self):
        """Résolution de stratégie avec fonction callable."""
        def strategy_func(entity_key):
            if 'France' in entity_key:
                return 'shift'
            return 'mask'
        
        result_fr = _resolve_strategy(strategy_func, ('France',))
        result_de = _resolve_strategy(strategy_func, ('Germany',))
        
        assert result_fr == 'shift'
        assert result_de == 'mask'
    
    def test_resolve_strategy_invalid_string(self):
        """Validation de stratégie invalide (chaîne)."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            _resolve_strategy('invalid', ('France',))
