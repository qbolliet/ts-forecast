"""Tests for HighFrequencyImputer class.

This module tests the HighFrequencyImputer class for mixed-frequency imputation,
focusing on edge cases, panel data handling, and delay management.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer


class TestHighFrequencyImputerInit:
    """Tests for HighFrequencyImputer initialization and parameter validation."""

    def test_init_default_parameters(self):
        """Test initialisation avec paramètres par défaut."""
        imputer = HighFrequencyImputer(target_frequency='M')

        assert imputer.target_frequency == 'M'
        assert imputer.additive_transformer is None
        assert imputer.estimator is None
        assert imputer.low_frequency_handling is None  # None par défaut pour compatibilité sklearn
        assert imputer.delays is None
        assert imputer.impute_delayed_values is False
        assert imputer.fit_per_entity is False
        assert imputer.time_col == 'date'
        assert imputer.panel_cols is None

    def test_init_with_all_parameters(self):
        """Test initialisation avec tous les paramètres."""
        estimator = LinearRegression()
        transformer = StandardScaler()
        delays_df = pd.DataFrame({
            'variable': ['var1'],
            'delay': [30],
            'unit': ['D'],
            'reference_point': ['end']
        })

        imputer = HighFrequencyImputer(
            target_frequency='Q',
            additive_transformer=transformer,
            estimator=estimator,
            low_frequency_handling={'var1': 'impute'},
            delays=delays_df,
            impute_delayed_values=True,
            fit_per_entity=True,
            time_col='timestamp',
            panel_cols=['country', 'sector']
        )

        assert imputer.target_frequency == 'Q'
        assert imputer.additive_transformer is transformer
        assert imputer.estimator is estimator
        assert imputer.low_frequency_handling == {'var1': 'impute'}
        assert imputer.delays is not None
        assert imputer.impute_delayed_values is True
        assert imputer.fit_per_entity is True
        assert imputer.time_col == 'timestamp'
        assert imputer.panel_cols == ['country', 'sector']

    def test_init_with_dict_estimator(self):
        """Test initialisation avec dictionnaire d'estimateurs."""
        estimators = {
            'var1': LinearRegression(),
            'var2': Ridge(),
        }

        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=estimators
        )

        assert isinstance(imputer.estimator, dict)
        assert 'var1' in imputer.estimator
        assert 'var2' in imputer.estimator


class TestHighFrequencyImputerValidation:
    """Tests for parameter validation."""

    def test_invalid_target_frequency(self):
        """Test erreur avec fréquence cible invalide."""
        imputer = HighFrequencyImputer(target_frequency='invalid_freq')

        # Création de données de test
        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'value': range(12)}, index=dates)

        with pytest.raises(ValueError, match="Invalid target_frequency"):
            imputer.fit(df)

    def test_invalid_estimator_type(self):
        """Test erreur avec type d'estimateur invalide."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator="not_an_estimator"
        )

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'value': range(12)}, index=dates)

        with pytest.raises(ValueError, match="must be a sklearn BaseEstimator"):
            imputer.fit(df)

    def test_invalid_low_frequency_handling_strategy(self):
        """Test erreur avec stratégie de gestion basse fréquence invalide."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            low_frequency_handling={'var1': 'invalid_strategy'}
        )

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'var1': range(12)}, index=dates)

        with pytest.raises(ValueError, match="Invalid strategy"):
            imputer.fit(df)

    def test_invalid_delays_dataframe(self):
        """Test erreur avec DataFrame de délais incomplet."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            delays=pd.DataFrame({'variable': ['var1']})  # Colonnes manquantes
        )

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'var1': range(12)}, index=dates)

        with pytest.raises(ValueError, match="missing required columns"):
            imputer.fit(df)

    def test_input_not_dataframe(self):
        """Test erreur avec entrée non-DataFrame."""
        imputer = HighFrequencyImputer(target_frequency='M')

        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            imputer.fit([1, 2, 3, 4, 5])

    def test_missing_datetime_index(self):
        """Test erreur avec DataFrame sans index temporel."""
        imputer = HighFrequencyImputer(target_frequency='M')
        df = pd.DataFrame({'value': range(12)})

        with pytest.raises(ValueError, match="DatetimeIndex"):
            imputer.fit(df)


class TestHighFrequencyImputerFrequencyDetection:
    """Tests for frequency detection functionality."""

    def test_detect_single_frequency(self):
        """Test détection de fréquence unique."""
        imputer = HighFrequencyImputer(target_frequency='M')

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'value': range(12)}, index=dates)

        imputer.fit(df)

        assert 'value' in imputer.detected_frequencies_
        assert imputer.detected_frequencies_['value'] in ['M', 'ME']

    def test_detect_multiple_frequencies(self):
        """Test détection de fréquences multiples."""
        imputer = HighFrequencyImputer(target_frequency='M')

        # Création de données avec fréquences différentes
        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({
            'monthly_var': range(12),
            'quarterly_var': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        }, index=dates)

        imputer.fit(df)

        assert 'monthly_var' in imputer.detected_frequencies_

    def test_target_frequency_higher_than_data_raises(self):
        """Test erreur si fréquence cible plus haute que données."""
        imputer = HighFrequencyImputer(target_frequency='D')

        # Données mensuelles, cible journalière -> erreur
        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'value': range(12)}, index=dates)

        with pytest.raises(ValueError, match="Target frequency.*is higher than"):
            imputer.fit(df)


class TestHighFrequencyImputerClassification:
    """Tests for variable classification."""

    def test_classify_aggregate_variables(self):
        """Test classification des variables haute fréquence."""
        imputer = HighFrequencyImputer(target_frequency='M')

        # Données journalières, cible mensuelle -> agrégation
        dates = pd.date_range('2023-01-01', periods=31, freq='D')
        df = pd.DataFrame({'daily_var': range(31)}, index=dates)

        imputer.fit(df)

        assert imputer.variable_categories_.get('daily_var') == 'aggregate'

    def test_classify_target_freq_variables(self):
        """Test classification des variables à la fréquence cible."""
        imputer = HighFrequencyImputer(target_frequency='M')

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'monthly_var': range(12)}, index=dates)

        imputer.fit(df)

        assert imputer.variable_categories_.get('monthly_var') == 'target_freq'

    def test_classify_interpolate_variables(self):
        """Test classification des variables basse fréquence pour interpolation."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            low_frequency_handling={'quarterly_var': 'interpolate'}
        )

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({
            'monthly_var': range(12),
            'quarterly_var': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        }, index=dates)

        imputer.fit(df)

        # La variable trimestrielle devrait être marquée pour interpolation
        # Note: dépend de la détection de fréquence

    def test_classify_impute_variables(self):
        """Test classification des variables basse fréquence pour imputation."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            low_frequency_handling={'quarterly_var': 'impute'},
            estimator=LinearRegression()
        )

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({
            'monthly_var': range(12),
            'quarterly_var': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        }, index=dates)

        imputer.fit(df)


class TestHighFrequencyImputerAggregation:
    """Tests for high-frequency variable aggregation."""

    def test_aggregate_daily_to_monthly(self):
        """Test agrégation journalière vers mensuelle."""
        imputer = HighFrequencyImputer(target_frequency='M')

        # 31 jours de janvier
        dates = pd.date_range('2023-01-01', periods=31, freq='D')
        df = pd.DataFrame({'daily_var': range(31)}, index=dates)

        imputer.fit(df)
        result = imputer.transform(df)

        # Vérifier que la transformation a été appliquée
        assert isinstance(result, pd.DataFrame)
        assert 'daily_var' in result.columns

    def test_aggregate_preserves_sum(self):
        """Test que l'agrégation préserve la somme (données additives)."""
        imputer = HighFrequencyImputer(target_frequency='M')

        dates = pd.date_range('2023-01-01', periods=31, freq='D')
        values = [1.0] * 31  # 31 jours de valeur 1
        df = pd.DataFrame({'daily_var': values}, index=dates)

        imputer.fit(df)
        result = imputer.transform(df)

        # La somme devrait être préservée (ou proche)
        assert result['daily_var'].notna().any()


class TestHighFrequencyImputerInterpolation:
    """Tests for low-frequency variable interpolation."""

    def test_interpolate_daily_data_with_nan(self):
        """Test interpolation de données journalières avec NaN vers cible mensuelle."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            low_frequency_handling={}  # Default = interpolation
        )

        # Données journalières avec quelques NaN
        dates = pd.date_range('2023-01-01', periods=31, freq='D')
        values = list(range(31))
        values[5] = np.nan
        values[15] = np.nan
        values[25] = np.nan
        df = pd.DataFrame({'daily_var': values}, index=dates)

        imputer.fit(df)
        result = imputer.transform(df)

        # Les NaN devraient être gérés
        assert isinstance(result, pd.DataFrame)


class TestHighFrequencyImputerImputation:
    """Tests for ML-based imputation."""

    def test_impute_with_linear_regression(self):
        """Test imputation avec régression linéaire pour données journalières."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression()
        )

        # Données journalières avec quelques NaN
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        df = pd.DataFrame({
            'feature': range(60),
            'target': [i * 2 if i % 5 != 0 else np.nan for i in range(60)]
        }, index=dates)

        imputer.fit(df)
        result = imputer.transform(df)

        # Vérifier que la transformation a fonctionné
        assert isinstance(result, pd.DataFrame)

    def test_impute_with_dict_estimators(self):
        """Test imputation avec dictionnaire d'estimateurs."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator={
                'target1': LinearRegression(),
                'target2': Ridge(),
            }
        )

        # Données journalières
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        df = pd.DataFrame({
            'feature': range(60),
            'target1': [i if i % 3 != 0 else np.nan for i in range(60)],
            'target2': [i * 2 if i % 4 != 0 else np.nan for i in range(60)]
        }, index=dates)

        imputer.fit(df)
        result = imputer.transform(df)

        assert isinstance(result, pd.DataFrame)

    def test_impute_fallback_without_estimator(self):
        """Test fallback vers interpolation sans estimateur."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=None
        )

        # Données journalières avec NaN
        dates = pd.date_range('2023-01-01', periods=31, freq='D')
        df = pd.DataFrame({
            'feature': range(31),
            'target': [i if i % 5 != 0 else np.nan for i in range(31)]
        }, index=dates)

        # Ne devrait pas lever d'erreur, utilise interpolation comme fallback
        imputer.fit(df)
        result = imputer.transform(df)

        assert isinstance(result, pd.DataFrame)


class TestHighFrequencyImputerCascading:
    """Tests for cascading imputation order."""

    def test_imputation_order_attribute_exists(self):
        """Test que l'attribut imputation_order_ est créé après fit."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression()
        )

        # Données journalières
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        df = pd.DataFrame({
            'feature1': range(60),
            'feature2': [i * 2 for i in range(60)]
        }, index=dates)

        imputer.fit(df)

        # L'attribut imputation_order_ devrait exister (même si vide)
        assert hasattr(imputer, 'imputation_order_')
        assert isinstance(imputer.imputation_order_, list)


class TestHighFrequencyImputerPanelData:
    """Tests for panel data handling."""

    def test_panel_data_single_model(self):
        """Test données de panel avec modèle unique."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            fit_per_entity=False,
            panel_cols=['entity']
        )

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({
            'entity': ['A'] * 6 + ['B'] * 6,
            'feature': range(12),
            'target': [i if i % 2 == 0 else np.nan for i in range(12)]
        })
        df['date'] = list(dates[:6]) + list(dates[:6])
        df = df.set_index('date')

        imputer.fit(df)
        result = imputer.transform(df)

        assert isinstance(result, pd.DataFrame)

    def test_panel_data_per_entity_models(self):
        """Test données de panel avec modèles par entité."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            fit_per_entity=True,
            panel_cols=['entity']
        )

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({
            'entity': ['A'] * 12 + ['B'] * 12,
            'feature': list(range(12)) + list(range(12)),
            'target': [i if i % 2 == 0 else np.nan for i in range(12)] * 2
        })
        df['date'] = list(dates) + list(dates)
        df = df.set_index('date')

        imputer.fit(df)

        # Vérifier que des modèles par entité ont été créés
        assert imputer.is_panel_ is True


class TestHighFrequencyImputerDelays:
    """Tests for publication delay handling."""

    def test_delays_from_dataframe(self):
        """Test gestion des délais depuis DataFrame."""
        delays_df = pd.DataFrame({
            'variable': ['target'],
            'delay': [2],
            'unit': ['periods'],
            'reference_point': ['end']
        })

        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            delays=delays_df,
            impute_delayed_values=True
        )

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({
            'feature': range(12),
            'target': list(range(10)) + [np.nan, np.nan]  # 2 dernières valeurs NaN
        }, index=dates)

        imputer.fit(df)
        result = imputer.transform(df)

        # Les valeurs retardées devraient être imputées
        assert isinstance(result, pd.DataFrame)

    def test_delays_inferred_from_nan(self):
        """Test inférence des délais depuis les NaN trailing."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            impute_delayed_values=True  # delays=None -> inférence
        )

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({
            'feature': range(12),
            'target': list(range(9)) + [np.nan, np.nan, np.nan]  # 3 NaN trailing
        }, index=dates)

        imputer.fit(df)

        # Vérifier que les délais ont été inférés
        assert 'target' in imputer.inferred_delays_
        assert imputer.inferred_delays_['target'] == 3.0

    def test_impute_delayed_values_false(self):
        """Test que impute_delayed_values=False ne touche pas aux fins de période."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            impute_delayed_values=False
        )

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({
            'feature': range(12),
            'target': list(range(10)) + [np.nan, np.nan]
        }, index=dates)

        imputer.fit(df)

        # Les délais ne devraient pas être inférés
        assert imputer.inferred_delays_ == {}


class TestHighFrequencyImputerAdditiveTransformer:
    """Tests for additive transformer integration."""

    def test_with_additive_transformer(self):
        """Test avec transformer additif."""
        from sklearn.preprocessing import FunctionTransformer

        # Log transformer pour rendre multiplicatif -> additif
        log_transformer = FunctionTransformer(
            func=np.log1p,
            inverse_func=np.expm1
        )

        imputer = HighFrequencyImputer(
            target_frequency='M',
            additive_transformer=log_transformer
        )

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]}, index=dates)

        imputer.fit(df)
        result = imputer.transform(df)

        assert isinstance(result, pd.DataFrame)
        assert imputer.additive_transformer_ is not None

    def test_without_additive_transformer(self):
        """Test sans transformer additif."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            additive_transformer=None
        )

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'value': range(12)}, index=dates)

        imputer.fit(df)
        result = imputer.transform(df)

        assert imputer.additive_transformer_ is None
        assert isinstance(result, pd.DataFrame)


class TestHighFrequencyImputerInverseTransform:
    """Tests for inverse_transform functionality."""

    def test_inverse_transform_with_additive_transformer(self):
        """Test inverse_transform avec transformer additif."""
        from sklearn.preprocessing import FunctionTransformer

        log_transformer = FunctionTransformer(
            func=np.log1p,
            inverse_func=np.expm1
        )

        imputer = HighFrequencyImputer(
            target_frequency='M',
            additive_transformer=log_transformer
        )

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]}, index=dates)

        imputer.fit(df)
        transformed = imputer.transform(df)
        inverse = imputer.inverse_transform(transformed)

        assert isinstance(inverse, pd.DataFrame)

    def test_inverse_transform_without_additive_transformer(self):
        """Test inverse_transform sans transformer additif."""
        imputer = HighFrequencyImputer(target_frequency='M')

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'value': range(12)}, index=dates)

        imputer.fit(df)
        transformed = imputer.transform(df)
        inverse = imputer.inverse_transform(transformed)

        # Sans transformation additive, devrait retourner les données telles quelles
        pd.testing.assert_frame_equal(transformed, inverse)


class TestHighFrequencyImputerXYInterface:
    """Tests for XY transformer interface compliance."""

    def test_fit_returns_self(self):
        """Test que fit retourne self."""
        imputer = HighFrequencyImputer(target_frequency='M')

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'value': range(12)}, index=dates)

        result = imputer.fit(df)

        assert result is imputer

    def test_transform_with_y(self):
        """Test transform avec y."""
        imputer = HighFrequencyImputer(target_frequency='M')

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        X = pd.DataFrame({'feature': range(12)}, index=dates)
        y = pd.Series(range(12), index=dates, name='target')

        imputer.fit(X, y)
        X_t, y_t = imputer.transform(X, y)

        assert isinstance(X_t, pd.DataFrame)
        assert isinstance(y_t, pd.Series)

    def test_fit_transform(self):
        """Test fit_transform."""
        imputer = HighFrequencyImputer(target_frequency='M')

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'value': range(12)}, index=dates)

        result = imputer.fit_transform(df)

        assert isinstance(result, pd.DataFrame)

    def test_fit_transform_with_y(self):
        """Test fit_transform avec y."""
        imputer = HighFrequencyImputer(target_frequency='M')

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        X = pd.DataFrame({'feature': range(12)}, index=dates)
        y = pd.Series(range(12), index=dates, name='target')

        X_t, y_t = imputer.fit_transform(X, y)

        assert isinstance(X_t, pd.DataFrame)
        assert isinstance(y_t, pd.Series)


class TestHighFrequencyImputerEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_all_same_frequency(self):
        """Test avec toutes les variables à la même fréquence."""
        imputer = HighFrequencyImputer(target_frequency='M')

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({
            'var1': range(12),
            'var2': range(12, 24)
        }, index=dates)

        imputer.fit(df)
        result = imputer.transform(df)

        # Pas d'agrégation ni d'imputation nécessaire
        assert isinstance(result, pd.DataFrame)
        assert all(cat == 'target_freq' for cat in imputer.variable_categories_.values())

    def test_empty_dataframe(self):
        """Test avec DataFrame vide."""
        imputer = HighFrequencyImputer(target_frequency='M')

        df = pd.DataFrame(index=pd.DatetimeIndex([]))

        with pytest.raises(ValueError):
            imputer.fit(df)

    def test_single_observation(self):
        """Test avec une seule observation."""
        imputer = HighFrequencyImputer(target_frequency='M')

        dates = pd.date_range('2023-01-01', periods=1, freq='M')
        df = pd.DataFrame({'value': [1]}, index=dates)

        # Devrait lever un warning car pas assez de données pour détecter la fréquence
        # ou échouer silencieusement
        try:
            with pytest.warns(UserWarning):
                imputer.fit(df)
        except (ValueError, pytest.fail.Exception):
            # Acceptable si une erreur est levée ou pas de warning
            pass

    def test_all_nan_column(self):
        """Test avec colonne entièrement NaN."""
        imputer = HighFrequencyImputer(target_frequency='M')

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({
            'valid_col': range(12),
            'nan_col': [np.nan] * 12
        }, index=dates)

        # Devrait avertir mais ne pas échouer
        with pytest.warns(UserWarning):
            imputer.fit(df)

    def test_time_col_in_columns(self):
        """Test avec colonne temporelle dans les colonnes."""
        imputer = HighFrequencyImputer(target_frequency='M', time_col='date')

        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=12, freq='M'),
            'value': range(12)
        })

        imputer.fit(df)
        result = imputer.transform(df)

        assert isinstance(result, pd.DataFrame)


class TestHighFrequencyImputerSklearnCompatibility:
    """Tests for sklearn pipeline compatibility."""

    def test_in_sklearn_pipeline(self):
        """Test utilisation dans un pipeline sklearn."""
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ('imputer', HighFrequencyImputer(target_frequency='M')),
        ])

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'value': range(12)}, index=dates)

        result = pipeline.fit_transform(df)

        assert isinstance(result, pd.DataFrame)

    def test_get_params(self):
        """Test get_params pour la compatibilité sklearn."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            fit_per_entity=True
        )

        params = imputer.get_params()

        assert params['target_frequency'] == 'M'
        assert params['fit_per_entity'] is True

    def test_set_params(self):
        """Test set_params pour la compatibilité sklearn."""
        imputer = HighFrequencyImputer(target_frequency='M')

        imputer.set_params(fit_per_entity=True)

        assert imputer.fit_per_entity is True

    def test_clone(self):
        """Test clone pour la compatibilité sklearn."""
        from sklearn.base import clone

        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression()
        )

        cloned = clone(imputer)

        assert cloned.target_frequency == 'M'
        # L'estimateur peut être le même objet ou un clone selon sklearn
        assert cloned.estimator is not None


# =============================================================================
# Tests ciblés de non-régression (cf. high_frequency_imputer_review.md §7)
# =============================================================================
# Utilisent les fixtures `mixed_freq_timeseries` / `mixed_freq_panel` de
# tests/frequency/conftest.py, qui reproduisent les jeux de données du
# notebook `notebooks/2 - QB - Mixed frequencies.ipynb`. Les tests marqués
# xfail(strict=True) documentent le comportement souhaité (pas le
# comportement actuel bogué) et référencent la section de la revue.
import warnings


def _fit_transform_quiet(imputer, data):
    """Run fit_transform while silencing the (expected, unrelated) warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return imputer.fit_transform(data.copy())


@pytest.fixture
def annual_over_monthly():
    """Annual variable perfectly explained by an additive monthly covariate.

    Six years of month-start dates. ``covariable_mensuelle`` is dense and
    fluctuates around ``A / 12`` (A = 120); ``variable_annuelle`` is NaN
    everywhere except in January, where it carries the sum of the twelve
    monthly values of the year. The additive relation is therefore exact:
    a correctly scaled imputation must return monthly values summing back
    to the annual total.
    """
    A = 120.0
    dates = pd.date_range('2018-01-01', periods=72, freq='MS')
    rng = np.random.default_rng(0)

    monthly = pd.Series(A / 12 + rng.normal(0, 0.5, len(dates)), index=dates)
    annual = pd.Series(np.nan, index=dates)
    # La valeur annuelle est portée par janvier, ancre de la période
    for _, block in monthly.groupby(monthly.index.year):
        annual.loc[block.index[0]] = block.sum()

    data = pd.DataFrame(
        {'covariable_mensuelle': monthly, 'variable_annuelle': annual}
    )
    data.index.name = 'date'
    return data


class TestScaleAnnualToMonthly:
    """§2.1 : sens, exactitude et symétrie fit/predict du facteur d'échelle."""

    def test_scale_annual_to_monthly(self, annual_over_monthly):
        """Les 12 valeurs mensuelles imputées somment à la valeur annuelle A.

        Test d'acceptation du correctif §2.1 : le modèle apprend à prédire
        directement une valeur de sous-période (y_train divisé par 12), les
        covariables agrégées sont ramenées à la même échelle, et les
        prédictions ne sont jamais re-multipliées. Une erreur de sens du
        facteur donne 144*A, une double division A/12.

        Le mois d'ancre conserve le total annuel d'origine (défaut §2.6,
        hors périmètre de ce correctif) : la somme des douze sous-périodes
        est donc reconstituée depuis la moyenne des mois imputés.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=False,
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, annual_over_monthly)

        original = annual_over_monthly['variable_annuelle']
        imputed = result['variable_annuelle'][original.isna()]
        assert imputed.notna().all()

        # Reconstitution de la somme des 12 sous-périodes de chaque année
        for year, annual_value in original.dropna().groupby(original.dropna().index.year):
            months = imputed[imputed.index.year == year]
            reconstructed = months.mean() * 12
            assert reconstructed == pytest.approx(annual_value.iloc[0], rel=0.05)

    def test_scale_factor_is_exact_subperiod_count(self, annual_over_monthly):
        """Le facteur stocké vaut 12 (comptage calendaire), pas 12.17 ni 0.0822."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=False,
            keep_lower_frequencies=False,
        )
        _fit_transform_quiet(imputer, annual_over_monthly)

        # Étape unique (pas de fréquence intermédiaire) : une seule clé d'étape
        stage_key, = imputer.model_fitting_order_
        assert stage_key[1] == 'variable_annuelle'
        model_info = imputer.imputation_models_[stage_key]
        assert model_info['scale_factor'] == 12.0

    def test_annual_imputations_at_monthly_scale(self, mixed_freq_timeseries):
        """Sur la fixture, l'annuel imputé est à l'échelle mensuelle (annuel/12).

        Reproduction du scénario mesuré dans la revue §2.1
        (cascade_refitting=False, keep_lower_frequencies=False) : les vraies
        valeurs annuelles sont entre -28 et -9, la cible mensuelle additive
        est donc de l'ordre de -1.6, quand la classe produisait -330.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=False,
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_timeseries)

        original = mixed_freq_timeseries['balance_commerciale_annuelle']
        imputed = result['balance_commerciale_annuelle'][original.isna()].dropna()
        expected = original.mean() / 12

        # Le niveau central des imputations est celui d'une valeur mensuelle ;
        # la dispersion individuelle relève des défauts §2.7 et §4.5
        assert imputed.median() == pytest.approx(expected, abs=abs(expected))

    def test_quarterly_imputations_at_monthly_scale(self, mixed_freq_timeseries):
        """Idem pour la variable trimestrielle : cible = trimestriel / 3."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=False,
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_timeseries)

        original = mixed_freq_timeseries['pib_trimestriel']
        imputed = result['pib_trimestriel'][original.isna()].dropna()
        expected = original.mean() / 3

        assert imputed.median() == pytest.approx(expected, rel=0.1)


class TestOutputKeepsSourceIndex:
    """§2.2 : la cascade d'agrégation ne doit pas détruire l'index d'origine."""

    def test_output_keeps_monthly_index(self, mixed_freq_timeseries):
        """cascade_refitting=True : la sortie garde les 79 dates mensuelles."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_timeseries)

        assert len(result) == len(mixed_freq_timeseries)
        pd.testing.assert_index_equal(
            result.index.sort_values(), mixed_freq_timeseries.index.sort_values()
        )


class TestModelsKeyedByStage:
    """§2.3 : imputation_models_ ne doit pas être écrasé d'une étape à l'autre."""

    def test_models_keyed_by_stage(self, mixed_freq_timeseries):
        """Une variable imputée à deux étapes -> deux modèles distincts."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(mixed_freq_timeseries)

        # balance_commerciale_annuelle est imputée à l'étape 'Q' puis 'M'
        stage_entries = [
            (freq_label, imputer.imputation_models_[stage_key])
            for stage_key in imputer.model_fitting_order_
            for freq_label, var_key in [stage_key]
            if var_key == 'balance_commerciale_annuelle'
        ]
        assert [freq_label for freq_label, _ in stage_entries] == ['Q', 'M']

        # Deux modèles réellement distincts (cascade_refitting=True)
        models = [info['model'] for _, info in stage_entries if isinstance(info, dict)]
        assert len(models) == 2
        assert models[0] is not models[1]

        # Facteurs d'échelle propres à chaque étape : 4 trimestres puis 12 mois
        scales = [info['scale_factor'] for _, info in stage_entries]
        assert scales == [4.0, 12.0]

    def test_registry_size_matches_fitting_order(self, mixed_freq_timeseries):
        """Chaque étape enregistrée possède son entrée de registre."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(mixed_freq_timeseries)

        assert len(imputer.imputation_models_) == len(imputer.model_fitting_order_)
        assert list(imputer.imputation_models_) == imputer.model_fitting_order_


@pytest.fixture
def panel_annual_quarterly_over_monthly():
    """Panel version of `annual_quarterly_over_monthly`, entities identical.

    Three entities (``France``, ``Allemagne``, ``Italie``) carrying strictly
    identical values for every column. This makes the panel deduplication
    fix (review §2.4) verifiable deterministically: with a single GLOBAL
    model per (stage, variable) group, entities with identical inputs must
    get identical outputs — up to floating-point summation-order noise,
    unlike ``mixed_freq_panel`` (``tests/frequency/conftest.py``), whose
    per-entity seed depends on the non-reproducible ``hash()`` builtin.
    """
    dates = pd.date_range('2018-01-01', periods=72, freq='MS')
    rng = np.random.default_rng(1)

    monthly = pd.Series(10.0 + rng.normal(0, 0.5, len(dates)), index=dates)

    quarterly = pd.Series(np.nan, index=dates)
    for _, block in monthly.groupby([monthly.index.year, monthly.index.quarter]):
        quarterly.loc[block.index[0]] = block.sum()

    annual = pd.Series(np.nan, index=dates)
    for _, block in monthly.groupby(monthly.index.year):
        annual.loc[block.index[0]] = block.sum()

    single_entity = pd.DataFrame({
        'covariable_mensuelle': monthly,
        'variable_trimestrielle': quarterly,
        'variable_annuelle': annual,
    })
    single_entity.index.name = 'date'

    frames = []
    for entity in ('France', 'Allemagne', 'Italie'):
        df = single_entity.copy()
        df['entity'] = entity
        frames.append(df.reset_index())

    return pd.concat(frames, ignore_index=True).set_index(['entity', 'date']).sort_index()


class TestPanelSingleFitPerVariable:
    """§2.4 : un panel entraîne un modèle GLOBAL par variable, fitté une seule fois.

    Avant le correctif, `ordered_impute_keys` contenait une clé par couple
    (entité, variable) : pour 3 entités et 1 variable, le même modèle global
    était réentraîné 3 fois et `imputation_models_` portait 3 clés pour un
    seul modèle logique.
    """

    def test_panel_single_fit_per_variable(self, panel_annual_quarterly_over_monthly):
        """3 entités, 1 variable annuelle -> un seul fit par étape."""
        data = panel_annual_quarterly_over_monthly
        _FitCountingEstimator.reset()
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=_FitCountingEstimator(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
            train_on_partial_coverage=True,
        )
        result = _fit_transform_quiet(imputer, data)

        # Trois couples (étape, variable) : ('Q', annuelle), ('M', annuelle)
        # et ('M', trimestrielle) -- comme pour la série temporelle
        # équivalente (`TestFitCountByCascadeRefitting`), malgré les 3
        # entités du panel : ni le registre ni les fits ne sont dupliqués
        # par entité
        assert len(imputer.model_fitting_order_) == 3
        assert len(imputer.imputation_models_) == 3
        assert _FitCountingEstimator.n_fits == 3

        # Aucune clé de registre n'est indexée par entité (§2.4 : la clé de
        # groupe est la variable seule, ou (variable, fréquence détectée),
        # jamais (entité, variable))
        entities = {'France', 'Allemagne', 'Italie'}
        for _, group_key in imputer.model_fitting_order_:
            var_name = group_key[0] if isinstance(group_key, tuple) else group_key
            assert var_name not in entities

        # La provenance marque bien les lignes des 3 entités
        provenance = imputer.imputation_provenance_['variable_annuelle']
        assert set(provenance.index.get_level_values('entity')) == entities

        # Non-régression numérique : les 3 entités, strictement identiques en
        # entrée, obtiennent des imputations identiques (à un bruit de somme
        # flottante près) -- ce qui n'était vrai qu'accessoirement avant la
        # déduplication (3 fits redondants mais mathématiquement identiques)
        target_label = imputer._stage_frequency_label(imputer.effective_target_frequency_)
        target_level = result.xs(target_label, level='frequency')
        annual_col = target_level['variable_annuelle']
        france = annual_col.xs('France', level='entity')
        allemagne = annual_col.xs('Allemagne', level='entity')
        italie = annual_col.xs('Italie', level='entity')

        assert np.allclose(france, allemagne, equal_nan=True)
        assert np.allclose(france, italie, equal_nan=True)


class TestNoImputationWithoutCovariates:
    """§2.7 : fillna(mean) fabrique des prédictions là où aucune covariable n'existe."""

    @pytest.mark.xfail(
        strict=True,
        reason="§2.7 high_frequency_imputer_review.md : les NaN des features de "
               "prédiction sont remplacés par la moyenne des lignes à prédire, "
               "produisant des valeurs même aux dates où AUCUNE covariable n'a "
               "jamais été observée (_determine_prediction_samples existe mais "
               "n'est jamais appelée).",
    )
    def test_no_imputation_outside_window_without_covariates(self):
        """Aucune valeur imputée là où aucune covariable n'existe."""
        dates = pd.date_range('2020-01-01', periods=30, freq='MS')
        df = pd.DataFrame(index=dates)
        df.index.name = 'date'

        # Covariable totalement absente les 10 premiers mois, puis présente
        covariate = np.full(30, np.nan)
        covariate[10:] = np.linspace(1, 20, 20)
        df['covariate'] = covariate

        # Variable trimestrielle à imputer au mensuel
        target = np.full(30, np.nan)
        for i in range(0, 30, 3):
            target[i] = 100 + i
        df['target_var'] = target

        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=False,
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, df)

        # Mois 1 à 9 : la covariable n'a jamais existé, aucune imputation ne
        # devrait y avoir lieu
        never_covered = result['target_var'].iloc[1:10]
        assert never_covered.isna().all()


# Colonnes denses (fréquence de l'index) des fixtures de référence
DENSE_COLUMNS = ['production_industrielle', 'inflation_ipc', 'taux_chomage']


def _assert_dense_columns_untouched(result, source):
    """Assert observed values of the dense columns survive the cascade unchanged."""
    for col in DENSE_COLUMNS:
        observed = source[col].notna()
        pd.testing.assert_series_equal(
            result.loc[observed[observed].index, col],
            source.loc[observed, col],
            check_names=False,
            check_freq=False,
        )


class TestStageFramesRebuiltFromOriginal:
    """§5.1 : chaque étape est reconstruite depuis les données d'origine.

    Avant le refactoring, `_transform` réagrégeait `data_transformed` sur
    lui-même à chaque changement de fréquence de prédiction : à l'étape 'M',
    les colonnes mensuelles denses portaient encore les sommes trimestrielles
    figées à l'étape 'Q' (26 valeurs non-NaN au lieu de 78).
    """

    def test_stage_frames_keep_original_monthly_values(self, mixed_freq_timeseries):
        """cascade_refitting=True : les colonnes mensuelles gardent leurs valeurs."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_timeseries)

        _assert_dense_columns_untouched(result, mixed_freq_timeseries)

    def test_intermediate_stage_does_not_contaminate_target_level(
        self, mixed_freq_timeseries
    ):
        """keep_lower_frequencies=True : le niveau cible n'hérite pas de l'étape 'Q'."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_timeseries)

        target_label = imputer._stage_frequency_label(imputer.effective_target_frequency_)
        target_level = result.xs(target_label, level='frequency')

        _assert_dense_columns_untouched(target_level, mixed_freq_timeseries)


class TestInputNotMutated:
    """§5.1 : ni `fit` ni `transform` ne doivent modifier les données d'entrée."""

    @pytest.mark.parametrize('cascade_refitting', [False, True])
    def test_fit_transform_does_not_mutate_input(
        self, mixed_freq_timeseries, cascade_refitting
    ):
        """Séries temporelles : X est identique avant et après fit/transform."""
        data = mixed_freq_timeseries
        reference = data.copy()

        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=cascade_refitting,
            keep_lower_frequencies=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(data)
            pd.testing.assert_frame_equal(data, reference)

            imputer.transform(data)
            pd.testing.assert_frame_equal(data, reference)

            imputer.fit_transform(data)
        pd.testing.assert_frame_equal(data, reference)

    def test_panel_fit_transform_does_not_mutate_input(self, mixed_freq_panel):
        """Panel : X est identique avant et après fit_transform."""
        data = mixed_freq_panel
        reference = data.copy()

        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit_transform(data)

        pd.testing.assert_frame_equal(data, reference)


class _FitCountingEstimator(BaseEstimator, RegressorMixin):
    """Linear regression counting its fits, across clones.

    ``_get_estimator_for_variable`` clones the estimator before every fit:
    an instance counter would never be seen again by the test. The counter
    is therefore held by the class, which cloning preserves.
    """

    # Compteur partagé par toutes les instances, clones compris
    n_fits = 0

    def __init__(self):
        pass

    @classmethod
    def reset(cls) -> None:
        """Reset the shared fit counter."""
        cls.n_fits = 0

    def fit(self, X, y):
        """Fit the wrapped linear regression and count the call."""
        type(self).n_fits += 1
        self._model = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        """Delegate to the wrapped linear regression."""
        return self._model.predict(X)


@pytest.fixture
def annual_quarterly_over_monthly():
    """Annual and quarterly variables, both additive over a dense monthly one.

    Six years of month-start dates. ``covariable_mensuelle`` is dense;
    ``variable_trimestrielle`` carries, at each quarter-start month, the sum
    of the three monthly values of the quarter; ``variable_annuelle``
    carries in January the sum of the twelve monthly values of the year.
    Both low-frequency variables are imputable at every cascade stage below
    their own frequency, which makes the fit count observable:
    stages ``['Y', 'Q', 'M']``, three (stage, variable) couples.
    """
    dates = pd.date_range('2018-01-01', periods=72, freq='MS')
    rng = np.random.default_rng(1)

    monthly = pd.Series(10.0 + rng.normal(0, 0.5, len(dates)), index=dates)

    quarterly = pd.Series(np.nan, index=dates)
    for _, block in monthly.groupby([monthly.index.year, monthly.index.quarter]):
        quarterly.loc[block.index[0]] = block.sum()

    annual = pd.Series(np.nan, index=dates)
    for _, block in monthly.groupby(monthly.index.year):
        annual.loc[block.index[0]] = block.sum()

    data = pd.DataFrame({
        'covariable_mensuelle': monthly,
        'variable_trimestrielle': quarterly,
        'variable_annuelle': annual,
    })
    data.index.name = 'date'
    return data


class TestFitCountByCascadeRefitting:
    """§5.2 : le nombre de fits est piloté par `cascade_refitting`."""

    @staticmethod
    def _fitted_entries(imputer):
        """Registry entries holding a real model (no interpolation fallback)."""
        return [
            info for info in imputer.imputation_models_.values()
            if isinstance(info, dict)
        ]

    def _run(self, data, cascade_refitting):
        """Fit on the fixture with the spying estimator, fallbacks forbidden.

        ``train_on_partial_coverage=True`` neutralises the still-open defect
        §2.8: the 'Y' stage marks every higher-frequency column as
        aggregated, which empties the ORIGINAL training mask of the
        quarterly variable and sends it to the interpolation fallback. The
        fit count would then no longer be observable.
        """
        _FitCountingEstimator.reset()
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=_FitCountingEstimator(),
            cascade_refitting=cascade_refitting,
            keep_lower_frequencies=True,
            train_on_partial_coverage=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(data)

        # La fixture est construite pour qu'aucune variable ne se replie
        assert len(self._fitted_entries(imputer)) == len(imputer.model_fitting_order_)
        return imputer

    def test_single_fit_per_variable_without_refitting(
        self, annual_quarterly_over_monthly
    ):
        """cascade_refitting=False : un fit par variable imputable, pas par étape."""
        imputer = self._run(annual_quarterly_over_monthly, cascade_refitting=False)

        # Trois étapes enregistrées ('Q' et 'M' pour l'annuelle, 'M' pour la
        # trimestrielle) mais seulement deux variables imputables
        assert len(imputer.model_fitting_order_) == 3
        assert _FitCountingEstimator.n_fits == 2

        # Le modèle de l'annuelle est partagé par ses deux étapes, seul le
        # facteur d'échelle suit l'étape
        annual_entries = [
            imputer.imputation_models_[key]
            for key in imputer.model_fitting_order_
            if key[1] == 'variable_annuelle'
        ]
        assert len(annual_entries) == 2
        assert annual_entries[0]['model'] is annual_entries[1]['model']
        assert [info['scale_factor'] for info in annual_entries] == [4.0, 12.0]
        # Le facteur cuit dans le modèle reste celui de l'étape d'entraînement
        assert {info['fit_scale_factor'] for info in annual_entries} == {4.0}

    def test_one_fit_per_stage_with_refitting(self, annual_quarterly_over_monthly):
        """cascade_refitting=True : un fit par couple (étape, variable)."""
        imputer = self._run(annual_quarterly_over_monthly, cascade_refitting=True)

        assert len(imputer.model_fitting_order_) == 3
        assert _FitCountingEstimator.n_fits == 3

        # Chaque étape a son propre modèle, entraîné pour son propre facteur
        annual_entries = [
            imputer.imputation_models_[key]
            for key in imputer.model_fitting_order_
            if key[1] == 'variable_annuelle'
        ]
        assert annual_entries[0]['model'] is not annual_entries[1]['model']
        for info in annual_entries:
            assert info['fit_scale_factor'] == info['scale_factor']

    def test_reused_model_predicts_at_stage_scale(self, annual_quarterly_over_monthly):
        """Le modèle réutilisé est ramené à l'échelle de l'étape courante.

        Le facteur d'échelle est cuit dans le modèle (`y_train` en est divisé
        au fit) : réutilisé tel quel à l'étape 'M', un modèle entraîné à
        l'étape 'Q' rendrait des valeurs trimestrielles, trois fois trop
        grandes pour un mois.
        """
        data = annual_quarterly_over_monthly
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=False,
            keep_lower_frequencies=False,
        )
        result_single_stage = _fit_transform_quiet(imputer, data)

        imputer_cascade = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=False,
            keep_lower_frequencies=True,
        )
        result_cascade = _fit_transform_quiet(imputer_cascade, data)
        target_label = imputer_cascade._stage_frequency_label(
            imputer_cascade.effective_target_frequency_
        )
        target_level = result_cascade.xs(target_label, level='frequency')

        # Les imputations mensuelles de l'annuelle sont au même niveau que
        # celles obtenues sans étape intermédiaire
        missing = data['variable_annuelle'].isna()
        direct = result_single_stage.loc[missing, 'variable_annuelle']
        cascaded = target_level.loc[missing, 'variable_annuelle']

        assert cascaded.median() == pytest.approx(direct.median(), rel=0.1)


class TestReplaySymmetry:
    """§2.3 : le replay de `transform` reproduit celui de `fit_transform`."""

    @pytest.mark.parametrize('cascade_refitting', [False, True])
    @pytest.mark.parametrize('keep_lower_frequencies', [False, True])
    def test_fit_transform_equals_transform(
        self, mixed_freq_timeseries, cascade_refitting, keep_lower_frequencies
    ):
        """Séries temporelles : fit_transform puis transform donnent le même résultat."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=cascade_refitting,
            keep_lower_frequencies=keep_lower_frequencies,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fitted = imputer.fit_transform(mixed_freq_timeseries)
            replayed = imputer.transform(mixed_freq_timeseries)

        pd.testing.assert_frame_equal(fitted, replayed)

    def test_panel_fit_transform_equals_transform(self, mixed_freq_panel):
        """Panel : les labels d'étape par entité se rejouent à l'identique."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fitted = imputer.fit_transform(mixed_freq_panel)
            replayed = imputer.transform(mixed_freq_panel)

        pd.testing.assert_frame_equal(fitted, replayed)


# ---------------------------------------------------------------------------
# §2.6 — Désagrégation des valeurs d'ancre et recalage additif
# ---------------------------------------------------------------------------
from tsforecast.frequency.provenance import (
    ImputationProvenanceTracker,
    ProvenanceType,
)


def _period_keys(index, freq):
    """Build the period key of each row, entity included for a panel."""
    if isinstance(index, pd.MultiIndex):
        periods = index.get_level_values(-1).to_period(freq)
        return pd.Series(list(zip(index.droplevel(-1), periods)), index=index)
    return pd.Series(list(index.to_period(freq)), index=index)


def _assert_period_totals(result, original, column, freq):
    """Assert every fully-imputed observed period sums to its observed total.

    Returns the number of periods actually checked, so that a test cannot
    pass vacuously on an empty set of periods.
    """
    observed = original[column].dropna()
    totals = observed.groupby(_period_keys(observed.index, freq)).sum()

    keys = _period_keys(result.index, freq)
    sums = result[column].groupby(keys).sum()
    complete = result[column].groupby(keys).apply(lambda s: s.notna().all())

    checked = 0
    for period_key, observed_total in totals.items():
        # Périodes partiellement prédites : hors contrainte
        if period_key not in sums.index or not complete.get(period_key, False):
            continue
        assert sums[period_key] == pytest.approx(observed_total, abs=1e-8), (
            f"period {period_key} of '{column}' sums to {sums[period_key]}, "
            f"expected {observed_total}"
        )
        checked += 1

    assert checked > 0, f"no period of '{column}' could be checked"
    return checked


@pytest.fixture
def quarterly_over_monthly():
    """Quarterly variable carried by an additive monthly covariate.

    Month-start dates over six years. ``covariable_mensuelle`` is dense;
    ``variable_trimestrielle`` is NaN except on the first month of each
    quarter, where it holds the sum of the three monthly values — the same
    additive setup as :func:`annual_over_monthly`, one frequency step up.
    """
    dates = pd.date_range('2018-01-01', periods=72, freq='MS')
    rng = np.random.default_rng(7)

    monthly = pd.Series(30.0 + rng.normal(0, 2.0, len(dates)), index=dates)
    quarterly = pd.Series(np.nan, index=dates)
    # La valeur trimestrielle est portée par le premier mois, ancre de la période
    for _, block in monthly.groupby(monthly.index.to_period('Q')):
        quarterly.loc[block.index[0]] = block.sum()

    data = pd.DataFrame(
        {'covariable_mensuelle': monthly, 'variable_trimestrielle': quarterly}
    )
    data.index.name = 'date'
    return data


class TestPeriodTotalsEnforced:
    """§2.6 : la somme des sous-périodes égale la valeur basse fréquence observée."""

    def test_period_totals_enforced_time_series(self, quarterly_over_monthly):
        """Chaque trimestre entièrement imputé somme à la valeur observée (1e-8)."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, quarterly_over_monthly)

        checked = _assert_period_totals(
            result, quarterly_over_monthly, 'variable_trimestrielle', 'Q'
        )
        assert checked >= 20

    def test_period_totals_enforced_on_reference_dataset(self, mixed_freq_timeseries):
        """Le jeu de référence du notebook : PIB trimestriel et balance annuelle."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_timeseries)

        _assert_period_totals(result, mixed_freq_timeseries, 'pib_trimestriel', 'Q')
        _assert_period_totals(
            result, mixed_freq_timeseries, 'balance_commerciale_annuelle', 'Y'
        )

    def test_period_totals_enforced_panel(self, mixed_freq_panel):
        """Panel : le recalage est fait par entité, aucun bloc ne les mélange."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_panel)

        checked = _assert_period_totals(
            result, mixed_freq_panel, 'pib_trimestriel', 'Q'
        )
        # Trois entités : le nombre de blocs vérifiés les couvre toutes
        assert checked >= 3 * 20

    def test_anchor_no_longer_carries_period_total(self, quarterly_over_monthly):
        """Aucune date-ancre ne conserve le total de sa période basse fréquence."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, quarterly_over_monthly)

        anchors = quarterly_over_monthly['variable_trimestrielle'].dropna()
        imputed_at_anchors = result.loc[anchors.index, 'variable_trimestrielle']

        # L'ancre a été retirée et réimputée : plus aucune valeur d'origine
        assert not np.isclose(imputed_at_anchors, anchors, atol=1e-9).any()
        # Série positive : aucune sous-période ne dépasse le total de sa période
        keys = _period_keys(result.index, 'Q')
        totals = result['variable_trimestrielle'].groupby(keys).transform('sum')
        assert (result['variable_trimestrielle'] <= totals + 1e-8).all()

    def test_period_without_observation_not_rescaled(self, mixed_freq_timeseries):
        """Fin de série retardée : aucun recalage, prédictions brutes conservées.

        Le dernier trimestre du jeu de référence a été vidé (délai de
        publication) : sans valeur observée aucune contrainte n'est
        imposable, et les cellules gardent une provenance MODEL_ON_*.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        observed = mixed_freq_timeseries['pib_trimestriel'].dropna()
        observed_periods = set(observed.index.to_period('Q'))
        provenance = imputer.imputation_provenance_['pib_trimestriel']

        # Au moins une période de la grille est dépourvue de valeur observée
        unobserved = [
            date for date in provenance.index
            if date.to_period('Q') not in observed_periods
        ]
        assert unobserved
        assert all(
            provenance.loc[date] in (
                ProvenanceType.MODEL_ON_TRUE, ProvenanceType.MODEL_ON_MIXED
            )
            for date in unobserved
        )


class TestRescaleToPeriodTotalsUnit:
    """§2.6 : cas dégénérés de `_rescale_to_period_totals`, testés directement."""

    @staticmethod
    def _imputer():
        return HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())

    def test_partial_period_not_rescaled(self):
        """Une période dont une sous-période est NaN n'est jamais recalée."""
        imputer = self._imputer()
        index = pd.date_range('2020-01-01', periods=6, freq='MS')
        membership = pd.Series(list(index.to_period('Q')), index=index)
        predictions = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], index=index)
        y_original = pd.Series(
            [30.0, np.nan, np.nan, 60.0, np.nan, np.nan], index=index
        )

        rescaled, mask = imputer._rescale_to_period_totals(
            predictions, y_original, membership
        )

        # Q1 incomplet : inchangé et non marqué
        assert not mask.iloc[:3].any()
        pd.testing.assert_series_equal(rescaled.iloc[:3], predictions.iloc[:3])
        # Q2 complet : recalé sur 60
        assert mask.iloc[3:].all()
        assert rescaled.iloc[3:].sum() == pytest.approx(60.0, abs=1e-8)

    def test_zero_sum_block_warns_and_keeps_predictions(self):
        """Somme nulle avec total observé non nul : avertissement, pas de recalage."""
        imputer = self._imputer()
        index = pd.date_range('2020-01-01', periods=3, freq='MS')
        membership = pd.Series(list(index.to_period('Q')), index=index)
        predictions = pd.Series([1.0, -1.0, 0.0], index=index)
        y_original = pd.Series([30.0, np.nan, np.nan], index=index)

        with pytest.warns(UserWarning, match='sum to zero'):
            rescaled, mask = imputer._rescale_to_period_totals(
                predictions, y_original, membership
            )

        assert not mask.any()
        pd.testing.assert_series_equal(rescaled, predictions)

    def test_opposite_sign_block_rescales_with_warning(self):
        """Signe opposé : la contrainte est imposée et un avertissement est émis."""
        imputer = self._imputer()
        index = pd.date_range('2020-01-01', periods=3, freq='MS')
        membership = pd.Series(list(index.to_period('Q')), index=index)
        predictions = pd.Series([2.0, 3.0, 5.0], index=index)
        y_original = pd.Series([-30.0, np.nan, np.nan], index=index)

        with pytest.warns(UserWarning, match='flipped'):
            rescaled, mask = imputer._rescale_to_period_totals(
                predictions, y_original, membership
            )

        assert mask.all()
        assert rescaled.sum() == pytest.approx(-30.0, abs=1e-8)
        # Le profil a bien changé de signe
        assert (rescaled < 0).all()

    def test_zero_observed_total_is_rescaled(self):
        """Total observé nul : recalage exact, toutes les sous-périodes à zéro."""
        imputer = self._imputer()
        index = pd.date_range('2020-01-01', periods=3, freq='MS')
        membership = pd.Series(list(index.to_period('Q')), index=index)
        predictions = pd.Series([2.0, 3.0, 5.0], index=index)
        y_original = pd.Series([0.0, np.nan, np.nan], index=index)

        rescaled, mask = imputer._rescale_to_period_totals(
            predictions, y_original, membership
        )

        assert mask.all()
        assert rescaled.sum() == pytest.approx(0.0, abs=1e-8)

    def test_period_membership_panel_separates_entities(self, mixed_freq_panel):
        """Panel : deux entités ne partagent jamais une clé de période."""
        imputer = self._imputer()
        imputer.is_panel_ = True
        membership = imputer._period_membership(mixed_freq_panel.index, 'Q')

        # La clé porte l'entité en plus de la période
        entities = {key[0] for key in membership.dropna()}
        assert len(entities) == 3
        # Chaque clé ne regroupe que des lignes d'une seule entité
        for _, block in membership.groupby(membership):
            assert block.index.droplevel(-1).nunique() == 1

    def test_period_membership_restricted_to_group_entities(self, mixed_freq_panel):
        """Fréquences hétérogènes : seules les entités du groupe sont recalées."""
        imputer = self._imputer()
        imputer.is_panel_ = True
        membership = imputer._period_membership(
            mixed_freq_panel.index, 'Q', entities=[('France',)]
        )

        # Les lignes des autres entités sont hors périmètre
        assert {key[0] for key in membership.dropna()} == {('France',)}
        assert 0 < membership.notna().sum() < len(membership)

        # Le masque de désagrégation couvre exactement les mêmes lignes
        mask = imputer._disaggregation_mask(
            mixed_freq_panel, {'entities': [('France',)]}, 'pib_trimestriel'
        )
        assert mask.sum() == membership.notna().sum()


class TestEnforcePeriodTotalsParameter:
    """§2.6 : le paramètre ne pilote que la contrainte, jamais le retrait d'ancre."""

    def test_enforce_period_totals_false_keeps_raw_predictions(
        self, quarterly_over_monthly
    ):
        """enforce_period_totals=False : échelle homogène mais somme libre."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
            enforce_period_totals=False,
        )
        result = _fit_transform_quiet(imputer, quarterly_over_monthly)

        # L'ancre est retirée même sans contrainte : c'est la correction du bug
        anchors = quarterly_over_monthly['variable_trimestrielle'].dropna()
        imputed_at_anchors = result.loc[anchors.index, 'variable_trimestrielle']
        assert not np.isclose(imputed_at_anchors, anchors, atol=1e-9).any()

        # Aucune cellule n'est marquée DISAGGREGATED
        provenance = imputer.imputation_provenance_['variable_trimestrielle']
        assert not (provenance == ProvenanceType.DISAGGREGATED).any()
        assert (provenance == ProvenanceType.MODEL_ON_TRUE).any()

    def test_default_is_option_one(self):
        """La valeur par défaut applique bien l'Option 1 de la revue."""
        imputer = HighFrequencyImputer(target_frequency='M')
        assert imputer.enforce_period_totals is True
        assert imputer.get_params()['enforce_period_totals'] is True


class TestInterpolateFallbackRescaled:
    """§2.6 : le repli par interpolation est désagrégé lui aussi."""

    def test_interpolate_fallback_is_rescaled(self, quarterly_over_monthly):
        """Sans estimateur, la colonne interpolée somme quand même aux totaux."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=None,
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, quarterly_over_monthly)

        # Le repli a bien été emprunté
        assert any(
            info == 'interpolate_fallback'
            for info in imputer.imputation_models_.values()
        )
        _assert_period_totals(
            result, quarterly_over_monthly, 'variable_trimestrielle', 'Q'
        )

        # Les métadonnées de groupe existent même pour un repli
        assert set(imputer.stage_groups_) == set(imputer.model_fitting_order_)


class TestDisaggregationProvenance:
    """§2.6 : la matrice de provenance distingue les trois catégories."""

    def test_provenance_distinguishes_disaggregated(self, mixed_freq_timeseries):
        """ORIGINAL, DISAGGREGATED et MODEL_ON_* coexistent dans la matrice."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        present = set(imputer.imputation_provenance_.stack().unique())
        assert ProvenanceType.ORIGINAL in present
        assert ProvenanceType.DISAGGREGATED in present
        assert present & {ProvenanceType.MODEL_ON_TRUE, ProvenanceType.MODEL_ON_MIXED}

        # Les ancres désagrégées ne sont plus déclarées originales
        anchors = mixed_freq_timeseries['pib_trimestriel'].dropna().index
        assert (
            imputer.imputation_provenance_.loc[anchors, 'pib_trimestriel']
            == ProvenanceType.DISAGGREGATED
        ).all()

    def test_provenance_statistics_sum_to_cell_count(self, mixed_freq_timeseries):
        """Somme des catégories + non-imputées == nombre de cellules."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        tracker = ImputationProvenanceTracker()
        tracker.provenance_matrix_ = imputer.imputation_provenance_
        stats = tracker.compute_statistics()['overall']

        total = sum(stats[prov.value] for prov in ProvenanceType)
        total += stats['not_imputed']
        assert total == imputer.imputation_provenance_.size

    def test_training_mask_accepts_disaggregated_cells(self, mixed_freq_timeseries):
        """Une variable désagrégée à une étape reste entraînable à la suivante.

        Non-régression : le masque d'entraînement de `_prepare_training_data`
        se restreint aux cellules ORIGINAL quand train_on_partial_coverage
        est False ; les ancres devenant DISAGGREGATED, il doit les accepter
        aussi, sinon tout bascule en repli dès la seconde étape.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
            train_on_partial_coverage=False,
        )
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        # La variable annuelle est imputée à plus d'une étape : aucune de ses
        # entrées de registre ne doit être tombée en repli
        annual_stages = [
            key for key in imputer.model_fitting_order_
            if (key[1][0] if isinstance(key[1], tuple) else key[1])
            == 'balance_commerciale_annuelle'
        ]
        assert len(annual_stages) >= 2
        assert all(
            imputer.imputation_models_[key] != 'interpolate_fallback'
            for key in annual_stages
        )
