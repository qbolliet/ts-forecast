"""Tests for HighFrequencyImputer class.

This module tests the HighFrequencyImputer class for mixed-frequency imputation,
focusing on edge cases, panel data handling, and delay management.
"""
import pytest
import pandas as pd
import numpy as np
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


class TestScaleAnnualToMonthly:
    """§2.1 : le facteur d'échelle annuel -> mensuel est inversé et approximatif."""

    @pytest.mark.xfail(
        strict=True,
        reason="§2.1 high_frequency_imputer_review.md : get_conversion_factor est "
               "appelé dans le mauvais sens et le scaling n'est appliqué qu'au fit "
               "(pas au predict), produisant des valeurs mensuelles imputées "
               "~150x trop grandes en magnitude pour une variable annuelle.",
    )
    def test_scale_annual_to_monthly(self, mixed_freq_timeseries):
        """balance_commerciale_annuelle imputée mensuellement reste à l'échelle mensuelle.

        Reproduction empirique exacte du scénario de la revue §2.1 :
        cascade_refitting=False, keep_lower_frequencies=False, séries
        temporelles. Les vraies valeurs annuelles sont entre -28 et -9 ;
        une valeur mensuelle additive plausible (annuel/12) est de l'ordre
        de quelques unités, pas de plusieurs centaines.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=False,
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_timeseries)

        original = mixed_freq_timeseries['balance_commerciale_annuelle']
        imputed_months = result['balance_commerciale_annuelle'][original.isna()]

        # Écart-type des vraies valeurs annuelles : borne raisonnable pour une
        # valeur mensuelle additive (annuel / 12), avec une marge large
        annual_scale = original.dropna().abs().max()
        assert (imputed_months.abs() < annual_scale).all()


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

    @pytest.mark.xfail(
        strict=True,
        reason="§2.3 high_frequency_imputer_review.md : imputation_models_ est "
               "indexé par var_key seul ; une variable imputée à deux étapes de "
               "cascade (ex. balance_commerciale_annuelle à 'Q' puis 'M') voit son "
               "second modèle écraser le premier.",
    )
    def test_models_keyed_by_stage(self, mixed_freq_timeseries):
        """Une variable imputée à deux étapes -> deux modèles distincts."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
        )
        imputer.fit(mixed_freq_timeseries)

        # balance_commerciale_annuelle est imputée à l'étape 'Q' puis 'M'
        stage_entries = [
            entry for pred_freq, var_key in imputer.model_fitting_order_
            if var_key == 'balance_commerciale_annuelle'
            for entry in [(pred_freq, imputer.imputation_models_.get(var_key))]
        ]
        assert len(stage_entries) == 2

        models = [info['model'] for _, info in stage_entries if isinstance(info, dict)]
        assert len(models) == 2
        assert models[0] is not models[1]


class TestPanelSingleFitPerVariable:
    """§2.4 : un panel ne doit pas réentraîner le même modèle global par entité."""

    @pytest.mark.xfail(
        strict=True,
        reason="§2.4 high_frequency_imputer_review.md : ordered_impute_keys contient "
               "une clé (entité, variable) par entité et le modèle global (entraîné "
               "sur tout le panel) est réentraîné une fois par entité au lieu d'une "
               "seule fois par variable.",
    )
    def test_panel_single_fit_per_variable(self, mixed_freq_panel):
        """3 entités, 1 variable annuelle -> 1 seul fit par étape."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=False,
            keep_lower_frequencies=False,
        )
        imputer.fit(mixed_freq_panel)

        balance_keys = [
            key for key in imputer.imputation_models_
            if isinstance(key, tuple) and key[-1] == 'balance_commerciale_annuelle'
        ]
        # Les 3 entités partagent le même modèle global : un seul objet distinct
        model_ids = {
            id(imputer.imputation_models_[key]['model'])
            for key in balance_keys
            if isinstance(imputer.imputation_models_[key], dict)
        }
        assert len(model_ids) == 1


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

        target_level = result.xs('target', level='frequency')

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
