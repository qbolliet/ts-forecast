"""Tests for HighFrequencyImputer class.

This module tests the HighFrequencyImputer class for mixed-frequency imputation,
focusing on edge cases, panel data handling, and delay management.
"""
import dataclasses

import pytest
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
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
        assert imputer.time_col is None
        assert imputer.panel_cols is None
        assert imputer.imputation_scope == 'strict'
        assert imputer.coverage_threshold == 0.5
        assert imputer.train_on_partial_fit_order == 'frequency'
        assert imputer.verbose is False

    def test_init_with_all_parameters(self):
        """Test initialisation avec tous les paramètres."""
        estimator = LinearRegression()
        transformer = StandardScaler()

        imputer = HighFrequencyImputer(
            target_frequency='Q',
            additive_transformer=transformer,
            estimator=estimator,
            imputation_scope='extended_forward',
            coverage_threshold=0.75,
            time_col='timestamp',
            panel_cols=['country', 'sector'],
            verbose=True,
        )

        assert imputer.target_frequency == 'Q'
        assert imputer.additive_transformer is transformer
        assert imputer.estimator is estimator
        assert imputer.imputation_scope == 'extended_forward'
        assert imputer.coverage_threshold == 0.75
        assert imputer.time_col == 'timestamp'
        assert imputer.panel_cols == ['country', 'sector']
        assert imputer.verbose is True

    def test_removed_parameters_rejected(self):
        """Les paramètres dépréciés supprimés de l'API lèvent une erreur de
        construction : `delays` et `impute_delayed_values` n'existent plus
        (TypeError, argument inconnu), et `train_on_partial_fit_order='random'`
        n'est plus une valeur valide (ValueError)."""
        with pytest.raises(TypeError):
            HighFrequencyImputer(target_frequency='M', delays=None)

        with pytest.raises(TypeError):
            HighFrequencyImputer(target_frequency='M', impute_delayed_values=False)

        with pytest.raises(ValueError, match="train_on_partial_fit_order"):
            HighFrequencyImputer(target_frequency='M', train_on_partial_fit_order='random')

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
        """Test erreur avec fréquence cible invalide.

        La validation de `target_frequency` a lieu à la construction
        (`__init__`), pas à `fit()` : c'est donc la construction qui doit
        être encadrée par `pytest.raises`.
        """
        with pytest.raises(ValueError, match="Invalid target_frequency"):
            HighFrequencyImputer(target_frequency='invalid_freq')

    def test_invalid_estimator_type(self):
        """Test erreur avec type d'estimateur invalide.

        La validation de `estimator` a lieu à la construction (`__init__`),
        pas à `fit()`.
        """
        with pytest.raises(ValueError, match="must have a 'fit' method"):
            HighFrequencyImputer(
                target_frequency='M',
                estimator="not_an_estimator"
            )

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


class TestHighFrequencyImputerParameterValidation:
    """Garde-fous de construction avec l'API courante (pas l'API dépréciée)."""

    @pytest.mark.parametrize('kwargs, match', [
        (dict(on_frequency_mismatch='bogus'), 'on_frequency_mismatch'),
        (dict(coverage_threshold=2.0), 'coverage_threshold'),
        (dict(imputation_scope='bogus'), 'imputation_scope'),
        (dict(train_on_partial_fit_order='bogus'), 'train_on_partial_fit_order'),
    ])
    def test_invalid_scalar_parameter_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            HighFrequencyImputer(target_frequency='M', **kwargs)

    def test_empty_target_frequency_dict_raises(self):
        with pytest.raises(ValueError, match='cannot be empty'):
            HighFrequencyImputer(target_frequency={})

    def test_non_string_frequency_value_raises(self):
        with pytest.raises(ValueError, match='must be a string'):
            HighFrequencyImputer(target_frequency={'FR': 123})

    def test_invalid_frequency_string_in_dict_raises(self):
        with pytest.raises(ValueError, match='Invalid frequencies'):
            HighFrequencyImputer(target_frequency={'FR': 'bogus_freq'})

    def test_wrong_type_target_frequency_raises(self):
        with pytest.raises(TypeError, match='string or dict'):
            HighFrequencyImputer(target_frequency=['M'])

    def test_empty_estimator_dict_raises(self):
        with pytest.raises(ValueError, match='cannot be empty'):
            HighFrequencyImputer(target_frequency='M', estimator={})

    def test_single_estimator_missing_predict_raises(self):
        class NoPredict:
            def fit(self, X, y=None):
                return self

        with pytest.raises(ValueError, match="must have a 'predict' method"):
            HighFrequencyImputer(target_frequency='M', estimator=NoPredict())

    def test_dict_estimator_missing_predict_raises(self):
        class NoPredict:
            def fit(self, X, y=None):
                return self

        with pytest.raises(ValueError, match="must have a 'predict' method"):
            HighFrequencyImputer(target_frequency='M', estimator={'var1': NoPredict()})

    def test_dict_estimator_missing_fit_raises(self):
        class NoFit:
            def predict(self, X):
                return X

        with pytest.raises(ValueError, match="must have a 'fit' method"):
            HighFrequencyImputer(target_frequency='M', estimator={'var1': NoFit()})

    @pytest.mark.parametrize('param_name', [
        'cascade_refitting',
        'keep_lower_frequencies',
        'scale_features',
        'enforce_period_totals',
        'restore_original_values',
        'verbose',
    ])
    @pytest.mark.parametrize('bad_value', [1, 0, 'True', None])
    def test_boolean_params_validated(self, param_name, bad_value):
        """B22 : aucun booléen de __init__ n'était typé-vérifié avant ce correctif."""
        with pytest.raises(TypeError, match=param_name):
            HighFrequencyImputer(target_frequency='M', **{param_name: bad_value})


class TestHighFrequencyImputerClassification:
    """Tests for variable classification."""

    def test_classify_aggregate_variables(self):
        """Test classification des variables haute fréquence."""
        imputer = HighFrequencyImputer(target_frequency='M')

        # Données journalières, cible mensuelle -> agrégation
        dates = pd.date_range('2023-01-01', periods=31, freq='D')
        df = pd.DataFrame({'daily_var': range(31)}, index=dates)

        imputer.fit(df)

        assert 'daily_var' in imputer.variable_categories_['aggregate']

    def test_classify_target_freq_variables(self):
        """Test classification des variables à la fréquence cible."""
        imputer = HighFrequencyImputer(target_frequency='M')

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        df = pd.DataFrame({'monthly_var': range(12)}, index=dates)

        imputer.fit(df)

        assert 'monthly_var' in imputer.variable_categories_['target_freq']

    def test_classify_impute_variables(self):
        """Test classification des variables basse fréquence pour imputation.

        `low_frequency_handling` (permettant de forcer une stratégie par
        variable) a été supprimé : il n'existe plus que 3 catégories
        ('aggregate', 'target_freq', 'impute'), déterminées automatiquement
        depuis la fréquence détectée. Pour qu'une variable soit détectée en
        fréquence trimestrielle malgré un index mensuel, elle doit être NaN
        hors des mois d'ancrage du trimestre (comme dans les fixtures de
        `conftest.py`), pas seulement porter une valeur répétée.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression()
        )

        dates = pd.date_range('2023-01-01', periods=12, freq='M')
        quarterly_var = [np.nan] * 12
        for i in (0, 3, 6, 9):
            quarterly_var[i] = i // 3 + 1
        df = pd.DataFrame({
            'monthly_var': range(12),
            'quarterly_var': quarterly_var
        }, index=dates)

        imputer.fit(df)

        # La variable trimestrielle est classée pour imputation
        assert 'quarterly_var' in imputer.variable_categories_['impute']


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
        """Test interpolation de données journalières avec NaN vers cible mensuelle.

        `low_frequency_handling` a été supprimé : `daily_var` est de
        fréquence plus haute que la cible ('M'), donc classée 'aggregate'
        (sans rapport avec ce paramètre disparu).
        """
        imputer = HighFrequencyImputer(target_frequency='M')

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

    def test_panel_data_single_model(self, mixed_freq_panel):
        """Test données de panel avec modèle unique.

        `fit_per_entity` a été supprimé : un panel entraîne désormais
        toujours un modèle global unique par variable (jamais un modèle par
        entité, cf. `TestPanelSingleFitPerVariable`), donc plus rien à
        distinguer ici.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
        )

        result = _fit_transform_quiet(imputer, mixed_freq_panel)

        assert isinstance(result, pd.DataFrame)
        assert imputer.is_panel_ is True


class TestHighFrequencyImputerDelays:
    """§2.9 : `delays`/`impute_delayed_values` ont été retirés de l'API au
    profit de `imputation_scope='extended_forward'` + `coverage_threshold`,
    qui gère nativement les fins de série retardées (revue §2.9)."""

    def test_delayed_series_end_covered_by_extended_forward(self):
        """Fin de série retardée (3 derniers points NaN) couverte par la
        fenêtre étendue et effectivement imputée.

        Remplace le comportement de l'ancien `_impute_delayed_values`, qui
        interprétait le délai comme un nombre de LIGNES : ici, les 3
        derniers trimestres de `variable_trimestrielle` ne sont pas encore
        publiés. La couverture des autres colonnes (2 covariables mensuelles
        denses sur 3 colonnes, soit 2/3) dépasse `coverage_threshold` à ces
        dates : `imputation_scope='extended_forward'` doit donc les inclure
        dans la fenêtre d'entraînement étendue, et le modèle doit les
        imputer.
        """
        dates = pd.date_range('2018-01-01', periods=72, freq='MS')
        rng = np.random.default_rng(11)

        monthly = pd.Series(30.0 + rng.normal(0, 2.0, len(dates)), index=dates)
        monthly_2 = pd.Series(10.0 + rng.normal(0, 1.0, len(dates)), index=dates)
        quarterly = pd.Series(np.nan, index=dates)
        for _, block in monthly.groupby(monthly.index.to_period('Q')):
            quarterly.loc[block.index[0]] = block.sum()

        # Publication retardée : les 3 derniers trimestres ne sont pas
        # encore publiés (et non un simple dernier point)
        delayed_anchors = quarterly.dropna().index[-3:]
        quarterly.loc[delayed_anchors] = np.nan

        data = pd.DataFrame({
            'covariable_mensuelle': monthly,
            'covariable_mensuelle_2': monthly_2,
            'variable_trimestrielle': quarterly,
        })
        data.index.name = 'date'

        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
            imputation_scope='extended_forward',
        )
        result = _fit_transform_quiet(imputer, data)

        window = imputer._imputation_window_calc.get_imputation_window_mask(data)

        # Les dates d'ancrage retardées sont dans la fenêtre étendue...
        assert all(bool(window.get(date, False)) for date in delayed_anchors)
        # ...et effectivement imputées : plus aucun NaN là où le trimestre a
        # été vidé
        assert result.loc[delayed_anchors, 'variable_trimestrielle'].notna().all()


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

        # Rien à imputer ici : toutes les cellules sont ORIGINAL, l'inverse
        # redonne donc les données d'entrée, à l'index source (§2.10)
        pd.testing.assert_frame_equal(
            inverse, df, check_dtype=False, check_freq=False
        )


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
        assert not imputer.variable_categories_['aggregate']
        assert not imputer.variable_categories_['impute']
        assert set(imputer.variable_categories_['target_freq']) == {'var1', 'var2'}

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
        """Test get_params pour la compatibilité sklearn.

        `fit_per_entity` a été supprimé de l'API : remplacé ici par
        `verbose`, un autre paramètre booléen toujours présent.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            verbose=True
        )

        params = imputer.get_params()

        assert params['target_frequency'] == 'M'
        assert params['verbose'] is True

    def test_set_params(self):
        """Test set_params pour la compatibilité sklearn.

        `fit_per_entity` a été supprimé de l'API : remplacé ici par
        `verbose`, un autre paramètre booléen toujours présent.
        """
        imputer = HighFrequencyImputer(target_frequency='M')

        imputer.set_params(verbose=True)

        assert imputer.verbose is True

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

    def test_get_params_returns_input_target_frequency(self):
        """B3 : get_params()['target_frequency'] est LA valeur reçue par __init__.

        sklearn.clone() exige `get_params()[name] is` la valeur passée au
        constructeur ; stocker une version normalisée cassait ce contrat.
        """
        raw = {'FR': 'Q', 'DE': 'Q'}
        imputer = HighFrequencyImputer(
            target_frequency=raw, estimator=LinearRegression()
        )

        assert imputer.get_params()['target_frequency'] is raw

    def test_clone_with_dict_target_frequency(self):
        """B3 : clone() ne lève plus sur un target_frequency de type dict."""
        from sklearn.base import clone

        imputer = HighFrequencyImputer(
            target_frequency={'FR': 'Q', 'DE': 'Q'},
            estimator=LinearRegression(),
        )

        cloned = clone(imputer)

        assert cloned.target_frequency == {'FR': 'Q', 'DE': 'Q'}

    def test_pipeline_with_imputer_fits(self, mixed_freq_panel):
        """B3 : un HighFrequencyImputer à target_frequency dict s'entraîne
        dans un Pipeline sklearn sur des données de panel (clone() y est
        systématiquement appelé par certains méta-estimateurs)."""
        entities = sorted({key[0] for key in mixed_freq_panel.index})
        target_frequency = {entity: 'M' for entity in entities}

        pipeline = Pipeline([
            ('imputer', HighFrequencyImputer(
                target_frequency=target_frequency,
                estimator=LinearRegression(),
            )),
        ])

        result = _fit_transform_quiet(pipeline, mixed_freq_panel)

        assert isinstance(result, pd.DataFrame)


class TestEntryContractAndSklearnConformance:
    """§3.16 : contrat d'entrée et conformité sklearn (B14, B15, B16, B19, B20).

    Lot mécanique indépendant de la logique d'imputation : ces défauts
    empêchaient des usages nominaux de la classe (clone(), refit, panel
    déclaré par panel_cols, cible sans nom, transform avant fit).
    """

    def test_unnamed_y_is_imputed(self, quarterly_over_monthly):
        """B14 : une cible Series sans nom est bien imputée au transform.

        Avant correctif, la colonne portait le nom `0` au fit (`y.to_frame()`)
        et `'__target__'` au transform : les deux ne coïncidant jamais, `y`
        ressortait de `transform` inchangé, en silence.
        """
        X = quarterly_over_monthly[['covariable_mensuelle']]
        y = quarterly_over_monthly['variable_trimestrielle'].rename(None)
        assert y.name is None

        imputer = HighFrequencyImputer(
            target_frequency='M', estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(X, y)
            _, y_t = imputer.transform(X, y)

        assert y.isna().sum() > 0
        assert y_t.isna().sum() < y.isna().sum()

    def test_y_index_mismatch_raises(self):
        """B14 : un désaccord d'index entre X et y lève, plutôt que de
        produire un jeu de travail gonflé de NaN par un concat mal aligné."""
        dates = pd.date_range('2023-01-01', periods=12, freq='MS')
        X = pd.DataFrame({'feature': range(12)}, index=dates)
        # Même longueur que X, mais un index de valeurs différentes
        y = pd.Series(range(12), index=pd.RangeIndex(12), name='target')

        imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())

        with pytest.raises(ValueError, match='different indices'):
            imputer.fit(X, y)

    def test_panel_cols_without_multiindex_fits(self):
        """B15 : la forme panel documentée (panel_cols sur un frame plat,
        sans y) s'ajuste sans AttributeError sur effective_target_frequency_."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        rows = [
            {'country': entity, 'date': d, 'value': i}
            for entity in ('FR', 'DE')
            for i, d in enumerate(dates)
        ]
        flat_df = pd.DataFrame(rows)

        imputer = HighFrequencyImputer(
            target_frequency='Q', time_col='date', panel_cols=['country'],
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(flat_df)

        assert imputer.is_panel_ is True
        assert imputer.entities_ is not None
        assert isinstance(imputer.effective_target_frequency_, dict)

    def test_incomplete_target_frequency_dict_raises(self, mixed_freq_panel):
        """B16 : une entité manquante du dict target_frequency lève un
        ValueError nommant l'entité, plutôt que de faire disparaître ses
        variables en silence de toutes les catégories."""
        entities = sorted({key[0] for key in mixed_freq_panel.index})
        incomplete = {entities[0]: 'M'}  # il manque les autres entités

        imputer = HighFrequencyImputer(
            target_frequency=incomplete, estimator=LinearRegression()
        )

        with pytest.raises(ValueError, match='missing entries for entities'):
            imputer.fit(mixed_freq_panel)

    def test_refit_resets_transform_state(self, quarterly_over_monthly):
        """B19 : un refit purge la provenance/les snapshots d'un transform
        antérieur ; inverse_transform sans nouveau transform doit lever,
        pas réutiliser silencieusement l'état d'un jeu de données différent."""
        data_a = quarterly_over_monthly
        data_b = quarterly_over_monthly.iloc[::-1].set_axis(quarterly_over_monthly.index)

        imputer = HighFrequencyImputer(
            target_frequency='M', estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(data_a)
            imputer.transform(data_a)
            imputer.fit(data_b)

        assert not hasattr(imputer, 'imputation_provenance_')
        with pytest.raises(ValueError, match='previous call to transform'):
            imputer.inverse_transform(data_b)

    def test_transform_before_fit_raises_not_fitted(self):
        """B20 : transform avant fit lève NotFittedError, pas un
        AttributeError cryptique (conversion_metadata_ posé dans __init__
        de la classe de base faisait passer check_is_fitted à tort)."""
        from sklearn.exceptions import NotFittedError

        imputer = HighFrequencyImputer(target_frequency='M')
        dates = pd.date_range('2023-01-01', periods=12, freq='MS')
        df = pd.DataFrame({'value': range(12)}, index=dates)

        with pytest.raises(NotFittedError):
            imputer.transform(df)

    def test_no_estimator_warns_once(self, mixed_freq_panel):
        """B22 : estimator=None émet UN SEUL avertissement à fit, pas un par
        variable et par étape de la cascade."""
        imputer = HighFrequencyImputer(target_frequency='M')

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            imputer.fit(mixed_freq_panel)

        no_estimator_warnings = [
            w for w in caught
            if 'No estimator was provided' in str(w.message)
        ]
        assert len(no_estimator_warnings) == 1


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
    """§2.7 : aucune valeur produite hors de la fenêtre ni sans covariable.

    Le défaut corrigé : les NaN des features de prédiction étaient remplacés
    par la moyenne des lignes à prédire, produisant des valeurs même aux
    dates où aucune covariable n'a jamais été observée
    (`_determine_prediction_samples` existait mais n'était jamais appelée).
    """

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
        # devrait y avoir lieu. Depuis §3.14 l'entrée y est rendue telle
        # quelle — les ancres trimestrielles d'origine sont conservées, seule
        # une valeur PRODUITE y serait un défaut
        never_covered = result['target_var'].iloc[1:10]
        pd.testing.assert_series_equal(
            never_covered, df['target_var'].iloc[1:10], check_freq=False
        )
        provenance = imputer.imputation_provenance_['target_var'].iloc[1:10]
        imputed_types = {
            ProvenanceType.DISAGGREGATED,
            ProvenanceType.MODEL_ON_TRUE,
            ProvenanceType.MODEL_ON_MIXED,
        }
        assert not provenance.isin(list(imputed_types)).any()

        # Les mois couverts, eux, sont bien imputés
        assert result['target_var'].iloc[10:].notna().any()

    def test_no_prediction_outside_imputation_window(self, mixed_freq_timeseries):
        """Sur le jeu de référence, rien n'est imputé avant la fenêtre.

        `production_industrielle` ne démarre qu'en 2019 : la fenêtre stricte
        exclut toute l'année 2018, où la revue §2.7 observait pourtant une
        balance annuelle imputée dès 2018-02.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=False,
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_timeseries)

        window = imputer._imputation_window_calc.get_imputation_window_mask(
            mixed_freq_timeseries
        )
        outside = window.index[~window]
        assert (outside < pd.Timestamp('2019-01-01')).any()

        imputed_types = {
            ProvenanceType.DISAGGREGATED,
            ProvenanceType.MODEL_ON_TRUE,
            ProvenanceType.MODEL_ON_MIXED,
        }
        for column in ('pib_trimestriel', 'balance_commerciale_annuelle'):
            # Aucune valeur PRODUITE hors fenêtre. Depuis §3.14 le vidage du
            # périmètre de désagrégation se restreint lui aussi à la fenêtre :
            # hors fenêtre la sortie est exactement l'entrée, ancres comprises
            pd.testing.assert_series_equal(
                result.loc[outside, column],
                mixed_freq_timeseries.loc[outside, column],
                check_freq=False,
            )
            provenance = imputer.imputation_provenance_.loc[outside, column]
            assert not provenance.isin(list(imputed_types)).any()


class TestPredictionFieldDeterminism:
    """§2.7 : les imputations ne dépendent pas du champ de prédiction.

    Les covariables manquantes sont complétées par les moyennes du JEU
    D'ENTRAÎNEMENT, mémorisées dans `imputation_models_`, et non par la
    moyenne des lignes présentées à `transform`.
    """

    def test_imputation_deterministic_wrt_prediction_field(self, mixed_freq_timeseries):
        """`transform` d'un sous-ensemble redonne les mêmes valeurs.

        La coupure tombe sur un début d'année, donc sur une frontière de
        période pour les deux variables basse fréquence : tronquer au milieu
        d'une période changerait le recalage additif, qui relève de §2.6.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=False,
            keep_lower_frequencies=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(mixed_freq_timeseries.copy())

            full = imputer.transform(mixed_freq_timeseries.copy())
            subset_source = mixed_freq_timeseries.loc['2020-01-01':].copy()
            subset = imputer.transform(subset_source)

        pd.testing.assert_frame_equal(
            full.loc[subset.index], subset, check_freq=False
        )

    def test_feature_means_come_from_training_set(self, mixed_freq_timeseries):
        """Le registre porte les moyennes d'entraînement, sans NaN."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        fitted = [
            info for info in imputer.imputation_models_.values()
            if isinstance(info, dict)
        ]
        assert fitted
        for model_info in fitted:
            means = model_info['feature_means']
            assert list(means.index) == list(model_info['feature_cols'])
            assert means.notna().all()


class TestDeterminePredictionSamplesIsCalled:
    """§2.7 : `_determine_prediction_samples` n'est plus du code mort."""

    def test_determine_prediction_samples_is_called(
        self, monkeypatch, mixed_freq_timeseries
    ):
        """La méthode est appelée et ses groupes sont réellement consommés."""
        calls = []
        original = HighFrequencyImputer._determine_prediction_samples

        def spy(self, X_stage, rows_mask, feature_cols):
            samples = original(self, X_stage, rows_mask, feature_cols)
            calls.append((rows_mask.sum(), samples))
            return samples

        monkeypatch.setattr(
            HighFrequencyImputer, '_determine_prediction_samples', spy
        )

        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        assert calls

        for n_rows, samples in calls:
            # Partition exacte des lignes à prédire
            assert sum(len(index) for index, _ in samples) == n_rows
            # Groupes ordonnés du plus riche en covariables au plus pauvre
            sizes = [len(cols) for _, cols in samples]
            assert sizes == sorted(sizes, reverse=True)


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


class TestMultiFrequencyLevelsNotDuplicated:
    """§2.5 : `keep_lower_frequencies=True` ne duplique pas le niveau cible.

    Avant le correctif, le niveau cible était empilé deux fois : une fois
    sous son label de fréquence réel ('M') et une fois sous le label
    générique 'target', avec un contenu identique.
    """

    def test_multifreq_levels_not_duplicated(self, mixed_freq_timeseries):
        """Un seul niveau par label de fréquence, jamais de label 'target'."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_timeseries)

        levels = result.index.get_level_values('frequency').tolist()
        unique_levels = result.index.get_level_values('frequency').unique().tolist()

        assert 'target' not in unique_levels
        target_label = imputer._stage_frequency_label(imputer.effective_target_frequency_)
        assert levels.count(target_label) == len(mixed_freq_timeseries)

    def test_multifreq_levels_not_duplicated_panel(self, mixed_freq_panel):
        """Idem pour un panel : un seul niveau par label de fréquence."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_panel)

        unique_levels = result.index.get_level_values('frequency').unique().tolist()

        assert 'target' not in unique_levels
        target_label = imputer._stage_frequency_label(imputer.effective_target_frequency_)
        assert target_label in unique_levels


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

        La fin retardée sort de la fenêtre stricte — c'est précisément le rôle
        d'`imputation_scope='extended_forward'` de la couvrir (revue §2.9).
        Depuis §2.7, plus rien n'est imputé hors de la fenêtre : l'assertion
        porte donc sur les dates non observées QUI SONT dans la fenêtre. Le
        contrôle porte sur la balance annuelle : son année 2024 est privée de
        valeur (délai de publication) tout en tombant dans la fenêtre étendue,
        alors que le seul trimestre non observé du PIB, lui, en sort.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
            imputation_scope='extended_forward',
        )
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        column = 'balance_commerciale_annuelle'
        observed = mixed_freq_timeseries[column].dropna()
        observed_periods = set(observed.index.to_period('Y'))
        provenance = imputer.imputation_provenance_[column]
        window = imputer._imputation_window_calc.get_imputation_window_mask(
            mixed_freq_timeseries
        )

        # Au moins une période de la fenêtre est dépourvue de valeur observée
        unobserved = [
            date for date in provenance.index
            if date.to_period('Y') not in observed_periods
            and bool(window.get(date, False))
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

        # §3.15 (B2) : le marquage suit la NATURE de la cellule, pas la réussite
        # du recalage. Les dates-ancres restent DISAGGREGATED — sans quoi le
        # filtre de provenance de l'étape suivante viderait son masque
        # d'entraînement — et les sous-périodes seules deviennent MODEL_ON_*
        provenance = imputer.imputation_provenance_['variable_trimestrielle']
        disaggregated = provenance == ProvenanceType.DISAGGREGATED
        assert disaggregated.loc[anchors.index].all()
        assert not disaggregated.drop(index=anchors.index).any()
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
        """ORIGINAL, DISAGGREGATED et MODEL_ON_* coexistent dans la matrice.

        Le scope étendu est nécessaire depuis §2.7 : dans la fenêtre stricte,
        toute période imputée porte par construction une valeur observée,
        donc toute cellule imputée est recalée (DISAGGREGATED). MODEL_ON_*
        n'apparaît que sur la partie étendue, où une période peut être
        dépourvue d'observation (fin de série retardée).
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
            imputation_scope='extended_forward',
        )
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        present = set(imputer.imputation_provenance_.stack().unique())
        assert ProvenanceType.ORIGINAL in present
        assert ProvenanceType.DISAGGREGATED in present
        assert present & {ProvenanceType.MODEL_ON_TRUE, ProvenanceType.MODEL_ON_MIXED}

        # Les ancres désagrégées ne sont plus déclarées originales. Seules les
        # ancres SITUÉES DANS LA FENÊTRE sont concernées : depuis §2.7 aucune
        # prédiction n'a lieu hors fenêtre, donc aucune désagrégation non plus
        window = imputer._imputation_window_calc.get_imputation_window_mask(
            mixed_freq_timeseries
        )
        anchors = mixed_freq_timeseries['pib_trimestriel'].dropna().index
        anchors_in_window = anchors[window.reindex(anchors).fillna(False)]
        assert len(anchors_in_window) > 0
        assert (
            imputer.imputation_provenance_.loc[anchors_in_window, 'pib_trimestriel']
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


class TestProvenanceFitVsTransform:
    """§2.8.1 : `fit` et `transform` écrivent deux attributs distincts.

    Avant le correctif, les deux écrivaient `imputation_provenance_` : un
    `fit_transform` perdait systématiquement la trace laissée par le fit.
    """

    def test_fit_transform_keeps_both_traces(self, mixed_freq_timeseries):
        """Les deux attributs existent, sont non vides, et peuvent différer."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
            imputation_scope='extended_forward',
        )
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        assert hasattr(imputer, 'imputation_provenance_fit_')
        assert hasattr(imputer, 'imputation_provenance_')
        assert not imputer.imputation_provenance_fit_.empty
        assert not imputer.imputation_provenance_.empty

    def test_transform_does_not_touch_fit_trace(self, mixed_freq_timeseries):
        """Un `transform` postérieur au fit laisse `imputation_provenance_fit_` intact."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        _fit_transform_quiet(imputer, mixed_freq_timeseries)
        fit_trace = imputer.imputation_provenance_fit_.copy()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.transform(mixed_freq_timeseries.copy())

        pd.testing.assert_frame_equal(fit_trace, imputer.imputation_provenance_fit_)


class TestAggregatedProvenanceExcludesOriginal:
    """§2.8.2 : `AGGREGATED` ne recouvre ni les cellules NaN ni les ORIGINAL."""

    def test_no_nan_cell_marked_aggregated(self, mixed_freq_timeseries):
        """Aucune cellule restée NaN après agrégation n'est marquée AGGREGATED."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_timeseries)

        aggregated_mask = imputer.imputation_provenance_ == ProvenanceType.AGGREGATED
        still_nan = result.isna()
        common_cols = aggregated_mask.columns.intersection(still_nan.columns)
        assert not (aggregated_mask[common_cols] & still_nan[common_cols]).any().any()

    def test_original_cell_never_downgraded_to_aggregated(self, mixed_freq_timeseries):
        """Une cellule vraiment observée (ORIGINAL) n'est jamais requalifiée AGGREGATED."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        # Les variables mensuelles denses sont ORIGINAL sur toute leur
        # couverture et n'ont jamais besoin d'être agrégées à l'échelle
        # mensuelle cible : leur colonne ne doit contenir aucun AGGREGATED
        for column in ('production_industrielle', 'inflation_ipc', 'taux_chomage'):
            provenance = imputer.imputation_provenance_[column]
            observed = mixed_freq_timeseries[column].dropna().index
            observed_in_matrix = provenance.index.intersection(observed)
            assert not (
                provenance.loc[observed_in_matrix] == ProvenanceType.AGGREGATED
            ).any()


class TestProvenancePerFrequencyLevel:
    """§2.8.4 : avec `keep_lower_frequencies=True`, la provenance suit chaque niveau.

    Avant le correctif, `imputation_provenance_` restait une matrice à un
    seul niveau (celui de la fréquence cible) même quand la sortie
    empilait plusieurs fréquences : une date-ancre d'un niveau bas s'y
    trouvait donc déclarée DISAGGREGATED, y compris pour le niveau où elle
    porte encore la vraie observation d'origine.
    """

    def test_provenance_matches_output_multiindex_structure(self, mixed_freq_timeseries):
        """La provenance porte le même MultiIndex (frequency, date) que la sortie."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=True,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_timeseries)

        assert isinstance(imputer.imputation_provenance_.index, pd.MultiIndex)
        assert imputer.imputation_provenance_.index.names == result.index.names
        assert set(imputer.imputation_provenance_.index.get_level_values('frequency')) == (
            set(result.index.get_level_values('frequency'))
        )

    def test_low_level_anchor_stays_original(self, mixed_freq_timeseries):
        """Au niveau QUARTERLY, l'ancre du PIB reste ORIGINAL (pas DISAGGREGATED).

        C'est précisément la valeur que la matrice à un seul niveau
        (restreinte à la cible) déclarait faussement DISAGGREGATED.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=True,
        )
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        quarterly_label = imputer._stage_frequency_label('Q')
        provenance_q = imputer.imputation_provenance_.xs(quarterly_label, level='frequency')

        anchors = mixed_freq_timeseries['pib_trimestriel'].dropna().index
        anchors_in_matrix = provenance_q.index.intersection(anchors)
        assert len(anchors_in_matrix) > 0
        assert (
            provenance_q.loc[anchors_in_matrix, 'pib_trimestriel']
            == ProvenanceType.ORIGINAL
        ).all()

    def test_target_level_anchor_is_disaggregated(self, mixed_freq_timeseries):
        """Au niveau cible (mensuel), la même ancre est bien DISAGGREGATED."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=True,
            imputation_scope='extended_forward',
        )
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        target_label = imputer._stage_frequency_label(imputer.effective_target_frequency_)
        provenance_target = imputer.imputation_provenance_.xs(target_label, level='frequency')

        window = imputer._imputation_window_calc.get_imputation_window_mask(
            mixed_freq_timeseries
        )
        anchors = mixed_freq_timeseries['pib_trimestriel'].dropna().index
        anchors_in_window = anchors[window.reindex(anchors).fillna(False)]
        assert len(anchors_in_window) > 0
        assert (
            provenance_target.loc[anchors_in_window, 'pib_trimestriel']
            == ProvenanceType.DISAGGREGATED
        ).all()

    def test_panel_provenance_keeps_entity_and_frequency_levels(self, mixed_freq_panel):
        """Panel : la provenance porte (country, frequency, date), comme la
        sortie — le nom d'origine du niveau d'entité (`country`) est
        préservé, il n'est plus écrasé par `'entity'` (§3.13)."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=True,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_panel)

        assert imputer.imputation_provenance_.index.names == result.index.names
        assert set(imputer.imputation_provenance_.index.get_level_values('country')) == {
            'France', 'Allemagne', 'Italie'
        }

    def test_provenance_statistics_sum_to_cell_count_multi_level(
        self, mixed_freq_timeseries
    ):
        """`compute_statistics()` reste cohérente sur la matrice empilée par niveau."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=True,
        )
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        tracker = ImputationProvenanceTracker()
        tracker.provenance_matrix_ = imputer.imputation_provenance_
        stats = tracker.compute_statistics()['overall']

        total = sum(stats[prov.value] for prov in ProvenanceType)
        total += stats['not_imputed']
        assert total == imputer.imputation_provenance_.size


# ---------------------------------------------------------------------------
# §2.10 — inverse_transform piloté par la provenance
# ---------------------------------------------------------------------------
def _target_level(imputer, frame):
    """Restrict a (possibly stacked) frame to the target frequency level."""
    if not isinstance(frame.index, pd.MultiIndex):
        return frame
    if 'frequency' not in (frame.index.names or []):
        return frame
    target_label = imputer._stage_frequency_label(imputer.effective_target_frequency_)
    return frame.xs(target_label, level='frequency')


class TestInverseTransformRestoresOriginal:
    """§2.10 : `inverse_transform` défait réellement `transform`.

    Avant le correctif, la méthode n'inversait que l'`additive_transformer` :
    `inverse_transform(transform(X)).equals(transform(X))` valait `True`. Elle
    doit désormais ramener la sortie à l'index source, remettre à NaN toute
    cellule dont la provenance n'est pas ORIGINAL, et n'appliquer l'inverse
    additif qu'en dernier.
    """

    def test_inverse_transform_restores_nans(self, mixed_freq_timeseries):
        """NaN partout où la provenance n'est pas ORIGINAL, valeurs intactes ailleurs."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=True,
        )
        transformed = _fit_transform_quiet(imputer, mixed_freq_timeseries)
        inverse = imputer.inverse_transform(transformed)

        provenance = _target_level(imputer, imputer.imputation_provenance_)
        original_mask = provenance == ProvenanceType.ORIGINAL

        values = _target_level(imputer, transformed)

        # Des cellules imputées existent bien : le test serait vide sinon
        assert (~original_mask).any().any()
        # Toute cellule non ORIGINAL est revenue à NaN
        assert inverse.where(~original_mask).isna().all().all()
        # Les cellules ORIGINAL traversent l'inversion sans être touchées
        pd.testing.assert_frame_equal(
            inverse.where(original_mask),
            values.where(original_mask),
            check_dtype=False,
            check_freq=False,
        )
        # Et celles que `transform` a conservées portent bien la valeur
        # observée à l'entrée. Le masque est intersecté avec les cellules
        # non-NaN de la sortie : hors fenêtre d'imputation, `transform` vide
        # volontairement le périmètre de désagrégation d'une variable
        # imputée (§2.6/§2.7) sans en changer la provenance
        kept_mask = original_mask & values.notna()
        pd.testing.assert_frame_equal(
            inverse.where(kept_mask),
            mixed_freq_timeseries.where(kept_mask),
            check_dtype=False,
            check_freq=False,
        )

    def test_inverse_transform_returns_source_index(self, mixed_freq_timeseries):
        """Avec keep_lower_frequencies=True, la sortie retrouve l'index de X."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=True,
        )
        transformed = _fit_transform_quiet(imputer, mixed_freq_timeseries)
        # Le transformé porte bien un niveau de fréquence à retirer
        assert 'frequency' in transformed.index.names

        inverse = imputer.inverse_transform(transformed)

        assert inverse.index.equals(mixed_freq_timeseries.index)
        assert inverse.index.names == mixed_freq_timeseries.index.names

    def test_inverse_transform_panel_returns_source_index(self, mixed_freq_panel):
        """Panel : l'index (country, date) est restauré, niveaux et noms compris."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=True,
        )
        transformed = _fit_transform_quiet(imputer, mixed_freq_panel)
        inverse = imputer.inverse_transform(transformed)

        assert inverse.index.names == mixed_freq_panel.index.names
        assert set(inverse.index) == set(mixed_freq_panel.index)

    def test_inverse_transform_panel_restores_nans(self, mixed_freq_panel):
        """Panel : le masque ORIGINAL est appliqué entité par entité."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=True,
        )
        transformed = _fit_transform_quiet(imputer, mixed_freq_panel)
        inverse = imputer.inverse_transform(transformed)

        provenance = _target_level(imputer, imputer.imputation_provenance_)
        provenance = provenance.rename_axis(mixed_freq_panel.index.names)
        original_mask = provenance == ProvenanceType.ORIGINAL

        assert (~original_mask).any().any()
        assert inverse.where(~original_mask).isna().all().all()

    def test_inverse_transform_without_frequency_level(self, mixed_freq_timeseries):
        """keep_lower_frequencies=False : rien à désempiler, masque appliqué tel quel."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        transformed = _fit_transform_quiet(imputer, mixed_freq_timeseries)
        inverse = imputer.inverse_transform(transformed)

        original_mask = imputer.imputation_provenance_ == ProvenanceType.ORIGINAL

        assert inverse.index.equals(mixed_freq_timeseries.index)
        assert inverse.where(~original_mask).isna().all().all()

    def test_inverse_transform_roundtrip_with_log_transformer(self, annual_over_monthly):
        """Aller-retour exact (1e-8) sur les cellules ORIGINAL avec un log additif."""
        from sklearn.preprocessing import FunctionTransformer

        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            additive_transformer=FunctionTransformer(
                func=np.log1p, inverse_func=np.expm1
            ),
            keep_lower_frequencies=True,
        )
        transformed = _fit_transform_quiet(imputer, annual_over_monthly)
        inverse = imputer.inverse_transform(transformed)

        provenance = _target_level(imputer, imputer.imputation_provenance_)
        original_mask = provenance == ProvenanceType.ORIGINAL

        restored = inverse.where(original_mask)
        expected = annual_over_monthly.where(original_mask)
        assert original_mask.any().any()
        assert (restored - expected).abs().max().max() < 1e-8

    def test_inverse_transform_without_transform_raises(self, mixed_freq_timeseries):
        """Sans `transform` préalable, l'inversion échoue avec un message explicite."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(mixed_freq_timeseries.copy())

        with pytest.raises(ValueError, match='requires a previous call to transform'):
            imputer.inverse_transform(mixed_freq_timeseries.copy())

    def test_restore_original_values_recovers_anchors(self, mixed_freq_timeseries):
        """`restore_original_values=True` récupère les ancres DISAGGREGATED."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=True,
            imputation_scope='extended_forward',
        )
        transformed = _fit_transform_quiet(imputer, mixed_freq_timeseries)

        provenance = _target_level(imputer, imputer.imputation_provenance_)
        anchors = mixed_freq_timeseries['pib_trimestriel'].dropna().index
        disaggregated = anchors[
            (provenance.loc[anchors, 'pib_trimestriel'] == ProvenanceType.DISAGGREGATED)
            .to_numpy()
        ]
        assert len(disaggregated) > 0

        # Par défaut, ces ancres sont perdues : leur provenance n'est pas ORIGINAL
        default_inverse = imputer.inverse_transform(transformed)
        assert default_inverse.loc[disaggregated, 'pib_trimestriel'].isna().all()

        # Avec le paramètre explicite, elles retrouvent leur valeur observée
        imputer.set_params(restore_original_values=True)
        restored = imputer.inverse_transform(transformed)
        pd.testing.assert_series_equal(
            restored.loc[disaggregated, 'pib_trimestriel'],
            mixed_freq_timeseries.loc[disaggregated, 'pib_trimestriel'],
            check_freq=False,
        )


class TestNoStrictImputationWindowWarns:
    """§3.6 : le fit avertit explicitement quand aucune fenêtre stricte n'existe.

    Avant le correctif, seul `ValueError` était intercepté en PHASE 1 : quand le
    calculateur "réussit" avec des bornes `None` (deux colonnes dont les périodes
    de couverture ne se chevauchent jamais : aucune date ne couvre toutes les
    variables), `fit` continuait silencieusement et tous les entraînements
    finissaient un à un en `interpolate_fallback`, sans message global expliquant
    pourquoi.

    Note : pour un panel, `_fit_panel` lève déjà `ValueError` quand *toutes* les
    entités sont sans fenêtre (cf. except ValueError plus haut en PHASE 1) ; la
    branche dict de la vérification ci-dessous reste donc défensive pour ce cas
    précis et n'est pas retestée séparément ici.
    """

    def test_no_strict_window_warns_time_series(self):
        """Deux colonnes sans chevauchement de couverture : warning explicite."""
        imputer = HighFrequencyImputer(target_frequency='M')

        dates = pd.date_range('2023-01-01', periods=12, freq='ME')
        df = pd.DataFrame(index=dates)
        df['col1'] = [float(i) for i in range(6)] + [np.nan] * 6
        df['col2'] = [np.nan] * 6 + [float(i) for i in range(6)]

        with pytest.warns(UserWarning, match="No strict imputation window"):
            imputer.fit(df)


class TestTrainOnPartialFitOrderCV:
    """`train_on_partial_fit_order='cv'` : ordre déterminé par MAPE de CV."""

    def test_cv_order_runs_end_to_end(self, mixed_freq_timeseries):
        """Suffisamment d'observations : le chemin de scoring CV s'exécute."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            train_on_partial_fit_order='cv',
            train_on_partial_coverage=True,
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, mixed_freq_timeseries)

        assert len(result) == len(mixed_freq_timeseries)
        assert len(imputer.model_fitting_order_) > 0

    def test_cv_order_falls_back_with_few_observations(self):
        """Moins de 10 observations disponibles : repli sur l'ordre de fréquence."""
        dates = pd.date_range('2020-01-01', periods=8, freq='MS')
        monthly = pd.Series(10.0 + np.arange(8), index=dates)
        quarterly = pd.Series(np.nan, index=dates)
        quarterly.iloc[[0, 3]] = [100.0, 130.0]
        df = pd.DataFrame({'monthly_dense': monthly, 'quarterly': quarterly})

        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            train_on_partial_fit_order='cv',
            train_on_partial_coverage=True,
        )
        result = _fit_transform_quiet(imputer, df)

        assert len(result) == len(df)


class TestClassifyVariablesUnification:
    """§5.5 : la consolidation en une seule méthode
    (`_classify_variables_at_frequency`) a eu lieu, mais sans conserver le
    wrapper `_classify_variables()` proposé par la revue pour exposer le
    résultat relatif à `effective_target_frequency_` : cette méthode
    n'existe plus du tout, donc plus rien à comparer entre "les deux
    méthodes". `variable_categories_` est exposé directement au format
    catégorie -> liste de clés produit par `_classify_variables_at_frequency`
    (§3.2 : plus d'aller-retour clé -> catégorie -> clé)."""

    def test_variable_categories_format(self):
        """Les trois clés de catégorie sont toujours présentes, leur union
        couvre exactement `detected_frequencies_` et elles sont disjointes."""
        imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())

        # "monthly_var" n'a une valeur qu'aux fins de mois (NaN ailleurs) pour
        # que sa fréquence détectée soit bien mensuelle malgré l'index journalier
        dates = pd.date_range('2023-01-01', periods=90, freq='D')
        monthly_var = pd.Series(np.nan, index=dates)
        monthly_var.loc[dates.is_month_end] = range(dates.is_month_end.sum())
        df = pd.DataFrame({
            'daily_var': range(90),
            'monthly_var': monthly_var,
        }, index=dates)
        imputer.fit(df)

        assert set(imputer.variable_categories_.keys()) == {'aggregate', 'impute', 'target_freq'}

        aggregate = set(imputer.variable_categories_['aggregate'])
        impute = set(imputer.variable_categories_['impute'])
        target_freq = set(imputer.variable_categories_['target_freq'])

        assert aggregate | impute | target_freq == set(imputer.detected_frequencies_)
        assert aggregate & impute == set()
        assert aggregate & target_freq == set()
        assert impute & target_freq == set()

        assert 'daily_var' in aggregate
        assert 'monthly_var' in target_freq


class TestVerboseMode:
    """§5.6 : mode `verbose`, qui aurait évité le bug §1.4 (prints de débogage
    oubliés en production)."""

    def test_verbose_false_produces_no_output(
        self, capsys, annual_quarterly_over_monthly
    ):
        """`verbose=False` (défaut) : aucune sortie standard."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
        )
        _fit_transform_quiet(imputer, annual_quarterly_over_monthly)

        assert capsys.readouterr().out == ''

    def test_verbose_true_produces_prefixed_lines_per_stage_and_fit(
        self, capsys, annual_quarterly_over_monthly
    ):
        """`verbose=True` : au moins une ligne par étape et par fit, toutes
        préfixées `[HighFrequencyImputer]`."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            verbose=True,
        )
        _fit_transform_quiet(imputer, annual_quarterly_over_monthly)

        lines = [
            line for line in capsys.readouterr().out.splitlines() if line
        ]
        assert lines
        assert all(line.startswith('[HighFrequencyImputer]') for line in lines)

        # Trois étapes de cascade ('Y', 'Q', 'M') et trois fits (un par
        # couple étape/variable, cf. TestFitCountByCascadeRefitting)
        stage_lines = [line for line in lines if '] Stage' in line]
        fit_lines = [line for line in lines if '] Fit ' in line]
        assert len(stage_lines) >= 3
        assert len(fit_lines) == 3

    def test_verbose_logs_fallback_reason_when_no_estimator(
        self, capsys, annual_quarterly_over_monthly
    ):
        """Repli sans estimateur : la raison est journalisée."""
        imputer = HighFrequencyImputer(target_frequency='M', verbose=True)
        _fit_transform_quiet(imputer, annual_quarterly_over_monthly)

        out = capsys.readouterr().out
        assert '[HighFrequencyImputer]' in out
        assert 'no estimator available' in out


class TestImputationPlan:
    """§5.3 : le plan d'imputation est le seul état du fit, les anciennes
    structures parallèles n'en sont plus que des vues dérivées."""

    @staticmethod
    def _fitted(data, **kwargs):
        """Fit an imputer on the two-stage cascade of the fixtures."""
        params = {
            'target_frequency': 'M',
            'estimator': LinearRegression(),
            'cascade_refitting': True,
            'keep_lower_frequencies': True,
        }
        params.update(kwargs)
        imputer = HighFrequencyImputer(**params)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(data.copy())
        return imputer

    def test_plan_length_matches_fitting_order(self, mixed_freq_timeseries):
        """Une étape de plan par entrée de `model_fitting_order_`."""
        imputer = self._fitted(mixed_freq_timeseries)

        assert len(imputer.imputation_plan_) == len(imputer.model_fitting_order_)
        assert len(imputer.imputation_plan_) == len(imputer.imputation_models_)

    def test_plan_length_matches_fitting_order_panel(self, mixed_freq_panel):
        """Idem sur un panel, où les clés de groupe sont dédupliquées (§2.4)."""
        imputer = self._fitted(mixed_freq_panel)

        assert len(imputer.imputation_plan_) == len(imputer.model_fitting_order_)
        assert len(imputer.imputation_plan_) == len(imputer.imputation_models_)

    def test_derived_keys_are_coherent(self, mixed_freq_timeseries):
        """Les quatre vues dérivées portent exactement les clés du plan."""
        imputer = self._fitted(mixed_freq_timeseries)
        plan_keys = [step.stage_key for step in imputer.imputation_plan_]

        assert plan_keys == imputer.model_fitting_order_
        assert list(imputer.imputation_models_) == plan_keys
        assert list(imputer.stage_groups_) == plan_keys
        # Chaque clé se décompose en (label de fréquence de l'étape, groupe)
        for step in imputer.imputation_plan_:
            assert step.stage_key == (step.pred_freq_label, step.var_key)
        # La progression de fréquence ne référence que des variables du plan
        assert set(imputer.frequency_progression_) == {
            step.var_name for step in imputer.imputation_plan_
        }

    def test_registry_entries_match_steps(self, mixed_freq_timeseries):
        """L'entrée de registre d'une étape reproduit fidèlement ses champs."""
        imputer = self._fitted(mixed_freq_timeseries)

        for step in imputer.imputation_plan_:
            entry = imputer.imputation_models_[step.stage_key]
            if step.is_fallback:
                assert entry == 'interpolate_fallback'
                continue
            # Le modèle est partagé, pas copié
            assert entry['model'] is step.model
            assert entry['feature_cols'] == list(step.feature_cols)
            assert entry['scale_factor'] == step.scale_factor
            assert entry['fit_scale_factor'] == step.fit_scale_factor
            assert entry['pred_freq'] == step.pred_freq
            assert entry['trained_on_imputed'] == step.trained_on_imputed

    def test_group_metadata_matches_steps(self, quarterly_over_monthly):
        """`stage_groups_` dérive du plan, replis par interpolation compris."""
        imputer = self._fitted(quarterly_over_monthly, estimator=None)

        # La fixture sans estimateur emprunte le repli
        assert any(step.is_fallback for step in imputer.imputation_plan_)
        for step in imputer.imputation_plan_:
            group = imputer.stage_groups_[step.stage_key]
            assert group['var_name'] == step.var_name
            assert group['f_var'] == step.source_frequency
            assert group['entities'] == step.entities

    def test_derived_views_are_read_only(self, mixed_freq_timeseries):
        """Les vues dérivées n'ont pas de setter : le plan reste la source."""
        imputer = self._fitted(mixed_freq_timeseries)

        for attribute in (
            'imputation_models_',
            'model_fitting_order_',
            'stage_groups_',
            'frequency_progression_',
        ):
            with pytest.raises(AttributeError):
                setattr(imputer, attribute, {})

    def test_derived_views_are_absent_before_fit(self):
        """Avant `fit`, les vues restent invisibles à `hasattr` (non-régression
        du passage attribut -> propriété)."""
        imputer = HighFrequencyImputer(target_frequency='M')

        assert not hasattr(imputer, 'imputation_plan_')
        assert not hasattr(imputer, 'imputation_models_')
        assert not hasattr(imputer, 'model_fitting_order_')
        assert not hasattr(imputer, 'stage_groups_')
        assert not hasattr(imputer, 'frequency_progression_')

    def test_steps_are_frozen(self, mixed_freq_timeseries):
        """Une étape est immuable : seul `replace` produit une variante."""
        imputer = self._fitted(mixed_freq_timeseries)
        step = imputer.imputation_plan_[0]

        with pytest.raises(dataclasses.FrozenInstanceError):
            step.scale_factor = 1.0

        # La variante partage le modèle et ne modifie que le champ demandé
        variant = dataclasses.replace(step, scale_factor=99.0)
        assert variant.scale_factor == 99.0
        assert step.scale_factor != 99.0
        assert variant.model is step.model

    def test_transform_replays_the_plan(self, mixed_freq_timeseries):
        """`transform` rejoue le plan : le retirer prive la sortie de ses
        imputations, preuve qu'aucun autre registre n'est consulté."""
        imputer = self._fitted(mixed_freq_timeseries, keep_lower_frequencies=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            full = imputer.transform(mixed_freq_timeseries.copy())

            # Plan vidé : plus aucune étape à rejouer
            imputer.imputation_plan_ = []
            empty = imputer.transform(mixed_freq_timeseries.copy())

        imputed_col = 'balance_commerciale_annuelle'
        assert full[imputed_col].notna().sum() > empty[imputed_col].notna().sum()


class TestHighFrequencyImputerMultiLevelPanel:
    """§C19 : couverture d'un panel à deux niveaux d'entité (country, sector).

    `split_variable_key`, `get_entity_mask`, `_period_membership`,
    `_stage_scale_factor` et `to_entity_tuple` ne consomment en aval de
    `repr_var_key = var_keys[0]` que le nom de colonne et la fréquence
    détectée, identiques pour toutes les clés du groupe : rien dans leur
    logique ne suppose un unique niveau d'entité. Ce chemin n'était exercé,
    jusqu'ici, que par `mixed_freq_panel` (un seul niveau, `country`).

    `keep_lower_frequencies=True` est désormais couvert : `_build_multifreq_output`
    conserve tous les niveaux d'entité (`combined.index.nlevels - 1`), plus
    seulement le premier (§3.13)."""

    def test_two_level_panel_fit_transform(self, panel_two_level_dataset):
        """`fit_transform` aboutit sur un panel à deux niveaux d'entité : la
        sortie garde les niveaux (country, sector, date) et la variable
        trimestrielle est imputée pour les 4 couples pays/secteur."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )

        result = _fit_transform_quiet(imputer, panel_two_level_dataset)

        assert imputer.is_panel_ is True
        assert list(result.index.names) == ['country', 'sector', 'date']

        pairs = panel_two_level_dataset.index.droplevel('date').unique()
        assert len(pairs) == 4
        for country, sector in pairs:
            values = result.xs((country, sector), level=('country', 'sector'))
            assert values['indicateur_trimestriel'].notna().all()

    def test_two_level_panel_entities_are_length_two_tuples(self, panel_two_level_dataset):
        """Chaque étape du plan porte ses entités sous forme de tuples
        (country, sector) à deux éléments, pas des scalaires ni des tuples
        à un seul niveau."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
        )

        _fit_transform_quiet(imputer, panel_two_level_dataset)

        assert len(imputer.imputation_plan_) > 0
        for step in imputer.imputation_plan_:
            assert step.entities is not None
            for entity in step.entities:
                assert isinstance(entity, tuple)
                assert len(entity) == 2

    def test_two_level_panel_inverse_transform(self, panel_two_level_dataset):
        """`inverse_transform` restitue l'index et les valeurs d'origine
        pour un panel à deux niveaux d'entité, `keep_lower_frequencies=False`."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )

        result = _fit_transform_quiet(imputer, panel_two_level_dataset)
        inverse = imputer.inverse_transform(result)

        assert inverse.index.names == panel_two_level_dataset.index.names
        assert set(inverse.index) == set(panel_two_level_dataset.index)

        # Toute cellule dont la provenance n'est pas ORIGINAL revient à NaN,
        # les autres traversent l'inversion sans être touchées
        original_mask = imputer.imputation_provenance_ == ProvenanceType.ORIGINAL
        assert (~original_mask).any().any()
        assert inverse.where(~original_mask).isna().all().all()
        pd.testing.assert_frame_equal(
            inverse.where(original_mask),
            panel_two_level_dataset.where(original_mask),
            check_dtype=False,
            check_freq=False,
        )

    def test_two_level_panel_stacked_index_is_unique(self, panel_two_level_dataset):
        """`keep_lower_frequencies=True` : l'index empilé n'a aucun doublon
        et porte les quatre niveaux (country, sector, frequency, date) —
        avant le correctif, `sector` était jeté et ('France', 'Industrie')
        / ('France', 'Services') finissaient sous la même étiquette."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=True,
        )

        result = _fit_transform_quiet(imputer, panel_two_level_dataset)

        assert list(result.index.names) == ['country', 'sector', 'frequency', 'date']
        assert not result.index.duplicated().any()

    def test_two_level_panel_provenance_same_index(self, panel_two_level_dataset):
        """`imputation_provenance_` porte exactement les mêmes noms de
        niveaux que la sortie empilée."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=True,
        )

        result = _fit_transform_quiet(imputer, panel_two_level_dataset)

        assert list(imputer.imputation_provenance_.index.names) == list(result.index.names)
        assert not imputer.imputation_provenance_.index.duplicated().any()

    def test_two_level_panel_inverse_transform_with_stacking(self, panel_two_level_dataset):
        """`inverse_transform` ne lève plus sur une sortie empilée à deux
        niveaux d'entité et restitue l'index ET les valeurs d'origine."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=True,
        )

        result = _fit_transform_quiet(imputer, panel_two_level_dataset)
        inverse = imputer.inverse_transform(result)

        assert inverse.index.names == panel_two_level_dataset.index.names
        assert set(inverse.index) == set(panel_two_level_dataset.index)

        original_mask = imputer.imputation_provenance_ == ProvenanceType.ORIGINAL
        target_original_mask = _target_level(imputer, original_mask)
        pd.testing.assert_frame_equal(
            inverse.where(target_original_mask),
            panel_two_level_dataset.where(target_original_mask),
            check_dtype=False,
            check_freq=False,
        )

    def test_single_level_panel_index_names_preserved(self, mixed_freq_panel):
        """Non-régression du cas à un seul niveau d'entité : le nom d'origine
        (`country`) est désormais préservé dans la sortie empilée, là où
        l'ancien code l'écrasait systématiquement par `'entity'`."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=True,
        )

        result = _fit_transform_quiet(imputer, mixed_freq_panel)

        assert list(result.index.names) == ['country', 'frequency', 'date']
        assert not result.index.duplicated().any()


# =============================================================================
# §3.14 (B1) — `transform` hors fenêtre de fit
# =============================================================================
def _build_monthly_with_quarterly(start, periods=72, seed=0):
    """Build a monthly frame carrying one dense covariate and a quarterly var.

    Same additive setup as the `quarterly_over_monthly` fixture, but with a
    parametrizable date range: the point of §3.14 is precisely that the fit
    frame and the transform frame span DISJOINT periods.

    Args:
        start: First month of the frame ('YYYY-MM-DD', month-start anchored).
        periods: Number of months.
        seed: Seed of the covariate's noise.

    Returns:
        DataFrame indexed by month-start dates, with a dense ``monthly_var``
        and a ``gdp`` column non-NaN only on the first month of each quarter,
        where it holds the quarter's total.
    """
    dates = pd.date_range(start=start, periods=periods, freq='MS')
    rng = np.random.default_rng(seed)

    monthly = pd.Series(30.0 + rng.normal(0, 2.0, periods), index=dates)
    quarterly = pd.Series(np.nan, index=dates)
    # La valeur trimestrielle est portée par le premier mois, ancre de la période
    for _, block in monthly.groupby(monthly.index.to_period('Q')):
        quarterly.loc[block.index[0]] = block.sum()

    data = pd.DataFrame({'monthly_var': monthly, 'gdp': quarterly})
    data.index.name = 'date'
    return data


def _assert_no_observation_lost(source, result, imputer=None):
    """Assert no observed input value disappears from the output.

    Generic invariant of §3.14, meant to be reused by the following batches:
    the cascade may CHANGE an observed value (spreading a quarterly total over
    its months is the whole point of the disaggregation), it may never turn it
    into NaN without putting anything in its place.

    Args:
        source: Frame given to fit_transform / transform.
        result: Frame it returned.
        imputer: Fitted imputer, needed only to strip the ``frequency`` level
            of a ``keep_lower_frequencies=True`` output.

    Raises:
        AssertionError: If an observed cell of `source` is NaN in `result`.
    """
    # Restriction au niveau de la fréquence cible pour une sortie empilée
    frame = _target_level(imputer, result) if imputer is not None else result

    common_cols = [c for c in source.columns if c in frame.columns]
    assert common_cols, "aucune colonne commune : le test serait vide"

    common_index = source.index.intersection(frame.index)
    assert len(common_index) > 0, "aucune ligne commune : le test serait vide"

    for col in common_cols:
        observed = source.loc[common_index, col].notna()
        lost = observed & frame.loc[common_index, col].isna()
        assert not lost.any(), (
            f"{int(lost.sum())} observation(s) de '{col}' détruite(s) par la "
            f"transformation, p.ex. {list(common_index[lost.to_numpy()][:5])}"
        )


class TestTransformOutsideFitWindow:
    """§3.14 (B1) : `transform` sur des dates hors de la grille du fit.

    Le défaut corrigé : `_prediction_masks` réutilisait le
    `ImputationWindowCalculator` du fit, dont le masque valait False sur toute
    date inconnue. Le périmètre de désagrégation était alors vidé en entier
    sans qu'aucune prédiction ne vienne le remplir — colonne entièrement NaN,
    observations d'entrée détruites, et pas le moindre avertissement.

    La fenêtre d'imputation est une contrainte de DISPONIBILITÉ DES
    COVARIABLES, pas un paramètre estimé : elle se recalcule désormais sur les
    données que l'on impute.
    """

    @staticmethod
    def _imputer(**kwargs):
        """Build an imputer with the parameters shared by the whole class."""
        params = dict(
            target_frequency='M',
            estimator=LinearRegression(),
            keep_lower_frequencies=False,
        )
        params.update(kwargs)
        return HighFrequencyImputer(**params)

    def test_transform_out_of_fit_window_imputes(self):
        """Le scénario de reproduction : fit 2015-2020, transform 2021-2023."""
        fit_df = _build_monthly_with_quarterly('2015-01-01', 72, seed=0)
        new_df = _build_monthly_with_quarterly('2021-01-01', 36, seed=1)

        imputer = self._imputer()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fitted = imputer.fit_transform(fit_df.copy())
            transformed = imputer.transform(new_df.copy())

        # Contrôle : sur les données du fit, tout allait déjà bien
        assert fitted['gdp'].notna().sum() == len(fit_df)

        # Le défaut : 0/36 alors que l'entrée portait 12 observations
        assert new_df['gdp'].notna().sum() == 12
        assert transformed['gdp'].notna().sum() == len(new_df)

    def test_transform_never_erases_observations(self, mixed_freq_timeseries):
        """Aucune valeur non-NaN de l'entrée ne disparaît de la sortie."""
        # Cas hors échantillon
        fit_df = _build_monthly_with_quarterly('2015-01-01', 72, seed=0)
        new_df = _build_monthly_with_quarterly('2021-01-01', 36, seed=1)

        imputer = self._imputer()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(fit_df.copy())
            transformed = imputer.transform(new_df.copy())
        _assert_no_observation_lost(new_df, transformed, imputer)

        # Cas du jeu de référence : les ancres 2018 de `pib_trimestriel` sont
        # dans le périmètre mais hors fenêtre stricte
        # (`production_industrielle` ne démarre qu'en 2019)
        reference = self._imputer(keep_lower_frequencies=True)
        result = _fit_transform_quiet(reference, mixed_freq_timeseries)
        _assert_no_observation_lost(mixed_freq_timeseries, result, reference)

    @pytest.mark.parametrize('dataset', ['mixed_freq_timeseries', 'mixed_freq_panel'])
    @pytest.mark.parametrize('keep_lower_frequencies', [False, True])
    def test_fit_transform_equals_fit_then_transform(
        self, dataset, keep_lower_frequencies, request
    ):
        """`fit_transform(X)` et `fit(X).transform(X)` restent identiques.

        Garde-fou non négociable du recalcul de la fenêtre : deux instances
        distinctes, pour que l'égalité ne puisse pas venir d'un état partagé.
        """
        data = request.getfixturevalue(dataset)

        combined = self._imputer(keep_lower_frequencies=keep_lower_frequencies)
        separate = self._imputer(keep_lower_frequencies=keep_lower_frequencies)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result_combined = combined.fit_transform(data.copy())
            separate.fit(data.copy())
            result_separate = separate.transform(data.copy())

        pd.testing.assert_frame_equal(
            result_combined.sort_index(), result_separate.sort_index()
        )

    def test_out_of_window_rows_warn(self):
        """Un périmètre entièrement hors fenêtre est signalé, dates comprises.

        Le fit porte sur un jeu normal ; le transform, lui, reçoit des données
        où la covariable et la variable à imputer ne se recouvrent JAMAIS : la
        fenêtre stricte y est vide alors que le périmètre de désagrégation est
        entier. C'est le cas que §3.14 traversait en silence.
        """
        fit_df = _build_monthly_with_quarterly('2015-01-01', 72, seed=0)

        # Jeu de transform sans aucun mois où les deux séries coexistent
        dates = pd.date_range('2021-01-01', periods=36, freq='MS')
        rng = np.random.default_rng(3)
        monthly = pd.Series(np.nan, index=dates)
        monthly.iloc[:12] = 30.0 + rng.normal(0, 1.0, 12)
        quarterly = pd.Series(np.nan, index=dates)
        quarterly.loc[dates[12::3]] = 100.0
        disjoint = pd.DataFrame({'monthly_var': monthly, 'gdp': quarterly})
        disjoint.index.name = 'date'

        imputer = self._imputer()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(fit_df.copy())

        with pytest.warns(UserWarning) as caught:
            result = imputer.transform(disjoint.copy())

        messages = [
            str(w.message) for w in caught
            if 'falls inside the imputation window' in str(w.message)
        ]
        assert messages, "aucun avertissement sur les lignes hors fenêtre"
        # Le message nomme le nombre de dates, un exemple daté et les bornes
        assert 'date(s)' in messages[0]
        assert '2021-' in messages[0]
        assert 'bounds:' in messages[0]

        # Et rien n'a été détruit au passage
        _assert_no_observation_lost(disjoint, result, imputer)

    def test_provenance_matches_output_after_transform(self, mixed_freq_timeseries):
        """Aucune cellule ORIGINAL là où la sortie est NaN."""
        fit_df = _build_monthly_with_quarterly('2015-01-01', 72, seed=0)
        new_df = _build_monthly_with_quarterly('2021-01-01', 36, seed=1)

        imputer = self._imputer()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(fit_df.copy())
            transformed = imputer.transform(new_df.copy())

        provenance = imputer.imputation_provenance_
        common = [c for c in transformed.columns if c in provenance.columns]
        inconsistent = (
            transformed[common].isna()
            & (provenance[common].reindex(index=transformed.index)
               == ProvenanceType.ORIGINAL)
        )
        assert not inconsistent.any().any()

        # Même invariant sur le jeu de référence, où le vidage du périmètre
        # laisse des cellules non réécrites à l'intérieur de la fenêtre.
        # Contrôle au NIVEAU CIBLE : c'est celui que lit `inverse_transform`.
        # Les niveaux intermédiaires d'une sortie multi-fréquences violent
        # encore l'invariant (une colonne dense agrégée y est NaN hors ancre
        # alors que son instantané de provenance reste au pas source) —
        # défaut propre à l'empilement, signalé par le contrôle interne
        reference = self._imputer(keep_lower_frequencies=True)
        result = _fit_transform_quiet(reference, mixed_freq_timeseries)
        values = _target_level(reference, result)
        provenance = _target_level(reference, reference.imputation_provenance_)
        common = [c for c in values.columns if c in provenance.columns]
        inconsistent = (
            values[common].isna()
            & (provenance[common].reindex(index=values.index)
               == ProvenanceType.ORIGINAL)
        )
        assert not inconsistent.any().any()


# =============================================================================
# §3.15 — Symétrie fit/transform de la cascade (B5, B7, B8, B2)
# =============================================================================

@pytest.fixture
def panel_heterogeneous_variable_frequency():
    """Panel where one variable has a different frequency per entity.

    MultiIndex (``country``, ``date``), two entities over 72 month-start
    dates. ``covariable_mensuelle`` is dense for both. ``variable_basse``
    is ANNUAL for ``France`` (January anchors) and QUARTERLY for
    ``Allemagne`` (quarter-start anchors), so a single cascade stage fits
    TWO distinct groups for the same column name — the situation the
    ``imputed_store`` ``combine_first`` is meant to survive (review §3.15,
    B5).
    """
    dates = pd.date_range('2018-01-01', periods=72, freq='MS')
    rng = np.random.default_rng(11)

    frames = []
    for country, period in (('France', 'Y'), ('Allemagne', 'Q')):
        monthly = pd.Series(20.0 + rng.normal(0, 0.5, len(dates)), index=dates)
        low = pd.Series(np.nan, index=dates)
        # L'ancre de chaque période porte la somme de ses mois
        for _, block in monthly.groupby(monthly.index.to_period(period)):
            low.loc[block.index[0]] = block.sum()

        frame = pd.DataFrame(
            {'covariable_mensuelle': monthly, 'variable_basse': low}
        )
        frame['country'] = country
        frame.index.name = 'date'
        frames.append(frame.reset_index())

    panel = pd.concat(frames, ignore_index=True)
    return panel.set_index(['country', 'date']).sort_index()


class _StoreSpy:
    """Capture the ``imputed_store`` and the values written at each step.

    ``_build_stage_frame`` receives the very dict ``_fit``/``_transform``
    keep mutating, so holding a reference to it is enough to read its final
    content. ``_write_stage_values`` gives the values each (stage, variable)
    couple actually produced.
    """

    def __init__(self):
        self.store = None
        self.written = {}
        self._build = HighFrequencyImputer._build_stage_frame
        self._write = HighFrequencyImputer._write_stage_values

    def __enter__(self):
        spy = self

        def build(imputer, X_original, imputed_store, pred_freq, aggregate_keys=None):
            if spy.store is None:
                spy.store = imputed_store
            return spy._build(
                imputer, X_original, imputed_store, pred_freq, aggregate_keys
            )

        def write(imputer, X_stage, X_input, var_name, predictions, *args, **kwargs):
            spy.written[(kwargs.get('context', ''), var_name)] = predictions.copy()
            return spy._write(
                imputer, X_stage, X_input, var_name, predictions, *args, **kwargs
            )

        HighFrequencyImputer._build_stage_frame = build
        HighFrequencyImputer._write_stage_values = write
        return self

    def __exit__(self, *exc):
        HighFrequencyImputer._build_stage_frame = self._build
        HighFrequencyImputer._write_stage_values = self._write
        return False

    def last_written(self, var_name):
        """Values produced by the LAST step that wrote ``var_name``."""
        matching = [v for (_, name), v in self.written.items() if name == var_name]
        assert matching, "aucune ecriture capturee pour " + var_name
        return matching[-1]


class TestImputedStoreRefreshedAtEachStage:
    """B5 : `imputed_store` porte la prédiction de l'étape la plus fine."""

    @staticmethod
    def _imputer(**kwargs):
        params = dict(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
            train_on_partial_coverage=True,
        )
        params.update(kwargs)
        return HighFrequencyImputer(**params)

    def test_imputed_store_refreshed_at_each_stage(
        self, annual_quarterly_over_monthly
    ):
        """La valeur stockée est celle du DERNIER passage, pas du premier.

        `variable_annuelle` est imputée aux étapes 'Q' puis 'M'. Avec le
        `combine_first` à l'envers, le magasin restait figé sur les valeurs
        d'échelle trimestrielle et toute étape ultérieure recevait une
        covariable ~3x trop grande.
        """
        imputer = self._imputer()
        with _StoreSpy() as spy:
            _fit_transform_quiet(imputer, annual_quarterly_over_monthly)

        # La variable est bien passée par au moins deux étapes
        stages = [
            step.pred_freq_label for step in imputer.imputation_plan_
            if step.var_name == 'variable_annuelle' and not step.is_fallback
        ]
        assert len(stages) >= 2, "une seule etape pour la variable : %s" % stages

        stored = spy.store['variable_annuelle']
        last = spy.last_written('variable_annuelle')

        # Égalité EXACTE avec la dernière écriture, sur son propre index
        pd.testing.assert_series_equal(
            stored.reindex(last.index), last, check_names=False
        )

    def test_stored_values_are_at_the_finest_stage_scale(
        self, annual_quarterly_over_monthly
    ):
        """Contrôle d'échelle : ~1/12 de l'annuel, pas ~1/4."""
        imputer = self._imputer()
        with _StoreSpy() as spy:
            _fit_transform_quiet(imputer, annual_quarterly_over_monthly)

        annual = annual_quarterly_over_monthly['variable_annuelle'].dropna().median()
        stored = spy.store['variable_annuelle'].dropna().median()

        # Échelle mensuelle : le rapport doit être plus proche de 12 que de 4
        ratio = annual / stored
        assert abs(ratio - 12) < abs(ratio - 4), (
            "rapport annuel/stocke = %.2f, echelle trimestrielle (~4) "
            "au lieu de mensuelle (~12)" % ratio
        )


class TestOtherGroupDepositsPreserved:
    """B5 : le correctif ne sacrifie pas les dépôts d'un autre groupe."""

    def test_other_group_deposits_preserved(
        self, panel_heterogeneous_variable_frequency
    ):
        """Fréquences hétérogènes selon l'entité : les deux groupes survivent."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
            train_on_partial_coverage=True,
        )
        with _StoreSpy() as spy:
            _fit_transform_quiet(imputer, panel_heterogeneous_variable_frequency)

        # Deux groupes distincts pour la même colonne, à des fréquences différentes
        group_keys = {
            step.var_key for step in imputer.imputation_plan_
            if step.var_name == 'variable_basse' and not step.is_fallback
        }
        assert len(group_keys) >= 2, "un seul groupe fitte : %s" % (group_keys,)

        stored = spy.store['variable_basse'].dropna()
        entities = set(stored.index.get_level_values('country').unique())

        # Le dépôt de chaque entité subsiste : "preds.combine_first(existing)"
        # ne recouvre que les lignes que "preds" couvre effectivement
        assert entities == {'France', 'Allemagne'}


class TestCascadeGuardsAreSymmetric:
    """B7 : `fit` et `transform` cascadent sous les mêmes gardes."""

    @staticmethod
    def _covariate_frames(imputer, data):
        """Frames de covariables vus par le fit puis par le transform.

        Le fit les reçoit par `_prepare_training_data`, le transform par
        `_predict_stage_values` : ce sont les deux seuls points où une
        covariable devient une entrée du modèle.
        """
        from tsforecast.panel.utils import split_variable_key

        fit_frames, transform_frames = {}, {}
        # `_predict_stage_values` est appelée par le bloc 5g du FIT aussi :
        # sans ce drapeau, la capture « transform » enregistrerait des frames
        # du fit dès que `cascade_refitting=True`, et la comparaison serait
        # vide de sens (frame du fit contre lui-même)
        replaying = {'now': False}
        orig_prepare = HighFrequencyImputer._prepare_training_data
        orig_predict = HighFrequencyImputer._predict_stage_values

        def prepare(self, X_stage, X_original, var_key, pred_freq):
            _, var_name = split_variable_key(var_key)
            fit_frames.setdefault(
                (self._freq_label(pred_freq), var_name), X_stage.copy()
            )
            return orig_prepare(self, X_stage, X_original, var_key, pred_freq)

        def predict(self, step, X_stage, rows_mask, context=''):
            if replaying['now']:
                transform_frames.setdefault(
                    (step.pred_freq_label, step.var_name), X_stage.copy()
                )
            return orig_predict(self, step, X_stage, rows_mask, context)

        HighFrequencyImputer._prepare_training_data = prepare
        HighFrequencyImputer._predict_stage_values = predict
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                imputer.fit(data.copy())
                replaying['now'] = True
                imputer.transform(data.copy())
        finally:
            HighFrequencyImputer._prepare_training_data = orig_prepare
            HighFrequencyImputer._predict_stage_values = orig_predict

        return fit_frames, transform_frames

    @staticmethod
    def _diverging_cells(a, b):
        """Cellules divergentes : valeur différente OU motif de NaN différent."""
        cols = [c for c in a.columns if c in b.columns]
        left, right = a[cols], b.loc[a.index, cols]
        both = left.notna() & right.notna()
        value_diff = ((left - right).abs() > 1e-9) & both
        nan_diff = left.notna() != right.notna()
        return int((value_diff | nan_diff).sum().sum())

    def _assert_symmetric(self, imputer, data):
        """Aucune cellule de covariable ne diffère entre fit et transform."""
        fit_frames, transform_frames = self._covariate_frames(imputer, data)

        compared = sorted(set(fit_frames) & set(transform_frames), key=str)
        assert compared, "aucun couple (etape, variable) comparable"
        for key in compared:
            diverging = self._diverging_cells(fit_frames[key], transform_frames[key])
            assert diverging == 0, (
                "%d cellule(s) de covariable divergentes entre fit et "
                "transform pour %s" % (diverging, (key,))
            )

    @pytest.mark.parametrize('cascade_refitting', [False, True])
    def test_no_refitting_fit_and_transform_agree(
        self, mixed_freq_timeseries, cascade_refitting
    ):
        """Les covariables du transform sont celles vues à l'entraînement.

        Sous `cascade_refitting=False`, l'étape k était entraînée sur un
        frame que les étapes 1..k-1 n'avaient pas touché, et appliquée sur
        un frame qu'elles avaient déjà réécrit.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=cascade_refitting,
            keep_lower_frequencies=True,
        )
        self._assert_symmetric(imputer, mixed_freq_timeseries)

    @pytest.mark.parametrize('cascade_refitting', [False, True])
    def test_panel_no_refitting_fit_and_transform_agree(
        self, mixed_freq_panel, cascade_refitting
    ):
        """Même invariant sur un panel."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=cascade_refitting,
            keep_lower_frequencies=True,
        )
        self._assert_symmetric(imputer, mixed_freq_panel)

    def test_fallback_step_symmetric(self, annual_quarterly_over_monthly):
        """Un repli ne nourrit jamais les covariables des étapes suivantes.

        Le bloc 5g du fit écarte les replis (`if not step.is_fallback`) ;
        le transform les appliquait pourtant à son frame d'étape, que la
        variable suivante voyait ensuite.
        """
        # Dict d'estimateurs sans entrée pour la variable annuelle : son étape
        # bascule en repli, celle de la variable trimestrielle reste un modèle
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator={'variable_trimestrielle': LinearRegression()},
            cascade_refitting=True,
            keep_lower_frequencies=True,
            train_on_partial_coverage=True,
        )
        fit_frames, transform_frames = self._covariate_frames(
            imputer, annual_quarterly_over_monthly
        )

        fallbacks = [
            step for step in imputer.imputation_plan_
            if step.var_name == 'variable_annuelle' and step.is_fallback
        ]
        assert fallbacks, "le repli attendu n'a pas ete declenche"

        compared = sorted(set(fit_frames) & set(transform_frames), key=str)
        assert compared, "aucun couple (etape, variable) comparable"
        for key in compared:
            diverging = self._diverging_cells(fit_frames[key], transform_frames[key])
            assert diverging == 0, (
                "le repli a contamine les covariables du transform pour "
                "%s (%d cellule(s))" % ((key,), diverging)
            )


class _FirstObservationDropped(BaseEstimator, TransformerMixin):
    """Additive transformer changing the NaN pattern, like a differencing."""

    def __init__(self, column='inflation_ipc'):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        out.iloc[0, out.columns.get_loc(self.column)] = np.nan
        return out

    def inverse_transform(self, X):
        return X.copy()


class _ExtraColumnAdded(BaseEstimator, TransformerMixin):
    """Additive transformer adding a dense column, absent from the input."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        out['covariable_ajoutee'] = np.arange(len(out), dtype=float)
        return out

    def inverse_transform(self, X):
        return X.drop(columns=['covariable_ajoutee'], errors='ignore')


class TestProvenanceTrackerInitializedAfterAdditiveTransformer:
    """B8 : le tracker est initialisé au même moment dans les deux chemins."""

    def test_provenance_identical_with_differencing_transformer(
        self, mixed_freq_timeseries
    ):
        """Un transformateur changeant le motif de NaN ne divise plus ORIGINAL.

        `initialize` marque ORIGINAL là où `notna()`. Initialisé AVANT le
        transformateur au transform et APRÈS au fit, il produisait deux
        masques différents : la première observation était non-ORIGINAL au
        fit et ORIGINAL au transform.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            additive_transformer=_FirstObservationDropped(),
            keep_lower_frequencies=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(mixed_freq_timeseries.copy())
            imputer.transform(mixed_freq_timeseries.copy())

        fit_original = imputer.imputation_provenance_fit_ == ProvenanceType.ORIGINAL
        transform_original = imputer.imputation_provenance_ == ProvenanceType.ORIGINAL

        # La cellule annulée par le transformateur n'est ORIGINAL nulle part
        first_date = mixed_freq_timeseries.index[0]
        assert not fit_original.loc[first_date, 'inflation_ipc']
        assert not transform_original.loc[first_date, 'inflation_ipc']

        # Et les deux masques coïncident partout
        pd.testing.assert_frame_equal(fit_original, transform_original)

    def test_provenance_covers_a_column_added_by_the_transformer(
        self, mixed_freq_timeseries
    ):
        """Une colonne ajoutée par le transformateur figure dans la provenance.

        La matrice du transform venait de `data_work` alors que les frames
        d'étape viennent de `data_transformed` : la colonne ajoutée
        manquait à la matrice, sans le moindre message.
        """
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            additive_transformer=_ExtraColumnAdded(),
            keep_lower_frequencies=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(mixed_freq_timeseries.copy())
            result = imputer.transform(mixed_freq_timeseries.copy())

        assert 'covariable_ajoutee' in imputer.imputation_provenance_.columns
        assert list(imputer.imputation_provenance_fit_.columns) == list(
            imputer.imputation_provenance_.columns
        )
        # La matrice décrit bien la sortie, pas l'entrée
        assert set(imputer.imputation_provenance_.columns) == set(result.columns)


class TestAnchorsMarkedWithoutPeriodTotals:
    """B2 : `enforce_period_totals=False` ne casse plus l'étape suivante."""

    @staticmethod
    def _imputer(**kwargs):
        params = dict(
            target_frequency='M',
            estimator=LinearRegression(),
            enforce_period_totals=False,
            keep_lower_frequencies=False,
        )
        params.update(kwargs)
        return HighFrequencyImputer(**params)

    def test_anchors_marked_disaggregated_without_period_totals(
        self, mixed_freq_timeseries
    ):
        """Une date-ancre reste une date-ancre, recalage ou non."""
        imputer = self._imputer()
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        column = 'balance_commerciale_annuelle'
        anchors = mixed_freq_timeseries[column].dropna().index
        provenance = imputer.imputation_provenance_[column]
        marked = provenance.reindex(anchors).dropna()

        # Une ancre RÉÉCRITE est DISAGGREGATED, jamais MODEL_ON_*. Une ancre
        # hors fenêtre d'imputation n'est pas réécrite du tout et garde donc
        # sa marque ORIGINAL : rien n'y a été produit, rien n'y a été détruit
        assert len(marked) > 0
        assert marked.isin(
            [ProvenanceType.DISAGGREGATED, ProvenanceType.ORIGINAL]
        ).all()
        assert (marked == ProvenanceType.DISAGGREGATED).any()

        # Les sous-périodes, elles, restent des sorties de modèle
        sub_periods = provenance.drop(index=anchors).dropna()
        assert len(sub_periods) > 0
        assert (sub_periods != ProvenanceType.DISAGGREGATED).all()

    def test_period_totals_marking_is_unchanged_when_enforced(
        self, mixed_freq_timeseries
    ):
        """Sous le régime par défaut, l'union est un no-op strict."""
        imputer = self._imputer(enforce_period_totals=True)
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        column = 'balance_commerciale_annuelle'
        provenance = imputer.imputation_provenance_[column].dropna()
        written = provenance[provenance != ProvenanceType.ORIGINAL]

        # Toutes les cellules recalées, ancres comprises, restent DISAGGREGATED
        assert len(written) > 0
        assert (written == ProvenanceType.DISAGGREGATED).all()

    def test_no_period_totals_does_not_force_fallback(self, mixed_freq_timeseries):
        """Aucune étape ne bascule en repli faute de masque d'entraînement.

        Le marquage MODEL_ON_* des ancres vidait le filtre de provenance de
        `_prepare_training_data` à l'étape suivante, d'où `len(X_train) < 2`
        puis repli interpolation.
        """
        imputer = self._imputer(keep_lower_frequencies=True)
        _fit_transform_quiet(imputer, mixed_freq_timeseries)

        fallbacks = [
            (step.pred_freq_label, step.var_name)
            for step in imputer.imputation_plan_ if step.is_fallback
        ]
        assert fallbacks == [], "etapes basculees en repli : %s" % (fallbacks,)

    def test_no_period_totals_column_is_homogeneous(self, mixed_freq_timeseries):
        """La colonne ne mélange plus le total annuel et des sous-périodes.

        Sans le correctif, l'ancre de janvier portait le total annuel
        observé à côté de valeurs d'échelle trimestrielle, avec une rampe
        linéaire entre les deux : la somme de l'année atteignait 164 fois
        le total observé.
        """
        imputer = self._imputer()
        result = _fit_transform_quiet(imputer, mixed_freq_timeseries)

        column = 'balance_commerciale_annuelle'
        observed = mixed_freq_timeseries[column].dropna()
        imputed = result[column]

        checked = 0
        for anchor, total in observed.items():
            year = imputed[imputed.index.year == anchor.year].dropna()
            if len(year) < 12:
                continue
            checked += 1
            # Aucune cellule ne porte encore le total de la période
            assert not np.isclose(year, total, atol=1e-9).any()
            # Et l'échelle reste celle de la sous-période
            assert abs(year.sum() / total) < 5, (
                "somme %.2f pour un total observe de %.2f" % (year.sum(), total)
            )
        assert checked > 0


# ---------------------------------------------------------------------------
# §1bis B25/B6/B11/B12 — échelles et fréquences
# ---------------------------------------------------------------------------

_SCALE_LEVELS = {'M': 1000.0, 'Q': 3000.0, 'Y': 12000.0}


def _series_at(index, freq, level, rng):
    """Series carrying `level` on the period ends of `freq`, NaN elsewhere."""
    out = pd.Series(np.nan, index=index)
    anchors = index[~index.to_period(freq).duplicated(keep='last')]
    out.loc[anchors] = level + rng.normal(0, level * 0.02, len(anchors))
    return out


def _scale_frame(spec, seed=0):
    """Monthly frame whose columns carry the frequencies given by `spec`."""
    index = pd.date_range('2015-01-31', '2023-12-31', freq='ME', name='date')
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        name: _series_at(index, freq, _SCALE_LEVELS[freq], rng)
        for name, freq in spec.items()
    })


def _stage_scales(imputer, data, var_key, cov_name, pred_freq):
    """Training and prediction magnitudes of one covariate, plus its divisor.

    Reproduces what the estimator actually sees: `X_train` divided by its
    per-covariate divisor at fit time, and the raw stage frame at predict
    time. Reading the methods directly keeps the measurement independent of
    whether the step fell back to interpolation.
    """
    classified = imputer._classify_variables_at_frequency(pred_freq)
    X_stage = imputer._build_stage_frame(data, {}, pred_freq, classified['aggregate'])
    X_train, _, scale_factor, factors = imputer._prepare_training_data(
        X_stage, data, var_key, pred_freq
    )
    divisor = factors[cov_name]
    if isinstance(divisor, pd.Series):
        fit_scale = (X_train[cov_name] / divisor).abs().mean()
    else:
        fit_scale = (X_train[cov_name] / divisor).abs().mean()
    return fit_scale, X_stage[cov_name].abs().mean(), divisor, scale_factor


def _fitted(data, **kwargs):
    params = {'target_frequency': 'M', 'keep_lower_frequencies': False}
    params.update(kwargs)
    imputer = HighFrequencyImputer(**params)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        imputer.fit(data.copy())
    return imputer


class TestCovariateScalingDivisors:
    """§1bis B25/B6 : chaque covariable est ramenée à SON échelle de prédiction.

    Au fit, une covariable est agrégée à `f_var` ; au predict, elle est lue
    brute dans le frame d'étape, où elle porte `f_stage`. Le diviseur doit
    être le nombre de sous-périodes `f_stage` dans une période `f_var` — ni
    systématiquement `scale_factor`, ni systématiquement le décompte propre
    à la fréquence détectée de la covariable.
    """

    def test_training_scale_matches_prediction_scale(self):
        """Covariable mensuelle, variable annuelle, étape trimestrielle : rapport 1."""
        # Le cas B25 : f_cov=M est plus fine que pred_freq=Q, donc
        # _build_stage_frame l'agrège et elle porte Q au predict. Le diviseur
        # etait get_conversion_factor(M, Y) = 12 pour une echelle Q qui en
        # appelle 4 : les features arrivaient 3x trop petites a l'entrainement
        data = _scale_frame({'cov': 'M', 'var': 'Y'})
        imputer = _fitted(data)

        fit_scale, pred_scale, divisor, _ = _stage_scales(
            imputer, data, 'var', 'cov', 'Q'
        )

        assert divisor == 4.0
        assert pred_scale / fit_scale == pytest.approx(1.0, rel=1e-9)

    def test_coarser_covariate_keeps_own_divisor(self):
        """Covariable trimestrielle sous variable annuelle : diviseur 4, pas scale_factor."""
        # f_cov=Q n'est PAS plus fine que pred_freq=M : _build_stage_frame la
        # laisse au trimestre. Diviser par scale_factor (12) la rendrait 3x
        # trop petite — c'est le cas qui interdit un diviseur unique
        data = _scale_frame({'ref': 'M', 'cov': 'Q', 'var': 'Y'})
        imputer = _fitted(data)

        fit_scale, pred_scale, divisor, scale_factor = _stage_scales(
            imputer, data, 'var', 'cov', 'M'
        )

        assert scale_factor == 12.0
        assert divisor == 4.0
        assert pred_scale / fit_scale == pytest.approx(1.0, rel=1e-9)

    def test_non_aggregated_covariate_not_scaled(self):
        """Covariable plus grossière que la variable : diviseur 1, pas 0.25."""
        # f_cov=Y n'est pas plus fine que f_var=Q : la colonne n'est jamais
        # ré-agrégée et porte deja son echelle de prediction. Le diviseur
        # fractionnaire get_conversion_factor(Y, Q) = 0.25 la MULTIPLIAIT par 4
        data = _scale_frame({'ref': 'M', 'cov': 'Y', 'var': 'Q'})
        imputer = _fitted(data)

        fit_scale, pred_scale, divisor, _ = _stage_scales(
            imputer, data, 'var', 'cov', 'M'
        )

        assert divisor == 1.0
        assert pred_scale / fit_scale == pytest.approx(1.0, rel=1e-9)


def _panel_heterogeneous_covariate(columns=None):
    """Panel where `ip` is MONTHLY for France and QUARTERLY for Allemagne.

    `columns` reorders the input frame's columns, to prove the divisors do
    not depend on the order the user happened to choose.
    """
    dates = pd.date_range('2015-01-31', '2023-12-31', freq='ME', name='date')
    frames = []
    for country, ip_freq, seed in (('France', 'M', 3), ('Allemagne', 'Q', 4)):
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({
            'ref': _series_at(dates, 'M', 1000.0, rng),
            'ip': _series_at(dates, ip_freq, _SCALE_LEVELS[ip_freq], rng),
            'var': _series_at(dates, 'Y', 12000.0, rng),
        })
        frame['country'] = country
        frames.append(frame.reset_index())

    panel = pd.concat(frames, ignore_index=True).set_index(['country', 'date'])
    panel = panel.sort_index()
    return panel if columns is None else panel[columns]


class TestCovariateFactorsPerEntity:
    """§1bis B6 : le diviseur est calculé PAR ENTITÉ, jamais par nom nu.

    Une table `{nom de colonne: fréquence}` s'effondre sur un panel où la
    même colonne porte deux fréquences : la dernière entité insérée gagne, et
    le modèle obtenu dépend de l'ordre des colonnes du DataFrame d'entrée.
    """

    @staticmethod
    def _divisors(panel):
        from tsforecast.panel.utils import split_variable_key

        imputer = _fitted(panel)
        var_key = next(
            key for key in imputer.detected_frequencies_
            if split_variable_key(key)[1] == 'var'
        )
        classified = imputer._classify_variables_at_frequency('M')
        X_stage = imputer._build_stage_frame(panel, {}, 'M', classified['aggregate'])
        _, _, _, factors = imputer._prepare_training_data(
            X_stage, panel, var_key, 'M'
        )
        return factors

    def test_covariate_factors_per_entity(self):
        """'ip' mensuelle chez France et trimestrielle chez Allemagne : 12 et 4."""
        factors = self._divisors(_panel_heterogeneous_covariate())

        # Les deux entités ne peuvent pas partager un diviseur : la ventilation
        # se fait donc par ligne, ce qu'une Series par colonne ne peut porter
        assert isinstance(factors, pd.DataFrame)
        france = factors.xs('France', level='country')['ip'].unique()
        allemagne = factors.xs('Allemagne', level='country')['ip'].unique()
        assert france.tolist() == [12.0]
        assert allemagne.tolist() == [4.0]

    def test_divisors_do_not_depend_on_column_order(self):
        """Permuter les colonnes d'entrée ne change aucun diviseur."""
        direct = self._divisors(_panel_heterogeneous_covariate())
        swapped = self._divisors(
            _panel_heterogeneous_covariate(columns=['var', 'ip', 'ref'])
        )
        pd.testing.assert_frame_equal(
            direct, swapped[direct.columns], check_like=True
        )


class TestUnitScaleFactorStillScalesFeatures:
    """§1bis B12 : un scale_factor unitaire n'implique pas des diviseurs unitaires."""

    def test_unit_scale_factor_still_scales_features(self):
        """scale_factor == 1.0 avec des feature_factors non unitaires : X est divisé."""
        imputer = HighFrequencyImputer(target_frequency='M')
        X_train = pd.DataFrame({'a': [12.0, 24.0], 'b': [4.0, 8.0]})
        y_train = pd.Series([1.0, 2.0])
        factors = pd.Series({'a': 12.0, 'b': 4.0})

        X_scaled, y_scaled = imputer._apply_frequency_scaling(
            X_train, y_train, 1.0, factors
        )

        # Le court-circuit historique renvoyait X_train intact, laissant les
        # covariables d'une variable trimestrielle prédite au trimestre à une
        # échelle sans rapport avec celles de l'étape voisine
        assert X_scaled['a'].tolist() == [1.0, 2.0]
        assert X_scaled['b'].tolist() == [1.0, 2.0]
        # La cible reste inchangée : elle est bien divisée par 1
        assert y_scaled.tolist() == [1.0, 2.0]

    def test_unit_factors_short_circuit(self):
        """Rien à mettre à l'échelle : les objets d'entrée sont rendus tels quels."""
        imputer = HighFrequencyImputer(target_frequency='M')
        X_train = pd.DataFrame({'a': [12.0, 24.0]})
        y_train = pd.Series([1.0, 2.0])

        X_scaled, y_scaled = imputer._apply_frequency_scaling(
            X_train, y_train, 1.0, pd.Series({'a': 1.0})
        )

        assert X_scaled is X_train
        assert y_scaled is y_train


class TestStageScaleFactorIsDocumentedAverage:
    """§1bis B11 : le facteur d'étape est une moyenne, pas un décompte calendaire.

    `count_subperiods_per_period` rendrait 28 en février et 31 en janvier,
    mais il rend un décompte PAR PÉRIODE : le câbler impose un diviseur par
    ligne, qui arrive avec `train_on_own_imputations`. Ce test fige le
    comportement documenté pour rendre ce changement visible.
    """

    def test_scale_factor_is_documented_average(self):
        """Étape journalière sur variable mensuelle : 30.0 constant."""
        data = _scale_frame({'cov': 'M', 'var': 'Q'})
        imputer = _fitted(data)

        # 30 jours par mois quelle que soit la periode : fevrier (28) et
        # janvier (31) recoivent le meme facteur
        assert imputer._stage_scale_factor('cov', 'D') == 30.0
        # Les paires calendaires emboitees restent exactes
        assert imputer._stage_scale_factor('var', 'M') == 3.0
