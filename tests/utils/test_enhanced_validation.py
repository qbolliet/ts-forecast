"""Tests for enhanced validation in base_transformers."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

from tsforecast.utils.base_transformers import PanelTimeSeriesTransformer


# Classe de test concrète (PanelTimeSeriesTransformer est abstraite)
class ConcreteTransformer(PanelTimeSeriesTransformer):
    """Implémentation concrète pour les tests."""

    def _fit(self, X: pd.DataFrame, y=None):
        """Fit factice."""
        pass

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform factice."""
        return X


@pytest.fixture
def sorted_time_series():
    """Série temporelle triée."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    data = {
        'value': np.random.randn(len(dates))
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'date'
    return df


@pytest.fixture
def unsorted_time_series():
    """Série temporelle non triée."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    data = {
        'date': dates,
        'value': np.random.randn(len(dates))
    }
    df = pd.DataFrame(data)
    # Mélanger les lignes
    df = df.sample(frac=1).reset_index(drop=True)
    return df


@pytest.fixture
def sorted_panel_data():
    """Données panel triées et groupées."""
    countries = ['France', 'Germany', 'Italy']
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='MS')

    data = []
    for country in countries:
        for date in dates:
            data.append({
                'country': country,
                'date': date,
                'value': np.random.randn()
            })

    return pd.DataFrame(data)


@pytest.fixture
def unsorted_panel_data():
    """Données panel non triées."""
    countries = ['France', 'Germany', 'Italy']
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='MS')

    data = []
    for country in countries:
        for date in dates:
            data.append({
                'country': country,
                'date': date,
                'value': np.random.randn()
            })

    df = pd.DataFrame(data)
    # Mélanger les lignes
    df = df.sample(frac=1).reset_index(drop=True)
    return df


@pytest.fixture
def interleaved_panel_data():
    """Données panel avec entités entrelacées (non contiguës)."""
    data = [
        {'country': 'France', 'date': pd.Timestamp('2023-01-01'), 'value': 1.0},
        {'country': 'Germany', 'date': pd.Timestamp('2023-01-01'), 'value': 2.0},
        {'country': 'France', 'date': pd.Timestamp('2023-02-01'), 'value': 3.0},  # France entrelacé
        {'country': 'Germany', 'date': pd.Timestamp('2023-02-01'), 'value': 4.0},
    ]
    return pd.DataFrame(data)


class TestStrictValidationMode:
    """Tests pour le mode de validation strict."""

    def test_strict_validation_sorted_time_series_passes(self, sorted_time_series):
        """Test données triées passent en mode strict."""
        transformer = ConcreteTransformer(
            time_col='date',
            strict_validation=True,
            auto_sort=False
        )

        # Ne devrait pas lever d'erreur
        transformer.fit(sorted_time_series)

    def test_strict_validation_unsorted_raises(self, unsorted_time_series):
        """Test données non triées lèvent erreur en mode strict."""
        transformer = ConcreteTransformer(
            time_col='date',
            panel_cols=None,
            strict_validation=True,
            auto_sort=False
        )

        with pytest.raises(ValueError, match="not sorted"):
            transformer.fit(unsorted_time_series)

    def test_permissive_validation_unsorted_warns(self, unsorted_time_series):
        """Test données non triées émettent warning en mode permissif."""
        transformer = ConcreteTransformer(
            time_col='date',
            panel_cols=None,
            strict_validation=False,
            auto_sort=False
        )

        with pytest.warns(UserWarning):
            transformer.fit(unsorted_time_series)


class TestAutoSortFeature:
    """Tests pour la fonctionnalité auto_sort."""

    def test_auto_sort_fixes_unsorted_data(self, unsorted_time_series):
        """Test auto_sort corrige le tri automatiquement."""
        transformer = ConcreteTransformer(
            time_col='date',
            panel_cols=None,
            strict_validation=True,
            auto_sort=True
        )

        # Ne devrait pas lever d'erreur grâce à auto_sort
        transformer.fit(unsorted_time_series)

    def test_auto_sort_preserves_sorted_data(self, sorted_time_series):
        """Test auto_sort ne modifie pas les données déjà triées."""
        transformer = ConcreteTransformer(
            time_col='date',
            strict_validation=True,
            auto_sort=True
        )

        original_index = sorted_time_series.index.copy()
        transformer.fit(sorted_time_series)

        # L'index devrait être inchangé
        assert sorted_time_series.index.equals(original_index)


class TestPanelDataValidation:
    """Tests pour la validation des données panel."""

    def test_sorted_panel_data_passes(self, sorted_panel_data):
        """Test données panel triées et groupées passent."""
        transformer = ConcreteTransformer(
            time_col='date',
            panel_cols=['country'],
            strict_validation=True,
            auto_sort=False
        )

        # Ne devrait pas lever d'erreur
        transformer.fit(sorted_panel_data)

    def test_interleaved_entities_raises(self, interleaved_panel_data):
        """Test entités entrelacées lèvent erreur."""
        transformer = ConcreteTransformer(
            time_col='date',
            panel_cols=['country'],
            strict_validation=True,
            auto_sort=False
        )

        with pytest.raises(ValueError, match="not contiguous"):
            transformer.fit(interleaved_panel_data)

    def test_unsorted_panel_within_groups_raises(self):
        """Test données non triées dans les groupes lèvent erreur."""
        # Panel avec dates non triées au sein d'un groupe
        data = [
            {'country': 'France', 'date': pd.Timestamp('2023-02-01'), 'value': 1.0},
            {'country': 'France', 'date': pd.Timestamp('2023-01-01'), 'value': 2.0},  # Non trié
            {'country': 'Germany', 'date': pd.Timestamp('2023-01-01'), 'value': 3.0},
            {'country': 'Germany', 'date': pd.Timestamp('2023-02-01'), 'value': 4.0},
        ]
        df = pd.DataFrame(data)

        transformer = ConcreteTransformer(
            time_col='date',
            panel_cols=['country'],
            strict_validation=True,
            auto_sort=False
        )

        with pytest.raises(ValueError, match="not sorted within panel groups"):
            transformer.fit(df)

    def test_auto_sort_with_panel_data(self, unsorted_panel_data):
        """Test auto_sort fonctionne avec données panel."""
        transformer = ConcreteTransformer(
            time_col='date',
            panel_cols=['country'],
            strict_validation=True,
            auto_sort=True
        )

        # Ne devrait pas lever d'erreur grâce à auto_sort
        transformer.fit(unsorted_panel_data)


class TestMissingPanelColumns:
    """Tests pour la détection de colonnes panel manquantes."""

    def test_missing_panel_cols_raises(self, sorted_panel_data):
        """Test erreur si colonnes panel spécifiées mais absentes."""
        transformer = ConcreteTransformer(
            time_col='date',
            panel_cols=['missing_column'],
            strict_validation=True
        )

        with pytest.raises(ValueError, match="Panel columns not found"):
            transformer.fit(sorted_panel_data)


class TestValidationPreservesData:
    """Tests pour vérifier que la validation préserve les données."""

    def test_validation_preserves_data_when_valid(self, sorted_time_series):
        """Test validation ne modifie pas les données valides."""
        transformer = ConcreteTransformer(
            time_col='date',
            strict_validation=True,
            auto_sort=False
        )

        original_values = sorted_time_series['value'].copy()
        transformer.fit(sorted_time_series)

        # Les valeurs devraient être inchangées
        assert (sorted_time_series['value'] == original_values).all()

    def test_auto_sort_only_sorts_not_modifies_values(self, unsorted_time_series):
        """Test auto_sort trie mais ne modifie pas les valeurs."""
        transformer = ConcreteTransformer(
            time_col='date',
            panel_cols=None,
            strict_validation=True,
            auto_sort=True
        )

        original_values_set = set(unsorted_time_series['value'].values)
        transformer.fit(unsorted_time_series)

        # Les valeurs devraient être les mêmes (juste réordonnées)
        # Note: auto_sort est appliqué sur une copie, donc original inchangé
        assert set(unsorted_time_series['value'].values) == original_values_set


class TestParameterDefaults:
    """Tests pour les valeurs par défaut des paramètres."""

    def test_default_strict_validation_is_true(self):
        """Test strict_validation=True par défaut."""
        transformer = ConcreteTransformer()
        assert transformer.strict_validation is True

    def test_default_auto_sort_is_false(self):
        """Test auto_sort=False par défaut."""
        transformer = ConcreteTransformer()
        assert transformer.auto_sort is False

    def test_default_validate_input_is_true(self):
        """Test validate_input=True par défaut."""
        transformer = ConcreteTransformer()
        assert transformer.validate_input is True


class TestValidationCanBeDisabled:
    """Tests pour désactiver la validation."""

    def test_validation_can_be_disabled(self, unsorted_time_series):
        """Test validation peut être complètement désactivée."""
        transformer = ConcreteTransformer(
            time_col='date',
            validate_input=False  # Désactive toute validation
        )

        # Ne devrait pas lever d'erreur même avec données non triées
        transformer.fit(unsorted_time_series)
