"""Tests for enhanced validation in base.transformers module - MultiIndex support.

This module tests the PanelTimeSeriesTransformer with both column-based
and MultiIndex-based panel data structures.
"""
import pandas as pd
import pytest
import numpy as np
from datetime import datetime
import warnings
from tsforecast.base.transformers import PanelTimeSeriesTransformer

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


# Créer un transformateur concret pour les tests
class SimpleScaler(PanelTimeSeriesTransformer):
    """Simple transformer for testing purposes."""

    def _fit(self, X, y=None):
        # Apprentissage des moyennes pour normalisation
        numeric_cols = X.select_dtypes(include='number').columns
        self.means_ = X[numeric_cols].mean()

    def _transform(self, X):
        # Application de la transformation
        X_transformed = X.copy()
        numeric_cols = X.select_dtypes(include='number').columns
        X_transformed[numeric_cols] = X[numeric_cols] - self.means_
        return X_transformed


# ==================== GROUPE 1: BACKWARD COMPATIBILITY ====================


def test_column_based_panel_backward_compatibility():
    """Vérifier que l'approche colonnes fonctionne toujours."""
    df = pd.DataFrame({
        'entity': ['A', 'A', 'B', 'B'],
        'date': pd.date_range('2023-01-01', periods=2).tolist() * 2,
        'value': [1, 2, 3, 4]
    })

    transformer = SimpleScaler(time_col='date', panel_cols=['entity'])
    result = transformer.fit_transform(df)

    assert isinstance(result, pd.DataFrame)
    assert transformer.is_panel_ is True
    assert len(result) == 4
    # Vérifier que les colonnes d'origine sont préservées ou dans l'index
    assert 'value' in result.columns


def test_column_based_time_series_backward_compatibility():
    """Vérifier que les séries temporelles colonnes fonctionnent."""
    dates = pd.date_range('2023-01-01', periods=5)
    df = pd.DataFrame({'date': dates, 'value': [1, 2, 3, 4, 5]})

    transformer = SimpleScaler(time_col='date')
    result = transformer.fit_transform(df)

    assert isinstance(result, pd.DataFrame)
    assert transformer.is_panel_ is False
    assert len(result) == 5


# ==================== GROUPE 2: MULTIINDEX PANEL ====================


def test_multiindex_panel_basic():
    """Test basique MultiIndex 2 niveaux."""
    idx = pd.MultiIndex.from_arrays([
        ['A', 'A', 'B', 'B'],
        pd.date_range('2023-01-01', periods=2).tolist() * 2
    ], names=['entity', 'date'])
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=idx)

    transformer = SimpleScaler(time_col=None, panel_cols=None)
    result = transformer.fit_transform(df)

    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.nlevels == 2
    assert transformer.is_panel_ is True
    assert transformer._panel_in_index is True
    assert transformer._time_in_index is True


def test_multiindex_panel_multi_entity_levels():
    """Test MultiIndex 3+ niveaux (ex: country + sector + date)."""
    idx = pd.MultiIndex.from_arrays([
        ['US', 'US', 'FR', 'FR'],
        ['Tech', 'Finance', 'Tech', 'Finance'],
        pd.date_range('2023-01-01', periods=1).tolist() * 4
    ], names=['country', 'sector', 'date'])
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=idx)

    transformer = SimpleScaler(time_col=None, panel_cols=None)
    result = transformer.fit_transform(df)

    assert result.index.nlevels == 3
    assert transformer.is_panel_ is True
    assert transformer._panel_level_names == ['country', 'sector']


def test_multiindex_panel_named_levels():
    """Vérifier que les noms de niveaux sont préservés."""
    idx = pd.MultiIndex.from_arrays([
        ['A', 'A', 'B', 'B'],
        pd.date_range('2023-01-01', periods=2).tolist() * 2
    ], names=['entity_id', 'timestamp'])
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=idx)

    transformer = SimpleScaler(time_col=None, panel_cols=None)
    result = transformer.fit_transform(df)

    assert result.index.names == ['entity_id', 'timestamp']
    assert transformer._panel_level_names == ['entity_id']


# ==================== GROUPE 3: DATETIMEINDEX TIME SERIES ====================


def test_datetimeindex_time_series():
    """Test série temporelle avec DatetimeIndex."""
    dates = pd.date_range('2023-01-01', periods=5)
    df = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=dates)

    transformer = SimpleScaler(time_col=None, panel_cols=None)
    result = transformer.fit_transform(df)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert not isinstance(result.index, pd.MultiIndex)
    assert transformer.is_panel_ is False
    assert transformer._time_in_index is True


# ==================== GROUPE 4: CAS LIMITES ====================


def test_invalid_config_panel_cols_without_time_col():
    """Erreur si panel_cols spécifié mais time_col=None."""
    df = pd.DataFrame({'entity': ['A'], 'value': [1]})

    transformer = SimpleScaler(time_col=None, panel_cols=['entity'])

    with pytest.raises(ValueError, match="Invalid configuration"):
        transformer.fit(df)


def test_single_level_multiindex_error():
    """Erreur si MultiIndex avec seulement 1 niveau."""
    idx = pd.MultiIndex.from_arrays([
        pd.date_range('2023-01-01', periods=4)
    ])
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=idx)

    transformer = SimpleScaler(time_col=None, panel_cols=None)

    with pytest.raises(ValueError, match="at least 2 levels"):
        transformer.fit(df)


def test_multiindex_unsorted_with_auto_sort():
    """MultiIndex non trié avec auto_sort=True."""
    # Créer des données avec entités non contiguës (mais sans doublons)
    dates_a = pd.date_range('2023-01-01', periods=2)
    dates_b = pd.date_range('2023-01-03', periods=2)

    idx = pd.MultiIndex.from_arrays([
        ['B', 'B', 'A', 'A'],  # Unsorted entities (B avant A)
        list(dates_b) + list(dates_a)
    ], names=['entity', 'date'])
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=idx)

    transformer = SimpleScaler(
        time_col=None,
        panel_cols=None,
        auto_sort=True
    )
    result = transformer.fit_transform(df)

    # Les entités devraient être triées (A avant B)
    entities = result.index.get_level_values(0).tolist()
    assert entities == ['A', 'A', 'B', 'B']


def test_multiindex_missing_values_in_entity_level():
    """Valeurs manquantes dans niveaux entité."""
    idx = pd.MultiIndex.from_arrays([
        ['A', pd.NA, 'B', 'B'],  # Missing entity
        pd.date_range('2023-01-01', periods=2).tolist() * 2
    ], names=['entity', 'date'])
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=idx)

    transformer = SimpleScaler(
        time_col=None,
        panel_cols=None,
        strict_validation=False  # Use warnings instead of errors
    )

    # Devrait émettre un warning
    with pytest.warns(UserWarning, match="Missing values in panel identifier"):
        transformer.fit(df)


# ==================== GROUPE 5: INTÉGRATION ====================


def test_multiindex_with_convert_cols_to_index():
    """convert_cols_to_index devrait être no-op pour MultiIndex."""
    idx = pd.MultiIndex.from_arrays([
        ['A', 'A', 'B', 'B'],
        pd.date_range('2023-01-01', periods=2).tolist() * 2
    ], names=['entity', 'date'])
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=idx)

    transformer = SimpleScaler(
        time_col=None,
        panel_cols=None,
        convert_cols_to_index=True  # Should be no-op
    )
    result = transformer.fit_transform(df)

    # Structure MultiIndex devrait être préservée
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.equals(df.index)


def test_structure_detection_all_scenarios():
    """Test _detect_data_structure pour tous les cas."""
    # Cas 1: Panel colonnes
    df1 = pd.DataFrame({
        'entity': ['A', 'A'],
        'date': pd.date_range('2023-01-01', periods=2),
        'value': [1, 2]
    })
    t1 = SimpleScaler(time_col='date', panel_cols=['entity'])
    is_panel, panel_in_idx, time_in_idx = t1._detect_data_structure(df1)
    assert (is_panel, panel_in_idx, time_in_idx) == (True, False, False)

    # Cas 2: Série temporelle colonnes
    df2 = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=2),
        'value': [1, 2]
    })
    t2 = SimpleScaler(time_col='date', panel_cols=None)
    is_panel, panel_in_idx, time_in_idx = t2._detect_data_structure(df2)
    assert (is_panel, panel_in_idx, time_in_idx) == (False, False, False)

    # Cas 3: Panel MultiIndex
    idx3 = pd.MultiIndex.from_arrays([
        ['A', 'A'],
        pd.date_range('2023-01-01', periods=2)
    ])
    df3 = pd.DataFrame({'value': [1, 2]}, index=idx3)
    t3 = SimpleScaler(time_col=None, panel_cols=None)
    is_panel, panel_in_idx, time_in_idx = t3._detect_data_structure(df3)
    assert (is_panel, panel_in_idx, time_in_idx) == (True, True, True)

    # Cas 4: Série temporelle DatetimeIndex
    df4 = pd.DataFrame(
        {'value': [1, 2]},
        index=pd.date_range('2023-01-01', periods=2)
    )
    t4 = SimpleScaler(time_col=None, panel_cols=None)
    is_panel, panel_in_idx, time_in_idx = t4._detect_data_structure(df4)
    assert (is_panel, panel_in_idx, time_in_idx) == (False, False, True)


def test_check_panel_consistency_multiindex():
    """Test _check_panel_consistency avec MultiIndex."""
    idx = pd.MultiIndex.from_arrays([
        ['A', 'A', 'B', 'B'],
        pd.date_range('2023-01-01', periods=2).tolist() * 2
    ], names=['entity', 'date'])
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=idx)

    transformer = SimpleScaler(time_col=None, panel_cols=None)
    transformer.fit(df)  # Setup structure detection

    is_consistent, issues = transformer._check_panel_consistency(df)

    assert is_consistent is True
    assert len(issues) == 0


def test_multiindex_validation_methods():
    """Test que les méthodes de validation MultiIndex fonctionnent."""
    idx = pd.MultiIndex.from_arrays([
        ['A', 'A', 'B', 'B'],
        pd.date_range('2023-01-01', periods=2).tolist() * 2
    ], names=['entity', 'date'])
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=idx)

    transformer = SimpleScaler(time_col=None, panel_cols=None)

    # Test _validate_multiindex_panel directement
    validated = transformer._validate_multiindex_panel(df)

    assert isinstance(validated.index, pd.MultiIndex)
    assert validated.index.nlevels == 2


# ==================== TESTS ADDITIONNELS ====================


def test_multiindex_panel_fit_metadata():
    """Vérifier que fit() stocke correctement les métadonnées MultiIndex."""
    idx = pd.MultiIndex.from_arrays([
        ['A', 'A', 'B', 'B'],
        pd.date_range('2023-01-01', periods=2).tolist() * 2
    ], names=['entity', 'date'])
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=idx)

    transformer = SimpleScaler(time_col=None, panel_cols=None)
    transformer.fit(df)

    # Vérifications métadonnées
    assert hasattr(transformer, 'is_panel_')
    assert transformer.is_panel_ is True
    assert hasattr(transformer, '_panel_in_index')
    assert transformer._panel_in_index is True
    assert hasattr(transformer, '_time_in_index')
    assert transformer._time_in_index is True
    assert hasattr(transformer, '_panel_level_names')
    assert transformer._panel_level_names == ['entity']


def test_multiindex_with_unnamed_levels():
    """Test MultiIndex avec niveaux sans nom."""
    idx = pd.MultiIndex.from_arrays([
        ['A', 'A', 'B', 'B'],
        pd.date_range('2023-01-01', periods=2).tolist() * 2
    ])  # Pas de names
    df = pd.DataFrame({'value': [1, 2, 3, 4]}, index=idx)

    transformer = SimpleScaler(time_col=None, panel_cols=None)
    result = transformer.fit_transform(df)

    assert isinstance(result.index, pd.MultiIndex)
    assert transformer.is_panel_ is True

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
