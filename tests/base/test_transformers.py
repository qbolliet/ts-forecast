"""Tests for base_transformers module - MultiIndex support.

This module tests the PanelTimeSeriesTransformer with both column-based
and MultiIndex-based panel data structures.
"""
import pandas as pd
import pytest
import numpy as np
from tsforecast.base.transformers import PanelTimeSeriesTransformer


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
