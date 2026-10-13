"""``HighFrequencyImputer.inverse_transform`` (§12.4, I8).

Covers the transform / inverse round trip, multi-level entity names (B4)
and the provenance-driven reset of imputed cells.
"""

# Modules de base
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

# Objets testés
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer

# Outils de test partagés
from tests.unit.frequency.high_frequency_imputer.helpers import (
    fit_quietly,
    fit_transform_quietly,
    make_imputer,
)


class TestInverseTransform:
    """I8 — aller-retour transform / inverse_transform."""

    def test_inverse_transform_roundtrip(self, reference_timeseries):
        """I8 — l'index et les noms sont restitués, les valeurs sous demande."""
        imputer = make_imputer(restore_original_values=True)
        transformed = fit_transform_quietly(imputer, reference_timeseries)
        restored = imputer.inverse_transform(transformed)

        # Index et noms de niveaux d'origine
        assert restored.index.equals(reference_timeseries.index)
        assert restored.index.names == reference_timeseries.index.names

        # Valeurs d'origine, cellule par cellule
        for column in reference_timeseries.columns:
            observed = reference_timeseries[column].notna()
            assert np.allclose(
                restored.loc[observed, column].to_numpy(),
                reference_timeseries.loc[observed, column].to_numpy(),
            )

    def test_inverse_transform_preserves_multilevel_entity_names(
        self, panel_two_level_dataset
    ):
        """I8, B4 — un panel à deux niveaux d'entité garde ses trois noms."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            restore_original_values=True,
        )
        transformed = fit_transform_quietly(imputer, panel_two_level_dataset)

        # Le niveau de fréquence s'insère du côté de l'entité (§5.9)
        assert list(transformed.index.names) == [
            'country', 'sector', 'frequency', 'date'
        ]

        restored = imputer.inverse_transform(transformed)
        assert list(restored.index.names) == ['country', 'sector', 'date']
        # Toute ligne d'entrée est restituée, avec ses valeurs d'origine. La
        # grille cible peut en porter d'autres — l'ancrage de la conversion de
        # fenêtre ajoute ici une fin de période au-delà des données, ce que
        # produit déjà le fit et qui ne relève pas de l'inversion
        assert panel_two_level_dataset.index.isin(restored.index).all()
        source = panel_two_level_dataset['indicateur_mensuel']
        assert np.allclose(
            restored['indicateur_mensuel'].reindex(source.index).to_numpy(),
            source.to_numpy(),
        )

    def test_inverse_transform_without_a_provenance_matrix_raises(
        self, reference_timeseries
    ):
        """Sans matrice de provenance, l'inversion refuse : c'est elle qui guide.

        Le fit en écrit déjà une — inverser juste après lui est donc licite,
        et lit alors la provenance du fit. La garde couvre l'état purgé.
        """
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        del imputer.__dict__['imputation_provenance_']
        with pytest.raises(ValueError, match='previous call to transform'):
            imputer.inverse_transform(reference_timeseries)

    def test_inverse_right_after_a_fit_reads_the_fit_provenance(
        self, reference_timeseries
    ):
        """Le fit écrit sa propre matrice : l'inversion la lit sans transform."""
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        restored = imputer.inverse_transform(reference_timeseries)
        assert restored.index.equals(reference_timeseries.index)

    def test_restoring_inverse_right_after_a_fit(self, reference_timeseries):
        """``restore_original_values=True`` inverts right after a fit, like after a transform.

        Inverting right after a fit is licit (the fit writes its own provenance
        matrix and snapshots its own input, ANO-FREQ-051); the exact restoration
        then refills the cells observed in the input of the fit.
        """
        imputer = fit_quietly(
            make_imputer(restore_original_values=True), reference_timeseries
        )
        restored = imputer.inverse_transform(reference_timeseries)

        assert restored.index.equals(reference_timeseries.index)
        for column in reference_timeseries.columns:
            observed = reference_timeseries[column].notna()
            assert np.allclose(
                restored.loc[observed, column].to_numpy(),
                reference_timeseries.loc[observed, column].to_numpy(),
            )

    def test_imputed_cells_are_set_back_to_nan(self, reference_timeseries):
        """Sans restore_original_values, toute cellule non ORIGINAL redevient NaN."""
        imputer = make_imputer()
        transformed = fit_transform_quietly(imputer, reference_timeseries)
        restored = imputer.inverse_transform(transformed)
        # m1 est observée partout : elle traverse intacte
        assert restored['m1'].notna().all()
        # a1 était imputée : elle ne peut pas être plus renseignée qu'à l'entrée
        assert restored['a1'].notna().sum() <= reference_timeseries['a1'].notna().sum()
