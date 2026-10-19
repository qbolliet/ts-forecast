"""Provenance families emitted by ``HighFrequencyImputer``.

Covers the five ``MODEL_*`` families and the absence of ``DISAGGREGATED``.
"""

# Modules de base
import pytest

# Objets testés
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer

# Outils de test partagés
from tests.unit.frequency.high_frequency_imputer.helpers import (
    fit_with_spy,
)


class TestProvenanceFamilies:
    """Les cinq familles MODEL_* sont émises exactement dans les cas documentés."""

    @staticmethod
    def _families(imputer: HighFrequencyImputer, column: str) -> set:
        """Provenances distinctes portées par une colonne."""
        return {str(value) for value in imputer.imputation_provenance_[column]}

    def test_interpolate_emits_model_on_interpolated(self, reference_timeseries):
        """Une covariable plus basse que la grille suffit à souiller l'étape."""
        imputer = fit_with_spy(reference_timeseries)
        for column in ('q1', 'a1', 'a2'):
            assert self._families(imputer, column) == {'model_on_interpolated'}

    def test_tolerate_nan_emits_model_on_true(self, reference_timeseries):
        """Aucune valeur interpolée ne circule : le modèle n'a vu que du vrai."""
        imputer = fit_with_spy(reference_timeseries, covariate_strategy='tolerate_nan')
        for column in ('q1', 'a1', 'a2'):
            assert self._families(imputer, column) == {'model_on_true'}

    def test_model_on_imputed_only_under_model_strategy(self, reference_timeseries):
        """MODEL_ON_IMPUTED n'est émis que sous covariate_strategy='model'."""
        under_model = fit_with_spy(reference_timeseries, covariate_strategy='model')
        # 'a1' est imputée la première : ses covariables ne sont qu'interpolées
        assert self._families(under_model, 'a1') == {'model_on_interpolated'}
        # 'a2' et 'q1' lisent ensuite l'imputation de 'a1' dans le miroir
        assert self._families(under_model, 'a2') == {'model_on_imputed'}
        assert self._families(under_model, 'q1') == {'model_on_imputed'}

        for strategy in ('interpolate', 'tolerate_nan'):
            imputer = fit_with_spy(reference_timeseries, covariate_strategy=strategy)
            emitted = set().union(
                *(self._families(imputer, column) for column in ('q1', 'a1', 'a2'))
            )
            assert 'model_on_imputed' not in emitted

    @pytest.mark.parametrize('strategy', ['tolerate_nan', 'interpolate', 'model'])
    def test_target_families_absent_under_false(self, reference_timeseries, strategy):
        """*_TARGET et *_BOTH exigent impute_intermediate_frequencies=True."""
        imputer = fit_with_spy(reference_timeseries, covariate_strategy=strategy)
        emitted = set().union(
            *(self._families(imputer, column) for column in ('q1', 'a1', 'a2'))
        )
        assert not emitted & {'model_on_imputed_target', 'model_on_imputed_both'}

    def test_no_disaggregated_cell_is_ever_emitted(self, reference_timeseries):
        """DISAGGREGATED n'est jamais émis par l'imputeur, recalage compris."""
        imputer = fit_with_spy(reference_timeseries)
        emitted = {
            str(value) for value in imputer.imputation_provenance_.to_numpy().ravel()
        }
        assert 'disaggregated' not in emitted
