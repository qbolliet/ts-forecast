"""Multi-frequency output of ``HighFrequencyImputer.transform``.

Scenarios: frequency control against the fit, ``keep_lower_frequencies`` and a target
coarser than the index.
"""

# Modules de base
import copy
import warnings

import numpy as np
import pandas as pd
import pytest

# Objets testés
from tsforecast.imputation._components.provenance import ProvenanceType

# Outils de test partagés
from tests.unit.imputation.high_frequency_imputer.helpers import (
    assert_same_state,
    fit_quietly,
    fit_transform_quietly,
    make_imputer,
    transform_quietly,
    _warning_messages,
)


class TestTransformFrequencyControl:
    """Frequencies detected at transform are checked against those of the fit."""

    def test_transform_diverging_frequency_warns_once_and_uses_fit_frequencies(
        self, reference_timeseries
    ):
        """Divergence : UN avertissement, puis les fréquences du fit."""
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        fit_frequencies = dict(imputer.detected_frequencies_)

        # q1 passe de trimestrielle à annuelle sur les données du transform
        diverging = reference_timeseries.copy()
        annual = diverging.index.month == 12
        diverging['q1'] = np.nan
        diverging.loc[annual, 'q1'] = [30.0, 90.0, 150.0]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            imputer.transform(diverging)

        divergence_warnings = [
            str(w.message) for w in caught
            if 'different frequency at transform time' in str(w.message)
        ]
        assert len(divergence_warnings) == 1
        assert 'q1' in divergence_warnings[0]
        assert 'fit=' in divergence_warnings[0] and 'transform=' in divergence_warnings[0]

        # Poursuite avec les fréquences du fit : elles ne sont jamais réécrites
        assert imputer.detected_frequencies_ == fit_frequencies

    def test_transform_missing_column_raises_naming_columns(
        self, reference_timeseries
    ):
        """Colonne du fit absente : ValueError nommant les colonnes."""
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        with pytest.raises(ValueError) as excinfo:
            imputer.transform(reference_timeseries.drop(columns=['q1', 'a2']))
        message = str(excinfo.value)
        assert "'q1'" in message and "'a2'" in message

    def test_transform_extra_column_ignored_silently(self, reference_timeseries):
        """Colonne supplémentaire : ignorée, sans avertissement ni erreur."""
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        extended = reference_timeseries.copy()
        extended['extra'] = np.arange(len(extended), dtype=float)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = imputer.transform(extended)

        assert not [w for w in caught if 'extra' in str(w.message)]
        # La colonne traverse le transform sans entrer dans aucun plan
        assert 'extra' in result.columns
        reference = transform_quietly(imputer, reference_timeseries)
        pd.testing.assert_frame_equal(
            result[reference.columns], reference
        )

    def test_per_entity_frequency_divergence_is_not_a_false_positive(
        self, mixed_freq_panel_multifrequency
    ):
        """La comparaison est par COUPLE : PANEL-F n'émet aucune divergence."""
        imputer = fit_quietly(make_imputer(), mixed_freq_panel_multifrequency)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            imputer.transform(mixed_freq_panel_multifrequency)
        assert not [
            w for w in caught
            if 'different frequency at transform time' in str(w.message)
        ]

    def test_fit_frequencies_drive_the_rescaling(self, reference_timeseries):
        """A divergence is replayed with the frequency of the fit, not the detected one.

        ``q1``, quarterly at fit time, is published in December only at
        transform time (30 / 90 / 150), which reads as annual. Kept at its fit
        frequency, the December value is the total of the fourth QUARTER:
        October-December sum to it, and the twelve months do not.
        """
        imputer = fit_quietly(make_imputer(keep_lower_frequencies=False), reference_timeseries)
        diverging = reference_timeseries.copy()
        diverging['q1'] = np.nan
        diverging.loc[diverging.index.month == 12, 'q1'] = [30.0, 90.0, 150.0]

        result = transform_quietly(imputer, diverging)

        # Valeur d'or : recalage trimestriel, T4 2021 = 30 (et non l'année)
        assert result.loc['2021-10-31':'2021-12-31', 'q1'].sum() == pytest.approx(30.0)
        assert result.loc['2021', 'q1'].sum() != pytest.approx(30.0)

    def test_divergence_is_named_by_pair_on_a_panel(self, mixed_freq_panel_multifrequency):
        """On a panel, the warning names the diverging (entity, column) pair and it alone."""
        imputer = fit_quietly(make_imputer(), mixed_freq_panel_multifrequency)
        diverging = mixed_freq_panel_multifrequency.copy()
        # q1 de DE publiée en décembre seulement : annuelle au transform
        dates = diverging.index.get_level_values('date')
        is_de = diverging.index.get_level_values(0) == 'DE'
        diverging.loc[is_de, 'q1'] = np.nan
        diverging.loc[is_de & (dates.month == 12), 'q1'] = [30.0, 90.0, 150.0]

        messages = _warning_messages(lambda: imputer.transform(diverging))

        divergences = [m for m in messages if 'different frequency at transform time' in m]
        assert len(divergences) == 1
        assert "('DE', 'q1')" in divergences[0]
        assert 'FR' not in divergences[0] and 'IT' not in divergences[0]

    def test_missing_column_raises_on_a_panel(self, mixed_freq_panel_multifrequency):
        """A column of the fit missing from a panel transform raises, naming it."""
        imputer = fit_quietly(make_imputer(), mixed_freq_panel_multifrequency)
        with pytest.raises(ValueError, match=r"missing 1 column\(s\).*\['v'\]"):
            imputer.transform(mixed_freq_panel_multifrequency.drop(columns=['v']))

    def test_empty_fit_column_is_still_required(self, reference_timeseries):
        """A column entirely NaN at fit time is part of the frozen contract all the same."""
        data = reference_timeseries.assign(vide=np.nan)
        imputer = fit_quietly(make_imputer(), data)

        with pytest.raises(ValueError, match=r"\['vide'\]"):
            imputer.transform(reference_timeseries)

    def test_failed_transform_leaves_the_last_pass_intact(self, reference_timeseries):
        """A transform refused by the frequency check leaves the inversion of the last pass intact.

        The inversion reads the input snapshot of the last pass. A call that
        raises is not a pass: it must leave that snapshot,
        the matrix and its statistics as it found them, or the exact
        restoration silently restores the refused data.
        """
        imputer = make_imputer(restore_original_values=True)
        transformed = fit_transform_quietly(imputer, reference_timeseries)
        pass_state = copy.deepcopy(
            (imputer.imputation_provenance_, imputer.provenance_statistics_)
        )
        refused = reference_timeseries.drop(columns=['q1']).assign(
            a1=reference_timeseries['a1'] * 2
        )
        with pytest.raises(ValueError):
            imputer.transform(refused)

        assert_same_state(
            pass_state, (imputer.imputation_provenance_, imputer.provenance_statistics_)
        )
        restored = imputer.inverse_transform(transformed)

        # Valeurs d'or : les ancres 120 / 132 / 150 de la dernière passe
        observed = reference_timeseries['a1'].notna()
        np.testing.assert_allclose(
            restored.loc[observed, 'a1'].to_numpy(), [120.0, 132.0, 150.0]
        )


class TestKeepLowerFrequencies:
    """``keep_lower_frequencies`` is a pure DISPLAY parameter."""

    def test_keep_lower_frequencies_is_display_only(self, reference_timeseries):
        """Les valeurs du niveau cible sont identiques sous True et sous False."""
        stacked = fit_transform_quietly(
            make_imputer(keep_lower_frequencies=True), reference_timeseries
        )
        flat = fit_transform_quietly(
            make_imputer(keep_lower_frequencies=False), reference_timeseries
        )

        # Sous False, aucun niveau de fréquence dans l'index
        assert not isinstance(flat.index, pd.MultiIndex)
        # Sous True, le niveau cible porte exactement les mêmes valeurs
        target_level = stacked.xs('M', level='frequency')
        pd.testing.assert_frame_equal(target_level, flat)

    def test_no_intermediate_level_without_axis_2(self, reference_timeseries):
        """Sous impute_intermediate_frequencies=False, un seul niveau empilé."""
        stacked = fit_transform_quietly(
            make_imputer(impute_intermediate_frequencies=False),
            reference_timeseries,
        )
        assert set(stacked.index.get_level_values('frequency')) == {'M'}

    def test_intermediate_levels_appear_under_axis_2(self, reference_timeseries):
        """Sous True, les étapes intermédiaires deviennent autant de niveaux."""
        stacked = fit_transform_quietly(
            make_imputer(impute_intermediate_frequencies=True),
            reference_timeseries,
        )
        levels = set(stacked.index.get_level_values('frequency'))
        assert 'M' in levels and len(levels) > 1

    @pytest.mark.slow
    def test_keep_lower_frequencies_is_display_only_on_a_panel(
        self, mixed_freq_panel_multifrequency
    ):
        """On PANEL-F under axis 2, the target level is the flat output, value for value."""
        stacked = fit_transform_quietly(
            make_imputer(impute_intermediate_frequencies=True),
            mixed_freq_panel_multifrequency,
        )
        flat = fit_transform_quietly(
            make_imputer(impute_intermediate_frequencies=True, keep_lower_frequencies=False),
            mixed_freq_panel_multifrequency,
        )
        pd.testing.assert_frame_equal(stacked.xs('M', level='frequency'), flat)

    def test_stacked_index_shape_on_a_panel(self, mixed_freq_panel_multifrequency):
        """The frequency level sits on the entity side, levels in progression order."""
        imputer = make_imputer(impute_intermediate_frequencies=True)
        stacked = fit_transform_quietly(imputer, mixed_freq_panel_multifrequency)

        assert list(stacked.index.names) == ['country', 'frequency', 'date']
        # Une étape de la progression, un niveau : Q puis M
        assert list(pd.unique(stacked.index.get_level_values('frequency'))) == ['Q', 'M']
        assert len(imputer.frequency_progression_) == 2

    @pytest.mark.parametrize('keep', [True, False], ids=['stacked', 'flat'])
    def test_provenance_follows_the_shape_of_the_output(self, reference_timeseries, keep):
        """The matrix of the transform shares the index of the output, stacked or flat."""
        imputer = make_imputer(impute_intermediate_frequencies=True, keep_lower_frequencies=keep)
        result = fit_transform_quietly(imputer, reference_timeseries)

        pd.testing.assert_index_equal(imputer.imputation_provenance_.index, result.index)

    @pytest.mark.parametrize('strategy', ['interpolate', 'model'])
    def test_each_level_carries_the_imputations_of_its_own_stage(
        self, reference_timeseries, strategy
    ):
        """Each stacked level holds what its stage produced, under every strategy.

        ``stage_frame`` reads the mirror under ``'model'`` only: under
        ``'interpolate'`` a level built from it alone would show no imputation
        at all. Gold values: ``a1`` totals 120 in 2021, whether split
        into four quarters at the Q level or into twelve months at the M one —
        and the monthly level never reuses a quarterly value.
        """
        stacked = fit_transform_quietly(
            make_imputer(covariate_strategy=strategy, impute_intermediate_frequencies=True),
            reference_timeseries,
        )
        quarterly = stacked.xs('Q', level='frequency').loc['2021', 'a1']
        monthly = stacked.xs('M', level='frequency').loc['2021', 'a1']

        # Niveau Q : quatre trimestres imputés, total annuel conservé
        assert len(quarterly) == 4 and quarterly.notna().all()
        assert quarterly.sum() == pytest.approx(120.0)
        # Niveau M : douze mois imputés, même total, aucune valeur trimestrielle reposée
        assert monthly.sum() == pytest.approx(120.0)
        assert not monthly.reindex(quarterly.index).isin(quarterly.to_numpy()).any()

    @pytest.mark.slow
    def test_entity_left_out_of_the_last_stage_keeps_its_last_passage(
        self, mixed_freq_panel_multifrequency
    ):
        """An entity no later stage concerns keeps the rows of its last stage.

        Targets ``{FR: M, DE: Q, IT: M}`` under axis 2: ``DE`` stops at the
        quarterly stage. At the monthly level it holds its quarter-end rows
        only, with its quarterly values: its other months would carry
        nothing, their values being aggregated on the row of their quarter
        The inversion gives them back.
        """
        imputer = make_imputer(
            target_frequency={'FR': 'M', 'DE': 'Q', 'IT': 'M'},
            impute_intermediate_frequencies=True, restore_original_values=True,
        )
        stacked = fit_transform_quietly(imputer, mixed_freq_panel_multifrequency)
        de_quarterly = stacked.xs('Q', level='frequency').xs('DE', level=0)
        de_monthly = stacked.xs('M', level='frequency').xs('DE', level=0)

        # Niveau mensuel de DE : ses seules fins de trimestre, valeurs du passage Q
        pd.testing.assert_frame_equal(de_monthly, de_quarterly, check_freq=False)
        assert len(de_monthly) == 12
        # L'inversion rend les 36 mois de DE, observations comprises
        restored = imputer.inverse_transform(stacked)
        pd.testing.assert_frame_equal(
            restored.loc['DE'], mixed_freq_panel_multifrequency.loc['DE'], check_freq=False
        )

    @pytest.mark.parametrize('keep', [True, False], ids=['stacked', 'flat'])
    def test_no_empty_cell_of_the_output_is_marked(
        self, mixed_freq_panel_multifrequency, keep
    ):
        """The provenance describes the output: an empty cell is never marked.

        Under ``{FR: M, DE: Q, IT: M}`` the final frame is completed
        with the 24 months of ``DE`` its quarterly grid lacks: these empty rows
        carry no mark, ``m1`` cells included.
        """
        imputer = make_imputer(
            target_frequency={'FR': 'M', 'DE': 'Q', 'IT': 'M'}, keep_lower_frequencies=keep
        )
        result = fit_transform_quietly(imputer, mixed_freq_panel_multifrequency)
        marks = imputer.imputation_provenance_

        marked_empty = marks.notna() & result.isna()
        assert int(marked_empty.to_numpy().sum()) == 0
        # Les 24 mois hors fin de trimestre de DE ne sont plus dans la sortie
        flat = result.droplevel('frequency') if keep else result
        assert len(flat.loc['DE']) == 12

    @pytest.mark.parametrize('keep', [True, False], ids=['stacked', 'flat'])
    def test_output_index_stays_sorted(self, mixed_freq_panel_multifrequency, keep):
        """The output index is sorted, entity blocks contiguous, like the input it came from.

        An unsorted input is refused; an output the class itself would
        refuse cannot be fed back to it, nor to any strict component.
        """
        result = fit_transform_quietly(
            make_imputer(
                target_frequency={'FR': 'M', 'DE': 'Q', 'IT': 'M'},
                keep_lower_frequencies=keep,
            ),
            mixed_freq_panel_multifrequency,
        )
        assert result.index.is_monotonic_increasing


class TestTargetCoarserThanTheIndex:
    """A target coarser than the index: one row per period, aggregations marked.

    ``TS`` is monthly, the target quarterly: ``m1`` is aggregated on each
    quarter end, ``q1`` is already at target, ``a1`` and ``a2`` are imputed.
    """

    def test_output_holds_one_sorted_row_per_quarter(self, reference_timeseries):
        """Twelve quarter ends, sorted; ``m1`` holds the sum of its three months."""
        imputer = make_imputer(target_frequency='Q', keep_lower_frequencies=False)

        result = fit_transform_quietly(imputer, reference_timeseries)

        assert len(result) == 12 and result.index.is_monotonic_increasing
        # Valeurs d'or : 100 + 101 + 102 au premier trimestre 2021
        assert result['m1'].iloc[0] == pytest.approx(303.0)
        pd.testing.assert_series_equal(
            result['q1'], reference_timeseries['q1'].dropna(), check_freq=False
        )

    def test_aggregations_are_marked_aggregated(self, reference_timeseries):
        """``m1`` sums are ``AGGREGATED``, never the ``ORIGINAL`` of the input cell of the date."""
        imputer = make_imputer(target_frequency='Q', keep_lower_frequencies=False)
        fit_transform_quietly(imputer, reference_timeseries)
        marks = imputer.imputation_provenance_

        assert set(marks['m1']) == {ProvenanceType.AGGREGATED}
        assert set(marks['q1']) == {ProvenanceType.ORIGINAL}

    @pytest.mark.parametrize('restore', [False, True], ids=['mask', 'restore'])
    def test_inverse_gives_back_the_monthly_index(self, reference_timeseries, restore):
        """The 36 months come back; a quarterly sum is never handed back as a month.

        Without ``restore_original_values`` only the ``ORIGINAL`` cells
        return: ``q1`` at its quarter ends, nothing of ``m1``. With it, every
        observation.
        """
        imputer = make_imputer(
            target_frequency='Q', keep_lower_frequencies=False, restore_original_values=restore
        )
        transformed = fit_transform_quietly(imputer, reference_timeseries)

        restored = imputer.inverse_transform(transformed)

        assert restored.index.equals(reference_timeseries.index)
        if restore:
            pd.testing.assert_frame_equal(restored, reference_timeseries, check_freq=False)
        else:
            assert restored['m1'].isna().all()
            pd.testing.assert_series_equal(
                restored['q1'], reference_timeseries['q1'], check_freq=False
            )
