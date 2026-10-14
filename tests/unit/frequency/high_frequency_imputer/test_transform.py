"""``HighFrequencyImputer.transform`` and its multi-frequency output (§12.4).

Covers the transform outside the fit window (I7), the extended window on
an irregular index, the frequency control (D11), ``keep_lower_frequencies``
and the stacked output (D35, D37, D38, D43), a target coarser than the index
(D43, D44), the transform state (B19, D45) — the replay never fits and never
mutates the fitted state (§12.1) —, the transform on dates after the fit, the
shared-model replay (§5.8 R6), new entities and pairs unknown to the fit
(D34, D42), columns without any observation in the transformed data (§12.1,
§5.10), the coverage of a short entity of the realistic heterogeneous panel,
the input contract of ``transform`` (``y``, additive transformer, one-row
input, truncated period — D46) and the reserved ``'frequency'`` level name.
"""

# Modules de base
import copy
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

# Objets testés
from tsforecast.frequency.provenance import ProvenanceType, provenance_statistics
from tsforecast.frequency.training_set_builder import TrainingSetBuilder
from tsforecast.utils.frequency import normalize_frequency

# Outils de test partagés
from tests.support.estimators import SpyEstimator
from tests.unit.frequency.high_frequency_imputer.helpers import (
    assert_same_state,
    fit_quietly,
    fit_transform_quietly,
    fit_with_spy,
    has_frequency_level_in_output,
    make_imputer,
    public_fitted_state,
    transform_quietly,
)


# Panel augmenté d'une entité inconnue du fit (D34)
def _with_new_entity(
    data: pd.DataFrame,
    source: str = 'IT',
    name: str = 'ES',
    clear: tuple = ('v',),
) -> pd.DataFrame:
    """Return a panel extended by a copy of one entity, renamed.

    Args:
        data: Panel with an ``(entity, date)`` MultiIndex.
        source: Entity copied.
        name: Name of the new entity.
        clear: Columns emptied on the copy.

    Returns:
        The extended panel, sorted.
    """
    added = data.xs(source, level=0, drop_level=False).copy()
    added.index = pd.MultiIndex.from_arrays(
        [[name] * len(added), added.index.get_level_values(-1)],
        names=data.index.names,
    )
    for column in clear:
        added[column] = np.nan
    return pd.concat([data, added]).sort_index()


# Messages d'avertissement d'un appel, relevés sans filtre
def _warning_messages(call) -> list:
    """Run a callable and return the messages of every warning it emitted.

    Args:
        call: Callable without arguments.

    Returns:
        The messages, as strings, in emission order.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        call()
    return [str(item.message) for item in caught]


# Ajustement puis rejeu silencieux d'un couple (X, y)
def _fit_transform_pair(imputer, X: pd.DataFrame, y: pd.Series) -> tuple:
    """Fit then transform ``(X, y)``, swallowing the legitimate warnings.

    Args:
        imputer: Imputer to fit.
        X: Features.
        y: Target, merged as a column of its own name.

    Returns:
        The pair ``(X_transformed, y_transformed)``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return imputer.fit_transform(X, y)


class TestTransformOutsideWindow:
    """I7 — le transform hors fenêtre impute au lieu de vider."""

    # Fixture privée d'un jeu dont la fin sort de la fenêtre stricte
    @staticmethod
    def _truncated(data: pd.DataFrame) -> pd.DataFrame:
        """Jeu TS dont les trois colonnes basses fréquences s'arrêtent en 2022."""
        frame = data.copy()
        tail = frame.index >= pd.Timestamp('2023-01-31')
        frame.loc[tail, ['q1', 'a1', 'a2']] = np.nan
        return frame

    def test_transform_outside_fit_window(self, reference_timeseries):
        """I7 — impute, ne détruit aucune observation, avertit UNE fois."""
        data = self._truncated(reference_timeseries)
        imputer = fit_quietly(make_imputer(), data)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = imputer.transform(data)

        # Un seul avertissement de fenêtre, nommant le nombre de lignes
        window_warnings = [
            str(w.message) for w in caught if 'outside' in str(w.message)
        ]
        assert len(window_warnings) == 1
        assert 'row(s)' in window_warnings[0]

        # Aucune ligne d'entrée perdue, aucune observation détruite
        flat = result.droplevel('frequency')
        assert len(flat) == len(data)
        observed = data['m1'].notna()
        assert np.allclose(flat['m1'].to_numpy(), data['m1'].to_numpy())
        assert observed.all()

        # Les lignes hors fenêtre gardent leurs valeurs d'entrée : la colonne
        # n'est jamais vidée sans être réécrite
        tail = flat.index >= pd.Timestamp('2023-01-31')
        assert flat.loc[tail, 'm1'].notna().all()

    def test_input_observations_survive_every_column(self, reference_timeseries):
        """Aucune cellule observée de m1 n'est jamais perdue au transform."""
        data = self._truncated(reference_timeseries)
        imputer = fit_quietly(make_imputer(), data)
        flat = transform_quietly(imputer, data).droplevel('frequency')
        assert flat['m1'].notna().sum() == data['m1'].notna().sum()


    def test_transform_outside_fit_window_on_panel(self, mixed_freq_panel_heterogeneous):
        """I7 sur PANEL — impute, ne détruit aucune observation, avertit UNE fois."""
        data = mixed_freq_panel_heterogeneous.copy()
        tail = data.index.get_level_values('date') >= pd.Timestamp('2023-01-31')
        data.loc[tail, ['q1', 'a1', 'a2']] = np.nan
        imputer = fit_quietly(make_imputer(), data)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = imputer.transform(data)
        window_warnings = [
            str(item.message) for item in caught if 'outside' in str(item.message)
        ]
        assert len(window_warnings) == 1

        # Aucune ligne d'entrée perdue, aucune observation à la fréquence cible
        # détruite ; les ancres basses fréquences sont, elles, ré-exprimées (D37)
        flat = result.droplevel('frequency').reindex(data.index)
        assert len(result.droplevel('frequency')) == len(data)
        for column in ['m1', 'climat_affaires']:
            observed = data[column].notna()
            np.testing.assert_allclose(
                flat.loc[observed, column].to_numpy(), data.loc[observed, column].to_numpy()
            )


class TestIrregularIndexExtendedWindow:
    """Fenêtre étendue vers des périodes où la grille cible n'existe pas dans l'index.

    Observations annuelles isolées AVANT le début de la grille mensuelle : sous
    un seuil nul, la fenêtre y remonte et la grille densifiée porte des mois
    absents de l'entrée. La matrice de provenance, initialisée sur l'index
    d'entrée, levait une KeyError à l'écriture.
    """

    @staticmethod
    def _irregular_timeseries() -> pd.DataFrame:
        """Mensuelles dès 2018, trimestrielle, annuelle dès 2015 (index irrégulier)."""
        rng = np.random.default_rng(0)
        monthly = pd.date_range('2018-01-01', '2021-12-01', freq='MS')
        frame = pd.DataFrame(index=monthly)
        frame['m1'] = 100 + rng.normal(0, 1, len(monthly)).cumsum()
        frame['m2'] = 50 + rng.normal(0, 1, len(monthly))
        frame['q1'] = np.where(monthly.month.isin([1, 4, 7, 10]),
                               300 + rng.normal(0, 5, len(monthly)), np.nan)
        annual = pd.date_range('2015-01-01', '2021-01-01', freq='YS')
        frame = frame.reindex(frame.index.union(annual))
        frame['a1'] = np.nan
        frame.loc[annual, 'a1'] = 1200 + rng.normal(0, 20, len(annual))
        frame.index.name = 'date'
        return frame

    @pytest.mark.parametrize('scope', ['extended_backward', 'extended_both'])
    @pytest.mark.parametrize('strategy', ['interpolate', 'model'])
    def test_fit_transform_on_densified_grid(self, scope, strategy):
        """Le fit aboutit ; la provenance couvre les mois absents de l'entrée."""
        data = self._irregular_timeseries()
        imputer = make_imputer(
            imputation_scope=scope, coverage_threshold=0.0,
            covariate_strategy=strategy, keep_lower_frequencies=False,
        )
        imputer = fit_quietly(imputer, data)

        # La fenêtre remonte bien jusqu'à la première observation annuelle
        assert imputer.imputation_window_[0] == pd.Timestamp('2015-01-01')

        # Les mois de 2015 absents de l'entrée portent une provenance imputée
        matrix = imputer.imputation_provenance_
        added = pd.date_range('2015-02-01', '2015-12-01', freq='MS')
        assert added.isin(matrix.index).all()
        assert matrix.loc[added, 'a1'].notna().all()
        # Aucune ligne créée ne se déclare ORIGINAL
        assert not (matrix.loc[added] == ProvenanceType.ORIGINAL).any().any()
        assert matrix.index.is_monotonic_increasing

        # Le transform rejoue sans erreur et produit ces mois
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = imputer.transform(data)
        assert result.loc[added, 'a1'].notna().all()

    def test_additivity_holds_on_the_isolated_years(self):
        """Sous 'sum', les douze mois d'une année isolée redonnent son total."""
        data = self._irregular_timeseries()
        imputer = fit_quietly(make_imputer(
            imputation_scope='extended_backward', coverage_threshold=0.0,
            keep_lower_frequencies=False,
        ), data)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = imputer.transform(data)
        for year in (2015, 2016, 2017):
            months = result.loc[f'{year}-01-01':f'{year}-12-01', 'a1']
            assert len(months) == 12
            assert months.sum() == pytest.approx(data.loc[f'{year}-01-01', 'a1'])


class TestTransformFrequencyControl:
    """D11 — contrôle des fréquences détectées au transform (§12.1)."""

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
        raises is not a pass (D45, ANO-FREQ-078): it must leave that snapshot,
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

        # Valeurs d'or du §2.2 : les ancres 120 / 132 / 150 de la dernière passe
        observed = reference_timeseries['a1'].notna()
        np.testing.assert_allclose(
            restored.loc[observed, 'a1'].to_numpy(), [120.0, 132.0, 150.0]
        )


class TestKeepLowerFrequencies:
    """`keep_lower_frequencies` est un paramètre d'AFFICHAGE pur (§12.4)."""

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
        """§5.9 — the frequency level sits on the entity side, levels in progression order."""
        imputer = make_imputer(impute_intermediate_frequencies=True)
        stacked = fit_transform_quietly(imputer, mixed_freq_panel_multifrequency)

        assert list(stacked.index.names) == ['country', 'frequency', 'date']
        # Une étape de la progression, un niveau : Q puis M (D35)
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
        """D37 — each stacked level holds what its stage produced, under every strategy.

        ``stage_frame`` reads the mirror under ``'model'`` only: under
        ``'interpolate'`` a level built from it alone would show no imputation
        at all. Gold values of §2.2: ``a1`` totals 120 in 2021, whether split
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
        """D38 / D43 — an entity no later stage concerns keeps the rows of its last stage.

        Targets ``{FR: M, DE: Q, IT: M}`` under axis 2: ``DE`` stops at the
        quarterly stage. At the monthly level it holds its quarter-end rows
        only, with its quarterly values: its other months would carry
        nothing, their values being aggregated on the row of their quarter
        (D43). The inversion gives them back (I8).
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
        """D44 — the provenance describes the output: an empty cell is never marked.

        Under ``{FR: M, DE: Q, IT: M}`` the final frame used to be completed
        with the 24 months of ``DE`` its quarterly grid lacks: empty rows whose
        ``m1`` cells kept the ``ORIGINAL`` mark of the input (ANO-FREQ-079).
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

        §12.5 refuses an unsorted input; an output the class itself would
        refuse cannot be fed back to it, nor to any strict component
        (ANO-FREQ-075).
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
    """D43 / D44 — a target coarser than the index: one row per period, aggregations marked.

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
        """I8 — the 36 months come back; a quarterly sum is never handed back as a month.

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


class TestTransformState:
    """B19 et §13.2 — l'état de transform est écrasé puis purgé."""

    def test_transform_state_purged_at_fit(self, reference_timeseries):
        """B19 — un fit efface l'état laissé par le transform précédent.

        Observé par l'API publique : après ``fit_transform(A)`` puis ``fit(B)``,
        l'inversion exacte restitue les valeurs de ``B`` — celles du dernier
        passage —, et la provenance est celle, à plat, du fit.
        """
        data_a = reference_timeseries
        data_b = reference_timeseries.copy()
        data_b['a1'] = data_b['a1'] * 2
        imputer = make_imputer(restore_original_values=True)
        fit_transform_quietly(imputer, data_a)
        assert has_frequency_level_in_output(imputer.imputation_provenance_)

        fit_quietly(imputer, data_b)
        # Le fit réécrit sa propre matrice, jamais celle d'un transform passé
        assert not has_frequency_level_in_output(imputer.imputation_provenance_)
        restored = imputer.inverse_transform(data_b)
        observed = data_b['a1'].notna()
        np.testing.assert_allclose(
            restored.loc[observed, 'a1'].to_numpy(), data_b.loc[observed, 'a1'].to_numpy()
        )

    def test_provenance_is_overwritten_by_each_transform(self, reference_timeseries):
        """§13.2 — l'attribut porte la provenance du DERNIER transform."""
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        after_fit = imputer.imputation_provenance_.copy()
        transform_quietly(imputer, reference_timeseries)
        after_transform = imputer.imputation_provenance_
        # La matrice du transform porte le niveau de fréquence, celle du fit non
        assert has_frequency_level_in_output(after_transform)
        assert not has_frequency_level_in_output(after_fit)

    @pytest.mark.parametrize('keep', [True, False], ids=['stacked', 'flat'])
    def test_provenance_statistics_describe_the_last_pass(self, reference_timeseries, keep):
        """D45 — ``provenance_statistics_`` follows the last pass, counted on its target level.

        After a transform of 2021 alone, the statistics count its twelve rows,
        whatever ``keep_lower_frequencies`` stacks — a display parameter
        changes no figure (ANO-FREQ-077).
        """
        imputer = make_imputer(impute_intermediate_frequencies=True, keep_lower_frequencies=keep)
        fit_quietly(imputer, reference_timeseries)
        fit_statistics = copy.deepcopy(imputer.provenance_statistics_)

        # Rejeu sur la seule année 2021 : une matrice trois fois plus courte
        transform_quietly(imputer, reference_timeseries.loc['2021'])
        marks = imputer.imputation_provenance_
        target = marks.xs('M', level='frequency') if keep else marks

        assert len(target) == 12
        assert_same_state(provenance_statistics(target), imputer.provenance_statistics_)
        assert fit_statistics['overall']['not_imputed'] + sum(
            fit_statistics['overall'][member.value] for member in ProvenanceType
        ) == 36 * 4

    def test_never_imputed_cells_stay_unmarked_after_transform(self, reference_timeseries):
        """A NaN cell no stage imputes stays unmarked after ``transform``, as after ``fit``.

        §5.10 and §6.4: ``ORIGINAL`` marks the cells non-null in the input only;
        a NaN cell never imputed is unmarked (``isna``). I1 demands the same
        provenance from the fit and from its replay.
        """
        # Jeu TS dont les trois colonnes basses fréquences s'arrêtent en 2022 :
        # leurs cellules de 2023 sont hors fenêtre, donc jamais imputées
        data = reference_timeseries.copy()
        tail = data.index >= pd.Timestamp('2023-01-31')
        data.loc[tail, ['q1', 'a1', 'a2']] = np.nan

        imputer = fit_quietly(make_imputer(keep_lower_frequencies=False), data)
        assert imputer.imputation_provenance_.loc[tail, 'a1'].isna().all()

        transform_quietly(imputer, data)
        assert imputer.imputation_provenance_.loc[tail, 'a1'].isna().all()


    def test_fit_and_replay_provenances_agree_on_a_panel(self, panel_f_without_it_v):
        """I1 sur PANEL — la matrice du fit est celle du rejeu, cellules muettes comprises.

        Les 36 cellules de ``v`` d'``IT``, jamais observées ni imputées, sont
        non marquées des deux côtés (ANO-FREQ-050).
        """
        imputer = fit_quietly(
            make_imputer(keep_lower_frequencies=False), panel_f_without_it_v
        )
        after_fit = imputer.imputation_provenance_.copy()
        transform_quietly(imputer, panel_f_without_it_v)
        after_transform = imputer.imputation_provenance_

        assert after_fit.loc['IT', 'v'].isna().all()
        pd.testing.assert_frame_equal(
            after_transform.astype(object).sort_index(),
            after_fit.astype(object).reindex(after_transform.index).sort_index(),
            check_freq=False,
        )


class TestTransformLeavesTheFitIntact:
    """§12.1 — the transform replays the frozen plan: it never fits, never mutates.

    Public counterpart of the former internal test
    ``TestTransformState::test_fit_state_is_restored_after_a_transform``
    (identity of two private collaborators): what matters is that no fitted
    attribute moves and that a transform depends on its own input only.
    """

    # Trois données de rejeu qui sollicitent chacune un chemin propre au transform
    @staticmethod
    def _replay_inputs(name: str, request) -> tuple:
        """Return ``(fit data, transform data, impute_unobserved_entities)``."""
        if name == 'timeseries_later_dates':
            # Fit sur 2021-2022, rejeu sur 2021-2023 : périodes nouvelles
            data = request.getfixturevalue('reference_timeseries')
            return data.loc[:'2022-12-31'], data, False
        if name == 'timeseries_diverging_frequency':
            # q1 publiée en décembre seulement : divergence de fréquence (D11)
            data = request.getfixturevalue('reference_timeseries')
            diverging = data.copy()
            diverging['q1'] = np.nan
            diverging.loc[diverging.index.month == 12, 'q1'] = [30.0, 90.0, 150.0]
            return data, diverging, False
        # Entité ES inconnue du fit, imputée sans ancre (D34) : étapes dérivées
        data = request.getfixturevalue('mixed_freq_panel_multifrequency')
        return data, _with_new_entity(data), True

    @pytest.mark.slow
    def test_transform_never_refits(self, mixed_freq_panel_multifrequency):
        """No training set is built and no estimator is fitted during a transform.

        The richest replay: ``'model'`` cascade, intermediate stages, and an
        entity born at transform time served by derived steps (D34). The
        public ``TrainingSetBuilder.build`` is patched at class level, so
        whatever path reaches it raises — and every call is recorded, so a
        caller swallowing the ``AssertionError`` still shows.
        """
        imputer = fit_with_spy(
            mixed_freq_panel_multifrequency,
            covariate_strategy='model', impute_intermediate_frequencies=True,
            impute_unobserved_entities=True,
        )
        fits_after_fit = SpyEstimator.n_fits
        calls = []

        def _forbidden(*args, **kwargs):
            calls.append(kwargs.get('column'))
            raise AssertionError('the training set must never be rebuilt at transform')

        with patch.object(TrainingSetBuilder, 'build', side_effect=_forbidden):
            transform_quietly(imputer, _with_new_entity(mixed_freq_panel_multifrequency))

        assert calls == []
        # Aucun estimateur ajusté pendant le rejeu : le compteur de classe reste
        assert SpyEstimator.n_fits == fits_after_fit

    @pytest.mark.parametrize(
        'scenario',
        ['timeseries_later_dates', 'timeseries_diverging_frequency', 'panel_new_entity'],
    )
    def test_transform_does_not_mutate_the_fitted_state(self, scenario, request):
        """Every public fitted attribute is, after a transform, what the fit left.

        A deep copy of the public ``_`` attributes is taken right after the
        fit and compared structurally after the transform (the fitted imputer
        copies since ANO-FREQ-049). ``imputation_provenance_`` and
        ``provenance_statistics_`` alone are left out: §13.2 and D45 make
        them describe the LAST pass, rewritten by design.
        """
        fit_data, transform_data, unobserved = self._replay_inputs(scenario, request)
        imputer = fit_quietly(
            make_imputer(covariate_strategy='model', impute_unobserved_entities=unobserved),
            fit_data,
        )
        snapshot = copy.deepcopy(public_fitted_state(imputer))

        transform_quietly(imputer, transform_data)

        assert_same_state(snapshot, public_fitted_state(imputer))

    @pytest.mark.slow
    def test_transform_depends_on_its_own_input_only(self, mixed_freq_panel_multifrequency):
        """A transform of A gives the same result before and after a transform of B.

        B carries an entity unknown to the fit (D34): its replay installs
        derived steps and extended bindings, none of which may leak into the
        next call.
        """
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True), mixed_freq_panel_multifrequency
        )
        first = transform_quietly(imputer, mixed_freq_panel_multifrequency)
        first_provenance = imputer.imputation_provenance_.copy()

        transform_quietly(imputer, _with_new_entity(mixed_freq_panel_multifrequency))
        again = transform_quietly(imputer, mixed_freq_panel_multifrequency)

        pd.testing.assert_frame_equal(again, first)
        pd.testing.assert_frame_equal(imputer.imputation_provenance_, first_provenance)


class TestTransformOnLaterDates:
    """I7 — a transform on dates after the fit imputes the new periods only.

    The fit sees 2021-2022, the transform 2021-2023: the 2023 anchors are new
    totals the frozen models disaggregate, while the cells of the fit period
    are recomputed identically — same models, same covariates, same totals.
    """

    @pytest.mark.parametrize('strategy', ['interpolate', 'model'])
    def test_new_periods_are_imputed_and_add_up(self, reference_timeseries, strategy):
        """The twelve months of 2023 are imputed and sum to the new annual anchors."""
        imputer = fit_quietly(
            make_imputer(covariate_strategy=strategy, keep_lower_frequencies=False),
            reference_timeseries.loc[:'2022-12-31'],
        )
        result = transform_quietly(imputer, reference_timeseries)
        provenance = imputer.imputation_provenance_.loc['2023']

        # Cellules 2023 de a1 et a2 toutes produites par un modèle
        for column in ('a1', 'a2'):
            assert provenance[column].map(lambda mark: mark.value.startswith('model_')).all()
        # Valeurs d'or du §2.2 : totaux 2023 de 150 (a1) et 72 (a2)
        assert result.loc['2023', 'a1'].sum() == pytest.approx(150.0)
        assert result.loc['2023', 'a2'].sum() == pytest.approx(72.0)

    @pytest.mark.parametrize('strategy', ['interpolate', 'model'])
    def test_fit_period_is_unchanged(self, reference_timeseries, strategy):
        """Values and provenances of 2021-2022 are those of the transform of the fit data."""
        fit_data = reference_timeseries.loc[:'2022-12-31']
        imputer = fit_quietly(
            make_imputer(covariate_strategy=strategy, keep_lower_frequencies=False), fit_data
        )
        on_fit_data = transform_quietly(imputer, fit_data)
        provenance_on_fit_data = imputer.imputation_provenance_.copy()

        extended = transform_quietly(imputer, reference_timeseries)

        pd.testing.assert_frame_equal(
            extended.loc[:'2022-12-31'], on_fit_data, check_freq=False
        )
        pd.testing.assert_frame_equal(
            imputer.imputation_provenance_.loc[:'2022-12-31'], provenance_on_fit_data,
            check_freq=False,
        )

    def test_new_periods_on_a_panel(self, mixed_freq_panel_multifrequency):
        """On PANEL-F, each 2023 period of ``v`` adds up to its new observation, per entity.

        ``FR`` publishes ``v`` once a year (gold value 150.0, §2.5), ``DE``
        once a quarter; ``IT`` observes it monthly and is never rewritten.
        """
        dates = mixed_freq_panel_multifrequency.index.get_level_values('date')
        imputer = fit_quietly(
            make_imputer(keep_lower_frequencies=False),
            mixed_freq_panel_multifrequency[dates <= '2022-12-31'],
        )
        result = transform_quietly(imputer, mixed_freq_panel_multifrequency)
        source = mixed_freq_panel_multifrequency['v']

        # FR : total annuel 2023 de 150, valeur d'or du §2.5
        assert result.loc['FR', 'v'].loc['2023'].sum() == pytest.approx(150.0)
        # DE : chaque trimestre 2023 somme à son observation trimestrielle
        de_months = result.loc['DE', 'v'].loc['2023']
        de_quarters = source.loc['DE'].loc['2023'].dropna()
        np.testing.assert_allclose(
            de_months.groupby(de_months.index.to_period('Q')).sum().to_numpy(),
            de_quarters.to_numpy(),
        )
        # IT : observations mensuelles intactes
        pd.testing.assert_series_equal(result.loc['IT', 'v'], source.loc['IT'])


class TestSharedModelReplay:
    """§5.8 R6 — un modèle partagé est rejoué par chacune de ses étapes."""

    @pytest.mark.slow
    def test_shared_model_replayed_by_each_step(
        self, mixed_freq_panel_multifrequency
    ):
        """Les deux étapes de 'v' rejouent le MÊME objet modèle, sur leurs entités."""
        imputer = fit_with_spy(mixed_freq_panel_multifrequency)
        monthly = [
            step for step in imputer.imputation_plan_
            if step.var_name == 'v' and step.pred_freq_label == 'M'
        ]
        assert len(monthly) == 2

        # Un seul objet modèle, partagé (§5.8 R6)
        assert monthly[0].model is monthly[1].model
        # Des entités et des recalages distincts
        assert set(monthly[0].entities) != set(monthly[1].entities)
        assert monthly[0].source_frequency != monthly[1].source_frequency

        # Chaque étape rejoue le modèle : autant d'appels de prédiction que
        # d'étapes, au fit comme au transform
        model = monthly[0].model
        fit_calls = len(model.predict_X_)
        transform_quietly(imputer, mixed_freq_panel_multifrequency)
        assert len(model.predict_X_) == 2 * fit_calls

        # Et la symétrie tient malgré le partage
        combined = fit_transform_quietly(
            make_imputer(estimator=SpyEstimator()), mixed_freq_panel_multifrequency
        )
        separate = make_imputer(estimator=SpyEstimator())
        fit_quietly(separate, mixed_freq_panel_multifrequency)
        pd.testing.assert_frame_equal(
            combined, transform_quietly(separate, mixed_freq_panel_multifrequency)
        )


class TestNewEntityAtTransform:
    """D34 — une entité absente du fit, rencontrée au transform."""

    # Fixture privée d'une entité ajoutée après le fit
    @staticmethod
    def _with_new_entity(data: pd.DataFrame) -> pd.DataFrame:
        """PANEL-F augmenté d'une entité ES, dont 'v' n'est jamais observée."""
        added = data.xs('IT', level=0, drop_level=False).copy()
        added.index = pd.MultiIndex.from_arrays(
            [
                ['ES'] * len(added),
                added.index.get_level_values(-1),
            ],
            names=data.index.names,
        )
        added['v'] = np.nan
        return pd.concat([data, added]).sort_index()

    @pytest.mark.slow
    def test_new_entity_is_left_untouched_by_default(
        self, mixed_freq_panel_multifrequency
    ):
        """Sous False, rien ne lui est écrit et un avertissement la nomme."""
        imputer = fit_quietly(make_imputer(), mixed_freq_panel_multifrequency)
        extended = self._with_new_entity(mixed_freq_panel_multifrequency)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = imputer.transform(extended)

        messages = [str(w.message) for w in caught if 'absent from the fit' in str(w.message)]
        assert len(messages) == 1
        assert 'ES' in messages[0]

        # Les 36 cellules de 'v' restent vides
        target = result.xs('M', level='frequency')
        assert target.xs('ES', level=0)['v'].isna().all()
        # Les autres entités sont intactes
        reference = transform_quietly(imputer, mixed_freq_panel_multifrequency)
        for entity in ('FR', 'DE', 'IT'):
            pd.testing.assert_series_equal(
                result.xs('M', level='frequency').xs(entity, level=0)['v'],
                reference.xs('M', level='frequency').xs(entity, level=0)['v'],
            )

    def test_new_entity_is_imputed_on_demand(self, mixed_freq_panel_multifrequency):
        """Sous True, elle est imputée sans ancre, par le modèle du plan."""
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True),
            mixed_freq_panel_multifrequency,
        )
        extended = self._with_new_entity(mixed_freq_panel_multifrequency)
        result = transform_quietly(imputer, extended)

        target = result.xs('M', level='frequency').xs('ES', level=0)
        # Les 36 cellules sont renseignées
        assert target['v'].notna().sum() == 36
        # Elles portent la provenance des cellules sans ancre
        provenance = (
            imputer.imputation_provenance_
            .xs('M', level='frequency').xs('ES', level=0)['v']
        )
        assert set(provenance.dropna().unique()) == {ProvenanceType.MODEL_UNANCHORED}

    @pytest.mark.slow
    def test_new_entity_is_imputed_as_if_present_at_fit(self, mixed_freq_panel_multifrequency):
        """D34 is the mechanism of §5.10, not a second one: same values, same provenance.

        ``ES`` copies ``IT`` without ``v``. Present at fit time, it is an
        unobserved pair of §5.10; born at transform time, it is served by
        derived steps. Since it contributes nothing to the pooled training
        set either way (R1), both routes give it the very same 36 cells.
        """
        extended = _with_new_entity(mixed_freq_panel_multifrequency)
        params = dict(impute_unobserved_entities=True, keep_lower_frequencies=False)

        # Voie du §5.10 : ES présente au fit
        at_fit = make_imputer(**params)
        expected = fit_transform_quietly(at_fit, extended).loc['ES', 'v']
        # Voie D34 : ES née au transform
        at_transform = fit_quietly(make_imputer(**params), mixed_freq_panel_multifrequency)
        actual = transform_quietly(at_transform, extended).loc['ES', 'v']

        pd.testing.assert_series_equal(actual, expected)
        assert set(at_transform.imputation_provenance_.loc['ES', 'v']) == {
            ProvenanceType.MODEL_UNANCHORED
        }

    def test_derived_steps_never_enter_the_plan(self, mixed_freq_panel_multifrequency):
        """D34 — the plan and ``unanchored_pairs_`` stay those of the fit after serving ES."""
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True), mixed_freq_panel_multifrequency
        )
        plan_before = imputer.imputation_plan_.to_diagnostic_frame()

        result = transform_quietly(imputer, _with_new_entity(mixed_freq_panel_multifrequency))

        # ES a bien été servie...
        assert result.xs('M', level='frequency').loc['ES', 'v'].notna().all()
        # ... sans qu'aucune étape dérivée n'entre au plan
        pd.testing.assert_frame_equal(imputer.imputation_plan_.to_diagnostic_frame(), plan_before)
        assert imputer.unanchored_pairs_ == ()

    def test_stage_with_diverging_frequencies_cannot_bind_a_new_entity(
        self, mixed_freq_panel_multifrequency
    ):
        """D34 — a stage whose entities disagree on its frequency leaves the new entity out.

        Targets ``{FR: M, DE: Q, IT: M}`` without axis 2: the single stage
        binds diverging frequencies, nothing departs them for ``ES``. One
        aggregated warning names the entity and the stage; its cells stay NaN
        and unmarked, even under ``impute_unobserved_entities=True``.
        """
        imputer = fit_quietly(
            make_imputer(
                target_frequency={'FR': 'M', 'DE': 'Q', 'IT': 'M'},
                impute_unobserved_entities=True, keep_lower_frequencies=False,
            ),
            mixed_freq_panel_multifrequency,
        )
        extended = _with_new_entity(mixed_freq_panel_multifrequency)
        captured = {}

        messages = _warning_messages(
            lambda: captured.setdefault('result', imputer.transform(extended))
        )

        skipped = [m for m in messages if 'disagree on the stage frequency' in m]
        assert len(skipped) == 1 and "('ES',)" in skipped[0]
        assert captured['result'].loc['ES', 'v'].isna().all()
        assert imputer.imputation_provenance_.loc['ES', 'v'].isna().all()

    def test_observations_of_a_new_entity_are_never_overwritten(
        self, mixed_freq_panel_multifrequency
    ):
        """I4 / I7 / D42 — the periods a new entity observes keep their observed totals.

        ``ES`` copies ``DE``, ``v`` and ``q1`` included: it observes both
        quarterly. It is served by anchored steps (ANO-FREQ-073): the months
        of each observed quarter sum to the observation — under ``'sum'`` an
        anchor is re-expressed, never lost — and its cells carry the
        ``MODEL_*`` of the column's step, never ``MODEL_UNANCHORED``.
        """
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True, keep_lower_frequencies=False),
            mixed_freq_panel_multifrequency,
        )
        extended = _with_new_entity(mixed_freq_panel_multifrequency, source='DE', clear=())
        result = transform_quietly(imputer, extended).loc['ES']
        source = extended.loc['ES']

        # Trimestres observés dont la somme mensuelle s'écarte de l'observation
        gaps = {}
        for column in ('v', 'q1'):
            sums = result[column].groupby(result.index.to_period('Q')).sum()
            observed = source[column].dropna()
            observed.index = observed.index.to_period('Q')
            gap = (sums.reindex(observed.index) - observed).abs()
            gaps[column] = int((gap > 1e-9).sum())
        assert gaps == {'v': 0, 'q1': 0}
        marks = imputer.imputation_provenance_.loc['ES', ['v', 'q1']]
        assert ProvenanceType.MODEL_UNANCHORED not in set(marks.stack())

    def test_new_entity_copy_is_imputed_like_its_model(self, mixed_freq_panel_multifrequency):
        """D42 — same covariates, same frozen model, same totals: ``ES`` gets ``DE``'s cells.

        ``ES`` copies ``DE`` at transform time. Its anchored steps share the
        model of ``DE``'s and rescale on the same quarterly totals: the two
        entities receive the very same values and provenance.
        """
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True, keep_lower_frequencies=False),
            mixed_freq_panel_multifrequency,
        )
        extended = _with_new_entity(mixed_freq_panel_multifrequency, source='DE', clear=())

        result = transform_quietly(imputer, extended)

        pd.testing.assert_frame_equal(result.loc['ES'], result.loc['DE'])
        marks = imputer.imputation_provenance_
        pd.testing.assert_frame_equal(marks.loc['ES'], marks.loc['DE'])

    def test_column_empty_at_fit_and_observed_at_transform_is_anchored(
        self, panel_f_without_it_v, mixed_freq_panel_multifrequency
    ):
        """D42 — a known entity's column the fit never observed is anchored once observed.

        ``IT`` leaves ``v`` empty at fit time: under ``True`` its pair is
        unanchored in the plan. At transform it publishes ``v`` quarterly,
        like ``DE``: the unanchored step would overwrite those observations,
        an anchored one rescales on them.
        """
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True, keep_lower_frequencies=False),
            panel_f_without_it_v,
        )
        assert imputer.unanchored_pairs_ == (('IT', 'v'),)
        published = panel_f_without_it_v.copy()
        published.loc[('IT', slice(None)), 'v'] = (
            mixed_freq_panel_multifrequency.loc['DE', 'v'].to_numpy()
        )

        result = transform_quietly(imputer, published).loc['IT', 'v']

        quarters = result.groupby(result.index.to_period('Q')).sum()
        observed = published.loc['IT', 'v'].dropna()
        np.testing.assert_allclose(quarters.to_numpy(), observed.to_numpy())
        marks = set(imputer.imputation_provenance_.loc['IT', 'v'])
        assert ProvenanceType.MODEL_UNANCHORED not in marks

    def test_new_pair_already_at_the_stage_frequency_is_left_as_given(
        self, mixed_freq_panel_multifrequency
    ):
        """D42 — a pair a new entity observes at the stage frequency has nothing to impute.

        ``ES`` copies ``IT``, which observes ``v`` monthly: at the monthly
        stage the pair is at target, as it would have been at fit time, and
        keeps its observations. Its quarterly ``q1`` is imputed and anchored.
        """
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True, keep_lower_frequencies=False),
            mixed_freq_panel_multifrequency,
        )
        extended = _with_new_entity(mixed_freq_panel_multifrequency, source='IT', clear=())

        result = transform_quietly(imputer, extended).loc['ES']

        pd.testing.assert_series_equal(result['v'], extended.loc['ES', 'v'], check_freq=False)
        assert result['q1'].notna().all()
        quarters = result['q1'].groupby(result.index.to_period('Q')).sum()
        np.testing.assert_allclose(
            quarters.to_numpy(), extended.loc['ES', 'q1'].dropna().to_numpy()
        )

    def test_new_pair_observed_off_its_calendar_is_left_as_given(
        self, mixed_freq_panel_multifrequency
    ):
        """D42 — a new pair dated off the bounds of its period is left out, as at fit.

        ``ES`` publishes ``v`` one month before each quarter end: a quarterly
        reading would attach each value to the wrong quarter. Neither
        anchored nor unanchored, the pair stays as given, and the note of the
        pairs outside the plan names it once.
        """
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True, keep_lower_frequencies=False),
            mixed_freq_panel_multifrequency,
        )
        extended = _with_new_entity(mixed_freq_panel_multifrequency, source='DE', clear=())
        new_rows = extended.index.get_level_values(0) == 'ES'
        extended.loc[new_rows, 'v'] = extended.loc[new_rows, 'v'].shift(-1).to_numpy()
        captured = {}

        messages = _warning_messages(
            lambda: captured.setdefault('result', imputer.transform(extended))
        )

        pd.testing.assert_series_equal(
            captured['result'].loc['ES', 'v'], extended.loc['ES', 'v'], check_freq=False
        )
        notes = [m for m in messages if 'off the calendar bounds' in m]
        assert len(notes) == 1 and "('ES', 'v')" in notes[0]

    def test_pairs_unknown_to_the_fit_stay_as_given_by_default(
        self, mixed_freq_panel_multifrequency
    ):
        """D34 — under ``False`` a new entity is left as given, observed columns included."""
        imputer = fit_quietly(
            make_imputer(keep_lower_frequencies=False), mixed_freq_panel_multifrequency
        )
        extended = _with_new_entity(mixed_freq_panel_multifrequency, source='DE', clear=())

        result = transform_quietly(imputer, extended)

        pd.testing.assert_frame_equal(result.loc['ES'], extended.loc['ES'], check_freq=False)
        assert imputer.imputation_provenance_.loc['ES', 'v'].dropna().eq(
            ProvenanceType.ORIGINAL
        ).all()


class TestColumnWithoutAnchorAtTransform:
    """§12.1 / §5.10 — a column observed at fit keeps its replayed, anchored step.

    ``a1`` is unpublished for 2023. Transformed together with its history,
    the 2023 rows fall outside the recomputed strict window and stay empty.
    Alone, the same rows see ``a1`` without any observation: the column
    leaves the coverage denominator (D39, warned), the window spans them, and
    the frozen step predicts every month without rescaling — the very cells
    the extended window of §7.3 produces with the history. The pair being
    observed at fit, its classification is replayed: ``MODEL_*``, never
    ``MODEL_UNANCHORED``, whatever ``impute_unobserved_entities`` says
    (ANO-FREQ-074, closed by the author).
    """

    # Jeu TS dont l'année 2023 de a1 n'est pas publiée
    @staticmethod
    def _unpublished_2023(data: pd.DataFrame) -> pd.DataFrame:
        """Return TS with ``a1`` blanked over 2023."""
        frame = data.copy()
        frame.loc['2023', 'a1'] = np.nan
        return frame

    def test_unpublished_year_stays_empty_with_its_history(self, reference_timeseries):
        """With its history, the unpublished year stays NaN and unmarked."""
        imputer = fit_quietly(make_imputer(keep_lower_frequencies=False), reference_timeseries)
        result = transform_quietly(imputer, self._unpublished_2023(reference_timeseries))

        assert result.loc['2023', 'a1'].isna().all()
        assert imputer.imputation_provenance_.loc['2023', 'a1'].isna().all()

    @pytest.mark.parametrize('unobserved', [False, True])
    def test_unpublished_year_alone_is_predicted_by_its_replayed_step(
        self, reference_timeseries, unobserved
    ):
        """Alone, the twelve months are predicted, ``MODEL_*``, the D39 warning naming ``a1``."""
        imputer = fit_quietly(
            make_imputer(keep_lower_frequencies=False, impute_unobserved_entities=unobserved),
            reference_timeseries,
        )
        alone = self._unpublished_2023(reference_timeseries).loc['2023']
        captured = {}

        messages = _warning_messages(
            lambda: captured.setdefault('result', imputer.transform(alone))
        )

        assert captured['result']['a1'].notna().sum() == 12
        marks = set(imputer.imputation_provenance_['a1'])
        assert len(marks) == 1 and marks.pop().value.startswith('model_on_')
        assert any('a1' in m and 'without any observation' in m for m in messages)

    def test_extended_window_predicts_it_with_its_history(self, reference_timeseries):
        """§7.3 — under ``'extended_forward'`` the history no longer hides the year."""
        imputer = fit_quietly(
            make_imputer(keep_lower_frequencies=False, imputation_scope='extended_forward'),
            reference_timeseries,
        )

        result = transform_quietly(imputer, self._unpublished_2023(reference_timeseries))

        assert result.loc['2023', 'a1'].notna().all()
        marks = set(imputer.imputation_provenance_.loc['2023', 'a1'])
        assert len(marks) == 1 and marks.pop().value.startswith('model_on_')


@pytest.mark.slow
class TestShortEntityCoverage:
    """I7 on the realistic panel — a short entity is never stretched past its coverage.

    ``heterogeneous_coverage_panel``: France and Italie run to 2024-07,
    Allemagne stops at 2024-04. The imputer is fitted on the data up to
    2023-12 and transforms the whole panel, so every 2024 row is new.
    """

    @pytest.fixture(scope='class', params=['interpolate', 'model'])
    def replay(self, request, _heterogeneous_coverage_panel_session):
        """Fit up to 2023-12, transform everything: ``(input, output, provenance, imputer)``."""
        data = _heterogeneous_coverage_panel_session.copy()
        dates = data.index.get_level_values('date')
        imputer = fit_quietly(
            make_imputer(covariate_strategy=request.param, keep_lower_frequencies=False),
            data[dates <= '2023-12-31'],
        )
        output = transform_quietly(imputer, data)
        return data, output, imputer.imputation_provenance_, imputer

    def test_short_entity_ends_where_its_input_ends(self, replay):
        """Allemagne's output stops at 2024-04-01, France's goes on to 2024-07-01."""
        data, output, _provenance, _imputer = replay
        last_dates = {
            entity: output.loc[entity].index.max() for entity in ('Allemagne', 'France')
        }
        assert last_dates == {
            'Allemagne': data.loc['Allemagne'].index.max(),
            'France': pd.Timestamp('2024-07-01'),
        }
        assert last_dates['Allemagne'] == pd.Timestamp('2024-04-01')

    def test_no_column_is_filled_past_its_coverage(self, replay):
        """No cell of Allemagne is filled after the period its column's last observation covers.

        A yearly value dated 2023-01-01 covers 2023 (period-start index); a
        quarterly one dated 2024-01-01 covers the first quarter of 2024.
        Beyond, a value would be an extrapolation no provenance announces.
        """
        data, output, _provenance, imputer = replay
        source = data.loc['Allemagne']
        filled_beyond = {}
        for column in source.columns:
            last = source[column].last_valid_index()
            detected = imputer.detected_frequencies_.get(('Allemagne', column))
            # Colonne jamais observée ou de fréquence indétectable : rien à couvrir
            if last is None or detected is None:
                continue
            frequency = normalize_frequency(detected, return_format='base')
            covered_until = pd.Period(last, freq=frequency).end_time
            beyond = output.loc['Allemagne', column].loc[lambda s: s.index > covered_until]
            filled_beyond[column] = int(beyond.notna().sum())
        # Toutes les colonnes observées sont contrôlées, aucune n'est débordée
        assert len(filled_beyond) == int(source.notna().any().sum())
        assert set(filled_beyond.values()) == {0}

    def test_every_filled_cell_of_the_short_entity_has_its_provenance(self, replay):
        """Each filled cell of Allemagne is marked, ``ORIGINAL`` only where the input observed it."""
        data, output, provenance, _imputer = replay
        values = output.loc['Allemagne']
        marks = provenance.loc['Allemagne'].reindex(index=values.index, columns=values.columns)
        observed = data.loc['Allemagne'].reindex(index=values.index, columns=values.columns).notna()

        # Aucune valeur sans provenance
        assert not (values.notna() & marks.isna()).any().any()
        # ORIGINAL réservé aux cellules observées en entrée
        is_original = marks.apply(lambda column: column == ProvenanceType.ORIGINAL)
        assert not (is_original & ~observed).any().any()


# =============================================================================
# Nom réservé du niveau de fréquence et lecture positionnelle (§5.9, §12.4)
# =============================================================================
class TestReservedFrequencyLevel:
    """The ``'frequency'`` level name is reserved, the level read by position."""

    @pytest.mark.parametrize(
        'names', [['frequency', 'date'], ['country', 'frequency']],
        ids=['entity_level', 'date_level'],
    )
    def test_fit_rejects_an_index_level_named_frequency(
        self, mixed_freq_panel_multifrequency, names
    ):
        """A stacked output would hold two levels of one name: ``fit`` refuses."""
        data = mixed_freq_panel_multifrequency.rename_axis(names)

        with pytest.raises(ValueError, match="'frequency' is reserved"):
            make_imputer().fit(data)

    def test_transform_rejects_an_index_level_named_frequency(
        self, mixed_freq_panel_multifrequency
    ):
        """Same guard at ``transform``, on data the fit never saw."""
        imputer = fit_quietly(make_imputer(), mixed_freq_panel_multifrequency)
        data = mixed_freq_panel_multifrequency.rename_axis(['frequency', 'date'])

        with pytest.raises(ValueError, match="'frequency' is reserved"):
            imputer.transform(data)

    def test_guard_does_not_depend_on_the_configuration(self, reference_timeseries):
        """``keep_lower_frequencies=False`` stacks nothing, yet the name stays reserved."""
        data = reference_timeseries.rename_axis('frequency')

        with pytest.raises(ValueError, match="'frequency' is reserved"):
            make_imputer(keep_lower_frequencies=False).fit(data)

    def test_inverse_reads_the_level_by_position(self, reference_timeseries):
        """Renamed output levels invert exactly like the original names."""
        imputer = make_imputer(
            impute_intermediate_frequencies=True, restore_original_values=True
        )
        transformed = fit_transform_quietly(imputer, reference_timeseries)
        reference = imputer.inverse_transform(transformed)

        # Renommage des niveaux de la sortie : une lecture par nom échouerait
        renamed = transformed.rename_axis(['pas', 'jour'])
        restored = imputer.inverse_transform(renamed)

        pd.testing.assert_frame_equal(restored, reference)
        assert restored.index.equals(reference_timeseries.index)

    def test_unstacked_output_is_left_as_is(self, reference_timeseries):
        """Under ``keep_lower_frequencies=False`` no level is ever dropped."""
        imputer = make_imputer(keep_lower_frequencies=False)
        transformed = fit_transform_quietly(imputer, reference_timeseries)

        restored = imputer.inverse_transform(transformed)

        assert restored.index.equals(transformed.index)


# =============================================================================
# Contrat d'entrée du transform : y, transformateur additif, entrée minimale
# =============================================================================
# Doublure : transformateur additif logarithmique dont l'inversion échoue
class _NonInvertibleLog(BaseEstimator, TransformerMixin):
    """Additive ``log1p`` transformer whose ``inverse_transform`` raises."""

    def fit(self, X, y=None):
        """Learn nothing.

        Args:
            X: Data.
            y: Ignored.

        Returns:
            self.
        """
        return self

    def transform(self, X):
        """Return ``log1p(X)``.

        Args:
            X: Data.

        Returns:
            The transformed frame.
        """
        return np.log1p(X)

    def inverse_transform(self, X):
        """Refuse to invert.

        Args:
            X: Data.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError('not invertible')


class TestTransformInputContract:
    """What ``transform`` accepts beside a frame: ``y``, an additive transformer, one row."""

    def test_series_is_refused(self, reference_timeseries):
        """A Series passes the base validation but is refused by the replay itself."""
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        with pytest.raises(ValueError, match='X must be a pandas DataFrame, got Series'):
            imputer.transform(reference_timeseries['m1'])

    def test_one_row_falls_back_on_every_row_with_one_warning(self, reference_timeseries):
        """One row carries no window: a single warning, every row used, the row kept."""
        imputer = fit_quietly(make_imputer(keep_lower_frequencies=False), reference_timeseries)
        one_row = reference_timeseries.iloc[[5]]
        captured = {}

        messages = _warning_messages(
            lambda: captured.setdefault('result', imputer.transform(one_row))
        )

        assert len([m for m in messages if 'Could not calculate the imputation window' in m]) == 1
        result = captured['result']
        assert result.index.equals(one_row.index)
        assert result['m1'].iloc[0] == one_row['m1'].iloc[0]

    def test_truncated_period_is_reported_and_restorable(self, reference_timeseries):
        """D46 — one month of a quarter cannot carry its total: warned once, restorable.

        The row 2021-06-30 holds ``q1 = 20``, the total of a quarter the
        call truncates to one month. It is not rescaled (D25): the month gets
        a third of it, and the quarterly total is no longer readable in the
        output — which one warning says. ``restore_original_values`` brings
        the observation back.
        """
        imputer = fit_quietly(
            make_imputer(keep_lower_frequencies=False, restore_original_values=True),
            reference_timeseries,
        )
        one_row = reference_timeseries.loc[['2021-06-30']]
        captured = {}

        messages = _warning_messages(
            lambda: captured.setdefault('result', imputer.transform(one_row))
        )

        truncated = [m for m in messages if m.startswith('Aggregation constraint not enforced')]
        assert len(truncated) == 1 and "['q1']" in truncated[0]
        assert captured['result']['q1'].iloc[0] == pytest.approx(20.0 / 3)
        restored = imputer.inverse_transform(captured['result'])
        assert restored['q1'].iloc[0] == 20.0

    def test_unnamed_target_comes_back_unnamed(self, reference_timeseries):
        """An unnamed ``y`` keeps no name through ``transform`` and ``inverse_transform``."""
        X = reference_timeseries[['m1', 'q1', 'a2']]
        y = reference_timeseries['a1'].rename(None)
        imputer = make_imputer(keep_lower_frequencies=False)

        X_out, y_out = _fit_transform_pair(imputer, X, y)
        _X_back, y_back = imputer.inverse_transform(X_out, y_out)

        assert y_out.name is None and y_back.name is None
        assert y_out.loc['2021'].sum() == pytest.approx(120.0)

    def test_transform_with_y_returns_the_named_pair(self, reference_timeseries):
        """B14 — ``(X, y)`` comes back as a pair, ``y`` under its own name, imputed like a column.

        Gold value of §2.2: the twelve months of ``a1`` in 2021 sum to 120.
        """
        X = reference_timeseries[['m1', 'q1', 'a2']]
        y = reference_timeseries['a1']
        imputer = make_imputer(keep_lower_frequencies=False)

        X_out, y_out = _fit_transform_pair(imputer, X, y)

        assert list(X_out.columns) == ['m1', 'q1', 'a2']
        assert y_out.name == 'a1'
        assert y_out.loc['2021'].sum() == pytest.approx(120.0)

    def test_plain_additive_transformer_is_applied_then_inverted(self, reference_timeseries):
        """An additive transformer returning a frame: imputed in its space, inverted back.

        The output lives in the additive space (``log1p``); the inversion
        applies ``expm1`` last, so the fully observed ``m1`` comes back exact.
        """
        imputer = make_imputer(
            additive_transformer=FunctionTransformer(np.log1p, np.expm1, check_inverse=False),
            keep_lower_frequencies=False,
        )
        transformed = fit_transform_quietly(imputer, reference_timeseries)

        np.testing.assert_allclose(
            transformed['m1'].to_numpy(), np.log1p(reference_timeseries['m1'].to_numpy())
        )
        restored = imputer.inverse_transform(transformed)
        np.testing.assert_allclose(
            restored['m1'].to_numpy(), reference_timeseries['m1'].to_numpy()
        )

    def test_failing_additive_inversion_warns_and_returns_the_additive_space(
        self, reference_timeseries
    ):
        """An additive inversion that raises is reported once; the data stay in its space."""
        imputer = make_imputer(
            additive_transformer=_NonInvertibleLog(), keep_lower_frequencies=False
        )
        transformed = fit_transform_quietly(imputer, reference_timeseries)
        captured = {}

        messages = _warning_messages(
            lambda: captured.setdefault('restored', imputer.inverse_transform(transformed))
        )

        assert len([m for m in messages if 'Failed to inverse transform' in m]) == 1
        np.testing.assert_allclose(
            captured['restored']['m1'].to_numpy(),
            np.log1p(reference_timeseries['m1'].to_numpy()),
        )
