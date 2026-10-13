"""``HighFrequencyImputer.transform`` and its multi-frequency output (§12.4).

Covers the transform outside the fit window (I7), the extended window on
an irregular index, the frequency control (D11), ``keep_lower_frequencies``
(D35), the transform state (B19), the shared-model replay (§5.8 R6), new
entities (D34) and the reserved ``'frequency'`` level name.
"""

# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest

# Objets testés
from tsforecast.frequency.provenance import ProvenanceType

# Outils de test partagés
from tests.support.estimators import SpyEstimator
from tests.unit.frequency.high_frequency_imputer.helpers import (
    fit_quietly,
    fit_transform_quietly,
    fit_with_spy,
    has_frequency_level_in_output,
    make_imputer,
    transform_quietly,
)


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

    # Interne : aucun effet public prouvé de la restauration des deux
    # collaborateurs ; le pendant public (« le transform ne mute pas l'état
    # ajusté ») relève du prompt F12
    @pytest.mark.internal
    def test_fit_state_is_restored_after_a_transform(self, reference_timeseries):
        """Le rejeu rend au fit son calculateur et son traceur de provenance."""
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        calculator = imputer._imputation_window_calc
        tracker = imputer._provenance_tracker
        transform_quietly(imputer, reference_timeseries)
        assert imputer._imputation_window_calc is calculator
        assert imputer._provenance_tracker is tracker

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
