"""Robustness of ``HighFrequencyImputer.fit`` to its inputs.

Covers the edge cases of ``CLAUDE.md`` on the imputer as a whole — unsorted
panel, entity of X absent from y and conversely, all-NaN column, all-NaN
entity, entity reduced to one row, single observation of the target, entirely
NaN target, no low-frequency column (empty plan), panel of a single entity —;
the invariance of the fit to the index perturbations of
``tests/support/perturbations.py`` on ``PANEL-X`` (reversed entities, special
column names, non-standard index names, three-level index; shuffled rows are
refused); and the start-of-period position, including the absence of an extra
row on an ``MS`` grid. The realistic datasets of notebook 3
are integration tests: ``tests/integration/frequency/test_hfi_on_realistic_dataset.py``.

The duplicated index and the unsorted time series are pinned in
``test_fit_phases.py::TestPhaseFiveEdgeCases``.
"""

# Modules de base
import warnings
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import pytest

# Objets testés
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer

# Outils de test partagés
from tests.support.datasets import build_panel_reference
from tests.support.perturbations import (
    drop_entity,
    reverse_entities,
    shuffle_rows,
    to_period_start,
    to_period_start_by_column,
    to_three_level_index,
    with_index_names,
    with_special_column_names,
)
from tests.unit.frequency.high_frequency_imputer.helpers import (
    FR,
    fit_quietly,
    fit_transform_quietly,
    make_imputer,
    replay,
    transform_quietly,
)


# Messages de tous les avertissements d'un appel
def _caught_messages(call: Callable[[], object]) -> List[str]:
    """Run a call and return every warning message it emitted.

    Args:
        call: Zero-argument callable.

    Returns:
        The messages, in emission order.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        call()
    return [str(item.message) for item in caught]


# =============================================================================
# Cas limites des entrées (CLAUDE.md)
# =============================================================================
class TestInputEdgeCases:
    """Cas limites de ``CLAUDE.md`` sur l'imputeur entier, un test chacun."""

    @pytest.mark.parametrize(
        'perturb',
        [
            shuffle_rows,
            lambda data: data.sort_index(level=[0, 1], ascending=[True, False]),
        ],
        ids=['shuffled_rows', 'dates_reversed_within_entities'],
    )
    def test_unsorted_panel_is_refused(self, panel_reference_full, perturb):
        """An unsorted panel is refused explicitly, never silently sorted.

        The time series counterpart is pinned by
        ``test_fit_phases.py::TestPhaseFiveEdgeCases::test_unsorted_index_is_refused_explicitly``.
        """
        with pytest.raises(ValueError, match='not contiguous|not sorted') as excinfo:
            fit_quietly(make_imputer(), perturb(panel_reference_full))

        # Le conseil ne propose pas auto_sort, que l'imputeur fige et n'expose pas
        assert 'auto_sort' not in str(excinfo.value)
        assert 'never sorts implicitly' in str(excinfo.value)

    @pytest.mark.parametrize(
        'drop_from, detail',
        [('y', 'lengths differ (108 vs 72)'), ('X', 'lengths differ (72 vs 108)')],
        ids=['entity_of_x_absent_from_y', 'entity_of_y_absent_from_x'],
    )
    def test_entity_missing_on_one_side_is_refused(
        self, panel_reference_full, drop_from, detail
    ):
        """Une entité présente d'un seul côté de (X, y) lève, sans réalignement (B14).

        Valeur d'or : 3 entités × 36 mois = 108 lignes, 72 sans IT.
        """
        features = panel_reference_full[['m1', 'q1', 'a2']]
        target = panel_reference_full['a1']
        if drop_from == 'y':
            target = drop_entity(target, 'IT')
        else:
            features = drop_entity(features, 'IT')

        with pytest.raises(ValueError, match='different indices') as excinfo:
            make_imputer().fit(features, target)

        assert detail in str(excinfo.value)

    def test_all_nan_column_on_a_panel_changes_nothing_else(
        self, mixed_freq_panel_multifrequency
    ):
        """An entirely NaN column is neither imputed, nor marked, nor influential.

        Excluded from the covariates (empty everywhere) and from the coverage
        denominator, it leaves the imputations of the other columns strictly
        identical to those of the same dataset without it.
        """
        data = mixed_freq_panel_multifrequency.copy()
        data['empty'] = np.nan

        imputer = fit_quietly(make_imputer(keep_lower_frequencies=False), data)
        values, provenance = replay(imputer, data)
        bare = fit_quietly(
            make_imputer(keep_lower_frequencies=False), mixed_freq_panel_multifrequency
        )
        bare_values, _bare_provenance = replay(bare, mixed_freq_panel_multifrequency)

        assert 'empty' not in {step.var_name for step in imputer.imputation_plan_}
        assert values['empty'].isna().all() and provenance['empty'].isna().all()
        pd.testing.assert_frame_equal(values.drop(columns='empty'), bare_values)

    def test_all_nan_entity_is_left_alone_and_costs_the_others_nothing(
        self, mixed_freq_panel_multifrequency
    ):
        """Une entité entièrement NaN reste vide et non marquée ; les autres sont inchangées.

        Elle n'observe aucune colonne : aucune fréquence, aucune fenêtre,
        aucune contribution au jeu mutualisé (R1). FR et DE reçoivent
        exactement les imputations d'un ajustement sans elle.
        """
        data = mixed_freq_panel_multifrequency.copy()
        data.loc[('IT',), :] = np.nan
        without = drop_entity(mixed_freq_panel_multifrequency, 'IT')

        imputer = fit_quietly(make_imputer(keep_lower_frequencies=False), data)
        values, provenance = replay(imputer, data)
        bare = fit_quietly(make_imputer(keep_lower_frequencies=False), without)
        bare_values, _bare_provenance = replay(bare, without)

        assert values.loc['IT'].isna().all().all()
        assert provenance.loc['IT'].isna().all().all()
        pd.testing.assert_frame_equal(values.drop(index='IT', level=0), bare_values)

    def test_all_nan_entity_is_not_said_absent_from_the_data(
        self, mixed_freq_panel_multifrequency
    ):
        """L'entité entièrement NaN fait partie des données : aucun message ne dit le contraire.

        ``target_frequency='M'`` est une chaîne : l'utilisateur n'a passé aucun
        dictionnaire, et IT est bien dans les données. L'avertissement de la
        fenêtre la nomme déjà (« no observation at all »).
        """
        data = mixed_freq_panel_multifrequency.copy()
        data.loc[('IT',), :] = np.nan

        messages = _caught_messages(lambda: make_imputer().fit(data))

        # Aucun message ne la dit absente ; celui de la
        # fenêtre la nomme, une fois
        assert [message for message in messages if 'not in data' in message] == []
        assert [message for message in messages if "[('IT',)]" in message] == [
            message for message in messages if 'without a determinable window' in message
        ]

    def test_all_nan_entity_is_named_once_at_transform(self, mixed_freq_panel_multifrequency):
        """Au transform, l'entité sans fenêtre est nommée une fois, jamais recomptée « hors fenêtre ».

        Le calculateur recalculé la nomme (« without a determinable window ») ;
        le message des lignes hors fenêtre recomptait en plus ses 36 lignes.
        """
        data = mixed_freq_panel_multifrequency.copy()
        data.loc[('IT',), :] = np.nan
        imputer = fit_quietly(make_imputer(), data)

        messages = _caught_messages(lambda: imputer.transform(data))

        assert [m for m in messages if 'fall outside' in m] == []
        assert len([m for m in messages if "[('IT',)]" in m]) == 1

    def test_entity_reduced_to_one_row_leaves_the_other_windows_intact(
        self, mixed_freq_panel_multifrequency
    ):
        """An entity of a single row does not cost the other entities their window.

        DE's m1 stops at the end of 2022: so does its imputation window. The entity
        LU, reduced to one row, has no detectable index frequency;
        it joins the entities without a window and leaves
        the masks of the other three intact, without making the computation fail
        for the whole panel.
        """
        base = mixed_freq_panel_multifrequency.copy()
        germany = base.index.get_level_values(0) == 'DE'
        base.loc[germany & (base.index.get_level_values(-1) >= '2023-01-01'), 'm1'] = np.nan
        single_row = pd.DataFrame(
            {'m1': [100.0], 'q1': [np.nan], 'v': [np.nan]},
            index=pd.MultiIndex.from_tuples(
                [('LU', pd.Timestamp('2021-01-31'))], names=base.index.names
            ),
        )
        data = pd.concat([base, single_row])

        with_single_row = fit_quietly(make_imputer(), data)
        without = fit_quietly(make_imputer(), base)

        pd.testing.assert_series_equal(
            with_single_row.imputation_window_mask_.drop(index='LU', level=0),
            without.imputation_window_mask_,
            check_names=False,
        )

    def test_single_observation_of_the_target_is_left_as_is(self, reference_timeseries):
        """A target observed only once is not imputed, and makes nothing fail.

        Its frequency is undetectable (fewer than two observations): the
        pair joins the undetected pairs, no stage
        imputes it, and its single value comes out as is.
        """
        features = reference_timeseries[['m1', 'q1', 'a2']]
        target = reference_timeseries['a1'].where(
            reference_timeseries.index == '2023-12-31'
        )
        imputer = make_imputer(keep_lower_frequencies=False)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(features, target)
            _features_out, target_out = imputer.transform(features, target)

        assert 'a1' not in imputer.detected_frequencies_
        assert 'a1' not in {step.var_name for step in imputer.imputation_plan_}
        pd.testing.assert_series_equal(
            target_out, target, check_names=False, check_freq=False
        )

    def test_undetectable_column_is_not_used_as_a_covariate(self, reference_timeseries):
        """An unusable pair does not make the fit of the others fail.

        The target a1, observed only once, has no frequency: it can be neither
        aggregated nor interpolated. It is not a candidate covariate for q1 and
        a2: it would bring only NaN to ``X_train`` and the estimator, which does not
        tolerate them, would fail, q1 and a2 falling back.
        """
        features = reference_timeseries[['m1', 'q1', 'a2']]
        target = reference_timeseries['a1'].where(
            reference_timeseries.index == '2023-12-31'
        )
        imputer = make_imputer()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(features, target)

        assert not any(step.is_fallback for step in imputer.imputation_plan_)
        assert all('a1' not in step.feature_cols for step in imputer.imputation_plan_)

    def test_entirely_nan_target_changes_nothing_else(self, reference_timeseries):
        """An entirely NaN target stays NaN; the imputations of X are those of a fit without y.

        Structurally absent column, empty on all
        grids, it enters neither the window nor the covariates.
        """
        features = reference_timeseries[['m1', 'q1', 'a2']]
        target = reference_timeseries['a1'] * np.nan
        imputer = make_imputer(keep_lower_frequencies=False)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(features, target)
            features_out, target_out = imputer.transform(features, target)
        bare = fit_quietly(make_imputer(keep_lower_frequencies=False), features)

        assert 'a1' not in {step.var_name for step in imputer.imputation_plan_}
        assert target_out.isna().all()
        pd.testing.assert_frame_equal(features_out, transform_quietly(bare, features))

    def test_no_low_frequency_column_gives_an_empty_plan(self, reference_timeseries):
        """Aucune colonne de basse fréquence : plan vide, sortie égale à l'entrée, en silence.

        Une seule colonne suffit à définir une fenêtre — sa propre
        couverture : le calculateur n'exige plus deux colonnes, et le fit
        n'émet plus « Could not calculate imputation window ». Rien n'étant
        imputable, la sortie reprend l'entrée.
        """
        data = reference_timeseries[['m1']]
        imputer = make_imputer(keep_lower_frequencies=False)

        messages = _caught_messages(lambda: imputer.fit(data))
        values = transform_quietly(imputer, data)

        assert len(imputer.imputation_plan_) == 0
        assert imputer.frequency_progression_ == ['M']
        assert messages == []
        assert imputer.imputation_window_ == (
            pd.Timestamp('2021-01-31'), pd.Timestamp('2023-12-31')
        )
        pd.testing.assert_frame_equal(values, data, check_freq=False)

    def test_single_entity_panel_matches_the_time_series(
        self, panel_reference_full, reference_timeseries
    ):
        """A single-entity panel gives exactly the imputations of the series.

        The time series is the one-entity degenerate case: the pooled
        set reduces to the single block, values and provenances bit for
        bit.
        """
        panel = panel_reference_full.loc[['FR'], ['m1', 'q1', 'a1', 'a2']]

        imputer = fit_quietly(make_imputer(keep_lower_frequencies=False), panel)
        values, provenance = replay(imputer, panel)
        series = fit_quietly(make_imputer(keep_lower_frequencies=False), reference_timeseries)
        series_values, series_provenance = replay(series, reference_timeseries)

        assert imputer.is_panel_ is True and imputer.entities_ == [FR]
        pd.testing.assert_frame_equal(values.xs('FR'), series_values, check_freq=False)
        pd.testing.assert_frame_equal(
            provenance.xs('FR'), series_provenance, check_freq=False
        )


# =============================================================================
# Invariance aux perturbations d'index (PANEL-X)
# =============================================================================
# Réglage commun : covariables éligibles sur toutes les entités (climat_affaires,
# absente pour IT, est écartée et tous les modèles s'ajustent), sortie à plat
# pour que chaque perturbation s'applique telle quelle à la sortie
_INVARIANCE_PARAMS = dict(covariate_eligibility='all_entities', keep_lower_frequencies=False)

# Préfixe d'entité rendu par "to_diagnostic_frame" sous to_three_level_index
_REGION_PREFIX = 'Zone euro|'


# Perturbations de PANEL-X, et inversion de leur renommage
def _special_names(data: pd.DataFrame) -> pd.DataFrame:
    """Rename every column with special characters, dropping the mapping."""
    return with_special_column_names(data)[0]


_PERTURBATIONS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    'reverse_entities': reverse_entities,
    'index_names': lambda data: with_index_names(data, ['pays', 'periode']),
    'three_level_index': to_three_level_index,
    'special_column_names': _special_names,
}


# Renommage inverse d'une trame de diagnostic
def _canonical_diagnostic(diagnostic: pd.DataFrame, names: Dict[str, str]) -> pd.DataFrame:
    """Undo the renamings a perturbation brings into a diagnostic frame.

    Args:
        diagnostic: Output of ``imputation_plan_.to_diagnostic_frame()``.
        names: Mapping from perturbed column name to original name.

    Returns:
        The frame with original column names in ``variable`` and
        ``materialization``, and the region prefix dropped from the entity
        columns.
    """
    out = diagnostic.copy()
    for column in ('entities', 'training_blocks'):
        out[column] = out[column].str.replace(_REGION_PREFIX, '', regex=False)
    out['variable'] = out['variable'].map(lambda name: names.get(name, name))

    def _rename_ways(text: str) -> str:
        # Sérialisation "colonne=voie, colonne=voie" : les noms spéciaux ne
        # portent ni virgule ni signe égal
        if not text:
            return text
        pairs = (item.split('=', 1) for item in text.split(', '))
        return ', '.join(f"{names.get(name, name)}={way}" for name, way in pairs)

    out['materialization'] = out['materialization'].map(_rename_ways)
    return out


@pytest.fixture(scope='module')
def panel_x_fits() -> Callable[[str], tuple]:
    """Fit ``PANEL-X`` once per perturbation, lazily, for the whole module.

    Returns:
        Callable ``name -> (values, provenance, diagnostic, names)``; the name
        ``'original'`` gives the unperturbed fit, ``names`` maps perturbed
        column names back to the original ones.
    """
    data = build_panel_reference()
    cache: Dict[str, tuple] = {}

    def _get(name: str) -> tuple:
        if name not in cache:
            perturbed = data if name == 'original' else _PERTURBATIONS[name](data)
            imputer = make_imputer(**_INVARIANCE_PARAMS)
            values = fit_transform_quietly(imputer, perturbed)
            names = dict(zip(perturbed.columns, data.columns))
            cache[name] = (
                values,
                imputer.imputation_provenance_.copy(),
                imputer.imputation_plan_.to_diagnostic_frame(),
                names,
            )
        return cache[name]

    return _get


@pytest.mark.slow
class TestPerturbationInvariance:
    """fit_transform(perturbé) == perturbation(fit_transform(original)), sur PANEL-X."""

    def test_shuffled_rows_are_refused(self, panel_reference_full):
        """Shuffled rows are not invariant: they are refused.

        Invariance to shuffle_rows would assume an implicit sort, which the
        class forbids itself: sorting would move windows and aggregates without
        the caller knowing.
        """
        with pytest.raises(ValueError, match='not contiguous'):
            fit_quietly(make_imputer(**_INVARIANCE_PARAMS), shuffle_rows(panel_reference_full))

    @pytest.mark.parametrize('name', list(_PERTURBATIONS))
    def test_values_follow_the_perturbation(self, panel_x_fits, name):
        """Les valeurs imputées sont celles de l'original, perturbées de la même façon.

        Au réordonnancement près : sous reverse_entities, les lignes du jeu
        mutualisé changent d'ordre et l'ajustement par moindres carrés ne
        coïncide qu'à l'arrondi (3e-14 mesuré), d'où ``rtol=1e-9``.
        """
        values, _provenance, _diagnostic, _names = panel_x_fits(name)
        original, _p, _d, _n = panel_x_fits('original')

        pd.testing.assert_frame_equal(
            values.sort_index(),
            _PERTURBATIONS[name](original).sort_index(),
            rtol=1e-9,
        )

    @pytest.mark.parametrize('name', list(_PERTURBATIONS))
    def test_provenance_follows_the_perturbation(self, panel_x_fits, name):
        """La matrice de provenance est celle de l'original, perturbée de la même façon."""
        _values, provenance, _diagnostic, _names = panel_x_fits(name)
        _v, original, _d, _n = panel_x_fits('original')

        pd.testing.assert_frame_equal(
            provenance.sort_index(), _PERTURBATIONS[name](original).sort_index()
        )

    @pytest.mark.parametrize('name', list(_PERTURBATIONS))
    def test_diagnostic_frame_is_unchanged_up_to_renaming(self, panel_x_fits, name):
        """The diagnostic of the plan is identical, up to the renaming of columns and entities.

        Under reverse_entities, ``training_blocks`` is sorted by entity key,
        not by input order of the entities.
        """
        _values, _provenance, diagnostic, names = panel_x_fits(name)
        _v, _p, original, _n = panel_x_fits('original')

        pd.testing.assert_frame_equal(_canonical_diagnostic(diagnostic, names), original)


# =============================================================================
# Position des dates dans la période
# =============================================================================
# Décalage d'un index fin de mois au début du même mois
def _to_month_start(frame: pd.DataFrame) -> pd.DataFrame:
    """Move every date of a month-end frame to the first day of its month.

    Args:
        frame: Time series or panel frame, month-end anchored.

    Returns:
        A copy whose date level is month-start anchored.
    """
    out = frame.copy()
    if isinstance(out.index, pd.MultiIndex):
        dates = out.index.get_level_values(-1).to_period('M').start_time
        out.index = pd.MultiIndex.from_arrays(
            [out.index.get_level_values(i) for i in range(out.index.nlevels - 1)] + [dates],
            names=out.index.names,
        )
    else:
        out.index = pd.DatetimeIndex(
            out.index.to_period('M').start_time, name=out.index.name
        )
    return out


class TestPeriodPositions:
    """Les positions début (MS / QS / YS) et fin (ME / QE / YE) donnent les mêmes imputations."""

    @pytest.mark.parametrize(
        'columns',
        [['m1', 'a1'], ['m1', 'q1'], ['m1', 'v']],
        ids=['yearly', 'quarterly', 'per_entity_frequency'],
    )
    def test_start_anchored_panel_gives_the_same_imputations(
        self, panel_reference_full, columns
    ):
        """PANEL-X anchored at period start: same imputations, up to the date shift.

        Each observation is redated to the start of ITS period (January 1st,
        first day of the quarter, 1st of the month). The only covariate, m1, is
        read as is on the monthly grid and summed over the periods
        of the anchors: these sums do not depend on the position, nor therefore do
        the model or the rescaled predictions. An INTERPOLATED covariate, however,
        depends on the position by construction under ``interpolation_anchor=None``
        (the value applies at the date where it is referenced).
        """
        end_anchored = panel_reference_full[columns]
        start_anchored = to_period_start_by_column(end_anchored)

        end = fit_quietly(make_imputer(keep_lower_frequencies=False), end_anchored)
        end_values, end_provenance = replay(end, end_anchored)
        start = fit_quietly(make_imputer(keep_lower_frequencies=False), start_anchored)
        start_values, start_provenance = replay(start, start_anchored)

        pd.testing.assert_frame_equal(
            start_values, _to_month_start(end_values), rtol=1e-9
        )
        pd.testing.assert_frame_equal(start_provenance, _to_month_start(end_provenance))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        'modality', [False, True], ids=['single_stage', 'intermediate_stages']
    )
    def test_start_anchored_grid_holds_one_row_per_period(
        self, panel_reference_full, modality
    ):
        """Grille cible ``MS`` : 36 mois et 12 trimestres par entité, pas une ligne de plus.

        Constat de la campagne HFI2 (« sur un index MS, la grille cible porte
        une ligne de plus par entité ») : NON reproduit sur PANEL-X ancré en
        début de période, ni au niveau mensuel ni au niveau trimestriel de la
        sortie multi-fréquences. Épinglé ici.
        """
        start_anchored = to_period_start_by_column(
            panel_reference_full[['m1', 'q1', 'a1', 'a2', 'v']]
        )
        imputer = make_imputer(impute_intermediate_frequencies=modality)

        values = fit_transform_quietly(imputer, start_anchored)

        counts = values.groupby(level=['country', 'frequency']).size().to_dict()
        expected = {(entity, 'M'): 36 for entity in ('FR', 'DE', 'IT')}
        if modality:
            expected.update({(entity, 'Q'): 12 for entity in ('FR', 'DE', 'IT')})
        assert counts == expected

    def test_native_month_start_dataset_keeps_its_rows(self, mixed_freq_timeseries):
        """Jeu natif ``MS`` du notebook 2 : la sortie porte exactement l'index d'entrée."""
        values = fit_transform_quietly(
            make_imputer(keep_lower_frequencies=False), mixed_freq_timeseries
        )

        pd.testing.assert_index_equal(
            values.index, mixed_freq_timeseries.index, exact=False
        )

    @pytest.mark.parametrize(
        'build',
        [
            lambda ts: to_period_start(ts[['m1', 'q1', 'a1']]),
            lambda ts: ts[['m1', 'q1']].assign(
                a1=pd.Series(
                    [120.0, 132.0, 150.0],
                    index=pd.to_datetime(['2021-06-30', '2022-06-30', '2023-06-30']),
                ).reindex(ts.index)
            ),
        ],
        ids=['yearly_anchor_on_december_first', 'fiscal_year_ending_in_june'],
    )
    def test_off_calendar_anchors_are_left_out_with_one_warning(
        self, reference_timeseries, build
    ):
        """Anchors off the calendar bounds of their frequency: pair dropped, one warning.

        ``to_period_start`` applied as is dates the annual anchor on December 1st
        (and the quarterly anchor on the first day of the last month of the
        quarter); a fiscal year ending end of June dates it on June 30th. No
        calendar reading of them is coherent: the training mask would discard all
        the anchors, the rescaling would attach them to the calendar year, and
        the column would fall back on a partial fallback. The pair is dropped —
        neither imputed nor covariate —, its observations come out as is, and a
        single warning names it.
        """
        data = build(reference_timeseries)
        imputer = make_imputer(keep_lower_frequencies=False)

        messages = _caught_messages(lambda: imputer.fit(data))
        values = transform_quietly(imputer, data)

        misaligned = [m for m in messages if 'off the calendar bounds' in m]
        assert len(misaligned) == 1 and "'a1'" in misaligned[0]
        assert 'a1' not in {step.var_name for step in imputer.imputation_plan_}
        assert all('a1' not in step.feature_cols for step in imputer.imputation_plan_)
        pd.testing.assert_series_equal(values['a1'], data['a1'], check_freq=False)

    def test_off_calendar_quarter_anchors_of_one_entity_only(self, mixed_freq_panel_multifrequency):
        """Sur un panel, seul le couple (entité, colonne) hors bornes est écarté.

        Les ancres trimestrielles de v pour DE sont décalées d'un mois (fin
        février, mai, août, novembre) : (DE, v) est écarté, FR garde son
        imputation de v, et IT — mensuelle, déjà à la cible — n'est pas
        concernée.
        """
        data = mixed_freq_panel_multifrequency.copy()
        germany = data.loc['DE', 'v'].dropna()
        shifted = pd.Series(
            germany.to_numpy(), index=germany.index - pd.offsets.MonthEnd(1)
        )
        data.loc[('DE',), 'v'] = shifted.reindex(data.loc['DE'].index).to_numpy()

        imputer = make_imputer()
        messages = _caught_messages(lambda: imputer.fit(data))

        assert [m for m in messages if 'off the calendar bounds' in m and "('DE', 'v')" in m]
        assert {(s.var_name, s.entities) for s in imputer.imputation_plan_ if s.var_name == 'v'} == {
            ('v', (FR,)),
        }
