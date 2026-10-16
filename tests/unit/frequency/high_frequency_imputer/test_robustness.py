"""Robustness of ``HighFrequencyImputer.fit`` to its inputs (prompt F11).

Covers the edge cases of ``CLAUDE.md`` on the imputer as a whole — unsorted
panel, entity of X absent from y and conversely, all-NaN column, all-NaN
entity, entity reduced to one row, single observation of the target, entirely
NaN target, no low-frequency column (empty plan), panel of a single entity —;
the invariance of the fit to the index perturbations of
``tests/support/perturbations.py`` on ``PANEL-X`` (reversed entities, special
column names, non-standard index names, three-level index; shuffled rows are
refused, §12.5); and the start-of-period position (§10.2), including the extra
row an ``MS`` grid was suspected to carry. The realistic datasets of notebook 3
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
        """Un panel non trié est refusé explicitement, jamais trié en silence (§12.5).

        Le pendant série temporelle est épinglé par
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
        """Une colonne entièrement NaN n'est ni imputée, ni marquée, ni influente.

        Exclue des covariables (vide partout) et du dénominateur de couverture
        (D39), elle laisse les imputations des autres colonnes rigoureusement
        identiques à celles du même jeu sans elle.
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

        # ANO-FREQ-062 : aucun message ne la dit absente ; celui de la
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
        """Une entité d'une seule ligne ne coûte pas leur fenêtre aux autres entités.

        m1 de DE s'arrête fin 2022 : sa fenêtre d'imputation aussi. L'entité
        LU, réduite à une ligne, n'a pas de fréquence d'index détectable ;
        elle rejoint les entités sans fenêtre (ANO-FREQ-020) et laisse
        intacts les masques des trois autres — elle faisait échouer le calcul
        pour tout le panel (ANO-FREQ-061).
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
        """Une cible observée une seule fois n'est pas imputée, et ne fait rien échouer.

        Sa fréquence est indétectable (moins de deux observations) : le
        couple rejoint les couples non détectés (§12.5), aucune étape ne
        l'impute, et sa valeur unique ressort telle quelle.
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
        """Un couple inutilisable ne fait pas échouer l'ajustement des autres (§12.5).

        La cible a1, observée une seule fois, n'a pas de fréquence : elle ne
        peut être ni agrégée ni interpolée. Retenue comme covariable de q1 et
        de a2, elle n'apportait que des NaN à ``X_train`` et l'estimateur, qui
        ne les tolère pas, échouait : q1 et a2 retombaient en repli
        (ANO-FREQ-060). Elle n'est plus candidate.
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
        """Une cible entièrement NaN reste NaN ; les imputations de X sont celles d'un fit sans y.

        Colonne structurellement absente (D39) et vide sur toutes les
        grilles, elle n'entre ni dans la fenêtre, ni dans les covariables.
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
        """Un panel d'une seule entité donne exactement les imputations de la série (§5.8, I16).

        La série temporelle est le cas dégénéré à une entité : le jeu
        mutualisé se réduit au bloc unique, valeurs et provenances au bit
        près.
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
        """Des lignes mélangées ne sont pas invariantes : elles sont refusées (§12.5).

        L'invariance à shuffle_rows supposerait un tri implicite, que la
        classe s'interdit : le tri déplacerait fenêtres et agrégats sans que
        l'appelant le sache.
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
        """Le diagnostic du plan est identique, au renommage des colonnes et entités près.

        Sous reverse_entities, ``training_blocks`` suivait l'ordre d'entrée des
        entités (ANO-FREQ-067) ; il est désormais trié par clé d'entité.
        """
        _values, _provenance, diagnostic, names = panel_x_fits(name)
        _v, _p, original, _n = panel_x_fits('original')

        pd.testing.assert_frame_equal(_canonical_diagnostic(diagnostic, names), original)


# =============================================================================
# Position des dates dans la période (§10.2)
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
        """PANEL-X ancré en début de période : mêmes imputations, au décalage de date près.

        Chaque observation est redatée au début de SA période (1er janvier,
        premier jour du trimestre, 1er du mois). La seule covariable, m1, est
        lue telle quelle sur la grille mensuelle et sommée sur les périodes
        des ancres : ces sommes ne dépendent pas de la position, ni donc le
        modèle, ni les prédictions recalées. Une covariable INTERPOLÉE, elle,
        dépend de la position par construction sous ``interpolation_anchor=None``
        (§10.2 : la valeur s'applique à la date où elle est référencée).
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
        """Ancres hors des bornes calendaires de leur fréquence : couple écarté, un avertissement.

        ANO-FREQ-065. ``to_period_start`` appliqué tel quel date l'ancre
        annuelle au 1er décembre (et l'ancre trimestrielle au premier jour du
        dernier mois du trimestre) ; un exercice clos fin juin la date au 30
        juin. Aucune lecture calendaire n'en est cohérente : le masque
        d'entraînement en écartait toutes les ancres, le recalage les
        rattachait à l'année civile, et la colonne retombait sur un repli
        partiel. Le couple est désormais écarté — ni imputé, ni covariable —,
        ses observations ressortent telles quelles, et un avertissement
        unique le nomme.
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
