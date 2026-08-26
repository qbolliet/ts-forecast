"""Tests d'intégration de non-régression pour HighFrequencyImputer.

Ce module exécute `fit_transform` sur les jeux de données de référence du
notebook `notebooks/2 - QB - Mixed frequencies.ipynb` (fixtures
`mixed_freq_timeseries` / `mixed_freq_panel`) pour toutes les combinaisons de
`cascade_refitting` x `keep_lower_frequencies` x {séries temporelles, panel},
et épingle des caractéristiques observables du résultat (nombre de lignes,
colonnes 100% NaN, clés de `imputation_models_`, niveaux de fréquence de
sortie).

Ce script est le filet de sécurité qui précède toute correction de
`tsforecast/frequency/high_frequency_imputer.py` : voir
`high_frequency_imputer_review.md` (§7) pour le diagnostic complet. Les tests
marqués ``xfail(strict=True)`` documentent le comportement **souhaité** (pas
le comportement actuel bogué) et référencent la section de la revue
concernée ; ils doivent tomber un par un au fil des correctifs.

Correctifs déjà appliqués et couverts sans ``xfail`` : §2.1 (facteur
d'échelle), §2.2 / §5.1 (frames d'étape), §2.3 / §5.2 (registre de modèles
indexé par étape), §2.4 (déduplication des fits panel, voir
`test_high_frequency_imputer.py::TestPanelSingleFitPerVariable`), §2.5
(sortie multi-fréquences : niveaux reconstruits après imputation, plus de
label 'target' dupliqué) et §2.6 (désagrégation des valeurs d'ancre et
recalage additif, voir
`test_high_frequency_imputer.py::TestPeriodTotalsEnforced`).

En plus de la matrice de base (`ALL_SCENARIOS`), `FULL_SCENARIOS` ajoute la
dimension `imputation_scope` (`extended_backward`/`extended_forward`/
`extended_both`, en plus de `strict`) sur la configuration la plus complète
(`cascade_refitting=True, keep_lower_frequencies=True`) pour les deux types
de données. Ces scénarios sont vérifiés sur trois axes supplémentaires :
forme de la sortie (`TestScopeScenarioShape`), absence de colonnes
intégralement NaN (`TestScopeScenarioNoFullyNanColumns`) et cohérence
additive des sous-périodes (`TestScopeScenarioAdditiveCoherence`, §2.6).
`TestFitTransformSymmetry` vérifie que `fit_transform(X)` sur une instance
équivaut à `fit(X)` puis `transform(X)` sur une instance séparée.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer


# ---------------------------------------------------------------------------
# Construction des scénarios et exécution de fit_transform
# ---------------------------------------------------------------------------
# (dataset_fixture_name, cascade_refitting, keep_lower_frequencies)
ALL_SCENARIOS = [
    ('mixed_freq_timeseries', False, False),
    ('mixed_freq_timeseries', False, True),
    ('mixed_freq_timeseries', True, False),
    ('mixed_freq_timeseries', True, True),
    ('mixed_freq_panel', False, False),
    ('mixed_freq_panel', False, True),
    ('mixed_freq_panel', True, False),
    ('mixed_freq_panel', True, True),
]


def _scenario_id(param: tuple) -> str:
    dataset_name, cascade_refitting, keep_lower_frequencies = param
    kind = 'ts' if dataset_name == 'mixed_freq_timeseries' else 'panel'
    return f"{kind}-cascade={cascade_refitting}-keep_lower={keep_lower_frequencies}"


def _run_hfi(
    data: pd.DataFrame,
    cascade_refitting: bool,
    keep_lower_frequencies: bool,
    imputation_scope: str = 'strict',
):
    """Fit_transform avec un régresseur tolérant les NaN, target_frequency='M'.

    Depuis le retrait de `feature_means`, l'imputer ne complète plus les
    covariables manquantes : sur les jeux de référence multi-fréquences, une
    `LinearRegression` nue lève au `predict` et bascule l'étape en repli par
    interpolation, qui ne recale pas les totaux de période. Le harnais fournit
    donc un `Pipeline` conforme au contrat NaN documenté sur `estimator`.
    """
    imputer = HighFrequencyImputer(
        target_frequency='M',
        estimator=make_pipeline(SimpleImputer(), LinearRegression()),
        cascade_refitting=cascade_refitting,
        keep_lower_frequencies=keep_lower_frequencies,
        imputation_scope=imputation_scope,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = imputer.fit_transform(data.copy())
    return imputer, result


def _target_level_frame(
    imputer: HighFrequencyImputer,
    result: pd.DataFrame,
    keep_lower_frequencies: bool,
) -> pd.DataFrame:
    """Extract the target-frequency level from a (possibly multi-freq) result.

    Since §2.5's fix, the target level keeps its real frequency label
    (e.g. 'M') instead of the generic 'target' one, so it must be looked
    up via `effective_target_frequency_` rather than a hardcoded label.
    """
    if not keep_lower_frequencies:
        return result
    target_label = imputer._stage_frequency_label(imputer.effective_target_frequency_)
    return result.xs(target_label, level='frequency')


def _index_as_set(index: pd.Index) -> set:
    """Compare indices as sets of tuples/timestamps, ignoring order and names."""
    return set(index.tolist())


@pytest.fixture(params=ALL_SCENARIOS, ids=_scenario_id)
def hfi_scenario(request):
    """Fit_transform result for one (dataset, cascade_refitting, keep_lower_frequencies)."""
    dataset_name, cascade_refitting, keep_lower_frequencies = request.param
    data = request.getfixturevalue(dataset_name)
    imputer, result = _run_hfi(data, cascade_refitting, keep_lower_frequencies)
    return {
        'param': request.param,
        'data': data,
        'imputer': imputer,
        'result': result,
        'cascade_refitting': cascade_refitting,
        'keep_lower_frequencies': keep_lower_frequencies,
    }


KEEP_LOWER_SCENARIOS = [p for p in ALL_SCENARIOS if p[2] is True]


class TestRowCountAndCoverage:
    """§2.2 : l'agrégation en cascade ne doit ni dupliquer ni supprimer de lignes."""

    @pytest.mark.parametrize('param', ALL_SCENARIOS, ids=_scenario_id)
    def test_target_level_matches_source_index(self, request, param):
        """Le niveau cible de la sortie couvre exactement l'index des données source."""
        dataset_name, cascade_refitting, keep_lower_frequencies = param
        data = request.getfixturevalue(dataset_name)
        imputer, result = _run_hfi(data, cascade_refitting, keep_lower_frequencies)

        target_frame = _target_level_frame(imputer, result, keep_lower_frequencies)

        assert _index_as_set(target_frame.index) == _index_as_set(data.index)

    @pytest.mark.parametrize('param', ALL_SCENARIOS, ids=_scenario_id)
    def test_target_level_has_no_fully_nan_column(self, request, param):
        """Aucune colonne du niveau cible n'est intégralement NaN."""
        dataset_name, cascade_refitting, keep_lower_frequencies = param
        data = request.getfixturevalue(dataset_name)
        imputer, result = _run_hfi(data, cascade_refitting, keep_lower_frequencies)

        target_frame = _target_level_frame(imputer, result, keep_lower_frequencies)
        fully_nan_cols = [c for c in target_frame.columns if target_frame[c].isna().all()]

        assert fully_nan_cols == []


class TestModelRegistry:
    """§2.3 : chaque étape de fit doit obtenir sa propre entrée de registre."""

    @pytest.mark.parametrize('param', ALL_SCENARIOS, ids=_scenario_id)
    def test_one_registry_entry_per_fitting_order_entry(self, request, param):
        """imputation_models_ doit contenir autant d'entrées que model_fitting_order_."""
        dataset_name, cascade_refitting, keep_lower_frequencies = param
        data = request.getfixturevalue(dataset_name)
        imputer, _ = _run_hfi(data, cascade_refitting, keep_lower_frequencies)

        assert len(imputer.imputation_models_) == len(imputer.model_fitting_order_)


class TestMultiFrequencyOutput:
    """§2.5 : la sortie multi-fréquences ne doit pas dupliquer le dernier niveau."""

    @pytest.mark.parametrize('param', KEEP_LOWER_SCENARIOS, ids=_scenario_id)
    def test_no_duplicate_target_label(self, request, param):
        """Le niveau 'target' ne doit pas coexister avec le label de fréquence réel."""
        dataset_name, cascade_refitting, keep_lower_frequencies = param
        data = request.getfixturevalue(dataset_name)
        imputer, result = _run_hfi(data, cascade_refitting, keep_lower_frequencies)

        levels = result.index.get_level_values('frequency').unique().tolist()

        assert 'target' not in levels

    @pytest.mark.parametrize('param', KEEP_LOWER_SCENARIOS, ids=_scenario_id)
    def test_annual_level_absent_when_nothing_to_impute_at_annual_stage(self, request, param):
        """§4.2 : aucune étape n'est gaspillée sur une fréquence sans variable à imputer.

        Dans nos jeux de données, `balance_commerciale_annuelle` est la seule
        variable annuelle : il n'existe aucune variable de fréquence
        strictement inférieure à imputer à l'étape 'Y', qui ne doit donc
        laisser aucune trace dans les niveaux de sortie.
        """
        dataset_name, cascade_refitting, keep_lower_frequencies = param
        data = request.getfixturevalue(dataset_name)
        imputer, result = _run_hfi(data, cascade_refitting, keep_lower_frequencies)

        levels = result.index.get_level_values('frequency').unique().tolist()

        assert 'Y' not in levels


# ---------------------------------------------------------------------------
# Scénarios A à I : ajout de la dimension `imputation_scope`
# ---------------------------------------------------------------------------
# `ALL_SCENARIOS` ci-dessus couvre déjà séries temporelles x panel x
# cascade_refitting x keep_lower_frequencies à imputation_scope='strict'
# (scénarios A à D du notebook, cf. section 4 de
# `notebooks/2 - QB - Mixed frequencies.ipynb`, doublés pour le panel). On
# ajoute ici les trois valeurs restantes d'`imputation_scope` sur la
# configuration la plus complète (cascade_refitting=True,
# keep_lower_frequencies=True), pour les deux types de données, ce qui
# couvre les scénarios E à J en pratique (le nom "A à I" de la revue ne
# correspond à aucune énumération explicite du notebook, qui n'en nomme que
# quatre : on interprète ici la consigne comme le produit cartésien complet
# des dimensions listées, pas une liste figée de neuf cas).
SCOPE_SCENARIOS = [
    (dataset_name, True, True, imputation_scope)
    for dataset_name in ('mixed_freq_timeseries', 'mixed_freq_panel')
    for imputation_scope in ('extended_backward', 'extended_forward', 'extended_both')
]

# (dataset_fixture_name, cascade_refitting, keep_lower_frequencies, imputation_scope)
FULL_SCENARIOS = [
    (dataset_name, cascade_refitting, keep_lower_frequencies, 'strict')
    for dataset_name, cascade_refitting, keep_lower_frequencies in ALL_SCENARIOS
] + SCOPE_SCENARIOS


def _full_scenario_id(param: tuple) -> str:
    dataset_name, cascade_refitting, keep_lower_frequencies, imputation_scope = param
    kind = 'ts' if dataset_name == 'mixed_freq_timeseries' else 'panel'
    return (
        f"{kind}-cascade={cascade_refitting}-keep_lower={keep_lower_frequencies}"
        f"-scope={imputation_scope}"
    )


def _object_array(values: list) -> np.ndarray:
    """1-D object array holding `values` as opaque elements (never 2-D).

    ``np.array(list_of_equal_length_tuples, dtype=object)`` silently
    broadcasts to a 2-D array instead of an array of tuple objects, which
    then crashes pandas' groupby hashing (``Buffer has wrong number of
    dimensions``). Pre-allocating and assigning by slice avoids the
    broadcast.
    """
    arr = np.empty(len(values), dtype=object)
    arr[:] = values
    return arr


def _period_keys(index: pd.Index, freq: str) -> np.ndarray:
    """Period key of each row, with the entity kept alongside for a panel.

    Returned as a numpy array (not a plain list) so pandas treats it as a
    single array-like grouper: ``.groupby(some_list)`` instead interprets
    each list element as a *separate* grouping key, raising a spurious
    ``KeyError`` on the first tuple.
    """
    if isinstance(index, pd.MultiIndex):
        entities = index.droplevel(-1)
        periods = index.get_level_values(-1).to_period(freq)
        return _object_array(list(zip(entities, periods)))
    return _object_array(index.to_period(freq).tolist())


def _assert_additive_coherence(
    target_frame: pd.DataFrame, original: pd.DataFrame, column: str, freq: str
) -> int:
    """Sum of imputed target-level values per low-frequency period == the observed total.

    Periods with no observed anchor at all (delayed end of series, §2.9 —
    intentionally left un-rescaled) and periods whose target level is still
    partially NaN (not fully predicted yet) are skipped, as are all-zero
    sums (a period entirely outside the imputation window sums to 0 by
    construction, which would otherwise wrongly read as "coherent").
    """
    result_keys = _period_keys(target_frame.index, freq)
    original_keys = _period_keys(original.index, freq)

    observed_totals = original[column].groupby(original_keys).sum()
    observed_present = original[column].groupby(original_keys).apply(lambda s: s.notna().any())
    imputed_sums = target_frame[column].groupby(result_keys).sum()
    imputed_complete = target_frame[column].groupby(result_keys).apply(lambda s: s.notna().all())

    checked = 0
    for period_key, observed_total in observed_totals.items():
        if not observed_present.get(period_key, False) or observed_total == 0:
            continue
        if period_key not in imputed_sums.index or not imputed_complete.get(period_key, False):
            continue
        assert imputed_sums[period_key] == pytest.approx(observed_total, abs=1e-6), (
            f"period {period_key} of '{column}' sums to {imputed_sums[period_key]}, "
            f"expected {observed_total}"
        )
        checked += 1
    return checked


class TestScopeScenarioShape:
    """Forme de la sortie : préservée sur toute la matrice, `imputation_scope` compris.

    `imputation_scope` ne pilote que la fenêtre d'ENTRAÎNEMENT (quelles
    dates comptent comme suffisamment couvertes pour fitter) : il ne doit en
    aucun cas changer le nombre de lignes ni l'index du niveau cible en
    sortie.
    """

    @pytest.mark.parametrize('param', FULL_SCENARIOS, ids=_full_scenario_id)
    def test_target_level_matches_source_index(self, request, param):
        """Le niveau cible de la sortie couvre exactement l'index des données source."""
        dataset_name, cascade_refitting, keep_lower_frequencies, imputation_scope = param
        data = request.getfixturevalue(dataset_name)
        imputer, result = _run_hfi(data, cascade_refitting, keep_lower_frequencies, imputation_scope)

        target_frame = _target_level_frame(imputer, result, keep_lower_frequencies)

        assert _index_as_set(target_frame.index) == _index_as_set(data.index)
        assert list(target_frame.columns) == list(data.columns)


class TestScopeScenarioNoFullyNanColumns:
    """Aucune colonne du niveau cible n'est intégralement NaN, quel que soit le scope."""

    @pytest.mark.parametrize('param', FULL_SCENARIOS, ids=_full_scenario_id)
    def test_target_level_has_no_fully_nan_column(self, request, param):
        dataset_name, cascade_refitting, keep_lower_frequencies, imputation_scope = param
        data = request.getfixturevalue(dataset_name)
        imputer, result = _run_hfi(data, cascade_refitting, keep_lower_frequencies, imputation_scope)

        target_frame = _target_level_frame(imputer, result, keep_lower_frequencies)
        fully_nan_cols = [c for c in target_frame.columns if target_frame[c].isna().all()]

        assert fully_nan_cols == []


class TestScopeScenarioAdditiveCoherence:
    """§2.6 : la somme des sous-périodes imputées égale la valeur basse fréquence observée."""

    @pytest.mark.parametrize('param', FULL_SCENARIOS, ids=_full_scenario_id)
    def test_quarterly_and_annual_totals_enforced(self, request, param):
        dataset_name, cascade_refitting, keep_lower_frequencies, imputation_scope = param
        data = request.getfixturevalue(dataset_name)
        imputer, result = _run_hfi(data, cascade_refitting, keep_lower_frequencies, imputation_scope)

        target_frame = _target_level_frame(imputer, result, keep_lower_frequencies)

        checked_q = _assert_additive_coherence(target_frame, data, 'pib_trimestriel', 'Q')
        checked_y = _assert_additive_coherence(
            target_frame, data, 'balance_commerciale_annuelle', 'Y'
        )
        assert checked_q > 0
        assert checked_y > 0


class TestFitTransformSymmetry:
    """La symétrie `fit_transform(X)` == `fit(X)` puis `transform(X)` (instances séparées)."""

    @pytest.mark.parametrize('param', ALL_SCENARIOS, ids=_scenario_id)
    def test_fit_transform_matches_separate_fit_and_transform(self, request, param):
        dataset_name, cascade_refitting, keep_lower_frequencies = param
        data = request.getfixturevalue(dataset_name)

        kwargs = dict(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=cascade_refitting,
            keep_lower_frequencies=keep_lower_frequencies,
        )
        combined = HighFrequencyImputer(**kwargs)
        separate = HighFrequencyImputer(**kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result_combined = combined.fit_transform(data.copy())
            separate.fit(data.copy())
            result_separate = separate.transform(data.copy())

        pd.testing.assert_frame_equal(
            result_combined.sort_index(), result_separate.sort_index()
        )
