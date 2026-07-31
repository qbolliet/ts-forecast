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
d'échelle), §2.2 / §5.1 (frames d'étape) et §2.3 / §5.2 (registre de modèles
indexé par étape).
"""
import warnings

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

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


def _run_hfi(data: pd.DataFrame, cascade_refitting: bool, keep_lower_frequencies: bool):
    """Fit_transform avec LinearRegression, target_frequency='M', warnings ignorés."""
    imputer = HighFrequencyImputer(
        target_frequency='M',
        estimator=LinearRegression(),
        cascade_refitting=cascade_refitting,
        keep_lower_frequencies=keep_lower_frequencies,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = imputer.fit_transform(data.copy())
    return imputer, result


def _target_level_frame(
    result: pd.DataFrame,
    keep_lower_frequencies: bool,
) -> pd.DataFrame:
    """Extract the target-frequency level from a (possibly multi-freq) result."""
    if not keep_lower_frequencies:
        return result
    return result.xs('target', level='frequency')


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

        target_frame = _target_level_frame(result, keep_lower_frequencies)

        assert _index_as_set(target_frame.index) == _index_as_set(data.index)

    @pytest.mark.parametrize('param', ALL_SCENARIOS, ids=_scenario_id)
    def test_target_level_has_no_fully_nan_column(self, request, param):
        """Aucune colonne du niveau cible n'est intégralement NaN."""
        dataset_name, cascade_refitting, keep_lower_frequencies = param
        data = request.getfixturevalue(dataset_name)
        imputer, result = _run_hfi(data, cascade_refitting, keep_lower_frequencies)

        target_frame = _target_level_frame(result, keep_lower_frequencies)
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

    @pytest.mark.parametrize(
        'param',
        [
            pytest.param(
                p, id=_scenario_id(p),
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="§2.5 high_frequency_imputer_review.md : le frame final est "
                           "ajouté deux fois, sous le label de fréquence réel ('M') et "
                           "sous le label générique 'target', avec un contenu identique.",
                ),
            )
            for p in KEEP_LOWER_SCENARIOS
        ],
    )
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
