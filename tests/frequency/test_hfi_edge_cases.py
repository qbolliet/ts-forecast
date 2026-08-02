"""Tests de cas limites pour `HighFrequencyImputer` et ses dépendances.

Complète `test_hfi_integration.py` (matrice de scénarios nominaux) avec les
cas limites priorisés par `CLAUDE.md` : données non triées, index dupliqués,
entités manquantes ou inconnues dans les panels, fréquences irrégulières,
dataset à une seule observation, colonne entièrement NaN, panel à une seule
entité et panel à deux niveaux d'entité.

Convention : quand le comportement actuel est un échec contrôlé (`ValueError`
avec un message clair), le test fige ce contrat plutôt que d'exiger un
succès silencieux — lever une erreur explicite est le comportement désiré
pour ces cas-là, pas un bug.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer


def _fit_transform_quiet(imputer: HighFrequencyImputer, data: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return imputer.fit_transform(data.copy())


# ---------------------------------------------------------------------------
# Données non triées (ordre temporel)
# ---------------------------------------------------------------------------
class TestUnsortedData:
    """`HighFrequencyImputer` fige `auto_sort=False` : les données non triées
    doivent lever une erreur claire plutôt que produire un résultat erroné en
    silence (contrairement à `validate_temporal_data(sort_data=True)`, le
    comportement par défaut ailleurs dans le package).
    """

    def test_unsorted_time_series_raises(self, mixed_freq_timeseries):
        shuffled = mixed_freq_timeseries.sample(frac=1.0, random_state=0)
        imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())

        with pytest.raises(ValueError, match='not sorted'):
            imputer.fit(shuffled)

    def test_unsorted_panel_raises(self, mixed_freq_panel):
        shuffled = mixed_freq_panel.sample(frac=1.0, random_state=0)
        imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())

        with pytest.raises(ValueError, match='not contiguous'):
            imputer.fit(shuffled)


# ---------------------------------------------------------------------------
# Index dupliqués
# ---------------------------------------------------------------------------
class TestDuplicateIndex:
    """Un index dupliqué doit lever une erreur claire, jamais produire un
    résultat silencieusement faux (agrégations/fenêtres mal comptées).
    """

    def test_duplicate_timestamp_raises(self, mixed_freq_timeseries):
        duplicated = pd.concat([mixed_freq_timeseries.iloc[[5]], mixed_freq_timeseries])
        imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())

        with pytest.raises(ValueError, match='duplicate'):
            imputer.fit(duplicated)

    def test_duplicate_entity_date_pair_raises(self, mixed_freq_panel):
        duplicated = pd.concat([mixed_freq_panel.iloc[[5]], mixed_freq_panel])
        imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())

        with pytest.raises(ValueError):
            imputer.fit(duplicated)


# ---------------------------------------------------------------------------
# Entités manquantes / inconnues dans un panel
# ---------------------------------------------------------------------------
class TestPanelEntityMismatchBetweenFitAndTransform:
    """§2.4 : le modèle GLOBAL par variable (pas un modèle par entité) doit
    rester utilisable même quand `transform` ne voit pas exactement les
    mêmes entités que `fit`.
    """

    def test_transform_with_fewer_entities_than_fit(self, mixed_freq_panel):
        """Une entité vue au fit peut être absente au transform sans crash."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(mixed_freq_panel)
            subset = mixed_freq_panel.loc[
                mixed_freq_panel.index.get_level_values(0) != 'Allemagne'
            ]
            result = imputer.transform(subset)

        remaining_entities = set(result.index.get_level_values(0).unique())
        assert remaining_entities == {'France', 'Italie'}
        target_label = imputer._stage_frequency_label(imputer.effective_target_frequency_)
        target_frame = result.xs(target_label, level='frequency')
        assert [c for c in target_frame.columns if target_frame[c].isna().all()] == []

    def test_transform_with_entity_unseen_at_fit(self, mixed_freq_panel):
        """Une entité absente au fit mais présente au transform est tout de
        même imputée par le modèle global, sans `KeyError`.
        """
        fit_subset = mixed_freq_panel.loc[
            mixed_freq_panel.index.get_level_values(0) != 'Italie'
        ]
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(fit_subset)
            result = imputer.transform(mixed_freq_panel)

        target_label = imputer._stage_frequency_label(imputer.effective_target_frequency_)
        italie_target = result.xs('Italie', level=0).xs(target_label, level='frequency')
        assert [c for c in italie_target.columns if italie_target[c].isna().all()] == []


# ---------------------------------------------------------------------------
# Panel à une seule entité
# ---------------------------------------------------------------------------
class TestSingleEntityPanel:
    """Un panel réduit à une seule entité doit se comporter comme une série
    temporelle unique, sans branche panel défaillante.
    """

    def test_fit_transform_single_entity_panel(self, mixed_freq_panel):
        single_entity = mixed_freq_panel.loc[
            mixed_freq_panel.index.get_level_values(0) == 'France'
        ]
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=True,
            keep_lower_frequencies=True,
        )
        result = _fit_transform_quiet(imputer, single_entity)

        assert imputer.is_panel_ is True
        assert set(result.index.get_level_values(0).unique()) == {'France'}
        target_label = imputer._stage_frequency_label(imputer.effective_target_frequency_)
        target_frame = result.xs(target_label, level='frequency')
        assert len(target_frame) == len(single_entity)
        assert [c for c in target_frame.columns if target_frame[c].isna().all()] == []


# ---------------------------------------------------------------------------
# Panel à deux niveaux d'entité
# ---------------------------------------------------------------------------
def _build_two_level_panel(seed: int = 11) -> pd.DataFrame:
    """Small panel with a 2-level entity MultiIndex (country, sector, date).

    Mirrors `mixed_freq_panel`'s structure (monthly covariate + quarterly
    additive variable) but with entities identified by TWO levels, to
    exercise the tuple-key normalization (§3.4/§5.4) all the way through a
    full `HighFrequencyImputer` fit/transform, not just
    `ImputationWindowCalculator` in isolation.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2019-01-01', periods=36, freq='MS')
    entities = [('FR', 'industrie'), ('FR', 'services'), ('DE', 'industrie')]

    frames = []
    for country, sector in entities:
        monthly = 50.0 + rng.normal(0, 2.0, len(dates))
        quarterly = pd.Series(np.nan, index=dates)
        for _, block in pd.Series(monthly, index=dates).groupby(dates.to_period('Q')):
            quarterly.loc[block.index[0]] = block.sum()

        frame = pd.DataFrame({'covariable_mensuelle': monthly, 'variable_trimestrielle': quarterly})
        frame['country'] = country
        frame['sector'] = sector
        frame['date'] = dates
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True).set_index(['country', 'sector', 'date'])
    return panel.sort_index()


class TestTwoLevelEntityPanel:
    """§3.4/§5.4 : un panel à 2 niveaux d'entité doit fonctionner de bout en
    bout, pas seulement au niveau du calculateur de fenêtre.
    """

    def test_fit_transform_two_level_entity_panel(self):
        panel = _build_two_level_panel()
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            cascade_refitting=False,
            keep_lower_frequencies=False,
        )
        result = _fit_transform_quiet(imputer, panel)

        assert len(result) == len(panel)
        assert set(result.index.get_level_values(0).unique()) == {'FR', 'DE'}
        assert set(result.index.get_level_values(1).unique()) == {'industrie', 'services'}
        assert result['variable_trimestrielle'].isna().sum() == 0

        # Les clés d'entité du registre de modèles restent des tuples à 2 niveaux
        for stage_key in imputer.model_fitting_order_:
            assert isinstance(stage_key, tuple)


# ---------------------------------------------------------------------------
# Dataset à une seule observation
# ---------------------------------------------------------------------------
class TestSingleObservationDataset:
    """Une seule observation ne fournit jamais assez d'information pour
    détecter une fréquence ou entraîner un modèle : le contrat attendu est
    un `ValueError` explicite, pas un résultat silencieusement vide.
    """

    def test_single_observation_raises(self):
        dates = pd.date_range('2023-01-01', periods=1, freq='MS')
        df = pd.DataFrame({'value': [1.0]}, index=dates)
        df.index.name = 'date'
        imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())

        with pytest.raises(ValueError, match='non-null observations'):
            imputer.fit_transform(df)


# ---------------------------------------------------------------------------
# Colonne entièrement NaN
# ---------------------------------------------------------------------------
class TestAllNaNColumn:
    """Une colonne sans aucune valeur observée ne peut être ni classifiée ni
    imputée : le contrat attendu est un `ValueError` explicite (garde-fou
    "minimum 2 observations non-nulles"), pas une colonne fantôme dans la
    sortie.
    """

    def test_fully_nan_column_raises(self, mixed_freq_timeseries):
        data = mixed_freq_timeseries.copy()
        data['completement_vide'] = np.nan
        imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())

        with pytest.raises(ValueError, match='non-null observations'):
            imputer.fit_transform(data)


# ---------------------------------------------------------------------------
# Fréquences irrégulières
# ---------------------------------------------------------------------------
class TestIrregularFrequency:
    """Une fréquence non détectable doit échouer proprement plutôt que
    produire un résultat incohérent.

    Connu et hors périmètre de cette campagne : le message effectivement
    levé ("Frequency must be a string, got <class 'NoneType'>",
    `tsforecast/utils/frequency/normalizer.py:109`) est peu informatif — il
    provient de `normalize_frequency(return_format='full')`
    (`tsforecast/utils/frequency/utils.py`) qui avale le `ValueError`
    explicite de `parse_frequency` ("Could not detect index frequency...")
    avant de retomber sur `_normalizer.normalize(None)`. Ces deux tests
    figent uniquement le fait qu'une erreur contrôlée est levée (pas un
    crash différent ni un résultat silencieux), sans figer ce message précis
    ni corriger `normalize_frequency` : ce fichier est hors du périmètre de
    la revue `HighFrequencyImputer` (`high_frequency_imputer.py`,
    `imputation_window.py`, `frequency_aligner.py`).
    """

    def test_undetectable_frequency_raises(self):
        dates = pd.to_datetime(
            ['2020-01-01', '2020-02-01', '2020-04-01', '2020-05-01', '2020-09-01', '2020-10-01']
        )
        df = pd.DataFrame(
            {'a': range(6), 'b': [1, np.nan, 3, np.nan, 5, np.nan]}, index=dates
        )
        df.index.name = 'date'
        imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())

        with pytest.raises(ValueError):
            imputer.fit_transform(df)

    def test_single_gap_in_otherwise_regular_monthly_series_raises(self):
        """Une série mensuelle régulière avec UN SEUL mois manquant échoue
        déjà à la détection de fréquence dès qu'une agrégation de fréquence
        est nécessaire (cas plus préoccupant qu'une série franchement
        irrégulière : c'est un accident de données courant). Une variable
        seule, sans covariable de fréquence différente, ne déclenche pas ce
        chemin de code et n'échoue donc pas : la seconde colonne
        (`quarterly`) est nécessaire pour reproduire le problème.
        """
        dates = pd.date_range('2020-01-01', periods=24, freq='MS').delete(10)
        rng = np.random.default_rng(3)
        monthly = 10 + rng.normal(0, 1, len(dates))
        quarterly = pd.Series(np.nan, index=dates)
        quarterly.iloc[::3] = 100 + np.arange(len(quarterly.iloc[::3]))
        df = pd.DataFrame({'monthly_dense': monthly, 'quarterly': quarterly}, index=dates)
        df.index.name = 'date'
        imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression())

        with pytest.raises(ValueError):
            imputer.fit_transform(df)
