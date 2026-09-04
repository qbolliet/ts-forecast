"""Tests for tsforecast.frequency.covariate_materializer.

Couvre §4.1 (classification identité / agrégation / stratégie), §4.2
('tolerate_nan'), §4.3 ('interpolate' et le repli), §4.4 ('model' et la
précédence de matérialisation à quatre rangs, décision D13), §4.5
(covariate_eligibility), §4.6 (unicité de la voie, socle du test I11), §6.2
(les trois registres et les origines de cellule) et §4.7 / D14 (formulation
testable de l'invariant central) de [SPEC]
high_frequency_imputer2_architecture.md.

Les jeux sont ceux du document : `reference_timeseries` (§2.2) et
`mixed_freq_panel_heterogeneous` (§2.3). Lot purement additif : hfi et ses
tests restent intacts.
"""
# Modules de base
import numpy as np
import pandas as pd
import pytest

# Objets testés
from tsforecast.frequency.covariate_materializer import (
    CovariateMaterializer,
    DEFAULT_MATERIALIZATION_KEY,
    _WAY_ORIGIN,
)


# Fréquences détectées du jeu TS (§2.2), partagées par la plupart des tests
TS_FREQUENCIES = {'m1': 'M', 'q1': 'Q', 'a1': 'Y', 'a2': 'Y'}
# Mêmes fréquences pour le panel hétérogène, plus la colonne mensuelle d'enquête
PANEL_FREQUENCIES = {**TS_FREQUENCIES, 'climat_affaires': 'M'}


# Grille mensuelle de fin de mois, celle des jeux de référence
def _monthly_grid(start: str = '2021-01-31', periods: int = 36) -> pd.DatetimeIndex:
    """Build a month-end grid."""
    return pd.date_range(start=start, periods=periods, freq='ME')


# Grille trimestrielle de fin de trimestre
def _quarterly_grid(start: str = '2021-03-31', periods: int = 12) -> pd.DatetimeIndex:
    """Build a quarter-end grid."""
    return pd.date_range(start=start, periods=periods, freq='QE')


# Grille annuelle de fin d'année
def _annual_grid(start: str = '2021-12-31', periods: int = 3) -> pd.DatetimeIndex:
    """Build a year-end grid."""
    return pd.date_range(start=start, periods=periods, freq='YE')


# Grille des trois ancres annuelles, sous-ensemble de la grille trimestrielle
def _annual_anchor_grid() -> pd.DatetimeIndex:
    """Build the three annual anchors of the ``TS`` dataset."""
    return pd.DatetimeIndex(
        pd.to_datetime(['2021-12-31', '2022-12-31', '2023-12-31'])
    )


# Amorçage manuel des trois registres, comme l'aurait fait une étape antérieure
def _prime_stores(
    materializer: CovariateMaterializer,
    column: str,
    values: pd.Series,
    freq,
    origin,
) -> None:
    """Prime the three stores of one column, as an earlier stage would have."""
    materializer.imputed_store[column] = values
    materializer.imputed_freq_store[column] = pd.Series(freq, index=values.index)
    materializer.origin_store[column] = pd.Series(
        origin, index=values.index, dtype=object
    )


# Grille de panel : produit cartésien des entités et des dates
def _panel_grid(data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.MultiIndex:
    """Build a panel MultiIndex pairing the data's entities with ``dates``."""
    entities = data.index.get_level_values(0).unique().tolist()
    return pd.MultiIndex.from_product(
        [entities, dates], names=list(data.index.names)
    )


class TestRankOneClassification:
    """Rang 1 du §4.1 : identité et agrégation exacte, origines 'observed'."""

    def test_identity_and_aggregate_ranks(self, reference_timeseries):
        """f_c == f donne 'identity', f_c plus fine donne 'aggregate'."""
        materializer = CovariateMaterializer()
        grid = _quarterly_grid()

        features, ways, origins = materializer.materialize(
            columns=['q1', 'm1'],
            grid_index=grid,
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        # Voies retenues
        assert ways == {'q1': 'identity', 'm1': 'aggregate'}
        # Origines agrégées : une agrégation exacte n'est pas une approximation
        assert origins == {'q1': 'observed', 'm1': 'observed'}

        # Identité : les valeurs trimestrielles portées telles quelles
        expected_q1 = 10.0 * np.arange(1, 13)
        np.testing.assert_allclose(features['q1'].to_numpy(), expected_q1)

        # Agrégation : m1 vaut 100 + i, sommée par trimestre
        assert features['m1'].iloc[0] == pytest.approx(100 + 101 + 102)
        assert features['m1'].iloc[1] == pytest.approx(103 + 104 + 105)
        # Aucune période incomplète sur cette grille
        assert features['m1'].notna().all()

        # Registre d'origines : 'observed' partout où une valeur est produite
        for column in ('q1', 'm1'):
            assert set(materializer.origin_store[column].unique()) == {'observed'}

    def test_rank1_aggregation_is_always_a_sum(self, reference_timeseries):
        """Une covariable plus fine est agrégée par SOMME même sous None (D20).

        ``aggregation_constraint`` ne gouverne que le recalage des covariables
        INTERPOLÉES (§4.3) : l'agrégation exacte du rang 1 n'y est jamais
        soumise.
        """
        materializer = CovariateMaterializer(aggregation_constraint=None)
        grid = _quarterly_grid()

        features, ways, origins = materializer.materialize(
            columns=['m1'],
            grid_index=grid,
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        assert ways == {'m1': 'aggregate'}
        assert origins == {'m1': 'observed'}
        # m1 vaut 100 + i, sommée par trimestre, comme sous le défaut 'sum'
        assert features['m1'].iloc[0] == pytest.approx(100 + 101 + 102)
        assert features['m1'].iloc[1] == pytest.approx(103 + 104 + 105)

    def test_incomplete_period_aggregation_is_nan(self, reference_timeseries):
        """Une période incomplète produit NaN, source légitime et non masquée."""
        materializer = CovariateMaterializer()
        # Retrait de décembre 2023 : l'année 2023 devient incomplète
        truncated = reference_timeseries.iloc[:-1]
        grid = _annual_grid()

        features, ways, _origins = materializer.materialize(
            columns=['m1'],
            grid_index=grid,
            stage_freq='Y',
            detected_frequencies=TS_FREQUENCIES,
            source_data=truncated,
        )

        assert ways['m1'] == 'aggregate'
        # Les deux années complètes sont sommées, la troisième est NaN
        assert features['m1'].loc['2021-12-31'] == pytest.approx(1266.0)
        assert features['m1'].loc['2022-12-31'] == pytest.approx(1410.0)
        assert np.isnan(features['m1'].loc['2023-12-31'])
        # La cellule non produite n'entre pas dans le registre
        assert pd.Timestamp('2023-12-31') not in materializer.origin_store['m1'].index


class TestTolerateNan:
    """Stratégie 'tolerate_nan' (§4.2) : aucune matérialisation."""

    def test_tolerate_nan_keeps_anchors_only(self, reference_timeseries):
        """a1 vaut 120 au 2021-12-31 et NaN sur les 11 autres mois, des deux côtés."""
        materializer = CovariateMaterializer(covariate_strategy='tolerate_nan')
        # Deux grilles mensuelles disjointes, jouant le fit et le predict
        grid_fit = _monthly_grid('2021-01-31', 12)
        grid_pred = _monthly_grid('2022-01-31', 12)

        x_train, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=grid_fit,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        # La voie est imposée à la seconde grille (§4.6)
        x_pred, ways_pred, _ = materializer.materialize(
            columns=['a1'],
            grid_index=grid_pred,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            materialization=ways,
        )

        assert ways == {'a1': 'raw_anchors'}
        assert ways_pred == ways
        assert origins == {'a1': 'observed'}

        # Valeur d'or du document à l'ancre, NaN sur les onze autres mois
        assert x_train['a1'].loc['2021-12-31'] == pytest.approx(120.0)
        assert x_train['a1'].isna().sum() == 11
        # Taux de NaN identique au predict : 11/12 des deux côtés
        assert x_pred['a1'].loc['2022-12-31'] == pytest.approx(132.0)
        assert x_pred['a1'].isna().sum() == 11


class TestInterpolate:
    """Stratégie 'interpolate' (§4.3), le défaut."""

    def test_interpolate_removes_all_nan_except_empty_entity(
        self, mixed_freq_panel_heterogeneous
    ):
        """Seule l'entité sans aucune observation garde des NaN (§4.5)."""
        data = mixed_freq_panel_heterogeneous
        materializer = CovariateMaterializer(covariate_strategy='interpolate')
        grid = _panel_grid(data, _monthly_grid())

        features, ways, origins = materializer.materialize(
            columns=['m1', 'q1', 'a1', 'a2', 'climat_affaires'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
        )

        # Colonnes plus basses que la grille : interpolées ; les autres, rang 1
        assert ways == {
            'm1': 'identity',
            'q1': 'interpolate',
            'a1': 'interpolate',
            'a2': 'interpolate',
            'climat_affaires': 'identity',
        }
        assert origins['q1'] == 'interpolated'
        assert origins['m1'] == 'observed'

        # Plus aucun NaN, sauf climat_affaires pour l'Italie
        assert features[['m1', 'q1', 'a1', 'a2']].notna().all().all()
        assert features.loc['IT', 'climat_affaires'].isna().all()
        for entity in ('FR', 'DE'):
            assert features.loc[entity, 'climat_affaires'].notna().all()

    def test_interpolate_uses_per_feature_method_and_anchor(self, reference_timeseries):
        """Forme dict avec '__default__' : méthode et ancrage résolus par feature."""
        materializer = CovariateMaterializer(
            interpolation_method={'a1': 'nearest', DEFAULT_MATERIALIZATION_KEY: 'linear'},
            interpolation_anchor={'a1': None, DEFAULT_MATERIALIZATION_KEY: 0.5},
        )

        # Résolution des réglages
        assert materializer.resolve_method('a1') == 'nearest'
        assert materializer.resolve_method('a2') == 'linear'
        assert materializer.resolve_anchor('a1') is None
        assert materializer.resolve_anchor('a2') == 0.5

        grid = _monthly_grid()
        features, _ways, _origins = materializer.materialize(
            columns=['a1', 'a2'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        # 'nearest' sans ancrage recopie l'ancre sur toute la période
        assert set(np.round(features['a1'].dropna().unique(), 6)) <= {120.0, 132.0, 150.0}

        # Référence linéaire sans ancrage, pour montrer que 0.5 déplace la forme
        reference = CovariateMaterializer()
        flat, _, _ = reference.materialize(
            columns=['a2'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        assert not np.allclose(
            features['a2'].to_numpy(), flat['a2'].to_numpy(), equal_nan=True
        )
        # L'ancrage au milieu de période retire la valeur d'ancre de la sortie
        assert features['a2'].loc['2022-12-31'] != pytest.approx(66.0)

    def test_unknown_dict_key_raises_listing_columns(self):
        """Les clés inconnues des dicts par feature sont refusées et listées."""
        materializer = CovariateMaterializer(interpolation_method={'zz': 'cubic'})
        with pytest.raises(ValueError, match=r"interpolation_method names unknown columns"):
            materializer.validate_columns(['a1', 'q1'])
        # La clé de repli n'est jamais signalée comme inconnue
        CovariateMaterializer(
            interpolation_anchor={DEFAULT_MATERIALIZATION_KEY: 0.5}
        ).validate_columns(['a1'])


class TestStores:
    """Les trois registres du §6.2, et leurs règles d'écriture."""

    def test_origin_store_values(self, reference_timeseries):
        """Les quatre règles d'origine du §6.2."""
        materializer = CovariateMaterializer(covariate_strategy='interpolate')
        grid = _quarterly_grid()

        materializer.materialize(
            columns=['q1', 'm1', 'a1'],
            grid_index=grid,
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        # 1. Une cellule d'entrée non NaN vaut 'observed'
        assert set(materializer.origin_store['q1'].unique()) == {'observed'}
        # 2. Une agrégation exacte vaut 'observed'
        assert set(materializer.origin_store['m1'].unique()) == {'observed'}
        # 3. Une interpolation vaut 'interpolated'
        assert set(materializer.origin_store['a1'].unique()) == {'interpolated'}

        # 3 bis. Un repli d'interpolation vaut aussi 'interpolated'
        fallback = CovariateMaterializer()
        fallback.interpolate_column(
            'a2', grid, 'Q', TS_FREQUENCIES, reference_timeseries
        )
        assert set(fallback.origin_store['a2'].unique()) == {'interpolated'}

        # 4. Une prédiction de modèle vaut 'model', report d'étape compris
        assert _WAY_ORIGIN['stage_model'] == 'model'
        assert _WAY_ORIGIN['carried_model'] == 'model'
        model = CovariateMaterializer(covariate_strategy='model')
        _prime_stores(
            model, 'a1', pd.Series(20.0 + np.arange(12), index=grid), 'Q', 'model'
        )
        _features, _ways, origins = model.materialize(
            columns=['a1'],
            grid_index=grid,
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            materialization={'a1': 'stage_model'},
        )
        assert origins == {'a1': 'model'}
        assert set(model.origin_store['a1'].unique()) == {'model'}

    def test_fallback_writes_to_all_three_stores(self, reference_timeseries):
        """« Le repli matérialise » : les trois registres sont alimentés."""
        materializer = CovariateMaterializer()
        grid = _monthly_grid()

        values = materializer.interpolate_column(
            'a1', grid, 'M', TS_FREQUENCIES, reference_timeseries
        )

        # Les trois registres portent la colonne, sur le même index
        for store in (
            materializer.imputed_store,
            materializer.imputed_freq_store,
            materializer.origin_store,
        ):
            assert 'a1' in store
            assert store['a1'].index.equals(values.dropna().index)

        # Miroir des valeurs, fréquence de production et origine
        pd.testing.assert_series_equal(
            materializer.imputed_store['a1'], values.dropna(), check_names=False
        )
        assert set(materializer.imputed_freq_store['a1'].unique()) == {'M'}
        assert set(materializer.origin_store['a1'].unique()) == {'interpolated'}

        # Le snapshot est indépendant des écritures ultérieures
        snapshot = materializer.snapshot()
        materializer.reset()
        assert materializer.origin_store == {}
        assert set(snapshot['origin']['a1'].unique()) == {'interpolated'}

    def test_combine_first_never_overwrites(self, reference_timeseries):
        """B5 : une écriture ne détruit jamais les cellules qu'elle ne couvre pas."""
        materializer = CovariateMaterializer(covariate_strategy='tolerate_nan')
        grid_first = _monthly_grid('2021-01-31', 12)
        grid_second = _monthly_grid('2022-01-31', 12)

        materializer.materialize(
            columns=['a1'],
            grid_index=grid_first,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        first_state = materializer.imputed_store['a1'].copy()
        assert first_state.index.tolist() == [pd.Timestamp('2021-12-31')]

        # Seconde écriture sur une grille disjointe
        materializer.materialize(
            columns=['a1'],
            grid_index=grid_second,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        store = materializer.imputed_store['a1']

        # L'index a crû vers l'union : la première ancre a survécu
        assert store.index.tolist() == [
            pd.Timestamp('2021-12-31'), pd.Timestamp('2022-12-31')
        ]
        assert store.loc['2021-12-31'] == pytest.approx(120.0)
        assert store.loc['2022-12-31'] == pytest.approx(132.0)
        # Les trois registres ont crû ensemble
        assert len(materializer.origin_store['a1']) == 2
        assert len(materializer.imputed_freq_store['a1']) == 2


class TestUniquenessOfTheWay:
    """§4.6 : la voie est choisie une fois et rejouée telle quelle (socle de I11)."""

    def test_replayed_materialization_is_identical(self, mixed_freq_panel_heterogeneous):
        """Choix puis rejeu produisent exactement les mêmes valeurs et voies."""
        data = mixed_freq_panel_heterogeneous
        columns = ['m1', 'q1', 'a1', 'climat_affaires']
        grid = _panel_grid(data, _monthly_grid())

        chooser = CovariateMaterializer()
        chosen, ways, origins = chooser.materialize(
            columns=columns,
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
        )

        replayer = CovariateMaterializer()
        replayed, ways_replayed, origins_replayed = replayer.materialize(
            columns=columns,
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
            materialization=ways,
        )

        assert ways_replayed == ways
        assert origins_replayed == origins
        pd.testing.assert_frame_equal(replayed, chosen)

    def test_replay_must_cover_exactly_the_columns(self, reference_timeseries):
        """Une matérialisation incomplète ou orpheline est refusée."""
        materializer = CovariateMaterializer()
        grid = _monthly_grid()
        with pytest.raises(ValueError, match=r"doit couvrir exactement columns"):
            materializer.materialize(
                columns=['m1', 'a1'],
                grid_index=grid,
                stage_freq='M',
                detected_frequencies=TS_FREQUENCIES,
                source_data=reference_timeseries,
                materialization={'m1': 'identity'},
            )


    def test_way_unicity_fit_and_pred(self, reference_timeseries):
        """§4.6 : sous 'model', le repli s'applique AUSSI au fit.

        Jeu `TS`, étape `Q`, ordre `a1` avant `a2` : au moment où `a1` est
        imputée, `a2` n'est matérialisée par aucun rang supérieur, elle relève
        donc du `covariate_fallback` — des deux côtés, y compris sur la grille
        d'entraînement où ses seules ancres suffiraient (les 3 ancres annuelles
        sont renseignées). Sinon le modèle apprendrait sur la covariable exacte
        et prédirait sur la covariable interpolée.
        """
        materializer = CovariateMaterializer(covariate_strategy='model')
        # L'étape est Q des deux côtés : la grille d'entraînement est la grille
        # Q masquée aux 3 ancres annuelles de a1, la grille de prédiction est
        # la grille Q complète
        grid_train = _annual_anchor_grid()
        grid_pred = _quarterly_grid()

        x_train, ways, origins = materializer.materialize(
            columns=['m1', 'q1', 'a2'],
            grid_index=grid_train,
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        x_pred, ways_pred, _origins_pred = materializer.materialize(
            columns=['m1', 'q1', 'a2'],
            grid_index=grid_pred,
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            materialization=ways,
        )

        # Rang 4 des deux côtés, jamais le rang 1 que les ancres autoriseraient
        assert ways['a2'] == 'interpolate'
        assert ways_pred == ways
        # La NATURE des valeurs, pas seulement le motif de NaN
        assert origins['a2'] == 'interpolated'
        assert x_train['a2'].notna().all()
        assert x_pred['a2'].notna().all()

        # Valeur par valeur : celle du repli, produite par la même routine
        reference = CovariateMaterializer(covariate_strategy='model')
        expected = reference.interpolate_column(
            'a2', grid_train, 'Q', TS_FREQUENCIES, reference_timeseries
        )
        np.testing.assert_allclose(
            x_train['a2'].to_numpy(), expected.to_numpy()
        )

        # Cas numériquement visible : avec un ancrage à mi-période, le repli
        # du fit ne redonne PAS les ancres 60 / 66 de a2
        anchored = CovariateMaterializer(
            covariate_strategy='model',
            interpolation_anchor={'a2': 0.5, DEFAULT_MATERIALIZATION_KEY: None},
        )
        x_anchored, ways_anchored, _ = anchored.materialize(
            columns=['a2'],
            grid_index=grid_train[:2],
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )
        assert ways_anchored == {'a2': 'interpolate'}
        np.testing.assert_allclose(
            x_anchored['a2'].to_numpy(), [62.99178082, 68.99178082]
        )


class TestPrecedenceRanks:
    """§4.4 : la précédence à quatre rangs, arrêt au premier rang applicable."""

    def test_rank_two_reads_stage_mirror(self, reference_timeseries):
        """Rang 2 : les valeurs viennent du miroir d'étape, pas de la source."""
        grid = _monthly_grid()
        materializer = CovariateMaterializer(covariate_strategy='model')
        mirror = pd.Series(7.0 + np.arange(len(grid)), index=grid)
        _prime_stores(materializer, 'a1', mirror, 'M', 'model')

        # Source aveugle : a1 n'y porte plus aucune observation
        blind = reference_timeseries.copy()
        blind['a1'] = np.nan

        features, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=blind,
        )

        assert ways == {'a1': 'stage_model'}
        assert origins == {'a1': 'model'}
        # Le miroir, cellule par cellule, alors que la source est vide
        np.testing.assert_allclose(features['a1'].to_numpy(), mirror.to_numpy())

    def test_rank_two_origin_follows_store(self, reference_timeseries):
        """Rang 2 : l'origine est LUE, jamais déduite de la présence d'un modèle."""
        grid = _monthly_grid()
        materializer = CovariateMaterializer(covariate_strategy='model')
        mirror = pd.Series(7.0 + np.arange(len(grid)), index=grid)
        # Étape produite par repli : le miroir porte 'interpolated'
        _prime_stores(materializer, 'a1', mirror, 'M', 'interpolated')

        features, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        # La voie reste celle du rang 2, seule l'origine change
        assert ways == {'a1': 'stage_model'}
        assert origins == {'a1': 'interpolated'}
        assert set(materializer.origin_store['a1'].unique()) == {'interpolated'}
        np.testing.assert_allclose(features['a1'].to_numpy(), mirror.to_numpy())

    def test_rank_three_carries_from_previous_stage(self, reference_timeseries):
        """Rang 3 : l'imputation trimestrielle de a1 est REPORTÉE sur la grille M.

        Réponse à la première cause de B28 : a1 n'est plus laissée NaN deux
        mois sur trois à l'étape mensuelle.
        """
        quarterly_grid = _quarterly_grid()
        grid = _monthly_grid()
        materializer = CovariateMaterializer(covariate_strategy='model')
        quarterly = pd.Series(20.0 + np.arange(12), index=quarterly_grid)
        _prime_stores(materializer, 'a1', quarterly, 'Q', 'model')

        features, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        assert ways == {'a1': 'carried_model'}
        # Aucune cellule NaN : c'est exactement le défaut corrigé
        assert features['a1'].notna().all()
        # L'interpolation d'une valeur de modèle RESTE de modèle
        assert origins == {'a1': 'model'}
        assert set(materializer.origin_store['a1'].unique()) == {'model'}
        # Les cellules reportées sont écrites à la fréquence de l'étape courante
        assert set(materializer.imputed_freq_store['a1'].unique()) == {'M'}

    def test_rank_three_rescales_to_origin_stage_totals(self, reference_timeseries):
        """Rang 3 : le recalage porte sur les totaux de f', pas sur ceux de f."""
        quarterly_grid = _quarterly_grid()
        grid = _monthly_grid()

        # Applier espion : il enregistre ce sur quoi il est appelé
        class _RecordingApplier:
            def __init__(self):
                self.calls = []

            def rescale(self, values, observations, period_freq, column=None):
                self.calls.append((observations.copy(), period_freq, column))
                return values, values.notna()

        applier = _RecordingApplier()
        materializer = CovariateMaterializer(
            covariate_strategy='model', aggregation_constraint_applier=applier
        )
        quarterly = pd.Series(20.0 + np.arange(12), index=quarterly_grid)
        _prime_stores(materializer, 'a1', quarterly, 'Q', 'model')

        _features, ways, _origins = materializer.materialize(
            columns=['a1'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        assert ways == {'a1': 'carried_model'}
        assert len(applier.calls) == 1
        observations, period_freq, column = applier.calls[0]
        # Les totaux de l'étape d'origine, jamais ceux de l'étape courante
        assert period_freq == 'Q'
        # La colonne est transmise, pour la résolution d'une contrainte par colonne
        assert column == 'a1'
        assert observations.index.equals(quarterly_grid)
        # Ni les ancres annuelles de la source, ni la grille mensuelle
        np.testing.assert_allclose(observations.to_numpy(), quarterly.to_numpy())

    def test_rank_four_fallback_interpolate(self, reference_timeseries):
        """Rang 4, registres vides : covariate_fallback='interpolate'."""
        grid = _monthly_grid()
        materializer = CovariateMaterializer(
            covariate_strategy='model', covariate_fallback='interpolate'
        )

        features, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        assert ways == {'a1': 'interpolate'}
        assert origins == {'a1': 'interpolated'}
        assert features['a1'].notna().all()

    def test_rank_four_fallback_tolerate_nan(self, reference_timeseries):
        """Rang 4, registres vides : covariate_fallback='tolerate_nan'."""
        grid = _monthly_grid()
        materializer = CovariateMaterializer(
            covariate_strategy='model', covariate_fallback='tolerate_nan'
        )

        features, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        assert ways == {'a1': 'raw_anchors'}
        assert origins == {'a1': 'observed'}
        # Les trois ancres annuelles, et NaN partout ailleurs
        assert features['a1'].notna().sum() == 3

    def test_precedence_stops_at_first_applicable_rank(self, reference_timeseries):
        """Présente au pas f ET à un pas antérieur, la covariable prend le rang 2."""
        grid = _monthly_grid()
        quarterly_grid = _quarterly_grid()
        materializer = CovariateMaterializer(covariate_strategy='model')

        # Miroir mixte : 2021 imputée à Q, 2022-2023 imputées à M
        carried = pd.Series(99.0, index=quarterly_grid[quarterly_grid.year == 2021])
        stage = pd.Series(1.0, index=grid[grid.year >= 2022])
        values = pd.concat([carried, stage]).sort_index()
        freqs = pd.Series('M', index=values.index)
        freqs.loc[carried.index] = 'Q'
        materializer.imputed_store['a1'] = values
        materializer.imputed_freq_store['a1'] = freqs
        materializer.origin_store['a1'] = pd.Series(
            'model', index=values.index, dtype=object
        )

        features, ways, _origins = materializer.materialize(
            columns=['a1'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
        )

        # Arrêt au rang 2 : aucun report du miroir trimestriel de 2021
        assert ways == {'a1': 'stage_model'}
        assert set(features['a1'].loc['2022'].unique()) == {1.0}
        assert features['a1'].loc['2021'].isna().all()

    def test_model_way_degrades_per_entity(self, mixed_freq_panel_heterogeneous):
        """La voie est une propriété de colonne, l'applicabilité se mesure par entité."""
        data = mixed_freq_panel_heterogeneous
        dates = _monthly_grid()
        grid = _panel_grid(data, dates)
        materializer = CovariateMaterializer(covariate_strategy='model')

        # a1 imputée au pas M pour la seule entité FR
        french = pd.MultiIndex.from_product(
            [['FR'], dates], names=data.index.names
        )
        _prime_stores(
            materializer, 'a1',
            pd.Series(7.0 + np.arange(len(dates)), index=french), 'M', 'model',
        )

        features, ways, origins = materializer.materialize(
            columns=['a1'],
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
        )

        # Une seule voie pour la colonne, la plus dégradée des verdicts
        assert ways == {'a1': 'stage_model'}
        assert origins == {'a1': 'model'}
        # FR lit le miroir ; DE et IT dégradent au rang 4 (repli interpolé)
        np.testing.assert_allclose(
            features.loc['FR', 'a1'].to_numpy(), 7.0 + np.arange(len(dates))
        )
        assert features.loc['DE', 'a1'].iloc[0] == pytest.approx(120.0)
        assert features['a1'].notna().all()
        # Chaque entité porte l'origine de la voie qui l'a effectivement servie
        assert set(materializer.origin_store['a1'].unique()) == {
            'model', 'interpolated'
        }

    def test_ranks_two_and_three_unreachable_outside_model(self, reference_timeseries):
        """§8.1 : hors 'model', aucune stratégie ne consulte les registres."""
        grid = _monthly_grid()
        quarterly = pd.Series(20.0 + np.arange(12), index=_quarterly_grid())
        monthly = pd.Series(7.0 + np.arange(len(grid)), index=grid)

        for strategy in ('tolerate_nan', 'interpolate'):
            materializer = CovariateMaterializer(covariate_strategy=strategy)
            # Registres amorcés aux deux rangs : rien ne doit les lire
            _prime_stores(materializer, 'a1', quarterly, 'Q', 'model')
            _prime_stores(materializer, 'a2', monthly, 'M', 'model')

            ways = materializer.decide_ways(
                columns=['m1', 'q1', 'a1', 'a2'],
                grid_index=grid,
                stage_freq='M',
                detected_frequencies=TS_FREQUENCIES,
            )
            assert not {'stage_model', 'carried_model'} & set(ways.values()), strategy

            expected = 'raw_anchors' if strategy == 'tolerate_nan' else 'interpolate'
            assert ways == {
                'm1': 'identity', 'q1': expected, 'a1': expected, 'a2': expected
            }


class TestReferenceExampleSpec47:
    """§4.7 : les quatre lignes du tableau de référence, valeurs d'or comprises."""

    # Colonnes et grilles de l'exemple : imputation de a1 à l'étape Q
    COLUMNS = ['m1', 'q1', 'a2']

    def _fit_and_predict(self, materializer, data):
        """Materialize on the training grid, then replay on the prediction grid."""
        # L'étape est Q des deux côtés ; X_train est la grille Q masquée aux
        # 3 ancres annuelles de a1, X_pred la grille Q complète (12 lignes)
        x_train, ways, origins = materializer.materialize(
            columns=self.COLUMNS,
            grid_index=_annual_anchor_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=data,
        )
        x_pred, ways_pred, _ = materializer.materialize(
            columns=self.COLUMNS,
            grid_index=_quarterly_grid(),
            stage_freq='Q',
            detected_frequencies=TS_FREQUENCIES,
            source_data=data,
            materialization=ways,
        )
        # La voie est la même des deux côtés, par construction
        assert ways_pred == ways
        return x_train, x_pred, ways, origins

    def test_reference_example_of_spec_4_7(self, reference_timeseries):
        """Les quatre configurations du tableau : voie de a2 et taux de NaN."""
        data = reference_timeseries

        # Ligne 1 — 'tolerate_nan' : a2 à ses ancres des deux côtés. Le taux
        # brut diffère uniquement parce que les grilles n'ont pas le même pas
        tolerate = CovariateMaterializer(covariate_strategy='tolerate_nan')
        x_train, x_pred, ways, _origins = self._fit_and_predict(tolerate, data)
        assert ways == {'m1': 'aggregate', 'q1': 'identity', 'a2': 'raw_anchors'}
        assert x_train['a2'].isna().mean() == pytest.approx(0.0)
        assert x_pred['a2'].isna().mean() == pytest.approx(0.75)

        # Ligne 2 — 'interpolate' : a2 interpolée sur la grille Q
        interpolate = CovariateMaterializer(covariate_strategy='interpolate')
        x_train, x_pred, ways, origins = self._fit_and_predict(interpolate, data)
        assert ways['a2'] == 'interpolate'
        assert origins['a2'] == 'interpolated'
        assert x_train['a2'].isna().mean() == pytest.approx(0.0)
        assert x_pred['a2'].isna().mean() == pytest.approx(0.0)

        # Ligne 3 — 'model', ordre a2 avant a1 : a2 est son imputation Q (rang 2)
        ordered = CovariateMaterializer(covariate_strategy='model')
        _prime_stores(
            ordered, 'a2', pd.Series(5.0 + np.arange(12), index=_quarterly_grid()),
            'Q', 'model',
        )
        x_train, x_pred, ways, origins = self._fit_and_predict(ordered, data)
        assert ways['a2'] == 'stage_model'
        assert origins['a2'] == 'model'
        assert x_train['a2'].isna().mean() == pytest.approx(0.0)
        assert x_pred['a2'].isna().mean() == pytest.approx(0.0)

        # Ligne 4 — 'model', ordre a1 avant a2 : a2 servie par covariate_fallback
        # (rang 4), des deux côtés. C'est le cas qui produisait 33-67 % de NaN
        # silencieux dans hfi
        unordered = CovariateMaterializer(covariate_strategy='model')
        x_train, x_pred, ways, origins = self._fit_and_predict(unordered, data)
        assert ways['a2'] == 'interpolate'
        assert origins['a2'] == 'interpolated'
        assert x_train['a2'].isna().mean() == pytest.approx(0.0)
        assert x_pred['a2'].isna().mean() == pytest.approx(0.0)
        # Valeurs d'or de a2 aux trois ancres annuelles (§2.2)
        np.testing.assert_allclose(x_train['a2'].to_numpy(), [60.0, 66.0, 72.0])


class TestCovariateEligibility:
    """§4.5 : feature sans aucune observation pour la totalité d'une entité."""

    def test_covariate_eligibility_any_vs_all(self, mixed_freq_panel_heterogeneous):
        """'any_entity' retient climat_affaires ; 'all_entities' l'écarte."""
        data = mixed_freq_panel_heterogeneous
        columns = ['m1', 'q1', 'climat_affaires']

        any_entity = CovariateMaterializer(covariate_eligibility='any_entity')
        all_entities = CovariateMaterializer(covariate_eligibility='all_entities')

        # Recensement des entités qui n'observent jamais la colonne
        assert any_entity.entities_without_column('climat_affaires', data) == (('IT',),)
        assert any_entity.entities_without_column('m1', data) == ()

        # 'any_entity' retient la colonne, 'all_entities' l'écarte
        assert any_entity.eligible_columns(columns, data) == tuple(columns)
        assert all_entities.eligible_columns(columns, data) == ('m1', 'q1')

        # Sous 'any_entity', les lignes de l'entité vide restent NaN
        grid = _panel_grid(data, _monthly_grid())
        features, _ways, _origins = any_entity.materialize(
            columns=list(any_entity.eligible_columns(columns, data)),
            grid_index=grid,
            stage_freq='M',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
        )
        assert features.loc['IT', 'climat_affaires'].isna().all()
        assert features.loc['FR', 'climat_affaires'].notna().all()


class TestCentralInvariant:
    """§3 et §4.7 : l'invariant NaN, dans sa formulation testable D14."""

    @staticmethod
    def _filled_dates(frame: pd.DataFrame, column: str, entity=None) -> set:
        """Return the dates where ``column`` is filled, for one entity."""
        series = frame[column] if entity is None else frame.loc[entity, column]
        return set(series.dropna().index)

    def test_invariant_nan_by_entity(self, reference_timeseries, mixed_freq_panel_heterogeneous):
        """L'ensemble des dates renseignées dans X_pred CONTIENT celui de X_train.

        Formulation D14 du §4.7, et non une comparaison de pourcentages : sous
        'tolerate_nan' la grille d'entraînement (annuelle) et la grille de
        prédiction (trimestrielle) n'ont pas le même pas, et le taux brut
        diffère légitimement.
        """
        columns = ['m1', 'q1', 'a2']
        grid_train = _annual_grid()
        grid_pred = _quarterly_grid()

        for strategy in ('tolerate_nan', 'interpolate'):
            materializer = CovariateMaterializer(covariate_strategy=strategy)
            x_train, ways, _ = materializer.materialize(
                columns=columns,
                grid_index=grid_train,
                stage_freq='Y',
                detected_frequencies=TS_FREQUENCIES,
                source_data=reference_timeseries,
            )
            x_pred, ways_pred, _ = materializer.materialize(
                columns=columns,
                grid_index=grid_pred,
                stage_freq='Q',
                detected_frequencies=TS_FREQUENCIES,
                source_data=reference_timeseries,
                materialization=ways,
            )
            # La voie est la même des deux côtés
            assert ways_pred == ways

            # Inclusion des dates renseignées, colonne par colonne
            for column in columns:
                trained = self._filled_dates(x_train, column)
                predicted = self._filled_dates(x_pred, column)
                assert trained <= predicted, (strategy, column)

        # Même mesure sur le panel, PAR ENTITÉ : l'entité sans la colonne est
        # l'unique exception à l'invariant (§4.5)
        data = mixed_freq_panel_heterogeneous
        panel_columns = ['m1', 'q1', 'a2', 'climat_affaires']
        materializer = CovariateMaterializer(covariate_strategy='tolerate_nan')
        x_train, ways, _ = materializer.materialize(
            columns=panel_columns,
            grid_index=_panel_grid(data, grid_train),
            stage_freq='Y',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
        )
        x_pred, ways_pred, _ = materializer.materialize(
            columns=panel_columns,
            grid_index=_panel_grid(data, grid_pred),
            stage_freq='Q',
            detected_frequencies=PANEL_FREQUENCIES,
            source_data=data,
            materialization=ways,
        )
        assert ways_pred == ways

        for entity in ('FR', 'DE', 'IT'):
            for column in panel_columns:
                trained = self._filled_dates(x_train, column, entity)
                predicted = self._filled_dates(x_pred, column, entity)
                assert trained <= predicted, (entity, column)
        # L'entité italienne est vide des deux côtés, pas dégradée d'un côté
        assert x_train.loc['IT', 'climat_affaires'].isna().all()
        assert x_pred.loc['IT', 'climat_affaires'].isna().all()


class TestInitValidation:
    """§13.1 : __init__ valide sans transformer (B3)."""

    @pytest.mark.parametrize(
        'kwargs, message',
        [
            ({'covariate_strategy': 'nope'}, 'covariate_strategy'),
            ({'covariate_fallback': 'model'}, 'covariate_fallback'),
            ({'covariate_eligibility': 'some'}, 'covariate_eligibility'),
            ({'interpolation_method': {}}, 'cannot be empty'),
            ({'interpolation_anchor': 1.5}, r'\[0, 1\]'),
            ({'aggregation_constraint': 'median'}, 'aggregation_constraint must be one of'),
            ({'aggregation_constraint': {'a1': 'median'}}, 'aggregation_constraint values must be one of'),
            ({'aggregation_constraint': {}}, 'cannot be empty'),
        ],
    )
    def test_invalid_settings_raise(self, kwargs, message):
        """Chaque réglage hors contrat lève un ValueError explicite."""
        with pytest.raises(ValueError, match=message):
            CovariateMaterializer(**kwargs)

    def test_init_stores_parameters_verbatim(self):
        """Aucun paramètre n'est transformé à l'initialisation."""
        setting = {'a1': 'cubic'}
        materializer = CovariateMaterializer(interpolation_method=setting)
        assert materializer.interpolation_method is setting
        assert materializer.aggregation_constraint_applier is None
