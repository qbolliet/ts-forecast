"""Tests for tsforecast.frequency.covariate_materializer.

Couvre §4.1 (classification identité / agrégation / stratégie), §4.2
('tolerate_nan'), §4.3 ('interpolate' et le repli), §4.5
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

        # 4. Une prédiction de modèle vaut 'model', report d'étape compris ;
        # ces deux voies relèvent du lot suivant et sont refusées ici
        assert _WAY_ORIGIN['stage_model'] == 'model'
        assert _WAY_ORIGIN['carried_model'] == 'model'
        with pytest.raises(NotImplementedError, match=r"stage_model"):
            materializer.materialize(
                columns=['a1'],
                grid_index=grid,
                stage_freq='Q',
                detected_frequencies=TS_FREQUENCIES,
                source_data=reference_timeseries,
                materialization={'a1': 'stage_model'},
            )

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
            ({'aggregation_constraint': 'mean'}, 'reserved for a later extension'),
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
