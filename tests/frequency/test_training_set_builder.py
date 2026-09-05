"""Tests for tsforecast.frequency.training_set_builder.

Couvre §5.8 EN ENTIER (les six règles R1 à R6, l'exemple chiffré sur le jeu
`PANEL-F`), §1.5 (défaut B29), §2.5 (le jeu `PANEL-F`), §5.3 (filtre
d'origine), §5.4 (échelle par ligne) et §4.6 (unicité de la voie) de [SPEC]
high_frequency_imputer2_architecture.md, ainsi que les décisions D17, D18 et
D19.

Les jeux sont ceux du document : `mixed_freq_panel_multifrequency` (§2.5) et
`reference_timeseries` (§2.2). Aucun estimateur n'est nécessaire : le
composant est testable seul. Lot purement additif : `hfi` et ses tests restent
intacts.
"""
# Modules de base
import numpy as np
import pandas as pd
import pytest

# Objets testés
from tsforecast.frequency.training_set_builder import (
    TrainingSet,
    TrainingSetBuilder,
)
from tsforecast.frequency.covariate_materializer import CovariateMaterializer
from tsforecast.frequency.stage_scaler import StageScaler
from tsforecast.utils.frequency.converter import FrequencyConverter


# Fréquences détectées du jeu PANEL-F (§2.5) : `v` est une propriété du couple
# (entité, colonne), jamais de la seule colonne
PANEL_F_FREQUENCIES = {
    'm1': 'M',
    'q1': 'Q',
    'v': {('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'},
}
# Fréquences détectées du jeu TS (§2.2)
TS_FREQUENCIES = {'m1': 'M', 'q1': 'Q', 'a1': 'Y', 'a2': 'Y'}

# Covariables des jeux d'entraînement testés
FEATURES = ['m1', 'q1']


# Fabrique de composant : matérialiseur réel, aucun estimateur
def _builder(**materializer_kwargs) -> TrainingSetBuilder:
    """Build a TrainingSetBuilder over a real materializer."""
    materializer = CovariateMaterializer(**materializer_kwargs)
    return TrainingSetBuilder(materializer)


# Fabrique d'appel : les paramètres communs à tous les tests du panel
def _build_panel_f(builder, data, stage_freq='M', eligible_origins=('observed',),
                   frequencies=None, column='v', feature_cols=None):
    """Compose the mutualized training set of ``v`` on ``PANEL-F``."""
    return builder.build(
        column=column,
        feature_cols=FEATURES if feature_cols is None else feature_cols,
        stage_freq=stage_freq,
        detected_frequencies=PANEL_F_FREQUENCIES if frequencies is None else frequencies,
        source_data=data,
        eligible_origins=set(eligible_origins),
    )


# Mise à l'échelle de la cible par le SEUL chemin d'échelle du package
def _scaled_target(training: TrainingSet, stage: str) -> pd.Series:
    """Apply ``StageScaler.target_divisor(produced_freq=...)`` to a raw target."""
    scaler = StageScaler()
    divisors = scaler.target_divisor(
        training.y.name,
        source_freq=dict(training.blocks),
        pred_freq=stage,
        index=training.y.index,
        produced_freq=training.row_frequency,
    )
    return scaler.apply(training.y, divisors)


# Comparaison de deux captures des trois registres
def _snapshots_equal(left, right) -> bool:
    """Tell whether two materializer snapshots hold the same content."""
    if sorted(left) != sorted(right):
        return False
    for store in left:
        if sorted(left[store]) != sorted(right[store]):
            return False
        for column, series in left[store].items():
            if not series.equals(right[store][column]):
                return False
    return True


class TestBlocksAndRows:
    """R1, R2 et R3 : périmètre, fréquence de bloc et lignes (§5.8)."""

    def test_blocks_on_panel_f_at_monthly_stage(self, mixed_freq_panel_multifrequency):
        """Chaque entité contribue à SA fréquence pour la colonne (R2)."""
        training = _build_panel_f(_builder(), mixed_freq_panel_multifrequency)

        assert dict(training.blocks) == {('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'}

    def test_row_counts_on_panel_f_at_monthly_stage(self, mixed_freq_panel_multifrequency):
        """51 lignes — 3 FR, 12 DE, 36 IT — contre 3 sans mutualisation (§5.8)."""
        training = _build_panel_f(_builder(), mixed_freq_panel_multifrequency)

        assert len(training) == 51
        assert training.y.groupby(level=0).size().to_dict() == {
            'DE': 12, 'FR': 3, 'IT': 36
        }
        # Les features couvrent exactement la grille mutualisée
        assert training.X.index.equals(training.y.index)
        assert list(training.X.columns) == FEATURES

    def test_scaled_target_on_panel_f_at_monthly_stage(self, mixed_freq_panel_multifrequency):
        """Les trois blocs se superposent à la même échelle mensuelle (§5.8)."""
        training = _build_panel_f(_builder(), mixed_freq_panel_multifrequency)
        scaled = _scaled_target(training, 'M')

        # FR : ancres annuelles divisées par 12
        assert scaled.loc['FR'].tolist() == [10.0, 11.0, 12.5]
        # IT : bloc à l'échelle de l'étape, diviseur 1.0 (identité)
        assert sorted(set(scaled.loc['IT'].round(6))) == [10.0, 11.0, 12.5]
        # DE : ancres trimestrielles divisées par 3
        expected_de = [
            value / 3.0 for value in
            [28, 30, 31, 31, 31, 33, 34, 34, 36, 37, 38, 39]
        ]
        np.testing.assert_allclose(scaled.loc['DE'].to_numpy(), expected_de)
        assert round(float(scaled.loc[('DE', '2021-03-31')]), 3) == 9.333

    def test_row_frequency_and_origin_of_observed_rows(self, mixed_freq_panel_multifrequency):
        """Une ligne observée porte la fréquence de son bloc (R5)."""
        training = _build_panel_f(_builder(), mixed_freq_panel_multifrequency)

        assert set(training.row_origin) == {'observed'}
        assert set(training.row_frequency.loc['FR']) == {'Y'}
        assert set(training.row_frequency.loc['DE']) == {'Q'}
        assert set(training.row_frequency.loc['IT']) == {'M'}


class TestQuarterlyStage:
    """D18 : les blocs ne dépendent pas de l'étape, les diviseurs si (§5.8)."""

    def test_blocks_and_rows_at_quarterly_stage(self, mixed_freq_panel_multifrequency):
        """Mêmes blocs et mêmes 51 lignes à l'étape Q, diviseurs différents."""
        training = _build_panel_f(
            _builder(), mixed_freq_panel_multifrequency, stage_freq='Q'
        )

        assert dict(training.blocks) == {('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'}
        assert len(training) == 51

        scaled = _scaled_target(training, 'Q')
        # FR : diviseur 4
        assert scaled.loc['FR'].tolist() == [30.0, 33.0, 37.5]
        # DE : diviseur 1.0, les 12 valeurs brutes inchangées
        assert scaled.loc['DE'].tolist() == [
            28.0, 30.0, 31.0, 31.0, 31.0, 33.0, 34.0, 34.0, 36.0, 37.0, 38.0, 39.0
        ]
        # IT : diviseur fractionnaire 1/3, ses 36 lignes mensuelles conservées
        assert len(scaled.loc['IT']) == 36
        assert sorted(set(scaled.loc['IT'].round(6))) == [30.0, 33.0, 37.5]

    def test_italian_rows_stay_monthly_at_quarterly_stage(
        self, mixed_freq_panel_multifrequency, monkeypatch
    ):
        """Le bloc IT porte 36 lignes de fin de MOIS, jamais 12 agrégats (R3)."""
        # Compteur d'agrégations : la cible ne doit JAMAIS y passer
        aggregated_columns = []
        original = FrequencyConverter.aggregate_to_lower_frequency

        def _spy(self, series, target_freq, *args, **kwargs):
            aggregated_columns.append(getattr(series, 'name', None))
            return original(self, series, target_freq, *args, **kwargs)

        monkeypatch.setattr(
            FrequencyConverter, 'aggregate_to_lower_frequency', _spy
        )

        training = _build_panel_f(
            _builder(), mixed_freq_panel_multifrequency, stage_freq='Q'
        )

        # Lignes italiennes : 36 fins de mois, dont des mois non trimestriels
        italian_dates = training.y.loc['IT'].index
        assert len(italian_dates) == 36
        assert not set(italian_dates.month).issubset({3, 6, 9, 12})
        # Cible BRUTE : c'est le diviseur qui porte l'échelle, pas l'agrégation
        assert sorted(set(training.y.loc['IT'].round(6))) == [10.0, 11.0, 12.5]
        assert 'v' not in aggregated_columns

    def test_blocks_are_stage_independent(self, mixed_freq_panel_multifrequency):
        """Mêmes blocs et même compte de lignes aux étapes Q et M (D18)."""
        builder = _builder()
        monthly = _build_panel_f(
            builder, mixed_freq_panel_multifrequency, stage_freq='M'
        )
        quarterly = _build_panel_f(
            builder, mixed_freq_panel_multifrequency, stage_freq='Q'
        )

        assert dict(monthly.blocks) == dict(quarterly.blocks)
        assert len(monthly) == len(quarterly) == 51
        assert monthly.y.index.equals(quarterly.y.index)
        # Seuls les diviseurs diffèrent
        assert _scaled_target(monthly, 'M').loc['FR'].tolist() == [10.0, 11.0, 12.5]
        assert _scaled_target(quarterly, 'Q').loc['FR'].tolist() == [30.0, 33.0, 37.5]


class TestSingleMaterialization:
    """R4 : un seul appel au matérialiseur, et aucun registre écrit."""

    def test_single_call_to_materialize(self, mixed_freq_panel_multifrequency):
        """`materialize` est appelé une fois, sur les blocs, sans enregistrer."""
        builder = _builder()
        calls = []
        original = builder.materializer.materialize

        def _spy(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        builder.materializer.materialize = _spy
        training = _build_panel_f(builder, mixed_freq_panel_multifrequency)

        assert len(calls) == 1
        assert calls[0]['stage_freq'] == dict(training.blocks)
        assert calls[0]['record'] is False
        assert calls[0]['grid_index'].equals(training.y.index)

    def test_stores_untouched_by_training_materialization(
        self, mixed_freq_panel_multifrequency
    ):
        """Les trois registres sont identiques avant et après `build`."""
        builder = _builder(covariate_strategy='model')
        before = builder.materializer.snapshot()

        _build_panel_f(builder, mixed_freq_panel_multifrequency)

        assert _snapshots_equal(before, builder.materializer.snapshot())

    def test_no_carry_phantom_after_build(self, mixed_freq_panel_multifrequency):
        """Aucun rang 3 ne devient éligible du seul fait du `build` (R4)."""
        builder = _builder(covariate_strategy='model')

        _build_panel_f(builder, mixed_freq_panel_multifrequency)

        # Le miroir reste vide : rien n'a jamais été imputé
        assert builder.materializer.imputed_store == {}
        # Et la classification ne peut donc pas conclure au report d'étape
        for entity in (('FR',), ('DE',), ('IT',)):
            way = builder.materializer.classify(
                'q1', 'M', PANEL_F_FREQUENCIES, entity
            )
            assert way != 'carried_model'


class TestScopeOfContributors:
    """R1 : qui contribue, et qui ne contribue rien sans lever."""

    def test_truncated_entity_keeps_all_its_observations(
        self, mixed_freq_panel_multifrequency
    ):
        """Un trimestre incomplet ne fait perdre AUCUNE ligne (R3, §4.1)."""
        data = mixed_freq_panel_multifrequency.copy()
        # Troncature des deux derniers mois d'IT : le dernier trimestre est
        # incomplet, ce que `full_periods_only` sanctionnerait sur une agrégation
        last_two = data.loc['IT'].index[-2:]
        data.loc[[('IT', date) for date in last_two], 'v'] = np.nan

        training = _build_panel_f(_builder(), data, stage_freq='Q')

        assert len(training.y.loc['IT']) == 34
        assert len(training) == 3 + 12 + 34

    def test_entity_without_the_column_contributes_nothing(
        self, mixed_freq_panel_multifrequency
    ):
        """Une entité n'observant jamais la colonne ne lève rien (§4.5)."""
        data = mixed_freq_panel_multifrequency.copy()
        data.loc['DE', 'v'] = np.nan

        training = _build_panel_f(_builder(), data)

        assert dict(training.blocks) == {('FR',): 'Y', ('IT',): 'M'}
        assert 'DE' not in training.y.index.get_level_values(0)
        assert len(training) == 39

    def test_entity_with_other_target_frequency_still_contributes(
        self, mixed_freq_panel_multifrequency
    ):
        """La fréquence cible propre d'une entité ne change rien (R1)."""
        # IT porte une fréquence cible distincte des deux autres entités
        stage_freq = {('FR',): 'M', ('DE',): 'M', ('IT',): 'Q'}

        training = _build_panel_f(
            _builder(), mixed_freq_panel_multifrequency, stage_freq=stage_freq
        )

        assert dict(training.blocks) == {('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'}
        assert len(training.y.loc['IT']) == 36
        assert len(training) == 51

    def test_column_absent_from_the_data_raises(self, mixed_freq_panel_multifrequency):
        """Une colonne absente du jeu est une erreur de branchement."""
        with pytest.raises(KeyError, match='missing from source_data'):
            _build_panel_f(
                _builder(), mixed_freq_panel_multifrequency, column='absente'
            )

    def test_no_entity_observes_the_column(self, mixed_freq_panel_multifrequency):
        """Aucun contributeur : jeu vide, sans levée."""
        data = mixed_freq_panel_multifrequency.copy()
        data['v'] = np.nan

        training = _build_panel_f(_builder(), data)

        assert dict(training.blocks) == {}
        assert len(training) == 0
        assert list(training.X.columns) == FEATURES


class TestOriginFilter:
    """§5.3 : le filtre d'origine, orthogonal à la mutualisation."""

    # Amorçage manuel des registres, comme l'aurait fait une étape antérieure
    @staticmethod
    def _seed_stores(materializer, dates, values, freq, origin):
        """Seed the three stores with French cells produced at an earlier stage."""
        index = pd.MultiIndex.from_tuples(
            [('FR', date) for date in pd.to_datetime(dates)],
            names=['country', 'date'],
        )
        materializer.imputed_store['v'] = pd.Series(values, index=index, dtype=float)
        materializer.imputed_freq_store['v'] = pd.Series(freq, index=index, dtype=object)
        materializer.origin_store['v'] = pd.Series(origin, index=index, dtype=object)

    def test_eligible_origins_filter(self, mixed_freq_panel_multifrequency):
        """`{'observed'}` exclut les cellules imputées, un jeu élargi les admet."""
        builder = _builder()
        self._seed_stores(
            builder.materializer,
            ['2021-03-31', '2021-06-30', '2021-09-30'],
            [28.0, 30.0, 31.0],
            'Q',
            'model',
        )

        strict = _build_panel_f(
            builder, mixed_freq_panel_multifrequency, eligible_origins={'observed'}
        )
        assert len(strict) == 51
        assert set(strict.row_origin) == {'observed'}

        widened = _build_panel_f(
            builder,
            mixed_freq_panel_multifrequency,
            eligible_origins={'observed', 'interpolated', 'model'},
        )
        assert len(widened) == 54
        # R5 : ces lignes entrent avec la fréquence lue dans imputed_freq_store
        model_rows = widened.row_origin == 'model'
        assert int(model_rows.sum()) == 3
        assert set(widened.row_frequency[model_rows.to_numpy()]) == {'Q'}
        assert widened.y[model_rows.to_numpy()].tolist() == [28.0, 30.0, 31.0]

    def test_interpolated_rows_excluded_under_covariates_only_of_a_model_cell(
        self, mixed_freq_panel_multifrequency
    ):
        """Sous `{'observed', 'interpolated'}`, une cellule 'model' reste dehors."""
        builder = _builder()
        self._seed_stores(
            builder.materializer, ['2021-03-31'], [28.0], 'Q', 'model'
        )

        training = _build_panel_f(
            builder,
            mixed_freq_panel_multifrequency,
            eligible_origins={'observed', 'interpolated'},
        )

        assert len(training) == 51
        assert set(training.row_origin) == {'observed'}

    def test_imputed_rows_scale_by_their_own_production_frequency(
        self, mixed_freq_panel_multifrequency
    ):
        """Le diviseur d'une ligne imputée est celui de SA fréquence (§5.4)."""
        builder = _builder()
        self._seed_stores(
            builder.materializer, ['2021-03-31'], [28.0], 'Q', 'model'
        )

        training = _build_panel_f(
            builder,
            mixed_freq_panel_multifrequency,
            eligible_origins={'observed', 'interpolated', 'model'},
        )
        scaled = _scaled_target(training, 'M')

        # 28 produit au pas trimestriel : diviseur 3, et non le scalaire annuel
        assert round(float(scaled.loc[('FR', '2021-03-31')]), 3) == 9.333
        assert float(scaled.loc[('FR', '2021-12-31')]) == 10.0


class TestTrainingWindow:
    """§7.2 : le masque 'training' est lu à la fréquence du bloc."""

    def test_window_restricts_each_block_at_its_own_frequency(
        self, mixed_freq_panel_multifrequency
    ):
        """Le masque est demandé une fois, aux fréquences de bloc."""
        asked = []

        def _mask(frequencies):
            asked.append(dict(frequencies))
            # Fenêtre limitée à 2021 et 2022, à la fréquence de chaque bloc
            index = mixed_freq_panel_multifrequency.index
            return pd.Series(
                index.get_level_values(-1).year < 2023, index=index
            )

        builder = TrainingSetBuilder(CovariateMaterializer(), training_mask=_mask)
        training = _build_panel_f(builder, mixed_freq_panel_multifrequency)

        assert asked == [{('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'}]
        assert len(training.y.loc['FR']) == 2
        assert len(training.y.loc['DE']) == 8
        assert len(training.y.loc['IT']) == 24

    def test_entity_absent_from_the_mask_is_unrestricted(
        self, mixed_freq_panel_multifrequency
    ):
        """Une entité sans masque ajusté n'est pas restreinte (§7.2)."""
        def _mask(frequencies):
            # Masque ne couvrant que FR, comme le calculateur le rend lorsque
            # les autres entités n'ont pas de masque valide
            index = mixed_freq_panel_multifrequency.loc[['FR']].index
            return pd.Series(
                index.get_level_values(-1).year < 2023, index=index
            )

        builder = TrainingSetBuilder(CovariateMaterializer(), training_mask=_mask)
        training = _build_panel_f(builder, mixed_freq_panel_multifrequency)

        assert len(training.y.loc['FR']) == 2
        assert len(training.y.loc['DE']) == 12
        assert len(training.y.loc['IT']) == 36


class TestDegenerateCases:
    """I16 et panel à une entité : la mutualisation est neutre au singulier."""

    def test_timeseries_is_the_degenerate_case(self, reference_timeseries):
        """Sur `TS`, le jeu mutualisé est identique au jeu d'origine (I16)."""
        builder = _builder()
        training = builder.build(
            column='a1',
            feature_cols=FEATURES,
            stage_freq='M',
            detected_frequencies=TS_FREQUENCIES,
            source_data=reference_timeseries,
            eligible_origins={'observed'},
        )

        assert dict(training.blocks) == {(): 'Y'}
        assert len(training) == 3
        assert training.y.tolist() == [120.0, 132.0, 150.0]
        assert isinstance(training.y.index, pd.DatetimeIndex)

        # Diviseur 12 à l'étape M : tous les chiffres des §4.7, §5.4 et §5.5
        # restent vrais
        scaler = StageScaler()
        divisors = scaler.target_divisor(
            'a1', source_freq='Y', pred_freq='M',
            produced_freq=training.row_frequency,
        )
        assert divisors.tolist() == [12.0, 12.0, 12.0]
        assert scaler.apply(training.y, divisors).tolist() == [10.0, 11.0, 12.5]

    def test_single_entity_panel(self, mixed_freq_panel_multifrequency, reference_timeseries):
        """Un panel à une entité rend le même jeu qu'une série temporelle."""
        data = mixed_freq_panel_multifrequency.loc[['FR']]

        training = _build_panel_f(_builder(), data)

        assert dict(training.blocks) == {('FR',): 'Y'}
        assert len(training) == 3
        assert training.y.tolist() == reference_timeseries['a1'].dropna().tolist()
        assert _scaled_target(training, 'M').tolist() == [10.0, 11.0, 12.5]


class TestWayContract:
    """§4.6 : les voies rendues sont celles imposées à la grille de prédiction."""

    def test_way_uniqueness_contract(self, mixed_freq_panel_multifrequency):
        """Les voies couvrent exactement `feature_cols` et sont rejouables."""
        builder = _builder()
        training = _build_panel_f(builder, mixed_freq_panel_multifrequency)

        assert set(training.ways) == set(FEATURES)
        assert set(training.column_origins) == set(FEATURES)

        # Rejeu tel quel sur une autre grille : celle de prédiction de FR
        prediction_grid = mixed_freq_panel_multifrequency.loc[['FR']].index
        _features, replayed, _origins = builder.materializer.materialize(
            columns=tuple(FEATURES),
            grid_index=prediction_grid,
            stage_freq='M',
            detected_frequencies=PANEL_F_FREQUENCIES,
            source_data=mixed_freq_panel_multifrequency,
            materialization=dict(training.ways),
            record=False,
        )
        assert replayed == dict(training.ways)
