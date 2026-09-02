"""Tests for tsforecast.frequency.stage_scaler.

Focus §9 (modalités 'constant'/'calendar', forme dict par feature), §9.2
(règle B25 et les trois diviseurs) et §5.4 (échelle par ligne, correctif B12)
de [SPEC] high_frequency_imputer2_architecture.md. Le composant est aussi un
transformer sklearn : fit gèle les trois diviseurs, transform divise,
inverse_transform remultiplie. Lot purement additif : hfi et son
`_covariate_scaling_divisors` restent intacts.
"""
# Modules de base
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

# Objets testés
from tsforecast.frequency.stage_scaler import (
    StageScaler,
    DEFAULT_SCALE_KEY,
    DEFAULT_SCALE_MODE,
)
from tsforecast.utils.frequency.converter import FrequencyConverter


# Fabrique de scaler : configuration par défaut cohérente, surchargeable
def _make_scaler(**overrides) -> StageScaler:
    """Build a StageScaler with sensible defaults for the tests."""
    params = dict(
        scale_features='constant',
        source_freq='Y',
        pred_freq='M',
        column_frequencies={'m1': 'M', 'q1': 'Q', 'a2': 'Y'},
    )
    params.update(overrides)
    return StageScaler(**params)


# Grille mensuelle de fin de mois, calquée sur le jeu TS du §2.2
def _monthly_grid(start: str = '2021-01-31', periods: int = 12) -> pd.DatetimeIndex:
    """Build a month-end grid."""
    return pd.date_range(start=start, periods=periods, freq='ME')


class TestConstantMode:
    """Modalité 'constant' : facteur invariant par couple de fréquences."""

    def test_constant_divisors_match_conversion_factor(self):
        """Les diviseurs valent exactement get_conversion_factor(étape, variable)."""
        scaler = StageScaler()
        converter = FrequencyConverter()

        # Variable annuelle prédite au mois
        assert scaler.target_divisor('a1', source_freq='Y', pred_freq='M') == 12.0
        # Variable trimestrielle prédite au mois
        assert scaler.target_divisor('q1', source_freq='Q', pred_freq='M') == 3.0
        # Variable mensuelle prédite au jour
        assert scaler.target_divisor('m1', source_freq='M', pred_freq='D') == 30.0

        # Contrôle du sens de la conversion : l'inverse est bien < 1
        assert scaler.target_divisor('m1', source_freq='M', pred_freq='Q') == pytest.approx(
            converter.get_conversion_factor('Q', 'M')
        )
        assert scaler.target_divisor('m1', source_freq='M', pred_freq='Q') < 1.0

    def test_constant_divisor_is_a_scalar_not_a_series(self):
        """Sous 'constant' et sans fréquence de production, le retour est scalaire."""
        scaler = _make_scaler()
        divisor = scaler.target_divisor(
            'a1', source_freq='Y', pred_freq='M', index=_monthly_grid()
        )
        assert isinstance(divisor, float)


class TestCalendarMode:
    """Modalité 'calendar' : décompte calendaire réel, une valeur par ligne."""

    def test_calendar_divisors_are_per_row(self):
        """Février vaut 28 ou 29, le premier trimestre 90 ou 91."""
        scaler = StageScaler(scale_features='calendar')

        # Étape journalière, variable mensuelle : février bissextile ou non
        dates = pd.to_datetime(['2021-02-15', '2024-02-15', '2021-01-15'])
        divisors = scaler.target_divisor(
            'm1', source_freq='M', pred_freq='D', index=dates
        )
        assert isinstance(divisors, pd.Series)
        assert len(divisors) == len(dates)
        pd.testing.assert_index_equal(divisors.index, dates)
        assert divisors.tolist() == [28.0, 29.0, 31.0]

        # Étape journalière, variable trimestrielle : T1 de 90 ou 91 jours
        quarters = pd.to_datetime(['2021-03-31', '2024-03-31'])
        quarterly = scaler.target_divisor(
            'q1', source_freq='Q', pred_freq='D', index=quarters
        )
        assert quarterly.tolist() == [90.0, 91.0]

    def test_calendar_without_index_raises(self):
        """Le décompte par ligne exige une grille : sans index, ValueError explicite."""
        scaler = StageScaler(scale_features='calendar')
        with pytest.raises(ValueError, match=r"calendar"):
            scaler.target_divisor('m1', source_freq='M', pred_freq='D')


class TestRuleB25:
    """Règle B25 : le diviseur d'une covariable dépend de sa ré-agrégation."""

    def test_b25_never_reaggregated_column_divides_by_one(self):
        """Une covariable jamais ré-agrégée vers f_var garde son échelle : 1.0."""
        scaler = StageScaler()
        # Covariable annuelle, variable trimestrielle, étape mensuelle :
        # Y n'est pas plus fine que Q, la colonne n'est jamais agrégée
        divisors = scaler.feature_divisors(
            columns=['a2'],
            column_frequencies={'a2': 'Y'},
            source_freq='Q',
            pred_freq='M',
        )
        assert divisors['a2'] == 1.0

    def test_b25_finer_column_uses_pred_freq(self):
        """Une covariable plus fine que l'étape porte f_stage = pred_freq."""
        scaler = StageScaler()
        # Covariable journalière, variable annuelle, étape mensuelle :
        # f_stage = M, diviseur = factor(M, Y) = 12
        divisors = scaler.feature_divisors(
            columns=['d1'],
            column_frequencies={'d1': 'D'},
            source_freq='Y',
            pred_freq='M',
        )
        assert divisors['d1'] == 12.0

    def test_b25_lower_column_uses_own_freq(self):
        """Une covariable plus basse que l'étape garde sa propre fréquence."""
        scaler = StageScaler()
        # Covariable trimestrielle, variable annuelle, étape mensuelle :
        # Q n'est pas plus fine que M, donc f_stage = Q, diviseur = factor(Q, Y) = 4
        divisors = scaler.feature_divisors(
            columns=['q1'],
            column_frequencies={'q1': 'Q'},
            source_freq='Y',
            pred_freq='M',
        )
        assert divisors['q1'] == 4.0

    def test_unknown_column_frequency_falls_back_on_default(self):
        """Une colonne sans fréquence détectée retombe sur le diviseur par défaut."""
        scaler = StageScaler(default_divisor=12.0)
        divisors = scaler.feature_divisors(
            columns=['inconnue'],
            column_frequencies={},
            source_freq='Y',
            pred_freq='M',
        )
        assert divisors['inconnue'] == 12.0


class TestB12ShortCircuit:
    """Correctif B12 : seul un scalaire strictement égal à 1.0 court-circuite."""

    def test_b12_series_of_ones_is_not_short_circuited(self):
        """Une Series de 1.0 passe par le chemin de mise à l'échelle, pas un scalaire 1.0."""
        scaler = StageScaler()
        index = _monthly_grid(periods=3)
        values = pd.Series([10.0, 20.0, 30.0], index=index)
        ones = pd.Series(1.0, index=index)

        # Le helper unique de décision
        assert StageScaler._is_identity(1.0) is True
        assert StageScaler._is_identity(ones) is False
        assert StageScaler._is_identity(pd.DataFrame({'a': ones})) is False

        # Le scalaire unitaire court-circuite : l'objet d'entrée est renvoyé tel quel
        assert scaler.apply(values, 1.0) is values
        assert scaler.invert(values, 1.0) is values

        # La Series unitaire ne court-circuite pas : un nouvel objet est produit
        scaled = scaler.apply(values, ones)
        assert scaled is not values
        pd.testing.assert_series_equal(scaled, values)

    def test_b12_series_of_ones_leaves_other_divisors_applied(self):
        """Le non-court-circuit vaut aussi pour une trame à diviseurs hétérogènes."""
        scaler = StageScaler()
        index = _monthly_grid(periods=2)
        frame = pd.DataFrame({'a': [12.0, 24.0], 'b': [3.0, 6.0]}, index=index)
        divisors = pd.Series({'a': 1.0, 'b': 3.0})

        scaled = scaler.apply(frame, divisors)
        assert scaled['a'].tolist() == [12.0, 24.0]
        assert scaled['b'].tolist() == [1.0, 2.0]


class TestPerRowTargetDivisor:
    """Échelle par ligne du §5.4 : chaque ligne porte sa fréquence de production."""

    def test_per_row_target_divisor(self):
        """Tableau chiffré du §5.4 : 120/Y -> 10.0, 28/Q -> 9.33, 30/Q -> 10.0."""
        scaler = StageScaler()
        index = pd.to_datetime(['2021-12-31', '2021-03-31', '2021-06-30'])
        y_train = pd.Series([120.0, 28.0, 30.0], index=index)
        produced_freq = pd.Series(['Y', 'Q', 'Q'], index=index)

        divisors = scaler.target_divisor(
            'a1', source_freq='Y', pred_freq='M', produced_freq=produced_freq
        )
        assert isinstance(divisors, pd.Series)
        assert divisors.tolist() == [12.0, 3.0, 3.0]

        scaled = scaler.apply(y_train, divisors)
        assert scaled.iloc[0] == pytest.approx(10.0, abs=1e-2)
        assert scaled.iloc[1] == pytest.approx(9.33, abs=1e-2)
        assert scaled.iloc[2] == pytest.approx(10.0, abs=1e-2)

    def test_per_row_divisor_is_always_a_series(self):
        """Même quand toutes les lignes partagent la fréquence, le retour reste une Series."""
        scaler = StageScaler()
        index = pd.to_datetime(['2021-12-31', '2022-12-31'])
        produced_freq = pd.Series(['Y', 'Y'], index=index)

        divisors = scaler.target_divisor(
            'a1', source_freq='Y', pred_freq='M', produced_freq=produced_freq
        )
        assert isinstance(divisors, pd.Series)
        assert divisors.tolist() == [12.0, 12.0]

    def test_per_row_divisor_under_calendar_mode(self):
        """Sous 'calendar', le diviseur par ligne suit le décompte réel de la période."""
        scaler = StageScaler(scale_features='calendar')
        index = pd.to_datetime(['2021-02-28', '2024-02-29'])
        produced_freq = pd.Series(['M', 'M'], index=index)

        divisors = scaler.target_divisor(
            'm1', source_freq='M', pred_freq='D', produced_freq=produced_freq
        )
        assert divisors.tolist() == [28.0, 29.0]


class TestDictForm:
    """Forme dict par feature, clé '__default__' et validation des clés."""

    def test_dict_form_and_default_key(self):
        """La clé propre prime sur '__default__', qui prime sur le défaut 'constant'."""
        scaler = _make_scaler(
            scale_features={'m1': 'calendar', DEFAULT_SCALE_KEY: 'constant'}
        )
        assert scaler.resolve_mode('m1') == 'calendar'
        assert scaler.resolve_mode('q1') == 'constant'

        # Sans clé de repli, une colonne non couverte retombe sur 'constant'
        partial = _make_scaler(scale_features={'m1': 'calendar'})
        assert partial.resolve_mode('q1') == DEFAULT_SCALE_MODE

        # Une seule colonne en 'calendar' suffit à basculer le retour en DataFrame
        index = _monthly_grid()
        divisors = scaler.feature_divisors(
            columns=['m1', 'q1'],
            column_frequencies={'m1': 'M', 'q1': 'Q'},
            source_freq='Y',
            pred_freq='M',
            index=index,
        )
        assert isinstance(divisors, pd.DataFrame)
        pd.testing.assert_index_equal(divisors.index, index)
        assert list(divisors.columns) == ['m1', 'q1']
        # 'm1' est agrégée : décompte calendaire des mois dans une année
        assert divisors['m1'].tolist() == [12.0] * len(index)
        # 'q1' reste en 'constant' : factor(Q, Y) = 4
        assert divisors['q1'].tolist() == [4.0] * len(index)

    def test_unknown_dict_key_raises_listing_columns(self):
        """Les clés absentes des colonnes réelles sont listées dans le ValueError."""
        scaler = _make_scaler(scale_features={'zz': 'calendar', 'yy': False, 'm1': 'constant'})
        with pytest.raises(ValueError) as excinfo:
            scaler.validate_columns(['m1', 'q1'])
        message = str(excinfo.value)
        assert 'zz' in message
        assert 'yy' in message
        # La colonne connue ne doit pas être dénoncée
        assert "'m1'" not in message

    def test_default_key_is_never_reported_as_unknown(self):
        """'__default__' n'est pas un nom de colonne et ne déclenche aucune erreur."""
        scaler = _make_scaler(scale_features={DEFAULT_SCALE_KEY: 'calendar'})
        scaler.validate_columns(['m1', 'q1'])

    def test_init_validates_without_transforming(self):
        """__init__ rejette une valeur illégale et stocke le paramètre tel que reçu."""
        with pytest.raises(ValueError, match=r"scale_features"):
            StageScaler(scale_features='exact')
        with pytest.raises(ValueError, match=r"scale_features"):
            StageScaler(scale_features={'m1': 'exact'})
        with pytest.raises(ValueError, match=r"empty"):
            StageScaler(scale_features={})

        setting = {'m1': 'calendar'}
        scaler = StageScaler(scale_features=setting)
        assert scaler.scale_features is setting


class TestScaleFeaturesFalse:
    """`False` ne divise rien : ni les features, ni la cible."""

    def test_scale_features_false_spares_y_too(self):
        """La cible est une colonne comme une autre : `False` la laisse intacte."""
        scaler = _make_scaler(scale_features=False)

        # Aucun diviseur sur les features
        divisors = scaler.feature_divisors(
            columns=['m1', 'q1', 'd1'],
            column_frequencies={'m1': 'M', 'q1': 'Q', 'd1': 'D'},
            source_freq='Y',
            pred_freq='M',
        )
        assert divisors.tolist() == [1.0, 1.0, 1.0]

        # La cible non plus n'est pas divisée, facteur cuit compris
        assert scaler.resolve_mode('a1') is False
        assert scaler.target_divisor('a1', source_freq='Y', pred_freq='M') == 1.0
        assert scaler.fit_scale_factor('a1', source_freq='Y', pred_freq='M') == 1.0

    def test_false_on_y_ignores_produced_frequencies(self):
        """Même mêlée de fréquences de production, `False` ne divise aucune ligne."""
        scaler = _make_scaler(scale_features=False)
        index = pd.to_datetime(['2021-12-31', '2021-03-31'])
        produced = pd.Series(['Y', 'Q'], index=index)

        divisor = scaler.target_divisor(
            'a1', source_freq='Y', pred_freq='M', produced_freq=produced
        )
        assert divisor == 1.0

    def test_false_on_y_only_through_the_dict_form(self):
        """Un dict peut dispenser la seule cible, les features restant divisées."""
        scaler = _make_scaler(
            scale_features={'a1': False, DEFAULT_SCALE_KEY: 'constant'}
        )
        divisors = scaler.feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': 'M'},
            source_freq='Y',
            pred_freq='M',
        )
        assert divisors['m1'] == 12.0
        assert scaler.target_divisor('a1', source_freq='Y', pred_freq='M') == 1.0

    def test_false_in_dict_form_only_spares_that_feature(self):
        """`False` dans un dict ne neutralise que la colonne visée."""
        scaler = _make_scaler(scale_features={'m1': False, DEFAULT_SCALE_KEY: 'constant'})
        divisors = scaler.feature_divisors(
            columns=['m1', 'q1'],
            column_frequencies={'m1': 'M', 'q1': 'Q'},
            source_freq='Y',
            pred_freq='M',
        )
        assert divisors['m1'] == 1.0
        assert divisors['q1'] == 4.0


class TestApplyInvert:
    """Routine unique d'application et d'inversion de l'échelle."""

    def test_apply_invert_roundtrip(self):
        """invert(apply(v, d), d) == v sur scalaire et sur Series."""
        scaler = StageScaler()
        index = _monthly_grid(periods=4)
        values = pd.Series([120.0, 132.0, 150.0, 90.0], index=index)

        # Diviseur scalaire
        pd.testing.assert_series_equal(
            scaler.invert(scaler.apply(values, 12.0), 12.0), values
        )

        # Diviseur par ligne
        divisors = pd.Series([12.0, 3.0, 4.0, 1.0], index=index)
        pd.testing.assert_series_equal(
            scaler.invert(scaler.apply(values, divisors), divisors), values
        )

        # Trame et diviseurs par colonne
        frame = pd.DataFrame({'a': values, 'b': values * 2})
        by_column = pd.Series({'a': 12.0, 'b': 4.0})
        pd.testing.assert_frame_equal(
            scaler.invert(scaler.apply(frame, by_column), by_column), frame
        )


class TestFitScaleFactor:
    """Facteur cuit dans le modèle : forme et invariance."""

    def test_fit_scale_factor_is_per_row_under_calendar(self):
        """Scalaire sous 'constant', Series sur la grille de prédiction sous 'calendar'."""
        index = pd.to_datetime(['2021-01-31', '2021-02-28'])

        constant = _make_scaler(scale_features='constant')
        assert constant.fit_scale_factor(
            'm1', source_freq='M', pred_freq='D', index=index
        ) == 30.0

        calendar = _make_scaler(scale_features='calendar')
        factor = calendar.fit_scale_factor(
            'm1', source_freq='M', pred_freq='D', index=index
        )
        assert isinstance(factor, pd.Series)
        pd.testing.assert_index_equal(factor.index, index)
        assert factor.tolist() == [31.0, 28.0]

    def test_fit_scale_factor_ignores_production_frequencies(self):
        """Le facteur cuit reste celui de l'étape, même quand y mêle des échelles."""
        scaler = StageScaler()
        index = pd.to_datetime(['2021-12-31', '2021-03-31'])
        produced_freq = pd.Series(['Y', 'Q'], index=index)

        baked = scaler.fit_scale_factor('a1', source_freq='Y', pred_freq='M')
        rows = scaler.target_divisor(
            'a1', source_freq='Y', pred_freq='M', produced_freq=produced_freq
        )
        assert baked == 12.0
        assert rows.tolist() == [12.0, 3.0]


class TestPanel:
    """Fréquences par entité : ventilation ligne à ligne quand elles divergent."""

    def _panel_index(self) -> pd.MultiIndex:
        """Build a two-entity month-end MultiIndex."""
        dates = _monthly_grid(periods=3)
        return pd.MultiIndex.from_product(
            [['DE', 'FR'], dates], names=['country', 'date']
        )

    def test_homogeneous_panel_keeps_the_compact_form(self):
        """Quand les entités s'accordent, le retour reste une Series par colonne."""
        scaler = StageScaler()
        divisors = scaler.feature_divisors(
            columns=['ip'],
            column_frequencies={'ip': {('FR',): 'M', ('DE',): 'M'}},
            source_freq='Y',
            pred_freq={('FR',): 'M', ('DE',): 'M'},
            index=self._panel_index(),
        )
        assert isinstance(divisors, pd.Series)
        assert divisors['ip'] == 12.0

    def test_panel_heterogeneous_frequencies_give_dataframe(self):
        """Une colonne mensuelle pour FR et trimestrielle pour DE se ventile par ligne."""
        scaler = StageScaler()
        index = self._panel_index()
        divisors = scaler.feature_divisors(
            columns=['ip'],
            column_frequencies={'ip': {('FR',): 'M', ('DE',): 'Q'}},
            source_freq='Y',
            pred_freq={('FR',): 'M', ('DE',): 'M'},
            index=index,
        )
        assert isinstance(divisors, pd.DataFrame)
        pd.testing.assert_index_equal(divisors.index, index)
        # DE : covariable trimestrielle, jamais agrégée à l'étape M -> factor(Q, Y) = 4
        assert divisors.loc['DE', 'ip'].tolist() == [4.0] * 3
        # FR : covariable mensuelle, agrégée -> factor(M, Y) = 12
        assert divisors.loc['FR', 'ip'].tolist() == [12.0] * 3

    def test_heterogeneous_stage_frequencies_give_per_row_target_divisor(self):
        """Des fréquences d'étape divergentes produisent un diviseur de y par ligne."""
        scaler = StageScaler()
        index = self._panel_index()
        divisors = scaler.target_divisor(
            'a1',
            source_freq='Y',
            pred_freq={('FR',): 'M', ('DE',): 'Q'},
            index=index,
        )
        assert isinstance(divisors, pd.Series)
        assert divisors.loc['DE'].tolist() == [4.0] * 3
        assert divisors.loc['FR'].tolist() == [12.0] * 3


class TestSklearnProtocol:
    """fit / transform / inverse_transform et conventions sklearn."""

    def _stage_frame(self) -> pd.DataFrame:
        """Build a small stage frame with two covariates."""
        index = _monthly_grid(periods=3)
        return pd.DataFrame(
            {'m1': [1200.0, 2400.0, 3600.0], 'q1': [40.0, 80.0, 120.0]},
            index=index,
        )

    def test_fit_stores_the_three_divisors(self):
        """fit gèle les diviseurs de covariables, de y et le facteur cuit."""
        X = self._stage_frame()
        scaler = _make_scaler(column_frequencies={'m1': 'M', 'q1': 'Q'}).fit(X)

        assert scaler.feature_divisors_.to_dict() == {'m1': 12.0, 'q1': 4.0}
        assert scaler.target_divisor_ == 12.0
        assert scaler.fit_scale_factor_ == 12.0
        assert scaler.n_features_in_ == 2
        assert list(scaler.feature_names_in_) == ['m1', 'q1']

    def test_transform_divides_and_inverse_transform_restores(self):
        """transform divise la trame, inverse_transform la restitue exactement."""
        X = self._stage_frame()
        scaler = _make_scaler(column_frequencies={'m1': 'M', 'q1': 'Q'}).fit(X)

        scaled = scaler.transform(X)
        assert scaled['m1'].tolist() == [100.0, 200.0, 300.0]
        assert scaled['q1'].tolist() == [10.0, 20.0, 30.0]
        pd.testing.assert_frame_equal(scaler.inverse_transform(scaled), X)

    def test_transform_on_a_series_uses_the_target_divisor(self):
        """Une Series est traitée comme la cible, une trame comme les covariables."""
        X = self._stage_frame()
        y = pd.Series([120.0, 132.0, 150.0], index=X.index)
        scaler = _make_scaler(column_frequencies={'m1': 'M', 'q1': 'Q'}).fit(X, y)

        scaled = scaler.transform(y)
        assert scaled.tolist() == [10.0, 11.0, 12.5]
        pd.testing.assert_series_equal(scaler.inverse_transform(scaled), y)

    def test_fit_accepts_production_frequencies(self):
        """produced_freq alimente target_divisor_ sans toucher à fit_scale_factor_."""
        X = self._stage_frame()
        y = pd.Series([120.0, 28.0, 30.0], index=X.index)
        produced_freq = pd.Series(['Y', 'Q', 'Q'], index=X.index)
        scaler = _make_scaler(column_frequencies={'m1': 'M', 'q1': 'Q'}).fit(
            X, y, produced_freq=produced_freq
        )

        assert scaler.fit_scale_factor_ == 12.0
        assert scaler.target_divisor_.tolist() == [12.0, 3.0, 3.0]
        assert scaler.transform(y).iloc[1] == pytest.approx(9.33, abs=1e-2)

    def test_fit_reads_the_imputed_column_from_the_name_of_y(self):
        """Le nom de `y` désigne à lui seul la colonne imputée."""
        X = self._stage_frame()
        # Cible nommée 'a1', dispensée d'échelle par le dict
        y = pd.Series([120.0, 132.0, 150.0], index=X.index, name='a1')
        scaler = _make_scaler(
            scale_features={'a1': False, DEFAULT_SCALE_KEY: 'constant'},
            column_frequencies={'m1': 'M', 'q1': 'Q'},
        ).fit(X, y)

        # La cible garde son échelle, les covariables sont bien divisées
        assert scaler.target_divisor_ == 1.0
        assert scaler.fit_scale_factor_ == 1.0
        assert scaler.feature_divisors_.to_dict() == {'m1': 12.0, 'q1': 4.0}
        pd.testing.assert_series_equal(scaler.transform(y), y)

    def test_unnamed_y_falls_back_on_the_global_setting(self):
        """Une cible anonyme ne résout aucune entrée : c'est le réglage global."""
        X = self._stage_frame()
        y = pd.Series([120.0, 132.0, 150.0], index=X.index)
        # L'entrée 'a1' ne peut plus être atteinte, seul '__default__' compte
        scaler = _make_scaler(
            scale_features={'a1': False, DEFAULT_SCALE_KEY: 'constant'},
            column_frequencies={'m1': 'M', 'q1': 'Q'},
        ).fit(X, y)

        assert scaler.target_divisor_ == 12.0

    def test_transform_before_fit_raises(self):
        """transform hors ajustement lève NotFittedError."""
        with pytest.raises(NotFittedError):
            _make_scaler().transform(self._stage_frame())

    def test_fit_without_frequencies_raises(self):
        """Les métadonnées de fréquence sont obligatoires au fit."""
        with pytest.raises(ValueError, match=r"source_freq"):
            StageScaler().fit(self._stage_frame())

    def test_get_params_and_clone(self):
        """Le composant respecte le protocole d'estimateur : get_params et clone."""
        scaler = _make_scaler(scale_features={'m1': 'calendar'})
        params = scaler.get_params()
        assert params['scale_features'] == {'m1': 'calendar'}
        assert params['source_freq'] == 'Y'

        twin = clone(scaler)
        assert twin.get_params()['pred_freq'] == 'M'
        assert not hasattr(twin, 'feature_divisors_')

    def test_transform_rejects_other_types(self):
        """Un type autre que Series ou DataFrame lève TypeError."""
        X = self._stage_frame()
        scaler = _make_scaler(column_frequencies={'m1': 'M', 'q1': 'Q'}).fit(X)
        with pytest.raises(TypeError):
            scaler.transform(np.arange(3))


class TestStatelessness:
    """Le scaler ne mémorise rien entre deux étapes hors sa configuration."""

    def test_scaler_is_stateless_across_stages(self):
        """Une même instance sert deux étapes et rend les mêmes résultats que deux neuves."""
        shared = StageScaler()

        first = shared.feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': 'M'},
            source_freq='Y',
            pred_freq='M',
        )
        second = shared.feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': 'M'},
            source_freq='Q',
            pred_freq='M',
        )
        # Le second appel ne doit rien devoir au premier
        third = shared.feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': 'M'},
            source_freq='Y',
            pred_freq='M',
        )

        assert first['m1'] == 12.0
        assert second['m1'] == 3.0
        assert third['m1'] == 12.0
        assert StageScaler().feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': 'M'},
            source_freq='Q',
            pred_freq='M',
        )['m1'] == second['m1']

    def test_divisor_methods_need_no_fit(self):
        """Les trois diviseurs sont utilisables sans ajustement préalable."""
        scaler = StageScaler()
        assert scaler.target_divisor('a1', source_freq='Y', pred_freq='M') == 12.0
        assert scaler.fit_scale_factor('a1', source_freq='Y', pred_freq='M') == 12.0
        assert scaler.feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': 'M'},
            source_freq='Y',
            pred_freq='M',
        )['m1'] == 12.0


class TestHfiUntouched:
    """Lot purement additif : la logique d'échelle de hfi reste en place."""

    def test_hfi_still_carries_its_own_divisors(self):
        """`_covariate_scaling_divisors` et `_covariate_divisor` existent toujours."""
        from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer

        assert hasattr(HighFrequencyImputer, '_covariate_scaling_divisors')
        assert hasattr(HighFrequencyImputer, '_covariate_divisor')
        assert hasattr(HighFrequencyImputer, '_apply_frequency_scaling')
        assert hasattr(HighFrequencyImputer, '_stage_scale_factor')
