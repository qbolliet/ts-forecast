"""Tests ciblés pour ImputationWindowCalculator (cf. high_frequency_imputer_review.md §3, §7).

Chaque test épingle un comportement précis identifié dans la revue. Les tests
marqués ``xfail(strict=True)`` documentent le comportement souhaité (pas le
comportement actuel bogué) et référencent la section de la revue concernée.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from tsforecast.utils.frequency.utils import detect_index_frequency
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer
from tsforecast.frequency.imputation_window import ImputationWindowCalculator
from tsforecast.panel.utils import get_unique_panel_entities, split_variable_key


# Fonction auxiliaire de construction d'un panel mixte à N niveaux d'entité
def _make_panel(entities, n_periods=36):
    """Build a two-column monthly panel indexed by the given entity tuples.

    Args:
        entities: List of entity tuples, all of the same length.
        n_periods: Number of monthly dates per entity.

    Returns:
        Panel DataFrame with a MultiIndex (entity levels..., date).
    """
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='MS')
    n_levels = len(entities[0])
    # Construction du MultiIndex (niveaux d'entité + date)
    tuples = [(*entity, date) for entity in entities for date in dates]
    names = [f'level_{i}' for i in range(n_levels)] + ['date']
    idx = pd.MultiIndex.from_tuples(tuples, names=names)
    n_rows = len(idx)
    return pd.DataFrame({
        'a': np.arange(n_rows, dtype=float),
        'b': np.arange(n_rows, dtype=float) * 2,
    }, index=idx)


# Fonction auxiliaire d'extraction des clés d'entité d'un masque de panel
def _entities_of(series):
    """Return the set of entity key tuples present in a MultiIndex mask Series.

    Args:
        series: Boolean ``pd.Series`` on a MultiIndex ``(entity..., date)``.

    Returns:
        Set of entity key tuples, normalized to tuples even for a
        single-level panel.
    """
    entity_index = series.index.droplevel(-1)
    return {key if isinstance(key, tuple) else (key,) for key in entity_index}


class TestGetImputationWindowMaskAlignment:
    """§1.3 : get_imputation_window_mask(data) aligne le masque sur data.index."""

    def test_get_imputation_window_mask_aligned_to_data(self):
        """Le masque aligné a exactement l'index des données, False hors grille."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        df = pd.DataFrame({
            'a': np.arange(24, dtype=float),
            'b': np.arange(24, dtype=float) * 2,
        }, index=dates)
        calc = ImputationWindowCalculator(coverage_threshold=0.5, imputation_scope='strict')
        calc.fit(df)

        # Données à aligner : l'index d'origine plus une date hors de la grille ajustée
        extra_dates = dates.append(pd.DatetimeIndex(['2022-06-01']))
        data_to_align = pd.DataFrame(index=extra_dates)

        aligned = calc.get_imputation_window_mask(data_to_align)

        # Index identique à celui des données passées en argument
        pd.testing.assert_index_equal(aligned.index, data_to_align.index)
        # La date hors grille est nécessairement False
        assert aligned.loc[pd.Timestamp('2022-06-01')] == False
        # Les dates de la fenêtre stricte (couverture totale) sont True
        assert aligned.loc[dates].all()


class TestExtensionContiguity:
    """§3.2 : l'extension du masque doit s'arrêter au premier trou de couverture."""

    def test_extension_stops_at_first_gap(self):
        """extended_backward n'active rien au-delà d'un trou sous le seuil."""
        dates = pd.date_range('2020-01-01', periods=20, freq='MS')
        # 'b' en premier : c'est la colonne contiguë sur la fenêtre stricte, utilisée
        # pour détecter la position (S/E) de la grille -- 'a' a des trous et ferait
        # échouer cette détection si elle était choisie en premier (hors périmètre
        # §3.2, bug latent indépendant dans _build_index_freq_grid).
        df = pd.DataFrame({'b': np.nan, 'a': np.nan}, index=dates, dtype=float)

        # Fenêtre stricte : couverture totale sur [10, 15)
        df.loc[dates[10:15], 'a'] = 1.0
        df.loc[dates[10:15], 'b'] = 1.0
        # Avant la fenêtre, au-delà d'un trou : couverture 50% (>= seuil) sur [0, 4)
        df.loc[dates[0:4], 'a'] = 1.0
        # Trou : couverture nulle (< seuil) sur [4, 8) -- doit bloquer l'extension
        # Immédiatement avant la fenêtre stricte : couverture 50% sur [8, 10)
        df.loc[dates[8:10], 'a'] = 1.0

        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='extended_backward', min_columns=2
        )
        calc.fit(df)

        extended_dates = calc.imputation_window_mask_.index[calc.imputation_window_mask_]

        # L'extension doit s'arrêter au trou : les dates [0, 4) restent hors fenêtre
        assert not calc.imputation_window_mask_.loc[dates[0:4]].any()
        # Les dates [8, 10) juste avant la fenêtre stricte sont, elles, incluses
        assert calc.imputation_window_mask_.loc[dates[8:10]].all()


class TestStrictVsScopeWindowBounds:
    """imputation_window_start_/_end_ suivent imputation_scope, tandis que
    imputation_strict_window_start_/_end_ restent figées sur la fenêtre stricte."""

    @staticmethod
    def _make_df_with_shoulder():
        dates = pd.date_range('2020-01-01', periods=20, freq='MS')
        # 'b' en premier : cf. commentaire de test_extension_stops_at_first_gap
        df = pd.DataFrame({'b': np.nan, 'a': np.nan}, index=dates, dtype=float)
        # Fenêtre stricte : couverture totale sur [10, 15)
        df.loc[dates[10:15], 'a'] = 1.0
        df.loc[dates[10:15], 'b'] = 1.0
        # Épaule avant la fenêtre stricte, à couverture == seuil (0.5)
        df.loc[dates[8:10], 'a'] = 1.0
        return dates, df

    def test_strict_scope_bounds_are_identical(self):
        """En scope 'strict' (pas d'extension), les deux jeux de bornes coïncident."""
        dates, df = self._make_df_with_shoulder()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='strict', min_columns=2
        )
        calc.fit(df)

        assert calc.imputation_window_start_ == calc.imputation_strict_window_start_
        assert calc.imputation_window_end_ == calc.imputation_strict_window_end_
        assert calc.imputation_window_start_ == dates[10]
        assert calc.imputation_window_end_ == dates[14]

    def test_extended_scope_moves_window_bounds_but_not_strict_bounds(self):
        """En scope étendu, seules les bornes scope-dépendantes bougent."""
        dates, df = self._make_df_with_shoulder()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='extended_backward', min_columns=2
        )
        calc.fit(df)

        # La fenêtre stricte reste [10, 14], indépendamment du scope
        assert calc.imputation_strict_window_start_ == dates[10]
        assert calc.imputation_strict_window_end_ == dates[14]

        # La fenêtre scope-dépendante s'étend jusqu'à l'épaule [8, 10)
        assert calc.imputation_window_start_ == dates[8]
        assert calc.imputation_window_end_ == dates[14]


class TestMaskAtFrequency:
    """§3.3 : get_mask_at_frequency ne doit pas systématiquement retourner False."""

    def test_mask_at_frequency_full_year_is_true(self):
        """12 mois couverts -> le masque annuel vaut True (bug du floor/12.1667)."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        df = pd.DataFrame({
            'a': np.arange(24, dtype=float),
            'b': np.arange(24, dtype=float) * 2,
        }, index=dates)
        calc = ImputationWindowCalculator(coverage_threshold=0.5, imputation_scope='strict')
        calc.fit(df)

        mask_year = calc.get_mask_at_frequency('YS')

        # Les deux années 2020 et 2021 sont intégralement couvertes par le masque mensuel
        assert mask_year.loc[pd.Timestamp('2020-01-01')] == True
        assert mask_year.loc[pd.Timestamp('2021-01-01')] == True

    def test_mask_at_frequency_partial_year_is_false(self):
        """11 mois couverts sur 12 -> le masque annuel vaut False."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        df = pd.DataFrame({
            'a': np.arange(24, dtype=float),
            'b': np.arange(24, dtype=float) * 2,
        }, index=dates)
        calc = ImputationWindowCalculator(coverage_threshold=0.5, imputation_scope='strict')
        calc.fit(df)
        # Un seul mois de 2021 sort de la fenêtre stricte : l'année n'est plus
        # intégralement couverte, même si 11 de ses 12 mois le sont encore
        calc.imputation_window_mask_.loc[pd.Timestamp('2021-06-01')] = False

        mask_year = calc.get_mask_at_frequency('YS')

        assert mask_year.loc[pd.Timestamp('2020-01-01')] == True
        assert mask_year.loc[pd.Timestamp('2021-01-01')] == False

    def test_mask_at_frequency_index_reindexes_losslessly_on_data(self):
        """L'index du masque annuel obtenu depuis une grille MS reste ancré en
        début de période, même si l'appelant demande 'YE' : le résultat se
        réindexe sans perte sur des données annuelles ancrées comme la grille."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        df = pd.DataFrame({
            'a': np.arange(24, dtype=float),
            'b': np.arange(24, dtype=float) * 2,
        }, index=dates)
        calc = ImputationWindowCalculator(coverage_threshold=0.5, imputation_scope='strict')
        calc.fit(df)

        # Fréquence cible explicitement ancrée en fin de période ('YE'), alors
        # que la grille source (MS) est ancrée en début de période
        mask_year = calc.get_mask_at_frequency('YE')

        # L'index résultat reste ancré en début de période ('YS'), comme la
        # grille source -- pas en fin de période ('YE') comme demandé par
        # l'appelant, ce qui le rendrait inutilisable pour un reindex
        assert mask_year.index.freqstr == 'YS-JAN'

        # Données annuelles ancrées comme la grille source : le reindex ne
        # doit perdre aucune date, quelle que soit la valeur du masque
        yearly_data = pd.Series(
            np.arange(len(mask_year)),
            index=pd.date_range(mask_year.index[0], periods=len(mask_year), freq='YS'),
        )
        reindexed = yearly_data.reindex(mask_year.index)
        assert not reindexed.isna().any()

        # Les deux années intégralement couvertes par le masque mensuel sont vraies
        assert mask_year.loc[pd.Timestamp('2020-01-01')] == True
        assert mask_year.loc[pd.Timestamp('2021-01-01')] == True


class TestColumnCoveragePanel:
    """§3.1 : column_coverage_ doit contenir une entrée par entité pour un panel."""

    def test_column_coverage_is_per_entity_for_panel(self):
        """column_coverage_ contient une entrée par entité, pas la dernière seule."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        entities = ['A', 'B']
        idx = pd.MultiIndex.from_product([entities, dates], names=['entity', 'date'])
        df = pd.DataFrame({
            'a': np.arange(48, dtype=float),
            'b': np.arange(48, dtype=float) * 2,
        }, index=idx)

        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        calc.fit(df)

        # Un dict par entité (une clé par entité), pas un unique dict de colonnes.
        # Les clés sont des tuples y compris à un seul niveau d'entité (§3.4).
        assert set(calc.column_coverage_.keys()) == {('A',), ('B',)}
        for col_coverage in calc.column_coverage_.values():
            assert set(col_coverage.keys()) == {'a', 'b'}


class TestEntityKeysAreAlwaysTuples:
    """§3.4/§5.4 : une représentation UNIQUE des clés d'entité, le tuple.

    Le tuple s'impose à tous les niveaux d'entité, y compris un seul :
    ``('France',)`` et jamais ``'France'``. Sans cela, chaque consommateur doit
    compenser par un double lookup défensif, et c'est ce masquage qui a laissé
    passer le ``KeyError`` de §1.5.
    """

    # Attributs dict de ImputationWindowCalculator indexés par entité. Les trois
    # masques et coverage_by_date_ n'en sont plus : ils portent une pd.Series à
    # MultiIndex unique (spec §7.2), couverte par _entities_of() ailleurs.
    ENTITY_KEYED_ATTRS = (
        'imputation_window_start_',
        'imputation_window_end_',
        'imputation_strict_window_start_',
        'imputation_strict_window_end_',
        'column_coverage_',
        'index_freq_',
    )

    @pytest.mark.parametrize(
        'entities',
        [
            pytest.param([('A',), ('B',)], id='one_entity_level'),
            pytest.param([('A', 'x'), ('B', 'y')], id='two_entity_levels'),
        ],
    )
    def test_entity_keys_are_always_tuples(self, entities):
        """Tous les dicts du calculateur sont indexés par get_unique_panel_entities."""
        df = _make_panel(entities)
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        calc.fit(df)

        expected = set(get_unique_panel_entities(df))
        # Les entités attendues sont bien des tuples, même à un seul niveau
        assert expected == set(entities)

        # Critère d'acceptation n°1 : mêmes clés que get_unique_panel_entities
        for attr in self.ENTITY_KEYED_ATTRS:
            keys = set(getattr(calc, attr))
            assert keys == expected, f"{attr} indexé par {keys}, attendu {expected}"
            # Aucune clé scalaire ne doit subsister pour un panel à 1 niveau
            assert all(isinstance(key, tuple) for key in keys), attr

    @pytest.mark.parametrize(
        'entities',
        [
            pytest.param([('A',), ('B',)], id='one_entity_level'),
            pytest.param([('A', 'x'), ('B', 'y')], id='two_entity_levels'),
        ],
    )
    def test_detect_index_frequency_keys_are_tuples(self, entities):
        """detect_index_frequency indexe par tuple, comme le promet sa docstring."""
        df = _make_panel(entities)

        result = detect_index_frequency(df.index)

        assert set(result) == set(entities)
        assert all(isinstance(key, tuple) for key in result)

    @pytest.mark.parametrize(
        'entities',
        [
            pytest.param([('A',), ('B',)], id='one_entity_level'),
            pytest.param([('A', 'x'), ('B', 'y')], id='two_entity_levels'),
        ],
    )
    def test_consumers_accept_the_tuple_key(self, entities):
        """get_columns_with_coverage et get_mask_at_frequency acceptent le tuple."""
        df = _make_panel(entities)
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        calc.fit(df)

        entity = entities[0]
        start = calc.imputation_window_start_[entity]
        end = calc.imputation_window_end_[entity]

        # Entité fournie : une liste de colonnes, pas un dict par entité
        columns = calc.get_columns_with_coverage(start, end, entity=entity)
        assert isinstance(columns, list)
        assert set(columns) == {'a', 'b'}

        # Entité omise : un dict indexé par tuple
        per_entity = calc.get_columns_with_coverage(start, end)
        assert set(per_entity) == set(entities)

        # Masque à fréquence inférieure : Series à MultiIndex, une entité par tuple
        mask_at_year = calc.get_mask_at_frequency('YS')
        assert _entities_of(mask_at_year) == set(entities)

    def test_unknown_entity_raises_instead_of_silently_returning_empty(self):
        """Une clé inconnue lève un KeyError : c'est un vrai bug, pas un repli."""
        df = _make_panel([('A',), ('B',)])
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        calc.fit(df)
        start = calc.imputation_window_start_[('A',)]
        end = calc.imputation_window_end_[('A',)]

        with pytest.raises(KeyError):
            calc.get_columns_with_coverage(start, end, entity=('ZZ',))

    def test_split_variable_key_normalizes_every_level(self):
        """split_variable_key rend toujours un tuple d'entité, () hors panel."""
        assert split_variable_key(('FR', 'gdp')) == (('FR',), 'gdp')
        assert split_variable_key(('FR', 'manufacturing', 'gdp')) == (
            ('FR', 'manufacturing'), 'gdp'
        )
        assert split_variable_key('gdp') == ((), 'gdp')

    @pytest.mark.parametrize(
        'entities',
        [
            pytest.param([('A',), ('B',)], id='one_entity_level'),
            pytest.param([('A', 'x'), ('B', 'y')], id='two_entity_levels'),
        ],
    )
    def test_imputer_normalizes_user_supplied_target_frequency_keys(self, entities):
        """Les clés d'entité fournies par l'utilisateur sont normalisées en tuple.

        La normalisation n'a plus lieu à __init__ (B3/§3.16) : elle est
        recalculée à chaque fit() dans effective_target_frequency_, pour que
        self.target_frequency reste IDENTIQUE à la valeur reçue (conformité
        sklearn.clone()). C'est donc effective_target_frequency_, après fit,
        qui porte les clés d'entité normalisées.
        """
        df = _make_panel(entities)
        # Clés utilisateur volontairement "brutes" : scalaires quand c'est possible
        user_keys = [
            entity[0] if len(entity) == 1 else entity
            for entity in entities
        ]
        raw_target_frequency = {key: 'M' for key in user_keys}
        imputer = HighFrequencyImputer(
            target_frequency=raw_target_frequency,
            estimator=LinearRegression(),
        )

        # self.target_frequency reste la valeur brute, non normalisée
        assert imputer.target_frequency is raw_target_frequency

        imputer.fit(df)

        assert set(imputer.effective_target_frequency_) == set(entities)
        assert all(isinstance(key, tuple) for key in imputer.effective_target_frequency_)


class TestImputationWindowCalculatorValidation:
    """Garde-fous de construction et de `fit()` (contrats d'entrée)."""

    def test_invalid_coverage_threshold_raises(self):
        with pytest.raises(ValueError, match='coverage_threshold'):
            ImputationWindowCalculator(coverage_threshold=1.5)

    def test_invalid_imputation_scope_raises(self):
        with pytest.raises(ValueError, match='imputation_scope'):
            ImputationWindowCalculator(imputation_scope='bogus')

    def test_invalid_min_columns_raises(self):
        with pytest.raises(ValueError, match='min_columns'):
            ImputationWindowCalculator(min_columns=0)

    def test_invalid_training_scope_raises(self):
        with pytest.raises(ValueError, match='training_scope'):
            ImputationWindowCalculator(training_scope='bogus')

    def test_invalid_training_coverage_threshold_raises(self):
        with pytest.raises(ValueError, match='training_coverage_threshold'):
            ImputationWindowCalculator(training_coverage_threshold=1.5)

    def test_invalid_kind_raises(self):
        """Un `kind` fautif échoue au lieu de retomber sur la fenêtre d'imputation."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        df = pd.DataFrame({'a': range(12), 'b': range(12)}, index=dates)
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        calc.fit(df)

        with pytest.raises(ValueError, match='kind'):
            calc.get_imputation_window_mask(kind='bogus')

    def test_fit_with_single_column(self):
        """min_columns=1 : la couverture se réduit à la présence de l'unique colonne."""
        dates = pd.date_range('2023-01-01', periods=12, freq='MS')
        df = pd.DataFrame({'a': [np.nan, np.nan] + list(range(10))}, index=dates)
        calc = ImputationWindowCalculator(min_columns=1)
        calc.fit(df)

        assert calc.imputation_strict_window_start_ == dates[2]
        assert calc.imputation_strict_window_end_ == dates[-1]

    def test_fit_rejects_non_dataframe(self):
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        with pytest.raises(ValueError, match='DataFrame'):
            calc.fit([1, 2, 3])

    def test_fit_rejects_empty_dataframe(self):
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        with pytest.raises(ValueError, match='empty'):
            calc.fit(pd.DataFrame(index=pd.DatetimeIndex([])))

    def test_fit_rejects_invalid_index_type(self):
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})  # RangeIndex, pas de temps

        with pytest.raises(ValueError, match='DatetimeIndex or MultiIndex'):
            calc.fit(df)

    def test_fit_rejects_too_few_columns(self):
        calc = ImputationWindowCalculator(coverage_threshold=0.5, min_columns=2)
        dates = pd.date_range('2023-01-01', periods=12, freq='MS')
        df = pd.DataFrame({'a': range(12)}, index=dates)

        with pytest.raises(ValueError, match='min_columns'):
            calc.fit(df)

    def test_fit_panel_rejects_when_no_entity_has_valid_window(self):
        """§3.6 : aucune entité n'a de fenêtre stricte -> ValueError explicite.

        `b` est entièrement NaN pour les deux entités : la couverture
        conjointe des deux colonnes n'atteint jamais 1.0, donc aucune des
        deux entités n'obtient de fenêtre stricte.
        """
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        idx = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
        df = pd.DataFrame(
            {'a': np.arange(24, dtype=float), 'b': [np.nan] * 24}, index=idx
        )
        calc = ImputationWindowCalculator(coverage_threshold=0.5)

        with pytest.raises(ValueError, match='No imputation window'):
            calc.fit(df)


class TestNotFittedGuards:
    """Chaque accesseur public doit exiger `fit()` au préalable."""

    def test_get_imputation_window_mask_before_fit_raises(self):
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        with pytest.raises(ValueError, match='not fitted'):
            calc.get_imputation_window_mask()

    def test_get_mask_at_frequency_before_fit_raises(self):
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        with pytest.raises(ValueError, match='not fitted'):
            calc.get_mask_at_frequency('QS')

    def test_get_columns_with_coverage_before_fit_raises(self):
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        with pytest.raises(ValueError, match='not fitted'):
            calc.get_columns_with_coverage(
                pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-01')
            )


class TestGetImputationWindowMaskNoData:
    """`get_imputation_window_mask()` sans argument renvoie le masque brut."""

    def test_returns_raw_fitted_mask_when_no_data_given(self):
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        df = pd.DataFrame({'a': range(12), 'b': range(12)}, index=dates)
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        calc.fit(df)

        result = calc.get_imputation_window_mask()

        pd.testing.assert_series_equal(result, calc.imputation_window_mask_)


class TestGetMaskAtFrequencyNormalizesDictKeys:
    """§3.4/§5.4 : les clés scalaires fournies par l'appelant sont normalisées."""

    def test_get_mask_at_frequency_normalizes_dict_keys(self):
        df = _make_panel([('A',), ('B',)])
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        calc.fit(df)

        result = calc.get_mask_at_frequency({'A': 'QS', 'B': 'QS'})

        assert _entities_of(result) == {('A',), ('B',)}


class TestExtensionNoOp:
    """§3.2 : `_extend_backward`/`_extend_forward` n'étendent rien quand il n'y a
    rien à étendre (masque entièrement False, ou fenêtre déjà aux bornes de la
    grille de couverture).
    """

    @pytest.fixture
    def calc(self):
        return ImputationWindowCalculator(coverage_threshold=0.5)

    @pytest.fixture
    def full_coverage(self):
        idx = pd.date_range('2020-01-01', periods=5, freq='MS')
        return pd.Series([1.0] * 5, index=idx)

    def test_extend_backward_noop_when_mask_entirely_false(self, calc, full_coverage):
        mask = pd.Series([False] * 5, index=full_coverage.index)

        result = calc._extend_backward(full_coverage, mask, 0.5)

        assert not result.any()

    def test_extend_forward_noop_when_mask_entirely_false(self, calc, full_coverage):
        mask = pd.Series([False] * 5, index=full_coverage.index)

        result = calc._extend_forward(full_coverage, mask, 0.5)

        assert not result.any()

    def test_extend_backward_noop_when_window_starts_at_grid_start(self, calc, full_coverage):
        mask = pd.Series([True] * 5, index=full_coverage.index)

        result = calc._extend_backward(full_coverage, mask, 0.5)

        pd.testing.assert_series_equal(result, mask)

    def test_extend_forward_noop_when_window_ends_at_grid_end(self, calc, full_coverage):
        mask = pd.Series([True] * 5, index=full_coverage.index)

        result = calc._extend_forward(full_coverage, mask, 0.5)

        pd.testing.assert_series_equal(result, mask)


# Fonction auxiliaire de construction d'un jeu à épaules avant ET arrière
def _make_df_with_shoulders():
    """Build a monthly frame with a strict window flanked by two half-coverage shoulders.

    Returns:
        Tuple ``(dates, df)`` where the strict window (both columns covered)
        spans ``dates[10:15]``, and both ``dates[8:10]`` and ``dates[15:17]``
        sit at coverage 0.5 — included by a 0.5 threshold, excluded by 0.9.
    """
    dates = pd.date_range('2020-01-01', periods=20, freq='MS')
    # 'b' en premier : cf. commentaire de test_extension_stops_at_first_gap
    df = pd.DataFrame({'b': np.nan, 'a': np.nan}, index=dates, dtype=float)
    # Fenêtre stricte : couverture totale sur [10, 15)
    df.loc[dates[10:15], 'a'] = 1.0
    df.loc[dates[10:15], 'b'] = 1.0
    # Épaules à couverture == seuil (0.5), de part et d'autre de la fenêtre stricte
    df.loc[dates[8:10], 'a'] = 1.0
    df.loc[dates[15:17], 'a'] = 1.0
    return dates, df


# Fonction auxiliaire de construction d'un jeu sans aucune fenêtre stricte
def _make_df_without_strict_window():
    """Build a monthly frame where the two columns never overlap.

    Returns:
        Tuple ``(dates, df)``: ``b`` covers ``dates[0:8]`` and ``a`` covers
        ``dates[12:20]``, so coverage never reaches 1.0 anywhere.
    """
    dates = pd.date_range('2020-01-01', periods=20, freq='MS')
    df = pd.DataFrame({'b': np.nan, 'a': np.nan}, index=dates, dtype=float)
    df.loc[dates[0:8], 'b'] = 1.0
    df.loc[dates[12:20], 'a'] = 1.0
    return dates, df


# Fonction auxiliaire de construction d'un panel dont une entité n'a pas de fenêtre
def _make_panel_with_none_entity():
    """Build a two-entity panel whose entity ``B`` has an undetectable frequency.

    Returns:
        Panel DataFrame indexed by (``entity``, ``date``). Entity ``A`` is a
        regular monthly series; entity ``B`` carries deliberately irregular
        dates, so ``detect_index_frequency`` yields None for it and every
        per-entity attribute of the calculator is None.
    """
    dates_a = pd.date_range('2020-01-01', periods=24, freq='MS')
    dates_b = pd.DatetimeIndex(['2020-01-01', '2020-01-03', '2020-06-17', '2023-02-28'])
    tuples = [('A', date) for date in dates_a] + [('B', date) for date in dates_b]
    idx = pd.MultiIndex.from_tuples(tuples, names=['entity', 'date'])
    n_rows = len(idx)
    return pd.DataFrame({
        'b': np.arange(n_rows, dtype=float),
        'a': np.arange(n_rows, dtype=float) * 2,
    }, index=idx)


class TestThreeWindowMasks:
    """§3.3 : le masque strict est conservé à côté du masque de scope."""

    def test_strict_mask_narrower_than_scope_mask(self):
        """Sous 'extended_both', le masque strict est strictement inclus dans celui de scope."""
        dates, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='extended_both', min_columns=2
        )
        calc.fit(df)

        strict = calc.imputation_strict_window_mask_
        scope = calc.imputation_window_mask_

        # Inclusion : aucune date stricte n'est absente du masque de scope
        assert not (strict & ~scope).any()
        # Stricte : le scope active des dates que le masque strict n'a pas
        assert (scope & ~strict).any()
        # Les épaules sont précisément ces dates supplémentaires
        assert not strict.loc[dates[8:10]].any()
        assert scope.loc[dates[8:10]].all()

    def test_strict_equals_scope_under_strict_scope(self):
        """En scope 'strict', aucune extension n'a lieu : les deux masques coïncident."""
        dates, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='strict', min_columns=2
        )
        calc.fit(df)

        pd.testing.assert_series_equal(
            calc.imputation_strict_window_mask_, calc.imputation_window_mask_
        )


class TestTrainingScope:
    """§3.3/§3.7 : la fenêtre d'entraînement se règle indépendamment de celle de prédiction."""

    def test_training_mask_follows_training_scope(self):
        """Deux valeurs de training_scope produisent deux masques d'étendue différente."""
        dates, df = _make_df_with_shoulders()
        narrow = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='strict',
            training_scope='strict', min_columns=2,
        ).fit(df)
        wide = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='strict',
            training_scope='extended_both', min_columns=2,
        ).fit(df)

        # La fenêtre d'entraînement s'élargit, celle de prédiction reste stricte
        assert wide.training_window_mask_.sum() > narrow.training_window_mask_.sum()
        pd.testing.assert_series_equal(
            wide.imputation_window_mask_, narrow.imputation_window_mask_
        )
        # Le masque large recouvre entièrement le masque étroit
        assert not (narrow.training_window_mask_ & ~wide.training_window_mask_).any()

    def test_training_scope_none_follows_imputation_scope(self):
        """Non-régression du défaut : à None, la fenêtre d'entraînement suit celle de prédiction."""
        dates, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='extended_both', min_columns=2
        )
        calc.fit(df)

        pd.testing.assert_series_equal(
            calc.training_window_mask_, calc.imputation_window_mask_
        )

    def test_training_coverage_threshold_independent(self):
        """À imputation_scope constant, deux seuils d'entraînement donnent deux extensions."""
        dates, df = _make_df_with_shoulders()
        kwargs = dict(
            coverage_threshold=0.5, imputation_scope='extended_both',
            training_scope='extended_both', min_columns=2,
        )
        permissive = ImputationWindowCalculator(training_coverage_threshold=0.5, **kwargs).fit(df)
        strict_thr = ImputationWindowCalculator(training_coverage_threshold=0.9, **kwargs).fit(df)

        # Le seuil 0.9 exclut les épaules à couverture 0.5, le seuil 0.5 les retient
        assert permissive.training_window_mask_.sum() > strict_thr.training_window_mask_.sum()
        assert not strict_thr.training_window_mask_.loc[dates[8:10]].any()
        assert permissive.training_window_mask_.loc[dates[8:10]].all()
        # La fenêtre de prédiction, elle, est identique de part et d'autre
        pd.testing.assert_series_equal(
            permissive.imputation_window_mask_, strict_thr.imputation_window_mask_
        )

    def test_unrestricted_training_scope_is_all_true(self):
        """'unrestricted' supprime toute restriction, sans toucher à la fenêtre de prédiction."""
        dates, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='strict',
            training_scope='unrestricted', min_columns=2,
        )
        calc.fit(df)

        assert calc.training_window_mask_.all()
        # La fenêtre de prédiction reste la fenêtre stricte
        pd.testing.assert_series_equal(
            calc.imputation_window_mask_, calc.imputation_strict_window_mask_
        )


class TestThreeWindowMasksPanel:
    """§3.3 : les trois masques sont calculés par entité, y compris les entités sans fenêtre."""

    def test_entity_without_window_is_all_false(self):
        """Une entité sans fréquence identifiable a toutes ses lignes à False.

        Elle contribue ses lignes du frame ajusté (valeur False, pas une
        absence de lignes) aux trois masques, et figure dans
        ``entities_without_window_`` (spec §7.2).
        """
        df = _make_panel_with_none_entity()
        calc = ImputationWindowCalculator(coverage_threshold=0.5, min_columns=2)
        calc.fit(df)

        # B est l'entité sans fenêtre déterminable
        assert calc.entities_without_window_ == (('B',),)

        b_rows = df.loc['B'].index
        for attr in (
            'imputation_strict_window_mask_',
            'imputation_window_mask_',
            'training_window_mask_',
        ):
            mask = getattr(calc, attr)
            # Series unique à MultiIndex couvrant les deux entités
            assert isinstance(mask, pd.Series), attr
            assert _entities_of(mask) == {('A',), ('B',)}, attr
            # Toutes les lignes de B présentes, toutes à False
            b_slice = mask.xs('B', level=0)
            pd.testing.assert_index_equal(b_slice.index, b_rows)
            assert not b_slice.any(), attr
            # A garde au moins une ligne True (fenêtre déterminée)
            assert mask.xs('A', level=0).any(), attr

        # column_coverage_ reste keyée par TOUTES les entités
        assert set(calc.column_coverage_) == {('A',), ('B',)}

    def test_training_scope_none_follows_imputation_scope_panel(self):
        """Non-régression du défaut : à None, la fenêtre d'entraînement suit celle de prédiction."""
        df = _make_panel_with_none_entity()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='extended_both', min_columns=2
        )
        calc.fit(df)

        # Les deux masques (Series à MultiIndex) coïncident exactement
        pd.testing.assert_series_equal(
            calc.training_window_mask_, calc.imputation_window_mask_
        )

    def test_unrestricted_training_scope_is_all_true_panel(self):
        """'unrestricted' vaut tout-vrai pour l'entité fittée, tout-faux pour l'entité sans fenêtre."""
        df = _make_panel_with_none_entity()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='strict',
            training_scope='unrestricted', min_columns=2,
        )
        calc.fit(df)

        assert calc.training_window_mask_.xs('A', level=0).all()
        # B n'a pas de fenêtre : ses lignes restent à False, même sous 'unrestricted'
        assert not calc.training_window_mask_.xs('B', level=0).any()
        assert ('B',) in calc.entities_without_window_

        # Aligné sur les données : l'entité sans fenêtre reste entièrement False
        aligned = calc.get_imputation_window_mask(df, kind='training')
        assert aligned.loc['A'].all()
        assert not aligned.loc['B'].any()


class TestNoStrictWindow:
    """B23 : la branche « aucune fenêtre stricte » doit renseigner les trois masques."""

    def test_no_strict_window_sets_three_masks(self):
        """Sans fenêtre stricte, les trois masques existent et les bornes restent None."""
        dates, df = _make_df_without_strict_window()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='extended_both', min_columns=2
        )
        with pytest.warns(UserWarning, match='no period'):
            calc.fit(df)

        # Trois Series, et non None : le calcul de couverture a bien eu lieu
        for attr in (
            'imputation_strict_window_mask_',
            'imputation_window_mask_',
            'training_window_mask_',
        ):
            assert isinstance(getattr(calc, attr), pd.Series), attr
            # L'extension ne peut rien activer à partir d'un masque strict vide
            assert not getattr(calc, attr).any(), attr

        # Les bornes restent indéterminées
        assert calc.imputation_window_start_ is None
        assert calc.imputation_window_end_ is None
        assert calc.imputation_strict_window_start_ is None
        assert calc.imputation_strict_window_end_ is None

    def test_no_strict_window_honours_unrestricted(self):
        """Seul 'unrestricted' produit une fenêtre d'entraînement exploitable dans ce cas."""
        dates, df = _make_df_without_strict_window()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='extended_both',
            training_scope='unrestricted', min_columns=2,
        )
        with pytest.warns(UserWarning, match="unrestricted"):
            calc.fit(df)

        # Fenêtre d'entraînement tout-vrai, fenêtres stricte et d'imputation tout-faux
        assert calc.training_window_mask_.all()
        assert not calc.imputation_strict_window_mask_.any()
        assert not calc.imputation_window_mask_.any()

    def test_no_strict_window_panel_allowed_under_unrestricted(self):
        """Panel sans aucune fenêtre stricte : rédhibitoire par défaut, toléré en 'unrestricted'."""
        dates, df = _make_df_without_strict_window()
        idx = pd.MultiIndex.from_tuples(
            [(entity, date) for entity in ('A', 'B') for date in dates],
            names=['entity', 'date'],
        )
        panel = pd.concat([df, df]).set_index(idx)

        # Défaut : aucune entité entraînable, l'échec reste explicite
        with pytest.raises(ValueError, match='No imputation window'):
            ImputationWindowCalculator(coverage_threshold=0.5, min_columns=2).fit(panel)

        # 'unrestricted' : le panel reste entraînable, sans rien pouvoir imputer
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, training_scope='unrestricted', min_columns=2
        )
        with pytest.warns(UserWarning):
            calc.fit(panel)
        assert calc.training_window_mask_[('A',)].all()
        assert not calc.imputation_window_mask_[('A',)].any()


class TestGetMaskAtFrequencyKind:
    """B24 : `kind` se propage à la conversion de fréquence."""

    def test_get_mask_at_frequency_kind(self):
        """Les masques strict et de scope donnent deux masques trimestriels distincts."""
        dates, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='extended_both', min_columns=2
        )
        calc.fit(df)

        strict_quarterly = calc.get_mask_at_frequency('QS', kind='strict')
        scope_quarterly = calc.get_mask_at_frequency('QS', kind='imputation')
        training_quarterly = calc.get_mask_at_frequency('QS', kind='training')

        # L'extension active des trimestres que la fenêtre stricte n'atteint pas
        assert scope_quarterly.sum() > strict_quarterly.sum()
        assert not (strict_quarterly & ~scope_quarterly).any()
        # training_scope à None suit imputation_scope
        pd.testing.assert_series_equal(training_quarterly, scope_quarterly)

    def test_get_mask_at_frequency_defaults_to_imputation(self):
        """Le défaut préserve le comportement historique de la méthode."""
        dates, df = _make_df_with_shoulders()
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='extended_both', min_columns=2
        )
        calc.fit(df)

        pd.testing.assert_series_equal(
            calc.get_mask_at_frequency('QS'),
            calc.get_mask_at_frequency('QS', kind='imputation'),
        )


class TestPanelMaskReturnType:
    """[SPEC] §7.2 : type de retour unifié — pd.Series à MultiIndex sur panel."""

    @pytest.mark.parametrize('kind', ['strict', 'imputation', 'training'])
    def test_panel_masks_are_multiindex_series(self, kind):
        """Les trois masques sont des Series booléennes à MultiIndex couvrant
        tout l'index du frame ajusté, pour les trois valeurs de `kind`."""
        df = _make_panel([('A', 'x'), ('B', 'y')])
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='extended_both', min_columns=2
        )
        calc.fit(df)

        mask = calc.get_imputation_window_mask(kind=kind)

        assert isinstance(mask, pd.Series)
        assert isinstance(mask.index, pd.MultiIndex)
        assert mask.dtype == bool
        # Couverture intégrale de l'index du frame ajusté (la grille peut
        # s'étendre au-delà, cf. docstring de classe)
        assert set(df.index).issubset(set(mask.index))
        assert _entities_of(mask) == {('A', 'x'), ('B', 'y')}
        assert list(mask.index.names) == list(df.index.names)
        # Les attributs bruts sont les mêmes objets
        pd.testing.assert_series_equal(mask, calc.get_imputation_window_mask(kind=kind))

    def test_get_imputation_window_mask_no_data_panel_returns_multiindex_series(self):
        """Sans `data`, l'appel panel renvoie désormais une Series à MultiIndex."""
        df = _make_panel([('A',), ('B',)])
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        calc.fit(df)

        result = calc.get_imputation_window_mask()

        assert isinstance(result, pd.Series)
        assert isinstance(result.index, pd.MultiIndex)
        pd.testing.assert_series_equal(result, calc.imputation_window_mask_)

    def test_get_mask_at_frequency_returns_multiindex_series(self):
        """get_mask_at_frequency renvoie une Series à MultiIndex (entity..., date)."""
        df = _make_panel([('A', 'x'), ('B', 'y')])
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        calc.fit(df)

        mask_year = calc.get_mask_at_frequency('YS')

        assert isinstance(mask_year, pd.Series)
        assert isinstance(mask_year.index, pd.MultiIndex)
        assert mask_year.dtype == bool
        assert _entities_of(mask_year) == {('A', 'x'), ('B', 'y')}
        # L'index temporel est ancré en début de période (comme la grille source)
        year_dates = set(mask_year.index.get_level_values(-1))
        assert {pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01')}.issubset(year_dates)
        # Les deux années intégralement couvertes sont vraies pour les deux entités
        assert mask_year.xs(('A', 'x'), level=[0, 1]).loc[pd.Timestamp('2020-01-01')]
        assert mask_year.xs(('B', 'y'), level=[0, 1]).loc[pd.Timestamp('2021-01-01')]

    def test_window_bounds_still_dict_per_entity(self):
        """Les BORNES (scalaires) n'ont PAS été converties : elles restent des dicts."""
        df = _make_panel([('A',), ('B',)])
        calc = ImputationWindowCalculator(coverage_threshold=0.5)
        calc.fit(df)

        for attr in (
            'imputation_window_start_',
            'imputation_window_end_',
            'imputation_strict_window_start_',
            'imputation_strict_window_end_',
        ):
            bounds = getattr(calc, attr)
            assert isinstance(bounds, dict), attr
            assert set(bounds) == {('A',), ('B',)}, attr
            assert all(isinstance(v, pd.Timestamp) for v in bounds.values()), attr

    def test_timeseries_masks_unchanged(self):
        """Sur série temporelle : aucun changement de type ni de valeur."""
        dates = pd.date_range('2020-01-01', periods=24, freq='MS')
        df = pd.DataFrame({
            'a': np.arange(24, dtype=float),
            'b': np.arange(24, dtype=float) * 2,
        }, index=dates)
        calc = ImputationWindowCalculator(
            coverage_threshold=0.5, imputation_scope='extended_both'
        )
        calc.fit(df)

        for attr in (
            'imputation_strict_window_mask_',
            'imputation_window_mask_',
            'training_window_mask_',
        ):
            mask = getattr(calc, attr)
            assert isinstance(mask, pd.Series), attr
            assert isinstance(mask.index, pd.DatetimeIndex), attr
            assert mask.dtype == bool, attr

        # Aucune entité sans fenêtre sur une série temporelle
        assert calc.entities_without_window_ == ()
        # coverage_by_date_ reste une Series sur DatetimeIndex
        assert isinstance(calc.coverage_by_date_.index, pd.DatetimeIndex)

