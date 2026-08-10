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

    # Attributs dict de ImputationWindowCalculator indexés par entité
    ENTITY_KEYED_ATTRS = (
        'imputation_window_start_',
        'imputation_window_end_',
        'imputation_strict_window_start_',
        'imputation_strict_window_end_',
        'imputation_window_mask_',
        'coverage_by_date_',
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

        # Masque à fréquence inférieure : dict indexé par tuple lui aussi
        mask_at_year = calc.get_mask_at_frequency('YS')
        assert set(mask_at_year) == set(entities)

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
        """Les clés d'entité fournies par l'utilisateur sont normalisées en tuple."""
        df = _make_panel(entities)
        # Clés utilisateur volontairement "brutes" : scalaires quand c'est possible
        user_keys = [
            entity[0] if len(entity) == 1 else entity
            for entity in entities
        ]
        imputer = HighFrequencyImputer(
            target_frequency={key: 'M' for key in user_keys},
            estimator=LinearRegression(),
        )

        assert set(imputer.target_frequency) == set(entities)
        assert all(isinstance(key, tuple) for key in imputer.target_frequency)


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

        assert set(result) == {('A',), ('B',)}


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

        result = calc._extend_backward(full_coverage, mask)

        assert not result.any()

    def test_extend_forward_noop_when_mask_entirely_false(self, calc, full_coverage):
        mask = pd.Series([False] * 5, index=full_coverage.index)

        result = calc._extend_forward(full_coverage, mask)

        assert not result.any()

    def test_extend_backward_noop_when_window_starts_at_grid_start(self, calc, full_coverage):
        mask = pd.Series([True] * 5, index=full_coverage.index)

        result = calc._extend_backward(full_coverage, mask)

        pd.testing.assert_series_equal(result, mask)

    def test_extend_forward_noop_when_window_ends_at_grid_end(self, calc, full_coverage):
        mask = pd.Series([True] * 5, index=full_coverage.index)

        result = calc._extend_forward(full_coverage, mask)

        pd.testing.assert_series_equal(result, mask)
