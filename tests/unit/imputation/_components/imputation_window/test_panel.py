"""Windows on panels.

Scenarios: panel return types, entities without a window, structurally absent columns,
heterogeneous coverage across entities and per-entity index frequencies (several entity
levels, missing entity labels).
"""

# Modules de base
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

# Objet testé
from tsforecast.imputation._components.imputation_window import ImputationWindowCalculator
from tsforecast.panel.utils import get_unique_panel_entities

# Outils de test partagés du paquet
from tests.unit.imputation._components.imputation_window.helpers import (
    _BOUND_ATTRIBUTES,
    _KINDS,
    _MASK_ATTRIBUTES,
    _entities_of,
    _fit_quietly,
    _frame,
    _make_panel,
    _warning_messages,
)


def _make_panel_with_none_entity() -> pd.DataFrame:
    """Build a panel whose entity ``B`` has an index frequency that cannot be detected.

    Entity ``A`` is a regular monthly series (24 months); entity ``B`` carries
    four irregularly spaced dates, so ``detect_index_frequency`` yields None
    for it.
    """
    dates_a = pd.date_range('2020-01-01', periods=24, freq='MS')
    dates_b = pd.DatetimeIndex(['2020-01-01', '2020-01-03', '2020-06-17', '2023-02-28'])
    tuples = [('A', date) for date in dates_a] + [('B', date) for date in dates_b]
    index = pd.MultiIndex.from_tuples(tuples, names=['entity', 'date'])
    n_rows = len(index)
    return pd.DataFrame({
        'b': np.arange(n_rows, dtype=float),
        'a': np.arange(n_rows, dtype=float) * 2,
    }, index=index)


def _make_panel_with_empty_entity() -> pd.DataFrame:
    """Build a regular monthly panel whose entity ``B`` observes nothing at all.

    ``B``'s index frequency is detected (regular monthly rows) but its
    coverage grid is empty: no observation, hence no period to expand.
    """
    dates = pd.date_range('2020-01-01', periods=12, freq='MS')
    index = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
    values = list(range(12)) + [np.nan] * 12
    return pd.DataFrame({'a': values, 'b': values}, index=index, dtype=float)


def _months(bounds: Dict[tuple, Optional[pd.Timestamp]]) -> Dict[str, Optional[str]]:
    """Render per-entity window bounds as ``{last entity level: 'YYYY-MM'}``."""
    return {
        key[-1]: (value.strftime('%Y-%m') if value is not None else None)
        for key, value in bounds.items()
    }


# =============================================================================
# Types de retour sur un panel
# =============================================================================
class TestPanelReturnTypes:
    """Masks are one MultiIndex Series, bounds are dicts keyed by entity tuples."""

    @pytest.mark.parametrize('kind', _KINDS)
    def test_masks_are_a_single_multiindex_series(self, kind):
        """Each mask is a boolean Series on ``(entity..., date)``, never a dict."""
        df = _make_panel([('A', 'x'), ('B', 'y')])
        calc = ImputationWindowCalculator(imputation_scope='extended_both', min_columns=2).fit(df)

        mask = calc.get_imputation_window_mask(kind=kind)

        assert (type(mask), type(mask.index), mask.dtype, list(mask.index.names),
                _entities_of(mask), set(df.index) <= set(mask.index)) == (
            pd.Series, pd.MultiIndex, bool, list(df.index.names), {('A', 'x'), ('B', 'y')}, True
        )

    @pytest.mark.parametrize('kind', _KINDS)
    def test_raw_mask_is_the_fitted_attribute(self, kind):
        """Without ``data``, the method returns the fitted attribute designated by ``kind``."""
        df = _make_panel([('A',), ('B',)])
        calc = ImputationWindowCalculator().fit(df)
        attribute = dict(zip(_KINDS, _MASK_ATTRIBUTES))[kind]

        assert calc.get_imputation_window_mask(kind=kind) is getattr(calc, attribute)

    def test_coverage_by_date_is_a_multiindex_series(self):
        """``coverage_by_date_`` follows the same unified type."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))

        assert (isinstance(calc.coverage_by_date_.index, pd.MultiIndex),
                _entities_of(calc.coverage_by_date_)) == (True, {('A',), ('B',)})

    @pytest.mark.parametrize(
        'entities',
        [
            pytest.param([('A',), ('B',)], id='one_entity_level'),
            pytest.param([('A', 'x'), ('B', 'y')], id='two_entity_levels'),
        ],
    )
    @pytest.mark.parametrize(
        'attribute',
        list(_BOUND_ATTRIBUTES) + ['column_coverage_', 'index_freq_', 'structurally_absent_columns_',
                                   'undetected_frequency_columns_'],
    )
    def test_dict_attributes_are_keyed_by_entity_tuples(self, entities, attribute):
        """Every per-entity dict uses the keys of ``get_unique_panel_entities``.

        Tuples even with a single entity level: ``('A',)``, never ``'A'``
        so a KeyError genuinely signals a bug.
        """
        df = _make_panel(entities)
        calc = ImputationWindowCalculator().fit(df)

        value = getattr(calc, attribute)
        assert (isinstance(value, dict), set(value)) == (True, set(get_unique_panel_entities(df)))

    def test_bounds_are_timestamps_per_entity(self):
        """Bounds stay scalar per entity: one Timestamp each."""
        calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))

        assert all(
            isinstance(v, pd.Timestamp)
            for attr in _BOUND_ATTRIBUTES for v in getattr(calc, attr).values()
        )

    def test_time_series_masks_are_on_a_datetime_index(self):
        """On a time series: Series on a DatetimeIndex, no entity without window."""
        calc = ImputationWindowCalculator(imputation_scope='extended_both').fit(
            _frame({'a': list(range(24)), 'b': list(range(24))})
        )

        assert (all(isinstance(getattr(calc, a).index, pd.DatetimeIndex) for a in _MASK_ATTRIBUTES),
                isinstance(calc.coverage_by_date_.index, pd.DatetimeIndex),
                calc.entities_without_window_) == (True, True, ())


# =============================================================================
# Entités sans fenêtre
# =============================================================================
class TestEntitiesWithoutWindow:
    """Entities whose window cannot be determined contribute False rows and are listed."""

    def test_undetected_index_frequency_gives_false_rows(self):
        """An entity with an irregular index contributes its frame rows, all False."""
        df = _make_panel_with_none_entity()
        calc = _fit_quietly(ImputationWindowCalculator(min_columns=2), df)

        b_rows = df.loc['B'].index
        assert calc.entities_without_window_ == (('B',),)
        for attr in _MASK_ATTRIBUTES:
            b_slice = getattr(calc, attr).xs('B', level=0)
            pd.testing.assert_index_equal(b_slice.index, b_rows)
            assert not b_slice.any(), attr
            assert getattr(calc, attr).xs('A', level=0).any(), attr

    def test_undetected_index_frequency_keeps_every_entity_key(self):
        """Bounds are None and ``column_coverage_`` still has an entry for the entity."""
        calc = _fit_quietly(ImputationWindowCalculator(min_columns=2), _make_panel_with_none_entity())

        assert ([getattr(calc, attr)[('B',)] for attr in _BOUND_ATTRIBUTES], calc.column_coverage_[('B',)]) == (
            [None] * len(_BOUND_ATTRIBUTES), None
        )

    def test_empty_coverage_grid_gives_false_rows(self):
        """An entity observing nothing (regular index) has no grid: False rows, listed."""
        df = _make_panel_with_empty_entity()
        calc = _fit_quietly(ImputationWindowCalculator(), df)

        counts = calc.imputation_window_mask_.groupby(level=0).agg(['sum', 'size'])
        assert (calc.entities_without_window_, counts.loc['B'].tolist(), counts.loc['A'].tolist(),
                calc.imputation_window_start_[('B',)], calc.column_coverage_[('B',)]) == (
            (('B',),), [0, 12], [12, 12], None, None
        )

    def test_empty_entity_reports_no_excluded_column(self):
        """An entity without window carries ``()`` in both exclusion attributes.

        The attributes are only meaningful for an entity with a coverage grid:
        ``B``'s columns were never weighed against one another (documented
        rule).
        """
        calc = _fit_quietly(ImputationWindowCalculator(), _make_panel_with_empty_entity())

        assert (calc.structurally_absent_columns_, calc.undetected_frequency_columns_) == (
            {('A',): (), ('B',): ()}, {('A',): (), ('B',): ()}
        )

    @pytest.mark.parametrize(
        'builder',
        [
            pytest.param(_make_panel_with_none_entity, id='undetected_index_frequency'),
            pytest.param(_make_panel_with_empty_entity, id='no_observation'),
        ],
    )
    def test_entities_without_window_are_named_by_one_warning(self, builder):
        """Losing a whole entity is never silent: one warning names every such entity."""
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            ImputationWindowCalculator(min_columns=2).fit(builder())

        named = [m for m in _warning_messages(records) if 'without a determinable window' in m]
        assert (len(named), "[('B',)]" in named[0]) == (1, True)

    def test_entity_reduced_to_one_row_has_no_window_and_costs_the_others_nothing(self):
        """An entity of a single row joins the entities without window.

        Its index frequency is undetectable (a single date): the detection
        on the whole index does not raise, and the computation does not fail for
        the whole panel.
        """
        panel = _make_panel([('A',), ('B',)])
        single_row = pd.DataFrame(
            {'a': [1.0], 'b': [2.0]},
            index=pd.MultiIndex.from_tuples(
                [('C', pd.Timestamp('2020-01-01'))], names=panel.index.names
            ),
        )

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            calc = ImputationWindowCalculator().fit(pd.concat([panel, single_row]))

        assert calc.entities_without_window_ == (('C',),)
        assert calc.index_freq_[('C',)] is None
        assert bool(calc.imputation_window_mask_.drop(index='C', level=0).all())
        named = [m for m in _warning_messages(records) if 'without a determinable window' in m]
        assert len(named) == 1 and "[('C',)]" in named[0]

    def test_no_warning_when_every_entity_has_a_window(self):
        """A panel where every entity has a window fits silently."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            calc = ImputationWindowCalculator().fit(_make_panel([('A',), ('B',)]))

        assert calc.entities_without_window_ == ()

    def test_unrestricted_does_not_revive_an_entity_without_window(self):
        """'unrestricted' is all True for the fitted entity, all False for the other."""
        df = _make_panel_with_none_entity()
        calc = _fit_quietly(ImputationWindowCalculator(training_scope='unrestricted', min_columns=2), df)

        aligned = calc.get_imputation_window_mask(df, kind='training')
        assert (bool(aligned.loc['A'].all()), bool(aligned.loc['B'].any())) == (True, False)

    def test_training_scope_none_follows_imputation_scope_on_a_panel(self):
        """Default training window = prediction window, entity without window included."""
        calc = _fit_quietly(ImputationWindowCalculator(imputation_scope='extended_both', min_columns=2), _make_panel_with_none_entity())

        pd.testing.assert_series_equal(calc.training_window_mask_, calc.imputation_window_mask_)

    @pytest.mark.parametrize(
        'builder',
        [
            pytest.param(_make_panel_with_none_entity, id='undetected_index_frequency'),
            pytest.param(_make_panel_with_empty_entity, id='no_observation'),
        ],
    )
    def test_entity_without_window_is_absent_from_mask_at_frequency(self, builder):
        """``get_mask_at_frequency`` leaves out every entity without window, whatever the cause.

        Absent means "no window", unlike ``False`` ("outside the window"):
        the imputer and ``TrainingSetBuilder`` read an absent entity as
        unrestricted. An entity with a known frequency but no observation
        is read as unrestricted too, not converted to all-False rows.
        """
        calc = _fit_quietly(ImputationWindowCalculator(min_columns=2), builder())

        assert _entities_of(calc.get_mask_at_frequency('QS')) == {('A',)}

    def test_dict_target_need_not_name_an_entity_without_window(self):
        """A per-entity target listing only the entities with a window is enough."""
        calc = _fit_quietly(ImputationWindowCalculator(), _make_panel_with_empty_entity())

        assert calc.get_mask_at_frequency({'A': 'QS'}).tolist() == [True] * 4

    def test_columns_with_coverage_is_empty_for_an_entity_without_window(self):
        """No coverage dict for the entity: no covered column."""
        calc = _fit_quietly(ImputationWindowCalculator(min_columns=2), _make_panel_with_none_entity())

        assert calc.get_columns_with_coverage(
            pd.Timestamp('2020-01-01'), pd.Timestamp('2023-12-31'), entity=('B',)
        ) == []


# =============================================================================
# Colonnes structurellement absentes
# =============================================================================
class TestStructurallyAbsentColumns:
    """A column never observed (per entity) is left out of the coverage denominator.

    It says nothing about the temporal extent of its entity: kept in the
    denominator, it would cap coverage at (n-1)/n, hence an empty strict
    window and an entity where nothing can be imputed.
    """

    @staticmethod
    def _panel_with_absent_column() -> pd.DataFrame:
        """Panel where ``b`` is never observed for entity ``B`` only."""
        dates = pd.date_range('2020-01-01', periods=12, freq='MS')
        index = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
        column_b = np.concatenate([np.arange(12, dtype=float), [np.nan] * 12])
        return pd.DataFrame({'a': np.arange(24, dtype=float), 'b': column_b}, index=index)

    def test_absent_column_does_not_veto_the_entity_window(self):
        """The entity missing a column keeps a full window."""
        calc = ImputationWindowCalculator()
        with pytest.warns(UserWarning, match='coverage denominator'):
            calc.fit(self._panel_with_absent_column())

        counts = calc.get_imputation_window_mask().groupby(level=0).sum().to_dict()
        assert (counts, calc.entities_without_window_) == ({'A': 12, 'B': 12}, ())

    def test_absent_columns_are_reported_per_entity(self):
        """The exclusion is named per entity; the reported coverage stays (None, None)."""
        calc = _fit_quietly(ImputationWindowCalculator(), self._panel_with_absent_column())

        assert (calc.structurally_absent_columns_, calc.column_coverage_[('B',)]['b']) == (
            {('A',): (), ('B',): ('b',)}, (None, None)
        )

    def test_empty_column_of_a_time_series_is_absent(self):
        """On a time series, an empty column is structurally absent too.

        It is excluded from the denominator without shrinking the strict
        window of the two other columns.
        """
        df = _frame({'a': list(range(12)), 'b': list(range(12)), 'c': [np.nan] * 12})
        calc = ImputationWindowCalculator()
        with pytest.warns(UserWarning, match='coverage denominator'):
            calc.fit(df)

        assert (calc.structurally_absent_columns_,
                calc.imputation_strict_window_start_,
                calc.imputation_strict_window_end_) == (('c',), df.index[0], df.index[-1])

    def test_no_absent_column_is_silent(self):
        """Without an absent column, no warning and an empty attribute."""
        df = _frame({'a': list(range(12)), 'b': list(range(12))})
        calc = ImputationWindowCalculator()
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            calc.fit(df)

        assert calc.structurally_absent_columns_ == ()

    def test_refit_resets_panel_only_attributes(self):
        """Re-fitting a panel-fitted calculator on a time series resets the per-entity attributes."""
        calc = _fit_quietly(ImputationWindowCalculator(min_columns=2), _make_panel_with_none_entity())
        calc.fit(_frame({'a': list(range(12)), 'b': list(range(12))}))

        assert (calc.structurally_absent_columns_, calc.undetected_frequency_columns_,
                calc.entities_without_window_) == ((), (), ())


class TestStructurallyAbsentOnHeterogeneousPanel:
    """On the realistic panel, ``climat_affaires`` is never observed for Italie."""

    def test_absent_columns_are_exact(self, heterogeneous_coverage_panel):
        """Only (Italie, climat_affaires) is structurally absent."""
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        assert calc.structurally_absent_columns_ == {
            ('Allemagne',): (), ('France',): (), ('Italie',): ('climat_affaires',),
        }

    def test_single_warning_names_the_entity_and_column(self, heterogeneous_coverage_panel):
        """Exactly one denominator warning per fit, naming Italie and its column."""
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            ImputationWindowCalculator().fit(heterogeneous_coverage_panel)

        denominator = [m for m in _warning_messages(records) if 'coverage denominator' in m]
        assert (len(denominator), "('Italie',): ['climat_affaires']" in denominator[0]) == (1, True)

    def test_italy_keeps_a_strict_window(self, heterogeneous_coverage_panel):
        """Without the exclusion, Italie's coverage would never exceed 6/7: its strict window exists."""
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        # Valeur d'or : production industrielle dès 2019-06 ; dépenses et balance jusqu'en 2023-12
        assert (calc.imputation_strict_window_start_[('Italie',)],
                calc.imputation_strict_window_end_[('Italie',)]) == (
            pd.Timestamp('2019-06-01'), pd.Timestamp('2023-12-01')
        )

    @pytest.mark.parametrize(
        'entity, date, expected',
        [
            # Italie : 6 colonnes au dénominateur ; en 2019-01, seule la production manque
            pytest.param('Italie', '2019-01-01', 5 / 6, id='italie_denominator_6'),
            # France : 7 colonnes ; en 2018-01, seule la production manque
            pytest.param('France', '2018-01-01', 6 / 7, id='france_denominator_7'),
        ],
    )
    def test_coverage_ratio_uses_present_columns_only(self, heterogeneous_coverage_panel, entity, date, expected):
        """The coverage ratio divides by the entity's present columns."""
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        assert calc.coverage_by_date_.loc[(entity, pd.Timestamp(date))] == pytest.approx(expected)

    def test_reported_coverage_of_the_absent_column_is_none(self, heterogeneous_coverage_panel):
        """The exclusion bears on the denominator, not on ``column_coverage_``."""
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        assert calc.column_coverage_[('Italie',)]['climat_affaires'] == (None, None)


# =============================================================================
# Jeux réalistes
# =============================================================================
class TestHeterogeneousCoveragePanel:
    """Per-entity windows consistent with each entity's own coverage.

    Golden bounds derived by hand from ``HETEROGENEOUS_PANEL_COUNTRIES``
    (monthly grids FR 2018-01→2024-07, DE 2018-07→2024-04, IT 2019-01→2024-07;
    production from FR 2018-06, DE 2019-01, IT 2019-06; last monthly value of
    inflation / unemployment, last quarter of GDP, last year of the annual
    series withheld; annual history from 2015 (FR) / 2016 (DE, IT)).
    """

    @pytest.mark.parametrize(
        'scope, threshold, expected_start, expected_end',
        [
            pytest.param('strict', 0.5,
                         {'Allemagne': '2019-01', 'France': '2018-06', 'Italie': '2019-06'},
                         {'Allemagne': '2023-12', 'France': '2023-12', 'Italie': '2023-12'},
                         id='strict'),
            # 6/7 (FR, DE) et 5/6 (IT) avant la production ; balance seule (1/7, 1/6) plus tôt
            pytest.param('extended_backward', 0.5,
                         {'Allemagne': '2018-07', 'France': '2018-01', 'Italie': '2019-01'},
                         {'Allemagne': '2023-12', 'France': '2023-12', 'Italie': '2023-12'},
                         id='extended_backward'),
            # 2024 : sans les séries annuelles, 5/7 (FR), 6/7 (DE), 4/6 (IT) ; dernier mois 2/7, 2/7, 1/6
            pytest.param('extended_forward', 0.5,
                         {'Allemagne': '2019-01', 'France': '2018-06', 'Italie': '2019-06'},
                         {'Allemagne': '2024-03', 'France': '2024-06', 'Italie': '2024-06'},
                         id='extended_forward'),
            # Seuil 0 : jusqu'au bout de la grille propre à chaque entité (DE finit en 2024-04)
            pytest.param('extended_forward', 0.0,
                         {'Allemagne': '2019-01', 'France': '2018-06', 'Italie': '2019-06'},
                         {'Allemagne': '2024-04', 'France': '2024-07', 'Italie': '2024-07'},
                         id='extended_forward_threshold_0'),
            # Seuil 0 : jusqu'à la première ancre annuelle de chaque entité
            pytest.param('extended_backward', 0.0,
                         {'Allemagne': '2016-01', 'France': '2015-01', 'Italie': '2016-01'},
                         {'Allemagne': '2023-12', 'France': '2023-12', 'Italie': '2023-12'},
                         id='extended_backward_threshold_0'),
        ],
    )
    def test_window_bounds_per_entity(self, heterogeneous_coverage_panel, scope, threshold,
                                      expected_start, expected_end):
        """Each entity's window follows its own coverage."""
        calc = _fit_quietly(
            ImputationWindowCalculator(imputation_scope=scope, coverage_threshold=threshold),
            heterogeneous_coverage_panel,
        )

        assert (_months(calc.imputation_window_start_), _months(calc.imputation_window_end_)) == (
            expected_start, expected_end
        )

    def test_annual_anchor_rows_stay_outside_the_window(self, heterogeneous_coverage_panel):
        """Rows before an entity's monthly grid (annual anchors only) are never in the window."""
        calc = _fit_quietly(
            ImputationWindowCalculator(imputation_scope='extended_both'), heterogeneous_coverage_panel
        )
        aligned = calc.get_imputation_window_mask(heterogeneous_coverage_panel)

        # Lignes antérieures au début de la grille mensuelle de chaque entité
        monthly_start = {'France': '2018-01-01', 'Allemagne': '2018-07-01', 'Italie': '2019-01-01'}
        dates = aligned.index.get_level_values('date')
        countries = aligned.index.get_level_values('country')
        before = np.array([d < pd.Timestamp(monthly_start[c]) for c, d in zip(countries, dates)])
        assert (int(before.sum()), bool(aligned[before].any())) == (9, False)

    def test_entities_all_have_a_window(self, heterogeneous_coverage_panel):
        """Every entity of the realistic panel has a window (index frequency detected)."""
        calc = _fit_quietly(ImputationWindowCalculator(), heterogeneous_coverage_panel)

        assert (calc.entities_without_window_, set(calc.index_freq_.values())) == ((), {'M'})


# =============================================================================
# Fréquence d'index entité par entité
# =============================================================================
class TestEntityIndexFrequencies:
    """The index frequency is detected per entity, entity labels being never missing."""

    def test_two_entity_levels_with_interleaved_rows(self):
        """Each ``(level 0, level 1)`` combination is one entity, whatever the row order.

        The rows of the entities are interleaved: each entity still reads its
        own dates only, in their original order.
        """
        months = pd.date_range('2020-01-31', periods=4, freq='ME')
        quarters = pd.date_range('2020-03-31', periods=4, freq='QE')
        tuples = [
            row
            for month, quarter in zip(months, quarters)
            for row in (('A', 'x', month), ('A', 'y', quarter))
        ] + [('B', 'x', months[0])]
        index = pd.MultiIndex.from_tuples(tuples, names=['country', 'sector', 'date'])

        assert ImputationWindowCalculator._detect_entity_index_frequencies(index) == {
            ('A', 'x'): 'M', ('A', 'y'): 'Q', ('B', 'x'): None,
        }

    @pytest.mark.parametrize('level', [0, 1], ids=['first_entity_level', 'second_entity_level'])
    def test_missing_entity_label_is_rejected(self, level):
        """A missing entity label raises instead of being dropped or kept as an entity."""
        df = _make_panel([('A', 'x'), ('B', 'y')])
        labels = [list(df.index.get_level_values(position)) for position in range(df.index.nlevels)]
        labels[level] = [np.nan if label == 'B' or label == 'y' else label for label in labels[level]]
        df.index = pd.MultiIndex.from_arrays(labels, names=df.index.names)

        with pytest.raises(ValueError, match='Entity labels must not be missing'):
            ImputationWindowCalculator._detect_entity_index_frequencies(df.index)
        with pytest.raises(ValueError, match='Entity labels must not be missing'):
            _fit_quietly(ImputationWindowCalculator(), df)
