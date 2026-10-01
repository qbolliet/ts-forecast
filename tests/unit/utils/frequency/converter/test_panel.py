"""Unit tests of ``FrequencyConverter.convert_frequency`` on panels (MultiIndex).

Scope: entity-by-entity conversion with a string target (two- and three-level
indexes, unsorted rows, downsampling and upsampling), ``target_freq``
dictionaries keyed by entity, by ``(entity, column)`` or by column and their
precedence, entities left untouched, panel Series, the column path
(``time_col`` + ``panel_cols``), ``target_position``, the boolean methods and
``full_periods_only`` per entity, the empty panel (refused), and the source
position and untargeted Series entities (ANO-UTILS-056 and 060, fixed).

The realistic heterogeneous panel of notebook 3 is covered by
``test_realistic_datasets.py``.
"""
import numpy as np
import pandas as pd
import pytest

from tests.support.perturbations import shuffle_rows, to_three_level_index
from tsforecast.utils.frequency.converter import FrequencyConverter


@pytest.fixture
def converter() -> FrequencyConverter:
    """A fresh ``FrequencyConverter``."""
    return FrequencyConverter()


def _panel_index(dates: pd.DatetimeIndex, entities=('A', 'B')) -> pd.MultiIndex:
    """(entity, date) index, entities outer."""
    return pd.MultiIndex.from_product([list(entities), dates], names=['entity', 'date'])


@pytest.fixture
def small_panel() -> pd.DataFrame:
    """Two entities on the month ends of 2024 H1: ``x`` = 1..6 (A), 7..12 (B); ``y`` = 10·x."""
    index = _panel_index(pd.date_range('2024-01-31', periods=6, freq='ME'))
    x = np.arange(1.0, 13.0)
    return pd.DataFrame({'x': x, 'y': 10 * x}, index=index)


@pytest.fixture
def quarterly_sums() -> pd.DataFrame:
    """Golden quarterly sums of ``small_panel``: A 6, 15 ; B 24, 33 (``y`` ten times more)."""
    index = _panel_index(pd.DatetimeIndex(['2024-03-31', '2024-06-30']))
    x = np.array([6.0, 15.0, 24.0, 33.0])
    return pd.DataFrame({'x': x, 'y': 10 * x}, index=index)


# =============================================================================
# Cible unique
# =============================================================================

class TestPanelStringTarget:
    """Each entity is converted on its own simple index."""

    def test_downsampling(self, converter, small_panel, quarterly_sums):
        """Quarterly sums per entity, entity level kept."""
        result = converter.convert_frequency(small_panel, 'QE', method='sum')
        pd.testing.assert_frame_equal(result, quarterly_sums)

    def test_unsorted_rows(self, converter, small_panel, quarterly_sums):
        """Shuffled rows give the same sorted result."""
        result = converter.convert_frequency(shuffle_rows(small_panel, seed=3), 'QE', method='sum')
        pd.testing.assert_frame_equal(result, quarterly_sums)

    def test_three_level_index(self, converter, small_panel, quarterly_sums):
        """An extra outer level is kept as part of the entity."""
        result = converter.convert_frequency(to_three_level_index(small_panel), 'QE', method='sum')
        pd.testing.assert_frame_equal(result, to_three_level_index(quarterly_sums))

    def test_upsampling(self, converter, quarterly_sums):
        """Quarterly to monthly, per entity, over the whole first quarter."""
        result = converter.convert_frequency(quarterly_sums[['x']], 'ME', method='linear')
        # Valeurs d'or : janvier-février comblés vers l'arrière, puis pas de 3 par mois
        assert result['x'].tolist() == [6.0, 6.0, 6.0, 9.0, 12.0, 15.0,
                                        24.0, 24.0, 24.0, 27.0, 30.0, 33.0]

    @pytest.mark.parametrize('as_series', [False, True], ids=['dataframe', 'series'])
    def test_never_observed_entity_is_left_out(self, converter, small_panel, quarterly_sums, as_series):
        """An entity without any observation disappears, for a Series as for a DataFrame panel."""
        panel = small_panel[['x']].copy()
        panel.loc['B', 'x'] = np.nan
        data = panel['x'] if as_series else panel
        result = converter.convert_frequency(data, 'QE', method='sum')
        # Valeurs d'or : A seule, sommes trimestrielles 6 et 15
        assert result.index.get_level_values('entity').unique().tolist() == ['A']

    def test_never_observed_panel_gives_no_row(self, converter, small_panel):
        """No entity observed: an empty panel, columns kept."""
        result = converter.convert_frequency(small_panel * np.nan, 'QE', method='sum')
        assert result.empty and list(result.columns) == ['x', 'y']

    def test_empty_panel_raises(self, converter, small_panel):
        """A panel without any row is refused, like an empty Series."""
        with pytest.raises(ValueError, match='Cannot convert empty data'):
            converter.convert_frequency(small_panel.iloc[0:0], 'QE', method='sum')

    def test_target_position(self, converter, small_panel):
        """``target_position`` relabels every entity at the start of the quarters."""
        result = converter.convert_frequency(small_panel, 'Q', method='sum', target_position='start')
        assert list(result.loc['A'].index) == [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-04-01')]

    def test_source_position_is_kept(self, converter, small_panel):
        """A position-less target inherits the start position of the entities."""
        month_starts = small_panel.copy()
        month_starts.index = _panel_index(pd.date_range('2024-01-01', periods=6, freq='MS'))
        result = converter.convert_frequency(month_starts, 'Q', method='sum')
        assert list(result.loc['A'].index) == [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-04-01')]


# =============================================================================
# Cibles par dictionnaire
# =============================================================================

class TestPanelDictTargets:
    """Dictionary keys: ``(entity,)`` > column, ``(entity, column)`` > ``(entity,)``."""

    def test_untargeted_entity_is_unchanged(self, converter, small_panel):
        """An entity absent from the dictionary keeps its rows as they are."""
        result = converter.convert_frequency(small_panel, {('A',): 'QE'}, method='sum')
        pd.testing.assert_frame_equal(result.loc[['B']], small_panel.loc[['B']])
        assert result.loc['A', 'x'].tolist() == [6.0, 15.0]

    def test_three_level_entity_key(self, converter, small_panel):
        """An entity key of a three-level panel names both entity levels."""
        panel = to_three_level_index(small_panel)
        result = converter.convert_frequency(panel, {('Zone euro', 'A'): 'QE'}, method='sum')
        # Valeurs d'or : A trimestrialisée (2 lignes), B inchangée (6 lignes)
        assert result.loc[('Zone euro', 'A'), 'x'].tolist() == [6.0, 15.0]
        assert len(result.loc[('Zone euro', 'B')]) == 6

    # Cible : x → 'YE' pour tous, A → 'YE', (A, x) → 'QE' ; y de B non ciblée
    # Valeurs d'or (alignment_method='none', NaN de l'union retirés) :
    # - (A, x) : clé (entité, colonne) la plus spécifique → trimestres 6, 15
    # - (A, y) : clé (entité,) → année 10 × 21 = 210
    # - (B, x) : clé de colonne → année 7 + … + 12 = 57
    # - (B, y) : aucune clé → valeurs mensuelles d'origine 70 … 120
    @pytest.mark.parametrize(
        'entity, column, dates, values',
        [
            pytest.param('A', 'x', ['2024-03-31', '2024-06-30'], [6.0, 15.0], id='entity-column-key'),
            pytest.param('A', 'y', ['2024-12-31'], [210.0], id='entity-key'),
            pytest.param('B', 'x', ['2024-12-31'], [57.0], id='column-key'),
            pytest.param('B', 'y', list(pd.date_range('2024-01-31', periods=6, freq='ME')),
                         [70.0, 80.0, 90.0, 100.0, 110.0, 120.0], id='untargeted'),
        ],
    )
    def test_key_precedence(self, converter, small_panel, entity, column, dates, values):
        """The most specific key wins for each (entity, column)."""
        target = {'x': 'YE', ('A',): 'YE', ('A', 'x'): 'QE'}
        result = converter.convert_frequency(small_panel, target, method='sum', alignment_method='none')

        observed = result.loc[entity, column].dropna()
        pd.testing.assert_series_equal(
            observed, pd.Series(values, index=pd.DatetimeIndex(dates, name='date')),
            check_names=False, check_freq=False,
        )


class TestPanelSeries:
    """A panel Series is converted entity by entity."""

    def test_string_target(self, converter, small_panel, quarterly_sums):
        """Same quarterly sums as the DataFrame column."""
        result = converter.convert_frequency(small_panel['x'], 'QE', method='sum')
        pd.testing.assert_series_equal(result, quarterly_sums['x'])

    def test_one_target_per_entity(self, converter, small_panel):
        """Entity keys give each entity its own frequency."""
        result = converter.convert_frequency(small_panel['x'], {('A',): 'QE', ('B',): 'YE'}, method='sum')
        # Valeurs d'or : A trimestres 6, 15 ; B année 57
        assert result.loc['A'].tolist() == [6.0, 15.0] and result.loc['B'].tolist() == [57.0]

    def test_untargeted_entity_is_unchanged(self, converter, small_panel):
        """An entity absent from the dictionary keeps its rows, as for a DataFrame."""
        result = converter.convert_frequency(small_panel['x'], {('A',): 'QE'}, method='sum')
        assert len(result.loc['B']) == 6

    def test_column_key_names_the_series(self, converter, small_panel, quarterly_sums):
        """A column key matches the name of the panel Series, for every entity."""
        result = converter.convert_frequency(small_panel['x'], {'x': 'QE'}, method='sum')
        pd.testing.assert_series_equal(result, quarterly_sums['x'])


# =============================================================================
# Chemin colonnes, booléens, périodes complètes
# =============================================================================

class TestPanelColumnPath:
    """Panels given as columns (``time_col`` + ``panel_cols``)."""

    @pytest.mark.filterwarnings('ignore:Index replaced')
    def test_time_and_panel_columns(self, converter, small_panel, quarterly_sums):
        """The output is indexed by (entity, date)."""
        result = converter.convert_frequency(
            small_panel.reset_index(), 'QE', method='sum', time_col='date', panel_cols=['entity']
        )
        pd.testing.assert_frame_equal(result, quarterly_sums)

    @pytest.mark.filterwarnings('ignore:Index replaced')
    def test_dict_target_with_panel_columns(self, converter, small_panel, quarterly_sums):
        """Column keys that do not name an identifier column are converted."""
        result = converter.convert_frequency(
            small_panel.reset_index(), {'x': 'QE', 'y': 'QE'},
            method='sum', time_col='date', panel_cols=['entity'],
        )
        pd.testing.assert_frame_equal(result, quarterly_sums)

    @pytest.mark.filterwarnings('ignore:Index replaced')
    def test_panel_column_cannot_be_a_target(self, converter, small_panel):
        """An identifier column is not a variable to convert."""
        with pytest.raises(ValueError, match='Panel columns cannot be in target_freq'):
            converter.convert_frequency(
                small_panel.reset_index(), {'x': 'QE', 'entity': 'QE'},
                method='sum', time_col='date', panel_cols=['entity'],
            )


class TestPanelBooleanAndCoverage:
    """Boolean methods and ``full_periods_only`` are evaluated entity by entity."""

    @pytest.fixture
    def flags(self) -> pd.DataFrame:
        """A: always True ; B: True in January and February only."""
        index = _panel_index(pd.date_range('2024-01-31', periods=6, freq='ME'))
        return pd.DataFrame({'flag': [True] * 6 + [True, True, False, False, False, False]}, index=index)

    # Valeurs d'or : A vrai partout ; B → T1 partiellement vrai, T2 entièrement faux
    @pytest.mark.parametrize(
        'method, expected',
        [
            pytest.param('all', [True, True, False, False], id='all'),
            pytest.param('any', [True, True, True, False], id='any'),
        ],
    )
    def test_boolean_methods(self, converter, flags, method, expected):
        """Quarterly booleans per entity."""
        result = converter.convert_frequency(flags, 'QE', method=method)
        assert result['flag'].tolist() == expected

    @pytest.mark.parametrize(
        'full_periods_only, expected',
        [
            pytest.param(True, [6.0, 15.0, 24.0, np.nan], id='incomplete-quarter-masked'),
            pytest.param(False, [6.0, 15.0, 24.0, 22.0], id='incomplete-quarter-summed'),
        ],
    )
    def test_full_periods_only(self, converter, small_panel, full_periods_only, expected):
        """Only B misses May: only B's second quarter is masked."""
        panel = small_panel[['x']].copy()
        panel.loc[('B', pd.Timestamp('2024-05-31')), 'x'] = np.nan
        result = converter.convert_frequency(panel, 'QE', method='sum', full_periods_only=full_periods_only)
        # Valeur d'or sans masquage : B T2 = 10 + 12 = 22
        assert result['x'].tolist() == pytest.approx(expected, nan_ok=True)
