"""``FrequencyAligner.convert_to_target`` on mixed inputs.

Scenarios: per-key routing between aggregation and interpolation, heterogeneous coverage
on a panel, robustness to row order and index shape, and the private guards.
"""

import numpy as np
import pandas as pd
import pytest

# Perturbations partagées
from tests.support.perturbations import (
    reverse_entities,
    shuffle_rows,
    to_three_level_index,
    with_duplicated_rows,
    with_index_names,
    with_special_column_names,
)

# Outils de test partagés du paquet
from tests.unit.frequency.frequency_aligner.helpers import _frame, _panel, _series


# Entités du panel réaliste à couvertures hétérogènes
_COUNTRIES = ['France', 'Allemagne', 'Italie']


# ------------------------------------------------------------------ #
#  convert_to_target : orientation et clés                            #
# ------------------------------------------------------------------ #
class TestConvertToTargetRouting:
    """Each key is aggregated or interpolated according to its own observed frequency."""

    def test_mixed_keys_are_routed_independently(self, aligner):
        """A monthly and a yearly column are respectively aggregated and interpolated to Q."""
        # Mensuelle 1..24 et annuelle (40 en 2022, 80 en 2023) sur une grille mensuelle
        df = _frame(np.arange(1, 25), '2022-01-01', 'MS', col='m')
        df['y'] = np.nan
        df.loc['2022-01-01', 'y'] = 40.0
        df.loc['2023-01-01', 'y'] = 80.0

        result = aligner.convert_to_target(df, ['m', 'y'], 'QS')

        # Valeur d'or : m = sommes trimestrielles (6, 15, ..., 69) ; y = rampe de +10 par
        # trimestre de 40 à 80, puis 80 maintenu
        expected = pd.DataFrame(
            {'m': [6.0, 15.0, 24.0, 33.0, 42.0, 51.0, 60.0, 69.0],
             'y': [40.0, 50.0, 60.0, 70.0, 80.0, 80.0, 80.0, 80.0]},
            index=pd.date_range('2022-01-01', periods=8, freq='QS'),
        )
        pd.testing.assert_frame_equal(result.dropna(how='all'), expected, check_freq=False)

    def test_same_frequency_leaves_values_unchanged(self, aligner):
        """A monthly column converted to monthly is a sum of one sub-period: identity."""
        df = _frame(np.arange(1, 7), '2023-01-01', 'MS')

        result = aligner.convert_to_target(df, ['x'], 'MS')

        pd.testing.assert_frame_equal(result, df)

    def test_empty_keys_return_the_frame_unchanged(self, aligner):
        """No key to convert: the frame is returned as is."""
        df = _frame(np.ones(4), '2023-01-01', 'MS')

        result = aligner.convert_to_target(df, [], 'QS')

        pd.testing.assert_frame_equal(result, df)

    @pytest.mark.parametrize('grid, values, target', [
        pytest.param(('2023-01-01', 'MS'), np.ones(4), 'QS', id='aggregation'),
        pytest.param(('2023-01-01', 'QS'), [100, 110, 120], 'MS', id='interpolation'),
    ])
    def test_missing_column_is_ignored(self, aligner, grid, values, target):
        """A key naming an absent column is skipped, not an error."""
        df = _frame(values, *grid)

        result = aligner.convert_to_target(df, ['missing'], target)

        pd.testing.assert_frame_equal(result, df)

    def test_panel_missing_column_is_ignored(self, aligner):
        """Same in a panel: an (entity, absent column) key is skipped."""
        panel = _panel({'A': _frame([100, 110, 120], '2023-01-01', 'QS')})

        result = aligner.convert_to_target(panel, [('A', 'missing')], 'MS')

        pd.testing.assert_frame_equal(result, panel)

    def test_unknown_entity_is_ignored(self, aligner):
        """A key naming an entity absent from the panel is skipped, like an absent column."""
        panel = _panel({'A': _frame(np.ones(6), '2023-01-01', 'MS')})

        result = aligner.convert_to_target(panel, [('Z', 'x')], 'QS')

        pd.testing.assert_frame_equal(result, panel)

    def test_panel_column_without_observation_is_left_as_is(self, aligner):
        """An entity whose column is entirely NaN is left untouched, without error."""
        panel = _panel({'A': _frame([np.nan] * 8, '2023-01-01', 'MS')})

        result = aligner.convert_to_target(panel, [('A', 'x')], 'QS')

        pd.testing.assert_frame_equal(result, panel)

    def test_per_entity_target_missing_an_entity_raises(self, aligner):
        """A target dict without one of the keyed entities raises a ValueError naming it."""
        panel = _panel({'A': _frame(np.ones(6), '2023-01-01', 'MS'),
                        'B': _frame(np.ones(6), '2023-01-01', 'MS')})

        with pytest.raises(ValueError, match="'B'"):
            aligner.convert_to_target(panel, [('A', 'x'), ('B', 'x')], {'A': 'QS'})

    def test_duplicated_dates_raise_on_interpolation(self, aligner):
        """Duplicated dates are rejected by the interpolation route, with the same message."""
        df = with_duplicated_rows(_frame([100, 130, 160], '2023-01-01', 'QS'), n=1)

        with pytest.raises(ValueError, match='Duplicate dates: 2023-01-01'):
            aligner.convert_to_target(df, ['x'], 'MS')

    def test_empty_frame_is_returned_unchanged(self, aligner):
        """A frame without rows has nothing to convert."""
        df = _frame([1.0], '2023-01-01', 'MS').iloc[0:0]

        result = aligner.convert_to_target(df, ['x'], 'QS')

        pd.testing.assert_frame_equal(result, df)

    def test_plain_column_name_on_a_panel_converts_every_entity(self, aligner):
        """A bare column name on a panel designates the column of every entity."""
        panel = _panel({'A': _frame(np.arange(1, 7), '2023-01-01', 'MS'),
                        'B': _frame(np.arange(1, 7) * 2, '2023-01-01', 'MS')})

        result = aligner.convert_to_target(panel, ['x'], 'QS')

        # Valeur d'or : T1 = 6, T2 = 15 pour A ; le double pour B
        assert result['x'].dropna().tolist() == [6.0, 15.0, 12.0, 30.0]

    def test_plain_and_tuple_keys_are_equivalent(self, aligner, heterogeneous_coverage_panel):
        """A column name gives the same result as its (entity, column) keys, per-entity frequencies included."""
        tuple_keys = [(country, 'depenses_publiques_pib') for country in _COUNTRIES]

        result = aligner.convert_to_target(
            heterogeneous_coverage_panel, ['depenses_publiques_pib'], 'QS'
        )

        pd.testing.assert_frame_equal(
            result, aligner.convert_to_target(heterogeneous_coverage_panel, tuple_keys, 'QS')
        )

    def test_several_plain_column_names_on_a_panel(self, aligner):
        """Several bare column names are each expanded to every entity."""
        frame = _frame(np.arange(1, 7), '2023-01-01', 'MS')
        frame['y'] = np.arange(1, 7) * 10.0
        panel = _panel({'A': frame, 'B': frame})

        result = aligner.convert_to_target(panel, ['x', 'y'], 'QS')

        # Valeur d'or : x = 6, 15 et y = 60, 150 pour chacune des deux entités
        expected = pd.DataFrame({'x': [6.0, 15.0] * 2, 'y': [60.0, 150.0] * 2},
                                index=result.dropna().index)
        pd.testing.assert_frame_equal(result.dropna(), expected)

    def test_plain_and_tuple_keys_for_the_same_column_are_merged(self, aligner):
        """Mixing ``'x'`` and ``('A', 'x')`` converts each entity once."""
        panel = _panel({'A': _frame(np.arange(1, 7), '2023-01-01', 'MS'),
                        'B': _frame(np.arange(1, 7), '2023-01-01', 'MS')})

        result = aligner.convert_to_target(panel, [('A', 'x'), 'x'], 'QS')

        pd.testing.assert_frame_equal(result, aligner.convert_to_target(panel, ['x'], 'QS'))

    @pytest.mark.parametrize('index_kind', ['period', 'range'])
    def test_non_datetime_time_series_index_raises(self, aligner, index_kind):
        """A PeriodIndex (or any non-datetime index) is rejected with an explicit message."""
        df = _frame(np.arange(1, 7), '2023-01-01', 'MS')
        df = df.to_period('M') if index_kind == 'period' else df.reset_index(drop=True)

        with pytest.raises(TypeError, match='requires a DatetimeIndex as the index'):
            aligner.convert_to_target(df, ['x'], 'QS')

    def test_period_index_message_suggests_the_conversion(self, aligner):
        """The PeriodIndex message points to ``to_timestamp``."""
        df = _frame(np.arange(1, 7), '2023-01-01', 'MS').to_period('M')

        with pytest.raises(TypeError, match=r"to_timestamp\(how='start'\)"):
            aligner.convert_to_target(df, ['x'], 'QS')

    def test_period_level_of_a_panel_raises(self, aligner):
        """On a panel, the message names the last level of the MultiIndex."""
        panel = pd.concat({'A': _frame(np.arange(1, 7), '2023-01-01', 'MS').to_period('M')})

        with pytest.raises(TypeError, match='last level of the panel MultiIndex, got PeriodIndex'):
            aligner.convert_to_target(panel, [('A', 'x')], 'QS')


# ------------------------------------------------------------------ #
#  Panel à fréquence par (entité, colonne) : jeu réaliste             #
# ------------------------------------------------------------------ #

# Dépenses publiques : annuelles pour France et Italie, trimestrielles pour Allemagne
_DEPENSES_KEYS = [(country, 'depenses_publiques_pib') for country in _COUNTRIES]


class TestHeterogeneousCoveragePanel:
    """Frequency is a property of the (entity, column) pair (``heterogeneous_coverage_panel``).

    ``depenses_publiques_pib`` is yearly for France and Italie, quarterly for
    Allemagne; ``climat_affaires`` is monthly for France and Allemagne,
    structurally absent for Italie; every entity has its own coverage.
    """

    def test_quarterly_target_keeps_the_panel_index(self, aligner, heterogeneous_coverage_panel):
        """Interpolating yearly entities to quarter starts adds no row (already on the grid)."""
        result = aligner.convert_to_target(heterogeneous_coverage_panel, _DEPENSES_KEYS, 'QS')

        pd.testing.assert_index_equal(result.index, heterogeneous_coverage_panel.index)

    def test_quarterly_entity_is_unchanged_by_a_quarterly_target(
        self, aligner, heterogeneous_coverage_panel
    ):
        """Allemagne is already quarterly: Q→Q is the identity for it."""
        result = aligner.convert_to_target(heterogeneous_coverage_panel, _DEPENSES_KEYS, 'QS')

        pd.testing.assert_series_equal(
            result.loc['Allemagne', 'depenses_publiques_pib'],
            heterogeneous_coverage_panel.loc['Allemagne', 'depenses_publiques_pib'],
        )

    @pytest.mark.parametrize('country', ['France', 'Italie'])
    def test_yearly_entities_are_interpolated_quarterly(
        self, aligner, heterogeneous_coverage_panel, country
    ):
        """Each quarter start of a yearly entity is a quarter of the way to the next year."""
        source = heterogeneous_coverage_panel.loc[country, 'depenses_publiques_pib'].dropna()

        result = aligner.convert_to_target(heterogeneous_coverage_panel, _DEPENSES_KEYS, 'QS')

        # Valeur d'or : v(an) + k/4 × (v(an+1) − v(an)) au k-ième début de trimestre,
        # puis dernière valeur maintenue sur les trois trimestres suivants (limite 4)
        expected = {}
        next_values = list(source.to_numpy()[1:]) + [None]
        for (date, value), next_value in zip(source.items(), next_values):
            step = 0.0 if next_value is None else (next_value - value) / 4
            for k in range(4):
                expected[date + pd.DateOffset(months=3 * k)] = value + k * step
        np.testing.assert_allclose(
            result.loc[country, 'depenses_publiques_pib'].dropna().to_numpy(),
            _series(expected).to_numpy(),
        )

    def test_yearly_entities_stay_nan_outside_quarter_starts(
        self, aligner, heterogeneous_coverage_panel
    ):
        """Months that are not quarter starts, and the withdrawn last publication, stay NaN."""
        result = aligner.convert_to_target(heterogeneous_coverage_panel, _DEPENSES_KEYS, 'QS')

        france = result.loc['France', 'depenses_publiques_pib']
        # Hors débuts de trimestre, et après le dernier trimestre couvert (janvier 2024
        # retiré du jeu) : aucune valeur inventée
        outside = ~france.index.month.isin([1, 4, 7, 10]) | (france.index >= '2024-01-01')
        assert france[outside].isna().all()

    def test_per_entity_target_aggregates_only_complete_years(
        self, aligner, heterogeneous_coverage_panel
    ):
        """With a target dict, Allemagne (Q→Y) sums its complete years only."""
        targets = {'France': 'QS', 'Allemagne': 'YS', 'Italie': 'QS'}
        quarterly = heterogeneous_coverage_panel.loc['Allemagne', 'depenses_publiques_pib'].dropna()

        result = aligner.convert_to_target(heterogeneous_coverage_panel, _DEPENSES_KEYS, targets)

        # Valeur d'or : somme des 4 trimestres de chaque année complète (2019-2023) ;
        # 2018 (T3-T4 seulement) et 2024 (T1 retiré) sont incomplètes
        by_year = quarterly.groupby(quarterly.index.year)
        complete = by_year.sum()[by_year.count() == 4]
        expected = pd.Series(
            complete.to_numpy(),
            index=pd.DatetimeIndex([f'{year}-01-01' for year in complete.index], name='date'),
            name='depenses_publiques_pib',
        )
        pd.testing.assert_series_equal(
            result.loc['Allemagne', 'depenses_publiques_pib'].dropna(), expected
        )

    def test_structurally_absent_column_is_left_untouched(self, aligner, heterogeneous_coverage_panel):
        """``climat_affaires`` of Italie (zero observation) stays NaN, without error."""
        keys = [(country, 'climat_affaires') for country in _COUNTRIES]

        result = aligner.convert_to_target(heterogeneous_coverage_panel, keys, 'QS')

        assert result.loc['Italie', 'climat_affaires'].isna().all()

    def test_monthly_entities_are_summed_by_quarter(self, aligner, heterogeneous_coverage_panel):
        """``climat_affaires`` of France is summed over the three months of each quarter."""
        keys = [(country, 'climat_affaires') for country in _COUNTRIES]
        monthly = heterogeneous_coverage_panel.loc['France', 'climat_affaires']

        result = aligner.convert_to_target(heterogeneous_coverage_panel, keys, 'QS')

        # Valeur d'or : T1 2019 = janvier + février + mars 2019
        expected = monthly['2019-01-01':'2019-03-01'].sum()
        assert result.loc[('France', pd.Timestamp('2019-01-01')), 'climat_affaires'] == (
            pytest.approx(expected)
        )

    @pytest.mark.parametrize('country, first, last', [
        # Bornes : première ancre annuelle (historique antérieur à la grille mensuelle)
        # et dernière date de la grille mensuelle de l'entité
        pytest.param('France', '2015-01-01', '2024-07-01', id='France'),
        pytest.param('Allemagne', '2016-01-01', '2024-04-01', id='Allemagne'),
        pytest.param('Italie', '2016-01-01', '2024-07-01', id='Italie'),
    ])
    def test_monthly_densification_respects_each_entity_bounds(
        self, aligner, heterogeneous_coverage_panel, country, first, last
    ):
        """Yearly ``balance_commerciale_annuelle`` → MS fills each entity's own span, monthly."""
        keys = [(entity, 'balance_commerciale_annuelle') for entity in _COUNTRIES]

        result = aligner.convert_to_target(heterogeneous_coverage_panel, keys, 'MS')

        pd.testing.assert_index_equal(
            result.loc[country].index,
            pd.date_range(first, last, freq='MS', name='date'),
            exact=False,
        )


# ------------------------------------------------------------------ #
#  Robustesse : désordre, noms, niveaux d'index, index irrégulier     #
# ------------------------------------------------------------------ #

# Clés mélangeant agrégation (climat_affaires → Q) et interpolation (annuelles → Q)
_MIXED_PANEL_KEYS = (
    [(country, 'climat_affaires') for country in _COUNTRIES]
    + _DEPENSES_KEYS
    + [(country, 'balance_commerciale_annuelle') for country in _COUNTRIES]
)


class TestRobustness:
    """The result does not depend on row order, column names, index names or index depth."""

    @pytest.fixture
    def reference(self, aligner, heterogeneous_coverage_panel) -> pd.DataFrame:
        """Conversion of the sorted panel, reference of the perturbed variants."""
        return aligner.convert_to_target(heterogeneous_coverage_panel, _MIXED_PANEL_KEYS, 'QS')

    @pytest.mark.parametrize('seed', [0, 7])
    def test_shuffled_panel_gives_the_same_result(
        self, aligner, heterogeneous_coverage_panel, reference, seed
    ):
        """Shuffling the panel rows gives the same values once sorted back."""
        result = aligner.convert_to_target(
            shuffle_rows(heterogeneous_coverage_panel, seed=seed), _MIXED_PANEL_KEYS, 'QS'
        )

        pd.testing.assert_frame_equal(result.sort_index(), reference.sort_index())

    def test_reversed_entities_give_the_same_result(
        self, aligner, heterogeneous_coverage_panel, reference
    ):
        """The order of entity blocks has no effect."""
        result = aligner.convert_to_target(
            reverse_entities(heterogeneous_coverage_panel), _MIXED_PANEL_KEYS, 'QS'
        )

        pd.testing.assert_frame_equal(result.sort_index(), reference.sort_index())

    @pytest.mark.parametrize('seed', [0, 7])
    def test_shuffled_time_series_gives_the_same_result(
        self, aligner, irregular_index_timeseries, seed
    ):
        """Shuffling a time series (aggregation and interpolation keys) has no effect."""
        keys = ['production_industrielle', 'balance_commerciale_annuelle']
        expected = aligner.convert_to_target(irregular_index_timeseries, keys, 'QS')

        result = aligner.convert_to_target(
            shuffle_rows(irregular_index_timeseries, seed=seed), keys, 'QS'
        )

        pd.testing.assert_frame_equal(result.sort_index(), expected.sort_index(), check_freq=False)

    def test_special_column_names_give_the_same_result(
        self, aligner, heterogeneous_coverage_panel, reference
    ):
        """Spaces, accents, ``%``, ``/`` and parentheses in column names change nothing."""
        renamed, mapping = with_special_column_names(heterogeneous_coverage_panel)
        keys = [(entity, mapping[col]) for entity, col in _MIXED_PANEL_KEYS]

        result = aligner.convert_to_target(renamed, keys, 'QS')

        # Restauration des noms d'origine pour la comparaison
        restored = result.rename(columns={new: old for old, new in mapping.items()})
        pd.testing.assert_frame_equal(restored, reference)

    def test_non_standard_index_names_give_the_same_result(
        self, aligner, heterogeneous_coverage_panel, reference
    ):
        """Levels named neither ``entity`` nor ``date`` are handled by position."""
        renamed = with_index_names(heterogeneous_coverage_panel, ['pays', 'periode'])

        result = aligner.convert_to_target(renamed, _MIXED_PANEL_KEYS, 'QS')

        pd.testing.assert_frame_equal(result, with_index_names(reference, ['pays', 'periode']))

    def test_three_level_index_gives_the_same_values(
        self, aligner, heterogeneous_coverage_panel, reference
    ):
        """A (region, country, date) index with (region, country, column) keys gives the same values."""
        three_levels = to_three_level_index(heterogeneous_coverage_panel)
        keys = [('Zone euro', entity, col) for entity, col in _MIXED_PANEL_KEYS]

        result = aligner.convert_to_target(three_levels, keys, 'QS')

        pd.testing.assert_frame_equal(result, to_three_level_index(reference))

    def test_irregular_time_series_is_densified_to_a_regular_grid(
        self, aligner, irregular_index_timeseries
    ):
        """Yearly anchors before the monthly grid are bridged: the output grid is regular."""
        # Ancres annuelles 2015-2017 isolées avant la grille mensuelle 2018-01 → 2024-07
        result = aligner.convert_to_target(
            irregular_index_timeseries, ['balance_commerciale_annuelle'], 'MS'
        )

        expected = pd.date_range('2015-01-01', '2024-07-01', freq='MS', name='date')
        pd.testing.assert_index_equal(result.index, expected, exact=False)

    def test_irregular_time_series_keeps_every_observation(
        self, aligner, irregular_index_timeseries
    ):
        """No observed value of any column is altered by the densification."""
        result = aligner.convert_to_target(
            irregular_index_timeseries, ['balance_commerciale_annuelle'], 'MS'
        )

        # Cellules observées seulement (``stack`` conserve les cellules vides)
        observed = irregular_index_timeseries.stack().dropna()
        pd.testing.assert_series_equal(result.stack().loc[observed.index], observed)


# ------------------------------------------------------------------ #
#  Gardes défensives des méthodes privées                             #
# ------------------------------------------------------------------ #
@pytest.mark.internal
class TestPrivateGuards:
    """Defensive guards of the private helpers, unreachable through ``convert_to_target``.

    ``convert_to_target`` only calls ``_aggregate_to_target`` and
    ``_interpolate_to_target`` with non-empty key lists, and routes every
    column without a detectable frequency (no or one observation) to the
    aggregation: these guards cannot be reached from the public API.
    """

    @pytest.mark.parametrize('method', ['_aggregate_to_target', '_interpolate_to_target'])
    def test_empty_keys_return_the_frame_unchanged(self, aligner, method):
        """Both dataset-level helpers return their input when no key is given."""
        df = _frame([100, 110, 120], '2023-01-01', 'QS')

        result = getattr(aligner, method)(df, [], 'MS')

        assert result is df

    @pytest.mark.parametrize('values', [
        pytest.param([np.nan] * 3, id='no-observation'),
        pytest.param([100.0, np.nan, np.nan], id='single-observation'),
    ])
    def test_interpolating_a_column_without_frequency_leaves_it_unchanged(self, aligner, values):
        """Fewer than two observations: the column is left as is."""
        df = _frame(values, '2023-01-01', 'QS')

        result = aligner._interpolate_to_target(df, ['x'], 'MS')

        pd.testing.assert_frame_equal(result, df)
