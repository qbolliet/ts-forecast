"""Hand-computed divisors of ``StageScaler``.

Scenarios: constant and calendar modes on every (source, prediction) frequency pair, the
period a covariate cell spans, and the per-row target divisor.
"""

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objets testés
from tsforecast.frequency.stage_scaler import StageScaler

# Outils de test partagés du paquet
from tests.unit.frequency.stage_scaler.helpers import (
    _EXACT,
    _PANEL_F_BLOCKS,
    _panel_grid,
)


# =============================================================================
# Valeurs d'or : 'constant'
# =============================================================================
# Valeur d'or : nombre de sous-périodes de prédiction dans une période de la
# variable, dérivé à la main (12 mois, 4 trimestres, 3 mois par trimestre)
_CONSTANT_GOLD = [
    ('Y', 'Y', 1.0), ('Y', 'Q', 4.0), ('Y', 'M', 12.0),
    ('Q', 'Y', 1.0 / 4.0), ('Q', 'Q', 1.0), ('Q', 'M', 3.0),
    ('M', 'Y', 1.0 / 12.0), ('M', 'Q', 1.0 / 3.0), ('M', 'M', 1.0),
]


class TestConstantGoldenValues:
    """'constant' : one invariant factor per (source, prediction) pair."""

    @pytest.mark.parametrize('position', ['', 'S', 'E'], ids=['base', 'start', 'end'])
    @pytest.mark.parametrize(
        'source, pred, expected', _CONSTANT_GOLD,
        ids=[f'{s}->{p}' for s, p, _ in _CONSTANT_GOLD],
    )
    def test_target_divisor_gold(self, source, pred, expected, position):
        """target_divisor counts the pred sub-periods of one source period, any position.

        A coarser stage than the variable (a monthly block at a quarterly
        stage) yields a fractional divisor, never floored at 1.0.
        """
        divisor = StageScaler().target_divisor(
            'v', source_freq=source + position, pred_freq=pred + position
        )
        assert divisor == pytest.approx(expected, rel=_EXACT)

    @pytest.mark.parametrize(
        'source, pred, expected', _CONSTANT_GOLD,
        ids=[f'{s}->{p}' for s, p, _ in _CONSTANT_GOLD],
    )
    def test_fit_scale_factor_equals_the_stage_scalar(self, source, pred, expected):
        """Without produced_freq the baked factor is the stage scalar itself."""
        factor = StageScaler().fit_scale_factor('v', source_freq=source, pred_freq=pred)
        assert factor == pytest.approx(expected, rel=_EXACT)

    @pytest.mark.parametrize(
        'source, pred, expected',
        [('M', 'D', 30.0), ('Q', 'D', 91.0), ('Y', 'D', 365.0)],
        ids=['M->D', 'Q->D', 'Y->D'],
    )
    def test_daily_stage_uses_the_mean_factor(self, source, pred, expected):
        """'constant' never looks at the calendar: one month is 30 days, whatever the month."""
        divisor = StageScaler().target_divisor(
            'v', source_freq=source, pred_freq=pred, index=pd.to_datetime(['2024-02-29'])
        )
        assert divisor == expected

    @pytest.mark.parametrize(
        'carried, stage, expected',
        [('Y', 'M', 12.0), ('Q', 'M', 3.0), ('Y', 'Q', 4.0), ('M', 'Q', 1.0 / 3.0), ('M', 'M', 1.0)],
        ids=['Y-cell@M', 'Q-cell@M', 'Y-cell@Q', 'M-cell@Q', 'M-cell@M'],
    )
    def test_feature_divisor_gold(self, carried, stage, expected):
        """A covariate divides by the stage sub-periods held by the period its cell spans.

        The cell spans a period of its grid ('identity' way): a block grid at
        the ``carried`` frequency carries cells of that period, coarser or
        finer than the stage.
        """
        divisors = StageScaler().feature_divisors(
            columns=['x'],
            column_frequencies={'x': carried},
            ways={'x': 'identity'},
            grid_freq=carried,
            stage_freq=stage,
        )
        assert divisors['x'] == pytest.approx(expected, rel=_EXACT)


# =============================================================================
# Valeurs d'or : 'calendar'
# =============================================================================
# Couples où l'étape est au moins aussi fine que la variable : le décompte
# calendaire de mois / trimestres est invariant (12, 4, 3, 1)
_CALENDAR_COARSER_SOURCE = [
    ('Y', 'Y', 1.0), ('Y', 'Q', 4.0), ('Y', 'M', 12.0),
    ('Q', 'Q', 1.0), ('Q', 'M', 3.0), ('M', 'M', 1.0),
]

# Valeur d'or : jours réels de la période contenant chaque date, à la main
# (février 2023 = 28, février 2024 = 29 ; T1 2023 = 31 + 28 + 31 = 90,
# T1 2024 = 91, T2 = 30 + 31 + 30 = 91, T3 = T4 = 92 ; 2023 = 365, 2024 = 366)
_CALENDAR_DAYS = [
    ('M', ['2023-02-01', '2024-02-01', '2024-01-01', '2024-04-01'],
     ['2023-02-28', '2024-02-29', '2024-01-31', '2024-04-30'], [28.0, 29.0, 31.0, 30.0]),
    ('Q', ['2023-01-01', '2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'],
     ['2023-03-31', '2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'],
     [90.0, 91.0, 91.0, 92.0, 92.0]),
    ('Y', ['2023-01-01', '2024-01-01'], ['2023-12-31', '2024-12-31'], [365.0, 366.0]),
]


class TestCalendarGoldenValues:
    """'calendar' : real calendar count, one divisor per row."""

    @pytest.mark.parametrize('grid_freq', ['MS', 'ME'], ids=['start', 'end'])
    @pytest.mark.parametrize(
        'source, pred, expected', _CALENDAR_COARSER_SOURCE,
        ids=[f'{s}->{p}' for s, p, _ in _CALENDAR_COARSER_SOURCE],
    )
    def test_target_divisor_gold_on_month_quarter_year(self, source, pred, expected, grid_freq):
        """Months in a year, quarters in a year, months in a quarter: one Series per row."""
        index = pd.date_range('2024-01-01', periods=12, freq=grid_freq)
        divisors = StageScaler(scale_features='calendar').target_divisor(
            'v', source_freq=source, pred_freq=pred, index=index
        )
        pd.testing.assert_series_equal(divisors, pd.Series(expected, index=index, dtype=float))

    @pytest.mark.parametrize(
        'source, pred, stage_period',
        [('M', 'Q', 'Q'), ('M', 'Y', 'Y'), ('Q', 'Y', 'Y')],
        ids=['M->Q', 'M->Y', 'Q->Y'],
    )
    def test_fractional_divisor_under_calendar(self, source, pred, stage_period):
        """A block finer than the stage divides by a fraction, < 1, summing to 1 per stage period.

        The divisor is "> 1 for a block coarser than the stage and
        < 1 for a finer one", and the mode applies to every divisor.
        The shares of one stage period sum to 1 exactly.
        """
        index = pd.date_range('2024-01-31', periods=12, freq='ME')
        if source == 'Q':
            index = pd.date_range('2024-03-31', periods=4, freq='QE')
        divisors = StageScaler(scale_features='calendar').target_divisor(
            'v', source_freq=source, pred_freq=pred, index=index
        )
        assert (divisors < 1.0).all()
        sums = divisors.groupby(index.to_period(stage_period)).sum()
        np.testing.assert_allclose(sums.to_numpy(), 1.0, rtol=0, atol=_EXACT)

    def test_fractional_feature_divisor_under_calendar(self):
        """A monthly cell at a quarterly stage divides by its share of days: 31 / 29 / 31 of 91.

        Monthly block grid, quarterly stage: each cell spans one
        month.
        """
        index = pd.date_range('2024-01-31', periods=3, freq='ME')
        divisors = StageScaler(scale_features='calendar').feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': 'M'},
            ways={'m1': 'identity'},
            grid_freq='M',
            stage_freq='Q',
            index=index,
        )
        np.testing.assert_allclose(
            divisors['m1'].to_numpy(), [31 / 91, 29 / 91, 31 / 91], rtol=_EXACT
        )

    @pytest.mark.parametrize('position', ['start', 'end'])
    @pytest.mark.parametrize(
        'source, pred, starts, ends, expected',
        [
            # Valeur d'or : jours du mois / jours du trimestre (T1 2024 = 91,
            # T2 = 91, T3 = 92 ; T1 2023 = 90)
            ('M', 'Q', ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-07-01', '2023-02-01'],
             ['2024-01-31', '2024-02-29', '2024-03-31', '2024-04-30', '2024-07-31', '2023-02-28'],
             [31 / 91, 29 / 91, 31 / 91, 30 / 91, 31 / 92, 28 / 90]),
            # Jours du mois / jours de l'année (2024 = 366, 2023 = 365)
            ('M', 'Y', ['2024-02-01', '2023-02-01'], ['2024-02-29', '2023-02-28'], [29 / 366, 28 / 365]),
            # Jours du trimestre / jours de l'année
            ('Q', 'Y', ['2024-01-01', '2024-07-01', '2023-01-01'], ['2024-03-31', '2024-09-30', '2023-03-31'],
             [91 / 366, 92 / 366, 90 / 365]),
        ],
        ids=['M->Q', 'M->Y', 'Q->Y'],
    )
    def test_calendar_share_of_days_gold(self, source, pred, starts, ends, expected, position):
        """'calendar' fractional divisor = real share of days of the stage period, any position.

        'constant' keeps the uniform ratio
        (1/3 for every month), 'calendar' weighs each block period by its
        days (January 31/91, February 29/91 in 2024).
        """
        dates = pd.DatetimeIndex(starts if position == 'start' else ends)
        suffix = 'S' if position == 'start' else 'E'
        divisors = StageScaler(scale_features='calendar').target_divisor(
            'v', source_freq=source + suffix, pred_freq=pred + suffix, index=dates
        )
        np.testing.assert_allclose(divisors.to_numpy(), expected, rtol=_EXACT)

    def test_calendar_share_is_an_exact_ratio_of_days(self):
        """The share is the exact ratio of whole period lengths, without sub-microsecond drift."""
        # Valeur d'or : 31 / 91, 29 / 91 et 31 / 91 au bit près (durées entières de nanosecondes,
        # une seule division flottante) ; une longueur de période amputée d'une fraction de
        # microseconde fausserait le dernier chiffre
        dates = pd.DatetimeIndex(['2024-01-31', '2024-02-29', '2024-03-31'])
        divisors = StageScaler(scale_features='calendar').target_divisor(
            'v', source_freq='ME', pred_freq='QE', index=dates
        )
        assert divisors.tolist() == [31 / 91, 29 / 91, 31 / 91]

    def test_calendar_share_falls_back_on_the_constant_ratio(self):
        """A frequency without pandas Period (semi-monthly) falls back on the constant ratio, 1/2."""
        index = pd.to_datetime(['2024-01-15', '2024-01-31'])
        divisors = StageScaler(scale_features='calendar').target_divisor(
            'v', source_freq='SM', pred_freq='M', index=index
        )
        assert divisors.tolist() == [0.5, 0.5]

    def test_calendar_share_on_a_mutualized_block(self, mixed_freq_panel_multifrequency):
        """PANEL-F, stage Q, 'calendar': the monthly IT block divides by its days in the quarter.

        IT carries 10.0 every month of 2021: January becomes 10 / (31/90),
        February 10 / (28/90), March 10 / (31/90) — weighted by days, their
        shares of the quarter sum to the quarterly scale, 3 months x 10.
        """
        y = mixed_freq_panel_multifrequency['v'].dropna()
        X = mixed_freq_panel_multifrequency.loc[y.index, ['m1']]
        scaler = StageScaler(
            scale_features='calendar', source_freq=_PANEL_F_BLOCKS, pred_freq='Q',
            column_frequencies={'m1': 'M'},
        ).fit(X, y)

        first_quarter = scaler.target_divisor_.loc['IT'].iloc[:3]

        np.testing.assert_allclose(first_quarter.to_numpy(), [31 / 90, 28 / 90, 31 / 90], rtol=_EXACT)

    @pytest.mark.parametrize('position', ['start', 'end'])
    @pytest.mark.parametrize(
        'source, starts, ends, expected', _CALENDAR_DAYS, ids=['M->D', 'Q->D', 'Y->D'],
    )
    def test_daily_stage_counts_real_days(self, source, starts, ends, expected, position):
        """February 28 / 29, quarters of 90 / 91 / 92 days, years of 365 / 366 days.

        The count is read off the period holding each date: a date at period
        start and the matching period end give the same divisor, and so do
        the ``S`` / ``E`` aliases of the variable's frequency.
        """
        dates = pd.DatetimeIndex(starts if position == 'start' else ends)
        alias = source + ('S' if position == 'start' else 'E')
        divisors = StageScaler(scale_features='calendar').target_divisor(
            'v', source_freq=alias, pred_freq='D', index=dates
        )
        assert divisors.tolist() == expected

    def test_feature_divisors_count_real_days_per_row(self):
        """A daily covariate aggregated onto months divides by the real days of each month.

        Daily covariate aggregated onto the monthly grid of a daily stage
        ('aggregate' way: the cell spans one month): divisor = days of each
        month, 31 / 29 / 31 for 2024.
        """
        index = pd.date_range('2024-01-31', periods=3, freq='ME')
        divisors = StageScaler(scale_features='calendar').feature_divisors(
            columns=['d1'],
            column_frequencies={'d1': 'D'},
            ways={'d1': 'aggregate'},
            grid_freq='M',
            stage_freq='D',
            index=index,
        )
        assert divisors['d1'].tolist() == [31.0, 29.0, 31.0]

    def test_fit_scale_factor_is_frozen_on_the_fit_grid(self):
        """Under 'calendar' the baked factor is a Series on the prediction grid of the fit."""
        index = pd.to_datetime(['2021-01-31', '2021-02-28'])
        factor = StageScaler(scale_features='calendar').fit_scale_factor(
            'm1', source_freq='M', pred_freq='D', index=index
        )
        pd.testing.assert_series_equal(factor, pd.Series([31.0, 28.0], index=index))

    @pytest.mark.parametrize('method', ['target_divisor', 'fit_scale_factor'])
    def test_calendar_without_index_raises(self, method):
        """The per-row count needs a grid: without index, an explicit ValueError."""
        scaler = StageScaler(scale_features='calendar')
        with pytest.raises(ValueError, match=r"calendar"):
            getattr(scaler, method)('m1', source_freq='M', pred_freq='D')

    def test_calendar_feature_divisors_without_index_raise(self):
        """Same prerequisite on the covariate side."""
        with pytest.raises(ValueError, match=r"index"):
            StageScaler(scale_features='calendar').feature_divisors(
                columns=['d1'],
                column_frequencies={'d1': 'D'},
                ways={'d1': 'aggregate'},
                grid_freq='M',
                stage_freq='D',
            )


# =============================================================================
# Règle de la période couverte et replis
# =============================================================================
class TestCarriedPeriodRule:
    """A covariate divides by the period its cell SPANS on its grid."""

    @pytest.mark.parametrize('way', ['identity', 'aggregate', 'interpolate', 'mirror'])
    def test_every_materialized_way_spans_one_grid_period(self, way):
        """Identity, aggregation, interpolation, mirror: the cell spans one period of the grid.

        Daily, monthly and quarterly covariates on a YEARLY grid all span one
        year, and come back to the monthly stage through the same divisor 12.
        """
        divisors = StageScaler().feature_divisors(
            columns=['d1', 'm1', 'q1'],
            column_frequencies={'d1': 'D', 'm1': 'M', 'q1': 'Q'},
            ways={'d1': way, 'm1': way, 'q1': way},
            grid_freq='Y',
            stage_freq='M',
        )
        assert divisors.to_dict() == {'d1': 12.0, 'm1': 12.0, 'q1': 12.0}

    def test_raw_anchors_column_spans_one_column_period(self):
        """Read at its anchors, the column spans ITS period, not the grid's."""
        divisors = StageScaler().feature_divisors(
            columns=['q1', 'a2'],
            column_frequencies={'q1': 'Q', 'a2': 'Y'},
            ways={'q1': 'raw_anchors', 'a2': 'raw_anchors'},
            grid_freq='M',
            stage_freq='M',
        )
        assert divisors.to_dict() == {'q1': 3.0, 'a2': 12.0}

    @pytest.mark.parametrize(
        'column_freq, expected', [('M', 1.0), ('Q', 1.0), ('Y', 4.0)], ids=['finer', 'equal', 'coarser'],
    )
    def test_raw_anchors_of_a_column_at_least_as_fine_as_the_grid_span_the_grid(
        self, column_freq, expected
    ):
        """Under 'raw_anchors', a column at least as fine as the grid spans one grid period.

        The materializer brings such a column back to identity or to exact
        aggregation (rank 1): on a quarterly grid at stage
        Q, a monthly or quarterly column divides by 1; only the
        annual column, read at its anchors, spans its own period (4).
        """
        divisors = StageScaler().feature_divisors(
            columns=['x'],
            column_frequencies={'x': column_freq},
            ways={'x': 'raw_anchors'},
            grid_freq='Q',
            stage_freq='Q',
        )
        assert divisors['x'] == pytest.approx(expected, rel=_EXACT)

    def test_raw_anchors_per_entity_follow_each_entity_column(self):
        """PANEL-F ``v`` at stage Q under 'raw_anchors': FR anchors 4, DE and IT one quarter.

        Valeur d'or : FR annuelle (ancre, 4 trimestres), DE trimestrielle
        (identité, 1), IT mensuelle (agrégation exacte, 1) — et non 1/3, la
        valeur ponctuelle d'un mois n'étant plus jamais servie.
        """
        index = _panel_grid(('DE', 'FR', 'IT'), periods=4)
        divisors = StageScaler().feature_divisors(
            columns=['v'],
            column_frequencies={'v': _PANEL_F_BLOCKS},
            ways={'v': 'raw_anchors'},
            grid_freq='Q',
            stage_freq='Q',
            index=index,
        )
        assert divisors['v'].tolist() == [1.0] * 4 + [4.0] * 4 + [1.0] * 4

    def test_training_and_prediction_grids_are_comparable(self):
        """On a time series at stage M, q1 aggregated yearly divides by 12, interpolated monthly by 1.

        The yearly mean of the training feature equals the mean of the
        monthly prediction feature exactly.
        """
        scaler = StageScaler()
        train = scaler.feature_divisors(
            columns=['q1'], column_frequencies={'q1': 'Q'}, ways={'q1': 'aggregate'},
            grid_freq='Y', stage_freq='M',
        )
        pred = scaler.feature_divisors(
            columns=['q1'], column_frequencies={'q1': 'Q'}, ways={'q1': 'interpolate'},
            grid_freq='M', stage_freq='M',
        )
        # Valeur d'or : 12 mois de 10 sommés en 120 sur l'année, ramenés à 10
        assert 120.0 / train['q1'] == 10.0 / pred['q1']

    def test_absent_way_defaults_to_the_grid_period(self):
        """A column missing from ``ways`` is materialized: its cell spans the grid."""
        divisors = StageScaler().feature_divisors(
            columns=['a2'], column_frequencies={'a2': 'Y'}, ways={},
            grid_freq='Q', stage_freq='M',
        )
        assert divisors['a2'] == 3.0


# =============================================================================
# Échelle par ligne de la cible
# =============================================================================
class TestPerRowTargetDivisor:
    """Per-row scale: each row of y carries its production frequency."""

    def test_per_row_divisors_table(self):
        """Per-row divisors: 120 / Y -> 10.0, 28 / Q -> 9.33, 30 / Q -> 10.0."""
        scaler = StageScaler()
        index = pd.to_datetime(['2021-12-31', '2021-03-31', '2021-06-30'])
        y_train = pd.Series([120.0, 28.0, 30.0], index=index)
        divisors = scaler.target_divisor(
            'a1', source_freq='Y', pred_freq='M',
            produced_freq=pd.Series(['Y', 'Q', 'Q'], index=index),
        )
        # Valeur d'or : 120 / 12, 28 / 3, 30 / 3
        np.testing.assert_allclose(
            scaler.apply(y_train, divisors).to_numpy(), [10.0, 28.0 / 3.0, 10.0], rtol=_EXACT
        )

    def test_unit_per_row_divisors_are_not_short_circuited(self):
        """B12: under 'constant', per-row divisors all equal to 1.0 stay a Series.

        A pure ``StageScaler`` case.
        """
        index = pd.to_datetime(['2021-01-31', '2021-02-28'])
        divisors = StageScaler(scale_features='constant').target_divisor(
            'a1', source_freq='M', pred_freq='M', index=index,
            produced_freq=pd.Series(['M', 'M'], index=index),
        )
        assert isinstance(divisors, pd.Series)
        assert divisors.tolist() == [1.0, 1.0]

    def test_per_row_divisor_under_calendar_mode(self):
        """Under 'calendar' the per-row divisor follows the real count of the period."""
        index = pd.to_datetime(['2021-02-28', '2024-02-29'])
        divisors = StageScaler(scale_features='calendar').target_divisor(
            'm1', source_freq='M', pred_freq='D', produced_freq=pd.Series(['M', 'M'], index=index)
        )
        assert divisors.tolist() == [28.0, 29.0]

    def test_baked_factor_ignores_production_frequencies(self):
        """The baked factor stays the stage scalar, even when y mixes scales."""
        index = pd.to_datetime(['2021-12-31', '2021-03-31'])
        scaler = StageScaler()
        rows = scaler.target_divisor(
            'a1', source_freq='Y', pred_freq='M', produced_freq=pd.Series(['Y', 'Q'], index=index)
        )
        assert (scaler.fit_scale_factor('a1', 'Y', 'M'), rows.tolist()) == (12.0, [12.0, 3.0])

    def test_per_entity_stage_frequency_with_production_frequencies(self):
        """Each row reads its stage frequency from its entity, then divides by its production one.

        DE at a quarterly stage, FR at a monthly one; rows produced at Y or Q.
        """
        index = pd.MultiIndex.from_tuples(
            [('DE', pd.Timestamp('2021-12-31')), ('DE', pd.Timestamp('2021-06-30')),
             ('FR', pd.Timestamp('2021-12-31')), ('FR', pd.Timestamp('2021-06-30'))],
            names=['country', 'date'],
        )
        divisors = StageScaler().target_divisor(
            'a1', source_freq='Y', pred_freq={('DE',): 'Q', ('FR',): 'M'},
            produced_freq=pd.Series(['Y', 'Q', 'Y', 'Q'], index=index),
        )
        # Valeur d'or : factor(Q, Y) = 4, factor(Q, Q) = 1, factor(M, Y) = 12, factor(M, Q) = 3
        assert divisors.tolist() == [4.0, 1.0, 12.0, 3.0]

    def test_per_entity_stage_frequency_requires_an_entity_level(self):
        """A per-entity stage binding with produced_freq on a flat index raises."""
        index = pd.to_datetime(['2021-12-31'])
        with pytest.raises(ValueError, match=r"MultiIndex"):
            StageScaler().target_divisor(
                'a1', source_freq='Y', pred_freq={('DE',): 'Q', ('FR',): 'M'},
                produced_freq=pd.Series(['Y'], index=index),
            )
