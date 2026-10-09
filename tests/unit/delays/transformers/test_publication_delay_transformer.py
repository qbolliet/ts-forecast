"""Unit tests for ``PublicationDelayTransformer`` (``tsforecast.delays.transformers``): parameters and ``fit``.

Covers ``__init__`` (validation of ``strategy``, ``reference_point``,
``handle_missing_delays`` and ``default_values``, warnings), the resolution done
by ``fit`` (``prediction_date_``, ``inferred_params_``, ``detected_frequencies_``,
``shift_params``, ``mask_params``, ``fit_report_``) through the private helpers
``_infer_parameters_from_delays``, ``_build_parameter_dict``,
``_build_target_frequency_dict``, ``_compute_shift_periods``,
``_compute_mask_periods``, ``_parameter_source`` and ``_build_fit_report``
(exercised through ``fit`` only), and the sklearn protocol. ``transform`` /
``inverse_transform`` are tested in ``test_publication_delay_transformer_transform.py``,
the per-entity factories in ``test_factories.py``.

Gold values. Monthly series (``MS``) of 2023, prediction date 2023-12-15. The
number of periods is counted on the calendar: ``n_periods = -k``, ``k`` being the
number of months between December and the last month published on December 15
(a month published on the prediction date itself counts as published). With 45
days from the start, the last period published on December 15 is October
(October 1 + 45 days = November 15; November 1 + 45 days = December 16), two
months before December, hence -2. From the end, the delay runs from the first
day of the next month (October ends on November 1).

The tests were triaged and completed by prompt D4 (``tests_and_refactoring_prompts.md``);
the anomalies found are registered in ``tests/ANOMALIES.md`` (``ANO-DELAYS-028`` to ``-041``, all fixed).
``TestCalendarBoundaries`` (prompt D5) covers ``ANO-DELAYS-042`` (fixed): the former
30-day month convention was one period off when the delay ended near a calendar
boundary.
"""
# Modules de base
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

# Classe à tester et producteurs réels de tableaux de délais
from tsforecast.delays.calculator import calculate_applicable_delay
from tsforecast.delays.data_manager import compare_and_detect_delays
from tsforecast.delays.transformers import PublicationDelayTransformer

TS = pd.Timestamp

# Date de prédiction des valeurs d'or : 14 jours écoulés depuis le début de décembre 2023
PREDICTION = '2023-12-15'


# =============================================================================
# Constructeurs locaux de petits jeux à valeurs d'or calculables
# =============================================================================
def _monthly(columns=('GDP', 'CPI'), periods: int = 12) -> pd.DataFrame:
    """Build a monthly (``MS``) frame of 2023 whose column ``i`` holds ``(i + 1) * [0, 1, ...]``.

    Args:
        columns: Column names.
        periods: Number of months from January 2023.

    Returns:
        Deterministic float frame indexed by month starts.
    """
    index = pd.date_range('2023-01-01', periods=periods, freq='MS')
    return pd.DataFrame({col: np.arange(periods, dtype=float) * (i + 1) for i, col in enumerate(columns)}, index=index)


def _daily(columns=('GDP',), periods: int = 90) -> pd.DataFrame:
    """Build a daily frame from 2024-01-01 (January to March 30, 2024 for 90 days).

    Args:
        columns: Column names.
        periods: Number of days.

    Returns:
        Deterministic float frame indexed by days.
    """
    index = pd.date_range('2024-01-01', periods=periods, freq='D')
    return pd.DataFrame({col: np.arange(periods, dtype=float) for col in columns}, index=index)


def _fit(X: pd.DataFrame, **kwargs) -> PublicationDelayTransformer:
    """Fit a transformer at the gold prediction date, the warnings of ``fit`` being silenced.

    Args:
        X: Data to fit on.
        **kwargs: Constructor arguments (``prediction_date`` defaults to 2023-12-15).

    Returns:
        The fitted transformer.
    """
    kwargs.setdefault('prediction_date', PREDICTION)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return PublicationDelayTransformer(**kwargs).fit(X)


def _delays_frame(columns=('GDP', 'CPI'), delays=(45.0, 20.0), unit='D', reference_point='start',
                  frequency='Q') -> pd.DataFrame:
    """Build a flat delays table with the columns expected by the transformer.

    Args:
        columns: Variable names (column ``column``).
        delays: Delay of each variable.
        unit: Unit of every delay.
        reference_point: Reference point of every delay.
        frequency: Target frequency of every delay.

    Returns:
        Table with the columns ``column``, ``delay``, ``unit``, ``reference_point``, ``frequency``.
    """
    n = len(columns)
    return pd.DataFrame({'column': list(columns), 'delay': list(delays), 'unit': [unit] * n,
                         'reference_point': [reference_point] * n, 'frequency': [frequency] * n})


# =============================================================================
# Fixtures (déterministes)
# =============================================================================
@pytest.fixture
def sample_time_series() -> pd.DataFrame:
    """Monthly 2023-2024 series with three deterministic columns."""
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='MS')
    steps = np.arange(len(dates), dtype=float)
    return pd.DataFrame({'GDP': 1000 + steps, 'inflation': 2 + steps / 10, 'unemployment': 5 - steps / 100},
                        index=dates)


@pytest.fixture
def delays_dict_simple() -> dict:
    """Delays in days of the three variables of ``sample_time_series``."""
    return {'GDP': 45.0, 'inflation': 30.0, 'unemployment': 15.0}


@pytest.fixture
def delays_dataframe() -> pd.DataFrame:
    """Flat delays table with the current column names (``column``, ``frequency``)."""
    # Noms de colonnes actuels : 'variable' -> 'column' (aed46c5), 'target_frequency' -> 'frequency' (ae347b2)
    return pd.DataFrame({
        'column': ['GDP', 'inflation', 'unemployment'],
        'delay': [45.0, 30.0, 15.0],
        'unit': ['D', 'D', 'D'],
        'reference_point': ['end', 'end', 'end'],
        'frequency': ['M', 'M', 'M'],
    })


# =============================================================================
# Initialisation et validation des paramètres
# =============================================================================
class TestPublicationDelayTransformerInit:
    """Constructor: parameters stored as given, invalid values rejected, warnings."""

    def test_init_with_dict(self, delays_dict_simple):
        """Parameters are stored unchanged."""
        transformer = PublicationDelayTransformer(delays=delays_dict_simple, strategy='shift',
                                                  prediction_date='2024-01-01')
        assert (transformer.delays, transformer.strategy, transformer.prediction_date) == (
            delays_dict_simple, 'shift', '2024-01-01')

    def test_init_with_dataframe(self, delays_dataframe):
        """A delays table and a ``datetime`` prediction date are stored unchanged."""
        transformer = PublicationDelayTransformer(delays=delays_dataframe, strategy='mask',
                                                  prediction_date=datetime(2024, 6, 15))
        assert transformer.delays is delays_dataframe and transformer.prediction_date == datetime(2024, 6, 15)

    def test_init_with_strategy_dict(self, delays_dict_simple):
        """A per-variable strategy dictionary is accepted at construction."""
        strategy = {'GDP': 'shift', 'inflation': 'mask', 'unemployment': 'shift'}
        assert PublicationDelayTransformer(delays=delays_dict_simple, strategy=strategy).strategy == strategy

    def test_init_default_values(self, delays_dict_simple):
        """A complete ``default_values`` dictionary is stored."""
        defaults = {'delay': 30.0, 'unit': 'D', 'reference_point': 'end', 'target_frequency': 'M'}
        transformer = PublicationDelayTransformer(delays=delays_dict_simple, strategy='mask', default_values=defaults)
        assert transformer.default_values == defaults

    def test_init_invalid_strategy_string(self, delays_dict_simple):
        """An unknown strategy name is rejected."""
        with pytest.raises(ValueError, match="strategy must be 'shift' or 'mask'"):
            PublicationDelayTransformer(delays=delays_dict_simple, strategy='invalid')

    def test_init_invalid_strategy_dict(self, delays_dict_simple):
        """An unknown strategy inside a dictionary is rejected and the variable named."""
        with pytest.raises(ValueError, match="for variable 'GDP'"):
            PublicationDelayTransformer(delays=delays_dict_simple, strategy={'GDP': 'invalid_strategy'})

    def test_init_invalid_strategy_type(self, delays_dict_simple):
        """A strategy that is neither a string nor a dictionary is rejected."""
        with pytest.raises(TypeError, match="'strategy' should be a string"):
            PublicationDelayTransformer(delays=delays_dict_simple, strategy=123)

    def test_init_invalid_reference_point(self, delays_dict_simple):
        """An unknown reference point is rejected."""
        with pytest.raises(ValueError, match="reference_point must be 'start' or 'end'"):
            PublicationDelayTransformer(delays=delays_dict_simple, reference_point='middle')

    def test_init_invalid_handle_missing(self, delays_dict_simple):
        """An unknown ``handle_missing_delays`` is rejected."""
        with pytest.raises(ValueError, match="'handle_missing_delays' must be"):
            PublicationDelayTransformer(delays=delays_dict_simple, handle_missing_delays='invalid')

    @pytest.mark.parametrize(
        ('strategy', 'defaults', 'missing'),
        [
            pytest.param('mask', {'delay': 30.0, 'unit': 'D'}, 'reference_point', id='mask-without-reference-point'),
            pytest.param('mask', {'delay': 30.0, 'unit': 'D', 'reference_point': 'end'}, 'target_frequency',
                         id='mask-without-target-frequency'),
            pytest.param('shift', {'delay': 30.0, 'reference_point': 'end'}, 'unit', id='shift-without-unit'),
        ],
    )
    def test_init_default_values_missing_keys(self, delays_dict_simple, strategy, defaults, missing):
        """``default_values`` must hold every key the strategy needs (``target_frequency`` for 'mask' only)."""
        with pytest.raises(ValueError, match=missing):
            PublicationDelayTransformer(delays=delays_dict_simple, strategy=strategy, default_values=defaults)

    def test_init_shift_default_values_need_no_target_frequency(self, delays_dict_simple):
        """For 'shift', ``default_values`` without ``target_frequency`` is complete."""
        defaults = {'delay': 30.0, 'unit': 'D', 'reference_point': 'end'}
        assert PublicationDelayTransformer(delays=delays_dict_simple, default_values=defaults).default_values == defaults

    def test_init_warning_strategy_dict_with_defaults(self, delays_dict_simple):
        """``default_values`` with a strategy dictionary is announced as ignored."""
        defaults = {'delay': 30.0, 'unit': 'D', 'reference_point': 'end', 'target_frequency': 'M'}
        with pytest.warns(UserWarning, match="'default_values' is ignored"):
            PublicationDelayTransformer(delays=delays_dict_simple, strategy={'GDP': 'shift', 'inflation': 'mask'},
                                        default_values=defaults)

    def test_init_warning_target_frequency_with_shift(self, delays_dict_simple):
        """``target_frequency`` with the 'shift' strategy is announced as ignored."""
        with pytest.warns(UserWarning, match="'target_frequency' is ignored"):
            PublicationDelayTransformer(delays=delays_dict_simple, strategy='shift', target_frequency='M')


# =============================================================================
# fit : attributs ajustés
# =============================================================================
class TestPublicationDelayTransformerFit:
    """``fit`` returns the transformer and sets the fitted attributes.

    Triage (prompt D4): the inherited tests passed a delays dictionary without
    ``delay_unit`` nor ``reference_point`` and failed with ``KeyError: 'GDP'``. No
    rename is involved (the signature is unchanged since ``d16b0b2``): nothing can
    infer the unit of a bare dictionary, so the tests were wrong (c) and now give
    both parameters. The raw ``KeyError`` itself is ``ANO-DELAYS-033``.
    """

    def test_fit_basic(self, sample_time_series, delays_dict_simple):
        """``fit`` returns ``self`` and resolves the prediction date."""
        transformer = PublicationDelayTransformer(delays=delays_dict_simple, delay_unit='D', reference_point='end',
                                                  prediction_date='2024-06-01')
        assert transformer.fit(sample_time_series) is transformer
        assert transformer.prediction_date_ == datetime(2024, 6, 1)

    def test_fit_sets_the_fitted_attributes(self, sample_time_series, delays_dict_simple):
        """Frequencies are detected per column, nothing is inferred from a dictionary."""
        transformer = _fit(sample_time_series, delays=delays_dict_simple, delay_unit='D', reference_point='end')
        assert transformer.detected_frequencies_ == {'GDP': 'M', 'inflation': 'M', 'unemployment': 'M'}
        assert transformer.inferred_params_ == {'delay_unit': {}, 'reference_point': {}, 'target_frequency': {}}

    def test_fit_creates_shift_params(self, sample_time_series, delays_dict_simple):
        """One shift per delayed column, in periods of its detected frequency.

        Gold values (prediction 2023-12-15, from the period end, last month published):
        45 days -> September (end October 1 + 45 days = November 15; October: December 16), -3;
        30 days -> October (November 1 + 30 days = December 1; November: December 31), -2;
        15 days -> October (November 16; November: December 16), -2.
        """
        transformer = _fit(sample_time_series, delays=delays_dict_simple, delay_unit='D', reference_point='end')
        assert transformer.shift_params == {
            'GDP': {'n_periods': -3, 'frequency': 'M'},
            'inflation': {'n_periods': -2, 'frequency': 'M'},
            'unemployment': {'n_periods': -2, 'frequency': 'M'},
        }

    def test_fit_shift_leaves_no_mask(self, sample_time_series, delays_dict_simple):
        """The 'shift' strategy builds no mask."""
        transformer = _fit(sample_time_series, delays=delays_dict_simple, delay_unit='D', reference_point='end')
        assert transformer.mask_params == {}

    def test_prediction_date_today_by_default(self):
        """Without ``prediction_date``, the date of the fit is used."""
        before = pd.Timestamp.now()
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D', reference_point='start')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            transformer.fit(_monthly(('GDP',)))
        assert before <= pd.Timestamp(transformer.prediction_date_) <= pd.Timestamp.now()

    def test_refit_replaces_the_parameters(self):
        """A second ``fit`` on other columns recomputes the parameters from scratch."""
        transformer = _fit(_monthly(), delays={'GDP': 45.0, 'CPI': 20.0}, delay_unit='D', reference_point='start')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            transformer.fit(_monthly(('CPI',)))
        assert transformer.shift_params == {'CPI': {'n_periods': -1, 'frequency': 'M'}}


# =============================================================================
# Valeurs d'or du décalage
# =============================================================================
class TestShiftGoldValues:
    """``n_periods`` is minus the number of months between December and the last month published on December 15."""

    @pytest.mark.parametrize(
        ('delay', 'reference_point', 'expected'),
        [
            # Valeur d'or : octobre est la dernière période publiée au 15 décembre (1er oct. + 45 j = 15 nov.,
            # 1er nov. + 45 j = 16 déc.) : -2
            pytest.param(45.0, 'start', -2, id='45d-start'),
            # Valeur d'or : septembre (fin le 1er oct. + 45 j = 15 nov. ; octobre : 1er nov. + 45 j = 16 déc.) : -3
            pytest.param(45.0, 'end', -3, id='45d-end'),
            # Valeur d'or : novembre (1er nov. + 20 j = 21 nov.) : -1
            pytest.param(20.0, 'start', -1, id='20d-start'),
            # Valeur d'or : octobre (fin le 1er nov. + 20 j = 21 nov. ; novembre : 1er déc. + 20 j = 21 déc.) : -2
            pytest.param(20.0, 'end', -2, id='20d-end'),
            # Valeur d'or : décembre publié le 1er décembre : 0
            pytest.param(0.0, 'start', 0, id='0d-start'),
            # Valeur d'or : décembre publié le 15 décembre, le jour même de la prédiction : 0
            pytest.param(14.0, 'start', 0, id='14d-start-published-on-the-day'),
            # Valeur d'or : décembre publié le 16, un jour trop tard : novembre, -1
            pytest.param(15.0, 'start', -1, id='15d-start-one-day-late'),
            # Valeur d'or : 1er nov. 2022 + 400 j = 5 déc. 2023 (publié) ; 1er déc. 2022 + 400 j = 4 janv. 2024 :
            # novembre 2022, treize mois avant décembre 2023 : -13
            pytest.param(400.0, 'start', -13, id='400d-start-longer-than-the-series'),
        ],
    )
    def test_monthly_series(self, delay, reference_point, expected):
        """Number of monthly periods to shift for a delay in days."""
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': delay}, delay_unit='D', reference_point=reference_point)
        assert transformer.shift_params['GDP']['n_periods'] == expected

    def test_quarterly_series(self):
        """On a quarterly series, the elapsed time counts from the quarter start.

        Gold value: the fourth quarter (October 1 + 45 days = November 15) is published on
        December 15: no shift.
        """
        quarterly = _monthly(('GDP',)).iloc[::3]
        transformer = _fit(quarterly, delays={'GDP': 45.0}, delay_unit='D', reference_point='start')
        assert transformer.shift_params['GDP'] == {'n_periods': 0, 'frequency': 'Q'}

    def test_frequency_is_the_detected_one(self):
        """The shift frequency of each column is its detected frequency."""
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 30.0}, delay_unit='D', reference_point='end')
        assert transformer.shift_params['GDP']['frequency'] == transformer.detected_frequencies_['GDP'] == 'M'

    def test_ordered_delays_give_ordered_shifts(self):
        """Increasing delays (10, 30, 60 days from the start) give shifts 0, -1, -2."""
        # Valeurs d'or : décembre publié le 11 déc. (0) ; novembre le 1er déc. (-1) ; octobre le 30 nov.,
        # novembre seulement le 31 déc. (-2)
        transformer = _fit(_monthly(('fast', 'medium', 'slow')), delays={'fast': 10.0, 'medium': 30.0, 'slow': 60.0},
                           delay_unit='D', reference_point='start')
        assert {col: p['n_periods'] for col, p in transformer.shift_params.items()} == {
            'fast': 0, 'medium': -1, 'slow': -2}


class TestCalendarBoundaries:
    """Near a calendar boundary, a value is visible at the prediction date if and only if it is published by then.

    ``n_periods`` used to count periods of 30 days (``ANO-DELAYS-042``, fixed): when
    the months between the delayed period and the prediction date did not last 30
    days each, the shift was one period off, in either direction.
    """

    def test_value_published_after_the_prediction_date_is_hidden(self):
        """February 2023 + 69 days = April 11: at April 10, the value of April must be January's, not February's.

        Gold value: 2023 is not a leap year, February + March = 59 days < 2 x 30;
        on the calendar, the last month published on April 10 is January
        (January 1 + 69 days = March 11), three months before April.
        """
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 69.0}, delay_unit='D', reference_point='start',
                           prediction_date='2023-04-10')
        shifted = transformer.transform(_monthly(('GDP',)))
        # Valeur d'or : valeur de janvier (0.0) ; février (1.0) n'est publié que le 11 avril
        assert shifted.loc[TS('2023-04-01'), 'GDP'] == 0.0

    def test_value_published_on_the_prediction_date_is_visible(self):
        """July 2023 + 76 days = September 15: at September 15, the value of September is July's.

        Gold value: July + August = 62 days > 2 x 30; on the calendar, July is
        published on the prediction date itself, two months before September.
        """
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 76.0}, delay_unit='D', reference_point='start',
                           prediction_date='2023-09-15')
        shifted = transformer.transform(_monthly(('GDP',)))
        # Valeur d'or : valeur de juillet (6.0), et non celle de juin (5.0)
        assert shifted.loc[TS('2023-09-01'), 'GDP'] == 6.0

    def test_business_days_skip_the_weekend(self):
        """Business-day series, 3 days from the start, prediction on Monday 2024-01-08: Friday's value lands on Monday.

        Gold value: Friday January 5 + 3 days = Monday January 8, published on the
        prediction date; Monday's own value is published on Thursday. One business
        day separates them (the weekend is not a period of a ``B`` index), hence -1.
        """
        X = pd.DataFrame({'GDP': np.arange(10.0)}, index=pd.bdate_range('2024-01-01', periods=10))
        transformer = _fit(X, delays={'GDP': 3.0}, delay_unit='D', reference_point='start',
                           prediction_date='2024-01-08')
        assert transformer.shift_params['GDP'] == {'n_periods': -1, 'frequency': 'B'}

    def test_negative_delay_on_business_days(self):
        """A delay of -1 day (forecast published the day before its date): Tuesday's value is visible on Monday.

        Gold value: Tuesday January 9 - 1 day = Monday January 8, published on the
        prediction date; Wednesday's value only on Tuesday: one business day ahead, +1.
        """
        X = pd.DataFrame({'GDP': np.arange(10.0)}, index=pd.bdate_range('2024-01-01', periods=10))
        transformer = _fit(X, delays={'GDP': -1.0}, delay_unit='D', reference_point='start',
                           prediction_date='2024-01-08')
        assert transformer.shift_params['GDP'] == {'n_periods': 1, 'frequency': 'B'}

    def test_end_of_a_quarter_counts_from_the_next_quarter(self):
        """Quarterly series, 45 days from the end, prediction on 2024-02-15: the fourth quarter of 2023 is published.

        Gold value: the fourth quarter ends on January 1, + 45 days = February 15
        (published on the prediction date), one quarter before the first quarter of
        2024, hence -1.
        """
        X = pd.DataFrame({'GDP': np.arange(8.0)}, index=pd.date_range('2022-01-01', periods=8, freq='QS'))
        transformer = _fit(X, delays={'GDP': 45.0}, delay_unit='D', reference_point='end',
                           prediction_date='2024-02-15')
        assert transformer.shift_params['GDP'] == {'n_periods': -1, 'frequency': 'Q'}


# =============================================================================
# Point de référence
# =============================================================================
class TestReferencePoint:
    """Origin of the delay: explicit, inferred from the delays table, or by default."""

    def test_inferred_from_the_delays_table(self):
        """The ``reference_point`` column of the table is used when nothing is given."""
        delays = _delays_frame(columns=('GDP',), delays=(45.0,), reference_point='end')
        transformer = _fit(_monthly(('GDP',)), delays=delays)
        # Valeur d'or : 45 j depuis la fin -> -3
        assert transformer.shift_params['GDP']['n_periods'] == -3

    def test_explicit_value_overrides_the_table(self):
        """An explicit reference point wins over the table."""
        delays = _delays_frame(columns=('GDP',), delays=(45.0,), reference_point='end')
        transformer = _fit(_monthly(('GDP',)), delays=delays, reference_point='start')
        # Valeur d'or : 45 j depuis le début -> -2
        assert transformer.shift_params['GDP']['n_periods'] == -2

    def test_default_value_is_imputed_with_a_warning(self):
        """Without explicit nor inferred value, ``default_values['reference_point']`` is imputed and announced."""
        delays = _delays_frame(columns=('GDP',), delays=(45.0,)).drop(columns='reference_point')
        defaults = {'delay': 1.0, 'unit': 'D', 'reference_point': 'end'}
        transformer = PublicationDelayTransformer(delays=delays, default_values=defaults, prediction_date=PREDICTION)
        with pytest.warns(UserWarning, match="Imputed default reference_point value 'end' for column 'GDP'"):
            transformer.fit(_monthly(('GDP',)))
        assert transformer.shift_params['GDP']['n_periods'] == -3

    def test_params_with_dict_reference_point(self):
        """A per-variable reference point applies to each variable.

        Triage (prompt D4): ``__init__`` rejected any non-string ``reference_point``
        although its signature and the factories use a dictionary (b, ANO-DELAYS-028,
        fixed). The former assertion (``a != b or |a - b| <= 1``) was always true; gold
        values instead: 45 days from the end -> -3, from the start -> -2.
        """
        transformer = _fit(_monthly(), delays={'GDP': 45.0, 'CPI': 45.0}, delay_unit='D',
                           reference_point={'GDP': 'end', 'CPI': 'start'})
        assert {col: p['n_periods'] for col, p in transformer.shift_params.items()} == {'GDP': -3, 'CPI': -2}

    def test_invalid_value_in_a_dict_is_rejected(self):
        """Each value of a per-variable reference point is validated, the variable named."""
        with pytest.raises(ValueError, match="for variable 'CPI' got 'middle'"):
            PublicationDelayTransformer(delays={'GDP': 45.0}, reference_point={'GDP': 'end', 'CPI': 'middle'})


# =============================================================================
# Unité des délais
# =============================================================================
class TestDelayUnit:
    """Unit of the delays: explicit (scalar or per variable) or inferred."""

    def test_days_and_hours_are_equivalent(self):
        """30 days and 720 hours give the same shift: November is published on December 1, -1."""
        days = _fit(_monthly(('GDP',)), delays={'GDP': 30.0}, delay_unit='D', reference_point='start')
        hours = _fit(_monthly(('GDP',)), delays={'GDP': 720.0}, delay_unit='h', reference_point='start')
        assert days.shift_params == hours.shift_params == {'GDP': {'n_periods': -1, 'frequency': 'M'}}

    @pytest.mark.parametrize(
        ('weeks', 'expected'),
        [
            # Valeur d'or : 6 semaines = 42 j, novembre publié le 13 déc. -> -1
            pytest.param(6.0, -1, id='6W'),
            # Valeur d'or : 7 semaines = 49 j, novembre publié le 20 déc., octobre le 19 nov. -> -2
            pytest.param(7.0, -2, id='7W'),
        ],
    )
    def test_weeks(self, weeks, expected):
        """A delay in weeks lasts seven days per week."""
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': weeks}, delay_unit='W', reference_point='start')
        assert transformer.shift_params['GDP']['n_periods'] == expected

    def test_unit_per_variable(self):
        """A per-variable unit applies to each variable: 30 days -> -1, 7 weeks -> -2."""
        transformer = _fit(_monthly(), delays={'GDP': 30.0, 'CPI': 7.0}, delay_unit={'GDP': 'D', 'CPI': 'W'},
                           reference_point='start')
        assert {col: p['n_periods'] for col, p in transformer.shift_params.items()} == {'GDP': -1, 'CPI': -2}

    def test_literal_unit_of_the_delays_table(self):
        """The literal unit returned by the delay detection (``'day'``) is understood."""
        delays = _delays_frame(columns=('GDP',), delays=(45.0,), unit='day')
        transformer = _fit(_monthly(('GDP',)), delays=delays)
        assert transformer.shift_params['GDP']['n_periods'] == -2


# =============================================================================
# Paramètres de masquage
# =============================================================================
class TestMaskParams:
    """Mask strategy: number of index observations hidden in each target period.

    ``n_obs`` is the number of index periods not yet published at the prediction
    date, counted on the calendar; masking is possible while
    ``n_obs`` is below the number of column periods in a target period, otherwise
    the column falls back to a shift.
    """

    @pytest.mark.parametrize(
        ('delay', 'reference_point', 'expected'),
        [
            # Valeur d'or : le 10 mars est publié le 15 ; 11 -> 15 mars non publiés -> 5
            pytest.param(5.0, 'start', 5, id='5d-start'),
            # Valeur d'or : depuis la fin du jour, le 9 mars est publié le 15 ; 10 -> 15 mars -> 6
            pytest.param(5.0, 'end', 6, id='5d-end'),
            # Valeur d'or : le 28 février (fin le 29) est publié le 15 mars ; 29 février -> 15 mars -> 16
            pytest.param(15.0, 'end', 16, id='15d-end'),
        ],
    )
    def test_daily_series_masked_by_month(self, delay, reference_point, expected):
        """On a daily series, one day is masked per day of delay not yet elapsed."""
        transformer = _fit(_daily(), delays={'GDP': delay}, strategy='mask', delay_unit='D',
                           reference_point=reference_point, target_frequency='M', prediction_date='2024-03-15')
        assert transformer.mask_params == {'GDP': {'n_obs': expected, 'mask_frequency': 'M', 'how': 'last'}}

    @pytest.mark.parametrize(
        'target_frequency',
        [pytest.param('Q', id='str'), pytest.param({'GDP': 'Q'}, id='dict'), pytest.param('quarterly', id='literal')],
    )
    def test_target_frequency_forms(self, target_frequency):
        """Scalar, per-variable and literal target frequencies give the normalized code.

        Gold value: monthly series, 20 days from the start: December (published December 21)
        is not published, November is -> 1 month masked per quarter.
        """
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 20.0}, strategy='mask', delay_unit='D',
                           reference_point='start', target_frequency=target_frequency)
        assert transformer.mask_params == {'GDP': {'n_obs': 1, 'mask_frequency': 'Q', 'how': 'last'}}

    def test_target_frequency_inferred_from_the_table(self):
        """The ``frequency`` column of the delays table is the target frequency."""
        transformer = _fit(_monthly(('GDP',)), delays=_delays_frame(columns=('GDP',), delays=(20.0,)), strategy='mask')
        assert transformer.mask_params['GDP']['mask_frequency'] == 'Q'

    @pytest.mark.parametrize(
        ('delay', 'masked'),
        [
            # Valeur d'or : 27 jours, le 16 février est publié le 15 mars ; 17 février -> 15 mars non publiés :
            # 12 + 15 = 27 jours < 28 jours de février 2023 -> masquage
            pytest.param(27.0, True, id='27d-masked'),
            # Valeur d'or : 28 jours, 16 février -> 15 mars non publiés : 13 + 15 = 28 jours, tout février
            # serait masqué -> repli sur le décalage (un mois de 30 jours le laissait masquer)
            pytest.param(28.0, False, id='28d-whole-february'),
        ],
    )
    def test_whole_february_cannot_be_masked(self, delay, masked):
        """Daily series masked by month: the check counts the 28 days of February 2023, not 30 days."""
        X = pd.DataFrame({'GDP': np.arange(90.0)}, index=pd.date_range('2023-01-01', periods=90, freq='D'))
        transformer = _fit(X, delays={'GDP': delay}, strategy='mask', delay_unit='D', reference_point='start',
                           target_frequency='M', prediction_date='2023-03-15')
        assert ('GDP' in transformer.mask_params, 'GDP' in transformer.shift_params) == (masked, not masked)

    def test_period_index_is_masked_like_its_month_starts(self):
        """A monthly ``PeriodIndex`` gets the mask of its month starts: 20 days, December unpublished, 1 month."""
        X = _monthly(('GDP',))
        X.index = X.index.to_period('M')
        transformer = _fit(X, delays={'GDP': 20.0}, strategy='mask', delay_unit='D', reference_point='start',
                           target_frequency='Q')
        assert transformer.mask_params == {'GDP': {'n_obs': 1, 'mask_frequency': 'Q', 'how': 'last'}}

    @pytest.mark.parametrize(
        ('delay', 'masked'),
        [
            # Valeur d'or : 3 jours non publiés au 20 février (18 -> 20) : moins qu'une quinzaine -> masquage
            pytest.param(3.0, True, id='3d-masked'),
            # Valeur d'or : 1er -> 20 février non publiés, 20 jours, plus qu'aucune quinzaine -> décalage
            pytest.param(20.0, False, id='20d-whole-semi-month'),
        ],
    )
    def test_semi_monthly_target(self, delay, masked):
        """Semi-monthly target periods (no pandas ``Period``): the check uses a semi-month of 15 days."""
        X = pd.DataFrame({'GDP': np.arange(60.0)}, index=pd.date_range('2024-01-01', periods=60, freq='D'))
        transformer = _fit(X, delays={'GDP': delay}, strategy='mask', delay_unit='D', reference_point='start',
                           target_frequency='SM', prediction_date='2024-02-20')
        assert ('GDP' in transformer.mask_params, 'GDP' in transformer.shift_params) == (masked, not masked)

    def test_last_maskable_number_of_months(self):
        """Up to two months of a quarter can be masked: 74 days, October published on December 14 -> 2."""
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 74.0}, strategy='mask', delay_unit='D',
                           reference_point='start', target_frequency='Q')
        assert transformer.mask_params['GDP']['n_obs'] == 2

    def test_fallback_to_shift_when_the_whole_period_would_be_masked(self):
        """80 days -> 3 unpublished months, a whole quarter: the column is shifted, with a warning.

        Gold value: on December 15, October (published December 20), November and
        December are unpublished; September is published on November 20.
        """
        transformer = PublicationDelayTransformer(delays={'GDP': 80.0}, strategy='mask', delay_unit='D',
                                                  reference_point='start', target_frequency='Q',
                                                  prediction_date=PREDICTION)
        with pytest.warns(UserWarning, match="Could not mask the column 'GDP'"):
            transformer.fit(_monthly(('GDP',)))
        assert (set(transformer.shift_params), transformer.mask_params) == ({'GDP'}, {})

    def test_fallback_shift_moves_values_to_later_dates(self):
        """The fallback shift is the one of the 'shift' strategy: 80 days from the start -> -3.

        The fallback used to store the (positive) number of observations to mask as
        ``n_periods``: the values moved three months **earlier**, before their
        publication (ANO-DELAYS-031, fixed).
        """
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 80.0}, strategy='mask', delay_unit='D',
                           reference_point='start', target_frequency='Q')
        assert transformer.shift_params['GDP']['n_periods'] == -3


# =============================================================================
# Stratégie
# =============================================================================
class TestStrategy:
    """Choice of the strategy per variable."""

    def test_mask_strategy_masks_every_delayed_column(self):
        """With 'mask', every column with a delay is masked, none shifted."""
        transformer = _fit(_monthly(), delays={'GDP': 20.0, 'CPI': 50.0}, strategy='mask', delay_unit='D',
                           reference_point='start', target_frequency='Q')
        # Valeurs d'or : 20 j, novembre publié le 21 nov. -> 1 ; 50 j, octobre publié le 20 nov.,
        # novembre seulement le 21 déc. -> 2
        assert ({c: p['n_obs'] for c, p in transformer.mask_params.items()}, transformer.shift_params) == (
            {'GDP': 1, 'CPI': 2}, {})

    def test_strategy_per_variable(self):
        """A per-variable strategy shifts some columns and masks the others."""
        transformer = _fit(_monthly(), delays={'GDP': 45.0, 'CPI': 20.0}, strategy={'GDP': 'shift', 'CPI': 'mask'},
                           delay_unit='D', reference_point='start', target_frequency={'CPI': 'Q'})
        assert (set(transformer.shift_params), set(transformer.mask_params)) == ({'GDP'}, {'CPI'})

    def test_strategy_per_variable_gold_values(self):
        """Shift of ``GDP`` (45 days -> -2) and mask of ``CPI`` (20 days -> 1 month per quarter)."""
        transformer = _fit(_monthly(), delays={'GDP': 45.0, 'CPI': 20.0}, strategy={'GDP': 'shift', 'CPI': 'mask'},
                           delay_unit='D', reference_point='start', target_frequency={'CPI': 'Q'})
        assert (transformer.shift_params, transformer.mask_params) == (
            {'GDP': {'n_periods': -2, 'frequency': 'M'}},
            {'CPI': {'n_obs': 1, 'mask_frequency': 'Q', 'how': 'last'}})

    def test_delayed_column_without_strategy_is_left_unchanged(self):
        """A column with a delay but absent from the strategy dict is announced and left unchanged."""
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0, 'CPI': 20.0}, strategy={'GDP': 'shift'},
                                                  delay_unit='D', reference_point='start', prediction_date=PREDICTION)
        with pytest.warns(UserWarning, match=r"\['CPI'\] have a delay but no strategy"):
            transformer.fit(_monthly())
        assert (set(transformer.shift_params), transformer.fit_report_.columns_unaffected) == ({'GDP'}, ('CPI',))

    def test_strategy_per_variable_ignores_default_values(self):
        """As announced by ``__init__``, ``default_values`` is ignored with a strategy dict: nothing is imputed.

        ``_build_parameter_dict`` used to impute the default reference point to the
        columns *outside* the strategy dict (``Z`` here), which are never delayed.
        """
        defaults = {'delay': 30.0, 'unit': 'D', 'reference_point': 'end', 'target_frequency': 'Q'}
        transformer = _fit(_monthly(('GDP', 'CPI', 'Z')), delays=_delays_frame(),
                           strategy={'GDP': 'shift', 'CPI': 'shift'}, default_values=defaults)
        assert transformer.fit_report_.defaults_imputed == ()

    def test_masked_variable_needs_a_target_frequency(self):
        """A masked variable without target frequency (defaults ignored with a dict) is rejected, named."""
        defaults = {'delay': 30.0, 'unit': 'D', 'reference_point': 'start', 'target_frequency': 'Q'}
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0, 'CPI': 20.0},
                                                  strategy={'GDP': 'shift', 'CPI': 'mask'}, delay_unit='D',
                                                  reference_point='start', default_values=defaults,
                                                  prediction_date=PREDICTION)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match=r"No 'target_frequency' for the delayed columns \['CPI'\]"):
                transformer.fit(_monthly())


# =============================================================================
# Formats du tableau des délais
# =============================================================================
class TestDelaysFormat:
    """Accepted forms of ``delays``, including the real output of the delay detection."""

    def test_dict_and_flat_table_are_equivalent(self):
        """The same delays as a dictionary or as a flat table give the same parameters."""
        from_dict = _fit(_monthly(), delays={'GDP': 45.0, 'CPI': 20.0}, delay_unit='D', reference_point='start')
        from_table = _fit(_monthly(), delays=_delays_frame(delays=(45.0, 20.0)))
        assert from_dict.shift_params == from_table.shift_params == {
            'GDP': {'n_periods': -2, 'frequency': 'M'}, 'CPI': {'n_periods': -1, 'frequency': 'M'}}

    def test_inferred_parameters_of_a_table(self):
        """Unit, reference point and target frequency are read per variable from the table."""
        transformer = _fit(_monthly(), delays=_delays_frame())
        assert transformer.inferred_params_ == {
            'delay_unit': {'GDP': 'D', 'CPI': 'D'},
            'reference_point': {'GDP': 'start', 'CPI': 'start'},
            'target_frequency': {'GDP': 'Q', 'CPI': 'Q'},
        }

    def test_table_without_frequency_column(self):
        """Without ``frequency`` column, no target frequency is inferred; the shift does not need one."""
        transformer = _fit(_monthly(('GDP',)), delays=_delays_frame(columns=('GDP',), delays=(45.0,)).drop(
            columns='frequency'))
        assert (transformer.inferred_params_['target_frequency'], transformer.shift_params['GDP']['n_periods']) == (
            {}, -2)

    @staticmethod
    def _applicable_delays() -> pd.DataFrame:
        """Detect and convert the delays of a monthly series downloaded on 2023-12-15.

        First download: the last observation of each column is its last date
        (``GDP`` stops in October, ``CPI`` in November), delays counted in days
        from the start of the month.
        """
        X = _monthly()
        X.loc['2023-11-01':, 'GDP'] = np.nan
        X.loc['2023-12-01':, 'CPI'] = np.nan
        detected = compare_and_detect_delays(X, None, PREDICTION, reference_point='start')
        return calculate_applicable_delay(detected, 'start', 'M', unit='D')

    def test_output_of_calculate_applicable_delay(self):
        """The applicable delays, index reset, give the shifts of the data they come from.

        Gold values: October 1 -> December 15 = 75 days, October published on the prediction
        date, -2; November 1 -> December 15 = 44 days, -1. The shift brings each last
        observation back onto December, the period of the prediction date (the 30-day
        convention gave -3 for GDP, ANO-DELAYS-042).
        """
        transformer = _fit(_monthly(), delays=self._applicable_delays().reset_index())
        assert {col: p['n_periods'] for col, p in transformer.shift_params.items()} == {'GDP': -2, 'CPI': -1}

    def test_raw_output_of_calculate_applicable_delay(self):
        """The variable in the index (level ``'column'``) is read as is (ANO-DELAYS-038, decided: accepted)."""
        raw = _fit(_monthly(), delays=self._applicable_delays())
        flat = _fit(_monthly(), delays=self._applicable_delays().reset_index())
        assert raw.shift_params == flat.shift_params

    def test_variable_in_the_last_unnamed_level(self):
        """Without ``column`` column nor level of that name, the variable is the last index level."""
        delays = _delays_frame().set_index('column').rename_axis(None)
        assert _fit(_monthly(), delays=delays).shift_params == _fit(_monthly(), delays=_delays_frame()).shift_params

    def test_per_entity_table_is_rejected(self):
        """A variable listed several times (per-entity delays) is rejected towards the factory."""
        delays = pd.concat({'FR': _delays_frame().set_index('column'), 'DE': _delays_frame().set_index('column')},
                           names=['country'])
        transformer = PublicationDelayTransformer(delays=delays, prediction_date=PREDICTION)
        with pytest.raises(ValueError, match='create_delay_transformer_factory'):
            transformer.fit(_monthly())

    def test_table_without_delay_column_is_rejected(self):
        """A table without ``delay`` column is rejected."""
        transformer = PublicationDelayTransformer(delays=_delays_frame().drop(columns='delay'),
                                                  prediction_date=PREDICTION)
        with pytest.raises(ValueError, match="'delay' column"):
            transformer.fit(_monthly())


# =============================================================================
# Délais manquants
# =============================================================================
class TestMissingDelays:
    """Columns without delay, delays without column, unresolved parameters."""

    def test_column_without_delay_is_left_out(self):
        """A column of ``X`` without delay is neither shifted nor masked and is reported as unaffected."""
        transformer = _fit(_monthly(('GDP', 'Z')), delays={'GDP': 45.0}, delay_unit='D', reference_point='start')
        assert (set(transformer.shift_params), transformer.fit_report_.columns_unaffected) == ({'GDP'}, ('Z',))

    def test_delay_of_an_absent_column_is_ignored(self):
        """A delay for a column absent from ``X`` is reported as ignored, without error."""
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 45.0, 'OLD': 5.0}, delay_unit='D',
                           reference_point='start')
        assert (set(transformer.shift_params), transformer.fit_report_.columns_ignored) == ({'GDP'}, ('OLD',))

    def test_unresolved_unit_raises_a_clear_error(self):
        """A delay whose unit cannot be resolved is reported by a ``ValueError`` naming the parameter."""
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0}, reference_point='start',
                                                  prediction_date=PREDICTION)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match='delay_unit'):
                transformer.fit(_monthly(('GDP',)))

    def test_column_without_delay_is_announced_by_default(self):
        """With ``handle_missing_delays='warn'`` (default), the columns without delay are named."""
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D', reference_point='start',
                                                  prediction_date=PREDICTION)
        with pytest.warns(UserWarning, match=r"No publication delay for the columns \['Z'\]"):
            transformer.fit(_monthly(('GDP', 'Z')))

    def test_handle_missing_delays_ignore(self):
        """With ``handle_missing_delays='ignore'``, nothing is announced."""
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D', reference_point='start',
                                                  handle_missing_delays='ignore', prediction_date=PREDICTION)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            transformer.fit(_monthly(('GDP', 'Z')))
        assert transformer.fit_report_.columns_unaffected == ('Z',)

    def test_handle_missing_delays_error(self):
        """``handle_missing_delays='error'`` rejects a column of ``X`` without delay."""
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D', reference_point='start',
                                                  handle_missing_delays='error', prediction_date=PREDICTION)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match='Z'):
                transformer.fit(_monthly(('GDP', 'Z')))

    def test_unknown_delay_is_treated_as_missing(self):
        """A ``NaN`` delay (couple whose delay is unknown) leaves its column unaffected."""
        transformer = _fit(_monthly(), delays={'GDP': 45.0, 'CPI': np.nan}, delay_unit='D', reference_point='start')
        assert (set(transformer.shift_params), transformer.fit_report_.columns_unaffected) == ({'GDP'}, ('CPI',))

    def test_unknown_delay_of_a_table_gets_the_default(self):
        """In a table, a ``NaN`` delay is missing: ``default_values['delay']`` replaces it (20 days -> -1)."""
        delays = _delays_frame(delays=(45.0, np.nan))
        defaults = {'delay': 20.0, 'unit': 'D', 'reference_point': 'start'}
        transformer = _fit(_monthly(), delays=delays, default_values=defaults)
        assert transformer.shift_params['CPI']['n_periods'] == -1


# =============================================================================
# Valeurs par défaut
# =============================================================================
class TestDefaultValues:
    """``default_values`` completes the parameters of every column of ``X``."""

    def test_default_delay_for_a_column_without_delay(self):
        """A column without delay gets ``default_values['delay']``: 20 days from the start -> -1."""
        defaults = {'delay': 20.0, 'unit': 'D', 'reference_point': 'start'}
        transformer = _fit(_monthly(), delays={'GDP': 45.0}, delay_unit='D', reference_point='start',
                           default_values=defaults)
        assert transformer.shift_params['CPI'] == {'n_periods': -1, 'frequency': 'M'}

    def test_default_delay_is_reported(self):
        """The default delay is listed among the defaults imputed, with its origin."""
        defaults = {'delay': 20.0, 'unit': 'D', 'reference_point': 'start'}
        report = _fit(_monthly(), delays={'GDP': 45.0}, delay_unit='D', reference_point='start',
                      default_values=defaults).fit_report_
        records = {record.column: record for record in report.columns}
        assert (report.defaults_imputed, records['GDP'].delay_source, records['CPI'].delay_source) == (
            (('CPI', 'delay'),), 'explicit', 'default')

    def test_default_unit(self):
        """Without explicit nor inferred unit, ``default_values['unit']`` is used: 45 days from the start -> -2."""
        defaults = {'delay': 20.0, 'unit': 'D', 'reference_point': 'start'}
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 45.0}, default_values=defaults)
        assert transformer.shift_params['GDP']['n_periods'] == -2

    def test_default_target_frequency_for_the_mask(self):
        """With 'mask', ``default_values['target_frequency']`` is imputed and announced."""
        defaults = {'delay': 20.0, 'unit': 'D', 'reference_point': 'start', 'target_frequency': 'Q'}
        transformer = PublicationDelayTransformer(delays={'GDP': 20.0}, strategy='mask', delay_unit='D',
                                                  reference_point='start', default_values=defaults,
                                                  prediction_date=PREDICTION)
        with pytest.warns(UserWarning, match="Imputed default target_frequency value 'Q' for column 'GDP'"):
            transformer.fit(_monthly(('GDP',)))
        assert transformer.mask_params == {'GDP': {'n_obs': 1, 'mask_frequency': 'Q', 'how': 'last'}}


# =============================================================================
# Avertissements
# =============================================================================
class TestWarnings:
    """``fit`` warns only about what it could not resolve."""

    def test_fully_resolved_fit_emits_no_warning(self):
        """Explicit unit and reference point for every column: nothing to report."""
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D', reference_point='start',
                                                  prediction_date=PREDICTION)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            transformer.fit(_monthly(('GDP',)))

    def test_parameters_of_undelayed_columns_are_not_required(self):
        """A column without delay needs no unit: only its missing delay is announced."""
        transformer = PublicationDelayTransformer(delays=_delays_frame(columns=('GDP',), delays=(45.0,)).drop(columns='unit'),
                                                  delay_unit={'GDP': 'D'}, prediction_date=PREDICTION)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            transformer.fit(_monthly(('GDP', 'Z')))
        assert [str(w.message) for w in caught] == ["No publication delay for the columns ['Z']: they are left unchanged"]


# =============================================================================
# Panel et types d'entrée
# =============================================================================
class TestInputTypes:
    """Inputs other than a ``DataFrame`` indexed by dates."""

    @staticmethod
    def _panel(**frames) -> pd.DataFrame:
        """Panel (country, date) of the given per-entity frames (``FR`` and ``DE`` monthly by default)."""
        frames = frames or {'FR': _monthly(), 'DE': _monthly() + 100}
        return pd.concat(frames, names=['country', 'date'])

    def test_panel_gets_the_shift_of_each_column(self):
        """A panel (entity, date) gets the same per-column shifts as each of its entities."""
        transformer = _fit(self._panel(), delays={'GDP': 45.0, 'CPI': 20.0}, delay_unit='D', reference_point='start')
        assert {col: p['n_periods'] for col, p in transformer.shift_params.items()} == {'GDP': -2, 'CPI': -1}

    def test_panel_warns_that_every_entity_gets_the_same_delays(self):
        """Fitting a panel directly is announced, pointing to the factory for per-entity delays."""
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0, 'CPI': 20.0}, delay_unit='D',
                                                  reference_point='start', prediction_date=PREDICTION)
        with pytest.warns(UserWarning, match='same publication delays are applied to every entity.*'
                                             'create_delay_transformer_factory'):
            transformer.fit(self._panel())

    def test_panel_transform_and_round_trip(self):
        """Each entity is shifted (last ``GDP`` value of December in February 2024), then restored exactly."""
        panel = self._panel()
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0, 'CPI': 20.0}, delay_unit='D',
                                                  reference_point='start', prediction_date=PREDICTION)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            shifted = transformer.fit_transform(panel)
        last = {entity: shifted.loc[entity, 'GDP'].last_valid_index() for entity in ('FR', 'DE')}
        assert last == {'FR': TS('2024-02-01'), 'DE': TS('2024-02-01')}
        pd.testing.assert_frame_equal(transformer.inverse_transform(shifted), panel)

    def test_panel_mask(self):
        """A panel can be masked: one month per quarter in every entity (20 days, target quarter)."""
        transformer = _fit(self._panel(), delays={'GDP': 20.0}, strategy='mask', delay_unit='D',
                           reference_point='start', target_frequency='Q')
        masked = transformer.transform(self._panel())
        assert masked['GDP'].isna().groupby(level='country').sum().to_dict() == {'DE': 4, 'FR': 4}

    def test_panel_column_with_different_frequencies_is_rejected(self):
        """A delayed column monthly for one entity and quarterly for another needs per-entity delays."""
        quarterly = _monthly()
        quarterly.loc[~quarterly.index.month.isin([1, 4, 7, 10]), 'GDP'] = np.nan
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D', reference_point='start',
                                                  prediction_date=PREDICTION)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match=r"\['GDP'\] have different frequencies across entities"):
                transformer.fit(self._panel(FR=_monthly(), DE=quarterly))

    def test_panel_column_empty_for_one_entity_takes_the_others_frequency(self):
        """A column entirely empty for one entity gets the frequency the other entities share."""
        empty_cpi = _monthly().assign(CPI=np.nan)
        transformer = _fit(self._panel(FR=_monthly(), DE=empty_cpi), delays={'GDP': 45.0, 'CPI': 20.0},
                           delay_unit='D', reference_point='start')
        assert transformer.shift_params['CPI'] == {'n_periods': -1, 'frequency': 'M'}

    def test_panel_mask_with_different_index_frequencies_is_rejected(self):
        """Masking needs one index frequency: a monthly and a quarterly index need per-entity delays.

        ``GDP`` is quarterly in both entities, stored on a monthly index for ``FR`` and on a
        quarterly one for ``DE``.
        """
        on_monthly_index = _monthly(('GDP',))
        on_monthly_index.loc[~on_monthly_index.index.month.isin([1, 4, 7, 10]), 'GDP'] = np.nan
        on_quarterly_index = _monthly(('GDP',)).iloc[::3]
        transformer = PublicationDelayTransformer(delays={'GDP': 20.0}, strategy='mask', delay_unit='D',
                                                  reference_point='start', target_frequency='Y',
                                                  prediction_date=PREDICTION)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match='different index frequencies'):
                transformer.fit(self._panel(FR=on_monthly_index, DE=on_quarterly_index))

    def test_series_is_accepted(self):
        """A named ``Series`` is handled like a one-column frame."""
        transformer = _fit(_monthly(('GDP',))['GDP'], delays={'GDP': 45.0}, delay_unit='D', reference_point='start')
        assert transformer.shift_params == {'GDP': {'n_periods': -2, 'frequency': 'M'}}

    def test_series_in_series_out(self):
        """``transform`` and ``inverse_transform`` of a Series return a Series with its name."""
        series = _monthly(('GDP',))['GDP']
        transformer = _fit(series, delays={'GDP': 45.0}, delay_unit='D', reference_point='start')
        shifted = transformer.transform(series)
        assert isinstance(shifted, pd.Series) and shifted.name == 'GDP'
        pd.testing.assert_series_equal(transformer.inverse_transform(shifted), series, check_freq=False)

    def test_other_input_types_are_rejected(self):
        """A numpy array is rejected with a ``TypeError``."""
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D', reference_point='start')
        with pytest.raises(TypeError, match='pandas Series or DataFrame'):
            transformer.fit(np.arange(12.0))


# =============================================================================
# Protocole sklearn
# =============================================================================
class TestSklearnProtocol:
    """``get_params`` / ``set_params``, ``clone`` and the not-fitted errors."""

    def test_get_params_returns_the_constructor_arguments(self):
        """Every constructor argument is a parameter."""
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D')
        assert set(transformer.get_params()) == {
            'delays', 'prediction_date', 'strategy', 'target_frequency', 'delay_unit', 'reference_point',
            'handle_missing_delays', 'default_values'}

    def test_clone_keeps_the_parameters_and_drops_the_fit(self):
        """A clone has the same parameters and no fitted attribute."""
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 45.0}, delay_unit='D', reference_point='start')
        cloned = clone(transformer)
        assert cloned.get_params() == transformer.get_params() and not hasattr(cloned, 'shift_params')

    def test_set_params_changes_the_fit(self):
        """``set_params`` then ``fit`` uses the new value (45 days from the end -> -3)."""
        transformer = PublicationDelayTransformer(delays={'GDP': 45.0}, delay_unit='D', reference_point='start',
                                                  prediction_date=PREDICTION)
        transformer.set_params(reference_point='end')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            transformer.fit(_monthly(('GDP',)))
        assert transformer.shift_params['GDP']['n_periods'] == -3

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_not_fitted(self, method, delays_dict_simple, sample_time_series):
        """``transform`` and ``inverse_transform`` require ``fit``."""
        transformer = PublicationDelayTransformer(delays=delays_dict_simple, strategy='shift')
        with pytest.raises(NotFittedError):
            getattr(transformer, method)(sample_time_series)

    def test_inverse_transform_before_transform(self):
        """``inverse_transform`` reverses the last ``transform``: without one, ``NotFittedError`` (ANO-DELAYS-037)."""
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 45.0}, delay_unit='D', reference_point='start')
        with pytest.raises(NotFittedError, match='call transform before inverse_transform'):
            transformer.inverse_transform(_monthly(('GDP',)))

    def test_refit_forgets_the_previous_transform(self):
        """After a new ``fit``, the helpers of the previous ``transform`` are no longer used."""
        transformer = _fit(_monthly(('GDP',)), delays={'GDP': 45.0}, delay_unit='D', reference_point='start')
        shifted = transformer.transform(_monthly(('GDP',)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            transformer.fit(_monthly(('GDP',)))
        with pytest.raises(NotFittedError):
            transformer.inverse_transform(shifted)


# =============================================================================
# Rapport d'ajustement (fit_report_)
# =============================================================================
class TestFitReport:
    """``fit_report_`` gathers what ``fit`` resolved, beyond the warnings it emits."""

    @staticmethod
    def _monthly(columns=('GDP', 'CPI'), periods=12):
        index = pd.date_range('2023-01-01', periods=periods, freq='MS')
        return pd.DataFrame({col: range(periods) for col in columns}, index=index)

    @staticmethod
    def _delays(columns=('GDP', 'CPI'), delays=(45.0, 20.0), **extra):
        data = {'column': list(columns), 'delay': list(delays), 'unit': ['D'] * len(columns),
                'reference_point': ['start'] * len(columns), 'frequency': ['Q'] * len(columns)}
        data.update(extra)
        return pd.DataFrame(data)

    def test_report_exists_only_after_fit(self):
        transformer = PublicationDelayTransformer(delays=self._delays(), prediction_date='2023-12-15')
        assert not hasattr(transformer, 'fit_report_')
        transformer.fit(self._monthly())
        assert transformer.fit_report_.prediction_date == datetime(2023, 12, 15)

    def test_report_matches_the_fitted_parameters(self):
        transformer = PublicationDelayTransformer(delays=self._delays(), prediction_date='2023-12-15')
        report = transformer.fit(self._monthly()).fit_report_
        records = {record.column: record for record in report.columns}
        assert set(records) == set(transformer.shift_params)
        for col, params in transformer.shift_params.items():
            assert records[col].strategy == 'shift'
            assert records[col].n_periods == params['n_periods']
            assert records[col].frequency == params['frequency']
        assert (records['GDP'].delay, records['GDP'].delay_unit, records['GDP'].reference_point) == (45.0, 'D', 'start')

    def test_columns_unaffected_and_ignored(self):
        X = self._monthly(columns=('GDP', 'Z'))
        delays = self._delays(columns=('GDP', 'OLD'), delays=(45.0, 5.0))
        with pytest.warns(UserWarning):
            report = PublicationDelayTransformer(delays=delays, prediction_date='2023-12-15').fit(X).fit_report_
        assert report.columns_unaffected == ('Z',)
        assert report.columns_ignored == ('OLD',)

    def test_parameter_sources_inferred_and_explicit(self):
        X = self._monthly()
        inferred = PublicationDelayTransformer(
            delays=self._delays(), prediction_date='2023-12-15').fit(X).fit_report_
        assert {r.reference_point_source for r in inferred.columns} == {'inferred'}
        assert {r.delay_unit_source for r in inferred.columns} == {'inferred'}
        explicit = PublicationDelayTransformer(
            delays=self._delays(), prediction_date='2023-12-15', reference_point='end').fit(X).fit_report_
        assert {r.reference_point_source for r in explicit.columns} == {'explicit'}
        assert {r.reference_point for r in explicit.columns} == {'end'}

    def test_explicit_dict_wins_for_its_columns_only(self):
        X = self._monthly()
        report = PublicationDelayTransformer(
            delays=self._delays(), prediction_date='2023-12-15', delay_unit={'GDP': 'W'}
        ).fit(X).fit_report_
        sources = {r.column: (r.delay_unit, r.delay_unit_source) for r in report.columns}
        assert sources == {'GDP': ('W', 'explicit'), 'CPI': ('D', 'inferred')}

    def test_defaults_imputed_are_listed(self):
        X = self._monthly()
        delays = self._delays().drop(columns='reference_point')
        defaults = {'delay': 1.0, 'unit': 'D', 'reference_point': 'end'}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            report = PublicationDelayTransformer(
                delays=delays, prediction_date='2023-12-15', default_values=defaults).fit(X).fit_report_
        assert report.defaults_imputed == (('GDP', 'reference_point'), ('CPI', 'reference_point'))
        assert {r.reference_point_source for r in report.columns} == {'default'}

    def test_mask_strategy_records_the_masked_observations(self):
        X = self._monthly()
        delays = self._delays(delays=(20.0, 20.0))
        report = PublicationDelayTransformer(
            delays=delays, strategy='mask', prediction_date='2023-12-15').fit(X).fit_report_
        for record in report.columns:
            assert record.strategy == 'mask'
            assert record.n_obs == 1 and record.n_periods is None
            assert (record.target_frequency, record.target_frequency_source) == ('Q', 'inferred')
            assert record.moved_from_mask is False
        assert report.mask_fallbacks == ()

    def test_mask_fallback_is_reported(self):
        X = self._monthly()
        delays = self._delays(delays=(400.0, 20.0))
        with pytest.warns(UserWarning, match="Could not mask the column 'GDP'"):
            transformer = PublicationDelayTransformer(delays=delays, strategy='mask', prediction_date='2023-12-15')
            report = transformer.fit(X).fit_report_
        assert report.mask_fallbacks == ('GDP',)
        records = {record.column: record for record in report.columns}
        assert (records['GDP'].strategy, records['GDP'].moved_from_mask) == ('shift', True)
        assert records['GDP'].n_periods == transformer.shift_params['GDP']['n_periods']
        assert records['CPI'].strategy == 'mask'

    def test_refit_replaces_the_report(self):
        transformer = PublicationDelayTransformer(delays=self._delays(), prediction_date='2023-12-15')
        first = transformer.fit(self._monthly()).fit_report_
        second = transformer.fit(self._monthly(columns=('GDP',))).fit_report_
        assert first is not second
        assert [r.column for r in second.columns] == ['GDP']

    def test_fit_is_logged_at_info_level(self, caplog):
        transformer = PublicationDelayTransformer(delays=self._delays(), prediction_date='2023-12-15')
        with caplog.at_level('INFO', logger='tsforecast.delays.transformers'):
            report = transformer.fit(self._monthly()).fit_report_
        assert report.summary() in caplog.messages

    def test_clone_does_not_copy_the_report(self):
        transformer = PublicationDelayTransformer(delays=self._delays(), prediction_date='2023-12-15')
        transformer.fit(self._monthly())
        assert not hasattr(clone(transformer), 'fit_report_')


# =============================================================================
# Masques nuls ou négatifs (ajout D3 : contrat de MaskTransformer, n_obs >= 0)
# =============================================================================
class TestZeroAndNegativeMasks:
    """A mask with nothing to mask leaves its column unchanged instead of building a ``MaskTransformer``.

    Gold values (monthly column, delays at monthly frequency, so the mask frequency
    equals the index frequency): ``n_obs`` is the number of months not yet published
    on 2023-12-15, the delay counted from the period start; a negative count (later
    months already published) is clamped at zero.
    """

    @staticmethod
    def _monthly() -> pd.DataFrame:
        index = pd.date_range('2023-01-01', periods=12, freq='MS')
        return pd.DataFrame({'GDP': np.arange(12.0), 'CPI': np.arange(12.0) * 2}, index=index)

    @staticmethod
    def _transformer(delay: float) -> PublicationDelayTransformer:
        delays = pd.DataFrame({'column': ['GDP'], 'delay': [delay], 'unit': ['D'],
                               'reference_point': ['start'], 'frequency': ['M']})
        return PublicationDelayTransformer(delays=delays, strategy='mask', prediction_date='2023-12-15')

    @pytest.mark.parametrize(
        'delay',
        [
            # Valeur d'or : décembre publié le 6 décembre -> 0
            pytest.param(5.0, id='zero'),
            # Valeur d'or : janvier publié dès le 22 novembre (1er janv. - 40 j) -> -1, ramené à 0
            pytest.param(-40.0, id='negative'),
        ],
    )
    def test_n_obs_is_never_negative(self, delay):
        """The number of observations to mask is clamped at zero."""
        transformer = self._transformer(delay).fit(self._monthly())
        assert transformer.mask_params['GDP']['n_obs'] == 0

    @pytest.mark.parametrize('delay', [pytest.param(5.0, id='zero'), pytest.param(-40.0, id='negative')])
    def test_zero_mask_leaves_the_data_unchanged(self, delay):
        """``transform`` returns the data as given: nothing is unpublished at the prediction date."""
        transformer = self._transformer(delay).fit(self._monthly())
        pd.testing.assert_frame_equal(transformer.transform(self._monthly()), self._monthly())

    def test_zero_mask_round_trip(self):
        """``inverse_transform`` of an unchanged frame returns it unchanged."""
        transformer = self._transformer(5.0).fit(self._monthly())
        recovered = transformer.inverse_transform(transformer.transform(self._monthly()))
        pd.testing.assert_frame_equal(recovered, self._monthly())
