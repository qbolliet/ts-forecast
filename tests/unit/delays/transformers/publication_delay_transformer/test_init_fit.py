"""Construction and ``fit`` of ``PublicationDelayTransformer``.

Scenarios: constructor validation, fitted attributes, fit warnings, accepted input
types, sklearn protocol and the ``fit_report_`` gathered by ``fit``.
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
from tsforecast.delays.transformers import PublicationDelayTransformer

# Outils de test partagés du paquet
from tests.unit.delays.transformers.publication_delay_transformer.helpers import (
    PREDICTION,
    TS,
    _delays_frame,
    _fit,
    _monthly,
)


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

    A bare delays dictionary carries neither ``delay_unit`` nor ``reference_point``:
    nothing can infer the unit of such a dictionary, so both parameters are given
    explicitly. Without them, ``fit`` raises ``KeyError: 'GDP'``.
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
        """``inverse_transform`` reverses the last ``transform``: without one, ``NotFittedError``."""
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
