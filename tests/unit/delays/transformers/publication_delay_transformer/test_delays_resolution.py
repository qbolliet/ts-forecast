"""Resolution of delay parameters by ``PublicationDelayTransformer``.

Scenarios: reference point, delay unit, accepted forms of ``delays``, missing delays and
``default_values``.
"""

# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest

# Classe à tester et producteurs réels de tableaux de délais
from tsforecast.delays.calculator import calculate_applicable_delay
from tsforecast.delays.data_manager import compare_and_detect_delays
from tsforecast.delays.transformers import PublicationDelayTransformer

# Outils de test partagés du paquet
from tests.unit.delays.transformers.publication_delay_transformer.helpers import (
    PREDICTION,
    _delays_frame,
    _fit,
    _monthly,
)


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

        ``reference_point`` accepts a dictionary giving one reference per variable, as
        its signature and the factories do. Gold values: 45 days from the end -> -3,
        from the start -> -2.
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
        observation back onto December, the period of the prediction date (counting
        30-day months would give -3 for GDP).
        """
        transformer = _fit(_monthly(), delays=self._applicable_delays().reset_index())
        assert {col: p['n_periods'] for col, p in transformer.shift_params.items()} == {'GDP': -2, 'CPI': -1}

    def test_raw_output_of_calculate_applicable_delay(self):
        """The variable in the index (level ``'column'``) is read as is."""
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
