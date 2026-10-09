"""Unit tests for ``PublicationDelayTransformer.transform`` / ``inverse_transform``.

Covers the application of the fitted parameters (``transform``,
``inverse_transform``, through the private helpers
``_apply_auxiliary_transformers``, ``_apply_inverse_transformers`` and
``_active_params``): gold frames for the 'shift' and 'mask' strategies, columns
left untouched, column order, round trip, and the robustness cases of
``CLAUDE.md`` (unsorted rows, special column names, end-of-period labels,
``PeriodIndex``, duplicated dates, too few observations). Split from
``test_publication_delay_transformer.py`` (mirror rule, theme of the
``tests/unit/delays/transformers/`` package) by prompt D4.

Gold values: monthly series of 2023 (``GDP = 0, 1, ..., 11``, ``CPI = 0, 2, ...,
22``), prediction date 2023-12-15. 45 days from the period start shift ``GDP``
by two months, 20 days shift ``CPI`` by one month (see the module docstring of
``test_publication_delay_transformer.py``). ``ShiftTransformer`` moves the dates
without losing any value, so the output index is the union of the shifted
indexes of the columns: it grows (and the dates no column occupies any more
disappear); ``inverse_transform`` drops the rows ``transform`` added, so that a
round trip restores the input exactly.
"""
# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest

# Classe à tester
from tsforecast.delays.transformers import PublicationDelayTransformer

# Perturbations partagées
from tests.support import perturbations as perturb

TS = pd.Timestamp
PREDICTION = '2023-12-15'


# =============================================================================
# Constructeurs locaux
# =============================================================================
def _monthly(columns=('GDP', 'CPI')) -> pd.DataFrame:
    """Build the gold monthly frame of 2023: column ``i`` holds ``(i + 1) * [0, ..., 11]``.

    Args:
        columns: Column names.

    Returns:
        Deterministic float frame indexed by month starts.
    """
    index = pd.date_range('2023-01-01', periods=12, freq='MS')
    return pd.DataFrame({col: np.arange(12, dtype=float) * (i + 1) for i, col in enumerate(columns)}, index=index)


def _transformer(delays=None, **kwargs) -> PublicationDelayTransformer:
    """Build a transformer at the gold prediction date, delays in days from the period start.

    Args:
        delays: Delays dictionary (``GDP`` 45 days, ``CPI`` 20 days by default).
        **kwargs: Other constructor arguments.

    Returns:
        Unfitted transformer.
    """
    params = dict(delays=delays or {'GDP': 45.0, 'CPI': 20.0}, delay_unit='D', reference_point='start',
                  prediction_date=PREDICTION)
    params.update(kwargs)
    return PublicationDelayTransformer(**params)


def _fit_transform(transformer: PublicationDelayTransformer, X: pd.DataFrame) -> pd.DataFrame:
    """Fit and transform with the warnings of ``fit`` silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return transformer.fit_transform(X)


def _shifted_gold() -> pd.DataFrame:
    """Gold output of the default shift: ``GDP`` two months later, ``CPI`` one month later.

    Returns:
        Frame indexed from February 2023 (first date of ``CPI``) to February 2024 (last date of ``GDP``).
    """
    gdp = pd.Series(np.arange(12, dtype=float), index=pd.date_range('2023-03-01', periods=12, freq='MS'), name='GDP')
    cpi = pd.Series(np.arange(12, dtype=float) * 2, index=pd.date_range('2023-02-01', periods=12, freq='MS'),
                    name='CPI')
    return pd.concat([gdp, cpi], axis=1)


# =============================================================================
# Valeurs d'or de transform
# =============================================================================
class TestTransformGoldValues:
    """Output of ``transform`` cell by cell."""

    def test_transform_basic(self):
        """Shifted columns keep their values and names; the index becomes the union of the shifted dates.

        Triage (prompt D4): the inherited test expected the shape of the input. The
        shift moves the dates without loss of data (deliberate, ``ShiftTransformer``
        contract of prompt D3): 12 + 1 rows here (February 2023 to February 2024,
        January 2023 holding no value any more), so the assertion was wrong (c).
        """
        result = _fit_transform(_transformer(), _monthly())
        pd.testing.assert_frame_equal(result, _shifted_gold(), check_freq=False)

    def test_fit_transform(self):
        """``fit_transform`` equals ``fit`` then ``transform``."""
        transformer = _transformer()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected = transformer.fit(_monthly()).transform(_monthly())
        pd.testing.assert_frame_equal(_fit_transform(_transformer(), _monthly()), expected)

    def test_last_value_lands_on_the_period_of_the_prediction_date(self):
        """With 45 days from the start, October is the last period published on December 15: it shows in December."""
        result = _fit_transform(_transformer({'GDP': 45.0}), _monthly(('GDP',)))
        # Valeur d'or : octobre 2023 vaut 9
        assert result.loc['2023-12-01', 'GDP'] == 9.0

    def test_mask_gold(self):
        """'mask', target quarter: per quarter, the months not yet published on December 15 are hidden.

        Gold values: ``GDP`` 45 days -> 2 months (November published on December 16, October on
        November 15: February-March, May-June, ...), ``CPI`` 20 days -> 1 month (December published on
        December 21: March, June, September, December).
        """
        transformer = _transformer(strategy='mask', target_frequency='Q')
        expected = _monthly()
        expected.loc[pd.to_datetime(['2023-03-01', '2023-06-01', '2023-09-01', '2023-12-01'])] = np.nan
        expected.loc[pd.to_datetime(['2023-02-01', '2023-05-01', '2023-08-01', '2023-11-01']), 'GDP'] = np.nan
        pd.testing.assert_frame_equal(_fit_transform(transformer, _monthly()), expected)

    def test_daily_mask_gold(self):
        """Daily series, 5 days from the day end, target month: the last 6 calendar days of each month are hidden."""
        X = pd.DataFrame({'GDP': np.arange(90.0)}, index=pd.date_range('2024-01-01', periods=90, freq='D'))
        transformer = _transformer({'GDP': 5.0}, strategy='mask', reference_point='end', target_frequency='M',
                                   prediction_date='2024-03-15')
        result = _fit_transform(transformer, X)
        # Valeur d'or : 26-31 janvier, 24-29 février, 26-31 mars (la série s'arrête le 30 mars)
        expected = (list(pd.date_range('2024-01-26', '2024-01-31')) + list(pd.date_range('2024-02-24', '2024-02-29'))
                    + list(pd.date_range('2024-03-26', '2024-03-30')))
        assert list(result.index[result['GDP'].isna()]) == expected

    def test_column_without_delay_is_untouched_and_order_kept(self):
        """A column without delay keeps its values; the columns keep the input order."""
        X = _monthly(('Z', 'GDP', 'CPI'))
        result = _fit_transform(_transformer(), X)
        assert list(result.columns) == ['Z', 'GDP', 'CPI']
        pd.testing.assert_series_equal(result['Z'].reindex(X.index), X['Z'])

    def test_columns_with_the_same_parameters_share_one_helper(self):
        """Columns with identical parameters go through one ``ShiftTransformer``, the others through their own."""
        X = _monthly(('GDP', 'CPI', 'PMI'))
        transformer = _transformer({'GDP': 45.0, 'CPI': 20.0, 'PMI': 45.0})
        result = _fit_transform(transformer, X)
        recovered = transformer.inverse_transform(result)
        # Valeur d'or : deux jeux de paramètres (-2 mois pour GDP et PMI, -1 mois pour CPI)
        assert (len(transformer.auxiliary_transformers_['shift']), result['PMI'].last_valid_index()) == (
            2, TS('2024-02-01'))
        pd.testing.assert_frame_equal(recovered.reindex(X.index), X, check_freq=False)

    def test_no_delayed_column_returns_the_input(self):
        """Without any delayed column, ``transform`` returns the data unchanged."""
        X = _monthly(('Z',))
        pd.testing.assert_frame_equal(_fit_transform(_transformer(), X), X)


# =============================================================================
# Aller-retour transform / inverse_transform
# =============================================================================
class TestRoundTrip:
    """``inverse_transform(transform(X))`` restores ``X`` exactly.

    The rows added by ``transform`` (dates outside its input), empty once inverted,
    are dropped by ``inverse_transform`` (decision of the author, prompt D4).
    """

    @staticmethod
    def _assert_restored(recovered: pd.DataFrame, X: pd.DataFrame) -> None:
        """Check that ``X`` is restored exactly, index included."""
        pd.testing.assert_frame_equal(recovered, X, check_freq=False)

    def test_shift_round_trip(self):
        """Shift then inverse restores every value."""
        transformer = _transformer()
        recovered = transformer.inverse_transform(_fit_transform(transformer, _monthly()))
        self._assert_restored(recovered, _monthly())

    def test_mask_round_trip_is_exact(self):
        """Mask then inverse restores the masked cells: identity."""
        transformer = _transformer(strategy='mask', target_frequency='Q')
        recovered = transformer.inverse_transform(_fit_transform(transformer, _monthly()))
        pd.testing.assert_frame_equal(recovered, _monthly())

    def test_shift_and_mask_round_trip(self):
        """A masked column and a column moved to the shift (fallback) are both restored."""
        transformer = _transformer({'GDP': 400.0, 'CPI': 20.0}, strategy='mask', target_frequency='Q')
        recovered = transformer.inverse_transform(_fit_transform(transformer, _monthly()))
        self._assert_restored(recovered, _monthly())

    def test_round_trip_keeps_the_untouched_column(self):
        """A column without delay goes through both steps unchanged."""
        X = _monthly(('Z', 'GDP'))
        transformer = _transformer({'GDP': 45.0})
        recovered = transformer.inverse_transform(_fit_transform(transformer, X))
        self._assert_restored(recovered, X)

    def test_empty_input_row_is_kept(self):
        """A row of the input that is entirely empty is not mistaken for an added row."""
        X = _monthly()
        X.loc['2023-06-01'] = np.nan
        transformer = _transformer()
        self._assert_restored(transformer.inverse_transform(_fit_transform(transformer, X)), X)

    def test_rows_with_values_outside_the_input_are_kept(self):
        """Inverting other data (later dates holding values) keeps every non-empty row.

        Gold values: the transformed frame extended by March 2024 (``GDP`` = 12) is
        inverted; that value goes back two months, to January 2024, outside the input of
        ``transform``, and is kept.
        """
        transformer = _transformer({'GDP': 45.0})
        shifted = _fit_transform(transformer, _monthly(('GDP',)))
        extended = pd.concat([shifted, pd.DataFrame({'GDP': [12.0]}, index=[TS('2024-03-01')])])
        recovered = transformer.inverse_transform(extended)
        assert recovered.loc[TS('2024-01-01'), 'GDP'] == 12.0

    def test_fallback_never_shows_a_value_before_its_date(self):
        """After the mask-to-shift fallback, no value appears before its own date (no look-ahead)."""
        transformer = _transformer({'GDP': 400.0}, strategy='mask', target_frequency='Q')
        result = _fit_transform(transformer, _monthly(('GDP',)))
        first = result['GDP'].first_valid_index()
        # Valeur d'or : la première valeur (janvier 2023) ne peut apparaître qu'à sa date ou plus tard
        assert first >= TS('2023-01-01')


# =============================================================================
# Robustesse (cas limites de CLAUDE.md)
# =============================================================================
class TestPanelOutputOrder:
    """A panel output is grouped by entity, in the input order, and sorted by date (ANO-DELAYS-043, fixed)."""

    @staticmethod
    def _panel() -> pd.DataFrame:
        """Two entities in non-alphabetical order (FR before DE), first half of 2024."""
        X = pd.DataFrame({'GDP': np.arange(6.0), 'CPI': np.arange(6.0)},
                         index=pd.date_range('2024-01-01', periods=6, freq='MS', name='date'))
        return pd.concat({'FR': X, 'DE': X}, names=['country', 'date'])

    def _transformed(self) -> pd.DataFrame:
        """Shift ``GDP`` (40 days, one month at 2024-06-15), leave ``CPI`` untouched."""
        transformer = _transformer({'GDP': 40.0}, prediction_date='2024-06-15')
        return _fit_transform(transformer, self._panel())

    def test_entities_keep_the_input_order_and_dates_are_sorted(self):
        """FR then DE, January to July 2024 each: the date added by the shift is not appended at the end."""
        dates = list(pd.date_range('2024-01-01', periods=7, freq='MS'))
        expected = [('FR', date) for date in dates] + [('DE', date) for date in dates]
        assert self._transformed().index.tolist() == expected

    def test_round_trip(self):
        """``inverse_transform`` restores the panel exactly."""
        transformer = _transformer({'GDP': 40.0}, prediction_date='2024-06-15')
        transformed = _fit_transform(transformer, self._panel())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            recovered = transformer.inverse_transform(transformed)
        pd.testing.assert_frame_equal(recovered, self._panel())


class TestRobustness:
    """Unsorted rows, special names, other index labels, degenerate inputs."""

    def test_unsorted_rows(self):
        """Shuffled rows give the gold output, sorted by date."""
        result = _fit_transform(_transformer(), perturb.shuffle_rows(_monthly(), seed=3))
        pd.testing.assert_frame_equal(result, _shifted_gold(), check_freq=False)

    def test_special_column_names(self):
        """Columns with spaces, accents and symbols are shifted under their own names."""
        renamed, mapping = perturb.with_special_column_names(_monthly())
        transformer = _transformer({mapping['GDP']: 45.0, mapping['CPI']: 20.0})
        result = _fit_transform(transformer, renamed)
        pd.testing.assert_frame_equal(result, _shifted_gold().rename(columns=mapping), check_freq=False)

    def test_end_of_period_labels(self):
        """Month-end labels are shifted by the same number of months: December (11) lands on February 29, 2024."""
        X = perturb.to_period_end(_monthly(('GDP',)))
        result = _fit_transform(_transformer({'GDP': 45.0}), X)
        assert (result['GDP'].last_valid_index(), result['GDP'].iloc[-1]) == (TS('2024-02-29'), 11.0)

    def test_period_index(self):
        """A ``PeriodIndex`` is shifted and returned as a ``PeriodIndex``."""
        result = _fit_transform(_transformer({'GDP': 45.0}), perturb.to_period_index(_monthly(('GDP',))))
        assert isinstance(result.index, pd.PeriodIndex)
        assert result['GDP'].last_valid_index() == pd.Period('2024-02', freq='M')

    def test_duplicated_dates_are_rejected(self):
        """Duplicated dates cannot be shifted: ``ValueError``."""
        transformer = _transformer()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            transformer.fit(_monthly())
            with pytest.raises(ValueError, match='duplicate'):
                transformer.transform(perturb.with_duplicated_rows(_monthly()))

    @pytest.mark.parametrize(
        'build',
        [pytest.param(perturb.single_observation, id='single-observation'),
         pytest.param(perturb.empty_like, id='empty')],
    )
    def test_too_few_observations_are_rejected(self, build):
        """No frequency can be detected: ``fit`` raises a ``ValueError``."""
        transformer = _transformer()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError):
                transformer.fit(build(_monthly()))

    def test_all_nan_delayed_column_is_left_as_is(self):
        """An all-NaN delayed column (no detectable frequency) is reported, the other columns are shifted."""
        X = _monthly().assign(CPI=np.nan)
        transformer = _transformer()
        with pytest.warns(UserWarning, match='CPI'):
            result = transformer.fit_transform(X)
        assert result['GDP'].last_valid_index() == TS('2024-02-01')
