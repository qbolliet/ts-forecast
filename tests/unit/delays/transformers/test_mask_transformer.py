"""Unit tests for ``MaskTransformer`` (``tsforecast.delays.transformers``).

Covers the public API of the class: ``__init__`` (``n_obs``, ``mask_frequency``,
``how``), the sklearn protocol (``get_params`` / ``set_params``, ``clone``,
``check_is_fitted``), ``fit``, ``transform`` and ``inverse_transform``. The
private helpers ``_mask_n_obs_per_period``, ``_extend_period_if_incomplete`` and
``_generate_periods`` are exercised through ``transform`` only.

Masking semantics (author decision, 2026-10-06): the index is cut into the
calendar periods of ``mask_frequency``; in each of them, ``how='last'`` masks the
observations on the last ``n_obs`` **positions** of the regular grid of the
index frequency detected at ``fit``, ``how='first'`` those on the first ones.
Positions are calendar positions everywhere: a position absent from the data
(start or end of the series, gap, irregular index) masks nothing, and an
observation is never masked because a neighbouring one is missing. This is the
purpose of the class within ``PublicationDelayTransformer``: reproduce, in every
period of the history, the information missing at the same position of the
period as the prediction date, without treating as unavailable an observation
that is.

Gold values are computed from the calendar, never copied from the output of the
code. The masked cells are stored at each ``transform`` call since ``fit`` (the
store accumulates, the latest value wins for a date masked twice) and put back
by ``inverse_transform``: a masked cell present in its input recovers its
**original** value, even when that input carries a prediction in it; any other
cell is returned as given, and the output has the index of the input.

Per-entity parameters go through ``PanelwiseTransformer`` (the class itself
rejects a ``MultiIndex``); per-variable parameters through one instance per
group of columns. The realistic scenario (notebook 3 datasets) lives in
``tests/integration/delays/test_shift_mask_realistic.py``.

Anomalies found while writing these tests are registered in
``tests/ANOMALIES.md`` (``ANO-DELAYS-016`` to ``-020``, ``-023``, ``-026`` and ``-027``,
all fixed).
"""
# Modules de base
import warnings
from typing import List

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

# Classe à tester et collaborateur réel pour les panels
from tsforecast.delays.transformers import MaskTransformer
from tsforecast.panel import PanelwiseTransformer

# Perturbations partagées
from tests.support.perturbations import (
    shuffle_rows,
    to_period_index,
    with_duplicated_rows,
    with_index_names,
    with_special_column_names,
)

TS = pd.Timestamp


# =============================================================================
# Constructeurs locaux de petits jeux à valeurs d'or calculables
# =============================================================================
def _series(freq: str = 'MS', periods: int = 12, start: str = '2024-01-01', name='x') -> pd.Series:
    """Build a regular float series ``0, 1, ..., periods - 1``.

    Args:
        freq: Pandas frequency of the index.
        periods: Number of observations.
        start: First date (rolled forward to the frequency anchor by pandas).
        name: Name of the series.

    Returns:
        The series, its values equal to their position.
    """
    index = pd.date_range(start, periods=periods, freq=freq)
    return pd.Series(np.arange(periods, dtype=float), index=index, name=name)


def _daily(start: str, end: str, name='x') -> pd.Series:
    """Build a daily float series between two dates (both included)."""
    index = pd.date_range(start, end, freq='D')
    return pd.Series(np.arange(len(index), dtype=float), index=index, name=name)


def _masked_dates(masked: pd.Series, fmt: str = '%Y-%m-%d') -> List[str]:
    """Return the dates of the masked (NaN) cells of a series, formatted."""
    return masked[masked.isna()].index.strftime(fmt).tolist()


def _days(month: str, days) -> List[str]:
    """Return ``['<month>-<day>', ...]`` for the given days of a ``'YYYY-MM'`` month."""
    return [f'{month}-{day:02d}' for day in days]


# Configurations de masquage aux valeurs d'or calculées à la main
# (index, mask_frequency, n_obs, how, dates masquées attendues)
GOLD_CASES = [
    # Valeur d'or : dernier mois de chaque trimestre
    pytest.param(_series('MS'), 'Q', 1, 'last', ['2024-03-01', '2024-06-01', '2024-09-01', '2024-12-01'],
                 id='MS-Q-1-last'),
    # Valeur d'or : premier mois de chaque trimestre
    pytest.param(_series('MS'), 'Q', 1, 'first', ['2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01'],
                 id='MS-Q-1-first'),
    # Valeur d'or : deux derniers mois de chaque trimestre
    pytest.param(_series('MS'), 'Q', 2, 'last',
                 ['2024-02-01', '2024-03-01', '2024-05-01', '2024-06-01',
                  '2024-08-01', '2024-09-01', '2024-11-01', '2024-12-01'], id='MS-Q-2-last'),
    # Valeur d'or : position fin de mois, dernier mois de chaque trimestre
    pytest.param(_series('ME', start='2024-01-31'), 'Q', 1, 'last',
                 ['2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31'], id='ME-Q-1-last'),
    # Valeur d'or : dernier trimestre de chaque année (2021-2023)
    pytest.param(_series('QE-DEC', start='2021-03-31'), 'Y', 1, 'last', ['2021-12-31', '2022-12-31', '2023-12-31'],
                 id='QE-Y-1-last'),
    # Valeur d'or : premier trimestre de chaque année (2022-2023)
    pytest.param(_series('QS', periods=8, start='2022-01-01'), 'Y', 1, 'first', ['2022-01-01', '2023-01-01'],
                 id='QS-Y-1-first'),
    # Valeur d'or : deux derniers jours de chaque mois, février 2024 bissextile
    pytest.param(_daily('2024-01-01', '2024-03-31'), 'M', 2, 'last',
                 ['2024-01-30', '2024-01-31', '2024-02-28', '2024-02-29', '2024-03-30', '2024-03-31'],
                 id='D-M-2-last'),
    # Valeur d'or : semaine du lundi au dimanche, le dimanche est masqué (2024-01-01 est un lundi)
    pytest.param(_series('D', periods=14), 'W', 1, 'last', ['2024-01-07', '2024-01-14'], id='D-W-1-last'),
    # Valeur d'or : semestres (multiplicateur de la fréquence de masque honoré)
    pytest.param(_series('MS'), '2Q', 1, 'last', ['2024-06-01', '2024-12-01'], id='MS-2Q-1-last'),
]


# =============================================================================
# Protocole sklearn
# =============================================================================
class TestSklearnProtocol:
    """``MaskTransformer`` follows the sklearn estimator conventions."""

    def test_get_params_returns_the_constructor_arguments(self):
        """``get_params`` exposes ``n_obs``, ``mask_frequency`` and ``how``."""
        params = MaskTransformer(n_obs=2, mask_frequency='Q', how='first').get_params()
        assert params == {'n_obs': 2, 'mask_frequency': 'Q', 'how': 'first'}

    def test_how_defaults_to_last(self):
        """``how`` defaults to ``'last'``."""
        assert MaskTransformer(n_obs=1, mask_frequency='Q').how == 'last'

    def test_set_params_changes_the_mask(self):
        """``set_params`` (used by ``PanelwiseTransformer`` per entity) changes the masked count."""
        masker = MaskTransformer(n_obs=1, mask_frequency='Q').set_params(n_obs=2)
        # Valeur d'or : deux mois masqués par trimestre sur douze mois
        assert masker.fit_transform(_series()).isna().sum() == 8

    def test_clone_keeps_the_parameters(self):
        """``clone`` rebuilds an estimator with the same parameters."""
        params = clone(MaskTransformer(n_obs=3, mask_frequency='M', how='first')).get_params()
        assert params == {'n_obs': 3, 'mask_frequency': 'M', 'how': 'first'}

    def test_clone_drops_the_masked_cells(self):
        """A clone of a transformed masker is unfitted and stores nothing."""
        masker = MaskTransformer(n_obs=1, mask_frequency='Q')
        masker.fit_transform(_series())
        assert not hasattr(clone(masker), 'masked_values_')

    def test_unfitted_masker_is_reported_unfitted(self):
        """``check_is_fitted`` raises ``NotFittedError`` before ``fit`` (ANO-DELAYS-018)."""
        with pytest.raises(NotFittedError):
            check_is_fitted(MaskTransformer(n_obs=1, mask_frequency='Q'))

    def test_fit_returns_self_and_is_fitted(self):
        """``fit`` returns the estimator itself, which ``check_is_fitted`` then accepts."""
        masker = MaskTransformer(n_obs=1, mask_frequency='Q')
        assert masker.fit(_series()) is masker
        check_is_fitted(masker)

    def test_fit_stores_the_index_components(self):
        """``fit`` stores the base, position and anchor of the index frequency."""
        masker = MaskTransformer(n_obs=1, mask_frequency='Y').fit(_series('QE-DEC', start='2021-03-31'))
        assert (masker.index_frequency_, masker.index_position_, masker.index_suffix_) == ('Q', 'E', 'DEC')

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_unfitted_masker_cannot_transform(self, method):
        """``transform`` and ``inverse_transform`` raise ``NotFittedError`` before ``fit``."""
        with pytest.raises(NotFittedError):
            getattr(MaskTransformer(n_obs=1, mask_frequency='Q'), method)(_series())

    def test_inverse_transform_after_fit_only_returns_its_input(self):
        """``fit`` alone masks nothing: ``inverse_transform`` has nothing to restore."""
        masker = MaskTransformer(n_obs=1, mask_frequency='Q').fit(_series())
        masked = _series().where(_series() > 2)
        pd.testing.assert_series_equal(masker.inverse_transform(masked), masked)

    def test_transform_uses_the_grid_detected_at_fit(self):
        """A single observation, whose frequency cannot be detected, is masked on the fitted grid."""
        masker = MaskTransformer(n_obs=1, mask_frequency='Q').fit(_series())
        # Valeur d'or : mars est le dernier mois du premier trimestre
        assert masker.transform(_series(periods=3).iloc[2:]).isna().all()


# =============================================================================
# Valeurs d'or : positions masquées dans des périodes complètes
# =============================================================================
class TestMaskGoldValues:
    """The ``n_obs`` first or last positions of each period are set to NaN, nothing else."""

    @pytest.mark.parametrize('series, mask_frequency, n_obs, how, expected', GOLD_CASES)
    def test_masked_positions(self, series, mask_frequency, n_obs, how, expected):
        """Masked dates match the calendar positions, for every index and mask frequency."""
        masked = MaskTransformer(n_obs=n_obs, mask_frequency=mask_frequency, how=how).fit_transform(series)
        assert _masked_dates(masked) == expected

    def test_sub_daily_index(self):
        """Two last hours of each day on an hourly index."""
        series = _series('h', periods=48)
        masked = MaskTransformer(n_obs=2, mask_frequency='D', how='last').fit_transform(series)
        # Valeur d'or : 22 h et 23 h des 1er et 2 janvier
        expected = ['2024-01-01 22', '2024-01-01 23', '2024-01-02 22', '2024-01-02 23']
        assert _masked_dates(masked, '%Y-%m-%d %H') == expected

    @pytest.mark.parametrize(
        'how, expected',
        [
            # Valeur d'or : dernier dimanche de janvier, février, mars 2024
            pytest.param('last', ['2024-01-28', '2024-02-25', '2024-03-31'], id='last'),
            # Valeur d'or : premier dimanche de chaque mois
            pytest.param('first', ['2024-01-07', '2024-02-04', '2024-03-03'], id='first'),
        ],
    )
    def test_weekly_index_masked_by_month(self, how, expected):
        """A weekly index (``W-SUN``) is masked per month like any finer index (ANO-DELAYS-027)."""
        series = _series('W-SUN', periods=13, start='2024-01-07')
        masked = MaskTransformer(n_obs=1, mask_frequency='M', how=how).fit_transform(series)
        assert _masked_dates(masked) == expected

    @pytest.mark.parametrize('n_obs', [pytest.param(3, id='equal'), pytest.param(5, id='larger')])
    def test_n_obs_covering_the_period_masks_everything(self, n_obs):
        """``n_obs`` at least the number of positions of a period masks every period entirely.

        ``PublicationDelayTransformer`` never asks for it (it falls back to a shift);
        the class itself does not prevent it, the result being well defined.
        """
        masked = MaskTransformer(n_obs=n_obs, mask_frequency='Q').fit_transform(_series())
        assert masked.isna().all()

    @pytest.mark.parametrize('data', [pytest.param(_series(), id='series'), pytest.param(_series().to_frame(), id='frame')])
    def test_zero_n_obs_returns_the_input(self, data):
        """``n_obs=0`` masks nothing."""
        masked = MaskTransformer(n_obs=0, mask_frequency='Q').fit_transform(data)
        assert masked.equals(data)

    @pytest.mark.parametrize('data', [pytest.param(_series(), id='series'), pytest.param(_series().to_frame(), id='frame')])
    def test_zero_n_obs_inverse_returns_its_input(self, data):
        """With nothing masked, ``inverse_transform`` returns a copy of its input."""
        masker = MaskTransformer(n_obs=0, mask_frequency='Q')
        masker.fit_transform(data)
        assert masker.inverse_transform(data).equals(data)

    def test_zero_n_obs_still_checks_the_frequency(self):
        """``n_obs=0`` with a mask frequency equal to the index one is rejected like any other."""
        with pytest.raises(ValueError, match="strictly higher than the mask frequency"):
            MaskTransformer(n_obs=0, mask_frequency='M').fit(_series())

    def test_output_is_a_copy(self):
        """Masking never modifies the input."""
        original = _series()
        MaskTransformer(n_obs=2, mask_frequency='Q').fit_transform(original)
        assert original.notna().all()


# =============================================================================
# Aller-retour transform / inverse_transform
# =============================================================================
class TestRoundTrip:
    """Unmasked cells are untouched, masked cells recover their original value."""

    @pytest.mark.parametrize('series, mask_frequency, n_obs, how, expected', GOLD_CASES)
    def test_unmasked_cells_are_unchanged(self, series, mask_frequency, n_obs, how, expected):
        """Every cell not masked keeps its input value."""
        masked = MaskTransformer(n_obs=n_obs, mask_frequency=mask_frequency, how=how).fit_transform(series)
        kept = masked.notna()
        pd.testing.assert_series_equal(masked[kept], series[kept], check_freq=False)

    @pytest.mark.parametrize('series, mask_frequency, n_obs, how, expected', GOLD_CASES)
    def test_inverse_transform_restores_the_input(self, series, mask_frequency, n_obs, how, expected):
        """``inverse_transform(transform(x)) == x`` (``DatetimeIndex.freq`` aside)."""
        masker = MaskTransformer(n_obs=n_obs, mask_frequency=mask_frequency, how=how)
        recovered = masker.inverse_transform(masker.fit_transform(series))
        pd.testing.assert_series_equal(recovered, series, check_freq=False)

    def test_masked_cell_recovers_its_original_value_over_a_prediction(self):
        """A masked cell filled with a prediction gets its original value back."""
        series = _daily('2024-01-01', '2024-02-29')
        masker = MaskTransformer(n_obs=2, mask_frequency='M', how='last')
        # Construction : les cellules masquées sont remplies par une « prédiction » arbitraire
        predictions = masker.fit_transform(series).fillna(-999.0)
        pd.testing.assert_series_equal(masker.inverse_transform(predictions), series, check_freq=False)

    def test_unmasked_cell_is_returned_as_given(self):
        """A cell that was not masked keeps the value of the ``inverse_transform`` input."""
        masker = MaskTransformer(n_obs=1, mask_frequency='Q', how='last')
        masked = masker.fit_transform(_series())
        # Construction : modification d'une cellule non masquée (janvier)
        masked.loc[TS('2024-01-01')] = 42.0
        assert masker.inverse_transform(masked).loc[TS('2024-01-01')] == 42.0

    def test_originally_missing_cell_stays_missing(self):
        """A NaN at a masked position is restored as NaN."""
        series = _series()
        series.loc[TS('2024-03-01')] = np.nan
        masker = MaskTransformer(n_obs=1, mask_frequency='Q', how='last')
        assert np.isnan(masker.inverse_transform(masker.fit_transform(series)).loc[TS('2024-03-01')])

    def test_train_and_test_sets_can_both_be_inverted(self):
        """Masked cells accumulate over the ``transform`` calls since ``fit``.

        Fitted and transformed on a training set, then transformed on a test set,
        the masker restores both.
        """
        train, test = _series(periods=6), _series(periods=6, start='2024-07-01') + 100.0
        masker = MaskTransformer(n_obs=1, mask_frequency='Q', how='last')
        masked_train = masker.fit_transform(train)
        masked_test = masker.transform(test)
        recovered = (masker.inverse_transform(masked_train), masker.inverse_transform(masked_test))
        pd.testing.assert_series_equal(pd.concat(recovered), pd.concat([train, test]), check_freq=False)

    def test_latest_transform_wins_for_a_date_masked_twice(self):
        """Transforming the same dates twice keeps the values of the latest call."""
        masker = MaskTransformer(n_obs=1, mask_frequency='Q', how='last')
        masker.fit_transform(_series())
        second = _series() + 100.0
        recovered = masker.inverse_transform(masker.transform(second))
        pd.testing.assert_series_equal(recovered, second, check_freq=False)

    def test_refit_empties_the_store(self):
        """A new ``fit`` forgets the cells masked before it."""
        masker = MaskTransformer(n_obs=1, mask_frequency='Q', how='last')
        masked = masker.fit_transform(_series())
        masker.fit(_series())
        assert masker.inverse_transform(masked).isna().sum() == 4

    def test_inverse_transform_keeps_the_index_of_its_input(self):
        """Inverting the first half of a masked series returns that half only (ANO-DELAYS-020)."""
        masker = MaskTransformer(n_obs=1, mask_frequency='Q', how='last')
        first_half = masker.fit_transform(_series()).iloc[:6]
        pd.testing.assert_index_equal(masker.inverse_transform(first_half).index, first_half.index)


# =============================================================================
# Bords de série : périodes incomplètes (tri des échecs hérités)
# =============================================================================
class TestBoundaries:
    """Periods cut by the start or the end of the data are masked on calendar positions."""

    def test_series_start_with_how_first(self):
        """Data starting on 10 January: the first days of January are absent, nothing is masked there.

        Legacy test (category (c)): it expected 10-12 January masked, a position relative
        to the data, which the extremities handling (commit ``76ed126``) deliberately
        replaced by calendar positions. February is complete and masked on 1-3.
        """
        series = _daily('2024-01-10', '2024-02-29')
        masked = MaskTransformer(n_obs=3, mask_frequency='M', how='first').fit_transform(series)
        # Valeur d'or : positions 1-3 janvier absentes ; 1-3 février masqués
        assert _masked_dates(masked) == _days('2024-02', [1, 2, 3])

    def test_series_start_with_how_last(self):
        """Data starting on 15 January: the last five days of each complete month are masked."""
        series = _daily('2024-01-15', '2024-02-29')
        masked = MaskTransformer(n_obs=5, mask_frequency='M', how='last').fit_transform(series)
        # Valeur d'or : 27-31 janvier, 25-29 février (2024 bissextile)
        assert _masked_dates(masked) == _days('2024-01', range(27, 32)) + _days('2024-02', range(25, 30))

    def test_series_end_with_how_first(self):
        """Data ending on 10 March: 1-4 March are the first positions of March and are masked (ANO-DELAYS-016)."""
        series = _daily('2024-01-01', '2024-03-10')
        masked = MaskTransformer(n_obs=4, mask_frequency='M', how='first').fit_transform(series)
        # Valeur d'or : 1-4 de chaque mois, mars inclus (les positions existent)
        expected = _days('2024-01', range(1, 5)) + _days('2024-02', range(1, 5)) + _days('2024-03', range(1, 5))
        assert _masked_dates(masked) == expected

    def test_series_end_with_how_last(self):
        """Data ending on 15 March: the last positions of March (29-31) are absent, March is not masked.

        Legacy test (category (c)): it expected 13-15 March masked, positions relative to
        the data; see ``test_series_start_with_how_first``.
        """
        series = _daily('2024-01-01', '2024-03-15')
        masked = MaskTransformer(n_obs=3, mask_frequency='M', how='last').fit_transform(series)
        # Valeur d'or : 29-31 janvier, 27-29 février ; 29-31 mars hors données
        assert _masked_dates(masked) == _days('2024-01', [29, 30, 31]) + _days('2024-02', [27, 28, 29])

    def test_first_incomplete_quarter_on_month_start_index(self):
        """Data starting in February: March, last month of Q1, is masked."""
        series = _series('MS', periods=11, start='2024-02-01')
        masked = MaskTransformer(n_obs=1, mask_frequency='Q', how='last').fit_transform(series)
        # Valeur d'or : mars, juin, septembre, décembre
        assert _masked_dates(masked) == ['2024-03-01', '2024-06-01', '2024-09-01', '2024-12-01']

    @pytest.mark.parametrize(
        'series',
        [
            pytest.param(_series('MS', periods=11), id='start-anchored'),
            pytest.param(_series('ME', periods=11, start='2024-01-31'), id='end-anchored'),
        ],
    )
    def test_last_incomplete_quarter_with_how_last(self, series):
        """Data ending in November: December, last position of Q4, is absent; Q4 is not masked."""
        masked = MaskTransformer(n_obs=1, mask_frequency='Q', how='last').fit_transform(series)
        # Valeur d'or : mars, juin, septembre (jour de position début ou fin)
        assert _masked_dates(masked, '%Y-%m') == ['2024-03', '2024-06', '2024-09']

    @pytest.mark.parametrize(
        'how, expected',
        [
            # Valeur d'or : 31 mars, dernière position du premier trimestre
            pytest.param('last', ['2024-03-31'], id='last'),
            # Valeur d'or : 31 janvier, première position, absente des données
            pytest.param('first', [], id='first'),
        ],
    )
    def test_single_quarter_cut_at_its_start(self, how, expected):
        """Month ends February and March only: one quarter, incomplete at its start, complete at its end."""
        series = pd.Series([1., 2.], index=pd.to_datetime(['2024-02-29', '2024-03-31']))
        masked = MaskTransformer(n_obs=1, mask_frequency='Q', how=how).fit_transform(series)
        assert _masked_dates(masked) == expected

    def test_single_incomplete_period_with_how_last(self):
        """Five days of January, ten last positions asked: none of them is in the data."""
        masked = MaskTransformer(n_obs=10, mask_frequency='M', how='last').fit_transform(_daily('2024-01-01', '2024-01-05'))
        # Valeur d'or : positions 22-31 janvier, toutes hors données
        assert masked.notna().all()

    def test_single_incomplete_period_with_how_first(self):
        """Five days of January, ten first positions asked: the five days are masked."""
        masked = MaskTransformer(n_obs=10, mask_frequency='M', how='first').fit_transform(_daily('2024-01-01', '2024-01-05'))
        assert masked.isna().all()

    @pytest.mark.parametrize(
        'mask_frequency, expected',
        [
            # Valeur d'or : 2 derniers jours de chaque mois ; avril (série jusqu'au 29) : le 29 est l'avant-dernier jour
            pytest.param('M', ['01-30', '01-31', '02-28', '02-29', '03-30', '03-31', '04-29'], id='monthly'),
            # Valeur d'or : seul le premier trimestre est complet ; positions 29-30 juin hors données
            pytest.param('Q', ['03-30', '03-31'], id='quarterly'),
        ],
    )
    def test_position_less_frequency_raises_no_pandas_deprecation(self, mask_frequency, expected):
        """A daily index has no position: ``'M'`` / ``'Q'`` must not reach pandas as bare aliases.

        Moved from ``TestMaskTransformerPandasAliases``; its monthly gold value pinned
        April as unmasked (ANO-DELAYS-016), while 29 April is a calendar position of
        the two last days of April.
        """
        series = pd.Series(range(120), index=pd.date_range('2024-01-01', periods=120, freq='D'), name='x', dtype=float)
        masker = MaskTransformer(n_obs=2, mask_frequency=mask_frequency, how='last')
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message=".*is deprecated", category=FutureWarning)
            masked = masker.fit_transform(series)
        assert _masked_dates(masked, '%m-%d') == expected

    def test_boundary_round_trip(self):
        """Incomplete periods at both ends: ``inverse_transform`` restores the input."""
        series = _daily('2024-01-10', '2024-03-15')
        masker = MaskTransformer(n_obs=3, mask_frequency='M', how='first')
        pd.testing.assert_series_equal(masker.inverse_transform(masker.fit_transform(series)), series, check_freq=False)

    def test_gap_inside_a_period_masks_nothing_on_the_missing_position(self):
        """A missing 31 January: the last position of January is absent, 30 January stays (ANO-DELAYS-017)."""
        index = pd.date_range('2024-01-01', '2024-02-29', freq='D').drop(TS('2024-01-31'))
        series = pd.Series(np.arange(len(index), dtype=float), index=index)
        masked = MaskTransformer(n_obs=1, mask_frequency='M', how='last').fit_transform(series)
        # Valeur d'or : seul le 29 février (dernière position de février) est masqué
        assert _masked_dates(masked) == ['2024-02-29']

    def test_lone_observation_of_a_period_is_masked_on_its_position_only(self):
        """Annual values on 1 January, alone in their quarter: January is not the last month, nothing is masked."""
        index = pd.DatetimeIndex(['2015-01-01', '2016-01-01', '2017-01-01']).append(
            pd.date_range('2018-01-01', periods=6, freq='MS'))
        series = pd.Series(np.arange(len(index), dtype=float), index=index)
        masked = MaskTransformer(n_obs=1, mask_frequency='Q', how='last').fit_transform(series)
        # Valeur d'or : mars et juin 2018 seulement
        assert _masked_dates(masked) == ['2018-03-01', '2018-06-01']


# =============================================================================
# DataFrame, paramètres par variable
# =============================================================================
class TestDataFrameAndVariables:
    """All columns of a frame share the masked rows; per-variable masks use one instance per group."""

    @staticmethod
    def _frame() -> pd.DataFrame:
        index = pd.date_range('2024-01-01', '2024-03-31', freq='D')
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            'col1': range(len(index)),
            'col2': range(100, 100 + len(index)),
            'col3': rng.normal(size=len(index)),
        }, index=index)

    def test_every_column_has_the_same_masked_rows(self):
        """A single masker masks whole rows."""
        masked = MaskTransformer(n_obs=2, mask_frequency='M', how='last').fit_transform(self._frame())
        # Valeur d'or : six lignes entièrement masquées, aucune cellule isolée
        assert masked.isna().all(axis=1).sum() == 6 and masked.isna().any(axis=1).sum() == 6

    def test_dataframe_inverse_transform_restores_the_values(self):
        """``inverse_transform`` restores every value of a frame.

        Legacy ``test_mask_dataframe_inverse_transform`` (category (c) for
        ``DatetimeIndex.freq``, a cached attribute not part of the contract); integer
        columns recover their dtype.
        """
        masker = MaskTransformer(n_obs=4, mask_frequency='M', how='first')
        recovered = masker.inverse_transform(masker.fit_transform(self._frame()))
        pd.testing.assert_frame_equal(recovered, self._frame(), check_freq=False)

    def test_masked_integer_columns_are_float(self):
        """Masked integer columns carry NaN, hence are float after ``transform``."""
        masked = MaskTransformer(n_obs=4, mask_frequency='M', how='first').fit_transform(self._frame())
        assert masked.dtypes.tolist() == [np.dtype('float64')] * 3

    def test_integer_column_with_a_prediction_left_stays_float(self):
        """A non-integral value in an unmasked cell keeps the column float at the inversion."""
        masker = MaskTransformer(n_obs=1, mask_frequency='M', how='first')
        masked = masker.fit_transform(self._frame()[['col1']])
        masked.loc[TS('2024-01-15'), 'col1'] = 14.5
        assert masker.inverse_transform(masked)['col1'].dtype == np.dtype('float64')

    def test_integer_column_with_a_missing_value_left_stays_float(self):
        """A NaN left in an unmasked cell keeps the integer column float at the inversion."""
        masker = MaskTransformer(n_obs=1, mask_frequency='M', how='first')
        masked = masker.fit_transform(self._frame()[['col1']])
        masked.loc[TS('2024-01-15'), 'col1'] = np.nan
        assert masker.inverse_transform(masked)['col1'].dtype == np.dtype('float64')

    def test_integer_column_with_text_left_is_returned_as_given(self):
        """An unmasked cell holding text prevents the cast back to integers: the column is returned as given."""
        masker = MaskTransformer(n_obs=1, mask_frequency='M', how='first')
        masked = masker.fit_transform(self._frame()[['col1']]).astype(object)
        masked.loc[TS('2024-01-15'), 'col1'] = 'prediction'
        assert masker.inverse_transform(masked).loc[TS('2024-01-15'), 'col1'] == 'prediction'

    def test_float_column_keeps_the_dtype_of_the_inverse_input(self):
        """Only integer and boolean columns are cast back: a float32 column given as float64 stays float64."""
        frame = self._frame()[['col3']].astype('float32')
        masker = MaskTransformer(n_obs=1, mask_frequency='M', how='first')
        masked = masker.fit_transform(frame).astype('float64')
        assert masker.inverse_transform(masked)['col3'].dtype == np.dtype('float64')

    def test_inverse_of_rows_without_masked_cells_returns_them_as_given(self):
        """Inverting rows none of which was masked changes no value (integer columns get their dtype back)."""
        masker = MaskTransformer(n_obs=1, mask_frequency='M', how='first')
        masked = masker.fit_transform(self._frame())
        unmasked_rows = masked.loc['2024-01-10':'2024-01-20']
        pd.testing.assert_frame_equal(masker.inverse_transform(unmasked_rows), self._frame().loc['2024-01-10':'2024-01-20'],
                                      check_freq=False)

    def test_string_column_round_trip(self):
        """An object column is masked and restored like the others."""
        frame = pd.DataFrame({'label': [f'v{i}' for i in range(12)]}, index=_series().index)
        masker = MaskTransformer(n_obs=1, mask_frequency='Q', how='last')
        pd.testing.assert_frame_equal(masker.inverse_transform(masker.fit_transform(frame)), frame, check_freq=False)

    def test_boundary_extension_of_a_frame(self):
        """A frame starting on 10 January is masked like a series (calendar positions)."""
        frame = self._frame().loc['2024-01-10':'2024-02-29']
        masked = MaskTransformer(n_obs=3, mask_frequency='M', how='first').fit_transform(frame)
        assert _masked_dates(masked['col1']) == _days('2024-02', [1, 2, 3])

    def test_last_incomplete_period_of_a_frame(self):
        """A monthly frame ending in November leaves Q4 unmasked."""
        frame = pd.DataFrame({'a': np.arange(11.), 'b': np.arange(11.)}, index=pd.date_range('2024-01-01', periods=11, freq='MS'))
        masked = MaskTransformer(n_obs=1, mask_frequency='Q', how='last').fit_transform(frame)
        assert _masked_dates(masked['b'], '%m') == ['03', '06', '09']

    def test_frame_with_nothing_masked_round_trip(self):
        """A frame whose masked positions are all absent restores as given."""
        frame = self._frame().loc['2024-01-01':'2024-01-05']
        masker = MaskTransformer(n_obs=10, mask_frequency='M', how='last')
        masked = masker.fit_transform(frame)
        pd.testing.assert_frame_equal(masker.inverse_transform(masked), masked)

    def test_special_column_names_are_preserved(self):
        """Columns with spaces, accents or symbols come out unchanged."""
        renamed, mapping = with_special_column_names(self._frame())
        masked = MaskTransformer(n_obs=1, mask_frequency='M').fit_transform(renamed)
        assert masked.columns.tolist() == list(mapping.values())

    def test_per_variable_masks_with_one_instance_per_group(self):
        """Two groups of columns masked with 1 and 2 observations per quarter."""
        frame = pd.DataFrame({'a': np.arange(12.), 'b': np.arange(12.)}, index=_series().index)
        first = MaskTransformer(n_obs=1, mask_frequency='Q').fit_transform(frame[['a']])
        second = MaskTransformer(n_obs=2, mask_frequency='Q').fit_transform(frame[['b']])
        # Valeur d'or : 4 et 8 mois masqués sur douze
        assert (first['a'].isna().sum(), second['b'].isna().sum()) == (4, 8)


# =============================================================================
# Validation des paramètres
# =============================================================================
class TestParameterValidation:
    """Mask frequency, ``how`` and ``n_obs`` are checked."""

    @pytest.mark.parametrize(
        'series, mask_frequency',
        [
            pytest.param(_series('D', periods=30), 'D', id='equal'),
            pytest.param(_series('MS'), 'D', id='finer'),
        ],
    )
    def test_mask_frequency_not_coarser_than_the_index_raises(self, series, mask_frequency):
        """The index must be strictly finer than the mask periods."""
        with pytest.raises(ValueError, match="should be strictly higher than the mask frequency"):
            MaskTransformer(n_obs=1, mask_frequency=mask_frequency).fit_transform(series)

    def test_unknown_how_raises(self):
        """``how`` other than ``'first'`` / ``'last'`` is rejected (ANO-DELAYS-019)."""
        with pytest.raises(ValueError, match="how must be 'first' or 'last', got 'middle'"):
            MaskTransformer(n_obs=1, mask_frequency='Q', how='middle').fit(_series())

    def test_negative_n_obs_raises(self):
        """A negative number of observations to mask is rejected (ANO-DELAYS-019)."""
        with pytest.raises(ValueError, match="'n_obs' must be a non-negative integer, got -1"):
            MaskTransformer(n_obs=-1, mask_frequency='Q').fit(_series())

    @pytest.mark.parametrize('value', [pytest.param(1.5, id='float'), pytest.param(True, id='bool')])
    def test_non_integer_n_obs_raises(self, value):
        """``n_obs`` must be an integer."""
        with pytest.raises(TypeError, match="'n_obs' must be an integer"):
            MaskTransformer(n_obs=value, mask_frequency='Q').fit(_series())

    def test_unknown_mask_frequency_raises(self):
        """An unknown mask frequency is rejected at ``fit``."""
        with pytest.raises(ValueError, match="Unsupported frequency: foo"):
            MaskTransformer(n_obs=1, mask_frequency='foo').fit(_series())

    def test_anchored_mask_frequency_is_honoured(self):
        """Fiscal years ending in November (``'YE-NOV'``): November is the last month of each year."""
        masked = MaskTransformer(n_obs=1, mask_frequency='YE-NOV').fit_transform(_series(periods=24))
        # Valeur d'or : novembre 2024 et novembre 2025
        assert _masked_dates(masked, '%Y-%m') == ['2024-11', '2025-11']


# =============================================================================
# Robustesse d'index et cas limites
# =============================================================================
class TestIndexRobustness:
    """Unsorted, Period, named, duplicated, short or non-datetime indexes."""

    def test_unsorted_input_is_sorted_then_masked(self):
        """A shuffled series gives the mask of the sorted series."""
        masked = MaskTransformer(n_obs=1, mask_frequency='Q').fit_transform(shuffle_rows(_series(), seed=1))
        pd.testing.assert_series_equal(masked, MaskTransformer(n_obs=1, mask_frequency='Q').fit_transform(_series()),
                                       check_freq=False)

    def test_period_index_is_returned_as_periods(self):
        """A ``PeriodIndex`` input gives a ``PeriodIndex`` output, masked on the last month of each quarter."""
        masked = MaskTransformer(n_obs=1, mask_frequency='Q').fit_transform(to_period_index(_series()))
        assert isinstance(masked.index, pd.PeriodIndex) and masked[masked.isna()].index.month.tolist() == [3, 6, 9, 12]

    def test_period_index_round_trip(self):
        """A ``PeriodIndex`` input is restored exactly."""
        original = to_period_index(_series())
        masker = MaskTransformer(n_obs=1, mask_frequency='Q')
        pd.testing.assert_series_equal(masker.inverse_transform(masker.fit_transform(original)), original)

    def test_index_name_is_preserved(self):
        """A non-standard index name survives the round trip."""
        masker = MaskTransformer(n_obs=1, mask_frequency='Q')
        recovered = masker.inverse_transform(masker.fit_transform(with_index_names(_series(), 'periode')))
        assert recovered.index.name == 'periode'

    def test_duplicated_index_is_rejected(self):
        """Duplicated dates are rejected by the validation."""
        with pytest.raises(ValueError, match="Index contains duplicate values"):
            MaskTransformer(n_obs=1, mask_frequency='Q').fit(with_duplicated_rows(_series()))

    @pytest.mark.parametrize('periods', [pytest.param(0, id='empty'), pytest.param(1, id='single')])
    def test_fewer_than_two_observations_are_rejected(self, periods):
        """An empty or single-observation series cannot carry a frequency."""
        with pytest.raises(ValueError, match="minimum required is 2"):
            MaskTransformer(n_obs=1, mask_frequency='Q').fit(_series(periods=periods))

    def test_non_datetime_index_is_rejected(self):
        """An integer index is rejected with an explicit message."""
        with pytest.raises(ValueError, match="Index cannot be converted to datetime"):
            MaskTransformer(n_obs=1, mask_frequency='M').fit(pd.Series([1., 2., 3.], index=[0, 1, 2]))

    def test_multiplied_index_is_rejected_at_fit(self):
        """A ``2MS`` index is not treated as monthly."""
        series = pd.Series(range(6), index=pd.date_range('2024-01-01', periods=6, freq='2MS'), dtype=float)
        with pytest.raises(ValueError, match="Multiplied index frequency"):
            MaskTransformer(n_obs=1, mask_frequency='Q', how='last').fit(series)

    def test_non_pandas_input_is_rejected_by_fit(self):
        """Lists are rejected by ``fit``."""
        with pytest.raises(ValueError, match="must be a pandas Series or DataFrame"):
            MaskTransformer(n_obs=2, mask_frequency='M').fit([1, 2, 3])

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_non_pandas_input_is_rejected_after_fit(self, method):
        """Lists are rejected by ``transform`` and ``inverse_transform``."""
        masker = MaskTransformer(n_obs=1, mask_frequency='Q').fit(_series())
        with pytest.raises(ValueError, match="must be a pandas Series or DataFrame"):
            getattr(masker, method)([1, 2, 3])

    def test_multiindex_input_raises_a_clear_error(self):
        """A panel passed directly raises a ``ValueError`` pointing to ``PanelwiseTransformer`` (ANO-DELAYS-023)."""
        index = pd.MultiIndex.from_product([['FR', 'DE'], pd.date_range('2024-01-01', periods=6, freq='MS')])
        panel = pd.DataFrame({'a': np.arange(12.)}, index=index)
        with pytest.raises(ValueError, match="wrap it in a PanelwiseTransformer"):
            MaskTransformer(n_obs=1, mask_frequency='Q').fit(panel)


# =============================================================================
# Panel : paramètres par entité, panel désordonné (via PanelwiseTransformer)
# =============================================================================
class TestPanelThroughPanelwise:
    """Per-entity masks through ``PanelwiseTransformer``, the panel path of the class."""

    @staticmethod
    def _panel() -> pd.DataFrame:
        index = pd.MultiIndex.from_product(
            [['FR', 'DE'], pd.date_range('2024-01-01', periods=6, freq='MS')], names=['country', 'date'])
        return pd.DataFrame({'a': np.arange(12.), 'b': np.arange(12.) * 10}, index=index)

    def test_per_entity_mask(self):
        """``entity_kwargs`` gives FR two masked months per quarter, DE keeps the default of one."""
        transformer = PanelwiseTransformer(
            transformer=MaskTransformer(n_obs=1, mask_frequency='Q'),
            entity_kwargs={('FR',): {'n_obs': 2}}, time_col=None, panel_cols=None)
        masked = transformer.fit_transform(self._panel())
        dates = {entity: _masked_dates(masked.loc[entity, 'a'], '%m') for entity in ('FR', 'DE')}
        # Valeur d'or : FR février, mars, mai, juin ; DE mars, juin
        assert dates == {'FR': ['02', '03', '05', '06'], 'DE': ['03', '06']}

    def test_per_entity_round_trip(self):
        """Each entity is restored by ``inverse_transform``, entities in their input order (FR, DE)."""
        transformer = PanelwiseTransformer(
            transformer=MaskTransformer(n_obs=1, mask_frequency='Q'),
            entity_kwargs={('FR',): {'n_obs': 2}}, time_col=None, panel_cols=None)
        recovered = transformer.inverse_transform(transformer.fit_transform(self._panel()))
        pd.testing.assert_frame_equal(recovered, self._panel())

    def test_unsorted_panel_gives_the_sorted_result(self):
        """A shuffled panel (``auto_sort=True``) is masked like the sorted one."""
        def mask(panel):
            return PanelwiseTransformer(
                transformer=MaskTransformer(n_obs=1, mask_frequency='Q'),
                time_col=None, panel_cols=None, auto_sort=True).fit_transform(panel).sort_index()

        pd.testing.assert_frame_equal(mask(shuffle_rows(self._panel(), seed=0)), mask(self._panel()))
