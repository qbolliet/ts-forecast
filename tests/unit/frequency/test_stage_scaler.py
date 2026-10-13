"""Tests for tsforecast.frequency.stage_scaler.

``StageScaler`` holds the scale arithmetic of one imputation stage of
``HighFrequencyImputer`` (spec ``high_frequency_imputer2_architecture.md``
§9): every divisor is a pure function of the configuration and of the
frequencies handed to it, never of the data. ``'constant'`` divides by
``FrequencyConverter.get_conversion_factor`` (M -> Y = 12), ``'calendar'`` by
the real calendar count, one value per row (February 28 or 29, quarters of 90
to 92 days), ``False`` divides nothing, the target included (D15).

Contracts pinned here:

- golden values, hand-derived, for every (source, prediction) pair of
  ``{Y, Q, M}²`` under ``'constant'`` (12, 4, 3, 1, uniform 1/3, 1/4, 1/12)
  and ``'calendar'`` (whole counts, and real shares of time when the stage is
  coarser: 31/91, 29/91, 91/366…), plus the day counts of ``'calendar'``
  (February 2023 / 2024, quarters of 90 / 91 / 92 days, years of 365 / 366
  days), with start and end positions (``YS`` / ``YE``, dates at period start
  or end);
- the shape rule: the return type (scalar / ``Series`` / ``DataFrame``)
  follows the configuration — mode, form of ``grid_freq`` / ``source_freq``,
  frequencies resolved per entity, ``produced_freq`` — never the divisor
  values, divisors all equal to ``1.0`` included (defect B12);
- the carried-period rule of §9.2 (``'raw_anchors'`` spans one period of a
  column coarser than the grid, every other way — and ``'raw_anchors'`` on a
  column at least as fine as the grid, read by the materializer's rank 1,
  ANO-FREQ-042 — one period of the grid), the fallbacks
  (incomparable frequency -> default divisor for a covariate, error for the
  target), the per-row target divisor of §5.4 and the dict form;
- ``inverse_transform(transform(X)) == X`` to ``1e-12`` on features and
  target, ``transform`` telling them apart by type alone;
- per-entity frequencies (§5.8 R5): ``PANEL-F`` golden values of the stage
  ``Q`` table (fractional divisor ``1/3`` of the monthly ``IT`` block) and
  the realistic ``heterogeneous_coverage_panel`` (``depenses_publiques_pib``
  annual for France / Italie, quarterly for Allemagne) and
  ``irregular_index_timeseries``;
- per-entity modes (D41): ``{column: {entity: mode}}`` closed on itself,
  ``False`` refused within it, full entity keys only (a partial key of a
  two-level panel matches none), golden divisors 31 / 28 against 30 on the
  target, the baked factor and the covariates;
- index robustness: shuffled, duplicated, empty, single-row grids,
  three-level ``MultiIndex``, special column names;
- sklearn protocol: ``clone``, ``get_params`` / ``set_params``,
  ``NotFittedError``, ``fit_transform``, divisor methods usable without
  ``fit``.

Corrections of ANO-FREQ-028 to ANO-FREQ-036, the ``'raw_anchors'`` side of
ANO-FREQ-042 and the plain entity keys of ANO-FREQ-037 (``tests/ANOMALIES.md``).

Covered symbols: ``StageScaler`` (``feature_divisors``, ``target_divisor``,
``fit_scale_factor``, ``apply``, ``invert``, ``resolve_mode``,
``validate_columns``, ``fit``, ``transform``, ``inverse_transform``),
``DEFAULT_SCALE_KEY``, ``DEFAULT_SCALE_MODE``.
"""
# Modules de base
import calendar
from typing import Tuple

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Sklearn
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

# Objets testés
from tsforecast.frequency.stage_scaler import (
    DEFAULT_SCALE_KEY,
    DEFAULT_SCALE_MODE,
    StageScaler,
)

# Perturbations partagées
from tests.support.perturbations import (
    shuffle_rows,
    to_three_level_index,
    with_duplicated_rows,
    with_special_column_names,
)


# Seuil de la symétrie transform / inverse_transform : un écart supérieur est
# une anomalie, jamais une tolérance à élargir
_EXACT = 1e-12

# Fréquences de bloc du jeu PANEL-F (§2.5) : v annuelle, trimestrielle, mensuelle
_PANEL_F_BLOCKS = {('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'}

# Fréquence de publication de depenses_publiques_pib par entité (jeu du notebook 3)
_DEPENSES_BLOCKS = {('France',): 'Y', ('Allemagne',): 'Q', ('Italie',): 'Y'}


# Fabrique de scaler : configuration par défaut cohérente, surchargeable
def _make_scaler(**overrides) -> StageScaler:
    """Build a StageScaler with a coherent default configuration.

    Args:
        **overrides: Constructor arguments replacing the defaults.

    Returns:
        An unfitted ``StageScaler`` (stage ``M``, variable ``Y``).
    """
    params = dict(
        scale_features='constant',
        source_freq='Y',
        pred_freq='M',
        column_frequencies={'m1': 'M', 'q1': 'Q', 'a2': 'Y'},
    )
    params.update(overrides)
    return StageScaler(**params)


# Grille mensuelle de fin de mois, calquée sur le jeu TS du §2.2
def _monthly_grid(start: str = '2021-01-31', periods: int = 12) -> pd.DatetimeIndex:
    """Build a month-end grid.

    Args:
        start: First month end.
        periods: Number of months.

    Returns:
        A ``DatetimeIndex`` of month ends.
    """
    return pd.date_range(start=start, periods=periods, freq='ME')


# Grille de panel (entité, date) en fin de mois
def _panel_grid(entities: Tuple[str, ...] = ('DE', 'FR'), periods: int = 3) -> pd.MultiIndex:
    """Build a month-end panel grid.

    Args:
        entities: Entity names, outer level.
        periods: Number of months per entity.

    Returns:
        A ``MultiIndex`` (``country``, ``date``).
    """
    return pd.MultiIndex.from_product(
        [list(entities), _monthly_grid(periods=periods)], names=['country', 'date']
    )


# Diffusion d'un diviseur de covariable, compact ou par ligne, sur une grille
def _feature_rows(divisors, column: str, index: pd.Index) -> np.ndarray:
    """Read the divisor of one covariate on every row of a grid, whatever its form.

    Lets a test check VALUES without depending on the return shape, which is
    pinned by dedicated tests (``TestShapeRule``).

    Args:
        divisors: Return of ``feature_divisors`` (``Series`` or ``DataFrame``).
        column: Covariate name.
        index: Grid the divisors apply to.

    Returns:
        Array of one divisor per row of ``index``.
    """
    if isinstance(divisors, pd.DataFrame):
        return divisors[column].to_numpy(dtype=float)
    return np.full(len(index), float(divisors[column]))


# Masque des lignes d'une entité sur le premier niveau d'un panel
def _rows_of(index: pd.MultiIndex, entity: str) -> np.ndarray:
    """Boolean mask of the rows of one entity.

    Args:
        index: Panel grid.
        entity: Entity name (first level).

    Returns:
        Boolean array aligned on ``index``.
    """
    return np.asarray(index.get_level_values(0) == entity)


# Nombre de jours d'un trimestre, par la bibliothèque standard (source
# indépendante de pandas et du convertisseur)
def _days_in_quarter(year: int, quarter: int) -> int:
    """Count the days of one calendar quarter with the standard library.

    Args:
        year: Year.
        quarter: Quarter number, 1 to 4.

    Returns:
        Number of days of the quarter.
    """
    months = range(3 * quarter - 2, 3 * quarter + 1)
    return sum(calendar.monthrange(year, month)[1] for month in months)


# =============================================================================
# Validation de la configuration
# =============================================================================
class TestValidation:
    """``scale_features`` validation, dict keys and mode resolution (§9.1)."""

    @pytest.mark.parametrize(
        'setting',
        [False, 'constant', 'calendar', {'m1': 'calendar'},
         {'m1': False, DEFAULT_SCALE_KEY: 'calendar'}],
        ids=['false', 'constant', 'calendar', 'dict', 'dict-default'],
    )
    def test_admissible_settings_are_stored_untouched(self, setting):
        """Every admissible setting is stored as received, same object."""
        scaler = StageScaler(scale_features=setting)
        assert scaler.scale_features is setting

    @pytest.mark.parametrize(
        'setting, message',
        [
            ('exact', r"scale_features must be one of"),
            (True, r"scale_features must be one of"),
            (1, r"scale_features must be one of"),
            (None, r"scale_features must be one of"),
            ({'m1': 'exact'}, r"scale_features values must be one of"),
            ({}, r"empty"),
        ],
        ids=['unknown-literal', 'true', 'one', 'none', 'dict-bad-value', 'empty-dict'],
    )
    def test_inadmissible_settings_raise(self, setting, message):
        """Anything else than False / 'constant' / 'calendar' (or their dict) raises."""
        with pytest.raises(ValueError, match=message):
            StageScaler(scale_features=setting)

    @pytest.mark.parametrize('form', ['scalar', 'dict'])
    @pytest.mark.parametrize('setting', [0, 0.0, np.False_], ids=['int-0', 'float-0', 'numpy-false'])
    def test_values_merely_equal_to_false_are_rejected(self, setting, form):
        """0, 0.0 and np.False_ are rejected: only False itself divides nothing (ANO-FREQ-034).

        The resolution reads ``mode is False``; a value merely equal to False
        would otherwise pass the validation and then act as 'constant'.
        """
        value = setting if form == 'scalar' else {'m1': setting}
        with pytest.raises(ValueError, match=r"scale_features"):
            StageScaler(scale_features=value)

    def test_validate_columns_lists_every_unknown_key(self):
        """Keys absent from the real columns are listed, sorted; known ones are not."""
        scaler = _make_scaler(scale_features={'zz': 'calendar', 'yy': False, 'm1': 'constant'})
        with pytest.raises(ValueError, match=r"unknown columns : \['yy', 'zz'\]"):
            scaler.validate_columns(['m1', 'q1'])

    def test_default_key_is_never_reported_as_unknown(self):
        """'__default__' is not a column name and raises nothing."""
        _make_scaler(scale_features={DEFAULT_SCALE_KEY: 'calendar'}).validate_columns(['m1'])

    @pytest.mark.parametrize('setting', [False, 'constant', 'calendar'], ids=str)
    def test_validate_columns_ignores_the_scalar_form(self, setting):
        """The scalar form names no column: nothing to check, whatever the columns."""
        StageScaler(scale_features=setting).validate_columns([])

    def test_dict_keys_with_special_characters(self):
        """Column names with spaces, accents and symbols are plain dict keys."""
        frame = pd.DataFrame({'m1': [1.0], 'q1': [2.0]}, index=_monthly_grid(periods=1))
        renamed, mapping = with_special_column_names(frame)
        scaler = StageScaler(scale_features={mapping['m1']: 'calendar', mapping['q1']: False})

        # Validation et résolution sur les noms spéciaux, sans normalisation
        scaler.validate_columns(renamed.columns)
        assert [scaler.resolve_mode(mapping[c]) for c in ('m1', 'q1')] == ['calendar', False]

    @pytest.mark.parametrize(
        'setting, column, expected',
        [
            ('calendar', 'm1', 'calendar'),
            ('calendar', None, 'calendar'),
            (False, 'm1', False),
            ({'m1': 'calendar', DEFAULT_SCALE_KEY: False}, 'm1', 'calendar'),
            ({'m1': 'calendar', DEFAULT_SCALE_KEY: False}, 'q1', False),
            ({'m1': 'calendar', DEFAULT_SCALE_KEY: False}, None, False),
            ({'m1': 'calendar'}, 'q1', DEFAULT_SCALE_MODE),
            ({'m1': 'calendar'}, None, DEFAULT_SCALE_MODE),
        ],
        ids=['scalar', 'scalar-global', 'false', 'dict-own-key', 'dict-default',
             'dict-global-default', 'dict-no-default', 'dict-global-no-default'],
    )
    def test_resolve_mode(self, setting, column, expected):
        """Own entry, then '__default__', then 'constant' — the scalar form for all."""
        assert StageScaler(scale_features=setting).resolve_mode(column) == expected

    def test_default_scale_mode_is_constant(self):
        """The documented default of the module and of __init__ is 'constant' (§9.1)."""
        assert DEFAULT_SCALE_MODE == 'constant'
        assert StageScaler().scale_features == 'constant'


class TestValidationThroughSetParams:
    """``set_params`` does not bypass the ``__init__`` validation (ANO-FREQ-032)."""

    @pytest.mark.parametrize(
        'entry_point', ['fit', 'target_divisor', 'fit_scale_factor', 'feature_divisors', 'resolve_mode']
    )
    def test_inadmissible_setting_raises_after_set_params(self, entry_point):
        """fit and the divisor methods usable without fit revalidate scale_features.

        Same arbitration as ANO-FREQ-023 (``AggregationConstraint``): the
        components are usable alone, ``set_params`` (``GridSearchCV``,
        pipelines) included.
        """
        scaler = _make_scaler().set_params(scale_features='exact')
        X = pd.DataFrame({'m1': [1.0]}, index=_monthly_grid(periods=1))
        calls = {
            'fit': lambda: scaler.fit(X),
            'target_divisor': lambda: scaler.target_divisor('a1', source_freq='Y', pred_freq='M'),
            'fit_scale_factor': lambda: scaler.fit_scale_factor('a1', source_freq='Y', pred_freq='M'),
            'feature_divisors': lambda: scaler.feature_divisors(
                columns=['m1'], column_frequencies={'m1': 'M'}, ways={}, grid_freq='Y', stage_freq='M'
            ),
            'resolve_mode': lambda: scaler.resolve_mode('m1'),
        }
        with pytest.raises(ValueError, match=r"scale_features"):
            calls[entry_point]()

    def test_admissible_setting_through_set_params(self):
        """A legitimate set_params keeps working and is read by the next call."""
        scaler = _make_scaler().set_params(scale_features=False)
        assert scaler.target_divisor('a1', source_freq='Y', pred_freq='M') == 1.0


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
    """'constant' : one invariant factor per (source, prediction) pair (§9.1)."""

    @pytest.mark.parametrize('position', ['', 'S', 'E'], ids=['base', 'start', 'end'])
    @pytest.mark.parametrize(
        'source, pred, expected', _CONSTANT_GOLD,
        ids=[f'{s}->{p}' for s, p, _ in _CONSTANT_GOLD],
    )
    def test_target_divisor_gold(self, source, pred, expected, position):
        """target_divisor counts the pred sub-periods of one source period, any position.

        A coarser stage than the variable (a monthly block at a quarterly
        stage) yields a fractional divisor, never floored at 1.0 (§5.8 R5).
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

        La cellule couvre une période de sa grille (voie 'identity') : une
        grille de bloc à la fréquence ``carried`` porte des cellules de cette
        période, plus grossière ou plus fine que l'étape (§5.8 R5).
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
    """'calendar' : real calendar count, one divisor per row (§9.1)."""

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

        §5.8 R5: the divisor is "> 1 for a block coarser than the stage and
        < 1 for a finer one", and §9.2 applies the mode to every divisor
        (ANO-FREQ-028). The shares of one stage period sum to 1 exactly.
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

        Grille de bloc mensuelle, étape trimestrielle : chaque cellule couvre
        un mois (§5.8 R5).
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

        Arbitration of ANO-FREQ-028: 'constant' keeps the uniform ratio
        (1/3 for every month), 'calendar' weighs each block period by its
        days (January 31/91, February 29/91 in 2024).
        """
        dates = pd.DatetimeIndex(starts if position == 'start' else ends)
        suffix = 'S' if position == 'start' else 'E'
        divisors = StageScaler(scale_features='calendar').target_divisor(
            'v', source_freq=source + suffix, pred_freq=pred + suffix, index=dates
        )
        np.testing.assert_allclose(divisors.to_numpy(), expected, rtol=_EXACT)

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
# Règle de forme
# =============================================================================
class TestShapeRule:
    """The return type follows the CONFIGURATION, never the values."""

    @pytest.mark.parametrize(
        'setting, source, pred, with_index, expected_type',
        [
            ('constant', 'Y', 'M', False, float),
            ('constant', 'Y', 'M', True, float),
            ('constant', 'M', 'M', True, float),
            (False, 'Y', 'M', True, float),
            ('calendar', 'Y', 'M', True, pd.Series),
            ('calendar', 'M', 'M', True, pd.Series),
        ],
        ids=['constant', 'constant-with-index', 'constant-all-ones', 'false',
             'calendar', 'calendar-all-ones'],
    )
    def test_target_divisor_shape(self, setting, source, pred, with_index, expected_type):
        """Scalar under 'constant' / False, Series under 'calendar' — even when every divisor is 1.0."""
        index = _monthly_grid(periods=3) if with_index else None
        divisor = StageScaler(scale_features=setting).target_divisor(
            'v', source_freq=source, pred_freq=pred, index=index
        )
        assert type(divisor) is expected_type

    @pytest.mark.parametrize(
        'setting, grid_freq, expected_type',
        [
            ('constant', 'Y', pd.Series),
            ('constant', 'M', pd.Series),
            (False, 'Y', pd.Series),
            ('calendar', 'M', pd.DataFrame),
            ({'m1': 'calendar', DEFAULT_SCALE_KEY: False}, 'M', pd.DataFrame),
        ],
        ids=['constant', 'constant-all-ones', 'false', 'calendar-all-ones',
             'one-calendar-column-among-false'],
    )
    def test_feature_divisors_shape(self, setting, grid_freq, expected_type):
        """Series per column unless one column is in 'calendar', 1.0 values included."""
        index = _monthly_grid(periods=3)
        divisors = StageScaler(scale_features=setting).feature_divisors(
            columns=['m1', 'q1'],
            column_frequencies={'m1': 'M', 'q1': 'Q'},
            ways={'m1': 'identity', 'q1': 'interpolate'},
            grid_freq=grid_freq,
            stage_freq='M',
            index=index,
        )
        assert type(divisors) is expected_type

    def test_calendar_divisors_of_one_are_not_collapsed(self):
        """'calendar' M -> M: a DataFrame of 1.0 aligned on the grid, never a scalar (B12)."""
        index = _monthly_grid(periods=3)
        divisors = StageScaler(scale_features='calendar').feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': 'M'},
            ways={'m1': 'identity'},
            grid_freq='M',
            stage_freq='M',
            index=index,
        )
        pd.testing.assert_frame_equal(divisors, pd.DataFrame({'m1': 1.0}, index=index))

    @pytest.mark.parametrize('produced', [['Y', 'Y'], ['Y', 'Q']], ids=['uniform', 'mixed'])
    def test_produced_freq_always_gives_a_series(self, produced):
        """With produced_freq the target divisor is a Series, even when every row agrees."""
        index = pd.to_datetime(['2021-12-31', '2022-12-31'])
        divisor = StageScaler().target_divisor(
            'a1', source_freq='Y', pred_freq='M',
            produced_freq=pd.Series(produced, index=index),
        )
        assert isinstance(divisor, pd.Series)

    def test_per_entity_source_freq_always_gives_a_series(self):
        """A per-entity source_freq gives a Series, even when every block agrees."""
        index = _panel_grid()
        divisor = StageScaler().target_divisor(
            'v', source_freq={('DE',): 'Y', ('FR',): 'Y'}, pred_freq='M', index=index
        )
        pd.testing.assert_series_equal(divisor, pd.Series(12.0, index=index))

    def test_disagreeing_blocks_give_a_dataframe(self):
        """A per-entity grid_freq whose blocks disagree gives a per-row DataFrame."""
        index = _panel_grid()
        divisors = StageScaler().feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': 'M'},
            ways={'m1': 'aggregate'},
            grid_freq={('DE',): 'Q', ('FR',): 'Y'},
            stage_freq='M',
            index=index,
        )
        assert isinstance(divisors, pd.DataFrame)

    @pytest.mark.parametrize(
        'grid_freq, entities',
        [({('DE',): 'Y', ('FR',): 'Y'}, ('DE', 'FR')), ({('IT',): 'M'}, ('IT',))],
        ids=['agreeing-blocks', 'single-entry'],
    )
    def test_per_entity_grid_freq_always_gives_a_dataframe(self, grid_freq, entities):
        """§9.2: a per-entity grid_freq returns a per-row DataFrame, whatever the divisors (ANO-FREQ-029)."""
        divisors = StageScaler().feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': 'M'},
            ways={'m1': 'aggregate'},
            grid_freq=grid_freq,
            stage_freq='Q',
            index=_panel_grid(entities),
        )
        assert isinstance(divisors, pd.DataFrame)

    def test_homogeneous_stage_binding_keeps_the_compact_form(self):
        """A stage binding whose entities agree on one frequency stays compact (ANO-FREQ-030).

        Arbitration: a stage / prediction binding is compact as long as its
        entities agree on the frequency — the configuration, not the
        divisors, decides. ``grid_freq`` / ``source_freq`` mappings describe
        a mutualized training set and are always per row.
        """
        index = _panel_grid()
        binding = {('DE',): 'M', ('FR',): 'M'}
        scaler = StageScaler()

        features = scaler.feature_divisors(
            columns=['m1'],
            column_frequencies={'m1': {('FR',): 'M', ('DE',): 'M'}},
            ways={'m1': 'aggregate'},
            grid_freq='Y',
            stage_freq=binding,
            index=index,
        )
        target = scaler.target_divisor('a1', source_freq='Y', pred_freq=binding, index=index)

        assert (type(features), features['m1'], type(target), target) == (pd.Series, 12.0, float, 12.0)

    def test_position_aliases_are_one_stage_frequency(self):
        """'M' for one entity and 'ME' for the other are one frequency: compact form."""
        index = _panel_grid()
        target = StageScaler().target_divisor(
            'a1', source_freq='Y', pred_freq={('DE',): 'M', ('FR',): 'ME'}, index=index
        )
        assert target == 12.0

    def test_disagreeing_stage_binding_is_per_row_even_with_equal_divisors(self):
        """Stage frequencies differing per entity give a DataFrame, even when every column is False.

        The divisors are all 1.0 and coincide, but the configuration varies
        per entity: the shape follows it, not the values.
        """
        divisors = StageScaler(scale_features=False).feature_divisors(
            columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
            grid_freq='Y', stage_freq={('DE',): 'Q', ('FR',): 'M'}, index=_panel_grid(),
        )
        pd.testing.assert_frame_equal(divisors, pd.DataFrame({'m1': 1.0}, index=_panel_grid()))

    def test_unread_column_binding_does_not_force_the_per_row_form(self):
        """A per-entity column frequency is not read by a materialized way: compact form.

        Only 'raw_anchors' reads the column's own frequency; aggregated onto
        the yearly grid, the column spans one year for every entity.
        """
        divisors = StageScaler().feature_divisors(
            columns=['ip'], column_frequencies={'ip': {('DE',): 'Q', ('FR',): 'M'}},
            ways={'ip': 'aggregate'}, grid_freq='Y', stage_freq='M', index=_panel_grid(),
        )
        assert (type(divisors), divisors['ip']) == (pd.Series, 12.0)


# =============================================================================
# Règle de la période couverte et replis
# =============================================================================
class TestCarriedPeriodRule:
    """A covariate divides by the period its cell SPANS on its grid (§9.2)."""

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

        Le matérialiseur ramène une telle colonne à l'identité ou à l'agrégation
        exacte (rang 1, ANO-FREQ-042) : sur une grille trimestrielle à l'étape
        Q, une colonne mensuelle ou trimestrielle divise par 1 ; seule la
        colonne annuelle, lue à ses ancres, couvre sa propre période (4).
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
        """I5 on TS at stage M: q1 aggregated yearly divides by 12, interpolated monthly by 1.

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


class TestFallbacks:
    """Incomparable frequencies: default divisor for a covariate, error for the target (§9.2)."""

    @pytest.mark.parametrize(
        'column_frequencies',
        [{}, {'x': 'zz'}],
        ids=['undetected', 'unknown-literal'],
    )
    def test_covariate_falls_back_on_default_divisor(self, column_frequencies):
        """An anchor column without a comparable frequency takes default_divisor."""
        divisors = StageScaler(default_divisor=12.0).feature_divisors(
            columns=['x'], column_frequencies=column_frequencies, ways={'x': 'raw_anchors'},
            grid_freq='M', stage_freq='M',
        )
        assert divisors['x'] == 12.0

    def test_default_argument_overrides_default_divisor(self):
        """The ``default`` argument of the call takes precedence over the constructor's."""
        divisors = StageScaler(default_divisor=12.0).feature_divisors(
            columns=['x'], column_frequencies={}, ways={'x': 'raw_anchors'},
            grid_freq='M', stage_freq='M', default=4.0,
        )
        assert divisors['x'] == 4.0

    @pytest.mark.parametrize('source_freq', ['zz', None], ids=['unknown-literal', 'none'])
    def test_target_with_incomparable_frequency_raises(self, source_freq):
        """The target's scale silently governs the model: never a fallback, always an error."""
        with pytest.raises(ValueError):
            StageScaler().target_divisor('v', source_freq=source_freq, pred_freq='M')


class TestScaleFeaturesFalse:
    """``False`` divides nothing: neither the features, nor the target (D15)."""

    def test_false_spares_features_target_and_baked_factor(self):
        """The target is a column like any other: False leaves it at its own scale."""
        scaler = _make_scaler(scale_features=False)
        divisors = scaler.feature_divisors(
            columns=['m1', 'q1', 'd1'],
            column_frequencies={'m1': 'M', 'q1': 'Q', 'd1': 'D'},
            ways={'m1': 'aggregate', 'q1': 'aggregate', 'd1': 'aggregate'},
            grid_freq='Y',
            stage_freq='M',
        )
        assert (
            divisors.tolist(),
            scaler.target_divisor('a1', source_freq='Y', pred_freq='M'),
            scaler.fit_scale_factor('a1', source_freq='Y', pred_freq='M'),
        ) == ([1.0, 1.0, 1.0], 1.0, 1.0)

    def test_false_on_y_ignores_produced_frequencies(self):
        """Even with mixed production frequencies, False divides no row."""
        index = pd.to_datetime(['2021-12-31', '2021-03-31'])
        divisor = _make_scaler(scale_features=False).target_divisor(
            'a1', source_freq='Y', pred_freq='M',
            produced_freq=pd.Series(['Y', 'Q'], index=index),
        )
        assert divisor == 1.0

    def test_false_on_y_only_through_the_dict_form(self):
        """I5: a dict may spare the target alone, the features staying divided."""
        scaler = _make_scaler(scale_features={'a1': False, DEFAULT_SCALE_KEY: 'constant'})
        divisors = scaler.feature_divisors(
            columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
            grid_freq='Y', stage_freq='M',
        )
        assert (divisors['m1'], scaler.target_divisor('a1', 'Y', 'M')) == (12.0, 1.0)

    def test_calendar_feature_with_constant_target(self):
        """I5: 'calendar' on a feature, 'constant' on y — each divisor follows its own mode."""
        scaler = _make_scaler(scale_features={'d1': 'calendar', DEFAULT_SCALE_KEY: 'constant'})
        index = pd.date_range('2024-01-31', periods=2, freq='ME')
        divisors = scaler.feature_divisors(
            columns=['d1'], column_frequencies={'d1': 'D'}, ways={'d1': 'aggregate'},
            grid_freq='M', stage_freq='D', index=index,
        )
        target = scaler.target_divisor('a1', source_freq='M', pred_freq='D', index=index)
        assert (divisors['d1'].tolist(), target) == ([31.0, 29.0], 30.0)

    def test_false_in_dict_form_only_spares_that_feature(self):
        """False in a dict only neutralizes the targeted column."""
        divisors = _make_scaler(
            scale_features={'m1': False, DEFAULT_SCALE_KEY: 'constant'}
        ).feature_divisors(
            columns=['m1', 'q1'],
            column_frequencies={'m1': 'M', 'q1': 'Q'},
            ways={'m1': 'aggregate', 'q1': 'aggregate'},
            grid_freq='Y',
            stage_freq='M',
        )
        assert divisors.to_dict() == {'m1': 1.0, 'q1': 12.0}


# =============================================================================
# Échelle par ligne de la cible (§5.4)
# =============================================================================
class TestPerRowTargetDivisor:
    """§5.4 per-row scale: each row of y carries its production frequency."""

    def test_spec_table_of_5_4(self):
        """§5.4 table: 120 / Y -> 10.0, 28 / Q -> 9.33, 30 / Q -> 10.0."""
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

        Moved from the ``HighFrequencyImputer`` test package (prompt F10): a
        pure ``StageScaler`` case.
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


# =============================================================================
# Application, inversion, B12
# =============================================================================
class TestApplyInvert:
    """Single scaling routine, and the B12 short-circuit rule."""

    def test_scalar_one_short_circuits(self):
        """Only a scalar strictly equal to 1.0 returns the input object itself."""
        values = pd.Series([10.0, 20.0], index=_monthly_grid(periods=2))
        scaler = StageScaler()
        assert scaler.apply(values, 1.0) is values and scaler.invert(values, 1.0) is values

    @pytest.mark.parametrize('kind', ['series', 'frame'])
    def test_b12_divisors_of_ones_go_through_the_scaling_path(self, kind):
        """A Series / DataFrame of 1.0 is NOT short-circuited: a new, equal object comes back (B12)."""
        index = _monthly_grid(periods=3)
        values = pd.DataFrame({'a': [10.0, 20.0, 30.0]}, index=index)
        ones = pd.DataFrame({'a': 1.0}, index=index)
        if kind == 'series':
            values, ones = values['a'], ones['a']

        scaled = StageScaler().apply(values, ones)

        assert scaled is not values and scaled.equals(values)

    def test_b12_through_transform(self):
        """Public path: a fitted 'calendar' M -> M stage divides by a Series of 1.0, never skipped."""
        X = pd.DataFrame({'m1': [1.0, 2.0, 3.0]}, index=_monthly_grid(periods=3))
        y = pd.Series([1.0, 2.0, 3.0], index=X.index)
        scaler = StageScaler(
            scale_features='calendar', source_freq='M', pred_freq='M', column_frequencies={'m1': 'M'}
        ).fit(X, y)

        scaled = scaler.transform(y)

        assert scaled is not y and scaled.equals(y)

    def test_heterogeneous_column_divisors(self):
        """A 1.0 divisor among others leaves the other columns divided."""
        frame = pd.DataFrame({'a': [12.0, 24.0], 'b': [3.0, 6.0]}, index=_monthly_grid(periods=2))
        scaled = StageScaler().apply(frame, pd.Series({'a': 1.0, 'b': 3.0}))
        assert scaled.to_dict('list') == {'a': [12.0, 24.0], 'b': [1.0, 2.0]}

    @pytest.mark.parametrize('divisor_kind', ['scalar', 'per-row', 'per-column'])
    def test_apply_invert_roundtrip(self, divisor_kind):
        """invert(apply(v, d), d) == v for every divisor form."""
        index = _monthly_grid(periods=4)
        values = pd.DataFrame({'a': [120.0, 132.0, 150.0, 90.0], 'b': [1.0, -2.0, 3.5, np.nan]}, index=index)
        divisor = {
            'scalar': 12.0,
            'per-row': pd.DataFrame({'a': [12.0, 3.0, 4.0, 1.0], 'b': [1 / 3, 3.0, 4.0, 1.0]}, index=index),
            'per-column': pd.Series({'a': 12.0, 'b': 1 / 3}),
        }[divisor_kind]
        scaler = StageScaler()
        pd.testing.assert_frame_equal(
            scaler.invert(scaler.apply(values, divisor), divisor), values, rtol=_EXACT
        )


# =============================================================================
# Symétrie transform / inverse_transform
# =============================================================================
# Configurations de symétrie : (paramètres du scaler, fabrique de (X, y, produced))
def _symmetry_case(case: str):
    """Build one (scaler, X, y, produced_freq) configuration of the symmetry tests.

    Args:
        case: Configuration identifier.

    Returns:
        Tuple ``(scaler, X, y, produced_freq)``; values drawn from a seeded
        generator, negative values and NaN included.
    """
    rng = np.random.default_rng(42)
    produced = None
    if case == 'panel-fractional':
        # Panel PANEL-F réduit : blocs Y / Q / M, étape Q, diviseur 1/3 pour IT
        index = _panel_grid(('DE', 'FR', 'IT'), periods=6)
        scaler = StageScaler(
            source_freq=_PANEL_F_BLOCKS, pred_freq='Q', column_frequencies={'m1': 'M', 'q1': 'Q'},
            materialization={'q1': 'raw_anchors'},
        )
    elif case == 'calendar-daily':
        # Étape journalière du premier trimestre 2024 : diviseurs par ligne
        index = pd.date_range('2024-01-01', '2024-03-31', freq='D')
        scaler = StageScaler(
            scale_features='calendar', source_freq='Q', pred_freq='D',
            column_frequencies={'m1': 'M', 'q1': 'Q'}, materialization={'m1': 'raw_anchors', 'q1': 'raw_anchors'},
        )
    else:
        index = _monthly_grid(periods=12)
        settings = {
            'constant': 'constant',
            'dict-mixed': {'m1': 'calendar', 'q1': False, DEFAULT_SCALE_KEY: 'constant'},
            'produced-freq': 'constant',
        }[case]
        scaler = StageScaler(
            scale_features=settings, source_freq='Y', pred_freq='M',
            column_frequencies={'m1': 'M', 'q1': 'Q'}, materialization={'q1': 'raw_anchors'},
        )
        if case == 'produced-freq':
            produced = pd.Series(['Y', 'Q', 'M'] * 4, index=index)

    X = pd.DataFrame(
        {'m1': rng.normal(0.0, 50.0, len(index)), 'q1': rng.normal(100.0, 30.0, len(index))},
        index=index,
    )
    X.iloc[::5, 0] = np.nan
    y = pd.Series(rng.normal(120.0, 40.0, len(index)), index=index, name='v')
    y.iloc[1::4] = np.nan
    return scaler, X, y, produced


_SYMMETRY_CASES = ['constant', 'dict-mixed', 'calendar-daily', 'panel-fractional', 'produced-freq']


class TestTransformSymmetry:
    """``inverse_transform ∘ transform`` is the identity, on features and target."""

    @pytest.mark.parametrize('case', _SYMMETRY_CASES)
    def test_roundtrip_on_features(self, case):
        """inverse_transform(transform(X)) == X to 1e-12, NaN kept in place."""
        scaler, X, y, produced = _symmetry_case(case)
        scaler.fit(X, y, produced_freq=produced)
        pd.testing.assert_frame_equal(
            scaler.inverse_transform(scaler.transform(X)), X, rtol=_EXACT, atol=0
        )

    @pytest.mark.parametrize('case', _SYMMETRY_CASES)
    def test_roundtrip_on_target(self, case):
        """inverse_transform(transform(y)) == y to 1e-12, NaN kept in place."""
        scaler, X, y, produced = _symmetry_case(case)
        scaler.fit(X, y, produced_freq=produced)
        pd.testing.assert_series_equal(
            scaler.inverse_transform(scaler.transform(y)), y, rtol=_EXACT, atol=0
        )

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    @pytest.mark.parametrize('case', ['calendar-daily', 'panel-fractional', 'produced-freq'])
    def test_name_of_y_is_kept_under_per_row_divisors(self, case, method):
        """The name of y designates the imputed column: scaling never erases it (ANO-FREQ-036).

        pandas drops the name of a Series divided by an unnamed per-row
        divisor; the scaler restores it.
        """
        scaler, X, y, produced = _symmetry_case(case)
        scaler.fit(X, y, produced_freq=produced)
        assert getattr(scaler, method)(y).name == 'v'

    def test_name_of_y_is_kept_under_a_scalar_divisor(self):
        """Contrast: under a scalar target divisor the name of y is kept."""
        scaler, X, y, produced = _symmetry_case('constant')
        scaler.fit(X, y, produced_freq=produced)
        assert (scaler.transform(y).name, scaler.inverse_transform(y).name) == ('v', 'v')

    @pytest.mark.parametrize('case', _SYMMETRY_CASES)
    def test_transform_introduces_no_nan(self, case):
        """Scaling never creates nor removes a missing value."""
        scaler, X, y, produced = _symmetry_case(case)
        scaler.fit(X, y, produced_freq=produced)
        assert scaler.transform(X).isna().equals(X.isna())

    def test_transform_tells_features_and_target_apart_by_type(self):
        """The same values divide by the covariate divisor as a DataFrame, by the target's as a Series."""
        index = _monthly_grid(periods=3)
        values = [120.0, 132.0, 150.0]
        X = pd.DataFrame({'m1': values}, index=index)
        y = pd.Series(values, index=index, name='m1')
        # 'm1' identité sur la grille d'étape : diviseur 1 ; cible Y -> M : diviseur 12
        scaler = _make_scaler(materialization={'m1': 'identity'}).fit(X, y)
        assert (scaler.transform(X)['m1'].tolist(), scaler.transform(y).tolist()) == (
            values, [10.0, 11.0, 12.5]
        )

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_other_types_are_rejected(self, method):
        """Neither a Series nor a DataFrame: TypeError naming the type."""
        X = pd.DataFrame({'m1': [1.0, 2.0]}, index=_monthly_grid(periods=2))
        scaler = _make_scaler().fit(X)
        with pytest.raises(TypeError, match=r"ndarray"):
            getattr(scaler, method)(np.arange(2.0))

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_transform_rejects_other_columns_than_fit(self, method):
        """sklearn convention (as ANO-FREQ-026): missing and unexpected columns are named (ANO-FREQ-033)."""
        index = _monthly_grid(periods=3)
        X = pd.DataFrame({'m1': [1.0, 2.0, 3.0], 'a2': [12.0, 24.0, 36.0]}, index=index)
        scaler = _make_scaler(materialization={'a2': 'raw_anchors'}).fit(X)
        with pytest.raises(ValueError, match=r"missing \['a2'\], unexpected \['q1'\]"):
            getattr(scaler, method)(X.rename(columns={'a2': 'q1'}))

    def test_transform_accepts_reordered_columns(self):
        """The column order is free: each column keeps its own divisor."""
        index = _monthly_grid(periods=3)
        X = pd.DataFrame({'m1': [1.0, 2.0, 3.0], 'a2': [12.0, 24.0, 36.0]}, index=index)
        scaler = _make_scaler(materialization={'a2': 'raw_anchors'}).fit(X)
        assert scaler.transform(X[['a2', 'm1']]).to_dict('list') == {'a2': [1.0, 2.0, 3.0], 'm1': [1.0, 2.0, 3.0]}


# =============================================================================
# Fréquences par entité
# =============================================================================
class TestPerEntityFrequencies:
    """Per-entity bindings spread row by row; uncovered entities never guessed."""

    def test_heterogeneous_anchor_frequencies(self):
        """Read at its anchors, a column monthly for FR and quarterly for DE spreads per row."""
        index = _panel_grid()
        divisors = StageScaler().feature_divisors(
            columns=['ip'],
            column_frequencies={'ip': {('FR',): 'M', ('DE',): 'Q'}},
            ways={'ip': 'raw_anchors'},
            grid_freq={('FR',): 'M', ('DE',): 'M'},
            stage_freq={('FR',): 'M', ('DE',): 'M'},
            index=index,
        )
        # Valeur d'or : DE, ancre trimestrielle -> factor(M, Q) = 3 ; FR, ancre mensuelle -> 1
        assert divisors['ip'].tolist() == [3.0] * 3 + [1.0] * 3

    def test_heterogeneous_stage_frequencies_give_per_row_target(self):
        """Disagreeing stage frequencies produce a per-row target divisor."""
        index = _panel_grid()
        divisors = StageScaler().target_divisor(
            'a1', source_freq='Y', pred_freq={('FR',): 'M', ('DE',): 'Q'}, index=index
        )
        # Valeur d'or : DE à l'étape Q -> 4, FR à l'étape M -> 12
        assert divisors.tolist() == [4.0] * 3 + [12.0] * 3

    def test_heterogeneous_stage_frequencies_give_per_row_features(self):
        """Disagreeing stage frequencies spread the covariate divisors per row too."""
        index = _panel_grid()
        divisors = StageScaler().feature_divisors(
            columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
            grid_freq='Y', stage_freq={('FR',): 'M', ('DE',): 'Q'}, index=index,
        )
        assert divisors['m1'].tolist() == [4.0] * 3 + [12.0] * 3

    def test_binding_entity_absent_from_the_grid_is_ignored(self):
        """An entity named by the binding but absent from the grid writes no row."""
        index = _panel_grid()
        divisors = StageScaler().target_divisor(
            'a1', source_freq='Y', pred_freq={('FR',): 'M', ('DE',): 'Q', ('IT',): 'M'}, index=index
        )
        pd.testing.assert_index_equal(divisors.index, index)

    @pytest.mark.parametrize('method', ['feature_divisors', 'target_divisor'])
    def test_grid_entity_uncovered_by_the_stage_binding_raises(self, method):
        """The scale of a block is never guessed: an uncovered grid entity is an error naming it (ANO-FREQ-031)."""
        index = _panel_grid(('DE', 'FR', 'IT'))
        binding = {('FR',): 'M', ('DE',): 'Q'}
        scaler = StageScaler()
        with pytest.raises(ValueError, match=r"IT"):
            if method == 'feature_divisors':
                scaler.feature_divisors(
                    columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
                    grid_freq='Y', stage_freq=binding, index=index,
                )
            else:
                scaler.target_divisor('a1', source_freq='Y', pred_freq=binding, index=index)

    def test_uncovered_grid_entity_raises_with_production_frequencies(self):
        """Disagreeing stage binding and produced_freq: the uncovered IT rows raise, never NaN."""
        index = _panel_grid(('DE', 'FR', 'IT'))
        with pytest.raises(ValueError, match=r"IT"):
            StageScaler().target_divisor(
                'a1', source_freq='Y', pred_freq={('FR',): 'M', ('DE',): 'Q'},
                produced_freq=pd.Series('Y', index=index),
            )

    @pytest.mark.parametrize('route', ['target', 'produced-freq', 'features'])
    def test_unanimous_stage_binding_covers_the_whole_panel(self, route):
        """An agreeing stage binding is the stage frequency: an entity it omits is scaled to it.

        Mutualized training set: DE skips the monthly stage (quarterly
        target) but contributes its block; its rows come to the monthly
        scale like the others — factor(M, Y) = 12 — instead of a NaN that
        would drop them from training.
        """
        index = _panel_grid(('DE', 'FR', 'IT'))
        binding = {('FR',): 'M', ('IT',): 'M'}
        scaler = StageScaler()
        if route == 'target':
            divisors = scaler.target_divisor('a1', source_freq='Y', pred_freq=binding, index=index)
            values = np.full(len(index), float(divisors))
        elif route == 'produced-freq':
            values = scaler.target_divisor(
                'a1', source_freq='Y', pred_freq=binding, produced_freq=pd.Series('Y', index=index)
            ).to_numpy()
        else:
            divisors = scaler.feature_divisors(
                columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
                grid_freq={('DE',): 'Y', ('FR',): 'Y', ('IT',): 'Y'}, stage_freq=binding, index=index,
            )
            values = divisors['m1'].to_numpy()
        np.testing.assert_array_equal(values, 12.0)

    @pytest.mark.parametrize(
        'parameter, expected',
        [
            # DE trimestrielle lue à ses ancres (3), FR mensuelle (1)
            ('column_frequencies', [3.0] * 3 + [1.0] * 3),
            # Blocs DE trimestriel (3) et FR annuel (12), étape M
            ('source_freq', [3.0] * 3 + [12.0] * 3),
            # Cible annuelle, étapes DE trimestrielle (4) et FR mensuelle (12)
            ('pred_freq', [4.0] * 3 + [12.0] * 3),
        ],
    )
    @pytest.mark.parametrize('form', ['plain', 'tuple'])
    def test_plain_entity_keys_behave_like_tuple_keys(self, parameter, expected, form):
        """``{'FR': ...}`` and ``{('FR',): ...}`` are the same binding, in every parameter.

        Clés scalaires normalisées, comme dans ``CovariateMaterializer``
        (ANO-FREQ-037) : les deux composants lisent les mêmes liaisons.
        """
        index = _panel_grid()
        fr, de = ('FR', 'DE') if form == 'plain' else (('FR',), ('DE',))
        scaler = StageScaler()
        if parameter == 'column_frequencies':
            divisors = _feature_rows(
                scaler.feature_divisors(
                    columns=['ip'], column_frequencies={'ip': {fr: 'M', de: 'Q'}},
                    ways={'ip': 'raw_anchors'}, grid_freq='M', stage_freq='M', index=index,
                ),
                'ip', index,
            )
        elif parameter == 'source_freq':
            divisors = scaler.target_divisor('v', {fr: 'Y', de: 'Q'}, 'M', index=index).to_numpy()
        else:
            divisors = scaler.target_divisor('v', 'Y', {fr: 'M', de: 'Q'}, index=index).to_numpy()
        np.testing.assert_array_equal(divisors, expected)

    def test_uncovered_entity_message_lists_normalized_keys(self):
        """An entity a plain-key binding does not cover is named, the keys normalized."""
        index = _panel_grid(('DE', 'FR', 'IT'))
        with pytest.raises(ValueError, match=r"entity \('IT',\).*got keys \[\('DE',\), \('FR',\)\]"):
            StageScaler().target_divisor('v', {'FR': 'Y', 'DE': 'Q'}, 'M', index=index)

    def test_grid_entity_uncovered_by_a_column_binding_takes_the_default(self):
        """An anchor column without frequency for one entity falls back on the default divisor, never NaN."""
        divisors = StageScaler(default_divisor=7.0).feature_divisors(
            columns=['ip'], column_frequencies={'ip': {('FR',): 'Q', ('DE',): 'Q'}},
            ways={'ip': 'raw_anchors'}, grid_freq='M', stage_freq='M', index=_panel_grid(('DE', 'FR', 'IT')),
        )
        assert divisors['ip'].tolist() == [3.0] * 6 + [7.0] * 3

    @pytest.mark.parametrize('method', ['feature_divisors', 'target_divisor'])
    def test_disagreeing_stage_binding_without_index_raises(self, method):
        """Per-row divisors without a grid: explicit ValueError."""
        binding = {('FR',): 'M', ('DE',): 'Q'}
        scaler = StageScaler()
        with pytest.raises(ValueError, match=r"index"):
            if method == 'feature_divisors':
                scaler.feature_divisors(
                    columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
                    grid_freq='Y', stage_freq=binding,
                )
            else:
                scaler.target_divisor('a1', source_freq='Y', pred_freq=binding)

    def test_disagreeing_entities_on_a_flat_index_raise(self):
        """Several entities to spread over a grid without entity level: explicit ValueError."""
        with pytest.raises(ValueError, match=r"MultiIndex"):
            StageScaler().feature_divisors(
                columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
                grid_freq='Y', stage_freq={('FR',): 'M', ('DE',): 'Q'}, index=_monthly_grid(periods=3),
            )

    def test_three_level_index_with_two_level_entity_keys(self):
        """(region, country, date): entity keys are 2-tuples, the date the last level."""
        frame = pd.DataFrame({'v': 0.0}, index=_panel_grid())
        index = to_three_level_index(frame).index
        divisors = StageScaler().target_divisor(
            'v', source_freq={('Zone euro', 'DE'): 'Q', ('Zone euro', 'FR'): 'Y'}, pred_freq='M',
            index=index,
        )
        assert divisors.tolist() == [3.0] * 3 + [12.0] * 3

    @pytest.mark.parametrize('method', ['feature_divisors', 'target_divisor'])
    def test_per_entity_binding_requires_an_entity_level(self, method):
        """A per-entity grid / source binding needs a MultiIndex — none, or a flat one, raises."""
        scaler = StageScaler()
        for index in (None, _monthly_grid()):
            with pytest.raises(ValueError, match=r"grid_freq|source_freq"):
                if method == 'feature_divisors':
                    scaler.feature_divisors(
                        columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
                        grid_freq=_PANEL_F_BLOCKS, stage_freq='M', index=index,
                    )
                else:
                    scaler.target_divisor('v', source_freq=_PANEL_F_BLOCKS, pred_freq='M', index=index)


# =============================================================================
# Jeu mutualisé PANEL-F (§5.8 R5)
# =============================================================================
class TestMutualizedTrainingSet:
    """§5.8 R5: per-entity source_freq / grid_freq, one block frequency per entity."""

    @pytest.mark.parametrize(
        'stage, expected',
        [
            # Valeur d'or du tableau de l'étape Q du §5.8 : FR / 4, DE / 1, IT / (1/3)
            ('Q', {'FR': [30.0, 33.0, 37.5], 'DE': [28, 30, 31, 31, 31, 33, 34, 34, 36, 37, 38, 39],
                   'IT': [30.0] * 12 + [33.0] * 12 + [37.5] * 12}),
            # Étape M : FR / 12, DE / 3, IT / 1
            ('M', {'FR': [10.0, 11.0, 12.5],
                   'DE': [v / 3.0 for v in (28, 30, 31, 31, 31, 33, 34, 34, 36, 37, 38, 39)],
                   'IT': [10.0] * 12 + [11.0] * 12 + [12.5] * 12}),
        ],
        ids=['stage-Q', 'stage-M'],
    )
    def test_panel_f_scaled_target_gold(self, mixed_freq_panel_multifrequency, stage, expected):
        """The 51 observed rows of v come back to the stage scale, block by block.

        FR 3 yearly anchors, DE 12 quarterly ones, IT 36 monthly values: the
        monthly block finer than the quarterly stage divides by 1/3.
        """
        y = mixed_freq_panel_multifrequency['v'].dropna()
        X = mixed_freq_panel_multifrequency.loc[y.index, ['m1', 'q1']]
        scaler = StageScaler(
            source_freq=_PANEL_F_BLOCKS, pred_freq=stage, column_frequencies={'m1': 'M', 'q1': 'Q'}
        ).fit(X, y)

        scaled = scaler.transform(y)

        for entity in ('FR', 'DE', 'IT'):
            np.testing.assert_allclose(
                scaled.loc[entity].to_numpy(), expected[entity], rtol=_EXACT, err_msg=entity
            )

    def test_feature_divisors_per_entity_matches_scalar_calls(self, mixed_freq_panel_multifrequency):
        """The per-entity form is an assembly: block by block, it equals the scalar call."""
        index = mixed_freq_panel_multifrequency.index
        columns = ['m1', 'q1']
        kwargs = dict(
            columns=columns, column_frequencies={'m1': 'M', 'q1': 'Q'},
            ways={'m1': 'aggregate', 'q1': 'aggregate'}, stage_freq='M',
        )
        scaler = StageScaler()
        mutualized = scaler.feature_divisors(grid_freq=_PANEL_F_BLOCKS, index=index, **kwargs)

        for (entity,), f_block in _PANEL_F_BLOCKS.items():
            rows = _rows_of(index, entity)
            scalar = scaler.feature_divisors(grid_freq=f_block, index=index[rows], **kwargs)
            for column in columns:
                np.testing.assert_array_equal(
                    _feature_rows(mutualized, column, index)[rows],
                    _feature_rows(scalar, column, index[rows]),
                )

    def test_feature_divisors_per_entity_gold(self, mixed_freq_panel_multifrequency):
        """m1 divides by 12 for FR (block Y), 3 for DE (block Q), 1 for IT (block M)."""
        index = mixed_freq_panel_multifrequency.index
        divisors = StageScaler().feature_divisors(
            columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
            grid_freq=_PANEL_F_BLOCKS, stage_freq='M', index=index,
        )
        per_entity = divisors['m1'].groupby(level='country').unique().map(list).to_dict()
        assert per_entity == {'DE': [3.0], 'FR': [12.0], 'IT': [1.0]}

    def test_fractional_divisor_for_a_block_finer_than_the_stage(self, mixed_freq_panel_multifrequency):
        """A monthly block at a quarterly stage (D18) divides by 1/3, by every route.

        Block divisor, production frequency (§5.4) and covariate finer than
        the block all give 1/3 — never floored at 1.0.
        """
        index = mixed_freq_panel_multifrequency.index
        it_rows = index[_rows_of(index, 'IT')]
        scaler = StageScaler()

        by_block = scaler.target_divisor('v', source_freq={('IT',): 'M'}, pred_freq='Q', index=it_rows)
        by_production = scaler.target_divisor(
            'v', source_freq='Q', pred_freq='Q', produced_freq=pd.Series('M', index=it_rows)
        )
        features = scaler.feature_divisors(
            columns=['d1'], column_frequencies={'d1': 'D'}, ways={'d1': 'aggregate'},
            grid_freq={('IT',): 'M'}, stage_freq='Q', index=it_rows,
        )

        np.testing.assert_allclose(
            [by_block.to_numpy(), by_production.to_numpy(), _feature_rows(features, 'd1', it_rows)],
            1.0 / 3.0, rtol=_EXACT,
        )

    def test_scalar_grid_freq_keeps_the_compact_form(self, mixed_freq_panel_multifrequency):
        """The scalar form keeps its Series per column, with or without index."""
        index = mixed_freq_panel_multifrequency.index
        kwargs = dict(
            columns=['m1', 'q1'], column_frequencies={'m1': 'M', 'q1': 'Q'},
            ways={'m1': 'aggregate', 'q1': 'aggregate'}, grid_freq='Y', stage_freq='M',
        )
        scaler = StageScaler()
        with_index = scaler.feature_divisors(index=index, **kwargs)
        pd.testing.assert_series_equal(with_index, scaler.feature_divisors(**kwargs))
        assert with_index.to_dict() == {'m1': 12.0, 'q1': 12.0}

    def test_unknown_entity_in_mapping_raises(self, mixed_freq_panel_multifrequency):
        """A grid entity absent from a multi-entry block mapping is an error naming it."""
        index = mixed_freq_panel_multifrequency.index
        partial = {('FR',): 'Y', ('DE',): 'Q'}
        with pytest.raises(ValueError, match=r"IT"):
            StageScaler().feature_divisors(
                columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
                grid_freq=partial, stage_freq='M', index=index,
            )
        with pytest.raises(ValueError, match=r"IT"):
            StageScaler().target_divisor('v', source_freq=partial, pred_freq='M', index=index)

    def test_single_entry_mapping_covers_the_whole_panel(self, mixed_freq_panel_multifrequency):
        """Documented fallback: a single-entry binding stands for every entity."""
        index = mixed_freq_panel_multifrequency.index
        divisors = StageScaler().feature_divisors(
            columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
            grid_freq={('FR',): 'Y'}, stage_freq='M', index=index,
        )
        np.testing.assert_array_equal(_feature_rows(divisors, 'm1', index), 12.0)

    def test_produced_freq_takes_precedence_over_source_freq(self, mixed_freq_panel_multifrequency):
        """With produced_freq, source_freq is not read at all, whatever its form."""
        index = mixed_freq_panel_multifrequency.index
        produced = pd.Series('Q', index=index)
        scaler = StageScaler()
        reference = scaler.target_divisor(
            'v', source_freq='Y', pred_freq='M', index=index, produced_freq=produced
        )
        for source_freq in ('M', _PANEL_F_BLOCKS, {('FR',): 'Y', ('DE',): 'Q'}):
            pd.testing.assert_series_equal(
                scaler.target_divisor(
                    'v', source_freq=source_freq, pred_freq='M', index=index, produced_freq=produced
                ),
                reference,
            )
        assert reference.unique().tolist() == [3.0]

    def test_calendar_mode_with_per_entity_blocks(self, mixed_freq_panel_multifrequency):
        """'calendar' combines with the per-entity form: per row on both axes.

        FR yearly blocks over 2021-2023 (no leap year): 365 days; DE monthly
        blocks: the days of each month.
        """
        index = mixed_freq_panel_multifrequency.index
        blocks = {('FR',): 'Y', ('DE',): 'M', ('IT',): 'M'}
        scaler = StageScaler(scale_features='calendar')
        features = scaler.feature_divisors(
            columns=['d1'], column_frequencies={'d1': 'D'}, ways={'d1': 'aggregate'},
            grid_freq=blocks, stage_freq='D', index=index,
        )
        target = scaler.target_divisor('v', source_freq=blocks, pred_freq='D', index=index)

        # Valeur d'or : jours réels de chaque mois par la bibliothèque standard
        de_dates = index[_rows_of(index, 'DE')].get_level_values('date')
        de_days = [calendar.monthrange(d.year, d.month)[1] for d in de_dates]
        for divisors in (features['d1'], target):
            assert divisors.loc['FR'].unique().tolist() == [365.0]
            assert divisors.loc['DE'].tolist() == de_days


# =============================================================================
# Jeux réalistes du notebook 3
# =============================================================================
class TestRealisticDatasets:
    """Realistic datasets: properties and stdlib-derived golden values."""

    def test_depenses_publiques_per_entity_divisors(self, heterogeneous_coverage_panel):
        """depenses_publiques_pib at stage M: 12 for France / Italie (Y), 3 for Allemagne (Q)."""
        y = heterogeneous_coverage_panel['depenses_publiques_pib'].dropna()
        # Prérequis : 6 ans français, 5 italiens, 23 trimestres allemands
        assert y.groupby(level='country').size().to_dict() == {'Allemagne': 23, 'France': 6, 'Italie': 5}

        divisors = StageScaler().target_divisor(
            'depenses_publiques_pib', source_freq=_DEPENSES_BLOCKS, pred_freq='M', index=y.index
        )
        expected = np.where(y.index.get_level_values('country') == 'Allemagne', 3.0, 12.0)
        np.testing.assert_array_equal(divisors.to_numpy(), expected)

    def test_depenses_publiques_calendar_days(self, heterogeneous_coverage_panel):
        """At a daily stage under 'calendar', each anchor divides by the days of its own period.

        Dates at period START (MS): year of 365 / 366 days for France /
        Italie, quarter of 90 to 92 days for Allemagne — counted with the
        standard library.
        """
        y = heterogeneous_coverage_panel['depenses_publiques_pib'].dropna()
        divisors = StageScaler(scale_features='calendar').target_divisor(
            'depenses_publiques_pib', source_freq=_DEPENSES_BLOCKS, pred_freq='D', index=y.index
        )
        expected = [
            float(_days_in_quarter(d.year, (d.month - 1) // 3 + 1)) if country == 'Allemagne'
            else (366.0 if calendar.isleap(d.year) else 365.0)
            for country, d in y.index
        ]
        assert divisors.tolist() == expected

    def test_fit_transform_roundtrip_on_the_whole_panel(self, heterogeneous_coverage_panel):
        """Whole irregular panel: NaN pattern kept, index restored, roundtrip exact."""
        X = heterogeneous_coverage_panel.drop(columns='depenses_publiques_pib')
        y = heterogeneous_coverage_panel['depenses_publiques_pib']
        scaler = StageScaler(
            source_freq=_DEPENSES_BLOCKS, pred_freq='M',
            column_frequencies={
                'production_industrielle': 'M', 'inflation_ipc': 'M', 'taux_chomage': 'M',
                'pib_trimestriel': 'Q', 'balance_commerciale_annuelle': 'Y', 'climat_affaires': 'M',
            },
            materialization={'pib_trimestriel': 'raw_anchors', 'balance_commerciale_annuelle': 'raw_anchors'},
        ).fit(X, y)

        scaled = scaler.transform(X)

        assert scaler.feature_divisors_.to_dict() == {
            'production_industrielle': 1.0, 'inflation_ipc': 1.0, 'taux_chomage': 1.0,
            'pib_trimestriel': 3.0, 'balance_commerciale_annuelle': 12.0, 'climat_affaires': 1.0,
        }
        assert scaled.isna().equals(X.isna())
        pd.testing.assert_frame_equal(scaler.inverse_transform(scaled), X, rtol=_EXACT, atol=0)
        pd.testing.assert_series_equal(
            scaler.inverse_transform(scaler.transform(y)), y, rtol=_EXACT, atol=0
        )

    def test_irregular_timeseries_calendar_divisor(self, irregular_index_timeseries):
        """Irregular index (yearly anchors before the monthly grid): 365 / 366 days per row."""
        index = irregular_index_timeseries.index
        divisors = StageScaler(scale_features='calendar').target_divisor(
            'balance_commerciale_annuelle', source_freq='YS', pred_freq='D', index=index
        )
        expected = [366.0 if calendar.isleap(d.year) else 365.0 for d in index]
        assert divisors.tolist() == expected


# =============================================================================
# Robustesse d'index
# =============================================================================
class TestIndexRobustness:
    """Shuffled, duplicated, empty and single-row grids; index names kept."""

    def test_shuffled_grid_keeps_each_divisor_on_its_row(self):
        """Per-row divisors follow their label, not their position."""
        frame = pd.DataFrame({'v': 0.0}, index=pd.date_range('2024-01-31', periods=12, freq='ME'))
        shuffled = shuffle_rows(frame, seed=3).index
        scaler = StageScaler(scale_features='calendar')
        sorted_divisors = scaler.target_divisor('v', source_freq='M', pred_freq='D', index=frame.index)
        shuffled_divisors = scaler.target_divisor('v', source_freq='M', pred_freq='D', index=shuffled)
        pd.testing.assert_series_equal(shuffled_divisors, sorted_divisors.reindex(shuffled))

    def test_duplicated_rows_are_scaled_positionally(self):
        """Duplicated dates: every row divides by its own period's count, no realignment."""
        index = pd.date_range('2024-01-31', periods=3, freq='ME')
        X = with_duplicated_rows(pd.DataFrame({'m1': [31.0, 29.0, 31.0]}, index=index), n=2)
        scaler = StageScaler(
            scale_features='calendar', source_freq='M', pred_freq='D',
            column_frequencies={'m1': 'M'}, materialization={'m1': 'raw_anchors'},
        ).fit(X, X['m1'])
        # Valeur d'or : chaque valeur vaut le nombre de jours de son mois -> 1.0 partout
        assert (scaler.transform(X)['m1'].tolist(), scaler.transform(X['m1']).tolist()) == (
            [1.0] * 5, [1.0] * 5
        )

    def test_empty_grid(self):
        """An empty grid gives empty divisors and an empty transform, without error."""
        X = pd.DataFrame({'m1': []}, index=pd.DatetimeIndex([]), dtype=float)
        scaler = StageScaler(
            scale_features='calendar', source_freq='M', pred_freq='D', column_frequencies={'m1': 'M'}
        ).fit(X)
        assert (scaler.target_divisor_.size, scaler.transform(X).shape) == (0, (0, 1))

    def test_single_observation(self):
        """A single row under 'calendar': one divisor, the days of its month."""
        index = pd.DatetimeIndex(['2024-02-10'])
        divisor = StageScaler(scale_features='calendar').target_divisor(
            'v', source_freq='M', pred_freq='D', index=index
        )
        assert divisor.tolist() == [29.0]

    def test_index_name_is_kept(self):
        """A non-standard index name is carried over to the per-row divisors."""
        index = pd.DatetimeIndex(['2024-05-15'], name='période de référence')
        divisor = StageScaler(scale_features='calendar').target_divisor(
            'v', source_freq='Q', pred_freq='D', index=index
        )
        assert (divisor.index.name, divisor.tolist()) == ('période de référence', [91.0])


# =============================================================================
# Protocole sklearn
# =============================================================================
class TestSklearnProtocol:
    """fit / transform / inverse_transform and sklearn conventions."""

    @staticmethod
    def _stage_frame() -> pd.DataFrame:
        """Build a small monthly stage frame with two covariates."""
        return pd.DataFrame(
            {'m1': [1200.0, 2400.0, 3600.0], 'a2': [120.0, 240.0, 360.0]},
            index=_monthly_grid(periods=3),
        )

    @staticmethod
    def _scaler(**overrides) -> StageScaler:
        """Scaler of the stage frame: 'm1' on the stage grid, 'a2' read at its yearly anchors."""
        params = dict(
            column_frequencies={'m1': 'M', 'a2': 'Y'},
            materialization={'m1': 'identity', 'a2': 'raw_anchors'},
        )
        params.update(overrides)
        return _make_scaler(**params)

    def test_fit_stores_the_three_divisors(self):
        """fit freezes the covariate divisors, the target divisor and the baked factor."""
        X = self._stage_frame()
        scaler = self._scaler().fit(X)
        assert (
            scaler.feature_divisors_.to_dict(), scaler.target_divisor_, scaler.fit_scale_factor_,
            scaler.n_features_in_, list(scaler.feature_names_in_),
        ) == ({'m1': 1.0, 'a2': 12.0}, 12.0, 12.0, 2, ['m1', 'a2'])

    def test_fit_returns_self(self):
        """fit returns the instance itself."""
        scaler = self._scaler()
        assert scaler.fit(self._stage_frame()) is scaler

    def test_transform_values(self):
        """transform divides 'a2' by 12 and leaves 'm1' (already at the stage scale)."""
        scaled = self._scaler().fit(self._stage_frame()).transform(self._stage_frame())
        assert scaled.to_dict('list') == {'m1': [1200.0, 2400.0, 3600.0], 'a2': [10.0, 20.0, 30.0]}

    def test_fit_transform_equals_fit_then_transform(self):
        """TransformerMixin.fit_transform gives the same frame as fit().transform()."""
        X = self._stage_frame()
        pd.testing.assert_frame_equal(
            self._scaler().fit_transform(X), self._scaler().fit(X).transform(X)
        )

    def test_fit_accepts_production_frequencies(self):
        """produced_freq feeds target_divisor_ without touching fit_scale_factor_."""
        X = self._stage_frame()
        y = pd.Series([120.0, 28.0, 30.0], index=X.index)
        scaler = self._scaler().fit(X, y, produced_freq=pd.Series(['Y', 'Q', 'Q'], index=X.index))
        assert (scaler.fit_scale_factor_, scaler.target_divisor_.tolist()) == (12.0, [12.0, 3.0, 3.0])

    def test_target_grid_defaults_to_the_grid_of_x(self):
        """Without y, the per-row target divisor runs on the grid of X."""
        X = pd.DataFrame({'m1': [1.0, 2.0]}, index=pd.to_datetime(['2024-01-31', '2024-02-29']))
        scaler = StageScaler(scale_features='calendar', source_freq='M', pred_freq='D').fit(X)
        pd.testing.assert_series_equal(scaler.target_divisor_, pd.Series([31.0, 29.0], index=X.index))

    def test_name_of_y_designates_the_imputed_column(self):
        """The name of y alone designates the imputed column whose mode applies."""
        X = self._stage_frame()
        y = pd.Series([120.0, 132.0, 150.0], index=X.index, name='a1')
        scaler = self._scaler(scale_features={'a1': False, DEFAULT_SCALE_KEY: 'constant'}).fit(X, y)
        assert (scaler.target_divisor_, scaler.fit_scale_factor_, scaler.feature_divisors_.to_dict()) == (
            1.0, 1.0, {'m1': 1.0, 'a2': 12.0}
        )

    def test_unnamed_y_falls_back_on_the_global_setting(self):
        """An anonymous target resolves no entry: the '__default__' setting applies."""
        X = self._stage_frame()
        y = pd.Series([120.0, 132.0, 150.0], index=X.index)
        scaler = self._scaler(scale_features={'a1': False, DEFAULT_SCALE_KEY: 'constant'}).fit(X, y)
        assert scaler.target_divisor_ == 12.0

    @pytest.mark.parametrize('missing', ['source_freq', 'pred_freq'])
    def test_fit_without_frequencies_raises(self, missing):
        """source_freq and pred_freq are mandatory at fit."""
        with pytest.raises(ValueError, match=missing):
            self._scaler(**{missing: None}).fit(self._stage_frame())

    @pytest.mark.parametrize('method', ['transform', 'inverse_transform'])
    def test_not_fitted_raises(self, method):
        """transform and inverse_transform before fit raise NotFittedError."""
        with pytest.raises(NotFittedError):
            getattr(self._scaler(), method)(self._stage_frame())

    def test_get_params_lists_every_constructor_argument(self):
        """get_params returns the seven constructor arguments, as received."""
        setting = {'m1': 'calendar'}
        params = self._scaler(scale_features=setting).get_params()
        assert sorted(params) == sorted([
            'scale_features', 'source_freq', 'pred_freq', 'column_frequencies',
            'materialization', 'default_divisor', 'converter',
        ]) and params['scale_features'] is setting

    def test_set_params_roundtrip(self):
        """set_params changes a parameter and returns the instance."""
        scaler = self._scaler()
        assert scaler.set_params(pred_freq='Q') is scaler and scaler.pred_freq == 'Q'

    def test_clone_of_a_fitted_scaler_is_unfitted_with_equal_params(self):
        """clone copies the configuration, not the fitted divisors."""
        scaler = self._scaler(scale_features={'m1': 'calendar'}).fit(self._stage_frame())
        twin = clone(scaler)
        assert twin.get_params() == scaler.get_params() and not hasattr(twin, 'feature_divisors_')

    def test_refit_replaces_every_divisor(self):
        """A second fit on another grid and stage leaves nothing of the first."""
        scaler = self._scaler()
        scaler.fit(self._stage_frame())
        scaler.set_params(pred_freq='Q', source_freq='Y', column_frequencies={'m1': 'M', 'a2': 'Y'})
        scaler.fit(self._stage_frame())
        assert (scaler.feature_divisors_.to_dict(), scaler.target_divisor_) == ({'m1': 1.0, 'a2': 4.0}, 4.0)

    def test_divisor_methods_need_no_fit(self):
        """The three divisor methods work on an unfitted, unconfigured instance."""
        scaler = StageScaler()
        features = scaler.feature_divisors(
            columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'},
            grid_freq='Y', stage_freq='M',
        )
        assert (
            scaler.target_divisor('a1', source_freq='Y', pred_freq='M'),
            scaler.fit_scale_factor('a1', source_freq='Y', pred_freq='M'),
            features['m1'],
        ) == (12.0, 12.0, 12.0)

    def test_one_instance_serves_every_stage(self):
        """Stateless: a shared instance gives the same divisors as fresh ones, in any order."""
        kwargs = dict(columns=['m1'], column_frequencies={'m1': 'M'}, ways={'m1': 'aggregate'}, stage_freq='M')
        shared = StageScaler()
        sequence = [shared.feature_divisors(grid_freq=grid, **kwargs)['m1'] for grid in ('Y', 'Q', 'Y')]
        fresh = [StageScaler().feature_divisors(grid_freq=grid, **kwargs)['m1'] for grid in ('Y', 'Q', 'Y')]
        assert sequence == fresh == [12.0, 3.0, 12.0]


# =============================================================================
# Modalités par entité (D41)
# =============================================================================
# Réglage de référence : 'v' brute pour DE, CVS pour toutes les autres entités
_DE_RAW = {'v': {'DE': 'calendar', DEFAULT_SCALE_KEY: 'constant'}}


# Grille de panel à deux niveaux d'entité (zone, pays, date)
def _two_level_grid(periods: int = 2) -> pd.MultiIndex:
    """Build a month-end grid with two entity levels.

    Args:
        periods: Number of months per entity.

    Returns:
        A ``MultiIndex`` (``zone``, ``country``, ``date``): entities
        ``('Zone euro', 'DE')`` then ``('Zone euro', 'FR')``.
    """
    return pd.MultiIndex.from_product(
        [['Zone euro'], ['DE', 'FR'], _monthly_grid(periods=periods)],
        names=['zone', 'country', 'date'],
    )


class TestPerEntityModes:
    """``scale_features`` as ``{column: {entity: mode}}`` (D41).

    Golden values: a monthly variable at a daily stage divides by 30.0 under
    ``'constant'`` and by the real days of each month under ``'calendar'``
    (January / February 2021: 31 / 28); a quarterly one by 91.0 under
    ``'constant'``.
    """

    # --- Validation ----------------------------------------------------------
    @pytest.mark.parametrize(
        'setting',
        [
            _DE_RAW,
            {'v': {'DE': 'calendar'}},
            {'v': {('Zone euro', 'DE'): 'calendar'}},
            {DEFAULT_SCALE_KEY: {'IT': 'calendar', DEFAULT_SCALE_KEY: 'constant'}},
            {'v': {'DE': 'calendar'}, 'q1': False},
        ],
        ids=['nested-default', 'no-nested-default', 'two-level-key', 'top-default', 'mixed-columns'],
    )
    def test_admissible_nested_settings_are_stored_untouched(self, setting):
        """The nested form passes the validation and is stored as the same object."""
        assert StageScaler(scale_features=setting).scale_features is setting

    @pytest.mark.parametrize(
        'setting',
        [
            {'v': {'DE': False}},
            {'v': {'DE': 'calendar', DEFAULT_SCALE_KEY: False}},
            {DEFAULT_SCALE_KEY: {'DE': False}},
        ],
        ids=['entity', 'nested-default', 'top-default'],
    )
    def test_false_within_a_nested_dict_raises(self, setting):
        """False applies to a whole column, never to some of its entities."""
        with pytest.raises(ValueError, match=r"never to some of its entities"):
            StageScaler(scale_features=setting)

    @pytest.mark.parametrize(
        'setting, message',
        [
            ({'v': {}}, r"per-entity dict cannot be empty"),
            ({'v': {'DE': 'exact'}}, r"only admits \('constant', 'calendar'\)"),
            ({'v': {'DE': np.False_}}, r"only admits"),
            ({'v': {'DE': 0}}, r"only admits"),
            ({'v': {'DE': {'x': 'calendar'}}}, r"only admits"),
        ],
        ids=['empty', 'unknown-literal', 'numpy-false', 'zero', 'third-level'],
    )
    def test_inadmissible_nested_values_raise(self, setting, message):
        """A nested dict holds 'constant' or 'calendar' only, and is never empty."""
        with pytest.raises(ValueError, match=message):
            StageScaler(scale_features=setting)

    def test_false_within_a_nested_dict_raises_after_set_params(self):
        """set_params bypasses __init__: the resolution validates again."""
        scaler = StageScaler().set_params(scale_features={'v': {'DE': False}})
        with pytest.raises(ValueError, match=r"never to some of its entities"):
            scaler.resolve_mode('v', 'DE')

    # --- Résolution -----------------------------------------------------------
    @pytest.mark.parametrize(
        'setting, column, entity, expected',
        [
            (_DE_RAW, 'v', 'DE', 'calendar'),
            (_DE_RAW, 'v', ('DE',), 'calendar'),
            (_DE_RAW, 'v', 'FR', 'constant'),
            (_DE_RAW, 'v', None, 'constant'),
            (_DE_RAW, 'q1', 'DE', 'constant'),
            ({'v': {'DE': 'constant'}, DEFAULT_SCALE_KEY: 'calendar'}, 'v', 'FR', 'constant'),
            ({'v': {'DE': 'constant'}, DEFAULT_SCALE_KEY: False}, 'v', 'FR', 'constant'),
            ({'v': {'DE': 'constant'}, DEFAULT_SCALE_KEY: 'calendar'}, 'q1', 'FR', 'calendar'),
            ({DEFAULT_SCALE_KEY: {'IT': 'calendar'}}, 'q1', 'IT', 'calendar'),
            ({DEFAULT_SCALE_KEY: {'IT': 'calendar'}, 'v': False}, 'v', 'IT', False),
            ({'v': {('DE',): 'calendar'}}, 'v', 'DE', 'calendar'),
        ],
        ids=['own-entity', 'tuple-entity', 'nested-default', 'no-entity', 'other-column',
             'closed-on-calendar-default', 'closed-on-false-default', 'top-default-elsewhere',
             'nested-top-default', 'column-scalar-wins', 'tuple-key-scalar-entity'],
    )
    def test_resolve_mode(self, setting, column, entity, expected):
        """Column entry first; a nested entry is resolved within itself only."""
        assert StageScaler(scale_features=setting).resolve_mode(column, entity) == expected

    @pytest.mark.parametrize(
        'entity, expected',
        [
            (('Zone euro', 'DE'), 'calendar'),
            (('Zone euro', 'FR'), 'constant'),
            ('Zone euro', 'constant'),
            (('Zone euro',), 'constant'),
            ('DE', 'constant'),
        ],
        ids=['full-key', 'other-entity', 'partial-scalar', 'partial-tuple', 'inner-level'],
    )
    def test_two_level_keys_match_full_entities_only(self, entity, expected):
        """A partial key — outer or inner level — designates no entity."""
        scaler = StageScaler(scale_features={'v': {('Zone euro', 'DE'): 'calendar'}})
        assert scaler.resolve_mode('v', entity) == expected

    def test_partial_key_does_not_cover_its_sub_entities(self):
        """'Zone euro' as a key leaves every ('Zone euro', *) entity on the nested default."""
        scaler = StageScaler(scale_features={'v': {'Zone euro': 'calendar'}})
        modes = [scaler.resolve_mode('v', e) for e in (('Zone euro', 'DE'), ('Zone euro', 'FR'))]
        assert modes == ['constant', 'constant']

    # --- Diviseurs de la cible ------------------------------------------------
    def test_target_divisor_gold(self):
        """DE divides by its real days (31, 28), FR by the constant 30."""
        index = _panel_grid(periods=2)
        divisors = StageScaler(scale_features=_DE_RAW).target_divisor(
            'v', source_freq='M', pred_freq='D', index=index
        )
        pd.testing.assert_series_equal(
            divisors, pd.Series([31.0, 28.0, 30.0, 30.0], index=index)
        )

    def test_fit_scale_factor_gold(self):
        """The baked factor follows the same per-entity modes, frozen on the fit grid."""
        index = _panel_grid(periods=2)
        factor = StageScaler(scale_features=_DE_RAW).fit_scale_factor(
            'v', source_freq='M', pred_freq='D', index=index
        )
        pd.testing.assert_series_equal(factor, pd.Series([31.0, 28.0, 30.0, 30.0], index=index))

    def test_target_divisor_with_production_frequencies(self):
        """Per-row production frequencies combine with per-entity modes."""
        index = _panel_grid(periods=2)
        produced = pd.Series(['M', 'D', 'M', 'D'], index=index)
        divisors = StageScaler(scale_features=_DE_RAW).target_divisor(
            'v', source_freq='M', pred_freq='D', index=index, produced_freq=produced
        )
        assert divisors.tolist() == [31.0, 1.0, 30.0, 1.0]

    def test_target_divisor_with_per_entity_blocks(self):
        """Per-entity block frequencies and per-entity modes, row by row.

        DE : bloc mensuel, jours réels ; FR : bloc trimestriel, facteur
        constant D -> Q = 91.
        """
        index = _panel_grid(periods=2)
        divisors = StageScaler(scale_features=_DE_RAW).target_divisor(
            'v', source_freq={'DE': 'M', 'FR': 'Q'}, pred_freq='D', index=index
        )
        assert divisors.tolist() == [31.0, 28.0, 91.0, 91.0]

    def test_rows_of_each_entity_keep_their_mode_on_a_shuffled_grid(self):
        """The divisors land on their own rows whatever the order of the grid."""
        frame = shuffle_rows(pd.DataFrame({'v': 0.0}, index=_panel_grid(periods=2)))
        divisors = StageScaler(scale_features=_DE_RAW).target_divisor(
            'v', source_freq='M', pred_freq='D', index=frame.index
        )
        january, february = pd.Timestamp('2021-01-31'), pd.Timestamp('2021-02-28')
        assert divisors.to_dict() == {
            ('DE', january): 31.0, ('DE', february): 28.0,
            ('FR', january): 30.0, ('FR', february): 30.0,
        }

    def test_two_level_entities(self):
        """On a two-level panel, only the full key ('Zone euro', 'DE') is raw."""
        divisors = StageScaler(
            scale_features={'v': {('Zone euro', 'DE'): 'calendar'}}
        ).target_divisor('v', source_freq='M', pred_freq='D', index=_two_level_grid())
        assert divisors.tolist() == [31.0, 28.0, 30.0, 30.0]

    def test_partial_key_on_a_two_level_panel_divides_by_the_nested_default(self):
        """'Zone euro' matches no entity: every row takes the nested default."""
        divisors = StageScaler(
            scale_features={'v': {'Zone euro': 'calendar'}}
        ).target_divisor('v', source_freq='M', pred_freq='D', index=_two_level_grid())
        assert divisors == 30.0

    def test_timeseries_grid_reads_the_nested_default(self):
        """No entity level: the nested default applies to every row."""
        scaler = StageScaler(scale_features={'v': {'DE': 'constant', DEFAULT_SCALE_KEY: 'calendar'}})
        divisors = scaler.target_divisor(
            'v', source_freq='M', pred_freq='D', index=_monthly_grid(periods=2)
        )
        assert divisors.tolist() == [31.0, 28.0]

    def test_grid_without_named_entity_keeps_the_scalar_form(self):
        """Every entity of the grid on one mode: the result is that of the scalar setting."""
        divisors = StageScaler(scale_features={'v': {'IT': 'calendar'}}).target_divisor(
            'v', source_freq='M', pred_freq='D', index=_panel_grid(periods=2)
        )
        assert divisors == 30.0

    def test_unanimous_nested_dict_needs_no_index(self):
        """Modes that cannot differ need no grid, exactly like the scalar form."""
        scaler = StageScaler(scale_features={'v': {'DE': 'constant'}})
        assert scaler.target_divisor('v', source_freq='M', pred_freq='D') == 30.0

    @pytest.mark.parametrize('method', ['target_divisor', 'fit_scale_factor'])
    def test_disagreeing_modes_without_index_raise(self, method):
        """Spreading per-entity modes needs a grid: never a silent nested default."""
        with pytest.raises(ValueError, match=r"per-entity modes that differ"):
            getattr(StageScaler(scale_features=_DE_RAW), method)('v', source_freq='M', pred_freq='D')

    def test_false_column_beside_a_nested_column(self):
        """A column in False stays exempt; the nested column next to it is per entity."""
        scaler = StageScaler(scale_features={'v': {'DE': 'calendar'}, 'w': False})
        index = _panel_grid(periods=2)
        assert scaler.target_divisor('w', source_freq='M', pred_freq='D', index=index) == 1.0
        divisors = scaler.target_divisor('v', source_freq='M', pred_freq='D', index=index)
        assert divisors.tolist() == [31.0, 28.0, 30.0, 30.0]

    # --- Diviseurs des covariables --------------------------------------------
    def test_feature_divisors_gold(self):
        """A daily covariate aggregated on months: real days for DE, 30 for FR."""
        divisors = StageScaler(
            scale_features={'d1': {'DE': 'calendar'}, 'd2': 'constant'}
        ).feature_divisors(
            columns=['d1', 'd2'],
            column_frequencies={'d1': 'D', 'd2': 'D'},
            ways={'d1': 'aggregate', 'd2': 'aggregate'},
            grid_freq='M',
            stage_freq='D',
            index=_panel_grid(periods=2),
        )
        assert isinstance(divisors, pd.DataFrame)
        assert divisors['d1'].tolist() == [31.0, 28.0, 30.0, 30.0]
        assert divisors['d2'].tolist() == [30.0] * 4

    def test_feature_divisors_on_a_mutualized_grid(self):
        """Per-entity blocks and per-entity modes on the same training grid."""
        divisors = StageScaler(scale_features={'d1': {'DE': 'calendar'}}).feature_divisors(
            columns=['d1'],
            column_frequencies={'d1': 'D'},
            ways={'d1': 'aggregate'},
            grid_freq={'DE': 'M', 'FR': 'M'},
            stage_freq='D',
            index=_panel_grid(periods=2),
        )
        assert divisors['d1'].tolist() == [31.0, 28.0, 30.0, 30.0]

    def test_feature_divisors_of_unanimous_modes_keep_the_compact_form(self):
        """A nested dict resolving to 'constant' everywhere changes nothing to the shape."""
        kwargs = dict(
            columns=['d1'], column_frequencies={'d1': 'D'}, ways={'d1': 'aggregate'},
            grid_freq='M', stage_freq='D', index=_panel_grid(periods=2),
        )
        nested = StageScaler(scale_features={'d1': {'IT': 'calendar'}}).feature_divisors(**kwargs)
        scalar = StageScaler(scale_features='constant').feature_divisors(**kwargs)
        pd.testing.assert_series_equal(nested, scalar)

    def test_feature_divisors_with_disagreeing_modes_without_index_raise(self):
        """Same prerequisite on the covariate side."""
        with pytest.raises(ValueError, match=r"per-entity modes that differ"):
            StageScaler(scale_features={'d1': {'DE': 'calendar'}}).feature_divisors(
                columns=['d1'], column_frequencies={'d1': 'D'}, ways={'d1': 'aggregate'},
                grid_freq='M', stage_freq='D',
            )

    def test_feature_divisors_two_level_entities(self):
        """Full two-level key on the covariate side."""
        divisors = StageScaler(
            scale_features={'d1': {('Zone euro', 'FR'): 'calendar'}}
        ).feature_divisors(
            columns=['d1'], column_frequencies={'d1': 'D'}, ways={'d1': 'aggregate'},
            grid_freq='M', stage_freq='D', index=_two_level_grid(),
        )
        assert divisors['d1'].tolist() == [30.0, 30.0, 31.0, 28.0]

    # --- Protocole sklearn ----------------------------------------------------
    def test_fit_transform_roundtrip(self):
        """fit / transform / inverse_transform of the target under per-entity modes."""
        index = _panel_grid(periods=2)
        X = pd.DataFrame({'d1': [1.0, 2.0, 3.0, 4.0]}, index=index)
        y = pd.Series([310.0, 280.0, 300.0, 300.0], index=index, name='v')
        scaler = StageScaler(
            scale_features=_DE_RAW, source_freq='M', pred_freq='D', column_frequencies={'d1': 'D'},
        )
        scaled = scaler.fit(X, y).transform(y)
        assert scaled.tolist() == [10.0, 10.0, 10.0, 10.0]
        np.testing.assert_allclose(scaler.inverse_transform(scaled), y, rtol=0, atol=_EXACT)

    def test_clone_keeps_the_nested_setting(self):
        """clone and get_params carry the nested dict as a plain parameter."""
        twin = clone(StageScaler(scale_features=_DE_RAW))
        assert twin.get_params()['scale_features'] == _DE_RAW
        assert twin.resolve_mode('v', 'DE') == 'calendar'
