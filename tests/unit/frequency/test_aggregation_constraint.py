"""Tests for tsforecast.frequency.aggregation_constraint.

The aggregation constraint rescales the sub-periods predicted for one period
of a lower-frequency variable by ``observed total / predicted aggregate``, so
that the column carries a genuine disaggregation of the observation. The
aggregate is always the sum: ``'mean'`` and ``'last'`` are not accepted,
additivity being the contract of the whole imputer.

Contracts pinned here:

- validation: exhaustive table of admissible and rejected settings (scalar,
  dict form with ``DEFAULT_CONSTRAINT_KEY``, invalid types), messages naming
  the admissible forms and ``additive_transformer``; dict keys checked
  against the real columns; resolution rules of the dict form; parity of the
  shared validation with ``CovariateMaterializer``;
- ``rescale``: the invariant "rescaled sub-periods sum to the observed total"
  checked as a PROPERTY (random predictions, totals of both signs) on Y→M,
  Y→Q and Q→M, start and end positions, to ``1e-9``; hand-computed golden
  values; the guards (NaN sub-period, period without observation,
  calendar truncation at both edges, zero aggregate) and the degenerate
  totals (zero, NEGATIVE — the trade balance of ``irregular_index_timeseries``
  —, all-zero sub-periods); panels whose source frequency differs per
  entity, three-level index, shuffled rows;
- ``anchor_cells_mask``: exactly the anchor cells, whatever the constraint
  and the rescaling outcome;
- sklearn protocol: ``clone``, ``get_params`` / ``set_params``,
  ``fit_transform == fit().transform()``, ``NotFittedError``, metadata errors;
- warnings: aggregated and unique per degenerate case, never one per period
  nor per column; ``deferred_warnings`` gathers a chain of calls (several
  columns, stages, covariates of the materializer) into one warning per case,
  each period counted once;
- edge cases: numerically zero aggregate (rounding
  bound of the summation), duplicated dates rejected, setting validated again
  after ``set_params``, ``grid_freq`` validated (readable, covering every
  entity), ``NaT`` rows outside any period, ``transform`` restricted to the
  fitted columns, ``rescaled_mask_`` purged at ``fit``.

Covered symbols: ``AggregationConstraint`` (``fit``, ``transform``,
``rescale``, ``anchor_cells_mask``, ``validate_columns``,
``resolve_constraint``), ``validate_aggregation_constraint``,
``resolve_aggregation_constraint``, ``validate_constraint_columns``,
``ConstraintKind``, ``DEFAULT_CONSTRAINT_KEY``. Realistic datasets:
``irregular_index_timeseries`` (negative annual totals, isolated anchors
before the monthly grid) and ``heterogeneous_coverage_panel``
(``depenses_publiques_pib`` annual for France / Italie, quarterly for
Allemagne).
"""
# Modules de base
import typing
import warnings

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
from tsforecast.frequency.aggregation_constraint import (
    AggregationConstraint,
    ConstraintKind,
    DEFAULT_CONSTRAINT_KEY,
    resolve_aggregation_constraint,
    validate_aggregation_constraint,
    validate_constraint_columns,
)
from tsforecast.frequency.covariate_materializer import CovariateMaterializer

# Perturbations partagées
from tests.support.perturbations import (
    reverse_entities,
    shuffle_rows,
    to_three_level_index,
    with_special_column_names,
)


# Seuil d'exactitude de la contrainte : un écart supérieur est une anomalie,
# jamais une tolérance à élargir
_EXACT = 1e-9

# Prédictions brutes des douze mois de 2021 : somme 112.5, décembre 10.5.
# Valeurs de l'exemple de référence — ratio 120 / 112.5, donc
# janvier 9.6, février 10.1333… et l'ancre 11.2
_RAW_2021 = [9.0, 9.5, 9.5, 9.5, 9.0, 9.5, 9.0, 9.5, 9.0, 9.5, 9.0, 10.5]

# Alias pandas d'une grille selon sa base et sa position dans la période
_GRID_ALIASES = {
    ('M', 'start'): 'MS', ('M', 'end'): 'ME',
    ('Q', 'start'): 'QS', ('Q', 'end'): 'QE',
    ('Y', 'start'): 'YS', ('Y', 'end'): 'YE',
}

# Couples (fréquence des totaux, base de la grille) du recalage
_FREQUENCY_PAIRS = {
    'Y->M': ('Y', 'M'),
    'Y->Q': ('Y', 'Q'),
    'Q->M': ('Q', 'M'),
}


# Fabrique de la grille mensuelle d'une année
def _monthly_grid(year: int = 2021) -> pd.DatetimeIndex:
    """Build the twelve month-end dates of one year.

    Args:
        year: Calendar year of the grid.

    Returns:
        Month-end ``DatetimeIndex`` of twelve rows.
    """
    return pd.date_range(f'{year}-01-31', periods=12, freq='ME')


# Fabrique des prédictions brutes d'une année
def _raw_predictions(year: int = 2021, values=None) -> pd.Series:
    """Build the raw sub-period predictions of one year.

    Args:
        year: Calendar year of the grid.
        values: Values to carry; defaults to the reference vector.

    Returns:
        ``Series`` named ``'a1'`` on the twelve month-end dates of ``year``.
    """
    return pd.Series(
        _RAW_2021 if values is None else values,
        index=_monthly_grid(year),
        name='a1',
    )


# Fabrique des observations annuelles sur une grille mensuelle
def _annual_observations(year: int = 2021, total: float = 120.0) -> pd.Series:
    """Build the annual observation of one year, on its monthly grid.

    Args:
        year: Calendar year of the grid.
        total: Observed annual total, carried by the December anchor.

    Returns:
        ``Series`` named ``'a1'``, non-null on the December anchor only.
    """
    observations = pd.Series(np.nan, index=_monthly_grid(year), name='a1')
    observations.iloc[-1] = total
    return observations


# Fabrique d'un bloc de panel à partir d'une série temporelle
def _to_panel(series: pd.Series, entities=('FR', 'DE')) -> pd.Series:
    """Replicate a time series over several entities, as a panel Series.

    Args:
        series: Time series to replicate.
        entities: Entity keys of the panel.

    Returns:
        ``Series`` on a ``MultiIndex`` ``(country, date)``.
    """
    blocks = []
    for entity in entities:
        block = series.copy()
        block.index = pd.MultiIndex.from_product(
            [[entity], series.index], names=['country', 'date']
        )
        blocks.append(block)
    return pd.concat(blocks)


# Fabrique des observations : une ancre par période, au début ou à la fin
def _anchor_observations(
    grid: pd.DatetimeIndex,
    period_freq: str,
    totals: typing.Sequence[float],
    position: str,
) -> pd.Series:
    """Place one observed total per period on its anchor row of the grid.

    Args:
        grid: Sorted grid covering whole periods.
        period_freq: Frequency of the periods (``'Y'`` or ``'Q'``).
        totals: One observed total per period, in chronological order.
        position: ``'start'`` anchors on the first row of each period (``YS``
            convention), ``'end'`` on the last one (``YE`` convention).

    Returns:
        ``Series`` on ``grid``, non-null on the anchor rows only.
    """
    observations = pd.Series(np.nan, index=grid)
    periods = grid.to_period(period_freq)
    for total, period in zip(totals, periods.unique()):
        rows = np.flatnonzero(periods == period)
        anchor = rows[0] if position == 'start' else rows[-1]
        observations.iloc[anchor] = total
    return observations


# Écart maximal entre la somme recalée de chaque période et son total observé
def _max_period_gap(
    rescaled: pd.Series, observations: pd.Series, period_freq: str
) -> float:
    """Return the largest |sum of rescaled sub-periods - observed total|.

    Args:
        rescaled: Rescaled values on a ``DatetimeIndex``.
        observations: Observed totals on their anchor rows.
        period_freq: Frequency of the periods.

    Returns:
        The worst gap over the periods holding an observation.
    """
    sums = rescaled.groupby(rescaled.index.to_period(period_freq)).sum()
    observed = observations.dropna()
    totals = observed.groupby(observed.index.to_period(period_freq)).sum()
    return float((sums.reindex(totals.index) - totals).abs().max())


# Capture de tous les avertissements d'un appel
def _record_warnings(function, *args, **kwargs):
    """Call ``function`` and return its result with every warning raised.

    Args:
        function: Callable to run.
        *args: Positional arguments of the call.
        **kwargs: Keyword arguments of the call.

    Returns:
        Tuple ``(result, user_warnings)`` — the list of ``UserWarning``
        messages, as strings.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = function(*args, **kwargs)
    messages = [
        str(w.message) for w in caught if issubclass(w.category, UserWarning)
    ]
    return result, messages


# =============================================================================
# Vocabulaire public
# =============================================================================
class TestPublicVocabulary:
    """Public constants of the module."""

    def test_constraint_kind_admits_only_sum(self):
        """ConstraintKind is Literal['sum']: 'mean' and 'last' are gone."""
        assert typing.get_args(ConstraintKind) == ('sum',)

    def test_default_constraint_key(self):
        """The fallback key of the dict form is '__default__'."""
        assert DEFAULT_CONSTRAINT_KEY == '__default__'

    def test_default_setting_is_sum(self):
        """The constructor defaults to 'sum'."""
        assert AggregationConstraint().aggregation_constraint == 'sum'


# =============================================================================
# Validation du paramètre public
# =============================================================================
# Réglages admissibles : deux scalaires, et les dicts qui les associent à une
# colonne, clé de repli comprise
_ACCEPTED_SETTINGS = {
    'sum': 'sum',
    'none': None,
    'dict-sum': {'a1': 'sum'},
    'dict-none': {'a1': None},
    'dict-default-only': {DEFAULT_CONSTRAINT_KEY: None},
    'dict-mixed-default': {'a1': None, 'a2': 'sum', DEFAULT_CONSTRAINT_KEY: 'sum'},
    'dict-non-str-key': {1: 'sum'},
    'numpy-str': np.str_('sum'),
}

# Réglages refusés à la forme scalaire : agrégateurs retirés, casse et
# espaces, types non textuels — True / 0 ne doivent pas passer pour 'sum' / None
_REJECTED_SCALARS = {
    'mean': 'mean',
    'last': 'last',
    'upper-case': 'SUM',
    'capitalized': 'Sum',
    'padded': ' sum',
    'empty-string': '',
    'other-aggregator': 'median',
    'true': True,
    'false': False,
    'zero': 0,
    'one': 1,
    'float': 1.5,
    'nan': float('nan'),
    'list': ['sum'],
    'tuple': ('sum',),
    'set': {'sum'},
}

# Réglages refusés à la forme dictionnaire : une seule valeur fautive suffit
_REJECTED_DICTS = {
    'dict-mean': {'a1': 'mean'},
    'dict-last': {'a1': 'last'},
    'dict-default-mean': {DEFAULT_CONSTRAINT_KEY: 'mean'},
    'dict-true': {'a1': True},
    'dict-list': {'a1': ['sum']},
    'dict-one-bad-among-good': {'a1': 'sum', 'a2': None, 'a3': 'last'},
}


class TestValidateAggregationConstraint:
    """Admissible forms of ``aggregation_constraint``."""

    @pytest.mark.parametrize(
        'setting', list(_ACCEPTED_SETTINGS.values()), ids=list(_ACCEPTED_SETTINGS)
    )
    def test_admissible_settings_pass(self, setting):
        """'sum', None, and dicts of these values pass silently."""
        assert validate_aggregation_constraint(setting) is None

    @pytest.mark.parametrize(
        'setting', list(_ACCEPTED_SETTINGS.values()), ids=list(_ACCEPTED_SETTINGS)
    )
    def test_constructor_stores_admissible_settings_untouched(self, setting):
        """__init__ stores the setting as received (B3)."""
        assert AggregationConstraint(setting).aggregation_constraint is setting

    @pytest.mark.parametrize(
        'setting', list(_REJECTED_SCALARS.values()), ids=list(_REJECTED_SCALARS)
    )
    def test_rejected_scalar_raises_value_error(self, setting):
        """Any other scalar raises a ValueError."""
        with pytest.raises(ValueError, match='aggregation_constraint must be one of'):
            validate_aggregation_constraint(setting)

    @pytest.mark.parametrize(
        'setting', list(_REJECTED_DICTS.values()), ids=list(_REJECTED_DICTS)
    )
    def test_rejected_dict_raises_value_error(self, setting):
        """A dict holding one inadmissible value raises a ValueError."""
        with pytest.raises(ValueError, match='aggregation_constraint values must be one of'):
            validate_aggregation_constraint(setting)

    @pytest.mark.parametrize('setting', ['mean', 'last'], ids=['mean', 'last'])
    def test_withdrawn_aggregator_message_lists_admissible_forms(self, setting):
        """The message of a withdrawn aggregator lists the forms and names additive_transformer.

        The message lists the admissible forms ('sum', None, the dict
        form with its '__default__' key) and names ``additive_transformer`` as
        the answer to a non-additive column, and repeats the value.
        """
        with pytest.raises(ValueError) as error:
            AggregationConstraint(setting)
        message = str(error.value)

        # Formes admises, échappatoire et valeur reçue, toutes nommées
        expected_fragments = [
            "('sum', None)", 'dict', repr(DEFAULT_CONSTRAINT_KEY),
            'additive_transformer', f'got {setting!r}',
        ]
        assert [fragment for fragment in expected_fragments if fragment not in message] == []

    def test_dict_message_names_only_the_offending_entries(self):
        """The dict message quotes the invalid entries, not the valid ones."""
        with pytest.raises(ValueError) as error:
            validate_aggregation_constraint({'a1': 'sum', 'a2': 'mean', 'a3': 'last'})
        message = str(error.value)

        # Valeur d'or : seules a2 et a3 sont fautives
        assert "got {'a2': 'mean', 'a3': 'last'}" in message
        assert 'additive_transformer' in message

    def test_empty_dict_raises(self):
        """An empty dict designates no column and raises."""
        with pytest.raises(ValueError, match='cannot be empty'):
            AggregationConstraint({})

    @pytest.mark.parametrize(
        'setting',
        ['mean', 'last', True, {'a1': 'mean'}, {}],
        ids=['mean', 'last', 'true', 'dict-mean', 'empty-dict'],
    )
    def test_covariate_materializer_rejects_the_same_settings(self, setting):
        """CovariateMaterializer shares the validation: same error, same message.

        Both components carry the parameter, a single implementation
        keeps them from drifting.
        """
        with pytest.raises(ValueError) as reference:
            validate_aggregation_constraint(setting)
        with pytest.raises(ValueError) as materializer:
            CovariateMaterializer(aggregation_constraint=setting)

        assert str(materializer.value) == str(reference.value)


class TestResolveAggregationConstraint:
    """Constraint effectively applied to one column."""

    # Table de résolution : (réglage, colonne, réglage attendu)
    _CASES = {
        'scalar-sum': ('sum', 'a1', 'sum'),
        'scalar-none': (None, 'a1', None),
        'scalar-global': ('sum', None, 'sum'),
        'own-entry-none': ({'a1': None, DEFAULT_CONSTRAINT_KEY: 'sum'}, 'a1', None),
        'own-entry-sum': ({'a1': 'sum', DEFAULT_CONSTRAINT_KEY: None}, 'a1', 'sum'),
        'fallback-on-default-key': ({'a1': 'sum', DEFAULT_CONSTRAINT_KEY: None}, 'q1', None),
        'fallback-on-document-default': ({'a1': None}, 'q1', 'sum'),
        'global-reads-default-key': ({'a1': 'sum', DEFAULT_CONSTRAINT_KEY: None}, None, None),
        'global-without-default-key': ({'a1': None}, None, 'sum'),
    }

    @pytest.mark.parametrize(
        'setting, column, expected', list(_CASES.values()), ids=list(_CASES)
    )
    def test_module_function(self, setting, column, expected):
        """Own entry, else '__default__', else 'sum'."""
        assert resolve_aggregation_constraint(setting, column) == expected

    @pytest.mark.parametrize(
        'setting, column, expected', list(_CASES.values()), ids=list(_CASES)
    )
    def test_method_matches_module_function(self, setting, column, expected):
        """resolve_constraint gives the same answer as the module function."""
        assert AggregationConstraint(setting).resolve_constraint(column) == expected


class TestValidateConstraintColumns:
    """Dict keys are checked against the real columns at fit."""

    @pytest.mark.parametrize('setting', ['sum', None], ids=['sum', 'none'])
    def test_scalar_form_has_no_key_to_check(self, setting):
        """A scalar setting never names a column, even against no column."""
        assert validate_constraint_columns(setting, []) is None

    def test_known_keys_and_default_key_pass(self):
        """The '__default__' key is not a column name and is never reported."""
        setting = {'a1': 'sum', DEFAULT_CONSTRAINT_KEY: None}
        assert validate_constraint_columns(setting, pd.Index(['a1', 'q1'])) is None

    def test_unknown_keys_are_listed_sorted(self):
        """Every unknown key is named, in sorted order."""
        setting = {'zz': 'sum', 'a1': None, 'yy': None, DEFAULT_CONSTRAINT_KEY: 'sum'}
        with pytest.raises(ValueError, match=r"unknown columns : \['yy', 'zz'\]"):
            validate_constraint_columns(setting, iter(['a1', 'q1']))

    def test_method_delegates_to_module_function(self):
        """validate_columns raises like the module function."""
        with pytest.raises(ValueError, match=r"unknown columns : \['zz'\]"):
            AggregationConstraint({'zz': 'sum'}).validate_columns(['a1'])

    def test_fit_checks_the_keys(self):
        """fit validates the dict keys against the columns of X."""
        grid = _monthly_grid()
        X = pd.DataFrame({'a1': _RAW_2021}, index=grid)
        constraint = AggregationConstraint(
            {'zz': 'sum'}, period_frequencies='Y',
            observations=pd.DataFrame({'a1': _annual_observations().to_numpy()}, index=grid),
        )
        with pytest.raises(ValueError, match='unknown columns'):
            constraint.fit(X)


class TestValidationThroughSetParams:
    """Settings injected by set_params are validated again.

    'mean' is not accepted by the API. ``set_params`` (GridSearchCV,
    pipelines) skips ``__init__``: every public entry point validates again,
    so that 'mean' is never silently treated as an active constraint.
    """

    @staticmethod
    def _stage():
        """Return a one-column stage frame and its annual observation."""
        grid = _monthly_grid()
        X = pd.DataFrame({'a1': _RAW_2021}, index=grid)
        observations = pd.DataFrame({'a1': _annual_observations().to_numpy()}, index=grid)
        return X, observations

    def test_rejected_at_fit(self):
        """fit raises the ValueError of __init__."""
        X, observations = self._stage()
        constraint = AggregationConstraint('sum', 'Y', observations).set_params(
            aggregation_constraint='mean'
        )

        with pytest.raises(ValueError, match='additive_transformer'):
            constraint.fit(X)

    def test_rejected_at_transform_after_fit(self):
        """A setting changed after fit is rejected at transform."""
        X, observations = self._stage()
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)
        constraint.set_params(aggregation_constraint={'a1': 'last'})

        with pytest.raises(ValueError, match='additive_transformer'):
            constraint.transform(X)

    def test_rejected_at_rescale(self):
        """rescale, usable without fit, validates too."""
        constraint = AggregationConstraint('sum').set_params(aggregation_constraint='mean')

        with pytest.raises(ValueError, match='additive_transformer'):
            constraint.rescale(_raw_predictions(), _annual_observations(), 'Y')

    def test_admissible_setting_through_set_params_is_honoured(self):
        """A valid setting injected by set_params is applied."""
        X, observations = self._stage()
        constraint = AggregationConstraint('sum', 'Y', observations).set_params(
            aggregation_constraint=None
        )

        pd.testing.assert_frame_equal(constraint.fit_transform(X), X)


# =============================================================================
# Recalage : invariant d'additivité par propriété
# =============================================================================
class TestRescaleSumInvariant:
    """Rescaled sub-periods sum to the observed total."""

    @pytest.mark.filterwarnings('ignore:Aggregation constraint flipped')
    @pytest.mark.parametrize('seed', range(5), ids=lambda seed: f'seed{seed}')
    @pytest.mark.parametrize('position', ['start', 'end'])
    @pytest.mark.parametrize(
        'pair', list(_FREQUENCY_PAIRS.values()), ids=list(_FREQUENCY_PAIRS)
    )
    def test_property_every_complete_period_sums_to_its_total(self, pair, position, seed):
        """Random predictions and totals of both signs: every period is exact to 1e-9.

        Three years of whole periods; per period, predictions of one random
        sign and magnitude 1-100, total drawn from N(0, 500) — so some periods
        are rescaled with a sign flip, which must not affect exactness.
        """
        period_freq, grid_base = pair
        rng = np.random.default_rng(seed)

        # Construction d'une grille couvrant exactement 2020-2022
        grid = pd.date_range(
            '2020-01-01', '2022-12-31', freq=_GRID_ALIASES[(grid_base, position)]
        )
        periods = grid.to_period(period_freq)
        n_periods = periods.nunique()

        # Prédictions : un signe par période, magnitudes strictement non nulles
        signs = rng.choice([-1.0, 1.0], size=n_periods)
        sign_by_row = signs[pd.factorize(periods)[0]]
        values = pd.Series(sign_by_row * rng.uniform(1.0, 100.0, len(grid)), index=grid)
        observations = _anchor_observations(
            grid, period_freq, rng.normal(0.0, 500.0, n_periods), position
        )

        rescaled, _mask = AggregationConstraint('sum').rescale(
            values, observations, period_freq,
            grid_freq=_GRID_ALIASES[(grid_base, position)],
        )

        assert _max_period_gap(rescaled, observations, period_freq) <= _EXACT

    @pytest.mark.filterwarnings('ignore:Aggregation constraint flipped')
    @pytest.mark.parametrize('position', ['start', 'end'])
    @pytest.mark.parametrize(
        'pair', list(_FREQUENCY_PAIRS.values()), ids=list(_FREQUENCY_PAIRS)
    )
    def test_property_every_cell_is_moved(self, pair, position):
        """With whole periods and non-zero aggregates, the mask covers the whole grid."""
        period_freq, grid_base = pair
        rng = np.random.default_rng(0)
        grid = pd.date_range(
            '2020-01-01', '2022-12-31', freq=_GRID_ALIASES[(grid_base, position)]
        )
        n_periods = grid.to_period(period_freq).nunique()
        values = pd.Series(rng.uniform(1.0, 100.0, len(grid)), index=grid)
        observations = _anchor_observations(
            grid, period_freq, rng.normal(0.0, 500.0, n_periods), position
        )

        _rescaled, mask = AggregationConstraint('sum').rescale(
            values, observations, period_freq,
            grid_freq=_GRID_ALIASES[(grid_base, position)],
        )

        assert mask.all()


class TestRescaleGoldenValues:
    """Hand-computed rescaling, for each frequency pair and position."""

    def test_reference_example_of_spec_6_5(self, reference_timeseries):
        """Reference example: a1 = 120 in 2021, stage M, raw predictions summing to 112.5."""
        # Grille et observations tirées du jeu de référence
        grid = reference_timeseries.loc['2021'].index
        observations = reference_timeseries.loc['2021', 'a1']
        values = pd.Series(_RAW_2021, index=grid, name='a1')

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        # Valeur d'or : chaque mois multiplié par 120 / 112.5 = 1.0666…
        expected = [value * 120.0 / 112.5 for value in _RAW_2021]
        np.testing.assert_allclose(rescaled.to_numpy(), expected, rtol=0, atol=1e-12)
        # La ligne d'ancre porte une sous-période (11.2), jamais le total
        assert rescaled.loc['2021-12-31'] == pytest.approx(11.2)
        assert mask.all()

    # Cas d'or : (fréquence des totaux, base de grille, prédictions, total, attendu)
    _GOLDEN = {
        # Valeur d'or : 10+20+30+40 = 100, ratio 120 / 100 = 1.2
        'Y->Q': ('Y', 'Q', [10.0, 20.0, 30.0, 40.0], 120.0, [12.0, 24.0, 36.0, 48.0]),
        # Valeur d'or : 1+2+3 = 6, ratio 12 / 6 = 2
        'Q->M': ('Q', 'M', [1.0, 2.0, 3.0], 12.0, [2.0, 4.0, 6.0]),
        # Valeur d'or : douze mois à 5, somme 60, ratio 120 / 60 = 2
        'Y->M': ('Y', 'M', [5.0] * 12, 120.0, [10.0] * 12),
    }

    @pytest.mark.parametrize('position', ['start', 'end'])
    @pytest.mark.parametrize(
        'period_freq, grid_base, raw, total, expected',
        list(_GOLDEN.values()), ids=list(_GOLDEN),
    )
    def test_golden_rescaling(self, period_freq, grid_base, raw, total, expected, position):
        """Sub-periods are multiplied by total / aggregate, at either anchor position."""
        grid = pd.date_range(
            '2021-01-01', periods=len(raw), freq=_GRID_ALIASES[(grid_base, position)]
        )
        values = pd.Series(raw, index=grid)
        observations = _anchor_observations(grid, period_freq, [total], position)

        rescaled, _mask = AggregationConstraint('sum').rescale(
            values, observations, period_freq,
            grid_freq=_GRID_ALIASES[(grid_base, position)],
        )

        assert rescaled.tolist() == expected

    def test_several_observations_of_one_period_are_summed(self):
        """Two observations falling in one period make a single target total.

        Pinned behaviour: the period frequency is a per-call setting. Quarterly
        observations rescaled with ``period_freq='Y'`` are summed into one
        annual target — the year is exact, the quarters are not. Hence the
        caller groups entities by source frequency (``HighFrequencyImputer``
        calls ``rescale`` with each step's own ``source_frequency``).
        """
        grid = _monthly_grid()
        values = pd.Series(1.0, index=grid)
        observations = pd.Series(np.nan, index=grid)
        # Valeur d'or : 4 trimestres de 30 → total annuel 120, 10 par mois
        observations.iloc[[2, 5, 8, 11]] = [30.0, 30.0, 30.0, 30.0]

        rescaled, _mask = AggregationConstraint('sum').rescale(
            values, observations, 'Y', grid_freq='M'
        )

        assert rescaled.tolist() == [10.0] * 12


# =============================================================================
# Recalage : gardes et totaux dégénérés
# =============================================================================
class TestRescaleGuards:
    """Guards against degenerate periods and totals."""

    def test_nan_sub_period_leaves_the_period_raw(self):
        """One NaN sub-period forbids the constraint: raw predictions kept, NaN kept."""
        values = _raw_predictions()
        values.iloc[3] = np.nan

        rescaled, mask = AggregationConstraint('sum').rescale(
            values, _annual_observations(), 'Y'
        )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    def test_nan_sub_period_does_not_block_the_other_periods(self):
        """Only the period holding the NaN escapes; the mask matches the moved cells."""
        grid = pd.date_range('2021-01-31', periods=24, freq='ME')
        values = pd.Series(_RAW_2021 * 2, index=grid, name='a1')
        values.loc['2022-05-31'] = np.nan
        observations = pd.Series(np.nan, index=grid, name='a1')
        observations.loc['2021-12-31'] = 120.0
        observations.loc['2022-12-31'] = 132.0

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        # Cellules effectivement modifiées (NaN exclu des deux côtés)
        changed = (rescaled != values) & values.notna()
        pd.testing.assert_series_equal(mask, changed, check_names=False)
        assert mask.loc['2021'].all() and not mask.loc['2022'].any()

    def test_period_without_observation_is_left_raw(self):
        """A delayed series end has no total to impose."""
        values = _raw_predictions()
        observations = pd.Series(np.nan, index=values.index, name='a1')

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    def test_observation_outside_the_grid_imposes_nothing(self):
        """A total whose period has no row on the grid is ignored."""
        values = _raw_predictions(2021)
        observations = pd.Series([132.0], index=[pd.Timestamp('2022-12-31')])

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    @pytest.mark.parametrize('position', ['start', 'end'])
    def test_periods_truncated_at_both_edges_are_left_raw(self, position):
        """Only the whole middle year is rescaled; the truncated edge years keep raw values.

        Grid March 2020 - October 2022: 2020 holds ten months, 2022 ten
        months, 2021 twelve. Spreading an annual total over ten months would
        over-attribute 20 % to each.
        """
        alias = _GRID_ALIASES[('M', position)]
        grid = pd.date_range('2020-03-01', '2022-10-31', freq=alias)
        values = pd.Series(1.0, index=grid)
        # Ancres de chaque année, au début ou à la fin, même hors grille
        anchor_dates = pd.date_range('2020-01-01', periods=3, freq=_GRID_ALIASES[('Y', position)])
        observations = pd.Series([100.0, 120.0, 140.0], index=anchor_dates)

        # Les deux années tronquées sont nommées par un avertissement unique
        with pytest.warns(UserWarning, match=r'not enforced.*2 observed period\(s\)'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, observations, 'Y', grid_freq=alias
            )

        # Valeur d'or : 2021 → 120 / 12 = 10 par mois ; 2020 et 2022 intactes
        expected = pd.Series(1.0, index=grid)
        expected.loc['2021'] = 10.0
        pd.testing.assert_series_equal(rescaled, expected)
        assert mask.loc['2021'].all() and not mask.loc['2020'].any() and not mask.loc['2022'].any()

    def test_truncation_guard_is_opt_in(self):
        """Without grid_freq, a truncated period still receives its whole total.

        The count of expected sub-periods cannot be read off the values: the
        missing ones are ABSENT from the grid, not empty.
        """
        grid = pd.date_range('2021-03-31', periods=10, freq='ME')
        values = pd.Series(1.0, index=grid)
        observations = pd.Series(120.0, index=[grid[-1]])

        rescaled, _mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        assert rescaled.sum() == pytest.approx(120.0, abs=_EXACT)

    def test_quarter_truncated_on_a_monthly_grid(self, reference_timeseries):
        """The guard holds for every frequency pair, not only the year."""
        # Deux mois pour un trimestre de trois
        grid = reference_timeseries.loc['2021-01-31':'2021-02-28'].index
        values = pd.Series([5.0, 5.0], index=grid, name='q1')
        observations = pd.Series([np.nan, 10.0], index=grid, name='q1')

        with pytest.warns(UserWarning, match='not enforced'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, observations, 'Q', grid_freq='M'
            )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    def test_zero_total_zeroes_every_sub_period(self):
        """A zero observed total is imposed: ratio 0, sum exactly 0, no warning."""
        values = _raw_predictions()

        (rescaled, mask), messages = _record_warnings(
            AggregationConstraint('sum').rescale,
            values, _annual_observations(total=0.0), 'Y',
        )

        assert rescaled.tolist() == [0.0] * 12
        assert mask.all() and messages == []

    def test_all_zero_sub_periods_against_a_non_zero_total(self):
        """All-zero predictions: no division by zero, raw values kept, one warning."""
        values = _raw_predictions(values=[0.0] * 12)

        with warnings.catch_warnings():
            # Division par zéro proscrite, y compris sous forme de RuntimeWarning
            warnings.simplefilter('error', RuntimeWarning)
            with pytest.warns(UserWarning, match='aggregate to zero for 1 period'):
                rescaled, mask = AggregationConstraint('sum').rescale(
                    values, _annual_observations(), 'Y'
                )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    def test_cancelling_sub_periods_against_a_non_zero_total(self):
        """Predictions summing exactly to zero are left raw, with the zero warning."""
        values = _raw_predictions(values=[1.0, -1.0] * 6)

        with pytest.warns(UserWarning, match='aggregate to zero'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, _annual_observations(), 'Y'
            )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    def test_all_zero_sub_periods_against_a_zero_total(self):
        """0 against 0: the constraint already holds, nothing moves, no warning."""
        values = _raw_predictions(values=[0.0] * 12)

        (rescaled, mask), messages = _record_warnings(
            AggregationConstraint('sum').rescale,
            values, _annual_observations(total=0.0), 'Y',
        )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any() and messages == []

    def test_negative_total_with_negative_predictions(self):
        """A negative total with same-sign predictions is rescaled silently and exactly."""
        values = _raw_predictions(values=[-value for value in _RAW_2021])

        (rescaled, mask), messages = _record_warnings(
            AggregationConstraint('sum').rescale,
            values, _annual_observations(total=-120.0), 'Y',
        )

        # Valeur d'or : même ratio que l'exemple de référence, 120 / 112.5, signes conservés
        assert rescaled.sum() == pytest.approx(-120.0, abs=_EXACT)
        assert rescaled.loc['2021-12-31'] == pytest.approx(-11.2)
        assert mask.all() and messages == []

    def test_negative_total_with_positive_predictions_flips_the_profile(self):
        """Opposite signs: the constraint prevails, every sub-period changes sign."""
        values = _raw_predictions()

        with pytest.warns(UserWarning, match='flipped the predicted profile'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, _annual_observations(total=-120.0), 'Y'
            )

        assert rescaled.sum() == pytest.approx(-120.0, abs=_EXACT)
        assert (rescaled < 0).all() and mask.all()

    def test_numerically_zero_aggregate_is_zero(self):
        """A float sum that is zero up to rounding takes the zero guard.

        0.1 + 0.2 - 0.3 is mathematically zero but evaluates to 5.55e-17: the
        ratio 12 / 5.55e-17 would multiply the predictions by 2e17 and the
        rescaled sum would come back as 16, not 12. Below the rounding bound of the
        summation (n * eps * sum|x| = 3 * 2.2e-16 * 0.6, about 4e-16), the
        aggregate counts as zero: raw values kept, zero warning.
        """
        grid = pd.date_range('2021-01-31', periods=3, freq='ME')
        values = pd.Series([0.1, 0.2, -0.3], index=grid)
        observations = pd.Series([np.nan, np.nan, 12.0], index=grid)

        with pytest.warns(UserWarning, match='aggregate to zero'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, observations, 'Q', grid_freq='M'
            )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    def test_tiny_same_sign_predictions_are_still_rescaled(self):
        """The zero bound is relative: tiny predictions of one sign are not zero."""
        values = _raw_predictions(values=[1e-20] * 12)

        rescaled, mask = AggregationConstraint('sum').rescale(
            values, _annual_observations(), 'Y'
        )

        # Valeur d'or : 120 / 12 = 10 par mois
        np.testing.assert_allclose(rescaled.to_numpy(), [10.0] * 12, rtol=1e-12)
        assert mask.all()

    def test_duplicated_observation_dates_are_rejected(self):
        """A duplicated anchor raises instead of doubling the observed total.

        The two copies of the December anchor are never summed: that would
        rescale the year to 240 instead of 120, silently.
        """
        grid = _monthly_grid()
        values = pd.Series(1.0, index=grid)
        observations = pd.Series(120.0, index=grid[[-1, -1]])

        with pytest.raises(
            ValueError, match=r"Duplicate dates in observations: \[Timestamp\('2021-12-31"
        ):
            AggregationConstraint('sum').rescale(values, observations, 'Y', grid_freq='M')

    def test_duplicated_grid_dates_are_rejected(self):
        """A duplicated sub-period would count twice in the aggregate: it raises."""
        grid = _monthly_grid()
        values = pd.Series(1.0, index=grid.append(grid[:1]))

        with pytest.raises(ValueError, match='Duplicate dates in values'):
            AggregationConstraint('sum').rescale(values, _annual_observations(), 'Y')

    def test_duplicated_panel_pairs_are_rejected(self):
        """On a panel, a repeated (entity, date) pair raises; one date per entity does not."""
        values = _to_panel(_raw_predictions())
        repeated = pd.concat([values, values.iloc[:1]])

        with pytest.raises(ValueError, match=r"Duplicate dates in values: \[\('FR'"):
            AggregationConstraint('sum').rescale(
                repeated, _to_panel(_annual_observations()), 'Y'
            )

    def test_nat_rows_never_form_a_period(self):
        """A row dated NaT belongs to no period: neither rescaled nor used as a total.

        It forms no block of its own: its observation never becomes a
        "total" imposed on itself.
        """
        index = pd.DatetimeIndex(list(_monthly_grid()) + [pd.NaT])
        values = pd.Series([10.0] * 12 + [7.0], index=index)
        observations = pd.Series(np.nan, index=index)
        observations.iloc[11] = 240.0
        observations.iloc[12] = 5.0

        rescaled, mask = AggregationConstraint('sum').rescale(
            values, observations, 'Y', grid_freq='M'
        )

        # Valeur d'or : 2021 → 240 / 12 = 20 par mois ; la ligne NaT garde 7
        assert rescaled.tolist() == [20.0] * 12 + [7.0]
        assert mask.tolist() == [True] * 12 + [False]

    def test_constraint_none_leaves_raw_values(self):
        """Under None, raw predictions are untouched and the total is overwritten."""
        values = _raw_predictions()

        rescaled, mask = AggregationConstraint(None).rescale(
            values, _annual_observations(), 'Y'
        )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()
        # Le total observé n'est pas reconstitué : 112.5 et non 120
        assert rescaled.sum() == pytest.approx(112.5)

    def test_dict_form_selects_the_column(self):
        """rescale reads the column's own entry of the dict form."""
        constraint = AggregationConstraint({'a1': None, DEFAULT_CONSTRAINT_KEY: 'sum'})
        values = _raw_predictions()

        rescaled_a1, _ = constraint.rescale(values, _annual_observations(), 'Y', column='a1')
        rescaled_q1, _ = constraint.rescale(values, _annual_observations(), 'Y', column='q1')

        assert rescaled_a1.sum() == pytest.approx(112.5)
        assert rescaled_q1.sum() == pytest.approx(120.0, abs=_EXACT)

    def test_empty_grid(self):
        """An empty grid comes back empty, with an empty mask."""
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))

        rescaled, mask = AggregationConstraint('sum').rescale(empty, empty, 'Y', grid_freq='M')

        assert rescaled.empty and mask.empty

    def test_single_row_grid_is_a_truncated_period(self):
        """A single month cannot carry an annual total once the grid is declared."""
        grid = pd.DatetimeIndex(['2021-12-31'])
        values = pd.Series([3.0], index=grid)
        observations = pd.Series([120.0], index=grid)

        with pytest.warns(UserWarning, match='not enforced'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, observations, 'Y', grid_freq='M'
            )

        assert rescaled.tolist() == [3.0] and not mask.any()


class TestGridFrequencyValidation:
    """grid_freq is validated once per call, never silently ignored.

    The truncation guard only exists through ``grid_freq``; an unreadable
    frequency or a forgotten entity would switch it off and hand a
    truncated period its whole total.
    """

    def test_unreadable_grid_frequency_raises(self):
        """An unparsable grid_freq raises, naming the parameter and the value."""
        grid = pd.date_range('2021-10-31', periods=3, freq='ME')
        values = pd.Series(1.0, index=grid)
        observations = pd.Series([np.nan, np.nan, 120.0], index=grid)

        with pytest.raises(ValueError, match="grid_freq must be a readable frequency, got 'zz'"):
            AggregationConstraint('sum').rescale(values, observations, 'Y', grid_freq='zz')

    def test_unreadable_frequency_inside_the_mapping_raises(self):
        """The values of the per-entity mapping are checked too."""
        values = _to_panel(_raw_predictions())

        with pytest.raises(ValueError, match='readable frequency, got 3'):
            AggregationConstraint('sum').rescale(
                values, _to_panel(_annual_observations()), 'Y',
                grid_freq={('FR',): 'M', ('DE',): 3},
            )

    def test_entity_missing_from_the_mapping_raises(self):
        """An entity the mapping does not name raises, listing the missing entities."""
        quarters = pd.date_range('2021-03-31', periods=2, freq='QE')
        values = pd.Series(30.0, index=pd.MultiIndex.from_product(
            [['DE'], quarters], names=['country', 'date']))
        observations = pd.Series([np.nan, 120.0], index=values.index)

        with pytest.raises(ValueError, match=r"does not cover entities \[\('DE',\)\]"):
            AggregationConstraint('sum').rescale(
                values, observations, 'Y', grid_freq={('FR',): 'M'}
            )

    def test_two_level_keys_on_a_three_level_index_raise(self):
        """('FR',) does not name the entity ('Zone euro', 'FR') of a three-level index."""
        values = to_three_level_index(_to_panel(_raw_predictions()).to_frame('a1'))['a1']
        observations = to_three_level_index(
            _to_panel(_annual_observations()).to_frame('a1')
        )['a1']

        with pytest.raises(ValueError, match='does not cover entities'):
            AggregationConstraint('sum').rescale(
                values, observations, 'Y', grid_freq={('FR',): 'M', ('DE',): 'M'}
            )

    def test_scalar_entity_keys_are_normalized(self):
        """'FR' and ('FR',) designate the same entity."""
        values = _to_panel(_raw_predictions())

        _rescaled, mask = AggregationConstraint('sum').rescale(
            values, _to_panel(_annual_observations()), 'Y',
            grid_freq={'FR': 'MS', 'DE': 'ME'},
        )

        assert mask.all()

    def test_mapping_may_name_extra_entities(self):
        """Keys of entities absent from the values are harmless."""
        values = _to_panel(_raw_predictions(), entities=('FR',))

        _rescaled, mask = AggregationConstraint('sum').rescale(
            values, _to_panel(_annual_observations(), entities=('FR',)), 'Y',
            grid_freq={'FR': 'M', 'IT': 'M'},
        )

        assert mask.all()

    def test_mapping_on_a_time_series_needs_the_empty_key(self):
        """A time series is the single entity (): a mapping must name it."""
        values = _raw_predictions()

        _rescaled, mask = AggregationConstraint('sum').rescale(
            values, _annual_observations(), 'Y', grid_freq={(): 'M'}
        )
        with pytest.raises(ValueError, match=r'does not cover entities \[\(\)\]'):
            AggregationConstraint('sum').rescale(
                values, _annual_observations(), 'Y', grid_freq={'FR': 'M'}
            )

        assert mask.all()


# =============================================================================
# Recalage : panel
# =============================================================================
class TestRescalePanel:
    """Rescaling on a panel: one block per (entity, period)."""

    def test_each_entity_is_rescaled_on_its_own_total(self):
        """On a panel, FR observes 120, DE 240, blocks never mix."""
        values = _to_panel(_raw_predictions())
        observations = _to_panel(_annual_observations())
        observations.loc[('DE', pd.Timestamp('2021-12-31'))] = 240.0

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        sums = rescaled.groupby(level='country').sum()
        assert sums.to_dict() == pytest.approx({'FR': 120.0, 'DE': 240.0}, abs=_EXACT)
        assert rescaled.loc[('DE', pd.Timestamp('2021-12-31'))] == pytest.approx(22.4)
        assert mask.all()

    def test_entities_never_share_a_rescaling_block(self):
        """A partly predicted entity does not prevent the other one from being rescaled."""
        values = _to_panel(_raw_predictions())
        values.loc[('DE', pd.Timestamp('2021-05-31'))] = np.nan
        observations = _to_panel(_annual_observations())

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        assert mask.loc['FR'].all() and not mask.loc['DE'].any()
        assert rescaled.loc['FR'].sum() == pytest.approx(120.0, abs=_EXACT)

    def test_entity_without_observation_is_left_raw(self):
        """An entity that never observes the column keeps free predictions."""
        values = _to_panel(_raw_predictions())
        observations = _to_panel(_annual_observations())
        observations.loc['DE'] = np.nan

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        pd.testing.assert_series_equal(rescaled.loc['DE'], values.loc['DE'])
        assert mask.loc['FR'].all() and not mask.loc['DE'].any()

    def test_source_frequency_differing_per_entity(self):
        """FR annual, DE quarterly: one call per source-frequency group, each exact.

        The period frequency is a per-call setting; ``HighFrequencyImputer``
        groups entities by detected source frequency and calls ``rescale``
        once per group. Grid frequencies may also differ per entity.
        """
        months = _monthly_grid()
        index = pd.MultiIndex.from_product([['FR', 'DE'], months], names=['country', 'date'])
        values = pd.Series(1.0, index=index)
        observations = pd.Series(np.nan, index=index)
        # FR : un total annuel ; DE : quatre totaux trimestriels
        observations.loc[('FR', months[-1])] = 120.0
        observations.loc[[('DE', months[i]) for i in (2, 5, 8, 11)]] = [3.0, 6.0, 9.0, 12.0]

        groups = {'Y': ['FR'], 'Q': ['DE']}
        pieces = []
        for period_freq, entities in groups.items():
            rows = index.get_level_values('country').isin(entities)
            rescaled_group, _ = AggregationConstraint('sum').rescale(
                values[rows], observations[rows], period_freq,
                grid_freq={(entity,): 'M' for entity in entities},
            )
            pieces.append(rescaled_group)
        rescaled = pd.concat(pieces)

        # Valeurs d'or : FR 120 / 12 = 10 par mois ; DE 1, 2, 3, 4 par mois selon le trimestre
        expected_de = [1.0] * 3 + [2.0] * 3 + [3.0] * 3 + [4.0] * 3
        assert rescaled.loc['FR'].tolist() == [10.0] * 12
        assert rescaled.loc['DE'].tolist() == expected_de

    def test_grid_frequency_read_per_entity(self):
        """FR complete on months, DE truncated on quarters: only FR is rescaled."""
        months = _monthly_grid()
        quarters = pd.date_range('2021-03-31', periods=2, freq='QE')
        france = pd.Series(10.0, index=pd.MultiIndex.from_product(
            [['FR'], months], names=['country', 'date']))
        germany = pd.Series(30.0, index=pd.MultiIndex.from_product(
            [['DE'], quarters], names=['country', 'date']))
        values = pd.concat([france, germany])
        observations = pd.Series(np.nan, index=values.index)
        observations.loc[('FR', months[-1])] = 120.0
        observations.loc[('DE', quarters[-1])] = 120.0

        with pytest.warns(UserWarning, match=r'not enforced.*1 observed period\(s\)'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, observations, 'Y', grid_freq={('FR',): 'M', ('DE',): 'Q'}
            )

        assert mask.loc['FR'].all() and not mask.loc['DE'].any()
        assert rescaled.loc['DE'].tolist() == [30.0, 30.0]

    def test_three_level_index(self):
        """A (region, entity, date) index keeps one block per (region, entity)."""
        values = to_three_level_index(_to_panel(_raw_predictions()).to_frame('a1'))['a1']
        observations = _to_panel(_annual_observations())
        observations.loc[('DE', pd.Timestamp('2021-12-31'))] = -60.0
        observations = to_three_level_index(observations.to_frame('a1'))['a1']

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            rescaled, _mask = AggregationConstraint('sum').rescale(
                values, observations, 'Y',
                grid_freq={('Zone euro', 'FR'): 'M', ('Zone euro', 'DE'): 'M'},
            )

        sums = rescaled.groupby(level=['region', 'country']).sum()
        assert sums.to_dict() == pytest.approx(
            {('Zone euro', 'FR'): 120.0, ('Zone euro', 'DE'): -60.0}, abs=_EXACT
        )

    @pytest.mark.parametrize(
        'perturb', [lambda s: shuffle_rows(s, seed=3), reverse_entities],
        ids=['shuffle_rows', 'reverse_entities'],
    )
    def test_row_order_does_not_change_any_cell(self, perturb):
        """Shuffled rows or reversed entities: same value for every cell."""
        values = _to_panel(_raw_predictions())
        observations = _to_panel(_annual_observations())
        observations.loc[('DE', pd.Timestamp('2021-12-31'))] = 240.0
        reference, _ = AggregationConstraint('sum').rescale(
            values, observations, 'Y', grid_freq='M'
        )

        perturbed, _ = AggregationConstraint('sum').rescale(
            perturb(values), observations, 'Y', grid_freq='M'
        )

        pd.testing.assert_series_equal(perturbed.reindex(reference.index), reference)


# =============================================================================
# Recalage : jeux réalistes
# =============================================================================
class TestRescaleRealisticDatasets:
    """Properties on the realistic datasets of notebook 3."""

    @staticmethod
    def _balance_case(irregular_index_timeseries: pd.DataFrame, sign: float):
        """Return random predictions of one sign and the negative annual trade balance."""
        observations = irregular_index_timeseries['balance_commerciale_annuelle']
        rng = np.random.default_rng(11)
        values = pd.Series(
            sign * rng.uniform(0.5, 5.0, len(observations)), index=observations.index
        )
        return values, observations

    def test_negative_annual_trade_balance_is_matched_exactly(self, irregular_index_timeseries):
        """Every complete year 2018-2023 sums to its NEGATIVE observed balance.

        Predictions are positive: each year is rescaled with a sign flip.
        """
        values, observations = self._balance_case(irregular_index_timeseries, +1.0)
        assert (observations.dropna() < 0).all()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            rescaled, _mask = AggregationConstraint('sum').rescale(
                values, observations, 'Y', grid_freq='MS'
            )

        # Années complètes et observées de la grille mensuelle
        complete = observations.loc['2018':'2023'].dropna()
        sums = rescaled.groupby(rescaled.index.year).sum().loc[2018:2023]
        assert np.abs(sums.to_numpy() - complete.to_numpy()).max() <= _EXACT

    def test_isolated_anchors_and_unpublished_year_stay_raw(self, irregular_index_timeseries):
        """2015-2017 hold one row each, 2024 has no total: none of them moves."""
        values, observations = self._balance_case(irregular_index_timeseries, +1.0)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, observations, 'Y', grid_freq='MS'
            )

        untouched = mask.index.year.isin([2015, 2016, 2017, 2024])
        assert not mask[untouched].any()
        pd.testing.assert_series_equal(rescaled[untouched], values[untouched])

    @pytest.mark.parametrize('sign, expected_warnings', [(+1.0, 1), (-1.0, 0)],
                             ids=['opposite-sign', 'same-sign'])
    def test_trade_balance_warns_once_for_all_flipped_years(
        self, irregular_index_timeseries, sign, expected_warnings
    ):
        """Six flipped years produce ONE warning; same-sign predictions none.

        The years the grid holds in part — the isolated anchors before the
        monthly history, the last one after it — are named by their own
        single warning, whatever the sign.
        """
        values, observations = self._balance_case(irregular_index_timeseries, sign)

        _result, messages = _record_warnings(
            AggregationConstraint('sum').rescale,
            values, observations, 'Y', grid_freq='MS',
        )

        flipped = [m for m in messages if 'flipped the predicted profile' in m]
        assert len(flipped) == expected_warnings
        if expected_warnings:
            assert '6 period(s)' in flipped[0]
        truncated = [m for m in messages if m.startswith('Aggregation constraint not enforced')]
        assert len(truncated) == 1 and len(messages) == len(flipped) + 1

    def test_per_entity_source_frequency_on_the_heterogeneous_panel(
        self, heterogeneous_coverage_panel
    ):
        """depenses_publiques_pib: annual (France, Italie) and quarterly (Allemagne) totals.

        One call per source-frequency group, as the imputer does; every
        observed period (6 French years, 5 Italian years, 23 German quarters,
        all whole on the monthly grid) sums to its total within 1e-9.
        """
        observations = heterogeneous_coverage_panel['depenses_publiques_pib']
        rng = np.random.default_rng(5)
        values = pd.Series(rng.uniform(1.0, 20.0, len(observations)), index=observations.index)
        countries = observations.index.get_level_values('country')

        gaps, n_periods = [], 0
        for period_freq, entities in {'Y': ['France', 'Italie'], 'Q': ['Allemagne']}.items():
            rows = countries.isin(entities)
            rescaled, _mask = AggregationConstraint('sum').rescale(
                values[rows], observations[rows], period_freq, grid_freq='MS'
            )
            for entity in entities:
                observed = observations.loc[entity].dropna()
                gaps.append(_max_period_gap(rescaled.loc[entity], observed, period_freq))
                n_periods += len(observed)

        assert n_periods == 6 + 5 + 23
        assert max(gaps) <= _EXACT


# =============================================================================
# Masque des ancres
# =============================================================================
class TestAnchorCellsMask:
    """Anchor rows re-expressed at the stage frequency.

    A DIAGNOSTIC mask: it locates the observed totals overwritten by a
    sub-period value, it qualifies no provenance.
    """

    def test_exactly_the_anchor_cells(self):
        """True on the anchor rows, and only there."""
        grid = pd.date_range('2021-01-01', periods=24, freq='MS')
        observations = pd.Series(np.nan, index=grid)
        anchors = pd.DatetimeIndex(['2021-01-01', '2022-01-01'])
        observations.loc[anchors] = [100.0, -3.0]

        mask = AggregationConstraint('sum').anchor_cells_mask(observations, grid)

        assert mask[mask].index.equals(anchors)

    @pytest.mark.parametrize('setting', ['sum', None, {'a1': None}],
                             ids=['sum', 'none', 'dict'])
    def test_independent_of_the_constraint(self, reference_timeseries, setting):
        """Same mask under every setting: the three annual anchors of a1."""
        grid = reference_timeseries.index
        observations = reference_timeseries['a1']

        mask = AggregationConstraint(setting).anchor_cells_mask(observations, grid)

        assert mask[mask].index.equals(observations.dropna().index)

    def test_independent_of_the_rescaling_outcome(self):
        """A period left raw keeps its anchor located."""
        values = _raw_predictions()
        values.iloc[3] = np.nan
        observations = _annual_observations()

        constraint = AggregationConstraint('sum')
        _, rescaled_mask = constraint.rescale(values, observations, 'Y')
        anchor_mask = constraint.anchor_cells_mask(observations, values.index)

        assert not rescaled_mask.any()
        assert anchor_mask.tolist() == [False] * 11 + [True]

    def test_dataframe_input_gives_one_column_per_variable(self, reference_timeseries):
        """A frame of observations yields a boolean frame, each column its own anchors."""
        observations = reference_timeseries[['a1', 'q1', 'm1']]

        mask = AggregationConstraint('sum').anchor_cells_mask(
            observations, reference_timeseries.index
        )

        assert isinstance(mask, pd.DataFrame)
        pd.testing.assert_frame_equal(mask, observations.notna())

    def test_rows_off_the_grid_are_ignored_and_new_rows_are_false(self):
        """Anchors outside the grid vanish; grid rows without observation are False."""
        observations = pd.Series([120.0, 132.0], index=pd.to_datetime(['2020-12-31', '2021-12-31']))
        grid = _monthly_grid(2021)

        mask = AggregationConstraint('sum').anchor_cells_mask(observations, grid)

        assert mask.tolist() == [False] * 11 + [True]

    def test_no_observation_gives_an_all_false_mask(self):
        """A column without anchor on the grid yields an all-False mask."""
        grid = _monthly_grid()
        observations = pd.Series(np.nan, index=grid, name='a1')

        assert not AggregationConstraint('sum').anchor_cells_mask(observations, grid).any()

    def test_panel_heterogeneous_anchors(self, heterogeneous_coverage_panel):
        """Panel: a MultiIndex Series, True on each entity's own anchors only.

        depenses_publiques_pib: 6 French and 5 Italian years, 23 German
        quarters; climat_affaires never observed by Italie.
        """
        observations = heterogeneous_coverage_panel[['depenses_publiques_pib', 'climat_affaires']]
        index = heterogeneous_coverage_panel.index

        mask = AggregationConstraint('sum').anchor_cells_mask(observations, index)

        counts = mask['depenses_publiques_pib'].groupby(level='country').sum().to_dict()
        assert counts == {'Allemagne': 23, 'France': 6, 'Italie': 5}
        assert not mask.loc['Italie', 'climat_affaires'].any()

    def test_shuffled_grid_is_aligned_by_label(self, reference_timeseries):
        """A shuffled grid gets the same mask, row for row."""
        observations = reference_timeseries['a1']
        shuffled = shuffle_rows(reference_timeseries, seed=1).index

        mask = AggregationConstraint('sum').anchor_cells_mask(observations, shuffled)

        pd.testing.assert_series_equal(mask, observations.notna().reindex(shuffled))

    def test_duplicated_observation_dates_raise(self):
        """Duplicated observation dates raise, with the message of the rescaling."""
        grid = _monthly_grid()
        observations = pd.Series(120.0, index=grid[[-1, -1]])

        with pytest.raises(ValueError, match='Duplicate dates in observations'):
            AggregationConstraint('sum').anchor_cells_mask(observations, grid)


# =============================================================================
# Protocole sklearn
# =============================================================================
@pytest.fixture
def two_column_stage(reference_timeseries):
    """Stage frame of a1 / a2 on the 2021 months and their annual observations."""
    grid = reference_timeseries.loc['2021'].index
    X = pd.DataFrame(
        {'a1': _RAW_2021, 'a2': [value / 2 for value in _RAW_2021]}, index=grid
    )
    observations = reference_timeseries.loc['2021', ['a1', 'a2']]
    return X, observations


class TestSklearnProtocol:
    """``fit`` / ``transform`` and sklearn conventions."""

    def test_get_params_returns_the_constructor_arguments_untouched(self, two_column_stage):
        """B3: the four parameters come back as the very objects received."""
        _X, observations = two_column_stage
        setting = {'a1': 'sum', DEFAULT_CONSTRAINT_KEY: None}
        frequencies = {'a1': 'Y', 'a2': 'Y'}
        constraint = AggregationConstraint(setting, frequencies, observations, 'ctx')

        params = constraint.get_params()

        assert set(params) == {'aggregation_constraint', 'period_frequencies',
                               'observations', 'context'}
        assert (params['aggregation_constraint'] is setting
                and params['period_frequencies'] is frequencies
                and params['observations'] is observations
                and params['context'] == 'ctx')

    def test_set_params_round_trip(self):
        """set_params changes the parameter read by the resolution."""
        constraint = AggregationConstraint('sum').set_params(aggregation_constraint=None)

        assert constraint.resolve_constraint('a1') is None

    def test_clone_is_an_unfitted_equivalent(self, two_column_stage):
        """clone of a fitted constraint: same parameters, no fitted attribute."""
        X, observations = two_column_stage
        fitted = AggregationConstraint(
            {'a1': None, DEFAULT_CONSTRAINT_KEY: 'sum'}, 'Y', observations, 'ctx'
        ).fit(X)

        cloned = clone(fitted)

        assert cloned.get_params()['aggregation_constraint'] == fitted.aggregation_constraint
        pd.testing.assert_frame_equal(cloned.observations, observations)
        assert not hasattr(cloned, 'anchor_mask_')

    def test_transform_before_fit_raises_not_fitted(self, two_column_stage):
        """B20: NotFittedError before fit."""
        X, observations = two_column_stage
        constraint = AggregationConstraint('sum', 'Y', observations)

        with pytest.raises(NotFittedError):
            constraint.transform(X)

    def test_fit_returns_self_and_sets_the_fitted_attributes(self, two_column_stage):
        """fit returns self, with anchor_mask_, n_features_in_, feature_names_in_."""
        X, observations = two_column_stage
        constraint = AggregationConstraint('sum', 'Y', observations)

        assert constraint.fit(X) is constraint
        assert constraint.n_features_in_ == 2
        assert list(constraint.feature_names_in_) == ['a1', 'a2']
        assert constraint.anchor_mask_.sum().to_dict() == {'a1': 1, 'a2': 1}

    def test_fit_transform_equals_fit_then_transform(self, two_column_stage):
        """fit_transform(X) == fit(X).transform(X), masks included."""
        X, observations = two_column_stage
        chained = AggregationConstraint('sum', 'Y', observations).fit(X)
        expected = chained.transform(X)

        combined = AggregationConstraint('sum', 'Y', observations)
        result = combined.fit_transform(X)

        pd.testing.assert_frame_equal(result, expected)
        pd.testing.assert_frame_equal(combined.rescaled_mask_, chained.rescaled_mask_)

    def test_transform_rescales_every_column(self, two_column_stage):
        """Two annual columns are rescaled in one pass; X is not mutated."""
        X, observations = two_column_stage
        original = X.copy()

        rescaled = AggregationConstraint('sum', 'Y', observations).fit(X).transform(X)

        assert rescaled.sum().to_dict() == pytest.approx({'a1': 120.0, 'a2': 60.0}, abs=_EXACT)
        pd.testing.assert_frame_equal(X, original)

    def test_transform_honours_the_per_column_dict(self, two_column_stage):
        """The dict form rescales a1 and leaves a2 raw."""
        X, observations = two_column_stage
        constraint = AggregationConstraint(
            {'a1': 'sum', DEFAULT_CONSTRAINT_KEY: None}, 'Y', observations
        ).fit(X)

        rescaled = constraint.transform(X)

        pd.testing.assert_series_equal(rescaled['a2'], X['a2'])
        assert constraint.rescaled_mask_.sum().to_dict() == {'a1': 12, 'a2': 0}

    def test_transform_on_a_panel_with_per_column_frequencies(self):
        """Panel stage frame: annual a1 and quarterly q1, per-column period frequencies."""
        months = _monthly_grid()
        index = pd.MultiIndex.from_product([['FR', 'DE'], months], names=['country', 'date'])
        X = pd.DataFrame({'a1': 1.0, 'q1': 1.0}, index=index)
        observations = pd.DataFrame(np.nan, index=index, columns=['a1', 'q1'])
        observations.loc[[('FR', months[-1]), ('DE', months[-1])], 'a1'] = [120.0, 24.0]
        observations.loc[[(entity, months[i]) for entity in ('FR', 'DE') for i in (2, 5, 8, 11)],
                         'q1'] = 6.0

        rescaled = AggregationConstraint(
            'sum', {'a1': 'Y', 'q1': 'Q'}, observations
        ).fit(X).transform(X)

        # Valeurs d'or : a1 → 10 par mois (FR), 2 par mois (DE) ; q1 → 6 / 3 = 2 par mois
        assert rescaled['a1'].groupby(level='country').first().to_dict() == {'DE': 2.0, 'FR': 10.0}
        assert (rescaled['q1'] == 2.0).all()

    def test_transform_is_positional_on_a_shuffled_frame(self, two_column_stage):
        """Shuffled rows: every cell gets the same value as on the sorted frame."""
        X, observations = two_column_stage
        reference = AggregationConstraint('sum', 'Y', observations).fit(X).transform(X)
        shuffled = shuffle_rows(X, seed=2)

        result = AggregationConstraint('sum', 'Y', observations).fit_transform(shuffled)

        pd.testing.assert_frame_equal(result.loc[X.index], reference, check_freq=False)

    def test_special_column_names(self, two_column_stage):
        """Spaces, accents and symbols in column names are carried through."""
        X, observations = two_column_stage
        X_special, mapping = with_special_column_names(X)
        observations_special = observations.rename(columns=mapping)
        setting = {mapping['a1']: 'sum', DEFAULT_CONSTRAINT_KEY: None}

        rescaled = AggregationConstraint(setting, 'Y', observations_special).fit_transform(X_special)

        assert rescaled[mapping['a1']].sum() == pytest.approx(120.0, abs=_EXACT)
        pd.testing.assert_series_equal(rescaled[mapping['a2']], X_special[mapping['a2']])

    def test_constraint_none_returns_raw_values_and_an_empty_mask(self, two_column_stage):
        """Under None, transform is the identity and rescaled_mask_ is all-False."""
        X, observations = two_column_stage
        constraint = AggregationConstraint(None, 'Y', observations).fit(X)

        rescaled = constraint.transform(X)

        pd.testing.assert_frame_equal(rescaled, X)
        assert not constraint.rescaled_mask_.any().any()

    def test_series_observations_serve_a_single_column(self):
        """observations may be one Series when X holds one column."""
        grid = _monthly_grid()
        X = pd.DataFrame({'a1': _RAW_2021}, index=grid)

        rescaled = AggregationConstraint(
            'sum', 'Y', _annual_observations()
        ).fit_transform(X)

        assert rescaled['a1'].sum() == pytest.approx(120.0, abs=_EXACT)

    # Métadonnées absentes ou incomplètes : erreurs de branchement levées au fit
    _METADATA_ERRORS = {
        'no-period-frequencies': (None, 'frame', 'requires period_frequencies'),
        'period-frequencies-missing-column': ({'a1': 'Y'}, 'frame',
                                              "does not cover column 'a2'"),
        'no-observations': ('Y', None, 'requires observations'),
        'observations-missing-column': ('Y', 'a1-only', "does not hold column 'a2'"),
    }

    @pytest.mark.parametrize(
        'frequencies, observations_kind, message',
        list(_METADATA_ERRORS.values()), ids=list(_METADATA_ERRORS),
    )
    def test_missing_metadata_raises_at_fit(
        self, two_column_stage, frequencies, observations_kind, message
    ):
        """Absent or incomplete metadata is a wiring error, raised at fit."""
        X, observations = two_column_stage
        observations = {
            'frame': observations, 'a1-only': observations[['a1']], None: None,
        }[observations_kind]

        with pytest.raises(ValueError, match=message):
            AggregationConstraint('sum', frequencies, observations).fit(X)

    def test_refit_follows_the_new_grid(self, two_column_stage, reference_timeseries):
        """A second fit replaces the anchor mask with the new grid's one."""
        X, observations = two_column_stage
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)
        X_2022 = pd.DataFrame(
            {'a1': 1.0, 'a2': 1.0}, index=reference_timeseries.loc['2022'].index
        )

        constraint.set_params(observations=reference_timeseries.loc['2022', ['a1', 'a2']]).fit(X_2022)

        assert constraint.anchor_mask_.index.equals(X_2022.index)

    def test_refit_purges_the_previous_rescaled_mask(self, two_column_stage):
        """A new fit drops the mask of the previous transform (B19 principle)."""
        X, observations = two_column_stage
        constraint = AggregationConstraint('sum', 'Y', observations)
        constraint.fit_transform(X)

        constraint.fit(X)

        assert not hasattr(constraint, 'rescaled_mask_')

    @pytest.mark.parametrize(
        'columns, message',
        [(['a1'], r"missing \['a2'\], unexpected \[\]"),
         (['a1', 'a2', 'q1'], r"missing \[\], unexpected \['q1'\]")],
        ids=['missing-column', 'unexpected-column'],
    )
    def test_transform_rejects_other_columns_than_fit(
        self, two_column_stage, columns, message
    ):
        """transform refuses a frame whose columns differ from those seen at fit."""
        X, observations = two_column_stage
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)
        other = X.assign(q1=1.0)[columns]

        with pytest.raises(ValueError, match=message):
            constraint.transform(other)

    def test_transform_accepts_reordered_columns(self, two_column_stage):
        """The same columns in another order are accepted, each on its own total."""
        X, observations = two_column_stage
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)

        rescaled = constraint.transform(X[['a2', 'a1']])

        assert rescaled.sum().to_dict() == pytest.approx({'a1': 120.0, 'a2': 60.0}, abs=_EXACT)


# =============================================================================
# Avertissements agrégés et uniques
# =============================================================================
class TestWarnings:
    """One warning per degenerate case and per operation."""

    def test_nominal_rescaling_is_silent(self, two_column_stage):
        """No warning when every period is rescaled normally."""
        X, observations = two_column_stage

        _result, messages = _record_warnings(
            AggregationConstraint('sum', 'Y', observations).fit_transform, X
        )

        assert messages == []

    def test_rescale_warns_once_for_several_flipped_periods(self):
        """Two flipped years: ONE warning naming the count and the column."""
        grid = pd.date_range('2021-01-31', periods=24, freq='ME')
        values = pd.Series(-1.0, index=grid, name='a1')
        observations = pd.Series(np.nan, index=grid)
        observations.loc['2021-12-31'] = 120.0
        observations.loc['2022-12-31'] = 132.0

        _result, messages = _record_warnings(
            AggregationConstraint('sum').rescale, values, observations, 'Y'
        )

        assert len(messages) == 1
        assert "2 period(s), in column(s) ['a1'] (e.g. 2021)" in messages[0]

    def test_zero_and_flipped_cases_give_one_warning_each(self):
        """2021 aggregates to zero, 2022 is flipped: exactly two warnings, one per case."""
        grid = pd.date_range('2021-01-31', periods=24, freq='ME')
        values = pd.Series([1.0, -1.0] * 6 + [-1.0] * 12, index=grid, name='a1')
        observations = pd.Series(np.nan, index=grid)
        observations.loc['2021-12-31'] = 120.0
        observations.loc['2022-12-31'] = 132.0

        _result, messages = _record_warnings(
            AggregationConstraint('sum').rescale, values, observations, 'Y'
        )

        kinds = sorted('zero' if 'aggregate to zero' in m else 'flipped' for m in messages)
        assert kinds == ['flipped', 'zero']

    def test_truncated_periods_are_named_once(self):
        """Observed totals the grid holds in part: ONE warning, their count, the column.

        Grid March 2020 - October 2022: 2020 and 2022 hold ten months each and
        stay raw; their totals are lost in the values, which the warning
        says. The whole year 2021 is rescaled and counts for nothing.
        """
        grid = pd.date_range('2020-03-31', '2022-10-31', freq='ME')
        values = pd.Series(1.0, index=grid, name='a1')
        observations = pd.Series(
            [100.0, 120.0, 140.0], index=pd.date_range('2020-12-31', periods=3, freq='YE')
        )

        _result, messages = _record_warnings(
            AggregationConstraint('sum').rescale, values, observations, 'Y', grid_freq='M'
        )

        assert len(messages) == 1
        assert messages[0].startswith('Aggregation constraint not enforced')
        assert "2 observed period(s)" in messages[0] and "['a1']" in messages[0]

    def test_truncation_is_not_reported_without_a_grid_frequency(self):
        """Without ``grid_freq`` no truncation can be seen, hence none is reported."""
        grid = pd.date_range('2021-03-31', periods=10, freq='ME')
        values = pd.Series(1.0, index=grid, name='a1')
        observations = pd.Series(120.0, index=[grid[-1]])

        _result, messages = _record_warnings(
            AggregationConstraint('sum').rescale, values, observations, 'Y'
        )

        assert messages == []

    def test_transform_warns_once_for_every_column(self):
        """Two flipped columns: one warning naming both, with the total count."""
        grid = _monthly_grid()
        X = pd.DataFrame({'a1': [-1.0] * 12, 'a2': [-1.0] * 12}, index=grid)
        observations = pd.DataFrame(
            {'a1': [np.nan] * 11 + [120.0], 'a2': [np.nan] * 11 + [60.0]}, index=grid
        )
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)

        _result, messages = _record_warnings(constraint.transform, X)

        assert len(messages) == 1
        assert "2 period(s), in column(s) ['a1', 'a2']" in messages[0]

    def test_transform_counts_zero_periods_over_every_column(self):
        """Zero aggregates in a1 (one year) and a2 (two years): one warning, 3 periods."""
        grid = pd.date_range('2021-01-31', periods=24, freq='ME')
        X = pd.DataFrame({'a1': [0.0] * 12 + [1.0] * 12, 'a2': 0.0}, index=grid)
        observations = pd.DataFrame(np.nan, index=grid, columns=['a1', 'a2'])
        observations.loc[['2021-12-31', '2022-12-31'], :] = 12.0
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)

        _result, messages = _record_warnings(constraint.transform, X)

        assert len(messages) == 1
        assert "3 period(s) with a non-zero observed total, in column(s) ['a1', 'a2']" in messages[0]

    def test_context_is_quoted_in_the_message(self):
        """The context label follows 'for' in the message."""
        values = _raw_predictions(values=[0.0] * 12)

        _result, messages = _record_warnings(
            AggregationConstraint('sum', context="'a1' at stage M").rescale,
            values, _annual_observations(), 'Y',
        )

        assert messages[0].startswith("Aggregation constraint skipped for 'a1' at stage M:")

    def test_constraint_none_never_warns(self):
        """Under None, degenerate predictions raise no warning: nothing is rescaled."""
        values = _raw_predictions(values=[-1.0] * 12)

        _result, messages = _record_warnings(
            AggregationConstraint(None).rescale, values, _annual_observations(), 'Y'
        )

        assert messages == []


# Exécution d'une suite d'appels dans un bloc différé
def _within_deferred_block(constraint: AggregationConstraint, *calls) -> None:
    """Run each call inside one ``deferred_warnings`` block of ``constraint``.

    Args:
        constraint: Constraint whose block gathers the warnings.
        *calls: Zero-argument callables, run in order.
    """
    with constraint.deferred_warnings():
        for call in calls:
            call()


class TestDeferredWarnings:
    """``deferred_warnings``: one warning per case for a whole chain of calls.

    ``HighFrequencyImputer`` rescales every (stage, column) and every
    interpolated covariate; one warning per family is wanted, never one
    per variable × stage: the warnings accumulate, then an aggregated message
    is emitted at the end of the phase.
    """

    def test_several_rescale_calls_give_one_warning(self):
        """Two columns rescaled by two calls: ONE warning naming both, 2 periods."""
        constraint = AggregationConstraint('sum')
        flipped = _raw_predictions(values=[-1.0] * 12)

        _result, messages = _record_warnings(
            _within_deferred_block,
            constraint,
            lambda: constraint.rescale(flipped, _annual_observations(), 'Y', column='a1'),
            lambda: constraint.rescale(flipped, _annual_observations(), 'Y', column='a2'),
        )

        assert len(messages) == 1
        assert "2 period(s), in column(s) ['a1', 'a2'] (e.g. 2021)" in messages[0]

    def test_a_period_met_twice_counts_once(self):
        """The same year of the same column, rescaled at two grids, counts once.

        Valeur d'or : l'année 2021 de a1, inversée sur la grille mensuelle
        puis sur la grille trimestrielle — une seule période observée.
        """
        constraint = AggregationConstraint('sum')
        monthly = _raw_predictions(values=[-1.0] * 12)
        quarters = pd.date_range('2021-03-31', periods=4, freq='QE')
        quarterly = pd.Series(-1.0, index=quarters, name='a1')
        observations = pd.Series([np.nan, np.nan, np.nan, 120.0], index=quarters)

        _result, messages = _record_warnings(
            _within_deferred_block,
            constraint,
            lambda: constraint.rescale(monthly, _annual_observations(), 'Y', column='a1'),
            lambda: constraint.rescale(quarterly, observations, 'Y', column='a1'),
        )

        assert len(messages) == 1
        assert "1 period(s), in column(s) ['a1']" in messages[0]

    def test_both_cases_give_one_warning_each_for_the_block(self):
        """A zero aggregate in one call, a flipped profile in another: two warnings."""
        constraint = AggregationConstraint('sum')

        _result, messages = _record_warnings(
            _within_deferred_block,
            constraint,
            lambda: constraint.rescale(
                _raw_predictions(values=[0.0] * 12), _annual_observations(), 'Y', column='a1'
            ),
            lambda: constraint.rescale(
                _raw_predictions(values=[-1.0] * 12), _annual_observations(), 'Y', column='a2'
            ),
        )

        kinds = sorted('zero' if 'aggregate to zero' in m else 'flipped' for m in messages)
        assert kinds == ['flipped', 'zero']

    def test_transform_joins_the_block(self):
        """``transform`` inside the block reports to it rather than warning."""
        grid = _monthly_grid()
        X = pd.DataFrame({'a1': [-1.0] * 12}, index=grid)
        observations = pd.DataFrame({'a1': [np.nan] * 11 + [120.0]}, index=grid)
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)
        flipped = _raw_predictions(values=[-1.0] * 12).rename('a2')

        _result, messages = _record_warnings(
            _within_deferred_block,
            constraint,
            lambda: constraint.transform(X),
            lambda: constraint.rescale(flipped, _annual_observations(), 'Y', column='a2'),
        )

        assert len(messages) == 1
        assert "in column(s) ['a1', 'a2']" in messages[0]

    def test_nested_blocks_warn_once_at_the_outer_exit(self):
        """A nested block joins the outer census: nothing at the inner exit."""
        constraint = AggregationConstraint('sum')
        flipped = _raw_predictions(values=[-1.0] * 12)
        inner_messages = []

        def _nested():
            # Bloc imbriqué : rien n'est émis à sa sortie
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with constraint.deferred_warnings():
                    constraint.rescale(flipped, _annual_observations(), 'Y', column='a2')
            inner_messages.extend(str(w.message) for w in caught)

        _result, messages = _record_warnings(
            _within_deferred_block,
            constraint,
            lambda: constraint.rescale(flipped, _annual_observations(), 'Y', column='a1'),
            _nested,
        )

        assert inner_messages == []
        assert len(messages) == 1
        assert "2 period(s), in column(s) ['a1', 'a2']" in messages[0]

    def test_an_exception_drops_the_census(self):
        """A block left on an exception emits nothing, and leaves no census behind."""
        constraint = AggregationConstraint('sum')
        flipped = _raw_predictions(values=[-1.0] * 12)

        def _failing_block():
            with constraint.deferred_warnings():
                constraint.rescale(flipped, _annual_observations(), 'Y', column='a1')
                raise RuntimeError('boom')

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(RuntimeError, match='boom'):
                _failing_block()
        assert caught == []

        # Hors bloc, l'appel suivant avertit immédiatement : aucun état résiduel
        _result, messages = _record_warnings(
            constraint.rescale, flipped, _annual_observations(), 'Y', column='a1'
        )
        assert len(messages) == 1

    def test_a_silent_block_stays_silent(self, two_column_stage):
        """No degenerate period in the block: no warning at its exit."""
        X, observations = two_column_stage
        constraint = AggregationConstraint('sum', 'Y', observations).fit(X)

        _result, messages = _record_warnings(
            _within_deferred_block, constraint, lambda: constraint.transform(X)
        )

        assert messages == []

    def test_the_block_leaves_the_parameters_untouched(self):
        """``get_params`` and ``clone`` are the same before, inside and after the block."""
        constraint = AggregationConstraint({'a1': None, DEFAULT_CONSTRAINT_KEY: 'sum'})
        before = constraint.get_params()

        with constraint.deferred_warnings():
            inside = clone(constraint).get_params()

        assert inside == before
        assert constraint.get_params() == before

    def test_the_materializer_covariates_join_the_block(self):
        """Covariates interpolated through the applier report to the block.

        Deux interpolations de la même colonne, ancres 100 puis -10 : 2022
        agrège positivement face à -10 à chaque appel. Une seule période.
        """
        constraint = AggregationConstraint('sum')
        materializer = CovariateMaterializer(aggregation_constraint_applier=constraint)
        grid = pd.date_range('2021-01-31', periods=24, freq='ME')
        source = pd.DataFrame({'a2': np.nan}, index=grid)
        source.loc['2021-12-31', 'a2'] = 100.0
        source.loc['2022-12-31', 'a2'] = -10.0

        def _interpolate():
            materializer.interpolate_column(
                column='a2', grid_index=grid, stage_freq='M',
                detected_frequencies={'a2': 'Y'}, source_data=source,
            )

        _result, messages = _record_warnings(
            _within_deferred_block, constraint, _interpolate, _interpolate
        )

        assert len(messages) == 1
        assert "1 period(s), in column(s) ['a2'] (e.g. 2022)" in messages[0]
