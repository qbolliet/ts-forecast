"""``HighFrequencyImputer`` end to end on the realistic datasets of notebook 3.

Runs ``tsforecast.frequency.HighFrequencyImputer`` (``fit_transform`` then
``inverse_transform``) on ``irregular_index_timeseries`` (annual anchors
2015-2017 before a monthly grid 2018-01 -> 2024-07, genuinely irregular index)
and ``heterogeneous_coverage_panel`` (coverage specific to each entity,
``depenses_publiques_pib`` annual or quarterly by entity, ``climat_affaires``
never observed for ``Italie``), under every ``covariate_strategy`` x
``impute_intermediate_frequencies`` combination, with the default stacked
output (``keep_lower_frequencies=True``).

No gold value exists on these datasets: the tests check **properties** whose
expected value is derived from the input or from the spec
(``high_frequency_imputer2_architecture.md``), never from the output:

- observed values are never lost: untouched at the target level for the pairs
  that are not imputed, restored by ``inverse_transform`` for all (§11.2, I8);
- complete periods sum to their observed total under ``'sum'`` (I4, §11.1);
- every filled cell carries a provenance, no empty cell does, ``ORIGINAL`` only
  on input observations (§6.4, §12.4, D44);
- residual NaN only where the spec announces them: rows outside the imputation
  window, and (``Italie``, ``climat_affaires``) (§3, §4.5, §7.2);
- no (entity, column) is filled outside the periods its own observations
  cover: a short entity is never stretched (§7.1, I7);
- the ``impute_intermediate_frequencies`` axis: ``'covariates_only'`` equals
  ``False`` under ``'interpolate'`` (I12); ``*_TARGET`` / ``*_BOTH`` marks only
  under ``True`` (I6);
- one warning per family at most, at fit as at transform (§12.5);
- a transform beyond the fit never stretches the short entity (I7).

The panel cases fit in 2.5-6 s each and are marked ``slow``; the time-series
cases stay in the fast run. Formerly unit tests, moved here because they run
one component on the realistic datasets (§3 of
``tests_and_refactoring_prompts.md``):
``test_robustness.py::TestRealisticDatasets`` - its observed-cells and
period-total tests are subsumed by :class:`TestObservedValuesAreKept` and
:class:`TestPeriodTotals` (same checks, whole matrix, same 1e-9 precision),
its warning test became :class:`TestAggregatedWarnings` - and
``test_transform.py::TestShortEntityCoverage``, moved as is.
"""
# Modules de base
import re
import warnings
from typing import List, Tuple

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Matrice des scénarios et ajustements partagés (fixtures ``realistic_run`` / ``record``)
from tests.integration.frequency.realistic_runs import CASES, DATASET_FIXTURES

# Composant testé
from tsforecast.frequency import HighFrequencyImputer, ProvenanceType
from tsforecast.utils.frequency import normalize_frequency


# =============================================================================
# Matrice des scénarios (realistic_runs.py) et attentes propres aux jeux
# =============================================================================
# Familles de provenance réservées à la modalité True (I6)
_TRUE_ONLY_MARKS = {
    ProvenanceType.MODEL_ON_IMPUTED_TARGET,
    ProvenanceType.MODEL_ON_IMPUTED_BOTH,
}

# Unique couple structurellement vide des jeux réalistes (§4.5)
_STRUCTURALLY_EMPTY = {(('Italie',), 'climat_affaires')}

# Cellules du panel seulement (couple structurellement vide, entité courte)
_PANEL_CASES = [case for case in CASES if case.values[0] == 'panel']



# =============================================================================
# Helpers de lecture (niveaux de fréquence, entités)
# =============================================================================
def _levels(frame: pd.DataFrame) -> List[str]:
    """Return the frequency levels of a stacked frame, coarsest first."""
    return list(frame.index.get_level_values('frequency').unique())


def _level(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Return one frequency level of a stacked frame, without the level."""
    return frame.xs(frequency, level='frequency')


def _target_level(record: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Return the target (finest, last) level of a stacked frame."""
    return _level(frame, _levels(record['output'])[-1])


def _entities(frame: pd.DataFrame) -> List[tuple]:
    """Return the entity keys of a flat frame, ``[()]`` on a time series."""
    if not isinstance(frame.index, pd.MultiIndex):
        return [()]
    return [
        key if isinstance(key, tuple) else (key,)
        for key in frame.index.droplevel(-1).unique()
    ]


def _entity_frame(frame: pd.DataFrame, entity: tuple) -> pd.DataFrame:
    """Return the rows of one entity of a flat frame, indexed by date only."""
    if not entity:
        return frame
    return frame.xs(entity, level=list(frame.index.names[:len(entity)]))


def _pair_key(entity: tuple, column: str):
    """Return the key of an (entity, column) pair as the imputer stores it."""
    return (*entity, column) if entity else column


def _covered_bounds(series: pd.Series, frequency: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return the start of the first and the end of the last observed period.

    Args:
        series: Input column of one entity, indexed by date.
        frequency: Frequency detected for the pair.

    Returns:
        ``(start, end)`` of the span the observations cover.
    """
    base = normalize_frequency(frequency, return_format='base')
    observed = series.dropna().index
    return (
        pd.Period(observed.min(), freq=base).start_time,
        pd.Period(observed.max(), freq=base).end_time,
    )


def _marked_original(marks: pd.DataFrame) -> pd.DataFrame:
    """Return the boolean frame of the cells marked ``ORIGINAL``."""
    return marks.apply(lambda column: column == ProvenanceType.ORIGINAL)


# =============================================================================
# P1 — Valeurs observées jamais perdues
# =============================================================================
class TestObservedValuesAreKept:
    """Observed values survive: unchanged where not imputed, restored by the inversion."""

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_observations_of_non_imputed_pairs_are_untouched(
        self, record, dataset, strategy, modality
    ):
        """Every observation of a pair that is not imputed keeps its value and ``ORIGINAL``.

        The anchors of the imputed pairs are excluded: by construction their
        row receives a sub-period value (§11.2), and their totals are checked
        by :class:`TestPeriodTotals`.
        """
        data = record['data']
        values = _target_level(record, record['output'])
        marks = _target_level(record, record['provenance'])
        imputable = set(record['imputer'].variable_categories_['impute'])

        offending = []
        for entity in _entities(data):
            source = _entity_frame(data, entity)
            entity_values = _entity_frame(values, entity)
            entity_marks = _entity_frame(marks, entity)
            for column in source.columns:
                if _pair_key(entity, column) in imputable:
                    continue
                for date, value in source[column].dropna().items():
                    # Valeur d'or : la valeur d'entrée elle-même
                    if (
                        entity_values.at[date, column] != value
                        or entity_marks.at[date, column] != ProvenanceType.ORIGINAL
                    ):
                        offending.append((entity, column, date))

        assert offending == []

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_inverse_transform_restores_the_input_index(
        self, record, dataset, strategy, modality
    ):
        """``inverse_transform`` drops the grid rows and restores the input index, in order (I8)."""

        assert record['inverse'].index.equals(record['data'].index)

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_inverse_transform_restores_every_observation(
        self, record, dataset, strategy, modality
    ):
        """With ``restore_original_values=True``, every observed cell comes back exactly (I8).

        Covers the anchors of the imputed pairs too, whose total the stacked
        output only keeps as the sum of the period.
        """
        data, inverse = record['data'], record['inverse']
        observed = data.notna()

        # Comparaison restreinte aux cellules observées en entrée
        restored = inverse.reindex(index=data.index, columns=data.columns).where(observed)
        pd.testing.assert_frame_equal(restored, data.where(observed), check_freq=False)


# =============================================================================
# P2 — Totaux de période conservés sous 'sum'
# =============================================================================
def _period_gaps(record: dict) -> List[tuple]:
    """List the complete periods whose imputed sub-periods miss their total.

    Same rule as ``_period_gaps`` in the unit ``test_robustness.py``, read on
    the target level of the stacked output: a period is complete when the
    target grid holds all its sub-periods and the step wrote every one of
    them (provenance other than ``ORIGINAL``).

    Args:
        record: One record of :func:`realistic_run`.

    Returns:
        ``(column, entity, period, gap)`` for every complete period whose
        absolute gap exceeds ``1e-9`` (precision of the former unit test).
    """
    imputer = record['imputer']
    values = _target_level(record, record['output'])
    marks = _target_level(record, record['provenance'])
    # Étiquette de l'étape cible : celle de la dernière étape du plan, ordonné par étape
    target_label = imputer.imputation_plan_[-1].pred_freq_label
    gaps = []
    for step in imputer.imputation_plan_:
        # Étapes de l'étape cible seulement, hors entités sans ancre (aucun total)
        if step.unanchored or step.pred_freq_label != target_label:
            continue
        base = normalize_frequency(step.source_frequency, return_format='base')
        column = step.var_name
        for entity in (step.entities or [()]):
            observed = _entity_frame(record['data'], entity)[column].dropna()
            entity_values = _entity_frame(values, entity)[column]
            entity_marks = _entity_frame(marks, entity)[column]
            written = entity_marks.notna() & (entity_marks != ProvenanceType.ORIGINAL)
            months = entity_values.index.to_period('M')
            for date, total in observed.items():
                period = pd.Period(date, freq=base)
                first, last = period.start_time.to_period('M'), period.end_time.to_period('M')
                expected = (last - first).n + 1
                in_period = (months >= first) & (months <= last)
                # Période tronquée par la grille ou la fenêtre : hors contrôle (§11.1)
                if in_period.sum() != expected or not written[in_period].all():
                    continue
                gap = abs(float(entity_values[in_period].sum()) - float(total))
                if gap > 1e-9:
                    gaps.append((column, entity, str(period), gap))
    return gaps


class TestPeriodTotals:
    """I4 — under ``aggregation_constraint='sum'``, complete periods keep their total."""

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_complete_periods_sum_to_their_observed_total(
        self, record, dataset, strategy, modality
    ):
        """Each complete period of an imputed pair sums to its observed total (1e-9).

        True of model predictions and of interpolation fallbacks alike, both
        rescaled (§11.1). Periods the grid or the window truncates - the
        isolated annual anchors before the monthly grid - are not constrained.
        """

        assert _period_gaps(record) == []

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_at_least_one_period_is_checked(self, record, dataset, strategy, modality):
        """The total check is not vacuous: the target stage writes cells of every imputed pair."""
        marks = _target_level(record, record['provenance'])
        imputed_columns = {
            key[-1] if isinstance(key, tuple) else key
            for key in record['imputer'].variable_categories_['impute']
        }

        # Chaque colonne imputée porte au moins une cellule écrite au niveau cible
        written = {
            column: int((marks[column].notna() & (marks[column] != ProvenanceType.ORIGINAL)).sum())
            for column in imputed_columns
        }
        assert min(written.values()) > 0


# =============================================================================
# P3 — Provenance complète
# =============================================================================
class TestProvenanceCoverage:
    """Every filled cell is marked, no empty cell is, ``ORIGINAL`` only on observations."""

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_every_filled_cell_has_a_provenance(self, record, dataset, strategy, modality):
        """On every frequency level, a non-NaN value always carries a mark (§6.4)."""
        output, provenance = record['output'], record['provenance']

        unmarked = int((output.notna() & provenance.isna()).sum().sum())

        assert unmarked == 0

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_no_empty_cell_is_marked(self, record, dataset, strategy, modality):
        """On every frequency level, a NaN value never carries a mark (§12.4)."""
        output, provenance = record['output'], record['provenance']

        marked_empty = int((output.isna() & provenance.notna()).sum().sum())

        assert marked_empty == 0

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_original_marks_only_input_observations(
        self, record, dataset, strategy, modality
    ):
        """At the target level, ``ORIGINAL`` is reserved to cells observed in the input.

        Grid rows added by the imputer and imputed anchors never claim to be
        observations.
        """
        data = record['data']
        marks = _target_level(record, record['provenance'])
        observed = data.notna().reindex(index=marks.index, columns=marks.columns, fill_value=False)

        claimed = int((_marked_original(marks) & ~observed).sum().sum())

        assert claimed == 0


# =============================================================================
# P4 — NaN résiduels seulement là où la spec les annonce
# =============================================================================
class TestResidualNaN:
    """§3 / §4.5 / §7.2 — where the output may still hold NaN, measured per entity."""

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_window_rows_are_filled_except_structurally_empty_pairs(
        self, record, dataset, strategy, modality
    ):
        """Inside the imputation window, the only NaN left are those of (``Italie``, ``climat_affaires``).

        The structurally empty pair is the single exception of §3, measured per
        entity (§4.5, ``covariate_eligibility='any_entity'``).
        """
        values = _target_level(record, record['output'])
        mask = record['imputer'].imputation_window_mask_
        in_window = values.loc[mask.index[mask.to_numpy()]]

        residual = set()
        for entity in _entities(in_window):
            frame = _entity_frame(in_window, entity)
            for column in frame.columns[frame.isna().any().to_numpy()]:
                residual.add((entity, column))

        expected = _STRUCTURALLY_EMPTY if dataset == 'panel' else set()
        assert residual == expected

    @pytest.mark.parametrize('dataset, strategy, modality', _PANEL_CASES)
    def test_structurally_empty_pair_stays_entirely_nan(
        self, record, dataset, strategy, modality
    ):
        """No strategy fabricates values for an entity that never observes a column (§4.5)."""

        filled = int(record['output'].xs('Italie', level='country')['climat_affaires'].notna().sum())

        assert filled == 0

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_input_rows_outside_the_window_are_left_as_is(
        self, record, dataset, strategy, modality
    ):
        """Input rows outside the imputation window keep their input values, NaN included (§7.2).

        I7: no input observation is ever destroyed, under every strategy.
        Under ``'model'`` the stage frame serves an imputed column from the
        mirror of the produced cells, which overlays its raw anchors without
        replacing them (formerly ANO-FREQ-083: anchors erased outside the window).
        """
        data = record['data']
        values = _target_level(record, record['output'])
        mask = record['imputer'].imputation_window_mask_
        outside = mask.index[~mask.to_numpy()]

        pd.testing.assert_frame_equal(
            values.loc[outside, data.columns], data.loc[outside], check_freq=False
        )

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_added_grid_rows_stay_empty(self, record, dataset, strategy, modality):
        """Rows the regular grid adds outside the input index carry no value.

        On these datasets they all fall before the monthly grid (months of the
        isolated annual anchors), hence outside the imputation window.
        """
        values = _target_level(record, record['output'])
        added = values.index.difference(record['data'].index)

        # Les lignes ajoutées existent bien (index irrégulier en entrée)
        assert len(added) > 0
        assert int(values.loc[added].notna().sum().sum()) == 0


# =============================================================================
# P5 — Couverture courte jamais prolongée
# =============================================================================
class TestCoverageIsNotStretched:
    """§7.1 / I7 — no (entity, column) is filled outside the periods its observations cover."""

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_no_cell_is_filled_outside_its_own_coverage(
        self, record, dataset, strategy, modality
    ):
        """Nothing before the first observed period nor after the last one, pair by pair.

        A yearly value dated 2023-01-01 covers 2023, a quarterly one dated
        2024-01-01 covers 2024-Q1; beyond, a value would be an extrapolation
        that ``imputation_scope='strict'`` (default) forbids.
        """
        data = record['data']
        values = _target_level(record, record['output'])
        detected = record['imputer'].detected_frequencies_

        stretched = {}
        for entity in _entities(data):
            source = _entity_frame(data, entity)
            output = _entity_frame(values, entity)
            for column in source.columns:
                frequency = detected.get(_pair_key(entity, column))
                # Couple jamais observé ou de fréquence indétectable : rien à borner
                if frequency is None or source[column].notna().sum() == 0:
                    continue
                start, end = _covered_bounds(source[column], frequency)
                filled = output[column].dropna().index
                outside = int(((filled < start) | (filled > end)).sum())
                if outside:
                    stretched[(entity, column)] = outside

        assert stretched == {}

    @pytest.mark.parametrize('dataset, strategy, modality', _PANEL_CASES)
    def test_short_entity_ends_where_its_input_ends(
        self, record, dataset, strategy, modality
    ):
        """Allemagne's grid stops at 2024-04-01, three months before the other entities."""
        values = _target_level(record, record['output'])

        last_dates = {
            entity: values.xs(entity, level='country').index.max()
            for entity in ('Allemagne', 'France', 'Italie')
        }

        assert last_dates == {
            'Allemagne': pd.Timestamp('2024-04-01'),
            'France': pd.Timestamp('2024-07-01'),
            'Italie': pd.Timestamp('2024-07-01'),
        }


# =============================================================================
# P6 — Axe impute_intermediate_frequencies
# =============================================================================
# Paires (False, 'covariates_only') comparables : stratégie 'interpolate' (I12)
_INTERPOLATE_PAIRS = [
    pytest.param(dataset, marks=[pytest.mark.slow] if dataset == 'panel' else [], id=dataset)
    for dataset in DATASET_FIXTURES
]


class TestIntermediateFrequencyAxis:
    """I12 / I6 — what the ``impute_intermediate_frequencies`` modality changes."""

    @pytest.mark.parametrize('dataset', _INTERPOLATE_PAIRS)
    def test_covariates_only_equals_false_under_interpolate(self, realistic_run, dataset):
        """Under ``'interpolate'``, ``'covariates_only'`` changes no target value (I12, §5.6)."""
        direct = realistic_run(dataset, 'interpolate', False)
        walked = realistic_run(dataset, 'interpolate', 'covariates_only')

        pd.testing.assert_frame_equal(
            _target_level(walked, walked['output']),
            _target_level(direct, direct['output']),
            check_freq=False,
        )

    @pytest.mark.parametrize('dataset', _INTERPOLATE_PAIRS)
    def test_covariates_only_marks_equal_false_under_interpolate(self, realistic_run, dataset):
        """Under ``'interpolate'``, ``'covariates_only'`` changes no target provenance either."""
        direct = realistic_run(dataset, 'interpolate', False)
        walked = realistic_run(dataset, 'interpolate', 'covariates_only')

        pd.testing.assert_frame_equal(
            _target_level(walked, walked['provenance']),
            _target_level(direct, direct['provenance']),
            check_freq=False,
        )

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_imputed_target_marks_only_under_true(self, record, dataset, strategy, modality):
        """``MODEL_ON_IMPUTED_TARGET`` / ``_BOTH`` appear only when ``y_train`` holds imputed cells (I6)."""
        marks = set(pd.unique(record['provenance'].to_numpy().ravel()))

        # Sous True, ces familles sont permises ; ailleurs, interdites
        allowed = _TRUE_ONLY_MARKS if modality is True else set()
        assert (marks & _TRUE_ONLY_MARKS) <= allowed

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_intermediate_levels_follow_the_modality(
        self, record, dataset, strategy, modality
    ):
        """``False`` emits the target level alone; the two other modalities walk the quarter first."""

        expected = ['M'] if modality is False else ['Q', 'M']
        assert _levels(record['output']) == expected


# =============================================================================
# Avertissements agrégés (§12.5) — déplacé de tests/unit/.../test_robustness.py
# =============================================================================
def _family(message: str) -> str:
    """Reduce a warning message to its family.

    The family is the text before the first colon, every number replaced by
    ``#``: two messages of one family differ only by their counts and the
    names they list.

    Args:
        message: Warning message.

    Returns:
        The family label.

    Examples:
        >>> _family("3 imputation step(s) degraded during the fit:\n  - 'a1'")
        '# imputation step(s) degraded during the fit'
    """
    return re.sub(r'\d+', '#', message.split(':')[0])


class TestAggregatedWarnings:
    """§12.5 — one warning per family at most, at fit as at transform."""

    @pytest.mark.parametrize('dataset, strategy, modality', CASES)
    def test_one_warning_per_family_at_most(self, record, dataset, strategy, modality):
        """No warning family is emitted twice by one pass (ANO-FREQ-059).

        Formerly ``test_robustness.py::TestRealisticDatasets`` (three
        strategies, ``impute_intermediate_frequencies=False``), now on the
        whole matrix.
        """
        repeated = {
            phase: [
                family for family, count in pd.Series(
                    [_family(message) for message in record[f'{phase}_messages']],
                    dtype=object,
                ).value_counts().items()
                if count > 1
            ]
            for phase in ('fit', 'transform')
        }

        assert repeated == {'fit': [], 'transform': []}


# =============================================================================
# Transform au-delà du fit sur le panel hétérogène (I7)
# — déplacé de tests/unit/.../test_transform.py
# =============================================================================
@pytest.mark.slow
class TestShortEntityCoverage:
    """I7 on the realistic panel — a short entity is never stretched past its coverage.

    ``heterogeneous_coverage_panel``: France and Italie run to 2024-07,
    Allemagne stops at 2024-04. The imputer is fitted on the data up to
    2023-12 and transforms the whole panel, so every 2024 row is new.
    """

    @pytest.fixture(scope='class', params=['interpolate', 'model'])
    def replay(self, request, _heterogeneous_coverage_panel_session):
        """Fit up to 2023-12, transform everything: ``(input, output, provenance, imputer)``."""
        data = _heterogeneous_coverage_panel_session.copy()
        dates = data.index.get_level_values('date')
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            covariate_strategy=request.param,
            keep_lower_frequencies=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            imputer.fit(data[dates <= '2023-12-31'])
            output = imputer.transform(data)
        return data, output, imputer.imputation_provenance_, imputer

    def test_short_entity_ends_where_its_input_ends(self, replay):
        """Allemagne's output stops at 2024-04-01, France's goes on to 2024-07-01."""
        data, output, _provenance, _imputer = replay
        last_dates = {
            entity: output.loc[entity].index.max() for entity in ('Allemagne', 'France')
        }
        assert last_dates == {
            'Allemagne': data.loc['Allemagne'].index.max(),
            'France': pd.Timestamp('2024-07-01'),
        }
        assert last_dates['Allemagne'] == pd.Timestamp('2024-04-01')

    def test_no_column_is_filled_past_its_coverage(self, replay):
        """No cell of Allemagne is filled after the period its column's last observation covers.

        A yearly value dated 2023-01-01 covers 2023 (period-start index); a
        quarterly one dated 2024-01-01 covers the first quarter of 2024.
        Beyond, a value would be an extrapolation no provenance announces.
        """
        data, output, _provenance, imputer = replay
        source = data.loc['Allemagne']
        filled_beyond = {}
        for column in source.columns:
            last = source[column].last_valid_index()
            detected = imputer.detected_frequencies_.get(('Allemagne', column))
            # Colonne jamais observée ou de fréquence indétectable : rien à couvrir
            if last is None or detected is None:
                continue
            frequency = normalize_frequency(detected, return_format='base')
            covered_until = pd.Period(last, freq=frequency).end_time
            beyond = output.loc['Allemagne', column].loc[lambda s: s.index > covered_until]
            filled_beyond[column] = int(beyond.notna().sum())
        # Toutes les colonnes observées sont contrôlées, aucune n'est débordée
        assert len(filled_beyond) == int(source.notna().any().sum())
        assert set(filled_beyond.values()) == {0}

    def test_every_filled_cell_of_the_short_entity_has_its_provenance(self, replay):
        """Each filled cell of Allemagne is marked, ``ORIGINAL`` only where the input observed it."""
        data, output, provenance, _imputer = replay
        values = output.loc['Allemagne']
        marks = provenance.loc['Allemagne'].reindex(index=values.index, columns=values.columns)
        observed = data.loc['Allemagne'].reindex(index=values.index, columns=values.columns).notna()

        # Aucune valeur sans provenance
        assert not (values.notna() & marks.isna()).any().any()
        # ORIGINAL réservé aux cellules observées en entrée
        is_original = marks.apply(lambda column: column == ProvenanceType.ORIGINAL)
        assert not (is_original & ~observed).any().any()
