"""``StageScaler`` on panels.

Scenarios: per-entity frequency bindings, the mutualized training set with one block
frequency per entity, and the realistic datasets.
"""

# Modules de base
import calendar

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objets testés
from tsforecast.imputation._components.stage_scaler import StageScaler

# Perturbations partagées
from tests.support.perturbations import to_three_level_index

# Outils de test partagés du paquet
from tests.unit.imputation._components.stage_scaler.helpers import (
    _EXACT,
    _PANEL_F_BLOCKS,
    _monthly_grid,
    _panel_grid,
)


# Fréquence de publication de depenses_publiques_pib par entité (jeu du notebook 3)
_DEPENSES_BLOCKS = {('France',): 'Y', ('Allemagne',): 'Q', ('Italie',): 'Y'}


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
        """The scale of a block is never guessed: an uncovered grid entity is an error naming it."""
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

        Scalar keys are normalized, as in ``CovariateMaterializer``: both components read the same bindings.
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
# Jeu mutualisé PANEL-F
# =============================================================================
class TestMutualizedTrainingSet:
    """Per-entity source_freq / grid_freq, one block frequency per entity."""

    @pytest.mark.parametrize(
        'stage, expected',
        [
            # Valeur d'or du tableau de l'étape Q : FR / 4, DE / 1, IT / (1/3)
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
        """A monthly block at a quarterly stage divides by 1/3, by every route.

        Block divisor, production frequency and covariate finer than
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
