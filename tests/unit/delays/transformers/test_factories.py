"""Unit tests for the per-entity factories of ``tsforecast.delays.transformers``.

Covers ``create_delay_transformer_factory`` and ``prepare_entity_kwargs_from_delays``
and their helpers ``_build_entity_params``, ``_extract_param_by_variable`` and
``_resolve_strategy``, all exercised through the two public functions (the factory
called with an entity key, or the ``entity_kwargs`` dictionary), then end to end
with ``PanelwiseTransformer``. Also keeps ``_detect_index_components`` (index
components read by ``ShiftTransformer`` / ``MaskTransformer``, moved here by
prompt D3).

Gold values: panel of two entities (``FR``, ``DE``) of monthly 2023 data,
prediction date 2023-12-15 (14 days elapsed in December, 30 days per month).
Delays in days from the period start: ``FR`` GDP 45 -> shift -2, ``FR`` CPI 20
-> -1, ``DE`` GDP 75 -> -ceil(61 / 30) = -3, ``DE`` CPI 20 -> -1.

Triage (prompt D4): the former tests of ``_extract_param_by_variable`` and
``_resolve_strategy`` called the private helpers directly although both are
reachable through the public functions; they were rewritten on the public path
with the same intentions (constant vs varying parameter, every form of strategy).
"""
# Modules de base
import doctest
import warnings

import numpy as np
import pandas as pd
import pytest

# Fonctions à tester et collaborateur réel pour les panels
from tsforecast.delays.transformers import (
    PublicationDelayTransformer,
    _detect_index_components,
    create_delay_transformer_factory,
    prepare_entity_kwargs_from_delays,
)
from tsforecast.delays import transformers as transformers_module
from tsforecast.panel import PanelwiseTransformer

TS = pd.Timestamp
PREDICTION = '2023-12-15'


# =============================================================================
# Constructeurs locaux
# =============================================================================
def _monthly() -> pd.DataFrame:
    """Monthly frame of 2023: ``GDP = 0..11``, ``CPI = 0, 2, ..., 22``."""
    index = pd.date_range('2023-01-01', periods=12, freq='MS')
    return pd.DataFrame({'GDP': np.arange(12.0), 'CPI': np.arange(12.0) * 2}, index=index)


def _panel(entities=('FR', 'DE')) -> pd.DataFrame:
    """Panel (country, date): entity ``i`` holds ``_monthly() + 100 * i``."""
    return pd.concat({entity: _monthly() + 100 * i for i, entity in enumerate(entities)}, names=['country', 'date'])


def _entity_delays(rows=None, frequency='M', reference_point='start') -> pd.DataFrame:
    """Per-entity delays table, as returned by ``calculate_applicable_delay(aggregate_by_panel=True)``.

    Args:
        rows: ``{(entity, column): delay}``; the gold delays by default.
        frequency: Target frequency of every row (or a ``{column: frequency}`` mapping).
        reference_point: Reference point of every row (or a ``{column: reference_point}`` mapping).

    Returns:
        Table indexed by (country, column) with the columns ``delay``, ``unit``, ``frequency``,
        ``reference_point``.
    """
    rows = rows or {('FR', 'GDP'): 45.0, ('FR', 'CPI'): 20.0, ('DE', 'GDP'): 75.0, ('DE', 'CPI'): 20.0}
    index = pd.MultiIndex.from_tuples(list(rows), names=['country', 'column'])

    def per_column(value):
        return [value[col] if isinstance(value, dict) else value for _, col in rows]

    return pd.DataFrame({'delay': list(rows.values()), 'unit': 'day', 'frequency': per_column(frequency),
                         'reference_point': per_column(reference_point)}, index=index)


def _panelwise(transformer, **kwargs) -> PanelwiseTransformer:
    """Wrap a transformer (or a factory) for a panel whose entity and date are index levels."""
    return PanelwiseTransformer(transformer=transformer, time_col=None, panel_cols=None, **kwargs)


def _fit_transform(transformer, X: pd.DataFrame) -> pd.DataFrame:
    """Fit and transform with the warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return transformer.fit_transform(X)


# =============================================================================
# Composantes de l'index (déplacé tel quel par D3)
# =============================================================================
class TestMultipliedIndexFrequency:
    """Les transformateurs décalent par périodes d'index entières : un index multiplié est rejeté."""

    def test_detect_index_components(self):
        """Base, position et ancre de l'index (sans multiplicateur)."""
        assert _detect_index_components(pd.date_range('2024-01-01', periods=6, freq='QS')) == ('Q', 'S', 'JAN')

    def test_multiplied_index_is_rejected(self):
        """Un index '2MS' ne doit pas être traité comme mensuel."""
        with pytest.raises(ValueError, match="Multiplied index frequency"):
            _detect_index_components(pd.date_range('2024-01-01', periods=6, freq='2MS'))


# =============================================================================
# Exemples des docstrings du module
# =============================================================================
class TestDocstringExamples:
    """The examples of the module docstrings (transformers, factories and helpers) run as written."""

    def test_docstring_examples_run(self):
        """``doctest`` finds examples in ``tsforecast.delays.transformers`` and all of them pass."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = doctest.testmod(transformers_module, optionflags=doctest.ELLIPSIS)
        assert results.attempted > 0 and results.failed == 0


# =============================================================================
# create_delay_transformer_factory
# =============================================================================
class TestCreateDelayTransformerFactory:
    """The factory builds one configured ``PublicationDelayTransformer`` per entity key."""

    def test_entity_transformer_is_configured_from_its_rows(self):
        """Delays, unit and reference point come from the rows of the entity; no target frequency for the shift."""
        transformer = create_delay_transformer_factory(_entity_delays(), prediction_date=PREDICTION)(('DE',))
        params = transformer.get_params()
        assert {key: params[key] for key in ('delays', 'delay_unit', 'reference_point', 'target_frequency',
                                             'strategy', 'prediction_date')} == {
            'delays': {'GDP': 75.0, 'CPI': 20.0}, 'delay_unit': 'day', 'reference_point': 'start',
            'target_frequency': None, 'strategy': 'shift', 'prediction_date': PREDICTION}

    def test_mask_transformer_gets_the_target_frequency(self):
        """With the 'mask' strategy, the target frequency of the rows is passed."""
        transformer = create_delay_transformer_factory(_entity_delays(), strategy='mask',
                                                       prediction_date=PREDICTION)(('DE',))
        assert transformer.target_frequency == 'M'

    def test_parameter_varying_by_variable_becomes_a_dict(self):
        """A parameter constant within an entity is a scalar, a varying one a ``{variable: value}`` dict."""
        delays = _entity_delays(frequency={'GDP': 'Q', 'CPI': 'M'})
        transformer = create_delay_transformer_factory(delays, strategy='mask', prediction_date=PREDICTION)(('FR',))
        assert (transformer.delay_unit, transformer.target_frequency) == ('day', {'GDP': 'Q', 'CPI': 'M'})

    def test_scalar_and_tuple_keys_are_equivalent(self):
        """``factory('FR')`` and ``factory(('FR',))`` build the same transformer."""
        factory = create_delay_transformer_factory(_entity_delays(), prediction_date=PREDICTION)
        assert factory('FR').get_params() == factory(('FR',)).get_params()

    def test_each_call_builds_a_new_transformer(self):
        """Two calls never share a transformer (``PanelwiseTransformer`` fits one per entity)."""
        factory = create_delay_transformer_factory(_entity_delays(), prediction_date=PREDICTION)
        assert factory(('FR',)) is not factory(('FR',))

    def test_unknown_entity_raises_a_key_error_listing_the_entities(self):
        """An entity absent from the table is reported with the available ones."""
        factory = create_delay_transformer_factory(_entity_delays(), prediction_date=PREDICTION)
        with pytest.raises(KeyError, match=r"\('IT',\) not found.*\('DE',\), \('FR',\)"):
            factory(('IT',))

    def test_more_than_ten_entities_are_summarized(self):
        """The error lists ten entities and counts the others."""
        rows = {(f'E{i:02d}', 'GDP'): 45.0 for i in range(12)}
        factory = create_delay_transformer_factory(_entity_delays(rows), prediction_date=PREDICTION)
        with pytest.raises(KeyError, match=r"\.\.\. and 2 more"):
            factory(('IT',))

    def test_default_transformer_kwargs_are_passed(self):
        """``default_transformer_kwargs`` reach every transformer; the factory's prediction date wins."""
        factory = create_delay_transformer_factory(
            _entity_delays(), prediction_date=PREDICTION,
            default_transformer_kwargs={'handle_missing_delays': 'ignore', 'prediction_date': '2000-01-01'})
        transformer = factory(('FR',))
        assert (transformer.handle_missing_delays, transformer.prediction_date) == ('ignore', PREDICTION)

    def test_custom_column_names(self):
        """Other column names are read through ``*_col``."""
        delays = _entity_delays().rename(columns={'delay': 'lag', 'unit': 'u', 'reference_point': 'rp',
                                                  'frequency': 'f'})
        factory = create_delay_transformer_factory(delays, prediction_date=PREDICTION, delay_col='lag', unit_col='u',
                                                   reference_point_col='rp', target_frequency_col='f')
        assert factory(('FR',)).delays == {'GDP': 45.0, 'CPI': 20.0}

    def test_missing_columns_are_rejected(self):
        """A table without one of the required columns is rejected at creation."""
        with pytest.raises(ValueError, match=r"Missing required columns in df_delays: \['unit'\]"):
            create_delay_transformer_factory(_entity_delays().drop(columns='unit'))

    def test_flat_index_is_rejected(self):
        """A table without entity level is rejected at creation."""
        with pytest.raises(ValueError, match='MultiIndex'):
            create_delay_transformer_factory(_entity_delays().droplevel('country'))

    def test_two_entity_levels(self):
        """With two entity levels (region, country), entity keys are pairs."""
        delays = pd.concat({'EU': _entity_delays()}, names=['region'])
        factory = create_delay_transformer_factory(delays, prediction_date=PREDICTION)
        assert factory(('EU', 'DE')).delays == {'GDP': 75.0, 'CPI': 20.0}

    def test_reference_point_varying_by_variable(self):
        """A reference point varying by variable gives a per-variable dict accepted by the transformer."""
        delays = _entity_delays(reference_point={'GDP': 'start', 'CPI': 'end'})
        transformer = create_delay_transformer_factory(delays, prediction_date=PREDICTION)(('FR',))
        assert transformer.reference_point == {'GDP': 'start', 'CPI': 'end'}

    def test_shift_factory_emits_no_warning(self):
        """Building a 'shift' transformer from a complete table needs no warning."""
        factory = create_delay_transformer_factory(_entity_delays(), prediction_date=PREDICTION)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            factory(('FR',))


# =============================================================================
# Stratégie par entité (_resolve_strategy, via la fabrique)
# =============================================================================
class TestStrategyResolution:
    """Every accepted form of ``strategy``, resolved for one entity."""

    @staticmethod
    def _strategy_of(strategy, entity_key=('FR',)):
        """Return the strategy of the transformer built for ``entity_key``."""
        return create_delay_transformer_factory(_entity_delays(), strategy=strategy,
                                                prediction_date=PREDICTION)(entity_key).strategy

    def test_global_string(self):
        """A string applies to every entity."""
        assert (self._strategy_of('mask', ('FR',)), self._strategy_of('mask', ('DE',))) == ('mask', 'mask')

    @pytest.mark.parametrize(
        'strategy',
        [pytest.param({('FR',): 'shift', ('DE',): 'mask'}, id='tuple-keys'),
         pytest.param({'FR': 'shift', 'DE': 'mask'}, id='scalar-keys')],
    )
    def test_per_entity_dict(self, strategy):
        """A per-entity dict, keyed by tuples or by scalars for one entity level."""
        assert (self._strategy_of(strategy, ('FR',)), self._strategy_of(strategy, ('DE',))) == ('shift', 'mask')

    def test_per_variable_dict_is_passed_on(self):
        """A dict keyed by variable names (no entity) is passed as is to every transformer."""
        strategy = {'GDP': 'shift', 'CPI': 'mask'}
        assert self._strategy_of(strategy) == strategy

    def test_callable(self):
        """A callable receives the entity key."""
        def selector(entity_key):
            return 'shift' if 'FR' in entity_key else 'mask'
        assert (self._strategy_of(selector, ('FR',)), self._strategy_of(selector, ('DE',))) == ('shift', 'mask')

    @pytest.mark.parametrize(
        ('strategy', 'error', 'match'),
        [
            pytest.param('invalid', ValueError, "Invalid strategy: 'invalid'", id='unknown-string'),
            pytest.param(lambda key: 'later', ValueError, "returned invalid value 'later'", id='callable-bad-value'),
            pytest.param(123, TypeError, 'strategy must be str, dict, or callable', id='bad-type'),
            pytest.param({('DE',): 'mask'}, KeyError, r"No strategy defined for entity \('FR',\)",
                         id='entity-missing-from-dict'),
        ],
    )
    def test_invalid_strategies(self, strategy, error, match):
        """Invalid strategies are rejected when the entity transformer is built."""
        with pytest.raises(error, match=match):
            self._strategy_of(strategy, ('FR',))


# =============================================================================
# prepare_entity_kwargs_from_delays
# =============================================================================
class TestPrepareEntityKwargs:
    """``entity_kwargs`` hold, per entity, the same configuration as the factory."""

    def test_kwargs_of_each_entity(self):
        """One kwargs dict per entity, built from its rows."""
        assert prepare_entity_kwargs_from_delays(_entity_delays()) == {
            ('FR',): {'delays': {'GDP': 45.0, 'CPI': 20.0}, 'delay_unit': 'day', 'reference_point': 'start',
                      'target_frequency': None, 'strategy': 'shift'},
            ('DE',): {'delays': {'GDP': 75.0, 'CPI': 20.0}, 'delay_unit': 'day', 'reference_point': 'start',
                      'target_frequency': None, 'strategy': 'shift'},
        }

    def test_same_configuration_as_the_factory(self):
        """Each kwargs dict matches the parameters of the transformer built by the factory."""
        delays = _entity_delays(frequency={'GDP': 'Q', 'CPI': 'M'})
        factory = create_delay_transformer_factory(delays, prediction_date=PREDICTION)
        for entity, kwargs in prepare_entity_kwargs_from_delays(delays).items():
            params = factory(entity).get_params()
            assert {key: params[key] for key in kwargs} == kwargs, entity

    def test_per_entity_strategy(self):
        """A per-entity strategy dict is resolved entity by entity."""
        kwargs = prepare_entity_kwargs_from_delays(_entity_delays(), strategy={'FR': 'shift', 'DE': 'mask'})
        assert {entity: kw['strategy'] for entity, kw in kwargs.items()} == {('FR',): 'shift', ('DE',): 'mask'}

    def test_two_entity_levels(self):
        """With two entity levels, keys are pairs."""
        delays = pd.concat({'EU': _entity_delays()}, names=['region'])
        assert set(prepare_entity_kwargs_from_delays(delays)) == {('EU', 'FR'), ('EU', 'DE')}

    def test_missing_columns_are_rejected(self):
        """A table without one of the required columns is rejected."""
        with pytest.raises(ValueError, match=r"Missing required columns in df_delays: \['frequency'\]"):
            prepare_entity_kwargs_from_delays(_entity_delays().drop(columns='frequency'))


# =============================================================================
# De bout en bout avec PanelwiseTransformer
# =============================================================================
class TestPanelwiseIntegration:
    """Factory or ``entity_kwargs`` -> ``PanelwiseTransformer`` -> per-entity delays."""

    @pytest.mark.parametrize(
        ('entity', 'column', 'last_date', 'last_value'),
        [
            # Valeurs d'or : décembre (11 / 22, +100 pour DE) déplacé de -n_periods mois
            pytest.param('FR', 'GDP', '2024-02-01', 11.0, id='FR-GDP-shift-2'),
            pytest.param('FR', 'CPI', '2024-01-01', 22.0, id='FR-CPI-shift-1'),
            pytest.param('DE', 'GDP', '2024-03-01', 111.0, id='DE-GDP-shift-3'),
            pytest.param('DE', 'CPI', '2024-01-01', 122.0, id='DE-CPI-shift-1'),
        ],
    )
    def test_factory_applies_the_delays_of_each_entity(self, entity, column, last_date, last_value):
        """The last value of each (entity, column) moves by the shift of its own delay."""
        factory = create_delay_transformer_factory(_entity_delays(), prediction_date=PREDICTION)
        series = _fit_transform(_panelwise(factory), _panel()).loc[entity, column]
        assert (series.last_valid_index(), series[series.last_valid_index()]) == (TS(last_date), last_value)

    def test_entity_kwargs_give_the_same_output_as_the_factory(self):
        """Both parameterizations of ``PanelwiseTransformer`` give the same panel."""
        factory = create_delay_transformer_factory(_entity_delays(), prediction_date=PREDICTION)
        base = PublicationDelayTransformer(delays={}, prediction_date=PREDICTION)
        with_kwargs = _panelwise(base, entity_kwargs=prepare_entity_kwargs_from_delays(_entity_delays()))
        pd.testing.assert_frame_equal(_fit_transform(with_kwargs, _panel()),
                                      _fit_transform(_panelwise(factory), _panel()))

    def test_panel_round_trip(self):
        """Transform then inverse restores every entity exactly (added dates dropped)."""
        transformer = _panelwise(create_delay_transformer_factory(_entity_delays(), prediction_date=PREDICTION))
        recovered = transformer.inverse_transform(_fit_transform(transformer, _panel()))
        pd.testing.assert_frame_equal(recovered, _panel())

    def test_variable_absent_for_an_entity_is_untouched(self):
        """A variable without row for an entity is left as is for that entity only."""
        rows = {('FR', 'GDP'): 45.0, ('FR', 'CPI'): 20.0, ('DE', 'GDP'): 75.0}
        factory = create_delay_transformer_factory(_entity_delays(rows), prediction_date=PREDICTION)
        result = _fit_transform(_panelwise(factory), _panel())
        pd.testing.assert_series_equal(result.loc['DE', 'CPI'].dropna(), _panel().loc['DE', 'CPI'], check_freq=False)

    def test_entity_absent_from_the_table_raises(self):
        """With ``error_handling='raise'`` (default), an entity without delays stops the fit."""
        factory = create_delay_transformer_factory(_entity_delays(), prediction_date=PREDICTION)
        with pytest.raises(KeyError, match=r"\('IT',\) not found"):
            _fit_transform(_panelwise(factory), _panel(('FR', 'DE', 'IT')))

    def test_entity_absent_from_the_table_is_kept_with_a_warning(self):
        """With ``error_handling='warn'``, the entity without delays is kept untransformed."""
        factory = create_delay_transformer_factory(_entity_delays(), prediction_date=PREDICTION)
        transformer = _panelwise(factory, error_handling='warn')
        with pytest.warns(UserWarning, match=r"\('IT',\)"):
            result = transformer.fit_transform(_panel(('FR', 'DE', 'IT')))
        pd.testing.assert_frame_equal(result.loc['IT'], _panel(('FR', 'DE', 'IT')).loc['IT'])

    def test_entity_absent_from_entity_kwargs_keeps_the_base_transformer(self):
        """In ``entity_kwargs`` mode, an entity without kwargs uses the base transformer (no delay)."""
        base = PublicationDelayTransformer(delays={}, prediction_date=PREDICTION)
        transformer = _panelwise(base, entity_kwargs=prepare_entity_kwargs_from_delays(_entity_delays()))
        result = _fit_transform(transformer, _panel(('FR', 'DE', 'IT')))
        pd.testing.assert_frame_equal(result.loc['IT'], _panel(('FR', 'DE', 'IT')).loc['IT'])

    def test_per_entity_strategy(self):
        """``FR`` shifted, ``DE`` masked (target quarter): ``DE`` keeps its dates, ``FR`` gets later ones.

        Gold values for ``DE`` (45 / 20 days from the start): GDP masked 2 months per quarter (8 cells),
        CPI 1 month per quarter (4 cells).
        """
        rows = {('FR', 'GDP'): 45.0, ('FR', 'CPI'): 20.0, ('DE', 'GDP'): 45.0, ('DE', 'CPI'): 20.0}
        factory = create_delay_transformer_factory(_entity_delays(rows, frequency='Q'),
                                                   strategy={'FR': 'shift', 'DE': 'mask'}, prediction_date=PREDICTION)
        result = _fit_transform(_panelwise(factory), _panel())
        de = result.loc['DE']
        assert (len(de), de.isna().sum().to_dict(), result.loc['FR'].index.max()) == (
            12, {'GDP': 8, 'CPI': 4}, TS('2024-02-01'))

    def test_per_variable_strategy(self):
        """A per-variable strategy dict shifts ``GDP`` and masks ``CPI`` in every entity."""
        factory = create_delay_transformer_factory(_entity_delays(frequency='Q'),
                                                   strategy={'GDP': 'shift', 'CPI': 'mask'}, prediction_date=PREDICTION)
        result = _fit_transform(_panelwise(factory), _panel())
        assert result.loc['FR', 'GDP'].last_valid_index() == TS('2024-02-01')
