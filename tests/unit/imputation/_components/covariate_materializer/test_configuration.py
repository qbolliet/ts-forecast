"""Tests of the configuration surface of ``CovariateMaterializer``.

Covers the validation at ``__init__`` (validate without
transforming), the defaults, the
resolution of the per-feature dict form (``'__default__'`` key, an uncovered
column receiving the parameter default), the per-column key check run once at ``fit``, and the
readable representation.

The ``'linear'`` default also applies to the columns a dict form leaves
uncovered.

Covered symbols: ``CovariateMaterializer.__init__``, ``resolve_method``,
``resolve_anchor``, ``resolve_aggregation_constraint``, ``validate_columns``,
``__repr__``, ``DEFAULT_MATERIALIZATION_KEY``, ``DEFAULT_INTERPOLATION_METHOD``.
"""
# Cadre de test
import pytest

# Objets testés
from tsforecast.frequency.covariate_materializer import (
    DEFAULT_INTERPOLATION_METHOD,
    DEFAULT_MATERIALIZATION_KEY,
    CovariateMaterializer,
)


class TestInitValidation:
    """``__init__`` validates every setting without transforming it."""

    @pytest.mark.parametrize(
        'kwargs, message',
        [
            ({'covariate_strategy': 'nope'}, 'covariate_strategy'),
            ({'covariate_fallback': 'model'}, 'covariate_fallback'),
            ({'covariate_eligibility': 'some'}, 'covariate_eligibility'),
            ({'interpolation_method': {}}, 'cannot be empty'),
            ({'interpolation_anchor': 1.5}, r'\[0, 1\]'),
            ({'aggregation_constraint': 'median'}, 'aggregation_constraint must be one of'),
            ({'aggregation_constraint': {'a1': 'median'}}, 'aggregation_constraint values must be one of'),
            ({'aggregation_constraint': {}}, 'cannot be empty'),
            # Formes de interpolation_method jusque-là non éprouvées
            ({'interpolation_method': ''}, 'non-empty method name'),
            ({'interpolation_method': 3}, 'non-empty method name'),
            ({'interpolation_method': {'a1': ''}}, 'non-empty method names'),
            ({'interpolation_method': {'a1': None}}, 'non-empty method names'),
            # Formes de interpolation_anchor : un booléen n'est pas un nombre
            ({'interpolation_anchor': True}, r'\[0, 1\]'),
            ({'interpolation_anchor': '0.5'}, r'\[0, 1\]'),
            ({'interpolation_anchor': -0.1}, r'\[0, 1\]'),
            ({'interpolation_anchor': {}}, 'cannot be empty'),
            ({'interpolation_anchor': {'a1': 2.0}}, r"interpolation_anchor\['a1'\]"),
        ],
        ids=[
            'strategy', 'fallback-model', 'eligibility', 'method-empty-dict',
            'anchor-above-1', 'constraint-median', 'constraint-dict-median',
            'constraint-empty-dict', 'method-empty-string', 'method-int',
            'method-dict-empty-string', 'method-dict-none', 'anchor-bool',
            'anchor-string', 'anchor-negative', 'anchor-empty-dict',
            'anchor-dict-out-of-range',
        ],
    )
    def test_invalid_settings_raise(self, kwargs, message):
        """Every setting outside its contract raises an explicit ValueError."""
        with pytest.raises(ValueError, match=message):
            CovariateMaterializer(**kwargs)

    @pytest.mark.parametrize(
        'anchor',
        [None, 0, 0.0, 0.5, 1, 1.0, {'a1': 0.0, DEFAULT_MATERIALIZATION_KEY: None}],
        ids=['none', 'int-0', 'float-0', 'half', 'int-1', 'float-1', 'dict'],
    )
    def test_admissible_anchors_are_stored_verbatim(self, anchor):
        """Both bounds of ``[0, 1]``, None and the dict form are admissible."""
        materializer = CovariateMaterializer(interpolation_anchor=anchor)
        assert materializer.interpolation_anchor is anchor

    def test_init_stores_parameters_verbatim(self):
        """No parameter is transformed at initialization."""
        setting = {'a1': 'cubic'}
        materializer = CovariateMaterializer(interpolation_method=setting)
        assert materializer.interpolation_method is setting
        assert materializer.aggregation_constraint_applier is None

    def test_defaults_are_the_documented_ones(self):
        """Defaults of the constructor parameters."""
        materializer = CovariateMaterializer()
        # Valeurs par défaut documentées, paramètre par paramètre
        assert (
            materializer.covariate_strategy,
            materializer.covariate_fallback,
            materializer.covariate_eligibility,
            materializer.interpolation_anchor,
            materializer.aggregation_constraint,
        ) == ('interpolate', 'interpolate', 'any_entity', None, 'sum')

    def test_two_instances_never_share_their_stores(self):
        """The three stores are created per instance, empty."""
        first, second = CovariateMaterializer(), CovariateMaterializer()
        # Écriture dans le premier composant seulement
        first.imputed_store['a1'] = None
        assert (second.imputed_store, second.imputed_freq_store, second.origin_store) == ({}, {}, {})


class TestRepr:
    """Readable one-line representation."""

    def test_repr_names_the_three_modalities(self):
        """The representation names the strategy, the fallback and the eligibility."""
        materializer = CovariateMaterializer(
            covariate_strategy='model',
            covariate_fallback='tolerate_nan',
            covariate_eligibility='all_entities',
        )
        assert repr(materializer) == (
            "CovariateMaterializer(covariate_strategy='model', "
            "covariate_fallback='tolerate_nan', "
            "covariate_eligibility='all_entities')"
        )


class TestPerFeatureResolution:
    """Resolution of the scalar-or-dict settings, column by column."""

    def test_public_constants(self):
        """The fallback key of the dict forms, and the default method."""
        assert (DEFAULT_MATERIALIZATION_KEY, DEFAULT_INTERPOLATION_METHOD) == ('__default__', 'linear')

    @pytest.mark.parametrize(
        'setting, column, expected',
        [
            ('linear', 'a1', 'linear'),
            ('linear', None, 'linear'),
            ({'a1': 'cubic', DEFAULT_MATERIALIZATION_KEY: 'nearest'}, 'a1', 'cubic'),
            ({'a1': 'cubic', DEFAULT_MATERIALIZATION_KEY: 'nearest'}, 'q1', 'nearest'),
            # column=None lit la clé de repli, jamais une entrée de colonne
            ({'a1': 'cubic', DEFAULT_MATERIALIZATION_KEY: 'nearest'}, None, 'nearest'),
        ],
        ids=['scalar', 'scalar-global', 'own-entry', 'default-key', 'global-reads-default-key'],
    )
    def test_resolve_method(self, setting, column, expected):
        """Own entry, else ``'__default__'``, scalar form for every column."""
        materializer = CovariateMaterializer(interpolation_method=setting)
        assert materializer.resolve_method(column) == expected

    @pytest.mark.parametrize(
        'setting', [{'a1': 'cubic'}, 'linear'], ids=['dict-without-default-key', 'default-setting'],
    )
    def test_uncovered_column_falls_back_to_the_parameter_default(self, setting):
        """An uncovered column without ``'__default__'`` is interpolated ``'linear'``.

        The default of ``interpolation_method`` is ``'linear'`` —
        the default of ``HighFrequencyImputer`` too — and the rule is:
        uncovered features without a ``'__default__'`` get the default.
        The imputer hands its dict over unchanged, so the materializer's own
        fallback is what every uncovered column receives.
        """
        materializer = (
            CovariateMaterializer() if setting == 'linear'
            else CovariateMaterializer(interpolation_method=setting)
        )
        assert materializer.resolve_method('q1') == 'linear'

    @pytest.mark.parametrize(
        'setting, column, expected',
        [
            (0.5, 'a1', 0.5),
            (None, 'a1', None),
            # Une valeur nulle (0.0, fausse en Python) reste l'entrée de la colonne
            ({'a1': 0.0, DEFAULT_MATERIALIZATION_KEY: 1.0}, 'a1', 0.0),
            ({'a1': 0.0, DEFAULT_MATERIALIZATION_KEY: 1.0}, 'q1', 1.0),
            ({'a1': 0.5}, 'q1', None),
            ({'a1': 0.5}, None, None),
        ],
        ids=['scalar', 'scalar-none', 'own-zero', 'default-key', 'no-entry', 'global-no-default'],
    )
    def test_resolve_anchor(self, setting, column, expected):
        """Own entry, else ``'__default__'``, else None."""
        materializer = CovariateMaterializer(interpolation_anchor=setting)
        assert materializer.resolve_anchor(column) == expected

    @pytest.mark.parametrize(
        'setting, column, expected',
        [
            ('sum', 'a1', 'sum'),
            (None, 'a1', None),
            ({'a1': None, DEFAULT_MATERIALIZATION_KEY: 'sum'}, 'a1', None),
            ({'a1': None, DEFAULT_MATERIALIZATION_KEY: 'sum'}, 'q1', 'sum'),
            ({'a1': None}, 'q1', 'sum'),
            ({DEFAULT_MATERIALIZATION_KEY: None}, 'q1', None),
        ],
        ids=['scalar-sum', 'scalar-none', 'own-none', 'default-key', 'no-entry', 'default-none'],
    )
    def test_resolve_aggregation_constraint(self, setting, column, expected):
        """Own entry, else ``'__default__'``, else ``'sum'``."""
        materializer = CovariateMaterializer(aggregation_constraint=setting)
        assert materializer.resolve_aggregation_constraint(column) == expected


class TestValidateColumns:
    """Per-column dict keys are checked once, against every column of the data."""

    def test_unknown_dict_key_raises_listing_columns(self):
        """Unknown keys of the per-feature dicts are rejected and listed."""
        materializer = CovariateMaterializer(interpolation_method={'zz': 'cubic'})
        with pytest.raises(ValueError, match=r"interpolation_method names unknown columns"):
            materializer.validate_columns(['a1', 'q1'])
        # La clé de repli n'est jamais signalée comme inconnue
        CovariateMaterializer(
            interpolation_anchor={DEFAULT_MATERIALIZATION_KEY: 0.5}
        ).validate_columns(['a1'])

    @pytest.mark.parametrize(
        'kwargs, name',
        [
            ({'interpolation_method': {'zz': 'cubic', 'aa': 'nearest'}}, 'interpolation_method'),
            ({'interpolation_anchor': {'zz': 0.5, 'aa': 0.0}}, 'interpolation_anchor'),
            ({'aggregation_constraint': {'zz': None, 'aa': 'sum'}}, 'aggregation_constraint'),
        ],
        ids=['method', 'anchor', 'constraint'],
    )
    def test_each_setting_lists_its_unknown_keys_sorted(self, kwargs, name):
        """The three per-column settings are checked, offending keys sorted."""
        materializer = CovariateMaterializer(**kwargs)
        with pytest.raises(ValueError) as error:
            materializer.validate_columns(['a1', 'q1'])
        assert str(error.value) == f"{name} names unknown columns : ['aa', 'zz']"

    def test_scalar_forms_have_no_key_to_check(self):
        """Scalar settings pass whatever the columns, even none."""
        materializer = CovariateMaterializer(
            interpolation_method='linear', interpolation_anchor=0.5, aggregation_constraint=None,
        )
        assert materializer.validate_columns([]) is None

    def test_known_keys_with_special_characters_pass(self):
        """Column names with spaces, accents and symbols are plain keys."""
        special = "climat d'affaires (%) à crédit/mois"
        materializer = CovariateMaterializer(
            interpolation_method={special: 'nearest'},
            interpolation_anchor={special: 0.5, DEFAULT_MATERIALIZATION_KEY: None},
            aggregation_constraint={special: None},
        )
        # Contrôle sur un itérable à usage unique : un générateur suffit
        assert materializer.validate_columns(column for column in ['a1', special]) is None
