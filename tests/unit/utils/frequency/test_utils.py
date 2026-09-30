"""Tests for ``tsforecast.utils.frequency.utils`` (everything but the detection, see ``detector/``).

Covers the module-level functions that wrap ``FrequencyNormalizer``:
``normalize_frequency`` (every ``return_format``, multipliers, unknown or removed
frequencies), ``canonicalize_frequency``, ``to_literal`` / ``to_code`` /
``to_pandas_freq`` / ``to_dateoffset`` / ``validate_frequency`` (delegation to the
normalizer, every code and position), ``is_higher_frequency`` and
``get_frequency_order`` (total order over the supported codes, antisymmetry and
transitivity as properties, multipliers).

Also covers the behaviours changed while fixing ``utils/position`` (prompt U3):
``normalize_frequency(return_format='full')`` keeps a leading multiplier
(``'2MS'``), and ``detect_index_frequency`` returns a canonical quarterly
anchor (``'QS-JAN'`` rather than the equivalent ``'QS-OCT'`` reported by
``pd.infer_freq``), through ``canonicalize_frequency``.

Detection itself (``detect_frequency``, ``detect_dataset_frequency``,
``detect_index_frequency``, ``target_offset_for_index``) is tested in
the ``detector/`` test package (prompt U7).
"""
from __future__ import annotations

import itertools
import warnings

import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset

from tsforecast.utils.frequency.normalizer import FrequencyNormalizer
from tsforecast.utils.frequency.utils import (
    canonicalize_frequency,
    detect_index_frequency,
    get_frequency_order,
    is_higher_frequency,
    normalize_frequency,
    to_code,
    to_dateoffset,
    to_literal,
    to_pandas_freq,
    validate_frequency,
)
from tsforecast.utils.parse import ParsedFrequency

# Codes supportés, du plus fin au plus grossier. Valeur d'or issue du domaine : durée
# nominale croissante, le jour ouvré ('B') se plaçant après le jour ('D') car il saute
# les week-ends, la quinzaine ('SM') entre la semaine et le mois
ORDERED_CODES = ['ns', 'us', 'ms', 's', 'min', 'h', 'D', 'B', 'W', 'SM', 'M', 'Q', 'Y']

# Noms littéraux associés aux codes
LITERAL_NAMES = {
    'ns': 'nanosecond', 'us': 'microsecond', 'ms': 'millisecond', 's': 'second',
    'min': 'minute', 'h': 'hourly', 'D': 'daily', 'B': 'business_daily',
    'W': 'weekly', 'SM': 'semi_monthly', 'M': 'monthly', 'Q': 'quarterly', 'Y': 'annual',
}

# Écritures équivalentes d'une même fréquence (sans multiplicateur) : code, nom littéral,
# positions début / fin, ancres
SPELLINGS = {
    'ns': ['ns', 'nanosecond'],
    'us': ['us', 'microsecond'],
    'ms': ['ms', 'millisecond'],
    's': ['s', 'second'],
    'min': ['min', 'minute'],
    'h': ['h', 'hourly'],
    'D': ['D', 'daily'],
    'B': ['B', 'business_daily'],
    'W': ['W', 'weekly', 'W-MON', 'W-SUN'],
    'SM': ['SM', 'semi_monthly', 'SMS', 'SME'],
    'M': ['M', 'monthly', 'MS', 'ME'],
    'Q': ['Q', 'quarterly', 'QS', 'QE', 'QE-DEC', 'QS-FEB'],
    'Y': ['Y', 'annual', 'YS', 'YE', 'YS-JAN', 'YE-DEC'],
}

# Codes pour lesquels pandas connaît des variantes début (S) / fin (E)
POSITION_AWARE_CODES = {'SM', 'M', 'Q', 'Y'}

# Fréquences avec multiplicateur, pour les propriétés d'ordre
MULTIPLIED_FREQUENCIES = (
    [f'{k}{code}' for code in ORDERED_CODES for k in ('', '2', '3', '7', '24')]
    + ['MS', 'ME', 'QS', 'QE-DEC', 'YS-JAN', 'W-MON', '2MS', '3QS-FEB', '15min', '12ME']
)

# Entrées qui ne sont pas des chaînes, avec un identifiant lisible
NON_STRING_INPUTS = [
    pytest.param(123, id="int"), pytest.param(3.14, id="float"),
    pytest.param(['M'], id="list"), pytest.param(b'M', id="bytes"),
]

# Fréquences non supportées : inconnues, ancienne écriture de l'année, casse ou espaces
UNSUPPORTED_FREQUENCIES = ['xyz', 'foo', '', '2foo', 'A', 'AS-JAN', 'T', 'MONTHLY', ' M']


# =============================================================================
# normalize_frequency : multiplicateur, canonicalisation (existant, U3)
# =============================================================================

class TestNormalizeFrequencyFullMultiplier:
    """``normalize_frequency(return_format='full')`` with a leading multiplier."""

    @pytest.mark.parametrize("frequency", ["2MS", "12ME", "3QS-FEB", "2D", "15min"])
    def test_multiplied_frequency_is_kept(self, frequency):
        """A valid multiplied frequency is returned unchanged."""
        assert normalize_frequency(frequency, return_format="full") == frequency

    @pytest.mark.parametrize("frequency", ["2A", "2foo"])
    def test_invalid_frequency_after_multiplier_raises(self, frequency):
        """Only the part after the multiplier is validated: it must be supported."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            normalize_frequency(frequency, return_format="full")


class TestDetectIndexFrequencyAnchors:
    """``detect_index_frequency`` reports multiplied and canonically anchored frequencies."""

    @pytest.mark.parametrize(
        "freq, expected",
        [
            # pd.infer_freq renvoie 'QS-OCT' pour un index 'QS' débutant en janvier
            pytest.param("QS", "QS-JAN", id="quarter-start-default"),
            pytest.param("QE", "QE-DEC", id="quarter-end-default"),
            pytest.param("QS-FEB", "QS-FEB", id="quarter-start-february"),
            # Trimestres terminés en nov., févr., mai, août : fin canonique en février
            pytest.param("QE-NOV", "QE-FEB", id="quarter-end-november"),
            pytest.param("2MS", "2MS", id="bimonthly"),
        ],
    )
    def test_full_format(self, freq, expected):
        """Detected string in 'full' format."""
        dates = pd.date_range("2023-01-01", periods=8, freq=freq)
        assert detect_index_frequency(dates, return_format="full") == expected

    @pytest.mark.parametrize("freq", ["QS", "QE", "QS-FEB", "QE-NOV", "QS-MAR"])
    def test_detected_frequency_regenerates_the_index(self, freq):
        """Property: the detected offset regenerates exactly the source dates."""
        dates = pd.date_range("2023-01-01", periods=8, freq=freq)
        detected = detect_index_frequency(dates, return_format="full")
        assert list(pd.date_range(dates[0], periods=8, freq=detected)) == list(dates)


class TestCanonicalizeFrequency:
    """``canonicalize_frequency``: one representative per class of equivalent spellings."""

    @pytest.mark.parametrize(
        "frequency, expected",
        [
            ("QS-JAN", "QS-JAN"), ("QS-APR", "QS-JAN"), ("QS-JUL", "QS-JAN"), ("QS-OCT", "QS-JAN"),
            ("QS-NOV", "QS-FEB"), ("QS-DEC", "QS-MAR"),
            ("QE-DEC", "QE-DEC"), ("QE-MAR", "QE-DEC"), ("QE-SEP", "QE-DEC"),
            ("QE-JAN", "QE-JAN"), ("QE-NOV", "QE-FEB"),
            # Ancre sans position : pandas la lit comme mois de fin
            ("Q-MAR", "Q-DEC"),
            # Multiplicateur conservé
            ("2QS-OCT", "2QS-JAN"), ("2QE-MAR", "2QE-DEC"), ("3Q-JUN", "3Q-DEC"),
        ],
    )
    def test_golden_values(self, frequency, expected):
        """Start anchors map to JAN / FEB / MAR, end anchors to the preceding month."""
        assert canonicalize_frequency(frequency) == expected

    @pytest.mark.parametrize("frequency", ["QS-JAN", "QS-OCT", "QE-NOV", "QE-MAR"])
    def test_canonical_offset_generates_the_same_dates(self, frequency):
        """Property: the canonical anchor generates the same dates as the original one."""
        original = pd.date_range("2023-01-01", periods=8, freq=frequency)
        canonical = pd.date_range("2023-01-01", periods=8, freq=canonicalize_frequency(frequency))
        assert list(original) == list(canonical)

    @pytest.mark.parametrize(
        "frequency",
        [
            None, "MS", "QS", "YS-JUL", "W-MON", "2D", "15min", "QS-XYZ",
            # Jours ouvrés trimestriels : non supportés par le package, laissés tels quels
            "BQS-APR",
            # Chaîne non analysable
            "", "not a frequency",
        ],
    )
    def test_other_inputs_unchanged(self, frequency):
        """Non-quarterly, unanchored, unknown or unparsable inputs are returned unchanged."""
        assert canonicalize_frequency(frequency) == frequency

    @pytest.mark.parametrize("frequency", ["QS-JAN", "QS-OCT", "QE-NOV", "2QE-MAR", "Q-JUN"])
    def test_is_idempotent(self, frequency):
        """A canonical spelling is a fixed point."""
        once = canonicalize_frequency(frequency)
        assert canonicalize_frequency(once) == once

    def test_canonical_pairs_describe_the_same_periods(self):
        """``QS-FEB`` and ``QE-JAN`` (both canonical) bound the same quarters."""
        starts = pd.date_range("2024-02-01", periods=4, freq="QS-FEB")
        ends = pd.date_range("2024-04-30", periods=4, freq="QE-JAN")
        assert list(starts + to_offset("QS-FEB") - pd.Timedelta(days=1)) == list(ends)


# =============================================================================
# normalize_frequency : return_format
# =============================================================================

# Une ligne par entrée : (entrée, 'base', 'with_position', 'full', 'components').
# Valeurs d'or écrites à la main : 'base' garde le code, 'with_position' y ajoute S / E,
# 'full' renvoie la chaîne d'entrée (validée), 'components' la décompose en
# (code, position, ancre, multiplicateur)
RETURN_FORMAT_TABLE = [
    pytest.param('D', 'D', 'D', 'D', ('D', None, None, 1), id="daily-code"),
    pytest.param('monthly', 'M', 'M', 'monthly', ('M', None, None, 1), id="monthly-literal"),
    pytest.param('MS', 'M', 'MS', 'MS', ('M', 'S', None, 1), id="month-start"),
    pytest.param('ME', 'M', 'ME', 'ME', ('M', 'E', None, 1), id="month-end"),
    pytest.param('QE-DEC', 'Q', 'QE', 'QE-DEC', ('Q', 'E', 'DEC', 1), id="quarter-end-december"),
    pytest.param('QS-FEB', 'Q', 'QS', 'QS-FEB', ('Q', 'S', 'FEB', 1), id="quarter-start-february"),
    pytest.param('YS-JAN', 'Y', 'YS', 'YS-JAN', ('Y', 'S', 'JAN', 1), id="year-start-january"),
    pytest.param('YE-DEC', 'Y', 'YE', 'YE-DEC', ('Y', 'E', 'DEC', 1), id="year-end-december"),
    pytest.param('annual', 'Y', 'Y', 'annual', ('Y', None, None, 1), id="annual-literal"),
    pytest.param('W-SUN', 'W', 'W', 'W-SUN', ('W', None, 'SUN', 1), id="week-sunday"),
    pytest.param('SMS', 'SM', 'SMS', 'SMS', ('SM', 'S', None, 1), id="semi-month-start"),
    pytest.param('SME', 'SM', 'SME', 'SME', ('SM', 'E', None, 1), id="semi-month-end"),
    pytest.param('business_daily', 'B', 'B', 'business_daily', ('B', None, None, 1), id="business-daily-literal"),
    pytest.param('minute', 'min', 'min', 'minute', ('min', None, None, 1), id="minute-literal"),
    pytest.param('ms', 'ms', 'ms', 'ms', ('ms', None, None, 1), id="millisecond-code"),
    pytest.param('2MS', 'M', 'MS', '2MS', ('M', 'S', None, 2), id="multiplied-month-start"),
    pytest.param('3QS-FEB', 'Q', 'QS', '3QS-FEB', ('Q', 'S', 'FEB', 3), id="multiplied-anchored-quarter"),
    pytest.param('15min', 'min', 'min', '15min', ('min', None, None, 15), id="multiplied-minute"),
    pytest.param('2W-MON', 'W', 'W', '2W-MON', ('W', None, 'MON', 2), id="multiplied-week-monday"),
]


class TestNormalizeFrequencyReturnFormats:
    """``normalize_frequency`` in each of its four ``return_format`` modalities."""

    @pytest.mark.parametrize("frequency, base, with_position, full, components", RETURN_FORMAT_TABLE)
    def test_base(self, frequency, base, with_position, full, components):
        """'base' keeps the code only: position, anchor and multiplier are left out."""
        assert normalize_frequency(frequency, return_format='base') == base

    @pytest.mark.parametrize("frequency, base, with_position, full, components", RETURN_FORMAT_TABLE)
    def test_with_position(self, frequency, base, with_position, full, components):
        """'with_position' adds the S / E position when there is one; anchor and multiplier stay out."""
        assert normalize_frequency(frequency, return_format='with_position') == with_position

    @pytest.mark.parametrize("frequency, base, with_position, full, components", RETURN_FORMAT_TABLE)
    def test_full(self, frequency, base, with_position, full, components):
        """'full' validates then hands back the input string (multiplier included), literals unchanged."""
        assert normalize_frequency(frequency, return_format='full') == full

    @pytest.mark.parametrize("frequency, base, with_position, full, components", RETURN_FORMAT_TABLE)
    def test_components(self, frequency, base, with_position, full, components):
        """'components' decomposes into ``(code, position, anchor, multiplier)``."""
        assert normalize_frequency(frequency, return_format='components') == components

    @pytest.mark.parametrize("frequency, base, with_position, full, components", RETURN_FORMAT_TABLE)
    def test_default_format_is_base(self, frequency, base, with_position, full, components):
        """Without ``return_format``, the code alone is returned (backward compatible)."""
        assert normalize_frequency(frequency) == base

    @pytest.mark.parametrize("frequency, base, with_position, full, components", RETURN_FORMAT_TABLE)
    def test_formats_are_consistent_with_each_other(self, frequency, base, with_position, full, components):
        """Property: the components rebuild the ``'base'`` and ``'with_position'`` outputs."""
        parsed = normalize_frequency(frequency, return_format='components')
        assert parsed.freq == normalize_frequency(frequency, return_format='base')
        assert f"{parsed.freq}{parsed.position or ''}" == normalize_frequency(frequency, return_format='with_position')

    def test_components_is_a_parsed_frequency(self):
        """The components are reachable by name."""
        parsed = normalize_frequency('3QS-FEB', return_format='components')
        assert isinstance(parsed, ParsedFrequency)
        assert (parsed.freq, parsed.position, parsed.suffix, parsed.multiplier) == ('Q', 'S', 'FEB', 3)

    @pytest.mark.parametrize("return_format", ['invalid', 'xyz', 'BASE', '', None, 3])
    def test_invalid_return_format_raises(self, return_format):
        """An unknown format is rejected, listing the four supported ones."""
        with pytest.raises(ValueError, match="Invalid return_format"):
            normalize_frequency('M', return_format=return_format)

    def test_invalid_return_format_is_reported_before_the_frequency(self):
        """The format is checked even when the frequency is unknown too."""
        with pytest.raises(ValueError, match="Invalid return_format"):
            normalize_frequency('xyz', return_format='bogus')

    @pytest.mark.parametrize("return_format", ['base', 'with_position', 'full', 'components'])
    @pytest.mark.parametrize("frequency", UNSUPPORTED_FREQUENCIES)
    def test_unsupported_frequency_raises_in_every_format(self, frequency, return_format):
        """Unknown, removed (``'A'``, ``'T'``), mis-cased or padded frequencies are rejected."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            normalize_frequency(frequency, return_format=return_format)

    @pytest.mark.parametrize("return_format", ['base', 'with_position', 'full', 'components'])
    def test_none_raises_value_error_in_every_format(self, return_format):
        """``None`` (no frequency detected upstream) is a ``ValueError``."""
        with pytest.raises(ValueError):
            normalize_frequency(None, return_format=return_format)

    @pytest.mark.parametrize("return_format", ['base', 'with_position', 'full', 'components'])
    @pytest.mark.parametrize("frequency", NON_STRING_INPUTS)
    def test_non_string_raises_value_error_in_every_format(self, frequency, return_format):
        """A non-string is rejected with the same ``ValueError`` in every format (ANO-UTILS-037)."""
        with pytest.raises(ValueError, match="must be a string"):
            normalize_frequency(frequency, return_format=return_format)

    @pytest.mark.parametrize("return_format", ['base', 'with_position', 'full', 'components'])
    @pytest.mark.parametrize("frequency", ['QS-XYZ', 'MS-JAN', 'D-MON', 'YE-FOO', 'W-XYZ', 'SMS-1', 'Q-'])
    def test_invalid_anchor_raises_in_every_format(self, frequency, return_format):
        """An anchor that does not exist in pandas is rejected in every format (ANO-UTILS-040)."""
        with pytest.raises(ValueError, match="Invalid anchor|does not accept an anchor"):
            normalize_frequency(frequency, return_format=return_format)

    @pytest.mark.parametrize("frequency", ['2A', '2foo', '0D'])
    def test_multiplier_does_not_make_an_invalid_base_valid(self, frequency):
        """Only the part after the multiplier counts: it must be a supported frequency."""
        with pytest.raises(ValueError):
            normalize_frequency(frequency, return_format='components')

    def test_literal_name_that_parse_frequency_cannot_read(self):
        """``'business_daily'`` (underscore) falls back on ``normalize`` in the 'components' format."""
        assert normalize_frequency('business_daily', return_format='components') == ('B', None, None, 1)


class TestNormalizeFrequencyRemovedAlias:
    """The removed ``'A'`` alias is rejected in every format (see ``test_normalizer.py``)."""

    @pytest.mark.parametrize("return_format", ['base', 'with_position', 'full', 'components'])
    @pytest.mark.parametrize("alias", ['A', 'AS', 'AE', 'AS-JAN', 'AE-DEC'])
    def test_alias_is_rejected(self, alias, return_format):
        """``normalize_frequency('AS-JAN', ...)`` raises instead of answering ``'Y'``."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            normalize_frequency(alias, return_format=return_format)

    @pytest.mark.parametrize(
        "detected_index_freq, expected",
        [
            pytest.param('YS', ('Y', 'S', 'JAN', 1), id="year-start"),
            pytest.param('YE', ('Y', 'E', 'DEC', 1), id="year-end"),
        ],
    )
    def test_detected_yearly_index_is_spelled_with_y(self, detected_index_freq, expected):
        """The frequency pandas detects on a yearly index is ``'YS-JAN'`` / ``'YE-DEC'``, never ``'AS-JAN'``."""
        index = pd.date_range('2020-01-01', periods=4, freq=detected_index_freq)
        assert detect_index_frequency(index, return_format='components') == expected


class TestDecompositionOfDetectedFrequencies:
    """The converter decomposes detected frequencies through ``return_format='components'``."""

    def test_decompose_detected_frequency(self):
        """A detected ``'QE-DEC'`` splits into code, position, anchor and multiplier."""
        base, position, anchor, multiplier = normalize_frequency('QE-DEC', return_format='components')
        assert (base, position, anchor, multiplier) == ('Q', 'E', 'DEC', 1)

    def test_decompose_source_and_target(self):
        """Source and target frequencies expose their base code by name."""
        assert normalize_frequency('QE-DEC', return_format='components').freq == 'Q'
        assert normalize_frequency('MS', return_format='components').freq == 'M'

    def test_preserve_position(self):
        """The position of a detected frequency is kept in the 'with_position' format."""
        assert normalize_frequency('QE-DEC', return_format='with_position') == 'QE'


# =============================================================================
# Fonctions de commodité : délégation à FrequencyNormalizer
# =============================================================================

# Entrées de la table de délégation : codes, noms littéraux, chaînes complexes
DELEGATION_INPUTS = (
    ORDERED_CODES + list(LITERAL_NAMES.values()) + ['MS', 'ME', 'QE-DEC', 'YS-JAN', 'W-MON', '2MS', '15min']
)


class TestConvenienceFunctionsDelegate:
    """The module-level functions give what the ``FrequencyNormalizer`` methods give."""

    @pytest.fixture(scope='class')
    def normalizer(self):
        """Create a FrequencyNormalizer instance for testing."""
        return FrequencyNormalizer()

    @pytest.mark.parametrize("value", DELEGATION_INPUTS)
    def test_to_literal(self, normalizer, value):
        """``to_literal`` is ``FrequencyNormalizer.to_literal``."""
        assert to_literal(value) == normalizer.to_literal(value)

    @pytest.mark.parametrize("value", DELEGATION_INPUTS)
    def test_to_code(self, normalizer, value):
        """``to_code`` is ``FrequencyNormalizer.to_code``."""
        assert to_code(value) == normalizer.to_code(value)

    @pytest.mark.parametrize("value", DELEGATION_INPUTS)
    def test_to_pandas_freq(self, normalizer, value):
        """``to_pandas_freq`` is ``FrequencyNormalizer.to_pandas_freq``."""
        assert to_pandas_freq(value) == normalizer.to_pandas_freq(value)

    @pytest.mark.parametrize("value", DELEGATION_INPUTS)
    def test_to_dateoffset(self, normalizer, value):
        """``to_dateoffset`` is ``FrequencyNormalizer.to_dateoffset``."""
        assert to_dateoffset(value) == normalizer.to_dateoffset(value)

    @pytest.mark.parametrize("value", DELEGATION_INPUTS + ['xyz', 'A', '', 'MONTHLY'])
    def test_validate_frequency(self, normalizer, value):
        """``validate_frequency`` is ``FrequencyNormalizer.validate``."""
        assert validate_frequency(value) is normalizer.validate(value)

    @pytest.mark.parametrize("value", ['xyz', 'A', '', None, 5])
    def test_validate_frequency_of_unsupported_values_is_false(self, value):
        """An unsupported value is reported, not raised."""
        assert validate_frequency(value) is False

    @pytest.mark.parametrize("function", [to_literal, to_code, to_pandas_freq, to_dateoffset], ids=lambda f: f.__name__)
    @pytest.mark.parametrize("value", ['xyz', 'A', 'AS-JAN', '', 'MONTHLY'])
    def test_unsupported_frequency_raises(self, function, value):
        """Every conversion rejects an unknown or removed frequency with a ``ValueError``."""
        with pytest.raises(ValueError):
            function(value)

    @pytest.mark.parametrize(
        "code, literal",
        [('ns', 'nanosecond'), ('h', 'hourly'), ('SM', 'semi_monthly'), ('Y', 'annual'), ('B', 'business_daily')],
    )
    def test_documented_examples(self, code, literal):
        """Golden values from the docstrings: code -> literal name -> code."""
        assert to_literal(code) == literal
        assert to_code(literal) == code


# =============================================================================
# to_pandas_freq / to_dateoffset : chaque fréquence, chaque position
# =============================================================================

class TestToPandasFreqEveryCodeAndPosition:
    """``to_pandas_freq`` over the whole table of codes and positions."""

    @pytest.mark.parametrize("position", [None, 'S', 'E'], ids=['no-position', 'start', 'end'])
    @pytest.mark.parametrize("code", ORDERED_CODES)
    def test_position_is_kept_only_where_pandas_defines_it(self, code, position):
        """``'MS'`` / ``'QE'`` / ``'YS'`` / ``'SME'`` keep their position; the other codes have none.

        A code with start / end variants and no position gets its end variant (ANO-UTILS-039).
        """
        # Valeur d'or : M / Q / Y / SM acceptent S ou E, la fin par défaut ; ns, us, ms, s, min,
        # h, D, B, W n'en ont pas
        given = f"{code}{position or ''}"
        if code in POSITION_AWARE_CODES:
            expected = f"{code}{position or 'E'}"
        else:
            expected = code
        assert to_pandas_freq(given) == expected

    @pytest.mark.parametrize("position", [None, 'S', 'E'], ids=['no-position', 'start', 'end'])
    @pytest.mark.parametrize("code", ORDERED_CODES)
    def test_result_is_a_valid_pandas_alias_without_warning(self, code, position):
        """Property: whatever the code and position, pandas reads the result and does not deprecate it."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert to_offset(to_pandas_freq(f"{code}{position or ''}")) is not None

    @pytest.mark.parametrize("code, literal", list(LITERAL_NAMES.items()))
    def test_literal_names_give_the_alias_of_their_code(self, code, literal):
        """Each literal name gives the pandas alias of its frequency (end variant by default)."""
        expected = f"{code}E" if code in POSITION_AWARE_CODES else code
        assert to_pandas_freq(literal) == expected

    @pytest.mark.parametrize(
        "value",
        ['QE-DEC', 'QS-FEB', 'QE-NOV', 'YS-JAN', 'YE-JUN', 'W-MON', 'W-SUN', '2MS', '3QS-FEB', '15min'],
    )
    def test_anchor_and_multiplier_are_kept(self, value):
        """The anchor after the dash and the leading multiplier survive."""
        assert to_pandas_freq(value) == value

    @pytest.mark.parametrize("value", ['xyz', 'A', 'AS', 'AE-DEC', '0D', '-1D', 'T', 'QS-XYZ', 'MS-JAN'])
    def test_unsupported_frequency_raises(self, value):
        """Unknown, removed aliases, non-positive multipliers and invalid anchors are rejected."""
        with pytest.raises(ValueError):
            to_pandas_freq(value)

    @pytest.mark.parametrize("value", ['M', 'monthly', 'Q', 'quarterly', 'Y', 'annual', 'SM', 'semi_monthly'])
    def test_base_code_is_still_available_through_normalization(self, value):
        """The bare code (``'M'``) is what ``normalize_frequency`` / ``to_code`` give, not ``to_pandas_freq``."""
        assert normalize_frequency(value) == to_code(value)
        assert to_pandas_freq(value) == f"{to_code(value)}E"


class TestToDateoffsetEveryCode:
    """``to_dateoffset`` over the whole table of codes."""

    @pytest.mark.parametrize("code", ORDERED_CODES)
    def test_offset_of_each_code(self, code):
        """Each supported code gives its pandas offset, without deprecation warning."""
        # Valeur d'or : les alias nus de pandas 2 ('M', 'Q', 'Y', 'SM') désignaient la fin de période
        expected_alias = f"{code}E" if code in POSITION_AWARE_CODES else code
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert to_dateoffset(code) == to_offset(expected_alias)

    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param('MS', pd.offsets.MonthBegin(), id="MS"),
            pytest.param('ME', pd.offsets.MonthEnd(), id="ME"),
            pytest.param('QS', pd.offsets.QuarterBegin(startingMonth=1), id="QS"),
            pytest.param('QE', pd.offsets.QuarterEnd(startingMonth=12), id="QE"),
            pytest.param('YS', pd.offsets.YearBegin(month=1), id="YS"),
            pytest.param('YE', pd.offsets.YearEnd(month=12), id="YE"),
            pytest.param('SMS', pd.offsets.SemiMonthBegin(), id="SMS"),
            pytest.param('SME', pd.offsets.SemiMonthEnd(), id="SME"),
        ],
    )
    def test_start_and_end_positions_give_distinct_offsets(self, value, expected):
        """Positions are not lost on the way to the offset, and raise no pandas warning."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert to_dateoffset(value) == expected


# =============================================================================
# is_higher_frequency : ordre total, antisymétrie, transitivité
# =============================================================================

def _order_matrix(frequencies: list[str]) -> dict[tuple[str, str], bool]:
    """Return ``is_higher_frequency`` for every ordered pair of ``frequencies``.

    Args:
        frequencies: Frequency strings to compare two by two.

    Returns:
        Dictionary ``{(a, b): is_higher_frequency(a, b)}`` over the Cartesian product.
    """
    return {(a, b): is_higher_frequency(a, b) for a in frequencies for b in frequencies}


class TestIsHigherFrequencyOrderTable:
    """``is_higher_frequency`` is the strict order of the table ``ORDERED_CODES``."""

    @pytest.mark.parametrize(
        "code_a, code_b",
        [pytest.param(a, b, id=f"{a}-vs-{b}") for a, b in itertools.product(ORDERED_CODES, repeat=2)],
    )
    def test_every_pair_follows_the_table(self, code_a, code_b):
        """A code is higher than another exactly when it comes earlier in the table."""
        expected = ORDERED_CODES.index(code_a) < ORDERED_CODES.index(code_b)
        assert is_higher_frequency(code_a, code_b) is expected

    def test_irreflexive(self):
        """No frequency is higher than itself."""
        assert not [c for c in ORDERED_CODES if is_higher_frequency(c, c)]

    def test_antisymmetric(self):
        """Property: ``a`` higher than ``b`` excludes ``b`` higher than ``a``."""
        higher = _order_matrix(ORDERED_CODES)
        assert not [(a, b) for a, b in higher if higher[a, b] and higher[b, a]]

    def test_total_on_distinct_codes(self):
        """Property: for two distinct codes, one is higher than the other."""
        higher = _order_matrix(ORDERED_CODES)
        assert all(higher[a, b] != higher[b, a] for a, b in itertools.combinations(ORDERED_CODES, 2))

    def test_transitive(self):
        """Property: ``a > b`` and ``b > c`` imply ``a > c``, over all the triples of codes."""
        higher = _order_matrix(ORDERED_CODES)
        violations = [
            (a, b, c) for a, b, c in itertools.product(ORDERED_CODES, repeat=3)
            if higher[a, b] and higher[b, c] and not higher[a, c]
        ]
        assert violations == []

    def test_chain_of_neighbours_is_enough(self):
        """The 12 neighbouring pairs of the table are each strictly ordered."""
        assert all(is_higher_frequency(a, b) for a, b in zip(ORDERED_CODES, ORDERED_CODES[1:]))

    @pytest.mark.parametrize(
        "code_a, code_b",
        [pytest.param(a, b, id=f"{a}-vs-{b}") for a, b in itertools.product(ORDERED_CODES, repeat=2)],
    )
    def test_answer_does_not_depend_on_the_spelling(self, code_a, code_b):
        """Code, literal name, position and anchor spellings of a frequency compare alike."""
        expected = ORDERED_CODES.index(code_a) < ORDERED_CODES.index(code_b)
        answers = {is_higher_frequency(x, y) for x in SPELLINGS[code_a] for y in SPELLINGS[code_b]}
        assert answers == {expected}

    def test_agrees_with_get_frequency_order(self):
        """Without multiplier, ``a`` is higher than ``b`` exactly when its order number is smaller."""
        assert all(
            is_higher_frequency(a, b) == (get_frequency_order(a) < get_frequency_order(b))
            for a, b in itertools.product(ORDERED_CODES, repeat=2)
        )

    @pytest.mark.parametrize(
        "freq1, freq2, expected",
        [
            pytest.param('daily', 'monthly', True, id="doc-daily-monthly"),
            pytest.param('quarterly', 'weekly', False, id="doc-quarterly-weekly"),
            pytest.param('MS', '2MS', True, id="doc-month-vs-two-months"),
            pytest.param('2MS', 'QS', True, id="doc-two-months-vs-quarter"),
            pytest.param('D', 'B', True, id="day-vs-business-day"),
            pytest.param('B', 'W', True, id="business-day-vs-week"),
            pytest.param('W', 'SM', True, id="week-vs-semi-month"),
            pytest.param('SM', 'M', True, id="semi-month-vs-month"),
        ],
    )
    def test_documented_examples(self, freq1, freq2, expected):
        """Golden values of the docstring and of the granularity chain D > B > W > SM > M."""
        assert is_higher_frequency(freq1, freq2) is expected

    @pytest.mark.parametrize("freq1, freq2", [('xyz', 'D'), ('D', 'xyz'), ('A', 'D'), ('D', 'AS-JAN'), (None, 'D'), ('D', 5)])
    def test_unsupported_frequency_raises(self, freq1, freq2):
        """An unsupported frequency on either side raises, without a silent default order."""
        with pytest.raises(ValueError):
            is_higher_frequency(freq1, freq2)

    def test_delegates_to_the_normalizer(self):
        """The function is ``FrequencyNormalizer.is_higher_frequency``."""
        normalizer = FrequencyNormalizer()
        assert all(
            is_higher_frequency(a, b) is normalizer.is_higher_frequency(a, b)
            for a, b in itertools.product(['D', 'monthly', '2MS', 'QE-DEC', 'W-MON'], repeat=2)
        )


class TestIsHigherFrequencyWithMultipliers:
    """A multiplier lengthens the period: the order stays a strict, antisymmetric, transitive relation."""

    @pytest.fixture(scope='class')
    def higher(self):
        """Matrix ``is_higher_frequency`` over ``MULTIPLIED_FREQUENCIES``."""
        return _order_matrix(MULTIPLIED_FREQUENCIES)

    def test_irreflexive(self, higher):
        """No frequency is higher than itself, multiplied or not."""
        assert not [f for f in MULTIPLIED_FREQUENCIES if higher[f, f]]

    def test_antisymmetric(self, higher):
        """Property: ``a`` higher than ``b`` excludes ``b`` higher than ``a``."""
        assert not [(a, b) for (a, b), value in higher.items() if value and higher[b, a]]

    def test_transitive(self, higher):
        """Property: ``a > b`` and ``b > c`` imply ``a > c``, over all the triples."""
        violations = [
            (a, b, c) for a, b, c in itertools.product(MULTIPLIED_FREQUENCIES, repeat=3)
            if higher[a, b] and higher[b, c] and not higher[a, c]
        ]
        assert violations == []

    @pytest.mark.parametrize("code", ORDERED_CODES)
    def test_smaller_multiplier_is_higher_frequency(self, code):
        """Same code: the smaller the multiplier, the shorter the period, the higher the frequency."""
        assert is_higher_frequency(code, f"2{code}") is True
        assert is_higher_frequency(f"2{code}", f"3{code}") is True
        assert is_higher_frequency(f"3{code}", f"2{code}") is False

    @pytest.mark.parametrize(
        "freq1, freq2, expected",
        [
            pytest.param('24h', 'D', False, id="24h-not-higher-than-day"),
            pytest.param('D', '24h', False, id="day-not-higher-than-24h"),
            pytest.param('7D', 'W', False, id="7-days-not-higher-than-week"),
            pytest.param('W', '7D', False, id="week-not-higher-than-7-days"),
            pytest.param('6D', 'W', True, id="6-days-higher-than-week"),
            pytest.param('2W', '15D', True, id="2-weeks-higher-than-15-days"),
        ],
    )
    def test_equal_nominal_durations_are_incomparable(self, freq1, freq2, expected):
        """Different codes of the same nominal duration (24 h and a day) are neither higher nor lower."""
        assert is_higher_frequency(freq1, freq2) is expected


class TestIsHigherFrequencyDayVersusBusinessDay:
    """``kD`` is a higher frequency than ``kB``, like ``'D'`` than ``'B'`` (ANO-UTILS-038).

    A business day skips the week-ends: 5 observations per week, hence 7/5 of a calendar day
    between two dates on average. The comparison of multiplied frequencies uses that duration.
    """

    def test_day_is_higher_than_business_day_without_multiplier(self):
        """Table order: the business day comes after the day."""
        assert is_higher_frequency('D', 'B') is True
        assert is_higher_frequency('B', 'D') is False

    @pytest.mark.parametrize("multiplier", [2, 3, 7, 24])
    def test_equal_multiplier_keeps_the_day_higher_than_the_business_day(self, multiplier):
        """With the same multiplier, ``kD`` is higher than ``kB`` and not the other way round."""
        assert is_higher_frequency(f"{multiplier}D", f"{multiplier}B") is True
        assert is_higher_frequency(f"{multiplier}B", f"{multiplier}D") is False

    @pytest.mark.parametrize(
        "freq1, freq2, expected",
        [
            pytest.param('4D', '3B', True, id="4-days-vs-3-business-days"),
            pytest.param('3B', '4D', False, id="3-business-days-vs-4-days"),
            pytest.param('2B', '3D', True, id="2-business-days-vs-3-days"),
            pytest.param('24h', 'B', True, id="24h-vs-business-day"),
            pytest.param('B', '24h', False, id="business-day-vs-24h"),
        ],
    )
    def test_business_day_lasts_seven_fifths_of_a_day(self, freq1, freq2, expected):
        """Golden values: 3B = 4.2 days (lower frequency than 4D), 2B = 2.8 days (higher than 3D)."""
        assert is_higher_frequency(freq1, freq2) is expected

    @pytest.mark.parametrize("freq1, freq2", [('5B', 'W'), ('5B', '7D'), ('W', '5B')])
    def test_five_business_days_are_a_week(self, freq1, freq2):
        """5 business days span 7 calendar days: neither is higher than a week."""
        assert is_higher_frequency(freq1, freq2) is False

    def test_incomparability_is_transitive(self):
        """Property: on multiplied frequencies, ``a ~ b`` and ``b ~ c`` (equal durations) imply ``a ~ c``.

        This is what the ``'D'`` / ``'24h'`` / ``'B'`` triple used to break.
        """
        higher = _order_matrix(MULTIPLIED_FREQUENCIES)
        tied = lambda a, b: not higher[a, b] and not higher[b, a]
        violations = [
            (a, b, c) for a, b, c in itertools.product(MULTIPLIED_FREQUENCIES, repeat=3)
            if tied(a, b) and tied(b, c) and not tied(a, c)
        ]
        assert violations == []


# =============================================================================
# get_frequency_order
# =============================================================================

class TestGetFrequencyOrder:
    """``get_frequency_order``: numeric rank of the granularity of a frequency."""

    def test_order_increases_along_the_granularity_table(self):
        """Each code has a strictly greater order than the finer one before it."""
        orders = [get_frequency_order(code) for code in ORDERED_CODES]
        assert all(finer < coarser for finer, coarser in zip(orders, orders[1:]))

    def test_documented_values(self):
        """Golden values of the docstring: daily is 7, monthly is 9, quarterly comes after monthly."""
        assert get_frequency_order('daily') == 7
        assert get_frequency_order('monthly') == 9
        assert get_frequency_order('quarterly') > get_frequency_order('monthly')

    @pytest.mark.parametrize("code", ORDERED_CODES)
    def test_every_spelling_has_the_order_of_its_code(self, code):
        """Literal name, position and anchor do not change the order."""
        assert {get_frequency_order(spelling) for spelling in SPELLINGS[code]} == {get_frequency_order(code)}

    @pytest.mark.parametrize("value, code", [('2MS', 'M'), ('3QS-FEB', 'Q'), ('15min', 'min'), ('2D', 'D')])
    def test_multiplier_is_ignored(self, value, code):
        """The order is the one of the granularity: a leading multiplier does not shift it.

        Unlike ``is_higher_frequency``, which accounts for the multiplier.
        """
        assert get_frequency_order(value) == get_frequency_order(code)

    def test_order_is_a_number(self):
        """The order is numeric (an ``int`` for most codes, a ``float`` for ``'B'`` and ``'SM'``)."""
        assert all(isinstance(get_frequency_order(c), (int, float)) for c in ORDERED_CODES)

    @pytest.mark.parametrize("value", ['xyz', 'A', 'AS-JAN', '', 'T', 'MONTHLY', None, 5])
    def test_unsupported_frequency_raises(self, value):
        """An unsupported frequency is rejected; it is never given the order 0.

        The docstring says the function "returns 0 if frequency is not found": see ANO-UTILS-035.
        """
        with pytest.raises(ValueError):
            get_frequency_order(value)
