"""Tests for the ``FrequencyNormalizer`` class.

Covers the mappings between codes and literal names, the extraction of the base
code from complex pandas strings (position, anchor, multiplier), the rebuilding
of pandas strings (``to_pandas_freq`` / ``to_dateoffset``), the explicit
rejection of the pandas aliases abandoned by the package (``'A'``, ``'AS'``,
``'AE'``), ``validate`` / ``are_compatible_frequencies`` and the comparison
``is_higher_frequency``.

The module-level functions of ``utils.py`` (``normalize_frequency`` and its
``return_format``, ``get_frequency_order``...) are tested in ``test_utils.py``.
"""
import warnings
from typing import get_args

import pandas as pd
import pytest

from tsforecast.utils.frequency.normalizer import FrequencyNormalizer
from tsforecast.utils.frequency.types import FrequencyType, UserFrequencyType

# Codes supportés par le package et noms littéraux associés (valeurs d'or, écrites
# indépendamment des dictionnaires internes du normaliseur)
CODE_TO_LITERAL = {
    'ns': 'nanosecond', 'us': 'microsecond', 'ms': 'millisecond', 's': 'second',
    'min': 'minute', 'h': 'hourly', 'D': 'daily', 'B': 'business_daily',
    'W': 'weekly', 'SM': 'semi_monthly', 'M': 'monthly', 'Q': 'quarterly', 'Y': 'annual',
}


class TestFrequencyNormalizerBasic:
    """Test basic normalization functionality (backward compatibility)."""

    @pytest.fixture
    def normalizer(self):
        """Create a FrequencyNormalizer instance for testing."""
        return FrequencyNormalizer()

    def test_normalize_simple_codes_unchanged(self, normalizer):
        """Test that simple pandas codes are returned unchanged."""
        assert normalizer.normalize('D') == 'D'
        assert normalizer.normalize('M') == 'M'
        assert normalizer.normalize('Q') == 'Q'
        assert normalizer.normalize('Y') == 'Y'
        assert normalizer.normalize('W') == 'W'
        assert normalizer.normalize('B') == 'B'
        assert normalizer.normalize('h') == 'h'
        assert normalizer.normalize('min') == 'min'
        assert normalizer.normalize('s') == 's'

    def test_normalize_literal_names(self, normalizer):
        """Test conversion from literal names to pandas codes."""
        assert normalizer.normalize('daily') == 'D'
        assert normalizer.normalize('monthly') == 'M'
        assert normalizer.normalize('quarterly') == 'Q'
        assert normalizer.normalize('annual') == 'Y'
        assert normalizer.normalize('weekly') == 'W'
        assert normalizer.normalize('business_daily') == 'B'
        assert normalizer.normalize('hourly') == 'h'

    def test_to_literal_conversion(self, normalizer):
        """Test conversion from codes to literal names."""
        assert normalizer.to_literal('D') == 'daily'
        assert normalizer.to_literal('M') == 'monthly'
        assert normalizer.to_literal('Q') == 'quarterly'
        assert normalizer.to_literal('Y') == 'annual'

    def test_to_code_alias(self, normalizer):
        """Test that to_code() is an alias for normalize()."""
        assert normalizer.to_code('daily') == 'D'
        assert normalizer.to_code('M') == 'M'

    def test_to_pandas_freq_of_plain_codes_and_literals(self, normalizer):
        """to_pandas_freq() maps literal names and plain codes to pandas aliases (end variant by default)."""
        assert normalizer.to_pandas_freq('monthly') == 'ME'
        assert normalizer.to_pandas_freq('Q') == 'QE'
        assert normalizer.to_pandas_freq('D') == 'D'

    def test_to_pandas_freq_is_no_longer_an_alias_of_normalize(self, normalizer):
        """Unlike normalize(), to_pandas_freq() keeps the position and the anchor."""
        assert normalizer.normalize('QE-DEC') == 'Q'
        assert normalizer.to_pandas_freq('QE-DEC') == 'QE-DEC'


class TestFrequencyNormalizerComplexStrings:
    """Test extraction of base frequencies from complex pandas strings."""

    @pytest.fixture
    def normalizer(self):
        """Create a FrequencyNormalizer instance for testing."""
        return FrequencyNormalizer()

    def test_normalize_position_strings_start(self, normalizer):
        """Test extraction from position strings with 'S' suffix (start)."""
        assert normalizer.normalize('MS') == 'M'
        assert normalizer.normalize('QS') == 'Q'
        assert normalizer.normalize('YS') == 'Y'
        assert normalizer.normalize('SMS') == 'SM'

    def test_normalize_position_strings_end(self, normalizer):
        """Test extraction from position strings with 'E' suffix (end)."""
        assert normalizer.normalize('ME') == 'M'
        assert normalizer.normalize('QE') == 'Q'
        assert normalizer.normalize('YE') == 'Y'
        assert normalizer.normalize('SME') == 'SM'

    def test_normalize_anchor_strings_quarter(self, normalizer):
        """Test extraction from quarter anchor strings."""
        assert normalizer.normalize('QE-DEC') == 'Q'
        assert normalizer.normalize('QS-JAN') == 'Q'
        assert normalizer.normalize('QE-MAR') == 'Q'
        assert normalizer.normalize('QS-APR') == 'Q'
        assert normalizer.normalize('QE-JUN') == 'Q'

    def test_normalize_anchor_strings_year(self, normalizer):
        """Test extraction from year anchor strings."""
        assert normalizer.normalize('YS-JAN') == 'Y'
        assert normalizer.normalize('YE-DEC') == 'Y'
        assert normalizer.normalize('YS-MAR') == 'Y'
        assert normalizer.normalize('YE-JUN') == 'Y'

    def test_normalize_week_anchor_strings(self, normalizer):
        """Test extraction from week anchor strings."""
        assert normalizer.normalize('W-SUN') == 'W'
        assert normalizer.normalize('W-MON') == 'W'


class TestFrequencyNormalizerMultiplier:
    """A leading multiplier ('2MS') is parsed by ``parse_frequency``."""

    @pytest.fixture
    def normalizer(self):
        """Create a FrequencyNormalizer instance for testing."""
        return FrequencyNormalizer()

    @pytest.mark.parametrize("value, expected", [("2MS", "M"), ("3QS-FEB", "Q"), ("2D", "D"), ("15min", "min")])
    def test_normalize_drops_the_multiplier(self, normalizer, value, expected):
        """Like the position and the anchor, the multiplier is left out of the base code."""
        assert normalizer.normalize(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [("2MS", ("M", 2)), ("3QS-FEB", ("Q", 3)), ("MS", ("M", 1)), ("daily", ("D", 1)),
         ("business_daily", ("B", 1)), ("15min", ("min", 15))],
    )
    def test_normalize_with_multiplier(self, normalizer, value, expected):
        """The code and the multiplier are returned together."""
        assert normalizer.normalize_with_multiplier(value) == expected

    @pytest.mark.parametrize("value", ["2foo", "", None])
    def test_normalize_with_multiplier_rejects_invalid_values(self, normalizer, value):
        """An invalid value is rejected like by ``normalize``."""
        with pytest.raises(ValueError):
            normalizer.normalize_with_multiplier(value)

    @pytest.mark.parametrize(
        "freq1, freq2, expected",
        [
            # Même base : le plus petit multiplicateur est la fréquence la plus élevée
            pytest.param("MS", "2MS", True, id="same-base-lower-multiplier"),
            pytest.param("2MS", "MS", False, id="same-base-higher-multiplier"),
            pytest.param("2MS", "2ME", False, id="same-multiplier-same-base"),
            # Bases différentes : durées nominales
            pytest.param("2MS", "QS", True, id="two-months-vs-quarter"),
            pytest.param("4MS", "QS", False, id="four-months-vs-quarter"),
            pytest.param("QS", "2MS", False, id="quarter-vs-two-months"),
            pytest.param("2D", "W", True, id="two-days-vs-week"),
            pytest.param("15min", "h", True, id="fifteen-minutes-vs-hour"),
            # Sans multiplicateur : ordre des codes inchangé
            pytest.param("daily", "monthly", True, id="no-multiplier"),
            pytest.param("B", "D", False, id="business-day-vs-day"),
        ],
    )
    def test_is_higher_frequency_with_multiplier(self, normalizer, freq1, freq2, expected):
        """A multiplier lengthens the period, hence lowers the frequency."""
        assert normalizer.is_higher_frequency(freq1, freq2) is expected

    def test_validate_accepts_a_multiplied_frequency(self, normalizer):
        """A multiplied frequency is a supported frequency."""
        assert normalizer.validate("2MS") is True

    @pytest.mark.parametrize("value", ["2MS", "3QS-FEB", "2W-MON", "15min", "2D"])
    def test_to_pandas_freq_keeps_the_multiplier(self, normalizer, value):
        """The multiplier is carried over when the string is rebuilt."""
        assert normalizer.to_pandas_freq(value) == value

    def test_to_dateoffset_honours_the_multiplier(self, normalizer):
        """The resulting offset steps by the multiplier."""
        assert normalizer.to_dateoffset('2MS').n == 2

    def test_unsupported_base_after_multiplier_raises(self, normalizer):
        """The multiplier does not make an unknown base frequency valid."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            normalizer.normalize('2foo')


class TestFrequencyNormalizerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def normalizer(self):
        """Create a FrequencyNormalizer instance for testing."""
        return FrequencyNormalizer()

    def test_normalize_invalid_type(self, normalizer):
        """Test that non-string types raise ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            normalizer.normalize(123)

        with pytest.raises(ValueError, match="must be a string"):
            normalizer.normalize(None)

        with pytest.raises(ValueError, match="must be a string"):
            normalizer.normalize(['M'])

    def test_normalize_invalid_string(self, normalizer):
        """Test that invalid frequency strings raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            normalizer.normalize('invalid_freq')

        with pytest.raises(ValueError, match="Unsupported frequency"):
            normalizer.normalize('XYZ')

        with pytest.raises(ValueError, match="Unsupported frequency"):
            normalizer.normalize('123')

    def test_normalize_empty_string(self, normalizer):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            normalizer.normalize('')

    def test_normalize_lowercase_not_supported(self, normalizer):
        """Test that lowercase complex strings are not currently supported."""
        # Note: This is a known limitation (future enhancement)
        with pytest.raises(ValueError, match="Unsupported frequency"):
            normalizer.normalize('qe-dec')

    def test_validate_method(self, normalizer):
        """Test validate() method returns correct boolean values."""
        # Valid frequencies
        assert normalizer.validate('D') is True
        assert normalizer.validate('monthly') is True
        assert normalizer.validate('QE-DEC') is True
        assert normalizer.validate('MS') is True

        # Invalid frequencies
        assert normalizer.validate('invalid_freq') is False
        assert normalizer.validate('XYZ') is False
        assert normalizer.validate('') is False

    def test_validate_with_invalid_type(self, normalizer):
        """Test validate() with invalid types."""
        assert normalizer.validate(123) is False
        assert normalizer.validate(None) is False


class TestFrequencyNormalizerBackwardCompatibility:
    """Test that existing functionality remains unchanged."""

    @pytest.fixture
    def normalizer(self):
        """Create a FrequencyNormalizer instance for testing."""
        return FrequencyNormalizer()

    def test_is_higher_frequency_simple_codes(self, normalizer):
        """Test is_higher_frequency() with simple codes."""
        assert normalizer.is_higher_frequency('D', 'M') is True
        assert normalizer.is_higher_frequency('M', 'Q') is True
        assert normalizer.is_higher_frequency('Q', 'Y') is True
        assert normalizer.is_higher_frequency('Y', 'Q') is False

    def test_is_higher_frequency_literal_names(self, normalizer):
        """Test is_higher_frequency() with literal names."""
        assert normalizer.is_higher_frequency('daily', 'monthly') is True
        assert normalizer.is_higher_frequency('monthly', 'quarterly') is True
        assert normalizer.is_higher_frequency('quarterly', 'annual') is True

    def test_are_compatible_frequencies(self, normalizer):
        """Test are_compatible_frequencies() method."""
        assert normalizer.are_compatible_frequencies('D', 'M') is True
        assert normalizer.are_compatible_frequencies('daily', 'monthly') is True
        assert normalizer.are_compatible_frequencies('B', 'M') is True

        # Invalid frequency should return False
        assert normalizer.are_compatible_frequencies('invalid', 'M') is False

    def test_to_dateoffset_simple(self, normalizer):
        """Test to_dateoffset() with simple frequencies."""
        offset = normalizer.to_dateoffset('M')
        assert isinstance(offset, pd.DateOffset)

        offset = normalizer.to_dateoffset('monthly')
        assert isinstance(offset, pd.DateOffset)


class TestFrequencyNormalizerIntegration:
    """Test integration with complex pandas strings in real-world scenarios."""

    @pytest.fixture
    def normalizer(self):
        """Create a FrequencyNormalizer instance for testing."""
        return FrequencyNormalizer()

    def test_normalize_then_to_literal(self, normalizer):
        """Test normalization followed by literal conversion."""
        # Complex string → normalize → to_literal
        normalized = normalizer.normalize('QE-DEC')
        assert normalized == 'Q'

        literal = normalizer.to_literal(normalized)
        assert literal == 'quarterly'

    def test_normalize_complex_then_is_higher_frequency(self, normalizer):
        """Test that normalized complex strings work with is_higher_frequency()."""
        # This should work without manual split() workaround
        assert normalizer.is_higher_frequency('MS', 'QE-DEC') is True
        assert normalizer.is_higher_frequency('QE-DEC', 'YS-JAN') is True
        assert normalizer.is_higher_frequency('D', 'MS') is True

    def test_normalize_complex_then_are_compatible(self, normalizer):
        """Test that complex strings work with are_compatible_frequencies()."""
        assert normalizer.are_compatible_frequencies('MS', 'QE-DEC') is True
        assert normalizer.are_compatible_frequencies('QE-DEC', 'M') is True

    def test_frequency_order_consistency(self, normalizer):
        """Test that frequency ordering is consistent across representations."""
        # All these should be considered same frequency level
        freqs = ['M', 'monthly', 'MS', 'ME']
        orders = [normalizer._frequency_order.get(normalizer.normalize(f), 0) for f in freqs]

        # All should have same order value (9 for monthly)
        assert all(order == 9 for order in orders), f"Orders not consistent: {orders}"

    def test_real_world_frequency_detector_output(self, normalizer):
        """Test normalization of typical FrequencyDetector output strings."""
        # These are typical outputs from FrequencyDetector.detect()
        detector_outputs = [
            'MS',      # Month start
            'QE-DEC',  # Quarter end December (fiscal year)
            'YS-JAN',  # Year start January
            'W-SUN',   # Week ending Sunday
            'D',       # Daily (unchanged)
            'M',       # Monthly (unchanged)
        ]

        expected = ['M', 'Q', 'Y', 'W', 'D', 'M']

        for output, expected_base in zip(detector_outputs, expected):
            assert normalizer.normalize(output) == expected_base

    def test_chaining_operations(self, normalizer):
        """Test chaining multiple operations without errors."""
        # Start with complex string
        freq = 'QE-DEC'

        # Chain: normalize → to_literal → normalize again
        step1 = normalizer.normalize(freq)
        assert step1 == 'Q'

        step2 = normalizer.to_literal(step1)
        assert step2 == 'quarterly'

        step3 = normalizer.normalize(step2)
        assert step3 == 'Q'

    def test_validate_complex_strings(self, normalizer):
        """Test that validate() works correctly with complex strings."""
        complex_strings = ['MS', 'ME', 'QS', 'QE', 'QE-DEC', 'YS-JAN', 'YE-DEC', 'W-SUN']

        for freq in complex_strings:
            assert normalizer.validate(freq) is True, f"Failed to validate {freq}"


# Alias pandas abandonnés : ancienne écriture de l'année (dépréciée par pandas 2.2 au profit
# de 'Y' / 'YS' / 'YE'), avec position, ancre et multiplicateur
REMOVED_YEAR_ALIASES = ['A', 'AS', 'AE', 'AS-JAN', 'AE-DEC', 'AS-MAR', 'A-DEC', '2A', '2AS-JAN']


class TestRemovedAliasRejection:
    """The pandas aliases ``'A'`` / ``'AS'`` / ``'AE'`` are explicitly rejected.

    Their abandon is deliberate: ``normalize`` stopped mapping ``'A'`` to ``'Y'`` in commit
    ``9ce944e`` and ``'A'`` left ``FrequencyType`` in ``0ec1f29``. The year is written
    ``'Y'`` / ``'YS'`` / ``'YE'``. Every entry point must fail loudly rather than translate
    the old spelling.
    """

    @pytest.fixture
    def normalizer(self):
        """Create a FrequencyNormalizer instance for testing."""
        return FrequencyNormalizer()

    @pytest.mark.parametrize("alias", REMOVED_YEAR_ALIASES)
    @pytest.mark.parametrize("method", ['normalize', 'to_literal', 'to_code', 'to_pandas_freq', 'to_dateoffset'])
    def test_every_conversion_rejects_the_alias(self, normalizer, method, alias):
        """Each conversion method raises ``ValueError('Unsupported frequency ...')``."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            getattr(normalizer, method)(alias)

    @pytest.mark.parametrize("alias", REMOVED_YEAR_ALIASES)
    def test_error_message_names_the_input_and_lists_the_year_code(self, normalizer, alias):
        """The message quotes the rejected string and offers ``'Y'`` among the supported codes."""
        with pytest.raises(ValueError) as error:
            normalizer.normalize(alias)
        message = str(error.value)
        assert alias in message
        assert "'Y'" in message and "'annual'" in message

    @pytest.mark.parametrize("alias", REMOVED_YEAR_ALIASES)
    def test_validate_is_false(self, normalizer, alias):
        """``validate`` reports the alias as unsupported instead of raising."""
        assert normalizer.validate(alias) is False

    @pytest.mark.parametrize("alias", ['A', 'AS-JAN', '2A'])
    def test_comparisons_reject_the_alias_on_either_side(self, normalizer, alias):
        """``is_higher_frequency`` does not fall back on a default order for the alias."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            normalizer.is_higher_frequency(alias, 'M')
        with pytest.raises(ValueError, match="Unsupported frequency"):
            normalizer.is_higher_frequency('M', alias)

    @pytest.mark.parametrize("alias", ['A', 'AS-JAN'])
    def test_are_compatible_frequencies_is_false(self, normalizer, alias):
        """A pair containing the alias is not compatible."""
        assert normalizer.are_compatible_frequencies(alias, 'M') is False
        assert normalizer.are_compatible_frequencies('M', alias) is False

    @pytest.mark.internal
    def test_alias_is_absent_from_the_internal_tables(self, normalizer):
        """``'A'`` has no entry in the code, literal or order tables."""
        assert 'A' not in normalizer._pandas_to_literal
        assert 'A' not in normalizer._frequency_order
        assert 'A' not in normalizer._literal_to_pandas.values()

    def test_alias_is_absent_from_the_declared_types(self):
        """``FrequencyType`` no longer declares ``'A'`` (nor the ``'T'`` minute alias)."""
        assert 'A' not in get_args(FrequencyType)
        assert 'T' not in get_args(FrequencyType)

    @pytest.mark.parametrize("alias", ['T', 'H'])
    def test_legacy_subdaily_aliases_are_rejected_too(self, normalizer, alias):
        """The historical ``'T'`` (minute) and ``'H'`` (hour) aliases are not supported either."""
        with pytest.raises(ValueError, match="Unsupported frequency"):
            normalizer.normalize(alias)

    @pytest.mark.parametrize(
        "modern", ['Y', 'YS', 'YE', 'YS-JAN', 'YE-DEC', 'annual'],
    )
    def test_year_is_written_with_y(self, normalizer, modern):
        """The supported spellings of the year all normalize to ``'Y'``."""
        assert normalizer.normalize(modern) == 'Y'


class TestSupportedFrequencyMappings:
    """Tables between codes and literal names, and their declared types."""

    @pytest.fixture
    def normalizer(self):
        """Create a FrequencyNormalizer instance for testing."""
        return FrequencyNormalizer()

    def test_declared_codes_are_the_supported_codes(self):
        """``FrequencyType`` declares exactly the supported codes (no phantom value)."""
        assert set(get_args(FrequencyType)) == set(CODE_TO_LITERAL)

    def test_declared_literals_are_the_supported_literals(self):
        """``UserFrequencyType`` declares exactly the supported literal names."""
        assert set(get_args(UserFrequencyType)) == set(CODE_TO_LITERAL.values())

    @pytest.mark.parametrize("code, literal", list(CODE_TO_LITERAL.items()))
    def test_code_and_literal_correspondence(self, normalizer, code, literal):
        """Each code is its own normal form and each literal name maps to its code."""
        assert normalizer.normalize(code) == code
        assert normalizer.normalize(literal) == code
        assert normalizer.to_literal(code) == literal

    @pytest.mark.parametrize("code, literal", list(CODE_TO_LITERAL.items()))
    def test_round_trip_code_literal_code(self, normalizer, code, literal):
        """``to_code(to_literal(code))`` is the identity, and ``to_literal`` is idempotent."""
        assert normalizer.to_code(normalizer.to_literal(code)) == code
        assert normalizer.to_literal(literal) == literal

    @pytest.mark.parametrize("code, literal", list(CODE_TO_LITERAL.items()))
    def test_to_pandas_freq_of_a_literal_name_is_the_alias_of_its_code(self, normalizer, code, literal):
        """A literal name is rebuilt as the pandas alias it stands for (end variant by default)."""
        expected = f"{code}E" if code in {'M', 'Q', 'Y', 'SM'} else code
        assert normalizer.to_pandas_freq(literal) == expected

    @pytest.mark.parametrize("value", ['M', 'monthly', 'QE-DEC', 'MS', 'W-MON', '2MS'])
    def test_to_code_is_normalize(self, normalizer, value):
        """``to_code`` gives the same result as ``normalize``."""
        assert normalizer.to_code(value) == normalizer.normalize(value)

    @pytest.mark.parametrize("value", ['xyz', '', 'A', None, 5])
    def test_to_code_fails_like_normalize(self, normalizer, value):
        """``to_code`` raises the very same error as ``normalize``."""
        with pytest.raises(ValueError) as expected:
            normalizer.normalize(value)
        with pytest.raises(ValueError) as obtained:
            normalizer.to_code(value)
        assert str(obtained.value) == str(expected.value)

    @pytest.mark.parametrize(
        "value, expected",
        [('QE-DEC', 'quarterly'), ('MS', 'monthly'), ('YS-JAN', 'annual'), ('W-MON', 'weekly'), ('2D', 'daily')],
    )
    def test_to_literal_of_complex_strings(self, normalizer, value, expected):
        """The literal name of a complex string is the one of its base code."""
        assert normalizer.to_literal(value) == expected

    @pytest.mark.internal
    def test_order_table_covers_exactly_the_codes(self, normalizer):
        """The granularity order is defined for every supported code and nothing else."""
        assert set(normalizer._frequency_order) == set(CODE_TO_LITERAL)


class TestRebuiltPandasStrings:
    """``to_pandas_freq`` / ``to_dateoffset`` keep position, anchor and multiplier."""

    @pytest.fixture
    def normalizer(self):
        """Create a FrequencyNormalizer instance for testing."""
        return FrequencyNormalizer()

    @pytest.mark.parametrize(
        "value",
        ['MS', 'ME', 'QS', 'QE', 'YS', 'YE', 'SMS', 'SME', 'QE-DEC', 'QS-FEB', 'YS-JAN', 'YE-JUN',
         'W-MON', 'W-SUN', '2MS', '3QS-FEB', '2W-MON', '15min', '2D'],
    )
    def test_valid_pandas_string_is_unchanged(self, normalizer, value):
        """A valid pandas string goes through the parse / normalize / rebuild round trip unchanged."""
        assert normalizer.to_pandas_freq(value) == value

    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param('MS', pd.offsets.MonthBegin(), id="month-start"),
            pytest.param('ME', pd.offsets.MonthEnd(), id="month-end"),
            pytest.param('QS', pd.offsets.QuarterBegin(startingMonth=1), id="quarter-start"),
            pytest.param('QE-DEC', pd.offsets.QuarterEnd(startingMonth=12), id="quarter-end"),
            pytest.param('YS-JAN', pd.offsets.YearBegin(month=1), id="year-start"),
            pytest.param('YE-JUN', pd.offsets.YearEnd(month=6), id="year-end-june"),
            pytest.param('SMS', pd.offsets.SemiMonthBegin(), id="semi-month-start"),
            pytest.param('W-MON', pd.offsets.Week(weekday=0), id="week-monday"),
            pytest.param('2MS', pd.offsets.MonthBegin(2), id="multiplied"),
            pytest.param('daily', pd.offsets.Day(), id="literal-daily"),
            pytest.param('business_daily', pd.offsets.BusinessDay(), id="literal-business-daily"),
        ],
    )
    def test_to_dateoffset_keeps_the_position(self, normalizer, value, expected):
        """The start / end position and the anchor survive the conversion into an offset.

        The start and end positions give different offsets (``MonthBegin`` vs ``MonthEnd``),
        contrary to what the first version of ``frequency_normalizer.ipynb`` reported.
        """
        # Aucun alias déprécié ici : toute alerte pandas serait une régression
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert normalizer.to_dateoffset(value) == expected

    def test_start_and_end_offsets_differ(self, normalizer):
        """``'MS'`` and ``'ME'`` are distinct offsets."""
        assert normalizer.to_dateoffset('MS') != normalizer.to_dateoffset('ME')

    @pytest.mark.parametrize(
        "value, multiplier", [('2MS', 2), ('3QS-FEB', 3), ('15min', 15), ('2D', 2), ('MS', 1)],
    )
    def test_to_dateoffset_honours_the_multiplier(self, normalizer, value, multiplier):
        """The offset steps by the leading multiplier."""
        assert normalizer.to_dateoffset(value).n == multiplier

    @pytest.mark.parametrize("value", ['DS', 'DE', 'WS', 'BE', 'hS', 'minE', 'sS', 'msE'])
    def test_position_is_dropped_where_pandas_has_no_start_end_variant(self, normalizer, value):
        """A position on a code without S / E variant is silently ignored (``build_frequency_string``)."""
        assert normalizer.to_pandas_freq(value) == normalizer.normalize(value)

    @pytest.mark.parametrize(
        "value",
        [pytest.param(123, id="int"), pytest.param(3.14, id="float"), pytest.param(['M'], id="list"),
         pytest.param({'a': 1}, id="dict"), pytest.param(b'M', id="bytes"), pytest.param(None, id="none")],
    )
    @pytest.mark.parametrize("method", ['to_pandas_freq', 'to_dateoffset'])
    def test_non_string_input_raises_value_error(self, normalizer, method, value):
        """Like ``normalize`` / ``to_literal``, a non-string is a ``ValueError`` (ANO-UTILS-037)."""
        with pytest.raises(ValueError, match="must be a string"):
            getattr(normalizer, method)(value)

    @pytest.mark.parametrize("value", ['AS', 'AS-JAN', 'BQS', 'xyz'])
    def test_error_names_the_whole_input(self, normalizer, value):
        """The message quotes the string given, not the fragment left after parsing (``'AS'``, not ``'A'``)."""
        with pytest.raises(ValueError) as error:
            normalizer.to_pandas_freq(value)
        assert f"Unsupported frequency: {value}." in str(error.value)

    @pytest.mark.parametrize("value", ['xyz', '', 'M ', ' M', '-1D', '0D', 'MONTHLY', 'qe-dec'])
    @pytest.mark.parametrize("method", ['to_pandas_freq', 'to_dateoffset'])
    def test_invalid_strings_raise_value_error(self, normalizer, method, value):
        """Unknown, blank-padded, non-positive multiplier or lower-cased strings are rejected."""
        with pytest.raises(ValueError):
            getattr(normalizer, method)(value)


# Chaînes base + position + ancre confrontées à pandas. Les fréquences sans variante début / fin
# (W, D, B) n'ont pas de position : le package en ignorerait une silencieusement
# (voir build_frequency_string), pandas la rejetterait
PANDAS_ANCHOR_CANDIDATES = [
    f"{base}{position}-{anchor}"
    for base, positions in [('M', 'SE'), ('Q', 'SE'), ('Y', 'SE'), ('SM', 'SE'), ('W', ''), ('D', ''), ('B', '')]
    for position in ('', *positions)
    for anchor in [
        'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',
        'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN', '1', '2', '15', '27', '28', 'XYZ',
    ]
]


class TestBareCodesGiveTheEndVariant:
    """A code with start / end variants and no position gets its end variant (ANO-UTILS-039).

    ``'M'``, ``'Q'``, ``'Y'`` and ``'SM'`` are the aliases pandas 2.2 deprecates (removed in
    pandas 3); they designated the end of the period. ``to_pandas_freq`` therefore answers
    ``'ME'``, ``'QE'``, ``'YE'`` and ``'SME'`` and the offsets are built without any warning.
    The base code (``'M'``) stays what ``normalize`` / ``to_code`` return.
    """

    @pytest.fixture
    def normalizer(self):
        """Create a FrequencyNormalizer instance for testing."""
        return FrequencyNormalizer()

    @pytest.mark.parametrize(
        "value, expected",
        [
            ('monthly', 'ME'), ('quarterly', 'QE'), ('annual', 'YE'), ('semi_monthly', 'SME'),
            ('M', 'ME'), ('Q', 'QE'), ('Y', 'YE'), ('SM', 'SME'),
            ('2M', '2ME'), ('3Q', '3QE'), ('Q-DEC', 'QE-DEC'), ('Y-JUN', 'YE-JUN'),
        ],
    )
    def test_bare_code_becomes_the_end_alias(self, normalizer, value, expected):
        """Literal names, bare codes, multiplied and anchored bare codes get the end variant."""
        assert normalizer.to_pandas_freq(value) == expected

    @pytest.mark.parametrize("value", ['MS', 'QS', 'YS-JAN', 'SMS', 'QS-FEB', '2MS'])
    def test_explicit_start_position_is_not_overridden(self, normalizer, value):
        """The default only applies without position."""
        assert normalizer.to_pandas_freq(value) == value

    @pytest.mark.parametrize(
        "value, expected",
        [
            ('M', pd.offsets.MonthEnd()), ('Q', pd.offsets.QuarterEnd(startingMonth=12)),
            ('Y', pd.offsets.YearEnd(month=12)), ('SM', pd.offsets.SemiMonthEnd()),
            ('monthly', pd.offsets.MonthEnd()),
        ],
    )
    def test_offset_of_a_bare_code_is_the_end_of_period_offset_without_warning(self, normalizer, value, expected):
        """The offset is the one pandas 2 used for the bare alias, and pandas does not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert normalizer.to_dateoffset(value) == expected

    @pytest.mark.parametrize("value, base", [('monthly', 'M'), ('Q', 'Q'), ('YS-JAN', 'Y'), ('SM', 'SM')])
    def test_base_code_comes_from_normalize(self, normalizer, value, base):
        """To get the base code rather than an alias, ``normalize`` / ``to_code`` are the entry points."""
        assert normalizer.normalize(value) == normalizer.to_code(value) == base

    @pytest.mark.parametrize("code", ['M', 'Q', 'Y', 'SM'])
    def test_end_alias_generates_the_dates_of_the_former_bare_alias(self, normalizer, code):
        """Property: the end alias generates the same dates as the bare alias of pandas 2."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            former = pd.date_range('2023-01-01', periods=6, freq=code)
        current = pd.date_range('2023-01-01', periods=6, freq=normalizer.to_pandas_freq(code))
        assert list(current) == list(former)


class TestAnchorValidation:
    """The anchor after the dash is checked against the base frequency (ANO-UTILS-040).

    Rules of pandas: a month for ``Q`` / ``Y`` (``'QS-JAN'``, ``'Y-DEC'``), a weekday for ``W``
    (``'W-MON'``), a day of the month behind a position for ``SM`` (``'SMS-15'``: 2 to 27,
    ``'SME-15'``: 1 to 27), and no anchor for the other frequencies.
    """

    MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    WEEKDAYS = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

    @pytest.fixture
    def normalizer(self):
        """Create a FrequencyNormalizer instance for testing."""
        return FrequencyNormalizer()

    @pytest.mark.parametrize("month", MONTHS)
    @pytest.mark.parametrize("base", ['QS', 'QE', 'Q', 'YS', 'YE', 'Y'])
    def test_every_month_is_a_valid_quarterly_or_yearly_anchor(self, normalizer, base, month):
        """The twelve months anchor quarters and years, with or without position."""
        assert normalizer.validate(f"{base}-{month}") is True

    @pytest.mark.parametrize("weekday", WEEKDAYS)
    def test_every_weekday_is_a_valid_weekly_anchor(self, normalizer, weekday):
        """The seven weekdays anchor weeks."""
        assert normalizer.normalize(f"W-{weekday}") == 'W'

    @pytest.mark.parametrize("value", ['SMS-2', 'SMS-15', 'SMS-27', 'SME-1', 'SME-15', 'SME-27', '2SME-10'])
    def test_semi_monthly_day_of_month_is_valid(self, normalizer, value):
        """A day of the month anchors the semi-month, within the bounds pandas accepts."""
        assert normalizer.normalize(value) == 'SM'

    @pytest.mark.parametrize(
        "value",
        [
            'QS-XYZ', 'YE-FOO', 'Q-', 'QS-13', 'Y-MON',   # ni mois ni vide pour Q / Y
            'W-XYZ', 'W-JAN', 'W-',                        # ni jour pour W
            'MS-JAN', 'ME-1', 'M-JAN', 'D-MON', 'B-MON', 'h-1', 'min-5',   # aucune ancre admise
            'SMS-1', 'SMS-28', 'SME-0', 'SME-28', 'SM-15', 'SMS-XYZ', 'SMS-',   # jour hors bornes ou sans position
            'QS-jan', 'W-mon',                             # casse : seules les majuscules
        ],
    )
    def test_invalid_anchor_is_rejected(self, normalizer, value):
        """An anchor pandas does not read makes the frequency unsupported, everywhere."""
        assert normalizer.validate(value) is False
        for method in (normalizer.normalize, normalizer.to_pandas_freq, normalizer.to_dateoffset):
            with pytest.raises(ValueError, match="Invalid anchor|does not accept an anchor"):
                method(value)

    def test_error_message_gives_the_anchor_the_base_and_the_expected_values(self, normalizer):
        """The message says what was rejected and what would be accepted."""
        with pytest.raises(ValueError) as error:
            normalizer.normalize('QS-XYZ')
        message = str(error.value)
        assert "Unsupported frequency: QS-XYZ" in message
        assert "'XYZ'" in message and "'Q'" in message and "JAN" in message and "DEC" in message

    def test_error_message_for_a_frequency_without_anchor(self, normalizer):
        """A frequency that takes no anchor says so."""
        with pytest.raises(ValueError, match=r"Base frequency 'D' does not accept an anchor \('-MON'\)"):
            normalizer.normalize('D-MON')

    @pytest.mark.parametrize("value", ['2QS-XYZ', '3MS-JAN'])
    def test_multiplied_frequency_is_checked_too(self, normalizer, value):
        """The anchor of a multiplied frequency is checked like any other."""
        with pytest.raises(ValueError, match="Invalid anchor|does not accept an anchor"):
            normalizer.normalize(value)

    @pytest.mark.parametrize("value", ['QS-XYZ', 'MS-JAN'])
    def test_normalize_with_multiplier_rejects_an_invalid_anchor(self, normalizer, value):
        """The multiplier-aware variant validates through ``normalize``."""
        with pytest.raises(ValueError, match="Invalid anchor|does not accept an anchor"):
            normalizer.normalize_with_multiplier(value)

    @pytest.mark.parametrize("candidate", PANDAS_ANCHOR_CANDIDATES)
    def test_validation_agrees_with_pandas(self, normalizer, candidate):
        """Property: for a frequency the package supports, a valid anchor is one pandas accepts."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                pd.tseries.frequencies.to_offset(candidate)
                pandas_accepts = True
            except ValueError:
                pandas_accepts = False
        assert normalizer.validate(candidate) is pandas_accepts


class TestValidateAndCompatibility:
    """``validate`` never raises; ``are_compatible_frequencies`` is ``validate`` on both sides."""

    @pytest.fixture
    def normalizer(self):
        """Create a FrequencyNormalizer instance for testing."""
        return FrequencyNormalizer()

    @pytest.mark.parametrize(
        "value",
        [pytest.param(123, id="int"), pytest.param(3.14, id="float"), pytest.param(None, id="none"),
         pytest.param(['D'], id="list"), pytest.param({'a': 1}, id="dict"), pytest.param(b'M', id="bytes"),
         pytest.param(object(), id="object")],
    )
    def test_validate_of_a_non_string_is_false(self, normalizer, value):
        """Whatever the type, ``validate`` answers ``False`` instead of raising."""
        assert normalizer.validate(value) is False

    @pytest.mark.parametrize("value", ['d', 'MONTHLY', 'Monthly', 'MIN', 'Min', 'qe-dec', ' M', 'M ', 'm', 'q'])
    def test_matching_is_case_and_whitespace_sensitive(self, normalizer, value):
        """Lookups are exact: another case or a padded string is not a supported frequency."""
        assert normalizer.validate(value) is False

    @pytest.mark.parametrize(
        "freq1, freq2",
        [('D', 'M'), ('ns', 'Y'), ('B', 'monthly'), ('QE-DEC', 'MS'), ('D', 'xyz'), ('xyz', 'D'),
         ('xyz', 'abc'), ('A', 'D'), (None, 'D'), ('D', 5)],
    )
    def test_are_compatible_is_validate_on_both(self, normalizer, freq1, freq2):
        """No conversion ratio is checked: two supported frequencies are always compatible."""
        assert normalizer.are_compatible_frequencies(freq1, freq2) is (
            normalizer.validate(freq1) and normalizer.validate(freq2)
        )


class TestIsHigherFrequencyStrictness:
    """``is_higher_frequency`` is a strict comparison that ignores the spelling."""

    @pytest.fixture
    def normalizer(self):
        """Create a FrequencyNormalizer instance for testing."""
        return FrequencyNormalizer()

    @pytest.mark.parametrize(
        "freq1, freq2",
        [('M', 'M'), ('M', 'monthly'), ('MS', 'ME'), ('QE-DEC', 'QS-FEB'), ('D', 'D')],
    )
    def test_equal_frequencies_are_not_higher(self, normalizer, freq1, freq2):
        """A frequency is never higher than itself, whatever its position or anchor."""
        assert normalizer.is_higher_frequency(freq1, freq2) is False

    @pytest.mark.parametrize(
        "freq1, freq2, expected",
        [
            pytest.param('daily', 'QE-DEC', True, id="daily-vs-quarter-end"),
            pytest.param('QE-DEC', 'daily', False, id="quarter-end-vs-daily"),
            pytest.param('hourly', 'W-MON', True, id="hourly-vs-weekly-anchor"),
            pytest.param('YS-JAN', 'min', False, id="year-start-vs-minute"),
        ],
    )
    def test_spellings_can_be_mixed(self, normalizer, freq1, freq2, expected):
        """Literal names, codes and complex strings are compared through their base code."""
        assert normalizer.is_higher_frequency(freq1, freq2) is expected
