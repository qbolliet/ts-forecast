"""Tests for FrequencyNormalizer class.

This test suite verifies:
1. Backward compatibility with simple codes and literal names
2. New functionality for complex pandas frequency strings
3. Edge cases and error handling
4. Integration with other frequency utilities
"""
import pytest
import pandas as pd
from tsforecast.utils.frequency.normalizer import FrequencyNormalizer


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

    def test_to_pandas_freq_alias(self, normalizer):
        """Test that to_pandas_freq() is an alias for normalize()."""
        assert normalizer.to_pandas_freq('monthly') == 'M'
        assert normalizer.to_pandas_freq('Q') == 'Q'


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
        assert normalizer.normalize('AS') == 'Y'  # 'A' normalized to 'Y'
        assert normalizer.normalize('YS') == 'Y'

    def test_normalize_position_strings_end(self, normalizer):
        """Test extraction from position strings with 'E' suffix (end)."""
        assert normalizer.normalize('ME') == 'M'
        assert normalizer.normalize('QE') == 'Q'
        assert normalizer.normalize('AE') == 'Y'  # 'A' normalized to 'Y'
        assert normalizer.normalize('YE') == 'Y'

    def test_normalize_anchor_strings_quarter(self, normalizer):
        """Test extraction from quarter anchor strings."""
        assert normalizer.normalize('QE-DEC') == 'Q'
        assert normalizer.normalize('QS-JAN') == 'Q'
        assert normalizer.normalize('QE-MAR') == 'Q'
        assert normalizer.normalize('QS-APR') == 'Q'
        assert normalizer.normalize('QE-JUN') == 'Q'

    def test_normalize_anchor_strings_year(self, normalizer):
        """Test extraction from year anchor strings."""
        assert normalizer.normalize('AS-JAN') == 'Y'
        assert normalizer.normalize('AE-DEC') == 'Y'
        assert normalizer.normalize('AS-MAR') == 'Y'
        assert normalizer.normalize('YS-JAN') == 'Y'
        assert normalizer.normalize('YE-DEC') == 'Y'

    def test_normalize_week_anchor_strings(self, normalizer):
        """Test extraction from week anchor strings."""
        assert normalizer.normalize('W-SUN') == 'W'
        assert normalizer.normalize('W-MON') == 'W'

    def test_extract_base_frequency_direct(self, normalizer):
        """Test _extract_base_frequency() method directly."""
        assert normalizer._extract_base_frequency('QE-DEC') == 'Q'
        assert normalizer._extract_base_frequency('MS') == 'M'
        assert normalizer._extract_base_frequency('AS-JAN') == 'Y'
        assert normalizer._extract_base_frequency('invalid') is None
        assert normalizer._extract_base_frequency('123') is None


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
        assert normalizer.is_higher_frequency('QE-DEC', 'AS-JAN') is True
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
            'AS-JAN',  # Year start January
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
        complex_strings = ['MS', 'ME', 'QS', 'QE', 'QE-DEC', 'AS-JAN', 'W-SUN']

        for freq in complex_strings:
            assert normalizer.validate(freq) is True, f"Failed to validate {freq}"
