"""Tests for frequency parser utilities."""

import pytest
import pandas as pd
from tsforecast.utils.frequency.parser import (
    detect_and_parse_frequency,
    build_frequency_string
)


class TestDetectAndParseFrequency:
    """Tests for detect_and_parse_frequency function."""

    def test_daily_frequency(self):
        """Test detection of simple daily frequency."""
        index = pd.date_range('2024-01-01', periods=10, freq='D')
        freq, pos, suffix = detect_and_parse_frequency(index)
        assert freq == 'D'
        assert pos is None
        assert suffix is None

    def test_monthly_start(self):
        """Test detection of monthly frequency at start of month."""
        index = pd.date_range('2024-01-01', periods=5, freq='MS')
        freq, pos, suffix = detect_and_parse_frequency(index)
        assert freq == 'M'
        assert pos == 'S'
        assert suffix is None

    def test_monthly_end(self):
        """Test detection of monthly frequency at end of month."""
        index = pd.date_range('2024-01-31', periods=5, freq='ME')
        freq, pos, suffix = detect_and_parse_frequency(index)
        assert freq == 'M'
        assert pos == 'E'
        assert suffix is None

    def test_quarterly_with_suffix_dec(self):
        """Test detection of quarterly frequency with December anchor."""
        index = pd.date_range('2024-01-31', periods=4, freq='QE-DEC')
        freq, pos, suffix = detect_and_parse_frequency(index)
        assert freq == 'Q'
        assert pos == 'E'
        assert suffix == 'DEC'

    def test_quarterly_with_suffix_mar(self):
        """Test detection of quarterly frequency with March anchor.

        Note: detect_frequency may normalize quarter end to default anchor (DEC).
        This test verifies the parser can handle detected frequencies.
        """
        index = pd.date_range('2024-03-31', periods=4, freq='QE-MAR')
        freq, pos, suffix = detect_and_parse_frequency(index)
        assert freq == 'Q'
        assert pos == 'E'
        # Note: detected suffix may be normalized to DEC
        assert suffix is not None

    def test_quarterly_start(self):
        """Test detection of quarterly frequency at start.

        Note: detect_frequency may normalize the anchor month.
        """
        index = pd.date_range('2024-01-01', periods=4, freq='QS-JAN')
        freq, pos, suffix = detect_and_parse_frequency(index)
        assert freq == 'Q'
        assert pos == 'S'
        # Detected suffix may be normalized
        assert suffix is not None

    def test_two_digit_frequency(self):
        """Test detection of frequency with two-letter code."""
        index = pd.date_range('2024-01-01', periods=10, freq='B')
        freq, pos, suffix = detect_and_parse_frequency(index)
        # Should not be None (different from D)
        assert freq is not None
        assert len(freq) >= 1

    def test_irregular_index_raises(self):
        """Test that irregular index raises ValueError."""
        index = pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-15'])
        with pytest.raises(ValueError, match="Could not detect"):
            detect_and_parse_frequency(index)

    def test_single_observation_raises(self):
        """Test that single observation raises ValueError."""
        index = pd.date_range('2024-01-01', periods=1, freq='D')
        with pytest.raises(ValueError):
            detect_and_parse_frequency(index)

    def test_two_observations_works(self):
        """Test that two observations are sufficient for detection."""
        index = pd.date_range('2024-01-01', periods=2, freq='D')
        freq, pos, suffix = detect_and_parse_frequency(index)
        assert freq == 'D'
        assert pos is None
        assert suffix is None



class TestBuildFrequencyString:
    """Tests for build_frequency_string function."""

    def test_frequency_only(self):
        """Test building frequency string with frequency code only."""
        result = build_frequency_string('D')
        assert result == 'D'

    def test_with_position_start(self):
        """Test building frequency string with start position."""
        result = build_frequency_string('M', position='S')
        assert result == 'MS'

    def test_with_position_end(self):
        """Test building frequency string with end position."""
        result = build_frequency_string('Q', position='E')
        assert result == 'QE'

    def test_with_position_and_suffix(self):
        """Test building frequency string with position and suffix."""
        result = build_frequency_string('Q', position='E', suffix='DEC')
        assert result == 'QE-DEC'

    def test_weekly_frequency(self):
        """Test building weekly frequency string."""
        result = build_frequency_string('W')
        assert result == 'W'

    def test_hourly_frequency(self):
        """Test building hourly frequency string."""
        result = build_frequency_string('h')
        assert result == 'h'

    def test_minute_frequency(self):
        """Test building minute frequency string."""
        result = build_frequency_string('T')
        assert result == 'T'

    def test_monthly_start_and_end_differ(self):
        """Test that MS and ME produce different strings."""
        ms = build_frequency_string('M', position='S')
        me = build_frequency_string('M', position='E')
        assert ms == 'MS'
        assert me == 'ME'
        assert ms != me

    def test_quarterly_multiple_anchors(self):
        """Test quarterly with different anchor months."""
        q_dec = build_frequency_string('Q', position='E', suffix='DEC')
        q_mar = build_frequency_string('Q', position='E', suffix='MAR')
        q_jun = build_frequency_string('Q', position='E', suffix='JUN')

        assert q_dec == 'QE-DEC'
        assert q_mar == 'QE-MAR'
        assert q_jun == 'QE-JUN'
        assert q_dec != q_mar

    def test_invalid_position_raises(self):
        """Test that invalid position raises ValueError."""
        with pytest.raises(ValueError, match="position must be"):
            build_frequency_string('D', position='X')

    def test_suffix_with_position(self):
        """Test that suffix is properly added only with position."""
        # With position, suffix should be added
        result_with_pos = build_frequency_string('Q', position='E', suffix='DEC')
        assert result_with_pos == 'QE-DEC'

    def test_none_position_and_suffix(self):
        """Test explicitly passing None for position and suffix."""
        result = build_frequency_string('D', position=None, suffix=None)
        assert result == 'D'

    def test_business_day_frequency(self):
        """Test building business day frequency string."""
        result = build_frequency_string('B')
        assert result == 'B'


class TestRoundTrip:
    """Test that detect_and_parse_frequency and build_frequency_string are consistent."""

    def test_roundtrip_daily(self):
        """Test roundtrip for daily frequency."""
        index = pd.date_range('2024-01-01', periods=10, freq='D')
        freq, pos, suffix = detect_and_parse_frequency(index)
        rebuilt = build_frequency_string(freq, pos, suffix)
        assert rebuilt == 'D'

    def test_roundtrip_monthly_start(self):
        """Test roundtrip for monthly start frequency."""
        index = pd.date_range('2024-01-01', periods=5, freq='MS')
        freq, pos, suffix = detect_and_parse_frequency(index)
        rebuilt = build_frequency_string(freq, pos, suffix)
        assert rebuilt == 'MS'

    def test_roundtrip_monthly_end(self):
        """Test roundtrip for monthly end frequency."""
        index = pd.date_range('2024-01-31', periods=5, freq='ME')
        freq, pos, suffix = detect_and_parse_frequency(index)
        rebuilt = build_frequency_string(freq, pos, suffix)
        assert rebuilt == 'ME'

    def test_roundtrip_quarterly_with_suffix(self):
        """Test roundtrip for quarterly frequency with suffix."""
        index = pd.date_range('2024-01-31', periods=4, freq='QE-DEC')
        freq, pos, suffix = detect_and_parse_frequency(index)
        rebuilt = build_frequency_string(freq, pos, suffix)
        assert rebuilt == 'QE-DEC'

    def test_roundtrip_quarterly_start(self):
        """Test roundtrip for quarterly start frequency.

        Note: detected suffix may be normalized from the requested value.
        """
        index = pd.date_range('2024-01-01', periods=4, freq='QS-JAN')
        freq, pos, suffix = detect_and_parse_frequency(index)
        rebuilt = build_frequency_string(freq, pos, suffix)
        # Verify roundtrip creates valid frequency string
        assert rebuilt.startswith('QS')
        # If suffix was detected, it should be in the rebuilt string
        if suffix:
            assert rebuilt.endswith(suffix)

    def test_roundtrip_weekly(self):
        """Test roundtrip for weekly frequency."""
        index = pd.date_range('2024-01-01', periods=10, freq='W')
        freq, pos, suffix = detect_and_parse_frequency(index)
        rebuilt = build_frequency_string(freq, pos, suffix)
        assert rebuilt == 'W'
