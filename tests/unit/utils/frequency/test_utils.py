"""Tests for ``tsforecast.utils.frequency.utils`` (partial, completed by prompt U6).

Covers the behaviours changed while fixing ``utils/position`` (prompt U3):
``normalize_frequency(return_format='full')`` keeps a leading multiplier
(``'2MS'``), and ``detect_index_frequency`` returns a canonical quarterly
anchor (``'QS-JAN'`` rather than the equivalent ``'QS-OCT'`` reported by
``pd.infer_freq``), through ``canonicalize_frequency``.
"""
from __future__ import annotations

import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset

from tsforecast.utils.frequency.utils import (
    canonicalize_frequency,
    detect_index_frequency,
    normalize_frequency,
)


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
