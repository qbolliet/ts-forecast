"""Tests unitaires du FrequencyConverter avec des fréquences multipliées ('2MS', '3QS-FEB').

La décomposition de ``target_freq`` passe par ``ParsedFrequency`` : le
multiplicateur est conservé jusqu'à ``resample`` / ``date_range``, il entre dans
le sens de la conversion (sur-échantillonnage ou agrégation) et dans les
facteurs de conversion. Les opérations qui comptent des périodes de base
(``full_periods_only``, ``method='all'``, ``anchor_fraction``) rejettent
explicitement une fréquence multipliée.
"""
import pandas as pd
import pytest

from tsforecast.utils.frequency.converter import FrequencyConverter, _modernize_resample_freq


@pytest.fixture
def converter():
    """A fresh ``FrequencyConverter``."""
    return FrequencyConverter()


@pytest.fixture
def daily():
    """120 days from 2024-01-01, values 0..119."""
    return pd.Series(
        range(120), index=pd.date_range("2024-01-01", periods=120, freq="D"), dtype=float
    )


@pytest.fixture
def monthly():
    """Six month starts, values 0..5."""
    return pd.Series(
        range(6), index=pd.date_range("2024-01-01", periods=6, freq="MS"), dtype=float
    )


@pytest.fixture
def bimonthly():
    """Three bimonthly starts (Jan, Mar, May), values 10, 20, 30."""
    return pd.Series([10.0, 20.0, 30.0], index=pd.date_range("2024-01-01", periods=3, freq="2MS"))


class TestDownsamplingToMultipliedFrequency:
    """The multiplier reaches ``resample``: each bin spans several base periods."""

    def test_daily_to_bimonthly_mean(self, converter, daily):
        """Bins of two months, labelled at the start of the bin."""
        result = converter.convert_frequency(daily, "2MS", method="mean")
        assert list(result.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")]
        # Janvier + février 2024 : jours 0 à 59
        assert result.iloc[0] == pytest.approx(29.5)

    def test_monthly_to_bimonthly_sum(self, converter, monthly):
        """Consecutive months are summed by pairs."""
        result = converter.convert_frequency(monthly, "2MS", method="sum")
        assert result.tolist() == [1.0, 5.0, 9.0]

    def test_bimonthly_to_quarterly(self, converter, bimonthly):
        """A multiplied source is aggregated to a coarser frequency."""
        result = converter.convert_frequency(bimonthly, "QS", method="mean")
        assert list(result.index) == list(pd.date_range("2024-01-01", periods=2, freq="QS"))

    def test_same_multiplied_frequency_is_returned_unchanged(self, converter, bimonthly):
        """Source and target identical, multiplier included: nothing to convert."""
        result = converter.convert_frequency(bimonthly, "2MS")
        assert list(result.index) == list(bimonthly.index)
        assert result.tolist() == bimonthly.tolist()

    def test_multiplier_makes_the_frequencies_differ(self, converter, monthly):
        """'MS' -> '2MS' is a conversion, not a no-op."""
        result = converter.convert_frequency(monthly, "2MS", method="sum")
        assert len(result) == 3


class TestUpsamplingFromMultipliedFrequency:
    """The direction of the conversion accounts for the multiplier."""

    def test_bimonthly_to_monthly_is_an_upsampling(self, converter, bimonthly):
        """'2MS' -> 'MS' interpolates between the bimonthly points."""
        result = converter.convert_frequency(bimonthly, "MS", method="linear")
        assert list(result.index[:5]) == list(pd.date_range("2024-01-01", periods=5, freq="MS"))
        assert result.iloc[:5].tolist() == [10.0, 15.0, 20.0, 25.0, 30.0]

    def test_bimonthly_to_monthly_covers_the_last_block(self, converter, bimonthly):
        """The index is extended to the end of the last bimonthly block (June)."""
        result = converter.convert_frequency(bimonthly, "MS", method="linear")
        assert result.index[-1] == pd.Timestamp("2024-06-01")

    def test_bimonthly_end_position_extends_backwards(self, converter):
        """A '2ME' stamp closes its block: the range starts one month earlier."""
        series = pd.Series([10.0, 20.0, 30.0], index=pd.date_range("2024-02-29", periods=3, freq="2ME"))
        result = converter.convert_frequency(series, "ME", method="linear")
        assert result.index[0] == pd.Timestamp("2024-01-31")
        assert result.index[-1] == pd.Timestamp("2024-06-30")

    def test_quarterly_to_bimonthly_uses_the_target_grid(self, converter):
        """A multiplied target is the ``date_range`` frequency of the extended index."""
        quarterly = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, freq="QS"))
        result = converter.convert_frequency(quarterly, "2MS", method="linear")
        assert list(result.index) == list(pd.date_range("2024-01-01", periods=3, freq="2MS"))


class TestDataFrameWithMultipliedFrequencies:
    """Column-wise conversions decompose each target through ``ParsedFrequency``."""

    def test_string_target(self, converter, daily):
        """One multiplied target for every column."""
        result = converter.convert_frequency(pd.DataFrame({"a": daily, "b": daily * 2}), "2MS", method="mean")
        assert list(result.index) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")]
        assert result["b"].iloc[0] == pytest.approx(59.0)

    def test_dict_target_with_mixed_multipliers(self, converter, daily):
        """Each column gets its own multiplier."""
        result = converter.convert_frequency(
            pd.DataFrame({"a": daily, "b": daily}), {"a": "2MS", "b": "MS"}, method="mean"
        )
        assert result["a"].nunique() == 2
        assert result["b"].nunique() == 4

    def test_explicit_target_position_overrides_the_position_of_the_target(self, converter, daily):
        """``target_position`` wins, the multiplier is kept."""
        result = converter.convert_frequency(daily, "2MS", method="mean", target_position="E")
        assert result.index.freqstr == "2ME"


class TestConversionFactors:
    """``get_conversion_factor`` compares the durations of the multiplied periods."""

    @pytest.mark.parametrize(
        "from_unit, to_unit, expected",
        [
            pytest.param("2MS", "QS", 1.5, id="two-months-in-quarter"),
            pytest.param("D", "2MS", 60.0, id="days-in-two-months"),
            pytest.param("MS", "2MS", 2.0, id="months-in-two-months"),
            pytest.param("monthly", "quarterly", 3.0, id="unchanged-without-multiplier"),
        ],
    )
    def test_factor(self, converter, from_unit, to_unit, expected):
        """How many ``from_unit`` periods fit in one ``to_unit`` period."""
        assert converter.get_conversion_factor(from_unit, to_unit) == pytest.approx(expected)

    @pytest.mark.internal
    def test_default_interpolation_limit_follows_the_multiplier(self, converter):
        """Target periods per source period: 3 months in a quarter, 1.5 bimonthly ones."""
        assert converter._resolve_interpolation_limit("default", "QS", "MS") == 3
        assert converter._resolve_interpolation_limit("default", "QS", "2MS") == 2


class TestOperationsRejectingMultipliedFrequencies:
    """Operations counting base periods fail explicitly rather than counting wrong."""

    def test_full_periods_only(self, converter, daily):
        """The expected number of days per bin is not defined for a two-month bin."""
        with pytest.raises(NotImplementedError, match="2MS"):
            converter.convert_frequency(daily, "2MS", method="sum", full_periods_only=True)

    def test_boolean_all(self, converter, daily):
        """``method='all'`` relies on the same coverage check."""
        with pytest.raises(NotImplementedError, match="2MS"):
            converter.convert_frequency(daily > 3, "2MS", method="all")

    def test_multiplied_source_in_coverage_check(self, converter, bimonthly):
        """A multiplied source frequency is rejected as well."""
        with pytest.raises(NotImplementedError, match="2MS"):
            converter.aggregate_to_lower_frequency(
                bimonthly, "QS", method="sum", full_periods_only=True, source_freq="2MS"
            )

    def test_anchor_fraction_with_multiplied_source(self, converter, bimonthly):
        """Each stamp would stand for a block of two months, not a month."""
        with pytest.raises(NotImplementedError, match="anchor_fraction"):
            converter.interpolate_to_higher_frequency(bimonthly, "MS", anchor_fraction=0.5)

    def test_anchor_fraction_with_multiplied_target_is_supported(self, converter):
        """Only the source periods matter for the anchor."""
        quarterly = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, freq="QS"))
        result = converter.interpolate_to_higher_frequency(quarterly, "2MS", anchor_fraction=0.5)
        assert list(result.index) == list(pd.date_range("2024-01-01", periods=3, freq="2MS"))

    def test_count_subperiods(self, converter):
        """The public counting helper rejects multiplied frequencies too."""
        index = pd.date_range("2024-01-31", periods=2, freq="ME")
        with pytest.raises(NotImplementedError, match="2D"):
            converter.count_subperiods_per_period(index, "M", "2D")

    def test_not_a_value_error(self, converter, daily):
        """``full_periods_only`` swallows ``ValueError``: the rejection must not be one."""
        assert not issubclass(NotImplementedError, ValueError)


class TestUnsupportedMultipliedTargets:
    """A multiplier does not make an unknown frequency valid."""

    def test_unknown_base_after_multiplier(self, converter, daily):
        """Unsupported base frequency."""
        with pytest.raises(ValueError, match="Invalid target frequency"):
            converter.convert_frequency(daily, "2foo")


class TestModernizeResampleFreq:
    """Deprecated bare aliases are modernised, multiplier and anchor kept."""

    @pytest.mark.internal
    @pytest.mark.parametrize(
        "freq, expected",
        [
            pytest.param("Q", "QE", id="bare-quarter"),
            pytest.param("M", "ME", id="bare-month"),
            pytest.param("Y", "YE", id="bare-year"),
            pytest.param("A", "YE", id="bare-annual-alias"),
            pytest.param("2M", "2ME", id="multiplied"),
            pytest.param("3Q-NOV", "3QE-NOV", id="multiplied-anchored"),
            pytest.param("1M", "ME", id="explicit-one-omitted"),
            # Position explicite ou fréquence non concernée : inchangées
            pytest.param("QS", "QS", id="start-position"),
            pytest.param("2ME", "2ME", id="already-modern"),
            pytest.param("D", "D", id="daily"),
            pytest.param("W-MON", "W-MON", id="weekly-anchor"),
            pytest.param("15min", "15min", id="subdaily"),
            pytest.param("", "", id="empty-string"),
            pytest.param("not a frequency", "not a frequency", id="unparsable"),
        ],
    )
    def test_modernize(self, freq, expected):
        """Only bare period-end aliases are rewritten."""
        assert _modernize_resample_freq(freq) == expected
