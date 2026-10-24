"""Fixtures of the ``FrequencyAligner`` test package."""

import pytest

# Objet testé
from tsforecast.utils.frequency.aligner import FrequencyAligner


@pytest.fixture
def aligner() -> FrequencyAligner:
    """Fresh FrequencyAligner instance."""
    return FrequencyAligner()
