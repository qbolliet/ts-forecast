"""Fixtures of the delay-calculator test package.

Publication-delay frames shared by several test files.
"""

import numpy as np
import pandas as pd
import pytest

# Outils de test partagés du paquet
from tests.unit.delays.calculator.helpers import TS, _detect


# =============================================================================
# Jeux issus de compare_and_detect_delays (contrat entre les deux fonctions)
# =============================================================================
@pytest.fixture
def monthly_publication_delays() -> pd.DataFrame:
    """Monthly delays of a two-country panel, detected by ``compare_and_detect_delays``.

    ``new`` holds ``PIB`` and ``inflation`` from January to June 2023 for France
    and Germany; ``existing`` lacks the last months, so these are the observations
    detected, downloaded on 2023-08-10 and counted from the period start:

    * (France, PIB): April, May, June, i.e. 131, 101 and 70 days
      (April 1st + 131 d = August 10th: 30 + 31 + 30 + 31 = 122 d to August 1st, then 9 d);
    * (France, inflation): May, June: 101 and 70 days;
    * (Germany, PIB): May, June: 101 and 70 days;
    * (Germany, inflation): June: 70 days.
    """
    months = pd.date_range('2023-01-01', periods=6, freq='MS')
    index = pd.MultiIndex.from_product([['France', 'Germany'], months], names=['country', 'date'])
    new = pd.DataFrame({'PIB': np.arange(1.0, 13.0), 'inflation': np.arange(101.0, 113.0)}, index=index)
    existing = new.copy()
    first_new = {('France', 'PIB'): '2023-04-01', ('France', 'inflation'): '2023-05-01',
                 ('Germany', 'PIB'): '2023-05-01', ('Germany', 'inflation'): '2023-06-01'}
    for (country, column), date in first_new.items():
        existing.loc[(country, slice(TS(date), None)), column] = np.nan

    return _detect(new, existing, download_date='2023-08-10', reference_point='start')


@pytest.fixture
def simple_time_series_delays() -> pd.DataFrame:
    """Delays of a plain time series (no panel), from a first download on 2024-01-15.

    ``PIB`` and ``inflation`` are observed up to December 2023 (delay from the period
    start: 45 days, December 1st to January 15th), ``unemployment`` up to November 2023
    (75 days).
    """
    months = pd.date_range('2023-01-01', periods=12, freq='MS')
    data = pd.DataFrame({'PIB': np.arange(12.0), 'inflation': np.arange(12.0) + 2.5, 'unemployment': np.arange(12.0)},
                        index=months)
    data.loc[TS('2023-12-01'), 'unemployment'] = np.nan

    return _detect(data, None, download_date='2024-01-15', reference_point='start')
