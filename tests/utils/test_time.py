"""Tests unitaires pour les fonctions time_utils étendues."""
# Importation des modules
# Modules de base
import pandas as pd
# Module de gestion des dates
from datetime import datetime, timedelta
# Module de test
import pytest

# Fonctions à tester d'extraction des limites de périodes
from tsforecast.utils.time import get_period_start, get_period_end, get_period_boundaries

class TestPeriodFunctions:
    """Test suite for period start and end functions."""

    def test_intraday_frequencies(self):
        """Test intraday frequency handling (hourly, minute, second)."""
        test_date = datetime(2023, 6, 15, 14, 35, 47, 123456)
        
        # Test horaire
        assert get_period_start(test_date, 'h') == datetime(2023, 6, 15, 14, 0, 0)
        assert get_period_end(test_date, 'h') == datetime(2023, 6, 15, 15, 0, 0)
        
        # Test minute
        assert get_period_start(test_date, 'min') == datetime(2023, 6, 15, 14, 35, 0)
        assert get_period_end(test_date, 'min') == datetime(2023, 6, 15, 14, 36, 0)
        
        # Test seconde
        assert get_period_start(test_date, 's') == datetime(2023, 6, 15, 14, 35, 47)
        assert get_period_end(test_date, 's') == datetime(2023, 6, 15, 14, 35, 48)
    
    def test_subsecond_frequencies(self):
        """Test sub-second frequency handling (ms, us)."""
        test_date = datetime(2023, 6, 15, 14, 35, 47, 123456)
        
        # Test milliseconde (123.456 ms)
        start_ms = get_period_start(test_date, 'ms')
        assert start_ms.microsecond == 123000  # Arrondi à la milliseconde
        
        # Test microseconde
        start_us = get_period_start(test_date, 'us')
        assert start_us.microsecond == 123456
        
    def test_daily_frequencies(self):
        """Test daily and business daily frequencies."""
        test_date = datetime(2023, 6, 15, 14, 30, 0)
        
        # Test jour calendaire
        assert get_period_start(test_date, 'D') == datetime(2023, 6, 15, 0, 0, 0)
        assert get_period_end(test_date, 'D') == datetime(2023, 6, 16, 0, 0, 0)
        
        # Test jour ouvré (même logique)
        assert get_period_start(test_date, 'B') == datetime(2023, 6, 15, 0, 0, 0)
        assert get_period_end(test_date, 'B') == datetime(2023, 6, 16, 0, 0, 0)
    
    def test_weekly_frequency(self):
        """Test weekly frequency (Monday-based)."""
        # Jeudi 15 juin 2023
        test_date = datetime(2023, 6, 15)
        
        # La semaine commence le lundi (12 juin)
        assert get_period_start(test_date, 'W') == datetime(2023, 6, 12, 0, 0, 0)
        # La semaine se termine le lundi suivant (19 juin)
        assert get_period_end(test_date, 'W') == datetime(2023, 6, 19, 0, 0, 0)
    
    def test_semi_monthly_frequency(self):
        """Test semi-monthly frequency (1-15, 16-end)."""
        # Première quinzaine
        date1 = datetime(2023, 6, 10)
        assert get_period_start(date1, 'SM') == datetime(2023, 6, 1, 0, 0, 0)
        assert get_period_end(date1, 'SM') == datetime(2023, 6, 16, 0, 0, 0)
        
        # Deuxième quinzaine
        date2 = datetime(2023, 6, 20)
        assert get_period_start(date2, 'SM') == datetime(2023, 6, 16, 0, 0, 0)
        assert get_period_end(date2, 'SM') == datetime(2023, 7, 1, 0, 0, 0)
        
        # Cas limite : le 15 est dans la première quinzaine
        date3 = datetime(2023, 6, 15)
        assert get_period_start(date3, 'SM') == datetime(2023, 6, 1, 0, 0, 0)
        
        # Cas limite : le 16 est dans la deuxième quinzaine
        date4 = datetime(2023, 6, 16)
        assert get_period_start(date4, 'SM') == datetime(2023, 6, 16, 0, 0, 0)
    
    def test_monthly_frequency(self):
        """Test monthly frequency."""
        test_date = datetime(2023, 6, 15)
        
        assert get_period_start(test_date, 'M') == datetime(2023, 6, 1, 0, 0, 0)
        assert get_period_end(test_date, 'M') == datetime(2023, 7, 1, 0, 0, 0)
        
        # Test décembre -> janvier
        dec_date = datetime(2023, 12, 15)
        assert get_period_end(dec_date, 'M') == datetime(2024, 1, 1, 0, 0, 0)
    
    def test_quarterly_frequency(self):
        """Test quarterly frequency."""
        # Q1 : janvier-mars
        q1_date = datetime(2023, 2, 15)
        assert get_period_start(q1_date, 'Q') == datetime(2023, 1, 1, 0, 0, 0)
        assert get_period_end(q1_date, 'Q') == datetime(2023, 4, 1, 0, 0, 0)
        
        # Q2 : avril-juin
        q2_date = datetime(2023, 5, 15)
        assert get_period_start(q2_date, 'Q') == datetime(2023, 4, 1, 0, 0, 0)
        assert get_period_end(q2_date, 'Q') == datetime(2023, 7, 1, 0, 0, 0)
        
        # Q3 : juillet-septembre
        q3_date = datetime(2023, 8, 15)
        assert get_period_start(q3_date, 'Q') == datetime(2023, 7, 1, 0, 0, 0)
        assert get_period_end(q3_date, 'Q') == datetime(2023, 10, 1, 0, 0, 0)
        
        # Q4 : octobre-décembre
        q4_date = datetime(2023, 11, 15)
        assert get_period_start(q4_date, 'Q') == datetime(2023, 10, 1, 0, 0, 0)
        assert get_period_end(q4_date, 'Q') == datetime(2024, 1, 1, 0, 0, 0)
    
    def test_annual_frequency(self):
        """Test annual frequency."""
        test_date = datetime(2023, 6, 15)
        
        assert get_period_start(test_date, 'Y') == datetime(2023, 1, 1, 0, 0, 0)
        assert get_period_end(test_date, 'Y') == datetime(2024, 1, 1, 0, 0, 0)
    
    def test_literal_frequency_names(self):
        """Test using literal frequency names instead of pandas codes."""
        test_date = datetime(2023, 6, 15)
        
        # Test avec noms littéraux
        assert get_period_start(test_date, 'monthly') == datetime(2023, 6, 1, 0, 0, 0)
        assert get_period_start(test_date, 'quarterly') == datetime(2023, 4, 1, 0, 0, 0)
        assert get_period_start(test_date, 'annual') == datetime(2023, 1, 1, 0, 0, 0)
    
    def test_pandas_timestamp_input(self):
        """Test using pandas Timestamp as input."""
        test_date = pd.Timestamp('2023-06-15 14:30:00')
        
        assert get_period_start(test_date, 'M') == datetime(2023, 6, 1, 0, 0, 0)
        assert get_period_end(test_date, 'M') == datetime(2023, 7, 1, 0, 0, 0)
    
    def test_get_period_boundaries(self):
        """Test combined period range function."""
        test_date = datetime(2023, 6, 15)
        
        start, end = get_period_boundaries(test_date, 'M')
        assert start == datetime(2023, 6, 1, 0, 0, 0)
        assert end == datetime(2023, 7, 1, 0, 0, 0)
    
    def test_unsupported_frequency_error(self):
        """Test error handling for unsupported frequencies."""
        test_date = datetime(2023, 6, 15)
        
        with pytest.raises(ValueError, match="Unsupported frequency"):
            get_period_start(test_date, 'invalid_freq')
        
        with pytest.raises(ValueError, match="Unsupported frequency"):
            get_period_end(test_date, 'invalid_freq')
