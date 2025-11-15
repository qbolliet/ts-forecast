"""Test script for panel position conversion."""
import pandas as pd
import numpy as np
from tsforecast.utils.position.converter import PeriodPositionConverter

# Initialisation du converter
converter = PeriodPositionConverter()

print("=" * 80)
print("TEST 1: Panel simple avec fréquence mensuelle identique")
print("=" * 80)

# Création d'un panel simple : 2 entités, 3 dates chacune, fréquence mensuelle
entities = ['A', 'A', 'A', 'B', 'B', 'B']
dates_ms = pd.date_range('2023-01-01', periods=6, freq='MS')  # Month Start
idx = pd.MultiIndex.from_arrays([entities, dates_ms], names=['entity', 'date'])
data = pd.Series(range(6), index=idx, name='value')

print("\nDonnées originales (MS - Month Start):")
print(data)

# Conversion de start à end
result = converter.convert(data, from_unit='start', to_unit='end', freq='M')
print("\nAprès conversion vers end position:")
print(result)
print("\nDates converties:")
print(result.index.get_level_values('date'))

print("\n" + "=" * 80)
print("TEST 2: Panel avec fréquences mixtes")
print("=" * 80)

# Création d'un panel avec fréquences mixtes
# Entité A: mensuelle (3 mois)
# Entité B: annuelle (3 ans)
entities_mixed = ['A', 'A', 'A', 'B', 'B', 'B']
dates_a = pd.date_range('2023-01-01', periods=3, freq='MS')
dates_b = pd.date_range('2023-01-01', periods=3, freq='YS')
dates_mixed = pd.DatetimeIndex(list(dates_a) + list(dates_b))
idx_mixed = pd.MultiIndex.from_arrays([entities_mixed, dates_mixed], names=['entity', 'date'])
data_mixed = pd.Series(range(6), index=idx_mixed, name='value')

print("\nDonnées originales (fréquences mixtes):")
print(data_mixed)
print("\nFréquences par entité:")
print("  Entité A:", pd.infer_freq(dates_a))
print("  Entité B:", pd.infer_freq(dates_b))

# Conversion sans spécifier freq (inférence automatique par groupe)
result_mixed = converter.convert(data_mixed, from_unit='start', to_unit='end')
print("\nAprès conversion vers end position (fréquences inférées):")
print(result_mixed)
print("\nDates converties:")
for entity in ['A', 'B']:
    entity_dates = result_mixed.loc[entity].index
    print(f"  Entité {entity}: {list(entity_dates)}")

print("\n" + "=" * 80)
print("TEST 3: DataFrame avec MultiIndex")
print("=" * 80)

# Test avec DataFrame
df = pd.DataFrame({
    'value1': range(6),
    'value2': range(10, 16)
}, index=idx)

print("\nDataFrame original:")
print(df)

result_df = converter.convert(df, from_unit='S', to_unit='E', freq='M')
print("\nDataFrame après conversion:")
print(result_df)

print("\n" + "=" * 80)
print("TEST 4: Test de validation (doit échouer)")
print("=" * 80)

# Test avec entités non groupées (interleaved)
entities_bad = ['A', 'B', 'A', 'B', 'A', 'B']
idx_bad = pd.MultiIndex.from_arrays([entities_bad, dates_ms], names=['entity', 'date'])
data_bad = pd.Series(range(6), index=idx_bad)

print("\nDonnées avec entités entrelacées:")
print(data_bad)

try:
    result_bad = converter.convert(data_bad, from_unit='start', to_unit='end', freq='M')
    print("ERREUR: La validation aurait dû échouer!")
except ValueError as e:
    print(f"\n[OK] Validation échouée comme prévu: {e}")

print("\n" + "=" * 80)
print("TEST 5: Conversion identique (start -> start)")
print("=" * 80)

result_same = converter.convert(data, from_unit='start', to_unit='start')
print("\nDonnées originales:")
print(data.head(3))
print("\nAprès conversion start -> start (doit être identique):")
print(result_same.head(3))
print(f"\n[OK] Identiques: {data.equals(result_same)}")

print("\n" + "=" * 80)
print("Tous les tests sont terminés!")
print("=" * 80)
