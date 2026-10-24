"""Standalone building blocks of ``HighFrequencyImputer``.

Each module (provenance, imputation plan, window calculator, covariate materializer,
aggregation constraint, stage scaler, variable orderer, training-set builder, target-frequency
validator) has its own API and is tested in isolation; they are imported by module path.
"""
