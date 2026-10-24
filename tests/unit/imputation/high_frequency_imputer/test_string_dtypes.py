"""String dtypes of the outputs of ``HighFrequencyImputer``.

The ``frequency`` level of the stacked output is in the ``str`` dtype, with one
stage or several; the provenance matrix has the categorical dtype
``PROVENANCE_DTYPE`` after ``fit`` and after ``transform`` (cells marked
``ORIGINAL`` or ``AGGREGATED`` at the replay included), every marked cell being
a ``ProvenanceType`` value; the diagnostic frame of the fitted plan carries its
provenance in the same dtype.
"""

# Manipulation de données
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Objets testés
from tsforecast.frequency import PROVENANCE_DTYPE, HighFrequencyImputer, ProvenanceType

# Jeux de données partagés
from tests.support.datasets import build_panel_reference


def _fitted(intermediate: bool) -> tuple:
    """Fit an imputer on the reference panel; return it with its input."""
    X = build_panel_reference()[['m1', 'q1', 'v']]
    imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                   impute_intermediate_frequencies=intermediate)
    imputer.fit(X)
    return imputer, X


def _marked_cells(provenance: pd.DataFrame) -> list:
    """Return the marked (non-missing) cells of a provenance matrix."""
    return [value for value in provenance.to_numpy().ravel() if not pd.isna(value)]


class TestStringDtypes:
    """Stage labels in ``str``; provenance in the categorical ``PROVENANCE_DTYPE``."""

    @pytest.mark.parametrize('intermediate, stages', [(False, ['M']), (True, ['Q', 'M'])],
                             ids=['one_stage', 'two_stages'])
    def test_frequency_level_is_str(self, intermediate, stages):
        imputer, X = _fitted(intermediate)

        output = imputer.transform(X)

        level = output.index.get_level_values('frequency')
        assert level.dtype == 'str' and level.unique().tolist() == stages

    @pytest.mark.parametrize('intermediate', [False, True], ids=['one_stage', 'two_stages'])
    def test_provenance_matrix_is_categorical(self, intermediate):
        """After ``fit`` and after ``transform``: a ``PROVENANCE_DTYPE`` matrix of ``ProvenanceType`` values."""
        imputer, X = _fitted(intermediate)
        after_fit = imputer.imputation_provenance_.copy()

        imputer.transform(X)
        after_transform = imputer.imputation_provenance_

        for provenance in (after_fit, after_transform):
            assert (provenance.dtypes == PROVENANCE_DTYPE).all()
            assert set(_marked_cells(provenance)) <= {member.value for member in ProvenanceType}
        if intermediate:
            # Les cellules agrégées de l'étape trimestrielle sont marquées au rejeu
            assert ProvenanceType.AGGREGATED in _marked_cells(after_transform)

    def test_diagnostic_frame_provenance_is_categorical(self):
        imputer, _X = _fitted(True)

        frame = imputer.imputation_plan_.to_diagnostic_frame()

        assert frame['emitted_provenance'].dtype == PROVENANCE_DTYPE
        assert frame['stage'].dtype == 'str' and frame['source_frequency'].dtype == 'str'
        assert frame['emitted_provenance'].tolist() == [
            step.emitted_provenance for step in imputer.imputation_plan_.steps
        ]
