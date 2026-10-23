"""String dtypes of the outputs of ``HighFrequencyImputer``.

The ``frequency`` level of the stacked output is in the ``str`` dtype, with one
stage or several; the provenance matrix is an ``object`` frame whose every
marked cell is a ``ProvenanceType`` member after ``fit`` and after ``transform``
(cells marked ``ORIGINAL`` or ``AGGREGATED`` at the replay included); the
diagnostic frame of the fitted plan keeps its provenance as members.
"""

# Manipulation de données
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Objets testés
from tsforecast.frequency import HighFrequencyImputer, ProvenanceType

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
    """Stage labels in ``str``; provenance cells as enumeration members."""

    @pytest.mark.parametrize('intermediate, stages', [(False, ['M']), (True, ['Q', 'M'])],
                             ids=['one_stage', 'two_stages'])
    def test_frequency_level_is_str(self, intermediate, stages):
        imputer, X = _fitted(intermediate)

        output = imputer.transform(X)

        level = output.index.get_level_values('frequency')
        assert level.dtype == 'str' and level.unique().tolist() == stages

    @pytest.mark.parametrize('intermediate', [False, True], ids=['one_stage', 'two_stages'])
    def test_provenance_cells_are_members(self, intermediate):
        """After ``fit`` and after ``transform``: an ``object`` matrix of ``ProvenanceType`` members."""
        imputer, X = _fitted(intermediate)
        after_fit = imputer.imputation_provenance_.copy()

        imputer.transform(X)
        after_transform = imputer.imputation_provenance_

        for provenance in (after_fit, after_transform):
            assert (provenance.dtypes == object).all()
            assert all(isinstance(value, ProvenanceType) for value in _marked_cells(provenance))
        if intermediate:
            # Les cellules agrégées de l'étape trimestrielle sont marquées au rejeu
            assert ProvenanceType.AGGREGATED in _marked_cells(after_transform)

    def test_diagnostic_frame_provenance_is_object(self):
        imputer, _X = _fitted(True)

        frame = imputer.imputation_plan_.to_diagnostic_frame()

        assert frame['emitted_provenance'].dtype == object
        assert frame['stage'].dtype == 'str' and frame['source_frequency'].dtype == 'str'
        assert all(isinstance(value, ProvenanceType) for value in frame['emitted_provenance'])
