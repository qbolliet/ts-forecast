"""Public components accept inputs whose values sit on read-only NumPy buffers.

Every float DataFrame or Series of the inputs of ``cases.CASES`` is rebuilt on
a ``writeable=False`` array (``as_read_only``), the way pandas exposes shared
data under Copy-on-Write: a component writing in place into the values it
receives, or into an array obtained by ``to_numpy()`` / ``.values`` without a
copy, fails with ``assignment destination is read-only``. Each case must run
and give the outputs it gives on writable inputs.
"""
import pytest

from tests.integration.input_contract.cases import CASES, as_read_only, assert_same_outputs, guarded_run


def _params():
    """One parameter per case, the slow ones marked ``slow``."""
    return [
        pytest.param(name, marks=[pytest.mark.slow] if case.slow else [], id=name)
        for name, case in CASES.items()
    ]


class TestReadOnlyInputs:
    """Read-only inputs give the same outputs as writable ones."""

    @pytest.mark.parametrize('name', _params())
    def test_same_outputs_as_on_writable_inputs(self, name):
        """The case runs on read-only inputs and its outputs equal those of writable inputs."""
        case = CASES[name]
        expected, _ = guarded_run(case, case.build())
        inputs = {key: as_read_only(value) for key, value in case.build().items()}

        outputs, _ = guarded_run(case, inputs)

        assert_same_outputs(outputs, expected)
