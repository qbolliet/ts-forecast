"""Public components never modify the objects they receive.

For every case of ``cases.CASES`` — validation, detection, conversion and
regularity functions, delay detection and application, ``ShiftTransformer`` /
``MaskTransformer`` / ``PublicationDelayTransformer``, ``PanelwiseTransformer``,
``HighFrequencyImputer`` (with and without ``y``), ``DropNaNTarget``,
``XYPipeline`` and the eight splitters —, every DataFrame, Series or array
handed to the package (inputs of ``fit`` / ``transform`` / ``split``, and the
output passed back to ``inverse_transform``) is deep-copied when it is handed
over, then compared with its copy once the case has run: values, dtypes, index,
columns and index frequency are unchanged.
"""
import pytest

from tests.integration.input_contract.cases import CASES, assert_unchanged, guarded_run


def _params():
    """One parameter per case, the slow ones marked ``slow``."""
    return [
        pytest.param(name, marks=[pytest.mark.slow] if case.slow else [], id=name)
        for name, case in CASES.items()
    ]


class TestInputsAreNotModified:
    """Nothing the package receives is modified in place."""

    @pytest.mark.parametrize('name', _params())
    def test_every_handed_object_is_unchanged(self, name):
        """Each object handed to the package equals its snapshot once the case has run."""
        case = CASES[name]

        _, registered = guarded_run(case, case.build())

        assert registered, 'the case hands no object to the package'
        for label, obj, frozen in registered:
            assert_unchanged(obj, frozen, f'{name}: {label}')
