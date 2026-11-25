"""Independence of the delay transformers from the resolution of the input dates.

``ShiftTransformer``, ``MaskTransformer``, ``PublicationDelayTransformer`` (shift
and mask strategies) and the per-entity factory run through
``PanelwiseTransformer``, on monthly data whose dates are stored in seconds,
milliseconds, microseconds or nanoseconds (``with_datetime_resolution``):
``transform`` and ``inverse_transform`` give the values and dates of the
nanosecond input, and the dates keep the input resolution.
"""
# Modules de base
import warnings

import pandas as pd
import pytest

# Jeux de données et perturbations partagés
from tests.support.datasets import build_mixed_frequency_panel, build_mixed_frequency_timeseries
from tests.support.perturbations import DATETIME_UNITS, datetime_units, with_datetime_resolution

# Transformateurs à tester et collaborateur réel pour les panels
from tsforecast.delays.transformers import (
    MaskTransformer,
    PublicationDelayTransformer,
    ShiftTransformer,
    create_delay_transformer_factory,
)
from tsforecast.panel import PanelwiseTransformer

# Délais en jours depuis le début de la période (valeurs du pipeline d'intégration)
DELAYS = {'inflation_ipc': 20.0, 'taux_chomage': 50.0, 'pib_trimestriel': 75.0}

# Constructeurs des transformateurs comparés, un par cas
TRANSFORMERS = {
    'shift': lambda: ShiftTransformer(n_periods=2, frequency='M'),
    'shift-backward': lambda: ShiftTransformer(n_periods=-1, frequency='Q'),
    'mask-last': lambda: MaskTransformer(n_obs=1, mask_frequency='Q', how='last'),
    'mask-first': lambda: MaskTransformer(n_obs=2, mask_frequency='Y', how='first'),
    'publication-shift': lambda: PublicationDelayTransformer(
        DELAYS, prediction_date='2024-08-15', strategy='shift', delay_unit='D',
        reference_point='start', handle_missing_delays='ignore',
    ),
    'publication-mask': lambda: PublicationDelayTransformer(
        DELAYS, prediction_date='2024-08-15', strategy='mask', target_frequency='Q', delay_unit='D',
        reference_point='start', handle_missing_delays='ignore',
    ),
}


def _round_trip(make, data: pd.DataFrame) -> tuple:
    """Fit, transform and invert a fresh transformer on ``data``, warnings silenced."""
    transformer = make()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        transformed = transformer.fit_transform(data)
        inverse = transformer.inverse_transform(transformed)
    return transformed, inverse


class TestDatetimeResolution:
    """Transformed and inverted frames do not depend on how the dates are stored."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('name', list(TRANSFORMERS), ids=list(TRANSFORMERS))
    def test_series(self, name, unit):
        """Monthly series: same values and dates as in nanoseconds, dates at the input resolution."""
        data = build_mixed_frequency_timeseries()
        expected, expected_inverse = _round_trip(TRANSFORMERS[name], with_datetime_resolution(data, 'ns'))

        transformed, inverse = _round_trip(TRANSFORMERS[name], with_datetime_resolution(data, unit))

        pd.testing.assert_frame_equal(transformed, expected, check_index_type=False)
        pd.testing.assert_frame_equal(inverse, expected_inverse, check_index_type=False)
        assert set(datetime_units(transformed).values()) == {unit}
        assert set(datetime_units(inverse).values()) == {unit}

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('strategy', ['shift', 'mask'])
    def test_panel_through_the_factory(self, strategy, unit):
        """Per-entity delays applied by ``PanelwiseTransformer``: same panel as in nanoseconds."""
        panel = build_mixed_frequency_panel()
        # Délais propres à chaque entité, table indexée par (pays, variable)
        delays = pd.DataFrame(
            {'delay': [20.0, 50.0, 30.0, 45.0], 'unit': 'D', 'reference_point': 'start', 'frequency': 'Q'},
            index=pd.MultiIndex.from_tuples(
                [('France', 'inflation_ipc'), ('France', 'taux_chomage'),
                 ('Allemagne', 'inflation_ipc'), ('Italie', 'taux_chomage')],
                names=['country', 'column'],
            ),
        )

        def run(data):
            factory = create_delay_transformer_factory(delays, strategy=strategy, prediction_date='2024-08-15')
            transformer = PanelwiseTransformer(transformer=factory, time_col=None, panel_cols=None)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return transformer.fit_transform(data)

        expected = run(with_datetime_resolution(panel, 'ns'))

        result = run(with_datetime_resolution(panel, unit))

        pd.testing.assert_frame_equal(result, expected, check_index_type=False)
        assert set(datetime_units(result).values()) == {unit}
