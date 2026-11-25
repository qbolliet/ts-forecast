"""Tests of the parallel fitting of ``PanelwiseTransformer`` (``n_jobs``).

Covers the equality of ``n_jobs=1`` and ``n_jobs=2`` (or ``-1``): same output, same
row order, same fitted entities in the same order, on a ``MultiIndex`` panel of two
and three levels, on a column panel and on reversed entities; and the error handling
of the parallel fit.
"""

import pandas as pd
import pytest

from tsforecast.panel.transformers import PanelwiseTransformer
from tests.support.perturbations import reverse_entities, to_three_level_index

from .conftest import FailingTransformer, centring

# Démarrage des processus joblib (plusieurs secondes sous Windows) : module lent
pytestmark = pytest.mark.slow

def _fit_transform(panel: pd.DataFrame, n_jobs: int, **kwargs) -> tuple:
    """Fit then transform ``panel`` with ``n_jobs`` workers; return the output and the transformer."""
    kwargs.setdefault('time_col', None)
    transformer = PanelwiseTransformer(centring(), n_jobs=n_jobs, **kwargs).fit(panel)
    return transformer.transform(panel), transformer


class TestParallelEqualsSequential:
    """``n_jobs`` changes how entities are fitted, never the result."""

    @pytest.mark.parametrize('n_jobs', [2, -1], ids=['two-workers', 'all-processors'])
    def test_same_output(self, index_panel, centred_index_panel, n_jobs):
        """The parallel output is the golden centred panel, rows in the input order."""
        output, _ = _fit_transform(index_panel, n_jobs)
        pd.testing.assert_frame_equal(output, centred_index_panel)

    @pytest.mark.parametrize(
        'perturb', [to_three_level_index, reverse_entities], ids=['three-levels', 'reversed-entities']
    )
    def test_same_output_on_perturbed_panels(self, index_panel, perturb):
        """Three levels or entities in another order: one and two workers agree."""
        panel = perturb(index_panel)
        pd.testing.assert_frame_equal(_fit_transform(panel, 2)[0], _fit_transform(panel, 1)[0])

    def test_same_output_on_a_column_panel(self, column_panel, centred_column_panel):
        """Entities in columns are fitted in parallel as well."""
        output, _ = _fit_transform(column_panel, 2, time_col='date', panel_cols=['country'])
        pd.testing.assert_frame_equal(output, centred_column_panel)

    def test_same_fitted_entities_in_the_same_order(self, index_panel):
        """The fitted transformers are listed in the order of the data, with the same means."""
        _, sequential = _fit_transform(index_panel, 1)
        _, parallel = _fit_transform(index_panel, 2)

        assert list(parallel.transformers_) == list(sequential.transformers_) == [('FR',), ('DE',)]
        assert [t.mean_.tolist() for t in parallel.transformers_.values()] == [[3.0, 4.0], [30.0, 3.0]]

    def test_parallel_fit_records_no_failure(self, index_panel):
        """Without any failure, ``failed_entities_`` is empty after a parallel fit."""
        assert _fit_transform(index_panel, 2)[1].failed_entities_ == []


class TestParallelErrorHandling:
    """The parallel fit follows ``error_handling`` like the sequential one."""

    def test_raise_propagates_the_error(self, index_panel):
        """With ``'raise'``, the entity error propagates."""
        transformer = PanelwiseTransformer(
            FailingTransformer(), time_col=None, n_jobs=2, entity_kwargs={('DE',): {'fail_on': 'fit'}}
        )
        with pytest.raises(RuntimeError, match='deliberate failure'):
            transformer.fit(index_panel)

    def test_warn_skips_the_failing_entity(self, index_panel):
        """With ``'warn'``, the failing entity is skipped and recorded, as with one worker."""
        transformer = PanelwiseTransformer(
            FailingTransformer(), time_col=None, n_jobs=2, error_handling='warn',
            entity_kwargs={('DE',): {'fail_on': 'fit'}},
        )
        with pytest.warns(UserWarning, match=r"Error fitting entity \('DE',\): RuntimeError: deliberate failure"):
            transformer.fit(index_panel)

        assert list(transformer.transformers_) == [('FR',)]
        assert transformer.failed_entities_ == [('DE',)]

    def test_ignore_matches_the_sequential_fit(self, index_panel):
        """With ``'ignore'``, one and two workers fit and record the same entities."""
        def fitted(n_jobs):
            return PanelwiseTransformer(
                FailingTransformer(), time_col=None, n_jobs=n_jobs, error_handling='ignore',
                entity_kwargs={('DE',): {'fail_on': 'fit'}},
            ).fit(index_panel)

        sequential, parallel = fitted(1), fitted(2)

        assert list(parallel.transformers_) == list(sequential.transformers_) == [('FR',)]
        assert parallel.failed_entities_ == sequential.failed_entities_ == [('DE',)]
