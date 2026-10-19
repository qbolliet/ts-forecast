"""Fixtures of the ``HighFrequencyImputer`` test package.

Two derivations of ``PANEL-F`` used by the tests of entities that never
observe the imputed column.
"""
# Modules de base
import numpy as np
import pandas as pd
import pytest


# Jeu PANEL-F dont 'v' est entièrement effacée pour IT
@pytest.fixture
def panel_f_without_it_v(mixed_freq_panel_multifrequency) -> pd.DataFrame:
    """``PANEL-F`` whose ``v`` is NEVER observed for ``IT``.

    The 36 monthly cells of ``v`` of ``IT`` are erased: the pair
    ``('IT', 'v')`` has no detectable frequency anymore and joins the
    undetected pairs. ``FR`` stays annual and ``DE`` quarterly, so that
    the pooled set keeps its 15 rows.
    """
    frame = mixed_freq_panel_multifrequency.copy()
    frame.loc[('IT',), 'v'] = np.nan
    return frame


# Entité totalement muette : ni cible, ni covariable
@pytest.fixture
def panel_f_with_a_mute_entity(panel_f_without_it_v) -> pd.DataFrame:
    """``PANEL-F`` whose ``IT`` has no usable covariate left for ``v``.

    ``q1`` is erased in turn: ``IT`` keeps only ``m1``, which leaves it in the
    set at the target frequency — an entity none of whose monthly columns is
    observed is dropped much earlier, by the validation of the target
    frequency — but deprives the prediction grid of ``v`` of one of its two
    covariates. This is the only failure path: the row is empty, and
    interpolation cannot serve as a fallback — there is nothing to
    interpolate.
    """
    frame = panel_f_without_it_v.copy()
    frame.loc[('IT',), 'q1'] = np.nan
    return frame
