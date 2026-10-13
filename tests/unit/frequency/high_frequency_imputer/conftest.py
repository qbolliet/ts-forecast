"""Fixtures of the ``HighFrequencyImputer`` test package.

Two derivations of ``PANEL-F`` (§2.5) used by the tests of entities that never
observe the imputed column (§5.10, D33).
"""
# Modules de base
import numpy as np
import pandas as pd
import pytest


# Jeu PANEL-F dont 'v' est entièrement effacée pour IT
@pytest.fixture
def panel_f_without_it_v(mixed_freq_panel_multifrequency) -> pd.DataFrame:
    """``PANEL-F`` dont ``v`` n'est JAMAIS observée pour ``IT`` (§5.10).

    Les 36 cellules mensuelles de ``v`` d'``IT`` sont effacées : le couple
    ``('IT', 'v')`` n'a plus de fréquence détectable et rejoint les couples
    non détectés. ``FR`` reste annuelle et ``DE`` trimestrielle, de sorte que
    le jeu mutualisé du §5.8 garde ses 15 lignes.
    """
    frame = mixed_freq_panel_multifrequency.copy()
    frame.loc[('IT',), 'v'] = np.nan
    return frame


# Entité totalement muette : ni cible, ni covariable
@pytest.fixture
def panel_f_with_a_mute_entity(panel_f_without_it_v) -> pd.DataFrame:
    """``PANEL-F`` dont ``IT`` n'a plus de covariable exploitable pour ``v``.

    ``q1`` est effacée à son tour : ``IT`` ne garde que ``m1``, ce qui la
    laisse dans le jeu à la fréquence cible — une entité dont plus aucune
    colonne mensuelle n'est observée est écartée bien plus haut, par la
    validation de la fréquence cible — mais prive la grille de prédiction de
    ``v`` d'une de ses deux covariables. C'est le seul chemin d'échec du
    §5.10 : la ligne est vide, et l'interpolation ne peut pas servir de repli
    — il n'y a rien à interpoler.
    """
    frame = panel_f_without_it_v.copy()
    frame.loc[('IT',), 'q1'] = np.nan
    return frame
