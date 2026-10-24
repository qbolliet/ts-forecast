"""Shared builders of the ``TrainingSetBuilder`` tests (not a test module).

Detected frequencies and golden values of the frozen reference datasets
(``PANEL-F``, ``TS``) and of ``heterogeneous_coverage_panel``,
builder and call factories, public-path seeding of the three stores (as an
earlier stage would have done), a ``'training'`` mask provider shaped like
``ImputationWindowCalculator.get_mask_at_frequency``, the target scaling
through ``StageScaler`` and the order-insensitive comparison of two sets.
"""
# Modules de base
from typing import Callable, Dict, List, Optional, Tuple

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Objets testés
from tsforecast.imputation._components.training_set_builder import (
    TrainingSet,
    TrainingSetBuilder,
)
from tsforecast.imputation._components.covariate_materializer import CovariateMaterializer
from tsforecast.imputation._components.stage_scaler import StageScaler


# Fréquences détectées du jeu PANEL-F : `v` est une propriété du couple
# (entité, colonne), jamais de la seule colonne
PANEL_F_FREQUENCIES = {
    'm1': 'M',
    'q1': 'Q',
    'v': {('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'},
}
# Blocs attendus de `v` sur PANEL-F, quelle que soit l'étape
PANEL_F_BLOCKS = {('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'}
# Fréquences détectées du jeu TS
TS_FREQUENCIES = {'m1': 'M', 'q1': 'Q', 'a1': 'Y', 'a2': 'Y'}

# Covariables des jeux d'entraînement testés
FEATURES = ['m1', 'q1']

# Valeurs d'or brutes de `v` : ancres annuelles de FR, trimestrielles de
# DE, mensuelles d'IT (constantes dans l'année)
V_FR = [120.0, 132.0, 150.0]
V_DE = [28.0, 30.0, 31.0, 31.0, 31.0, 33.0, 34.0, 34.0, 36.0, 37.0, 38.0, 39.0]
V_IT = [10.0] * 12 + [11.0] * 12 + [12.5] * 12

# Imputations trimestrielles de `a1` / `a2` posées à la main dans le miroir,
# comme les aurait produites l'étape Q sous 'sum' : chaque année somme à son
# ancre (120 / 132 / 150 et 60 / 66 / 72)
QUARTERLY_MIRROR = {
    'a1': [28.0, 30.0, 31.0, 31.0, 31.0, 33.0, 34.0, 34.0, 36.0, 37.0, 38.0, 39.0],
    'a2': [14.0, 15.0, 15.5, 15.5, 15.5, 16.5, 17.0, 17.0, 18.0, 18.0, 18.0, 18.0],
}

# Fréquences détectées de `heterogeneous_coverage_panel` utiles aux tests :
# depenses_publiques_pib annuelle pour France / Italie, trimestrielle pour
# Allemagne ; climat_affaires jamais observée par Italie (aucune entrée)
HETEROGENEOUS_FREQUENCIES = {
    'inflation_ipc': 'M',
    'pib_trimestriel': 'Q',
    'climat_affaires': {('France',): 'M', ('Allemagne',): 'M'},
    'depenses_publiques_pib': {
        ('France',): 'Y', ('Allemagne',): 'Q', ('Italie',): 'Y'
    },
}
# Covariables retenues sur le jeu réaliste, dont une structurellement absente
HETEROGENEOUS_FEATURES = ['inflation_ipc', 'pib_trimestriel', 'climat_affaires']

# Alias pandas de fin de période des fréquences de base, pour les masques
_END_ALIASES = {'Y': 'YE', 'Q': 'QE', 'M': 'ME'}


# Fabrique de composant : matérialiseur réel, aucun estimateur
def make_builder(training_mask=None, log=None, **materializer_kwargs) -> TrainingSetBuilder:
    """Build a TrainingSetBuilder over a real materializer.

    Args:
        training_mask: Optional training-window mask provider.
        log: Optional logging callback.
        **materializer_kwargs: Arguments of the ``CovariateMaterializer``.

    Returns:
        A fresh builder, its materializer's stores empty.
    """
    materializer = CovariateMaterializer(**materializer_kwargs)
    return TrainingSetBuilder(materializer, training_mask=training_mask, log=log)


# Fabrique d'appel : les paramètres communs à tous les tests du panel
def build_panel_f(builder, data, stage_freq='M', eligible_origins=('observed',),
                   frequencies=None, column='v', feature_cols=None,
                   keep_coincident_cells=False, materialization=None) -> TrainingSet:
    """Compose the mutualized training set of ``v`` on ``PANEL-F``.

    Args:
        builder: Builder under test.
        data: ``PANEL-F`` or a perturbation of it.
        stage_freq: Stage frequency, scalar or per entity.
        eligible_origins: Admissible cell origins.
        frequencies: Detected frequencies, ``PANEL_F_FREQUENCIES`` when None.
        column: Imputed column.
        feature_cols: Covariates, ``FEATURES`` when None.
        keep_coincident_cells: Coincident-cell rule.
        materialization: Imposed ways, or None.

    Returns:
        The composed training set.
    """
    return builder.build(
        column=column,
        feature_cols=FEATURES if feature_cols is None else feature_cols,
        stage_freq=stage_freq,
        detected_frequencies=PANEL_F_FREQUENCIES if frequencies is None else frequencies,
        source_data=data,
        eligible_origins=set(eligible_origins),
        materialization=materialization,
        keep_coincident_cells=keep_coincident_cells,
    )


# Fabrique d'appel : le jeu de `a1` / `a2` sur TS
def build_ts(builder, data, column='a1', feature_cols=('m1', 'q1'),
              eligible_origins=('observed', 'interpolated', 'model'),
              keep_coincident_cells=False) -> TrainingSet:
    """Compose the training set of an annual column of ``TS`` at stage ``M``.

    Args:
        builder: Builder under test.
        data: ``TS``.
        column: ``'a1'`` or ``'a2'``.
        feature_cols: Covariates.
        eligible_origins: Admissible cell origins.
        keep_coincident_cells: Coincident-cell rule.

    Returns:
        The composed training set.
    """
    return builder.build(
        column=column,
        feature_cols=list(feature_cols),
        stage_freq='M',
        detected_frequencies=TS_FREQUENCIES,
        source_data=data,
        eligible_origins=set(eligible_origins),
        keep_coincident_cells=keep_coincident_cells,
    )


# Amorçage du miroir d'une colonne de TS, comme l'aurait fait l'étape Q
def seed_quarterly_mirror(builder: TrainingSetBuilder, column: str) -> TrainingSetBuilder:
    """Record the twelve quarterly imputations of ``column`` in the three stores.

    Args:
        builder: Builder whose materializer is seeded.
        column: ``'a1'`` or ``'a2'``.

    Returns:
        The same builder, for chaining.
    """
    # Douze fins de trimestre 2021-2023 : une sur quatre coïncide avec une
    # ancre annuelle (le 31 décembre)
    dates = pd.date_range('2021-03-31', periods=12, freq='QE')
    builder.materializer.record_production(
        column,
        pd.Series(QUARTERLY_MIRROR[column], index=dates),
        pd.Series(['model'] * 12, index=dates),
        pd.Series(['Q'] * 12, index=dates),
    )
    return builder


# Amorçage du miroir de `v` pour FR sur PANEL-F
def seed_panel_mirror(builder: TrainingSetBuilder, cells: List[Tuple[str, float, str]],
                       freq: str = 'Q') -> TrainingSetBuilder:
    """Record cells of ``v`` produced for ``FR`` by an earlier stage.

    Args:
        builder: Builder whose materializer is seeded.
        cells: ``(date, value, origin)`` triples.
        freq: Production frequency of every cell.

    Returns:
        The same builder, for chaining.
    """
    index = pd.MultiIndex.from_tuples(
        [('FR', pd.Timestamp(date)) for date, _value, _origin in cells],
        names=['country', 'date'],
    )
    builder.materializer.record_production(
        'v',
        pd.Series([value for _date, value, _origin in cells], index=index, dtype=float),
        pd.Series([origin for _date, _value, origin in cells], index=index, dtype='str'),
        pd.Series([freq] * len(cells), index=index, dtype='str'),
    )
    return builder


# Mise à l'échelle de la cible par le SEUL chemin d'échelle du package
def scaled_target(training: TrainingSet, stage: str) -> pd.Series:
    """Apply ``StageScaler.target_divisor(produced_freq=...)`` to a raw target.

    Args:
        training: Composed training set.
        stage: Stage frequency.

    Returns:
        The target brought back to the stage scale.
    """
    scaler = StageScaler()
    divisors = scaler.target_divisor(
        training.y.name,
        source_freq=dict(training.blocks),
        pred_freq=stage,
        index=training.y.index,
        produced_freq=training.row_frequency,
    )
    return scaler.apply(training.y, divisors)


# Comparaison de deux captures des trois registres
def snapshots_equal(left, right) -> bool:
    """Tell whether two materializer snapshots hold the same content.

    Args:
        left: First snapshot.
        right: Second snapshot.

    Returns:
        True when the stores, their columns and their series coincide.
    """
    if sorted(left) != sorted(right):
        return False
    for store in left:
        if sorted(left[store]) != sorted(right[store]):
            return False
        for column, series in left[store].items():
            if not series.equals(right[store][column]):
                return False
    return True


# Fabrique de masque 'training' : une grille de fin de période par entité, à la
# fréquence demandée, et une règle d'appartenance (entité, fréquence, date)
def layer_mask_provider(
    keep: Callable[[Tuple, str, pd.Timestamp], bool],
    asked: Optional[List[Dict]] = None,
    names: Tuple[str, str] = ('country', 'date'),
) -> Callable[[Dict], pd.Series]:
    """Return a mask provider shaped like ``get_mask_at_frequency``.

    Args:
        keep: Rule telling, for an (entity, frequency, date), whether the
            date is inside the training window.
        asked: List collecting each binding the builder asks for.
        names: Index level names of the returned panel mask.

    Returns:
        Callable mapping ``{entity: frequency}`` to a boolean Series on the
        period-end grid of each entity at its frequency, 2021-2023.
    """
    def _provider(frequencies):
        if asked is not None:
            asked.append(dict(frequencies))
        keys, values = [], []
        for entity, frequency in frequencies.items():
            grid = pd.date_range('2021-01-01', '2023-12-31', freq=_END_ALIASES[frequency])
            for date in grid:
                keys.append((*entity, date))
                values.append(keep(entity, frequency, date))
        # Série temporelle : entité dégénérée, masque indexé par la date seule
        if all(len(key) == 1 for key in keys):
            return pd.Series(values, index=pd.DatetimeIndex([key[0] for key in keys]))
        return pd.Series(values, index=pd.MultiIndex.from_tuples(keys, names=list(names)))
    return _provider


# Construction de PANEL-F positionné en DÉBUT de période
def panel_f_at_period_start(panel_f: pd.DataFrame) -> pd.DataFrame:
    """Rebuild ``PANEL-F`` with every observation dated at its period start.

    ``to_period_start`` moves each date to its MONTH start (the grid is
    monthly), which would date a yearly anchor on December 1st. A consistent
    start-positioned dataset dates the yearly anchor on January 1st and the
    quarterly ones on the first day of their quarter.

    Args:
        panel_f: The end-positioned ``PANEL-F``.

    Returns:
        Same values, ``MS`` grid, ``YS`` / ``QS`` / ``MS`` anchors.
    """
    frames = []
    # Entités dans l'ordre des données d'origine
    for entity in panel_f.index.get_level_values(0).unique():
        block = panel_f.loc[entity]
        out = pd.DataFrame(
            np.nan,
            index=pd.date_range('2021-01-01', periods=36, freq='MS'),
            columns=block.columns,
        )
        out['m1'] = block['m1'].to_numpy()
        # Ancres trimestrielles : premier jour du trimestre
        for date, value in block['q1'].dropna().items():
            out.loc[pd.Timestamp(date.year, date.month - 2, 1), 'q1'] = value
        # Ancres de `v` : début d'année, de trimestre ou de mois selon le bloc
        for date, value in block['v'].dropna().items():
            if entity == 'FR':
                start = pd.Timestamp(date.year, 1, 1)
            elif entity == 'DE':
                start = pd.Timestamp(date.year, date.month - 2, 1)
            else:
                start = pd.Timestamp(date.year, date.month, 1)
            out.loc[start, 'v'] = value
        out.index = pd.MultiIndex.from_product(
            [[entity], out.index], names=['country', 'date']
        )
        frames.append(out)
    return pd.concat(frames)


# Comparaison de deux jeux d'entraînement au tri près
def assert_same_rows(left: TrainingSet, right: TrainingSet) -> None:
    """Assert two training sets hold the same rows, up to their order.

    Args:
        left: First training set.
        right: Second training set.
    """
    assert dict(left.blocks) == dict(right.blocks)
    pd.testing.assert_series_equal(left.y.sort_index(), right.y.sort_index())
    pd.testing.assert_series_equal(
        left.row_frequency.sort_index(), right.row_frequency.sort_index()
    )
    pd.testing.assert_series_equal(
        left.row_origin.sort_index(), right.row_origin.sort_index()
    )
    pd.testing.assert_frame_equal(left.X.sort_index(), right.X.sort_index())
    assert dict(left.ways) == dict(right.ways)
    assert dict(left.column_origins) == dict(right.column_origins)
