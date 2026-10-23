"""Tests of the grid decomposition and of the ``TrainingSet`` container.

``split_training_index`` is the single place reading the two shapes of a
training grid — ``(entity..., date)`` and, under ``keep_coincident_cells``,
``(entity..., frequency, date)``. Pinned here: round trip with and without the frequency level (the
real grid, the entity keys and the row frequencies rebuild the stamped grid
exactly), three-level entities, empty grids, and the flag deciding alone —
an entity level NAMED ``'frequency'`` is never sniffed, a frequency level
under any name is dropped by position. ``TrainingSet``: exact fields,
immutability, ``has_frequency_level`` default, ``len`` and the shared grid.

Covered symbols: ``split_training_index``, ``TrainingSet``,
``FREQUENCY_LEVEL_NAME``.
"""
# Modules de base
import dataclasses

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objets testés
from tsforecast.frequency.training_set_builder import (
    FREQUENCY_LEVEL_NAME,
    TrainingSet,
    split_training_index,
)

# Constantes et fabriques partagées du paquet
from tests.unit.frequency.training_set_builder.helpers import (
    build_panel_f,
    make_builder,
    seed_panel_mirror,
)


# =============================================================================
# split_training_index
# =============================================================================
class TestSplitTrainingIndex:
    """The single decomposition of a training grid, stamped or not."""

    # Dates communes des index construits à la main
    DATES = pd.to_datetime(['2021-03-31', '2021-12-31'])

    def test_time_series_without_level_is_returned_as_is(self):
        """An unstamped time-series grid is its own real index, entity ``()``."""
        index = pd.DatetimeIndex(self.DATES, name='date')

        real, entities = split_training_index(index, has_frequency_level=False)

        # Aucun niveau ajouté : la grille réelle EST l'entrée
        assert real is index
        assert entities == [(), ()]

    @pytest.mark.parametrize(
        'entity_levels, expected',
        [
            ([['FR', 'DE']], [('FR',), ('DE',)]),
            ([['Zone euro', 'Zone euro'], ['FR', 'DE']],
             [('Zone euro', 'FR'), ('Zone euro', 'DE')]),
        ],
        ids=['two_levels', 'three_levels'],
    )
    def test_panel_without_level_yields_every_entity_level(self, entity_levels, expected):
        """Unstamped panel: every level but the date belongs to the entity."""
        index = pd.MultiIndex.from_arrays([*entity_levels, self.DATES])

        real, entities = split_training_index(index, has_frequency_level=False)

        assert real is index
        assert entities == expected

    @pytest.mark.parametrize(
        'entity_levels, expected_entities, expected_names',
        [
            ([], [(), ()], ['date']),
            ([['FR', 'FR']], [('FR',), ('FR',)], ['country', 'date']),
            ([['Zone euro', 'Zone euro'], ['FR', 'FR']],
             [('Zone euro', 'FR'), ('Zone euro', 'FR')],
             ['region', 'country', 'date']),
        ],
        ids=['time_series', 'two_levels', 'three_levels'],
    )
    def test_stamped_grid_drops_the_frequency_level(
        self, entity_levels, expected_entities, expected_names
    ):
        """Stamped grid: the level before the date is dropped, entities kept."""
        # Grille estampillée : les deux cellules du 2021-12-31 (année, T4) et
        # celle du 2021-03-31 ; le niveau de fréquence est l'avant-dernier
        dates = pd.to_datetime(['2021-12-31', '2021-12-31'])
        names = ['region', 'country'][-len(entity_levels):] if entity_levels else []
        index = pd.MultiIndex.from_arrays(
            [*entity_levels, ['Y', 'Q'], dates],
            names=names + [FREQUENCY_LEVEL_NAME, 'date'],
        )

        real, entities = split_training_index(index, has_frequency_level=True)

        assert entities == expected_entities
        assert list(real.names) == expected_names
        # La date réelle est conservée, doublon compris : deux cellules
        # coïncidentes partagent par définition une date
        assert list(real.get_level_values(-1)) == list(dates)

    def test_stamped_time_series_gives_a_datetime_index(self):
        """A stamped ``(freq, date)`` grid comes back as a plain DatetimeIndex."""
        index = pd.MultiIndex.from_arrays(
            [['Y', 'Q'], self.DATES], names=[FREQUENCY_LEVEL_NAME, 'date']
        )

        real, _entities = split_training_index(index, has_frequency_level=True)

        assert isinstance(real, pd.DatetimeIndex)

    def test_level_named_frequency_is_never_sniffed(self):
        """An ENTITY level named ``'frequency'`` stays an entity without the flag."""
        # Cas piège : un niveau d'entité porte le nom du niveau de fréquence ;
        # seul le drapeau décide, jamais les noms
        index = pd.MultiIndex.from_arrays(
            [['FR', 'DE'], self.DATES], names=[FREQUENCY_LEVEL_NAME, 'date']
        )

        real, entities = split_training_index(index, has_frequency_level=False)

        assert real is index
        assert entities == [('FR',), ('DE',)]

    def test_frequency_level_under_any_name_is_dropped_by_position(self):
        """With the flag, the level before the date goes, whatever its name."""
        index = pd.MultiIndex.from_arrays(
            [['FR', 'FR'], ['Y', 'Q'], self.DATES], names=['pays', 'pas', 'periode']
        )

        real, entities = split_training_index(index, has_frequency_level=True)

        assert list(real.names) == ['pays', 'periode']
        assert entities == [('FR',), ('FR',)]

    @pytest.mark.parametrize('has_level', [False, True], ids=['plain', 'stamped'])
    def test_empty_grid(self, has_level):
        """An empty grid yields an empty real index and no entity."""
        names = ['country', FREQUENCY_LEVEL_NAME, 'date'] if has_level else ['country', 'date']
        index = pd.MultiIndex.from_tuples([], names=names)

        real, entities = split_training_index(index, has_frequency_level=has_level)

        assert len(real) == 0
        assert entities == []

    def test_round_trip_on_a_stamped_training_set(self, mixed_freq_panel_multifrequency):
        """Real index + entities + row frequency rebuild the stamped grid exactly."""
        # Coïncidence FR au 2021-12-31 : ancre annuelle et imputation du T4
        builder = seed_panel_mirror(make_builder(), [('2021-12-31', 31.0, 'model')])
        training = build_panel_f(
            builder, mixed_freq_panel_multifrequency,
            eligible_origins=('observed', 'model'), keep_coincident_cells=True,
        )
        assert training.has_frequency_level

        real, entities = split_training_index(
            training.X.index, has_frequency_level=training.has_frequency_level
        )

        # Aller-retour : la fréquence de ligne réinsérée avant la date redonne
        # la grille estampillée à l'identique
        rebuilt = pd.MultiIndex.from_tuples(
            [
                (*entity, frequency, date)
                for entity, frequency, date in zip(
                    entities, training.row_frequency, real.get_level_values(-1)
                )
            ],
            names=training.X.index.names,
        )
        assert rebuilt.equals(training.X.index)

    def test_round_trip_real_grid_matches_the_unstamped_set(
        self, mixed_freq_panel_multifrequency
    ):
        """The real grid of a stamped set covers the same (entity, date) keys."""
        builder = seed_panel_mirror(make_builder(), [('2021-12-31', 31.0, 'model')])
        stamped = build_panel_f(
            builder, mixed_freq_panel_multifrequency,
            eligible_origins=('observed', 'model'), keep_coincident_cells=True,
        )
        plain = build_panel_f(
            builder, mixed_freq_panel_multifrequency,
            eligible_origins=('observed', 'model'), keep_coincident_cells=False,
        )

        real, _entities = split_training_index(stamped.X.index, has_frequency_level=True)

        # Grille réelle = grille sans estampille, plus le seul doublon de la
        # cellule coïncidente conservée
        assert real.duplicated().sum() == 1
        assert real[~real.duplicated()].sort_values().equals(plain.X.index.sort_values())
        # Grille sans estampille : la décomposition est l'identité
        plain_real, _ = split_training_index(plain.X.index, has_frequency_level=False)
        assert plain_real is plain.X.index


# =============================================================================
# TrainingSet
# =============================================================================
class TestTrainingSet:
    """The immutable container handed back by ``build``."""

    def test_fields(self):
        """The fields are exactly the documented ones, in order."""
        assert [field.name for field in dataclasses.fields(TrainingSet)] == [
            'X', 'y', 'row_frequency', 'row_origin', 'blocks', 'ways',
            'column_origins', 'has_frequency_level',
        ]

    def test_has_frequency_level_defaults_to_false(self):
        """Without an explicit flag the grid is the historical one."""
        empty = TrainingSet(
            X=pd.DataFrame(), y=pd.Series(dtype=float),
            row_frequency=pd.Series(dtype='str'), row_origin=pd.Series(dtype='str'),
            blocks={}, ways={}, column_origins={},
        )

        assert empty.has_frequency_level is False
        assert len(empty) == 0

    def test_frozen(self, mixed_freq_panel_multifrequency):
        """A field cannot be reassigned."""
        training = build_panel_f(make_builder(), mixed_freq_panel_multifrequency)

        with pytest.raises(dataclasses.FrozenInstanceError):
            training.has_frequency_level = True

    def test_len_is_the_number_of_target_rows(self, mixed_freq_panel_multifrequency):
        """``len`` counts the rows of ``y``, the measure of the mutualization gain."""
        training = build_panel_f(make_builder(), mixed_freq_panel_multifrequency)

        assert len(training) == len(training.y) == 51

    def test_every_series_shares_the_grid(self, mixed_freq_panel_multifrequency):
        """``X``, ``y``, ``row_frequency`` and ``row_origin`` share one index."""
        training = build_panel_f(make_builder(), mixed_freq_panel_multifrequency)

        for series in (training.y, training.row_frequency, training.row_origin):
            assert series.index.equals(training.X.index)
        assert training.y.name == 'v'
        assert training.row_frequency.name == 'row_frequency'
        assert training.row_origin.name == 'row_origin'
