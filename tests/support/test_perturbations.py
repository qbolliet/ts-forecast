"""Tests de ``tests/support/perturbations.py``.

Une fonction pure par cas limite de ``CLAUDE.md`` — chaque test vérifie que la
perturbation produit exactement la forme annoncée par sa docstring, sur de
petits jeux construits à la main et sur les jeux réalistes du notebook 3
(``nb3_panel``) pour les perturbations spécifiques aux panels.
"""
# Manipulation de données
import pandas as pd
import pytest

from tests.support.perturbations import (
    drop_entity,
    empty_like,
    reverse_entities,
    shuffle_rows,
    single_observation,
    to_period_end,
    to_period_index,
    to_period_start,
    to_three_level_index,
    with_duplicated_rows,
    with_index_names,
    with_special_column_names,
)


@pytest.fixture
def small_timeseries() -> pd.DataFrame:
    """Small hand-built time series, month-start anchored, 4 rows."""
    dates = pd.date_range('2020-01-01', periods=4, freq='MS')
    return pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [10.0, 20.0, 30.0, 40.0]}, index=dates)


@pytest.fixture
def small_panel() -> pd.DataFrame:
    """Small hand-built panel, two entities, MultiIndex (entity, date)."""
    dates = pd.date_range('2020-01-01', periods=2, freq='MS')
    idx = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
    return pd.DataFrame({'v': [1.0, 2.0, 3.0, 4.0]}, index=idx)


class TestShuffleRows:
    """``shuffle_rows`` : désordonnancement pur, sans perte ni gain de ligne."""

    def test_same_rows_different_order(self, small_timeseries: pd.DataFrame) -> None:
        """L'ensemble des lignes (index, valeurs) est préservé, l'ordre non."""
        shuffled = shuffle_rows(small_timeseries, seed=0)

        assert sorted(shuffled.index) == sorted(small_timeseries.index)
        assert not shuffled.index.equals(small_timeseries.index)
        for date in small_timeseries.index:
            assert (shuffled.loc[date] == small_timeseries.loc[date]).all()

    def test_reproducible_with_same_seed(self, small_timeseries: pd.DataFrame) -> None:
        """La même graine rend la même permutation."""
        first = shuffle_rows(small_timeseries, seed=1)
        second = shuffle_rows(small_timeseries, seed=1)
        pd.testing.assert_frame_equal(first, second)

    def test_does_not_mutate_input(self, small_timeseries: pd.DataFrame) -> None:
        """L'entrée reste triée après appel (fonction pure)."""
        original_index = small_timeseries.index.copy()
        shuffle_rows(small_timeseries, seed=0)
        assert small_timeseries.index.equals(original_index)


class TestReverseEntities:
    """``reverse_entities`` : ordre des blocs d'entités inversé, ordre interne intact."""

    def test_entity_block_order_is_reversed(self, small_panel: pd.DataFrame) -> None:
        """Les entités apparaissent dans l'ordre inverse de leur première apparition."""
        reversed_panel = reverse_entities(small_panel)

        seen_order = list(dict.fromkeys(reversed_panel.index.get_level_values('entity')))
        assert seen_order == ['B', 'A']

    def test_within_entity_row_order_preserved(self, small_panel: pd.DataFrame) -> None:
        """Au sein de chaque entité, l'ordre des dates n'est pas modifié."""
        reversed_panel = reverse_entities(small_panel)

        for entity in ('A', 'B'):
            pd.testing.assert_frame_equal(
                reversed_panel.loc[[entity]], small_panel.loc[[entity]]
            )

    def test_raises_on_non_multiindex(self, small_timeseries: pd.DataFrame) -> None:
        """Une série temporelle simple (index à un seul niveau) n'a pas d'entité à inverser."""
        with pytest.raises(TypeError):
            reverse_entities(small_timeseries)

    def test_nb3_panel_entity_order_is_reversed(self, nb3_panel: pd.DataFrame) -> None:
        """Propriété (pas de valeur d'or) sur le jeu réaliste : mêmes entités, ordre inversé."""
        reversed_panel = reverse_entities(nb3_panel)
        original_entities = list(dict.fromkeys(nb3_panel.index.get_level_values('country')))
        reversed_entities = list(dict.fromkeys(reversed_panel.index.get_level_values('country')))
        assert reversed_entities == list(reversed(original_entities))
        assert len(reversed_panel) == len(nb3_panel)


class TestWithSpecialColumnNames:
    """``with_special_column_names`` : espaces, accents, ``/ % (`` dans les noms."""

    def test_mapping_matches_renamed_columns(self, small_timeseries: pd.DataFrame) -> None:
        """``mapping`` associe chaque nom d'origine à son remplacement spécial."""
        renamed, mapping = with_special_column_names(small_timeseries)

        assert set(mapping.keys()) == set(small_timeseries.columns)
        assert list(renamed.columns) == [mapping[col] for col in small_timeseries.columns]

    def test_special_characters_present(self) -> None:
        """Au moins un nom contient chacun des caractères spéciaux ciblés (5 colonnes : un par suffixe)."""
        df = pd.DataFrame({f'col{i}': [0] for i in range(5)})
        renamed, _ = with_special_column_names(df)
        joined = ' '.join(renamed.columns)

        for char in (' ', 'é', '/', '%', '('):
            assert char in joined

    def test_values_untouched(self, small_timeseries: pd.DataFrame) -> None:
        """Seuls les noms de colonnes changent, les valeurs restent identiques."""
        renamed, _ = with_special_column_names(small_timeseries)
        pd.testing.assert_frame_equal(
            renamed.set_axis(small_timeseries.columns, axis=1), small_timeseries
        )


class TestWithIndexNames:
    """``with_index_names`` : renommage de l'index simple ou de chaque niveau."""

    def test_single_index_renamed(self, small_timeseries: pd.DataFrame) -> None:
        """Un index simple prend le nouveau nom directement."""
        renamed = with_index_names(small_timeseries, 'periode')
        assert renamed.index.name == 'periode'

    def test_multiindex_levels_renamed(self, small_panel: pd.DataFrame) -> None:
        """Chaque niveau d'un ``MultiIndex`` prend le nom correspondant."""
        renamed = with_index_names(small_panel, ['pays', 'periode'])
        assert list(renamed.index.names) == ['pays', 'periode']


class TestToThreeLevelIndex:
    """``to_three_level_index`` : ajout d'un niveau région au-dessus de l'entité."""

    def test_adds_outer_level_with_default_region(self, small_panel: pd.DataFrame) -> None:
        """Sans mappage, toutes les lignes reçoivent la même région par défaut."""
        three_level = to_three_level_index(small_panel)

        assert list(three_level.index.names) == ['region', 'entity', 'date']
        assert set(three_level.index.get_level_values('region')) == {'Zone euro'}

    def test_entity_and_date_levels_unchanged(self, small_panel: pd.DataFrame) -> None:
        """Les niveaux entité et date conservent leurs valeurs d'origine."""
        three_level = to_three_level_index(small_panel)

        assert list(three_level.index.get_level_values('entity')) == list(
            small_panel.index.get_level_values('entity')
        )
        assert list(three_level.index.get_level_values('date')) == list(
            small_panel.index.get_level_values('date')
        )

    def test_custom_region_mapping(self, small_panel: pd.DataFrame) -> None:
        """Un mappage explicite affecte une région différente par entité."""
        three_level = to_three_level_index(small_panel, region_by_entity={'A': 'Nord'})

        regions = dict(zip(
            three_level.index.get_level_values('entity'),
            three_level.index.get_level_values('region'),
        ))
        assert regions['A'] == 'Nord'
        assert regions['B'] == 'Zone euro'  # absente du mappage : région par défaut

    def test_raises_on_two_level_requirement(self, small_timeseries: pd.DataFrame) -> None:
        """Un index à un seul niveau n'est pas un panel à deux niveaux."""
        with pytest.raises(TypeError):
            to_three_level_index(small_timeseries)


class TestPeriodPosition:
    """``to_period_start`` / ``to_period_end`` / ``to_period_index``."""

    def test_start_to_end_moves_off_month_start(self, small_timeseries: pd.DataFrame) -> None:
        """Un index ``MS`` bascule vers des dates de fin de période."""
        end_anchored = to_period_end(small_timeseries)
        assert not any(date.day == 1 for date in end_anchored.index)

    def test_round_trip_recovers_month_start(self, small_timeseries: pd.DataFrame) -> None:
        """Un aller-retour début -> fin -> début retombe sur la grille d'origine."""
        round_tripped = to_period_start(to_period_end(small_timeseries))
        pd.testing.assert_index_equal(
            round_tripped.index.sort_values(), small_timeseries.index.sort_values()
        )

    def test_already_start_anchored_is_a_no_op(self, small_timeseries: pd.DataFrame) -> None:
        """Demander le début de période sur un index déjà ``MS`` ne change rien."""
        result = to_period_start(small_timeseries)
        pd.testing.assert_frame_equal(result, small_timeseries)

    def test_panel_positions_flip_per_entity(self, small_panel: pd.DataFrame) -> None:
        """La conversion s'applique à chaque entité du panel, pas seulement à la première."""
        end_anchored = to_period_end(small_panel)
        dates = end_anchored.index.get_level_values('date')
        assert not any(date.day == 1 for date in dates)

    def test_irregular_index_still_converts_the_common_grid(
        self, nb3_timeseries: pd.DataFrame
    ) -> None:
        """Un index irrégulier (ancres annuelles hors grille) est tout de même converti.

        ``convert_position`` n'exige qu'un pas localement détectable, pas une
        grille entièrement régulière : contrairement à ``to_period_index``
        (``pd.infer_freq``, strict), l'irrégularité ne bloque pas ici la
        conversion.
        """
        result = to_period_end(nb3_timeseries)
        assert not any(date.day == 1 and date.hour == 0 for date in result.index)

    def test_empty_index_is_a_no_op(self, small_timeseries: pd.DataFrame) -> None:
        """Un jeu vide n'a aucune date dont inférer la position : no-op documenté."""
        empty = small_timeseries.iloc[0:0]
        result = to_period_end(empty)
        assert len(result) == 0

    def test_to_period_index_returns_period_index(self, small_timeseries: pd.DataFrame) -> None:
        """Sur un index régulier, le résultat est bien un ``PeriodIndex``."""
        converted = to_period_index(small_timeseries)
        assert isinstance(converted.index, pd.PeriodIndex)
        assert converted.index.freqstr == 'M'

    def test_to_period_index_on_panel_converts_date_level_only(self) -> None:
        """Sur un panel, seul le dernier niveau (date) devient un ``PeriodIndex``.

        ``pd.infer_freq`` exige au moins 3 dates distinctes : un panel à 3
        périodes par entité (au lieu des 2 de ``small_panel``) est nécessaire
        ici.
        """
        dates = pd.date_range('2020-01-01', periods=3, freq='MS')
        idx = pd.MultiIndex.from_product([['A', 'B'], dates], names=['entity', 'date'])
        panel = pd.DataFrame({'v': range(6)}, index=idx)

        converted = to_period_index(panel)
        assert isinstance(converted.index.get_level_values('date'), pd.PeriodIndex)
        assert list(converted.index.get_level_values('entity')) == list(
            panel.index.get_level_values('entity')
        )

    def test_to_period_index_irregular_is_a_no_op(self, nb3_timeseries: pd.DataFrame) -> None:
        """``pd.infer_freq`` échoue sur un index irrégulier : la conversion est sautée."""
        result = to_period_index(nb3_timeseries)
        assert isinstance(result.index, pd.DatetimeIndex)


class TestDropEntity:
    """``drop_entity`` : entité manquante, ni observée ni présente en NaN."""

    def test_entity_rows_removed(self, small_panel: pd.DataFrame) -> None:
        """L'entité retirée n'apparaît plus du tout dans l'index."""
        dropped = drop_entity(small_panel, 'A')
        assert 'A' not in dropped.index.get_level_values('entity')
        assert set(dropped.index.get_level_values('entity')) == {'B'}

    def test_other_entities_untouched(self, small_panel: pd.DataFrame) -> None:
        """Les lignes des autres entités restent identiques."""
        dropped = drop_entity(small_panel, 'A')
        pd.testing.assert_frame_equal(dropped.loc[['B']], small_panel.loc[['B']])

    def test_raises_on_non_multiindex(self, small_timeseries: pd.DataFrame) -> None:
        """Pas de notion d'entité sur une série temporelle simple."""
        with pytest.raises(TypeError):
            drop_entity(small_timeseries, 'A')


class TestWithDuplicatedRows:
    """``with_duplicated_rows`` : index dupliqué, en queue de frame."""

    def test_length_increases_by_n(self, small_timeseries: pd.DataFrame) -> None:
        """La longueur croît exactement de ``n``."""
        duplicated = with_duplicated_rows(small_timeseries, n=2)
        assert len(duplicated) == len(small_timeseries) + 2

    def test_duplicated_index_values_appear_twice(self, small_timeseries: pd.DataFrame) -> None:
        """Les dates dupliquées apparaissent deux fois dans l'index résultant."""
        duplicated = with_duplicated_rows(small_timeseries, n=1)
        first_date = small_timeseries.index[0]
        assert (duplicated.index == first_date).sum() == 2


class TestSingleObservation:
    """``single_observation`` : jeu réduit à une seule ligne."""

    def test_returns_one_row(self, small_timeseries: pd.DataFrame) -> None:
        """Une seule ligne, celle d'origine, colonnes inchangées."""
        single = single_observation(small_timeseries)
        assert len(single) == 1
        pd.testing.assert_frame_equal(single, small_timeseries.iloc[[0]])


class TestEmptyLike:
    """``empty_like`` : jeu vide, forme (colonnes, dtypes, noms d'index) conservée."""

    def test_zero_rows_same_columns(self, small_timeseries: pd.DataFrame) -> None:
        """Zéro ligne, mêmes colonnes et mêmes dtypes que l'original."""
        empty = empty_like(small_timeseries)
        assert len(empty) == 0
        assert list(empty.columns) == list(small_timeseries.columns)
        pd.testing.assert_series_equal(empty.dtypes, small_timeseries.dtypes)

    def test_index_name_preserved(self, small_timeseries: pd.DataFrame) -> None:
        """Le nom de l'index reste renseigné malgré l'absence de lignes."""
        empty = empty_like(small_timeseries)
        assert empty.index.name == small_timeseries.index.name
