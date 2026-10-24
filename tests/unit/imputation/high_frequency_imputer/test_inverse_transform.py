"""``HighFrequencyImputer.inverse_transform``.

Covers the transform / inverse round trip — index, level names and observed
values given back on perturbed panels (three-level index, non-standard
names, special column names), the ``(X, y)`` pair, a pass on new data, the
input index and order given back whatever rows the target grid added or
left out —, multi-level entity names, the provenance-driven reset
of imputed cells, the choice of the stacked level to keep — read on the data
to invert, a level the last pass never produced being refused —, and the
columns the last pass never saw.
"""

# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Objets testés
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer

# Outils de test partagés
from tests.support.perturbations import (
    to_three_level_index,
    with_index_names,
    with_special_column_names,
)
from tests.unit.frequency.high_frequency_imputer.helpers import (
    fit_quietly,
    fit_transform_quietly,
    make_imputer,
    transform_quietly,
)


class TestInverseTransform:
    """Aller-retour transform / inverse_transform."""

    def test_inverse_transform_roundtrip(self, reference_timeseries):
        """The index and names are given back, the values on request."""
        imputer = make_imputer(restore_original_values=True)
        transformed = fit_transform_quietly(imputer, reference_timeseries)
        restored = imputer.inverse_transform(transformed)

        # Index et noms de niveaux d'origine
        assert restored.index.equals(reference_timeseries.index)
        assert restored.index.names == reference_timeseries.index.names

        # Valeurs d'origine, cellule par cellule
        for column in reference_timeseries.columns:
            observed = reference_timeseries[column].notna()
            assert np.allclose(
                restored.loc[observed, column].to_numpy(),
                reference_timeseries.loc[observed, column].to_numpy(),
            )

    def test_inverse_transform_preserves_multilevel_entity_names(
        self, panel_two_level_dataset
    ):
        """A panel with two entity levels keeps its three names."""
        imputer = HighFrequencyImputer(
            target_frequency='M',
            estimator=LinearRegression(),
            restore_original_values=True,
        )
        transformed = fit_transform_quietly(imputer, panel_two_level_dataset)

        # Le niveau de fréquence s'insère du côté de l'entité
        assert list(transformed.index.names) == [
            'country', 'sector', 'frequency', 'date'
        ]

        restored = imputer.inverse_transform(transformed)
        assert list(restored.index.names) == ['country', 'sector', 'date']
        # L'index d'entrée exactement, avec ses valeurs d'origine : les lignes
        # que la grille cible porte au-delà des données n'y reviennent pas
        pd.testing.assert_index_equal(restored.index, panel_two_level_dataset.index)
        source = panel_two_level_dataset['indicateur_mensuel']
        assert np.allclose(restored['indicateur_mensuel'].to_numpy(), source.to_numpy())

    def test_inverse_transform_without_a_provenance_matrix_raises(
        self, reference_timeseries
    ):
        """Sans matrice de provenance, l'inversion refuse : c'est elle qui guide.

        Le fit en écrit déjà une — inverser juste après lui est donc licite,
        et lit alors la provenance du fit. La garde couvre l'état purgé.
        """
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        del imputer.__dict__['imputation_provenance_']
        with pytest.raises(ValueError, match='previous call to transform'):
            imputer.inverse_transform(reference_timeseries)

    def test_inverse_right_after_a_fit_reads_the_fit_provenance(
        self, reference_timeseries
    ):
        """Le fit écrit sa propre matrice : l'inversion la lit sans transform."""
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        restored = imputer.inverse_transform(reference_timeseries)
        assert restored.index.equals(reference_timeseries.index)

    def test_restoring_inverse_right_after_a_fit(self, reference_timeseries):
        """``restore_original_values=True`` inverts right after a fit, like after a transform.

        Inverting right after a fit is licit (the fit writes its own provenance
        matrix and snapshots its own input); the exact restoration
        then refills the cells observed in the input of the fit.
        """
        imputer = fit_quietly(
            make_imputer(restore_original_values=True), reference_timeseries
        )
        restored = imputer.inverse_transform(reference_timeseries)

        assert restored.index.equals(reference_timeseries.index)
        for column in reference_timeseries.columns:
            observed = reference_timeseries[column].notna()
            assert np.allclose(
                restored.loc[observed, column].to_numpy(),
                reference_timeseries.loc[observed, column].to_numpy(),
            )

    def test_imputed_cells_are_set_back_to_nan(self, reference_timeseries):
        """Sans restore_original_values, toute cellule non ORIGINAL redevient NaN."""
        imputer = make_imputer()
        transformed = fit_transform_quietly(imputer, reference_timeseries)
        restored = imputer.inverse_transform(transformed)
        # m1 est observée partout : elle traverse intacte
        assert restored['m1'].notna().all()
        # a1 est imputée : elle ne peut pas être plus renseignée qu'à l'entrée
        assert restored['a1'].notna().sum() <= reference_timeseries['a1'].notna().sum()


# Données perturbées du panel PANEL-X, par nom de perturbation
def _perturbed(name: str, data: pd.DataFrame) -> pd.DataFrame:
    """Return PANEL-X under one of the index or column perturbations.

    Args:
        name: ``'three_level_index'``, ``'non_standard_index_names'`` or
            ``'special_column_names'``.
        data: PANEL-X.

    Returns:
        The perturbed frame.
    """
    if name == 'three_level_index':
        return to_three_level_index(data)
    if name == 'non_standard_index_names':
        return with_index_names(data, ['pays ', 'période (fin)'])
    return with_special_column_names(data)[0]


PERTURBATIONS = ['three_level_index', 'non_standard_index_names', 'special_column_names']


class TestRoundTripRestoresObservations:
    """``inverse_transform(transform(X))`` gives X back: index, names, observations."""

    @pytest.mark.slow
    @pytest.mark.parametrize('perturbation', PERTURBATIONS)
    def test_index_and_observed_values_come_back(self, panel_reference_full, perturbation):
        """Under ``restore_original_values=True``, X comes back exactly, NaN layout included."""
        data = _perturbed(perturbation, panel_reference_full)
        imputer = make_imputer(restore_original_values=True)
        restored = imputer.inverse_transform(fit_transform_quietly(imputer, data))

        pd.testing.assert_index_equal(restored.index, data.index, exact=True)
        # Cellules observées : valeurs d'origine ; cellules vides : vides
        pd.testing.assert_frame_equal(restored.where(data.notna()), data, check_freq=False)

    @pytest.mark.parametrize('keep', [True, False], ids=['stacked', 'flat'])
    def test_without_restoration_nothing_is_invented(self, mixed_freq_panel_multifrequency, keep):
        """Without exact restoration, every value given back is an input observation.

        The anchors re-expressed by the imputation are not ``ORIGINAL``
        and come back NaN; nothing imputed may leak through, and the fully
        observed ``m1`` comes back whole.
        """
        data = mixed_freq_panel_multifrequency
        imputer = make_imputer(keep_lower_frequencies=keep)
        restored = imputer.inverse_transform(fit_transform_quietly(imputer, data))

        # Aucune cellule rendue que l'entrée n'observait pas, aucune valeur altérée
        assert not (restored.notna() & data.isna()).any().any()
        pd.testing.assert_frame_equal(
            restored.where(restored.notna()), data.where(restored.notna()), check_freq=False
        )
        pd.testing.assert_series_equal(restored['m1'], data['m1'], check_freq=False)

    def test_inverse_of_a_pass_on_new_data_gives_that_data_back(self, reference_timeseries):
        """The inversion undoes the LAST pass: a transform on later dates comes back whole."""
        imputer = fit_quietly(
            make_imputer(restore_original_values=True),
            reference_timeseries.loc[:'2022-12-31'],
        )
        transformed = transform_quietly(imputer, reference_timeseries)

        restored = imputer.inverse_transform(transformed)

        pd.testing.assert_index_equal(restored.index, reference_timeseries.index)
        pd.testing.assert_frame_equal(
            restored.where(reference_timeseries.notna()), reference_timeseries,
            check_freq=False,
        )

    def test_target_pair_round_trip(self, reference_timeseries):
        """``(X, y)`` in, ``(X, y)`` out, both on X's index, the anchors of y restored."""
        X = reference_timeseries[['m1', 'q1', 'a2']]
        y = reference_timeseries['a1']
        imputer = make_imputer(restore_original_values=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            X_out, y_out = imputer.fit_transform(X, y)

        X_back, y_back = imputer.inverse_transform(X_out, y_out)

        pd.testing.assert_index_equal(X_back.index, X.index)
        assert y_back.name == 'a1'
        # Valeurs d'or : ancres 120 / 132 / 150
        np.testing.assert_allclose(y_back.dropna().to_numpy(), [120.0, 132.0, 150.0])

    def test_columns_unknown_to_the_last_pass_go_through_untouched(self, reference_timeseries):
        """Columns neither tracked nor in the snapshot are left as given, NaN and values alike."""
        imputer = make_imputer(restore_original_values=True, keep_lower_frequencies=False)
        transformed = fit_transform_quietly(imputer, reference_timeseries)
        renamed = transformed.add_suffix(' (copie)')

        restored = imputer.inverse_transform(renamed)

        pd.testing.assert_frame_equal(restored, renamed)

    def test_inverse_restores_the_row_order(self, mixed_freq_panel_multifrequency):
        """Per-entity targets: the inversion gives the rows back in X's order.

        ``DE``'s months off the quarter ends are not in the output; the
        inversion gives them back, each observation restored.
        """
        imputer = make_imputer(
            target_frequency={'FR': 'M', 'DE': 'Q', 'IT': 'M'}, restore_original_values=True
        )
        restored = imputer.inverse_transform(
            fit_transform_quietly(imputer, mixed_freq_panel_multifrequency)
        )
        pd.testing.assert_index_equal(restored.index, mixed_freq_panel_multifrequency.index)
        pd.testing.assert_frame_equal(
            restored.where(mixed_freq_panel_multifrequency.notna()),
            mixed_freq_panel_multifrequency, check_freq=False,
        )

    @pytest.mark.parametrize('restore', [False, True], ids=['mask', 'restore'])
    def test_inverse_drops_the_rows_the_grid_added(self, irregular_index_timeseries, restore):
        """On the realistic irregular index, X's index comes back, not the densified grid.

        82 rows in, 115 on the target grid: the 33 months the
        grid adds before the monthly history carry nothing once inverted.
        """
        imputer = make_imputer(restore_original_values=restore)
        transformed = fit_transform_quietly(imputer, irregular_index_timeseries)
        assert len(transformed) > len(irregular_index_timeseries)

        restored = imputer.inverse_transform(transformed)

        pd.testing.assert_index_equal(restored.index, irregular_index_timeseries.index)


class TestInverseLevelSelection:
    """Which stacked level the inversion keeps: source, else target, else the last one."""

    def test_target_level_kept_when_the_source_level_was_never_a_stage(
        self, reference_timeseries
    ):
        """A quarterly target on a monthly index: the Q level is kept, silently.

        Its quarter ends are read back onto the monthly index of X.
        """
        imputer = make_imputer(target_frequency='Q')
        transformed = fit_transform_quietly(imputer, reference_timeseries)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            restored = imputer.inverse_transform(transformed)

        assert not [item for item in caught if 'Falling back' in str(item.message)]
        pd.testing.assert_index_equal(restored.index, reference_timeseries.index)
        # q1, déjà trimestrielle, est ORIGINAL au niveau Q : ses observations reviennent
        pd.testing.assert_series_equal(
            restored['q1'], reference_timeseries['q1'], check_freq=False
        )

    def test_last_stacked_level_kept_with_one_warning(self, panel_reference_full):
        """Neither the source level (M) nor a composite target is stacked: last level, one warning.

        Targets ``{FR: Q, DE: Y, IT: Q}`` under axis 2 stack ``Y`` then ``Q``;
        the target label names three entities and is no stage label.
        """
        imputer = make_imputer(
            target_frequency={'FR': 'Q', 'DE': 'Y', 'IT': 'Q'},
            impute_intermediate_frequencies=True,
        )
        transformed = fit_transform_quietly(
            imputer, panel_reference_full[['m1', 'q1', 'a1', 'v']]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            restored = imputer.inverse_transform(transformed)

        fallbacks = [str(item.message) for item in caught if 'Falling back' in str(item.message)]
        assert len(fallbacks) == 1 and "'Q'" in fallbacks[0]
        assert restored.index.nlevels == 2

    def test_hand_cut_level_is_the_level_inverted(self, reference_timeseries):
        """A level cut out of the stacked output is inverted on its own rows, without noise.

        The level is chosen among those of the data to invert: the source
        level, present in the provenance only, would select nothing in them.
        The quarterly level falls back with its single warning and inverts
        like any level: ``q1`` observed at its quarter ends comes back, on
        the monthly index of X.
        """
        imputer = make_imputer(impute_intermediate_frequencies=True)
        transformed = fit_transform_quietly(imputer, reference_timeseries)
        quarterly = transformed.xs('Q', level='frequency', drop_level=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            restored = imputer.inverse_transform(quarterly)

        assert not [item for item in caught if issubclass(item.category, FutureWarning)]
        assert len([item for item in caught if 'Falling back' in str(item.message)]) == 1
        pd.testing.assert_index_equal(restored.index, reference_timeseries.index)
        pd.testing.assert_series_equal(
            restored['q1'], reference_timeseries['q1'], check_freq=False
        )

    def test_level_the_last_pass_never_produced_is_refused(self, reference_timeseries):
        """A stacked level unknown to the last pass cannot be inverted: an explicit error.

        No cell of such data lines up with the provenance; the inversion used
        to return an all-NaN frame and a pandas ``FutureWarning``.
        """
        imputer = make_imputer(impute_intermediate_frequencies=True)
        transformed = fit_transform_quietly(imputer, reference_timeseries)
        foreign = transformed.xs('Q', level='frequency', drop_level=False).rename(
            index={'Q': 'W'}, level='frequency'
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match="frequency level 'W'"):
                imputer.inverse_transform(foreign)
