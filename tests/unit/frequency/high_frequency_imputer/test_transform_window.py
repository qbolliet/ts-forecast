"""``HighFrequencyImputer.transform`` against the fit window.

Scenarios: transform outside the fit window, extended window on an irregular index,
dates after the fit, a column without anchor at transform and the reserved
``'frequency'`` level name.
"""

# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest

# Objets testés
from tsforecast.frequency.provenance import ProvenanceType

# Outils de test partagés
from tests.unit.frequency.high_frequency_imputer.helpers import (
    fit_quietly,
    fit_transform_quietly,
    make_imputer,
    transform_quietly,
    _warning_messages,
)


class TestTransformOutsideWindow:
    """A transform outside the fit window imputes instead of emptying."""

    # Fixture privée d'un jeu dont la fin sort de la fenêtre stricte
    @staticmethod
    def _truncated(data: pd.DataFrame) -> pd.DataFrame:
        """Jeu TS dont les trois colonnes basses fréquences s'arrêtent en 2022."""
        frame = data.copy()
        tail = frame.index >= pd.Timestamp('2023-01-31')
        frame.loc[tail, ['q1', 'a1', 'a2']] = np.nan
        return frame

    def test_transform_outside_fit_window(self, reference_timeseries):
        """Imputes, destroys no observation, warns ONCE."""
        data = self._truncated(reference_timeseries)
        imputer = fit_quietly(make_imputer(), data)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = imputer.transform(data)

        # Un seul avertissement de fenêtre, nommant le nombre de lignes
        window_warnings = [
            str(w.message) for w in caught if 'outside' in str(w.message)
        ]
        assert len(window_warnings) == 1
        assert 'row(s)' in window_warnings[0]

        # Aucune ligne d'entrée perdue, aucune observation détruite
        flat = result.droplevel('frequency')
        assert len(flat) == len(data)
        observed = data['m1'].notna()
        assert np.allclose(flat['m1'].to_numpy(), data['m1'].to_numpy())
        assert observed.all()

        # Les lignes hors fenêtre gardent leurs valeurs d'entrée : la colonne
        # n'est jamais vidée sans être réécrite
        tail = flat.index >= pd.Timestamp('2023-01-31')
        assert flat.loc[tail, 'm1'].notna().all()

    def test_input_observations_survive_every_column(self, reference_timeseries):
        """Aucune cellule observée de m1 n'est jamais perdue au transform."""
        data = self._truncated(reference_timeseries)
        imputer = fit_quietly(make_imputer(), data)
        flat = transform_quietly(imputer, data).droplevel('frequency')
        assert flat['m1'].notna().sum() == data['m1'].notna().sum()


    def test_transform_outside_fit_window_on_panel(self, mixed_freq_panel_heterogeneous):
        """On a panel, imputes, destroys no observation, warns ONCE."""
        data = mixed_freq_panel_heterogeneous.copy()
        tail = data.index.get_level_values('date') >= pd.Timestamp('2023-01-31')
        data.loc[tail, ['q1', 'a1', 'a2']] = np.nan
        imputer = fit_quietly(make_imputer(), data)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = imputer.transform(data)
        window_warnings = [
            str(item.message) for item in caught if 'outside' in str(item.message)
        ]
        assert len(window_warnings) == 1

        # Aucune ligne d'entrée perdue, aucune observation à la fréquence cible
        # détruite ; les ancres basses fréquences sont, elles, ré-exprimées
        flat = result.droplevel('frequency').reindex(data.index)
        assert len(result.droplevel('frequency')) == len(data)
        for column in ['m1', 'climat_affaires']:
            observed = data[column].notna()
            np.testing.assert_allclose(
                flat.loc[observed, column].to_numpy(), data.loc[observed, column].to_numpy()
            )


class TestIrregularIndexExtendedWindow:
    """Fenêtre étendue vers des périodes où la grille cible n'existe pas dans l'index.

    Observations annuelles isolées AVANT le début de la grille mensuelle : sous
    un seuil nul, la fenêtre y remonte et la grille densifiée porte des mois
    absents de l'entrée. La matrice de provenance, initialisée sur l'index
    d'entrée, levait une KeyError à l'écriture.
    """

    @staticmethod
    def _irregular_timeseries() -> pd.DataFrame:
        """Mensuelles dès 2018, trimestrielle, annuelle dès 2015 (index irrégulier)."""
        rng = np.random.default_rng(0)
        monthly = pd.date_range('2018-01-01', '2021-12-01', freq='MS')
        frame = pd.DataFrame(index=monthly)
        frame['m1'] = 100 + rng.normal(0, 1, len(monthly)).cumsum()
        frame['m2'] = 50 + rng.normal(0, 1, len(monthly))
        frame['q1'] = np.where(monthly.month.isin([1, 4, 7, 10]),
                               300 + rng.normal(0, 5, len(monthly)), np.nan)
        annual = pd.date_range('2015-01-01', '2021-01-01', freq='YS')
        frame = frame.reindex(frame.index.union(annual))
        frame['a1'] = np.nan
        frame.loc[annual, 'a1'] = 1200 + rng.normal(0, 20, len(annual))
        frame.index.name = 'date'
        return frame

    @pytest.mark.parametrize('scope', ['extended_backward', 'extended_both'])
    @pytest.mark.parametrize('strategy', ['interpolate', 'model'])
    def test_fit_transform_on_densified_grid(self, scope, strategy):
        """Le fit aboutit ; la provenance couvre les mois absents de l'entrée."""
        data = self._irregular_timeseries()
        imputer = make_imputer(
            imputation_scope=scope, coverage_threshold=0.0,
            covariate_strategy=strategy, keep_lower_frequencies=False,
        )
        imputer = fit_quietly(imputer, data)

        # La fenêtre remonte bien jusqu'à la première observation annuelle
        assert imputer.imputation_window_[0] == pd.Timestamp('2015-01-01')

        # Les mois de 2015 absents de l'entrée portent une provenance imputée
        matrix = imputer.imputation_provenance_
        added = pd.date_range('2015-02-01', '2015-12-01', freq='MS')
        assert added.isin(matrix.index).all()
        assert matrix.loc[added, 'a1'].notna().all()
        # Aucune ligne créée ne se déclare ORIGINAL
        assert not (matrix.loc[added] == ProvenanceType.ORIGINAL).any().any()
        assert matrix.index.is_monotonic_increasing

        # Le transform rejoue sans erreur et produit ces mois
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = imputer.transform(data)
        assert result.loc[added, 'a1'].notna().all()

    def test_additivity_holds_on_the_isolated_years(self):
        """Sous 'sum', les douze mois d'une année isolée redonnent son total."""
        data = self._irregular_timeseries()
        imputer = fit_quietly(make_imputer(
            imputation_scope='extended_backward', coverage_threshold=0.0,
            keep_lower_frequencies=False,
        ), data)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = imputer.transform(data)
        for year in (2015, 2016, 2017):
            months = result.loc[f'{year}-01-01':f'{year}-12-01', 'a1']
            assert len(months) == 12
            assert months.sum() == pytest.approx(data.loc[f'{year}-01-01', 'a1'])


class TestTransformOnLaterDates:
    """A transform on dates after the fit imputes the new periods only.

    The fit sees 2021-2022, the transform 2021-2023: the 2023 anchors are new
    totals the frozen models disaggregate, while the cells of the fit period
    are recomputed identically — same models, same covariates, same totals.
    """

    @pytest.mark.parametrize('strategy', ['interpolate', 'model'])
    def test_new_periods_are_imputed_and_add_up(self, reference_timeseries, strategy):
        """The twelve months of 2023 are imputed and sum to the new annual anchors."""
        imputer = fit_quietly(
            make_imputer(covariate_strategy=strategy, keep_lower_frequencies=False),
            reference_timeseries.loc[:'2022-12-31'],
        )
        result = transform_quietly(imputer, reference_timeseries)
        provenance = imputer.imputation_provenance_.loc['2023']

        # Cellules 2023 de a1 et a2 toutes produites par un modèle
        for column in ('a1', 'a2'):
            assert provenance[column].map(lambda mark: mark.startswith('model_')).all()
        # Valeurs d'or : totaux 2023 de 150 (a1) et 72 (a2)
        assert result.loc['2023', 'a1'].sum() == pytest.approx(150.0)
        assert result.loc['2023', 'a2'].sum() == pytest.approx(72.0)

    @pytest.mark.parametrize('strategy', ['interpolate', 'model'])
    def test_fit_period_is_unchanged(self, reference_timeseries, strategy):
        """Values and provenances of 2021-2022 are those of the transform of the fit data."""
        fit_data = reference_timeseries.loc[:'2022-12-31']
        imputer = fit_quietly(
            make_imputer(covariate_strategy=strategy, keep_lower_frequencies=False), fit_data
        )
        on_fit_data = transform_quietly(imputer, fit_data)
        provenance_on_fit_data = imputer.imputation_provenance_.copy()

        extended = transform_quietly(imputer, reference_timeseries)

        pd.testing.assert_frame_equal(
            extended.loc[:'2022-12-31'], on_fit_data, check_freq=False
        )
        pd.testing.assert_frame_equal(
            imputer.imputation_provenance_.loc[:'2022-12-31'], provenance_on_fit_data,
            check_freq=False,
        )

    def test_new_periods_on_a_panel(self, mixed_freq_panel_multifrequency):
        """On PANEL-F, each 2023 period of ``v`` adds up to its new observation, per entity.

        ``FR`` publishes ``v`` once a year (gold value 150.0), ``DE``
        once a quarter; ``IT`` observes it monthly and is never rewritten.
        """
        dates = mixed_freq_panel_multifrequency.index.get_level_values('date')
        imputer = fit_quietly(
            make_imputer(keep_lower_frequencies=False),
            mixed_freq_panel_multifrequency[dates <= '2022-12-31'],
        )
        result = transform_quietly(imputer, mixed_freq_panel_multifrequency)
        source = mixed_freq_panel_multifrequency['v']

        # FR : total annuel 2023 de 150, valeur d'or
        assert result.loc['FR', 'v'].loc['2023'].sum() == pytest.approx(150.0)
        # DE : chaque trimestre 2023 somme à son observation trimestrielle
        de_months = result.loc['DE', 'v'].loc['2023']
        de_quarters = source.loc['DE'].loc['2023'].dropna()
        np.testing.assert_allclose(
            de_months.groupby(de_months.index.to_period('Q')).sum().to_numpy(),
            de_quarters.to_numpy(),
        )
        # IT : observations mensuelles intactes (valeurs et dates ; la fréquence portée par
        # l'index d'une sélection de MultiIndex n'est pas l'objet du test)
        pd.testing.assert_series_equal(result.loc['IT', 'v'], source.loc['IT'], check_freq=False)


class TestColumnWithoutAnchorAtTransform:
    """A column observed at fit keeps its replayed, anchored step.

    ``a1`` is unpublished for 2023. Transformed together with its history,
    the 2023 rows fall outside the recomputed strict window and stay empty.
    Alone, the same rows see ``a1`` without any observation: the column
    leaves the coverage denominator (with a warning), the window spans them, and
    the frozen step predicts every month without rescaling — the very cells
    the extended window produces with the history. The pair being
    observed at fit, its classification is replayed: ``MODEL_*``, never
    ``MODEL_UNANCHORED``, whatever ``impute_unobserved_entities`` says
    """

    # Jeu TS dont l'année 2023 de a1 n'est pas publiée
    @staticmethod
    def _unpublished_2023(data: pd.DataFrame) -> pd.DataFrame:
        """Return TS with ``a1`` blanked over 2023."""
        frame = data.copy()
        frame.loc['2023', 'a1'] = np.nan
        return frame

    def test_unpublished_year_stays_empty_with_its_history(self, reference_timeseries):
        """With its history, the unpublished year stays NaN and unmarked."""
        imputer = fit_quietly(make_imputer(keep_lower_frequencies=False), reference_timeseries)
        result = transform_quietly(imputer, self._unpublished_2023(reference_timeseries))

        assert result.loc['2023', 'a1'].isna().all()
        assert imputer.imputation_provenance_.loc['2023', 'a1'].isna().all()

    @pytest.mark.parametrize('unobserved', [False, True])
    def test_unpublished_year_alone_is_predicted_by_its_replayed_step(
        self, reference_timeseries, unobserved
    ):
        """Alone, the twelve months are predicted, ``MODEL_*``, with a warning naming ``a1``."""
        imputer = fit_quietly(
            make_imputer(keep_lower_frequencies=False, impute_unobserved_entities=unobserved),
            reference_timeseries,
        )
        alone = self._unpublished_2023(reference_timeseries).loc['2023']
        captured = {}

        messages = _warning_messages(
            lambda: captured.setdefault('result', imputer.transform(alone))
        )

        assert captured['result']['a1'].notna().sum() == 12
        marks = set(imputer.imputation_provenance_['a1'])
        assert len(marks) == 1 and marks.pop().startswith('model_on_')
        assert any('a1' in m and 'without any observation' in m for m in messages)

    def test_extended_window_predicts_it_with_its_history(self, reference_timeseries):
        """Under ``'extended_forward'``, the history does not hide the year."""
        imputer = fit_quietly(
            make_imputer(keep_lower_frequencies=False, imputation_scope='extended_forward'),
            reference_timeseries,
        )

        result = transform_quietly(imputer, self._unpublished_2023(reference_timeseries))

        assert result.loc['2023', 'a1'].notna().all()
        marks = set(imputer.imputation_provenance_.loc['2023', 'a1'])
        assert len(marks) == 1 and marks.pop().startswith('model_on_')


# =============================================================================
# Nom réservé du niveau de fréquence et lecture positionnelle
# =============================================================================
class TestReservedFrequencyLevel:
    """The ``'frequency'`` level name is reserved, the level read by position."""

    @pytest.mark.parametrize(
        'names', [['frequency', 'date'], ['country', 'frequency']],
        ids=['entity_level', 'date_level'],
    )
    def test_fit_rejects_an_index_level_named_frequency(
        self, mixed_freq_panel_multifrequency, names
    ):
        """A stacked output would hold two levels of one name: ``fit`` refuses."""
        data = mixed_freq_panel_multifrequency.rename_axis(names)

        with pytest.raises(ValueError, match="'frequency' is reserved"):
            make_imputer().fit(data)

    def test_transform_rejects_an_index_level_named_frequency(
        self, mixed_freq_panel_multifrequency
    ):
        """Same guard at ``transform``, on data the fit never saw."""
        imputer = fit_quietly(make_imputer(), mixed_freq_panel_multifrequency)
        data = mixed_freq_panel_multifrequency.rename_axis(['frequency', 'date'])

        with pytest.raises(ValueError, match="'frequency' is reserved"):
            imputer.transform(data)

    def test_guard_does_not_depend_on_the_configuration(self, reference_timeseries):
        """``keep_lower_frequencies=False`` stacks nothing, yet the name stays reserved."""
        data = reference_timeseries.rename_axis('frequency')

        with pytest.raises(ValueError, match="'frequency' is reserved"):
            make_imputer(keep_lower_frequencies=False).fit(data)

    def test_inverse_reads_the_level_by_position(self, reference_timeseries):
        """Renamed output levels invert exactly like the original names."""
        imputer = make_imputer(
            impute_intermediate_frequencies=True, restore_original_values=True
        )
        transformed = fit_transform_quietly(imputer, reference_timeseries)
        reference = imputer.inverse_transform(transformed)

        # Renommage des niveaux de la sortie : une lecture par nom échouerait
        renamed = transformed.rename_axis(['pas', 'jour'])
        restored = imputer.inverse_transform(renamed)

        pd.testing.assert_frame_equal(restored, reference)
        assert restored.index.equals(reference_timeseries.index)

    def test_unstacked_output_is_left_as_is(self, reference_timeseries):
        """Under ``keep_lower_frequencies=False`` no level is ever dropped."""
        imputer = make_imputer(keep_lower_frequencies=False)
        transformed = fit_transform_quietly(imputer, reference_timeseries)

        restored = imputer.inverse_transform(transformed)

        assert restored.index.equals(transformed.index)
