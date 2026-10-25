"""State and input contract of ``HighFrequencyImputer.transform``.

Scenarios: transform state overwritten then purged, fitted state left intact, shared-
model replay, and what ``transform`` accepts beside a frame (``y``, additive
transformer, one row).
"""

# Modules de base
import copy
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

# Objets testés
from tsforecast.imputation._components.provenance import ProvenanceType, provenance_statistics
from tsforecast.imputation._components.training_set_builder import TrainingSetBuilder

# Outils de test partagés
from tests.support.estimators import SpyEstimator
from tests.unit.imputation.high_frequency_imputer.helpers import (
    assert_same_state,
    fit_quietly,
    fit_transform_quietly,
    fit_with_spy,
    has_frequency_level_in_output,
    make_imputer,
    public_fitted_state,
    transform_quietly,
    _warning_messages,
    _with_new_entity,
)


# Ajustement puis rejeu silencieux d'un couple (X, y)
def _fit_transform_pair(imputer, X: pd.DataFrame, y: pd.Series) -> tuple:
    """Fit then transform ``(X, y)``, swallowing the legitimate warnings.

    Args:
        imputer: Imputer to fit.
        X: Features.
        y: Target, merged as a column of its own name.

    Returns:
        The pair ``(X_transformed, y_transformed)``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return imputer.fit_transform(X, y)


class TestTransformState:
    """The transform state is overwritten by each pass, then purged."""

    def test_transform_state_purged_at_fit(self, reference_timeseries):
        """B19 — un fit efface l'état laissé par le transform précédent.

        Observé par l'API publique : après ``fit_transform(A)`` puis ``fit(B)``,
        l'inversion exacte restitue les valeurs de ``B`` — celles du dernier
        passage —, et la provenance est celle, à plat, du fit.
        """
        data_a = reference_timeseries
        data_b = reference_timeseries.copy()
        data_b['a1'] = data_b['a1'] * 2
        imputer = make_imputer(restore_original_values=True)
        fit_transform_quietly(imputer, data_a)
        assert has_frequency_level_in_output(imputer.imputation_provenance_)

        fit_quietly(imputer, data_b)
        # Le fit réécrit sa propre matrice, jamais celle d'un transform passé
        assert not has_frequency_level_in_output(imputer.imputation_provenance_)
        restored = imputer.inverse_transform(data_b)
        observed = data_b['a1'].notna()
        np.testing.assert_allclose(
            restored.loc[observed, 'a1'].to_numpy(), data_b.loc[observed, 'a1'].to_numpy()
        )

    def test_provenance_is_overwritten_by_each_transform(self, reference_timeseries):
        """The attribute carries the provenance of the LAST transform."""
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        after_fit = imputer.imputation_provenance_.copy()
        transform_quietly(imputer, reference_timeseries)
        after_transform = imputer.imputation_provenance_
        # La matrice du transform porte le niveau de fréquence, celle du fit non
        assert has_frequency_level_in_output(after_transform)
        assert not has_frequency_level_in_output(after_fit)

    @pytest.mark.parametrize('keep', [True, False], ids=['stacked', 'flat'])
    def test_provenance_statistics_describe_the_last_pass(self, reference_timeseries, keep):
        """``provenance_statistics_`` follows the last pass, counted on its target level.

        After a transform of 2021 alone, the statistics count its twelve rows,
        whatever ``keep_lower_frequencies`` stacks — a display parameter
        changes no figure.
        """
        imputer = make_imputer(impute_intermediate_frequencies=True, keep_lower_frequencies=keep)
        fit_quietly(imputer, reference_timeseries)
        fit_statistics = copy.deepcopy(imputer.provenance_statistics_)

        # Rejeu sur la seule année 2021 : une matrice trois fois plus courte
        transform_quietly(imputer, reference_timeseries.loc['2021'])
        marks = imputer.imputation_provenance_
        target = marks.xs('M', level='frequency') if keep else marks

        assert len(target) == 12
        assert_same_state(provenance_statistics(target), imputer.provenance_statistics_)
        assert fit_statistics['overall']['not_imputed'] + sum(
            fit_statistics['overall'][member.value] for member in ProvenanceType
        ) == 36 * 4

    def test_never_imputed_cells_stay_unmarked_after_transform(self, reference_timeseries):
        """A NaN cell no stage imputes stays unmarked after ``transform``, as after ``fit``.

        ``ORIGINAL`` marks the cells non-null in the input only;
        a NaN cell never imputed is unmarked (``isna``). The fit and its replay
        must give the same
        provenance from the fit and from its replay.
        """
        # Jeu TS dont les trois colonnes basses fréquences s'arrêtent en 2022 :
        # leurs cellules de 2023 sont hors fenêtre, donc jamais imputées
        data = reference_timeseries.copy()
        tail = data.index >= pd.Timestamp('2023-01-31')
        data.loc[tail, ['q1', 'a1', 'a2']] = np.nan

        imputer = fit_quietly(make_imputer(keep_lower_frequencies=False), data)
        assert imputer.imputation_provenance_.loc[tail, 'a1'].isna().all()

        transform_quietly(imputer, data)
        assert imputer.imputation_provenance_.loc[tail, 'a1'].isna().all()


    def test_fit_and_replay_provenances_agree_on_a_panel(self, panel_f_without_it_v):
        """On a panel, the fit matrix is the replay matrix, mute cells included.

        The 36 cells of ``v`` of ``IT``, never observed nor imputed, are
        unmarked on both sides.
        """
        imputer = fit_quietly(
            make_imputer(keep_lower_frequencies=False), panel_f_without_it_v
        )
        after_fit = imputer.imputation_provenance_.copy()
        transform_quietly(imputer, panel_f_without_it_v)
        after_transform = imputer.imputation_provenance_

        assert after_fit.loc['IT', 'v'].isna().all()
        pd.testing.assert_frame_equal(
            after_transform.astype(object).sort_index(),
            after_fit.astype(object).reindex(after_transform.index).sort_index(),
            check_freq=False,
        )


class TestTransformLeavesTheFitIntact:
    """The transform replays the frozen plan: it never fits, never mutates.

    Public counterpart of the former internal test
    ``TestTransformState::test_fit_state_is_restored_after_a_transform``
    (identity of two private collaborators): what matters is that no fitted
    attribute moves and that a transform depends on its own input only.
    """

    # Trois données de rejeu qui sollicitent chacune un chemin propre au transform
    @staticmethod
    def _replay_inputs(name: str, request) -> tuple:
        """Return ``(fit data, transform data, impute_unobserved_entities)``."""
        if name == 'timeseries_later_dates':
            # Fit sur 2021-2022, rejeu sur 2021-2023 : périodes nouvelles
            data = request.getfixturevalue('reference_timeseries')
            return data.loc[:'2022-12-31'], data, False
        if name == 'timeseries_diverging_frequency':
            # q1 publiée en décembre seulement : divergence de fréquence
            data = request.getfixturevalue('reference_timeseries')
            diverging = data.copy()
            diverging['q1'] = np.nan
            diverging.loc[diverging.index.month == 12, 'q1'] = [30.0, 90.0, 150.0]
            return data, diverging, False
        data = request.getfixturevalue('mixed_freq_panel_multifrequency')
        # Entité ES inconnue du fit, laissée intacte : cas hors du plan figé
        if name == 'panel_new_entity_untouched':
            return data, _with_new_entity(data), False
        # Entité ES inconnue du fit, imputée sans ancre : étapes dérivées
        return data, _with_new_entity(data), True

    @pytest.mark.slow
    def test_transform_never_refits(self, mixed_freq_panel_multifrequency):
        """No training set is built and no estimator is fitted during a transform.

        The richest replay: ``'model'`` cascade, intermediate stages, and an
        entity born at transform time served by derived steps. The
        public ``TrainingSetBuilder.build`` is patched at class level, so
        whatever path reaches it raises — and every call is recorded, so a
        caller swallowing the ``AssertionError`` still shows.
        """
        imputer = fit_with_spy(
            mixed_freq_panel_multifrequency,
            covariate_strategy='model', impute_intermediate_frequencies=True,
            impute_unobserved_entities=True,
        )
        fits_after_fit = SpyEstimator.n_fits
        calls = []

        def _forbidden(*args, **kwargs):
            calls.append(kwargs.get('column'))
            raise AssertionError('the training set must never be rebuilt at transform')

        with patch.object(TrainingSetBuilder, 'build', side_effect=_forbidden):
            transform_quietly(imputer, _with_new_entity(mixed_freq_panel_multifrequency))

        assert calls == []
        # Aucun estimateur ajusté pendant le rejeu : le compteur de classe reste
        assert SpyEstimator.n_fits == fits_after_fit

    @pytest.mark.parametrize(
        'scenario',
        ['timeseries_later_dates', 'timeseries_diverging_frequency', 'panel_new_entity'],
    )
    def test_transform_does_not_mutate_the_fitted_state(self, scenario, request):
        """Every public fitted attribute is, after a transform, what the fit left.

        A deep copy of the public ``_`` attributes is taken right after the
        fit and compared structurally after the transform (the fitted imputer
        copies). ``imputation_provenance_`` and
        ``provenance_statistics_`` alone are left out: they
        describe the LAST pass, rewritten by design.
        """
        fit_data, transform_data, unobserved = self._replay_inputs(scenario, request)
        imputer = fit_quietly(
            make_imputer(covariate_strategy='model', impute_unobserved_entities=unobserved),
            fit_data,
        )
        snapshot = copy.deepcopy(public_fitted_state(imputer))

        transform_quietly(imputer, transform_data)

        assert_same_state(snapshot, public_fitted_state(imputer))

    @pytest.mark.internal
    @pytest.mark.parametrize(
        'scenario',
        ['timeseries_later_dates', 'timeseries_diverging_frequency', 'panel_new_entity'],
    )
    def test_transform_leaves_the_fitted_components_intact(self, scenario, request):
        """The private fitted state is, after a transform, what the fit left.

        The transform reads the components the fit configured — aggregation
        constraint, stage scaler, materializer configuration — and the
        frequency layout, and never writes them: its materializer, registers
        and accumulators are its own.
        A deep copy taken right after the fit is compared structurally.
        """
        fit_data, transform_data, unobserved = self._replay_inputs(scenario, request)
        imputer = fit_quietly(
            make_imputer(covariate_strategy='model', impute_unobserved_entities=unobserved),
            fit_data,
        )
        names = (
            '_fitted_components_', '_frequency_layout_', '_source_index_frequency_label',
        )
        snapshot = copy.deepcopy({name: getattr(imputer, name) for name in names})

        transform_quietly(imputer, transform_data)

        assert_same_state(snapshot, {name: getattr(imputer, name) for name in names})

    @pytest.mark.parametrize(
        'scenario',
        [
            'timeseries_later_dates', 'timeseries_diverging_frequency',
            'panel_new_entity', 'panel_new_entity_untouched',
        ],
    )
    def test_two_successive_transforms_are_identical(self, scenario, request):
        """Two transforms of the same data agree on output, provenance, statistics and warnings.

        Each replay starts from an execution context of its own, built from the
        fitted state alone: nothing of the first one — registers, derived steps,
        pending warnings — reaches the second.
        """
        fit_data, transform_data, unobserved = self._replay_inputs(scenario, request)
        imputer = fit_quietly(
            make_imputer(covariate_strategy='model', impute_unobserved_entities=unobserved),
            fit_data,
        )

        passes = []
        for _ in range(2):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                output = imputer.transform(transform_data)
            passes.append((
                output,
                imputer.imputation_provenance_.copy(),
                copy.deepcopy(imputer.provenance_statistics_),
                [(item.category, str(item.message)) for item in caught],
            ))
        (output, provenance, statistics, emitted), (again, provenance_again,
                                                    statistics_again, emitted_again) = passes

        pd.testing.assert_frame_equal(again, output)
        pd.testing.assert_frame_equal(provenance_again, provenance)
        assert_same_state(statistics, statistics_again)
        assert emitted_again == emitted

    @pytest.mark.slow
    def test_transform_depends_on_its_own_input_only(self, mixed_freq_panel_multifrequency):
        """A transform of A gives the same result before and after a transform of B.

        B carries an entity unknown to the fit: its replay installs
        derived steps and extended bindings, none of which may leak into the
        next call.
        """
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True), mixed_freq_panel_multifrequency
        )
        first = transform_quietly(imputer, mixed_freq_panel_multifrequency)
        first_provenance = imputer.imputation_provenance_.copy()

        transform_quietly(imputer, _with_new_entity(mixed_freq_panel_multifrequency))
        again = transform_quietly(imputer, mixed_freq_panel_multifrequency)

        pd.testing.assert_frame_equal(again, first)
        pd.testing.assert_frame_equal(imputer.imputation_provenance_, first_provenance)


class TestSharedModelReplay:
    """A shared model is replayed by each of its stages."""

    @pytest.mark.slow
    def test_shared_model_replayed_by_each_step(
        self, mixed_freq_panel_multifrequency
    ):
        """Les deux étapes de 'v' rejouent le MÊME objet modèle, sur leurs entités."""
        imputer = fit_with_spy(mixed_freq_panel_multifrequency)
        monthly = [
            step for step in imputer.imputation_plan_
            if step.var_name == 'v' and step.pred_freq_label == 'M'
        ]
        assert len(monthly) == 2

        # Un seul objet modèle, partagé
        assert monthly[0].model is monthly[1].model
        # Des entités et des recalages distincts
        assert set(monthly[0].entities) != set(monthly[1].entities)
        assert monthly[0].source_frequency != monthly[1].source_frequency

        # Chaque étape rejoue le modèle : autant d'appels de prédiction que
        # d'étapes, au fit comme au transform
        model = monthly[0].model
        fit_calls = len(model.predict_X_)
        transform_quietly(imputer, mixed_freq_panel_multifrequency)
        assert len(model.predict_X_) == 2 * fit_calls

        # Et la symétrie tient malgré le partage
        combined = fit_transform_quietly(
            make_imputer(estimator=SpyEstimator()), mixed_freq_panel_multifrequency
        )
        separate = make_imputer(estimator=SpyEstimator())
        fit_quietly(separate, mixed_freq_panel_multifrequency)
        pd.testing.assert_frame_equal(
            combined, transform_quietly(separate, mixed_freq_panel_multifrequency)
        )


# =============================================================================
# Contrat d'entrée du transform : y, transformateur additif, entrée minimale
# =============================================================================
# Doublure : transformateur additif logarithmique dont l'inversion échoue
class _NonInvertibleLog(BaseEstimator, TransformerMixin):
    """Additive ``log1p`` transformer whose ``inverse_transform`` raises."""

    def fit(self, X, y=None):
        """Learn nothing.

        Args:
            X: Data.
            y: Ignored.

        Returns:
            self.
        """
        return self

    def transform(self, X):
        """Return ``log1p(X)``.

        Args:
            X: Data.

        Returns:
            The transformed frame.
        """
        return np.log1p(X)

    def inverse_transform(self, X):
        """Refuse to invert.

        Args:
            X: Data.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError('not invertible')


class TestTransformInputContract:
    """What ``transform`` accepts beside a frame: ``y``, an additive transformer, one row."""

    def test_series_is_refused(self, reference_timeseries):
        """A Series passes the base validation but is refused by the replay itself."""
        imputer = fit_quietly(make_imputer(), reference_timeseries)
        with pytest.raises(ValueError, match='X must be a pandas DataFrame, got Series'):
            imputer.transform(reference_timeseries['m1'])

    def test_one_row_falls_back_on_every_row_with_one_warning(self, reference_timeseries):
        """One row carries no window: a single warning, every row used, the row kept."""
        imputer = fit_quietly(make_imputer(keep_lower_frequencies=False), reference_timeseries)
        one_row = reference_timeseries.iloc[[5]]
        captured = {}

        messages = _warning_messages(
            lambda: captured.setdefault('result', imputer.transform(one_row))
        )

        assert len([m for m in messages if 'Could not calculate the imputation window' in m]) == 1
        result = captured['result']
        assert result.index.equals(one_row.index)
        assert result['m1'].iloc[0] == one_row['m1'].iloc[0]

    def test_truncated_period_is_reported_and_restorable(self, reference_timeseries):
        """One month of a quarter cannot carry its total: warned once, restorable.

        The row 2021-06-30 holds ``q1 = 20``, the total of a quarter the
        call truncates to one month. It is not rescaled: the month gets
        a third of it, and the quarterly total cannot be read in the
        output — which one warning says. ``restore_original_values`` brings
        the observation back.
        """
        imputer = fit_quietly(
            make_imputer(keep_lower_frequencies=False, restore_original_values=True),
            reference_timeseries,
        )
        one_row = reference_timeseries.loc[['2021-06-30']]
        captured = {}

        messages = _warning_messages(
            lambda: captured.setdefault('result', imputer.transform(one_row))
        )

        truncated = [m for m in messages if m.startswith('Aggregation constraint not enforced')]
        assert len(truncated) == 1 and "['q1']" in truncated[0]
        assert captured['result']['q1'].iloc[0] == pytest.approx(20.0 / 3)
        restored = imputer.inverse_transform(captured['result'])
        assert restored['q1'].iloc[0] == 20.0

    def test_unnamed_target_comes_back_unnamed(self, reference_timeseries):
        """An unnamed ``y`` keeps no name through ``transform`` and ``inverse_transform``."""
        X = reference_timeseries[['m1', 'q1', 'a2']]
        y = reference_timeseries['a1'].rename(None)
        imputer = make_imputer(keep_lower_frequencies=False)

        X_out, y_out = _fit_transform_pair(imputer, X, y)
        _X_back, y_back = imputer.inverse_transform(X_out, y_out)

        assert y_out.name is None and y_back.name is None
        assert y_out.loc['2021'].sum() == pytest.approx(120.0)

    def test_transform_with_y_returns_the_named_pair(self, reference_timeseries):
        """B14 — ``(X, y)`` comes back as a pair, ``y`` under its own name, imputed like a column.

        Gold value: the twelve months of ``a1`` in 2021 sum to 120.
        """
        X = reference_timeseries[['m1', 'q1', 'a2']]
        y = reference_timeseries['a1']
        imputer = make_imputer(keep_lower_frequencies=False)

        X_out, y_out = _fit_transform_pair(imputer, X, y)

        assert list(X_out.columns) == ['m1', 'q1', 'a2']
        assert y_out.name == 'a1'
        assert y_out.loc['2021'].sum() == pytest.approx(120.0)

    def test_plain_additive_transformer_is_applied_then_inverted(self, reference_timeseries):
        """An additive transformer returning a frame: imputed in its space, inverted back.

        The output lives in the additive space (``log1p``); the inversion
        applies ``expm1`` last, so the fully observed ``m1`` comes back exact.
        """
        imputer = make_imputer(
            additive_transformer=FunctionTransformer(np.log1p, np.expm1, check_inverse=False),
            keep_lower_frequencies=False,
        )
        transformed = fit_transform_quietly(imputer, reference_timeseries)

        np.testing.assert_allclose(
            transformed['m1'].to_numpy(), np.log1p(reference_timeseries['m1'].to_numpy())
        )
        restored = imputer.inverse_transform(transformed)
        np.testing.assert_allclose(
            restored['m1'].to_numpy(), reference_timeseries['m1'].to_numpy()
        )

    def test_failing_additive_inversion_warns_and_returns_the_additive_space(
        self, reference_timeseries
    ):
        """An additive inversion that raises is reported once; the data stay in its space."""
        imputer = make_imputer(
            additive_transformer=_NonInvertibleLog(), keep_lower_frequencies=False
        )
        transformed = fit_transform_quietly(imputer, reference_timeseries)
        captured = {}

        messages = _warning_messages(
            lambda: captured.setdefault('restored', imputer.inverse_transform(transformed))
        )

        assert len([m for m in messages if 'Failed to inverse transform' in m]) == 1
        np.testing.assert_allclose(
            captured['restored']['m1'].to_numpy(),
            np.log1p(reference_timeseries['m1'].to_numpy()),
        )
