"""Axis 2 of ``HighFrequencyImputer``: intermediate frequencies.

Covers the frequency progression, the stage plan, the origin filter of
``y_train``, coincident cells and the per-row scale.
"""

# Modules de base
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Objets testés
from tsforecast.frequency.high_frequency_imputer import ELIGIBLE_ORIGINS
from tsforecast.frequency.provenance import ProvenanceType

# Outils de test partagés
from tests.support.estimators import RecordingEstimator, SpyEstimator
from tests.unit.frequency.high_frequency_imputer.helpers import (
    FR,
    DE,
    IT,
    capture_fits,
    capture_variable_fits,
    fit_quietly,
    fit_with_spy,
    imputed_cells,
    make_imputer,
    plan_groups,
    plan_pairs,
    replay,
    transform_quietly,
)
from tests.unit.frequency.training_set_builder.helpers import (
    TS_FREQUENCIES,
    make_builder,
)


class TestFrequencyProgression:
    """Construction de la progression, identique au fit et au transform."""

    @pytest.mark.slow
    def test_frequency_progression_on_reference_ts(self, reference_timeseries):
        """Les trois modalités sur le jeu TS, valeurs exactes."""
        # F = {Q, Y, M} : sous False, la cible seule
        imputer = fit_with_spy(reference_timeseries, impute_intermediate_frequencies=False)
        assert imputer.frequency_progression_ == ['M']

        # Sous les deux autres modalités, LE MÊME plan : Y, la plus basse
        # fréquence, n'est pas une étape — rien n'est à y imputer
        for modality in ('covariates_only', True):
            imputer = fit_with_spy(
                reference_timeseries, impute_intermediate_frequencies=modality
            )
            assert imputer.frequency_progression_ == ['Q', 'M']

    def test_imputable_variables_of_each_stage_on_reference_ts(self, reference_timeseries):
        """Étape Q : {a1, a2} ; étape M : {q1, a1, a2}."""
        imputer = fit_with_spy(
            reference_timeseries, impute_intermediate_frequencies='covariates_only'
        )
        # Groupes imputables lus sur le plan figé : une étape par groupe
        assert {column for column, _ in plan_groups(imputer, 'Q')} == {'a1', 'a2'}
        assert {column for column, _ in plan_groups(imputer, 'M')} == {'q1', 'a1', 'a2'}

    @pytest.mark.slow
    def test_progression_on_panel_f_uses_per_entity_frequencies(
        self, mixed_freq_panel_multifrequency
    ):
        """Sur PANEL-F, F est lu par couple (entité, colonne)."""
        imputer = fit_with_spy(
            mixed_freq_panel_multifrequency,
            impute_intermediate_frequencies='covariates_only',
        )
        # v est annuelle pour FR, trimestrielle pour DE, mensuelle pour IT :
        # une lecture par colonne n'aurait vu qu'une fréquence et manqué Q
        # Chaque étape lie toutes ses entités à une seule fréquence
        assert [
            set(stage.values()) for stage in imputer.frequency_progression_
        ] == [{'Q'}, {'M'}]

        # Étape Q : v n'est imputable que pour FR
        at_quarter = plan_groups(imputer, 'Q')
        assert at_quarter[('v', 'Y')] == (FR,)
        assert ('v', 'Q') not in at_quarter
        # Étape M : v est imputable pour FR et DE, jamais pour IT
        at_month = plan_groups(imputer, 'M')
        assert at_month[('v', 'Y')] == (FR,)
        assert at_month[('v', 'Q')] == (DE,)
        # IT observe déjà v mensuellement : elle n'est imputable à aucune étape
        for (column, _source), entities in at_month.items():
            if column == 'v':
                assert IT not in entities

    def test_progression_per_target_frequency_group_on_panel(
        self, mixed_freq_panel_multifrequency
    ):
        """Un dict de cibles produit une progression par groupe, fusionnée."""
        imputer = fit_with_spy(
            mixed_freq_panel_multifrequency,
            target_frequency={FR: 'M', DE: 'Q', IT: 'M'},
            impute_intermediate_frequencies=True,
        )
        # DE a Q pour cible : son groupe s'arrête à l'étape Q, dont il est absent
        assert imputer.frequency_progression_ == [
            {FR: 'Q', DE: 'Q', IT: 'Q'},
            {FR: 'M', IT: 'M'},
        ]
        # Aucune étape mensuelle n'écrit pour DE
        for step in imputer.imputation_plan_.steps:
            if step.pred_freq_label == 'M':
                assert DE not in (step.entities or ())

    def test_entity_stopping_early_still_feeds_the_finer_stage(
        self, mixed_freq_panel_multifrequency
    ):
        """DE, cible Q, contribue au ``y_train`` de l'étape M sans y être écrite."""
        data = mixed_freq_panel_multifrequency
        imputer, fits = capture_variable_fits(
            data,
            target_frequency={FR: 'M', DE: 'Q', IT: 'M'},
            impute_intermediate_frequencies=True,
        )
        contributors = {key[0] for key in fits[('M', 'v')].y_train.index}
        assert 'DE' in contributors

        # Aucune cellule de DE n'est écrite à la fréquence M : le niveau M de la
        # sortie lui reporte ses valeurs trimestrielles, sans aucune imputation
        output = transform_quietly(imputer, data)
        germany_at_m = output.xs((DE[0], 'M'), level=('country', 'frequency'))['v']
        assert set(germany_at_m.dropna().index.month) <= {3, 6, 9, 12}
        marks = imputer.imputation_provenance_.xs(
            (DE[0], 'M'), level=('country', 'frequency')
        )['v']
        assert set(marks.dropna().astype(str)) == {'original'}


class TestStagePlanAxis2:
    """Le plan d'étapes complet sur le jeu TS."""

    @pytest.mark.slow
    def test_stage_plan_of_spec_5_5(self, reference_timeseries):
        """2 étapes, 5 modèles, y_train filtré par l'origine."""
        expected_pairs = [
            ('Q', 'a1'), ('Q', 'a2'), ('M', 'a1'), ('M', 'a2'), ('M', 'q1'),
        ]
        # Sous 'covariates_only', y_train tient les 3 ancres partout : le
        # filtre est celui de False, seul le plan diffère
        imputer, fits = capture_variable_fits(
            reference_timeseries,
            covariate_strategy='model',
            fit_predict_order='frequency',
            impute_intermediate_frequencies='covariates_only',
        )
        assert imputer.frequency_progression_ == ['Q', 'M']
        assert plan_pairs(imputer) == expected_pairs
        assert SpyEstimator.n_fits == 5
        for stage in ('Q', 'M'):
            for column in ('a1', 'a2'):
                assert len(fits[(stage, column)].y_train) == 3
        assert len(fits[('M', 'q1')].y_train) == 12

        # Sous True, les étapes M de a1 et a2 gagnent leurs imputations Q.
        # Douze lignes et non quinze : la quatrième imputation trimestrielle
        # de chaque année tombe sur l'ancre annuelle, où l'observation gagne
        imputer, fits = capture_variable_fits(
            reference_timeseries,
            covariate_strategy='model',
            fit_predict_order='frequency',
            impute_intermediate_frequencies=True,
        )
        assert plan_pairs(imputer) == expected_pairs
        assert SpyEstimator.n_fits == 5
        for column in ('a1', 'a2'):
            assert len(fits[('Q', column)].y_train) == 3
            training = fits[('M', column)].training
            assert len(fits[('M', column)].y_train) == 12
            assert sorted(training.row_origin.value_counts().to_dict().items()) == [
                ('model', 9), ('observed', 3)
            ]

    def test_carried_model_rank_reached_under_covariates_only(self, reference_timeseries):
        """Le rang 3 — report d'étape — devient atteignable."""
        imputer = fit_with_spy(
            reference_timeseries,
            covariate_strategy='model',
            impute_intermediate_frequencies='covariates_only',
        )
        ways = [
            way
            for step in imputer.imputation_plan_.steps
            for way in step.materialization.values()
        ]
        assert 'carried_model' in ways
        # Sous False, aucune étape antérieure : le rang 3 reste hors d'atteinte
        without = fit_with_spy(
            reference_timeseries,
            covariate_strategy='model',
            impute_intermediate_frequencies=False,
        )
        assert 'carried_model' not in [
            way
            for step in without.imputation_plan_.steps
            for way in step.materialization.values()
        ]


    def test_x_train_is_full_on_the_quarterly_rows(self, reference_timeseries):
        """Sous True, à l'étape M de ``a1``, ``X_train`` est plein sur les lignes Q.

        ``y_train`` porte des lignes Y et Q, ``target_taint`` vaut
        ``'imputed'``, et ``m1`` vaut sur chaque ligne trimestrielle la somme du
        trimestre — ramenée à l'échelle de l'étape (diviseur 3) —, jamais celle
        de l'année, jamais NaN.
        """
        imputer, fits = capture_variable_fits(
            reference_timeseries, impute_intermediate_frequencies=True
        )
        fit = fits[('M', 'a1')]
        assert set(fit.training.row_frequency.unique()) == {'Y', 'Q'}
        step = next(
            step for step in imputer.imputation_plan_
            if (step.pred_freq_label, step.var_name) == ('M', 'a1')
        )
        assert step.target_taint == 'imputed'

        x_train = step.model.fit_X_
        assert x_train.notna().all().all()
        quarterly = fit.training.row_frequency[fit.training.row_frequency == 'Q'].index
        quarter_sums = reference_timeseries['m1'].resample('QE').sum()
        for date in quarterly:
            assert x_train.loc[date, 'm1'] == pytest.approx(quarter_sums.loc[date] / 3)


class TestOriginFilter:
    """Le filtre d'origine de y_train, et son piège."""

    @pytest.mark.slow
    def test_covariates_only_differs_from_true_on_panel(self, mixed_freq_panel_heterogeneous):
        """Sur panel, filtre 'observed' seul, valeurs finales distinctes de True."""
        data = mixed_freq_panel_heterogeneous
        _, covariates_only = capture_variable_fits(
            data, covariate_strategy='model',
            impute_intermediate_frequencies='covariates_only',
        )
        for fit in covariates_only.values():
            assert set(fit.training.row_origin.unique()) <= {'observed'}
        _, cascaded = capture_variable_fits(
            data, covariate_strategy='model', impute_intermediate_frequencies=True,
        )
        assert any(
            'model' in set(fit.training.row_origin.unique()) for fit in cascaded.values()
        )

        # La cascade change y_train : lignes 'model' en plus sous True
        assert len(cascaded[('M', 'a1')].y_train) != len(
            covariates_only[('M', 'a1')].y_train
        )

        # Valeurs finales lues au niveau cible de la sortie publique
        def _final(modality, strategy):
            imputer = fit_quietly(
                make_imputer(
                    covariate_strategy=strategy,
                    impute_intermediate_frequencies=modality,
                ),
                data,
            )
            values, _ = replay(imputer, data)
            return values[['q1', 'a1', 'a2']]

        # Sous 'interpolate', 'covariates_only' rend les valeurs de False
        pd.testing.assert_frame_equal(
            _final('covariates_only', 'interpolate'), _final(False, 'interpolate')
        )

    @pytest.mark.slow
    def test_covariates_only_differs_from_true(self, reference_timeseries):
        """Même plan, filtre différent, valeurs différentes."""
        # Sous 'model', aucune ligne d'origine 'model' n'entre dans y_train
        _, covariates_only = capture_variable_fits(
            reference_timeseries,
            covariate_strategy='model',
            impute_intermediate_frequencies='covariates_only',
        )
        for fit in covariates_only.values():
            assert set(fit.training.row_origin.unique()) <= {'observed'}

        # ... et les valeurs finales diffèrent de celles de True
        _, cascaded = capture_variable_fits(
            reference_timeseries,
            covariate_strategy='model',
            impute_intermediate_frequencies=True,
        )
        assert 'model' in set(cascaded[('M', 'a1')].training.row_origin.unique())
        assert not np.allclose(
            covariates_only[('M', 'a1')].y_train.to_numpy()[:3],
            cascaded[('M', 'a1')].y_train.to_numpy()[:3],
        ) or len(cascaded[('M', 'a1')].y_train) != len(
            covariates_only[('M', 'a1')].y_train
        )

        # Sous 'interpolate', les rangs 2 et 3 sont hors d'atteinte :
        # 'covariates_only' rend exactement les valeurs FINALES de False, lues
        # au niveau cible de la sortie publique
        values_false, _ = replay(
            fit_with_spy(
                reference_timeseries,
                covariate_strategy='interpolate',
                impute_intermediate_frequencies=False,
            ),
            reference_timeseries,
        )
        values_only, _ = replay(
            fit_with_spy(
                reference_timeseries,
                covariate_strategy='interpolate',
                impute_intermediate_frequencies='covariates_only',
            ),
            reference_timeseries,
        )
        for column in ('a1', 'a2', 'q1'):
            # "check_freq=False" : l'attribut de fréquence de l'index diffère
            # après deux étapes, les valeurs sont ce qui est comparé
            pd.testing.assert_series_equal(
                values_false[column],
                values_only[column],
                check_names=False,
                check_freq=False,
            )

    def test_y_train_filter_reads_origin_store_not_provenance(self, reference_timeseries):
        """Le filtre lit origin_store, jamais la provenance publique.

        Composants publics assemblés comme l'imputeur les assemble : un
        constructeur de jeu sur un matérialiseur réel, et la table publique
        "ELIGIBLE_ORIGINS" de l'imputeur, qui fixe les origines admises par
        modalité de l'axe 2.
        """
        builder = make_builder()
        materializer = builder.materializer
        frequencies = TS_FREQUENCIES

        # Deux cellules de MÊME provenance publique — toutes deux produites
        # par une même étape trimestrielle — mais d'origines différentes
        dates = pd.to_datetime(['2021-03-31', '2021-06-30', '2021-09-30'])
        materializer.record_production(
            'a1',
            pd.Series([30.0, 31.0, 32.0], index=dates),
            pd.Series(['observed', 'model', 'interpolated'], index=dates),
            pd.Series(['Q', 'Q', 'Q'], index=dates),
        )

        def _origins(modality):
            training = builder.build(
                column='a1',
                feature_cols=(),
                stage_freq='M',
                detected_frequencies=frequencies,
                source_data=reference_timeseries,
                eligible_origins=ELIGIBLE_ORIGINS[modality],
            )
            return training.row_origin.reindex(dates).dropna().to_list()

        # Sous 'covariates_only', seule la cellule observée entre ; la cellule
        # de repli 'interpolated' est exclue au même titre que celle de modèle
        assert _origins('covariates_only') == ['observed']
        # Sous True, les trois entrent
        assert sorted(_origins(True)) == ['interpolated', 'model', 'observed']

    @pytest.mark.slow
    def test_target_taint_families(self, reference_timeseries):
        """Les deux familles de souillure de cible n'existent que sous True."""
        tainted = {
            ProvenanceType.MODEL_ON_IMPUTED_TARGET,
            ProvenanceType.MODEL_ON_IMPUTED_BOTH,
        }
        for modality in (False, 'covariates_only'):
            imputer = fit_with_spy(
                reference_timeseries,
                covariate_strategy='model',
                impute_intermediate_frequencies=modality,
            )
            for step in imputer.imputation_plan_.steps:
                assert step.target_taint == 'none'
                assert step.emitted_provenance not in tainted

        imputer = fit_with_spy(
            reference_timeseries,
            covariate_strategy='model',
            impute_intermediate_frequencies=True,
        )
        emitted = {step.emitted_provenance for step in imputer.imputation_plan_.steps}
        assert emitted & tainted


class TestCoincidentCells:
    """Cellules coïncidentes et niveau de fréquence de l'index."""

    # Fabrique privée : les paramètres communs des mesures
    @staticmethod
    def _cascade(data, constraint, modality=True, **overrides):
        """Ajuste sur le jeu TS sous l'axe 2, et rend les contextes captés."""
        return capture_variable_fits(
            data,
            covariate_strategy='model',
            fit_predict_order='frequency',
            impute_intermediate_frequencies=modality,
            aggregation_constraint=constraint,
            **overrides,
        )

    # Fabrique privée : la même capture, sous un estimateur RÉEL
    @staticmethod
    def _capture_with_regression(data, **overrides):
        """Capte les contextes d'un fit mené par une régression linéaire.

        Les valeurs mesurées supposent un modèle réel : l'estimateur
        espion des autres tests ne prédit rien d'exploitable. L'enveloppe
        "RecordingEstimator" délègue à la régression et retient le jeu ajusté,
        lu ensuite sur le modèle de chaque étape du plan.
        """
        _imputer, fits = capture_fits(
            lambda: fit_quietly(
                make_imputer(estimator=RecordingEstimator(LinearRegression()), **overrides),
                data,
            )
        )
        return fits

    @pytest.mark.slow
    def test_coincident_cells_kept_only_without_constraint(self, reference_timeseries):
        """12 lignes sous 'sum', 15 sous None, à l'étape M."""
        _, under_sum = self._cascade(reference_timeseries, 'sum')
        _, under_none = self._cascade(reference_timeseries, None)
        for column in ('a1', 'a2'):
            # Trois ancres annuelles + neuf imputations Q : la quatrième de
            # chaque année tombe sur l'ancre, l'observation gagne
            assert len(under_sum[('M', column)].y_train) == 12
            # Aucun recalage : les quatre imputations de chaque année entrent
            assert len(under_none[('M', column)].y_train) == 15

    def test_collinearity_is_exact_under_sum(self, reference_timeseries):
        """La règle suit la structure algébrique du jeu, pas une préférence."""
        gold = {2021: 120.0, 2022: 132.0, 2023: 150.0}

        # Sortie de l'étape Q : les quatre imputations trimestrielles de a1
        def _totals(constraint):
            imputer = fit_with_spy(
                reference_timeseries,
                target_frequency='Q',
                covariate_strategy='model',
                impute_intermediate_frequencies=True,
                aggregation_constraint=constraint,
            )
            # Imputations de l'étape Q, lues au niveau cible de la sortie
            values, provenance = replay(imputer, reference_timeseries)
            produced = imputed_cells(values, provenance, 'a1')
            years = pd.DatetimeIndex(produced.index.get_level_values(-1)).year
            return {year: produced[years == year].sum() for year in gold}

        # Sous 'sum', le recalage impose Somme(sous-périodes) = total observé :
        # la ligne annuelle EST la somme des quatre trimestrielles
        exact = _totals('sum')
        for year, anchor in gold.items():
            assert exact[year] == pytest.approx(anchor, abs=1e-9)

        # Sous None, la colinéarité est rompue (écarts -0.65 / +2.37 / -0.60)
        free = _totals(None)
        assert any(abs(free[year] - anchor) > 0.1 for year, anchor in gold.items())

    @pytest.mark.slow
    def test_training_index_gains_a_frequency_level(self, reference_timeseries):
        """Sous None, l'index porte la fréquence ; sous 'sum', il est inchangé."""
        _, under_sum = self._cascade(reference_timeseries, 'sum')
        _, under_none = self._cascade(reference_timeseries, None)
        stamped = under_none[('M', 'a1')].training
        plain = under_sum[('M', 'a1')].training

        # Sous 'sum', l'index reste STRICTEMENT celui d'aujourd'hui : les
        # douze fins de trimestre, sans niveau ajoute
        expected = pd.DatetimeIndex(
            [
                date
                for date in pd.date_range('2021-01-31', '2023-12-31', freq='ME')
                if date.month in (3, 6, 9, 12)
            ],
            name='date',
        )
        assert not plain.has_frequency_level
        assert not isinstance(plain.X.index, pd.MultiIndex)
        assert plain.X.index.equals(expected)

        # Sous None, un niveau de plus, nommé 'frequency' et placé du côté de
        # l'entité — le bloc devient le couple (entité, fréquence)
        assert stamped.has_frequency_level
        assert list(stamped.X.index.names) == ['frequency', 'date']
        assert stamped.X.index.nlevels == plain.X.index.nlevels + 1

        # Les deux cellules du 2021-12-31 coexistent
        anchor = pd.Timestamp('2021-12-31')
        assert ('Y', anchor) in stamped.X.index
        assert ('Q', anchor) in stamped.X.index

        # Leurs diviseurs valent 12 (l'année) et 3 (le trimestre)
        raw = stamped.y
        scaled = under_none[('M', 'a1')].y_train
        assert raw[('Y', anchor)] / scaled[('Y', anchor)] == pytest.approx(12.0)
        assert raw[('Q', anchor)] / scaled[('Q', anchor)] == pytest.approx(3.0)

    def test_scaled_target_stays_homogeneous_across_levels(self, reference_timeseries):
        """Les deux cellules du 2021-12-31 valent ~10.0 de part et d'autre."""
        fits = self._capture_with_regression(
            reference_timeseries,
            covariate_strategy='model',
            fit_predict_order='frequency',
            impute_intermediate_frequencies=True,
            aggregation_constraint=None,
        )
        scaled = fits[('M', 'a1')].y_train
        anchor = pd.Timestamp('2021-12-31')
        # 120 / 12 d'un côté, ~30 / 3 de l'autre : une seule échelle mensuelle
        assert scaled[('Y', anchor)] == pytest.approx(10.0)
        assert scaled[('Q', anchor)] == pytest.approx(10.0, rel=0.1)

    @pytest.mark.parametrize('modality', [False, 'covariates_only'])
    @pytest.mark.slow
    def test_constraint_is_inert_on_y_train_without_axis_2(
        self, reference_timeseries, modality
    ):
        """Hors axe 2, aucune coïncidence n'est possible : l'effet est nul."""
        _, under_sum = self._cascade(reference_timeseries, 'sum', modality=modality)
        _, under_none = self._cascade(reference_timeseries, None, modality=modality)
        assert set(under_sum) == set(under_none)
        for key, fit in under_sum.items():
            pd.testing.assert_series_equal(fit.y_train, under_none[key].y_train)

    def test_per_column_constraint_is_read_per_column(self, reference_timeseries):
        """La forme dictionnaire est lue PAR COLONNE, jamais globalement."""
        _, fits = self._cascade(
            reference_timeseries, {'a1': None, '__default__': 'sum'}
        )
        # a1 garde ses cellules coïncidentes, a2 les perd
        assert len(fits[('M', 'a1')].y_train) == 15
        assert fits[('M', 'a1')].training.has_frequency_level
        assert len(fits[('M', 'a2')].y_train) == 12
        assert not fits[('M', 'a2')].training.has_frequency_level


class TestPerRowScale:
    """Le diviseur d'échelle est par ligne, jamais par étape."""

    @pytest.mark.slow
    def test_row_frequency_mixes_block_and_store_sources(
        self, mixed_freq_panel_multifrequency
    ):
        """Les deux sources de la fréquence de ligne coexistent."""
        _, fits = capture_variable_fits(
            mixed_freq_panel_multifrequency,
            covariate_strategy='interpolate',
            impute_intermediate_frequencies=True,
        )
        training = fits[('M', 'v')].training
        # Fréquences de BLOC : Y pour FR, Q pour DE, M pour IT
        assert dict(training.blocks) == {FR: 'Y', DE: 'Q', IT: 'M'}
        by_entity = {}
        for entity, frequency, origin in zip(
            [tuple(key[:-1]) for key in training.row_frequency.index],
            training.row_frequency,
            training.row_origin,
        ):
            by_entity.setdefault(entity, set()).add((frequency, origin))
        # FR porte ses 3 ancres annuelles ET ses imputations trimestrielles
        assert ('Y', 'observed') in by_entity[FR]
        assert ('Q', 'model') in by_entity[FR]
        # DE et IT n'apportent que des observations, à leur fréquence propre
        assert by_entity[DE] == {('Q', 'observed')}
        assert by_entity[IT] == {('M', 'observed')}

        # La cible mise à l'échelle reste homogène : le diviseur fractionnaire
        # du bloc IT (1/1 à l'étape M) n'est pas planchéré, celui de FR vaut 12
        scaled = fits[('M', 'v')].y_train
        for year, level in ((2021, 10.0), (2022, 11.0), (2023, 12.5)):
            values = scaled[
                scaled.index.get_level_values(-1).year == year
            ].to_numpy()
            assert values.mean() == pytest.approx(level, rel=0.2)

    @pytest.mark.slow
    def test_target_taint_of_a_contributing_entity_propagates(
        self, mixed_freq_panel_multifrequency
    ):
        """Une cellule 'model' d'une entité dégrade toute l'étape."""
        imputer = fit_with_spy(
            mixed_freq_panel_multifrequency,
            covariate_strategy='interpolate',
            impute_intermediate_frequencies=True,
        )
        monthly = [
            step
            for step in imputer.imputation_plan_.steps
            if step.pred_freq_label == 'M' and step.var_name == 'v'
        ]
        assert monthly
        # Les imputations trimestrielles de FR souillent la cible de l'étape,
        # donc la provenance des cellules produites pour DE aussi
        for step in monthly:
            assert step.target_taint == 'imputed'
        assert any(DE in (step.entities or ()) for step in monthly)
