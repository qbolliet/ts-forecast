"""Inter-entity mutualization of the training set (§5.8, §5.10).

Covers the pooled training set and the single fit per (stage, variable)
(I14, I15, D17-D19), and entities never observing the imputed column
(``impute_unobserved_entities``, D33, I21).
"""

# Modules de base
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Objets testés
from tsforecast.frequency.high_frequency_imputer import HighFrequencyImputer
from tsforecast.frequency.covariate_materializer import CovariateMaterializer
from tsforecast.frequency.provenance import resolve_model_provenance

# Outils de test partagés
from tests.support.estimators import SpyEstimator
from tests.unit.frequency.high_frequency_imputer.helpers import (
    FR,
    DE,
    IT,
    by_entity,
    capture_variable_fits,
    entity_block,
    fit_quietly,
    fit_with_spy,
    imputed_cells,
    make_imputer,
    replay,
    target_level,
)


class TestMutualizedTrainingSet:
    """I14 et I15 — mutualisation inter-entités et ajustement unique."""

    @staticmethod
    def _variable_steps(imputer: HighFrequencyImputer) -> list:
        """Étapes de la variable 'v' du jeu PANEL-F."""
        return [step for step in imputer.imputation_plan_ if step.var_name == 'v']

    def test_pooled_training_set_on_panel_f(self, mixed_freq_panel_multifrequency):
        """I14 — 51 lignes (3 FR + 12 DE + 36 IT), toutes à l'échelle de l'étape."""
        imputer = fit_with_spy(mixed_freq_panel_multifrequency)
        step = self._variable_steps(imputer)[0]
        y_train = step.model.fit_y_

        assert len(y_train) == 51
        counts = pd.Series(
            [key[0] for key in y_train.index]
        ).value_counts().to_dict()
        assert counts == {'IT': 36, 'DE': 12, 'FR': 3}
        assert dict(step.training_blocks) == {('FR',): 'Y', ('DE',): 'Q', ('IT',): 'M'}

        # Valeurs d'or du §5.8 : FR et IT à 10.0 / 11.0 / 12.5, DE aux tiers
        france = entity_block(y_train, 'FR')
        assert [round(value, 6) for value in france] == [10.0, 11.0, 12.5]

        italy = entity_block(y_train, 'IT')
        assert [round(value, 6) for value in italy.loc['2021']] == [10.0] * 12
        assert [round(value, 6) for value in italy.loc['2022']] == [11.0] * 12
        assert [round(value, 6) for value in italy.loc['2023']] == [12.5] * 12

        germany = entity_block(y_train, 'DE')
        assert [round(value, 3) for value in germany[:4]] == [
            round(28 / 3, 3), round(30 / 3, 3), round(31 / 3, 3), round(31 / 3, 3)
        ]

        # Aucune ligne ne mêle deux échelles : toutes tiennent dans la plage
        # mensuelle du jeu
        assert y_train.min() > 9.0 and y_train.max() < 14.0

    def test_pooled_training_set_at_stage_q(self, mixed_freq_panel_multifrequency):
        """I14 — à l'étape Q, 51 lignes : FR divisé par 4, IT par 1/3, DE brut.

        Les blocs ne dépendent pas de l'étape (D18) : IT garde ses 36 lignes
        mensuelles, ramenées à l'échelle trimestrielle (diviseur 1/3).
        """
        _, fits = capture_variable_fits(
            mixed_freq_panel_multifrequency,
            impute_intermediate_frequencies='covariates_only',
        )
        y_train = fits[('Q', 'v')].y_train

        assert len(y_train) == 51
        counts = pd.Series([key[0] for key in y_train.index]).value_counts().to_dict()
        assert counts == {'IT': 36, 'DE': 12, 'FR': 3}

        france = entity_block(y_train, 'FR')
        assert [round(value, 6) for value in france] == [30.0, 33.0, 37.5]
        italy = entity_block(y_train, 'IT')
        assert [round(value, 6) for value in italy.loc['2021']] == [30.0] * 12
        assert [round(value, 6) for value in italy.loc['2023']] == [37.5] * 12
        germany = entity_block(y_train, 'DE')
        observed = entity_block(mixed_freq_panel_multifrequency['v'], 'DE').dropna()
        np.testing.assert_allclose(germany.to_numpy(), observed.to_numpy())

    def test_single_fit_shared_between_source_frequency_groups(
        self, mixed_freq_panel_multifrequency
    ):
        """I15 — un seul ajustement, un seul objet modèle, deux recalages distincts."""
        imputer = fit_with_spy(mixed_freq_panel_multifrequency)
        steps = self._variable_steps(imputer)

        # Deux étapes de plan, une par groupe de fréquence source
        assert len(steps) == 2
        assert {step.source_frequency for step in steps} == {'Y', 'Q'}
        assert {step.entities for step in steps} == {(('FR',),), (('DE',),)}

        # Un seul ajustement pour la variable, quel que soit le nombre de
        # groupes : 'q1' et 'v' font deux ajustements en tout
        assert SpyEstimator.n_fits == 2

        # MÊME objet modèle, mêmes features, mêmes blocs, mêmes souillures
        first, second = steps
        assert first.model is second.model
        assert first.feature_cols == second.feature_cols
        assert dict(first.training_blocks) == dict(second.training_blocks)
        assert dict(first.materialization) == dict(second.materialization)
        assert (first.covariate_taint, first.target_taint) == (
            second.covariate_taint, second.target_taint
        )

        # Recalages distincts : total annuel exact pour FR, trimestriel pour DE
        values, _provenance = replay(imputer, mixed_freq_panel_multifrequency)
        store = values['v']
        assert entity_block(store, 'FR').loc['2021'].sum() == pytest.approx(120.0)
        assert entity_block(store, 'DE').loc['2021-01':'2021-03'].sum() == (
            pytest.approx(28.0)
        )

    def test_contributing_entity_is_never_rewritten(
        self, mixed_freq_panel_multifrequency
    ):
        """IT fournit 36 des 51 lignes d'entraînement et n'est jamais réécrite."""
        imputer = fit_with_spy(mixed_freq_panel_multifrequency)

        # Aucune étape ne nomme IT parmi ses entités
        for step in self._variable_steps(imputer):
            assert ('IT',) not in (step.entities or ())

        # Ses cellules restent ORIGINAL
        provenance = entity_block(imputer.imputation_provenance_['v'], 'IT')
        assert {str(value) for value in provenance} == {'original'}

        # Et aucune n'est écrite : seules FR et DE portent des cellules
        # imputées de 'v' dans la sortie, IT ne l'étant jamais
        values, provenance = replay(imputer, mixed_freq_panel_multifrequency)
        written = by_entity(imputed_cells(values, provenance, 'v').index)
        assert set(written) == {('FR',), ('DE',)}

    def test_target_taint_is_computed_by_the_origin_filter(
        self, mixed_freq_panel_multifrequency
    ):
        """Sous False, y_train n'est fait que d'observations : souillure 'none'."""
        imputer = fit_with_spy(mixed_freq_panel_multifrequency)
        for step in imputer.imputation_plan_:
            assert step.target_taint == 'none'


class TestUnobservedEntities:
    """I21 — imputation complète d'une entité qui n'observe jamais la colonne."""

    @staticmethod
    def _v_steps(imputer: HighFrequencyImputer) -> list:
        """Étapes de plan de la variable 'v'."""
        return [step for step in imputer.imputation_plan_ if step.var_name == 'v']

    @staticmethod
    def _imputed(
        imputer: HighFrequencyImputer, data: pd.DataFrame, entity: str
    ) -> pd.Series:
        """Cellules de 'v' imputées pour une entité, lues sur la sortie publique.

        Le rejeu réécrit "imputation_provenance_" : "_provenance" la lit donc
        au niveau cible, identique avant et après le transform (I1).
        """
        values, provenance = replay(imputer, data)
        store = imputed_cells(values, provenance, 'v')
        if entity not in set(store.index.get_level_values(0)):
            return pd.Series(dtype=float)
        return entity_block(store, entity)

    @staticmethod
    def _provenance(imputer: HighFrequencyImputer, entity: str) -> pd.Series:
        """Provenance de 'v' pour une entité, au niveau cible.

        Lue au niveau cible, la matrice est la même après le fit et après un
        transform des mêmes données (I1) : l'ordre des lectures est indifférent.
        """
        provenance = target_level(imputer, imputer.imputation_provenance_)
        return entity_block(provenance['v'], entity)

    def test_unobserved_entity_is_left_alone_by_default(self, panel_f_without_it_v):
        """I21 — sous le défaut False, IT n'est ni imputée ni nommée par le plan."""
        imputer = fit_quietly(make_imputer(), panel_f_without_it_v)

        # Les 36 cellules restent vides et non imputées : la matrice de
        # provenance du FIT laisse une cellule NaN sans marque, aucune ne
        # portant donc la moindre provenance de modèle. Lue AVANT le rejeu, qui
        # la réécrit (et la marque à tort ORIGINAL, ANO-FREQ-050)
        provenance = self._provenance(imputer, 'IT')
        assert len(provenance) == 36
        assert provenance.isna().all()

        # Aucune cellule produite : seules FR et DE portent des imputations
        assert self._imputed(imputer, panel_f_without_it_v, 'IT').empty

        # Aucune étape de plan ne nomme IT parmi ses entités
        for step in self._v_steps(imputer):
            assert IT not in (step.entities or ())
            assert step.source_frequency is not None
            assert step.unanchored is False

        # L'attribut est écrit, et vide
        assert imputer.unanchored_pairs_ == ()

    def test_unobserved_entity_is_imputed_on_demand(self, panel_f_without_it_v):
        """I21 — sous True, les 36 cellules sont produites et marquées sans ancre."""
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True), panel_f_without_it_v
        )

        # Les 36 cellules mensuelles sont renseignées
        imputed = self._imputed(imputer, panel_f_without_it_v, 'IT')
        assert len(imputed) == 36
        assert imputed.notna().all()

        # Toutes portent MODEL_UNANCHORED, et elles seules
        assert {str(value) for value in self._provenance(imputer, 'IT')} == {
            'model_unanchored'
        }

        # Le couple est rendu par l'attribut ajusté, sous-ensemble des couples
        # sans fréquence détectée
        assert imputer.unanchored_pairs_ == (('IT', 'v'),)
        # sans fréquence détectée : l'entité et la colonne existent, le couple
        # n'a pas de fréquence
        assert IT in imputer.entities_ and 'v' in imputer.feature_columns_
        assert not set(imputer.unanchored_pairs_) & set(imputer.detected_frequencies_)

    def test_unanchored_step_shares_the_model_and_carries_no_scale(
        self, panel_f_without_it_v
    ):
        """L'étape (v, None) partage le modèle des groupes ancrés, sans diviseur."""
        imputer = fit_quietly(
            make_imputer(impute_unobserved_entities=True), panel_f_without_it_v
        )
        steps = {step.source_frequency: step for step in self._v_steps(imputer)}
        assert set(steps) == {'Y', 'Q', None}

        unanchored = steps[None]
        assert unanchored.source_frequency is None
        assert unanchored.unanchored is True
        assert unanchored.entities == (IT,)
        assert unanchored.scale_factor == 1.0
        assert unanchored.fit_scale_factor == 1.0

        # MÊME objet modèle que les deux étapes ancrées de la même étape de
        # fréquence (§5.8 R6, D19)
        assert unanchored.model is steps['Y'].model
        assert unanchored.model is steps['Q'].model

    @pytest.mark.slow
    def test_unanchored_entity_contributes_nothing_to_training(
        self, panel_f_without_it_v
    ):
        """R1 inchangée — IT n'apporte aucune ligne au jeu mutualisé."""
        imputer = fit_with_spy(
            panel_f_without_it_v, impute_unobserved_entities=True
        )
        steps = self._v_steps(imputer)
        assert len(steps) == 3

        # Les blocs ne nomment que les entités qui OBSERVENT la colonne
        for step in steps:
            assert dict(step.training_blocks) == {FR: 'Y', DE: 'Q'}

        # 15 lignes mutualisées : 3 annuelles de FR, 12 trimestrielles de DE
        y_train = steps[0].model.fit_y_
        assert len(y_train) == 15
        assert pd.Series(
            [key[0] for key in y_train.index]
        ).value_counts().to_dict() == {'DE': 12, 'FR': 3}

        # Le même compte sans le paramètre : R1 ne bouge pas
        without = fit_with_spy(panel_f_without_it_v)
        assert len(self._v_steps(without)[0].model.fit_y_) == 15

    @pytest.mark.parametrize('constraint', ['sum', None])
    def test_no_aggregation_constraint_on_unanchored_cells(
        self, panel_f_without_it_v, constraint
    ):
        """Aucun total de période n'est imposé aux cellules sans ancre."""
        imputer = fit_quietly(
            make_imputer(
                impute_unobserved_entities=True, aggregation_constraint=constraint
            ),
            panel_f_without_it_v,
        )

        # La somme 2021 d'IT n'a aucune raison de valoir 120 : ses cellules
        # sont des prédictions libres, jamais la désagrégation d'un total
        italy = self._imputed(imputer, panel_f_without_it_v, 'IT')
        assert italy.loc['2021'].sum() != pytest.approx(120.0)

        # Le recalage des groupes ANCRÉS, lui, reste celui du §11.1
        if constraint == 'sum':
            france = self._imputed(imputer, panel_f_without_it_v, 'FR')
            assert france.loc['2021'].sum() == pytest.approx(120.0)

    @pytest.mark.parametrize('modality', [False, 'covariates_only', True])
    @pytest.mark.slow
    def test_progression_is_unchanged_by_the_parameter(
        self, panel_f_without_it_v, modality
    ):
        """Le couple sans ancre n'entre dans aucun ensemble F : la progression est identique."""
        without = fit_with_spy(
            panel_f_without_it_v, impute_intermediate_frequencies=modality
        )
        with_parameter = fit_with_spy(
            panel_f_without_it_v,
            impute_intermediate_frequencies=modality,
            impute_unobserved_entities=True,
        )
        assert (
            with_parameter.frequency_progression_ == without.frequency_progression_
        )
        assert (
            with_parameter.variable_categories_ == without.variable_categories_
        )

        # La capacité joue sous les TROIS modalités, l'axe 2 n'y étant pour rien
        assert with_parameter.unanchored_pairs_ == (('IT', 'v'),)

    def test_unanchored_pair_without_covariates_warns_once_and_stays_nan(
        self, panel_f_with_a_mute_entity
    ):
        """Sans covariable exploitable, la ligne est vide : NaN, et un seul avertissement."""
        imputer = make_imputer(impute_unobserved_entities=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            imputer.fit(panel_f_with_a_mute_entity)

        # Aucune provenance de modèle (matrice du fit, lue avant le rejeu), et
        # aucune cellule produite pour IT
        assert self._provenance(imputer, 'IT').isna().all()
        assert self._imputed(imputer, panel_f_with_a_mute_entity, 'IT').empty
        assert imputer.unanchored_pairs_ == ()

        # UN SEUL avertissement nomme le couple, agrégé en fin de fit
        naming = [
            str(warning.message) for warning in caught
            if "('IT', 'v')" in str(warning.message)
        ]
        assert len(naming) == 1

    @pytest.mark.slow
    def test_model_unanchored_is_never_emitted_by_default(
        self, panel_f_without_it_v, mixed_freq_panel_multifrequency
    ):
        """I6 non-régression — l'ajout est additif, aucune cellule ne change de marque."""
        for data in (panel_f_without_it_v, mixed_freq_panel_multifrequency):
            imputer = fit_with_spy(data)
            emitted = {
                str(value)
                for value in imputer.imputation_provenance_.to_numpy().ravel()
            }
            assert 'model_unanchored' not in emitted

            # Les cinq familles restent résolues par les deux seules souillures
            for step in imputer.imputation_plan_:
                if step.is_fallback:
                    continue
                assert step.emitted_provenance == resolve_model_provenance(
                    step.covariate_taint, step.target_taint
                )

    @pytest.mark.parametrize('modality', [False, True])
    @pytest.mark.slow
    def test_adding_the_unanchored_group_leaves_the_others_intact(
        self, panel_f_without_it_v, modality
    ):
        """I15 non-régression — X_train, y_train et les voies des groupes ancrés.

        Sous ``True``, la mesure porte aussi ce que la règle promet : les
        passes intermédiaires de l'entité sans ancre ne changent PAS le
        modèle de l'étape cible, R1 la tenant hors du jeu mutualisé qu'elle
        porte ou non des cellules imputées. Même jeu, même modèle, donc mêmes
        valeurs finales.
        """
        without = fit_with_spy(
            panel_f_without_it_v, impute_intermediate_frequencies=modality
        )
        with_parameter = fit_with_spy(
            panel_f_without_it_v,
            impute_intermediate_frequencies=modality,
            impute_unobserved_entities=True,
        )

        def _anchored(imputer, stage):
            return {
                step.source_frequency: step
                for step in self._v_steps(imputer)
                if step.source_frequency is not None
                and step.pred_freq_label == stage
            }

        anchored = _anchored(with_parameter, 'M')
        reference = _anchored(without, 'M')
        assert set(anchored) == set(reference) == {'Y', 'Q'}

        for source_frequency, step in anchored.items():
            other = reference[source_frequency]
            # Même jeu mutualisé, aux valeurs près
            pd.testing.assert_frame_equal(step.model.fit_X_, other.model.fit_X_)
            pd.testing.assert_series_equal(step.model.fit_y_, other.model.fit_y_)
            # Mêmes voies de matérialisation, mêmes entités, même échelle
            assert dict(step.materialization) == dict(other.materialization)
            assert step.entities == other.entities
            assert step.scale_factor == other.scale_factor

    def test_unanchored_pair_travels_every_stage_of_its_progression(
        self, panel_f_without_it_v
    ):
        """Le couple rejoint TOUTES les étapes que sa progression traverse.

        L'étape unique de ``False`` est le cas dégénéré de cette règle, non
        une exception : sous ``True`` la progression de ``IT`` compte deux
        étapes, et le couple les rejoint toutes les deux.
        """
        imputer = fit_with_spy(
            panel_f_without_it_v,
            impute_intermediate_frequencies=True,
            impute_unobserved_entities=True,
        )

        # Deux étapes dans la progression, l'intermédiaire et la cible
        assert [
            set(stage.values()) for stage in imputer.frequency_progression_
        ] == [{'Q'}, {'M'}]

        # Une étape de plan sans ancre à CHACUNE, jamais à la seule dernière
        unanchored = [step for step in self._v_steps(imputer) if step.unanchored]
        assert {step.pred_freq_label for step in unanchored} == {'Q', 'M'}
        for step in unanchored:
            assert step.source_frequency is None
            assert step.entities == (IT,)
            assert step.scale_factor == 1.0
            # R1 inchangée à toutes les étapes : IT n'entre dans aucun bloc,
            # qu'elle porte ou non les cellules d'une passe antérieure
            assert IT not in dict(step.training_blocks)

        # Le couple n'est nommé qu'une fois, quel que soit le nombre de passes
        assert imputer.unanchored_pairs_ == (('IT', 'v'),)

    def test_every_pass_stays_unanchored_and_unrescaled(self, panel_f_without_it_v):
        """Une passe ancrée sur une prédiction d'elle-même n'est pas ancrée.

        Aucune contrainte d'agrégation ne relie les niveaux d'une entité sans
        ancre — pas plus qu'elle n'en relie ceux d'une entité ANCRÉE, dont les
        deux niveaux sont recalés sur le total observé commun et jamais l'un
        sur l'autre.
        """
        productions: list = []
        real = CovariateMaterializer.record_production

        def _spy(materializer, column, values, origins, freqs):
            """Retient chaque production, le miroir n'en gardant que la dernière."""
            productions.append((column, freqs.iloc[0], values.copy()))
            return real(materializer, column, values, origins, freqs)

        # Estimateur réel, et non l'espion : celui-ci prédit une constante,
        # de sorte que les deux niveaux coïncideraient par construction et
        # que la mesure ne dirait plus rien
        with patch.object(CovariateMaterializer, 'record_production', _spy):
            imputer = fit_quietly(
                make_imputer(
                    impute_intermediate_frequencies=True,
                    impute_unobserved_entities=True,
                ),
                panel_f_without_it_v,
            )

        # Productions de 'v' pour IT, une par étape
        by_stage = {
            stage: values
            for column, stage, values in productions
            if column == 'v' and set(by_entity(values.index)) == {IT}
        }
        assert set(by_stage) == {'Q', 'M'}

        # Le trimestre imputé à l'étape Q et la somme des trois mois de la
        # même période à l'étape M ne coïncident pas : rien ne les lie
        quarterly = entity_block(by_stage['Q'], 'IT').loc['2021-01':'2021-03']
        monthly = entity_block(by_stage['M'], 'IT').loc['2021-01':'2021-03']
        assert len(quarterly) == 1 and len(monthly) == 3
        assert float(monthly.sum()) != pytest.approx(float(quarterly.sum()))

        # Toutes les cellules, des deux niveaux, portent MODEL_UNANCHORED
        assert {str(value) for value in self._provenance(imputer, 'IT')} == {
            'model_unanchored'
        }
