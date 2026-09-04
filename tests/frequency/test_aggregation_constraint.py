"""Tests du lot L7 : ``tsforecast/frequency/aggregation_constraint.py``.

Couvre [SPEC] ``high_frequency_imputer2_architecture.md`` §11.1 (contrainte
d'agrégation et ses quatre gardes), §11.2 (désagrégation non paramétrable des
ancres et invariance de provenance), §6.4 et §6.5 (provenance et exemple chiffré
de référence),
§7.2 (masques de panel à MultiIndex), décisions D7 et D8 du §14.2.

Lot purement additif : ``hfi`` reste intact, sa méthode
``_rescale_to_period_totals`` y demeure en place et n'est pas touchée.
"""
# Importation des modules
# Modules de base
import warnings

# Calcul numérique
import numpy as np

# Manipulation de données
import pandas as pd

# Cadre de test
import pytest

# Objets testés
from tsforecast.frequency.aggregation_constraint import (
    AggregationConstraint,
    DEFAULT_CONSTRAINT_KEY,
)


# Prédictions brutes des douze mois de 2021 : somme 112.5, décembre 10.5.
# Ce sont les valeurs de l'exemple de référence du §6.5 — le ratio de recalage
# vaut 120 / 112.5, donc janvier 9.6, février 10.13… et l'ancre 11.2.
_RAW_2021 = [9.0, 9.5, 9.5, 9.5, 9.0, 9.5, 9.0, 9.5, 9.0, 9.5, 9.0, 10.5]


# Fabrique de la grille mensuelle d'une année
def _monthly_grid(year: int = 2021) -> pd.DatetimeIndex:
    """Build the twelve month-end dates of one year.

    Args:
        year: Calendar year of the grid.

    Returns:
        Month-end ``DatetimeIndex`` of twelve rows.
    """
    return pd.date_range(f'{year}-01-31', periods=12, freq='ME')


# Fabrique des prédictions brutes d'une année
def _raw_predictions(year: int = 2021, values=None) -> pd.Series:
    """Build the raw sub-period predictions of one year.

    Args:
        year: Calendar year of the grid.
        values: Values to carry; defaults to the §6.5 reference vector.

    Returns:
        ``Series`` named ``'a1'`` on the twelve month-end dates of ``year``.
    """
    return pd.Series(
        _RAW_2021 if values is None else values,
        index=_monthly_grid(year),
        name='a1',
    )


# Fabrique des observations annuelles sur une grille mensuelle
def _annual_observations(year: int = 2021, total: float = 120.0) -> pd.Series:
    """Build the annual observation of one year, on its monthly grid.

    Args:
        year: Calendar year of the grid.
        total: Observed annual total, carried by the December anchor.

    Returns:
        ``Series`` named ``'a1'``, non-null on the December anchor only.
    """
    observations = pd.Series(np.nan, index=_monthly_grid(year), name='a1')
    observations.iloc[-1] = total
    return observations


# Fabrique d'un bloc de panel à partir d'une série temporelle
def _to_panel(series: pd.Series, entities=('FR', 'DE')) -> pd.Series:
    """Replicate a time series over several entities, as a panel Series.

    Args:
        series: Time series to replicate.
        entities: Entity keys of the panel.

    Returns:
        ``Series`` on a ``MultiIndex`` ``(country, date)``.
    """
    blocks = []
    for entity in entities:
        block = series.copy()
        block.index = pd.MultiIndex.from_product(
            [[entity], series.index], names=['country', 'date']
        )
        blocks.append(block)
    return pd.concat(blocks)


# Tests de la validation du paramètre public (§11.1, D8)
class TestValidation:
    """Contrat de validation du paramètre ``aggregation_constraint``."""

    # Formes scalaires acceptées
    @pytest.mark.parametrize('setting', ['sum', 'mean', 'last', None])
    def test_scalar_settings_accepted(self, setting):
        """Les quatre réglages scalaires sont acceptés."""
        assert AggregationConstraint(setting).resolve_constraint() == setting

    # Forme dictionnaire de valeurs valides acceptée
    def test_dict_of_valid_values_accepted(self):
        """Un dict associant une colonne à un réglage valide est accepté."""
        constraint = AggregationConstraint(
            {'a1': 'mean', 'a2': 'last', DEFAULT_CONSTRAINT_KEY: None}
        )
        assert constraint.resolve_constraint('a1') == 'mean'
        assert constraint.resolve_constraint('a2') == 'last'
        assert constraint.resolve_constraint('q1') is None

    # Repli sur le défaut du document en l'absence de clé de repli
    def test_dict_without_default_key_falls_back_on_sum(self):
        """Sans clé de repli, une colonne non couverte retombe sur 'sum'."""
        assert AggregationConstraint({'a1': None}).resolve_constraint('q1') == 'sum'

    # Message d'erreur citant 'mean', 'last' et la forme dictionnaire
    def test_dict_form_raises_with_reserved_extension_message(self):
        """Une valeur non valide, seule ou dans un dict, lève un ValueError explicite."""
        # Forme scalaire non valide
        with pytest.raises(ValueError) as scalar_error:
            AggregationConstraint('median')
        message = str(scalar_error.value)
        assert "'sum'" in message and "'mean'" in message and "'last'" in message
        assert 'dict' in message and DEFAULT_CONSTRAINT_KEY in message

        # Forme dictionnaire portant une valeur non valide
        with pytest.raises(ValueError) as dict_error:
            AggregationConstraint({'a1': 'median'})
        dict_message = str(dict_error.value)
        assert "'mean'" in dict_message and "'last'" in dict_message
        assert 'dict' in dict_message and DEFAULT_CONSTRAINT_KEY in dict_message

    # Dictionnaire vide
    def test_empty_dict_raises(self):
        """Un dict vide ne désigne aucune colonne et lève."""
        with pytest.raises(ValueError, match='cannot be empty'):
            AggregationConstraint({})

    # Clés de colonnes inconnues
    def test_validate_columns_rejects_unknown_keys(self):
        """Les clés du dict sont contrôlées contre les colonnes réelles."""
        AggregationConstraint({'a1': 'sum'}).validate_columns(['a1', 'q1'])
        with pytest.raises(ValueError, match='unknown columns'):
            AggregationConstraint({'zz': 'sum'}).validate_columns(['a1'])


# Tests du recalage sur série temporelle (§11.1)
class TestRescaleTimeSeries:
    """Recalage aux totaux de période et ses quatre gardes, en série temporelle."""

    # Exemple chiffré de référence du document
    def test_reference_example_of_spec_6_5(self, reference_timeseries):
        """§6.5 : a1 = 120 en 2021, étape M, prédictions brutes sommant à 112.5."""
        # Grille et observations tirées du jeu de référence
        grid = reference_timeseries.loc['2021'].index
        observations = reference_timeseries.loc['2021', 'a1']
        values = pd.Series(_RAW_2021, index=grid, name='a1')

        # Vérification du point de départ : 112.5 et une ancre à 120
        assert values.sum() == pytest.approx(112.5)
        assert observations.loc['2021-12-31'] == 120.0

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        # La somme 2021 vaut EXACTEMENT le total observé
        assert rescaled.sum() == pytest.approx(120.0)
        # La ligne d'ancre porte une valeur de sous-période, non le total
        assert rescaled.loc['2021-12-31'] == pytest.approx(11.2)
        assert rescaled.loc['2021-12-31'] != 120.0
        # Les deux premières valeurs du tableau du §6.5
        assert rescaled.iloc[0] == pytest.approx(9.6)
        assert rescaled.iloc[1] == pytest.approx(10.1333333, abs=1e-6)
        # Toute la période est recalée, ancre comprise
        assert mask.all()

    # Garde 1 : période partiellement prédite
    def test_partial_period_not_rescaled(self):
        """Une sous-période NaN interdit la contrainte : valeurs brutes gardées."""
        values = _raw_predictions()
        values.iloc[3] = np.nan
        observations = _annual_observations()

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        # Prédictions brutes conservées, aucune cellule recalée
        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    # Garde 2 : période sans aucune observation
    def test_period_without_observation_not_rescaled(self):
        """Une fin de série retardée n'a pas de total à imposer."""
        values = _raw_predictions()
        # Aucune ancre : l'année 2021 n'est pas encore publiée
        observations = pd.Series(np.nan, index=values.index, name='a1')

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    # Garde 3 : agrégat prédit nul
    def test_zero_predicted_total_not_rescaled(self):
        """Un agrégat prédit nul rend le ratio indéfini : non recalée."""
        # Prédictions de somme nulle face à un total observé de 120
        values = _raw_predictions(values=[1.0, -1.0] * 6)
        observations = _annual_observations()

        with pytest.warns(UserWarning, match='aggregate to zero'):
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, observations, 'Y'
            )

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()

    # Garde 4 : signe opposé, un seul avertissement pour N périodes
    def test_opposite_sign_is_rescaled_and_warns_once(self):
        """Le recalage a lieu ; un SEUL UserWarning couvre les deux périodes."""
        # Deux années de prédictions négatives face à des totaux positifs
        grid = pd.date_range('2021-01-31', periods=24, freq='ME')
        values = pd.Series(-1.0, index=grid, name='a1')
        observations = pd.Series(np.nan, index=grid, name='a1')
        observations.loc['2021-12-31'] = 120.0
        observations.loc['2022-12-31'] = 132.0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            rescaled, mask = AggregationConstraint('sum').rescale(
                values, observations, 'Y'
            )

        # La contrainte prime : les deux périodes sont bien recalées
        assert mask.all()
        assert rescaled.loc['2021'].sum() == pytest.approx(120.0)
        assert rescaled.loc['2022'].sum() == pytest.approx(132.0)
        # Toutes les sous-périodes ont changé de signe
        assert (rescaled > 0).all()

        # UN SEUL avertissement, nommant les deux périodes concernées
        flipped = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(flipped) == 1
        assert '2 period(s)' in str(flipped[0].message)
        assert 'a1' in str(flipped[0].message)

    # Coïncidence du masque et des cellules modifiées
    def test_rescaled_mask_matches_modified_cells(self):
        """Le masque retourné coïncide exactement avec les cellules modifiées."""
        # Deux années : 2021 recalable, 2022 partiellement prédite
        grid = pd.date_range('2021-01-31', periods=24, freq='ME')
        values = pd.Series(_RAW_2021 * 2, index=grid, name='a1')
        values.loc['2022-05-31'] = np.nan
        observations = pd.Series(np.nan, index=grid, name='a1')
        observations.loc['2021-12-31'] = 120.0
        observations.loc['2022-12-31'] = 132.0

        rescaled, mask = AggregationConstraint('sum').rescale(values, observations, 'Y')

        # Cellules dont la valeur a effectivement changé (NaN exclu des deux côtés)
        changed = (rescaled != values) & values.notna()
        pd.testing.assert_series_equal(mask, changed, check_names=False)
        # 2021 recalée, 2022 laissée brute par la garde des périodes partielles
        assert mask.loc['2021'].all()
        assert not mask.loc['2022'].any()

    # Contrainte désactivée
    def test_constraint_none_leaves_raw_values(self):
        """Sous None, les prédictions brutes sont intactes et rien n'est recalé."""
        values = _raw_predictions()
        observations = _annual_observations()

        rescaled, mask = AggregationConstraint(None).rescale(values, observations, 'Y')

        pd.testing.assert_series_equal(rescaled, values)
        assert not mask.any()
        # Le total observé est bien écrasé : la somme ne fait plus 120
        assert rescaled.sum() == pytest.approx(112.5)

    # Contraintes 'mean' et 'last'
    def test_mean_and_last_constraints(self):
        """'mean' recale sur la moyenne, 'last' sur la dernière sous-période."""
        values = _raw_predictions()
        observations = _annual_observations(total=10.0)

        mean_rescaled, mean_mask = AggregationConstraint('mean').rescale(
            values, observations, 'Y'
        )
        assert mean_rescaled.mean() == pytest.approx(10.0)
        assert mean_mask.all()

        last_rescaled, last_mask = AggregationConstraint('last').rescale(
            values, observations, 'Y'
        )
        assert last_rescaled.iloc[-1] == pytest.approx(10.0)
        assert last_mask.all()
        # Profil conservé : toutes les sous-périodes suivent le même ratio
        ratio = 10.0 / values.iloc[-1]
        pd.testing.assert_series_equal(last_rescaled, values * ratio)


# Tests de la désagrégation des ancres (§11.2, D7)
class TestAnchorCellsMask:
    """Masque des ancres ré-exprimées à la fréquence d'étape.

    Masque de DIAGNOSTIC — il localise les totaux observés écrasés par une
    valeur de sous-période, il ne qualifie aucune provenance (§11.2).
    """

    # Indépendance du masque vis-à-vis de la contrainte
    def test_anchor_mask_independent_of_constraint(self, reference_timeseries):
        """Le masque d'ancre est identique sous 'sum' et sous None."""
        grid = reference_timeseries.index
        observations = reference_timeseries['a1']

        under_sum = AggregationConstraint('sum').anchor_cells_mask(observations, grid)
        under_none = AggregationConstraint(None).anchor_cells_mask(observations, grid)

        pd.testing.assert_series_equal(under_sum, under_none)
        # Les trois ancres annuelles, et elles seules
        assert under_sum.sum() == 3
        assert under_sum.loc['2021-12-31']
        assert not under_sum.loc['2021-11-30']

    # Indépendance vis-à-vis de la réussite du recalage
    def test_anchor_mask_independent_of_rescaling_success(self):
        """Une période non recalée garde sa ligne d'ancre repérée."""
        # Période partiellement prédite : le recalage n'a pas lieu
        values = _raw_predictions()
        values.iloc[3] = np.nan
        observations = _annual_observations()

        constraint = AggregationConstraint('sum')
        _, rescaled_mask = constraint.rescale(values, observations, 'Y')
        anchor_mask = constraint.anchor_cells_mask(observations, values.index)

        # Aucune cellule recalée, mais l'ancre reste repérée
        assert not rescaled_mask.any()
        assert anchor_mask.loc['2021-12-31']

    # Absence d'observation : aucune ancre
    def test_anchor_mask_all_false_without_observation(self):
        """Une colonne sans ancre sur la grille rend un masque tout à False."""
        grid = _monthly_grid()
        observations = pd.Series(np.nan, index=grid, name='a1')

        mask = AggregationConstraint('sum').anchor_cells_mask(observations, grid)

        assert not mask.any()


# Tests du protocole sklearn, sur plusieurs colonnes à la fois
class TestTransformerProtocol:
    """``fit`` / ``transform`` sur une trame de plusieurs colonnes."""

    # Recalage simultané de deux colonnes
    def test_transform_rescales_every_column(self, reference_timeseries):
        """Deux colonnes annuelles sont recalées en une passe."""
        grid = reference_timeseries.loc['2021'].index
        X = pd.DataFrame(
            {'a1': _RAW_2021, 'a2': [value / 2 for value in _RAW_2021]},
            index=grid,
        )
        observations = reference_timeseries.loc['2021', ['a1', 'a2']]

        constraint = AggregationConstraint(
            'sum', period_frequencies='Y', observations=observations
        ).fit(X)
        rescaled = constraint.transform(X)

        assert rescaled['a1'].sum() == pytest.approx(120.0)
        assert rescaled['a2'].sum() == pytest.approx(60.0)
        assert constraint.rescaled_mask_.all().all()
        # Masque d'ancre gelé au fit, indépendant des valeurs
        assert constraint.anchor_mask_['a1'].sum() == 1
        assert list(constraint.feature_names_in_) == ['a1', 'a2']
        assert constraint.n_features_in_ == 2

    # Contrainte par colonne
    def test_transform_honours_per_column_dict(self, reference_timeseries):
        """La forme dict applique une contrainte différente par colonne."""
        grid = reference_timeseries.loc['2021'].index
        X = pd.DataFrame(
            {'a1': _RAW_2021, 'a2': [value / 2 for value in _RAW_2021]},
            index=grid,
        )
        observations = reference_timeseries.loc['2021', ['a1', 'a2']]

        constraint = AggregationConstraint(
            {'a1': 'sum', DEFAULT_CONSTRAINT_KEY: None},
            period_frequencies='Y',
            observations=observations,
        ).fit(X)
        rescaled = constraint.transform(X)

        # a1 recalée, a2 laissée brute
        assert rescaled['a1'].sum() == pytest.approx(120.0)
        pd.testing.assert_series_equal(rescaled['a2'], X['a2'])
        assert constraint.rescaled_mask_['a1'].all()
        assert not constraint.rescaled_mask_['a2'].any()

    # Un seul avertissement pour toutes les colonnes
    def test_transform_warns_once_for_every_column(self):
        """Deux colonnes retournées n'émettent qu'un seul avertissement."""
        grid = _monthly_grid()
        X = pd.DataFrame({'a1': [-1.0] * 12, 'a2': [-1.0] * 12}, index=grid)
        observations = pd.DataFrame(
            {'a1': [np.nan] * 11 + [120.0], 'a2': [np.nan] * 11 + [60.0]},
            index=grid,
        )

        constraint = AggregationConstraint(
            'sum', period_frequencies='Y', observations=observations
        ).fit(X)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            constraint.transform(X)

        flipped = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(flipped) == 1
        assert "'a1'" in str(flipped[0].message) and "'a2'" in str(flipped[0].message)
        assert '2 period(s)' in str(flipped[0].message)

    # Métadonnées manquantes
    def test_fit_requires_metadata(self):
        """L'absence de fréquence de période est une erreur de branchement."""
        grid = _monthly_grid()
        X = pd.DataFrame({'a1': _RAW_2021}, index=grid)
        observations = pd.DataFrame({'a1': [np.nan] * 11 + [120.0]}, index=grid)

        with pytest.raises(ValueError, match='period_frequencies'):
            AggregationConstraint('sum', observations=observations).fit(X)

        with pytest.raises(ValueError, match='observations'):
            AggregationConstraint('sum', period_frequencies='Y').fit(X)


# Tests sur données de panel (§7.2)
class TestPanel:
    """Recalage et masques sur un panel à MultiIndex ``(country, date)``."""

    # Exemple de référence, version panel
    def test_reference_example_of_spec_6_5_panel(self):
        """§6.5 sur panel : chaque entité est recalée sur son propre total."""
        values = _to_panel(_raw_predictions())
        # FR observe 120 en 2021, DE observe 240 : les blocs sont indépendants
        observations = _to_panel(_annual_observations())
        observations.loc[('DE', pd.Timestamp('2021-12-31'))] = 240.0

        rescaled, mask = AggregationConstraint('sum').rescale(
            values, observations, 'Y'
        )

        assert rescaled.loc['FR'].sum() == pytest.approx(120.0)
        assert rescaled.loc['DE'].sum() == pytest.approx(240.0)
        # La ligne d'ancre porte une valeur de sous-période, non le total
        assert rescaled.loc[('FR', pd.Timestamp('2021-12-31'))] == pytest.approx(11.2)
        assert rescaled.loc[('DE', pd.Timestamp('2021-12-31'))] == pytest.approx(22.4)
        assert mask.all()

    # Masque d'ancre, version panel
    def test_anchor_mask_independent_of_constraint_panel(
        self, mixed_freq_panel_heterogeneous
    ):
        """Sur panel : masque MultiIndex identique sous 'sum' et sous None."""
        grid = mixed_freq_panel_heterogeneous.index
        observations = mixed_freq_panel_heterogeneous['a1']

        under_sum = AggregationConstraint('sum').anchor_cells_mask(observations, grid)
        under_none = AggregationConstraint(None).anchor_cells_mask(observations, grid)

        pd.testing.assert_series_equal(under_sum, under_none)
        # Type de retour unifié : une Series booléenne à MultiIndex (§7.2)
        assert isinstance(under_sum, pd.Series)
        assert isinstance(under_sum.index, pd.MultiIndex)
        # Trois ancres annuelles pour chacune des trois entités
        assert under_sum.sum() == 9
        assert under_sum.loc[('FR', pd.Timestamp('2021-12-31'))]

    # Étanchéité des blocs entre entités
    def test_entities_never_share_a_rescaling_block(self):
        """Une entité partiellement prédite n'empêche pas le recalage de l'autre."""
        values = _to_panel(_raw_predictions())
        # DE porte une sous-période manquante : seule DE échappe au recalage
        values.loc[('DE', pd.Timestamp('2021-05-31'))] = np.nan
        observations = _to_panel(_annual_observations())

        rescaled, mask = AggregationConstraint('sum').rescale(
            values, observations, 'Y'
        )

        assert mask.loc['FR'].all()
        assert not mask.loc['DE'].any()
        assert rescaled.loc['FR'].sum() == pytest.approx(120.0)
