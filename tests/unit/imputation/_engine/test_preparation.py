"""Tests of ``prepare_work_frame`` and its helpers in ``tsforecast.imputation._engine.preparation``.

Scenarios: target naming and alignment (equal indices, another date unit, pre-conversion index,
mismatched labels), date units per level, reserved level name, window calculator factory and
readable bounds; preparation of a fit (working frame and snapshots, frequency layout, entities,
input index label, window masks, additive transformer cloned and fitted, provenance tracker on
the transformed frame, nothing written on the imputer) and of a transform (same frame and windows
as the fit on the fit data, prediction pass, frequency check, window failure, fitted state left
intact), at every date resolution.
"""

# Modules de base
import warnings

# Manipulation de données
import numpy as np
import pandas as pd

# Tests
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

from tests.support.perturbations import (
    DATETIME_UNITS,
    datetime_units as datetime_units_of,
    drop_entity,
    with_datetime_resolution,
)
from tsforecast.imputation import HighFrequencyImputer, ProvenanceType
from tsforecast.imputation._components.imputation_window import ImputationWindowCalculator
from tsforecast.imputation._engine import preparation
from tsforecast.imputation._engine.frequency_layout import plan_frequencies
from tsforecast.imputation._engine.preparation import (
    align_target_index,
    check_reserved_level_name,
    datetime_units,
    make_window_calculator,
    prepare_work_frame,
    resolve_target_column_name,
    zip_window_bounds,
)

# Trois genres de masque de fenêtre, dans l'ordre où le fit les calcule
_MASK_KINDS = ('strict', 'imputation', 'training')


# Transformateur additif qui vide une colonne après une date et garde sa dernière entrée
class _BlankAfter(BaseEstimator, TransformerMixin):
    """Additive transformer blanking one column after a date, recording its last input.

    It changes what is observed — the window and the provenance read it —
    without changing the frequency of the column, so a whole fit runs on it.

    Args:
        column: Column blanked.
        after: Last date of the column kept.
    """

    def __init__(self, column: str = 'a1', after: str = '2022-12-31'):
        self.column = column
        self.after = after

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
        """Record a copy of ``X`` and return it with the column blanked after the date.

        Args:
            X: Data, a time series or a panel.

        Returns:
            The blanked copy.
        """
        self.last_input_ = X.copy()
        blanked = X.copy()
        dates = blanked.index.get_level_values(-1)
        blanked.loc[dates > pd.Timestamp(self.after), self.column] = np.nan
        return blanked

    def inverse_transform(self, X):
        """Return ``X`` unchanged.

        Args:
            X: Data.

        Returns:
            ``X``.
        """
        return X


# Transformateur additif à la manière d'un transformateur XY : il rend un couple
class _PairReturningIdentity(BaseEstimator, TransformerMixin):
    """Identity additive transformer returning the ``(X, y)`` pair of an XY transformer."""

    def fit(self, X, y=None):
        """Learn nothing.

        Args:
            X: Data.
            y: Ignored.

        Returns:
            self.
        """
        return self

    def transform(self, X, y=None):
        """Return a copy of ``X`` paired with ``y``.

        Args:
            X: Data.
            y: Optional target.

        Returns:
            The pair ``(X, y)``.
        """
        return X.copy(), y

    def inverse_transform(self, X):
        """Return ``X`` unchanged.

        Args:
            X: Data.

        Returns:
            ``X``.
        """
        return X


# Imputeur valide, surchargeable paramètre par paramètre
def _imputer(**overrides) -> HighFrequencyImputer:
    """Build an unfitted imputer targeting the monthly frequency.

    Args:
        **overrides: Constructor parameters to change.

    Returns:
        The imputer.
    """
    params = dict(target_frequency='M', estimator=LinearRegression())
    params.update(overrides)
    return HighFrequencyImputer(**params)


# Imputeur ajusté sans avertissement affiché
def _fitted(data: pd.DataFrame, y: pd.Series = None, **overrides) -> HighFrequencyImputer:
    """Fit an imputer, swallowing the warnings the fit legitimately emits.

    Args:
        data: Features of the fit.
        y: Target of the fit, optional.
        **overrides: Constructor parameters to change.

    Returns:
        The fitted imputer.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _imputer(**overrides).fit(data, y)


# Préparation sans avertissement affiché
def _prepare(imputer, X, y=None, **kwargs):
    """Prepare a pass, swallowing the warnings the preparation legitimately emits.

    Args:
        imputer: Imputer of the pass.
        X: Features.
        y: Target, optional.
        **kwargs: ``fitting`` and ``prediction``.

    Returns:
        The prepared data.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return prepare_work_frame(imputer, X, y, **kwargs)


# Cible mensuelle partageant l'index d'un jeu
def _target(data: pd.DataFrame, name='y') -> pd.Series:
    """Build a dense monthly target on the index of a frame.

    Args:
        data: Frame whose index the target takes.
        name: Name of the target.

    Returns:
        The target, ``2 * m1``.
    """
    return (data['m1'] * 2.0).rename(name)


# Instantané des attributs d'un imputeur : noms et identités des valeurs
def _attribute_identities(imputer: HighFrequencyImputer) -> dict:
    """Map every attribute of the instance to the identity of its value.

    Args:
        imputer: Imputer.

    Returns:
        ``{name: id(value)}`` over ``vars(imputer)``.
    """
    return {name: id(value) for name, value in vars(imputer).items()}


# =============================================================================
# Cible : nom et alignement
# =============================================================================
class TestResolveTargetColumnName:
    """The target is merged under its own name, or under the fallback."""

    @pytest.mark.parametrize(
        ('name', 'expected'),
        [('gdp', 'gdp'), (None, '__target__'), ('cible é', 'cible é'), (0, 0)],
        ids=['named', 'unnamed', 'accented', 'integer'],
    )
    def test_name_rule(self, name, expected):
        """A named target keeps its name, an unnamed one becomes ``'__target__'``."""
        assert resolve_target_column_name(pd.Series([1.0], name=name)) == expected


class TestAlignTargetIndex:
    """``y`` is repositioned onto ``X``'s index when that is safe, refused otherwise."""

    @staticmethod
    def _frames(unit_X='ns', unit_y='ns'):
        """Return ``(X, y)`` sharing three month-end instants, each at its unit."""
        dates = pd.date_range('2024-01-31', periods=3, freq='ME')
        X = pd.DataFrame({'m1': [1.0, 2.0, 3.0]}, index=dates.as_unit(unit_X).rename('date'))
        y = pd.Series([4.0, 5.0, 6.0], index=dates.as_unit(unit_y).rename('when'), name='y')
        return X, y

    def test_equal_indices_return_y_itself(self):
        """Equal indices at the same unit: ``y`` comes back untouched, the very object."""
        X, y = self._frames()
        assert align_target_index(X, y) is y

    def test_other_unit_takes_the_dates_of_X(self):
        """Same instants at another unit: ``y`` takes ``X``'s dates and keeps its level names."""
        X, y = self._frames(unit_X='s', unit_y='ns')
        aligned = align_target_index(X, y)
        # Unité de X, nom de niveau de y, valeurs inchangées
        assert (aligned.index.unit, aligned.index.name) == ('s', 'when')
        assert aligned.tolist() == [4.0, 5.0, 6.0]

    def test_pre_conversion_index_is_repositioned(self):
        """``y`` on the index ``X`` had before its columns became the index is repositioned."""
        X, _ = self._frames()
        y = pd.Series([4.0, 5.0, 6.0], name='y')
        metadata = {'index_was_replaced': True, 'original_index': y.index}
        aligned = align_target_index(X, y, metadata)
        assert aligned.index.equals(X.index)

    def test_metadata_without_replacement_is_not_used(self):
        """Metadata recording no index replacement repositions nothing."""
        X, _ = self._frames()
        y = pd.Series([4.0, 5.0, 6.0], name='y')
        metadata = {'index_was_replaced': False, 'original_index': y.index}
        with pytest.raises(ValueError, match='label'):
            align_target_index(X, y, metadata)

    def test_lengths_differ(self):
        """Different lengths are named as such."""
        X, y = self._frames()
        with pytest.raises(ValueError, match=r'lengths differ \(3 vs 2\)'):
            align_target_index(X, y.iloc[:2])

    def test_same_length_with_other_labels(self):
        """Same length, labels absent from ``y``: the message counts them."""
        X, y = self._frames()
        # Décalage d'un mois de la dernière date de y : une étiquette de X manque
        shifted = y.set_axis(y.index[:2].append(pd.DatetimeIndex(['2024-04-30'])))
        with pytest.raises(ValueError, match=r'same length \(3\) but 1 label'):
            align_target_index(X, shifted)


class TestDatetimeUnits:
    """The storage unit of the dates is read level by level."""

    @pytest.mark.parametrize(
        ('index', 'expected'),
        [
            (pd.date_range('2024-01-31', periods=2, freq='ME').as_unit('ms'), ('ms',)),
            (
                pd.MultiIndex.from_product(
                    [['FR'], pd.date_range('2024-01-31', periods=2, freq='ME').as_unit('s')]
                ),
                (None, 's'),
            ),
            (pd.RangeIndex(2), (None,)),
        ],
        ids=['flat', 'multiindex', 'not_dates'],
    )
    def test_units_per_level(self, index, expected):
        """One entry per level: the unit of a date level, None otherwise."""
        assert datetime_units(index) == expected


# =============================================================================
# Contrat d'entrée et fenêtres
# =============================================================================
class TestCheckReservedLevelName:
    """No index level, present or to come, may be named ``'frequency'``."""

    @pytest.mark.parametrize(
        ('index_name', 'panel_cols', 'time_col'),
        [
            ('frequency', None, None),
            (None, 'frequency', None),
            (None, ['country', 'frequency'], None),
            (None, None, 'frequency'),
        ],
        ids=['index_level', 'panel_cols_str', 'panel_cols_list', 'time_col'],
    )
    def test_reserved_name_is_refused(self, index_name, panel_cols, time_col):
        """The reserved name is refused wherever the future index takes it from."""
        frame = pd.DataFrame({'v': [1.0]}, index=pd.Index(['x'], name=index_name))
        with pytest.raises(ValueError, match="'frequency' is reserved"):
            check_reserved_level_name(frame, panel_cols=panel_cols, time_col=time_col)

    def test_other_names_pass(self):
        """A column named ``'frequency'`` that stays a column is no index level."""
        frame = pd.DataFrame({'frequency': ['M']}, index=pd.Index(['x'], name='date'))
        assert check_reserved_level_name(frame, panel_cols=['country'], time_col='date') is None


class TestMakeWindowCalculator:
    """The factory hands its four window parameters to an unfitted calculator."""

    def test_four_parameters_are_forwarded(self):
        """Each parameter reaches the calculator unchanged."""
        calculator = make_window_calculator(
            coverage_threshold=0.75,
            imputation_scope='extended_forward',
            training_scope='strict',
            training_coverage_threshold=0.25,
        )
        assert isinstance(calculator, ImputationWindowCalculator)
        assert (
            calculator.coverage_threshold, calculator.imputation_scope,
            calculator.training_scope, calculator.training_coverage_threshold,
        ) == (0.75, 'extended_forward', 'strict', 0.25)


class TestZipWindowBounds:
    """Window bounds are paired as ``(start, end)``, per entity on a panel."""

    def test_scalar_bounds(self):
        """A time series has one pair."""
        start, end = pd.Timestamp('2021-01-31'), pd.Timestamp('2021-12-31')
        assert zip_window_bounds(start, end) == (start, end)

    def test_per_entity_bounds_with_a_missing_end(self):
        """A panel has one pair per entity of the starts, None for a missing end."""
        start = {('FR',): pd.Timestamp('2021-01-31'), ('DE',): pd.Timestamp('2021-03-31')}
        end = {('FR',): pd.Timestamp('2021-12-31')}
        assert zip_window_bounds(start, end) == {
            ('FR',): (pd.Timestamp('2021-01-31'), pd.Timestamp('2021-12-31')),
            ('DE',): (pd.Timestamp('2021-03-31'), None),
        }


# =============================================================================
# Préparation du fit
# =============================================================================
class TestPrepareFit:
    """``fitting=True``: the working frame, the frequency plan, the windows, the additive fit."""

    def test_working_frame_merges_the_target(self, reference_timeseries):
        """``X_work`` is ``X`` plus the target, under the target's name."""
        y = _target(reference_timeseries)
        prepared = _prepare(_imputer(), reference_timeseries, y, fitting=True)
        expected = reference_timeseries.assign(y=y)
        pd.testing.assert_frame_equal(prepared.X_work, expected)
        assert prepared.target_column == 'y'

    @pytest.mark.parametrize(
        ('name', 'column'), [(None, '__target__'), ('cible é', 'cible é')],
        ids=['unnamed', 'accented'],
    )
    def test_target_column_name(self, reference_timeseries, name, column):
        """An unnamed target is merged as ``'__target__'``, an accented name is kept."""
        prepared = _prepare(
            _imputer(), reference_timeseries, _target(reference_timeseries, name=name),
            fitting=True,
        )
        assert prepared.target_column == column
        assert column in prepared.X_work.columns

    def test_without_target_the_frame_is_a_copy_of_X(self, reference_timeseries):
        """Without ``y``: no target column, a working frame equal to ``X`` yet distinct."""
        prepared = _prepare(_imputer(), reference_timeseries, fitting=True)
        pd.testing.assert_frame_equal(prepared.X_work, reference_timeseries)
        assert prepared.X_work is not reference_timeseries
        assert (prepared.target_column, prepared.y, prepared.snapshot_y) == (None, None, None)

    def test_snapshots_copy_the_input(self, reference_timeseries):
        """The snapshots equal the input of the pass, as copies."""
        y = _target(reference_timeseries)
        prepared = _prepare(_imputer(), reference_timeseries, y, fitting=True)
        pd.testing.assert_frame_equal(prepared.snapshot_X, reference_timeseries)
        pd.testing.assert_series_equal(prepared.snapshot_y, y)
        assert prepared.snapshot_X is not reference_timeseries
        assert prepared.snapshot_y is not y

    def test_target_on_the_pre_conversion_index(self, reference_timeseries):
        """A target on the flat index of column-based data is repositioned onto ``X``."""
        imputer = _imputer(time_col='date')
        flat = reference_timeseries.reset_index()
        # Validation de base : la colonne temporelle passe en index et la
        # conversion est consignée pour l'appel en cours
        X = imputer._validate_input(flat)
        y = _target(flat)
        prepared = _prepare(imputer, X, y, fitting=True)
        assert prepared.y.index.equals(X.index)
        assert prepared.X_work['y'].tolist() == y.tolist()

    def test_layout_is_the_frequency_plan_of_the_working_frame(self, reference_timeseries):
        """The layout is ``plan_frequencies`` run on the frame before the additive transformer.

        The transformer empties ``a1`` entirely: a plan computed after it would
        find no frequency for that column.
        """
        prepared = _prepare(
            _imputer(additive_transformer=_BlankAfter(after='2020-12-31')),
            reference_timeseries,
            fitting=True,
        )
        expected = plan_frequencies(
            reference_timeseries, 'M', is_panel=False, entities=None,
            impute_intermediate_frequencies=False, on_frequency_mismatch='error',
        )
        assert prepared.fit.layout == expected

    def test_time_series_structure(self, reference_timeseries):
        """A time series: no panel, no entity, a monthly input index."""
        fit = _prepare(_imputer(), reference_timeseries, fitting=True).fit
        assert (fit.is_panel, fit.entities, fit.source_index_frequency_label) == (
            False, None, 'M'
        )

    def test_panel_structure(self, mixed_freq_panel_multifrequency):
        """A panel: its entities, sorted, and a monthly input index."""
        fit = _prepare(_imputer(), mixed_freq_panel_multifrequency, fitting=True).fit
        assert fit.is_panel
        assert fit.entities == [('DE',), ('FR',), ('IT',)]
        assert fit.source_index_frequency_label == 'M'

    def test_undetectable_index_has_no_label(self, reference_timeseries, monkeypatch):
        """An input index whose frequency cannot be detected yields no label, never an error.

        The detection of the index is robust enough to label every index a fit
        accepts; its failure is simulated.
        """
        def _undetectable(index, return_format='base'):
            raise ValueError('index too short')

        monkeypatch.setattr(preparation, 'detect_index_frequency', _undetectable)
        fit = _prepare(_imputer(), reference_timeseries, fitting=True).fit
        assert fit.source_index_frequency_label is None

    @pytest.mark.parametrize('kind', _MASK_KINDS)
    def test_masks_are_those_of_the_calculator(self, mixed_freq_panel_multifrequency, kind):
        """Each mask of the fit is the calculator's mask of its kind on the working frame."""
        prepared = _prepare(_imputer(), mixed_freq_panel_multifrequency, fitting=True)
        expected = prepared.window_calculator.get_imputation_window_mask(
            mixed_freq_panel_multifrequency, kind=kind
        )
        pd.testing.assert_series_equal(getattr(prepared.fit, f'{kind}_window_mask'), expected)

    def test_bounds_pair_the_calculator_bounds(self, mixed_freq_panel_multifrequency):
        """The readable bounds pair the calculator's starts and ends of each window."""
        prepared = _prepare(_imputer(), mixed_freq_panel_multifrequency, fitting=True)
        calculator = prepared.window_calculator
        assert prepared.fit.imputation_window == zip_window_bounds(
            calculator.imputation_window_start_, calculator.imputation_window_end_
        )
        assert prepared.fit.training_window == zip_window_bounds(
            calculator.training_window_start_, calculator.training_window_end_
        )

    def test_additive_transformer_is_a_fitted_clone(self, reference_timeseries):
        """The parameter stays untouched; a fitted clone transforms the working frame."""
        parameter = _BlankAfter()
        prepared = _prepare(
            _imputer(additive_transformer=parameter), reference_timeseries, fitting=True
        )
        fitted = prepared.fit.additive_transformer
        assert fitted is not parameter
        assert not hasattr(parameter, 'last_input_')
        # Valeur d'or : a1 vidée après 2022, le reste de l'entrée inchangé
        expected = reference_timeseries.copy()
        expected.loc[expected.index > '2022-12-31', 'a1'] = np.nan
        pd.testing.assert_frame_equal(prepared.X_work, expected)

    def test_no_additive_transformer(self, reference_timeseries):
        """Without a transformer, none is recorded."""
        prepared = _prepare(_imputer(), reference_timeseries, fitting=True)
        assert prepared.fit.additive_transformer is None

    def test_pair_returned_by_the_transformer_is_unpacked(self, reference_timeseries):
        """A transformer returning the ``(X, y)`` pair hands its ``X`` to the pass."""
        prepared = _prepare(
            _imputer(additive_transformer=_PairReturningIdentity()), reference_timeseries,
            fitting=True,
        )
        pd.testing.assert_frame_equal(prepared.X_work, reference_timeseries)

    def test_tracker_scans_the_transformed_frame(self, reference_timeseries):
        """The tracker reads the frame the transformer produced: it starts after it."""
        prepared = _prepare(
            _imputer(additive_transformer=_BlankAfter()), reference_timeseries, fitting=True
        )
        provenance = prepared.provenance_tracker.get_provenance_matrix()
        # Valeur d'or : l'ancre 2022 de a1 reste originale ; celle de 2023,
        # observée en entrée mais vidée par le transformateur, n'est pas marquée
        assert provenance.loc['2022-12-31', 'a1'] == ProvenanceType.ORIGINAL
        assert pd.isna(provenance.loc['2023-12-31', 'a1'])

    def test_window_failure_raises(self, reference_timeseries, monkeypatch):
        """A window that cannot be computed fails the fit."""
        def _failing_fit(self, data):
            raise ValueError('no window')

        monkeypatch.setattr(ImputationWindowCalculator, 'fit', _failing_fit)
        with pytest.raises(ValueError, match='no window'):
            _prepare(_imputer(), reference_timeseries, fitting=True)

    def test_reserved_level_name_is_refused(self, reference_timeseries):
        """An input index level named ``'frequency'`` is refused."""
        with pytest.raises(ValueError, match="'frequency' is reserved"):
            _prepare(_imputer(), reference_timeseries.rename_axis('frequency'), fitting=True)

    def test_prediction_is_refused_at_fit(self, reference_timeseries):
        """The prediction pass is a transform pass."""
        with pytest.raises(ValueError, match='prediction pass is a transform pass'):
            prepare_work_frame(_imputer(), reference_timeseries, fitting=True, prediction=True)

    def test_nothing_is_written_on_the_imputer(self, mixed_freq_panel_multifrequency):
        """The preparation publishes nothing: no attribute added or rebound."""
        imputer = _imputer(additive_transformer=_BlankAfter())
        before = _attribute_identities(imputer)
        _prepare(imputer, mixed_freq_panel_multifrequency, fitting=True)
        assert _attribute_identities(imputer) == before


# =============================================================================
# Préparation du transform
# =============================================================================
class TestPrepareTransform:
    """``fitting=False``: the same steps, on the fitted state, plus the frequency check."""

    def test_fit_data_give_the_frame_and_windows_of_the_fit(self, reference_timeseries):
        """On the fit data, the transform prepares the frame and the windows of the fit.

        The transformer empties ``a1`` in 2023: a window computed after it
        would stop in 2022. Both passes compute it before.
        """
        params = dict(additive_transformer=_BlankAfter())
        fit = _prepare(_imputer(**params), reference_timeseries, fitting=True)
        replay = _prepare(
            _fitted(reference_timeseries, **params), reference_timeseries, fitting=False
        )

        pd.testing.assert_frame_equal(replay.X_work, fit.X_work)
        for kind in _MASK_KINDS:
            pd.testing.assert_series_equal(
                replay.window_calculator.get_imputation_window_mask(
                    reference_timeseries, kind=kind
                ),
                getattr(fit.fit, f'{kind}_window_mask'),
            )
        # Le test discrimine : la fenêtre de la frame transformée diffère
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            after_additive = make_window_calculator(
                coverage_threshold=0.5, imputation_scope='strict',
                training_scope=None, training_coverage_threshold=None,
            ).fit(fit.X_work)
        assert not after_additive.get_imputation_window_mask(fit.X_work, kind='strict').equals(
            fit.fit.strict_window_mask
        )

    def test_fit_part_and_check_results(self, reference_timeseries):
        """No fit part, and nothing new on the fit data."""
        replay = _prepare(_fitted(reference_timeseries), reference_timeseries, fitting=False)
        assert replay.fit is None
        assert (replay.new_frequencies, replay.new_misaligned, replay.skipped_columns) == (
            {}, (), frozenset()
        )

    def test_training_replay_merges_the_target(self, reference_timeseries):
        """Outside the prediction pass, the target is merged and snapshotted as at fit."""
        y = _target(reference_timeseries)
        replay = _prepare(_fitted(reference_timeseries, y), reference_timeseries, y, fitting=False)
        assert replay.target_column == 'y'
        pd.testing.assert_series_equal(replay.snapshot_y, y)

    def test_prediction_pass_leaves_the_target_out(self, reference_timeseries):
        """The prediction pass neither merges nor snapshots ``y``, and skips its column."""
        y = _target(reference_timeseries)
        imputer = _fitted(reference_timeseries, y)
        replay = _prepare(imputer, reference_timeseries, y, fitting=False, prediction=True)

        assert replay.target_column is None
        assert 'y' not in replay.X_work.columns
        assert replay.snapshot_y is None
        assert replay.skipped_columns == frozenset({'y'})
        # y rendu tel quel : index déjà égaux, même unité
        assert replay.y is y

    def test_prediction_pass_hands_the_target_empty_to_the_transformer(
        self, reference_timeseries
    ):
        """The transformer fitted with the target receives it empty, and it is dropped."""
        y = _target(reference_timeseries)
        imputer = _fitted(reference_timeseries, y, additive_transformer=_BlankAfter())
        replay = _prepare(imputer, reference_timeseries, y, fitting=False, prediction=True)

        received = imputer.additive_transformer_.last_input_
        assert received.columns.tolist() == ['m1', 'q1', 'a1', 'a2', 'y']
        assert bool(received['y'].isna().all())
        assert 'y' not in replay.X_work.columns

    def test_missing_column_of_the_fit_raises(self, reference_timeseries):
        """A column of the fit absent from the transform is named in the error."""
        imputer = _fitted(reference_timeseries)
        with pytest.raises(ValueError, match=r"missing 1 column\(s\) seen at fit time: \['a2'\]"):
            prepare_work_frame(
                imputer, reference_timeseries.drop(columns='a2'), fitting=False
            )

    def test_diverging_frequency_warns_once(self, reference_timeseries):
        """A pair observed at another frequency than at fit warns once."""
        imputer = _fitted(reference_timeseries)
        # q1 publiée en décembre seulement : annuelle au transform, trimestrielle au fit
        diverging = reference_timeseries.copy()
        diverging['q1'] = np.nan
        diverging.loc[diverging.index.month == 12, 'q1'] = [30.0, 90.0, 150.0]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            prepare_work_frame(imputer, diverging, fitting=False)
        messages = [str(item.message) for item in caught if 'different frequency' in str(item.message)]
        assert len(messages) == 1
        assert "'q1'" in messages[0]

    def test_pairs_unknown_to_the_fit_are_detected(self, mixed_freq_panel_multifrequency):
        """The pairs of an entity the fit never saw carry the frequency detected on them."""
        imputer = _fitted(drop_entity(mixed_freq_panel_multifrequency, 'IT'))
        replay = _prepare(imputer, mixed_freq_panel_multifrequency, fitting=False)
        assert set(replay.new_frequencies) == {('IT', 'm1'), ('IT', 'q1'), ('IT', 'v')}

    def test_window_failure_warns_and_restricts_nothing(self, reference_timeseries):
        """A single row carries no window: warning, and an unfitted calculator."""
        imputer = _fitted(reference_timeseries)
        with pytest.warns(UserWarning, match='Could not calculate the imputation window'):
            replay = prepare_work_frame(imputer, reference_timeseries.iloc[:1], fitting=False)
        assert replay.window_calculator.imputation_window_mask_ is None

    def test_rows_outside_the_window_warn(self, reference_timeseries):
        """Rows the recomputed window leaves out are announced once."""
        imputer = _fitted(reference_timeseries)
        # Couverture de a1 et a2 retirée en 2023 : la fenêtre recalculée s'arrête avant
        shortened = reference_timeseries.copy()
        shortened.loc['2023', ['a1', 'a2']] = np.nan
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            prepare_work_frame(imputer, shortened, fitting=False)
        assert sum('fall outside the imputation window' in str(w.message) for w in caught) == 1

    def test_series_is_refused(self, reference_timeseries):
        """``X`` must be a DataFrame."""
        imputer = _fitted(reference_timeseries)
        with pytest.raises(ValueError, match='X must be a pandas DataFrame, got Series'):
            prepare_work_frame(imputer, reference_timeseries['m1'], fitting=False)

    def test_fitted_state_is_left_intact(self, mixed_freq_panel_multifrequency):
        """The preparation of a transform rebinds no attribute and adds none."""
        imputer = _fitted(
            drop_entity(mixed_freq_panel_multifrequency, 'IT'),
            additive_transformer=_PairReturningIdentity(),
        )
        before = _attribute_identities(imputer)
        _prepare(imputer, mixed_freq_panel_multifrequency, fitting=False)
        assert _attribute_identities(imputer) == before


# =============================================================================
# Résolution des dates
# =============================================================================
class TestDatetimeResolution:
    """The working frame comes out at the unit of ``X``, whatever the unit of ``y``."""

    @pytest.mark.parametrize('unit', DATETIME_UNITS)
    @pytest.mark.parametrize('fitting', [True, False], ids=['fit', 'transform'])
    def test_frame_at_the_unit_of_X(self, reference_timeseries, unit, fitting):
        """``X`` at ``unit``, ``y`` in nanoseconds: the frame and ``y`` are at ``unit``."""
        X = with_datetime_resolution(reference_timeseries, unit)
        y = with_datetime_resolution(_target(reference_timeseries), 'ns')
        imputer = _fitted(X, y) if not fitting else _imputer()
        prepared = _prepare(imputer, X, y, fitting=fitting)
        assert datetime_units_of(prepared.X_work) == {('index', 0): unit}
        assert prepared.y.index.unit == unit
