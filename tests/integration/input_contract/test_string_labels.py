"""Entity names, column names and index level names held in the ``str`` dtype come back unchanged.

Panels whose entity level and columns are stored either as ``object`` or as the
NaN-backed string dtype (``pd.StringDtype(na_value=np.nan)``, the default
string dtype of pandas 3), with entity and column names holding spaces,
accents, slashes, parentheses and percent signs: ``PanelwiseTransformer``
(round trip through ``inverse_transform``), the per-entity delay factory,
``HighFrequencyImputer`` (stacked output and inversion),
``compare_and_detect_delays``, ``detect_dataset_frequency``,
``validate_temporal_data`` on the column path and the panel splitters (test
dates given as ``(entity, date)`` tuples) keep the labels, their order and the
level names. The text these components produce (delay frequencies, units and
reference points, empty delay results included, applicable delays, the
detection report's index, the imputer's ``frequency`` level) is in the ``str``
dtype, and the imputer's provenance holds ``ProvenanceType`` members.
"""
# Modules de base
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from tests.support.datasets import build_mixed_frequency_panel, build_panel_reference
from tests.support.perturbations import string_dtypes
from tsforecast.crossvals import PanelOutOfSampleSplit, PanelOutOfSampleSplitPerEntity
from tsforecast.delays import calculate_applicable_delay, compare_and_detect_delays, create_delay_transformer_factory
from tsforecast.frequency import PROVENANCE_DTYPE, HighFrequencyImputer, ProvenanceType
from tsforecast.panel import PanelwiseTransformer
from tsforecast.utils.frequency import detect_dataset_frequency
from tsforecast.utils.validation import validate_temporal_data

# Dtype chaîne à valeur manquante NaN (dtype chaîne par défaut de pandas 3)
STR = pd.StringDtype(na_value=np.nan)

# Noms d'entités à caractères spéciaux, dans l'ordre des blocs
ENTITIES = {'France': 'Côte d’Ivoire', 'Allemagne': 'A/B (n°2)', 'Italie': 'É-1 % x'}

# Noms de colonnes à caractères spéciaux
COLUMNS = {
    'production_industrielle': 'production (brute)',
    'inflation_ipc': 'inflation %',
    'taux_chomage': 'chômage/mois',
    'pib_trimestriel': 'PIB trimestriel (n°2)',
    'balance_commerciale_annuelle': 'balance à l’export',
}


def _quiet(function, *args, **kwargs):
    """Call ``function`` with the warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


def _relabel(panel: pd.DataFrame, entities: dict, columns: dict, dtype) -> pd.DataFrame:
    """Rename entities and columns of a two-level panel, the labels stored in ``dtype``."""
    entity_level = pd.Index([entities[label] for label in panel.index.get_level_values(0)], dtype=dtype)
    out = panel.copy()
    out.index = pd.MultiIndex.from_arrays(
        [entity_level, panel.index.get_level_values(1)], names=['pays (zone)', 'date']
    )
    out.columns = pd.Index([columns.get(col, col) for col in panel.columns], dtype=dtype)
    return out


@pytest.fixture(params=['object', 'str'])
def label_dtype(request):
    """Dtype of the labels: ``object`` or the NaN-backed string dtype."""
    return object if request.param == 'object' else STR


@pytest.fixture
def panel(label_dtype) -> pd.DataFrame:
    """Mixed-frequency panel of three countries with special entity and column names."""
    return _relabel(build_mixed_frequency_panel(), ENTITIES, COLUMNS, label_dtype)


def _assert_same_labels(result: pd.Index, expected: pd.Index) -> None:
    """Same labels in the same order and same names, whatever the dtype of the labels."""
    pd.testing.assert_index_equal(result, expected, exact=False, check_exact=True)
    assert list(result.names) == list(expected.names)


class TestPanelTransformersRestoreLabels:
    """Panel transformers give back the index and the columns they received."""

    def test_panelwise_round_trip(self, panel):
        """``PanelwiseTransformer`` around a scaler: output and inversion carry the input labels."""
        transformer = PanelwiseTransformer(StandardScaler())

        output = _quiet(transformer.fit_transform, panel.fillna(0.0))
        inverse = _quiet(transformer.inverse_transform, output)

        _assert_same_labels(output.index, panel.index)
        _assert_same_labels(output.columns, panel.columns)
        _assert_same_labels(inverse.index, panel.index)
        # Inversion du centrage-réduction : égalité au bruit d'arrondi près
        np.testing.assert_allclose(inverse.to_numpy(), panel.fillna(0.0).to_numpy(), rtol=1e-12, atol=1e-9)

    def test_delay_factory(self, panel):
        """Per-entity delays keyed by special names reach their entity; labels are kept."""
        # Un délai par entité (la fabrique exige chaque entité du panel)
        delays = pd.DataFrame(
            {'delay': [20.0, 50.0, 50.0], 'unit': 'D', 'reference_point': 'start', 'frequency': 'Q'},
            index=pd.MultiIndex.from_tuples(
                [(ENTITIES['France'], COLUMNS['inflation_ipc']), (ENTITIES['Allemagne'], COLUMNS['taux_chomage']),
                 (ENTITIES['Italie'], COLUMNS['taux_chomage'])],
                names=['pays (zone)', 'column'],
            ),
        )
        factory = create_delay_transformer_factory(delays, strategy='mask', prediction_date='2024-08-15')
        transformer = PanelwiseTransformer(transformer=factory, time_col=None, panel_cols=None)

        output = _quiet(transformer.fit_transform, panel)

        _assert_same_labels(output.index, panel.index)
        _assert_same_labels(output.columns, panel.columns)
        # Masque appliqué à la seule entité visée : la dernière valeur d'inflation de la France disparaît
        french_inflation = output.xs(ENTITIES['France'], level=0)[COLUMNS['inflation_ipc']]
        assert french_inflation.notna().sum() < panel.xs(ENTITIES['France'], level=0)[COLUMNS['inflation_ipc']].notna().sum()

    def test_imputer_output_and_inversion(self, label_dtype):
        """``HighFrequencyImputer``: entities and columns of the output are the input ones."""
        reference = build_panel_reference()[['m1', 'q1', 'v']]
        data = _relabel(reference, {'FR': 'Côte d’Ivoire', 'DE': 'A/B (n°2)', 'IT': 'É-1 % x'},
                        {'m1': 'mensuel (brut)', 'q1': 'trimestriel %', 'v': 'v/an'}, label_dtype)
        imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                       restore_original_values=True)

        output = _quiet(imputer.fit_transform, data)
        inverse = _quiet(imputer.inverse_transform, output)

        assert list(dict.fromkeys(output.index.get_level_values(0))) == list(dict.fromkeys(data.index.get_level_values(0)))
        assert output.index.names[0] == 'pays (zone)'
        assert list(output.columns) == list(data.columns)
        _assert_same_labels(inverse.index, data.index)
        _assert_same_labels(inverse.columns, data.columns)


class TestFunctionsKeepLabels:
    """Detection, validation and delay functions report the labels they received."""

    def test_detected_frequencies_are_keyed_by_the_labels(self, panel):
        """Each (entity, column) couple is a key, written exactly as in the panel."""
        frequencies = detect_dataset_frequency(panel)

        assert set(frequencies) == {(entity, column) for entity in ENTITIES.values() for column in COLUMNS.values()}

    def test_detected_delays_are_indexed_by_the_labels(self, panel):
        """The delay table is indexed by (entity, column) with the special names."""
        delays = _quiet(compare_and_detect_delays, panel, None, '2024-10-15')

        assert set(delays.index.get_level_values(0)) == set(ENTITIES.values())
        assert set(delays.index.get_level_values(-1)) <= set(COLUMNS.values())
        assert delays.index.names == ['pays (zone)', 'column']

    def test_column_path_validation(self, panel):
        """Entity and date given as columns: the validated panel carries the same index."""
        flat = panel.reset_index()

        validated = _quiet(validate_temporal_data, flat, time_col='date', panel_cols=['pays (zone)'])

        _assert_same_labels(validated.index, panel.index)
        _assert_same_labels(validated.columns, panel.columns)


class TestSplittersFindTheEntities:
    """Test dates given as ``(entity, date)`` tuples reach the entity whatever the label dtype."""

    @pytest.mark.parametrize('splitter_class', [PanelOutOfSampleSplit, PanelOutOfSampleSplitPerEntity],
                             ids=['global', 'per-entity'])
    def test_tuple_test_indices(self, panel, splitter_class):
        """The test fold holds exactly the requested entity's row."""
        target = (ENTITIES['Allemagne'], '2023-06-01')
        splitter = splitter_class(test_indices=[target])

        folds = _quiet(lambda: list(splitter.split(panel)))

        tested = {panel.index[position] for _, test in folds for position in test}
        assert tested == {(ENTITIES['Allemagne'], pd.Timestamp('2023-06-01'))}


class TestOutputStringDtypes:
    """Text produced from special labels is ``str``; labels keep their dtype, empty result or not."""

    def test_delay_tables_and_report(self, panel, label_dtype):
        """Delays, empty delays, applicable delays and the detection report, labels kept."""
        result, report = _quiet(compare_and_detect_delays, panel, None, '2024-10-15', return_report=True)
        empty = _quiet(compare_and_detect_delays, panel * np.nan, None, '2024-10-15')
        applicable = calculate_applicable_delay(result, 'end', 'quarterly', aggregate_by_panel=True)

        expected_entity = 'str' if label_dtype is STR else 'object'
        text = {('column', name): 'str' for name in ('frequency', 'reference_point', 'unit')}
        expected = {('index', 0): expected_entity, ('index', 1): 'str', **text}
        assert string_dtypes(result) == expected and string_dtypes(empty) == expected
        assert {key: kind for key, kind in string_dtypes(applicable).items() if key[0] == 'column'} == {
            **text, ('column', 'aggregation_method'): 'str'
        }
        frame = report.to_frame()
        assert frame.index.dtype == 'str' and set(frame.index) == set(COLUMNS.values())

    def test_imputer_frequency_level_and_provenance(self, label_dtype):
        """The stacked output's ``frequency`` level is ``str``; the provenance is categorical."""
        reference = build_panel_reference()[['m1', 'q1', 'v']]
        data = _relabel(reference, {'FR': 'Côte d’Ivoire', 'DE': 'A/B (n°2)', 'IT': 'É-1 % x'},
                        {'m1': 'mensuel (brut)', 'q1': 'trimestriel %', 'v': 'v/an'}, label_dtype)
        imputer = HighFrequencyImputer(target_frequency='M', estimator=LinearRegression(),
                                       impute_intermediate_frequencies=True)

        output = _quiet(imputer.fit_transform, data)

        assert output.index.get_level_values('frequency').dtype == 'str'
        assert list(imputer.imputation_provenance_.columns) == list(data.columns)
        assert (imputer.imputation_provenance_.dtypes == PROVENANCE_DTYPE).all()
        cells = [value for value in imputer.imputation_provenance_.to_numpy().ravel() if not pd.isna(value)]
        assert cells and set(cells) <= {member.value for member in ProvenanceType}
