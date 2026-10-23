"""Panel data tools: ``PanelwiseTransformer`` and the entity helpers of ``panel.utils``.

Re-exports the per-entity transformer and every public function of ``panel/utils.py``
(panel detection, entity keys and their integer codes, entity masks and blocks, per-entity
frequencies).
"""
# Importation des modules
from .transformers import PanelwiseTransformer
from .utils import (
    is_panel_data,
    get_entity_levels,
    detect_panel_structure,
    extract_time_series_from_multiindex,
    normalize_entity_key,
    factorize_row_keys,
    get_entity_codes,
    normalize_entity_mapping,
    split_variable_key,
    get_unique_panel_entities,
    group_keys_by_entity_and_variable,
    extract_column_names,
    get_entity_mask,
    get_entity_target_frequency,
    resolve_entity_column_frequencies,
    iter_entity_blocks,
    build_panel_index,
)

# Exportation des éléments du module
__all__ = [
    'PanelwiseTransformer',
    'is_panel_data',
    'get_entity_levels',
    'detect_panel_structure',
    'extract_time_series_from_multiindex',
    'normalize_entity_key',
    'factorize_row_keys',
    'get_entity_codes',
    'normalize_entity_mapping',
    'split_variable_key',
    'get_unique_panel_entities',
    'group_keys_by_entity_and_variable',
    'extract_column_names',
    'get_entity_mask',
    'get_entity_target_frequency',
    'resolve_entity_column_frequencies',
    'iter_entity_blocks',
    'build_panel_index',
]
