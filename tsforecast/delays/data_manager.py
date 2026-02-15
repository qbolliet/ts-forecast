"""Data manager for comparing and storing publication delays information.

This module provides tools for comparing new datasets with existing data
and managing publication delay information using pandas DataFrames.
"""
# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple, List, Any, Literal
from datetime import datetime, timedelta
import warnings
# Module de détection de la fréquence des séries
from ..frequency.detector import detect_frequency
# Module de validation des données temporelles
from ..utils.validation import validate_temporal_data
# Module de manipulation temporelle
from ..utils.time import resolve_date, get_period_boundaries

# /!\ Faire un prompt pour intégrer un logger à cette fonction : comment mettre du logging optionnel + implémentation
# Fonction de comparaison et d'inférence des délais de publication
def compare_and_detect_delays(new_data: pd.DataFrame, existing_data: Optional[pd.DataFrame] = None, download_date: Union[str, datetime] = None, detection_mode: str = 'new_only', reference_point: str = 'start', delay_unit: Literal['us', 's', 'D', 'microsecond', 'second', 'day'] = 'day', time_col: Optional[str] = None, panel_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Compare new data with existing data and detect publication delays.

    This function identifies new or changed observations by comparing new_data with
    existing_data (if provided), then calculates publication delays for these observations.
    
    When existing_data is None, the function identifies the most recent non-null observation
    for each variable (and each panel entity if panel data is provided).

    Args:
        new_data: New data DataFrame to analyze
        existing_data: Existing data DataFrame for comparison. If None, identifies the most
            recent non-null observation for each variable/entity
        download_date: Date when the data was downloaded. If None, uses current datetime
        detection_mode: Detection mode - 'new_only' (only null→non-null transitions) or
            'all_changes' (all value changes). Only used when existing_data is provided
        reference_point: Reference point for delay calculation - 'start' or 'end' of the period
        delay_unit: Unit for delay calculation - 'day'/'D', 'second'/'s', or 'microsecond'/'us'
        time_col: Name of the time column (optional)
        panel_cols: List of panel column names for panel data (optional)

    Returns:
        DataFrame containing detected observations with publication delay information
        It contains the following columns:
        - 'observation_date': The date of the observation
        - 'column': The column name
        - 'download_date': Date when data was downloaded
        - 'frequency': Detected frequency of the data
        - 'period_start': Start boundary of the period
        - 'period_end': End boundary of the period
        - 'reference_point': Reference point used ('start' or 'end')
        - 'delay': Calculated publication delay (ceil-rounded)
        - 'unit': Unit of the delay value

    Raises:
        ValueError: If invalid parameters are provided
    """
    # Validation des arguments
    # Validation des jeux de données
    new_data = _validate_input_data(data=new_data, time_col=time_col, panel_cols=panel_cols)
    if existing_data is not None:
        existing_data = _validate_input_data(data=existing_data, time_col=time_col, panel_cols=panel_cols)
    
    # Validation de la date de téléchargement
    if download_date is None:
        download_date = datetime.now()
    download_date = resolve_date(date=download_date)
    
    # Validation du point de référence
    if reference_point not in ['start', 'end']:
        raise ValueError("reference_point must be 'start' or 'end'")
    # Validation du mode de détection
    if detection_mode not in ['new_only', 'all_changes']:
        raise ValueError("detection_mode must be 'new_only' or 'all_changes'")

    # Identification des nouvelles observations
    new_observations = _identify_new_observations(
        new_data=new_data, 
        existing_data=existing_data, 
        detection_mode=detection_mode
    )

    # Fonction de calcul des délais associés aux observations nouvellement publiées
    new_observations = _calculate_publication_delays(
        new_observations=new_observations,
        new_data=new_data,
        download_date=download_date,
        reference_point=reference_point,
        unit=delay_unit
    )

    return new_observations


# Fonction auxiliaire de validation des jeux de données en entrée
def _validate_input_data(data: pd.DataFrame, time_col: Optional[str] = None, panel_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Validate and prepare input data for analysis.

    This function uses the validate_temporal_data() function to validate and prepare
    time series or panel data structures.

    Args:
        data: Input pandas DataFrame to validate
        time_col: Name of the time column (optional)
        panel_cols: List of panel column names (optional)

    Returns:
        Validated and sorted DataFrame

    Raises:
        ValueError: If validation fails or invalid parameter combination
    """
    # Utilisation de validate_temporal_data pour valider les données
    data_validated = validate_temporal_data(
        data=data,
        time_col=time_col,
        panel_cols=panel_cols,
        strict=True,
        sort_data=True,
        return_metadata=False  # Pas besoin de métadonnées pour la reversion ici
    )

    return data_validated


# Fonction auxiliaire d'identification des nouvelles observations
def _identify_new_observations(new_data: pd.DataFrame, existing_data: Optional[pd.DataFrame] = None, detection_mode: str = 'new_only') -> pd.DataFrame:
    """Identify new or changed observations.

    When existing_data is provided, compares the two DataFrames to identify changes.
    When existing_data is None, identifies the most recent non-null observation for
    each variable (and each panel entity if panel data).

    Args:
        new_data: New data DataFrame
        existing_data: Existing data DataFrame for comparison. If None, identifies the
            most recent non-null observation for each variable/entity
        detection_mode: 'new_only' or 'all_changes'. Only used when existing_data is provided

    Returns:
        DataFrame containing only new or changed observations with columns 'column' and 'has_changes'
    """
    # Cas où existing_data n'est pas fourni : identification des observations les plus récentes
    if existing_data is None:
        # Initialisation de la liste des résultats
        results = []
        
        # Obtention des niveaux d'index (pour distinguer time vs panel)
        is_panel = isinstance(new_data.index, pd.MultiIndex)
        
        # Identification des colonnes de valeurs (exclure les niveaux d'index)
        value_columns = [col for col in new_data.columns]
        
        # Traitement selon la structure des données (panel ou non)
        if is_panel:
            # Cas panel data : l'index est multi-niveau (entités + temps)
            # Groupement par toutes les dimensions sauf la dernière (qui est le temps)
            panel_levels = list(range(new_data.index.nlevels - 1))
            
            # Parcours des colonnes
            for col in value_columns:
                # Pour chaque colonne, groupement par entité et identification de la dernière observation non-nulle
                grouped = new_data.groupby(level=panel_levels)
                # Parcours des groupes
                for _, group_data in grouped:
                    # Filtre des valeurs non-nulles
                    non_null_mask = group_data[col].notna()
                    non_null_data = group_data[non_null_mask]
                    
                    # Si des observations non-nulles existent, extraction de la plus récente
                    if len(non_null_data) > 0:
                        last_idx = non_null_data.index[-1]  # Dernière observation (données triées)
                        results.append({
                            'index': last_idx,
                            'column': col,
                            'has_changes': True
                        })
        else:
            # Cas série temporelle simple : l'index est uni-niveau (temps uniquement)
            for col in value_columns:
                # Filtre des valeurs non-nulles
                non_null_mask = new_data[col].notna()
                non_null_data = new_data[non_null_mask]
                
                # Si des observations non-nulles existent, extraction de la plus récente
                if len(non_null_data) > 0:
                    last_idx = non_null_data.index[-1]  # Dernière observation (données triées)
                    results.append({
                        'index': last_idx,
                        'column': col,
                        'has_changes': True
                    })
        
        # Création du DataFrame de résultats
        if results:
            df_changes = pd.DataFrame(results)
            df_changes = df_changes.set_index('index')
            # Création d'un multiindex s'il s'agit de données de panel
            if is_panel:
                df_changes.index = pd.MultiIndex.from_tuples(df_changes.index, names=new_data.index.names)
            else:
                df_changes.index.names = new_data.index.names
        else:
            # Aucune observation non-nulle trouvée
            df_changes = pd.DataFrame(columns=['column', 'has_changes'])
            df_changes.index.names = new_data.index.names
        
        return df_changes
    else :
        # Cas où existing_data est fourni : comportement existant
        # Vérification que les deux DataFrames ont des colonnes en commun
        common_columns = list(set(new_data.columns).intersection(set(existing_data.columns)))

        # Restriction aux colonnes communes
        new_common_data = new_data[common_columns]
        existing_common_data = existing_data[common_columns]

        # Alignement des DataFrames sur l'index commun
        aligned_new, aligned_existing = new_common_data.align(existing_common_data, fill_value=np.nan)

        # Création des masques booléens pour les valeurs nulles
        new_isnull = aligned_new.isnull()
        existing_isnull = aligned_existing.isnull()

        # Détection des changements
        if detection_mode == 'new_only':
            # Détection uniquement des valeurs null → non-null
            # Condition: valeur existante était null ET nouvelle valeur n'est pas null
            changes_mask = existing_isnull & ~new_isnull
        else:  # 'all_changes'
            # Détection de tous les changements (null → non-null ET valeur → nouvelle valeur)
            # Condition: (ancien null ET nouveau non-null) OU (valeurs différentes)
            changes_mask = (existing_isnull & ~new_isnull) | (
                ~new_isnull & ~existing_isnull & (aligned_new != aligned_existing)
            )

        # Mise en forme du jeu de données
        df_changes = pd.melt(changes_mask, var_name="column", value_name="has_changes", ignore_index=False)
        # Filtre sur les changements
        df_changes = df_changes.loc[df_changes["has_changes"]]

        return df_changes


# Fonction de calcul des délais de publication
def _calculate_publication_delays(new_observations: pd.DataFrame,
                                new_data: pd.DataFrame,
                                download_date: datetime,
                                reference_point: Literal['start', 'end'],
                                unit: Literal['us', 's', 'D', 'microsecond', 'second', 'day']
                                ) -> pd.DataFrame :
    """Calculate publication delays for observed data points.

    Computes the delay between the observation date (period start or end) and the
    download date. Also enriches the observations with metadata including frequency,
    period boundaries, and delay values in the specified unit.

    Args:
        new_observations: DataFrame with detected observations, indexed by datetime/multi-level
            and containing 'column' and 'has_changes' columns
        new_data: Original DataFrame containing the time series data
        download_date: Date when the data was downloaded (as datetime object)
        reference_point: Point used for delay calculation - 'start' (period start) or 'end'
            (period end)
        unit: Unit for delay values - 'day'/'D', 'second'/'s', or 'microsecond'/'us'

    Returns:
        DataFrame with computed publication delays containing columns:
        - 'observation_date': The date of the observation
        - 'column': The column name
        - 'download_date': Date when data was downloaded
        - 'frequency': Detected frequency of the data
        - 'period_start': Start boundary of the period
        - 'period_end': End boundary of the period
        - 'reference_point': Reference point used ('start' or 'end')
        - 'delay': Calculated publication delay (ceil-rounded)
        - 'unit': Unit of the delay value

    Raises:
        ValueError: If unit is not one of 'us', 's', 'D', 'microsecond', 'second', 'day'
    """
    # Copie indépendante du jeu de données
    publication_delays = new_observations.copy()

    # Ajout de l'indicateur à l'index et suppression de la date
    publication_delays.set_index("column", drop=True, append=True, inplace=True)
    publication_delays.reset_index(
        level=publication_delays.index.nlevels-2,
        drop=False,
        inplace=True,
        names=[
            "observation_date" if i == publication_delays.index.nlevels-2 else i
            for i in range(publication_delays.index.nlevels)
        ],
    )

    # Ajout d'informations d'intérêt
    # Date de téléchargement
    publication_delays["download_date"] = download_date

    # Fréquence
    # Détection de la fréquence pour chaque indicateur dans new_data
    frequency_map = detect_frequency(new_data, literal=True)

    # Conversion du dictionnaire en DataFrame ou création de la série
    if isinstance(frequency_map, dict):
        # Si c'est un dictionnaire (cas panel data ou multi-colonnes)
        freq_df = pd.Series(frequency_map, name="frequency").to_frame()
        # Si les clés sont des tuples (panel_id, column), décomposer
        if freq_df.index.nlevels == 1 and isinstance(freq_df.index[0], tuple):
            freq_df.index = pd.MultiIndex.from_tuples(freq_df.index)
    else:
        # Si c'est une fréquence unique
        freq_df = pd.Series(frequency_map, index=publication_delays.index, name="frequency").to_frame()

    # Ajout de la fréquence au jeu de données
    publication_delays = publication_delays.join(freq_df, on=publication_delays.index.names)

    # Dates de début et de fin de la période sur laquelle porte l'observation
    # Calcul des bornes de période en fonction de la fréquence
    publication_delays[["period_start", "period_end"]] = publication_delays[["observation_date", "frequency"]].apply(
        lambda row: pd.Series(get_period_boundaries(date=row["observation_date"], frequency=row["frequency"]), index=["period_start", "period_end"]),
        axis=1
    )

    # Point de référence
    publication_delays["reference_point"] = reference_point

    # Délai de publication
    # Le délai de publication est toujours arrondi à l'entier supérieur
    # Calcul du délai selon le point de référence
    if reference_point == "start":
        reference_dates = publication_delays["period_start"]
    else:  # "end"
        reference_dates = publication_delays["period_end"]

    # Calcul des délais en timedelta
    delays_timedelta = download_date - reference_dates

    # Conversion selon l'unité spécifiée
    # Timedelta n'a que pour attributs "day", "second", "microsecond"
    if unit in ["D", "day"]:
        publication_delays["delay"] = np.ceil(delays_timedelta.dt.total_seconds() / 86400)
        publication_delays["unit"] = "day"
    elif unit in ["s", "second"]:
        publication_delays["delay"] = np.ceil(delays_timedelta.dt.total_seconds())
        publication_delays["unit"] = "second"
    elif unit in ["us", "microsecond"]:
        publication_delays["delay"] = np.ceil(delays_timedelta.dt.total_seconds() * 1_000_000)
        publication_delays["unit"] = "microsecond"
    else:
        raise ValueError(f"Unit must be one of 'us', 's', 'D', 'microsecond', 'second', 'day', got {unit}") 


    return publication_delays