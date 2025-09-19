"""Data manager for comparing and storing publication delays information.

This module provides tools for comparing new datasets with existing data
and managing publication delay information using pandas DataFrames.
"""
# Importation des modules
# Modules de base
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple, List, Any
from datetime import datetime, timedelta
import warnings
# Module de détection de la fréquence des séries
from ..frequency.detector import FrequencyDetector

# /!\ Voir si on a besoin de "time_col" et "panels_cols" ou si on peut utiliser les index (cohérent avec le comportement des crossvals)

# Classe de création des données de publication sur la base de comparaison de jeux de données
class ReleaseDataManager:
    """Manager for comparing and storing publication delay data.

    This class compares new datasets with existing data, identifies
    new observations and calculates publication delays for each indicator.

    Args:
        time_col: Name of the time column in DataFrames
        panel_cols: List of columns identifying panel dimensions
        frequency_detector: Custom frequency detector (optional)
        default_reference_point: Default reference point ('start' or 'end')

    Attributes:
        time_col_ (str): Name of the time column
        panel_cols_ (Optional[List[str]]): Panel columns
        frequency_detector_ (FrequencyDetector): Frequency detector
        default_reference_point_ (str): Default reference point
        delay_records_ (List[Dict]): Stored delay records

    Examples:
        >>> from tsforecast.delays import ReleaseDataManager
        >>>
        >>> # Create manager
        >>> manager = ReleaseDataManager(
        ...     time_col='date',
        ...     panel_cols=['country', 'indicator']
        ... )
        >>>
        >>> # Compare datasets
        >>> new_data = pd.DataFrame({...})
        >>> existing_data = pd.DataFrame({...})
        >>> delays_info = manager.compare_and_calculate_delays(
        ...     new_data, existing_data, download_date='2023-12-01'
        ... )
    """
    # Initialisation
    def __init__(self,
                 time_col: str = 'date',
                 panel_cols: Optional[List[str]] = None,
                 frequency_detector: Optional[FrequencyDetector] = None,
                 default_reference_point: str = 'end'):
        """Initialize the release data manager.

        Args:
            time_col: Name of the time column
            panel_cols: Columns identifying panel dimensions
            frequency_detector: Frequency detector (created by default if None)
            default_reference_point: 'start' or 'end' of period for delay calculation
        """
        # Initialisation des attributs
        self.time_col_ = time_col
        self.panel_cols_ = panel_cols or []
        self.frequency_detector_ = frequency_detector or FrequencyDetector()
        self.default_reference_point_ = default_reference_point
        self.delay_records_ = []  # Stockage des enregistrements de délais

        # Validation du point de référence
        if default_reference_point not in ['start', 'end']:
            raise ValueError("default_reference_point must be 'start' or 'end'")

    # Méthode principale d'imputation des délais de publication à partir de deux jeux de données
    def compare_and_calculate_delays(self,
                                    new_data: pd.DataFrame,
                                    existing_data: pd.DataFrame,
                                    download_date: Union[str, datetime],
                                    reference_point: Optional[str] = None,
                                    detection_mode: str = 'new_only') -> Dict[str, Any]:
        """Compare two datasets and calculate publication delay information.

        This method identifies new or changed observations in new_data compared
        to existing_data, then calculates publication delays.

        Args:
            new_data: Dataset containing new downloaded data
            existing_data: Dataset of already stored data
            download_date: Download date of new data
            reference_point: Reference point for delay calculation ('start' or 'end')
            detection_mode: Type of changes to detect:
                - 'new_only': Only null → non-null changes (default)
                - 'all_changes': Both null → non-null and value changes

        Returns:
            Dictionary containing comparison and calculation statistics

        Raises:
            ValueError: If data does not respect expected format or invalid detection_mode
        """
        # Validation des paramètres d'entrée
        new_data = self._validate_input_data(new_data)
        existing_data = self._validate_input_data(existing_data)
        download_date = self._parse_download_date(download_date)
        reference_point = reference_point or self.default_reference_point_

        # Validation du mode de détection
        if detection_mode not in ['new_only', 'all_changes']:
            raise ValueError("detection_mode must be 'new_only' or 'all_changes'")

        # Identification des nouvelles observations
        new_observations = self._identify_new_observations(new_data, existing_data, detection_mode)

        # Vérification que des nouvelles observations existent
        if new_observations.empty:
            return {
                'new_observations_count': 0,
                'delays_calculated': 0,
                'processing_time': datetime.now(),
                'delay_records': []
            }

        # Calcul des délais de publication
        delay_records = self._calculate_release_delays(
            new_observations, download_date, reference_point
        )

        # Stockage des enregistrements
        self.delay_records_.extend(delay_records)

        return {
            'new_observations_count': len(new_observations),
            'delays_calculated': len(delay_records),
            'processing_time': datetime.now(),
            # /!\ Revoir si le filtre sur 10 est pertinent
            'delay_records': delay_records[:10] if len(delay_records) > 10 else delay_records  # Premiers 10 pour debug
        }

    # Méthode auxiliaire de validation des jeux de données en entrée
    def _validate_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize input data.

        Args:
            data: DataFrame to validate

        Returns:
            Validated and normalized DataFrame

        Raises:
            ValueError: If data does not respect required format
        """
        # Vérification que le jeu de données est un pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        # Vérification de la présence de la colonne temporelle
        if self.time_col_ not in data.columns:
            raise ValueError(f"Time column '{self.time_col_}' not found")

        # Vérification des colonnes panel si spécifiées
        if self.panel_cols_:
            missing_panel_cols = set(self.panel_cols_) - set(data.columns)
            if missing_panel_cols:
                raise ValueError(f"Missing panel columns: {missing_panel_cols}")

        # Conversion de la colonne temporelle
        data = data.copy()
        data[self.time_col_] = pd.to_datetime(data[self.time_col_])

        # Tri des données par ordre temporel et panel si nécessaire
        sort_cols = self.panel_cols_ + [self.time_col_] if self.panel_cols_ else [self.time_col_]
        data = data.sort_values(sort_cols)

        return data

    # Méthode auxiliaire de conversion et de validation de la date de téléchargement
    def _parse_download_date(self, download_date: Union[str, datetime]) -> datetime:
        """Parse and validate download date.

        Args:
            download_date: Date as string or datetime

        Returns:
            Validated download date

        Raises:
            ValueError: If date cannot be parsed
        """
        if isinstance(download_date, str):
            try:
                return pd.to_datetime(download_date).to_pydatetime()
            except:
                raise ValueError(f"Invalid date format: {download_date}")
        elif isinstance(download_date, datetime):
            return download_date
        else:
            raise ValueError("download_date must be a string or datetime")

    # Méthode auxiliaire d'identification des nouvelles observations
    def _identify_new_observations(self,
                                  new_data: pd.DataFrame,
                                  existing_data: pd.DataFrame,
                                  detection_mode: str = 'new_only') -> pd.DataFrame:
        """Identify new or changed observations.

        This method uses pandas vectorized operations instead of loops for
        significantly better performance on large datasets.

        Args:
            new_data: New data DataFrame
            existing_data: Existing data DataFrame
            detection_mode: 'new_only' or 'all_changes'

        Returns:
            DataFrame containing only new or changed observations
        """
        # Définition des colonnes d'identification et de données
        id_cols = self.panel_cols_ + [self.time_col_]
        data_cols = [col for col in new_data.columns if col not in id_cols]

        # Vérification que les deux DataFrames ont les mêmes colonnes de données
        existing_data_cols = [col for col in existing_data.columns if col not in id_cols]
        common_data_cols = list(set(data_cols).intersection(set(existing_data_cols)))

        if not common_data_cols:
            # Aucune colonne commune, toutes les observations sont nouvelles
            return self._format_observations_output(new_data, data_cols)

        # Création d'index multi-niveaux pour l'alignement
        new_indexed = new_data.set_index(id_cols)[common_data_cols]
        existing_indexed = existing_data.set_index(id_cols)[common_data_cols]

        # Alignement des DataFrames sur l'index commun
        aligned_new, aligned_existing = new_indexed.align(existing_indexed, fill_value=np.nan)

        # Création des masques booléens pour les valeurs nulles
        new_isnull = aligned_new.isnull()
        existing_isnull = aligned_existing.isnull()

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

        # Identification des observations avec des changements
        changed_observations = []

        # Parcours des lignes avec changements
        for idx, row_changes in changes_mask.iterrows():
            if row_changes.any():  # S'il y a au moins un changement dans cette ligne
                # Récupération des valeurs d'index (panel + time)
                if isinstance(idx, tuple):
                    index_values = dict(zip(id_cols, idx))
                else:
                    index_values = {id_cols[0]: idx}

                # Parcours des colonnes avec changements
                for col in row_changes[row_changes].index:
                    new_value = aligned_new.loc[idx, col]

                    # Ignore les valeurs NaN dans les nouvelles données
                    if pd.isna(new_value):
                        continue

                    # Création de l'observation
                    obs_dict = {
                        **index_values,
                        'indicator_name': col,
                        'value': new_value
                    }
                    changed_observations.append(obs_dict)

        # Conversion en DataFrame
        if changed_observations:
            return pd.DataFrame(changed_observations)
        else:
            # Retour d'un DataFrame vide avec les bonnes colonnes
            empty_cols = id_cols + ['indicator_name', 'value']
            return pd.DataFrame(columns=empty_cols)

    # Méthode auxiliaire pour formater la sortie des observations
    def _format_observations_output(self, data: pd.DataFrame, data_cols: List[str]) -> pd.DataFrame:
        """Format observations data into required output format.

        Args:
            data: Input DataFrame
            data_cols: List of data columns to process

        Returns:
            Formatted DataFrame with observations
        """
        id_cols = self.panel_cols_ + [self.time_col_]
        observations = []

        # Parcours des lignes et colonnes pour créer les observations
        for _, row in data.iterrows():
            for col in data_cols:
                if not pd.isna(row[col]):
                    obs_dict = {
                        **{id_col: row[id_col] for id_col in id_cols},
                        'indicator_name': col,
                        'value': row[col]
                    }
                    observations.append(obs_dict)

        return pd.DataFrame(observations) if observations else pd.DataFrame(columns=id_cols + ['indicator_name', 'value'])

    # Méthode de calcul des délais de publication
    def _calculate_release_delays(self,
                                 new_observations: pd.DataFrame,
                                 download_date: datetime,
                                 reference_point: str) -> List[Dict[str, Any]]:
        """Calculate publication delays for new observations.

        Args:
            new_observations: DataFrame of new observations
            download_date: Download date
            reference_point: Reference point ('start' or 'end')

        Returns:
            List of calculated delay records
        """
        # Initialisation de la liste des délais
        delay_records = []

        # Parcours des nouvelles observations
        for _, obs in new_observations.iterrows():
            try:
                # Détection de la fréquence de l'indicateur
                indicator_name = obs['indicator_name']
                observation_date = obs['observation_date']

                # Détermination de la période (début et fin)
                period_info = self._determine_period_boundaries(
                    observation_date, indicator_name, new_observations
                )

                # Calcul du délai selon le point de référence
                if reference_point == 'start':
                    reference_date = period_info['period_start']
                else:  # 'end'
                    reference_date = period_info['period_end']

                release_delay_days = (download_date - reference_date).days

                # Création de l'enregistrement
                delay_record = {
                    'indicator_name': indicator_name,
                    'observation_date': observation_date,
                    'period_start': period_info['period_start'],
                    'period_end': period_info['period_end'],
                    'download_date': download_date,
                    'release_delay_days': float(release_delay_days),
                    'is_period_start_reference': (reference_point == 'start'),
                    'data_frequency': period_info.get('frequency'),
                    'metadata': {
                        'calculation_method': 'automatic',
                        'reference_point': reference_point,
                        'original_value': float(obs['value']) if not pd.isna(obs['value']) else None
                    }
                }

                # Ajout de l'entité si données panel
                if self.panel_cols_:
                    entity_parts = []
                    for col in self.panel_cols_:
                        if col in obs:
                            entity_parts.append(str(obs[col]))
                    delay_record['entity_id'] = '|'.join(entity_parts) if entity_parts else None

                delay_records.append(delay_record)

            except Exception as e:
                warnings.warn(f"Error calculating delay for {obs.get('indicator_name', 'unknown')}: {str(e)}")
                continue

        return delay_records

    # Détermination des dates de début et de fin de la période à laquelle se réfère une observation
    def _determine_period_boundaries(self,
                                   observation_date: datetime,
                                   indicator_name: str,
                                   df : pd.DataFrame) -> Dict[str, Any]:
        """Determine period boundaries for a given observation.

        Args:
            observation_date: Date of observation
            indicator_name: Name of indicator
            df: DataFrames containing the indicator

        Returns:
            Dictionary containing period information
        """
        # Tentative de détection de la fréquence avec le FrequencyDetector
        frequency = FrequencyDetector().detect_frequency(df[indicator_name])

        if not frequency:
            # Fréquence par défaut si non détectée
            frequency = 'monthly'
            warnings.warn(f"Frequency not detected for {indicator_name}, using 'monthly' as default")

        # Calcul des limites selon la fréquence
        if frequency == 'daily':
            period_start = observation_date.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=1) - timedelta(microseconds=1)
        elif frequency == 'weekly':
            # Début de semaine (lundi)
            days_since_monday = observation_date.weekday()
            period_start = observation_date - timedelta(days=days_since_monday)
            period_start = period_start.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=7) - timedelta(microseconds=1)
        elif frequency == 'monthly':
            period_start = observation_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if observation_date.month == 12:
                next_month = period_start.replace(year=period_start.year + 1, month=1)
            else:
                next_month = period_start.replace(month=period_start.month + 1)
            period_end = next_month - timedelta(microseconds=1)
        elif frequency == 'quarterly':
            quarter_start_month = ((observation_date.month - 1) // 3) * 3 + 1
            period_start = observation_date.replace(month=quarter_start_month, day=1,
                                                  hour=0, minute=0, second=0, microsecond=0)
            if quarter_start_month == 10:
                next_quarter = period_start.replace(year=period_start.year + 1, month=1)
            else:
                next_quarter = period_start.replace(month=quarter_start_month + 3)
            period_end = next_quarter - timedelta(microseconds=1)
        elif frequency == 'annual':
            period_start = observation_date.replace(month=1, day=1,
                                                  hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start.replace(year=period_start.year + 1) - timedelta(microseconds=1)
        else:
            # Fréquence inconnue : utiliser le jour comme période
            period_start = observation_date.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=1) - timedelta(microseconds=1)

        return {
            'period_start': period_start,
            'period_end': period_end,
            'frequency': frequency
        }

    # Méthode d'extraction des délais de publication
    def get_delay_records(self) -> List[Dict[str, Any]]:
        """Get all stored delay records.

        Returns:
            List of delay records
        """
        return self.delay_records_.copy()

    # Méthode de réinitialisation des délais de publication
    def clear_delay_records(self) -> None:
        """Clear all stored delay records."""
        self.delay_records_.clear()

    # Méthode résumant les délais de publication stockés
    def get_delays_summary(self,
                          indicator_name: Optional[str] = None,
                          entity_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of stored delays.

        Args:
            indicator_name: Filter by indicator name (optional)
            entity_id: Filter by entity (optional)

        Returns:
            Dictionary containing delays summary
        """
        if not self.delay_records_:
            return {
                'record_count': 0,
                'message': 'No records found'
            }

        # Filtrage des enregistrements
        filtered_records = self.delay_records_
        if indicator_name:
            filtered_records = [r for r in filtered_records if r.get('indicator_name') == indicator_name]
        if entity_id:
            filtered_records = [r for r in filtered_records if r.get('entity_id') == entity_id]

        if not filtered_records:
            return {
                'record_count': 0,
                'message': 'No records found for specified criteria'
            }

        # Calcul des statistiques
        delays = [r['release_delay_days'] for r in filtered_records]
        summary = {
            'record_count': len(filtered_records),
            'median_delay': float(np.median(delays)),
            'mean_delay': float(np.mean(delays)),
            'std_delay': float(np.std(delays)),
            'min_delay': float(np.min(delays)),
            'max_delay': float(np.max(delays)),
            'unique_indicators': len(set(r['indicator_name'] for r in filtered_records)),
            'unique_entities': len(set(r.get('entity_id') for r in filtered_records if r.get('entity_id'))),
        }

        return summary