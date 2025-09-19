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
                                    reference_point: Optional[str] = None) -> Dict[str, Any]:
        """Compare two datasets and calculate publication delay information.

        This method identifies new observations in new_data that were not
        present in existing_data, then calculates publication delays.

        Args:
            new_data: Dataset containing new downloaded data
            existing_data: Dataset of already stored data
            download_date: Download date of new data
            reference_point: Reference point for delay calculation ('start' or 'end')

        Returns:
            Dictionary containing comparison and calculation statistics

        Raises:
            ValueError: If data does not respect expected format
        """
        # Validation des paramètres d'entrée
        new_data = self._validate_input_data(new_data)
        existing_data = self._validate_input_data(existing_data)
        download_date = self._parse_download_date(download_date)
        reference_point = reference_point or self.default_reference_point_

        # Identification des nouvelles observations
        new_observations = self._identify_new_observations(new_data, existing_data)

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
                                  existing_data: pd.DataFrame) -> pd.DataFrame:
        """Identify new observations in new_data compared to existing_data.

        Args:
            new_data: New data
            existing_data: Existing data

        Returns:
            DataFrame containing only new observations
        """
        # Définition des colonnes pour l'identification unique des observations
        id_cols = self.panel_cols_ + [self.time_col_]

        # Colonnes de données (indicateurs)
        data_cols = [col for col in new_data.columns
                    if col not in id_cols]

        # Initialisation de la liste des nouvelles observations
        new_observations = []

        # Groupement pour optimiser la comparaison
        if self.panel_cols_:
            # Données panel : comparaison par groupe
            for panel_values, new_group in new_data.groupby(self.panel_cols_):
                # Recherche du groupe correspondant dans les données existantes
                mask = np.ones(len(existing_data), dtype=bool)
                for i, col in enumerate(self.panel_cols_):
                    if isinstance(panel_values, tuple):
                        mask &= (existing_data[col] == panel_values[i])
                    else:
                        mask &= (existing_data[col] == panel_values)

                existing_group = existing_data[mask]

                # Identification des nouvelles observations pour ce groupe
                group_new_obs = self._compare_group_data(
                    new_group, existing_group, data_cols, panel_values
                )
                new_observations.extend(group_new_obs)
        else:
            # Série temporelle simple
            group_new_obs = self._compare_group_data(
                new_data, existing_data, data_cols, None
            )
            new_observations.extend(group_new_obs)

        # Conversion en DataFrame
        if new_observations:
            return pd.DataFrame(new_observations)
        else:
            return pd.DataFrame(columns=new_data.columns)

    # Méthode auxiliaire de comparaison des données pour chaque groupe
    # /!\ Je vérifie ici l'égalité des anciennes et nouvelles valeurs. En faisant cela je capte également les révisions. J'aimerais ajouter un argument qui permette de signifier si je souhaite identifier toutes les modifications entre anciennes et nouvelles données ou seulement les remplacements des null / nan par des valeurs non nulles.             
    def _compare_group_data(self,
                           new_group: pd.DataFrame,
                           existing_group: pd.DataFrame,
                           data_cols: List[str],
                           panel_values: Optional[Union[str, Tuple]]) -> List[Dict]:
        """Compare les données d'un groupe et identifie les nouvelles observations.

        Args:
            new_group: Nouvelles données du groupe
            existing_group: Données existantes du groupe
            data_cols: Liste des colonnes de données à comparer
            panel_values: Valeurs des colonnes panel pour ce groupe

        Returns:
            Liste des nouvelles observations sous forme de dictionnaires
        """
        # Initialisation de la liste des nouvelles observations
        new_observations = []

        # Parcours des lignes du nouveau groupe
        for _, new_row in new_group.iterrows():
            observation_date = new_row[self.time_col_]

            # Recherche de l'observation correspondante dans les données existantes
            existing_match = existing_group[
                existing_group[self.time_col_] == observation_date
            ]

            # Vérification pour chaque indicateur
            for col in data_cols:
                new_value = new_row[col]

                # Si la valeur est NaN, on l'ignore
                if pd.isna(new_value):
                    continue

                # Vérification si cette observation est nouvelle
                is_new_observation = False

                if existing_match.empty:
                    # Aucune observation à cette date dans les données existantes
                    is_new_observation = True
                else:
                    existing_value = existing_match[col].iloc[0]
                    if pd.isna(existing_value) and not pd.isna(new_value):
                        # La valeur était manquante et est maintenant disponible
                        is_new_observation = True
                # Caractérisation de la nouvelle observation
                if is_new_observation:
                    obs_dict = {
                        'observation_date': observation_date,
                        'indicator_name': col,
                        'value': new_value
                    }

                    # Ajout des colonnes panel si présentes
                    if self.panel_cols_ and panel_values is not None:
                        if isinstance(panel_values, tuple):
                            for i, panel_col in enumerate(self.panel_cols_):
                                obs_dict[panel_col] = panel_values[i]
                        else:
                            obs_dict[self.panel_cols_[0]] = panel_values

                    new_observations.append(obs_dict)

        return new_observations

    # Méthode de calcul des délais de publication
    def _calculate_release_delays(self,
                                 new_observations: pd.DataFrame,
                                 download_date: datetime,
                                 reference_point: str) -> List[Dict[str, Any]]:
        """Calcule les délais de publication pour les nouvelles observations.

        Args:
            new_observations: DataFrame des nouvelles observations
            download_date: Date de téléchargement
            reference_point: Point de référence ('start' ou 'end')

        Returns:
            Liste des enregistrements de délais calculés
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
                    observation_date, indicator_name
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
                                   indicator_name: str) -> Dict[str, Any]:
        """Determine period boundaries for a given observation.

        Args:
            observation_date: Date of observation
            indicator_name: Name of indicator

        Returns:
            Dictionary containing period information
        """
        # Tentative de détection de la fréquence
        frequency = self._detect_frequency_from_name(indicator_name)

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

    def _detect_frequency_from_name(self, indicator_name: str) -> Optional[str]:
        """Detect frequency from indicator name using simple heuristics.

        Args:
            indicator_name: Name of the indicator

        Returns:
            Frequency of the indicator or None if not detected
        """
        # Heuristiques simples basées sur le nom de l'indicateur
        name_lower = indicator_name.lower()

        if any(keyword in name_lower for keyword in ['daily', 'jour', 'day']):
            return 'daily'
        elif any(keyword in name_lower for keyword in ['weekly', 'semaine', 'week']):
            return 'weekly'
        elif any(keyword in name_lower for keyword in ['quarterly', 'trimestre', 'quarter', 'q1', 'q2', 'q3', 'q4']):
            return 'quarterly'
        elif any(keyword in name_lower for keyword in ['annual', 'yearly', 'annuel', 'year']):
            return 'annual'
        else:
            # Par défaut, supposer mensuel
            return 'monthly'

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