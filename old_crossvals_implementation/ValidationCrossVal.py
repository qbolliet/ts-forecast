# Scikit-Learn
from sklearn.base import is_classifier, clone, BaseEstimator, TransformerMixin
from sklearn.utils import indexable
from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils.validation import _num_samples
#from sklearn.utils.validation import _deprecate_positional_args
from sklearn.model_selection._split import check_cv, _BaseKFold
from sklearn.model_selection._validation import _fit_and_predict
from sklearn.utils.metaestimators import _safe_split
from sklearn.exceptions import FitFailedWarning
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd


class ForwardWalkingValidationPanelData(_BaseKFold):

    def __init__(self, PanelCV, PeriodVal, Gap, TestSize=1):

        self.PanelCV = PanelCV
        self.PeriodVal = PeriodVal
        self.Gap = Gap
        self.n_splits = PeriodVal # Ne sert pas à grand chose à part à l'héritage avec _BaseKFold
        self.TestSize = TestSize
        

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        # Test initial
        X, y, groups = indexable(X, y, groups)
        
        # Construction des Indices à retourner
        NumSamples = _num_samples(X)

        # Numérotation des indices
        Indices = np.arange(NumSamples)

        # Gestion de PeriodVal
        if self.PeriodVal < 1:
            self.PeriodVal = int(NumSamples*self.PeriodVal)

        if self.PanelCV :
            
            # Construction de la liste correspondant aux des périodes de Train
            ListeIDebutTrain = [int(X.index.get_locs([Pays, slice(None)])[0]) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]
            
            # Construction de la liste correspondant à la fin des périodes de Val
            ListeIFinVal = [i for i in ListeIDebutTrain[1:]].append(X.shape[0])

            # Construction de la liste correspondant aux débuts des périodes de Val
            ListeIDebutVal = [i - self.PeriodVal for i in ListeIFinVal]


            # Vérification qu'il y a autant de DebutTrain que de DebutVal et de FinVal
            if len(ListeIDebutTrain) != len(ListeIDebutVal) :
                raise ValueError(("Cannot have a number of train debuts = {0} different from the number of val debuts = {1}").format(len(ListeIDebutTrain), len(ListeIDebutVal)))
            elif len(ListeIDebutVal) != len(ListeIFinVal) :
                raise ValueError(("Cannot have a number of val debuts = {0} different from the number of val ends = {1}").format(len(ListeIDebutVal), len(ListeIFinVal)))
            
            # Vérification que le ValSize est compris dans l'intervalle DebutVal,FinVal
            for i in range(len(ListeIDebutVal)):
                IDebutVal = (ListeIDebutVal)[i]
                IFinVal = (ListeIFinVal)[i]
                if self.TestSize > IFinVal-IDebutVal:
                    raise ValueError(("Cannot have the val size ={0} greater than the val interval: {1}.").format(self.TestSize,IFinVal-IDebutVal))
            
            # Vérification que toutes les périodes de Val sont de la même longueur
            if len(ListeIDebutVal) > 1 :
                for i in range(1,len(ListeIDebutVal)):
                    if (ListeIFinVal[i]-ListeIDebutVal[i]) != (ListeIFinVal[i-1]-ListeIDebutVal[i-1]) :
                        raise ValueError(("The length of the {0}th val set = {1} is different from the length of the {2}th val set = {3}.").format(i-1, ListeIFinVal[i-1]-ListeIDebutVal[i-1], i, ListeIFinVal[i]-ListeIDebutVal[i]))

            # Yield des indices
            for i in range(self.PeriodVal):

                # Construction des Index correspondant aux périodes de Train et de Val
                IndexTrain = np.concatenate([np.arange(ListeIDebutTrain[j], ((ListeIDebutVal)[j] + i - self.Gap[j])) for j in range(len(ListeIDebutVal))], axis=0).tolist()
                IndexTest= np.concatenate([np.arange(IDebutVal + i, (IDebutVal + i +self.TestSize)) for IDebutVal in ListeIDebutVal], axis=0).tolist()
            
                yield (Indices[IndexTrain], Indices[IndexTest])

        else :

            # Indices de début et de Fin
            IDebutVal = X.shape[0] - self.PeriodVal

            # Yield des indices
            for i in range(self.PeriodVal):

                IndexTrain = np.arange(0, IDebutVal + i - self.Gap)
                IndexTest = np.arange(IDebutVal + i, IDebutVal + i + self.TestSize)

                yield (Indices[IndexTrain], Indices[IndexTest])


class ForwardWalkingValidation2PanelData(_BaseKFold):

    def __init__(self, PanelCV, PeriodVal, TestSize=1):
        self.PanelCV = PanelCV
        self.PeriodVal = PeriodVal
        self.TestSize = TestSize
        self.n_splits = PeriodVal
        

    def split(self, X, y=None, groups=None):
        
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        # Test initial
        X, y, groups = indexable(X, y, groups)
        
        # Construction des Indices à retourner
        NumSamples = _num_samples(X)

        # Numérotation des indices
        Indices = np.arange(NumSamples)

        # Gestion de PeriodVal
        if self.PeriodVal < 1:
            self.PeriodVal = int(NumSamples*self.PeriodVal)

        if self.PanelCV :
            
            # Construction de la liste correspondant aux des périodes de Train
            ListeIDebutTrain = [int(X.index.get_locs([Pays, slice(None)])[0]) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]
            
            # Construction de la liste correspondant à la fin des périodes de Val
            ListeIFinVal = [i for i in ListeIDebutTrain[1:]].append(X.shape[0])

            # Construction de la liste correspondant aux débuts des périodes de Val
            ListeIDebutVal = [i - self.PeriodVal for i in ListeIFinVal]


            # Vérification qu'il y a autant de DebutTrain que de DebutVal et de FinVal
            if len(ListeIDebutTrain) != len(ListeIDebutVal) :
                raise ValueError(("Cannot have a number of train debuts = {0} different from the number of val debuts = {1}").format(len(ListeIDebutTrain), len(ListeIDebutVal)))
            elif len(ListeIDebutVal) != len(ListeIFinVal) :
                raise ValueError(("Cannot have a number of val debuts = {0} different from the number of val ends = {1}").format(len(ListeIDebutVal), len(ListeIFinVal)))
            
            # Vérification que le ValSize est compris dans l'intervalle DebutVal,FinVal
            for i in range(len(ListeIDebutVal)):
                IDebutVal = (ListeIDebutVal)[i]
                IFinVal = (ListeIFinVal)[i]
                if self.TestSize > IFinVal-IDebutVal:
                    raise ValueError(("Cannot have the val size ={0} greater than the val interval: {1}.").format(self.TestSize,IFinVal-IDebutVal))
            
            # Vérification que toutes les périodes de Val sont de la même longueur
            if len(ListeIDebutVal) > 1 :
                for i in range(1,len(ListeIDebutVal)):
                    if (ListeIFinVal[i]-ListeIDebutVal[i]) != (ListeIFinVal[i-1]-ListeIDebutVal[i-1]) :
                        raise ValueError(("The length of the {0}th val set = {1} is different from the length of the {2}th val set = {3}.").format(i-1, ListeIFinVal[i-1]-ListeIDebutVal[i-1], i, ListeIFinVal[i]-ListeIDebutVal[i]))

            # Yield des indices
            for i in range(self.PeriodVal):

                # Construction des Index correspondant aux périodes de Train et de Val
                IndexTrain = np.concatenate([np.arange(ListeIDebutTrain[j], ((ListeIDebutVal)[j] + i)) for j in range(len(ListeIDebutVal))], axis=0).tolist()
                IndexTest= np.concatenate([np.arange(IDebutVal + i, (IDebutVal + i +self.TestSize)) for IDebutVal in ListeIDebutVal], axis=0).tolist()
            
                yield (Indices[IndexTrain], Indices[IndexTest])

        else :

            # Indices de début et de Fin
            IDebutVal = X.shape[0] - self.PeriodVal

            # Yield des indices
            for i in range(self.PeriodVal):

                IndexTrain = np.arange(0, IDebutVal + i)
                IndexTest = np.arange(IDebutVal + i, IDebutVal + i + self.TestSize)

                yield (Indices[IndexTrain], Indices[IndexTest])


# Threshold Validation
class ThresholdCVPanelData(_BaseKFold):

    def __init__(self):
        self.n_splits =1 

    def split(self, X, y=None, groups=None):

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)

        indices = np.arange(n_samples)

        # Yield des indices
        yield (indices, indices)