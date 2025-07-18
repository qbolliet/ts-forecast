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

# Forecasting CrossVal

class PanelCVOutSamplePanelData():

    def __init__(self, ListeDateTest, Gap, TestSize=1):
        self.ListeDateTest = ListeDateTest
        self.Gap = Gap
        self.TestSize = TestSize

    def split(self, X, y=None, groups=None):

        # Test initial
        X, y, groups = indexable(X, y, groups)
        
        # Construction des Indices à retourner
        NumSamples = _num_samples(X)

        Indices = np.arange(NumSamples)

        # Construction de la liste correspondant aux des périodes de Train
        ListeIDebutTrain = [int(X.index.get_locs([Pays, slice(None)])[0]) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]
            
        for i in range(0,len(self.ListeDateTest), self.TestSize):
            
            # Construction des listes correspondant aux débuts des périodes de Test
            ListeIDebutTest = [X.index.get_loc(key=(Pays, self.ListeDateTest[i])) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]
            
            # Vérification que les listes de début de train et de début de test sont le même longueur
            if len(ListeIDebutTrain) != len(ListeIDebutTest) :
                raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrain), len(ListeIDebutTest)))
        
            # Construction des Index correspondant aux périodes de Train et de Test
            IndexTrain = np.concatenate([np.arange(ListeIDebutTrain[j], ((ListeIDebutTest)[j] - self.Gap[j])) for j in range(len(ListeIDebutTest))], axis=0).tolist()
            IndexTest= np.concatenate([np.arange(IDebutTest, (IDebutTest+self.TestSize)) for IDebutTest in ListeIDebutTest], axis=0).tolist()
            
            yield (Indices[IndexTrain], Indices[IndexTest])


class PanelCVOnlineOutSamplePanelData():

    def __init__(self, ListeDateTest, Gap, TestSize=1):
        self.ListeDateTest = ListeDateTest
        self.Gap = Gap
        self.TestSize = TestSize

    def split(self, X, y=None, groups=None):

        # Test initial
        X, y, groups = indexable(X, y, groups)
        
        # Construction des Indices à retourner
        NumSamples = _num_samples(X)

        Indices = np.arange(NumSamples)

        # Construction de la liste correspondant aux des périodes de Train
        ListeIDebutTrain = [int(X.index.get_locs([Pays, slice(None)])[0]) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]

        for i in range(0,len(self.ListeDateTest), self.TestSize):

            # Construction des listes correspondant aux débuts des périodes de Test
            ListeIDebutTest = [X.index.get_loc(key=(Pays, self.ListeDateTest[i])) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]
            
            # Vérification que les listes de début de train et de début de test sont le même longueur
            if len(ListeIDebutTrain) != len(ListeIDebutTest) :
                raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrain), len(ListeIDebutTest)))
        
            # Construction des Index correspondant aux périodes de Train et de Test
            IndexTrain = np.concatenate([np.arange(ListeIDebutTrain[j], ((ListeIDebutTest)[j] - self.Gap[j])) for j in range(len(ListeIDebutTest))], axis=0).tolist()
            IndexTest= np.concatenate([np.arange(IDebutTest, (IDebutTest+self.TestSize)) for IDebutTest in ListeIDebutTest], axis=0).tolist()
            
            # Remplacement des indices de Train par les indices de Test pour la prochaine intération
            ListeIDebutTrain = ListeIDebutTest.copy()

            yield (Indices[IndexTrain], Indices[IndexTest])



class OneCountryOneModelCVOutSamplePanelData():
    
    def __init__(self, ListeDateTest, Gap, TestSize=1):
        
        self.ListeDateTest = ListeDateTest
        self.Gap = Gap
        self.TestSize = TestSize

    def split(self, X, y=None, groups=None):

        # Test initial
        X, y, groups = indexable(X, y, groups)
        
        # Construction des Indices à retourner
        NumSamples = _num_samples(X)

        Indices = np.arange(NumSamples)

        # Construction de la liste correspondant aux des périodes de Train
        ListeIDebutTrain = [int(X.index.get_locs([Pays, slice(None)])[0]) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]
         
        for i in range(0,len(self.ListeDateTest), self.TestSize):
            
            # Construction des listes correspondant aux débuts des périodes de Test
            ListeIDebutTest = [X.index.get_loc(key=(Pays, self.ListeDateTest[i])) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]
            
            # Vérification que les listes de début de train et de début de test sont le même longueur
            if len(ListeIDebutTrain) != len(ListeIDebutTest) :
                raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrain), len(ListeIDebutTest)))
        
            for j in range(len(ListeIDebutTest)) :

                # Construction des Index correspondant aux périodes de Train et de Test
                IndexTrain = np.arange(ListeIDebutTrain[j], ((ListeIDebutTest)[j] - self.Gap[j]))
                IndexTest = np.arange(ListeIDebutTest[j], (ListeIDebutTest[j] + self.TestSize)).tolist()
            
                yield (Indices[IndexTrain], Indices[IndexTest])



class OneCountryOneModelCVOnlineOutSamplePanelData():
    
    def __init__(self, ListeDateTest, Gap, TestSize=1):
        
        self.ListeDateTest = ListeDateTest
        self.Gap = Gap
        self.TestSize = TestSize

    def split(self, X, y=None, groups=None):

        # Test initial
        X, y, groups = indexable(X, y, groups)
        
        # Construction des Indices à retourner
        NumSamples = _num_samples(X)

        Indices = np.arange(NumSamples)

        # Construction de la liste correspondant aux des périodes de Train
        ListeIDebutTrain = [int(X.index.get_locs([Pays, slice(None)])[0]) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]
         
        for i in range(0,len(self.ListeDateTest), self.TestSize):
            
            # Construction des listes correspondant aux débuts des périodes de Test
            ListeIDebutTest = [X.index.get_loc(key=(Pays, self.ListeDateTest[i])) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]
            
            # Vérification que les listes de début de train et de début de test sont le même longueur
            if len(ListeIDebutTrain) != len(ListeIDebutTest) :
                raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrain), len(ListeIDebutTest)))
        
            for j in range(len(ListeIDebutTest)) :

                # Construction des Index correspondant aux périodes de Train et de Test
                IndexTrain = np.arange(ListeIDebutTrain[j], ((ListeIDebutTest)[j] - self.Gap[j]))
                IndexTest = np.arange(ListeIDebutTest[j], (ListeIDebutTest[j] + self.TestSize)).tolist()
            
                yield (Indices[IndexTrain], Indices[IndexTest])



# Ne renvoie que les indices avant la première date de test
class PanelCVInSamplePanelData():
    
    def __init__(self, ListeDateTest):
        self.ListeDateTest = ListeDateTest

    def split(self, X, y=None, groups=None):

        # Test initial
        X, y, groups = indexable(X, y, groups)
        
        # Construction des Indices à retourner
        NumSamples = _num_samples(X)

        Indices = np.arange(NumSamples)

        # Construction des listes correspondant aux débuts des périodes de Train
        ListeIDebutTrain = [int(X.index.get_locs([Pays, slice(None)])[0]) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]

        # Construction des listes correspondant aux débuts de la première période de Test
        ListeIDebutTest = [X.index.get_loc(key=(Pays, self.ListeDateTest[0])) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]   

        # Vérification que les listes sont de même longeur
        if len(ListeIDebutTrain) != len(ListeIDebutTest) :
                raise ValueError(("Cannot have a number of train debuts with FreqTrain = {0} different from the number of train debuts with FreqTest = {1}").format(len(ListeIDebutTrain), len(ListeIDebutTest)))

        # Construction des indices de Train et Test
        Index = np.concatenate([np.arange(ListeIDebutTrain[j], ListeIDebutTest[j]) for j in range(len(ListeIDebutTest))], axis=0).tolist()

        yield (Indices[Index], Indices[Index])


# Ne renvoie que les indices avant la première date de test
class OneCountryOneModelCVInSamplePanelData():
    
    def __init__(self, ListeDateTest):
        self.ListeDateTest = ListeDateTest

    def split(self, X, y=None, groups=None):

        # Test initial
        X, y, groups = indexable(X, y, groups)
        
        # Construction des Indices à retourner
        NumSamples = _num_samples(X)

        Indices = np.arange(NumSamples)

        # Construction des listes correspondant aux débuts des périodes de Train
        ListeIDebutTrain = [int(X.index.get_locs([Pays, slice(None)])[0]) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]

        # Construction des listes correspondant aux débuts de la première période de Test
        ListeIDebutTest = [X.index.get_loc(key=(Pays, self.ListeDateTest[0])) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]  

        # Vérification que les listes sont de même longeur
        if len(ListeIDebutTrain) != len(ListeIDebutTest) :
                raise ValueError(("Cannot have a number of train debuts with FreqTrain = {0} different from the number of train debuts with FreqTest = {1}").format(len(ListeIDebutTrain), len(ListeIDebutTest)))

        for j in range(len(ListeIDebutTest)):
            
            # Construction des indices de Train et Test
            Index = np.arange(ListeIDebutTrain[j], ListeIDebutTest[j])

            yield (Indices[Index], Indices[Index])


# Threshold CrossVals

class PanelThresholdPanelData(_BaseKFold):

    def __init__(self, TestSize = 1):
        self.TestSize = 1

    def split(self, X, y=None, groups=None):

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)

        indices = np.arange(n_samples)

        # Indices de début de train pour chaque pays
        IndexPays = list(X.index.levels[0].values)
        Index = list(X.index)
        ListeIDebutTrain = []
        for IRow in range(X.shape[0]):
            if Index[IRow][0] in IndexPays :
                ListeIDebutTrain.append(IRow)
                IndexPays.remove(Index[IRow][0])
        
        # Indices fin de période de Test pour chaque pays
        ListeIFinTest = []
        for IRow in range(X.shape[0]-1):
            if Index[IRow][0] != Index[IRow+1][0] :
                ListeIFinTest.append(IRow+1)
        ListeIFinTest.append(X.shape[0])

        # Vérification que les périodes comptent le même nombre d'observations

        LenTest = ListeIFinTest[0] - ListeIDebutTrain[0]
        self.n_splits = LenTest

        # Yield des indices
        for i in range(LenTest):
            IndexTrain = [idx for j in range(len(ListeIDebutTrain)) for idx in range(ListeIDebutTrain[j],(ListeIDebutTrain[j] + i))]
            IndexTest = [idx for j in range(len(ListeIDebutTrain)) for idx in range((ListeIDebutTrain[j] + i),(ListeIDebutTrain[j] + i + self.TestSize))]
            yield (indices[IndexTrain], indices[IndexTest])


class OneCountryOneModelThresholdPanelData(_BaseKFold):

    def __init__(self, TestSize = 1):
        self.TestSize = TestSize

    def split(self, X, y=None, groups=None):

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)

        indices = np.arange(n_samples)

        # Indices de début de train pour chaque pays
        IndexPays = list(X.index.levels[0].values)
        Index = list(X.index)
        ListeIDebutTrain = []
        for IRow in range(X.shape[0]):
            if Index[IRow][0] in IndexPays :
                ListeIDebutTrain.append(IRow)
                IndexPays.remove(Index[IRow][0])
        
        # Indices fin de période de Test pour chaque pays
        ListeIFinTest = []
        for IRow in range(X.shape[0]-1):
            if Index[IRow][0] != Index[IRow+1][0] :
                ListeIFinTest.append(IRow+1)
        ListeIFinTest.append(X.shape[0])

        # Vérification que les périodes comptent le même nombre d'observations

        LenTest = ListeIFinTest[0] - ListeIDebutTrain[0]
        self.n_splits = LenTest

        # Yield des indices
        for i in range(LenTest):
            for j in range(len(ListeIDebutTrain)) :
                IndexTrain = [idx for idx in range(ListeIDebutTrain[j],(ListeIDebutTrain[j] + i))]
                IndexTest = [idx for idx in range((ListeIDebutTrain[j] + i),(ListeIDebutTrain[j] + i + self.TestSize))]
                yield (indices[IndexTrain], indices[IndexTest])


class PanelThresholdInSamplePanelData(_BaseKFold):

    def __init__(self):
        self.n_splits = 1

    def split(self, X, y=None, groups=None):

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)

        indices = np.arange(n_samples)

        # Yield des indices
        yield (indices, indices)

class OneCountryOneModelInSampleThresholdPanelData(_BaseKFold):

    def __init__(self):
        pass

    def split(self, X, y=None, groups=None):

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)

        indices = np.arange(n_samples)

        # Indices de début de train pour chaque pays
        IndexPays = list(X.index.levels[0].values)
        Index = list(X.index)
        ListeIDebutTrain = []
        for IRow in range(X.shape[0]):
            if Index[IRow][0] in IndexPays :
                ListeIDebutTrain.append(IRow)
                IndexPays.remove(Index[IRow][0])
        ListeIDebutTrain.append(X.shape[0])

        self.n_splits = len(ListeIDebutTrain)

        # Yield des indices
        for j in range(len(ListeIDebutTrain)-1) :
            IndexTrain = [idx for idx in range(ListeIDebutTrain[j],(ListeIDebutTrain[j+1]))]
            yield (indices[IndexTrain], indices[IndexTrain])


# Forecasting CrossVal2

# _BaseKFold est peut être superflu
# S'assurer que les index sont triés
class PanelCVOutSamplePanelData2(_BaseKFold):

    def __init__(self, ListeDateTest, GapFreqTrain, GapFreqTest, TestSize=1):
        self.ListeDateTest = ListeDateTest
        self.GapFreqTrain = GapFreqTrain
        self.GapFreqTest = GapFreqTest
        self.TestSize = TestSize
        self.n_splits = len(ListeDateTest)

    def split(self, XFreqTrain, XFreqTest, yFreqTrain=None, yFreqTest=None, groups=None):

        # Test initial
        XFreqTrain, yFreqTrain, groups = indexable(XFreqTrain, yFreqTrain, groups)
        XFreqTest, yFreqTest, groups = indexable(XFreqTest, yFreqTest, groups)
        
        # Construction des Indices à retourner
        NumSamplesFreqTrain = _num_samples(XFreqTrain)
        NumSamplesFreqTest = _num_samples(XFreqTest)

        IndicesFreqTrain = np.arange(NumSamplesFreqTrain)
        IndicesFreqTest = np.arange(NumSamplesFreqTest)

        # Construction de la liste correspondant aux des périodes de Train
        ListeIDebutTrainFreqTrain = [int(XFreqTrain.index.get_locs([Pays, slice(None)])[0]) for Pays in XFreqTrain.index.get_level_values(0).drop_duplicates().tolist()]
        ListeIDebutTrainFreqTest = [int(XFreqTest.index.get_locs([Pays, slice(None)])[0]) for Pays in XFreqTest.index.get_level_values(0).drop_duplicates().tolist()]

        # Vérification que les listes ListeIDebutTrainFreqTrain et ListeIDebutTrainFreqTest sont de même longueur
        if len(ListeIDebutTrainFreqTrain) != len(ListeIDebutTrainFreqTest) :
                raise ValueError(("Cannot have a number of train debuts with FreqTrain = {0} different from the number of train debuts with FreqTest = {1}").format(len(ListeIDebutTrainFreqTrain), len(ListeIDebutTrainFreqTest)))
            
        for i in range(0,len(self.ListeDateTest), self.TestSize):
            
            # Construction des listes correspondant aux débuts des périodes de Test
            ListeIDebutTestFreqTrain = [XFreqTrain.index.get_loc(key=(Pays, self.ListeDateTest[i])) for Pays in XFreqTrain.index.get_level_values(0).drop_duplicates().tolist()]
            ListeIDebutTestFreqTest = [XFreqTest.index.get_loc(key=(Pays, self.ListeDateTest[i])) for Pays in XFreqTest.index.get_level_values(0).drop_duplicates().tolist()]
            
            # Vérification que les listes de début de train et de début de test sont le même longueur
            if len(ListeIDebutTrainFreqTrain) != len(ListeIDebutTestFreqTrain) :
                raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrainFreqTrain), len(ListeIDebutTestFreqTrain)))
            elif len(ListeIDebutTrainFreqTest) != len(ListeIDebutTestFreqTest) :
                raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrainFreqTest), len(ListeIDebutTestFreqTest)))
        
            # Construction des Index correspondant aux périodes de Train et de Test
            IndexTrainFreqTrain = np.concatenate([np.arange(ListeIDebutTrainFreqTrain[j], ((ListeIDebutTestFreqTrain)[j] - self.GapFreqTrain[j])) for j in range(len(ListeIDebutTestFreqTrain))], axis=0).tolist()
            IndexTrainFreqTest = np.concatenate([np.arange(ListeIDebutTrainFreqTest[j], ((ListeIDebutTestFreqTest)[j] - self.GapFreqTest[j])) for j in range(len(ListeIDebutTestFreqTest))], axis=0).tolist()
            IndexTestFreqTest = np.concatenate([np.arange(IDebutTest, (IDebutTest+self.TestSize)) for IDebutTest in ListeIDebutTestFreqTest], axis=0).tolist()
            
            yield (IndicesFreqTrain[IndexTrainFreqTrain], IndicesFreqTest[IndexTrainFreqTest], IndicesFreqTest[IndexTestFreqTest])


class OneCountryOneModelCVOutSamplePanelData2(_BaseKFold):
    
    def __init__(self, ListeDateTest, GapFreqTrain, GapFreqTest, TestSize=1):
        self.ListeDateTest = ListeDateTest
        self.GapFreqTrain = GapFreqTrain
        self.GapFreqTest = GapFreqTest
        self.TestSize = TestSize
        self.n_splits = len(ListeDateTest)

    def split(self, XFreqTrain, XFreqTest, yFreqTrain=None, yFreqTest=None, groups=None):

        # Test initial
        XFreqTrain, yFreqTrain, groups = indexable(XFreqTrain, yFreqTrain, groups)
        XFreqTest, yFreqTest, groups = indexable(XFreqTest, yFreqTest, groups)
        
        # Construction des Indices à retourner
        NumSamplesFreqTrain = _num_samples(XFreqTrain)
        NumSamplesFreqTest = _num_samples(XFreqTest)

        IndicesFreqTrain = np.arange(NumSamplesFreqTrain)
        IndicesFreqTest = np.arange(NumSamplesFreqTest)

        # Construction de la liste correspondant aux des périodes de Train
        ListeIDebutTrainFreqTrain = [int(XFreqTrain.index.get_locs([Pays, slice(None)])[0]) for Pays in XFreqTrain.index.get_level_values(0).drop_duplicates().tolist()]
        ListeIDebutTrainFreqTest = [int(XFreqTest.index.get_locs([Pays, slice(None)])[0]) for Pays in XFreqTest.index.get_level_values(0).drop_duplicates().tolist()]

        # Vérification que les listes ListeIDebutTrainFreqTrain et ListeIDebutTrainFreqTest sont de même longueur
        if len(ListeIDebutTrainFreqTrain) != len(ListeIDebutTrainFreqTest) :
                raise ValueError(("Cannot have a number of train debuts with FreqTrain = {0} different from the number of train debuts with FreqTest = {1}").format(len(ListeIDebutTrainFreqTrain), len(ListeIDebutTrainFreqTest)))
         
        for i in range(0,len(self.ListeDateTest), self.TestSize):
            
            # Construction des listes correspondant aux débuts des périodes de Test
            ListeIDebutTestFreqTrain = [XFreqTrain.index.get_loc(key=(Pays, self.ListeDateTest[i])) for Pays in XFreqTrain.index.get_level_values(0).drop_duplicates().tolist()]
            ListeIDebutTestFreqTest = [XFreqTest.index.get_loc(key=(Pays, self.ListeDateTest[i])) for Pays in XFreqTest.index.get_level_values(0).drop_duplicates().tolist()]
            
            # Vérification que les listes de début de train et de début de test sont le même longueur
            if len(ListeIDebutTrainFreqTrain) != len(ListeIDebutTestFreqTrain) :
                raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrainFreqTrain), len(ListeIDebutTestFreqTrain)))
            elif len(ListeIDebutTrainFreqTest) != len(ListeIDebutTestFreqTest) :
                raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrainFreqTest), len(ListeIDebutTestFreqTest)))
        
            for j in range(len(ListeIDebutTestFreqTrain)) :

                # Construction des Index correspondant aux périodes de Train et de Test
                IndexTrainFreqTrain = np.arange(ListeIDebutTrainFreqTrain[j], ((ListeIDebutTestFreqTrain)[j] - self.GapFreqTrain[j]))
                IndexTrainFreqTest = np.arange(ListeIDebutTrainFreqTest[j], ((ListeIDebutTestFreqTest)[j] - self.GapFreqTest[j]))
                IndexTestFreqTest = np.arange(ListeIDebutTestFreqTest[j], (ListeIDebutTestFreqTest[j] + self.TestSize)).tolist()
            
                yield (IndicesFreqTrain[IndexTrainFreqTrain], IndicesFreqTest[IndexTrainFreqTest], IndicesFreqTest[IndexTestFreqTest])

# Ne renvoie que les indices avant la première date de test
class PanelCVInSamplePanelData2(_BaseKFold):
    
    def __init__(self, ListeDateTest):
        self.ListeDateTest = ListeDateTest
        self.n_splits = 1

    def split(self, XFreqTrain, XFreqTest, yFreqTrain=None, yFreqTest=None, groups=None):

        # Test initial
        XFreqTrain, yFreqTrain, groups = indexable(XFreqTrain, yFreqTrain, groups)
        XFreqTest, yFreqTest, groups = indexable(XFreqTest, yFreqTest, groups)
        
        # Construction des Indices à retourner
        NumSamplesFreqTrain = _num_samples(XFreqTrain)
        NumSamplesFreqTest = _num_samples(XFreqTest)

        IndicesFreqTrain = np.arange(NumSamplesFreqTrain)
        IndicesFreqTest = np.arange(NumSamplesFreqTest)

        # Construction des listes correspondant aux débuts des périodes de Train
        ListeIDebutTrainFreqTrain = [int(XFreqTrain.index.get_locs([Pays, slice(None)])[0]) for Pays in XFreqTrain.index.get_level_values(0).drop_duplicates().tolist()]
        ListeIDebutTrainFreqTest = [int(XFreqTest.index.get_locs([Pays, slice(None)])[0]) for Pays in XFreqTest.index.get_level_values(0).drop_duplicates().tolist()]

        # Construction des listes correspondant aux débuts de la première période de Test
        ListeIDebutTestFreqTrain = [XFreqTrain.index.get_loc(key=(Pays, self.ListeDateTest[0])) for Pays in XFreqTrain.index.get_level_values(0).drop_duplicates().tolist()]
        ListeIDebutTestFreqTest = [XFreqTest.index.get_loc(key=(Pays, self.ListeDateTest[0])) for Pays in XFreqTest.index.get_level_values(0).drop_duplicates().tolist()]
            

        # Vérification que les listes sont de même longeur
        if len(ListeIDebutTrainFreqTrain) != len(ListeIDebutTrainFreqTest) :
                raise ValueError(("Cannot have a number of train debuts with FreqTrain = {0} different from the number of train debuts with FreqTest = {1}").format(len(ListeIDebutTrainFreqTrain), len(ListeIDebutTrainFreqTest)))
        elif len(ListeIDebutTrainFreqTrain) != len(ListeIDebutTestFreqTrain) :
            raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrainFreqTrain), len(ListeIDebutTestFreqTrain)))
        elif len(ListeIDebutTrainFreqTest) != len(ListeIDebutTestFreqTest) :
            raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrainFreqTest), len(ListeIDebutTestFreqTest)))

        # Construction des indices de Train et Test
        IndexTrainFreqTrain = np.concatenate([np.arange(ListeIDebutTrainFreqTrain[j], ListeIDebutTestFreqTrain[j]) for j in range(len(ListeIDebutTestFreqTrain))], axis=0).tolist()
        IndexFreqTest = np.concatenate([np.arange(ListeIDebutTrainFreqTest[j], ListeIDebutTestFreqTest[j]) for j in range(len(ListeIDebutTestFreqTest))], axis=0).tolist()

        yield (IndicesFreqTrain[IndexTrainFreqTrain], IndicesFreqTest[IndexFreqTest], IndicesFreqTest[IndexFreqTest])


class OneCountryOneModelCVInSamplePanelData2():
    
    def __init__(self, ListeDateTest):
        self.ListeDateTest = ListeDateTest

    def split(self, XFreqTrain, XFreqTest, yFreqTrain=None, yFreqTest=None, groups=None):

        # Test initial
        XFreqTrain, yFreqTrain, groups = indexable(XFreqTrain, yFreqTrain, groups)
        XFreqTest, yFreqTest, groups = indexable(XFreqTest, yFreqTest, groups)
        
        # Construction des Indices à retourner
        NumSamplesFreqTrain = _num_samples(XFreqTrain)
        NumSamplesFreqTest = _num_samples(XFreqTest)

        IndicesFreqTrain = np.arange(NumSamplesFreqTrain)
        IndicesFreqTest = np.arange(NumSamplesFreqTest)

        # Construction des listes correspondant aux débuts des périodes de Train
        ListeIDebutTrainFreqTrain = [int(XFreqTrain.index.get_locs([Pays, slice(None)])[0]) for Pays in XFreqTrain.index.get_level_values(0).drop_duplicates().tolist()]
        ListeIDebutTrainFreqTest = [int(XFreqTest.index.get_locs([Pays, slice(None)])[0]) for Pays in XFreqTest.index.get_level_values(0).drop_duplicates().tolist()]

        # Construction des listes correspondant aux débuts de la première période de Test
        ListeIDebutTestFreqTrain = [XFreqTrain.index.get_loc(key=(Pays, self.ListeDateTest[0])) for Pays in XFreqTrain.index.get_level_values(0).drop_duplicates().tolist()]
        ListeIDebutTestFreqTest = [XFreqTest.index.get_loc(key=(Pays, self.ListeDateTest[0])) for Pays in XFreqTest.index.get_level_values(0).drop_duplicates().tolist()]
            

        # Vérification que les listes sont de même longeur
        if len(ListeIDebutTrainFreqTrain) != len(ListeIDebutTrainFreqTest) :
                raise ValueError(("Cannot have a number of train debuts with FreqTrain = {0} different from the number of train debuts with FreqTest = {1}").format(len(ListeIDebutTrainFreqTrain), len(ListeIDebutTrainFreqTest)))
        elif len(ListeIDebutTrainFreqTrain) != len(ListeIDebutTestFreqTrain) :
            raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrainFreqTrain), len(ListeIDebutTestFreqTrain)))
        elif len(ListeIDebutTrainFreqTest) != len(ListeIDebutTestFreqTest) :
            raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrainFreqTest), len(ListeIDebutTestFreqTest)))
        
        for j in range(len(ListeIDebutTestFreqTrain)):
            
            # Construction des indices de Train et Test
            IndexTrainFreqTrain = np.arange(ListeIDebutTrainFreqTrain[j], ListeIDebutTestFreqTrain[j])
            IndexFreqTest = np.arange(ListeIDebutTrainFreqTest[j], ListeIDebutTestFreqTest[j]).tolist()

            yield (IndicesFreqTrain[IndexTrainFreqTrain], IndicesFreqTest[IndexFreqTest], IndicesFreqTest[IndexFreqTest])


# CrossVal de complétion de fréquence
class PanelCVCompletionPanelData():

    def __init__(self, DateDebutInpute, DateFinInpute):
        self.DateDebutInpute = DateDebutInpute
        self.DateFinInpute = DateFinInpute


    def split(self, DataTrain, DataInput, groups=None):
        
        # Construction des Indices à retourner
        NumSamplesTrain = _num_samples(DataTrain)
        NumSamplesInput = _num_samples(DataInput)

        IndicesTrain = np.arange(NumSamplesTrain)
        IndicesInput = np.arange(NumSamplesInput)

        # Construction de la liste correspondant aux des périodes de Train
        ListeIDebutTrain = [int(DataTrain.index.get_locs([Pays, slice(None)])[0]) for Pays in DataTrain.index.get_level_values(0).drop_duplicates().tolist()]

        # Construction des listes correspondant aux débuts de la période de test
        if self.DateDebutInpute is None :
            ListeIDebutInput = [int(DataInput.index.get_locs([Pays, slice(None)])[0]) for Pays in DataInput.index.get_level_values(0).drop_duplicates().tolist()]
        else :
            ListeIDebutInput = [DataInput.index.get_loc(key=(Pays, self.DateDebutInpute)) for Pays in DataInput.index.get_level_values(0).drop_duplicates().tolist()]
        
        # Construction des listes correspondant à la fin de la période de Train et de Test
        if self.DateFinInpute is None :
            ListeIFinTrain = [int(DataTrain.index.get_locs([Pays, slice(None)])[-1])+1 for Pays in DataTrain.index.get_level_values(0).drop_duplicates().tolist()]
            ListeIFinInput = [int(DataInput.index.get_locs([Pays, slice(None)])[-1])+1 for Pays in DataInput.index.get_level_values(0).drop_duplicates().tolist()]
        else :
            ListeIFinTrain = [DataTrain.index.get_loc(key=(Pays, self.DateFinInpute)) for Pays in DataTrain.index.get_level_values(0).drop_duplicates().tolist()]
            ListeIFinInput = [DataInput.index.get_loc(key=(Pays, self.DateFinInpute)) for Pays in DataInput.index.get_level_values(0).drop_duplicates().tolist()]
        
        # Vérification que les listes sont de même longueur
        if len(ListeIDebutTrain) != len(ListeIFinTrain) :
            raise ValueError(("Cannot have a number of train debuts = {0} different from the number of train ends = {1}").format(len(ListeIDebutTrain), len(ListeIFinTrain)))
        elif len(ListeIDebutTrain) != len(ListeIDebutInput) :
            raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrain), len(ListeIDebutInput)))
        elif len(ListeIDebutInput) != len(ListeIFinInput) :
            raise ValueError(("Cannot have a number of test debuts = {0} different from the number of test ends = {1}").format(len(ListeIDebutInput), len(ListeIFinInput)))
        
        # Construction des index correspondant aux périodes de Train et d'Imputation
        IndexTrain = np.concatenate([np.arange(ListeIDebutTrain[j], ListeIFinTrain[j]) for j in range(len(ListeIDebutTrain))], axis=0).tolist()
        IndexInpute = np.concatenate([np.arange(ListeIDebutInput[j], ListeIFinInput[j]) for j in range(len(ListeIDebutInput))], axis=0).tolist()
        
        yield (IndicesTrain[IndexTrain], IndicesInput[IndexInpute])


class OneCountryOneModelCVCompletionPanelData(_BaseKFold):
    
    def __init__(self, DateDebutInpute, DateFinInpute):
        self.DateDebutInpute = DateDebutInpute
        self.DateFinInpute = DateFinInpute


    def split(self, DataTrain, DataInput, groups=None):
        
        # Construction des Indices à retourner
        NumSamplesTrain = _num_samples(DataTrain)
        NumSamplesInput = _num_samples(DataInput)

        IndicesTrain = np.arange(NumSamplesTrain)
        IndicesInput = np.arange(NumSamplesInput)

        # Construction de la liste correspondant aux des périodes de Train
        ListeIDebutTrain = [int(DataTrain.index.get_locs([Pays, slice(None)])[0]) for Pays in DataTrain.index.get_level_values(0).drop_duplicates().tolist()]

        # Construction des listes correspondant aux débuts de la période de test
        if self.DateDebutInpute is None :
            ListeIDebutInput = [int(DataInput.index.get_locs([Pays, slice(None)])[0]) for Pays in DataInput.index.get_level_values(0).drop_duplicates().tolist()]
        else :
            ListeIDebutInput = [DataInput.index.get_loc(key=(Pays, self.DateDebutInpute)) for Pays in DataInput.index.get_level_values(0).drop_duplicates().tolist()]
        
        # Construction des listes correspondant à la fin de la période de Train et de Test
        if self.DateFinInpute is None :
            ListeIFinTrain = [int(DataTrain.index.get_locs([Pays, slice(None)])[-1])+1 for Pays in DataTrain.index.get_level_values(0).drop_duplicates().tolist()]
            ListeIFinInput = [int(DataInput.index.get_locs([Pays, slice(None)])[-1])+1 for Pays in DataInput.index.get_level_values(0).drop_duplicates().tolist()]
        else :
            ListeIFinTrain = [DataTrain.index.get_loc(key=(Pays, self.DateFinInpute)) for Pays in DataTrain.index.get_level_values(0).drop_duplicates().tolist()]
            ListeIFinInput = [DataInput.index.get_loc(key=(Pays, self.DateFinInpute)) for Pays in DataInput.index.get_level_values(0).drop_duplicates().tolist()]
        
        # Vérification que les listes sont de même longueur
        if len(ListeIDebutTrain) != len(ListeIFinTrain) :
            raise ValueError(("Cannot have a number of train debuts = {0} different from the number of train ends = {1}").format(len(ListeIDebutTrain), len(ListeIFinTrain)))
        elif len(ListeIDebutTrain) != len(ListeIDebutInput) :
            raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrain), len(ListeIDebutInput)))
        elif len(ListeIDebutInput) != len(ListeIFinInput) :
            raise ValueError(("Cannot have a number of test debuts = {0} different from the number of test ends = {1}").format(len(ListeIDebutInput), len(ListeIFinInput)))
        
        # Construction des index correspondant aux périodes de Train et d'Imputation
        for j in range(len(ListeIDebutTrain)):
            IndexTrain = np.arange(ListeIDebutTrain[j], ListeIFinTrain[j]).tolist()
            IndexInpute = np.arange(ListeIDebutInput[j], ListeIFinInput[j]).tolist()
            
            yield (IndicesTrain[IndexTrain], IndicesInput[IndexInpute])