# Modules de bases
import numpy as np
import pandas as pd

# Modules Scikit-learn
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

class PanelCVRollingOutSamplePanelData3():

    def __init__(self, ListeDateTest, Horizon, TestSize=1):
        self.ListeDateTest = ListeDateTest
        self.Horizon = Horizon
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
            IndexTrain = np.concatenate([np.arange(ListeIDebutTrain[j], ((ListeIDebutTest)[j] - self.Horizon)) for j in range(len(ListeIDebutTest))], axis=0).tolist()
            IndexTest= np.concatenate([np.arange(IDebutTest, (IDebutTest+self.TestSize)) for IDebutTest in ListeIDebutTest], axis=0).tolist()
            
            yield (Indices[IndexTrain], Indices[IndexTest])


class PanelCVFixOutSamplePanelData3():

    def __init__(self, ListeDateTest, Horizon):

        self.ListeDateTest = ListeDateTest
        self.Horizon = Horizon

    def split(self, X, y=None, groups=None):

        # Test initial
        X, y, groups = indexable(X, y, groups)
        
        # Construction des Indices à retourner
        NumSamples = _num_samples(X)

        Indices = np.arange(NumSamples)

        # Construction de la liste correspondant aux des périodes de Train
        ListeIDebutTrain = [int(X.index.get_locs([Pays, slice(None)])[0]) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]

        # Date de début et de fin de test
        DateDebutTest = self.ListeDateTest[0]
        DateFinTest = self.ListeDateTest[-1]

        # Construction des listes correspondant aux débuts des périodes de Test
        ListeIDebutTest = [X.index.get_loc(key=(Pays, DateDebutTest)) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]
        ListeIFinTest = [X.index.get_loc(key=(Pays, DateFinTest)) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]

        # Vérification que les listes de début de train et de début de test sont le même longueur
        if len(ListeIDebutTrain) != len(ListeIDebutTest) :
            raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrain), len(ListeIDebutTest)))
        elif len(ListeIDebutTest) != len(ListeIFinTest) :
            raise ValueError(("Cannot have a number of test debuts = {0} different from the number of test ends = {1}").format(len(ListeIDebutTest), len(ListeIFinTest)))

        # Construction des Index correspondant aux périodes de Train et de Test
        IndexTrain = np.concatenate([np.arange(ListeIDebutTrain[j], (ListeIDebutTest[j] - self.Horizon)) for j in range(len(ListeIDebutTest))], axis=0).tolist()
        IndexTest= np.concatenate([np.arange(ListeIDebutTest[j], (ListeIFinTest[j]+1)) for j in range(len(ListeIDebutTest))], axis=0).tolist()
            
        yield (Indices[IndexTrain], Indices[IndexTest])


class OneCountryOneModelCVRollingOutSamplePanelData3():
    
    def __init__(self, ListeDateTest, Horizon, TestSize=1):
        
        self.ListeDateTest = ListeDateTest
        self.Horizon = Horizon
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
                IndexTrain = np.arange(ListeIDebutTrain[j], ((ListeIDebutTest)[j] - self.Horizon))
                IndexTest = np.arange(ListeIDebutTest[j], (ListeIDebutTest[j] + self.TestSize)).tolist()
            
                yield (Indices[IndexTrain], Indices[IndexTest])


class OneCountryOneModelCVFixOutSamplePanelData3():
    
    def __init__(self, ListeDateTest, Horizon):
        
        self.ListeDateTest = ListeDateTest
        self.Horizon = Horizon

    def split(self, X, y=None, groups=None):

        # Test initial
        X, y, groups = indexable(X, y, groups)
        
        # Construction des Indices à retourner
        NumSamples = _num_samples(X)

        Indices = np.arange(NumSamples)

        # Construction de la liste correspondant aux des périodes de Train
        ListeIDebutTrain = [int(X.index.get_locs([Pays, slice(None)])[0]) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]

        # Date de début et de fin de test
        DateDebutTest = self.ListeDateTest[0]
        DateFinTest = self.ListeDateTest[-1]

        # Construction des listes correspondant aux débuts des périodes de Test
        ListeIDebutTest = [X.index.get_loc(key=(Pays, DateDebutTest)) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]
        # Construction des listes correspondant aux fins des périodes de Test
        ListeIFinTest = [X.index.get_loc(key=(Pays, DateFinTest)) for Pays in X.index.get_level_values(0).drop_duplicates().tolist()]
        
        # Vérification que les listes de début de train et de début de test sont le même longueur
        if len(ListeIDebutTrain) != len(ListeIDebutTest) :
            raise ValueError(("Cannot have a number of train debuts = {0} different from the number of test debuts = {1}").format(len(ListeIDebutTrain), len(ListeIDebutTest)))
        elif len(ListeIDebutTest) != len(ListeIFinTest) :
            raise ValueError(("Cannot have a number of test debuts = {0} different from the number of test ends = {1}").format(len(ListeIDebutTest), len(ListeIFinTest)))

        for j in range(len(ListeIDebutTest)) :

            # Construction des Index correspondant aux périodes de Train et de Test
            IndexTrain = np.arange(ListeIDebutTrain[j], ((ListeIDebutTest)[j] - self.Horizon))
            IndexTest = np.arange(ListeIDebutTest[j], (ListeIFinTest[j] + 1)).tolist()
            
            yield (Indices[IndexTrain], Indices[IndexTest])