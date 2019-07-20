import numpy #for numpy storage
import os #to find files
import time #for time to complete
from sklearn.tree import DecisionTreeClassifier
import pickle

#Import Training Data & Labels
data = numpy.load("Data/1UkiTrainingData.npy")
data = data.reshape(1634,(3*3*78))
isnot = numpy.load("Data/1UkiTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None ,max_depth = 13,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "1UKIDaisyProb.DT", "wb" ))

#######
#######

data = numpy.load("Data/2LReTrainingData.npy")
data = data.reshape(1790,(3*3*78))
isnot = numpy.load("Data/2LReTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 15,min_samples_split = 0.0610, min_samples_leaf = 15)
machine.fit(data,isnot)

pickle.dump(machine, open( "2LReDaisyProb.DT", "wb" ))

#######
#######

odata = numpy.load("Data/3MinTrainingData.npy")
data = odata.reshape(1860,(3*3*78))
isnot = numpy.load("Data/3MinTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 16,min_samples_split = 8)
machine.fit(data,isnot)

pickle.dump(machine, open( "3MinDaisyProb.DT", "wb" ))

############################################
############################################

odata = numpy.load("Data/4HReTrainingData.npy")
data = odata.reshape(1862,(3*3*78))
isnot = numpy.load("Data/4HReTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 12,min_samples_split = 64)
machine.fit(data,isnot)

pickle.dump(machine, open( "4HRenDaisyProb.DT", "wb" ))

#######
#######

odata = numpy.load("Data/5ERenTrainingData.npy")
data = odata.reshape(1938,(3*3*78))
isnot = numpy.load("Data/5ERenTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 20,min_samples_split = 4)
machine.fit(data,isnot)

pickle.dump(machine, open( "5ERenDaisyProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/6PopTrainingData.npy")
data = odata.reshape(2076,(3*3*78))
isnot = numpy.load("Data/6PopTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 15,min_samples_split = 8)
machine.fit(data,isnot)

pickle.dump(machine, open( "6PopDaisyProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/7CFPTrainingData.npy")
data = odata.reshape(2262,(3*3*78))
isnot = numpy.load("Data/7CFPTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 13,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "7CFPDaisyProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/8RocTrainingData.npy")
data = odata.reshape(2924,(3*3*78))
isnot = numpy.load("Data/8RocTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 17,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "8RocDaisyProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/9CubTrainingData.npy")
data = odata.reshape(3150,(3*3*78))
isnot = numpy.load("Data/9CubTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 13,min_samples_split = 64)
machine.fit(data,isnot)

pickle.dump(machine, open( "9CubDaisyProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/10NAPTrainingData.npy")
data = odata.reshape(3368,(3*3*78))
isnot = numpy.load("Data/10NAPTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 16,min_samples_split = 16)
machine.fit(data,isnot)

pickle.dump(machine, open( "10NAPDaisyProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/11NRTrainingData.npy")
data = odata.reshape(3572,(3*3*78))
isnot = numpy.load("Data/11NRTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 16,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "11NRDaisyProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/12AETrainingData.npy")
data = odata.reshape(3894,(3*3*78))
isnot = numpy.load("Data/12AETrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 8,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "12AEDaisyProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/13BTrainingData.npy")
data = odata.reshape(5938,(3*3*78))
isnot = numpy.load("Data/13BTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 29,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "13BDaisyProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/14ANMTrainingData.npy")
data = odata.reshape(6068,(3*3*78))
isnot = numpy.load("Data/14ANMTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 20,min_samples_split = 16)
machine.fit(data,isnot)

pickle.dump(machine, open( "14ANMDaisyProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/15SymTrainingData.npy")
data = odata.reshape(6340,(3*3*78))
isnot = numpy.load("Data/15SymTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 30,min_samples_split = 32)
machine.fit(data,isnot)

pickle.dump(machine, open( "15SymDaisyProb.DT", "wb" ))

#######
#######

odata = numpy.load("Data/16PITrainingData.npy")
data = odata.reshape(9032,(3*3*78))
isnot = numpy.load("Data/16PITrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 29,min_samples_split = 128)
machine.fit(data,isnot)

pickle.dump(machine, open( "16PIDaisyProb.DT", "wb" ))

#######
#######

odata = numpy.load("Data/17ETrainingData.npy")
data = odata.reshape(9430,(3*3*78))
isnot = numpy.load("Data/17ETrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 25,min_samples_split = 8)
machine.fit(data,isnot)

pickle.dump(machine, open( "17EDaisyProb.DT", "wb" ))

#######
#######

odata = numpy.load("Data/18RomTrainingData.npy")
data = odata.reshape(9826,(3*3*78))
isnot = numpy.load("Data/18RomTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 12,min_samples_split = 0.03125)
machine.fit(data,isnot)

pickle.dump(machine, open( "18RomDaisyProb.DT", "wb" ))

#######
#######

odata = numpy.load("Data/19RelTrainingData.npy")
data = odata.reshape(9826,(3*3*78))
isnot = numpy.load("Data/19RelTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 29,min_samples_split = 128)
machine.fit(data,isnot)

pickle.dump(machine, open( "19RelDaisyProb.DT", "wb" ))

#######
#######

odata = numpy.load("Data/20ImpTrainingData.npy")
data = odata.reshape(9826,(3*3*78))
isnot = numpy.load("Data/20ImpTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=None,max_depth = 26,min_samples_split = 0.03125)
machine.fit(data,isnot)

pickle.dump(machine, open( "20ImpDaisyProb.DT", "wb" ))

#######
#######



