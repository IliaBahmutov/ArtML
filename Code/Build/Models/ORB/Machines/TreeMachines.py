import numpy #for numpy storage
import os #to find files
import time #for time to complete
from sklearn.tree import DecisionTreeClassifier
import pickle

#Import Training Data & Labels
data = numpy.load("Data/1UkiORBTrainingData.npy")
data = data.reshape(1476,(10*256))
isnot = numpy.load("Data/1UkiORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 9,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "1UKIORBProb.DT", "wb" ))

#######
#######

data = numpy.load("Data/2LReORBTrainingData.npy")
data = data.reshape(1656,(10*256))
isnot = numpy.load("Data/2LReORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 9,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "2LReORBProb.DT", "wb" ))

#######
#######

odata = numpy.load("Data/3MinORBTrainingData.npy")
data = odata.reshape(1062,(10*256))
isnot = numpy.load("Data/3MinORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 16,min_samples_split = 16)
machine.fit(data,isnot)

pickle.dump(machine, open( "3MinORBProb.DT", "wb" ))

############################################
############################################

odata = numpy.load("Data/4HReORBTrainingData.npy")
data = odata.reshape(1724,(10*256))
isnot = numpy.load("Data/4HReORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 9,min_samples_split = 4)
machine.fit(data,isnot)

pickle.dump(machine, open( "4HRenORBProb.DT", "wb" ))

#######
#######

odata = numpy.load("Data/5ERenORBTrainingData.npy")
data = odata.reshape(1796,(10*256))
isnot = numpy.load("Data/5ERenORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 9,min_samples_split = 64)
machine.fit(data,isnot)

pickle.dump(machine, open( "5ERenORBProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/6PopORBTrainingData.npy")
data = odata.reshape(1924,(10*256))
isnot = numpy.load("Data/6PopORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 8,min_samples_split = 8)
machine.fit(data,isnot)

pickle.dump(machine, open( "6PopORBProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/7CFPORBTrainingData.npy")
data = odata.reshape(1080,(10*256))
isnot = numpy.load("Data/7CFPORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 9,min_samples_split = 8)
machine.fit(data,isnot)

pickle.dump(machine, open( "7CFPORBProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/8RocORBTrainingData.npy")
data = odata.reshape(2688,(10*256))
isnot = numpy.load("Data/8RocORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 10,min_samples_split = 4)
machine.fit(data,isnot)

pickle.dump(machine, open( "8RocORBProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/9CubORBTrainingData.npy")
data = odata.reshape(2686,(10*256))
isnot = numpy.load("Data/9CubORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 12,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "9CubORBProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/10NAPORBTrainingData.npy")
data = odata.reshape(3078,(10*256))
isnot = numpy.load("Data/10NAPORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 7,min_samples_split = 32)
machine.fit(data,isnot)

pickle.dump(machine, open( "10NAPORBProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/11NRORBTrainingData.npy")
data = odata.reshape(3280,(10*256))
isnot = numpy.load("Data/11NRORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 5,min_samples_split = 0.0625)
machine.fit(data,isnot)

pickle.dump(machine, open( "11NRORBProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/12AEORBTrainingData.npy")
data = odata.reshape(3562,(10*256))
isnot = numpy.load("Data/12AEORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 6,min_samples_split = 64)
machine.fit(data,isnot)

pickle.dump(machine, open( "12AEORBProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/13BORBTrainingData.npy")
data = odata.reshape(5276,(10*256))
isnot = numpy.load("Data/13BORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 6,min_samples_split = 0.125)
machine.fit(data,isnot)

pickle.dump(machine, open( "13BORBProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/14ANMORBTrainingData.npy")
data = odata.reshape(5538,(10*256))
isnot = numpy.load("Data/14ANMORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 8,min_samples_split = 32)
machine.fit(data,isnot)

pickle.dump(machine, open( "14ANMORBProb.DT", "wb" ))

#######
#######


odata = numpy.load("Data/15SymORBTrainingData.npy")
data = odata.reshape(5796,(10*256))
isnot = numpy.load("Data/15SymORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 12,min_samples_split = 4)
machine.fit(data,isnot)

pickle.dump(machine, open( "15SymORBProb.DT", "wb" ))

#######
#######

odata = numpy.load("Data/16PIORBTrainingData.npy")
data = odata.reshape(8278,(10*256))
isnot = numpy.load("Data/16PIORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 4,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "16PIORBProb.DT", "wb" ))

#######
#######

odata = numpy.load("Data/17EORBTrainingData.npy")
data = odata.reshape(8646,(10*256))
isnot = numpy.load("Data/17EORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 9,min_samples_split = 0.5)
machine.fit(data,isnot)

pickle.dump(machine, open( "17EORBProb.DT", "wb" ))

#######
#######

odata = numpy.load("Data/18RomORBTrainingData.npy")
data = odata.reshape(9050,(10*256))
isnot = numpy.load("Data/18RomORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 4,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "18RomORBProb.DT", "wb" ))

#######
#######

odata = numpy.load("Data/19RelORBTrainingData.npy")
data = odata.reshape(9022,(10*256))
isnot = numpy.load("Data/19RelORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 12,min_samples_split = 4)
machine.fit(data,isnot)

pickle.dump(machine, open( "19RelORBProb.DT", "wb" ))

#######
#######

odata = numpy.load("Data/20ImpORBTrainingData.npy")
data = odata.reshape(9022,(10*256))
isnot = numpy.load("Data/20ImpORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = DecisionTreeClassifier(splitter='best', random_state=1, max_depth = 10,min_samples_split = 16)
machine.fit(data,isnot)

pickle.dump(machine, open( "20ImpORBProb.DT", "wb" ))

#######
#######