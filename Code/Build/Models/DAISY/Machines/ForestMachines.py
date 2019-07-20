import numpy #for numpy storage
import os #to find files
import time #for time to complete
from sklearn.ensemble import RandomForestClassifier
import pickle

#Import Training Data & Labels
data = numpy.load("Data/1UkiTrainingData.npy")
data = data.reshape(1634,(3*3*78))
isnot = numpy.load("Data/1UkiTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 12,min_samples_split = 8)
machine.fit(data,isnot)

pickle.dump(machine, open( "1UKIDaisyProb.DF", "wb" ))

#######
#######

data = numpy.load("Data/2LReTrainingData.npy")
data = data.reshape(1790,(3*3*78))
isnot = numpy.load("Data/2LReTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 12,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "2LReDaisyProb.DF", "wb" ))

#######
#######

odata = numpy.load("Data/3MinTrainingData.npy")
data = odata.reshape(1860,(3*3*78))
isnot = numpy.load("Data/3MinTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 11,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "3MinDaisyProb.DF", "wb" ))

############################################
############################################

odata = numpy.load("Data/4HReTrainingData.npy")
data = odata.reshape(1862,(3*3*78))
isnot = numpy.load("Data/4HReTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 17,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "4HRenDaisyProb.DF", "wb" ))

#######
#######

odata = numpy.load("Data/5ERenTrainingData.npy")
data = odata.reshape(1938,(3*3*78))
isnot = numpy.load("Data/5ERenTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 18,min_samples_split = 8)
machine.fit(data,isnot)

pickle.dump(machine, open( "5ERenDaisyProb.DF", "wb" ))

#######
#######


odata = numpy.load("Data/6PopTrainingData.npy")
data = odata.reshape(2076,(3*3*78))
isnot = numpy.load("Data/6PopTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 11,min_samples_split = 8)
machine.fit(data,isnot)

pickle.dump(machine, open( "6PopDaisyProb.DF", "wb" ))

#######
#######


odata = numpy.load("Data/7CFPTrainingData.npy")
data = odata.reshape(2262,(3*3*78))
isnot = numpy.load("Data/7CFPTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 12,min_samples_split = 4)
machine.fit(data,isnot)

pickle.dump(machine, open( "7CFPDaisyProb.DF", "wb" ))

#######
#######


odata = numpy.load("Data/8RocTrainingData.npy")
data = odata.reshape(2924,(3*3*78))
isnot = numpy.load("Data/8RocTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 14,min_samples_split = 16)
machine.fit(data,isnot)

pickle.dump(machine, open( "8RocDaisyProb.DF", "wb" ))

#######
#######


odata = numpy.load("Data/9CubTrainingData.npy")
data = odata.reshape(3150,(3*3*78))
isnot = numpy.load("Data/9CubTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 16,min_samples_split = 8)
machine.fit(data,isnot)

pickle.dump(machine, open( "9CubDaisyProb.DF", "wb" ))

#######
#######


odata = numpy.load("Data/10NAPTrainingData.npy")
data = odata.reshape(3368,(3*3*78))
isnot = numpy.load("Data/10NAPTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 15,min_samples_split = 4)
machine.fit(data,isnot)

pickle.dump(machine, open( "10NAPDaisyProb.DF", "wb" ))

#######
#######


odata = numpy.load("Data/11NRTrainingData.npy")
data = odata.reshape(3572,(3*3*78))
isnot = numpy.load("Data/11NRTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 22,min_samples_split = 8)
machine.fit(data,isnot)

pickle.dump(machine, open( "11NRDaisyProb.DF", "wb" ))

#######
#######


odata = numpy.load("Data/12AETrainingData.npy")
data = odata.reshape(3894,(3*3*78))
isnot = numpy.load("Data/12AETrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 28,min_samples_split = 4)
machine.fit(data,isnot)

pickle.dump(machine, open( "12AEDaisyProb.DF", "wb" ))

#######
#######


odata = numpy.load("Data/13BTrainingData.npy")
data = odata.reshape(5938,(3*3*78))
isnot = numpy.load("Data/13BTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 12,min_samples_split = 8)
machine.fit(data,isnot)

pickle.dump(machine, open( "13BDaisyProb.DF", "wb" ))

#######
#######


odata = numpy.load("Data/14ANMTrainingData.npy")
data = odata.reshape(6068,(3*3*78))
isnot = numpy.load("Data/14ANMTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 30,min_samples_split = 32)
machine.fit(data,isnot)

pickle.dump(machine, open( "14ANMDaisyProb.DF", "wb" ))

#######
#######


odata = numpy.load("Data/15SymTrainingData.npy")
data = odata.reshape(6340,(3*3*78))
isnot = numpy.load("Data/15SymTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 30,min_samples_split = 128)
machine.fit(data,isnot)

pickle.dump(machine, open( "15SymDaisyProb.DF", "wb" ))

#######
#######

odata = numpy.load("Data/16PITrainingData.npy")
data = odata.reshape(9032,(3*3*78))
isnot = numpy.load("Data/16PITrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 16,min_samples_split = 2)
machine.fit(data,isnot)

pickle.dump(machine, open( "16PIDaisyProb.DF", "wb" ))

#######
#######

odata = numpy.load("Data/17ETrainingData.npy")
data = odata.reshape(9430,(3*3*78))
isnot = numpy.load("Data/17ETrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 30,min_samples_split = 32)
machine.fit(data,isnot)

pickle.dump(machine, open( "17EDaisyProb.DF", "wb" ))

#######
#######

odata = numpy.load("Data/18RomTrainingData.npy")
data = odata.reshape(9826,(3*3*78))
isnot = numpy.load("Data/18RomTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 27,min_samples_split = 64)
machine.fit(data,isnot)

pickle.dump(machine, open( "18RomDaisyProb.DF", "wb" ))

#######
#######

odata = numpy.load("Data/19RelTrainingData.npy")
data = odata.reshape(9826,(3*3*78))
isnot = numpy.load("Data/19RelTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 28,min_samples_split = 64)
machine.fit(data,isnot)

pickle.dump(machine, open( "19RelDaisyProb.DF", "wb" ))

#######
#######

odata = numpy.load("Data/20ImpTrainingData.npy")
data = odata.reshape(9826,(3*3*78))
isnot = numpy.load("Data/20ImpTrainingDataLabels.npy")

print ("Training Test Data")
machine = RandomForestClassifier(random_state=1,n_estimators = 100,max_depth = 100000,min_samples_split = 10000000)
machine.fit(data,isnot)

pickle.dump(machine, open( "20ImpDaisyProb.DF", "wb" ))

#######
#######



