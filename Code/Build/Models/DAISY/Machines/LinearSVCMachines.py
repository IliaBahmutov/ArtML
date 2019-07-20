import numpy #for numpy storage
import os #to find files
import time #for time to complete
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
import pickle

#Import Training Data & Labels
data = numpy.load("Data/1UkiTrainingData.npy")
data = data.reshape(1634,(3*3*78))
isnot = numpy.load("Data/1UkiTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 64, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "1UKIDaisyProb.LinearSVC", "wb" ))

#######
#######

data = numpy.load("Data/2LReTrainingData.npy")
data = data.reshape(1790,(3*3*78))
isnot = numpy.load("Data/2LReTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 128, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "2LReDaisyProb.LinearSVC", "wb" ))

#######
#######

odata = numpy.load("Data/3MinTrainingData.npy")
data = odata.reshape(1860,(3*3*78))
isnot = numpy.load("Data/3MinTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 16, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "3MinDaisyProb.LinearSVC", "wb" ))

############################################
############################################

odata = numpy.load("Data/4HReTrainingData.npy")
data = odata.reshape(1862,(3*3*78))
isnot = numpy.load("Data/4HReTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 16, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "4HRenDaisyProb.LinearSVC", "wb" ))

#######
#######

odata = numpy.load("Data/5ERenTrainingData.npy")
data = odata.reshape(1938,(3*3*78))
isnot = numpy.load("Data/5ERenTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.25, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "5ERenDaisyProb.LinearSVC", "wb" ))

#######
#######


odata = numpy.load("Data/6PopTrainingData.npy")
data = odata.reshape(2076,(3*3*78))
isnot = numpy.load("Data/6PopTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 1, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "6PopDaisyProb.LinearSVC", "wb" ))

#######
#######


odata = numpy.load("Data/7CFPTrainingData.npy")
data = odata.reshape(2262,(3*3*78))
isnot = numpy.load("Data/7CFPTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 4, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "7CFPDaisyProb.LinearSVC", "wb" ))

#######
#######


odata = numpy.load("Data/8RocTrainingData.npy")
data = odata.reshape(2924,(3*3*78))
isnot = numpy.load("Data/8RocTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 16, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "8RocDaisyProb.LinearSVC", "wb" ))

#######
#######


odata = numpy.load("Data/9CubTrainingData.npy")
data = odata.reshape(3150,(3*3*78))
isnot = numpy.load("Data/9CubTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.5, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "9CubDaisyProb.LinearSVC", "wb" ))

#######
#######


odata = numpy.load("Data/10NAPTrainingData.npy")
data = odata.reshape(3368,(3*3*78))
isnot = numpy.load("Data/10NAPTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 16, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "10NAPDaisyProb.LinearSVC", "wb" ))

#######
#######


odata = numpy.load("Data/11NRTrainingData.npy")
data = odata.reshape(3572,(3*3*78))
isnot = numpy.load("Data/11NRTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 2, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "11NRDaisyProb.LinearSVC", "wb" ))

#######
#######


odata = numpy.load("Data/12AETrainingData.npy")
data = odata.reshape(3894,(3*3*78))
isnot = numpy.load("Data/12AETrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 64, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "12AEDaisyProb.LinearSVC", "wb" ))

#######
#######


odata = numpy.load("Data/13BTrainingData.npy")
data = odata.reshape(5938,(3*3*78))
isnot = numpy.load("Data/13BTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 128, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "13BDaisyProb.LinearSVC", "wb" ))

#######
#######


odata = numpy.load("Data/14ANMTrainingData.npy")
data = odata.reshape(6068,(3*3*78))
isnot = numpy.load("Data/14ANMTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 64, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "14ANMDaisyProb.LinearSVC", "wb" ))

#######
#######


odata = numpy.load("Data/15SymTrainingData.npy")
data = odata.reshape(6340,(3*3*78))
isnot = numpy.load("Data/15SymTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 16, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "15SymDaisyProb.LinearSVC", "wb" ))

#######
#######

odata = numpy.load("Data/16PITrainingData.npy")
data = odata.reshape(9032,(3*3*78))
isnot = numpy.load("Data/16PITrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 8, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "16PIDaisyProb.LinearSVC", "wb" ))

#######
#######

odata = numpy.load("Data/17ETrainingData.npy")
data = odata.reshape(9430,(3*3*78))
isnot = numpy.load("Data/17ETrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 8, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "17EDaisyProb.LinearSVC", "wb" ))

#######
#######

odata = numpy.load("Data/18RomTrainingData.npy")
data = odata.reshape(9826,(3*3*78))
isnot = numpy.load("Data/18RomTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 16, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "18RomDaisyProb.LinearSVC", "wb" ))

#######
#######

odata = numpy.load("Data/19RelTrainingData.npy")
data = odata.reshape(9826,(3*3*78))
isnot = numpy.load("Data/19RelTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 64, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "19RelDaisyProb.LinearSVC", "wb" ))

#######
#######

odata = numpy.load("Data/20ImpTrainingData.npy")
data = odata.reshape(9826,(3*3*78))
isnot = numpy.load("Data/20ImpTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 4, random_state = 2,max_iter = 1000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "20ImpDaisyProb.LinearSVC", "wb" ))

#######
#######



