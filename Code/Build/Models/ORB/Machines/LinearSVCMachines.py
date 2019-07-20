import numpy #for numpy storage
import os #to find files
import time #for time to complete
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm
import pickle


#Import Training Data & Labels
data = numpy.load("Data/1UkiORBTrainingData.npy")
data = data.reshape(1476,(10*256))
isnot = numpy.load("Data/1UkiORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.25, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "1UKIORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None

data = numpy.load("Data/2LReORBTrainingData.npy")
data = data.reshape(1656,(10*256))
isnot = numpy.load("Data/2LReORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.25, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "2LReORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None

odata = numpy.load("Data/3MinORBTrainingData.npy")
data = odata.reshape(1062,(10*256))
isnot = numpy.load("Data/3MinORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.25, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "3MinORBProb.LinearSVC", "wb" ))

############################################
############################################

data = None
isnot = None
machine = None

odata = numpy.load("Data/4HReORBTrainingData.npy")
data = odata.reshape(1724,(10*256))
isnot = numpy.load("Data/4HReORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.25, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "4HRenORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None

odata = numpy.load("Data/5ERenORBTrainingData.npy")
data = odata.reshape(1796,(10*256))
isnot = numpy.load("Data/5ERenORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.25, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "5ERenORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None

odata = numpy.load("Data/6PopORBTrainingData.npy")
data = odata.reshape(1924,(10*256))
isnot = numpy.load("Data/6PopORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.25, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "6PopORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None


odata = numpy.load("Data/7CFPORBTrainingData.npy")
data = odata.reshape(1080,(10*256))
isnot = numpy.load("Data/7CFPORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.25, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "7CFPORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None


odata = numpy.load("Data/8RocORBTrainingData.npy")
data = odata.reshape(2688,(10*256))
isnot = numpy.load("Data/8RocORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.03125, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "8RocORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None


odata = numpy.load("Data/9CubORBTrainingData.npy")
data = odata.reshape(2686,(10*256))
isnot = numpy.load("Data/9CubORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.25, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "9CubORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None


odata = numpy.load("Data/10NAPORBTrainingData.npy")
data = odata.reshape(3078,(10*256))
isnot = numpy.load("Data/10NAPORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.03125, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "10NAPORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None


odata = numpy.load("Data/11NRORBTrainingData.npy")
data = odata.reshape(3280,(10*256))
isnot = numpy.load("Data/11NRORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.03125, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "11NRORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None


odata = numpy.load("Data/12AEORBTrainingData.npy")
data = odata.reshape(3562,(10*256))
isnot = numpy.load("Data/12AEORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.0625, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "12AEORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None


odata = numpy.load("Data/13BORBTrainingData.npy")
data = odata.reshape(5276,(10*256))
isnot = numpy.load("Data/13BORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.03125, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "13BORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None


odata = numpy.load("Data/14ANMORBTrainingData.npy")
data = odata.reshape(5538,(10*256))
isnot = numpy.load("Data/14ANMORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.03125, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "14ANMORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None


odata = numpy.load("Data/15SymORBTrainingData.npy")
data = odata.reshape(5796,(10*256))
isnot = numpy.load("Data/15SymORBTrainingDataLabels.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.0625, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "15SymORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None

odata = numpy.load("Data/16PIORBTrainingDataSPLIT.npy")
data = odata.reshape(4139,(10*256))
isnot = numpy.load("Data/16PIORBTrainingDataLabelsSPLIT.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.03125, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "16PIORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None

odata = numpy.load("Data/17EORBTrainingDataSPLIT.npy")
data = odata.reshape(4323,(10*256))
isnot = numpy.load("Data/17EORBTrainingDataLabelsSPLIT.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.0625, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "17EORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None

odata = numpy.load("Data/18RomORBTrainingDataSPLIT.npy")
data = odata.reshape(4525,(10*256))
isnot = numpy.load("Data/18RomORBTrainingDataLabelsSPLIT.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.03125, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "18RomORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None

odata = numpy.load("Data/19RelORBTrainingDataSPLIT.npy")
data = odata.reshape(4511,(10*256))
isnot = numpy.load("Data/19RelORBTrainingDataLabelsSPLIT.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.03125, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "19RelORBProb.LinearSVC", "wb" ))

#######
#######

data = None
isnot = None
machine = None

odata = numpy.load("Data/20ImpORBTrainingDataSPLIT.npy")
data = odata.reshape(4511,(10*256))
isnot = numpy.load("Data/20ImpORBTrainingDataLabelsSPLIT.npy")

print ("Training Test Data")
machine = svm.LinearSVC(C = 0.0625, random_state = 2,max_iter = 10000000,loss="hinge")
machine = CalibratedClassifierCV(machine, cv = 3)
machine.fit(data,isnot)

pickle.dump(machine, open( "20ImpORBProb.LinearSVC", "wb" ))

#######
#######



