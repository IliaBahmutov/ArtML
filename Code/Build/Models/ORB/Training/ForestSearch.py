import numpy #for numpy storage
import os #to find files
import time #for time to complete
from sklearn.ensemble import RandomForestClassifier
import pickle
from statistics import mean
start = time.time()

#Import Training Data & Labels

odata = numpy.load("Data/20ImpORBTrainingData.npy")
data = odata.reshape(9022,(10*256))
isnot = numpy.load("Data/20ImpORBTrainingDataLabels.npy")

#import wrong test data
wdata0 = numpy.load("Data/UkiORBTestingData.npy")
wdata0 = wdata0.reshape(174,(10*256))

wdata1 = numpy.load("Data/LReORBTestingData.npy")
wdata1 = wdata1.reshape(172,(10*256))

wdata2 = numpy.load("Data/MinORBTestingData.npy")
wdata2 = wdata2.reshape(90,(10*256))

wdata3 = numpy.load("Data/HReORBTestingData.npy")
wdata3 = wdata3.reshape(187,(10*256))

wdata4 = numpy.load("Data/EreORBTestingData.npy")
wdata4 = wdata4.reshape(197,(10*256))

wdata5 = numpy.load("Data/PopORBTestingData.npy")
wdata5 = wdata5.reshape(208,(10*256))

wdata6 = numpy.load("Data/CFPORBTestingData.npy")
wdata6 = wdata6.reshape(93,(10*256))

wdata7 = numpy.load("Data/RocORBTestingData.npy")
wdata7 = wdata7.reshape(282,(10*256))

wdata8 = numpy.load("Data/CubORBTestingData.npy")
wdata8 = wdata8.reshape(317,(10*256))

wdata9 = numpy.load("Data/NAPORBTestingData.npy")
wdata9 = wdata9.reshape(355,(10*256))

wdata10 = numpy.load("Data/NreORBTestingData.npy")
wdata10 = wdata10.reshape(358,(10*256))

wdata11 = numpy.load("Data/AExORBTestingData.npy")
wdata11 = wdata11.reshape(380,(10*256))

wdata12 = numpy.load("Data/BORBTestingData.npy")
wdata12 = wdata12.reshape(589,(10*256))

wdata13 = numpy.load("Data/ANMORBTestingData.npy")
wdata13 = wdata13.reshape(633,(10*256))

wdata14 = numpy.load("Data/SymORBTestingData.npy")
wdata14 = wdata14.reshape(638,(10*256))

wdata15 = numpy.load("Data/PImORBTestingData.npy")
wdata15 = wdata15.reshape(946,(10*256))

wdata16 = numpy.load("Data/ExpORBTestingData.npy")
wdata16 = wdata16.reshape(981,(10*256))

wdata17 = numpy.load("Data/RomORBTestingData.npy")
wdata17 = wdata17.reshape(964,(10*256))

wdata18 = numpy.load("Data/RelORBTestingData.npy")
wdata18 = wdata18.reshape(1543,(10*256))

wdata19 = numpy.load("Data/ImpORBTestingData.npy")
wdata19 = wdata19.reshape(1913,(10*256))

"""
cval = 21 length from 2^-5 to 2^15
gval = 18 length from 2^-15 to 2^2
"""

madepth = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] #1,2,3 removed
mins = [0.03125,0.0625,0.125,0.25,0.5,1.0,2,4,8,16,32,64,128] #,256,512,1024 removed

print ("Training Test Data")

results = [0] *19
checkagainst = [0]
falsepositive = 0;
falsenegative = 0;
truepositive = 0;
truenegative = 0;


for manums in madepth: 
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	for minvals in mins:
		machine = RandomForestClassifier(n_estimators = 10, random_state=1,max_depth = manums,min_samples_split = minvals)
		machine.fit(data,isnot)
		#score the data 
		#change predictpoba(wdataXX) to correct data
		checkagainst[0] = mean(machine.predict_proba(wdata19)[:,1]) #true positive
		falsenegative = 1-checkagainst[0]
		#make sure correct wdataXX isn't in the results and that the other 19 are
		results[0] = mean(machine.predict_proba(wdata0)[:,1])
		results[1] = mean(machine.predict_proba(wdata1)[:,1])
		results[2] = mean(machine.predict_proba(wdata2)[:,1])
		results[3] = mean(machine.predict_proba(wdata3)[:,1])
		results[4] = mean(machine.predict_proba(wdata4)[:,1])
		results[5] = mean(machine.predict_proba(wdata5)[:,1])
		results[6] = mean(machine.predict_proba(wdata6)[:,1])
		results[7] = mean(machine.predict_proba(wdata7)[:,1])
		results[8] = mean(machine.predict_proba(wdata8)[:,1])
		results[9] = mean(machine.predict_proba(wdata9)[:,1])
		results[10] = mean(machine.predict_proba(wdata10)[:,1])
		results[11] = mean(machine.predict_proba(wdata11)[:,1])
		results[12] = mean(machine.predict_proba(wdata12)[:,1])
		results[13] = mean(machine.predict_proba(wdata13)[:,1])
		results[14] = mean(machine.predict_proba(wdata14)[:,1])
		results[15] = mean(machine.predict_proba(wdata15)[:,1])
		results[16] = mean(machine.predict_proba(wdata16)[:,1])
		results[17] = mean(machine.predict_proba(wdata17)[:,1])
		results[18] = mean(machine.predict_proba(wdata18)[:,1])
		for numbers in results:
			falsepositive = falsepositive+numbers
			truenegative = truenegative+(1-numbers)
		#ACC = (TP+TN)/(TP+TN+FP+FN)
		accuracy = ((truepositive+truenegative)/(truepositive+truenegative+falsepositive+falsenegative))
		print (str(accuracy))
		checkagainst = [0]
		falsepositive = 0;
		falsenegative = 0;
		truepositive = 0;
		truenegative = 0;


end = time.time()
print (str(round((end - start),2)) + " seconds to complete")
