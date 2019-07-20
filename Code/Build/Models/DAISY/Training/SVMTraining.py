import numpy #for numpy storage
import os #to find files
import time #for time to complete
from sklearn import svm
import pickle
start = time.time()
from sklearn.calibration import CalibratedClassifierCV
from statistics import mean

#Import Training Data & Labels
odata = numpy.load("Data/20ImpTrainingData.npy")
data = odata.reshape(9826,(3*3*78))
isnot = numpy.load("Data/20ImpTrainingDataLabels.npy")

#Import Test Data & Labels
wdata0 = numpy.load("Data/UkiTestingData.npy")
wdata0 = wdata0.reshape(350,(3*3*78))

wdata1 = numpy.load("Data/LReTestingData.npy")
wdata1 = wdata1.reshape(384,(3*3*78))

wdata2 = numpy.load("Data/MinTestingData.npy")
wdata2 = wdata2.reshape(401,(3*3*78))

wdata3 = numpy.load("Data/HreTestingData.npy")
wdata3 = wdata3.reshape(403,(3*3*78))

wdata4 = numpy.load("Data/EreTestingData.npy")
wdata4 = wdata4.reshape(417,(3*3*78))

wdata5 = numpy.load("Data/PopTestingData.npy")
wdata5 = wdata5.reshape(445,(3*3*78))

wdata6 = numpy.load("Data/CFPTestingData.npy")
wdata6 = wdata6.reshape(484,(3*3*78))

wdata7 = numpy.load("Data/ROCTestingData.npy")
wdata7 = wdata7.reshape(627,(3*3*78))

wdata8 = numpy.load("Data/CubTestingData.npy")
wdata8 = wdata8.reshape(660,(3*3*78))

wdata9 = numpy.load("Data/NAPTestingData.npy")
wdata9 = wdata9.reshape(721,(3*3*78))

wdata10 = numpy.load("Data/NreTestingData.npy")
wdata10 = wdata10.reshape(766,(3*3*78))

wdata11 = numpy.load("Data/AExTestingData.npy")
wdata11 = wdata11.reshape(835,(3*3*78))

wdata12 = numpy.load("Data/BarTestingData.npy")
wdata12 = wdata12.reshape(1272,(3*3*78))

wdata13 = numpy.load("Data/AMNTestingData.npy")
wdata13 = wdata13.reshape(1300,(3*3*78))

wdata14 = numpy.load("Data/SymTestingData.npy")
wdata14 = wdata14.reshape(1358,(3*3*78))

wdata15 = numpy.load("Data/PImTestingData.npy")
wdata15 = wdata15.reshape(1934,(3*3*78))

wdata16 = numpy.load("Data/ExpTestingData.npy")
wdata16 = wdata16.reshape(2021,(3*3*78))

wdata17 = numpy.load("Data/RomTestingData.npy")
wdata17 = wdata17.reshape(2106,(3*3*78))

wdata18 = numpy.load("Data/RelTestingData.npy")
wdata18 = wdata18.reshape(3220,(3*3*78))

wdata19 = numpy.load("Data/ImpTestingData.npy")
wdata19 = wdata19.reshape(3918,(3*3*78))


#cval = 21 length from 2^-5 to 2^15


cval = [0.03125,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,64,128]
print ("Training Test Data")


results = [0] *19
checkagainst = [0]
falsepositive = 0;
falsenegative = 0;
truepositive = 0;
truenegative = 0;


for cavls in cval:
	machine = svm.LinearSVC(C = cavls, random_state = 2,max_iter = 1000000,loss="hinge")
	machine = CalibratedClassifierCV(machine, cv = 3)
	machine.fit(data,isnot)
	#score the data 
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
