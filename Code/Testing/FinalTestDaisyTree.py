import numpy #for numpy storage
import os #to find files
import time #for time to complete
import sklearn
import pickle

#Load Test Data

running = True
selection = 1

while running:

	if (selection == 21):
		print ("done")
		exit()


	elif (selection == 1):
		testdata = numpy.load("Data/1UkiTestingV.npy")
		testdata = testdata.reshape(175,(3*3*78))
		correctnum = 0 
		results = numpy.zeros(shape = (20,175))

	elif (selection == 2):
		testdata = numpy.load("Data/2LReTestingV.npy")
		testdata = testdata.reshape(192,(3*3*78))
		correctnum = 1 
		results = numpy.zeros(shape = (20,192))

	elif (selection == 3):
		testdata = numpy.load("Data/3MinTestingV.npy")
		testdata = testdata.reshape(201,(3*3*78))
		correctnum = 2
		results = numpy.zeros(shape = (20,201))

	elif (selection == 4):
		testdata = numpy.load("Data/4HreTestingV.npy")
		testdata = testdata.reshape(202,(3*3*78))
		correctnum = 3 
		results = numpy.zeros(shape = (20,202))

	elif (selection == 5):
		testdata = numpy.load("Data/5EreTestingV.npy")
		testdata = testdata.reshape(209,(3*3*78))
		correctnum = 4
		results = numpy.zeros(shape = (20,209))

	elif (selection == 6):
		testdata = numpy.load("Data/6PopTestingV.npy")
		testdata = testdata.reshape(223,(3*3*78))
		correctnum = 5
		results = numpy.zeros(shape = (20,223))

	elif (selection == 7):
		testdata = numpy.load("Data/7CFPTestingV.npy")
		testdata = testdata.reshape(242,(3*3*78))
		correctnum = 6
		results = numpy.zeros(shape = (20,242))

	elif (selection == 8):
		testdata = numpy.load("Data/8ROCTestingV.npy")
		testdata = testdata.reshape(314,(3*3*78))
		correctnum = 7 
		results = numpy.zeros(shape = (20,314))

	elif (selection == 9):
		testdata = numpy.load("Data/9CubTestingV.npy")
		testdata = testdata.reshape(330,(3*3*78))
		correctnum = 8
		results = numpy.zeros(shape = (20,330))

	elif (selection == 10):
		testdata = numpy.load("Data/10NAPTestingV.npy")
		testdata = testdata.reshape(361,(3*3*78))
		correctnum = 9
		results = numpy.zeros(shape = (20,361))

	elif (selection == 11):
		testdata = numpy.load("Data/11NreTestingV.npy")
		testdata = testdata.reshape(383,(3*3*78))
		correctnum = 10
		results = numpy.zeros(shape = (20,383))

	elif (selection == 12):
		testdata = numpy.load("Data/12AExTestingV.npy")
		testdata = testdata.reshape(418,(3*3*78))
		correctnum = 11
		results = numpy.zeros(shape = (20,418))

	elif (selection == 13):
		testdata = numpy.load("Data/13BarTestingV.npy")
		testdata = testdata.reshape(636,(3*3*78))
		correctnum = 12
		results = numpy.zeros(shape = (20,636))

	elif (selection == 14):
		testdata = numpy.load("Data/14AMNTestingV.npy")
		testdata = testdata.reshape(650,(3*3*78))
		correctnum = 13
		results = numpy.zeros(shape = (20,650))

	elif (selection == 15):
		testdata = numpy.load("Data/15SymTestingV.npy")
		testdata = testdata.reshape(679,(3*3*78))
		correctnum = 14
		results = numpy.zeros(shape = (20,679))

	elif (selection == 16):
		testdata = numpy.load("Data/16PImTestingV.npy")
		testdata = testdata.reshape(967,(3*3*78))
		correctnum = 15
		results = numpy.zeros(shape = (20,967))

	elif (selection == 17):
		testdata = numpy.load("Data/17ExpTestingV.npy")
		testdata = testdata.reshape(1011,(3*3*78))
		correctnum = 16
		results = numpy.zeros(shape = (20,1011))

	elif (selection == 18):
		testdata = numpy.load("Data/18RomTestingV.npy")
		testdata = testdata.reshape(1053,(3*3*78))
		correctnum = 17
		results = numpy.zeros(shape = (20,1053))

	elif (selection == 19):
		testdata = numpy.load("Data/19RelTestingV.npy")
		testdata = testdata.reshape(1610,(3*3*78))
		correctnum = 18
		results = numpy.zeros(shape = (20,1610))

	elif (selection == 20):
		testdata = numpy.load("Data/20ImpTestingV.npy")
		testdata = testdata.reshape(1959,(3*3*78))
		correctnum = 19
		results = numpy.zeros(shape = (20,1959))


	#Load All DT's
	UKIDT = pickle.load(open("1UKIDaisyProb.DTGS", "rb"))
	LReDT = pickle.load(open("2LReDaisyProb.DTGS", "rb"))
	MinDT = pickle.load(open("3MinDaisyProb.DTGS", "rb"))
	HRenDT = pickle.load(open("4HRenDaisyProb.DTGS", "rb"))
	ERenDT = pickle.load(open("5ERenDaisyProb.DTGS", "rb"))
	PopDT = pickle.load(open("6PopDaisyProb.DTGS", "rb"))
	CFPDT = pickle.load(open("7CFPDaisyProb.DTGS", "rb"))
	RocDT = pickle.load(open("8RocDaisyProb.DTGS", "rb"))
	CubDT = pickle.load(open("9CubDaisyProb.DTGS", "rb"))
	NAPDT = pickle.load(open("10NAPDaisyProb.DTGS", "rb"))
	NRDT = pickle.load(open("11NRDaisyProb.DTGS", "rb"))
	AEDT = pickle.load(open("12AEDaisyProb.DTGS", "rb"))
	BDT = pickle.load(open("13BDaisyProb.DTGS", "rb"))
	ANMDT = pickle.load(open("14ANMDaisyProb.DTGS", "rb"))
	SymDT = pickle.load(open("15SymDaisyProb.DTGS", "rb"))
	PIDT = pickle.load(open("16PIDaisyProb.DTGS", "rb"))
	EDT = pickle.load(open("17EDaisyProb.DTGS", "rb"))
	RomDT = pickle.load(open("18RomDaisyProb.DTGS", "rb"))
	RelDT = pickle.load(open("19RelDaisyProb.DTGS", "rb"))
	ImpDT = pickle.load(open("20ImpDaisyProb.DTGS", "rb"))

	results[0] = UKIDT.predict_proba(testdata)[:,1]
	results[1] = LReDT.predict_proba(testdata)[:,1]
	results[2] = MinDT.predict_proba(testdata)[:,1]
	results[3] = HRenDT.predict_proba(testdata)[:,1]
	results[4] = ERenDT.predict_proba(testdata)[:,1]
	results[5] = PopDT.predict_proba(testdata)[:,1]
	results[6] = CFPDT.predict_proba(testdata)[:,1]
	results[7] = RocDT.predict_proba(testdata)[:,1]
	results[8] = CubDT.predict_proba(testdata)[:,1]
	results[9] = NAPDT.predict_proba(testdata)[:,1]
	results[10] = NRDT.predict_proba(testdata)[:,1]
	results[11] = AEDT.predict_proba(testdata)[:,1]
	results[12] = BDT.predict_proba(testdata)[:,1]
	results[13] = ANMDT.predict_proba(testdata)[:,1]
	results[14] = SymDT.predict_proba(testdata)[:,1]
	results[15] = PIDT.predict_proba(testdata)[:,1]
	results[16] = EDT.predict_proba(testdata)[:,1]
	results[17] = RomDT.predict_proba(testdata)[:,1]
	results[18] = RelDT.predict_proba(testdata)[:,1]
	results[19] = ImpDT.predict_proba(testdata)[:,1]

	labels = ["UKI","LRe","Min","HRen","ERen","Pop","CFP","Roc","Cub","NAP","NR","AE","B","ANM","Sym","PI","ES","RoM","Rel","Imp"]

	resultsmax = results.argmax(axis=0)
	correct = list(resultsmax).count(correctnum)
	print(correct)
	selection = selection +1;


 
	