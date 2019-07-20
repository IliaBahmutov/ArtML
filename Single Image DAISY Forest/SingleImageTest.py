import numpy #for numpy storage
import os #to find files
import time #for time to complete
import sklearn
import pickle
from PIL import Image
from skimage.feature import daisy
from skimage import data 
from skimage import color
from skimage import io 
while (True):
	filetotest = input("Enter filename: ")

	if (filetotest == "e" or filetotest == "exit"):
		break
	"""
	Crop Start
	"""
	im = Image.open(filetotest)
	w, h = im.size #get resolution
	cropim = im.crop((w//2 - 128, h//2 - 128, w//2 + 128, h//2 + 128)) #256/2 = 128
	cropim.save("Testing","png")
	"""
	Crop End
	"""

	"""
	Extraction Start
	"""
	data = numpy.zeros(shape=(1,3,3,78))
	filetotest = ("Testing")
	img = color.rgb2gray(io.imread(filetotest))
	descs, descs_img = daisy(img, step=95, radius=30, rings=2, histograms=6, orientations=6, visualize=True)
	data = descs
	numpy.save("testingfeaturedata",data) 
	"""
	Extraction End
	"""


	"""
	Testing Start
	"""
	results = numpy.zeros(shape = (20,1))

	testdata = numpy.load("testingfeaturedata.npy")
	testdata = testdata.reshape(1,(3*3*78))

	#Load All DT's
	UKIDT = pickle.load(open("Machines/1UKIDaisyProb.DF", "rb"))
	LReDT = pickle.load(open("Machines/2LReDaisyProb.DF", "rb"))
	MinDT = pickle.load(open("Machines/3MinDaisyProb.DF", "rb"))
	HRenDT = pickle.load(open("Machines/4HRenDaisyProb.DF", "rb"))
	ERenDT = pickle.load(open("Machines/5ERenDaisyProb.DF", "rb"))
	PopDT = pickle.load(open("Machines/6PopDaisyProb.DF", "rb"))
	CFPDT = pickle.load(open("Machines/7CFPDaisyProb.DF", "rb"))
	RocDT = pickle.load(open("Machines/8RocDaisyProb.DF", "rb"))
	CubDT = pickle.load(open("Machines/9CubDaisyProb.DF", "rb"))
	NAPDT = pickle.load(open("Machines/10NAPDaisyProb.DF", "rb"))
	NRDT = pickle.load(open("Machines/11NRDaisyProb.DF", "rb"))
	AEDT = pickle.load(open("Machines/12AEDaisyProb.DF", "rb"))
	BDT = pickle.load(open("Machines/13BDaisyProb.DF", "rb"))
	ANMDT = pickle.load(open("Machines/14ANMDaisyProb.DF", "rb"))
	SymDT = pickle.load(open("Machines/15SymDaisyProb.DF", "rb"))
	PIDT = pickle.load(open("Machines/16PIDaisyProb.DF", "rb"))
	EDT = pickle.load(open("Machines/17EDaisyProb.DF", "rb"))
	RomDT = pickle.load(open("Machines/18RomDaisyProb.DF", "rb"))
	RelDT = pickle.load(open("Machines/19RelDaisyProb.DF", "rb"))
	ImpDT = pickle.load(open("Machines/20ImpDaisyProb.DF", "rb"))

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
	labels = ["Ukiyo e","Mannerism Late Renaissance","Minimalism","High Renaissance","Early Renaissance","Pop Art","Color Field Painting","Rococo","Cubism","Naive Art Primitivism","Northen Renaissance","Abstract Expressionism","Baroque","Art Nouveau Modren","Symbolism","Post Impressionism","Expressionism","Romanticism","Realism","Impressionism"]


	resultsmax = results.argmax() #find most correct classifier position
	highest = (-results).argsort(axis=0)[:20]
	highest.ravel()
	count = 1

	for index in highest: 
		anumber = int(index)
		print (str(count) + ":" + str(labels[anumber]) + ": " + str(results[anumber]))
		count +=1
	deletequestion = input("Clean up?: ")
	if (deletequestion == "y"):
		os.remove("Testing")
		os.remove("testingfeaturedata.npy")
	if (deletequestion == "yq"):
		break
"""
Testing End
"""
 
	