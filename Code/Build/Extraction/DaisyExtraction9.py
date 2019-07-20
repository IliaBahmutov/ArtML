"""
Daisy Extractor Script
This script extracts the daisy data from all files within the directory so that a 256x256 image
has 9 features extracted with 2 rings and 6 orientations for each feature. 

Script has hardcoded vales for array sizes and filename differentiation so they need to be changed
on each run. Script will fail at the end if there are more than the specified amount of files within
the directory ignoring the script file itself
"""

from skimage.feature import daisy #used to extract image features 
from skimage import data #used to save files
from skimage import color #used to change the images to grayscale
from skimage import io #used to save files
import numpy #for numpy storage
import os #to find files
import time #for time to complete
start = time.time()

sizeoflearningsample = 3894;
nameofthesample = "11NR";

errorlist = [] #used for error checking
count = 1; #used for progress 
filenumb = 0; 
filecount = len(os.listdir('./')) #used for progress bar
thedata = numpy.zeros(shape=(sizeoflearningsample,3,3,78)) #create array to hold all 3d image arrays, fill with 0's
isnot = numpy.zeros(shape=(sizeoflearningsample)) #create array to hold label data of image data


for filename in sorted(os.listdir('./')):
	try:
		if not (filename.endswith("py")): #all non script files
			img = color.rgb2gray(io.imread(filename)) #convert image into grayscale
			descs, descs_img = daisy(img, step=95, radius=30, rings=2, histograms=6, orientations=6, visualize=True)
			thedata[filenumb] = descs #Add daisy data to 4d numpy array
			if (filename.startswith(nameofthesample)):
				isnot[filenumb] = 1
				filenumb += 1
			else:
				isnot[filenumb] = 0
				filenumb += 1
			count += 1
			perc = round(100*(float(count)/float(filecount)),2) #percentage of files looped to two decimal places
			print (str(perc) + "%") # pring percentage
		else: 
			errorlist.append(filename)
	except:
		errorlist.append(filename)

numpy.save(nameofthesample+"TrainingData",thedata)		#Save FeatureData
numpy.save(nameofthesample+"TrainingDataLabels", isnot)	#Save FeatureDataLabel

end = time.time()
print (str(round((end - start),2)) + " seconds to complete")

if errorlist:
	try:
		for x in errorlist: print ("File: " + x + ": Failed")
	except:
		print ("Error List Error")
else: 
	print ("No Errors During Feature Extraction")