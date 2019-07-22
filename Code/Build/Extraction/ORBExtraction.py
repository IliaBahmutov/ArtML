from skimage.feature import ORB
from skimage import data
from skimage import color
from skimage import io
import numpy #for numpy storage
import os #to find files
import time #for time to complete

start = time.time()
errorlist = [] #used for error checking
deadend = 0;

sizeoflearningsample = 1476
nameofsample = "UKI"

#Orb can only extract x number of keypoints from a given image
#x can be 0
#need to resize the array after finishing

count = 1 #used for progress 
filenumb = 0
filecount = len(os.listdir('./')) #used for progress bar
thedata = numpy.zeros(shape=(sizeoflearningsample,100,256))
isnot = numpy.zeros(shape=(sizeoflearningsample)) 

for filename in os.listdir('./'):
    try:
        if not(filename.endswith("py")): #all cropped files are png
            img = color.rgb2gray(io.imread(filename))
            descriptor_extractor = ORB(n_keypoints=100)
            descriptor_extractor.detect_and_extract(img)
            thedata[filenumb] = descriptor_extractor.descriptors #add data to array
            if (filename.startswith(nameofsample)):
                isnot[filenumb] = 1
                filenumb += 1
            else:
                isnot[filenumb] = 0
                filenumb += 1 
            count += 1
            perc = round(100*(float(count)/float(filecount)),2)
            print (str(perc) + "%")
        else: 
            errorlist.append(filename)
    except Exception as e: 
        print(e)
        errorlist.append(filename)
        deadend = deadend+1
    

numpy.save(nameofsample+"ORBTrainingData",thedata)		#Save FeatureData
numpy.save(nameofsample+"ORBTrainingDataLabels", isnot)	#Save FeatureDataLabel

end = time.time()
print (str(round((end - start),2)) + " seconds to complete")



if errorlist:
    try:
        for x in errorlist: print (x)
    except:
        print ("Error List Error")
else: 
    print ("No Errors During Feature Extraction")

print("To remove with zCLEANUP :" + str(deadend))