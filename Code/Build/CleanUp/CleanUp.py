from PIL import Image
import os 
import time
start = time.time()

for filename in os.listdir('./'): #Iterate over all files in folder
	if filename.endswith(".jpg"): #Has to be a target file
	  	os.remove(filename)

end = time.time()
print str(round((end - start),2)) + " seconds to complete"