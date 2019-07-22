from PIL import Image
import os 
import time
start = time.time()

count = 1; #used for naming
filecount = len(os.listdir('./')) #used for progress bar
errorlist = []

for filename in os.listdir('./'): #Iterate over all files in folder
    if filename.endswith(".jpg"): #Has to be a target file
        im = Image.open(filename) 
        w, h = im.size #get resolution
        cropim = im.crop((w//2 - 128, h//2 - 128, w//2 + 128, h//2 + 128)) #256/2 = 128
        newname = str(count)+"C"
        cropim.save(newname,"png")
        count += 1
        perc = round(100*(float(count)/float(filecount)),2)
        print str(perc) + "%"
    else:
        errorlist.append(filename)


end = time.time()
print str(round((end - start),2)) + " seconds to complete"
if errorlist:
    try:
        for x in errorlist: print "File: " + x + ": Failed"
    except:
        print "Error List Error" 
else: 
    print "No Errors During Cropping"