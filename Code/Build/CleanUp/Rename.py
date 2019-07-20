import os 

path = './' #change path to folder path
i = 0
for filename in sorted(os.listdir(path)) :
    os.rename(os.path.join(path,filename), os.path.join(path,'TEST'+str(i))) #change test to artwork name
    i = i +1