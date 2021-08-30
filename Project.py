# -*- coding: utf-8 -*-

# -- Sheet --

"""
CNG 483 PROJECT 1: CONTENT BASED IMAGE CLASSIFICATION
Sedat Ali ZEVİT 		-	2152221
Mehmet Dağhan Namlıoğlu -   2152106
Group no: 10
"""



#This project is created in Datalore Jupyter notebook 
import shutil
import os
from setuptools import glob
from PIL import Image  
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def binarraygen(binnum,rangeof = 255): #Create an bin array for histograms if user want to plot graph manually
    binarray=[]
    for i in range(0,binnum+1):
        binarray.append((rangeof/binnum)*i)
    return binarray 
    
def plotcolorhistnormal(arr,bin_value=64):#Function for plotting RGB color histogram on top of each other.
    red_plot=arr[:,:,0]
    green_plot=arr[:,:,1]
    blue_plot=arr[:,:,2]
    plt.hist(red_plot.ravel(),bins=bin_value,color="r", alpha = 0.5)
    plt.hist(green_plot.ravel(),bins=bin_value,color="g", alpha = 0.5)
    plt.hist(blue_plot.ravel(),bins=bin_value,color="b", alpha = 0.5)
    plt.hist(arr.ravel(),bins=bin_value,color="black", alpha = 0.4)
    
def colorhistnormal(arr,bin_value=64):  # function for creating color histogram 
    return np.histogram(arr.ravel(), bin_value)

def greyhist(arr,bin_value=64):# function for creating greyscale  histogram 
    grayscale = rgb2gray(arr)
    return np.histogram(grayscale.ravel(), bin_value)

"""
This approach will create a combined color rgb histogram as requested in assignment. In our tests with cat photo, our function
provides similar results as in wikipedia page given in assignment. We manipulated intensity of RGB and use them as indexes.
First, intensity of each pixel is multiplied by bin_value/256. This new value will indicate which bin that color belongs to.
After calculating RGB bins, we use this values as index for a 3D array and increase value of this array element by one.
This process must be repeated for every pixel. Because of this, although this process is still fast enough for couple of
photographs, it can take hours to complete big datasets. We didnt have enough time to verify accuracy of this approach
in given dataset for our report.
"""

def colorhistcomb(arr,bin_value=16):
    red=arr[:,:,0].ravel() # Channels are divided into 3 variables for combination.
    green=arr[:,:,1].ravel()
    blue=arr[:,:,2].ravel()
    hisarr = np.zeros((bin_value,bin_value,bin_value))
    width,height,channels=arr.shape
    for val in range(int(height*width)):
        hisarr[int(red[val]*bin_value/256)][int(green[val]*bin_value/256)][int(blue[val]*bin_value/256)] = int(hisarr[int(red[val]*bin_value/256)][int(green[val]*bin_value/256)][int(blue[val]*bin_value/256)] +1) 
    return hisarr.flatten()

def imagecropper(img, lvl):# function for cropping images acording to levels
    width, height = img.size
    lvl = lvl-1 #Level of cropping needed
    croppedarray = [] #This array will hold cropped images
    for y in range(0,2**lvl):
        for x in range(0,2**lvl):
            croppedarray.append(img.crop(((x*(width/(2**lvl))),(y*(height/(2**lvl))),((x+1)*(width/(2**lvl))),((y+1)*(height/(2**lvl))))))  
    return croppedarray



def makeimagearr(directorypath): # function for creating an image array list
    imagelist = []
    for pathImg in glob.glob(directorypath + "/*"):
        imgload = Image.open(pathImg)
        imagelist.append(imgload)
        imgload.show()
    return imagelist

def makeimagearrarray(directorypath):# function for returning image array's array
    imagelist = []
    for pathImg in glob.glob(directorypath + "/*"):
        imgload = Image.open(pathImg)
        imagelist.append(np.array(imgload))

    return imagelist

def greyhistlist(directorypath,bin_val):# function for creating greyscale histogram list from directory path 
    list = makeimagearrarray(directorypath)
    histlist = []
    for i in range(0,len(list)):
        histlist.append(greyhist(list[i],bin_val)[0])
    return histlist

def normalhistogramlist(directorypath,bin_val):# function for creating color histogram list from directory path 
    list = makeimagearrarray(directorypath)
    histlist = []
    for i in range(0,len(list)):
        histlist.append(colorhistnormal(list[i],bin_val)[0])
    return histlist

def combhistogramlist(directorypath,bin_val):# function for creating combined color histogram list from directory path 
    list = makeimagearrarray(directorypath)
    histlist = []
    for i in range(0,len(list)):
        histlist.append(colorhistcomb(list[i],bin_val))
    return histlist




def imagecropperarr(img, lvl):# function for cropping images acording to levels and returns array
    width, height = img.size
    lvl = lvl-1 #Level of cropping needed
    croppedarray = [] #This array will hold cropped images
    for y in range(0,2**lvl):
        for x in range(0,2**lvl):
            croppedarray.append(np.array(img.crop(((x*(width/(2**lvl))),(y*(height/(2**lvl))),((x+1)*(width/(2**lvl))),((y+1)*(height/(2**lvl)))))))  
    return croppedarray


def makeimagearrarraylevel(directorypath,level):# This function reads images from a directory and crops them according to level
    imagelist = []                              # Then returns array of all images ready for process
    for pathImg in glob.glob(directorypath + "/*"):
        imgload = Image.open(pathImg)
        imagelist.append(imagecropperarr(imgload,level))
    return imagelist


def greyhistlistlevel(directorypath,bin_val,level):#This function creates greyscale histograms of cropped images from directory.
    imlist = makeimagearrarraylevel(directorypath,level)
    histlist = []
    croppedphotohistarray = []
    for i in range(0,len(imlist)):
        for k in range(0,len(imlist[0])):
            croppedphotohistarray.append(greyhist(imlist[i][k],bin_val)[0])
        histlist.append(croppedphotohistarray)
        croppedphotohistarray = []
    return histlist
#leveltestgrey = greyhistlistlevel("dataset",4,2)

def normalhistogramlistlevel(directorypath,bin_val,level):#This function creates color histograms of cropped images from directory.
    imlist = makeimagearrarraylevel(directorypath,level)
    histlist = []
    croppedphotohistarray = []
    for i in range(0,len(imlist)):
        for k in range(0,len(imlist[0])):
            croppedphotohistarray.append(colorhistnormal(imlist[i][k],bin_val)[0])
        histlist.append(croppedphotohistarray)
        croppedphotohistarray = []
    return histlist
#leveltestnormal = normalhistogramlistlevel("dataset",4,2)


def combhistogramlistlevel(directorypath,bin_val,level):#This function creates combined color histograms of cropped images from directory.
    imlist = makeimagearrarraylevel(directorypath,level)
    histlist = []
    croppedphotohistarray = []
    for i in range(0,len(imlist)):
        for k in range(0,len(imlist[0])):
            croppedphotohistarray.append(colorhistcomb(imlist[i][k],bin_val))
        histlist.append(croppedphotohistarray)
        croppedphotohistarray = []
    return histlist
#leveltestcomb = combhistogramlistlevel("dataset",4,2)


"""
Preprocessor prepares dataset for KNN clasifier according to datapath, binsize, level and algoritm selection.This function 
creates target array and a bunch type variable. Then target array and data array are placed into this variable.
"""
def datasetpreprocessor(directorypath = "dataset",binsize = 4,level = 2,type = 1): 
    target = 0
    targetarr=[]
    for pathImg in glob.glob(directorypath + "/*"):
        
        if "cloudy" in pathImg:
            target=0
        elif "shine" in pathImg:
            target=1
        elif "sunrise" in pathImg:
            target=2 
        else:
            target=-1
        targetarr.append(target)
    ds = Bunch()
    ds.target = targetarr
    if type == 1:
        ds.data = arrayflatter(greyhistlistlevel(directorypath,binsize,level))
    elif type == 2:
        ds.data = arrayflatter(combhistogramlistlevel(directorypath,binsize,level))
    else:
        ds.data = arrayflatter(normalhistogramlistlevel(directorypath,binsize,level))
    ds.target_names = ["cloudy","shine","sunrise"]
    return ds

def arrayflatter(dataset): #This function flattens cropped images into one array.
    length = len(dataset[0])
    inlength = len(dataset[0][0])
    temparray = []
    flatarray = []
    for i in range(0,len(dataset)):
        for k in range(0,length):
            temparray.extend(dataset[i][k])
        flatarray.append(temparray)
        temparray = []
    return flatarray


bin = 8
level = 3
method = 3
#Method 1 = greyscale, method 2 = composite color, method 3 = color histogram

ds = datasetpreprocessor("testntrain", bin,level,method)
xtrain,xtest,ytrain,ytest = train_test_split(ds.data,ds.target,test_size = 0.25, random_state = 4) # This function call splits data into 2 sets.
scorewithbin = []
score = {}
score_list = []
for k in [9]:#Can add different neighborhood numbers to train
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(xtrain,ytrain)
    predictions = knn.predict(xtest)
    score[k]=metrics.accuracy_score(ytest,predictions)
    score_list.append(metrics.accuracy_score(ytest,predictions))
    
#plt.plot(list(range(1,31)),score_list)
#print(score)

def query(path):#Query is configured according to our best results. Bin = 8, level = 3
    shutil.copyfile(path, "queryfile/"+path.split('/')[1])
    ans = (knn.predict(arrayflatter(normalhistogramlistlevel("queryfile",8,3))))
    for i in range(0,len(ans)):
        if ans[i] == 0:
            print("cloudy")
        elif ans[i] == 1:
            print("shine")
        else:
            print("sunrise")
    os.remove("queryfile/"+path.split('/')[1])


query("testntrain/cloudy126.jpg")





