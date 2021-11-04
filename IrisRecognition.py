import os 
import cv2 
import matplotlib.pyplot as plt
import math
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestCentroid as NC
from scipy import signal
import scipy
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')
from IrisLocalization import dist1
from IrisLocalization import localizer
from IrisNormalization import normalizer
from ImageEnhancement import enhancer
from FeatureExtraction import feature_extractor
from IrisMatching import modtrain
from PerformanceEvaluation import CRRcalc
from PerformanceEvaluation import FNMRcalc
from PerformanceEvaluation import CRRoutput
from PerformanceEvaluation import FNMRoutput

def pipeline(imgname):
  locinfo=localizer(imgname)
  # 7 angles to be shifted
  angles=[np.pi/20,np.pi/30,np.pi/60,0,-np.pi/20,-np.pi/30,-np.pi/60]
  imgvec=[]
  for shift in angles:
    normalized=normalizer(locinfo[0],locinfo[1],shift)
    enhanced=enhancer(normalized)
    imgvec.append(enhanced)
  return imgvec
def pipelinetest(imgname):
  locinfo=localizer(imgname)
  imgvec=[]
  normalized=normalizer(locinfo[0],locinfo[1])
  enhanced=enhancer(normalized)
  imgvec.append(enhanced)
  return imgvec

train=[]
test=[]
basename='CASIA Iris Image Database (version 1.0)/CASIA Iris Image Database (version 1.0)/'
# get training and testing images
for i in range(1,109):
  for j in range(1,4):
    nz=3-len(str(i))
    newname=basename
    newname+=('0'*nz+str(i))
    newname+='/1/'
    newname+=('0'*nz+str(i)+"_"+"1"+"_"+str(j)+".bmp")
    res=pipeline(newname)
    train+=res
  for k in range(1,5):
    nz=3-len(str(i))
    newname=basename
    newname+=('0'*nz+str(i))
    newname+='/2/'
    newname+=('0'*nz+str(i)+"_"+"2"+"_"+str(k)+".bmp")
    res=pipelinetest(newname)
    test+=res
#extract feature
fvec=feature_extractor(train,test)
X=fvec[0]
Xtest=fvec[1]
y=np.array([i for i in range(1,109) for j in range(21) ])
ytest=np.array([i for i in range(1,109) for j in range(4) ])
#train model and generate CRR, FMR and FNMR results
models=modtrain(X,y,100)
print("The best CRR is reached for n=100, where CRR>80% for all metrics(eu,mh and cos)")
print(CRRcalc(Xtest,ytest,models[0],models[1],models[2],models[3]))
print("\n")
CRRoutput(X,y,Xtest,ytest)
print("\n")
print("\n")
FNMRoutput(X,y,Xtest,ytest)
