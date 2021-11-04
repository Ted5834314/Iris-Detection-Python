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
from IrisMatching import modtrain
from IrisMatching import modtrain_raw
from IrisMatching import eu
from IrisMatching import cos
from IrisMatching import mh

#calculate CRR for 3 distance measures
def CRRcalc(Xtest,ytest,lda,model1,model2,model3):
  features_new=lda.transform(Xtest)
  pred1=model1.predict(features_new)
  pred2=model2.predict(features_new)
  pred3=model3.predict(features_new)
  CRR1= np.sum(ytest==pred1)/np.size(ytest)
  CRR2= np.sum(ytest==pred2)/np.size(ytest)
  CRR3= np.sum(ytest==pred3)/np.size(ytest)
  return [CRR1,CRR2,CRR3]

#calculate CRR for original feature
def CRRcalc_raw(Xtest,ytest,model4,model5,model6):
  pred4=model4.predict(Xtest)
  pred5=model5.predict(Xtest)
  pred6=model6.predict(Xtest)
  CRR4= np.sum(ytest==pred4)/np.size(ytest)
  CRR5= np.sum(ytest==pred5)/np.size(ytest)
  CRR6= np.sum(ytest==pred6)/np.size(ytest)
  return [CRR4,CRR5,CRR6]


#calculate mindistance of 21 observations in the same class
def calcmindist(x,cat,dist,gsize,totalcat,X):
  cent=np.array([0.0]*1536)
  for i in range(gsize):
    pos=cat*gsize+i
    cent+=X[pos]
  cent=cent/gsize
  return dist(x,cent)

#calculate FNMR by fixing FMR, choosing corresponding threshold by percentile
def FNMRcalc(Xtest,ytest,X,dist):
  np.random.seed(32)
  totalcat=108
  gsize=4
  distlst=[]
  for j in range(20):
    ytest2=sklearn.utils.shuffle(ytest)
  # to calculate FMR, we need non matching lables so random shuffle labels and match to the test features
  # loop 20 times to increase the size of this testing sample and get a more stable rate
    for i in range(totalcat*gsize):
      if ytest2[i]!=ytest[i]:
       
        distlst.append(calcmindist(Xtest[i],ytest2[i]-1,dist,21,108,X))
  # the ith percentile of the distance array is just the threshold i% for FMR rate
  # specifically, distance<threshold would be FM case
  # set 7 thresholds
  thresh_1p=np.percentile(distlst, 1)
  thresh_5p=np.percentile(distlst, 5)
  thresh_10p=np.percentile(distlst, 10)
  thresh_15p=np.percentile(distlst, 15)
  thresh_20p=np.percentile(distlst, 20)
  thresh_25p=np.percentile(distlst, 25)
  thresh_30p=np.percentile(distlst, 30)
  threshlst=[thresh_1p,thresh_5p,thresh_10p,thresh_15p,thresh_20p,thresh_25p,thresh_30p]
  FNMRrate=[]
  # calculate FNMR with the original labels since we want false non-matching cases
  for thresh in threshlst:
    falsepred=0
    for i in range(totalcat*gsize):
      if calcmindist(Xtest[i],ytest[i]-1,dist,21,108,X)>thresh:
        falsepred+=1
    FNMRrate.append(falsepred/totalcat/gsize)
  return threshlst,FNMRrate


#output table and graph for CRR and FNMR
def CRRoutput(X,y,Xtest,ytest):
  models=modtrain(X,y,100)
  result = CRRcalc(Xtest,ytest,models[0],models[1],models[2],models[3])

  model_raw = modtrain_raw(X,y)
  result_raw = CRRcalc_raw(Xtest,ytest,model_raw[0],model_raw[1],model_raw[2])


  table = [['similarity measure','correct recognition rate of original feature set','correct recognition rate of reduced feature set(dimension=100)'],
             ['L1 distance measure',result_raw[1],result[1]],
             ['L2 distance measure',result_raw[0],result[0]],
             ['cosine similarity measure',result_raw[2],result[2]]]
  print(tabulate(table,headers = 'firstrow',tablefmt='fancy_grid'))
  print("unlike the paper, LDA boost the CRR significantly in our cases")
  print("generate CRR over feature dimension for cos similarity, which has best performance in terms of CRR")

  ls_x = [20,40,60,80,100,105,107]
  ls_y = []
  for i in ls_x:
    models=modtrain(X,y,i)
    ls_y.append(CRRcalc(Xtest,ytest,models[0],models[1],models[2],models[3])[2])
  line = plt.plot(ls_x,ls_y)
  plt.plot(ls_x,ls_y,'ro-') 
  plt.title('recognition results using features of different dimensionality')
  plt.xlabel('dimensionality of feature vector')
  plt.ylabel('correct recognition rate')
  plt.show()


def FNMRoutput(X,y,Xtest,ytest):
  FMR_cos = FNMRcalc(Xtest,ytest,X,cos)
  FMR_eu = FNMRcalc(Xtest,ytest,X,eu)
  FMR_mh = FNMRcalc(Xtest,ytest,X,mh)

  print("Now Generate tables and ROC curves for the 3 metrics")
  print("\n")
  table_1 = [['Threshold for cos','False match rate ','False non-match rate'],[FMR_cos[0][2],0.10,FMR_cos[1][2]],
           [FMR_cos[0][3],0.15,FMR_cos[1][3]],[FMR_cos[0][4],0.20,FMR_cos[1][4]]]
  print(tabulate(table_1,headers = 'firstrow',tablefmt='fancy_grid'))

  table_2 = [['Threshold for eu ','False match rate ','False non-match rate'],
             [FMR_eu[0][2],0.10,FMR_eu[1][2]],
             [FMR_eu[0][3],0.15,FMR_eu[1][3]],
             [FMR_eu[0][4],0.20,FMR_eu[1][4]]]
  print(tabulate(table_2,headers = 'firstrow',tablefmt='fancy_grid'))

  table_3 = [['Threshold for mh','False match rate ','False non-match rate'],
             [FMR_mh[0][2],0.10,FMR_mh[1][2]],
             [FMR_mh[0][3],0.15,FMR_mh[1][3]],
             [FMR_mh[0][4],0.20,FMR_mh[1][4]]]
  print(tabulate(table_3,headers = 'firstrow',tablefmt='fancy_grid'))
  FMR=[0.01,0.05,0.1,0.15,0.2,0.25,0.3]

  plt.plot(FMR,FMR_cos[1],'ro-')
  plt.title(' FNMR vs FMR for cos')
  plt.xlabel('False match rate')
  plt.ylabel('False non-match rate')
  plt.show()


  plt.plot(FMR,FMR_eu[1],'ro-')
  plt.title(' FNMR vs FMR for eu')
  plt.xlabel('False match rate')
  plt.ylabel('False non-match rate')
  plt.show()



  plt.plot(FMR,FMR_mh[1],'ro-')
  plt.title(' FNMR vs FMR for mh')
  plt.xlabel('False match rate')
  plt.ylabel('False non-match rate')
  plt.show()
  print("\n")
  print("From the reult above, mh is the best of the 3 in terms of FNMR")
