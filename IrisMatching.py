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
#modtrain outputs a vector [LDA,mod1,mod2,mod3]
#LDA is the feature reduction strategy from training set
#NC is the nearest center classifier from training set NC=[NC1,NC2,NC3] corresponding to different distance metrics
#As I posted in ed, the original method won't work out in our case, so I simply view the 3x7=21 images as same class and use ordinary distance functions to do machine learning
def cos(x,y):
  return 1-np.dot(x,y)/ np.linalg.norm(x)/np.linalg.norm(y)
def mh(x,y):
  return np.linalg.norm(x-y,ord=1)
def eu(x,y):
  return np.linalg.norm(x-y,ord=2)


# In the paper, the author took the "min of the seven scores". which may be
# well defined where each class only occur once in the training set
# However when there are more than one observation for each class in the training set,
# this distance measure becomes dubious. For example we apply the nearest center classifier
# and the center of the 3 observations would be their mean. But this mean,
# as a feature vector, has lost all information as an image, so we are not
# able to acquire the shifted features of this "mean vector"
# The only reasonable thing to do here, is to set 
def modtrain(X,y,n):
  LDA_transform = LDA(n_components=n)
  LDA_transform.fit(X, y)
  features_new = LDA_transform.transform(X)
  model1 = NC(metric='euclidean')
  model2 = NC(metric='manhattan')
  model3 = NC(metric=cos)
  model1.fit(features_new,y)
  model2.fit(features_new,y)
  model3.fit(features_new,y)
  return [LDA_transform,model1,model2,model3]

def modtrain_raw(X,y):
  model4 = NC(metric='euclidean')
  model5 = NC(metric='manhattan')
  model6 = NC(metric=cos)
  model4.fit(X,y)
  model5.fit(X,y)
  model6.fit(X,y)
  return [model4,model5,model6]
