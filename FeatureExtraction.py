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
# This implementation follows the formula given in the paper
# and follows similar logic in wikipedia's python implementation
# of gabor filter. See https://en.wikipedia.org/wiki/Gabor_filter
# That is, the origin of a gabor filter matrix is set to be its center
# The kernel size is set to (9,9) as suggested
def ROI(image):
    return image[0:48,:]


def newfilter(x, y, sigma_x, sigma_y,f):
    return (1/(2*(math.pi)*sigma_x*sigma_y))*math.exp(-0.5*(x**2/sigma_x**2 + y**2/sigma_y**2)) * M(x,y,f)
    
    
    
def M(x,y,f):
    # f is the frequency of sinusoidal function
    return math.cos(2*math.pi*f*math.sqrt(x**2 + y**2))


# compute kernel apply convolution
def getKernel(sigma_x, sigma_y,f):
    kernel = np.zeros((9,9))
    for i in range(0,9):
        for j in range(0,9):
            kernel[i,j] = newfilter(i-4, j-4, sigma_x, sigma_y,f)
    return kernel

            
            
def getConvolution(img, sigma_x, sigma_y,f):
    roi = ROI(img)
    kernel = getKernel(sigma_x, sigma_y,f)
    return scipy.signal.convolve2d(roi, kernel, mode='same', boundary='wrap')


# extract statistics from 8x8 block
def get_feature_vector(image, sigma_x, sigma_y,f):
    img = getConvolution(image, sigma_x, sigma_y,f)
    len_row = len(img)
    len_col = len(img[0])
    rows = len_row//8
    cols = len_col//8
    feature_vector=[]
    for r in range(0,rows):
        for c in range(0,cols):
            mean = np.mean(np.abs(img[8*r:8*(r+1), 8*c:8*(c+1)]))
            sd = np.mean(np.abs((np.abs(img[8*r:8*(r+1), 8*c:8*(c+1)]) - mean)))
            feature_vector.append(mean)
            feature_vector.append(sd)
    return feature_vector

# extract featue vectors with two
# set of sigmax and sigmay as described in the
# paper and glue them to a length 1536 vector

def feature_extractor(train,test):
  trainf=[]
  testf=[]
  for i in range(len(train)):
    # f is set to 1/sigmax
    fvec1=get_feature_vector(train[i], 4.5, 1.5,1/4.5)
    fvec2=get_feature_vector(train[i], 3, 1.5,1/3)
    trainf.append(fvec1+fvec2)
  for j in range(len(test)):
    fvec1=get_feature_vector(test[j], 4.5, 1.5,1/4.5)
    fvec2=get_feature_vector(test[j], 3, 1.5,1/3)
    testf.append(fvec1+fvec2)
  return [np.array(trainf),np.array(testf)]
