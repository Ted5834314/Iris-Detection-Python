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
def enhancer(normalized):
  xlim=64
  ylim=512
  blursize=16
  blurred=np.zeros(xlim*ylim//blursize//blursize, dtype=np.uint8).reshape(xlim//blursize, ylim//blursize)
  # blur by taking mean in 16x16 blocks
  for i in range(len(blurred)):
    for j in range(len(blurred[0])):
      sum=0.0
      for k in range(blursize*blursize):
        t=blursize*i+k//blursize
        s=blursize*j+k%blursize
        sum+=normalized[t][s]
      sum/=(blursize*blursize)
      blurred[i][j]=np.uint8(sum)
  # resize by bicubic interpolation
  background= cv2.resize(blurred, (ylim, xlim), interpolation=cv2.INTER_CUBIC)
  # since negative numbers of type uint8 doesn't exist, first convert to int
  # and shift above by the min intensity
  realenhanced=normalized.astype(int)-background.astype(int)
  realenhanced=realenhanced+np.min(realenhanced)
  realenhanced[realenhanced>255]=255
  realenhanced=realenhanced.astype(np.uint8)
  # do histogram equalization for each 32x32 blocks
  for i in range(len(realenhanced)//32):
    for j in range(len(realenhanced[0])//32):
      realenhanced[i*32:(i+1)*32,j*32:(j+1)*32]=cv2.equalizeHist(realenhanced[i*32:(i+1)*32,j*32:(j+1)*32])
  return realenhanced
