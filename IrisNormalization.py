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
def normalizer(locinfo,img,shift=0):
  # locinfo is the first element of the return value of localizer
  # img is the second element, which is the original image
  # shift is the angle to shift for image augmentation described in the paper
  if not locinfo:
    return None
  #normalized image size
  xlim=64
  ylim=512
  r1=locinfo[0]
  c1=locinfo[1]
  r2=locinfo[2]
  c2=locinfo[3]
  xl=locinfo[4]
  yb=locinfo[5]
  normalized=np.zeros(xlim*ylim, dtype=np.uint8).reshape(xlim, ylim)
  rec=0

  # for each i,j of normalized image, map to the corresponding position in the original image
  for i in range(xlim):
    for j in range(ylim):
      # prop is the relative distance from the line segment p1->p2, that is, Y/M in the paper.
      prop=i/(xlim-1)
      # because the coordinate system of python image is different from normal cartaesian coordinate,
      # following the same formula on the paper there would be a pi/2 rotation. Specifically: (image)(dx,dy)=(-dy,dx)(cartaesian)
      theta=2*math.pi*j/(ylim-1)+math.pi/2+shift   
      xd=math.cos(theta)
      yd=math.sin(theta)
      # p1 and p2 are the corresponding points with angle theta on inner and outer boundaries
      p1=(xd*r1+c1[1],yd*r1+c1[0])
      p2=(xd*r2+c2[1],yd*r2+c2[0])
      # get p on the line segment corresponding to prop
      p=(int(round(p1[0]+prop*(p2[0]-p1[0]))),int(round(p1[1]+prop*(p2[1]-p1[1]))))
      # p is the coordinate relative to the origin of localized image (xl,yb), so need to adjust it
      xfin=xl+p[0]
      yfin=yb+p[1]
      # in case exceeds the boundary of the original image
      # this rarely happens
      if xfin<0:
        xfin=0
      if yfin<0:
        yfin=0
      if xfin>=280:
        xfin=279
      if yfin>=320:
        yfin=319
      normalized[i][j]=img[xfin][yfin]

  return normalized
