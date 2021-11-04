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

# The localizer won't generate a localized graph as in the paper(although when tuning it I did store the localized images),
# since it is really not necessary to just set the outer pixels of the
# original image to black. We only need to record radius and center of boundaries to do machine learning.
# The localizer generates a vector ((r1,c1,r2,c2,xl,yb),img) where r1 c1 are the radius and center of the inner boundary circle
# and r2 c2 are that of outer boundary circle. img is the original image,
# (xl,yb) is the origin of these coordinates, meaning that the actually coordinate
# should be (x+xl,y+yb) in the original image.
# I did not take pixels from the localized image since the outer circle may go beyond the boundary, so we need the full
# image to get pixels.
def dist1(a,b):
  return (a[0]-b[0])*(a[0]-b[0])+(a[1]-b[1])*(a[1]-b[1])
  
def localizer(s):
  res=[]
  img = cv2.imread(s)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_blur = cv2.GaussianBlur(img, (5,5), 0) # need blur to remove noise for better detection, 5 is just a standard kernel size
  # instead of following the projection technique described by the paper, I decided to first
  # find the pupil and set the center to be the center of the pupil
  # This is because that the technique in the paper would fail for some
  # images in the training set
  edges = cv2.Canny(image=img_blur, threshold1=150, threshold2=150)
  circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1000,
                               param1=100, param2=10,
                               minRadius=25, maxRadius=80) 
  if circles is None:
    return None
  center=(int(round(circles[0][0][1])),int(round(circles[0][0][0])))
  # parameter explanation:
  # threshold in canny need to be high to remove noise. Pupil boundary has very high gradient so can always be detected
  # set the mindistance between circles to be 1000 so that there will be only one circle
  # param1 and param2 are not very important, just set to 100,10 or 100,30 so that a cycle can be detected, is param2 is too large
  # circle detection may fail
  # set a smaller range for radius so that pupil can be detected
  # the center is output as (y,x) so reverting is needed
  

  # use the 240x240 region around the center as our ROI in localization stage, 120x120 region in the paper would be too small
  # (xl,yb) is our new origin
  r=120
  xl=max(0,center[0]-r)
  xr=min(len(img),center[0]+r)
  yb=max(0,center[1]-r)
  yt=min(len(img[0]),center[1]+r)
  loc=np.copy(img[xl:xr,yb:yt])

  # repeat the same pupil finding process to more accurately locate the pupil
  # find the center and radius of pupil
  img_blur = cv2.GaussianBlur(loc, (5,5), 0)
  edges = cv2.Canny(image=img_blur, threshold1=150, threshold2=150)
  circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1000,
                               param1=100, param2=10,
                               minRadius=25, maxRadius=80)
  cent = (circles[0][0][0], circles[0][0][1])
  radius = circles[0][0][2]
  if circles is None:
    return None
  

  # two different set of edges are created
  # one to detect obscure outer boundaries(10,30), the other to detect clear boundaries(20,30)
  edges2 = cv2.Canny(image=img_blur, threshold1=10, threshold2=30)
  edges3 = cv2.Canny(image=img_blur, threshold1=20, threshold2=30)
  

  # set the area around and within pupil boundary to black, so that we can detect
  # the outer boundary without being influenced by the pupil
  for i in range(len(edges2)):
    for j in range(len(edges2[0])):
      if (i-cent[1])*(i-cent[1])+(j-cent[0])*(j-cent[0])<=radius*radius+1000:
        edges2[i][j]=0
        edges3[i][j]=0


  # find two circles from two edges, the radius range is set to be larger to detect outer boundary
  # maxRadius of the first one is smaller just because some how it can detect boundary better for
  # some individual cases
  # other parameters just follw the same idea described in pupil detection phase
  circles2 = cv2.HoughCircles(edges2, cv2.HOUGH_GRADIENT, 1, 1000,
                               param1=100, param2=30,
                               minRadius=80, maxRadius=170)
  circles3 = cv2.HoughCircles(edges3, cv2.HOUGH_GRADIENT, 1, 1000,
                               param1=100, param2=30,
                               minRadius=80, maxRadius=180)
  realr=0
  realcent=0

  # pick the one whose center is closer to pupil center as our real outer boundary
  if circles3 is None and circles2 is None:
    return None
  elif circles3 is not None and circles2 is not None:
    cent2=(circles2[0][0][0], circles2[0][0][1])
    r2=circles2[0][0][2]
    cent3=(circles3[0][0][0], circles3[0][0][1])
    r3=circles3[0][0][2]
    if dist1(cent,cent2)>dist1(cent,cent3):
      realr=r3
      realcent=cent3
    else:
      realr=r2
      realcent=cent2
  elif circles3 is not None:
    realr=circles3[0][0][2]
    cent3=(circles3[0][0][0], circles3[0][0][1])
    realcent=cent3
  else:
    realr=circles2[0][0][2]
    cent2=(circles2[0][0][0], circles2[0][0][1])
    realcent=cent2
  
  # actually the None cases would not happen in our localization for all images, but they are written just in case

  return [[radius,cent,realr,realcent,xl,yb,],img]
  
