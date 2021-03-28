import cv2
import imutils
import os
import os.path
import json
import sys
import pytesseract
import re
import difflib
import csv
import dateutil.parser as dparser
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


#enter image, eg test.jpg

img = cv2.imread("test.jpg")

image = img.copy()

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 199, 15)

text = pytesseract.image_to_string(thresh2, lang='eng') 
fnameflag = 0
lnameflag = 0
fatherfnameflag = 0
fatherlnameflag = 0
dobflag= 0

flags= {'fnameflag' : 0
'lnameflag' : 0
'fatherfnameflag' : 0
'fatherlnameflag' :0
'dobflag': 0}


words= text.split(" ")
for i in words:
    j = i.split(" ")
    for k in j:
        if k in flags.keys():
            flags[k] = 1

q = 0
for title in flags:
    if flag[title] == 0:
        q=1
        break:
#if q == 1:
#  print("not verified")
#else:
#    print("verified")

#CHECK DETAILS IN THIS
imgdet = cv2.detailEnhance(image, sigma_s=50, sigma_r=20)
t = text.split(" ")


####SEGMENTED IMAGE SEARCH
  
# Read in the image
test = cv2.imread('pass.jpg')
# Change color to RGB (from BGR)

image = cv2.cvtColor(imgGray, cv2.COLOR_BGR2RGB)

pixel_vals = image.reshape((-1,3))
  
# Convert to float type
pixel_vals = np.float32(pixel_vals)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
  
# then perform k-means clustering wit h number of clusters defined as 3
#also random centres are initally chosed for k-means clustering
k = 2
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  
# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
  
# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
  

'''
img
del
gray or Ulta
kmean
canny
dilate
try erosion
cotour'''

imgCanny1 = cv2.Canny(segmented_image, 100,200)
cv2_imshow(imgCanny1)


'''kernel = np.ones((2,2), np.uint8)
imgdi = cv2.dilate(imgCanny1, kernel, iterations = 1)
cv2_imshow(imgdi)'''


imgcount = img.copy()

countors = cv2.findContours(imgdi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
countors = imutils.grab_contours(countors)
countors = sorted(countors, key = cv2.contourArea, reverse = True)[:5]

counterslist = []
for c in countors:

  peri = cv2.arcLength(c, True)
  approx = cv2.approxPolyDP(c, 0.02 * peri, False)
  #print(len(approx), approx, "\n\n\n")
  screenCnt = approx
  cv2.drawContours(imgcount, [screenCnt], -1, (0, 255, 0), 2)
  cv2_imshow(imgcount)

  if len(approx) >=4 and len(approx) <=6:
    for a in range(len(approx)-1):
      
      l1 = approx[a][0][0]
      l2 = approx[a][0][1]
      r1 = approx[a+1][0][0]
      r2 = approx[a+1][0][1]
      #print(l1,l2,r1,r2)
      eval = ((l1-r1)**2+(l2-r2)**2)**(0.5)
      
      counterslist.append(eval)

      #print(eval,"\n\n\n")
    e1 = approx[0][0][0]
    e2 = approx[0][0][1]
    f1 = approx[len(approx)-1][0][0]
    f2 = approx[len(approx)-1][0][1]
    print(e1,e2,f1,f2)
    evalend = ((e1-f1)**2+(e2-f2)**2)**(0.5)
    counterslist.append(evalend)
    print(counterslist)

    #print(counterslist)
    maxcount = max(counterslist)
    counterslist = [x/maxcount for x in counterslist]
    print(counterslist)

    for i in range(len(counterslist)):
      print(i)
      if counterslist[i]<0.3:
        print(counterslist[i])
        approx = np.delete(approx,i, axis=0)
      #print(approx)

  if len(approx) == 4:
    
    print(approx)
    screenCnt = approx
    cv2.drawContours(imgcount, [screenCnt], -1, (0, 255, 0), 2)
    cv2_imshow(imgcount)
    x,y,w,h=cv2.boundingRect(c)
    
    rect = cv2.rectangle(imgcount,(x,y),(x+w,y+h),(0,0,255),2)
    cv2_imshow(rect[x:, y:])
    break


m1 = approx[0][0][0]
m2 = approx[1][0][0]
y1 = (int)((m1+m2)/2)

n1 = approx[2][0][0]
n2 = approx[3][0][0]
y2 = (int)((n1+n2)/2)

p1 = approx[0][0][1]
p2 = approx[3][0][1]
x1 = (int)((p1+p2)/2)

q1 = approx[1][0][1]
q2 = approx[2][0][1]
x2 = (int)((q1+q2)/2)


z = rect[x1:x2, y1:y2]
z = cv2.resize(z, (750,500))



kernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
im = cv2.filter2D(z, -1, kernel)

zGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


Bar = zGray[105:250,25:170]

sign = zGray[360:440,350:500]

