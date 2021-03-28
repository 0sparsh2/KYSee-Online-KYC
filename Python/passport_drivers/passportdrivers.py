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

flags= {'city' : 0
'state' : 0
'pincode' : 0
'number on id' :0
}


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