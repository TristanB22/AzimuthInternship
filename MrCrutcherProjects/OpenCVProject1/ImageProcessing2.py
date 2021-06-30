#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
from subprocess import run
output = run("pwd", capture_output=True).stdout
img = cv.imread("/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/OpenCVProject1/color-theory-1-1.jpg")
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgray = cv.GaussianBlur(imgray, (5, 5), 0)
thresh, imgray = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
imgray = cv.Canny(imgray, 250, 254)
contours, hierarchy = cv.findContours(imgray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
imgray = cv.cvtColor(imgray, cv.COLOR_GRAY2BGR)
cv.drawContours(imgray, contours, -1, (255, 0, 0), 2)
cv.imshow('Contours', imgray)
cv.waitKey()
cv.destroyAllWindows()

