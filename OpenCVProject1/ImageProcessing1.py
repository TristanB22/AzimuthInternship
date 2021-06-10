#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
img = cv.imread("White_text_Black_Background.jpg")
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgray = cv.GaussianBlur(imgray, (5, 5), 0)
thresh,imgBW = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
plt.imshow(imgBW, cmap = "gray")
cv.waitKey(0)
