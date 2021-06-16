#!/usr/bin/env python
# coding: utf-8

import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np

img = cv.imread("/Users/tristanbrigham/GithubProjects/AzimuthInternship/OpenCVProject1/White_text_Black_Background.jpg")
imgray = img.copy()
imgray = cv.cvtColor(imgray, cv.COLOR_BGR2GRAY)
imgray = cv.GaussianBlur(imgray, (5, 5), 0)
thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# imgray = cv.Canny(imgray, 250, 254)
contours, hierarchy = cv.findContours(imgray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
imgray = cv.cvtColor(imgray, cv.COLOR_GRAY2BGR)
cv.drawContours(imgray, contours, -1, (255, 255, 255), 10)
contours = contours[0] if len(contours)==2 else contours
# cv.imshow("grayImage", imgray)
# cv.waitKey(0)
# cv.destroyAllWindows()
regionOfInterest = 0
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    ROI = imgray[y:y + h, x:x + w]
    cv.rectangle(imgray, (x, y), (x + w, y + h), (0, 255, 0), 5)
    cv.imwrite("ROI_0-{}.png".format(regionOfInterest), ROI)
    regionOfInterest += 1
cv.imshow("Bounded", imgray)
cv.waitKey(0)
cv.destroyAllWindows()

