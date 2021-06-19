#!/usr/bin/env python
# coding: utf-8

import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import pytesseract

img = cv.imread("/Users/tristanbrigham/GithubProjects/AzimuthInternship/OpenCVProject1/White_text_Black_Background.jpg")
imgray = img.copy()
imgray = cv.cvtColor(imgray, cv.COLOR_BGR2GRAY)
imgray = cv.GaussianBlur(imgray, (7, 7), 0)

# imgTemp = imgray[0 : int(29 * imgray.shape[0] / 30)]
# imgTemp2 = imgray[int(29 * imgray.shape[0] / 30) : imgray.shape[0]]
# print(dtype(imgray))

# imgShift = np.zeros((imgray.shape[0], imgray.shape[1]), dtype=np.int)

# imgShift[0 : int(imgray.shape[0] / 30)] = imgTemp2
# imgShift[int(imgray.shape[0] / 30) : imgray.shape[0]] = imgTemp

# imgray = cv.addWeighted(imgray, 1, imgShift, 1, 0)

imgray = cv.dilate(imgray, (3, 50), iterations = 20)
thresh = cv.threshold(imgray, 0, 127, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# imgray = cv.Canny(imgray, 250, 254)
contours, _ = cv.findContours(imgray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
imgray = cv.cvtColor(imgray, cv.COLOR_GRAY2BGR)
cv.drawContours(imgray, contours, -1, (255, 255, 255), 10)
contours = contours[0] if len(contours)==2 else contours
# cv.imshow("grayImage", imgray)
# cv.waitKey(0)
# cv.destroyAllWindows()
regionOfInterest = 0

print(pytesseract.image_to_string(img))

for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    ROI = imgray[y:y + h, x:x + w]
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    # cv.imwrite("ROI_0-{}.png".format(regionOfInterest), ROI)
    regionOfInterest += 1
cv.imshow("Bounded Original", img)
cv.waitKey(15000)
cv.destroyAllWindows()

