import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imutils
# import pytesseract

class FindPlate:
    def __init__(self):
        self.min = 1000
        self.max = 60000
        self.maxAspect = 1
        self.minAspect = 0.15
        self.img = cv.resize(cv.imread("/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate.jpeg"), (640, 480))
        self.element_structure = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(5, 5))

    def preprocessCanny(self, keep=-1):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)
        edged = cv.Canny(gray, 75, 200)
        contours = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key = cv.contourArea, reverse = True)
        
        ret = []

        if keep == -1:
            keep=len(contours)
        for c in contours[:keep]:
            peri = cv.arcLength(c, True)    
            approx = cv.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                ret.append(c)

        return ret 

    def drawContours(self, contours):
        cv.drawContours(self.img, contours, -1, (0, 255, 0), thickness=3)
        
        # for c in contours:
        #     rect = cv.boundingRect(c)

imageToProcess = FindPlate()
contours = imageToProcess.preprocessCanny()
imageToProcess.drawContours(contours)            

cv.imshow("Original", imutils.resize(imageToProcess.img, height = 650))
cv.waitKey(0)
if cv.waitKey(0) & 0xFF == ord('q'):
    exit(0)