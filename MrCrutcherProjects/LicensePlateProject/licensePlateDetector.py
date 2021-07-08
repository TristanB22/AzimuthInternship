import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imutils
# import pytesseract

class FindPlate:
    def __init__(self, imgAddress):
        self.min = 1000
        self.max = 60000
        self.maxAspect = 1
        self.minAspect = 0.1
        self.img = cv.resize(cv.imread(imgAddress), (640, 480))
        self.element_structure = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(5, 5))

    def preprocessCanny(self, keep=-1):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)
        edged = cv.Canny(gray, 100, 200)
        cv.imshow("Canny", edged)
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
                x, y, w, h = cv.boundingRect(c)
                ratio = h / w
                if ratio < self.maxAspect and ratio > self.minAspect:
                    ret.append(c)

        return ret 

    def drawContours(self, contours):
        cv.drawContours(self.img, contours, -1, (0, 255, 0), thickness=2)
        
    def run(self):
        contours = self.preprocessCanny()
        self.drawContours(contours) 

        cv.imshow("Original", imutils.resize(imageToProcess.img, height = 650))
        while True:
            key = cv.waitKey(0) & 0xFF
            if  key == ord('p'):
                break
            elif key == ord('q'):
                exit(0)

imageAddresses = [
    "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate.jpeg",
    "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate2.jpeg",
    "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate3.jpeg",
    "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate4.jpeg",
    "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate5.jpeg",
]

print("\n\nWelcome\nPlease press p to continue through the images\nPlease press q to quit the program")

for image in imageAddresses:
    imageToProcess = FindPlate(image)
    imageToProcess.run()
