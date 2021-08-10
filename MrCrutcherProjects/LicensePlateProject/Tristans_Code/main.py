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
        self.img = cv.resize(cv.imread("/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate3.jpeg"), (640, 480))
        self.element_structure = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(20, 5))

    def preprocessing(self):
        imgBlur = cv.GaussianBlur(self.img, (21, 21), 0)
        gray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)
        sobelx = cv.Sobel(gray, cv.CV_8U, 1, 0, ksize=5)
        sobely = cv.Sobel(gray, cv.CV_8U, 0, 1, ksize=5)
        # gray = cv.bitwise_not(gray)
        # sobelx = np.abs(sobelx)
        # sobelx64f = cv.Sobel(self.img,cv.CV_64F,1,0,ksize=5)
        # abs_sobel64f = np.absolute(sobelx64f)
        # sobel_8u = np.uint8(abs_sobel64f)
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        element = self.element_structure
        morph_thresh = thresh.copy()
        cv.morphologyEx(morph_thresh, op=cv.MORPH_CLOSE, kernel=element, dst=morph_thresh, iterations=1)
        # cv.morphologyEx(morph_thresh, op=cv.MORPH_ERODE, kernel=element, dst=morph_thresh, iterations=2)
        # cv.morphologyEx(morph_thresh, op=cv.MORPH_OPEN, kernel=element, dst=morph_thresh, iterations=2)

        return morph_thresh

    def preprocessingCanny(self):
        imgBlur = cv.GaussianBlur(self.img, (19, 19), 0)
        gray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)
        # ret, gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        # element = self.element_structure
        # morph_thresh = gray.copy()
        cv.morphologyEx(gray, op=cv.MORPH_CLOSE, kernel=self.element_structure, dst=gray)
        edged = cv.Canny(gray, 50, 220)
        cv.imshow("Gray", gray)
        cv.imshow("Canny", edged)
        return edged

    def extract_largest_contours(self, after_preprocess, keep=10):
        contours= cv.findContours(after_preprocess.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        print("Length of Contours: {}".format(len(contours)))
        return contours[:keep]

    def findRectsandContours(self, after_preprocess, keep=-1):
        contours= cv.findContours(after_preprocess.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        print("Length of Contours: {}".format(len(contours)))
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        ret = []
        if keep == -1:
            keep=len(contours)
        for c in contours[:keep]:
            peri = cv.arcLength(c, True)    
            approx = cv.approxPolyDP(c, 0.02 * peri, True,)
            x, y, w, h = cv.boundingRect(c)
            ratio = h / w
            if ratio < self.maxAspect and ratio > self.minAspect and len(approx) == 4:
                cv.drawContours(self.img, [c], -1, (0, 255, 0))
                cv.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0))
                ret.append(c)
        return ret

    def preprocessCanny2(self):
        imgBlur = cv.GaussianBlur(self.img, (19, 19), 0)
        gray = cv.cvtColor(imgBlur, cv.COLOR_BGR2GRAY)
        cont = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(cont)
        contours = sorted(contours, key=cv.contourArea, reverse=True)


# class TensorFlow:


image = FindPlate()
preprocessing_image = image.preprocessingCanny()
# rectangles = image.findRectsandContours(preprocessing_image, keep=20)

preprocessing_image = cv.cvtColor(preprocessing_image, cv.COLOR_GRAY2BGR)

for rectangle in preprocessing_image:
    x, y, w, h = cv.boundingRect(rectangle)
    cv.rectangle(preprocessing_image, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)
    cv.rectangle(image.img, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)


cv.imshow("Img", image.img)
cv.imshow("Preprocess", preprocessing_image)
# cv.imshow("contours", cv.drawContours(contourImg, contours, -1, (0, 255, 0)))
if cv.waitKey(0) & 0xFF == ord('q'):
    exit()
