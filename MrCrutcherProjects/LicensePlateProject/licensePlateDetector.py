import cv2 as cv
import numpy as np
import tensorflow as tf
import imutils
from skimage.filters import threshold_local
from skimage import measure
# import pytesseract




class FindPlate:

    def __init__(self, checkWait = False, imgAddress = None, img = None):
        self.minArea = 200
        self.maxArea = 9000
        self.maxAspect = 1
        self.minAspect = 0.1
        self.element_structure = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(5, 5))
        self.setupAndExec(checkWait = checkWait, imgAddress = imgAddress, img = img)
    
    def setupAndExec(self, checkWait, imgAddress = None, img = None):
        if imgAddress is None and img is not None:
            self.img = img
        elif imgAddress is not None and img is None:
            self.img = cv.resize(cv.imread(imgAddress), (860, 480))
        else:
            print("-----------------------ERROR FINDING IMAGE-----------------------")
            exit(0)
        self.x = self.img.shape[1]
        self.imgCopy = self.img.copy()
        self.imgAreaRects = self.img.copy()
        self.Canny = None
        self.run(checkWait)

    def preprocessing(self, input_img):
        imgBlurred = cv.GaussianBlur(input_img, (5,5), 0) # n, n is the size of the window, let's see if it's okay
        convertGray = cv.cvtColor(imgBlurred, cv.COLOR_BGR2GRAY) # grayscale conversion on a now blurred image
        sobelx = cv.Sobel(convertGray, cv.CV_8U, 1, 0, ksize=3) # veritcle x edges
        ret2, threshold_img = cv.threshold(sobelx, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU) #thresholding

        element = self.element_structure #grabbing the structure of the element we created in __init__
        morph_thresh = threshold_img.copy() # returning a copy of the tresholding for morphology
        cv.morphologyEx(src=threshold_img, op=cv.MORPH_CLOSE, kernel=element, dst=morph_thresh)#Morphological output, whitespacing essentially. 
        #The iterations for the morph control of course how much more white spacing you want, it appears 2 is "okay" 3 is better for 1 than the other. 

        cv.imshow("sobelx", sobelx)
        cv.imshow('thresh', threshold_img)
        cv.imshow('Blur', imgBlurred)
        return morph_thresh

    def extracted_contours(self, after_preprocess):
        contours, _ = cv.findContours(after_preprocess, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE) 
        cv.imshow('conts', after_preprocess) #show contours
        contours = imutils.grab_contours(contours)
        return contours

    def preprocessCannyandContours(self, keep = -1):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)
        edged = cv.Canny(gray, 75, 200)
        self.Canny = edged
        contours = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)
        
        cv.drawContours(self.imgCopy, contours, -1, (0, 0, 255), thickness=2)
        cv.imshow("Contours", self.imgCopy)
        return contours

    def sortContours(self, contours):
        rects = []
        for c in contours:
            rects.append((cv.boundingRect(c), c))
        retContours = []
        retContourMapping = []

        for i in range(len(rects)):
            rect, contour = rects[i]
            x, _, w, _ = rect
            x = int(self.x / 2) - x
            x = abs(x + int(w/2))
            retContourMapping.append((i, x, rects[i], contour))

        retContourMapping.sort(key=lambda tup: tup[1])  # sorts in place by distance from vertical horizontal line

        keys = []
        for index, _, _, _ in retContourMapping:
            keys.append(index)
        return keys


    def contourManipulation(self, contours):
        # contours = self.sortContours(contours, key=lambda x: abs(int(self.x / 2) - x[0] + int(w/2)))
        # contours = sorted(contours, key=lambda x: abs(int(self.x / 2) - x[0] + int(x[2]/2)), reverse=True)
        keys = self.sortContours(contours)
        checkIndividual = False
        ret = []

        for key in keys:
            c = contours[key]
            # peri = cv.arcLength(c, True)    
            # approx = cv.approxPolyDP(c, 0.02 * peri, True)

            rect, boolRect = self.getMinRect(c)

            if boolRect:
                ret.append(c)
            
            if checkIndividual:
                print("\n\nCONTOUR: {}".format(cv.contourArea(c)))
                cv.imshow("Bounding Rects", imutils.resize(self.imgAreaRects, height = 900))
                if cv.waitKey(0) & 0xFF == ord('c'):
                    continue
                elif cv.waitKey(0) & 0xFF == ord('f'):
                    checkIndividual = False
                elif cv.waitKey(0) & 0xFF == ord('q'):
                    exit()
        return ret 

    def drawContours(self, contours):
        cv.drawContours(self.img, contours, -1, (0, 255, 0), thickness=2)

    def getRect(self, contour):
        return cv.boundingRect(contour)

    def getMinRect(self, contour):
        rect = cv.minAreaRect(contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        if self.validateRatio(rect, contour):
            cv.drawContours(self.imgAreaRects,[box], 0, (0, 255, 0), 2)
            return rect, True
        else:
            cv.drawContours(self.imgAreaRects,[box], 0, (0, 0, 255), 2)
            return rect, False
        
    def preRatCheck(self, area, width, height):
        minVal = self.minArea
        maxVal = self.maxArea

        ratioMin = 1.5
        ratioMax = 7

        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        
        return not ((area < minVal or area > maxVal) or (ratio < ratioMin or ratio > ratioMax))

	### Validate that the ratio is correct. 
    def validateRatio(self, rect, c):
        (x, y), (width, height), angle = rect
        area = width * height
        if width * height < self.minArea:
            return False
        rx, ry, rw, rh = cv.boundingRect(c)
        if rw < rh:
            return False

        angle = angle % 90
        
        if not (angle < 10 or angle > 80):
            return False

        if (height == 0 or width == 0):
            return False

        return self.preRatCheck(area, width, height)



    def run(self, checkWait):
        contours = self.contourManipulation(self.preprocessCannyandContours())

        # preprocessedImage = self.preprocessing(self.img)
        # contours = self.contourManipulation(preprocessedImage)
        self.drawContours(contours) 
        self.showImages(checkWait)

    def showImages(self, checkWait, height = 300):
        cv.imshow("Original", imutils.resize(self.img, height = height))
        cv.imshow("Contours", imutils.resize(self.imgCopy, height = height))
        cv.imshow("Bounding Rects", imutils.resize(self.imgAreaRects, height = height * 3))
        cv.imshow("Canny", imutils.resize(self.Canny, height = height))

        #comment out the 3 lines below if you would like for the windows to be able to move
        cv.moveWindow("Contours", 530, -100)
        cv.moveWindow("Bounding Rects", 530, 285)
        cv.moveWindow("Canny", 0, 285)

        if checkWait:
            key = cv.waitKey(0) & 0xFF
            print("NEXT IMAGE")
            if key == ord('q'):
                exit(0)
        else:
            if cv.waitKey(25) & 0xFF == ord('q'):
                exit(0)

if __name__ == "__main__":

    imageAddresses = [
        "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate.jpeg",
        "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate2.jpeg",
        "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate3.jpeg",
        "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate4.jpeg",
        "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate5.jpeg",
    ]

    print("\n\nWelcome\nPlease press q to quit the program\nPlease press anything else to continue through the images")
    print("\nOnce you have looked at all of the still images, the video will begin\n\n")
    print("Green boxes signify possible license plate regions \nwhile red ones show other ROI's which were picked up and discarded")

    for image in imageAddresses:
        imageToProcess = FindPlate(checkWait = True, imgAddress = image)
    
    cap = cv.VideoCapture('/Users/tristanbrigham/Downloads/NewYorkVid.mp4')
    print("Starting Video")
    while(cap.isOpened()):
        ret, img = cap.read()
        img = imutils.resize(img, width=640)
        if ret == True:
            FindPlate(imgAddress = None, img = img)
            print("Analyzed")
            
        else:
            break
    
    cap.release()
    cv.destroyAllWindows()
