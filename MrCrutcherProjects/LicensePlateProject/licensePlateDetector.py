print("\n\nLOADING PROGRAM\n\n")

import cv2 as cv
print("Loaded CV")
import numpy as np
print("Loaded NP")
# import tensorflow as tf
# print("Loaded TF")
import imutils
print("Loaded IMUTILS")
# from skimage.filters import threshold_local
print("Loaded THRESHOLD")
# from skimage import measure
print("Loaded MEASURE")
# import pytesseract


class FindPlate:

    # Have to adjust so that the min and max are larger when analyzing the images and smaller when looking at the vids
    def __init__(self, checkWait = False, optimize=True, imgAddress = None, img = None):
        self.divideArea = 2.5 #This is the denominator for how much of the screen is analyzed (analyzes the [1/(variable)] portion of the image/vid)
                            #For example, if the bottom half should be analyzed, put in '2' 

        self.ratio_max = 3       #This is the maximum width to height ratio that a license plate can be in the program (for the US, about 4 is good while for EU plates, about 6-7 is good)
        self.ratio_min = 2      #This is the minimum width to height ratio 

        self.angle_min = 83      #After the angle of the cv.areaMinRect has a modulo 90 applied to it, the angle either needs to be close to upright (this value or above)
        self.angle_max = 7      # or close to horizontal (this value or below) in degrees

        self.area_min = 50       #minimum area of the accepted bounding boxes -- it recognizes plates with smaller values but there is no way that characters can be picked out. No use to have smaller
        self.area_max = 350      #max area of the accepted bounding boxes

        self.lower_canny = 90    #upper value for canny thresholding
        self.upper_canny = 140   #Lower value for canny thresholding

        #ASPECT variables are not used:
        self.aspect_max = 1      #the max amount of area that the license plate can cover within a bounding box to be considered
        self.aspect_min = 0.3    #the minimum amount of area that a license plate can cover within a bounding box to be considered

        self.blur = (11, 11)    #initializing the size component of the gaussian blur that is applied to the image
        self.offset = 0         #initializing the variable which keeps track of how far from the top of the image the program begins to analyze
        self.top_img = None     #initializing the variable which may hold the top part of the image for later
        self.element_structure = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(5, 5)) #basic elem structure for blurring

        self.roiArray = []      #array for holding the ROI's

        self.setupAndExec(checkWait = checkWait, optimize=optimize, imgAddress = imgAddress, img = img) #execute the program
    
    def setupAndExec(self, checkWait, optimize, imgAddress = None, img = None):
        if imgAddress is None and img is not None:
            self.img = img
        elif imgAddress is not None and img is None:
            self.img = cv.resize(cv.imread(imgAddress), (860, 480))
        else:
            print("-----------------------ERROR FINDING IMAGE-----------------------")
            exit(0)

        if optimize:        #Currently, optimize makes it so that only the bottom portion of the screen is analyzed
            self.offset = int(self.img.shape[0] * (self.divideArea - 1) / self.divideArea)  #How many pixels in the y direction are not analyzed from the top
            self.top_img = self.img[ : self.offset ]    #Taking the top potion of the image and saving it for later
            self.img = self.img[self.offset : ]         #reassigning the image to the portion being analyed
        
        self.x = self.img.shape[1]                      #getting the width of the image

        self.imgCopy = self.img.copy()                  #getting copies for analysis
        self.imgAreaRects = self.img.copy()             #the copy that will be used for bounding rectangles
        self.Canny = None                               #Initializing variable to hold Canny image
        self.run()

        if optimize:                                    #After being run, rejoin the images if optimize is on
            self.img = np.append(self.top_img, self.img, axis=0)
            self.imgAreaRects = np.append(self.top_img, self.imgAreaRects, axis=0)
        self.showImages(checkWait)                      #Show the images

    def preprocessCannyandContours(self):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY) #get grayscale image
        gray = cv.GaussianBlur(gray, self.blur, 0)         #Apply a blur
        edged = cv.Canny(gray, self.lower_canny, self.upper_canny)  #Getting the canny contours
        self.Canny = edged                              #assign the variable
        contours = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)    #Get the contours of the Canny image [remember that this will return more contours than we need
                                                                                            #Because Canny just returns lines]
        contours = imutils.grab_contours(contours)      #Get the contours using imutils
        
        cv.drawContours(self.imgCopy, contours, -1, (0, 0, 255), thickness=2)   #Draw the contours onto the copied image
        cv.imshow("Contours", self.imgCopy)             #show image
        return contours

    def sortContours(self, contours):                   #This sorts the contours based on how far the contours are from the middle of the screen (only looks at the x-pos)
        rects = []
        for c in contours:
            rects.append((cv.boundingRect(c), c))       #Creating a tuple array with bouding rects and the contours
        retContourMapping = []

        for i in range(len(rects)):                     #for every contour, first get the middle part of the bouding box x-pos wise
            rect, contour = rects[i]                    #Then we are going to subtract that value from the middle of the screen 
            x, _, w, _ = rect                           #then we take the abs. value of that and sort those values in increasing fashion
            x = int(self.x / 2) - x                     #If necessary, this would allow us to put a cap on processing and only look at contours in the middle of the screen
            x = abs(x + int(w/2))
            retContourMapping.append((i, x, rects[i], contour))

        retContourMapping.sort(key=lambda tup: tup[1])  # sorts in place by distance from vertical horizontal line

        keys = []
        for index, _, _, _ in retContourMapping:
            keys.append(index)
        return keys


    def contourManipulation(self, contours):
        # keys = self.sortContours(contours)            #This is not currently being used -- it would only be used if we wanted to optimize by looking at the 
        checkIndividual = False                         #contours which are in the middle of the screen x-wise. It can also be changed to look at contours close to 
        ret = []                                        #a certain y-val

        # for key in keys:
        #     c = contours[key]
        
        for c in contours:
            boundingRectangle, boolRect = self.checkMinRect(c)

            if boolRect:
                ret.append(boundingRectangle)
            
            if checkIndividual:                         #if the check individual option is on, then go through the contours one-by-one, write them to the image, and show the image
                print("\n\nCONTOUR: {}".format(cv.contourArea(c)))
                cv.imshow("Bounding Rects", imutils.resize(self.imgAreaRects, height = 900))
                if cv.waitKey(0) & 0xFF == ord('c'):    #This cycles through to the next contour
                    continue
                elif cv.waitKey(0) & 0xFF == ord('f'):  #This makes it so that the rest of the contours are drawn in an instant
                    checkIndividual = False
                elif cv.waitKey(0) & 0xFF == ord('q'):  #quits the program
                    exit()
        return ret 

    def drawContoursThick(self, contours):              #Draws the passed contours with a thickness of 2
        cv.drawContours(self.img, contours, -1, (255, 0, 0), thickness=2)

    def getRect(self, contour):                         #Method to just get the bounding rectangles
        return cv.boundingRect(contour)

    def checkMinRect(self, contour):                    #function for getting the min-area rectangle and validating whether it is ok
        rect = cv.minAreaRect(contour)                  #get the min area rect
        box = cv.boxPoints(rect)                       
        box = np.int0(box)                              #for drawing the min area rectangles
        brect= cv.boundingRect(contour)
        _, _, rw, rh = brect
        if self.validateRatio(rect, rw, rh):
            cv.drawContours(self.imgAreaRects,[box], 0, (0, 255, 0), 1)
            return brect, True                                 #if everything is right, then return the contour and true to show that it is valid
        else:
            # cv.drawContours(self.imgAreaRects,[box], 0, (0, 0, 255), 1)
            return None, False                          #else, return the contour and false



    def ratCheck(self, width, height):         
        ratio = float(width) / float(height)            #check whether the width to height ratio is wrong
        if ratio < 1:
            ratio = 1 / ratio
        
        return not (ratio < self.ratio_min or ratio > self.ratio_max) #if the area is not in range or the ratio is off, return false


    def validateRatio(self, rect, rw, rh):                   #more checking that the contour could be a license-plate
        (x, y), (width, height), angle = rect           #get all of the data about the minarea bounding rectangle
        if width == 0 or height == 0:
            return False

        angle = angle % 90
        area = width * height                           
        
        if not ((angle < self.angle_max or angle > self.angle_min) and (area > self.area_min and area < self.area_max)):
            return False

        if rw < rh:
            return False

        return self.ratCheck(width, height)

    def initializeImages(): #putting the images in the right places on the screen so that they do not all stack up on run
        cv.imshow("Original", np.zeros((10, 3)))
        cv.imshow("Contours", np.zeros((10, 3)))
        cv.imshow("Bounding Rects", np.zeros((10, 3)))
        cv.imshow("Canny", np.zeros((10, 3)))

        #comment out the 3 lines below if you would like for the windows to be able to move
        # cv.moveWindow("Contours", 530, -100)
        # cv.moveWindow("Bounding Rects", 530, 285)
        # cv.moveWindow("Canny", 530, 100)

    def run(self):                                      #master run function for the program
        boundingRectangles = self.contourManipulation(self.preprocessCannyandContours())
        ###FIX THIS:::
        # for count, brect in enumerate(boundingRectangles):
        #     x, y, w, h = brect
        #     if h > 0 and w > 0:
        #         img = self.img[x : x + w, y + self.offset : y + self.offset + h]
        #         cv.imshow("ROI {}".format(count), img)

    def showImages(self, checkWait, height = 300):      #showing the images and putting them in the right place on the screen

        cv.imshow("Original", imutils.resize(self.img, height = height))
        cv.imshow("Contours", imutils.resize(self.imgCopy, height = height))
        cv.imshow("Bounding Rects", imutils.resize(self.imgAreaRects, height = height * 4))
        cv.imshow("Canny", imutils.resize(self.Canny, height = height))

        if checkWait:                                   #if going through the contours, check if q is pressed
            key = cv.waitKey(0) & 0xFF
            print("NEXT IMAGE")
            if key == ord('q'):
                exit(0)
        else:
            key = cv.waitKey(1)
            if key & 0xFF == ord('q'):                  #exit button
                exit(0)
            elif key & 0xFF == ord('p'):                # this creates a pause button for the video, in essence
                while True:
                    key = cv.waitKey(25) & 0xFF
                    if key == ord('p'):                 #unpause
                        break
                    elif key == ord('q'):               #quit the program button
                        exit(0)

if __name__ == "__main__":

    imageAddresses = [
        "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate.jpeg",
        "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate2.jpeg",
        "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate3.jpeg",
        "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate4.jpeg",
        "/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate5.jpeg",
    ]

    print("\n\nWelcome\nPlease press q to quit the program\nPlease press p to pause and unpause during the video\nPlease press anything else to continue through the images")
    print("\nOnce you have looked at all of the still images, the video will begin\n\n")
    print("Green boxes signify possible license plate regions \nwhile red ones show other ROI's which were picked up and discarded")

# Uncomment the lines below to see the still image recognitions
    # for image in imageAddresses:
    #     imageToProcess = FindPlate(checkWait = True, imgAddress = image)
    
    cap = cv.VideoCapture('/Users/tristanbrigham/Downloads/BostonVid.mp4')
    print("Starting Video")

    FindPlate.initializeImages()

    while(cap.isOpened()):                          #reading and analyzing the video as it runs
        ret, img = cap.read()
        img = imutils.resize(img, width=640)
        if ret == True:
            FindPlate(imgAddress = None, img = img)
        else:
            break
    
    cap.release()
    cv.destroyAllWindows()
