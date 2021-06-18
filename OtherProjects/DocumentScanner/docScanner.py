from skimage.filters import threshold_local    # help from https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
import numpy as np                             # A lot of the code is based off of the site above, but I coded everything by hand so I would understand it
import argparse
import cv2 as cv
import imutils

def orderPts(pts):
    rect = np.zeros((4, 2), dtype= "float32")

    s = np.sum(pts, axis = 1)
    
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmax(d)]
    rect[3] = pts[np.argmin(d)]

    return rect
    

def warpImage(image, pts):
    rect = orderPts(pts)
    (topLeft, topRight, bottomRight, bottomLeft) = rect
    
    widthTopottom = (((bottomRight[0] - bottomLeft[0]) ** 2) + ((bottomRight[1] - bottomLeft[1])) ** 2) ** 0.5
    widthTop = (((topRight[0] - topLeft[0]) ** 2) + ((topRight[1] - topLeft[1]) ** 2)) ** 0.5
    width = max(int(widthTopottom), int(widthTop))
    
    heightRight = (((topRight[0] - bottomRight[0]) ** 2) + ((topRight[1] - bottomRight[1]))) ** 0.5
    heightLeft = (((topLeft[0] - bottomLeft[0]) ** 2) + ((topLeft[1] - bottomLeft[1]))) ** 0.5
    height = max(int(heightRight), int(heightLeft))
    
    dimensions = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    temp = cv.getPerspectiveTransform(rect, dimensions)
    warped = cv.warpPerspective(image, temp, (width, height))
    return warped        

def showImageOutline(img):
    ratio = img.shape[0] / 500.0
    frame = img.copy()
    img = imutils.resize(img, height = 500)

    # thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # imgray = cv.Canny(imgray, 250, 254)
    # contours, hierarchy = cv.findContours(imgray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(gray, 75, 200)
    contours = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv.contourArea, reverse = True)

    for c in contours:
        peri = cv.arcLength(c, True)    
        approx = cv.approxPolyDP(c, 0.02 * peri, True,)

        if len(approx) == 4:
            screenCnt = approx
            cv.drawContours(img, [screenCnt], -1, (0, 255, 0), thickness = 2)
            
            warped = warpImage(frame, screenCnt.reshape(4, 2) * ratio)
            warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
            T = threshold_local(warped, 11, offset = 10, method = "gaussian")       #I don't get this line or the next one
            warped = (warped > T).astype("uint8") * 255

            cv.imshow("Original", imutils.resize(img, height = 650))
            cv.imshow("Scanned", imutils.resize(warped, height = 650))
            if cv.waitKey(10) & 0xFF == ord('q'):
                exit(0)


ap = argparse.ArgumentParser()
ap.add_argument("--image", type = int, default = 0,
	help = "Path to the image to be scanned\nOr 0 to open camera")
args = ap.parse_args()

if(args.image == 0):
    cap = cv.VideoCapture(0)

    if not cap.isOpened:
        print("Failed to Open Camera")
        exit(0)

    while True:
        ret, frame = cap.read()
        if(frame is None):
            print("Nothing to Capture!")
            break
        
        showImageOutline(frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break