import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# import pytesseract

img = cv.imread("/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate.jpeg")
img = cv.resize(img, (640, 480))

imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgray = cv.dilate(imgray, (3, 3), iterations=2)
imgray = cv.GaussianBlur(imgray, (5, 5), 0)
hist = cv.calcHist(imgray, [0], None, [256], [0, 256])
ret, thresh = cv.threshold(imgray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# imgray = cv.Canny(imgray, 100, 255)
# thresh = cv.bitwise_not(thresh)
# cv.imshow("thresh", thresh)

contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
img = cv.drawContours(img, contours, -1, (0, 255, 0), thickness=5)

# c = max(contours, key = cv.contourArea)
# x, y, w, h = cv.boundingRect(c)
# largestContour = imgray[y : y + h, x : x + w]
# largestContour = cv.bitwise_not(largestContour)
# cv.imshow("LargestContour", largestContour)

imgray = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)
print(len(contours))

for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness = 3)
    cv.rectangle(imgray, (x, y), (x + w, y + h), (0, 255, 0), thickness = 3)
    # cv.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), thickness = 3)

cv.imshow("Img with Contours", img)
cv.imshow("Grayscale", imgray)
# cv.imshow("Thresh", thresh)
plt.plot(hist)
plt.show()
if cv.waitKey(0) & 0xFF == ord('q'):
    exit()