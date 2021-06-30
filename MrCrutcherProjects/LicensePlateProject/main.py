import cv2 as cv
import numpy as np
import pytesseract

img = cv.imread("/Users/tristanbrigham/GithubProjects/AzimuthInternship/MrCrutcherProjects/LicensePlateProject/licensePlate.jpeg")

imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgray = cv.GaussianBlur(imgray, (5, 5), 0)

# imgray = cv.dilate(imgray (3, 50), iterations = 5)
# imgray = cv.Canny(imgray, 50)
ret, thresh = cv.threshold(imgray, 100, 255, cv.THRESH_OTSU)
# thresh = cv.bitwise_not(thresh)
cv.imshow("thresh", thresh)

contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# imgray = cv.drawContours(imgray, contours, -1, (0, 255, 0), thickness=5)

c = max(contours, key = cv.contourArea)
x, y, w, h = cv.boundingRect(c)
largestContour = img[y : y + h, x : x + w]
cv.imshow("LargestContour", largestContour)


thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)
print(len(contours))

for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness = 5)
    cv.rectangle(imgray, (x, y), (x + w, y + h), (0, 255, 0), thickness = 5)
    cv.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), thickness = 5)

cv.imshow("Img with Contours", img)
cv.imshow("Grayscale", imgray)
cv.imshow("Thresh", thresh)
if cv.waitKey(0) & 0xFF == ord('q'):
    exit()