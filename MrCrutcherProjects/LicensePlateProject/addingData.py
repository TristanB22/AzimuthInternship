import cv2 as cv
import numpy as np
import imutils

counter = 2655

imageAddresses2 = [
        "license_plate_letter_0.png",
        "license_plate_letter_1.png",
        "license_plate_letter_2.png",
        "license_plate_letter_3.png"
    ]

training_file_keys ="/Users/tristanbrigham/GithubProjects/AI_Training_Data/LicensePlateProject/training_data.txt"

for imageaddr in imageAddresses2:
    roi = cv.imread("/Users/tristanbrigham/Desktop/" + str(imageaddr))

    regionOfInterest = roi
    name = "ROI"                           #format the name
    regionOfInterest = cv.cvtColor(regionOfInterest, cv.COLOR_BGR2GRAY)

    regionOfInterest[: int(regionOfInterest.shape[0] / 10), :]  += 50  #Increasing the brightness of the top of the image (BREAKS WITH HIGH VALUES because of overflow)

    regionOfInterest = imutils.resize(regionOfInterest, height=200, inter=cv.INTER_AREA)
    image = cv.GaussianBlur(regionOfInterest, (0, 0), 3)
    image = cv.addWeighted(image, 1.5, regionOfInterest, -0.5, 0)

    ret, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # ret, thresh = cv.threshold(image, self.contour_license_plate, 255, cv.THRESH_BINARY)
    # thresh = cv.dilate(thresh, (3, 5), iterations = 1)
    thresh = cv.bitwise_not(thresh)
    thresh = cv.erode(thresh, (81, 61), iterations = 12)
    # thresh = cv.dilate(thresh, (71, 3))
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    cv.imshow("GRAY", imutils.resize(thresh, height=200))

    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    letters = []
    for contour in contours:
        if cv.contourArea(contour) > 100:
            x, y, w, h = cv.boundingRect(contour)
            letterInterest = thresh[0 : y + h, x : x + w]
            # cv.imshow("Letter {}".format(count), imutils.resize(letterInterest, height = 100))
            # cv.moveWindow("Letter {}".format(count), 600, 110 * count)
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))
            letterImage = cv.resize(letterInterest, (60, 80))
            letters.append(letterImage)

    cv.imshow(name, image)   #showing and resizing image
    cv.imshow("LETTER", letters[0])
    cv.waitKey(10000)
    letter = input("WRITE THE LETTER: ").upper()[0]
    for i in range(100):
        counter = counter + 1
        file = open("/Users/tristanbrigham/GithubProjects/AI_Training_Data/LicensePlateProject/" + str(counter) + ".txt", "w")
        for row in letters[0]:
            np.savetxt(file, row)
        print("Letter passed: " + letter)
        training_file = open(training_file_keys, "a")
        training_file.write("\n" + str(letter))
cv.destroyAllWindows()