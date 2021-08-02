from re import T
import cv2 as cv
import numpy as np

file = open("/Users/tristanbrigham/GithubProjects/AI_Training_Data/LicensePlateProject/training_data", "r")
# cv.imwrite("/Users/tristanbrigham/GithubProjects/AI_Training_Data/LicensePlateProject/0.txt", np.zeros((60, 80)))
linesarr = file.readlines()
print("LENGTH: {}".format(len(linesarr)))
for count in range(1730, len(linesarr)):
    array = np.loadtxt("/Users/tristanbrigham/GithubProjects/AI_Training_Data/LicensePlateProject/" + str(count) + ".txt")
    if len(array) < 4800:
        print("APPENDING {}".format(4800 - len(array)))
        array2 = np.zeros(4800 - len(array))
        array = np.append(array, array2)
        file = "/Users/tristanbrigham/GithubProjects/AI_Training_Data/LicensePlateProject/" + str(count) + ".txt"
        np.savetxt(file, array)
    array = np.reshape(array, (80, 60))
    cv.imshow("Image", array)
    print(linesarr[count])
    cv.waitKey(0)

0
P
Z

