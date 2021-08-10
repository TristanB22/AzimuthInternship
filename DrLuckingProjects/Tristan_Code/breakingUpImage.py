import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

img = plt.imread("/Users/tristanbrigham/GithubProjects/AzimuthInternship/DrLuckingProjects/STMProject1/firstSTM.png")
print("Size: {}".format(img.shape))

finalArray = np.zeros((436, 17, 15, 4))

yAmt = int(img.shape[1] /  14)
offsetY = int(yAmt / 2)

xAmt = int(img.shape[0] / 12)
# offsetX = int(xAmt / 2)
offsetX = xAmt     #uncomment the above line and comment this one to add overlap

count = 0
count2 = 0
x = xAmt

while x < img.shape[1]:
    y = yAmt
    while y < img.shape[0] - offsetY:
        imgTemp = img[y - yAmt:y, x - xAmt:x]
        finalArray[count] = imgTemp
        count += 1
        count2 += 1
        y += offsetY
        if(count > 1000):
            exit()
        if(count2 % 30 == 0):
            finalArray[count: count + 10] = np.ones((10))
            count += 10
            count2 = 0
    x += offsetX
    
print(finalArray.shape)
print("----------------------------—SUCCESS----------------------------—")