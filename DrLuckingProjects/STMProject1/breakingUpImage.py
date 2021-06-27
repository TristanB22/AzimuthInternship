import numpy as np
import matplotlib.pyplot as plt
import os

img = plt.imread("/Users/tristanbrigham/GithubProjects/AzimuthInternship/DrLuckingProjects/STMProject1/firstSTM.png")
print("Size: {}".format(img.shape))

yAmt = int(img.shape[1] /  14)
offset = int(yAmt / 2)

xAmt = int(img.shape[0] / 12)

count = 0
x = xAmt

for i in range(500):
    os.system("rm Portion{}.png".format(i))

while x < img.shape[1]:
    y = yAmt
    while y < img.shape[0] - offset:
        imgTemp = img[y - yAmt:y, x - xAmt:x]
        plt.imsave("Portion{}.png".format(count), imgTemp)
        count += 1
        y += offset
        if(count > 1000):
            exit()
    x += xAmt
print("----------------------------—SUCCESS----------------------------—")