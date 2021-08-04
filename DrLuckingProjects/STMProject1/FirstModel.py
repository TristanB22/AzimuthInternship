import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from functools import partial

def train_model(labels, finalArray):
    DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")
    print(labels.shape)
    print(finalArray.shape)
    model = keras.models.Sequential([
        #first layer is 64 7x7 filters
        DefaultConv2D(filters=64, kernel_size=7, input_shape=[17, 15, 4]),
        # pool reduce each spatial dimension by factor of 2
        keras.layers.MaxPooling2D(pool_size=2),
        #Repeat 2 convolution layers followed by one pool twice
        #number of filters increases as go further along
        DefaultConv2D(filters=128),
        DefaultConv2D(filters=128),
        keras.layers.MaxPooling2D(pool_size=2),
        DefaultConv2D(filters=256),
        DefaultConv2D(filters=256),
        keras.layers.MaxPooling2D(pool_size=2),
        #need to flatten input for dense layer because need 1D array for input
        keras.layers.Flatten(),
        #Fully connected network, dropout layer to reduce overfit
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=2, activation='softmax'),
    ])


    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    history = model.fit(finalArray, labels, epochs=30, validation_data=(finalArray[300:, :, :, :], labels[300:]))
    score = model.evaluate(finalArray[325:, :, :, :], labels[325:])

    print(model.summary())

    countTrain = 0

    X_new = finalArray # pretend we have new images
    y_pred = model.predict(X_new)
    for i in range(436):
        pred = y_pred[i]
        if possibleResults[pred.argmax()] != labels[i]:
            countTrain += 1
            print("Validation Image {} was incorrect :: {}".format(i, labels[count]))

    if countTrain > 10 :
        return train_model(labels, finalArray)
    return model, score

factor = 2
imagesWithDefectsNumbers = [23, 24, 25, 179, 199, 200, 201, 221, 222, 223]

img1 = plt.imread("/Users/tristanbrigham/GithubProjects/AzimuthInternship/DrLuckingProjects/STMProject1/firstSTM.png")
img2 = plt.imread("/Users/tristanbrigham/GithubProjects/AzimuthInternship/DrLuckingProjects/STMProject1/image2.png")
img3 = plt.imread("/Users/tristanbrigham/GithubProjects/AzimuthInternship/DrLuckingProjects/STMProject1/image3.png")

imgCopy = img1.copy()
img2 = cv.resize(img2, dsize=(img1.shape[1], img1.shape[0]), interpolation=cv.INTER_NEAREST)
img3 = cv.resize(img3, dsize=(img1.shape[1], img1.shape[0]), interpolation=cv.INTER_NEAREST)

print("Size1: {}".format(img1.shape))
print("Size2: {}".format(img2.shape))
print("Size3: {}".format(img3.shape))

imagesWithDefects = np.zeros((10, 17, 15, 4))

finalArray = np.zeros((436, 17, 15, 4))
# finalArray2 = np.zeros((336, 17, 15, 4))
# finalArray3 = np.zeros((336, 17, 15, 4))

img2Array = np.zeros((336, 17, 15, 4))
img3Array = np.zeros((336, 17, 15, 4))

yAmt = int(img1.shape[1] /  14)
offsetY = int(yAmt / 2)

xAmt = int(img1.shape[0] / 12)
# offsetX = int(xAmt / 2)
offsetX = xAmt     #uncomment the above line and comment this one to add overlap

count = 0
x = xAmt

possibleResults = [0, 1]

while x < img1.shape[1]:
    y = yAmt
    while y < img1.shape[0] - offsetY:
        if count in imagesWithDefectsNumbers:
            cv.rectangle(imgCopy, (x - xAmt, y - yAmt), (x, y), (0, 255, 0))
        imgTemp1 = img1[y - yAmt:y, x - xAmt:x]
        finalArray[count] = imgTemp1
        
        imgTemp2 = img2[y - yAmt:y, x - xAmt:x]
        img2Array[count] = imgTemp2

        imgTemp3 = img3[y - yAmt:y, x - xAmt:x]
        img3Array[count] = imgTemp3
        count += 1
        y += offsetY
        if(count > 1000):
            exit()
    x += offsetX
# 
# finalArray[336: 672] = finalArray2
# finalArray[672: 1008] = finalArray3

# count = 0
# for img in finalArray[:1008]:
#     cv.imwrite("Img{}.png".format(count), img)
#     count += 1
    
print(finalArray.shape)

# labels = np.zeros((1108)) #going to be a zero if there is no defect
# labels[1008 : 1109] = np.ones((100))

labels = np.zeros((436))
labels[336: 436] = np.ones((100))

count = 0
for i in imagesWithDefectsNumbers:
    labels[i] = 1
    imagesWithDefects[count] = finalArray[i]
    count += 1

for i in range(10):
    # finalArray[1008 + (i * 10) : 1018 + (i * 10)] = imagesWithDefects[0 : 10]
    finalArray[336 + (i * 10) : 346 + (i * 10)] = imagesWithDefects[0 : 10]

xSections = int(img1.shape[1]/finalArray.shape[2])
ySections = int(336/xSections) #using 336 because count is wonky -- should be final Array y shape

print("x sect # is {} and y sect # is {}".format(xSections, ySections))
# 21 sections in y direction
# 16 sections in x direction

model, score = train_model(labels, finalArray)

# print("Accuracy: {}".format(score[0]['accuracy']))

for count in imagesWithDefectsNumbers:
    x = offsetX * int(count / ySections) + xAmt
    y = offsetY * (count % ySections) + yAmt
    cv.rectangle(img1, (x - xAmt, y - yAmt), (x, y), (0, 255, 0))

newPred2 = model.predict(img2Array)
count = -1
defective2 = []
nonDefective2 = []
for pred in newPred2:
    count += 1
    if possibleResults[pred.argmax()] == 1:
        defective2.append(count)
        x = offsetX * int(count / ySections) + xAmt
        y = offsetY * (count % ySections) + yAmt
        cv.rectangle(img2, (x - xAmt, y - yAmt), (x, y), (0, 255, 0))
    else:
        nonDefective2.append(count)

print("2:\nDEFECTIVE: {} \n\nNONDEFECTIVE: {}\n\n".format(defective2, nonDefective2))

newPred3 = model.predict(img3Array)
count = -1
defective3 = []
nonDefective3 = []
for pred in newPred3:
    count += 1
    if possibleResults[pred.argmax()] == 1:
        defective3.append(count) 
        x = offsetX * int(count / ySections) + xAmt
        y = offsetY * (count % ySections) + yAmt
        cv.rectangle(img3, (x - xAmt, y - yAmt), (x, y), (0, 255, 0))
    else:
        nonDefective3.append(count)

print("3:\nDEFECTIVE: {} \n\nNONDEFECTIVE: {}\n\n".format(defective3, nonDefective3))

imgCopyOUT = cv.resize(imgCopy, dsize=(img1.shape[1]*factor, img1.shape[0]*factor), interpolation=cv.INTER_NEAREST)
img1OUT = cv.resize(img1, dsize=(img1.shape[1]*factor, img1.shape[0]*factor), interpolation=cv.INTER_NEAREST)
img2OUT = cv.resize(img2, dsize=(img1.shape[1]*factor, img1.shape[0]*factor), interpolation=cv.INTER_NEAREST)
img3OUT = cv.resize(img3, dsize=(img1.shape[1]*factor, img1.shape[0]*factor), interpolation=cv.INTER_NEAREST)

cv.imshow("ImgCopy Bounded", imgCopyOUT)
cv.imshow("Img1 Bounded", img1OUT)
cv.imshow("Img2 Bounded", img2OUT)
cv.imshow("Img3 Bounded", img3OUT)
if cv.waitKey(0) & 0xFF == ord('q'):
    cv.destroyAllWindows()
