import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from functools import partial

img = plt.imread("/Users/tristanbrigham/GithubProjects/AzimuthInternship/DrLuckingProjects/STMProject1/firstSTM.png")
print("Size: {}".format(img.shape))

finalArray = np.zeros((436, 17, 15, 4))

yAmt = int(img.shape[1] /  14)
offsetY = int(yAmt / 2)

xAmt = int(img.shape[0] / 12)
# offsetX = int(xAmt / 2)
offsetX = xAmt     #uncomment the above line and comment this one to add overlap

count = 0
x = xAmt

possibleResults = [0, 1]

while x < img.shape[1]:
    y = yAmt
    while y < img.shape[0] - offsetY:
        imgTemp = img[y - yAmt:y, x - xAmt:x]
        finalArray[count] = imgTemp
        count += 1
        y += offsetY
        if(count > 1000):
            exit()
    x += offsetX
    
print(finalArray.shape)

imagesWithDefectsNumbers = [23, 24, 25, 179, 199, 200, 201, 221, 222, 223]
imagesWithDefects = np.zeros((10, 17, 15, 4))

labels = np.zeros((436)) #going to be a zero if there is no defect
labels[336 : 437] = np.ones((100))

count = 0
for i in imagesWithDefectsNumbers:
    labels[i] = 1
    imagesWithDefects[count] = finalArray[i]
    count += 1

for i in range(10):
    finalArray[336 + (i * 10) : 346 + (i * 10)] = imagesWithDefects[0 : 10]

DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")
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
history = model.fit(finalArray, labels, epochs=20, validation_data=(finalArray[325:, :, :, :], labels[325:]))
score = model.evaluate(finalArray[325:, :, :, :], labels[325:])

# print("Accuracy: {}".format(score[0]['accuracy']))
X_new = finalArray # pretend we have new images
y_pred = model.predict(X_new)
count = 0
for pred in y_pred:
    if possibleResults[pred.argmax()] != labels[count]:
        print("Image {} was incorrect :: {}".format(count, labels[count]))
        plt.imshow(finalArray[count])
    count += 1
plt.show()