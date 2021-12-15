import cv2
from matplotlib import pyplot as plot
import numpy as np
from PIL import Image
import numpy as np
import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import time
import datetime

inputFiles = []
targetValues = []

data = open('/content/optdigits-orig.windep')
readData = data.readlines()
for i in range(0, 1797):
  sampleArr = []
  for j in range(33*i, 33*i+33):
    testArr = []
    for s in (list(readData[21 + j].strip())):
      testArr.append(int(s))
    sampleArr.append(testArr)
  img = sampleArr[:-1]
  inputFiles.append(np.array(img)) #Creating the input files
  targetValues.append(sampleArr[-1]) #Extracting the class or target values

#The target values are then converted into an array with the shape of (10, 1) which has 10 elements in each array and the index of max value in the array corresponds to the output.
np.shape(np.array(inputFiles).reshape(-1, 32, 32, 1))

def buildNetwork():
  model = Sequential()
  model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(32,32, 1)))
  model.add(Conv2D(32, kernel_size=3, activation='relu'))
  model.add(Flatten())
  model.add(Dense(10, activation='softmax'))
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model


def checkValuesAndPlot(history): #Plotting the cross enropy and classification accuracy
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='red', label='train')
	pyplot.plot(history.history['val_loss'], color='green', label='test')
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='red', label='train')
	pyplot.plot(history.history['val_accuracy'], color='green', label='test')
	filename = sys.argv[0].split('/')[-1]
	filename = "outputGraph"
	pyplot.savefig(filename + '.png')
	pyplot.close()

def loadingDataset(): #Loading the dataset
  (trainX, trainY), (testX, testY) = (inputFiles[:1400], targetValues[:1400]), (inputFiles[1401:1797], targetValues[1401:1797])
  trainX = np.array(trainX)
  trainY = to_categorical(trainY)
  testY = to_categorical(testY)
  return np.array(trainX).reshape(-1, 32, 32, 1), np.array(trainY), np.array(testX).reshape(-1, 32, 32, 1), np.array(testY)

def checkingEachPixel(train, test): #casting training and testing object object to float dtype.
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	return train_norm, test_norm



 
startTime = datetime.datetime.now()
trainX, trainY, testX, testY = loadingDataset()
trainX, testX = checkingEachPixel(trainX, testX)
print(np.shape(trainX), np.shape(trainY))
model = buildNetwork()
checkModelValue = model.fit(trainX, trainY, epochs=2, batch_size=10, validation_data=(testX, testY), verbose=1)
_, acc = model.evaluate(testX, testY, verbose=1)
print('> %.3f' % (acc * 100.0))
endTime = datetime.datetime.now()
print("Total Time Take")
print(endTime - startTime)
checkValuesAndPlot(checkModelValue)