
#importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras_preprocessing.image import ImageDataGenerator
import cv2
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns


#define the classes
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }


#read the classes from the path

path = "mydata"
count = 0
images = []
label = []
classes_list = os.listdir(path)
print("Total Classes:",len(classes_list))
no_of_classes = len(classes_list)
for x in range(0, len(classes_list)):
    img_list = os.listdir(path + "/" + str(count))
    for y in img_list:
        img = cv2.imread(path + "/" + str(count) + "/" + y)
        img = cv2.resize(img, (32,32))
        images.append(img)
        label.append(count)
    print(count, end = " ")
    print("th class imported")
    count += 1
print(" ")

#data resizing
images = np.array(images)
classNo = np.array(label)
data = images.reshape(-1,32,32,3)
images = np.array(images)
classNo = np.array(label)
data= np.array(images).reshape(-1, 32, 32, 3)
data.shape

X_train, X_test, y_train, y_test = train_test_split(data, classNo, test_size=0.2)
y_Tests = y_test
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
print("Dataset Shapes")
print("Train Size", end = "")
print(X_train.shape, y_train.shape)
print("Validation Size", end = "")
print(X_val.shape, y_val.shape)
print("Test Size", end = "")
print(X_test.shape, y_test.shape)

#image preprocessing
def grey_scale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grey_scale(img)
    img = equalize(img)
    img = img/255
    return img
X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

# Reshape into channel 1 we have converted rgb to greyscale
X_train = X_train.reshape(-1, 32, 32, 1)
X_val = X_val.reshape(-1, 32, 32, 1)
X_test = X_test.reshape(-1, 32, 32, 1)
X_test.shape

#data augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

y_train = to_categorical(y_train, no_of_classes)
y_val = to_categorical(y_val, no_of_classes)
y_test = to_categorical(y_test, no_of_classes)
y_test.shape

#model architecture
def seq_model():
    no_filters     = 60
    size_of_filter1 = (5,5)
    size_of_filter2 = (3,3)
    size_of_pool    = (2,2)
    no_of_nodes     = 500
    model           = Sequential()
    model.add((Conv2D(no_filters, size_of_filter1, input_shape = (32,32,1), activation = "relu")))
    model.add((Conv2D(no_filters, size_of_filter1, activation = "relu")))
    model.add(MaxPooling2D(pool_size = size_of_pool))
    
    model.add((Conv2D(no_filters//2, size_of_filter2, activation = "relu")))
    model.add((Conv2D(no_filters//2, size_of_filter2, activation = "relu")))
    model.add(MaxPooling2D(pool_size = size_of_pool))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(no_of_nodes, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(no_of_classes, activation = "softmax"))
    model.compile(Adam(learning_rate = 0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model
model = seq_model()
print(model.summary())

batch_size_val = 30
steps_per_epoch_val = 500
epochs_val = 40

#Train the model
history = model.fit(dataGen.flow(X_train, y_train, batch_size = batch_size_val), 
                   steps_per_epoch = steps_per_epoch_val, epochs = epochs_val,
                   validation_data = (X_val, y_val), shuffle = 1)

#graph plotting
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()

#testing the model 
score = model.evaluate(X_test,y_test,verbose=0)       
print('Test Score:',score[0])
print('Test Accuracy:',score[1])
model.save("Traffic_class_model.h5")