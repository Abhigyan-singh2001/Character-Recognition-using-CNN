# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:53:00 2021

@author: ABHIGYAN
"""
# importing the required libraries...

from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.utils import shuffle

#importing dataset...

Data = pd.read_csv(r"D:\A_Z Handwritten Data.csv").astype('float32')

#spliting data as column '0' contains the lables...

X = Data.drop('0',axis=1)
Y = Data['0']

#reshaping of data into 28 * 28 pixels as image...

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2)

X_train = np.reshape(X_train.values, (X_train.shape[0],28,28))
X_test = np.reshape(X_test.values, (X_test.shape[0],28,28))

print("Data for training: ", X_train.shape)
print("Data for testing: ", X_test.shape)

# create dictionary to map values from index to letters
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}

# plotting the frequency of alphabets in the dataset..

train_Y = np.int0(Y)
count = np.zeros(26 , dtype='int')

for i in train_Y:
    count[i]+=1
    
alphabets=[]
for i in word_dict.values():
    alphabets.append(i)
    
fig, ax = plt.subplots(nrows=1,ncols=1 ,figsize=(10,10))
ax.barh(alphabets,count, height=0.1, color="green")

plt.xlabel("Number of elements ")
plt.ylabel("Alphabets")
plt.grid()
plt.show()

#shuffling the data to get random alphabets rom data set...
shuffling = shuffle(X_train[:100])

fig, ax= plt.subplots(nrows=3,ncols=3, figsize=(10,10))
axes = ax.flatten()

for i in range(9):
    axes[i].imshow(np.reshape(shuffling[i], (28,28)) , cmap="Greys")
    
plt.show()


#again reshaping the dataset...

train_x = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],1)
print("New shape of training data is: ",train_x.shape)

test_x = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],1)
print("New shape of test data is: ", test_x.shape)

#converting single float values to catagorical data for CNN
y_train_categorical = to_categorical(Y_train, num_classes=26, dtype='int')
print("New shape og train labels id: ", y_train_categorical.shape)

y_test_categorical = to_categorical(Y_test, num_classes=26, dtype='int')
print("New shape of test labels is: ", y_test_categorical.shape)


# building a convolution neural Network
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2),strides=2))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(26, activation='softmax'))

# compiling the model..
model.compile(optimizer= Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
 
history=model.fit(train_x, y_train_categorical, epochs=1, validation_data=(test_x,y_test_categorical))
model.summary()
model.save(r'model_hand.h5')

#predicting from test set
fig, axes = plt.subplots(3,3, figsize=(8,9))
axes = axes.flatten()

for i,ax in enumerate(axes):
    img = np.reshape(test_x[i], (28,28))
    ax.imshow(img, cmap="Greys")

    pred = word_dict[np.argmax(y_test_categorical[i])]
    ax.set_title("Prediction: "+pred)
    ax.grid()

# prediction on user input image
img = cv2.imread(r'D:\image_c.png')
img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (400,440))

img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

img_final = cv2.resize(img_thresh, (28,28))
img_final =np.reshape(img_final, (1,28,28,1))


img_pred = word_dict[np.argmax(model.predict(img_final))]

cv2.putText(img, "Sample Image...", (20,25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color = (0,0,230))
cv2.putText(img, "Prediction: " + img_pred, (20,410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color = (255,0,30))
cv2.imshow('handwritten character recognition', img)


while (1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()


