from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import layers
import tensorflow as tf
from keras import optimizers

import numpy as np

# loading the data, split between train and test sets (each consisting of corresponding image and label arrays)
(images_train, labels_train), (images_test, labels_test) = fashion_mnist.load_data()

#########################################################################
# Optional visualisation tools to help you understand the data.         #
# You can modify and this section (or ignore it) as you wish.           #
# The visualisation will no longer work after the data is preprocessed. #
# None of this will impact  points or grading of this task!             #
#########################################################################
import matplotlib.pyplot as plt

#creating an array with class names in correct position
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#print the shape of the training data
print('images_train shape:', images_train.shape)

#print the amount of training and test samples
print(images_train.shape[0], 'train samples')
print(images_test.shape[0], 'test samples')

#how visualize a single image
print('\n colored example visualization of an image:')
plt.figure()
plt.imshow(images_train[7])
plt.colorbar()
plt.grid(False)
plt.show()

#how the data of a single image is stored in the dataset
print('\n example array structure of an image (28 arrays with 28 entries each):\n', images_train[7] )

#how the data of a single label is stored in the dataset (one single value between 0-10 representing the class)
label = labels_train[7]
print('\n example label of an image:', label)
print(' this corresponds to class:', class_names[label])

#how to visualize the first 15 images with class name
#this can be used to verfiy that the data is still in correct format e.g. after transforming it
print('\n example visualization of 15 images with class name:')
plt.figure(figsize=(10,10))
for i in range(15):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels_train[i]])
plt.show()

################################################
# Preprocessing. Leave this section unchanged! #
################################################

# define number of classes
num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    images_train = images_train.reshape(images_train.shape[0], 1, img_rows, img_cols)
    images_test = images_test.reshape(images_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    images_train = images_train.reshape(images_train.shape[0], img_rows, img_cols, 1)
    images_test = images_test.reshape(images_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# scale pixel values (ranging from 0 to 255) to range of 0 to 1
# this is a normal pre-processing step for efficiency reasons
images_train = images_train.astype('float32')
images_test = images_test.astype('float32')
images_train /= 255
images_test /= 255


# convert class vectors to binary class matrices
# e.g. 6 becomes [0,0,0,0,0,1,0,0,0,0]
# this step is required to train your model with the categorical cross entropy loss function
# a function especially suited for multiclass classification tasks
labels_train = keras.utils.to_categorical(labels_train, num_classes)
labels_test = keras.utils.to_categorical(labels_test, num_classes)


#create sequential model
model = keras.Sequential()

################################
# Implement your solution here #
################################

#Convolution layer with 32 output filters, a kernel size of 3x3
model.add(layers.Conv2D(32,(3,3),activation=tf.nn.relu,
                                 input_shape=(28,28,1)))

model.add(layers.MaxPooling2D((2, 2))) #maxpooling of 2*2

#Convolution layer with 64 output filters, a kernel size of 4x4
model.add(layers.Conv2D(64, (4, 4), activation=tf.nn.relu))


model.add(layers.MaxPooling2D((2, 2)))#maxpooling of 2*2


#Dropout layer with a drop fraction of 0.3
model.add(layers.Dropout(0.3))

#Flatten layer
model.add(layers.Flatten())

#Fully-connected layer with 256 neurons
model.add(layers.Dense(256,activation=tf.nn.relu))

#Dropout layer with a drop fraction of 0.5
model.add(layers.Dropout(0.5))

#Fully-connected layer with as many neurons as there are classes in the problem (Output layer), activation function: Softmax
model.add(layers.Dense(10, activation=tf.nn.softmax))

#Learning rate: 0.005 with adam optimizer
sgd = optimizers.adam(lr=0.0006)

model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(images_train, labels_train, batch_size=100, epochs=3, verbose=1)



test_loss, test_acc = model.evaluate(images_test,  labels_test, verbose=2)

print('\nTest accuracy:', test_acc)

prediction = model.predict(images_test)

print('\n prediction of image_test[100]',prediction[100])




