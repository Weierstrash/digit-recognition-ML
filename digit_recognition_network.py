from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

#Loading the training set and test set
(train_images, train_labels), (test_images,test_labels) = mnist.load_data()

#creating the network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape = (28*28,)))

#adding a final layer for the classification
network.add(layers.Dense(10,activation = 'softmax'))

#compilation step
network.compile(optimizer = 'adagrad',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

#preprocessing the data before training
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')

#setting up the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

