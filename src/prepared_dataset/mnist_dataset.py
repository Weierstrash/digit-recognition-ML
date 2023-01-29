from keras.datasets import mnist
from keras.utils import to_categorical

#Loading the training set and test set
(train_images, train_labels), (test_images,test_labels) = mnist.load_data()

#preprocessing the data before training
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')

#setting up the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)