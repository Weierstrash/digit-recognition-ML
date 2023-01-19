from digit_recognition_network import network
from digit_recognition_network import train_images
from digit_recognition_network import train_labels

#training
def return_trained_network(epochs):
    network.fit(train_images, train_labels, epochs = epochs)
    return network

if __name__ == "__main__":
    #training for 5 epochs
    network = return_trained_network(epochs = 10)

    #saving the model
    network.save('saved_model')



    

