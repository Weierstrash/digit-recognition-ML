import keras
import matplotlib.pyplot as plt
from src.util.prepare_image import imageprepare
import numpy as np



if __name__ == "__main__":
    use_cnn_network = False

    if use_cnn_network:
        #importing the saved model
        model = keras.models.load_model('src/saved_model_cnn')

        #image prepared for classification
        tensor_image = imageprepare('example_images/image.png')

        classes = model.predict(tensor_image.reshape(1,28,28))
        print(classes)
        
        #printing the index which has the highest weight
        #which is therefore the model prediction
        print(np.argmax(classes))

        #plotting
        plt.imshow(tensor_image.reshape(28,28),cmap = plt.cm.binary)
        plt.show()
        
    else:
        #image prepared for classification
        tensor_image = imageprepare('example_images/image.png')

        #importing the saved model
        model = keras.models.load_model('src/saved_model_dense')

        classes = model.predict(tensor_image.reshape(1,28*28))
        print(classes)
        
        #printing the index which has the highest weight
        #which is therefore the model prediction
        print(np.argmax(classes))

        #plotting
        plt.imshow(tensor_image.reshape(28,28),cmap = plt.cm.binary)
        plt.show()



