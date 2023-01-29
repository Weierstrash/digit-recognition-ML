import keras
import matplotlib.pyplot as plt
from prepare_image import imageprepare
import numpy as np


#importing the saved model
model = keras.models.load_model('src/saved_model')

if __name__ == "__main__":
    #image prepared for classification
    tensor_image = imageprepare('image.png')

    classes = model.predict(tensor_image.reshape(1,28*28))
    print(classes)
    
    #printing the index which has the highest weight
    #which is therefore the model prediction
    print(np.argmax(classes))

    #plotting
    plt.imshow(tensor_image.reshape(28,28),cmap = plt.cm.binary)
    plt.show()
    


