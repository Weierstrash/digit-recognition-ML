from src.network.digit_recognition_network import test_images
from src.network.digit_recognition_network import test_labels
import keras

#loading the saved model
model = keras.models.load_model('src/saved_model')

if __name__ == "__main__":
    test_loss, test_acc = model.evaluate(test_images,
                                          test_labels)
    print('test_acc:',test_acc)
