from src.prepared_dataset.mnist_dataset import test_images
from src.prepared_dataset.mnist_dataset import test_labels
import keras

#loading the saved model
model = keras.models.load_model('src/saved_model_dense')

if __name__ == "__main__":
    test_loss, test_acc = model.evaluate(test_images,
                                          test_labels)
    print('test_acc:',test_acc)
