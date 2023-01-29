import keras
from src.prepared_dataset.mnist_dataset_for_cnn import test_images
from src.prepared_dataset.mnist_dataset_for_cnn import test_labels

#loading the model
model = keras.models.load_model("src/saved_model_cnn")

if __name__ == "__main__":
    test_loss,test_acc = model.evaluate(test_images,test_labels)
    print(test_acc)



