from src.CNN_network.digit_recognition_cnn import model
from src.prepared_dataset.mnist_dataset_for_cnn import train_images
from src.prepared_dataset.mnist_dataset_for_cnn import train_labels

#compiling the model
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


if __name__ == "__main__":
    #fitting the model
    model.fit(train_images , 
            train_labels, 
            epochs = 5,
            batch_size = 64)

    model.save('src/saved_model_cnn')




    

