import tensorflow as tf
import numpy as np
import flwr as fl
import sys
import splitfolders
from tensorflow import keras


# Client-specific data paths
data_path = ".\output\\Part3"  # Change this for each client

IMG_HEIGHT = 128
IMG_WIDTH = 128

splitfolders.ratio('.\OuputDataset\Part3', output="output\Part3", seed=1345, ratio=(.8,0.1,0.1))  
                                                                 
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_path + "\\train",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_path + "/test",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_path + "/val",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64
)

class_names = train_ds.class_names
print(class_names)

model = keras.models.Sequential()
model.add(keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Dropout(0.20))

model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(4, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="Adam", metrics=["accuracy"])

class DnnClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r = model.fit(train_ds, validation_data=val_ds, epochs=10, batch_size=64, verbose=1)
        hist = r.history
        print("Fit history: ", hist)
        return model.get_weights(), len(train_ds), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_ds)
        print("Eval accuracy: ", accuracy)
        return loss, len(test_ds), {"accuracy": accuracy}

fl.client.start_numpy_client(
    server_address="localhost:" + str(sys.argv[1]),
    client=DnnClient(),
    grpc_max_message_length=1024 * 1024 * 1024
)