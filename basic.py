import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import keras.api._v2.keras as keras
from keras import layers
import flwr as fl
import numpy as np

# Set the path to the directory containing the HAM10000 dataset
data_dir = 'D:\dataset1'

# Load the metadata
metadata = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))

# Load the image data
image_data = pd.read_csv(os.path.join(data_dir, 'hmnist_28_28_RGB.csv'))

# Extract the image and label data
x = image_data.drop('label', axis=1).values
x = x.reshape((-1, 28, 28, 3))
y = image_data['label'].values

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
# Number of clients
num_clients = 2

# Split the training data into partitions
x_train_partitions = np.array_split(x_train, num_clients)
y_train_partitions = np.array_split(y_train, num_clients)

# Get the partition index from the command line arguments
partition_index = int(sys.argv[1])

# Select the partition for this client
x_train = x_train_partitions[partition_index]
y_train = y_train_partitions[partition_index]

# Preprocess the data
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Build the model
model = keras.Sequential([
    layers.InputLayer(input_shape=(28, 28, 3)),
    layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dense(7, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
#model.fit(x_train, y_train, epochs=2)

# Evaluate the model
#model.evaluate(x_test, y_test)

class SkinCancerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}
    
# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=SkinCancerClient())