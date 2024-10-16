import pickle
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

from nn.dense import Dense
from nn.activations import Tanh
from nn.losses import mse, mse_prime
from nn.network import train, predict


def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

def save_model(network, filename='model.pkl'):
    params = []
    for layer in network:
        if isinstance(layer, Dense):
            # Save the weights and biases for each Dense layer
            params.append((layer.weights, layer.bias))

    # Save the parameters to a file
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
    print(f"Model saved to {filename}")

def load_model(network, filename='model.pkl'):
    # Load the parameters from the file
    with open(filename, 'rb') as f:
        params = pickle.load(f)

    # Assign the loaded parameters to the respective layers
    param_idx = 0
    for layer in network:
        if isinstance(layer, Dense):
            layer.weights, layer.bias = params[param_idx]
            param_idx += 1
    print(f"Model loaded from {filename}")



# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 20)

# neural network
network = [
    Dense(28 * 28, 40),
    Tanh(),
    Dense(40, 10),
    Tanh()
]

# train
train(network, mse, mse_prime, x_train, y_train, epochs=100, learning_rate=0.1)

save_model(network, 'model.pkl')

# Load the model (e.g., in a new session or after retraining)
load_model(network, 'model.pkl')

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))
