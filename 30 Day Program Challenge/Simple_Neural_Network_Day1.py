# Marketne Noel
## 30 Day Code Challenge Day 1
## C: Simple Neural Network

# set imports 
import numpy as np


# create class 
class NeuralNetwork():
    def __init__(self):
        np.random.seed(1) # set seed

        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1 # make random weight

    # make sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_div(self, x):
        return x * (1 - x)

    # create assumption 
    def think(self, inputs):
        inputs = inputs.astype(float)
        outputs = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return outputs

    # Train NN 
    def train(self, inputs, outputs, i_max):
        for i in range(i_max):
            out = self.think(inputs)
            error = outputs - out
            adjust = np.dot(inputs.T, error * self.sigmoid_div(out))
            self.synaptic_weights += adjust

if __name__ == "__main__":
    nn = NeuralNetwork()


    # Training model 
    training_input = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])

    # Make Training  Output array 
    training_output = np.array([[0,1,1,0]]).T

    # Train Model 
    nn.train(training_input, training_output, 10000)

    print("Input data (1 or 0)")

    A = str(input("input 1: "))
    B = str(input("input 2: "))
    C = str(input("input 3: "))

    response = nn.think(np.array([A, B, C]))
    print("Prediction: ")
    print(round(response[0]))