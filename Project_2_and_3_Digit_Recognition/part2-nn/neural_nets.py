import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    if x > 0:
        return x
    else: 
        return 0
    

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    if x > 0:
        return 1
    else: 
        return 0

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS
        self.input_to_hidden_weights = np.matrix('1 1; 1 1; 1 1')
        self.hidden_to_output_weights = np.matrix('1 1 1')
        self.biases = np.matrix('0; 0; 0')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]

    def train(self, x1, x2, y):

        ReLU = np.vectorize(rectified_linear_unit)
        dReLU = np.vectorize(rectified_linear_unit_derivative)


        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = np.matmul(self.input_to_hidden_weights, input_values) + self.biases # (3 by 1 matrix)
        # print()
        # print("Input-Hidden Weights: " + str(self.input_to_hidden_weights))
        # print("Input Walues: " + str(input_values))
        # print("Biases: " + str(self.biases))
        # print('hidden_layer_weighted_input: ' + str(hidden_layer_weighted_input))
        hidden_layer_activation = ReLU(hidden_layer_weighted_input)  # (3 by 1 matrix)
        #print("Hidden Layer Acrivation: " + str(hidden_layer_activation))

        #print(np.matmul(self.hidden_to_output_weights, hidden_layer_activation))
        #print(self.hidden_to_output_weights)
        #print(hidden_layer_activation)
        output =  np.dot(self.hidden_to_output_weights, hidden_layer_activation).item()
        # print("Output Weights: " + str(self.hidden_to_output_weights))
        # print("Correct Answer: " + str(y))
        # print("Output: " + str(output))
        # print("Error: " + str((1/2) * ((y - output) ** 2)))
        activated_output = output_layer_activation(output)

        ### Backpropagation ###

        # Compute gradients
        output_layer_error = activated_output - y # derivative of loss
        #print(y)
        #print(activated_output)
        #print(output_layer_error)

        print("Output Error: " + str(output_layer_error))
        print("Hidden-Output Weights: " + str(self.hidden_to_output_weights))
        print("dReLU: " + str(dReLU(hidden_layer_weighted_input)))
        hidden_layer_error = output_layer_error * np.multiply(dReLU(hidden_layer_weighted_input), np.transpose(self.hidden_to_output_weights)) # (3 by 1 matrix)

        bias_gradients = hidden_layer_error
        hidden_to_output_weight_gradients = output_layer_error * hidden_layer_activation
        print("Input: " + str(input_values))
        print("Zi: " + str(hidden_layer_error))
        input_to_hidden_weight_gradients = np.outer(hidden_layer_error, input_values) # (3 neurons by 2 weights matrix)
        print("Input-Hidden Weights Gradients: " + str(input_to_hidden_weight_gradients))

        # Use gradients to adjust weights and biases using gradient descent
        self.biases = self.biases - self.learning_rate * bias_gradients
        #print("Biases: " + str(self.biases))
        self.input_to_hidden_weights = self.input_to_hidden_weights - self.learning_rate * input_to_hidden_weight_gradients
        #print("Input-Hidden Weights: " + str(self.input_to_hidden_weights))

        #print("Before changing: " + str(self.hidden_to_output_weights))
        #print("Changing rate: " + str(self.learning_rate))
        #print("Gradients: " + str(hidden_to_output_weight_gradients))
        self.hidden_to_output_weights = self.hidden_to_output_weights - (self.learning_rate * hidden_to_output_weight_gradients).reshape(self.hidden_to_output_weights.shape)
        #print("After change: " + str(self.hidden_to_output_weights))


    def predict(self, x1, x2):

        ReLU = np.vectorize(rectified_linear_unit)
        input_values = np.matrix([[x1],[x2]])

        # Compute output for a single input(should be same as the forward propagation in training)
        hidden_layer_weighted_input = np.matmul(self.input_to_hidden_weights, input_values) # (3 by 1 matrix)
        hidden_layer_activation = ReLU(hidden_layer_weighted_input)  # (3 by 1 matrix)
        output = np.matmul(self.hidden_to_output_weights, hidden_layer_activation)
        activated_output = output_layer_activation(output)

        return activated_output.item()

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            print("#######################  EPOCH " + str(epoch + 1) + "  ##########################")
            for x,y in self.training_points:
                self.train(x[0], x[1], y)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
        return
    
    def print_errors_on_training_set(self):

        for point in self.training_points:
            print("Point,", point[0], "Error,", (1/2) * (point[1] - self.predict(point[0][0], point[0][1])) ** 2)
            if abs(self.predict(point[0][0], point[0][1]) - 7*point[0][0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0][0], point[0][1], " failed to be predicted correctly.")
        return

x = NeuralNetwork()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
x.test_neural_network()
