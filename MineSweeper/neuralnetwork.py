# Python modules
import numpy as np
import pandas as pd
import random
import sys
import time


# # Project modules
import util
from generateminesweeper import *


# Python modules
import numpy as np
import pandas as pd
import random
import sys
import time

# Project modules
import util

# class NeuralNetwork:
#     def __init__(self, num_input= 4, num_hidden =18, num_output=1, learning_rate=0.01, num_iterations=10000):
#         ''' Init neural network parameters
#                * num_input: number of input nodes
#                * num_hidden: number of hidden nodes
#                * num_output: number of output nodes
#                * learning_rate: gradient descent learning rate
#                * num_iterations: iterations for gradient descent

#                  Neural network with one hidden layer

#                               hid1
#                       W1, b1/      \  W2, b2
#                           /          \
#                      X ------ hid2 --- Y_pred
#                           \          /
#                             \      /
#                               hid3

#                          Z1 = W1 dot X + b1
#                          A1 = tanh(Z1)
#                          Z2 = W2 dot A1 + b2
#                  Y_est = A2 = sigmoid(Z2)
#         '''
#         self.num_input = num_input
#         self.num_hidden = num_hidden
#         self.num_output = num_output
#         self.learning_rate = learning_rate
#         self.num_iterations = num_iterations
#         self.parameters = self.init_parameters()

#     def init_parameters(self):
#         ''' Initialize weights and biases for neural network.
#             Weights are initialized randomly, biases are set to 0
#             Return: dictionary containing weights and biases
#         '''
#         # Use numpy arrays to create
#         W1 = np.random.randn(self.num_hidden, self.num_input) * 0.01  # Size num_hidden x num_input
#         b1 = np.zeros((self.num_hidden, 1))                           # Size num_hidden x 1
#         W2 = np.random.randn(self.num_output, self.num_hidden) * 0.01 # Size num_output x num_hidden
#         b2 = np.zeros((self.num_output, 1))                           # Size num_output x 1

#         # Create dictionary of parameters
#         parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
#         return parameters

#     def update_parameters(self, gradients):
#         ''' Update parameters using gradient descent rule '''
#         self.parameters["W1"] = self.parameters["W1"] - self.learning_rate * gradients["dW1"]
#         self.parameters["b1"] = self.parameters["b1"] - self.learning_rate * gradients["db1"]
#         self.parameters["W2"] = self.parameters["W2"] - self.learning_rate * gradients["dW2"]
#         self.parameters["b2"] = self.parameters["b2"] - self.learning_rate * gradients["db2"]

#     def get_loss(self, Y_true, Y_pred):
#         ''' Compute cross-entropy loss function over all examples '''
#         num_examples = Y_true.shape[0]
#         logprobs = np.multiply(np.log(Y_pred), Y_true) + \
#                    np.multiply(np.log(1 - Y_pred), 1 - Y_true)
#         loss = - np.sum(logprobs) / num_examples
#         return loss

#     def get_accuracy(self, Y_true, Y_pred):
#         return 1 - sum(abs(Y_true - Y_pred)) / len(Y_true)

#     def forward_pass(self, X):
#         ''' Compute output of neural network
#             Return: output computed, intermediate steps
#         '''
#         Z1 = np.dot(self.parameters["W1"], X) + self.parameters["b1"]
#         A1 = np.tanh(Z1)
#         Z2 = np.dot(self.parameters["W2"], A1) + self.parameters["b2"]
#         A2 = util.sigmoid(Z2)

#         # Create dictionary to cache intermediate steps as well as output
#         cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
#         return A2, cache

#     def backward_pass(self, cache, X, Y):
#         ''' Backpropagation algorithm to update neural network weights
#             Sum over number of examples and divide by number
#         '''
#         num_examples = X.shape[1]

#         # Compute gradients for layer 2
#         dZ2 = cache["A2"] - Y # diff for each example, vector of length num_examples
#         dW2 = np.dot(dZ2, cache["A1"].T) / num_examples
#         db2 = np.sum(dZ2, axis=1, keepdims=True) / num_examples

#         # Compute gradients for layer 1
#         dZ1 = np.dot(self.parameters["W2"].T, dZ2) * (1 - np.power(cache["A1"], 2))
#         dW1 = np.dot(dZ1, X.T) / num_examples
#         db1 = np.sum(dZ1, axis=1, keepdims=True) / num_examples


#         # Create dictionary of gradients
#         gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
#         return gradients

#     def train(self, X, Y):
#         ''' Train neural network given data '''
#         accuracy = None
#         loss = None
#         for i in range(0, self.num_iterations):
#             Y_pred, cache = self.forward_pass(X)
#             loss = self.get_loss(Y, Y_pred)
#             accuracy = self.get_accuracy(Y, np.squeeze(np.round(Y_pred)))
#             gradients = self.backward_pass(cache, X, Y)
#             self.update_parameters(gradients)
#             # print("Iteration %i: loss is %f, accuracy is %f" % (i, loss, accuracy))
#         return loss, accuracy

#     def predict(self, X, Y):
#         Y_pred, cache = self.forward_pass(X)
#         loss = self.get_loss(Y, Y_pred)
#         accuracy = self.get_accuracy(Y, np.squeeze(np.round(Y_pred)))
#         return Y_pred, loss, accuracy

# def cross_validation(df, num_folds, features, target, num_input, num_hidden, num_output, learning_rate, num_iterations):
#     ''' Perform cross-validataion using neural network

#         Inputs:
#             * df: dataframe containing data
#             * num_folds: number of folds
#             * features: set of feature names to use corresponding to columns in df
#             * target: name of target column (i.e., feature to predict) in df
#             * num_input: number of input nodes in neural network
#             * num_hidden: number of hidden nodes in neural network
#             * num_output: number of output nodes in neural entwork
#             * learning_rate: to use in linear regression
#             * num_iterations: to use in linear regression
#         Outputs:
#             * Per fold root mean square error predicted target and true target,
#     '''
#     validation_accuracies = []
#     validation_losses = []
#     folds = util.divide_k_folds(df, num_folds)
#     for i in range(num_folds):

#         # Train and get error
#         train = pd.concat(folds[:i] + folds[i+1:])
#         X, Y = util.get_X_y_data(train, features, target)
#         nn = NeuralNetwork(num_input, num_hidden, num_output, learning_rate, num_iterations)
#         loss, accuracy = nn.train(X, Y)

#         # Validation
#         validation = folds[i]
#         X, Y = util.get_X_y_data(validation, features, target)
#         Y_pred, loss, accuracy = nn.predict(X, Y)

#         # Save metrics
#         validation_accuracies.append(accuracy)
#         validation_losses.append(loss)

#     return sum(validation_losses) / num_folds, sum(validation_accuracies) / num_folds


# def get_best_hyperparameters(df, features, target, num_input, num_hidden, num_output, num_folds = 5):
#     ''' Inputs:
#             * df: dataframe containing data
#             * features: set of feature names to use corresponding to columns in df
#             * target: name of target column (i.e., feature to predict) in df
#         Outputs:
#             * best_r: best learning rate to use
#             * best_i: best number of iterations to use
#     '''

#     best_loss = sys.maxsize * 2 + 1
#     best_r = None
#     best_i = None

#     iterations = [1000, 5000, 10000, 20000]
#     learning_rates = [0.001, 0.01, 0.1]

#     for r in learning_rates:
#         for i in iterations:

#             loss, accuracy = cross_validation(
#                     df, num_folds, features, target, num_input, num_hidden, num_output, r, i)

#             print('learning_rate:', r, ', \t num_iterations: ', i, ' \t loss:', loss, ', \t accuracy:', accuracy)

#             if loss < best_loss:
#                 best_loss = loss
#                 best_r = r
#                 best_i = i

#     return best_r, best_i





def main():
    import warnings
    warnings.filterwarnings("ignore", message = r"Passing", category = FutureWarning)
    warnings.filterwarnings("ignore")

    random.seed(time.time())

    MAX_ERROR = sys.maxsize * 2 + 1 # Maximum RMSE
    num_folds = 10 # Number of folds to use in cross-validation

    # Load data and normalize
    target = "Safe"
    features = {"Xcord", "Ycord", "Value", "Neighbors"}
   
    df = boardTodf()


    num_examples = df.shape[0]
    num_features = df.shape[1]

   # Normalize features by max-min difference
    f = df[features]
   # print(f)
    f = (f - f.min()) / (f.max() - f.min())
    df.update(f)
    print(df)

    # Split data into training and test
    train_proportion = 0.7
    train_df, test_df = util.split_data(df, train_proportion)

    # Set size of neural network layers, input and output should match data
    num_input = num_features -1
    num_hidden = 7
    num_output = 1
    nn = NeuralNetwork(num_input, num_hidden, num_output)
    nn.__init__()

    # Training results
    X, Y = util.get_X_y_data(train_df, features, target)
    Y_pred, loss, accuracy = nn.predict(X, Y)
    print("Training: loss", loss, "accuracy", accuracy)

    # Testing results
    X, Y = util.get_X_y_data(test_df, features, target)
    Y_pred, loss, accuracy = nn.predict(X, Y)
    print("Testing: loss", loss, "accuracy", accuracy)


    print("#### Find best hyperparameter settings ##############################")

    best_lr, best_r = get_best_hyperparameters(train_df, features, target, num_folds, num_input, num_hidden, num_output)
    print("best learning rate is: ", best_lr, "best num iterations is: ", best_r, "\n")



if __name__ == '__main__':
    main()

