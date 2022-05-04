# Python modules
import numpy as np
import pandas as pd
import random
import sys
import time


# Project modules
import util
from generateminesweeper import *
class Game:
    def guess(self, position):
        pass 
    def is_won(self):
        pass 
    def view(self):
        pass 

class Player:
    #this implements the game player
    def __init__(self, h, w):
        self.height = h
        self.width = w
        self.mines = mines 
        self.data = []

    def play(self):
        #right now this play with just one round 
        won = 0
        g = Game(self.height, self.width, self.mines)
        won += self.playGame(g)

    def playGame(self, game):
        #plays the game and returns bool.
        #T if game is won and F if anything else
        mine = False
        while not mine:
            mine = self.playMove(game, debug)
            self.data.append((game.view(), game.mines))
            if game.is_won():
                return True 
        return False

    def playMove(self,game):
        view = game.view()
        pred = self.predict_mines(view)
        pos = np.unravel_index(np.argmin(pred), (self.height, self.width))
        if not game.guessed:
            #if the first step has not been taken the guess in the middle of the board (5X5 board)
            pos = (3,3)
        return game.guess(pos)

    def predict_mines(self, view):
        game_input = self.get_model_input(view)
        pred = predict(game_input)[0]
        pred[view.flatten()!=9]=1
        return pred


class ModelBatchResults:
    '''The result of training the model on one batch. '''
    def __init__(self, loss, precision, recall, accuracy):
        pass 













class NeuralNetwork:
    def __init__(self, num_input = 175, num_hidden = 140, num_output = 25, learning_rate=0.01, num_iterations=1000):
        pass

    def init_parameters(self):
        ''' Initialize weights and biases for neural network.
            Weights are initialized randomly, biases are set to 0
            Return: dictionary containing weights and biases
        '''
        # Use numpy arrays to create
        W1 = np.random.randn(self.num_hidden, self.num_input) * 0.01  # Size num_hidden x num_input
        b1 = np.zeros((self.num_hidden, 1))                           # Size num_hidden x 1
        W2 = np.random.randn(self.num_output, self.num_hidden) * 0.01 # Size num_output x num_hidden
        b2 = np.zeros((self.num_output, 1))                           # Size num_output x 1

        # Create dictionary of parameters
        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return parameters


    def forward_pass(self, X):
        ''' Compute output of neural network
            Return: output computed, intermediate steps
        '''
        Z1 = np.dot(self.parameters["W1"], X) + self.parameters["b1"]
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.parameters["W2"], A1) + self.parameters["b2"]
        A2 = util.sigmoid(Z2)

        # Create dictionary to cache intermediate steps as well as output
        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def back_prop(self, X, Y_true):
        """
        Compute the output of the neural network
        Retrurn: output computed, intermediate steps
        """

        #calling forward pass
        Y_pred, cache = self.forward_pass(X)


        step1 = Y_pred - Y_true
        step2 = Y_pred - Y_true * cache["A1"]
        step3 = np.matmul(self.parameters["W2"], (Y_pred - Y_true) *(1 - np.square(np.tanh(np.dot(self.parameters["W1"], X)))))
        step4 = np.matmul(self.parameters["W2"],(Y_pred - Y_true) * (1 - np.square(np.tanh(np.dot(self.parameters["W1"], X)))))

        avg1 = np.sum(step1) / Y_true.shape[0]
        avg2 = np.sum(step2) / Y_true.shape[0]
        avg3 = np.sum(step3) / Y_true.shape[0]
        avg4 = np.sum(step4) / Y_true.shape[0]

        return avg1, avg2, avg3, avg4

    def update_weights(self, X, y):
        for i in range(self.num_iterations):
            #random.shuffle(X)
            avg1, avg2, avg3, avg4 = self.back_prop(X, y)
            for row in range(len(X)):

                self.parameters["W1"] -= avg4 * self.learning_rate
                self.parameters["W2"] -= avg2 * self.learning_rate
                self.parameters["b1"] -= avg3 * self.learning_rate
                self.parameters["b2"] -= avg1 * self.learning_rate
        return self.parameters

    def predict(self, X, Y):
        Y_pred, cache = self.forward_pass(X)
        loss = self.get_loss(Y, Y_pred)
        Y_pred_rounded = np.round(Y_pred)
        accuracy = self.get_accuracy(Y, np.squeeze(Y_pred_rounded))
        return Y_pred, loss, accuracy

    def get_loss(self, Y_true, Y_pred):
        ''' Compute cross-entropy loss function over all examples '''
        num_examples = Y_true.shape[0]
        logprobs = np.multiply(np.log(Y_pred+1e-15), Y_true) + \
                   np.multiply(np.log(1 - Y_pred+1e-15), 1 - Y_true)
        loss = - np.sum(logprobs) / num_examples
        return loss

    def get_accuracy(self, Y_true, Y_pred):
        return 1 - sum(abs(Y_true - Y_pred)) / len(Y_true)

def cross_validation(df, num_folds, features, target, learning_rate, num_iterations, num_input, num_hidden, num_output):
    ''' Perform cross-validataion using linear regression

        Inputs:
            * df: dataframe containing data
            * num_folds: number of folds
            * features: set of feature names to use corresponding to columns in df
            * target: name of target column (i.e., feature to predict) in df
            * learning_rate: to use in linear regression
            * num_iterations: to use in linear regression
        Outputs:
            * Per fold root mean square error predicted target and true target,
    '''
    validation_loss = []
    accuracies = []
    folds = util.divide_k_folds(df, num_folds)


    for i in range(num_folds):
        validation = folds[i]
        train = pd.concat(folds[:i] + folds[i+1:])

        train_X, train_Y = util.get_X_y_data(train, features, target)
        nn = NeuralNetwork(num_input, num_hidden, num_output, learning_rate, num_iterations)
        params = nn.update_weights(train_X,train_Y)

        Y_pred, loss, accuracy = nn.predict(train_X, train_Y)

        validation_loss.append(loss)
        accuracies.append(accuracy)


    return sum(validation_loss) / num_folds, sum(accuracies)/ num_folds

def main():
    import warnings
    warnings.filterwarnings("ignore", message = r"Passing", category = FutureWarning)
    warnings.filterwarnings("ignore")

    random.seed(time.time())

    MAX_ERROR = sys.maxsize * 2 + 1 # Maximum RMSE
    num_folds = 10 # Number of folds to use in cross-validation

    # Load data and normalize
    target = "Safe"
    features = {"Coordinates", "Value", "Neighbors"}
   
    df = boardTodf()


    num_examples = df.shape[0]
    num_features = df.shape[1]

    # Normalize features by max-min difference
    f = df[features]
    print(f)
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
