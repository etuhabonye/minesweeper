import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
# warnings.filterwarnings("ignore", message=r"Maximum number of iteration", category=FutureWarning)
from minesweeper import MineSweeper
from generateminesweeper import *
import util


best_num_iterations = 100
def split_data(df, train_proportion):
    ''' Inputs
            * df: dataframe containing data
            * train_proportion: proportion of data in df that will be used for
                training. 1-train_proportion is proportion of data to be used
                for testing
        Output
            * train_df: dataframe containing training data
            * test_df: dataframe containing testing data
    '''
    # Make sure there are row numbers
    df = df.reset_index(drop=True)

    # Reorder examples and split data according to train proportion
    train = df.sample(frac=train_proportion, axis=0)
    test = df.drop(index=train.index)
    return train, test

def get_X_y_data(df, features, label):
    X = np.array([np.array(x) for _, x in df[features].iterrows()])
    y = np.array(df[label])
    return X, y

def divide_k_folds(df, k):
    ''' Inputs
            * df: dataframe containing data
            * k: number of folds
        Output
            * folds: lists of folds, each fold is subset of df dataframe
    '''
    folds = []
    for subset in np.array_split(df, num_folds):
        folds.append(subset)
    return folds

def get_scikit_perceptron_accuracy(clf, X, y):
    accuracy = clf.score(X, y)
    return accuracy

def scikit_perceptron(X, y, num_iterations):

    ####
    # Todo: return trained scikit perceptron classifier
    ####
    clf = Perceptron(max_iter = num_iterations)
    clf.fit(X, y)

    return clf


def scikit_perceptron_cross_validation(df, num_folds, features, label, num_iterations):
    # Do cross-validation
    validation_accuracies = 0

    folds = divide_k_folds(df, num_folds)
    acc_total = 0

    for i in range(0, num_folds):
        validation_data = folds[i]
        train_list = folds[:i] + folds[i + 1:]
        train_data = pd.concat(train_list)
        X, y = get_X_y_data(train_data, features, label)

        clf = scikit_perceptron(X, y, num_iterations)
        accuracy = get_scikit_perceptron_accuracy(clf, X, y)
        validation_accuracies += accuracy

    ####
    # Todo: implement cross-validation for scikit perceptron
    ####

    return validation_accuracies / num_folds

learning_rate = 1
num_iterations = [1, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
# num_iterations = [1, 10, 100]
num_folds = 5

label = "Safe"
features = {"Xcord", "Ycord", "Value", "Neighbors"}

df = boardTodf()

# Add bias feature to data
df['bias'] = 1
features.add('bias')

# Convert 0 labels to -1
# df['class'] = df['class'].apply(lambda x: 1 if x > 0 else -1)

# Split data into training and test
train_proportion = 0.70
train_df, test_df = split_data(df, train_proportion)

for iter in num_iterations:
    X, y = get_X_y_data(train_df, features, label)
    clf = scikit_perceptron(X, y, iter)
    train_accuracy = get_scikit_perceptron_accuracy(clf, X, y)
    X, y = get_X_y_data(test_df, features, label)
    test_accuracy = get_scikit_perceptron_accuracy(clf, X, y)
    validation_accuracy = scikit_perceptron_cross_validation(train_df, num_folds, features, label, iter)
    print('num_iterations:', iter, ', validation accuracy:', validation_accuracy,'train accuracy:', train_accuracy, ', test accuracy:', test_accuracy)

# Accuracy on training and testing data
X, y = get_X_y_data(train_df, features, label)
clf = scikit_perceptron(X, y, best_num_iterations)
train_accuracy = get_scikit_perceptron_accuracy(clf, X, y)
X, y = get_X_y_data(test_df, features, label)
test_accuracy = get_scikit_perceptron_accuracy(clf, X, y)
print('train accuracy:', train_accuracy, ', test accuracy:', test_accuracy)
