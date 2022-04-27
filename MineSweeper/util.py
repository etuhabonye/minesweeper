'''
Wesleyan University, COMP 343, Spring 2022
Homework 8: Neural Network Training
Name: Your name here!

Important: If you modify this file, you should submit to your homework directory

'''


import numpy as np
import pandas as pd


########################## Data helper functions #############################
def load_data(filename):
    ''' Returns a dataframe (df) containing the data in filename. You should
        specify the full path plus the name of the file in filename, relative
        to where you are running code
    '''
    df = pd.read_csv(filename)
    return df

def sigmoid(z):
    ''' Sigmoid function '''
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    ''' Derivative of sigmoid function '''
    sig_z = sigmoid(z)
    return sig_z * (1-sig_z)

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

def divide_k_folds(df, num_folds):
    ''' Inputs
            * df: dataframe containing data
            * num_folds: number of folds
        Output
            * folds: lists of folds, each fold is subset of df dataframe
    '''
    folds = []
    for subset in np.array_split(df, num_folds):
        folds.append(subset)

    return folds

def to_numpy(df):
    a = df.to_numpy()
    return a.T

def get_X_y_data(df, features, target):
    ''' Split dataframe into X and y numpy arrays '''
    X_df = df[features]
    Y_df = df[target]
    X = to_numpy(X_df)
    Y = to_numpy(Y_df)
    return X, Y

def get_rmse(vec_true, vec_pred):
    ''' Compute root mean square error between two numpy arrays '''
    rmse = np.sqrt(np.mean(np.subtract(vec_pred,vec_true)**2))
    return rmse

def validateNeighbords(neighbors):
    """
    input: neighbors, list of neighbors
    returns: list of valid neighbors
    """
    for pair in neighbors:
        if (pair[0] == -1) or (pair[0] == 5) or (pair[1] == -1) or (pair[1] == 5):
            neighbors.remove(pair)
    return neighbors

def get_neighbors(coord):
    """
    input: coord, tuple of coordinates
    output: neighbors, list of valid neighbors
    """
    x = coord[0]
    y = coord[1]
    neighbors = []

    neighbors.append((x - 1, y - 1))
    neighbors.append((x - 1, y + 1))
    neighbors.append((x + 1, y - 1))
    neighbors.append((x + 1, y + 1))
    neighbors.append((x, y - 1))
    neighbors.append((x, y + 1))
    neighbors.append((x - 1, y))
    neighbors.append((x + 1, y))

    return(validateNeighbords(neighbors))
