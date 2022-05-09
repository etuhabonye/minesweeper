# example of training a final classification model
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from generateminesweeper import *
import util

target = "Safe"
features = {"Xcord", "Ycord", "Value", "Neighbors"}

df = boardTodf()

train_proportion = 0.7
train_df, test_df = util.split_data(df, train_proportion)


X, Y = util.get_X_y_data(train_df, features, target)

# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, Y, epochs=200, verbose=0)

