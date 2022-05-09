import numpy as np
import pandas as pd
from minesweeper import MineSweeper
import generateminesweeper
import util
import math

df = generateminesweeper.boardTodf()
df['Probability'] = 1



# features = {"Coordinates", "Value", "Neighbors"}
# target = "Safe"
# X, y = util.get_X_y_data(df, features, target)

for i in range(len(df["Xcord"])):
    coord = df["Coordinates"][i]
    if df["Value"][i]< 0:
        neighbors = util.get_neighbors(coord)
        n = len(df.loc[(df["Coordinates"].isin(neighbors)) & (df['Value'].isin([0, 1, 2, 3]) == False)])
        if n != 0:
            prob = 100 / n
            df.loc[(df["Coordinates"].isin(neighbors)) & (df['Value'].isin([0, 1, 2, 3]) == False), "Probability"] = (df.loc[(df["Coordinates"].isin(neighbors)) & (df['Value'].isin([0, 1, 2, 3]) == False), "Probability"] * (1 / n)) / (df.loc[(df["Coordinates"].isin(neighbors)) & (df['Value'].isin([0, 1, 2, 3]) == False), "Probability"])

df["Probability"] = (1 - df["Probability"]) * 100

print(df)
# Our next steps are:
# 1. Have the algorithm make the "best move" based off the probability it calculated
# 2. Get the updated board and recalculate the probabilities, repeat until board is "cleared"
# 3. If a "bad move" is made restart...?