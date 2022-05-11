import numpy as np
import pandas as pd
from minesweeper import MineSweeper
from random import randrange

np.set_printoptions(linewidth = 1000, precision = 3, suppress = True)

def generatedf(board, mines, neighbors):
    coordList = []
    for i in range(len(board[0])):
        for j in range(len(board[0])):
            coordList.append((i, j))

    rows = [ [] for k in range(len(coordList))]
    curr = coordList[7]

    # print(coordList)
    for l in range(len(coordList)):
        curr = coordList[l]

        rows[l] = [coordList[l], coordList[l][0], coordList[l][1], board[curr[0]][curr[1]], neighbors[curr[0]][curr[1]], mines[curr[0]][curr[1]]]

    df = pd.DataFrame(rows, columns = ['Coordinates', 'Xcord', 'Ycord', 'Value', 'Neighbors', 'Safe'])

    return df

def boardTodf():
    game = MineSweeper()
    x = randrange(game.X)
    y = randrange(game.Y)
    coordinates = (x, y)
    game.selectCell(coordinates)

    board = game.state
    mines = game.mines
    neighbors = game.neighbors

    df1 = generatedf(board, mines, neighbors)

    safe = False
    while not safe:
        x = randrange(game.X)
        y = randrange(game.Y)
        if mines[x][y] == 0:
            safe = True

    coordinates = (x, y)
    game.selectCell(coordinates)

    board = game.state
    mines = game.mines
    neighbors = game.neighbors

    df2 = generatedf(board, mines, neighbors)

    return df1, df2


boardTodf()
