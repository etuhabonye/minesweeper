import numpy as np
import pandas as pd
from minesweeper import MineSweeper

np.set_printoptions(linewidth = 1000, precision = 3, suppress = True)

def generateMineSweeper():
    game = MineSweeper()
    coordinates = (3, 3)
    game.selectCell(coordinates)

    board = game.state
    mines = game.mines
    neighbors = game.neighbors

    print(board)

    return board, mines, neighbors

def boardTodf():
    board, mines, neighbors = generateMineSweeper()
    coordList = []
    for i in range(len(board[0])):
        for j in range(len(board[0])):
            coordList.append((i, j))

    rows = [ [] for k in range(len(coordList))]
    curr = coordList[7]

    # print(coordList)
    for l in range(len(coordList)):
        curr = coordList[l]

        rows[l] = [coordList[l], board[curr[0]][curr[1]], neighbors[curr[0]][curr[1]], mines[curr[0]][curr[1]]]

    df = pd.DataFrame(rows, columns = ['Coordinates', 'Value', 'Neighbors', 'Safe'])

    return df

print(boardTodf())
