import numpy as np
import pandas as pd

class MineSweeper:
    def __init__(self):
        # params
        self.X = 5
        self.Y = 5
        self.totalCells = self.X * self.Y
        self.nMines = 5
        self.mines = np.zeros([self.X, self.Y])
        self.neighbors = np.zeros([self.X, self.Y])
        self.state = np.zeros([self.X, self.Y])
        self.state.fill(-1)
        self.initialized = False
        self.gameOver = False
        self.victory = False

    def initialize(self, coordinates):
        availableCells = range(self.totalCells)
        selected = coordinates[0]*self.Y + coordinates[1]
        offLimits = np.array([selected-self.Y-1, selected-self.Y, selected-self.Y+1, selected-1, selected, selected+1, selected+self.Y-1, selected+self.Y, selected+self.Y+1])
        availableCells = np.setdiff1d(availableCells, offLimits)
        self.nMines = np.minimum(self.nMines, len(availableCells)) #in case there are fewer remaining cells than mines to place
        minesFlattened = np.zeros([self.totalCells])
        minesFlattened[np.random.choice(availableCells, self.nMines, replace=False)] = 1
        self.mines = minesFlattened.reshape([self.X, self.Y])
        #set up neighbors
        for i in range(self.X):
            for j in range(self.Y):
                nNeighbors = 0
                for k in range(-1, 2):
                    if i + k >= 0 and i + k < self.X:
                        for l in range(-1, 2):
                            if j + l >= 0 and j + l < self.Y and (k != 0 or l != 0):
                                nNeighbors += self.mines[i + k, j + l]
                self.neighbors[i, j] = nNeighbors

        self.initialized = True

    def clearEmptyCell(self, coordinates):
        x = coordinates[0]
        y = coordinates[1]
        self.state[x, y] = self.neighbors[x, y]
        if self.state[x, y] == 0:
            for i in range(-1, 2):
                if x + i >= 0 and x + i < self.X:
                    for j in range(-1, 2):
                        if y + j >= 0 and y + j < self.Y:
                            if (self.state[x + i, y + j]) < 0:
                                self.clearEmptyCell((x + i, y + j))
    def selectCell(self, coordinates):
        self.initialize(coordinates)
        self.clearEmptyCell(coordinates)
