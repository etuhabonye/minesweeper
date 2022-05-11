import pygame
import math
import numpy as np
import random
import time
import mss

class Minesweeper(object):
    def __init__(self, rows = 8, cols = 10, sizeofsq = 100, mines = 10, display = True):
        """ Initialize Minesweeper
            Rows, Cols: int  - Number of rows and cols on the board
            SIZEOFSQ: pixels -  Determines the size of the window, reduce to get smaller window
            Mines: integer - Number of mines generated on the board
            display: bool - chooses weather to display the game with pygame
        """
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.display = display

        self.grid = np.zeros((self.rows, self.cols), dtype=object)
        self.state = np.zeros((self.rows, self.cols), dtype=object)
        self.state_last = np.copy(self.state) #saving the state that happened beforehand

        self.won = 0
        self.lost = 0

        #to display the game in PyGame display = True
        if display:
             #Scale to resolutions
            with mss.mss() as sct:
                img = np.array(sct.grab(sct.monitors[1]))
                self.sizeofsq = int(sizeofsq * img.shape[1] / 3840)
                sizeofsq = self.sizeofsq

            pygame.init()
            pygame.font.init()

            self.myfont = pygame.font.SysFont("monospace-bold", sizeofsq)
            self.screen = pygame.display.set_mode((cols * sizeofsq, rows * sizeofsq))

        self.initGame()

    def initGame(self):
            if self.display:
                self.initPattern()
                pygame.display.set_caption('Minesweeper |            Won: {}   Lost: {}'.format(self.won, self.lost))
            self.grid = self.initBoard(startcol = 5, startrow = 5)
            self.state = np.ones((self.rows, self.cols), dtype=object) * 'U'
            self.state_last = np.copy(self.state)


            # self.action(5,5) #Hack alert, to start off with non empty board. Can be removed but then agent has to learn
                             #what to do when the board starts out empty.
            print(self.mines)

    def initBoard(self, startcol, startrow):
            """ Initializes the board """

            cols = self.cols
            rows = self.rows
            grid = np.zeros((self.rows, self.cols), dtype=object)
            mines = self.mines

            #Randomly place bombs
            while mines > 0:
                (row,col) = (random.randint(0, rows-1), random.randint(0, cols-1))
                #if (col,row) not in findNeighbors(startcol, startrow, grid) and grid[col][row] != 'B' and (col, row) not in (startcol, startrow):
                if (row,col) not in self.findNeighbors(startrow, startcol) and (row,col) != (startrow, startcol) and grid[row][col] != 'B':
                    grid[row][col] = 'B'
                    mines = mines - 1


            #Get rest of board when bombs have been placed
            for col in range(cols):
                for row in range(rows):
                    if grid[row][col] != 'B':
                        totMines = self.sumMines(col, row, grid)
                        if totMines > 0:
                            grid[row][col] = totMines
                        else:
                            grid[row][col] = 'E'


            return grid

    def findNeighbors(self, rowin, colin):
        """ Takes col, row and grid as input and returns as list of neighbors
        """
        cols = self.grid.shape[1]
        rows = self.grid.shape[0]
        neighbors = []
        for col in range(colin-1, colin+2):
            for row in range(rowin-1, rowin+2):
                if (-1 < rowin < rows and
                    -1 < colin < cols and
                    (rowin != row or colin != col) and
                    (0 <= col < cols) and
                    (0 <= row < rows)):
                    neighbors.append((row,col))

        return neighbors

    def sumMines(self, col, row, grid):
        """ Finds amount of mines adjacent to a field.
        """
        mines = 0
        neighbors = self.findNeighbors(row,col)
        for n in neighbors:
            if grid[n[0],n[1]] == 'B':
                mines = mines + 1
        return mines


    def printState(self):
        """Prints the current state"""
        grid = self.state
        cols = grid.shape[1]
        rows = grid.shape[0]
        for row in range(0,rows):
            print(' ')
            for col in range(0,cols):
                print(grid[row][col], end=' ')


    def printBoard(self):
        """Prints the board """
        grid = self.grid
        cols = grid.shape[1]
        rows = grid.shape[0]
        for row in range(0, rows):
            print(' ')
            for col in range(0, cols):
                print(grid[row][col], end=' ')

    def reveal(self, col, row, checked, press = "LM"):
        """Finds out which values to show in the state when a square is pressed
           Checked : np.array((row,col)) to check which squares has already been checked
                     If the field is not a bomb we want to reveal it, if the field is empty
                     we want to find it's neighbors and reveal them too if they are not a bomb.
        """
        if press == "LM":
            if checked[row][col] != 0:
                return
            checked[row][col] = checked[row][col] + 1
            if self.grid[row][col] != 'B':

                if self.display:
                    self.drawSpirit(col, row, self.grid[row][col])

                #Reveal to state space
                self.state[row][col] = self.grid[row][col]

                if self.grid[row][col] == 'E':
                    neighbors = self.findNeighbors(row,col)
                    for n in neighbors:
                        if not checked[n[0],n[1]]:
                            self.reveal(n[1], n[0], checked)

        elif press == "RM":
            #Draw flag, not used for agent
            pass

            if self.display:
                self.drawSpirit(col, row, "flag")

    def action(self, row, col):
        """ External action, taken by human or agent
            a: tuple - row and column of the tile to act on
         """

        #If press a bomb game over, start new game and return bad reward, -10 in this case
        if self.grid[row][col] == "B":
            self.lost += 1
            self.initGame()
            return({"s" : self.state, "r" : -10000})

        #Take action and reveal new state
        self.reveal(col, row , np.zeros_like(self.grid))
        if self.display == True:
            pygame.display.flip()


        #Winning condition
        if np.sum(self.state == "U") == self.mines:
            self.won += 1
            self.initGame()
            return({"s" : self.state, "r" : 10})

        #Get the reward for the given action
        reward = self.compute_reward()

        #return the state and the reward
        return({"s" : self.state, "r" : reward})

    def compute_reward(self):
        """Computes the reward for a given action"""

        #Reward = 1 if we get less unknowns, 0 otherwise
        if (np.sum(self.state_last == 'U') - np.sum(self.state == 'U')) > 0:
            reward = 1
        else:
            reward = -100


        self.state_last = np.copy(self.state)
        return(reward)


    def drawSpirit(self, col, row, type):
        """
        Draws a spirit at pos col, row of type = [E (empty), B (bomb), 1, 2, 3, 4, 5, 6, F (flag)]
        """
        myfont = self.myfont
        sizeofsq = self.sizeofsq

        if type == 'E':
            if self.checkpattern(col,row):
                c = (242, 244, 247)
            else:
                c = (247, 249, 252)
            pygame.draw.rect(self.screen, c, pygame.Rect(col*sizeofsq, row*sizeofsq, sizeofsq, sizeofsq))

        else:
            self.drawSpirit(col, row, 'E')
            if type == 1:
                text = myfont.render("1", 1, (0, 204, 0))
            elif type == 2:
                text = myfont.render("2", 1, (255, 204, 0))
            elif type == 3:
                text = myfont.render("3", 1, (204, 0, 0))
            elif type == 4:
                text = myfont.render("4", 1, (0, 51, 153))
            elif type == 5:
                text = myfont.render("5", 1, (255, 102, 0))
            elif type == 6:
                text = myfont.render("6", 1, (255, 102, 0))
            elif type == 'flag':
                text = myfont.render("F", 1, (255, 0, 0))

            #Get the text rectangle and center it inside the rectangles
            textRect = text.get_rect()
            textRect.center = (col*self.sizeofsq + int(0.5*self.sizeofsq)),(row*self.sizeofsq + int(0.5*self.sizeofsq))
            self.screen.blit(text, textRect)

    def checkpattern(self, col, row):
        #Function to construct the checked pattern in pygame
        if row % 2:
            if col % 2: #If unequal
                return True
            else: #if equal
                return False
        else:
            if col % 2: #If unequal
                return False
            else: #if equal
                return True

    def initPattern(self):
        #Initialize pattern:

        c1 = (4, 133, 223) #color1
        c2 = (4, 145, 223) #color2
        rects = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.checkpattern(col, row):
                    c = c1
                else:
                    c = c2

                rects.append(pygame.draw.rect(self.screen, c, pygame.Rect(col*self.sizeofsq, row*self.sizeofsq, self.sizeofsq, self.sizeofsq)))

        pygame.display.flip()

    def get_state(self):
        return self.state

if __name__ == "__main__":

    game = Minesweeper(display=True)
    game.printState()

    i = 0
    #start = time.time()
    while True:

        inp = input("Enter input (ROW,COL)")
        # print(inp)
        row = int(inp[1])
        # print("row", row)
        col = int(inp[3])
        # print("col", col)
        v = game.action(row,col)
        game.printState()
        print("\nReward = {}".format(v["r"]))
