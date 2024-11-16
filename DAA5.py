#!/usr/bin/env python
# coding: utf-8

# In[24]:


def isSafe(board, row, col):
    # Check column for any queen in the same column
    for i in range(row):
        if board[i][col] == 'Q':
            return False

    # Check upper-left diagonal
    i, j = row, col
    while i >= 0 and j >= 0:
        if board[i][j] == 'Q':
            return False
        i -= 1
        j -= 1

    # Check upper-right diagonal
    i, j = row, col
    while i >= 0 and j < len(board):
        if board[i][j] == 'Q':
            return False
        i -= 1
        j += 1

    return True

def printBoard(board):
    for row in board:
        print(" ".join(row))
    print()

def placeQueens(board, row):
    if row == len(board):
        printBoard(board)
        return True  # Stop after finding the first solution

    for col in range(len(board)):
        if isSafe(board, row, col):
            board[row][col] = 'Q'
            if placeQueens(board, row + 1):
                return True  # Stop recursion once a solution is found
            board[row][col] = '.'  # Backtrack

    return False


def solveNQueens(n):
    # Initialize board with all empty spaces
    board = [['.' for _ in range(n)] for _ in range(n)]

    # Start placing queens from the first row
    if not placeQueens(board, 0):  # Start from the first row
        print("No solution found")

# Example usage for an 8x8 board
solveNQueens(8)


# In[ ]:





# In[ ]:




