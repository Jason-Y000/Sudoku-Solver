# Sudoku-Solver

Capture an image of a 9x9 Sudoku board and output the solved board. Used a CNN to do the digit prediction of the board and a backtracking algorithm to solve the resultant Sudoku board. 

sudoku.py - Contains the class for Sudoku
SudokuSolver.py - Function for image processing of the Sudoku board, digit extraction, solving board and then projecting back to the original image

The CNN model was trained with images from the MNIST dataset and a custom dataset
