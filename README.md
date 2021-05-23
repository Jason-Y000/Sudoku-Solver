# Sudoku-Solver

Capture an image of a 9x9 Sudoku board and output the solved board. Used a CNN to do the digit prediction of the board and a backtracking algorithm to solve the resultant Sudoku board. The CNN was trained with data from the MNIST dataset and the computer font digits from the Chars74k dataset

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
sudoku.py - Contains the class for Sudoku. Has two functions for solving a sudoku: opt_solve and solve. The solve function is a brute force backtracking algorithm while the opt_solve does some simple elimination based on sudoku rules before running backtracking. 

SudokuSolverStill.py - Functions for image processing of the Sudoku board, digit extraction, solving board and then projecting back to the original image. These functions are intended for still images of an unsolved sudoku board.

SudokuSolverAR.py - Fucntions for processing a sudoku board in real-time. Essentially the same as SudokuSolverStill.py with some minor changes.

ARsudoku.py - Program to use a camera to solve the sudoku puzzle in real-time.

model.py - Contains the class definition of the CNN model.

sudoku_model_batch128_lr0.001_epochs20.pt - File containing the pre-trained weights of the CNN model. Model weights were trained on GPU, so if you computer doesn't have GPU support be sure to load the model weights to CPU.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Helpful tutorials:

StackOverflow for help when there were errors

https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/#pyis-cta-modal

https://becominghuman.ai/image-processing-sudokuai-opencv-45380715a629

https://github.com/anhminhtran235/real_time_sudoku_solver

http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
