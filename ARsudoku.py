from SudokuSolverAR import *
from sudoku import *
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

model_path = '/Users/jasonyuan/Desktop/Sudoku Project/sudoku_model_batch128_lr0.001_epochs20.pt'

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No Camera")
    exit()

frame_rate = 10
prev = 0
while True:
    time_elapsed = time.time() - prev
    # Capture frame-by-frame
    ret,frame = cap.read()
    # print(frame.shape)

    # If frame is read correctly, ret = True
    if not ret:
        print("Can't receive frame")
        break

    if time_elapsed > 1./frame_rate:
        prev = time.time()
        frame_temp = frame.copy()
        mask = np.zeros((frame.shape[0],frame.shape[1]),'uint8')
        rect = None
        frame_temp, mask, rect = detectSudoku(frame_temp)

        if (np.array_equal(frame_temp,frame)):
            print('same')
            frame = cv2.resize(frame,(1066,600))
            mask = cv2.resize(mask,(1066,600))
            cv2.imshow('frame',frame)
            # cv2.imshow('solved',mask)
        else:
            print('different')
            model = DigitClassifier()
            solution = Sudoku('E',100)

            cells,transformed_pts,im_adjusted = getCells(frame_temp,mask,rect)
            solution.game = predictDigits(model,model_path,cells)

            # print(len(cells),len(cells[0]))
            # print(solution.game)

            solution.solved = [x[:] for x in solution.game]
            solved = solution.opt_solve(solution.solved)
            if not solved:
                frame_temp = cv2.resize(frame_temp,(1066,600))
                cv2.imshow('frame',frame_temp)
                # cv2.destroyAllWindows('solved')
                continue

            annotated = printDigits(frame,im_adjusted,solution.solved,transformed_pts,rect)
            frame_temp = cv2.resize(frame_temp,(1066,600))
            # mask = cv2.resize(mask,(1066,600))
            annotated = cv2.resize(annotated,(1066,600))
            cv2.imshow('frame',annotated)
            # cv2.imshow('solved',annotated)

    if cv2.waitKey(1) == ord('q'):
        break

# When finished, release the capture
cap.release()
cv2.destroyAllWindows()
