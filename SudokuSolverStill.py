from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
# from google.colab import drive
import cv2
import os
import matplotlib.pyplot as plt
import imutils

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from sudoku import *

def preprocess_image(path): # Returns a list of the individual cells of the sudoku board
  # Read in the image as a grayscale image (wihout camera)
  im = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

  # Blur the image
  im_blur = cv2.GaussianBlur(im,(9,9),0)

  # Apply adaptive thresholding
  im_thresh = cv2.adaptiveThreshold(im_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #11 and 2 would be tunable parameters

  # Dilate the image
  kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
  im_dilate = cv2.dilate(cv2.bitwise_not(im_thresh),kernel)

  # Identify all contours in the image
  contours,_ = cv2.findContours(im_dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

  # Find largest contour
  contours = sorted(contours,key=cv2.contourArea,reverse=True)

  for i in contours:
    perimeter = cv2.arcLength(i,True)
    approx = cv2.approxPolyDP(i,0.02*perimeter,True)
    if len(approx) == 4:
      best_cont = i
      break

  # Create a mask and draw the largest contour
  mask = np.zeros((im.shape),np.uint8)
  mask = cv2.drawContours(mask,[best_cont],-1,255,2)

  # Find the perimeter and approximate quadrilateral that bounds sudoku
  perimeter = cv2.arcLength(best_cont,True)
  approx = cv2.approxPolyDP(best_cont,0.02*perimeter,True)

  rect = np.zeros((4,2),'float32')

  #Top left and Bottom right:
  approx = sorted(approx.reshape(4,2),key=sum)
  rect[0] = approx[0] #top left
  rect[2] = approx[3] #bottom right

  #Top right and Bottom left:
  approx = sorted(approx,key=np.diff)
  rect[1] = approx[0] #top right
  rect[3] = approx[3] #bottom left

  width1 = np.sqrt((rect[1][0]-rect[0][0])**2+(rect[1][1]-rect[0][1])**2)
  width2 = np.sqrt((rect[2][0]-rect[3][0])**2+(rect[2][1]-rect[3][1])**2)
  width = int(max(width1,width2))

  height1 = np.sqrt((rect[0][0]-rect[3][0])**2+(rect[0][1]-rect[3][1])**2)
  height2 = np.sqrt((rect[1][0]-rect[2][0])**2+(rect[1][1]-rect[2][1])**2)
  height = int(max(height1,height2))

  # top-left, top-right, bottom-right, bottom-left transformed points
  transformed_pts = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype='float32')

  # Transform perspective of the original image into a top-down view (without camera)
  im1 = cv2.imread(path)

  M = cv2.getPerspectiveTransform(rect,transformed_pts)
  im_adjusted = cv2.warpPerspective(im1,M,(width,height))
  # plt.imshow(im_adjusted)
  # plt.savefig('/Users/jasonyuan/Desktop/adjusted.jpg')
  # print(im_adjusted.shape)

  # This is the simple way of extracting the cells, obviously this is not the best
  # A better way would be to identify all corners of the transformed grid through
  # horizontal and vertical lines to extract cells exactly
  cell_w = im_adjusted.shape[1]//9
  cell_h = im_adjusted.shape[0]//9

  cells = []
  for row in range(0,cell_h*9,cell_h):
    temp = im_adjusted[row:row+cell_h+1]
    temps = []
    for col in range(0,cell_w*9,cell_w):
      tempgrid = []
      for r in temp:
        tempgrid.append(r[col:col+cell_w+1])
      temps.append(tempgrid)
    cells.append(temps)

  for row in range(0,len(cells)):
    for col in range(0,len(cells[0])):
      cells[row][col] = np.array(cells[row][col])

  return cells

class DigitClassifier(nn.Module):
  def __init__(self):
    super(DigitClassifier,self).__init__()
    self.conv1 = nn.Conv2d(1,32,3)
    self.pool1 = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(32,64,3)
    self.pool2 = nn.MaxPool2d(2,2)
    self.conv3 = nn.Conv2d(64,64,3)
    self.pool3 = nn.MaxPool2d(2,2)
    self.fc1 = nn.Linear(1*1*64,100)
    self.fc2 = nn.Linear(100,32)
    self.fc3 = nn.Linear(32,10)
    self.dropout = nn.Dropout(p=0.3)

  def forward(self,x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = self.pool3(F.relu(self.conv3(x)))
    x = x.reshape(-1,1*1*64)
    x = self.dropout(x)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

def getSudokuBoard(model,path,cells):
  if torch.cuda.is_available():
    model.load_state_dict(torch.load(path))
  else:
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

  board = [[],[],[],[],[],[],[],[],[]]

  for i in range(0,9):
    for j in range(0,9):
      # print("i: {}, j: {}".format(i,j))

      # Make a copy of the cell image and proces
      test = cells[i][j].copy()
      test = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
      test = cv2.erode(test,(1,1),iterations=1,borderType=cv2.BORDER_DEFAULT)
      _,test = cv2.threshold(test,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
      test = cv2.copyMakeBorder(test,6,6,4,4,cv2.BORDER_WRAP)

      # Find the largest contour in the cell image
      edges,_ = cv2.findContours(test.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      e = max(edges,key=cv2.contourArea)

      # Extract the digit from the cell
      mask_d = np.zeros(test.shape,dtype="uint8")
      digit = np.zeros(test.shape,dtype="uint8")

      mask_d = cv2.drawContours(mask_d,[e],-1,255,-1)
      test = cv2.bitwise_not(test)
      digit[mask_d == 255] = test[mask_d == 255]
      digit = cv2.erode(digit,(3,3),iterations=1,borderType=cv2.BORDER_DEFAULT)

      # if (i == 2) and (j == 3):
      #     plt.imshow(test)
      #     plt.show()
      #     return True

      # Find the contours in the extracted digit
      d_cnt,_ = cv2.findContours(digit.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      # print("Contours: ",len(d_cnt))

      # Check to see if the cell is empty by checking number of contours
      # and the largest bounding rectangle in extracted digit image
      if (len(d_cnt) == 0):
        board[i].insert(j,0)
        continue
      else:
        max_area = 0
        for c in d_cnt:
          _,_,wid,hei = cv2.boundingRect(c)
          area = wid*hei
          if (area > max_area):
            max_area = area
        percent = ((max_area)/(28*28))*100
        # print("Percent :",percent)
        if percent < 5:
          board[i].insert(j,0)
          continue

      # Find the bounding rectangle of the extracted digit and center the digit
      digit_copy = digit.copy()

      if len(d_cnt) > 0:
        max_area = 0
        for c in d_cnt:
          _,_,wid,hei = cv2.boundingRect(c)
          area = wid*hei
          if (area > max_area):
            max_area = area
            x,y,w,h = cv2.boundingRect(c)

        # print(x,y,w,h)

        if h > w:
          digit_copy = digit_copy[int(y-0.1*h):int(y+1.1*h),int(x-0.5*w):int(x+1.5*w)]
        elif w > h:
          digit_copy = digit_copy[int(y-0.5*h):int(y+1.5*h),int(x-0.1*w):int(x+1.1*w)]
        else:
          digit_copy = digit_copy[int(y-0.1*h):int(y+1.1*h),int(x-0.1*w):int(x+1.1*w)]

      # Find and then draw the contour in centered digit image
      d_cnt_new,_ = cv2.findContours(digit_copy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      cv2.drawContours(digit_copy,d_cnt_new,-1,255,1)

      # Process the digit for model prediction
      digit_copy = cv2.resize(digit_copy,(28,28),cv2.INTER_AREA)
      digit_copy = cv2.GaussianBlur(digit_copy,(3,3),0)

      digit_copy = digit_copy.astype('float')
      digit_copy = digit_copy.reshape(1,1,28,28)
      digit_copy /= 255

      digit_copy = torch.Tensor(digit_copy)

      # Make prediction of the cell image
      model.eval()
      if torch.cuda.is_available():
        model.cuda()
        out = model(digit_copy.cuda())
      else:
        out = model(digit_copy)

      out = F.softmax(out,dim=1)
      val = torch.argmax(out).item()

      # Take second likeliest prediction if 0 was initially chosen
      if val == 0:
        new = torch.cat((out[0][0:val],out[0][val+1:]))
        new = new.reshape(1,-1)
        val = torch.argmax(new).item()
        val = val + 1

      board[i].insert(j,val)
      # print(board[i][j],i,j)
  return board

def projectToImg(solution,im_path):
  # Read in image (without camera)
  img = cv2.imread(im_path)

  # Read in the image as a grayscale image (without camera)
  im = cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)

  # Blur the image
  im_blur = cv2.GaussianBlur(im,(9,9),0)

  # Apply adaptive thresholding
  im_thresh = cv2.adaptiveThreshold(im_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #11 and 2 would be tunable parameters

  # Dilate the image
  kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
  im_dilate = cv2.dilate(cv2.bitwise_not(im_thresh),kernel)

  # Identify all contours in the image
  contours,_ = cv2.findContours(im_dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

  # Find largest contour
  contours = sorted(contours,key=cv2.contourArea,reverse=True)

  for i in contours:
    perimeter = cv2.arcLength(i,True)
    approx = cv2.approxPolyDP(i,0.02*perimeter,True)
    if len(approx) == 4:
      best_cont = i
      break

  # Create a mask and draw the largest contour
  mask = np.zeros((im.shape),np.uint8)
  mask = cv2.drawContours(mask,[best_cont],-1,255,2)

  # Find the perimeter and approximate quadrilateral that bounds sudoku
  perimeter = cv2.arcLength(best_cont,True)
  approx = cv2.approxPolyDP(best_cont,0.02*perimeter,True)

  rect = np.zeros((4,2),'float32')

  #Top left and Bottom right:
  approx = sorted(approx.reshape(4,2),key=sum)
  rect[0] = approx[0] #top left
  rect[2] = approx[3] #bottom right

  #Top right and Bottom left:
  approx = sorted(approx,key=np.diff)
  rect[1] = approx[0] #top right
  rect[3] = approx[3] #bottom left

  width1 = np.sqrt((rect[1][0]-rect[0][0])**2+(rect[1][1]-rect[0][1])**2)
  width2 = np.sqrt((rect[2][0]-rect[3][0])**2+(rect[2][1]-rect[3][1])**2)
  width = int(max(width1,width2))

  height1 = np.sqrt((rect[0][0]-rect[3][0])**2+(rect[0][1]-rect[3][1])**2)
  height2 = np.sqrt((rect[1][0]-rect[2][0])**2+(rect[1][1]-rect[2][1])**2)
  height = int(max(height1,height2))

  # top-left, top-right, bottom-right, bottom-left transformed points
  transformed_pts = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype='float32')

  # Transform perspective of the original image into a top-down view (without camera)
  im1 = cv2.imread(im_path)

  M = cv2.getPerspectiveTransform(rect,transformed_pts)
  im_adjusted = cv2.warpPerspective(im1,M,(width,height))

  w = im_adjusted.shape[1]
  h = im_adjusted.shape[0]
  cell_w = im_adjusted.shape[1]//9
  cell_h = im_adjusted.shape[0]//9

  if (cell_w  == 0 or cell_h == 0):
      return []

  for row, i in zip(range(0,h,cell_h),range(0,9)):
    startY = row
    endY = row+cell_h+1
    for col, j in zip(range(0,w,cell_w),range(0,9)):
      startX = col
      endX = col+cell_w+1

      textX = int((endX - startX)*0.33)
      textY = int((endY - startY)*(-0.2))
      textX += startX
      textY += endY

      scale = np.ceil(0.02*cell_h)
      cv2.putText(im_adjusted,str(solution[i][j]),(textX,textY),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,255),1,cv2.LINE_AA)

  inv_M = cv2.getPerspectiveTransform(transformed_pts,rect)
  im_annotate = cv2.warpPerspective(im_adjusted,inv_M,(img.shape[1],img.shape[0]))

  img[im_annotate != 0] = im_annotate[im_annotate != 0]

  return img

def main(path,model_path):
  solution = Sudoku('E',100)
  model = DigitClassifier()

  cells = preprocess_image(path)
  solution.game = getSudokuBoard(model,model_path,cells)
  print(solution.game)
  solution.solved = [x[:] for x in solution.game]
  solution.solve(solution.solved)
  f_img = projectToImg(solution.solved,path)

  return f_img
