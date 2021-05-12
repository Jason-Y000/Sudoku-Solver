import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

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
