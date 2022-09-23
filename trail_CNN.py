# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 19:03:22 2021

@author: narut
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrailCNN(nn.Module):
  def __init__(self, kernel_size=3):
    super().__init__()
    self.pool = nn.MaxPool2d(2, 2)
    self.conv1 = nn.Conv2d(3, 64, kernel_size)
    self.conv2 = nn.Conv2d(64, 64, kernel_size)
    self.bn1 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(64, 128, kernel_size)
    self.conv4 = nn.Conv2d(128, 128, kernel_size)
    self.bn2 = nn.BatchNorm2d(128)
    self.conv5 = nn.Conv2d(128, 256, kernel_size)
    self.bn3 = nn.BatchNorm2d(256)
    self.fc1 = nn.Linear(4563200, 512)
    self.fc2 = nn.Linear(512, 18)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = self.pool(x)
    x = self.bn1(x)
    
    x = self.conv3(x)
    x = F.relu(x)
    x = self.conv4(x)
    x = F.relu(x)
    x = self.pool(x)
    x = self.bn2(x)
    
    x = self.conv5(x)
    x = F.relu(x)
    x = self.bn3(x)
    
    
    x = torch.flatten(x, start_dim=1)
    print(x.shape)
    
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    
    return x