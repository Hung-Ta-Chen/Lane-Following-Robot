import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets ,transforms
import torchvision
from matplotlib import pyplot as plt
import numpy as np
#from google.colab.patches import cv2_imshow
import torch.optim as optim
from torch.autograd import Variable
#from torch.utils.tensorboard import SummaryWriter
#from tensorflow import summary
from torchvision.utils import make_grid
import os
from trail_CNN import TrailCNN

def train(net, train_loader, optimizer, criterion, epoch_idx, device=None):
  net.train()
  running_loss = 0.0
  batch_cnt = 0
  for batch_idx, (inputs, labels) in enumerate(train_loader):
    if device != None:
      inputs, labels = inputs.to(device), labels.to(device)
  
    optimizer.zero_grad()   # zero the parameter gradients
    outputs = net(inputs)   # Forward
    loss = criterion(outputs, labels)
    loss.backward()   # Backprop
    optimizer.step()  # Update parameters

    running_loss += loss.item()
    batch_cnt = batch_idx

    if batch_idx % 10 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(  
        epoch_idx, batch_idx * len(inputs), len(train_loader.dataset),      
        100. * batch_idx / len(train_loader), loss.item()))
      
  return (running_loss / batch_cnt)

def test(net, test_loader, criterion, device=None, set_name="Test"):
  net.eval()
  test_loss = 0.0
  correct = 0.0
  batch_cnt = 0
  with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):
      if device != None:
        inputs, labels = inputs.to(device), labels.to(device)
      outputs = net(inputs)
      c = criterion(outputs, labels)
      test_loss += c.item()  # sum up batch loss
      pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(labels.view_as(pred)).sum().item()
      batch_cnt = batch_idx
      

  test_loss /= batch_cnt

  print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    set_name, test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
  return (100. * correct / len(test_loader.dataset))

def main():
  #Load data set
  train_data = datasets.ImageFolder(
      './trail_dataset/train_data',
      transform = transforms.Compose([transforms.ToTensor()])                         
  )
  
  test_data = datasets.ImageFolder(
      './trail_dataset/test_data',
      transform = transforms.Compose([transforms.ToTensor()])                         
  )
  
  train_loader = DataLoader(train_data, batch_size=50,shuffle= True)
  test_loader = DataLoader(test_data, batch_size=1,shuffle=False)
  train_val_loader = DataLoader(train_data, batch_size=1, shuffle=False)
  
  #show labels
  print(train_data.classes)
  print(train_data.class_to_idx)
  
  #check availability of gpu
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  net = TrailCNN(kernel_size=3).to(device)
  optimizer = optim.Adam(net.parameters(), lr=0.003)
  criterion = nn.CrossEntropyLoss()
  epoch_num = 100
  
  # Training
  loss_list = []
  train_acc_list = []
  test_acc_list = []
  # Training
  loss_list = []
  train_acc_list = []
  test_acc_list = []
  for epoch in range(1, epoch_num + 1):
    loss = train(net, train_loader, optimizer, criterion, epoch, device)
    loss_list.append(loss)
    
    if epoch % 10 == 0:
      torch.save(net.state_dict(), 'trail_cnn_{}.pth'.format(epoch))

    train_acc = test(net, train_val_loader, criterion, device, 'Train')
    train_acc_list.append(train_acc)
    test_acc = test(net, test_loader, criterion, device)
    test_acc_list.append(test_acc)

  # Save parameters of the model
  torch.save(net.state_dict(), 'trail_cnn_fin.pth')
  """
  # Load the parameters of model
  net.load_state_dict(torch.load('net_cifar_1.pt'))
  """
  
  # Plot Accuracy
  
  print("=== Show accuracy plot ===>>")
  fig , ax = plt.subplots()
  #plt.rcParams["figure.figsize"] = (8, 3.5)
  plt.plot(range(len(train_acc_list)), train_acc_list, label = "training accuracy", color = "blue", linewidth = 0.5)
  plt.plot(range(len(test_acc_list)), test_acc_list, label = "testing accuracy", color = "orange", linewidth = 0.5)
  plt.title("Accuracy of the Model")
  plt.ylabel("Accuracy Rate(%)")
  plt.xlabel("Epoch")
  leg = ax.legend(loc='lower right') 
  plt.savefig('trail_acc.png')
  plt.show()
  
  print(" ")
  
  # Plot learning curve
  print("=== Show learning plot ===>>")
  plt.plot(loss_list)
  plt.ylabel("Loss")
  plt.xlabel("Epoch")
  plt.title("Learning Curve")
  plt.savefig('trail_lc.png')
  plt.show()
  
  print(" ")
  
  
if __name__=="__main__":
  main()






