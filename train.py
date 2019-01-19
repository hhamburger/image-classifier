import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import futils

ap = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments

ap.add_argument('data_dir', nargs='*', action="store", type=str, default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", type=str, default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", type=float, default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", type=float, default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--arch', dest="arch", action="store", type=str, default="vgg16")
ap.add_argument('--hidden_units', dest="hidden_units", action="store", type=int, default=120)

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs


train_datasets, trainloader, valid_loader, testloader = futils.load_data(where)


model, criterion, optimizer = futils.nn_setup(structure,dropout,hidden_layer1,lr,power)


futils.train_network(model, criterion, optimizer, epochs, 20, trainloader, validloader, power)


futils.save_checkpoint(train_datasets, path,structure,hidden_layer1,dropout,lr)


print("The Model is trained")