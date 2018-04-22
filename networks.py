import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler

from Capsules import PrimaryCaps, ConvCaps
from utils import get_args, get_dataloader
import torchvision as vision
from torchvision import transforms
from pdb import set_trace as st
import numpy as np
from logger import Logger
from time import gmtime, strftime
from collections import OrderedDict

class CapsNet(nn.Module):
    def __init__(self,in_channels=1,A=32,B=32,C=32,D=32, E=10,r = 3):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=A,
                               kernel_size=5, stride=2)
        self.primary_caps = PrimaryCaps(A,B)
        self.convcaps1 = ConvCaps(B, C, kernel = 3, stride=2,iteration=r,
                                  coordinate_add=False, transform_share = False)
        self.convcaps2 = ConvCaps(C, D, kernel = 3, stride=1,iteration=r,
                                  coordinate_add=False, transform_share = True)
        self.classcaps = ConvCaps(D, E, kernel = 0, stride=1,iteration=r,
                                  coordinate_add=True, transform_share = True)
        self.batchnorm = nn.BatchNorm2d(in_channels)

    def forward(self,x,lambda_): #b,1,28,28
        x = self.batchnorm(x)
        x = F.relu(self.conv1(x)) #b,32,12,12
        x = self.primary_caps(x) #b,32*(4*4+1),12,12
        x = self.convcaps1(x,lambda_) #b,32*(4*4+1),5,5
        x = self.convcaps2(x,lambda_) #b,32*(4*4+1),3,3
        x = self.classcaps(x,lambda_).view(-1,10*16+10) #b,10*16+10
        return x

    def loss(self, x, target, m): #x:b,10 target:b
        b = x.size(0)
        a_t = torch.cat([x[i][target[i]] for i in range(b)]) #b
        a_t_stack = a_t.view(b,1).expand(b,10).contiguous() #b,10
        u = m-(a_t_stack-x) #b,10
        mask = u.ge(0).float() #max(u,0) #b,10
        loss = ((mask*u)**2).sum()/b - m**2  #float
        return loss

    def loss2(self,x ,target):
        loss = F.cross_entropy(x,target)
        return loss

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x,_):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def loss(self,x,target,m=None):
        loss = F.cross_entropy(x,target)
        return loss