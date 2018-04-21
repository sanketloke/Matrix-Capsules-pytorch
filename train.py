# -*- coding: utf-8 -*-

'''
The Capsules Network.

@author: Yuxian Meng
'''


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


    def forward(self,x,lambda_): #b,1,28,28
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



def adjust_learning_rate(optimizer, iter, total,power=0.9):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.01*(1 - float(iter)/total)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == "__main__":
    args = get_args()

    if args.dataset=='CIFAR10':
        transform = vision.transforms.ToTensor()
        train_data = vision.datasets.CIFAR10(".", train=True, transform=transform, download=True )
        test_data = vision.datasets.CIFAR10(".", train=False, transform=transform ,download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=args.batch_size,
                                                  shuffle=True)
        use_cuda = args.use_cuda
        steps = len(train_loader.dataset)//args.batch_size
        lambda_ = 1e-3 #TODO:find a good schedule to increase lambda and m
        m = 0.2
        # A=32, B=8, C=16, D=16, batch_size=50, iteration number of EM routing: 2
        A,B,C,D,E,r = 32,8,16,16,10,2 #args.r # a small CapsNet
    #    A,B,C,D,E,r = 32,32,32,32,10,args.r # a classic CapsNet
        model = CapsNet(3,A,B,C,D,E,r)
    elif args.dataset=='MNIST':
        transform = transforms.ToTensor()
        train_data = vision.datasets.MNIST(".", train=True, transform=transform , download=True )
        test_data = vision.datasets.MNIST(".", train=False, transform=transform ,download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=args.batch_size,
                                                  shuffle=True)
        use_cuda = args.use_cuda
        steps = len(train_loader.dataset)//args.batch_size
        lambda_ = 1e-3 #TODO:find a good schedule to increase lambda and m
        m = 0.2
        A,B,C,D,E,r = 64,8,16,16,10,args.r # a small CapsNet
    #    A,B,C,D,E,r = 32,32,32,32,10,args.r # a classic CapsNet
        model = CapsNet(1,A,B,C,D,E,r)
    elif args.dataset=='SVHN':
        transform = vision.transforms.ToTensor()
        train_data = vision.datasets.SVHN(".", split='train', transform=transform ,download=True )
        test_data = vision.datasets.SVHN(".", split='test', transform=transform , download=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=args.batch_size,
                                                  shuffle=True)
        use_cuda = args.use_cuda
        steps = len(train_loader.dataset)//args.batch_size
        lambda_ = 1e-3 #TODO:find a good schedule to increase lambda and m
        m = 0.2
        A,B,C,D,E,r = 64,8,16,16,10,args.r # a small CapsNet
    #    A,B,C,D,E,r = 32,32,32,32,10,args.r # a classic CapsNet
        model = CapsNet(3,A,B,C,D,E,r)
    print("Steps:"+str(steps))
    with torch.cuda.device(args.gpu):
#        print(args.gpu, type(args.gpu))
        if args.pretrained:
            model.load_state_dict(torch.load(args.pretrained))
            m = 0.8
            lambda_ = 0.9
        if use_cuda:
            print("activating cuda")
            model.cuda()

        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=0.00005 )
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience = 1)
        b = 0
        total = args.num_epochs * steps
        if args.dataset=='MNIST':
            batchnorm = nn.BatchNorm2d(1)
        else:
            batchnorm = nn.BatchNorm2d(3)
        for epoch in range(args.num_epochs):
            #Train
            print("Epoch {}".format(epoch))
            correct = 0
            loss=0
            for data in train_loader:
                b += 1

                if lambda_ < 1:
                    lambda_ += 2e-3/steps
                if m < 0.9:
                    m += 2e-3/steps
                optimizer.zero_grad()
                imgs,labels = data #b,1,28,28; #b
                imgs,labels = Variable(imgs),Variable(labels)
                imgs = batchnorm(imgs)
                if use_cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                out = model(imgs,lambda_) #b,10,17
                out_poses, out_labels = out[:,:-10],out[:,-10:] #b,16*10; b,10
                loss = model.loss(out_labels, labels, m)
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
                loss.backward()
                optimizer.step()
                adjust_learning_rate(optimizer,b,total)
                #print(lambda_,m,loss.data.cpu().numpy())
                #stats
                pred = out_labels.max(1)[1] #b
                acc = pred.eq(labels).cpu().sum().data[0]
                correct += acc
                if b % args.print_freq == 0:
                    print("batch:{}, loss:{:.4f}, acc:{:}/{} : {}".format(
                            b, loss.data[0],acc, args.batch_size, float(acc)/args.batch_size))
            acc = correct/len(train_loader.dataset)
            print("Epoch{} Train acc:{:4}".format(epoch, acc))
            scheduler.step(acc)
            torch.save(model.state_dict(), "./model_{}.pth".format(epoch))
            #Test
            print('Testing...')
            correct = 0
            for data in test_loader:
                imgs,labels = data #b,1,28,28; #b
                imgs,labels = Variable(imgs),Variable(labels)
                if use_cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                out = model(imgs,lambda_) #b,10,17
                out_poses, out_labels = out[:,:-10],out[:,-10:] #b,16*10; b,10
                loss = model.loss(out_labels, labels, m)
                #stats
                pred = out_labels.max(1)[1] #b
                acc = pred.eq(labels).cpu().sum().data[0]
                correct += acc
            acc = correct/len(test_loader.dataset)
            print("Epoch{} Test acc:{:4}".format(epoch, acc))







