# -*- coding: utf-8 -*-

'''
The Capsules Network.

@author: Yuxian Meng
'''

#import datetime
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
from networks import CapsNet,LeNet5

def to_np(x):
    return x.data.cpu().numpy()





def adjust_learning_rate(optimizer, iter, total,power=0.9):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.01*(1 - float(iter)/total)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def validate(test_loader,model,samples = 20):
    i = 1
    model.eval()
    correct = 0
    for data in test_loader:
        if i > samples:
            break
        imgs,labels = data #b,1,28,28; #b
        imgs,labels = Variable(imgs),Variable(labels)
        if use_cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()
        out = model(imgs,lambda_) #b,10,17
        if type(model)==LeNet5:
            out_labels  = out
        else:
            out_poses, out_labels = out[:,:-10],out[:,-10:] #b,16*10; b,10
        loss = model.loss(out_labels, labels, m)
        l2 = float(out_labels.size()[0])
        pred = out_labels.max(1)[1] #b
        acc = pred.eq(labels).cpu().sum().data[0]
        correct += acc
        i += 1
    model.train()
    acc = correct/(samples*l2)
    return acc

if __name__ == "__main__":
    args = get_args()
    time = strftime("%Y%m%d%H%M%S", gmtime())
    if args.fname!='':
        time = args.fname
    logger = Logger('./logs/'+time)
    #    A,B,C,D,E,r = 32,32,32,32,10,args.r # a classic CapsNet
    # A=32, B=8, C=16, D=16, batch_size=50, iteration number of EM routing: 2
    if args.dataset=='CIFAR10':
        transform = vision.transforms.ToTensor()
        train_data = vision.datasets.CIFAR10(".", train=True, transform=transform, download=True )
        test_data = vision.datasets.CIFAR10(".", train=False, transform=transform ,download=True)
        A,B,C,D,E,r = 32,8,16,16,10,2 #args.r # a small CapsNet
        n_channels =3
    elif args.dataset=='MNIST':
        transform = transforms.ToTensor()
        train_data = vision.datasets.MNIST(".", train=True, transform=transform , download=True )
        test_data = vision.datasets.MNIST(".", train=False, transform=transform ,download=True)
        A,B,C,D,E,r = 64,8,16,16,10,args.r # a small CapsNet
        n_channels = 1
    elif args.dataset=='SVHN':
        transform = vision.transforms.ToTensor()
        train_data = vision.datasets.SVHN(".", split='train', transform=transform ,download=True )
        test_data = vision.datasets.SVHN(".", split='test', transform=transform , download=True)
        n_channels =3
        A,B,C,D,E,r = 64,8,16,16,10,args.r # a small CapsNet


    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=args.batch_size,
                                           shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=args.batch_size,
                                              shuffle=True)
    use_cuda = args.use_cuda
    steps = len(train_loader.dataset)//args.batch_size
    lambda_ = args.lambda_ #1e-3 #TODO:find a good schedule to increase lambda and m
    m = args.m #0.2

    if args.model=='MatrixCaps':
        if args.custom==0:
            model = CapsNet(n_channels,A,B,C,D,E,r)
        else:
            model = CapsNet(n_channels,args.A,args.B,args.C,args.D,args.E,r)
    elif args.model=='CNN':
        model = LeNet5()

    use_cuda = args.use_cuda
    steps = len(train_loader.dataset)//args.batch_size

    print("Steps:"+str(steps))
    with torch.cuda.device(args.gpu):
        if args.pretrained:
            model.load_state_dict(torch.load(args.pretrained))
            m = 0.8
            lambda_ = 0.9
        if use_cuda:
            print("activating cuda")
            model.cuda()

        #optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=0.00005 )
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience = 1)
        b = 0
        total = args.num_epochs * steps
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
                if use_cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                out = model(imgs,lambda_) #b,10,17
                if type(model)==LeNet5:
                    out_labels  = out
                else:
                    out_poses, out_labels = out[:,:-10],out[:,-10:] #b,16*10; b,10
                loss = model.loss(out_labels, labels, m)
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
                loss.backward()
                optimizer.step()
#                adjust_learning_rate(optimizer,b,total)

                pred = out_labels.max(1)[1] #b
                acc = pred.eq(labels).cpu().sum().data[0]
                correct += acc
                if b % args.print_freq == 0:
                    print("batch:{}, loss:{:.4f}, acc:{:}/{} : {}".format(
                            b, loss.data[0],acc, args.batch_size, float(acc)/args.batch_size))
                    info = {
                            'loss': loss.data[0],
                            'accuracy': float(acc)/args.batch_size
                        }
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, b)

                if b % args.val_freq == 0:
                    val_acc = validate(test_loader,model,samples= 20 )
                    info = {
                            'val_acc': val_acc
                        }
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, b)
                if args.dist_freq!=0 and b % args.dist_freq == 0:
                    for tag, value in model.named_parameters():
                        try:
                            tag = tag.replace('.', '/')
                            logger.histo_summary(tag, to_np(value), b)
                            logger.histo_summary(tag+'/grad', to_np(value.grad), b)
                        except:
                            print(tag)


            acc = correct/len(train_loader.dataset)
            print("Epoch{} Train acc:{:4}".format(epoch, acc))
            scheduler.step(acc)
            torch.save(model.state_dict(), "./model_{}.pth".format(epoch))
            #Test
            print('Testing...')
            correct = 0
            model.eval()
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
            model.train()
            print("Epoch{} Test acc:{:4}".format(epoch, acc))







