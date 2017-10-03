from __future__ import print_function
import torch.utils.data as data
from os import listdir
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as t
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from YLNet3D import *
from sampling import make_training_samples
import os
import time
import numpy as np
import math

#input = Variable(torch.randn(1,1,256,256))
#net = YLNet2D()
#output = net(input)
#print(output)
in_channels = 1
out_channels = 1
net = YLNet3D(out_channels)
net.train()

#------------------ Datasets ------------------------------
img_path = '/Users/Elaine/desktop/MICCAI/Training'
label_path = '/Users/Elaine/desktop/MICCAI/Labels'

num_patches = 1000
w_train = 0.7
w_val = 1 - w_train
patch_size = [27,27,27]

#----------- Hyperparameters for training -------------------
optimizer = optim.SGD(net.parameters(),lr=0.000001, momentum=0.9)
scheduler = MultiStepLR(optimizer,milestones=[10,20],gamma=0.1)#Set the learning rate of each parameter group to the initial lr decayed by gamma once
                                                             #the number of epoch reaches one of the milestones.
criterion = nn.MSELoss()
use_gpu = torch.cuda.is_available()
num_epochs = 5
batch_size = 50

#-------------------------------------------------------------
'''
def makeDatasets(img_path,label_path,w_train,w_val):
    labels,trainset,valset = [], [], []
    
    imgs, labels = listdir(img_path),listdir(label_path)
    for imgname,labelname in zip(imgs,labels):#Go through all subjects
        img_patches,label_patches = make_training_samples(os.path.join(img_path,imgname),os.path.join(label_path,labelname),num_patches,patch_size)
        print(np.array(img_patches).shape)
        trainset.append(img_patches[:int(len(img_patches)*w_train)])
        print(np.array(trainset).shape)
        valset.append(img_patches[int(len(label_patches)*w_val):])
        labels.append(label_patches)

    return np.array(trainset), np.array(valset)

trainset,valset = makeDatasets(img_path,label_path,w_train,w_val)

        

import nibabel as nib

def default_loader(path):
    return nib.load(path)

clas myDataset(data.Dataset):

    def _init_(self,root,transform=None,target_transform=None,
               loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root,class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root))
        
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def _getitem_(self,index):
        img,target = self.imgs[index]

myDataset = myDataset()
imgLoader = data.Dataset(myDataset(root = img_path,label = label_path,                

'''


    
#def train(model, img_path,label_path,criterion, optimizer,scheduler,num_epochs,batch_size,num_patches,patch_size):
since = time.time()

best_model_wts = net.state_dict()
best_acc = 0.0
    
imgs, labels = listdir(img_path),listdir(label_path)
for epoch in range(num_epochs): #In every epoch, go through all subjects
    for imgname,labelname in zip(imgs,labels): #For every subject
        img_patches,label_patches = make_training_samples(os.path.join(img_path,imgname),os.path.join(label_path,labelname),num_patches,patch_size)
        #Wrap them in tensors
        img_patches = torch.FloatTensor(np.array(img_patches.astype('float'),dtype=np.int64))
        label_patches = torch.FloatTensor(np.array(label_patches.astype('float'),dtype=np.int64))
        img_patches = img_patches.unsqueeze(1) #insert dimension for input channel
        label_patches = label_patches.unsqueeze(1)

        #Phase: training
        next = 0                
        for i in xrange(0,len(img_patches),batch_size):#Go through every batch
            running_loss = 0.0
            running_corrects = 0
            if i != 0:
                batch_indices = i
                    #Iterate over data in a batch
                imgdata = img_patches[next:batch_indices]
                labels = label_patches[next:batch_indices]

                        # wrap them in Variable
                if use_gpu:
                   imgdata = Variable(imgdata.cuda())
                   labels = Variable(labels.cuda())
                else:
                   imgdata,labels = Variable(imgdata),Variable(labels) 
 #                   if imgdata.size() == label.size():
                        #zero the parameter gradients
                optimizer.zero_grad()

                print('imgdata.sum(): ',imgdata.sum())
                print('imgdata.size: ',imgdata.size())
                print('labels.size: ', labels.size())
                output = net(imgdata)
                _, preds = torch.max(output.data,1)
                loss = criterion(output,labels)
                print('loss ',loss.data[0])
                       
                loss.backward()
                optimizer.step()
                next = batch_indices
                print('next ', next)
                         # statistics
                running_loss += loss.data[0]
                preds = preds.type(torch.FloatTensor)
                running_corrects += torch.sum(preds == label.data)
                
            subepoch_loss = running_loss / batch_size #training dataset size
            print('sbuepoch_loss {:.02f}'.format(subepoch_loss))

         #Phase: validation
         

         
 #          epoch_acc = running_corrects /  # validation dataset size

    print (epoch_loss)
           
           

                    
        
            

#model_ft = train(net,img_path,label_path,criterion,optimizer,scheduler,num_epochs,batch_size,num_patches,patch_size)

        
      
        

