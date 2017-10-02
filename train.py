from __future__ import print_function
import torch.utils.data as data
from os import listdir
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as t
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data as data_utils

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
num_classes = 1
net = YLNet3D(num_classes)
net.train()

#------------------ Datasets ------------------------------
img_path = '/Users/Elaine/desktop/MICCAI/Training'
label_path = '/Users/Elaine/desktop/MICCAI/Labels'
num_patches = 1000
patch_size = [27,27,27]


#----------- Hyperparameters for training -------------------
optimizer = optim.SGD(net.parameters(),lr=0.0001, momentum=0.9)
scheduler = MultiStepLR(optimizer,milestones=[10,20],gamma=0.1)#Set the learning rate of each parameter group to the initial lr decayed by gamma once
                                                             #the number of epoch reaches one of the milestones.
criterion = nn.MSELoss()
use_gpu = torch.cuda.is_available()
num_epochs = 5
batch_size = 5

#-------------------------------------------------------------               
                      

    
#def train(model, img_path,label_path,criterion, optimizer,scheduler,num_epochs,batch_size,num_patches,patch_size):
since = time.time()

best_model_wts = net.state_dict()
best_acc = 0.0
    
imgs, labels = listdir(img_path),listdir(label_path)
for epoch in range(num_epochs): #In every epoch, go through all subjects
    for imgname,labelname in zip(imgs,labels): #For every subject
        img_patches,label_patches = make_training_samples(os.path.join(img_path,imgname),os.path.join(label_path,labelname),num_patches,patch_size)


        next = 0 #next batch           
        for i in xrange(0,len(img_patches),batch_size):#Go through every batch
            running_loss = 0.0
            running_corrects = 0
            if i != 0:
                batch_indices = i
                    #Iterate over data in a batch
                for j in xrange(next,batch_indices):
                    img, label = img_patches[j], label_patches[j]
                    imgdata = torch.FloatTensor(np.array(img,dtype=np.int64)) #Wrap them in tensor
                    imgdata = imgdata.unsqueeze(0).unsqueeze(0)#insert dimension
                    label = torch.FloatTensor(np.array(label,dtype=np.int64))
                    label = label.unsqueeze(0).unsqueeze(0)#insert dimension

                        # wrap them in Variable
                    if use_gpu:
                       imgdata = Variable(imgdata.cuda())
                       label = Variable(label.cuda())
                    else:
                       imgdata,label = Variable(imgdata),Variable(label) 
                    if imgdata.size() == label.size():
                        #zero the parameter gradients
                       optimizer.zero_grad()

                        #forward
 #                      print('imgdata ', imgdata)
                       output = net(imgdata)
                       _, preds = torch.max(output.data,1)
                       loss = criterion(output,label)
                       print('loss ',loss.data[0])
                        #backward + optimize only if in training phase
                       # if phase == 'train':
                       loss.backward()
                       optimizer.step()

                         # statistics
                       running_loss += loss.data[0]
                       preds = preds.type(torch.FloatTensor)
                       running_corrects += torch.sum(preds == label.data)
                next = batch_indices
                print('next ', next)
                subepoch_loss = running_loss / batch_size #training dataset size
                print('sbuepoch_loss ', subepoch_loss)
 #          epoch_acc = running_corrects /  # validation dataset size

    print (epoch_loss)
           
           

                    
        
            

#model_ft = train(net,img_path,label_path,criterion,optimizer,scheduler,num_epochs,batch_size,num_patches,patch_size)

         
        
        
    

    
