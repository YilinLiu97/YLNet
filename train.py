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

#input = Variable(torch.randn(1,1,256,256))
#net = YLNet2D()
#output = net(input)
#print(output)

net = YLNet3D()
net.train()

#------------------ Datasets ------------------------------
img_path = '/Users/Elaine/desktop/MICCAI/Testing'
label_path = '/Users/Elaine/desktop/MICCAI/Labels'
num_patches = 1000
patch_size = [27,27,27]


#----------- Hyperparameters for training -------------------
optimizer = optim.SGD(net.parameters(),lr=0.001, momentum=0.9)
scheduler = MultiStepLR(optimizer,milestones=[10,20],gamma=0.1)#Set the learning rate of each parameter group to the initial lr decayed by gamma once
                                                             #the number of epoch reaches one of the milestones.
criterion = nn.CrossEntropyLoss()
use_gpu = torch.cuda.is_available()
num_epochs = 5
batch_size = 5
model_ft = train(net,img_path,label_path,criterion,optimizer,scheduler,num_epochs,batch_size,num_patches,patch_size)
#-------------------------------------------------------------               
                      

    
def train(model, img_path,label_path,criterion, optimizer,scheduler,num_epochs,batch_size,num_patches,patch_size):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    imgs, labels = listdir(img_path),listdir(label_path)
    for epoch in range(num_epochs): #In every epoch, go through all subjects
        for imgname,labelname in zip(imgs,labels): #For every subject
            img_patches,label_patches = make_training_samples(os.path.join(img_path,imgname),os.path.join(label_path,labelname),num_patches,patch_size)


            running_loss = 0.0
            running_corrects = 0
            
            for i in xrange(0,len(img_patches),batch_size):#Go through every batch
                if i != 0:
                    batch_indices = i
                    #Iterate over data in a batch
                    for j in xrange(0,batch_indices):
                        img, label = img_patches[j], label_patches[j]
                        imgdata = torch.FloatTensor(np.array(img,dtype=np.int64)) #Wrap them in tensor
                        label = torch.FloatTensor(np.array(label,atype=np.int64))

                        # wrap them in Variable
                        if use_gpu:
                            imgdata = Variable(imgdata.cuda())
                            label = Variable(label.cuda())
                        else:
                            imgdata,labels = Variable(imgdata),Variable(label) 

                        #zero the parameter gradients
                        optimizer.zero_grad()

                        #forward
                        output = model(imgdata)
                        _, preds = torch.max(output.data,1)
                        loss = criterion(output,label)

                        #backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                         # statistics
                         running_loss += loss.data[0]
                         running_corrects += torch.sum(preds == label.data)
                         
           epoch_loss = running_loss / (len(imgname)*num_patches) #training dataset size
 #          epoch_acc = running_corrects /  # validation dataset size
           print '{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss)
           
           

                    
        
               
    return model



         
        
        
    

    

