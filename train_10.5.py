from os import listdir
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from YLNet3D import *
from sampling import make_training_samples
import os
import time
import numpy as np
import math
import nibabel as nib

#----------------------- GPU -------------------------------
use_gpu = torch.cuda.is_available()
if use_gpu:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
print "Number of GPUs: ", torch.cuda.device_count()
print "Current device:", torch.cuda.current_device()

#-------------------- Net --------------------------------
in_channels = 1
out_channels = 9
net = YLNet3D(in_channels,out_channels)
if use_gpu:
	net = net.cuda()
net.train()

#------------------ Datasets ------------------------------
# img_path = '/Users/Elaine/desktop/MICCAI_old/Training'
img_path = '/home/yilin/MICCAI/Training'
# label_path = '/Users/Elaine/desktop/MICCAI_old/LabelsCorrected'
label_path = '/home/yilin/MICCAI/LabelsCorrected'

num_patches = 100
w_train = 0.7
patch_size = [50,50,50]

#----------- Hyperparameters for training -------------------
optimizer = optim.SGD(net.parameters(),lr=0.000001, momentum=0.9)
scheduler = MultiStepLR(optimizer,milestones=[10,20],gamma=0.1)#Set the learning rate of each parameter group to the initial lr decayed by gamma once
                                                             #the number of epoch reaches one of the milestones.
criterion = nn.MSELoss()

num_epochs = 5
batch_size = 10

#-------------------------------------------------------------

def makeDatasets(img_path,label_path,w_train):
    '''
    Split dataset
    '''
    labels,trainset,valset = [], [], []
    
    imgs, labels = listdir(img_path),listdir(label_path)
    trainset = imgs[:int(len(imgs)*w_train)]
    labels_tr = labels[:int(len(labels)*w_train)]
    valset = imgs[int(len(imgs)*w_train):]
    labels_val = labels[int(len(labels)*w_train):]

    return np.array(trainset), np.array(labels_tr), np.array(valset), np.array(labels_val)

    
#def train(model, img_path,label_path,criterion, optimizer,scheduler,num_epochs,batch_size,num_patches,patch_size):
since = time.time()

best_model_wts = net.state_dict()
best_acc = 0.0
    

trainset,labels_tr,valset,labels_val = makeDatasets(img_path,label_path,w_train)

for epoch in range(num_epochs): #In every epoch, go through all subjects

    # Each epoch has a training and validation phase
    for phase in ['train','val']:
        if phase == 'train':
            Dataset = trainset
            Labels = labels_tr
        else:
            dataset = valset
            Labels = labels_val

        running_loss = 0.0
        running_corrects = 0
        
        for imgname,labelname in zip(Dataset,Labels): #For every subject
            affine,img_patches,label_patches = make_training_samples(os.path.join(img_path,imgname),os.path.join(label_path,labelname),num_patches,patch_size)
            #Wrap them in tensors
            img_patches = torch.FloatTensor(np.array(img_patches.astype('float'),dtype=np.int64))
            label_patches = torch.FloatTensor(np.array(label_patches.astype('float'),dtype=np.int64))
            img_patches = img_patches.unsqueeze(1) #insert dimension for input channel
            label_patches = label_patches.unsqueeze(1)
        
            next = 0
            batch_indices = 0
            for i in xrange(0,len(img_patches),batch_size):#Go through every batch

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
                    
                    print('imgdata.sum(): ',imgdata.sum())
                    print('imgdata.size: ',imgdata.size())
                    
                    optimizer.zero_grad()
                    # forward
                    output = net(imgdata)
                    labels = labels.expand_as(output)  #expand the 2nd dimension: output channels --> num_classes
                    print('output.size: ',output.size())
                    print('labels.size: ', labels.size())
                    _, preds = torch.max(output.data,1) #_ - maximum prob values, preds - argmax

 #                   nib.save(nib.Nifti1Image(np.int32(preds.numpy()),affine,os.path.join('/Users/Elaine/desktop','preds.nii')))
                    
                    loss = criterion(output,labels)
                    print('Loss per batch ',loss.data[0])
 #                   nib.save(preds,'preds.nii')
                     # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    next = batch_indices
                    print('next ', next)
                    
                    # statistics
                    running_loss += loss.data[0]
                            
    epoch_loss = running_loss / len(Dataset)*num_patches
    '''
    epoch_acc = running_corrects / len(Dataset)*num_patches
    
    print('{} Loss: {:.4f} '.format(phase,epoch_loss))

    #deep copy the model
    if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = model.state_dict()

print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))
    
# load best model weights
model.load_state_dict(best_model_wts)

         
'''
    '''
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
      
        

