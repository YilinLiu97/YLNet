from os import listdir
import torch
from torch.autograd import Variable

import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torch.optim.lr_scheduler import MultiStepLR,ReduceLROnPlateau

from deepYLNet import *
from YLNet4_LK import *
from YLNet4 import *
from YLNet3D import *
from sampling import *
import os
import time
import numpy as np
import math
import nibabel as nib

import pdb


#----------------------- GPU -------------------------------

use_gpu = torch.cuda.is_available()
if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
print "Number of GPUs: ", torch.cuda.device_count()
print "Current device:", torch.cuda.current_device()

#use_gpu = 0
#-------------------- Net --------------------------------
in_channels = 1
num_classes = 4
net = deepYLNet(in_channels,num_classes)

nn.init.xavier_uniform(net.encoder_1[0].weight)
nn.init.xavier_uniform(net.encoder_2[0].weight)
nn.init.xavier_uniform(net.encoder_3[0].weight)
nn.init.xavier_uniform(net.decoder_1[0].weight)
nn.init.xavier_uniform(net.decoder_2[0].weight)
nn.init.xavier_uniform(net.decoder_3[0].weight)
nn.init.xavier_uniform(net.conv_4[0].weight)

#------------------ Datasets ------------------------------
use_gpu = torch.cuda.is_available()
img_path = '/home/yilin/Dataset-brain/Training'
label_path = '/home/yilin/Dataset-brain/Labels'

num_patches = 6000
w_train = 0.9
patch_size = [25,25,25]

#----------- Hyperparameters for training -------------------
optimizer = optim.Adam(net.parameters(),lr=0.01,weight_decay=0)
scheduler = MultiStepLR(optimizer, milestones=[5,20,35,40,60,80,100], gamma=0.1) #MultiStepLR(optimizer,milestones=[10,20],gamma=0.1)#Set the learning rate of each parameter group to the initial lr decayed by gamma once
#scheduler = ReduceLROnPlateau(optimizer,'min')                                                            #the number of epoch reaches one of the milestones.
criterion = nn.CrossEntropyLoss(Variable(torch.FloatTensor([0,0.1,0.45,0.45])).cuda())
num_epochs = 210
batch_size = 64
clip_norm = 1
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

    
def train(net, img_path,label_path,criterion,scheduler,num_epochs,batch_size,num_patches,patch_size,num_classes,w_train):
    since = time.time()

    best_model_wts = net.state_dict()
    best_acc = 520667
    lowest_loss = 1.012
    iteration = 0
    ok = 0
    trainset,labels_tr,valset,labels_val = makeDatasets(img_path,label_path,w_train)
 
    for epoch in xrange(0,num_epochs): #In every epoch, go through all subjects
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                Dataset = trainset
                Labels = labels_tr
                scheduler.step()
                net.train(True)
            else:
                net.train(False)
                Dataset = valset
                Labels = labels_val
     
            running_loss = 0.0
            running_corrects = 0
           
            for imgname,labelname in zip(Dataset,Labels): #For every subject
                affine,img_patches,label_patches = make_training_samples(os.path.join(img_path,imgname),os.path.join(label_path,labelname),num_patches,patch_size,num_classes)
                datasetSize = len(img_patches)
    
                #Wrap them in tensors
                img_patches = torch.FloatTensor(np.array(img_patches.astype('float'),dtype=np.int64))
                label_patches = torch.FloatTensor(np.array(label_patches.astype('int'),dtype=np.int64))
                img_patches = img_patches.unsqueeze(1) #insert dimension for input channel

                next = 0
                batch_indices = 0
                num_batches = 0
	
                for i in xrange(0,len(img_patches),batch_size): #Go through every batch
		    loss = 0
                    if i != 0:
                        batch_indices = i
                        #Iterate over data in a batch
                        imgdata = img_patches[next:batch_indices]
                        labels = label_patches[next:batch_indices]
       
                        if len(imgdata) == len(labels):
                            # wrap them in Variable
                            if use_gpu:
                               imgdata = Variable(imgdata.cuda())
                               labels = Variable(labels.cuda())
                            else:
                               imgdata,labels = Variable(imgdata),Variable(labels)
     
                            
                            # forward
                            output = net(imgdata)
                        
                            output = output.view(-1,num_classes)
                            labels = labels.type(torch.LongTensor).view(-1).cuda()
                            loss = criterion(output,labels)
                          

                            #zero the parameter gradients
                            optimizer.zero_grad()
                            _,preds = torch.max(output.data,1)
                            corrects_perBatch = torch.sum(preds == labels.data)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                clip_grad_norm(net.parameters(),1)
  
                             	torch.save(net.state_dict(),'/home/yilin/YLNet/weights_10.23(weighted+MoreKs).pt')
 	      
                                optimizer.step()
                                iteration = iteration + 1
                         
       #                         print('{} Iteration: {: d} LossPerBatch: {:.4f}'.format(phase,iteration,loss.data[0]))
                                 
                                             
                            if phase == 'val'and ok == 1:
                                #labels = labels.view(batch_size,1,patch_size[0],patch_size[1],patch_size[2]) #unfold the matrix
                               dice = computeDice(preds.cpu().numpy(),(labels.data).cpu().numpy())
                               #print('Dice score: ', dice)
				
	                    
		            running_loss += loss.data[0]
			    running_corrects += torch.sum(preds == labels.data)
			    num_batches = num_batches + 1
			    #print('num_batches ',num_batches)
                            next = batch_indices #To next batch
            		  
            if (epoch+1)%3 == 0:
		ok = 1                                                                          
            epoch_loss = running_loss /(len(Dataset)*num_batches)
	    epoch_acc = running_corrects / (len(Dataset)*num_batches)
            print('{} Epoch: {} Loss: {:.4f} Acc: {:.4f} '.format(phase,epoch+1,epoch_loss,epoch_acc))

        #deep copy the model
            if phase == 'val'and best_acc < epoch_acc:
               best_acc = epoch_acc
               print('Current best accuracy is: ',best_acc)
#            best_model_wts = net.state_dict()
               torch.save(net.state_dict(),'/home/yilin/YLNet/bestWeights_10.25(bestAcc).pt')
           # if phase == 'val' and best_acc < epoch_acc:
	    #   best_acc = epoch_acc
             #  print('Current best accuracy: ',best_acc) 
               #torch.save(net.state_dict(),'home/yilin/YLNet/bestAcc.pt')           
 
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    best_model_wts = net.state_dict()
    return net

def saveCheckpoint(state,is_best,filename='checkpoint.pth.tar'):
    torch.save(state,filename)
    if is_best:
        shutil.copyfile(filename,'model_best.pth.tar')

    
#net.load_state_dict(torch.load('/home/yilin/YLNet/bestWeights_10.25(bestAcc).pt'))
#net.load_state_dict(torch.load('/home/yilin/YLNet/bestWeights_10.22.pt', map_location=lambda storage, loc: storage)) #For cpu
train(net,img_path,label_path,criterion,scheduler,num_epochs,batch_size,num_patches,patch_size,num_classes,w_train)
#affine,img_patches,label_patches = make_training_samples(os.path.join(img_path,'subject205_noskl_mid_s205abcd_superseg_contrasted_path.nii'),os.path.join(label_path,'subject205_RIGHT_all_labels_8bit_path_RightLeftAmygdalaSubfields.nii'),num_patches,patch_size,num_classes
'''
vol = nib.load('/home/yilin/Dataset/Training/subject220_noskl_mid_s220abcd_superseg_contrasted_path.nii')
label = nib.load('/home/yilin/Dataset/Labels/subject218_RIGHT_all_labels_8bit_path_RightLeftAmygdalaNOSUBFIELDS.nii')
affine = vol.affine
coordinates,patches = crop_det_patches(vol,[25,25,25])
#print('patches.shape ',patches.shape)
probMaps = np.array(np.zeros(list(vol.get_shape())), dtype="int16")
next = 0
batch_indices= 0
batch_size = 32
for i in xrange(0,len(patches),batch_size):
   if i != 0:
      batch_indices = i
      imgdata = torch.FloatTensor(np.array(patches[next:batch_indices].astype('float'),dtype=np.int64)).cuda().unsqueeze(1).cuda()
     # imgdata = torch.FloatTensor(np.array(patches[next:batch_indices].astype('float'),dtype=np.int64)).unsqueeze(1)
      ## Garble up imgdata shuffle it or something like that. permute the tensor.
      output = net(Variable(imgdata))
      print('output.shape ', output.size())
      print('imgdata.shape ',imgdata.size())
      # _,preds = torch.max(output,1)
      _,preds = torch.max(output,1)
      #preds = torch.ones(32,1,25,25,25)
      print('preds.shape ',preds.size())
      #preds = preds.view(-1,25,25,25).unsqueeze(1)
      #print('preds.shape ',preds.size())
      # print preds.data.cpu().numpy()
      #probMaps = stitch_Patches(probMaps,coordinates[next:batch_indices],imgdata.cpu().numpy(),batch_size)
      probMaps = stitch_Patches(probMaps,coordinates[next:batch_indices],preds.data.cpu().numpy(),batch_size)
   next = batch_indices
saveImage(probMaps,affine)

'''

