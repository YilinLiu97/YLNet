from os import listdir
import torch
from torch.autograd import Variable

import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from torch.optim.lr_scheduler import MultiStepLR,ReduceLROnPlateau

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

#-------------------- Net --------------------------------
in_channels = 1
num_classes = 3
net = YLNet3D(in_channels,num_classes)

nn.init.xavier_uniform(net.encoder_1[0].weight)
nn.init.xavier_uniform(net.encoder_2[0].weight)
nn.init.xavier_uniform(net.encoder_3[0].weight)
nn.init.xavier_uniform(net.decoder_1[0].weight)
nn.init.xavier_uniform(net.decoder_2[0].weight)
nn.init.xavier_uniform(net.decoder_3[0].weight)
nn.init.xavier_uniform(net.conv_4[0].weight)
'''
net.encoder_1[0].weight.data.copy_(torch.eye(3))
net.encoder_2[0].weight.data.copy_(torch.eye(3))
net.encoder_3[0].weight.data.copy_(torch.eye(3))
net.decoder_1[0].weight.data.copy_(torch.eye(3))
net.decoder_2[0].weight.data.copy_(torch.eye(3))
net.decoder_3[0].weight.data.copy_(torch.eye(3))
'''
#------------------ Datasets ------------------------------
use_gpu = torch.cuda.is_available()
img_path = '/Users/Elaine/desktop/Dataset/Training'
label_path = '/Users/Elaine/desktop/Dataset/Labels'

num_patches = 1000
w_train = 0.7
patch_size = [25,25,25]

#----------- Hyperparameters for training -------------------
optimizer = optim.Adam(net.parameters(),lr=0.0005)
scheduler = MultiStepLR(optimizer,milestones=[2,3],gamma=0.1)#Set the learning rate of each parameter group to the initial lr decayed by gamma once
#scheduler = ReduceLROnPlateau(optimizer,'min')                                                            #the number of epoch reaches one of the milestones.
criterion = nn.CrossEntropyLoss()
num_epochs = 20
batch_size = 32
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

    
def train(model, img_path,label_path,criterion,scheduler,num_epochs,batch_size,num_patches,patch_size,num_classes,clip_norm):
    since = time.time()

    best_model_wts = net.state_dict()
    best_acc = 0.0
    iteration = 0
    
    trainset,labels_tr,valset,labels_val = makeDatasets(img_path,label_path,w_train)
   
    for epoch in xrange(0,num_epochs): #In every epoch, go through all subjects
        # Each epoch has a training and validation phase
        for phase in ['val','train']:
            if phase == 'train':
                print('Epoch ',epoch+1)
                Dataset = trainset
                Labels = labels_tr
                scheduler.step()
                net.train(True)
            else:
                net.train(False)
                Dataset = valset
    
                Labels = labels_val
            print('phase: ', phase)
            running_loss = 0.0
            running_corrects = 0
            
            for imgname,labelname in zip(Dataset,Labels): #For every subject
                print('imgname ',imgname)
                affine,img_patches,label_patches = make_training_samples(os.path.join(img_path,imgname),os.path.join(label_path,labelname),num_patches,patch_size,num_classes)
                datasetSize = len(img_patches)
                vol = nib.load(os.path.join(img_path,imgname))
                label = nib.load(os.path.join(label_path,labelname))
                print('imagname',imgname)
                print('vol.size(): ',nib.load(os.path.join(img_path,imgname)).get_shape())
                print('label.size(): ',nib.load(os.path.join(label_path,labelname)).get_shape())
                #Wrap them in tensors
                img_patches = torch.FloatTensor(np.array(img_patches.astype('float'),dtype=np.int64))
                label_patches = torch.FloatTensor(np.array(label_patches.astype('int'),dtype=np.int64))
                img_patches = img_patches.unsqueeze(1) #insert dimension for input channel
                print('label_patches ',label_patches.size())
                print('img_patches ', img_patches.size())

                next = 0
                batch_indices = 0
                
                for i in xrange(0,len(img_patches),batch_size):#Go through every batch
                    loss = 0
                    if i != 0:
                        batch_indices = i
                        #Iterate over data in a batch
                        imgdata = img_patches[next:batch_indices]
                        labels = label_patches[next:batch_indices]
                        print('minimum imgdata ', np.min(imgdata.numpy()))
                        print('minimum label ', np.min(label_patches.numpy()))
                        if len(imgdata) == len(labels):
                            # wrap them in Variable
                            if use_gpu:
                               imgdata = Variable(imgdata.cuda())
                               labels = Variable(labels.cuda())
                            else:
                               imgdata,labels = Variable(imgdata),Variable(labels)
     
                            print('imgdata.size: ',imgdata.size())
                            
                            # forward
                            output = net(imgdata)
                            print('output.size: ',output.size())
                            output = output.view(-1,num_classes)
                            labels = labels.type(torch.LongTensor).view(-1)
                            loss = criterion(output,labels)
                            print('loss has 0? ',np.any(loss==0))
 #                           print('net.encoder_1[0].weight ', net.encoder_1[0].weight)

                            print('output.size(folded): ',output.size())
                            print('labels.size: ', labels.size())
                            print('smallest output value ',np.min(output.data.numpy())) 
                            
        
                            #zero the parameter gradients
                            optimizer.zero_grad()
           
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
 #                               clip_grad_norm(net.parameters(),2)

                                for g in net.parameters():
#                                    print('before clipping ',g.grad.data)
                                    g.grad.data.clamp_(-3,2)
 #                                   print('after clipping ',g.grad.data)
 #                               print('before:encoder_1.gradient: ',net.encoder_1[0].weight.grad.data[0])
#                                print('before:conv_4.gradient: ',net.conv_4[0].weight.grad.data[0])
                                if np.isnan(np.min(net.encoder_1[0].weight.grad.data[0].numpy())) == False:
                                     torch.save(net.state_dict(),'/Users/Elaine/desktop/weights_10.17.pt')
                                else:
                                     for param_group in optimizer.param_groups:
                                         param_group['lr'] = param_group['lr']*0.001
        #                            pdb.set_trace()
                                optimizer.step()
                                iteration = iteration + 1
                                print('after Loss ', loss)
                                print('updated:(min) encoder_1.gradient ',np.min(net.encoder_1[0].weight.grad.data[0].numpy()))
                                print('updated:(min) output.gradient ',np.min(net.conv_4[0].weight.grad.data[0].numpy()))
                                print('{} Iteration: {: d} LossPerBatch: {:.4f}'.format(phase,iteration,loss.data[0]))
                                torch.save(net.state_dict(),'/Users/Elaine/desktop/weights_backup.pt') #copy weights per batch

                                print('Loss per batch: ', loss.data[0])                               
                            
                               # statistics
                                running_loss += loss.data[0]
      
                            if phase == 'val':
                                #labels = labels.view(batch_size,1,patch_size[0],patch_size[1],patch_size[2]) #unfold the matrix
                                _,preds = torch.max(output.data,1)
                                dice = computeDice(preds.numpy(),labels.data.numpy())
                                print('Dice score: ', dice)


                            next = batch_indices #To next batch
                            print('next ', next)
                                                                                      
 #       epoch_loss = running_loss / datasetSize
#        print('{} Loss: {:.4f} Acc:{:.4f}'.format(phase,epoch_loss))
            
        #deep copy the model
#        if phase == 'val':
 #           best_acc = epoch_acc
#            best_model_wts = net.state_dict()
#            torch.save(net.state_dict(),'/Users/Elaine/desktop/bestWeights.pt')
                        
 
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#    print('Best val Acc: {:4f}'.format(best_acc))

    best_model_wts = net.state_dict()
    return net

def save_checkpoint(state,is_best,filename='checkpoint.pth.tar'):
    torch.save(state,filename)
    if is_best:
        shutil.copyfile(filename,'model_best.pth.tar')

    
#net.load_state_dict(torch.load('/Users/Elaine/desktop/weights_retrain.pt'))
train(net,img_path,label_path,criterion,scheduler,num_epochs,batch_size,num_patches,patch_size,num_classes,clip_norm)
#affine,img_patches,label_patches = make_training_samples(os.path.join(img_path,'subject205_noskl_mid_s205abcd_superseg_contrasted_path.nii'),os.path.join(label_path,'subject205_RIGHT_all_labels_8bit_path_RightLeftAmygdalaSubfields.nii'),num_patches,patch_size,num_classes)

