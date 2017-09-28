import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as t
import torch.optim as optim

from YLNet3D import *
from Sampling import getPatchSamples
import os

#input = Variable(torch.randn(1,1,256,256))
#net = YLNet2D()
#output = net(input)
#print(output)

#The input to the forward is an autograd.Variable, so is the output
#train_dataset = DataLoader("path/to/dataset", batch_size=4,shuffle=True,num_workers=4) #Combines a dataset and a sampler,provides single- or multi-process iteration over the dataset
#test_dataset = DataLoader("path/to/dataset",batch_size=4,shffule=True,num_workers=4)
folder_path = '/Users/Elaine/desktop/MICCAI'
net = YLNet3D()
net.train()

train_dataset = getPatchSamples(os.path.join(folder_path,'/Training'))
test_dataset = getPatchSamples(os.path.join(folder_path, '/Testing'))
labels = getPatchSamples(os.path.join(folder_path, '/Testing'))

def train(net, criterion, optimizer,scheduler,num_epochs):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)

        #Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True) #Set model to training mode
            else:
                 model.train(False) #Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

              #Iterate over data
            for data, label in zip(train_dataset,labels):
                #Wrap the input in Variable
                if use_gpu:
                    input = Variable(input.cuda())
                    label = Variable(label.cuda())
                else:
                    input, label = Variable(input),Varialbe(label)
                      
                  # zero the parameter gradients  
                optimizer.zero_grad()

                  #forward
                output = model(input)
                _, preds = torch.max(output.data,1)
                loss = criterion(output,label)

                  # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects/dataset_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
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
        return model


                
                      

         
        
        
    

    

