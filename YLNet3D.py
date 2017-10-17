import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as Funct
import numpy as np

import torch.optim as optim #Define a loss function and optimizer

class MyLogSoftmax(nn.Module):

    def forward(self, input_):
        batch_size = input_.size()[0]
        output_ = torch.stack([Funct.log_softmax(input_[i]) for i in range(batch_size)], 0)
        return output_
    
   
class YLNet3D(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(YLNet3D, self).__init__()
        
        #Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        #nn.Sequential is a container for Module.Module will work in order during forward
        self.encoder_1 = nn.Sequential(
            nn.Conv3d(in_channels,25,3,padding=2,dilation=2),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) #first encoder


        self.encoder_2 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=2,dilation=2),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) #second encoder


        self.encoder_3 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=2,dilation=2),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) #third encoder

        
        self.maxpool_1 = nn.MaxPool3d(2, stride=2, return_indices=True) #create masks
        self.maxpool_2 = nn.MaxPool3d(2, stride=2, return_indices=True)
        self.maxpool_3 = nn.MaxPool3d(2, stride=2, return_indices=True)
        
        self.unpool_1 = nn.MaxUnpool3d(2, stride=2) 
        self.unpool_2 = nn.MaxUnpool3d(2, stride=2)
        self.unpool_3 = nn.MaxUnpool3d(2, stride=2)

        
        self.decoder_1 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=1),#the number of kernels can be easily changed here
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) # first decoder


        self.decoder_2 = nn.Sequential(
            nn.Conv3d(50,25,3,padding=1),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) # second decoder

        self.decoder_3 = nn.Sequential(
            nn.Conv3d(50,25,3,padding=1),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) # third decoder

        self.conv_4 = nn.Sequential(
            nn.Conv3d(50,out_channels,1,padding=0) #the number of output columns depend on the number of classes
            ) # last conv layer

    def forward(self,x): #x is an input that needs to go forward
        size_1 = x.size()

        en_1 = self.encoder_1(x)

        
        en1_maxpool,indices_1 = self.maxpool_1(en_1) 

        size_2 = en1_maxpool.size()
        en_2 = self.encoder_2(en1_maxpool)
        en2_maxpool,indices_2 = self.maxpool_2(en_2)

        size_3 = en2_maxpool.size()
        en_3 = self.encoder_3(en2_maxpool)
        en3_maxpool,indices_3 = self.maxpool_3(en_3)


        de_1 = self.decoder_1(en3_maxpool)
        de1_unpool = self.unpool_1(de_1, indices_3, output_size=size_3) #transfer of indices
        merge1_3 = torch.cat((de1_unpool,en_3), 1)

        de_2 = self.decoder_2(merge1_3)
        de2_unpool = self.unpool_2(de_2, indices_2, output_size=size_2) #transfer of indices
        merge2_2 = torch.cat((de2_unpool,en_2), 1)

        de_3 = self.decoder_3(merge2_2)
        de3_unpool = self.unpool_3(de_3, indices_1, output_size=size_1) #transfer of indices
        merge3_1 = torch.cat((de3_unpool,en_1), 1)


        conv_4 = self.conv_4(merge3_1)


        logsoftmax = MyLogSoftmax()


        return logsoftmax(conv_4) 
'''
if __name__ == "__main__":
    
    criterion = nn.CrossEntropyLoss()
    input = Variable(torch.randn(10,1,27,27,27).float())
    label = Variable(torch.LongTensor(10,27,27,27).random_(9))
    net = YLNet3D(1,9)
    params = list(net.parameters())

    optimizer = optim.SGD(net.parameters(),lr=0.0001, momentum=0.9)
    output = net(input)
    output = output.view(-1,9)
    label = label.view(-1)

      
    
   # print output

#params = list(net.parameters())
#print(len(params))
#print(params[0].size())

'''
