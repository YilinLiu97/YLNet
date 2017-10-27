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
    
   
class YLNet4(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(YLNet4, self).__init__()
        
        #Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        #nn.Sequential is a container for Module.Module will work in order during forward
        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_channels,25,3,padding=2,dilation=2),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) #conv 1st
        self.encoder_1 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=2,dilation=2),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) #first encoder

        self.conv_2 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=2,dilation=2),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) #conv 2nd
        self.encoder_2 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=2,dilation=2),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) #second encoder

        self.conv_3 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=2,dilation=2),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) #conv 3rd

        self.encoder_3 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=2,dilation=2),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) #third encoder

        self.conv_4 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=2,dilation=2),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) #conv 2nd
        self.encoder_4 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=2,dilation=2),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) #fourth encoder

        
        self.maxpool_1 = nn.MaxPool3d(2, stride=2, return_indices=True) #create masks
        self.maxpool_2 = nn.MaxPool3d(2, stride=2, return_indices=True)
        self.maxpool_3 = nn.MaxPool3d(2, stride=2, return_indices=True)
        self.maxpool_4 = nn.MaxPool3d(2, stride=2, return_indices=True)
        
        self.unpool_1 = nn.MaxUnpool3d(2, stride=2) 
        self.unpool_2 = nn.MaxUnpool3d(2, stride=2)
        self.unpool_3 = nn.MaxUnpool3d(2, stride=2)
        self.unpool_4 = nn.MaxUnpool3d(2, stride=2)

        self.conv_5 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=1),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) #conv 2nd
        self.decoder_1 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=1),#the number of kernels can be easily changed here
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) # first decoder

        self.conv_6 = nn.Sequential(
            nn.Conv3d(50,25,3,padding=1),
            nn.BatchNorm3d(25),
            nn.ReLU())
        
        self.decoder_2 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=1),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) # second decoder

        self.conv_7 = nn.Sequential(
            nn.Conv3d(50,25,3,padding=1),
            nn.BatchNorm3d(25),
            nn.ReLU())
        
        self.decoder_3 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=1),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) # third decoder

        self.conv_8 = nn.Sequential(
            nn.Conv3d(50,25,3,padding=1),
            nn.BatchNorm3d(25),
            nn.ReLU())
        
        self.decoder_4 = nn.Sequential(
            nn.Conv3d(25,25,3,padding=1),
            nn.BatchNorm3d(25),
            nn.ReLU()
            ) # third decoder
        #xavier weights initialization
 #       nn.init.xavier_uniform(self.decoder_3[0].weight)
        self.conv_9 = nn.Sequential(
            nn.Conv3d(50,out_channels,1,padding=0) #the number of output columns depend on the number of classes
            ) # last conv layer

    def forward(self,x): #x is an input that needs to go forward
        size_1 = x.size()
        print('size_1 ',size_1)

        conv_1 = self.conv_1(x)
        en_1 = self.encoder_1(conv_1)
        en1_maxpool,indices_1 = self.maxpool_1(en_1) 

        size_2 = en1_maxpool.size()
        conv_2 = self.conv_2(en1_maxpool)
        print('size_2 ',size_2)
        en_2 = self.encoder_2(conv_2)
        en2_maxpool,indices_2 = self.maxpool_2(en_2)

        size_3 = en2_maxpool.size()
        conv_3 = self.conv_3(en2_maxpool)
        print('size_3 ',size_3)
        en_3 = self.encoder_3(conv_3)
        en3_maxpool,indices_3 = self.maxpool_3(en_3)

        size_4 = en3_maxpool.size()
        conv_4 = self.conv_4(en3_maxpool)
        print('size_4 ',size_4)
        en_4 = self.encoder_4(conv_4)
        en4_maxpool,indices_4 = self.maxpool_4(en_4)
        print('en4_maxpool.size() ',en4_maxpool.size())
        
        conv_5 = self.conv_5(en4_maxpool)
        print('conv_5.size() ',conv_5.size())
        de_1 = self.decoder_1(conv_5)
        print('de_1 ',de_1.size())
        de1_unpool = self.unpool_1(de_1, indices_4, output_size=size_4) #transfer of indices
        print('de1_unpool.size() ',de1_unpool.size())
        merge1_4 = torch.cat((de1_unpool,en_4), 1)

        conv_6 = self.conv_6(merge1_4)
        print('conv_6.size() ',conv_6.size())
        de_2 = self.decoder_2(conv_6)
        print('de_2 ',de_2.size())
        de2_unpool = self.unpool_2(de_2, indices_3, output_size=size_3) #transfer of indices
        merge2_3 = torch.cat((de2_unpool,en_3), 1)

        conv_7 = self.conv_7(merge2_3)
        de_3 = self.decoder_3(conv_7)
        print('de_3 ',de_3.size())
        de3_unpool = self.unpool_3(de_3, indices_2, output_size=size_2) #transfer of indices
        merge3_2 = torch.cat((de3_unpool,en_2), 1)

        conv_8 = self.conv_8(merge3_2)
        de_4 = self.decoder_4(conv_8)
        print('de_4 ',de_4.size())
        de4_unpool = self.unpool_4(de_4, indices_1, output_size=size_1) #transfer of indices
        merge4_1 = torch.cat((de4_unpool,en_1), 1)

        conv_9 = self.conv_9(merge4_1)
        print('conv_4 ',conv_9.size())

        logsoftmax = MyLogSoftmax()
        return logsoftmax(conv_9)


