import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as Funct

import torch.optim as optim #Define a loss function and optimizer

class YLNet(nn.Module):
    def __init__(self):
        super(YLNet, self).__init__()

        #Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        #nn.Sequential is a container for Module.Module will work in order during forward
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(1,64,7,padding=3,dilation=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
            ) #first encoder

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(64,64,7,padding=3,dilation=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
            ) #second encoder

        self.encoder_3 = nn.Sequential(
            nn.Conv2d(64,64,7,padding=3,dilation=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
            ) #third encoder

        
        self.maxpool_1 = nn.MaxPool2d(2, stride=2, return_indices=True) #create masks
        self.maxpool_2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.maxpool_3 = nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.unpool_1 = nn.MaxUnpool2d(2, stride=2) 
        self.unpool_2 = nn.MaxUnpool2d(2, stride=2)
        self.unpool_3 = nn.MaxUnpool2d(2, stride=2)

        
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(64,64,7,padding=3),
            nn.BatchNorm2d(64),
            nn.PReLU()
            ) # first decoder

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(128,64,7,padding=3),
            nn.BatchNorm2d(64),
            nn.PReLU()
            ) # second decoder

        self.decoder_3 = nn.Sequential(
            nn.Conv2d(128,64,7,padding=3),
            nn.BatchNorm2d(64),
            nn.PReLU()
            ) # third decoder

        self.conv_4 = nn.Sequential(
            nn.Conv2d(128,64,7,padding=3), #the number of output columns depend on
            nn.BatchNorm2d(64),            #the situations
            nn.PReLU()
            ) # last conv layer


    def forward(self,x): #x is an input that needs to go forward
        size_1 = x.size() #256x256x1
        en_1 = self.encoder_1(x) #256x256x64
        en1_maxpool,indices_1 = self.maxpool_1(en_1) #128x128x64
        
        size_2 = en1_maxpool.size()#(128x128x128)
        en_2 = self.encoder_2(en1_maxpool) #128x128x64
        en2_maxpool,indices_2 = self.maxpool_2(en_2)#64x64x64

        size_3 = en2_maxpool.size()#(64x64x64)
        en_3 = self.encoder_3(en2_maxpool) #64x64x64
        en3_maxpool,indices_3 = self.maxpool_3(en_3)#32x32x64

        de_1 = self.decoder_1(en3_maxpool)#32x32x64
        de1_unpool = self.unpool_1(en3_maxpool, indices_3, output_size=size_3) #transfer of indices
        merge1_3 = torch.cat((de1_unpool,en_3), 1)#64x64x128
        print 'merge1_3 ', merge1_3.size()
        de_2 = self.decoder_2(merge1_3)#64x64x64
        de2_unpool = self.unpool_2(en2_maxpool, indices_2, output_size=size_2) #transfer of indices
        merge2_2 = torch.cat((de2_unpool,en_2), 1)#128x128x128
        print 'merge2_2 ', merge2_2.size()
        de_3 = self.decoder_3(merge2_2)#128x128x64
        de3_unpool = self.unpool_3(en1_maxpool, indices_1, output_size=size_1) #transfer of indices
        merge3_1 = torch.cat((de3_unpool,en_1), 1)#256x256x128
        print 'merge3_1 ', merge3_1.size()
        conv_4 = self.conv_4(merge3_1)#256x256x64

        output = Funct.softmax(conv_4)

        return output

if __name__ == "__main__":     
    input = Variable(torch.randn(1, 1, 256,256))
    net = YLNet()
    output = net(input)
#params = list(net.parameters())
#print(len(params))
#print(params[0].size())

