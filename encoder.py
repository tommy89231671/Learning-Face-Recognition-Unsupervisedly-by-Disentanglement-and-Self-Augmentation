# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:08:12 2018

@author: tommy
"""
import torch.nn as nn
class Encoder(nn.Module):
    
    
    def __init__(self,c_size,z_size):
        super(Encoder, self).__init__()
        nc=3
        ndf=128
        latent_variable_c=c_size
        latent_variable_z=z_size
        #nc->ndf->2ndf->4ndf->8ndf->ndf*8*4*4->latent_variable_size
        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*4, 2*latent_variable_c)#inputsize,outputsize
        self.fc2 = nn.Linear(ndf*8*4, 2*latent_variable_z)
        
        self.fc3 = nn.Sequential(
            nn.Linear(ndf*8*4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10),
            nn.Linear(10, 5))
        #self.fc_c = nn.Linear(ndf*8*4*4, latent_variable_c)
        #self.fc_z = nn.Linear(ndf*8*4*4, latent_variable_z)
        self.leakyrelu = nn.LeakyReLU(0.2)
    '''
    def siamese_forward(self,input1,input2):
        a1 = self.leakyrelu(self.bn1(self.e1(input1)))
        a2 = self.leakyrelu(self.bn2(self.e2(a1)))
        a3 = self.leakyrelu(self.bn3(self.e3(a2)))
        a4 = self.leakyrelu(self.bn4(self.e4(a3)))
        a5 = self.leakyrelu(self.bn5(self.e5(a4)))
        
        b1 = self.leakyrelu(self.bn1(self.e1(input2)))
        b2 = self.leakyrelu(self.bn2(self.e2(b1)))
        b3 = self.leakyrelu(self.bn3(self.e3(b2)))
        b4 = self.leakyrelu(self.bn4(self.e4(b3)))
        b5 = self.leakyrelu(self.bn5(self.e5(b4)))
        
        return 
        
    def normal_forward(self,x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5=h5.view(-1,128*8*4)
        return self.fc1(h5), self.fc2(h5)
    '''    
    def forward(self, x):
        #print(self.e1(x))
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        
        h5=h5.view(-1,128*8*4)
        
        
        return self.fc1(h5), self.fc2(h5)#size -> latent_variable_size
        
        
        
        