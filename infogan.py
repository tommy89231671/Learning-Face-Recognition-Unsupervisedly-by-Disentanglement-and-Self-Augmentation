# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:01:54 2018

@author: tommy
"""

import torch.nn as nn



class Generator(nn.Module):
    
    def __init__(self,c_z_dim):
        super(Generator, self).__init__()
        self.c_z_dim=c_z_dim     
        self.main = nn.Sequential( 
            nn.ConvTranspose2d(self.c_z_dim,1024,4,1,0,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # state size. 1024x 4 x 4
            nn.ConvTranspose2d(1024,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(128,3,4,2,1,bias=False),
            
            nn.Tanh(),
            nn.Sigmoid()
        )
    def forward(self,x):
        output = self.main(x)
        return output

class D_Q_commonlayer(nn.Module):
    def __init__(self):
        super(D_Q_commonlayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(128,256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(256,512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(512,1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            
        )
    def forward(self,x):
        output = self.main(x)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main=nn.Sequential(
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
                
        )
    def forward(self, x):
        output = self.main(x).view(-1, 1)
        return output

class Q(nn.Module):

  def __init__(self,c_size):
    super(Q, self).__init__()
    self.c_size=c_size
    self.conv = nn.Conv2d(1024, 128, 1, bias=False)
    self.bn = nn.BatchNorm2d(128)
    self.lReLU = nn.LeakyReLU(0.1, inplace=True)
   
    self.conv_out = nn.Conv2d(128, self.c_size, 4)
    

  def forward(self, x):
    y = self.conv(x)
    Q_output = self.conv_out(y).squeeze()
    return Q_output 