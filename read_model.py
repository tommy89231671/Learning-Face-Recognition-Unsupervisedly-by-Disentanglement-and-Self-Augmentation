# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 12:30:06 2018

@author: tommy
"""
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
from torchvision.utils import save_image
import siamese
import torchvision.datasets as dset
import tsne
import matplotlib.pyplot as plt

class Generator(nn.Module):
    
    def __init__(self,c_z_dim):
        super(Generator, self).__init__()
        self.c_z_dim=c_z_dim
        #self.conv1=nn.ConvTranspose2d(c_z_dim,1024,4,1,0,bias=False),        
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
            
        )
    def forward(self,x):
        output = self.main(x)
        #output=self.conv1(x)
        #print(output)
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
    #self.conv_disc = nn.Conv2d(128,5,4)
    self.conv_out = nn.Conv2d(128, self.c_size, 4)
    #self.conv_var = nn.Conv2d(128, 2, 4)

  def forward(self, x):
    y = self.conv(x)
    Q_output = self.conv_out(y).squeeze()
    return Q_output  

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
        
        
        
        
version=input('version to read:')
#arg=input('arg:')
#version='v2'
path='/home/eternalding/tommy/project/result_'+version+'/'

gpath=path+'Generator.pkl'
dpath=path+'Discriminator.pkl'
qpath=path+'Q.pkl'
dqpath=path+'DQ.pkl'
epath=path+'Encoder.pkl'

c_size=1
z_size=19
batch_size=100
img_size=64
g=Generator(c_size+z_size).cuda(1)
d=Discriminator().cuda(1)
q=Q(c_size).cuda(1)
dq=D_Q_commonlayer().cuda(1)
encoder=Encoder(c_size,z_size).cuda(1)

g.load_state_dict(torch.load(gpath)) 
d.load_state_dict(torch.load(dpath)) 
q.load_state_dict(torch.load(qpath)) 
dq.load_state_dict(torch.load(dqpath)) 
encoder.load_state_dict(torch.load(epath))

c=torch.FloatTensor(batch_size,c_size).cuda(1)
c=Variable(c,volatile=True)

z=torch.FloatTensor(batch_size,z_size).cuda(1)
z=Variable(z)
zz=torch.randn(z_size,1)
zz=zz.expand(z_size,100)
z.data.copy_(torch.t(zz))

folder_dataset = dset.ImageFolder("/home/eternalding/tommy/training_data/")
siamese_dataset = siamese.SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize(img_size),transforms.CenterCrop(img_size),transforms.ToTensor()]),
                                        should_invert=False)                             
                                                                      
        
        
train_dataloader = DataLoader(siamese_dataset,
                                      shuffle=False,
                                      #num_workers=8,
                                      batch_size=batch_size)
       
test_img=torch.FloatTensor(batch_size, 3, img_size, img_size).cuda(1)
test_img = Variable(test_img)
for ni, data in enumerate(train_dataloader,0):
      img00, img11 ,_,person_class = data
      test_img.data.resize_(img00.size())
      test_img.data.copy_(img00)
      c_en,z_en=encoder(test_img)
      c_mean,c_logvar=torch.chunk(c_en,2,dim=1)
      z_mean,z_logvar=torch.chunk(z_en,2,dim=1)
              
      qn = torch.norm(z_mean, p=2, dim=0).detach()
      z_mean = z_mean.div(qn.expand_as(z_mean))
              
      cz_mean=torch.cat([c_mean,z_mean], 1).view(-1,c_size+z_size) 
      qn = torch.norm(cz_mean, p=2, dim=0).detach()
      cz_mean = cz_mean.div(qn.expand_as(cz_mean))
      
      break
      
      
ss=torch.zeros(batch_size)

YY = tsne.tsne(cz_mean.data.cpu().numpy(), 2, 10, 30)
    #print(YY.shape)
    #pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
plt.scatter(YY[:,0],YY[:,1],20,person_class)
fname='/home/eternalding/tommy/project/test_result/'+'test_tsne_'+version+'.png'
plt.savefig(fname)
plt.clf()
YY = tsne.tsne(c_mean.data.cpu().numpy(), 2, 10, 30)
    #print(YY.shape)
    #pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
plt.scatter(YY[:,0],YY[:,1],20,person_class)
fname='/home/eternalding/tommy/project/test_result/'+'test_tsnee_'+version+'.png'
plt.savefig(fname)
plt.clf()        

plt.scatter(c_mean.data.cpu().numpy()[:,0],ss.numpy(),20,person_class)

            
#print(c_mean)

fname='/home/eternalding/tommy/project/test_result/'+'test_c_vs_class_'+version+'.png'
plt.savefig(fname)
plt.clf()



#print(torch.t(zz))

#cc=torch.FloatTensor([20,10,0,-10,-20,-30]).cuda(1)
cc=torch.randn(1,6).cuda(1)
ccc=torch.FloatTensor([0,0,0,0]).cuda(1)
for i in range(4):
  cc=torch.cat([cc,cc])
cc=cc.view(-1,1)
ccc=ccc.view(-1,1)
#print(cc)
cc=torch.cat([cc,ccc])
#print(cc)
cc=torch.range(-299,300,6)
cc.view(batch_size,1)
#c.data.copy_(torch.t(cc))
c.data.copy_(cc)
print(c)
#print(c)
  
G_input=torch.cat([c,z],1).view(-1,c_size+z_size,1,1)
  
G_output = g(G_input)
#print(G_output)
fname='/home/eternalding/tommy/project/test_result/'+'test_result_'+version+'.png'
save_image(G_output.data,fname, nrow=6)
  
DQ_output=dq(G_output)
Q_output=Variable(torch.FloatTensor(batch_size,1),volatile=True)
Q_output=q(DQ_output)
  
D_output=d(DQ_output)
#print('Q_output:',Q_output)
criterionMSE = nn.MSELoss().cuda(1)
C_loss =criterionMSE(c,Q_output)
print(C_loss)
