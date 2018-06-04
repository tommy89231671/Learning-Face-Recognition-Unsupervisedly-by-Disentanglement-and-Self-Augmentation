# -*- coding: utf-8 -*-
"""
Created on Thu May 24 01:50:58 2018

@author: tommy
"""

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #print(self.imageFolderDataset.labels)
        #img0 = Image.open(img0_tuple[0])
        #print(img0_tuple[1])
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        
        #img0 = img0.convert("L")
        #img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        #print(img0)
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32)),img0_tuple[1]
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
'''        
folder_dataset = dset.ImageFolder('/home/eternalding/tommy/training_data/')
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)'''

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self,margin,batch_size):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.batch_size=batch_size
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        #print(euclidean_distance)
        #print(output2)
        """
        a=euclidean_distance**2
        
        b=torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        print(a.size()[0])
        sum=0
        
        for i in range(a.size()[0]):
            sum+=float(label[i])*float(a[i])+float(1-label[i])*float(b[i])
        
        #print(sum)
        loss_contrastive=Variable(torch.FloatTensor(1),requires_grad=True)
        loss_contrastive.data.fill_(sum/(self.batch_size))
        
        return 10*loss_contrastive"""
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive        
'''        
train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=train_batch_size)
net = encoder().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
counter = []
loss_history = [] 
iteration_number= 0
for epoch in range(0,train_number_epochs):
    for i, data in enumerate(train_dataloader,0):
        #print(data)
        img0, img1 , label = data
        #print(label)
        img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
        #loss_contrastive=
        optimizer.zero_grad()
        output1,output2 = net(img0,img1)
        
        #print(img0)
        print(output1)
        #print(type(output2))
           
        loss_contrastive = criterion(output1,output2,label)
        #loss_contrastive=Variable(loss_contrastive)
        #print(criterion(output1,output2,label))
        #loss_contrastive.data.fill_()
        loss_contrastive.backward()
        
        optimizer.step()
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,float(loss_contrastive[0])))
            iteration_number +=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive[0])
#show_plot(counter,loss_history)
  '''
