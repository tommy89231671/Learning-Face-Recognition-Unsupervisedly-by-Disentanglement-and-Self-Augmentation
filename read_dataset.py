# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:06:46 2018

@author: tommy
"""


import torch

import torchvision.transforms as transforms

import torchvision.datasets as dataset

def read_dataset(path,img_size,batch_size):
    
    transform=transforms.Compose([transforms.Resize(img_size),transforms.CenterCrop(img_size),transforms.ToTensor()])
    dt=dataset.ImageFolder(path,transform)
    train_loader = torch.utils.data.DataLoader(dt,batch_size,shuffle=False)

    return train_loader