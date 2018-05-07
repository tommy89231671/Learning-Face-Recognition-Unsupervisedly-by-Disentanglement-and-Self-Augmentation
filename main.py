# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:19:01 2018

@author: tommy
"""

import torchvision.transforms as transforms

from infogan import Generator,Discriminator,Q,D_Q_commonlayer
from encoder import Encoder
from trainer import Trainer
from read_dataset import read_dataset
from read_result import read_result

epoch=20
batch_size=100
img_size=64
c_size=1
z_size=50
dataloader=read_dataset('../pic',img_size,batch_size)
version=input('result version:')


c_loss_weight=0.2
RF_loss_weight=0.8
generator_loss_weight=0.8


unloader = transforms.ToPILImage()
encoder=Encoder(c_size,z_size)
g=Generator(c_size+z_size)
d=Discriminator()
q=Q(c_size)
dq=D_Q_commonlayer()

for i in [dq, d, q, g,encoder]:
  i.cuda()
  #i.apply(weights_init)

trainer = Trainer(g,dq, d, q,encoder,batch_size,img_size,c_size,z_size,dataloader,version
                  ,c_loss_weight,RF_loss_weight,generator_loss_weight,epoch)
trainer.train()

read_result(version,epoch)