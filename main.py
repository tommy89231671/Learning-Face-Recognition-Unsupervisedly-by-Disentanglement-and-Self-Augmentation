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
from classifier import Classifier

import time
tStart = time.time()

epoch=20
pre_epoch=10
batch_size=100
img_size=64
c_size=1
z_size=19
margin=0.5
dataloader=read_dataset("/home/eternalding/tommy/pic/",img_size,batch_size)
version=input('result version:')

c_loss_weight=1
RF_loss_weight=2
generator_loss_weight=2
kl_loss_weight=5
reconstruction_loss_weight=10
contrastive_loss_weight=20


path='./result_'+version+'/arg_'+version+'.txt'
f=open(path,'a+')
arg='epoch='+str(epoch)+'\n'+'batch_size='+str(batch_size)+'\n'+'img_size='+str(img_size)+'\n'+\
  'c_size='+str(c_size)+'\n'+'z_size='+str(z_size)+'\n'+'RF_loss_weight=generator_loss_weight='+str(RF_loss_weight)+'\n'+'c_loss_weight='+str(c_loss_weight)+'\n'

f.write(arg+'\n') 
f.close()  
unloader = transforms.ToPILImage()
encoder=Encoder(c_size,z_size)

g=Generator(c_size+z_size)
#g.apply(weights_init)

d=Discriminator()
#d.apply(weights_init)

q=Q(c_size)
#q.apply(weights_init)

dq=D_Q_commonlayer()
#dq.apply(weights_init)  
classifier=Classifier(z_size)

for i in [dq, d, q, g,encoder,classifier]:
  i.cuda(1)
    #i.apply(weights_init)
  
trainer = Trainer(g,dq, d, q,encoder,classifier,batch_size,img_size,c_size,z_size,dataloader,version
                    ,c_loss_weight,RF_loss_weight,generator_loss_weight,reconstruction_loss_weight,kl_loss_weight,epoch,pre_epoch,margin,contrastive_loss_weight)
trainer.train()

tEnd = time.time()

print('Time:'+str(tEnd - tStart)+' sec\n')

#read_result(version,epoch)
"""
for j in range(1,9):
  v=version+str(j+1)
  t1=0.1*j+0.1
  t2=1-t1
  

  c_loss_weight=t1
  RF_loss_weight=t2
  generator_loss_weight=t2
  
  
  unloader = transforms.ToPILImage()
  encoder=Encoder(c_size,z_size)
  g=Generator(c_size+z_size)
  d=Discriminator()
  q=Q(c_size)
  dq=D_Q_commonlayer()
  
  for i in [dq, d, q, g,encoder]:
    i.cuda()
    #i.apply(weights_init)
  
  trainer = Trainer(g,dq, d, q,encoder,batch_size,img_size,c_size,z_size,dataloader,v
                    ,c_loss_weight,RF_loss_weight,generator_loss_weight,epoch)
  trainer.train()
"""  