# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:15:26 2018
@author: tommy
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision.utils import save_image
import torch.nn.functional as F
import siamese

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import tsne


class Trainer:

  def __init__(self,G,DQ,D,Q,Encoder,batch_size,img_size,c_size,z_size,dataloader,version
               ,c_loss_weight,RF_loss_weight,generator_loss_weight,reconstruction_loss_weight,kl_loss_weight,epoch,pre_epoch):

    self.G = G
    self.DQ =DQ
    self.D = D
    self.Q = Q
    self.Encoder=Encoder
    self.batch_size = batch_size
    self.img_size=img_size
    self.c_size=c_size
    self.z_size=z_size
    self.dataloader=dataloader
    self.version=version
    self.c_loss_weight=c_loss_weight
    self.RF_loss_weight=RF_loss_weight
    self.generator_loss_weight=generator_loss_weight
    self.epoch=epoch
    self.reconstruction_loss_weight=reconstruction_loss_weight
    self.kl_loss_weight=kl_loss_weight
    self.pre_epoch=pre_epoch
  def train(self):
    x_real = torch.FloatTensor(self.batch_size, 1, self.img_size, self.img_size).cuda(1)
    img0=torch.FloatTensor(self.batch_size, 3, self.img_size, self.img_size).cuda(1)
    img1=torch.FloatTensor(self.batch_size, 3, self.img_size, self.img_size).cuda(1)
    label_fix=torch.FloatTensor(self.batch_size,1).cuda(1)
    label = torch.FloatTensor(self.batch_size,1).cuda(1)
    #dis_c = torch.FloatTensor(self.batch_size, dis_c_size).cuda()
    c = torch.FloatTensor(self.batch_size, self.c_size).cuda(1)
    z = torch.FloatTensor(self.batch_size, self.z_size).cuda(1)
    img0 = Variable(img0)
    img1 = Variable(img1)
    x_real = Variable(x_real)
    label_fix = Variable(label_fix)
    label = Variable(label)
    #dis_c = Variable(dis_c)
    c = Variable(c)
    z = Variable(z)
    
    KL=torch.FloatTensor(1).cuda(1)
    KL = Variable(KL)
    z_kl_divergence=torch.FloatTensor(1).cuda(1)
    z_kl_divergence = Variable(z_kl_divergence)  
    c_kl_divergence=torch.FloatTensor(1).cuda(1)
    c_kl_divergence = Variable(c_kl_divergence)    
    
    
    
    
    criterionBCE = nn.BCELoss( ).cuda(1)
    criterionMSE = nn.MSELoss().cuda(1)
    criterionContrastive = siamese.ContrastiveLoss(margin=20.0,batch_size=self.batch_size)
    

    optimD = optim.Adam([{'params':self.DQ.parameters()}, {'params':self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimQ = optim.Adam([{'params':self.G.parameters()},{'params':self.DQ.parameters()}, {'params':self.Q.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimG = optim.Adam([{'params':self.G.parameters()}], lr=0.002, betas=(0.5, 0.99))
    optimVAE =optim.Adam([{'params':self.Encoder.parameters()}, {'params':self.G.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimEncoder = optim.Adam([{'params':self.Encoder.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    
    
    
    for i in range(self.pre_epoch):
        folder_dataset = dset.ImageFolder('"/home/eternalding/tommy/training_data/"')
        siamese_dataset = siamese.SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize(self.img_size),transforms.CenterCrop(self.img_size),transforms.ToTensor()]),
                                        should_invert=False)                             
                                                                      
        
        
        train_dataloader = DataLoader(siamese_dataset,
                                      shuffle=False,
                                      #num_workers=8,
                                      batch_size=self.batch_size)
       
        test_img=torch.FloatTensor(self.batch_size, 3, self.img_size, self.img_size).cuda(1)
        test_img = Variable(test_img)
        
       
        
        for num_iters, data in enumerate(train_dataloader,0):
          img00, img11 , labelx,person_class = data
          #img0, img1 , label = img0.cuda(1), img1.cuda(1) , label.cuda(1)
          
          #print(person_class)
          label_fix.data.resize_(labelx.size())
          label_fix.data.copy_(labelx)
          img0.data.resize_(img00.size())
          img0.data.copy_(img00)
          img1.data.resize_(img11.size())
          img1.data.copy_(img11)
          optimEncoder.zero_grad()
           
          
          #output1=self.Encoder(img0,0)
          #output2=self.Encoder(img1,0)
          
          c1_en,z1_en=self.Encoder(img0)
          c1_mean,c1_logvar=torch.chunk(c1_en,2,dim=1)
          z1_mean,z1_logvar=torch.chunk(z1_en,2,dim=1)
          
          qn = torch.norm(z1_mean, p=2, dim=0).detach()
          z1_mean = z1_mean.div(qn.expand_as(z1_mean))
          
          cz1_mean=torch.cat([c1_mean,z1_mean], 1).view(-1,self.c_size+self.z_size) 
          qn = torch.norm(cz1_mean, p=2, dim=0).detach()
          cz1_mean = cz1_mean.div(qn.expand_as(cz1_mean))
          
          c2_en,z2_en=self.Encoder(img1)
          c2_mean,c2_logvar=torch.chunk(c2_en,2,dim=1)
          z2_mean,z2_logvar=torch.chunk(z2_en,2,dim=1)
          
          qn = torch.norm(z2_mean, p=2, dim=0).detach()
          z2_mean = z2_mean.div(qn.expand_as(z2_mean))
          
          cz2_mean=torch.cat([c2_mean,z2_mean], 1).view(-1,self.c_size+self.z_size) 
          qn = torch.norm(cz2_mean, p=2, dim=0).detach()
          cz2_mean = cz2_mean.div(qn.expand_as(cz2_mean))
          #print(img0)
          #print(img00)
          #input('enter')
          #print(img1)
          #output1 = torch.cat([z1,c1], 1).view(-1,self.c_size+self.z_size) 
          #output2 = torch.cat([z2,c2], 1).view(-1,self.c_size+self.z_size)   
          #print(output1)
          #print(output2)
          loss_contrastive = criterionContrastive(cz1_mean,cz2_mean,label_fix)
          #print(loss_contrastive)
          #input('enter')
          #loss_contrastive=Variable(loss_contrastive)
          #print(criterion(output1,output2,label))
          #loss_contrastive.data.fill_()
          loss_contrastive.backward()
          
          optimEncoder.step()
        result='Epoch/Iter:{0}/{1}, Contrastive loss: {2}'.format(
                i, num_iters, loss_contrastive.data.cpu().numpy())
        
        print(result)
        cz_mean=torch.FloatTensor(self.batch_size,-1).cuda(1)
        cz_mean=Variable(cz_mean)
        for num_iters, data in enumerate(train_dataloader,0):
          img00, img11 , labelx,person_class = data
          test_img.data.resize_(img00.size())
          test_img.data.copy_(img00)
          c_en,z_en=self.Encoder(test_img)
          c_mean,c_logvar=torch.chunk(c_en,2,dim=1)
          z_mean,z_logvar=torch.chunk(z_en,2,dim=1)
          
          qn = torch.norm(z_mean, p=2, dim=0).detach()
          z_mean = z_mean.div(qn.expand_as(z_mean))
          
          cz_mean=torch.cat([c_mean,z_mean], 1).view(-1,self.c_size+self.z_size) 
          qn = torch.norm(cz_mean, p=2, dim=0).detach()
          cz_mean = cz_mean.div(qn.expand_as(cz_mean))
          break
        tsne.main(cz_mean.data.cpu().numpy(),i,self.version,person_class)
    
    
    
    
    for epoch in range(self.epoch):
        
        for num_iters, batch_data in enumerate(self.dataloader, 0):
          
            """Discriminator's real part"""
            optimD.zero_grad()
            
            x, _ = batch_data
    
            bs = x.size(0)
            #if bs!=self.batch_size:
                
                
                #break
            x_real.data.resize_(x.size())
            #print(x_real)
            label.data.resize_(bs,1)
            c.data.resize_(bs, self.c_size)
            z.data.resize_(bs,self.z_size)
            #print(x)
            x_real.data.copy_(x)
            #print(x_real)
            #input('enter')
            dq1 = self.DQ(x_real)
            #print(dq1)
            x_real_result = self.D(dq1)
            label.data.fill_(1.0)
            
            #print()
    
            loss_real = self.RF_loss_weight*criterionBCE(x_real_result, label)
            #print(loss_real)
            loss_real.backward()
          
            """Encoder part"""
            
            
            if epoch<20:
              #print(x_real)
              c_en,z_en=self.Encoder(x_real)
              c_mean,c_logvar=torch.chunk(c_en,2,dim=1)
              z_mean,z_logvar=torch.chunk(z_en,2,dim=1)
              c_distribution =torch.distributions.Normal(c_mean, torch.exp(c_logvar))
              z_distribution= torch.distributions.Normal(z_mean, torch.exp(z_logvar))  
              c=c_distribution.sample()
              z=z_distribution.sample()
            
            
            else:
              z.data=torch.randn(z.size()).cuda(1)#changing noise to train
              c.data=torch.randn(c.size()).cuda(1)
            
            
            #z.data.copy_(fix_noise)#fix noise to train
            
            '''
            c_en,z_en=self.Encoder(x_real)
            c_mean,c_logvar=torch.chunk(c_en,2,dim=1)
            z_mean,z_logvar=torch.chunk(z_en,2,dim=1)
            c_distribution =torch.distributions.Normal(c_mean, torch.exp(c_logvar))
            z_distribution= torch.distributions.Normal(z_mean, torch.exp(z_logvar))  
            c=c_distribution.sample()
            z=z_distribution.sample()
            '''
            G_input = torch.cat([z,c], 1).view(-1,self.c_size+self.z_size, 1, 1)
            
            x_fake = self.G(G_input)
            
            dq2 = self.DQ(x_fake.detach())
            x_fake_result = self.D(dq2)
            label.data.fill_(0.0)
            loss_fake = self.RF_loss_weight*criterionBCE(x_fake_result, label)
            loss_fake.backward()
            
            
            optimD.step()
            
            optimG.zero_grad()
            
            dq = self.DQ(x_fake)
            x_fake_result = self.D(dq)
            label.data.fill_(1.0)
    
            generator_loss = self.generator_loss_weight*criterionBCE(x_fake_result, label)
            generator_loss.backward(retain_graph=True)
            optimG.step()
            
            optimQ.zero_grad()
            q= self.Q(dq)
            
            """Find C_losss and combine Closs and generator_loss"""
            C_loss = self.c_loss_weight*criterionMSE(q,c)
            C_loss.backward(retain_graph=True)
            #G_loss = generator_loss + C_loss
            #G_loss.backward(retain_graph=True)
            optimQ.step()
            
            
            optimVAE.zero_grad()
            if epoch<20:
              optimEncoder.zero_grad()
              z_kl_divergence = torch.sum(0.5 * (z_mean**2 + torch.exp(z_logvar) - z_logvar -1))
              c_kl_divergence = 10*torch.sum(0.5 * (c_mean**2 + torch.exp(c_logvar) - c_logvar -1))
              KL=self.kl_loss_weight*(z_kl_divergence+c_kl_divergence)
              KL.backward(retain_graph=True)
              optimEncoder.step()
            else:
              KL.data.fill_(0)
              z_kl_divergence.data.fill_(0)
              c_kl_divergence.data.fill_(0)
            vae_reconstruct_loss = self.reconstruction_loss_weight*criterionMSE(x_fake.view(x_fake.size(0),-1),x_real.view(x_real.size(0),-1))
            
            #print(vae_reconstruct_loss)
            vae_reconstruct_loss.backward(retain_graph=True)
            #vae_loss=vae_reconstruct_loss+KL
            #vae_loss.backward()
            optimVAE.step()
            
            
            '''
            optimVAE.zero_grad()
            optimEncoder.zero_grad()
            z_kl_divergence = torch.sum(0.5 * (z_mean**2 + torch.exp(z_logvar) - z_logvar -1))
            c_kl_divergence = 10*torch.sum(0.5 * (c_mean**2 + torch.exp(c_logvar) - c_logvar -1))
            KL=self.kl_loss_weight*(z_kl_divergence+c_kl_divergence)
            KL.backward(retain_graph=True)
            optimEncoder.step()
            #print((x_fake.view(x_fake.size(0),-1),x_real.view(x_real.size(0),-1)))
            
            #input('en')
            vae_reconstruct_loss = self.reconstruction_loss_weight*criterionMSE(x_fake.view(x_fake.size(0),-1),x_real.view(x_real.size(0),-1))
            
            #print(vae_reconstruct_loss)
            vae_reconstruct_loss.backward(retain_graph=True)
            #vae_loss=vae_reconstruct_loss+KL
            #vae_loss.backward()
            optimVAE.step()
            '''
            
            """update generator more"""
           
            
            
            
        result='Epoch/Iter:{0}/{1}, Real loss: {2},fake loss: {3},c loss: {4}, generator_loss: {5},reconstruction_loss: {6},KL_loss: {7}:'.format(
                epoch, num_iters, loss_real.data.cpu().numpy(),loss_fake.data.cpu().numpy(),C_loss.data.cpu().numpy(),
                generator_loss.data.cpu().numpy(),vae_reconstruct_loss.data.cpu().numpy(),KL.data.cpu().numpy())
        
        print(result)
        klresult='c_kl:'+str(c_kl_divergence)+'/'+'z_kl:'+str(z_kl_divergence)
        #print('c_kl:'+str(c_kl_divergence))
        #print('z_kl:'+str(z_kl_divergence))
        pathkl='./result_'+self.version+'/kl_'+self.version+'.txt'
        
        f1=open(pathkl,'a+')
        f1.write(klresult+'\n')
        f1.close()
        
        path='./result_'+self.version+'/result_'+self.version+'.txt'
        
        f=open(path,'a+')
        f.write(result+'\n')
        f.close()
        
        '''
        if epoch%2==0:
              c_en,z_en=self.Encoder(x_real)
              c_mean,c_logvar=torch.chunk(c_en,2,dim=1)
              z_mean,z_logvar=torch.chunk(z_en,2,dim=1)
              c_distribution =torch.distributions.Normal(c_mean, torch.exp(c_logvar))
              z_distribution= torch.distributions.Normal(z_mean, torch.exp(z_logvar))  
              c=c_distribution.sample()
              z=z_distribution.sample()
            
            
        else:
              z.data=torch.randn(z.size()).cuda(1)#changing noise to train
              c.data=torch.randn(c.size()).cuda(1)
        
        c_en,z_en=self.Encoder(x_real)
        c_mean,c_logvar=torch.chunk(c_en,2,dim=1)
        z_mean,z_logvar=torch.chunk(z_en,2,dim=1)
            
        c_distribution =torch.distributions.Normal(c_mean, torch.exp(c_logvar))
        z_distribution= torch.distributions.Normal(z_mean, torch.exp(z_logvar))
        c=c_distribution.sample()
        z=z_distribution.sample()
        '''
        c_en,z_en=self.Encoder(x_real)
        c_mean,c_logvar=torch.chunk(c_en,2,dim=1)
        z_mean,z_logvar=torch.chunk(z_en,2,dim=1)
        c_distribution =torch.distributions.Normal(c_mean, torch.exp(c_logvar))
        z_distribution= torch.distributions.Normal(z_mean, torch.exp(z_logvar))  
        c=c_distribution.sample()
        z=z_distribution.sample()
        G_input = torch.cat([z,c], 1).view(-1,self.c_size+self.z_size, 1, 1).cuda(1)
         
        x_save = self.G(G_input)
          
       
        f1name='./result_'+self.version+'/'+str(epoch)+'_G(E(x)).png'
        
        
        
        z.data=torch.randn(z.size()).cuda(1)#changing noise to train
        c.data=torch.randn(c.size()).cuda(1)  
        save_image(x_save.data,f1name, nrow=int(self.batch_size/5))
        G_input = torch.cat([z,c], 1).view(-1,self.c_size+self.z_size, 1, 1).cuda(1)
         
        x_save = self.G(G_input)
          
       
        f1name='./result_'+self.version+'/'+str(epoch)+'_rand.png'
          
        save_image(x_save.data,f1name, nrow=int(self.batch_size/5))
    
    gpath='./model/Generator_'+self.version+'.pkl'
    dpath='./model/Discriminator_'+self.version+'.pkl'
    qpath='./model/Q_'+self.version+'.pkl'
    dqpath='./model/DQ_'+self.version+'.pkl'
    enpath='./model/Encoder_'+self.version+'.pkl'
    torch.save(self.G.state_dict(), gpath)
    torch.save(self.D.state_dict(),dpath)
    torch.save(self.Q.state_dict(), qpath)
    torch.save(self.DQ.state_dict(), dqpath)
    torch.save(self.Encoder.state_dict(), enpath)