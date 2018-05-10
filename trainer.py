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

'''
def KL_divergence(c_distribution,z_distribution,batch_size,c_size,z_size):
    c_zero_mean= torch.FloatTensor(batch_size, c_size).()
    c_zero_mean=Variable(c_zero_mean)
    c_zero_mean.data.fill_(0.0)
    z_zero_mean = torch.FloatTensor(batch_size, z_size).cuda()
    z_zero_mean=Variable(z_zero_mean)
    z_zero_mean.data.fill_(0.0)
    c_one_var= torch.FloatTensor(batch_size, c_size).cuda()
    c_one_var=Variable(c_one_var)
    c_one_var.data.fill_(1.0)
    z_one_var= torch.FloatTensor(batch_size, z_size).cuda()
    z_one_var=Variable(z_one_var)
    z_one_var.data.fill_(1.0)
    return torch.distributions.kl_divergence(c_distribution,torch.distributions.Normal(c_zero_mean,c_one_var)),torch.distributions.kl_divergence(z_distribution,torch.distributions.Normal(z_zero_mean,z_one_var))'''

class Trainer:

  def __init__(self,G,DQ,D,Q,Encoder,batch_size,img_size,c_size,z_size,dataloader,version
               ,c_loss_weight,RF_loss_weight,generator_loss_weight,epoch):

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
  
  def train(self):
    x_real = torch.FloatTensor(self.batch_size, 1, self.img_size, self.img_size).cuda()
    label = torch.FloatTensor(self.batch_size,1).cuda()
    #dis_c = torch.FloatTensor(self.batch_size, dis_c_size).cuda()
    c = torch.FloatTensor(self.batch_size, self.c_size).cuda()
    z = torch.FloatTensor(self.batch_size, self.z_size).cuda()

    x_real = Variable(x_real)
    label = Variable(label)
    #dis_c = Variable(dis_c)
    c = Variable(c)
    z = Variable(z)

    criterionBCE = nn.BCELoss( ).cuda()
    #criterionKL=nn.KLDivLoss(size_average=False)
    criterionMSE = nn.MSELoss().cuda()
    #criterionReconstuct = F.binary_cross_entropy().cuda()
    

    optimD = optim.Adam([{'params':self.DQ.parameters()}, {'params':self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimQ = optim.Adam([{'params':self.G.parameters()},{'params':self.DQ.parameters()}, {'params':self.Q.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimG = optim.Adam([{'params':self.G.parameters()}], lr=0.002, betas=(0.5, 0.99))
    optimVAE =optim.Adam([{'params':self.Encoder.parameters()}, {'params':self.G.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimEncoder = optim.Adam([{'params':self.Encoder.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    
    for epoch in range(self.epoch):
        
        for num_iters, batch_data in enumerate(self.dataloader, 0):
          
            """Discriminator's real part"""
            optimD.zero_grad()
            
            x, _ = batch_data
    
            bs = x.size(0)
            if bs!=self.batch_size:
                
                
                break
            x_real.data.resize_(x.size())
            label.data.resize_(bs,1)
            c.data.resize_(bs, self.c_size)
            z.data.resize_(bs,self.z_size)
            
            x_real.data.copy_(x)
            dq1 = self.DQ(x_real)
            #print(dq1)
            x_real_result = self.D(dq1)
            label.data.fill_(1.0)
            
            #print()
    
            loss_real = self.RF_loss_weight*criterionBCE(x_real_result, label)
            loss_real.backward()
          
            """Encoder part"""
            c_en,z_en=self.Encoder(x_real)
            c_mean,c_logvar=torch.chunk(c_en,2,dim=1)
            z_mean,z_logvar=torch.chunk(z_en,2,dim=1)
            
            
            
            """"Discriminator's fake part"""
            c_distribution =torch.distributions.Normal(c_mean, torch.exp(c_logvar))
            z_distribution= torch.distributions.Normal(z_mean, torch.exp(z_logvar))
              
            
            c=c_distribution.sample()
            z=z_distribution.sample()
            
            
            #z.data.copy_(fix_noise)#fix noise to train
            
           
            G_input = torch.cat([z,c], 1).view(-1,self.c_size+self.z_size, 1, 1)
            
            x_fake = self.G(G_input)
            dq2 = self.DQ(x_fake.detach())
            x_fake_result = self.D(dq2)
            label.data.fill_(0.0)
            loss_fake = self.RF_loss_weight*criterionBCE(x_fake_result, label)
            loss_fake.backward()
            
            
            optimD.step()
            '''
            """Find generator_loss"""
            optimG.zero_grad()
            optimQ.zero_grad()
            dq = self.DQ(x_fake)
            x_fake_result = self.D(dq)
            label.data.fill_(1.0)
    
            generator_loss = self.generator_loss_weight*criterionBCE(x_fake_result, label)
            #generator_loss.backward()
            
            
            q= self.Q(dq)
            
            """Find C_losss and combine Closs and generator_loss"""
            C_loss = self.c_loss_weight*criterionMSE(q,c)
            
            G_loss = generator_loss + C_loss
            G_loss.backward(retain_graph=True)
            optimQ.step()
            optimG.step()
            '''
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
            
            
            
            
            #vae_reconstruct_loss = F.binary_cross_entropy(x_fake,x_real, size_average=False).cuda()
            optimVAE.zero_grad()
            optimEncoder.zero_grad()
            z_kl_divergence = torch.sum(0.5 * (z_mean**2 + torch.exp(z_logvar) - z_logvar -1))
            c_kl_divergence = torch.sum(0.5 * (c_mean**2 + torch.exp(c_logvar) - c_logvar -1))
            KL=z_kl_divergence+c_kl_divergence
            KL.backward(retain_graph=True)
            optimEncoder.step()
            #print((x_fake.view(x_fake.size(0),-1),x_real.view(x_real.size(0),-1)))
            
            #input('en')
            vae_reconstruct_loss = criterionMSE(x_fake.view(x_fake.size(0),-1),x_real.view(x_real.size(0),-1))
            
            #print(vae_reconstruct_loss)
            vae_reconstruct_loss.backward(retain_graph=True)
            #vae_loss=vae_reconstruct_loss+KL
            #vae_loss.backward()
            optimVAE.step()
            
            """update generator more"""
            
            
            
            
            
            #c_kl_divergence,z_kl_divergence=KL_divergence(c_distribution,z_distribution,self.batch_size,self.c_size,self.z_size)
            
            
            
            
        result='Epoch/Iter:{0}/{1}, Real loss: {2},fake loss: {3},c loss: {4}, generator_loss: {5},reconstruction_loss: {6},KL_loss: {7}:'.format(
                epoch, num_iters, loss_real.data.cpu().numpy(),loss_fake.data.cpu().numpy(),C_loss.data.cpu().numpy(),
                generator_loss.data.cpu().numpy(),vae_reconstruct_loss.data.cpu().numpy(),KL.data.cpu().numpy())
        
        print(result)
        
        path='./result_'+self.version+'/result_'+self.version+'.txt'
        
        f=open(path,'a+')
        f.write(result+'\n')
        f.close()
        
        
        
        c_en,z_en=self.Encoder(x_real)
        c_mean,c_logvar=torch.chunk(c_en,2,dim=1)
        z_mean,z_logvar=torch.chunk(z_en,2,dim=1)
            
        c_distribution =torch.distributions.Normal(c_mean, torch.exp(c_logvar))
        z_distribution= torch.distributions.Normal(z_mean, torch.exp(z_logvar))
        c=c_distribution.sample()
        z=z_distribution.sample()
        
        
        
        G_input = torch.cat([z,c], 1).view(-1,self.c_size+self.z_size, 1, 1).cuda()
         
        x_save = self.G(G_input)
          
       
        f1name='./result_'+self.version+'/'+str(epoch)+'_'+self.version+'.png'
          
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
