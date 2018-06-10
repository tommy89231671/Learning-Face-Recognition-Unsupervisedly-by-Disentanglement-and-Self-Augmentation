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
import matplotlib.pyplot as plt


class Trainer:

  def __init__(self,G,DQ,D,Q,Encoder,Classifier,batch_size,img_size,c_size,z_size,dataloader,version
               ,c_loss_weight,RF_loss_weight,generator_loss_weight,reconstruction_loss_weight,kl_loss_weight,epoch,pre_epoch,margin,contrastive_loss_weight):

    self.G = G
    self.DQ =DQ
    self.D = D
    self.Q = Q
    self.Encoder=Encoder
    self.Classifier=Classifier
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
    self.margin=margin
    self.contrastive_loss_weight=contrastive_loss_weight
  def do_siamese(self,epoch,training_iteration,optimEncoder,pre_or_inner_or_result,out_epoch,weight):
    img0=torch.FloatTensor(self.batch_size, 3, self.img_size, self.img_size).cuda(1)
    img1=torch.FloatTensor(self.batch_size, 3, self.img_size, self.img_size).cuda(1)
    img0 = Variable(img0)
    img1 = Variable(img1)
    classifier_label=torch.FloatTensor(self.batch_size,2).cuda(1)
    classifier_label=Variable(classifier_label)
    #classifier_input=torch.FloatTensor(self.batch_size,self.z_size*2).cuda(1)
    #classifier_input=Variable(classifier_input)
    criterionContrastive = siamese.ContrastiveLoss(margin=self.margin,batch_size=self.batch_size)
    folder_dataset = dset.ImageFolder("/home/eternalding/tommy/training_data/")
    siamese_dataset = siamese.SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize(self.img_size),transforms.CenterCrop(self.img_size),transforms.ToTensor()]),
                                        should_invert=False)                             
    criterionBCE = nn.BCELoss( ).cuda(1)                                                                   
    optimClassifier = optim.Adam([{'params':self.Classifier.parameters()}], lr=0.002, betas=(0.5, 0.99))    
        
    train_dataloader = DataLoader(siamese_dataset,
                                      shuffle=False,
                                      #num_workers=8,
                                      batch_size=self.batch_size)
    
    for i in range(epoch):
        
       
        test_img=torch.FloatTensor(self.batch_size, 3, self.img_size, self.img_size).cuda(1)
        test_img = Variable(test_img)
        
        
        
        for num_iters, data in enumerate(train_dataloader,0):
          img00, img11 , labelx,_= data
          #img0, img1 , label = img0.cuda(1), img1.cuda(1) , label.cuda(1)
          if num_iters==training_iteration:
            break
          #print(person_class)
          labelx=Variable(labelx.cuda(1))
          #label_fix.data.resize_(labelx.size())
          #label_fix.data.copy_(labelx)
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
          ss=torch.zeros(self.batch_size)
        
          #print(ss.numpy().shape)
          #print(c1_mean.data.cpu().numpy()[:,0].shape)
          #plt.plot(c1_mean.data.cpu().numpy()[:,0],ss.numpy())
          #plt.show()
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
          #print(c1_mean.data.cpu().numpy()[:,0].shape)
          cz2_mean=torch.cat([c2_mean,z2_mean], 1).view(-1,self.c_size+self.z_size) 
          qn = torch.norm(cz2_mean, p=2, dim=0).detach()
          cz2_mean = cz2_mean.div(qn.expand_as(cz2_mean))
          loss_contrastive = weight*criterionContrastive(cz1_mean,cz2_mean,labelx)
          #print(loss_contrastive)
          loss_contrastive.backward(retain_graph=True)
          #print(c_mean.data.cpu().numpy())
          optimEncoder.step()
          
          classifier_input=torch.cat([z1_mean,z2_mean],1).cuda(1)
          #print(classifier_input)
          classifier_output=self.Classifier(classifier_input)
          temp=labelx*(-1)+1
          classifier_label_real=torch.cat([labelx,temp],1)
          classifier_label_fake=torch.FloatTensor(self.batch_size,2).cuda(1)
          classifier_label_fake=Variable(classifier_label_fake)
          classifier_label_fake.data.fill_(0.5)
          
          optimClassifier.zero_grad()
          classifier_loss=weight*criterionBCE(classifier_output,classifier_label_real)
          classifier_loss.backward(retain_graph=True)
          optimClassifier.step()
          
          optimEncoder.zero_grad()
          classifier_for_encoder_loss=weight*criterionBCE(classifier_output,classifier_label_fake)
          classifier_for_encoder_loss.backward()
          
          #print(classifier_label)
          optimEncoder.step()
          
        #print(classifier_loss)
        #print(classifier_for_encoder_loss)  
        if pre_or_inner_or_result==1:
            break
        else:
            result='Epoch/Iter:{0}/{1}, Contrastive loss: {2}, Classifier_loss: {3}, Classifier_for_encoder_loss: {4}'.format(
                i, num_iters, loss_contrastive.data.cpu().numpy(),classifier_loss.data.cpu().numpy(),classifier_for_encoder_loss.data.cpu().numpy())
        
            print(result)
        
        cz_mean=torch.FloatTensor(self.batch_size,-1).cuda(1)
        cz_mean=Variable(cz_mean)
        
        for ni, data in enumerate(train_dataloader,0):
            img00, img11 ,_,person_class = data
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
        ss=torch.zeros(self.batch_size)
            
        tsne.main(cz_mean.data.cpu().numpy(),i,self.version,person_class,30,pre_or_inner_or_result,out_epoch)
        plt.scatter(c_mean.data.cpu().numpy()[:,0],ss.numpy(),20,person_class)
        cresult='Epoch{0}, c_mean: {2}, class:{3}'.format(
                    i, num_iters, c_mean.data.cpu().numpy()[:,0],person_class)
            
            #print(cresult)
        cpath='./result_'+self.version+'/c_vs_class_'+self.version+'.txt'
            
        f=open(cpath,'a+')
        f.write(cresult+'\n')
        f.close()
        path=''
        if pre_or_inner_or_result==0: 
          path='./result_'+self.version+'/'+'scatter_c_pre_'+str(i)+'.png'
        elif pre_or_inner_or_result==2:
          path='./result_'+self.version+'/'+'scatter_c_result_'+str(out_epoch)+'.png'
        plt.savefig(path)
        plt.clf()
    
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
    criterionContrastive = siamese.ContrastiveLoss(margin=self.margin,batch_size=self.batch_size)
    

    optimD = optim.Adam([{'params':self.DQ.parameters()}, {'params':self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimQ = optim.Adam([{'params':self.G.parameters()},{'params':self.DQ.parameters()}, {'params':self.Q.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimG = optim.Adam([{'params':self.G.parameters()}], lr=0.002, betas=(0.5, 0.99))
    optimClassifier = optim.Adam([{'params':self.Classifier.parameters()}], lr=0.002, betas=(0.5, 0.99))
    optimVAE =optim.Adam([{'params':self.Encoder.parameters()}, {'params':self.G.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    optimEncoder = optim.Adam([{'params':self.Encoder.parameters()}], lr=0.0002, betas=(0.5, 0.99))
    

    self.do_siamese(self.pre_epoch,50,optimEncoder,0,0,self.contrastive_loss_weight)
    
    
    num_iters=0
    for epoch in range(self.epoch):
        
        for num_iters, batch_data in enumerate(self.dataloader, 0):
          
            """Discriminator's real part"""
            optimD.zero_grad()
            
            x, _ = batch_data
    
            bs = x.size(0)
            x_real.data.resize_(x.size())
            label.data.resize_(bs,1)
            c.data.resize_(bs, self.c_size)
            z.data.resize_(bs,self.z_size)
            x_real.data.copy_(x)
            dq1 = self.DQ(x_real)
            x_real_result = self.D(dq1)
            label.data.fill_(1.0)
    
            loss_real = self.RF_loss_weight*criterionBCE(x_real_result, label)
            
            loss_real.backward(retain_graph=True)
          
            
            
            
              #print(x_real)
            c_en,z_en=self.Encoder(x_real)
            c_mean,c_logvar=torch.chunk(c_en,2,dim=1)
            z_mean,z_logvar=torch.chunk(z_en,2,dim=1)
            c_distribution =torch.distributions.Normal(c_mean, torch.exp(c_logvar))
            z_distribution= torch.distributions.Normal(z_mean, torch.exp(z_logvar))  
            c=c_distribution.sample()
            z=z_distribution.sample()
            G_input = torch.cat([z,c], 1).view(-1,self.c_size+self.z_size, 1, 1)
            
            x_fake = self.G(G_input)
            
            dq2 = self.DQ(x_fake.detach())
            x_fake_result = self.D(dq2)
            label.data.fill_(0.0)
            loss_fake = self.RF_loss_weight*criterionBCE(x_fake_result, label)
            loss_fake.backward(retain_graph=True)
            
            
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
            optimQ.step()
            
            
            optimVAE.zero_grad()
            
            optimEncoder.zero_grad()
            z_kl_divergence = torch.sum(0.5 * (z_mean**2 + torch.exp(z_logvar) - z_logvar -1))
            c_kl_divergence = 10*torch.sum(0.5 * (c_mean**2 + torch.exp(c_logvar) - c_logvar -1))
            KL=self.kl_loss_weight*(z_kl_divergence+c_kl_divergence)
            KL.backward(retain_graph=True)
            optimEncoder.step()
            vae_reconstruct_loss = self.reconstruction_loss_weight*criterionMSE(x_fake.view(x_fake.size(0),-1),x_real.view(x_real.size(0),-1))
            
            #print(vae_reconstruct_loss)
            vae_reconstruct_loss.backward(retain_graph=True)
            optimVAE.step()
            #self.do_siamese(1,1,optimEncoder,1,epoch,self.contrastive_loss_weight)  
            
            """randn"""  
            optimD.zero_grad()
            z.data=torch.randn(z.size()).cuda(1)#changing noise to train
            c.data=torch.randn(c.size()).cuda(1)
            G_input = torch.cat([z,c], 1).view(-1,self.c_size+self.z_size, 1, 1)
            x_fake = self.G(G_input)
            dq2 = self.DQ(x_fake.detach())
            x_fake_result = self.D(dq2)
            label.data.fill_(0.0)
            loss_fake = self.RF_loss_weight*criterionBCE(x_fake_result, label)
            loss_fake.backward()
            loss_real.backward()
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
            C_loss = self.c_loss_weight*criterionMSE(q,c)
            C_loss.backward(retain_graph=True)
            optimQ.step()
            
              
            #if num_iters%3==0:
            self.do_siamese(1,3,optimEncoder,1,epoch,self.contrastive_loss_weight)  
            """contrastive"""
            #self.do_siamese(1,10,optimEncoder,1)
        self.do_siamese(1,0,optimEncoder,2,epoch,self.contrastive_loss_weight)
          
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
        save_image(x_save.data,f1name, nrow=int(self.batch_size/6))
        G_input = torch.cat([z,c], 1).view(-1,self.c_size+self.z_size, 1, 1).cuda(1)
         
        x_save = self.G(G_input)
          
       
        f1name='./result_'+self.version+'/'+str(epoch)+'_rand.png'
          
        save_image(x_save.data,f1name, nrow=int(self.batch_size/6))
    
    

    
    
    gpath='./result_'+self.version+'/Generator.pkl'
    dpath='./result_'+self.version+'/Discriminator.pkl'
    qpath='./result_'+self.version+'/Q.pkl'
    dqpath='./result_'+self.version+'/DQ.pkl'
    enpath='./result_'+self.version+'/Encoder.pkl'
    torch.save(self.G.state_dict(), gpath)
    torch.save(self.D.state_dict(),dpath)
    torch.save(self.Q.state_dict(), qpath)
    torch.save(self.DQ.state_dict(), dqpath)
    torch.save(self.Encoder.state_dict(), enpath)