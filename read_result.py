# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:00:45 2018

@author: tommy
"""
import matplotlib.pyplot as plt


def read_result(version,epoch):
    fname='./result_'+version+'/result_'+version+'.txt'
    f=open(fname,'r')
    
    
    
    real_result=list()
    fake_result=list()
    c_result=list()
    G_result=list()
    recon_result=list()
    kl_result=list()
    for i in range(epoch):
        real_result.append([])
        fake_result.append([])
        c_result.append([])
        G_result.append([])
        recon_result.append([])
        kl_result.append([])
    #counter=0
    
    for line in f:
        #print(line[line.find(':')+1:line.find('/',6)])
        index=int(line[line.find(':')+1:line.find('/',6)])
        real_result[index].append(float(line[line.find('Real loss: [')+12:line.find(']')]))
        fake_result[index].append(float(line[line.find('fake loss: [')+13:line.find('],c')]))
        c_result[index].append(float(line[line.find('c loss')+9:line.find('], gen')]))
        #G_result[index].append(float(line[line.find('reconstruction_loss: [')+22:line.find('],time taken')-1]))
        G_result[index].append(float(line[line.find('generator_loss: [')+17:line.find('],recon')-1]))
        recon_result[index].append(float(line[line.find('reconstruction_loss: [')+22:line.find('],KL')-1]))
        kl_result[index].append(float(line[line.find('KL_loss: [')+10:line.find('\n')-2]))
    Dloss=[0]*epoch
    for i in range(epoch):
        Dloss[i]=real_result[i][0]+fake_result[i][0]
    
    f,(plt1, plt2,plt3,plt4,plt5,plt6)=plt.subplots(6)
    f.subplots_adjust(hspace = 1)
    plt1.set_title('real_loss')
    plt1.plot(range(epoch),real_result)
    #plt1.show()
    plt2.set_title('fake_loss')
    plt2.plot(range(epoch),fake_result)
    
    plt3.set_title('c loss')
    plt3.plot(range(epoch),c_result)
    #plt2.show()
    
    plt4.set_title('Generator_loss')
    plt4.plot(range(epoch),G_result)
    
    plt5.set_title('reconstruction_loss')
    plt5.plot(range(epoch),recon_result)
    #plt.savefig('./result_'+version+'/LOSS.png')
    #plt.show()
    
    plt6.set_title('kl_loss')
    plt6.plot(range(epoch),kl_result)
    plt.savefig('./result_'+version+'/LOSS.png')
    plt.show()
    
    
    plt.title('Real+fake vs reconstruction')
    plt.plot(range(epoch),Dloss)
    plt.plot(range(epoch),G_result)
    plt.savefig('./result_'+version+'/Real+fake vs reconstruction.png')
    plt.show()