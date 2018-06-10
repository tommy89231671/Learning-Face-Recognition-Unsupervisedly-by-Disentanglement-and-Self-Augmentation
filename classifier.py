
import torch.nn as nn
import torch
from torch.autograd import Variable
class Classifier(nn.Module):
    def __init__(self,z_size):
        super(Classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(z_size*2, z_size),
            nn.LeakyReLU(0.2),
            nn.Linear(z_size,5),
             nn.LeakyReLU(0.2),
            nn.Linear(5,2),    
            nn.Softmax()
            
        )
        #self.fc1 = nn.Linear(z_size*2, z_size)#inputsize,outputsize
        #self.fc2 = nn.Linear(z_size, 5)
        #self.leakyrelu=nn.LeakyReLU(0.2)
    def forward(self,x):
        
        return self.main(x)