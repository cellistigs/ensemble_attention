## Create RFF models. 
import torch
from torch import nn
import numpy as np

class RFF_Projection(nn.Module):
    def __init__(self,input_size,project_size,sigma):
        """Create an rff projection

        """
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(project_size,input_size).float(),requires_grad = False)
        self.offset = nn.Parameter(torch.Tensor(project_size).float(),requires_grad = False)
        nn.init.normal_(self.weights,0,1/np.sqrt(sigma*input_size))
        nn.init.uniform_(self.offset,0,2*np.pi)
        
    def forward(self,x):    
        return torch.cos(torch.matmul(x,self.weights.T)+self.offset)

class RFF(nn.Module): 
    """Random Fourier Features module.

    """
    def __init__(self,input_size,project_size,output_size,sigma = 1):
        super().__init__()
        self.project = RFF_Projection(input_size,project_size,sigma) 
        self.regress = nn.Linear(project_size,output_size) 
        self.sigma = sigma

    def forward(self,x):    
        proj = self.project(x)
        out = self.regress(proj)
        return out

class LinearRegression(nn.Module):
    """Standard Linear Regression

    """
    def __init__(self,input_size,output_size):
        super().__init__()
        self.regress = nn.Linear(input_size,output_size) 

    def forward(self,x):    
        out = self.regress(x)
        return out

def linreg_wine():
    return LinearRegression(11,1)

def rff_regress_1000_wine():    
    return RFF(11,1000,1)

def rff_regress_10000_wine():    
    return RFF(11,10000,1)

def rff_regress_100000_wine():    
    return RFF(11,100000,1)

def rff_casregress_1000_mnist():    
    return RFF(784,1000,10)

def rff_casregress_10000_mnist():    
    return RFF(784,10000,10)

def rff_casregress_100000_mnist():    
    return RFF(784,100000,10)
