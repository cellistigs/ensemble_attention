## smaller datasets that aren't cifar10
import os 
import zipfile
from PIL import Image
import numpy as np 
import pytorch_lightning as pl
import requests
from torch.utils.data import Dataset,Dataloader
import torch
from tqdm import dqdm 
import pandas as pd

def WineDataset(Dataset):
    def __init__(self,root_dir,transform =None,target_transform = None,seed = None,color = None):
        #super().__init__(root_dir,transform = transform,target_transform = target_transform)
        self.root_dir = root_dir
        self.raw_data_red = pd.read_csv(os.path.join(root_dir,"winequality-red.csv"),sep = ";").values()
        self.raw_data_white = pd.read_csv(os.path.join(root_dir,"winequality-white.csv"),sep = ";").values()
        self.transform = transform
        self.target_transform = target_transform
        if color == None:
            self.raw_data = np.concatenate([self.raw_data_red,self.raw_data_white],axis = 0)
        elif color == "red":    
            self.raw_data = self.raw_data_red
        elif color == "white":    
            self.raw_data = self.raw_data_white
        self.features = self.raw_data[:,:-1]    
        self.targets = self.raw_data[:,-1]
        
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self,idx):
        feature,target = self.features[i],self.targets[i]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:    
            target = self.target_transform(target)
        return feature,target    

        
        


