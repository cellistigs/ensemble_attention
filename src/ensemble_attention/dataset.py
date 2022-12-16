## smaller datasets that aren't cifar10
import os 
import zipfile
from PIL import Image
import numpy as np 
import pytorch_lightning as pl
import requests
from torch.utils.data import Dataset,DataLoader
import torch
from tqdm import tqdm 
import pandas as pd

class WineDataset(Dataset):
    def __init__(self,root_dir,transform =None,target_transform = None,seed = 0,test_size = 1000,color = None,train = False):
        if seed is not None:
            np.random.seed(seed)
        #super().__init__(root_dir,transform = transform,target_transform = target_transform)
        self.root_dir = root_dir
        self.raw_data_red = pd.read_csv(os.path.join(root_dir,"winequality-red.csv"),sep = ";").values
        self.raw_data_white = pd.read_csv(os.path.join(root_dir,"winequality-white.csv"),sep = ";").values
        self.transform = transform
        self.target_transform = target_transform
        if color == None:
            self.raw_data = np.concatenate([self.raw_data_red,self.raw_data_white],axis = 0)
        elif color == "red":    
            self.raw_data = self.raw_data_red
        elif color == "white":    
            self.raw_data = self.raw_data_white
        indices = np.random.permutation(range(len(self.raw_data)))    
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        if train == False:
            self.features = self.raw_data[test_indices,:-1]    
            self.targets = self.raw_data[test_indices,-1]
        else:    
            self.features = self.raw_data[train_indices,:-1]    
            self.targets = self.raw_data[train_indices,-1]
        
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self,idx):
        feature,target = self.features[idx],self.targets[idx]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:    
            target = self.target_transform(target)
        return feature,target    

class WineDataModule(pl.LightningDataModule):
    def __init__(self,args):
        super().__init__()
        self.hparams = args
        self.wine_predict = WineDataset(self.hparams.data_dir,train = False)
        self.wine_train = WineDataset(self.hparams.data_dir,train = True)
    def train_dataloader(self):
        return DataLoader(self.wine_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,)
    def val_dataloader(self):
        return DataLoader(self.wine_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,)
    def test_dataloader(self):
        return self.val_dataloader()

