## smaller datasets that aren't cifar10
import os 
import zipfile
from PIL import Image
import numpy as np 
import pytorch_lightning as pl
import requests
import sklearn.preprocessing
from torch.utils.data import Dataset,DataLoader,Subset
import torch
from torchvision.transforms import ToTensor,Compose,Normalize
from torchvision.datasets import MNIST
from tqdm import tqdm 
import pandas as pd

def replace_with_indices(df):
    df_mapped = df.copy()
    for column in df.columns:
        mapping = {value: index for index, value in enumerate(df[column].unique())}
        df_mapped[column] = df[column].map(mapping)
    return df_mapped

class AdultDataset(Dataset):
    def __init__(self,root_dir, transform = None,target_transform = None, seed = 0, test_size = 10000, train = False):
        self.seed = seed
        self.tensorize = Compose([ToTensor()])
        numerical_features = ["age","fnlwgt","educational-num","capital-gain","capital-loss","hours-per-week"]
        categorical_features = ["workclass","education","marital-status","occupation","relationship","race","gender","native-country"]
        target = "income"
        if seed is not None: 
            np.random.seed(seed)
        self.root_dir = root_dir    
        self.raw_data = pd.read_csv(os.path.join(root_dir,"adult.csv"),na_values ="?").dropna() ## remove datapoints with nan values. 
        num_data = self.raw_data[numerical_features].values
        ## now convert category labels to indices. 
        cat_data = replace_with_indices(self.raw_data[categorical_features])
        
        self.cat_sizes = [len(np.unique(di)) for di in cat_data.T]

        targets = self.raw_data[target].values==">50K" ## transform to binary. 
        self.transform = transform
        self.target_transform = target_transform

        # Determine data split to use. 
        indices = np.random.permutation(range(len(self.raw_data)))
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        train_features = num_data[train_indices,:],cat_data[train_indices,:]
        train_targets = targets[train_indices]

        ## get normalization 
        if transform is None:
            self.normalizer = self.get_normalizer(train_features[0])
        else:
            self.normalizer = self.get_normalizer(train_features[0],normtype = transform)

        if train == False:
            self.features = num_data[test_indices,:],cat_data[test_indices,:]
            self.targets = targets[test_indices]
        else:
            self.features = train_features
            self.targets = train_targets


    def get_normalizer(self,traindata,normtype="standard"):
        """Normalizes numerical data according standard scaling or quantile transform.

        """
        if normtype == "standard":
            normalizer = sklearn.preprocessing.StandardScaler()
        elif normtype == "quantile":
            normalizer = sklearn.preprocessing.QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles = max(min(traindata.shape[0] // 30, 1000),10),
                    subsample=int(1e9),
                    random_state = self.seed
                    )
        else:
            raise NotImplementedError("transform given is {}".format(normtype))
        normalizer.fit(traindata)
        return normalizer

    def __len__(self):
        return len(self.features[0])

    def __getitem__(self,idx):
        num,cat,target = self.features[0][idx],self.features[1][idx],self.targets[idx]
        if len(np.shape(num)) == 1:
            num = num.reshape(1,-1)
        if len(np.shape(cat)) == 1:    
            cat = cat.reshape(1,-1)
        num = self.normalizer.transform(num)
        if self.target_transform:
            target = self.target_transform(target)
        return self.tensorize(num),self.tensorize(cat),self.tensorize(target)

class WineDataset(Dataset):
    def __init__(self,root_dir,transform =None,target_transform = None,seed = 0,test_size = 1000,color = None,train = False):
        if seed is not None:
            np.random.seed(seed)
        #super().__init__(root_dir,transform = transform,target_transform = target_transform)
        self.root_dir = root_dir
        self.raw_data_red = pd.read_csv(os.path.join(root_dir,"winequality-red.csv"),sep = ";").values.astype(np.float32)
        self.raw_data_white = pd.read_csv(os.path.join(root_dir,"winequality-white.csv"),sep =";").values.astype(np.float32)
        self.transform = transform
        self.target_transform = target_transform
        if color == None:
            self.raw_data = np.concatenate([self.raw_data_red,self.raw_data_white],axis = 0)
        elif color == "red":    
            self.raw_data = self.raw_data_red
        elif color == "white":    
            self.raw_data = self.raw_data_white

        # normalize data. 
        mean,std = np.mean(self.raw_data,axis = 0), np.std(self.raw_data,axis = 0)
        self.normed_data = (self.raw_data-mean)/std
        indices = np.random.permutation(range(len(self.raw_data)))    
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        if train == False:
            self.features = self.normed_data[test_indices,:-1]    
            self.targets = self.normed_data[test_indices,-1:]
        else:    
            self.features = self.normed_data[train_indices,:-1]  
            self.targets = self.normed_data[train_indices,-1:]
        
    def __len__(self):
        return len(self.features)

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
        self.wine_predict = WineDataset(self.hparams.data_dir,train = False,color =
                self.hparams.winecolor,test_size=1000)
        self.wine_train = WineDataset(self.hparams.data_dir,train = True,color =
                self.hparams.winecolor,test_size=self.hparams.testset_size)
    def train_dataloader(self,shuffle = False,aug = False):
        return DataLoader(self.wine_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle = shuffle)
    def val_dataloader(self):
        return DataLoader(self.wine_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,)
    def test_dataloader(self):
        return self.val_dataloader()

class OneHotTransform():
    def __init__(self,num_classes):
        self.num_classes = num_classes
    def __call__(self,data):    
        return torch.nn.functional.one_hot(torch.tensor(data),self.num_classes).float()

class MNISTModule(pl.LightningDataModule):
    def __init__(self,args):
        super().__init__()
        self.hparams = args
        self.mnist_predict = MNIST(self.hparams.data_dir,train = False,transform=ToTensor(),target_transform=OneHotTransform(num_classes=10),download = True)
        self.mnist_train = MNIST(self.hparams.data_dir,train = True,transform =ToTensor(),target_transform=OneHotTransform(num_classes=10),download =True)
    def train_dataloader(self,shuffle = True,aug = False):
        return DataLoader(self.mnist_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle = shuffle)
    def val_dataloader(self):
        return DataLoader(self.mnist_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,)
    def test_dataloader(self):
        return self.val_dataloader()

class MNISTModule_class(pl.LightningDataModule):
    def __init__(self,args):
        super().__init__()
        self.hparams = args
        self.mnist_predict = MNIST(self.hparams.data_dir,train = False,transform=ToTensor(),download = True)
        self.mnist_train = MNIST(self.hparams.data_dir,train = True,transform =ToTensor(),download =True)
    def train_dataloader(self,shuffle = True,aug = False):
        return DataLoader(self.mnist_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle = shuffle)
    def val_dataloader(self):
        return DataLoader(self.mnist_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,)
    def test_dataloader(self):
        return self.val_dataloader()
class MNIST10000Module_class(pl.LightningDataModule):
    def __init__(self,args):
        super().__init__()
        self.hparams = args
        self.mnist_predict = MNIST(self.hparams.data_dir,train =
                False,transform=Compose([ToTensor(),Normalize((0.1307,),(0.3081))]),download = True)
        self.mnist_train = Subset(MNIST(self.hparams.data_dir,train = True,transform=Compose([ToTensor(),Normalize((0.1307,),(0.3081))]),download=True),np.arange(10000))
    def train_dataloader(self,shuffle = True,aug = False):
        return DataLoader(self.mnist_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle = shuffle)
    def val_dataloader(self):
        return DataLoader(self.mnist_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,)
    def test_dataloader(self):
        return self.val_dataloader()

class MNIST5000Module_class(pl.LightningDataModule):
    """Just select 5000 training examples, and test on 10000.
    """
    def __init__(self,args):
        super().__init__()
        self.hparams = args
        self.mnist_predict = MNIST(self.hparams.data_dir,train =
                False,transform=Compose([ToTensor(),Normalize((0.1307,),(0.3081))]),download = True)
        self.mnist_train = Subset(MNIST(self.hparams.data_dir,train =
            True,transform=Compose([ToTensor(),Normalize((0.1307,),(0.3081))]),download=True),np.arange(5000))
    def train_dataloader(self,shuffle = True,aug = False):
        return DataLoader(self.mnist_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle = shuffle)
    def val_dataloader(self):
        return DataLoader(self.mnist_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,)
    def test_dataloader(self):
        return self.val_dataloader()
class MNIST10000Module(pl.LightningDataModule):
    def __init__(self,args):
        super().__init__()
        self.hparams = args
        self.mnist_predict = MNIST(self.hparams.data_dir,train = False,transform=ToTensor(),target_transform=OneHotTransform(num_classes=10),download = True)
        self.mnist_train = Subset(MNIST(self.hparams.data_dir,train = True,transform =ToTensor(),target_transform=OneHotTransform(num_classes=10),download =True),np.arange(10000))
    def train_dataloader(self,shuffle = False,aug = False):
        return DataLoader(self.mnist_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle = shuffle)
    def val_dataloader(self):
        return DataLoader(self.mnist_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=False,
            pin_memory=True,)
    def test_dataloader(self):
        return self.val_dataloader()
