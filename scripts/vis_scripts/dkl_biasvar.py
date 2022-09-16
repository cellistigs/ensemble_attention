# Estimate the bias average single model brier score and variance of the dkl models. 
import json
import os
import hydra
import torch 
import numpy 
from ensemble_attention.module import CIFAR10EnsembleDKLModule
from cifar10_ood.data import CIFAR10Data,CIFAR10_1Data,CINIC10_Data
import pytorch_lightning as pl

script_dir = os.path.abspath(os.path.dirname(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets = {"cifar10":CIFAR10Data,"cifar10.1":CIFAR10_1Data,"cinic10":CINIC10_Data}

def get_resultspaths():
    """Get the paths to all individual results so we can iterate over them

    :returns: a dictionary of resultpaths and corresponding gamma values. 
    """

def get_modelpath(resultspath):
    """Given the path to a set of results, recovers the path to the corresponding saved model. 

    :param resultspath:
    :returns: modelpath
    """
    metafile = os.path.join(resultspath,"meta.json")
    with open(metafile,"r") as f:
        config = json.load(f)
    return config["model_save_path"]    


def get_model(modelpath,args):    
    """Given the path to saved model, reloads that model into memory and returns it. 

    :param modelpath:
    """
    model = CIFAR10EnsembleDKLModule.load_from_checkpoint(checkpoint_path=modelpath,hparams=args)
    return model
    

def eval_model(model,device,dataset_name = "cifar10"):    
    """given a dkl model, evaluates each of its individual constituent models on a given dataset (default: cifar10)

    :param model:
    :param dataset:
    :returns: returns the M individual sets of data outputs. 
    """
    softmax = torch.nn.Softmax(dim=1)
    data = dataset[dataset_name]
    model.eval()
    with torch.no_grad():
        preds = []
        for m in model.models:
            modelpreds = []
            for idx,batch in tqdm(enumerate(data.test_dataloader())):
                ims = batch[0].to(device)
                labels = batch[1]#.to(device)
                pred = m(ims.to(device)) ## this is the logit.  
                modelpreds.append(pred)
            preds.append(pred)    
    return preds, labels

@hydra.main(config_path=os.path.join(script_dir,"../../configs/"),config_name="run_default_gpu")
def main(args):
    #resultspaths= get_resultspath
    resultspaths = ["/Users/taigaabe/Code/ensemble_attention/results/test_kl_data_resnet18_cifar10_1/gamma_0/"]
    for resultspath in resultspaths:
        modelpath = get_modelpath(resultspath)
        model = get_model(modelpath)
        outputs = eval_model(model)


