# Estimate the bias average single model brier score and variance of the js models. load data as in https://github.com/Lightning-AI/lightning/issues/2909
import json
import numpy as np
from tqdm import tqdm
import os
import hydra
import torch 
import numpy 
from ensemble_attention.module import CIFAR10EnsembleJS_Avg_Module
from cifar10_ood.data import CIFAR10Data,CIFAR10_1Data,CINIC10_Data
import pytorch_lightning as pl

script_dir = os.path.abspath(os.path.dirname(__file__))
results_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)),"results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets = {"cifar10":CIFAR10Data,"cifar10.1":CIFAR10_1Data,"cinic10":CINIC10_Data}

resultdict = {
        -1.4:"multirun/2022-11-09/00-21-36/0",
        -1.9:"multirun/2022-11-09/00-21-36/1",
        -2.4:"multirun/2022-11-09/00-21-36/2",
        -2.9:"multirun/2022-11-09/00-21-36/3",
        -3.4:"multirun/2022-11-09/00-21-36/4",
        -3.9:"multirun/2022-11-09/00-21-36/5",
        1.4:"multirun/2022-11-09/00-21-36/6",
        1.9:"multirun/2022-11-09/00-21-36/7",
        2.4:"multirun/2022-11-09/00-21-36/8",
        2.9:"multirun/2022-11-09/00-21-36/9",
        3.4:"multirun/2022-11-09/00-21-36/10",
        3.9:"multirun/2022-11-09/00-21-36/11",
        -0.4:"multirun/2022-11-09/00-21-36/12",
        -0.8:"multirun/2022-11-09/00-21-36/13",
        0.4:"multirun/2022-11-09/00-21-36/14",
        0.8:"multirun/2022-11-09/00-21-36/15",
        }

def get_resultspaths(resultpaths):
    """Get the paths to all individual results so we can iterate over them

    :returns: a dictionary of resultpaths and corresponding gamma values. 
    """
    dict_to_return = {}
    for gamma,path in resultpaths.items():
        dict_to_return[os.path.join(script_dir,"../",path)] = gamma
    return dict_to_return 


def get_modelpath(resultspath):
    """Given the path to a set of results, recovers the path to the corresponding saved model. 

    :param resultspath:
    :returns: modelpath
    """
    metafile = os.path.join(resultspath,"meta.json")
    with open(metafile,"r") as f:
        config = json.load(f)
    ckpt = os.listdir(config["model_save_path"])
    assert len(ckpt) == 1,"hope there's only one"  
    return os.path.join(config["model_save_path"],ckpt[0])    


def get_model(modelpath,args):    
    """Given the path to saved model, reloads that model into memory and returns it. 

    :param modelpath:
    """
    ckpt = torch.load(modelpath)

    model = CIFAR10EnsembleJS_Avg_Module(args)
    model.load_state_dict(ckpt["state_dict"])
    return model
    

def eval_model(model,device,args,dataset_name = "cifar10"):    
    """given a dkl model, evaluates each of its individual constituent models on a given dataset (default: cifar10)

    :param model:
    :param dataset:
    :returns: returns the M individual sets of data outputs. 
    """
    softmax = torch.nn.Softmax(dim=1)
    data = datasets[dataset_name](args)
    model.to(device)
    model.eval()
    with torch.no_grad():
        preds = []
        all_labels = []
        for m in model.models:
            modelpreds = []
            for idx,batch in tqdm(enumerate(data.test_dataloader())):
                ims = batch[0].to(device)
                labels = batch[1]#.to(device)
                pred = m(ims.to(device)) ## this is the logit.  
                modelpreds.append(pred.to("cpu"))
            preds.append(np.concatenate(modelpreds,axis=0))    
    return preds, labels

@hydra.main(config_path=os.path.join(script_dir,"../../configs/"),config_name="run_default_gpu")
def main(args):
    modeltype = "resnet18"
    resultspaths= get_resultspaths(resultdict)
    print("gammas: {}".format(resultspaths))
    for resultspath,gamma in resultspaths.items():
        print("creating labels for: {}, gamma = {}".format(modeltype, gamma))
        modelpath = get_modelpath(resultspath)
        model = get_model(modelpath,args)
        outputs = eval_model(model,device,args)
        labelpath = os.path.join(results_dir,"js_avg_indiv_true","ensemble_js_avg_model_{}_{}_ind_{}".format(gamma,modeltype,"labels"))
        np.save(labelpath,outputs[1].to("cpu"))
        for i in range(len(outputs[0])):
            path = os.path.join(results_dir,"js_avg_indiv_true","ensemble_js_avg_model_{}_{}_{}_ind_{}".format(i,gamma,modeltype,"preds"))
            np.save(path,outputs[0][i])

if __name__ == "__main__":
    main()