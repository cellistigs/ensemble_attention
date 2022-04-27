## Script to visualize the effect of increasing or decreasing the effective KL weighting on standard ensemble training relative to training ensemble probabilities directly. 

import os
import numpy as np 
import itertools
from scipy.special import softmax
from ensemble_attention.metrics import AccuracyData
import matplotlib.pyplot as plt

base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),"../../")
results_dir = os.path.join(base_dir,"results/")
acc = AccuracyData()


## For visualization scripts, we do not take parameters. 
base_resnets = {
    "ResNet 18":"robust_results11-15-21_02:46.11_base_resnet18",
    "ResNet 18.7":"robust_results11-15-21_03:01.22_base_resnet18",
    "ResNet 18.8":"robust_results11-15-21_03:16.31_base_resnet18",
    "ResNet 18.9":"robust_results11-15-21_03:31.40_base_resnet18",
    "ResNet 18.10":"robust_results11-15-21_03:46.53_base_resnet18"
}


datatype_suffix = {
        "InD Labels":"ind_labels.npy",
        "InD Probs": "ind_preds.npy",
        "OOD Labels":"ood_labels.npy",
        "OOD Probs": "ood_preds.npy",
        "meta":"_meta.json"
        }

def single_model_performance(base_resnets,datatype_suffix):
    """returns mean and standard deviation of performance for in and out of distribution. 

    :param base_resnets: a dictionary of base resnet stubs giving individual resnet outputs.
    :param datatype_suffix: a dictionary specifying which outputs give which suffixes. 
    :returns: two dictionaries, one involving the mean and standard deviation performance of single models on ind data, the other on ood data. 
    """
    outputs = []
    ind = {
            "labels":"ind_labels.npy",
            "probs": "ind_preds.npy",
            "accs":[]
            }
    ood = {
            "labels":"ood_labels.npy",
            "probs": "ood_preds.npy",
            "accs":[]
            }
    for resnet,path in base_resnets.items():
        full_path = os.path.join(results_dir,"benchmark_ensemble_data",path)
        for setting in [ind,ood]:
            labels = np.load(full_path+setting["labels"])
            probs = softmax(np.load(full_path+setting["probs"]),axis = 1)
            setting["accs"].append(acc.accuracy(probs,labels)) 
    ind_results = {"mean":np.mean(ind["accs"]),"std":np.std(ind["accs"])}
    ood_results = {"mean":np.mean(ood["accs"]),"std":np.std(ood["accs"])}
    return ind_results,ood_results

def ensemble_performance(base_resnets,datatype_suffix,ensemble_size):    
    """returns mean and standard deviation of performance for ensembles of a given size. 

    :param base_resnets: a dictionary of base resnet stubs giving individual resnet outputs.
    :param datatype_suffix: a dictionary specifying which outputs give which suffixes. 
    :param ensemble_size: what size of ensembles to sample. 
    """
    outputs = []
    ind = {
            "labels":"ind_labels.npy",
            "probs": "ind_preds.npy",
            "ens_probs":[],
            "ens_accs":[]
            }
    ood = {
            "labels":"ood_labels.npy",
            "probs": "ood_preds.npy",
            "ens_probs":[],
            "ens_accs":[]
            }
    for resnet,path in base_resnets.items():
        full_path = os.path.join(results_dir,"benchmark_ensemble_data",path)
        for setting in [ind,ood]:
            labels = np.load(full_path+setting["labels"])
            probs = softmax(np.load(full_path+setting["probs"]),axis = 1)
            setting["ens_probs"].append(probs)

    for setting in [ind,ood]:
        labels = np.load(full_path+setting["labels"])
        subsets = itertools.combinations(range(len(setting["ens_probs"])),ensemble_size)
        for ens_index in subsets:
            ens_avg = np.mean(np.stack([setting["ens_probs"][i] for i in ens_index],axis = 0),axis = 0)
            setting["ens_accs"].append(acc.accuracy(ens_avg,labels)) 
    ind_results = {"mean":np.mean(ind["ens_accs"]),"std":np.std(ind["ens_accs"])}
    ood_results = {"mean":np.mean(ood["ens_accs"]),"std":np.std(ood["ens_accs"])}
    return ind_results,ood_results

def plot(all_data_ind,all_data_ood):    
    """Plot all data on two subplots. 

    :param all_data_ind: dictionary of ind data. 
    :param all_data_ood: dictionary of ood data. 
    """
    fig,ax = plt.subplots(2,1)
    all_data = [all_data_ind,all_data_ood]
    for i,data in enumerate(all_data):
        ax[i].axhline(y = data["baselines"]["single_models"]["mean"],color = "black",label = "single model")
        ax[i].axhline(y = data["baselines"]["single_models"]["mean"]+data["baselines"]["single_models"]["std"],color = "black",linestyle = "--")
        ax[i].axhline(y = data["baselines"]["single_models"]["mean"]-data["baselines"]["single_models"]["std"],color = "black",linestyle = "--")

        ax[i].axhline(y = data["baselines"]["ensembles"]["mean"],color = "red",label = "ensemble")
        ax[i].axhline(y = data["baselines"]["ensembles"]["mean"]+data["baselines"]["ensembles"]["std"],color = "red",linestyle = "--")
        ax[i].axhline(y = data["baselines"]["ensembles"]["mean"]-data["baselines"]["ensembles"]["std"],color = "red",linestyle = "--")

    ax[0].set_title("InD Performance")    
    ax[1].set_title("OOD Performance")
    plt.suptitle("ResNet 18 (M = 4) Ensemble Weighting")
    plt.legend()
    plt.show()


def main():
   """This script plots the baseline performance on CIFAR 10 and CIFAR 10.1 of resnet 18s and ensembles of resnet 18s. Relative to these baselines, it also plots the performance of models trained using the D_{KL} interpretation of the ensemble training loss as a regularized single model training loss. 
   Each step of this script fills out a different field of the dictionary "all_data{ind,ood}" for ind or ood data separately. These dictionaries are structured as follows: 
   {
   "baselines":
       {
           single_models: {mean:val,std:val},
           ensembles: {mean:val,std:val},
           ensembles_joint: {mean:val, std:val},
       }
   "lambda_weighted":    
       {
           0: {mean:val,std:val}
           0.2: {mean:val,std:val}
           0.4: {mean:val,std:val}
           ...
           4: {mean:val,std:val}
       }
   }
   """

   all_data_ind = {
           "baselines":{},
           "lambda_weighted":{}
           }
   all_data_ood = {
           "baselines":{},
           "lambda_weighted":{}
           }
   # Calculate baselines
   ## single models:
   indresults,oodresults = single_model_performance(base_resnets,datatype_suffix)

   ## ensembles: 
   ens_indresults,ens_oodresults = ensemble_performance(base_resnets,datatype_suffix,ensemble_size=4)

   all_data_ind["baselines"] = {
           "single_models":indresults,
           "ensembles":ens_indresults
           }
   all_data_ood["baselines"] = {
           "single_models":oodresults,
           "ensembles":ens_oodresults
           }

   ## jointly trained ensembles: 

   # Calculate weighted model performance: 

   # Plot: 
   plot(all_data_ind,all_data_ood)
   
if __name__ == "__main__":
    main()


