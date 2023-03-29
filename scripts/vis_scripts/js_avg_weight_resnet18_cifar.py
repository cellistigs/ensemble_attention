## Script to visualize the effect of increasing or decreasing the effective Variance weighting on standard ensemble training relative to training ensemble probabilities directly. 

import os
import numpy as np 
import itertools
from scipy.special import softmax
from ensemble_attention.metrics import AccuracyData
import matplotlib.pyplot as plt

output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),"vis_figures")
base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),"../../")
plt.style.use(os.path.join(base_dir,"etc/config/stylesheet.mplstyle"))
results_dir = os.path.join(base_dir,"results/")
scripts_dir = os.path.join(base_dir,"scripts/")

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

simul_ensembles = {
        "Ensemble_0":"ensemble0",
        "Ensemble_1":"ensemble1",
        "Ensemble_2":"ensemble2",
        "Ensemble_3":"ensemble3"
        }

gamma_ensembles = {
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

def simul_ensemble_performance(ensemble_dirs,datatype_suffix):
    """returns mean and standard deviation of performance for ensembles trained simultaneously. 

    :param ensemble_dirs: a dictionary of directories specifying where ensemble info is located.  
    :param datatype_suffix: a dictionary specifying which outputs give which suffixes. 
    :param ensemble_size: what size of ensembles to sample. 
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
    for resnet,path in ensemble_dirs.items():
        full_path = os.path.join(results_dir,"simultrained_ensemble_data",path)
        for setting in [ind,ood]:
            labels = np.load(os.path.join(full_path,setting["labels"]))
            probs = softmax(np.load(os.path.join(full_path,setting["probs"])),axis = 1)
            setting["accs"].append(acc.accuracy(probs,labels)) 
    ind_results = {"mean":np.mean(ind["accs"]),"std":np.std(ind["accs"])}
    ood_results = {"mean":np.mean(ood["accs"]),"std":np.std(ood["accs"])}
    return ind_results,ood_results

def gamma_performance(gammas,datatype_suffix):
    """returns performance for ensembles of a given gamma. 

    :param gammas: a dictionary indexed by gamma weights giving corresponding directory names.  
    :param datatype_suffix: a dictionary specifying which outputs give which suffixes. 
    :returns: two dictionaries, for ind vs ood, specifying the gamma values for each. 
    """
    outputs = []
    ind = {
            "labels":"ind_labels.npy",
            "probs": "ind_preds.npy",
            "gammas":{}
            }
    ood = {
            "labels":"ood_labels.npy",
            "probs": "ood_preds.npy",
            "gammas":{}
            }
    for gamma,path in gammas.items():
        full_path = os.path.join(scripts_dir,path)
        for setting in [ind,ood]:
            labels = np.load(os.path.join(full_path,setting["labels"]))
            probs = softmax(np.load(os.path.join(full_path,setting["probs"])),axis = 1)
            gamma_acc= acc.accuracy(probs,labels)
            setting["gammas"][gamma] = gamma_acc

    return ind,ood  

def plot(all_data_ind,all_data_ood):    
    """Plot all data on two subplots. 

    :param all_data_ind: dictionary of ind data. 
    :param all_data_ood: dictionary of ood data. 
    """
    fig,ax = plt.subplots(2,1,figsize=(6,10))
    all_data = [all_data_ind,all_data_ood]
    for i,data in enumerate(all_data):
        x = np.linspace(-4,4)
        ax[i].axhline(y = data["baselines"]["single_models"]["mean"],color = "black",label = "single model")
        single_upper = data["baselines"]["single_models"]["mean"]+data["baselines"]["single_models"]["std"]
        single_lower = data["baselines"]["single_models"]["mean"]-data["baselines"]["single_models"]["std"]
        ax[i].fill_between(x,single_upper,single_lower,color = "black",alpha = 0.5)

        #ax[i].axhline(y = data["baselines"]["ensembles"]["mean"],color = "red",label = "ensemble")
        #ax[i].axhline(y = data["baselines"]["ensembles"]["mean"]+data["baselines"]["ensembles"]["std"],color = "red",linestyle = "--")
        #ax[i].axhline(y = data["baselines"]["ensembles"]["mean"]-data["baselines"]["ensembles"]["std"],color = "red",linestyle = "--")

        ax[i].axhline(y = data["baselines"]["ensembles_joint"]["mean"],color = "blue",label = "ensemble")
        ensemble_upper = data["baselines"]["ensembles_joint"]["mean"]+data["baselines"]["ensembles_joint"]["std"]
        ensemble_lower = data["baselines"]["ensembles_joint"]["mean"]-data["baselines"]["ensembles_joint"]["std"]
        ax[i].fill_between(x,ensemble_upper,ensemble_lower,color ="blue",alpha = 0.5)
        ax[i].axvline(x = 0,color = "black",linestyle = "--")
        for gamma, model in data["gamma_weighted"]["gammas"].items():
            ax[i].plot(gamma,model,"X",color = "black",markersize = 10)

    ax[0].set_title("InD Performance")    
    ax[1].set_title("OOD Performance")
    ax[0].set_xlabel("$\gamma$ (predictive diversity penalty)")
    ax[1].set_xlabel("$\gamma$ (predictive diversity penalty)")
    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("Accuracy")
    ax[0].set_xlim([x[0],x[-1]])
    ax[1].set_xlim([x[0],x[-1]])
    #ax[0].set_xlim([-5.5,5.5])
    #ax[1].set_xlim([-5.5,5.5])
    plt.suptitle(r'ResNet 18 Ensemble (M = 4)'+"\n" +'Weighted Training: $L_{test}+ \gamma JS Avg()$',size = "xx-large")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"js_avg_weight_resnet18_cifar10.pdf"))


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
   "gamma_weighted":    
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
           }
   all_data_ood = {
           }
   # Calculate baselines
   ## single models:
   indresults,oodresults = single_model_performance(base_resnets,datatype_suffix)

   ## ensembles: 
   ens_indresults,ens_oodresults = ensemble_performance(base_resnets,datatype_suffix,ensemble_size=4)

   ## ensembles_joint:
   sim_ens_indresults,sim_ens_oodresults = simul_ensemble_performance(simul_ensembles,datatype_suffix)

   all_data_ind["baselines"] = {
           "single_models":indresults,
           "ensembles":ens_indresults,
           "ensembles_joint":sim_ens_indresults
           }
   all_data_ood["baselines"] = {
           "single_models":oodresults,
           "ensembles":ens_oodresults,
           "ensembles_joint":sim_ens_oodresults
           }

   # Calculate weighted model performance: 
   gamma_ens_indresults,gamma_ens_oodresults = gamma_performance(gamma_ensembles,datatype_suffix)
   print(gamma_ens_indresults)
   all_data_ind["gamma_weighted"] = gamma_ens_indresults
   all_data_ood["gamma_weighted"] = gamma_ens_oodresults

   # Plot: 
   plot(all_data_ind,all_data_ood)
   
if __name__ == "__main__":
    main()


