## Script to visualize the effect of increasing or decreasing the effective KL weighting on standard ensemble training relative to training ensemble probabilities directly. 

import os
import numpy as np 
import itertools
from scipy.special import softmax
from ensemble_attention.metrics import AccuracyData
import matplotlib.pyplot as plt

output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),"vis_figures")
base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),"../../")
here = os.path.join(os.path.abspath(os.path.dirname(__file__)))
plt.style.use(os.path.join(here,"../../etc/config/stylesheet.mplstyle"))
results_dir = os.path.join(base_dir,"results/")
acc = AccuracyData()


## For visualization scripts, we do not take parameters. 
base_resnets = {
    "ResNet 8":os.path.join(results_dir,"dkl_resnet8_indiv/ensemble_dkl_model_0_gamma_1_resnet8_"),#"resnet8_0/",
    "ResNet 8.7":os.path.join(results_dir,"dkl_resnet8_indiv/ensemble_dkl_model_1_gamma_1_resnet8_"),
    "ResNet 8.8":os.path.join(results_dir,"dkl_resnet8_indiv/ensemble_dkl_model_2_gamma_1_resnet8_"),
    "ResNet 8.9":os.path.join(results_dir,"dkl_resnet8_indiv/ensemble_dkl_model_3_gamma_1_resnet8_"),
    "labels":os.path.join(results_dir,"robust_results03-15-22_15:52.36_base_resnet18ind_labels.npy")
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
        }


gamma_ensembles_dkl = {
        -1:"1", 
        -0.8:"3", 
        -0.6:"5", 
        -0.4:"7", 
        -0.2:"9", 
        0:"11", 
        0.5:"13", 
        1:"15", 
        1.5:"17", 
        2:"19", 
        }

datafolder_dkl = "test_kl_data_rff_mnist"

all_gamma_ensembles = {"dkl":gamma_ensembles_dkl,
                       }
all_datafolders = {"dkl":datafolder_dkl,
                   }

simul_ensembles = {}
for i,(reg,reg_path) in enumerate(all_datafolders.items()): 
    try:
        simul_ensembles["Ensemble_{}".format(i)] = os.path.join(reg_path,all_gamma_ensembles[reg][0]) 
    except:     
        pass
        
all_settings = {"dkl":{"lims":[-1.2,3.2]},
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
        if not resnet == "labels":
            full_path = os.path.join(results_dir,"benchmark_ensemble_data",path)
            for setting in [ind]:
                labels = np.load(base_resnets["labels"])
                probs = softmax(np.load(full_path+setting["probs"]),axis = 1)
                setting["accs"].append(acc.accuracy(probs,labels)) 
    ind_results = {"mean":np.mean(ind["accs"]),"se":2*np.std(ind["accs"])/np.sqrt(len(base_resnets))}
    ood_results = {"mean":np.mean(ood["accs"]),"se":2*np.std(ood["accs"])/np.sqrt(len(base_resnets))}
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
    ind_results = {"mean":np.mean(ind["ens_accs"]),"se":2*np.std(ind["ens_accs"])/np.sqrt(len(base_resnets))}
    ood_results = {"mean":np.mean(ood["ens_accs"]),"se":2*np.std(ood["ens_accs"])/np.sqrt(len(base_resnets))}
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
        full_path = os.path.join(results_dir,path)
        for setting in [ind]:
            labels = np.load(os.path.join(full_path,setting["labels"]))
            probs = softmax(np.load(os.path.join(full_path,setting["probs"])),axis = 1)
            setting["accs"].append(acc.accuracy(probs,labels)) 
    ind_results = {"mean":np.mean(ind["accs"]),"se":2*np.std(ind["accs"])/np.sqrt(len(ensemble_dirs))}
    ood_results = {"mean":np.mean(ood["accs"]),"se":2*np.std(ood["accs"])/np.sqrt(len(ensemble_dirs))}
    return ind_results,ood_results

def gamma_performance(gammas,datatype_suffix,regularizer):
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
        full_path = os.path.join(results_dir,all_datafolders[regularizer],path)
        for setting in [ind]:
            labels = np.load(os.path.join(full_path,setting["labels"]))
            probs = softmax(np.load(os.path.join(full_path,setting["probs"])),axis = 1)
            gamma_acc= acc.accuracy(probs,labels)
            setting["gammas"][gamma] = gamma_acc

    return ind,ood  

def plot_main(all_data_ind,all_data_ood):    
    """Plot all data on two subplots. For the main fig.  

    :param all_data_ind: dictionary of ind data. 
    :param all_data_ood: dictionary of ood data. 
    """
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax = [ax]
    for i,reg in enumerate(["dkl"]):
        x = np.linspace(*all_settings[reg]["lims"])
        #single_upper = all_data_ind["baselines"]["single_models"]["mean"]+all_data_ind["baselines"]["single_models"]["se"]
        #single_lower = all_data_ind["baselines"]["single_models"]["mean"]-all_data_ind["baselines"]["single_models"]["se"]
        #ax[i].fill_between(x,single_upper,single_lower,color = "black",alpha = 0.5)
        #ax[i].axhline(y = all_data_ind["baselines"]["single_models"]["mean"],color = "black",label = "single model")
        #
        ensemble_upper = all_data_ind["baselines"]["ensembles_joint"]["mean"]+all_data_ind["baselines"]["ensembles_joint"]["se"]
        ensemble_lower = all_data_ind["baselines"]["ensembles_joint"]["mean"]-all_data_ind["baselines"]["ensembles_joint"]["se"]
        ax[i].fill_between(x,ensemble_upper,ensemble_lower,color = "blue",alpha = 0.5)
        ax[i].axhline(y = all_data_ind["baselines"]["ensembles_joint"]["mean"],color = "blue",label = "ensemble ($\gamma = 0$)")


        ax[i].axvline(x = 0,linestyle = "--",color = "black")
        for gamma, model in all_data_ind["gamma_weighted"][reg]["gammas"].items():
            ax[i].plot(gamma,model,"X",color = "black",markersize = 10)
        ax[i].set_xlim([x[0],x[-1]])

    ax[0].set_title("Jensen Gap")    
    ax[0].set_xlabel("$\gamma$ (predictive diversity penalty)")
    ax[0].set_ylabel("Accuracy")
    plt.suptitle(r'RFF (Width 100) Ensemble (M = 4)' +"\n"+'Weighted Training: MNIST',size = "xx-large")
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"all_weight_rff100_mnist.pdf"))


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

    all_data_ind = {"gamma_weighted":{}}

    all_data_ood = {"gamma_weighted":{}
            }
    # Calculate baselines
    ## single models:
    #indresults,oodresults = single_model_performance(base_resnets,datatype_suffix)

    ## ensembles: 
    #ens_indresults,ens_oodresults = ensemble_performance(base_resnets,datatype_suffix,ensemble_size=4)

    ## ensembles_joint:
    sim_ens_indresults,sim_ens_oodresults = simul_ensemble_performance(simul_ensembles,datatype_suffix)

    all_data_ind["baselines"] = {
            #"single_models":indresults,
    #        "ensembles":ens_indresults,
            "ensembles_joint":sim_ens_indresults
            }
    all_data_ood["baselines"] = {
            #"single_models":oodresults,
    #        "ensembles":ens_oodresults,
            "ensembles_joint":sim_ens_oodresults
            }

    # Calculate weighted model performance: 
    for i,reg in enumerate(["dkl"]):
        gamma_ens_indresults,gamma_ens_oodresults = gamma_performance(all_gamma_ensembles[reg],datatype_suffix,reg)
        all_data_ind["gamma_weighted"][reg] = gamma_ens_indresults
        all_data_ood["gamma_weighted"][reg] = gamma_ens_oodresults

    # Plot: 
    plot_main(all_data_ind,all_data_ood)
    #plot_appx(all_data_ind,all_data_ood)
   
if __name__ == "__main__":
    main()


