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
        }


gamma_ensembles_resnet18 = {
        -1:"gamma_0", 
        -0.999:"gamma_0.001",
        -0.99:"gamma_0.01",
        -0.9:"gamma_0.1",
        -0.8:"gamma_0.2",
        -0.5:"gamma_0.5",
        -0.2:"gamma_0.8",
        0:"gamma_1",
        0.1:"gamma_1.1",
        0.2:"gamma_1.2",
        0.3:"gamma_1.3",
        0.4:"gamma_1.4",
        0.45:"gamma_1.45",
        0.5:"gamma_1.5",
        0.55:"gamma_1.55",
        0.6:"gamma_1.6",
        0.65:"gamma_1.65",
        0.7:"gamma_1.7",
        0.8:"gamma_1.8",
        0.9:"gamma_1.9",
        1:"gamma_2",
        2:"gamma_3",
        4:"gamma_5",
        }
gamma_ensembles_densenet121 = {
        -1:"gamma_0", 
        -0.9:"gamma_0.1",
        -0.8:"gamma_0.2",
        -0.5:"gamma_0.5",
        -0.2:"gamma_0.8",
        0:"gamma_1",
        0.1:"gamma_1.1",
        0.2:"gamma_1.2",
        0.3:"gamma_1.3",
        0.4:"gamma_1.4",
        0.45:"gamma_1.45",
        0.5:"gamma_1.5",
        0.55:"gamma_1.55",
        0.6:"gamma_1.6",
        0.65:"gamma_1.65",
        0.7:"gamma_1.7",
        0.8:"gamma_1.8",
        1:"gamma_2",
        2:"gamma_3",
        4:"gamma_5",
        }
gamma_ensembles_vgg13 = {
        -1:"0", 
        -0.8:"1", 
        -0.6:"2", 
        -0.4:"3", 
        -0.2:"4", 
        0:"5", 
        0.5:"6", 
        1:"7", 
        1.5:"8", 
        2:"9", 
        2.5:"10", 
        }
gamma_ensembles_wideresnet18 = {
        -1:"0", 
        -0.8:"1", 
        -0.6:"2", 
        -0.4:"3", 
        -0.2:"4", 
        0:"5", 
        0.5:"6", 
        1:"7", 
        #1.5:"8", 
        2:"9", 
        2.5:"10", 
        }

datafolder_resnet18 = "test_kl_data_resnet18_cifar10_1"
datafolder_densenet121 = "test_kl_data_densenet121_cifar10_1"
datafolder_vgg13 = "test_dkl_data_vgg13_cifar10_1/17-44-34"
datafolder_wideresnet = "test_kl_data_wideresnet18_cifar100"

all_gamma_ensembles = {"resnet":gamma_ensembles_resnet18,
                       "densenet":gamma_ensembles_densenet121,
                       "vgg":gamma_ensembles_vgg13,
                       "wideresnet":gamma_ensembles_wideresnet18}
all_datafolders = {"resnet":datafolder_resnet18,
                   "densenet":datafolder_densenet121,
                   "vgg":datafolder_vgg13,
                   "wideresnet":datafolder_wideresnet}

simul_ensembles = {"Ensemble_{}".format(i):os.path.join(reg_path,"5") for i,reg_path in enumerate(all_datafolders.values())
        }

all_settings = {"resnet":{"lims":[-1.2,3.7]},
                "densenet":{"lims":[-1.2,3.7]},
                "vgg":{"lims":[-1.2,3.7]},
                "wideresnet":{"lims":[-1.2,3.7]}}

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
    fig,ax = plt.subplots(1,4,figsize=(20,5))
    for i,reg in enumerate(["resnet","densenet","vgg","wideresnet"]):
        x = np.linspace(*all_settings[reg]["lims"])
        #single_upper = data["baselines"]["single_models"]["mean"]+data["baselines"]["single_models"]["se"]
        #single_lower = data["baselines"]["single_models"]["mean"]-data["baselines"]["single_models"]["se"]
        #ax[i].fill_between(x,single_upper,single_lower,color = "black",alpha = 0.5)
        #ax[i].axhline(y = data["baselines"]["single_models"]["mean"],color = "black",label = "single model")
        #
        #ensemble_upper = all_data_ind["baselines"]["ensembles_joint"]["mean"]+all_data_ind["baselines"]["ensembles_joint"]["se"]
        #ensemble_lower = all_data_ind["baselines"]["ensembles_joint"]["mean"]-all_data_ind["baselines"]["ensembles_joint"]["se"]
        #ax[i].fill_between(x,ensemble_upper,ensemble_lower,color = "blue",alpha = 0.5)
        #ax[i].axhline(y = all_data_ind["baselines"]["ensembles_joint"]["mean"],color = "blue",label = "ensemble ($\gamma = 1$)")


        ax[i].axvline(x = 0,linestyle = "--",color = "black")
        for gamma, model in all_data_ind["gamma_weighted"][reg]["gammas"].items():
            ax[i].plot(gamma,model,"X",color = "black",markersize = 10)
        ax[i].set_xlim([x[0],x[-1]])

    ax[0].set_title("ResNet18")    
    ax[1].set_title("DenseNet121")
    ax[2].set_title("VGG13")
    ax[3].set_title("WideResNet")
    ax[0].set_xlabel("$\gamma$ (predictive diversity penalty)")
    ax[1].set_xlabel("$\gamma$ (predictive diversity penalty)")
    ax[2].set_xlabel("$\gamma$ (predictive diversity penalty)")
    ax[3].set_xlabel("$\gamma$ (predictive diversity penalty)")
    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("Accuracy")
    plt.suptitle(r'ResNet 18 Ensemble (M = 4)' +"\n"+'Weighted Training: CIFAR10',size = "xx-large")
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"kl_weight_all_models_cifar10.pdf"))


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
    #sim_ens_indresults,sim_ens_oodresults = simul_ensemble_performance(simul_ensembles,datatype_suffix)

    #all_data_ind["baselines"] = {
    ##        "single_models":indresults,
    ##        "ensembles":ens_indresults,
    #        "ensembles_joint":sim_ens_indresults
    #        }
    #all_data_ood["baselines"] = {
    ##        "single_models":oodresults,
    ##        "ensembles":ens_oodresults,
    #        "ensembles_joint":sim_ens_oodresults
    #        }

    # Calculate weighted model performance: 
    for i,reg in enumerate(["resnet","densenet","vgg","wideresnet"]):
        gamma_ens_indresults,gamma_ens_oodresults = gamma_performance(all_gamma_ensembles[reg],datatype_suffix,reg)
        all_data_ind["gamma_weighted"][reg] = gamma_ens_indresults
        all_data_ood["gamma_weighted"][reg] = gamma_ens_oodresults

    # Plot: 
    plot_main(all_data_ind,all_data_ood)
    #plot_appx(all_data_ind,all_data_ood)
   
if __name__ == "__main__":
    main()


