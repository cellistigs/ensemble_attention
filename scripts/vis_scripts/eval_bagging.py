## evaluate bagging vs. standard training. 
import numpy as np 
import os 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join(os.path.dirname(here),"outputs/")

labels = ""
standard_resnet8s = [
        "2023-03-29/21-50-43",
        "2023-03-29/22-00-56",
        "2023-03-29/22-11-12",
        "2023-03-29/22-21-34",
        "2023-03-30/15-04-42", 
        "2023-03-30/15-14-18",
        "2023-03-30/15-24-05",
        "2023-03-30/15-33-55",
        "2023-03-30/20-34-39", 
        "2023-03-30/20-44-13",
        "2023-03-30/20-53-48",
        ]
bagged_resnet8s = [
        "2023-03-29/22-31-52", 
        "2023-03-29/22-42-11",
        "2023-03-29/22-52-28",
        "2023-03-29/23-02-46",
        "2023-03-30/15-43-38",
        "2023-03-30/15-53-27",
        "2023-03-30/16-03-20",
        "2023-03-30/16-13-12",
        "2023-03-30/21-13-10",  
        "2023-03-30/21-22-54",
        "2023-03-30/21-32-37",
        ]

def extract_preds_labels(s,condition = "ind"):
    all_preds = []
    all_labels = []
    for si in s:
        preds = np.load(os.path.join(datadir,si,"{}_preds.npy".format(condition)))
        labels = np.load(os.path.join(datadir,si,"{}_labels.npy".format(condition)))
        all_preds.append(preds)
        all_labels.append(labels)
    assert np.all([all_labels[i] == all_labels[0] for i in range(len(all_labels))]); "sanity check should share labels."   
    assert np.all([np.allclose(np.sum(p,axis =1),1) for p in all_preds]); "should normalize already"
    return all_preds,labels

def evaluate(preds,labels):
    accs = []
    for p in preds:
        acc = len(np.where(np.argmax(p,axis = 1)==labels)[0])/float(len(labels))
        accs.append(acc)
    return accs    

def ens_evaluate(preds,labels):
    ens_preds = np.mean(preds,axis = 0)
    ens_acc = len(np.where(np.argmax(ens_preds,axis = 1)==labels)[0])/float(len(labels))
    return ens_acc

def bootstrap_ens_evaluate(preds,labels,members = 4,samples = 100):
    preds_subsamples = [np.array(preds)[np.random.choice(len(preds),members,replace =False).astype(int)] for i in range(samples)]
    ###### fix to iterate over. 
    ens_acc = []
    for preds in preds_subsamples:
        ens_preds = np.mean(preds,axis = 0)
        ens_acc.append(len(np.where(np.argmax(ens_preds,axis = 1)==labels)[0])/float(len(labels)))
    return ens_acc



def main():
    fig,ax = plt.subplots(1,2)
    all_preds = []
    for ci, condition in enumerate(["ind","ood"]):
        for si,s in enumerate([standard_resnet8s,bagged_resnet8s]):
            preds,labels = extract_preds_labels(s,condition)
            acc = evaluate(preds,labels)
            ens_acc = bootstrap_ens_evaluate(preds,labels)
            ax[ci].plot([si for a in acc],acc,"o",color = "blue",label = "indiv")
            #ax[ci].plot([si for a in ens_acc],ens_acc,"x",color = "red",label = "ens")
            ax[ci].errorbar(si,np.mean(ens_acc),yerr=np.std(ens_acc),marker="x",color = "red",label = "ens")
            ax[ci].set_xticks([0,1])
            ax[ci].set_xticklabels(["Standard\n Ensembling","Bagged\n Ensembles"])
            ax[ci].set_xlim([-0.25,1.25])
        ax[ci].set_title("Ensemble Comparison:\n {} Data".format(condition))    
        plt.legend()
    plt.savefig("eval_bagging.png")    
        

if __name__ == "__main__":    
    main()
