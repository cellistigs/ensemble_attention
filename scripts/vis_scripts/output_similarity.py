## Measures the similarity of outputs between a set of networks on ind and ood test data. Assumed that one network generated labels for all others. 
import numpy as np 
import os 
import hydra
import matplotlib.pyplot as plt
here = os.path.abspath(os.path.dirname(__file__))
filename_dict={
        "ind":"ind_preds.npy",
        "ood":"ood_preds.npy",
        }

@hydra.main(config_path = os.path.join(here,"./configs/"),config_name = "output_similarity_config")
def main(cfg):
    for output in ["ind","ood"]:
        filelist = [os.path.join(fi,filename_dict[output]) for fi in [cfg.generator_path]+list(cfg.ensemble_paths)]
        grid_kld = grid_kl(filelist)
        grid_hammingd = grid_hamming(filelist)
        plot(grid_kld,grid_hammingd,cfg,output)    

def compute_kl(patha,pathb):    
    """Compute the average KL divergence across outputs between softmax output probabilities stored at patha and pathb.  

    :param patha: path to an array of size (examples,classes) containing softmax outputs for each class. 
    :param pathb:
    """
    proba = load_data(patha)
    probb = load_data(pathb)
    assert not np.any(proba==0) # no zeroes for the following calculation
    assert not np.any(probb==0)
    logdiv = np.log(proba/probb)
    elementwise = proba*logdiv
    kls =np.sum(elementwise,axis = 1)
    avg_kl = np.mean(kls) 
    return avg_kl

def compute_hamming(patha,pathb):
    """Compute the average hamming distance between predictions stored at patha and pathb.  

    :param patha: path to an array of size (examples,classes) containing softmax outputs for each class. 
    :param pathb:
    """
    proba = load_data(patha)
    probb = load_data(pathb)
    assert not np.any(proba==0) # no zeroes for the following calculation
    assert not np.any(probb==0)
    predsa = np.argmax(proba,axis = 1)
    predsb = np.argmax(probb,axis = 1)
    return sum(predsa!=predsb)/len(predsa)

def load_data(path):
    """load in data. 

    """
    return np.load(path)

def grid_kl(pathlist):
    """given a list of paths, computes, the average KL divergence across outputs pairwise, and stores them in a grid. 

    """
    grid = np.array([[None for i in range(len(pathlist))] for j in range(len(pathlist))])
    for fi,f in enumerate(pathlist):
        for gi,g in enumerate(pathlist):
            grid[fi,gi] = compute_kl(f,g)
    return grid.astype(float)        

def grid_hamming(pathlist):
    """given a list of paths, computes, the average hamming distance across outputs pairwise, and stores them in a grid. 

    """
    grid = np.array([[None for i in range(len(pathlist))] for j in range(len(pathlist))])
    for fi,f in enumerate(pathlist):
        for gi,g in enumerate(pathlist):
            grid[fi,gi] = compute_hamming(f,g)
    return grid.astype(float)        

def plot(grid_kl,grid_hamming,cfg,output):    
    """Plot the average kl divergence as a grid., and also the average hamming distance.  

    """
    fig,ax = plt.subplots(2,1)
    ## plot distances with kl grid
    gridded = ax[0].matshow(grid_kl)
    ax[0].set_title(cfg.plot_title_1)
    ax[0].set_xticklabels(['']+list(cfg.labels),rotation=30)
    ax[0].set_yticklabels(['']+list(cfg.labels))
    plt.colorbar(gridded,ax = ax[0])
    ## plot distances with hamming grid
    gridded = ax[1].matshow(grid_hamming)
    ax[1].set_title(cfg.plot_title_2)
    ax[1].set_xticklabels(['']+list(cfg.labels),rotation=30)
    ax[1].set_yticklabels(['']+list(cfg.labels))
    plt.colorbar(gridded,ax = ax[1])

    plt.tight_layout()
    plt.savefig("{}_avg_kl_hamming".format(output))

if __name__ == "__main__":    
    main()
