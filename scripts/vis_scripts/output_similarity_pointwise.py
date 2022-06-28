## MEasure the output similarity across the deep ensemble as a function of distance to "ground truth" output
import numpy as np
import os
import matplotlib.pyplot as plt 
import hydra
from scipy.stats import gaussian_kde

here = os.path.abspath(os.path.dirname(__file__))
filename_dict={
        "ind":"ind_preds.npy",
        "ood":"ood_preds.npy",
        }

@hydra.main(config_path = os.path.join(here,"./configs/"),config_name = "output_similarity_config_2")
def main(cfg):
    for output in ["ind","ood"]:
        filelist = [os.path.join(fi,filename_dict[output]) for fi in [cfg.generator_path]+list(cfg.ensemble_paths)]
        ens_vars = variance(cfg.ensemble_paths,output) # variances of shape (examples,)
        ens_dists_kl = distance_kl(cfg.ensemble_paths,cfg.generator_path,output)
        tf = correct(cfg.ensemble_paths,cfg.generator_path,output)
        ens_dists_hamming = distance_hamming(cfg.ensemble_paths,cfg.generator_path,output)
        plot(ens_dists_kl,ens_dists_hamming,ens_vars,tf,output,cfg.plot_title_1,cfg.plot_title_2)

def variance(ensemble_path_list,output):    
    """Variance of the ensemble itself.

    """
    ens_preds = np.stack([load_data(os.path.join(p,filename_dict[output])) for p in ensemble_path_list],axis = 0)
    ens_vars = np.sum(np.var(ens_preds,axis = 0),axis = -1)
    return ens_vars
    
def correct(ensemble_path_list,generator,output):    
    """Distance to the ground truth labels from the mean output labels

    """
    ens_preds = np.stack([load_data(os.path.join(p,filename_dict[output])) for p in ensemble_path_list],axis = 0)
    ens_means = np.mean(ens_preds,axis = 0)
    generator_preds = load_data(os.path.join(generator,filename_dict[output])) ## examples, classes
    generator_labels = np.argmax(generator_preds,axis = 1)
    ens_labels = np.argmax(ens_means,axis = 1)
    return ens_labels==generator_labels

def distance_hamming(ensemble_path_list,generator,output):    
    """Distance to the ground truth labels from the mean output labels

    """
    ens_preds = np.stack([load_data(os.path.join(p,filename_dict[output])) for p in ensemble_path_list],axis = 0)
    #ens_means = np.mean(ens_preds,axis = 0)
    generator_preds = load_data(os.path.join(generator,filename_dict[output])) ## examples, classes
    generator_preds = generator_preds[None,:,:]
    predsa = np.argmax(ens_preds,axis = 2)
    predsb = np.argmax(generator_preds,axis = 2)
    ham =  sum(predsa!=predsb)/predsa.shape[0]
    return ham

def distance_kl(ensemble_path_list,generator,output):    
    """Distance to the softmax output from the mean output 

    """
    ens_preds = np.stack([load_data(os.path.join(p,filename_dict[output])) for p in ensemble_path_list],axis = 0)
    ens_means = np.mean(ens_preds,axis = 0)
    generator_preds = load_data(os.path.join(generator,filename_dict[output])) ## examples, classes
    logdiv = np.log(generator_preds/ens_means)
    elementwise = generator_preds*logdiv
    kls =np.sum(elementwise,axis = 1)
    #dist = np.linalg.norm(generator_preds-ens_means,axis= -1 )
    return kls

def load_data(path):
    """load in data. 

    """
    return np.load(path)

def plot(ens_dists_kl,ens_dists_ham,ens_vars,tf,output,title_kl,title_ham):     
    """plot 

    """
    eps = 0.01
    fig, ax = plt.subplots(3,1,figsize=(15,15))

    kde_exp = -0.125 ## 1/4*nb_dims
    ## generate grid
    xmin_kl = np.min(ens_dists_kl)
    xmax_kl = np.max(ens_dists_kl)
    xmin_ham = np.min(ens_dists_ham)
    xmax_ham = np.max(ens_dists_ham)
    ymin = np.min(ens_vars)
    ymax = np.max(ens_vars)
    true = np.where(tf)
    false = np.where(~tf)

    ## for kl
    X, Y = np.mgrid[xmin_kl:xmax_kl:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    kernel = gaussian_kde(np.vstack([ens_dists_kl[true],ens_vars[true]]), bw_method=len(ens_dists_kl) ** (kde_exp))
    Z = np.reshape(kernel(positions).T, X.shape)
    ax[0].imshow(np.rot90(Z),extent=[xmin_kl, xmax_kl, ymin, ymax],cmap=plt.cm.gist_earth_r,aspect='auto')
    ax[0].plot(ens_dists_kl[true], ens_vars[true], 'k.', markersize=2,color="red")
    ax[0].set_xlim([xmin_kl-eps, xmax_kl+eps])
    ax[0].set_ylim([ymin-eps, ymax+eps])
    ax[0].set_title("Correct: "+title_kl)
    ax[0].set_xlabel("KL (ground_truth|ensemble avg)")
    ax[0].set_ylabel("ensemble variance")

    X, Y = np.mgrid[xmin_kl:xmax_kl:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    kernel = gaussian_kde(np.vstack([ens_dists_kl[false],ens_vars[false]]), bw_method=len(ens_dists_kl) ** (kde_exp))
    Z = np.reshape(kernel(positions).T, X.shape)
    ax[1].imshow(np.rot90(Z),extent=[xmin_kl, xmax_kl, ymin, ymax],cmap=plt.cm.gist_earth_r,aspect='auto')
    ax[1].plot(ens_dists_kl[false], ens_vars[false], 'k.', markersize=2,color="red")
    ax[1].set_xlim([xmin_kl-eps, xmax_kl+eps])
    ax[1].set_ylim([ymin-eps, ymax+eps])
    ax[1].set_title("Incorrect: " + title_kl)
    ax[1].set_xlabel("KL (ground_truth|ensemble avg)")
    ax[1].set_ylabel("ensemble variance")

    ## for hamming 
    X, Y = np.mgrid[xmin_ham:xmax_ham:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    kernel = gaussian_kde(np.vstack([ens_dists_ham,ens_vars]), bw_method=len(ens_dists_ham) ** (kde_exp))
    Z = np.reshape(kernel(positions).T, X.shape)
    ax[2].imshow(np.rot90(Z),extent=[xmin_ham, xmax_ham, ymin, ymax],cmap=plt.cm.gist_earth_r,aspect='auto')
    ax[2].plot(ens_dists_ham, ens_vars, 'k.', markersize=2,color="red")
    ax[2].set_xlim([xmin_ham-eps, xmax_ham+eps])
    ax[2].set_ylim([ymin-eps, ymax+eps])
    ax[2].set_title(title_ham)
    ax[2].set_xlabel("avg. hamming distance (labels)")
    ax[2].set_ylabel("ensemble variance")

    #plt.scatter(ens_dists_kl,ens_vars)
    plt.tight_layout()
    plt.savefig("{}_pointwise_per_dist".format(output))
    plt.close()

if __name__ == "__main__":    
    main()
