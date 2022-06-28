## Computes Esnembel performance vs.individual model performance. 
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
        model_perfs = [model_correct(mi,cfg.generator_path,output) for mi in cfg.ensemble_paths]
        ens_perf = ens_correct(cfg.ensemble_paths,cfg.generator_path,output)
        print("{} Performance: \nModels: {}\nEnsemble: {} ".format(output,model_perfs,ens_perf))

def model_correct(modelpath,generator,output):    
    """Distance to the ground truth labels from the mean output labels

    """
    ens_preds = load_data(os.path.join(modelpath,filename_dict[output]))
    generator_preds = load_data(os.path.join(generator,filename_dict[output])) ## examples, classes
    ens_labels = np.argmax(ens_preds,axis = 1)
    generator_labels = np.argmax(generator_preds,axis = 1)
    return sum(ens_labels==generator_labels)/len(ens_labels)
def ens_correct(ensemble_path_list,generator,output):    
    """Distance to the ground truth labels from the mean output labels

    """
    ens_preds = np.stack([load_data(os.path.join(p,filename_dict[output])) for p in ensemble_path_list],axis = 0)
    ens_means = np.mean(ens_preds,axis = 0)
    generator_preds = load_data(os.path.join(generator,filename_dict[output])) ## examples, classes
    generator_labels = np.argmax(generator_preds,axis = 1)
    ens_labels = np.argmax(ens_means,axis = 1)
    return sum(ens_labels==generator_labels)/len(ens_labels)

def load_data(path):
    """load in data. 

    """
    return np.load(path)


if __name__ == "__main__":    
    main()
