## Copied from interp_ensembles.
import os
import hydra
from tqdm import tqdm
from argparse import ArgumentParser
import datetime
import torch
import json
import numpy as np
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from ensemble_attention.callback import Check_GradNorm

from ensemble_attention.module import CIFAR10Module,CIFAR10EnsembleModule,\
    CIFAR10AttentionEnsembleModule,CIFAR10AttentionEnsembleSkipModule,CIFAR10AttentionEnsembleMLPSkipModule,\
    CIFAR10EnsembleDKLModule,CIFAR10EnsemblePAC2BModule,CIFAR10EnsembleJS_Unif_Module,CIFAR10EnsembleJS_Avg_Module, \
    CIFAR10EnsembleDKL_Avg_Module, CIFAR10EnsembleJGAPModule, CIFAR10EnsembleJGAPLModule, \
    RegressionSingleModel, RegressionEnsembleModel, RegressionEnsemble_JGModel, \
    ClassasRegressionSingleModel,ClassasRegressionEnsembleModel, ClassasRegressionEnsemble_JGModel, \
    ClassasRegressionSingleModelOneHot, ClassasRegressionEnsembleModelOneHot, ClassasRegressionEnsembleJGAPModelOneHot

#from ensemble_attention.module import CIFAR100Module,CIFAR100EnsembleModule,CIFAR100EnsembleDKLModule, \
#    CIFAR100EnsemblePAC2BModule,CIFAR100EnsembleJS_Unif_Module,CIFAR100EnsembleJS_Avg_Module

#from ensemble_attention.callback import Check_GradNorm
from ensemble_attention.callback import GradNormCallbackSplit, GradNormCallback
from pytorch_lightning.plugins import ddp_plugin

from ensemble_attention.dataset import WineDataModule,MNISTModule,MNISTModule_class
from cifar10_ood.data import CIFAR10Data,CIFAR10_1Data,CINIC10_Data,CIFAR10_CData, CIFAR100Data, CIFAR100CoarseData


modules = {"base":CIFAR10Module,
        "ensemble":CIFAR10EnsembleModule,  # train time ensemble
        "ensemble_dkl":CIFAR10EnsembleDKLModule,  #jgap ensemble with kl divergence
        "ensemble_jgap":CIFAR10EnsembleJGAPModule,  #jgap ensemble with jgap
        "ensemble_jgapl":CIFAR10EnsembleJGAPLModule,  #jgap ensemble with jgap w logit averaging.
        "ensemble_p2b":CIFAR10EnsemblePAC2BModule,  # Ortega ensemble*
        "ensemble_js_unif":CIFAR10EnsembleJS_Unif_Module,  # co-training ensemble*
        "ensemble_js_avg":CIFAR10EnsembleJS_Avg_Module,  # Mishtal ensemble*
        "ensemble_dkl_avg":CIFAR10EnsembleDKL_Avg_Module,  # Webb ensembling* (not using logits for E)
        "attention":CIFAR10AttentionEnsembleModule,
        "attentionskip":CIFAR10AttentionEnsembleSkipModule,
        "attentionmlpskip":CIFAR10AttentionEnsembleMLPSkipModule,
        "regress":RegressionSingleModel,
        "regress_ensemble":RegressionEnsembleModel,
        "regress_ensemble_dkl":RegressionEnsemble_JGModel,
        "casregress":ClassasRegressionSingleModel,
        "casregress_ensemble":ClassasRegressionEnsembleModel,
        "casregress_ensemble_dkl":ClassasRegressionEnsemble_JGModel,
        "casregress_onehot":ClassasRegressionSingleModelOneHot,
        "casregress_ensemble_onehot":ClassasRegressionEnsembleModelOneHot,
        "casregress_ensemble_jgap_onehot":ClassasRegressionEnsembleJGAPModelOneHot,
        #"base_100":CIFAR100Module,
        #"ensemble_100":CIFAR100EnsembleModule,  # train time ensemble
        #"ensemble_dkl_100":CIFAR100EnsembleDKLModule,  #jgap ensemble
        #"ensemble_p2b_100":CIFAR100EnsemblePAC2BModule,  # Ortega ensemble
        #"ensemble_js_unif_100":CIFAR100EnsembleJS_Unif_Module,  # co-training ensemble
        #"ensemble_js_avg_100":CIFAR100EnsembleJS_Avg_Module,  # Mishtal ensemble
}

script_dir = os.path.abspath(os.path.dirname(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def traindata_eval(model,ind_data,device,softmax = True, store_split = False):
    """Custom evaluation function to output logits as arrays from models given the trained model on the training data. Used to generate training examples from random labels. 

    :param model: a model from interpensembles.modules. Should have a method "calibration" that outputs predictions (logits) and labels given images and labels. 
    :param ind_data: an instance of a data class (like CIFAR10Data,CIFAR10_1Data) that has a corresponding test_dataloader. 
    :param device: device to run computations on.
    :param softmax: whether or not to apply softmax to predictions. 
    :returns: four arrays corresponding to predictions (array of shape (batch,classes)), and labels (shape (batch,)) for ind and ood data respectively. 

    """
    ## This is the only place where we need to worry about devices. The model should already know what device to use. 
    all_preds= []
    all_labels = []

    ## model, cifart10data,cifart10_1data,
    model.eval()
    with torch.no_grad():
        for idx,batch in tqdm(enumerate(ind_data.train_dataloader(shuffle=False,aug=False))):
            ims = batch[0].to(device)
            labels = batch[1].to(device)
            pred, label = model.calibration((ims,labels),store_split = True)
            ## to cpu
            predarray = pred.cpu().numpy() ## 256x10
            labelarray = label.cpu().numpy() ## 
            all_preds.append(predarray)
            all_labels.append(labelarray)

    all_preds_array = np.concatenate(all_preds,axis = 0)
    all_labels_array = np.concatenate(all_labels,axis = 0)
    return all_preds_array,all_labels_array

def custom_eval(model,ind_data,ood_data,device,softmax = True, store_split=False):
    """Custom evaluation function to output logits as arrays from models given the trained model, in distribution data and out of distribution data. 

    :param model: a model from interpensembles.modules. Should have a method "calibration" that outputs predictions (logits) and labels given images and labels. 
    :param ind_data: an instance of a data class (like CIFAR10Data,CIFAR10_1Data) that has a corresponding test_dataloader. 
    :param ood_data: an instance of a data class (like CIFAR10Data,CIFAR10_1Data) that has a corresponding test_dataloader.
    :param device: device to run computations on.
    :param softmax: whether or not to apply softmax to predictions. 
    :returns: four arrays corresponding to predictions (array of shape (batch,classes)), and labels (shape (batch,)) for ind and ood data respectively. 

    """
    ## This is the only place where we need to worry about devices. The model should already know what device to use. 
    all_preds_ind = []
    all_labels_ind = []
    all_preds_ood = []
    all_labels_ood = []

    ## model, cifart10data,cifart10_1data,
    model.eval()
    with torch.no_grad():
        for idx,batch in tqdm(enumerate(ind_data.test_dataloader())):
            ims = batch[0].to(device)
            labels = batch[1].to(device)
            pred, label = model.calibration((ims, labels), store_split=store_split)
            ## to cpu
            predarray = pred.cpu().numpy() ## 256x10
            labelarray = label.cpu().numpy() ## 
            all_preds_ind.append(predarray)
            all_labels_ind.append(labelarray)
        for idx,batch in tqdm(enumerate(ood_data.test_dataloader())):
            ims = batch[0].to(device)
            labels = batch[1].to(device)
            pred,label = model.calibration((ims,labels), store_split=store_split)
            ## to cpu
            predarray = pred.cpu().numpy() ## 256x10
            labelarray = label.cpu().numpy() ## 
            all_preds_ood.append(predarray)
            all_labels_ood.append(labelarray)

    all_preds_ind_array = np.concatenate(all_preds_ind,axis = 0)
    all_labels_ind_array = np.concatenate(all_labels_ind,axis = 0)
    all_preds_ood_array = np.concatenate(all_preds_ood,axis = 0)
    all_labels_ood_array = np.concatenate(all_labels_ood,axis = 0)
    return all_preds_ind_array,all_labels_ind_array,all_preds_ood_array,all_labels_ood_array

@hydra.main(config_path = os.path.join(script_dir,"../configs/"),config_name = "run_default_gpu_tinyimagenet")
def main(args):

    ## Set seeds if given.  
    if args.seed is not None:
        seed_everything(args.seed)
#    if torch.cuda.is_available():
#        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    ## Set up logging. 
    if args.logger == "wandb":
        logger = WandbLogger(name=args.classifier, project=args.test_set)
    elif args.logger == "tensorboard":
        logger = TensorBoardLogger(args.test_set, name=args.classifier)

    ## Configure checkpoint and trainer: 
    checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", save_last=False, dirpath = os.path.join(script_dir,"../","models",args.classifier,args.module,datetime.datetime.now().strftime("%m-%d-%y"),datetime.datetime.now().strftime("%H_%M_%S")))

    trainerargs = {
        #"default_root_dir":os.path.join(script_dir,"../","models",args.classifier,args.module),    
        "fast_dev_run":bool(args.dev),
        "logger":logger if not bool(args.dev + args.test_phase) else None,
        "deterministic":bool(args.deterministic),
        "weights_summary":None,
        "log_every_n_steps":1,
        "max_epochs":args.max_epochs,
        "checkpoint_callback":checkpoint,
        "precision":args.precision,
        "auto_lr_find": bool(args.get('auto_lr_find', 0)),
        "gradient_clip_val": args.get('gradient_clip_val', 0),
        }

    if torch.cuda.is_available():
        print("training on GPU(s) ".format(args.get('gpus', -1)))
        trainerargs["gpus"] = args.get('gpus', -1)

    if args.get('accelerator', False):
        trainerargs['accelerator'] = args.accelerator

        if args.accelerator == "ddp":
            # do we need this for args?
            args.batch_size = int(args.batch_size / max(1, args.gpus))
            args.num_workers = int(args.num_workers / max(1, args.gpus))
            trainerargs['plugins'] = [ddp_plugin.DDPPlugin(find_unused_parameters=False)]

    if args.callbacks:
       #trainer = Trainer(**trainerargs,callbacks = [Check_GradNorm()])
        trainer = Trainer(**trainerargs,callbacks = [GradNormCallbackSplit()])
    else:
        trainer = Trainer(**trainerargs)

    ## define arguments for each model class: 
    all_args = {"hparams":args} 
    if args.module == "base":
        pass
    elif args.module == "ensemble":
        pass
    elif args.module == "attention":
        pass

    ## we can load in pretrained models stored as weights. 
    if bool(args.test_phase) and not bool(args.pretrained): ## if loading from checkpoints: 
        if args.module == "base":
            model = modules[args.module].load_from_checkpoint(checkpoint_path=args.checkpoint,hparams = args)
        elif args.module == "ensemble":    
            model = modules[args.module].load_from_checkpoint(checkpoint_path=args.checkpoint,hparams = args)
        elif args.module == "attention":    
            model = modules[args.module].load_from_checkpoint(checkpoint_path=args.checkpoint,hparams = args)
        ## Really should be the case for anything
        else:    
            #model = modules[args.module].load_from_checkpoint(checkpoint_path=args.checkpoint,hparams = args)
            ckpt = torch.load(args.checkpoint)
            model = modules[args.module](**all_args)
            model.load_state_dict(ckpt["state_dict"])

    else: ## if training from scratch or loading from state dict:    
        model = modules[args.module](**all_args)
        ## if loading from state dictionary instead of checkpoint: 
        if bool(args.pretrained):
            if args.pretrained_path is None:
                state_dict = os.path.join(
                    script_dir,"../","models",
                    "cifar10_models", "state_dicts", args.classifier + ".pt"
                )
            else:     
                state_dict = args.pretrained_path
            model.model.load_state_dict(torch.load(state_dict))

    ## what dataset should we evaluate on?
    if args.test_set == "CIFAR10":
        ind_data = CIFAR10Data(args)
    elif args.test_set == "wine":    
        ind_data = WineDataModule(args)
    elif args.test_set == "mnist":    
        ind_data = MNISTModule_class(args)
    elif args.test_set == "CIFAR100":    
        ind_data = CIFAR100Data(args)
    elif args.test_set == "CIFAR100Coarse":
        ind_data = CIFAR100CoarseData(args)
    else:
        raise ValueError("Unknown dataset")

    if args.ood_dataset == "cifar10_1":
        ood_data = CIFAR10_1Data(args,version =args.version)
    elif args.ood_dataset == "cinic10":    
        ood_data = CINIC10_Data(args)
    elif args.ood_dataset == "cifar10_c":    
        assert args.corruption, "for cifar10_c, corruption must be given."
        assert args.level, "for cifar10_c, level must be given"
        ood_data = CIFAR10_CData(args)
    elif args.ood_dataset == "wine":    
        ood_data = WineDataModule(args)
    elif args.ood_dataset == "mnist":    
        ood_data = MNISTModule_class(args)
    elif args.ood_dataset == "CIFAR100":    
        ood_data = CIFAR100Data(args)
    elif args.ood_dataset == "CIFAR100Coarse":
        ood_data = CIFAR100CoarseData(args)
    else:
        raise ValueError("Unknown dataset")

    ## do we train the model or not? 
    if bool(args.test_phase) or bool(args.random_eval):
        pass
    else:
        trainer.fit(model, ind_data)

    ## testing and evaluation :
    data = {"in_dist_acc":None,"out_dist_acc":None}
    data["in_dist_acc"] = trainer.test(model, ind_data.test_dataloader())[0]["acc/test"]
    data["out_dist_acc"] = trainer.test(model, ood_data.test_dataloader())[0]["acc/test"]

    store_split = args.get('store_split', False)
    preds_ind, labels_ind, preds_ood, labels_ood = custom_eval(model,ind_data,ood_data,device,softmax = bool(args.softmax), store_split=store_split)
    preds_train,labels_train = traindata_eval(model,ind_data,device,softmax = bool(args.softmax), store_split=store_split)

    full_path = "." #os.path.join(results_dir,"robust_results{}_{}_{}".format(datetime.datetime.now().strftime("%m-%d-%y_%H:%M.%S"),args.module,args.classifier))
    np.save("ind_preds",preds_ind)
    np.save("ind_labels",labels_ind)
    np.save("train_preds",preds_train)
    np.save("train_labels",labels_train)
    if args.ood_dataset == "cifar10_1":
        np.save("ood_preds",preds_ood)
        np.save("ood_labels",labels_ood)
    elif args.ood_dataset == "cinic10":    
        np.save("ood_cinic_preds",preds_ood)
        np.save("ood_cinic_labels",labels_ood)
    elif args.ood_dataset == "cifar10_c":    
        np.save("ood_cifar10_c_{}_{}_preds".format(args.corruption,args.level),preds_ood)
        np.save("ood_cifar10_c_{}_{}_labels".format(args.corruption,args.level),labels_ood)
    elif args.ood_dataset == "wine":
        np.save("ood_wine_{}_{}_preds".format(args.corruption,args.level),preds_ood)
        np.save("ood_wine_{}_{}_labels".format(args.corruption,args.level),labels_ood)
    elif args.ood_dataset == "mnist":
        np.save("ood_mnist_{}_{}_preds".format(args.corruption,args.level),preds_ood)
        np.save("ood_mnist_{}_{}_labels".format(args.corruption,args.level),labels_ood)
    elif args.ood_dataset == "CIFAR100":
        np.save("ood_cifar100_{}_{}_preds".format(args.corruption,args.level),preds_ood)
        np.save("ood_cifar100_{}_{}_labels".format(args.corruption,args.level),labels_ood)
    elif args.ood_dataset == "CIFAR100Coarse":
        np.save("ood_cifar100coarse_{}_{}_preds".format(args.corruption, args.level), preds_ood)
        np.save("ood_cifar100coarse_{}_{}_labels".format(args.corruption, args.level), labels_ood)
    else:
        raise Exception("option for ood dataset not recognized.")
    ## write metadata
    metadata = {}
    metadata["model_save_path"] = trainer.checkpoint_callback.dirpath
    with open("meta.json","w") as f:
        json.dump(metadata,f)


if __name__ == "__main__":
    main()

