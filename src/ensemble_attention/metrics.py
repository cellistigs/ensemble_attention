## Tools to help calculate calibration related metrics given a predictions and labels.  
import numpy as np
import torch

class BrierScoreData(object):
    """Calculates brier score. 

    """
    def __init__(self):
        pass

    def brierscore(self,prob,target):
        """Given an array of probabilities, `prob`, (batch,dim) and an array of targets `target` (dim), calculates the brier score as if we were in the binary case: take the predicted probability of the target class, and just calculate based on that.
        :param prob: array of probabilities per class. 
        :param target: list/array of true targets. 

        """
        probs = prob[np.arange(len(target)),target]
        deviance = probs-np.ones(probs.shape)
        
        return np.mean(deviance**2)

    def brierscore_multi(self,prob,target):
        """The "original" brier score definition that accounts for other classes explicitly. Note the range of this test is 0-2. 
        :param prob: array of probabilities per class. 
        :param target: list/array of true targets. 

        """
        target_onehot = np.zeros(prob.shape)
        target_onehot[np.arange(len(target)),target] = 1 ## onehot encoding. 
        deviance = prob-target_onehot
        return np.mean(np.sum(deviance**2,axis = 1))

class VarianceData(object):
    """Calculates variance/related metrics. In particular, this is the variance in the confidence of the top predicted label. 

    """
    def __init__(self,modelprefix,data):
        """Takes a modelprefix that specifies the kinds of models over which we will be calculating variance. 

        :param modelprefix: a string specifying what model names should start with. 
        :param data: ind/ood, specifying if this class is calculating variance for in or out of distribution data. 
        """
        self.modelprefix = modelprefix
        self.models = {} ## dict of dicts- key is modelname, value is dictionary of preds/labels.
        self.data = data
    
    def register(self,preds,labels,modelname):
        """Takes predictions and labels. as a rough check, will assert that the labels match those that have already been registered.  
        :param preds: assumes softmax applied. 
        :param labels: vector of integers. 
        :param modelname: enforce that models we are calculating diversity over have the same model prefix. 
        """
        assert modelname.startswith(self.modelprefix); "modelname must start with modelprefix"
        for model,modeldata in self.models.items():
            assert np.all(labels == modeldata["labels"]); "labels must match already registered." 
            assert np.allclose(np.sum(preds,axis = 1), 1); "predictions must be probabilities"
        self.models[modelname] = {"preds":preds,"labels":labels}    
    
    def mean_conf(self):
        """Calculates mean confidence across all softmax output. 

        :return: array of shape samples, classes giving per class variance. 
        """
        all_probs = []
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0)    
        return np.mean(array_probs, axis = 0)

    def variance(self):
        """Calculates variance in confidence across all softmax output. 

        :return: array of shape samples,classes giving per class variance. 
        """
        all_probs = []
        for model,modeldata in self.models.items():
            probs = modeldata["preds"]
            all_probs.append(probs)
        array_probs = np.stack(all_probs,axis = 0)    
        return np.var(array_probs, axis = 0)

    def variance_confidence(self):
        """Calculates an array of shape (sample, 2) that gives the mean and variance of confidence estimates together per datapoint.   

        """
        target = self.models[list(self.models.keys())[0]]["labels"]

        all_vars = self.variance()
        all_conf = self.mean_conf()
        tprob_vars = all_vars[np.arange(len(target)),target]
        tprob_confs = all_conf[np.arange(len(target)),target]
        return np.stack([tprob_confs,tprob_vars],axis = 1)

    def expected_variance(self):
        """Calculates expected variance across y|x and x. This is just selecting the variance in the top probability for y|x, and then averaging over all examples for x. 

        :return: scalar expected variance
        """
        target = self.models[list(self.models.keys())[0]]["labels"]

        all_vars = self.variance()
        tprob_vars = all_vars[np.arange(len(target)),target]
        return np.mean(tprob_vars)

class AccuracyData(object):
    """Calculates accuracy related metrics. 

    """
    def __init__(self):
        pass
    
    def accuracy(self,prob,target):
        """Given predictions (example,class) and targets (class), will calculate the accuracy.  

        """
        selected = np.argmax(prob, axis = 1)
        correct = target == selected
        accuracy = sum(correct)/len(target)
        return accuracy

class NLLData(object):
    """Calculates the negative log likelihood of the data. 

    """
    def __init__(self):
        pass
    
    def nll(self,prob,target,normalize = False):
        """Given predictions (example,class) and targets (class), will calculate the negative log likelihood. Important here that the probs are expected to be outputs of softmax functions.   

        """
        probs = prob[np.arange(len(target)),target]
        logprobs = np.log(probs)

        nll = -sum(logprobs)
        if normalize:
            nll= nll/len(logprobs)
        return nll

class CalibrationData(object):
    """Initializes an object to bin predictions that are fed to it in batches.  

    """
    def __init__(self,binedges):
        """Initialize with a set of floats giving the interval spacing between different bins

        :param binedges: list of edges of the bins, not including 0 and 1. Will create intervals like `[[0,binedges[0]),[binedges[0],binedges[1]),...,[binedges[-1],100]]`  
        """
        assert binedges[0] > 0 and binedges[-1] < 1, "bin edges must be strictly within limits."
        assert np.all(np.diff(binedges)> 0), "bin edges must be ordered" 
        assert type(binedges) == list
        padded = [0] + binedges + [1] 
        self.binedges = [(padded[i],padded[i+1]) for i in range(len(padded)-1)]

    def bin(self,prob,target):
        """Given predictions  (example, class) and targets  (class), will bin them according to the binedges parameter.
        Returns a dictionary with keys giving bin intervals, and values another dictionary giving the accuracy, confidence, and number of examples in the bin. 
        """
        data = self.analyze_batch(prob,target)
        ## first let's divide the data by bin: 
        bininds = np.array(list(data["bin"]))
        bin_assigns = [np.where(bininds == i) for i in range(len(self.binedges))]
        ## now we want per-bin stats:
        all_stats = {} 
        for ai,assignments in enumerate(bin_assigns):
            bin_card = len(assignments[0]) 
            name = self.binedges[ai]
            if bin_card == 0:
                bin_conf = np.nan
                bin_acc = np.nan
            else:    
                bin_conf = sum(data["maxprob"][assignments])/bin_card
                bin_acc = sum(data["correct"][assignments])/bin_card
            all_stats[name] = {"bin_card":bin_card,"bin_conf":bin_conf,"bin_acc":bin_acc}
        return all_stats    

    def ece(self,prob,target):
        """Calculate the expected calibration error across bins given a probability and target. 

        """
        all_stats = self.bin(prob,target)
        ece_nonnorm = 0 
        for interval,intervalstats in all_stats.items():
            if intervalstats["bin_card"] == 0:
                continue
            else:
                factor = intervalstats["bin_card"]*abs(intervalstats["bin_acc"]-intervalstats["bin_conf"])
                ece_nonnorm += factor
        ece = ece_nonnorm/len(target)         
        return ece


    def getbin(self,data):
        """Halt and return the index where your maximum prediction fits into a bin. 

        """
        index = len(self.binedges)-1 
        for b in self.binedges[::-1]: ## iterate in reverse order
            if data >= b[0]:
                break
            else:
                index -= 1
        return index        

    def analyze_batch(self,prob,target):
        """Given a matrix of class probabilities (batch, class) and a target (class), returns calibration related info about that datapoint: 
        {"prob":prob,"target":target,"correct":bool,"bin":self.binedges[index]}

        """
        assert len(prob.shape) == 2
        assert prob.shape[0] == len(target)
        maxprob,maxind = np.amax(prob,axis = 1),np.argmax(prob,axis= 1)
        correct = maxind == target
        binind = map(self.getbin,maxprob)
        return {"maxprob":maxprob,"maxind":maxind,"target":target,"correct":correct,"bin":binind}

class Model_D_KL(object):
    """Calculate the KL divergence between the empirical distribution of log probs output by individual model predictions that together are averaged to form an ensemble. 

    :param cost_format: output either "torch", or "numpy", indicating if we want to create a metric for training on torch tensors or experimenting with numpy arrays. 
    """
    def __init__(self,cost_format):
        """Decide if we're going to use torch or numpy kl:
        """
        assert cost_format in ["torch","numpy"], "format must be either 'torch' or 'numpy'"
        self.format = cost_format

    def kl(self,probs,labels):     
       if self.format == "numpy":
           return self.kl_numpy(probs,labels)
       elif self.format == "torch":
           return self.kl_torch(probs,labels)
  
    def kl_numpy(self,probs,labels):    
        """
        :param probs: an iterable of probabilities, each of which has identical shape (batch, class)
        :param labels: a set of labels of shape (batch,)
        :return: kls of shape (batch,)
        """
        M = len(probs) 
        prob_array = np.stack(probs,axis = -1)
        correct_probs = prob_array[np.arange(prob_array.shape[0]),labels,:] # shape of (batch,models)
        normalization = np.sum(correct_probs,axis = 1)
        normed_probs = correct_probs/normalization[:,None]
        kls = (1./M)*np.sum(np.log(1./M)-np.log(normed_probs),axis = 1) 
        return kls

    def kl_torch(self,probs,labels):    
        """
        :param probs: an iterable of probabilities, each of which has identical shape (batch, class)
        :param labels: a set of labels of shape (batch,)
        """
        M = len(probs) 
        prob_array = torch.stack(probs,axis = -1)
        correct_probs = prob_array[np.arange(prob_array.shape[0]),labels,:] # shape of (batch,models)
        normalization = torch.sum(correct_probs,axis = 1)[:,None]
        normed_probs = torch.div(correct_probs,normalization)
        kls = (1./M)*torch.sum(torch.sub(np.log(1./M),torch.log(normed_probs)),axis = 1) 
        return kls

class Model_Ortega_Variance(object):
    """Calculate diversity used as an upper bound on ensemble loss in https://proceedings.mlr.press/v151/ortega22a/ortega22a.pdf
    This is the "tighter" upper bound that includes an additional hmax term. 

    :param cost_format: output either "torch", or "numpy", indicating if we want to create a metric for training on torch tensors or experimenting with numpy arrays. 
    """
    def __init__(self,cost_format):
        """

        """
        assert cost_format in ["torch","numpy"], "format must be either `torch` or `numpy`"
        self.format = cost_format

    def var(self,probs,labels):
        if self.format == "numpy":
            return self.var_numpy(probs,labels)
        elif self.format == "torch":
            return self.var_torch(probs,labels)

    def var_numpy(self,logprobs,labels):
        """Follows implementation from https://github.com/PGM-Lab/2022-AISTATS-diversity/blob/63df2e5f29cdaefe49626439bbe13289f37eed36/baselines/utils/varianceBound.py#L75
        :param logprobs: an iterable of probabilities, each of which has identical shape (batch, class)
        :param labels: a set of labels of shape (batch,)
        """
        M = len(logprobs)
        log_prob_array = np.stack(logprobs,axis = 0)
        correct_log_probs = log_prob_array[:,np.arange(log_prob_array.shape[1]),labels] # (models,batch)

        ## get scaling factor for tighter bound:
        logmean = np.log(np.sum(np.exp(correct_log_probs),axis = 0))-np.log(M)
        logmax = np.max(correct_log_probs,axis = 0)
        inc = logmean-logmax
        inc = np.clip(inc,-10,-0.01)
        hmax1 = (inc/np.power(1-np.exp(inc),2))/(np.exp(logmax)**2)
        hmax2 = (np.power(np.exp(inc)*(1-np.exp(inc)),-1))/(np.exp(logmax)**2)
        hmax = hmax1 + hmax2

        variance = np.mean(np.exp(2*correct_log_probs-2*logmax),axis = 0)
        for j in range(M):
            variance -= np.mean(np.exp(correct_log_probs+correct_log_probs[j,:]-2*logmax),axis =0) / M
        full_variance =  variance    
        return full_variance


    def var_torch(self,logprobs,labels):
        """
        :param logprobs: an iterable of probabilities, each of which has identical shape (batch, class)
        :param labels: a set of labels of shape (batch,)
        """
        M = len(logprobs)
        log_prob_array = torch.stack(logprobs,axis = 0)
        correct_log_probs = log_prob_array[:,np.arange(log_prob_array.shape[1]),labels] # (models,batch)

        logmean = torch.log(torch.sum(torch.exp(correct_log_probs),axis=0))-np.log(M)
        logmax = torch.max(correct_log_probs,0)[0]
        inc = logmean-logmax
        inc = torch.clip(inc,-10.-0.01)
        hmax1 = (inc/torch.pow(1-torch.exp(inc),2))/(torch.exp(logmax)**2)
        hmax2 = (torch.pow(torch.exp(inc)*(1-torch.exp(inc)),-1))/(torch.exp(logmax)**2)
        hmax = hmax1+hmax2

        variance = torch.mean(torch.exp(2*correct_log_probs-2*logmax),axis = 0)
        for j in range(M):
            variance -= torch.mean(torch.exp(correct_log_probs+correct_log_probs[j,:]-2*logmax),axis = 0)/M
        #full_variance = hmax*variance
        full_variance = variance
        return full_variance

class Model_JS_Unif(object):
    """Calculate diversity using the multi-distribution Jensen-Shannon divergence between all ensemble members. Among other sources, this is the diversity measure used in Diversity and Co-training. 
    
    """
    def __init__(self,cost_format):
        """

        """
        assert cost_format in ["torch","numpy"], "format must be either `torch` or `numpy`"
        self.format = cost_format

    def js_unif(self,probs):    
        """probs should be an iterable of probabilities, each of which has identical shape (batch,classes)

        """
        if self.format == "numpy":
            return self.js_unif_numpy(probs)
        elif self.format == "torch":
            return self.js_unif_torch(probs)

    def js_unif_numpy(self,probs):
        probs_array =np.stack(probs,axis = 0)
        avg_probs = np.mean(probs_array,axis = 0)

        logprobs = np.log(probs_array) # shape (models,batch,classes)
        avg_logprobs = np.log(avg_probs) # shape (batch,classes)

        shannon_inf = -probs_array*logprobs
        avg_shannon_inf = -avg_probs*avg_logprobs

        entropy = np.sum(shannon_inf,axis = -1)
        avg_entropy = np.sum(avg_shannon_inf,axis = -1)

        jsd = avg_entropy - np.mean(entropy,axis = 0)

        return jsd

    def js_unif_torch(self,probs):
        """
        """
        probs_array = torch.stack(probs,axis = 0)
        avg_probs = torch.mean(probs_array,axis = 0)

        logprobs = torch.log(probs_array)
        avg_logprobs = torch.log(avg_probs)

        shannon_inf = -probs_array*logprobs
        avg_shannon_inf = -avg_probs*avg_logprobs

        entropy = torch.sum(shannon_inf,axis = -1)
        avg_entropy = torch.sum(avg_shannon_inf,axis = -1)

        jsd = avg_entropy - torch.mean(entropy,axis = 0)

        return jsd

class Model_JS_Avg(object):
    """Calculates diversity using the Jensen-Shannon-divergence between an ensemble member and the ensemble prediction. Among other sources, this is the diversity measure used in Jensen-Shannon Divergence in Ensembles of Concurrently-Trained Neural Networks. 

    """
    def __init__(self,cost_format):
        """

        """
        assert cost_format in ["torch","numpy"], "format must be either `torch` or `numpy`"
        self.format = cost_format

    def js_avg(self,probs):    
        if self.format == "numpy":
            return self.js_avg_numpy(probs)
        elif self.format == "torch":
            return self.js_avg_torch(probs)

    def js_avg_numpy(self,probs):    
        probs_array =np.stack(probs,axis = 0)
        #avg_probs = np.mean(probs_array,axis = 0)
        masked_arrays = [np.ma.array(probs_array,mask = False) for i in range(len(probs))]
        for i,m in enumerate(masked_arrays):
            m.mask[i,:,:] = True 
        avg_probs = np.stack([m.mean(axis = 0) for m in masked_arrays],axis = 0) # shape (batch,classes)

        logprobs = np.log(probs_array) # shape (models,batch,classes)
        avg_logprobs = np.log(avg_probs) # shape (avg_probs,batch,classes)

        combined_probs = 0.5*probs_array+0.5*avg_probs ## broadcast average probabilities on first dim. 
        log_combined_probs = np.log(combined_probs)

        avg_kl_terms = np.sum(avg_probs*(avg_logprobs-log_combined_probs),axis = -1)
        single_kl_terms = np.sum(probs_array*(logprobs-log_combined_probs),axis = -1)
        single_model_jsds = 0.5*(avg_kl_terms+single_kl_terms) ## shape (models,batch)

        avg_jsd = np.mean(single_model_jsds,axis = 0)
        return avg_jsd
    
    def js_avg_torch(self,probs):
        probs_array = torch.stack(probs,axis = 0)
        inds = []
        for j in range(len(probs)):
            inds.append([i for i in range(len(probs)) if i is not j])
        sub_arrays = [probs_array[ind] for ind in inds]
        avg_probs = torch.stack([torch.mean(si,axis = 0) for si in sub_arrays],axis = 0)
        #avg_probs = torch.mean(probs_array,axis = 0)


        logprobs = torch.log(probs_array)
        avg_logprobs = torch.log(avg_probs)

        combined_probs = 0.5*probs_array+0.5*avg_probs ## broadcast average probabilities on first dim. 
        log_combined_probs = torch.log(combined_probs)

        avg_kl_terms = torch.sum(avg_probs*(avg_logprobs-log_combined_probs),axis = -1)
        single_kl_terms = torch.sum(probs_array*(logprobs-log_combined_probs),axis = -1)
        single_model_jsds = 0.5*(avg_kl_terms+single_kl_terms) ## shape (models,batch)

        avg_jsd = torch.mean(single_model_jsds,axis = 0)
        return avg_jsd

class Model_DKL_Avg(object):    
    """Calculates diversity using the KL divergence from the ensemble prediction to a single model prediction (ensemble as first argument). 

    """
    def __init__(self,cost_format):
        """

        """
        assert cost_format in ["torch","numpy"], "format must be either `torch` or `numpy`"
        self.format = cost_format

    def dkl_avg(self,probs):    
        if self.format == "numpy":
            return self.dkl_avg_numpy(probs)
        elif self.format == "torch":
            return self.dkl_avg_torch(probs)
    
    def dkl_avg_numpy(self,probs):
        probs_array =np.stack(probs,axis = 0)
        avg_probs = np.mean(probs_array,axis = 0)

        logprobs = np.log(probs_array) # shape (models,batch,classes)
        avg_logprobs = np.log(avg_probs) # shape (batch,classes)

        single_model_kl = np.sum(avg_probs*(avg_logprobs-logprobs),axis = -1) ## shape (models,batch)
        avg_kl = np.mean(single_model_kl,axis = 0)
        return avg_kl

    def dkl_avg_torch(self,probs):
        probs_array = torch.stack(probs,axis = 0)
        avg_probs = torch.mean(probs_array,axis = 0)

        logprobs = torch.log(probs_array)
        avg_logprobs = torch.log(avg_probs)

        single_model_kl = torch.sum(avg_probs*(avg_logprobs-logprobs),axis = -1) ## shape (models,batch)
        avg_kl = torch.mean(single_model_kl,axis = 0)
        
        return avg_kl

class Regression_Var(object):
    def __init__(self,cost_format):
        assert cost_format in ["torch","numpy"]
    def var(self,preds):
        if self.format == "numpy":
            return self.var_numpy(preds)
        elif self.format == "torch":
            return self.var_torch(preds)
    def var_numpy(preds):    
        preds_array = np.stack(preds,axis = 0)
        var = np.var(preds_array,axis = 0)
        return var

    def var_torch(preds):
        preds_array = np.stack(preds,axis = 0)
        var = torch.var(preds_array,axis = 0, unbiased = False)
        return var
