## Test gradients you get from dkl against those with a single model. 
from ensemble_attention.module import CIFAR10EnsembleDKLModule
from ensemble_attention.metrics import Model_D_KL
import numpy as np
from hydra import initialize, compose
import torch

softmax = torch.nn.Softmax(dim = 1)

def random_preds(batch_size,classes):
    """given a bunch of scores, compare the gradients we get to these scores from 
    """
    scores = np.random.randn(batch_size,classes)
    return scores

class Test_Model_D_KL():
    np_scores = []
    for i in range(5):
        scores = random_preds(100,10)
        np_scores.append(scores)
    np_labels = np.random.randint(0,10,100)
    torch_scores = [torch.tensor(nd,requires_grad=True) for nd in np_scores]
    torch_softmaxes = [softmax(torch_score) for torch_score in torch_scores]
    torch_labels = torch.tensor(np_labels)

    ## for dkl
    traincriterion = torch.nn.NLLLoss()
    kl = Model_D_KL("torch")

    ## for ensemble
    criterion = torch.nn.CrossEntropyLoss()
    def test_grads(self):    
        """Using autograd, computes gradients with: the dkl loss, the ensemble loss, and the single model loss on the same set of softmax vectors. we expect that the single model loss should be a factor of M greater than the other two, where M is the number of ensemble members. 

        """
        model = 1
        ## use the dkl model:

        logoutput = torch.log(torch.mean(torch.stack(self.torch_softmaxes),dim = 0)) ## get the 
        mloss = self.traincriterion(logoutput, self.torch_labels)
        dklloss = torch.mean(self.kl.kl(self.torch_softmaxes,self.torch_labels))
        eloss = mloss + dklloss ## with gamma equal to 1, this is the same as the standard ensemble training loss (independent). 

        eloss.backward()
        grad_first_input_ens = self.torch_scores[model].grad[0,:]
        print(grad_first_input_ens)
        self.torch_scores[model].grad.data.zero_()

        ## now use the ensemble model. 

        losses = []
        for i in range(len(self.torch_scores)):
            mloss = self.criterion(self.torch_scores[i], self.torch_labels)
            losses.append(mloss)
        iloss = sum(losses)/len(self.torch_scores) ## calculate the sum with pure python functions.    
        iloss.backward()
        grad_first_input_ind = self.torch_scores[model].grad[0,:]
        print(grad_first_input_ind)
        self.torch_scores[model].grad.data.zero_()

        loss = self.criterion(self.torch_scores[model],self.torch_labels)
        loss.backward()
        grad_first_input_single = self.torch_scores[model].grad[0,:]
        print(grad_first_input_single)
        assert 0

    def test_full_models(self):
        """Now use an optimizer against a given model that has both the ensemble loss and the dkl loss, and see if there's any difference in gradient. . 

        """
