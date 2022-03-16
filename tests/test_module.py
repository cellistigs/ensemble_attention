## test module 
import pytest
import numpy as np
import torch
from hydra import compose, initialize
import ensemble_attention
from ensemble_attention.module import CIFAR10EnsembleModule,CIFAR10AttentionEnsembleModule



class Test_CIFAR10EnsembleModule():
    """Run the following tests: 

    1. are individual submodels different in initialization from each other? 
    2. are the learning rates set properly? 
    """
    with initialize(config_path="../configs/", job_name="test_app"):
        cfg = compose(config_name="run_default_cpu", overrides=["nb_models=3"])
    
    def test_init(self):
        ens = CIFAR10EnsembleModule(self.cfg)
        assert len(ens.models) == self.cfg.nb_models
        print(ens.models[0].layer1.__dict__)
        init_weight_matrices = [m.layer1[0].conv1.weight.detach().numpy() for m in ens.models]
        for i in range(len(init_weight_matrices)):
            for j in range(i):
                assert not np.all(init_weight_matrices[i] == init_weight_matrices[j])
    def test_opt(self):
        ens = CIFAR10EnsembleModule(self.cfg)
        opt,sched = ens.configure_optimizers()
        print(opt.__dict__)
        assert 0, "still need a good way to test this."

class Test_CIFAR10AttentionEnsembleModule():
    """Run the following tests: 

    1. are individual submodels different in initialization from each other? 
    2. if we use the right query and key, is the output identical to the expected single model output? 
    3. how can we save weight outputs? 
    """
    with initialize(config_path="../configs/", job_name="test_app"):
        cfg = compose(config_name="run_default_cpu", overrides=["nb_models=3","+embedding_dim=10"])
    
    def test_init(self):
        ens = CIFAR10AttentionEnsembleModule(self.cfg)
        assert len(ens.models) == self.cfg.nb_models
        print(ens.models[0].layer1.__dict__)
        init_weight_matrices = [m.layer1[0].conv1.weight.detach().numpy() for m in ens.models]
        for i in range(len(init_weight_matrices)):
            for j in range(i):
                assert not np.all(init_weight_matrices[i] == init_weight_matrices[j])
    
    def test_degen(self,monkeypatch):
        """If we set the attention matrix correctly, we expect the output to be equal to the output of the first model only. 
        """
        # set inputs
        dummy_input = torch.tensor(np.random.randn(5,3,32,32)).double()
        dummy_output  = torch.tensor(np.array([1,2,1,1,1])).long()
        dummy_batch = (dummy_input,dummy_output)

        # set fixed attention matrix that just selects outputs from first model.  
        def fixed_attnmatrix(a,b):
            select_matrix = np.zeros((self.cfg.nb_models,self.cfg.nb_models))
            select_matrix[0,0] = 1
            select_matrix = torch.tensor(select_matrix).double()
            return select_matrix
        def get_fixed_attnmatrix(s,a,b):
            return fixed_attnmatrix
        monkeypatch.setattr(ensemble_attention.module.CIFAR10AttentionEnsembleModule,"get_attnlayer",get_fixed_attnmatrix)
        ens = CIFAR10AttentionEnsembleModule(self.cfg).double()

        # test 
        ens.eval()
        with torch.no_grad():
            y = ens(dummy_batch)
            y0 = ens.models[0](dummy_batch[0])

            assert torch.all(y[0]==y0)
            assert torch.all(y[1] == fixed_attnmatrix(0,0))


