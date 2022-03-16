## Test the Attetnion Comparison head. 
import torch
from scipy.special import softmax
from ensemble_attention import layers
import numpy as np


class Test_AttnComparison():
    """We want to test that this attention head works as intended. 

    1. test to make sure that weight matrices are initialized correctly, and parameters are as expected (scale parameter is consistent, for example)
    2. test to make sure that with known inputs and projection matrices, the outputs are as intended. 
    """
    ident_matrix = np.eye(3)
    ident_matrix = torch.nn.Parameter(torch.tensor(ident_matrix)).double()
    angle = 36 
    rot_matrix = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    rot_matrix = torch.nn.Parameter(torch.tensor(rot_matrix)).double()
    #cls.orth_matrix = torch.nn.Parameter(torch.tensor(ident_matrix))
    def test_init(self):
        in_dim = 3 
        out_dim = 100
        scale = np.sqrt(out_dim) ## sqrt of 100
        layer = layers.AttnComparison(in_dim,out_dim)
        assert layer.linear_q.weight.shape == (100,3)
        assert layer.linear_k.weight.shape == (100,3)
        assert layer.scale == 1/scale

    def test_forward_ident(self):    
        """Now we want to ensure that compute works as intended. Given a fixed linear projection matrix and fixed inputs, we expect outputs to be fixed. 
        """
        ## initialize test input:
        np_queries = np.array([[0,0,1],[0,1,0],[1,0,0]]).reshape(1,3,3)
        np_keys = np.array([[1,0,0],[0,1,0],[1,0,0],[0,0,1]]).reshape(1,4,3)

        test_queries = torch.tensor(np_queries).double()
        test_keys = torch.tensor(np_keys).double()

        ## initialize weights: 
        layer = layers.AttnComparison(3,3)
        layer.linear_q.weight = self.ident_matrix
        layer.linear_k.weight = self.ident_matrix

        overlap = layer(test_queries,test_keys)

        assert np.allclose(overlap.detach().numpy(),softmax(np.array([[0,0,0,1],[0,1,0,0],[1,0,1,0]])/np.sqrt(3),axis = 1))

    def test_forward_rot(self):    
        ## initialize test input:
        np_queries = np.array([[0,0,1],[0,1,0],[1,0,0]]).reshape(1,3,3)
        np_keys = np.array([[1,0,0],[0,1,0],[1,0,0],[0,0,1]]).reshape(1,4,3)

        test_queries = torch.tensor(np_queries).double()
        test_keys = torch.tensor(np_keys).double()

        ## initialize weights: 
        layer = layers.AttnComparison(3,3)
        layer.linear_q.weight = self.rot_matrix ## multiplication by an orthonormal matrix should not change the attention weights. 
        layer.linear_k.weight = self.rot_matrix

        overlap = layer(test_queries,test_keys)

        assert np.allclose(overlap.detach().numpy(),softmax(np.array([[0,0,0,1],[0,1,0,0],[1,0,1,0]])/np.sqrt(3),axis = 1))


