## test dkl metric 
from ensemble_attention.metrics import Model_D_KL
import numpy as np
import torch
from scipy.special import softmax

def random_preds(batch_size,classes):
    """Generate an array of shape (batch, classes) giving softmax probability output for a batch of size `batch_size` in a `classes`-way classification task. 
    """
    scores = np.random.randn(batch_size,classes)
    probs = softmax(scores,axis = 1)
    return probs

class Test_Model_D_KL():
    np_data = [np.log(random_preds(100,10)) for i in range(5)]
    np_labels = np.random.randint(0,10,100)
    torch_data = [torch.tensor(nd) for nd in np_data]
    torch_labels = torch.tensor(np_labels)
    def test_vals(self):
        kl = Model_D_KL("numpy")
        same = [self.np_data[0] for i in range(5)]
        same_torch = [self.torch_data[0] for i in range(5)]
        out_numpy = kl.kl_numpy(same,self.np_labels)
        out_torch = kl.kl_torch(same_torch,self.torch_labels)
        assert np.allclose(out_numpy,0)
        assert np.allclose(out_torch.numpy(),0)

    def test_numpy(self):
        kl = Model_D_KL("numpy")
        out = kl.kl(self.np_data,self.np_labels)
        base = kl.kl_numpy(self.np_data,self.np_labels)
        assert np.all(out == base)
        print(out)
        assert np.all(out > 0)

    def test_torch(self):    
        kl = Model_D_KL("torch")
        out = kl.kl(self.torch_data,self.torch_labels)
        base = kl.kl_torch(self.torch_data,self.torch_labels)
        assert np.all(out.numpy() == base.numpy())

    def test_cross(self):    
        kl = Model_D_KL("torch")
        out_numpy = kl.kl_numpy(self.np_data,self.np_labels)
        out_torch = kl.kl_torch(self.torch_data,self.torch_labels)
        assert np.allclose(out_torch.numpy(),out_numpy)
