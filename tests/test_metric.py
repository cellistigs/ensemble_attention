## test dkl metric 
from ensemble_attention.metrics import Model_D_KL,Model_Ortega_Variance,Model_JS_Unif,Model_JS_Avg,Model_DKL_Avg
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

class Test_Model_Ortega_Variance():       
    M = 5
    np_data = [np.log(random_preds(100,10)) for i in range(5)]
    np_labels = np.random.randint(0,10,100)
    torch_data = [torch.tensor(nd) for nd in np_data]
    torch_labels = torch.tensor(np_labels)
    def test_vals(self):
        var = Model_Ortega_Variance("numpy")
        ## setup 
        same =  self.np_data#[self.np_data[0] for i in range(self.M)]
        out_numpy = var.var_numpy(same,self.np_labels)
        ## check code: 
        same_probs = np.stack(np.exp(same),axis = 0)
        same_array = same_probs[:,np.arange(same_probs.shape[1]),self.np_labels]

        max_probs = np.max(same_array,axis = 0)
        mean_probs = np.mean(same_array,axis = 0)
        hmax1 = (np.log(mean_probs)-np.log(max_probs))/(max_probs-mean_probs)**2 
        hmax2 = 1/(mean_probs*(max_probs-mean_probs))
        hmax = hmax1+hmax2

        var_sq_terms = np.mean(same_array**2/(max_probs**2),axis = 0)
        var = var_sq_terms
        for j in range(self.M):
            var_cross_term = np.mean(same_array*same_array[j,:]/(max_probs**2),axis = 0)/self.M
            var -= var_cross_term

        full_var = var#*hmax
        assert np.allclose(full_var,out_numpy)
        assert np.all(full_var>0)
        assert np.all(out_numpy>0)

    def test_torch(self):    
        var = Model_Ortega_Variance("numpy")
        ## setup 
        numpy =  self.np_data#[self.np_data[0] for i in range(self.M)]
        out_numpy = var.var_numpy(self.np_data,self.np_labels)
        out_torch = var.var_torch(self.torch_data,self.torch_labels)
        assert np.allclose(out_torch.numpy(),out_numpy)
        assert np.all(out_torch.numpy()>0)

class Test_Model_JS_Unif():       
    M = 5
    np_data = [random_preds(100,10) for i in range(5)]
    np_labels = np.random.randint(0,10,100)
    torch_data = [torch.tensor(nd) for nd in np_data]
    torch_labels = torch.tensor(np_labels)
    def test_vals(self):
        js = Model_JS_Unif("numpy")
        ## setup 
        same =  self.np_data#[self.np_data[0] for i in range(self.M)]
        out_numpy = js.js_unif_numpy(same)
        assert np.all(out_numpy>0)
        out_torch = js.js_unif_torch(self.torch_data)
        assert np.all(out_torch.numpy()>0)
        assert np.allclose(out_torch.numpy()-out_numpy,0)

        same =  [self.np_data[0] for i in range(self.M)]
        out_numpy = js.js_unif_numpy(same)
        assert np.allclose(out_numpy,0)


class Test_Model_JS_Avg():       
    M = 5
    np_data = [random_preds(100,10) for i in range(5)]
    np_labels = np.random.randint(0,10,100)
    torch_data = [torch.tensor(nd) for nd in np_data]
    torch_labels = torch.tensor(np_labels)
    def test_vals(self):
        js = Model_JS_Avg("numpy")
        ## setup 
        same =  self.np_data#[self.np_data[0] for i in range(self.M)]
        out_numpy = js.js_avg_numpy(same)
        assert np.all(out_numpy>0)

        out_torch = js.js_avg_torch(self.torch_data)
        assert np.all(out_torch.numpy()>0)
        assert np.allclose(out_torch.numpy()-out_numpy,0)

        same =  [self.np_data[0] for i in range(self.M)]
        out_numpy = js.js_avg_numpy(same)
        assert np.allclose(out_numpy,0)

class Test_Model_DKL_Avg():       
    M = 5
    np_data = [random_preds(100,10) for i in range(5)]
    np_labels = np.random.randint(0,10,100)
    torch_data = [torch.tensor(nd) for nd in np_data]
    torch_labels = torch.tensor(np_labels)
    def test_vals(self):
        dkl= Model_DKL_Avg("numpy")
        ## setup 
        same =  self.np_data#[self.np_data[0] for i in range(self.M)]
        out_numpy = dkl.dkl_avg_numpy(same)
        assert np.all(out_numpy>0)

        out_torch = dkl.dkl_avg_torch(self.torch_data)
        assert np.all(out_torch.numpy()>0)
        assert np.allclose(out_torch.numpy()-out_numpy,0)

        same =  [self.np_data[0] for i in range(self.M)]
        out_numpy = dkl.dkl_avg_numpy(same)
        assert np.allclose(out_numpy,0)

