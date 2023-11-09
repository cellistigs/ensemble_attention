"""Much of this code is used to implement models from https://arxiv.org/pdf/2106.11959.pdf, with implementation details taken from https://github.com/yandex-research/tabular-dl-revisiting-models/blob/main/bin/mlp.py 
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import os 

class MLP(nn.Module):
    """MLP that takes both categorical and numerical data as input. We separate out the number of numerical inputs from the categorical, which we keep track of separately. 
    We assume numerical data is normalized, and we embed all categorical features in the same space. 
    :param d_in: number of numerical input dimensions
    :param d_layers: size each hidden layer in the network. provided as a list of ints.  
    :param d_out: output dimensionality of the mlp. 
    :param categories: number of categories in each categorical feature (list of ints)
    :param d_embedding: dimension of categorical data embedding. 

    """
    def __init__(
            self,
            d_in,
            d_layers,
            d_out,
            categories,
            d_embedding
            ):
        super().__init__()
        if categories is not None: 
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0) # offsets so we can project the categorical data into the right spaces. 
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a = math.sqrt(5))

        self.layers = nn.ModuleList(
                [
                    nn.Linear(d_layers[i-1] if i else d_in, x)
                    for i, x in enumerate(d_layers)
                ]
            )
        self.dropout = 0 
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x_num, x_cat):
        x = []
        if x_num is not None:
            x.append(x_num.float())
        if x_cat is not None:
            x.append(
                    self.category_embeddings(x_cat + self.category_offsets[None]).view(x_cat.size(0),-1) # add offsets to categorical data, then embed this into higher dimensional space. 
                    )
        x = torch.cat(x,dim=-1)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        x = x.squeeze(-1)
        return x

def MLP_Adult(d_layers,d_embedding):
    return MLP(6,d_layers,2,[7,16,7,14,6,5,2,41],d_embedding)



