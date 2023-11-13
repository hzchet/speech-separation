from typing import List, Union

import torch.nn as nn
import torch


class GlobalLayerNorm(nn.Module):
    def __init__(
        self, 
        normalized_shape: Union[int, List[int]], 
        eps=1e-05
    ):
        super().__init__()
        
        self.eps = eps
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_dim = normalized_shape
    
        self.beta = nn.Parameter(torch.zeros(*normalized_shape, 1))
        self.gamma = nn.Parameter(torch.ones(*normalized_shape, 1))


    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} requires a 3D tensor input".format(self.__name__))
        
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
