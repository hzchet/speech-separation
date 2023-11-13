import torch.nn as nn


class ChannelwiseLayerNorm(nn.Module):
    def __init__(self, channel_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channel_dim)
        
    def forward(self, x):
        assert x.dim() == 3
        
        return self.layer_norm(x.transpose(-1, -2)).transpose(-1, -2)
