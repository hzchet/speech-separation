import torch
import torch.nn as nn
from torch import Tensor


class SISDRLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, preds: Tensor, target: Tensor, **batch):
        dot_prod = preds @ target.transpose(-1, -2)

        signal = dot_prod * target / torch.norm(target, dim=-1).unsqueeze(1) ** 2
        distortion = signal - preds
        
        return -20 * (
            torch.log10(torch.norm(signal, dim=-1)) - torch.log10(torch.norm(distortion, dim=-1))
        ).mean()
