import torch
import torch.nn as nn

from src.model.global_layernorm import GlobalLayerNorm


class ResNetBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int = 256,
        out_channels: int = 256,
        pooling_kernel_size: int = 3
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.pooling = nn.Sequential(
            nn.PReLU(),
            nn.MaxPool1d(pooling_kernel_size)   
        )
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.downsample = nn.Identity()
        
    def forward(self, x):
        return self.pooling(self.conv_layers(x) + self.downsample(x))


class TCNConcatBLock(nn.Module):
    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 256,
        depthwise_channels: int = 512,
        depthwise_kernel_size: int = 3,
        dilation: int = 1
    ):
        super().__init__()
        # padding to not shrink the time dimension
        depthwise_padding = dilation * (depthwise_kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, depthwise_channels, kernel_size=1),
            nn.PReLU(),
            GlobalLayerNorm(depthwise_channels),
            nn.Conv1d(depthwise_channels, depthwise_channels, kernel_size=depthwise_kernel_size, 
                      dilation=dilation, padding=depthwise_padding, groups=depthwise_channels),
            nn.PReLU(),
            GlobalLayerNorm(depthwise_channels),
            nn.Conv1d(depthwise_channels, out_channels, kernel_size=1)
        )
    
    def forward(self, x, condition):
        return x + self.net(torch.cat([x, condition.repeat(1, 1, x.shape[-1])], dim=1))


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        depthwise_channels: int = 512,
        depthwise_kernel_size: int = 3,
        dilation: int = 1
    ):
        super().__init__()
        # padding to not shrink the time dimension
        depthwise_padding = dilation * (depthwise_kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, depthwise_channels, kernel_size=1),
            nn.PReLU(),
            GlobalLayerNorm(depthwise_channels),
            nn.Conv1d(depthwise_channels, depthwise_channels, kernel_size=depthwise_kernel_size, 
                      dilation=dilation, padding=depthwise_padding, groups=depthwise_channels),
            nn.PReLU(),
            GlobalLayerNorm(depthwise_channels),
            nn.Conv1d(depthwise_channels, in_channels, kernel_size=1)
        )
    
    def forward(self, x):
        return x + self.net(x)


class TCNStackedBlock(nn.Module):
    def __init__(
        self, 
        num_blocks: int = 8,
        in_channels: int = 256,
        embed_dim: int = 256,
        *args,
        **kwargs
    ):
        super().__init__()
        tcn_blocks = []
        dilation = 1
        self.concat_block = TCNConcatBLock(in_channels + embed_dim, in_channels, *args, **kwargs)
        dilation *= 2
        for i in range(num_blocks - 1):
            tcn_blocks.append(TCNBlock(in_channels, *args, **kwargs, dilation=dilation))
            dilation *= 2
        
        self.net = nn.Sequential(*tcn_blocks)
    
    def forward(self, x, condition):
        x = self.concat_block(x, condition)
        return self.net(x)
