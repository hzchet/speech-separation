from typing import List, Dict

import torch
import pyloudnorm as pyln
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.base.base_model import BaseModel
from src.model.spex_blocks import TCNStackedBlock, ResNetBlock
from src.model.channelwise_layernorm import ChannelwiseLayerNorm


class SpExPlus(BaseModel):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 256,
        speech_kernel_sizes: List[int] = [20, 80, 160],
        extractor_channels: int = 256,
        resnet_channels: List[int] = [256, 256, 512, 512],
        speaker_embed_dim: int = 256,
        pooling_kernel_size: int = 3,
        tcn_stacked_block_num: int = 4,
        tcn_stacked_block_args: Dict = None,
        n_classes: int = None,
        **batch
    ):
        super().__init__(**batch)
        assert n_classes is not None
        assert len(speech_kernel_sizes) == 3, 'provide kernel_size for every time scale encoder'
        
        L1, L2, L3 = speech_kernel_sizes
        
        self.short_speech_encoder = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                                              kernel_size=L1, stride=L1 // 2)
        self.middle_speech_encoder = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                                               kernel_size=L2, stride=L1 // 2)
        self.long_speech_encoder = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                                             kernel_size=L3, stride=L1 // 2)
        
        self.speaker_extractor = nn.ModuleDict({
            "conv_proj": nn.Sequential(
                ChannelwiseLayerNorm(3 * out_channels),
                nn.Conv1d(3 * out_channels, extractor_channels, kernel_size=1)
            ),
            "tcn_blocks": nn.ModuleList([
                TCNStackedBlock(embed_dim=speaker_embed_dim, **tcn_stacked_block_args) for _ in range(tcn_stacked_block_num)
            ]),
            "short_mask": nn.Sequential(
                nn.Conv1d(extractor_channels, out_channels, kernel_size=1),
                nn.ReLU()
            ),
            "middle_mask": nn.Sequential(
                nn.Conv1d(extractor_channels, out_channels, kernel_size=1),
                nn.ReLU()
            ),
            "long_mask": nn.Sequential(
                nn.Conv1d(extractor_channels, out_channels, kernel_size=1),
                nn.ReLU()
            )
        })
        
        speaker_encoder_resnet_blocks = []
        for in_ch, out_ch in zip(resnet_channels[:-1], resnet_channels[1:]):
            speaker_encoder_resnet_blocks.append(
                ResNetBlock(in_ch, out_ch, pooling_kernel_size)
            )

        self.speaker_encoder = nn.ModuleDict({
            "conv_proj": nn.Sequential(
                ChannelwiseLayerNorm(3 * out_channels),
                nn.Conv1d(3 * out_channels, resnet_channels[0], kernel_size=1)
            ),
            "encoder": nn.Sequential(
                *speaker_encoder_resnet_blocks,
                nn.Conv1d(resnet_channels[-1], speaker_embed_dim, kernel_size=1),
            ),
            "classifier": nn.Sequential(
                nn.Linear(speaker_embed_dim, n_classes)
            )
        })

        self.short_speech_decoder = nn.ConvTranspose1d(out_channels, 1, kernel_size=L1, 
                                                       stride=L1 // 2)
        self.middle_speech_decoder = nn.ConvTranspose1d(out_channels, 1, kernel_size=L2,
                                                        stride=L1 // 2)
        self.long_speech_decoder = nn.ConvTranspose1d(out_channels, 1, kernel_size=L3,
                                                      stride=L1 // 2)

    def forward(self, mixed_wave: Tensor, reference_wave: Tensor, **batch):
        assert mixed_wave.dim() == reference_wave.dim() == 3
        
        X1 = F.relu(self.short_speech_encoder(reference_wave))
        X2 = self.middle_speech_encoder(reference_wave)
        X2 = F.relu(F.pad(X2, pad=(0, X1.shape[-1] - X2.shape[-1]), mode='constant', value=0))
        X3 = self.long_speech_encoder(reference_wave)
        X3 = F.relu(F.pad(X3, pad=(0, X1.shape[-1] - X3.shape[-1]), mode='constant', value=0))
        
        X = torch.cat([X1, X2, X3], dim=1)
        X_proj = self.speaker_encoder["conv_proj"](X)
        X_encoded = self.speaker_encoder["encoder"](X_proj)
        speaker_embedding = torch.nn.functional.avg_pool1d(
            X_encoded, 
            kernel_size=X_encoded.shape[-1]
        )
        
        Y1 = F.relu(self.short_speech_encoder(mixed_wave))
        Y2 = self.middle_speech_encoder(mixed_wave)
        Y2 = F.relu(F.pad(Y2, pad=(0, Y1.shape[-1] - Y2.shape[-1]), mode='constant', value=0))
        Y3 = self.long_speech_encoder(mixed_wave)
        Y3 = F.relu(F.pad(Y3, pad=(0, Y1.shape[-1] - Y3.shape[-1]), mode='constant', value=0))
        
        Y = torch.cat([Y1, Y2, Y3], dim=1)
        Y_proj = self.speaker_extractor["conv_proj"](Y)
        tcn_input = Y_proj
        for tcn_block in self.speaker_extractor["tcn_blocks"]:
            tcn_input = tcn_block(tcn_input, speaker_embedding)
        
        mask_1 = self.speaker_extractor["short_mask"](tcn_input)
        mask_2 = self.speaker_extractor["middle_mask"](tcn_input)
        mask_3 = self.speaker_extractor["long_mask"](tcn_input)
        
        Y1_masked = Y1 * mask_1
        Y2_masked = Y2 * mask_2
        Y3_masked = Y3 * mask_3
        
        s1 = self.short_speech_decoder(Y1_masked)
        s2 = self.middle_speech_decoder(Y2_masked)
        s3 = self.long_speech_decoder(Y3_masked)

        return {
            "s1": F.pad(s1, pad=(0, mixed_wave.shape[-1] - s1.shape[-1]), mode='constant', 
                        value=0.0),
            "s2": F.pad(s2, pad=(0, mixed_wave.shape[-1] - s2.shape[-1]), mode='constant', 
                        value=0.0),
            "s3": F.pad(s3, pad=(0, mixed_wave.shape[-1] - s3.shape[-1]), mode='constant',
                          value=0.0),
            "clf_logits": self.speaker_encoder["classifier"](speaker_embedding.squeeze(-1))
        }
