from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torch import Tensor

from src.base.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, zero_mean: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_mean = zero_mean
        
    def __call__(self, s1: Tensor, target_wave: Tensor, *args, **kwargs):
        si_sdr = ScaleInvariantSignalDistortionRatio(zero_mean=self.zero_mean).to(s1.device)
        return si_sdr(s1, target_wave)
