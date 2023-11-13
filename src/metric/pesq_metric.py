from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torch import Tensor
import torch
import pyloudnorm as pyln

from src.base.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    def __init__(self, fs: int = 16000, mode: str = 'wb', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meter = pyln.Meter(fs)
        self.fs = fs
        self.mode = mode
    
    def __call__(self, s1: Tensor, target_wave: Tensor, *args, **kwargs):
        if s1.dim() == 2:
            s1 = s1.unsqueeze(1)
            target_wave = target_wave.unsqueeze(1)
        audios = []
        for audio in s1:
            audio = audio.view(-1, 1).detach().cpu().numpy()
            loudness = self.meter.integrated_loudness(audio)
            audio = pyln.normalize.loudness(audio, loudness, -20)
            audios.append(torch.from_numpy(audio).view(1, -1))
        
        # try:
        audios = torch.stack(audios, dim=0).to(target_wave.device)
        pesq = PerceptualEvaluationSpeechQuality(self.fs, self.mode).to(target_wave.device)
        return pesq(audios, target_wave)
        # except:
        #     return torch.tensor(-0.5)
