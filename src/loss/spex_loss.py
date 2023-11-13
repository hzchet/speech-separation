import torch.nn as nn

from src.loss.si_sdr_loss import SISDRLoss


class SpExLoss(nn.Module):
    def __init__(self, alpha: float = 0.1, beta: float = 0.1, gamma: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.si_sdr_loss = SISDRLoss()
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, s1, s2, s3, clf_logits, target_wave, speaker_id, **batch):
        if self.training:
            return (1 - self.alpha - self.beta) * self.si_sdr_loss(s1, target_wave) + \
                self.alpha * self.si_sdr_loss(s2, target_wave) + \
                self.beta * self.si_sdr_loss(s3, target_wave) + \
                self.gamma * self.ce_loss(clf_logits, speaker_id)
        else:
            return (1 - self.alpha - self.beta) * self.si_sdr_loss(s1, target_wave) + \
                self.alpha * self.si_sdr_loss(s2, target_wave) + \
                self.beta * self.si_sdr_loss(s3, target_wave)
