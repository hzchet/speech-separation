from torch import Tensor

from src.base.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, clf_logits: Tensor, speaker_id: Tensor, **batch):
        preds = clf_logits.argmax(dim=1)
        return (preds == speaker_id).sum() / len(speaker_id)
