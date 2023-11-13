import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.T for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.transpose(1, 2)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    target_wave = [item['target_wave'] for item in dataset_items]
    mixed_wave = [item['mixed_wave'] for item in dataset_items]
    reference_wave = [item['reference_wave'] for item in dataset_items]
    
    target_paths = [item['target_path'] for item in dataset_items]
    mixed_paths = [item['mixed_path'] for item in dataset_items]
    reference_paths = [item['reference_path'] for item in dataset_items]
    
    speaker_id = torch.tensor([item['speaker_id'] for item in dataset_items])
    target_len = torch.tensor([item['target_len'] for item in dataset_items])
    
    reference_wave = pad_sequence(reference_wave)
    target_wave = pad_sequence(target_wave)
    mixed_wave = pad_sequence(mixed_wave)

    return {
        "target_wave": target_wave,
        "target_len": target_len,
        "mixed_wave": mixed_wave,
        "reference_wave": reference_wave,
        "target_path": target_paths,
        "mixed_paths": mixed_paths,
        "reference_paths": reference_paths,
        "speaker_id": speaker_id
    }
