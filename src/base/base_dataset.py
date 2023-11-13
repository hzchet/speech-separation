import logging
import random

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            wave_augs=None,
            limit=None
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs

        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, limit)
        self._index = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        
        target_wave = self.load_audio(data_dict["target_path"])
        target_wave = self.process_wave(target_wave)
        
        reference_wave = self.load_audio(data_dict["reference_path"])
        reference_wave = self.process_wave(reference_wave)
        
        mixed_wave = self.load_audio(data_dict["mixed_path"])
        mixed_wave = self.process_wave(mixed_wave)
        
        return {
            "target_wave": target_wave,
            "target_len": target_wave.shape[-1],
            "target_path": data_dict["target_path"],
            "reference_wave": reference_wave,
            "reference_path": data_dict["reference_path"],
            "mixed_wave": mixed_wave,
            "mixed_path": data_dict["mixed_path"],
            "speaker_id": int(data_dict["target_id"])
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)

            return audio_tensor_wave

    @staticmethod
    def _filter_records_from_dataset(
            index: list, limit
    ) -> list:
        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "target_path" in entry, (
                "Each dataset item should include field 'target_path' - path to target audio."
            )
            assert "reference_path" in entry, (
                "Each dataset item should include field 'reference_path' - path to reference audio."
            )
            assert "mixed_path" in entry, (
                "Each dataset item should include field 'mixed_path' - path to mixed audio."
            )
