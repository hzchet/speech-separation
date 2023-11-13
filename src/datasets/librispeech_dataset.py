import json
import os

from sklearn.model_selection import train_test_split

from src.utils import ROOT_PATH
from src.base.base_dataset import BaseDataset


class LibriSpeechDataset(BaseDataset):
    def __init__(self, split: str, data_dir: str = None, *args, **kwargs):
        assert split in ('train', 'test_snr0', 'valid'), split
        
        if data_dir is None:
            data_dir = ROOT_PATH / "data"
            
        self._data_dir = data_dir
        
        index = self._get_or_create_index(split)
        super().__init__(index, *args, **kwargs)
        
    def _get_or_create_index(self, split: str):
        index_path = self._data_dir / f"{split}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(split)
        
        class_info_path = self._data_dir / "classes_info.json"
        with class_info_path.open() as f:
            self.class_mapping = json.load(f)
        
        return index
    
    def _create_train_valid_index(self, split: str):
        mixes_dir = self._data_dir / f'mixes_train_val'
        
        audio_paths = os.listdir(mixes_dir)
        speaker_ids = [int(path.split('_')[0]) for path in audio_paths]
        train_audios, valid_audios, train_speakers, valid_speakers = train_test_split(
            audio_paths,
            speaker_ids,
            stratify=speaker_ids,
            test_size=0.3,
            random_state=42
        )
        train_index = []
        valid_index = []
        speaker_mapping = dict()
        current_speaker_id = 0
        for file in train_audios:
            if file.endswith('-ref.wav'):
                audio_id = file.replace('-ref.wav', '')
                target_id = int(file.split('_')[0])
                if target_id not in speaker_mapping:
                    speaker_mapping[target_id] = current_speaker_id
                    current_speaker_id += 1
                train_index.append({
                    'reference_path': os.path.join(mixes_dir, file),
                    'target_path': os.path.join(mixes_dir, f'{audio_id}-target.wav'),
                    'mixed_path': os.path.join(mixes_dir, f'{audio_id}-mixed.wav'),
                    'target_id': speaker_mapping[target_id]
                })
        for file in valid_audios:
            if file.endswith('-ref.wav'):
                audio_id = file.replace('-ref.wav', '')
                target_id = int(file.split('_')[0])
                assert target_id in speaker_mapping
                valid_index.append({
                    'reference_path': os.path.join(mixes_dir, file),
                    'target_path': os.path.join(mixes_dir, f'{audio_id}-target.wav'),
                    'mixed_path': os.path.join(mixes_dir, f'{audio_id}-mixed.wav'),
                    'target_id': speaker_mapping[target_id]
                })

        classes_info_path = self._data_dir / "classes_info.json" 
        train_index_path = self._data_dir / "train_index.json"
        valid_index_path = self._data_dir / "valid_index.json"
        with train_index_path.open("w") as f:
            json.dump(train_index, f, indent=2)
        with valid_index_path.open("w") as f:
            json.dump(valid_index, f, indent=2)
        with classes_info_path.open("w") as f:
            json.dump(speaker_mapping, f, indent=2)
        
        if split == 'train':
            return train_index
        elif split == 'valid':
            return valid_index
    
    def _create_index(self, split: str):
        index = []
        if split in ('train', 'valid'):
            return self._create_train_valid_index(split)
        
        split_dir = self._data_dir / f'mixes_{split}'
        assert split_dir.exists(), f'{split_dir} does not exist! run `python3 create_dataset.py`'

        files = os.listdir(split_dir)
        
        for file in files:
            if file.endswith('-ref.wav'):
                audio_id = file.replace('-ref.wav', '')
                index.append({
                    'reference_path': os.path.join(split_dir, file),
                    'target_path': os.path.join(split_dir, f'{audio_id}-target.wav'),
                    'mixed_path': os.path.join(split_dir, f'{audio_id}-mixed.wav'),
                    'target_id': -1
                })
        index_path = self._data_dir / f"{split}_index.json"
        with index_path.open("w") as f:
            json.dump(index, f, indent=2)

        return index
