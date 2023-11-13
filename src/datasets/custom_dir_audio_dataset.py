import json
from pathlib import Path

from src.utils import ROOT_PATH
from src.base.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, path_to_mixes: str, path_to_refs: str, path_to_targets: str, index_name: str = 'final_test.json', *args, **kwargs):
        index = self._create_or_load_index(index_name, path_to_mixes, path_to_refs, path_to_targets)
        super().__init__(index, *args, **kwargs)
        
    def _create_or_load_index(self, index_name: str, path_to_mixes, path_to_refs, path_to_targets):
        index_path = ROOT_PATH / 'data' / index_name
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
            
            return index
        
        index = []
        for path in Path(path_to_mixes).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                audio_id = path.stem.split('-')[0]
                entry["mixed_path"] = str(path)
                if path_to_targets and Path(path_to_targets).exists():
                    target_path = Path(path_to_targets) / (audio_id + '-target.wav')
                    if target_path.exists():
                        entry["target_path"] = str(target_path)
                if path_to_refs and Path(path_to_refs).exists():
                    reference_path = Path(path_to_refs) / (audio_id + '-ref.wav')
                    if reference_path.exists():
                        entry["reference_path"] = str(reference_path)
                entry['target_id'] = -1

            if len(entry) > 0:
                index.append(entry)

        with index_path.open('w') as f:
            json.dump(index, f, indent=2)
        
        return index
