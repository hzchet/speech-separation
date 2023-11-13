import os
import argparse
import shutil
import warnings

from speechbrain.utils.data_utils import download_file

import src.audio_mixer as module_mixer
from src.audio_mixer.utils import LibriSpeechSpeakerFiles
from src.utils.parse_config import ConfigParser
from src.utils import ROOT_PATH

warnings.filterwarnings("ignore")

URL_LINKS_ = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


def install_librispeech(config):
    logger = config.get_logger()
    part = config["part"]
    assert part in URL_LINKS_
    data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
    data_dir.mkdir(exist_ok=True, parents=True)
    
    arch_path = data_dir / f"{part}.tar.gz"
    logger.info(f"Loading part {part}")
    download_file(URL_LINKS_[part], arch_path)
    shutil.unpack_archive(arch_path, data_dir)
    for fpath in (data_dir / "LibriSpeech").iterdir():
        shutil.move(str(fpath), str(data_dir / fpath.name))
    os.remove(str(arch_path))
    shutil.rmtree(str(data_dir / "LibriSpeech"))


def main(config: ConfigParser):
    audios_dir = f'{str(ROOT_PATH)}/data/datasets/librispeech/{config["part"]}'
    if not os.path.exists(audios_dir):
        install_librispeech(config)

    speaker_files = [
        LibriSpeechSpeakerFiles(speaker_id.name, audios_dir, config["audio_template"]) \
            for speaker_id in os.scandir(audios_dir)
    ]
    mixture_generator = config.init_obj(config["mixture_generator"], module_mixer, 
                                        speakers_files=speaker_files)
    mixture_generator.generate_mixes(
        **config["mixture_generator"]["mix_args"]
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    
    config = ConfigParser.from_args(args)
    main(config)
