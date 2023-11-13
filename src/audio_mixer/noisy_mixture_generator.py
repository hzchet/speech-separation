import os
import random
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from src.audio_mixer.utils import create_mix, create_noisy_mix


logger = logging.getLogger(__name__)


class NoisyMixtureGenerator:
    def __init__(self, speakers_files, out_folder, noise_folder, nfiles=5000, test=False, random_state=42):
        self.speakers_files = speakers_files # list of SpeakerFiles for every speaker_id
        self.nfiles = nfiles
        self.random_state = random_state
        self.out_folder = out_folder
        self.test = test
        self.noise_folder = Path(noise_folder)
        self.noises = os.listdir(noise_folder)
        random.seed(self.random_state)
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

    def generate_triplets(self):
        i = 0
        all_triplets = {"reference": [], "target": [], "noise": [], "wham_noise": [], "target_id": [], "noise_id": []}
        while i < self.nfiles:
            spk1, spk2 = random.sample(self.speakers_files, 2)

            if len(spk1.files) < 2 or len(spk2.files) < 2:
                continue

            wham_noise = random.choice(self.noises)
            target, reference = random.sample(spk1.files, 2)
            noise = random.choice(spk2.files)
            all_triplets["reference"].append(reference)
            all_triplets["target"].append(target)
            all_triplets["noise"].append(noise)
            all_triplets["wham_noise"].append(str(self.noise_folder / wham_noise))
            all_triplets["target_id"].append(spk1.id)
            all_triplets["noise_id"].append(spk2.id)
            i += 1

        return all_triplets

    def generate_mixes(self, snr_levels=[0], num_workers=10, update_steps=10, **kwargs):

        triplets = self.generate_triplets()

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = []

            for i in range(self.nfiles):
                triplet = {"reference": triplets["reference"][i],
                           "target": triplets["target"][i],
                           "noise": triplets["noise"][i],
                           "wham_noise": triplets["wham_noise"][i],
                           "target_id": triplets["target_id"][i],
                           "noise_id": triplets["noise_id"][i]
                        }

                futures.append(pool.submit(create_noisy_mix, i, triplet,
                                           snr_levels, os.path.join(self.out_folder, 'mix'),
                                           os.path.join(self.out_folder, 'refs'),
                                           os.path.join(self.out_folder, 'targets'),
                                           test=self.test, **kwargs))

            for i, future in enumerate(futures):
                future.result()
                if (i + 1) % max(self.nfiles // update_steps, 1) == 0:
                    logger.info(f"Files Processed | {i + 1} out of {self.nfiles}")
