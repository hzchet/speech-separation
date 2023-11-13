import os
import glob
from glob import glob

import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln


class LibriSpeechSpeakerFiles:
    def __init__(self, speaker_id, audios_dir, audio_template="*.flac"):
        self.id = speaker_id
        self.files = []
        self.audio_template = audio_template
        self.files = self.find_files_by_worker(audios_dir)

    def find_files_by_worker(self, audios_dir):
        speaker_dir = os.path.join(audios_dir, self.id) #it is a string
        chapter_dirs = os.scandir(speaker_dir)
        files=[]
        for chapter_dir in chapter_dirs:
            files = files + [file for file in glob(os.path.join(speaker_dir,chapter_dir.name)+"/"+self.audio_template)]
        return files


def snr_mixer(clean, noise, snr):
    amp_noise = np.linalg.norm(clean) / 10**(snr / 20)

    noise_norm = (noise / np.linalg.norm(noise)) * amp_noise

    mix = clean + noise_norm

    return mix


def vad_merge(w, top_db):
    intervals = librosa.effects.split(w, top_db=top_db)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def cut_audios(s1, s2, sec, sr):
    cut_len = sr * sec
    len1 = len(s1)
    len2 = len(s2)

    s1_cut = []
    s2_cut = []

    segment = 0
    while (segment + 1) * cut_len < len1 and (segment + 1) * cut_len < len2:
        s1_cut.append(s1[segment * cut_len:(segment + 1) * cut_len])
        s2_cut.append(s2[segment * cut_len:(segment + 1) * cut_len])

        segment += 1

    return s1_cut, s2_cut


def fix_length(s1, s2, min_or_max='max'):
    # Fix length
    if min_or_max == 'min':
        utt_len = np.minimum(len(s1), len(s2))
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
    else:  # max
        utt_len = np.maximum(len(s1), len(s2))
        s1 = np.append(s1, np.zeros(utt_len - len(s1)))
        s2 = np.append(s2, np.zeros(utt_len - len(s2)))
    return s1, s2


def create_noisy_mix(idx, triplet, snr_levels, mix_dir, ref_dir, target_dir, trim_db, vad_db, audio_len=3, sr=16000, test=True):
    s1_path = triplet["target"]
    s2_path = triplet["noise"]
    noise_path = triplet["wham_noise"]
    ref_path = triplet["reference"]
    target_id = triplet["target_id"]
    noise_id = triplet["noise_id"]

    s1, _ = sf.read(os.path.join('', s1_path))
    s2, _ = sf.read(os.path.join('', s2_path))
    ref, _ = sf.read(os.path.join('', ref_path))
    noise, _ = sf.read(os.path.join('', noise_path))
    
    meter = pyln.Meter(sr) # create BS.1770 meter

    louds1 = meter.integrated_loudness(s1)
    louds2 = meter.integrated_loudness(s2)
    louds_noise = meter.integrated_loudness(noise)
    louds_ref = meter.integrated_loudness(ref)

    s1_norm = pyln.normalize.loudness(s1, louds1, -29)
    s2_norm = pyln.normalize.loudness(s2, louds2, -29)
    ref_norm = pyln.normalize.loudness(ref, louds_ref, -23.0)
    noise_norm = pyln.normalize.loudness(noise, louds_noise, -29)
    
    amp_s1 = np.max(np.abs(s1_norm))
    amp_s2 = np.max(np.abs(s2_norm))
    amp_ref = np.max(np.abs(ref_norm))
    amp_noise = np.max(np.abs(noise_norm))

    if amp_s1 == 0 or amp_s2 == 0 or amp_ref == 0 or amp_noise == 0:
        return

    if trim_db:
        ref, _ = librosa.effects.trim(ref_norm, top_db=trim_db)
        s1, _ = librosa.effects.trim(s1_norm, top_db=trim_db)
        s2, _ = librosa.effects.trim(s2_norm, top_db=trim_db)
        noise, _ = librosa.effects.trim(noise_norm, top_db=trim_db)

    if len(ref) < sr:
        return

    path_mix = os.path.join(mix_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-mixed.wav")
    path_target = os.path.join(target_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-target.wav")
    path_ref = os.path.join(ref_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-ref.wav")
    
    snr = np.random.choice(snr_levels, 1).item()

    s1, s2 = fix_length(s1, s2, 'max')
    mix = snr_mixer(s1, s2, snr)
    louds1 = meter.integrated_loudness(s1)
    s1 = pyln.normalize.loudness(s1, louds1, -20.0)

    loud_mix = meter.integrated_loudness(mix)
    mix = pyln.normalize.loudness(mix, loud_mix, -20.0)

    mix, noise = fix_length(mix, noise, 'max')
    mix = snr_mixer(mix, noise, snr)
    
    loud_mix = meter.integrated_loudness(mix)
    mix = pyln.normalize.loudness(mix, loud_mix, -20.0)
    
    sf.write(path_mix, mix, sr)
    sf.write(path_target, s1, sr)
    sf.write(path_ref, ref, sr)


def create_mix(idx, triplet, snr_levels, out_dir, trim_db, vad_db, audio_len=3, test=False, sr=16000):
    s1_path = triplet["target"]
    s2_path = triplet["noise"]
    ref_path = triplet["reference"]
    target_id = triplet["target_id"]
    noise_id = triplet["noise_id"]

    s1, _ = sf.read(os.path.join('', s1_path))
    s2, _ = sf.read(os.path.join('', s2_path))
    ref, _ = sf.read(os.path.join('', ref_path))

    meter = pyln.Meter(sr) # create BS.1770 meter

    louds1 = meter.integrated_loudness(s1)
    louds2 = meter.integrated_loudness(s2)
    louds_ref = meter.integrated_loudness(ref)

    s1_norm = pyln.normalize.loudness(s1, louds1, -29)
    s2_norm = pyln.normalize.loudness(s2, louds2, -29)
    ref_norm = pyln.normalize.loudness(ref, louds_ref, -23.0)

    amp_s1 = np.max(np.abs(s1_norm))
    amp_s2 = np.max(np.abs(s2_norm))
    amp_ref = np.max(np.abs(ref_norm))

    if amp_s1 == 0 or amp_s2 == 0 or amp_ref == 0:
        return

    if trim_db:
        ref, _ = librosa.effects.trim(ref_norm, top_db=trim_db)
        s1, _ = librosa.effects.trim(s1_norm, top_db=trim_db)
        s2, _ = librosa.effects.trim(s2_norm, top_db=trim_db)

    if len(ref) < sr:
        return

    path_mix = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-mixed.wav")
    path_target = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-target.wav")
    path_ref = os.path.join(out_dir, f"{target_id}_{noise_id}_" + "%06d" % idx + "-ref.wav")

    snr = np.random.choice(snr_levels, 1).item()

    if not test:
        s1, s2 = vad_merge(s1, vad_db), vad_merge(s2, vad_db)
        s1_cut, s2_cut = cut_audios(s1, s2, audio_len, sr)

        for i in range(len(s1_cut)):
            mix = snr_mixer(s1_cut[i], s2_cut[i], snr)

            louds1 = meter.integrated_loudness(s1_cut[i])
            s1_cut[i] = pyln.normalize.loudness(s1_cut[i], louds1, -23.0)
            loud_mix = meter.integrated_loudness(mix)
            mix = pyln.normalize.loudness(mix, loud_mix, -23.0)

            path_mix_i = path_mix.replace("-mixed.wav", f"_{i}-mixed.wav")
            path_target_i = path_target.replace("-target.wav", f"_{i}-target.wav")
            path_ref_i = path_ref.replace("-ref.wav", f"_{i}-ref.wav")
            sf.write(path_mix_i, mix, sr)
            sf.write(path_target_i, s1_cut[i], sr)
            sf.write(path_ref_i, ref, sr)
    else:
        s1, s2 = fix_length(s1, s2, 'max')
        mix = snr_mixer(s1, s2, snr)
        louds1 = meter.integrated_loudness(s1)
        s1 = pyln.normalize.loudness(s1, louds1, -23.0)

        loud_mix = meter.integrated_loudness(mix)
        mix = pyln.normalize.loudness(mix, loud_mix, -23.0)

        sf.write(path_mix, mix, sr)
        sf.write(path_target, s1, sr)
        sf.write(path_ref, ref, sr)
