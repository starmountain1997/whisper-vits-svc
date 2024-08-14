import os
from functools import partial

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm.contrib.concurrent import process_map


def resample_wave(wav_in, wav_out, sample_rate):
    wav, _ = librosa.load(wav_in, sr=sample_rate)
    wav = wav / np.abs(wav).max() * 0.6
    wav = wav / max(0.01, np.max(np.abs(wav))) * 32767 * 0.6
    wavfile.write(wav_out, sample_rate, wav.astype(np.int16))


def main(wav_path: str, out_path: str, sample_rate: int, thread_count: int):
    wav_ins = []
    wav_outs = []

    for spks in os.listdir(wav_path):
        if os.path.isdir(os.path.join(wav_path, spks)):
            os.makedirs(os.path.join(out_path, spks), exist_ok=True)
            wav_ins = wav_ins + [
                os.path.join(wav_path, spks, f)
                for f in os.listdir(os.path.join(wav_path, spks))
                if f.endswith(".wav")
            ]
            wav_outs = wav_outs + [
                os.path.join(out_path, spks, f)
                for f in os.listdir(os.path.join(wav_path, spks))
                if f.endswith(".wav")
            ]

    process_map(
        partial(resample_wave, sample_rate=sample_rate),
        wav_ins,
        wav_outs,
        max_workers=thread_count,
    )
