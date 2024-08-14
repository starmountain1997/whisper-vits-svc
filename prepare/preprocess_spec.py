import os
import os.path as osp
from functools import partial

import torch
from omegaconf import OmegaConf
from tqdm.contrib.concurrent import process_map

from vits import spectrogram, utils


def compute_spec(filename, specname, hps):
    audio, sampling_rate = utils.load_wav_to_torch(filename)
    assert (
        sampling_rate == hps.sampling_rate
    ), f"{sampling_rate} is not {hps.sampling_rate}"
    audio_norm = audio / hps.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    n_fft = hps.filter_length
    sampling_rate = hps.sampling_rate
    hop_size = hps.hop_length
    win_size = hps.win_length
    spec = spectrogram.spectrogram_torch(
        audio_norm, n_fft, sampling_rate, hop_size, win_size, center=False
    )
    spec = torch.squeeze(spec, 0)
    torch.save(spec, specname)


def main(wav_path: str, out_path: str, thread_count: int):
    wav_ins = []
    wav_outs = []
    for spks in os.listdir(wav_path):
        if os.path.isdir(osp.join(wav_path, spks)):
            os.makedirs(osp.join(out_path, spks), exist_ok=True)
            wav_ins = wav_ins + [
                osp.join(wav_path, spks, f)
                for f in os.listdir(osp.join(wav_path, spks))
                if f.endswith(".wav")
            ]
            wav_outs = wav_outs + [
                osp.join(out_path, spks, f[:-4] + ".pt")
                for f in os.listdir(osp.join(wav_path, spks))
                if f.endswith(".wav")
            ]
    hps = OmegaConf.load("./configs/base.yaml")
    process_map(
        partial(compute_spec, hps=hps.data),
        wav_ins,
        wav_outs,
        max_workers=thread_count,
    )
