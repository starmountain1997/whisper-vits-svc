import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim
from whisper.model import ModelDimensions, Whisper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model(device: str) -> Whisper:
    path = os.path.join("whisper_pretrain", "large-v2.pt")
    checkpoint = torch.load(path, map_location="cpu")
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    del model.decoder
    cut = len(model.encoder.blocks) // 4
    cut = -1 * cut
    del model.encoder.blocks[cut:]
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    model.half()
    model.to(device)
    return model


def pred_ppg(device, wav_paths, ppg_paths):
    def _pred_ppg(wav_path, ppg_path):
        audio = load_audio(wav_path)
        audln = audio.shape[0]
        ppgln = audln // 320
        audio = pad_or_trim(audio)
        mel = log_mel_spectrogram(audio).half().to(whisper.device)
        with torch.no_grad():
            ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
            ppg = ppg[:ppgln,]  # [length, dim=1280]
            np.save(ppg_path, ppg, allow_pickle=False)
        pbar.update()

    whisper = load_model(device)
    pbar = tqdm(total=len(wav_paths), desc=device)
    for wav_path, ppg_path in zip(wav_paths, ppg_paths):
        _pred_ppg(wav_path, ppg_path)
