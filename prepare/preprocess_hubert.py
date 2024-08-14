import os

import librosa
import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from hubert import hubert_model


def load_model(device):
    model = hubert_model.hubert_soft(
        os.path.join("hubert_pretrain", "hubert-soft-0d54a1f4.pt"),
    )
    model.eval()
    # FIXME:
    # RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
    # The operator 'aten::transformer_encoder_layer_fwd'
    # is not currently supprted on theNPU backend and will fall back to run on the CPU.
    model.half().to(device)
    return model


def pred_vec(device,wav_paths, vec_paths):
    def _pred_vec(wav_path, vec_path):
        feats, _ = librosa.load(wav_path)
        feats = torch.from_numpy(feats).to(device)
        feats = feats[None, None, :].half()
        with torch.no_grad():
            vec = hubert.units(feats).squeeze().data.cpu().float().numpy()
            np.save(vec_path, vec, allow_pickle=False)
        pbar.update()

    hubert = load_model(device)
    pbar = tqdm(total=len(wav_paths), desc=device)
    for wav_path, vec_path in zip(wav_paths, vec_paths):
        _pred_vec(wav_path, vec_path)


