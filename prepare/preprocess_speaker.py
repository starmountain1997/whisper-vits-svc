import os.path as osp

import numpy as np
import torch
import torch.multiprocessing.pool
from tqdm import tqdm

from speaker.config import SpeakerEncoderConfig
from speaker.infer import read_json
from speaker.models.lstm import LSTMSpeakerEncoder
from speaker.utils.audio import AudioProcessor


def extract_speaker_embeddings(
    device,
    filenames,
    saves,
):
    def _extract_speaker_embeddings(wav_file, save):
        waveform = speaker_encoder_ap.load_wav(
            wav_file, sr=speaker_encoder_ap.sample_rate
        )
        spec = speaker_encoder_ap.melspectrogram(waveform)
        spec = torch.from_numpy(spec.T).to(device)
        spec = spec.unsqueeze(0)
        embed = speaker_encoder.compute_embedding(spec).detach().cpu().numpy()
        embed = embed.squeeze()
        np.save(save[:-4] + ".spk", embed, allow_pickle=False)
        pbar.update()

    config_dict = read_json(osp.join("speaker_pretrain", "config.json"))

    # model
    config = SpeakerEncoderConfig(config_dict)
    config.from_dict(config_dict)

    speaker_encoder = LSTMSpeakerEncoder(
        config.model_params["input_dim"],
        config.model_params["proj_dim"],
        config.model_params["lstm_dim"],
        config.model_params["num_lstm_layers"],
    )

    from svc_preprocessing import PROJECT_DIR

    speaker_encoder.load_checkpoint(
        osp.join(PROJECT_DIR, "speaker_pretrain", "best_model.pth.tar"),
        eval=True,
        device=device,
    )

    # preprocess
    speaker_encoder_ap = AudioProcessor(**config.audio)
    # normalize the input audio level and trim silences
    speaker_encoder_ap.do_sound_norm = True
    speaker_encoder_ap.do_trim_silence = True

    pbar = tqdm(total=len(filenames), desc=device)
    for wav_file, save in zip(filenames, saves):
        _extract_speaker_embeddings(wav_file, save)
