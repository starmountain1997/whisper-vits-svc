import numpy as np
import torchcrepe
from tqdm import tqdm


def compute_f0(device, filenames, saves):
    def _compute_f0(filename, save):
        audio, sr = torchcrepe.load.audio(filename)
        # Here we'll use a 10 millisecond hop length
        hop_length = int(sr / 200.0)
        # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
        # This would be a reasonable range for speech
        fmin = 50
        fmax = 1000
        # Select a model capacity--one of "tiny" or "full"
        model = "full"
        # Pick a batch size that doesn't cause memory errors on your gpu
        batch_size = 2048
        # Compute pitch using first gpu
        pitch, periodicity = torchcrepe.predict(
            audio,
            sr,
            hop_length,
            fmin,
            fmax,
            model,
            batch_size=batch_size,
            device=device,
            return_periodicity=True,
        )
        # CREPE was not trained on silent audio. some error on silent need
        # filter.pit_path
        periodicity = torchcrepe.filter.median(periodicity, 7)
        pitch = torchcrepe.filter.mean(pitch, 5)
        pitch[periodicity < 0.5] = 0
        pitch = pitch.squeeze(0)
        np.save(save, pitch, allow_pickle=False)
        pbar.update()

    pbar = tqdm(total=len(filenames), desc=device)
    for filename, save in zip(filenames, saves):
        _compute_f0(filename, save)
