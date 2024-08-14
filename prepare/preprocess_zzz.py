from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from vits.data_utils import (DistributedBucketSampler, TextAudioSpeakerCollate,
                             TextAudioSpeakerSet)


def main():
    hps = OmegaConf.load("./configs/base.yaml")
    dataset = TextAudioSpeakerSet("files/valid.txt", hps.data)

    for _ in tqdm(dataset):
        pass

    sampler = DistributedBucketSampler(
        dataset, 4, [150, 300, 450], num_replicas=1, rank=0, shuffle=True
    )
    collate_fn = TextAudioSpeakerCollate()
    loader = DataLoader(
        dataset,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=sampler,
    )

    for _ in tqdm(loader):
        pass
