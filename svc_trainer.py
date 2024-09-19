import os
import sys

import svc_preprocessing

import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from vits_extend.train import train

import click


@click.command()
@click.option("--config", "-c", help="yaml file for configuration.")
@click.option(
    "--checkpoint_path",
    "-p",
    default=None,
    help="path of checkpoint pt file to resume training.",
)
@click.option("--name", "-n", help="name of the model for logging, saving checkpoint.")
@click.option("--use_npu", is_flag=True, help="Flag to use NPU.")
def main(config, checkpoint_path, name, use_npu):
    hp = OmegaConf.load(config)
    with open(config, "r") as f:
        hp_str = "".join(f.readlines())

    assert hp.data.hop_length == 320, (
        "hp.data.hop_length must be equal to 320, got %d" % hp.data.hop_length
    )
    device, device_num = svc_preprocessing.get_device(use_npu)

    torch.manual_seed(hp.train.seed)

    if device_num > 1: 
        mp.spawn(
            train,
            nprocs=device_num,
            args=(
                device,
                device_num,
                name,
                checkpoint_path,
                hp,
                hp_str,
                use_npu
            ),
        )
    else:
        train(0, device, 1, name, checkpoint_path, hp, hp_str)


if __name__ == "__main__":
    main()
