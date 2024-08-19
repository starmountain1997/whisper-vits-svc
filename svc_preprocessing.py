import multiprocessing
import os
from typing import Callable

import click
import torch
from loguru import logger

from prepare import preprocess_crepe

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_device(use_npu: bool):
    if use_npu:
        import torch_npu

        logger.info(f"import torch_npu: {torch_npu.__version__}.")
        device = "npu" if torch.npu.is_available() else "cpu"
        device_num = torch.npu.device_count() if device == "npu" else 1
    else:
        torch.backends.cudnn.benchmark = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_num = torch.cuda.device_count() if device == "cuda" else 1

    return device, device_num


def call_torch_mp(
    func: Callable,
    wav_path: str,
    output_path: str,
    device: str,
    p_num: int,
    output_ext: str = "wav",
):

    files = []

    for spks in os.listdir(wav_path):
        if os.path.isdir(os.path.join(wav_path, spks)):
            os.makedirs(os.path.join(output_path, spks), exist_ok=True)
            files = files + [
                f"{spks}/{f[:-4]}"
                for f in os.listdir(os.path.join(wav_path, spks))
                if f.endswith(".wav")
            ]
    if p_num == 1:
        func(
            device,
            [os.path.join(wav_path, f"{file}.wav") for file in files],
            [os.path.join(output_path, f"{file}.{output_ext}") for file in files],
        )
    else:
        files_chunk = [files[i::p_num] for i in range(p_num)]
        task_args = [
            (
                f"{device}:{gpu_id}" if device != "cpu" else "cpu",
                [os.path.join(wav_path, f"{file}.wav") for file in chunk],
                [os.path.join(output_path, f"{file}.{output_ext}") for file in chunk],
            )
            for gpu_id, chunk in enumerate(files_chunk)
        ]

        processes = []
        for task_arg in task_args:
            if device == "cpu":
                p = multiprocessing.Process(target=func, args=task_arg)
            else:
                p = torch.multiprocessing.Process(target=func, args=task_arg)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


@click.command()
@click.option("--thread_count", "-t", default=0, help="Number of threads to use.")
@click.option("--use_npu", is_flag=True, help="Flag to use NPU.")
def main(thread_count, use_npu):
    if thread_count == 0:
        thread_count = os.cpu_count() // 2 + 1
    device, device_num = get_device(use_npu)
    logger.info(
        f"Device: {device}, Device number: {device_num}, thread_count: {thread_count}."
    )
    # preprocess_resample.main(
    #     os.path.join(PROJECT_DIR, "dataset_raw"),
    #     os.path.join(PROJECT_DIR, "data_svc/waves-32k"),
    #     32000,
    #     thread_count,
    # )
    # preprocess_resample.main(
    #     os.path.join(PROJECT_DIR, "dataset_raw"),
    #     os.path.join(PROJECT_DIR, "data_svc/waves-16k"),
    #     16000,
    #     thread_count,
    # )
    call_torch_mp(
        preprocess_crepe.compute_f0,
        os.path.join(PROJECT_DIR, "data_svc/waves-16k"),
        os.path.join(PROJECT_DIR, "data_svc/pitch"),
        device,
        device_num,
        output_ext="pit",
    )
    # call_torch_mp(
    #     preprocess_ppg.pred_ppg,
    #     os.path.join(PROJECT_DIR, "data_svc/waves-16k"),
    #     os.path.join(PROJECT_DIR, "data_svc/whisper"),
    #     device,
    #     device_num,
    #     output_ext="ppg",
    # )

    # call_torch_mp(
    #     preprocess_hubert.pred_vec,
    #     os.path.join(PROJECT_DIR, "data_svc/waves-16k"),
    #     os.path.join(PROJECT_DIR, "data_svc/hubert"),
    #     device,
    #     device_num,
    #     output_ext="vec",
    # )
    # call_torch_mp(
    #     preprocess_speaker.extract_speaker_embeddings,
    #     os.path.join(PROJECT_DIR, "data_svc/waves-16k"),
    #     os.path.join(PROJECT_DIR, "data_svc/speaker"),
    #     device,
    #     device_num,
    # )
    # preprocess_speaker_ave.main(
    #     os.path.join(PROJECT_DIR, "data_svc/speaker"),
    #     os.path.join(PROJECT_DIR, "data_svc/singer"),
    # )
    # preprocess_spec.main(
    #     os.path.join(PROJECT_DIR, "data_svc/waves-32k"),
    #     os.path.join(PROJECT_DIR, "data_svc/specs"),
    #     thread_count,
    # )
    # preprocess_train.main()
    # preprocess_zzz.main()


if __name__ == "__main__":
    main()
