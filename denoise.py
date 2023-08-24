from pathlib import Path
import os
import math
import sys
import random
import numpy as np


import typer
from tqdm import tqdm

import torch
from accelerate import PartialState

from diffusers import DDPMScheduler, UNet2DModel
from tpdmpipeline import TPDMPipeline

cur_dir = Path(__file__).parent
stylegan_dir = cur_dir / "stylegan"
sys.path.append(stylegan_dir.as_posix())

# StyleGAN imports
import dnnlib
import legacy

app = typer.Typer(pretty_exceptions_show_locals=False)


def seed_all(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def chunk_into_n_list(lst, n):
    size = math.ceil(len(lst) / n)
    return list(map(lambda x: lst[x * size : x * size + size], list(range(n))))


def chunk_into_list_of_n(lst, n):
    return [lst[i * n : (i + 1) * n] for i in range((len(lst) + n - 1) // n)]


@app.command()
def main(
    unet_ckpt_path: Path = typer.Argument(
        ..., help="ckpt path", file_okay=True, exists=True
    ),
    gan_ckpt_path: Path = typer.Argument(
        ..., help="ckpt path", file_okay=True, exists=True
    ),
    output_dir: Path = typer.Argument(..., help="output dir"),
    n_samples: int = typer.Option(50000, "-n", help="Number of samples to generate"),
    batch_size: int = typer.Option(16, "-b", help="Batchsize"),
    truncated_step: int = typer.Option(499, "-s", help="truncated step"),
    load_ema: bool = typer.Option(True, help="Load EMA Model"),
    use_safetensors: bool = typer.Option(True, help="Use Safetensors"),
    unet_subfolder_name: str = typer.Option("unet", help="Subfolder name"),
    gpu: bool = typer.Option(True, help="Use GPU"),
    seed: int = typer.Option(0, help="Seed"),
):
    if load_ema:
        unet_subfolder_name += "_ema"
    device = torch.device("cpu") if not gpu else torch.device("cuda:0")
    # device = None
    parent_folder = unet_ckpt_path.parent
    unet = UNet2DModel.from_pretrained(
        unet_ckpt_path.as_posix(),
        subfolder=unet_subfolder_name,
        use_safetensors=use_safetensors,
    )
    unet = unet.eval()
    scheduler = DDPMScheduler.from_pretrained(
        parent_folder.as_posix(), subfolder="scheduler"
    )

    with dnnlib.util.open_url(gan_ckpt_path.as_posix()) as f:
        G = legacy.load_network_pkl(f)["G_ema"].eval()  # type: ignore
    G.dtype = torch.float16

    pipeline = TPDMPipeline(unet=unet, scheduler=scheduler, gan_generator=G)
    distributed_state = PartialState()
    world_size = distributed_state.num_processes
    pipeline.to(distributed_state.device)

    chunks = chunk_into_n_list(list(range(n_samples)), world_size)
    output_dir.mkdir(exist_ok=True, parents=True)
    with distributed_state.split_between_processes(chunks) as chunk:
        seed_all(seed + distributed_state.local_process_index)
        chunk = chunk[0]

        batched = chunk_into_list_of_n(chunk, batch_size)
        print(f"{distributed_state.local_process_index}: batched={batched}")
        pbar = (
            tqdm(range(len(batched)), desc="Generating...")
            if distributed_state.is_main_process
            else range(len(batched))
        )
        for i in pbar:
            real_bs = len(batched[i])
            assert real_bs <= batch_size
            images = pipeline(
                batch_size=real_bs,
                truncated_step=truncated_step,
            ).images
            for j, image in zip(batched[i], images):
                outpath = output_dir / f"{str(j).zfill(8)}.png"
                image.save(outpath.as_posix())


if __name__ == "__main__":
    app()
