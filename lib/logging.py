import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import wandb
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image, ImageDraw

from io import BytesIO
from PIL import Image
import numpy as np
import base64

from argparse import ArgumentParser
from einops import rearrange

from .utils import min_pool
from .metrics import compute_metrics


def log_metrics(metrics, prefix, step, do_print=True, do_wandb=True):
    logged_metrics = {}
    for key, values in metrics.items():
        if key.endswith('_premetrics'):
          values = {k: np.sum(np.stack([batch[k] for batch in values])) for k in values[0]}
          tag = key.removesuffix('_premetrics')
          new_metrics = compute_metrics(values)
          logged_metrics.update({ f'{tag}_{k}': new_metrics[k] for k in new_metrics })
        else:
          logged_metrics[key] = np.mean(values)

    if do_wandb:
        wandb.log({f"{prefix}/{k}": v for k, v in logged_metrics.items()}, step=step)
    if do_print:
        print(f"{prefix}/metrics")
        print(", ".join(f"{k}: {v:.3f}" for k, v in logged_metrics.items()))


def get_rgb(data):
    img = data["imagery"]
    if img.shape[-1] == 1:
        rgb = np.concatenate([img] * 3, axis=-1)
    elif img.shape[-1] == 10:
        rgb = img[..., 3:0:-1]
    rgb = np.clip(255 * rgb, 0, 255).astype(np.uint8)
    return rgb


def log_segmentation(data, tag, step):
    H, W, C = data["imagery"].shape

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    for ax in axs:
        ax.axis("off")
    axs[0].imshow(get_rgb(data))
    axs[1].imshow(
        np.asarray(data["segmentation"][:, :, 0]), cmap="gray", vmin=-1, vmax=1
    )
    axs[2].imshow(np.asarray(data["mask"]), cmap="gray", vmin=0, vmax=1)

    wandb.log({tag: wandb.Image(fig)}, step=step)
    plt.close(fig)
