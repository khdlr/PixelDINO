import jax
import jax.numpy as jnp
import haiku as hk
from ..config_mod import config
from . import backbones
from .backbones import nnutils as nn


class DeepLabv3p:
  def __init__(self):
    self.backbone = getattr(backbones, config.model.backbone)()

  def __call__(self, inp):
    x_hr, x_lr = self.backbone(inp)
    x_hr = nn.ConvLNAct(64, 3)(x_hr)

    B, H, W, C = x_hr.shape

    x = jnp.concatenate([
      x_hr,
      jax.image.resize(x_lr, [B, H, W, x_lr.shape[-1]], method='bilinear')
    ], axis=-1)

    x = nn.ConvLNAct(64, 3)(x)
    x = nn.ConvLNAct(64, 3)(x)

    x = jax.image.resize(x, inp.shape[:-1] + (64, ), method='bilinear')
    x = hk.Conv2D(1, 1)(x)
    return x


