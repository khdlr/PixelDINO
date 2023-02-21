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
    xs = self.backbone(inp)
    x_hr, *_, x = xs
    x_hr = nn.ConvLNAct(64, 3)(x_hr)

    # ASPP
    # Image Feature branch
    bD = hk.max_pool(x, window_shape=2, strides=2, padding='SAME')
    bD = nn.ConvLNAct(256, 1, act='elu')(bD)
    bD = nn.upsample(bD, factor=2)

    b0 = nn.ConvLNAct(32, 1, act='elu')(x)
    b1 = nn.SepConvLN(32, rate=1)(x)
    b2 = nn.SepConvLN(32, rate=2)(x)
    b3 = nn.SepConvLN(32, rate=3)(x)
    b4 = nn.SepConvLN(32, rate=4)(x)
    b5 = nn.SepConvLN(32, rate=5)(x)
    x = jnp.concatenate([bD, b0, b1, b2, b3, b4, b5], axis=-1)

    x = nn.ConvLNAct(128, 1, act='elu')(x)
    # skip3 = nn.ConvLNAct(64, 1, act='elu')(skip3)

    B, H, W, _ = x_hr.shape

    x = jnp.concatenate([
      x_hr,
      jax.image.resize(x, [B, H, W, x.shape[-1]], method='bilinear')
    ], axis=-1)

    x = nn.ConvLNAct(64, 3)(x)
    x = nn.ConvLNAct(64, 3)(x)

    x = jax.image.resize(x, inp.shape[:-1] + (64, ), method='bilinear')
    x = hk.Conv2D(1, 1)(x)
    return x


