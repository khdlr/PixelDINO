import jax
import jax.numpy as jnp
import haiku as hk
from ..config_mod import config


class Discriminator:
  def __init__(self):
    self.width = config.model.width

  def __call__(self, x):
    """cf. Hung et al. -- Adversarial learning for semi-supervised semantic segmentation, 2018"""
    B, H, W, C = x.shape

    x = hk.Conv2D( 64, 4, stride=2)(x)
    x = jax.nn.leaky_relu(x, negative_slope=0.2)
    x = hk.Conv2D(128, 4, stride=2)(x)
    x = jax.nn.leaky_relu(x, negative_slope=0.2)
    x = hk.Conv2D(256, 4, stride=2)(x)
    x = jax.nn.leaky_relu(x, negative_slope=0.2)
    x = hk.Conv2D(512, 4, stride=2)(x)
    x = jax.nn.leaky_relu(x, negative_slope=0.2)
    x = hk.Conv2D(1, 1)(x)
    return jax.image.resize(x, [B, H, W, 1], method='bilinear')
