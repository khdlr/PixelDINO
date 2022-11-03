import jax
import jax.numpy as jnp
import haiku as hk
from .. import config


class UNet:
  def __init__(self):
    self.width = config.model.width

  def __call__(self, x):
    skip_connections = []

    W = self.width
    channel_seq = [W, 2*W, 4*W, 8*W]
    for channels in channel_seq:
      x = Convx2(x, channels)
      skip_connections.append(x)
      x = hk.max_pool(x, 2, 2, padding='SAME')

    x = Convx2(x, 16*W)

    for channels, skip in zip(reversed(channel_seq), reversed(skip_connections)):
      B,  H,  W,  C  = x.shape
      B_, H_, W_, C_ = skip.shape

      upsampled = jax.image.resize(x, [B, H_, W_, C], method='bilinear')
      x = hk.Conv2D(C_, 2)(upsampled)
      x = Norm()(x)
      x = jax.nn.relu(x)
      x = Convx2(jnp.concatenate([x, skip], axis=-1), channels)

    x = hk.Conv2D(1, 1)(x)
    return x


def Norm():
  return hk.LayerNorm(axis=-1, param_axis=-1,
          create_scale=True, create_offset=True)


def Convx2(x, channels):
  x = hk.Conv2D(channels, 3)(x)
  x = Norm()(x)
  x = jax.nn.relu(x)
  x = hk.Conv2D(channels, 3)(x)
  x = Norm()(x)
  x = jax.nn.relu(x)
  return x
