"""
Ported to haiku from https://keras.io/examples/vision/deeplabv3_plus/
"""
import jax
import jax.numpy as jnp
import haiku as hk
from . import backbones
from ..config_mod import config
from einops import repeat

class DeepLabv3p:
  def __init__(self):
    self.backbone = getattr(backbones, config.model.backbone)()

  def __call__(self, inp):
    B, H, W, C = inp.shape
    xs = self.backbone(inp)
    for i, x in enumerate(xs):
      print(f'xs[{i}]:', x.shape)
    _, input_b, _, input_a = xs

    B, H_, W_, _ = input_b.shape

    x_a = ASPP(input_a)
    x_a = jax.image.resize(x_a, [B, H_, W_, C], method='bilinear')

    x_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = jnp.concatenate([x_a, x_b], axis=-1)
    x = convolution_block(x)
    x = convolution_block(x)
    x = jax.image.resize(x, [B, H, W, x.shape[-1]], method='bilinear')

    x = hk.Conv2D(1, 1)(x)
    return x


def ASPP(x_in):
  B, H, W, C = x_in.shape
  x = jnp.mean(x_in, axis=[1,2], keepdims=True)
  out_pool = convolution_block(x, kernel_size=1, use_bias=True)
  out_pool = repeat(out_pool, 'B 1 1 C -> B H W C', H=H, W=W)

  out_1 = convolution_block(x_in, kernel_size=1, dilation_rate=1)
  out_3 = convolution_block(x_in, kernel_size=3, dilation_rate=3)
  out_6 = convolution_block(x_in, kernel_size=3, dilation_rate=6)
  out_9 = convolution_block(x_in, kernel_size=3, dilation_rate=9)
  # out_12 = convolution_block(x_in, kernel_size=3, dilation_rate=12)
  # out_18 = convolution_block(x_in, kernel_size=3, dilation_rate=18)

  x = jnp.concatenate([out_pool, out_1, out_3, out_6, out_9], axis=-1)
  output = convolution_block(x, kernel_size=1)
  return output


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    use_bias=False,
):
    x = hk.Conv2D(num_filters, kernel_size, 
        rate=dilation_rate, padding="same", with_bias=use_bias,
        w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')
    )(block_input)

    x = hk.LayerNorm(axis=-1, param_axis=-1,
                    create_scale=True, create_offset=True)(x)
    return jax.nn.relu(x)


