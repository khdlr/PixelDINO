import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd
import haiku as hk
from functools import partial
from . import nnutils as nn


class Xception(hk.Module):
    """Xception backbone like the one used in CALFIN"""
    def __call__(self, x):
        B, H, W, C = x.shape

        # Backbone
        x, skip1 = XceptionBlock([32, 32, 32], stride=2, return_skip=True)(x)
        x, skip2 = XceptionBlock([64, 64, 64], stride=2, return_skip=True)(x)
        x, skip3 = XceptionBlock([96, 96, 96], stride=2, return_skip=True)(x)
        for i in range(8):
            x = XceptionBlock([96, 96, 96], skip_type='sum', stride=1)(x)

        x = XceptionBlock([ 96, 128, 128], stride=2)(x)
        x = XceptionBlock([192, 192, 256], stride=1, rate=(1, 2, 4))(x)

        return [skip2, x]


class XceptionBlock(hk.Module):
    def __init__(self, depth_list, stride, skip_type='conv',
                 rate=1, return_skip=False):
        super().__init__()
        self.blocks = []
        if rate == 1:
            rate = [1, 1, 1]
        for i in range(3):
            self.blocks.append(nn.SepConvLN(
                depth_list[i],
                stride=stride if i == 2 else 1,
                rate=rate[i],
            ))

        if skip_type == 'conv':
            self.shortcut = nn.ConvLNAct(depth_list[-1], 1, stride=stride, act=None)
        elif skip_type == 'sum':
            self.shortcut = nn.identity
        self.return_skip = return_skip

    def __call__(self, inputs):
        residual = inputs
        for i, block in enumerate(self.blocks):
            residual = block(residual)
            if i == 1:
                skip = residual

        shortcut = self.shortcut(inputs)
        outputs = residual + shortcut

        if self.return_skip:
            return outputs, skip
        else:
            return outputs
