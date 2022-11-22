import jax
import jax.numpy as jnp
import haiku as hk


def identity(x, *aux, **kwaux):
    return x


class ReLU(hk.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.relu(x)


class LeakyReLU(hk.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.leaky_relu(x)


class ConvLNAct(hk.Module):
    def __init__(self, *args, ln=True, act='relu', **kwargs):
        super().__init__()
        kwargs['with_bias'] = False
        self.conv = hk.Conv2D(*args, **kwargs)

        if ln:
            self.ln = hk.LayerNorm(axis=-1, param_axis=-1,
                        create_scale=True, create_offset=True)
        else:
            self.ln = identity

        if act is None:
            self.act = identity
        elif hasattr(jax.nn, act):
            self.act = getattr(jax.nn, act)
        else:
            raise ValueError(f"no activation called {act}")

    def __call__(self, x):
        x = self.conv(x)
        x = self.ln(x)
        x = self.act(x)
        return x


class SepConvLN(hk.Module):
    def __init__(self, filters, stride=1, kernel_size=3, rate=1, depth_activation=False):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.rate = rate
        self.filters = filters
        self.depth_activation = depth_activation

    def __call__(self, x):
        if self.depth_activation:
            x = jax.nn.relu(x)
        x = hk.Conv2D(self.filters, 1)(x)
        x = ConvLNAct(self.filters, self.kernel_size, stride=self.stride,
            rate=self.rate, feature_group_count=self.filters)(x)

        return x


def upsample(x, factor=None, shp=None):
    B, H, W, C = x.shape
    if factor is not None:
        H *= factor
        W *= factor
    else:
        H, W = shp
    return jax.image.resize(x, [B, H, W, C], 'bilinear')


def channel_dropout(x, rate):
    if rate < 0 or rate >= 1:
        raise ValueError("rate must be in [0, 1).")

    if rate == 0.0:
        return x

    keep_rate = 1.0 - rate
    mask_shape = (x.shape[0], *((1,) * (x.ndim - 2)), x.shape[-1])

    keep = jax.random.bernoulli(hk.next_rng_key(), keep_rate, shape=mask_shape)
    return keep * x / keep_rate
