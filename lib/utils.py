import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import augmax
import optax
from subprocess import check_output
from einops import rearrange
from typing import Union, Sequence, Optional, Tuple
from typing import NamedTuple
import pickle
from . import models
from .config_mod import config


class TrainingState(NamedTuple):
    params: hk.Params
    opt: optax.OptState


def changed_state(state, params=None, opt=None):
    return TrainingState(
        params=state.params if params is None else params,
        opt=state.opt if opt is None else opt,
    )


def save_state(state, out_path):
    state = jax.device_get(state)
    with out_path.open("wb") as f:
        pickle.dump(state, f)


def load_state(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        state = pickle.load(f)
    return state


def get_model(*dummy_in, seed=jax.random.PRNGKey(39)):
  model_cls = getattr(models, config.model.type)
  model = hk.without_apply_rng(hk.transform(lambda x: model_cls()(x)))
  params = model.init(seed, *jax.tree_map(lambda x: x[:1], dummy_in))

  return model, params


def prep(batch, augment_key=None):
    ops = []
    if augment_key is not None:
        ops += [
            augmax.HorizontalFlip(),
            augmax.VerticalFlip(),
            augmax.Rotate90(),
        ]
    # if augment: ops += [
    #     augmax.ChannelShuffle(p=0.1),
    #     augmax.Solarization(p=0.1),
    # ]

    all_types = {
        'Sentinel2': augmax.InputType.IMAGE,
        'Mask': augmax.InputType.MASK,
    }
    input_types = {k: all_types[k] for k in batch}
    chain = augmax.Chain(*ops, input_types=input_types)
    if augment_key is None:
        augment_key = jax.random.PRNGKey(0)

    batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
    subkeys = jax.random.split(augment_key, batch_size)
    transformation = jax.vmap(chain)
    outputs = transformation(subkeys, batch)

    return outputs


def distort(batch, augment_key):
    ops = [
        augmax.Rotate(),
        augmax.Warp(coarseness=16, strength=2),
        # augmax.RandomGamma(p=1., range=[0.75, 1.33]),
        # augmax.RandomBrightness(p=1.),
        # augmax.RandomContrast(p=1.),
        augmax.GaussianBlur(sigma=2)
    ]

    all_types = {
        'Sentinel2': augmax.InputType.IMAGE,
        'Mask': augmax.InputType.MASK,
    }
    input_types = {k: all_types[k] for k in batch}
    chain = augmax.Chain(*ops, input_types=input_types)

    batch_size = jax.tree_util.tree_leaves(batch)[0].shape[0]
    subkeys = jax.random.split(augment_key, batch_size)
    transformation = jax.vmap(chain)
    outputs = transformation(subkeys, batch)

    return outputs


def deterministic_loop(length):
  while True:
    for i in range(length):
      yield i


def shuffle_loop(length):
  while True:
    for i in np.random.permutation(length):
      yield i


def distance_matrix(a, b):
    a = rearrange(a, "(true pred) d -> true pred d", true=1)
    b = rearrange(b, "(true pred) d -> true pred d", pred=1)
    D = jnp.sum(jnp.square(a - b), axis=-1)
    return D


def pad_inf(inp, before, after):
    return jnp.pad(inp, (before, after), constant_values=jnp.inf)


def fmt_num(x):
    if jnp.isinf(x):
        return "∞".rjust(8)
    else:
        return f"{x:.2f}".rjust(8)


def fmt(xs, extra=None):
    tag = ""
    if isinstance(xs, str):
        tag = xs
        xs = extra
    rank = len(xs.shape)
    if rank == 1:
        print(tag, ",".join([fmt_num(x) for x in xs]))
    elif rank == 2:
        print("\n".join(",".join([fmt_num(x) for x in row]) for row in xs))
        print()


def assert_git_clean():
    diff = (
        check_output(["git", "diff", "--name-only", "HEAD"])
        .decode("utf-8")
        .splitlines()
    )
    if diff and diff != ["config.yml"]:
        assert False, "Won't run on a dirty git state!"


def _infer_shape(
    x: jnp.ndarray,
    size: Union[int, Sequence[int]],
    channel_axis: Optional[int] = -1,
) -> Tuple[int, ...]:
    """Infer shape for pooling window or strides."""
    if isinstance(size, int):
        if channel_axis and not 0 <= abs(channel_axis) < x.ndim:
            raise ValueError(f"Invalid channel axis {channel_axis} for {x.shape}")
        if channel_axis and channel_axis < 0:
            channel_axis = x.ndim + channel_axis
        return (1,) + tuple(size if d != channel_axis else 1 for d in range(1, x.ndim))
    elif len(size) < x.ndim:
        # Assume additional dimensions are batch dimensions.
        return (1,) * (x.ndim - len(size)) + tuple(size)
    else:
        assert x.ndim == len(size)
        return tuple(size)


def min_pool(
    value: jnp.ndarray,
    window_shape: Union[int, Sequence[int]],
    strides: Union[int, Sequence[int]],
    padding: str = "SAME",
    channel_axis: Optional[int] = -1,
) -> jnp.ndarray:
    """Min pool.
    Args:
      value: Value to pool.
      window_shape: Shape of the pooling window, an int or same rank as value.
      strides: Strides of the pooling window, an int or same rank as value.
      padding: Padding algorithm. Either ``VALID`` or ``SAME``.
      channel_axis: Axis of the spatial channels for which pooling is skipped,
        used to infer ``window_shape`` or ``strides`` if they are an integer.
    Returns:
      Pooled result. Same rank as value.
    """
    if padding not in ("SAME", "VALID"):
        raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")

    window_shape = _infer_shape(value, window_shape, channel_axis)
    strides = _infer_shape(value, strides, channel_axis)

    return jax.lax.reduce_window(
        value, jnp.inf, jax.lax.min, window_shape, strides, padding
    )


def fnot(fun):
    return lambda x: not fun(x)
