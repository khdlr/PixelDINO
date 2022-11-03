from .unet import UNet

import jax
import haiku as hk
from collections import namedtuple

model_type = namedtuple("Model", "get_features predict_next")


def get_model(*dummy_in, seed=jax.random.PRNGKey(39)):
    model = hk.without_apply_rng(hk.transform(lambda x: UNet()(x)))
    params = model.init(seed, *jax.tree_map(lambda x: x[:1], dummy_in))

    return model, params
