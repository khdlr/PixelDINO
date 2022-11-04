import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.experimental.host_callback import id_print

def bce(y_true, y_pred):
    ignore = (y_true < -.5) | (y_true > 1.5)

    log_y_true  = jax.nn.log_sigmoid( y_pred)
    log_y_false = jax.nn.log_sigmoid(-y_pred)

    loss = -(y_true * log_y_true + (1-y_true) * log_y_false)
    loss = jnp.where(ignore, 0., loss)
    return jnp.mean(loss)


def focal_loss(y_true, y_pred):
    ignore = (y_true < -.5) | (y_true > 1.5)
    log_y_true  = jax.nn.log_sigmoid( y_pred)
    log_y_false = jax.nn.log_sigmoid(-y_pred)
    bce_loss = -(y_true * log_y_true + (1-y_true) * log_y_false)
    p_true = jnp.exp(-bce_loss)

    loss = bce_loss * (1 - p_true) ** 2.
    loss = jnp.where(ignore, 0., loss)
    return jnp.mean(loss)


def noop(y_true, y_pred):
  return 0
