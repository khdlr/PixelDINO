import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.experimental.host_callback import id_print

def bce(y_true, y_pred):
    ignore = y_true < -.5
    
    log_y_true  = jax.nn.log_sigmoid( y_pred)
    log_y_false = jax.nn.log_sigmoid(-y_pred)

    loss = -(y_true * log_y_true + (1-y_true) * log_y_false)
    loss = jnp.where(ignore, 0., loss)
    return jnp.mean(loss)
