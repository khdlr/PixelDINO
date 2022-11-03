import jax
import jax.numpy as jnp
from . import config
import math
from einops import rearrange



def get_alpha(t):
  schedule = config.diffusion.alpha_schedule
  x = (t / (1 + config.diffusion.steps_train))

  alpha = 0
  if schedule == 'linear':
    alpha = 1 - x
  elif schedule == 'circular':
    alpha = 1 - jnp.sqrt(2*x - x*x)
  elif schedule == 'sinusoidal':
    alpha = jnp.sin(jnp.pi / 2 * (1 - x))
  elif schedule == 'cosine':
    alpha = 0.5 + 0.5 * jnp.cos(x * jnp.pi)
  else:
    raise ValueError(f'{config.diffusion.alpha_schedule!r}')
  return alpha[..., None, None]


# DDIM sampling procedure for a single image
def ddim_sample(model, params, img_features, timesteps, init):
  # DDIM inference step
  def inference_step(x_t, step_vars):
    t, tm1 = step_vars
    a_t, a_tm1 = get_alpha(t), get_alpha(tm1)

    eps = model.predict_next(params, x_t, img_features, t)
    x0_est = (x_t - eps * jnp.sqrt(1 - a_t)) / jnp.sqrt(a_t)
    x_tm1 = jnp.sqrt(a_tm1) * x0_est + jnp.sqrt(1 - a_tm1) * eps

    # Clipping
    x_tm1 = jnp.clip(x_tm1, -1, 1)

    return x_tm1, x_tm1

  step_vars = (timesteps[:-1], timesteps[1:])
  # Vmap over samples axis
  inference_step = jax.vmap(inference_step, in_axes=[1, None], out_axes=1)
  final, steps = jax.lax.scan(inference_step, init, step_vars)
  return rearrange(steps, 'diffusion_step B S T C -> B S diffusion_step T C')
