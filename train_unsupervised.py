import argparse
import yaml
from pathlib import Path

import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from tempfile import mkstemp
from lib.lion import lion
from collections import defaultdict
from PIL import Image
import matplotlib as mpl
from einops import rearrange

import wandb
from tqdm import tqdm

from munch import munchify
from lib.data_loading import get_datasets, get_unlabelled
from lib import utils, logging, losses
from lib.config_mod import config
from lib.metrics import compute_premetrics
from lib.utils import DINOState, prep, distort, changed_state, save_state

jax.config.update("jax_numpy_rank_promotion", "raise")

def get_optimizer():
  conf = config.optimizer
  schedule = getattr(optax, conf.schedule)(**conf.schedule_args)
  if conf.type == 'lion':
    opt_class = lion
  else:
    opt_class = getattr(optax, conf.type)
  return opt_class(schedule, **conf.args)


def get_loss_fn(mode):
  name = config.loss_functions[f'{mode}']
  return getattr(losses, name)


@jax.jit
def train_step(unlabelled, state, key):
  _, optimizer = get_optimizer()

  key_1a, key_1b, key_2, key_3, key_4 = jax.random.split(key, 5)

  batch = prep(unlabelled, key_2)
  img_1 = img_2 = batch['img']
  img_2 = distort({'img': img_2}, key_3)['img']
  _, feat_1 = model(state.teacher, img_1, return_features=True)

  center = feat_1.mean(axis=[0, 1, 2], keepdims=True)
  feat_1 = (feat_1 - state.center) / config.train.temperature
  feat_1 = jax.nn.softmax(feat_1, axis=-1)
  feat_1 = distort({'features': feat_1}, key_4)['features']

  def get_loss(params):
    terms = {}
    _, feat_2 = model(params, img_2, return_features=True)

    # Dino-Style loss: feat_1 == "teacher", feat_2 == "student"
    terms['loss'] = optax.softmax_cross_entropy(feat_2, feat_1).mean()

    return terms['loss'], terms

  gradients, terms = jax.grad(get_loss, has_aux=True)(state.params)
  updates, new_opt = optimizer(gradients, state.opt, state.params)
  new_params = optax.apply_updates(state.params, updates)

  # EMA steps
  progress = new_opt[0].count / config.train.steps
  ema = config.train.teacher_ema
  ema_sched = 0.5 - 0.5 * jnp.cos(jnp.pi * progress)
  ema =  (1 - ema_sched) * ema + ema_sched # * 1  # Increases to 1 with cosine schedule
  teacher = jax.tree_map(lambda old, new: ema * old + (1 - ema) * new, state.teacher, new_params)
  c = config.train.center_ema
  center = c * state.center + (1 - c) * center
  return terms, changed_state(state,
      params=new_params,
      teacher=teacher,
      center=center,
      opt=new_opt,
  )


@jax.jit
def test_step(batch, state):
    batch = prep(batch)
    img = batch['s2']
    pred, features = model(state.teacher, img, return_features=True)
    return jax.nn.softmax(features, axis=-1)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog="Permafrost SSL Training script")
  parser.add_argument('config', type=Path)
  parser.add_argument('-s', '--seed', type=int, required=True)
  parser.add_argument('-n', '--name', type=str, required=True)
  parser.add_argument('-f', '--skip-git-check', action='store_true')
  parser.add_argument('-w', '--unlabelled_weight', type=float)
  parser.add_argument('-t', '--dino_temperature', type=float)
  args = parser.parse_args()

  train_key = jax.random.PRNGKey(args.seed)
  persistent_val_key = jax.random.PRNGKey(27)

  config.update(munchify(yaml.safe_load(args.config.open())))
  if args.dino_temperature is not None:
    config.train.temperature = args.dino_temperature
  if args.unlabelled_weight is not None:
    config.train.unlabelled_weight = args.unlabelled_weight

  # initialize data loading
  train_key, subkey = jax.random.split(train_key)

  datasets = get_datasets(config['datasets'])
  val_data = {k: datasets[k] for k in datasets if k.startswith('val')}
  unlabelled_data = get_unlabelled(config['train']['unlabelled_bs'])

  S, params = utils.get_model(np.ones([1, 128, 128, 12]))

  # Initialize model and optimizer state
  opt_init, _ = get_optimizer()
  model = S.apply

  center = jnp.zeros([1, 1, 1, config.train.n_pseudoclasses])
  state = DINOState(params=params, teacher=params, opt=opt_init(params), center=center)

  wandb.init(project=f'PixelDINO Unsupervised', config=config, name=args.name, group=config.train.group)

  run_dir = Path(f'runs/{wandb.run.id}/')
  assert not run_dir.exists(), f"Previous run exists at {run_dir}"

  run_dir.mkdir(parents=True)
  config.run_id = wandb.run.id
  with open(run_dir / 'config.yml', 'w') as f:
      f.write(yaml.dump(config, default_flow_style=False))

  unlabelled_gen = iter(unlabelled_data)
  trn_metrics = defaultdict(list)
  for step in tqdm(range(1, 1+config.train.steps), ncols=80):
    unlabelled = next(unlabelled_gen)
    unlabelled = {'img': unlabelled['img']}

    train_key, subkey = jax.random.split(train_key)
    terms, state = train_step(unlabelled, state, subkey)

    for k in terms:
        trn_metrics[k].append(terms[k])

    # """
    # Metrics logging and Validation
    # """
    if step % config.validation.frequency != 0:
      continue

    logging.log_metrics(trn_metrics, 'trn', step, do_print=False)
    trn_metrics = defaultdict(list)

    if step not in config.validation.image_steps:
      continue

    for tag, dataset in val_data.items():
      print(f'Validating on {tag}')
      # Validate
      val_key = persistent_val_key
      val_outputs = defaultdict(list)
      for step_test, data in enumerate(dataset):
          val_key, subkey = jax.random.split(val_key, 2)
          data_inp = {'s2': data['s2']}
          pseudo_classes = jax.device_put(test_step(data_inp, state), jax.devices('cpu')[0])

          for i in range(pseudo_classes.shape[0]):
            key = data['source'][i].decode('utf8')
            val_outputs[key].append({
              'pseudo_classes': pseudo_classes[i],
              **jax.tree_map(lambda x: x[i], data),
            })

      # Save Checkpoint
      save_state(state, run_dir / f'step_{step:07d}.pkl')
      save_state(state, run_dir / f'latest.pkl')

      for tile, data in val_outputs.items():
        name = Path(tile).stem
        y_max = max(d['box'][3] for d in data)
        x_max = max(d['box'][2] for d in data)

        weight = np.zeros([y_max, x_max, 1], dtype=np.float64)
        pseudo_classes = np.zeros([y_max, x_max, config.train.n_pseudoclasses])
        window = np.concatenate([
          np.linspace(0, 1, 96),
          np.linspace(0, 1, 96)[::-1],
        ]).reshape(-1, 1)
        stencil = (window * window.T).reshape(192, 192, 1)

        for patch in data:
          x0, y0, x1, y1 = patch['box']
          weight[y0:y1, x0:x1] += stencil
          pseudo_classes[y0:y1, x0:x1] += stencil * patch['pseudo_classes']

        weight = np.where(weight == 0, 1, weight)
        pseudo_classes = (pseudo_classes / weight).argmax(axis=-1)

        cmap = mpl.colormaps['hsv'].resampled(config.train.n_pseudoclasses)
        colors = np.stack([np.asarray(cmap(i))[:3] for i in range(config.train.n_pseudoclasses)])
        pc_rgb = colors[pseudo_classes]

        wandb.log({f'pseudo_class/{name}': wandb.Image(pc_rgb)}, step=step)
