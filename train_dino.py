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
from lib.data_loading import get_datasets
from lib import utils, logging, losses
from lib.config_mod import config
from lib.metrics import compute_premetrics
from lib.utils import TrainingState, prep, distort, changed_state, save_state

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


@partial(jax.jit, static_argnums=3)
def train_step(data, state, key, do_augment=True):
  _, optimizer = get_optimizer()

  key_1a, key_1b, key_2, key_3, key_4 = jax.random.split(key, 5)

  batch = distort(prep(data['train'], key_1a), key_1b)
  img, mask = batch['s2'], batch['mask']

  batch = data['train_unlabelled']
  batch = prep(batch, key_2)
  img_1, img_2 = batch['img_1'], batch['img_2']
  img_2 = distort({'img_2': img_2}, key_3)['img_2']
  imgs = jnp.stack([img, img_1, img_2], axis=0)

  def get_loss(params):
    terms = {}
    preds, features = jax.vmap(lambda x: model(params, x, return_features=True))(imgs)
    pred, _, _ = preds
    _, feat_1, feat_2 = features

    terms['loss_super'] = get_loss_fn('train')(mask, pred)

    # Dino-Style loss: feat_1 == "teacher", feat_2 == "student"
    center = feat_1.mean(axis=[0, 1, 2], keepdims=True)
    feat_1 = (feat_1 - center) / config.train.temperature
    feat_1 = jax.nn.softmax(feat_1, axis=-1)
    feat_1 = distort({'features': feat_1}, key_4)['features']
    feat_1 = jax.lax.stop_gradient(feat_1)

    terms['loss_unlabelled'] = optax.softmax_cross_entropy(feat_2, feat_1).mean()
    terms['loss'] = terms['loss_super'] + \
        config.train.unlabelled_weight * terms['loss_unlabelled']
    terms['super_premetrics'] = compute_premetrics(mask, pred)

    return terms['loss'], terms

  gradients, terms = jax.grad(get_loss, has_aux=True)(state.params)
  updates, new_opt = optimizer(gradients, state.opt, state.params)
  new_params = optax.apply_updates(state.params, updates)

  return terms, changed_state(state,
      params=new_params,
      opt=new_opt,
  )


@jax.jit
def test_step(batch, state):
    batch = prep(batch)
    img = batch['s2']
    mask = batch['mask']

    pred, features = model(state.params, img, return_features=True)
    loss = get_loss_fn('train')(mask, pred)

    terms = {
        'loss': loss,
        'val_premetrics': compute_premetrics(mask, pred),
    }

    return terms, jax.nn.sigmoid(pred), features.argmax(axis=-1)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog="Permafrost SSL Training script")
  parser.add_argument('config', type=Path)
  parser.add_argument('-s', '--seed', type=int, required=True)
  parser.add_argument('-n', '--name', type=str, required=True)
  parser.add_argument('-f', '--skip-git-check', action='store_true')
  args = parser.parse_args()

  run_dir = Path(f'runs/{args.name}/')
  assert not run_dir.exists(), f"Previous run exists at {run_dir}"

  train_key = jax.random.PRNGKey(args.seed)
  persistent_val_key = jax.random.PRNGKey(27)

  config.update(munchify(yaml.safe_load(args.config.open())))

  # initialize data loading
  train_key, subkey = jax.random.split(train_key)

  datasets = get_datasets(config['datasets'])
  val_data = {k: datasets[k] for k in datasets if k.startswith('val')}
  trn_data = {k: datasets[k] for k in datasets if not k.startswith('val')}

  S, params = utils.get_model(np.ones([1, 128, 128, 12]))

  # Initialize model and optimizer state
  opt_init, _ = get_optimizer()
  model = S.apply

  state = TrainingState(params=params, opt=opt_init(params))

  wandb.init(project=f'semi', config=config, name=args.name, group=args.config.stem)

  run_dir.mkdir(parents=True)
  config.run_id = wandb.run.id
  with open(run_dir / 'config.yml', 'w') as f:
      f.write(yaml.dump(config, default_flow_style=False))

  generators = jax.tree_map(iter, trn_data)
  trn_metrics = defaultdict(list)
  for step in tqdm(range(1, 1+config.train.steps), ncols=80):
    data = jax.tree_map(next, generators)
    data = {k: {kk: v for kk, v in data[k].items() if kk in {'s2', 'img_1', 'img_2', 'mask'}} for k in data}

    train_key, subkey = jax.random.split(train_key)
    terms, state = train_step(data, state, subkey, do_augment=config.datasets.train.augment)

    for k in terms:
        trn_metrics[k].append(terms[k])

    # """
    # Metrics logging and Validation
    # """
    if step % config.validation.frequency != 0:
      continue

    logging.log_metrics(trn_metrics, 'trn', step, do_print=False)
    trn_metrics = defaultdict(list)
    # Save Checkpoint
    save_state(state, run_dir / f'step_{step:07d}.pkl')
    save_state(state, run_dir / f'latest.pkl')

    for tag, dataset in val_data.items():
      # Validate
      val_key = persistent_val_key
      val_metrics = defaultdict(list)
      val_outputs = defaultdict(list)
      for step_test, data in enumerate(dataset):
          val_key, subkey = jax.random.split(val_key, 2)
          data_inp = {'s2': data['s2'], 'mask': data['mask']}
          metrics, preds, pseudo_classes = test_step(data_inp, state)

          for m in metrics:
            val_metrics[m].append(metrics[m])

          for i in range(preds.shape[0]):
            key = data['source'][i].decode('utf8')
            val_outputs[key].append({
              'pred': preds[i],
              'pseudo_classes': pseudo_classes[i],
              **jax.tree_map(lambda x: x[i], data),
            })

      logging.log_metrics(val_metrics, tag, step)

      if step % config.validation.image_frequency != 0:
        continue

      for tile, data in val_outputs.items():
        name = Path(tile).stem
        y_max = max(d['box'][3] for d in data)
        x_max = max(d['box'][2] for d in data)

        weight = np.zeros([y_max, x_max, 1], dtype=np.float64)
        rgb    = np.zeros([y_max, x_max, 3], dtype=np.float64)
        mask   = np.zeros([y_max, x_max, 1], dtype=np.float64)
        pred   = np.zeros([y_max, x_max, 1], dtype=np.float64)
        pseudo_classes = np.zeros([y_max, x_max], dtype=int)
        window = np.concatenate([
          np.linspace(0, 1, 96),
          np.linspace(0, 1, 96)[::-1],
        ]).reshape(-1, 1)
        stencil = (window * window.T).reshape(192, 192, 1)

        for patch in data:
          x0, y0, x1, y1 = patch['box']
          patch_rgb  = patch['s2'][:, :, [3,2,1]]
          patch_rgb  = np.clip(patch_rgb, 0, 255)
          patch_mask = np.where(patch['mask'] == 127, 64, patch['mask'])
          patch_mask = np.clip(patch_mask, 0, 255)
          patch_pred = np.clip(255 * patch['pred'], 0, 255)

          patch_rgb = np.asarray(patch_rgb).astype(np.float64)
          patch_mask = np.asarray(patch_mask).astype(np.float64)
          patch_pred = np.asarray(patch_pred).astype(np.float64)

          weight[y0:y1, x0:x1] += stencil
          rgb[y0:y1, x0:x1]    += stencil * patch_rgb
          mask[y0:y1, x0:x1]   += stencil * patch_mask
          pred[y0:y1, x0:x1]   += stencil * patch_pred
          pseudo_classes[y0:y1, x0:x1] = patch['pseudo_classes']

        weight = np.where(weight == 0, 1, weight)
        rgb  = np.clip(rgb / weight, 0, 255).astype(np.uint8)
        mask = np.clip(mask / weight, 0, 255).astype(np.uint8)
        pred = np.clip(pred / weight, 0, 255).astype(np.uint8)

        stacked = np.concatenate([mask, pred, np.zeros_like(mask)], axis=-1)
        stacked = Image.fromarray(stacked)
        _, stacked_jpg = mkstemp('.jpg')
        stacked.save(stacked_jpg)

        @jax.jit
        def mark_edges(mask, threshold):
          mask = (mask > threshold).astype(np.float32)
          if mask.ndim > 2:
            mask = mask[..., 0]
          padded = jnp.pad(mask, 1, mode='edge')
          padded = rearrange(padded, 'H W -> 1 H W 1')
          min_pooled = -hk.max_pool(-padded, 3, 1, 'VALID')
          max_pooled = hk.max_pool(padded, 3, 1, 'VALID')
          is_edge = min_pooled != max_pooled
          is_edge = rearrange(is_edge, '1 H W 1 -> H W')
          return 255 * is_edge.astype(np.uint8)

        mask_img = mark_edges(mask, 0.5)
        pred_img = mark_edges(pred, 0.7)
        annot = np.stack([
          mask_img,
          pred_img,
          np.zeros_like(mask_img),
        ], axis=-1)
        contour_img = np.where(np.all(annot == 0, axis=-1, keepdims=True), rgb, annot)
        contour_img = Image.fromarray(contour_img)
        _, contour_jpg = mkstemp('.jpg')
        contour_img.save(contour_jpg)

        n_classes = config.model.width
        cmap = mpl.colormaps['hsv'].resampled(n_classes)
        colors = np.stack([np.asarray(cmap(i))[:3] for i in range(n_classes)])
        pc_rgb = colors[pseudo_classes]

        wandb.log({f'contour/{name}': wandb.Image(contour_jpg),
                   f'imgs/{name}': wandb.Image(stacked_jpg),
                   f'pseudo_class/{name}': wandb.Image(pc_rgb),
        }, step=step)