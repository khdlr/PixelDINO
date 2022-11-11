import yaml
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial
from collections import defaultdict
from PIL import Image, ImageDraw
from skimage import measure

import wandb
from tqdm import tqdm

import sys
from munch import munchify
from lib.data_loading import get_loader
from lib import utils, logging, models, config, losses
from lib.metrics import compute_premetrics, compute_metrics
from lib.utils import TrainingState, prep, distort, changed_state, save_state

jax.config.update("jax_numpy_rank_promotion", "raise")


def get_optimizer():
  conf = config.optimizer
  return getattr(optax, conf.type)(**conf.args)


def get_loss_fn(mode):
  name = config.train[f'loss_{mode}']
  return getattr(losses, name)


@partial(jax.jit, static_argnums=4)
def train_step(labelled, unlabelled, state, key, model):
  _, optimizer = get_optimizer()

  aug_key_1a, aug_key_1b, aug_key_2, aug_key_3 = jax.random.split(key, 4)
  batch_l = distort(prep(labelled, augment_key=aug_key_1a), aug_key_1b)
  img_l   = batch_l['Sentinel2']
  mask_l  = batch_l['Mask']
  img_u   = prep(unlabelled, augment_key=aug_key_2)['Sentinel2']
  
  def get_loss(params):
    pred_l = model(params, img_l)
    pred_u = model(params, img_u)

    distorted = distort({'Sentinel2': img_u, 'Mask': pred_u}, aug_key_3)
    img_d  = distorted['Sentinel2']
    mask_d = jax.nn.sigmoid(distorted['Mask'])
    mask_d = jnp.where( mask_d > 0.8,  1,
             jnp.where( mask_d < 0.2,  0,
                                      -1))
    mask_d_counts = jnp.bincount(mask_d.reshape(-1) + 1, length=3) / np.prod(mask_d.shape)

    pred_d = model(params, img_d)

    loss_supervised     = get_loss_fn('supervised')(mask_l, pred_l)
    loss_semisupervised = get_loss_fn('semisupervised')(mask_d, pred_d)

    loss = loss_supervised + config.train.semisupervised_weight * loss_semisupervised

    terms = {
        'loss_supervised': loss_supervised,
        'loss_semisupervised': loss_semisupervised,
        'loss': loss,
        'supervised_premetrics': compute_premetrics(mask_l, pred_l),
        'semisupervised_premetrics': compute_premetrics(mask_d, pred_d),
        'pseudolabels_undetermined': mask_d_counts[0],
        'pseudolabels_negative': mask_d_counts[1],
        'pseudolabels_positive': mask_d_counts[2],
    }

    return loss, terms

  gradients, terms = jax.grad(get_loss, has_aux=True)(state.params)
  updates, new_opt = optimizer(gradients, state.opt, state.params)
  new_params = optax.apply_updates(state.params, updates)

  return terms, changed_state(state,
      params=new_params,
      opt=new_opt,
  )


@partial(jax.jit, static_argnums=3)
def test_step(batch, state, key, model):
    batch = prep(batch)
    img = batch['Sentinel2']
    mask = batch['Mask']

    pred = model(state.params, img)
    loss = get_loss_fn('supervised')(mask, pred)

    terms = {
        'loss': loss,
        'val_premetrics': compute_premetrics(mask, pred),
    }

    # Convert from normalized to to pixel coordinates
    return terms, jax.nn.sigmoid(pred)



if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] != '-f':
        utils.assert_git_clean()
    train_key = jax.random.PRNGKey(42)
    persistent_val_key = jax.random.PRNGKey(27)

    config.update(munchify(yaml.load(open('config.yml'), Loader=yaml.SafeLoader)))

    # initialize data loading
    train_key, subkey = jax.random.split(train_key)

    data_trn   = get_loader(config.datasets.train_labelled)
    data_trn_u = get_loader(config.datasets.train_unlabelled)
    data_val   = get_loader(config.datasets.val)
    
    sample_data, sample_meta = next(iter(data_trn))
    S, params = models.get_model(sample_data['Sentinel2'])

    # Initialize model and optimizer state
    opt_init, _ = get_optimizer()
    state = TrainingState(params=params, opt=opt_init(params))
    net = S.apply
    wandb.init(project=f'RTS', config=config)

    run_dir = Path(f'runs/{wandb.run.id}/')
    run_dir.mkdir(parents=True)
    config.run_id = wandb.run.id
    with open(run_dir / 'config.yml', 'w') as f:
        f.write(yaml.dump(config, default_flow_style=False))

    train_gen = iter(data_trn)
    train_gen_u = iter(data_trn_u)

    trn_metrics = defaultdict(list)
    for step in tqdm(range(1, 1+config.train.steps)):
        labelled, meta_labelled = next(train_gen)
        unlabelled, meta_labelled = next(train_gen_u)

        train_key, subkey = jax.random.split(train_key)
        terms, state = train_step(labelled, unlabelled, state, subkey, net)

        for k in terms:
            trn_metrics[k].append(terms[k])

        """
        Metrics logging and Validation
        """
        if step % config.validation.frequency == 0:
            logging.log_metrics(trn_metrics, 'trn', step, do_print=False)
            trn_metrics = defaultdict(list)
            # Save Checkpoint
            save_state(state, run_dir / f'latest.pkl')

            # Validate
            val_key = persistent_val_key
            val_metrics = defaultdict(list)
            val_outputs = defaultdict(list)
            for step_test, (batch, meta) in enumerate(data_val):
                val_key, subkey = jax.random.split(val_key)
                metrics, preds = test_step(batch, state, subkey, net)

                for m in metrics:
                  val_metrics[m].append(metrics[m])

                for i in range(preds.shape[0]):
                  val_outputs[meta['source_file'][i]].append({
                    'Prediction': preds[i],
                    **jax.tree_map(lambda x: x[i], batch),
                    **jax.tree_map(lambda x: x[i], meta),
                  })

            logging.log_metrics(val_metrics, 'val', step)

            if step % config.validation.image_frequency == 0:
              for tile, data in val_outputs.items():
                name = Path(tile).stem
                y_max = max(d['y1'] for d in data)
                x_max = max(d['x1'] for d in data)

                rgb  = np.zeros([y_max, x_max, 3], dtype=np.uint8)
                mask = np.zeros([y_max, x_max, 1], dtype=np.uint8)
                pred = np.zeros([y_max, x_max, 1], dtype=np.uint8)

                for patch in data:
                  y0, x0, y1, x1 = [patch[k] for k in ['y0', 'x0', 'y1', 'x1']]
                  patch_rgb = patch['Sentinel2'][:, :, [3,2,1]]
                  patch_rgb = np.clip(2 * 255 * patch_rgb, 0, 255).astype(np.uint8)
                  patch_mask = np.clip(255 * patch['Mask'], 0, 255).astype(np.uint8)
                  patch_pred = np.clip(255 * patch['Prediction'], 0, 255).astype(np.uint8)

                  rgb[y0:y1, x0:x1]  = patch_rgb
                  mask[y0:y1, x0:x1] = patch_mask
                  pred[y0:y1, x0:x1] = patch_pred

                stacked = np.concatenate([
                  rgb,
                  np.concatenate([
                    mask,
                    pred,
                    np.zeros_like(mask)
                  ], axis=-1),
                ], axis=1)

                stacked = Image.fromarray(stacked)
                stacked_path = Path(wandb.run.dir) / f'imgs_{name}.jpg'
                stacked.save(stacked_path)

                mask_img = Image.new("L", (x_max, y_max), 0)
                mask_draw = ImageDraw.Draw(mask_img)
                for contour in measure.find_contours(mask[..., 0], 0.5):
                  mask_draw.polygon([(x,y) for y,x in contour],
                                    fill=0, outline=255, width=3)

                pred_img = Image.new("L", (x_max, y_max), 0)
                pred_draw = ImageDraw.Draw(pred_img)
                for contour in measure.find_contours(pred[..., 0], 0.5):
                  pred_draw.polygon([(x,y) for y,x in contour],
                                    fill=0, outline=255, width=3)
                mask_img = np.asarray(mask_img)
                pred_img = np.asarray(pred_img)
                annot = np.stack([
                  mask_img,
                  pred_img,
                  np.zeros_like(mask_img),
                ], axis=-1)
                rgb_with_annot = np.where(np.all(annot == 0, axis=-1, keepdims=True),
                                          rgb, annot)
                rgb_with_annot = Image.fromarray(rgb_with_annot)
                annot_path = Path(wandb.run.dir) / f'contour_{name}.jpg'
                rgb_with_annot.save(annot_path)
                wandb.log({f'contour/{name}': wandb.Image(str(annot_path)),
                           f'imgs/{name}': wandb.Image(str(stacked_path)),
                }, step=step)



