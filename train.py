import yaml
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial
from collections import defaultdict
from torch.utils.data import DataLoader

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
  return getattr(optax, conf.type)(1e-4)


def get_loss_fn(mode):
  name = config.train[f'loss_{mode}']
  return getattr(losses, name)


@partial(jax.jit, static_argnums=4)
def train_step(labelled, unlabelled, state, key, model):
  _, optimizer = get_optimizer()

  aug_key_1, aug_key_2, aug_key_3 = jax.random.split(key, 3)
  batch_l = prep(labelled, augment_key=aug_key_1)
  img_l   = batch_l['Sentinel2']
  mask_l  = batch_l['Mask']
  img_u   = prep(unlabelled, augment_key=aug_key_2)['Sentinel2']
  
  def get_loss(params):
    pred_l = model(params, img_l)
    pred_u = model(params, img_u)

    distorted = distort({'Sentinel2': img_u, 'Mask': pred_u}, aug_key_3)
    img_d  = distorted['Sentinel2']
    mask_d = jax.nn.sigmoid(distorted['Mask'])
    mask_d = jnp.where( mask_d > 0.5,  1,
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
    return terms



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
            for step_test, (batch, meta) in enumerate(data_val):
                val_key, subkey = jax.random.split(val_key)
                metrics = test_step(batch, state, subkey, net)

                for m in metrics:
                  val_metrics[m].append(metrics[m])

            logging.log_metrics(val_metrics, 'val', step)
