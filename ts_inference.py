"""permafrost dataset."""

import yaml
from itertools import zip_longest
import jax
import jax.numpy as jnp
from munch import munchify
import rasterio as rio
import argparse
from pathlib import Path
import math
import numpy as np
from scipy.ndimage import zoom
from scipy.special import expit
from einops import rearrange
from tqdm import tqdm
from multiprocessing import Pool

from lib.config_mod import config
from lib.utils import prep, get_model, load_state

bands = ['B02', 'B03', 'B04', 'B08']


@jax.jit
def normalize(ary):
  'DynamicEarth-Style Normalization'
  ary = 2.5 * (jnp.log(1 + 0.005 * ary) - 2.1)
  ary = jax.nn.sigmoid(ary)
  ary = jnp.clip(255 * ary, 0, 255)
  ary = jnp.nan_to_num(ary).astype(jnp.uint8)
  return ary


def load_data(tile):
  """Yields examples."""
  cache_path = tile.with_suffix('.npy')
  if cache_path.exists():
    return cache_path
  print(f'Starting processing {tile.stem}')
  platform, instrument, acquisition_date, *_ = tile.stem.split('_')
  year = int(acquisition_date[:4])
  
  scene = np.zeros([10980, 10980, 4], dtype=np.uint8)
  was_loaded = [False for _ in bands] 

  for i, band in enumerate(bands):
    data = rio.open(tile / f'{band}.tif').read(1)
    print(f'Loading Band[{i}] {band}')
    scale_factor = scene.shape[0] // data.shape[0] 
    if (year >= 2022):
      # Subtract 1000 for data from 2022 onwards (new processing baseline)
      data = data - 1000

    assert scale_factor == 1
    # if scale_factor != 1:
    # data = zoom(data, zoom=scale_factor, grid_mode=True, order=1, mode='nearest')

    scene[:, :, i] = data
    was_loaded[i] = True

  assert all(was_loaded), 'Not all channels were loaded: {was_loaded}'

  return scene


def flow_from_cache(npy):
  tile_size = 384
  H, W, C = npy.shape
  y_range = np.linspace(0, H-tile_size, 1 + math.ceil((H-tile_size) / 256)).astype(int)
  x_range = np.linspace(0, W-tile_size, 1 + math.ceil((W-tile_size) / 256)).astype(int)

  for y0 in y_range:
    for x0 in x_range:
      y1 = y0 + tile_size
      x1 = x0 + tile_size

      tile = npy[y0:y1, x0:x1]
      box = np.asarray([y0, x0, y1, x1], dtype=np.int32)

      yield {
        'img': tile,
        'box': box,
      }


def chunks(iterable, n, *, fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


@jax.jit
def inference_step(acc, batch):
  prepped = prep({'img': normalize(batch['img'])})
  preds = jax.nn.sigmoid(model(state.params, prepped['img']))
  # Sanity check:
  # preds = prepped['img'][..., 3:4]

  window = jnp.concatenate([
    jnp.linspace(0, 1, 192),
    jnp.linspace(0, 1, 192)[::-1],
  ]).reshape(-1, 1)
  stencil = (window * window.T).reshape(1, 384, 384, 1)
  y, x = jnp.meshgrid(jnp.arange(384), jnp.arange(384))
  y, x = y[jnp.newaxis], x[jnp.newaxis]

  x0, y0, x1, y1 = batch['box'].T
  y0 = y0[:, np.newaxis, np.newaxis]
  x0 = x0[:, np.newaxis, np.newaxis]

  acc['pred'] = acc['pred'].at[y0 + y, x0 + x].add(stencil * preds)
  acc['weight'] = acc['weight'].at[y0 + y, x0 + x].add(stencil)

  return acc


def inference_for_tile(tile, data):
  BS = 43

  # No idea why this is needed... :
  data = rearrange(data, 'W H C -> H W C')
  imgs = flow_from_cache(data) # , total=86*86)
  batches = chunks(imgs, BS)

  acc = {
    'weight': jnp.zeros([10980, 10980, 1], dtype=np.float32),
    'pred': jnp.zeros([10980, 10980, 1], dtype=np.float32),
  }

  for batch in batches:
    batch = [b for b in batch if b is not None]
    batch = {k: jnp.stack([b[k] for b in batch]) for k in batch[0]}
    acc = inference_step(acc, batch)

  weight = acc['weight']
  pred = acc['pred']
  print('pred', pred.min(), pred.max())

  weight = np.where(weight == 0, 1, weight)
  pred = (pred / weight).astype(np.float32)

  with rio.open(tile / 'B04.tif') as src:
    profile = {**src.profile}

  # profile['nodata'] = 255
  # profile['dtype'] = np.uint8
  # with rio.open(f'peel/masks/{tile.stem}.tif', 'w', **profile) as out:
  #   out.write(pred[..., 0] > 0.5, 1)

  profile['nodata'] = 255
  profile['dtype'] = np.uint8
  with rio.open(f'peel/results_u8/{tile.stem}_{args.run.stem}.tif', 'w', **profile) as out:
    pred_u8 = np.clip(254 * pred[..., 0], 0, 254)
    out.write(pred_u8, 1)

  # profile['nodata'] = -1.
  # profile['dtype'] = np.float32
  # with rio.open(f'peel/results/{tile.stem}.tif', 'w', **profile) as out:
  #   out.write(pred[..., 0], 1)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog="Permafrost TimeSeries Inference Script")
  parser.add_argument('run', type=Path)
  args = parser.parse_args()
  config.update(munchify(yaml.safe_load((args.run / 'config.yml').open())))

  tiles = [p for p in Path('peel/s2').glob('*/') if p.is_dir()]
  # tiles = [p for p in tiles if '20210712' in p.stem]

  S, _ = get_model(np.ones([1, 64, 64, 4]))
  model = S.apply
  state = load_state(args.run / 'latest.pkl')
  for tile in tiles:
    data = load_data(tile)
    # mmaps = [np.load(npy, mmap_mode='r') for npy in npys]

    out = inference_for_tile(tile, data)
