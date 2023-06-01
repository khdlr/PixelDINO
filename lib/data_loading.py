# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import webdataset as wds
from webdataset.multi import MultiLoader
# import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
from einops import rearrange
from PIL import Image
import re
from pathlib import Path
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

def decode(name_buf):
  name, buf = name_buf
  if name.endswith('.jpg'):
    return name, jpeg.decode(buf.read())
  else:
    return name, np.asarray(Image.open(buf))


def prepare(sample):
  s2 = np.concatenate([
    sample['156.jpg'][..., :1],
    sample['rgb.jpg'][..., ::-1],
    sample['156.jpg'][..., 1:],
    sample['788a.jpg'],
    sample['101112.jpg']
  ], axis=-1)
  
  *scene, x, y = sample['__key__'].split('_')
  scene = '_'.join(scene)
  x = int(x)
  y = int(y)
  
  out = dict(
    scene=scene,
    x=x,
    y=y,
    s2=s2,
    mask=rearrange(sample['mask.png'], 'H W -> H W 1')
  )
  return out


def split(gen):
  for sample in gen:
    scn = sample['scene']
    x, y = sample['x'], sample['y']
    s2 = sample['s2']
    mask = sample['mask']

    for dy in [0, 96, 192]:
      for dx in [0, 96, 192]:
        yield dict(
          scene=scn,
          x = x+dx,
          y = y+dy,
          s2 = s2[dy:dy+192, dx:dx+192],
          mask = mask[dy:dy+192, dx:dx+192],
        )


def numpy_collate(batch):
  """Collate tensors as numpy arrays, taken from
  https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html"""
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple, list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  elif isinstance(batch[0], dict):
    return {k: numpy_collate([sample[k] for sample in batch]) for k in batch[0]}
  else:
    return np.array(batch)


def get_loader_pytorch(config, cycle=False):
  path = Path(config['path'])
  dp = FileLister(str(path.parent), path.name)

  if cycle:
    dp = dp.cycle()
  if config['shuffle']:
    dp = dp.shuffle(buffer_size=100)
  dp = dp.sharding_filter()
  dp = FileOpener(dp, mode='b')
  dp = dp.load_from_tar().map(decode).webdataset()
  dp = dp.map(prepare)
  if config['shuffle']:
    dp = dp.shuffle(buffer_size=500)
  dp = dp.flatmap(split)
  if config['shuffle']:
    dp = dp.shuffle(buffer_size=800)
  dp = dp.batch(config['batch_size'])
  dp = dp.map(numpy_collate)
  # dp = dp.prefetch(8)
  # rs = MultiProcessingReadingService(num_workers=config['threads'])
  # loader = DataLoader2(dp, reading_service=rs)

  return dp


def get_loader(config, cycle=False):
  ds = wds.DataPipeline(
    wds.SimpleShardList(config['path']),
    # cycle,
    # wds.shuffle(100),
    wds.split_by_worker,
    wds.tarfile_to_samples(),
    # wds.shuffle(500),
    wds.decode('rgb'),
    wds.map(prepare),
    split,
    # wds.shuffle(800),
    wds.batched(config['batch_size'], numpy_collate, partial=False),
  )
  return ds


if __name__ == '__main__':
  config = dict(
    batch_size=16,
    threads=0,
    shuffle=True,
    path='/mnt/SSD1/konrad/data/RTS/webds/train-st-{000..002}.tar'
  )

  dataset = get_loader(config, cycle=True)
  for i in tqdm(dataset):
    pass
