# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds
from einops import rearrange


@tf.function
def prepare(sample):
  s2 = rearrange(sample['img'], '... (col W) C -> ... W (col C)', col=4, C=3)

  out = dict(
    region=sample['region'],
    source=sample['source'],
    date=sample['date'],
    box=sample['box'],
    s2=s2,
    mask=sample['mask'],
  )
  return out


@tf.function
def prepare_unlabelled(sample):
  img = rearrange(sample['img'], '... (col W) C -> ... W (col C)', col=4, C=3)

  out = dict(
    box=sample['box'],
    img=img,
  )
  return out


@tf.function
def split(sample):
  x, y = sample['box'][..., 0], sample['box'][..., 1]
  offsets = [(  0,   0), (  0, 96), (  0, 192),
             ( 96,   0), ( 96, 96), ( 96, 192),
             (192,   0), (192, 96), (192, 192),]
  out = dict(
    region = tf.stack([sample['region']] * 9),
    source = tf.stack([sample['source']] * 9),
    date   = tf.stack([sample['date']] * 9),
    box = tf.stack([tf.stack([x+dx, y+dy, x+dx+192, y+dy+192], axis=-1) for dy, dx in offsets]),
    s2 = tf.stack([sample['s2'][..., dy:dy+192, dx:dx+192, :] for dy, dx in offsets]),
    mask = tf.stack([sample['mask'][..., dy:dy+192, dx:dx+192, :] for dy, dx in offsets]),
  )
  return tf.data.Dataset.from_tensor_slices(out)


def get_datasets(config_ds):
  data = tfds.load('rts', shuffle_files=True)
  if any(config_ds[c]['split'].startswith('unlabelled') for c in config_ds):
    data.update(tfds.load('rts_unlabelled', shuffle_files=True))

  datasets = {}
  for key in config_ds:
    split = config_ds[key]['split']
    conf = config_ds[key]

    bs = conf['batch_size']
    ds = data[split]
    if not key.startswith('val'):
      ds = ds.repeat()
      ds = ds.shuffle(500)
    ds = ds.batch(bs)

    ds = ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    # ds = ds.interleave(split)
    if key != 'val':
      ds = ds.unbatch()
      ds = ds.shuffle(1024)
      ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    datasets[key] = tfds.as_numpy(ds)
  return datasets


def get_unlabelled(batch_size):
  ds = tfds.load('permafrost', shuffle_files=True)['train']
  ds = ds.repeat()
  ds = ds.shuffle(500)
  ds = ds.batch(batch_size)
  ds = ds.map(prepare_unlabelled, num_parallel_calls=tf.data.AUTOTUNE)

  ds = ds.unbatch()
  ds = ds.shuffle(1024)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  ds = tfds.as_numpy(ds)
  return ds


if __name__ == '__main__':
  config = {'unlabelled': {
    'batch_size': 16,
    'shuffle': True,
    'split': 'unlabelled'
  }}

  dataset = get_datasets(config)['unlabelled']
  for i in tqdm(dataset):
    pass
