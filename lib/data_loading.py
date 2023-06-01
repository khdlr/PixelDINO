# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds


@tf.function
def prepare(sample):
  s2 = tf.concat([
    sample['156'][..., :1],
    sample['rgb'][..., ::-1],
    sample['156'][..., 1:],
    sample['788a'],
    sample['101112']
  ], axis=-1)

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
def split(sample):
  y, x = sample['box'][..., 0], sample['box'][..., 1]
  offsets = [(  0,   0), (  0, 96), (  0, 192),
             ( 96,   0), ( 96, 96), ( 96, 192),
             (192,   0), (192, 96), (192, 192),]
  out = dict(
    region = tf.stack([sample['region']] * 9),
    source = tf.stack([sample['source']] * 9),
    date   = tf.stack([sample['date']] * 9),
    box = tf.stack([tf.stack([y+dy, x+dx, y+dy+192, x+dx+192], axis=-1) for dy, dx in offsets]),
    s2 = tf.stack([sample['s2'][..., dy:dy+192, dx:dx+192, :] for dy, dx in offsets]),
    mask = tf.stack([sample['mask'][..., dy:dy+192, dx:dx+192, :] for dy, dx in offsets]),
  )
  return tf.data.Dataset.from_tensor_slices(out)


def get_datasets(config_ds):
  data = tfds.load('rts', shuffle_files=True)

  datasets = {}
  for key in config_ds:
    conf = config_ds[key]
    bs = conf['batch_size']
    ds = data[conf['split']]
    if key != 'val':
      ds = ds.repeat()
      ds = ds.shuffle(500)
    ds = ds.batch(bs)
    ds = ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    if key != 'val':
      ds = ds.interleave(split)
      ds = ds.unbatch()
      ds = ds.shuffle(1024)
      ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    datasets[key] = tfds.as_numpy(ds)
  return datasets

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
