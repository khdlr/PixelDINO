from munch import munchify
import yaml

import jax
import numpy as np
from einops import rearrange
from lib.data_loading import get_loader
from lib import utils
from lib import config
from PIL import Image


def rgb(batch, i):
  image = batch['Sentinel2'][i]
  uint8 = np.clip(2 * 255 * image, 0, 255).astype(np.uint8)
  rgb = uint8[:, :, [3,2,1]]
  rgb = np.asarray(rgb)
  return Image.fromarray(rgb)


def mask(batch, i):
  image = batch['Mask'][i]
  uint8 = np.clip(255 * image, 0, 255).astype(np.uint8)
  rgb = np.concatenate([uint8]*3, axis=2)
  rgb = np.asarray(rgb)
  return Image.fromarray(rgb)



if __name__ == '__main__':
    config.update(munchify(yaml.load(open('config.yml'), Loader=yaml.SafeLoader)))
    data_trn   = get_loader(config.datasets.train_labelled)

    @jax.jit
    def augment(batch):
      val_prepped = utils.prep(batch)
      weakly_prepped = utils.prep(batch, jax.random.PRNGKey(129837))
      strongly_prepped = utils.distort(weakly_prepped, jax.random.PRNGKey(87298))
      return val_prepped, weakly_prepped, strongly_prepped

    for B, meta in data_trn:
      V, W, S = augment(B)
      for i in range(B['Sentinel2'].shape[0]):
        ary = V['Sentinel2'][i]
        print(i, ary.min(), ary.mean(), ary.max())
        rgb(V, i).save(f'augs/{i}_raw_rgb.jpg')
        mask(V, i).save(f'augs/{i}_raw_mask.png')
        rgb(W, i).save(f'augs/{i}_weak_rgb.jpg')
        mask(W, i).save(f'augs/{i}_weak_mask.png')
        rgb(S, i).save(f'augs/{i}_strong_rgb.jpg')
        mask(S, i).save(f'augs/{i}_strong_mask.png')

      break


