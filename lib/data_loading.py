# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import xarray
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from einops import rearrange
from tqdm import tqdm
from pathlib import Path
from skimage.measure import find_contours
from .utils import shuffle_loop, deterministic_loop


class NCDataset(Dataset):
  def __init__(self, netcdf_path, config):
    self.netcdf_path = netcdf_path
    self.tile_size = config.tile_size
    self.datasets = config.datasets
    self.data = xarray.open_dataset(netcdf_path, cache=False)
    self.sampling_mode = config.sampling_mode

    self.H, self.W = len(self.data.y), len(self.data.x)
    self.H_tile = self.H // self.tile_size
    self.W_tile = self.W // self.tile_size

    self.scale = {'Sentinel2': 1/10000}

    if self.sampling_mode == 'targets_only':
      targets = (self.data.Mask == 1).squeeze('mask_band').values
      # Find Bounding boxes of targets
      contours = find_contours(targets)

      self.bboxes = []
      for contour in contours:
        ymin, xmin = np.floor(contour.min(axis=0)).astype(int)
        ymax, xmax = np.ceil(contour.max(axis=0)).astype(int)
        self.bboxes.append([ymin, xmin, ymax, xmax])

  def __getitem__(self, idx):
    if self.sampling_mode == 'deterministic':
      y_tile, x_tile = divmod(idx, self.W_tile)
      y0 = y_tile * self.tile_size
      x0 = x_tile * self.tile_size
    elif self.sampling_mode == 'random':
      y0 = int(torch.randint(0, self.H - self.tile_size, ()))
      x0 = int(torch.randint(0, self.W - self.tile_size, ()))
    elif self.sampling_mode == 'targets_only':
      bbox_idx = int(torch.randint(0, len(self.bboxes), ()))
      ymin, xmin, ymax, xmax = self.bboxes[bbox_idx]

      y_start = max(0, ymin - self.tile_size)
      y_end   = min(self.H - self.tile_size, ymax)

      x_start = max(0, xmin - self.tile_size)
      x_end   = min(self.W - self.tile_size, xmax )

      if y_start >= y_end or x_start >= x_end:
        print("Nasty BBox!")
        print(f'y range: {ymin} -- {ymax}')
        print(f'x range: {xmin} -- {xmax}')
        print('Derived:')
        print(f'Sample y from [{y_start}, {y_end})')
        print(f'Sample x from [{x_start}, {x_end})')
        print(f'Image size: {self.H} x {self.W}')

      y0 = int(torch.randint(y_start, y_end, ()))
      x0 = int(torch.randint(x_start, x_end, ()))
    else:
      raise ValueError(f'Unsupported tiling mode: {self.sampling_mode!r}')
    y1 = y0 + self.tile_size
    x1 = x0 + self.tile_size

    metadata = {
      'source_file': self.netcdf_path,
      'y0': y0, 'x0': x0,
      'y1': y1, 'x1': x1,
    }
    tile = {k: self.data[k][:, y0:y1, x0:x1].fillna(0).values for k in self.datasets}
    tile = {k: rearrange(v, 'C H W -> H W C') for k, v in tile.items()}
    for k in tile:
      if k in self.scale:
        tile[k] = (tile[k] * self.scale[k]).clip(0, 1)
    return tile, metadata

  def __len__(self):
    return self.H_tile * self.W_tile


class TemporalNCDataset(Dataset):
  def __init__(self, netcdf_path, config):
    self.netcdf_path = netcdf_path
    self.tile_size = config.tile_size
    self.datasets = config.datasets
    self.data = xarray.open_dataset(netcdf_path, cache=False)
    self.sampling_mode = config.sampling_mode

    self.T, self.H, self.W = len(self.data.time), len(self.data.y), len(self.data.x)
    self.H_tile = self.H // self.tile_size
    self.W_tile = self.W // self.tile_size

    self.scale = {'Sentinel2': 1/10000}

    if self.sampling_mode == 'targets_only':
      targets = (self.data.Mask == 1).squeeze('mask_band').values
      # Find Bounding boxes of targets
      contours = find_contours(targets)

      self.bboxes = []
      for contour in contours:
        ymin, xmin = np.floor(contour.min(axis=0)).astype(int)
        ymax, xmax = np.ceil(contour.max(axis=0)).astype(int)
        self.bboxes.append([ymin, xmin, ymax, xmax])

  def __getitem__(self, idx):
    if self.sampling_mode == 'deterministic':
      t0, rem    = divmod(idx, self.H_tile * self.W_tile)
      t1 = t0 + 1
      y_tile, x_tile = divmod(idx, self.W_tile)
      y0 = y_tile * self.tile_size
      x0 = x_tile * self.tile_size
    elif self.sampling_mode == 'random':
      t0 = int(torch.randint(0, self.T - 1, ()))
      t1 = t0 + 1
      if torch.randint(0, 2, ()):
        t0, t1 = t1, t0
      y0 = int(torch.randint(0, self.H - self.tile_size, ()))
      x0 = int(torch.randint(0, self.W - self.tile_size, ()))
    else:
      raise ValueError(f'Unsupported tiling mode: {self.sampling_mode!r}')
    y1 = y0 + self.tile_size
    x1 = x0 + self.tile_size

    metadata = {
      'source_file': self.netcdf_path,
      't0': t0, 't1': t1,
      'y0': y0, 'x0': x0,
      'y1': y1, 'x1': x1,
    }
    tile0 = {k: self.data[k][t0, :, y0:y1, x0:x1].fillna(0).values for k in self.datasets}
    tile1 = {k: self.data[k][t1, :, y0:y1, x0:x1].fillna(0).values for k in self.datasets}

    for tile in [tile0, tile1]:
      for k in tile:
        if k in self.scale:
          tile[k] = (tile[k] * self.scale[k]).clip(0, 1)
          tile[k] = rearrange(tile[k], 'C H W -> H W C')

    return tile0, tile1, metadata

  def __len__(self):
    return (self.T - 1) * self.H_tile * self.W_tile


def get_loader(config, dsclass=NCDataset):
  root = config.root
  scene_names = config.scenes
  scenes = [dsclass(f'{root}/{scene}.nc', config) for scene in scene_names]
  all_data = ConcatDataset(scenes)

  if config.sampling_mode == 'deterministic':
    sampler = None
  else:
    sampler = shuffle_loop(len(all_data))

  return DataLoader(all_data,
      sampler=sampler,
      batch_size=config.batch_size,
      num_workers=config.threads,
      collate_fn=numpy_collate,
      pin_memory=True)


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



if __name__ == '__main__':
    from sys import argv
    from einops import rearrange
    from PIL import Image
    from lib.utils.plot_info import grid

    file = Path(argv[1])
    dataset = TimeseriesDataset(file, sampling_mode='deterministic')
    # loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    for i in tqdm(range(len(dataset))):
        img, mask = dataset[i]
        if not (mask == 1).any():
            continue
        img  = img.numpy()
        mask = mask.numpy()

        mask = np.where(mask == 255, np.uint8(127),
               np.where(mask == 1,   np.uint8(255),
                                     np.uint8(  0)))

        print(img.min(), img.max(), end=' -> ')
        img = rearrange(img[[3,2,7]], 'C H W -> H W C')
        img  = np.clip(2 * 255 * img, 0, 255).astype(np.uint8)
        print(img.min(), img.max())
        combined = Image.fromarray(grid([[img, mask]]))
        combined.save(f'img/{file.stem}_{i}.jpg')

