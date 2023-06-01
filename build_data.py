#!/usr/bin/env python
# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Usecase 2 Data Preprocessing Script
"""
import argparse
import math
from datetime import datetime
from pathlib import Path
import os
import random
from io import BytesIO
import xarray
from PIL import Image
import numpy as np
from tqdm import tqdm
from einops import rearrange

import pandas as pd
import geopandas as gpd
from shapely import unary_union, MultiPolygon
from joblib import Parallel, delayed

from lib import data

parser = argparse.ArgumentParser()
parser.add_argument("--n_jobs", default=-1, type=int, help="number of parallel joblib jobs")
parser.add_argument("--data_dir", default='data', type=Path, help="Path to data processing dir")
parser.add_argument("--log_dir", default='logs', type=Path, help="Path to log dir")
parser.add_argument("--mode", default='s2', choices=['s2', 's2_4w', 's2_ts'], help="The type of data cubes to build.")

def tiles_to_jpg(basepath, data):
  tile_size = 192
  assert 'time' in data.dims
  basepath.mkdir(exist_ok=True)

  H, W = len(data.y), len(data.x)
  y_range = np.linspace(0, H-tile_size, 1 + math.ceil((H-tile_size) / 128)).astype(int)
  x_range = np.linspace(0, W-tile_size, 1 + math.ceil((W-tile_size) / 128)).astype(int)

  for y0 in y_range:
    for x0 in x_range:
      y1 = y0 + tile_size
      x1 = x0 + tile_size

      maskpart = ''
      if 'Mask' in data:
        mask = data.Mask[:, y0:y1, x0:x1]
        if mask.min() >= 250:
          maskpart = '_na'
        elif np.any(mask.values.reshape(-1) == 1):
          maskpart = '_rts'
        else:
          maskpart = '_bg'
      out_base = basepath / f'{x0}_{y0}{maskpart}'

      if 'Mask' in data:
        mask = np.where(mask == 255, 127, 255 * mask)
        mask = np.where(np.isnan(mask), 127, mask)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask[0])
        mask.save(out_base.with_suffix('.mask.png', format='png'))

        for t_i, t in enumerate(data.time):
          ts = t.dt.strftime('%Y-%m-%d').item()
          tile = rearrange(data.Sentinel2[t_i, :, y0:y1, x0:x1].values, 'C H W -> H W C')

          if tile.max() <= 0:
            continue
          # DynamicEarth-Style Normalization
          tile = np.log(tile * 0.005 + 1)
          tile = (tile - 1.7) / 2.0
          tile = np.exp(tile * 5 - 1)
          tile = tile / (tile + 1)

          tile = np.clip(255 * tile, 0, 255).astype(np.uint8)

          def add_img(channels, suffix):
            img = Image.fromarray(tile[..., channels])
            img.save(out_base.with_suffix(f'{suffix}.jpg'))

          add_img([3, 2, 1], 'rgb')
          add_img([0, 4, 5], '156')
          add_img([6, 7, 8], '788a')
          add_img([10, 11, 12], '101112')


def build_sentinel2_4w(site_poly, image_id, image_date, tiles, targets, out_dir: Path):
  out_subdir = out_dir / f'{image_id}'
  # if any(out_subdir.glob('*.tar')):
  #   print(f'Skipping {image_id} as it has already been built')
  #   return
  data.init_data_paths(out_dir.parent)

  start_date = image_date - pd.to_timedelta('21 days')
  end_date = image_date + pd.to_timedelta('21 days')
  scenes = data.Sentinel2.build_scenes(
      bounds=site_poly, crs='EPSG:3995',
      start_date=start_date, end_date=end_date,
      prefix=image_id,
  )
  print(f'Found {len(scenes)} valid image found for {image_id}')
  if not scenes:
    return

  mask = data.Mask(targets.geometry, tiles.geometry)

  xarrs = []
  print('Starting extracting Scenes')
  for scene in tqdm(scenes):
      xarr = scene.to_xarray()
      date = pd.to_datetime(xarr.Sentinel2.date)
      xarr = xarr.expand_dims({'time': [date]}, axis=0)
      xarrs.append(xarr)
  combined = xarray.concat(xarrs, 'time')
  combined['Mask'] = mask.get_raster_data(scenes[0])

  combined['Sentinel2'].rio.write_crs(xarrs[0].rio.crs, inplace=True)
  combined['Sentinel2'].rio.write_transform(xarrs[0].rio.transform(), inplace=True)
  combined['Mask'].rio.write_nodata(255, inplace=True)

  print('Writing...')
  tiles_to_jpg(out_subdir, combined)
  print('Done')


def build_s2_ts(tile_id, out_dir: Path):
  out_subdir = out_dir / f'{tile_id}'
  data.init_data_paths(out_dir.parent)
  if any(out_dir.glob(f'{tile_id}*')):
    print(f'Skipping {tile_id} as it has already been built')
    return
  data.init_data_paths(out_dir.parent)

  random.seed(tile_id)
  year = random.choice([2017, 2018, 2019, 2020, 2021])
  start_date = f'{year}-07-01'
  end_date = f'{year}-10-01'
  scenes = data.Sentinel2.scenes_for_tile(tile_id,
    start_date=start_date, end_date=end_date,
  )

  xarrs = []
  for scene in scenes:
    xarr = scene.to_xarray()
    date = pd.to_datetime(xarr.Sentinel2.date)
    xarr = xarr.expand_dims({'time': [date]}, axis=0)
    xarrs.append(xarr)

  combined = xarray.concat(xarrs, 'time')
  combined['Sentinel2'].encoding.update({
    'chunksizes': (1, 13, 128, 128),
  })
  combined['Sentinel2'].rio.write_crs(xarrs[0].rio.crs, inplace=True)
  combined['Sentinel2'].rio.write_transform(xarrs[0].rio.transform(), inplace=True)

  tiles_to_jpg(out_subdir, combined)
  # combined.to_netcdf(out_dir / f'{tile_id}_{year}.nc', engine='h5netcdf')


def load_annotations(shapefile_root):
    targets = map(gpd.read_file, shapefile_root.glob('*/TrainingLabel*.shp'))
    targets = pd.concat(targets).to_crs('EPSG:3995').reset_index(drop=True)
    targets['geometry'] = targets['geometry'].buffer(0)
    targets['image_date'] = pd.to_datetime(targets['image_date'])
    id2date = dict(targets[['image_id', 'image_date']].values)

    scenes = map(gpd.read_file, shapefile_root.glob('*/ImageFootprints*.shp'))
    scenes = pd.concat(scenes).to_crs('EPSG:3995').reset_index(drop=True)
    scenes = scenes[scenes.image_id.isin(targets.image_id)]
    scenes['geometry'] = scenes['geometry'].buffer(0)
    scenes['image_date'] = scenes.image_id.apply(id2date.get)

    # Semijoin targets and sites:
    targets = targets[targets.image_id.isin(scenes.image_id)]

    sites = unary_union(scenes.geometry)
    sites = list(sites.geoms)

    merged_scenes = []
    merged_targets = []
    for date, date_scenes in scenes.groupby('image_date'):
      union = unary_union(date_scenes.geometry)
      if type(union) is MultiPolygon:
        geoms = list(union.geoms)
      else:
        geoms = [union]
      for geom in geoms:
        tgts = targets[targets.intersects(geom) & targets.image_id.isin(date_scenes.image_id)].copy()
        regions = list(tgts.region.unique())
        assert len(regions) == 1, f"Merging data from multiple regions: {regions}"
        region = regions[0]
        if region == 'Lena Delta':
          region = 'Lena'
        region = region.replace(' ', '_')

        representatives = date_scenes[date_scenes.intersects(geom)]
        image_id = representatives.iloc[0].image_id
        image_id = region + '--' + image_id + '_merged'
        scene = representatives.iloc[:1].copy()
        scene['image_id'] = image_id
        scene['geometry'] = [geom]
        tgts['image_id'] = image_id

        merged_targets.append(tgts)
        merged_scenes.append(scene)

    merged_scenes = pd.concat(merged_scenes)
    merged_targets = pd.concat(merged_targets)

    return targets, scenes, sites, merged_scenes, merged_targets


def run_jobs(function, n_jobs, out_dir, args_list):
  if n_jobs == 0:
    for args in tqdm(args_list):
      function(*args, out_dir)
  else:
    Parallel(n_jobs=n_jobs)(delayed(function)(*args, out_dir) for args in args_list)


if __name__ == "__main__":
    args = parser.parse_args()

    global DATA_ROOT, INPUT_DATA_DIR, BACKUP_DIR, DATA_DIR, AUX_DIR

    DATA_ROOT = Path(args.data_dir)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = Path(args.log_dir) / f'build_data_cubes-{timestamp}.log'
    if not Path(args.log_dir).exists():
        os.mkdir(Path(args.log_dir))

    shapefile_root = DATA_ROOT / 'ML_training_labels' / 'retrogressive_thaw_slumps'
    targets, scenes, sites, merged_scenes, merged_targets = load_annotations(shapefile_root)

    if args.mode == 's2':
        out_dir = DATA_ROOT / 's2_cubes'; out_dir.mkdir(exist_ok=True)
        scene_infos = []
        for _, row in scenes.iterrows():
          scene_infos.append((row.geometry, row.image_id, row.image_date,
            scenes[scenes.image_id == row.image_id], # scenes
            targets[targets.image_id == row.image_id], # targets
          ))
        run_jobs(build_sentinel2_cubes, args.n_jobs, out_dir, scene_infos)
    elif args.mode == 's2_4w':
      out_dir = DATA_ROOT / 's2_4w'; out_dir.mkdir(exist_ok=True)
      scene_infos = []
      for _, row in merged_scenes.iterrows():
        scene_infos.append((row.geometry, row.image_id, row.image_date,
          merged_scenes[merged_scenes.image_id == row.image_id], # scenes
          merged_targets[merged_targets.image_id == row.image_id], # targets
        ))
      run_jobs(build_sentinel2_4w, args.n_jobs, out_dir, scene_infos)
    elif args.mode == 's2_ts':
      out_dir = DATA_ROOT / 's2u_jpg'
      out_dir.mkdir(exist_ok=True)

      tiles = gpd.read_file(DATA_ROOT / 'active_tiles.geojson')
      site_info = [(x,) for x in tiles.Name]
      run_jobs(build_s2_ts, args.n_jobs, out_dir, site_info)
