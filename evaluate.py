import yaml
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from collections import defaultdict
from PIL import Image, ImageDraw
from skimage import measure
from shapely import Polygon
import geopandas as gpd

import xarray
import rioxarray
from tqdm import tqdm
from munch import munchify
import argparse
from train import test_step

from lib import config, utils
from lib.data_loading import get_loader


if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog="Permafrost SSL Evaluation Script")
  parser.add_argument('run_dir', nargs='+', help='Run directories to evaluate', type=Path)
  args = parser.parse_args()

  config.update(yaml.safe_load((args.run_dir[0] / 'config.yml').open()))
  data_val   = get_loader(config.datasets.val)

  sample_data, sample_meta = next(iter(data_val))
  print('Building model')
  S, params = utils.get_model(jax.tree_map(lambda x: x[:1], sample_data['Sentinel2']))
  net = S.apply

  for run_dir in args.run_dir:
    print(f'Loading Weights for {run_dir.name}')
    state = utils.load_state(run_dir / 'latest.pkl')
    # Validate
    val_key = jax.random.PRNGKey(27)
    val_metrics = defaultdict(list)
    val_outputs = defaultdict(list)

    print('Starting eval for', run_dir.name)
    for step_test, (batch, meta) in enumerate(tqdm(data_val)):
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

    for tile, data in val_outputs.items():
      name = Path(tile).stem
      source = xarray.open_dataset(tile, decode_coords='all')
      tx = np.asarray(source.rio.transform().column_vectors)

      def apply_tx(points):
        x, y = points.T
        o = np.ones_like(y)
        points = np.stack([y, x, o], axis=1)
        return Polygon(points @ tx)

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

      # stacked = np.concatenate([
      #   rgb,
      #   np.concatenate([
      #     mask,
      #     pred,
      #     np.zeros_like(mask)
      #   ], axis=-1),
      # ], axis=1)

      # stacked = Image.fromarray(stacked)

      # mask_img = Image.new("L", (x_max, y_max), 0)
      # mask_draw = ImageDraw.Draw(mask_img)

      # for contour in measure.find_contours(mask[..., 0], 0.5):
      #   mask_draw.polygon([(x,y) for y,x in contour],
      #                     fill=0, outline=255, width=3)

      # pred_img = Image.new("L", (x_max, y_max), 0)
      # pred_draw = ImageDraw.Draw(pred_img)

      contours = measure.find_contours(pred[..., 0], 0.7)
      projected = [apply_tx(points) for points in contours if len(points) >= 4]
      gdf = gpd.GeoDataFrame(geometry=projected, crs=source.rio.crs)
      print(f'Saving to eval/{name}_{run_dir.name}.gpkg')
      gdf.to_file(f'eval/{name}_{run_dir.name}.gpkg')
      # for contour in measure.find_contours(pred[..., 0], 0.7):
      #   pred_draw.polygon([(x,y) for y,x in contour],
      #                     fill=0, outline=255, width=3)
      # mask_img = np.asarray(mask_img)
      # pred_img = np.asarray(pred_img)
      # annot = np.stack([
      #   mask_img,
      #   pred_img,
      #   np.zeros_like(mask_img),
      # ], axis=-1)
      # rgb_with_annot = np.where(np.all(annot == 0, axis=-1, keepdims=True),
      #                           rgb, annot)
      # rgb_with_annot = Image.fromarray(rgb_with_annot)

      # rgb_with_annot.save(f'eval/contour_{name}_{run_dir.name}.jpg')
      # stacked.save(f'eval/mask_{name}_{run_dir.name}.jpg')
