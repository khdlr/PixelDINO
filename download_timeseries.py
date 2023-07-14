import pystac_client
import planetary_computer
import pandas as pd
from IPython.terminal.embed import embed
import numpy as np
from scipy.special import expit


def normalize(ary):
  'DynamicEarth-Style Normalization'
  ary = 0.005 * ary
  np.add(ary, 1., out=ary)
  np.log(ary, out=ary)
  np.add(ary, -2.1, out=ary)
  np.multiply(ary, 2.5, out=ary)
  expit(ary, out=ary)
  ary = np.clip(255 * ary, 0, 255)
  ary = np.nan_to_num(ary).astype(np.uint8)
  return ary


catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(collections=["sentinel-2-l2a"], query={
  "s2:mgrs_tile": dict(eq="08WMA"),
  "eo:cloud_cover": {"lt": 20},
})
items = search.item_collection()

df = pd.DataFrame([res.properties for res in items])
df['item'] = items
df['useful_frac'] = (1 - df['s2:nodata_pixel_percentage']/100) * \
                    (1 - df['s2:high_proba_clouds_percentage']/100) * \
                    (1 - df['s2:snow_ice_percentage']/100)
df = df[df['useful_frac'] > 0.8].sort_values('useful_frac', ascending=False)

bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
         'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

with open('urls.aria', 'w') as f:
  for item in df.item:
    out_root = f's2_peel/{item.id}'
    for band in bands:
      out_name = f'{out_root}/{band}.tif'
      print(item.assets[band].href, file=f)
      print(f'\tout={out_name}', file=f)

with open('rgb.aria', 'w') as f:
  for item in df.item:
    out_name = f'peel/s2rgb/{item.id}.tif'
    print(item.assets['visual'].href, file=f)
    print(f'\tout={out_name}', file=f)
