import xarray as xr
import rioxarray
import ee
import geedim as gd
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from .base import TileSource, Scene, cache_path, safe_download


class Sentinel2(TileSource):
    def __init__(self, s2sceneid: str):
        self.s2sceneid = s2sceneid

    def get_raster_data(self, scene: Scene) -> xr.Dataset:
        _cache_path = cache_path('Sentinel2', f'{scene.id}.tif')
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not _cache_path.exists():
            utm_zone = s2sceneid.split('_')[-1][1:3]
            crs = f'EPSG:326{utm_zone}'
            Sentinel2.download_tile(_cache_path, self.s2sceneid, scene.ee_bounds(crs), crs=crs)

        data = rioxarray.open_rasterio(_cache_path, chunks=True)
        data = data.isel(band=[0,1,2,3,4,5,6,7,8,9,10,11,12])
        # We could transfer the band names as coordinate labels here
        # But other tools don't seem to be compatible with that (i.e. QGIS)
        # data = data.assign_coords({'band': list(data.long_name[:13])})

        data.attrs['date'] = str(pd.to_datetime(scene.id.split('_')[-3]))
        return data

    @staticmethod
    def download_tile(out_path, s2sceneid, bounds=None):
        if not out_path.exists():
            gd.Initialize()
            img = ee.Image(s2sceneid)
            img = img.select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'])
            if bounds is None:
              # Multiply operation deletes the footprint for some reason
              bounds = gd.MaskedImage(img).footprint
            img = gd.MaskedImage(img)
            safe_download(img, out_path,
                region=bounds,
                scale=10,
                dtype='uint16',
                max_tile_size=2,
                max_tile_dim=2000,
            )

    @staticmethod
    def build_scene(bounds, crs, start_date, end_date, prefix, min_coverage=90, max_cloudy_pixels=20):
        gd.Initialize()
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        s2 = gd.MaskedCollection(s2)

        bounds = ee.Geometry.Polygon(
            list(bounds.exterior.coords),
            proj=None if crs is None else str(crs),
            evenOdd=False)

        imgs = s2.search(
          start_date=start_date,
          end_date=end_date,
          region=bounds,
          fill_portion=min_coverage,
          custom_filter=f'CLOUDY_PIXEL_PERCENTAGE < {max_cloudy_pixels}'
        )

        metadata = imgs.properties
        if len(metadata) == 0:
          return None
        best = max(metadata, key=lambda x: metadata[x]['CLOUDLESS_PORTION'])
        assert metadata[best]['FILL_PORTION'] > 90

        s2_id = best.split('/')[-1]

        scene_id = f'{prefix}_{s2_id}'
        _cache_path = cache_path('Sentinel2', f'{scene_id}.tif')
        Sentinel2.download_tile(_cache_path, best, bounds)
        ds = rioxarray.open_rasterio(_cache_path)

        scene = Scene(
            id=scene_id,
            crs=ds.rio.crs,
            transform=ds.rio.transform(),
            size=ds.shape[-2:],
            layers=[Sentinel2(s2_id)])
        return scene

    @staticmethod
    def build_scenes(bounds, crs, start_date, end_date, prefix, min_coverage=90, max_cloudy_pixels=20):
        gd.Initialize()
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        s2 = gd.MaskedCollection(s2)

        bounds = ee.Geometry.Polygon(
            list(bounds.exterior.coords),
            proj=None if crs is None else str(crs),
            evenOdd=False)

        imgs = s2.search(
          start_date=start_date,
          end_date=end_date,
          region=bounds,
          fill_portion=min_coverage,
          custom_filter=f'CLOUDY_PIXEL_PERCENTAGE < {max_cloudy_pixels}'
        )

        scenes = []

        props = imgs.properties
        old_len = len(props)
        if len(props) > 4:
          sort = sorted(props, key=lambda k: props[k]['CLOUDLESS_PORTION'] +
                         props[k]['FILL_PORTION'], reverse=True)

          dates = set()
          props = []
          for p in sort:
            date = p.split('/')[2][:8]
            if date in dates:
              continue
            dates.add(date)
            props.append(p)
            if len(props) >= 4:
              break

        print(f'Found {old_len} scenes, proceeding with {len(props)} scenes...')


        for img in tqdm(props):
            s2_id = img.split('/')[-1]
            scene_id = f'{prefix}_{s2_id}'
            _cache_path = cache_path('Sentinel2', f'{scene_id}.tif')
            Sentinel2.download_tile(_cache_path, img, bounds)
            ds = rioxarray.open_rasterio(_cache_path)

            scene = Scene(
                id=scene_id,
                crs=ds.rio.crs,
                transform=ds.rio.transform(),
                size=ds.shape[-2:],
                layers=[Sentinel2(s2_id)])
            scenes.append(scene)
        return scenes

    @staticmethod
    def scene_for_tile(tile_id, start_date, end_date,
                       max_cloudy_pixels=20):
        gd.Initialize()
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        s2 = s2.filterMetadata('MGRS_TILE', 'equals', tile_id)
        s2 = s2.filterDate(start_date, end_date)

        n_found = s2.size().getInfo()
        if n_found == 0:
          return None

        best = s2.sort('CLOUDY_PIXEL_PERCENTAGE').first()
        best = best.get('system:id').getInfo()

        s2_id = best.split('/')[-1]
        scene_id = f'{s2_id}'
        _cache_path = cache_path('Sentinel2', f'{scene_id}.tif')
        Sentinel2.download_tile(_cache_path, best)
        ds = rioxarray.open_rasterio(_cache_path)

        scene = Scene(
            id=scene_id,
            crs=ds.rio.crs,
            transform=ds.rio.transform(),
            size=ds.shape[-2:],
            layers=[Sentinel2(s2_id)])
        return scene

    @staticmethod
    def scenes_for_tile(tile_id, start_date, end_date, max_cloudy_pixels=20):
        gd.Initialize()
        s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        s2 = s2.filterMetadata('MGRS_TILE', 'equals', tile_id)
        s2 = gd.MaskedCollection(s2)

        imgs = s2.search(
          start_date=start_date,
          end_date=end_date,
          fill_portion=90,
          custom_filter=f'CLOUDY_PIXEL_PERCENTAGE < {max_cloudy_pixels}'
        )

        metadata = metadata_all = imgs.properties

        if len(metadata) > 4:
          metadata = sorted(metadata, key=lambda k: metadata[k]['CLOUDLESS_PORTION'] + metadata[k]['FILL_PORTION'], reverse=True)[:4]

        print(f'Found {len(metadata_all)} scenes, proceeding with {len(metadata)} scenes...')

        def build_scene(s2_id, _cache_path):
          Sentinel2.download_tile(_cache_path, img)
          ds = rioxarray.open_rasterio(_cache_path)
          return Scene(
            id=s2_id,
            crs=ds.rio.crs,
            transform=ds.rio.transform(),
            size=ds.shape[-2:],
            layers=[Sentinel2(s2_id)])

        with ThreadPoolExecutor(4) as workers:
          futures = []
          for img in metadata:
            s2_id = img.split('/')[-1]
            _cache_path = cache_path('Sentinel2', f'{s2_id}.tif')
            futures.append(workers.submit(build_scene, s2_id, _cache_path))

          scenes = [future.result() for future in futures]
        return scenes

    def __repr__(self):
        return f'Sentinel2({self.s2sceneid})'
