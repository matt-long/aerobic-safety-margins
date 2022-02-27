import os
import time
from collections.abc import Iterable

import cftime
import dask
import intake
import numpy as np
import xarray as xr
import yaml
from dask.distributed import Client
from dask_jobqueue import PBSCluster

path_to_here = os.path.dirname(os.path.realpath(__file__))

USER = os.environ['USER']
PBS_PROJECT = 'NCGD0011'


def attrs_label(attrs):
    """generate a label from long_name and units"""
    da_name = ''
    if isinstance(attrs, xr.DataArray):
        da_name = attrs.name
        attrs = attrs.attrs
    name = da_name if 'long_name' not in attrs else attrs['long_name']

    if len(name) > 30:
        name = '\n'.join([name[:30], name[30:]])
    units = '' if 'units' not in attrs else f' [{attrs["units"]}]'
    return name + units


def label_plots(fig, axs, xoff=-0.04, yoff=0.02):
    alp = [chr(i).upper() for i in range(97, 97 + 26)]
    for i, ax in enumerate(axs):
        p = ax.get_position()
        x = p.x0 + xoff
        y = p.y1 + yoff
        fig.text(x, y, f'{alp[i]}', fontsize=14, fontweight='semibold')


def get_ClusterClient(memory='25GB'):
    """get cluster and client"""
    cluster = PBSCluster(
        cores=1,
        memory=memory,
        processes=1,
        queue='casper',
        local_directory=f'/glade/scratch/{USER}/dask-workers',
        log_directory=f'/glade/scratch/{USER}/dask-workers',
        resource_spec=f'select=1:ncpus=1:mem={memory}',
        project=PBS_PROJECT,
        walltime='06:00:00',
        interface='ib0',
    )

    jupyterhub_server_name = os.environ.get('JUPYTERHUB_SERVER_NAME', None)
    dashboard_link = 'https://jupyterhub.hpc.ucar.edu/stable/user/{USER}/proxy/{port}/status'
    if jupyterhub_server_name:
        dashboard_link = (
            'https://jupyterhub.hpc.ucar.edu/stable/user/'
            + '{USER}'
            + f'/{jupyterhub_server_name}/proxy/'
            + '{port}/status'
        )
    dask.config.set({'distributed.dashboard.link': dashboard_link})
    client = Client(cluster)
    return cluster, client


class timer(object):
    """support reporting timing info with named tasks"""

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tic = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(f'[{self.name}]: ', end='')
        toc = time.time() - self.tic
        print(f'{toc:0.5f}s')


def to_datenum(y, m, d, time_units='days since 0001-01-01 00:00:00'):
    """convert year, month, day to number"""
    return cftime.date2num(cftime.datetime(y, m, d), units=time_units)


def nday_per_year(year):
    return 365


def year_frac(time):
    """compute year fraction"""

    year = [d.year for d in time.values]
    month = [d.month for d in time.values]
    day = [d.day for d in time.values]

    t0_year = np.array([to_datenum(y, 1, 1) - 1 for y in year])
    t_year = np.array([to_datenum(y, m, d) for y, m, d in zip(year, month, day)])
    nday_year = np.array([nday_per_year(y) for y in year])

    return year + (t_year - t0_year) / nday_year


def pop_add_cyclic(ds):
    """Make POP grid easily plottable"""
    ni = ds.TLONG.shape[1]

    xL = int(ni / 2 - 1)
    xR = int(xL + ni)

    tlon = ds.TLONG.data
    tlat = ds.TLAT.data

    tlon = np.where(np.greater_equal(tlon, min(tlon[:, 0])), tlon - 360.0, tlon)
    lon = np.concatenate((tlon, tlon + 360.0), 1)
    lon = lon[:, xL:xR]

    if ni == 320:
        lon[367:-3, 0] = lon[367:-3, 0] + 360.0
    lon = lon - 360.0

    lon = np.hstack((lon, lon[:, 0:1] + 360.0))
    if ni == 320:
        lon[367:, -1] = lon[367:, -1] - 360.0

    # -- trick cartopy into doing the right thing:
    #   it gets confused when the cyclic coords are identical
    lon[:, 0] = lon[:, 0] - 1e-8

    # -- periodicity
    lat = np.concatenate((tlat, tlat), 1)
    lat = lat[:, xL:xR]
    lat = np.hstack((lat, lat[:, 0:1]))

    TLAT = xr.DataArray(lat, dims=('nlat', 'nlon'))
    TLONG = xr.DataArray(lon, dims=('nlat', 'nlon'))

    dso = xr.Dataset({'TLAT': TLAT, 'TLONG': TLONG})

    # copy vars
    varlist = [v for v in ds.data_vars if v not in ['TLAT', 'TLONG']]
    for v in varlist:
        v_dims = ds[v].dims
        if not ('nlat' in v_dims and 'nlon' in v_dims):
            dso[v] = ds[v]
        else:
            # determine and sort other dimensions
            other_dims = set(v_dims) - {'nlat', 'nlon'}
            other_dims = tuple([d for d in v_dims if d in other_dims])
            lon_dim = ds[v].dims.index('nlon')
            field = ds[v].data
            field = np.concatenate((field, field), lon_dim)
            field = field[..., :, xL:xR]
            field = np.concatenate((field, field[..., :, 0:1]), lon_dim)
            dso[v] = xr.DataArray(field, dims=other_dims + ('nlat', 'nlon'), attrs=ds[v].attrs)

    # copy coords
    for v, da in ds.coords.items():
        if not ('nlat' in da.dims and 'nlon' in da.dims):
            dso = dso.assign_coords(**{v: da})

    return dso


class curator_local_assets(object):
    """Curate an intake catalog with locally-cached assets"""

    def __init__(self):

        cache_dir = 'data/cache'
        os.makedirs(cache_dir, exist_ok=True)

        self.catalog_file = f'{path_to_here}/data/catalogs/catalog-local.yml'
        if os.path.exists(self.catalog_file):
            with open(self.catalog_file, 'r') as fid:
                self.catalog = yaml.safe_load(fid)
        else:
            self.catalog = yaml.safe_load(
                """
                description: Local assets

                plugins:
                  source:
                    - module: intake_xarray

                sources: {}
                """
            )

    def add_source(self, key, urlpath, description, driver='netcdf', overwrite=False, **kwargs):
        """add a new source to the catalog"""

        if key in self.catalog['sources']:
            if not overwrite:
                raise ValueError(f'source {key} exists; set `overwrite` to true to overwrite')
            else:
                print(f'overwriting "{key}" key in "sources"')

        args = dict(urlpath=urlpath)
        args.update(kwargs)

        self.catalog['sources'][key] = dict(
            driver=driver,
            description=description,
            args=args,
        )
        self.persist()

    def persist(self):
        """write the catalog to disk"""
        with open(self.catalog_file, 'w') as fid:
            yaml.dump(self.catalog, fid)

    def open_catalog(self):
        """return as intake catalog"""
        return intake.open_catalog(self.catalog_file)

    def __repr__(self):
        return self.catalog.__repr__()


def infer_lat_name(ds):
    lat_names = ['latitude', 'lat']
    for n in lat_names:
        if n in ds:
            return n
    raise ValueError('could not determine lat name')


def infer_lon_name(ds):
    lon_names = ['longitude', 'lon']
    for n in lon_names:
        if n in ds:
            return n
    raise ValueError('could not determine lon name')


def lat_weights_regular_grid(lat):
    """
    Generate latitude weights for equally spaced (regular) global grids.
    Weights are computed as sin(lat+dlat/2)-sin(lat-dlat/2) and sum to 2.0.
    """
    dlat = np.abs(np.diff(lat))
    np.testing.assert_almost_equal(dlat, dlat[0])
    w = np.abs(np.sin(np.radians(lat + dlat[0] / 2.0)) - np.sin(np.radians(lat - dlat[0] / 2.0)))

    if np.abs(lat[0]) > 89.9999:
        w[0] = np.abs(1.0 - np.sin(np.radians(np.pi / 2 - dlat[0])))

    if np.abs(lat[-1]) > 89.9999:
        w[-1] = np.abs(1.0 - np.sin(np.radians(np.pi / 2 - dlat[0])))

    return w


def compute_grid_area(ds, check_total=True):
    """Compute the area of grid cells.

    Parameters
    ----------

    ds : xarray.Dataset
      Input dataset with latitude and longitude fields

    check_total : Boolean, optional
      Test that total area is equal to area of the sphere.

    Returns
    -------

    area : xarray.DataArray
       DataArray with area field.

    """

    radius_earth = 6.37122e6  # m, radius of Earth
    area_earth = 4.0 * np.pi * radius_earth ** 2  # area of earth [m^2]e

    lon_name = infer_lon_name(ds)
    lat_name = infer_lat_name(ds)

    weights = lat_weights_regular_grid(ds[lat_name])
    area = weights + 0.0 * ds[lon_name]  # add 'lon' dimension
    area = (area_earth / area.sum(dim=(lat_name, lon_name))) * area

    if check_total:
        np.testing.assert_approx_equal(np.sum(area), area_earth)

    return xr.DataArray(
        area, dims=(lat_name, lon_name), attrs={'units': 'm^2', 'long_name': 'area'}
    )
