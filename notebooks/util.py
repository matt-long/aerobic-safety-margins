import os

from collections.abc import Iterable
import yaml

import time
import cftime

import intake
import numpy as np
import xarray as xr

import dask
from dask_jobqueue import PBSCluster
from dask.distributed import Client

path_to_here = os.path.dirname(os.path.realpath(__file__))

USER = os.environ['USER']    
PBS_PROJECT = 'NCGD0011'


def attrs_label(attrs): 
    """generate a label from long_name and units"""
    name = attrs["long_name"]
    units = '' if 'units' not in attrs else f' [{attrs["units"]}]' 
    return name + units


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


def retrieve_woa_dataset(variable, value):
    cat = intake.open_catalog("data/catalogs/woa2018-catalog.yml")
    if isinstance(variable, list):
        return xr.merge(
            [cat[v](time_code=value).to_dask() for v in variable]
        )
    else:
        return cat[variable](time_code=value).to_dask()

    
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
    
    nj = ds.TLAT.shape[0]
    ni = ds.TLONG.shape[1]

    xL = int(ni/2 - 1)
    xR = int(xL + ni)

    tlon = ds.TLONG.data
    tlat = ds.TLAT.data
    
    tlon = np.where(np.greater_equal(tlon, min(tlon[:,0])), tlon-360., tlon)    
    lon  = np.concatenate((tlon, tlon + 360.), 1)
    lon = lon[:, xL:xR]

    if ni == 320:
        lon[367:-3, 0] = lon[367:-3, 0] + 360.        
    lon = lon - 360.
    
    lon = np.hstack((lon, lon[:, 0:1] + 360.))
    if ni == 320:
        lon[367:, -1] = lon[367:, -1] - 360.

    #-- trick cartopy into doing the right thing:
    #   it gets confused when the cyclic coords are identical
    lon[:, 0] = lon[:, 0] - 1e-8

    #-- periodicity
    lat = np.concatenate((tlat, tlat), 1)
    lat = lat[:, xL:xR]
    lat = np.hstack((lat, lat[:,0:1]))

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
            dso[v] = xr.DataArray(field, dims=other_dims+('nlat', 'nlon'), 
                                  attrs=ds[v].attrs)


    # copy coords
    for v, da in ds.coords.items():
        if not ('nlat' in da.dims and 'nlon' in da.dims):
            dso = dso.assign_coords(**{v: da})
                
            
    return dso


class curator_local_assets(object):    
    
    def __init__(self):
        
        cache_dir = 'data/cache'
        os.makedirs(cache_dir, exist_ok=True)
        
        self.catalog_file = f"{path_to_here}/data/catalogs/catalog-local.yml"
        if os.path.exists(self.catalog_file):
            with open(self.catalog_file, "r") as fid:
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
        with open(self.catalog_file, "w") as fid:
            yaml.dump(self.catalog, fid)    
    
    def open_catalog(self):
        """return as intake catalog"""
        return intake.open_catalog(self.catalog_file)
    
    def __repr__(self):
        return self.catalog.__repr__()
    