import os

import numpy as np
import xarray as xr

import funnel as fn
import operators as ops
import thermodyn


USER = os.environ['USER']    
catalog_json = 'data/catalogs/glade-cesm1-le.json'
cache_dir = f'/glade/scratch/{USER}/ocean-metabolism/funnel-cache'


use_only_ocean_bgc_member_ids = True
ocean_bgc_member_ids = [
    1, 2, 9, 10, 
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
    31, 32, 34, 35, # bad data on 33?
    101, 102, 103, 104, 105,
]

time_slice = slice('1920', '2100')
drift_year_0 = 1920


def get_cdf_kwargs(stream):
    if stream in ['pop.h']:
        return {
            'chunks': {'nlat': 384, 'nlon': 320, 'z_t': 10}, 
            'decode_times': False,
        }
    else:
        raise ValueError(f'cdf_kwargs for "{stream}" not defined')


def _preprocess_pop_h_upper_1km(ds):
    """drop unneeded variables and set grid var to coords"""      
    grid_vars = ['KMT', 'TAREA', 'TLAT', 'TLONG', 'z_t', 'dz', 'z_t_150m', 'time', 'time_bound']    
    data_vars = list(filter(lambda v: 'time' in ds[v].dims, ds.data_vars))
    ds = ds[data_vars + grid_vars].sel(z_t=slice(0, 1000e2)) # top 1000 m
    new_coords = set(grid_vars) - set(ds.coords)   
    return ds.set_coords(new_coords)


def drift(**query):
    assert 'stream' in query    
    postproccess = [
        compute_time,
        sel_time_slice, 
        compute_drift,
    ]    
    return fn.Collection(
        name='linear-drift',
        esm_collection_json=catalog_json,
        preprocess=_preprocess_pop_h_upper_1km,
        postproccess=postproccess,  
        query=query,
        cache_dir=cache_dir,
        persist=True,        
        cdf_kwargs=get_cdf_kwargs(query['stream']), 
    )


def drift_corrected_ens(**query): 
    assert 'stream' in query

    postproccess = [
        compute_time, 
        sel_time_slice, 
        compute_drift_correction,
    ]
    
    if use_only_ocean_bgc_member_ids:
        query['member_id'] = ocean_bgc_member_ids      
    
    return fn.Collection(
        name='drift-corrected',
        esm_collection_json=catalog_json,
        preprocess=_preprocess_pop_h_upper_1km,
        postproccess=postproccess,  
        query=query,
        cache_dir=cache_dir,
        persist=True,        
        cdf_kwargs=get_cdf_kwargs(query['stream']), 
    )


    
@fn.register_query_dependent_op(
    query_keys=['variable', 'stream'],
)
def compute_drift_correction(ds, variable, stream):
    dsets_drift = drift(
        experiment='CTRL',
        stream=stream,
    ).to_dataset_dict(variable=variable)    
    assert len(dsets_drift.keys()) == 1
    
    key, ds_drift = dsets_drift.popitem()
    
    year_frac = xr.DataArray(
        ops.year_frac_noleap(ds.time) - drift_year_0, 
        dims=('time'), 
        name='year_frac',
    )
    
    chunks_dict = ops.get_chunks_dict(ds)
    year_frac = year_frac.chunk({'time': chunks_dict['time']})
    da_drift = year_frac * ds_drift[variable]
    da_drift = da_drift.chunk({d: chunks_dict[d] for d in da_drift.dims}).persist()
    
    attrs = ds[variable].attrs
    ds[variable] = ds[variable] - da_drift
    attrs['note'] = 'corrected for drift in control integration'
    ds[variable].attrs = attrs
    
    return ds.chunk({'time': 12})


@fn.register_query_dependent_op(
    query_keys=['experiment'],
)
def compute_time(ds, experiment):
    offset_days = (1850 - 402) * 365 if experiment == 'CTRL' else 0.
    return ops.center_decode_time(ds, offset_days=offset_days)


def sel_time_slice(ds):
    """select time index"""
    return ds.sel(time=time_slice)
    
    
def compute_drift(ds_ctrl):
    """return a dataset of the linear trend in time"""
    ds_drift = xr.Dataset()
    year_frac = ops.year_frac_noleap(ds_ctrl.time)

    for v, da in ds_ctrl.data_vars.items():
        if 'time' not in da.dims:
            continue
        da_drift = ops.linear_trend(da, x=year_frac)
        da_drift.attrs = da.attrs        
        if 'units' in da_drift.attrs:
            da_drift.attrs['units'] += '/yr'
            
        ds_drift[v] = da_drift
    
    return ds_drift


@fn.register_derived_var(
    varname='pO2',
    dependent_vars=['TEMP', 'SALT', 'O2'],
)
def compute_pO2(ds):
    """Compute the partial pressure of O2 and drop dependent variables.
    """
    ds['pO2'] = thermodyn.compute_pO2(ds.O2, ds.TEMP, ds.SALT, ds.z_t * 1e-2, isPOP=True)    
    return ds.drop(['TEMP', 'SALT', 'O2'])