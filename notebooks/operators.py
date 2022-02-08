import cftime
import numpy as np
import xarray as xr

import pop_tools

R_gasconst = 8.3144621 # J/mol/K
T0_Kelvin = 273.15
XiO2 = 0.209 # Mean atmospheric O2 mixing ratio


def _get_tb_name_and_tb_dim(ds):
    """return the name of the time 'bounds' variable and its second dimension"""
    assert 'bounds' in ds.time.attrs, 'missing "bounds" attr on time'
    tb_name = ds.time.attrs['bounds']        
    assert tb_name in ds, f'missing "{tb_name}"'    
    tb_dim = ds[tb_name].dims[-1]
    return tb_name, tb_dim


def get_chunks_dict(obj):
    """get dictionary of chunks for each dimension"""

    if not obj.chunks:
        return {}        
    
    if isinstance(obj, xr.Dataset):
        return {
            d: chunks[0] for d, chunks in obj.chunks.items()
        }
    else:
        return {
            d: chunks[0] 
            for d, chunks in zip(obj.dims, obj.chunks)
        }    


def _gen_time_weights(ds):
    """compute temporal weights using time_bound attr"""    
    tb_name, tb_dim = _get_tb_name_and_tb_dim(ds)
    wgt = ds[tb_name].compute().diff(tb_dim).squeeze().astype(float)    
    chunks_dict = get_chunks_dict(ds[tb_name])    
    if chunks_dict:        
        del chunks_dict[tb_dim]
        return wgt.chunk(chunks_dict)
    else:
        return wgt


def center_decode_time(ds, offset_days=0.):
    """
    make time the center of the time bounds
    assumes time has not been decoded
    """    
    ds = ds.copy()
    attrs = ds.time.attrs
    encoding = ds.time.encoding
    tb_name, tb_dim = _get_tb_name_and_tb_dim(ds)
    
    time_values = cftime.num2date(
        ds[tb_name].compute().mean(tb_dim).squeeze() + offset_days,
        units=ds.time.units,
        calendar=ds.time.calendar,
    )
    ds['time'] = xr.DataArray(time_values, dims=('time'))
    attrs['note'] = f'time recomputed as {tb_name}.mean({tb_dim})'
    
    encoding['units'] = attrs.pop('units')
    encoding['calendar'] = attrs.pop('calendar')
    ds.time.attrs = attrs
    ds.time.encoding = encoding
    return ds


def resample_ann(ds):
    """
    compute the annual mean of an xarray.Dataset
    assumes time has been centered
    """
    
    weights = _gen_time_weights(ds)
    weights = weights.groupby('time.year') / weights.groupby('time.year').sum()
   
    # ensure they all add to one
    # TODO: build support for situations when they don't, 
    # i.e. define min coverage threshold
    nyr = len(weights.groupby('time.year'))
    np.testing.assert_allclose(weights.groupby('time.year').sum().values, np.ones(nyr))
        
    # ascertain which variables have time and which don't
    tb_name, tb_dim = _get_tb_name_and_tb_dim(ds)
    time_vars = [v for v in ds.data_vars if 'time' in ds[v].dims and v != tb_name]
    other_vars = list(set(ds.variables) - set(time_vars) - {tb_name, 'time'} )

    # compute
    with xr.set_options(keep_attrs=True):        
        return xr.merge((
            ds[other_vars],         
            (ds[time_vars] * weights).groupby('time.year').sum(dim='time'),
        )).rename({'year': 'time'})    


def global_mean(ds, normalize=True, include_ms=False):
    """
    Compute the global mean on a POP dataset. 
    Return computed quantity in conventional units.
    """

    compute_vars = [
        v for v in ds 
        if 'time' in ds[v].dims and ('nlat', 'nlon') == ds[v].dims[-2:]
    ]
    other_vars = list(set(ds.variables) - set(compute_vars))

    if include_ms:
        surface_mask = ds.TAREA.where(ds.KMT > 0).fillna(0.)
    else:
        surface_mask = ds.TAREA.where(ds.REGION_MASK > 0).fillna(0.)        
    
    masked_area = {
        v: surface_mask.where(ds[v].notnull()).fillna(0.) 
        for v in compute_vars
    }

    with xr.set_options(keep_attrs=True):
        
        dso = xr.Dataset({
            v: (ds[v] * masked_area[v]).sum(['nlat', 'nlon'])
            for v in compute_vars
        })
        if normalize:
            dso = xr.Dataset({
                v: dso[v] / masked_area[v].sum(['nlat', 'nlon'])
                for v in compute_vars
            })            
        else:
            for v in compute_vars:
                if v in variable_defs.C_flux_vars:
                    dso[v] = dso[v] * nmols_to_PgCyr
                    dso[v].attrs['units'] = 'Pg C yr$^{-1}$'
                
        return xr.merge([dso, ds[other_vars]]).drop(
            [c for c in ds.coords if ds[c].dims == ('nlat', 'nlon')]
        )
    

def mean_time(ds, sel_dict):
    """compute the mean over a time range"""
    ds = ds.sel(sel_dict)
    try:
        weights = _gen_time_weights(ds)
    except AssertionError as error:
        traceback.print_tb(error.__traceback__) 
        warnings.warn('could not generate time_weights\nusing straight average')        
        return ds.sel(sel_dict).mean('time')       
    
    tb_name, _ = _get_tb_name_and_tb_dim(ds)
    time_vars = [v for v in ds.data_vars if 'time' in ds[v].dims and v != tb_name]
    other_vars = list(set(ds.variables) - set(time_vars) - {tb_name, 'time'})
    
    with xr.set_options(keep_attrs=True):
        dso = (ds[time_vars] * weights).sum('time') / weights.sum('time')
        return xr.merge([dso, ds[other_vars]])
    
    
def pop_mean_z(ds, max_depth_m=2000.):
    """compute vertical mean"""
    sel_dict = dict(z_t=slice(0, max_depth_m*1e2))
    dz_wgts = ds.dz.sel(sel_dict) / ds.dz.sel(sel_dict).sum()    
    with xr.set_options(keep_attrs=True):    
        return (ds.sel(sel_dict) * dz_wgts).sum('z_t')

    
def linear_trend(da, x=None, dim='time'):
    """compute linear trend using `apply_ufunc`"""

    if x is None:
        x = np.arange(ds.sizes[dim])
    
    def _linear_trend(y):
        """ufunc to be used by linear_trend"""
        return np.polyfit(x, y, 1)[0]
            
    da_chunk = da.chunk({dim: -1}).persist()        
    
    trend = xr.apply_ufunc(
        _linear_trend, 
        da_chunk, 
        input_core_dims=[[dim]],
        output_core_dims=[[]],
        output_dtypes=[np.float],
        vectorize=True,        
        dask='parallelized',
    )
        
    return trend

    
def to_datenum(y, m, d, time_units='days since 0001-01-01 00:00:00'):
    """convert year, month, day to number"""      
    return cftime.date2num(cftime.datetime(y, m, d), units=time_units)


def year_frac_noleap(time):
    """compute year fraction"""
    nt = len(time)
    year = [time.values[i].year for i in range(nt)]
    month = [time.values[i].month for i in range(nt)]
    day = [time.values[i].day for i in range(nt)]        
    t0_year = np.array([to_datenum(y, 1, 1) - 1 for y in year])
    t_year = np.array([to_datenum(y, m, d) for y, m, d in zip(year, month, day)])    
    return year + (t_year - t0_year) / 365.    


