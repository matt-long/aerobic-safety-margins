import os
from functools import partial

import constants
import intake
import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr
from scipy import stats as scistats
from scipy.optimize import newton

Tref = 15.0  # °C
Tref_K = Tref + constants.T0_Kelvin

dEodT_bar = 0.022  # eV/°C


def open_traits_df(pressure_kPa=True, add_ATmax=False, pull_remote=False):
    """Open the MI traits dataset from Deutsh et al. (2020); return a pandas.DataFrame"""

    path_to_here = os.path.dirname(os.path.realpath(__file__))
    cache_file = f'{path_to_here}/data/MI-traits-data/MI-traits-Deutsch-etal-2020.json'

    if pull_remote:
        os.rename(cache_file, f'{cache_file}.old')

    try:
        cat = intake.open_catalog('data/MI-traits-data/catalog-metabolic-index-traits.yml')
        data = cat['MI-traits'].read()

        df = pd.DataFrame()
        save_attrs = {}
        for key, info in data.items():
            attrs = info['attrs']
            data_type = info['data_type']

            if data_type == 'string':
                values = np.array(info['data'])
            else:
                values = np.array(info['data']).astype(np.float64)
                scale_factor = 1.0
                if pressure_kPa:
                    if 'units' in attrs:
                        if attrs['units'] == '1/atm':
                            scale_factor = 1.0 / constants.kPa_per_atm
                            attrs['units'] = '1/kPa'
                        elif attrs['units'] == 'atm':
                            scale_factor = constants.kPa_per_atm
                            attrs['units'] = 'kPa'
                values *= scale_factor

            df[key] = values
            save_attrs[key] = attrs

        if pull_remote:
            os.remove(f'{cache_file}.old')

    except:
        print('trait db access failed')
        if pull_remote:
            os.rename(f'{cache_file}.old', cache_file)
        raise

    if add_ATmax:
        PO2_atm = constants.XiO2 * constants.kPa_per_atm
        ATmax_active = []
        ATmax_resting = []
        for i, rowdata in df.iterrows():
            ATmax_active.append(
                compute_ATmax(PO2_atm, rowdata['Ac'], rowdata['Eo'], dEodT=dEodT_bar)
            )
            ATmax_resting.append(
                compute_ATmax(PO2_atm, rowdata['Ao'], rowdata['Eo'], dEodT=dEodT_bar)
            )
        df['ATmax_active'] = ATmax_active
        df['ATmax_resting'] = ATmax_resting

    # add the attribute
    for key, attrs in save_attrs.items():
        df[key].attrs = attrs

    return df


class trait_pdf(object):
    """Class to simplify fitting trait PDFs and returning functions"""

    normal_traits = ['Eo']

    def __init__(self, df, trait, N=None, bounds=None, coord_data=None):

        if coord_data is None:
            assert N is not None
            if bounds is None:
                bounds = df[trait].min(), df[trait].max()

        if trait in self.normal_traits:
            self.pdf_func = scistats.norm
            if coord_data is None:
                coord_data = np.round(np.linspace(bounds[0], bounds[1], N), 4)
        else:
            self.pdf_func = scistats.lognorm
            if coord_data is None:
                coord_data = np.round(np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), N), 4)

        self.coord = xr.DataArray(
            coord_data,
            dims=(trait),
            name=trait,
            attrs=df[trait].attrs,
            coords={trait: coord_data},
        )
        self.beta = self.pdf_func.fit(df[trait].values)

    @property
    def pdf(self):
        return self.pdf_func.pdf(self.coord, *self.beta)

    @property
    def cdf(self):
        return self.pdf_func.cdf(self.coord, *self.beta)

    def median(self):
        return self.pdf_func.median(*self.beta)


def open_CTmax_data():
    """Return an xarray.Dataset with CTmax data from OBIS.
    TODO: improve provenance!
    """
    matdata = sio.loadmat(
        'data/MI-traits-data/obis_Tmax.mat', struct_as_record=False, squeeze_me=True
    )
    data = matdata['obis']

    return xr.Dataset(
        dict(
            lat=xr.DataArray(
                data.Ymed,
                dims=('N'),
                attrs={
                    'long_name': 'Median latitude of observed occurrence',
                    'units': 'degrees_north',
                },
            ),
            lat_dist=xr.DataArray(
                data.Yprct,
                dims=('N', 'percentile'),
                attrs={
                    'long_name': 'Latitude distribution of observed occurence',
                    'units': 'degrees_north',
                },
                coords={'percentile': data.prct},
            ),
            Thabitat_dist=xr.DataArray(
                data.Tprct,
                dims=('N', 'percentile'),
                attrs={
                    'long_name': 'Distribution of inhabited temperature',
                    'units': '°C',
                },
                coords={'percentile': data.prct},
            ),
            CTmax=xr.DataArray(
                data.CTmax,
                dims=('N'),
                attrs={
                    'long_name': 'CTmax',
                    'units': '°C',
                },
            ),
            Species=xr.DataArray(
                data.species,
                dims=('N'),
                attrs={
                    'long_name': 'Species',
                },
            ),
            Phylum=xr.DataArray(
                data.phylum,
                dims=('N'),
                attrs={
                    'long_name': 'Phylum',
                },
            ),
        )
    )


def compute_ATmax(pO2, Ac, Eo, dEodT=0.0):
    """
    Compute the maximum temperature at which resting or active (sustained)
    metabolic rate can be realized at a given pO2.

    Parameters
    ----------
    pO2 : float
        Ambient O2 pressure (atm or kPa)

    Ac : float
        Hypoxia tolerance at Tref (1/atm) or (1/kPa)
        Can be either at rest (Ao) or at an active state.
        For active thermal tolerance, argument should be
           Ac = Ao / Phi_crit

    Eo : float
        Temperature sensitivity of hypoxia tolerance (eV)

    dEdT: float
        Rate of change of Eo with temperature

    Note: Ac*Po2 must be unitless, but the units of either one are arbitrary

    Returns
    -------
    ATmax : float
        The maximum temperature for sustained respiration at PO2.
    """

    def Phi_opt(T):
        return Phi(pO2, T, Ac, Eo, dEodT) - 1.0

    # make a good initial guess for Tmax
    # - evaluate function over large temperature range
    # - find the zero crossings
    # - pick the highest
    trange = np.arange(-200.0, 201.0, 1.0)
    fvalue = Phi_opt(trange)
    fvalue[fvalue == 0.0] = np.nan
    sign = fvalue / np.abs(fvalue)
    ndx = np.where(sign[:-1] != sign[1:])[0]

    # no solution
    if len(ndx) == 0:
        return np.nan

    return newton(Phi_opt, trange[ndx[-1]])


def compute_ASM_Tmax(pO2_v_T_slope, pO2_v_T_intercept, Ac, Eo, dEodT=0.0):
    """
    Compute the maximum temperature at which resting or active (sustained)
    metabolic rate can be realized at a given po2.

    Parameters
    ----------
    pO2_v_T_slope : float
        Slope of pO2 versus T relationship (kPa/°C)

    pO2_v_T_intercept : float
        Intercept of pO2 versus T relationship (kPa)

    Ac : float
        Hypoxia tolerance at Tref (1/kPa)
        Can be either at rest (Ao) or at an active state.
        For active thermal tolerance, argument should be
           Ac = Ao / Phi_crit

    Eo : float
        Temperature sensitivity of hypoxia tolerance (eV)

    dEdT: float
        Rate of change of Eo with temperature

    Returns
    -------
    ASM_Tmax : float
        The maximum temperature for sustained respiration presuming correlative
        relationship between pO2 and T.

        If there is no intersection between the pO2-T relationship and the
        pO2 at Phi = 1 (or pO2 at Phi = Phi_crit) line, the function returns
        ATmax.
    """
    if np.isnan(pO2_v_T_slope) or np.isnan(pO2_v_T_intercept):
        return np.nan

    PO2_atm = constants.XiO2 * constants.kPa_per_atm
    ATmax = compute_ATmax(PO2_atm, Ac, Eo, dEodT)

    def func_opt(T):
        return pO2_v_T_slope * T + pO2_v_T_intercept - pO2_at_Phi_one(T, Ac, Eo, dEodT)

    # make a good initial guess for Tmax
    # - evaluate function over large temperature range
    # - find the zero crossings
    # - pick the highest
    trange = np.arange(-200.0, 201.0, 1.0)
    fvalue = func_opt(trange)
    fvalue[fvalue == 0.0] = np.nan
    sign = fvalue / np.abs(fvalue)
    ndx = np.where(sign[:-1] != sign[1:])[0]

    # no solution
    if len(ndx) == 0:
        return ATmax

    try:
        ASM_Tmax = newton(func_opt, trange[ndx[-1]])
    except:
        print('ASM_Tmax convergence failed:')
        print(f'pO2_v_T_slope={pO2_v_T_slope}', end=',')
        print(f'pO2_v_T_intercept={pO2_v_T_intercept}', end=',')
        print(f'Ac={Ac}', end=',')
        print(f'Eo={Eo}', end=',')
        print(f'dEodT={dEodT}', end=',')
        print()
        ASM_Tmax = ATmax

    if ASM_Tmax > ATmax:
        return ATmax
    else:
        return ASM_Tmax


def Phi(pO2, T, Ac, Eo, dEodT=0.0):
    """
    Compute the metabolic index.

    References:
     1. Deutsch, C., A. Ferrel, B. Seibel, H. O. Pörtner,
         R. B. Huey, Climate change tightens a metabolic constraint
         on marine habitats. Science 348, 1132–1135 (2015)
         doi:10.1126/science.aaa1605.

     2. Deutsch, C., J. L. Penn, B. Seibel, Metabolic
         trait diversity shapes marine biogeography, Nature,
         585, (2020) doi:10.1038/s41586-020-2721-y.

    Parameters
    ----------

    pO2 : float, array_like
       Partial pressure of oxygen (atm or kPa)

    T : float, array_like
       Temperature (°C)

    Ac : float
      Hypoxic tolerante (1/atm or 1/kPa)

    Eo : float
      Temperature dependence of metabolic rate (eV)

    dEodT : float
       Temperature sensitivty of Eo

    Returns
    -------

    Phi : float
      The Metabolic Index

    """
    return Ac * pO2 * _Phi_exp(T, Eo, dEodT)


def pO2_at_Phi_one(T, Ac, Eo, dEodT=0.0):
    """compute pO2 at Φ = 1"""
    return np.reciprocal(Ac * _Phi_exp(T, Eo, dEodT))


def _Phi_exp(T, Eo, dEodT):
    T_K = T + constants.T0_Kelvin
    return np.exp(constants.kb_inv * (Eo + dEodT * (T_K - Tref_K)) * (1.0 / T_K - 1.0 / Tref_K))
