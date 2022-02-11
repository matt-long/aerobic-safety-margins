import os
from functools import partial

import constants
import intake
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats as scistats
from scipy.optimize import newton

Tref = 15.0  # °C
Tref_K = Tref + constants.T0_Kelvin

dEodT_bar = 0.022  # eV/°C


def open_traits_df(pressure_kPa=True):
    """Open the MI traits dataset from Deutsh et al. (2020); return a pandas.DataFrame"""

    path_to_here = os.path.dirname(os.path.realpath(__file__))
    cache_file = f'{path_to_here}/data/MI-traits-data/MI-traits-Deutsch-etal-2020.json'
    os.rename(cache_file, f'{cache_file}.old')

    try:
        cat = intake.open_catalog('data/MI-traits-data/catalog-metabolic-index-traits.yml')
        data = cat['MI-traits'].read()

        df = pd.DataFrame()
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
            df[key].attrs = attrs
        os.remove(f'{cache_file}.old')
    except:
        print('trait db access failed')
        os.rename(f'{cache_file}.old', cache_file)
        raise

    return df


class trait_pdf(object):
    """Class to simplify fitting trait PDFs and returning functions"""

    normal_traits = ['Eo']

    def __init__(self, df, trait, N, bounds=None):

        if bounds is None:
            bounds = np.percentile(df[trait].values, [0.5, 99.5])

        if trait in self.normal_traits:
            self.pdf_func = scistats.norm
            coord_data = np.round(np.linspace(bounds[0], bounds[1], N), 4)
        else:
            self.pdf_func = scistats.lognorm
            coord_data = np.round(np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), N), 4)

        self.coord = xr.DataArray(coord_data, dims=(trait), name=trait, attrs=df[trait].attrs)
        self.beta = self.pdf_func.fit(df[trait].values)

    def fitted(self, bins):
        return self.pdf_func.pdf(bins, *self.beta)

    def median(self):
        return self.pdf_func.median(*self.beta)


def compute_ATmax(pO2, Ac, Eo, dEodT=0.0):
    """
    Compute the maximum temperature at which resting or active (sustained)
    metabolic rate can be realized at a given po2.

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
