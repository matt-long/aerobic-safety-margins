import os
from functools import partial

import intake
import numpy as np
from scipy import stats as scistats

import pandas as pd
from scipy.optimize import newton

import constants


Tref = 15. # °C
Tref_K = Tref + constants.T0_Kelvin


def open_traits_df(pressure_kPa=True):
    """Open the MI traits dataset from Deutsh et al. (2020); return a pandas.DataFrame
    
    """
    
    path_to_here = os.path.dirname(os.path.realpath(__file__))
    cache_file = f'{path_to_here}/data/MI-traits-data/MI-traits-Deutsch-etal-2020.json'
    os.rename(cache_file, f'{cache_file}.old')

    try:
        cat = intake.open_catalog("data/MI-traits-data/catalog-metabolic-index-traits.yml")
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
    def __init__(self, df, trait):
        self.dist_type = 'norm' if trait in ['Eo'] else 'lognorm'
        self.pdf_func = scistats.norm if trait in ['Eo'] else scistats.lognorm
        self.beta = self.pdf_func.fit(df[trait].values)            
        
    def fitted(self, bins):
        return self.pdf_func.pdf(bins, *self.beta)

    def median(self):
        return self.pdf_func.median(*self.beta)

    
def compute_ATmax(pO2, Ac, Eo, dEodT=0.):
    """
    Compute the maximum temperature at which resting or active (sustained)
    metabolic rate can be realized at a given po2.

    Parameters
    ----------
    Po2 : float
        Ambient O2 pressure (atm)

    Ac : float
        Hypoxia tolerance at Tref (1/atm) - can be either at rest (Ao) or at
        an active state.  For active thermal tolerance, argument should
        be Ac = Ao / Phi_crit

    Eo : float
        Temperature sensitivity of hypoxia tolerance (eV)
    
    dEdT: float
        Rate of change of Eo with temperature
    

    Note: Ac*Po2 must be unitless, but the units of either one are arbitrary

    Returns
    -------
    Tmax : float
        The 
    """
    
    def Phi_opt(T):
        return Phi(pO2, T, Ac, Eo, dEodT) - 1.
    
    # make a good initial guess for Tmax
    # - evaluate function over large temperature range
    # - find the zero crossings
    # - pick the highest 
    trange = np.arange(-2., 201., 1.)
    fvalue = Phi_opt(trange)
    fvalue[fvalue==0.] = np.nan
    sign = fvalue / np.abs(fvalue)
    ndx = np.where(sign[:-1] != sign[1:])[0]

    # no solution
    if len(ndx) == 0:
        return np.nan
    
    return newton(Phi_opt, trange[ndx[-1]])    


def Phi(pO2, T, Ac, Eo, dEodT=0.):
    """compute the metabolic index"""
    return Ac * pO2 * _Phi_exp(T, Eo, dEodT)


def pO2_at_Phi_one(T, Ac, Eo, dEodT=0.):
    """compute pO2 at Φ = 1"""
    return np.reciprocal(Ac * _Phi_exp(T, Eo, dEodT))


def _Phi_exp(T, Eo, dEodT):
    T_K = T + constants.T0_Kelvin
    return np.exp(constants.kb_inv * (Eo + dEodT * (T_K - Tref_K)) * (1.0 / T_K - 1.0 / Tref_K))





