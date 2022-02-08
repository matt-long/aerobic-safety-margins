import numpy as np
import xarray as xr

import seawater as sw
import pop_tools


import constants

V = 32e-6 # partial molar volume of O2 (m3/mol)


def compute_pO2(O2, T, S, depth, isPOP=False):
    """
    Compute the partial pressure of O2 in seawater including 
    correction for the effect of hydrostatic pressure of the 
    water column based on Enns et al., J. Phys. Chem. 1964
      d(ln p)/dP = V/RT
    where p = partial pressure of O2, P = hydrostatic pressure
    V = partial molar volume of O2, R = gas constant, T = temperature
    
    Parameters
    ----------
    O2 : float
      Oxygen concentration (mmol/m3)
    
    T : float
       Temperature (°C)
    
    S : float
       Salinity 
       
    depth : float
       Depth (m)
       
    Returns
    -------
    pO2 : float
       Partial pressure (kPa)
       
    """    
    T_K = T + constants.T0_Kelvin

    db2Pa = 1e4 # convert pressure: decibar to Pascal    

    # Solubility with pressure effect 
    if isPOP:
        P = pop_tools.compute_pressure(depth) * 10. # pressure [dbar]: 10 = bars to dbar
        rho = 1026. # use reference density [kg/m3]
    else:
        P = sw.pres(depth, lat=0.)  # seawater pressure [db] !! Warning - z*0 neglects gravity differences w/ latitude
        rho = sw.dens(S, T, depth)  # seawater density [kg/m3]
        
    dP = P * db2Pa
    pCor = np.exp(V * dP / (constants.R_gasconst * T_K))

    Kh = 1e-3 * O2sol(S, T) * rho / constants.XiO2 # implicit division by Patm = 1 atm; solubility [mmol/m3/atm]
    
    pO2 = (xr.where(O2 < 0., 0., O2) / Kh) * pCor * constants.kPa_per_atm
    if isinstance(pO2, xr.DataArray):
        pO2.attrs['units'] = 'kPa'
        pO2.attrs['long_name'] = 'pO$_2$'
        pO2.name = 'pO2'
    return pO2


def O2sol(S, T):
    """
    Solubility of O2 in sea water
    INPUT:
    S = salinity    [PSS]
    T = temperature [degree C]
    conc = solubility of O2 [mmol/m^3]
    REFERENCE:
    Hernan E. Garcia and Louis I. Gordon, 1992.
    "Oxygen solubility in seawater: Better fitting equations"
    Limnology and Oceanography, 37, pp. 1307-1312.
    """

    # constants from Table 4 of Hamme and Emerson 2004
    return _garcia_gordon_polynomial(S, T,
                                     A0=5.80871,
                                     A1=3.20291,
                                     A2=4.17887,
                                     A3=5.10006,
                                     A4=-9.86643e-2,
                                     A5=3.80369,
                                     B0=-7.01577e-3,
                                     B1=-7.70028e-3,
                                     B2=-1.13864e-2,
                                     B3=-9.51519e-3,
                                     C0=-2.75915e-7)


def _garcia_gordon_polynomial(S,T,
                              A0 = 0., A1 = 0., A2 = 0., A3 = 0., A4 = 0., A5 = 0.,
                              B0 = 0., B1 = 0., B2 = 0., B3 = 0.,
                              C0 = 0.):

    T_scaled = np.log((298.15 - T) /(constants.T0_Kelvin + T))
    return np.exp(A0 + A1*T_scaled + A2*T_scaled**2. + A3*T_scaled**3. + A4*T_scaled**4. + A5*T_scaled**5. + \
                  S*(B0 + B1*T_scaled + B2*T_scaled**2. + B3*T_scaled**3.) + C0 * S**2.)