import constants
import numpy as np
import pop_tools
import seawater as sw
import xarray as xr

V = 32e-6  # partial molar volume of O2 (m3/mol)


def compute_pO2(O2, T, S, depth, isPOP=False, gravimetric_units=False):
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

    isPOP : boolean, optional
      If True, assume constant reference density and use function from
      `pop_tools` to compute pressure from depth.

    gravimetric_units : boolean, optional
      Set to "True" if O2 is passed in with units of µmol/kg.

    Returns
    -------
    pO2 : float
       Partial pressure (kPa)

    """
    T_K = T + constants.T0_Kelvin

    db2Pa = 1e4  # convert pressure: decibar to Pascal

    # Solubility with pressure effect
    if isPOP:
        # pressure [dbar]: 10 = bars to dbar
        P = pop_tools.compute_pressure(depth) * 10.0
        rho = 1026.0  # use reference density [kg/m3]
    else:
        # seawater pressure [db]
        # neglects gravity differences w/ latitude
        P = sw.pres(depth, lat=0.0)
        # seawater density [kg/m3]
        rho = sw.dens(S, T, depth)

    dP = P * db2Pa
    pCor = np.exp(V * dP / (constants.R_gasconst * T_K))

    # implicit division by Patm = 1 atm
    Kh = O2sol(S, T) / constants.XiO2  # solubility [µmol/kg/atm]
    if not gravimetric_units:
        Kh *= 1e-3 * rho  # solubility [mmol/m3/atm]

    with xr.set_options(keep_attrs=True):
        pO2 = (xr.where(O2 < 0.0, 0.0, O2) / Kh) * pCor * constants.kPa_per_atm

    if isinstance(pO2, xr.DataArray):
        pO2.attrs['standard_name'] = 'partial_pressure_oxygen_kPa'
        pO2.attrs['units'] = 'kPa'
        pO2.attrs['long_name'] = 'pO$_2$'
        pO2.attrs['note'] = 'computed from O2, T, and S'
        pO2.name = 'pO2'

    return pO2


def O2sol(S, T):
    """
    Solubility of O2 in sea water

    Reference:
    Hernan E. Garcia and Louis I. Gordon, 1992.
      "Oxygen solubility in seawater: Better fitting equations"
      Limnology and Oceanography, 37, pp. 1307-1312.
      https://doi.org/10.4319/lo.1992.37.6.1307

    Coefficients are in Table 1, using the fit to the Benson & Krause (1984) data.
    Check value in Table 1: 274.610 at S = 35.0 and T = 10.0

    Parameters
    ----------

    S : float, array_like
      Salinity [PSS]
    T : float, array_like
      Temperature [degree C]

    Returns
    -------
    conc : float, array_like
      Solubility of O2 [µmol/kg]

    """

    # constants from Table 4 of Hamme and Emerson 2004
    return _garcia_gordon_polynomial(
        S,
        T,
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
        C0=-2.75915e-7,
    )


def _garcia_gordon_polynomial(
    S,
    T,
    A0=0.0,
    A1=0.0,
    A2=0.0,
    A3=0.0,
    A4=0.0,
    A5=0.0,
    B0=0.0,
    B1=0.0,
    B2=0.0,
    B3=0.0,
    C0=0.0,
):

    T_scaled = np.log((298.15 - T) / (constants.T0_Kelvin + T))
    return np.exp(
        A0
        + A1 * T_scaled
        + A2 * T_scaled ** 2
        + A3 * T_scaled ** 3
        + A4 * T_scaled ** 4
        + A5 * T_scaled ** 5
        + S * (B0 + B1 * T_scaled + B2 * T_scaled ** 2 + B3 * T_scaled ** 3)
        + C0 * S ** 2
    )
