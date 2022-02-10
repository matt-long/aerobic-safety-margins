import constants
import numpy as np
import pop_tools
import thermodyn


def test_solubility():
    """check value from literature
    Reference:
    Hernan E. Garcia and Louis I. Gordon, 1992.
      "Oxygen solubility in seawater: Better fitting equations"
      Limnology and Oceanography, 37, pp. 1307-1312.
      https://doi.org/10.4319/lo.1992.37.6.1307

    Check value in Table 1: 274.610 at S = 35.0 and T = 10.0
    """
    func_val = thermodyn.O2sol(35.0, 10.0)
    check_val = 274.610
    np.testing.assert_almost_equal(func_val, check_val, decimal=3)


def test_compute_pO2():
    """
    pO2 at a saturated concentration should equal
    the atmospheric mixing ratio
    """
    saturated_conc = thermodyn.O2sol(35.0, 10.0)
    pO2 = thermodyn.compute_pO2(
        saturated_conc,
        10.0,
        35.0,
        0.0,
        gravimetric_units=True,
    )
    np.testing.assert_almost_equal(pO2 / constants.kPa_per_atm, constants.XiO2)


def test_compute_pO2_pop():
    """demonstrate that POP/non-POP results are pretty close"""

    grid = pop_tools.get_grid('POP_gx1v6')

    O2 = np.arange(10.0, 355, 5.0)
    T = 10.0
    S = 34.7
    k = 40

    Z = grid.z_t[k] * 1e-2

    for o2 in O2:
        check_value = thermodyn.compute_pO2(o2, T, S, Z)
        pop_value = thermodyn.compute_pO2(o2, T, S, grid.z_t.data[k] * 1e-2, isPOP=True)
        np.testing.assert_allclose(check_value, pop_value, rtol=1e-2)
