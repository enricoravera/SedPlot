import math
import numpy as np

N_A = 6.02214076e23
k_B = 1.380649e-23
Rb  = 8.31
EINSTEIN_COEFF = 2.5
D_CH_RIGID = 2 * np.pi * 23.0e3   # rad/s
D_HH_RIGID = 2 * np.pi * 40.0e3   # rad/s


def water_viscosity(T):
    """Return dynamic viscosity of water (Pa·s) at temperature T (K).
    Correlation: 2.414e-5 * 10^(247.8 / (T - 140)).
    """
    return 2.414e-5 * (10 ** (247.8 / (T - 140.0)))


def perrin_shape_factors(p):
    """Perrin (1936) translational frictional ratio F_T and empirical
    rotational shape factor F_R = F_T**4 (HullRad, Fleming & Fleming 2018)
    for a prolate (p>1) or oblate (p<1) ellipsoid of revolution with axial
    ratio p = a/b relative to a sphere of equal volume.
    Returns (1.0, 1.0) for p == 1.

    Verified values: F_T ≈ 1.044 at p=2, ≈ 1.25 at p=5, ≈ 1.54 at p=10.
    """
    if abs(p - 1.0) < 1e-6:
        return 1.0, 1.0
    if p > 1.0:
        F_T = math.sqrt(p**2 - 1.0) / (p**(1.0/3.0) * math.log(p + math.sqrt(p**2 - 1.0)))
    else:
        F_T = math.sqrt(1.0 - p**2) / (p**(1.0/3.0) * math.atan(math.sqrt(1.0 - p**2) / p))
    F_R = F_T**4
    return F_T, F_R


def compute_cp_efficiency_vec(
        tau_c, omega_R, omega_0_H, omega_0_C, omega_1_H, omega_1_C,
        b, S, tau_s, tau_CP,
        d_ch_rigid=D_CH_RIGID, d_hh_rigid=D_HH_RIGID):
    """Vectorized CP efficiency ratio I_CP/I_DP (Topgaard model with
    mobile + rigid-lattice channels and MAS sideband weighting).
    """
    gamma_H = 267.513e6
    gamma_C = 67.262e6
    b_tesla = b * 1e-3
    w_R = 2 * np.pi * omega_R
    w_0_H = 2 * np.pi * omega_0_H
    w_1_H = 2 * np.pi * omega_1_H
    w_1_C = 2 * np.pi * omega_1_C

    def j_0(w):
        term1 = (1 - S**2) * (2 * tau_c) / (1 + (w * tau_c)**2)
        term2 = S**2 * (2 * tau_s) / (1 + (w * tau_s)**2)
        return term1 + term2

    # Rigid-lattice spectral density: same Lorentzian form as j_0, but the
    # fluctuation that broadens it is the homonuclear (1H-1H) dipolar
    # network (tau_HH = 1/D_HH_RIGID), not molecular tumbling. Unlike j_0,
    # this does NOT vanish as tau_c -> infinity, which is exactly the
    # "solid-like" behaviour a fully jammed/rigid protein should show.
    tau_HH = 1.0 / d_hh_rigid

    def j_0_rigid(w):
        return (2 * tau_HH) / (1 + (w * tau_HH)**2)

    def sidebands(j0_func, w):
        return (1/3) * (j0_func(w - w_R) + j0_func(w + w_R)) + (1/6) * (j0_func(w - 2*w_R) + j0_func(w + 2*w_R))

    def j(w):
        return sidebands(j_0, w)

    def j_rigid(w):
        return sidebands(j_0_rigid, w)

    R_1rho_H = ((gamma_H * b_tesla)**2 / 2.0) * (j(w_1_H) + j(w_0_H))

    # Heteronuclear CP rate = mobile (motionally-averaged) channel + rigid
    # (solid-like) channel. The two add because they are independent
    # transfer pathways; R_CH_mobile dominates while the molecule still
    # tumbles fast enough to fall in the BWR/motional-narrowing regime,
    # R_CH_rigid takes over (and keeps CP efficient) once tau_c grows long
    # enough that the mobile channel's spectral density has collapsed.
    # Both still carry the same first/second-order MAS-sideband weighting
    # (1/3, 1/6 at +-wR, +-2wR), which is what lets the rigid channel
    # remain MAS-frequency dependent even though it sits exactly on the
    # n=0 (MAS-independent) Hartmann-Hahn condition used experimentally.
    R_CH_mobile = ((gamma_C * b_tesla)**2 / 2.0) * j(w_1_H - w_1_C)

    # Motional-narrowing gate: the static D_CH_RIGID coupling is only
    # "seen" once tumbling is slow enough that it is no longer averaged
    # away. Uses an exponential (Arrhenius/mollifier-type) onset rather
    # than a power law: f_rigid ~ exp(-1/x) vanishes to ALL polynomial
    # orders as x=D_CH_RIGID*tau_c -> 0, so it suppresses fast-tumbling/
    # dilute leakage much more strongly than a x^2/(1+x^2) gate, while
    # still saturating to 1 once tau_c grows past the 1/D_CH_RIGID
    # threshold. Also has a gentler slope right at its half-max point,
    # which is what removes the artificial extra steepness this channel
    # was otherwise adding on top of the already-steep Ross-Minton rise.
    with np.errstate(divide='ignore', over='ignore'):
        f_rigid = np.where(tau_c > 0, np.exp(-1.0 / (d_ch_rigid * tau_c)), 0.0)
    R_CH_rigid = f_rigid * (d_ch_rigid**2 / 2.0) * j_rigid(w_1_H - w_1_C)
    R_CH = R_CH_mobile + R_CH_rigid

    with np.errstate(divide='ignore'):
        T_1rho_H = np.where(R_1rho_H > 0, 1.0 / R_1rho_H, np.inf)
        T_CH = np.where(R_CH > 0, 1.0 / R_CH, np.inf)

    efficiency_term = np.zeros_like(T_CH)
    inf_mask = np.isinf(T_1rho_H)
    close_mask = np.isclose(T_CH, T_1rho_H) & ~inf_mask
    normal_mask = ~inf_mask & ~close_mask

    if np.any(close_mask):
        efficiency_term[close_mask] = (tau_CP / T_1rho_H[close_mask]) * np.exp(-tau_CP / T_1rho_H[close_mask])
    if np.any(normal_mask):
        t1, tch = T_1rho_H[normal_mask], T_CH[normal_mask]
        ratio = tch / t1
        efficiency_term[normal_mask] = (np.exp(-tau_CP / t1) - np.exp(-tau_CP / tch)) / (1.0 - ratio)

    return (gamma_H / gamma_C) * efficiency_term


def ross_minton_tau_c(c_ratio, tau_c0, einstein_coeff_eff, phi_m, delta_smooth):
    """Ross-Minton concentration-dependent correlation time.

    ln(eta_rel) = einstein_coeff_eff * c_ratio / (1 - c_ratio + delta_smooth)
    tau_c = tau_c0 * exp(clipped exponent)

    Parameters
    ----------
    c_ratio            : ndarray  c / cl0, clipped to [0, 0.99] before calling
    tau_c0             : float    dilute-limit correlation time (s)
    einstein_coeff_eff : float    EINSTEIN_COEFF * F_T * phi_m  (caller constructs this)
    phi_m              : float    max packing fraction (kept for signature symmetry)
    delta_smooth       : float    jamming-rounding parameter (0 → original hard clip)

    Returns
    -------
    tau_c : ndarray  clipped to [0, 1e6]
    """
    exponent = np.clip(
        einstein_coeff_eff * c_ratio / (1.0 - c_ratio + delta_smooth), 0, 50
    )
    return np.clip(tau_c0 * np.exp(exponent), 0, 1e6)


def derive_hydrodynamics(MW, T, v_eff, p_axial):
    """Compute dilute-limit hydrodynamic quantities for a globular protein.

    Parameters
    ----------
    MW      : float  molecular weight (kDa)
    T       : float  temperature (K)
    v_eff   : float  effective specific volume (mL/g) = v_bar + delta_hyd + v_cav
    p_axial : float  Perrin axial ratio a/b

    Returns
    -------
    tau_c0 : float  dilute-limit rotational correlation time (s)
    F_T    : float  translational Perrin shape factor
    F_R    : float  rotational shape factor (F_T**4)
    """
    F_T, F_R = perrin_shape_factors(p_axial)
    M_mol = MW * 1000.0
    eta0 = water_viscosity(T)
    V_h = M_mol * v_eff * 1e-6 / N_A * F_R
    tau_c0 = eta0 * V_h / (k_B * T)
    return tau_c0, F_T, F_R


def c_cylindrical_vec(r, A, k2, cl0):
    """Radial sedimentation concentration profile c(r) for a cylindrical rotor
    (sigmoid in r²), clipped to prevent float overflow.
    """
    return cl0 / (1.0 + np.exp(np.clip(A - k2 * r * r, -500.0, 500.0)))
