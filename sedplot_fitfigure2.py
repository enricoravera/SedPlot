#!/usr/bin/env python
"""
recompute_cp_trend.py
Recomputes the expected CP efficiency trend for a 480 kDa protein sedimented 
in a 1.5 mm internal diameter rotor, comparing it to digitized experimental points.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from rm_cp_physics import (
    N_A, k_B, Rb, EINSTEIN_COEFF, D_CH_RIGID, D_HH_RIGID,
    water_viscosity, perrin_shape_factors, compute_cp_efficiency_vec,
    ross_minton_tau_c, derive_hydrodynamics, c_cylindrical_vec,
)

# ─── Constants & Physical Models ──────────────────────────────────────────────

# PHI_M: maximum (random close) packing fraction in the Ross-Minton crowding
# model. It is the volume fraction at which the hydrated protein "jams" and
# the suspension viscosity formally diverges (the cl0 concentration below is
# the mass-concentration equivalent of this packing limit). For hard, fairly
# monodisperse spheres this sits close to 0.64 (random close packing); real
# globular proteins are usually reported in the ~0.6-0.7 range depending on
# shape and polydispersity. This script has no interactive UI, so PHI_M is
# kept as a fixed, documented constant here — edit this value directly if a
# different particle/packing assumption is needed for a given fit.
PHI_M = 0.64

# DELTA_HYDRATION: thickness of the bound-water/hydration shell, expressed as
# g water per g protein (equivalently mL/g since water density ~1). It is
# added to the protein's intrinsic specific volume (1/rp0) to give the
# "effective" specific volume (v_eff) that actually gets crowded out as
# concentration rises. Like PHI_M, this is a fixed constant in this
# non-interactive script; see hydration_optimizer.py for a version where
# both are adjustable sliders.
DELTA_HYDRATION = 0.40

# DELTA_SMOOTH: rounding parameter (dimensionless) added to the Ross-Minton
# denominator (1 - c/cl0), turning the crowding exponent into
# EINSTEIN_COEFF*PHI_M*c_ratio/(1 - c_ratio + DELTA_SMOOTH). The unmodified
# Ross-Minton form formally diverges at c -> cl0; the code already
# truncates that with np.clip(exponent, 0, 50), but the clip only acts
# after the fact (the slope stays at full, unregularized steepness right
# up to the ceiling, producing a kink). DELTA_SMOOTH rounds off the
# approach itself, which is a reasonable stand-in for the polydispersity/
# finite-size effects that round off any real jamming transition rather
# than letting it diverge exactly. DELTA_SMOOTH = 0 exactly recovers the
# original (still hard-clipped) behaviour.
DELTA_SMOOTH = 0.03

# V_CAVITY: extra specific volume (mL/g) contributed by the empty interior
# of a hollow-sphere protein, on top of the protein-shell partial specific
# volume (v_bar = 1/rp0) and the surface hydration shell (DELTA_HYDRATION).
# A hollow shell tumbles, and jams against its neighbours, according to its
# full OUTER volume -- not the volume implied by its mass and shell density
# -- because the solvent-filled cavity adds no buoyant mass but adds just as
# much excluded/hydrodynamic volume as if it were solid protein. Importantly
# this term is added only to v_eff (used below for cl0 and tau_c0/V_h); it
# is deliberately NOT mixed into rp0/v_bar, which still correctly sets the
# buoyant-mass term (1 - rs0/rp0) used in the sedimentation exponent k2.
# That keeps the sedimentation (concentration-profile) physics untouched
# while correctly inflating the crowding/tumbling volume. Set to 0 to
# recover the original solid-particle behaviour.
V_CAVITY = 0.35

# P_AXIAL: axial ratio p = a/b of an equivalent ellipsoid of revolution used
# to approximate the particle's departure from spherical shape (prolate
# "rod/dumbbell"-like for p>1, oblate "disc"-like for p<1, sphere at p=1).
# This is the classic Perrin (1936) whole-body approximation: a single
# number, no coordinates or surface triangulation needed, that captures how
# much slower a non-spherical particle tumbles (and how much more it raises
# solution viscosity) than a sphere of the same volume. See
# perrin_shape_factors() below for how it is used. A "dumbbell" or other
# elongated multi-lobed assembly is conventionally represented this way too,
# via its experimentally-equivalent axial ratio (e.g. fit from a measured
# frictional ratio f/f0), rather than with an explicit two-lobe geometry.
P_AXIAL = 1.0


# ─── Experimental & Input Parameters ─────────────────────────────────────────
# User-provided specs
rotor_inner_diameter = 1.5 / 1000.0  # 1.5 mm to meters
be = rotor_inner_diameter / 2.0      # Inner radius (meters)
c0 = 60.0                             # Initial concentration (mg/mL)
T = 290.0                             # Temperature (K)
rs0 = 997 / 1000.0                    # Solvent density (g/mL)
rp0 = 1370 / 1000.0                   # Protein density (g/mL)
m0 = 480288.0 / 1000.0                # MW in kDa (480 kDa)

# NMR parameters
w0H = 700.0 * 1e6                     # 1H Larmor frequency (Hz)
w0C = w0H * (67.262e6 / 267.513e6)    # 13C Larmor frequency derived from gyromagnetic ratios
tau_CP = 83e-6                       # Contact time (seconds)
w1H = 41.0 * 1e3                      # 1H Nutation frequency (Hz)

# Underlying Topgaard interaction assumptions (from standard defaults)
b = 0.20                              # Local field fluctuation (mT)
S = 0.0                               # Order parameter
tau_s = 1.0 * 1e-3                    # Slow correlation time (seconds)
n_match = 0                           # Hartmann-Hahn match index (n = 0, actual acquisition condition)

# ─── Hydrodynamic & Jamming Concentration Setup ──────────────────────────────
v_bar = 1.0 / rp0
v_eff = v_bar + DELTA_HYDRATION + V_CAVITY
cl0 = PHI_M / v_eff * 1000.0           # Max packing limit

tau_c0, F_T_AXIAL, F_R_AXIAL = derive_hydrodynamics(m0, T, v_eff, P_AXIAL)

p = {'be': be, 'c0': c0, 'cl0': cl0, 'm0': m0, 'T': T, 'rs0': rs0, 'rp0': rp0, 'tau_c0': tau_c0}

# ─── Digitized Experimental Data ────────────────────────────────────────────
mas_exp = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000])
y_exp = np.array([0, 0, 0, 0.10112, 0.22472, 0.45506, 0.75281, 0.87079, 0.93258, 0.94944])
y_err = 0.10

# ─── Compute Theoretical CP Profiles ─────────────────────────────────────────
# Generate a smooth resolution trace for the model line
mas_dense = np.linspace(0, 13000, 300)
r_space = np.linspace(0., be, 200)

def calculate_weighted_cp(mas_array, cl0=None, tau_c0=None, einstein_coeff_eff=None, delta_smooth=None, d_hh_rigid=None):
    cl0_loc = p['cl0'] if cl0 is None else cl0
    tau_c0_loc = p['tau_c0'] if tau_c0 is None else tau_c0
    einstein_eff_loc = EINSTEIN_COEFF * F_T_AXIAL * PHI_M if einstein_coeff_eff is None else einstein_coeff_eff
    delta_smooth_loc = DELTA_SMOOTH if delta_smooth is None else delta_smooth
    d_hh_rigid_loc = D_HH_RIGID if d_hh_rigid is None else d_hh_rigid

    cp_averages = []
    for w_R in mas_array:
        if w_R < 10.0:
            c_r_local = np.full_like(r_space, c0)
        else:
            k2 = p['m0']*(1.-p['rs0']/p['rp0'])*(w_R*2*np.pi)**2/(2.*Rb*p['T'])
            alpha = k2*p['be']**2*(1.-p['c0']/cl0_loc)
            beta  = k2*p['be']**2*(p['c0']/cl0_loc)
            lnum  = alpha if alpha>100 else math.log(max(math.expm1(min(alpha,500.)),1e-300))
            lden  = 0.    if beta >100 else math.log(max(-math.expm1(-min(beta,500.)),1e-300))
            c_r_local = c_cylindrical_vec(r_space, lnum-lden, k2, cl0_loc)
        
        c_ratio = np.clip(c_r_local / cl0_loc, 0, 0.99)
        # Ross-Minton crowding exponent: ln(eta_rel) = EINSTEIN_COEFF * v_eff * c / (1 - c/cl0).
        # Re-expressed in terms of c_ratio = c/cl0, the v_eff dependence cancels exactly
        # (v_eff * cl0 == PHI_M * 1000 by construction), leaving EINSTEIN_COEFF * PHI_M.
        # F_T_AXIAL is folded in as a simple proxy for the Simha (1940) non-
        # spherical intrinsic-viscosity correction: non-spherical particles
        # raise solution viscosity faster per unit packing fraction than
        # spheres do, so the crowding exponent steepens with elongation too.
        tau_c_local = ross_minton_tau_c(c_ratio, tau_c0_loc, einstein_eff_loc, PHI_M, delta_smooth_loc)
        
        w1C_hz = w1H - n_match * w_R
        cp_eff = compute_cp_efficiency_vec(tau_c_local, w_R, w0H, w0C, w1H, w1C_hz, b, S, tau_s, tau_CP,
                                            d_hh_rigid=d_hh_rigid_loc)
        cp_eff = np.nan_to_num(cp_eff, nan=0.0)
        
        # Spatial mass weighting over the cylinder cross section
        weights = c_r_local * r_space
        tot_weight = np.sum(weights)
        cp_averages.append(np.sum(cp_eff * weights) / tot_weight if tot_weight > 0 else 0.0)
        
    return np.array(cp_averages)

# Calculate at discrete experimental points and along dense curve
cp_at_exp_points = calculate_weighted_cp(mas_exp)
cp_dense_curve = calculate_weighted_cp(mas_dense)

# Least-squares scaling factor to address arbitrary intensity scale
scaling_factor = np.sum(y_exp * cp_at_exp_points) / np.sum(cp_at_exp_points**2)
cp_dense_scaled = cp_dense_curve * scaling_factor

# ─── lmfit-Based Parameter Optimization ──────────────────────────────────────
# Lets a handful of the more uncertain, sample-specific knobs (hydration
# shell, cavity volume, axial ratio, crowding rounding, homonuclear coupling,
# max packing fraction) be optimized against the digitized experimental
# points, rather than left at the hand-picked values above. D_CH_RIGID and
# the geometry/NMR acquisition parameters are kept fixed -- those represent
# either genuine bond physics or things that were actually measured/set on
# the spectrometer, not free fit knobs.
#
# Requires lmfit (pip install lmfit). The intensity scale is NOT treated as
# an independent nonlinear fit parameter: at every trial point it is solved
# for analytically via the same linear projection used above, which removes
# one degree of freedom and keeps the fit well-posed for only 10 data points.
import lmfit

def derive_quantities(v_extra, p_axial, phi_m):
    """Thin wrapper used by the lmfit residual.
    NOTE: delta_hyd and v_cav enter only through their sum v_extra = v_bar + delta_hyd + v_cav,
    which is the only combination identifiable from the CP-vs-MAS curve.
    """
    v_eff_t = v_bar + v_extra
    cl0_t   = phi_m / v_eff_t * 1000.0
    tau_c0_t, F_T_t, F_R_t = derive_hydrodynamics(m0, T, v_eff_t, p_axial)
    einstein_eff_t = EINSTEIN_COEFF * F_T_t * phi_m
    return cl0_t, tau_c0_t, einstein_eff_t

def model_unscaled(mas_array, params):
    cl0_t, tau_c0_t, einstein_eff_t = derive_quantities(
        params['v_extra'].value, params['p_axial'].value, params['phi_m'].value)
    return calculate_weighted_cp(
        mas_array, cl0=cl0_t, tau_c0=tau_c0_t, einstein_coeff_eff=einstein_eff_t,
        delta_smooth=params['delta_smooth'].value, d_hh_rigid=2*np.pi*params['d_hh_khz'].value*1e3)

def residual(params, mas_array, y_obs, y_err_arr):
    model = model_unscaled(mas_array, params)
    denom = np.sum(model**2)
    scale = np.sum(y_obs * model) / denom if denom > 0 else 0.0
    return (scale * model - y_obs) / y_err_arr

# Only 10 (noisy) data points are available. Letting delta_smooth and/or
# d_hh_khz float alongside v_extra was tried and barely improved chi-square
# (1.63 -> 1.50 for 1 -> 3 free parameters) while blowing up the parameter
# uncertainties to 100-300% and producing strong (0.85-0.95) correlations --
# textbook signs of an underdetermined, overfit model. v_extra alone is the
# one combination this curve shape genuinely constrains (it shifts both the
# jamming point and the baseline tau_c0 together, exactly mirroring what
# hand-tuning DELTA_HYDRATION/V_CAVITY does); delta_smooth, d_hh_khz, p_axial
# and phi_m are kept fixed at their hand-set/literature values.
fit_params = lmfit.Parameters()
fit_params.add('v_extra',      value=DELTA_HYDRATION + V_CAVITY, min=0.10, max=1.00)
fit_params.add('p_axial',      value=P_AXIAL,         vary=False)  # hollow SPHERE -> keep p=1
fit_params.add('delta_smooth', value=DELTA_SMOOTH,    vary=False)
fit_params.add('d_hh_khz',     value=D_HH_RIGID/(2*np.pi)/1e3, vary=False)
fit_params.add('phi_m',        value=PHI_M,           vary=False)

y_err_arr = np.full_like(y_exp, y_err)
fit_result = lmfit.minimize(residual, fit_params, args=(mas_exp, y_exp, y_err_arr), method='leastsq')
print(lmfit.fit_report(fit_result))

best = fit_result.params
best_unscaled_exp = model_unscaled(mas_exp, best)
best_scale = np.sum(y_exp * best_unscaled_exp) / np.sum(best_unscaled_exp**2)
best_unscaled_dense = model_unscaled(mas_dense, best)
best_curve_dense = best_unscaled_dense * best_scale

chi2_before = np.sum(((scaling_factor * cp_at_exp_points - y_exp) / y_err)**2)
chi2_after = np.sum(((best_scale * best_unscaled_exp - y_exp) / y_err)**2)
dof_before, dof_after = len(y_exp) - 0, fit_result.nfree  # "before" curve wasn't fit, just hand-set

# ─── Plotting Comparison ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
plt.rcParams['font.sans-serif'] = 'Arial'

for ax in (ax1, ax2):
    ax.errorbar(mas_exp, y_exp, yerr=y_err, fmt='o', color='#d62728', elinewidth=1.5,
                capsize=3, label='Experimental Points (Arb. Scale)', zorder=4)
    ax.set_xlabel('MAS Frequency (Hz)', fontsize=10)
    ax.set_xlim(-500, 13000)
    ax.set_ylim(-0.1, 1.3)
    ax.grid(True, linestyle=':', alpha=0.6)

ax1.plot(mas_dense, cp_dense_scaled, color='#1f77b4', lw=2.5,
         label=f'Recomputed CP Trend (Scaled x {scaling_factor:.3f})', zorder=3)
ax1.set_title(f'Hand-Set Parameters\n$\\chi^2$={chi2_before:.2f} ({dof_before} pts, 0 fitted)',
              fontsize=11, fontweight='bold', pad=10)
ax1.set_ylabel('Signal Intensity / Efficiency', fontsize=10)
ax1.legend(loc='upper left', fontsize=8.5, frameon=True, facecolor='#ffffff', edgecolor='#e0e0e0')

ax2.plot(mas_dense, best_curve_dense, color='#2ca02c', lw=2.5,
         label=f'lmfit-Optimized Trend (Scaled x {best_scale:.3f})', zorder=3)
v_extra_val = best['v_extra'].value
v_extra_err = best['v_extra'].stderr if best['v_extra'].stderr is not None else float('nan')
fit_summary = (f"v_extra = {v_extra_val:.3f} $\\pm$ {v_extra_err:.3f} mL/g\n"
               f"(hand-set: {DELTA_HYDRATION + V_CAVITY:.3f} mL/g)\n"
               f"$\\delta_{{smooth}}$, D$_{{HH}}$, $\\phi_m$, p fixed")
ax2.set_title(f'lmfit-Optimized Parameters\n$\\chi^2$={chi2_after:.2f} ({dof_after} dof)',
              fontsize=11, fontweight='bold', pad=10)
ax2.legend(loc='upper left', fontsize=8.5, frameon=True, facecolor='#ffffff', edgecolor='#e0e0e0')
ax2.text(0.97, 0.03, fit_summary, transform=ax2.transAxes, fontsize=8, ha='right', va='bottom',
         family='monospace', bbox=dict(boxstyle='round', facecolor='#f3f3f3', edgecolor='#cccccc'))

fig.suptitle('CP Signal Evolution: Simulation vs. Experiment', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()